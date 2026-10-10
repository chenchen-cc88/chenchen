"""Ansys Fluent 后端（通过 PyFluent / gRPC 实时耦合）。

运行时只依赖 5 个 Fluent 操作，这几个接口在 Fluent 2024R2 ~ 2026R1 中名称一致
（已对照 PyFluent 0.42 自带的 settings_242/251/252/261 API 定义核对）：

    settings.file.read_case(file_name=...)
    settings.file.read_data(file_name=...)                       # 每回合 reset
    settings.setup.named_expressions[name].definition = "..."    # 写入控制量/阵风
    settings.solution.run_calculation.dual_time_iterate(time_step_count=1, max_iter_per_step=N)
    settings.solution.report_definitions.compute(report_defs=[...])  # 读取 Cl/Cd/探头

控制量与阵风都通过“命名表达式 (Named Expression)”传入 Fluent，边界条件在算例中引用这些
表达式，因此运行时不需要修改任何边界条件对象，跨版本最稳健：

    u_ctrl    : 无量纲控制量 ∈ [-1, 1]
    gust_aoa  : 阵风攻角，例如 "1.234 [deg]"
    jet_lower 速度 = Vjet_max * max(u_ctrl, 0)    （下表面尾缘吹气 → 增升）
    jet_upper 速度 = Vjet_max * max(-u_ctrl, 0)   （上表面尾缘吹气 → 减升）
    远场入口速度分量 = U_inf * (cos(alpha0 + gust_aoa), sin(alpha0 + gust_aoa))
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import yaml

from .backend_base import Backend, Measurement, SimulationDiverged
from .config import Config, FluentConfig
from .gust import GustSignal

log = logging.getLogger(__name__)


# ============================================================================ session
def launch_fluent_session(
    fcfg: FluentConfig, workdir: Optional[Path] = None, processors: Optional[int] = None, rank: int = 0
) -> Any:
    """启动（或连接）一个 2D 双精度 Fluent 求解器会话。

    server_info_file 可包含 ``{rank}`` 占位符（如 ``fluent_servers/server_{rank}.txt``），
    便于在集群上预先手动启动多个 Fluent（``fluent 2ddp -t8 -sifile=server_0.txt``）供并行环境连接。
    """
    try:
        import ansys.fluent.core as pyfluent
    except ImportError as exc:  # pragma: no cover - 取决于用户环境
        raise ImportError("需要安装 PyFluent: pip install ansys-fluent-core（并安装 Ansys Fluent 2024R2+）") from exc

    if fcfg.server_info_file:
        sifile = fcfg.server_info_file.format(rank=rank)
        log.info("连接已有 Fluent 会话: %s", sifile)
        return pyfluent.connect_to_fluent(server_info_file_name=sifile)

    kwargs: dict = dict(
        precision="double",
        processor_count=processors or fcfg.processors,
        mode="solver",
        start_timeout=fcfg.start_timeout,
        additional_arguments=fcfg.additional_arguments,
        start_transcript=False,
        cleanup_on_exit=True,
    )
    if workdir is not None:
        kwargs["cwd"] = str(workdir)
    if fcfg.product_version:
        kwargs["product_version"] = fcfg.product_version
    try:
        return pyfluent.launch_fluent(dimension=2, ui_mode=fcfg.ui_mode, **kwargs)
    except TypeError:
        # 旧版 PyFluent (<0.20) 的参数名
        kwargs.pop("cwd", None)
        return pyfluent.launch_fluent(version="2d", show_gui=False, **kwargs)


def _parse_compute_result(res: Any) -> dict:
    """解析 report_definitions.compute 的返回值，例如 [{'cl': [0.61, 0]}, {'cd': [0.014, 0]}]。"""
    out: dict = {}

    def visit(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (int, float)):
                    out[k] = float(v)
                elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                    out[k] = float(v[0])
                else:
                    visit(v)
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)

    visit(res)
    return out


class FluentHandle:
    """对 PyFluent settings API 的薄封装：多版本路径回退 + 缓存。"""

    def __init__(self, session: Any):
        self.session = session
        self.settings = getattr(session, "settings", session)
        self._expr_cache: dict = {}
        self._iterate_style: Optional[str] = None

    # ------------------------------------------------------------------ 通用路径
    def node(self, *paths: str) -> Any:
        """按候选路径依次尝试，返回第一个存在的 settings 对象。路径可含 ``name[key]``。"""
        errors = []
        for path in paths:
            obj = self.settings
            try:
                for part in path.split("."):
                    m = re.fullmatch(r"(\w+)\[(.+)\]", part)
                    if m:
                        obj = getattr(obj, m.group(1))[m.group(2).strip("'\"")]
                    else:
                        obj = getattr(obj, part)
                return obj
            except (AttributeError, KeyError) as exc:
                errors.append(f"{path}: {exc}")
        raise AttributeError("找不到 settings 路径：\n  " + "\n  ".join(errors))

    def try_set(self, value: Any, *paths: str) -> str:
        """把 value 写到第一个可用的路径，返回实际使用的路径。"""
        errors = []
        for path in paths:
            parent_path, _, leaf = path.rpartition(".")
            try:
                parent = self.node(parent_path) if parent_path else self.settings
                setattr(parent, leaf, value)
                return path
            except Exception as exc:  # noqa: BLE001 - PyFluent 抛出多种异常类型
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
        raise RuntimeError("设置失败：\n  " + "\n  ".join(errors))

    def tui(self, command: str) -> None:
        self.session.execute_tui(command)

    # ------------------------------------------------------------------ 文件
    def read_case(self, path: str) -> None:
        self.settings.file.read_case(file_name=str(path))

    def read_mesh(self, path: str) -> None:
        self.settings.file.read_mesh(file_name=str(path))

    def read_data(self, path: str) -> None:
        try:
            self.settings.file.read_data(file_name=str(path))
        except AttributeError:
            self.tui(f'/file/read-data "{path}"')

    def write_case_data(self, path: str) -> None:
        self.settings.file.write_case_data(file_name=str(path))

    def write_data(self, path: str) -> None:
        self.settings.file.write_data(file_name=str(path))

    # ------------------------------------------------------------------ 命名表达式
    def expression_names(self) -> list:
        return list(self.settings.setup.named_expressions.get_object_names())

    def ensure_expression(self, name: str, definition: str, description: str = "") -> None:
        ne = self.settings.setup.named_expressions
        if name in ne.get_object_names():
            ne[name].definition = definition
        else:
            ne[name] = {"definition": definition}
        if description:
            try:
                ne[name].description = description
            except Exception:  # noqa: BLE001
                pass
        self._expr_cache.pop(name, None)

    def set_expression(self, name: str, definition: str) -> None:
        node = self._expr_cache.get(name)
        if node is None:
            ne = self.settings.setup.named_expressions
            if name not in ne.get_object_names():
                raise KeyError(f"算例中没有命名表达式 '{name}'，请先运行 scripts/setup_fluent_case.py 或按文档手动创建")
            node = ne[name]
            self._expr_cache[name] = node
        node.definition = definition

    # ------------------------------------------------------------------ 求解
    def set_time_step(self, dt: float) -> str:
        return self.try_set(
            dt,
            "solution.run_calculation.transient_controls.time_step_size",  # 24R2 ~ 25R2
            "solution.run_calculation.parameters.time_step_size",  # 26R1+
        )

    def flow_time(self) -> Optional[float]:
        for path in ("solution.run_calculation.transient_controls.flow_time",):
            try:
                return float(self.node(path)())
            except Exception:  # noqa: BLE001
                pass
        try:
            return float(self.session.scheme.eval("(rpgetvar 'flow-time)"))
        except Exception:  # noqa: BLE001
            return None

    def dual_time_iterate(self, n_steps: int, max_iter: int) -> None:
        rc = self.settings.solution.run_calculation
        if self._iterate_style in (None, "new"):
            try:
                rc.dual_time_iterate(time_step_count=n_steps, max_iter_per_step=max_iter)
                self._iterate_style = "new"
                return
            except (TypeError, KeyError, AttributeError) as exc:
                if self._iterate_style == "new":
                    raise
                log.warning("dual_time_iterate 新参数名不可用（%s），尝试旧版参数名", exc)
        if self._iterate_style in (None, "old"):
            try:
                rc.dual_time_iterate(number_of_time_steps=n_steps, maximum_number_of_iterations_per_time_step=max_iter)
                self._iterate_style = "old"
                return
            except (TypeError, KeyError, AttributeError) as exc:
                if self._iterate_style == "old":
                    raise
                log.warning("dual_time_iterate 旧参数名也不可用（%s），改用 TUI", exc)
        self._iterate_style = "tui"
        self.tui(f"/solve/dual-time-iterate {n_steps} {max_iter}")

    def compute_reports(self, names: Iterable[str]) -> dict:
        names = list(names)
        res = self.settings.solution.report_definitions.compute(report_defs=names)
        out = _parse_compute_result(res)
        missing = [n for n in names if n not in out]
        if missing:
            raise KeyError(f"报告定义 {missing} 没有返回值，原始结果: {res!r}")
        return out

    def exit(self) -> None:
        try:
            self.session.exit()
        except Exception:  # noqa: BLE001
            pass


def load_meta(path: str | Path) -> dict:
    p = Path(path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ============================================================================ backend
class FluentBackend(Backend):
    """每个实例对应一个 Fluent 进程。并行训练时每个子进程各自启动一个实例（不同工作目录）。"""

    def __init__(self, cfg: Config, rank: int = 0, session: Any = None):
        self.cfg = cfg
        fc = cfg.fluent
        self.fc = fc
        self.dt_cfd = cfg.control.dt_cfd
        self.n_substeps = cfg.control.n_substeps
        self.workdir = Path(fc.workdir).resolve() / f"env{rank}"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.case_file = str(Path(fc.case_file).resolve())
        self.init_data_file = str(Path(fc.init_data_file).resolve())
        if fc.server_info_file is None:
            for f in (self.case_file, self.init_data_file):
                if not os.path.isfile(f):
                    raise FileNotFoundError(f"找不到 {f}；请先按 docs/fluent_setup.md 准备算例")

        meta = load_meta(fc.meta_file)
        self._cl_ref = fc.cl_ref if fc.cl_ref is not None else meta.get("cl_ref")
        self._cd_ref = fc.cd_ref if fc.cd_ref is not None else meta.get("cd_ref")
        if "dt_cfd" in meta and not math.isclose(float(meta["dt_cfd"]), self.dt_cfd, rel_tol=1e-6):
            log.warning("初始流场的时间步 %.3g 与配置 dt_cfd %.3g 不一致，以配置为准", meta["dt_cfd"], self.dt_cfd)

        self.report_names = [fc.report_cl, fc.report_cd] + list(fc.probe_reports)
        log.info("[env%d] 启动 Fluent，工作目录 %s", rank, self.workdir)
        self.h = FluentHandle(session or launch_fluent_session(fc, self.workdir, rank=rank))
        self.h.read_case(self.case_file)
        self.h.set_time_step(self.dt_cfd)
        names = self.h.expression_names()
        for n in (fc.expr_control, fc.expr_gust):
            if n not in names:
                raise KeyError(f"算例缺少命名表达式 {n}（现有: {names}）")
        self.gust: Optional[GustSignal] = None
        self.t = 0.0

    @property
    def cl_ref(self) -> float:
        if self._cl_ref is None:
            raise RuntimeError("cl_ref 未知：请在配置 fluent.cl_ref 或 meta 文件中给出，或先调用 reset()")
        return float(self._cl_ref)

    @property
    def cd_ref(self) -> float:
        return float(self._cd_ref) if self._cd_ref is not None else 0.0

    @property
    def n_probes(self) -> int:
        return len(self.fc.probe_reports)

    # ------------------------------------------------------------------
    def _apply_inputs(self, u: float, t: float) -> None:
        assert self.gust is not None
        self.h.set_expression(self.fc.expr_control, f"{u:.6f}")
        self.h.set_expression(self.fc.expr_gust, f"{math.degrees(self.gust(t)):.6f} [deg]")

    def _read(self) -> tuple[float, float, Optional[np.ndarray]]:
        r = self.h.compute_reports(self.report_names)
        cl, cd = r[self.fc.report_cl], r[self.fc.report_cd]
        if not (math.isfinite(cl) and math.isfinite(cd)) or abs(cl) > self.fc.cl_abs_limit:
            raise SimulationDiverged(f"Fluent 结果异常: Cl={cl}, Cd={cd}")
        probes = np.array([r[n] for n in self.fc.probe_reports]) if self.fc.probe_reports else None
        return cl, cd, probes

    def reset(self, gust: GustSignal) -> Measurement:
        self.gust = gust
        self.t = 0.0
        self.h.read_data(self.init_data_file)
        self._apply_inputs(0.0, 0.0)
        cl, cd, probes = self._read()
        if self._cl_ref is None:
            log.warning("未提供 cl_ref，使用初始流场瞬时值 %.4f", cl)
            self._cl_ref, self._cd_ref = cl, cd
        return Measurement(self.t, cl, cd, np.array([cl]), np.array([cd]), gust(0.0), probes)

    def step(self, u_old: float, u_new: float) -> Measurement:
        cl_s, cd_s = [], []
        probes = None
        n = self.n_substeps
        for k in range(1, n + 1):
            u = u_old + (u_new - u_old) * k / n
            t_next = self.t + self.dt_cfd
            # 隐式格式求解 t_{n+1} 时刻的流场，边界条件取 t_{n+1} 的值
            self._apply_inputs(u, t_next)
            self.h.dual_time_iterate(1, self.fc.max_iter_per_step)
            self.t = t_next
            cl, cd, probes = self._read()
            cl_s.append(cl)
            cd_s.append(cd)
        assert self.gust is not None
        return Measurement(self.t, cl_s[-1], cd_s[-1], np.array(cl_s), np.array(cd_s), self.gust(self.t), probes)

    def close(self) -> None:
        self.h.exit()
