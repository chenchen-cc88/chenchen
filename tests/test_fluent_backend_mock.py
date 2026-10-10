"""用一个模拟 PyFluent settings API 的假会话测试 FluentBackend 的耦合逻辑（无需安装 Fluent）。

假会话内部用代理模型充当“流场”，从命名表达式字符串中解析 u_ctrl 与 gust_aoa，
因此能检验：表达式字符串格式、时间步推进次数、报告解析、reset 时重新读 data 等环节。
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airfoil_rl.config import Config  # noqa: E402
from airfoil_rl.controllers import NoControl, PIDController, run_episode  # noqa: E402
from airfoil_rl.env import AirfoilLiftEnv  # noqa: E402
from airfoil_rl.fluent_backend import FluentBackend  # noqa: E402
from airfoil_rl.gust import GustSignal  # noqa: E402
from airfoil_rl.surrogate import SurrogateAirfoil, SurrogateParams  # noqa: E402


class _Expr:
    def __init__(self, definition: str):
        self.definition = definition


class _NamedExpressions:
    def __init__(self) -> None:
        self._d: dict = {}

    def get_object_names(self) -> list:
        return list(self._d)

    def __getitem__(self, k: str) -> _Expr:
        return self._d[k]

    def __setitem__(self, k: str, v: dict) -> None:
        self._d[k] = _Expr(v.get("definition", ""))


class _LiveGust(GustSignal):
    """阵风值由“边界条件表达式”实时给定。"""

    def __init__(self) -> None:
        super().__init__(np.array([0.0, 1.0]), np.zeros(2), "live")
        self.value = 0.0

    def __call__(self, t: float) -> float:
        return self.value

    def rate(self, t: float) -> float:
        return 0.0


class FakeFluent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ne = _NamedExpressions()
        self.ne["u_ctrl"] = {"definition": "0"}
        self.ne["gust_aoa"] = {"definition": "0 [deg]"}
        self.model = SurrogateAirfoil(SurrogateParams(), cfg.flow)
        self.gust = _LiveGust()
        self.t = 0.0
        self.n_iterate_calls = 0
        self.n_read_data = 0
        self.dt = None
        s = self
        self.settings = type("S", (), {})()
        self.settings.file = type("F", (), {"read_case": lambda _s, file_name: None, "read_data": lambda _s, file_name: s._read_data()})()
        self.settings.setup = type("Setup", (), {"named_expressions": self.ne})()
        tc_obj = type("TC", (), {})()
        rc = type("RC", (), {"transient_controls": tc_obj, "dual_time_iterate": lambda _s, **kw: s._iterate(**kw)})()
        rd = type("RD", (), {"compute": lambda _s, report_defs: s._compute(report_defs)})()
        self.settings.solution = type("Sol", (), {"run_calculation": rc, "report_definitions": rd})()
        self._tc = tc_obj

    def _read_data(self) -> None:
        self.n_read_data += 1
        self.t = 0.0
        self.gust.value = 0.0
        self.model.reset_steady(self.gust)

    def _inputs(self) -> float:
        u = float(self.ne["u_ctrl"].definition)
        m = re.fullmatch(r"\s*([-+0-9.eE]+)\s*\[deg\]\s*", self.ne["gust_aoa"].definition)
        assert m, f"阵风表达式格式错误: {self.ne['gust_aoa'].definition!r}"
        self.gust.value = math.radians(float(m.group(1)))
        return u

    def _iterate(self, time_step_count: int, max_iter_per_step: int) -> None:
        self.n_iterate_calls += 1
        dt = self._tc.time_step_size
        u = self._inputs()
        for _ in range(time_step_count):
            self.model.rk4(self.t, dt, u, self.gust)
            self.t += dt

    def _compute(self, report_defs: list) -> list:
        u = float(self.ne["u_ctrl"].definition)
        cl, cd, _ = self.model.outputs(self.t, u, self.gust)
        vals = {"cl": cl, "cd": cd}
        return [{n: [vals[n], 0]} for n in report_defs]

    def exit(self) -> None:
        pass


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.backend = "fluent"
    c.control.episode_steps = 80
    c.surrogate = {"jet_delay": 0.0}  # 假 Fluent 内部没有执行器延迟，对比基准保持一致
    c.fluent.workdir = str(tmp_path / "work")
    c.fluent.server_info_file = "fake"  # 跳过 case 文件存在性检查
    c.fluent.meta_file = str(tmp_path / "none.yaml")
    return c


def test_fluent_backend_matches_surrogate(cfg: Config) -> None:
    fake = FakeFluent(cfg)
    be = FluentBackend(cfg, session=fake)
    assert fake._tc.time_step_size == cfg.control.dt_cfd
    env = AirfoilLiftEnv(cfg, backend=be)
    gust = env.gust_gen.sine(3.0, 0.2)
    tr = run_episode(env, PIDController(dt=cfg.control.dt_ctrl), gust, seed=0)
    assert fake.n_iterate_calls == cfg.control.episode_steps * cfg.control.n_substeps
    assert fake.n_read_data == 1
    assert be.cl_ref == pytest.approx(tr["cl"][0], abs=1e-9)  # 未提供 cl_ref 时取初始流场值

    # 与直接用代理模型后端的结果对比（差异仅来自阵风在时间步内的保持方式）
    cfg2 = Config()
    cfg2.control.episode_steps = cfg.control.episode_steps
    cfg2.surrogate = dict(cfg.surrogate)
    env2 = AirfoilLiftEnv(cfg2)
    env2.backend.stochastic = False
    tr2 = run_episode(env2, PIDController(dt=cfg.control.dt_ctrl), gust, seed=0)
    assert np.sqrt(np.mean((tr["cl"] - tr2["cl"]) ** 2)) < 5e-3


def test_fluent_backend_detects_divergence(cfg: Config) -> None:
    fake = FakeFluent(cfg)
    be = FluentBackend(cfg, session=fake)
    env = AirfoilLiftEnv(cfg, backend=be)
    env.reset(seed=0, options={"gust": env.gust_gen.zero()})
    fake.model.y[:] = np.nan  # 模拟求解发散
    _, r, terminated, _, info = env.step(np.array([0.0]))
    assert terminated and r == -cfg.reward.diverge_penalty and "diverged" in info
    # 下一回合重新读 data，流场恢复
    run_episode(env, NoControl(), env.gust_gen.zero())
    assert fake.n_read_data == 2
