#!/usr/bin/env python
"""由网格自动建立 RL 所需的 Fluent 瞬态算例，并计算充分发展的初始流场。

前提：2D 网格（.msh / .msh.h5 / .cas.h5）包含以下边界（名称可在命令行修改）：
    airfoil    翼型壁面（不含射流缝）
    jet_upper  上表面尾缘射流缝（约 x/c = 0.95~0.96）
    jet_lower  下表面尾缘射流缝
    inlet      远场速度入口（C 型/圆形远场的上游与上下边界）
    outlet     压力出口（下游边界）

本脚本完成：
  1. 湍流模型 SST k-ω（可选 transition-sst），2 阶隐式瞬态，Coupled 求解
  2. 命名表达式：U_inf, alpha0, Vjet_max, u_ctrl, gust_aoa, aoa_in, jet_lower_vel, jet_upper_vel
  3. 边界条件引用表达式：入口速度分量 U_inf*cos/sin(aoa_in)，射流 Vjet_max*max(±u_ctrl,0)
  4. 参考值与 Cl/Cd 报告定义（升力方向垂直于平均来流），可选表面压力探头
  5. 稳态迭代 -> 瞬态推进若干对流时间 -> 统计 Cl_ref/Cd_ref -> 保存 case/data/meta

用法：
  python scripts/setup_fluent_case.py --mesh fluent_case/naca0012_jets.msh --config configs/fluent.yaml
  python scripts/setup_fluent_case.py --mesh ... --probes 6 --viscous transition-sst --processors 8

注：脚本基于 Fluent 2024R2~2026R1 的 settings API 编写（已对照 PyFluent 自带 API 定义），
    个别项设置失败时会打印清晰提示，可按 docs/fluent_setup.md 在 GUI 中手动补齐后保存。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airfoil_rl.config import load_config  # noqa: E402
from airfoil_rl.fluent_backend import FluentHandle, launch_fluent_session  # noqa: E402

FAILED: list = []


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def soft(desc: str, fn, *args, **kwargs):
    """非关键设置：失败时记录并继续。"""
    try:
        out = fn(*args, **kwargs)
        print(f"    [ok] {desc}")
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"    [!!] {desc} 失败: {type(exc).__name__}: {exc}")
        FAILED.append(desc)
        return None


def set_velocity_inlet_components(h: FluentHandle, zone: str, vx: str, vy: str) -> None:
    vi = h.settings.setup.boundary_conditions.velocity_inlet[zone]
    vi.momentum.velocity_specification_method = "Components"
    comps = vi.momentum.velocity_components
    comps[0].value = vx
    comps[1].value = vy


def set_velocity_inlet_normal(h: FluentHandle, zone: str, vmag_expr: str) -> None:
    vi = h.settings.setup.boundary_conditions.velocity_inlet[zone]
    vi.momentum.velocity_specification_method = "Magnitude, Normal to Boundary"
    mom = vi.momentum
    if hasattr(mom, "velocity_magnitude"):  # 25R1+
        mom.velocity_magnitude.value = vmag_expr
    else:  # 24R2
        mom.velocity.value = vmag_expr


def set_inlet_turbulence(h: FluentHandle, zone: str, intensity: float, ratio: float) -> None:
    turb = h.settings.setup.boundary_conditions.velocity_inlet[zone].turbulence
    turb.turbulence_specification = "Intensity and Viscosity Ratio"
    turb.turbulent_intensity = intensity
    turb.turbulent_viscosity_ratio = ratio


def create_force_report(h: FluentHandle, kind: str, name: str, zones: list, direction: tuple, output_type: str) -> None:
    group = getattr(h.settings.solution.report_definitions, kind)
    if name not in group.get_object_names():
        group[name] = {}
    rep = group[name]
    rep.zones = zones
    try:
        rep.force_vector = list(direction)
    except Exception:  # noqa: BLE001  某些版本 2D 也要求 3 个分量
        rep.force_vector = list(direction) + [0.0]
    rep.report_output_type = output_type


def create_probe(h: FluentHandle, name: str, x: float, y: float) -> None:
    ps = h.settings.results.surfaces.point_surface
    pname = f"pt_{name}"
    if pname not in ps.get_object_names():
        ps[pname] = {}
    try:
        ps[pname].point = [x, y]
    except Exception:  # noqa: BLE001
        ps[pname].point = [x, y, 0.0]
    sr = h.settings.solution.report_definitions.surface
    if name not in sr.get_object_names():
        sr[name] = {}
    sr[name].report_type = "surface-vertexavg"
    sr[name].field = "pressure-coefficient"
    sr[name].surface_names = [pname]


def naca_y(xc: float, t: float = 0.12) -> float:
    return 5 * t * (0.2969 * math.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="网格或已有 case 文件")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out-dir", default="fluent_case")
    ap.add_argument("--name", default="airfoil_rl")
    ap.add_argument("--processors", type=int, default=None)
    ap.add_argument("--jet-vmax", type=float, default=20.0, help="射流最大速度 [m/s]（u=±1 时）")
    ap.add_argument("--viscous", default="sst", choices=["sst", "transition-sst"])
    ap.add_argument("--turb-intensity", type=float, default=0.001, help="来流湍流度（小数，0.001=0.1%%）")
    ap.add_argument("--turb-visc-ratio", type=float, default=1.0)
    ap.add_argument("--steady-iters", type=int, default=800)
    ap.add_argument("--init-conv", type=float, default=25.0, help="瞬态预推进时长（对流时间 c/U 个数）")
    ap.add_argument("--stat-conv", type=float, default=5.0, help="最后多少个对流时间用于统计 Cl_ref")
    ap.add_argument("--probes", type=int, default=0, help="上/下表面各布置 N/2 个 Cp 探头（0 表示不建）")
    ap.add_argument("--zone-airfoil", default="airfoil")
    ap.add_argument("--zone-jet-upper", default="jet_upper")
    ap.add_argument("--zone-jet-lower", default="jet_lower")
    ap.add_argument("--zone-inlet", default="inlet")
    ap.add_argument("--zone-outlet", default="outlet")
    ap.add_argument("--keep-open", action="store_true", help="结束后不关闭 Fluent（ui_mode=gui 时便于检查）")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    fl, fc, ctl = cfg.flow, cfg.fluent, cfg.control
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = fl.convective_time
    a0 = math.radians(fl.alpha0_deg)
    print(cfg.summary())

    step("启动 Fluent（2D, 双精度）")
    h = FluentHandle(launch_fluent_session(fc, out_dir, args.processors))
    mesh = str(Path(args.mesh).resolve())
    if mesh.endswith((".cas", ".cas.h5", ".cas.gz")):
        h.read_case(mesh)
    else:
        h.read_mesh(mesh)

    bc = h.settings.setup.boundary_conditions
    step("检查/设置边界类型")
    z_vi = [args.zone_inlet, args.zone_jet_upper, args.zone_jet_lower]
    for z in z_vi:
        if z not in bc.velocity_inlet.get_object_names():
            soft(f"{z} -> velocity-inlet", bc.set_zone_type, zone_list=[z], new_type="velocity-inlet")
    if args.zone_outlet not in bc.pressure_outlet.get_object_names():
        soft(f"{args.zone_outlet} -> pressure-outlet", bc.set_zone_type, zone_list=[args.zone_outlet], new_type="pressure-outlet")
    print("    velocity_inlet:", list(bc.velocity_inlet.get_object_names()))
    print("    pressure_outlet:", list(bc.pressure_outlet.get_object_names()))
    print("    wall:", list(bc.wall.get_object_names()))

    step("求解器与模型")
    s = h.settings
    soft("稳态（先做稳态初值）", setattr, s.setup.general.solver, "time", "steady")
    soft("粘性模型 k-omega", setattr, s.setup.models.viscous, "model", "k-omega" if args.viscous == "sst" else "transition-sst")
    if args.viscous == "sst":
        soft("k-omega SST", setattr, s.setup.models.viscous, "k_omega_model", "sst")
    air = s.setup.materials.fluid["air"]
    soft("空气密度", setattr, air.density, "value", fl.density)
    soft("空气黏度", setattr, air.viscosity, "value", fl.viscosity)
    soft("Coupled 压力-速度耦合", setattr, s.solution.methods.p_v_coupling, "flow_scheme", "Coupled")

    step("命名表达式（RL 运行时只修改 u_ctrl 与 gust_aoa）")
    exprs = {
        "U_inf": (f"{fl.u_inf} [m/s]", "来流速度"),
        "alpha0": (f"{fl.alpha0_deg} [deg]", "平均攻角"),
        "Vjet_max": (f"{args.jet_vmax} [m/s]", "射流最大速度"),
        fc.expr_control: ("0", "RL 控制量 u ∈ [-1,1]"),
        fc.expr_gust: ("0 [deg]", "阵风攻角（RL 每个时间步写入）"),
        "aoa_in": (f"alpha0 + {fc.expr_gust}", "瞬时来流方向"),
        "jet_lower_vel": (f"Vjet_max*max({fc.expr_control}, 0)", "下表面射流：增升"),
        "jet_upper_vel": (f"Vjet_max*max(-{fc.expr_control}, 0)", "上表面射流：减升"),
    }
    for name, (definition, desc) in exprs.items():
        soft(f"{name} = {definition}", h.ensure_expression, name, definition, desc)

    step("边界条件")
    soft("入口速度分量 = U_inf*(cos, sin)(aoa_in)", set_velocity_inlet_components, h, args.zone_inlet, "U_inf*cos(aoa_in)", "U_inf*sin(aoa_in)")
    soft("入口湍流", set_inlet_turbulence, h, args.zone_inlet, args.turb_intensity, args.turb_visc_ratio)
    soft("下射流速度 = jet_lower_vel", set_velocity_inlet_normal, h, args.zone_jet_lower, "jet_lower_vel")
    soft("上射流速度 = jet_upper_vel", set_velocity_inlet_normal, h, args.zone_jet_upper, "jet_upper_vel")
    for z in (args.zone_jet_lower, args.zone_jet_upper):
        soft(f"{z} 湍流", set_inlet_turbulence, h, z, 0.01, 1.0)

    step("参考值（2D：面积 = 弦长 × 1 m 展向）")
    rv = s.setup.reference_values
    for key, val in dict(area=fl.chord, length=fl.chord, velocity=fl.u_inf, density=fl.density, viscosity=fl.viscosity).items():
        soft(f"reference {key} = {val}", setattr, rv, key, val)

    step("报告定义")
    zones = [args.zone_airfoil, args.zone_jet_upper, args.zone_jet_lower]
    lift_dir = (-math.sin(a0), math.cos(a0))
    drag_dir = (math.cos(a0), math.sin(a0))
    soft(f"{fc.report_cl}: 升力系数（方向 {lift_dir[0]:.4f}, {lift_dir[1]:.4f}）", create_force_report, h, "lift", fc.report_cl, zones, lift_dir, "Lift Coefficient")
    soft(f"{fc.report_cd}: 阻力系数", create_force_report, h, "drag", fc.report_cd, zones, drag_dir, "Drag Coefficient")
    probe_names = []
    if args.probes > 0:
        n_side = max(args.probes // 2, 1)
        xs = np.linspace(0.1, 0.85, n_side)
        for side, sign in (("u", 1.0), ("l", -1.0)):
            for i, xc in enumerate(xs, 1):
                name = f"cp_{side}{i}"
                # 点略微离开壁面（0.2% 弦长），vertex-average 插值得到壁面附近 Cp
                y = sign * (naca_y(xc) + 0.002) * fl.chord
                try:
                    create_probe(h, name, xc * fl.chord, y)
                    probe_names.append(name)
                    print(f"    [ok] 探头 {name} @ x/c={xc:.2f}")
                except Exception as exc:  # noqa: BLE001
                    print(f"    [!!] 探头 {name} 失败: {exc}")
                    FAILED.append(f"探头 {name}")
        if probe_names:
            print(f"    在配置中设置 fluent.probe_reports: {probe_names} 并打开 obs.include_probes")
    report_defs = [fc.report_cl, fc.report_cd] + probe_names
    soft(
        "报告文件 (每时间步写 Cl/Cd)",
        s.solution.monitor.report_files.__setitem__,
        "rl-report-file",
        {"report_defs": [fc.report_cl, fc.report_cd], "file_name": str(out_dir / f"{args.name}_reports.out"), "frequency_of": "time-step"},
    )

    step(f"稳态初始化与迭代 {args.steady_iters} 步")
    soft("混合初始化", s.solution.initialization.hybrid_initialize)
    soft("稳态迭代", s.solution.run_calculation.iterate, iter_count=args.steady_iters)
    r = soft("稳态 Cl/Cd", h.compute_reports, [fc.report_cl, fc.report_cd])
    if r:
        print(f"    稳态 Cl={r[fc.report_cl]:.4f}, Cd={r[fc.report_cd]:.5f}")

    step("切换为 2 阶隐式瞬态")
    soft("unsteady-2nd-order", setattr, s.setup.general.solver, "time", "unsteady-2nd-order")
    soft(f"时间步 {ctl.dt_cfd:g} s", h.set_time_step, ctl.dt_cfd)
    n_total = int(round(args.init_conv * tc / ctl.dt_cfd))
    n_stat = int(round(args.stat_conv * tc / ctl.dt_cfd))
    n_pre = max(n_total - n_stat, 0)
    print(f"    预推进 {n_pre} 步 + 统计 {n_stat} 步（每步 {fc.max_iter_per_step} 次内迭代）")
    t0 = time.time()
    if n_pre:
        h.dual_time_iterate(n_pre, fc.max_iter_per_step)
    cls, cds, probes = [], [], []
    for _ in range(n_stat):
        h.dual_time_iterate(1, fc.max_iter_per_step)
        rr = h.compute_reports(report_defs)
        cls.append(rr[fc.report_cl])
        cds.append(rr[fc.report_cd])
        probes.append([rr[p] for p in probe_names])
    wall = time.time() - t0
    cls, cds = np.array(cls), np.array(cds)
    print(f"    Cl = {cls.mean():.5f} ± {cls.std():.2e}，Cd = {cds.mean():.5f}；耗时 {wall:.0f} s（{wall / max(n_total, 1):.2f} s/步）")
    if cls.std() > 0.02:
        print("    [警告] Cl 波动较大（可能有涡脱落或未收敛），检查网格/时间步，或增大 --init-conv")

    step("保存")
    case_path = out_dir / f"{args.name}.cas.h5"
    data_path = out_dir / f"{args.name}_init.dat.h5"
    h.settings.file.write_case(file_name=str(case_path))
    h.write_data(str(data_path))
    meta = {
        "cl_ref": float(cls.mean()),
        "cd_ref": float(cds.mean()),
        "cl_std": float(cls.std()),
        "probe_reports": probe_names,
        "probe_ref": [float(v) for v in np.mean(probes, axis=0)] if probe_names else [],
        "dt_cfd": ctl.dt_cfd,
        "alpha0_deg": fl.alpha0_deg,
        "u_inf": fl.u_inf,
        "chord": fl.chord,
        "jet_vmax": args.jet_vmax,
        "seconds_per_time_step": wall / max(n_total, 1),
        "failed_settings": FAILED,
    }
    meta_path = out_dir / f"{args.name}_meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, allow_unicode=True, sort_keys=False)
    print(f"    case: {case_path}\n    data: {data_path}\n    meta: {meta_path}")
    if FAILED:
        print("\n以下设置未成功，请在 GUI 中按 docs/fluent_setup.md 手动设置后重新保存 case：")
        for f in FAILED:
            print("   -", f)
    if not args.keep_open:
        h.exit()


if __name__ == "__main__":
    main()
