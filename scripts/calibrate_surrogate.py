#!/usr/bin/env python
"""用 Fluent 开环数据标定代理模型（缩小 sim-to-CFD 差距，使预训练策略能直接迁移）。

  # 1) 在 Fluent 上采集开环数据（8 组实验，每组 120 控制步 ≈ 600 个时间步）
  python scripts/calibrate_surrogate.py collect --config configs/fluent.yaml --out fluent_case/sysid.npz

  # 2) 拟合代理模型参数，输出可直接叠加的 YAML
  python scripts/calibrate_surrogate.py fit --config configs/surrogate.yaml --data fluent_case/sysid.npz \
      --out configs/surrogate_calibrated.yaml

  # 3) 用标定后的代理模型预训练
  python scripts/train.py --config configs/surrogate.yaml --config configs/surrogate_calibrated.yaml

  # 自测：用代理模型自身生成“伪 Fluent”数据，验证辨识流程能找回真值
  python scripts/calibrate_surrogate.py selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airfoil_rl.config import load_config  # noqa: E402
from airfoil_rl.env import make_backend  # noqa: E402
from airfoil_rl.surrogate import SurrogateBackend, SurrogateParams  # noqa: E402
from airfoil_rl.sysid import (  # noqa: E402
    DEFAULT_FIT_KEYS,
    collect,
    default_experiments,
    fit_surrogate,
    load_dataset,
    run_openloop,
    save_dataset,
)


def plot_fit(cfg, data: dict, params: SurrogateParams, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from airfoil_rl.sysid import data_to_experiments

    be = SurrogateBackend(cfg, stochastic=False, params=params)
    exps = data_to_experiments(data)
    n = len(exps)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(11, 2.4 * ((n + 1) // 2)), squeeze=False)
    tc = cfg.flow.convective_time
    for ax, exp in zip(axes.ravel(), exps):
        ref = data[exp.name]
        sim = run_openloop(be, exp)
        ax.plot(ref["t"] / tc, ref["cl"], "k-", lw=1.2, label="Fluent")
        ax.plot(sim["t"] / tc, sim["cl"], "r--", lw=1.2, label="surrogate")
        ax.set_title(exp.name, fontsize=9)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    fig.supxlabel(r"$tU/c$")
    fig.supylabel(r"$C_L$")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def cmd_collect(args) -> None:
    cfg = load_config(args.config, args.set)
    be = make_backend(cfg, rank=0)
    try:
        data = collect(be, default_experiments(cfg, n_steps=args.n_steps))
    finally:
        be.close()
    save_dataset(args.out, data, {"backend": cfg.backend, "dt_cfd": cfg.control.dt_cfd, "n_substeps": cfg.control.n_substeps})
    print(f"开环数据已保存到 {args.out}")


def cmd_fit(args) -> None:
    cfg = load_config(args.config, args.set)
    data, meta = load_dataset(args.data)
    if abs(meta.get("dt_cfd", cfg.control.dt_cfd) - cfg.control.dt_cfd) > 1e-12:
        raise ValueError("数据的 dt_cfd 与当前配置不一致")
    keys = args.keys.split(",") if args.keys else DEFAULT_FIT_KEYS
    params, summ = fit_surrogate(cfg, data, keys, max_nfev=args.max_nfev)
    print(f"Cl/Cd 残差 RMS: {summ['rms_before']:.5f} -> {summ['rms_after']:.5f}")
    for k in keys:
        print(f"  {k:20s} {summ['x0'][k]:10.4f} -> {summ['x'][k]:10.4f}")
    out = {"surrogate": {k: round(float(getattr(params, k)), 6) for k in keys}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# 由 scripts/calibrate_surrogate.py 生成：与 configs/surrogate.yaml 叠加使用\n")
        yaml.safe_dump(out, fh, allow_unicode=True, sort_keys=False)
    png = Path(args.out).with_suffix(".png")
    plot_fit(cfg, data, params, png)
    print(f"参数写入 {args.out}，拟合对比图 {png}")


def cmd_selftest(args) -> None:
    cfg = load_config(args.config, args.set)
    truth = SurrogateParams(cl_alpha=5.3, alpha_stall_deg=10.5, jet_gain_pos=0.3, jet_gain_neg=0.25, jet_tau_slow=2.2, wagner_time_scale=1.4)
    be = SurrogateBackend(cfg, stochastic=False, params=truth)
    data = collect(be, default_experiments(cfg, n_steps=80), verbose=False)
    keys = ["cl_alpha", "alpha_stall_deg", "jet_gain_pos", "jet_gain_neg", "jet_tau_slow", "wagner_time_scale"]
    params, summ = fit_surrogate(cfg, data, keys, init=SurrogateParams(), max_nfev=80, verbose=0)
    print(f"残差 RMS: {summ['rms_before']:.5f} -> {summ['rms_after']:.6f}")
    for k in keys:
        print(f"  {k:20s} 真值 {getattr(truth, k):8.4f}  初值 {summ['x0'][k]:8.4f}  辨识 {getattr(params, k):8.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--config", action="append", default=[])
    c.add_argument("--set", action="append", default=[])
    c.add_argument("--n-steps", type=int, default=120)
    c.add_argument("--out", default="fluent_case/sysid.npz")
    f = sub.add_parser("fit")
    f.add_argument("--config", action="append", default=[])
    f.add_argument("--set", action="append", default=[])
    f.add_argument("--data", required=True)
    f.add_argument("--keys", default=None, help="逗号分隔的待辨识参数（默认见 sysid.DEFAULT_FIT_KEYS）")
    f.add_argument("--max-nfev", type=int, default=200)
    f.add_argument("--out", default="configs/surrogate_calibrated.yaml")
    s = sub.add_parser("selftest")
    s.add_argument("--config", action="append", default=[])
    s.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    {"collect": cmd_collect, "fit": cmd_fit, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
