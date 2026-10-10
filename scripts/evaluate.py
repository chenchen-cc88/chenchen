#!/usr/bin/env python
"""在固定阵风工况上对比：无控制 / PID(+前馈) / RL 策略，输出指标表与时程曲线。

示例：
  python scripts/evaluate.py --run runs/sac_surrogate                 # 使用运行目录中的 config.yaml 与 best_model.zip
  python scripts/evaluate.py --run runs/sac_surrogate --tune-pid      # 先在随机阵风上整定 PID 再对比
  python scripts/evaluate.py --run runs/sac_surrogate --config configs/fluent.yaml   # 把代理模型策略直接放到 Fluent 上检验
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airfoil_rl.config import load_config  # noqa: E402
from airfoil_rl.controllers import NoControl, PIDController, PolicyController, run_episode, tune_pid  # noqa: E402
from airfoil_rl.env import AirfoilLiftEnv  # noqa: E402


def plot_scenario(name: str, trajs: dict, out: Path, tc: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    colors = {"no_control": "#9a9a9a", "pid": "#1f77b4", "rl": "#d62728"}
    first = next(iter(trajs.values()))
    ax[0].plot(first["t"] / tc, first["alpha_g_deg"], "k-", lw=1.2)
    ax[0].set_ylabel(r"gust $\alpha_g$ [deg]")
    ax[1].axhline(first["cl_ref"], color="k", ls="--", lw=1, label=r"$C_{L,ref}$")
    for key, tr in trajs.items():
        m = tr["metrics"]
        ax[1].plot(tr["t"] / tc, tr["cl"], color=colors.get(key), lw=1.2, label=f"{key} (RMSE {m['cl_rmse']:.4f})")
        if key != "no_control":
            ax[2].plot(tr["t"] / tc, tr["u"], color=colors.get(key), lw=1.2, label=key)
    ax[1].set_ylabel(r"$C_L$")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].set_ylabel("jet command u")
    ax[2].set_ylim(-1.05, 1.05)
    ax[2].set_xlabel(r"$tU_\infty/c$")
    ax[2].legend(loc="upper right", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="训练输出目录（读取 config.yaml 与模型）")
    ap.add_argument("--model", default=None, help="模型路径（默认 <run>/best_model.zip，其次 final_model.zip）")
    ap.add_argument("--config", action="append", default=[], help="额外 YAML（覆盖运行目录中的配置，例如切换到 Fluent）")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tune-pid", action="store_true", help="在随机阵风（非评估工况）上整定 PID")
    ap.add_argument("--pid", default=None, help='PID 增益 JSON，如 \'{"kp":1.0,"ki":200,"kd":0.001,"kff":0.05}\'')
    ap.add_argument("--no-pid", action="store_true")
    ap.add_argument("--scenarios", default=None, help="逗号分隔的工况名，默认全部")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_files = ([str(Path(args.run) / "config.yaml")] if args.run else []) + args.config
    cfg = load_config(cfg_files, args.set)
    out = Path(args.out or (Path(args.run) / "eval" if args.run else "runs/eval"))
    out.mkdir(parents=True, exist_ok=True)

    env = AirfoilLiftEnv(cfg)
    tc = cfg.flow.convective_time
    controllers: dict = {"no_control": NoControl()}

    if not args.no_pid:
        if args.pid:
            pid = PIDController(dt=cfg.control.dt_ctrl, cl_scale=cfg.reward.cl_scale, **json.loads(args.pid))
        elif args.tune_pid:
            rng = np.random.default_rng(777)
            train_gusts = [env.gust_gen.sample(rng) for _ in range(6)]
            print("整定 PID 中（Nelder–Mead，基于随机训练阵风）...")
            pid = tune_pid(env, train_gusts)
        else:
            pid = PIDController(dt=cfg.control.dt_ctrl, cl_scale=cfg.reward.cl_scale)
        print(f"PID: kp={pid.kp:.4g} ki={pid.ki:.4g} kd={pid.kd:.4g} kff={pid.kff:.4g}")
        controllers["pid"] = pid

    model_path = args.model
    if model_path is None and args.run:
        for cand in ("best_model.zip", "final_model.zip"):
            if (Path(args.run) / cand).is_file():
                model_path = str(Path(args.run) / cand)
                break
    if model_path:
        from airfoil_rl.training import load_model

        model = load_model(model_path, cfg.train.algo, device="cpu")
        if model.observation_space.shape != env.observation_space.shape:
            raise ValueError(f"模型观测维度 {model.observation_space.shape} 与环境 {env.observation_space.shape} 不一致")
        controllers["rl"] = PolicyController(model)
        print(f"RL 模型: {model_path}")

    scenarios = env.gust_gen.standard_scenarios()
    if args.scenarios:
        scenarios = {k: scenarios[k] for k in args.scenarios.split(",")}

    rows = []
    for i, (sname, gust) in enumerate(scenarios.items()):
        trajs = {}
        for cname, ctrl in controllers.items():
            tr = run_episode(env, ctrl, gust, seed=1000 + i)
            trajs[cname] = tr
            rows.append({"scenario": sname, "controller": cname, **tr["metrics"]})
            np.savez_compressed(out / f"traj_{sname}_{cname}.npz", **{k: v for k, v in tr.items() if k != "metrics"})
        base = trajs["no_control"]["metrics"]["cl_rmse"]
        msg = " | ".join(
            f"{c}: RMSE={t['metrics']['cl_rmse']:.4f} ({100 * (1 - t['metrics']['cl_rmse'] / base):+.1f}%)"
            for c, t in trajs.items()
            if c != "no_control"
        )
        print(f"{sname:18s} 无控制 RMSE={base:.4f} | {msg}")
        plot_scenario(sname, trajs, out / f"{sname}.png", tc)

    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 汇总（各工况平均）
    lines = ["| 控制器 | Cl RMSE | 最大偏差 | u_rms | 平均 Cd | 回报 |", "|---|---|---|---|---|---|"]
    for cname in controllers:
        sel = [r for r in rows if r["controller"] == cname]
        lines.append(
            f"| {cname} | {np.mean([r['cl_rmse'] for r in sel]):.4f} | {np.mean([r['cl_max_dev'] for r in sel]):.4f} | "
            f"{np.mean([r['u_rms'] for r in sel]):.3f} | {np.mean([r['cd_mean'] for r in sel]):.4f} | "
            f"{np.mean([r['return'] for r in sel]):.1f} |"
        )
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\n结果保存在 {out}")
    env.close()


if __name__ == "__main__":
    main()
