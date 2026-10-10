#!/usr/bin/env python
"""Fluent 耦合自检：在正式训练前验证 PyFluent 接口在你的 Fluent 版本上可用。

检查项：启动/连接 → 读 case/data → 命名表达式写入 → 推进时间步 → 读取 Cl/Cd →
        射流指令阶跃后 Cl 是否变化（确认执行器连通）→ 阵风写入后 Cl 是否变化 → 单步耗时。

  python scripts/check_fluent.py --config configs/fluent.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airfoil_rl.config import load_config  # noqa: E402
from airfoil_rl.fluent_backend import FluentBackend  # noqa: E402
from airfoil_rl.gust import GustGenerator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=["configs/fluent.yaml"])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--steps", type=int, default=8, help="每项测试推进的控制步数")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    cfg.backend = "fluent"

    t0 = time.time()
    be = FluentBackend(cfg, rank=0)
    print(f"[1] 启动并读入 case: {time.time() - t0:.1f} s")
    gen = GustGenerator(cfg.gust, cfg.flow, 1.0, cfg.control.dt_cfd)

    m = be.reset(gen.zero())
    print(f"[2] reset 后 Cl={m.cl:.5f} Cd={m.cd:.5f}；cl_ref={be.cl_ref:.5f}")

    t0 = time.time()
    cl_base = [be.step(0.0, 0.0).cl_mean for _ in range(args.steps)]
    dt_step = (time.time() - t0) / (args.steps * cfg.control.n_substeps)
    print(f"[3] 无控制推进 {args.steps} 控制步：Cl 漂移 {cl_base[-1] - cl_base[0]:+.5f}；平均 {dt_step:.2f} s/时间步")

    be.reset(gen.zero())
    cl_jet = [be.step(0.0 if k == 0 else 1.0, 1.0).cl_mean for k in range(args.steps)]
    print(f"[4] 下射流 u=+1：Cl 变化 {cl_jet[-1] - cl_base[-1]:+.5f}（应为正）")
    be.reset(gen.zero())
    cl_jet2 = [be.step(0.0 if k == 0 else -1.0, -1.0).cl_mean for k in range(args.steps)]
    print(f"[5] 上射流 u=-1：Cl 变化 {cl_jet2[-1] - cl_base[-1]:+.5f}（应为负）")

    be.reset(gen.step(3.0, 0.0, 0.2))
    cl_g = [be.step(0.0, 0.0).cl_mean for _ in range(args.steps)]
    print(f"[6] 阵风 +3°：Cl 变化 {cl_g[-1] - cl_base[-1]:+.5f}（应为正，约 +0.1~0.3）")

    ok = cl_jet[-1] > cl_base[-1] and cl_jet2[-1] < cl_base[-1] and cl_g[-1] > cl_base[-1]
    n_ep = cfg.control.episode_steps * cfg.control.n_substeps
    print(f"\n每回合 {n_ep} 个时间步，预计 {n_ep * dt_step / 60:.1f} 分钟/回合/实例")
    print("自检通过 ✔" if ok else "自检未通过：检查边界条件是否引用了 u_ctrl / gust_aoa 表达式（见 docs/fluent_setup.md）")
    be.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    np.set_printoptions(precision=5)
    main()
