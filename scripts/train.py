#!/usr/bin/env python
"""训练 RL 升力稳定控制器。

示例：
  # 1) 代理模型上预训练（CPU 约 20~40 分钟）
  python scripts/train.py --config configs/surrogate.yaml --run-name sac_surrogate

  # 2) 迁移到 Fluent 微调（加载代理模型上的策略；Fluent 并行 4 个实例）
  python scripts/train.py --config configs/fluent.yaml \
      --init-model runs/sac_surrogate/best_model.zip --run-name sac_fluent

  # 命令行覆盖任意参数
  python scripts/train.py --config configs/surrogate.yaml --set train.total_timesteps=50000 --set seed=3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airfoil_rl.config import load_config, save_config  # noqa: E402
from airfoil_rl.training import (  # noqa: E402
    build_model,
    build_vec_env,
    default_run_dir,
    load_model,
    make_scenario_callback,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[], help="YAML 配置，可多次给出（后者覆盖前者）")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖单个参数，如 train.n_envs=8")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--init-model", default=None, help="从已有模型继续训练（如代理模型预训练结果）")
    ap.add_argument("--keep-hparams", action="store_true", help="加载模型时保留其原超参数（默认用当前配置覆盖）")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = default_run_dir(cfg, args.run_name)
    save_config(cfg, Path(run_dir) / "config.yaml")
    print(f"运行目录: {run_dir}")
    print(f"后端: {cfg.backend}  算法: {cfg.train.algo}  {cfg.summary()}")

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure

    env = build_vec_env(cfg, cfg.train.n_envs, cfg.seed, monitor_dir=str(Path(run_dir) / "monitor"))

    if args.init_model:
        t = cfg.train
        overrides = {}
        if not args.keep_hparams and t.algo in ("sac", "td3"):
            overrides = dict(
                learning_rate=t.learning_rate,
                learning_starts=t.learning_starts,
                gradient_steps=t.gradient_steps,
                train_freq=t.train_freq,
                batch_size=t.batch_size,
                gamma=t.gamma,
            )
        model = load_model(args.init_model, t.algo, env=env, device=t.device, **overrides)
        print(f"已加载初始模型 {args.init_model}（回放缓存为空，前 {model.learning_starts} 步只收集数据）")
    else:
        model = build_model(cfg, env, run_dir)

    fmt = ["stdout", "csv"] + (["tensorboard"] if model.tensorboard_log else [])
    model.set_logger(configure(run_dir, fmt))

    callbacks = [
        CheckpointCallback(
            save_freq=max(cfg.train.checkpoint_freq // cfg.train.n_envs, 1),
            save_path=str(Path(run_dir) / "checkpoints"),
            name_prefix="model",
            save_replay_buffer=cfg.backend == "fluent",  # Fluent 样本昂贵，保存回放缓存以便断点续训
        )
    ]
    if cfg.train.eval_freq > 0:
        callbacks.append(make_scenario_callback(cfg, run_dir))

    try:
        model.learn(total_timesteps=cfg.train.total_timesteps, callback=CallbackList(callbacks), log_interval=4)
    finally:
        model.save(str(Path(run_dir) / "final_model"))
        if cfg.backend == "fluent" and hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(str(Path(run_dir) / "replay_buffer"))
        env.close()
    print(f"训练结束，模型保存在 {run_dir}")


if __name__ == "__main__":
    main()
