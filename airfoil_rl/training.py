"""训练工具：向量化环境、算法构建、固定工况评估回调。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config, clone_config
from .controllers import PolicyController, run_episode
from .env import AirfoilLiftEnv, make_env_fn


def build_vec_env(cfg: Config, n_envs: int, seed: int, monitor_dir: Optional[str] = None):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env_fn(cfg, rank=i, seed=seed, monitor_dir=monitor_dir) for i in range(n_envs)]
    kind = cfg.train.vec_env
    if kind == "auto":
        # Fluent 每个环境是一个独立进程，用子进程并行；代理模型很快，单进程即可
        kind = "subproc" if (cfg.backend == "fluent" and n_envs > 1) else "dummy"
    venv = SubprocVecEnv(fns, start_method="spawn") if kind == "subproc" else DummyVecEnv(fns)
    venv.seed(seed)
    return venv


def build_model(cfg: Config, env, run_dir: Optional[str] = None):
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise

    t = cfg.train
    tb = None
    try:
        import tensorboard  # noqa: F401

        tb = os.path.join(run_dir, "tb") if run_dir else None
    except ImportError:
        pass
    common = dict(seed=cfg.seed, device=t.device, verbose=1, tensorboard_log=tb)
    pk = dict(net_arch=list(t.net_arch))
    if t.algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            learning_rate=t.learning_rate,
            buffer_size=t.buffer_size,
            batch_size=t.batch_size,
            gamma=t.gamma,
            tau=t.tau,
            learning_starts=t.learning_starts,
            train_freq=t.train_freq,
            gradient_steps=t.gradient_steps,
            ent_coef=t.ent_coef,
            policy_kwargs=pk,
            **common,
        )
    if t.algo == "td3":
        noise = NormalActionNoise(mean=np.zeros(1), sigma=0.2 * np.ones(1))
        return TD3(
            "MlpPolicy",
            env,
            learning_rate=t.learning_rate,
            buffer_size=t.buffer_size,
            batch_size=t.batch_size,
            gamma=t.gamma,
            tau=t.tau,
            learning_starts=t.learning_starts,
            train_freq=t.train_freq,
            gradient_steps=t.gradient_steps,
            action_noise=noise,
            policy_kwargs=pk,
            **common,
        )
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=t.learning_rate,
        n_steps=t.n_steps,
        batch_size=t.ppo_batch_size,
        n_epochs=t.n_epochs,
        gamma=t.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=list(t.net_arch), vf=list(t.net_arch))),
        **common,
    )


def load_model(path: str, algo: str, env=None, device: str = "auto", **overrides):
    from stable_baselines3 import PPO, SAC, TD3

    cls = {"sac": SAC, "td3": TD3, "ppo": PPO}[algo]
    return cls.load(path, env=env, device=device, **overrides)


def make_finetune_callback(collect_steps: int, critic_steps: int):
    """迁移微调的三段式预热（仅用于 SAC/TD3 加载预训练模型时）：

    1. [0, collect_steps)：gradient_steps=0，只用预训练策略（而不是 SB3 默认的均匀随机动作）采样；
    2. [collect_steps, collect_steps+critic_steps)：恢复梯度更新，但冻结 Actor，只让 Critic 适应新环境；
    3. 之后：正常更新。

    直接在空回放池上以较高 UTD 更新时，Critic 会在少量新数据上过拟合并把策略带偏
    （代理模型上实测评估 RMSE 先从 0.027 劣化到 0.045，约 120 回合后才恢复到 0.029）。
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class FineTuneWarmup(BaseCallback):
        def _set_actor_trainable(self, flag: bool) -> None:
            for p in self.model.actor.parameters():
                p.requires_grad_(flag)

        def _on_training_start(self) -> None:
            self.grad_steps = self.model.gradient_steps
            self.model.learning_starts = 0
            self.model.gradient_steps = 0
            self._set_actor_trainable(False)
            self.phase = 0
            print(f"[finetune] 阶段1：用预训练策略采样 {collect_steps} 步（不更新）")

        def _on_step(self) -> bool:
            if self.phase == 0 and self.num_timesteps >= collect_steps:
                self.model.gradient_steps = self.grad_steps
                self.phase = 1
                print(f"[finetune] 阶段2：只更新 Critic {critic_steps} 步")
            if self.phase == 1 and self.num_timesteps >= collect_steps + critic_steps:
                self._set_actor_trainable(True)
                self.phase = 2
                print("[finetune] 阶段3：正常 SAC 更新")
            return True

        def _on_training_end(self) -> None:
            self._set_actor_trainable(True)
            self.model.gradient_steps = self.grad_steps

    return FineTuneWarmup()


def make_scenario_callback(
    cfg: Config, run_dir: str, eval_env: Optional[AirfoilLiftEnv] = None, eval_at_start: bool = False
):
    """在固定阵风工况上评估确定性策略，按平均 Cl RMSE 保存 best_model.zip。

    eval_at_start=True（加载预训练模型微调时）：训练前先评估一次作为基线，微调后的模型只有更好时才覆盖 best_model。
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class ScenarioEvalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.env = eval_env
            self.best = np.inf
            self.history: list = []
            self.last_eval = 0

        def _on_training_start(self) -> None:
            if self.env is None:
                ecfg = clone_config(cfg)
                self.env = AirfoilLiftEnv(ecfg, rank=10_000)  # rank 区分 Fluent 工作目录
            scen = self.env.gust_gen.standard_scenarios()
            names = list(scen)[: max(cfg.train.n_eval_episodes, 1)]
            self.scenarios = {n: scen[n] for n in names}
            if eval_at_start:
                self._evaluate()

        def _on_step(self) -> bool:
            if cfg.train.eval_freq <= 0 or self.num_timesteps - self.last_eval < cfg.train.eval_freq:
                return True
            self.last_eval = self.num_timesteps
            self._evaluate()
            return True

        def _evaluate(self) -> None:
            ctrl = PolicyController(self.model)
            res = {n: run_episode(self.env, ctrl, g, seed=i)["metrics"] for i, (n, g) in enumerate(self.scenarios.items())}
            rmse = float(np.mean([m["cl_rmse"] for m in res.values()]))
            ret = float(np.mean([m["return"] for m in res.values()]))
            u_rms = float(np.mean([m["u_rms"] for m in res.values()]))
            self.logger.record("eval/cl_rmse", rmse)
            self.logger.record("eval/return", ret)
            self.logger.record("eval/u_rms", u_rms)
            self.history.append({"timesteps": self.num_timesteps, "cl_rmse": rmse, "return": ret, "u_rms": u_rms})
            with open(os.path.join(run_dir, "eval_history.json"), "w", encoding="utf-8") as fh:
                json.dump(self.history, fh, indent=1)
            if rmse < self.best:
                self.best = rmse
                self.model.save(os.path.join(run_dir, "best_model"))
                print(f"[eval] step={self.num_timesteps}: 新的最佳 Cl RMSE={rmse:.5f}（return={ret:.2f}）")
            else:
                print(f"[eval] step={self.num_timesteps}: Cl RMSE={rmse:.5f}（最佳 {self.best:.5f}）")

        def _on_training_end(self) -> None:
            if self.env is not None and eval_env is None:
                self.env.close()

    return ScenarioEvalCallback()


def default_run_dir(cfg: Config, name: Optional[str] = None) -> str:
    import time

    name = name or f"{cfg.backend}_{cfg.train.algo}_{time.strftime('%Y%m%d-%H%M%S')}"
    path = Path(cfg.train.save_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
