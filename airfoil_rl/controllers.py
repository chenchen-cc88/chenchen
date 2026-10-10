"""对比用控制器与评估工具：无控制、PID(+阵风前馈)、RL 策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from .env import AirfoilLiftEnv
from .gust import GustSignal


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, info: dict) -> float: ...


class NoControl:
    name = "no_control"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> float:
        return 0.0


@dataclass
class PIDController:
    """经典基准：e = Cl - Cl_ref，u = -(Kp e + Ki ∫e + Kd de/dt) - Kff * alpha_g。

    增益以 cl_scale 归一化后的误差为输入，输出为动作 a∈[-1,1]，带积分抗饱和。
    """

    kp: float = 0.6
    ki: float = 6.0  # [1/s]
    kd: float = 0.0  # [s]
    kff: float = 0.0  # 阵风前馈 [1/deg]
    dt: float = 2e-3
    cl_scale: float = 0.1
    name: str = "pid"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_e: Optional[float] = None

    def act(self, obs: np.ndarray, info: dict) -> float:
        e = (info["cl_meas"] - info["cl_ref"]) / self.cl_scale
        de = 0.0 if self.prev_e is None else (e - self.prev_e) / self.dt
        self.prev_e = e
        u_unsat = -(self.kp * e + self.ki * self.integral + self.kd * de) - self.kff * info["alpha_g_meas_deg"]
        u = float(np.clip(u_unsat, -1.0, 1.0))
        # 抗饱和：饱和时只允许让输出退出饱和方向的积分（u 含 -Ki∫e，故 e 与 u 同号时积分会减小 |u|）
        if u == u_unsat or np.sign(e) == np.sign(u_unsat):
            self.integral += e * self.dt
        return u


class PolicyController:
    """包装 stable-baselines3 模型。"""

    def __init__(self, model: Any, name: str = "rl", deterministic: bool = True):
        self.model = model
        self.name = name
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> float:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return float(np.asarray(action).ravel()[0])


# ============================================================================ 评估
def run_episode(env: AirfoilLiftEnv, ctrl: Any, gust: Optional[GustSignal] = None, seed: int = 0) -> dict:
    """运行一个完整回合并记录时间历程（以 CFD 时间步为分辨率）。"""
    obs, info = env.reset(seed=seed, options={"gust": gust} if gust is not None else None)
    ctrl.reset()
    dt = env.cfg.control.dt_cfd
    t_f, cl_f, u_f, ag_f = [0.0], [info["cl"]], [0.0], [info["alpha_g_deg"]]
    rewards, cds = [], [info["cd"]]
    done = False
    while not done:
        a = ctrl.act(obs, info)
        u_old = env.u
        obs, r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        n = len(info["cl_series"])
        t0 = t_f[-1]
        t_f.extend(t0 + dt * np.arange(1, n + 1))
        cl_f.extend(info["cl_series"])
        u_f.extend(u_old + (info["u"] - u_old) * np.arange(1, n + 1) / n)
        ag_f.extend([info["alpha_g_deg"]] * n)
        cds.append(info["cd"])
        rewards.append(r)
        done = term or trunc
    traj = {
        "t": np.asarray(t_f),
        "cl": np.asarray(cl_f),
        "u": np.asarray(u_f),
        "alpha_g_deg": np.asarray(ag_f),
        "cd": np.asarray(cds),
        "reward": np.asarray(rewards),
        "cl_ref": env.cl_ref,
        "terminated_early": bool(len(rewards) < env.cfg.control.episode_steps),
    }
    traj["metrics"] = episode_metrics(traj, env.cfg.flow.convective_time)
    return traj


def episode_metrics(traj: dict, tc: float, skip_conv: float = 1.0) -> dict:
    """skip_conv: 丢弃前若干对流时间（阵风尚未施加）后统计。"""
    mask = traj["t"] >= skip_conv * tc
    e = traj["cl"][mask] - traj["cl_ref"]
    u = traj["u"][mask]
    return {
        "cl_rmse": float(np.sqrt(np.mean(e**2))),
        "cl_max_dev": float(np.max(np.abs(e))),
        "cl_std": float(np.std(traj["cl"][mask])),
        "u_rms": float(np.sqrt(np.mean(u**2))),
        "du_rms": float(np.sqrt(np.mean(np.diff(u) ** 2))) if len(u) > 1 else 0.0,
        "cd_mean": float(np.mean(traj["cd"])),
        "return": float(np.sum(traj["reward"])),
        "steps": int(len(traj["reward"])),
    }


def tune_pid(env: AirfoilLiftEnv, gusts: list, use_ff: bool = True, maxiter: int = 150) -> PIDController:
    """在给定工况上用 Nelder–Mead 整定 PID（以累计奖励为目标，与 RL 公平对比）。"""
    from scipy.optimize import minimize

    dt = env.cfg.control.dt_ctrl
    cs = env.cfg.reward.cl_scale

    def make(x: np.ndarray) -> PIDController:
        kp, ki, kd = np.exp(x[:3])
        kff = x[3] if use_ff else 0.0
        return PIDController(kp=kp, ki=ki, kd=kd, kff=kff, dt=dt, cl_scale=cs)

    def cost(x: np.ndarray) -> float:
        ctrl = make(x)
        return -float(np.mean([run_episode(env, ctrl, g, seed=i)["metrics"]["return"] for i, g in enumerate(gusts)]))

    # 初值：kff 取“准静态完全抵消”的量级 Clα/(K·57.3)≈0.25 [1/deg]；显式给出初始单纯形，
    # 否则 Nelder–Mead 对零初值分量的初始步长只有 2.5e-4，前馈增益实际上不会被搜索
    x0 = np.array([np.log(0.6), np.log(30.0), np.log(1e-3), 0.2 if use_ff else 0.0])
    steps = np.diag([0.7, 1.0, 1.0, 0.15])
    simplex = np.vstack([x0, x0 + steps])
    res = minimize(
        cost,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2, "initial_simplex": simplex},
    )
    best = make(res.x)
    return best
