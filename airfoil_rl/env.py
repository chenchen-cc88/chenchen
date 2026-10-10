"""Gymnasium 环境：阵风下的翼型升力稳定控制。

MDP 定义（详细说明见 docs/algorithm.md 第 2 节）：

状态/观测 o_k（全部归一化到 O(1)）：
    [ (Cl_{k-i} - Cl_ref)/cl_scale, i=0..n_hist-1 ]      控制周期平均升力历史（含测量噪声）
    [ (Cd_k - Cd_ref)/cd_scale ]                          可选
    [ alpha_g,{k-i} / gust_scale, i=0..n_gust_hist-1 ]   可选：机头攻角探头（前馈）
    [ (Cp_j - Cp_j,ref)/probe_scale ]                     可选：表面压力探头
    [ EMA((Cl - Cl_ref)/cl_scale) ]                       可选：误差滑动平均（积分作用）
    [ u_{k-i}, i=0..n_u_hist-1 ]                          执行器指令历史（延迟系统的“在途”控制量）

动作 a_k ∈ [-1, 1]，经一阶平滑得到执行器指令：
    u_{k+1} = clip(u_k + beta (a_k - u_k), -1, 1)
    （周期内 u 从 u_k 线性过渡到 u_{k+1}，避免边界条件突变）

奖励（二次型，类 LQR）：
    r_k = -[ w_cl * mean_n((Cl_n - Cl_ref)/cl_scale)^2       # 周期内每个 CFD 步的升力偏差
           + w_u  * u_{k+1}^2                                # 射流能耗
           + w_du * (u_{k+1} - u_k)^2                        # 抖振
           + w_cd * max(Cd - Cd_ref, 0)/cd_scale ]           # 可选：阻力代价
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backend_base import Backend, Measurement, SimulationDiverged
from .config import Config
from .gust import GustGenerator, GustSignal


def make_backend(cfg: Config, rank: int = 0, stochastic: bool = True) -> Backend:
    if cfg.backend == "surrogate":
        from .surrogate import SurrogateBackend

        return SurrogateBackend(cfg, stochastic=stochastic)
    if cfg.backend == "fluent":
        from .fluent_backend import FluentBackend

        return FluentBackend(cfg, rank=rank)
    raise ValueError(cfg.backend)


class AirfoilLiftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, backend: Optional[Backend] = None, rank: int = 0):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.backend = backend if backend is not None else make_backend(cfg, rank)
        c = cfg.control
        duration = c.episode_steps * c.dt_ctrl + 2 * c.dt_ctrl
        self.gust_gen = GustGenerator(cfg.gust, cfg.flow, duration, c.dt_cfd)
        self.fixed_gust: Optional[GustSignal] = None

        o = cfg.obs
        self.n_probes = self.backend.n_probes if o.include_probes else 0
        dim = o.n_hist + max(o.n_u_hist, 1)  # Cl 历史 + u 历史
        dim += 1 if o.include_cd else 0
        dim += o.n_gust_hist if o.include_gust_sensor else 0
        dim += self.n_probes
        dim += 1 if o.include_err_ema else 0
        self.ema_decay = float(np.exp(-c.dt_ctrl / (o.err_ema_conv * cfg.flow.convective_time)))
        self.err_ema = 0.0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.u = 0.0
        self.k = 0
        self.cl_ref = 0.0
        self.cd_ref = 0.0
        self.probe_ref: Optional[np.ndarray] = None
        self.cl_hist: deque = deque(maxlen=o.n_hist)
        self.gust_hist: deque = deque(maxlen=max(o.n_gust_hist, 1))
        self.u_hist: deque = deque(maxlen=max(o.n_u_hist, 1))
        self.last_meas: Optional[Measurement] = None
        self.gust: Optional[GustSignal] = None
        self.cl_meas = 0.0
        self.alpha_g_meas_deg = 0.0

    # ------------------------------------------------------------------ 观测
    def _sense(self, meas: Measurement) -> None:
        """带噪声的传感器读数（RL 与 PID 使用同一份测量值）。"""
        self.cl_meas = meas.cl_mean + self.cfg.obs.noise_cl * self.np_random.standard_normal()
        self.alpha_g_meas_deg = float(np.rad2deg(meas.alpha_g)) + self.cfg.gust.sensor_noise_deg * self.np_random.standard_normal()

    def _obs(self, meas: Measurement) -> np.ndarray:
        o, r = self.cfg.obs, self.cfg.reward
        parts = [np.asarray(self.cl_hist, dtype=float)]
        if o.include_cd:
            parts.append([(meas.cd_mean - self.cd_ref) / r.cd_scale])
        if o.include_gust_sensor:
            parts.append(np.asarray(self.gust_hist, dtype=float)[-o.n_gust_hist :])
        if self.n_probes:
            assert meas.probes is not None and self.probe_ref is not None
            parts.append((meas.probes - self.probe_ref) / o.probe_scale)
        if o.include_err_ema:
            parts.append([self.err_ema])
        parts.append(np.asarray(self.u_hist, dtype=float))
        return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)

    # ------------------------------------------------------------------ 奖励
    def compute_reward(self, meas: Measurement, u_old: float, u_new: float) -> tuple[float, dict]:
        r = self.cfg.reward
        e = (meas.cl_series - self.cl_ref) / r.cl_scale
        terms = {
            "cl": r.w_cl * float(np.mean(e**2)),
            "u": r.w_u * u_new**2,
            "du": r.w_du * (u_new - u_old) ** 2,
            "cd": r.w_cd * max(meas.cd_mean - self.cd_ref, 0.0) / r.cd_scale,
        }
        reward = -sum(terms.values())
        return max(reward, -r.clip), terms

    # ------------------------------------------------------------------ gym API
    def set_fixed_gust(self, gust: Optional[GustSignal]) -> None:
        """评估时固定阵风工况；传 None 恢复随机采样。"""
        self.fixed_gust = gust

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None and hasattr(self.backend, "seed"):
            self.backend.seed(seed + 7919 * self.rank)
        options = options or {}
        gust = options.get("gust") or self.fixed_gust
        if gust is None:
            gust = self.gust_gen.sample(self.np_random, options.get("gust_kind"))
        self.gust = gust

        meas = self.backend.reset(gust)
        rc = self.cfg.reward
        self.cl_ref = rc.cl_target if rc.cl_target is not None else self.backend.cl_ref
        self.cd_ref = self.backend.cd_ref
        self.probe_ref = None if meas.probes is None else np.array(meas.probes, dtype=float)
        self.u = 0.0
        self.k = 0
        self._sense(meas)
        self.cl_hist.clear()
        self.cl_hist.extend([(self.cl_meas - self.cl_ref) / rc.cl_scale] * self.cfg.obs.n_hist)
        self.gust_hist.clear()
        self.gust_hist.extend([self.alpha_g_meas_deg / self.cfg.obs.gust_scale_deg] * self.gust_hist.maxlen)
        self.u_hist.clear()
        self.u_hist.extend([0.0] * self.u_hist.maxlen)
        self.err_ema = self.cl_hist[-1]
        self.last_meas = meas
        return self._obs(meas), self._info(meas, {})

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        c, rc = self.cfg.control, self.cfg.reward
        a = float(np.clip(np.asarray(action, dtype=float).ravel()[0], -1.0, 1.0))
        u_old = self.u
        u_new = u_old + c.action_smoothing * (a - u_old)
        if c.max_du is not None:
            u_new = u_old + float(np.clip(u_new - u_old, -c.max_du, c.max_du))
        u_new = float(np.clip(u_new, -1.0, 1.0))

        try:
            meas = self.backend.step(u_old, u_new)
        except SimulationDiverged as exc:
            assert self.last_meas is not None
            info = self._info(self.last_meas, {"diverged": str(exc)})
            return self._obs(self.last_meas), -rc.diverge_penalty, True, False, info

        self.u = u_new
        self.k += 1
        reward, terms = self.compute_reward(meas, u_old, u_new)
        self._sense(meas)
        self.cl_hist.append((self.cl_meas - self.cl_ref) / rc.cl_scale)
        self.gust_hist.append(self.alpha_g_meas_deg / self.cfg.obs.gust_scale_deg)
        self.u_hist.append(u_new)
        self.err_ema = self.ema_decay * self.err_ema + (1.0 - self.ema_decay) * self.cl_hist[-1]
        self.last_meas = meas

        terminated = bool(abs(meas.cl - self.cl_ref) > rc.diverge_dcl)
        if terminated:
            reward = -rc.diverge_penalty
        truncated = self.k >= c.episode_steps
        return self._obs(meas), float(reward), terminated, truncated, self._info(meas, terms)

    def _info(self, meas: Measurement, terms: dict) -> dict:
        return {
            "t": meas.t,
            "cl": meas.cl,
            "cl_mean": meas.cl_mean,
            "cd": meas.cd_mean,
            "cl_ref": self.cl_ref,
            "cl_meas": self.cl_meas,  # 带噪声测量值（控制器只应使用测量值）
            "alpha_g_meas_deg": self.alpha_g_meas_deg,
            "u": self.u,
            "alpha_g_deg": float(np.rad2deg(meas.alpha_g)),
            "cl_series": meas.cl_series,
            "reward_terms": terms,
            **{k: v for k, v in meas.extra.items() if np.isscalar(v)},
            **({"diverged": terms["diverged"]} if "diverged" in terms else {}),
        }

    def close(self) -> None:
        self.backend.close()


def make_env_fn(cfg: Config, rank: int = 0, seed: int = 0, monitor_dir: Optional[str] = None) -> Callable[[], gym.Env]:
    """供 SB3 的 DummyVecEnv/SubprocVecEnv 使用的工厂函数。"""

    def _init() -> gym.Env:
        from stable_baselines3.common.monitor import Monitor

        env = AirfoilLiftEnv(cfg, rank=rank)
        env.action_space.seed(seed + rank)
        filename = None
        if monitor_dir is not None:
            import os

            os.makedirs(monitor_dir, exist_ok=True)
            filename = os.path.join(monitor_dir, f"env{rank}")
        return Monitor(env, filename=filename)

    return _init
