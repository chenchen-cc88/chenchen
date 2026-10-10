"""翼型非定常气动降阶代理模型（用于快速预训练与调试，参数可由 Fluent 数据辨识）。

模型由三部分组成（详细推导见 docs/algorithm.md 第 3 节）：

1. Wagner 函数（R.T. Jones 两指数近似）描述环量升力对攻角变化的迟滞：
       phi(s) ≈ 1 - A1 exp(-b1 s) - A2 exp(-b2 s),  s = 2 U t / c
   状态空间形式：
       x1' = -b1 w x1 + a,  x2' = -b2 w x2 + a,  w = 2U/c
       a_eff = a (1 - A1 - A2) + w (A1 b1 x1 + A2 b2 x2)

2. Goman–Khrabrov 分离点动力学 + Kirchhoff 升力，产生失速与动态迟滞：
       tau1 f' = f_st(a_eff - tau2 a') - f,  f_st(a) = 1 / (1 + exp(s (|a| - a_ss)))
       Cn = Cla a_eff ((1 + sqrt f)/2)^2,   Cc = eta Cla a_eff^2 sqrt f

3. 尾缘射流（流体 Gurney 襟翼）执行器：
       u>0 下表面尾缘吹气 -> 增升；u<0 上表面尾缘吹气 -> 减升
       J(u) = K± |u(t - t_d)|^p * eff(f)，t_d 为执行器纯延迟，经快/慢两条一阶通道作用于升力：
       dCl_jet = phi_j * c1 + (1 - phi_j) * c2,  c_i' = (J - c_i) / tau_i

另外叠加 Ornstein–Uhlenbeck 过程模拟湍流/分离引起的随机升力脉动。
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .backend_base import Backend, Measurement, SimulationDiverged
from .config import Config, FlowConfig
from .gust import GustSignal


@dataclass
class SurrogateParams:
    # 静态气动
    cl_alpha: float = 5.9  # 升力线斜率 [1/rad]（NACA0012, Re~2e5）
    alpha_zl_deg: float = 0.0  # 零升攻角
    alpha_stall_deg: float = 12.0  # f_st = 0.5 对应的攻角
    stall_sharpness: float = 40.0  # f_st 的陡峭度 [1/rad]
    eta_chord: float = 0.95  # 前缘吸力恢复系数
    cd0: float = 0.011
    # Wagner 函数（R.T. Jones 近似）
    wagner_A1: float = 0.165
    wagner_b1: float = 0.0455
    wagner_A2: float = 0.335
    wagner_b2: float = 0.3
    wagner_time_scale: float = 1.0  # >1 表示环量建立更慢（辨识参数）
    # 分离动力学（对流时间 c/U 为单位）
    tau_sep: float = 3.0
    tau_sep_delay: float = 1.5
    # 尾缘射流执行器
    jet_gain_pos: float = 0.40  # u=+1 时稳态 dCl
    jet_gain_neg: float = 0.35  # u=-1 时稳态 |dCl|
    jet_exponent: float = 1.0
    jet_fast_frac: float = 0.4  # 快通道占比
    jet_tau_fast: float = 0.15  # [c/U]
    jet_tau_slow: float = 1.5  # [c/U]
    jet_delay: float = 0.3  # 纯延迟 [c/U]：射流在腔体/缝道内建立及尾缘涡结构形成所需时间（不参与梯度辨识）
    jet_eff_separated: float = 0.4  # 完全分离时的射流效率（附着时为 1）
    jet_cd_gain: float = 0.004  # 射流引起的阻力增量（每单位 |u|）
    # 随机脉动
    turb_sigma: float = 0.003
    turb_sigma_sep: float = 0.03  # 分离程度 (1-f) 引起的附加脉动
    turb_tau: float = 0.5  # [c/U]

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "SurrogateParams":
        p = cls()
        for k, v in (d or {}).items():
            if not hasattr(p, k):
                raise KeyError(f"未知代理模型参数: {k}")
            setattr(p, k, float(v))
        return p

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class SurrogateAirfoil:
    """纯模型（不含 RL 接口）。状态 y = [x1, x2, f, c1, c2]。"""

    def __init__(self, params: SurrogateParams, flow: FlowConfig):
        self.p = params
        self.flow = flow
        self.tc = flow.chord / flow.u_inf
        self.alpha0 = math.radians(flow.alpha0_deg)
        self.y = np.zeros(5)
        self.turb = 0.0

    # ------------------------------------------------------------------ static
    def f_static(self, a: float) -> float:
        p = self.p
        z = p.stall_sharpness * (abs(a) - math.radians(p.alpha_stall_deg))
        z = min(max(z, -60.0), 60.0)
        return 1.0 / (1.0 + math.exp(z))

    def _jet(self, u: float, f: float) -> float:
        p = self.p
        mag = abs(u) ** p.jet_exponent
        j = p.jet_gain_pos * mag if u >= 0 else -p.jet_gain_neg * mag
        eff = p.jet_eff_separated + (1.0 - p.jet_eff_separated) * f
        return j * eff

    def _alpha_in(self, gust: GustSignal, t: float) -> tuple[float, float]:
        a = self.alpha0 + gust(t) - math.radians(self.p.alpha_zl_deg)
        return a, gust.rate(t)

    def _alpha_eff(self, y: np.ndarray, a_in: float) -> float:
        p = self.p
        w = 2.0 / self.tc / p.wagner_time_scale
        return a_in * (1.0 - p.wagner_A1 - p.wagner_A2) + w * (p.wagner_A1 * p.wagner_b1 * y[0] + p.wagner_A2 * p.wagner_b2 * y[1])

    # ------------------------------------------------------------------ dynamics
    def derivs(self, y: np.ndarray, t: float, u: float, gust: GustSignal) -> np.ndarray:
        p = self.p
        tc = self.tc
        w = 2.0 / tc / p.wagner_time_scale
        a_in, a_dot = self._alpha_in(gust, t)
        a_eff = self._alpha_eff(y, a_in)
        f = min(max(y[2], 0.0), 1.0)
        f_target = self.f_static(a_eff - p.tau_sep_delay * tc * a_dot)
        j = self._jet(u, f)
        return np.array(
            [
                -p.wagner_b1 * w * y[0] + a_in,
                -p.wagner_b2 * w * y[1] + a_in,
                (f_target - y[2]) / (p.tau_sep * tc),
                (j - y[3]) / (p.jet_tau_fast * tc),
                (j - y[4]) / (p.jet_tau_slow * tc),
            ]
        )

    def rk4(self, t: float, dt: float, u: float, gust: GustSignal) -> None:
        y = self.y
        k1 = self.derivs(y, t, u, gust)
        k2 = self.derivs(y + 0.5 * dt * k1, t + 0.5 * dt, u, gust)
        k3 = self.derivs(y + 0.5 * dt * k2, t + 0.5 * dt, u, gust)
        k4 = self.derivs(y + dt * k3, t + dt, u, gust)
        self.y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.y[2] = min(max(self.y[2], 0.0), 1.0)

    def update_turbulence(self, dt: float, rng: Optional[np.random.Generator]) -> None:
        if rng is None:
            self.turb = 0.0
            return
        p = self.p
        tau = p.turb_tau * self.tc
        sigma = p.turb_sigma + p.turb_sigma_sep * (1.0 - self.y[2])
        decay = math.exp(-dt / tau)
        self.turb = self.turb * decay + sigma * math.sqrt(1.0 - decay * decay) * rng.standard_normal()

    def outputs(self, t: float, u: float, gust: GustSignal) -> tuple[float, float, dict]:
        p = self.p
        a_in, _ = self._alpha_in(gust, t)
        a_eff = self._alpha_eff(self.y, a_in)
        f = self.y[2]
        sf = math.sqrt(max(f, 0.0))
        cn = p.cl_alpha * a_eff * ((1.0 + sf) / 2.0) ** 2
        cc = p.eta_chord * p.cl_alpha * a_eff * a_eff * sf
        a_geo = a_eff + math.radians(p.alpha_zl_deg)
        cl_aero = cn * math.cos(a_geo) + cc * math.sin(a_geo)
        cl_jet = p.jet_fast_frac * self.y[3] + (1.0 - p.jet_fast_frac) * self.y[4]
        cl = cl_aero + cl_jet + self.turb
        cd = p.cd0 + cn * math.sin(a_geo) - cc * math.cos(a_geo) + p.jet_cd_gain * abs(u)
        return cl, cd, {"alpha_eff": a_eff, "f": f, "cl_jet": cl_jet}

    def reset_steady(self, gust: GustSignal) -> None:
        p = self.p
        w = 2.0 / self.tc / p.wagner_time_scale
        a_in, _ = self._alpha_in(gust, 0.0)
        self.y = np.array([a_in / (p.wagner_b1 * w), a_in / (p.wagner_b2 * w), self.f_static(a_in), 0.0, 0.0])
        self.turb = 0.0

    def steady_coeffs(self) -> tuple[float, float]:
        """无阵风、无控制的稳态 (Cl, Cd)。"""
        zero = GustSignal(np.array([0.0, 1.0]), np.zeros(2), "zero")
        saved = (self.y.copy(), self.turb)
        self.reset_steady(zero)
        cl, cd, _ = self.outputs(0.0, 0.0, zero)
        self.y, self.turb = saved
        return cl, cd


class SurrogateBackend(Backend):
    def __init__(self, cfg: Config, stochastic: bool = True, params: Optional[SurrogateParams] = None):
        self.cfg = cfg
        self.dt_cfd = cfg.control.dt_cfd
        self.n_substeps = cfg.control.n_substeps
        self.params = params or SurrogateParams.from_dict(cfg.surrogate)
        self.model = SurrogateAirfoil(self.params, cfg.flow)
        self.stochastic = stochastic
        self.rng = np.random.default_rng(cfg.seed)
        self._cl_ref, self._cd_ref = self.model.steady_coeffs()
        self.gust: Optional[GustSignal] = None
        self.t = 0.0
        self.n_delay = int(round(self.params.jet_delay * cfg.flow.convective_time / self.dt_cfd))
        self.u_buf: deque = deque()

    def seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @property
    def cl_ref(self) -> float:
        return self._cl_ref

    @property
    def cd_ref(self) -> float:
        return self._cd_ref

    def _measure(self, cl_s: list, cd_s: list, extra: dict) -> Measurement:
        return Measurement(
            t=self.t,
            cl=cl_s[-1],
            cd=cd_s[-1],
            cl_series=np.asarray(cl_s),
            cd_series=np.asarray(cd_s),
            alpha_g=self.gust(self.t),
            extra=extra,
        )

    def reset(self, gust: GustSignal) -> Measurement:
        self.gust = gust
        self.t = 0.0
        self.u_buf = deque([0.0] * self.n_delay)
        self.model.reset_steady(gust)
        cl, cd, extra = self.model.outputs(0.0, 0.0, gust)
        return self._measure([cl], [cd], extra)

    def step(self, u_old: float, u_new: float) -> Measurement:
        assert self.gust is not None, "先调用 reset()"
        rng = self.rng if self.stochastic else None
        cl_s, cd_s = [], []
        extra: dict = {}
        n = self.n_substeps
        for k in range(1, n + 1):
            u = u_old + (u_new - u_old) * k / n
            if self.n_delay:
                self.u_buf.append(u)
                u = self.u_buf.popleft()  # 执行器实际输出 = n_delay 个时间步之前的指令
            self.model.rk4(self.t, self.dt_cfd, u, self.gust)
            self.t += self.dt_cfd
            self.model.update_turbulence(self.dt_cfd, rng)
            cl, cd, extra = self.model.outputs(self.t, u, self.gust)
            cl_s.append(cl)
            cd_s.append(cd)
        if not np.all(np.isfinite(cl_s)):
            raise SimulationDiverged("代理模型出现非有限值")
        return self._measure(cl_s, cd_s, extra)
