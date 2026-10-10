"""阵风（来流攻角扰动）信号生成。

阵风以“来流方向角扰动 alpha_g(t)”的形式施加：
  * 代理模型：有效攻角 alpha = alpha0 + alpha_g(t)
  * Fluent：远场速度入口方向 = alpha0 + gust_aoa，gust_aoa 每个时间步由 Python 写入命名表达式

所有信号预先在细时间网格上生成并插值，保证同一个随机种子在两个后端上得到完全相同的扰动，
从而可以做公平对比与系统辨识。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import FlowConfig, GustConfig


@dataclass
class GustSignal:
    t: np.ndarray  # 时间网格 [s]
    alpha: np.ndarray  # 阵风攻角 [rad]
    kind: str = "custom"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.alpha = np.asarray(self.alpha, dtype=float)
        self._rate = np.gradient(self.alpha, self.t) if len(self.t) > 1 else np.zeros_like(self.alpha)

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.t, self.alpha))

    def rate(self, t: float) -> float:
        return float(np.interp(t, self.t, self._rate))

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def describe(self) -> str:
        p = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())
        return f"{self.kind}({p})"


def _envelope(t: np.ndarray, t_start: float, t_ramp: float) -> np.ndarray:
    """0 -> 1 的平滑包络（smoothstep），避免突加扰动引起数值冲击。"""
    if t_ramp <= 0:
        return (t >= t_start).astype(float)
    x = np.clip((t - t_start) / t_ramp, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class GustGenerator:
    """按配置随机生成阵风信号。"""

    def __init__(self, cfg: GustConfig, flow: FlowConfig, duration: float, dt: float):
        self.cfg = cfg
        self.flow = flow
        self.duration = float(duration)
        self.dt = float(dt)
        n = int(np.ceil(self.duration / self.dt)) + 2
        # 网格比 CFD 步长细一倍，插值误差可忽略
        self.t = np.linspace(0.0, (n - 1) * self.dt, 2 * n - 1)
        self.tc = flow.convective_time

    # ------------------------------------------------------------------ helpers
    def _freq_from_k(self, k: float) -> float:
        return k * self.flow.u_inf / (np.pi * self.flow.chord)

    def _wrap(self, alpha: np.ndarray, kind: str, params: dict, ramp: bool = True) -> GustSignal:
        if ramp:
            alpha = alpha * _envelope(self.t, self.cfg.start_delay_conv * self.tc, self.cfg.ramp_conv * self.tc)
        return GustSignal(self.t.copy(), alpha, kind, params)

    # ------------------------------------------------------------------ builders
    def zero(self) -> GustSignal:
        return GustSignal(self.t.copy(), np.zeros_like(self.t), "zero", {})

    def sine(self, amp_deg: float, k: float, phase: float = 0.0) -> GustSignal:
        f = self._freq_from_k(k)
        alpha = np.deg2rad(amp_deg) * np.sin(2 * np.pi * f * self.t + phase)
        return self._wrap(alpha, "sine", {"amp_deg": amp_deg, "k": k, "phase": phase})

    def multisine(self, amp_deg: float, ks: list, phases: list) -> GustSignal:
        alpha = np.zeros_like(self.t)
        for k, ph in zip(ks, phases):
            alpha += np.sin(2 * np.pi * self._freq_from_k(k) * self.t + ph)
        peak = np.max(np.abs(alpha)) or 1.0
        alpha *= np.deg2rad(amp_deg) / peak
        return self._wrap(alpha, "multisine", {"amp_deg": amp_deg, "ks": list(ks)})

    def one_minus_cos(self, gusts: list) -> GustSignal:
        """gusts: [(amp_deg, length_chords, t0_conv), ...]，离散 1-cos 阵风（适航规范常用）。"""
        alpha = np.zeros_like(self.t)
        for amp_deg, length, t0c in gusts:
            T = length * self.flow.chord / self.flow.u_inf
            t0 = t0c * self.tc
            mask = (self.t >= t0) & (self.t <= t0 + T)
            alpha[mask] += 0.5 * np.deg2rad(amp_deg) * (1 - np.cos(2 * np.pi * (self.t[mask] - t0) / T))
        return GustSignal(self.t.copy(), alpha, "one_minus_cos", {"n": len(gusts), "gusts": gusts})

    def dryden(self, amp_deg: float, scale_chords: float, rng: np.random.Generator) -> GustSignal:
        """Dryden 垂向湍流谱的频域合成；amp_deg 定义为 2 倍 RMS。"""
        n = len(self.t)
        dt = self.t[1] - self.t[0]
        L = scale_chords * self.flow.chord
        U = self.flow.u_inf
        omega = 2 * np.pi * np.fft.rfftfreq(n, dt)
        x = L * omega / U
        psd = (L / (np.pi * U)) * (1 + 3 * x**2) / (1 + x**2) ** 2
        spec = np.sqrt(psd) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(omega)))
        spec[0] = 0.0
        w = np.fft.irfft(spec, n)
        w = w / (np.std(w) + 1e-12) * np.deg2rad(amp_deg) / 2.0
        w = np.clip(w, -1.5 * np.deg2rad(amp_deg), 1.5 * np.deg2rad(amp_deg))
        return self._wrap(w, "dryden", {"amp_deg": amp_deg, "L_c": scale_chords})

    def step(self, amp_deg: float, t0_conv: float, rise_conv: float = 0.3) -> GustSignal:
        """平滑阶跃，用于系统辨识。"""
        alpha = np.deg2rad(amp_deg) * _envelope(self.t, t0_conv * self.tc, rise_conv * self.tc)
        return GustSignal(self.t.copy(), alpha, "step", {"amp_deg": amp_deg, "t0_conv": t0_conv})

    # ------------------------------------------------------------------ random
    def sample(self, rng: np.random.Generator, kind: Optional[str] = None) -> GustSignal:
        cfg = self.cfg
        if kind is None:
            kinds = list(cfg.kinds)
            probs = None if cfg.probs is None else np.asarray(cfg.probs, dtype=float) / np.sum(cfg.probs)
            kind = str(rng.choice(kinds, p=probs))
        amp = float(rng.uniform(*cfg.amp_deg)) * (1.0 if rng.random() < 0.5 else -1.0)
        k_lo, k_hi = cfg.reduced_freq
        if kind == "sine":
            return self.sine(abs(amp), float(rng.uniform(k_lo, k_hi)), float(rng.uniform(0, 2 * np.pi)))
        if kind == "multisine":
            ks = sorted(rng.uniform(k_lo, k_hi, size=3).tolist())
            return self.multisine(abs(amp), ks, rng.uniform(0, 2 * np.pi, size=3).tolist())
        if kind == "one_minus_cos":
            total_conv = self.duration / self.tc
            n_g = int(rng.integers(1, 4))
            gusts = []
            t_cursor = cfg.start_delay_conv + rng.uniform(0.0, 3.0)
            for _ in range(n_g):
                length = float(rng.uniform(*cfg.gust_length_chords))
                if t_cursor + length > total_conv - 1.0:
                    break
                a = float(rng.uniform(*cfg.amp_deg)) * (1.0 if rng.random() < 0.5 else -1.0)
                gusts.append((a, length, float(t_cursor)))
                t_cursor += length + rng.uniform(1.0, 6.0)
            if not gusts:
                gusts.append((amp, float(cfg.gust_length_chords[0]), cfg.start_delay_conv))
            return self.one_minus_cos(gusts)
        if kind == "dryden":
            return self.dryden(abs(amp), float(rng.uniform(*cfg.dryden_scale_chords)), rng)
        if kind == "zero":
            return self.zero()
        raise ValueError(f"未知阵风类型: {kind}")

    def standard_scenarios(self) -> dict:
        """固定的评估工况（不随训练随机性变化），用于对比 无控制/PID/RL。"""
        rng = np.random.default_rng(12345)
        return {
            "sine_k0.1_3deg": self.sine(3.0, 0.1),
            "sine_k0.3_3deg": self.sine(3.0, 0.3),
            "1-cos_H5_4deg": self.one_minus_cos([(4.0, 5.0, 2.0), (-3.0, 8.0, 12.0)]),
            "dryden_L5_3deg": self.dryden(3.0, 5.0, rng),
            "multisine_3deg": self.multisine(3.0, [0.07, 0.17, 0.33], [0.0, 1.3, 2.1]),
        }
