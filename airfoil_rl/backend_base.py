"""后端（代理模型 / Fluent）统一接口。

环境 (env.py) 只依赖这里定义的接口，因此同一套 RL 代码可以先在代理模型上快速训练，
再无缝切换到 Fluent 上微调与验证。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .gust import GustSignal


class SimulationDiverged(RuntimeError):
    """求解发散或得到非物理结果。环境捕获后终止回合，下一回合 reset 时重新读入初始流场。"""


@dataclass
class Measurement:
    t: float  # 回合内时间 [s]
    cl: float  # 控制周期末的瞬时升力系数
    cd: float
    cl_series: np.ndarray  # 本控制周期内每个 CFD 时间步的 Cl
    cd_series: np.ndarray
    alpha_g: float  # 控制周期末的阵风攻角（真值）[rad]
    probes: Optional[np.ndarray] = None  # 表面压力系数等
    extra: dict = field(default_factory=dict)

    @property
    def cl_mean(self) -> float:
        return float(np.mean(self.cl_series))

    @property
    def cd_mean(self) -> float:
        return float(np.mean(self.cd_series))


class Backend(ABC):
    dt_cfd: float
    n_substeps: int

    @property
    def dt_ctrl(self) -> float:
        return self.dt_cfd * self.n_substeps

    @property
    @abstractmethod
    def cl_ref(self) -> float:
        """无阵风、无控制时的稳态升力系数（控制目标的默认值）。"""

    @property
    @abstractmethod
    def cd_ref(self) -> float: ...

    @property
    def n_probes(self) -> int:
        return 0

    @abstractmethod
    def reset(self, gust: GustSignal) -> Measurement:
        """回到初始（充分发展、无控制）流场，并装载新的阵风信号。"""

    @abstractmethod
    def step(self, u_old: float, u_new: float) -> Measurement:
        """推进一个控制周期；控制量在周期内从 u_old 线性过渡到 u_new。"""

    def close(self) -> None:  # noqa: B027
        pass
