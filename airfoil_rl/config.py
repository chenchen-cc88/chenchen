"""配置定义与加载。

所有参数集中在 dataclass 中，YAML 文件只需写出想覆盖的字段；
可以叠加多个 YAML（后者覆盖前者），也可以用 ``--set a.b=value`` 在命令行覆盖。
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class FlowConfig:
    """来流与翼型参数（SI 单位）。"""

    chord: float = 0.2  # 弦长 c [m]
    u_inf: float = 15.0  # 来流速度 U [m/s]
    alpha0_deg: float = 6.0  # 平均攻角 [deg]
    density: float = 1.225  # [kg/m^3]
    viscosity: float = 1.7894e-5  # 动力黏度 [Pa s]

    @property
    def convective_time(self) -> float:
        """对流时间 c/U [s]。"""
        return self.chord / self.u_inf

    @property
    def reynolds(self) -> float:
        return self.density * self.u_inf * self.chord / self.viscosity

    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.density * self.u_inf**2


@dataclass
class GustConfig:
    """训练时随机采样的阵风扰动。"""

    kinds: list = field(default_factory=lambda: ["sine", "one_minus_cos", "dryden", "multisine"])
    probs: Optional[list] = None  # None 表示均匀采样
    amp_deg: tuple = (1.0, 4.0)  # 阵风攻角幅值范围 [deg]
    reduced_freq: tuple = (0.05, 0.4)  # 折合频率 k = pi f c / U
    gust_length_chords: tuple = (2.0, 10.0)  # 1-cos 离散阵风长度 H/c
    dryden_scale_chords: tuple = (3.0, 10.0)  # Dryden 湍流积分尺度 L/c
    ramp_conv: float = 1.0  # 起始缓升时间（对流时间单位），避免突加扰动
    start_delay_conv: float = 0.5  # 扰动开始前的静默时间（对流时间单位）
    sensor_noise_deg: float = 0.05  # 阵风传感器（攻角探头）噪声 [deg]


@dataclass
class ControlConfig:
    """时间离散与执行器。

    控制周期 dt_ctrl = dt_cfd * n_substeps。Fluent 每个控制周期推进 n_substeps 个物理时间步，
    代理模型用同样的步长积分，保证两个后端的时间尺度一致（便于迁移）。
    """

    dt_cfd: float = 4.0e-4  # CFD 物理时间步 [s]（≈0.03 c/U）
    n_substeps: int = 5  # 每个控制周期包含的 CFD 时间步数
    episode_steps: int = 200  # 每回合控制步数（200*2ms=0.4s≈30 c/U）
    action_smoothing: float = 0.5  # 一阶平滑 u <- u + beta*(a-u)，beta∈(0,1]
    max_du: Optional[float] = None  # 每个控制周期最大变化量（None 不限）

    @property
    def dt_ctrl(self) -> float:
        return self.dt_cfd * self.n_substeps


@dataclass
class RewardConfig:
    cl_target: Optional[float] = None  # None: 使用后端给出的无阵风稳态 Cl
    cl_scale: float = 0.1  # 升力误差归一化尺度
    cd_scale: float = 0.02
    w_cl: float = 1.0  # 升力跟踪误差权重
    w_u: float = 0.02  # 控制能量（射流动量）权重
    w_du: float = 0.1  # 控制变化率权重（抑制抖振）
    w_cd: float = 0.0  # 阻力增量权重（需要兼顾阻力时打开）
    clip: float = 20.0  # 单步奖励下限 -clip
    diverge_dcl: float = 3.0  # |Cl-Cl_ref| 超过此值视为发散，回合终止
    diverge_penalty: float = 50.0


@dataclass
class ObsConfig:
    n_hist: int = 6  # Cl 历史长度（控制周期数）
    include_cd: bool = True
    include_gust_sensor: bool = True  # 上游/机头攻角探头（前馈信息）
    n_gust_hist: int = 3
    n_u_hist: int = 3  # 执行器指令历史：有执行器延迟时，“在途”指令是系统状态的一部分
    include_err_ema: bool = True  # 升力误差的指数滑动平均（相当于积分作用，消除慢变扰动下的静差）
    err_ema_conv: float = 3.0  # 滑动平均时间常数（对流时间 c/U）
    gust_scale_deg: float = 4.0
    include_probes: bool = False  # 表面压力探头（仅 Fluent 可用，需在算例中定义）
    probe_scale: float = 1.0  # 压力系数归一化尺度
    noise_cl: float = 0.002  # 升力测量噪声（标准差）


@dataclass
class FluentConfig:
    """PyFluent 耦合参数。算例准备见 docs/fluent_setup.md。"""

    case_file: str = "fluent_case/airfoil_rl.cas.h5"
    init_data_file: str = "fluent_case/airfoil_rl_init.dat.h5"  # 充分发展流场，每回合 reset 时读入
    meta_file: str = "fluent_case/airfoil_rl_meta.yaml"  # setup 脚本写出的 cl_ref/cd_ref 等
    processors: int = 4
    product_version: Optional[str] = None  # 例如 "25.1.0"；None 自动选择已安装版本
    ui_mode: str = "no_gui"
    additional_arguments: str = ""
    start_timeout: int = 600
    server_info_file: Optional[str] = None  # 连接已启动的 Fluent（集群上手动启动），可含 {rank} 占位符
    workdir: str = "runs/fluent_work"
    max_iter_per_step: int = 20  # 每个物理时间步的内迭代数
    # 名称约定（与 setup_fluent_case.py / 文档一致）
    expr_control: str = "u_ctrl"
    expr_gust: str = "gust_aoa"
    report_cl: str = "cl"
    report_cd: str = "cd"
    probe_reports: list = field(default_factory=list)  # 例如 ["p_u1","p_u2",...]（Cp 报告）
    cl_ref: Optional[float] = None  # None: 读 meta_file，再不行则用 reset 时的瞬时 Cl
    cd_ref: Optional[float] = None
    cl_abs_limit: float = 5.0  # |Cl| 超过此值认为求解发散


@dataclass
class TrainConfig:
    algo: str = "sac"  # sac | td3 | ppo
    total_timesteps: int = 200_000
    n_envs: int = 4
    vec_env: str = "auto"  # auto | dummy | subproc
    learning_rate: float = 1e-4
    buffer_size: int = 300_000
    batch_size: int = 256
    gamma: float = 0.95
    tau: float = 0.005
    learning_starts: int = 2_000
    train_freq: int = 1
    gradient_steps: int = 1  # Fluent 下样本昂贵，可调大（UTD 比）
    ent_coef: Any = "auto"
    net_arch: list = field(default_factory=lambda: [256, 256])
    # PPO 专用
    n_steps: int = 1024
    ppo_batch_size: int = 256
    n_epochs: int = 10
    # 迁移微调（--init-model）：先用预训练策略采样、再只更新 Critic、最后正常 SAC 更新，避免空回放池导致策略退化
    finetune_collect_steps: int = 800  # 只采样不更新的步数（用策略本身而非随机动作）
    finetune_critic_steps: int = 2_000  # 冻结 Actor、只更新 Critic 的步数
    eval_freq: int = 10_000  # 以总环境步计
    n_eval_episodes: int = 4
    checkpoint_freq: int = 20_000
    save_dir: str = "runs"
    device: str = "auto"


@dataclass
class Config:
    seed: int = 0
    backend: str = "surrogate"  # surrogate | fluent
    flow: FlowConfig = field(default_factory=FlowConfig)
    gust: GustConfig = field(default_factory=GustConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    surrogate: dict = field(default_factory=dict)  # SurrogateParams 字段的覆盖值
    fluent: FluentConfig = field(default_factory=FluentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        f, c = self.flow, self.control
        return (
            f"Re={f.reynolds:.3g}, c/U={f.convective_time * 1e3:.2f} ms, "
            f"dt_cfd={c.dt_cfd * f.u_inf / f.chord:.3f} c/U, dt_ctrl={c.dt_ctrl * f.u_inf / f.chord:.3f} c/U, "
            f"episode={c.episode_steps * c.dt_ctrl * f.u_inf / f.chord:.1f} c/U"
        )


def _coerce(value: Any, type_str: str) -> Any:
    """YAML 1.1 把 ``1e-4``（无小数点）解析为字符串，这里按字段类型转换。"""
    if isinstance(value, str) and value.strip().lower() not in ("auto", "none", "null"):
        if "float" in type_str:
            try:
                return float(value)
            except ValueError:
                pass
        elif type_str.startswith("int") or "Optional[int]" in type_str:
            try:
                return int(float(value))
            except ValueError:
                pass
    if isinstance(value, int) and not isinstance(value, bool) and "float" in type_str:
        return float(value)
    return value


def _update_dataclass(obj: Any, data: dict, prefix: str = "") -> None:
    types = {f.name: str(f.type) for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"未知配置项: {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value, prefix=f"{prefix}{key}.")
        elif isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, _coerce(value, types.get(key, "")))


def _parse_override(item: str) -> dict:
    if "=" not in item:
        raise ValueError(f"--set 需要 key=value 格式，收到: {item}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    out: dict = {}
    node = out
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def load_config(paths: Optional[list] = None, overrides: Optional[list] = None) -> Config:
    """按顺序叠加 YAML 文件与命令行覆盖项。"""
    cfg = Config()
    for p in paths or []:
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _update_dataclass(cfg, data)
    for item in overrides or []:
        _update_dataclass(cfg, _parse_override(item))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.backend not in ("surrogate", "fluent"):
        raise ValueError(f"backend 只能是 surrogate 或 fluent，收到 {cfg.backend}")
    if not 0.0 < cfg.control.action_smoothing <= 1.0:
        raise ValueError("control.action_smoothing 必须在 (0, 1]")
    if cfg.control.n_substeps < 1:
        raise ValueError("control.n_substeps 必须 >= 1")
    if cfg.train.algo not in ("sac", "td3", "ppo"):
        raise ValueError("train.algo 只能是 sac/td3/ppo")
    if cfg.obs.n_hist < 1:
        raise ValueError("obs.n_hist 必须 >= 1")
    if not math.isfinite(cfg.flow.u_inf) or cfg.flow.u_inf <= 0:
        raise ValueError("flow.u_inf 必须为正")


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _plain(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_plain(v) for v in x]
        return x

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_plain(cfg.to_dict()), fh, allow_unicode=True, sort_keys=False)


def clone_config(cfg: Config) -> Config:
    return copy.deepcopy(cfg)
