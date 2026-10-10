"""强化学习 + Fluent 的翼型升力稳定（阵风载荷减缓）控制。

主要模块：
    config         配置 dataclass 与 YAML 加载
    gust           阵风扰动信号
    surrogate      非定常气动降阶代理模型（快速预训练）
    fluent_backend PyFluent 实时耦合后端
    env            Gymnasium 环境（MDP 定义）
    controllers    无控制 / PID / RL 策略 及评估工具
    sysid          用 Fluent 开环数据标定代理模型
"""

from .config import Config, load_config, save_config

__all__ = ["Config", "load_config", "save_config"]
__version__ = "0.1.0"
