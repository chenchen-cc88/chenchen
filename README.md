# 强化学习 + Fluent：阵风下的翼型升力稳定控制

用**深度强化学习（SAC）**在线耦合 **Ansys Fluent**（PyFluent / gRPC），控制 NACA0012 尾缘上下表面的**射流（流体 Gurney 襟翼）**，
在各种阵风扰动下把升力系数稳定在目标值，即阵风载荷减缓（gust load alleviation）。

本项目提供完整、可运行的工程实现：

* **两个可互换的环境后端**：Fluent 实时耦合；以及标定过的非定常气动**代理模型**，用于分钟级预训练与调试；
* **MDP 设计**：多类阵风（正弦 / 1-cos 离散阵风 / Dryden 湍流 / 多频）、带噪传感器、执行器平滑与延迟、二次型奖励；
* **Fluent 耦合**：运行时只用命名表达式 + 报告定义。不需要 UDF 和动网格，接口已对照 PyFluent 自带的 Fluent 2024R2~2026R1 API 定义核对；
* **算例自动建立脚本**：网格 → 湍流模型 / 表达式 / 边界 / 报告 → 充分发展的初始流场；
* **sim-to-CFD 迁移**：用 Fluent 开环数据做**系统辨识**标定代理模型 → 预训练 → Fluent 在线微调；
* **公平基准**：自动整定的 PID + 阵风前馈，统一的评估工况、指标表与时程图；
* 单元测试：包含一个模拟 PyFluent API 的假 Fluent 会话，在无 Fluent 的机器上也能验证耦合逻辑。

> 详细算法推导见 [docs/algorithm.md](docs/algorithm.md)，Fluent 操作步骤见 [docs/fluent_setup.md](docs/fluent_setup.md)。

---

## 方法概览

```text
           观测 o_k：Cl 历史、Cd、攻角探头、(表面 Cp)、误差滑动平均、执行器指令历史
        ┌──────────────────────────────────────────────────────────────┐
        │                                                              │
  ┌─────┴──────┐  a_k∈[-1,1]  ┌──────────────┐  u_ctrl, gust_aoa  ┌────┴─────────────────────────┐
  │  SAC 策略   │─────────────▶│ 动作平滑      │───────────────────▶│ Fluent 2D URANS (SST k-ω)     │
  │  π_φ(a|o)   │              │ u←u+β(a−u)   │  (命名表达式,       │  入口: U∞(cos,sin)(α0+α_g)    │
  └─────▲──────┘              │ 周期内线性过渡 │   每个时间步写入)    │  下缝: Vmax·max(u,0)  → 增升   │
        │                     └──────────────┘                    │  上缝: Vmax·max(−u,0) → 减升   │
        │  r_k = −[(Cl−Cl_ref)²/s² + w_u u² + w_Δu Δu²]             │  每控制周期 5 个时间步          │
        └──────────────────────────────────────────────────────────┴──────────────────────────────┘
                     （同一接口下可换成代理模型：Wagner + Goman–Khrabrov + 射流动力学）
```

| 参数 | 默认值 |
|---|---|
| 翼型 / 弦长 / 来流 | NACA0012，c = 0.2 m，U∞ = 15 m/s，Re ≈ 2.05×10⁵，α₀ = 6° |
| 阵风 | 幅值 1°~4°，折合频率 k = 0.05~0.4 |
| 执行器 | 上/下表面尾缘射流缝（x/c≈0.96，缝宽 2 mm），最大 20 m/s |
| 时间步 / 控制周期 / 回合 | 4×10⁻⁴ s（0.03 c/U）/ 5 个时间步 / 200 个控制步（30 c/U） |
| 算法 | SAC，MLP 256×256，γ = 0.95，学习率 1e-4 |

---

## 目录结构

```text
airfoil_rl/
  config.py           配置 dataclass、YAML 叠加、命令行覆盖
  gust.py             阵风信号（正弦 / 1-cos / Dryden / 多频 / 阶跃）
  backend_base.py     后端统一接口
  surrogate.py        非定常气动降阶代理模型
  fluent_backend.py   PyFluent 实时耦合后端
  env.py              Gymnasium 环境（观测 / 动作 / 奖励）
  controllers.py      无控制、PID(+前馈)、RL 策略；回合运行与指标
  sysid.py            系统辨识：用 Fluent 数据标定代理模型
  training.py         SB3 算法构建、并行环境、固定工况评估回调
scripts/
  naca_geometry.py        生成带射流缝的翼型坐标（DesignModeler / CSV）
  setup_fluent_case.py    由网格自动建立 Fluent 算例并计算初始流场
  check_fluent.py         Fluent 耦合自检
  calibrate_surrogate.py  开环数据采集 / 代理模型标定 / 自测
  train.py                训练（代理模型或 Fluent，支持加载预训练模型）
  evaluate.py             无控制 / PID / RL 对比评估与绘图
configs/
  surrogate.yaml          代理模型预训练配置
  fluent.yaml             Fluent 在线训练 / 微调配置
docs/
  algorithm.md            算法详解（MDP、代理模型方程、SAC、耦合时序、训练流程）
  fluent_setup.md         Fluent 几何、网格、算例、自检、训练全流程
tests/                    单元测试（含假 Fluent 会话测试）
```

---

## 快速开始（不需要 Fluent）

```bash
pip install -r requirements.txt
pytest -q                                   # 12 个测试，约 10 s

# 代理模型上训练 SAC（4 个并行环境，CPU 约 10~15 分钟）
python scripts/train.py --config configs/surrogate.yaml --run-name sac_surrogate

# 对比 无控制 / 自动整定的 PID / RL，输出指标表与时程图到 runs/sac_surrogate/eval/
python scripts/evaluate.py --run runs/sac_surrogate --tune-pid
```

任何参数都可以在命令行覆盖，例如 `--set train.total_timesteps=300000 --set reward.w_du=0.2 --set flow.alpha0_deg=8`。

---

## 在 Fluent 上运行（完整流程）

```bash
pip install ansys-fluent-core
# 1. 几何 → 网格（边界名 airfoil / jet_upper / jet_lower / inlet / outlet，见 docs/fluent_setup.md）
python scripts/naca_geometry.py
# 2. 自动建立算例并计算初始流场 → fluent_case/airfoil_rl.cas.h5, *_init.dat.h5, *_meta.yaml
python scripts/setup_fluent_case.py --mesh fluent_case/naca0012_jets.msh --config configs/fluent.yaml
# 3. 耦合自检（射流增/减升方向、阵风响应、单步耗时）
python scripts/check_fluent.py --config configs/fluent.yaml
# 4.（可选）系统辨识，标定代理模型
python scripts/calibrate_surrogate.py collect --config configs/fluent.yaml --out fluent_case/sysid.npz
python scripts/calibrate_surrogate.py fit --config configs/surrogate.yaml --data fluent_case/sysid.npz
# 5. 代理模型预训练 → 6. Fluent 在线微调 → 7. Fluent 上评估
python scripts/train.py --config configs/surrogate.yaml --config configs/surrogate_calibrated.yaml --run-name pre
python scripts/train.py --config configs/fluent.yaml --init-model runs/pre/best_model.zip --run-name finetune
python scripts/evaluate.py --run runs/finetune
```

计算量参考（2D、约 10 万单元、每步 20 次内迭代、每个 Fluent 4 核）：单回合 10~25 分钟；
4 个 Fluent 并行微调 150 回合约 8~16 小时。因此推荐“代理模型预训练 + Fluent 微调”的路线，详见 [docs/algorithm.md §5–6](docs/algorithm.md)。

---

## 代理模型上的示例结果

RESULTS_PLACEHOLDER

---

## 关键设计要点

1. **命名表达式作为控制总线**：边界条件在算例里一次性引用 `u_ctrl`、`gust_aoa`。运行时 Python 每个时间步只改这两个表达式的字符串，再读 `cl`/`cd` 报告，跨版本稳定、开销是毫秒级。
2. **控制周期 ≠ CFD 时间步**：每个控制周期推进 5 个 CFD 步，动作在周期内线性过渡，奖励用周期内每一步的 Cl。
3. **每回合从同一个充分发展流场开始**（`read_data`），Fluent 发散时自动终止回合并在下一回合恢复。
4. **状态设计**：执行器有延迟，所以把“在途”的指令历史放进观测；攻角探头提供前馈信息。观测都经过归一化。
5. **离策略 SAC + 多 Fluent 并行 + 预训练迁移**：这是 CFD 环境下样本效率的关键。

## 许可

MIT
