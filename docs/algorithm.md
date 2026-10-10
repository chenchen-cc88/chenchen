# 算法详解：基于深度强化学习 + Fluent 的翼型升力稳定控制

本文档给出完整的问题建模、MDP 定义、代理模型方程、SAC 算法细节、Fluent 实时耦合机制与训练流程。
代码与文档一一对应，章节末尾标注了对应的源文件。

---

## 1. 问题描述

### 1.1 物理场景

| 项目 | 取值（默认，可在 `configs/*.yaml` 修改） |
|---|---|
| 翼型 | NACA 0012，弦长 $c = 0.2\,\text{m}$，钝尾缘 |
| 来流 | $U_\infty = 15\,\text{m/s}$，空气 $\rho = 1.225$，$\mu = 1.7894\times10^{-5}$ |
| 雷诺数 | $Re = \rho U c/\mu \approx 2.05\times10^5$ |
| 对流时间 | $t_c = c/U_\infty = 13.3\,\text{ms}$ |
| 平均攻角 | $\alpha_0 = 6^\circ$（静态 $C_L \approx 0.62$） |
| 扰动 | 来流方向阵风 $\alpha_g(t)$，幅值 1°~4°，折合频率 $k = \pi f c/U \in [0.05, 0.4]$ |
| 控制目标 | 使升力系数保持在无扰动值：$C_L(t) \to C_{L,\text{ref}}$（阵风载荷减缓 / 升力稳定） |

阵风会使有效攻角 $\alpha = \alpha_0 + \alpha_g(t)$ 变化，升力随之波动（约 $\Delta C_L \approx 2\pi\,\Delta\alpha$，3° 阵风即可引起 ±0.3 的升力波动），
攻角较大时还会进入失速区（本例约 11°~12°），升力-攻角关系呈强非线性与迟滞。

### 1.2 执行器：尾缘射流（流体 Gurney 襟翼）

在上、下表面靠近尾缘处（默认 $x/c = 0.955\sim0.965$，缝宽 $0.01c = 2\,\text{mm}$）各开一条射流缝，
垂直于壁面吹气：

$$
V_\text{lower} = V_{j,\max}\,\max(u, 0),\qquad V_\text{upper} = V_{j,\max}\,\max(-u, 0),\qquad u \in [-1, 1]
$$

* $u>0$：下表面尾缘吹气，相当于在尾缘下方加了一块 Gurney 襟翼，改变尾缘 Kutta 条件、增大有效弯度 → **增升**；
* $u<0$：上表面尾缘吹气 → **减升**。

射流动量系数 $C_\mu = \dfrac{2 h V_j^2}{c U_\infty^2}$，默认 $V_{j,\max} = 20\,\text{m/s}$ 时 $C_\mu \approx 0.036$。
只吹不吸，避免速度入口出现负速度带来的数值问题；$u = 0$ 时两条缝速度均为 0，等价于无滑移壁面。

> 为什么选射流而不是襟翼：射流只需修改边界条件值，不需要动网格，Fluent 求解稳定、每步开销小、
> 运行时与 Fluent 的交互只有“改一个数”，最适合 RL 这种需要上万次交互的场景。
> 襟翼 / 俯仰控制的改法见第 9 节。

### 1.3 为什么用强化学习

* 对象非线性（失速、射流效率随分离程度变化）、有执行器延迟、扰动统计特性多样；
* 无需显式模型，直接以 CFD 为环境优化闭环性能；
* 可以自然地融合多种传感器（升力、攻角探头、表面压力）并学到前馈 + 反馈的组合策略；
* 本项目同时提供经过整定的 PID（含阵风前馈）作为基准，结果可直接对比。

---

## 2. 马尔可夫决策过程（MDP）定义

> 源码：`airfoil_rl/env.py`

### 2.1 时间离散

| 量 | 默认值 | 无量纲 |
|---|---|---|
| CFD 物理时间步 $\Delta t$ | $4\times10^{-4}$ s | $0.03\,t_c$ |
| 控制周期 $\Delta t_a = N_s\Delta t$ | $N_s = 5$，2 ms | $0.15\,t_c$ |
| 回合长度 | 200 个控制步，0.4 s | $30\,t_c$ |

阵风最高频率 $k=0.4$ 对应周期约 $7.9\,t_c \approx 52$ 个控制周期，控制带宽充足；
控制周期又远大于 Fluent 单步，使得每次与 Python 交互的开销可以忽略。

### 2.2 观测 $o_k$

所有分量都归一化到 $O(1)$，便于神经网络学习：

$$
o_k = \Big[\underbrace{\tfrac{\tilde C_{L,k-i} - C_{L,\text{ref}}}{s_{C_L}}}_{i=0..n_h-1},\;
\underbrace{\tfrac{\bar C_{D,k} - C_{D,\text{ref}}}{s_{C_D}}}_{\text{可选}},\;
\underbrace{\tfrac{\tilde\alpha_{g,k-i}}{s_\alpha}}_{i=0..n_g-1,\ \text{可选}},\;
\underbrace{\tfrac{C_{p,j} - C_{p,j,\text{ref}}}{s_p}}_{\text{可选}},\;
\underbrace{\bar e_k}_{\text{误差滑动平均}},\;
\underbrace{u_{k-i}}_{i=0..n_u-1}\Big]
$$

* $\tilde C_{L}$：控制周期内平均升力 + 高斯测量噪声（$\sigma = 0.002$）；默认 $n_h = 6$ 段历史，隐含了升力变化率信息；
* $\tilde\alpha_g$：机头/上游攻角探头读数（带 0.05° 噪声），默认 3 段历史，提供**前馈**信息；
* $C_p$：Fluent 中的表面压力探头（可选；需与预训练时观测维度一致）；
* $\bar e_k = \lambda \bar e_{k-1} + (1-\lambda) e_k$：归一化升力误差的指数滑动平均（时间常数 $3\,t_c$），相当于给策略提供“积分项”，
  消除慢变阵风下的静差（没有它时，RL 在 k = 0.1 的慢正弦阵风上明显不如带积分的 PID）；
* $u_{k-i}$：最近 $n_u = 3$ 个执行器指令。动作经过平滑、执行器又有延迟，“已发出但尚未生效”的指令是系统状态的一部分，
  缺少它，策略无法区分“射流已经在起作用”与“射流还在路上”（理论上延迟系统的马尔可夫状态必须包含这部分信息）。

由于存在执行器延迟和流动记忆效应，单帧观测不满足马尔可夫性，用短历史堆叠近似（也可换成 LSTM 策略，见 9 节）。

### 2.3 动作与平滑

策略输出 $a_k \in [-1,1]$，执行器指令经一阶平滑（Rabault 等 2019 的做法），防止边界条件突变引起数值冲击和高频抖振：

$$
u_{k+1} = \operatorname{clip}\big(u_k + \beta\,(a_k - u_k),\,-1,\,1\big),\qquad \beta = 0.5
$$

在一个控制周期内，$u$ 在 $N_s$ 个 CFD 时间步上从 $u_k$ **线性过渡**到 $u_{k+1}$。可选 `max_du` 对单周期变化量限幅。

### 2.4 奖励

类 LQR 的二次型代价，取负作为奖励：

$$
r_k = -\Big[\,w_{C_L}\,\frac{1}{N_s}\sum_{n=1}^{N_s}\Big(\frac{C_{L,n} - C_{L,\text{ref}}}{s_{C_L}}\Big)^2
+ w_u\,u_{k+1}^2 + w_{\Delta u}\,(u_{k+1}-u_k)^2
+ w_{C_D}\,\frac{\max(\bar C_D - C_{D,\text{ref}},0)}{s_{C_D}}\Big]
$$

| 权重 | 默认 | 含义 |
|---|---|---|
| $w_{C_L}$ | 1.0 | 升力偏差（用周期内**每个 CFD 步**的 $C_L$，奖励能感知周期内的波动） |
| $w_u$ | 0.02 | 射流能耗（$C_\mu \propto u^2$） |
| $w_{\Delta u}$ | 0.1 | 抖振惩罚 |
| $w_{C_D}$ | 0 | 需要同时兼顾阻力时打开 |
| $s_{C_L}$ | 0.1 | 升力偏差 0.1 时代价为 1 |

奖励下限截断为 $-20$。若 $|C_L - C_{L,\text{ref}}| > 3$ 或 Fluent 发散，回合终止并给予 $-50$ 惩罚。

### 2.5 折扣因子

$\gamma = 0.95$，有效视界 $1/(1-\gamma) = 20$ 步 $= 3\,t_c$，覆盖执行器延迟（$0.3\,t_c$）与射流慢通道响应时间（$1.5\,t_c$）。
升力跟踪是“短视”的调节问题，过长的视界只会增大 Critic 的估计方差；在代理模型上实测 $\gamma=0.95$、学习率 $10^{-4}$ 的组合
比 $\gamma=0.97$、$3\times10^{-4}$ 收敛更平稳、最终误差更低。

---

## 3. 代理模型（降阶非定常气动模型）

> 源码：`airfoil_rl/surrogate.py`。用途：① 分钟级预训练与调参；② 程序调试；③ 经 Fluent 数据标定后缩小 sim-to-CFD 差距。

### 3.1 环量升力迟滞：Wagner 函数

阶跃攻角的升力响应 $C_L(s) = C_{L\alpha}\,\alpha\,\phi(s)$，$s = 2U t/c$ 为半弦长距离。R.T. Jones 两指数近似：

$$\phi(s) \approx 1 - A_1 e^{-b_1 s} - A_2 e^{-b_2 s},\quad A_1 = 0.165,\ b_1 = 0.0455,\ A_2 = 0.335,\ b_2 = 0.3$$

写成状态空间（$w = 2U/c$，$\kappa$ 为可辨识的时间尺度因子 `wagner_time_scale`）：

$$
\dot x_1 = -\frac{b_1 w}{\kappa} x_1 + \alpha,\qquad
\dot x_2 = -\frac{b_2 w}{\kappa} x_2 + \alpha,\qquad
\alpha_e = \alpha(1 - A_1 - A_2) + \frac{w}{\kappa}(A_1 b_1 x_1 + A_2 b_2 x_2)
$$

其中 $\alpha = \alpha_0 + \alpha_g(t) - \alpha_{zl}$。稳态时 $\alpha_e = \alpha$，阶跃瞬间 $\alpha_e = 0.5\alpha$，与 Wagner 函数一致。

### 3.2 分离与失速：Goman–Khrabrov 模型 + Kirchhoff 升力

分离点位置 $f\in[0,1]$（1 为完全附着）满足一阶动力学：

$$
\tau_1 t_c\,\dot f = f_{st}\big(\alpha_e - \tau_2 t_c\,\dot\alpha\big) - f,\qquad
f_{st}(\alpha) = \frac{1}{1 + e^{\,s(|\alpha| - \alpha_{ss})}}
$$

法向力与弦向力（Kirchhoff 近似）：

$$
C_N = C_{L\alpha}\,\alpha_e\Big(\frac{1+\sqrt f}{2}\Big)^2,\qquad C_C = \eta\,C_{L\alpha}\,\alpha_e^2\sqrt f
$$

$$
C_L = C_N\cos\alpha_e + C_C\sin\alpha_e + \Delta C_{L,\text{jet}} + \varepsilon(t),\qquad
C_D = C_{D0} + C_N\sin\alpha_e - C_C\cos\alpha_e + k_{d}|u|
$$

默认参数下静态极曲线：$C_L(6^\circ) = 0.62$，最大升力 $\approx 0.94$（$\alpha \approx 10.5^\circ$），16° 时降到 0.64，符合 NACA0012 在 $Re\sim2\times10^5$ 的量级。

### 3.3 射流执行器

$$
J(u) = \begin{cases} K_+\,|u|^p & u\ge0 \\ -K_-\,|u|^p & u<0\end{cases}\times\big[\eta_s + (1-\eta_s) f\big]
$$

（$f$ 越小即分离越严重，射流效率越低）。$J$ 先经过纯延迟 $t_d$（射流在腔体/缝道内建立、尾缘涡结构形成，默认 $0.3\,t_c$），
再经快、慢两条一阶通道作用于升力：

$$
\dot c_1 = \frac{J(u(t-t_d)) - c_1}{\tau_f t_c},\quad
\dot c_2 = \frac{J(u(t-t_d)) - c_2}{\tau_s t_c},\quad
\Delta C_{L,\text{jet}} = \varphi\,c_1 + (1-\varphi)\,c_2
$$

默认 $K_+ = 0.40$、$K_- = 0.35$、$\tau_f = 0.15$、$\tau_s = 1.5$、$\varphi = 0.4$，射流满量程约可抵消 ±3.5° 的阵风。

### 3.4 随机脉动

$\varepsilon(t)$ 为 Ornstein–Uhlenbeck 过程：相关时间 $0.5\,t_c$，标准差 $\sigma = 0.003 + 0.03(1-f)$（分离越严重脉动越大）。

### 3.5 数值积分

与 Fluent 相同的步长 $\Delta t$ 上做 RK4 积分，射流指令在步内保持为该步终点值（与 Fluent 隐式格式一致）。
单回合（1000 个时间步）耗时约 50 ms。

---

## 4. 强化学习算法：Soft Actor-Critic（SAC）

> 源码：`airfoil_rl/training.py`（基于 stable-baselines3；也可选 TD3 / PPO）

### 4.1 选择 SAC 的理由

* **离策略（off-policy）+ 经验回放**：每个 CFD 样本可以被反复利用，样本效率比 PPO 高数倍，CFD 环境下这是决定性因素；
* 连续动作、最大熵目标带来稳健的探索，对超参数不敏感；
* 支持多环境并行采样（多个 Fluent 实例同时运行）。

### 4.2 目标函数与更新公式

最大熵目标：

$$ J(\pi) = \sum_k \mathbb E\big[r_k + \alpha\,\mathcal H(\pi(\cdot\mid o_k))\big] $$

双 Q 网络 $Q_{\theta_1}, Q_{\theta_2}$ 与目标网络 $Q_{\bar\theta_i}$，Critic 损失：

$$
y = r + \gamma(1-d)\Big[\min_{i=1,2} Q_{\bar\theta_i}(o', a') - \alpha\log\pi_\phi(a'\mid o')\Big],\quad a'\sim\pi_\phi(\cdot\mid o')
$$

$$
L_Q(\theta_i) = \mathbb E_{(o,a,r,o',d)\sim\mathcal D}\big[(Q_{\theta_i}(o,a) - y)^2\big]
$$

策略（重参数化 $a = \tanh(\mu_\phi(o) + \sigma_\phi(o)\odot\xi),\ \xi\sim\mathcal N(0,I)$）：

$$ L_\pi(\phi) = \mathbb E_{o\sim\mathcal D,\,\xi}\big[\alpha\log\pi_\phi(a\mid o) - \min_i Q_{\theta_i}(o,a)\big] $$

温度自动调节（目标熵 $\bar{\mathcal H} = -\dim(a) = -1$）：

$$ L(\alpha) = \mathbb E_{a\sim\pi}\big[-\alpha\,(\log\pi(a\mid o) + \bar{\mathcal H})\big] $$

目标网络软更新 $\bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta$，$\tau = 0.005$。

### 4.3 与 CFD 耦合的训练伪代码

```text
初始化 Actor π_φ、Critic Q_θ1,Q_θ2、目标网络、回放池 D
启动 N 个环境（代理模型或 N 个独立 Fluent 进程）
for 每个环境:  读入充分发展流场（read_data），随机采样阵风 α_g(t)，u ← 0
for 迭代 = 1, 2, ...:
    并行地，对每个环境 e:
        a_e ← π_φ(o_e) 采样（learning_starts 之前均匀随机）
        u_new ← u + β (a_e − u)
        for n = 1..N_s:                         # 一个控制周期 = N_s 个 CFD 时间步
            写命名表达式 u_ctrl ← u + (u_new − u)·n/N_s
            写命名表达式 gust_aoa ← α_g(t + Δt)
            Fluent dual_time_iterate(1 步, 20 次内迭代)
            读 Cl、Cd（report_definitions.compute）
        计算 r_e，组装 o'_e，存 (o_e, a_e, r_e, o'_e, d_e) 进 D
        若回合结束：重新读入初始流场并采样新阵风
    重复 gradient_steps 次：从 D 采样 batch，按 4.2 更新 Q、π、α 与目标网络
    每 eval_freq 步：在固定阵风工况上评估确定性策略，保存最优模型
```

### 4.4 默认超参数

| 参数 | 代理模型预训练 | Fluent 微调 |
|---|---|---|
| 网络 | MLP 256×256（Actor/Critic 各自独立） | 同左 |
| 学习率 | 1e-4 | 5e-5 |
| batch / 回放池 | 256 / 3e5 | 256 / 1e5 |
| $\gamma$ / $\tau$ | 0.95 / 0.005 | 0.95 / 0.005 |
| 并行环境数 | 4 | 4（4 个 Fluent，每个 4 核） |
| gradient_steps（每次采样后的更新次数） | 1 | 4（提高样本利用率） |
| learning_starts | 2000 | 800 |
| 总步数 | 2e5（1000 回合） | 3e4（150 回合） |

---

## 5. Fluent 实时耦合机制

> 源码：`airfoil_rl/fluent_backend.py`，算例准备：`scripts/setup_fluent_case.py`、`docs/fluent_setup.md`

### 5.1 核心思想：命名表达式作为“控制总线”

边界条件在算例中**一次性**写成引用命名表达式的形式：

| 命名表达式 | 定义 | 被谁引用 |
|---|---|---|
| `U_inf` | `15 [m/s]` | 入口 |
| `alpha0` | `6 [deg]` | `aoa_in` |
| `gust_aoa` | **Python 每个时间步写入**，如 `1.2345 [deg]` | `aoa_in` |
| `aoa_in` | `alpha0 + gust_aoa` | 入口速度分量 |
| `u_ctrl` | **Python 每个时间步写入**，如 `0.3127` | 射流 |
| `Vjet_max` | `20 [m/s]` | 射流 |
| `jet_lower_vel` | `Vjet_max*max(u_ctrl, 0)` | `jet_lower` 速度大小（法向） |
| `jet_upper_vel` | `Vjet_max*max(-u_ctrl, 0)` | `jet_upper` 速度大小（法向） |

入口速度分量：$(U_\infty\cos\text{aoa\_in},\ U_\infty\sin\text{aoa\_in})$。

运行时 Python 只做两件事：改 `u_ctrl` / `gust_aoa` 两个表达式的定义字符串、读报告定义的值。
**不需要 UDF、不需要动网格、不需要修改任何边界条件对象**，接口在 Fluent 2024R2~2026R1 中一致（已对照 PyFluent 自带 API 定义核对），稳健且开销小（每次 gRPC 调用约几毫秒）。

### 5.2 单个控制周期的时序

```text
Python (env.step)                                     Fluent (gRPC)
───────────────────────────────────────────────────────────────────────────────
a_k = π(o_k);  u_{k+1} = u_k + β(a_k − u_k)
for n in 1..N_s:
    named_expressions["u_ctrl"].definition = "..."   ─▶  边界条件更新（射流速度）
    named_expressions["gust_aoa"].definition = "..." ─▶  边界条件更新（来流方向）
    dual_time_iterate(time_step_count=1,             ─▶  求解 t_{n+1} 时刻流场
                      max_iter_per_step=20)               （2 阶隐式，20 次内迭代）
    report_definitions.compute(["cl","cd",...])      ◀─  返回 [{'cl':[0.63,0]}, ...]
r_k, o_{k+1}  ← 由 N_s 个 Cl 计算
```

隐式格式求解的是 $t_{n+1}$ 时刻流场，所以写入的是 $t_{n+1}$ 时刻的阵风与控制值。

### 5.3 回合重置

每回合开始时 `file.read_data(初始流场)` 读回同一个充分发展的无控制流场（由 setup 脚本预先计算），
保证每个回合的初始状态一致；若 Fluent 发散（Cl 非有限或 $|C_L| > 5$），环境捕获异常、终止回合，下一回合重新读 data 即可恢复。

### 5.4 并行与计算量估算

* 每个并行环境是一个独立的 Fluent 进程，各自的工作目录 `runs/fluent_work/env{rank}`；SB3 的 `SubprocVecEnv` 并行采样；
* 典型 2D 网格 8~15 万单元、每步 20 次内迭代，4 核约 0.5~1.5 s/时间步 → 每回合（1000 步）10~25 分钟；
* Fluent 微调 150 回合、4 个并行实例：约 8~16 小时；从零开始在 Fluent 上训练通常需要 400~800 回合（数天），因此强烈建议先在代理模型上预训练；
* 许可证：每个 Fluent 进程占用 1 个求解器许可，超过 4 核需要 HPC 许可。

---

## 6. 训练流程（推荐的三阶段迁移）

```text
┌──────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│ 阶段 1 代理模型预训练 │──▶│ 阶段 2 系统辨识（可选）    │──▶│ 阶段 3 Fluent 在线微调     │
│ 分钟级，验证 MDP 设计 │   │ Fluent 开环实验 → 标定代理 │   │ 加载预训练策略继续 SAC     │
│ train.py surrogate    │   │ calibrate_surrogate.py    │   │ train.py --init-model     │
└──────────────────────┘   │ → 用标定后的代理重新预训练 │   └──────────────────────────┘
                           └──────────────────────────┘                │
                                                                       ▼
                                                    evaluate.py：Fluent 上对比 无控制 / PID / RL
```

1. **阶段 1**：`python scripts/train.py --config configs/surrogate.yaml`，检查奖励设计、观测、超参数是否合理；
2. **阶段 2**：`calibrate_surrogate.py collect` 在 Fluent 上运行 8 组开环实验（射流阶跃 ±1、+0.5，阵风阶跃 ±3°/+6°，正弦阵风，PRBS 射流 + Dryden 阵风；共约 5000 个时间步），
   `fit` 用带边界的非线性最小二乘（`scipy.optimize.least_squares`）拟合代理模型参数，再用标定后的代理模型重新预训练；
3. **阶段 3**：`train.py --config configs/fluent.yaml --init-model runs/xxx/best_model.zip`，回放池清空后在 Fluent 上继续训练；
   观测、动作、时间步必须与预训练一致（配置中 flow/control/obs 三节保持相同）。

也可以跳过阶段 2，直接把阶段 1 的策略放到 Fluent 上评估（`evaluate.py --run runs/xxx --config configs/fluent.yaml`）看迁移效果。

---

## 7. 评估指标

在 5 个固定阵风工况上（正弦 k=0.1 / k=0.3、两次 1-cos 离散阵风、Dryden 连续湍流、多频正弦），比较：

* **Cl RMSE** $=\sqrt{\overline{(C_L - C_{L,\text{ref}})^2}}$（主要指标）与相对无控制的降低百分比；
* **最大偏差** $\max|C_L - C_{L,\text{ref}}|$（结构载荷峰值）；
* **u_rms**、**du_rms**（射流能耗与抖振）；
* 平均 $C_D$；累计回报。

基准控制器：PID + 阵风前馈 $u = -(K_p e + K_i\!\int\! e + K_d\dot e) - K_{ff}\,\tilde\alpha_g$（积分抗饱和），
`evaluate.py --tune-pid` 在**随机训练阵风**（非评估工况）上用 Nelder–Mead 整定，与 RL 使用相同的带噪测量值和相同的奖励目标，对比公平。

---

## 8. 调参经验与常见问题

| 现象 | 原因与处理 |
|---|---|
| 策略输出在 ±1 之间高频切换 | 增大 `w_du` 或减小 `action_smoothing`；检查测量噪声是否过大 |
| 升力被压住但均值偏离 Cl_ref | 增加观测历史 `n_hist`，或在观测中加入误差积分项；检查 Cl_ref 是否来自同一网格/时间步 |
| 阵风大时控制失效 | 执行器能力不足（看 u 是否长时间饱和）：增大 `Vjet_max` 或缩小阵风幅值范围 |
| 代理模型上好、Fluent 上差 | 做阶段 2 系统辨识；在 Fluent 上微调；检查射流方向（u>0 应增升，用 `check_fluent.py` 验证） |
| Fluent 单步太慢 | 减少 `max_iter_per_step`（残差在 10~15 次内已收敛即可）、粗化远场网格、增加核数 |
| Fluent 偶尔发散 | 减小 `dt_cfd` 或射流最大速度；检查射流缝网格（缝宽方向至少 10 个单元） |
| 评估曲线震荡 | SAC 在小回放池上可能过拟合：降低学习率、增大 batch，或用 `best_model.zip` |

---

## 9. 扩展方向

* **后缘襟翼**：Fluent 动网格（刚体运动 + 光顺/重构，或重叠网格），襟翼角速度用表达式/UDF 读 `u_ctrl`；运行时接口不变；
* **俯仰控制**：整个计算域刚体旋转（Cell Zone → Mesh Motion，转速 = 表达式 `omega_ctrl`），不需要额外边界；
* **对流阵风**：用 Fluent 的场速度法 / 上游阵风发生器使阵风以 $U_\infty$ 向下游传播，此时上游攻角探头具有真正的“预见”作用；
* **多执行器 / 多传感器**：动作维度 >1（多条射流缝）、观测加入表面压力分布（`--probes 6`）；
* **部分可观测**：把 MLP 换成 LSTM（`sb3-contrib` 的 `RecurrentPPO`）或增加历史长度；
* **鲁棒性**：训练时随机化 $U_\infty$、$\alpha_0$、执行器增益（domain randomization）以提升迁移能力；
* **三维机翼**：同一框架，每个控制周期的 CFD 开销增加，建议更强地依赖代理模型与离策略算法。

---

## 10. 参考文献

1. J. Rabault, M. Kuchta, A. Jensen, U. Réglade, N. Cerardi. Artificial neural networks trained through deep reinforcement learning discover control strategies for active flow control. *J. Fluid Mech.* 865, 281–302, 2019.
2. J. Rabault, A. Kuhnle. Accelerating deep reinforcement learning strategies of flow control through a multi-environment approach. *Phys. Fluids* 31, 094105, 2019.
3. J. Viquerat, P. Meliga, A. Larcher, E. Hachem. A review on deep reinforcement learning for fluid mechanics: an update. *Phys. Fluids* 34, 111301, 2022.
4. T. Haarnoja, A. Zhou, P. Abbeel, S. Levine. Soft Actor-Critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*, 2018.
5. S. Fujimoto, H. van Hoof, D. Meger. Addressing function approximation error in actor-critic methods (TD3). *ICML*, 2018.
6. J. G. Leishman. *Principles of Helicopter Aerodynamics*, 2nd ed., Cambridge University Press, 2006（Wagner/Küssner 函数、Beddoes–Leishman 模型）。
7. M. Goman, A. Khrabrov. State-space representation of aerodynamic characteristics of an aircraft at high angles of attack. *J. Aircraft* 31(5), 1994.
8. R. H. Liebeck. Design of subsonic airfoils for high lift. *J. Aircraft* 15(9), 1978（Gurney 襟翼）。
9. Ansys Fluent User's Guide — Expressions；PyFluent 文档 <https://fluent.docs.pyansys.com>。
