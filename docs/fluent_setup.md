# Fluent 算例准备与耦合运行指南

本指南从几何、网格到 RL 在线训练，逐步说明如何在 Ansys Fluent 上运行本项目。
命名约定（边界名、表达式名、报告名）与代码默认值一致，照做即可无缝衔接。

---

## 0. 软件环境

| 软件 | 版本要求 | 说明 |
|---|---|---|
| Ansys Fluent | 2024R2 及以上（推荐 2025R1+） | 2D 双精度求解器；需要命名表达式（Expressions）功能 |
| Ansys Meshing / Fluent Meshing / ICEM | 任意 | 生成 2D 网格并定义边界名 |
| Python | 3.10~3.13 | |
| PyFluent | `ansys-fluent-core` ≥ 0.26 | `pip install ansys-fluent-core` |
| RL 依赖 | 见 `requirements.txt` | stable-baselines3、gymnasium、torch |

PyFluent 通过环境变量 `AWP_ROOT251`（版本号对应）找到 Fluent 安装目录，例如：

```bash
# Linux
export AWP_ROOT251=/ansys_inc/v251
# Windows（PowerShell）
$env:AWP_ROOT251 = "C:\Program Files\ANSYS Inc\v251"
```

验证：`python -c "import ansys.fluent.core as pf; s=pf.launch_fluent(dimension=2, ui_mode='no_gui'); print(s.get_fluent_version()); s.exit()"`

---

## 1. 几何

```bash
python scripts/naca_geometry.py --naca 0012 --chord 0.2 --slot-center 0.96 --slot-width 0.01
```

生成 `fluent_case/naca0012_jets_dm.txt`（DesignModeler 3D 曲线格式）。坐标约定：**前缘在原点、弦线沿 +x、几何攻角 0°**
（攻角由来流方向 `alpha0 + gust_aoa` 施加，阵风时不需要转动网格）。

文件中 7 组曲线：

| 组 | 内容 | 网格边界名 |
|---|---|---|
| 1 | 上表面 前缘 → 上射流缝 | `airfoil` |
| 2 | 上射流缝（2 mm） | `jet_upper` |
| 3 | 上射流缝 → 尾缘 | `airfoil` |
| 4 | 下表面 前缘 → 下射流缝 | `airfoil` |
| 5 | 下射流缝（2 mm） | `jet_lower` |
| 6 | 下射流缝 → 尾缘 | `airfoil` |
| 7 | 钝尾缘底边 | `airfoil` |

**DesignModeler 操作**：Concept → 3D Curve → 选坐标文件 → Generate；在 XY 平面画 C 型远场：
以尾缘为圆心、半径 $15c$（3 m）的半圆（上游）+ 向下游延伸 $20c$（4 m）的矩形；
Concept → Surfaces From Edges（远场外边界）→ Boolean Subtract 掉翼型围成的面（或用 Sketch 直接构造带孔的面）。
在 Workbench 中设置 Analysis Type = 2D。

> 用 SpaceClaim 时：Design → Insert → File 导入 CSV（`naca0012_jets.csv`），或用 Script 读点生成样条。

---

## 2. 网格（以 Ansys Meshing 为例）

### 2.1 命名选择（Named Selections）——必须与下表一致

| 名称 | 选择对象 | Fluent 边界类型 |
|---|---|---|
| `airfoil` | 翼型除两条射流缝外的所有边 | wall |
| `jet_upper` | 组 2 边 | velocity-inlet |
| `jet_lower` | 组 5 边 | velocity-inlet |
| `inlet` | C 型半圆 + 上、下水平边 | velocity-inlet |
| `outlet` | 下游竖直边 | pressure-outlet |
| `fluid` | 面（流体域） | fluid |

（名称含 inlet/outlet 时 Fluent 会自动识别类型；`jet_*` 默认被识别为 wall，setup 脚本会用 `set_zone_type` 改为 velocity-inlet。）

### 2.2 尺寸与边界层

第一层网格高度按 $y^+ \approx 1$ 估算（平板湍流 $C_f = 0.026/Re^{1/7}$）：

$$
C_f \approx 0.0045,\quad u_\tau = U\sqrt{C_f/2} \approx 0.71\,\text{m/s},\quad
\Delta y_1 = \frac{y^+\mu}{\rho u_\tau} \approx 2\times10^{-5}\,\text{m}\ (=10^{-4}c)
$$

| 设置 | 推荐值 |
|---|---|
| 翼型表面 Edge Sizing | 上下表面各 200~250 个单元，前缘、尾缘加密（Bias） |
| 射流缝 Edge Sizing | 每条缝 ≥ 12 个单元（单元约 0.15 mm） |
| 尾缘底边 | ≥ 8 个单元 |
| Inflation（翼型 + 射流缝） | First Layer Thickness = 2e-5 m，Growth Rate = 1.12，Layers = 35 |
| 尾迹加密（Body of Influence 或 Sizing） | 尾缘下游 $2c$ 范围单元尺寸 ≈ 0.01c |
| 远场单元 | 0.5c~1c |
| 方法 | Quadrilateral Dominant / MultiZone Quad，总单元数 8~15 万 |
| 质量 | Orthogonal Quality 最小 > 0.15，Skewness 最大 < 0.85 |

导出：File → Export → Mesh → **Fluent Input File (.msh)**，保存为 `fluent_case/naca0012_jets.msh`。

> 也可以用 Fluent Meshing 的 2D Meshing 工作流或 ICEM CFD 结构化 C 网格，只要边界名一致即可。

---

## 3. 自动建立算例（推荐）

```bash
python scripts/setup_fluent_case.py --mesh fluent_case/naca0012_jets.msh --config configs/fluent.yaml \
    --processors 8 --jet-vmax 20 --init-conv 25
# 需要表面压力探头时加 --probes 6（上下表面各 3 个 Cp 点）
# 低雷诺数层流分离泡明显时可用 --viscous transition-sst
```

脚本依次完成：

1. 读网格，`jet_upper/jet_lower/inlet` 设为 velocity-inlet、`outlet` 设为 pressure-outlet；
2. SST k-ω、空气物性、Coupled 压力-速度耦合；
3. 创建命名表达式：

   | 名称 | 定义 |
   |---|---|
   | `U_inf` | `15 [m/s]` |
   | `alpha0` | `6 [deg]` |
   | `Vjet_max` | `20 [m/s]` |
   | `u_ctrl` | `0` |
   | `gust_aoa` | `0 [deg]` |
   | `aoa_in` | `alpha0 + gust_aoa` |
   | `jet_lower_vel` | `Vjet_max*max(u_ctrl, 0)` |
   | `jet_upper_vel` | `Vjet_max*max(-u_ctrl, 0)` |

4. 边界条件：`inlet` 速度分量 X = `U_inf*cos(aoa_in)`、Y = `U_inf*sin(aoa_in)`，湍流度 0.1%、黏度比 1；
   `jet_lower` / `jet_upper`：Magnitude, Normal to Boundary，速度 = `jet_lower_vel` / `jet_upper_vel`；
5. 参考值：Area = 0.2（弦长 × 1 m 展向），Length = 0.2，Velocity = 15，Density = 1.225；
6. 报告定义：`cl`（Lift，力方向 $(-\sin\alpha_0, \cos\alpha_0)$，翼型 + 两条射流缝），`cd`（Drag，方向 $(\cos\alpha_0, \sin\alpha_0)$）；
7. 稳态 800 步 → 切换 2 阶隐式瞬态，$\Delta t = 4\times10^{-4}$ s → 推进 25 个对流时间（约 830 步），最后 5 个对流时间统计 $C_{L,\text{ref}}$；
8. 保存：
   * `fluent_case/airfoil_rl.cas.h5`
   * `fluent_case/airfoil_rl_init.dat.h5`（每回合 reset 读入的初始流场）
   * `fluent_case/airfoil_rl_meta.yaml`（`cl_ref`、`cd_ref`、单步耗时等）

如有某项打印 `[!!] ... 失败`，按下一节在 GUI 中补齐该项后 File → Write → Case & Data 覆盖保存即可。

---

## 4. 手动建立算例（GUI，备用）

1. **General**：Solver Type = Pressure-Based，Time = Steady（先做稳态初值）。
2. **Models → Viscous**：k-omega，SST。
3. **Materials → air**：Density 1.225，Viscosity 1.7894e-05。
4. **Parameters & Customization → Expressions → New**：按第 3 节表格逐个创建 8 个命名表达式（注意单位方括号）。
5. **Boundary Conditions**：
   * `inlet`（velocity-inlet）→ Velocity Specification Method = Components；X-Velocity 输入框右侧切换为表达式，填 `U_inf*cos(aoa_in)`，Y-Velocity 填 `U_inf*sin(aoa_in)`；Turbulence：Intensity 0.1 %、Viscosity Ratio 1；
   * `jet_lower`（类型改为 velocity-inlet）→ Magnitude, Normal to Boundary，Velocity Magnitude = `jet_lower_vel`；Intensity 1 %、Viscosity Ratio 1；
   * `jet_upper` → 同上，`jet_upper_vel`；
   * `outlet`（pressure-outlet）→ Gauge Pressure 0；
   * `airfoil` → 无滑移壁面（默认）。
6. **Reference Values**：Area 0.2、Length 0.2、Velocity 15、Density 1.225。
7. **Methods**：Scheme = Coupled；Spatial Discretization 全部 Second Order；Transient Formulation = Second Order Implicit（切到瞬态后）。
8. **Report Definitions → New → Force Report → Lift**：名称 `cl`，Report Output Type = Lift Coefficient，Force Vector = (-0.104528, 0.994522)（$\alpha_0 = 6°$），Zones 选 `airfoil, jet_upper, jet_lower`；
   **Drag**：名称 `cd`，Force Vector = (0.994522, 0.104528)。
9. **Initialization**：Hybrid → Initialize；**Run Calculation**：800 iterations。
10. **General → Time = Transient**；Run Calculation：Time Step Size = 0.0004，Max Iterations/Time Step = 20，推进约 830 步。
11. 记录最后约 170 步 `cl`、`cd` 的平均值。
12. File → Write → Case：`fluent_case/airfoil_rl.cas.h5`；File → Write → Data：`fluent_case/airfoil_rl_init.dat.h5`。
13. 新建 `fluent_case/airfoil_rl_meta.yaml`：

    ```yaml
    cl_ref: 0.6xx   # 第 11 步的平均 Cl
    cd_ref: 0.01xx
    dt_cfd: 0.0004
    ```

---

## 5. 耦合自检（训练前必做）

```bash
python scripts/check_fluent.py --config configs/fluent.yaml
```

期望输出（数值因网格而异）：

```text
[1] 启动并读入 case: 35.2 s
[2] reset 后 Cl=0.6512 Cd=0.01503；cl_ref=0.6508
[3] 无控制推进 8 控制步：Cl 漂移 +0.00021；平均 0.74 s/时间步
[4] 下射流 u=+1：Cl 变化 +0.1x（应为正）
[5] 上射流 u=-1：Cl 变化 -0.1x（应为负）
[6] 阵风 +3°：Cl 变化 +0.1x（应为正，约 +0.1~0.3）
每回合 1000 个时间步，预计 12.3 分钟/回合/实例
自检通过 ✔
```

若 [4]/[5] 的符号相反，说明射流缝命名与上下表面对调了；若 Cl 不变化，说明边界条件没有引用表达式。

**建议的验证**（论文/报告中需要）：

* 网格无关性：粗/中/细三套网格比较静态 $C_L$ 与射流阶跃响应；
* 时间步无关性：$\Delta t$ 与 $\Delta t/2$ 下射流阶跃、阵风阶跃的 $C_L$ 响应差异 < 2%；
* 若改变 `dt_cfd`，须同步修改 `control.n_substeps` 保持控制周期不变，并重新生成初始流场。

---

## 6. 训练与评估

```bash
# (a) 可选：系统辨识，用 Fluent 开环数据标定代理模型（约 5000 个时间步）
python scripts/calibrate_surrogate.py collect --config configs/fluent.yaml --out fluent_case/sysid.npz
python scripts/calibrate_surrogate.py fit --config configs/surrogate.yaml --data fluent_case/sysid.npz \
       --out configs/surrogate_calibrated.yaml

# (b) 代理模型预训练（若做了 (a)，叠加标定参数）
python scripts/train.py --config configs/surrogate.yaml --config configs/surrogate_calibrated.yaml --run-name pre

# (c) 预训练策略直接在 Fluent 上评估（零样本迁移）
python scripts/evaluate.py --run runs/pre --config configs/fluent.yaml --out runs/pre/eval_fluent

# (d) Fluent 在线微调（4 个 Fluent 并行）
python scripts/train.py --config configs/fluent.yaml --init-model runs/pre/best_model.zip --run-name finetune

# (e) 最终评估（无控制 / PID / RL）
python scripts/evaluate.py --run runs/finetune --pid '{"kp":0.49,"ki":47,"kd":0.0015,"kff":0.16}'
```

说明：

* `configs/fluent.yaml` 中 `train.n_envs` 个 Fluent 进程同时运行，每个 `fluent.processors` 核；总核数 = 两者乘积（评估回调还会多开 1 个实例，可用 `train.eval_freq: 0` 关闭）；
* 训练中断后可从 `runs/<name>/checkpoints/` 中的模型与回放池继续：`--init-model runs/finetune/checkpoints/model_XXXX_steps.zip`；
* Fluent 上整定 PID 很昂贵，可在代理模型上整定后把增益通过 `--pid` 传入。

### 集群 / 手动启动 Fluent

在计算节点上预先启动 Fluent（每个环境一个）：

```bash
for r in 0 1 2 3; do
  fluent 2ddp -t8 -gu -sifile=$PWD/fluent_servers/server_$r.txt &
done
```

配置中设置 `fluent.server_info_file: fluent_servers/server_{rank}.txt`，代码会按环境编号连接对应实例。

---

## 7. 常见问题

| 问题 | 处理 |
|---|---|
| `KeyError: 算例缺少命名表达式 u_ctrl` | 确认第 3/4 节的表达式名称；或修改 `fluent.expr_control/expr_gust` |
| `报告定义 ['cl'] 没有返回值` | 报告名必须为 `cl`、`cd`（或修改 `fluent.report_cl/report_cd`） |
| 读 data 时提示 case/data 不匹配 | case 与 init data 必须由同一次 setup 保存；改网格后需重新运行 setup |
| `launch_fluent` 找不到 Fluent | 设置 `AWP_ROOTxxx` 环境变量，或在配置中给出 `fluent.product_version` |
| 求解发散 | 减小 `dt_cfd`、`Vjet_max`；检查射流缝附近网格质量；把 `max_iter_per_step` 增加到 30 |
| 速度太慢 | `max_iter_per_step` 降到 10~15（观察每步残差是否下降 3 个量级）；远场粗化；用更多并行实例而不是更多核 |
| Windows 上 `SubprocVecEnv` 报错 | 脚本已使用 `spawn` 启动方式；确保从命令行运行 `python scripts/train.py`，不要在交互式环境中运行 |
