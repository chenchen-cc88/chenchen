# fluent_case/

Fluent 相关的输入与生成文件存放在这里（大文件已在 .gitignore 中忽略）：

| 文件 | 来源 |
|---|---|
| `naca0012_jets_dm.txt` / `.csv` / `.png` | `scripts/naca_geometry.py` |
| `naca0012_jets.msh` | 自己生成的网格（边界名见 docs/fluent_setup.md §2） |
| `airfoil_rl.cas.h5` | `scripts/setup_fluent_case.py` |
| `airfoil_rl_init.dat.h5` | 同上：每个 RL 回合 reset 时读入的充分发展流场 |
| `airfoil_rl_meta.yaml` | 同上：`cl_ref`、`cd_ref`、单步耗时等 |
| `sysid.npz` | `scripts/calibrate_surrogate.py collect` |
