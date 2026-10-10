#!/usr/bin/env python
"""生成带尾缘射流缝的 NACA 4 位数翼型坐标，用于 DesignModeler / SpaceClaim 建模。

输出（默认 fluent_case/）：
  naca0012_jets_dm.txt   DesignModeler “3D Curve” 坐标文件（每组一条曲线，单位 m）
                         组 1 上表面 LE→上缝前缘   组 2 上射流缝 (jet_upper)   组 3 上缝后缘→TE
                         组 4 下表面 LE→下缝前缘   组 5 下射流缝 (jet_lower)   组 6 下缝后缘→TE
                         组 7 钝尾缘底边
  naca0012_jets.csv      同样的点（含 segment 列），便于其他前处理软件
  naca0012_jets.png      预览图

约定：前缘在原点，弦线沿 +x，几何攻角 0°（攻角通过来流方向 alpha0 + gust_aoa 施加）。

  python scripts/naca_geometry.py --naca 0012 --chord 0.2 --slot-center 0.96 --slot-width 0.01
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def naca4(code: str, x: np.ndarray, closed_te: bool = False) -> tuple:
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    if m == 0 or p == 0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return xu, yu, xl, yl


def cosine_grid(a: float, b: float, n: int) -> np.ndarray:
    beta = np.linspace(0, np.pi, n)
    return a + (b - a) * (1 - np.cos(beta)) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--naca", default="0012")
    ap.add_argument("--chord", type=float, default=0.2)
    ap.add_argument("--slot-center", type=float, default=0.96, help="射流缝中心 x/c")
    ap.add_argument("--slot-width", type=float, default=0.01, help="射流缝宽度（弦长比例）")
    ap.add_argument("--n-main", type=int, default=161, help="主段点数（余弦分布）")
    ap.add_argument("--closed-te", action="store_true", help="尖尾缘（默认钝尾缘，O/C 网格更易生成）")
    ap.add_argument("--out-dir", default="fluent_case")
    args = ap.parse_args()

    c = args.chord
    x0 = args.slot_center - args.slot_width / 2
    x1 = args.slot_center + args.slot_width / 2
    if not 0.5 < x0 < x1 < 1.0:
        raise ValueError("射流缝必须位于 0.5 < x/c < 1")

    segs_x = {
        "main": cosine_grid(0.0, x0, args.n_main),
        "slot": np.linspace(x0, x1, 9),
        "aft": np.linspace(x1, 1.0, 9),
    }
    segments = []
    for side in ("upper", "lower"):
        for key, label in (("main", f"{side}_surface_front"), ("slot", f"jet_{side}"), ("aft", f"{side}_surface_aft")):
            xu, yu, xl, yl = naca4(args.naca, segs_x[key], args.closed_te)
            xs, ys = (xu, yu) if side == "upper" else (xl, yl)
            segments.append((label, xs * c, ys * c))
    if not args.closed_te:
        xu, yu, xl, yl = naca4(args.naca, np.array([1.0]), False)
        segments.append(("te_base", np.array([xu[0], xl[0]]) * c, np.array([yu[0], yl[0]]) * c))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"naca{args.naca}_jets"
    with open(out / f"{stem}_dm.txt", "w", encoding="utf-8") as fh:
        fh.write(f"# NACA {args.naca}, chord={c} m, jets at x/c=[{x0:.4f},{x1:.4f}]; columns: group point x y z\n")
        for g, (label, xs, ys) in enumerate(segments, 1):
            fh.write(f"# group {g}: {label}\n")
            for i, (x, y) in enumerate(zip(xs, ys), 1):
                fh.write(f"{g} {i} {x:.8f} {y:.8f} 0.0\n")
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["segment", "x", "y"])
        for label, xs, ys in segments:
            for x, y in zip(xs, ys):
                w.writerow([label, f"{x:.8f}", f"{y:.8f}"])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw={"width_ratios": [3, 1]})
        for ax in axes:
            for label, xs, ys in segments:
                lw = 3 if label.startswith("jet") else 1
                ax.plot(xs / c, ys / c, lw=lw, label=label if ax is axes[0] else None)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
        axes[1].set_xlim(x0 - 0.03, 1.005)
        axes[1].set_ylim(-0.02, 0.02)
        axes[1].set_title("trailing edge detail")
        axes[0].legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=120)
    except ImportError:
        pass
    print(f"已生成 {out / (stem + '_dm.txt')}, {out / (stem + '.csv')}；射流缝宽 {args.slot_width * c * 1000:.1f} mm")


if __name__ == "__main__":
    main()
