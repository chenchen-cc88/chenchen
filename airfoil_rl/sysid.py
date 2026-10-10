"""系统辨识：用 Fluent 开环数据标定代理模型参数（sim-to-CFD 迁移的关键一步）。

流程：
1. ``default_experiments`` 生成一组开环激励（射流阶跃、阵风阶跃、正弦阵风、PRBS 射流 + 随机阵风）；
2. ``collect`` 在任意后端（通常是 Fluent）上逐个运行，记录每个 CFD 时间步的 Cl/Cd；
3. ``fit_surrogate`` 用同样的输入回放代理模型，最小化 Cl/Cd 误差（scipy least_squares，带边界）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .backend_base import Backend
from .config import Config
from .gust import GustGenerator, GustSignal
from .surrogate import SurrogateBackend, SurrogateParams

DEFAULT_BOUNDS = {
    "cl_alpha": (3.0, 8.0),
    "alpha_zl_deg": (-3.0, 3.0),
    "alpha_stall_deg": (6.0, 20.0),
    "stall_sharpness": (5.0, 120.0),
    "wagner_time_scale": (0.3, 3.0),
    "tau_sep": (0.3, 10.0),
    "tau_sep_delay": (0.0, 5.0),
    "jet_gain_pos": (0.0, 1.5),
    "jet_gain_neg": (0.0, 1.5),
    "jet_exponent": (0.3, 2.0),
    "jet_fast_frac": (0.0, 1.0),
    "jet_tau_fast": (0.02, 1.0),
    "jet_tau_slow": (0.2, 6.0),
    "cd0": (0.002, 0.05),
    "jet_cd_gain": (0.0, 0.05),
}
DEFAULT_FIT_KEYS = [
    "cl_alpha",
    "alpha_zl_deg",
    "alpha_stall_deg",
    "wagner_time_scale",
    "tau_sep",
    "jet_gain_pos",
    "jet_gain_neg",
    "jet_fast_frac",
    "jet_tau_fast",
    "jet_tau_slow",
    "cd0",
]


@dataclass
class Experiment:
    name: str
    gust: GustSignal
    u: np.ndarray  # 每个控制周期末的执行器指令（周期内线性过渡）


def default_experiments(cfg: Config, n_steps: int = 120, seed: int = 2024) -> list:
    c = cfg.control
    tc = cfg.flow.convective_time
    gen = GustGenerator(cfg.gust, cfg.flow, (n_steps + 2) * c.dt_ctrl, c.dt_cfd)
    rng = np.random.default_rng(seed)
    k0 = int(round(2.0 * tc / c.dt_ctrl))  # 2 个对流时间后施加阶跃

    def step_u(level: float) -> np.ndarray:
        u = np.zeros(n_steps)
        u[k0:] = level
        return u

    prbs = np.zeros(n_steps)
    k = k0
    while k < n_steps:
        dur = int(rng.integers(3, 15))
        prbs[k : k + dur] = rng.uniform(-1, 1)
        k += dur

    return [
        Experiment("jet_step_+1", gen.zero(), step_u(1.0)),
        Experiment("jet_step_-1", gen.zero(), step_u(-1.0)),
        Experiment("jet_step_+0.5", gen.zero(), step_u(0.5)),
        Experiment("gust_step_+3deg", gen.step(3.0, 2.0), np.zeros(n_steps)),
        Experiment("gust_step_+6deg", gen.step(6.0, 2.0), np.zeros(n_steps)),  # 进入失速区
        Experiment("gust_step_-3deg", gen.step(-3.0, 2.0), np.zeros(n_steps)),
        Experiment("gust_sine_k0.2", gen.sine(3.0, 0.2), np.zeros(n_steps)),
        Experiment("prbs_jet+dryden", gen.dryden(2.0, 5.0, rng), prbs),
    ]


def run_openloop(backend: Backend, exp: Experiment) -> dict:
    backend.reset(exp.gust)
    cl, cd, uf = [], [], []
    u_prev = 0.0
    n = backend.n_substeps
    for u in exp.u:
        m = backend.step(u_prev, float(u))
        cl.extend(m.cl_series)
        cd.extend(m.cd_series)
        uf.extend(u_prev + (u - u_prev) * np.arange(1, n + 1) / n)
        u_prev = float(u)
    t = backend.dt_cfd * np.arange(1, len(cl) + 1)
    return {"t": t, "cl": np.asarray(cl), "cd": np.asarray(cd), "u_fine": np.asarray(uf)}


def collect(backend: Backend, experiments: list, verbose: bool = True) -> dict:
    data = {}
    for exp in experiments:
        if verbose:
            print(f"[sysid] 运行开环实验 {exp.name} ({len(exp.u)} 控制步)")
        rec = run_openloop(backend, exp)
        rec.update({"u": exp.u, "gust_t": exp.gust.t, "gust_alpha": exp.gust.alpha})
        data[exp.name] = rec
    return data


def save_dataset(path: str, data: dict, meta: Optional[dict] = None) -> None:
    flat = {f"{name}::{key}": np.asarray(val) for name, rec in data.items() for key, val in rec.items()}
    flat["__meta__"] = np.array(json.dumps({"names": list(data.keys()), **(meta or {})}))
    np.savez_compressed(path, **flat)


def load_dataset(path: str) -> tuple:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    data: dict = {n: {} for n in meta["names"]}
    for key in z.files:
        if key == "__meta__":
            continue
        name, field = key.split("::", 1)
        data[name][field] = z[key]
    return data, meta


def data_to_experiments(data: dict) -> list:
    return [Experiment(n, GustSignal(r["gust_t"], r["gust_alpha"], "replay"), r["u"]) for n, r in data.items()]


def fit_surrogate(
    cfg: Config,
    data: dict,
    fit_keys: Optional[list] = None,
    init: Optional[SurrogateParams] = None,
    cd_weight: float = 5.0,
    max_nfev: int = 200,
    verbose: int = 1,
) -> tuple:
    """返回 (标定后的参数, 结果摘要 dict)。"""
    from scipy.optimize import least_squares

    keys = list(fit_keys or DEFAULT_FIT_KEYS)
    base = init or SurrogateParams.from_dict(cfg.surrogate)
    lo = np.array([DEFAULT_BOUNDS[k][0] for k in keys])
    hi = np.array([DEFAULT_BOUNDS[k][1] for k in keys])
    x0 = np.clip([getattr(base, k) for k in keys], lo + 1e-9, hi - 1e-9)
    exps = data_to_experiments(data)

    def params_of(x: np.ndarray) -> SurrogateParams:
        p = SurrogateParams.from_dict(base.to_dict())
        for k, v in zip(keys, x):
            setattr(p, k, float(v))
        return p

    def residual(x: np.ndarray) -> np.ndarray:
        be = SurrogateBackend(cfg, stochastic=False, params=params_of(x))
        res = []
        for exp in exps:
            sim = run_openloop(be, exp)
            ref = data[exp.name]
            res.append(sim["cl"] - ref["cl"])
            res.append(cd_weight * (sim["cd"] - ref["cd"]))
        return np.concatenate(res)

    r0 = residual(x0)
    sol = least_squares(residual, x0, bounds=(lo, hi), x_scale=(hi - lo), diff_step=1e-3, max_nfev=max_nfev, verbose=verbose)
    best = params_of(sol.x)
    n_cl = sum(len(data[e.name]["cl"]) for e in exps)
    summary = {
        "keys": keys,
        "x0": dict(zip(keys, map(float, x0))),
        "x": dict(zip(keys, map(float, sol.x))),
        "rms_before": float(np.sqrt(np.mean(r0**2))),
        "rms_after": float(np.sqrt(np.mean(sol.fun**2))),
        "n_samples": int(n_cl),
        "status": sol.message,
    }
    return best, summary
