"""核心功能单元测试（只依赖代理模型，无需 Fluent）。运行: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airfoil_rl.config import Config, load_config  # noqa: E402
from airfoil_rl.controllers import NoControl, PIDController, run_episode  # noqa: E402
from airfoil_rl.env import AirfoilLiftEnv  # noqa: E402
from airfoil_rl.fluent_backend import _parse_compute_result  # noqa: E402
from airfoil_rl.gust import GustGenerator  # noqa: E402
from airfoil_rl.surrogate import SurrogateBackend, SurrogateParams  # noqa: E402
from airfoil_rl.sysid import collect, default_experiments, fit_surrogate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def small_cfg() -> Config:
    cfg = Config()
    cfg.control.episode_steps = 60
    return cfg


def test_configs_load() -> None:
    for name in ("surrogate.yaml", "fluent.yaml"):
        cfg = load_config([str(ROOT / "configs" / name)])
        assert cfg.control.dt_ctrl == pytest.approx(2e-3)
    cfg = load_config([], ["train.n_envs=8", "flow.alpha0_deg=4.5"])
    assert cfg.train.n_envs == 8 and cfg.flow.alpha0_deg == 4.5
    with pytest.raises(KeyError):
        load_config([], ["flow.not_a_field=1"])


def test_surrogate_static_polar_and_stall() -> None:
    cls = []
    for a in (0, 4, 8, 11, 16):
        cfg = Config()
        cfg.flow.alpha0_deg = a
        cls.append(SurrogateBackend(cfg, stochastic=False).cl_ref)
    assert cls[0] == pytest.approx(0.0, abs=1e-9)
    assert cls[1] < cls[2] < cls[3]  # 线性段单调
    assert cls[4] < cls[3]  # 失速后升力下降


def test_jet_authority_signs() -> None:
    cfg = Config()
    be = SurrogateBackend(cfg, stochastic=False)
    gen = GustGenerator(cfg.gust, cfg.flow, 1.0, cfg.control.dt_cfd)
    for u, sign in ((1.0, 1), (-1.0, -1)):
        be.reset(gen.zero())
        for _ in range(60):
            m = be.step(u, u)
        assert sign * (m.cl - be.cl_ref) > 0.2


def test_env_api_and_determinism() -> None:
    from gymnasium.utils.env_checker import check_env

    cfg = small_cfg()
    env = AirfoilLiftEnv(cfg)
    check_env(env, skip_render_check=True)
    g = env.gust_gen.sine(3.0, 0.2)
    r1 = run_episode(env, NoControl(), g, seed=3)
    r2 = run_episode(env, NoControl(), g, seed=3)
    np.testing.assert_allclose(r1["cl"], r2["cl"])
    assert r1["metrics"]["steps"] == cfg.control.episode_steps


def test_pid_beats_no_control() -> None:
    cfg = small_cfg()
    cfg.control.episode_steps = 150
    env = AirfoilLiftEnv(cfg)
    g = env.gust_gen.sine(3.0, 0.1)
    m0 = run_episode(env, NoControl(), g)["metrics"]
    m1 = run_episode(env, PIDController(dt=cfg.control.dt_ctrl), g)["metrics"]
    assert m1["cl_rmse"] < 0.6 * m0["cl_rmse"]


def test_action_smoothing_limits_rate() -> None:
    cfg = small_cfg()
    cfg.control.action_smoothing = 0.3
    env = AirfoilLiftEnv(cfg)
    env.reset(seed=0)
    env.step(np.array([1.0]))
    assert env.u == pytest.approx(0.3)


def test_gust_kinds_and_amplitude() -> None:
    cfg = Config()
    gen = GustGenerator(cfg.gust, cfg.flow, 0.4, cfg.control.dt_cfd)
    rng = np.random.default_rng(0)
    for kind in cfg.gust.kinds:
        g = gen.sample(rng, kind)
        assert np.all(np.isfinite(g.alpha))
        assert np.max(np.abs(np.rad2deg(g.alpha))) <= 1.5 * cfg.gust.amp_deg[1] + 1e-9
        assert abs(g(0.0)) < 1e-12  # 起始时刻无扰动（缓升包络）


def test_parse_compute_result() -> None:
    res = [{"cl": [0.61, 0]}, {"cd": [0.0142, 0]}]
    assert _parse_compute_result(res) == {"cl": 0.61, "cd": 0.0142}
    assert _parse_compute_result({"cl": 0.5}) == {"cl": 0.5}


def test_sysid_recovers_parameters() -> None:
    cfg = Config()
    truth = SurrogateParams(jet_gain_pos=0.3, cl_alpha=5.4)
    be = SurrogateBackend(cfg, stochastic=False, params=truth)
    data = collect(be, default_experiments(cfg, n_steps=60), verbose=False)
    est, summ = fit_surrogate(cfg, data, ["jet_gain_pos", "cl_alpha"], init=SurrogateParams(), max_nfev=40, verbose=0)
    assert est.jet_gain_pos == pytest.approx(0.3, rel=1e-3)
    assert est.cl_alpha == pytest.approx(5.4, rel=1e-3)
    assert summ["rms_after"] < 1e-4


def test_sac_smoke(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    from airfoil_rl.training import build_model, build_vec_env

    cfg = small_cfg()
    cfg.train.learning_starts = 50
    cfg.train.batch_size = 32
    cfg.train.net_arch = [16, 16]
    env = build_vec_env(cfg, 2, 0)
    model = build_model(cfg, env)
    model.learn(200)
    path = tmp_path / "m.zip"
    model.save(path)
    env.close()
    assert path.is_file()

    # 迁移微调路径：加载 + 三段式预热（采样 → 只更新 Critic → 正常更新）
    from stable_baselines3.common.callbacks import CallbackList

    from airfoil_rl.training import load_model, make_finetune_callback

    env2 = build_vec_env(cfg, 2, 1)
    m2 = load_model(str(path), "sac", env=env2, device="cpu", learning_starts=10)
    actor_before = [p.detach().clone() for p in m2.actor.parameters()]
    cb = make_finetune_callback(collect_steps=40, critic_steps=40)
    m2.learn(60, callback=CallbackList([cb]))  # 只到阶段 2：Actor 必须保持不变
    for a, b in zip(actor_before, m2.actor.parameters()):
        assert np.allclose(a.numpy(), b.detach().numpy())
    assert all(p.requires_grad for p in m2.actor.parameters())  # 训练结束后恢复
    env2.close()
