"""End-to-end sanity tests for v2.0.9 Phase Dynamics main path (M1.4c).

PLAN v2.0.9 PATCH 12 (D.6 신규) — verifies the full Stage-1 → Stage-2 entry
sequence on the M1.4b winner setting (V_raw=3, K=3, reg_covar=5e-3, n_init=5).

Scope (integration-level, not unit-level):
  1. M1.4b winner config produces a healthy GaussianHMM on synthetic data
     (no dead state, no cov collapse).
  2. PhaseModule cfg-driven instantiation (S-7 pattern).
  3. Stage 2 entry sequence (T-1): cache HMM → build optimizer → forward → backward.
  4. Determinism: same input + cached HMM → same output (torch.no_grad).
  5. Cross-seed κ on synthetic data: two seeds fit on the same data agree.

Unit-level coverage is in src/tests/test_phase_module.py.

Run: pytest -xvs src/tests/test_phase_dynamics_main.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.gaussian_hmm import GaussianHMM
from src.models.phase_module import PhaseModule
from src.utils.config import CGMambaConfig
from src.utils.metrics import (
    cohens_kappa_aligned,
    is_dead_state,
    state_occupancy,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures: M1.4b winner config + deterministic synthetic data
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CGMambaConfig()


@pytest.fixture
def synthetic_x_raw_aug():
    """Smooth deterministic trajectory; returns (x_raw [T,V_raw], x_aug [T-1,V_aug])."""
    np.random.seed(42)
    T = 300
    V_raw = 3
    x_raw = np.cumsum(np.random.randn(T, V_raw).astype(np.float64) * 0.05, axis=0)
    delta = x_raw[1:] - x_raw[:-1]
    x_aug = np.concatenate([x_raw[1:], delta], axis=-1)
    return x_raw, x_aug


# ─────────────────────────────────────────────────────────────────
# T1 — M1.4b winner config (V_raw=3, K=3, reg_covar=5e-3, n_init=5)
# ─────────────────────────────────────────────────────────────────

def test_m1_4b_winner_config_is_default(cfg):
    """CGMambaConfig defaults match M1.4b winner setting."""
    assert cfg.V_hmm_raw == 3
    assert cfg.K_phase == 3
    assert cfg.hmm_reg_covar == 5e-3
    assert cfg.hmm_n_init == 5
    assert cfg.stage3_enabled is False


# ─────────────────────────────────────────────────────────────────
# T2 — Stage 1: GaussianHMM fits healthy on M1.4b winner setting
# ─────────────────────────────────────────────────────────────────

def test_stage1_healthy_fit(cfg, synthetic_x_raw_aug):
    """Stage 1 (offline EM) with winner reg_covar produces no dead state / no cov collapse."""
    _, x_aug = synthetic_x_raw_aug
    hmm = GaussianHMM(
        n_states=cfg.K_phase,
        n_features=2 * cfg.V_hmm_raw,
        reg_covar=cfg.hmm_reg_covar,
        seed=42 * 1000,
    ).fit(x_aug)
    viterbi = hmm.viterbi(x_aug)
    occ = state_occupancy(viterbi, K=cfg.K_phase)
    assert not is_dead_state(occ), f"dead state in winner setting: occupancy={occ}"
    # Check covariance well-conditioned (relative to reg_covar floor)
    for k in range(cfg.K_phase):
        cov_reg = hmm.covars[k] + hmm.reg_covar * np.eye(hmm.V)
        min_eig = np.linalg.eigvalsh(cov_reg).min()
        assert min_eig > 0.5 * cfg.hmm_reg_covar, \
            f"state {k}: cov_reg min eigval {min_eig:.2e} < 0.5·reg_covar"


# ─────────────────────────────────────────────────────────────────
# T3 — S-7: cfg-driven PhaseModule instantiation
# ─────────────────────────────────────────────────────────────────

def test_s7_cfg_driven_instantiation(cfg, synthetic_x_raw_aug):
    """PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model) — S-7 pattern."""
    _, x_aug = synthetic_x_raw_aug
    hmm = GaussianHMM(
        n_states=cfg.K_phase,
        n_features=2 * cfg.V_hmm_raw,
        reg_covar=cfg.hmm_reg_covar,
        seed=42 * 1000,
    ).fit(x_aug)
    pm = PhaseModule(
        V_raw=cfg.V_hmm_raw,
        K=cfg.K_phase,
        d_embed=cfg.d_model,
        hmm_fitted=hmm,
    )
    assert pm.V_raw == cfg.V_hmm_raw
    assert pm.V_aug == 2 * cfg.V_hmm_raw
    assert pm.K == cfg.K_phase
    assert pm.d_embed == cfg.d_model
    assert pm._hmm_cached


# ─────────────────────────────────────────────────────────────────
# T4 — T-1 Stage 2 entry sequence: cache → optimizer → forward → backward
# ─────────────────────────────────────────────────────────────────

def test_t1_stage2_entry_sequence(cfg, synthetic_x_raw_aug):
    """Verifies the documented Stage 2 entry order produces grad on state_embeddings only."""
    _, x_aug = synthetic_x_raw_aug
    hmm = GaussianHMM(
        n_states=cfg.K_phase,
        n_features=2 * cfg.V_hmm_raw,
        reg_covar=cfg.hmm_reg_covar,
        seed=42 * 1000,
    ).fit(x_aug)

    # Step 1: instantiate (no cache yet)
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    # Step 2: cache HMM (T-1, mandatory before forward)
    pm._cache_hmm_torch(hmm)
    # Step 3: build optimizer over model.parameters() (buffers auto-excluded)
    optimizer = torch.optim.AdamW(pm.parameters(), lr=1e-3)
    # state_embeddings should be the only trainable param in Stage 2
    n_trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    assert n_trainable == cfg.K_phase * cfg.d_model, \
        f"Stage 2 should have only state_embeddings trainable (got {n_trainable})"

    # Forward + backward + step
    pm.train()
    x_raw = torch.randn(2, 30, cfg.V_hmm_raw)
    gate_phase, phase_post = pm(x_raw)
    loss = gate_phase.pow(2).mean() + phase_post.pow(2).mean()
    loss.backward()
    optimizer.step()

    # state_embeddings should have moved off zero after one step
    assert pm.state_embeddings.abs().sum().item() > 0


# ─────────────────────────────────────────────────────────────────
# T5 — Determinism: same x_raw + cached HMM → identical output (eval mode)
# ─────────────────────────────────────────────────────────────────

def test_determinism_eval_mode(cfg, synthetic_x_raw_aug):
    """Same x_raw → same (gate_phase, phase_post) in eval mode (no dropout, deterministic FB)."""
    _, x_aug = synthetic_x_raw_aug
    hmm = GaussianHMM(
        n_states=cfg.K_phase,
        n_features=2 * cfg.V_hmm_raw,
        reg_covar=cfg.hmm_reg_covar,
        seed=42 * 1000,
    ).fit(x_aug)
    pm = PhaseModule(
        V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model, hmm_fitted=hmm,
    )
    pm.eval()
    torch.manual_seed(0)
    x_raw = torch.randn(2, 50, cfg.V_hmm_raw)
    with torch.no_grad():
        gp1, pp1 = pm(x_raw)
        gp2, pp2 = pm(x_raw)
    assert torch.equal(gp1, gp2), "forward not deterministic on identical input"
    assert torch.equal(pp1, pp2)


# ─────────────────────────────────────────────────────────────────
# T6 — Cross-seed κ on synthetic data (deterministic, simpler than M1.4b real run)
# ─────────────────────────────────────────────────────────────────

def test_cross_seed_kappa_on_synthetic(cfg, synthetic_x_raw_aug):
    """Two seeds fit on the same synthetic data should yield similar Viterbi paths.

    Looser bound than M1.4b production (κ_min=1.000 on real data with n_init=5):
    here single-init on synthetic random walk → κ ≥ 0.3 is acceptable sanity.
    """
    _, x_aug = synthetic_x_raw_aug
    paths = []
    for seed in [42, 123]:
        hmm = GaussianHMM(
            n_states=cfg.K_phase,
            n_features=2 * cfg.V_hmm_raw,
            reg_covar=cfg.hmm_reg_covar,
            seed=seed,
        ).fit(x_aug)
        paths.append(hmm.viterbi(x_aug))
    kappa = cohens_kappa_aligned(paths[0], paths[1], K=cfg.K_phase)
    assert kappa > 0.3, f"cross-seed κ={kappa:.4f} too low on synthetic data"
