"""Unit tests for `src/models/gaussian_hmm.py` (§7.4 ablation baseline).

Tests moved from test_phase_module.py (M1.4 file restructure):
  - GaussianHMM tests (T1-T9 + T18 in original numbering) → here
  - Cohen's κ tests (T16-T17) → already in test_metrics_kappa.py
  - PhaseModule tests (T10-T15) → kept in test_phase_module.py

This module verifies the simpler i.i.d. Gaussian-emission HMM used for paper
§7.4 ablation comparison against the main NeuralSwitchingVARHMM path.

Run: pytest -xvs src/tests/test_gaussian_hmm.py
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.gaussian_hmm import (
    GaussianHMM,
    generate_synthetic_hmm_data,
)
from src.utils.metrics import cohens_kappa_aligned


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Well-separated 3-state synthetic HMM data."""
    x, states = generate_synthetic_hmm_data(K=3, V=4, T=500, seed=42)
    return x, states


@pytest.fixture
def fitted_hmm(synthetic_data):
    """HMM fitted on synthetic data."""
    x, _ = synthetic_data
    hmm = GaussianHMM(n_states=3, n_features=4, n_iter=50, seed=42)
    hmm.fit(x)
    return hmm


# ─────────────────────────────────────────────────────────────────
# T1 — fit + posterior shape
# ─────────────────────────────────────────────────────────────────

def test_hmm_fit_posterior_shape(fitted_hmm, synthetic_data):
    """HMM fit produces posteriors with shape [T, K]."""
    x, _ = synthetic_data
    gamma = fitted_hmm.posteriors(x)
    assert gamma.shape == (500, 3), f"Expected (500, 3), got {gamma.shape}"


# ─────────────────────────────────────────────────────────────────
# T2 — posteriors sum to 1
# ─────────────────────────────────────────────────────────────────

def test_posterior_sums_to_one(fitted_hmm, synthetic_data):
    """γ[t, :].sum() == 1 for all t (probability simplex constraint)."""
    x, _ = synthetic_data
    gamma = fitted_hmm.posteriors(x)
    row_sums = gamma.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
                               err_msg="Posterior rows don't sum to 1")


# ─────────────────────────────────────────────────────────────────
# T3 — posteriors in [0, 1]
# ─────────────────────────────────────────────────────────────────

def test_posterior_range(fitted_hmm, synthetic_data):
    """γ[t, k] ∈ [0, 1] (probability constraint)."""
    x, _ = synthetic_data
    gamma = fitted_hmm.posteriors(x)
    assert gamma.min() >= 0.0, f"Negative posterior: {gamma.min()}"
    assert gamma.max() <= 1.0, f"Posterior > 1: {gamma.max()}"


# ─────────────────────────────────────────────────────────────────
# T4 — EM convergence + monotone non-decreasing LL
# ─────────────────────────────────────────────────────────────────

def test_em_convergence(fitted_hmm):
    """EM converges before max_iter; LL monotonically non-decreasing (theorem)."""
    assert fitted_hmm.n_iter_run < fitted_hmm.n_iter, \
        f"EM didn't converge: ran all {fitted_hmm.n_iter} iterations"
    # Log-likelihood should be monotonically non-decreasing (Dempster-Laird-Rubin 1977)
    ll = fitted_hmm.ll_history
    for i in range(1, len(ll)):
        assert ll[i] >= ll[i - 1] - 1e-6, \
            f"LL decreased at step {i}: {ll[i]:.4f} < {ll[i-1]:.4f}"


# ─────────────────────────────────────────────────────────────────
# T5 — BIC computation + K selection consistency
# ─────────────────────────────────────────────────────────────────

def test_bic_computation(synthetic_data):
    """BIC is finite and K=3 (true K) has lower BIC than K=1 (under-fit)."""
    x, _ = synthetic_data

    hmm_1 = GaussianHMM(n_states=1, n_features=4, n_iter=50, seed=42).fit(x)
    hmm_3 = GaussianHMM(n_states=3, n_features=4, n_iter=50, seed=42).fit(x)

    bic_1 = hmm_1.bic(x)
    bic_3 = hmm_3.bic(x)

    assert np.isfinite(bic_1), "BIC(K=1) is not finite"
    assert np.isfinite(bic_3), "BIC(K=3) is not finite"
    assert bic_3 < bic_1, \
        f"True K=3 should have lower BIC: BIC(3)={bic_3:.1f} >= BIC(1)={bic_1:.1f}"


# ─────────────────────────────────────────────────────────────────
# T6 — dead state detection
# ─────────────────────────────────────────────────────────────────

def test_dead_state_detection(synthetic_data):
    """K=3 on well-separated 3-state data → no dead states."""
    x, _ = synthetic_data
    hmm = GaussianHMM(n_states=3, n_features=4, n_iter=50, seed=42).fit(x)
    dead = hmm.dead_states(x, threshold=0.05)
    assert len(dead) == 0, f"Unexpected dead states: {dead}"


def test_dead_state_over_parameterized(synthetic_data):
    """K=5 on 3-state data → mean state mass diagnostic (informational)."""
    x, _ = synthetic_data
    hmm = GaussianHMM(n_states=5, n_features=4, n_iter=50, seed=42).fit(x)
    gamma = hmm.posteriors(x)
    mean_post = gamma.mean(axis=0)
    # All 5 states should sum to 1 (sanity)
    np.testing.assert_allclose(mean_post.sum(), 1.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────
# T7 — Viterbi decoding (shape + integer range + true-state recovery)
# ─────────────────────────────────────────────────────────────────

def test_viterbi_shape(fitted_hmm, synthetic_data):
    """Viterbi output: [T] integers in [0, K)."""
    x, _ = synthetic_data
    states = fitted_hmm.viterbi(x)
    assert states.shape == (500,)
    assert states.dtype == int or np.issubdtype(states.dtype, np.integer)
    assert states.min() >= 0
    assert states.max() < 3


def test_viterbi_recovers_true_states(synthetic_data):
    """Viterbi on well-separated data recovers true states (aligned κ > 0.5)."""
    x, true_states = synthetic_data
    hmm = GaussianHMM(n_states=3, n_features=4, n_iter=50, seed=42).fit(x)
    pred_states = hmm.viterbi(x)

    # Aligned κ from metrics.py (orthogonal label permutation)
    kappa = cohens_kappa_aligned(true_states, pred_states, K=3)
    assert kappa > 0.5, f"κ={kappa:.3f} too low — Viterbi can't recover true states"


# ─────────────────────────────────────────────────────────────────
# T8 — serialization round-trip
# ─────────────────────────────────────────────────────────────────

def test_serialization_roundtrip(fitted_hmm, synthetic_data):
    """state_dict → JSON → from_state_dict → posteriors identical."""
    x, _ = synthetic_data

    # Save → JSON string → reload
    sd = fitted_hmm.state_dict()
    json_str = json.dumps(sd)
    sd_loaded = json.loads(json_str)
    hmm2 = GaussianHMM.from_state_dict(sd_loaded)

    # Posteriors must match
    gamma1 = fitted_hmm.posteriors(x)
    gamma2 = hmm2.posteriors(x)
    np.testing.assert_allclose(gamma1, gamma2, atol=1e-10,
                               err_msg="Posteriors differ after round-trip")


# ─────────────────────────────────────────────────────────────────
# T9 — free parameter count (full + diag)
# ─────────────────────────────────────────────────────────────────

def test_free_param_count():
    """Verify _n_free_params formula for full and diag covariance."""
    # Full: K=3, V=4
    hmm_full = GaussianHMM(n_states=3, n_features=4, covariance_type="full")
    # π: 2, A: 6, μ: 12, Σ (sym): 3×(4×5/2)=30 → total=50
    assert hmm_full._n_free_params() == 2 + 6 + 12 + 30

    # Diag: K=3, V=4
    hmm_diag = GaussianHMM(n_states=3, n_features=4, covariance_type="diag")
    # π: 2, A: 6, μ: 12, σ²: 3×4=12 → total=32
    assert hmm_diag._n_free_params() == 2 + 6 + 12 + 12


# ─────────────────────────────────────────────────────────────────
# T10 — NaN/Inf safety on all outputs
# ─────────────────────────────────────────────────────────────────

def test_no_nan_inf(fitted_hmm, synthetic_data):
    """No NaN/Inf in posteriors, BIC, Viterbi outputs."""
    x, _ = synthetic_data

    gamma = fitted_hmm.posteriors(x)
    assert not np.any(np.isnan(gamma)), "NaN in posteriors"
    assert not np.any(np.isinf(gamma)), "Inf in posteriors"

    bic = fitted_hmm.bic(x)
    assert np.isfinite(bic), f"BIC not finite: {bic}"

    states = fitted_hmm.viterbi(x)
    assert not np.any(np.isnan(states)), "NaN in Viterbi states"


# ─────────────────────────────────────────────────────────────────
# T11 — diag covariance mode (alternative emission)
# ─────────────────────────────────────────────────────────────────

def test_diag_covariance(synthetic_data):
    """Diagonal covariance mode fits and produces valid posteriors."""
    x, _ = synthetic_data
    hmm = GaussianHMM(n_states=3, n_features=4, covariance_type="diag",
                       n_iter=50, seed=42).fit(x)
    gamma = hmm.posteriors(x)
    assert gamma.shape == (500, 3)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────
# T12 — K range support (§3.7 grid K ∈ {3, 4, 5})
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [3, 4, 5])
def test_K_grid_fits(K, synthetic_data):
    """All K in PLAN §3.7 grid {3, 4, 5} fit on synthetic data without error."""
    x, _ = synthetic_data
    hmm = GaussianHMM(n_states=K, n_features=4, n_iter=50, seed=42).fit(x)
    gamma = hmm.posteriors(x)
    assert gamma.shape == (500, K)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)
    assert hmm._fitted
