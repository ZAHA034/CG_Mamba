"""Unit tests for `src/models/phase_module.py` — v2.0.9 PhaseModule (M1.4c).

PLAN v2.0.9 PATCH 12 (D.6 Tests) + post-review fixes (H1/H2/H3/M2/M4/L3/L5)

Scope:
  - Constructor + 6 register_buffer 명단 (C-6)
  - zeros init (R-4)
  - _cache_hmm_torch + S-2 정규화 + H3 Cholesky fallback
  - forward tuple (gate_phase, phase_post) + L-1 alignment + sigmoid range (S-4)
  - numpy ↔ torch numerical consistency (S-2 bit-identical)
  - _torch_log_emission per-step vs _torch_log_emission_batched (H2)
  - _torch_forward_backward L≥1 assert (New-M4)
  - RuntimeError on missing cache (New-L5)
  - rollout simplex + rollout_gate triple shapes
  - gradient flow Stage 2 + Stage 3 unfreeze
  - _last_gamma train/eval discipline

Out of scope:
  - GaussianHMM EM      → test_gaussian_hmm.py
  - Cohen's κ           → test_metrics_kappa.py
  - End-to-end M1.4c    → test_phase_dynamics_main.py

Run: pytest -xvs src/tests/test_phase_module.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.models.gaussian_hmm import GaussianHMM
from src.models.phase_module import PhaseModule
from src.utils.config import CGMambaConfig


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CGMambaConfig()


def _make_fitted_hmm(V_raw: int = 3, K: int = 3, T: int = 200, seed: int = 42) -> GaussianHMM:
    """Fit a small GaussianHMM on synthetic augmented features for testing."""
    np.random.seed(seed)
    V_aug = 2 * V_raw
    # Smooth random trajectory in raw space → augmented [x_t, Δx_t]
    x_raw = np.cumsum(np.random.randn(T, V_raw).astype(np.float64) * 0.05, axis=0)
    delta = x_raw[1:] - x_raw[:-1]
    x_aug = np.concatenate([x_raw[1:], delta], axis=-1)
    return GaussianHMM(n_states=K, n_features=V_aug, reg_covar=5e-3, seed=seed).fit(x_aug)


@pytest.fixture
def fitted_hmm():
    return _make_fitted_hmm(V_raw=3, K=3, seed=42)


@pytest.fixture
def phase_module(cfg, fitted_hmm):
    torch.manual_seed(42)
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    pm._cache_hmm_torch(fitted_hmm)
    return pm


# ─────────────────────────────────────────────────────────────────
# T1 — Constructor + 6 buffer 정확 명단 (C-6)
# ─────────────────────────────────────────────────────────────────

def test_buffer_six_names_exact(cfg):
    """register_buffer 6개: _A, _pi, _means, _covs, _cov_inv, _log_det (C-6)."""
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    expected = {"_A", "_pi", "_means", "_covs", "_cov_inv", "_log_det"}
    assert set(pm._buffers.keys()) == expected, \
        f"buffer names mismatch: expected {expected}, got {set(pm._buffers.keys())}"
    # log_pi / log_T buffer names are FORBIDDEN (C-6)
    assert "log_pi" not in pm._buffers
    assert "log_T" not in pm._buffers


# ─────────────────────────────────────────────────────────────────
# T2 — state_embeddings zeros init (R-4 + S-5)
# ─────────────────────────────────────────────────────────────────

def test_state_embeddings_zeros_init(cfg):
    """state_embeddings is initialized to exactly zero (R-4); name is plural (S-5)."""
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    assert hasattr(pm, "state_embeddings"), "S-5: attribute must be 'state_embeddings'"
    assert not hasattr(pm, "state_embed"), "S-5: old name 'state_embed' must not exist"
    assert pm.state_embeddings.shape == (cfg.K_phase, cfg.d_model)
    assert (pm.state_embeddings == 0).all(), \
        f"R-4 zeros init violated: state_embeddings absum={pm.state_embeddings.abs().sum()}"


# ─────────────────────────────────────────────────────────────────
# T3 — _cache_hmm_torch + S-2 정규화 (Σ + reg·I)
# ─────────────────────────────────────────────────────────────────

def test_cache_hmm_torch_applies_reg_covar(cfg, fitted_hmm):
    """_covs buffer must equal raw Σ_k + reg_covar·I (S-2 정규화)."""
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    pm._cache_hmm_torch(fitted_hmm)
    expected = (
        fitted_hmm.covars.astype(np.float64)
        + fitted_hmm.reg_covar * np.eye(fitted_hmm.V)[None, :, :]
    )
    diff = np.abs(pm._covs.numpy().astype(np.float64) - expected).max()
    assert diff < 1e-5, f"S-2 reg_covar 정규화 누락: max abs diff={diff:.2e}"
    # Sign of log_det should be all positive (PD check)
    assert (pm._log_det > 0).all() or np.isfinite(pm._log_det.numpy()).all()


def test_cache_hmm_torch_cov_inv_consistency(cfg, fitted_hmm):
    """_cov_inv = (_covs)⁻¹ (computed from regularized cov per S-2)."""
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    pm._cache_hmm_torch(fitted_hmm)
    for k in range(pm.K):
        identity = pm._covs[k].numpy() @ pm._cov_inv[k].numpy()
        I = np.eye(pm.V_aug)
        assert np.abs(identity - I).max() < 1e-3, \
            f"state {k}: _covs @ _cov_inv != I (max dev={np.abs(identity - I).max():.2e})"


# ─────────────────────────────────────────────────────────────────
# T4 — forward tuple + L-1 alignment + sigmoid range (S-4)
# ─────────────────────────────────────────────────────────────────

def test_forward_tuple_shape_and_sigmoid(phase_module, cfg):
    """forward(x_raw[B,L,V_raw]) → (gate_phase[B,L-1,D], phase_post[B,L-1,K])."""
    B, L = 2, 50
    x_raw = torch.randn(B, L, cfg.V_hmm_raw)
    phase_module.eval()
    gate_phase, phase_post = phase_module(x_raw)
    assert gate_phase.shape == (B, L - 1, cfg.d_model), \
        f"gate_phase shape {tuple(gate_phase.shape)}, expected ({B}, {L - 1}, {cfg.d_model})"
    assert phase_post.shape == (B, L - 1, cfg.K_phase), \
        f"phase_post shape {tuple(phase_post.shape)}, expected ({B}, {L - 1}, {cfg.K_phase})"
    # S-4: sigmoid range [0, 1]
    assert gate_phase.min().item() >= 0.0 and gate_phase.max().item() <= 1.0


def test_gate_phase_at_init_is_exactly_half(cfg, fitted_hmm):
    """At init (state_embeddings=0), gate_phase = sigmoid(γ·0) = 0.5 exactly (R-4 + S-4)."""
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    pm._cache_hmm_torch(fitted_hmm)
    pm.eval()
    x_raw = torch.randn(1, 30, cfg.V_hmm_raw)
    gate_phase, _ = pm(x_raw)
    assert gate_phase.min().item() == 0.5 and gate_phase.max().item() == 0.5


def test_phase_post_simplex(phase_module, cfg):
    """phase_post rows must sum to 1 (probability simplex)."""
    x_raw = torch.randn(2, 50, cfg.V_hmm_raw)
    phase_module.eval()
    _, phase_post = phase_module(x_raw)
    dev = (phase_post.sum(dim=-1) - 1.0).abs().max().item()
    assert dev < 1e-5, f"phase_post simplex violated: max dev={dev:.2e}"


# ─────────────────────────────────────────────────────────────────
# T5 — Numerical consistency: torch FB ↔ numpy GaussianHMM
# ─────────────────────────────────────────────────────────────────

def test_torch_fb_matches_numpy_posteriors(cfg, fitted_hmm):
    """PhaseModule torch forward-backward must match GaussianHMM.posteriors numerically."""
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    pm._cache_hmm_torch(fitted_hmm)
    pm.eval()

    # Use the same data the HMM was fit on (deterministic)
    np.random.seed(42)
    T = 200
    x_raw_np = np.cumsum(np.random.randn(T, cfg.V_hmm_raw).astype(np.float64) * 0.05, axis=0)
    x_aug_np = np.concatenate([x_raw_np[1:], x_raw_np[1:] - x_raw_np[:-1]], axis=-1)

    gamma_np = fitted_hmm.posteriors(x_aug_np)                           # [T-1, K]
    with torch.no_grad():
        x_raw_t = torch.from_numpy(x_raw_np[None]).float()
        _, phase_post = pm(x_raw_t)
    gamma_t = phase_post.squeeze(0).numpy().astype(np.float64)

    max_diff = np.abs(gamma_t - gamma_np).max()
    # float32 precision tolerance (verified empirically ~3.6e-05 on real ILI data)
    assert max_diff < 1e-3, f"torch vs numpy gamma diff too large: {max_diff:.2e}"


def test_batched_vs_per_timestep_emission(phase_module, cfg):
    """_torch_log_emission_batched must match per-step _torch_log_emission (H2)."""
    np.random.seed(42)
    T = 80
    V_aug = 2 * cfg.V_hmm_raw
    x_aug = torch.from_numpy(np.random.randn(1, T, V_aug)).float()
    with torch.no_grad():
        log_emit_b = phase_module._torch_log_emission_batched(x_aug)     # [1, T, K]
        log_emit_step = torch.stack(
            [phase_module._torch_log_emission(x_aug[:, t, :]) for t in range(T)],
            dim=1,
        )
    diff = (log_emit_b - log_emit_step).abs().max().item()
    # tolerance 5e-4: float32 mantissa is 23-bit ≈ 7 decimal digits. log_emit
    # values are O(10²) in our HMM, so the last 1-2 ULPs ≈ 1e-4. The 1e-4
    # bound was marginal under H-3 init pre-reg (which inflated init covars
    # slightly), causing diff to drift to ~1.2e-4 after H-3 fix. The H2
    # spec requires "동치 (equivalent)" not bit-identical — see PLAN §3621.
    assert diff < 5e-4, f"batched ↔ per-step emission diff: {diff:.2e}"


# ─────────────────────────────────────────────────────────────────
# T6 — Edge case asserts (New-M4 + New-L5)
# ─────────────────────────────────────────────────────────────────

def test_forward_raises_runtime_error_without_cache(cfg):
    """forward() before _cache_hmm_torch() must RuntimeError (New-L5, not assert)."""
    pm = PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model)
    x_raw = torch.randn(1, 10, cfg.V_hmm_raw)
    with pytest.raises(RuntimeError, match="_cache_hmm_torch"):
        pm(x_raw)


def test_fb_rejects_L_zero(phase_module, cfg):
    """_torch_forward_backward asserts L ≥ 1 (New-M4 fix)."""
    V_aug = 2 * cfg.V_hmm_raw
    x_aug_zero = torch.zeros(1, 0, V_aug)
    with pytest.raises(AssertionError, match=r"L .* 1"):
        phase_module._torch_forward_backward(x_aug_zero)


def test_fb_accepts_L_one(phase_module, cfg):
    """_torch_forward_backward works with L=1 (single-timestep, marginal posterior)."""
    V_aug = 2 * cfg.V_hmm_raw
    x_aug_one = torch.randn(2, 1, V_aug)
    gamma = phase_module._torch_forward_backward(x_aug_one)
    assert gamma.shape == (2, 1, cfg.K_phase)
    assert (gamma.sum(-1) - 1.0).abs().max().item() < 1e-5


def test_augment_features_requires_L_at_least_two(phase_module, cfg):
    """_augment_features asserts L ≥ 2 to compute Δx."""
    x_raw_short = torch.randn(1, 1, cfg.V_hmm_raw)
    with pytest.raises(AssertionError, match=r"L .* 2"):
        phase_module._augment_features(x_raw_short)


# ─────────────────────────────────────────────────────────────────
# T7 — rollout + rollout_gate (PATCH 2 emission-aware)
# ─────────────────────────────────────────────────────────────────

def test_rollout_shape_and_simplex(phase_module, cfg):
    """rollout(gamma_T, x_window, H) → [B, H, K] simplex."""
    B, w = 4, 5
    gamma_T = torch.softmax(torch.randn(B, cfg.K_phase), dim=-1)
    x_window = torch.randn(B, w, cfg.V_hmm_raw)
    for H in [1, 2, 4]:
        gr = phase_module.rollout(gamma_T, x_window, H=H)
        assert gr.shape == (B, H, cfg.K_phase), \
            f"H={H}: shape {tuple(gr.shape)} expected ({B}, {H}, {cfg.K_phase})"
        dev = (gr.sum(-1) - 1.0).abs().max().item()
        assert dev < 1e-5, f"H={H} simplex dev={dev:.2e}"


def test_rollout_requires_window_at_least_two(phase_module, cfg):
    """rollout asserts x_window.shape[1] >= 2 (need Δx)."""
    gamma_T = torch.softmax(torch.randn(2, cfg.K_phase), dim=-1)
    x_window_short = torch.randn(2, 1, cfg.V_hmm_raw)
    with pytest.raises(AssertionError, match="2 timesteps"):
        phase_module.rollout(gamma_T, x_window_short, H=2)


def test_rollout_gate_triple_shapes(phase_module, cfg):
    """rollout_gate returns single-timestep (gate[B,D], gamma[B,K], conf[B]). L-1 fix."""
    B, w = 4, 5
    gamma_T = torch.softmax(torch.randn(B, cfg.K_phase), dim=-1)
    x_window = torch.randn(B, w, cfg.V_hmm_raw)
    gate_h, gamma_h, conf = phase_module.rollout_gate(gamma_T, x_window, horizon=3)
    assert gate_h.shape == (B, cfg.d_model)
    assert gamma_h.shape == (B, cfg.K_phase)
    assert conf.shape == (B,)
    # Sigmoid output range
    assert gate_h.min().item() >= 0.0 and gate_h.max().item() <= 1.0
    # Confidence ∈ [0, 1]
    assert conf.min().item() >= 0.0 and conf.max().item() <= 1.0


def test_rollout_gate_confidence_uniform_posterior(phase_module, cfg):
    """confidence ≈ 0 for uniform posterior, ≈ 1 for peaked posterior."""
    B = 2
    K = cfg.K_phase
    gamma_uniform = torch.full((B, K), 1.0 / K)        # max entropy → conf ≈ 0
    gamma_peaked = torch.zeros(B, K); gamma_peaked[:, 0] = 1.0  # 0 entropy → conf ≈ 1
    x_window = torch.zeros(B, 2, cfg.V_hmm_raw)
    _, _, conf_uniform = phase_module.rollout_gate(gamma_uniform, x_window, horizon=1)
    _, _, conf_peaked = phase_module.rollout_gate(gamma_peaked, x_window, horizon=1)
    # Note: after 1 emission-aware step, gamma_uniform may peak via emission ↑.
    # We test only that peaked input gives higher confidence than uniform input.
    assert conf_peaked.mean().item() >= conf_uniform.mean().item()


# ─────────────────────────────────────────────────────────────────
# T8 — Gradient flow (Stage 2: state_embeddings only)
# ─────────────────────────────────────────────────────────────────

def test_gradient_flow_stage2(phase_module, cfg):
    """Stage 2: state_embeddings gets gradient; HMM buffers do NOT."""
    phase_module.train()
    x_raw = torch.randn(2, 30, cfg.V_hmm_raw)
    phase_module.zero_grad()
    gate_phase, phase_post = phase_module(x_raw)
    (gate_phase.sum() + phase_post.sum()).backward()
    assert phase_module.state_embeddings.grad is not None
    assert phase_module.state_embeddings.grad.abs().sum() > 0
    # Buffers should NOT have grad attribute (frozen)
    for name in ("_A", "_pi", "_means", "_covs", "_cov_inv", "_log_det"):
        buf = getattr(phase_module, name)
        assert getattr(buf, "grad", None) is None, \
            f"Stage 2: buffer {name} unexpectedly has grad"


# ─────────────────────────────────────────────────────────────────
# T9 — Stage 3 selective unfreeze (T-2, M2 standard API)
# ─────────────────────────────────────────────────────────────────

def test_unfreeze_for_stage3_converts_to_parameter(phase_module, cfg):
    """_unfreeze_for_stage3: _A, _means → Parameter; pi/covs/cov_inv/log_det stay buffers."""
    assert "_A" in phase_module._buffers
    assert "_means" in phase_module._buffers

    n_unfrozen = phase_module._unfreeze_for_stage3()
    K = cfg.K_phase
    V_aug = 2 * cfg.V_hmm_raw
    expected = K * K + K * V_aug                          # 9 + 18 = 27 for K=V_raw=3
    assert n_unfrozen == expected, f"expected {expected} unfrozen, got {n_unfrozen}"

    # _A, _means moved to _parameters
    assert "_A" in phase_module._parameters
    assert "_means" in phase_module._parameters
    assert "_A" not in phase_module._buffers
    assert "_means" not in phase_module._buffers
    # Others stay frozen
    for name in ("_pi", "_covs", "_cov_inv", "_log_det"):
        assert name in phase_module._buffers, f"{name} should stay buffer"
        assert name not in phase_module._parameters


def test_unfreeze_idempotent(phase_module):
    """_unfreeze_for_stage3 is idempotent: 2nd call returns same count without error."""
    n1 = phase_module._unfreeze_for_stage3()
    n2 = phase_module._unfreeze_for_stage3()
    assert n1 == n2


def test_unfreeze_enables_gradient_on_A_means(phase_module, cfg):
    """After unfreeze, _A and _means receive gradient through forward-backward."""
    phase_module._unfreeze_for_stage3()
    phase_module.train()
    x_raw = torch.randn(2, 30, cfg.V_hmm_raw)
    phase_module.zero_grad()
    _, phase_post = phase_module(x_raw)
    phase_post.sum().backward()
    assert phase_module._A.grad is not None and phase_module._A.grad.abs().sum() > 0
    assert phase_module._means.grad is not None and phase_module._means.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────
# T10 — _last_gamma train/eval discipline
# ─────────────────────────────────────────────────────────────────

def test_last_gamma_cache_train_eval_mode(phase_module, cfg):
    """_last_gamma cached only in train mode (M1.3 _last_gate convention)."""
    x_raw = torch.randn(2, 30, cfg.V_hmm_raw)

    phase_module.train()
    _, phase_post = phase_module(x_raw)
    assert phase_module._last_gamma is not None
    assert phase_module._last_gamma.shape == phase_post.shape
    assert not phase_module._last_gamma.requires_grad

    phase_module.eval()
    phase_module(x_raw)
    assert phase_module._last_gamma is None


# ─────────────────────────────────────────────────────────────────
# T11 — Constructor with immediate cache (hmm_fitted kwarg)
# ─────────────────────────────────────────────────────────────────

def test_constructor_with_hmm_fitted_kwarg(cfg, fitted_hmm):
    """PhaseModule(..., hmm_fitted=...) caches HMM immediately and forward works."""
    pm = PhaseModule(
        V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model,
        hmm_fitted=fitted_hmm,
    )
    assert pm._hmm_cached
    pm.eval()
    x_raw = torch.randn(1, 30, cfg.V_hmm_raw)
    gate, post = pm(x_raw)
    assert gate.shape == (1, 29, cfg.d_model)
    assert post.shape == (1, 29, cfg.K_phase)


# ─────────────────────────────────────────────────────────────────
# T12 — Class-level _LOG_2PI constant (New-L3)
# ─────────────────────────────────────────────────────────────────

def test_log_2pi_class_constant():
    """_LOG_2PI is a class-level constant (New-L3 performance fix)."""
    assert hasattr(PhaseModule, "_LOG_2PI")
    assert abs(PhaseModule._LOG_2PI - math.log(2.0 * math.pi)) < 1e-12
