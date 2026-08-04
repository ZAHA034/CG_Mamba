"""EGCPMPhaseModule unit tests — entropy gating, shape, NumPy parity, K-flex.

Run: pytest -xvs src/tests/test_egcpm_phase_module.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.egcpm_phase_module import EGCPMPhaseModule
from src.models.cell_cycle_hmm import entropy_gated_phase_embedding


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_module():
    """K=4, d_embed=8 EGCPMPhaseModule with entropy gating ON."""
    torch.manual_seed(42)
    return EGCPMPhaseModule(K=4, d_embed=8, use_entropy_gating=True)


@pytest.fixture
def random_gamma_3d():
    """[B=2, L=10, K=4] valid posterior (Dirichlet sampled, sums to 1)."""
    rng = np.random.RandomState(42)
    gamma_np = rng.dirichlet(np.ones(4), size=(2, 10))    # [2, 10, 4]
    return torch.tensor(gamma_np, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────
# T1 — Shape & forward
# ──────────────────────────────────────────────────────────────────

def test_T1_forward_shape(small_module, random_gamma_3d):
    """[B, L, K] γ → [B, L, d_embed] gate."""
    gate = small_module(random_gamma_3d)
    assert gate.shape == (2, 10, 8), f"Expected (2, 10, 8), got {tuple(gate.shape)}"


# ──────────────────────────────────────────────────────────────────
# T2 — NumPy parity with cell_cycle_hmm.entropy_gated_phase_embedding
# ──────────────────────────────────────────────────────────────────

def test_T2_numpy_parity():
    """Torch implementation matches NumPy reference within 1e-5.

    NumPy reference clips γ before BOTH factors of `γ log γ`, while torch
    clips only the log factor. For γ ≥ eps=1e-12 (always true after a
    healthy HMM fit), both formulations are numerically identical to FP32.
    """
    torch.manual_seed(0)
    K, D, B, L = 4, 16, 2, 8

    # Use the same state_embeddings (random) in both NumPy and torch.
    E_np = np.random.RandomState(0).randn(K, D).astype(np.float64)
    rng = np.random.RandomState(1)
    gamma_np = rng.dirichlet(np.ones(K), size=(B, L)).astype(np.float64)  # [B, L, K]

    # NumPy reference
    gate_np, H_np, c_np = entropy_gated_phase_embedding(gamma_np, E_np)

    # Torch implementation — inject the same E into state_embeddings
    mod = EGCPMPhaseModule(K=K, d_embed=D, use_entropy_gating=True)
    with torch.no_grad():
        mod.state_embeddings.copy_(torch.tensor(E_np, dtype=torch.float32))
    gamma_torch = torch.tensor(gamma_np, dtype=torch.float32)
    mod.train()
    gate_torch = mod(gamma_torch)

    # Compare gate
    np.testing.assert_allclose(
        gate_torch.detach().numpy().astype(np.float64), gate_np,
        atol=1e-5, rtol=1e-5,
    )
    # Compare H and c via stored attributes
    np.testing.assert_allclose(
        mod._last_entropy.numpy().astype(np.float64), H_np,
        atol=1e-5, rtol=1e-5,
    )
    np.testing.assert_allclose(
        mod._last_confidence.numpy().astype(np.float64), c_np,
        atol=1e-5, rtol=1e-5,
    )


# ──────────────────────────────────────────────────────────────────
# T3 — use_entropy_gating=False ablation path
# ──────────────────────────────────────────────────────────────────

def test_T3_ablation_path_no_gating():
    """use_entropy_gating=False returns plain γ @ E (no c scaling)."""
    torch.manual_seed(0)
    mod = EGCPMPhaseModule(K=4, d_embed=8, use_entropy_gating=False)
    with torch.no_grad():
        mod.state_embeddings.copy_(torch.randn(4, 8))

    rng = np.random.RandomState(7)
    gamma = torch.tensor(
        rng.dirichlet(np.ones(4), size=(2, 5)),
        dtype=torch.float32,
    )

    gate = mod(gamma)
    expected = gamma @ mod.state_embeddings   # γ @ E, no c

    torch.testing.assert_close(gate, expected, atol=1e-6, rtol=1e-6)

    # Monitoring fields cleared when gating disabled
    assert mod._last_entropy is None
    assert mod._last_confidence is None


# ──────────────────────────────────────────────────────────────────
# T4 — zeros init: gate ≈ 0 at construction
# ──────────────────────────────────────────────────────────────────

def test_T4_zeros_init_gate_is_zero(random_gamma_3d):
    """Newly constructed module has state_embeddings=0 → gate=0."""
    mod = EGCPMPhaseModule(K=4, d_embed=8, use_entropy_gating=True)
    # Verify init
    assert torch.allclose(mod.state_embeddings, torch.zeros(4, 8))

    gate = mod(random_gamma_3d)
    assert torch.allclose(gate, torch.zeros_like(gate))


# ──────────────────────────────────────────────────────────────────
# T5 — confidence is always in [0, 1]
# ──────────────────────────────────────────────────────────────────

def test_T5_confidence_in_unit_interval():
    """For arbitrary valid γ, _last_confidence ∈ [0, 1]."""
    mod = EGCPMPhaseModule(K=4, d_embed=8, use_entropy_gating=True)
    mod.train()

    rng = np.random.RandomState(42)
    for _ in range(50):
        gamma = torch.tensor(
            rng.dirichlet(np.ones(4), size=(3, 20)),
            dtype=torch.float32,
        )
        _ = mod(gamma)
        c = mod._last_confidence
        assert c.min().item() >= 0.0, f"c.min = {c.min().item()}"
        assert c.max().item() <= 1.0, f"c.max = {c.max().item()}"


# ──────────────────────────────────────────────────────────────────
# T6 — uniform γ → c=0 → gate=0
# ──────────────────────────────────────────────────────────────────

def test_T6_uniform_gamma_zero_gate():
    """Maximum-entropy γ (uniform) yields c=0 and gate=0 regardless of E."""
    K, D = 4, 16
    mod = EGCPMPhaseModule(K=K, d_embed=D, use_entropy_gating=True)
    # Non-zero E so the gate would be non-zero if c were not 0
    with torch.no_grad():
        mod.state_embeddings.copy_(torch.randn(K, D))

    gamma = torch.full((1, 5, K), 1.0 / K, dtype=torch.float32)
    mod.train()
    gate = mod(gamma)

    assert torch.allclose(gate, torch.zeros_like(gate), atol=1e-6)
    # Confidence is mathematically 0 (modulo float)
    assert mod._last_confidence.abs().max().item() < 1e-6


# ──────────────────────────────────────────────────────────────────
# T7 — K=3 compatibility (G2/M merged case)
# ──────────────────────────────────────────────────────────────────

def test_T7_K3_compatibility():
    """K=3 module accepts [B, L, 3] γ and emits [B, L, d_embed] gate."""
    mod = EGCPMPhaseModule(K=3, d_embed=12, use_entropy_gating=True)
    with torch.no_grad():
        mod.state_embeddings.copy_(torch.randn(3, 12))

    rng = np.random.RandomState(0)
    gamma = torch.tensor(
        rng.dirichlet(np.ones(3), size=(2, 7)),
        dtype=torch.float32,
    )
    gate = mod(gamma)
    assert gate.shape == (2, 7, 12)

    # K mismatch should raise RuntimeError (A.6 fix: was AssertionError)
    bad_gamma = torch.tensor(
        rng.dirichlet(np.ones(4), size=(2, 7)),
        dtype=torch.float32,
    )
    with pytest.raises(RuntimeError, match="K=3"):
        mod(bad_gamma)


# ──────────────────────────────────────────────────────────────────
# T8a — ablation path clears _last_* unconditionally (no stale state)
# ──────────────────────────────────────────────────────────────────

def test_T8a_ablation_clears_stale_metrics():
    """Switching to ablation mode after a training pass must not leave
    _last_* with stale tensor values (F1 regression guard)."""
    mod = EGCPMPhaseModule(K=4, d_embed=8, use_entropy_gating=True)
    with torch.no_grad():
        mod.state_embeddings.copy_(torch.randn(4, 8))
    rng = np.random.RandomState(0)
    gamma = torch.tensor(
        rng.dirichlet(np.ones(4), size=(2, 5)), dtype=torch.float32,
    )

    # 1) Train forward with entropy gating → _last_* populated
    mod.train()
    _ = mod(gamma)
    assert mod._last_entropy is not None
    assert mod._last_confidence is not None

    # 2) Flip to ablation mode (still training) → _last_* must clear
    mod.use_entropy_gating = False
    _ = mod(gamma)
    assert mod._last_entropy is None, "stale entropy after ablation flip"
    assert mod._last_confidence is None, "stale confidence after ablation flip"

    # 3) Also in eval mode (ablation)
    mod.eval()
    _ = mod(gamma)
    assert mod._last_entropy is None
    assert mod._last_confidence is None


# ──────────────────────────────────────────────────────────────────
# T8 — state_embeddings is learnable (gradient flow)
# ──────────────────────────────────────────────────────────────────

def test_T8_gradient_flow():
    """Backprop through gate fills state_embeddings.grad."""
    K, D = 4, 8
    mod = EGCPMPhaseModule(K=K, d_embed=D, use_entropy_gating=True)
    # Break the zeros init so the loss has signal
    with torch.no_grad():
        mod.state_embeddings.add_(torch.randn(K, D) * 0.1)

    rng = np.random.RandomState(0)
    gamma = torch.tensor(
        rng.dirichlet(np.ones(K), size=(2, 5)),
        dtype=torch.float32,
    )

    gate = mod(gamma)
    loss = gate.pow(2).sum()
    loss.backward()

    assert mod.state_embeddings.grad is not None
    assert torch.isfinite(mod.state_embeddings.grad).all()
    # Some entries should have non-trivial gradient
    assert mod.state_embeddings.grad.abs().max().item() > 1e-6


# ──────────────────────────────────────────────────────────────────
# T9-T13 — init_from_hmm B-4 symmetry hazard fix (Direction Message §2.3)
# ──────────────────────────────────────────────────────────────────

def _build_fitted_hmm(K: int = 4, V: int = 16, seed: int = 42):
    """Construct a fitted CellCycleHMM on synthetic data (helper)."""
    from src.models.cell_cycle_hmm import CellCycleHMM, generate_synthetic_cell_cycle
    x, _, _ = generate_synthetic_cell_cycle(
        n_genes=V, n_timepoints=48, K=K, seed=seed,
    )
    hmm = CellCycleHMM(n_states=K, n_features=V, n_iter=40, seed=seed)
    hmm.fit(x)
    return hmm


def test_T9_init_from_hmm_breaks_symmetry():
    """B-4 core claim: K rows of state_embeddings are distinct after init."""
    mod = EGCPMPhaseModule(K=4, d_embed=16, use_entropy_gating=True)
    # Before: zeros init → all K rows identical
    assert (mod.state_embeddings.var(dim=0).max() < 1e-6).item()

    hmm = _build_fitted_hmm(K=4, V=16)
    mod.init_from_hmm(hmm)

    # After: each pair of rows is distinct
    K = mod.K
    for i in range(K):
        for j in range(i + 1, K):
            dist = (mod.state_embeddings[i] - mod.state_embeddings[j]).norm().item()
            assert dist > 1e-3, f"rows {i} and {j} too similar: {dist:.6f}"


def test_T10_init_from_hmm_deterministic():
    """Same hmm + same seed → identical init across two modules."""
    hmm = _build_fitted_hmm(K=4, V=16)
    mod_a = EGCPMPhaseModule(K=4, d_embed=16)
    mod_b = EGCPMPhaseModule(K=4, d_embed=16)
    mod_a.init_from_hmm(hmm, seed=42)
    mod_b.init_from_hmm(hmm, seed=42)
    torch.testing.assert_close(mod_a.state_embeddings, mod_b.state_embeddings,
                                atol=1e-10, rtol=1e-10)


def test_T11_init_from_hmm_different_seed_differs():
    """Different seed → different projection → different init."""
    hmm = _build_fitted_hmm(K=4, V=16)
    mod_a = EGCPMPhaseModule(K=4, d_embed=16)
    mod_b = EGCPMPhaseModule(K=4, d_embed=16)
    mod_a.init_from_hmm(hmm, seed=42)
    mod_b.init_from_hmm(hmm, seed=99)
    diff = (mod_a.state_embeddings - mod_b.state_embeddings).abs().max().item()
    assert diff > 1e-3


def test_T12_init_from_hmm_K_mismatch_raises():
    """K mismatch between module and hmm raises ValueError."""
    hmm = _build_fitted_hmm(K=4, V=16)
    mod = EGCPMPhaseModule(K=3, d_embed=16)
    with pytest.raises(ValueError, match="K mismatch"):
        mod.init_from_hmm(hmm)


def test_T13_init_from_hmm_unfitted_raises():
    """Unfitted hmm raises ValueError."""
    from src.models.cell_cycle_hmm import CellCycleHMM
    hmm = CellCycleHMM(n_features=16)
    mod = EGCPMPhaseModule(K=4, d_embed=16)
    with pytest.raises(ValueError, match="fitted"):
        mod.init_from_hmm(hmm)


def test_T14_init_from_hmm_K3():
    """K=3 case works (12 markers, K=3 cyclic mask)."""
    hmm = _build_fitted_hmm(K=3, V=12)
    mod = EGCPMPhaseModule(K=3, d_embed=12, use_entropy_gating=True)
    mod.init_from_hmm(hmm)
    # All 3 rows distinct
    for i in range(3):
        for j in range(i + 1, 3):
            dist = (mod.state_embeddings[i] - mod.state_embeddings[j]).norm().item()
            assert dist > 1e-3


def test_T15_init_with_full_covariance():
    """init_from_hmm extracts diagonal correctly when covars is [K, V, V]."""
    from src.models.cell_cycle_hmm import CellCycleHMM, generate_synthetic_cell_cycle
    x, _, _ = generate_synthetic_cell_cycle(
        n_genes=16, n_timepoints=48, K=4, seed=42,
    )
    hmm = CellCycleHMM(
        n_states=4, n_features=16, covariance_type="full",
        n_iter=20, seed=42,
    )
    hmm.fit(x)
    mod = EGCPMPhaseModule(K=4, d_embed=16)
    mod.init_from_hmm(hmm)
    # No NaN, rows distinct
    assert torch.isfinite(mod.state_embeddings).all()
    var_per_dim = mod.state_embeddings.var(dim=0)
    assert var_per_dim.max().item() > 1e-6


# ──────────────────────────────────────────────────────────────────
# F12 — K-dim guard raises RuntimeError (5차 review §2.3 / A.6 fix)
# ──────────────────────────────────────────────────────────────────

def test_F12_K_mismatch_raises_RuntimeError():
    """K-mismatch in gamma_window raises RuntimeError with explicit message.

    Why RuntimeError (not AssertionError):
      - `python -O` strips `assert` statements → silent bypass in production
      - RuntimeError survives optimization mode
      - Mirrors ILI PhaseModule.forward convention

    Why this matters for Step 5 ablation:
      - K=3 vs K=4 HMMs produce γ of shape [B, L, 3] vs [B, L, 4]
      - Feeding γ from the wrong HMM into a forecaster of the other K
        would silently produce shape-broken outputs without this guard.
    """
    K_module = 3
    mod = EGCPMPhaseModule(K=K_module, d_embed=12)

    # Wrong K (4 instead of 3)
    rng = np.random.RandomState(0)
    bad_gamma = torch.tensor(
        rng.dirichlet(np.ones(4), size=(2, 7)),
        dtype=torch.float32,
    )
    with pytest.raises(RuntimeError) as exc_info:
        mod(bad_gamma)

    # Error message should mention both expected and actual K, plus
    # the diagnostic hint about init_from_hmm
    msg = str(exc_info.value)
    assert "K=3" in msg, f"Missing expected K in message: {msg}"
    assert "init_from_hmm" in msg, f"Missing diagnostic hint: {msg}"


def test_F12_wrong_dim_raises_RuntimeError():
    """Non-3D gamma (e.g., 2D [T, K]) also caught by the same RuntimeError."""
    mod = EGCPMPhaseModule(K=4, d_embed=16)
    bad_gamma = torch.zeros(48, 4)   # 2D, missing batch dim
    with pytest.raises(RuntimeError, match="K=4"):
        mod(bad_gamma)
