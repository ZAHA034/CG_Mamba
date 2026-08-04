"""Unit tests for `src/models/entropy_decoder.py` — EntropyAwareDecoder (M1.6, v2.1.1).

PLAN v2.0.9 PATCH 12 (D.6 Tests) — v2.1.1 refactor: pure tensor I/O (A-1).

Scope (10 tests after A-1 refactor):
  1. Constructor + M-3 fix (proj dim = len(horizons))
  2. gate_init=-1.1 → α ≈ 0.25 init
  3. Forward shape (dense horizons)
  4. Forward shape (ragged horizons (1,2,4,8) — M-3 verification)
  5. LOGIC-1 confidence formula (peaked vs uniform γ)
  6. Residual form bound (pred near last_value at init)
  7. Gradient flow (proj + gate; backprop through gamma_all + state_embeddings)
  8. Shape validation errors (RuntimeError on wrong dims, wrong K, insufficient rollout length)
  9. Monitoring cache (_last_eff_gate, _last_confidence) train/eval discipline
  10. Pure tensor I/O — A-1 verification (no PhaseModule reference in signature)
"""
from __future__ import annotations

import inspect

import pytest
import torch

from src.models.entropy_decoder import EntropyAwareDecoder
from src.utils.config import CGMambaConfig


# ─────────────────────────────────────────────────────────────────
# Fixtures (pure tensors — A-1: no PhaseModule needed)
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CGMambaConfig()


@pytest.fixture
def decoder(cfg):
    return EntropyAwareDecoder(
        d_model=cfg.d_model, horizons=cfg.horizons, K=cfg.K_phase, gate_init=-1.1,
    )


@pytest.fixture
def tensor_inputs(cfg):
    """Pure-tensor inputs for decoder.forward() (A-1: no PhaseModule fixture)."""
    B, L_minus1 = 2, 100
    torch.manual_seed(0)
    return {
        "encoder_out": torch.randn(B, L_minus1, cfg.d_model),
        "last_value_normalized": torch.randn(B),
        "gamma_all": torch.softmax(torch.randn(B, max(cfg.horizons), cfg.K_phase), dim=-1),
        "state_embeddings": torch.zeros(cfg.K_phase, cfg.d_model),  # mirror R-4 zeros init
    }


# ─────────────────────────────────────────────────────────────────
# T1 — Constructor + M-3 fix: proj dim = len(horizons)
# ─────────────────────────────────────────────────────────────────

def test_constructor_proj_dim_equals_len_horizons(cfg):
    """M-3 fix: proj.out_features must equal len(horizons), not max(horizons)."""
    # Dense grid (1,2,3,4) — len == max, both 4
    dec = EntropyAwareDecoder(d_model=cfg.d_model, horizons=(1, 2, 3, 4), K=cfg.K_phase)
    assert dec.proj.out_features == 4
    assert dec.max_horizon == 4

    # Ragged grid (1,2,4,8) — len=4, max=8. M-3 fix: proj uses len.
    dec_ragged = EntropyAwareDecoder(d_model=cfg.d_model, horizons=(1, 2, 4, 8), K=cfg.K_phase)
    assert dec_ragged.proj.out_features == 4, \
        f"M-3 fix violated: proj.out_features={dec_ragged.proj.out_features} (expected 4=len)"
    assert dec_ragged.max_horizon == 8


# ─────────────────────────────────────────────────────────────────
# T2 — gate_init=-1.1 → sigmoid(-1.1) ≈ 0.25
# ─────────────────────────────────────────────────────────────────

def test_gate_init_alpha_quarter(decoder):
    """gate_init=-1.1 → α = sigmoid(-1.1) ≈ 0.2507 (soft correction start)."""
    alpha = torch.sigmoid(decoder.gate).item()
    assert abs(alpha - 0.2507) < 0.001, f"alpha={alpha:.4f} (expected ≈ 0.2507)"


# ─────────────────────────────────────────────────────────────────
# T3 — Forward shape (dense horizons)
# ─────────────────────────────────────────────────────────────────

def test_forward_shape_dense(decoder, tensor_inputs, cfg):
    """forward returns [B, len(horizons)] with dense horizons (1,2,3,4)."""
    decoder.eval()
    preds = decoder(**tensor_inputs)
    B = tensor_inputs["encoder_out"].shape[0]
    assert preds.shape == (B, len(cfg.horizons))


# ─────────────────────────────────────────────────────────────────
# T4 — Forward shape (ragged horizons — M-3 verification)
# ─────────────────────────────────────────────────────────────────

def test_forward_shape_ragged(cfg):
    """forward returns [B, len(horizons)] for ragged grid (1,2,4,8)."""
    horizons_ragged = (1, 2, 4, 8)
    dec = EntropyAwareDecoder(
        d_model=cfg.d_model, horizons=horizons_ragged, K=cfg.K_phase,
    )
    B = 3
    inputs = {
        "encoder_out": torch.randn(B, 80, cfg.d_model),
        "last_value_normalized": torch.randn(B),
        "gamma_all": torch.softmax(torch.randn(B, max(horizons_ragged), cfg.K_phase), dim=-1),
        "state_embeddings": torch.zeros(cfg.K_phase, cfg.d_model),
    }
    dec.eval()
    preds = dec(**inputs)
    assert preds.shape == (B, len(horizons_ragged)) == (B, 4)


# ─────────────────────────────────────────────────────────────────
# T5 — LOGIC-1: confidence formula
# ─────────────────────────────────────────────────────────────────

def test_confidence_peaked_and_uniform(cfg):
    """_compute_confidence: peaked γ → ≈ 1.0, uniform γ → 0.0."""
    K = cfg.K_phase
    B = 4
    gamma_peaked = torch.zeros(B, K)
    gamma_peaked[:, 0] = 1.0
    conf_peaked = EntropyAwareDecoder._compute_confidence(gamma_peaked, K)
    assert conf_peaked.mean().item() > 0.99, f"peaked conf={conf_peaked.mean().item()}"

    gamma_uniform = torch.full((B, K), 1.0 / K)
    conf_uniform = EntropyAwareDecoder._compute_confidence(gamma_uniform, K)
    assert conf_uniform.mean().item() < 0.01, f"uniform conf={conf_uniform.mean().item()}"


# ─────────────────────────────────────────────────────────────────
# T6 — Residual form bound (pred near last_value at init)
# ─────────────────────────────────────────────────────────────────

def test_residual_form(decoder, cfg):
    """At init (α ≈ 0.25, state_embeddings=0), |pred - last_value| stays bounded."""
    B = 4
    inputs = {
        "encoder_out": torch.randn(B, 50, cfg.d_model),
        "last_value_normalized": torch.full((B,), 5.0),
        "gamma_all": torch.softmax(torch.randn(B, max(cfg.horizons), cfg.K_phase), dim=-1),
        "state_embeddings": torch.zeros(cfg.K_phase, cfg.d_model),
    }
    decoder.eval()
    preds = decoder(**inputs)
    max_dev = (preds - 5.0).abs().max().item()
    # α ≈ 0.25, correction ~ O(0.1-1), eff_gate ∈ [0.5, 1.0] → |dev| < 5
    assert max_dev < 5.0, f"residual deviation too large: {max_dev}"


# ─────────────────────────────────────────────────────────────────
# T7 — Gradient flow (proj + gate, backprop through gamma_all + state_embeddings)
# ─────────────────────────────────────────────────────────────────

def test_gradient_flow(decoder, cfg):
    """Backward: decoder.proj.grad, decoder.gate.grad nonzero.
    Also backprop through gamma_all and state_embeddings (caller-provided tensors)."""
    B = 2
    gamma_all = torch.softmax(torch.randn(B, max(cfg.horizons), cfg.K_phase), dim=-1).requires_grad_(True)
    state_emb = torch.zeros(cfg.K_phase, cfg.d_model, requires_grad=True)
    encoder_out = torch.randn(B, 50, cfg.d_model, requires_grad=True)
    last_value = torch.randn(B)

    decoder.train()
    decoder.zero_grad()
    preds = decoder(encoder_out, last_value, gamma_all, state_emb)
    preds.sum().backward()

    assert decoder.proj.weight.grad is not None and decoder.proj.weight.grad.abs().sum() > 0
    assert decoder.gate.grad is not None and decoder.gate.grad.abs() > 0
    # A-1: gradient flows through caller-provided tensors too
    assert gamma_all.grad is not None and gamma_all.grad.abs().sum() > 0
    assert state_emb.grad is not None and state_emb.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────
# T8 — Shape validation errors
# ─────────────────────────────────────────────────────────────────

def test_shape_validation_errors(decoder, cfg):
    """RuntimeError on wrong encoder_out dim, wrong gamma_all K, insufficient rollout length."""
    B = 2

    # Wrong encoder_out last dim
    bad_enc = torch.randn(B, 50, cfg.d_model + 16)
    with pytest.raises(RuntimeError, match="encoder_out last dim"):
        decoder(
            bad_enc, torch.randn(B),
            torch.softmax(torch.randn(B, 4, cfg.K_phase), dim=-1),
            torch.zeros(cfg.K_phase, cfg.d_model),
        )

    # Wrong gamma_all K
    with pytest.raises(RuntimeError, match="gamma_all expected"):
        decoder(
            torch.randn(B, 50, cfg.d_model), torch.randn(B),
            torch.softmax(torch.randn(B, 4, cfg.K_phase + 1), dim=-1),
            torch.zeros(cfg.K_phase, cfg.d_model),
        )

    # Insufficient rollout length (gamma_all.shape[1] < max_horizon=4)
    with pytest.raises(RuntimeError, match="rollout length"):
        decoder(
            torch.randn(B, 50, cfg.d_model), torch.randn(B),
            torch.softmax(torch.randn(B, 2, cfg.K_phase), dim=-1),   # max_h=2 < 4
            torch.zeros(cfg.K_phase, cfg.d_model),
        )

    # Wrong state_embeddings shape
    with pytest.raises(RuntimeError, match="state_embeddings expected"):
        decoder(
            torch.randn(B, 50, cfg.d_model), torch.randn(B),
            torch.softmax(torch.randn(B, 4, cfg.K_phase), dim=-1),
            torch.zeros(cfg.K_phase + 1, cfg.d_model),  # wrong K
        )


# ─────────────────────────────────────────────────────────────────
# T9 — Monitoring cache: train-only, eval-None (both _last_eff_gate, _last_confidence)
# ─────────────────────────────────────────────────────────────────

def test_last_eff_gate_and_confidence_discipline(decoder, tensor_inputs, cfg):
    """_last_eff_gate + _last_confidence: cached in train mode (detached), None in eval."""
    B = tensor_inputs["encoder_out"].shape[0]

    decoder.train()
    _ = decoder(**tensor_inputs)
    assert decoder._last_eff_gate is not None
    assert decoder._last_eff_gate.shape == (B, len(cfg.horizons))
    assert not decoder._last_eff_gate.requires_grad
    assert decoder._last_confidence is not None
    assert decoder._last_confidence.shape == (B, len(cfg.horizons))

    decoder.eval()
    _ = decoder(**tensor_inputs)
    assert decoder._last_eff_gate is None
    assert decoder._last_confidence is None


# ─────────────────────────────────────────────────────────────────
# T10 — A-1: pure tensor I/O (no PhaseModule in signature)
# ─────────────────────────────────────────────────────────────────

def test_a1_pure_tensor_signature():
    """A-1 verification: EntropyAwareDecoder.forward signature contains no nn.Module reference."""
    sig = inspect.signature(EntropyAwareDecoder.forward)
    params = list(sig.parameters.values())
    # First param is `self`; remaining must be tensor-typed.
    forward_args = [p.name for p in params[1:]]
    assert forward_args == [
        "encoder_out", "last_value_normalized", "gamma_all", "state_embeddings",
    ], f"A-1 violation: forward params = {forward_args}"
    # No annotation references "PhaseModule" (it should be tensor-only)
    for p in params[1:]:
        ann = p.annotation
        if ann is not inspect.Parameter.empty:
            ann_str = str(ann)
            assert "PhaseModule" not in ann_str, \
                f"A-1 violation: parameter {p.name} annotation contains PhaseModule"
