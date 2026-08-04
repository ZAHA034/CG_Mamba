"""M1.3 Exit Criteria — ContextGatedMambaBlock unit tests.

PLAN v2.0.8a D.6.1 canonical test cases.
Run: pytest -xvs src/tests/test_context_gated_mamba.py
"""
from __future__ import annotations

import pytest
import torch

from src.models.context_gate import ContextGatedMambaBlock
from src.utils.config import CGMambaConfig


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CGMambaConfig()


@pytest.fixture
def block(cfg):
    """Default ContextGatedMambaBlock with gate_rank=8, gate_bias_init=2.0."""
    torch.manual_seed(42)
    return ContextGatedMambaBlock(cfg)


# ───────────────────────────────────────────────────────────────────
# [SHAPE] T1
# ───────────────────────────────────────────────────────────────────

def test_shape(block, cfg):
    """[B,L,D] + ctx[B,L,D] → [B,L,D]."""
    B, L = 2, 100
    x = torch.randn(B, L, cfg.d_model)
    ctx = torch.randn(B, L, cfg.d_model)
    y = block(x, ctx)
    assert y.shape == (B, L, cfg.d_model)


# ───────────────────────────────────────────────────────────────────
# [LOGIC] T2 — disable_gate equivalence
# ───────────────────────────────────────────────────────────────────

def test_disable_gate_matches_vanilla(block, cfg):
    """context_vec=None ↔ underlying CGMambaBlock(gate=None). Bit-identical.

    L2 fix: both paths execute the identical autograd graph (vanilla branch
    returns mamba(x, gate=None) directly with no intermediate ops), so we can
    assert exact equality rather than relying on a numerical tolerance.
    """
    B, L = 2, 100
    x = torch.randn(B, L, cfg.d_model)

    y_via_wrapper = block(x, context_vec=None)
    y_via_direct = block.mamba(x, gate=None)

    assert torch.equal(y_via_wrapper, y_via_direct), \
        f"max diff = {(y_via_wrapper - y_via_direct).abs().max():.2e}"


# ───────────────────────────────────────────────────────────────────
# [GRAD] T3 — gate_proj receives gradient
# ───────────────────────────────────────────────────────────────────

def test_gate_proj_grad(block, cfg):
    """After backward, gate_proj.weight.grad must be non-zero."""
    B, L = 2, 100
    x = torch.randn(B, L, cfg.d_model)
    ctx = torch.randn(B, L, cfg.d_model)
    y = block(x, ctx)
    loss = y.pow(2).mean()           # MSE-like loss for richer gradient
    loss.backward()

    for name, p in block.gate_proj.named_parameters():
        assert p.grad is not None, f"gate_proj.{name} has None grad"
        assert p.grad.abs().sum() > 0, f"gate_proj.{name} grad is all-zero"


# ───────────────────────────────────────────────────────────────────
# [GRAD] T4 — end-to-end gradient to context_vec input
# ───────────────────────────────────────────────────────────────────

def test_gradient_to_context(block, cfg):
    """loss → gate → context_vec input must carry gradient."""
    B, L = 2, 100
    x = torch.randn(B, L, cfg.d_model)
    ctx = torch.randn(B, L, cfg.d_model, requires_grad=True)
    y = block(x, ctx)
    y.sum().backward()

    assert ctx.grad is not None, "context_vec.grad is None"
    assert ctx.grad.abs().sum() > 0, "context_vec.grad is all-zero"


# ───────────────────────────────────────────────────────────────────
# [LOGIC] T5 — sigmoid output range
# ───────────────────────────────────────────────────────────────────

def test_sigmoid_range(block, cfg):
    """gate output ∈ [0, 1]."""
    B, L = 2, 100
    ctx = torch.randn(B, L, cfg.d_model)
    with torch.no_grad():
        gate = torch.sigmoid(block.gate_proj(ctx))
    assert gate.min() >= 0.0
    assert gate.max() <= 1.0


# ───────────────────────────────────────────────────────────────────
# [LOGIC] T6 — initial gate near 0.88 (v2.0.7 A-3)
# ───────────────────────────────────────────────────────────────────

def test_initial_gate_near_identity(block, cfg):
    """At init: gate ≈ sigmoid(gate_bias_init=2.0) ≈ 0.8808."""
    B, L = 2, 100
    ctx = torch.randn(B, L, cfg.d_model)
    with torch.no_grad():
        gate = torch.sigmoid(block.gate_proj(ctx))

    expected = torch.sigmoid(torch.tensor(2.0)).item()  # 0.8808
    mean_gate = gate.mean().item()
    assert abs(mean_gate - expected) < 0.05, \
        f"initial gate mean={mean_gate:.4f} not near {expected:.4f} (tol 0.05)"


# ───────────────────────────────────────────────────────────────────
# [EDGE] T7 — zero context → near-pure sigmoid(bias), no NaN
# ───────────────────────────────────────────────────────────────────

def test_zero_context_no_nan(block, cfg):
    """context=0 → gate = sigmoid(b1 path → b2 ≈ bias_init). No NaN/Inf."""
    B, L = 2, 100
    x = torch.randn(B, L, cfg.d_model)
    ctx = torch.zeros(B, L, cfg.d_model)
    y = block(x, ctx)
    assert not torch.isnan(y).any(), "NaN in output with context=0"
    assert not torch.isinf(y).any(), "Inf in output with context=0"

    with torch.no_grad():
        gate = torch.sigmoid(block.gate_proj(ctx))
    expected = torch.sigmoid(torch.tensor(2.0)).item()
    # With context=0, gate converges more tightly to sigmoid(bias) (no W1 contribution)
    assert abs(gate.mean().item() - expected) < 0.02, \
        f"context=0 gate mean={gate.mean().item():.4f} should be very near {expected:.4f}"


# ───────────────────────────────────────────────────────────────────
# [EDGE] T8 — long sequence stability (L=2048)
# ───────────────────────────────────────────────────────────────────

def test_long_sequence(block, cfg):
    """L=2048 doesn't OOM or produce NaN."""
    B, L = 1, 2048
    x = torch.randn(B, L, cfg.d_model)
    ctx = torch.randn(B, L, cfg.d_model)
    y = block(x, ctx)
    assert y.shape == (B, L, cfg.d_model)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


# ───────────────────────────────────────────────────────────────────
# [PARAM] T9 — parameter count check (M1.2 vanilla 35,712 + gate 1,672)
# ───────────────────────────────────────────────────────────────────

def test_param_count(block):
    """gate_proj=1,672, vanilla=35,712, total=37,384."""
    gate_params = sum(p.numel() for p in block.gate_proj.parameters())
    vanilla_params = sum(p.numel() for p in block.mamba.parameters())
    total = sum(p.numel() for p in block.parameters())

    assert gate_params == 1_672, f"gate_proj: expected 1,672, got {gate_params}"
    assert vanilla_params == 35_712, \
        f"M1.2 vanilla mismatch (CGMambaBlock changed?): expected 35,712, got {vanilla_params}"
    assert total == 37_384, f"total: expected 37,384, got {total}"


# ───────────────────────────────────────────────────────────────────
# [PARAM] T10 — ablation rank sweep params (PLAN §7.4 A1 per-layer)
# ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gate_rank,expected", [
    (4, 900),       # 64*4+4 + 4*128+128 = 260 + 640
    (8, 1_672),     # default
    (16, 3_216),    # 64*16+16 + 16*128+128 = 1,040 + 2,176
    (128, 8_320),   # full-rank: single Linear(64, 128) = 64*128+128
])
def test_ablation_rank_params(cfg, gate_rank, expected):
    """A1 ablation per-layer params match PLAN §7.4."""
    block = ContextGatedMambaBlock(cfg, gate_rank=gate_rank)
    actual = sum(p.numel() for p in block.gate_proj.parameters())
    assert actual == expected, \
        f"gate_rank={gate_rank}: expected {expected}, got {actual}"


# ───────────────────────────────────────────────────────────────────
# [LOGIC] T11 — _last_gate monitoring cache
# ───────────────────────────────────────────────────────────────────

def test_disable_gate_flag(cfg):
    """F7: PLAN D.4.1 `disable_gate` constructor flag works as runtime-equivalent
    to context_vec=None. Both paths must be bit-identical to underlying
    CGMambaBlock(gate=None)."""
    torch.manual_seed(42)
    block_flag = ContextGatedMambaBlock(cfg, disable_gate=True)
    torch.manual_seed(42)
    block_normal = ContextGatedMambaBlock(cfg, disable_gate=False)

    B, L = 2, 100
    x = torch.randn(B, L, cfg.d_model)
    ctx = torch.randn(B, L, cfg.d_model)

    # disable_gate=True ignores context_vec (even when provided)
    y_flag_with_ctx = block_flag(x, ctx)
    y_flag_no_ctx = block_flag(x, None)
    assert torch.equal(y_flag_with_ctx, y_flag_no_ctx), \
        "disable_gate=True should ignore context_vec"

    # disable_gate=False + ctx=None must match disable_gate=True path
    y_normal_no_ctx = block_normal(x, None)
    # Two blocks have different gate_proj inits (different seed), but vanilla path
    # is gate_proj-free, so as long as in_proj weights are same seed=42, results match
    assert torch.equal(y_flag_no_ctx, y_normal_no_ctx), \
        "disable_gate=True ↔ context_vec=None should produce identical output"


def test_last_gate_cache(block, cfg):
    """_last_gate cached after forward in TRAIN mode; skipped in eval (L4 guard)."""
    B, L = 2, 100
    x = torch.randn(B, L, cfg.d_model)
    ctx = torch.randn(B, L, cfg.d_model)

    # Default state: training=True (nn.Module default)
    assert block.training

    # Train mode + vanilla path → no cache
    block(x, context_vec=None)
    assert block._last_gate is None

    # Train mode + with context → cache populated, detached
    block(x, ctx)
    assert block._last_gate is not None
    assert block._last_gate.shape == (B, L, cfg.d_inner)
    assert not block._last_gate.requires_grad

    # Eval mode + with context → cache skipped (L4 fix: avoid stale state
    # + memory persistence during inference / W&B-disabled runs)
    block.eval()
    block(x, ctx)
    assert block._last_gate is None
    block.train()
