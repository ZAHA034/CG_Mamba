"""M1.5 Exit Criteria — EnvModule unit tests.

PLAN v2.0.8a §3.3 + v2.0.7 A-2 canonical test matrix.
Run: pytest -xvs src/tests/test_env_module.py

Test matrix:
    [SHAPE]   T1  — forward: [B,L,2] → [B,L,D]
    [SHAPE]   T2  — decode:  [B,L,D] → [B,L,2]
    [GRAD]    T3  — loss → encoder params gradient
    [GRAD]    T4  — loss → env input gradient (future joint training)
    [GRAD]    T5  — recon_loss → decoder params gradient
    [GRAD]    T6  — context_vec chain: gate_env → gate_phase gradient via ⊙
    [INIT]    T7  — encoder output scale at init (normal, NOT near-zero)
    [PARAM]   T8  — encoder/decoder param counts match budget
    [RECON]   T9  — reconstruction loss decreases with training
    [CACHE]   T10 — _last_env cache (train vs eval mode)
    [EDGE]    T11 — zero input: no NaN/Inf
    [EDGE]    T12 — large input: no NaN/Inf
    [EDGE]    T13 — long sequence (L=2048): shape + no NaN
    [VALID]   T14 — config validation rejects bad configs
    [COMPAT]  T15 — output compatible with PhaseModule ⊙ product
    [API]     T16 — reconstruction_loss convenience method consistency
    [SHAPE]   T17 — input validation (3D enforcement + dim match)
    [STAGE2]  T18 — freeze_decoder_for_stage2: decoder frozen, encoder trainable
    [STAGE2]  T19 — encoder_parameters() / decoder_parameters() iterator helpers
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.models.env_module import EnvModule
from src.utils.config import CGMambaConfig


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CGMambaConfig()


@pytest.fixture
def env_module(cfg):
    torch.manual_seed(42)
    return EnvModule(cfg)


@pytest.fixture
def sample_env():
    """Simulated z-scored env batch: [B=2, L=104, V_env=2]."""
    torch.manual_seed(0)
    return torch.randn(2, 104, 2)


# ───────────────────────────────────────────────────────────────────
# [SHAPE] T1 — forward shape
# ───────────────────────────────────────────────────────────────────

def test_forward_shape(env_module, cfg, sample_env):
    """env [B, L, 2] → gate_env [B, L, D=64]."""
    gate_env = env_module(sample_env)
    B, L = sample_env.shape[:2]
    assert gate_env.shape == (B, L, cfg.d_model), \
        f"Expected ({B}, {L}, {cfg.d_model}), got {tuple(gate_env.shape)}"


# ───────────────────────────────────────────────────────────────────
# [SHAPE] T2 — decode shape
# ───────────────────────────────────────────────────────────────────

def test_decode_shape(env_module, cfg, sample_env):
    """gate_env [B, L, D] → env_recon [B, L, 2]."""
    gate_env = env_module(sample_env)
    env_recon = env_module.decode(gate_env)
    B, L = sample_env.shape[:2]
    assert env_recon.shape == (B, L, cfg.env_input_dim), \
        f"Expected ({B}, {L}, {cfg.env_input_dim}), got {tuple(env_recon.shape)}"


# ───────────────────────────────────────────────────────────────────
# [GRAD] T3 — gradient flows to encoder params
# ───────────────────────────────────────────────────────────────────

def test_encoder_gradient(env_module, sample_env):
    """loss → gate_env → encoder must carry gradient."""
    gate_env = env_module(sample_env)
    loss = gate_env.pow(2).mean()
    loss.backward()

    for name, p in env_module.encoder.named_parameters():
        assert p.grad is not None, f"encoder.{name} has None grad"
        assert p.grad.abs().sum() > 0, f"encoder.{name} grad is all-zero"


# ───────────────────────────────────────────────────────────────────
# [GRAD] T4 — gradient flows to env input
# ───────────────────────────────────────────────────────────────────

def test_gradient_to_env_input(env_module):
    """loss → gate_env → env input must carry gradient (future joint training)."""
    env = torch.randn(2, 104, 2, requires_grad=True)
    gate_env = env_module(env)
    gate_env.sum().backward()

    assert env.grad is not None, "env.grad is None"
    assert env.grad.abs().sum() > 0, "env.grad is all-zero"


# ───────────────────────────────────────────────────────────────────
# [GRAD] T5 — gradient flows through decoder
# ───────────────────────────────────────────────────────────────────

def test_decoder_gradient(env_module, sample_env):
    """recon_loss → decoder params must carry gradient."""
    gate_env = env_module(sample_env)
    recon_loss = env_module.reconstruction_loss(sample_env, gate_env)
    recon_loss.backward()

    for name, p in env_module.decoder.named_parameters():
        assert p.grad is not None, f"decoder.{name} has None grad"
        assert p.grad.abs().sum() > 0, f"decoder.{name} grad is all-zero"


# ───────────────────────────────────────────────────────────────────
# [GRAD] T6 — context_vec chain: gate_env → gate_phase gradient
# ───────────────────────────────────────────────────────────────────

def test_context_vec_gradient_chain(env_module, cfg):
    """Simulates context_vec = gate_phase ⊙ gate_env.

    Gradient must flow from loss through ⊙ to both branches:
      ∂L/∂gate_phase via gate_env (∂(a*b)/∂a = b)
      ∂L/∂encoder via gate_phase (∂(a*b)/∂b = a)
    """
    B, L = 2, 104
    env = torch.randn(B, L, 2)
    # Simulate PhaseModule output (requires_grad for test)
    gate_phase = torch.randn(B, L, cfg.d_model, requires_grad=True)

    gate_env = env_module(env)
    context_vec = gate_phase * gate_env   # element-wise ⊙

    loss = context_vec.pow(2).mean()
    loss.backward()

    # gate_phase gets gradient through gate_env
    assert gate_phase.grad is not None, "gate_phase.grad is None"
    assert gate_phase.grad.abs().sum() > 0, "gate_phase.grad is all-zero"

    # encoder gets gradient through gate_phase
    enc_grads = [p.grad for p in env_module.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0, "No encoder params received gradient through ⊙ chain"
    assert all(g.abs().sum() > 0 for g in enc_grads), \
        "Some encoder param grad is all-zero — ⊙ chain may be broken"


# ───────────────────────────────────────────────────────────────────
# [INIT] T7 — encoder output scale (normal init, NOT near-zero)
# ───────────────────────────────────────────────────────────────────

def test_init_output_scale(env_module, sample_env):
    """At init, gate_env should have O(0.1-1) scale (standard PyTorch init).

    v2.0.9 update: PhaseModule.state_embeddings (S-5 rename) is now zeros init
    (R-4), giving gate_phase = sigmoid(0) = 0.5 exactly at init. If gate_env
    were near-zero, context_vec = 0.5 · gate_env(~0) ≈ 0 — no contextual
    signal, and ∂L/∂state_embeddings (chain rule via gate_env) collapses.
    Normal-init gate_env ensures gate_env ~ O(1) → context_vec ~ O(1) →
    healthy gradient signal for state_embeddings during Stage 2/3 training.
    """
    with torch.no_grad():
        gate_env = env_module(sample_env)

    mean_abs = gate_env.abs().mean().item()
    # Standard init: O(0.1-1). NOT near-zero (< 0.05); v2.0.9 PhaseModule
    # gives a uniform 0.5 gate_phase, so gate_env must carry the contextual signal.
    assert mean_abs > 0.05, \
        f"gate_env mean|val|={mean_abs:.4f} too small — should be O(0.1-1) with normal init"
    # Not exploding either
    assert mean_abs < 10.0, \
        f"gate_env mean|val|={mean_abs:.4f} too large — possible init issue"


# ───────────────────────────────────────────────────────────────────
# [PARAM] T8 — param count matches budget
# ───────────────────────────────────────────────────────────────────

def test_param_count(env_module, cfg):
    """Encoder: (V*H+H) + (H*D+D) = 2,208. Decoder: (D*H+H) + (H*V+V) = 2,146."""
    V = cfg.env_input_dim    # 2
    H = cfg.env_hidden_dim   # 32
    D = cfg.d_model          # 64

    expected_encoder = (V * H + H) + (H * D + D)    # 96 + 2112 = 2,208
    expected_decoder = (D * H + H) + (H * V + V)    # 2080 + 66 = 2,146
    expected_total = expected_encoder + expected_decoder

    assert env_module.encoder_param_count() == expected_encoder, \
        f"Encoder: expected {expected_encoder}, got {env_module.encoder_param_count()}"
    assert env_module.decoder_param_count() == expected_decoder, \
        f"Decoder: expected {expected_decoder}, got {env_module.decoder_param_count()}"
    actual_total = sum(p.numel() for p in env_module.parameters())
    assert actual_total == expected_total, \
        f"Total: expected {expected_total}, got {actual_total}"


# ───────────────────────────────────────────────────────────────────
# [RECON] T9 — reconstruction loss decreases
# ───────────────────────────────────────────────────────────────────

def test_reconstruction_loss_decreases(cfg):
    """30 SGD steps should reduce reconstruction loss by ≥ 50%."""
    torch.manual_seed(42)
    module = EnvModule(cfg)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

    # Fixed mini-batch
    env = torch.randn(4, 50, 2)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        gate_env = module(env)
        loss = module.reconstruction_loss(env, gate_env)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, \
        f"Loss didn't decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}"


# ───────────────────────────────────────────────────────────────────
# [CACHE] T10 — _last_env cache behavior
# ───────────────────────────────────────────────────────────────────

def test_last_env_cache(env_module, cfg, sample_env):
    """_last_env populated in train mode (detached), None in eval mode."""
    # Train mode → cache populated
    env_module.train()
    env_module(sample_env)
    assert env_module._last_env is not None, \
        "_last_env should be populated in train mode"
    assert env_module._last_env.shape == (2, 104, cfg.d_model), \
        f"Unexpected cache shape: {tuple(env_module._last_env.shape)}"
    assert not env_module._last_env.requires_grad, \
        "_last_env should be detached"

    # Eval mode → cache is None
    env_module.eval()
    env_module(sample_env)
    assert env_module._last_env is None, \
        "_last_env should be None in eval mode"


# ───────────────────────────────────────────────────────────────────
# [EDGE] T11 — zero input safety
# ───────────────────────────────────────────────────────────────────

def test_zero_input_no_nan(env_module, cfg):
    """env = 0 → no NaN/Inf in gate_env or reconstruction."""
    env = torch.zeros(2, 104, cfg.env_input_dim)
    gate_env = env_module(env)
    env_recon = env_module.decode(gate_env)

    assert not torch.isnan(gate_env).any(), "NaN in gate_env with zero input"
    assert not torch.isinf(gate_env).any(), "Inf in gate_env with zero input"
    assert not torch.isnan(env_recon).any(), "NaN in env_recon with zero input"
    assert not torch.isinf(env_recon).any(), "Inf in env_recon with zero input"


# ───────────────────────────────────────────────────────────────────
# [EDGE] T12 — large input safety
# ───────────────────────────────────────────────────────────────────

def test_large_input_no_nan(env_module, cfg):
    """Large env values (±10 std) → no NaN/Inf."""
    env = torch.randn(2, 104, cfg.env_input_dim) * 10.0
    gate_env = env_module(env)
    env_recon = env_module.decode(gate_env)

    assert not torch.isnan(gate_env).any(), "NaN in gate_env with large input"
    assert not torch.isinf(gate_env).any(), "Inf in gate_env with large input"
    assert not torch.isnan(env_recon).any(), "NaN in env_recon with large input"


# ───────────────────────────────────────────────────────────────────
# [EDGE] T13 — long sequence stability
# ───────────────────────────────────────────────────────────────────

def test_long_sequence(env_module, cfg):
    """L=2048 → no OOM, correct shape, no NaN."""
    B, L = 1, 2048
    env = torch.randn(B, L, cfg.env_input_dim)
    gate_env = env_module(env)

    assert gate_env.shape == (B, L, cfg.d_model)
    assert not torch.isnan(gate_env).any()
    assert not torch.isinf(gate_env).any()


# ───────────────────────────────────────────────────────────────────
# [VALID] T14 — config validation
# ───────────────────────────────────────────────────────────────────

def test_config_validation_rejects_bad():
    """Bad config values should raise AssertionError at construction.

    Three failure modes:
      1) env_input_dim < 1 (degenerate input)
      2) env_hidden_dim < env_input_dim (sub-input bottleneck)
      3) d_model < env_hidden_dim (inverted bottleneck — encoder hidden > output)
    """
    base_cfg = CGMambaConfig()

    # env_input_dim = 0 (degenerate)
    with pytest.raises(AssertionError, match="env_input_dim"):
        bad_cfg = replace(base_cfg, env_input_dim=0)
        EnvModule(bad_cfg)

    # env_hidden_dim < env_input_dim
    with pytest.raises(AssertionError, match="env_hidden_dim"):
        bad_cfg = replace(base_cfg, env_hidden_dim=1)  # 1 < 2
        EnvModule(bad_cfg)

    # d_model < env_hidden_dim (inverted bottleneck)
    with pytest.raises(AssertionError, match="d_model"):
        bad_cfg = replace(base_cfg, d_model=16, env_hidden_dim=32)  # 16 < 32
        EnvModule(bad_cfg)


# ───────────────────────────────────────────────────────────────────
# [COMPAT] T15 — compatible with PhaseModule ⊙ product
# ───────────────────────────────────────────────────────────────────

def test_phase_module_compatibility(env_module, cfg):
    """gate_env [B,L,D] ⊙ gate_phase [B,L,D] → context_vec [B,L,D].

    context_vec must be valid input for ContextGatedMambaBlock (dim D=64).
    v2.0.9 update: PhaseModule outputs gate_phase = sigmoid(phase_post @
    state_embeddings) with state_embeddings = zeros at init → gate_phase = 0.5
    exact. context_vec = 0.5 · gate_env is then driven by gate_env's normal
    init (no longer "driven by near-zero phase"), and stays bounded ~ O(0.5).
    """
    B, L = 2, 104
    env = torch.randn(B, L, cfg.env_input_dim)

    # Simulate v2.0.9 PhaseModule output: gate_phase = sigmoid(0) = 0.5 exact
    gate_phase = torch.full((B, L, cfg.d_model), 0.5)

    gate_env = env_module(env)
    context_vec = gate_phase * gate_env

    assert context_vec.shape == (B, L, cfg.d_model), \
        f"context_vec shape {tuple(context_vec.shape)} != (B, L, D={cfg.d_model})"

    # At init: gate_phase = 0.5 exact, gate_env ~ O(0.1-1) → ctx ~ O(0.05-0.5)
    with torch.no_grad():
        mean_ctx = context_vec.abs().mean().item()
    assert mean_ctx < 1.0, \
        f"context_vec too large at init: {mean_ctx:.4f} — " \
        f"uniform 0.5 gate_phase should keep context_vec bounded by ~0.5·|gate_env|"


# ───────────────────────────────────────────────────────────────────
# [API] T16 — reconstruction_loss convenience consistency
# ───────────────────────────────────────────────────────────────────

def test_reconstruction_loss_consistency(env_module, sample_env):
    """reconstruction_loss(env, gate_env) == reconstruction_loss(env, None).

    Both paths must produce identical loss:
      Path 1: caller pre-computes gate_env, passes it in
      Path 2: caller passes gate_env=None, method runs forward() internally
    """
    # Path 1: explicit forward + pass gate_env
    gate_env = env_module(sample_env)
    loss_explicit = env_module.reconstruction_loss(sample_env, gate_env)

    # Path 2: let reconstruction_loss call forward() internally
    loss_internal = env_module.reconstruction_loss(sample_env, gate_env=None)

    assert torch.allclose(loss_explicit, loss_internal, atol=1e-7), \
        f"Loss mismatch: explicit={loss_explicit.item():.6f} vs " \
        f"internal={loss_internal.item():.6f}"


# ───────────────────────────────────────────────────────────────────
# [SHAPE] T17 — input validation
# ───────────────────────────────────────────────────────────────────

def test_input_validation(env_module, cfg):
    """Wrong input shape should raise ValueError for forward AND decode."""
    # ── forward() validation ──
    # 2D input (missing batch dim)
    with pytest.raises(ValueError, match="3D"):
        env_module(torch.randn(104, 2))

    # Wrong feature dim
    with pytest.raises(ValueError, match="env_input_dim"):
        env_module(torch.randn(2, 104, 3))

    # ── decode() validation ──
    # 2D input
    with pytest.raises(ValueError, match="3D"):
        env_module.decode(torch.randn(104, cfg.d_model))

    # Wrong feature dim (should be d_model=64, not 32)
    with pytest.raises(ValueError, match="d_model"):
        env_module.decode(torch.randn(2, 104, 32))


# ───────────────────────────────────────────────────────────────────
# [STAGE2] T18 — freeze_decoder_for_stage2 (Major G fix)
# ───────────────────────────────────────────────────────────────────

def test_freeze_decoder_for_stage2(env_module, cfg, sample_env):
    """freeze_decoder_for_stage2(): decoder fully frozen, encoder unchanged.

    Mirrors hmm_stage1.freeze_hmm_for_stage2() pattern:
      1) returns "newly frozen" count (= decoder param count on first call)
      2) post-condition: all decoder params requires_grad=False
      3) post-condition: encoder remains trainable
      4) idempotent: second call returns 0 (no new freezes)
      5) functional: decoder still callable but produces no gradient
    """
    # Before freeze: all params trainable
    enc_trainable_before = sum(1 for p in env_module.encoder.parameters() if p.requires_grad)
    dec_trainable_before = sum(1 for p in env_module.decoder.parameters() if p.requires_grad)
    assert enc_trainable_before > 0, "Pre-condition: encoder must be trainable initially"
    assert dec_trainable_before > 0, "Pre-condition: decoder must be trainable initially"

    # Call freeze
    n_frozen = env_module.freeze_decoder_for_stage2()

    # Post-condition 1 (C-1 fix, v2.1.6): returned count is `numel sum` of
    # newly frozen scalar params, NOT param-tensor count. Linear weight + bias
    # are 2 tensors but together contribute (D·V + V) scalars. Docstring
    # explicitly says "typically 2,146 on first invocation".
    n_decoder_numel = sum(p.numel() for p in env_module.decoder.parameters())
    assert n_frozen == n_decoder_numel, (
        f"freeze returned {n_frozen}, expected {n_decoder_numel} (= decoder numel sum)"
    )

    # Post-condition 2: ALL decoder params now frozen
    dec_trainable_after = sum(1 for p in env_module.decoder.parameters() if p.requires_grad)
    assert dec_trainable_after == 0, \
        f"Expected 0 trainable decoder params, got {dec_trainable_after}"

    # Post-condition 3: encoder unchanged
    enc_trainable_after = sum(1 for p in env_module.encoder.parameters() if p.requires_grad)
    assert enc_trainable_after == enc_trainable_before, \
        f"Encoder trainable changed: {enc_trainable_before} → {enc_trainable_after}"

    # Idempotent: second call returns 0
    n_frozen_2 = env_module.freeze_decoder_for_stage2()
    assert n_frozen_2 == 0, \
        f"Second call should freeze 0 new params, got {n_frozen_2}"

    # Functional: decoder still callable (forward works), but no gradient
    gate_env = env_module(sample_env)
    env_recon = env_module.decode(gate_env)  # decode works
    assert env_recon.shape == (2, 104, cfg.env_input_dim)

    # If we tried to backward through decoder-only, decoder params would NOT
    # accumulate gradient (since they are frozen). We verify this:
    recon_loss = env_module.reconstruction_loss(sample_env, gate_env)
    recon_loss.backward()
    for name, p in env_module.decoder.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, \
            f"Frozen decoder.{name} accumulated gradient {p.grad}"

    # Encoder still receives gradient (through recon path: gate_env→decoder→recon)
    enc_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in env_module.encoder.parameters()
    )
    assert enc_has_grad, "Encoder must receive gradient even when decoder is frozen"


# ───────────────────────────────────────────────────────────────────
# [STAGE2] T19 — parameter iterator helpers (Medium A fix)
# ───────────────────────────────────────────────────────────────────

def test_encoder_decoder_parameters_iterators(env_module, cfg):
    """encoder_parameters() / decoder_parameters() return correct iterables.

    M1.7 optimizer group construction에서 사용:
        optimizer_groups = [
            {"name": "env_encoder",
             "params": list(env_module.encoder_parameters()), "lr": 1e-4},
        ]

    Properties verified:
      1) encoder_parameters() yields ONLY encoder params (not decoder)
      2) decoder_parameters() yields ONLY decoder params (not encoder)
      3) Counts match encoder_param_count() / decoder_param_count()
      4) iterables are independent (set non-overlap)
      5) id() check: returned param objects ARE the actual encoder/decoder params
    """
    enc_params = list(env_module.encoder_parameters())
    dec_params = list(env_module.decoder_parameters())

    # Counts match the existing count helpers
    enc_count_from_iter = sum(p.numel() for p in enc_params)
    dec_count_from_iter = sum(p.numel() for p in dec_params)
    assert enc_count_from_iter == env_module.encoder_param_count(), \
        f"encoder_parameters count mismatch: {enc_count_from_iter} vs " \
        f"{env_module.encoder_param_count()}"
    assert dec_count_from_iter == env_module.decoder_param_count(), \
        f"decoder_parameters count mismatch: {dec_count_from_iter} vs " \
        f"{env_module.decoder_param_count()}"

    # Non-overlapping (encoder ∩ decoder == ∅) using id() identity
    enc_ids = {id(p) for p in enc_params}
    dec_ids = {id(p) for p in dec_params}
    assert enc_ids.isdisjoint(dec_ids), \
        "encoder_parameters() and decoder_parameters() share param objects"

    # Identity: returned objects ARE the actual encoder/decoder params
    actual_enc_ids = {id(p) for p in env_module.encoder.parameters()}
    actual_dec_ids = {id(p) for p in env_module.decoder.parameters()}
    assert enc_ids == actual_enc_ids, "encoder_parameters() returns wrong objects"
    assert dec_ids == actual_dec_ids, "decoder_parameters() returns wrong objects"

    # Union == all EnvModule params (no other params exist)
    all_module_ids = {id(p) for p in env_module.parameters()}
    assert enc_ids | dec_ids == all_module_ids, \
        "encoder ∪ decoder ≠ all EnvModule params — unaccounted params exist"
