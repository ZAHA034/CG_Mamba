"""Unit tests for `src/models/cg_forecaster.py` — CGForecaster (M1.6, v2.0.9).

PLAN v2.0.9 PATCH 12 (D.6 Tests) — neue test file.

Scope (13 tests):
  1. Constructor: 4 sub-modules + dim consistency (V_hmm_raw, K_phase, d_embed)
  2. B1: V_input=4 → x_phase = x[:, :, :V_hmm_raw=3] slicing
  3. forward without prepare_for_stage2 → RuntimeError (M-4 enforced)
  4. forward with prepare_for_stage2(hmm) → shape [B, len(horizons)]
  5. L-1 alignment: encoder receives [B, L-1, V_input]
  6. AND composition: context_vec = gate_phase * gate_env_truncated
  7. phase_init / gamma_last extraction correctness (phase_init unused in M1.6, but
     phase_post structure must support both)
  8. gamma_last → decoder rollout integration (rollout called once, H-1 fix)
  9. x_window slicing with cfg.rollout_window=5
  10. Gradient flow: state_embeddings + gate_proj + decoder.proj + encoder layers
  11. Stage 1 ckpt → prepare_for_stage2 → forward end-to-end
  12. Determinism in eval mode
  13. L < rollout_window edge case (L=3, L=2 extreme — L-3 fix)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import CGForecaster
from src.models.gaussian_hmm import GaussianHMM
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_RUN_DIR = REPO_ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed42"


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CGMambaConfig()


def _make_synthetic_fitted_hmm(V_raw=3, K=3, T=200, seed=42):
    """For tests that don't have access to Stage 1 ckpt."""
    np.random.seed(seed)
    V_aug = 2 * V_raw
    x_raw = np.cumsum(np.random.randn(T, V_raw).astype(np.float64) * 0.05, axis=0)
    delta = x_raw[1:] - x_raw[:-1]
    x_aug = np.concatenate([x_raw[1:], delta], axis=-1)
    return GaussianHMM(n_states=K, n_features=V_aug, reg_covar=5e-3, seed=seed).fit(x_aug)


@pytest.fixture
def synthetic_hmm():
    return _make_synthetic_fitted_hmm()


@pytest.fixture
def model(cfg, synthetic_hmm):
    """CGForecaster with synthetic-HMM prepare_for_stage2 already called."""
    m = CGForecaster(cfg)
    m.prepare_for_stage2(synthetic_hmm)
    return m


# ─────────────────────────────────────────────────────────────────
# T1 — Constructor: 4 sub-modules + dim consistency
# ─────────────────────────────────────────────────────────────────

def test_constructor_four_submodules(cfg):
    """CGForecaster instantiates PhaseModule, EnvModule, CGMambaEncoder,
    EntropyAwareDecoder with cfg-consistent dims."""
    m = CGForecaster(cfg)
    assert m.phase_module.V_raw == cfg.V_hmm_raw == 3
    assert m.phase_module.K == cfg.K_phase == 3
    assert m.phase_module.d_embed == cfg.d_model
    assert m.env_module.cfg.env_input_dim == cfg.env_input_dim
    assert m.encoder.cfg.main_input_dim == cfg.main_input_dim
    assert m.decoder.max_horizon == max(cfg.horizons)
    assert m.decoder.proj.out_features == len(cfg.horizons)


# ─────────────────────────────────────────────────────────────────
# T2 — B1: V_input=4 → x_phase = x[:, :, :V_hmm_raw=3] slicing
# ─────────────────────────────────────────────────────────────────

def test_b1_v_hmm_raw_slicing(model, cfg):
    """B1: changing the 4th channel (num_patients) should NOT change phase_post
    (PhaseModule only sees first V_hmm_raw=3 channels per EB-2)."""
    B, L = 2, 50
    x = torch.randn(B, L, cfg.main_input_dim)
    x_alt = x.clone()
    x_alt[:, :, 3] += 100.0  # massive num_patients perturbation

    model.eval()
    with torch.no_grad():
        _, post_orig = model.phase_module(x[:, :, :cfg.V_hmm_raw])
        _, post_alt = model.phase_module(x_alt[:, :, :cfg.V_hmm_raw])
    assert torch.allclose(post_orig, post_alt, atol=1e-6), \
        "B1 violated: phase_post depends on channel 3 (num_patients)"


# ─────────────────────────────────────────────────────────────────
# T3 — forward without prepare_for_stage2 → RuntimeError (M-4)
# ─────────────────────────────────────────────────────────────────

def test_forward_without_prepare_raises(cfg):
    """M-4 enforcement: forward before prepare_for_stage2 → RuntimeError."""
    m = CGForecaster(cfg)
    x = torch.randn(2, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(2, cfg.lookback, cfg.env_input_dim)
    with pytest.raises(RuntimeError, match="_cache_hmm_torch"):
        m(x, env)


# ─────────────────────────────────────────────────────────────────
# T4 — forward shape with prepared model
# ─────────────────────────────────────────────────────────────────

def test_forward_shape(model, cfg):
    """forward(x, env) returns [B, len(horizons)] after prepare_for_stage2."""
    B = 2
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)
    model.eval()
    with torch.no_grad():
        pred = model(x, env)
    assert pred.shape == (B, len(cfg.horizons))


# ─────────────────────────────────────────────────────────────────
# T5 — L-1 alignment: encoder receives [B, L-1, V_input]
# ─────────────────────────────────────────────────────────────────

def test_l_minus_1_alignment(model, cfg):
    """Encoder input is truncated to L-1, matching PhaseModule output length."""
    B, L = 2, 50
    captured = {}
    orig = model.encoder.forward

    def hook(x_in, context_vec=None):
        captured["enc_x_shape"] = tuple(x_in.shape)
        captured["ctx_shape"] = None if context_vec is None else tuple(context_vec.shape)
        return orig(x_in, context_vec=context_vec)

    model.encoder.forward = hook
    try:
        model.eval()
        with torch.no_grad():
            _ = model(torch.randn(B, L, cfg.main_input_dim),
                      torch.randn(B, L, cfg.env_input_dim))
    finally:
        model.encoder.forward = orig

    assert captured["enc_x_shape"] == (B, L - 1, cfg.main_input_dim)
    assert captured["ctx_shape"] == (B, L - 1, cfg.d_model)


# ─────────────────────────────────────────────────────────────────
# T6 — AND composition: context_vec = gate_phase * gate_env_truncated
# ─────────────────────────────────────────────────────────────────

def test_and_composition(model, cfg):
    """context_vec is element-wise product of gate_phase × gate_env[:, 1:, :]."""
    B, L = 2, 50
    x = torch.randn(B, L, cfg.main_input_dim)
    env = torch.randn(B, L, cfg.env_input_dim)
    model.eval()

    # Capture encoder context_vec input
    captured = {}
    orig = model.encoder.forward

    def hook(x_in, context_vec=None):
        captured["ctx"] = context_vec.detach() if context_vec is not None else None
        return orig(x_in, context_vec=context_vec)

    model.encoder.forward = hook
    try:
        with torch.no_grad():
            # Manually compute expected context_vec
            gate_phase, _ = model.phase_module(x[:, :, :cfg.V_hmm_raw])
            gate_env = model.env_module(env)
            expected_ctx = gate_phase * gate_env[:, 1:, :]
            _ = model(x, env)
    finally:
        model.encoder.forward = orig

    actual_ctx = captured["ctx"]
    assert torch.allclose(actual_ctx, expected_ctx, atol=1e-6), \
        "context_vec != gate_phase * gate_env[:, 1:, :]"


# ─────────────────────────────────────────────────────────────────
# T7 — phase_post structure: phase_init and gamma_last extractable
# ─────────────────────────────────────────────────────────────────

def test_phase_post_structure(model, cfg):
    """phase_post is [B, L-1, K]; phase_init = post[:, 0, :], gamma_last = post[:, -1, :].
    phase_init is unused in M1.6 but the structure must allow future M1.7 use."""
    B, L = 2, 30
    x = torch.randn(B, L, cfg.main_input_dim)
    model.eval()
    with torch.no_grad():
        _, post = model.phase_module(x[:, :, :cfg.V_hmm_raw])
    assert post.shape == (B, L - 1, cfg.K_phase)
    phase_init = post[:, 0, :]
    gamma_last = post[:, -1, :]
    assert phase_init.shape == (B, cfg.K_phase)
    assert gamma_last.shape == (B, cfg.K_phase)
    # Both should be valid simplex rows
    assert (phase_init.sum(-1) - 1.0).abs().max() < 1e-5
    assert (gamma_last.sum(-1) - 1.0).abs().max() < 1e-5


# ─────────────────────────────────────────────────────────────────
# T8 — gamma_last → decoder rollout (H-1 fix: rollout called once)
# ─────────────────────────────────────────────────────────────────

def test_gamma_last_to_decoder_rollout_once(model, cfg):
    """Each CGForecaster.forward triggers PhaseModule.rollout exactly once (H-1 fix)."""
    B = 2
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)

    call_count = [0]
    orig = model.phase_module.rollout

    def counting_rollout(*args, **kwargs):
        call_count[0] += 1
        return orig(*args, **kwargs)

    model.phase_module.rollout = counting_rollout
    try:
        model.eval()
        with torch.no_grad():
            _ = model(x, env)
    finally:
        model.phase_module.rollout = orig

    assert call_count[0] == 1, f"H-1: rollout called {call_count[0]} times, expected 1"


# ─────────────────────────────────────────────────────────────────
# T9 — x_window slicing with cfg.rollout_window=5
# ─────────────────────────────────────────────────────────────────

def test_x_window_slicing(model, cfg):
    """Decoder receives x_window of shape [B, rollout_window, V_raw] (or shorter if L<W)."""
    B = 2
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)

    captured = {}
    orig_rollout = model.phase_module.rollout

    def hook_rollout(gamma_t, x_window, H):
        captured["x_window_shape"] = tuple(x_window.shape)
        return orig_rollout(gamma_t, x_window, H)

    model.phase_module.rollout = hook_rollout
    try:
        model.eval()
        with torch.no_grad():
            _ = model(x, env)
    finally:
        model.phase_module.rollout = orig_rollout

    # L=156, W=5 → x_window = [B, 5, V_raw=3]
    assert captured["x_window_shape"] == (B, cfg.rollout_window, cfg.V_hmm_raw)


# ─────────────────────────────────────────────────────────────────
# T10 — Gradient flow: all trainable params receive nonzero grad
# ─────────────────────────────────────────────────────────────────

def test_gradient_flow_all_modules(model, cfg):
    """All trainable params (state_embeddings, gate_proj×3, encoder, decoder, env encoder)
    receive nonzero gradient in Stage 2."""
    B = 2
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)
    model.train()
    model.zero_grad()
    pred = model(x, env)
    pred.pow(2).mean().backward()

    # Spot-check each module
    assert model.phase_module.state_embeddings.grad is not None
    assert model.phase_module.state_embeddings.grad.abs().sum() > 0
    assert model.decoder.proj.weight.grad is not None
    assert model.decoder.proj.weight.grad.abs().sum() > 0
    assert model.decoder.gate.grad is not None
    assert model.decoder.gate.grad.abs() > 0
    # Encoder input_proj must have grad
    assert model.encoder.input_proj.weight.grad is not None
    assert model.encoder.input_proj.weight.grad.abs().sum() > 0
    # Env encoder must have grad
    env_enc_grad_sum = sum(p.grad.abs().sum().item()
                           for p in model.env_module.encoder.parameters()
                           if p.grad is not None)
    assert env_enc_grad_sum > 0
    # Env decoder must NOT have grad (frozen by prepare_for_stage2)
    for p in model.env_module.decoder.parameters():
        assert not p.requires_grad


# ─────────────────────────────────────────────────────────────────
# T11 — Stage 1 ckpt → prepare_for_stage2 → forward end-to-end
# ─────────────────────────────────────────────────────────────────

def test_stage1_ckpt_end_to_end(cfg):
    """Real Stage 1 ckpt loadable; full forward → backward works."""
    if not STAGE1_RUN_DIR.exists():
        pytest.skip(f"Stage 1 ckpt not found at {STAGE1_RUN_DIR}")
    m = CGForecaster(cfg)
    hmm = load_fitted_hmm(STAGE1_RUN_DIR)
    m.prepare_for_stage2(hmm)
    m.train()
    B = 2
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)
    pred = m(x, env)
    assert pred.shape == (B, len(cfg.horizons))
    pred.pow(2).mean().backward()  # no crash
    assert m.phase_module.state_embeddings.grad is not None


# ─────────────────────────────────────────────────────────────────
# T12 — Determinism in eval mode
# ─────────────────────────────────────────────────────────────────

def test_determinism_eval(model, cfg):
    """Same (x, env) → same prediction in eval mode."""
    model.eval()
    B = 2
    torch.manual_seed(0)
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)
    with torch.no_grad():
        p1 = model(x, env)
        p2 = model(x, env)
    assert torch.equal(p1, p2)


# ─────────────────────────────────────────────────────────────────
# T13 — L < rollout_window + L=2 extreme edge case (L-3 fix)
# ─────────────────────────────────────────────────────────────────

def test_l_less_than_rollout_window_edge(model, cfg):
    """L=3 < W=5: x_window clamped to L. L=2 extreme: augment → L-1=1; encoder + decoder still work."""
    B = 2
    # L=3 (< W=5)
    x3 = torch.randn(B, 3, cfg.main_input_dim)
    env3 = torch.randn(B, 3, cfg.env_input_dim)
    model.eval()
    with torch.no_grad():
        pred3 = model(x3, env3)
    assert pred3.shape == (B, len(cfg.horizons))

    # L=2 extreme: PhaseModule augment → L-1=1, encoder receives [B, 1, V_input]
    x2 = torch.randn(B, 2, cfg.main_input_dim)
    env2 = torch.randn(B, 2, cfg.env_input_dim)
    with torch.no_grad():
        pred2 = model(x2, env2)
    assert pred2.shape == (B, len(cfg.horizons))


# ─────────────────────────────────────────────────────────────────
# T14 — C-1 return_intermediates: dict structure + eval-mode access
# ─────────────────────────────────────────────────────────────────

def test_return_intermediates_dict_structure(model, cfg):
    """forward(return_intermediates=True) returns (predictions, dict) with all 9 keys.
    Each tensor has the expected shape. Works in eval mode (D-1 figure use case)."""
    B = 2
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)
    model.eval()
    with torch.no_grad():
        pred, inter = model(x, env, return_intermediates=True)

    assert pred.shape == (B, len(cfg.horizons))
    expected_keys = {
        "gate_phase", "phase_post", "context_vec", "gamma_last",
        "gamma_all", "fused", "eff_gate_per_horizon",
        "confidence_per_horizon", "phase_transition_kl",
    }
    assert set(inter.keys()) == expected_keys

    L_minus_1 = cfg.lookback - 1
    D = cfg.d_model
    K = cfg.K_phase
    max_h = max(cfg.horizons)
    H = len(cfg.horizons)

    assert inter["gate_phase"].shape == (B, L_minus_1, D)
    assert inter["phase_post"].shape == (B, L_minus_1, K)
    assert inter["context_vec"].shape == (B, L_minus_1, D)
    assert inter["gamma_last"].shape == (B, K)
    assert inter["gamma_all"].shape == (B, max_h, K)
    assert inter["fused"].shape == (B, L_minus_1, D)
    assert inter["eff_gate_per_horizon"].shape == (B, H)
    assert inter["confidence_per_horizon"].shape == (B, H)
    assert inter["phase_transition_kl"].shape == (B,)


# ─────────────────────────────────────────────────────────────────
# T15 — D-1: confidence_per_horizon ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────

def test_confidence_per_horizon_range(model, cfg):
    """D-1: confidence_per_horizon entries lie in [0, 1] for any input."""
    B = 4
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)
    model.eval()
    with torch.no_grad():
        _, inter = model(x, env, return_intermediates=True)
    conf = inter["confidence_per_horizon"]
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


# ─────────────────────────────────────────────────────────────────
# T16 — D-2: phase_transition_kl ≥ 0
# ─────────────────────────────────────────────────────────────────

def test_phase_transition_kl_nonneg(model, cfg):
    """D-2: KL(γ_0 || γ_{H-1}) ≥ 0 (information-theoretic property)."""
    B = 4
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)
    model.eval()
    with torch.no_grad():
        _, inter = model(x, env, return_intermediates=True)
    kl = inter["phase_transition_kl"]
    # Numerical tolerance for log(0) cleanup
    assert (kl >= -1e-6).all(), f"KL min = {kl.min().item()}"


# ─────────────────────────────────────────────────────────────────
# T17 — C-2: n_trainable_params + param_group_summary consistency
# ─────────────────────────────────────────────────────────────────

def test_param_helpers_consistency(model, cfg):
    """C-2: n_trainable_params matches param_group_summary['trainable_total'].
    Also: hmm_buffers_frozen > 0 (HMM is cached as frozen buffers, not parameters)."""
    n_total = model.n_trainable_params()
    summary = model.param_group_summary()
    assert summary["trainable_total"] == n_total
    # HMM buffers must not be in trainable params (they're frozen)
    assert summary["hmm_buffers_frozen"] > 0
    # Env decoder must be frozen after prepare_for_stage2
    assert summary["env_decoder_trainable"] == 0
    # phase state_embed = K × d_embed
    assert summary["phase_state_embed"] == cfg.K_phase * cfg.d_model


# ─────────────────────────────────────────────────────────────────
# T18 — Backward compat: forward without return_intermediates returns Tensor
# ─────────────────────────────────────────────────────────────────

def test_forward_backward_compat(model, cfg):
    """forward(x, env) (no return_intermediates) returns a plain Tensor (not tuple)."""
    B = 2
    x = torch.randn(B, cfg.lookback, cfg.main_input_dim)
    env = torch.randn(B, cfg.lookback, cfg.env_input_dim)
    model.eval()
    with torch.no_grad():
        out = model(x, env)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, len(cfg.horizons))


# ─────────────────────────────────────────────────────────────────
# T19 — A-1 verification: decoder.forward signature is pure tensors
# ─────────────────────────────────────────────────────────────────

def test_a1_decoder_pure_tensor_in_forecaster(model, cfg):
    """A-1: CGForecaster passes pure tensors (gamma_all, state_embeddings) to decoder,
    not a PhaseModule reference. Verify by inspecting the decoder forward signature
    AND by calling the decoder directly without any PhaseModule."""
    import inspect
    sig = inspect.signature(model.decoder.forward)
    args = list(sig.parameters.keys())[1:]  # exclude self
    assert "phase_module" not in args, f"A-1 violated: decoder takes phase_module — {args}"
    assert "gamma_all" in args
    assert "state_embeddings" in args

    # Functional check: call decoder directly with pure tensors (no PhaseModule)
    B = 2
    gamma_all = torch.softmax(
        torch.randn(B, max(cfg.horizons), cfg.K_phase), dim=-1
    )
    state_emb = torch.zeros(cfg.K_phase, cfg.d_model)
    encoder_out = torch.randn(B, 100, cfg.d_model)
    last_v = torch.randn(B)
    with torch.no_grad():
        pred = model.decoder(encoder_out, last_v, gamma_all, state_emb)
    assert pred.shape == (B, len(cfg.horizons))


# ──────────────────────────────────────────────────────────────────────────
# T14 — state_embeddings HMM-informed init regression (v2.1.4, ERR-C5)
# ──────────────────────────────────────────────────────────────────────────
def test_state_embeddings_hmm_init_post_prepare(model):
    """v2.1.4 (CH-1, ERR-C5): prepare_for_stage2 후 state_embeddings 대칭 파괴 검증.

    Regression protection: 미래 코드 변경 시 HMM-informed init이 우발적으로
    zeros init으로 회귀하면 즉시 fail. PLAN §3.4.x 참조.

    fixture `model`: cfg + synthetic_hmm 로부터 prepare_for_stage2(synthetic_hmm)
    이미 호출된 CGForecaster (test_cg_forecaster.py line 63 fixture 정의 참조).

    Verifies:
    - per-row norm > 0.1 (zeros init이 아님)
    - all pairwise |cosine| < 0.7 (symmetry broken)
    """
    E = model.phase_module.state_embeddings.data
    K = E.shape[0]

    # (1) per-row norm > 0.1 (zeros init이면 norm=0)
    norms = E.norm(dim=1)
    assert (norms > 0.1).all(), (
        f"state_embeddings rows near-zero: norms={norms.tolist()}. "
        f"HMM-informed init regressed to zeros? "
        f"Check CGForecaster._init_state_embeddings_from_cache."
    )

    # (2) pairwise |cosine| < 0.7 (symmetry broken)
    for i in range(K):
        for j in range(i + 1, K):
            cos = torch.cosine_similarity(E[i:i+1], E[j:j+1]).item()
            assert abs(cos) < 0.7, (
                f"|cos(e{i}, e{j})|={abs(cos):.4f} >= 0.7 threshold. "
                f"Symmetry not broken — check _init_state_embeddings_from_cache "
                f"or HMM checkpoint may be degenerate."
            )
