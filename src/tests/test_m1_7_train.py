"""M1.7 Stage 2 trainer + optimizer/scheduler/loss tests.

Test scope (Direction Message T1-T15 alignment):
  Core auto-regressed tests (always-on regression protection):
    T1-T6: Param group consistency (ERR-C2, group sizes, WD/LR per group).
    T11:   Loss arithmetic accuracy on known inputs.
    T12:   Scheduler LR curve over 200 epochs (C-1 zero drift + boundary continuity).
    T13:   sanity assert at total_epochs == P1+TR (Step 6 misuse-proofing).

  Lightweight regression (smoke run already validated):
    T-smoke: Stage 2 trainer smoke run produces all expected artifacts.

Tests deliberately omitted (covered by smoke or upstream):
    T7  (loss decreases)        — smoke run already shows monotonic ↓.
    T8  (grad clip)             — torch primitive, tested upstream.
    T9  (early stopping)        — patience logic is trivial counter.
    T10 (env pretrain)          — owned by m1_7_env_pretrain.py + diagnostics.json.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.utils.losses import cg_mamba_loss, mase_loss
from src.utils.optimizer import build_stage2_optimizer
from src.utils.scheduler import build_warm_gamma_scheduler


_HMM_DIR = (
    Path(__file__).resolve().parents[2]
    / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed42"
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def cfg():
    return CGMambaConfig()


@pytest.fixture(scope="module")
def stage2_ready_model(cfg):
    """CGForecaster with prepare_for_stage2() called (env decoder frozen, HMM cached)."""
    model = CGForecaster(cfg)
    if not _HMM_DIR.exists():
        pytest.skip(f"Stage 1 HMM checkpoint missing: {_HMM_DIR}")
    hmm = load_fitted_hmm(_HMM_DIR)
    model.prepare_for_stage2(hmm)
    return model


@pytest.fixture(scope="module")
def optimizer(cfg, stage2_ready_model):
    return build_stage2_optimizer(stage2_ready_model, cfg)


# ──────────────────────────────────────────────────────────────────────────
# T1-T6 — Param group consistency (PLAN D.5.2)
# ──────────────────────────────────────────────────────────────────────────
def test_t1_optimizer_has_four_named_groups(optimizer):
    names = [g["name"] for g in optimizer.param_groups]
    assert names == ["gate_proj", "decoder_gate", "context_embed", "backbone"], (
        f"Group order/names mismatch: {names}"
    )


def test_t2_per_group_param_counts(optimizer):
    """PLAN D.5.2 group sizes (after prepare_for_stage2 freeze)."""
    expected = {
        "gate_proj": (12, 5_016),     # 4 sub-modules × 3 layers
        "decoder_gate": (1, 1),        # scalar α
        "context_embed": (1, 192),     # K=3 × d_embed=64 state_embeddings
        "backbone": (39, 110_180),     # remainder (encoder + env encoder + decoder.proj)
    }
    for g in optimizer.param_groups:
        exp_entries, exp_numel = expected[g["name"]]
        actual_entries = len(g["params"])
        actual_numel = sum(p.numel() for p in g["params"])
        assert actual_entries == exp_entries, (
            f"{g['name']}: entries {actual_entries} ≠ {exp_entries}"
        )
        assert actual_numel == exp_numel, (
            f"{g['name']}: numel {actual_numel} ≠ {exp_numel}"
        )


def test_t3_per_group_lr_matches_plan_d52(optimizer, cfg):
    """LR values from PLAN D.5.2 (lines 893-906).

    Note: context_embed=1e-5 is a v2.1.3 ablation deviation from PLAN's 1e-6
    near-freeze, motivated by Step 8 finding (state_embeddings barely moved).
    """
    expected_lr = {
        "gate_proj": cfg.stage2_gate_lr,           # 1e-3
        "decoder_gate": cfg.stage2_backbone_lr,    # 5e-5
        "context_embed": 1e-5,                      # v2.1.3 (was 1e-6, see optimizer.py comment)
        "backbone": cfg.stage2_backbone_lr,        # 5e-5
    }
    for g in optimizer.param_groups:
        assert g["lr"] == expected_lr[g["name"]], (
            f"{g['name']}: lr {g['lr']} ≠ {expected_lr[g['name']]}"
        )


def test_t4_per_group_weight_decay_matches_plan_d52(optimizer, cfg):
    """WD values: backbone=0.01, gate_proj=1e-3, others=0 (PLAN D.5.2)."""
    expected_wd = {
        "gate_proj": 1e-3,
        "decoder_gate": 0.0,
        "context_embed": 0.0,
        "backbone": cfg.stage2_backbone_wd,        # 0.01
    }
    for g in optimizer.param_groups:
        assert g["weight_decay"] == expected_wd[g["name"]], (
            f"{g['name']}: wd {g['weight_decay']} ≠ {expected_wd[g['name']]}"
        )


def test_t5_err_c2_no_duplicate_no_unassigned(optimizer, stage2_ready_model):
    """ERR-C2: every trainable param is in exactly one group, no duplicates."""
    trainable = {id(p) for p in stage2_ready_model.parameters() if p.requires_grad}
    assigned = []
    for g in optimizer.param_groups:
        for p in g["params"]:
            assigned.append(id(p))
    # Bijection check
    assert len(assigned) == len(trainable), (
        f"Group total {len(assigned)} ≠ trainable {len(trainable)} (duplicates or missing)"
    )
    assert set(assigned) == trainable, "Group assignment ≠ trainable set (mismatch)"


def test_t6_grand_total_trainable_equals_115389(optimizer):
    """Defense-in-depth: total trainable count is exactly 115,389 (PLAN §3.0 budget)."""
    total = sum(
        sum(p.numel() for p in g["params"])
        for g in optimizer.param_groups
    )
    assert total == 115_389, f"Trainable total {total} ≠ 115,389 (PLAN budget)"


# ──────────────────────────────────────────────────────────────────────────
# T11 — Loss arithmetic accuracy (known inputs)
# ──────────────────────────────────────────────────────────────────────────
def test_t11_loss_zero_when_pred_equals_target():
    """pred == target → MSE=0, MASE=0, total=0 (composite identity)."""
    pred = torch.randn(8, 4)
    seasonal_mae = torch.tensor(0.5)
    total = cg_mamba_loss(pred, pred, seasonal_mae, lambda_mase=0.3)
    assert total.item() == pytest.approx(0.0, abs=1e-7)


def test_t11_mase_scales_as_mae_over_seasonal():
    """MAE(pred, target) = const → MASE = const / seasonal_mae."""
    target = torch.zeros(8, 4)
    pred = torch.full_like(target, 0.7)            # |pred - target| = 0.7
    seasonal_mae = torch.tensor(0.5)
    mase = mase_loss(pred, target, seasonal_mae)
    assert mase.item() == pytest.approx(0.7 / 0.5, rel=1e-5)


def test_t11_composite_loss_equals_mse_plus_lambda_mase():
    """cg_mamba_loss = MSE + λ·MASE (identity check, PLAN D.5.2 §5.1)."""
    target = torch.zeros(8, 4)
    pred = torch.full_like(target, 0.3)            # MSE=0.09, MASE=0.3/0.5=0.6
    seasonal_mae = torch.tensor(0.5)
    composite = cg_mamba_loss(pred, target, seasonal_mae, lambda_mase=0.3)
    expected = 0.09 + 0.3 * 0.6                    # = 0.27
    assert composite.item() == pytest.approx(expected, rel=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# T12 — Scheduler LR curve (200 epoch sweep + C-1 zero drift + boundaries)
# ──────────────────────────────────────────────────────────────────────────
def test_t12_context_embed_zero_drift_200_epochs(optimizer, cfg):
    """C-1 fix: context_embed LR must remain at its base value across all 200 epochs.

    Spec value is v2.1.3 ablation 1e-5 (raised from PLAN D.5.2's 1e-6 near-freeze).
    The C-1 mechanism (scheduler early-return `1.0` multiplier) is invariant to
    the base LR — what matters is *zero drift*, not the specific value.
    """
    sched = build_warm_gamma_scheduler(optimizer, total_epochs=cfg.stage2_n_epochs)
    ctx_idx = [g["name"] for g in optimizer.param_groups].index("context_embed")
    optimizer.step()  # PyTorch convention: opt.step before sched.step
    for _ in range(cfg.stage2_n_epochs):
        sched.step()
        lr = optimizer.param_groups[ctx_idx]["lr"]
        assert abs(lr - 1e-5) < 1e-12, f"context_embed LR drifted to {lr:.6e}"


def test_t12_boundary_continuity_warmup_to_phase1(cfg):
    """LR curve continuity: warmup last epoch == Phase 1 first epoch."""
    model = CGForecaster(cfg)
    opt = build_stage2_optimizer(model, cfg)
    sched = build_warm_gamma_scheduler(opt, total_epochs=cfg.stage2_n_epochs)
    gate_idx = [g["name"] for g in opt.param_groups].index("gate_proj")
    opt.step()
    # Default: warmup=2, P1=10. Step through warmup → Phase 1 transition.
    lrs = []
    for _ in range(3):                    # captures epochs 0, 1, 2
        sched.step()
        lrs.append(opt.param_groups[gate_idx]["lr"])
    # warmup last step (epoch 1) should hit base LR (1.0 multiplier), and Phase 1 (epoch 2) holds
    base = 1e-3
    assert lrs[1] == pytest.approx(base, rel=1e-5), f"warmup endpoint {lrs[1]} ≠ base"
    assert lrs[2] == pytest.approx(base, rel=1e-5), f"Phase 1 LR {lrs[2]} ≠ base"


def test_t12_boundary_continuity_transition_to_cosine(cfg):
    """LR curve continuity: Transition endpoint ≈ Cosine start (phase2_start_ratio)."""
    model = CGForecaster(cfg)
    opt = build_stage2_optimizer(model, cfg)
    sched = build_warm_gamma_scheduler(opt, total_epochs=cfg.stage2_n_epochs)
    gate_idx = [g["name"] for g in opt.param_groups].index("gate_proj")
    opt.step()
    # Default: P1=10, TR=10 → Transition ends at epoch 19, Cosine starts at 20
    for _ in range(20):                   # advance to epoch 19
        sched.step()
    lr_tr_end = opt.param_groups[gate_idx]["lr"]
    sched.step()                          # epoch 20: Cosine start
    lr_cos_start = opt.param_groups[gate_idx]["lr"]
    # phase2_start_ratio = 0.5 (default) → both ≈ 0.5 × base_lr = 5e-4
    expected = 0.5 * 1e-3
    assert lr_tr_end == pytest.approx(expected, rel=5e-3), (
        f"Transition end {lr_tr_end} ≠ expected {expected}"
    )
    # Continuity: Cosine start should be ≈ Transition end (within first cosine step delta)
    assert abs(lr_cos_start - lr_tr_end) / lr_tr_end < 0.01, (
        f"Discontinuity Trans→Cosine: {lr_tr_end} → {lr_cos_start}"
    )


def test_t12_cosine_floor_reached_at_last_epoch(cfg):
    """Cosine annealing reaches eta_min=1e-6 floor at epoch=stage2_n_epochs-1."""
    model = CGForecaster(cfg)
    opt = build_stage2_optimizer(model, cfg)
    sched = build_warm_gamma_scheduler(opt, total_epochs=cfg.stage2_n_epochs)
    gate_idx = [g["name"] for g in opt.param_groups].index("gate_proj")
    opt.step()
    for _ in range(cfg.stage2_n_epochs):
        sched.step()
    # gate_proj final LR should hit floor 1e-6
    final_lr = opt.param_groups[gate_idx]["lr"]
    assert final_lr == pytest.approx(1e-6, rel=1e-3), (
        f"gate_proj at last epoch: {final_lr} ≠ 1e-6 floor"
    )


# ──────────────────────────────────────────────────────────────────────────
# T13 — Sanity assert (Step 6 misuse-proofing)
# ──────────────────────────────────────────────────────────────────────────
def test_t13_scheduler_rejects_too_small_total_epochs(cfg):
    """build_warm_gamma_scheduler asserts total_epochs > P1+TR (default 20)."""
    model = CGForecaster(cfg)
    opt = build_stage2_optimizer(model, cfg)
    # Boundary: total_epochs == P1+TR=20 must be rejected
    with pytest.raises(AssertionError, match="must be > P1\\+TR"):
        build_warm_gamma_scheduler(opt, total_epochs=20)
    # total_epochs == 21 (just above) must succeed
    sched = build_warm_gamma_scheduler(opt, total_epochs=21)
    assert sched is not None


# ──────────────────────────────────────────────────────────────────────────
# T-smoke — Lightweight regression on trainer output artifacts
# (smoke run already validated dynamics; this just protects file shape)
# ──────────────────────────────────────────────────────────────────────────
def test_smoke_trainer_artifacts_shape():
    """Sanity: latest smoke run produced best.pt + history.json + final_metrics.json
    with required keys. Skipped if no smoke run available.
    """
    smoke_runs = sorted(
        (Path(__file__).resolve().parents[2] / "runs" / "m1_7_train").glob("smoke_*")
    )
    if not smoke_runs:
        pytest.skip("No smoke run artifacts available (run scripts/m1_7_train.py --smoke first)")
    latest = smoke_runs[-1]
    assert (latest / "best.pt").exists()
    assert (latest / "history.json").exists()
    assert (latest / "final_metrics.json").exists()
    assert (latest / "config.json").exists()
    import json
    final = json.loads((latest / "final_metrics.json").read_text())
    for key in ("best_epoch", "best_val_total", "best_val_mse", "best_val_mase",
                "test_total", "test_mse", "test_mase", "seasonal_mae"):
        assert key in final, f"Missing key in final_metrics.json: {key}"
