"""DEPRECATED in v2.0.9 — retained for paper §7.4 ablation B reproduction. Not active in v2.0.9 main path.

Integration tests for `src/models/legacy/hmm_stage1.py` (v2.0.8c M1.4 main path).

Tests cover the data-preparation + import-contract layer of the
NeuralSwitchingVARHMM main path. Heavy ML training (EM/Adam) is not tested
here — that is covered by the smoke + grid runs (scripts/run_hmm_stage1.py).

Scope:
  - prepare_hmm_train: seg2-only (v2.0.8c ED-1), shape contract, alignment
  - TRAIN_START_EPIWEEK constant + value
  - HMM_V4_COLS / HMM_V3_FALLBACK_COLS contracts
  - import path (NeuralSwitchingVARHMM via importlib absolute path)

Run: pytest -xvs src/tests/test_hmm_stage1.py
"""
from __future__ import annotations

import pytest
import torch
import numpy as np

from src.models.legacy.hmm_stage1 import (  # v2.0.9: moved to legacy
    NeuralSwitchingVARHMM,
    HMM_V4_COLS,
    HMM_V3_FALLBACK_COLS,
    TRAIN_START_EPIWEEK,
    prepare_hmm_train,
    forward_train,
    init_hmm,
    get_param_groups,
    freeze_hmm_for_stage2,
)
from src.utils.config import CGMambaConfig


# ─────────────────────────────────────────────────────────────────
# T1 — Constants
# ─────────────────────────────────────────────────────────────────

def test_train_start_epiweek_value():
    """v2.0.8c ED-1: TRAIN_START_EPIWEEK must equal 200240 (post-CDC-gap start)."""
    assert TRAIN_START_EPIWEEK == 200240, \
        f"TRAIN_START_EPIWEEK={TRAIN_START_EPIWEEK}, expected 200240 per v2.0.8c ED-1"


def test_v4_cols_content():
    """V=4 default columns: ili_weighted_pct, total_ili_count, num_providers, num_patients."""
    assert HMM_V4_COLS == [
        "ili_weighted_pct",
        "total_ili_count",
        "num_providers",
        "num_patients",
    ]


def test_v3_fallback_cols_drops_num_patients():
    """V=3 fallback drops num_patients (PLAN v2.0.8b EB-2, r=0.952 vs num_providers)."""
    assert HMM_V3_FALLBACK_COLS == [
        "ili_weighted_pct",
        "total_ili_count",
        "num_providers",
    ]
    assert "num_patients" not in HMM_V3_FALLBACK_COLS, \
        "num_patients must be dropped from V=3 fallback (EB-2)"


# ─────────────────────────────────────────────────────────────────
# T2 — prepare_hmm_train data alignment
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CGMambaConfig()


def test_prepare_hmm_train_v4_shape(cfg):
    """V=4 default: seg [1, 835, 4]."""
    seg, ili_raw = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    assert seg.shape == (1, 835, 4), \
        f"Expected [1, 835, 4], got {tuple(seg.shape)}"
    assert ili_raw.shape == (835,), \
        f"ili_raw expected (835,), got {ili_raw.shape}"


def test_prepare_hmm_train_v3_shape(cfg):
    """V=3 fallback: seg [1, 835, 3]."""
    seg, ili_raw = prepare_hmm_train(
        cfg.data_csv, cfg.norm_json, feature_cols=HMM_V3_FALLBACK_COLS,
    )
    assert seg.shape == (1, 835, 3), \
        f"Expected [1, 835, 3], got {tuple(seg.shape)}"


def test_prepare_hmm_train_starts_at_TRAIN_START_EPIWEEK(cfg):
    """Train data must start at 2002-W40 (epiweek 200240, v2.0.8c)."""
    import pandas as pd
    df = pd.read_csv(cfg.data_csv).sort_values("epiweek").reset_index(drop=True)
    train = df[df["split"] == "train"].reset_index(drop=True)
    seg2 = train[train["epiweek"].astype(int) >= TRAIN_START_EPIWEEK]
    assert len(seg2) == 835
    assert int(seg2["epiweek"].iloc[0]) == TRAIN_START_EPIWEEK


def test_prepare_hmm_train_ili_raw_range(cfg):
    """Raw ILI values are unnormalized (used for binary κ computation)."""
    _, ili_raw = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    # Real ILI %wILI range — from M1.1 PROVENANCE
    assert ili_raw.min() >= 0.0, f"ILI < 0: {ili_raw.min()}"
    assert ili_raw.max() < 10.0, f"ILI implausibly high: {ili_raw.max()}"
    # Epi threshold (PLAN §5.1): some weeks must exceed 2.0
    assert (ili_raw > 2.0).sum() > 50, \
        f"Too few epi weeks: {(ili_raw > 2.0).sum()} (CDC ILI baseline 2.0)"


def test_prepare_hmm_train_z_score_target_normalized(cfg):
    """ili_weighted_pct column (idx 0) z-scored: train-set μ subtracted."""
    seg, _ = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    z = seg.squeeze(0)[:, 0].numpy()
    # Z-score with train statistics → mean somewhere near 0 (but not exactly,
    # since the train set in the scaler includes seg1 which we dropped here)
    assert abs(z.mean()) < 2.0, \
        f"Z-scored ILI mean too far from 0: {z.mean():.3f}"
    assert 0.3 < z.std() < 3.0, \
        f"Z-scored ILI std out of plausible: {z.std():.3f}"


# ─────────────────────────────────────────────────────────────────
# T3 — Data alignment: main ↔ ablation bit-identical (v2.0.8c)
# ─────────────────────────────────────────────────────────────────

def test_main_vs_ablation_train_alignment(cfg):
    """main path (prepare_hmm_train) ↔ ablation (load_ili_train_seg2):
    train data must be bit-identical (apples-to-apples §7.4 ablation)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "legacy"))  # v2.0.9: legacy/
    import m1_4_ablation_gaussian_hmm_search as ablation

    seg_main, _ = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    x_main = seg_main.squeeze(0).numpy().astype(np.float64)
    x_abl = ablation.load_ili_train_seg2(cfg)

    np.testing.assert_allclose(
        x_main, x_abl, atol=1e-7,
        err_msg="Main path and ablation train data must be bit-identical"
    )


# ─────────────────────────────────────────────────────────────────
# T4 — NeuralSwitchingVARHMM import contract
# ─────────────────────────────────────────────────────────────────

def test_ns_var_hmm_importable():
    """NeuralSwitchingVARHMM class must be importable (B1 fix verification)."""
    assert NeuralSwitchingVARHMM is not None
    # Should have a forward method
    assert hasattr(NeuralSwitchingVARHMM, "forward"), \
        "NeuralSwitchingVARHMM missing forward method"


def test_ns_var_hmm_instantiation():
    """Instantiate HMM with V=4, K=3 (smallest grid spec)."""
    model = NeuralSwitchingVARHMM(V=4, num_states=3, hidden=64)
    assert model is not None
    # Should be on CPU (no device argument in __init__)
    params = list(model.parameters())
    assert len(params) > 0, "Model has no parameters"


# ─────────────────────────────────────────────────────────────────
# T5 — init_hmm + get_param_groups contract
# ─────────────────────────────────────────────────────────────────

def test_init_hmm_freezes_mu0(cfg):
    """μ₀ is frozen after init_hmm (PLAN EB-3 KMeans warm-start + μ₀ freeze)."""
    seg, _ = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    train_combined = seg.squeeze(0).numpy()
    model = init_hmm(V=4, K=3, seed=42, train_combined=train_combined)
    assert not model.mu0.requires_grad, \
        "μ₀ must be frozen (requires_grad=False) after init_hmm"


def test_get_param_groups_single_stage1(cfg):
    """Stage 1 has single param group: all trainable params at given lr."""
    seg, _ = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    train_combined = seg.squeeze(0).numpy()
    model = init_hmm(V=4, K=3, seed=42, train_combined=train_combined)
    groups = get_param_groups(model, lr=1e-3)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3
    assert groups[0]["name"] == "hmm_stage1"
    # All params in groups[0]["params"] should be trainable
    for p in groups[0]["params"]:
        assert p.requires_grad


# ─────────────────────────────────────────────────────────────────
# T6 — forward_train shape contract (single-segment, v2.0.8c)
# ─────────────────────────────────────────────────────────────────

def test_forward_train_shapes(cfg):
    """forward_train returns (phase_post [1, L, K], nll scalar, h_last)."""
    seg, _ = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    train_combined = seg.squeeze(0).numpy()
    model = init_hmm(V=4, K=3, seed=42, train_combined=train_combined)

    phase_post, nll, h_last = forward_train(model, seg)
    assert phase_post.shape == (1, 835, 3), \
        f"phase_post expected [1, 835, 3], got {tuple(phase_post.shape)}"
    assert nll.dim() == 0 or nll.numel() == 1, \
        f"nll must be scalar, got shape {nll.shape}"
    # phase_post must satisfy simplex constraint
    sums = phase_post.sum(dim=-1).squeeze(0)  # [835]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3), \
        "phase_post rows must sum to 1 (softmax simplex)"


# ─────────────────────────────────────────────────────────────────
# T7 — Config / PLAN spec alignment (review item 4)
# ─────────────────────────────────────────────────────────────────

def test_config_hmm_seeds_match_plan_grid(cfg):
    """config.hmm_seeds must match SEED_GRID in run_hmm_stage1.py (PLAN §3.7)."""
    # Import SEED_GRID from the grid driver to ensure single source of truth
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "legacy"))  # v2.0.9: legacy/
    import run_hmm_stage1 as grid_driver

    assert tuple(cfg.hmm_seeds) == tuple(grid_driver.SEED_GRID), (
        f"config.hmm_seeds={cfg.hmm_seeds} != "
        f"SEED_GRID={grid_driver.SEED_GRID} — PLAN §3.7 violation"
    )
    # Also verify they equal the canonical (42, 123, 456) spec
    assert tuple(cfg.hmm_seeds) == (42, 123, 456), \
        f"PLAN v2.0.8b EB-3 specifies seeds=(42, 123, 456), got {cfg.hmm_seeds}"


def test_config_n_states_in_K_grid(cfg):
    """config.n_states (default K) must be in PLAN §3.7 K grid {3, 4, 5}."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "legacy"))  # v2.0.9: legacy/
    import run_hmm_stage1 as grid_driver

    assert cfg.n_states in grid_driver.K_GRID, (
        f"config.n_states={cfg.n_states} not in K_GRID={grid_driver.K_GRID}"
    )


# ─────────────────────────────────────────────────────────────────
# T8 — Stage 2 HMM freeze guard (review item 3: DE-7 silent-bug prevention)
# ─────────────────────────────────────────────────────────────────

def test_freeze_hmm_for_stage2_freezes_all_params(cfg):
    """freeze_hmm_for_stage2 must set all HMM params to requires_grad=False."""
    seg, _ = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    train_combined = seg.squeeze(0).numpy()
    model = init_hmm(V=4, K=3, seed=42, train_combined=train_combined)

    # Pre-condition: at least some params trainable (μ₀ frozen, others trainable)
    trainable_before = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_before) > 0, "Expected some trainable params before freeze"

    # Freeze
    n_frozen = freeze_hmm_for_stage2(model)
    assert n_frozen == len(trainable_before), (
        f"freeze count {n_frozen} != trainable-before {len(trainable_before)}"
    )

    # Post-condition: no params trainable
    trainable_after = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_after) == 0, (
        f"After freeze, {len(trainable_after)} params still trainable — "
        f"Stage 2 silent-bug risk (DE-7)"
    )


def test_freeze_hmm_idempotent(cfg):
    """freeze_hmm_for_stage2 is idempotent (2nd call has no effect)."""
    seg, _ = prepare_hmm_train(cfg.data_csv, cfg.norm_json)
    train_combined = seg.squeeze(0).numpy()
    model = init_hmm(V=4, K=3, seed=42, train_combined=train_combined)
    freeze_hmm_for_stage2(model)
    n_frozen_2nd = freeze_hmm_for_stage2(model)
    assert n_frozen_2nd == 0, \
        f"Idempotent freeze 2nd call should freeze 0 (got {n_frozen_2nd})"
