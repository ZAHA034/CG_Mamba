"""DEPRECATED in v2.0.9 — retained for paper §7.4 ablation B reproduction (NSVARHMM autoregressive emission). Not active in v2.0.9 main path.

HMM Stage 1 wrapper for CG-Mamba (v2.0.8c).

Spec (PLAN §3.4, §5.1):
  - Input V=4 default: ili_weighted_pct, total_ili_count, num_providers, num_patients
  - V=3 fallback: drop num_patients (multicollinearity r=0.952)
  - K ∈ {3, 4, 5} × seeds ∈ {42, 123, 456} = 9 runs
  - **Train data: seg2-only (2002-W40 ~ 2018-W39, 835 rows)** — v2.0.8c
      Drop seg1 (200140 ~ 200220, 33 rows) for system-wide consistency with
      sliding-window models (LSTM, CG-Mamba) which auto-exclude seg1 because
      L=104+ lookback exceeds seg1 length 33. CDC 2002 summer gap
      (200220 → 200240, 19 weeks missing) is upstream of train start.
      *Previous (v2.0.8b): gap-aware 2-segment forward with seg1=33 included.*
  - KMeans warm-start (initialize_from_data, seed=s) → μ0 freeze
  - Fallback trigger (PLAN EB-3): dead<5% / 3-seed 전부 final κ<0.50 / σ collapse

Import policy (cross-project NeuralSwitchingVARHMM via importlib absolute path):
  CM_Mamba/legacy/v5_2_x/models/ns_var_hmm.py 절대 경로 로드 (B1 sys.path 순서
  충돌 회피, M1.4 review). 의존성 단방향.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cross-project import: NeuralSwitchingVARHMM
# ---------------------------------------------------------------------------
# Use importlib + absolute file path to avoid sys.path order collisions with
# src/models/ (which has no ns_var_hmm.py — would shadow the legacy lookup if
# `src/` is on sys.path before _LEGACY_PATH, as happens when running via
# `python scripts/run_hmm_stage1.py`).
_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[3]  # CG_Mamba/ (legacy/ moved file: src/models/legacy/hmm_stage1.py)
_CM_MAMBA_ROOT = _CG_MAMBA_ROOT.parent / "CM_Mamba"
_NS_VAR_HMM_PATH = _CM_MAMBA_ROOT / "legacy" / "v5_2_x" / "models" / "ns_var_hmm.py"

if not _NS_VAR_HMM_PATH.exists():
    raise ImportError(
        f"NeuralSwitchingVARHMM source not found at {_NS_VAR_HMM_PATH}. "
        f"Expected CM_Mamba/legacy/v5_2_x/models/ns_var_hmm.py."
    )

import importlib.util as _importlib_util

_spec = _importlib_util.spec_from_file_location(
    "_legacy_ns_var_hmm", str(_NS_VAR_HMM_PATH)
)
_mod = _importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
NeuralSwitchingVARHMM = _mod.NeuralSwitchingVARHMM

# Local imports — use absolute `src.data.loader` form (works when CG_Mamba root
# is the python cwd, which is the convention for all M1.x runs).
from src.data.loader import load_dataset_csv, load_norm_params  # noqa: E402


# ---------------------------------------------------------------------------
# Constants (PLAN spec, v2.0.8b)
# ---------------------------------------------------------------------------
HMM_V4_COLS = [
    "ili_weighted_pct",
    "total_ili_count",
    "num_providers",
    "num_patients",
]
HMM_V3_FALLBACK_COLS = [
    "ili_weighted_pct",
    "total_ili_count",
    "num_providers",
    # num_patients dropped (EB-2, r=0.952 multicollinearity)
]
# Train data start cutoff (v2.0.8c — system-wide alignment with sliding-window
# models). seg1 (200140 ~ 200220, 33 rows) is excluded because LSTM/CG-Mamba's
# gap-aware sliding window (loader.py:WeeklyDataset) cannot use it (L=104+ >
# seg1 length = 33). HMM main path follows the same boundary for fair comparison.
# Original CDC gap: 200220 → 200240 (19 weeks missing, 2002 summer anomaly).
TRAIN_START_EPIWEEK = 200240   # 2002-W40, post-gap start


# ---------------------------------------------------------------------------
# Data preparation (seg2-only, v2.0.8c — sliding-window model alignment)
# ---------------------------------------------------------------------------
def prepare_hmm_train(
    csv_path: str | Path,
    norm_path: str | Path,
    feature_cols: list[str] = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Train HMM on seg2 only (post-gap, 2002-W40 ~ 2018-W39, 835 rows).

    v2.0.8c change (was: prepare_hmm_train_segments returning 2 segments):
        Drop seg1 (200140 ~ 200220, 33 rows) for system-wide consistency
        with sliding-window models (LSTM, CG-Mamba) which auto-exclude seg1
        because L=104+ lookback > seg1 length = 33. Same effective training
        data across all forecasting models in paper §7.1 comparison.

    Args:
        feature_cols: HMM input features (None → HMM_V4_COLS default)

    Returns:
        seg: [1, L=835, V] — train seg2 only (2002-W40 ~ 2018-W39)
        ili_raw_train: [835] — raw ili_weighted_pct from seg2 (κ 계산용)
    """
    if feature_cols is None:
        feature_cols = HMM_V4_COLS

    df = load_dataset_csv(Path(csv_path))
    norm = load_norm_params(Path(norm_path))

    train = df[df["split"] == "train"].reset_index(drop=True)
    train["epiweek"] = train["epiweek"].astype(int)
    assert len(train) == 868, f"Expected 868 train rows, got {len(train)}"

    # seg2-only: drop pre-gap rows (epiweek < TRAIN_START_EPIWEEK)
    seg_df = train[train["epiweek"] >= TRAIN_START_EPIWEEK].reset_index(drop=True)
    L = len(seg_df)
    assert L == 835, f"Expected 835 seg2 rows (post-gap), got {L}"
    assert int(seg_df["epiweek"].iloc[0]) == TRAIN_START_EPIWEEK, \
        f"seg2 must start at {TRAIN_START_EPIWEEK}, got {seg_df['epiweek'].iloc[0]}"

    # Build feature tensor (z-score / log1p — loader.py 패턴 일관)
    cols = []
    for c in feature_cols:
        if c == "ili_weighted_pct":
            m = norm["ili_weighted_pct"]["mean"]
            s = norm["ili_weighted_pct"]["std"]
            cols.append((seg_df[c].to_numpy() - m) / s)
        elif c in ("total_ili_count", "num_providers", "num_patients"):
            cols.append(np.log1p(seg_df[c].to_numpy()))
        elif c == "temperature_c":
            m = norm["temperature_c"]["mean"]
            s = norm["temperature_c"]["std"]
            cols.append((seg_df[c].to_numpy() - m) / s)
        elif c == "specific_humidity_g_per_kg":
            m = norm["specific_humidity_g_per_kg"]["mean"]
            s = norm["specific_humidity_g_per_kg"]["std"]
            cols.append((seg_df[c].to_numpy() - m) / s)
        else:
            raise ValueError(f"Unknown feature column: {c}")
    feat = np.stack(cols, axis=-1).astype(np.float32)  # [L, V]
    seg = torch.from_numpy(feat).unsqueeze(0)          # [1, L, V]

    ili_raw_train = seg_df["ili_weighted_pct"].to_numpy().astype(np.float64)
    return seg, ili_raw_train


# ---------------------------------------------------------------------------
# Training utilities (single-segment forward, v2.0.8c)
# ---------------------------------------------------------------------------
def forward_train(
    model: NeuralSwitchingVARHMM,
    seg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-segment forward on seg2-only train data.

    v2.0.8c change (was: forward_two_segments with NLL weighted average):
        seg1 excluded upstream in prepare_hmm_train; this function now
        takes a single contiguous tensor [1, 835, V] and returns the
        standard forward outputs. No segment weighting needed.

    seg: [1, L=835, V]
    Returns:
        phase_post: [1, L, K] soft posteriors
        nll: scalar
        h_last: model._h_last (last hidden state, for rollout)
    """
    phase_post, _, nll = model(seg)
    h_last = model._h_last
    return phase_post, nll, h_last


def init_hmm(
    V: int,
    K: int,
    seed: int,
    train_combined: np.ndarray,
    hidden: int = 64,
) -> NeuralSwitchingVARHMM:
    """HMM 인스턴스 생성 + KMeans warm-start + μ0 freeze.

    Args:
        V: feature dim
        K: number of states
        seed: torch + KMeans seed (동일값 binding)
        train_combined: [L1+L2, V] flat array for KMeans init
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NeuralSwitchingVARHMM(V=V, num_states=K, hidden=hidden)
    # ns_var_hmm.py line 236-261: initialize_from_data(data_flat, seed=...)
    model.initialize_from_data(train_combined, seed=seed)
    model.mu0.requires_grad = False  # freeze anchor (EB-3)
    return model


def get_param_groups(model: NeuralSwitchingVARHMM, lr: float = 1e-3) -> list[dict]:
    """Stage 1 단일 group (μ0 frozen, 나머지 학습)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return [{"name": "hmm_stage1", "params": trainable, "lr": lr}]


# ---------------------------------------------------------------------------
# Stage 2 freeze guard (M1.7 silent-bug prevention, review item DE-7)
# ---------------------------------------------------------------------------
def freeze_hmm_for_stage2(model: NeuralSwitchingVARHMM) -> int:
    """DEPRECATED in v2.0.9 — NSVARHMM only. Phase Dynamics GaussianHMM (v2.0.9 main path) uses `register_buffer` instead (T-1), so freezing is not needed. This function remains for legacy test coverage only.

    Freeze all HMM parameters in-place for Stage 2 (CG-Mamba M1.7).

    PLAN v2.0.8c §5.1 EB-3 spec requires HMM weights to be FROZEN during
    Stage 2 backbone training. After Stage 1 produces an HMM checkpoint,
    Stage 2 only learns:
      - state_embeddings (via PhaseModule.state_embed)
      - gate_proj (per CGMambaBlock)
      - backbone / decoder
    HMM (NeuralSwitchingVARHMM) provides γ posteriors as fixed inputs.

    ⚠️ Silent-bug risk if NOT called: passing `model.parameters()` to a single
    optimizer would silently train the HMM in Stage 2, contaminating phase
    semantics. This helper makes the freeze explicit and asserts the result.

    M1.7 usage pattern:
        # After Stage 1 ckpt loaded into `model.hmm`:
        n_frozen = freeze_hmm_for_stage2(model.hmm)
        assert n_frozen > 0, "Expected HMM params to freeze"

        # Then build optimizer over OTHER components only:
        optimizer = torch.optim.AdamW([
            {"params": state_embed.parameters(), "lr": 1e-6},
            {"params": gate_proj.parameters(),   "lr": 1e-3},
            {"params": backbone.parameters(),    "lr": 5e-5},
            {"params": decoder.parameters(),     "lr": 5e-5},
        ])

    Args:
        model: fitted NeuralSwitchingVARHMM (loaded from Stage 1 ckpt)

    Returns:
        Number of params frozen (count of model.parameters() with
        requires_grad set False by this call).
    """
    n_frozen = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n_frozen += 1

    # Post-condition: ALL params frozen (Stage 2 invariant)
    remaining_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert remaining_trainable == 0, (
        f"freeze_hmm_for_stage2 failed: {remaining_trainable} HMM params still "
        f"trainable. Stage 2 spec violation (PLAN §5.1 EB-3)."
    )
    return n_frozen
