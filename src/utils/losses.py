"""Loss functions for CG-Mamba Stage 2 training (M1.7).

PLAN v2.0.9 §5.1 D.5.2: `L = MSE + 0.3 · MASE`, λ_sparse=0 default.

Functions:
    compute_seasonal_mae(df, norm, season_lag=52) → scalar Tensor
        Pre-compute seasonal naive MAE on the train split (normalized space).
        Used as MASE denominator. Includes epiweek gap guard (M1.7 review fix).
    mase_loss(pred, target, seasonal_mae) → scalar Tensor
    cg_mamba_loss(pred, target, seasonal_mae, lambda_mase=0.3) → scalar Tensor

Design notes:
    - All losses operate in NORMALIZED space (z-score for ili_weighted_pct), so
      MASE denominator and MSE/MAE numerators share the same scale (R-2).
    - compute_seasonal_mae touches ILI columns. It is intentionally kept here
      (NOT in m1_7_env_pretrain.py) to preserve the "Env pretrain is ILI-blind"
      structural guarantee (PLAN §5.1 A-2).
    - λ_sparse binary_entropy_loss is NOT implemented here — A-4 ablation only,
      out of M1.7 scope.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.loader import is_consecutive_epiweek


def compute_seasonal_mae(
    df: pd.DataFrame,
    norm: dict,
    season_lag: int = 52,
) -> torch.Tensor:
    """Seasonal naive MAE on the train split, in normalized space.

    Seasonal naive forecast: y_hat(t) = y(t - season_lag). This serves as the
    MASE denominator (forecast benchmark). Computing it in normalized space
    keeps the scale consistent with predictions (R-2).

    M1.7 review fix (Finding 2 — epiweek gap guard, M-2 optimization):
        Indices (i, i - season_lag) are only compared if the entire chain
        between them is consecutive in the MMWR calendar (no gaps inside
        the 52-week span). Train split (seg2-only, 2002-W40 ~ 2018-W39) is
        gap-free, but we add the guard for robustness against future
        split-policy changes.

        M-2 optimization: O(N · season_lag) double loop replaced with O(N)
        prefix-sum scan. Pre-compute `consec[j] = is_consecutive(eps[j], eps[j+1])`
        once, then `gap_cumsum[k]` gives the number of gaps in [0, k).
        Any [i-season_lag, i) span has zero gaps iff
            gap_cumsum[i-1] - gap_cumsum[i-season_lag-1] == 0
        (with appropriate boundary handling for i-season_lag == 0).

    M-3: If any gap-spanning pairs are skipped, emit a `warnings.warn` so the
        diagnostic is visible (previously `skipped_gap` was computed but never
        exposed outside the function).

    Args:
        df:          full dataset DataFrame (with 'split', 'epiweek',
                     'ili_weighted_pct' columns).
        norm:        normalization params dict (from normalization_params.json).
        season_lag:  lag in weeks (default 52, annual seasonality).

    Returns:
        Scalar torch.Tensor — seasonal naive MAE (normalized space).
    """
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    epiweeks = train_df["epiweek"].astype(int).to_numpy()
    vals_z = (
        (train_df["ili_weighted_pct"].to_numpy()
         - norm["ili_weighted_pct"]["mean"])
        / norm["ili_weighted_pct"]["std"]
    )
    N = len(train_df)

    # M-2: single O(N) scan to build the consecutive-pair boolean array
    # `consec[j] == True` iff epiweeks[j] → epiweeks[j+1] is one MMWR step.
    consec = np.fromiter(
        (
            is_consecutive_epiweek(int(epiweeks[j]), int(epiweeks[j + 1]))
            for j in range(N - 1)
        ),
        dtype=bool,
        count=N - 1,
    )
    # `gap_cumsum[k]` = number of gaps inside indices [0, k).
    # gap_cumsum is length N (gap_cumsum[0] = 0, gap_cumsum[N] = total_gaps),
    # and the span [start, end) contains gap_cumsum[end] - gap_cumsum[start] gaps.
    gap_cumsum = np.concatenate([[0], np.cumsum(~consec)])

    diffs = []
    skipped_gap = 0
    for i in range(season_lag, N):
        # The 52-week chain (i - season_lag) → i covers adjacency pairs
        # at indices [i - season_lag, i), so gap count in that range is:
        gaps_in_span = int(gap_cumsum[i] - gap_cumsum[i - season_lag])
        if gaps_in_span > 0:
            skipped_gap += 1
            continue
        diffs.append(abs(vals_z[i] - vals_z[i - season_lag]))

    if len(diffs) == 0:
        raise RuntimeError(
            f"compute_seasonal_mae: no valid (i, i-{season_lag}) pairs found "
            f"in train split (gap-skipped {skipped_gap}, len {N}). "
            f"Check season_lag and train split contiguity."
        )

    # M-3: surface the diagnostic so callers know the guard fired. The exact
    # `expected_skipped` count for the current dataset depends on where the
    # gap falls in the train split:
    #   - WeeklyDataset.train = seg1 (200140-200220, 33 rows) + seg2 (200240-
    #     201839, 835 rows), with a 19-week CDC anomaly gap at row 32 → 33.
    #   - For season_lag=52, the 52-week chain check fails iff the gap is
    #     inside [i - season_lag, i), which is true for i ∈ [season_lag,
    #     gap_idx + season_lag] = [52, 84] → **33 expected skips**.
    # If `skipped_gap` deviates from 33, the train-split structure changed
    # (e.g., a new gap was introduced) and the caller should investigate.
    EXPECTED_GAP_SKIP_FOR_SEG1_PLUS_SEG2_AT_52WEEK = 33
    if skipped_gap > 0 and skipped_gap != EXPECTED_GAP_SKIP_FOR_SEG1_PLUS_SEG2_AT_52WEEK:
        warnings.warn(
            f"compute_seasonal_mae: skipped {skipped_gap} gap-spanning pairs "
            f"(expected {EXPECTED_GAP_SKIP_FOR_SEG1_PLUS_SEG2_AT_52WEEK} for "
            f"seg1+seg2 train with CDC-2002 gap at season_lag={season_lag}). "
            f"Train split structure may have changed; investigate."
        )

    return torch.tensor(float(np.mean(diffs)), dtype=torch.float32)


def mase_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    seasonal_mae: torch.Tensor,
) -> torch.Tensor:
    """Mean Absolute Scaled Error.

    MASE = MAE(pred, target) / seasonal_naive_MAE_train.
    MASE < 1 means the model beats the seasonal naive benchmark.

    Args:
        pred:         [B, H] predictions (normalized space).
        target:       [B, H] targets (normalized space).
        seasonal_mae: scalar — pre-computed seasonal naive MAE on train
                      (normalized space, from compute_seasonal_mae).

    Returns:
        Scalar MASE loss.
    """
    mae = (pred - target).abs().mean()
    return mae / seasonal_mae.clamp(min=1e-8)


def cg_mamba_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    seasonal_mae: torch.Tensor,
    lambda_mase: float = 0.3,
) -> torch.Tensor:
    """CG-Mamba composite loss: MSE + λ·MASE (PLAN D.5.2 active spec).

    Args:
        pred:         [B, H] predictions (normalized space, R-2).
        target:       [B, H] targets (normalized space).
        seasonal_mae: scalar — MASE denominator (see compute_seasonal_mae).
        lambda_mase:  MASE weight (default 0.3, PLAN D.5.2).

    Returns:
        Scalar composite loss.
    """
    mse = F.mse_loss(pred, target)
    mase = mase_loss(pred, target, seasonal_mae)
    return mse + lambda_mase * mase
