"""Sliding-window construction + k-fold cross-validation for small time series.

Used by Step 4 (CellCycleForecaster) to address two Direction Message v2 §17
gaps identified in the 5th review:

  - 2.1 sliding window: Exp3 has T=48 timepoints. A single contiguous window
    of length L_win=48 would yield exactly one training sample. Sliding the
    window over the time axis gives multiple training pairs (x_window,
    y_future) suitable for SGD.

  - 2.2 train/val split: with only ~20 sliding windows from Exp3, a single
    held-out validation set has high variance. k-fold CV (default k=5)
    averages over folds, yielding a more stable early-stopping signal.

API:
    sliding_window_splits(T, L_win, horizon[, stride])
        → list[(t_start, t_end_x, t_end_y)] for each window

    kfold_window_indices(n_windows, n_folds=5, seed=42)
        → list[(train_idx, val_idx)] for each fold, using contiguous folds
          (sliding windows preserve temporal locality, so contiguous folds
          avoid information leakage between train and val).
"""
from __future__ import annotations

import numpy as np


def sliding_window_splits(
    T: int,
    L_win: int,
    horizon: int,
    stride: int = 1,
) -> list[tuple[int, int, int]]:
    """Construct sliding-window (x, y) index triples over a time series of length T.

    Each window covers:
        x: time indices [t, t + L_win)         — input
        y: time indices [t + L_win, t + L_win + horizon)  — forecast target

    Walked with `stride` between consecutive windows.

    Args:
        T:       total time series length (e.g., 48 for Exp3).
        L_win:   window length (lookback context).
        horizon: forecast length (longest horizon to support).
        stride:  step between consecutive windows (default 1).

    Returns:
        List of (t_start, t_end_x, t_end_y) triples:
          t_start ≤ t < t_end_x for x;  t_end_x ≤ t < t_end_y for y.
        t_end_y - t_start = L_win + horizon for every window.

    Raises:
        ValueError: if L_win + horizon > T (no valid window possible).

    Example:
        >>> sliding_window_splits(T=48, L_win=24, horizon=5)[:3]
        [(0, 24, 29), (1, 25, 30), (2, 26, 31)]
        >>> len(sliding_window_splits(T=48, L_win=24, horizon=5))
        20
    """
    if L_win < 1:
        raise ValueError(f"L_win must be >= 1, got {L_win}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if L_win + horizon > T:
        raise ValueError(
            f"L_win ({L_win}) + horizon ({horizon}) > T ({T}); "
            f"no valid sliding window exists."
        )

    splits: list[tuple[int, int, int]] = []
    t = 0
    while t + L_win + horizon <= T:
        splits.append((t, t + L_win, t + L_win + horizon))
        t += stride
    return splits


def kfold_window_indices(
    n_windows: int,
    n_folds: int = 5,
    contiguous: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """k-fold CV indices for an ordered list of sliding windows.

    Two modes:
      - `contiguous=True` (default): each fold is a contiguous block of
        consecutive window indices. Preserves temporal ordering between
        train and val (important for time series — random shuffle would
        leak future-into-past via overlapping sliding windows).
      - `contiguous=False`: round-robin assignment (window i → fold i % k).
        Less leakage-safe but balances class distribution if folds have
        differing characteristics.

    Args:
        n_windows: total number of sliding windows (returned by
                   sliding_window_splits).
        n_folds:   number of CV folds (default 5).
        contiguous: True = contiguous folds; False = round-robin.

    Returns:
        List of `n_folds` (train_idx, val_idx) tuples. `train_idx` and
        `val_idx` are np.ndarray of dtype int64, disjoint, union = all
        n_windows indices.

    Raises:
        ValueError: if n_windows < n_folds or n_folds < 2.

    Example:
        >>> folds = kfold_window_indices(20, n_folds=5)
        >>> len(folds), [len(tr) + len(va) for tr, va in folds]
        (5, [20, 20, 20, 20, 20])
        >>> [len(va) for _, va in folds]
        [4, 4, 4, 4, 4]
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_windows < n_folds:
        raise ValueError(
            f"n_windows ({n_windows}) < n_folds ({n_folds}); "
            f"cannot construct CV folds."
        )

    all_idx = np.arange(n_windows, dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    if contiguous:
        # Equal-ish contiguous chunks.
        fold_sizes = np.full(n_folds, n_windows // n_folds, dtype=int)
        fold_sizes[: n_windows % n_folds] += 1  # distribute remainder
        boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])
        for f in range(n_folds):
            val_idx = all_idx[boundaries[f]: boundaries[f + 1]]
            train_idx = np.concatenate([
                all_idx[: boundaries[f]],
                all_idx[boundaries[f + 1]:],
            ])
            folds.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))
    else:
        for f in range(n_folds):
            mask = (all_idx % n_folds) == f
            val_idx = all_idx[mask]
            train_idx = all_idx[~mask]
            folds.append((train_idx, val_idx))

    return folds
