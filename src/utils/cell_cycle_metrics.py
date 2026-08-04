"""Cell cycle domain-specific evaluation metrics.

Used by:
    - CellCycleForecaster training loop (Step 4): per-epoch monitoring
    - K=3 vs K=4 BIC + emission ablation scripts (Step 5): reporting
    - paper §results: phase classification accuracy, cyclic MSE

Metrics:
    1. phase_classification_accuracy — Viterbi-predicted state labels vs
       ground-truth phase labels with optimal permutation alignment.
       HMM states are unordered, so we brute-force all K! permutations
       (feasible for K ≤ 8) and report the best accuracy.

    2. gene_expression_mse / gene_expression_mae — gene-level prediction
       error, optionally restricted to a marker subset for interpretability.

    3. phase_angle_error — circular distance between predicted phase angle
       (γ-weighted) and ground-truth angle (radians, wraps at 2π). Returns
       mean absolute error in degrees for readability.

    4. cyclic_correlation — Pearson correlation between predicted and
       observed time series after subtracting a moving-average trend.
       Captures whether the model gets the *cyclic shape* right even if
       absolute values drift.

Reference:
    Whitfield ML et al. (2002), Bar-Joseph Z et al. (2008).
"""
from __future__ import annotations

from itertools import permutations
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


# ──────────────────────────────────────────────────────────────────
# 1. Phase classification accuracy (with permutation alignment)
# ──────────────────────────────────────────────────────────────────

def phase_classification_accuracy(
    pred_states: np.ndarray,
    true_states: np.ndarray,
    K: int,
) -> tuple[float, tuple[int, ...]]:
    """Best accuracy under optimal HMM-state ↔ true-phase permutation.

    HMM state labels are arbitrary (state 0 from one fit may correspond
    to phase 2 from the ground truth). We brute-force all K! relabelings
    of `pred_states` and return the best match.

    Args:
        pred_states: [T] integer Viterbi predictions in [0, K).
        true_states: [T] integer ground-truth phase labels in [0, K).
        K:           number of phases (must be K ≤ 8 for brute force).

    Returns:
        accuracy:        best fraction of timepoints correctly classified.
        best_permutation: K-tuple p such that p[pred[t]] is the relabeled
                          prediction. Useful for downstream interpretation.
    """
    assert pred_states.shape == true_states.shape, (
        f"Shape mismatch: pred {pred_states.shape} vs true {true_states.shape}"
    )
    assert K <= 8, f"K={K} too large for brute-force permutation (K! = {math.factorial(K)})"
    pred = np.asarray(pred_states, dtype=int)
    true = np.asarray(true_states, dtype=int)

    best_acc = -1.0
    best_perm: tuple[int, ...] = tuple(range(K))
    for perm in permutations(range(K)):
        remapped = np.asarray(perm, dtype=int)[pred]
        acc = float((remapped == true).mean())
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_acc, best_perm


# ──────────────────────────────────────────────────────────────────
# 2. Gene expression MSE / MAE (with optional marker subset)
# ──────────────────────────────────────────────────────────────────

def gene_expression_mse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    gene_indices: list[int] | None = None,
) -> float:
    """Mean squared error over gene expression predictions.

    Args:
        y_pred:       [T, G] or [T, horizon, G] predicted expression.
        y_true:       same shape as y_pred.
        gene_indices: optional subset of columns to evaluate (e.g., the
                      16 marker genes for marker-only MSE). If None,
                      evaluates over all columns.

    Returns:
        Scalar MSE.
    """
    assert y_pred.shape == y_true.shape, (
        f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
    )
    if gene_indices is not None:
        y_pred = y_pred[..., gene_indices]
        y_true = y_true[..., gene_indices]
    return float(np.mean((y_pred - y_true) ** 2))


def gene_expression_mae(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    gene_indices: list[int] | None = None,
) -> float:
    """Mean absolute error over gene expression predictions.

    Same args as gene_expression_mse. MAE is reported alongside MSE in
    the paper because gene expression has outliers (highly variable
    genes) where MSE over-weights.
    """
    assert y_pred.shape == y_true.shape, (
        f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
    )
    if gene_indices is not None:
        y_pred = y_pred[..., gene_indices]
        y_true = y_true[..., gene_indices]
    return float(np.mean(np.abs(y_pred - y_true)))


# ──────────────────────────────────────────────────────────────────
# 3. Phase angle error (circular distance)
# ──────────────────────────────────────────────────────────────────

def phase_angle_from_posterior(gamma: np.ndarray, K: int) -> np.ndarray:
    """Compute γ-weighted phase angle [0, 2π) from posterior.

    Each HMM state k is anchored at the **center** of its true range on the
    unit circle. State k covers phase_angle ∈ [k·2π/K, (k+1)·2π/K), so its
    center anchor is

        θ_k = (k + 0.5) · 2π / K.

    Anchoring at the range-start (θ_k = k·2π/K) would create a systematic
    backward offset: for state-k timepoints uniformly distributed in
    [θ_k, θ_k + 2π/K), the predicted angle (=θ_k) lags the true cell-center
    by π/K — e.g., 45° for K=4, yielding mean |error| of 45° even with a
    perfectly-confident posterior. Center-anchoring halves this systematic
    error to a non-systematic ~22.5° residual driven solely by within-cell
    uniform sampling (E[|U[-π/2K, π/2K)|] = π/(4K) in radians).

    The predicted phase angle is the circular mean of these anchors weighted
    by the posterior:

        cos_avg = Σ_k γ_k · cos(θ_k)
        sin_avg = Σ_k γ_k · sin(θ_k)
        angle   = atan2(sin_avg, cos_avg)  ∈ [-π, π) → mapped to [0, 2π)

    Args:
        gamma: [T, K] posterior probabilities.
        K:     number of phases.

    Returns:
        angle: [T] phase angle in radians, [0, 2π).
    """
    assert gamma.shape[-1] == K, f"gamma last dim {gamma.shape[-1]} != K={K}"
    # Center-anchor: (k + 0.5) / K to align with phase-range midpoint.
    anchors = 2 * np.pi * (np.arange(K) + 0.5) / K       # [K]
    cos_avg = (gamma * np.cos(anchors)).sum(axis=-1)
    sin_avg = (gamma * np.sin(anchors)).sum(axis=-1)
    angle = np.arctan2(sin_avg, cos_avg)                 # [-π, π)
    return np.mod(angle, 2 * np.pi)                      # [0, 2π)


def circular_distance(angle_a: np.ndarray, angle_b: np.ndarray) -> np.ndarray:
    """Smallest angular distance in [0, π] between two angles in radians.

    Element-wise. Handles wrap-around: e.g., dist(0.1, 6.2) ≈ 0.183
    (NOT 6.1) because going the other way around the circle is shorter.

    Args:
        angle_a, angle_b: arrays in radians (any matching shape).

    Returns:
        distances in [0, π].
    """
    diff = np.mod(angle_a - angle_b, 2 * np.pi)
    return np.minimum(diff, 2 * np.pi - diff)


def phase_angle_error(
    gamma: np.ndarray,
    true_angle: np.ndarray,
    K: int,
    in_degrees: bool = True,
) -> float:
    """Mean circular distance between γ-derived phase angle and ground truth.

    Args:
        gamma:      [T, K] posterior probabilities.
        true_angle: [T] ground-truth phase angle in radians [0, 2π).
        K:          number of HMM states.
        in_degrees: return error in degrees (default) vs radians.

    Returns:
        Mean absolute circular error.
    """
    pred_angle = phase_angle_from_posterior(gamma, K)        # [T]
    dist = circular_distance(pred_angle, true_angle)         # [T] in radians
    mean_dist = float(dist.mean())
    return mean_dist * 180.0 / math.pi if in_degrees else mean_dist


# ──────────────────────────────────────────────────────────────────
# 4. Cyclic correlation (detrended Pearson)
# ──────────────────────────────────────────────────────────────────

def cyclic_correlation(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    period: int,
) -> float:
    """Pearson correlation after subtracting a centered moving-average trend.

    Captures whether the predicted curve has the right *cyclic shape* even
    if the absolute level drifts. The moving-average window length equals
    `period` (one full cycle) so the trend captures slow drift.

    Args:
        y_pred: [T] or [T, G] predicted time series.
        y_true: same shape as y_pred.
        period: cycle period in timepoints (window length for detrending).

    Returns:
        Mean correlation across columns (single float).
        Returns 0.0 if either detrended series is constant (zero variance).
    """
    assert y_pred.shape == y_true.shape
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
        y_true = y_true[:, None]
    T, G = y_pred.shape

    pred_dt = y_pred - _moving_average_trend(y_pred, period)
    true_dt = y_true - _moving_average_trend(y_true, period)

    corrs = []
    for g in range(G):
        a = pred_dt[:, g]
        b = true_dt[:, g]
        sa = a.std()
        sb = b.std()
        if sa < 1e-12 or sb < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs))


# ──────────────────────────────────────────────────────────────────
# 5. Per-gene Pearson / Spearman correlation (auxiliary metrics, 2.6)
# ──────────────────────────────────────────────────────────────────
# Direction Message v2 §17 update: MSE alone misses "value-converges-but-
# shape-broken" failure modes. Per-gene rank/value correlations catch
# these and align with cell-cycle gene-expression forecasting standards
# (Bar-Joseph 2008, Sherlock 2002).

def gene_pearson_median(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Median per-gene Pearson correlation across columns.

    For each gene column g, compute Pearson r between y_pred[:, g] and
    y_true[:, g] across timepoints. Return the median r across all genes —
    robust to a few badly-predicted genes overwhelming the mean.

    Args:
        y_pred, y_true: [T, G] matched-shape arrays.

    Returns:
        Median per-gene Pearson r ∈ [-1, 1]. Returns 0.0 for any single
        gene whose pred or true is constant (zero variance).
    """
    assert y_pred.shape == y_true.shape
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
        y_true = y_true[:, None]
    rs = []
    for g in range(y_pred.shape[1]):
        a = y_pred[:, g]
        b = y_true[:, g]
        if a.std() < 1e-12 or b.std() < 1e-12:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.median(rs))


def gene_spearman_median(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Median per-gene Spearman rank correlation across columns.

    Spearman ρ on ranks is robust to outliers — useful when a few timepoints
    have extreme expression values that would dominate Pearson.

    Args:
        y_pred, y_true: [T, G] matched-shape arrays.

    Returns:
        Median per-gene Spearman ρ ∈ [-1, 1].
    """
    assert y_pred.shape == y_true.shape
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
        y_true = y_true[:, None]
    rhos = []
    for g in range(y_pred.shape[1]):
        a_rank = np.argsort(np.argsort(y_pred[:, g]))
        b_rank = np.argsort(np.argsort(y_true[:, g]))
        if a_rank.std() < 1e-12 or b_rank.std() < 1e-12:
            rhos.append(0.0)
        else:
            rhos.append(float(np.corrcoef(a_rank, b_rank)[0, 1]))
    return float(np.median(rhos))


# ──────────────────────────────────────────────────────────────────
# 6. Dynamic time warping (DTW) — phase-shift-tolerant distance
# ──────────────────────────────────────────────────────────────────

def dtw_distance(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    window: int | None = None,
) -> float:
    """Mean per-gene DTW distance — captures shape match modulo time warping.

    DTW measures alignment cost when the time axis is allowed to stretch /
    compress. Cell cycle data has variable phase durations across cells,
    making strict timepoint-wise MSE overly sensitive to slight phase shifts.
    DTW captures whether the predicted *shape* (peaks, troughs, sequence)
    matches even if the timing is offset.

    Implementation: Sakoe-Chiba band-restricted DTW. O(T·window) per gene.
    For T=22 and window=5 this is ~110 cells per gene → trivially fast.

    Args:
        y_pred:  [T_pred, G] or [T_pred] predicted series.
        y_true:  [T_true, G] or [T_true] target series.
        window:  Sakoe-Chiba band radius (default None = full DTW). For
                 cell cycle with cycle_period=22, window=5 (~¼ cycle)
                 prevents pathological warps.

    Returns:
        Mean DTW distance across genes. Lower is better.
    """
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
        y_true = y_true[:, None]
    T_p, G = y_pred.shape
    T_t = y_true.shape[0]
    assert y_true.shape[1] == G, (
        f"Gene dim mismatch: pred {G} vs true {y_true.shape[1]}"
    )

    total = 0.0
    for g in range(G):
        total += _dtw_single(y_pred[:, g], y_true[:, g], window=window)
    return float(total / G)


def _dtw_single(
    a: np.ndarray, b: np.ndarray, window: int | None = None,
) -> float:
    """Single-sequence DTW with optional Sakoe-Chiba band restriction."""
    n, m = len(a), len(b)
    if window is None:
        window = max(n, m)
    else:
        window = max(window, abs(n - m))

    INF = float("inf")
    cost = np.full((n + 1, m + 1), INF, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - window)
        j_hi = min(m, i + window)
        for j in range(j_lo, j_hi + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


# ──────────────────────────────────────────────────────────────────
# 7. Peak detection F1 — peak timing accuracy
# ──────────────────────────────────────────────────────────────────

def peak_detection_f1(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    tolerance: int = 2,
    prominence: float | None = None,
) -> float:
    """F1 score of peak detection (predicted peaks vs true peaks).

    Each gene's time series is scanned for local maxima ("peaks"). A
    predicted peak is a true positive if a true peak exists within
    `tolerance` timesteps; otherwise false positive. True peaks with no
    nearby predicted peak are false negatives.

    Args:
        y_pred, y_true: [T, G] matched-shape arrays.
        tolerance:      max distance (timesteps) for peak match.
        prominence:     scipy.signal.find_peaks prominence parameter. None
                        defaults to 0.1 × per-gene std (data-adaptive).

    Returns:
        Macro-averaged F1 score across genes ∈ [0, 1].
    """
    assert y_pred.shape == y_true.shape
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
        y_true = y_true[:, None]
    T, G = y_pred.shape

    f1s = []
    for g in range(G):
        prom_g = (
            prominence if prominence is not None
            else max(0.1 * float(y_true[:, g].std()), 1e-6)
        )
        peaks_pred, _ = find_peaks(y_pred[:, g], prominence=prom_g)
        peaks_true, _ = find_peaks(y_true[:, g], prominence=prom_g)

        if len(peaks_pred) == 0 and len(peaks_true) == 0:
            f1s.append(1.0)   # both empty = perfect (definitional)
            continue
        if len(peaks_pred) == 0 or len(peaks_true) == 0:
            f1s.append(0.0)
            continue

        # Match predicted peaks to true peaks within tolerance.
        # Greedy: each true peak claims the closest unmatched pred peak.
        matched_pred = set()
        matched_true = 0
        for tp in peaks_true:
            best_d = None
            best_pp = None
            for pp in peaks_pred:
                if pp in matched_pred:
                    continue
                d = abs(int(tp) - int(pp))
                if d <= tolerance and (best_d is None or d < best_d):
                    best_d = d
                    best_pp = pp
            if best_pp is not None:
                matched_pred.add(best_pp)
                matched_true += 1

        tp = matched_true
        fp = len(peaks_pred) - len(matched_pred)
        fn = len(peaks_true) - matched_true
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# ──────────────────────────────────────────────────────────────────
# 8. Hungarian state alignment (cross-K / cross-seed comparison)
# ──────────────────────────────────────────────────────────────────

def align_states_hungarian(
    means_a: np.ndarray, means_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one state matching between two HMMs via Hungarian.

    Two HMMs fit to similar data (same data, different K; same K, different
    seed; or same fit applied to two experiments) typically use different
    state-index conventions: state 0 from HMM-A may correspond to state 2
    from HMM-B. To compare metrics (occupancy, Viterbi accuracy, BIC) it is
    necessary to align state indices first.

    Algorithm: O(K^3) linear assignment minimizing the sum of pairwise
    Euclidean distances between μ vectors.

    Args:
        means_a: [K_a, V] state means of HMM-A.
        means_b: [K_b, V] state means of HMM-B. V must match.
                 K_a and K_b need NOT match — when K_b < K_a, the extra
                 a-states are unmatched (returned as -1 in `mapping_b`).

    Returns:
        a_indices:  [K_min] array of indices into HMM-A's states.
        b_indices:  [K_min] array of indices into HMM-B's states (paired).
                    means_a[a_indices[i]] is matched to means_b[b_indices[i]].
    """
    means_a = np.asarray(means_a, dtype=np.float64)
    means_b = np.asarray(means_b, dtype=np.float64)
    assert means_a.ndim == 2 and means_b.ndim == 2
    assert means_a.shape[1] == means_b.shape[1], (
        f"Feature dim mismatch: {means_a.shape[1]} vs {means_b.shape[1]}"
    )

    # Pairwise Euclidean distance matrix [K_a, K_b]
    diff = means_a[:, None, :] - means_b[None, :, :]
    cost = np.linalg.norm(diff, axis=-1)

    row_idx, col_idx = linear_sum_assignment(cost)
    return row_idx, col_idx


def _moving_average_trend(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding (reflect mode).

    Args:
        y:      [T, G] time series.
        window: window size (must be >= 1).

    Returns:
        trend: [T, G] same shape, smoothed via centered MA.
    """
    assert window >= 1
    T, G = y.shape
    if window == 1:
        return y.copy()

    half = window // 2
    # Reflect padding on both ends so edge timepoints aren't biased downward.
    padded = np.pad(y, ((half, half), (0, 0)), mode="reflect")
    kernel = np.ones(window) / window
    trend = np.empty_like(y)
    for g in range(G):
        # 'valid' convolution on padded → length T
        trend[:, g] = np.convolve(padded[:, g], kernel, mode="valid")[:T]
    return trend
