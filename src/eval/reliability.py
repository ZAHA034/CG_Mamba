"""Reliability diagram + Wilcoxon signed-rank for WIS Phase D analysis.

PLAN Q5: per-horizon 0.05-bin reliability (20 bins × 4 horizons).
PLAN Phase D: Wilcoxon signed-rank with Bonferroni correction across baselines.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from src.eval.wis import REQUIRED_QUANTILES


def reliability_per_horizon(
    quantile_forecasts: dict[float, np.ndarray],
    y: np.ndarray,
    bin_width: float = 0.05,
) -> dict[int, dict[str, np.ndarray]]:
    """Per-horizon reliability for each nominal quantile level.

    For each horizon h and quantile level q ∈ REQUIRED_QUANTILES, compute
    empirical coverage = fraction of y_h <= q_hat_h(q). Plot vs nominal q;
    perfect calibration lies on the y = q diagonal.

    Args:
        quantile_forecasts: dict q → array [N, H].
        y: array [N, H].
        bin_width: nominal-q bin width for grouping (default 0.05 — PLAN Q5).

    Returns:
        dict h → {"nominal": [Q], "empirical": [Q], "bin_centers": [B], ...}.
    """
    y = np.asarray(y)
    H = y.shape[1]
    out: dict[int, dict[str, np.ndarray]] = {}
    for h in range(H):
        nominal = np.array(REQUIRED_QUANTILES)
        empirical = np.array([
            float(np.mean(y[:, h] <= quantile_forecasts[q][:, h]))
            for q in REQUIRED_QUANTILES
        ])
        n_bins = int(round(1.0 / bin_width))
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        binned = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = (nominal >= bin_edges[i]) & (nominal < bin_edges[i + 1])
            if mask.any():
                binned[i] = empirical[mask].mean()
        out[h] = {
            "nominal": nominal,
            "empirical": empirical,
            "bin_centers": bin_centers,
            "bin_empirical": binned,
        }
    return out


def wilcoxon_signed_rank_per_horizon(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "less",
) -> list[dict[str, float]]:
    """Per-horizon Wilcoxon signed-rank between two paired WIS arrays.

    Args:
        scores_a, scores_b: shape [N, H] per-observation WIS for two models.
        alternative: 'less' → test H1: scores_a < scores_b (a better than b).

    Returns:
        Length-H list of {'statistic', 'pvalue', 'n', 'mean_diff'}.
    """
    if scores_a.shape != scores_b.shape:
        raise ValueError(f"shape mismatch: {scores_a.shape} vs {scores_b.shape}")
    H = scores_a.shape[1]
    out = []
    for h in range(H):
        diff = scores_a[:, h] - scores_b[:, h]
        nonzero = diff[diff != 0.0]
        if len(nonzero) < 2:
            out.append({"statistic": np.nan, "pvalue": np.nan,
                        "n": int(len(nonzero)), "mean_diff": float(diff.mean())})
            continue
        stat, p = wilcoxon(scores_a[:, h], scores_b[:, h],
                           alternative=alternative, zero_method="wilcox")
        out.append({
            "statistic": float(stat),
            "pvalue": float(p),
            "n": int(len(nonzero)),
            "mean_diff": float(diff.mean()),
        })
    return out


def bonferroni_adjust(pvalues: list[float], k: int | None = None) -> list[float]:
    """Bonferroni-correct a list of p-values for k comparisons.

    PLAN Phase D: CG-Mamba vs each of 10 baselines per horizon → k=10.
    """
    k = len(pvalues) if k is None else k
    return [min(1.0, p * k) for p in pvalues]
