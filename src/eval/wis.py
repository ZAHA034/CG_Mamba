"""Weighted Interval Score (WIS) — Bracher et al. 2021 PLOS Comp Bio.

Reference:
    Bracher J, Ray EL, Gneiting T, Reich NG (2021). "Evaluating epidemic
    forecasts in an interval format." PLoS Comput Biol 17(2): e1008618.
    eq. (3) — WIS = (1 / (K + 0.5)) * (0.5 * |y - m| + Σ_k (α_k/2) * IS_{α_k})

CDC FluSight / COVID-19 Forecast Hub 표준: K=11 central prediction intervals
plus median → 23 quantile levels total. Used in:
    - Cramer et al. 2022 PNAS (90+ heterogeneous COVID forecasting models)
    - Reich et al. 2019 PNAS (CDC FluSight ensemble)
    - Borchering et al. 2024 Nat Comms (latest seasonal flu challenge)
"""
from __future__ import annotations

import numpy as np


REQUIRED_QUANTILES = (
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99,
)

ALPHA_LEVELS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

INTERVAL_PAIRS = tuple(
    (round(a / 2, 4), round(1 - a / 2, 4)) for a in ALPHA_LEVELS
)


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    """Bracher 2021 eq. (1) — interval score for a single (1-α) PI.

    IS_α(F, y) = (u - l) + (2/α) * (l - y) * 1{y<l} + (2/α) * (y - u) * 1{y>u}

    Lower is better. All inputs are element-wise broadcastable arrays.
    """
    width = upper - lower
    under = np.where(y < lower, (2.0 / alpha) * (lower - y), 0.0)
    over = np.where(y > upper, (2.0 / alpha) * (y - upper), 0.0)
    return width + under + over


def wis(
    y: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> np.ndarray:
    """Weighted Interval Score — Bracher 2021 eq. (3).

    WIS = (1 / (K + 0.5)) * (0.5 * |y - m| + Σ_k (α_k/2) * IS_{α_k}(F, y))

    Args:
        y: shape [N] truth values.
        quantile_forecasts: dict mapping q ∈ REQUIRED_QUANTILES → array [N].
            Must contain all 23 quantile levels.

    Returns:
        shape [N] per-observation WIS (lower is better).
    """
    missing = [q for q in REQUIRED_QUANTILES if q not in quantile_forecasts]
    if missing:
        raise ValueError(f"quantile_forecasts missing levels: {missing}")

    y = np.asarray(y, dtype=np.float64)
    median = np.asarray(quantile_forecasts[0.5], dtype=np.float64)

    K = len(ALPHA_LEVELS)
    weighted_sum = 0.5 * np.abs(y - median)
    for alpha, (q_lo, q_hi) in zip(ALPHA_LEVELS, INTERVAL_PAIRS):
        lo = np.asarray(quantile_forecasts[q_lo], dtype=np.float64)
        hi = np.asarray(quantile_forecasts[q_hi], dtype=np.float64)
        weighted_sum = weighted_sum + (alpha / 2.0) * interval_score(y, lo, hi, alpha)

    return weighted_sum / (K + 0.5)


def wis_decomposed(
    y: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> dict[str, np.ndarray]:
    """WIS decomposed into (dispersion, under-prediction, over-prediction).

    Bracher 2021 §2.3 — diagnostic for sharpness vs calibration trade-off.
    All three terms sum to wis(y, quantile_forecasts).
    """
    y = np.asarray(y, dtype=np.float64)
    median = np.asarray(quantile_forecasts[0.5], dtype=np.float64)

    K = len(ALPHA_LEVELS)
    Z = K + 0.5

    dispersion = 0.5 * np.abs(y - median) / Z
    under = np.zeros_like(y, dtype=np.float64)
    over = np.zeros_like(y, dtype=np.float64)

    for alpha, (q_lo, q_hi) in zip(ALPHA_LEVELS, INTERVAL_PAIRS):
        lo = np.asarray(quantile_forecasts[q_lo], dtype=np.float64)
        hi = np.asarray(quantile_forecasts[q_hi], dtype=np.float64)
        w = alpha / (2.0 * Z)
        dispersion = dispersion + w * (hi - lo)
        under = under + w * (2.0 / alpha) * np.maximum(lo - y, 0.0)
        over = over + w * (2.0 / alpha) * np.maximum(y - hi, 0.0)

    return {"dispersion": dispersion, "under": under, "over": over,
            "total": dispersion + under + over}


def quantile_loss(y: np.ndarray, q_hat: np.ndarray, q: float) -> np.ndarray:
    """Pinball loss for a single quantile q ∈ (0, 1).

    QL_q(F, y) = (y - q_hat) * (q - 1{y < q_hat})
    Used for the properscoring-independent WIS cross-check:
    Bracher 2021 Appendix shows WIS = (2 / (2K+1)) * Σ_{k=0}^{2K} QL_{q_k}(F, y)
    summed over the 2K+1 = 23 quantile levels.
    """
    y = np.asarray(y, dtype=np.float64)
    q_hat = np.asarray(q_hat, dtype=np.float64)
    indicator = (y < q_hat).astype(np.float64)
    return (y - q_hat) * (q - indicator)


def wis_via_quantile_loss(
    y: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> np.ndarray:
    """Alternative WIS formula via pinball loss aggregation.

    Bracher 2021 §2.4:
        WIS ≈ (2 / (2K + 1)) * Σ_{q ∈ REQUIRED_QUANTILES} QL_q(F, y)

    Equivalent to wis() up to round-off — cross-check that catches any
    indexing or weight bug in the interval-form implementation.
    """
    y = np.asarray(y, dtype=np.float64)
    total = np.zeros_like(y, dtype=np.float64)
    for q in REQUIRED_QUANTILES:
        total = total + quantile_loss(y, np.asarray(quantile_forecasts[q]), q)
    return total * (2.0 / len(REQUIRED_QUANTILES))


def coverage(
    y: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
    alpha: float,
) -> float:
    """Empirical coverage of the (1-α) PI: fraction of y inside [q_{α/2}, q_{1-α/2}]."""
    q_lo = round(alpha / 2, 4)
    q_hi = round(1 - alpha / 2, 4)
    lo = np.asarray(quantile_forecasts[q_lo])
    hi = np.asarray(quantile_forecasts[q_hi])
    y = np.asarray(y)
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean())
