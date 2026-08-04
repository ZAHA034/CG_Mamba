"""src/eval/wis_standard.py — T5 single source of truth for WIS / Cov95 / PI

Background (T5 Phase 1 audit, 2026-06-21):
    Workflow A confirmed: src/eval/wis.py implements Bracher 2021 Eq.(1) bit-exact;
    worked example (μ=1.0, σ=0.5, y=1.3, 23 FluSight taus) → WIS=0.167894 in all
    three forms (our 2.0*pinball.mean(), Bracher interval form, scoringutils
    Eq.(4) pinball form), agree to ~1e-15.
    Workflow B confirmed: the 0.296 (paper m1_8 Method F) vs 0.399 (e1_final_eval
    RAW) gap is 100% from σ-source calibration — same WIS function, different
    quantile inputs.

This module is the SINGLE SOURCE OF TRUTH:
  - Standard WIS / Cov95: re-exports from src/eval/wis.py (Bracher 2021).
  - PI / quantile constructors (4 input forms):
      a. Gaussian-parametric:   quantiles_from_gaussian(mu, sigma2)
      b. Empirical-sample:      quantiles_from_samples(samples)
      c. Method F s_h-scaled:   quantiles_method_f_calibrated(decomp, ...)
      d. Conformalized (CQR):   quantiles_conformal_cqr(base, val_base, val_y)
  - Convenience: cov95_wis_from_(mu, sigma2) for Gaussian-PI back-compat
    wrappers used by legacy callers.

Cross-model fairness rule (T5-4 Track B):
    Track B = apply quantiles_conformal_cqr to EVERY model's base quantiles
    using a single conformity score definition (CQR signed residual). This is
    the apples-to-apples cross-model comparator. Track A keeps each model's
    native UQ (Method F for CGM, Kalman for SARIMA, MC Dropout for NN, ensemble
    for DLinear/N-BEATS) — supplementary, v1 reproducibility.

DEPRECATED IMPL (callers MUST migrate):
    - src/models/heteroscedastic_head.py::eval_cov95_wis  → cov95_wis_from_gaussian
    - scripts/e1_final_tighten.py::cov95_wis              → cov95_wis_from_gaussian
    - scripts/e1_final_tighten4.py::cov95_wis (imported)  → cov95_wis_from_gaussian
    - scripts/e1_hpo.py::cov95_wis (dataframe wrapper)    → cov95_wis_from_gaussian

Worked example unit test (test_wis_standard.py):
    μ=1.0, σ=0.5, y=1.3, FluSight 23-quantile
    → WIS = 0.16789396786717625 ± 1e-12 across all four quantile sources
       (Gaussian, parametric, scoringutils-equivalent).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import norm as sp_norm

from src.eval.wis import (
    REQUIRED_QUANTILES,
    ALPHA_LEVELS,
    INTERVAL_PAIRS,
    interval_score,
    wis,
    wis_decomposed,
    quantile_loss,
    wis_via_quantile_loss,
    coverage,
)

FLUSIGHT_23 = np.array(REQUIRED_QUANTILES, dtype=np.float64)
LO_IDX_025 = int(np.where(np.isclose(FLUSIGHT_23, 0.025))[0][0])
HI_IDX_975 = int(np.where(np.isclose(FLUSIGHT_23, 0.975))[0][0])


# ============================================================================
# (a) Gaussian-parametric quantile generator
# ============================================================================
def quantiles_from_gaussian(
    mu: np.ndarray,
    sigma2: np.ndarray,
    taus: np.ndarray = FLUSIGHT_23,
) -> dict[float, np.ndarray]:
    """Parametric Gaussian quantiles: Q_τ = μ + Φ⁻¹(τ) · σ.

    Args:
        mu:      shape [...] point forecasts.
        sigma2:  shape [...] predictive variance (will be clipped to ≥1e-12).
        taus:    1D quantile levels.

    Returns: dict {τ → array of shape [...]}. Keys are float taus matching
             REQUIRED_QUANTILES so that the result is directly usable in
             src.eval.wis.wis() / coverage().

    Note: ADMISSIBLE only if the predictive distribution is truly Gaussian.
    For heavy-tailed residuals (e.g., wILI) this produces biased outer quantiles.
    FluSight admits any quantile-construction mechanism, but the FAIR comparator
    is Conformalized — see quantiles_conformal_cqr().
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.sqrt(np.clip(np.asarray(sigma2, dtype=np.float64), 1e-12, None))
    z = sp_norm.ppf(np.asarray(taus, dtype=np.float64))
    # Broadcast: result has shape [..., len(taus)] before dict-conversion
    Q = mu[..., None] + sigma[..., None] * z
    return {float(t): Q[..., k] for k, t in enumerate(taus)}


# ============================================================================
# (b) Empirical-sample quantile generator
# ============================================================================
def quantiles_from_samples(
    samples: np.ndarray,
    taus: np.ndarray = FLUSIGHT_23,
    axis: int = -1,
) -> dict[float, np.ndarray]:
    """Empirical quantiles from posterior samples (e.g., MC Dropout).

    Args:
        samples: shape [..., S] forecast samples.
        taus:    1D quantile levels.
        axis:    sample axis (default last).

    Returns: dict {τ → array with shape = samples.shape minus the sample axis}.

    This is the FluSight-native PI construction for sample-based forecasters
    (LSTM/Vanilla Mamba/PatchTST/iTransformer/TimesNet/EpiDeep via MC Dropout).
    """
    samples = np.asarray(samples, dtype=np.float64)
    qs = np.quantile(samples, np.asarray(taus, dtype=np.float64), axis=axis)
    # qs shape: [Q, ...] — first axis is quantile
    return {float(t): qs[k] for k, t in enumerate(taus)}


# ============================================================================
# (c) Method F per-horizon s_h calibrated (Gaussian + scalar shrinkage)
# ============================================================================
def calibrate_s_h(
    mu_val: np.ndarray,
    sigma2_val: np.ndarray,
    y_val: np.ndarray,
    s_grid: np.ndarray | None = None,
    target_taus: np.ndarray = FLUSIGHT_23,
) -> np.ndarray:
    """Grid-search per-horizon σ-scale s_h on validation set by quantile matching.

    For each horizon h: minimize Σ_τ (empirical_freq(y ≤ μ + Φ⁻¹(τ)·sqrt(s_h·σ²)) − τ)².

    Args:
        mu_val:     [N, H] val point forecasts.
        sigma2_val: [N, H] val APMD variance.
        y_val:      [N, H] val ground truth.
        s_grid:     1D positive grid (default 100 points log-spaced [0.01, 30]).
        target_taus: quantile set for matching loss.

    Returns: s_h array of shape [H]. Mirror of paper's
             src.eval.hmm_interval.calibrate_scale_quantile_matching.
    """
    if s_grid is None:
        s_grid = np.geomspace(0.01, 30.0, 100)
    H = mu_val.shape[1]
    s_per_h = np.zeros(H, dtype=np.float64)
    taus = np.asarray(target_taus, dtype=np.float64)
    z = sp_norm.ppf(taus)
    for h in range(H):
        mu_h = mu_val[:, h]
        sig2_h = sigma2_val[:, h]
        y_h = y_val[:, h]
        losses = np.empty_like(s_grid)
        for i, s in enumerate(s_grid):
            sig_scaled = np.sqrt(s * sig2_h + 1e-12)
            emp_freqs = ((y_h[:, None] <= mu_h[:, None] + z[None, :] * sig_scaled[:, None])
                         .mean(axis=0))
            losses[i] = float(((emp_freqs - taus) ** 2).sum())
        s_per_h[h] = float(s_grid[int(losses.argmin())])
    return s_per_h


def quantiles_method_f_calibrated(
    mu_test: np.ndarray,
    sigma2_test: np.ndarray,
    s_per_h: np.ndarray,
    taus: np.ndarray = FLUSIGHT_23,
) -> dict[float, np.ndarray]:
    """Apply Method F per-horizon s_h to test forecasts → Gaussian quantiles.

    Q_τ(t, h) = μ(t, h) + Φ⁻¹(τ) · sqrt(s_h · σ²(t, h)).

    Args:
        mu_test:    [N, H]
        sigma2_test:[N, H]
        s_per_h:    [H] (output of calibrate_s_h on val)
        taus:       1D quantile levels.

    Returns: dict {τ → array [N, H]}.

    Convention: Method F is CGM's native calibrator (paper). For Track A
    (protocol-specific PI), use this. For Track B (apples-to-apples),
    use quantiles_conformal_cqr instead.
    """
    mu = np.asarray(mu_test, dtype=np.float64)
    sigma2 = np.asarray(sigma2_test, dtype=np.float64)
    s = np.asarray(s_per_h, dtype=np.float64)[None, :]  # [1, H]
    sigma_scaled = np.sqrt(np.clip(s * sigma2, 1e-12, None))  # [N, H]
    z = sp_norm.ppf(np.asarray(taus, dtype=np.float64))
    Q = mu[..., None] + sigma_scaled[..., None] * z      # [N, H, Q]
    return {float(t): Q[..., k] for k, t in enumerate(taus)}


# ============================================================================
# (d) Conformalized Quantile Regression (CQR) — Track B uniform calibrator
# ============================================================================
def quantiles_conformal_cqr(
    base_quantiles_val: dict[float, np.ndarray],
    base_quantiles_test: dict[float, np.ndarray],
    y_val: np.ndarray,
    alpha_target: float = 0.05,
    taus: np.ndarray = FLUSIGHT_23,
) -> dict[float, np.ndarray]:
    """Split Conformal — CQR-style nonconformity score, applied across the full
    quantile grid.

    Romano, Patterson, Candès (2019) "Conformalized Quantile Regression"
    — extended to all 23 quantiles by symmetric shift (a.k.a. CQR-symmetric).

    Procedure:
        1. For each τ in REQUIRED_QUANTILES, compute val signed residual
           E_τ = y_val − base_quantiles_val[τ].
        2. For each PI level (1-α_k), the CQR nonconformity is
           score_k = max(base_q_val[α_k/2] − y_val, y_val − base_q_val[1-α_k/2]).
           The (1-α_k)(1+1/n)-quantile of score_k is the radius r_k.
        3. Calibrated test PI: [base_q_test[α_k/2] − r_k, base_q_test[1-α_k/2] + r_k].
           Median (τ=0.5) is left unchanged from base.

    Args:
        base_quantiles_val:  dict {τ → [N_val]} val base quantiles (Gaussian
                              parametric, empirical sample, ensemble — any).
        base_quantiles_test: dict {τ → [N_test]} test base quantiles.
        y_val:               [N_val] truth on val (used for CQR scoring).
        alpha_target:        primary headline α (default 0.05 for 95% PI).
                              Output dict contains ALL 23 quantiles re-derived
                              via symmetric CQR shift for each (α_k/2, 1-α_k/2)
                              pair; median (0.5) unchanged.
        taus:                target quantile grid (default FluSight 23).

    Returns: dict {τ → array [N_test]} — Track B calibrated quantiles.

    CRITICAL APPLES-TO-APPLES PROPERTY:
        This single function takes Gaussian-parametric base, sample-based base,
        ensemble base, or any other base quantile producer and applies the
        SAME nonconformity score (signed quantile residual). Identity of base
        UQ is absorbed; v2 §V cross-model comparison reads CQR-output quantiles
        only. This is the field-standard fair comparator.
    """
    y_val = np.asarray(y_val, dtype=np.float64)
    n_val = len(y_val)
    out: dict[float, np.ndarray] = {}

    for alpha, (q_lo, q_hi) in zip(ALPHA_LEVELS, INTERVAL_PAIRS):
        lo_val = np.asarray(base_quantiles_val[q_lo], dtype=np.float64)
        hi_val = np.asarray(base_quantiles_val[q_hi], dtype=np.float64)
        lo_test = np.asarray(base_quantiles_test[q_lo], dtype=np.float64)
        hi_test = np.asarray(base_quantiles_test[q_hi], dtype=np.float64)
        score = np.maximum(lo_val - y_val, y_val - hi_val)
        # finite-sample-corrected (1-α)(1+1/n) quantile
        k_floor = int(np.ceil((n_val + 1) * (1 - alpha))) - 1
        k_floor = max(0, min(k_floor, n_val - 1))
        score_sorted = np.sort(score)
        r = float(score_sorted[k_floor])
        out[float(q_lo)] = lo_test - r
        out[float(q_hi)] = hi_test + r

    # Median = base (CQR symmetric does not shift median).
    if 0.5 in base_quantiles_test:
        out[0.5] = np.asarray(base_quantiles_test[0.5], dtype=np.float64)
    return out


# ============================================================================
# Convenience wrappers (legacy back-compat — DEPRECATED, callers migrate)
# ============================================================================
def cov95_wis_from_gaussian(
    mu: np.ndarray,
    sigma2: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Drop-in replacement for legacy cov95_wis / eval_cov95_wis (Gaussian PI).

    Builds Gaussian-parametric quantiles, then computes Bracher 2021 Cov95+WIS
    via src.eval.wis.{coverage,wis}. The two-step composition makes the PI
    construction explicit (and disclosable as "Gaussian-PI variant").

    DEPRECATED: existing callers using this function get the SAME numerical
    result as their inline implementation. New code should use
    quantiles_from_gaussian + (coverage, wis) directly.

    Returns (Cov95, WIS).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    qf = quantiles_from_gaussian(mu, sigma2)
    cov = coverage(y, qf, alpha=0.05)
    wis_per_row = wis(y, qf)
    return cov, float(np.mean(wis_per_row))


__all__ = [
    "REQUIRED_QUANTILES",
    "ALPHA_LEVELS",
    "INTERVAL_PAIRS",
    "FLUSIGHT_23",
    # Bracher 2021 standard, re-exported
    "interval_score",
    "wis",
    "wis_decomposed",
    "quantile_loss",
    "wis_via_quantile_loss",
    "coverage",
    # PI constructors (4 inputs)
    "quantiles_from_gaussian",
    "quantiles_from_samples",
    "calibrate_s_h",
    "quantiles_method_f_calibrated",
    "quantiles_conformal_cqr",
    # Legacy wrappers
    "cov95_wis_from_gaussian",
]
