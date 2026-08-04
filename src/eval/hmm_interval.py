"""HMM-Derived Calibrated Predictive Intervals (Method F).

Constructs prediction intervals for CG-Mamba via 3-component decomposition
of the HMM emission mixture, centered at CG-Mamba's point prediction:

    sigma2_within     = Σ_k γ_h[k] · σ²_k_ili         (aleatoric per-phase)
    sigma2_between_HMM = Σ_k γ_h[k] · (μ_k - μ_HMM)²   (pure phase uncertainty)
    bias_sq            = (μ_HMM - μ_CGM)²              (CG-Mamba refinement)
    sigma2_total       = sigma2_within + sigma2_between_HMM
    (bias_sq excluded from sigma2_total — kept as separate interpretability component)

Note: sigma2_between_HMM + bias_sq = sigma2_between_CGM (algebraic identity).

Quantile construction:
    - Gaussian approximation (default, fast):
        q(level) = μ_CGM + Φ⁻¹(level) · sqrt(s · sigma2_total)
    - Numerical mixture quantile (when sigma2_between/total > 0.3):
        F_mix(y) = Σ_k γ_h[k] · Φ((y - μ_k)/σ_k); solve via brentq

Per-horizon calibration scale s_h learned on validation by quantile matching.

References:
    - Law of Total Variance: standard Gaussian mixture theory
    - Lakshminarayanan et al. 2017 NeurIPS: predictive μ/σ decoupling pattern
    - Bracher et al. 2021 PLOS Comp Bio: WIS evaluation target metric
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src.eval.wis import REQUIRED_QUANTILES


# ─── 3-component decomposition ──────────────────────────────────────────────


@dataclass
class HMMDecomposition:
    """Per-sample, per-horizon Method F decomposition (z-scored space).

    All shapes: [N, H] where N=samples, H=horizons.
    """
    mu_CGM: np.ndarray
    mu_HMM: np.ndarray
    sigma2_within: np.ndarray          # aleatoric per-phase
    sigma2_between_HMM: np.ndarray     # pure phase uncertainty
    bias_sq: np.ndarray                # CG-Mamba refinement
    sigma2_total: np.ndarray           # within + between + bias

    @property
    def sigma_total(self) -> np.ndarray:
        return np.sqrt(self.sigma2_total + 1e-12)


def compute_decomposition(
    mu_CGM: np.ndarray,           # [N, H]
    gamma_all: np.ndarray,        # [N, H, K]
    mu_k_ili: np.ndarray,         # [K] HMM emission means (z-scored)
    sigma2_k_ili: np.ndarray,     # [K] HMM emission variances (z-scored)
) -> HMMDecomposition:
    """3-component Method F decomposition.

    All inputs in z-scored space (HMM trained on standardized features,
    CG-Mamba output also z-scored target). Caller denormalizes if needed.
    """
    N, H = mu_CGM.shape
    K = len(mu_k_ili)
    assert gamma_all.shape == (N, H, K), f"gamma_all {gamma_all.shape} != {(N, H, K)}"
    assert sigma2_k_ili.shape == (K,)

    # mu_HMM_h = Σ_k γ_h[k] · μ_k_ili
    mu_HMM = (gamma_all * mu_k_ili[None, None, :]).sum(axis=-1)               # [N, H]

    # sigma2_within = Σ_k γ_h[k] · σ²_k_ili
    sigma2_within = (gamma_all * sigma2_k_ili[None, None, :]).sum(axis=-1)    # [N, H]

    # sigma2_between_HMM = Σ_k γ_h[k] · (μ_k - μ_HMM)²
    mu_k_centered = mu_k_ili[None, None, :] - mu_HMM[..., None]               # [N, H, K]
    sigma2_between_HMM = (gamma_all * mu_k_centered ** 2).sum(axis=-1)        # [N, H]

    # bias² = (μ_HMM - μ_CGM)²  — DETERMINISTIC offset (not random uncertainty)
    # Kept as separate component for §V.X interpretability (anomaly signal),
    # NOT included in sigma2_total (which represents random predictive variance).
    bias_sq = (mu_HMM - mu_CGM) ** 2                                          # [N, H]

    # sigma2_total = within + between (uncertainty components only)
    # bias_sq is a deterministic offset — including it would conflate
    # "data + phase uncertainty" with "model refinement signal", causing
    # calibration to fail (s hits upper bound).
    sigma2_total = sigma2_within + sigma2_between_HMM
    # Guard: float32→float64 gamma precision can yield tiny negatives
    sigma2_total = np.maximum(sigma2_total, 1e-12)

    return HMMDecomposition(
        mu_CGM=mu_CGM, mu_HMM=mu_HMM,
        sigma2_within=sigma2_within,
        sigma2_between_HMM=sigma2_between_HMM,
        bias_sq=bias_sq,
        sigma2_total=sigma2_total,
    )


# ─── Per-horizon calibration scale s_h ──────────────────────────────────────


def calibrate_scale_quantile_matching(
    y_val: np.ndarray,             # [N_val, H]
    decomp_val: HMMDecomposition,
    target_quantiles: tuple[float, ...] = (0.025, 0.05, 0.1, 0.25, 0.5,
                                            0.75, 0.9, 0.95, 0.975),
) -> np.ndarray:                   # [H] per-horizon scale
    """Learn s_h per horizon by grid-search quantile matching.

    For each horizon h:
        argmin_s Σ_q (empirical_cov(s, q) - q)²

    Grid search over s ∈ logspace(0.01, 100) more robust than minimize_scalar
    (loss can be non-convex due to discrete empirical cdf).
    """
    N, H = y_val.shape
    s_grid = np.concatenate([np.linspace(0.01, 0.5, 20),
                              np.linspace(0.5, 3.0, 30),
                              np.linspace(3.0, 30.0, 15)])
    s_per_h = np.zeros(H)
    for h in range(H):
        y_h = y_val[:, h]
        mu_h = decomp_val.mu_CGM[:, h]
        sig2_h = decomp_val.sigma2_total[:, h]
        losses = []
        for s in s_grid:
            sig_scaled = np.sqrt(s * sig2_h + 1e-12)
            err = 0.0
            for q in target_quantiles:
                z = norm.ppf(q)
                q_pred = mu_h + z * sig_scaled
                emp = float((y_h <= q_pred).mean())
                err += (emp - q) ** 2
            losses.append(err)
        s_per_h[h] = float(s_grid[int(np.argmin(losses))])
    return s_per_h


def calibrate_scale_simple_ratio(
    y_val: np.ndarray,             # [N_val, H]
    decomp_val: HMMDecomposition,
) -> np.ndarray:                   # [H]
    """Simple scale: s_h = (residual std) / (sigma_total std) per horizon."""
    N, H = y_val.shape
    s_per_h = np.zeros(H)
    for h in range(H):
        residuals = y_val[:, h] - decomp_val.mu_CGM[:, h]
        sigma_total = np.sqrt(decomp_val.sigma2_total[:, h] + 1e-12)
        s_per_h[h] = float(residuals.std() / (sigma_total.std() + 1e-12))
    return s_per_h


# ─── Quantile construction ──────────────────────────────────────────────────


def gaussian_quantiles(
    decomp: HMMDecomposition,
    s_per_h: np.ndarray,             # [H]
) -> dict[float, np.ndarray]:
    """Gaussian approximation: q(level) = μ_CGM + Φ⁻¹(level) · sqrt(s · σ²_total)."""
    sig_scaled = np.sqrt(s_per_h[None, :] * decomp.sigma2_total + 1e-12)  # [N, H]
    out = {}
    for q in REQUIRED_QUANTILES:
        z = norm.ppf(q)
        out[q] = decomp.mu_CGM + z * sig_scaled
    return out


def _mixture_cdf(y: float, mu_k: np.ndarray, sigma_k: np.ndarray,
                 gamma_h: np.ndarray) -> float:
    """F_mix(y) = Σ_k γ_h[k] · Φ((y - μ_k)/σ_k)."""
    return float(np.sum(gamma_h * norm.cdf(y, loc=mu_k, scale=sigma_k)))


def mixture_quantile_one(
    target_q: float,
    mu_k: np.ndarray, sigma_k: np.ndarray, gamma_h: np.ndarray,
    lower: float = -10.0, upper: float = 10.0,
) -> float:
    """Numerical solve F_mix(y) = target_q via Brent's method."""
    f = lambda y: _mixture_cdf(y, mu_k, sigma_k, gamma_h) - target_q
    # Expand bounds if needed
    f_lo, f_hi = f(lower), f(upper)
    while f_lo > 0 and lower > -1e6:
        lower *= 2; f_lo = f(lower)
    while f_hi < 0 and upper < 1e6:
        upper *= 2; f_hi = f(upper)
    if f_lo > 0 or f_hi < 0:
        # Bracketing failed — fallback to Gaussian approx
        return float(np.nan)
    return brentq(f, lower, upper, xtol=1e-4)


def mixture_quantiles_per_sample(
    gamma_all: np.ndarray,           # [N, H, K]
    mu_k_ili: np.ndarray,            # [K]
    sigma2_k_ili: np.ndarray,        # [K]
    mu_shift: np.ndarray,            # [N, H] center shift (μ_CGM - μ_HMM)
    s_per_h: np.ndarray,             # [H] calibration
) -> dict[float, np.ndarray]:
    """Numerical mixture quantile per sample (more accurate, slower).

    Returns dict q -> [N, H] quantile values.
    Quantile is computed from the HMM mixture distribution, then SHIFTED by
    (μ_CGM - μ_HMM) to recenter on CG-Mamba. Calibration s_h scales spread.

    Note: shifting and scaling a quantile preserves quantile structure:
        q(F(y)) of shifted scaled mixture = mu_shift + sqrt(s) · q(F(y)) of original
        (approximate — exact only if mixture is location-scale family)
    """
    N, H, K = gamma_all.shape
    sigma_k = np.sqrt(sigma2_k_ili)
    out = {q: np.zeros((N, H)) for q in REQUIRED_QUANTILES}
    for n in range(N):
        for h in range(H):
            g_h = gamma_all[n, h, :]
            sqrt_s = np.sqrt(s_per_h[h])
            for q in REQUIRED_QUANTILES:
                # Quantile in original mixture space
                y_mix = mixture_quantile_one(q, mu_k_ili, sigma_k * sqrt_s, g_h)
                if np.isnan(y_mix):
                    # Fallback: Gaussian approx
                    mu_HMM = float((g_h * mu_k_ili).sum())
                    sig2_w = float((g_h * sigma2_k_ili).sum())
                    sig2_b = float((g_h * (mu_k_ili - mu_HMM) ** 2).sum())
                    sig = np.sqrt(s_per_h[h] * (sig2_w + sig2_b) + 1e-12)
                    y_mix = mu_HMM + norm.ppf(q) * sig
                # Shift to CG-Mamba center
                out[q][n, h] = mu_shift[n, h] + y_mix
    return out


def construct_quantiles(
    decomp: HMMDecomposition,
    gamma_all: np.ndarray,           # [N, H, K]
    mu_k_ili: np.ndarray,            # [K]
    sigma2_k_ili: np.ndarray,        # [K]
    s_per_h: np.ndarray,             # [H]
    mode: str = "auto",              # "gaussian", "mixture", "auto"
    between_ratio_threshold: float = 0.3,
) -> tuple[dict[float, np.ndarray], str]:
    """Construct 23-quantile predictions. Returns (quantiles, mode_used).

    Auto mode selects "mixture" if avg sigma2_between/sigma2_total > threshold,
    otherwise "gaussian" (faster). Mixture quantile call is O(N×H×K×23) brentq's
    — slow but accurate for multi-modal mixtures.
    """
    if mode == "auto":
        ratio_mean = (decomp.sigma2_between_HMM
                      / (decomp.sigma2_total + 1e-12)).mean()
        mode_used = "mixture" if ratio_mean > between_ratio_threshold else "gaussian"
    else:
        mode_used = mode

    if mode_used == "gaussian":
        return gaussian_quantiles(decomp, s_per_h), "gaussian"
    elif mode_used == "mixture":
        # mu_shift: μ_CGM - μ_HMM (recenter mixture quantile)
        mu_shift = decomp.mu_CGM - decomp.mu_HMM
        return mixture_quantiles_per_sample(
            gamma_all, mu_k_ili, sigma2_k_ili, mu_shift, s_per_h,
        ), "mixture"
    else:
        raise ValueError(f"Unknown mode: {mode_used}")


# ─── End-to-end pipeline ────────────────────────────────────────────────────


def method_f_predict_quantiles(
    mu_CGM_test: np.ndarray,
    gamma_all_test: np.ndarray,
    mu_CGM_val: np.ndarray,
    gamma_all_val: np.ndarray,
    y_val: np.ndarray,
    mu_k_ili: np.ndarray,
    sigma2_k_ili: np.ndarray,
    target_mean: float,
    target_std: float,
    calibration: str = "quantile_matching",   # "quantile_matching" or "simple_ratio"
    mode: str = "auto",
) -> tuple[dict[float, np.ndarray], dict]:
    """End-to-end Method F: validation calibration + test quantile prediction.

    Returns (quantiles_test [raw scale], metadata).
    All inputs in z-scored space; output quantiles denormalized to raw.
    """
    decomp_val = compute_decomposition(mu_CGM_val, gamma_all_val,
                                       mu_k_ili, sigma2_k_ili)
    decomp_test = compute_decomposition(mu_CGM_test, gamma_all_test,
                                        mu_k_ili, sigma2_k_ili)
    if calibration == "quantile_matching":
        s_per_h = calibrate_scale_quantile_matching(y_val, decomp_val)
    elif calibration == "simple_ratio":
        s_per_h = calibrate_scale_simple_ratio(y_val, decomp_val)
    else:
        raise ValueError(calibration)

    q_z, mode_used = construct_quantiles(
        decomp_test, gamma_all_test, mu_k_ili, sigma2_k_ili,
        s_per_h, mode=mode,
    )
    # Denormalize z-score → raw
    q_raw = {q: arr * target_std + target_mean for q, arr in q_z.items()}

    metadata = {
        "calibration_method": calibration,
        "quantile_mode": mode_used,
        "s_per_h": s_per_h.tolist(),
        "decomp_test": {
            "sigma2_within_mean": float(decomp_test.sigma2_within.mean()),
            "sigma2_between_HMM_mean": float(decomp_test.sigma2_between_HMM.mean()),
            "bias_sq_mean": float(decomp_test.bias_sq.mean()),
            "sigma2_total_mean": float(decomp_test.sigma2_total.mean()),
        },
    }
    return q_raw, metadata
