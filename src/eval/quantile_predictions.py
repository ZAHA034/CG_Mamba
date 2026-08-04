"""Quantile forecast generators for the WIS evaluation pipeline (PLAN J.10).

Per-baseline UQ method (PLAN J.2/J.3):
    - MC Dropout (LSTM, CG-Mamba, PatchTST, iTrans, TimesNet, EpiDeep)
    - Empirical residual quantiles (Persistence)
    - 5-seed Gaussian fit (DLinear, N-BEATS, Vanilla Mamba)
    - Parametric Gaussian from Kalman variance (SARIMA)

All generators return dict[q: float, np.ndarray of shape [N, H]] covering the
23 quantile levels in src.eval.wis.REQUIRED_QUANTILES.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm, t as student_t

from src.eval.wis import REQUIRED_QUANTILES


@contextmanager
def _dropout_train_mode(model: nn.Module):
    """Temporarily enable only Dropout layers' .train() while keeping the rest
    in eval mode — MC Dropout (Gal & Ghahramani 2016) without enabling
    BatchNorm/LayerNorm running-stat updates.

    v2.1.7-B (2026-05-26): Extended to handle nn.LSTM/GRU/RNN with built-in
    dropout. PyTorch's recurrent modules apply dropout internally (CuDNN fused
    kernel) and check the module's own .training flag, not a separate nn.Dropout
    instance. Without this, LSTM MC Dropout produces zero variance → cov95=0.
    """
    original = {}
    for name, m in model.named_modules():
        # Standard standalone Dropout layers
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            original[name] = m.training
            m.train()
        # Recurrent layers with built-in dropout (LSTM/GRU/RNN with dropout > 0)
        elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)) and getattr(m, "dropout", 0) > 0:
            original[name] = m.training
            m.train()
    try:
        yield
    finally:
        for name, m in model.named_modules():
            if name in original:
                m.train(original[name])


@torch.no_grad()
def mc_dropout_quantiles(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 100,
    dropout_layers_only: bool = True,
    device: str = "cuda",
) -> dict[float, np.ndarray]:
    """MC Dropout quantile forecast — Gal & Ghahramani 2016.

    Args:
        model: forecast model with non-zero dropout. Must return [B, H].
        x: input tensor [B, L, V].
        n_samples: number of stochastic forward passes (PLAN Q2: 100 default).
        dropout_layers_only: if True, only Dropout layers run in train() mode
            (recommended — preserves BN/LN behavior). If False, entire model.

    Returns:
        dict mapping q → np.ndarray of shape [B, H]. Quantiles computed
        empirically from the n_samples MC draws.
    """
    model.eval()
    x = x.to(device)
    samples = []
    cm = _dropout_train_mode(model) if dropout_layers_only else _no_op()
    with cm:
        for _ in range(n_samples):
            preds = model(x)               # [B, H]
            samples.append(preds.cpu().numpy())
    samples = np.stack(samples, axis=0)    # [S, B, H]
    out: dict[float, np.ndarray] = {}
    for q in REQUIRED_QUANTILES:
        out[q] = np.quantile(samples, q, axis=0)
    return out


@contextmanager
def _no_op():
    yield


def residual_quantiles_h_specific(
    point_preds: np.ndarray,
    val_residuals_per_h: list[np.ndarray],
) -> dict[float, np.ndarray]:
    """Empirical h-specific residual quantile forecast (PLAN Q1).

    For Persistence (and as a fallback for any point-only baseline):
        q_h(level) = point_pred_h + np.quantile(val_residuals_h, level)

    Args:
        point_preds: shape [N, H].
        val_residuals_per_h: length-H list, each array of validation residuals
            for horizon h. Residual = (y_val_h - pred_val_h).

    Returns:
        dict mapping q → array [N, H]. Each horizon uses its own residual
        distribution, avoiding the h=1 vs h=4 noise scale mismatch.
    """
    N, H = point_preds.shape
    if len(val_residuals_per_h) != H:
        raise ValueError(
            f"val_residuals_per_h length {len(val_residuals_per_h)} != H={H}"
        )
    out: dict[float, np.ndarray] = {}
    for q in REQUIRED_QUANTILES:
        q_arr = np.zeros((N, H), dtype=np.float64)
        for h in range(H):
            offset = np.quantile(val_residuals_per_h[h], q)
            q_arr[:, h] = point_preds[:, h] + offset
        out[q] = q_arr
    return out


def parametric_gaussian_quantiles(
    mean: np.ndarray, variance: np.ndarray,
) -> dict[float, np.ndarray]:
    """Parametric Gaussian quantiles (PLAN Q3 — SARIMA Kalman variance path).

    q(level) = mean + sqrt(variance) * Φ^{-1}(level)

    Args:
        mean: shape [N, H].
        variance: shape [N, H], per-(obs, horizon) forecast variance.

    Returns:
        dict q → array [N, H].
    """
    sd = np.sqrt(np.maximum(variance, 0.0))
    out: dict[float, np.ndarray] = {}
    for q in REQUIRED_QUANTILES:
        z = norm.ppf(q)
        out[q] = mean + sd * z
    return out


def ensemble_gaussian_quantiles(
    member_preds: np.ndarray,
    ddof: int = 1,
) -> dict[float, np.ndarray]:
    """5-seed (or other ensemble) Gaussian-fit quantiles (PLAN J.6 Main).

    Computes per-(obs, horizon) ensemble mean and std → Gaussian quantiles.
    Used for DLinear, N-BEATS, Vanilla Mamba where dropout is absent.

    Args:
        member_preds: shape [S, N, H] (S = number of ensemble members).
        ddof: degrees-of-freedom correction for std (default 1, sample std).

    Returns:
        dict q → array [N, H].
    """
    mean = member_preds.mean(axis=0)
    var = member_preds.var(axis=0, ddof=ddof)
    return parametric_gaussian_quantiles(mean, var)


def ensemble_student_t_quantiles(
    member_preds: np.ndarray,
    df: int = 4,
    ddof: int = 1,
) -> dict[float, np.ndarray]:
    """Student-t quantiles (PLAN J.6 §S.X sensitivity, df=4).

    Heavy-tailed alternative for small ensembles. Use as supplementary
    robustness check vs Gaussian.
    """
    mean = member_preds.mean(axis=0)
    sd = member_preds.std(axis=0, ddof=ddof)
    out: dict[float, np.ndarray] = {}
    for q in REQUIRED_QUANTILES:
        out[q] = mean + sd * student_t.ppf(q, df=df)
    return out
