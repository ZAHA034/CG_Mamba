"""N-BEATS weekly baseline for CG-Mamba (v2.1.7-A++ M2.3 expansion) — Multivariate.

Direct from-scratch implementation following Oreshkin et al. (2020, ICLR):
  "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"

Paper-canonical Generic block architecture (V=6 multivariate extension):
  FC stack (n_layers fully-connected, hidden width) → ReLU
  Two linear heads:
    theta_b: backcast coefficients (output dim = lookback × V, full reconstruction)
    theta_f: forecast coefficients (output dim = horizon, target only)
  Generic mode: theta IS the output (no basis expansion)

Stack (residual forecasting in flattened L×V space):
  x_flat ∈ R^{L·V}  (flatten time × features)
  residual_0 = x_flat
  forecast_sum = 0
  for block in blocks:
    backcast, forecast = block(residual)        # backcast [B, L·V], forecast [B, H]
    residual = residual - backcast              # next-block input
    forecast_sum += forecast
  return forecast_sum                            # [B, H] target prediction

Multivariate extension (v2.1.7-A++ fairness fix):
  - Input: x [B, L=104, V=6] z-scored (same as all NN baselines: LSTM, PatchTST,
    iTransformer, DLinear, TimesNet, Vanilla Mamba, CG-Mamba).
  - Flatten to [B, L·V=624] for FC stack consumption.
  - Backcast in same flattened space; forecast only for target channel (H=4).
  - Univariate ablation (paper-faithful only-target) available via
    `target_only=True` flag in adapter (used for Supplementary §S.X ablation).

References:
  - Oreshkin, B.N., Carpov, D., Chapados, N., & Bengio, Y. (2020). "N-BEATS:
    Neural basis expansion analysis for interpretable time series forecasting."
    ICLR 2020. arXiv:1905.10437
  - M4 Competition winner (2018).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    """Generic N-BEATS block: n_layers-FC + theta_b/theta_f linear heads.

    Operates in flattened L·V space. theta_b reconstructs all V channels over
    lookback; theta_f predicts target only over horizon.
    """

    def __init__(
        self,
        lookback: int = 104,
        horizon: int = 4,
        enc_in: int = 6,
        hidden: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.enc_in = enc_in
        self.in_dim = lookback * enc_in       # flattened input/backcast space

        # n_layers FC stack with ReLU activations (paper §3)
        layers = []
        in_dim = self.in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        self.fc_stack = nn.Sequential(*layers)

        # Generic mode: theta_b/theta_f are direct outputs (no basis expansion)
        self.theta_b = nn.Linear(hidden, self.in_dim)   # full multivariate backcast
        self.theta_f = nn.Linear(hidden, horizon)        # target-only forecast

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, L·V] (flattened multivariate lookback).
        Returns: (backcast [B, L·V], forecast [B, H])."""
        h = self.fc_stack(x)
        backcast = self.theta_b(h)
        forecast = self.theta_f(h)
        return backcast, forecast


class NBeatsStack(nn.Module):
    """Sequential N-BEATS blocks with residual forecasting (multivariate)."""

    def __init__(
        self,
        n_blocks: int = 3,
        lookback: int = 104,
        horizon: int = 4,
        enc_in: int = 6,
        hidden: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        self.horizon = horizon
        self.blocks = nn.ModuleList([
            NBeatsBlock(lookback, horizon, enc_in, hidden, n_layers)
            for _ in range(n_blocks)
        ])

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """x_flat: [B, L·V]. Returns: forecast [B, H] (sum across blocks)."""
        forecast = torch.zeros(
            x_flat.size(0), self.horizon,
            device=x_flat.device, dtype=x_flat.dtype,
        )
        residual = x_flat
        for block in self.blocks:
            backcast, fc = block(residual)
            residual = residual - backcast
            forecast = forecast + fc
        return forecast


class NBeatsForecaster(nn.Module):
    """N-BEATS multivariate adapter — V=6 input, H=4 target forecast output.

    Input/output match all NN baselines for fair comparison (v2.1.7-A++ fix):
      x: [B, L=104, V=6] z-scored (target_z at feature 0, env channels 1-5)
      y: [B, H=4]        z-scored ili_weighted_pct (target only)

    Multivariate flatten extension:
      x → x_flat ∈ R^{L·V=624} → Generic block FC stack → backcast L·V + forecast H

    Paper-faithful univariate ablation (Supplementary §S.X):
      Set `target_only=True` → x[:, :, 0] only (input dim = L=104, no env).

    Stack composition (v2.1.7-A++ capacity sweep):
      Single stack of n_blocks Generic blocks.
      n_blocks default 3; validation-based grid sweeps {3, 6, 12, 24, 30}
      to test paper-faithful Generic-30 against data-appropriate sizes.
    """

    TARGET_IDX = 0   # ili_weighted_pct is feature 0 in V=6 layout

    def __init__(
        self,
        seq_len: int = 104,
        pred_len: int = 4,
        enc_in: int = 6,
        hidden: int = 512,
        n_blocks: int = 3,
        n_layers: int = 4,
        target_only: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.target_only = target_only
        effective_V = 1 if target_only else enc_in
        self.stack = NBeatsStack(
            n_blocks=n_blocks,
            lookback=seq_len,
            horizon=pred_len,
            enc_in=effective_V,
            hidden=hidden,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, V=6]. Returns: forecast [B, H=4] (target only)."""
        if self.target_only:
            x_sel = x[:, :, self.TARGET_IDX:self.TARGET_IDX + 1]   # [B, L, 1]
        else:
            x_sel = x                                                # [B, L, V]
        x_flat = x_sel.reshape(x_sel.size(0), -1)                   # [B, L·V or L]
        return self.stack(x_flat)
