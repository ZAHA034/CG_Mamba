"""Vanilla Mamba weekly baseline for CG-Mamba (v2.1.7-A++ M2.3 expansion).

Reuses internal CGMambaBackbone (M1.2 vanilla architecture, src/models/backbone.py)
with use_gate=False. Wrapped as a standard Pattern A baseline with multi-horizon
output, matching other NN baselines (LSTM, PatchTST, DLinear, TimesNet, etc.).

A1 ablation rationale (PLAN §7.4):
  CG-Mamba = CGMambaEncoder (ContextGatedMambaBlock with gate_proj per layer)
  Vanilla Mamba = CGMambaBackbone (CGMambaBlock, gate_proj absent)
  Same SSM core (selective_scan, dt-projection); only the per-layer gating differs.

For fair comparison, Vanilla Mamba receives its own Pattern A HPO grid
(d_model × n_layers × lr, 12 cfg), matching LSTM/PatchTST/iTransformer/DLinear/
TimesNet/N-BEATS/EpiDeep — so the comparison is "best-tuned Vanilla Mamba"
vs CG-Mamba, not "CG-Mamba's HP minus gate" (which would be an under-tuned
Vanilla Mamba unfairly handicapping the baseline).

Input/output match all NN baselines (v2.1.7-A++ multivariate):
  x: [B, L=104, V=6] z-scored (target_z at feature 0, env channels 1-5)
  y: [B, H=4]        z-scored ili_weighted_pct (target only)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.nn as nn

import sys as _sys
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from src.models.backbone import CGMambaBackbone           # noqa: E402
from src.utils.config import CGMambaConfig                # noqa: E402


class VanillaMambaForecaster(nn.Module):
    """Vanilla Mamba (gate-less) multi-horizon forecaster.

    Architecture:
      input_proj (V → d_model) → n_layers × CGMambaBlock (no gate) → RMSNorm
                              → take last timestep h[:, -1, :]
                              → Linear (d_model → pred_len) → forecast [B, H]

    Multi-horizon strategy: pool sequence by last-timestep (consistent with
    LSTM-style decoders). Alternative pooling (mean, attention) considered but
    last-timestep is the canonical SSM-decoder convention.
    """
    TARGET_IDX = 0   # ili_weighted_pct is feature 0 in V=6 layout

    def __init__(
        self,
        seq_len: int = 104,
        pred_len: int = 4,
        enc_in: int = 6,
        d_model: int = 64,
        n_layers: int = 3,
        d_state: int = 16,
        dt_rank: int = 16,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Build CGMambaConfig with multivariate input + no gate
        cfg = dataclasses.replace(
            CGMambaConfig(),
            d_model=d_model,
            d_state=d_state,
            dt_rank=dt_rank,
            expand=expand,
            n_layers=n_layers,
            dropout=dropout,
            use_gate=False,                   # vanilla mamba (M1.2)
            lookback=seq_len,
            main_input_dim=enc_in,            # V=6 multivariate
            horizons=tuple(range(1, pred_len + 1)),
        )
        self.cfg = cfg

        # Vanilla backbone (gate-less)
        self.backbone = CGMambaBackbone(cfg)

        # Multi-horizon forecast head
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, V=6]. Returns: forecast [B, H=4] (target ili z-scored)."""
        h = self.backbone(x, gates=None)       # [B, L, D]
        h_last = h[:, -1, :]                    # [B, D]
        return self.head(h_last)                # [B, H]
