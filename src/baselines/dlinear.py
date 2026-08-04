"""DLinear weekly baseline for CG-Mamba (v2.1.7-A++ baseline expansion).

References thuml/Time-Series-Library `models/DLinear.py` via sys.path injection.
DLinear (Zeng et al. 2023, AAAI) — "Are Transformers Effective for Time Series Forecasting?"
A simple decomposition (trend + seasonal) + linear projection. Standard mandatory
baseline for the forecasting community.

Spec (PLAN §7.1 v2.1.7-A++):
  - Input V=6, lookback L=104, pred_len H=4
  - Channel-independent (when individual=True) or shared linears (individual=False)
  - Multi-horizon head via linear projection
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[2]
_TSLIB_ROOT = _CG_MAMBA_ROOT.parent / "Time-Series-Library"

if not _TSLIB_ROOT.exists():
    raise ImportError(
        f"Time-Series-Library not found at {_TSLIB_ROOT}. "
        "Expected layout: /A.I_DATA/jbnu/JeongHa/{CG_Mamba, Time-Series-Library}"
    )

if str(_TSLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TSLIB_ROOT))

from models.DLinear import Model as _TslibDLinear  # type: ignore  # noqa: E402


class DLinearForecaster(nn.Module):
    """thuml DLinear adapter — V=6 input, multi-horizon target output.

    Input/output match LSTM/PatchTST baselines for fair comparison:
      x: [B, L=104, V=6] z-scored (target_z at feature 0)
      y: [B, H=4]        z-scored ili_weighted_pct
    """

    TARGET_IDX = 0   # ili_weighted_pct is feature 0 in V=6 layout

    def __init__(
        self,
        seq_len: int = 104,
        pred_len: int = 4,
        enc_in: int = 6,
        moving_avg: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        cfg = SimpleNamespace(
            task_name="short_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            moving_avg=moving_avg,
        )
        self.backbone = _TslibDLinear(cfg, individual=individual)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg = moving_avg
        self.individual = individual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # thuml signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
        # DLinear ignores x_mark_*/x_dec/x_mark_dec.
        out = self.backbone(x, None, None, None)   # [B, pred_len, V]
        return out[:, :, self.TARGET_IDX]          # [B, pred_len]
