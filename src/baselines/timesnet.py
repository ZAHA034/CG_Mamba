"""TimesNet weekly baseline for CG-Mamba (v2.1.7-A++ M2.3 expansion).

References thuml/Time-Series-Library `models/TimesNet.py` via sys.path injection.
TimesNet (Wu et al. 2023, ICLR) — "TimesNet: Temporal 2D-Variation Modeling for
General Time Series Analysis". FFT-based period extraction + Inception 2D conv.
Standard recent transformer baseline for time series forecasting.

Spec (PLAN §7.1 v2.1.7-A++ M2.3):
  - Input V=6, lookback L=104, pred_len H=4
  - 2D-variation modeling via FFT top-k periods + Inception blocks
  - Multi-horizon head via linear projection
  - Channel-mixing (enc_in=6 → enc_in=6)
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

from models.TimesNet import Model as _TslibTimesNet  # type: ignore  # noqa: E402


class TimesNetForecaster(nn.Module):
    """thuml TimesNet adapter — V=6 input, multi-horizon target output.

    Input/output match other baselines for fair comparison:
      x: [B, L=104, V=6] z-scored (target_z at feature 0)
      y: [B, H=4]        z-scored ili_weighted_pct
    """

    TARGET_IDX = 0   # ili_weighted_pct is feature 0 in V=6 layout

    def __init__(
        self,
        seq_len: int = 104,
        pred_len: int = 4,
        enc_in: int = 6,
        d_model: int = 64,
        d_ff: int = 128,
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1,
        embed: str = "timeF",
        freq: str = "w",
        label_len: int = 0,
    ):
        super().__init__()
        cfg = SimpleNamespace(
            task_name="short_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            label_len=label_len,
            enc_in=enc_in,
            c_out=enc_in,        # output channels = input channels (channel-mixing)
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            top_k=top_k,
            num_kernels=num_kernels,
            dropout=dropout,
            embed=embed,
            freq=freq,
            num_class=0,
        )
        self.backbone = _TslibTimesNet(cfg)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # thuml signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
        # TimesNet's short_term_forecast path uses x_enc + x_mark_enc only.
        out = self.backbone(x, None, None, None)   # [B, pred_len, V]
        return out[:, :, self.TARGET_IDX]          # [B, pred_len]
