"""iTransformer weekly baseline for CG-Mamba (v2.1.6 baseline expansion).

References thuml/Time-Series-Library `models/iTransformer.py` via sys.path
injection (MIT license, external standard library, not modified).

Spec (PLAN §7.1):
  - Input V=6 (same as PatchTST/LSTM baselines)
  - Lookback L=104, pred_len H=4
  - Inverted attention: variates-as-tokens (channel-mixing via attention)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cross-project import: thuml Time-Series-Library iTransformer
# ---------------------------------------------------------------------------
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

from models.iTransformer import Model as _TslibITransformer  # type: ignore  # noqa: E402


class ITransformerForecaster(nn.Module):
    """thuml iTransformer adapter — V=6 input, multi-horizon target output.

    Input/output:
      x: [B, L=104, V=6] z-scored
      y: [B, H=4]        z-scored ili_weighted_pct

    iTransformer treats each variate as a token; inversion does channel-mixing
    via attention. forecast() projects from [B, V, d_model] → [B, V, pred_len];
    we slice the target channel.
    """

    TARGET_IDX = 0   # ili_weighted_pct is feature 0

    def __init__(
        self,
        seq_len: int = 104,
        pred_len: int = 4,
        enc_in: int = 6,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        factor: int = 1,
        activation: str = "gelu",
        embed: str = "timeF",
        freq: str = "w",
    ):
        super().__init__()
        cfg = SimpleNamespace(
            task_name="short_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            factor=factor,
            activation=activation,
            embed=embed,
            freq=freq,
        )
        self.backbone = _TslibITransformer(cfg)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mark_* set to None: DataEmbedding_inverted handles x_mark=None branch
        out = self.backbone(x, None, None, None)   # [B, pred_len, V]
        return out[:, :, self.TARGET_IDX]          # [B, pred_len]
