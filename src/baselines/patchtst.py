"""PatchTST weekly baseline for CG-Mamba (v2.1.6 baseline expansion).

References thuml/Time-Series-Library `models/PatchTST.py` via sys.path injection
(thuml repo is local at /A.I_DATA/jbnu/JeongHa/Time-Series-Library — MIT license,
external standard library, not modified).

Spec (PLAN §7.1):
  - Input V=6: ili_weighted_pct, total_ili_count, num_providers, num_patients,
    temperature_c, specific_humidity_g_per_kg (target = feature index 0)
  - Lookback L=104 weeks, pred_len H=4
  - Multi-horizon head (same MSE loss as LSTM / CG-Mamba for fair comparison)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cross-project import: thuml Time-Series-Library PatchTST
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

from models.PatchTST import Model as _TslibPatchTST  # type: ignore  # noqa: E402


class PatchTSTForecaster(nn.Module):
    """thuml PatchTST adapter — V=6 input, multi-horizon target output.

    Input/output match the LSTM baseline interface:
      x: [B, L=104, V=6] z-scored (target_z at feature 0)
      y: [B, H=4]        z-scored ili_weighted_pct

    thuml PatchTST predicts all V channels for H horizons (channel-independence
    Transformer); we slice the target channel.
    """

    TARGET_IDX = 0   # ili_weighted_pct is feature 0 in V=6 layout

    def __init__(
        self,
        seq_len: int = 104,
        pred_len: int = 4,
        enc_in: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        factor: int = 1,
        activation: str = "gelu",
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
        )
        self.backbone = _TslibPatchTST(cfg, patch_len=patch_len, stride=stride)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # thuml signature: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
        # PatchTST forecast() does not use x_mark_*/x_dec/x_mark_dec.
        out = self.backbone(x, None, None, None)   # [B, pred_len, V]
        return out[:, :, self.TARGET_IDX]          # [B, pred_len]
