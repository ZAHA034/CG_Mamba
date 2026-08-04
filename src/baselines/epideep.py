"""EpiDeep weekly baseline for CG-Mamba (v2.1.7-A++ M2.3 expansion) — Faithful.

Faithful from-scratch implementation following Adhikari et al. (2019, KDD):
  "EpiDeep: Exploiting Embeddings for Epidemic Forecasting"

Paper-canonical architecture:
  1. Season embedding bank: learnable [N_train_seasons, d_emb] table.
     Each slot represents an "ILI season pattern" captured during training.
  2. Current observation encoder: LSTM over the lookback window (multivariate
     V=6 input → query vector ∈ R^{d_emb}).
  3. Similarity attention: cosine-scored softmax over the embedding bank.
     ctx = softmax(query · E^T / √d) · E.
  4. Decoder: MLP([query ‖ ctx]) → forecast h=1..H.

  + (OPTIONAL, OFF by default) Season alignment auxiliary loss:
      Each training sample's target is in season s_id (CDC W40–W39 convention).
      L_align = MSE(query, E[s_id])  → encourages the bank to become
      season-discriminative.

      Default α=0.0 (paper-faithful end-to-end attention only).
      α>0 reserved for Supplementary §S.X ablation.

  Total loss = L_forecast + α · L_align  (α=0.0 default = paper-faithful).

Reference:
  Adhikari, B., Xu, X., Ramakrishnan, N., & Prakash, B.A. (2019).
  "EpiDeep: Exploiting Embeddings for Epidemic Forecasting." KDD 2019.
  https://dl.acm.org/doi/10.1145/3292500.3330917

Notes (v2.1.7-A++ fairness fix)
-------------------------------
- Multivariate V=6 input: encoder LSTM takes all 6 features (ILI + 5 env channels),
  consistent with all other NN baselines (LSTM/PatchTST/iTransformer/DLinear/
  TimesNet/Vanilla Mamba/CG-Mamba). Original paper EpiDeep is univariate on
  surveillance signal alone; the multivariate extension is trivial (encoder
  input_size change) and faithful to the paper's spirit (the season-retrieval
  mechanism is unchanged). Univariate ablation available via `target_only=True`.
- Training season span: 17 seasons (2001-02 ~ 2017-18) → N_train_seasons=17.
- At validation/test (post-2018 seasons), the embedding bank attends over the 17
  learned past-season embeddings to forecast new seasons.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Season definition (CDC FluSight convention):
#   Season s = W40-(s+SEASON_BASE_YEAR) ~ W39-(s+SEASON_BASE_YEAR+1)
#   season 0 = 2001-02 season (W40-2001 ~ W39-2002)
SEASON_BASE_YEAR = 2001
N_TRAIN_SEASONS = 17        # 2001-02 ~ 2017-18 = 17 seasons


def season_id_of_epiweek(epiweek: int) -> int:
    """Convert epiweek to season_id (relative to SEASON_BASE_YEAR=2001-02).
    Returns -1 if before training start, ≥N_TRAIN_SEASONS if after training end.
    """
    year = epiweek // 100
    week = epiweek % 100
    season_year = year if week >= 40 else year - 1
    return season_year - SEASON_BASE_YEAR


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class EpiDeepDataset(Dataset):
    """Weekly ILI dataset with season_id annotation per sample.

    Each sample:
      x:         [L=104, V=6] z-scored lookback features (target_z at feat 0)
      season_id: int (the season of the FIRST target week)
                 - for training-split samples: 0..16
                 - for val/test-split samples: ≥ N_TRAIN_SEASONS (used only for
                   logging, NOT for attention bank lookup)
      y:         [H=4] z-scored target ili_weighted_pct
    """
    TARGET_COL = "ili_weighted_pct"
    FEATURE_COLS = [
        "ili_weighted_pct",
        "total_ili_count",
        "num_providers",
        "num_patients",
        "temperature_c",
        "specific_humidity_g_per_kg",
    ]

    def __init__(self, df: pd.DataFrame, split: str, norm: dict,
                 lookback: int = 104, pred_len: int = 4):
        assert split in ("train", "val", "test"), split
        self.df = df.reset_index(drop=True).copy()
        self.df["epiweek"] = self.df["epiweek"].astype(int)
        self.split = split
        self.lookback = lookback
        self.pred_len = pred_len
        self.norm = norm
        self._build_features()
        self._build_windows()

    def _build_features(self):
        df = self.df
        N = len(df)
        ili_p = self.norm["ili_weighted_pct"]
        temp_p = self.norm["temperature_c"]
        hum_p = self.norm["specific_humidity_g_per_kg"]

        feats = np.zeros((N, 6), dtype=np.float32)
        feats[:, 0] = (df["ili_weighted_pct"].to_numpy() - ili_p["mean"]) / ili_p["std"]
        feats[:, 1] = np.log1p(df["total_ili_count"].to_numpy())
        feats[:, 2] = np.log1p(df["num_providers"].to_numpy())
        feats[:, 3] = np.log1p(df["num_patients"].to_numpy())
        feats[:, 4] = (df["temperature_c"].to_numpy() - temp_p["mean"]) / temp_p["std"]
        feats[:, 5] = (df["specific_humidity_g_per_kg"].to_numpy() - hum_p["mean"]) / hum_p["std"]
        self.feats = feats
        target_z = (df["ili_weighted_pct"].to_numpy() - ili_p["mean"]) / ili_p["std"]
        self.target_z = target_z.astype(np.float32)
        self.target_mean = ili_p["mean"]
        self.target_std = ili_p["std"]

    def _build_windows(self):
        from data.loader import is_consecutive_epiweek  # type: ignore
        df = self.df
        eps = df["epiweek"].to_numpy()
        splits = df["split"].to_numpy()
        N = len(df)
        L, H = self.lookback, self.pred_len

        indices: list[tuple[int, int, int]] = []   # (end, target_first_idx, season_id)
        for end in range(L - 1, N - H):
            ok = True
            # lookback continuity
            for j in range(end - L + 1, end):
                if not is_consecutive_epiweek(int(eps[j]), int(eps[j + 1])):
                    ok = False; break
            if not ok:
                continue
            # target continuity
            for j in range(end, end + H):
                if not is_consecutive_epiweek(int(eps[j]), int(eps[j + 1])):
                    ok = False; break
            if not ok:
                continue
            # target rows in this split
            if not all(splits[end + 1 + k] == self.split for k in range(H)):
                continue
            target_first = end + 1
            s_id = season_id_of_epiweek(int(eps[target_first]))
            indices.append((end, target_first, s_id))
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end, target_first, season_id = self.indices[idx]
        L, H = self.lookback, self.pred_len
        x = self.feats[end - L + 1 : end + 1]            # [L, V]
        y = self.target_z[target_first : target_first + H]  # [H]
        return (
            torch.from_numpy(x),
            torch.tensor(season_id, dtype=torch.long),
            torch.from_numpy(y),
        )


def build_epideep_loaders(csv_path: Path, norm_path: Path,
                          lookback: int = 104, pred_len: int = 4,
                          batch_size: int = 16):
    """Build train/val EpiDeep loaders."""
    import json
    from torch.utils.data import DataLoader
    import sys as _sys
    _REPO = Path(__file__).resolve().parents[2]
    _SRC = _REPO / "src"
    if str(_SRC) not in _sys.path:
        _sys.path.insert(0, str(_SRC))
    from data.loader import load_dataset_csv  # type: ignore

    df = load_dataset_csv(csv_path)
    norm = json.loads(Path(norm_path).read_text())["params"]

    train_ds = EpiDeepDataset(df, "train", norm, lookback, pred_len)
    val_ds = EpiDeepDataset(df, "val", norm, lookback, pred_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ili_p = norm["ili_weighted_pct"]
    meta = {
        "target_mean": ili_p["mean"],
        "target_std": ili_p["std"],
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "feature_cols": EpiDeepDataset.FEATURE_COLS,
        "n_train_seasons": N_TRAIN_SEASONS,
    }
    return train_loader, val_loader, meta


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class SeasonEmbeddingAttention(nn.Module):
    """Cosine-scored softmax attention over a learnable season embedding bank."""

    def __init__(self, n_seasons: int, d_emb: int):
        super().__init__()
        self.n_seasons = n_seasons
        self.d_emb = d_emb
        self.bank = nn.Embedding(n_seasons, d_emb)
        nn.init.normal_(self.bank.weight, std=0.1)
        self.scale = d_emb ** 0.5

    def forward(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """query: [B, d_emb]. Returns: (context [B, d_emb], weights [B, N])."""
        keys = self.bank.weight       # [N, d_emb]
        scores = (query @ keys.T) / self.scale     # [B, N]
        weights = F.softmax(scores, dim=-1)
        context = weights @ keys                   # [B, d_emb]
        return context, weights


class EpiDeepForecaster(nn.Module):
    """EpiDeep faithful — Adhikari et al. 2019 KDD (v2.1.7-A++ multivariate fix).

    Forward signature:
      forward(x, season_id=None) → forecast
        x:         [B, L, V=6]  z-scored multivariate input
        season_id: [B] (long) — used only by alignment auxiliary loss (α>0 case)
        forecast:  [B, H=4]    z-scored predictions for target channel

    Loss (during training):
      L = MSE(forecast, y) + α · MSE(query, bank[season_id])
      α (`alignment_weight`) defaults to **0.0** (paper-faithful Adhikari 2019).
      α>0 reserved for Supplementary §S.X ablation.
      At eval, return forecast only.

    Univariate (paper-original) ablation:
      `target_only=True` → encoder uses only TARGET_IDX=0 channel
                          (matches strictest paper-faithful EpiDeep).
    """
    TARGET_IDX = 0

    def __init__(
        self,
        seq_len: int = 104,
        pred_len: int = 4,
        enc_in: int = 6,
        d_emb: int = 64,
        encoder_hidden: int = 128,
        decoder_hidden: int = 128,
        n_train_seasons: int = N_TRAIN_SEASONS,
        alignment_weight: float = 0.0,      # v2.1.7-A++: paper-faithful default
        dropout: float = 0.1,
        target_only: bool = False,           # v2.1.7-A++: V=6 multivariate default
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_emb = d_emb
        self.alignment_weight = alignment_weight
        self.n_train_seasons = n_train_seasons
        self.target_only = target_only

        # 1. Encoder: LSTM over multivariate input (V=6) or univariate target
        effective_input_size = 1 if target_only else enc_in
        self.encoder = nn.LSTM(
            input_size=effective_input_size, hidden_size=encoder_hidden,
            num_layers=1, batch_first=True, dropout=0.0,
        )
        self.encoder_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden, d_emb),
        )

        # 2. Season embedding bank + attention
        self.attention = SeasonEmbeddingAttention(n_train_seasons, d_emb)

        # 3. Decoder: [query ‖ context] → forecast
        self.decoder = nn.Sequential(
            nn.Linear(2 * d_emb, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, pred_len),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, V]. Returns: query [B, d_emb]."""
        if self.target_only:
            x_in = x[:, :, self.TARGET_IDX:self.TARGET_IDX + 1]   # [B, L, 1]
        else:
            x_in = x                                                # [B, L, V=6]
        _, (h_n, _) = self.encoder(x_in)
        query = self.encoder_proj(h_n.squeeze(0))    # [B, d_emb]
        return query

    def forward(self, x: torch.Tensor,
                season_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.encode(x)                       # [B, d_emb]
        context, _ = self.attention(query)           # [B, d_emb]
        combined = torch.cat([query, context], dim=-1)   # [B, 2*d_emb]
        forecast = self.decoder(combined)            # [B, H]
        return forecast

    def compute_alignment_loss(self, x: torch.Tensor, season_id: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss: encoder(x) should be close to bank[season_id_clamped]."""
        # Clamp season_id to training range so val/test samples (s_id ≥ N) are ignored
        s_id_clamped = season_id.clamp(min=0, max=self.n_train_seasons - 1)
        valid_mask = (season_id >= 0) & (season_id < self.n_train_seasons)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=x.device)
        query = self.encode(x)                       # [B, d_emb]
        target_emb = self.attention.bank(s_id_clamped)   # [B, d_emb]
        diff = (query - target_emb) ** 2
        return diff[valid_mask].mean()
