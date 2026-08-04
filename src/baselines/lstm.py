"""LSTM weekly baseline for CG-Mamba (v2.0.8b).

Spec (PLAN §7.1, §5.3 EB-6/EB-7):
  - Input V=6 (main model과 동일): ili_weighted_pct, total_ili_count,
    num_providers, num_patients, temperature_c, specific_humidity_g_per_kg
  - Multi-horizon head: pred_len=4 (1~4주 동시 출력)
  - Grid HP: hidden×layers×lr×batch_size = 4×3×3×2 = 72 configs
  - Fixed: lookback=104, dropout=0.0, epochs=100, patience=20
  - Selection: val_MAE @ h=1 minimum
  - Final: top-1% within best → 5-seed × 4-horizon mean ± std

Import policy (v2.0.8b Q-2 Option A):
  sys.path injection으로 CM_Mamba/cm_mamba/baselines/lstm_baseline.py의
  LSTMForecaster를 직접 import (복사 X, 의존성 단방향).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Cross-project import: CM_Mamba LSTMForecaster
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[2]  # CG_Mamba/
_PARENT_DIR = _CG_MAMBA_ROOT.parent  # JeongHa/
_CM_MAMBA_ROOT = _PARENT_DIR / "CM_Mamba"

if str(_CM_MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CM_MAMBA_ROOT))

from cm_mamba.baselines.lstm_baseline import LSTMForecaster  # type: ignore  # noqa: E402

# Local imports (loader's is_consecutive_epiweek)
_SRC_ROOT = _CG_MAMBA_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data.loader import is_consecutive_epiweek, load_dataset_csv, load_norm_params  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Constants (PLAN spec)
# ---------------------------------------------------------------------------
# V=6 input (main model과 동일, EB-6/EB-7)
LSTM_FEATURE_COLS = [
    "ili_weighted_pct",
    "total_ili_count",
    "num_providers",
    "num_patients",
    "temperature_c",
    "specific_humidity_g_per_kg",
]
TARGET_COL = "ili_weighted_pct"


# ---------------------------------------------------------------------------
# Multi-horizon dataset (gap-aware, loader.WeeklyDataset 동일 logic 확장)
# ---------------------------------------------------------------------------
class WeeklyMultiHorizonDataset(Dataset):
    """Gap-aware sliding window for LSTM multi-horizon (pred_len=4).

    Each sample:
        x: [L=104, V=6] — z-score / log1p (train-only fit)
        y: [H=4]        — z-scored target ili_weighted_pct at t+{1,2,3,4}

    Window 유효 조건 (loader.py WeeklyDataset 일관):
      * Lookback rows [start..end] 연속 epiweek
      * Target rows [end+1..end+H] 연속 epiweek AND 모두 self.split
      * Lookback은 cross-split 허용 (predictors only)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        norm: dict,
        lookback: int = 104,
        pred_len: int = 4,
    ):
        assert split in ("train", "val", "test"), split
        assert lookback >= 1 and pred_len >= 1
        self.df = df.reset_index(drop=True).copy()
        self.df["epiweek"] = self.df["epiweek"].astype(int)
        self.split = split
        self.lookback = lookback
        self.pred_len = pred_len
        self.norm = norm

        self._build_features()
        self._build_windows()

    def _build_features(self) -> None:
        """Z-score / log1p 정규화 → self.feats [N, V=6], self.target_z [N]."""
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

    def _build_windows(self) -> None:
        """Valid window end indices 구축 (gap-aware)."""
        df = self.df
        eps = df["epiweek"].to_numpy()
        splits = df["split"].to_numpy()
        N = len(df)
        L, H = self.lookback, self.pred_len
        valid = []

        # end ∈ [L-1, N-H-1]: lookback [end-L+1..end] (size L), target [end+1..end+H]
        # range exclusive end = N-H이므로 target_end = end + H ≤ N-1 (in bounds)
        for end in range(L - 1, N - H):
            target_start = end + 1
            target_end = end + H  # inclusive last target row

            # 모든 target이 self.split 안에 있어야 함
            if not all(splits[t] == self.split for t in range(target_start, target_end + 1)):
                continue

            # Lookback [end-L+1 .. end] 연속 epiweek
            lookback_start = end - L + 1
            ok = True
            for j in range(lookback_start, end):
                if not is_consecutive_epiweek(int(eps[j]), int(eps[j + 1])):
                    ok = False
                    break
            if not ok:
                continue

            # Lookback → target 연속 (end → end+1)
            if not is_consecutive_epiweek(int(eps[end]), int(eps[target_start])):
                continue

            # Target 윈도우 내부 연속 (end+1 → end+H)
            target_ok = True
            for t in range(target_start, target_end):
                if not is_consecutive_epiweek(int(eps[t]), int(eps[t + 1])):
                    target_ok = False
                    break
            if not target_ok:
                continue

            valid.append(end)

        self.window_ends = np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.window_ends))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.window_ends[idx])
        L, H = self.lookback, self.pred_len
        x = self.feats[end - L + 1 : end + 1]  # [L, V=6]
        y = self.target_z[end + 1 : end + H + 1]  # [H=4]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_lstm_loaders(
    csv_path: str | Path,
    norm_path: str | Path,
    lookback: int = 104,
    pred_len: int = 4,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build train + val DataLoaders for LSTM grid search.

    Test split은 grid phase에서 사용하지 않음 (val_MAE @ h=1으로 선정).
    """
    df = load_dataset_csv(Path(csv_path))
    norm = load_norm_params(Path(norm_path))

    train_ds = WeeklyMultiHorizonDataset(df, "train", norm, lookback, pred_len)
    val_ds = WeeklyMultiHorizonDataset(df, "val", norm, lookback, pred_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )

    metadata = {
        "n_train_windows": int(len(train_ds)),
        "n_val_windows": int(len(val_ds)),
        "lookback": lookback,
        "pred_len": pred_len,
        "input_dim": 6,
        "target_mean": float(train_ds.target_mean),
        "target_std": float(train_ds.target_std),
        "feature_cols": LSTM_FEATURE_COLS,
    }
    return train_loader, val_loader, metadata
