"""Gap-aware sliding-window loader for the merged weekly ILI + env dataset.

Design (PLAN §4.2 single dataset + boundary metadata):
  * Single CSV with `split` column (`train` / `val` / `covid_excluded` / `test`).
  * Sliding window predicts y_{t+horizon} from features [t-L+1, ..., t].
  * **Cross-split lookback IS allowed** (predictors only): a val/test window's
    lookback can include preceding rows from other splits — but the TARGET
    row at position `end + horizon` must be in `self.split`.
  * Gap-aware: train has one CDC anomaly gap (200220 -> 200240). Any window
    that would span an epiweek gap is dropped, regardless of split.
  * Standardization: target + env features use the train-only mean/std loaded
    from `normalization_params.json` (no leakage).

Returned tensors:
    x        [L, main_input_dim=4]   — standardized ILI features
                                        [ili_weighted_pct_z,
                                         total_ili_count_z,
                                         num_providers_z,
                                         num_patients_z]
    env      [L, 2]                  — standardized env predictors
                                        [specific_humidity_g_per_kg_z,
                                         temperature_c_z]
    y        scalar                  — standardized target ili_weighted_pct_{t+horizon}
    info     dict                    — diagnostic metadata (epiweek, raw target)

For M1.2 we use ili counts as raw (no standardization) since they're not in
normalization_params.json. Only target + 2 env predictors are z-scored. ILI
counts are min-max-ish wide — we'll do a simple log1p transform inside this
loader (a known-good standardization for count-style ILI features).

⚠️ M1.7 DECISION POINT (L3 review): count features (`total_ili_count`,
`num_providers`, `num_patients`) currently use log1p ONLY (no z-score).
For M1.7 full integration + LSTM baseline fairness comparison, decide whether
to additionally z-score log1p outputs. Both options must be applied identically
to CG-Mamba and all baselines (LSTM, Vanilla Mamba + env concat, etc.). The
build_splits.py scaler currently fits only the 3 columns in `SCALER_COLUMNS`
(target + 2 env predictors); extending to count features requires extending
that fit list and re-emitting normalization_params.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def is_consecutive_epiweek(prev_ep: int, curr_ep: int) -> bool:
    """True iff curr_ep = next valid MMWR epiweek after prev_ep."""
    py, pw = prev_ep // 100, prev_ep % 100
    cy, cw = curr_ep // 100, curr_ep % 100
    if py == cy:
        return cw == pw + 1
    if cy == py + 1 and cw == 1 and pw in (52, 53):
        return True
    return False


@dataclass
class WindowSpec:
    start_row: int       # inclusive start of lookback in df
    end_row: int         # inclusive end of lookback in df  (= start + L - 1)
    target_row: int      # row used for target (= end_row + horizon)
    target_epiweek: int  # diagnostic


def _preprocess_window(
    sub: pd.DataFrame,
    norm: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract standardized (x_main, env) features from a single window slice.

    Shared by `WeeklyDataset.__getitem__` and `MultiHorizonDataset.__getitem__`
    (M1.7 L-1 cleanup) so the standardization policy lives in one place. Any
    future change (e.g., z-scoring log1p outputs per the L3 DECISION POINT
    above) must be applied here and propagates to both datasets.

    Args:
        sub:  window rows from the full DataFrame (len L).
        norm: normalization params dict (from normalization_params.json),
              providing mean/std for `ili_weighted_pct`,
              `specific_humidity_g_per_kg`, `temperature_c`.

    Returns:
        x_main: [L, 4] float64 — [ili_w_z, ili_count_log1p, n_prov_log1p,
                                   n_pat_log1p]
        env:    [L, 2] float64 — [humidity_z, temperature_z]
    """
    ili_w = (
        (sub["ili_weighted_pct"].to_numpy()
         - norm["ili_weighted_pct"]["mean"])
        / norm["ili_weighted_pct"]["std"]
    )
    ili_count = np.log1p(sub["total_ili_count"].to_numpy())
    n_prov = np.log1p(sub["num_providers"].to_numpy())
    n_pat = np.log1p(sub["num_patients"].to_numpy())
    x_main = np.stack([ili_w, ili_count, n_prov, n_pat], axis=-1)         # [L, 4]

    env_h = (
        (sub["specific_humidity_g_per_kg"].to_numpy()
         - norm["specific_humidity_g_per_kg"]["mean"])
        / norm["specific_humidity_g_per_kg"]["std"]
    )
    env_t = (
        (sub["temperature_c"].to_numpy()
         - norm["temperature_c"]["mean"])
        / norm["temperature_c"]["std"]
    )
    env = np.stack([env_h, env_t], axis=-1)                                # [L, 2]
    return x_main, env


class WeeklyDataset(Dataset):
    """Single split, single horizon, gap-aware sliding windows.

    Args:
        df:        full dataset DataFrame, sorted by epiweek.
        split:    'train' | 'val' | 'test'  — only windows fully inside the split.
        lookback: L (weeks)
        horizon:  forecast horizon (weeks ahead), 1..4.
        norm:     normalization params loaded from normalization_params.json.
    """

    MAIN_COLS = ["ili_weighted_pct", "total_ili_count",
                 "num_providers", "num_patients"]
    ENV_COLS = ["specific_humidity_g_per_kg", "temperature_c"]
    TARGET_COL = "ili_weighted_pct"

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        lookback: int,
        horizon: int,
        norm: dict,
    ):
        assert split in ("train", "val", "test"), split
        assert horizon >= 1
        self.df = df.reset_index(drop=True)
        self.split = split
        self.lookback = lookback
        self.horizon = horizon
        self.norm = norm
        self.windows = self._build_windows()

    def _build_windows(self) -> list[WindowSpec]:
        df = self.df
        eps = df["epiweek"].to_numpy()
        splits = df["split"].to_numpy()
        L, h = self.lookback, self.horizon
        N = len(df)
        windows = []
        for start in range(N - L - h + 1):
            end = start + L - 1
            target = end + h
            # Target row must be in self.split. Lookback rows may come from any
            # other split (predictors only — PLAN §4.2 single dataset design).
            if splits[target] != self.split:
                continue
            # All consecutive epiweeks (no gap inside window or window→target).
            # This is the only structural constraint on lookback rows.
            ok = True
            for j in range(start, target):
                if not is_consecutive_epiweek(int(eps[j]), int(eps[j + 1])):
                    ok = False
                    break
            if not ok:
                continue
            windows.append(WindowSpec(start, end, target, int(eps[target])))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        df = self.df
        sub = df.iloc[w.start_row:w.end_row + 1]                          # [L, ...]

        # L-1 cleanup: use shared `_preprocess_window` helper
        x_main, env = _preprocess_window(sub, self.norm)

        # Target: standardized ili_weighted_pct at horizon
        target_raw = float(df.iloc[w.target_row]["ili_weighted_pct"])
        y_std = (target_raw - self.norm["ili_weighted_pct"]["mean"]
                 ) / self.norm["ili_weighted_pct"]["std"]

        return {
            "x": torch.from_numpy(x_main.astype(np.float32)),
            "env": torch.from_numpy(env.astype(np.float32)),
            "y": torch.tensor(y_std, dtype=torch.float32),
            "y_raw": torch.tensor(target_raw, dtype=torch.float32),
            "target_epiweek": w.target_epiweek,
        }


class MultiHorizonDataset(Dataset):
    """Multi-horizon variant of WeeklyDataset (M1.7).

    Returns ALL horizon targets for each window in a single dict, so CGForecaster.
    forward(x, env) → [B, len(horizons)] can be matched against y [B, len(horizons)]
    in one loss computation pass.

    Sliding window structure (max_horizon-based):
      - Each window's lookback is [start, end] = [end - L + 1, end].
      - The maximum-horizon target is at end + max(horizons).
      - The window is admitted iff:
          (a) ALL target rows {end + h : h ∈ horizons} are in self.split.
              (M1.7 direction L-3: at the train/val boundary, the max-h target may
              fall outside split even when h<max-h targets are inside.)
          (b) Lookback + ALL target spans are consecutive in MMWR epiweek (gap-aware).

    Cross-split lookback policy (PLAN §4.2, retained from WeeklyDataset):
      Lookback rows may come from any other split (predictors only). TARGET rows
      must be in self.split.

    Returned dict:
        x        [L, main_input_dim=4]    standardized ILI features (z-score + log1p mix)
        env      [L, 2]                    standardized env predictors
        y        [len(horizons)]           z-scored target ili_weighted_pct per horizon
        y_raw    [len(horizons)]           raw %wILI per horizon (for denormalization)
        target_epiweeks  [len(horizons)]   diagnostic (list, kept off-tensor for collate)
    """

    MAIN_COLS = WeeklyDataset.MAIN_COLS
    ENV_COLS = WeeklyDataset.ENV_COLS
    TARGET_COL = WeeklyDataset.TARGET_COL

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        lookback: int,
        horizons: tuple[int, ...],
        norm: dict,
    ):
        assert split in ("train", "val", "test"), split
        assert len(horizons) >= 1 and all(int(h) >= 1 for h in horizons), horizons
        self.df = df.reset_index(drop=True)
        self.split = split
        self.lookback = lookback
        self.horizons = tuple(int(h) for h in horizons)
        self.max_horizon = max(self.horizons)
        self.norm = norm
        self.windows = self._build_windows()

    def _build_windows(self) -> list[WindowSpec]:
        df = self.df
        eps = df["epiweek"].to_numpy()
        splits = df["split"].to_numpy()
        L, max_h = self.lookback, self.max_horizon
        N = len(df)
        windows = []
        for start in range(N - L - max_h + 1):
            end = start + L - 1
            # (a) ALL horizon targets must be in self.split — explicit check
            # (max-h target containment does NOT guarantee h<max-h target containment
            #  at split boundaries, so verify each).
            all_targets_in_split = True
            for h in self.horizons:
                if splits[end + h] != self.split:
                    all_targets_in_split = False
                    break
            if not all_targets_in_split:
                continue
            # (b) Gap-aware: consecutive epiweeks across the full span
            # [start, end + max_h] — covers lookback and all per-horizon targets.
            ok = True
            for j in range(start, end + max_h):
                if not is_consecutive_epiweek(int(eps[j]), int(eps[j + 1])):
                    ok = False
                    break
            if not ok:
                continue
            windows.append(
                WindowSpec(start, end, end + max_h, int(eps[end + max_h]))
            )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        df = self.df
        sub = df.iloc[w.start_row:w.end_row + 1]                       # [L, ...]

        # L-1 cleanup: shared `_preprocess_window` helper (single source of truth
        # for feature standardization; both WeeklyDataset and MultiHorizonDataset
        # use this so future scaling-policy changes propagate to both).
        x_main, env = _preprocess_window(sub, self.norm)

        # Multi-horizon targets
        mean_ili = self.norm["ili_weighted_pct"]["mean"]
        std_ili = self.norm["ili_weighted_pct"]["std"]
        ys_z, ys_raw, eps_list = [], [], []
        for h in self.horizons:
            target_row = w.end_row + h
            raw = float(df.iloc[target_row]["ili_weighted_pct"])
            ys_raw.append(raw)
            ys_z.append((raw - mean_ili) / std_ili)
            eps_list.append(int(df.iloc[target_row]["epiweek"]))

        return {
            "x": torch.from_numpy(x_main.astype(np.float32)),
            "env": torch.from_numpy(env.astype(np.float32)),
            "y": torch.tensor(ys_z, dtype=torch.float32),               # [H]
            "y_raw": torch.tensor(ys_raw, dtype=torch.float32),         # [H]
            "target_epiweeks": eps_list,                                # list[int]
        }


def load_dataset_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("epiweek").reset_index(drop=True)
    return df


def load_norm_params(json_path: Path) -> dict:
    with open(json_path) as f:
        blob = json.load(f)
    return blob["params"]


def collate_dict(batch: list[dict]) -> dict:
    """Stack tensor entries; pass through scalars."""
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out
