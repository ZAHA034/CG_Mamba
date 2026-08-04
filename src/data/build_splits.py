"""Assign train/val/test split labels + fit train-only standardization scaler.

Architecture (per PLAN v2.0.5 §4.2):
  Single dataset + boundary metadata (NOT physical 3-file split). This avoids
  losing the lookback window at split boundaries: when serving val/test, the
  loader can still read predictors from preceding train weeks.

Split design (per PLAN v2.0.5 §4.1):
  Train: epiweek 200140 ~ 201839  (17 seasons: 2001-02 ~ 2017-18)
  Val:   epiweek 201840 ~ 202010  (truncate at W10-2020, pre-COVID onset)
  COVID excluded: epiweek 202011 ~ 202039 (held out — neither val nor test)
  Test:  epiweek 202040 ~ 202535  (~5 partial seasons: 2020-21 ~ 2024-25 partial)

  Test reporting (two rows in result tables):
    Test full:        all test epiweeks (includes COVID-era 2020-21 season)
    Test w/o COVID:   epiweek 202140 ~ 202535 (post-COVID 2021-22 ~ 2024-25)

Scaler: fit StandardScaler (mean, std) on TRAIN ONLY, save params to JSON.
  Applies to: ili_weighted_pct (target), temperature_c, specific_humidity_g_per_kg.

Outputs:
  data/processed/ili_env_weekly_split.csv     # original + 'split' column
  data/processed/split_boundaries.json        # epiweek boundary spec
  data/processed/normalization_params.json    # train-fit mean/std per column

Usage:
    python -m src.data.build_splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = REPO_ROOT / "data" / "processed" / "ili_env_weekly.csv"
OUT_DIR = REPO_ROOT / "data" / "processed"


# PLAN v2.0.5 §4.1 boundaries (inclusive ranges in MMWR epiweek YYYYWW form)
SPLIT_BOUNDARIES = {
    "train":           {"epiweek_first": 200140, "epiweek_last": 201839,
                        "description": "17 seasons (2001-02 ~ 2017-18)"},
    "val":             {"epiweek_first": 201840, "epiweek_last": 202010,
                        "description": "Val: W40-2018 ~ W10-2020 (truncate at COVID onset)"},
    "covid_excluded":  {"epiweek_first": 202011, "epiweek_last": 202039,
                        "description": "COVID hole: W11-2020 ~ W39-2020 — neither val nor test"},
    "test":            {"epiweek_first": 202040, "epiweek_last": 202535,
                        "description": "Test: W40-2020 ~ W35-2025 (~5 partial seasons)"},
}

# Sub-region of test for "w/o COVID" reporting row
TEST_POST_COVID = {"epiweek_first": 202140, "epiweek_last": 202535,
                   "description": "Test w/o COVID: 2021-22 ~ 2024-25 (post-COVID 4 seasons partial)"}

# Columns to standardize (target + env predictors)
SCALER_COLUMNS = ["ili_weighted_pct", "temperature_c", "specific_humidity_g_per_kg"]


def assign_split(epiweek: int) -> str:
    for name, b in SPLIT_BOUNDARIES.items():
        if b["epiweek_first"] <= epiweek <= b["epiweek_last"]:
            return name
    return "out_of_range"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Split] Reading: {IN_PATH.relative_to(REPO_ROOT)}")
    df = pd.read_csv(IN_PATH)
    print(f"  {len(df):,} rows, epiweek {df['epiweek'].min()} → {df['epiweek'].max()}")

    # Assign split labels
    df["split"] = df["epiweek"].apply(assign_split)
    n_oor = (df["split"] == "out_of_range").sum()
    if n_oor > 0:
        oor_eps = df.loc[df["split"] == "out_of_range", "epiweek"].tolist()
        print(f"  WARN: {n_oor} rows outside defined split boundaries: "
              f"{oor_eps[:5]}{'...' if n_oor > 5 else ''}", file=sys.stderr)

    counts = df["split"].value_counts().to_dict()
    print(f"  Split counts:")
    for name in ["train", "val", "covid_excluded", "test", "out_of_range"]:
        n = counts.get(name, 0)
        if n > 0:
            sub = df[df["split"] == name]
            print(f"    {name:<16}: {n:>4} rows  "
                  f"(epiweek {sub['epiweek'].min()} → {sub['epiweek'].max()})")

    # Fit scaler on TRAIN ONLY
    print(f"[Split] Fitting train-only StandardScaler on {SCALER_COLUMNS}...")
    train = df[df["split"] == "train"]
    norm_params = {}
    for col in SCALER_COLUMNS:
        mean = float(train[col].mean())
        std = float(train[col].std(ddof=0))   # population std (sklearn default)
        norm_params[col] = {
            "mean": mean, "std": std,
            "fit_on": "train",
            "fit_n_rows": len(train),
            "fit_epiweek_first": int(train["epiweek"].min()),
            "fit_epiweek_last": int(train["epiweek"].max()),
        }
        # Show per-split summary stat for sanity (no leakage — just diagnostic)
        print(f"  {col}:")
        print(f"    train fit:  mean={mean:.4f}, std={std:.4f}, n={len(train)}")
        for name in ["val", "test"]:
            sub = df[df["split"] == name]
            if len(sub) > 0:
                print(f"    {name:<10}: mean={sub[col].mean():.4f}, "
                      f"std={sub[col].std(ddof=0):.4f}, n={len(sub)}")

    # Save outputs
    out_csv = OUT_DIR / "ili_env_weekly_split.csv"
    df.to_csv(out_csv, index=False)
    sha_csv = hashlib.sha256(out_csv.read_bytes()).hexdigest()

    out_boundaries = OUT_DIR / "split_boundaries.json"
    boundaries_blob = {
        "_schema": "split_boundaries_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "design": ("Single dataset + boundary metadata (PLAN v2.0.5 §4.2). "
                   "Loader uses these epiweek ranges to filter; lookback windows "
                   "cross boundaries (predictors only)."),
        "splits": SPLIT_BOUNDARIES,
        "test_post_covid": TEST_POST_COVID,
        "scaler_columns": SCALER_COLUMNS,
        "row_counts": {name: int(counts.get(name, 0))
                       for name in SPLIT_BOUNDARIES.keys()},
    }
    with open(out_boundaries, "w") as f:
        json.dump(boundaries_blob, f, indent=2)

    out_norm = OUT_DIR / "normalization_params.json"
    norm_blob = {
        "_schema": "normalization_params_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "method": "StandardScaler (z-score: (x - mean) / std), std = population std (ddof=0)",
        "fit_on": "train split only — no test/val leakage",
        "input_dataset": str(IN_PATH.relative_to(REPO_ROOT)),
        "params": norm_params,
    }
    with open(out_norm, "w") as f:
        json.dump(norm_blob, f, indent=2)

    print()
    print(f"[Split] Saved:")
    print(f"  csv:         {out_csv.relative_to(REPO_ROOT)} "
          f"({out_csv.stat().st_size:,} bytes, sha256 {sha_csv[:16]}...)")
    print(f"  boundaries:  {out_boundaries.relative_to(REPO_ROOT)}")
    print(f"  norm params: {out_norm.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
