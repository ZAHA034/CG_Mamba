"""Merge CDC ILI + national weekly environment into single weekly dataset.

Inputs:
  - data/raw/cdc_ilinet/national_weekly.csv      (CDC FluView national)
  - data/processed/env_national_weekly.csv       (NOAA ISD population-weighted national)

Join key: epiweek (YYYYWW).  Inner join — both sides must have the week.

Output schema (data/processed/ili_env_weekly.csv):
  date              MMWR Sunday (canonical from CDC; verified == env's date)
  year, week, epiweek
  ili_weighted_pct           CDC %wILI (target, national weighted average)
  ili_unweighted_pct         CDC unweighted ILI (auxiliary)
  total_ili_count            CDC raw ILI counts
  num_providers              CDC reporting providers
  num_patients               CDC patient denominator
  temperature_c              NOAA pop-weighted weekly mean temp (°C)
  specific_humidity_g_per_kg NOAA pop-weighted weekly mean q (g/kg)
  n_stations_available       (env diagnostic) — should be 10 throughout
  weight_sum_raw             (env diagnostic) — 1.0 means all 10 stations present

Quality checks:
  - epiweek alignment: CDC date == env date (MMWR Sunday) for every joined row
  - no duplicate epiweeks
  - no NaN in target (ili_weighted_pct) or env predictors

Usage:
    python -m src.data.build_merged_weekly
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
CDC_PATH = REPO_ROOT / "data" / "raw" / "cdc_ilinet" / "national_weekly.csv"
ENV_PATH = REPO_ROOT / "data" / "processed" / "env_national_weekly.csv"
OUT_DIR = REPO_ROOT / "data" / "processed"


def load_cdc() -> pd.DataFrame:
    df = pd.read_csv(CDC_PATH)
    df["epiweek"] = df["year"].astype(int) * 100 + df["week"].astype(int)
    keep = [
        "date", "year", "week", "epiweek",
        "ili_weighted_pct", "ili_unweighted_pct",
        "total_ili_count", "num_providers", "num_patients",
    ]
    return df[keep].copy()


def load_env() -> pd.DataFrame:
    df = pd.read_csv(ENV_PATH)
    keep = [
        "date", "year", "week", "epiweek",
        "temperature_c", "specific_humidity_g_per_kg",
        "n_stations_available", "weight_sum_raw",
    ]
    return df[keep].copy()


def merge(cdc: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on epiweek; verify date alignment; assemble final schema."""
    cdc_e = set(cdc["epiweek"])
    env_e = set(env["epiweek"])
    only_cdc = sorted(cdc_e - env_e)
    only_env = sorted(env_e - cdc_e)
    print(f"  CDC-only epiweeks ({len(only_cdc)}): "
          f"{only_cdc[:5]}{'...' if len(only_cdc) > 5 else ''}")
    print(f"  ENV-only epiweeks ({len(only_env)}): "
          f"{only_env[:5]}{'...' if len(only_env) > 5 else ''}")

    # Inner join on epiweek using suffixes to catch any date mismatch
    merged = cdc.merge(env, on="epiweek", how="inner", suffixes=("_cdc", "_env"))

    # Verify date alignment (MMWR Sunday must match between sources)
    bad = merged[merged["date_cdc"] != merged["date_env"]]
    if len(bad) > 0:
        print(f"  WARN: {len(bad)} rows have date mismatch between CDC and ENV",
              file=sys.stderr)
        print(bad[["epiweek", "date_cdc", "date_env"]].head(5), file=sys.stderr)
        raise ValueError("CDC/ENV date alignment failed for some epiweeks")
    print(f"  Date alignment OK: all {len(merged):,} joined rows agree on MMWR Sunday")

    # Use CDC date as canonical (identical to env date), drop dup columns
    out = pd.DataFrame({
        "date": merged["date_cdc"],
        "year": merged["year_cdc"].astype(int),
        "week": merged["week_cdc"].astype(int),
        "epiweek": merged["epiweek"].astype(int),
        "ili_weighted_pct": merged["ili_weighted_pct"].astype(float),
        "ili_unweighted_pct": merged["ili_unweighted_pct"].astype(float),
        "total_ili_count": merged["total_ili_count"].astype(int),
        "num_providers": merged["num_providers"].astype(int),
        "num_patients": merged["num_patients"].astype(int),
        "temperature_c": merged["temperature_c"].astype(float),
        "specific_humidity_g_per_kg": merged["specific_humidity_g_per_kg"].astype(float),
        "n_stations_available": merged["n_stations_available"].astype(int),
        "weight_sum_raw": merged["weight_sum_raw"].astype(float),
    })
    out = out.sort_values("epiweek").reset_index(drop=True)
    return out


def quality_checks(df: pd.DataFrame) -> None:
    """Fail loudly on any data quality red flag."""
    assert df["epiweek"].is_unique, "duplicate epiweeks in merged dataset"
    assert df["epiweek"].is_monotonic_increasing, "epiweeks not sorted"
    for col in ["ili_weighted_pct", "temperature_c", "specific_humidity_g_per_kg"]:
        n_nan = df[col].isna().sum()
        assert n_nan == 0, f"{col} has {n_nan} NaN rows"
    # Sanity ranges
    assert (df["ili_weighted_pct"] >= 0).all(), "negative ILI %"
    assert (df["ili_weighted_pct"] <= 20).all(), "ILI % > 20 (unphysical)"
    assert (df["temperature_c"] > -50).all() and (df["temperature_c"] < 50).all(), \
        "temperature_c out of plausible national-mean range"
    assert (df["specific_humidity_g_per_kg"] > 0).all() \
        and (df["specific_humidity_g_per_kg"] < 30).all(), "q out of plausible range"
    print(f"  Quality checks passed:")
    print(f"    rows:    {len(df):,}")
    print(f"    epiweek: {df['epiweek'].min()} → {df['epiweek'].max()}")
    print(f"    date:    {df['date'].min()} → {df['date'].max()}")
    print(f"    ILI %wILI:  mean={df['ili_weighted_pct'].mean():.2f}, "
          f"min={df['ili_weighted_pct'].min():.2f}, "
          f"max={df['ili_weighted_pct'].max():.2f}")
    print(f"    Temp (°C):  mean={df['temperature_c'].mean():.2f}, "
          f"min={df['temperature_c'].min():.2f}, "
          f"max={df['temperature_c'].max():.2f}")
    print(f"    q (g/kg):   mean={df['specific_humidity_g_per_kg'].mean():.2f}, "
          f"min={df['specific_humidity_g_per_kg'].min():.2f}, "
          f"max={df['specific_humidity_g_per_kg'].max():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str,
                    default=str(OUT_DIR / "ili_env_weekly.csv"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    print(f"[Merge] Reading CDC FluView: {CDC_PATH.relative_to(REPO_ROOT)}")
    cdc = load_cdc()
    print(f"  {len(cdc):,} CDC weekly records "
          f"(epiweek {cdc['epiweek'].min()} → {cdc['epiweek'].max()})")

    print(f"[Merge] Reading env weekly: {ENV_PATH.relative_to(REPO_ROOT)}")
    env = load_env()
    print(f"  {len(env):,} env weekly records "
          f"(epiweek {env['epiweek'].min()} → {env['epiweek'].max()})")

    print(f"[Merge] Inner-join on epiweek...")
    merged = merge(cdc, env)

    print(f"[Merge] Quality checks...")
    quality_checks(merged)

    # Save
    merged.to_csv(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    size = out_path.stat().st_size

    # Manifest
    manifest = {
        "_schema": "ili_env_weekly_manifest_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "pipeline": "src.data.build_merged_weekly (CG-Mamba M1.1)",
        "join": {
            "key": "epiweek",
            "how": "inner",
            "date_alignment_verified": "CDC MMWR Sunday == NOAA MMWR Sunday for all joined rows",
        },
        "sources": {
            "cdc_ilinet": {
                "path": str(CDC_PATH.relative_to(REPO_ROOT)),
                "n_records": len(cdc),
                "epiweek_first": int(cdc["epiweek"].min()),
                "epiweek_last": int(cdc["epiweek"].max()),
            },
            "env_national_weekly": {
                "path": str(ENV_PATH.relative_to(REPO_ROOT)),
                "n_records": len(env),
                "epiweek_first": int(env["epiweek"].min()),
                "epiweek_last": int(env["epiweek"].max()),
            },
        },
        "output": {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "size_bytes": size,
            "sha256": sha,
            "n_records": len(merged),
            "epiweek_first": int(merged["epiweek"].min()),
            "epiweek_last": int(merged["epiweek"].max()),
            "date_first": str(merged["date"].min()),
            "date_last": str(merged["date"].max()),
        },
        "columns": list(merged.columns),
        "target_column": "ili_weighted_pct",
        "predictor_columns": ["temperature_c", "specific_humidity_g_per_kg"],
        "auxiliary_columns": [
            "ili_unweighted_pct", "total_ili_count",
            "num_providers", "num_patients",
        ],
        "diagnostic_columns": ["n_stations_available", "weight_sum_raw"],
    }
    manifest_path = OUT_DIR / "ili_env_weekly_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"[Merge] Saved: {out_path.relative_to(REPO_ROOT)}")
    print(f"  size:     {size:,} bytes")
    print(f"  sha256:   {sha[:16]}...")
    print(f"  manifest: {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
