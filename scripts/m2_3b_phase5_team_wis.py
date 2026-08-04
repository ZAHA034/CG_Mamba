"""FluSight 2018-2019 historical teams — per-team WIS scoring (US National).

Phase 5 (PLAN §18.7 Plan A, CP 5.5b) — joins:
  1. Harmonized team submissions (23 quantiles per row, from CP 5.3 harmonizer)
  2. Ground-truth wILI (data/processed/ili_env_weekly_split.csv)

Computes WIS per (team, submission_ew, target_h) for **US National** only
(HHS-region ground truth requires separate fetch — Phase 3 integration).

Output:
  runs/phase_5_flusight/team_wis_2018_2019.csv  (long-form per-row WIS)
  runs/phase_5_flusight/team_wis_summary_2018_2019.csv  (aggregated by team × h)

Usage:
  python3 scripts/m2_3b_phase5_team_wis.py
  python3 scripts/m2_3b_phase5_team_wis.py --location 'US National'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.eval.wis import wis, coverage, REQUIRED_QUANTILES   # noqa: E402


HARMONIZED_CSV = _ROOT / "runs" / "phase_5_flusight" / "harmonized_2018-2019.csv"
GROUND_TRUTH_CSV = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
OUT_ROOT = _ROOT / "runs" / "phase_5_flusight"


def compute_row_wis_cov(row: pd.Series) -> tuple[float, float, float]:
    """Per-row WIS + 95% coverage + 50% coverage from harmonized quantile cols.

    Returns (wis, cov95, cov50). NaN tuple if y_true unavailable.
    """
    y = row["y_true"]
    if pd.isna(y):
        return (np.nan, np.nan, np.nan)
    # Build per-q dict of length-1 arrays (wis API expects array forecast)
    qf = {q: np.array([row[f"q_{q}"]]) for q in REQUIRED_QUANTILES}
    if any(np.isnan(v[0]) for v in qf.values()):
        return (np.nan, np.nan, np.nan)
    y_arr = np.array([y])
    wis_val = float(wis(y_arr, qf).mean())
    cov95 = float(coverage(y_arr, qf, alpha=0.05))
    cov50 = float(coverage(y_arr, qf, alpha=0.50))
    return (wis_val, cov95, cov50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="US National",
                    help="Currently only US National supported (HHS via Phase 3)")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    print(f"Loading harmonized teams: {HARMONIZED_CSV}")
    if not HARMONIZED_CSV.exists():
        raise FileNotFoundError(
            f"Missing {HARMONIZED_CSV}. Run m2_3b_phase5_harmonize_flusight.py first.")
    teams = pd.read_csv(HARMONIZED_CSV)
    teams = teams[teams["location"] == args.location].copy()
    print(f"  {len(teams)} rows after filter location='{args.location}'")
    print(f"  unique teams: {teams['team'].nunique()}, weeks: {teams['submission_ew'].nunique()}")

    print(f"Loading ground truth: {GROUND_TRUTH_CSV}")
    gt = pd.read_csv(GROUND_TRUTH_CSV)
    # NB: data/processed/ili_env_weekly_split.csv contains US National only
    gt_lookup = dict(zip(gt["epiweek"], gt["ili_weighted_pct"]))
    teams["y_true"] = teams["target_epiweek"].map(gt_lookup)
    n_with_gt = teams["y_true"].notna().sum()
    print(f"  matched ground truth for {n_with_gt}/{len(teams)} rows")

    # Compute WIS per row
    print("Computing WIS + Cov95 + Cov50 per row...")
    metrics = teams.apply(compute_row_wis_cov, axis=1, result_type="expand")
    metrics.columns = ["wis", "cov95", "cov50"]
    teams = pd.concat([teams, metrics], axis=1)
    n_valid_wis = teams["wis"].notna().sum()
    print(f"  valid WIS: {n_valid_wis}/{len(teams)}")

    # Per-row long output
    long_cols = ["team", "season", "submission_ew", "submission_epiweek",
                 "submission_date", "target", "target_h", "target_epiweek",
                 "location", "point", "y_true", "wis", "cov95", "cov50"]
    long_path = args.out_dir / "team_wis_2018_2019.csv"
    teams[long_cols].to_csv(long_path, index=False)
    print(f"Saved long: {long_path}")

    # Aggregate per team × target_h
    summary = (teams.dropna(subset=["wis"])
               .groupby(["team", "target_h"])
               .agg(wis_mean=("wis", "mean"),
                    wis_std=("wis", "std"),
                    cov95_mean=("cov95", "mean"),
                    cov50_mean=("cov50", "mean"),
                    n_weeks=("wis", "count"))
               .reset_index())
    summary_path = args.out_dir / "team_wis_summary_2018_2019.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Team-level avg-WIS ranking (mean across h=1..4)
    print("\n=== Per-team avg WIS ranking (lower = better) ===")
    rank = (teams.dropna(subset=["wis"])
            .groupby("team")
            .agg(wis_mean=("wis", "mean"),
                 cov95_mean=("cov95", "mean"),
                 n_obs=("wis", "count"))
            .sort_values("wis_mean")
            .reset_index())
    rank["rank"] = range(1, len(rank) + 1)
    rank_path = args.out_dir / "team_wis_ranking_2018_2019.csv"
    rank.to_csv(rank_path, index=False)
    print(f"Saved ranking: {rank_path}\n")
    print(rank.head(15).to_string(index=False))
    print(f"\n  ... ({len(rank)} total teams)")
    print(f"\nMedian WIS:   {rank['wis_mean'].median():.3f}")
    print(f"Best team:    {rank.iloc[0]['team']} (WIS={rank.iloc[0]['wis_mean']:.3f})")
    print(f"Median team:  {rank.iloc[len(rank)//2]['team']} (WIS={rank.iloc[len(rank)//2]['wis_mean']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
