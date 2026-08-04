"""FluSight 2014-2020 historical submission harmonizer — Bin → 23 FluSight quantiles.

Phase 5 (PLAN §18.7 Plan A, B1 external validation): converts historical
team submissions (cdcepi/FluSight-forecasts, SHA b1ac383) from the legacy
"bin histogram" format to the standardized 23-quantile FluSight protocol
used by Bracher et al. 2021 WIS scoring.

Input format (per file EW{NN}-{team}-{date}.csv):
    location, target, unit, type, bin_start_incl, bin_end_notincl, value
    Types: "Bin" (~131 bins of 0.1% wILI width covering 0.0~13.0%)
           "Point" (1 row per target — point forecast)
    Targets: "1 wk ahead", "2 wk ahead", "3 wk ahead", "4 wk ahead"
    Locations: "US National" + "HHS Region 1" ... "HHS Region 10"

Output format (long-form CSV per season):
    team, season, submission_ew, target, location, point,
    q_0.01, q_0.025, q_0.05, q_0.1, q_0.15, ..., q_0.95, q_0.975, q_0.99

Quantile interpolation:
    CDF(x_end_of_bin) = cumulative sum of bin probabilities (post-normalize).
    Q(q) = linear interpolation of x_end_of_bin at level q.

Usage:
    python3 scripts/m2_3b_phase5_harmonize_flusight.py
    python3 scripts/m2_3b_phase5_harmonize_flusight.py --seasons 2018-2019 2019-2020
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
from epiweeks import Week


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.eval.wis import REQUIRED_QUANTILES                          # noqa: E402


FLUSIGHT_ROOT = _ROOT / "external" / "FluSight-forecasts"
OUT_ROOT = _ROOT / "runs" / "phase_5_flusight"

# Targets used in the paper (matches our H = [1, 2, 3, 4])
ACCEPTED_TARGETS = ("1 wk ahead", "2 wk ahead", "3 wk ahead", "4 wk ahead")

# Filename pattern: EW{week:NN}-{team}-{YYYY-MM-DD}.csv
# Per README.md: EW{week} = latest MMWR week of data used in the forecast.
# So "1 wk ahead" = forecast for EW{week+1}.
_FN_PATTERN = re.compile(r"^EW(\d+)-(.+?)-(\d{4})-(\d{2})-(\d{2})\.csv$")


def parse_filename(name: str) -> dict | None:
    """Extract submission_ew + team name + submission_date + submission_epiweek.

    Year resolution heuristic (handles cross-year season files):
      - filename EW ≥ 40 AND submit month ≤ 6 → year = submit_year − 1
      - filename EW ≤ 30 AND submit month ≥ 10 → year = submit_year + 1
      - else → year = submit_year

    Returns None on unparseable filename.
    """
    m = _FN_PATTERN.match(name)
    if not m:
        return None
    ew_num = int(m.group(1))
    team = m.group(2)
    submit_date = Date(int(m.group(3)), int(m.group(4)), int(m.group(5)))

    sub_year = submit_date.year
    if ew_num >= 40 and submit_date.month <= 6:
        sub_year -= 1
    elif ew_num <= 30 and submit_date.month >= 10:
        sub_year += 1

    submission_week = Week(sub_year, ew_num)
    submission_epiweek = sub_year * 100 + ew_num
    return {
        "submission_ew": ew_num,
        "submission_year": sub_year,
        "submission_epiweek": submission_epiweek,
        "submission_date": submit_date.isoformat(),
        "_week_obj": submission_week,
        "team_from_file": team,
    }


def target_epiweek_for(submission_week: Week, h: int) -> int:
    """Compute target epiweek (year*100+week) for h-week-ahead.

    Per FluSight 2018-19 README: "1 wk ahead" = forecast week immediately
    after the latest-data EW. So h=1 → submission_week + 1.
    epiweeks library handles year boundary automatically.
    """
    tw = submission_week + h
    return tw.year * 100 + tw.week


def bin_to_quantiles(bins_df: pd.DataFrame) -> dict[float, float]:
    """Convert bin histogram → 23 FluSight quantiles via inverse-CDF.

    Args:
        bins_df: DataFrame with columns `bin_start_incl`, `bin_end_notincl`,
                 `value`. Filtered to one (target, location), Bin type only.

    Returns:
        dict mapping quantile level → wILI value. NaN if total probability ~0.
    """
    df = bins_df.copy()
    df["bin_start_incl"] = pd.to_numeric(df["bin_start_incl"], errors="coerce")
    df["bin_end_notincl"] = pd.to_numeric(df["bin_end_notincl"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["bin_start_incl", "bin_end_notincl", "value"])
    df = df[df["value"] >= 0]
    df = df.sort_values("bin_start_incl").reset_index(drop=True)

    if len(df) == 0:
        return {q: float("nan") for q in REQUIRED_QUANTILES}

    total = df["value"].sum()
    if total < 1e-9:
        return {q: float("nan") for q in REQUIRED_QUANTILES}

    probs = (df["value"].to_numpy() / total).astype(np.float64)
    cdf = np.cumsum(probs)
    # CDF reaches "value at end of bin" → x = bin_end_notincl
    x = df["bin_end_notincl"].to_numpy(dtype=np.float64)

    out: dict[float, float] = {}
    for q in REQUIRED_QUANTILES:
        # np.interp clamps to first/last value beyond range — desired behavior.
        out[float(q)] = float(np.interp(q, cdf, x))
    return out


def harmonize_team_file(file: Path, submission_week: Week | None = None) -> pd.DataFrame | None:
    """Convert one team-week submission → long DataFrame.

    Returns DataFrame with one row per (target, location) pair, columns:
        target, location, target_h, target_epiweek, point, q_<q1>, q_<q2>, ...
    """
    try:
        df = pd.read_csv(file)
    except Exception:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"target", "location", "type", "value",
              "bin_start_incl", "bin_end_notincl"}
    if not needed.issubset(df.columns):
        return None

    rows = []
    for (target, location), sub in df.groupby(["target", "location"]):
        if target not in ACCEPTED_TARGETS:
            continue
        h = int(target.split()[0])  # "1 wk ahead" → 1
        target_ew = (target_epiweek_for(submission_week, h)
                     if submission_week is not None else None)
        pt = sub[sub["type"].str.lower() == "point"]
        point = float(pt["value"].iloc[0]) if len(pt) == 1 else float("nan")
        bins = sub[sub["type"].str.lower() == "bin"]
        if len(bins) < 10:
            continue   # malformed bin set — skip
        qmap = bin_to_quantiles(bins[["bin_start_incl", "bin_end_notincl", "value"]])
        row = {"target": target, "target_h": h, "target_epiweek": target_ew,
               "location": location, "point": point}
        row.update({f"q_{q}": v for q, v in qmap.items()})
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def harmonize_season(season: str, season_dir: Path) -> pd.DataFrame:
    """Walk all teams in a season → concatenated long DataFrame."""
    if not season_dir.exists():
        print(f"  [skip] {season} dir not found")
        return pd.DataFrame()
    parts = []
    teams = sorted([d for d in season_dir.iterdir() if d.is_dir()])
    for team_dir in teams:
        team = team_dir.name
        csvs = sorted(team_dir.glob("EW*-*-*.csv"))
        if not csvs:
            continue
        n_ok, n_total = 0, len(csvs)
        for fp in csvs:
            meta = parse_filename(fp.name)
            if meta is None:
                continue
            sub = harmonize_team_file(fp, submission_week=meta["_week_obj"])
            if sub is None or len(sub) == 0:
                continue
            sub.insert(0, "team", team)
            sub.insert(1, "season", season)
            sub.insert(2, "submission_ew", meta["submission_ew"])
            sub.insert(3, "submission_epiweek", meta["submission_epiweek"])
            sub.insert(4, "submission_date", meta["submission_date"])
            parts.append(sub)
            n_ok += 1
        print(f"  {team:30s}  {n_ok}/{n_total} weeks parsed")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+",
                    default=["2018-2019"],
                    help="Season directory names under external/FluSight-forecasts/")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for season in args.seasons:
        print(f"\n=== Season {season} ===")
        df = harmonize_season(season, FLUSIGHT_ROOT / season)
        if len(df) == 0:
            print(f"  no rows produced for {season}")
            continue
        out_path = args.out_dir / f"harmonized_{season}.csv"
        df.to_csv(out_path, index=False)
        s = {
            "season": season,
            "rows": len(df),
            "teams": df["team"].nunique(),
            "weeks": df["submission_ew"].nunique(),
            "targets": df["target"].nunique(),
            "locations": df["location"].nunique(),
            "file_kb": int(out_path.stat().st_size / 1024),
        }
        summary.append(s)
        print(f"  Saved: {out_path}")
        print(f"    rows={s['rows']}  teams={s['teams']}  weeks={s['weeks']}  "
              f"targets={s['targets']}  locations={s['locations']}  "
              f"({s['file_kb']} KB)")
    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['season']:12s}  {s['rows']:>7d} rows  {s['teams']:>3d} teams  {s['weeks']:>3d} weeks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
