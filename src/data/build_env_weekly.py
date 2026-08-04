"""Build national-level weekly environmental time series from per-station daily data.

Pipeline:
  1. Read per-station daily Parquet (data/interim/noaa_daily/{isd_id}_daily.parquet)
  2. Map each date to MMWR (year, week) using `epiweeks`
  3. Weekly mean per station (temp_c, dew_c, specific_humidity_g_per_kg)
  4. Aggregate across 10 stations with population-weighted mean (Census 2020)
  5. Output single national weekly CSV aligned to CDC FluView's MMWR weeks

Output schema:
  date (MMWR Sunday) | year | week | epiweek (YYYYWW)
  temperature_c                  (population-weighted weekly mean air temperature, °C)
  specific_humidity_g_per_kg     (population-weighted weekly mean specific humidity, g/kg)
  n_stations_available           (count of stations with valid data this week, ≤ 10)
  n_daily_obs_mean               (mean daily observations across stations × days)

Output: data/processed/env_national_weekly.csv

Usage:
    python -m src.data.build_env_weekly
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from epiweeks import Week

from src.data.noaa_stations import MSA_STATIONS, STATION_WEIGHTS_DICT, TOTAL_POPULATION_2020


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "noaa_daily"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def date_to_epiweek(d: pd.Timestamp) -> tuple[int, int]:
    """MMWR (year, week) from any date."""
    w = Week.fromdate(d.date() if hasattr(d, "date") else d)
    return w.year, w.week


def date_to_mmwr_sunday(year: int, week: int) -> str:
    """MMWR (year, week) → ISO Sunday date string (week start)."""
    return Week(year, week).startdate().isoformat()


def load_station_daily(isd_id: str) -> pd.DataFrame:
    """Load daily Parquet for one station + attach MMWR (year, week)."""
    path = INTERIM_DIR / f"{isd_id}_daily.parquet"
    df = pd.read_parquet(path)
    # Daily 'date' field comes back as object — coerce
    df["date"] = pd.to_datetime(df["date"])
    # Assign MMWR (year, week) per row
    mmwr = df["date"].apply(date_to_epiweek)
    df["mmwr_year"] = mmwr.apply(lambda x: x[0])
    df["mmwr_week"] = mmwr.apply(lambda x: x[1])
    return df


def aggregate_station_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily → weekly mean per station, keyed by (mmwr_year, mmwr_week)."""
    agg = (
        daily.groupby(["mmwr_year", "mmwr_week"], as_index=False)
             .agg(
                 isd_id=("isd_id", "first"),
                 temp_c=("temp_c", "mean"),
                 dew_c=("dew_c", "mean"),
                 specific_humidity_g_per_kg=("specific_humidity_g_per_kg", "mean"),
                 n_days_with_data=("temp_c", "size"),
                 n_daily_obs_mean=("n_obs_hourly", "mean"),
             )
    )
    return agg


def population_weighted_aggregate(per_station_weekly: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate 10 per-station weekly DataFrames into one national weekly time series.

    Population weights from STATION_WEIGHTS_DICT (sums to 1.0).
    If a station is missing for a given week, its weight is redistributed
    proportionally among the available stations (renormalization).
    """
    # Concatenate all station weeklies
    long = pd.concat(per_station_weekly, ignore_index=True)

    # Attach population weight per station
    long["weight"] = long["isd_id"].map(STATION_WEIGHTS_DICT)

    # Group by (year, week); within each group, renormalize weights to sum to 1.0
    def _renorm_weighted_mean(group: pd.DataFrame) -> pd.Series:
        w = group["weight"]
        w_norm = w / w.sum()      # renormalize over available stations
        return pd.Series({
            "temperature_c": float((group["temp_c"] * w_norm).sum()),
            "specific_humidity_g_per_kg": float(
                (group["specific_humidity_g_per_kg"] * w_norm).sum()),
            "n_stations_available": int(len(group)),
            "weight_sum_raw": float(w.sum()),     # diagnostic: < 1.0 if stations missing
            "n_daily_obs_mean": float(group["n_daily_obs_mean"].mean()),
        })

    agg = (
        long.groupby(["mmwr_year", "mmwr_week"])
            .apply(_renorm_weighted_mean, include_groups=False)
            .reset_index()
    )

    # Add canonical MMWR Sunday date + epiweek (YYYYWW)
    agg["date"] = agg.apply(
        lambda r: date_to_mmwr_sunday(int(r["mmwr_year"]), int(r["mmwr_week"])), axis=1)
    agg["epiweek"] = agg["mmwr_year"] * 100 + agg["mmwr_week"]
    agg = agg.rename(columns={"mmwr_year": "year", "mmwr_week": "week"})

    # Final column order
    cols = ["date", "year", "week", "epiweek", "temperature_c",
            "specific_humidity_g_per_kg", "n_stations_available",
            "weight_sum_raw", "n_daily_obs_mean"]
    return agg[cols].sort_values("epiweek").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str,
                    default=str(PROCESSED_DIR / "env_national_weekly.csv"))
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    print(f"[Env weekly] Reading 10 station daily Parquets...")
    per_station_weekly = []
    for s in MSA_STATIONS:
        daily = load_station_daily(s.isd_id)
        weekly = aggregate_station_to_weekly(daily)
        per_station_weekly.append(weekly)
        print(f"  {s.isd_id} ({s.short_name:>12}): {len(daily):,} daily → "
              f"{len(weekly):,} weekly  (weight={STATION_WEIGHTS_DICT[s.isd_id]:.4f})")

    print()
    print(f"[Env weekly] Population-weighted aggregation across 10 MSAs...")
    national = population_weighted_aggregate(per_station_weekly)
    print(f"  {len(national):,} national-level weekly records")
    print(f"  Date range: {national['date'].min()} → {national['date'].max()}")
    print(f"  Epiweek range: {national['epiweek'].min()} → {national['epiweek'].max()}")
    print(f"  Temperature (°C):  mean={national['temperature_c'].mean():.2f}, "
          f"min={national['temperature_c'].min():.2f}, max={national['temperature_c'].max():.2f}")
    print(f"  Spec.hum (g/kg):   mean={national['specific_humidity_g_per_kg'].mean():.2f}, "
          f"min={national['specific_humidity_g_per_kg'].min():.2f}, "
          f"max={national['specific_humidity_g_per_kg'].max():.2f}")
    print(f"  Stations/week:     mean={national['n_stations_available'].mean():.2f}, "
          f"min={national['n_stations_available'].min()}, "
          f"max={national['n_stations_available'].max()}")
    print(f"  weight_sum_raw:    mean={national['weight_sum_raw'].mean():.4f}, "
          f"min={national['weight_sum_raw'].min():.4f}, "
          f"max={national['weight_sum_raw'].max():.4f}  (1.0 means all 10 stations had data)")

    # Save
    national.to_csv(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    size = out_path.stat().st_size

    # Manifest
    manifest = {
        "_schema": "env_weekly_national_manifest_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "pipeline": "src.data.build_env_weekly (CG-Mamba M1.1)",
        "source": "NOAA NCEI ISD daily → MMWR weekly → population-weighted national",
        "n_stations_used": len(MSA_STATIONS),
        "total_population_2020": TOTAL_POPULATION_2020,
        "method": (
            "1. Per station: hourly TMP+DEW (FM-15+FM-12, QC valid) → daily mean → "
            "specific humidity via Bolton (1980) formula.\n"
            "2. Daily → weekly mean per station, keyed by MMWR (year, week).\n"
            "3. Population-weighted aggregate across 10 MSAs (Census 2020). "
            "Missing-station weights redistributed proportionally."
        ),
        "citations": [
            "Bolton, D. (1980). The computation of equivalent potential temperature. "
            "Monthly Weather Review, 108(7), 1046-1053.",
            "Shaman, J. et al. (2013). Real-time influenza forecasts during the 2012-2013 season.",
            "Reich, N.G. et al. (2019). A collaborative multiyear, multimodel assessment "
            "of seasonal influenza forecasting in the United States. PNAS.",
            "US Census Bureau (2021). 2020 Census Apportionment Results.",
        ],
        "output": {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "size_bytes": size,
            "sha256": sha,
            "n_records": len(national),
            "date_first": str(national["date"].min()),
            "date_last": str(national["date"].max()),
        },
        "columns": list(national.columns),
        "stations_used": [
            {"isd_id": s.isd_id, "short_name": s.short_name,
             "iata": s.airport_iata, "pop2020": s.population_2020,
             "weight": STATION_WEIGHTS_DICT[s.isd_id]}
            for s in MSA_STATIONS
        ],
    }
    with open(PROCESSED_DIR / "env_weekly_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"[Env weekly] Saved: {out_path.relative_to(REPO_ROOT)}")
    print(f"  size:     {size:,} bytes")
    print(f"  sha256:   {sha[:16]}...")
    print(f"  manifest: data/processed/env_weekly_MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
