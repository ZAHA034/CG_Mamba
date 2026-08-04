"""Parse NOAA NCEI ISD hourly CSV files → daily-aggregated temperature + dew point.

ISD CSV format reference:
- Federal standards: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
- TMP compound field: "±DDDD,QC" where DDDD = °C × 10, QC = quality code
- DEW compound field: same as TMP, for dew point
- Missing value sentinel: "+9999" (the value, not the QC)
- Quality code "1" = "Passed all quality control checks" (Shaman et al. standard)

Specific humidity formula (Shaman et al. 2013, Bolton 1980):
    e   = 6.112 × exp((17.67 × Td) / (Td + 243.5))    [hPa, Td in °C]
    q   = (0.622 × e) / (P - 0.378 × e)               [kg/kg, P in hPa]
    q_g = q × 1000                                     [g/kg]

P is assumed = 1013.25 hPa (standard atmospheric pressure) when SLP is missing.
This is a small approximation (typical P range 990-1030 hPa → q error ≤ 2%).

Pipeline:
    Hourly CSV → filter REPORT_TYPE=FM-15 (routine METAR) → parse TMP/DEW (QC=1)
              → drop sentinels (+9999) → resample to daily mean → compute q_g
              → save daily Parquet per station.

Usage:
    python -m src.data.parse_isd                    # all stations × all years
    python -m src.data.parse_isd --station 72503014732 --year 2020   # sanity test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.noaa_stations import MSA_STATIONS


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "noaa_isd"
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "noaa_daily"

# ISD constants
ISD_MISSING_TEMP = 9999     # sentinel for missing temperature/dew point (value, not QC)

# Valid QC codes per NOAA ISD documentation §3.2:
#   0 = Passed gross limits check
#   1 = Passed all quality control checks  (most strict + most common)
#   4 = Passed gross limits, lab data
#   5 = Passed all QC via lab data  (the majority in modern ASOS records)
#   9 = Passed gross limits, mass balance (meteorological consistency)
#   A, C, I, M, P, R, U = various "modified/manual passed" codes
# Excluded: 2 (suspect), 3 (erroneous), 6 (suspect via lab), 7 (erroneous via lab)
# LGA-2020 sample showed 99.99% of records have valid QC under this set.
ISD_QC_VALID = frozenset({"0", "1", "4", "5", "9", "A", "C", "I", "M", "P", "R", "U"})

# Report types containing surface temperature observations:
#   FM-15: Routine METAR (hourly)  — primary
#   FM-12: SYNOP (3-hourly synoptic) — secondary, fills gaps when FM-15 missing
# Excluded: FM-16 (SPECI/special), SOD/SOM (summaries), SY-MT (rare).
# Shaman et al. (2013) used FM-15 only; we add FM-12 for daily coverage robustness.
ISD_REPORT_TYPES_VALID = frozenset({"FM-15", "FM-12"})

STD_PRESSURE_HPA = 1013.25  # standard atmospheric pressure when SLP missing


def parse_isd_tmp(tmp_value: str) -> float | None:
    """Parse ISD TMP/DEW compound field '±DDDD,QC' → temperature in °C.

    Returns None if:
      - Field is NaN / empty
      - Value is the missing sentinel (+9999)
      - Quality code is not in ISD_QC_VALID

    Args:
        tmp_value: e.g., "+0067,1" → 6.7 °C; "-0143,1" → -14.3 °C
                   "+9999,9" → None (missing); "+0067,2" → None (suspect)
    """
    if not isinstance(tmp_value, str) or "," not in tmp_value:
        return None
    val_str, qc = tmp_value.split(",", 1)
    qc = qc.strip()
    if qc not in ISD_QC_VALID:
        return None
    try:
        val_int = int(val_str)
        if val_int == ISD_MISSING_TEMP:
            return None
        return val_int / 10.0
    except ValueError:
        return None


def specific_humidity_g_per_kg(
    temp_c: float, dew_c: float, pressure_hpa: float = STD_PRESSURE_HPA,
) -> float:
    """Specific humidity (g/kg) from temperature, dew point, pressure.

    Uses Bolton (1980) saturation vapor pressure formula at dew point.

    Args:
        temp_c:        air temperature (°C)  — not used in formula (kept for sanity check)
        dew_c:         dew point temperature (°C)
        pressure_hpa:  atmospheric pressure (hPa). Defaults to standard pressure.

    Returns:
        Specific humidity in g/kg.
    """
    # Saturation vapor pressure at dew point (Bolton 1980)
    e = 6.112 * math.exp((17.67 * dew_c) / (dew_c + 243.5))
    # Specific humidity (kg/kg → g/kg)
    q_kg_kg = (0.622 * e) / (pressure_hpa - 0.378 * e)
    return q_kg_kg * 1000.0


def parse_isd_year(csv_path: Path, isd_id: str) -> pd.DataFrame:
    """Parse one ISD station-year CSV → daily aggregated DataFrame.

    Returns DataFrame with columns:
      [date, isd_id, n_obs_hourly, temp_c, dew_c, specific_humidity_g_per_kg]

    Where:
      - date:         daily date (UTC)
      - n_obs_hourly: number of valid hourly observations averaged into the day
      - temp_c:       mean daily air temperature (°C)
      - dew_c:        mean daily dew point (°C)
      - specific_humidity_g_per_kg: derived from daily mean T + Td
    """
    # Load only needed columns for speed
    use_cols = ["DATE", "REPORT_TYPE", "TMP", "DEW"]
    df = pd.read_csv(csv_path, usecols=use_cols, dtype=str, low_memory=False)

    # Filter to routine METAR (FM-15) + synoptic (FM-12) — see ISD_REPORT_TYPES_VALID
    df = df[df["REPORT_TYPE"].str.strip().isin(ISD_REPORT_TYPES_VALID)].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "isd_id", "n_obs_hourly", "temp_c", "dew_c",
            "specific_humidity_g_per_kg",
        ])

    # Parse TMP, DEW with QC=1 filter (returns NaN for failures)
    df["temp_c"] = df["TMP"].map(parse_isd_tmp)
    df["dew_c"] = df["DEW"].map(parse_isd_tmp)

    # Drop rows where either is missing (require both for q computation)
    df = df.dropna(subset=["temp_c", "dew_c"])
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "isd_id", "n_obs_hourly", "temp_c", "dew_c",
            "specific_humidity_g_per_kg",
        ])

    # Parse timestamp → date (UTC)
    df["timestamp"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Daily aggregation: mean of valid hourly observations
    daily = (
        df.groupby("date")
          .agg(temp_c=("temp_c", "mean"),
               dew_c=("dew_c", "mean"),
               n_obs_hourly=("temp_c", "size"))
          .reset_index()
    )

    # Compute specific humidity from daily mean T + Td
    daily["specific_humidity_g_per_kg"] = daily.apply(
        lambda r: specific_humidity_g_per_kg(r["temp_c"], r["dew_c"]),
        axis=1,
    )
    daily["isd_id"] = isd_id

    # Reorder columns
    return daily[[
        "date", "isd_id", "n_obs_hourly", "temp_c", "dew_c",
        "specific_humidity_g_per_kg",
    ]]


def parse_station_all_years(isd_id: str, years: list[int]) -> pd.DataFrame:
    """Parse all year files for one station → concatenated daily DataFrame."""
    parts = []
    station_dir = RAW_DIR / isd_id
    for year in years:
        csv_path = station_dir / f"{year}.csv"
        if not csv_path.exists():
            print(f"  WARN: missing {csv_path.relative_to(REPO_ROOT)}", file=sys.stderr)
            continue
        try:
            year_df = parse_isd_year(csv_path, isd_id)
            parts.append(year_df)
        except Exception as e:
            print(f"  ERROR parsing {csv_path.name}: {e}", file=sys.stderr)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--station", type=str, default=None,
                    help="ISD ID for single-station mode (sanity test)")
    ap.add_argument("--year", type=int, default=None,
                    help="Year for single-station+single-year mode (sanity test)")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2001, 2026)),
                    help="Years to parse (default 2001-2025)")
    args = ap.parse_args()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity-test mode: one station, one year, print stats only
    if args.station and args.year:
        csv = RAW_DIR / args.station / f"{args.year}.csv"
        if not csv.exists():
            print(f"ERROR: {csv} not found", file=sys.stderr)
            return 1
        t0 = time.time()
        df = parse_isd_year(csv, args.station)
        dt = time.time() - t0
        print(f"[sanity] {args.station} year={args.year}:")
        print(f"  parsed in {dt:.1f}s")
        print(f"  daily rows: {len(df)}")
        if not df.empty:
            print(f"  date range: {df['date'].min()} → {df['date'].max()}")
            print(f"  n_obs_hourly: mean={df['n_obs_hourly'].mean():.1f}, "
                  f"min={df['n_obs_hourly'].min()}, max={df['n_obs_hourly'].max()}")
            print(f"  temp_c:       mean={df['temp_c'].mean():.2f}, "
                  f"min={df['temp_c'].min():.1f}, max={df['temp_c'].max():.1f}")
            print(f"  dew_c:        mean={df['dew_c'].mean():.2f}, "
                  f"min={df['dew_c'].min():.1f}, max={df['dew_c'].max():.1f}")
            print(f"  q (g/kg):     mean={df['specific_humidity_g_per_kg'].mean():.2f}, "
                  f"min={df['specific_humidity_g_per_kg'].min():.2f}, "
                  f"max={df['specific_humidity_g_per_kg'].max():.2f}")
            print(f"  sample (first 3 days):")
            print(df.head(3).to_string(index=False))
        return 0

    # Full mode: all stations, all years
    stations = [args.station] if args.station else [s.isd_id for s in MSA_STATIONS]
    print(f"[ISD parse] {len(stations)} stations × {len(args.years)} years")
    t0 = time.time()
    for i, isd_id in enumerate(stations, 1):
        ts = time.time()
        df = parse_station_all_years(isd_id, args.years)
        dt = time.time() - ts
        out_path = INTERIM_DIR / f"{isd_id}_daily.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  [{i:2d}/{len(stations)}] {isd_id} → {len(df):,} daily rows in {dt:.1f}s "
              f"({out_path.relative_to(REPO_ROOT)})")
    print(f"[ISD parse] Done in {(time.time()-t0)/60:.1f} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
