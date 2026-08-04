"""Download CDC FluView national weekly ILI data via Delphi Epidata API.

Source:
- CMU Delphi Epidata API: https://api.delphi.cmu.edu/epidata/fluview/
- Underlying data: CDC FluView (CDC Influenza Division). Public domain.
- Anonymous access (no API key required for fluview endpoint).

Output:
- data/raw/cdc_ilinet/national_weekly.csv
- Columns: date (MMWR Sunday), year, week, region, ili_weighted_pct,
           total_ili_count, num_providers, num_patients

Time range: 2001-W40 ~ 2025-W39 (24 seasons, covers PLAN v2.0.5 Train+Val+Test).

Usage:
    python -m src.data.download_cdc_fluview
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from epiweeks import Week


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "raw" / "cdc_ilinet"

API_BASE = "https://api.delphi.cmu.edu/epidata/fluview/"
DEFAULT_EPIWEEKS = "200140-202539"   # MMWR 2001-W40 to 2025-W39


def epiweek_to_sunday(epiweek: int) -> str:
    """200140 → '2001-09-30' (Sunday of MMWR 2001-W40)."""
    year, week = divmod(epiweek, 100)
    return Week(year, week).startdate().isoformat()


def fetch_fluview(epiweeks: str, region: str = "nat") -> dict:
    """Single Delphi Epidata API call.

    Args:
        epiweeks: e.g., '200140-202539' or '200140,200141,200142'
        region:   'nat' (national) | 'hhs1' | 'cen1' | etc.

    Returns:
        Parsed JSON dict: {'result': int, 'epidata': [...], 'message': str}
    """
    params = {"regions": region, "epiweeks": epiweeks}
    r = requests.get(API_BASE, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epiweeks", type=str, default=DEFAULT_EPIWEEKS,
                    help=f"MMWR range (default: {DEFAULT_EPIWEEKS})")
    ap.add_argument("--region", type=str, default="nat",
                    help="Region code (default 'nat' = National)")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "national_weekly.csv"),
                    help="Output CSV path")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[CDC FluView] Endpoint: {API_BASE}")
    print(f"[CDC FluView] Range: {args.epiweeks}  Region: {args.region}")

    t0 = time.time()
    resp = fetch_fluview(args.epiweeks, args.region)
    dt = time.time() - t0

    if resp.get("result") != 1:
        print(f"[CDC FluView] API error: {resp.get('message', 'unknown')}",
              file=sys.stderr)
        return 1

    records = resp.get("epidata", [])
    print(f"[CDC FluView] {len(records):,} records returned in {dt:.1f}s")

    # Build DataFrame
    df = pd.DataFrame(records)

    # Map Delphi field names to CDC FluView convention used in PLAN v2.0.5
    out = pd.DataFrame({
        "date": df["epiweek"].apply(epiweek_to_sunday),
        "year": df["epiweek"] // 100,
        "week": df["epiweek"] % 100,
        "region": df["region"],
        "ili_weighted_pct": df["wili"],
        "ili_unweighted_pct": df["ili"],
        "total_ili_count": df["num_ili"],
        "num_providers": df["num_providers"],
        "num_patients": df["num_patients"],
        # Delphi extras (useful for downstream analysis / revisions)
        "release_date": df["release_date"],
        "issue": df["issue"],
        "lag": df["lag"],
    })

    # Sort by date ascending, deduplicate (Delphi sometimes has overlapping issues)
    out = out.sort_values(["date", "issue"]).drop_duplicates("date", keep="last")
    out = out.reset_index(drop=True)

    out.to_csv(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    size = out_path.stat().st_size

    # Manifest
    manifest = {
        "_schema": "cdc_fluview_download_manifest_v1",
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "source": "CDC FluView National via CMU Delphi Epidata API",
        "api_endpoint": API_BASE,
        "license": "Public domain (US federal government work, 17 USC §105)",
        "citation": (
            "Centers for Disease Control and Prevention. FluView Interactive Dashboard. "
            "https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html. "
            "Accessed via CMU Delphi Epidata API "
            "(https://cmu-delphi.github.io/delphi-epidata/api/fluview.html)."
        ),
        "query": {"epiweeks": args.epiweeks, "region": args.region},
        "n_records": len(out),
        "date_first": str(out["date"].min()),
        "date_last": str(out["date"].max()),
        "epiweek_first": int(out["year"].iloc[0] * 100 + out["week"].iloc[0]),
        "epiweek_last": int(out["year"].iloc[-1] * 100 + out["week"].iloc[-1]),
        "output": {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "size_bytes": size,
            "sha256": sha,
        },
        "columns": list(out.columns),
        "known_caveats": [
            "CDC revises recent-week values retroactively. Latest issue used (lag=0 = most recent revision).",
            "MMWR week format: epiweek = YYYY*100 + WW; week 53 occurs in some years.",
            "Pre-2010 wili reflects original CDC reporting prior to denominator revisions."
        ],
    }
    manifest_path = OUT_DIR / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[CDC FluView] Saved: {out_path.relative_to(REPO_ROOT)}")
    print(f"  size:    {size:,} bytes")
    print(f"  records: {len(out):,}")
    print(f"  range:   {out['date'].min()} → {out['date'].max()}")
    print(f"  sha256:  {sha[:16]}...")
    print(f"  manifest: {manifest_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
