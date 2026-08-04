"""Download NOAA NCEI ISD hourly CSV files for 10 MSA stations × 25 years.

Default time range: 2001 - 2025 (covers PLAN v2.0.5 Train+Val+Test windows).
Concurrent download with 5 workers + per-file retry.

Layout:
    data/raw/noaa_isd/{isd_id}/{year}.csv          # one file per station-year
    data/raw/noaa_isd/MANIFEST.json                # download status + checksums

Usage:
    python -m src.data.download_noaa_isd                   # default 2001-2025
    python -m src.data.download_noaa_isd --years 2024 2025 # incremental update
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

from src.data.noaa_stations import MSA_STATIONS, get_isd_url


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "noaa_isd"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

DEFAULT_YEARS = list(range(2001, 2026))     # 2001-2025 inclusive
TIMEOUT_SEC = 90
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(isd_id: str, year: int, out_dir: Path) -> dict:
    """Download one station-year file with retries. Returns status dict."""
    url = get_isd_url(isd_id, year)
    out_path = out_dir / f"{year}.csv"

    # Skip if exists + non-empty
    if out_path.exists() and out_path.stat().st_size > 0:
        return {
            "isd_id": isd_id, "year": year, "status": "skipped",
            "url": url, "path": str(out_path.relative_to(REPO_ROOT)),
            "size": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
        }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            r = requests.get(url, timeout=TIMEOUT_SEC)
            dt = time.time() - t0
            if r.status_code == 404:
                return {
                    "isd_id": isd_id, "year": year, "status": "not_found",
                    "url": url, "http_status": 404, "attempt": attempt,
                    "note": f"Station {isd_id} has no data for year {year}",
                }
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
            return {
                "isd_id": isd_id, "year": year, "status": "ok",
                "url": url, "path": str(out_path.relative_to(REPO_ROOT)),
                "size": len(r.content),
                "sha256": hashlib.sha256(r.content).hexdigest(),
                "elapsed_sec": round(dt, 2),
                "attempt": attempt,
            }
        except (requests.RequestException, IOError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
                continue
            return {
                "isd_id": isd_id, "year": year, "status": "failed",
                "url": url, "error": str(e), "attempts": MAX_RETRIES,
            }
    # unreachable
    return {"isd_id": isd_id, "year": year, "status": "failed", "error": "unknown"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                    help="Years to download (default: 2001-2025)")
    ap.add_argument("--workers", type=int, default=5,
                    help="Concurrent download workers (default 5; NCEI server-friendly)")
    ap.add_argument("--out", type=str, default=str(RAW_DIR),
                    help=f"Output directory (default: {RAW_DIR})")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    n_stations = len(MSA_STATIONS)
    n_years = len(args.years)
    total = n_stations * n_years
    print(f"[NOAA ISD] {n_stations} stations × {n_years} years = {total} files")
    print(f"[NOAA ISD] workers={args.workers}, out={out_root.relative_to(REPO_ROOT)}/")
    print(f"[NOAA ISD] years: {min(args.years)} - {max(args.years)}")
    print()

    # Build task list
    tasks = []
    for s in MSA_STATIONS:
        station_dir = out_root / s.isd_id
        for y in args.years:
            tasks.append((s.isd_id, y, station_dir))

    # Concurrent download
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, isd, y, d): (isd, y) for (isd, y, d) in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            results.append(r)
            elapsed = time.time() - t0
            if r["status"] == "ok":
                mb = r["size"] / 1024 / 1024
                print(f"  [{done:3d}/{total}] {r['isd_id']} {r['year']} → "
                      f"{mb:.1f} MB in {r['elapsed_sec']}s  (total {elapsed/60:.1f}min)")
            elif r["status"] == "skipped":
                print(f"  [{done:3d}/{total}] {r['isd_id']} {r['year']} → skipped (exists)")
            else:
                print(f"  [{done:3d}/{total}] {r['isd_id']} {r['year']} → {r['status']}: "
                      f"{r.get('error', r.get('note', ''))}")

    total_dt = time.time() - t0
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_404 = sum(1 for r in results if r["status"] == "not_found")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    total_bytes = sum(r.get("size", 0) for r in results if r["status"] in ("ok", "skipped"))

    # Write manifest
    manifest = {
        "_schema": "noaa_isd_download_manifest_v1",
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "source": "NOAA NCEI Integrated Surface Database (ISD), Global Hourly",
        "base_url": "https://www.ncei.noaa.gov/data/global-hourly/access/{YEAR}/{ISD_ID}.csv",
        "license": "Public domain (US federal government work, 17 USC §105)",
        "n_stations": n_stations,
        "n_years_requested": n_years,
        "n_files_ok": n_ok,
        "n_files_skipped": n_skip,
        "n_files_not_found": n_404,
        "n_files_failed": n_fail,
        "total_bytes": total_bytes,
        "elapsed_sec": round(total_dt, 1),
        "files": sorted(results, key=lambda r: (r["isd_id"], r["year"])),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"[NOAA ISD] Done in {total_dt/60:.1f} min")
    print(f"  ok:        {n_ok}")
    print(f"  skipped:   {n_skip}")
    print(f"  not_found: {n_404}")
    print(f"  failed:    {n_fail}")
    print(f"  total:     {total_bytes/1024/1024:.1f} MB")
    print(f"  manifest:  {MANIFEST_PATH.relative_to(REPO_ROOT)}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
