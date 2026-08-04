#!/usr/bin/env python3
"""
p3_table_iv_verification.py
===========================

Verify Table IV (label: tab:regional_h1) numbers in
latex_submission/tex/results_main.tex against disk artifacts under runs/.

Table IV reports cross-region WIS and Cov95, h=1..4 average across 10 HHS regions,
with cross-region std. Rows: SARIMAX, CG-Mamba (APMD), LSTM (MCD d=0.3),
PatchTST (MCD d=0.1), Vanilla Mamba (MCD d=0.1), DLinear (5-seed ensemble),
EpiDeep (MCD d=0.1).

Aggregation policy (per LaTeX caption + iv_x_region prose):
  - DL models: per (baseline, region) 5-seed mean of strict-test WIS/Cov95
    at each h, then average h=1..4, then mean/std across 10 regions.
  - SARIMAX/DLinear: deterministic per region (no seed dimension); same
    h=1..4 average then mean/std across regions.

Disk sources used (most specific available per row):
  - SARIMAX:        runs/phase_3_sarima_wis_region.json (test_strict block)
  - CG-Mamba APMD:  runs/phase_3_cgm_method_f_region.csv
  - LSTM, PatchTST, Vanilla Mamba: runs/phase_3_region_wis.csv
  - EpiDeep:        runs/phase_3_region_wis_extras.csv
  - DLinear (5-seed ensemble Gaussian): runs/phase_3_dlinear_ensemble_region.csv

All numbers use the test_strict (n_strict=149/region) view, matching the
caption.

Tolerance: |delta| < 0.005.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, pstdev, stdev

ROOT = Path("/A.I_DATA/jbnu/JeongHa/CG_Mamba")
RUNS = ROOT / "runs"
TEX = ROOT / "latex_submission" / "tex" / "results_main.tex"
TOL = 0.005

# ---------- LaTeX values (from results_main.tex Table IV) ----------
LATEX = {
    "SARIMAX":       {"wis": 0.301, "wis_std": 0.060, "cov": 0.916, "cov_std": 0.031},
    "CG-Mamba":      {"wis": 0.368, "wis_std": 0.076, "cov": 0.930, "cov_std": 0.020},
    "LSTM":          {"wis": 0.416, "wis_std": 0.142, "cov": 0.513, "cov_std": 0.108},
    "PatchTST":      {"wis": 0.423, "wis_std": 0.095, "cov": 0.695, "cov_std": 0.044},
    "Vanilla Mamba": {"wis": 0.463, "wis_std": 0.163, "cov": 0.571, "cov_std": 0.161},
    "DLinear":       {"wis": 0.509, "wis_std": 0.116, "cov": 0.286, "cov_std": 0.050},
    "EpiDeep":       {"wis": 0.515, "wis_std": 0.227, "cov": 0.382, "cov_std": 0.138},
}

HORIZONS = (1, 2, 3, 4)


# ---------- Helpers ----------

def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def avg_h(row: dict, prefix: str) -> float:
    """Average h=1..4 of strict columns (tS_<metric>_h{h}). prefix is wis|cov95."""
    vals = [float(row[f"tS_{prefix}_h{h}"]) for h in HORIZONS]
    return sum(vals) / len(vals)


def cross_region_stats(per_region_vals: dict[str, float]) -> tuple[float, float]:
    """Return (mean, std) across regions. Use sample (ddof=1) std."""
    vs = list(per_region_vals.values())
    m = mean(vs)
    s = stdev(vs) if len(vs) > 1 else 0.0
    return m, s


# ---------- Per-baseline aggregators ----------

def agg_seed_csv(path: Path, baseline_name: str, expected_dropout: str | None = None) -> dict[str, dict[str, float]]:
    """
    Aggregate a CSV that has per-(region, seed) rows.
    Returns {region: {'wis': h14avg, 'cov': h14avg}} after 5-seed mean per region.
    """
    rows = [r for r in read_csv(path) if r["baseline"] == baseline_name]
    if expected_dropout is not None:
        rows = [r for r in rows if (r.get("dropout") or "") == expected_dropout]
    # group by region
    by_region: dict[str, list[dict]] = {}
    for r in rows:
        by_region.setdefault(r["region"], []).append(r)
    out: dict[str, dict[str, float]] = {}
    for region, rs in by_region.items():
        wis_h14 = [avg_h(r, "wis") for r in rs]
        cov_h14 = [avg_h(r, "cov95") for r in rs]
        out[region] = {
            "wis": sum(wis_h14) / len(wis_h14),
            "cov": sum(cov_h14) / len(cov_h14),
            "n_seeds": len(rs),
        }
    return out


def agg_dlinear(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out[r["region"]] = {
            "wis": avg_h(r, "wis"),
            "cov": avg_h(r, "cov95"),
            "n_seeds": 1,  # 5-seed ensemble Gaussian already aggregated into single row
        }
    return out


def agg_sarima(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for region, payload in data.items():
        ts = payload["test_strict"]
        wis = sum(ts[f"wis_h{h}"] for h in HORIZONS) / len(HORIZONS)
        cov = sum(ts[f"cov95_h{h}"] for h in HORIZONS) / len(HORIZONS)
        out[region] = {"wis": wis, "cov": cov, "n_seeds": 1}
    return out


# ---------- Build disk side ----------

def build_disk_table() -> dict[str, dict[str, float]]:
    disk: dict[str, dict[str, float]] = {}

    sources = {
        "SARIMAX":       ("phase_3_sarima_wis_region.json (test_strict)", agg_sarima(RUNS / "phase_3_sarima_wis_region.json")),
        "CG-Mamba":      ("phase_3_cgm_method_f_region.csv (5-seed mean)", agg_seed_csv(RUNS / "phase_3_cgm_method_f_region.csv", "cg_mamba_method_F")),
        "LSTM":          ("phase_3_region_wis.csv lstm d=0.3 (5-seed mean)", agg_seed_csv(RUNS / "phase_3_region_wis.csv", "lstm", "0.3")),
        "PatchTST":      ("phase_3_region_wis.csv patchtst d=0.1 (5-seed mean)", agg_seed_csv(RUNS / "phase_3_region_wis.csv", "patchtst", "0.1")),
        "Vanilla Mamba": ("phase_3_region_wis.csv vanilla_mamba d=0.1 (5-seed mean)", agg_seed_csv(RUNS / "phase_3_region_wis.csv", "vanilla_mamba", "0.1")),
        "DLinear":       ("phase_3_dlinear_ensemble_region.csv (deterministic ensemble Gaussian)", agg_dlinear(RUNS / "phase_3_dlinear_ensemble_region.csv")),
        "EpiDeep":       ("phase_3_region_wis_extras.csv epideep d=0.1 (5-seed mean)", agg_seed_csv(RUNS / "phase_3_region_wis_extras.csv", "epideep", "0.1")),
    }

    for model, (source_desc, per_region) in sources.items():
        wis_map = {r: v["wis"] for r, v in per_region.items()}
        cov_map = {r: v["cov"] for r, v in per_region.items()}
        wis_m, wis_s = cross_region_stats(wis_map)
        cov_m, cov_s = cross_region_stats(cov_map)
        disk[model] = {
            "wis": wis_m,
            "wis_std": wis_s,
            "cov": cov_m,
            "cov_std": cov_s,
            "n_regions": len(per_region),
            "n_seeds": max((v.get("n_seeds", 1) for v in per_region.values()), default=0),
            "source": source_desc,
        }
    return disk


# ---------- Verification ----------

def verdict(delta: float) -> str:
    return "PASS" if abs(delta) < TOL else "FAIL"


def main() -> int:
    print("=" * 78)
    print("Table IV (tab:regional_h1) verification against disk artifacts")
    print(f"Tolerance: |delta| < {TOL}")
    print("=" * 78)

    disk = build_disk_table()

    # Print sources
    print("\nDisk artifact sources used:")
    for model, d in disk.items():
        print(f"  {model:14s}  <- {d['source']}  (regions={d['n_regions']}, seeds_per_region<= {d['n_seeds']})")

    # Per-cell comparison
    print("\nPer-cell comparison (h=1..4 strict average; mean +/- std across 10 regions):")
    header = f"{'Model':14s} {'Metric':9s} {'LaTeX':>10s} {'Disk':>10s} {'Delta':>10s} {'Verdict':>8s}"
    print(header)
    print("-" * len(header))

    pf_rows: list[dict] = []
    all_pass = True
    failed = 0

    for model in ["SARIMAX", "CG-Mamba", "LSTM", "PatchTST", "Vanilla Mamba", "DLinear", "EpiDeep"]:
        lat = LATEX[model]
        d = disk[model]

        cells = [
            ("WIS_mean",  lat["wis"],     d["wis"]),
            ("WIS_std",   lat["wis_std"], d["wis_std"]),
            ("Cov95_mean",lat["cov"],     d["cov"]),
            ("Cov95_std", lat["cov_std"], d["cov_std"]),
        ]
        for metric, lv, dv in cells:
            delta = dv - lv
            v = verdict(delta)
            if v == "FAIL":
                all_pass = False
                failed += 1
            print(f"{model:14s} {metric:9s} {lv:>10.4f} {dv:>10.4f} {delta:>+10.4f} {v:>8s}")
            pf_rows.append({
                "model": model,
                "metric": metric,
                "latex_value": round(lv, 4),
                "disk_value": round(dv, 4),
                "delta": round(delta, 4),
                "verdict": v,
            })

    print("-" * len(header))
    n_cells = len(pf_rows)
    print(f"\nTotal cells: {n_cells}  passed: {n_cells - failed}  failed: {failed}")

    # Coverage / completeness notes
    print("\nCompleteness check:")
    for model, d in disk.items():
        if d["n_regions"] != 10:
            print(f"  WARN: {model} has {d['n_regions']} regions on disk (expected 10)")
        else:
            print(f"  OK:   {model} 10 regions on disk")

    # Baselines absent from Table IV
    print("\nBaselines absent from Table IV (deliberate scoping check):")
    absent = ["iTransformer", "TimesNet", "N-BEATS", "Persistence"]
    wis_phase_b = RUNS / "wis_phase_b"
    if wis_phase_b.exists():
        present_dirs = {p.name for p in wis_phase_b.iterdir() if p.is_dir()}
    else:
        present_dirs = set()
    for b in absent:
        key = b.lower().replace("-", "")
        on_disk = key in present_dirs
        print(f"  {b:14s} disk(wis_phase_b/) = {'present' if on_disk else 'absent'} ; Table IV omission consistent with Section IV-x_region 'five DL baselines' scope")

    print("\nOverall verdict:", "PASS" if all_pass else f"FAIL ({failed}/{n_cells} cells out of tolerance)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
