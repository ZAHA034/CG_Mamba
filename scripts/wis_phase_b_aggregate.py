"""WIS Phase B aggregator — pull all 8 no-retrain baselines' wis_results.json
into a single unified table for Phase D analysis.

Output:
  runs/wis_phase_b/summary_table.csv     ← CSV view
  runs/wis_phase_b/summary.json          ← structured nested view

Columns (per baseline × per split):
  baseline | cfg_name | split | n | wis_h1 | wis_h2 | wis_h3 | wis_h4 |
  wis_avg | dispersion_avg | under_avg | over_avg | coverage_50 | coverage_95 |
  wis_avg_std (if multi-seed) | n_seeds (if applicable)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]

PHASE_B_ROOT = _ROOT / "runs" / "wis_phase_b"

# Order baselines for display (Tier 1/2 + Tier 3)
BASELINE_ORDER = [
    "sarima",        # Tier 1 parametric
    "persistence",   # Tier 1 residual
    "dlinear",       # Tier 2 ensemble
    "nbeats",        # Tier 2 ensemble
    "patchtst",      # Tier 3 MC dropout
    "itransformer",  # Tier 3 MC dropout
    "timesnet",      # Tier 3 MC dropout
    "epideep",       # Tier 3 MC dropout
]
SPLITS = ("val", "test_full", "test_strict")


def _flatten_single_seed(blob: dict, split: str) -> dict | None:
    """For single-seed baselines (SARIMA, Persistence, DLinear, N-BEATS),
    return flattened split-level metrics or None if missing."""
    s = blob.get("splits", {}).get(split)
    if s is None:
        return None
    return {
        "n": s["n"],
        "wis_h1": s["wis_per_horizon"][0],
        "wis_h2": s["wis_per_horizon"][1],
        "wis_h3": s["wis_per_horizon"][2],
        "wis_h4": s["wis_per_horizon"][3],
        "wis_avg": s["wis_avg"],
        "dispersion_avg": s["wis_decomposed"]["dispersion_avg"],
        "under_avg": s["wis_decomposed"]["under_avg"],
        "over_avg": s["wis_decomposed"]["over_avg"],
        "coverage_50": s["coverage_50"],
        "coverage_95": s["coverage_95"],
        "wis_avg_std": "",
        "n_seeds": blob.get("n_seeds", 1),
    }


def _flatten_multi_seed(blob: dict, split: str) -> dict | None:
    """For Tier-3 MC Dropout baselines that store per-seed + aggregated."""
    agg = blob.get("aggregated", {}).get(split)
    if agg is None:
        return None
    return {
        "n": agg["n"],
        "wis_h1": agg["wis_per_horizon_mean"][0],
        "wis_h2": agg["wis_per_horizon_mean"][1],
        "wis_h3": agg["wis_per_horizon_mean"][2],
        "wis_h4": agg["wis_per_horizon_mean"][3],
        "wis_avg": agg["wis_avg_mean"],
        "dispersion_avg": "",  # decomposition is per-seed only in current spec
        "under_avg": "",
        "over_avg": "",
        "coverage_50": agg["coverage_50_mean"],
        "coverage_95": agg["coverage_95_mean"],
        "wis_avg_std": agg["wis_avg_std"],
        "n_seeds": len(blob.get("per_seed", {})),
    }


def main():
    if not PHASE_B_ROOT.exists():
        raise SystemExit(f"Phase B root missing: {PHASE_B_ROOT}")

    rows = []
    structured = {}

    for baseline in BASELINE_ORDER:
        json_path = PHASE_B_ROOT / baseline / "wis_results.json"
        if not json_path.exists():
            print(f"  [{baseline:14s}] SKIP — wis_results.json not yet produced")
            continue
        blob = json.loads(json_path.read_text())
        structured[baseline] = blob

        # Detect single-seed vs multi-seed schema
        is_multi = "aggregated" in blob and "per_seed" in blob
        flatten = _flatten_multi_seed if is_multi else _flatten_single_seed

        for split in SPLITS:
            flat = flatten(blob, split)
            if flat is None:
                continue
            row = {
                "baseline": baseline,
                "cfg_name": blob.get("cfg_name", ""),
                "split": split,
                **flat,
            }
            rows.append(row)

    if not rows:
        print("No baseline wis_results.json found yet — run Phase B scripts first.")
        return 1

    # CSV output
    csv_path = PHASE_B_ROOT / "summary_table.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path.relative_to(_ROOT)} ({len(rows)} rows)")

    # JSON structured
    json_path = PHASE_B_ROOT / "summary.json"
    json_path.write_text(json.dumps(structured, indent=2))
    print(f"Saved: {json_path.relative_to(_ROOT)}")

    # Pretty-print main table (test_strict only — paper main column)
    print(f"\n{'='*100}")
    print(f"WIS Phase B — Main test_strict comparison")
    print(f"{'='*100}")
    print(f"{'Baseline':<15s} {'cfg':<28s} {'WIS_avg':>10s} {'±std':>8s} "
          f"{'cov50':>8s} {'cov95':>8s} {'n':>5s} {'n_seeds':>8s}")
    print("-" * 100)
    for r in rows:
        if r["split"] != "test_strict":
            continue
        std_str = f"±{r['wis_avg_std']:.4f}" if r["wis_avg_std"] not in ("", None) else "       "
        cfg_disp = r["cfg_name"][:28]
        print(f"{r['baseline']:<15s} {cfg_disp:<28s} {r['wis_avg']:>10.4f} {std_str:>8s} "
              f"{r['coverage_50']:>8.3f} {r['coverage_95']:>8.3f} {r['n']:>5d} {r['n_seeds']:>8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
