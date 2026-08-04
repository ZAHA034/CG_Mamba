"""STEP 2 — transition-point computation under LOCK v2.2 + §14.3 + §14.4.

Outputs: runs/interpretability/transition_points_locked.json (canonical)
         runs/interpretability/transition_smoke.json (smoke 4-item report)

Spec (LOCK pre-registration interpretability_pre_registration.md):
- §3.2 peak: argmax_t y_raw within season window (W40_year .. W20_(year+1)).
- §3.3 turning: Savitzky-Golay smooth (w=5, p=2, mode='interp'), first-derivative
              sign-change weeks; WHOLE-SERIES, no season restriction.
- §3.4 P3: combined coverage |(peak union turning)| / |block| <= 0.35 per
           (region, season). > 0.35 in any block -> STOP per §8 (a).
- §14.3: scipy >=1.11,<1.14, savgol_mode='interp' frozen.
- §14.4: H1-onset NOT EVALUABLE; coverage computed on (peak union turning) only.
- §3.5 P5 byte-identity: re-derivation must match byte-identically.

Usage:
    cd /A.I_DATA/jbnu/JeongHa/CG_Mamba
    source runs/interpretability/.venv_p5/bin/activate
    python scripts/p5_step2_transitions.py
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "runs/interpretability/sigma_components.parquet"
OUT_LOCKED = ROOT / "runs/interpretability/transition_points_locked.json"
OUT_SMOKE = ROOT / "runs/interpretability/transition_smoke.json"

REGIONS = tuple(f"hhs{i}" for i in range(1, 11))
SG_WINDOW = 5
SG_POLY = 2
SG_MODE = "interp"
P3_THRESHOLD = 0.35


def epiweek_to_year_week(ep: int) -> tuple[int, int]:
    return ep // 100, ep % 100


def season_of(ep: int) -> str:
    y, w = epiweek_to_year_week(ep)
    start = y if w >= 40 else y - 1
    return f"{start}-{start+1}"


def in_peak_window(ep: int, season: str) -> bool:
    """Per §3.2: peak window = W40_year .. W20_(year+1) inclusive."""
    start_year = int(season.split("-")[0])
    y, w = epiweek_to_year_week(ep)
    if y == start_year and 40 <= w <= 53:
        return True
    if y == start_year + 1 and 1 <= w <= 20:
        return True
    return False


def compute_turning_whole_series(y: np.ndarray) -> np.ndarray:
    """§3.3 — savgol smooth (w=5, p=2, mode='interp'), first-derivative sign-change.

    NO season window restriction (whole-series; user 2026-06-28 catch).
    Returns boolean array of length len(y); True at first-derivative sign-change weeks.
    """
    smooth = savgol_filter(y, window_length=SG_WINDOW, polyorder=SG_POLY, mode=SG_MODE)
    deriv = savgol_filter(
        y, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=1, mode=SG_MODE
    )
    sign = np.sign(deriv)
    turn = np.zeros(len(y), dtype=bool)
    # Sign change at index t means deriv flipped between t-1 and t.
    for t in range(1, len(y)):
        if sign[t] != 0 and sign[t - 1] != 0 and sign[t] != sign[t - 1]:
            turn[t] = True
    return turn, smooth, deriv


def compute_transitions(df_h1: pd.DataFrame) -> dict:
    """Returns canonical dict.

    Structure:
      {
        "by_region": {
            "hhs1": {
                "epiweeks": [202240, ..., 202532],  # length 149
                "y_raw": [...],
                "peak": {"2022-2023": <epiweek>, "2023-2024": <epiweek>, "2024-2025": <epiweek>},
                "turning_epiweeks": [...],
                ...
            }, ...
        },
        "by_block_coverage": {"hhs1__2022-2023": 0.xxx, ...},
        "p3_max": 0.xxx,
        "p3_pass": bool,
        ...
      }
    """
    by_region = {}
    by_block_coverage = {}
    peak_total = 0
    turning_total = 0
    block_sizes = {}

    for region in REGIONS:
        sub = df_h1[df_h1.region == region].sort_values("week_idx")
        epiweeks = sub.eps_h1.astype(int).to_numpy()
        y_raw = sub.y_raw.astype(float).to_numpy()
        assert len(epiweeks) == 149, f"{region}: expected 149, got {len(epiweeks)}"

        seasons = np.array([season_of(ep) for ep in epiweeks])
        unique_seasons = ["2022-2023", "2023-2024", "2024-2025"]

        # PEAK per season (season-restricted per §3.2)
        peak_by_season = {}
        peak_label = np.zeros(len(epiweeks), dtype=bool)
        for s in unique_seasons:
            mask = np.array([(seasons[i] == s) and in_peak_window(epiweeks[i], s)
                             for i in range(len(epiweeks))])
            if not mask.any():
                peak_by_season[s] = None
                continue
            idxs = np.where(mask)[0]
            vals = y_raw[idxs]
            peak_idx = idxs[int(np.argmax(vals))]
            peak_by_season[s] = int(epiweeks[peak_idx])
            peak_label[peak_idx] = True
            peak_total += 1

        # TURNING whole-series (no season window) per §3.3 + user 2026-06-28 catch
        turn_label, smooth, deriv = compute_turning_whole_series(y_raw)
        turn_epiweeks = epiweeks[turn_label].tolist()
        turning_total += int(turn_label.sum())

        union = peak_label | turn_label

        # P3 coverage per (region, season)
        per_season_coverage = {}
        for s in unique_seasons:
            block_mask = seasons == s
            block_size = int(block_mask.sum())
            block_sizes.setdefault(s, block_size)
            union_in_block = int((union & block_mask).sum())
            cov = union_in_block / block_size if block_size > 0 else 0.0
            per_season_coverage[s] = cov
            by_block_coverage[f"{region}__{s}"] = cov

        # Also count turning by season (output convenience only — calc was whole-series)
        turn_count_by_season = {}
        for s in unique_seasons:
            block_mask = seasons == s
            turn_count_by_season[s] = int((turn_label & block_mask).sum())

        by_region[region] = {
            "epiweeks": epiweeks.tolist(),
            "y_raw": y_raw.tolist(),
            "season_of_each_week": seasons.tolist(),
            "peak_by_season": peak_by_season,
            "turning_epiweeks": turn_epiweeks,
            "turning_count_whole_series": int(turn_label.sum()),
            "turning_count_by_season_output_only": turn_count_by_season,
            "coverage_per_season_peak_union_turning": per_season_coverage,
            "peak_label": peak_label.tolist(),
            "turning_label": turn_label.tolist(),
            "onset_label": [False] * len(epiweeks),  # NOT EVALUABLE per §14.4
            "smooth_y_raw": smooth.tolist(),
            "first_derivative": deriv.tolist(),
        }

    p3_values = list(by_block_coverage.values())
    p3_max = max(p3_values) if p3_values else 0.0

    return {
        "lock_version": "v2.2 + §14.3 + §14.4",
        "spec_constants": {
            "savgol_window": SG_WINDOW,
            "savgol_polyorder": SG_POLY,
            "savgol_mode": SG_MODE,
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "p3_threshold": P3_THRESHOLD,
            "peak_window_inclusive_weeks": "W40 .. W20+1",
            "turning_window": "WHOLE SERIES (no season restriction; §3.3 + user 2026-06-28 catch)",
            "onset_status": "NOT EVALUABLE per §14.4",
        },
        "block_sizes_per_season": block_sizes,
        "n_regions": len(REGIONS),
        "n_weeks_per_region": 149,
        "totals": {
            "peak_total_30_cells": peak_total,
            "turning_total_whole_series": turning_total,
            "onset_total": 0,
        },
        "by_region": by_region,
        "by_block_coverage": by_block_coverage,
        "p3_max_coverage": p3_max,
        "p3_pass": p3_max <= P3_THRESHOLD,
    }


def canonical_json_bytes(obj: dict) -> bytes:
    """Sort keys + compact separators -> stable byte representation."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def smoke_robustness_compare() -> dict:
    """Run turning calc once with current (1.13.x) scipy and report what we'd see
    if computed with the system 1.16.1 — for robustness only, NOT canonical.

    We don't actually have 1.16 in this venv. Instead, we report scipy version
    used and hash the savgol_filter input/output for future comparison.
    """
    # Sample series for deterministic comparison probe
    rng = np.random.default_rng(seed=0)
    sample = rng.normal(size=149)
    smooth = savgol_filter(sample, window_length=SG_WINDOW, polyorder=SG_POLY, mode=SG_MODE)
    deriv = savgol_filter(sample, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=1, mode=SG_MODE)
    return {
        "scipy_version": scipy.__version__,
        "savgol_sample_sha256": sha256(smooth.tobytes() + deriv.tobytes()),
        "note": "comparison vs scipy 1.16.1 to be reported by smoke driver (separate venv invocation)",
    }


def main():
    print(f"=== STEP 2 transition compute (scipy {scipy.__version__}) ===")
    print(f"Reading parquet: {PARQUET}")
    df = pd.read_parquet(PARQUET)
    h1 = df[df.h == 1].drop_duplicates(["region", "week_idx"])[
        ["region", "week_idx", "eps_h1", "y_raw"]
    ]
    print(f"h=1 cells: {len(h1)} (expected 1490 = 10 × 149)")
    assert len(h1) == 1490, f"unexpected cell count: {len(h1)}"

    # First pass
    print("Computing transitions (pass 1)...")
    result_1 = compute_transitions(h1)

    # Second pass (byte-identity check, P5)
    print("Computing transitions (pass 2, P5 byte-identity)...")
    result_2 = compute_transitions(h1)

    b1 = canonical_json_bytes(result_1)
    b2 = canonical_json_bytes(result_2)
    sha_1 = sha256(b1)
    sha_2 = sha256(b2)
    p5_pass = sha_1 == sha_2

    # Save canonical
    OUT_LOCKED.write_bytes(b1)

    # Smoke report
    smoke = {
        "report": "STEP 2 smoke (4 mandatory items + robustness)",
        "item_1_turning_whole_series_confirmation": {
            "spec": "§3.3 whole-series; user 2026-06-28 catch — no season restriction",
            "code_path": "compute_turning_whole_series() — savgol on full 149-week series, no season-mask before sign-change detection",
            "evidence": "per-season turning counts are reported separately (turning_count_by_season_output_only) but the *computation* used the whole-series; sum of per-season counts == whole-series total per region",
            "per_region_check": {
                r: {
                    "whole_series_total": result_1["by_region"][r]["turning_count_whole_series"],
                    "by_season_sum": sum(result_1["by_region"][r]["turning_count_by_season_output_only"].values()),
                    "match": (
                        result_1["by_region"][r]["turning_count_whole_series"]
                        == sum(result_1["by_region"][r]["turning_count_by_season_output_only"].values())
                    ),
                }
                for r in REGIONS
            },
        },
        "item_2_p3_coverage": {
            "threshold": P3_THRESHOLD,
            "p3_max": result_1["p3_max_coverage"],
            "p3_pass": result_1["p3_pass"],
            "per_block_coverage": result_1["by_block_coverage"],
        },
        "item_3_counts_per_cell": {
            "peak_total_30_cells_expected": 30,
            "peak_total_observed": result_1["totals"]["peak_total_30_cells"],
            "turning_total_whole_series": result_1["totals"]["turning_total_whole_series"],
            "block_sizes_per_season": result_1["block_sizes_per_season"],
            "turning_per_region": {
                r: result_1["by_region"][r]["turning_count_whole_series"] for r in REGIONS
            },
        },
        "item_4_p5_byte_identity": {
            "pass1_sha256": sha_1,
            "pass2_sha256": sha_2,
            "p5_pass": p5_pass,
            "canonical_path": str(OUT_LOCKED),
            "scipy_version": scipy.__version__,
            "savgol_mode": SG_MODE,
            "savgol_window": SG_WINDOW,
            "savgol_polyorder": SG_POLY,
        },
        "robustness_probe_versions_used": smoke_robustness_compare(),
    }
    OUT_SMOKE.write_bytes(canonical_json_bytes(smoke))

    # Stdout summary
    print()
    print("=== SMOKE SUMMARY ===")
    print(f"Item 1 — turning whole-series: per-region {sum(1 for r in REGIONS if smoke['item_1_turning_whole_series_confirmation']['per_region_check'][r]['match'])}/10 sum-match")
    print(f"Item 2 — P3 max coverage: {result_1['p3_max_coverage']:.4f} (threshold {P3_THRESHOLD}) — {'PASS' if result_1['p3_pass'] else 'FAIL'}")
    print(f"Item 3 — peak total: {result_1['totals']['peak_total_30_cells']}/30 ; turning total: {result_1['totals']['turning_total_whole_series']}")
    print(f"Item 4 — P5 byte-identity: {'PASS' if p5_pass else 'FAIL'}")
    print(f"        sha256 = {sha_1}")
    print(f"Canonical saved: {OUT_LOCKED}")
    print(f"Smoke saved: {OUT_SMOKE}")
    print()
    print("=== P3 coverage per (region, season) ===")
    seasons = ["2022-2023", "2023-2024", "2024-2025"]
    print(f"{'region':<8} {seasons[0]:<12} {seasons[1]:<12} {seasons[2]:<12}")
    for r in REGIONS:
        row = result_1["by_region"][r]["coverage_per_season_peak_union_turning"]
        print(f"{r:<8} {row[seasons[0]]:<12.4f} {row[seasons[1]]:<12.4f} {row[seasons[2]]:<12.4f}")
    print()
    print("=== Turning count per region (whole-series) ===")
    for r in REGIONS:
        print(f"  {r}: {result_1['by_region'][r]['turning_count_whole_series']}")
    print()
    return 0 if (result_1["p3_pass"] and p5_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
