"""Validation tests for the merged weekly dataset + split + scaler.

Tests:
  T1. MMWR Sunday alignment between CDC and NOAA in merged dataset
  T2. No duplicate epiweeks, monotonic increasing
  T3. No NaN in target (ili_weighted_pct) and env predictors
  T4. Plausible ranges (ILI %, temperature, specific humidity)
  T5. Split boundaries are non-overlapping, inclusive, complete coverage
  T6. Row counts sum to total (train + val + covid_excluded + test = N)
  T7. test_post_covid is strict subset of test
  T8. Scaler fit on train ONLY — no val/test leakage
       - normalization_params.json fit_n_rows == train row count
       - mean/std recomputed from train slice equals saved params
  T9. SHA256 of output files match what's in manifests (env weekly, merged)
  T10. Enumerate epiweek gaps per split (CDC summer-2002 reporting anomaly in train)
       - Expect exactly 1 train gap (200220 -> 200240). Other splits should be contiguous.

Usage:
    python -m src.data.validate_dataset
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

MERGED_CSV = PROCESSED / "ili_env_weekly.csv"
MERGED_MANIFEST = PROCESSED / "ili_env_weekly_MANIFEST.json"
SPLIT_CSV = PROCESSED / "ili_env_weekly_split.csv"
BOUNDARIES_JSON = PROCESSED / "split_boundaries.json"
NORM_JSON = PROCESSED / "normalization_params.json"

CDC_CSV = REPO_ROOT / "data" / "raw" / "cdc_ilinet" / "national_weekly.csv"
ENV_CSV = PROCESSED / "env_national_weekly.csv"
ENV_MANIFEST = PROCESSED / "env_weekly_MANIFEST.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"  ❌ FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def main() -> int:
    failed = []

    print("="*72)
    print("CG-Mamba M1.1 dataset validation")
    print("="*72)

    # Load everything once
    merged = pd.read_csv(MERGED_CSV)
    split = pd.read_csv(SPLIT_CSV)
    cdc = pd.read_csv(CDC_CSV)
    env = pd.read_csv(ENV_CSV)
    with open(BOUNDARIES_JSON) as f:
        boundaries = json.load(f)
    with open(NORM_JSON) as f:
        norm = json.load(f)
    with open(MERGED_MANIFEST) as f:
        merged_man = json.load(f)
    with open(ENV_MANIFEST) as f:
        env_man = json.load(f)

    # T1. CDC and NOAA dates align in merged dataset
    print("\n[T1] MMWR Sunday alignment between sources (re-check)")
    cdc_dates = dict(zip(cdc["year"] * 100 + cdc["week"], cdc["date"]))
    env_dates = dict(zip(env["epiweek"].astype(int), env["date"]))
    mismatches = []
    for _, r in merged.iterrows():
        ep = int(r["epiweek"])
        if cdc_dates.get(ep) != env_dates.get(ep):
            mismatches.append((ep, cdc_dates.get(ep), env_dates.get(ep)))
    if mismatches:
        fail(f"{len(mismatches)} date mismatches; first: {mismatches[:3]}")
    else:
        ok(f"All {len(merged):,} joined rows have matching MMWR Sunday across sources")

    # T2. No duplicate epiweeks, monotonic increasing
    print("\n[T2] No duplicate / monotonic epiweeks")
    if not merged["epiweek"].is_unique:
        fail("duplicate epiweeks in merged")
    if not merged["epiweek"].is_monotonic_increasing:
        fail("epiweeks not monotonic in merged")
    if not split["epiweek"].is_unique:
        fail("duplicate epiweeks in split file")
    ok("epiweek column is unique + monotonic in both merged and split")

    # T3. No NaN in target / predictors
    print("\n[T3] No NaN in target + env predictors")
    for col in ["ili_weighted_pct", "temperature_c", "specific_humidity_g_per_kg"]:
        n = merged[col].isna().sum()
        if n > 0:
            fail(f"{col} has {n} NaN rows")
    ok("target + temperature_c + specific_humidity_g_per_kg are NaN-free")

    # T4. Plausible value ranges
    print("\n[T4] Plausible value ranges")
    rng_tests = [
        ("ili_weighted_pct", 0, 20, "%wILI"),
        ("temperature_c", -30, 40, "°C (national pop-weighted mean)"),
        ("specific_humidity_g_per_kg", 0, 25, "g/kg"),
    ]
    for col, lo, hi, unit in rng_tests:
        v_min, v_max = merged[col].min(), merged[col].max()
        if v_min < lo or v_max > hi:
            fail(f"{col} out of range: [{v_min:.2f}, {v_max:.2f}] not in [{lo}, {hi}]")
        ok(f"{col}: [{v_min:.2f}, {v_max:.2f}] within [{lo}, {hi}] {unit}")

    # T5. Split boundaries non-overlapping + complete coverage of merged epiweeks
    print("\n[T5] Split boundaries non-overlapping + complete")
    boundary_ranges = [(b["epiweek_first"], b["epiweek_last"], name)
                       for name, b in boundaries["splits"].items()]
    boundary_ranges.sort()
    for i in range(len(boundary_ranges) - 1):
        end_i = boundary_ranges[i][1]
        start_j = boundary_ranges[i + 1][0]
        if end_i >= start_j:
            fail(f"overlap between {boundary_ranges[i][2]} and "
                 f"{boundary_ranges[i+1][2]}: end={end_i}, next_start={start_j}")
    ok("No overlap between train/val/covid_excluded/test boundary ranges")

    # Every merged row falls into a defined split
    oor = (split["split"] == "out_of_range").sum()
    if oor > 0:
        fail(f"{oor} rows out of all defined split boundaries")
    ok(f"All {len(split):,} rows fall into a defined split")

    # T6. Row counts sum to total
    print("\n[T6] Row counts sum")
    counts = split["split"].value_counts().to_dict()
    n_train = counts.get("train", 0)
    n_val = counts.get("val", 0)
    n_cov = counts.get("covid_excluded", 0)
    n_test = counts.get("test", 0)
    total = n_train + n_val + n_cov + n_test
    if total != len(split):
        fail(f"Sum {total} != total {len(split)}")
    ok(f"train ({n_train}) + val ({n_val}) + covid_excluded ({n_cov}) "
       f"+ test ({n_test}) = {total} ✓")

    # T7. test_post_covid is strict subset of test
    print("\n[T7] test_post_covid is strict subset of test")
    tpc = boundaries["test_post_covid"]
    test_b = boundaries["splits"]["test"]
    if not (test_b["epiweek_first"] <= tpc["epiweek_first"]
            and tpc["epiweek_last"] <= test_b["epiweek_last"]):
        fail(f"test_post_covid [{tpc['epiweek_first']}, {tpc['epiweek_last']}] "
             f"not contained in test [{test_b['epiweek_first']}, {test_b['epiweek_last']}]")
    n_tpc = ((split["split"] == "test")
             & (split["epiweek"] >= tpc["epiweek_first"])
             & (split["epiweek"] <= tpc["epiweek_last"])).sum()
    ok(f"test_post_covid ⊂ test  ({n_tpc} of {n_test} test rows are post-COVID)")

    # T8. Scaler fit on train ONLY — no leakage
    print("\n[T8] Scaler train-only fit verification (no leakage)")
    train = split[split["split"] == "train"]
    for col, p in norm["params"].items():
        if p["fit_n_rows"] != len(train):
            fail(f"{col}: fit_n_rows={p['fit_n_rows']} != train rows {len(train)}")
        recomputed_mean = float(train[col].mean())
        recomputed_std = float(train[col].std(ddof=0))
        if abs(recomputed_mean - p["mean"]) > 1e-9:
            fail(f"{col}: saved mean {p['mean']} != recomputed {recomputed_mean}")
        if abs(recomputed_std - p["std"]) > 1e-9:
            fail(f"{col}: saved std {p['std']} != recomputed {recomputed_std}")
        ok(f"{col}: train-only fit verified  (mean={p['mean']:.4f}, "
           f"std={p['std']:.4f}, n={p['fit_n_rows']})")

    # T9. SHA256 of merged CSV matches its manifest
    print("\n[T9] SHA256 file integrity vs manifest")
    s_merged = sha256_file(MERGED_CSV)
    if s_merged != merged_man["output"]["sha256"]:
        fail(f"merged CSV sha256 {s_merged[:16]} != manifest "
             f"{merged_man['output']['sha256'][:16]}")
    ok(f"ili_env_weekly.csv sha256 matches manifest")
    s_env = sha256_file(ENV_CSV)
    if s_env != env_man["output"]["sha256"]:
        fail(f"env CSV sha256 {s_env[:16]} != manifest "
             f"{env_man['output']['sha256'][:16]}")
    ok(f"env_national_weekly.csv sha256 matches manifest")

    # T10. Enumerate epiweek gaps in each split (report, do not fail).
    # Expected: train has CDC pre-2014 summer off-season gaps; other splits have none.
    print("\n[T10] Epiweek gap enumeration (informational, loader must be gap-aware)")

    def find_gaps(eps: list[int]) -> list[tuple[int, int, int]]:
        """Return list of (prev_ep, next_ep, weeks_skipped) where the sequence is
        not strictly +1 epiweek (accounting for year boundary W52/W53 -> W1)."""
        gaps = []
        for i in range(1, len(eps)):
            prev_y, prev_w = eps[i-1] // 100, eps[i-1] % 100
            curr_y, curr_w = eps[i] // 100, eps[i] % 100
            if prev_y == curr_y:
                expected = prev_w + 1
                if curr_w != expected:
                    gaps.append((eps[i-1], eps[i], curr_w - expected))
            else:
                # year boundary: prev should be 52 or 53, curr should be 1
                if not ((prev_w in (52, 53)) and curr_w == 1
                        and curr_y == prev_y + 1):
                    # Compute approximate weeks skipped by counting via 52-week year
                    weeks_skipped = ((curr_y - prev_y) * 52
                                     + (curr_w - 1) - (prev_w - 52))
                    gaps.append((eps[i-1], eps[i], weeks_skipped))
        return gaps

    # Train should have exactly 1 gap: 200220 -> 200240 (CDC 2002 summer anomaly,
    # W21-W39 of 2002 not reported). All other splits should be contiguous.
    EXPECTED_TRAIN_GAPS = [(200220, 200240)]
    for name in ["train", "val", "covid_excluded", "test"]:
        eps = split[split["split"] == name]["epiweek"].tolist()
        gaps = find_gaps(eps)
        if name == "train":
            actual = [(g[0], g[1]) for g in gaps]
            if actual != EXPECTED_TRAIN_GAPS:
                fail(f"train: gap set mismatch.\n"
                     f"  expected: {EXPECTED_TRAIN_GAPS}\n"
                     f"  actual:   {actual}")
            for prev_ep, next_ep, skip in gaps:
                ok(f"train: 1 expected gap  {prev_ep} -> {next_ep}  "
                   f"({skip} weeks skipped; CDC 2002 summer reporting hiatus)")
        else:
            if gaps:
                fail(f"{name}: unexpected {len(gaps)} gap(s) "
                     f"(first: {gaps[0]}); only train should have gaps")
            ok(f"{name}: no gaps (contiguous epiweek sequence)")

    print("\n" + "="*72)
    print(f"ALL TESTS PASSED ✅  ({len(merged):,} merged rows, splits: "
          f"train={n_train} val={n_val} test={n_test} covid_excl={n_cov})")
    print("="*72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
