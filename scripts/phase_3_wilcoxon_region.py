"""
Phase 3 Region-Level Wilcoxon Signed-Rank Tests
================================================
Compare CG-Mamba vs each baseline (LSTM, Vanilla Mamba, PatchTST, DLinear)
using paired Wilcoxon signed-rank tests.

Two test levels:
  A) Per-region (n=5 seeds): within each of the 10 HHS regions, compare
     per-seed mean-over-horizons MAE values between CG-Mamba and a baseline.
  B) Cross-region (n=10 regions): use the 10 region-mean MAE values as
     paired observations.

Statistical notes:
  - With n=5, the minimum achievable exact p-value is 0.0625 (one-sided)
    or 0.125 (two-sided), so no per-region test can reach p<0.05 two-sided.
  - With n=10, minimum two-sided p-value ≈ 0.002.
  - Holm-Bonferroni correction is applied across all CG-Mamba vs baseline
    pairs within each test level.
  - Cliff's delta effect size is reported for all comparisons.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = "/A.I_DATA/jbnu/JeongHa/CG_Mamba"
EVAL_CSV = os.path.join(ROOT, "runs/phase_3_region_eval.csv")
DLINEAR_CSV = os.path.join(ROOT, "runs/phase_3_dlinear_mae_region.csv")
OUT_JSON = os.path.join(ROOT, "runs/phase_3_wilcoxon_region.json")

# ── helpers ─────────────────────────────────────────────────────────────────

# Horizon columns for "full" (tF) split — primary metric used for ranking
# tS = strict, tF = full; we average all tF columns as per-run MAE
TF_COLS = ["tF_h1", "tF_h2", "tF_h3", "tF_h4"]
TS_COLS = ["tS_h1", "tS_h2", "tS_h3", "tS_h4"]
ALL_MAE_COLS = TF_COLS + TS_COLS  # 8 columns


def mean_over_horizons(row, cols=ALL_MAE_COLS):
    """Return average MAE across all horizon columns for one row."""
    return row[cols].mean()


def cliffs_delta(a, b):
    """
    Cliff's delta: probability that a random value from 'a' is less than
    a random value from 'b', minus the probability it is greater.
    Range [-1, 1]; negative means 'a' tends to be smaller (better if MAE).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a) * len(b)
    dominance = sum(1 if ai < bj else (-1 if ai > bj else 0)
                    for ai in a for bj in b)
    return dominance / n


def interpret_cliffs(d):
    """Qualitative label for |Cliff's delta|."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.330:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def run_wilcoxon(x, y, alternative="two-sided"):
    """
    Paired Wilcoxon signed-rank test: x vs y (same length, paired).
    Returns (statistic, p_value) or (nan, nan) if all differences are zero.
    """
    diffs = np.asarray(x) - np.asarray(y)
    if np.all(diffs == 0):
        return (np.nan, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
    return (float(stat), float(p))


def holm_bonferroni(p_values):
    """
    Holm-Bonferroni correction.
    Returns list of (reject_H0, corrected_p) in original order.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    results = [None] * n
    reject = True  # once we fail to reject, all remaining are accepted too
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = 0.05 / (n - rank)
        if reject and p <= threshold:
            results[orig_idx] = (True, min(p * (n - rank), 1.0))
        else:
            reject = False
            results[orig_idx] = (False, min(p * (n - rank), 1.0))
    return results


# ── load data ───────────────────────────────────────────────────────────────
print("=" * 72)
print("Phase 3 — Region-Level Wilcoxon Signed-Rank Tests")
print("=" * 72)

df = pd.read_csv(EVAL_CSV)

# Compute per-row mean MAE (average over all 8 horizon columns)
df["mean_mae"] = df[ALL_MAE_COLS].mean(axis=1)

print(f"\nLoaded {EVAL_CSV}")
print(f"  Rows: {len(df)}")
print(f"  Baselines: {sorted(df['baseline'].unique())}")
print(f"  Regions:   {sorted(df['region'].unique())}")
print(f"  Seeds:     {sorted(df['seed'].unique())}")

# Load DLinear (single run, per-region)
dlinear_df = pd.read_csv(DLINEAR_CSV)
dlinear_df["mean_mae"] = dlinear_df[ALL_MAE_COLS].mean(axis=1)
dlinear_mean_by_region = dlinear_df.set_index("region")["mean_mae"].to_dict()
print(f"\nLoaded DLinear (single run) for {len(dlinear_mean_by_region)} regions")

REGIONS = sorted(df["region"].unique())  # hhs1 … hhs10
BASELINES = ["lstm", "vanilla_mamba", "patchtst"]  # seeded baselines
SEEDS = sorted(df["seed"].unique())

# ── A) Per-region tests (n=5 seeds) ─────────────────────────────────────────
print("\n" + "=" * 72)
print("A) PER-REGION Wilcoxon (n=5 seeds each)")
print("   ⚠  With n=5, minimum two-sided p-value = 0.0625 (cannot reach p<0.05)")
print("=" * 72)

per_region_results = {}
all_per_region_records = []  # for Holm-Bonferroni across all region×baseline pairs

for region in REGIONS:
    per_region_results[region] = {}
    cgm_rows = df[(df["region"] == region) & (df["baseline"] == "cg_mamba")]
    cgm_by_seed = cgm_rows.set_index("seed")["mean_mae"]

    for baseline in BASELINES:
        bl_rows = df[(df["region"] == region) & (df["baseline"] == baseline)]
        bl_by_seed = bl_rows.set_index("seed")["mean_mae"]

        # Align on seeds (should be identical)
        common_seeds = sorted(set(cgm_by_seed.index) & set(bl_by_seed.index))
        if len(common_seeds) < 2:
            per_region_results[region][baseline] = {
                "error": f"insufficient common seeds: {common_seeds}"
            }
            continue

        cgm_vals = [cgm_by_seed[s] for s in common_seeds]
        bl_vals = [bl_by_seed[s] for s in common_seeds]

        stat, p = run_wilcoxon(cgm_vals, bl_vals)
        cd = cliffs_delta(cgm_vals, bl_vals)
        cgm_wins = sum(c < b for c, b in zip(cgm_vals, bl_vals))
        bl_wins = sum(b < c for c, b in zip(cgm_vals, bl_vals))

        rec = {
            "region": region,
            "baseline": baseline,
            "n_seeds": len(common_seeds),
            "cgm_mean_mae": float(np.mean(cgm_vals)),
            "baseline_mean_mae": float(np.mean(bl_vals)),
            "cgm_better_count": cgm_wins,
            "baseline_better_count": bl_wins,
            "wilcoxon_stat": stat if not np.isnan(stat) else None,
            "p_value_raw": p,
            "cliffs_delta": cd,
            "effect_size_label": interpret_cliffs(cd),
        }
        per_region_results[region][baseline] = rec
        all_per_region_records.append(rec)

# Holm-Bonferroni across all per-region pairs
valid_recs = [r for r in all_per_region_records if "p_value_raw" in r]
p_vals = [r["p_value_raw"] for r in valid_recs]
hb_results = holm_bonferroni(p_vals)
for rec, (reject, p_adj) in zip(valid_recs, hb_results):
    rec["p_value_adjusted"] = p_adj
    rec["reject_H0_alpha05"] = reject

# Print per-region table
header = f"{'Region':<8} {'Baseline':<16} {'CGM_MAE':>9} {'BL_MAE':>9} {'CGM_wins':>9} {'W_stat':>8} {'p_raw':>8} {'p_adj':>8} {'Cliff_d':>9} {'Effect':<12}"
print(f"\n{header}")
print("-" * len(header))

for region in REGIONS:
    for baseline in BASELINES:
        r = per_region_results[region][baseline]
        if "error" in r:
            print(f"{region:<8} {baseline:<16}  ERROR: {r['error']}")
            continue
        reject_str = "*" if r.get("reject_H0_alpha05") else " "
        print(
            f"{region:<8} {baseline:<16}"
            f" {r['cgm_mean_mae']:>9.4f}"
            f" {r['baseline_mean_mae']:>9.4f}"
            f" {r['cgm_better_count']:>4}/{r['n_seeds']:<4}"
            f" {(r['wilcoxon_stat'] if r['wilcoxon_stat'] is not None else float('nan')):>8.1f}"
            f" {r['p_value_raw']:>8.4f}"
            f" {r['p_value_adjusted']:>7.4f}{reject_str}"
            f" {r['cliffs_delta']:>9.3f}"
            f" {r['effect_size_label']:<12}"
        )

# DLinear per-region: n=1 (no Wilcoxon possible)
print("\nDLinear per-region: single run, no seed-based test possible")
print(f"{'Region':<8} {'CGM_MAE':>9} {'DLinear_MAE':>12} {'CGM_better':>11}")
for region in REGIONS:
    cgm_rows = df[(df["region"] == region) & (df["baseline"] == "cg_mamba")]
    cgm_mean = cgm_rows["mean_mae"].mean()
    dl_mean = dlinear_mean_by_region.get(region, float("nan"))
    better = "YES" if cgm_mean < dl_mean else "NO"
    print(f"{region:<8} {cgm_mean:>9.4f} {dl_mean:>12.4f} {better:>11}")

# ── B) Cross-region tests (n=10 regions) ─────────────────────────────────────
print("\n" + "=" * 72)
print("B) CROSS-REGION Wilcoxon (n=10 region-mean MAE values)")
print("   Paired observations: mean MAE per region (averaged over seeds × horizons)")
print("=" * 72)

# Compute per-region, per-baseline mean MAE (mean over seeds and horizons)
region_means = (
    df.groupby(["region", "baseline"])["mean_mae"]
    .mean()
    .reset_index()
    .rename(columns={"mean_mae": "region_mean_mae"})
)

cross_region_results = []

for baseline in BASELINES:
    cgm_vals = []
    bl_vals = []
    for region in REGIONS:
        cgm_val = region_means.loc[
            (region_means["region"] == region) & (region_means["baseline"] == "cg_mamba"),
            "region_mean_mae"
        ].values
        bl_val = region_means.loc[
            (region_means["region"] == region) & (region_means["baseline"] == baseline),
            "region_mean_mae"
        ].values
        if len(cgm_val) == 1 and len(bl_val) == 1:
            cgm_vals.append(float(cgm_val[0]))
            bl_vals.append(float(bl_val[0]))

    if len(cgm_vals) < 2:
        continue

    stat, p = run_wilcoxon(cgm_vals, bl_vals)
    cd = cliffs_delta(cgm_vals, bl_vals)
    cgm_wins = sum(c < b for c, b in zip(cgm_vals, bl_vals))

    rec = {
        "baseline": baseline,
        "n_regions": len(cgm_vals),
        "cgm_overall_mean": float(np.mean(cgm_vals)),
        "baseline_overall_mean": float(np.mean(bl_vals)),
        "cgm_better_count": cgm_wins,
        "wilcoxon_stat": stat if not np.isnan(stat) else None,
        "p_value_raw": p,
        "cliffs_delta": cd,
        "effect_size_label": interpret_cliffs(cd),
        "cgm_region_means": cgm_vals,
        "baseline_region_means": bl_vals,
        "regions": REGIONS,
    }
    cross_region_results.append(rec)

# DLinear cross-region
cgm_for_dl = []
dl_vals = []
for region in REGIONS:
    cgm_val = region_means.loc[
        (region_means["region"] == region) & (region_means["baseline"] == "cg_mamba"),
        "region_mean_mae"
    ].values
    dl_val = dlinear_mean_by_region.get(region, None)
    if len(cgm_val) == 1 and dl_val is not None:
        cgm_for_dl.append(float(cgm_val[0]))
        dl_vals.append(float(dl_val))

if len(cgm_for_dl) >= 2:
    stat, p = run_wilcoxon(cgm_for_dl, dl_vals)
    cd = cliffs_delta(cgm_for_dl, dl_vals)
    cgm_wins_dl = sum(c < b for c, b in zip(cgm_for_dl, dl_vals))
    cross_region_results.append({
        "baseline": "dlinear",
        "n_regions": len(cgm_for_dl),
        "cgm_overall_mean": float(np.mean(cgm_for_dl)),
        "baseline_overall_mean": float(np.mean(dl_vals)),
        "cgm_better_count": cgm_wins_dl,
        "wilcoxon_stat": stat if not np.isnan(stat) else None,
        "p_value_raw": p,
        "cliffs_delta": cd,
        "effect_size_label": interpret_cliffs(cd),
        "cgm_region_means": cgm_for_dl,
        "baseline_region_means": dl_vals,
        "regions": REGIONS,
        "note": "DLinear has single run (no seeds); MAE from phase_3_dlinear_mae_region.csv"
    })

# Holm-Bonferroni on cross-region results
cr_p_vals = [r["p_value_raw"] for r in cross_region_results]
cr_hb = holm_bonferroni(cr_p_vals)
for rec, (reject, p_adj) in zip(cross_region_results, cr_hb):
    rec["p_value_adjusted"] = p_adj
    rec["reject_H0_alpha05"] = reject

print(f"\n{'Baseline':<16} {'n':>3} {'CGM_MAE':>9} {'BL_MAE':>9} {'CGM_wins':>9} {'W_stat':>8} {'p_raw':>8} {'p_adj':>8} {'Cliff_d':>9} {'Effect':<12}")
print("-" * 100)
for r in cross_region_results:
    reject_str = "*" if r.get("reject_H0_alpha05") else " "
    print(
        f"{r['baseline']:<16}"
        f" {r['n_regions']:>3}"
        f" {r['cgm_overall_mean']:>9.4f}"
        f" {r['baseline_overall_mean']:>9.4f}"
        f" {r['cgm_better_count']:>4}/{r['n_regions']:<4}"
        f" {(r['wilcoxon_stat'] if r['wilcoxon_stat'] is not None else float('nan')):>8.1f}"
        f" {r['p_value_raw']:>8.4f}"
        f" {r['p_value_adjusted']:>7.4f}{reject_str}"
        f" {r['cliffs_delta']:>9.3f}"
        f" {r['effect_size_label']:<12}"
    )

# ── Summary notes ────────────────────────────────────────────────────────────
NOTES = [
    "Sample size limitations:",
    "  Per-region tests (n=5 seeds): minimum achievable exact two-sided p = 0.0625.",
    "    No per-region test can reach p<0.05 under the exact Wilcoxon distribution.",
    "    Results should be interpreted via effect size (Cliff's delta) and win-counts.",
    "  Cross-region tests (n=10 regions): minimum two-sided p ≈ 0.002.",
    "    Significance claims at α=0.05 are achievable here.",
    "  DLinear has no seed replication in phase_3_region_eval.csv.",
    "    Per-region DLinear comparison uses point estimate only (no test).",
    "    Cross-region DLinear Wilcoxon uses n=10 region means paired with CGM means.",
    "Holm-Bonferroni correction applied separately within each test level (A and B).",
    "MAE is averaged over all 8 horizon×split columns (tF_h1..tS_h4) per seed/row.",
    "* = reject H0 after Holm-Bonferroni at α=0.05",
]
print("\n")
print("NOTES:")
for note in NOTES:
    print(f"  {note}")

# ── Save JSON ────────────────────────────────────────────────────────────────
output = {
    "meta": {
        "description": "Phase 3 region-level Wilcoxon signed-rank tests: CG-Mamba vs baselines",
        "test_levels": {
            "A_per_region": "n=5 seeds, min two-sided p=0.0625 (cannot reach p<0.05 exactly)",
            "B_cross_region": "n=10 region-means, min two-sided p≈0.002"
        },
        "mae_definition": "mean over tF_h1, tF_h2, tF_h3, tF_h4, tS_h1, tS_h2, tS_h3, tS_h4",
        "correction": "Holm-Bonferroni within each test level",
        "effect_size": "Cliff's delta (negative = CG-Mamba tends lower/better MAE)",
        "notes": NOTES,
    },
    "A_per_region": {
        region: {
            baseline: rec
            for baseline, rec in per_region_results[region].items()
        }
        for region in REGIONS
    },
    "B_cross_region": cross_region_results,
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to: {OUT_JSON}")
print("Done.")
