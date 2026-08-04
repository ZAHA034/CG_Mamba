"""
analyze_dsigma_dt_roc.py
========================
Tests whether the RATE OF CHANGE of sigma2_between (dσ²/dt) is a better
detector of HMM phase transition weeks than the raw level.

Background
----------
A previous ROC analysis used raw σ²_between as a binary classifier for
"is this a phase transition week?" and got AUC = 0.204 (POOR).

Hypothesis: Actual transitions cause a SHARP RISE in σ²_between, while
boundary-hovering gives sustained medium values. The derivative might
separate these two cases better.

Detectors evaluated
-------------------
  D0  raw σ²_between (baseline, should reproduce ~0.204)
  D1  dσ²/dt  = σ²[t] − σ²[t−1]   (signed first difference)
  D2  |dσ²/dt|                      (unsigned rate of change)
  D3  rolling 2-week max of |dσ²/dt|
  D4  rolling 3-week max of |dσ²/dt|

Transition labels
-----------------
Two schemes are tried:

  SCHEME A – ILI threshold crossings (heuristic epidemic onset/offset):
    A week t is labelled "transition=1" if ILI_t crosses 2.0% relative to
    ILI_{t-1} (i.e. sign(ILI_t − 2.0) ≠ sign(ILI_{t-1} − 2.0)).
    Also tested at thresholds 1.5% and 2.5%.

  SCHEME B – Steepest ILI gradient (empirical inflection points):
    Top-quartile |ΔILI_t| weeks are labelled transition=1.
    These capture rapid changes regardless of absolute level.

Outputs
-------
  stdout : summary table of AUC values
  runs/wis_method_f/dsigma_dt_roc.pdf : ROC curves + ILI/σ² time series
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent
DECOMP = BASE / "runs/wis_method_f/decomposition_temporal.csv"
ILI    = BASE / "data/processed/ili_env_weekly_split.csv"
OUT    = BASE / "runs/wis_method_f/dsigma_dt_roc.pdf"

# ── 1. Load and aggregate decomposition data ───────────────────────────────────
df = pd.read_csv(DECOMP)
print(f"Decomposition: {len(df)} rows, {df['sample_idx'].nunique()} windows, "
      f"horizons {sorted(df['horizon'].unique())}")

# Average sigma2_between across h=1..4 per sample_idx (window)
agg = (df.groupby("sample_idx")["sigma2_between_HMM"]
         .mean()
         .reset_index()
         .rename(columns={"sigma2_between_HMM": "s2b_mean"}))

# Attach the h=1 target_ep as the window's calendar date
h1 = df[df["horizon"] == 1][["sample_idx", "target_ep"]].copy()
agg = agg.merge(h1, on="sample_idx").sort_values("sample_idx").reset_index(drop=True)

print(f"Aggregated windows: {len(agg)}  "
      f"(epiweeks {agg['target_ep'].min()} – {agg['target_ep'].max()})")

# ── 2. Load ILI data aligned to test_strict window ────────────────────────────
ili_df = pd.read_csv(ILI)
strict = (ili_df[(ili_df["split"] == "test") & (ili_df["epiweek"] >= 202240)]
          [["epiweek", "ili_weighted_pct"]]
          .sort_values("epiweek")
          .reset_index(drop=True))

print(f"Strict-test ILI: {len(strict)} weeks  "
      f"({strict['epiweek'].min()} – {strict['epiweek'].max()})")

# Merge ILI onto aggregated windows by epiweek
agg = agg.merge(strict.rename(columns={"epiweek": "target_ep"}),
                on="target_ep", how="left")

missing_ili = agg["ili_weighted_pct"].isna().sum()
if missing_ili:
    print(f"WARNING: {missing_ili} windows have no matching ILI → will be dropped")
    agg = agg.dropna(subset=["ili_weighted_pct"]).reset_index(drop=True)

N = len(agg)
print(f"Final analysis windows: {N}")

# ── 3. Compute detector signals ────────────────────────────────────────────────
s2b = agg["s2b_mean"].values.copy()   # raw sigma2_between, shape (N,)

# First difference (NaN-padded at position 0)
d_s2b     = np.empty(N); d_s2b[:]  = np.nan
d_s2b[1:] = s2b[1:] - s2b[:-1]     # dσ²/dt

abs_d     = np.abs(d_s2b)           # |dσ²/dt|

# Rolling max helpers (NaN-safe, min_periods=1)
def rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window, min_periods=1).max().values

roll2 = rolling_max(np.nan_to_num(abs_d, nan=0.0), 2)
roll3 = rolling_max(np.nan_to_num(abs_d, nan=0.0), 3)

detectors = {
    "D0  raw σ²":          s2b,
    "D1  dσ²/dt":          np.nan_to_num(d_s2b,  nan=0.0),
    "D2  |dσ²/dt|":        np.nan_to_num(abs_d,  nan=0.0),
    "D3  roll2-max|dσ²|":  roll2,
    "D4  roll3-max|dσ²|":  roll3,
}

# ── 4. Build transition labels ─────────────────────────────────────────────────
ili_vals = agg["ili_weighted_pct"].values

def threshold_crossing_label(ili: np.ndarray, thresh: float) -> np.ndarray:
    """1 where ILI crosses thresh relative to previous week."""
    above = (ili > thresh).astype(int)
    lbl   = np.abs(np.diff(above, prepend=above[0]))
    lbl[0] = 0   # first week has no predecessor
    return lbl.astype(int)

def steepest_ili_label(ili: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    """Top-quantile |ΔILI| weeks are labelled 1."""
    delta = np.abs(np.diff(ili, prepend=ili[0]))
    delta[0] = 0.0
    cutoff = np.quantile(delta[1:], quantile)
    lbl    = (delta >= cutoff).astype(int)
    lbl[0] = 0
    return lbl

# Scheme A at three thresholds
labels = {
    "A-1.5%":  threshold_crossing_label(ili_vals, 1.5),
    "A-2.0%":  threshold_crossing_label(ili_vals, 2.0),
    "A-2.5%":  threshold_crossing_label(ili_vals, 2.5),
    "B-top25%": steepest_ili_label(ili_vals, 0.75),
    "B-top33%": steepest_ili_label(ili_vals, 0.67),
}

for name, lbl in labels.items():
    pos = lbl.sum()
    print(f"Label '{name}': {pos} positive / {len(lbl)} total  "
          f"({100*pos/len(lbl):.1f}%)")

# ── 5. Compute ROC AUC for every (detector × label) pair ──────────────────────
print("\n" + "="*70)
print(f"{'Detector':<26}  " + "  ".join(f"{k:>9}" for k in labels))
print("="*70)

results: dict[str, dict[str, float]] = {}
for det_name, det_vals in detectors.items():
    row = {}
    for lbl_name, lbl in labels.items():
        if lbl.sum() < 2 or (len(lbl) - lbl.sum()) < 2:
            row[lbl_name] = float("nan")
            continue
        try:
            auc = roc_auc_score(lbl, det_vals)
        except Exception as e:
            auc = float("nan")
            print(f"  WARNING: AUC failed for {det_name}/{lbl_name}: {e}")
        row[lbl_name] = auc
    results[det_name] = row
    vals_str = "  ".join(f"{row[k]:>9.3f}" if not np.isnan(row[k]) else f"{'nan':>9}"
                         for k in labels)
    print(f"{det_name:<26}  {vals_str}")

print("="*70)

# ── 6. Best result summary ─────────────────────────────────────────────────────
print("\n--- Best AUC per label scheme ---")
for lbl_name in labels:
    best_det  = max(results, key=lambda d: results[d].get(lbl_name, 0.0))
    best_auc  = results[best_det][lbl_name]
    base_auc  = results["D0  raw σ²"][lbl_name]
    delta     = best_auc - base_auc
    print(f"  {lbl_name}: best={best_auc:.3f} ({best_det.strip()})  "
          f"vs baseline={base_auc:.3f}  Δ={delta:+.3f}")

# ── 7. Plot ────────────────────────────────────────────────────────────────────
N_lbl = len(labels)
N_det = len(detectors)

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, N_lbl, figure=fig, hspace=0.55, wspace=0.4)

COLORS  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
det_items = list(detectors.items())

# ── Row 0: ROC curves per label scheme ────────────────────────────────────────
for j, (lbl_name, lbl) in enumerate(labels.items()):
    ax = fig.add_subplot(gs[0, j])
    for i, (det_name, det_vals) in enumerate(det_items):
        if np.isnan(results[det_name][lbl_name]):
            continue
        fpr, tpr, _ = roc_curve(lbl, det_vals)
        auc = results[det_name][lbl_name]
        short = det_name.strip().split()[0]
        ax.plot(fpr, tpr, color=COLORS[i], lw=1.5,
                label=f"{short} {auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=0.8, alpha=0.5)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"Label: {lbl_name}", fontsize=9)
    ax.set_xlabel("FPR", fontsize=8)
    if j == 0:
        ax.set_ylabel("TPR", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)

# ── Row 1: Time series – ILI + transition labels ──────────────────────────────
ax_ili = fig.add_subplot(gs[1, :])
t = np.arange(N)

ax_ili.plot(t, ili_vals, color="steelblue", lw=1.4, label="ILI %", zorder=3)
ax_ili.axhline(2.0, color="grey", lw=0.8, linestyle=":", alpha=0.7)
ax_ili.axhline(1.5, color="grey", lw=0.8, linestyle=":", alpha=0.5)
ax_ili.axhline(2.5, color="grey", lw=0.8, linestyle=":", alpha=0.5)

# Mark A-2.0% transitions
lbl_A20 = labels["A-2.0%"]
for idx in np.where(lbl_A20 == 1)[0]:
    ax_ili.axvline(idx, color="red", lw=0.7, alpha=0.5)

# Mark B-top25% transitions
lbl_B25 = labels["B-top25%"]
for idx in np.where(lbl_B25 == 1)[0]:
    ax_ili.axvline(idx, color="darkorange", lw=0.7, alpha=0.4, linestyle="--")

ax_ili.set_ylabel("ILI %", fontsize=9)
ax_ili.set_xlabel("Window index", fontsize=9)
ax_ili.set_title("ILI time series with transition labels "
                 "(red=A-2.0% crossings, orange-dashed=B-top25% steepest)", fontsize=9)
ax_ili.legend(fontsize=8)
ax_ili.tick_params(labelsize=8)

# ── Row 2: Time series – sigma2 and derivatives ───────────────────────────────
ax_sig = fig.add_subplot(gs[2, :])
ax_sig2 = ax_sig.twinx()

ax_sig.plot(t, s2b,    color="steelblue",  lw=1.2, label="σ² (raw)", alpha=0.9)
ax_sig.plot(t, roll3,  color="purple",     lw=1.2, label="|dσ²| roll3", alpha=0.8,
            linestyle="--")
ax_sig2.plot(t, np.nan_to_num(d_s2b, 0.0), color="coral", lw=0.9,
             label="dσ²/dt", alpha=0.7, linestyle=":")

# Mark A-2.0% transitions
for idx in np.where(lbl_A20 == 1)[0]:
    ax_sig.axvline(idx, color="red", lw=0.7, alpha=0.4)

ax_sig.set_ylabel("σ²_between (level / roll-max)", fontsize=9)
ax_sig2.set_ylabel("dσ²/dt (signed)", fontsize=9, color="coral")
ax_sig2.tick_params(axis="y", colors="coral", labelsize=7)
ax_sig.set_xlabel("Window index", fontsize=9)
ax_sig.set_title("σ²_between level vs rate-of-change (red verticals = A-2.0% transitions)", fontsize=9)
lines1, labs1 = ax_sig.get_legend_handles_labels()
lines2, labs2 = ax_sig2.get_legend_handles_labels()
ax_sig.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")
ax_sig.tick_params(labelsize=8)

fig.suptitle("dσ²_between/dt as Phase-Transition Detector\n"
             "AUC comparison: raw level vs first-difference vs rolling max",
             fontsize=11, fontweight="bold", y=0.99)

fig.savefig(OUT, bbox_inches="tight", dpi=150)
print(f"\nSaved: {OUT}")

# ── 8. Print epiweek-annotated AUC table for the main label scheme (A-2.0%) ───
print("\n--- Epiweek annotation for A-2.0% transition weeks ---")
trans_idx = np.where(labels["A-2.0%"] == 1)[0]
for idx in trans_idx:
    ep  = int(agg.loc[idx, "target_ep"])
    ili = float(agg.loc[idx, "ili_weighted_pct"])
    s2b_v  = float(s2b[idx])
    d_v    = float(0.0 if np.isnan(d_s2b[idx]) else d_s2b[idx])
    r3_v   = float(roll3[idx])
    print(f"  ep={ep}  ILI={ili:.3f}%  σ²={s2b_v:.4f}  dσ²={d_v:+.4f}  roll3={r3_v:.4f}")

print("\nDone.")
