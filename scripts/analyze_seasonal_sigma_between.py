"""
analyze_seasonal_sigma_between.py
==================================
Seasonal overlay analysis of sigma2_between from WIS decomposition.

Question: Does sigma2_between (HMM-vs-CGM disagreement) spike at predictable
seasonal times (onset W40-48, peak W4-8, decline W10-20), or is it noise?

Data:
  - runs/wis_method_f/decomposition_temporal.csv
      sample_idx | horizon | target_ep (YYYYWW) | sigma2_between_HMM | ...
  - data/processed/ili_env_weekly.csv
      epiweek | ili_weighted_pct | ...

Test period: test_strict = W40-2022 ~ W35-2025 (3 post-recovery seasons)
Seasons:  2022-23  (W40-2022 ~ W39-2023)
          2023-24  (W40-2023 ~ W39-2024)
          2024-25  (W40-2024 ~ W35-2025, partial)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent
DECOMP = BASE / "runs/wis_method_f/decomposition_temporal.csv"
ILI    = BASE / "data/processed/ili_env_weekly.csv"
OUT    = BASE / "runs/wis_method_f/seasonal_overlay.pdf"

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DECOMP)
ili = pd.read_csv(ILI)

print(f"Decomposition rows: {len(df)}  |  horizons: {sorted(df['horizon'].unique())}")
print(f"target_ep range: {df['target_ep'].min()} – {df['target_ep'].max()}")

# ── 2. Derive epi-week and season labels ──────────────────────────────────────
df["year_ep"]  = df["target_ep"] // 100
df["epiweek_num"] = df["target_ep"] % 100

# Flu season: W40 of year Y → W39 of year Y+1  ⇒  season labelled by Y
def _season(row):
    y, w = int(row["year_ep"]), int(row["epiweek_num"])
    sy = y if w >= 40 else y - 1
    return f"{sy}-{str(sy+1)[2:]}"

df["season"] = df.apply(_season, axis=1)

# Season-relative week: map to a single 1–52 axis where W40 = week_pos 1
def _season_pos(w):
    """Map CDC epiweek to season-relative position: W40=1, W52=13, W1=14, W39=52."""
    if w >= 40:
        return w - 39          # W40→1, W41→2, …, W52→13
    else:
        return w + 13          # W1→14, W2→15, …, W39→52

df["season_pos"] = df["epiweek_num"].apply(_season_pos)

# ── 3. Average sigma2_between across horizons (also keep h=1) ─────────────────
# Mean across h=1..4 per (sample_idx, season, epiweek_num)
agg = (df.groupby(["sample_idx", "season", "epiweek_num", "season_pos"])
         ["sigma2_between_HMM"]
         .mean()
         .reset_index()
         .rename(columns={"sigma2_between_HMM": "sigma2_between_mean_h"}))

h1  = (df[df["horizon"] == 1]
       [["sample_idx","season","epiweek_num","season_pos","sigma2_between_HMM"]]
       .rename(columns={"sigma2_between_HMM": "sigma2_between_h1"}))

wide = agg.merge(h1, on=["sample_idx","season","epiweek_num","season_pos"])

seasons = sorted(wide["season"].unique())
print(f"\nSeasons found: {seasons}")
for s in seasons:
    n = wide[wide["season"]==s].shape[0]
    print(f"  {s}: {n} windows")

# ── 4. ILI actual for the test period (overlay reference) ─────────────────────
ili_test = ili[ili["epiweek"] >= 202240].copy()
ili_test["year_ep"]    = ili_test["epiweek"] // 100
ili_test["epiweek_num"] = ili_test["epiweek"] % 100
ili_test["season"]     = ili_test.apply(
    lambda r: (lambda sy: f"{sy}-{str(sy+1)[2:]}")(
        int(r["year_ep"]) if int(r["epiweek_num"]) >= 40 else int(r["year_ep"]) - 1),
    axis=1)
ili_test["season_pos"] = ili_test["epiweek_num"].apply(_season_pos)

# ── 5. Cross-season mean σ²_between per season_pos ────────────────────────────
mean_by_pos = (wide.groupby("season_pos")["sigma2_between_mean_h"]
                   .agg(["mean","std","count"])
                   .reset_index()
                   .rename(columns={"mean":"mu","std":"sd","count":"n"}))
mean_by_pos["se"] = mean_by_pos["sd"] / np.sqrt(mean_by_pos["n"])

# ── 6. Print summary statistics ───────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY: σ²_between (mean across h=1..4) per epi-season-position")
print("="*65)
print(f"{'season_pos':>10}  {'epiweek':>7}  {'mu':>8}  {'sd':>8}  {'n':>4}")
for _, row in mean_by_pos.sort_values("season_pos").iterrows():
    # Convert back to epiweek label
    sp = int(row["season_pos"])
    ew = sp + 39 if sp <= 13 else sp - 13
    label = f"W{ew:02d}"
    print(f"{sp:>10}  {label:>7}  {row['mu']:>8.4f}  {row['sd']:>8.4f}  {int(row['n']):>4}")

# Peak / trough
peak_row  = mean_by_pos.loc[mean_by_pos["mu"].idxmax()]
trough_row = mean_by_pos.loc[mean_by_pos["mu"].idxmin()]
sp_to_ew  = lambda sp: sp + 39 if sp <= 13 else sp - 13
print(f"\nPeak   σ²_between: {peak_row['mu']:.4f} at season_pos={int(peak_row['season_pos'])}"
      f"  (W{sp_to_ew(int(peak_row['season_pos'])):02d})")
print(f"Trough σ²_between: {trough_row['mu']:.4f} at season_pos={int(trough_row['season_pos'])}"
      f"  (W{sp_to_ew(int(trough_row['season_pos'])):02d})")
print(f"Peak/Trough ratio: {peak_row['mu']/trough_row['mu']:.2f}×")

# Correlation with ILI mean per season_pos
ili_mean_pos = (ili_test.groupby("season_pos")["ili_weighted_pct"].mean().reset_index()
                .rename(columns={"ili_weighted_pct":"ili_mean"}))
merged_corr = mean_by_pos.merge(ili_mean_pos, on="season_pos", how="inner")
if len(merged_corr) > 3:
    r = merged_corr["mu"].corr(merged_corr["ili_mean"])
    print(f"\nCorrelation(σ²_between, ILI_mean) across season positions: r = {r:.3f}")

# Epidemic phase breakdown
phase_map = {
    "onset  (W40-W48, pos 1-9)":  (1,  9),
    "peak   (W49-W08, pos10-21)": (10, 21),
    "decline(W09-W20, pos22-33)": (22, 33),
    "off    (W21-W39, pos34-52)": (34, 52),
}
print("\nPhase averages:")
for label, (lo, hi) in phase_map.items():
    subset = mean_by_pos[(mean_by_pos["season_pos"]>=lo) & (mean_by_pos["season_pos"]<=hi)]
    if len(subset):
        print(f"  {label}:  mu={subset['mu'].mean():.4f}  (n_weeks={len(subset)})")

# ── 7. Build the figure ───────────────────────────────────────────────────────
SEASON_COLORS = {
    "2022-23": "#1f77b4",
    "2023-24": "#ff7f0e",
    "2024-25": "#2ca02c",
}
ILI_COLORS = {
    "2022-23": "#aec7e8",
    "2023-24": "#ffbb78",
    "2024-25": "#98df8a",
}

fig, axes = plt.subplots(3, 1, figsize=(13, 14), gridspec_kw={"height_ratios":[2.5,1.5,1]})
fig.suptitle(
    "Seasonal Overlay of σ²_between (HMM–CGM Disagreement)\n"
    "US ILI Forecasts · Test-strict period (2022-25, 3 seasons)",
    fontsize=13, fontweight="bold", y=0.995)

# ── Panel A: per-season lines + cross-season mean ─────────────────────────────
ax = axes[0]
for s in seasons:
    sub = wide[wide["season"]==s].sort_values("season_pos")
    ax.plot(sub["season_pos"], sub["sigma2_between_mean_h"],
            color=SEASON_COLORS.get(s, "grey"),
            linewidth=1.6, alpha=0.75, label=f"Season {s}")

# Cross-season mean ± 1 SE
pos_sorted = mean_by_pos.sort_values("season_pos")
ax.fill_between(pos_sorted["season_pos"],
                pos_sorted["mu"] - pos_sorted["se"],
                pos_sorted["mu"] + pos_sorted["se"],
                color="black", alpha=0.12)
ax.plot(pos_sorted["season_pos"], pos_sorted["mu"],
        color="black", linewidth=2.5, linestyle="--", label="Cross-season mean ±SE")

# Phase shading
phase_shades = [
    (1,  9,  "onset\n(W40-48)",  "#ffd700", 0.12),
    (10, 21, "peak\n(W49-W08)", "#ff6b6b", 0.12),
    (22, 33, "decline\n(W09-20)","#90ee90", 0.10),
]
for lo, hi, lbl, col, alpha in phase_shades:
    ax.axvspan(lo, hi, color=col, alpha=alpha, zorder=0)
# Add phase labels after axis limits are set
ax.autoscale_view()
y_top = ax.get_ylim()[1]
for lo, hi, lbl, col, alpha in phase_shades:
    ax.text((lo+hi)/2, y_top * 0.98, lbl, ha="center", va="top",
            fontsize=7, color="grey")

ax.set_xlim(1, 52)
ax.set_xticks([1,5,10,14,17,21,26,30,35,40,45,50,52])
ew_labels = [f"W{sp_to_ew(p):02d}" for p in [1,5,10,14,17,21,26,30,35,40,45,50,52]]
ax.set_xticklabels(ew_labels, fontsize=8)
ax.set_ylabel("σ²_between (mean h=1–4)", fontsize=10)
ax.set_title("Panel A: Per-season overlay + cross-season mean", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", linestyle=":", alpha=0.5)

# Annotate peak in mean
peak_sp  = int(peak_row["season_pos"])
peak_mu  = float(peak_row["mu"])
ax.annotate(f"Peak W{sp_to_ew(peak_sp):02d}\n{peak_mu:.3f}",
            xy=(peak_sp, peak_mu), xytext=(peak_sp+3, peak_mu*1.05),
            fontsize=8, color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

# ── Panel B: ILI actuals per season (reference signal) ────────────────────────
ax2 = axes[1]
for s in seasons:
    sub_ili = ili_test[ili_test["season"]==s].sort_values("season_pos")
    ax2.plot(sub_ili["season_pos"], sub_ili["ili_weighted_pct"],
             color=SEASON_COLORS.get(s, "grey"),
             linewidth=1.6, alpha=0.8, label=f"ILI {s}")

for lo, hi, lbl, col, alpha in phase_shades:
    ax2.axvspan(lo, hi, color=col, alpha=alpha, zorder=0)

ax2.set_xlim(1, 52)
ax2.set_xticks([1,5,10,14,17,21,26,30,35,40,45,50,52])
ax2.set_xticklabels(ew_labels, fontsize=8)
ax2.set_ylabel("ILI weighted % (actual)", fontsize=10)
ax2.set_title("Panel B: Actual ILI (reference — aligned season axis)", fontsize=10)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(axis="y", linestyle=":", alpha=0.5)

# ── Panel C: scatter σ²_between vs ILI (cross-season mean per pos) ────────────
ax3 = axes[2]
if len(merged_corr) > 3:
    sc = ax3.scatter(merged_corr["ili_mean"], merged_corr["mu"],
                     c=merged_corr["season_pos"], cmap="plasma",
                     s=40, alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=ax3, label="season_pos (1=W40, 52=W39)")
    # Trend line
    z = np.polyfit(merged_corr["ili_mean"], merged_corr["mu"], 1)
    xfit = np.linspace(merged_corr["ili_mean"].min(), merged_corr["ili_mean"].max(), 100)
    ax3.plot(xfit, np.poly1d(z)(xfit), "r--", linewidth=1.5, alpha=0.8,
             label=f"trend (r={r:.2f})")
    ax3.set_xlabel("Mean ILI % (cross-season average per week position)", fontsize=9)
    ax3.set_ylabel("Mean σ²_between", fontsize=9)
    ax3.set_title(
        f"Panel C: σ²_between vs ILI by season position  (r = {r:.3f})", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(linestyle=":", alpha=0.5)
else:
    ax3.text(0.5, 0.5, "Not enough overlap for scatter", ha="center", va="center",
             transform=ax3.transAxes)

# ── Finalise ──────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(OUT, bbox_inches="tight", dpi=150)
print(f"\nPlot saved → {OUT}")

# ── 8. Additional CSV summary ─────────────────────────────────────────────────
csv_out = OUT.parent / "seasonal_overlay_summary.csv"
# Pivot: rows=season_pos, cols=seasons + mean
pivot = (wide.pivot_table(index="season_pos", columns="season",
                          values="sigma2_between_mean_h", aggfunc="mean")
            .merge(mean_by_pos[["season_pos","mu","sd"]].set_index("season_pos"),
                   left_index=True, right_index=True, how="left"))
pivot.index.name = "season_pos"
pivot.to_csv(csv_out)
print(f"Summary CSV saved → {csv_out}")
print("\nDone.")
