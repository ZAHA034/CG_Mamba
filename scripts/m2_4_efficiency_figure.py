"""
m2_4_efficiency_figure.py
=========================
Publication-quality 2-panel data efficiency figure for CG-Mamba.

Panel A: test_strict avg MAE vs training seasons
Panel B: test_strict avg WIS vs training seasons

Both panels: one 5-seed-mean line per baseline,
             no error bar for SARIMA (deterministic, seed=-1).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# IEEE-compliant font embedding: TrueType (Type42) instead of matplotlib's default
# Type3 (not text-searchable -> IEEE Xplore/PDF eXpress flag), Helvetica/Arial-class
# family, ASCII minus so no DejaVu fallback. Matches Fig 3/4 (phase_3_region_hybrid.py).
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": False,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "runs" / "m2_4_data_efficiency"
OUT_DIR  = DATA_DIR

MAE_CSV = DATA_DIR / "m2_4_test_strict_all_baselines.csv"
WIS_CSV = DATA_DIR / "m2_4_wis_protocol.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VARIANT_ORDER = ["3_seasons", "4_seasons", "5_seasons", "7_seasons",
                 "10_seasons", "13_seasons", "17_seasons_full"]
SEASON_LABELS = [3, 4, 5, 7, 10, 13, 17]

BASELINES = ["cg_mamba", "vanilla_mamba", "lstm", "patchtst", "epideep", "dlinear"]
# SARIMA removed from main-text figure for DL-family-only visual scope;
# SARIMA reference trajectories provided in Supplementary Material.

DISPLAY_NAMES = {
    "cg_mamba":      "CG-Mamba",
    "sarima":        "SARIMA",
    "vanilla_mamba": "Vanilla Mamba",
    "lstm":          "LSTM",
    "patchtst":      "PatchTST",
    "epideep":       "EpiDeep",
    "dlinear":       "DLinear",
}

# Visual style. Colors = Okabe-Ito CVD-safe palette, identical to Fig 3/4
# (phase_3_region_hybrid.py COLORS) so each model has one consistent, colour-blind-
# safe colour paper-wide (removes the earlier red+green CVD-unsafe pair).
STYLE = {
    "cg_mamba":      {"color": "#0072B2", "lw": 1.3, "ls": "-",  "marker": "o", "ms": 4,  "zorder": 10},
    "sarima":        {"color": "#888888", "lw": 0.8, "ls": "--", "marker": "s", "ms": 3,  "zorder": 3},
    "vanilla_mamba": {"color": "#E69F00", "lw": 0.8, "ls": "-",  "marker": "^", "ms": 3,  "zorder": 7},
    "lstm":          {"color": "#D55E00", "lw": 0.8, "ls": "-",  "marker": "D", "ms": 3,  "zorder": 6},
    "patchtst":      {"color": "#009E73", "lw": 0.8, "ls": "-",  "marker": "v", "ms": 3,  "zorder": 5},
    "epideep":       {"color": "#56B4E9", "lw": 0.8, "ls": "-",  "marker": "X", "ms": 3,  "zorder": 4},
    "dlinear":       {"color": "#CC79A7", "lw": 0.8, "ls": "-",  "marker": "P", "ms": 3,  "zorder": 4},
}

DETERMINISTIC_BASELINES = {"sarima"}  # seed=-1, no std

# ---------------------------------------------------------------------------
# Load & aggregate MAE
# ---------------------------------------------------------------------------
mae_df = pd.read_csv(MAE_CSV)

# Compute per-row mean of the 4 strict horizons
mae_cols = ["test_strict_mae_h1", "test_strict_mae_h2",
            "test_strict_mae_h3", "test_strict_mae_h4"]
mae_df["avg_strict_mae"] = mae_df[mae_cols].mean(axis=1)

# Aggregate: mean and std across seeds per (baseline, variant)
mae_agg = (
    mae_df.groupby(["baseline", "variant"])["avg_strict_mae"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "mae_mean", "std": "mae_std"})
)
# Map variants to integer season counts
var2season = dict(zip(VARIANT_ORDER, SEASON_LABELS))
mae_agg["seasons"] = mae_agg["variant"].map(var2season)
mae_agg = mae_agg.dropna(subset=["seasons"])

# ---------------------------------------------------------------------------
# Load & aggregate WIS
# ---------------------------------------------------------------------------
wis_df = pd.read_csv(WIS_CSV)
# Sweep protocol fixes Vanilla Mamba MC-Dropout at d=0.2 across all sizes; the CSV also
# carries a d=0.1 row at 17_seasons_full (merged from the main-table eval). Drop it so all
# seven points use one rate (otherwise the 17-season point silently averages two rates).
wis_df = wis_df[~((wis_df["baseline"] == "vanilla_mamba") & (wis_df["uq_method"] == "mc_dropout_p0.1"))]

wis_cols = ["test_strict_wis_h1", "test_strict_wis_h2",
            "test_strict_wis_h3", "test_strict_wis_h4"]
wis_df["avg_strict_wis"] = wis_df[wis_cols].mean(axis=1)

# For DLinear the WIS CSV has one aggregated row per (baseline, variant) with seed=-1
# (ensemble_gaussian_5seed). For CG-Mamba it's per seed.
# We aggregate by taking mean across rows that have per-seed data,
# but keep deterministic rows (seed=-1) as-is.
wis_agg = (
    wis_df.groupby(["baseline", "variant"])["avg_strict_wis"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "wis_mean", "std": "wis_std"})
)
wis_agg["seasons"] = wis_agg["variant"].map(var2season)
wis_agg = wis_agg.dropna(subset=["seasons"])

# ---------------------------------------------------------------------------
# Aggregate Cov95 from same WIS CSV
# ---------------------------------------------------------------------------
cov_cols = ["test_strict_cov95_h1", "test_strict_cov95_h2",
            "test_strict_cov95_h3", "test_strict_cov95_h4"]
wis_df["avg_strict_cov95"] = wis_df[cov_cols].mean(axis=1)

cov_agg = (
    wis_df.groupby(["baseline", "variant"])["avg_strict_cov95"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "cov_mean", "std": "cov_std"})
)
cov_agg["seasons"] = cov_agg["variant"].map(var2season)
cov_agg = cov_agg.dropna(subset=["seasons"])

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_panel(ax, agg_df, y_col_mean, y_col_std, ylabel, baselines, title_letter,
               show_xlabel=True):
    # Numerical scale: actual season counts (3, 4, 5, 7, 10, 13, 17) on a linear axis
    x_positions = np.array(SEASON_LABELS, dtype=float)

    for bl in baselines:
        sub = agg_df[agg_df["baseline"] == bl].copy()
        if sub.empty:
            continue
        sub = sub.set_index("seasons").reindex(SEASON_LABELS)
        y_mean = sub[y_col_mean].values
        y_std  = sub[y_col_std].values

        sty = STYLE[bl]
        label = DISPLAY_NAMES[bl]

        # mask NaN positions
        valid = ~np.isnan(y_mean)

        ax.plot(
            x_positions[valid], y_mean[valid],
            label=label,
            color=sty["color"],
            lw=sty["lw"],
            ls=sty["ls"],
            marker=sty["marker"],
            markersize=sty["ms"],
            zorder=sty["zorder"],
            clip_on=False,
        )

        # Mean line only (±1 std bands removed: cleaner for the 6-model overlay;
        # 5-seed mean is stated in the caption).

    # X-axis formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SEASON_LABELS, fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Training set size (seasons)", fontsize=8, labelpad=3)
    else:
        ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=8, labelpad=3)

    # Panel letter label (placed clearly above y-axis area, not overlapping ticks)
    ax.text(-0.18, 1.10, title_letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")

    # Minimal spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", direction="out",
                   labelsize=8, length=3, width=0.8)

    # Light horizontal grid only
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    return ax


# ---------------------------------------------------------------------------
# Build figure (single-column stacked layout: 2 rows x 1 column)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.2),
                         constrained_layout=False,
                         gridspec_kw={"hspace": 0.55})

# --- Panel A: WIS (no xlabel; shares x-axis with Panel B) ---
ax_a = axes[0]
plot_panel(
    ax_a, wis_agg,
    y_col_mean="wis_mean",
    y_col_std="wis_std",
    ylabel="WIS",
    baselines=BASELINES,
    title_letter="A",
    show_xlabel=False,
)
ax_a.set_title("WIS (test_strict)", fontsize=9, pad=4)

# --- Panel B: Cov95 (xlabel shown) ---
ax_b = axes[1]
plot_panel(
    ax_b, cov_agg,
    y_col_mean="cov_mean",
    y_col_std="cov_std",
    ylabel="Cov95",
    baselines=BASELINES,
    title_letter="B",
    show_xlabel=True,
)
ax_b.set_title("Cov95 (test_strict)", fontsize=9, pad=4)
# Nominal coverage reference line at 0.95
ax_b.axhline(0.95, color="#444444", lw=0.8, ls=":", zorder=0, alpha=0.6)
ax_b.text(0.02, 0.95, "nominal 0.95", transform=ax_b.get_yaxis_transform(),
          fontsize=8, color="#444444", va="bottom", ha="left", alpha=0.8)

# --- Manual layout adjustment: leave space at bottom for legend ---
fig.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.20)

# --- Shared legend (compact, 3 columns x ~2 rows, placed below panel B's xlabel) ---
handles, labels = ax_a.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.00),
    ncol=3,
    fontsize=8,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.0,
    handletextpad=0.5,
)

# Super-title removed — content conveyed by LaTeX caption to avoid overlap with panel titles.

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_pdf = OUT_DIR / "m2_4_efficiency_figure.pdf"
out_png = OUT_DIR / "m2_4_efficiency_figure.png"

fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")

print(f"Saved:\n  {out_pdf}\n  {out_png}")
plt.close(fig)
