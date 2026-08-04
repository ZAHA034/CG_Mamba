"""Paper draft figure candidates — preview generation for user review.

Output: runs/figures/paper_drafts/
- C_alt1_reliability.pdf  (re-uses existing method_f_reliability_figure.pdf reference)
- C_alt2_pareto.pdf       (Sharpness-Calibration Pareto from Table I)
- D_alt1_top10.pdf        (Top-10 FluSight 2018-19 + CG-Mamba horizontal bar)
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "figures" / "paper_drafts"
OUT.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Figure C-Alt2: Sharpness-Calibration Pareto (national, Table I data)
# =====================================================================
# Data verified from latex_submission/tex/results_main.tex Table I
methods = [
    # (name, WIS, Cov95, marker, color, ms)
    ("SARIMA",          0.218, 0.888, "D",  "#2ca02c", 90),
    ("CG-Mamba (APMD)", 0.290, 0.889, "*",  "#1f77b4", 320),
    ("Vanilla Mamba",   0.372, 0.370, "^",  "#ff7f0e", 80),
    ("LSTM",            0.466, 0.342, "s",  "#ff7f0e", 80),
    ("PatchTST",        0.368, 0.698, "v",  "#9467bd", 80),
    ("EpiDeep",         0.394, 0.377, "P",  "#8c564b", 80),
    ("DLinear",         0.441, 0.289, "X",  "#9467bd", 80),
    ("iTransformer",    0.521, 0.270, "h",  "#9467bd", 80),
    ("N-BEATS",         0.487, 0.272, "p",  "#9467bd", 80),
    ("TimesNet",        0.597, 0.225, "<",  "#9467bd", 80),
]

fig, ax = plt.subplots(figsize=(5.2, 4.0))
for name, wis, cov, marker, color, size in methods:
    edgecolor = "black" if "CG-Mamba" in name else color
    lw = 1.5 if "CG-Mamba" in name else 0.5
    ax.scatter(wis, cov, marker=marker, color=color, s=size,
               edgecolors=edgecolor, linewidths=lw, zorder=5, label=name)
    # Annotate with team name
    offset_y = 0.04 if "CG-Mamba" in name else 0.02
    if "CG-Mamba" in name:
        ax.annotate(name, (wis, cov), xytext=(8, 8), textcoords="offset points",
                    fontsize=8, fontweight="bold", color="#1f77b4")
    elif name in ("SARIMA", "PatchTST"):
        ax.annotate(name, (wis, cov), xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color="dimgrey")

# Nominal coverage reference
ax.axhline(0.95, color="grey", lw=0.8, ls="--", alpha=0.7)
ax.text(0.59, 0.96, "nominal 0.95", fontsize=7, color="grey", ha="right")

# Sweet spot annotation (top-left)
ax.annotate("Pareto-optimal\n(sharp + calibrated)",
            xy=(0.218, 0.888), xytext=(0.32, 0.75),
            fontsize=7.5, color="dimgrey",
            arrowprops=dict(arrowstyle="->", color="dimgrey", lw=0.6))

ax.set_xlabel("Method-specific WIS (lower is better)", fontsize=9)
ax.set_ylabel("Method-specific Cov95 (higher is better)", fontsize=9)
ax.set_title("Sharpness-Calibration Trade-off (national, test_strict)", fontsize=10)
ax.legend(loc="lower right", fontsize=6.5, ncol=1, frameon=True)
ax.grid(alpha=0.3)
ax.set_xlim(0.18, 0.65)
ax.set_ylim(0.18, 1.05)
ax.tick_params(labelsize=8)

fig.savefig(OUT / "C_alt2_pareto.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT / "C_alt2_pareto.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {OUT}/C_alt2_pareto.{{pdf,png}}")


# =====================================================================
# Figure D-Alt1: Top-10 FluSight 2018-19 + CG-Mamba
# =====================================================================
team_df = pd.read_csv(ROOT / "runs/phase_5_flusight/team_wis_ranking_2018_2019.csv")
# Take top 22 to include EpiDeep (rank 22, the only documented prior DL submission)
top22 = team_df.nsmallest(22, "wis_mean").copy()

# Insert CG-Mamba (WIS 0.233, held-out checkpoint)
cg_row = pd.DataFrame({
    "team": ["CG-Mamba"],
    "wis_mean": [0.233],
    "cov95_mean": [0.860],
    "n_obs": [149],
    "rank": [4.5],
})
combined = pd.concat([top22, cg_row], ignore_index=True)
combined = combined.sort_values("wis_mean").reset_index(drop=True)
combined["new_rank"] = range(1, len(combined) + 1)

# Methodology categories — VERIFIED FluSight teams only (paper §IV.F.1 + Intro)
# SARIMA removed (not a FluSight team — internal reference only)
# ARIMA category dropped (single team Protea-Springbok grouped as Unclassified for compactness)
DL_BLUE = "#1f77b4"
methodology = {
    "LANL-DBMplus":      ("Bayesian dynamic", "#d62728"),  # paper §IV.F.1
    "LANL-Dante":        ("Mechanistic",      "#ff7f0e"),  # paper Intro
    "FluSightNetwork":   ("Ensemble",         "#9467bd"),  # paper §IV.F.1
    "PSI-s":             ("Mechanistic",      "#ff7f0e"),  # paper §IV.F.1
    "CG-Mamba":          ("DL",               DL_BLUE),    # ours
    "EpiDeep":           ("DL",               DL_BLUE),    # paper §IV.F.1
    "KoT":               ("Unspecified",      "#7f7f7f"),  # paper §IV.F.1
}

colors = [methodology.get(t, ("Unclassified", "#cccccc"))[1] for t in combined["team"]]

fig, ax = plt.subplots(figsize=(6.5, 8.5))
y_pos = list(range(len(combined)))
bars = ax.barh(y_pos, combined["wis_mean"], color=colors,
               edgecolor="black", lw=0.5)

# Highlight CG-Mamba + EpiDeep (DL family — paper signature comparison)
cg_idx = combined.index[combined["team"] == "CG-Mamba"].tolist()[0]
epi_idx = combined.index[combined["team"] == "EpiDeep"].tolist()[0]

bars[cg_idx].set_edgecolor(DL_BLUE)
bars[cg_idx].set_linewidth(2.5)
bars[epi_idx].set_edgecolor(DL_BLUE)
bars[epi_idx].set_linewidth(2.0)
bars[epi_idx].set_hatch("//")  # distinguish from CG-Mamba via hatching

ax.set_yticks(y_pos)
labels = []
for t in combined["team"]:
    if t == "CG-Mamba":
        labels.append(f"{t} (Ours)")
    elif t == "EpiDeep":
        labels.append(f"{t} (prior DL)")
    else:
        labels.append(t)
ax.set_yticklabels(labels, fontsize=9)
for label in ax.get_yticklabels():
    if "Ours" in label.get_text() or "prior DL" in label.get_text():
        label.set_fontweight("bold")

ax.set_xlabel("WIS (test_strict mean, 2018-2019 FluSight)", fontsize=10)
ax.invert_yaxis()


# WIS value annotations
for i, val in enumerate(combined["wis_mean"]):
    ax.text(val + 0.005, i, f"{val:.3f}", fontsize=7.5, va="center", color="dimgrey")

# CG-Mamba WIS reference (vertical blue dashed line — DL-internal comparison)
ax.axvline(0.233, color=DL_BLUE, lw=1.0, ls="--", alpha=0.7)
ax.text(0.233, -0.7, "CG-Mamba WIS 0.233", fontsize=7.5, color=DL_BLUE,
        ha="center", va="bottom", fontweight="bold")

ax.tick_params(labelsize=8)
ax.set_xlim(0, 0.42)
ax.grid(axis="x", alpha=0.3)

# Legend BELOW figure (horizontal)
categories_in_plot = []
for t in combined["team"]:
    cat, color = methodology.get(t, ("Unclassified", "#cccccc"))
    if (cat, color) not in categories_in_plot:
        categories_in_plot.append((cat, color))
patches = [mpatches.Patch(color=color, label=cat) for cat, color in categories_in_plot]
fig.legend(handles=patches,
           loc="lower center",
           bbox_to_anchor=(0.5, -0.02),
           ncol=len(patches),
           fontsize=8,
           frameon=True,
           title="Methodology",
           title_fontsize=8.5)

fig.savefig(OUT / "D_alt1_top22.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT / "D_alt1_top22.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {OUT}/D_alt1_top22.{{pdf,png}}")


# =====================================================================
# Figure C-Alt1: Reliability Diagram — re-uses existing data
# =====================================================================
# Existing figure already generated: runs/figures/method_f_reliability_figure.pdf
# It contains Panel A (Method F decomposition) + Panel B (Reliability)
# For paper Figure C-Alt1, we want Panel B standalone.
# Generate standalone version here:

import json
import numpy as np
from scipy.stats import norm

FLUSIGHT_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99
])

# Try to load Method F (HMM-derived) reliability data
decomp_path = ROOT / "runs/wis_method_f/decomposition_temporal.csv"
if decomp_path.exists():
    df = pd.read_csv(decomp_path)
    if all(c in df.columns for c in ["mu_CGM_raw", "sigma2_total", "y_raw"]):
        mu = df["mu_CGM_raw"].values
        sigma_total = np.sqrt(df["sigma2_total"].values)
        y_true = df["y_raw"].values

        # Compute empirical coverage at each quantile level
        emp_cov = np.zeros(len(FLUSIGHT_QUANTILES))
        for i, alpha in enumerate(FLUSIGHT_QUANTILES):
            z = norm.ppf(alpha)
            q = mu + z * sigma_total
            emp_cov[i] = (y_true <= q).mean()

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
        # CG-Mamba APMD
        ax.plot(FLUSIGHT_QUANTILES, emp_cov, "o-", color="#1f77b4",
                lw=2, markersize=5, label="CG-Mamba APMD (HMM-derived)")
        # Reference dots: SARIMA & MC Dropout at 0.95
        ax.scatter([0.95], [0.888], marker="D", color="#2ca02c", s=100, zorder=5,
                   edgecolors="black", linewidths=0.8, label="SARIMA @ Cov95")
        ax.scatter([0.95], [0.342], marker="s", color="#ff7f0e", s=100, zorder=5,
                   edgecolors="black", linewidths=0.8, label="LSTM MC Dropout @ Cov95")
        ax.scatter([0.95], [0.236], marker="x", color="#d62728", s=120, zorder=5,
                   linewidths=2.5, label="CG-Mamba w/ MC Dropout @ Cov95 (FM-2)")

        # Annotate under/over coverage zones
        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.06, color="green", label=None)
        ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.06, color="red", label=None)
        ax.text(0.78, 0.20, "under-coverage", fontsize=7, color="darkred", style="italic")
        ax.text(0.20, 0.85, "over-coverage", fontsize=7, color="darkgreen", style="italic")

        ax.set_xlabel("Nominal quantile level", fontsize=9)
        ax.set_ylabel("Empirical coverage", fontsize=9)
        ax.set_title("Calibration Reliability (test_strict)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=7, frameon=True)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

        fig.savefig(OUT / "C_alt1_reliability.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(OUT / "C_alt1_reliability.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {OUT}/C_alt1_reliability.{{pdf,png}}")
    else:
        print(f"NOTE: {decomp_path} missing columns mu/sigma_total/y_true")
        print(f"  Existing reference: runs/figures/method_f_reliability_figure.pdf (Panel B)")
else:
    print(f"NOTE: {decomp_path} not found")
    print(f"  Existing reference: runs/figures/method_f_reliability_figure.pdf (Panel B)")

# =====================================================================
# Figure APMD-Decomposition: σ²_within / σ²_between / bias² over time
# (Clean standalone version — replaces C-Alt1 Reliability)
# =====================================================================
import matplotlib.dates as mdates

decomp_path = ROOT / "runs/wis_method_f/decomposition_temporal.csv"
df_d = pd.read_csv(decomp_path)
df_h1 = df_d[df_d.horizon == 1].sort_values("target_ep").reset_index(drop=True)

def epiweek_to_date(ep: int) -> pd.Timestamp:
    year = ep // 100
    week = ep % 100
    jan4 = pd.Timestamp(year=year, month=1, day=4)
    iso_week_1_start = jan4 - pd.Timedelta(days=jan4.isoweekday() % 7)
    return iso_week_1_start + pd.Timedelta(weeks=week - 1)

df_h1["date"] = df_h1.target_ep.apply(epiweek_to_date)

fig, ax = plt.subplots(figsize=(6.5, 3.8))

# Stacked variance components
ax.fill_between(df_h1.date, 0, df_h1.sigma2_within,
                color="#2ca02c", alpha=0.65,
                label=r"$\sigma^2_{\mathrm{within}}$ (aleatoric, per-phase noise)")
ax.fill_between(df_h1.date, df_h1.sigma2_within,
                df_h1.sigma2_within + df_h1.sigma2_between_HMM,
                color="#ff7f0e", alpha=0.65,
                label=r"$\sigma^2_{\mathrm{between}}$ (phase identifiability)")
ax.fill_between(df_h1.date,
                df_h1.sigma2_within + df_h1.sigma2_between_HMM,
                df_h1.sigma2_within + df_h1.sigma2_between_HMM + df_h1.bias_sq,
                color="#d62728", alpha=0.55,
                label=r"$\mathrm{bias}^2$ (model refinement beyond HMM mean)")

ax.set_ylabel(r"Variance contribution ($z$-scored$^2$)", fontsize=9.5)
ax.set_xlabel("Date (test_strict period, $h=1$ forecast)", fontsize=9.5)
ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))
ax.tick_params(axis="x", which="major", labelsize=8.5)
ax.tick_params(axis="y", labelsize=8.5)

# Secondary axis: actual wILI overlay
ax_r = ax.twinx()
ax_r.plot(df_h1.date, df_h1.y_raw, "k-", linewidth=1.0, alpha=0.6,
          label="actual %wILI")
ax_r.set_ylabel("Actual %wILI", fontsize=9.5, color="dimgrey")
ax_r.tick_params(axis="y", colors="dimgrey", labelsize=8.5)
ax_r.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95)

ax.set_title("APMD three-component decomposition over test_strict period",
              fontsize=10.5, loc="left")

fig.savefig(OUT / "APMD_decomposition.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT / "APMD_decomposition.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {OUT}/APMD_decomposition.{{pdf,png}}")


print("\nAll figures generated. Review at:")
print(f"  {OUT}")
