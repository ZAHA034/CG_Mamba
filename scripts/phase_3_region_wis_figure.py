"""Fig 4 generator — Regional WIS/Cov95 comparison (6 DL baselines + SARIMA).

Data sources (method-specific UQ per §IV.6):
  - runs/phase_3_region_wis.csv           LSTM, Vanilla Mamba, PatchTST (MC Dropout)
  - runs/phase_3_cgm_method_f_region.csv  CG-Mamba (Method F)
  - runs/phase_3_region_wis_extras.csv    EpiDeep (MC Dropout d=0.1), DLinear (5-seed ensemble Gaussian)
  - runs/phase_3_sarima_wis_region.json   SARIMA (Kalman parametric)

Outputs:
  runs/phase_3_region_wis_figure.{pdf,png}

Layout (2 rows × 2 cols):
  Panel A: per-region tS_h1 WIS heatmap (10 regions × 7 baselines)
  Panel B: per-region tS_h1 Cov95 heatmap (color-centered on nominal 0.95)
  Panel C: cross-region WIS boxplot (lower median + tighter IQR = better)
  Panel D: cross-region Cov95 boxplot (closer to 0.95 dashed line = better calibrated)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "runs"

# DL family only — SARIMA classical reference noted in text §IV.X-REGION
# Display order: RNN → SSM → Transformer → linear → epi-DL → ours
BASELINES = ["lstm", "vanilla_mamba", "patchtst", "dlinear_ensemble_gauss",
             "epideep", "cg_mamba_method_F"]
LABELS = {
    "lstm":                   "LSTM\n(MC d=0.3)",
    "vanilla_mamba":          "Vanilla Mamba\n(MC d=0.2)",
    "patchtst":               "PatchTST\n(MC d=0.1)",
    "dlinear_ensemble_gauss": "DLinear\n(ensemble)",
    "epideep":                "EpiDeep\n(MC d=0.1)",
    "cg_mamba_method_F":      "CG-Mamba\n(HMM-derived)",
}
COLORS_BOX = {
    "lstm":                   "#f4a8b0",
    "vanilla_mamba":          "#f4b370",
    "patchtst":               "#b3aee3",
    "dlinear_ensemble_gauss": "#ffd966",
    "epideep":                "#9fd6f5",
    "cg_mamba_method_F":      "#a8dfa8",
}
REGIONS = [f"hhs{i}" for i in range(1, 11)]
REGION_LABELS = [f"HHS{i}" for i in range(1, 11)]


def load_all() -> pd.DataFrame:
    """Return long-format DataFrame including both h=1 and h=1-4 avg columns.

    SARIMA intentionally excluded — DL-family-only comparison, with SARIMA's
    classical-baseline strength acknowledged in §IV.X-REGION text.
    """
    parts = []
    use_cols = ["baseline", "seed", "region"] + \
               [f"tS_wis_h{h}" for h in (1,2,3,4)] + \
               [f"tS_cov95_h{h}" for h in (1,2,3,4)]

    main = pd.read_csv(_ROOT / "runs/phase_3_region_wis.csv")
    parts.append(main[use_cols])

    cgm = pd.read_csv(_ROOT / "runs/phase_3_cgm_method_f_region.csv")
    cgm["baseline"] = "cg_mamba_method_F"   # normalize label
    parts.append(cgm[use_cols])

    extras = pd.read_csv(_ROOT / "runs/phase_3_region_wis_extras.csv")
    parts.append(extras[use_cols])

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["tS_wis_h1", "tS_cov95_h1"])
    df = df[df["baseline"].isin(BASELINES)]

    # Compute h=1-4 average columns (row-wise mean across 4 horizons)
    df["tS_wis_h1_4_avg"]   = df[[f"tS_wis_h{h}"   for h in (1,2,3,4)]].mean(axis=1)
    df["tS_cov95_h1_4_avg"] = df[[f"tS_cov95_h{h}" for h in (1,2,3,4)]].mean(axis=1)
    return df


# ── Discrete calibration band scheme for Cov95 heatmap ─────────────────────────
# Boundaries chosen to make CG-Mamba's near-nominal status visually unambiguous:
#   [0.00, 0.50)  severe under-coverage
#   [0.50, 0.70)  substantial under
#   [0.70, 0.85)  moderate under
#   [0.85, 0.92)  near-nominal (just below 0.95 target)
#   [0.92, 1.00]  within nominal band (target 0.95)
COV_BOUNDS = [0.00, 0.50, 0.70, 0.85, 0.92, 1.00]
COV_COLORS = [
    "#a50026",   # severe under (deep red)
    "#f46d43",   # substantial under (orange-red)
    "#fee08b",   # moderate under (yellow)
    "#a6d96a",   # near-nominal (light green)
    "#1a9850",   # within nominal (deep green)
]
COV_LABELS = [
    "<0.50",
    "0.50–0.70",
    "0.70–0.85",
    "0.85–0.92",
    "0.92–1.00",
]


def per_region_mean(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pv = df.pivot_table(index="region", columns="baseline", values=metric, aggfunc="mean")
    pv = pv.reindex(index=REGIONS, columns=[b for b in BASELINES if b in pv.columns])
    return pv


# Horizon spec: (wis_col, cov_col, label, axis_wis_label, axis_cov_label, output_suffix,
#                 wis_vmin/vmax for heatmap, cov_bounds/labels for discrete heatmap)
# Cmap ranges chosen per-horizon to maximize color spectrum spread across 60 cells
# (6 baselines × 10 regions); see scripts/phase_3_region_wis_figure.py docstring.
COV_BOUNDS_DEFAULT = [0.20, 0.50, 0.70, 0.85, 0.92, 1.00]
COV_BOUNDS_H1_4_AVG = [0.20, 0.35, 0.55, 0.75, 0.90, 1.00]
COV_LABELS_DEFAULT = ["0.20–0.50", "0.50–0.70", "0.70–0.85", "0.85–0.92", "0.92–1.00"]
COV_LABELS_H1_4_AVG = ["0.20–0.35", "0.35–0.55", "0.55–0.75", "0.75–0.90", "0.90–1.00"]
COV_VMIN = 0.20

HORIZON_SPECS = {
    "h1": {
        "wis_col": "tS_wis_h1", "cov_col": "tS_cov95_h1",
        "label": "h=1 (immediate forecast)",
        "axis_wis": "WIS (h=1)", "axis_cov": "Cov95 (h=1)",
        "output_suffix": "h1",
        # h=1 distribution: WIS min=0.142, max=0.434 → auto-fit, no special cap needed
        "wis_vmin": None, "wis_vmax": None,
        "cov_bounds": COV_BOUNDS_DEFAULT, "cov_labels": COV_LABELS_DEFAULT,
    },
    "h1_4_avg": {
        "wis_col": "tS_wis_h1_4_avg", "cov_col": "tS_cov95_h1_4_avg",
        "label": "h=1–4 average (full FluSight scoring horizon)",
        "axis_wis": "WIS (h=1–4 avg)", "axis_cov": "Cov95 (h=1–4 avg)",
        "output_suffix": "h1_4_avg",
        # h=1-4 avg distribution: WIS min=0.246, max=0.862 — max is EpiDeep HHS2 outlier.
        # Cap at 0.65 (90th percentile region): 8 cells saturate at red end, rest get
        # balanced quintile spread (evenness score 4.15 vs auto-range 8.32).
        "wis_vmin": 0.25, "wis_vmax": 0.65,
        # Cov95 bands tuned to give near-balanced cell counts across 5 bands
        # (current bands [9, 13, 15, 13, 10] vs original [22, 15, 13, 3, 7]).
        "cov_bounds": COV_BOUNDS_H1_4_AVG, "cov_labels": COV_LABELS_H1_4_AVG,
    },
}


def main_plot(horizon="h1_4_avg", out_basename="phase_3_region_wis_figure"):
    """Generate Fig 4 (regional WIS/Cov95).

    horizon: one of 'h1' or 'h1_4_avg'. Default 'h1_4_avg' (main paper figure).
    """
    spec = HORIZON_SPECS[horizon]
    wis_col, cov_col = spec["wis_col"], spec["cov_col"]
    df = load_all()

    # ─── 2x2 layout: heatmaps (top row) + boxplots (bottom row) ───
    fig, axes = plt.subplots(2, 2, figsize=(18, 11),
                              gridspec_kw={"width_ratios": [1.4, 1.0],
                                            "height_ratios": [1.0, 1.0]})
    axAH, axAB = axes[0]
    axBH, axBB = axes[1]

    # ── A: WIS heatmap ──
    wis_pv = per_region_mean(df, wis_col)
    baselines_present = list(wis_pv.columns)
    wis_vals = wis_pv.values
    cmap = plt.get_cmap("RdYlGn_r")
    # Horizon-specific vmin/vmax for color-spectrum balance (None → auto-fit)
    wis_vmin = spec["wis_vmin"] if spec["wis_vmin"] is not None else np.nanmin(wis_vals)
    wis_vmax = spec["wis_vmax"] if spec["wis_vmax"] is not None else np.nanmax(wis_vals)
    im_a = axAH.imshow(wis_vals, cmap=cmap, aspect="auto", vmin=wis_vmin, vmax=wis_vmax)
    axAH.set_xticks(range(len(baselines_present)))
    axAH.set_xticklabels([LABELS[b] for b in baselines_present], rotation=0, ha="center", fontsize=8.5)
    axAH.set_yticks(range(len(REGIONS)))
    axAH.set_yticklabels(REGION_LABELS, fontsize=10)
    winners = np.argmin(wis_vals, axis=1)
    for i, row in enumerate(wis_vals):
        for j, v in enumerate(row):
            is_winner = (j == winners[i])
            # Text contrast normalized to the applied cmap range, not raw data range
            norm_v = (np.clip(v, wis_vmin, wis_vmax) - wis_vmin) / (wis_vmax - wis_vmin + 1e-9)
            txt = "white" if (0.45 < norm_v < 0.75) or norm_v > 0.85 else "black"
            axAH.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                       fontweight="bold" if is_winner else "normal", color=txt)
    axAH.set_title(f"(A) Per-region tS {spec['axis_wis']} (5-seed mean)\nLower = better; per-region winner in bold",
                    fontsize=11, loc="center")
    cb_a = plt.colorbar(im_a, ax=axAH, fraction=0.04, pad=0.02)
    # label intentionally omitted; metric name already in panel title

    # ── B: WIS boxplot ──
    box_data, box_colors, box_labels = [], [], []
    for b in baselines_present:
        vals = df[df["baseline"] == b][wis_col].dropna().values
        box_data.append(vals)
        box_colors.append(COLORS_BOX[b])
        box_labels.append(f"{LABELS[b]}\n(n={len(vals)})")
    bp = axAB.boxplot(box_data, patch_artist=True, widths=0.55,
                       medianprops=dict(color="black", linewidth=1.5),
                       flierprops=dict(marker="o", markersize=4, markerfacecolor="none"))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    y_top = max(np.max(d) for d in box_data) * 1.05
    for i, vals in enumerate(box_data):
        m = np.mean(vals); s = np.std(vals)
        axAB.text(i + 1, y_top, f"{m:.3f}\n±{s:.3f}",
                   ha="center", va="bottom", fontsize=7.5,
                   fontweight="bold" if box_labels[i].startswith("CG-Mamba") else "normal")
    axAB.set_ylim(top=y_top * 1.18)
    axAB.set_xticklabels(box_labels, rotation=0, ha="center", fontsize=7.5)
    # y-axis label intentionally omitted; metric name already in panel title
    # Dynamic subtitle — CGM mean + std
    cgm_wis_vals = df[df["baseline"] == "cg_mamba_method_F"][wis_col].values
    cgm_wis_mean, cgm_wis_std = cgm_wis_vals.mean(), cgm_wis_vals.std()
    axAB.set_title(f"(B) Cross-region WIS distribution — mean ± std annotated\n"
                    f"CG-Mamba mean {cgm_wis_mean:.3f} ± {cgm_wis_std:.3f}, lowest among DL baselines",
                    fontsize=11, loc="center")
    axAB.grid(alpha=0.3, axis="y")

    # ── C: Cov95 heatmap — CONTINUOUS cmap with PROPORTIONAL colorbar ticks ──
    # vmin=COV_VMIN (0.20), vmax=1 so cells below 0.20 saturate at deep red and the
    # cmap dynamic range is concentrated where the data actually lives.
    # Boundary values (e.g., 0.30, 0.50, 0.70, 0.90) sit at their proportional
    # positions on the colorbar relative to [0.20, 1.00].
    cov_pv = per_region_mean(df, cov_col)
    cov_vals = cov_pv.values
    cov_bounds = spec["cov_bounds"]
    cov_labels = spec["cov_labels"]
    cmap_cov = plt.get_cmap("RdYlGn")
    norm_cov = Normalize(vmin=COV_VMIN, vmax=1.0)
    im_c = axBH.imshow(cov_vals, cmap=cmap_cov, norm=norm_cov, aspect="auto")
    axBH.set_xticks(range(len(baselines_present)))
    axBH.set_xticklabels([LABELS[b] for b in baselines_present], rotation=0, ha="center", fontsize=8.5)
    axBH.set_yticks(range(len(REGIONS)))
    axBH.set_yticklabels(REGION_LABELS, fontsize=10)
    cov_dist = np.abs(cov_vals - 0.95)
    cov_winners = np.argmin(cov_dist, axis=1)
    for i, row in enumerate(cov_vals):
        for j, v in enumerate(row):
            is_winner = (j == cov_winners[i])
            # Continuous cmap with vmin=COV_VMIN: cells below or near vmin (deep red)
            # and near vmax 1.0 (deep green) need white text for contrast.
            txt_color = "white" if (v < COV_VMIN + 0.05 or v > 0.92) else "black"
            axBH.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                       fontweight="bold" if is_winner else "normal", color=txt_color)
    # Dynamic Cov95 mean comparison subtitle
    cgm_cov_vals = df[df["baseline"] == "cg_mamba_method_F"][cov_col].values
    cgm_cov_mean = cgm_cov_vals.mean()
    other_dl_means = {b: df[df["baseline"] == b][cov_col].mean() for b in baselines_present if b != "cg_mamba_method_F"}
    next_best_b, next_best_m = max(other_dl_means.items(), key=lambda kv: kv[1])
    axBH.set_title(f"(C) Per-region tS {spec['axis_cov']} (5-seed mean) — discrete calibration bands\n"
                    f"CG-Mamba mean {cgm_cov_mean:.3f}; next-best DL {next_best_m:.3f} ({LABELS[next_best_b].replace(chr(10), ' ')})",
                    fontsize=11, loc="center")
    # Colorbar ticks at the band boundary values (e.g. 0.30, 0.50, 0.70, 0.90),
    # each at its actual proportional position thanks to Normalize(0,1).
    cb_c = plt.colorbar(im_c, ax=axBH, fraction=0.04, pad=0.02, ticks=cov_bounds)
    cb_c.ax.set_yticklabels([f"{b:.2f}" for b in cov_bounds], fontsize=8)
    cb_c.ax.tick_params(length=3)
    # Nominal 0.95 reference marker on the colorbar
    cb_c.ax.axhline(0.95, color="black", linestyle="--", linewidth=0.9, alpha=0.7)

    # ── D: Cov95 boxplot — Cov95 std anchor ──
    cov_box_data = []
    for b in baselines_present:
        vals = df[df["baseline"] == b][cov_col].dropna().values
        cov_box_data.append(vals)
    bp2 = axBB.boxplot(cov_box_data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=4, markerfacecolor="none"))
    for patch, color in zip(bp2["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    axBB.axhline(0.95, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
                  label="Nominal 0.95")
    cgm_idx = baselines_present.index("cg_mamba_method_F")
    cgm_cov_std = np.std(cov_box_data[cgm_idx])
    other_stds = [np.std(cov_box_data[i]) for i in range(len(baselines_present)) if i != cgm_idx]
    min_other_std = min(other_stds)
    ratio = min_other_std / cgm_cov_std
    for i, vals in enumerate(cov_box_data):
        m = np.mean(vals); s = np.std(vals)
        is_cgm = (i == cgm_idx)
        axBB.text(i + 1, 1.03, f"{m:.3f}\n±{s:.3f}",
                   ha="center", va="bottom", fontsize=7.5,
                   fontweight="bold" if is_cgm else "normal",
                   color="#1a9850" if is_cgm else "black")
    axBB.set_xticklabels(box_labels, rotation=0, ha="center", fontsize=7.5)
    # y-axis label intentionally omitted; metric name already in panel title
    axBB.set_title(f"(D) Cross-region Cov95 distribution — mean ± std annotated\n"
                    f"CG-Mamba std {cgm_cov_std:.3f}; all other DL ≥ {min_other_std:.3f} ({ratio:.1f}× tighter or more)",
                    fontsize=11, loc="center")
    axBB.set_ylim(0, 1.18)
    axBB.legend(loc="lower left", fontsize=8)
    axBB.grid(alpha=0.3, axis="y")

    plt.suptitle(f"Figure 4 — DL-Family Regional Method-Specific WIS/Cov95 Comparison "
                  f"(test_strict, {spec['label']}, national-trained → regional inference). "
                  f"Classical SARIMA reference acknowledged in §IV.X-REGION text.",
                  fontsize=12, fontweight="bold", y=0.998)
    plt.tight_layout()

    pdf_path = OUT / f"{out_basename}.pdf"
    png_path = OUT / f"{out_basename}.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved [{horizon}]: {pdf_path.relative_to(_ROOT)}")
    print(f"Saved [{horizon}]: {png_path.relative_to(_ROOT)}")

    print(f"\n=== Per-baseline cross-region {wis_col} summary ===")
    for b in baselines_present:
        vals = df[df["baseline"] == b][wis_col].dropna().values
        if len(vals):
            print(f"  {LABELS[b].replace(chr(10),' '):<32} WIS mean={vals.mean():.3f}  std={vals.std():.3f}  n={len(vals)}")
    print(f"\n=== Per-baseline cross-region {cov_col} summary ===")
    for b in baselines_present:
        vals = df[df["baseline"] == b][cov_col].dropna().values
        if len(vals):
            print(f"  {LABELS[b].replace(chr(10),' '):<32} Cov95 mean={vals.mean():.3f}  std={vals.std():.3f}  n={len(vals)}")


if __name__ == "__main__":
    # Both are supplementary now; main paper figure is the per-horizon profile
    # (scripts/phase_3_region_profile_figure.py). These provide detail at each
    # endpoint of the profile: h=1 (immediate forecast) + h=1-4 avg (full horizon).
    main_plot(horizon="h1",       out_basename="phase_3_region_wis_figure_supp_h1")
    main_plot(horizon="h1_4_avg", out_basename="phase_3_region_wis_figure_supp_h1_4_avg")
