"""Fig 3 generator — HHS region stratified evaluation (6 DL baselines).

Reads:
  runs/phase_3_region_eval.csv         (LSTM, Vanilla Mamba, PatchTST, CG-Mamba)
  runs/phase_3_region_eval_extras.csv  (DLinear, EpiDeep — produced by extras script)

Outputs:
  runs/phase_3_region_figure.pdf
  runs/phase_3_region_figure.png

Layout (1 row × 2 cols):
  Panel A: per-region tS_h1 MAE heatmap (10 regions × 6 baselines)
           color = greener lower MAE; per-region winner cell shown in bold.
  Panel B: cross-region distribution boxplot (50 obs each, 10 regions × 5 seeds)
           lower median + tighter IQR = better geographic robustness.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "runs"

# Display order (left to right): RNN → SSM → Transformer → linear → epi-DL → ours
BASELINES = ["lstm", "vanilla_mamba", "patchtst", "dlinear", "epideep", "cg_mamba"]
LABELS = {
    "lstm":          "LSTM",
    "vanilla_mamba": "Vanilla Mamba",
    "patchtst":      "PatchTST",
    "dlinear":       "DLinear",
    "epideep":       "EpiDeep",
    "cg_mamba":      "CG-Mamba (ours)",
}
COLORS_BOX = {
    "lstm":          "#f4a8b0",
    "vanilla_mamba": "#f4b370",
    "patchtst":      "#b3aee3",
    "dlinear":       "#ffd966",
    "epideep":       "#9fd6f5",
    "cg_mamba":      "#a8dfa8",
}
REGIONS = [f"hhs{i}" for i in range(1, 11)]
REGION_LABELS = [f"HHS{i}" for i in range(1, 11)]


def load_data() -> pd.DataFrame:
    main = pd.read_csv(_ROOT / "runs/phase_3_region_eval.csv")
    extras_path = _ROOT / "runs/phase_3_region_eval_extras.csv"
    if extras_path.exists():
        extras = pd.read_csv(extras_path)
        df = pd.concat([main, extras], ignore_index=True)
    else:
        print(f"WARNING: {extras_path} not found — Fig 3 will fall back to 4 baselines.")
        df = main
    df = df[df["baseline"].isin(BASELINES)].copy()
    if "error" in df.columns:
        df = df[df["error"].isna()] if df["error"].notna().any() else df
    return df


def per_region_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame: rows=region, cols=baseline. Values = 5-seed mean tS_h1."""
    pivot = df.pivot_table(index="region", columns="baseline", values="tS_h1", aggfunc="mean")
    pivot = pivot.reindex(index=REGIONS, columns=[b for b in BASELINES if b in pivot.columns])
    return pivot


def main_plot():
    df = load_data()
    pivot = per_region_mean(df)
    baselines_present = list(pivot.columns)
    n_baselines = len(baselines_present)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 5.6),
                                     gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ─── Panel A: heatmap ───
    values = pivot.values  # shape (10 regions, n_baselines)
    cmap = plt.get_cmap("RdYlGn_r")  # green-low, red-high
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    im = axA.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axA.set_xticks(range(n_baselines))
    axA.set_xticklabels([LABELS[b] for b in baselines_present], rotation=0, ha="center", fontsize=8.5)
    axA.set_yticks(range(len(REGIONS)))
    axA.set_yticklabels(REGION_LABELS, fontsize=10)

    # Bold the per-region winner; print MAE in each cell.
    winners = np.argmin(values, axis=1)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            is_winner = (j == winners[i])
            # Choose readable text color
            norm_v = (v - vmin) / (vmax - vmin + 1e-9)
            txt_color = "white" if 0.45 < norm_v < 0.75 or norm_v > 0.85 else "black"
            axA.text(j, i, f"{v:.3f}", ha="center", va="center",
                      fontsize=9,
                      fontweight="bold" if is_winner else "normal",
                      color=txt_color)
    axA.set_title("Per-region tS_h1 MAE (5-seed mean)\nLower = better (greener); per-region winner in bold",
                   fontsize=11, loc="center")
    cbar = plt.colorbar(im, ax=axA, fraction=0.04, pad=0.02)
    # label intentionally omitted; metric name already in panel title

    # ─── Panel B: cross-region boxplot ───
    box_data = []
    box_colors = []
    box_labels = []
    for b in baselines_present:
        vals = df[df["baseline"] == b]["tS_h1"].dropna().values
        box_data.append(vals)
        box_colors.append(COLORS_BOX[b])
        n_obs = len(vals)
        box_labels.append(f"{LABELS[b]}\n(n={n_obs})")

    bp = axB.boxplot(box_data, patch_artist=True, widths=0.55,
                      medianprops=dict(color="black", linewidth=1.5),
                      flierprops=dict(marker="o", markersize=4, markerfacecolor="none"))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axB.set_xticklabels(box_labels, rotation=0, ha="center", fontsize=8)
    # y-axis label intentionally omitted; metric name already in panel title
    axB.set_title("Cross-region distribution (10 regions × 5 seeds per baseline)\n"
                   "Lower median + tighter IQR = better geographic robustness",
                   fontsize=11, loc="center")
    axB.grid(alpha=0.3, axis="y")

    plt.suptitle("Figure 3 — HHS Region Stratified Evaluation (national-trained, regional inference)",
                  fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()

    pdf_path = OUT / "phase_3_region_figure.pdf"
    png_path = OUT / "phase_3_region_figure.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {pdf_path.relative_to(_ROOT)}")
    print(f"Saved: {png_path.relative_to(_ROOT)}")

    # Print diagnostic summary
    print("\n=== Per-baseline cross-region summary (tS_h1 MAE) ===")
    for b in baselines_present:
        vals = df[df["baseline"] == b]["tS_h1"].dropna().values
        if len(vals):
            print(f"  {LABELS[b]:<20} mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"min={vals.min():.3f}  max={vals.max():.3f}  n={len(vals)}")


if __name__ == "__main__":
    main_plot()
