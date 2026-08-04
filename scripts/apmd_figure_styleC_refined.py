"""APMD architecture figure — Style C refined (paper-quality).

Design principles inspired by Transformer/U-Net/DETR architecture diagrams:
- Clean left-to-right pipeline with NO arrow overlap
- K=3 phase mini-graph ABOVE main pipeline (visible aggregation)
- bias² terminator BELOW main pipeline (clear separation)
- μ_CGM skip arc at BOTTOM (fully visible, dashed blue)
- Math labels on inter-box arrows (not inside boxes)
- Uniform box heights, minimal color palette

Output: runs/figures/paper_drafts/apmd_arch_styleC_refined.{pdf,png}
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

ROOT = Path("/A.I_DATA/jbnu/JeongHa/CG_Mamba")
OUT = ROOT / "runs" / "figures" / "paper_drafts"
OUT.mkdir(parents=True, exist_ok=True)

# Colors (minimal palette inspired by Transformer/U-Net)
COLOR_INPUT_HMM = "#dde7f0"   # very light blue (HMM stream)
COLOR_INPUT_DL  = "#fde6cc"   # very light orange (DL stream)
COLOR_OP        = "#f2f0e8"   # neutral light grey (operations)
COLOR_BIAS      = "#fadcdc"   # very light pink (bias² stream)
COLOR_OUTPUT    = "#e1ebd3"   # very light green (output)
COLOR_PHASE     = "#e8e0f0"   # very light purple (K=3 phases)
EDGE_DARK       = "#333333"
EDGE_LIGHT      = "#666666"
DL_BLUE         = "#1f6faf"
BIAS_RED        = "#a04040"

# Figure: wider, more breathing room
fig, ax = plt.subplots(figsize=(9.0, 4.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 55)
ax.axis("off")

# ----------------------------------------------------------------------
# Main pipeline: 5 boxes left-to-right
# ----------------------------------------------------------------------
main_y = 23
main_h = 12

main_boxes = [
    {"x": 2,   "w": 16, "title": "INPUTS",     "color": COLOR_INPUT_HMM,
     "body": r"$\gamma_T$, $\mu_k$, $\sigma^2_k$" + "\n(HMM, K=3, frozen)\n" + r"$\mu_{\mathrm{CGM}}$ (DL)"},
    {"x": 24,  "w": 19, "title": "DECOMPOSE",  "color": COLOR_OP,
     "body": r"$\sigma^2_{\mathrm{within}}\!=\!\sum_k\gamma_k\sigma^2_k$" + "\n" +
             r"$\sigma^2_{\mathrm{between}}\!=\!\sum_k\gamma_k(\mu_k\!-\!\bar\mu)^2$"},
    {"x": 49,  "w": 14, "title": "COMBINE",    "color": COLOR_OP,
     "body": r"$\sigma^2_{\mathrm{total}}=$" + "\n" +
             r"$\sigma^2_{\mathrm{within}}+\sigma^2_{\mathrm{between}}$"},
    {"x": 69,  "w": 12, "title": "CALIBRATE",  "color": COLOR_OP,
     "body": r"$s_h\cdot\sigma^2_{\mathrm{total}}$" + "\n" + r"($s_h$ val-grid)"},
    {"x": 87,  "w": 11, "title": "OUTPUT",     "color": COLOR_OUTPUT,
     "body": r"$q_\alpha\!=\!\mu_{\mathrm{CGM}}\!+\!z_\alpha\sqrt{s_h\sigma^2_{\mathrm{tot}}}$" +
             "\n23 FluSight quantiles"},
]

box_centers = []
for p in main_boxes:
    box = FancyBboxPatch((p["x"], main_y), p["w"], main_h,
                          boxstyle="round,pad=0.4,rounding_size=0.6",
                          facecolor=p["color"], edgecolor=EDGE_DARK, linewidth=1.0)
    ax.add_patch(box)
    cx = p["x"] + p["w"] / 2
    cy = main_y + main_h
    ax.text(cx, cy - 2.5, p["title"], ha="center", va="top",
            fontsize=10, fontweight="bold", color=EDGE_DARK)
    ax.text(cx, cy - 5, p["body"], ha="center", va="top",
            fontsize=8.0, color=EDGE_DARK)
    box_centers.append((p["x"], p["x"] + p["w"], cy - main_h/2))

# Arrows between main boxes — VISIBLE with math labels
arrow_labels = [
    r"$\gamma_T,\mu_k,\sigma^2_k$",
    r"$\sigma^2_w,\sigma^2_b$",
    r"$\sigma^2_{\mathrm{total}}$",
    r"$s_h\sigma^2_{\mathrm{total}}$",
]
for i in range(len(box_centers) - 1):
    src_x = box_centers[i][1]      # right edge of source
    dst_x = box_centers[i+1][0]    # left edge of dest
    y = box_centers[i][2]
    arr = FancyArrowPatch((src_x + 0.3, y), (dst_x - 0.3, y),
                            arrowstyle="-|>", mutation_scale=12,
                            color=EDGE_DARK, linewidth=1.3)
    ax.add_patch(arr)
    # Label above arrow
    mid_x = (src_x + dst_x) / 2
    ax.text(mid_x, y + 1.0, arrow_labels[i], ha="center", va="bottom",
            fontsize=7.5, color=EDGE_LIGHT, style="italic")

# ----------------------------------------------------------------------
# K=3 phase mini-graph ABOVE DECOMPOSE (visible aggregation)
# ----------------------------------------------------------------------
phase_y = main_y + main_h + 4.5
phase_h = 4
decompose_x = main_boxes[1]["x"]
decompose_w = main_boxes[1]["w"]

phase_x_starts = [decompose_x + 1.0, decompose_x + 7.5, decompose_x + 14]
for i, px in enumerate(phase_x_starts):
    box = FancyBboxPatch((px, phase_y), 4, phase_h,
                          boxstyle="round,pad=0.2,rounding_size=0.3",
                          facecolor=COLOR_PHASE, edgecolor=EDGE_DARK, linewidth=0.8)
    ax.add_patch(box)
    ax.text(px + 2, phase_y + phase_h/2, f"$k={i+1}$",
            ha="center", va="center", fontsize=8.5, color=EDGE_DARK)
    # Arrow down into DECOMPOSE
    arr = FancyArrowPatch((px + 2, phase_y), (px + 2, main_y + main_h + 0.3),
                          arrowstyle="-|>", mutation_scale=8,
                          color=EDGE_LIGHT, linewidth=0.9, linestyle=":")
    ax.add_patch(arr)

# Σγ_k(·)=1 simplex constraint annotation above phases
ax.text(decompose_x + decompose_w/2, phase_y + phase_h + 1.5,
        r"K=3 HMM phases (frozen)  $\sum_k\gamma_k=1$",
        ha="center", va="bottom", fontsize=8, fontstyle="italic", color=EDGE_LIGHT)

# ----------------------------------------------------------------------
# bias² terminator BELOW DECOMPOSE (clear separation, report-only)
# ----------------------------------------------------------------------
bias_y = main_y - 8
bias_x = decompose_x + 2
bias_w = decompose_w - 4
bias_h = 4

bias_box = FancyBboxPatch((bias_x, bias_y), bias_w, bias_h,
                           boxstyle="round,pad=0.3,rounding_size=0.4",
                           facecolor=COLOR_BIAS, edgecolor=BIAS_RED,
                           linewidth=1.0, linestyle="--")
ax.add_patch(bias_box)
ax.text(bias_x + bias_w/2, bias_y + bias_h/2,
        r"$\mathrm{bias}^2=(\bar\mu-\mu_{\mathrm{CGM}})^2$  (interpretability triple — report only, $\notin\sigma^2_{\mathrm{total}}$)",
        ha="center", va="center", fontsize=8, color=BIAS_RED)

# Arrow down from DECOMPOSE to bias²
arr = FancyArrowPatch((decompose_x + decompose_w/2, main_y - 0.3),
                      (decompose_x + decompose_w/2, bias_y + bias_h + 0.3),
                      arrowstyle="-|>", mutation_scale=10,
                      color=BIAS_RED, linewidth=1.0, linestyle="--")
ax.add_patch(arr)
ax.text(decompose_x + decompose_w/2 + 1.5, (main_y + bias_y + bias_h)/2,
        r"$\mathrm{bias}^2$", fontsize=7.5, color=BIAS_RED, style="italic")

# ----------------------------------------------------------------------
# μ_CGM skip arc at BOTTOM (fully visible, dashed blue)
# ----------------------------------------------------------------------
skip_y = 6
output_x = main_boxes[-1]["x"]
output_w = main_boxes[-1]["w"]
input_x = main_boxes[0]["x"]
input_w = main_boxes[0]["w"]

# Vertical drop from INPUTS to skip level
ax.plot([input_x + input_w/2, input_x + input_w/2],
        [main_y, skip_y], color=DL_BLUE, lw=1.4, ls="--", alpha=0.85)
# Horizontal arc
ax.plot([input_x + input_w/2, output_x + output_w/2],
        [skip_y, skip_y], color=DL_BLUE, lw=1.4, ls="--", alpha=0.85)
# Vertical rise to OUTPUT (with arrowhead)
arr = FancyArrowPatch((output_x + output_w/2, skip_y),
                      (output_x + output_w/2, main_y - 0.3),
                      arrowstyle="-|>", mutation_scale=12,
                      color=DL_BLUE, linewidth=1.4, linestyle="--")
ax.add_patch(arr)

# Label on the horizontal arc
ax.text((input_x + input_w + output_x)/2 + 5, skip_y - 1.8,
        r"$\mu_{\mathrm{CGM}}$  (DL skip — location parameter of $q_\alpha$)",
        ha="center", va="top", fontsize=8.5, color=DL_BLUE, fontweight="bold",
        style="italic")

# ----------------------------------------------------------------------
# Top title bar (subtle, gives context)
# ----------------------------------------------------------------------
ax.text(50, 52, "APMD: HMM-derived calibrated intervals with decomposable interpretability",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1a4d7a")

# ----------------------------------------------------------------------
# Legend (bottom-right corner, compact)
# ----------------------------------------------------------------------
legend_x = 70
legend_y = 1.5
ax.add_patch(Rectangle((legend_x, legend_y), 28, 4.5,
                        facecolor="white", edgecolor=EDGE_LIGHT, linewidth=0.5))
ax.text(legend_x + 1, legend_y + 3.5, "→ data flow", fontsize=7, color=EDGE_DARK)
ax.text(legend_x + 1, legend_y + 2.3, "⇢ skip / report-only", fontsize=7, color=EDGE_DARK)
ax.text(legend_x + 1, legend_y + 1.1,
        r"$\gamma_k$ : HMM phase weight  ·  $\bar\mu=\sum_k\gamma_k\mu_k$",
        fontsize=7, color=EDGE_DARK)

fig.savefig(OUT / "apmd_arch_styleC_refined.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT / "apmd_arch_styleC_refined.png", bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Saved: {OUT}/apmd_arch_styleC_refined.{{pdf,png}}")
