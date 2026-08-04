"""APMD architecture figure — Style C v2 (paper-quality, text overflow fixed).

Key fixes vs v1:
- All boxes sized to fit their text (no overflow)
- Legend moved to top-left (no skip arc collision)
- Shorter text inside boxes (math moved to arrow labels where possible)
- Larger fonts where space permits
- bias² label fits inside its box
- μ_CGM skip arc clearly visible

Output: runs/figures/paper_drafts/apmd_arch_styleC_v2.{pdf,png}
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path("/A.I_DATA/jbnu/JeongHa/CG_Mamba")
OUT = ROOT / "runs" / "figures" / "paper_drafts"
OUT.mkdir(parents=True, exist_ok=True)

# Color palette
COLOR_INPUT  = "#dde7f0"
COLOR_OP     = "#f2f0e8"
COLOR_OUTPUT = "#e1ebd3"
COLOR_BIAS   = "#fadcdc"
COLOR_PHASE  = "#e8e0f0"
EDGE_DARK    = "#333333"
EDGE_LIGHT   = "#666666"
DL_BLUE      = "#1f6faf"
BIAS_RED     = "#a04040"

fig, ax = plt.subplots(figsize=(11.0, 5.0))
ax.set_xlim(0, 110)
ax.set_ylim(0, 55)
ax.axis("off")

# ============================================================
# Main pipeline (5 boxes, sized to fit content)
# ============================================================
main_y = 22
main_h = 13

# Boxes with widths sized to text content
main_boxes = [
    {"x": 2,    "w": 17,  "title": "INPUTS",
     "color": COLOR_INPUT,
     "body": r"$\gamma_T,\,\mu_k,\,\sigma^2_k$" + "\n(HMM, K=3,\nfrozen)\n" + r"$\mu_{\mathrm{CGM}}$ (DL)"},
    {"x": 25,   "w": 19,  "title": "DECOMPOSE",
     "color": COLOR_OP,
     "body": r"$\sigma^2_w=\sum_k\gamma_k\sigma^2_k$" + "\n" +
             r"$\sigma^2_b=\sum_k\gamma_k(\mu_k\!-\!\bar\mu)^2$"},
    {"x": 50,   "w": 15,  "title": "COMBINE",
     "color": COLOR_OP,
     "body": r"$\sigma^2_{\mathrm{total}}\!=$" + "\n" + r"$\sigma^2_w+\sigma^2_b$"},
    {"x": 71,   "w": 14,  "title": "CALIBRATE",
     "color": COLOR_OP,
     "body": r"$s_h\cdot\sigma^2_{\mathrm{total}}$" + "\n" + r"($s_h$ val-grid)"},
    {"x": 91,   "w": 17,  "title": "OUTPUT",
     "color": COLOR_OUTPUT,
     "body": r"$q_\alpha=\mu_{\mathrm{CGM}}$" + "\n" +
             r"$+z_\alpha\sqrt{s_h\sigma^2_{\mathrm{tot}}}$" + "\n" +
             "23 FluSight\nquantiles"},
]

box_info = []
for p in main_boxes:
    box = FancyBboxPatch((p["x"], main_y), p["w"], main_h,
                          boxstyle="round,pad=0.4,rounding_size=0.6",
                          facecolor=p["color"], edgecolor=EDGE_DARK, linewidth=1.0)
    ax.add_patch(box)
    cx = p["x"] + p["w"] / 2
    cy_top = main_y + main_h
    ax.text(cx, cy_top - 2.5, p["title"], ha="center", va="top",
            fontsize=10, fontweight="bold", color=EDGE_DARK)
    ax.text(cx, cy_top - 5.5, p["body"], ha="center", va="top",
            fontsize=8.0, color=EDGE_DARK)
    box_info.append((p["x"], p["x"] + p["w"], cy_top - main_h/2))

# Inter-box arrows with math labels
arrow_labels = [
    r"$\gamma_T,\mu_k,\sigma^2_k$",
    r"$\sigma^2_w,\sigma^2_b$",
    r"$\sigma^2_{\mathrm{total}}$",
    r"$s_h\sigma^2_{\mathrm{total}}$",
]
for i in range(len(box_info) - 1):
    src_x = box_info[i][1]
    dst_x = box_info[i+1][0]
    y = box_info[i][2]
    arr = FancyArrowPatch((src_x + 0.3, y), (dst_x - 0.3, y),
                          arrowstyle="-|>", mutation_scale=12,
                          color=EDGE_DARK, linewidth=1.3)
    ax.add_patch(arr)
    mid_x = (src_x + dst_x) / 2
    ax.text(mid_x, y + 1.2, arrow_labels[i], ha="center", va="bottom",
            fontsize=7.5, color=EDGE_LIGHT, style="italic")

# ============================================================
# K=3 phase mini-graph ABOVE DECOMPOSE
# ============================================================
phase_y = main_y + main_h + 4
phase_h = 4
decompose = main_boxes[1]
phase_x_starts = [decompose["x"] + 2, decompose["x"] + 8, decompose["x"] + 14]

for i, px in enumerate(phase_x_starts):
    box = FancyBboxPatch((px, phase_y), 3.5, phase_h,
                          boxstyle="round,pad=0.2,rounding_size=0.3",
                          facecolor=COLOR_PHASE, edgecolor=EDGE_DARK, linewidth=0.8)
    ax.add_patch(box)
    ax.text(px + 1.75, phase_y + phase_h/2, f"$k\\!=\\!{i+1}$",
            ha="center", va="center", fontsize=8.5, color=EDGE_DARK)
    arr = FancyArrowPatch((px + 1.75, phase_y),
                          (px + 1.75, main_y + main_h + 0.3),
                          arrowstyle="-|>", mutation_scale=8,
                          color=EDGE_LIGHT, linewidth=0.9, linestyle=":")
    ax.add_patch(arr)

# Phase constraint annotation
ax.text(decompose["x"] + decompose["w"]/2, phase_y + phase_h + 1.5,
        r"K=3 HMM phases (frozen)  $\quad\sum_k\gamma_k=1$",
        ha="center", va="bottom", fontsize=8.5, fontstyle="italic", color=EDGE_LIGHT)

# ============================================================
# bias² terminator BELOW DECOMPOSE — wider, text fits inside
# ============================================================
bias_y = main_y - 7.5
bias_x = decompose["x"] - 5  # extend left for label fit
bias_w = decompose["w"] + 10
bias_h = 4.5

bias_box = FancyBboxPatch((bias_x, bias_y), bias_w, bias_h,
                           boxstyle="round,pad=0.3,rounding_size=0.4",
                           facecolor=COLOR_BIAS, edgecolor=BIAS_RED,
                           linewidth=1.0, linestyle="--")
ax.add_patch(bias_box)
ax.text(bias_x + bias_w/2, bias_y + bias_h/2,
        r"$\mathrm{bias}^2=(\bar\mu-\mu_{\mathrm{CGM}})^2$" +
        "  —  interpretability triple (report only, $\\notin\\sigma^2_{\\mathrm{total}}$)",
        ha="center", va="center", fontsize=8.5, color=BIAS_RED)

# Arrow from DECOMPOSE down to bias²
arr = FancyArrowPatch((decompose["x"] + decompose["w"]/2, main_y - 0.3),
                      (decompose["x"] + decompose["w"]/2, bias_y + bias_h + 0.3),
                      arrowstyle="-|>", mutation_scale=10,
                      color=BIAS_RED, linewidth=1.0, linestyle="--")
ax.add_patch(arr)

# ============================================================
# μ_CGM skip arc at BOTTOM (clear, no overlap)
# ============================================================
skip_y = 4.5
input_box = main_boxes[0]
output_box = main_boxes[-1]
input_cx = input_box["x"] + input_box["w"]/2
output_cx = output_box["x"] + output_box["w"]/2

# Vertical drop from INPUTS
ax.plot([input_cx, input_cx], [main_y, skip_y],
        color=DL_BLUE, lw=1.5, ls="--", alpha=0.85)
# Horizontal arc
ax.plot([input_cx, output_cx], [skip_y, skip_y],
        color=DL_BLUE, lw=1.5, ls="--", alpha=0.85)
# Vertical rise to OUTPUT
arr = FancyArrowPatch((output_cx, skip_y),
                      (output_cx, main_y - 0.3),
                      arrowstyle="-|>", mutation_scale=12,
                      color=DL_BLUE, linewidth=1.5, linestyle="--")
ax.add_patch(arr)

# Skip label (centered, no legend collision)
ax.text((input_cx + output_cx)/2, skip_y - 2.0,
        r"$\mu_{\mathrm{CGM}}$  (DL skip — location parameter of $q_\alpha$)",
        ha="center", va="top", fontsize=8.5, color=DL_BLUE,
        fontweight="bold", style="italic")

# ============================================================
# Title at TOP
# ============================================================
ax.text(55, 52, "APMD: HMM-derived calibrated intervals with decomposable interpretability",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1a4d7a")

# ============================================================
# Legend at TOP-LEFT corner (away from skip arc)
# ============================================================
legend_x = 0.5
legend_y = 45
legend_w = 24
legend_h = 6
ax.add_patch(Rectangle((legend_x, legend_y), legend_w, legend_h,
                        facecolor="white", edgecolor=EDGE_LIGHT, linewidth=0.5))
ax.text(legend_x + 1, legend_y + 4.7, "→  data flow", fontsize=7.5, color=EDGE_DARK)
ax.text(legend_x + 1, legend_y + 3.0, "⇢  skip / report-only", fontsize=7.5, color=EDGE_DARK)
ax.text(legend_x + 1, legend_y + 1.3,
        r"$\gamma_k\!:$ HMM phase weight,  $\bar\mu\!=\!\sum_k\gamma_k\mu_k$",
        fontsize=7, color=EDGE_DARK)

fig.savefig(OUT / "apmd_arch_styleC_v2.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT / "apmd_arch_styleC_v2.png", bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Saved: {OUT}/apmd_arch_styleC_v2.{{pdf,png}}")
