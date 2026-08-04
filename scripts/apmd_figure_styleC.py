"""APMD architecture figure - STYLE C: Functional pipeline (left-to-right) with branching subgraph.

Layout:
  - Main pipeline (horizontal): INPUTS -> DECOMPOSE -> COMBINE (sigma^2_total)
    -> CALIBRATE -> OUTPUT (q_alpha)
  - Above pipeline: K=3 phase contribution mini-graph (small branch boxes
    converging into DECOMPOSE)
  - Below pipeline: bias^2 separate annotation channel (interpretability only,
    NOT fed into sigma^2_total)

Output: runs/figures/paper_drafts/apmd_arch_styleC.{pdf,png}
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path("/A.I_DATA/jbnu/JeongHa/CG_Mamba")
OUT = ROOT / "runs" / "figures" / "paper_drafts"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Canvas
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.set_xlim(0, 100)
ax.set_ylim(-2, 44)
ax.axis("off")

# ----------------------------------------------------------------------------
# Palette (ReILIF-inspired: light brown for operations, light grey for layers)
# ----------------------------------------------------------------------------
COLOR_INPUT    = "#e0e8f0"   # light blue - inputs/data
COLOR_DECOMP   = "#f4e4c8"   # light brown - decomposition operation
COLOR_COMBINE  = "#e8f0d8"   # light green - aggregation
COLOR_CALIB    = "#f0dce4"   # light pink - calibration
COLOR_OUT      = "#d4e4f4"   # blue-tinted - quantile output
COLOR_PHASE    = "#ececec"   # light grey - phase contribution (small branches)
COLOR_BIAS     = "#fdf2e0"   # very light tan - bias annotation
COLOR_BIAS_EDGE = "#b89060"

EDGE      = "#3a3a3a"
EDGE_SOFT = "#888888"
TEXT      = "#1a1a1a"
DL_BLUE   = "#1f4e79"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def draw_box(ax, x, y, w, h, color, edge=EDGE, lw=1.0, radius=0.6):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=color, zorder=2,
    )
    ax.add_patch(box)


def title_text(ax, x, y, txt, size=8.5, weight="bold", color=TEXT):
    ax.text(x, y, txt, ha="center", va="center", fontsize=size,
            weight=weight, color=color, zorder=4)


def body_text(ax, x, y, txt, size=7.0, color=TEXT):
    ax.text(x, y, txt, ha="center", va="center", fontsize=size,
            color=color, zorder=4)


def edge_label(ax, x, y, txt, size=6.8, color=TEXT, bg="white"):
    ax.text(x, y, txt, ha="center", va="center", fontsize=size,
            color=color, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=bg,
                      edgecolor="none", alpha=0.92))


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.2, style="-|>", mut=10):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mut,
        linewidth=lw, color=color, zorder=3,
    )
    ax.add_patch(arr)


def dashed_arrow(ax, x1, y1, x2, y2, color=EDGE_SOFT, lw=0.9):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=color, zorder=3,
        linestyle=(0, (3, 2)),
    )
    ax.add_patch(arr)


# ----------------------------------------------------------------------------
# Main pipeline (5 boxes, horizontal, y-band ~ 14-26)
# ----------------------------------------------------------------------------
MAIN_Y    = 14         # box bottom
MAIN_H    = 12         # box height
MAIN_CY   = MAIN_Y + MAIN_H / 2   # = 20

# Box widths chosen so total + gaps ~ 96
box_specs = [
    {"label": "INPUTS",       "w": 14, "color": COLOR_INPUT,
     "body": r"$\gamma_T,\ \mu_k,\ \sigma^2_k$" + "\n" + r"$\mu_{\mathrm{CGM}}$"},
    {"label": "DECOMPOSE",    "w": 20, "color": COLOR_DECOMP,
     "body": r"$\bar\mu=\sum_k\!\gamma_k\mu_k$" + "\n"
             r"$\sigma^2_{\mathrm{w}}\!=\!\sum_k\!\gamma_k\sigma_k^2$" + "\n"
             r"$\sigma^2_{\mathrm{b}}\!=\!\sum_k\!\gamma_k(\mu_k\!-\!\bar\mu)^2$"},
    {"label": "COMBINE",      "w": 16, "color": COLOR_COMBINE,
     "body": r"$\sigma^2_{\mathrm{total}}$" + "\n"
             r"$=\sigma^2_{\mathrm{w}}+\sigma^2_{\mathrm{b}}$"},
    {"label": "CALIBRATE",    "w": 16, "color": COLOR_CALIB,
     "body": r"$s_h\!\cdot\!\sigma^2_{\mathrm{total}}$" + "\n"
             r"($s_h$ val-grid)"},
    {"label": "OUTPUT",       "w": 18, "color": COLOR_OUT,
     "body": r"$q_\alpha\!=\!\mu_{\mathrm{CGM}}\!+\!z_\alpha\sqrt{s_h\sigma^2_{\mathrm{tot}}}$"
             + "\n" + r"23 FluSight quantiles"},
]

GAP = 2.5
x_cursor = 2.0
box_centers = []
for spec in box_specs:
    x = x_cursor
    w = spec["w"]
    draw_box(ax, x, MAIN_Y, w, MAIN_H, spec["color"], lw=1.1, radius=0.8)
    title_text(ax, x + w / 2, MAIN_Y + MAIN_H - 2.2, spec["label"],
               size=8.5, weight="bold")
    body_text(ax, x + w / 2, MAIN_Y + MAIN_H / 2 - 1.8, spec["body"], size=6.8)
    box_centers.append({"label": spec["label"],
                        "xl": x, "xr": x + w, "cx": x + w / 2,
                        "cy": MAIN_CY})
    x_cursor = x + w + GAP

# Arrows between adjacent main boxes, with math edge labels
edge_labels = [
    r"$(\gamma_T,\mu_k,\sigma^2_k,\mu_{\mathrm{CGM}})$",
    r"$(\sigma^2_{\mathrm{w}},\sigma^2_{\mathrm{b}})$",
    r"$\sigma^2_{\mathrm{total}}$",
    r"$s_h\sigma^2_{\mathrm{total}}$",
]
for i in range(len(box_centers) - 1):
    L = box_centers[i]
    R = box_centers[i + 1]
    arrow(ax, L["xr"], MAIN_CY, R["xl"], MAIN_CY, lw=1.3, mut=11)
    midx = (L["xr"] + R["xl"]) / 2
    edge_label(ax, midx, MAIN_CY + 0.05, edge_labels[i], size=6.4)

# ----------------------------------------------------------------------------
# Above pipeline: K=3 phase contribution mini-graph
# Three small grey boxes feeding into DECOMPOSE
# ----------------------------------------------------------------------------
PHASE_Y  = 32
PHASE_H  = 6.5
PHASE_W  = 11
decomp = box_centers[1]   # DECOMPOSE box

# Three phase mini-boxes centered above DECOMPOSE
phase_cx_list = [decomp["cx"] - 13, decomp["cx"], decomp["cx"] + 13]
phase_labels  = [
    (r"$k\!=\!1$", r"$\gamma_1,\mu_1,\sigma^2_1$"),
    (r"$k\!=\!2$", r"$\gamma_2,\mu_2,\sigma^2_2$"),
    (r"$k\!=\!3$", r"$\gamma_3,\mu_3,\sigma^2_3$"),
]
for cx, (kk, lab) in zip(phase_cx_list, phase_labels):
    x0 = cx - PHASE_W / 2
    draw_box(ax, x0, PHASE_Y, PHASE_W, PHASE_H, COLOR_PHASE,
             edge=EDGE_SOFT, lw=0.8, radius=0.4)
    title_text(ax, cx, PHASE_Y + PHASE_H - 1.6, kk, size=7.2,
               weight="bold", color="#444444")
    body_text(ax, cx, PHASE_Y + 1.7, lab, size=6.2, color="#333333")

# Bracket label for K=3 phases
ax.text(decomp["cx"] - 23.5, PHASE_Y + PHASE_H / 2,
        "K=3 phases\n(frozen HMM)",
        ha="right", va="center", fontsize=6.8,
        color="#555555", style="italic")

# Curved arrows from each phase down into DECOMPOSE top edge
decomp_top_y = MAIN_Y + MAIN_H
for cx in phase_cx_list:
    tgt_x = decomp["cx"] + (cx - decomp["cx"]) * 0.35
    arr = FancyArrowPatch(
        (cx, PHASE_Y),
        (tgt_x, decomp_top_y),
        arrowstyle="-|>", mutation_scale=8,
        linewidth=0.9, color=EDGE_SOFT, zorder=2,
        connectionstyle="arc3,rad=0.18",
    )
    ax.add_patch(arr)

# Sigma symbol above DECOMPOSE indicating aggregation Sum_k
ax.text(decomp["cx"], decomp_top_y + 0.9,
        r"$\sum_{k=1}^{K}\!\gamma_k(\cdot)$",
        ha="center", va="bottom", fontsize=7.2, color="#444444",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=EDGE_SOFT, linewidth=0.6, alpha=0.95))

# ----------------------------------------------------------------------------
# Below pipeline: bias^2 annotation channel (interpretability only)
# ----------------------------------------------------------------------------
BIAS_Y = 0.5
BIAS_H = 6.0
BIAS_W = 26
bias_cx = (box_centers[0]["cx"] + box_centers[2]["cx"]) / 2

draw_box(ax, bias_cx - BIAS_W / 2, BIAS_Y, BIAS_W, BIAS_H, COLOR_BIAS,
         edge=COLOR_BIAS_EDGE, lw=0.9, radius=0.5)
title_text(ax, bias_cx, BIAS_Y + BIAS_H - 1.7,
           r"bias$^2 = (\bar\mu - \mu_{\mathrm{CGM}})^2$",
           size=7.3, weight="bold", color="#704020")
body_text(ax, bias_cx, BIAS_Y + 1.7,
          "interpretability only (excluded from $\\sigma^2_{\\mathrm{total}}$)",
          size=6.2, color="#704020")

# Dashed arrow: DECOMPOSE -> bias channel (computed here, but routed aside)
dashed_arrow(ax, decomp["cx"] - 2, MAIN_Y,
             bias_cx + 2, BIAS_Y + BIAS_H,
             color=COLOR_BIAS_EDGE, lw=0.9)
# Dashed arrow: bias channel -> exits sideways (not into COMBINE)
# emphasize "excluded": small X-mark style is too noisy; instead show no
# arrow into COMBINE/CALIBRATE — just a terminator label.
ax.text(bias_cx + BIAS_W / 2 + 0.5, BIAS_Y + BIAS_H / 2,
        "(report\nonly)", ha="left", va="center", fontsize=6.0,
        color="#704020", style="italic")

# ----------------------------------------------------------------------------
# DL skip connection: mu_CGM bypasses to OUTPUT (used inside q_alpha formula)
# ----------------------------------------------------------------------------
# A subtle blue dashed arc from INPUTS bottom to OUTPUT bottom showing
# that mu_CGM enters the quantile formula directly as the location parameter.
# Routed BELOW the main row, between main row and bias channel, with
# limited arc curvature so it doesn't dip into the bias band.
inp = box_centers[0]
outp = box_centers[-1]
arr_skip = FancyArrowPatch(
    (inp["cx"] + 2.0, MAIN_Y),
    (outp["cx"] - 2.0, MAIN_Y),
    arrowstyle="-|>", mutation_scale=8,
    linewidth=0.9, color=DL_BLUE, zorder=2,
    linestyle=(0, (4, 2)),
    connectionstyle="arc3,rad=0.16",
)
ax.add_patch(arr_skip)
ax.text((inp["cx"] + outp["cx"]) / 2 + 6, MAIN_Y - 2.2,
        r"$\mu_{\mathrm{CGM}}$ (DL skip: location of $q_\alpha$)",
        ha="center", va="center", fontsize=6.3, color=DL_BLUE,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.92))

# ----------------------------------------------------------------------------
# Stage banner across the top
# ----------------------------------------------------------------------------
ax.text(50, 40.5,
        "APMD (Analytic Phase-Mixture Decomposition)",
        ha="center", va="center", fontsize=9.5, weight="bold",
        color=TEXT)

# ----------------------------------------------------------------------------
# Legend (compact, bottom-right)
# ----------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=COLOR_INPUT,   edgecolor=EDGE,
                   label="data / inputs"),
    mpatches.Patch(facecolor=COLOR_DECOMP,  edgecolor=EDGE,
                   label="decomposition op"),
    mpatches.Patch(facecolor=COLOR_PHASE,   edgecolor=EDGE_SOFT,
                   label="phase $k$ (HMM)"),
    mpatches.Patch(facecolor=COLOR_BIAS,    edgecolor=COLOR_BIAS_EDGE,
                   label="bias$^2$ (report only)"),
]
leg = ax.legend(handles=legend_handles, loc="lower right",
                bbox_to_anchor=(1.0, -0.05),
                fontsize=6.2, frameon=True, ncol=4, columnspacing=0.8,
                handlelength=1.0, handletextpad=0.4, borderpad=0.3)
leg.get_frame().set_edgecolor("#bbbbbb")
leg.get_frame().set_linewidth(0.5)

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
plt.tight_layout(pad=0.3)
pdf_path = OUT / "apmd_arch_styleC.pdf"
png_path = OUT / "apmd_arch_styleC.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print(f"[OK] PDF -> {pdf_path}")
print(f"[OK] PNG -> {png_path}")
