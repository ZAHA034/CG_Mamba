"""APMD architecture figure — STYLE A: bottom-up hierarchy mirroring ReILIF.

Vertical flow (bottom -> top):
  Inputs row (HMM stats + CG-Mamba point forecast)
    -> Three parallel K=3 branches, each with within-stream (gamma_k * sigma_k^2)
       and between-stream (gamma_k * (mu_k - mu_bar)^2)
    -> Aggregation Sigma at top of each variance stream
    -> Merge sigma_within + sigma_between => sigma_total
    -> Calibration s_h
    -> Quantile output q_alpha = mu_CGM + z_alpha * sqrt(s_h * sigma_total^2)
  bias^2 = (mu_bar - mu_CGM)^2  shown as a side branch (interpretability only).

Output:
  runs/figures/paper_drafts/apmd_arch_styleA.{pdf,png}
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "figures" / "paper_drafts"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Style: light pastel palette, grouped by computation layer
# ----------------------------------------------------------------------
C_INPUT_HMM = "#e6eef7"   # very light blue  (frozen HMM stats)
C_INPUT_DL  = "#dde7f3"   # slightly darker blue (DL point forecast)
C_OP        = "#f5e9d3"   # warm light brown (atomic op, ReILIF-style)
C_AGG       = "#efe2c8"   # darker brown (Sigma aggregator)
C_WITHIN    = "#e3eed8"   # pale green (aleatoric stream)
C_BETWEEN   = "#f1e0e0"   # pale rose (epistemic stream)
C_TOTAL     = "#dfe9d4"   # green-tinted merge
C_CALIB     = "#e8def0"   # pale lavender
C_OUTPUT    = "#cfddee"   # output blue
C_BIAS      = "#f7f1da"   # pale yellow (interp. side branch)

EDGE  = "#3a3a3a"
TEXT  = "#1c1c1c"
ARROW = "#4a4a4a"
DL    = "#1f5fa8"

# ----------------------------------------------------------------------
# Canvas (2-col span — density requires it)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 8.4))
ax.set_xlim(-2, 112)
ax.set_ylim(0, 120)
ax.axis("off")


def add_box(x, y, w, h, color, text, *, fontsize=7.5, weight="normal",
            edge=EDGE, lw=0.8, rounding=0.020, italic=False):
    """Centered rounded box with text. (x,y) is the bottom-left corner."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding * max(w, h)}",
        linewidth=lw, edgecolor=edge, facecolor=color, zorder=2,
    )
    ax.add_patch(box)
    style = "italic" if italic else "normal"
    ax.text(x + w / 2.0, y + h / 2.0, text,
            ha="center", va="center", fontsize=fontsize, color=TEXT,
            weight=weight, style=style, zorder=3)


def arrow(x0, y0, x1, y1, *, color=ARROW, lw=0.9, style="-|>",
          mut=8, connect="arc3,rad=0.0", zorder=1):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=mut, color=color,
        linewidth=lw, connectionstyle=connect, zorder=zorder,
    )
    ax.add_patch(a)


def edge_label(x, y, text, *, fontsize=6.4, color=TEXT, bg=True):
    if bg:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.92))
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=color, zorder=4)


# ======================================================================
# LAYER 0 (bottom): INPUTS — HMM stats + CG-Mamba point forecast
# ======================================================================
y_in = 4.0
in_h = 6.5
in_w = 13.0

# Three per-phase input tiles (k=1,2,3): each shows (gamma_k, mu_k, sigma_k^2)
phase_x = [10.0, 32.0, 54.0]
phase_labels = [
    (r"$\gamma_1$, $\mu_1$, $\sigma_1^2$", "phase k = 1"),
    (r"$\gamma_2$, $\mu_2$, $\sigma_2^2$", "phase k = 2"),
    (r"$\gamma_3$, $\mu_3$, $\sigma_3^2$", "phase k = 3"),
]
for px, (lab, sub) in zip(phase_x, phase_labels):
    add_box(px, y_in, in_w, in_h, C_INPUT_HMM,
            lab + "\n" + sub,
            fontsize=7.2)

# CG-Mamba point forecast input (right side)
add_box(78.0, y_in, 16.0, in_h, C_INPUT_DL,
        r"$\mu_{\mathrm{CGM}}$" + "\n" + "(CG-Mamba)",
        fontsize=7.5, weight="bold", edge=DL, lw=1.1)

# Footer caption under inputs row
ax.text(50.0, y_in - 2.4,
        r"frozen Stage 1 HMM statistics  $+$  Stage 3 deep forecast",
        ha="center", va="center", fontsize=6.6, style="italic", color="#444")

# ======================================================================
# LAYER 1: mu_bar mixture mean (needed by both bias^2 and between-stream)
# ======================================================================
y_mubar = 18.0
add_box(28.0, y_mubar, 24.0, 5.6, C_OP,
        r"$\bar\mu \;=\; \sum_k \gamma_k\,\mu_k$",
        fontsize=8.0)
# Connect each phase tile to mu_bar
for px in phase_x:
    arrow(px + in_w / 2.0, y_in + in_h, 40.0, y_mubar, lw=0.7)

# ======================================================================
# LAYER 2: three parallel branches — each phase contributes to within & between
# Operation boxes (ReILIF-style) for each k:
#    within_k  = gamma_k * sigma_k^2
#    between_k = gamma_k * (mu_k - mu_bar)^2
# ======================================================================
y_branch = 31.0
br_h = 5.2
br_w = 13.0

# WITHIN-stream branch boxes (left half of canvas, one per k)
within_x = [3.0, 19.0, 35.0]
for k, bx in enumerate(within_x, start=1):
    add_box(bx, y_branch, br_w, br_h, C_WITHIN,
            r"$\gamma_{%d}\,\sigma_{%d}^2$" % (k, k),
            fontsize=8.4)
    # arrow from corresponding input
    arrow(phase_x[k - 1] + in_w / 2.0, y_in + in_h,
          bx + br_w / 2.0, y_branch, lw=0.7,
          connect="arc3,rad=-0.15")

# BETWEEN-stream branch boxes (right half), depend on mu_bar
between_x = [52.0, 68.0, 84.0]
for k, bx in enumerate(between_x, start=1):
    add_box(bx, y_branch, br_w, br_h, C_BETWEEN,
            r"$\gamma_{%d}(\mu_{%d}\!-\!\bar\mu)^2$" % (k, k),
            fontsize=7.8)
    # arrow from input
    arrow(phase_x[k - 1] + in_w / 2.0, y_in + in_h,
          bx + br_w / 2.0, y_branch, lw=0.7,
          connect="arc3,rad=0.25")
    # arrow from mu_bar
    arrow(40.0, y_mubar + 5.6, bx + br_w / 2.0, y_branch,
          lw=0.6, color="#777", style="-|>", mut=6,
          connect="arc3,rad=0.20")

# Stream column labels
ax.text(22.0, y_branch + br_h + 2.6, r"within-phase (aleatoric)",
        ha="center", va="center", fontsize=7.4, style="italic", color="#3a5a3a")
ax.text(72.0, y_branch + br_h + 2.6, r"between-phase (epistemic)",
        ha="center", va="center", fontsize=7.4, style="italic", color="#7a3a3a")

# ======================================================================
# LAYER 3: aggregation Sigma at top of each variance stream
# ======================================================================
y_agg = 47.0
agg_h = 6.4

# Sigma aggregator — WITHIN
add_box(13.0, y_agg, 26.0, agg_h, C_AGG,
        r"$\sum_k\;\;\Rightarrow\;\;\sigma^2_{\mathrm{within}}$",
        fontsize=8.6, weight="bold")
for bx in within_x:
    arrow(bx + br_w / 2.0, y_branch + br_h,
          26.0, y_agg, lw=0.8, connect="arc3,rad=0.0")

# Sigma aggregator — BETWEEN
add_box(61.0, y_agg, 26.0, agg_h, C_AGG,
        r"$\sum_k\;\;\Rightarrow\;\;\sigma^2_{\mathrm{between}}$",
        fontsize=8.6, weight="bold")
for bx in between_x:
    arrow(bx + br_w / 2.0, y_branch + br_h,
          74.0, y_agg, lw=0.8, connect="arc3,rad=0.0")

# ======================================================================
# LAYER 4: merge to sigma_total
# ======================================================================
y_tot = 62.0
add_box(30.0, y_tot, 40.0, 7.2, C_TOTAL,
        r"$\sigma^2_{\mathrm{total}} \;=\; "
        r"\sigma^2_{\mathrm{within}} + \sigma^2_{\mathrm{between}}$",
        fontsize=9.0, weight="bold")
arrow(26.0, y_agg + agg_h, 42.0, y_tot, lw=1.0)
arrow(74.0, y_agg + agg_h, 58.0, y_tot, lw=1.0)

# ======================================================================
# LAYER 5: calibration s_h
# ======================================================================
y_cal = 76.5
add_box(33.0, y_cal, 34.0, 6.4, C_CALIB,
        r"$s_h \cdot \sigma^2_{\mathrm{total}}$"
        + "\n"
        + "calibration (val grid-search, per horizon)",
        fontsize=7.6, italic=False)
arrow(50.0, y_tot + 7.2, 50.0, y_cal, lw=1.1)

# ======================================================================
# LAYER 6 (top): quantile output  q_alpha = mu_CGM + z_alpha * sqrt(s_h * sigma_total^2)
# ======================================================================
y_out = 91.0
add_box(14.0, y_out, 72.0, 9.0, C_OUTPUT,
        r"$q_\alpha \;=\; \mu_{\mathrm{CGM}} \;+\; "
        r"z_\alpha \sqrt{\,s_h \cdot \sigma^2_{\mathrm{total}}\,}$",
        fontsize=10.2, weight="bold", edge=DL, lw=1.3)
arrow(50.0, y_cal + 6.4, 50.0, y_out, lw=1.3)

# mu_CGM feeds into the quantile output directly (long curved bypass on the right)
arrow(86.0, y_in + in_h * 0.5, 86.0, y_out + 2.0,
      lw=0.9, color=DL, connect="arc3,rad=0.0", style="-|>", mut=8)
edge_label(89.0, (y_in + y_out) / 2.0,
           r"$\mu_{\mathrm{CGM}}$", fontsize=6.8, color=DL, bg=True)

# Output caption
ax.text(50.0, y_out + 10.5,
        r"23 FluSight quantile levels  ($\alpha \in \{0.01,0.025,\dots,0.99\}$)",
        ha="center", va="center", fontsize=7.0, style="italic", color="#333")

# ======================================================================
# SIDE BRANCH: bias^2 — interpretability only (not part of sigma_total)
# Drawn on the far left, parallel to layers 3-4.
# ======================================================================
y_bias = 47.0
add_box(0.5, y_bias, 11.0, 6.4, C_BIAS,
        r"$\mathrm{bias}^2$" + "\n"
        + r"$(\bar\mu\!-\!\mu_{\mathrm{CGM}})^2$",
        fontsize=7.0, edge="#a59060", lw=0.9)
# mu_bar -> bias
arrow(28.0, y_mubar + 2.8, 11.5, y_bias + 3.2,
      lw=0.6, color="#a59060", connect="arc3,rad=0.30", style="-|>", mut=6)
# mu_CGM -> bias (dashed-feel: just use lighter color)
arrow(78.0, y_in + in_h * 0.5, 6.0, y_bias,
      lw=0.6, color="#a59060", connect="arc3,rad=-0.45", style="-|>", mut=6)
ax.text(6.0, y_bias - 2.2, r"interpretability only",
        ha="center", va="center", fontsize=6.0, style="italic", color="#6a5a30")
ax.text(6.0, y_bias - 4.0, r"(not in $\sigma^2_{\mathrm{total}}$)",
        ha="center", va="center", fontsize=6.0, style="italic", color="#6a5a30")

# ======================================================================
# Layer labels on the far right (ReILIF-style "row tags")
# ======================================================================
row_tags = [
    (y_in + in_h / 2.0,      "Inputs"),
    (y_mubar + 2.8,          r"$\bar\mu$"),
    (y_branch + br_h / 2.0,  "Per-phase ops"),
    (y_agg + agg_h / 2.0,    r"$\sum_k$ aggregate"),
    (y_tot + 3.6,            "Total variance"),
    (y_cal + 3.2,            "Calibration"),
    (y_out + 4.5,            "Quantile output"),
]
for yy, tag in row_tags:
    ax.text(110.5, yy, tag, ha="right", va="center", fontsize=6.4,
            style="italic", color="#666",
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor="#fafafa", edgecolor="#cccccc", lw=0.4))

# ======================================================================
# Title
# ======================================================================
ax.text(50.0, 117.0,
        "APMD: Analytic Phase-Mixture Decomposition",
        ha="center", va="center", fontsize=11.0, weight="bold", color=TEXT)
ax.text(50.0, 113.5,
        r"Frozen HMM ($\gamma_T, \mu_k, \sigma_k^2$) "
        r"$\oplus$ CG-Mamba point forecast $\mu_{\mathrm{CGM}}$  "
        r"$\rightarrow$  calibrated Gaussian quantiles",
        ha="center", va="center", fontsize=7.6, style="italic", color="#444")

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
plt.tight_layout(pad=0.3)
pdf_path = OUT / "apmd_arch_styleA.pdf"
png_path = OUT / "apmd_arch_styleA.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
plt.close(fig)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
