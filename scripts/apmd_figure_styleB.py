"""APMD architecture — STYLE B: Three-column streams converging to quantile output.

Bottom→top vertical flow inspired by the ReILIF fusion schema:
  - shared INPUT row at bottom (γ_T, μ_k, σ²_k, μ_CGM)
  - three parallel lanes in the middle:
        Lane 1 (left)   σ²_within stream
        Lane 2 (middle) σ²_between stream
        Lane 3 (right)  bias² stream  (interpretability triple component)
  - lanes 1+2 merge into σ²_total; lane 3 stays separate
  - calibration s_h applied above the merge
  - q_α formula at the top

Output:
    /A.I_DATA/jbnu/JeongHa/CG_Mamba/runs/figures/paper_drafts/
        apmd_arch_styleB.pdf
        apmd_arch_styleB.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "figures" / "paper_drafts"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette  (light-brown for operations, light-grey for "layer"/input boxes,
#           tinted accents for stream-specific identity, ReILIF-inspired)
# ---------------------------------------------------------------------------
COL_INPUT       = "#e6e6e6"   # light grey  – frozen inputs row
COL_OP          = "#f1e3c8"   # light brown – operation boxes
COL_LANE_WITHIN = "#dbe7f3"   # cool blue   – aleatoric lane
COL_LANE_BETW   = "#dfeede"   # cool green  – epistemic lane
COL_LANE_BIAS   = "#f3dfe2"   # warm pink   – interpretability lane
COL_TOTAL       = "#e8f4d8"   # light green – σ²_total merge
COL_CALIB       = "#f0e0e8"   # light pink  – calibration
COL_OUT         = "#d8e8f4"   # blue        – final q_α

EDGE = "#333333"
TEXT = "#1a1a1a"
ACC  = "#444444"

# ---------------------------------------------------------------------------
# Figure (2-col span: 7 in wide, 8.2 in tall for vertical stack)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 8.4))
ax.set_xlim(0, 105)
ax.set_ylim(0, 130)
ax.axis("off")


def box(x, y, w, h, color, text, *, fontsize=8, weight="normal",
        edge=EDGE, lw=0.9, radius=0.02, italic=False, text_color=TEXT):
    """Rounded rectangle with centered multi-line text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.15,rounding_size={radius * 100:.1f}",
        linewidth=lw, edgecolor=edge, facecolor=color, zorder=2,
    )
    ax.add_patch(patch)
    style = "italic" if italic else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight=weight, fontstyle=style, zorder=3)


def arrow(x1, y1, x2, y2, *, color=ACC, lw=1.2, style="-|>",
          mutation=12, ls="-", zorder=1):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation,
        linewidth=lw, color=color, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(arr)


def edge_label(x, y, txt, *, fontsize=7.2, color=ACC, ha="left"):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=fontsize,
            color=color, zorder=4)


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
LANE_W = 24.0
LANE_GAP = 4.0
LANE_X = {
    "within":  6.0,
    "between": 6.0 + (LANE_W + LANE_GAP),
    "bias":    6.0 + 2 * (LANE_W + LANE_GAP),
}
LANE_COL = {
    "within":  COL_LANE_WITHIN,
    "between": COL_LANE_BETW,
    "bias":    COL_LANE_BIAS,
}

# vertical band y-coordinates (in 0..130)
Y_INPUTS    = 4.0      # bottom: inputs row
Y_INPUT_H   = 10.0
Y_LANE_HDR  = 22.0     # lane header (label band)
Y_LANE_HDR_H = 5.5
Y_FORMULA   = 35.0     # per-lane formula
Y_FORMULA_H = 10.0
Y_DESC      = 52.0     # per-lane semantic tag
Y_DESC_H    = 6.5
Y_TOTAL     = 72.0     # σ²_total merge box (spans within+between)
Y_TOTAL_H   = 8.5
Y_CALIB     = 87.0     # calibration
Y_CALIB_H   = 8.5
Y_OUT       = 103.0    # final quantile box (spans whole width)
Y_OUT_H     = 13.0
Y_TITLE     = 124.0

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(50, Y_TITLE,
        "APMD: Analytic Phase-Mixture Decomposition",
        ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=TEXT)
ax.text(50, Y_TITLE - 4.5,
        r"three parallel variance streams $\rightarrow$ calibrated quantile forecast",
        ha="center", va="center",
        fontsize=8.5, color=ACC, fontstyle="italic")

# ---------------------------------------------------------------------------
# (1) Bottom: shared INPUT row -----------------------------------------------
# Frozen HMM stats (left) + CG-Mamba point forecast (right)
# ---------------------------------------------------------------------------
# Frozen-HMM input panel (spans within+between lanes)
hmm_x = LANE_X["within"]
hmm_w = (LANE_X["between"] + LANE_W) - LANE_X["within"]
box(hmm_x, Y_INPUTS, hmm_w, Y_INPUT_H, COL_INPUT,
    r"FROZEN HMM (Stage 1, $K{=}3$):  $\gamma_T\!\in\!\mathbb{R}^K$   "
    r"$\mu_k\!\in\!\mathbb{R}^K$   $\sigma^2_k\!\in\!\mathbb{R}^K$",
    fontsize=8.2, weight="bold")

# DL input panel (bias lane only)
dl_x = LANE_X["bias"]
box(dl_x, Y_INPUTS, LANE_W, Y_INPUT_H, COL_INPUT,
    "CG-Mamba (DL):" + "\n" + r"$\mu_{\mathrm{CGM}}$  point forecast",
    fontsize=8.2, weight="bold")

# tiny "inputs" gutter label on the left
ax.text(2.8, Y_INPUTS + Y_INPUT_H / 2, "inputs",
        ha="center", va="center", rotation=90,
        fontsize=7.5, color=ACC, fontstyle="italic")

# ---------------------------------------------------------------------------
# (2) Lane headers -----------------------------------------------------------
# ---------------------------------------------------------------------------
lane_titles = {
    "within":  (r"Lane 1 — $\sigma^2_{\mathrm{within}}$",  "aleatoric"),
    "between": (r"Lane 2 — $\sigma^2_{\mathrm{between}}$", "epistemic"),
    "bias":    (r"Lane 3 — $\mathrm{bias}^2$",             "interpretability"),
}
for key, (title, _sub) in lane_titles.items():
    box(LANE_X[key], Y_LANE_HDR, LANE_W, Y_LANE_HDR_H,
        LANE_COL[key], title, fontsize=8.5, weight="bold")

# ---------------------------------------------------------------------------
# (3) Per-lane formula (operation box, light-brown) -------------------------
# ---------------------------------------------------------------------------
formulas = {
    "within":  r"$\sigma^2_{\mathrm{within}} = \sum_{k=1}^{K} \gamma_k\,\sigma^2_k$",
    "between": (r"$\bar\mu = \sum_k \gamma_k\,\mu_k$" + "\n" +
                r"$\sigma^2_{\mathrm{between}} = \sum_k \gamma_k\,(\mu_k - \bar\mu)^2$"),
    "bias":    (r"$\bar\mu = \sum_k \gamma_k\,\mu_k$" + "\n" +
                r"$\mathrm{bias}^2 = (\bar\mu - \mu_{\mathrm{CGM}})^2$"),
}
for key, expr in formulas.items():
    box(LANE_X[key], Y_FORMULA, LANE_W, Y_FORMULA_H, COL_OP,
        expr, fontsize=8.3)

# ---------------------------------------------------------------------------
# (4) Per-lane semantic description -----------------------------------------
# ---------------------------------------------------------------------------
desc = {
    "within":  "per-phase noise\n(irreducible)",
    "between": "regime ambiguity\n(reducible w/ data)",
    "bias":    "CGM vs HMM mean gap\n(diagnostic only)",
}
for key, txt in desc.items():
    box(LANE_X[key], Y_DESC, LANE_W, Y_DESC_H, LANE_COL[key],
        txt, fontsize=7.7, italic=True)

# ---------------------------------------------------------------------------
# (5) σ²_total merge (within + between only — bias² is EXCLUDED) ------------
# ---------------------------------------------------------------------------
total_x = LANE_X["within"]
total_w = (LANE_X["between"] + LANE_W) - LANE_X["within"]
box(total_x, Y_TOTAL, total_w, Y_TOTAL_H, COL_TOTAL,
    r"$\sigma^2_{\mathrm{total}} = \sigma^2_{\mathrm{within}} + "
    r"\sigma^2_{\mathrm{between}}$"
    "    (bias$^2$ excluded — already absorbed into $\\mu_{\\mathrm{CGM}}$)",
    fontsize=8.4, weight="bold")

# ---------------------------------------------------------------------------
# (6) Calibration s_h (spans the σ²_total band) ------------------------------
# ---------------------------------------------------------------------------
box(total_x, Y_CALIB, total_w, Y_CALIB_H, COL_CALIB,
    r"Calibration:  $s_h$  (learned per horizon $h$ on validation grid-search)",
    fontsize=8.4, weight="bold")

# ---------------------------------------------------------------------------
# (7) Quantile output (spans the whole width) -------------------------------
# ---------------------------------------------------------------------------
out_x = LANE_X["within"]
out_w = (LANE_X["bias"] + LANE_W) - LANE_X["within"]
box(out_x, Y_OUT, out_w, Y_OUT_H, COL_OUT,
    r"$q_\alpha = \mu_{\mathrm{CGM}} + z_\alpha\,\sqrt{s_h \cdot \sigma^2_{\mathrm{total}}}$"
    "\n"
    r"$\alpha \in \{0.01, 0.025, 0.05, \ldots, 0.975, 0.99\}$"
    "  (23 FluSight quantile levels)",
    fontsize=9.0, weight="bold")

# ---------------------------------------------------------------------------
# Arrows -- bottom→top wiring
# ---------------------------------------------------------------------------
def lane_cx(key):
    return LANE_X[key] + LANE_W / 2.0


# (a) Inputs → lane headers
for key, label in [
    ("within",  r"$\gamma_k,\sigma^2_k$"),
    ("between", r"$\gamma_k,\mu_k$"),
]:
    cx = lane_cx(key)
    arrow(cx, Y_INPUTS + Y_INPUT_H, cx, Y_LANE_HDR, lw=1.2)
    edge_label(cx + 1.0, (Y_INPUTS + Y_INPUT_H + Y_LANE_HDR) / 2, label,
               fontsize=7.0)

# bias lane receives both μ_CGM (from DL panel directly above)
# and γ_k,μ_k (cross-lane edge from frozen HMM panel)
cx_b = lane_cx("bias")
arrow(cx_b, Y_INPUTS + Y_INPUT_H, cx_b, Y_LANE_HDR, lw=1.2)
edge_label(cx_b + 1.0, (Y_INPUTS + Y_INPUT_H + Y_LANE_HDR) / 2,
           r"$\mu_{\mathrm{CGM}}$", fontsize=7.0)

# cross-lane: γ_k,μ_k from HMM panel into bias lane (curved-ish via diagonal)
arrow(LANE_X["between"] + LANE_W - 2,
      Y_INPUTS + Y_INPUT_H,
      LANE_X["bias"] + 5,
      Y_LANE_HDR,
      lw=0.9, style="-|>", mutation=10, ls=(0, (3, 2)))
edge_label(LANE_X["bias"] + 3.0,
           Y_INPUTS + Y_INPUT_H + 5.0,
           r"$\gamma_k,\mu_k$", fontsize=6.8)

# (b) lane header → formula → description
for key in ("within", "between", "bias"):
    cx = lane_cx(key)
    arrow(cx, Y_LANE_HDR + Y_LANE_HDR_H, cx, Y_FORMULA, lw=1.1)
    arrow(cx, Y_FORMULA + Y_FORMULA_H, cx, Y_DESC, lw=1.1)

# (c) within & between → σ²_total
cx_w = lane_cx("within")
cx_bt = lane_cx("between")
total_top_y = Y_TOTAL
# converge diagonals
arrow(cx_w,  Y_DESC + Y_DESC_H, (cx_w + cx_bt) / 2, total_top_y,
      lw=1.3)
arrow(cx_bt, Y_DESC + Y_DESC_H, (cx_w + cx_bt) / 2, total_top_y,
      lw=1.3)

# (d) bias lane → side note (interpretability only, NOT into σ²_total)
cx_bias = lane_cx("bias")
# vertical stub going up but ending below σ²_total band
stub_top = Y_DESC + Y_DESC_H + 3.5
arrow(cx_bias, Y_DESC + Y_DESC_H, cx_bias, stub_top,
      lw=1.0, style="-|>", mutation=10, ls=(0, (4, 2)))
# dashed "report-only" box (placed first so text overlays)
rep_x = LANE_X["bias"] + 1.0
rep_y = stub_top + 0.3
rep_w = LANE_W - 2.0
rep_h = Y_TOTAL - rep_y - 1.0           # fill up to just below σ²_total band
rep_patch = FancyBboxPatch(
    (rep_x, rep_y), rep_w, rep_h,
    boxstyle="round,pad=0.05,rounding_size=1.5",
    linewidth=0.7, edgecolor="#7a2e2e", facecolor="none",
    linestyle=(0, (3, 2)), zorder=1,
)
ax.add_patch(rep_patch)
# annotation centered inside the dashed box
ax.text(rep_x + rep_w / 2, rep_y + rep_h / 2,
        "interpretability\ntriple component\n"
        r"(not in $\sigma^2_{\mathrm{total}}$)",
        ha="center", va="center", fontsize=7.2,
        color="#7a2e2e", fontstyle="italic", zorder=3)

# (e) σ²_total → calibration
arrow((LANE_X["within"] + LANE_X["between"] + LANE_W) / 2,
      Y_TOTAL + Y_TOTAL_H,
      (LANE_X["within"] + LANE_X["between"] + LANE_W) / 2,
      Y_CALIB,
      lw=1.4)
edge_label((LANE_X["within"] + LANE_X["between"] + LANE_W) / 2 + 1.0,
           (Y_TOTAL + Y_TOTAL_H + Y_CALIB) / 2,
           r"$\sigma^2_{\mathrm{total}}$", fontsize=7.2)

# (f) calibration → quantile output (merge with μ_CGM bypass from bias-lane input)
mid_x = (LANE_X["within"] + LANE_X["between"] + LANE_W) / 2
arrow(mid_x, Y_CALIB + Y_CALIB_H, mid_x, Y_OUT, lw=1.4)
edge_label(mid_x + 1.0, (Y_CALIB + Y_CALIB_H + Y_OUT) / 2,
           r"$s_h \cdot \sigma^2_{\mathrm{total}}$", fontsize=7.2)

# μ_CGM bypass: from DL input panel directly to quantile output (side-channel),
# routed along the right margin so it does NOT cross the bias-lane content.
bypass_x = LANE_X["bias"] + LANE_W + 2.5      # just outside the bias lane
# segment 1: up from the right edge of the DL input panel
arrow(LANE_X["bias"] + LANE_W - 1.0, Y_INPUTS + Y_INPUT_H / 2,
      bypass_x, Y_INPUTS + Y_INPUT_H / 2,
      lw=1.0, color="#1f4e79", style="-", mutation=8,
      ls=(0, (5, 2)))
arrow(bypass_x, Y_INPUTS + Y_INPUT_H / 2,
      bypass_x, Y_OUT + Y_OUT_H / 2,
      lw=1.0, color="#1f4e79", style="-", mutation=8,
      ls=(0, (5, 2)))
arrow(bypass_x, Y_OUT + Y_OUT_H / 2,
      LANE_X["bias"] + LANE_W - 1.0, Y_OUT + Y_OUT_H / 2,
      lw=1.0, color="#1f4e79", style="-|>", mutation=10,
      ls=(0, (5, 2)))
ax.text(bypass_x + 0.8, (Y_INPUTS + Y_OUT) / 2,
        r"$\mu_{\mathrm{CGM}}$" + "\n" + "(point)\nbypass",
        ha="left", va="center", fontsize=6.8,
        color="#1f4e79", rotation=0)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
pdf_path = OUT / "apmd_arch_styleB.pdf"
png_path = OUT / "apmd_arch_styleB.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=220)
plt.close(fig)

print(f"[apmd_styleB] wrote {pdf_path}")
print(f"[apmd_styleB] wrote {png_path}")
