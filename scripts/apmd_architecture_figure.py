"""Horizontal APMD architecture diagram (matplotlib mock-up for review).

Output: runs/figures/paper_drafts/apmd_architecture.{pdf,png}

5-box horizontal pipeline:
  INPUTS → DECOMPOSITION → σ²_total → CALIBRATION → QUANTILES (output)
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/A.I_DATA/jbnu/JeongHa/CG_Mamba")
OUT = ROOT / "runs" / "figures" / "paper_drafts"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.0, 2.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")

# Color palette
COLOR_INPUT  = "#e0e8f0"   # light blue
COLOR_DECOMP = "#fff4d8"   # light orange
COLOR_TOTAL  = "#e8f4d8"   # light green
COLOR_CALIB  = "#f0e0e8"   # light pink
COLOR_OUT    = "#d8e8f4"   # blue-tinted

EDGE = "#444444"
TEXT = "#222222"
DL_BLUE = "#1f77b4"

box_w = 16
box_h = 16
gap = 4
y_box = 9

# Box positions
positions = [
    {"x": 2,                "color": COLOR_INPUT,  "title": "INPUTS",
     "body": r"$\gamma_T$ (HMM post.)" + "\n" +
             r"$\mu_k$, $\sigma^2_k$" + "\n" +
             r"$\mu_{\mathrm{CGM}}$"},
    {"x": 2 + (box_w + gap), "color": COLOR_DECOMP, "title": "DECOMPOSITION",
     "body": r"$\sigma^2_{\mathrm{within}}\!=\!\sum_k\gamma_k\sigma_k^2$" + "\n" +
             r"$\sigma^2_{\mathrm{between}}\!=\!\sum_k\gamma_k(\mu_k-\bar\mu)^2$" + "\n" +
             r"$\mathrm{bias}^2\!=\!(\bar\mu-\mu_{\mathrm{CGM}})^2$"},
    {"x": 2 + 2*(box_w + gap), "color": COLOR_TOTAL,  "title": r"$\sigma^2_{\mathrm{total}}$",
     "body": r"$\sigma^2_{\mathrm{within}}$" + "\n" +
             r"$+\,\sigma^2_{\mathrm{between}}$" + "\n" +
             r"(bias² excluded)"},
    {"x": 2 + 3*(box_w + gap), "color": COLOR_CALIB,  "title": "CALIBRATION",
     "body": r"$s_h$ via grid-" + "\n" +
             "search on val\n" +
             "(per horizon)"},
    {"x": 2 + 4*(box_w + gap), "color": COLOR_OUT,    "title": "QUANTILES",
     "body": r"$q_\alpha\!=\!\mu_{\mathrm{CGM}}\!+\!z_\alpha\sqrt{s_h\sigma^2_{\mathrm{total}}}$" + "\n" +
             "23 FluSight levels\n" +
             "+ interpretability"},
]

# Draw boxes
for i, p in enumerate(positions):
    box = FancyBboxPatch(
        (p["x"], y_box), box_w, box_h,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor=p["color"], edgecolor=EDGE, linewidth=1.0
    )
    ax.add_patch(box)
    # Title
    ax.text(p["x"] + box_w/2, y_box + box_h - 2.2, p["title"],
            ha="center", va="top", fontsize=9, fontweight="bold", color=TEXT)
    # Body
    ax.text(p["x"] + box_w/2, y_box + box_h/2 - 1.5, p["body"],
            ha="center", va="center", fontsize=7.5, color=TEXT)

# Draw arrows between boxes
for i in range(len(positions) - 1):
    src_x = positions[i]["x"] + box_w
    dst_x = positions[i+1]["x"]
    arrow = FancyArrowPatch(
        (src_x + 0.2, y_box + box_h/2),
        (dst_x - 0.2, y_box + box_h/2),
        arrowstyle="->", mutation_scale=14,
        color=EDGE, linewidth=1.2,
    )
    ax.add_patch(arrow)

# Bottom interpretability annotation
ax.text(2 + 1*(box_w + gap) + box_w/2, y_box - 2.8,
        "Aleatoric  /  Epistemic  /  Refinement",
        ha="center", va="top", fontsize=7, color="#666666", style="italic")
ax.annotate("",
            xy=(2 + 1*(box_w + gap) + box_w/2, y_box - 0.5),
            xytext=(2 + 1*(box_w + gap) + box_w/2, y_box - 2.0),
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

# Top label: highlighting unique architectural property
ax.text(50, y_box + box_h + 3.5,
        "APMD: HMM-derived calibrated intervals with decomposable interpretability",
        ha="center", va="bottom", fontsize=8.5, color=DL_BLUE, fontweight="bold")

fig.savefig(OUT / "apmd_architecture.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT / "apmd_architecture.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {OUT}/apmd_architecture.{{pdf,png}}")
