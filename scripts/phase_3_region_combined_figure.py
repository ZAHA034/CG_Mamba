"""Regional zero-shot figure -- 3 panels, publication-grade rebuild (IEEE JBHI).

Panels (data unchanged; restyle only):
  (A) Cov95 (y) vs WIS (x) aggregate scatter, per-baseline mean + cross-region std (y bars).
      Carries the ranking, per-region robustness, and the "only CG-Mamba near-nominal" story.
  (B) Cov95 across horizons h=1..4 : CG-Mamba near nominal (0.998->0.910), baselines collapse.
  (C) WIS  across horizons h=1..4 (lower better): rises for all; CG-Mamba rises most gently and
      is lowest from h=2, but is 4th of 6 at h=1 (DLinear/LSTM/EpiDeep lower) -- kept honest.

Design: authored at final print size (7.16 x 3.35 in), STIX-serif to match IEEEtran body,
Okabe-Ito colour-blind-safe palette with a blue CG-Mamba hero (consistent with Fig. 2 / the
efficiency figure), distinct markers for grayscale, decluttered labels. NO x error bars on A:
the cross-region WIS std is a between-region scale artefact (~3.9x the mean spread) and would
misrepresent the paired-Wilcoxon comparison reported in the text.

Aggregation: per-region 5-seed mean, then cross-region mean (WIS 0.393 / Cov95 0.954).
Output: runs/phase_3_region_combined_figure.{pdf,png}  (+ copy to CGM_v2_paper/figures/).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "runs"
PAPER_FIG = _ROOT / "CGM_v2_paper" / "figures" / "phase_3_region_combined.pdf"

BASELINES = ["lstm", "vanilla_mamba", "patchtst", "dlinear_ensemble_gauss",
             "epideep", "cg_mamba"]                     # cg_mamba last -> drawn on top
HORIZONS = [1, 2, 3, 4]

SHORT = {"lstm": "LSTM", "vanilla_mamba": "Vanilla", "patchtst": "PatchTST",
         "dlinear_ensemble_gauss": "DLinear", "epideep": "EpiDeep", "cg_mamba": "CG-Mamba"}
# Okabe-Ito subset; CG-Mamba hero = blue #0072B2 (darkest -> pops in grayscale, matches Fig. 2)
COLORS = {"cg_mamba": "#0072B2", "lstm": "#D55E00", "vanilla_mamba": "#E69F00",
          "patchtst": "#009E73", "dlinear_ensemble_gauss": "#CC79A7", "epideep": "#56B4E9"}
MARK = {"cg_mamba": "o", "lstm": "s", "vanilla_mamba": "^", "patchtst": "D",
        "dlinear_ensemble_gauss": "v", "epideep": "X"}

HALO = [pe.withStroke(linewidth=1.4, foreground="white")]
CASE = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
NEAR_BLACK = "#1A1A1A"


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 8, "axes.titleweight": "normal",
        "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.axisbelow": True,
        "axes.linewidth": 0.6, "axes.edgecolor": "#4D4D4D",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.color": "#4D4D4D", "ytick.color": "#4D4D4D",
        "xtick.labelcolor": "black", "ytick.labelcolor": "black",
        "grid.color": "#CCCCCC", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    })


def _hero(b):
    return b == "cg_mamba"


def load_long():
    use = ["baseline", "seed", "region"] + \
          [f"tS_wis_h{h}" for h in HORIZONS] + [f"tS_cov95_h{h}" for h in HORIZONS]
    df = pd.concat([pd.read_csv(_ROOT / "runs/phase_3_region_wis.csv")[use],
                    pd.read_csv(_ROOT / "runs/e1_final/n3_d64_regional_perhorizon_raw.csv")[use],
                    pd.read_csv(_ROOT / "runs/phase_3_region_wis_extras.csv")[use]],
                   ignore_index=True).dropna(subset=["tS_wis_h1"])
    df = df[df["baseline"].isin(BASELINES)]
    df["wis_avg"] = df[[f"tS_wis_h{h}" for h in HORIZONS]].mean(axis=1)
    df["cov_avg"] = df[[f"tS_cov95_h{h}" for h in HORIZONS]].mean(axis=1)
    return df


def per_horizon(df, b, prefix):
    s = df[df.baseline == b]
    return np.array([s.groupby("region")[f"{prefix}_h{h}"].mean().mean() for h in HORIZONS])


def agg(df, b):
    s = df[df.baseline == b]
    w = s.groupby("region").wis_avg.mean()
    c = s.groupby("region").cov_avg.mean()
    return w.mean(), w.std(ddof=1), c.mean(), c.std(ddof=1)


def _chrome(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def _letter(ax, L):
    ax.annotate(L, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9, fontweight="bold", annotation_clip=False)


# ---------------------------------------------------------------- Panel A
def plot_scatter(ax, df):
    # neutral target-zone band (not the hero hue) + single nominal reference
    ax.axhspan(0.93, 0.97, color="#ECECEC", zorder=0)
    ax.axhline(0.95, color="#333333", ls=":", lw=0.9, zorder=1)
    ax.text(0.995, 0.952, "nominal 0.95", transform=ax.get_yaxis_transform(),
            fontsize=6.5, color="#555555", va="bottom", ha="right")

    # label offsets in data units: (dx, dy, ha, va). Every model shows its exact
    # (WIS, Cov95) pair -- this panel carries the per-model values (replaces Table III).
    OFF = {"cg_mamba": (0.008, 0.020, "left", "center"),
           "patchtst": (0.009, 0.000, "left", "center"),
           "vanilla_mamba": (0.009, 0.015, "left", "center"),
           "lstm": (-0.009, 0.000, "right", "center"),
           "epideep": (0.009, 0.020, "left", "center"),
           "dlinear_ensemble_gauss": (0.009, -0.018, "left", "center")}
    for b in BASELINES:
        wm, ws, cm, cs = agg(df, b)
        hero = _hero(b)
        ax.errorbar(wm, cm, yerr=cs, fmt=MARK[b], color=COLORS[b], ecolor=COLORS[b],
                    elinewidth=1.2 if hero else 0.8, capsize=2.5,
                    ms=6.5 if hero else 4.5, mec="black", mew=0.8 if hero else 0.5,
                    zorder=10 if hero else 5)
        dx, dy, ha, va = OFF[b]
        ax.text(wm + dx, cm + dy, f"{SHORT[b]}\n({wm:.3f}, {cm:.3f})", ha=ha, va=va,
                fontsize=7, linespacing=1.25,
                color=COLORS["cg_mamba"] if hero else NEAR_BLACK,
                fontweight="bold" if hero else "normal",
                zorder=11, path_effects=HALO)
    ax.text(0.985, 0.905, "(WIS, Cov95)", transform=ax.transAxes, fontsize=6,
            style="italic", color="#888888", ha="right", va="top")

    ax.set_xlabel("Interval score, WIS (lower is better)")
    ax.set_ylabel("95% coverage (Cov95)")
    ax.set_xlim(0.36, 0.56)
    ax.set_ylim(0.18, 1.02)
    ax.set_xticks([0.40, 0.45, 0.50, 0.55])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title("Only CG-Mamba holds near-nominal coverage (10-region aggregate)")
    _chrome(ax)
    _letter(ax, "A")


# ---------------------------------------------------------------- Panels B, C
def _place_labels(ax, items, x_at, min_gap, dx=0.12):
    """items: list of (y, text, color, hero). Nudge apart, place at x_at+dx."""
    items = sorted(items, key=lambda t: t[0])
    adj = [it[0] for it in items]
    for i in range(1, len(adj)):
        if adj[i] - adj[i - 1] < min_gap:
            adj[i] = adj[i - 1] + min_gap
    for (y, txt, col, hero), ya in zip(items, adj):
        ax.annotate(txt, xy=(x_at, y), xytext=(x_at + dx, ya), textcoords="data",
                    fontsize=6.5, color=col, fontweight="bold" if hero else "normal",
                    va="center", ha="left", path_effects=HALO, annotation_clip=False)


def _profile(ax, df, prefix):
    x = np.array(HORIZONS, float)
    ys = {}
    for b in BASELINES:
        y = per_horizon(df, b, prefix)
        ys[b] = y
        hero = _hero(b)
        ax.plot(x, y, color=COLORS[b], lw=2.0 if hero else 1.1,
                marker=MARK[b], ms=5.0 if hero else 3.5, mec="black",
                mew=0.5 if hero else 0.4, zorder=10 if hero else 5,
                path_effects=CASE if hero else None)
    return ys


def plot_cov(ax, df):
    ys = _profile(ax, df, "tS_cov95")
    ax.axhline(0.95, color="#333333", ls=":", lw=0.9, zorder=3)
    ax.text(4.02, 0.95, "nominal 0.95", fontsize=6.5, color="#555555",
            va="bottom", ha="right")
    # end-of-line model names
    items = [(ys[b][-1], SHORT[b], COLORS[b], _hero(b)) for b in BASELINES]
    _place_labels(ax, items, x_at=4.0, min_gap=0.058)
    # hero drift numeric (near-black), placed up-right of the h=1 marker to clear the panel letter
    ax.annotate("0.998", xy=(1, ys["cg_mamba"][0]), xytext=(1.12, 1.005),
                fontsize=6.5, color=NEAR_BLACK, fontweight="bold", va="center", ha="left",
                path_effects=HALO)
    ax.set_ylabel("95% coverage (Cov95)")
    ax.set_xticks(HORIZONS)
    ax.set_xlim(0.85, 4.95)
    ax.set_ylim(0.17, 1.03)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.setp(ax.get_xticklabels(), visible=False)   # x shared with Panel C
    ax.set_title("Coverage by horizon")
    _chrome(ax)
    _letter(ax, "B")


def plot_wis(ax, df):
    ys = _profile(ax, df, "tS_wis")
    # end-of-line model names
    items = [(ys[b][-1], SHORT[b], COLORS[b], _hero(b)) for b in BASELINES]
    _place_labels(ax, items, x_at=4.0, min_gap=0.040)
    # honest h=1-ranking caveat in the open upper-left (calibration-vs-sharpness -> caption)
    ax.annotate("At $h{=}1$, CG-Mamba is 4th of 6\n(DLinear lowest, WIS 0.221).",
                xy=(1.0, ys["cg_mamba"][0]), xytext=(1.08, 0.79),
                fontsize=6.5, color="#555555", style="italic", va="top", ha="left",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=0.6,
                                shrinkA=2, shrinkB=3))
    ax.set_xlabel("Forecast horizon $h$ (weeks)")
    ax.set_ylabel("WIS (interval score; lower is better)")
    ax.set_xticks(HORIZONS)
    ax.set_xlim(0.85, 4.95)
    ax.set_ylim(0.19, 0.82)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title("Interval score by horizon")
    _chrome(ax)
    _letter(ax, "C")


def main():
    set_style()
    df = load_long()
    fig = plt.figure(figsize=(7.16, 3.35), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.03, wspace=0.03)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0])
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1], sharex=axB)
    plot_scatter(axA, df)
    plot_cov(axB, df)
    plot_wis(axC, df)

    pdf = OUT / "phase_3_region_combined_figure.pdf"
    png = OUT / "phase_3_region_combined_figure.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    shutil.copyfile(pdf, PAPER_FIG)
    print(f"Saved: {pdf}\n       {png}\n  copy: {PAPER_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
