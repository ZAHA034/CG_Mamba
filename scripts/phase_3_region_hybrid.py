"""Figure 3 (hybrid): US HHS-region coverage maps + horizon line panels.

Top  (A): 6 US HHS-region choropleths, one per model, shaded by Cov95 (h=1-4 avg).
          Single national-trained model, zero-shot to all 10 regions. CG-Mamba near-
          nominal everywhere (green); baselines under-cover (red).
Bottom(B): Cov95 vs horizon -- CG-Mamba near nominal, baselines collapse.
Bottom(C): WIS  vs horizon (lower better) -- CG-Mamba rises most gently but is 4th at
          h=1 (DLinear/LSTM/EpiDeep lower); honest non-uniform WIS story.

Data = native-APMD CSVs (identical to the prior Figure 3; NOT the scaled variant).
Contiguous US only (AK/HI/PR dropped); regions 9 & 10 shown via their mainland states.
Output: runs/phase_3_region_hybrid.{pdf,png} (+ copy to CGM_v2_paper/figures/).
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
import geopandas as gpd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "runs"
GEO = _ROOT / "data/geo/us_states.geojson"
PAPER_FIG = _ROOT / "CGM_v2_paper" / "figures" / "phase_3_region_hybrid.pdf"

BASELINES = ["lstm", "vanilla_mamba", "patchtst", "dlinear_ensemble_gauss",
             "epideep", "cg_mamba"]
HORIZONS = [1, 2, 3, 4]
SHORT = {"lstm": "LSTM", "vanilla_mamba": "Vanilla", "patchtst": "PatchTST",
         "dlinear_ensemble_gauss": "DLinear", "epideep": "EpiDeep", "cg_mamba": "CG-Mamba"}
MAP_TITLE = {"lstm": "LSTM", "vanilla_mamba": "Vanilla Mamba", "patchtst": "PatchTST",
             "dlinear_ensemble_gauss": "DLinear", "epideep": "EpiDeep",
             "cg_mamba": "CG-Mamba (ours)"}
COLORS = {"cg_mamba": "#0072B2", "lstm": "#D55E00", "vanilla_mamba": "#E69F00",
          "patchtst": "#009E73", "dlinear_ensemble_gauss": "#CC79A7", "epideep": "#56B4E9"}
MARK = {"cg_mamba": "o", "lstm": "s", "vanilla_mamba": "^", "patchtst": "D",
        "dlinear_ensemble_gauss": "v", "epideep": "X"}
HALO = [pe.withStroke(linewidth=1.4, foreground="white")]
CASE = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
NEAR_BLACK = "#1A1A1A"

HHS = {1: ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],
       2: ["New Jersey", "New York"],
       3: ["Delaware", "District of Columbia", "Maryland", "Pennsylvania", "Virginia", "West Virginia"],
       4: ["Alabama", "Florida", "Georgia", "Kentucky", "Mississippi", "North Carolina",
           "South Carolina", "Tennessee"],
       5: ["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"],
       6: ["Arkansas", "Louisiana", "New Mexico", "Oklahoma", "Texas"],
       7: ["Iowa", "Kansas", "Missouri", "Nebraska"],
       8: ["Colorado", "Montana", "North Dakota", "South Dakota", "Utah", "Wyoming"],
       9: ["Arizona", "California", "Hawaii", "Nevada"],
       10: ["Alaska", "Idaho", "Oregon", "Washington"]}
STATE2HHS = {s: k for k, v in HHS.items() for s in v}
DROP = {"Alaska", "Hawaii", "Puerto Rico"}


def set_style():
    # IEEE-listed font (Nimbus Sans = Helvetica/Arial equivalent), ~9-10 pt at full size.
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
        "axes.formatter.use_mathtext": False,
        "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal",
        "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.axisbelow": True, "axes.linewidth": 0.6, "axes.edgecolor": "#4D4D4D",
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
    df["cov_avg"] = df[[f"tS_cov95_h{h}" for h in HORIZONS]].mean(axis=1)
    return df


def per_horizon(df, b, prefix):
    s = df[df.baseline == b]
    return np.array([s.groupby("region")[f"{prefix}_h{h}"].mean().mean() for h in HORIZONS])


def per_region_cov(df):
    """DataFrame index=hhs int, columns=baseline, value=Cov95 (h1-4 avg)."""
    g = df.groupby(["baseline", "region"]).cov_avg.mean().reset_index()
    g["hhs"] = g["region"].str.extract(r"(\d+)").astype(int)
    return g.pivot(index="hhs", columns="baseline", values="cov_avg")


# --------------------------------------------------------------- horizon-lane panels
# Each model occupies its own horizontal lane; the four forecast horizons are drawn as
# connected dots with marker size growing h1 -> h4. Isolating each model in a lane keeps
# the per-horizon trajectory the text cites (0.998->0.910; h=1 4th, h>=2 lowest) while
# removing the multi-line overlap that made the earlier spaghetti version unreadable.
ORDER = ["cg_mamba", "lstm", "patchtst", "vanilla_mamba",
         "dlinear_ensemble_gauss", "epideep"]      # ascending mean WIS; hero on top lane
_HSIZE = np.array([3.2, 4.5, 5.8, 7.2])            # h1 (small) -> h4 (large)


def _lighten(hex_color, f):
    c = np.array(mpl.colors.to_rgb(hex_color))
    return tuple(c + (1.0 - c) * float(np.clip(f, 0.0, 1.0)))


def _lane(ax, df, prefix, order):
    n = len(order)
    for row, b in enumerate(order):
        y = n - 1 - row                            # first in `order` -> top lane
        v = per_horizon(df, b, prefix)
        hero = _hero(b)
        col = COLORS[b]
        ax.plot(v, np.full(4, y), color=col, lw=1.7 if hero else 1.05,
                alpha=0.95, solid_capstyle="round", zorder=6 if hero else 4,
                path_effects=CASE if hero else None)
        for h in range(4):
            ax.plot(v[h], y, marker="o", ms=_HSIZE[h],
                    mfc=_lighten(col, 0.55 - 0.18 * h), mec=col,
                    mew=1.1 if hero else 0.8, zorder=7 if hero else 5)
    labels = list(reversed(order))
    ax.set_yticks(range(n))
    ax.set_yticklabels([SHORT[b] for b in labels])
    for tick, b in zip(ax.get_yticklabels(), labels):
        tick.set_color(COLORS[b] if _hero(b) else "black")
        tick.set_fontweight("bold" if _hero(b) else "normal")
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x")


def _horizon_legend(fig):
    """Standalone size legend (outside the panels): four graded dots = horizon h1..h4."""
    handles = [Line2D([], [], marker="o", linestyle="", ms=_HSIZE[h],
                      mfc=_lighten("#808080", 0.55 - 0.18 * h), mec="#606060", mew=0.8)
               for h in range(4)]
    leg = fig.legend(handles, [f"h={h}" for h in HORIZONS],
                     title="Forecast horizon (marker size)", loc="lower center",
                     bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False,
                     handletextpad=0.25, columnspacing=1.1, fontsize=8, title_fontsize=8)
    leg.get_title().set_color("#555555")


def _letter(ax_or_fig, L, x, y):
    ax_or_fig.text(x, y, L, fontsize=11, fontweight="bold", va="bottom", ha="left")


def plot_cov(ax, df, order=ORDER):
    _lane(ax, df, "tS_cov95", order)
    ax.axvline(0.95, color="#333333", ls=":", lw=0.9, zorder=3)
    ax.text(0.941, 2.5, "nominal 0.95", rotation=90, fontsize=7.5,
            color="#555555", va="center", ha="right")
    ax.set_xlim(0.16, 1.06)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("Cov95  (closer to 0.95 is better)")
    ax.set_title("Coverage across horizon")


def plot_wis(ax, df, order=ORDER):
    _lane(ax, df, "tS_wis", order)
    ax.set_xlim(0.18, 0.80)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("WIS  (lower is better)")
    ax.set_title("Interval score across horizon")


# ------------------------------------------------------------------ maps
def build_regions():
    g = gpd.read_file(GEO)
    g = g[~g["name"].isin(DROP)].copy()
    g["hhs"] = g["name"].map(STATE2HHS)
    reg = g.dissolve(by="hhs").reset_index()
    return reg.set_crs(4326).to_crs(5070)


def plot_maps(fig, df):
    reg = build_regions()
    piv = per_region_cov(df)
    # Diverging norm CENTERED on the 0.95 nominal so the pale midpoint marks "on target":
    # a deviation in either direction reads as a departure from nominal, and over-coverage is
    # no longer rewarded with the deepest (hero) colour. TwoSlopeNorm maps [0.2,0.95]->[0,0.5]
    # (under-covers -> orange, unsafe side) and [0.95,1.0]->[0.5,1.0] (over-covers -> blue,
    # safe side); blue<->orange stays the colour-blind-safe Okabe-Ito diverging pair.
    norm = TwoSlopeNorm(vmin=0.2, vcenter=0.95, vmax=1.0)
    cmap = LinearSegmentedColormap.from_list(
        "cov_bluor", ["#D55E00", "#EFA24A", "#F3EEE6", "#5AA6D0", "#0072B2"])
    # single row of 6; thin vertical colorbar on the LEFT so the maps keep their height
    x0, w, gap = 0.084, 0.146, 0.005
    y0, hbox = 0.300, 0.500
    for i, b in enumerate(BASELINES):
        ax_x = x0 + i * (w + gap)
        ax = fig.add_axes([ax_x, y0, w, hbox])
        d = reg.merge(piv[b].rename("val"), left_on="hhs", right_index=True)
        d.plot(ax=ax, column="val", cmap=cmap, norm=norm, edgecolor="#7A7A7A", linewidth=0.35)
        ax.set_axis_off()
        hero = _hero(b)
        cx = ax_x + w / 2
        fig.text(cx, y0 + hbox + 0.004, MAP_TITLE[b], ha="center", va="bottom", fontsize=10,
                 fontweight="bold" if hero else "normal", color="#0072B2" if hero else "black")
        fig.text(cx, y0 - 0.004, f"mean {piv[b].mean():.3f}", ha="center", va="top", fontsize=9,
                 color="#0072B2" if hero else "#555555", fontweight="bold" if hero else "normal")
        if hero:
            ax.set_axis_on()
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor("#1A1A1A"); sp.set_linewidth(1.6)
    # vertical colorbar on the LEFT of the map row (blue=near nominal, red=under-covers -> caption)
    cax = fig.add_axes([0.033, y0 + 0.012, 0.013, hbox - 0.024])
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.axhline(0.95, color="black", lw=1.2)
    cb.ax.text(1.3, 0.95, "0.95", transform=cb.ax.get_yaxis_transform(), ha="left",
               va="center", fontsize=8, style="italic", color="#333333")
    cb.set_ticks([0.2, 0.6, 1.0]); cb.ax.tick_params(labelsize=9)
    cb.ax.yaxis.set_label_position("left"); cb.set_label("Cov95", fontsize=9, labelpad=1)


def main():
    set_style()
    df = load_long()
    figdir = _ROOT / "CGM_v2_paper" / "figures"

    # ---- Fig 3: HHS-region coverage maps (full width) ----
    fig_m = plt.figure(figsize=(7.16, 1.65))
    plot_maps(fig_m, df)
    fig_m.savefig(OUT / "phase_3_region_maps.pdf")
    fig_m.savefig(OUT / "phase_3_region_maps.png", dpi=300)
    plt.close(fig_m)
    shutil.copyfile(OUT / "phase_3_region_maps.pdf", figdir / "phase_3_region_maps.pdf")

    # ---- Fig 4: horizon-lane dot panels (single column, stacked) ----
    fig_h = plt.figure(figsize=(3.45, 4.25))
    axA = fig_h.add_axes([0.255, 0.650, 0.715, 0.298])   # coverage (top)
    axB = fig_h.add_axes([0.255, 0.205, 0.715, 0.298])   # WIS (bottom)
    plot_cov(axA, df)
    plot_wis(axB, df)
    _horizon_legend(fig_h)                               # standalone key below the panels
    _letter(fig_h, "A", 0.015, 0.958)
    _letter(fig_h, "B", 0.015, 0.505)
    fig_h.savefig(OUT / "phase_3_region_horizon.pdf")
    fig_h.savefig(OUT / "phase_3_region_horizon.png", dpi=300)
    plt.close(fig_h)
    shutil.copyfile(OUT / "phase_3_region_horizon.pdf", figdir / "phase_3_region_horizon.pdf")

    print("Saved: phase_3_region_maps + phase_3_region_horizon (pdf/png) + copied to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
