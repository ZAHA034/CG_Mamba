"""PROTOTYPE: US HHS 10-region choropleth small-multiples (native APMD, Cov95).

Single national-trained model applied zero-shot to all 10 HHS regions.
6 small US maps (one per model), each HHS region shaded by its Cov95 (h=1-4 avg).
CG-Mamba = near-nominal everywhere (green); baselines under-cover (red).
Data source = the SAME native-APMD CSVs as Figure 3 (NOT the older scaled variant).

Contiguous US only (AK/HI/PR dropped); regions 9 & 10 shown via their mainland states.
Output: runs/map_proto/hhs_cov95_maps.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
import geopandas as gpd
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
GEO = _ROOT / "data/geo/us_states.geojson"
MET = _ROOT / "runs/map_proto/per_region_metrics.csv"
OUT = _ROOT / "runs/map_proto"

HHS = {
    1: ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],
    2: ["New Jersey", "New York"],
    3: ["Delaware", "District of Columbia", "Maryland", "Pennsylvania", "Virginia", "West Virginia"],
    4: ["Alabama", "Florida", "Georgia", "Kentucky", "Mississippi", "North Carolina",
        "South Carolina", "Tennessee"],
    5: ["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"],
    6: ["Arkansas", "Louisiana", "New Mexico", "Oklahoma", "Texas"],
    7: ["Iowa", "Kansas", "Missouri", "Nebraska"],
    8: ["Colorado", "Montana", "North Dakota", "South Dakota", "Utah", "Wyoming"],
    9: ["Arizona", "California", "Hawaii", "Nevada"],
    10: ["Alaska", "Idaho", "Oregon", "Washington"],
}
STATE2HHS = {s: k for k, v in HHS.items() for s in v}
DROP = {"Alaska", "Hawaii", "Puerto Rico"}

ORDER = ["lstm", "vanilla_mamba", "patchtst", "dlinear_ensemble_gauss", "epideep", "cg_mamba"]
TITLE = {"lstm": "LSTM", "vanilla_mamba": "Vanilla Mamba", "patchtst": "PatchTST",
         "dlinear_ensemble_gauss": "DLinear", "epideep": "EpiDeep", "cg_mamba": "CG-Mamba (ours)"}


def set_style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9,
    })


def build_regions():
    g = gpd.read_file(GEO)
    g = g[~g["name"].isin(DROP)].copy()
    g["hhs"] = g["name"].map(STATE2HHS)
    assert g["hhs"].notna().all(), g[g.hhs.isna()]["name"].tolist()
    reg = g.dissolve(by="hhs").reset_index()
    reg = reg.set_crs(4326).to_crs(5070)          # CONUS Albers equal-area
    return reg


def main():
    set_style()
    reg = build_regions()
    met = pd.read_csv(MET)
    met["hhs"] = met["region"].str.extract(r"(\d+)").astype(int)
    piv = met.pivot(index="hhs", columns="baseline", values="cov95")

    norm = Normalize(vmin=0.2, vmax=1.0)
    cmap = plt.get_cmap("RdYlGn")

    fig, axes = plt.subplots(1, 6, figsize=(13.5, 3.2))
    for ax, b in zip(axes, ORDER):
        d = reg.merge(piv[b].rename("val"), left_on="hhs", right_index=True)
        d.plot(ax=ax, column="val", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.6)
        ax.set_axis_off()
        hero = (b == "cg_mamba")
        ax.set_title(TITLE[b], fontsize=9.5, fontweight="bold" if hero else "normal",
                     color="#0072B2" if hero else "black", pad=3)
        mean = piv[b].mean()
        ax.text(0.5, -0.02, f"mean Cov95 {mean:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5,
                color="#0072B2" if hero else "#555555",
                fontweight="bold" if hero else "normal")
        if hero:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor("#0072B2"); sp.set_linewidth(1.6)
            ax.set_frame_on(True)

    fig.suptitle("Zero-shot national$\\rightarrow$regional transfer: interval coverage (Cov95) across the "
                 "10 HHS regions from a single national-trained model ($h{=}1$–4 avg)",
                 fontsize=10, fontweight="bold", y=0.965)

    cax = fig.add_axes([0.32, 0.10, 0.36, 0.030])
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Cov95   (green = near nominal 0.95;  red = under-covers)", fontsize=8)
    cb.ax.axvline(0.95, color="black", lw=1.2)
    cb.ax.text(0.95, 1.7, "nominal 0.95", transform=cb.ax.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=7, style="italic")
    cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.20, wspace=0.02)
    pdf = OUT / "hhs_cov95_maps.pdf"; png = OUT / "hhs_cov95_maps.png"
    fig.savefig(pdf); fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"Saved: {pdf}\n       {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
