"""FluSight 2018-2019 — per-team WIS distribution boxplot (4 horizons).

Phase 5 §IV.X-flusight figure (PLAN §18.7 Plan A, CP 5.5c).

Reads:  runs/phase_5_flusight/team_wis_2018_2019.csv
Writes: notebooks/figures/phase_5/team_wis_boxplot.pdf + .png

Layout: 2×2 panels, one per h={1,2,3,4}. Each panel shows boxplot of WIS per
team (one box per team) sorted by median ascending. CG-Mamba slot reserved as
a dashed horizontal reference line per panel (filled when CP 5.4 retrospective
is available).

Style: monochrome-friendly, IEEE single-column 3.5in width per row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[1]
LONG_CSV = _ROOT / "runs" / "phase_5_flusight" / "team_wis_2018_2019.csv"
FIG_DIR = _ROOT / "notebooks" / "figures" / "phase_5"


def render(cg_mamba_wis: dict[int, float] | None = None, out_dir: Path | None = None):
    """Build the 4-panel boxplot.

    Args:
        cg_mamba_wis: optional dict {h: mean_wis} from CP 5.4 retrospective.
                      Plotted as horizontal dashed reference line per panel.
        out_dir:      output directory (default notebooks/figures/phase_5).
    """
    out_dir = out_dir or FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not LONG_CSV.exists():
        raise FileNotFoundError(f"Missing {LONG_CSV}. Run team_wis script first.")
    df = pd.read_csv(LONG_CSV)
    df = df.dropna(subset=["wis"])
    print(f"Loaded {len(df)} rows ({df['team'].nunique()} teams)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    horizons = [1, 2, 3, 4]
    for ax, h in zip(axes.flat, horizons):
        sub = df[df["target_h"] == h]
        # Sort teams by median WIS ascending
        team_order = sub.groupby("team")["wis"].median().sort_values().index.tolist()
        data = [sub[sub["team"] == t]["wis"].values for t in team_order]
        bp = ax.boxplot(data, vert=True, showfliers=False,
                         medianprops=dict(color="black", linewidth=1.2),
                         boxprops=dict(facecolor="lightgray", edgecolor="black"),
                         whiskerprops=dict(color="black"),
                         patch_artist=True)
        ax.set_xticks(range(1, len(team_order) + 1))
        ax.set_xticklabels(team_order, rotation=90, fontsize=6)
        ax.set_title(f"h={h} (1-{h} wk ahead)", fontsize=10)
        ax.set_ylabel("WIS", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # CG-Mamba reference line
        if cg_mamba_wis and h in cg_mamba_wis:
            ax.axhline(y=cg_mamba_wis[h], linestyle="--", color="red", linewidth=1.5,
                       label=f"CG-Mamba: {cg_mamba_wis[h]:.3f}")
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("FluSight 2018-2019 — Per-team WIS Distribution (US National, n=29 weeks)",
                  fontsize=11, y=0.995)
    fig.tight_layout()
    pdf = out_dir / "team_wis_boxplot.pdf"
    png = out_dir / "team_wis_boxplot.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"       {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cg-mamba-h1", type=float, default=None,
                    help="(optional) CG-Mamba retrospective WIS at h=1 (from CP 5.4)")
    ap.add_argument("--cg-mamba-h2", type=float, default=None)
    ap.add_argument("--cg-mamba-h3", type=float, default=None)
    ap.add_argument("--cg-mamba-h4", type=float, default=None)
    args = ap.parse_args()
    cg_wis = {h: getattr(args, f"cg_mamba_h{h}") for h in [1, 2, 3, 4]
              if getattr(args, f"cg_mamba_h{h}") is not None}
    cg_wis = cg_wis if cg_wis else None
    render(cg_mamba_wis=cg_wis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
