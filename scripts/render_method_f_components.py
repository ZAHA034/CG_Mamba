"""Method F 3-Component UQ Figure — focused standalone version for paper §IV.6.

Uses runs/wis_method_f/decomposition_temporal.csv (seed42 test_strict h=1).

Layout: 3 panels (top to bottom)
  Panel A: Per-component σ² stacked over time (absolute magnitude)
  Panel B: Per-component fraction over time (proportional view)
  Panel C: Aggregate component contribution bar (mean ± std across all weeks)

Output:
  notebooks/figures/method_f_components/method_f_components.pdf + .png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = _ROOT / "runs" / "wis_method_f" / "decomposition_temporal.csv"
OUT_DIR = _ROOT / "notebooks" / "figures" / "method_f_components"

COLOR_WITHIN = "#2ca02c"     # green — aleatoric, per-phase
COLOR_BETWEEN = "#ff7f0e"    # orange — phase uncertainty
COLOR_BIAS = "#d62728"       # red — CG-Mamba refinement


def epiweek_to_date(ep: int) -> pd.Timestamp:
    year = ep // 100
    week = ep % 100
    jan4 = pd.Timestamp(year=year, month=1, day=4)
    iso_week_1_start = jan4 - pd.Timedelta(days=jan4.isoweekday() % 7)
    return iso_week_1_start + pd.Timedelta(weeks=week - 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df = df[df.horizon == 1].sort_values("target_ep").reset_index(drop=True)
    df["date"] = df.target_ep.apply(epiweek_to_date)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9),
                              gridspec_kw={"height_ratios": [2, 1.5, 1.4]})

    # ─── Panel A: stacked σ² absolute magnitude ───
    ax = axes[0]
    within = df.sigma2_within.values
    between = df.sigma2_between_HMM.values
    bias = df.bias_sq.values
    ax.fill_between(df.date, 0, within,
                     color=COLOR_WITHIN, alpha=0.75,
                     label="σ²_within (aleatoric, per-phase emission)")
    ax.fill_between(df.date, within, within + between,
                     color=COLOR_BETWEEN, alpha=0.75,
                     label="σ²_between (phase uncertainty)")
    ax.fill_between(df.date, within + between, within + between + bias,
                     color=COLOR_BIAS, alpha=0.55,
                     label="bias² (CG-Mamba refinement)")
    ax.set_ylabel("σ² (z-score² units)", fontsize=10)
    ax.set_title("(A) Decomposable σ²_total over test_strict (h=1, seed=42)\n"
                 "Components are additive: σ²_total = σ²_within + σ²_between + bias²",
                 fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))

    # ─── Panel B: fraction view ───
    ax = axes[1]
    total = within + between + bias + 1e-12
    f_within = within / total
    f_between = between / total
    f_bias = bias / total
    ax.fill_between(df.date, 0, f_within,
                     color=COLOR_WITHIN, alpha=0.75)
    ax.fill_between(df.date, f_within, f_within + f_between,
                     color=COLOR_BETWEEN, alpha=0.75)
    ax.fill_between(df.date, f_within + f_between,
                     f_within + f_between + f_bias,
                     color=COLOR_BIAS, alpha=0.55)
    ax.set_ylabel("Fraction of σ²_total", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("(B) Per-component fraction of σ²_total (proportional view)",
                 fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))

    # ─── Panel C: aggregate contribution bar (across all 4 horizons) ───
    ax = axes[2]
    df_all = pd.read_csv(CSV_PATH)
    horizons = sorted(df_all.horizon.unique())
    n_h = len(horizons)
    x = np.arange(n_h)
    bar_w = 0.7

    within_means = []
    between_means = []
    bias_means = []
    for h in horizons:
        sub = df_all[df_all.horizon == h]
        t = sub.sigma2_within + sub.sigma2_between_HMM + sub.bias_sq + 1e-12
        within_means.append((sub.sigma2_within / t).mean())
        between_means.append((sub.sigma2_between_HMM / t).mean())
        bias_means.append((sub.bias_sq / t).mean())
    within_means = np.array(within_means)
    between_means = np.array(between_means)
    bias_means = np.array(bias_means)

    ax.bar(x, within_means, bar_w, color=COLOR_WITHIN, alpha=0.75,
            edgecolor="black", linewidth=0.6, label="σ²_within")
    ax.bar(x, between_means, bar_w, bottom=within_means,
            color=COLOR_BETWEEN, alpha=0.75,
            edgecolor="black", linewidth=0.6, label="σ²_between")
    ax.bar(x, bias_means, bar_w,
            bottom=within_means + between_means,
            color=COLOR_BIAS, alpha=0.55,
            edgecolor="black", linewidth=0.6, label="bias²")
    for i in range(n_h):
        ax.text(i, within_means[i] / 2, f"{within_means[i]*100:.1f}%",
                ha="center", va="center", fontsize=9, color="white", weight="bold")
        ax.text(i, within_means[i] + between_means[i] / 2,
                f"{between_means[i]*100:.1f}%",
                ha="center", va="center", fontsize=9, color="white", weight="bold")
        ax.text(i, within_means[i] + between_means[i] + bias_means[i] / 2,
                f"{bias_means[i]*100:.1f}%",
                ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean fraction of σ²_total", fontsize=10)
    ax.set_xlabel("Forecast horizon", fontsize=10)
    ax.set_title("(C) Aggregate component contribution by horizon "
                 "(mean across all test_strict weeks)",
                 fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Figure 7 — Method F Decomposable Uncertainty: 3 Additive Components\n"
        "(test_strict period W40-2022 ~ W35-2025, h=1 in panels A/B, all horizons in C)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf = OUT_DIR / "method_f_components.pdf"
    png = OUT_DIR / "method_f_components.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"       {png}")

    print("\n=== Per-horizon mean component fractions ===")
    for h, w, b, x in zip(horizons, within_means, between_means, bias_means):
        print(f"  h={h}: within={w:.3f}  between={b:.3f}  bias²={x:.3f}")


if __name__ == "__main__":
    main()
