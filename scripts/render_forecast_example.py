"""Forecast Example Figure — single-season qualitative forecast plot with CI.

Uses runs/wis_method_f/decomposition_temporal.csv (seed42 test_strict predictions,
W40-2022 ~ W35-2025). Renders all 4 horizons in a 2x2 panel with y_true,
CG-Mamba point forecast, and 95% prediction interval (μ ± 1.96·σ_total).

Output:
  notebooks/figures/forecast_example/forecast_example.pdf + .png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = _ROOT / "runs" / "wis_method_f" / "decomposition_temporal.csv"
OUT_DIR = _ROOT / "notebooks" / "figures" / "forecast_example"


def epiweek_to_date(ep: int) -> pd.Timestamp:
    year = ep // 100
    week = ep % 100
    jan4 = pd.Timestamp(year=year, month=1, day=4)
    iso_week_1_start = jan4 - pd.Timedelta(days=jan4.isoweekday() % 7)
    return iso_week_1_start + pd.Timedelta(weeks=week - 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df["date"] = df.target_ep.apply(epiweek_to_date)
    df["sigma_total"] = np.sqrt(df.sigma2_total)
    df["lo95"] = df.mu_CGM_raw - 1.96 * df.sigma_total
    df["hi95"] = df.mu_CGM_raw + 1.96 * df.sigma_total
    df["lo80"] = df.mu_CGM_raw - 1.28 * df.sigma_total
    df["hi80"] = df.mu_CGM_raw + 1.28 * df.sigma_total

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    horizons = [1, 2, 3, 4]
    horizon_labels = ["1-week ahead", "2-week ahead", "3-week ahead", "4-week ahead"]

    cov95_per_h = {}
    mae_per_h = {}

    for ax, h, hlabel in zip(axes.flat, horizons, horizon_labels):
        sub = df[df.horizon == h].sort_values("date").reset_index(drop=True)
        inside = ((sub.y_raw >= sub.lo95) & (sub.y_raw <= sub.hi95)).mean()
        cov95_per_h[h] = inside
        mae_per_h[h] = (sub.y_raw - sub.mu_CGM_raw).abs().mean()

        ax.fill_between(sub.date, sub.lo95, sub.hi95,
                         color="#1f77b4", alpha=0.18, label="CG-Mamba 95% PI")
        ax.fill_between(sub.date, sub.lo80, sub.hi80,
                         color="#1f77b4", alpha=0.30, label="CG-Mamba 80% PI")
        ax.plot(sub.date, sub.mu_CGM_raw, color="#1f77b4", linewidth=1.6,
                label="CG-Mamba forecast (μ)")
        ax.plot(sub.date, sub.y_raw, color="black", linewidth=1.4,
                label="Observed ILI (ground truth)", alpha=0.9)

        # Highlight 2024-25 season (a wide season with a clear peak)
        season_start = pd.Timestamp("2024-08-04")
        season_end = pd.Timestamp("2025-08-31")
        ax.axvspan(season_start, season_end, color="orange", alpha=0.07,
                    zorder=0)

        ax.set_title(f"h = {h} ({hlabel}) — Empirical Cov95 = {inside:.3f}, MAE = {mae_per_h[h]:.3f}",
                     fontsize=10, loc="left")
        ax.set_ylabel("Weighted ILI (%)" if ax.get_subplotspec().is_first_col() else "",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))
        ax.tick_params(axis="x", which="major", labelsize=9)
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Date (MMWR week)", fontsize=10)

    # One unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10,
                bbox_to_anchor=(0.5, 0.965))

    fig.suptitle(
        "Figure 5 — CG-Mamba qualitative forecast on test_strict (seed=42, W40-2022 ~ W35-2025)\n"
        "Predicted distribution from Method F decomposable uncertainty "
        "(orange shaded: 2024-25 season for visual reference)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    pdf = OUT_DIR / "forecast_example.pdf"
    png = OUT_DIR / "forecast_example.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"       {png}")

    print("\n=== Per-horizon empirical metrics on test_strict ===")
    for h in horizons:
        print(f"  h={h}: empirical Cov95 = {cov95_per_h[h]:.3f}  MAE = {mae_per_h[h]:.3f}")


if __name__ == "__main__":
    main()
