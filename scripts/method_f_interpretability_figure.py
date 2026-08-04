"""§V.X Interpretability figure — 3-component decomposition over test_strict period.

Loads runs/wis_method_f/decomposition_temporal.csv (seed42 test_strict period
W40-2022 ~ W35-2025, 152 weeks × 4 horizons).

Produces 4-row stack plot:
  Row 1: y_true vs μ_CGM vs μ_HMM (raw ILI %)
  Row 2: σ²_total breakdown (within / between_HMM / bias² stacked)
  Row 3: Fraction of σ²_total per component (proportional view)
  Row 4: Component-specific signals (interpretability)

Save: runs/wis_method_f/interpretability_figure.png + .pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]

CSV_PATH = _ROOT / "runs" / "wis_method_f" / "decomposition_temporal.csv"
OUT_DIR = _ROOT / "runs" / "wis_method_f"


def epiweek_to_date(ep: int) -> pd.Timestamp:
    """Convert epiweek (YYYYWW) to first day of MMWR week (Sunday)."""
    year = ep // 100
    week = ep % 100
    # MMWR week: ISO week-like but starts Sunday. Approximation via ISO + shift.
    jan4 = pd.Timestamp(year=year, month=1, day=4)
    # Find Sunday of that ISO week
    iso_week_1_start = jan4 - pd.Timedelta(days=jan4.isoweekday() % 7)
    return iso_week_1_start + pd.Timedelta(weeks=week - 1)


def main():
    df = pd.read_csv(CSV_PATH)
    # Focus on h=1 for cleanest signal
    df_h1 = df[df.horizon == 1].sort_values("target_ep").reset_index(drop=True)
    df_h1["date"] = df_h1.target_ep.apply(epiweek_to_date)
    df_h1["sigma_total"] = np.sqrt(df_h1.sigma2_total)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1.2, 1.5]})

    # ─── Row 1: predictions vs truth ───
    ax = axes[0]
    ax.plot(df_h1.date, df_h1.y_raw, "k-", linewidth=1.4, label="y_true (ILI %wILI)", alpha=0.8)
    ax.plot(df_h1.date, df_h1.mu_CGM_raw, "b-", linewidth=1.2, label="μ_CGM (CG-Mamba pred)", alpha=0.7)
    ax.plot(df_h1.date, df_h1.mu_HMM_raw, "r--", linewidth=1.0, label="μ_HMM (HMM mixture mean)", alpha=0.6)
    ax.fill_between(df_h1.date,
                    df_h1.mu_CGM_raw - 1.96 * df_h1.sigma_total,
                    df_h1.mu_CGM_raw + 1.96 * df_h1.sigma_total,
                    alpha=0.15, color="blue", label="μ_CGM ± 1.96σ_total")
    ax.set_ylabel("ILI %wILI", fontsize=10)
    ax.set_title("Row 1: Predictions vs. truth (h=1 forecast on test_strict W40-2022 ~ W35-2025)",
                 fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ─── Row 2: σ² components stacked ───
    ax = axes[1]
    ax.fill_between(df_h1.date, 0, df_h1.sigma2_within,
                    color="#2ca02c", alpha=0.7, label="σ²_within (aleatoric per-phase)")
    ax.fill_between(df_h1.date, df_h1.sigma2_within,
                    df_h1.sigma2_within + df_h1.sigma2_between_HMM,
                    color="#ff7f0e", alpha=0.7, label="σ²_between (phase uncertainty)")
    ax.fill_between(df_h1.date,
                    df_h1.sigma2_within + df_h1.sigma2_between_HMM,
                    df_h1.sigma2_within + df_h1.sigma2_between_HMM + df_h1.bias_sq,
                    color="#d62728", alpha=0.5, label="bias² (CG-Mamba refinement)")
    ax.set_ylabel("σ² (z-score² units)", fontsize=10)
    ax.set_title("Row 2: 3-component decomposition (stacked) — σ²_total = within + between + bias²",
                 fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ─── Row 3: fraction view ───
    ax = axes[2]
    total = df_h1.sigma2_within + df_h1.sigma2_between_HMM + df_h1.bias_sq + 1e-12
    f_within = df_h1.sigma2_within / total
    f_between = df_h1.sigma2_between_HMM / total
    f_bias = df_h1.bias_sq / total
    ax.fill_between(df_h1.date, 0, f_within,
                    color="#2ca02c", alpha=0.7)
    ax.fill_between(df_h1.date, f_within, f_within + f_between,
                    color="#ff7f0e", alpha=0.7)
    ax.fill_between(df_h1.date, f_within + f_between,
                    f_within + f_between + f_bias,
                    color="#d62728", alpha=0.5)
    ax.set_ylabel("Fraction of σ²_total", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Row 3: Per-component fraction (interpretability signal)",
                 fontsize=10, loc="left")
    ax.grid(alpha=0.3)

    # ─── Row 4: anomaly signal (bias² in raw units) ───
    ax = axes[3]
    bias_raw = np.sqrt(df_h1.bias_sq) * 1.76  # target_std ≈ 1.76
    bias_threshold = np.quantile(bias_raw, 0.90)
    ax.fill_between(df_h1.date, 0, bias_raw, color="#d62728", alpha=0.5,
                    label="|μ_HMM - μ_CGM| (anomaly signal, raw ILI %)")
    ax.axhline(bias_threshold, color="black", linestyle="--", linewidth=1,
               label=f"90th percentile = {bias_threshold:.2f} (anomaly flag threshold)")
    ax.set_ylabel("|μ_HMM - μ_CGM| (ILI %)", fontsize=10)
    ax.set_xlabel("Date (MMWR week)", fontsize=10)
    ax.set_title("Row 4: CG-Mamba refinement signal (bias¹/²) — anomaly attribution",
                 fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Date formatting on x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))
    ax.tick_params(axis='x', which='major', labelsize=9)

    plt.suptitle("CG-Mamba Method F: Decomposable Uncertainty for Clinical Interpretability\n"
                 "(test_strict period, seed=42, h=1 forecast)",
                 fontsize=12, y=0.995)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_png = OUT_DIR / "interpretability_figure.png"
    fig_pdf = OUT_DIR / "interpretability_figure.pdf"
    plt.savefig(fig_png, dpi=180, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    print(f"Saved: {fig_png.relative_to(_ROOT)}")
    print(f"Saved: {fig_pdf.relative_to(_ROOT)}")

    # Statistics for paper caption
    print("\n=== Figure caption stats ===")
    print(f"  Period: W40-2022 to W35-2025 ({len(df_h1)} weeks)")
    print(f"  Mean fractions: within {f_within.mean():.1%}, "
          f"between {f_between.mean():.1%}, bias {f_bias.mean():.1%}")
    print(f"  bias² (raw ILI %) range: {bias_raw.min():.2f} ~ {bias_raw.max():.2f}")
    print(f"  Anomaly threshold (90th pct): {bias_threshold:.2f} %wILI")
    n_anomaly = (bias_raw > bias_threshold).sum()
    print(f"  Anomaly weeks (above 90th pct): {n_anomaly}/{len(df_h1)} = {n_anomaly/len(df_h1):.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
