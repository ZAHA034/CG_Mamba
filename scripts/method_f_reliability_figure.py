"""Generate Fig 4: Method F decomposition + Reliability diagram (2-panel).

Panel A: Stacked area chart of Method F variance components over time.
Panel B: Reliability diagram (predicted vs empirical coverage) for CG-Mamba
         Method F across 23 FluSight quantile levels, with reference points
         for SARIMA Kalman and MC Dropout d=0.1 at Cov95.

Output: runs/figures/method_f_reliability_figure.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "runs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# 23 FluSight quantile levels (Bracher et al. 2021)
FLUSIGHT_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99
])


def epiweek_to_date(ep: int) -> pd.Timestamp:
    year = ep // 100
    week = ep % 100
    jan4 = pd.Timestamp(year=year, month=1, day=4)
    iso_week_1_start = jan4 - pd.Timedelta(days=jan4.isoweekday() % 7)
    return iso_week_1_start + pd.Timedelta(weeks=week - 1)


def compute_reliability(mu: np.ndarray, sigma_total: np.ndarray, y_true: np.ndarray,
                         s_scale: float, quantile_levels: np.ndarray):
    """Compute empirical coverage at each predicted quantile level.

    For each quantile level alpha, predicted_q_alpha = mu + z_alpha * sqrt(s * sigma2_total).
    Empirical coverage = fraction of y_true <= predicted_q_alpha.
    For perfect calibration, this should equal alpha.
    """
    sigma_eff = np.sqrt(s_scale) * sigma_total
    emp_cov = np.zeros(len(quantile_levels))
    for i, alpha in enumerate(quantile_levels):
        z = norm.ppf(alpha)
        q = mu + z * sigma_eff
        emp_cov[i] = (y_true <= q).mean()
    return emp_cov


def main():
    # Load decomposition data
    df = pd.read_csv(_ROOT / "runs/wis_method_f/decomposition_temporal.csv")

    # Load calibration scale s_per_h from val
    with open(_ROOT / "runs/wis_method_f/wis_results.json") as f:
        results = json.load(f)
    s_per_h_first = results['per_seed']['42']['splits']['test_strict']['calibration_meta']['s_per_h']
    print(f"Calibration s_per_h = {s_per_h_first}")

    # ── Panel A: Stacked variance components over time (h=1 forecast) ──
    df_h1 = df[df.horizon == 1].sort_values('target_ep').reset_index(drop=True)
    df_h1["date"] = df_h1.target_ep.apply(epiweek_to_date)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2),
                                     gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Stacked components
    axA.fill_between(df_h1.date, 0, df_h1.sigma2_within,
                      color="#2ca02c", alpha=0.75,
                      label=r"$\sigma^2_{within}$ (aleatoric, per-phase noise)")
    axA.fill_between(df_h1.date, df_h1.sigma2_within,
                      df_h1.sigma2_within + df_h1.sigma2_between_HMM,
                      color="#ff7f0e", alpha=0.75,
                      label=r"$\sigma^2_{between}$ (phase identifiability)")
    axA.fill_between(df_h1.date,
                      df_h1.sigma2_within + df_h1.sigma2_between_HMM,
                      df_h1.sigma2_within + df_h1.sigma2_between_HMM + df_h1.bias_sq,
                      color="#d62728", alpha=0.55,
                      label=r"bias$^2$ (model refinement)")

    # Overlay actual ILI on secondary axis
    axA_r = axA.twinx()
    axA_r.plot(df_h1.date, df_h1.y_raw, "k-", linewidth=1.2, alpha=0.8,
                label="actual %wILI (right axis)")
    axA_r.set_ylabel("Actual %wILI", fontsize=10)
    axA_r.legend(loc="upper right", fontsize=8)

    axA.set_ylabel(r"Variance ($z$-scored$^2$)", fontsize=10)
    axA.set_title("(A) HMM-derived variance decomposition over test_strict period (h=1 forecast)",
                   fontsize=11, loc="left")
    axA.legend(loc="upper left", fontsize=8)
    axA.grid(alpha=0.3)
    axA.xaxis.set_major_locator(mdates.YearLocator())
    axA.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axA.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))
    axA.tick_params(axis='x', which='major', labelsize=9)

    # ── Panel B: Per-horizon Cov95 bar chart (M2.3 verified 5-seed means) ──
    # All values from M2.3 final table (5-seed mean) and M2.3 per-horizon Cov95
    # Data sources: runs/M2_3_final_table.csv (aggregate) + per-horizon from m2_4_wis_protocol.csv
    horizons = ["h=1", "h=2", "h=3", "h=4", "avg"]
    n_h = len(horizons)
    x = np.arange(n_h)
    width = 0.155

    # Per-horizon Cov95 values verified from runs/m2_4_data_efficiency/m2_4_wis_protocol.csv
    # (5-seed mean for DL baselines, n=1 deterministic for SARIMA)
    cgm_methodf  = [0.932, 0.914, 0.901, 0.898, 0.911]  # CG-Mamba Method F
    sarima       = [0.934, 0.862, 0.849, 0.803, 0.862]  # SARIMA Kalman parametric
    vmamba_mc    = [0.936, 0.879, 0.784, 0.682, 0.820]  # Vanilla Mamba MC d=0.2
    patchtst_mc  = [0.745, 0.717, 0.670, 0.655, 0.697]  # PatchTST MC d=0.1
    lstm_mc      = [0.928, 0.713, 0.554, 0.443, 0.659]  # LSTM MC d=0.3
    # CG-Mamba MC d=0.1 only available as aggregate in M2.3 (Cov95=0.236, FM-2 mismatch)
    cgm_mc_agg = 0.236

    axB.bar(x - 2*width, cgm_methodf, width, color="#1a6faf",
            label="CG-Mamba (HMM-derived)")
    axB.bar(x - 1*width, sarima,      width, color="#444444",
            label="SARIMA Kalman parametric")
    axB.bar(x + 0*width, vmamba_mc,   width, color="#ff7f0e",
            label="Vanilla Mamba (MC d=0.2)")
    axB.bar(x + 1*width, patchtst_mc, width, color="#2ca02c",
            label="PatchTST (MC d=0.1)")
    axB.bar(x + 2*width, lstm_mc,     width, color="#9467bd",
            label="LSTM (MC d=0.3)")

    # Add FM-2 reference (CG-Mamba MC d=0.1, aggregate only) as horizontal line
    axB.axhline(cgm_mc_agg, color="#d62728", linestyle=":", linewidth=1.6,
                 alpha=0.85,
                 label=f"CG-Mamba MC d=0.1 aggregate (FM-2): {cgm_mc_agg:.3f}")

    axB.axhline(0.95, color="black", linestyle="--", linewidth=1.0,
                 alpha=0.7, label="Nominal 95%")
    axB.set_xticks(x)
    axB.set_xticklabels(horizons)
    axB.set_xlabel("Forecast horizon", fontsize=10)
    axB.set_ylabel("Empirical Cov95", fontsize=10)
    axB.set_title("(B) Per-horizon Cov95 — HMM-derived vs alternative UQ (test_strict, 5-seed mean)",
                   fontsize=11, loc="left")
    axB.set_ylim(0, 1.05)
    axB.legend(loc="lower left", fontsize=7.5, framealpha=0.95)
    axB.grid(alpha=0.3, axis="y")

    plt.suptitle("Figure 2: CG-Mamba HMM-derived Decomposable Uncertainty + Calibration Diagnostic",
                  fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    pdf_path = OUT / "method_f_reliability_figure.pdf"
    png_path = OUT / "method_f_reliability_figure.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {pdf_path.relative_to(_ROOT)}")
    print(f"Saved: {png_path.relative_to(_ROOT)}")

    # Diagnostic printing (no reliability curve in new Panel B design)
    print()
    print("=== Verified Cov95 data sources ===")
    print(f"  CG-Mamba Method F per-h: {cgm_methodf}")
    print(f"  SARIMA Kalman per-h:    {sarima}")
    print(f"  Vanilla Mamba MC per-h: {vmamba_mc}")
    print(f"  PatchTST MC per-h:      {patchtst_mc}")
    print(f"  LSTM MC per-h:          {lstm_mc}")
    print(f"  M2.3 aggregate Cov95 (5-seed mean):")
    print(f"    CG-Mamba Method F: 0.889")
    print(f"    CG-Mamba MC d=0.1: 0.236 (FM-2 mismatch)")


if __name__ == "__main__":
    main()
