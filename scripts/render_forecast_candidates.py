"""Render Figure 5 candidates: 5 per-seed trajectories + 5-seed mean overlay.

Reads runs/wis_method_f/decomposition_temporal_5seed.csv (after wis_method_f.py re-run).
Outputs 6 candidate PDFs to runs/figures/forecast_candidates/ for user selection.
Prints per-seed Cov95 h=1 + identifies representative seed (closest to 0.889 headline).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
CSV = _ROOT / "runs/wis_method_f/decomposition_temporal_5seed.csv"
OUT_DIR = _ROOT / "runs/figures/forecast_candidates"


def epiweek_to_date(ep: int) -> pd.Timestamp:
    y, w = ep // 100, ep % 100
    jan4 = pd.Timestamp(year=y, month=1, day=4)
    iso1 = jan4 - pd.Timedelta(days=jan4.isoweekday() % 7)
    return iso1 + pd.Timedelta(weeks=w - 1)


def render_figure(df_sub, title, out_stem):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    horizons = [1, 2, 3, 4]
    labels = ["1-week ahead", "2-week ahead", "3-week ahead", "4-week ahead"]
    cov95s = {}
    for ax, h, hl in zip(axes.flat, horizons, labels):
        sub = df_sub[df_sub.horizon == h].sort_values("date").reset_index(drop=True)
        inside = ((sub.y_raw >= sub.lo95) & (sub.y_raw <= sub.hi95)).mean()
        mae = (sub.y_raw - sub.mu_CGM_raw).abs().mean()
        cov95s[h] = inside
        ax.fill_between(sub.date, sub.lo95, sub.hi95, color="#1f77b4", alpha=0.18, label="CG-Mamba 95% PI")
        ax.fill_between(sub.date, sub.lo80, sub.hi80, color="#1f77b4", alpha=0.30, label="CG-Mamba 80% PI")
        ax.plot(sub.date, sub.mu_CGM_raw, color="#1f77b4", linewidth=1.6, label="CG-Mamba forecast (μ)")
        ax.plot(sub.date, sub.y_raw, color="black", linewidth=1.4, label="Observed ILI (ground truth)", alpha=0.9)
        ax.axvspan(pd.Timestamp("2024-08-04"), pd.Timestamp("2025-08-31"), color="orange", alpha=0.07, zorder=0)
        ax.set_title(f"h = {h} ({hl}) — Empirical Cov95 = {inside:.3f}, MAE = {mae:.3f}", fontsize=10, loc="left")
        ax.set_ylabel("Weighted ILI (%)" if ax.get_subplotspec().is_first_col() else "", fontsize=10)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Date (MMWR week)", fontsize=10)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{out_stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return cov95s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    df["date"] = df.target_ep.apply(epiweek_to_date)
    df["sigma_total"] = np.sqrt(df.sigma2_total)
    df["lo95"] = df.mu_CGM_raw - 1.96 * df.sigma_total
    df["hi95"] = df.mu_CGM_raw + 1.96 * df.sigma_total
    df["lo80"] = df.mu_CGM_raw - 1.28 * df.sigma_total
    df["hi80"] = df.mu_CGM_raw + 1.28 * df.sigma_total

    seeds = sorted(df["seed"].unique())
    print(f"Loaded {len(df)} rows across {len(seeds)} seeds: {seeds}")

    # Render per-seed figures
    per_seed_cov = {}
    for s in seeds:
        df_s = df[df.seed == s].copy()
        cov_h1 = (
            (df_s[df_s.horizon == 1].y_raw >= df_s[df_s.horizon == 1].lo95)
            & (df_s[df_s.horizon == 1].y_raw <= df_s[df_s.horizon == 1].hi95)
        ).mean()
        per_seed_cov[s] = cov_h1
        title = f"Figure 5 — CG-Mamba forecast on test_strict (seed={s})\nMethod F decomposable uncertainty (Cov95 h=1 for this seed: {cov_h1:.3f})"
        render_figure(df_s, title, OUT_DIR / f"forecast_seed{s}")

    # 5-seed mean overlay (single trajectory = avg μ, avg σ across seeds)
    df_mean = df.groupby(["horizon", "target_ep"]).agg(
        mu_CGM_raw=("mu_CGM_raw", "mean"),
        y_raw=("y_raw", "mean"),
        sigma_total=("sigma_total", "mean"),
    ).reset_index()
    df_mean["date"] = df_mean.target_ep.apply(epiweek_to_date)
    df_mean["lo95"] = df_mean.mu_CGM_raw - 1.96 * df_mean.sigma_total
    df_mean["hi95"] = df_mean.mu_CGM_raw + 1.96 * df_mean.sigma_total
    df_mean["lo80"] = df_mean.mu_CGM_raw - 1.28 * df_mean.sigma_total
    df_mean["hi80"] = df_mean.mu_CGM_raw + 1.28 * df_mean.sigma_total

    cov_mean = (
        (df_mean[df_mean.horizon == 1].y_raw >= df_mean[df_mean.horizon == 1].lo95)
        & (df_mean[df_mean.horizon == 1].y_raw <= df_mean[df_mean.horizon == 1].hi95)
    ).mean()
    title_mean = (
        f"Figure 5 — CG-Mamba forecast on test_strict (5-seed mean overlay)\n"
        f"Method F decomposable uncertainty (mean μ / mean σ across 5 seeds; Cov95 h=1 = {cov_mean:.3f})"
    )
    render_figure(df_mean, title_mean, OUT_DIR / "forecast_5seed_mean")

    print("\n=== Per-seed Cov95 h=1 summary ===")
    print(f"Target (5-seed mean from Table I): 0.889\n")
    print(f"{'seed':>6} {'Cov95 h=1':>12} {'|Δ to 0.889|':>14}")
    print("-" * 38)
    headline = 0.889
    best_seed = min(seeds, key=lambda s: abs(per_seed_cov[s] - headline))
    for s in seeds:
        d = abs(per_seed_cov[s] - headline)
        marker = "  ← representative" if s == best_seed else ""
        print(f"  {s:>4} {per_seed_cov[s]:>12.3f} {d:>14.3f}{marker}")
    print(f"\n5-seed mean overlay Cov95 h=1 = {cov_mean:.3f}")
    print(f"\nOutput PDFs at: {OUT_DIR.relative_to(_ROOT)}/")
    print("  forecast_seed{42,123,456,789,1024}.pdf (5 candidates)")
    print("  forecast_5seed_mean.pdf (mean overlay candidate)")


if __name__ == "__main__":
    main()
