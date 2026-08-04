"""scripts/e1_final_tighten.py — 3 조임 재집계 (paper Stage 5 전 카드 강도 정밀화)

조임 1: per-seed test_strict national n2_d128 vs n3_d64 (MAE / WIS / Cov95) Welch
조임 2: PC2-a per-region phase Cov95 / s_h Cov95 수준 + WIS 비교 (방향 검증)
조임 3: clean n3_d64 vs paper leaky full × 5 seeds (WIS / Cov95 / MAE) Welch
"""
from __future__ import annotations
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

E1_DIR = _ROOT / "runs/e1_final"
AR_DIR = _ROOT / "runs/ablation_retrain"
SEEDS = [42, 123, 456, 789, 1024]
HORIZONS = [1, 2, 3, 4]

PC2A_FLUSIGHT_23 = np.array([
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 0.975, 0.99
])
_LO_IDX = int(np.where(np.isclose(PC2A_FLUSIGHT_23, 0.025))[0][0])
_HI_IDX = int(np.where(np.isclose(PC2A_FLUSIGHT_23, 0.975))[0][0])


def cov95_wis(mu, s2, y):
    """T5 (2026-06-21): redirect to single source of truth — numerically identical."""
    from src.eval.wis_standard import cov95_wis_from_gaussian
    return cov95_wis_from_gaussian(mu, s2, y)


def welch_t(a, b):
    """Two-sample Welch t-test. Returns (Δ=mean(a)-mean(b), t, p_two_sided, df_approx)."""
    na, nb = len(a), len(b)
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va/na + vb/nb)
    t = (ma - mb) / se if se > 0 else 0.0
    # Welch-Satterthwaite df
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1)) if se > 0 else 0
    # two-sided p (use normal approx since df>=4 sufficient for small n)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return ma - mb, t, p, df


def per_seed_from_pool(df_pool: pd.DataFrame, n_per_seed: int):
    """Split pooled parquet (concat order SEEDS) into 5 per-seed slices."""
    return [df_pool.iloc[i*n_per_seed:(i+1)*n_per_seed].reset_index(drop=True)
            for i in range(len(SEEDS))]


def compute_per_seed_metrics(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    n_per_seed = len(df) // len(SEEDS)
    per_seed = per_seed_from_pool(df, n_per_seed)
    out = []
    for s_idx, df_s in enumerate(per_seed):
        mu = df_s.mu.to_numpy()
        s2 = df_s.s2_total.to_numpy()
        y  = df_s.y_true.to_numpy()
        mae = float(np.abs(mu - y).mean())
        cov, wis = cov95_wis(mu, s2, y)
        out.append(dict(seed=SEEDS[s_idx], mae=mae, wis=wis, cov95=cov, n=len(df_s)))
    return out


def read_paper_full_per_seed():
    """Paper ablation_retrain.full 5 seeds: read summary.csv 'full' rows."""
    rows = []
    with (AR_DIR / "ablation_retrain_summary.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row["ablation"] == "full":
                rows.append(dict(seed=int(row["seed"]),
                                  mae=float(row["mae_avg"]),
                                  wis=float(row["wis_avg"]),
                                  cov95=float(row["cov95_avg"])))
    return sorted(rows, key=lambda x: x["seed"])


def main():
    print("=" * 80)
    print("# E1 Stage 5 카드 조임 — paper §V 재작성 전 정밀화")
    print("=" * 80)

    # ──────────────── 조임 1 ────────────────
    print("\n## 조임 1 — test_strict national 5-seed: n2_d128 vs n3_d64 Welch\n")
    n2_per = compute_per_seed_metrics(E1_DIR / "n2_d128_test_strict_national.parquet")
    n3_per = compute_per_seed_metrics(E1_DIR / "n3_d64_test_strict_national.parquet")

    print(f"  {'seed':>5s} | {'n2_d128 MAE':>12s} | {'n3_d64 MAE':>11s} | "
          f"{'n2 WIS':>8s} | {'n3 WIS':>8s} | {'n2 Cov':>7s} | {'n3 Cov':>7s}")
    for s2_r, s3_r in zip(n2_per, n3_per):
        print(f"  {s2_r['seed']:>5d} | {s2_r['mae']:>12.4f} | {s3_r['mae']:>11.4f} | "
              f"{s2_r['wis']:>8.4f} | {s3_r['wis']:>8.4f} | "
              f"{s2_r['cov95']:>7.4f} | {s3_r['cov95']:>7.4f}")

    for metric_name in ("mae", "wis", "cov95"):
        a = [r[metric_name] for r in n2_per]
        b = [r[metric_name] for r in n3_per]
        d, t, p, df = welch_t(a, b)
        ma, mb = st.mean(a), st.mean(b)
        sa, sb = st.stdev(a), st.stdev(b)
        winner = "n2_d128" if d < 0 else "n3_d64"
        if metric_name == "cov95":
            winner = "closer to 0.95: " + ("n2_d128" if abs(ma-0.95) < abs(mb-0.95) else "n3_d64")
        print(f"\n  {metric_name.upper()}:  n2_d128={ma:.4f}±{sa:.4f}  n3_d64={mb:.4f}±{sb:.4f}  "
              f"Δ={d:+.4f}  t={t:+.2f}  p={p:.3f}  df={df:.1f}  → {winner}")

    # ──────────────── 조임 2 ────────────────
    print("\n## 조임 2 — PC2-a per-region phase/s_h 수준 + WIS 방향 검증\n")
    d_eval = json.load(open(E1_DIR / "e1_final_eval.json"))
    pc2a = d_eval["pc2a_recheck_n2_d128"]
    per_region = pc2a["per_region"]
    s_h = pc2a["s_h_per_horizon"]

    print(f"  s_h per horizon (held-out fit): {[f'{s:.3f}' for s in s_h]}")
    print(f"\n  {'region':>7s} | {'cov_phase':>9s} | {'cov_sh':>7s} | {'Δcov':>7s} | "
          f"{'wis_phase':>9s} | {'wis_sh':>7s} | {'Δwis':>7s}")
    cov_p_list, cov_s_list = [], []
    wis_p_list, wis_s_list = [], []
    for r in sorted(per_region):
        d = per_region[r]
        cov_p_list.append(d["cov_phase"]); cov_s_list.append(d["cov_sh"])
        wis_p_list.append(d["wis_phase"]); wis_s_list.append(d["wis_sh"])
        print(f"  {r:>7s} | {d['cov_phase']:>9.4f} | {d['cov_sh']:>7.4f} | "
              f"{d['dcov']:>+7.4f} | {d['wis_phase']:>9.4f} | {d['wis_sh']:>7.4f} | {d['dwis']:>+7.4f}")
    print(f"\n  mean phase Cov95 across regions: {st.mean(cov_p_list):.4f}  "
          f"(distance from 0.95: {abs(st.mean(cov_p_list)-0.95):.4f})")
    print(f"  mean s_h   Cov95 across regions: {st.mean(cov_s_list):.4f}  "
          f"(distance from 0.95: {abs(st.mean(cov_s_list)-0.95):.4f})")
    print(f"  mean phase WIS:  {st.mean(wis_p_list):.4f}")
    print(f"  mean s_h   WIS:  {st.mean(wis_s_list):.4f}")

    closer_to_95 = "phase" if abs(st.mean(cov_p_list)-0.95) < abs(st.mean(cov_s_list)-0.95) else "s_h"
    winner_wis = "phase (lower WIS)" if st.mean(wis_p_list) < st.mean(wis_s_list) else "s_h (lower WIS)"
    print(f"\n  → Cov95 가까운: {closer_to_95}  /  WIS 우위: {winner_wis}")
    print(f"  → paper bootstrap: ΔCov95={pc2a['bootstrap']['dcov']['mean']:+.4f} "
          f"CI[{pc2a['bootstrap']['dcov']['lo']:+.4f},{pc2a['bootstrap']['dcov']['hi']:+.4f}] "
          f"excludes_0={pc2a['bootstrap']['dcov']['excludes_0']}")
    print(f"  → paper bootstrap: ΔWIS  ={pc2a['bootstrap']['dwis']['mean']:+.4f} "
          f"CI[{pc2a['bootstrap']['dwis']['lo']:+.4f},{pc2a['bootstrap']['dwis']['hi']:+.4f}] "
          f"excludes_0={pc2a['bootstrap']['dwis']['excludes_0']}")

    # ──────────────── 조임 3 ────────────────
    print("\n## 조임 3 — clean n3_d64 vs paper leaky full × 5 seeds: Welch (MAE / WIS / Cov95)\n")
    paper_full = read_paper_full_per_seed()
    # Sort our n3_per by seed too
    clean_n3 = sorted(n3_per, key=lambda x: x["seed"])

    print(f"  {'seed':>5s} | {'leaky MAE':>10s} | {'clean MAE':>10s} | "
          f"{'leaky WIS':>10s} | {'clean WIS':>10s} | {'leaky Cov':>10s} | {'clean Cov':>10s}")
    for pr, cr in zip(paper_full, clean_n3):
        assert pr["seed"] == cr["seed"], f"seed mismatch: paper={pr['seed']}, clean={cr['seed']}"
        print(f"  {pr['seed']:>5d} | {pr['mae']:>10.4f} | {cr['mae']:>10.4f} | "
              f"{pr['wis']:>10.4f} | {cr['wis']:>10.4f} | "
              f"{pr['cov95']:>10.4f} | {cr['cov95']:>10.4f}")

    for metric_name in ("mae", "wis", "cov95"):
        a_leaky = [r[metric_name] for r in paper_full]
        b_clean = [r[metric_name] for r in clean_n3]
        d, t, p, df = welch_t(a_leaky, b_clean)
        ml, mc = st.mean(a_leaky), st.mean(b_clean)
        sl, sc = st.stdev(a_leaky), st.stdev(b_clean)
        verdict = "clean ≈ leaky" if p > 0.05 else "DIFFERENT (clean ≠ leaky)"
        print(f"\n  {metric_name.upper()}:  paper leaky={ml:.4f}±{sl:.4f}  clean={mc:.4f}±{sc:.4f}  "
              f"Δ={d:+.4f}  t={t:+.2f}  p={p:.3f}  → {verdict}")

    # Save aggregated result
    out = {
        "tightening_1_n2_vs_n3_test_strict": {
            "per_seed_n2_d128": n2_per,
            "per_seed_n3_d64": n3_per,
        },
        "tightening_2_pc2a_level_check": {
            "s_h_per_horizon": s_h,
            "per_region_summary": per_region,
            "mean_phase_cov": st.mean(cov_p_list),
            "mean_sh_cov": st.mean(cov_s_list),
            "mean_phase_wis": st.mean(wis_p_list),
            "mean_sh_wis": st.mean(wis_s_list),
        },
        "tightening_3_clean_vs_leaky_n3_d64": {
            "paper_leaky_per_seed": paper_full,
            "clean_per_seed": clean_n3,
        },
    }
    with (E1_DIR / "e1_final_tightening.json").open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: runs/e1_final/e1_final_tightening.json")


if __name__ == "__main__":
    main()
