"""scripts/p1_card1_cal_retest.py — refined (D) P1: 카드 1 의 CAL-WIS + CAL-Cov95 재검정

기존 tightening 1 = RAW WIS 위 검정 (RAW = diagnostic only 확정 후 invalid).
P1 = CAL-WIS + CAL-Cov95 (nominal 0.95 거리) per-seed Welch 재검정.

또한 Method F sign check: HeteroHead vs Method F 두 calibrator 에서 sign 가 유지되나.
유지 안 되면 within-noise → 카드 강도 조정 필요.

T5 anchor: src.eval.wis_standard (single source of truth).
"""
from __future__ import annotations
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.eval.wis_standard import (
    FLUSIGHT_23,
    cov95_wis_from_gaussian,
    calibrate_s_h,
    quantiles_method_f_calibrated,
    wis,
    coverage,
)

E1_DIR = _ROOT / "runs/e1_final"
SEEDS = [42, 123, 456, 789, 1024]
HORIZONS = [1, 2, 3, 4]


def welch_t(a, b):
    na, nb = len(a), len(b)
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va/na + vb/nb)
    t = (ma - mb) / se if se > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return ma - mb, t, p


def per_seed_split(df_pool, n_per_seed):
    return [df_pool.iloc[i*n_per_seed:(i+1)*n_per_seed].reset_index(drop=True)
            for i in range(len(SEEDS))]


def apply_hetero_head(df_s, alpha_h, beta_h):
    """Apply cal head α_h · sw + β_h · sb to per-row → return (mu, s2_cal, y, h_arr)."""
    mu = df_s.mu.to_numpy()
    sw = df_s.s2_within.to_numpy()
    sb = df_s.s2_between.to_numpy()
    y = df_s.y_true.to_numpy()
    h_arr = df_s.horizon.to_numpy()
    s2_cal = np.empty_like(mu)
    for h_idx, h in enumerate(HORIZONS):
        idx = h_arr == h
        s2_cal[idx] = alpha_h[h_idx] * sw[idx] + beta_h[h_idx] * sb[idx]
    return mu, s2_cal, y, h_arr


def main():
    print("=" * 80)
    print("# P1 — 카드 1 CAL-WIS + CAL-Cov95 재검정 (단일 source = wis_standard)")
    print("=" * 80)

    # ----- HeteroHead cal params from e1_final_eval.json -----
    e1_eval = json.load(open(E1_DIR / "e1_final_eval.json"))
    n2_cal = e1_eval["n2_d128"]["test_strict_national"]["calibrated"]
    n3_cal = e1_eval["n3_d64"]["test_strict_national"]["calibrated"]
    n2_alpha = np.array(n2_cal["alpha"]); n2_beta = np.array(n2_cal["beta"])
    n3_alpha = np.array(n3_cal["alpha"]); n3_beta = np.array(n3_cal["beta"])
    print(f"\nHeteroHead cal params (from e1_final_eval.json):")
    print(f"  n2_d128: α={[f'{a:.3f}' for a in n2_alpha]}  β={[f'{b:.3f}' for b in n2_beta]}")
    print(f"  n3_d64 : α={[f'{a:.3f}' for a in n3_alpha]}  β={[f'{b:.3f}' for b in n3_beta]}")

    # ----- per-seed CAL WIS + CAL Cov95 (HeteroHead) -----
    print("\n## HeteroHead per-seed CAL metrics (test_strict national)")
    n2_pq = pd.read_parquet(E1_DIR / "n2_d128_test_strict_national.parquet")
    n3_pq = pd.read_parquet(E1_DIR / "n3_d64_test_strict_national.parquet")
    n_per = len(n2_pq) // len(SEEDS)
    n2_seeds = per_seed_split(n2_pq, n_per)
    n3_seeds = per_seed_split(n3_pq, n_per)

    n2_results = []
    n3_results = []
    print(f"  {'seed':>5s} | {'n2 cal WIS':>10s} | {'n3 cal WIS':>10s} | "
          f"{'n2 cal Cov95':>12s} | {'n3 cal Cov95':>12s}")
    for i, s in enumerate(SEEDS):
        n2_mu, n2_s2cal, n2_y, _ = apply_hetero_head(n2_seeds[i], n2_alpha, n2_beta)
        n3_mu, n3_s2cal, n3_y, _ = apply_hetero_head(n3_seeds[i], n3_alpha, n3_beta)
        n2_cov, n2_wis = cov95_wis_from_gaussian(n2_mu, n2_s2cal, n2_y)
        n3_cov, n3_wis = cov95_wis_from_gaussian(n3_mu, n3_s2cal, n3_y)
        n2_results.append(dict(seed=s, cal_wis=n2_wis, cal_cov95=n2_cov))
        n3_results.append(dict(seed=s, cal_wis=n3_wis, cal_cov95=n3_cov))
        print(f"  {s:>5d} | {n2_wis:>10.4f} | {n3_wis:>10.4f} | "
              f"{n2_cov:>12.4f} | {n3_cov:>12.4f}")

    # ----- Welch on CAL-WIS + CAL-Cov95 -----
    n2_wis_list = [r["cal_wis"] for r in n2_results]
    n3_wis_list = [r["cal_wis"] for r in n3_results]
    n2_cov_list = [r["cal_cov95"] for r in n2_results]
    n3_cov_list = [r["cal_cov95"] for r in n3_results]

    d_wis, t_wis, p_wis = welch_t(n2_wis_list, n3_wis_list)
    d_cov, t_cov, p_cov = welch_t(n2_cov_list, n3_cov_list)
    # |Cov95 - 0.95| distance
    n2_cov_dist = [abs(c - 0.95) for c in n2_cov_list]
    n3_cov_dist = [abs(c - 0.95) for c in n3_cov_list]
    d_cd, t_cd, p_cd = welch_t(n2_cov_dist, n3_cov_dist)

    print(f"\n## HeteroHead CAL Welch (n2_d128 vs n3_d64)")
    print(f"  CAL WIS:   n2={st.mean(n2_wis_list):.4f}±{st.stdev(n2_wis_list):.4f}  "
          f"n3={st.mean(n3_wis_list):.4f}±{st.stdev(n3_wis_list):.4f}  "
          f"Δ={d_wis:+.4f}  t={t_wis:+.2f}  p={p_wis:.4f}  "
          f"→ winner: {'n2_d128' if d_wis < 0 else 'n3_d64'}")
    print(f"  CAL Cov95: n2={st.mean(n2_cov_list):.4f}±{st.stdev(n2_cov_list):.4f}  "
          f"n3={st.mean(n3_cov_list):.4f}±{st.stdev(n3_cov_list):.4f}  "
          f"Δ={d_cov:+.4f}  p={p_cov:.4f}")
    print(f"  |Cov95-0.95|: n2={st.mean(n2_cov_dist):.4f}±{st.stdev(n2_cov_dist):.4f}  "
          f"n3={st.mean(n3_cov_dist):.4f}±{st.stdev(n3_cov_dist):.4f}  "
          f"Δ={d_cd:+.4f}  p={p_cd:.4f}  "
          f"→ closer to nominal: {'n2_d128' if d_cd < 0 else 'n3_d64'}")

    # ----- Method F sign check (calibrate s_h on held-out, apply on test) -----
    print(f"\n## Method F sign check (calibrate s_h per-seed on held-out)")
    n2_ho_pq = pd.read_parquet(E1_DIR / "n2_d128_held_out_national.parquet")
    n3_ho_pq = pd.read_parquet(E1_DIR / "n3_d64_held_out_national.parquet")
    n_per_ho = len(n2_ho_pq) // len(SEEDS)
    n2_ho_seeds = per_seed_split(n2_ho_pq, n_per_ho)
    n3_ho_seeds = per_seed_split(n3_ho_pq, n_per_ho)

    n2_mf_wis_list, n3_mf_wis_list = [], []
    n2_mf_cov_list, n3_mf_cov_list = [], []
    print(f"  {'seed':>5s} | {'n2 MF WIS':>9s} | {'n3 MF WIS':>9s} | "
          f"{'n2 MF Cov':>9s} | {'n3 MF Cov':>9s} | {'n2 s_h':>23s} | {'n3 s_h':>23s}")
    for i, s in enumerate(SEEDS):
        for cfg_id, ho_pq, test_pq_seed, mf_wis_acc, mf_cov_acc in [
            ('n2', n2_ho_seeds[i], n2_seeds[i], n2_mf_wis_list, n2_mf_cov_list),
            ('n3', n3_ho_seeds[i], n3_seeds[i], n3_mf_wis_list, n3_mf_cov_list),
        ]:
            # Pivot to [N, H]
            ho_w = ho_pq.pivot(index='target_ep', columns='horizon',
                                values=['mu', 's2_total', 'y_true']).dropna()
            mu_val = ho_w['mu'].to_numpy(); s2_val = ho_w['s2_total'].to_numpy()
            y_val = ho_w['y_true'].to_numpy()
            test_w = test_pq_seed.pivot(index='target_ep', columns='horizon',
                                          values=['mu', 's2_total', 'y_true']).dropna()
            mu_te = test_w['mu'].to_numpy(); s2_te = test_w['s2_total'].to_numpy()
            y_te = test_w['y_true'].to_numpy()

            s_h = calibrate_s_h(mu_val, s2_val, y_val)
            qf_te = quantiles_method_f_calibrated(mu_te, s2_te, s_h)
            # Flatten to per-row for cov95 / wis
            qf_flat = {tau: q.reshape(-1) for tau, q in qf_te.items()}
            y_flat = y_te.reshape(-1)
            cov = coverage(y_flat, qf_flat, alpha=0.05)
            wis_v = float(np.mean(wis(y_flat, qf_flat)))
            mf_wis_acc.append(wis_v); mf_cov_acc.append(cov)
            if cfg_id == 'n2':
                s_h_n2 = s_h
            else:
                s_h_n3 = s_h
        print(f"  {s:>5d} | {n2_mf_wis_list[-1]:>9.4f} | {n3_mf_wis_list[-1]:>9.4f} | "
              f"{n2_mf_cov_list[-1]:>9.4f} | {n3_mf_cov_list[-1]:>9.4f} | "
              f"[{','.join(f'{x:.3f}' for x in s_h_n2):>21s}] | "
              f"[{','.join(f'{x:.3f}' for x in s_h_n3):>21s}]")

    d_mf_wis, t_mf_wis, p_mf_wis = welch_t(n2_mf_wis_list, n3_mf_wis_list)
    d_mf_cov, t_mf_cov, p_mf_cov = welch_t(n2_mf_cov_list, n3_mf_cov_list)

    print(f"\n## Method F Welch (n2_d128 vs n3_d64)")
    print(f"  MF WIS:   n2={st.mean(n2_mf_wis_list):.4f}±{st.stdev(n2_mf_wis_list):.4f}  "
          f"n3={st.mean(n3_mf_wis_list):.4f}±{st.stdev(n3_mf_wis_list):.4f}  "
          f"Δ={d_mf_wis:+.4f}  p={p_mf_wis:.4f}  "
          f"→ {'n2_d128' if d_mf_wis < 0 else 'n3_d64'}")
    print(f"  MF Cov95: n2={st.mean(n2_mf_cov_list):.4f}±{st.stdev(n2_mf_cov_list):.4f}  "
          f"n3={st.mean(n3_mf_cov_list):.4f}±{st.stdev(n3_mf_cov_list):.4f}  "
          f"Δ={d_mf_cov:+.4f}  p={p_mf_cov:.4f}")

    # ----- Sign consistency check -----
    he_winner = 'n2_d128' if d_wis < 0 else 'n3_d64'
    mf_winner = 'n2_d128' if d_mf_wis < 0 else 'n3_d64'
    print(f"\n## Sign consistency (calibrator robustness)")
    print(f"  HeteroHead WIS winner: {he_winner}  (p={p_wis:.3f})")
    print(f"  Method F WIS winner:   {mf_winner}  (p={p_mf_wis:.3f})")
    if he_winner == mf_winner:
        print(f"  ✓ SIGN CONSISTENT — 카드 1 의 calibration-우위 결론은 calibrator 무관 robust")
    else:
        print(f"  ⚠ SIGN FLIPS — within-noise → 카드 1 의 WIS leg = noise, MAE만 살아남")

    # Save
    out = {
        "p1_card1_cal_retest_2026-06-21": {
            "hetero_head": {
                "n2_d128_per_seed": n2_results,
                "n3_d64_per_seed": n3_results,
                "welch_cal_wis": {"delta": d_wis, "t": t_wis, "p": p_wis,
                                   "n2_mean": st.mean(n2_wis_list), "n3_mean": st.mean(n3_wis_list)},
                "welch_cal_cov95": {"delta": d_cov, "t": t_cov, "p": p_cov},
                "welch_cov95_distance_from_nominal": {"delta": d_cd, "t": t_cd, "p": p_cd,
                                                       "n2_mean": st.mean(n2_cov_dist),
                                                       "n3_mean": st.mean(n3_cov_dist)},
            },
            "method_f": {
                "n2_d128_per_seed_wis": n2_mf_wis_list,
                "n3_d64_per_seed_wis": n3_mf_wis_list,
                "n2_d128_per_seed_cov95": n2_mf_cov_list,
                "n3_d64_per_seed_cov95": n3_mf_cov_list,
                "welch_wis": {"delta": d_mf_wis, "t": t_mf_wis, "p": p_mf_wis,
                                "n2_mean": st.mean(n2_mf_wis_list), "n3_mean": st.mean(n3_mf_wis_list)},
                "welch_cov95": {"delta": d_mf_cov, "t": t_mf_cov, "p": p_mf_cov},
            },
            "sign_consistency": he_winner == mf_winner,
            "hetero_head_winner_wis": he_winner,
            "method_f_winner_wis": mf_winner,
        }
    }
    with (E1_DIR / "p1_card1_cal_retest.json").open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: runs/e1_final/p1_card1_cal_retest.json")


if __name__ == "__main__":
    main()
