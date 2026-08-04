"""STEP 3 — Pre-flight P1/P4/P6/P7 under LOCK v2.2.

P1: HMM γ_all sanity (degeneracy STOP signal)
P4: synthetic-null FP rate ≈ α (independent noise null)
P6: M0/M1 statsmodels formula string lock
P7: collinearity-preserving permutation null p-uniformity (KS test)

Outputs:
  runs/interpretability/preflight_locked.json
  runs/interpretability/preflight_smoke.json (stdout-mirror)

Usage:
  source runs/interpretability/.venv_p5/bin/activate
  python scripts/p5_step3_preflight.py
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "runs/interpretability/sigma_components.parquet"
TRANS_JSON = ROOT / "runs/interpretability/transition_points_locked.json"
OUT = ROOT / "runs/interpretability/preflight_locked.json"

REGIONS = tuple(f"hhs{i}" for i in range(1, 11))
SEASONS = ("2022-2023", "2023-2024", "2024-2025")

# LOCK §13 constants
LOGIT_KW = dict(method="newton", maxiter=100, disp=0, tol=1e-8)

# Pre-flight budget (calibration; not main-test B_perm=1000)
P_PERM_PREFLIGHT = 200
P_SIMS = 100
RNG_SEED_P4 = 1729
RNG_SEED_P7 = 31337


def safe_log1p(x):
    return np.log1p(np.clip(x, 0.0, None))


def fit_logit(y, X):
    """Unpenalized statsmodels Logit (LOCK §13)."""
    X = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.Logit(y, X).fit(**LOGIT_KW)
    return res


def delta_auc(y, X0, X1):
    """ΔAUC = AUC(M1) − AUC(M0); both fit in-sample; AUC in-sample."""
    r0 = fit_logit(y, X0)
    r1 = fit_logit(y, X1)
    X0c = sm.add_constant(X0, has_constant="add")
    X1c = sm.add_constant(X1, has_constant="add")
    p0 = r0.predict(X0c)
    p1 = r1.predict(X1c)
    return float(roc_auc_score(y, p1) - roc_auc_score(y, p0))


def within_block_shuffle(values, block_ids, rng):
    """Per LOCK §5.2.1: shuffle values within each block."""
    out = values.copy()
    for b in np.unique(block_ids):
        idx = np.where(block_ids == b)[0]
        perm = rng.permutation(len(idx))
        out[idx] = values[idx[perm]]
    return out


def build_h1_panel(parquet_path, transitions_json):
    """Returns (y_transition, X0 [sigma_total only], X1 [+sigma_between], sigma_between, sigma_total, block_ids).

    Uses H1-turning as the positive class (the primary-powered subtest per Fix #4).
    N = 5 seeds × 10 regions × 149 weeks = 7,450 rows.
    """
    df = pd.read_parquet(parquet_path).query("h == 1").copy()

    with open(transitions_json) as f:
        trans = json.load(f)

    # Build turning_label per (region, week_idx) — same across all 5 seeds
    label_map = {}
    season_map = {}
    for r in REGIONS:
        blk = trans["by_region"][r]
        for wi, ep, lab, s in zip(
            range(len(blk["epiweeks"])),
            blk["epiweeks"],
            blk["turning_label"],
            blk["season_of_each_week"],
        ):
            label_map[(r, wi)] = bool(lab)
            season_map[(r, wi)] = s

    df["y_turning"] = df.apply(lambda r: label_map[(r.region, r.week_idx)], axis=1).astype(int)
    df["season"] = df.apply(lambda r: season_map[(r.region, r.week_idx)], axis=1)

    # σ² components — already in z-space from STEP 1; but LOCK uses log1p of raw σ²
    # The parquet has sigma2_within_z, sigma2_between_z, sigma2_total_z. These are σ² values
    # (variances) computed by the model, named "_z" because they live in z-target space.
    # log1p(σ²) is the LOCK-frozen transform (§13). Use log1p on these σ² values as-is.
    sigma_total = df.sigma2_total_z.to_numpy()
    sigma_between = df.sigma2_between_z.to_numpy()

    block_ids = np.array([f"{r}__{s}" for r, s in zip(df.region, df.season)])
    return (
        df.y_turning.to_numpy().astype(int),
        sigma_total,
        sigma_between,
        block_ids,
        df,
    )


def p1_gamma_sanity(parquet_path: Path) -> dict:
    """P1 — HMM γ_all sanity (degeneracy STOP)."""
    df = pd.read_parquet(parquet_path)
    g = df[["gamma_all_0", "gamma_all_1", "gamma_all_2"]].to_numpy()
    row_sums = g.sum(axis=1)
    sum_dev_max = float(np.abs(row_sums - 1.0).max())

    # Global marginal: mean γ per state
    marginal_global = g.mean(axis=0)
    # Per-region marginal
    per_region = {}
    for r in REGIONS:
        sub = df[df.region == r][["gamma_all_0", "gamma_all_1", "gamma_all_2"]].to_numpy()
        per_region[r] = sub.mean(axis=0).tolist()

    # STOP rule: any region's max marginal > 0.90
    max_per_region = {r: float(max(per_region[r])) for r in REGIONS}
    global_max = float(marginal_global.max())
    degenerate = (global_max > 0.90) or any(v > 0.90 for v in max_per_region.values())

    return {
        "sum_to_one_max_abs_deviation": sum_dev_max,
        "marginal_global": marginal_global.tolist(),
        "marginal_per_region": per_region,
        "max_per_region": max_per_region,
        "global_max_state_share": global_max,
        "degenerate_state_present": degenerate,
        "stop_signal": degenerate or sum_dev_max > 1e-3,
    }


def p4_independent_noise_calibration(y, sigma_total, sigma_between, block_ids, n_sims, b_perm) -> dict:
    """P4 — independent-noise synthetic-null FP rate.

    Generate σ²_between' as random noise matched to marginal mean/std of real σ²_between.
    Run permutation null procedure; expect FP rate ≈ α (Bonferroni 0.0125).
    """
    rng = np.random.default_rng(RNG_SEED_P4)
    mu, sd = float(sigma_between.mean()), float(sigma_between.std())
    log_total = safe_log1p(sigma_total)

    p_values = []
    delta_obs_list = []
    for sim in range(n_sims):
        # Independent noise null (clip to non-negative to keep log1p sensible)
        between_null = np.clip(rng.normal(loc=mu, scale=sd, size=len(sigma_between)), 0.0, None)
        log_between_null = safe_log1p(between_null)

        X0 = log_total.reshape(-1, 1)
        X1 = np.column_stack([log_total, log_between_null])
        try:
            d_obs = delta_auc(y, X0, X1)
        except Exception:
            continue

        deltas = []
        for _ in range(b_perm):
            shuffled = within_block_shuffle(between_null, block_ids, rng)
            X1p = np.column_stack([log_total, safe_log1p(shuffled)])
            try:
                deltas.append(delta_auc(y, X0, X1p))
            except Exception:
                pass
        if not deltas:
            continue
        deltas = np.array(deltas)
        p = (1 + np.sum(np.abs(deltas) >= np.abs(d_obs))) / (len(deltas) + 1)
        p_bonf = min(1.0, 4 * p)
        p_values.append(p_bonf)
        delta_obs_list.append(d_obs)

    p_arr = np.array(p_values)
    fp_rate_bonf = float((p_arr < 0.0125).mean()) if len(p_arr) else float("nan")
    ks_stat, ks_p = stats.kstest(p_arr / p_arr.max() if p_arr.max() > 0 else p_arr, "uniform")
    # use uncorrected p for KS uniformity check (Bonferroni warps the distribution)
    # Recompute uniform check on raw p (before ×4)
    raw_p = np.array([min(p, 1.0) for p in (p_arr / 4)])
    ks_stat_raw, ks_p_raw = stats.kstest(raw_p, "uniform")
    return {
        "n_sims_completed": len(p_values),
        "fp_rate_bonf_observed": fp_rate_bonf,
        "fp_rate_bonf_expected": 0.0125,
        "ks_uniform_raw_p_stat": float(ks_stat_raw),
        "ks_uniform_raw_p_pvalue": float(ks_p_raw),
        "mean_delta_obs": float(np.mean(delta_obs_list)) if delta_obs_list else None,
        "calibration_pass": fp_rate_bonf <= 0.05 if not np.isnan(fp_rate_bonf) else False,
    }


def p7_collinearity_preserving_calibration(y, sigma_total, sigma_between, block_ids, n_sims, b_perm) -> dict:
    """P7 — collinearity-preserving synthetic-null calibration (the CRITICAL one).

    Synthetic null: σ²_between'_null = within-block shuffle of REAL σ²_between
    (preserves marginal + mechanical relation to σ²_total per block).
    Run permutation procedure on synthetic; p-values should be uniform.
    """
    rng = np.random.default_rng(RNG_SEED_P7)
    log_total = safe_log1p(sigma_total)
    X0 = log_total.reshape(-1, 1)

    # Sanity: original collinearity between log_total and log_between
    log_between_real = safe_log1p(sigma_between)
    corr_orig = float(np.corrcoef(log_total, log_between_real)[0, 1])

    # Sanity: after within-block shuffle, does collinearity persist?
    corrs_after_shuffle = []
    for _ in range(20):
        sh = within_block_shuffle(sigma_between, block_ids, rng)
        corrs_after_shuffle.append(float(np.corrcoef(log_total, safe_log1p(sh))[0, 1]))
    corr_shuffle_mean = float(np.mean(corrs_after_shuffle))
    corr_shuffle_std = float(np.std(corrs_after_shuffle))

    p_values_raw = []
    delta_obs_list = []
    for sim in range(n_sims):
        # Generate one synthetic-null realization
        between_null = within_block_shuffle(sigma_between, block_ids, rng)
        log_between_null = safe_log1p(between_null)
        X1 = np.column_stack([log_total, log_between_null])
        try:
            d_obs = delta_auc(y, X0, X1)
        except Exception:
            continue

        # Apply LOCK §5.2.1 permutation procedure to this null instance
        deltas = []
        for _ in range(b_perm):
            sh = within_block_shuffle(between_null, block_ids, rng)
            X1p = np.column_stack([log_total, safe_log1p(sh)])
            try:
                deltas.append(delta_auc(y, X0, X1p))
            except Exception:
                pass
        if not deltas:
            continue
        deltas = np.array(deltas)
        p = (1 + np.sum(np.abs(deltas) >= np.abs(d_obs))) / (len(deltas) + 1)
        p_values_raw.append(p)
        delta_obs_list.append(d_obs)

    p_arr = np.array(p_values_raw)
    fp_rate_raw = float((p_arr < 0.05).mean()) if len(p_arr) else float("nan")
    fp_rate_bonf = float((np.minimum(1.0, 4 * p_arr) < 0.0125).mean()) if len(p_arr) else float("nan")
    ks_stat, ks_p = stats.kstest(p_arr, "uniform")

    return {
        "collinearity_preservation": {
            "corr_log_total_vs_log_between_real": corr_orig,
            "corr_log_total_vs_log_between_shuffled_mean": corr_shuffle_mean,
            "corr_log_total_vs_log_between_shuffled_std": corr_shuffle_std,
            "preserved_within_1pct": abs(corr_orig - corr_shuffle_mean) < 0.01,
        },
        "n_sims_completed": len(p_values_raw),
        "ks_uniform_stat": float(ks_stat),
        "ks_uniform_pvalue": float(ks_p),
        "fp_rate_raw_at_alpha_0.05": fp_rate_raw,
        "fp_rate_bonf_at_0.0125": fp_rate_bonf,
        "fp_rate_bonf_expected": 0.0125,
        "mean_delta_obs": float(np.mean(delta_obs_list)) if delta_obs_list else None,
        "calibration_pass": (ks_p > 0.05) and (fp_rate_bonf <= 0.05) if not np.isnan(fp_rate_bonf) else False,
    }


def p6_formula_lock() -> dict:
    """P6 — M0/M1 formula string lock."""
    m0 = "y_transition ~ log1p(sigma2_total)"
    m1 = "y_transition ~ log1p(sigma2_total) + log1p(sigma2_between)"
    return {
        "m0_formula": m0,
        "m1_formula": m1,
        "engine": "statsmodels.Logit (unpenalized, Newton, maxiter=100, tol=1e-8)",
        "frozen_sha256_m0": hashlib.sha256(m0.encode()).hexdigest(),
        "frozen_sha256_m1": hashlib.sha256(m1.encode()).hexdigest(),
    }


def main():
    print(f"=== STEP 3 pre-flight (P1/P4/P6/P7) ===")
    t0 = time.time()

    print("[P1] HMM γ_all sanity ...")
    p1 = p1_gamma_sanity(PARQUET)
    print(f"  γ sum-to-1 max dev: {p1['sum_to_one_max_abs_deviation']:.2e}")
    print(f"  global marginal: {[f'{v:.3f}' for v in p1['marginal_global']]}")
    print(f"  global max state share: {p1['global_max_state_share']:.4f} (degeneracy thresh 0.90)")
    print(f"  STOP signal: {p1['stop_signal']}")
    print()

    print("[P6] Formula string lock ...")
    p6 = p6_formula_lock()
    print(f"  M0: {p6['m0_formula']}")
    print(f"  M1: {p6['m1_formula']}")
    print()

    print("[Setup] Building H1-turning panel ...")
    y, sigma_total, sigma_between, block_ids, df = build_h1_panel(PARQUET, TRANS_JSON)
    n_pos = int(y.sum())
    n_total = len(y)
    print(f"  N={n_total} rows (5 seeds × 10 regions × 149 weeks), positives={n_pos} ({n_pos/n_total*100:.2f}%)")
    print()

    print(f"[P4] Independent-noise calibration (n_sims={P_SIMS}, b_perm={P_PERM_PREFLIGHT}) ...")
    t_p4 = time.time()
    p4 = p4_independent_noise_calibration(y, sigma_total, sigma_between, block_ids, P_SIMS, P_PERM_PREFLIGHT)
    p4["wall_time_sec"] = time.time() - t_p4
    print(f"  FP rate (Bonf<0.0125): {p4['fp_rate_bonf_observed']:.4f} (expected ≈ 0.0125)")
    print(f"  KS uniform raw-p: D={p4['ks_uniform_raw_p_stat']:.4f}, p={p4['ks_uniform_raw_p_pvalue']:.4f}")
    print(f"  calibration_pass: {p4['calibration_pass']}")
    print(f"  ({p4['wall_time_sec']:.1f}s)")
    print()

    print(f"[P7] Collinearity-preserving calibration (n_sims={P_SIMS}, b_perm={P_PERM_PREFLIGHT}) ...")
    t_p7 = time.time()
    p7 = p7_collinearity_preserving_calibration(y, sigma_total, sigma_between, block_ids, P_SIMS, P_PERM_PREFLIGHT)
    p7["wall_time_sec"] = time.time() - t_p7
    cp = p7["collinearity_preservation"]
    print(f"  corr log(total) vs log(between): real={cp['corr_log_total_vs_log_between_real']:.4f}, shuffled={cp['corr_log_total_vs_log_between_shuffled_mean']:.4f} ± {cp['corr_log_total_vs_log_between_shuffled_std']:.4f}")
    print(f"  collinearity preserved (within 1%): {cp['preserved_within_1pct']}")
    print(f"  KS uniform: D={p7['ks_uniform_stat']:.4f}, p={p7['ks_uniform_pvalue']:.4f}")
    print(f"  FP rate (Bonf<0.0125): {p7['fp_rate_bonf_at_0.0125']:.4f} (expected ≈ 0.0125)")
    print(f"  calibration_pass: {p7['calibration_pass']}")
    print(f"  ({p7['wall_time_sec']:.1f}s)")
    print()

    result = {
        "lock_version": "v2.2 + §14.3 + §14.4",
        "preflight_budget": {"n_sims": P_SIMS, "b_perm": P_PERM_PREFLIGHT,
                              "note": "main test uses B_perm=1000 (LOCK §5.2.1); pre-flight smaller for time budget"},
        "P1_gamma_sanity": p1,
        "P4_independent_noise_calibration": p4,
        "P6_formula_lock": p6,
        "P7_collinearity_preserving_calibration": p7,
        "total_wall_time_sec": time.time() - t0,
    }
    OUT.write_bytes(json.dumps(result, sort_keys=True, separators=(",", ":")).encode())

    print("=== SUMMARY ===")
    print(f"P1 STOP signal: {p1['stop_signal']}")
    print(f"P4 calibration pass: {p4['calibration_pass']}")
    print(f"P6 formulas locked: yes")
    print(f"P7 calibration pass: {p7['calibration_pass']}")
    print(f"Saved: {OUT}")
    all_pass = (not p1["stop_signal"]) and p4["calibration_pass"] and p7["calibration_pass"]
    print(f"ALL PRE-FLIGHT PASS: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
