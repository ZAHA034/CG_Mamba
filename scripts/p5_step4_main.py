"""STEP 4 — Main analysis under LOCK v2.2 + §14.3 + §14.4.

Tests (m=4 family, Bonferroni alpha=0.0125):
  - H1-onset:   NOT EVALUABLE per §14.4
  - H1-peak:    nested Logit, under-powered (30 positives), disclosed-weak
  - H1-turning: nested Logit, primary-powered (725 positives)
  - H2-transition-error: nested OLS on |y - mu| within (peak ∪ turning) ±2 weeks

Per test:
  - observed Δ (seed-pooled, 5 seeds concatenated)
  - block-conditional permutation null p (B_perm=1000) + Bonferroni p*4
  - block-bootstrap BCa 95% CI + Bonf 98.75% CI (B_boot=1000)
  - 5/5 seed sign-consistency
  - §7 wILI-adjusted effect + permutation + CI

Verdict per LOCK §2:
  STRONG: (a) and (b) and (c) and (d) all pass
  MARGINAL: middle bands
  FORBIDDEN: all primary tests Δ<0.03 AND p_Bonf≥0.0125; OR ≤3/5 sign; OR wILI-adjusted all <0.03

Usage:
  source runs/interpretability/.venv_p5/bin/activate
  python scripts/p5_step4_main.py
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
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "runs/interpretability/sigma_components.parquet"
TRANS_JSON = ROOT / "runs/interpretability/transition_points_locked.json"
OUT = ROOT / "runs/interpretability/main_analysis_locked.json"

REGIONS = tuple(f"hhs{i}" for i in range(1, 11))
SEASONS = ("2022-2023", "2023-2024", "2024-2025")
SEEDS = (42, 123, 456, 789, 1024)

# LOCK §5.2 + §13
B_PERM = 1000
B_BOOT = 1000
ALPHA = 0.05
ALPHA_BONF = 0.0125  # = 0.05 / 4
TRANSITION_WINDOW_RADIUS = 2  # §3.4

# LOCK §13 statsmodels Logit / OLS settings
LOGIT_KW = dict(method="newton", maxiter=100, disp=0, tol=1e-8)

RNG_SEED_MAIN = 8675309


def safe_log1p(x):
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0.0, None))


def fit_logit(y, X):
    X = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(y, X).fit(**LOGIT_KW)
            return res, X
        except PerfectSeparationError:
            return None, X
        except Exception:
            return None, X


def fit_ols(y, X):
    X = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.OLS(y, X).fit()
            return res, X
        except Exception:
            return None, X


def delta_auc(y, X0, X1):
    r0, X0c = fit_logit(y, X0)
    r1, X1c = fit_logit(y, X1)
    if r0 is None or r1 is None:
        return None
    try:
        p0 = r0.predict(X0c)
        p1 = r1.predict(X1c)
        return float(roc_auc_score(y, p1) - roc_auc_score(y, p0))
    except Exception:
        return None


def delta_r2(y, X0, X1):
    r0, _ = fit_ols(y, X0)
    r1, _ = fit_ols(y, X1)
    if r0 is None or r1 is None:
        return None
    return float(r1.rsquared - r0.rsquared)


def within_block_shuffle(values, block_ids, rng):
    out = values.copy()
    unique_blocks = np.unique(block_ids)
    for b in unique_blocks:
        idx = np.where(block_ids == b)[0]
        if len(idx) > 1:
            perm = rng.permutation(len(idx))
            out[idx] = values[idx[perm]]
    return out


def block_bootstrap_resample(N, block_ids, rng):
    """Resample blocks WITH replacement; return row indices."""
    unique_blocks = np.unique(block_ids)
    chosen = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
    rows = []
    for b in chosen:
        idx = np.where(block_ids == b)[0]
        rows.extend(idx.tolist())
    return np.array(rows, dtype=int)


def bca_ci(boot_estimates, theta_hat, jackknife_estimates, alpha):
    """BCa percentile interval. Returns (lo, hi)."""
    boot = np.array([b for b in boot_estimates if b is not None and np.isfinite(b)])
    jk = np.array([j for j in jackknife_estimates if j is not None and np.isfinite(j)])
    if len(boot) < 100 or len(jk) < 5:
        return None, None
    # bias-correction
    z0 = stats_norm_ppf((boot < theta_hat).mean())
    # acceleration via jackknife
    jk_mean = jk.mean()
    num = ((jk_mean - jk) ** 3).sum()
    den = 6 * (((jk_mean - jk) ** 2).sum()) ** 1.5
    a_hat = num / den if den != 0 else 0.0
    z_alpha = stats_norm_ppf(alpha / 2)
    z_1mAlpha = stats_norm_ppf(1 - alpha / 2)
    p_lo = stats_norm_cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    p_hi = stats_norm_cdf(z0 + (z0 + z_1mAlpha) / (1 - a_hat * (z0 + z_1mAlpha)))
    lo = float(np.quantile(boot, p_lo))
    hi = float(np.quantile(boot, p_hi))
    return lo, hi


def stats_norm_ppf(p):
    from scipy.stats import norm
    return float(norm.ppf(p))


def stats_norm_cdf(z):
    from scipy.stats import norm
    return float(norm.cdf(z))


def run_logistic_test(y, sigma_total, sigma_between, wILI_level, block_ids, seeds, label_name, b_perm, b_boot, rng):
    """One H1 test (logistic ΔAUC), seed-pooled.

    Returns dict with: delta_obs, perm_p, perm_p_bonf, bca_lo, bca_hi, per_seed_deltas,
                       sign_consistency_n_positive, adj_delta_obs, adj_perm_p, adj_perm_p_bonf,
                       adj_bca_lo, adj_bca_hi, separation_detected
    """
    log_total = safe_log1p(sigma_total)
    log_between = safe_log1p(sigma_between)
    log_wILI = safe_log1p(wILI_level)
    X0 = log_total.reshape(-1, 1)
    X1 = np.column_stack([log_total, log_between])

    # ---- Unadjusted ----
    d_obs = delta_auc(y, X0, X1)
    sep = d_obs is None
    if sep:
        return {
            "delta_obs": None, "perm_p": None, "perm_p_bonf": None,
            "bca_lo": None, "bca_hi": None, "per_seed_deltas": [None]*5,
            "sign_consistency_n_positive": 0,
            "adj_delta_obs": None, "adj_perm_p": None, "adj_perm_p_bonf": None,
            "adj_bca_lo": None, "adj_bca_hi": None,
            "separation_detected": True,
        }

    # Permutation null (B_perm)
    deltas = []
    for _ in range(b_perm):
        shuffled = within_block_shuffle(sigma_between, block_ids, rng)
        X1p = np.column_stack([log_total, safe_log1p(shuffled)])
        d = delta_auc(y, X0, X1p)
        if d is not None:
            deltas.append(d)
    deltas_arr = np.array(deltas)
    perm_p = float((1 + np.sum(np.abs(deltas_arr) >= abs(d_obs))) / (len(deltas_arr) + 1))
    perm_p_bonf = min(1.0, 4 * perm_p)

    # Block-bootstrap CI (B_boot)
    boot_estimates = []
    for _ in range(b_boot):
        rows = block_bootstrap_resample(len(y), block_ids, rng)
        d = delta_auc(y[rows], log_total[rows].reshape(-1, 1),
                      np.column_stack([log_total[rows], log_between[rows]]))
        if d is not None:
            boot_estimates.append(d)
    # Jackknife for acceleration: delete-one-block
    jk_estimates = []
    unique_blocks = np.unique(block_ids)
    for b in unique_blocks:
        rows = np.where(block_ids != b)[0]
        d = delta_auc(y[rows], log_total[rows].reshape(-1, 1),
                      np.column_stack([log_total[rows], log_between[rows]]))
        if d is not None:
            jk_estimates.append(d)
    bca_lo, bca_hi = bca_ci(boot_estimates, d_obs, jk_estimates, alpha=ALPHA)

    # Per-seed sign consistency
    per_seed = []
    for s_val in SEEDS:
        mask = seeds == s_val
        if mask.sum() == 0:
            per_seed.append(None)
            continue
        d_s = delta_auc(y[mask], log_total[mask].reshape(-1, 1),
                        np.column_stack([log_total[mask], log_between[mask]]))
        per_seed.append(d_s)
    sign_pos = sum(1 for d in per_seed if d is not None and d >= 0)

    # ---- wILI-adjusted (§7) ----
    X0_adj = np.column_stack([log_total, log_wILI])
    X1_adj = np.column_stack([log_total, log_between, log_wILI])
    d_obs_adj = delta_auc(y, X0_adj, X1_adj)
    if d_obs_adj is not None:
        deltas_adj = []
        for _ in range(b_perm):
            shuffled = within_block_shuffle(sigma_between, block_ids, rng)
            X1p_adj = np.column_stack([log_total, safe_log1p(shuffled), log_wILI])
            d = delta_auc(y, X0_adj, X1p_adj)
            if d is not None:
                deltas_adj.append(d)
        deltas_adj_arr = np.array(deltas_adj)
        perm_p_adj = float((1 + np.sum(np.abs(deltas_adj_arr) >= abs(d_obs_adj))) / (len(deltas_adj_arr) + 1))
        perm_p_bonf_adj = min(1.0, 4 * perm_p_adj)
        # bootstrap for adj CI
        boot_adj = []
        for _ in range(b_boot):
            rows = block_bootstrap_resample(len(y), block_ids, rng)
            d = delta_auc(y[rows],
                          np.column_stack([log_total[rows], log_wILI[rows]]),
                          np.column_stack([log_total[rows], log_between[rows], log_wILI[rows]]))
            if d is not None:
                boot_adj.append(d)
        jk_adj = []
        for b in unique_blocks:
            rows = np.where(block_ids != b)[0]
            d = delta_auc(y[rows],
                          np.column_stack([log_total[rows], log_wILI[rows]]),
                          np.column_stack([log_total[rows], log_between[rows], log_wILI[rows]]))
            if d is not None:
                jk_adj.append(d)
        bca_lo_adj, bca_hi_adj = bca_ci(boot_adj, d_obs_adj, jk_adj, alpha=ALPHA)
    else:
        perm_p_adj = perm_p_bonf_adj = bca_lo_adj = bca_hi_adj = None

    return {
        "delta_obs": float(d_obs),
        "perm_p": perm_p,
        "perm_p_bonf": perm_p_bonf,
        "bca_lo_95": bca_lo, "bca_hi_95": bca_hi,
        "per_seed_deltas": per_seed,
        "sign_consistency_n_positive": sign_pos,
        "adj_delta_obs": float(d_obs_adj) if d_obs_adj is not None else None,
        "adj_perm_p": perm_p_adj, "adj_perm_p_bonf": perm_p_bonf_adj,
        "adj_bca_lo_95": bca_lo_adj, "adj_bca_hi_95": bca_hi_adj,
        "separation_detected": False,
        "n_rows": len(y), "n_positives": int(y.sum()),
        "n_perm_completed": len(deltas), "n_boot_completed": len(boot_estimates),
    }


def run_ols_test(y_abs_err, sigma_total, sigma_between, wILI_level, block_ids, seeds, b_perm, b_boot, rng):
    """H2: nested OLS on |y - mu|. Returns same shape as logistic test."""
    log_total = safe_log1p(sigma_total)
    log_between = safe_log1p(sigma_between)
    log_wILI = safe_log1p(wILI_level)
    X0 = log_total.reshape(-1, 1)
    X1 = np.column_stack([log_total, log_between])

    d_obs = delta_r2(y_abs_err, X0, X1)
    if d_obs is None:
        return {"separation_detected": True, "delta_obs": None}

    deltas = []
    for _ in range(b_perm):
        shuffled = within_block_shuffle(sigma_between, block_ids, rng)
        X1p = np.column_stack([log_total, safe_log1p(shuffled)])
        d = delta_r2(y_abs_err, X0, X1p)
        if d is not None:
            deltas.append(d)
    deltas_arr = np.array(deltas)
    perm_p = float((1 + np.sum(np.abs(deltas_arr) >= abs(d_obs))) / (len(deltas_arr) + 1))
    perm_p_bonf = min(1.0, 4 * perm_p)

    boot = []
    for _ in range(b_boot):
        rows = block_bootstrap_resample(len(y_abs_err), block_ids, rng)
        d = delta_r2(y_abs_err[rows], log_total[rows].reshape(-1, 1),
                     np.column_stack([log_total[rows], log_between[rows]]))
        if d is not None:
            boot.append(d)
    jk = []
    for b in np.unique(block_ids):
        rows = np.where(block_ids != b)[0]
        d = delta_r2(y_abs_err[rows], log_total[rows].reshape(-1, 1),
                     np.column_stack([log_total[rows], log_between[rows]]))
        if d is not None:
            jk.append(d)
    bca_lo, bca_hi = bca_ci(boot, d_obs, jk, alpha=ALPHA)

    per_seed = []
    for s_val in SEEDS:
        mask = seeds == s_val
        if mask.sum() == 0:
            per_seed.append(None); continue
        d_s = delta_r2(y_abs_err[mask], log_total[mask].reshape(-1, 1),
                       np.column_stack([log_total[mask], log_between[mask]]))
        per_seed.append(d_s)
    sign_pos = sum(1 for d in per_seed if d is not None and d >= 0)

    # wILI-adjusted
    X0_adj = np.column_stack([log_total, log_wILI])
    X1_adj = np.column_stack([log_total, log_between, log_wILI])
    d_obs_adj = delta_r2(y_abs_err, X0_adj, X1_adj)
    if d_obs_adj is not None:
        deltas_adj = []
        for _ in range(b_perm):
            shuffled = within_block_shuffle(sigma_between, block_ids, rng)
            X1p_adj = np.column_stack([log_total, safe_log1p(shuffled), log_wILI])
            d = delta_r2(y_abs_err, X0_adj, X1p_adj)
            if d is not None:
                deltas_adj.append(d)
        deltas_adj_arr = np.array(deltas_adj)
        perm_p_adj = float((1 + np.sum(np.abs(deltas_adj_arr) >= abs(d_obs_adj))) / (len(deltas_adj_arr) + 1))
        perm_p_bonf_adj = min(1.0, 4 * perm_p_adj)
        boot_adj = []
        for _ in range(b_boot):
            rows = block_bootstrap_resample(len(y_abs_err), block_ids, rng)
            d = delta_r2(y_abs_err[rows],
                         np.column_stack([log_total[rows], log_wILI[rows]]),
                         np.column_stack([log_total[rows], log_between[rows], log_wILI[rows]]))
            if d is not None:
                boot_adj.append(d)
        jk_adj = []
        for b in np.unique(block_ids):
            rows = np.where(block_ids != b)[0]
            d = delta_r2(y_abs_err[rows],
                         np.column_stack([log_total[rows], log_wILI[rows]]),
                         np.column_stack([log_total[rows], log_between[rows], log_wILI[rows]]))
            if d is not None:
                jk_adj.append(d)
        bca_lo_adj, bca_hi_adj = bca_ci(boot_adj, d_obs_adj, jk_adj, alpha=ALPHA)
    else:
        perm_p_adj = perm_p_bonf_adj = bca_lo_adj = bca_hi_adj = None

    return {
        "delta_obs": float(d_obs),
        "perm_p": perm_p, "perm_p_bonf": perm_p_bonf,
        "bca_lo_95": bca_lo, "bca_hi_95": bca_hi,
        "per_seed_deltas": per_seed,
        "sign_consistency_n_positive": sign_pos,
        "adj_delta_obs": float(d_obs_adj) if d_obs_adj is not None else None,
        "adj_perm_p": perm_p_adj, "adj_perm_p_bonf": perm_p_bonf_adj,
        "adj_bca_lo_95": bca_lo_adj, "adj_bca_hi_95": bca_hi_adj,
        "separation_detected": False,
        "n_rows": len(y_abs_err), "n_observations": int((y_abs_err >= 0).sum()),
        "n_perm_completed": len(deltas), "n_boot_completed": len(boot),
    }


def build_panels(parquet_path, trans_json):
    """Returns dict with keys: 'h1_peak', 'h1_turning', 'h2'."""
    df = pd.read_parquet(parquet_path).query("h == 1").copy()
    with open(trans_json) as f:
        trans = json.load(f)

    # Build labels per (region, week_idx)
    peak_map, turn_map, season_map, eps_map = {}, {}, {}, {}
    for r in REGIONS:
        blk = trans["by_region"][r]
        for wi, ep, pk, tn, s in zip(
            range(len(blk["epiweeks"])),
            blk["epiweeks"],
            blk["peak_label"],
            blk["turning_label"],
            blk["season_of_each_week"],
        ):
            peak_map[(r, wi)] = bool(pk)
            turn_map[(r, wi)] = bool(tn)
            season_map[(r, wi)] = s
            eps_map[(r, wi)] = ep

    df["y_peak"] = df.apply(lambda r: peak_map[(r.region, r.week_idx)], axis=1).astype(int)
    df["y_turning"] = df.apply(lambda r: turn_map[(r.region, r.week_idx)], axis=1).astype(int)
    df["season"] = df.apply(lambda r: season_map[(r.region, r.week_idx)], axis=1)
    df["block_id"] = df.region.astype(str) + "__" + df.season.astype(str)

    # H1 panels use all rows
    h1_panel = df.copy()
    h1_panel["abs_err_h1"] = np.abs(
        h1_panel.y_raw.to_numpy()
        - (h1_panel.mu_cgm_z.to_numpy() * h1_panel.target_std.to_numpy() + h1_panel.target_mean.to_numpy())
    )

    # H2 panel: (peak ∪ turning) ±2 weeks
    # For each (region, week_idx) compute window membership
    n_weeks = 149
    in_window = {}
    for r in REGIONS:
        wks_in_union = [wi for wi in range(n_weeks) if peak_map[(r, wi)] or turn_map[(r, wi)]]
        wset = set()
        for wi in wks_in_union:
            for off in range(-TRANSITION_WINDOW_RADIUS, TRANSITION_WINDOW_RADIUS + 1):
                if 0 <= wi + off < n_weeks:
                    wset.add(wi + off)
        for wi in range(n_weeks):
            in_window[(r, wi)] = wi in wset

    h2_panel = h1_panel.copy()
    h2_panel["in_window"] = h2_panel.apply(lambda r: in_window[(r.region, r.week_idx)], axis=1)
    h2_panel = h2_panel[h2_panel["in_window"]].copy()

    return {
        "h1_peak": h1_panel,
        "h1_turning": h1_panel,
        "h2": h2_panel,
    }


def verdict(results):
    """Per LOCK §2 + §14.4 Refinement #1: 'at least one of {peak, turning}'."""
    # (a) H1 — at least one of {peak, turning}
    h1_subtests = [results["H1_peak"], results["H1_turning"]]
    h1_pass_a, h1_marginal_a = [], []
    for r in h1_subtests:
        if r.get("separation_detected") or r["delta_obs"] is None:
            continue
        d = r["delta_obs"]
        p = r["perm_p_bonf"]
        lo, hi = r.get("bca_lo_95"), r.get("bca_hi_95")
        ci_excludes_zero = (lo is not None and hi is not None and (lo > 0 or hi < 0))
        if d >= 0.05 and p < ALPHA_BONF and ci_excludes_zero:
            h1_pass_a.append(r)
        elif 0.03 <= d < 0.05 and p < ALPHA_BONF:
            h1_marginal_a.append(r)
    a_strong = len(h1_pass_a) >= 1

    # (b) H2
    h2 = results["H2_transition_error"]
    if h2.get("separation_detected") or h2["delta_obs"] is None:
        b_strong = b_marginal = False
    else:
        d = h2["delta_obs"]; p = h2["perm_p_bonf"]
        lo, hi = h2.get("bca_lo_95"), h2.get("bca_hi_95")
        ci_excludes_zero = (lo is not None and hi is not None and (lo > 0 or hi < 0))
        b_strong = d >= 0.05 and p < ALPHA_BONF and ci_excludes_zero
        b_marginal = 0.03 <= d < 0.05 and p < ALPHA_BONF

    # (c) 5/5 sign consistency on the passing subtest of (a) and on (b)
    # Per LOCK §2(c): "the point estimates of ΔAUC (passing subtest) and ΔR² are BOTH ≥ 0"
    c_strong = True
    sign_details = {}
    for r in h1_pass_a:
        n = r["sign_consistency_n_positive"]
        sign_details["h1_passing"] = n
        if n < 5:
            c_strong = False
    if not h2.get("separation_detected") and h2.get("delta_obs") is not None:
        n = h2["sign_consistency_n_positive"]
        sign_details["h2"] = n
        if n < 5:
            c_strong = False

    # (d) wILI-adjusted ≥0.03 per H1-passing subtest and H2
    d_strong = True
    d_pass_details = {}
    for r in h1_pass_a:
        adj_d = r.get("adj_delta_obs")
        adj_p = r.get("adj_perm_p_bonf")
        adj_lo = r.get("adj_bca_lo_95"); adj_hi = r.get("adj_bca_hi_95")
        adj_ci = adj_lo is not None and adj_hi is not None and (adj_lo > 0 or adj_hi < 0)
        d_pass_details["h1_passing_adj"] = {"d": adj_d, "p_bonf": adj_p, "ci_excludes_zero": adj_ci}
        if not (adj_d is not None and adj_d >= 0.03 and adj_p is not None and adj_p < ALPHA_BONF and adj_ci):
            d_strong = False
    if not h2.get("separation_detected") and h2.get("delta_obs") is not None:
        adj_d = h2.get("adj_delta_obs")
        adj_p = h2.get("adj_perm_p_bonf")
        adj_lo = h2.get("adj_bca_lo_95"); adj_hi = h2.get("adj_bca_hi_95")
        adj_ci = adj_lo is not None and adj_hi is not None and (adj_lo > 0 or adj_hi < 0)
        d_pass_details["h2_adj"] = {"d": adj_d, "p_bonf": adj_p, "ci_excludes_zero": adj_ci}
        if not (adj_d is not None and adj_d >= 0.03 and adj_p is not None and adj_p < ALPHA_BONF and adj_ci):
            d_strong = False

    is_strong = a_strong and b_strong and c_strong and d_strong

    # FORBIDDEN check
    # "All four primary tests Δ<0.03 AND p_Bonf>=0.0125" — for our case onset=not_evaluable, so check 3 tests
    primary_tests = [results["H1_peak"], results["H1_turning"], results["H2_transition_error"]]
    all_below_threshold = True
    for r in primary_tests:
        if r.get("separation_detected") or r["delta_obs"] is None:
            continue
        d = r["delta_obs"]; p = r["perm_p_bonf"]
        if not (d < 0.03 and p >= ALPHA_BONF):
            all_below_threshold = False
            break
    if all_below_threshold:
        forbidden = True
    else:
        forbidden = False
    # Sign consistency ≤3/5
    for r in primary_tests + [h2]:
        if r.get("separation_detected") or r.get("delta_obs") is None:
            continue
        if r["sign_consistency_n_positive"] <= 3:
            forbidden = True
            break
    # wILI-adjusted all below 0.03
    all_adj_below = True
    for r in primary_tests:
        if r.get("separation_detected") or r.get("delta_obs") is None:
            continue
        adj_d = r.get("adj_delta_obs")
        if adj_d is not None and adj_d >= 0.03:
            all_adj_below = False
            break
    if all_adj_below:
        forbidden = True

    if is_strong:
        v = "STRONG"
    elif forbidden:
        v = "FORBIDDEN"
    else:
        v = "MARGINAL"

    return {
        "verdict": v,
        "criteria_breakdown": {
            "(a)_h1_strong": a_strong,
            "(b)_h2_strong": b_strong,
            "(c)_sign_5of5": c_strong,
            "(d)_wILI_adjusted": d_strong,
            "forbidden_triggered": forbidden,
        },
        "h1_passing_count": len(h1_pass_a),
        "sign_details": sign_details,
        "wILI_adj_details": d_pass_details,
    }


def main():
    print(f"=== STEP 4 main analysis (B_perm={B_PERM}, B_boot={B_BOOT}) ===")
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED_MAIN)

    panels = build_panels(PARQUET, TRANS_JSON)
    h1 = panels["h1_turning"]
    h2_p = panels["h2"]
    print(f"H1 panel: N={len(h1)} (5 seeds × 10 regions × 149 weeks)")
    print(f"  positives — peak: {int(h1.y_peak.sum())} ({h1.y_peak.mean()*100:.2f}%) — under-powered (disclosed)")
    print(f"  positives — turning: {int(h1.y_turning.sum())} ({h1.y_turning.mean()*100:.2f}%) — primary-powered")
    print(f"H2 panel: N={len(h2_p)} (within (peak ∪ turning) ±2 weeks)")
    print()

    results = {}

    # ---- H1-onset: NOT EVALUABLE per §14.4 ----
    results["H1_onset"] = {"status": "not_evaluable", "reason": "§14.4 — regional baseline unavailable; CDC prohibits national-uniform"}
    print("H1-onset: NOT EVALUABLE (per §14.4)")
    print()

    # ---- H1-peak ----
    print(f"[H1-peak] (under-powered, disclosed-weak)")
    t = time.time()
    results["H1_peak"] = run_logistic_test(
        h1.y_peak.to_numpy(),
        h1.sigma2_total_z.to_numpy(),
        h1.sigma2_between_z.to_numpy(),
        h1.y_raw.to_numpy(),
        h1.block_id.to_numpy(),
        h1.seed.to_numpy(),
        "H1_peak", B_PERM, B_BOOT, rng,
    )
    print(f"  Δ_obs={results['H1_peak'].get('delta_obs')}, p_Bonf={results['H1_peak'].get('perm_p_bonf')}, sign={results['H1_peak'].get('sign_consistency_n_positive')}/5  ({time.time()-t:.1f}s)")

    # ---- H1-turning ----
    print(f"[H1-turning] (primary-powered)")
    t = time.time()
    results["H1_turning"] = run_logistic_test(
        h1.y_turning.to_numpy(),
        h1.sigma2_total_z.to_numpy(),
        h1.sigma2_between_z.to_numpy(),
        h1.y_raw.to_numpy(),
        h1.block_id.to_numpy(),
        h1.seed.to_numpy(),
        "H1_turning", B_PERM, B_BOOT, rng,
    )
    print(f"  Δ_obs={results['H1_turning'].get('delta_obs')}, p_Bonf={results['H1_turning'].get('perm_p_bonf')}, sign={results['H1_turning'].get('sign_consistency_n_positive')}/5  ({time.time()-t:.1f}s)")

    # ---- H2-transition-error ----
    print(f"[H2-transition-error] (window: peak ∪ turning ±2 weeks)")
    t = time.time()
    results["H2_transition_error"] = run_ols_test(
        h2_p.abs_err_h1.to_numpy(),
        h2_p.sigma2_total_z.to_numpy(),
        h2_p.sigma2_between_z.to_numpy(),
        h2_p.y_raw.to_numpy(),
        h2_p.block_id.to_numpy(),
        h2_p.seed.to_numpy(),
        B_PERM, B_BOOT, rng,
    )
    print(f"  ΔR²_obs={results['H2_transition_error'].get('delta_obs')}, p_Bonf={results['H2_transition_error'].get('perm_p_bonf')}, sign={results['H2_transition_error'].get('sign_consistency_n_positive')}/5  ({time.time()-t:.1f}s)")
    print()

    # ---- Verdict ----
    v = verdict(results)
    out = {
        "lock_version": "v2.2 + §14.3 + §14.4 (m=4 family; H1-onset NOT EVALUABLE)",
        "alpha_bonf": ALPHA_BONF,
        "b_perm": B_PERM, "b_boot": B_BOOT,
        "rng_seed_main": RNG_SEED_MAIN,
        "results": results,
        "judgment": v,
        "total_wall_time_sec": time.time() - t0,
    }
    OUT.write_bytes(json.dumps(out, sort_keys=True, separators=(",", ":"), default=str).encode())

    print("=" * 60)
    print(f"VERDICT: {v['verdict']}")
    print(f"  (a) H1 (peak/turning, ≥0.05+p<0.0125+CI∌0): {v['criteria_breakdown']['(a)_h1_strong']}")
    print(f"  (b) H2 (≥0.05+p<0.0125+CI∌0):              {v['criteria_breakdown']['(b)_h2_strong']}")
    print(f"  (c) 5/5 sign:                              {v['criteria_breakdown']['(c)_sign_5of5']}")
    print(f"  (d) wILI-adjusted ≥0.03:                   {v['criteria_breakdown']['(d)_wILI_adjusted']}")
    print(f"  forbidden triggered:                       {v['criteria_breakdown']['forbidden_triggered']}")
    print(f"Saved: {OUT}")
    print(f"Total wall: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
