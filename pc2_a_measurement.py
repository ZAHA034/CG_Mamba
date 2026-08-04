"""pc2_a_measurement.py — PC2-(a) 사전등록 측정 (LOCKED)
================================================================================
목적: env 고정 시, phase-anchored 분산(√s²_total) vs 상수 s_h scalar 가
transfer regime(regional test_strict)에서 Cov95/WIS 우위?

데이터 셋업 (#② guardrail — 사용자 확인):
  - s_h fit (freeze): national val 만
  - 평가 (test): regional test_strict (hhs1-10) 만, national test 제외

LOCKED bar:
  - PASS: ΔCov95 (또는 ΔWIS) region-cluster bootstrap 95% CI 가 0 제외
          + 방향 일관 (Cov95: phase 가 nominal 에 더 가까움; WIS: phase 더 낮음)
  - FAIL: CI ∋ 0 OR 방향 어긋남 → floor-full-negative
  - PASS 시 narrative lock: transfer 한정 mechanism 가치만 인정, headline 부활 금지
  - FAIL 시: narrative tie 절대 금지, 무조건 floor

LOCKED 운영 상수:
  #1 region-cluster bootstrap: 10 HHS = 10 cluster, B=1000, percentile 95% CI, seed=42
  #5 s_h: national val 1회 quantile-matching → freeze → regional 적용 (재-fit 금지)
  #5 narrative: CI∋0 → 무조건 floor, tie 서사 금지
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent

# ============================================================================
# LOCKED PC2-a CONSTANTS
# ============================================================================
PC2A_BOOTSTRAP_B = 1000
PC2A_BOOTSTRAP_SEED = 42
PC2A_CI_LEVEL = 0.95
PC2A_HORIZONS = [1, 2, 3, 4]
PC2A_REGIONS_EVAL = [f"hhs{i}" for i in range(1, 11)]
PC2A_FLUSIGHT_23 = np.array([
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 0.975, 0.99
])
PC2A_SH_GRID = np.concatenate([
    np.linspace(0.05, 0.5, 30),
    np.linspace(0.5, 2.0, 30),
    np.linspace(2.0, 5.0, 15),
])
PARQUET_PATH = _ROOT / "runs/regime_shift/per_origin_forecasts.parquet"
OUT_DIR = _ROOT / "runs/pc2_a"


# ============================================================================
# Variant 1 — phase-anchored PI: μ + Φ⁻¹(τ) · √s²_total
# ============================================================================
def phase_anchored_pi(mu: np.ndarray, s2_total: np.ndarray,
                       taus: np.ndarray) -> np.ndarray:
    z = norm.ppf(taus)
    sigma = np.sqrt(np.clip(s2_total, 1e-12, None))
    return mu[:, None] + sigma[:, None] * z[None, :]


# ============================================================================
# Variant 2 — scalar s_h per horizon (fit-freeze on national val)
# ============================================================================
def fit_sh_quantile_matching(mu_val: np.ndarray, y_val: np.ndarray,
                              taus: np.ndarray, s_grid: np.ndarray) -> float:
    """s_h = argmin_s Σ_τ (P(y ≤ μ + Φ⁻¹(τ)·s) − τ)². Scalar per horizon."""
    losses = np.zeros(len(s_grid))
    for i, s in enumerate(s_grid):
        err = 0.0
        for tau in taus:
            z_tau = norm.ppf(tau)
            q_pred = mu_val + z_tau * s
            emp = float((y_val <= q_pred).mean())
            err += (emp - tau) ** 2
        losses[i] = err
    return float(s_grid[int(np.argmin(losses))])


def sh_scalar_pi(mu: np.ndarray, s_h: float, taus: np.ndarray) -> np.ndarray:
    z = norm.ppf(taus)
    return mu[:, None] + s_h * z[None, :]


# ============================================================================
# Metrics — Cov95, 23-quantile WIS
# ============================================================================
_LO_IDX = int(np.where(np.isclose(PC2A_FLUSIGHT_23, 0.025))[0][0])
_HI_IDX = int(np.where(np.isclose(PC2A_FLUSIGHT_23, 0.975))[0][0])


def cov95(Q: np.ndarray, y: np.ndarray) -> float:
    lo = Q[:, _LO_IDX]; hi = Q[:, _HI_IDX]
    return float(((y >= lo) & (y <= hi)).mean())


def wis_23(Q: np.ndarray, y: np.ndarray) -> float:
    tau = PC2A_FLUSIGHT_23
    y_b = y[:, None]
    pinball = np.where(y_b >= Q, tau * (y_b - Q), (1 - tau) * (Q - y_b))
    return float(2.0 * pinball.mean())


# ============================================================================
# Per-region (Cov95, WIS) for both variants
# ============================================================================
def per_region_metrics(df_test_region: pd.DataFrame, s_h_per_h: list) -> dict:
    cov_p = np.full(len(PC2A_HORIZONS), np.nan)
    wis_p = np.full(len(PC2A_HORIZONS), np.nan)
    cov_s = np.full(len(PC2A_HORIZONS), np.nan)
    wis_s = np.full(len(PC2A_HORIZONS), np.nan)
    for hi, h in enumerate(PC2A_HORIZONS):
        sub = df_test_region[df_test_region.horizon == h].sort_values("target_ep")
        if len(sub) == 0:
            continue
        mu = sub.mu.to_numpy()
        y = sub.y_true.to_numpy()
        s2 = sub.s2_total.to_numpy()
        Q_phase = phase_anchored_pi(mu, s2, PC2A_FLUSIGHT_23)
        Q_sh = sh_scalar_pi(mu, s_h_per_h[hi], PC2A_FLUSIGHT_23)
        cov_p[hi] = cov95(Q_phase, y); wis_p[hi] = wis_23(Q_phase, y)
        cov_s[hi] = cov95(Q_sh, y);    wis_s[hi] = wis_23(Q_sh, y)
    return dict(
        cov_phase=cov_p, wis_phase=wis_p, cov_sh=cov_s, wis_sh=wis_s,
        cov_phase_avg=float(np.nanmean(cov_p)),
        wis_phase_avg=float(np.nanmean(wis_p)),
        cov_sh_avg=float(np.nanmean(cov_s)),
        wis_sh_avg=float(np.nanmean(wis_s)),
    )


# ============================================================================
# #1 Region-cluster bootstrap
# ============================================================================
def cluster_bootstrap(per_region: dict, B: int = PC2A_BOOTSTRAP_B,
                       seed: int = PC2A_BOOTSTRAP_SEED,
                       level: float = PC2A_CI_LEVEL) -> tuple:
    """region 단위 cluster bootstrap. region 10개 with replacement resample, B 회."""
    regions = list(per_region.keys())
    vals = np.array([per_region[r] for r in regions], dtype=np.float64)
    n = len(vals)
    assert n == 10, f"expected 10 HHS clusters, got {n}: {regions}"
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = vals[idx].mean()
    alpha = (1 - level) / 2
    return (float(vals.mean()),
            float(np.percentile(boot, 100 * alpha)),
            float(np.percentile(boot, 100 * (1 - alpha))))


# ============================================================================
def run_pc2_a():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET_PATH)
    cgm = df[df.model == "cg_mamba"].copy()

    # === #② guardrail: data slicing ===
    val_nat = cgm[(cgm.region == "national") & (cgm.split == "val")]
    test_reg = cgm[(cgm.region.isin(PC2A_REGIONS_EVAL)) & (cgm.split == "test_strict")]
    assert val_nat.split.unique().tolist() == ["val"], f"val leak: {val_nat.split.unique()}"
    assert val_nat.region.unique().tolist() == ["national"], (
        f"region leak in val: {val_nat.region.unique()}")
    assert test_reg.split.unique().tolist() == ["test_strict"], (
        f"split leak in test: {test_reg.split.unique()}")
    assert "national" not in test_reg.region.unique(), (
        f"national leaked to eval: {test_reg.region.unique()}")
    assert set(test_reg.region.unique()) == set(PC2A_REGIONS_EVAL), (
        f"region set mismatch: {test_reg.region.unique()}")
    print("=== Data slicing (#② guardrail) ===")
    print(f"  s_h fit data:  national val, n={len(val_nat)} rows")
    print(f"  eval data:     {sorted(test_reg.region.unique())} × test_strict, "
          f"n={len(test_reg)} rows")

    # === #5 narrative lock: s_h fit-freeze on national val ===
    val_avg = (val_nat.groupby(["target_ep", "horizon"])
               .agg(mu=("mu", "mean"), y_true=("y_true", "first"))
               .reset_index())
    print(f"\n=== s_h fit-freeze (national val, 5-seed avg μ; per-horizon quantile-matching) ===")
    s_h_per_h = []
    for h in PC2A_HORIZONS:
        sub = val_avg[val_avg.horizon == h].sort_values("target_ep")
        mu_h = sub.mu.to_numpy(); y_h = sub.y_true.to_numpy()
        s_h = fit_sh_quantile_matching(mu_h, y_h, PC2A_FLUSIGHT_23, PC2A_SH_GRID)
        s_h_per_h.append(s_h)
        boundary_warn = ""
        if s_h <= PC2A_SH_GRID[0] + 1e-6 or s_h >= PC2A_SH_GRID[-1] - 1e-6:
            boundary_warn = "  ⚠ grid boundary"
        print(f"  s_h[h={h}] = {s_h:.4f}  (fit on n={len(mu_h)} origins, FROZEN){boundary_warn}")

    # === per-region metrics on regional test_strict (5-seed avg) ===
    test_avg = (test_reg.groupby(["region", "target_ep", "horizon"])
                .agg(mu=("mu", "mean"),
                     s2_total=("s2_total", "mean"),
                     y_true=("y_true", "first"))
                .reset_index())
    print(f"\n=== Per-region metrics (regional test_strict, 4-horizon avg) ===")
    print(f"  {'region':8s}  {'phase Cov95':>11s} {'phase WIS':>10s}  |  {'s_h Cov95':>10s} {'s_h WIS':>9s}  |  {'ΔCov95':>8s} {'ΔWIS':>8s}")
    region_metrics = {}
    for region in PC2A_REGIONS_EVAL:
        sub = test_avg[test_avg.region == region]
        m = per_region_metrics(sub, s_h_per_h)
        region_metrics[region] = m
        dcov = m['cov_phase_avg'] - m['cov_sh_avg']
        dwis = m['wis_phase_avg'] - m['wis_sh_avg']
        print(f"  {region:8s}  {m['cov_phase_avg']:>11.4f} {m['wis_phase_avg']:>10.4f}  |  "
              f"{m['cov_sh_avg']:>10.4f} {m['wis_sh_avg']:>9.4f}  |  "
              f"{dcov:>+8.4f} {dwis:>+8.4f}")

    # === #1 region-cluster bootstrap CI ===
    print(f"\n=== Region-cluster bootstrap (B={PC2A_BOOTSTRAP_B}, percentile 95% CI, seed={PC2A_BOOTSTRAP_SEED}) ===")
    cov_phase_per_r = {r: region_metrics[r]['cov_phase_avg'] for r in PC2A_REGIONS_EVAL}
    cov_sh_per_r    = {r: region_metrics[r]['cov_sh_avg']    for r in PC2A_REGIONS_EVAL}
    wis_phase_per_r = {r: region_metrics[r]['wis_phase_avg'] for r in PC2A_REGIONS_EVAL}
    wis_sh_per_r    = {r: region_metrics[r]['wis_sh_avg']    for r in PC2A_REGIONS_EVAL}
    dcov_per_r = {r: cov_phase_per_r[r] - cov_sh_per_r[r] for r in PC2A_REGIONS_EVAL}
    dwis_per_r = {r: wis_phase_per_r[r] - wis_sh_per_r[r] for r in PC2A_REGIONS_EVAL}

    cp_m, cp_lo, cp_hi = cluster_bootstrap(cov_phase_per_r)
    cs_m, cs_lo, cs_hi = cluster_bootstrap(cov_sh_per_r)
    wp_m, wp_lo, wp_hi = cluster_bootstrap(wis_phase_per_r)
    ws_m, ws_lo, ws_hi = cluster_bootstrap(wis_sh_per_r)
    dc_m, dc_lo, dc_hi = cluster_bootstrap(dcov_per_r)
    dw_m, dw_lo, dw_hi = cluster_bootstrap(dwis_per_r)

    print(f"  phase-anchored Cov95: {cp_m:.4f}  CI[{cp_lo:.4f}, {cp_hi:.4f}]")
    print(f"  s_h scalar     Cov95: {cs_m:.4f}  CI[{cs_lo:.4f}, {cs_hi:.4f}]")
    print(f"  ΔCov95 (phase−s_h):   {dc_m:+.4f}  CI[{dc_lo:+.4f}, {dc_hi:+.4f}]")
    print(f"  phase-anchored WIS:   {wp_m:.4f}  CI[{wp_lo:.4f}, {wp_hi:.4f}]")
    print(f"  s_h scalar     WIS:   {ws_m:.4f}  CI[{ws_lo:.4f}, {ws_hi:.4f}]")
    print(f"  ΔWIS (phase−s_h):     {dw_m:+.4f}  CI[{dw_lo:+.4f}, {dw_hi:+.4f}]  (negative = phase 우위)")

    # === LOCKED verdict (binary, narrative lock 그대로) ===
    excludes_0_cov = (dc_lo > 0) or (dc_hi < 0)
    excludes_0_wis = (dw_lo > 0) or (dw_hi < 0)
    phase_closer_nominal = abs(cp_m - 0.95) < abs(cs_m - 0.95)
    phase_wins_cov = excludes_0_cov and phase_closer_nominal
    phase_wins_wis = (dw_hi < 0)                                          # WIS 더 낮음 + CI < 0

    if phase_wins_wis or phase_wins_cov:
        verdict = ("PASS — phase-anchored 분산이 s_h scalar 를 transfer regime 에서 유의하게 이김. "
                   "**transfer 한정** mechanism 가치만 인정 (in-distribution tie 0.270 vs 0.268 그대로). "
                   "headline 부활 금지 — floor 본문 §IV-x_region 또는 §V-D mechanism note 한정.")
    else:
        verdict = ("FAIL — CI ∋ 0 또는 방향 어긋남. narrative tie 절대 금지 (락 §5). "
                   "HMM novelty 폐기, floor-full-negative 분기로 §2 reposition 진행.")

    print(f"\n=== PC2-a JUDGMENT (locked bar) ===")
    print(f"  ΔCov95 CI 0 제외: {excludes_0_cov}  |  phase 가 nominal(0.95) 에 더 가까움: {phase_closer_nominal}")
    print(f"  ΔWIS CI 0 제외(phase 우위 방향): {phase_wins_wis}")
    print(f"  → {verdict}")

    # Persist
    out = dict(
        locked_constants=dict(
            bootstrap_B=PC2A_BOOTSTRAP_B,
            bootstrap_seed=PC2A_BOOTSTRAP_SEED,
            ci_level=PC2A_CI_LEVEL,
            regions_eval=PC2A_REGIONS_EVAL,
            sh_grid=(float(PC2A_SH_GRID.min()), float(PC2A_SH_GRID.max()), int(len(PC2A_SH_GRID))),
        ),
        s_h_per_horizon=[float(s) for s in s_h_per_h],
        region_metrics={
            r: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
            for r, m in region_metrics.items()
        },
        bootstrap=dict(
            cov_phase=dict(mean=cp_m, lo=cp_lo, hi=cp_hi),
            cov_sh=dict(mean=cs_m, lo=cs_lo, hi=cs_hi),
            wis_phase=dict(mean=wp_m, lo=wp_lo, hi=wp_hi),
            wis_sh=dict(mean=ws_m, lo=ws_lo, hi=ws_hi),
            dcov=dict(mean=dc_m, lo=dc_lo, hi=dc_hi, excludes_0=excludes_0_cov),
            dwis=dict(mean=dw_m, lo=dw_lo, hi=dw_hi, excludes_0=excludes_0_wis),
        ),
        verdict=verdict,
    )
    out_path = OUT_DIR / "pc2_a_measurement.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None)
    print(f"\n  saved: {out_path}")
    return out


if __name__ == "__main__":
    run_pc2_a()
