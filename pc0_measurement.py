"""pc0_measurement.py — PC0 사전등록 측정 (LOCKED)
================================================================================
가설: CGM 전이 보정의 *이유* = phase-emission 통계(μ_k, σ_k)가 지역간 안정 →
분산이 타깃에 fit되지 않고 모델에 실려 따라감. PC0 = 기계론 사망 필터.

LOCKED bar (수정 금지, 결과 후 변경 금지):
- PASS:  지배상태(stationary π ≥ 0.15) median CV(σ_k) < 0.25 AND state 순서 보존
         → prior *무갱신* ("기계론이 죽지 않았다"일 뿐, evidence-for 아님), PC1 진행
- CLEAR FAIL: median CV(σ_k) > 0.40 → floor 직행
- GAP (0.25, 0.40): proceed with caveat — *재결정 점 아님*, PC1 진행
- 상태정체성 붕괴: → floor

LOCKED 운영 상수 #4 (실행 락):
- σ_k = z-ili 차원 상태별 표준편차 (diag, NOT trace/Frobenius)
- 지배상태 = national stationary probability ≥ 0.15
- 할당 = Viterbi MAP hard
- CV gap (0.25, 0.40) = "proceed with caveat" *재결정 점 아님*
- soft posterior / full-cov / top-1 dominant = 부록 sensitivity로만

Run: python pc0_measurement.py
Out: runs/pc0/pc0_measurement.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from regime_shift_drivers import NORM, _build_region_df, HMM_TPL  # noqa: E402
from src.utils.checkpoints import load_fitted_hmm                  # noqa: E402
from scripts.m1_4_phase_dynamics_search import (                   # noqa: E402
    featurize_raw, augment_features, RAW_COLS_V3,
)

# ============================================================================
# LOCKED PC0 CONSTANTS (실행 락 — 결과 본 후 수정 금지)
# ============================================================================
PC0_CV_PASS = 0.25                              # median CV < 이면 PASS
PC0_CV_CLEAR_FAIL = 0.40                        # median CV > 이면 CLEAR FAIL (floor 직행)
PC0_DOMINANT_PI_THRESHOLD = 0.15                # stationary π ≥ 이면 지배상태
PC0_HMM_SEED = 42                               # frozen national HMM (K-selection seed-invariant)
PC0_MIN_STATE_OCCUPANCY_PER_REGION = 5          # state 당 ≥ 5 weeks 있어야 σ 계산 (ddof=1 안정)
PC0_MIN_REGIONS_PER_STATE_FOR_CV = 3            # state 당 ≥ 3 region 있어야 CV 신뢰
PC0_REGIONS = ["national"] + [f"hhs{i}" for i in range(1, 11)]
OUT_DIR = _ROOT / "runs/pc0"


# ============================================================================
# Frozen HMM utilities
# ============================================================================
def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """A [K, K] → π ∈ Δ^{K-1} (left eigenvector of A with eigenvalue 1)."""
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)                                       # eigvec sign arbitrary; magnitude OK
    s = pi.sum()
    assert s > 1e-12 and np.all(pi >= -1e-12), (
        f"stationary distribution invalid: pi={pi}, sum={s}")
    return pi / s


def viterbi_hard_assign(
    x_aug: np.ndarray, A: np.ndarray, pi: np.ndarray,
    means: np.ndarray, covars: np.ndarray,
) -> np.ndarray:
    """Viterbi MAP hard assignment in log space using full covariance emissions.

    Args:
        x_aug: [T, V_aug] augmented features
        A: [K, K] transition, pi: [K] initial, means: [K, V_aug], covars: [K, V_aug, V_aug]
    Returns:
        path: [T] int64 hard state assignment
    """
    T = x_aug.shape[0]
    K = A.shape[0]
    # Log-emission [T, K]
    log_emit = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        rv = multivariate_normal(mean=means[k], cov=covars[k], allow_singular=True)
        log_emit[:, k] = rv.logpdf(x_aug)
    log_pi = np.log(np.clip(pi, 1e-30, None))
    log_A = np.log(np.clip(A, 1e-30, None))
    # Forward pass
    delta = np.full((T, K), -np.inf, dtype=np.float64)
    psi = np.zeros((T, K), dtype=np.int64)
    delta[0] = log_pi + log_emit[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A          # [K_prev, K_curr]
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(K)] + log_emit[t]
    # Guard: singular emissions can make entire path -inf
    if not np.isfinite(delta[-1]).any():
        raise ValueError("Viterbi divergence: all final-state log-likelihoods -inf "
                          "(likely singular covariance / extreme outlier)")
    # Backward trace
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = int(psi[t + 1, path[t + 1]])
    return path


# ============================================================================
# Per-region emission statistics
# ============================================================================
def per_region_emission_stats(region: str, hmm) -> dict:
    """Frozen national HMM Viterbi 적용 → per-state (μ, σ) on z-ili 차원."""
    df = _build_region_df(region)
    x_raw = featurize_raw(df, NORM, RAW_COLS_V3)         # [T, V_raw=3]
    x_aug = augment_features(x_raw)                       # [T-1, V_aug=6]
    if x_aug.shape[0] == 0:
        raise ValueError(f"{region}: empty x_aug after augment_features "
                          f"(raw rows = {x_raw.shape[0]})")
    path = viterbi_hard_assign(x_aug, hmm.A, hmm.pi, hmm.means, hmm.covars)
    z_ili = x_aug[:, 0]                                   # z-ili dim
    K = hmm.A.shape[0]
    T = len(path)
    out = {}
    for k in range(K):
        mask = path == k
        n_k = int(mask.sum())
        if n_k < PC0_MIN_STATE_OCCUPANCY_PER_REGION:
            out[int(k)] = dict(
                n=n_k, mu=float("nan"), sigma=float("nan"),
                occupancy=n_k / T,
            )
        else:
            out[int(k)] = dict(
                n=n_k,
                mu=float(z_ili[mask].mean()),
                sigma=float(z_ili[mask].std(ddof=1)),
                occupancy=n_k / T,
            )
    return out


# ============================================================================
# PC0 judgment (binary, locked)
# ============================================================================
def judge(median_cv: float, order_preserved: bool) -> str:
    """LOCKED bar 그대로 적용 — 재해석 금지."""
    if not np.isfinite(median_cv):
        return "INVALID — dominant state 당 region 수 부족 (≥3 필요)"
    if median_cv > PC0_CV_CLEAR_FAIL:
        return f"CLEAR FAIL (median CV {median_cv:.4f} > {PC0_CV_CLEAR_FAIL}) → floor 직행"
    if not order_preserved:
        return "FAIL (state identity collapse in dominant states) → floor"
    if median_cv < PC0_CV_PASS:
        return (f"PASS (median CV {median_cv:.4f} < {PC0_CV_PASS}) — "
                "*prior 무갱신* (기계론 사망 안 함), PC1 진행")
    return (f"GAP ({PC0_CV_PASS} ≤ median CV {median_cv:.4f} ≤ {PC0_CV_CLEAR_FAIL}) — "
            "proceed with caveat (*재결정 점 아님*), PC1 진행")


# ============================================================================
def run_pc0():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Frozen national HMM
    hmm_dir = Path(str(HMM_TPL).format(seed=PC0_HMM_SEED))
    hmm = load_fitted_hmm(hmm_dir)
    K = hmm.A.shape[0]
    pi_stat = stationary_distribution(hmm.A)
    dominant = [k for k in range(K) if pi_stat[k] >= PC0_DOMINANT_PI_THRESHOLD]

    print(f"=== Frozen national HMM (seed {PC0_HMM_SEED}) ===")
    print(f"  K = {K}")
    print(f"  stationary π = {pi_stat.round(4).tolist()}")
    print(f"  dominant states (π ≥ {PC0_DOMINANT_PI_THRESHOLD}): {dominant}")
    print(f"  national-fit means μ_k (z-ili dim, x_aug[:, 0]): {hmm.means[:, 0].round(3).tolist()}")
    print()

    # Per-region stats
    print(f"=== Per-region emission stats (Viterbi MAP hard, z-ili diag std) ===")
    region_stats = {}
    for region in PC0_REGIONS:
        try:
            s = per_region_emission_stats(region, hmm)
        except Exception as e:
            print(f"  {region}: FAIL {type(e).__name__}: {e}")
            region_stats[region] = None
            continue
        region_stats[region] = s
        occ = "  ".join(f"k{k}:{s[k]['occupancy']:.3f}" for k in range(K))
        mu_s = "  ".join(
            (f"k{k}:{s[k]['mu']:>+6.3f}" if not np.isnan(s[k]['mu']) else f"k{k}:   NaN")
            for k in range(K))
        sig_s = "  ".join(
            (f"k{k}:{s[k]['sigma']:.3f}" if not np.isnan(s[k]['sigma']) else f"k{k}:  NaN")
            for k in range(K))
        print(f"  {region:10s}  occ[{occ}]  μ[{mu_s}]  σ[{sig_s}]")

    # State identity (rank order of μ_k restricted to dominant states)
    print(f"\n=== State identity (μ_k rank order in dominant states, low→high) ===")
    rankings = {}
    for region, s in region_stats.items():
        if s is None:
            rankings[region] = None
            continue
        mus_dom = np.array([s[k]['mu'] for k in dominant])
        if np.isnan(mus_dom).any():
            rankings[region] = None
            continue
        rankings[region] = tuple(int(k) for k in sorted(dominant, key=lambda k: s[k]['mu']))
    ref = rankings.get("national")
    print(f"  reference (national): {ref}")
    if ref is None:
        print("  ⚠ national reference unobtainable — entire PC0 INVALID")
        order_preserved = False
        regions_diverging = []
    else:
        regions_diverging = []
        for region, r in rankings.items():
            if region == "national":
                continue
            if r is None:
                regions_diverging.append((region, "no_data"))
                print(f"  {region:10s}  ranking=None (no_data)")
                continue
            ok = r == ref
            flag = "OK" if ok else "DIFFER"
            print(f"  {region:10s}  ranking={r}  {flag}")
            if not ok:
                regions_diverging.append((region, str(r)))
        order_preserved = (len(regions_diverging) == 0)

    # CV(σ_k) per dominant state across regions
    print(f"\n=== CV(σ_k) across regions, dominant states only ===")
    cv_by_state = {}
    for k in dominant:
        sigs, used_regions = [], []
        for region, s in region_stats.items():
            if s is None:
                continue
            sig = s[k]['sigma']
            if np.isnan(sig):
                continue
            sigs.append(sig); used_regions.append(region)
        sigs = np.array(sigs)
        if len(sigs) < PC0_MIN_REGIONS_PER_STATE_FOR_CV:
            print(f"  state {k}: n_regions={len(sigs)} < {PC0_MIN_REGIONS_PER_STATE_FOR_CV} → CV=NaN")
            cv_by_state[k] = float("nan")
            continue
        cv_k = float(sigs.std(ddof=1) / sigs.mean()) if sigs.mean() > 1e-12 else float("nan")
        cv_by_state[k] = cv_k
        print(f"  state {k}: n_regions={len(sigs)}  mean σ={sigs.mean():.4f}  "
              f"std σ={sigs.std(ddof=1):.4f}  CV={cv_k:.4f}  regions={used_regions}")

    valid_cvs = [v for v in cv_by_state.values() if np.isfinite(v)]
    median_cv = float(np.median(valid_cvs)) if valid_cvs else float("nan")
    print(f"\n  median CV across dominant states = {median_cv:.4f}")

    # Judgment (locked, binary)
    print(f"\n=== PC0 JUDGMENT (locked bar, no post-hoc adjustment) ===")
    print(f"  bar: PASS<{PC0_CV_PASS}, CLEAR FAIL>{PC0_CV_CLEAR_FAIL}")
    print(f"  measured median CV = {median_cv:.4f}")
    print(f"  state identity preserved (dominant): {order_preserved}")
    verdict = judge(median_cv, order_preserved)
    print(f"  → {verdict}")

    # Persist artifact
    def _json_safe(x):
        if isinstance(x, float) and np.isnan(x):
            return None
        return x
    out = dict(
        locked_constants=dict(
            cv_pass=PC0_CV_PASS,
            cv_clear_fail=PC0_CV_CLEAR_FAIL,
            dominant_pi_threshold=PC0_DOMINANT_PI_THRESHOLD,
            hmm_seed=PC0_HMM_SEED,
            min_state_occupancy_per_region=PC0_MIN_STATE_OCCUPANCY_PER_REGION,
            min_regions_per_state_for_cv=PC0_MIN_REGIONS_PER_STATE_FOR_CV,
            sigma_definition="diag std on z-ili dim x_aug[:, 0]",
            assignment="viterbi MAP hard, full covariance emission",
        ),
        K=K,
        stationary_pi=pi_stat.tolist(),
        dominant_states=dominant,
        national_means_zili=hmm.means[:, 0].tolist(),
        region_stats={
            r: ({k: {kk: _json_safe(vv) for kk, vv in s[k].items()} for k in s}
                if s is not None else None)
            for r, s in region_stats.items()
        },
        rankings={r: (list(rk) if rk else None) for r, rk in rankings.items()},
        regions_diverging_state_identity=regions_diverging,
        cv_by_state={int(k): _json_safe(float(v)) for k, v in cv_by_state.items()},
        median_cv_dominant=_json_safe(median_cv),
        order_preserved=order_preserved,
        verdict=verdict,
    )
    out_path = OUT_DIR / "pc0_measurement.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None)
    print(f"\n  saved: {out_path}")
    return out


if __name__ == "__main__":
    run_pc0()
