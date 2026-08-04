"""
regime_shift_experiment.py  (v2 — build_load_data 연동 + epiweek/row 인덱싱 수정)
================================================================================
conformal 로 calibration 을 동등화한 뒤, 전환(regime-shift) 구간에서
CG-Mamba backbone 이 Vanilla Mamba 를 이기는지 검정한다. (reject 1, 2 양면 방어)

실행:  (regime_shift_drivers.py 가 같은 디렉토리에 있어야 함)
    python regime_shift_experiment.py

— v2 핵심 수정 (조용한 결과 오염 방지):
  (1) epiweek 산술 금지 — 202252 다음이 202301 이므로 t+h 의 정수산술이 연말에 깨짐.
      → γ 의 target_ep 배열을 master clock 으로, 모든 윈도우를 row-index 공간에서 계산.
  (2) gamma_smoothed_fn 이 (gamma, epiweek) 튜플 반환 → 언팩 + ep2idx 매핑.
  (3) no_env arm 추가 (NO_ENV) + NO_ENV vs NO_ENCGATES 대비 (phase-env 결합 probe).
  (4) observed wILI 를 γ 타임라인에 정렬 → 불일치 시 큰 소리로 assert (조용한 오염 방지).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

# regime_shift_drivers.build_load_data 가 데이터 dict 를 공급
from regime_shift_drivers import build_load_data
def load_data():
    return build_load_data()

# ----------------------------------------------------------------------------
FLUSIGHT_23 = np.array([0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                        0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                        0.85, 0.90, 0.95, 0.975, 0.99])
HORIZONS = [1, 2, 3, 4]
REGIONS  = ["national"] + [f"hhs{i}" for i in range(1, 11)]
REGIONAL = [f"hhs{i}" for i in range(1, 11)]          # 1차 종속 지역 트랙(n=10)
MODELS   = ["CGM", "VANILLA", "NO_ENCGATES", "NO_ENV"]
MA_WINDOW = 3
PROMINENCE_SIGMA = 0.5
COV_DIVERGENCE_PP = 0.03

# — 정확한 Table III 비교성을 원하면 여기에 src/eval/wis.py 의 함수를 연결.
#   (모델 간 비교는 동일 함수면 일관되므로 reference 로도 결론적 유효)
def wis_23(quantiles: np.ndarray, y: float) -> float:
    q = np.asarray(quantiles, float)
    tau = FLUSIGHT_23
    pinball = np.where(y >= q, tau * (y - q), (1 - tau) * (q - y))
    return 2.0 * pinball.mean()

def conformal_offsets(val_residuals: np.ndarray) -> np.ndarray:
    """national val 잔차(=y-mu)의 23-level 분위수 = 점예측에 더할 오프셋(전역 고정)."""
    return np.quantile(np.asarray(val_residuals, float), FLUSIGHT_23)

def conformal_quantiles(point: float, offsets: np.ndarray) -> np.ndarray:
    return point + offsets

def covered95(quantiles: np.ndarray, y: float) -> bool:
    lo = quantiles[np.where(FLUSIGHT_23 == 0.025)[0][0]]
    hi = quantiles[np.where(FLUSIGHT_23 == 0.975)[0][0]]
    return bool(lo <= y <= hi)

# ---- 전환 라벨링: 전부 row-index 공간 -------------------------------------
def turning_point_rows(y: np.ndarray, prominence: float) -> set:
    """모델-프리 변곡점 row 집합 (피크 + prominence). 끝단 rolling NaN 으로 제외."""
    ys = pd.Series(y).rolling(MA_WINDOW, center=True).mean().to_numpy()
    valid = ~np.isnan(ys)
    idx = np.where(valid)[0]
    ysv = ys[valid]
    peaks, _   = find_peaks(ysv,  prominence=prominence)
    troughs, _ = find_peaks(-ysv, prominence=prominence)
    return set(idx[np.concatenate([peaks, troughs])].tolist())

def _boundary_crossing(phase: np.ndarray, o: int, t: int, persist: int) -> bool:
    """phase[o..t] 에서 dominant 상태가 양쪽 persist 주 유지하며 바뀌면 True."""
    seg = phase[o:t + 1]
    if len(seg) < 2:
        return False
    for i in range(1, len(seg)):
        if seg[i] != seg[i - 1]:
            gpos = o + i
            left_ok  = np.all(phase[max(0, gpos - persist):gpos] == seg[i - 1])
            right_ok = np.all(phase[gpos:gpos + persist] == seg[i])
            if left_ok and right_ok:
                return True
    return False

def cohens_kappa(a, b) -> float:
    a = np.array([1 if x == "transition" else 0 for x in a])
    b = np.array([1 if x == "transition" else 0 for x in b])
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe + 1e-12))

# ============================================================================
# 채점 — model × region × horizon × origin (전부 row-index 공간)
# ============================================================================
def score_all(D) -> pd.DataFrame:
    offsets = {(m, h): conformal_offsets(D["national_val_residuals"][m][h])
               for m in MODELS for h in HORIZONS if m in D["national_val_residuals"]}
    rows = []
    for region in REGIONS:
        gamma, ep = D["gamma_smoothed_fn"](region)                 # (2) 튜플 언팩
        ep2idx = {int(e): i for i, e in enumerate(ep)}
        phase  = np.asarray(gamma).argmax(1)
        obs    = np.asarray(D["observed_wILI"][region], float)
        assert len(obs) == len(ep), (                              # (4) 정렬 강제
            f"[{region}] observed_wILI len {len(obs)} != gamma len {len(ep)}; "
            "observed wILI 를 γ epiweek 타임라인에 정렬하라")
        tp_rows = turning_point_rows(obs, PROMINENCE_SIGMA * D["sd_train"])

        for m in MODELS:
            pf = D["point_forecast"].get(m, {}).get(region, {})
            for t in D["origins_fn"](region):                      # t = target_ep
                for h in HORIZONS:
                    if (t, h) not in pf:
                        continue
                    tr = ep2idx.get(int(t))
                    if tr is None:
                        continue
                    orow = tr - h                                  # (1) row 공간, 연경계 안전
                    if orow < 0:
                        continue
                    lab_free = "transition" if any(orow <= r <= tr for r in tp_rows) else "stable"
                    lab_hmm  = "transition" if _boundary_crossing(phase, orow, tr, max(1, h - 1)) else "stable"
                    Q = conformal_quantiles(pf[(t, h)], offsets[(m, h)])
                    y = D["y_at_fn"](region, t, h)
                    sw = sb = st = np.nan
                    if m == "CGM":
                        comp = D["apmd_components"].get(region, {}).get((t, h))
                        if comp is not None:
                            sw, sb, st = comp
                    rows.append(dict(model=m, region=region, target_ep=int(t), horizon=h,
                                     label_free=lab_free, label_hmm=lab_hmm,
                                     wis=wis_23(Q, y), cov=covered95(Q, y),
                                     s2_within=sw, s2_between=sb, s2_total=st))
    return pd.DataFrame(rows)

# ============================================================================
# 집계 — pseudoreplication-safe (region = 독립 단위, n=10)
# ============================================================================
def aggregate(df, regions, label_col="label_free", treat="CGM", ctrl="VANILLA", subset="transition"):
    sub = df[(df[label_col] == subset) & (df.region.isin(regions))]
    per = sub.groupby(["region", "model"])["wis"].mean().unstack("model")
    if treat not in per or ctrl not in per:
        return dict(n=0, mean_delta=np.nan, wilcoxon_p=np.nan, boot_ci=(np.nan, np.nan),
                    rank_biserial=np.nan, favors_treat=0, per_region=pd.Series(dtype=float))
    delta = (per[ctrl] - per[treat]).dropna()                      # Δ>0 = treat 우세
    n = len(delta)
    try:
        w_p = stats.wilcoxon(delta.values).pvalue if n >= 1 and np.any(delta.values != 0) else np.nan
    except ValueError:
        w_p = np.nan
    rng = np.random.default_rng(0)
    boot = [rng.choice(delta.values, size=n, replace=True).mean() for _ in range(10000)] if n else []
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5)) if boot else (np.nan, np.nan)
    return dict(n=n, mean_delta=float(delta.mean()) if n else np.nan, wilcoxon_p=float(w_p),
                boot_ci=ci, rank_biserial=_rank_biserial(delta.values),
                favors_treat=int((delta > 0).sum()), per_region=delta)

def _rank_biserial(d):
    d = np.asarray(d); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())

def coverage_diagnostic(df, regions, subset="transition", treat="CGM", ctrl="VANILLA"):
    sub = df[(df.label_free == subset) & (df.region.isin(regions))]
    cov = sub.groupby("model")["cov"].mean()
    diff = abs(cov.get(treat, np.nan) - cov.get(ctrl, np.nan))
    return dict(cov_treat=float(cov.get(treat, np.nan)), cov_ctrl=float(cov.get(ctrl, np.nan)),
                abs_diff=float(diff), needs_per_regime_sensitivity=bool(diff > COV_DIVERGENCE_PP))

def decomposition_table(df, regions):
    """Table D — CGM 적용. subset별 mean r = mean(σ²_between/σ²_total)."""
    cgm = df[(df.model == "CGM") & (df.region.isin(regions))].copy()
    cgm["r"] = cgm.s2_between / (cgm.s2_total + 1e-12)
    t = cgm.groupby("label_free").agg(mean_within=("s2_within", "mean"),
                                      mean_between=("s2_between", "mean"),
                                      mean_r=("r", "mean"))
    t["between_pct"] = t.mean_between / (t.mean_within + t.mean_between) * 100
    t["r_gt_0.3 (eq.15 조건)"] = t.mean_r > 0.3
    return t

# ============================================================================
def main():
    D = load_data()
    df = score_all(D)

    # 검정력 먼저 — per-horizon 전환 표본 수 (지역 트랙)
    reg = df[df.region.isin(REGIONAL)]
    print("=== 전환 표본 수 (지역 트랙, per-horizon, model-free) ===")
    print(reg[reg.model == "CGM"].groupby(["horizon", "label_free"]).size().unstack(fill_value=0))
    print(f"\n=== 라벨 일치도 (Table E) Cohen's κ(model-free vs HMM) = "
          f"{cohens_kappa(reg.label_free, reg.label_hmm):.3f} ===")

    # 1차: model-free × 지역 × CGM vs Vanilla
    primary = aggregate(df, REGIONAL, treat="CGM", ctrl="VANILLA", subset="transition")
    stable  = aggregate(df, REGIONAL, treat="CGM", ctrl="VANILLA", subset="stable")
    cov     = coverage_diagnostic(df, REGIONAL)
    # secondary: phase-env 결합 probe
    cgm_vs_noenc = aggregate(df, REGIONAL, treat="CGM", ctrl="NO_ENCGATES", subset="transition")
    noenv_vs_noenc = aggregate(df, REGIONAL, treat="NO_ENV", ctrl="NO_ENCGATES", subset="transition")
    decomp  = decomposition_table(df, REGIONAL)

    def fmt(a): return (f"n={a['n']} meanΔ={a['mean_delta']:.4f} p={a['wilcoxon_p']:.4f} "
                        f"CI=({a['boot_ci'][0]:.4f},{a['boot_ci'][1]:.4f}) rb={a['rank_biserial']:.3f} "
                        f"favors={a['favors_treat']}/{a['n']}")
    print("\n=== Table A/B (1차: 전환, CGM vs Vanilla) ===\n " + fmt(primary))
    print(f"[안정 대비] CGM vs Vanilla meanΔ={stable['mean_delta']:.4f}  (H2: aggregate tie ≠ subset tie)")
    print(f"[Table C] Cov95 CGM={cov['cov_treat']:.3f} Vanilla={cov['cov_ctrl']:.3f} "
          f"|Δ|={cov['abs_diff']:.3f}  per-regime sensitivity 필요={cov['needs_per_regime_sensitivity']}")
    print("\n=== secondary arms ===")
    print(" CGM vs NO_ENCGATES (encoder gate 통째): " + fmt(cgm_vs_noenc))
    print(" NO_ENV vs NO_ENCGATES (phase-env 결합): " + fmt(noenv_vs_noenc))
    print("   ↳ Δ>0 = no_env 우세 = '전환에서 phase-conditional env 가 값' (critique #7 반박)")
    print("   ↳ Δ≤0 = 곱셈결합이 전환에서도 destructive (phase-env 주장 약함; 헤드라인은 CGM vs Vanilla)")
    print("\n=== Table D (분해 자기일관성, CGM 적용) ===")
    print(decomp)

    # 시나리오 분기 (§5)
    win_v = (primary["wilcoxon_p"] < 0.05) and (primary["mean_delta"] > 0)
    print("\n=== 결과 시나리오 ===")
    if win_v and cov["abs_diff"] <= COV_DIVERGENCE_PP:
        print(" CLEAR WIN → Table A/B 본문화, §IV-B Finding 2 reframe, §V-A 재서술. (이 reject 두 번 해석)")
    elif win_v:
        print(" WIN(coverage 갈림) → per-regime conformal sensitivity 첨부 후 본문화.")
    elif primary["n"] >= 10:
        print(" NULL(검정력 적절) → PIVOT: '추가비용 없는 decomposable calibration + interpretability'.")
    else:
        print(" INCONCLUSIVE → 검정력 부족 가능. power deficit 공개 + 보강 후 재실행. NULL과 혼동 금지.")
    print("\n(N3 이중 분포변화·전환 독립 사건 수·재정 coverage drift 한계는 §9 로 공개.)")

if __name__ == "__main__":
    main()
