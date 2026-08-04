"""kappa_recheck.py — γ.4 design-train HMM 재적합 + cross-seed κ + sanity ①②
================================================================================
목적:
  1. HMM 재적합 — design-train (seg2, 200240-201539) 만으로 multi-seed EM
  2. cross-seed Cohen's κ 측정 → γ.2 grid 분기 (κ≥0.8 → 45 / κ<0.8 → 90)
  3. sanity ①: κ-drop 해석 어휘 commit (data-size vs K-instability 구분)
  4. sanity ②: design-val state 점유율 collapse 체크

m1_4_phase_dynamics_search.py 의 multi-init EM 패턴 + cross_seed κ 재사용.
원 m1_4 와 차이: train 범위 만 cut (200240-201839 → 200240-201539).
"""
from __future__ import annotations
import json
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.models.gaussian_hmm import GaussianHMM
from src.utils.metrics import cohens_kappa_aligned, state_occupancy
from src.data.loader import load_norm_params
from scripts.m1_4_phase_dynamics_search import (
    featurize_raw, augment_features, RAW_COLS_V3,
)

# ============================================================================
# γ.4 LOCKED constants
# ============================================================================
DESIGN_TRAIN_START = 200240             # W40-2002 (seg2 start, post-gap)
DESIGN_TRAIN_END   = 201539             # W39-2015 (FluSight 평가시즌 -3 시즌)
DESIGN_VAL_START   = 201540             # W40-2015
DESIGN_VAL_END     = 201839             # W39-2018 (FluSight 시작 직전 주)
KAPPA_GRID_THRESHOLD = 0.8              # γ.2 분기점
HMM_SEEDS = [42, 123, 456]
N_INIT = 5
K = 3
V_RAW = 3
V_AUG = 2 * V_RAW
REG_COVAR = 5e-3
SPLIT_CSV = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_JSON = _ROOT / "data/processed/normalization_params.json"
OUT_DIR = _ROOT / "runs/m1_4_design_split"


def fit_multi_init(x_aug, seed, n_init=N_INIT):
    """m1_4 패턴: n_init 독립 EM 후 best-LL 선택."""
    best, best_ll, inits = None, -np.inf, []
    for i in range(n_init):
        init_seed = seed * 1000 + i
        cand = GaussianHMM(
            n_states=K, n_features=V_AUG, covariance_type="full",
            reg_covar=REG_COVAR, n_iter=200, tol=1e-4, seed=init_seed,
        ).fit(x_aug)
        ll = float(cand.ll_history[-1]) if cand.ll_history else -np.inf
        inits.append(dict(i=i, init_seed=init_seed, final_ll=ll,
                            n_iter=int(cand.n_iter_run)))
        if ll > best_ll:
            best, best_ll = cand, ll
    return best, best_ll, inits


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    norm = load_norm_params(NORM_JSON)
    df = pd.read_csv(SPLIT_CSV)

    # design-train (seg2-only) + design-val
    dt_df = df[(df["split"] == "train") &
               (df["epiweek"] >= DESIGN_TRAIN_START) &
               (df["epiweek"] <= DESIGN_TRAIN_END)].reset_index(drop=True)
    dv_df = df[(df["epiweek"] >= DESIGN_VAL_START) &
               (df["epiweek"] <= DESIGN_VAL_END)].reset_index(drop=True)

    print("=== Data slicing (γ.4) ===")
    print(f"  design-train (seg2): n={len(dt_df)} rows, ep[{int(dt_df.epiweek.min())}..{int(dt_df.epiweek.max())}]")
    print(f"  design-val:          n={len(dv_df)} rows, ep[{int(dv_df.epiweek.min())}..{int(dv_df.epiweek.max())}]")

    x_raw_dt = featurize_raw(dt_df, norm, RAW_COLS_V3)
    x_aug_dt = augment_features(x_raw_dt)
    x_raw_dv = featurize_raw(dv_df, norm, RAW_COLS_V3)
    x_aug_dv = augment_features(x_raw_dv)
    print(f"  x_aug design-train: {x_aug_dt.shape}")
    print(f"  x_aug design-val:   {x_aug_dv.shape}")

    # === Multi-seed HMM 재적합 ===
    print(f"\n=== HMM 재적합 (design-train only, K={K}, n_init={N_INIT}, reg_covar={REG_COVAR}) ===")
    hmms, viterbi_dt, occ_dt, inits = {}, {}, {}, {}
    for seed in HMM_SEEDS:
        hmm, ll, init_res = fit_multi_init(x_aug_dt, seed=seed)
        hmms[seed] = hmm
        viterbi_dt[seed] = hmm.viterbi(x_aug_dt)
        occ_dt[seed] = state_occupancy(viterbi_dt[seed], K=K)
        inits[seed] = init_res
        print(f"  seed {seed}: final_ll={ll:.2f}  occ_train={occ_dt[seed].round(3).tolist()}  μ_k(z-ili)={hmm.means[:, 0].round(3).tolist()}")
        seed_dir = OUT_DIR / f"V_raw3_regcov5e-03_K{K}_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.savez(seed_dir / "hmm_params.npz",
                  A=hmm.A, pi=hmm.pi, means=hmm.means, covars=hmm.covars,
                  reg_covar=hmm.reg_covar, K=hmm.K, V=hmm.V,
                  covariance_type=hmm.covariance_type,
                  n_iter_run=hmm.n_iter_run, final_ll=ll)
        np.save(seed_dir / "viterbi_path.npy", viterbi_dt[seed])

    # === Cross-seed κ on design-train ===
    print(f"\n=== Cross-seed Cohen's κ (design-train Viterbi) ===")
    pairs = []
    for s1, s2 in combinations(HMM_SEEDS, 2):
        k = float(cohens_kappa_aligned(viterbi_dt[s1], viterbi_dt[s2], K=K))
        pairs.append({"seeds": [s1, s2], "kappa": k})
        print(f"  seeds ({s1}, {s2}):  κ = {k:.4f}")
    kappas = [p["kappa"] for p in pairs]
    kappa_min = float(min(kappas))
    kappa_mean = float(np.mean(kappas))
    print(f"\n  κ_min  = {kappa_min:.4f}")
    print(f"  κ_mean = {kappa_mean:.4f}")

    # === γ.2 grid 분기 ===
    if kappa_min >= KAPPA_GRID_THRESHOLD:
        grid_status = f"PASS (κ_min {kappa_min:.4f} ≥ {KAPPA_GRID_THRESHOLD})"
        grid_decision = "K=3 고정 → grid = 9 configs × 5 seeds = 45 runs"
        n_configs = 9
    else:
        grid_status = f"FAIL (κ_min {kappa_min:.4f} < {KAPPA_GRID_THRESHOLD})"
        grid_decision = "K∈{3, 4} 추가 → grid = 18 configs × 5 seeds = 90 runs"
        n_configs = 18
    print(f"\n=== γ.2 grid 분기 결정 ===")
    print(f"  {grid_status}")
    print(f"  → {grid_decision}")

    # === Sanity ② — design-val state 점유율 ===
    print(f"\n=== Sanity ② — design-val state 점유율 (design-train HMM 의 Viterbi 적용) ===")
    dv_results = {}
    any_collapse = False
    for seed in HMM_SEEDS:
        path_dv = hmms[seed].viterbi(x_aug_dv)
        occ_dv = state_occupancy(path_dv, K=K)
        collapsed = [int(k) for k in range(K) if occ_dv[k] < 0.01]
        dv_results[seed] = dict(occ=occ_dv.tolist(), collapsed=collapsed)
        flag = "OK" if not collapsed else f"⚠ COLLAPSE state {collapsed}"
        print(f"  seed {seed}: occ_design-val = {occ_dv.round(3).tolist()}   {flag}")
        if collapsed:
            any_collapse = True
    if any_collapse:
        print(f"\n  ⚠ design-val state collapse 감지 — phase feature selection 가치 약한 신호 (정보용, γ.2 결정 무관)")
    else:
        print(f"\n  ✓ design-val state collapse 없음 — phase feature selection 의미 유지 (PC0 의 regional collapse 와 별개)")

    # === Sanity ① — κ-drop 해석 어휘 commit (사후 변경 금지) ===
    print(f"\n=== Sanity ① — κ 해석 어휘 commit (사후 변경 금지) ===")
    if kappa_min >= KAPPA_GRID_THRESHOLD:
        kappa_interp = (
            f"κ_min={kappa_min:.4f} ≥ {KAPPA_GRID_THRESHOLD} → K=3 가 design-train(seg2 {len(x_aug_dt)} pts) "
            f"위에서도 안정. 데이터 크기 -3 시즌 변화에 robust. K=3 고정 정당."
        )
    else:
        kappa_interp = (
            f"κ_min={kappa_min:.4f} < {KAPPA_GRID_THRESHOLD} → 가능 원인 둘: "
            f"(A) K=3 instability — design-train 크기 무관, K 자체 부적합. "
            f"(B) data-size effect — design-train ({len(x_aug_dt)} pts) 이 원 train ({len(x_aug_dt)}+...) 보다 작아 EM 덜 안정. "
            f"원인 분리 불가하므로 보수적으로 grid 90 (K∈{{3,4}}) 진행. paper §V-D 의 K-selection robustness 한계 명시 필수."
        )
    print(f"  {kappa_interp}")

    # === Persist ===
    out = dict(
        locked_constants=dict(
            design_train_epiweek=[DESIGN_TRAIN_START, DESIGN_TRAIN_END],
            design_val_epiweek=[DESIGN_VAL_START, DESIGN_VAL_END],
            kappa_threshold=KAPPA_GRID_THRESHOLD,
            seeds=HMM_SEEDS, n_init=N_INIT, K=K, V_raw=V_RAW,
            reg_covar=REG_COVAR,
        ),
        data_sizes=dict(
            design_train_seg2_rows=int(len(dt_df)),
            design_val_rows=int(len(dv_df)),
            x_aug_design_train_shape=list(x_aug_dt.shape),
            x_aug_design_val_shape=list(x_aug_dv.shape),
        ),
        per_seed=[
            dict(
                seed=seed,
                final_ll=float(hmms[seed].ll_history[-1]),
                means_zili=hmms[seed].means[:, 0].tolist(),
                occupancy_design_train=occ_dt[seed].tolist(),
                occupancy_design_val=dv_results[seed]["occ"],
                state_collapse_design_val=dv_results[seed]["collapsed"],
                init_results=inits[seed],
            ) for seed in HMM_SEEDS
        ],
        cross_seed_kappa=dict(
            pairs=pairs,
            kappa_min=kappa_min,
            kappa_mean=kappa_mean,
        ),
        grid_decision=dict(
            kappa_min=kappa_min, threshold=KAPPA_GRID_THRESHOLD,
            status=grid_status, n_configs=n_configs,
            total_runs=n_configs * 5, decision=grid_decision,
        ),
        sanity_design_val_collapse_detected=any_collapse,
        kappa_interpretation=kappa_interp,
    )
    out_path = OUT_DIR / "kappa_recheck.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved: {out_path}")
    return out


if __name__ == "__main__":
    main()
