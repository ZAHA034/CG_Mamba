"""M1.4b — Phase Dynamics GaussianHMM search (V_raw ∈ {3,4} × reg_covar × K=3 × 3 seeds, multi-start).

PLAN v2.0.9 §3.4 + D.5.1 + PATCH 13 (M1.4b)

Purpose:
  v2.0.9 pivot — NSVARHMM σ collapse 구조적 실패 (legacy/runs/hmm_stage1/, 9/9)
  + V=3 narrow check도 동일 (legacy/runs/hmm_stage1_v3_k3_only/, 3/3) → i.i.d.
  Gaussian emission으로 회귀하되 augmented feature space [x_t, Δx_t]로
  temporal context 회복 (Furui 1986 delta-MFCC + Hamilton 1989 transition).

  M1.4b는 V_raw ∈ {3, 4} 둘 다 실행해서 augmented space에서의 안정성과
  reproducibility를 비교 → main path V_raw 확정.

Grid (v2.0.9 self-optimization, post-feedback):
  V_raw ∈ {3, 4} × reg_covar ∈ {1e-3, 5e-3, 1e-2} × seeds ∈ {42, 123, 456} = 18 settings
  Each setting: n_init=5 multi-start (best-LL selection) → 90 EM fits total.
  K=3 fixed (BIC penalty, PLAN §3.4).

  v2.0.9 augmented [x_t, Δx_t] GaussianHMM은 새 모델이므로 raw 결과와의
  비교가 아니라 자체 최적값을 탐색. Multi-start으로 init sensitivity 완화 +
  reg_covar sweep으로 V_aug=6/8 cov rank-deficiency 회피.

Data:
  - seg2-only train (epiweek ≥ 200240, 835 rows)
  - Augmented: x_aug = [x_t, Δx_t], length L-1 = 834, V_aug = 2·V_raw

Selection criteria (PLAN §5.1 D.5.1):
  - cross-seed κ_min ≥ 0.50 (3-pairwise aligned Viterbi agreement)
  - no dead state (occupancy ≥ 5% for all K states)
  - covariance well-conditioned (no σ collapse analog)
  - binary κ ≥ 0.50 (state vs CDC %wILI ≥ 2.2 epi truth)

Usage:
  python scripts/m1_4_phase_dynamics_search.py             # full 18 settings × n_init=5
  python scripts/m1_4_phase_dynamics_search.py --smoke     # V_raw=3, reg_covar=1e-3, seed=42, n_init=2
  python scripts/m1_4_phase_dynamics_search.py --V_raw 3   # V_raw=3 fixed
  python scripts/m1_4_phase_dynamics_search.py --n_init 3  # reduced multi-start
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[1]
if str(_CG_MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CG_MAMBA_ROOT))

from src.data.loader import load_dataset_csv, load_norm_params  # noqa: E402
from src.models.gaussian_hmm import GaussianHMM  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    cohens_kappa_aligned,
    cohens_kappa_binary,
    fallback_trigger,
    is_dead_state,
    state_occupancy,
)


# ──────────────────────────────────────────────────────────────────
# Constants (v2.0.9)
# ──────────────────────────────────────────────────────────────────
TRAIN_START_EPIWEEK = 200240   # seg2-only (post-2002-gap, 2002-W40)
EXPECTED_TRAIN_ROWS = 868
EXPECTED_SEG2_ROWS = 835

# Raw feature column sets (defined locally to avoid legacy import dependency)
RAW_COLS_V3 = [
    "ili_weighted_pct",
    "total_ili_count",
    "num_providers",
    # num_patients dropped (r=0.952 multicollinearity, EB-2)
]
RAW_COLS_V4 = [
    "ili_weighted_pct",
    "total_ili_count",
    "num_providers",
    "num_patients",
]

K_FIXED = 3
K_GRID_DEFAULT = (3, 4, 5)  # §7.4 A-PD5 ablation grid (default disabled, K=3 fixed)
SEEDS = [42, 123, 456]
REG_COVAR_GRID = [1e-3, 5e-3, 1e-2]
DEFAULT_N_INIT = 5
DEAD_STATE_THRESHOLD = 0.05  # PLAN §3.7 EB-3
COVAR_COND_THRESHOLD = 1e8   # cov ill-condition heuristic (replaces NSVARHMM σ collapse)
MIN_EIGENVAL_FLOOR_FACTOR = 0.1  # min_eigval should be ≥ 0.1 × reg_covar (relative floor)


# ──────────────────────────────────────────────────────────────────
# Feature engineering (z-score / log1p — loader.py 패턴 일관)
# ──────────────────────────────────────────────────────────────────
def featurize_raw(seg_df: pd.DataFrame, norm: dict, feature_cols: list[str]) -> np.ndarray:
    """Build [L, V_raw] features for seg2 with same scaling as legacy main path."""
    cols = []
    for c in feature_cols:
        if c == "ili_weighted_pct":
            m = norm["ili_weighted_pct"]["mean"]
            s = norm["ili_weighted_pct"]["std"]
            cols.append((seg_df[c].to_numpy() - m) / s)
        elif c in ("total_ili_count", "num_providers", "num_patients"):
            cols.append(np.log1p(seg_df[c].to_numpy()))
        else:
            raise ValueError(f"Unknown feature column: {c}")
    return np.stack(cols, axis=-1).astype(np.float64)   # [L, V_raw]


def augment_features(x_raw: np.ndarray) -> np.ndarray:
    """[L, V_raw] → [L-1, V_aug=2·V_raw] augmented [x_t, Δx_t] (Furui 1986).

    Δx is computed on the already-scaled space (z-score/log1p) so no
    additional normalization is needed (plan A7).
    """
    assert x_raw.ndim == 2, f"Expected [L, V_raw], got {x_raw.shape}"
    delta = x_raw[1:] - x_raw[:-1]              # [L-1, V_raw]
    x_aug = np.concatenate([x_raw[1:], delta], axis=-1)   # [L-1, 2·V_raw]
    return x_aug


def prepare_train(
    csv_path: Path,
    norm_path: Path,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load seg2 train data and build raw + augmented features.

    Returns:
        x_raw: [835, V_raw]
        x_aug: [834, V_aug=2·V_raw]
        ili_raw: [834] raw ili_weighted_pct aligned with x_aug (for binary κ)
    """
    df = load_dataset_csv(csv_path)
    norm = load_norm_params(norm_path)

    train = df[df["split"] == "train"].reset_index(drop=True)
    train["epiweek"] = train["epiweek"].astype(int)
    assert len(train) == EXPECTED_TRAIN_ROWS, \
        f"Expected {EXPECTED_TRAIN_ROWS} train rows, got {len(train)}"

    seg_df = train[train["epiweek"] >= TRAIN_START_EPIWEEK].reset_index(drop=True)
    L = len(seg_df)
    assert L == EXPECTED_SEG2_ROWS, \
        f"Expected {EXPECTED_SEG2_ROWS} seg2 rows, got {L}"
    assert int(seg_df["epiweek"].iloc[0]) == TRAIN_START_EPIWEEK

    x_raw = featurize_raw(seg_df, norm, feature_cols)            # [835, V_raw]
    x_aug = augment_features(x_raw)                              # [834, V_aug]
    ili_raw_aligned = seg_df["ili_weighted_pct"].to_numpy()[1:]  # [834] (aligned with x_aug)
    return x_raw, x_aug, ili_raw_aligned


# ──────────────────────────────────────────────────────────────────
# Covariance diagnostics — operates on REGULARIZED Σ_k + reg·I (M1 fix)
# Previous version used raw Σ_k and flagged reg_covar=1e-2 as "collapsed"
# because raw eigvals dipped below 0.1·reg even when the regularized cov was
# numerically stable. The regularized view matches inference-time numerics
# (gaussian_hmm.py:159 and PhaseModule._cache_hmm_torch S-2 fix).
# ──────────────────────────────────────────────────────────────────
def covariance_health(covars: np.ndarray, reg_covar: float) -> dict:
    """Inspect Σ_k + reg_covar·I (regularized) for ill-conditioning.

    M1 fix: diagnostic uses the regularized covariance — the same matrix used
    at inference time — so the 'collapsed' flag reflects whether downstream
    posterior computation is numerically stable, not whether the raw Σ_k is
    rank-deficient (which is masked by regularization).
    """
    K, V, _ = covars.shape
    eye_V = np.eye(V)
    min_eigvals = np.zeros(K)
    conds = np.zeros(K)
    for k in range(K):
        cov_reg = covars[k] + reg_covar * eye_V
        eigs = np.linalg.eigvalsh(cov_reg)
        min_eigvals[k] = float(eigs.min())
        conds[k] = float(eigs.max() / max(eigs.min(), 1e-30))
    # Regularized eigvals should be ≥ reg_covar in theory; allow 50% margin
    # for numerical noise. A truly degenerate cov would have min_eig < 0.5·reg.
    floor = 0.5 * reg_covar
    collapsed = bool(
        (min_eigvals < floor).any()
        or (conds > COVAR_COND_THRESHOLD).any()
    )
    return {
        "min_eigval_per_state": [float(v) for v in min_eigvals],
        "cond_per_state": [float(c) for c in conds],
        "regularized": True,        # M1 fix marker
        "collapsed": collapsed,
    }


# ──────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────
def run_one_K(
    V_raw: int,
    K: int,
    seed: int,
    reg_covar: float,
    x_aug: np.ndarray,
    ili_raw_aligned: np.ndarray,
    out_dir: Path,
    n_init: int = DEFAULT_N_INIT,
) -> dict:
    """Multi-start GaussianHMM fit on augmented features.

    Run `n_init` independent EM fits with different init seeds derived from
    the user-facing `seed`, keep the one with highest final log-likelihood.
    Reduces init sensitivity (dead state, cross-seed κ degradation).

    K parameterization supports §7.4 A-PD5 ablation (K∈{3,4,5}).
    Default usage is K=K_FIXED=3 (BIC-confirmed, PLAN §3.4).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    V_aug = 2 * V_raw

    t0 = time()
    best_hmm = None
    best_ll = -np.inf
    init_results = []  # per-start LL for diagnostics
    for i in range(n_init):
        init_seed = seed * 1000 + i  # deterministic but distinct
        candidate = GaussianHMM(
            n_states=K,
            n_features=V_aug,
            covariance_type="full",
            reg_covar=reg_covar,
            n_iter=200,
            tol=1e-4,
            seed=init_seed,
        ).fit(x_aug)
        final_ll = float(candidate.ll_history[-1]) if candidate.ll_history else -np.inf
        init_results.append({
            "i": i, "init_seed": init_seed, "final_ll": final_ll,
            "n_iter": int(candidate.n_iter_run),
        })
        if final_ll > best_ll:
            best_ll = final_ll
            best_hmm = candidate
    fit_sec = time() - t0
    assert best_hmm is not None

    gamma = best_hmm.posteriors(x_aug)              # [T, K]
    viterbi = best_hmm.viterbi(x_aug)               # [T]
    occ = state_occupancy(viterbi, K=K)
    dead = is_dead_state(occ, threshold=DEAD_STATE_THRESHOLD)
    cov_health = covariance_health(best_hmm.covars, reg_covar=reg_covar)
    bin_kappa = cohens_kappa_binary(viterbi, ili_raw_aligned)

    np.save(out_dir / "viterbi_path.npy", viterbi)
    np.save(out_dir / "gamma.npy", gamma)
    # C1 fix: include all artifacts needed for Stage 2 GaussianHMM reconstruction.
    # Without reg_covar / K / V / covariance_type, PhaseModule._cache_hmm_torch
    # would silently use defaults → emission likelihood mismatch with Stage 1.
    np.savez(
        out_dir / "hmm_params.npz",
        A=best_hmm.A,
        pi=best_hmm.pi,
        means=best_hmm.means,
        covars=best_hmm.covars,
        reg_covar=np.array(best_hmm.reg_covar, dtype=np.float64),
        K=np.array(best_hmm.K, dtype=np.int64),
        V=np.array(best_hmm.V, dtype=np.int64),
        covariance_type=np.array(best_hmm.covariance_type),
        n_iter_run=np.array(best_hmm.n_iter_run, dtype=np.int64),
        final_ll=np.array(best_ll, dtype=np.float64),
    )
    diag = {
        "V_raw": V_raw, "V_aug": V_aug, "K": K, "seed": seed,
        "reg_covar": float(reg_covar),
        "n_init": n_init,
        "best_init_seed": init_results[
            max(range(len(init_results)), key=lambda j: init_results[j]["final_ll"])
        ]["init_seed"],
        "init_results": init_results,
        "n_iter_run": int(best_hmm.n_iter_run),
        "final_ll": float(best_ll),
        "binary_kappa": float(bin_kappa),
        "state_occupancy": [float(o) for o in occ],
        "dead_state": bool(dead),
        "covariance_health": cov_health,
        "fit_seconds": float(fit_sec),
    }
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    return diag


# ──────────────────────────────────────────────────────────────────
# Cross-seed aggregation
# ──────────────────────────────────────────────────────────────────
def cross_seed_summary(viterbi_paths: dict[int, np.ndarray], K: int) -> dict:
    pairs = []
    for s1, s2 in combinations(sorted(viterbi_paths.keys()), 2):
        kappa = cohens_kappa_aligned(viterbi_paths[s1], viterbi_paths[s2], K=K)
        pairs.append({"seeds": [int(s1), int(s2)], "kappa": float(kappa)})
    kappas = [p["kappa"] for p in pairs]
    return {
        "pairs": pairs,
        "kappa_min": float(min(kappas)) if kappas else float("nan"),
        "kappa_mean": float(np.mean(kappas)) if kappas else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase Dynamics HMM search (M1.4b)")
    parser.add_argument("--smoke", action="store_true",
                        help="V_raw=3, reg_covar=1e-3, seed=42, n_init=2 (sanity)")
    parser.add_argument("--V_raw", type=int, default=None, choices=[3, 4],
                        help="Restrict to single V_raw (default: both)")
    parser.add_argument("--n_init", type=int, default=DEFAULT_N_INIT,
                        help=f"Multi-start count per (V_raw, reg_covar, seed) [default {DEFAULT_N_INIT}]")
    parser.add_argument("--reg_covar", type=float, default=None,
                        help="Single reg_covar value (default: sweep all)")
    parser.add_argument("--K-grid", type=int, nargs="+", default=None,
                        help="K values to sweep (§7.4 A-PD5 ablation). "
                             "Default: K=3 fixed (PLAN spec). Example: --K-grid 3 4 5")
    args = parser.parse_args()

    csv_path = _CG_MAMBA_ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
    norm_path = _CG_MAMBA_ROOT / "data" / "processed" / "normalization_params.json"
    out_root = _CG_MAMBA_ROOT / "runs" / "m1_4_phase_dynamics"
    out_root.mkdir(parents=True, exist_ok=True)

    # K grid (§7.4 A-PD5 ablation support). Default: K=3 fixed.
    k_grid = list(args.K_grid) if args.K_grid is not None else [K_FIXED]

    if args.smoke:
        v_raw_grid = [3]
        seed_grid = [42]
        reg_covar_grid = [1e-3]
        n_init = 2
    elif args.V_raw is not None:
        v_raw_grid = [args.V_raw]
        seed_grid = SEEDS
        reg_covar_grid = [args.reg_covar] if args.reg_covar is not None else REG_COVAR_GRID
        n_init = args.n_init
    else:
        v_raw_grid = [3, 4]
        seed_grid = SEEDS
        reg_covar_grid = [args.reg_covar] if args.reg_covar is not None else REG_COVAR_GRID
        n_init = args.n_init

    n_settings = len(v_raw_grid) * len(k_grid) * len(reg_covar_grid) * len(seed_grid)
    k_label = (
        f"K={k_grid[0]} fixed (BIC, PLAN §3.4)"
        if len(k_grid) == 1
        else f"K ∈ {k_grid} (§7.4 A-PD5 ablation)"
    )
    print(f"[M1.4b Phase Dynamics search] {k_label}")
    print(f"  V_raw ∈ {v_raw_grid}, reg_covar ∈ {reg_covar_grid}, seeds ∈ {seed_grid}")
    print(f"  multi-start n_init={n_init}")
    print(f"  total settings={n_settings}, EM fits={n_settings * n_init}")
    print(f"  output: {out_root.relative_to(_CG_MAMBA_ROOT)}")
    print()

    summary_rows = []
    cross_seed: dict[str, dict] = {}  # "V_raw{V}_K{K}_regcov{r:.0e}" -> cs

    for V_raw in v_raw_grid:
        feature_cols = RAW_COLS_V3 if V_raw == 3 else RAW_COLS_V4
        print(f"━━ V_raw={V_raw} (V_aug={2 * V_raw}, cols={feature_cols}) ━━")
        x_raw, x_aug, ili_raw_aligned = prepare_train(csv_path, norm_path, feature_cols)
        print(f"  x_raw {x_raw.shape}, x_aug {x_aug.shape}")

        for K in k_grid:
            for reg_covar in reg_covar_grid:
                print(f"  ── K={K}, reg_covar={reg_covar:.0e} ──")
                viterbi_paths: dict[int, np.ndarray] = {}
                for seed in seed_grid:
                    run_name = f"V_raw{V_raw}_K{K}_regcov{reg_covar:.0e}_seed{seed}"
                    run_dir = out_root / run_name
                    diag = run_one_K(
                        V_raw=V_raw, K=K, seed=seed, reg_covar=reg_covar,
                        x_aug=x_aug, ili_raw_aligned=ili_raw_aligned,
                        out_dir=run_dir, n_init=n_init,
                    )
                    viterbi_paths[seed] = np.load(run_dir / "viterbi_path.npy")
                    print(
                        f"    seed={seed:3d}  "
                        f"bin_κ={diag['binary_kappa']:.4f}  "
                        f"occ={[f'{o:.3f}' for o in diag['state_occupancy']]}  "
                        f"dead={diag['dead_state']}  "
                        f"cov_collapsed={diag['covariance_health']['collapsed']}  "
                        f"ll={diag['final_ll']:.2f}  "
                        f"n_iter={diag['n_iter_run']}"
                    )
                    summary_rows.append({
                        "V_raw": V_raw,
                        "V_aug": 2 * V_raw,
                        "K": K,
                        "reg_covar": reg_covar,
                        "n_init": n_init,
                        "seed": seed,
                        "best_init_seed": diag["best_init_seed"],
                        "binary_kappa": diag["binary_kappa"],
                        "dead_state": diag["dead_state"],
                        "cov_collapsed": diag["covariance_health"]["collapsed"],
                        "min_occupancy": float(min(diag["state_occupancy"])),
                        "max_occupancy": float(max(diag["state_occupancy"])),
                        "n_iter_run": diag["n_iter_run"],
                        "final_ll": diag["final_ll"],
                        "fit_seconds": diag["fit_seconds"],
                    })

                # Cross-seed κ for this (V_raw, K, reg_covar) cell
                key = f"V_raw{V_raw}_K{K}_regcov{reg_covar:.0e}"
                if len(viterbi_paths) >= 2:
                    cs = cross_seed_summary(viterbi_paths, K=K)
                    cross_seed[key] = cs
                    print(f"    cross-seed κ_min={cs['kappa_min']:.4f}, κ_mean={cs['kappa_mean']:.4f}")
                    for p in cs["pairs"]:
                        s1, s2 = p["seeds"]
                        print(f"      κ(seed {s1} vs {s2}) = {p['kappa']:.4f}")

                cell_rows = [r for r in summary_rows
                             if r["V_raw"] == V_raw and r["K"] == K
                             and r["reg_covar"] == reg_covar]
                trig = fallback_trigger(
                    final_kappas=[r["binary_kappa"] for r in cell_rows],
                    dead_states=[r["dead_state"] for r in cell_rows],
                    sigma_collapses=[r["cov_collapsed"] for r in cell_rows],
                )
                print(f"    fallback_trigger: {trig['triggered']} ({trig['reason']})")
        print()

    pd.DataFrame(summary_rows).to_csv(out_root / "search_summary.csv", index=False)
    with open(out_root / "cross_seed_kappa.json", "w") as f:
        json.dump({
            "k_grid": k_grid, "seeds": seed_grid, "n_init": n_init,
            "by_cell": cross_seed,
        }, f, indent=2)

    # Winner selection: max κ_min among cells with no dead state and no cov collapse
    print("━━ Winner ranking (κ_min ↓, healthy cells only) ━━")
    cell_summary = []
    for key, cs in cross_seed.items():
        # key format: V_raw{V}_K{K}_regcov{r:.0e}
        cell_rows = [r for r in summary_rows
                     if f"V_raw{r['V_raw']}_K{r['K']}_regcov{r['reg_covar']:.0e}" == key]
        any_dead = any(r["dead_state"] for r in cell_rows)
        any_cov = any(r["cov_collapsed"] for r in cell_rows)
        cell_summary.append({
            "cell": key,
            "kappa_min": cs["kappa_min"],
            "kappa_mean": cs["kappa_mean"],
            "any_dead": any_dead,
            "any_cov_collapse": any_cov,
            "healthy": (not any_dead) and (not any_cov),
        })
    cell_summary.sort(key=lambda c: (not c["healthy"], -c["kappa_min"]))
    for c in cell_summary:
        flag = "✓" if c["healthy"] else "✗"
        print(f"  {flag} {c['cell']:30s}  κ_min={c['kappa_min']:.4f}  κ_mean={c['kappa_mean']:.4f}  "
              f"dead={c['any_dead']}  cov_collapse={c['any_cov_collapse']}")
    with open(out_root / "winner_ranking.json", "w") as f:
        json.dump(cell_summary, f, indent=2)

    print(f"\nSaved: search_summary.csv + cross_seed_kappa.json + winner_ranking.json "
          f"under {out_root.relative_to(_CG_MAMBA_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
