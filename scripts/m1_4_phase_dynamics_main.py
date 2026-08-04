"""M1.4c — Phase Dynamics GaussianHMM main run (Stage 1 offline EM).

PLAN v2.0.9 PATCH 4 (D.5.1 Stage 1) + PATCH 14 (script rename)

Purpose:
  Final Stage-1 HMM artifacts for M1.4b winner config:
    V_raw=3 (V_aug=6) × K=3 × reg_covar=5e-3 × n_init=5
  Runs across 3 seeds (42, 123, 456) and persists everything PhaseModule
  needs for Stage 2 caching: A, π, means, covars, reg_covar, K, V,
  covariance_type, n_iter_run, final_ll, viterbi_path, gamma.

This is the Stage 1 producer (offline EM, NumPy). Stage 2 consumer is
PhaseModule._cache_hmm_torch (R-1/T-1).

Defaults come from CGMambaConfig (V_hmm_raw, K_phase, hmm_reg_covar,
hmm_n_init, hmm_seeds). Override individually on the CLI for ablations.

Usage:
  python scripts/m1_4_phase_dynamics_main.py            # cfg defaults (3 seeds)
  python scripts/m1_4_phase_dynamics_main.py --smoke    # seed=42 only, n_init=2
  python scripts/m1_4_phase_dynamics_main.py --V_raw 4  # ablation override

Output:
  runs/m1_4_phase_dynamics_main/V_raw{V}_regcov{r}_K{K}_seed{s}/
    ├── hmm_params.npz       (A, pi, means, covars, reg_covar, K, V, ...)
    ├── viterbi_path.npy
    ├── gamma.npy
    └── diagnostics.json
  runs/m1_4_phase_dynamics_main/main_summary.csv
  runs/m1_4_phase_dynamics_main/cross_seed_kappa.json
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
from src.utils.config import CGMambaConfig  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    cohens_kappa_aligned,
    cohens_kappa_binary,
    fallback_trigger,
    is_dead_state,
    state_occupancy,
)


# Shared helpers — reuse from search script to avoid divergence
from scripts.m1_4_phase_dynamics_search import (  # noqa: E402
    RAW_COLS_V3,
    RAW_COLS_V4,
    TRAIN_START_EPIWEEK,
    augment_features,
    covariance_health,
    featurize_raw,
    prepare_train,
)


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
DEAD_STATE_THRESHOLD = 0.05    # PLAN §3.7 EB-3


# ──────────────────────────────────────────────────────────────────
# Stage 1 fitter (multi-start, best-LL selection)
# ──────────────────────────────────────────────────────────────────
def fit_stage1(
    x_aug: np.ndarray,
    K: int,
    V_aug: int,
    reg_covar: float,
    seed: int,
    n_init: int,
) -> tuple[GaussianHMM, list[dict]]:
    """Multi-start EM fit. Returns best-LL HMM + per-start diagnostics."""
    best_hmm = None
    best_ll = -np.inf
    init_results = []
    for i in range(n_init):
        init_seed = seed * 1000 + i
        cand = GaussianHMM(
            n_states=K,
            n_features=V_aug,
            covariance_type="full",
            reg_covar=reg_covar,
            n_iter=200,
            tol=1e-4,
            seed=init_seed,
        ).fit(x_aug)
        ll = float(cand.ll_history[-1]) if cand.ll_history else -np.inf
        init_results.append({
            "i": i, "init_seed": init_seed, "final_ll": ll,
            "n_iter": int(cand.n_iter_run),
        })
        if ll > best_ll:
            best_ll = ll
            best_hmm = cand
    assert best_hmm is not None
    return best_hmm, init_results


def persist_run(
    out_dir: Path,
    hmm: GaussianHMM,
    x_aug: np.ndarray,
    ili_raw_aligned: np.ndarray,
    V_raw: int,
    K: int,
    seed: int,
    init_results: list[dict],
    final_ll: float,
    fit_sec: float,
) -> dict:
    """Compute Stage-1 diagnostics and write all artifacts (C1 metadata included)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    V_aug = 2 * V_raw

    gamma = hmm.posteriors(x_aug)
    viterbi = hmm.viterbi(x_aug)
    occ = state_occupancy(viterbi, K=K)
    dead = is_dead_state(occ, threshold=DEAD_STATE_THRESHOLD)
    cov_health = covariance_health(hmm.covars, reg_covar=hmm.reg_covar)
    bin_kappa = cohens_kappa_binary(viterbi, ili_raw_aligned)

    np.save(out_dir / "viterbi_path.npy", viterbi)
    np.save(out_dir / "gamma.npy", gamma)
    # C1: complete metadata for Stage 2 GaussianHMM reconstruction
    np.savez(
        out_dir / "hmm_params.npz",
        A=hmm.A, pi=hmm.pi, means=hmm.means, covars=hmm.covars,
        reg_covar=np.array(hmm.reg_covar, dtype=np.float64),
        K=np.array(hmm.K, dtype=np.int64),
        V=np.array(hmm.V, dtype=np.int64),
        covariance_type=np.array(hmm.covariance_type),
        n_iter_run=np.array(hmm.n_iter_run, dtype=np.int64),
        final_ll=np.array(final_ll, dtype=np.float64),
    )
    diag = {
        "V_raw": V_raw, "V_aug": V_aug, "K": K, "seed": seed,
        "reg_covar": float(hmm.reg_covar),
        "n_init": len(init_results),
        "init_results": init_results,
        "best_init_seed": max(init_results, key=lambda r: r["final_ll"])["init_seed"],
        "n_iter_run": int(hmm.n_iter_run),
        "final_ll": float(final_ll),
        "binary_kappa": float(bin_kappa),
        "state_occupancy": [float(o) for o in occ],
        "dead_state": bool(dead),
        "covariance_health": cov_health,
        "fit_seconds": float(fit_sec),
    }
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    return diag


# M1.6 (v2.0.9): load_fitted_hmm은 src/utils/checkpoints.py로 이동.
# 본 모듈에서는 backward compat 위해 re-export — 기존 호출자
# (test_phase_dynamics_main 등)는 변경 없이 `from scripts.m1_4_phase_dynamics_main
# import load_fitted_hmm` 그대로 사용 가능.
from src.utils.checkpoints import load_fitted_hmm  # noqa: E402, F401


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase Dynamics HMM main Stage-1 run (M1.4c)")
    parser.add_argument("--smoke", action="store_true",
                        help="seed=42 only, n_init=2 (sanity)")
    parser.add_argument("--V_raw", type=int, default=None, choices=[3, 4],
                        help="Override cfg.V_hmm_raw")
    parser.add_argument("--K", type=int, default=None,
                        help="Override cfg.K_phase")
    parser.add_argument("--reg_covar", type=float, default=None,
                        help="Override cfg.hmm_reg_covar")
    parser.add_argument("--n_init", type=int, default=None,
                        help="Override cfg.hmm_n_init")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override cfg.hmm_seeds (e.g., --seeds 789 1024 for M2.1 extension)")
    args = parser.parse_args()

    cfg = CGMambaConfig()
    V_raw = args.V_raw if args.V_raw is not None else cfg.V_hmm_raw
    K = args.K if args.K is not None else cfg.K_phase
    reg_covar = args.reg_covar if args.reg_covar is not None else cfg.hmm_reg_covar
    n_init = args.n_init if args.n_init is not None else cfg.hmm_n_init
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.smoke:
        seeds = [42]
    else:
        seeds = list(cfg.hmm_seeds)
    if args.smoke:
        n_init = min(n_init, 2)

    csv_path = _CG_MAMBA_ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
    norm_path = _CG_MAMBA_ROOT / "data" / "processed" / "normalization_params.json"
    out_root = _CG_MAMBA_ROOT / "runs" / "m1_4_phase_dynamics_main"
    out_root.mkdir(parents=True, exist_ok=True)

    feature_cols = RAW_COLS_V3 if V_raw == 3 else RAW_COLS_V4
    V_aug = 2 * V_raw

    print(f"[M1.4c Phase Dynamics MAIN] cfg-driven Stage 1 EM")
    print(f"  V_raw={V_raw} (V_aug={V_aug}) K={K} reg_covar={reg_covar:.0e} n_init={n_init}")
    print(f"  seeds={seeds}, cols={feature_cols}")
    print(f"  output: {out_root.relative_to(_CG_MAMBA_ROOT)}")
    print()

    x_raw, x_aug, ili_raw_aligned = prepare_train(csv_path, norm_path, feature_cols)
    print(f"  data: x_raw {x_raw.shape}, x_aug {x_aug.shape}")
    print()

    summary_rows = []
    viterbi_paths: dict[int, np.ndarray] = {}
    for seed in seeds:
        run_name = f"V_raw{V_raw}_regcov{reg_covar:.0e}_K{K}_seed{seed}"
        run_dir = out_root / run_name
        t0 = time()
        hmm, init_results = fit_stage1(x_aug, K, V_aug, reg_covar, seed, n_init)
        fit_sec = time() - t0
        final_ll = float(hmm.ll_history[-1])
        diag = persist_run(
            run_dir, hmm, x_aug, ili_raw_aligned,
            V_raw, K, seed, init_results, final_ll, fit_sec,
        )
        viterbi_paths[seed] = np.load(run_dir / "viterbi_path.npy")
        print(
            f"  seed={seed:3d}  "
            f"bin_κ={diag['binary_kappa']:.4f}  "
            f"occ={[f'{o:.3f}' for o in diag['state_occupancy']]}  "
            f"dead={diag['dead_state']}  "
            f"cov_collapsed={diag['covariance_health']['collapsed']}  "
            f"ll={diag['final_ll']:.2f}  "
            f"n_iter={diag['n_iter_run']}"
        )
        summary_rows.append({
            "V_raw": V_raw, "V_aug": V_aug, "K": K, "seed": seed,
            "reg_covar": reg_covar, "n_init": n_init,
            "binary_kappa": diag["binary_kappa"],
            "dead_state": diag["dead_state"],
            "cov_collapsed": diag["covariance_health"]["collapsed"],
            "min_occupancy": float(min(diag["state_occupancy"])),
            "max_occupancy": float(max(diag["state_occupancy"])),
            "n_iter_run": diag["n_iter_run"],
            "final_ll": diag["final_ll"],
            "fit_seconds": diag["fit_seconds"],
        })

    # Cross-seed κ
    print()
    if len(viterbi_paths) >= 2:
        pairs = []
        for s1, s2 in combinations(sorted(viterbi_paths.keys()), 2):
            kappa = cohens_kappa_aligned(viterbi_paths[s1], viterbi_paths[s2], K=K)
            pairs.append({"seeds": [int(s1), int(s2)], "kappa": float(kappa)})
            print(f"  cross-seed κ(seed {s1} vs {s2}) = {kappa:.4f}")
        kappas = [p["kappa"] for p in pairs]
        kmin, kmean = float(min(kappas)), float(np.mean(kappas))
        print(f"  cross-seed κ_min={kmin:.4f}, κ_mean={kmean:.4f}")
    else:
        pairs, kmin, kmean = [], float("nan"), float("nan")

    trig = fallback_trigger(
        final_kappas=[r["binary_kappa"] for r in summary_rows],
        dead_states=[r["dead_state"] for r in summary_rows],
        sigma_collapses=[r["cov_collapsed"] for r in summary_rows],
    )
    print(f"  fallback_trigger: {trig['triggered']} ({trig['reason']})")

    # Persist
    pd.DataFrame(summary_rows).to_csv(out_root / "main_summary.csv", index=False)
    with open(out_root / "cross_seed_kappa.json", "w") as f:
        json.dump({
            "V_raw": V_raw, "K": K, "reg_covar": reg_covar, "n_init": n_init,
            "seeds": seeds, "pairs": pairs, "kappa_min": kmin, "kappa_mean": kmean,
        }, f, indent=2)

    print(f"\nSaved: main_summary.csv + cross_seed_kappa.json under "
          f"{out_root.relative_to(_CG_MAMBA_ROOT)}")
    print(f"Use load_fitted_hmm(run_dir) to load any seed's HMM into PhaseModule "
          f"(Stage 2 entry sequence: ._cache_hmm_torch).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
