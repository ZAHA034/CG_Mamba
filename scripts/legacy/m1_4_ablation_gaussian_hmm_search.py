"""DEPRECATED in v2.0.9 — retained for paper §7.4 ablation B reproduction (v2.0.8c ablation B = i.i.d. raw GaussianHMM). Replaced by m1_4_phase_dynamics_search.py + m1_4_phase_dynamics_main.py.

M1.4 §7.4 Ablation — GaussianHMM K-selection search (simpler HMM baseline).

PLAN v2.0.8c §3.7 + §7.4 + Appendix D ablation

**Status**: §7.4 ablation candidate, NOT the M1.4 main path.

Main path = NeuralSwitchingVARHMM (GRU + VAR emission)
            → scripts/run_hmm_stage1.py
Ablation   = GaussianHMM (i.i.d. emission)
            → THIS SCRIPT

Ablation purpose (PLAN §7.4 narrative outcomes):
  - Similar downstream MAE → "phase separation is the contribution,
                              HMM complexity is secondary" (CG-Mamba generality)
  - GaussianHMM significantly worse → "temporal-aware phase detection
                                       justifies NeuralSwitchingVARHMM"

Data alignment (v2.0.8c, ED-1):
  - **seg2-only train**: 2002-W40 ~ 2018-W39 (835 rows, 16 full seasons)
  - Identical to HMM main path + LSTM/CG-Mamba sliding-window models
  - seg1 (200140 ~ 200220) excluded — uniform across all forecasting models

Grid: K ∈ {3, 4, 5} × seed ∈ {42, 123, 456} = 9 runs

Selection criteria (per K, PLAN §3.7):
    1. BIC ↓ (median across 3 seeds)
    2. Dead state < 5% mean posterior mass (any seed → reject K)
    3. Cohen's κ ≥ 0.50 (pairwise aligned Viterbi agreement across 3 seeds)

Fallback (PLAN v2.0.8b EB-2):
    If all K fail with V=4 → re-run with V=3 (drop num_patients,
    r=0.952 multicollinearity with num_providers).

Usage:
    # With real ILI data (seg2-only, v2.0.8c):
    python -m scripts.m1_4_ablation_gaussian_hmm_search

    # Synthetic-only smoke test:
    python -m scripts.m1_4_ablation_gaussian_hmm_search --synthetic

    # V=3 fallback only:
    python -m scripts.m1_4_ablation_gaussian_hmm_search --V 3

    # Single-K sanity (quick):
    python -m scripts.m1_4_ablation_gaussian_hmm_search --sanity

Output:
    runs/m1_4_ablation_gaussian_hmm/k_search_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add CG_Mamba root to sys.path so absolute imports work when launched as
# `python scripts/m1_4_ablation_gaussian_hmm_search.py` (no -m). When launched
# via `python -m scripts...` this is unnecessary but harmless.
_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[2]  # legacy/ moved: scripts/legacy/m1_4_ablation_gaussian_hmm_search.py
if str(_CG_MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CG_MAMBA_ROOT))

from src.models.gaussian_hmm import (  # noqa: E402
    GaussianHMM,
    generate_synthetic_hmm_data,
)
from src.models.legacy.hmm_stage1 import TRAIN_START_EPIWEEK  # noqa: E402 — v2.0.9: moved to legacy
from src.utils.config import CGMambaConfig  # noqa: E402
from src.utils.metrics import cohens_kappa_aligned  # noqa: E402


REPO_ROOT = _CG_MAMBA_ROOT

# PLAN §3.7 grid + thresholds
K_GRID = [3, 4, 5]
SEED_GRID = [42, 123, 456]
DEAD_STATE_THRESHOLD = 0.05
KAPPA_THRESHOLD = 0.50
MAX_EM_ITER = 100


# ─────────────────────────────────────────────────────────────────
# Data loading — v2.0.8c alignment (seg2-only train, full ts for eval)
# ─────────────────────────────────────────────────────────────────

def _featurize_v4(df_sub, norm: dict) -> np.ndarray:
    """Apply same preprocessing as src/data/loader.py for V=4 features.

    Z-score ili_weighted_pct (train-fit μ, σ); log1p the 3 count features.
    Output: [T, 4] np.float64.
    """
    ili_w = ((df_sub["ili_weighted_pct"].values
              - norm["ili_weighted_pct"]["mean"])
             / norm["ili_weighted_pct"]["std"])
    ili_count = np.log1p(df_sub["total_ili_count"].values)
    n_prov = np.log1p(df_sub["num_providers"].values)
    n_pat = np.log1p(df_sub["num_patients"].values)
    return np.stack([ili_w, ili_count, n_prov, n_pat], axis=-1).astype(np.float64)


def load_ili_train_seg2(cfg: CGMambaConfig) -> np.ndarray | None:
    """Load HMM training data: seg2-only (v2.0.8c).

    Returns [835, 4] training feature array (epiweek >= 200240), or None if
    data files unavailable.
    """
    import pandas as pd

    if not cfg.data_csv.exists() or not cfg.norm_json.exists():
        return None

    df = pd.read_csv(cfg.data_csv).sort_values("epiweek").reset_index(drop=True)
    with open(cfg.norm_json) as f:
        norm = json.load(f)["params"]

    train = df[df["split"] == "train"].reset_index(drop=True)
    seg2 = train[train["epiweek"].astype(int) >= TRAIN_START_EPIWEEK].reset_index(drop=True)
    assert len(seg2) == 835, (
        f"Expected 835 seg2 rows (post-CDC-gap), got {len(seg2)}. "
        f"v2.0.8c ED-1 spec violated."
    )
    assert int(seg2["epiweek"].iloc[0]) == TRAIN_START_EPIWEEK, (
        f"seg2 must start at {TRAIN_START_EPIWEEK}, got {seg2['epiweek'].iloc[0]}"
    )

    return _featurize_v4(seg2, norm)


def load_ili_full_for_eval(cfg: CGMambaConfig) -> np.ndarray | None:
    """Load full time series (1,229 rows) for evaluation κ comparison.

    Returns [T_all, 4] array spanning train + val + covid_excluded + test.
    """
    import pandas as pd

    if not cfg.data_csv.exists() or not cfg.norm_json.exists():
        return None

    df = pd.read_csv(cfg.data_csv).sort_values("epiweek").reset_index(drop=True)
    with open(cfg.norm_json) as f:
        norm = json.load(f)["params"]

    return _featurize_v4(df, norm)


# ─────────────────────────────────────────────────────────────────
# K-search logic
# ─────────────────────────────────────────────────────────────────

def run_k_search(
    x_train: np.ndarray,
    x_eval: np.ndarray | None = None,
    V: int = 4,
) -> dict:
    """Run K ∈ {3,4,5} × seed ∈ {42,123,456} grid (9 fits).

    Args:
        x_train: [T_train, V] training features (seg2-only for v2.0.8c)
        x_eval:  [T_eval, V] evaluation features for cross-seed Viterbi κ
                  (None → use x_train)
        V: number of features to use (4 default, 3 fallback)

    Returns:
        results dict with per-K metrics and selection decision.
    """
    if x_eval is None:
        x_eval = x_train

    # Trim features if V < 4 (V=3 fallback)
    if V < x_train.shape[1]:
        x_train = x_train[:, :V]
        x_eval = x_eval[:, :V]

    results: dict = {
        "V": V,
        "K_grid": K_GRID,
        "seed_grid": SEED_GRID,
        "n_train": int(x_train.shape[0]),
        "n_eval": int(x_eval.shape[0]),
        "per_K": {},
    }

    for K in K_GRID:
        k_results: dict = {
            "K": K,
            "per_seed": {},
            "bics": [],
            "dead_states": [],
            "kappas_pairwise": [],
        }
        viterbi_seqs = {}

        for seed in SEED_GRID:
            print(f"  Fitting K={K}, seed={seed} ...", end=" ", flush=True)

            hmm = GaussianHMM(
                n_states=K,
                n_features=V,
                covariance_type="full",
                reg_covar=1e-3,
                n_iter=MAX_EM_ITER,
                tol=1e-4,
                seed=seed,
            )
            hmm.fit(x_train)

            bic_val = hmm.bic(x_train)
            dead = hmm.dead_states(x_train, threshold=DEAD_STATE_THRESHOLD)
            ll = hmm.log_likelihood(x_train)
            viterbi_states = hmm.viterbi(x_eval)
            gamma = hmm.posteriors(x_train)
            mean_post = gamma.mean(axis=0).tolist()

            seed_result = {
                "seed": seed,
                "bic": float(bic_val),
                "log_likelihood": float(ll),
                "n_iter_run": int(hmm.n_iter_run),
                "dead_states": dead,
                "mean_posterior_per_state": mean_post,
            }
            k_results["per_seed"][str(seed)] = seed_result
            k_results["bics"].append(float(bic_val))
            k_results["dead_states"].append(dead)
            viterbi_seqs[seed] = viterbi_states

            converged = "converged" if hmm.n_iter_run < MAX_EM_ITER else "MAX ITER"
            dead_str = f"dead={dead}" if dead else "no dead"
            print(f"BIC={bic_val:.1f}, {converged} ({hmm.n_iter_run} iter), {dead_str}")

        # Pairwise aligned κ (cross-seed stability)
        seeds = list(SEED_GRID)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                s1 = viterbi_seqs[seeds[i]]
                s2 = viterbi_seqs[seeds[j]]
                kappa = cohens_kappa_aligned(s1, s2, K)
                k_results["kappas_pairwise"].append({
                    "seed_pair": [seeds[i], seeds[j]],
                    "kappa": float(kappa),
                })
                print(f"    κ(seed {seeds[i]} vs {seeds[j]}): {kappa:.3f}")

        # Per-K summary
        k_results["bic_median"] = float(np.median(k_results["bics"]))
        k_results["has_dead_state"] = any(len(d) > 0 for d in k_results["dead_states"])
        kappas = [kp["kappa"] for kp in k_results["kappas_pairwise"]]
        k_results["kappa_min"] = float(min(kappas)) if kappas else 0.0
        k_results["kappa_mean"] = float(np.mean(kappas)) if kappas else 0.0

        results["per_K"][str(K)] = k_results

    results["selection"] = _select_k(results)
    return results


def _select_k(results: dict) -> dict:
    """K selection criteria (PLAN §3.7):
        1. Reject K with dead state (any seed) — over-parameterization
        2. Reject K with min pairwise κ < 0.50 — unstable across seeds
        3. Among remaining, pick lowest median BIC — Schwarz penalty
    """
    candidates = []
    for K_str, kr in results["per_K"].items():
        K = int(K_str)
        rejected = False
        reason = None

        if kr["has_dead_state"]:
            rejected = True
            reason = "dead state detected"
        elif kr["kappa_min"] < KAPPA_THRESHOLD:
            rejected = True
            reason = f"κ_min={kr['kappa_min']:.3f} < {KAPPA_THRESHOLD}"

        candidates.append({
            "K": K,
            "bic_median": kr["bic_median"],
            "kappa_min": kr["kappa_min"],
            "has_dead_state": kr["has_dead_state"],
            "rejected": rejected,
            "reason": reason,
        })

    valid = [c for c in candidates if not c["rejected"]]
    if valid:
        best = min(valid, key=lambda c: c["bic_median"])
        return {
            "selected_K": best["K"],
            "decision": "PASS",
            "candidates": candidates,
            "note": f"K={best['K']} selected (lowest BIC among valid candidates)",
        }
    else:
        return {
            "selected_K": None,
            "decision": "FAIL_ALL_REJECTED",
            "candidates": candidates,
            "note": (
                f"All K rejected with V={results['V']}. "
                f"{'Retry with V=3 (drop num_patients).' if results['V'] == 4 else 'Manual investigation needed.'}"
            ),
        }


# ─────────────────────────────────────────────────────────────────
# Sanity (single K, single seed) — Step E-3 토대
# ─────────────────────────────────────────────────────────────────

def run_sanity(x_train: np.ndarray, x_eval: np.ndarray | None = None) -> dict:
    """Single (K=3, seed=42) GaussianHMM fit on real ILI data.

    Used as Step E sanity gate before running full 9-run grid in production.
    """
    if x_eval is None:
        x_eval = x_train
    print(f"  GaussianHMM K=3 seed=42 V={x_train.shape[1]} on {x_train.shape[0]} rows")
    hmm = GaussianHMM(
        n_states=3, n_features=x_train.shape[1], covariance_type="full",
        reg_covar=1e-3, n_iter=MAX_EM_ITER, tol=1e-4, seed=42,
    ).fit(x_train)
    bic = hmm.bic(x_train)
    dead = hmm.dead_states(x_train)
    gamma = hmm.posteriors(x_train)
    viterbi = hmm.viterbi(x_eval)
    occ = gamma.mean(axis=0)
    print(f"  → BIC={bic:.1f}, iters={hmm.n_iter_run}, dead={dead}, "
          f"occ=[{', '.join(f'{o:.3f}' for o in occ)}]")
    return {
        "K": 3, "seed": 42, "V": int(x_train.shape[1]),
        "n_train": int(x_train.shape[0]),
        "bic": float(bic),
        "n_iter_run": int(hmm.n_iter_run),
        "dead_states": dead,
        "mean_posterior": occ.tolist(),
        "viterbi_state_counts": np.bincount(viterbi, minlength=3).tolist(),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="M1.4 §7.4 ablation: GaussianHMM K-search")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of real ILI")
    parser.add_argument("--sanity", action="store_true",
                        help="Single K=3 seed=42 sanity only (skip grid)")
    parser.add_argument("--V", type=int, choices=[3, 4], default=4,
                        help="Number of HMM input features (V=4 default, V=3 fallback)")
    args = parser.parse_args()

    cfg = CGMambaConfig()
    out_dir = REPO_ROOT / "runs" / "m1_4_ablation_gaussian_hmm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("M1.4 §7.4 Ablation — GaussianHMM K-selection (PLAN v2.0.8c)")
    print("=" * 78)
    print(f"  V = {args.V}")
    print(f"  TRAIN_START_EPIWEEK = {TRAIN_START_EPIWEEK}  (v2.0.8c, seg2-only)")

    # ── Data loading ──
    if args.synthetic:
        print("\n[SYNTHETIC] 3-state generator (T=500, V=4, seed=0)")
        x_train, _ = generate_synthetic_hmm_data(K=3, V=4, T=500, seed=0)
        x_eval = x_train
    else:
        print("\nLoading real ILI data (v2.0.8c seg2-only train)...")
        x_train = load_ili_train_seg2(cfg)
        if x_train is None:
            print(f"  ⚠️  Data not found at {cfg.data_csv}. Falling back to synthetic.")
            x_train, _ = generate_synthetic_hmm_data(K=3, V=4, T=500, seed=0)
            x_eval = x_train
        else:
            print(f"  Loaded x_train: {x_train.shape}  (seg2 [835, 4])")
            x_eval = load_ili_full_for_eval(cfg)
            print(f"  Loaded x_eval (full series): {x_eval.shape}")

    # ── Sanity mode ──
    if args.sanity:
        print(f"\n[SANITY] Single K=3 seed=42 fit")
        sanity_result = run_sanity(x_train, x_eval)
        out_path = out_dir / "sanity_result.json"
        with open(out_path, "w") as f:
            json.dump(sanity_result, f, indent=2)
        print(f"\nSaved: {out_path.relative_to(REPO_ROOT)}")
        return 0

    # ── Phase 1: V=4 grid ──
    print(f"\n{'━' * 78}")
    print(f"Phase 1: V={args.V} grid (K ∈ {K_GRID})")
    print(f"{'━' * 78}")
    results = run_k_search(x_train, x_eval, V=args.V)
    sel = results["selection"]
    print(f"\n{'=' * 78}")
    print(f"V={args.V} Result: {sel['decision']}")
    if sel["selected_K"] is not None:
        print(f"  ✓ Selected K = {sel['selected_K']}")
        print(f"     {sel['note']}")
    else:
        print(f"  ✗ {sel['note']}")

    # ── Phase 2: V=3 fallback (if V=4 failed) ──
    results_v3 = None
    if args.V == 4 and sel["selected_K"] is None and x_train.shape[1] >= 4:
        print(f"\n{'━' * 78}")
        print("Phase 2: V=3 fallback (drop num_patients, v2.0.8b EB-2)")
        print(f"{'━' * 78}")
        results_v3 = run_k_search(x_train, x_eval, V=3)
        sel_v3 = results_v3["selection"]
        print(f"\n{'=' * 78}")
        print(f"V=3 Result: {sel_v3['decision']}")
        if sel_v3["selected_K"] is not None:
            print(f"  ✓ Selected K = {sel_v3['selected_K']} (V=3 fallback)")
        else:
            print(f"  ✗ Both V=4 and V=3 failed. Manual investigation needed.")

    # ── Save ──
    all_results = {
        "v4": results if args.V == 4 else None,
        "v3": results_v3 if results_v3 is not None else (results if args.V == 3 else None),
        "final_decision": {
            "V": (results["selection"]["selected_K"] is not None and args.V) or
                 (results_v3 is not None and results_v3["selection"]["selected_K"] is not None and 3) or
                 None,
            "K": (results["selection"]["selected_K"] or
                  (results_v3["selection"]["selected_K"] if results_v3 else None)),
        },
        "data_alignment": "v2.0.8c seg2-only (TRAIN_START_EPIWEEK=200240, 835 rows)",
    }
    out_path = out_dir / "k_search_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path.relative_to(REPO_ROOT)}")

    final_K = all_results["final_decision"]["K"]
    if final_K is not None:
        print(f"\n✓ M1.4 §7.4 ablation complete: K={final_K}")
        return 0
    else:
        print("\n✗ M1.4 §7.4 ablation failed: no valid K found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
