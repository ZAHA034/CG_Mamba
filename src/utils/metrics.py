"""Diagnostic metrics for HMM Stage 1 (v2.0.8b).

Phase 0 표준에 정합:
  - epi_truth = (ili_weighted_pct > 2.0) — CDC ILI baseline threshold
  - auto_relabel_states: 각 HMM state의 epi 비율 ≥ 0.5면 "epi state"로 라벨
  - Cohen's kappa: epi_truth vs HMM-derived epi prediction

Fallback trigger 정량 기준 (PLAN §5.1, v2.0.8b EB-3):
  1) dead state: 어떤 state의 점유율 < 5%
  2) κ failure: 해당 K의 3 seeds 전부에서 최종 수렴 κ < 0.50
  3) σ collapse: log_sigma가 clamp boundary stuck

Two κ functions provided (orthogonal quality dimensions):
  - cohens_kappa_binary  : HMM state vs CDC epi truth (semantic agreement)
                           → PLAN EB-3 fallback trigger uses this
  - cohens_kappa_aligned : Viterbi seq A vs Viterbi seq B (cross-seed stability)
                           → Paper §5.1 reproducibility reporting + ablation
                             baseline comparison (GaussianHMM vs NeuralSwitchingVARHMM)
"""
from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


EPI_THRESHOLD: float = 2.0  # CDC ILI baseline (Phase 0 standard, k_selection.py)
DEAD_STATE_THRESHOLD: float = 0.05  # 5% 점유율 미만이면 dead
KAPPA_FAIL_THRESHOLD: float = 0.50  # 최종 수렴 κ가 이 미만이면 fail
SIGMA_CLAMP_MIN: float = 0.0
SIGMA_CLAMP_MAX: float = 2.0
SIGMA_BOUNDARY_TOL: float = 0.01


def cohens_kappa_binary(viterbi_path: np.ndarray, ili_raw: np.ndarray) -> float:
    """Binary epi/non-epi κ vs CDC threshold (Phase 0 표준).

    Args:
        viterbi_path: [L] integer state assignments (0..K-1)
        ili_raw:      [L] RAW ili_weighted_pct (정규화 X)

    Returns:
        kappa: float ∈ [-1, 1]. ≥0.60 강한 일치, 0.50-0.60 약, <0.50 fail.
    """
    epi_truth = (ili_raw > EPI_THRESHOLD).astype(int)

    K = int(viterbi_path.max()) + 1
    state_is_epi = []
    for k in range(K):
        mask = viterbi_path == k
        if mask.sum() == 0:
            state_is_epi.append(0)
        else:
            state_is_epi.append(int(epi_truth[mask].mean() >= 0.5))
    state_is_epi = np.asarray(state_is_epi, dtype=int)

    epi_pred = state_is_epi[viterbi_path]
    return float(cohen_kappa_score(epi_truth, epi_pred))


def state_occupancy(viterbi_path: np.ndarray, K: int) -> np.ndarray:
    """각 state의 시간 점유율 [K], 합=1.0."""
    counts = np.bincount(viterbi_path, minlength=K).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.zeros(K)
    return counts / total


def is_dead_state(occupancy: np.ndarray, threshold: float = DEAD_STATE_THRESHOLD) -> bool:
    """True iff 어떤 state의 점유율 < threshold."""
    return bool((occupancy < threshold).any())


def is_sigma_collapse(
    log_sigma: torch.Tensor,
    clamp_min: float = SIGMA_CLAMP_MIN,
    clamp_max: float = SIGMA_CLAMP_MAX,
    tol: float = SIGMA_BOUNDARY_TOL,
) -> bool:
    """True iff 어떤 log_sigma가 clamp boundary stuck."""
    at_min = (log_sigma <= clamp_min + tol).any().item()
    at_max = (log_sigma >= clamp_max - tol).any().item()
    return bool(at_min or at_max)


def fallback_trigger(
    final_kappas: list[float],
    dead_states: list[bool],
    sigma_collapses: list[bool],
) -> dict:
    """V=3 fallback 판단 (PLAN §5.1 EB-3 정량 기준).

    Args:
        final_kappas:    해당 K의 모든 seeds의 최종 수렴 κ (e.g., 3 seeds → 3 values)
        dead_states:     해당 K의 모든 seeds의 dead state 발생 여부 (List[bool])
        sigma_collapses: 동일, σ collapse 발생 여부

    Returns:
        dict with keys:
            triggered: bool
            reason: str
            metrics: dict (진단 수치)
    """
    all_kappa_fail = all(k < KAPPA_FAIL_THRESHOLD for k in final_kappas)
    any_dead = any(dead_states)
    any_sigma = any(sigma_collapses)

    triggered = all_kappa_fail or any_dead or any_sigma
    reasons = []
    if all_kappa_fail:
        reasons.append(f"all {len(final_kappas)} seeds final κ<{KAPPA_FAIL_THRESHOLD}")
    if any_dead:
        reasons.append("dead_state observed")
    if any_sigma:
        reasons.append("sigma collapse observed")

    return {
        "triggered": triggered,
        "reason": " + ".join(reasons) if reasons else "none",
        "metrics": {
            "final_kappas": [float(k) for k in final_kappas],
            "all_kappa_fail": all_kappa_fail,
            "any_dead_state": any_dead,
            "any_sigma_collapse": any_sigma,
        },
    }


# ──────────────────────────────────────────────────────────────────
# Cross-seed stability κ (cohens_kappa_aligned)
# ──────────────────────────────────────────────────────────────────
# Orthogonal to cohens_kappa_binary above:
#   binary κ measures "did HMM learn semantically meaningful states"
#   aligned κ measures "do different seeds converge to same state structure"
#
# Used by:
#   - scripts/run_hmm_stage1.py: cross-seed reproducibility (per K, 3-pairwise)
#   - scripts/m1_4_ablation_gaussian_hmm_search.py: GaussianHMM K-selection (§7.4 ablation)
#   - paper §5.1: reproducibility table (κ_min across seed pairs per K)

def _cohens_kappa(s1: np.ndarray, s2: np.ndarray, K: int) -> float:
    """Plain Cohen's κ (no label alignment).

    p_o = observed agreement
    p_e = chance agreement
    κ   = (p_o - p_e) / (1 - p_e)
    """
    T = len(s1)
    if T == 0:
        return 0.0
    p_o = float(np.mean(s1 == s2))
    p_e = sum(float(np.mean(s1 == k)) * float(np.mean(s2 == k)) for k in range(K))
    if p_e >= 1.0 - 1e-10:
        return 1.0 if p_o >= 1.0 - 1e-10 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def cohens_kappa_aligned(s1: np.ndarray, s2: np.ndarray, K: int) -> float:
    """Cohen's κ with optimal permutation alignment of HMM state labels.

    HMM states are unordered — state 0 from seed=42 may correspond to state
    2 from seed=123. We brute-force all K! label permutations on `s2` and
    return the maximum κ (feasible for K ≤ 8: max 40,320 permutations).

    For the CG-Mamba use case (K ∈ {3,4,5}), max 120 perms × T=868 train
    rows runs in ~ms.

    Args:
        s1: [T] integer state sequence (reference, e.g., seed 1 Viterbi)
        s2: [T] integer state sequence (to align, e.g., seed 2 Viterbi)
        K:  number of HMM states

    Returns:
        Best κ across all K! label permutations of s2. Range [-1, 1].
        κ ≥ 0.60 strong agreement, 0.50-0.60 moderate, < 0.50 fail.
    """
    assert len(s1) == len(s2), f"Length mismatch: {len(s1)} vs {len(s2)}"
    assert K <= 8, (
        f"K={K} too large for brute-force permutation alignment "
        f"(K! = {math.factorial(K)}). Use Hungarian assignment if K > 8."
    )

    s2_int = np.asarray(s2, dtype=int)
    best_kappa = -1.0
    for perm in permutations(range(K)):
        # Apply permutation as a lookup (fast vectorized remapping)
        remapped = np.asarray(perm, dtype=int)[s2_int]
        kappa = _cohens_kappa(s1, remapped, K)
        if kappa > best_kappa:
            best_kappa = kappa
    return best_kappa
