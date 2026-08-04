"""Unit tests for `cohens_kappa_aligned` in `src/utils/metrics.py`.

Cross-seed stability κ: orthogonal to the binary κ (HMM state vs CDC epi truth)
already covered elsewhere in metrics.py. These tests verify the brute-force
permutation alignment used for cross-seed reproducibility (paper §5.1).

Run: pytest -xvs src/tests/test_metrics_kappa.py
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from src.utils.metrics import (
    _cohens_kappa,
    cohens_kappa_aligned,
)


# ─────────────────────────────────────────────────────────────────
# T1 — perfect agreement with permuted labels (1.0)
# ─────────────────────────────────────────────────────────────────

def test_perfect_agreement_permuted_labels():
    """Two identical sequences with a permuted label mapping → κ = 1.0."""
    s1 = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    # Permutation 0→2, 1→0, 2→1
    s2 = np.array([2, 2, 0, 0, 1, 1, 2, 0, 1])
    kappa = cohens_kappa_aligned(s1, s2, K=3)
    assert abs(kappa - 1.0) < 1e-9, f"κ = {kappa:.6f}, expected 1.0"


def test_identical_sequences():
    """Identity (no permutation needed) → κ = 1.0."""
    rng = np.random.RandomState(0)
    s1 = rng.randint(0, 3, size=200)
    kappa = cohens_kappa_aligned(s1, s1.copy(), K=3)
    assert abs(kappa - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────
# T2 — random sequences → κ ≈ 0
# ─────────────────────────────────────────────────────────────────

def test_random_sequences_low_kappa():
    """Independent random sequences → aligned κ near 0 (low agreement).

    Note: 'aligned' κ on random sequences is slightly biased upward (we pick
    the best of K! permutations), but for T=1000, K=3 it should stay < 0.2.
    """
    rng = np.random.RandomState(42)
    s1 = rng.randint(0, 3, size=1000)
    s2 = rng.randint(0, 3, size=1000)
    kappa = cohens_kappa_aligned(s1, s2, K=3)
    assert kappa < 0.2, f"κ = {kappa:.4f}, expected < 0.2 for random sequences"


# ─────────────────────────────────────────────────────────────────
# T3 — K range support (3, 4, 5) — CG-Mamba K-search grid
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [3, 4, 5])
def test_aligned_kappa_handles_full_K_grid(K):
    """All K in CG-Mamba grid {3,4,5} run without error and produce valid κ."""
    rng = np.random.RandomState(K)
    s1 = rng.randint(0, K, size=500)
    s2 = rng.randint(0, K, size=500)
    kappa = cohens_kappa_aligned(s1, s2, K=K)
    assert -1.0 <= kappa <= 1.0, f"κ={kappa} out of [-1, 1]"


# ─────────────────────────────────────────────────────────────────
# T4 — K > 8 rejection (factorial explosion guard)
# ─────────────────────────────────────────────────────────────────

def test_rejects_K_too_large():
    """K > 8 raises AssertionError (would require > 40,320 permutations)."""
    s1 = np.array([0, 1, 2])
    s2 = np.array([0, 1, 2])
    with pytest.raises(AssertionError, match="K=9 too large"):
        cohens_kappa_aligned(s1, s2, K=9)


# ─────────────────────────────────────────────────────────────────
# T5 — length mismatch rejection
# ─────────────────────────────────────────────────────────────────

def test_rejects_length_mismatch():
    s1 = np.array([0, 1, 2])
    s2 = np.array([0, 1, 2, 0])
    with pytest.raises(AssertionError, match="Length mismatch"):
        cohens_kappa_aligned(s1, s2, K=3)


# ─────────────────────────────────────────────────────────────────
# T6 — plain _cohens_kappa (no alignment) for direct verification
# ─────────────────────────────────────────────────────────────────

def test_plain_kappa_matches_known_value():
    """_cohens_kappa with hand-computed example.

    s1 = [0, 0, 1, 1], s2 = [0, 1, 1, 1]
    p_o = 3/4 = 0.75
    p_e = (2/4)(1/4) + (2/4)(3/4) = 0.125 + 0.375 = 0.5
    κ = (0.75 - 0.5) / (1 - 0.5) = 0.5
    """
    s1 = np.array([0, 0, 1, 1])
    s2 = np.array([0, 1, 1, 1])
    k = _cohens_kappa(s1, s2, K=2)
    assert abs(k - 0.5) < 1e-9, f"κ = {k:.6f}, expected 0.5"


def test_plain_kappa_perfect():
    s = np.array([0, 1, 2, 0, 1, 2])
    assert abs(_cohens_kappa(s, s, K=3) - 1.0) < 1e-9


def test_plain_kappa_edge_empty():
    """Empty input → 0 (no division by zero)."""
    assert _cohens_kappa(np.array([], dtype=int), np.array([], dtype=int), K=3) == 0.0


# ─────────────────────────────────────────────────────────────────
# T7 — production timing (CG-Mamba grid use case)
# ─────────────────────────────────────────────────────────────────

def test_production_timing_K5_T868():
    """K=5, T=868 (train length) must run in < 100ms (3-pairwise × 9 runs ≈ 1s)."""
    rng = np.random.RandomState(123)
    s1 = rng.randint(0, 5, size=868)
    s2 = rng.randint(0, 5, size=868)
    t0 = time.time()
    _ = cohens_kappa_aligned(s1, s2, K=5)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms < 100, f"Too slow: {elapsed_ms:.1f}ms (budget 100ms)"
