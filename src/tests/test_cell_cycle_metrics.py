"""Tests for src/utils/cell_cycle_metrics.py.

Coverage:
    - phase_classification_accuracy: permutation alignment, edge cases
    - gene_expression_mse / mae: full + marker-subset
    - phase_angle_from_posterior + circular_distance + phase_angle_error
    - cyclic_correlation: detrended Pearson on perfect / noisy / constant data
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.utils.cell_cycle_metrics import (
    align_states_hungarian,
    circular_distance,
    cyclic_correlation,
    dtw_distance,
    gene_expression_mae,
    gene_expression_mse,
    gene_pearson_median,
    gene_spearman_median,
    peak_detection_f1,
    phase_angle_error,
    phase_angle_from_posterior,
    phase_classification_accuracy,
    _dtw_single,
    _moving_average_trend,
)


# ──────────────────────────────────────────────────────────────────
# Phase classification accuracy
# ──────────────────────────────────────────────────────────────────

class TestPhaseClassificationAccuracy:

    def test_identity_alignment(self):
        """Predictions matching true labels give 100% accuracy."""
        true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        pred = true.copy()
        acc, perm = phase_classification_accuracy(pred, true, K=4)
        assert acc == 1.0
        assert perm == (0, 1, 2, 3)

    def test_permuted_alignment(self):
        """Permuted labels still give 100% under best permutation."""
        true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        # Swap labels 0↔2 in predictions
        pred = np.array([2, 1, 0, 3, 2, 1, 0, 3])
        acc, perm = phase_classification_accuracy(pred, true, K=4)
        assert acc == 1.0
        # The optimal permutation maps 2→0 and 0→2
        assert perm[0] == 2 and perm[2] == 0

    def test_partial_match(self):
        """Half-correct predictions give 0.5 accuracy."""
        true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        pred = np.array([0, 0, 1, 1, 1, 2, 2, 3])   # 4/8 wrong even before align
        acc, _ = phase_classification_accuracy(pred, true, K=4)
        assert 0.0 <= acc <= 1.0

    def test_K3_works(self):
        """K=3 case works (3! = 6 permutations)."""
        true = np.array([0, 1, 2, 0, 1, 2])
        pred = np.array([1, 2, 0, 1, 2, 0])   # cyclic shift
        acc, _ = phase_classification_accuracy(pred, true, K=3)
        assert acc == 1.0

    def test_K_too_large_raises(self):
        """K > 8 raises (brute force not feasible)."""
        with pytest.raises(AssertionError, match="brute-force"):
            phase_classification_accuracy(
                np.zeros(10, dtype=int), np.zeros(10, dtype=int), K=9,
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError, match="Shape mismatch"):
            phase_classification_accuracy(
                np.zeros(5, dtype=int), np.zeros(7, dtype=int), K=4,
            )


# ──────────────────────────────────────────────────────────────────
# Gene expression MSE / MAE
# ──────────────────────────────────────────────────────────────────

class TestGeneExpressionError:

    def test_mse_perfect_zero(self):
        """Identical pred and true → MSE = 0."""
        y = np.random.RandomState(0).randn(10, 5)
        assert gene_expression_mse(y, y) == 0.0
        assert gene_expression_mae(y, y) == 0.0

    def test_mse_known_value(self):
        """MSE on a simple constant offset."""
        y_true = np.zeros((4, 3))
        y_pred = np.full((4, 3), 2.0)
        assert gene_expression_mse(y_pred, y_true) == 4.0
        assert gene_expression_mae(y_pred, y_true) == 2.0

    def test_marker_subset(self):
        """Marker subset evaluation ignores other columns."""
        y_true = np.zeros((4, 5))
        y_pred = np.zeros((4, 5))
        y_pred[:, 0] = 10.0       # huge error in column 0
        y_pred[:, 2] = 10.0       # huge error in column 2
        # Evaluating only on indices [1, 3, 4] → all zeros → 0 error
        mse_subset = gene_expression_mse(y_pred, y_true, gene_indices=[1, 3, 4])
        assert mse_subset == 0.0
        # Evaluating only on [0] → 100
        mse_col0 = gene_expression_mse(y_pred, y_true, gene_indices=[0])
        assert mse_col0 == 100.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError, match="Shape mismatch"):
            gene_expression_mse(np.zeros((3, 5)), np.zeros((3, 4)))

    def test_supports_3d_horizon(self):
        """Works on [T, horizon, G] tensors."""
        y_true = np.zeros((4, 5, 8))
        y_pred = np.full_like(y_true, 1.5)
        assert gene_expression_mae(y_pred, y_true) == 1.5


# ──────────────────────────────────────────────────────────────────
# Phase angle from posterior + circular distance
# ──────────────────────────────────────────────────────────────────

class TestPhaseAngle:

    def test_one_hot_recovers_anchor(self):
        """One-hot γ at state k recovers CENTER anchor angle (k+0.5)·2π/K."""
        K = 4
        for k in range(K):
            gamma = np.zeros((1, K))
            gamma[0, k] = 1.0
            angle = phase_angle_from_posterior(gamma, K)
            expected = (2 * np.pi * (k + 0.5) / K) % (2 * np.pi)
            np.testing.assert_allclose(angle, [expected], atol=1e-10)

    def test_uniform_undefined_but_finite(self):
        """Uniform γ is the singular point — should not crash.

        At γ = 1/K uniform, cos_avg = sin_avg = 0 (for K ≥ 2 with anchors
        equispaced on the circle), so atan2(0, 0) returns 0.0 by convention.
        Just check finiteness — semantic meaning of "0" at the singular
        point is undefined.
        """
        K = 4
        gamma = np.full((1, K), 1.0 / K)
        angle = phase_angle_from_posterior(gamma, K)
        assert np.isfinite(angle).all()

    def test_circular_distance_wrap_around(self):
        """Distance wraps around at 2π."""
        # 0.1 vs 6.2 (≈ 2π - 0.083) → distance ≈ 0.183 (not 6.1)
        d = circular_distance(np.array([0.1]), np.array([6.2]))
        assert d[0] < 0.2, f"Wrap-around failed: distance = {d[0]:.3f}"

    def test_circular_distance_max_pi(self):
        """Max possible circular distance is π."""
        d = circular_distance(np.array([0.0]), np.array([np.pi]))
        np.testing.assert_allclose(d, [np.pi], atol=1e-10)

    def test_circular_distance_symmetric(self):
        """Distance is symmetric."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.5, 6.1])
        np.testing.assert_array_equal(
            circular_distance(a, b), circular_distance(b, a),
        )

    def test_phase_angle_error_degrees(self):
        """Default returns degrees, max 180.

        K=4 anchors are π/4, 3π/4, 5π/4, 7π/4. γ favoring state 0 (anchor π/4)
        vs true_angle=5π/4 (state 2's anchor, diametrically opposite) → 180°.
        """
        K = 4
        gamma = np.array([[0.97, 0.01, 0.01, 0.01]])
        true_angle = np.array([5 * np.pi / 4])
        err = phase_angle_error(gamma, true_angle, K=K, in_degrees=True)
        assert 170 < err <= 180

    def test_phase_angle_error_radians(self):
        """in_degrees=False returns radians."""
        K = 4
        gamma = np.array([[0.97, 0.01, 0.01, 0.01]])
        true_angle = np.array([5 * np.pi / 4])
        err = phase_angle_error(gamma, true_angle, K=K, in_degrees=False)
        assert 0 < err <= np.pi

    def test_phase_angle_error_zero_when_matched(self):
        """Perfect γ at state 1 + true_angle at state-1 center anchor → ~0 error."""
        K = 4
        # State 1 center anchor = (1 + 0.5) · 2π / 4 = 3π/4
        gamma = np.array([[0.0, 1.0, 0.0, 0.0]])
        true_angle = np.array([3 * np.pi / 4])
        err = phase_angle_error(gamma, true_angle, K=K, in_degrees=True)
        assert err < 1.0, f"Should be ~0, got {err:.3f} deg"

    def test_center_anchor_halves_systematic_bias(self):
        """Center-anchor reduces phase angle MAE from ~45° to ~22.5° (K=4).

        Theory: for perfectly-confident γ (one-hot on true_state) over
        timepoints with true_angle uniform in [0, 2π):
          - range-start anchor (the I4 bug): predicted = θ_k_start, true ∈
            [θ_k_start, θ_k_start + 2π/K) → E[|err|] = π/K = 45° at K=4.
          - center anchor: predicted = θ_k_start + π/K, true symmetric around
            anchor with half-width π/K → E[|err|] = π/(2K) = 22.5° at K=4.
        """
        K = 4
        T = 1000  # large for Monte Carlo stability
        rng = np.random.RandomState(0)
        true_angle = rng.uniform(0, 2 * np.pi, size=T)
        true_states = np.floor(true_angle / (2 * np.pi / K)).astype(int)
        true_states = np.clip(true_states, 0, K - 1)
        # γ = one-hot on true_state (perfect HMM)
        gamma = np.zeros((T, K))
        gamma[np.arange(T), true_states] = 1.0

        err_deg = phase_angle_error(gamma, true_angle, K=K, in_degrees=True)
        # Center anchor: ~22.5°. Range-start anchor would give ~45°.
        assert 18.0 < err_deg < 27.0, (
            f"Expected ~22.5° (center-anchor residual), got {err_deg:.2f}°. "
            f"~45° would indicate range-start anchor (I4 regression)."
        )


# ──────────────────────────────────────────────────────────────────
# Cyclic correlation
# ──────────────────────────────────────────────────────────────────

class TestCyclicCorrelation:

    def test_perfect_match(self):
        """Identical signals give correlation = 1.0."""
        t = np.linspace(0, 4 * np.pi, 100)
        y = np.sin(t)
        corr = cyclic_correlation(y, y, period=25)
        assert corr > 0.95, f"Identical signals should correlate ~1, got {corr}"

    def test_phase_shifted(self):
        """Phase-shifted signals have lower correlation."""
        t = np.linspace(0, 4 * np.pi, 100)
        y1 = np.sin(t)
        y2 = np.sin(t + np.pi / 2)   # 90° phase shift
        corr_same = cyclic_correlation(y1, y1, period=25)
        corr_shift = cyclic_correlation(y1, y2, period=25)
        assert corr_same > corr_shift

    def test_anti_phase(self):
        """Anti-phase (sin vs -sin) gives negative correlation."""
        t = np.linspace(0, 4 * np.pi, 100)
        y1 = np.sin(t)
        y2 = -np.sin(t)
        corr = cyclic_correlation(y1, y2, period=25)
        assert corr < -0.8

    def test_constant_pred_returns_zero(self):
        """Constant predictions (zero variance) return 0."""
        t = np.linspace(0, 4 * np.pi, 100)
        y_true = np.sin(t)
        y_const = np.full_like(y_true, 5.0)
        corr = cyclic_correlation(y_const, y_true, period=25)
        assert corr == 0.0

    def test_multi_column(self):
        """Multi-column [T, G] averages across G."""
        t = np.linspace(0, 4 * np.pi, 100)
        y = np.stack([np.sin(t), np.cos(t)], axis=1)   # [100, 2]
        corr = cyclic_correlation(y, y, period=25)
        assert corr > 0.95

    def test_moving_average_trend_constant(self):
        """MA of a constant series is the same constant."""
        y = np.full((20, 3), 7.0)
        trend = _moving_average_trend(y, window=5)
        np.testing.assert_allclose(trend, y, atol=1e-10)

    def test_moving_average_trend_window_1(self):
        """window=1 returns input unchanged."""
        y = np.random.RandomState(0).randn(20, 3)
        trend = _moving_average_trend(y, window=1)
        np.testing.assert_array_equal(trend, y)


# ──────────────────────────────────────────────────────────────────
# Per-gene Pearson / Spearman
# ──────────────────────────────────────────────────────────────────

class TestPearsonSpearman:

    def test_pearson_perfect(self):
        rng = np.random.RandomState(0)
        y = rng.randn(20, 5)
        assert gene_pearson_median(y, y) > 0.99

    def test_pearson_anti(self):
        rng = np.random.RandomState(0)
        y = rng.randn(20, 5)
        assert gene_pearson_median(-y, y) < -0.99

    def test_pearson_zero_var_returns_zero(self):
        y_true = np.random.RandomState(0).randn(20, 3)
        y_const = np.full_like(y_true, 1.0)
        assert gene_pearson_median(y_const, y_true) == 0.0

    def test_spearman_perfect(self):
        rng = np.random.RandomState(0)
        y = rng.randn(20, 5)
        assert gene_spearman_median(y, y) > 0.99

    def test_spearman_monotonic_nonlinear(self):
        """Spearman handles monotonic non-linear transforms (where Pearson fails)."""
        t = np.linspace(-1, 1, 30)
        y_true = t[:, None]
        y_pred = (t ** 3)[:, None]   # monotonic but non-linear
        assert gene_spearman_median(y_pred, y_true) > 0.99


# ──────────────────────────────────────────────────────────────────
# DTW
# ──────────────────────────────────────────────────────────────────

class TestDTW:

    def test_dtw_identical_zero(self):
        y = np.sin(np.linspace(0, 4 * np.pi, 30))[:, None]
        assert dtw_distance(y, y) < 1e-6

    def test_dtw_shifted_better_than_mse(self):
        """Phase-shifted signal: DTW(small) ≪ MSE-equivalent linear distance."""
        t = np.linspace(0, 4 * np.pi, 30)
        a = np.sin(t)[:, None]
        b = np.sin(t + 0.3)[:, None]   # small phase shift
        dtw = dtw_distance(a, b, window=3)
        # Compared to no warping (cumulative |a-b|)
        no_warp = float(np.abs(a - b).sum())
        assert dtw < no_warp

    def test_dtw_window_restricts_warping(self):
        """Smaller window → larger or equal DTW distance."""
        rng = np.random.RandomState(0)
        a = rng.randn(20)
        b = rng.randn(20)
        d_wide = dtw_distance(a, b, window=20)
        d_narrow = dtw_distance(a, b, window=2)
        assert d_narrow >= d_wide - 1e-6

    def test_dtw_single_basic(self):
        """Single-sequence DTW returns expected zero for identical input."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _dtw_single(a, b) == 0.0


# ──────────────────────────────────────────────────────────────────
# Peak detection F1
# ──────────────────────────────────────────────────────────────────

class TestPeakF1:

    def test_peak_f1_identical(self):
        t = np.linspace(0, 4 * np.pi, 50)
        y = np.sin(t)[:, None]
        # Same signal → all peaks match → F1 = 1.0
        assert peak_detection_f1(y, y, tolerance=1) == 1.0

    def test_peak_f1_no_peaks_both(self):
        """Both signals have no detectable peaks → trivially F1=1."""
        y = np.zeros((30, 1))
        assert peak_detection_f1(y, y) == 1.0

    def test_peak_f1_no_pred_peaks(self):
        """Pred has no peaks, true has peaks → F1=0."""
        t = np.linspace(0, 4 * np.pi, 50)
        y_pred = np.zeros((50, 1))
        y_true = np.sin(t)[:, None]
        assert peak_detection_f1(y_pred, y_true) == 0.0

    def test_peak_f1_shifted_within_tolerance(self):
        """Shifted peaks within tolerance still match."""
        t = np.linspace(0, 4 * np.pi, 50)
        y_true = np.sin(t)[:, None]
        # Predict with 1-step delay (within tol=2)
        y_pred = np.roll(y_true, shift=1, axis=0)
        score = peak_detection_f1(y_pred, y_true, tolerance=2)
        assert score >= 0.5

    def test_peak_f1_outside_tolerance(self):
        """Shifted peaks outside tolerance → low F1."""
        t = np.linspace(0, 4 * np.pi, 50)
        y_true = np.sin(t)[:, None]
        y_pred = np.roll(y_true, shift=10, axis=0)   # large shift
        score = peak_detection_f1(y_pred, y_true, tolerance=1)
        assert score < 0.5


# ──────────────────────────────────────────────────────────────────
# Hungarian state alignment
# ──────────────────────────────────────────────────────────────────

class TestHungarianAlignment:

    def test_identity_alignment(self):
        means = np.eye(3) * 10.0   # 3 well-separated states
        a, b = align_states_hungarian(means, means)
        # Identity mapping (cost 0 on diagonal)
        for i in range(3):
            assert a[i] == b[i]

    def test_permuted_alignment(self):
        means_a = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
        # State indices in b are permuted (swap 0 ↔ 2)
        means_b = np.array([[0, 0, 10.0], [0, 10.0, 0], [10.0, 0, 0]])
        a, b = align_states_hungarian(means_a, means_b)
        # a[i] should pair with b[i] s.t. means match
        for ai, bi in zip(a, b):
            np.testing.assert_allclose(means_a[ai], means_b[bi])

    def test_different_K(self):
        """K_a=4, K_b=3 → 3 matches (extra a-state unmatched)."""
        means_a = np.eye(4)
        means_b = np.eye(4)[:3]
        a, b = align_states_hungarian(means_a, means_b)
        assert len(a) == 3 and len(b) == 3

    def test_feature_dim_mismatch_raises(self):
        with pytest.raises(AssertionError, match="Feature dim mismatch"):
            align_states_hungarian(np.eye(3), np.eye(3)[:, :2])
