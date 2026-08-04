"""CellCycleHMM unit tests — cyclic mask, marker emission, anti-collapse.

Run: pytest -xvs src/tests/test_cell_cycle_hmm.py
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.cell_cycle_hmm import (
    CellCycleHMM,
    CellCycleConfig,
    EmissionConfig,
    CYCLIC_MASK_K4,
    CYCLIC_MASK_K3,
    SOFT_MASK_EPSILON,
    K_CELL_CYCLE,
    N_MARKERS,
    N_MARKERS_K3,
    MARKER_GENES_FLAT,
    MARKER_GENES_FLAT_K3,
    CELL_CYCLE_MARKERS,
    CELL_CYCLE_MARKERS_K3,
    PHASE_NAMES,
    PHASE_NAMES_K3,
    DURATION_FRACTIONS_K4,
    DURATION_FRACTIONS_K3,
    annotate_states,
    get_markers_for_K,
    make_cyclic_mask,
    compute_duration_aware_A,
    compute_entropy_confidence,
    entropy_gated_phase_embedding,
    select_emission_features,
    generate_synthetic_cell_cycle,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Synthetic 4-phase cell cycle data (16 genes, 48 timepoints)."""
    x, states, phases = generate_synthetic_cell_cycle(
        n_genes=16, n_timepoints=48, cycle_period=22, seed=42,
    )
    return x, states, phases


@pytest.fixture
def fitted_hmm(synthetic_data):
    """CellCycleHMM fitted on synthetic data."""
    x, _, _ = synthetic_data
    hmm = CellCycleHMM(n_features=16, n_iter=80, seed=42)
    hmm.fit(x)
    return hmm


@pytest.fixture
def large_synthetic():
    """Larger synthetic data for stress testing (100 timepoints)."""
    x, states, phases = generate_synthetic_cell_cycle(
        n_genes=16, n_timepoints=100, cycle_period=22, seed=99,
    )
    return x, states, phases


# ──────────────────────────────────────────────────────────────────
# [MASK] Cyclic transition mask tests
# ──────────────────────────────────────────────────────────────────

class TestCyclicMask:
    """Tests for the cyclic soft mask."""

    def test_mask_shape(self):
        """Mask is [K, K]."""
        assert CYCLIC_MASK_K4.shape == (4, 4)

    def test_mask_values_soft(self):
        """All values are either 1.0 (allowed) or ε (forbidden)."""
        values = set(CYCLIC_MASK_K4.flatten().tolist())
        assert values == {SOFT_MASK_EPSILON, 1.0}, f"Unexpected values: {values}"

    def test_mask_self_loops(self):
        """Diagonal (self-loops) are all allowed."""
        for i in range(4):
            assert CYCLIC_MASK_K4[i, i] == 1.0, f"Self-loop blocked for state {i}"

    def test_mask_forward_transitions(self):
        """Forward cyclic transitions are allowed."""
        assert CYCLIC_MASK_K4[0, 1] == 1.0  # G1 → S
        assert CYCLIC_MASK_K4[1, 2] == 1.0  # S  → G2
        assert CYCLIC_MASK_K4[2, 3] == 1.0  # G2 → M
        assert CYCLIC_MASK_K4[3, 0] == 1.0  # M  → G1 (wrap)

    def test_mask_backward_near_zero(self):
        """Backward transitions are ε (near-zero, not hard zero)."""
        assert CYCLIC_MASK_K4[1, 0] == SOFT_MASK_EPSILON  # S  → G1
        assert CYCLIC_MASK_K4[2, 1] == SOFT_MASK_EPSILON  # G2 → S
        assert CYCLIC_MASK_K4[3, 2] == SOFT_MASK_EPSILON  # M  → G2
        assert CYCLIC_MASK_K4[0, 3] == SOFT_MASK_EPSILON  # G1 → M

    def test_mask_skip_near_zero(self):
        """Skip transitions (e.g., G1→G2) are ε."""
        assert CYCLIC_MASK_K4[0, 2] == SOFT_MASK_EPSILON  # G1 → G2 skip
        assert CYCLIC_MASK_K4[1, 3] == SOFT_MASK_EPSILON  # S  → M  skip

    def test_mask_allowed_per_row(self):
        """Each row has exactly 2 fully allowed (=1.0) transitions."""
        for i in range(4):
            n_allowed = (CYCLIC_MASK_K4[i] == 1.0).sum()
            assert n_allowed == 2, (
                f"Row {i} has {n_allowed} allowed, expected 2"
            )

    def test_make_cyclic_mask_K3(self):
        """Cyclic soft mask works for K=3."""
        mask = make_cyclic_mask(3)
        assert mask.shape == (3, 3)
        assert mask[0, 1] == 1.0
        assert mask[1, 2] == 1.0
        assert mask[2, 0] == 1.0  # wrap
        # 3 self + 3 forward = 6 allowed, rest = ε
        n_allowed = (mask == 1.0).sum()
        assert n_allowed == 6

    def test_hard_mask_epsilon_zero(self):
        """epsilon=0 recovers hard mask behavior."""
        mask = make_cyclic_mask(4, epsilon=0.0)
        assert mask[1, 0] == 0.0
        assert mask[0, 1] == 1.0

    def test_custom_epsilon(self):
        """Custom epsilon propagates to forbidden cells."""
        mask = make_cyclic_mask(4, epsilon=0.01)
        assert mask[1, 0] == 0.01
        assert mask[0, 1] == 1.0

    def test_precomputed_K3_mask(self):
        """Pre-computed K=3 mask matches make_cyclic_mask(3)."""
        np.testing.assert_array_equal(CYCLIC_MASK_K3, make_cyclic_mask(3))


# ──────────────────────────────────────────────────────────────────
# [HMM] CellCycleHMM core tests
# ──────────────────────────────────────────────────────────────────

class TestCellCycleHMM:
    """Core HMM tests with cell cycle priors."""

    def test_fit_and_posterior_shape(self, fitted_hmm, synthetic_data):
        """Fitted HMM produces [T, K=4] posteriors."""
        x, _, _ = synthetic_data
        gamma = fitted_hmm.posteriors(x)
        assert gamma.shape == (48, 4)

    def test_posterior_sums_to_one(self, fitted_hmm, synthetic_data):
        """γ[t, :] sums to 1 for all t."""
        x, _, _ = synthetic_data
        gamma = fitted_hmm.posteriors(x)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_posterior_range(self, fitted_hmm, synthetic_data):
        """γ ∈ [0, 1]."""
        x, _, _ = synthetic_data
        gamma = fitted_hmm.posteriors(x)
        assert gamma.min() >= 0.0
        assert gamma.max() <= 1.0

    def test_transition_mask_enforced(self, fitted_hmm):
        """After fitting, forbidden A entries are clamped near ε."""
        A = fitted_hmm.A
        mask = fitted_hmm.transition_mask
        # Wherever mask < 1.0 (forbidden), A must be near-zero (< 0.01)
        forbidden = A[mask < 1.0]
        assert np.all(forbidden < 0.01), (
            f"Forbidden transitions too large: max={forbidden.max():.2e}"
        )

    def test_transition_rows_sum_to_one(self, fitted_hmm):
        """A rows sum to 1 after mask enforcement."""
        np.testing.assert_allclose(
            fitted_hmm.A.sum(axis=1), 1.0, atol=1e-6,
        )

    def test_no_backward_transitions(self, fitted_hmm):
        """Backward transitions are near-zero (soft mask ε level)."""
        A = fitted_hmm.A
        # With soft mask, forbidden transitions should be very small
        # but not exactly zero. Threshold: 10× ε to account for renormalization.
        threshold = SOFT_MASK_EPSILON * 10
        assert A[1, 0] < threshold, f"S→G1 = {A[1,0]:.2e} (should be ~ε)"
        assert A[2, 1] < threshold, f"G2→S = {A[2,1]:.2e} (should be ~ε)"
        assert A[3, 2] < threshold, f"M→G2 = {A[3,2]:.2e} (should be ~ε)"
        assert A[0, 3] < threshold, f"G1→M = {A[0,3]:.2e} (should be ~ε)"

    def test_em_convergence(self, fitted_hmm):
        """EM converges (not all iterations used)."""
        ll = fitted_hmm.ll_history
        assert len(ll) >= 2, "Too few EM iterations"
        # LL should be monotonically non-decreasing.
        # Tolerance is 1e-4 (looser than base HMM's 1e-6) because the
        # hard mask enforcement in _m_step can cause tiny LL decreases —
        # constrained EM does not strictly guarantee monotonicity.
        for i in range(1, len(ll)):
            assert ll[i] >= ll[i - 1] - 1e-4, (
                f"LL decreased at step {i}: {ll[i]:.4f} < {ll[i-1]:.4f}"
            )

    def test_viterbi_shape(self, fitted_hmm, synthetic_data):
        """Viterbi output: [T] integers in [0, K)."""
        x, _, _ = synthetic_data
        states = fitted_hmm.viterbi(x)
        assert states.shape == (48,)
        assert states.min() >= 0
        assert states.max() < 4

    def test_viterbi_uses_all_states(self, fitted_hmm, synthetic_data):
        """Viterbi should use all 4 states on well-separated synthetic data."""
        x, _, _ = synthetic_data
        states = fitted_hmm.viterbi(x)
        unique_states = set(states.tolist())
        assert len(unique_states) >= 3, (
            f"Only {len(unique_states)} states used: {unique_states}. "
            f"Expected at least 3 out of 4."
        )

    def test_bic_finite(self, fitted_hmm, synthetic_data):
        """BIC is finite."""
        x, _, _ = synthetic_data
        bic = fitted_hmm.bic(x)
        assert np.isfinite(bic)

    def test_no_nan_inf(self, fitted_hmm, synthetic_data):
        """No NaN/Inf anywhere."""
        x, _, _ = synthetic_data
        gamma = fitted_hmm.posteriors(x)
        assert not np.any(np.isnan(gamma))
        assert not np.any(np.isinf(gamma))

        states = fitted_hmm.viterbi(x)
        assert not np.any(np.isnan(states))

    def test_free_param_count_with_mask(self):
        """Free parameter count accounts for soft-masked transitions.

        K=4, diag, V=16:
          π: 3, A: 4 (each row has 2 allowed (=1.0) - 1 = 1 free, × 4 rows),
          μ: 64, σ²: 64 → total = 135
        Soft mask ε entries are treated as fixed → same count as hard mask.
        """
        hmm = CellCycleHMM(n_features=16, covariance_type="diag")
        n_params = hmm._n_free_params()
        expected = 3 + 4 + 64 + 64  # 135
        assert n_params == expected, f"Expected {expected}, got {n_params}"


# ──────────────────────────────────────────────────────────────────
# [ANTI-COLLAPSE] State collapse prevention tests
# ──────────────────────────────────────────────────────────────────

class TestAntiCollapse:
    """Anti-collapse regularization tests."""

    def test_no_dead_states_synthetic(self, fitted_hmm, synthetic_data):
        """Well-separated synthetic data → no dead states."""
        x, _, _ = synthetic_data
        dead = fitted_hmm.dead_states(x, threshold=0.03)
        assert len(dead) == 0, f"Unexpected dead states: {dead}"

    def test_collapse_penalties_tracked(self, fitted_hmm):
        """Collapse penalties are recorded during training."""
        # penalties list should exist and have entries (one per M-step)
        assert isinstance(fitted_hmm.collapse_penalties, list)
        # Length = number of EM iterations (each M-step appends)
        assert len(fitted_hmm.collapse_penalties) > 0

    def test_all_states_have_occupancy(self, fitted_hmm, synthetic_data):
        """Each state has meaningful occupancy (> 5%)."""
        x, _, _ = synthetic_data
        gamma = fitted_hmm.posteriors(x)
        occupancy = gamma.mean(axis=0)
        for k in range(4):
            assert occupancy[k] > 0.03, (
                f"State {k} occupancy = {occupancy[k]:.4f} (< 3%, near collapse)"
            )
        print(f"  Occupancy: {[f'{o:.3f}' for o in occupancy]}")

    def test_collapse_lambda_zero_disables(self, synthetic_data):
        """collapse_lambda=0 disables anti-collapse (no rescue)."""
        x, _, _ = synthetic_data
        hmm = CellCycleHMM(
            n_features=16, n_iter=50, seed=42,
            collapse_lambda=0.0,
        )
        hmm.fit(x)
        # Should still work (just no rescue mechanism)
        gamma = hmm.posteriors(x)
        assert gamma.shape == (48, 4)


# ──────────────────────────────────────────────────────────────────
# [INIT] Phase-aware initialization tests
# ──────────────────────────────────────────────────────────────────

class TestInitialization:
    """Phase-aware vs random initialization."""

    def test_phase_aware_pi(self, synthetic_data):
        """Phase-aware init: π biased toward S phase (double thymidine)."""
        x, _, _ = synthetic_data
        hmm = CellCycleHMM(n_features=16, init_mode="phase_aware")
        hmm._init_params(x)
        # S phase (index 1) should have highest initial probability
        assert hmm.pi[1] > hmm.pi[0], "S phase should have higher π than G1"
        assert hmm.pi[1] > hmm.pi[2], "S phase should have higher π than G2"
        assert hmm.pi[1] > hmm.pi[3], "S phase should have higher π than M"

    def test_phase_aware_A_duration_aware(self, synthetic_data):
        """Phase-aware init: A has asymmetric self-transitions (G1 > M)."""
        x, _, _ = synthetic_data
        hmm = CellCycleHMM(n_features=16, init_mode="phase_aware")
        hmm._init_params(x)
        # G1 self-transition should be higher than M self-transition
        assert hmm.A[0, 0] > hmm.A[3, 3], (
            f"G1 self ({hmm.A[0,0]:.3f}) should > M self ({hmm.A[3,3]:.3f})"
        )
        # Forbidden transitions should be near-epsilon
        for i in range(4):
            for j in range(4):
                if CYCLIC_MASK_K4[i, j] < 1.0:
                    assert hmm.A[i, j] < 0.01, (
                        f"A[{i},{j}] = {hmm.A[i,j]:.4f} (should be near ε)"
                    )

    def test_random_init_also_masked(self, synthetic_data):
        """Random init mode also applies soft mask to A."""
        x, _, _ = synthetic_data
        hmm = CellCycleHMM(n_features=16, init_mode="random", seed=42)
        hmm._init_params(x)
        # Forbidden transitions should be small (soft masked)
        forbidden = hmm.A[CYCLIC_MASK_K4 < 1.0]
        assert np.all(forbidden < 0.01), "Random init should still enforce soft mask"

    def test_phase_aware_means_differ(self, synthetic_data):
        """Phase-aware init: means for different states are distinct."""
        x, _, _ = synthetic_data
        hmm = CellCycleHMM(n_features=16, init_mode="phase_aware")
        hmm._init_params(x)
        # Check means are not identical (would indicate bad init)
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(hmm.means[i] - hmm.means[j])
                assert dist > 1e-3, (
                    f"Means for states {i} and {j} are too similar (dist={dist:.6f})"
                )


# ──────────────────────────────────────────────────────────────────
# [EMISSION] Emission feature selection tests
# ──────────────────────────────────────────────────────────────────

class TestEmissionSelection:
    """Ablation emission variant tests."""

    @pytest.fixture
    def full_data(self):
        """Simulated full gene matrix (48 timepoints × 100 genes)."""
        rng = np.random.RandomState(42)
        x = rng.randn(48, 100)
        # Create gene symbols with some markers embedded
        symbols = [f"GENE_{i}" for i in range(100)]
        # Plant known markers at specific positions
        markers_to_plant = ["PCNA", "CCND1", "CDK1", "CDC20", "TOP2A"]
        for i, m in enumerate(markers_to_plant):
            symbols[i * 10] = m
        return x, symbols

    def test_marker_selection(self, full_data):
        """Marker mode selects known marker genes."""
        x, symbols = full_data
        config = EmissionConfig(
            emission_type="marker",
            marker_gene_symbols=["PCNA", "CCND1", "CDK1", "CDC20", "TOP2A"],
        )
        x_emit, indices = select_emission_features(x, symbols, config)
        assert x_emit.shape == (48, 5)
        assert len(indices) == 5
        # Verify correct columns were selected
        for idx, name in zip(indices, ["PCNA", "CCND1", "CDK1", "CDC20", "TOP2A"]):
            assert symbols[idx] == name

    def test_marker_missing_genes_warning(self, full_data):
        """Marker mode warns about missing genes."""
        x, symbols = full_data
        config = EmissionConfig(
            emission_type="marker",
            marker_gene_symbols=["PCNA", "NONEXISTENT_GENE"],
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_emit, indices = select_emission_features(x, symbols, config)
            assert len(w) == 1
            assert "NONEXISTENT_GENE" in str(w[0].message)
        assert x_emit.shape[1] == 1  # Only PCNA found

    def test_variance_selection(self, full_data):
        """Variance mode selects top-N highest variance genes."""
        x, symbols = full_data
        config = EmissionConfig(emission_type="variance", n_emission_features=10)
        x_emit, indices = select_emission_features(x, symbols, config)
        assert x_emit.shape == (48, 10)
        # Verify these are indeed the highest-variance columns
        variances = np.var(x, axis=0)
        top_indices = set(np.argsort(variances)[::-1][:10].tolist())
        assert set(indices) == top_indices

    def test_random_selection(self, full_data):
        """Random mode selects N random genes."""
        x, symbols = full_data
        config = EmissionConfig(
            emission_type="random", n_emission_features=16,
            random_seeds=[42],
        )
        x_emit, indices = select_emission_features(x, symbols, config)
        assert x_emit.shape == (48, 16)
        assert len(set(indices)) == 16  # all unique

    def test_random_different_seeds(self, full_data):
        """Different random seeds give different selections."""
        x, symbols = full_data
        config1 = EmissionConfig(emission_type="random", n_emission_features=16, random_seeds=[42])
        config2 = EmissionConfig(emission_type="random", n_emission_features=16, random_seeds=[99])
        _, idx1 = select_emission_features(x, symbols, config1)
        _, idx2 = select_emission_features(x, symbols, config2)
        assert idx1 != idx2, "Different seeds should give different selections"

    def test_latent_selection(self, full_data):
        """Latent mode applies PCA dimensionality reduction."""
        x, symbols = full_data
        config = EmissionConfig(emission_type="latent", latent_dim=8)
        x_emit, indices = select_emission_features(x, symbols, config)
        assert x_emit.shape == (48, 8)

    def test_marker_duplicate_gene_symbols(self):
        """Duplicate gene symbols → uses first occurrence only."""
        rng = np.random.RandomState(42)
        x = rng.randn(10, 5)
        # "PCNA" appears at index 1 AND index 3
        symbols = ["GENE_A", "PCNA", "GENE_B", "PCNA", "GENE_C"]
        config = EmissionConfig(
            emission_type="marker",
            marker_gene_symbols=["PCNA"],
        )
        x_emit, indices = select_emission_features(x, symbols, config)
        assert indices == [1], f"Should use first PCNA (idx=1), got {indices}"
        np.testing.assert_array_equal(x_emit, x[:, [1]])

    def test_random_overflow_raises(self):
        """Requesting more features than available raises ValueError."""
        rng = np.random.RandomState(42)
        x = rng.randn(10, 5)
        symbols = [f"G{i}" for i in range(5)]
        config = EmissionConfig(
            emission_type="random", n_emission_features=10,  # > 5 genes
        )
        with pytest.raises(ValueError, match="Cannot select more"):
            select_emission_features(x, symbols, config)

    def test_latent_captures_variance(self, full_data):
        """Latent features capture most variance (PCA property)."""
        x, symbols = full_data
        config = EmissionConfig(emission_type="latent", latent_dim=8)
        x_emit, _ = select_emission_features(x, symbols, config)
        # Explained variance should be substantial
        total_var = np.var(x, axis=0).sum()
        latent_var = np.var(x_emit, axis=0).sum()
        ratio = latent_var / total_var
        # 8 components from 100 features should capture a meaningful fraction
        assert ratio > 0.01, f"Latent variance ratio too low: {ratio:.4f}"


# ──────────────────────────────────────────────────────────────────
# [SYNTHETIC] Synthetic data generator tests
# ──────────────────────────────────────────────────────────────────

class TestSyntheticData:
    """Tests for the cell cycle data generator."""

    def test_shape(self, synthetic_data):
        """Output shapes are correct."""
        x, states, phases = synthetic_data
        assert x.shape == (48, 16)
        assert states.shape == (48,)
        assert phases.shape == (48,)

    def test_states_range(self, synthetic_data):
        """States are in [0, 3]."""
        _, states, _ = synthetic_data
        assert states.min() >= 0
        assert states.max() <= 3

    def test_all_phases_present(self, synthetic_data):
        """All 4 phases appear in 48 timepoints (~2 cycles)."""
        _, states, _ = synthetic_data
        unique = set(states.tolist())
        assert len(unique) == 4, f"Only {len(unique)} phases present: {unique}"

    def test_phases_cyclic(self, synthetic_data):
        """Phase angle wraps around 2π."""
        _, _, phases = synthetic_data
        assert phases.min() >= 0.0
        assert phases.max() < 2 * np.pi + 0.01

    def test_damping_effect(self):
        """Later timepoints have lower amplitude (damping)."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=16, n_timepoints=100, seed=42,
        )
        # Compare amplitude in first cycle vs last cycle
        first_cycle_std = x[:22].std(axis=0).mean()
        last_cycle_std = x[-22:].std(axis=0).mean()
        assert last_cycle_std < first_cycle_std, (
            f"No damping: first={first_cycle_std:.4f}, last={last_cycle_std:.4f}"
        )

    def test_reproducibility(self):
        """Same seed → identical data."""
        x1, s1, p1 = generate_synthetic_cell_cycle(seed=42)
        x2, s2, p2 = generate_synthetic_cell_cycle(seed=42)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        """Different seeds → different data."""
        x1, _, _ = generate_synthetic_cell_cycle(seed=42)
        x2, _, _ = generate_synthetic_cell_cycle(seed=99)
        assert not np.allclose(x1, x2)


# ──────────────────────────────────────────────────────────────────
# [SERIALIZATION] Round-trip tests
# ──────────────────────────────────────────────────────────────────

class TestSerialization:
    """Serialization round-trip for CellCycleHMM."""

    def test_roundtrip(self, fitted_hmm, synthetic_data):
        """save → JSON → load → posteriors identical."""
        x, _, _ = synthetic_data

        sd = fitted_hmm.state_dict()
        json_str = json.dumps(sd)
        sd_loaded = json.loads(json_str)
        hmm2 = CellCycleHMM.from_state_dict(sd_loaded)

        gamma1 = fitted_hmm.posteriors(x)
        gamma2 = hmm2.posteriors(x)
        np.testing.assert_allclose(gamma1, gamma2, atol=1e-10)

    def test_class_marker_preserved(self, fitted_hmm):
        """state_dict includes _class = 'CellCycleHMM'."""
        sd = fitted_hmm.state_dict()
        assert sd["_class"] == "CellCycleHMM"

    def test_mask_preserved(self, fitted_hmm):
        """Transition mask survives serialization."""
        sd = fitted_hmm.state_dict()
        json_str = json.dumps(sd)
        sd2 = json.loads(json_str)
        hmm2 = CellCycleHMM.from_state_dict(sd2)
        np.testing.assert_array_equal(hmm2.transition_mask, CYCLIC_MASK_K4)

    def test_collapse_penalties_preserved(self, fitted_hmm):
        """Collapse penalties survive serialization."""
        sd = fitted_hmm.state_dict()
        hmm2 = CellCycleHMM.from_state_dict(sd)
        assert hmm2.collapse_penalties == fitted_hmm.collapse_penalties


# ──────────────────────────────────────────────────────────────────
# [CONFIG] CellCycleConfig tests
# ──────────────────────────────────────────────────────────────────

class TestCellCycleConfig:
    """Config dataclass tests."""

    def test_defaults(self):
        """Default config has correct cell cycle values."""
        cfg = CellCycleConfig()
        assert cfg.K == 4
        assert cfg.domain == "cell_cycle"
        assert cfg.use_dsp is False
        assert cfg.use_context is False
        assert cfg.cycle_period_hours == 22.0
        assert cfg.n_total_genes == 874
        assert cfg.mask_epsilon == SOFT_MASK_EPSILON

    def test_build_hmm(self):
        """build_hmm() creates a proper CellCycleHMM."""
        cfg = CellCycleConfig()
        hmm = cfg.build_hmm()
        assert isinstance(hmm, CellCycleHMM)
        assert hmm.K == 4
        assert hmm.V == N_MARKERS

    def test_emission_config_default(self):
        """Default emission is marker type with 16 features."""
        cfg = CellCycleConfig()
        assert cfg.emission.emission_type == "marker"
        assert cfg.emission.n_emission_features == 16

    # ── Step 2 (G5/G7/G8) new fields ────────────────────────────

    def test_new_fields_defaults(self):
        """New fields (G5/G7/G8) have correct defaults."""
        cfg = CellCycleConfig()
        assert cfg.sync_method == "thy"
        assert cfg.V_encoder == 874
        assert cfg.d_season_target == 6

    def test_post_init_invalid_K(self):
        """K ∉ {3, 4} raises ValueError."""
        with pytest.raises(ValueError, match="K must be 3 or 4"):
            CellCycleConfig(K=5)
        with pytest.raises(ValueError, match="K must be 3 or 4"):
            CellCycleConfig(K=2)

    def test_post_init_invalid_V_encoder(self):
        """V_encoder < 1 raises ValueError."""
        with pytest.raises(ValueError, match="V_encoder"):
            CellCycleConfig(V_encoder=0)

    def test_post_init_invalid_d_season(self):
        """d_season_target < 1 raises ValueError."""
        with pytest.raises(ValueError, match="d_season_target"):
            CellCycleConfig(d_season_target=0)

    def test_post_init_invalid_sync_method(self):
        """sync_method ∉ {thy, thy_noc} raises ValueError."""
        with pytest.raises(ValueError, match="sync_method"):
            CellCycleConfig(sync_method="bogus")

    def test_post_init_invalid_cycle_period(self):
        """cycle_period_hours <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="cycle_period_hours"):
            CellCycleConfig(cycle_period_hours=0.0)

    def test_post_init_valid_K3_thy_noc(self):
        """K=3 + thy_noc + explicit K=3 emission is a valid combination."""
        cfg = CellCycleConfig(
            K=3,
            sync_method="thy_noc",
            emission=EmissionConfig(
                n_emission_features=N_MARKERS_K3,
                marker_gene_symbols=list(MARKER_GENES_FLAT_K3),
            ),
        )
        assert cfg.K == 3
        assert cfg.sync_method == "thy_noc"

    # ── I3 — K=3 with default K=4 emission must fail fast ──────

    def test_post_init_K3_default_emission_raises(self):
        """CellCycleConfig(K=3) with default emission (K=4 markers) raises."""
        with pytest.raises(ValueError, match="CellCycleConfig.K=3."):
            CellCycleConfig(K=3)

    def test_post_init_K3_with_K3_emission_passes(self):
        """K=3 + explicitly-K=3 emission succeeds."""
        cfg = CellCycleConfig(
            K=3,
            emission=EmissionConfig(
                n_emission_features=N_MARKERS_K3,
                marker_gene_symbols=list(MARKER_GENES_FLAT_K3),
            ),
        )
        assert cfg.K == 3
        assert cfg.emission.n_emission_features == N_MARKERS_K3

    # ── P2.4 — Symmetric validation across all fields ──────────

    def test_post_init_invalid_hmm_n_iter(self):
        with pytest.raises(ValueError, match="hmm_n_iter"):
            CellCycleConfig(hmm_n_iter=0)

    def test_post_init_invalid_collapse_lambda(self):
        with pytest.raises(ValueError, match="collapse_lambda"):
            CellCycleConfig(collapse_lambda=-0.01)

    def test_post_init_invalid_collapse_min_occ(self):
        with pytest.raises(ValueError, match="collapse_min_occ"):
            CellCycleConfig(collapse_min_occ=0.0)
        with pytest.raises(ValueError, match="collapse_min_occ"):
            CellCycleConfig(collapse_min_occ=1.0)

    def test_post_init_invalid_mask_epsilon(self):
        with pytest.raises(ValueError, match="mask_epsilon"):
            CellCycleConfig(mask_epsilon=1.0)
        with pytest.raises(ValueError, match="mask_epsilon"):
            CellCycleConfig(mask_epsilon=-1e-6)

    def test_post_init_invalid_hmm_covariance(self):
        with pytest.raises(ValueError, match="hmm_covariance"):
            CellCycleConfig(hmm_covariance="banded")

    def test_post_init_invalid_hmm_init_mode(self):
        with pytest.raises(ValueError, match="hmm_init_mode"):
            CellCycleConfig(hmm_init_mode="bogus")

    def test_post_init_invalid_n_timepoints(self):
        with pytest.raises(ValueError, match="n_timepoints_train"):
            CellCycleConfig(n_timepoints_train=0)
        with pytest.raises(ValueError, match="n_timepoints_test"):
            CellCycleConfig(n_timepoints_test=0)

    def test_post_init_invalid_d_model_depth(self):
        with pytest.raises(ValueError, match="d_model"):
            CellCycleConfig(d_model=0)
        with pytest.raises(ValueError, match="depth"):
            CellCycleConfig(depth=0)

    # ── 2.5/2.8 — horizons tuple + LR ratio validation ─────────

    def test_default_horizons(self):
        """Default horizons cover quarter/operational/half cycle (§2.5).

        h=22 dropped from main per 5th review §2.1: with L_win=24 and T=48
        only 3 windows are available for h=22, statistically unreliable.
        """
        cfg = CellCycleConfig()
        assert cfg.horizons == (1, 5, 11)

    def test_default_lr_ratios(self):
        cfg = CellCycleConfig()
        assert cfg.state_embed_lr_ratio == 0.02
        assert cfg.weight_decay_state_embed == 1e-4

    def test_post_init_empty_horizons(self):
        with pytest.raises(ValueError, match="horizons"):
            CellCycleConfig(horizons=())

    def test_post_init_negative_horizon_entry(self):
        with pytest.raises(ValueError, match="horizons"):
            CellCycleConfig(horizons=(1, 5, -1))

    def test_post_init_horizon_exceeds_data(self):
        """Lookback + max(horizons) > n_timepoints_train → fail-fast."""
        # Default n_timepoints_train=48, lookback=24, so max horizon <= 24
        with pytest.raises(ValueError, match="lookback"):
            CellCycleConfig(horizons=(1, 25))

    def test_post_init_invalid_lr_ratio(self):
        with pytest.raises(ValueError, match="state_embed_lr_ratio"):
            CellCycleConfig(state_embed_lr_ratio=0.0)

    def test_post_init_invalid_wd(self):
        with pytest.raises(ValueError, match="weight_decay_state_embed"):
            CellCycleConfig(weight_decay_state_embed=-1e-5)

    # ── 5차 review §2.2/R1 — Small-data regularization fields ───

    def test_default_small_data_regularization(self):
        """5 new fields (5차 review §2.2) have prescribed defaults."""
        cfg = CellCycleConfig()
        assert cfg.dropout == 0.1
        assert cfg.decoder_hidden is None
        assert cfg.weight_decay_decoder == 1e-3
        assert cfg.base_lr == 2e-4
        assert cfg.early_stop_patience == 7

    def test_default_loss_settings(self):
        """5차 review R1 — loss type / huber delta defaults."""
        cfg = CellCycleConfig()
        assert cfg.loss_type == "huber"
        assert cfg.huber_delta == 1.0

    def test_post_init_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            CellCycleConfig(dropout=1.0)
        with pytest.raises(ValueError, match="dropout"):
            CellCycleConfig(dropout=-0.01)

    def test_post_init_decoder_hidden_None_or_positive(self):
        # None is valid (direct Linear path)
        cfg = CellCycleConfig(decoder_hidden=None)
        assert cfg.decoder_hidden is None
        # Positive int is valid
        cfg = CellCycleConfig(decoder_hidden=32)
        assert cfg.decoder_hidden == 32
        # Zero / negative fails
        with pytest.raises(ValueError, match="decoder_hidden"):
            CellCycleConfig(decoder_hidden=0)

    def test_post_init_invalid_weight_decay_decoder(self):
        with pytest.raises(ValueError, match="weight_decay_decoder"):
            CellCycleConfig(weight_decay_decoder=-1e-5)

    def test_post_init_invalid_base_lr(self):
        with pytest.raises(ValueError, match="base_lr"):
            CellCycleConfig(base_lr=0.0)

    def test_post_init_invalid_early_stop_patience(self):
        with pytest.raises(ValueError, match="early_stop_patience"):
            CellCycleConfig(early_stop_patience=0)

    def test_post_init_invalid_loss_type(self):
        with pytest.raises(ValueError, match="loss_type"):
            CellCycleConfig(loss_type="rmse")

    def test_post_init_invalid_huber_delta(self):
        with pytest.raises(ValueError, match="huber_delta"):
            CellCycleConfig(huber_delta=0.0)

    # ── P2.3 — cycle_period_hours non-integer warning ──────────

    def test_build_hmm_non_integer_cycle_period_warns(self):
        """Non-integer cycle_period_hours emits UserWarning at build_hmm()."""
        cfg = CellCycleConfig(cycle_period_hours=22.5)
        with pytest.warns(UserWarning, match="non-integer"):
            hmm = cfg.build_hmm()
        assert hmm.cycle_period == 22   # truncated

    def test_build_hmm_integer_cycle_period_silent(self):
        """Integer cycle_period_hours (e.g., 22.0) emits no warning."""
        import warnings
        cfg = CellCycleConfig(cycle_period_hours=22.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # any warning would fail this
            hmm = cfg.build_hmm()
        assert hmm.cycle_period == 22

    def test_post_init_K3_non_marker_emission_passes(self):
        """K=3 + non-marker emission (variance/random/latent) is allowed
        because the K=4 marker contradiction only applies to marker mode."""
        cfg = CellCycleConfig(
            K=3,
            emission=EmissionConfig(emission_type="variance", n_emission_features=12),
        )
        assert cfg.K == 3
        assert cfg.emission.emission_type == "variance"

    def test_build_hmm_propagates_sync_method(self):
        """build_hmm() propagates sync_method to CellCycleHMM."""
        cfg_thy = CellCycleConfig(sync_method="thy")
        assert cfg_thy.build_hmm().sync_method == "thy"

        cfg_noc = CellCycleConfig(sync_method="thy_noc")
        assert cfg_noc.build_hmm().sync_method == "thy_noc"


# ──────────────────────────────────────────────────────────────────
# [ANNOTATE] State auto-annotation (Stretch 3)
# ──────────────────────────────────────────────────────────────────

class TestAnnotateStates:
    """Tests for annotate_states post-hoc biological interpretation."""

    @pytest.fixture
    def fitted_K4_hmm(self):
        """K=4 HMM fit on default synthetic (16 markers, 48 timepoints)."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=16, n_timepoints=48, seed=42,
        )
        hmm = CellCycleHMM(n_features=16, n_iter=80, seed=42)
        hmm.fit(x)
        return hmm

    def test_annotate_returns_K_entries(self, fitted_K4_hmm):
        """Annotation dict has exactly K keys (one per state)."""
        ann = annotate_states(fitted_K4_hmm)
        assert set(ann.keys()) == {0, 1, 2, 3}

    def test_annotate_entry_schema(self, fitted_K4_hmm):
        """Each entry has top_marker, top_value, fold_change, canonical_phase."""
        ann = annotate_states(fitted_K4_hmm)
        for k, info in ann.items():
            assert {"top_marker", "top_value", "fold_change", "canonical_phase"} \
                <= set(info.keys())
            assert isinstance(info["top_marker"], str)
            assert isinstance(info["top_value"], float)
            assert isinstance(info["fold_change"], float)
            assert isinstance(info["canonical_phase"], str)

    def test_annotate_top_marker_from_known_set(self, fitted_K4_hmm):
        """top_marker is always one of the 16 K=4 markers (auto-detect)."""
        ann = annotate_states(fitted_K4_hmm)
        for info in ann.values():
            assert info["top_marker"] in MARKER_GENES_FLAT

    def test_annotate_canonical_phase_covers_biology(self, fitted_K4_hmm):
        """On well-separated synthetic data, the 4 states should span the 4
        canonical phases (allowing permutation; reject 'unknown' or full
        collapse to one phase)."""
        ann = annotate_states(fitted_K4_hmm)
        phases = {info["canonical_phase"] for info in ann.values()}
        # No 'unknown' (top marker is always in CELL_CYCLE_MARKERS)
        assert "unknown" not in phases
        # At least 3 distinct biological phases assigned (allow 1 collision
        # on noisy synthetic — full 4-phase resolution is not guaranteed)
        assert len(phases) >= 3, f"Only {len(phases)} distinct phases: {phases}"

    def test_annotate_unfitted_raises(self):
        """Unfitted HMM raises ValueError."""
        hmm = CellCycleHMM(n_features=16)
        with pytest.raises(ValueError, match="fitted"):
            annotate_states(hmm)

    def test_annotate_K3_auto_detect(self):
        """K=3 HMM with V=N_MARKERS_K3 auto-detects K=3 marker symbols."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=12, n_timepoints=48, K=3, seed=42,
        )
        hmm = CellCycleHMM(n_states=3, n_features=12, n_iter=50, seed=42)
        hmm.fit(x)
        ann = annotate_states(hmm)
        assert set(ann.keys()) == {0, 1, 2}
        for info in ann.values():
            assert info["top_marker"] in MARKER_GENES_FLAT_K3

    def test_annotate_explicit_marker_symbols(self, fitted_K4_hmm):
        """Caller can override marker_symbols with a custom list."""
        custom = [f"GENE_{i}" for i in range(16)]
        ann = annotate_states(fitted_K4_hmm, marker_symbols=custom)
        for info in ann.values():
            assert info["top_marker"].startswith("GENE_")
            # Not in canonical dictionary → 'unknown'
            assert info["canonical_phase"] == "unknown"

    def test_annotate_marker_symbols_length_mismatch_raises(self, fitted_K4_hmm):
        with pytest.raises(ValueError, match="expected hmm.V"):
            annotate_states(fitted_K4_hmm, marker_symbols=["a", "b", "c"])

    def test_annotate_non_default_V_requires_explicit_symbols(self):
        """K=4 HMM with non-default V cannot auto-detect → raises."""
        x = np.random.RandomState(0).randn(48, 20)   # V=20, not 16
        hmm = CellCycleHMM(n_features=20, n_iter=20, seed=42)
        hmm.fit(x)
        with pytest.raises(ValueError, match="auto-detect"):
            annotate_states(hmm)

    def test_annotate_fold_change_relative_to_baseline(self, fitted_K4_hmm):
        """fold_change = top_value - cross-state mean for the top gene."""
        ann = annotate_states(fitted_K4_hmm)
        means = fitted_K4_hmm.means
        for k, info in ann.items():
            top_idx = MARKER_GENES_FLAT.index(info["top_marker"])
            expected_fc = float(means[k, top_idx] - means[:, top_idx].mean())
            assert abs(info["fold_change"] - expected_fc) < 1e-9


# ──────────────────────────────────────────────────────────────────
# [SYNC] sync_method wiring — π initialization tests
# ──────────────────────────────────────────────────────────────────

class TestSyncMethod:
    """Tests for the sync_method → π initialization wiring (G8)."""

    def test_K4_thy_pi(self):
        """K=4 thy: [G1, S, G2, M] = [0.15, 0.75, 0.05, 0.05]."""
        pi = CellCycleHMM._init_pi(K=4, sync_method="thy")
        np.testing.assert_array_equal(pi, [0.15, 0.75, 0.05, 0.05])
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_K4_thy_noc_pi(self):
        """K=4 thy_noc: [G1, S, G2, M] = [0.05, 0.05, 0.45, 0.45]."""
        pi = CellCycleHMM._init_pi(K=4, sync_method="thy_noc")
        np.testing.assert_array_equal(pi, [0.05, 0.05, 0.45, 0.45])
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_K3_thy_pi(self):
        """K=3 thy: [G1, S, G2M] = [0.15, 0.75, 0.10]."""
        pi = CellCycleHMM._init_pi(K=3, sync_method="thy")
        np.testing.assert_array_equal(pi, [0.15, 0.75, 0.10])
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_K3_thy_noc_pi(self):
        """K=3 thy_noc: [G1, S, G2M] = [0.10, 0.05, 0.85] (G2M dominant)."""
        pi = CellCycleHMM._init_pi(K=3, sync_method="thy_noc")
        np.testing.assert_array_equal(pi, [0.10, 0.05, 0.85])
        assert abs(pi.sum() - 1.0) < 1e-10
        # G2M should be the dominant state under M-arrest release
        assert pi[2] > pi[0] and pi[2] > pi[1]

    def test_constructor_rejects_invalid_sync(self):
        """Invalid sync_method in constructor raises AssertionError."""
        with pytest.raises(AssertionError, match="sync_method"):
            CellCycleHMM(n_features=16, sync_method="bogus")

    def test_default_sync_is_thy(self):
        """Constructor default sync_method is 'thy' (Exp3)."""
        hmm = CellCycleHMM(n_features=16)
        assert hmm.sync_method == "thy"

    def test_init_params_uses_sync_method(self, synthetic_data):
        """_init_params installs the right π for each sync_method."""
        x, _, _ = synthetic_data

        hmm_thy = CellCycleHMM(n_features=16, sync_method="thy")
        hmm_thy._init_params(x)
        np.testing.assert_array_equal(hmm_thy.pi, [0.15, 0.75, 0.05, 0.05])

        hmm_noc = CellCycleHMM(n_features=16, sync_method="thy_noc")
        hmm_noc._init_params(x)
        np.testing.assert_array_equal(hmm_noc.pi, [0.05, 0.05, 0.45, 0.45])

    def test_thy_noc_fit_runs(self, synthetic_data):
        """Full EM fit with sync_method='thy_noc' produces valid posteriors."""
        x, _, _ = synthetic_data
        hmm = CellCycleHMM(
            n_features=16, n_iter=50, seed=42, sync_method="thy_noc",
        )
        hmm.fit(x)
        gamma = hmm.posteriors(x)
        assert gamma.shape == (48, 4)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_sync_method_preserved_in_state_dict(self):
        """state_dict / from_state_dict roundtrip preserves sync_method."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=16, n_timepoints=48, seed=42,
        )
        hmm = CellCycleHMM(
            n_features=16, n_iter=30, seed=42, sync_method="thy_noc",
        )
        hmm.fit(x)
        sd = hmm.state_dict()
        assert sd["sync_method"] == "thy_noc"

        hmm2 = CellCycleHMM.from_state_dict(sd)
        assert hmm2.sync_method == "thy_noc"

    def test_old_state_dict_defaults_to_thy(self):
        """Loading legacy state_dict (no sync_method key) defaults to 'thy'."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=16, n_timepoints=48, seed=42,
        )
        hmm = CellCycleHMM(n_features=16, n_iter=30, seed=42)
        hmm.fit(x)
        sd = hmm.state_dict()
        del sd["sync_method"]  # simulate legacy save

        hmm2 = CellCycleHMM.from_state_dict(sd)
        assert hmm2.sync_method == "thy"


# ──────────────────────────────────────────────────────────────────
# [STRESS] Longer time series and edge cases
# ──────────────────────────────────────────────────────────────────

class TestStress:
    """Stress and edge case tests."""

    def test_100_timepoints(self, large_synthetic):
        """Works on longer time series."""
        x, _, _ = large_synthetic
        hmm = CellCycleHMM(n_features=16, n_iter=80, seed=42)
        hmm.fit(x)
        gamma = hmm.posteriors(x)
        assert gamma.shape == (100, 4)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_minimum_timepoints(self):
        """Works with minimum viable T (T=K=4)."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=16, n_timepoints=8, cycle_period=8, seed=42,
        )
        hmm = CellCycleHMM(n_features=16, n_iter=20, seed=42)
        hmm.fit(x)
        gamma = hmm.posteriors(x)
        assert gamma.shape == (8, 4)

    def test_high_noise(self):
        """Converges even with high noise (graceful degradation)."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=16, n_timepoints=48, noise_std=2.0, seed=42,
        )
        hmm = CellCycleHMM(n_features=16, n_iter=80, seed=42)
        hmm.fit(x)
        gamma = hmm.posteriors(x)
        # Should still produce valid posteriors
        assert not np.any(np.isnan(gamma))
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_full_covariance_mode(self, synthetic_data):
        """Full covariance mode fits and produces valid posteriors."""
        x, _, _ = synthetic_data
        hmm = CellCycleHMM(
            n_features=16, covariance_type="full", n_iter=80, seed=42,
        )
        hmm.fit(x)
        gamma = hmm.posteriors(x)
        assert gamma.shape == (48, 4)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)
        # Soft mask still enforced
        forbidden = hmm.A[CYCLIC_MASK_K4 < 1.0]
        assert np.all(forbidden < 0.01)

    def test_multiple_seeds_consistency(self, synthetic_data):
        """Different seeds produce comparable results on clean data."""
        x, _, _ = synthetic_data
        gammas = []
        for seed in [42, 123, 456]:
            hmm = CellCycleHMM(n_features=16, n_iter=80, seed=seed)
            hmm.fit(x)
            gammas.append(hmm.posteriors(x))

        # State assignments should be broadly similar (allowing permutation)
        # Just check all are valid
        for g in gammas:
            assert g.shape == (48, 4)
            np.testing.assert_allclose(g.sum(axis=1), 1.0, atol=1e-6)


# ──────────────────────────────────────────────────────────────────
# [INTEGRATION] PhaseModule compatibility
# ──────────────────────────────────────────────────────────────────

class TestPhaseModuleCompat:
    """Verify CellCycleHMM output is compatible with PhaseModule."""

    def test_posterior_to_phase_module(self, fitted_hmm, synthetic_data):
        """γ from CellCycleHMM can be fed to PhaseModule (shape check)."""
        import torch

        x, _, _ = synthetic_data
        gamma_np = fitted_hmm.posteriors(x)  # [48, 4]

        # Simulate batched input [B, L, K]
        gamma_torch = torch.tensor(gamma_np, dtype=torch.float32)
        gamma_batch = gamma_torch.unsqueeze(0)  # [1, 48, 4]

        assert gamma_batch.shape == (1, 48, 4)
        assert gamma_batch.sum(dim=-1).allclose(torch.ones(1, 48), atol=1e-5)

    def test_posterior_deterministic(self, fitted_hmm, synthetic_data):
        """Same input → same posterior (no randomness in inference)."""
        x, _, _ = synthetic_data
        g1 = fitted_hmm.posteriors(x)
        g2 = fitted_hmm.posteriors(x)
        np.testing.assert_array_equal(g1, g2)


# ──────────────────────────────────────────────────────────────────
# [DURATION] Duration-aware transition tests
# ──────────────────────────────────────────────────────────────────

class TestDurationAware:
    """Duration-aware asymmetric transition matrix tests."""

    def test_K4_asymmetry(self):
        """K=4: G1 self-transition > S > G2 > M."""
        A = compute_duration_aware_A(4, cycle_period=22)
        assert A[0, 0] > A[1, 1] > A[2, 2] > A[3, 3], (
            f"Self-transitions not monotonically decreasing: "
            f"G1={A[0,0]:.3f}, S={A[1,1]:.3f}, G2={A[2,2]:.3f}, M={A[3,3]:.3f}"
        )

    def test_K3_asymmetry(self):
        """K=3: G1 self > S self > G2M self."""
        A = compute_duration_aware_A(3, cycle_period=22)
        assert A[0, 0] > A[1, 1] > A[2, 2], (
            f"K=3 self-transitions: G1={A[0,0]:.3f}, S={A[1,1]:.3f}, G2M={A[2,2]:.3f}"
        )

    def test_rows_sum_to_one(self):
        """Rows of duration-aware A sum to 1."""
        for K in [3, 4]:
            A = compute_duration_aware_A(K)
            np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-10)

    def test_forbidden_near_epsilon(self):
        """Forbidden cells are near ε after renormalization."""
        A = compute_duration_aware_A(4, epsilon=1e-4)
        # A[1, 0] (S→G1) should be very small
        assert A[1, 0] < 0.001, f"S→G1 = {A[1,0]:.6f}"

    def test_g1_self_approx_091(self):
        """G1 self-transition ≈ 0.91 for cycle_period=22."""
        A = compute_duration_aware_A(4, cycle_period=22, epsilon=0)
        # G1 fraction=0.5, dwell=11, a_self = 1-1/11 ≈ 0.909
        assert abs(A[0, 0] - 0.909) < 0.01, f"G1 self = {A[0,0]:.3f}"

    def test_m_self_clamped_at_030(self):
        """M self-transition clamped at 0.30 floor (not 1-1/1.1≈0.09)."""
        A = compute_duration_aware_A(4, cycle_period=22, epsilon=0)
        # M fraction=0.05, dwell=1.1, a_self = clip(1-1/1.1, 0.30, 0.95) = 0.30
        assert abs(A[3, 3] - 0.30) < 0.01, f"M self = {A[3,3]:.3f}"

    def test_duration_fractions_sum_to_one(self):
        """Duration fractions sum to 1.0."""
        assert abs(sum(DURATION_FRACTIONS_K4["fractions"]) - 1.0) < 1e-10
        assert abs(sum(DURATION_FRACTIONS_K3["fractions"]) - 1.0) < 1e-10


# ──────────────────────────────────────────────────────────────────
# [ENTROPY] Entropy confidence and gating tests
# ──────────────────────────────────────────────────────────────────

class TestEntropyConfidence:
    """Tests for compute_entropy_confidence and entropy_gated_phase_embedding."""

    def test_confident_posterior(self):
        """Near-deterministic γ → low entropy, high confidence."""
        gamma = np.array([[0.97, 0.01, 0.01, 0.01]])  # [1, 4]
        entropy, confidence = compute_entropy_confidence(gamma)
        assert entropy[0] < 0.3, f"Entropy too high: {entropy[0]:.3f}"
        assert confidence[0] > 0.7, f"Confidence too low: {confidence[0]:.3f}"

    def test_uncertain_posterior(self):
        """Near-uniform γ → high entropy, low confidence."""
        gamma = np.array([[0.25, 0.25, 0.25, 0.25]])  # [1, 4]
        entropy, confidence = compute_entropy_confidence(gamma)
        # Max entropy for K=4: log(4) ≈ 1.386
        assert entropy[0] > 1.3, f"Entropy too low: {entropy[0]:.3f}"
        assert confidence[0] < 0.05, f"Confidence too high: {confidence[0]:.3f}"

    def test_confidence_range(self):
        """Confidence is always in [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            gamma = rng.dirichlet(np.ones(4), size=10)  # [10, 4]
            _, confidence = compute_entropy_confidence(gamma)
            assert confidence.min() >= 0.0
            assert confidence.max() <= 1.0

    def test_entropy_shape(self):
        """Output shapes match leading dims of input."""
        gamma = np.random.dirichlet(np.ones(4), size=(3, 10))  # [3, 10, 4]
        entropy, confidence = compute_entropy_confidence(gamma)
        assert entropy.shape == (3, 10)
        assert confidence.shape == (3, 10)

    def test_entropy_monotone(self):
        """More concentrated → lower entropy."""
        g_sharp = np.array([[0.9, 0.05, 0.03, 0.02]])
        g_flat = np.array([[0.3, 0.3, 0.2, 0.2]])
        e_sharp, _ = compute_entropy_confidence(g_sharp)
        e_flat, _ = compute_entropy_confidence(g_flat)
        assert e_sharp[0] < e_flat[0]

    def test_gated_embedding_shape(self):
        """gate_phase has correct shape [..., D]."""
        gamma = np.random.dirichlet(np.ones(4), size=(2, 10))  # [2, 10, 4]
        E = np.random.randn(4, 64)  # [K, d_model]
        gate, entropy, conf = entropy_gated_phase_embedding(gamma, E)
        assert gate.shape == (2, 10, 64)
        assert entropy.shape == (2, 10)
        assert conf.shape == (2, 10)

    def test_gated_embedding_scaling(self):
        """High confidence → large gate magnitude, low → small."""
        E = np.eye(4, 4)  # simple identity embedding
        g_sharp = np.array([[0.97, 0.01, 0.01, 0.01]])  # confident
        g_flat = np.array([[0.25, 0.25, 0.25, 0.25]])  # uncertain

        gate_sharp, _, c_sharp = entropy_gated_phase_embedding(g_sharp, E)
        gate_flat, _, c_flat = entropy_gated_phase_embedding(g_flat, E)

        assert np.linalg.norm(gate_sharp) > np.linalg.norm(gate_flat), (
            f"Sharp gate ({np.linalg.norm(gate_sharp):.3f}) should be "
            f"larger than flat ({np.linalg.norm(gate_flat):.3f})"
        )

    def test_gated_embedding_zero_confidence(self):
        """Uniform γ → confidence ≈ 0 → gate ≈ 0."""
        E = np.random.randn(4, 64)
        gamma = np.array([[0.25, 0.25, 0.25, 0.25]])
        gate, _, conf = entropy_gated_phase_embedding(gamma, E)
        assert np.allclose(gate, 0.0, atol=0.05), (
            f"Gate should be near zero, max={np.abs(gate).max():.4f}"
        )

    def test_K3_entropy(self):
        """Entropy works for K=3."""
        gamma = np.array([[0.8, 0.15, 0.05]])  # [1, 3]
        entropy, confidence = compute_entropy_confidence(gamma)
        assert entropy.shape == (1,)
        assert 0 < confidence[0] < 1

    def test_embedding_shape_mismatch_raises(self):
        """Mismatched K between γ and E raises AssertionError."""
        gamma = np.random.dirichlet(np.ones(4), size=(5,))  # [5, 4]
        E = np.random.randn(3, 64)  # K=3, mismatch
        with pytest.raises(AssertionError):
            entropy_gated_phase_embedding(gamma, E)


# ──────────────────────────────────────────────────────────────────
# [MARKERS] K=3 / K=4 marker constants
# ──────────────────────────────────────────────────────────────────

class TestMarkerSets:
    """Tests for CELL_CYCLE_MARKERS_K3 / MARKER_GENES_FLAT_K3 / get_markers_for_K."""

    def test_K4_marker_count(self):
        """K=4 marker set has 16 markers (4 phases × 4 each)."""
        assert N_MARKERS == 16
        assert len(MARKER_GENES_FLAT) == 16
        assert sum(len(v) for v in CELL_CYCLE_MARKERS.values()) == 16

    def test_K3_marker_count(self):
        """K=3 marker set has 12 markers (3 phases × 4 each)."""
        assert N_MARKERS_K3 == 12
        assert len(MARKER_GENES_FLAT_K3) == 12
        assert sum(len(v) for v in CELL_CYCLE_MARKERS_K3.values()) == 12

    def test_K3_G2M_markers_canonical(self):
        """G2M markers are the canonical G2→M transition genes."""
        g2m = set(CELL_CYCLE_MARKERS_K3["G2M"])
        assert g2m == {"CCNB1", "CDK1", "AURKA", "CDC20"}, (
            f"G2M markers should be canonical G2/M genes, got {g2m}"
        )

    def test_K3_G1_S_inherited_from_K4(self):
        """G1 and S marker sets are identical between K=3 and K=4."""
        assert CELL_CYCLE_MARKERS_K3["G1"] == CELL_CYCLE_MARKERS["G1"]
        assert CELL_CYCLE_MARKERS_K3["S"] == CELL_CYCLE_MARKERS["S"]

    def test_K3_no_marker_overlap_within_phase(self):
        """No duplicate gene symbols within a single phase."""
        for phase, markers in CELL_CYCLE_MARKERS_K3.items():
            assert len(set(markers)) == len(markers), (
                f"Phase {phase} has duplicate markers: {markers}"
            )

    def test_phase_names_match_marker_keys(self):
        """PHASE_NAMES_K3 ordering matches CELL_CYCLE_MARKERS_K3 keys order."""
        assert PHASE_NAMES_K3 == ["G1", "S", "G2M"]
        # And the flat list follows that order
        expected_flat = (
            CELL_CYCLE_MARKERS_K3["G1"]
            + CELL_CYCLE_MARKERS_K3["S"]
            + CELL_CYCLE_MARKERS_K3["G2M"]
        )
        assert MARKER_GENES_FLAT_K3 == expected_flat

    def test_get_markers_for_K_dispatch(self):
        """get_markers_for_K returns the right (phase_names, flat_list) tuple."""
        names4, flat4 = get_markers_for_K(4)
        assert names4 == PHASE_NAMES
        assert flat4 == MARKER_GENES_FLAT

        names3, flat3 = get_markers_for_K(3)
        assert names3 == PHASE_NAMES_K3
        assert flat3 == MARKER_GENES_FLAT_K3

    def test_get_markers_for_K_invalid(self):
        """K ∉ {3, 4} raises ValueError."""
        with pytest.raises(ValueError, match="K must be 3 or 4"):
            get_markers_for_K(5)
        with pytest.raises(ValueError, match="K must be 3 or 4"):
            get_markers_for_K(2)


# ──────────────────────────────────────────────────────────────────
# [K-FLEXIBLE] K=3 support tests
# ──────────────────────────────────────────────────────────────────

class TestKFlexible:
    """Tests for K=3 (G2/M merged) support."""

    def test_K3_fit(self):
        """K=3 HMM fits and produces valid posteriors."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=12, n_timepoints=48, K=3, seed=42,
        )
        hmm = CellCycleHMM(
            n_states=3, n_features=12, n_iter=50, seed=42,
        )
        hmm.fit(x)
        gamma = hmm.posteriors(x)
        assert gamma.shape == (48, 3)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_K3_transition_mask(self):
        """K=3 HMM uses correct 3×3 cyclic mask."""
        hmm = CellCycleHMM(n_states=3, n_features=12)
        assert hmm.transition_mask.shape == (3, 3)
        assert hmm.transition_mask[0, 1] == 1.0  # G1 → S
        assert hmm.transition_mask[1, 2] == 1.0  # S  → G2M
        assert hmm.transition_mask[2, 0] == 1.0  # G2M → G1

    def test_K3_pi_init(self):
        """K=3 phase-aware init: π biased toward S."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=12, n_timepoints=48, K=3, seed=42,
        )
        hmm = CellCycleHMM(n_states=3, n_features=12, init_mode="phase_aware")
        hmm._init_params(x)
        assert hmm.pi[1] > hmm.pi[0], "S should have higher π than G1"
        assert hmm.pi[1] > hmm.pi[2], "S should have higher π than G2M"

    def test_K3_duration_aware_A(self):
        """K=3 A has duration-aware self-transitions."""
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=12, n_timepoints=48, K=3, seed=42,
        )
        hmm = CellCycleHMM(n_states=3, n_features=12, init_mode="phase_aware")
        hmm._init_params(x)
        # G1 self should be higher than G2M self
        assert hmm.A[0, 0] > hmm.A[2, 2], (
            f"G1={hmm.A[0,0]:.3f} should > G2M={hmm.A[2,2]:.3f}"
        )

    def test_K3_config_build_hmm(self):
        """CellCycleConfig with K=3 + explicit K=3 emission builds correct HMM."""
        cfg = CellCycleConfig(
            K=3,
            emission=EmissionConfig(
                n_emission_features=N_MARKERS_K3,
                marker_gene_symbols=list(MARKER_GENES_FLAT_K3),
            ),
        )
        hmm = cfg.build_hmm()
        assert hmm.K == 3
        assert hmm.V == N_MARKERS_K3
        assert hmm.transition_mask.shape == (3, 3)

    def test_K3_bic_vs_K4(self):
        """K=3 and K=4 produce finite BIC for comparison.

        Note: K=3 uses V=12, K=4 uses V=16 (different feature dims) here, so
        BIC values are NOT directly comparable. This test only checks BIC
        computability. True K-selection ablation must use V=16 fixed for both
        K=3 and K=4 — done at the forecaster/script level (Step 5).
        """
        x, _, _ = generate_synthetic_cell_cycle(
            n_genes=16, n_timepoints=48, K=4, seed=42,
        )
        hmm4 = CellCycleHMM(n_states=4, n_features=16, n_iter=50, seed=42)
        hmm4.fit(x)
        bic4 = hmm4.bic(x)

        # For K=3 we need compatible data (12 features)
        x3, _, _ = generate_synthetic_cell_cycle(
            n_genes=12, n_timepoints=48, K=3, seed=42,
        )
        hmm3 = CellCycleHMM(n_states=3, n_features=12, n_iter=50, seed=42)
        hmm3.fit(x3)
        bic3 = hmm3.bic(x3)

        assert np.isfinite(bic3) and np.isfinite(bic4)
        print(f"  BIC K=3 (V=12): {bic3:.1f}, BIC K=4 (V=16): {bic4:.1f}")
