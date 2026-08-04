"""CellCycleHMM — Entropy-Gated Cyclic Phase Module (EGCPM) for cell cycle domain.

Extends the base GaussianHMM (gaussian_hmm.py) with cell-cycle-specific priors
and an entropy-based phase confidence mechanism:

  1. **Cyclic soft mask** on transition matrix (G1→S→G2→M→G1 only).
     Encodes the irreversible biochemical ratchet of cell cycle progression.
     Reverse transitions are near-zero (ε=1e-4) rather than hard zero,
     avoiding log(0) while maintaining biological constraint (~1/3000 of
     normal transition probability).

  2. **Duration-aware asymmetric transitions**: Initial self-transition
     probabilities reflect biological phase durations (G1≈50% → a_kk≈0.91,
     M≈5% → a_kk≈0.30). EM refines from this biologically grounded start.

  3. **Marker gene emission** (Option A): HMM emission uses a small subset
     of known phase-marker genes (~16) rather than all 874 periodic genes.
     Posterior γ is then applied to the full gene set via Mamba gating.
     Marker selection follows MSigDB / Whitfield 2002 phase-specific gene sets.

  4. **Entropy-gated phase confidence**: Shannon entropy of posterior γ
     quantifies phase assignment uncertainty. Confidence c(t) = 1 - H/log(K)
     gates the phase embedding: gate_phase = c(t) · (γ @ E).
     Self-regulating: strong gating at stable phases, weak at boundaries.

  5. **Anti-collapse mechanism**: State occupancy monitoring with active rescue.
     If any state's posterior mass drops below threshold, its emission mean
     is interpolated toward the data center to prevent K=4 collapse
     (G2/M are short phases, ~25% combined).

  6. **K-flexible design**: Supports K=3 (G2/M merged) and K=4. BIC-based
     model selection determines optimal K. Duration profiles and masks
     adapt automatically to K.

  7. **Ablation emission variants** for reviewer defense:
     (a) MSigDB markers (primary)
     (b) Top-N by variance (data-driven, no biology)
     (c) Random N genes (3-seed average)
     (d) All genes, latent space (Option C)

Integration with CG-Mamba (EGCPM pipeline):
    1. Offline: CellCycleHMM.fit(x_markers) → EM on marker gene subset
    2. Offline: CellCycleHMM.posteriors(x_markers) → γ [T, K]
    3. Online:  compute_entropy_confidence(γ) → (H, c)
    4. Online:  gate_phase = c · (γ @ E) → Mamba encoder prefix

Reference:
    Whitfield ML et al. (2002) Mol. Biol. Cell 13(6):1977-2000.
    GEO GSE3497, GPL3001 platform. 874 periodically expressed genes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Re-use base HMM infrastructure (EM, forward-backward, Viterbi, serialization)
from src.models.gaussian_hmm import GaussianHMM


# ──────────────────────────────────────────────────────────────────
# Constants — Cell cycle phase marker genes (MSigDB / Whitfield 2002)
# ──────────────────────────────────────────────────────────────────

# Phase-specific marker genes from Whitfield et al. 2002 + MSigDB canonical sets.
# These are gene symbols; actual probe IDs on GPL3001 are resolved at data loading.
# Order: G1 markers, S markers, G2 markers, M markers.
CELL_CYCLE_MARKERS: dict[str, list[str]] = {
    "G1": ["CCND1", "CCNE1", "E2F1", "CDK4"],
    "S":  ["PCNA", "RRM2", "MCM3", "RFC4"],
    "G2": ["CCNB1", "TOP2A", "CDK1", "AURKA"],
    "M":  ["CDC20", "BUB1", "PLK1", "MAD2L1"],
}

# K=3 variant: G2 and M merged into a single G2M phase.
# Markers are the canonical G2→M transition genes most commonly cited in
# MSigDB G2M_CHECKPOINT and Whitfield 2002 G2/M cluster: CCNB1 (Cyclin B1
# late-G2 accumulation), CDK1 (G2→M transition kinase), AURKA (Aurora-A,
# late G2 / early M centrosome activator), CDC20 (early-M APC/C activator).
# G1 and S marker sets are identical to the K=4 case.
CELL_CYCLE_MARKERS_K3: dict[str, list[str]] = {
    "G1":  ["CCND1", "CCNE1", "E2F1", "CDK4"],
    "S":   ["PCNA", "RRM2", "MCM3", "RFC4"],
    "G2M": ["CCNB1", "CDK1", "AURKA", "CDC20"],
}

# Flat list (canonical ordering for emission vector)
MARKER_GENES_FLAT: list[str] = []
for _phase in ["G1", "S", "G2", "M"]:
    MARKER_GENES_FLAT.extend(CELL_CYCLE_MARKERS[_phase])
N_MARKERS = len(MARKER_GENES_FLAT)  # 16

MARKER_GENES_FLAT_K3: list[str] = []
for _phase in ["G1", "S", "G2M"]:
    MARKER_GENES_FLAT_K3.extend(CELL_CYCLE_MARKERS_K3[_phase])
N_MARKERS_K3 = len(MARKER_GENES_FLAT_K3)  # 12

# Phase indices
PHASE_NAMES = ["G1", "S", "G2", "M"]
PHASE_NAMES_K3 = ["G1", "S", "G2M"]
K_CELL_CYCLE = 4


def get_markers_for_K(K: int) -> tuple[list[str], list[str]]:
    """Return (phase_names, flat_marker_list) for the requested K.

    Args:
        K: 3 or 4. Other values raise ValueError.

    Returns:
        (phase_names, marker_genes_flat) tuple matching the K-variant marker set.
    """
    if K == 4:
        return PHASE_NAMES, MARKER_GENES_FLAT
    if K == 3:
        return PHASE_NAMES_K3, MARKER_GENES_FLAT_K3
    raise ValueError(f"K must be 3 or 4, got {K}")


# ──────────────────────────────────────────────────────────────────
# Cyclic transition mask (soft mask)
# ──────────────────────────────────────────────────────────────────

# Default soft mask epsilon: near-zero but avoids log(0).
# ε=1e-4 → 1/3000 of normal forward transition (~0.3).
# Biological justification: "virtually impossible, not absolutely impossible."
SOFT_MASK_EPSILON: float = 1e-4


def make_cyclic_mask(K: int = 4, epsilon: float = SOFT_MASK_EPSILON) -> np.ndarray:
    """Soft mask enforcing forward-only cyclic transitions.

    mask[i, j] = 1.0 if transition i→j is allowed (self-loop or forward),
                 epsilon if transition i→j is forbidden (reverse/skip).

    Using epsilon > 0 instead of hard zero:
      - Avoids log(0) = -∞ in forward-backward (numerical stability)
      - Encodes "virtually impossible" rather than "absolutely impossible"
      - ε=1e-4 means forbidden transitions are ~1/3000 of normal forward (0.3)

    Biological basis: cell cycle progression is irreversible due to
    proteolytic degradation of phase-specific cyclins (ratchet mechanism).
    G1→S: Cyclin D degradation. S→G2: origin licensing reset.
    G2→M: Cyclin B accumulation. M→G1: APC/C-mediated Cyclin B destruction.

    Args:
        K:       number of phases (3 or 4)
        epsilon: near-zero floor for forbidden transitions (default: 1e-4)

    Returns:
        mask: [K, K] numpy array with 1.0 (allowed) or epsilon (forbidden)
    """
    assert epsilon >= 0, f"epsilon must be non-negative, got {epsilon}"
    assert K >= 2, f"K must be >= 2, got {K}"
    mask = np.full((K, K), epsilon, dtype=np.float64)
    for i in range(K):
        mask[i, i] = 1.0                    # self-loop
        mask[i, (i + 1) % K] = 1.0          # forward transition
    return mask


# Pre-computed for K=4 and K=3
CYCLIC_MASK_K4 = make_cyclic_mask(4)
CYCLIC_MASK_K3 = make_cyclic_mask(3)


# ──────────────────────────────────────────────────────────────────
# Duration-aware transition profiles
# ──────────────────────────────────────────────────────────────────

# Biological phase duration fractions (approximate, HeLa cells):
#   G1 ≈ 50%, S ≈ 30%, G2 ≈ 15%, M ≈ 5%  (for K=4)
#   G1 ≈ 50%, S ≈ 30%, G2M ≈ 20%          (for K=3, G2/M merged)
# Self-transition a_kk = 1 - 1/(fraction × cycle_length_in_steps)
# For cycle_period=22: G1 self ≈ 1 - 1/(0.5*22) = 0.909, M self ≈ 1 - 1/(0.05*22) = 0.091
# Clamped to [0.30, 0.95] for numerical safety.

DURATION_FRACTIONS_K4: dict[str, list[float]] = {
    "phases": ["G1", "S", "G2", "M"],
    "fractions": [0.50, 0.30, 0.15, 0.05],
}

DURATION_FRACTIONS_K3: dict[str, list[float]] = {
    "phases": ["G1", "S", "G2M"],
    "fractions": [0.50, 0.30, 0.20],
}


def compute_duration_aware_A(
    K: int,
    cycle_period: int = 22,
    epsilon: float = SOFT_MASK_EPSILON,
) -> np.ndarray:
    """Build duration-aware cyclic transition matrix.

    Self-transition probability reflects expected phase dwell time:
        a_kk = 1 - 1 / (fraction_k × cycle_period)
    Forward transition = 1 - a_kk (for allowed transitions).
    Forbidden transitions get epsilon, then rows are renormalized.

    Args:
        K:            number of phases (3 or 4)
        cycle_period: cycle length in timepoints (default: 22)
        epsilon:      floor for forbidden transitions

    Returns:
        A: [K, K] transition matrix (rows sum to 1)
    """
    if K == 4:
        fractions = DURATION_FRACTIONS_K4["fractions"]
    elif K == 3:
        fractions = DURATION_FRACTIONS_K3["fractions"]
    else:
        # Uniform fallback for arbitrary K
        fractions = [1.0 / K] * K

    A = np.full((K, K), epsilon, dtype=np.float64)
    for i in range(K):
        dwell = fractions[i] * cycle_period
        a_self = np.clip(1.0 - 1.0 / max(dwell, 1.0), 0.30, 0.95)
        a_fwd = 1.0 - a_self
        A[i, i] = a_self
        A[i, (i + 1) % K] = a_fwd

    # Renormalize rows (accounts for epsilon in forbidden cells)
    A /= A.sum(axis=1, keepdims=True)
    return A


# ──────────────────────────────────────────────────────────────────
# CellCycleHMM — GaussianHMM with cell-cycle priors
# ──────────────────────────────────────────────────────────────────

class CellCycleHMM(GaussianHMM):
    """Gaussian HMM with cell-cycle-specific structural priors (EGCPM core).

    Key differences from base GaussianHMM:
      - Transition matrix uses soft cyclic mask (ε=1e-4 instead of hard 0)
      - Duration-aware asymmetric self-transitions (G1≈0.91, M≈0.30)
      - K-flexible: supports K=3 (G2/M merged) and K=4
      - Initialization uses phase-aware priors (biological ordering)
      - Anti-collapse regularization during EM M-step
      - Entropy confidence computation for phase gating
      - Emission can operate on marker gene subset (V_marker < V_total)

    Args:
        n_states:           K — number of phases (3 or 4, default 4)
        n_features:         V — observation dimension (16 for marker emission)
        covariance_type:    'diag' recommended for marker emission (16-dim)
        reg_covar:          covariance regularization
        n_iter:             max EM iterations
        tol:                LL convergence tolerance
        seed:               random seed
        transition_mask:    [K, K] soft mask (default: cyclic with ε=1e-4)
        collapse_lambda:    anti-collapse monitoring strength (0 = disabled)
        collapse_min_occ:   minimum state occupancy fraction (below → rescue)
        init_mode:          'phase_aware' (default) or 'random' (base class)
        cycle_period:       approximate cycle length in timepoints (for init segmentation)
        mask_epsilon:       near-zero floor for forbidden transitions (default: 1e-4)
        sync_method:        cell synchronization protocol — controls π initial
                            distribution under 'phase_aware' init.
                            'thy'     = double thymidine (G1/S boundary release,
                                        Exp3 default). t=0 cells are post-G1/S.
                            'thy_noc' = thymidine + nocodazole (M-phase arrest
                                        release, Exp4). t=0 cells are post-M
                                        (G2/M predominant before re-entry).
    """

    _VALID_SYNC_METHODS = ("thy", "thy_noc")

    def __init__(
        self,
        n_states: int = K_CELL_CYCLE,
        n_features: int = N_MARKERS,
        covariance_type: str = "diag",
        reg_covar: float = 1e-3,
        n_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
        transition_mask: np.ndarray | None = None,
        collapse_lambda: float = 0.1,
        collapse_min_occ: float = 0.05,
        init_mode: str = "phase_aware",
        cycle_period: int = 22,
        mask_epsilon: float = SOFT_MASK_EPSILON,
        sync_method: str = "thy",
    ):
        super().__init__(
            n_states=n_states,
            n_features=n_features,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_iter=n_iter,
            tol=tol,
            seed=seed,
        )
        assert sync_method in self._VALID_SYNC_METHODS, (
            f"sync_method must be one of {self._VALID_SYNC_METHODS}, "
            f"got {sync_method!r}"
        )
        self.mask_epsilon = mask_epsilon
        self.transition_mask = (
            transition_mask if transition_mask is not None
            else make_cyclic_mask(self.K, epsilon=mask_epsilon)
        )
        assert self.transition_mask.shape == (self.K, self.K), (
            f"Transition mask shape {self.transition_mask.shape} != ({self.K}, {self.K})"
        )
        self.collapse_lambda = collapse_lambda
        self.collapse_min_occ = collapse_min_occ
        self.init_mode = init_mode
        self.cycle_period = cycle_period
        self.sync_method = sync_method

        # Tracking
        self.collapse_penalties: list[float] = []

    # ── Phase-aware initialization ────────────────────────────────

    @staticmethod
    def _init_pi(K: int, sync_method: str) -> np.ndarray:
        """Build initial distribution π based on synchronization method.

        Biological basis:
          - Double thymidine arrest releases cells at the G1/S boundary,
            so at t=0 most cells have just entered S (Exp3 protocol).
          - Thymidine + nocodazole arrests cells in mitosis (M); release
            re-enters G1, but at t=0 the population is still G2/M-dominant
            with Cyclin B accumulation (Exp4 protocol).

        Args:
            K:           3 or 4 phases.
            sync_method: 'thy' or 'thy_noc'.

        Returns:
            pi: [K] initial distribution, sums to 1.

        Raises:
            ValueError: if (K, sync_method) is not a supported combination.
        """
        if K == 4 and sync_method == "thy":
            return np.array([0.15, 0.75, 0.05, 0.05])
        if K == 4 and sync_method == "thy_noc":
            return np.array([0.05, 0.05, 0.45, 0.45])
        if K == 3 and sync_method == "thy":
            return np.array([0.15, 0.75, 0.10])
        if K == 3 and sync_method == "thy_noc":
            # G2M dominant; remaining mass split between G1 (post-mitotic entry)
            # and a small S residual.
            return np.array([0.10, 0.05, 0.85])
        # Arbitrary K fallback (sync_method ignored): uniform.
        if K not in (3, 4):
            return np.ones(K) / K
        raise ValueError(
            f"Unsupported (K={K}, sync_method={sync_method!r}) combination"
        )

    def _init_params(self, x: np.ndarray) -> None:
        """Phase-aware parameter initialization with duration-aware transitions.

        For 'phase_aware' mode:
          - π biased per sync_method:
              'thy'     → G1/S boundary release (Exp3, t=0 cells in S)
              'thy_noc' → M-phase arrest release (Exp4, t=0 cells in G2/M)
          - A initialized with duration-aware asymmetric self-transitions
            (G1≈0.91, S≈0.85, G2≈0.70, M≈0.30 for K=4)
          - μ_k initialized by splitting data into K temporal segments
            (assumes roughly cyclic ordering in synchronized time course)
          - Σ from segment variance (raw, no reg_covar — `_log_emission` adds
            it as the single source of truth; cf. base class H-3 fix)

        For 'random' mode: falls back to base class + soft mask.
        """
        if self.init_mode == "random":
            super()._init_params(x)
            self._apply_mask_to_A()
            return

        rng = np.random.RandomState(self.seed)
        T, V = x.shape
        K = self.K

        # π: synchronization-method-dependent initial distribution.
        # State ordering convention:
        #   K=4: 0=G1, 1=S, 2=G2, 3=M
        #   K=3: 0=G1, 1=S, 2=G2M (G2/M merged)
        self.pi = self._init_pi(K, self.sync_method)

        # A: duration-aware cyclic forward-only transitions.
        # Self-transition reflects biological dwell time per phase.
        self.A = compute_duration_aware_A(
            K, cycle_period=self.cycle_period, epsilon=self.mask_epsilon,
        )

        # μ_k: split time course into K segments (phase-ordered).
        # For a ~22h cycle with 48 timepoints (~2 cycles):
        #   Segment 0 (G1): t=0..5, t=22..27
        #   Segment 1 (S):  t=5..11, t=27..33
        # This is approximate — EM will refine.
        cycle_len = min(T, self.cycle_period)
        self.means = np.zeros((K, V))
        self.covars = (
            np.zeros((K, V, V)) if self.covariance_type == "full"
            else np.zeros((K, V))
        )

        for k in range(K):
            # Segment for phase k: [k/K * cycle, (k+1)/K * cycle)
            start = int(k * cycle_len / K)
            end = int((k + 1) * cycle_len / K)
            if end <= start:
                end = start + 1

            # Collect all time points belonging to this segment
            # (wrapping for multi-cycle data)
            indices = []
            for c in range(0, T, cycle_len):
                for t in range(c + start, min(c + end, T)):
                    indices.append(t)

            if len(indices) == 0:
                # Fallback: random data point
                indices = [rng.randint(0, T)]

            segment = x[indices]
            self.means[k] = segment.mean(axis=0)

            # H-3 fix (base class convention): init covar is RAW data variance.
            # _log_emission adds reg_covar as the single source of regularization.
            # Adding reg_covar here would cause iter-0 emission to use 2·reg_covar.
            data_var = np.var(segment, axis=0)
            if self.covariance_type == "full":
                self.covars[k] = np.diag(data_var)
            else:
                self.covars[k] = data_var

        # Add scale-adaptive noise to break symmetry if segments overlap.
        # Noise magnitude = 5% of per-feature std across full data.
        data_std = np.std(x, axis=0, keepdims=False)  # [V]
        noise_scale = np.maximum(data_std * 0.05, 1e-4)
        self.means += rng.randn(K, V) * noise_scale[None, :]

    def _apply_mask_to_A(self) -> None:
        """Apply soft mask to A: clamp forbidden transitions to ε, renormalize.

        Soft mask strategy:
          - Where mask[i,j] = 1.0 (allowed): A[i,j] keeps its EM-estimated value
          - Where mask[i,j] = ε (forbidden): A[i,j] is clamped to ε
          - Rows are renormalized to sum to 1

        This replaces the old hard mask (multiply by 0/1). The soft approach
        avoids log(0) = -∞ while maintaining the cyclic constraint.
        """
        for i in range(self.K):
            for j in range(self.K):
                if self.transition_mask[i, j] < 1.0:
                    # Forbidden transition: clamp to epsilon
                    self.A[i, j] = self.transition_mask[i, j]
        # Renormalize rows
        row_sums = self.A.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        self.A /= row_sums

    # ── Override M-step to enforce mask + anti-collapse ───────────

    def _m_step(self, x: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """M-step with soft cyclic mask enforcement and anti-collapse rescue.

        1. Standard M-step (base class updates π, A, μ, Σ)
        2. Apply soft mask to A (clamp forbidden transitions to ε, renormalize)
        3. Anti-collapse monitoring: track -log(occupancy) for diagnostics
        4. Active rescue: if state occupancy < threshold/2, gently interpolate
           its mean toward data center (not a true EM penalty — a heuristic
           that gives near-dead states a second chance to attract posterior mass)

        Note: The collapse_penalty is a monitoring metric (logged per iteration),
        NOT a term in the EM objective. True constrained-EM would modify the
        E-step posterior, but that risks destabilizing convergence on T=48 data.
        The rescue heuristic is simpler and empirically sufficient.
        """
        # Standard M-step
        super()._m_step(x, gamma, xi)

        # Soft mask enforcement on A
        self._apply_mask_to_A()

        # Anti-collapse monitoring + rescue
        if self.collapse_lambda > 0:
            occupancy = gamma.mean(axis=0)  # [K] — mean posterior per state
            penalty = 0.0
            for k in range(self.K):
                if occupancy[k] < self.collapse_min_occ:
                    penalty += -np.log(max(occupancy[k], 1e-10))
            self.collapse_penalties.append(penalty)

            # Active rescue: if a state is near-dead (occupancy < threshold/2),
            # interpolate its mean 50% toward the data center + small noise.
            # Gentler than full replacement — preserves partial information.
            # Per-state seed (iter*K + k) ensures distinct noise vectors when
            # multiple states collapse in the same iteration → better symmetry
            # breaking than a single per-iteration seed.
            rescue_threshold = self.collapse_min_occ * 0.5
            data_mean = x.mean(axis=0)
            iter_idx = len(self.collapse_penalties)
            for k in range(self.K):
                if occupancy[k] < rescue_threshold:
                    rng = np.random.RandomState(
                        self.seed + iter_idx * self.K + k
                    )
                    noise = rng.randn(self.V) * np.std(x, axis=0) * 0.05
                    # 50% interpolation: keep some of current mean, pull toward center
                    self.means[k] = 0.5 * self.means[k] + 0.5 * data_mean + noise

    # ── Override fit to track collapse penalties ──────────────────

    def fit(self, x: np.ndarray) -> "CellCycleHMM":
        """Baum-Welch EM with cyclic constraints.

        Identical to base fit() but initializes collapse tracking
        and applies mask throughout EM iterations.
        """
        self.collapse_penalties = []
        return super().fit(x)

    # ── Override _n_free_params (fewer due to mask) ───────────────

    def _n_free_params(self) -> int:
        """Number of free parameters (accounts for soft-masked transitions).

        Soft-masked (ε) transitions are effectively fixed — not free params.
        Only transitions with mask value = 1.0 are considered "allowed".
        Each row has (n_allowed - 1) free params (sum-to-1 constraint).
        """
        K, V = self.K, self.V
        n_pi = K - 1

        # Transition: count fully allowed transitions (mask == 1.0) per row
        n_A = 0
        for i in range(K):
            n_allowed = int((self.transition_mask[i] >= 1.0).sum())
            n_A += max(n_allowed - 1, 0)

        n_mu = K * V
        if self.covariance_type == "full":
            n_cov = K * V * (V + 1) // 2
        else:
            n_cov = K * V
        return n_pi + n_A + n_mu + n_cov

    # ── Serialization (extend base) ──────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        """Serialize with cell-cycle-specific fields."""
        d = super().state_dict()
        d["transition_mask"] = self.transition_mask.tolist()
        d["collapse_lambda"] = self.collapse_lambda
        d["collapse_min_occ"] = self.collapse_min_occ
        d["collapse_penalties"] = self.collapse_penalties
        d["init_mode"] = self.init_mode
        d["cycle_period"] = self.cycle_period
        d["mask_epsilon"] = self.mask_epsilon
        d["sync_method"] = self.sync_method
        d["_class"] = "CellCycleHMM"
        return d

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> "CellCycleHMM":
        """Reconstruct from serialized dict."""
        hmm = cls(
            n_states=d.get("K", K_CELL_CYCLE),
            n_features=d["V"],
            covariance_type=d["covariance_type"],
            reg_covar=d["reg_covar"],
            seed=d["seed"],
            transition_mask=np.array(d["transition_mask"]),
            collapse_lambda=d.get("collapse_lambda", 0.1),
            collapse_min_occ=d.get("collapse_min_occ", 0.05),
            init_mode=d.get("init_mode", "phase_aware"),
            cycle_period=d.get("cycle_period", 22),
            mask_epsilon=d.get("mask_epsilon", SOFT_MASK_EPSILON),
            sync_method=d.get("sync_method", "thy"),
        )
        hmm.pi = np.array(d["pi"])
        hmm.A = np.array(d["A"])
        hmm.means = np.array(d["means"])
        hmm.covars = np.array(d["covars"])
        hmm.n_iter_run = d["n_iter_run"]
        hmm.ll_history = d.get("ll_history", [])
        hmm.collapse_penalties = d.get("collapse_penalties", [])
        hmm._fitted = True
        return hmm


# ──────────────────────────────────────────────────────────────────
# State auto-annotation (post-hoc biological interpretation)
# ──────────────────────────────────────────────────────────────────

def annotate_states(
    hmm: "CellCycleHMM",
    marker_symbols: list[str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Auto-annotate each fitted HMM state with its top-expressed marker.

    For each state k, identifies the gene with the highest mean expression
    in μ_k, and maps it to its canonical cell cycle phase (G1, S, G2, or M
    for K=4; G1, S, or G2M for K=3) using the CELL_CYCLE_MARKERS lookup.
    Also reports the *fold change* relative to the cross-state baseline
    (mean of μ across states) — a state with `fold_change >> 0` is one
    where the top marker is significantly elevated.

    This is post-hoc, training-free interpretation: it does not modify the
    HMM and adds zero parameters. Its purpose is reviewer / paper-figure
    support — demonstrating that the *unsupervised* HMM states correspond
    to *biologically meaningful* phases without any explicit phase labels
    during fitting.

    Args:
        hmm:            a fitted CellCycleHMM instance.
        marker_symbols: gene symbol list aligned with columns of hmm.means.
                        If None, defaults are inferred from hmm.K:
                          - K=4, V=N_MARKERS=16  → MARKER_GENES_FLAT
                          - K=3, V=N_MARKERS_K3=12 → MARKER_GENES_FLAT_K3
                        Otherwise, must be provided explicitly.

    Returns:
        dict mapping state_index → dict with keys:
            top_marker (str):       gene symbol with highest mean expression
            top_value (float):      mean expression value
            fold_change (float):    top_value - mean(μ[:, top_idx]) across states
            canonical_phase (str):  phase label from CELL_CYCLE_MARKERS lookup
                                    ("G1", "S", "G2", "M", "G2M") or "unknown"
                                    if the gene is not in the marker dictionary.

    Raises:
        ValueError: if hmm is not fitted, or marker_symbols length ≠ hmm.V,
                    or auto-detection of marker_symbols fails for non-default V.

    Example:
        >>> hmm = CellCycleHMM(n_features=16).fit(x_markers)
        >>> ann = annotate_states(hmm)
        >>> for k, info in ann.items():
        ...     print(f"State {k}: {info['canonical_phase']} "
        ...           f"({info['top_marker']}↑ {info['fold_change']:.2f})")
        State 0: G1 (CCND1↑ 0.87)
        State 1: S  (PCNA↑ 1.34)
        ...
    """
    if not getattr(hmm, "_fitted", False):
        raise ValueError("annotate_states requires a fitted CellCycleHMM")

    # Auto-detect marker symbols if not provided.
    if marker_symbols is None:
        if hmm.K == 4 and hmm.V == N_MARKERS:
            marker_symbols = list(MARKER_GENES_FLAT)
        elif hmm.K == 3 and hmm.V == N_MARKERS_K3:
            marker_symbols = list(MARKER_GENES_FLAT_K3)
        else:
            raise ValueError(
                f"Cannot auto-detect marker symbols for K={hmm.K}, V={hmm.V}. "
                f"Pass `marker_symbols` explicitly (list of {hmm.V} gene names)."
            )

    if len(marker_symbols) != hmm.V:
        raise ValueError(
            f"marker_symbols has {len(marker_symbols)} entries, "
            f"expected hmm.V = {hmm.V}"
        )

    # Build gene symbol → canonical phase lookup matching this K.
    markers_dict = CELL_CYCLE_MARKERS if hmm.K == 4 else CELL_CYCLE_MARKERS_K3
    symbol_to_phase: dict[str, str] = {}
    for phase, syms in markers_dict.items():
        for s in syms:
            symbol_to_phase[s] = phase

    # Per-gene baseline mean across all K states (for fold-change reporting).
    baseline_per_gene = hmm.means.mean(axis=0)   # [V]

    annotations: dict[int, dict[str, Any]] = {}
    for k in range(hmm.K):
        top_idx = int(np.argmax(hmm.means[k]))
        top_symbol = marker_symbols[top_idx]
        top_value = float(hmm.means[k, top_idx])
        fold_change = top_value - float(baseline_per_gene[top_idx])
        canonical = symbol_to_phase.get(top_symbol, "unknown")
        annotations[k] = {
            "top_marker": top_symbol,
            "top_value": top_value,
            "fold_change": fold_change,
            "canonical_phase": canonical,
        }
    return annotations


# ──────────────────────────────────────────────────────────────────
# Entropy-Gated Phase Confidence (EGCPM core)
# ──────────────────────────────────────────────────────────────────

def compute_entropy_confidence(
    gamma: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Shannon entropy and normalized confidence from posterior γ.

    Entropy H(t) = -Σ_k γ[t,k] log γ[t,k] measures phase assignment
    uncertainty. Confidence c(t) = 1 - H(t)/log(K) normalizes to [0,1]:
      - c ≈ 1: one phase dominates (low entropy, high confidence)
      - c ≈ 0: uniform distribution (max entropy, phase boundary)

    This is the information-theoretic core of EGCPM: it quantifies
    "how much do we know about the current phase?"

    Equivalent KL-divergence formulation
    ────────────────────────────────────
    Let U_K = (1/K, ..., 1/K) be the uniform (maximum-entropy) prior over
    the K phases. Shannon entropy and KL-divergence from U_K satisfy

        D_KL(γ || U_K) = log K - H(γ)

    so the confidence is exactly the divergence from uncertainty, normalized
    by its maximum:

        c(t) = D_KL(γ_t || U_K) / log K  ∈ [0, 1]

    This re-framing motivates c(t) as the *normalized information gain*
    of the HMM posterior relative to an uninformative prior: c = 1 means
    "the data fully resolves phase identity", c = 0 means "the data is
    indistinguishable from the uniform prior". The implementation below
    computes the entropy form for numerical convenience; the two are
    mathematically identical.

    Args:
        gamma: [..., K] posterior probabilities (last dim = K states).
               Supports any leading dims: [T, K], [B, L, K], etc.
        eps:   numerical floor to avoid log(0)

    Returns:
        entropy:    [...] Shannon entropy H(t) in nats
        confidence: [...] normalized confidence c(t) ∈ [0, 1]
    """
    K = gamma.shape[-1]
    assert K >= 2, f"Need K >= 2 for entropy, got {K}"

    # Clamp for numerical safety
    gamma_safe = np.clip(gamma, eps, 1.0)

    # Shannon entropy: H = -Σ γ_k log γ_k
    entropy = -np.sum(gamma_safe * np.log(gamma_safe), axis=-1)

    # Normalize: c = 1 - H / log(K)
    max_entropy = np.log(K)
    confidence = 1.0 - entropy / max_entropy

    # Clamp confidence to [0, 1] (numerical safety)
    confidence = np.clip(confidence, 0.0, 1.0)

    return entropy, confidence


def entropy_gated_phase_embedding(
    gamma: np.ndarray,
    embedding: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute entropy-gated phase embedding (full EGCPM forward pass).

    gate_phase = c(t) · (γ[t] @ E)

    where c(t) is the entropy-based confidence and E is the phase embedding
    matrix. The gating is self-regulating: strong at stable phases (high c),
    weak at phase boundaries (low c).

    Args:
        gamma:     [..., K] posterior probabilities
        embedding: [K, D] phase embedding matrix (D = d_model)
        eps:       numerical floor for entropy computation

    Returns:
        gate_phase:  [..., D] gated phase embedding
        entropy:     [...] Shannon entropy
        confidence:  [...] normalized confidence
    """
    K = gamma.shape[-1]
    D = embedding.shape[-1]
    assert embedding.shape == (K, D), (
        f"Embedding shape {embedding.shape} != ({K}, {D})"
    )

    # Step 1: entropy confidence
    entropy, confidence = compute_entropy_confidence(gamma, eps=eps)

    # Step 2: phase embedding = γ @ E → [..., D]
    phase_embed = gamma @ embedding  # [..., K] @ [K, D] → [..., D]

    # Step 3: gating = c · phase_embed
    # c is [...], phase_embed is [..., D], need broadcasting
    gate_phase = confidence[..., np.newaxis] * phase_embed

    return gate_phase, entropy, confidence


# ──────────────────────────────────────────────────────────────────
# Ablation emission variants (reviewer defense)
# ──────────────────────────────────────────────────────────────────

@dataclass
class EmissionConfig:
    """Configuration for emission feature selection (ablation study).

    Variants:
        'marker'   — MSigDB phase-specific markers (primary, Option A)
        'variance' — Top-N genes by variance (data-driven, no biology)
        'random'   — Random N genes (3-seed average for stability)
        'latent'   — All genes → PCA/encoder → low-dim (Option C)
    """
    emission_type: str = "marker"             # marker | variance | random | latent
    n_emission_features: int = N_MARKERS      # 16 for marker, configurable for others
    marker_gene_symbols: list[str] = field(
        default_factory=lambda: MARKER_GENES_FLAT.copy()
    )
    random_seeds: list[int] = field(
        default_factory=lambda: [42, 123, 456]
    )
    latent_dim: int = 16                      # for 'latent' mode PCA components


def select_emission_features(
    x_full: np.ndarray,
    gene_symbols: list[str],
    config: EmissionConfig,
) -> tuple[np.ndarray, list[int]]:
    """Select emission features from a full feature matrix.

    The `gene_symbols` parameter is named for the canonical use case
    (HGNC gene symbols matching `MARKER_GENES_FLAT`), but in practice it
    accepts **any string identifier per column**:

      - 'marker' mode: identifiers must match `config.marker_gene_symbols`
                       entries for the lookup to succeed. Missing markers
                       emit a warning and are dropped.
      - 'variance'/'random' modes: identifiers are unused during selection
                       but the caller can recover the names of selected
                       columns via the returned `selected_indices`.
      - 'latent' mode: identifiers are entirely ignored (PCA components
                       have no per-feature name).

    For the Whitfield 2002 use case (Step 3 preprocessor), the platform
    annotation provides probe IDs (cDNA spot identifiers), not gene symbols
    — pass `probe_metadata["probe_ids"]` subset to `kept_probe_indices` as
    `gene_symbols`. Marker mode will fail to find canonical symbols and
    fall back to whatever was passed; use variance/random/latent until
    Tier 3 (HGNC mapping) is implemented.

    Args:
        x_full:       [T, G] full feature matrix.
        gene_symbols: [G] feature labels (gene symbols, probe IDs, or any
                      string identifier per column).
        config:       EmissionConfig specifying selection strategy.

    Returns:
        x_emission:      [T, V_emission] selected feature matrix.
        selected_indices: [V_emission] column indices in x_full.
    """
    T, G = x_full.shape
    assert len(gene_symbols) == G, f"Gene symbols ({len(gene_symbols)}) != columns ({G})"

    if config.emission_type == "marker":
        # Find column indices for marker genes.
        # Use first occurrence if gene symbol appears multiple times
        # (common in microarray data: multiple probes per gene).
        symbol_to_idx: dict[str, int] = {}
        for i, s in enumerate(gene_symbols):
            if s not in symbol_to_idx:  # keep first occurrence only
                symbol_to_idx[s] = i
        indices = []
        missing = []
        for symbol in config.marker_gene_symbols:
            if symbol in symbol_to_idx:
                indices.append(symbol_to_idx[symbol])
            else:
                missing.append(symbol)
        if missing:
            import warnings
            warnings.warn(
                f"Marker genes not found in data: {missing}. "
                f"Using {len(indices)}/{len(config.marker_gene_symbols)} available markers."
            )
        if len(indices) == 0:
            raise ValueError("No marker genes found in the provided gene symbols.")
        return x_full[:, indices], indices

    elif config.emission_type == "variance":
        # Top-N genes by temporal variance (data-driven)
        variances = np.var(x_full, axis=0)  # [G]
        indices = np.argsort(variances)[::-1][:config.n_emission_features].tolist()
        indices.sort()  # deterministic ordering
        return x_full[:, indices], indices

    elif config.emission_type == "random":
        # Random N genes — for single-seed call; multi-seed averaging
        # is handled at the experiment level (run 3 seeds, average metrics)
        n_select = config.n_emission_features
        if n_select > G:
            raise ValueError(
                f"n_emission_features ({n_select}) > total genes ({G}). "
                f"Cannot select more features than available."
            )
        rng = np.random.RandomState(config.random_seeds[0])
        indices = sorted(rng.choice(G, size=n_select, replace=False).tolist())
        return x_full[:, indices], indices

    elif config.emission_type == "latent":
        # PCA-based dimensionality reduction (Option C)
        from numpy.linalg import svd
        # Center
        x_centered = x_full - x_full.mean(axis=0, keepdims=True)
        # Truncated SVD
        U, S, Vt = svd(x_centered, full_matrices=False)
        # Project to top-k components
        n_comp = min(config.latent_dim, min(T, G))
        x_latent = U[:, :n_comp] * S[:n_comp]  # [T, n_comp]
        # indices are meaningless for latent (return range for API compat)
        return x_latent, list(range(n_comp))

    else:
        raise ValueError(f"Unknown emission_type: {config.emission_type!r}")


# ──────────────────────────────────────────────────────────────────
# Synthetic cell cycle data generator (for testing)
# ──────────────────────────────────────────────────────────────────

def generate_synthetic_cell_cycle(
    n_genes: int = 16,
    n_timepoints: int = 48,
    cycle_period: int = 22,
    K: int = 4,
    noise_std: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic cell cycle gene expression data.

    Each gene group (4 per phase) peaks during its assigned phase.
    The time course covers ~2 full cycles (48 timepoints, 22h period).

    Models a synchronized cell population with damping (amplitude
    decreases over time as synchronization is lost).

    Args:
        n_genes:       total genes (must be divisible by K)
        n_timepoints:  length of time course
        cycle_period:  period in timepoints (22 ≈ HeLa cycle hours)
        K:             number of phases
        noise_std:     Gaussian noise level
        seed:          random seed

    Returns:
        x:           [T, n_genes] expression matrix (log2 ratio scale)
        true_states: [T] ground truth phase labels (0=G1, 1=S, 2=G2, 3=M)
        true_phases: [T] continuous phase angle in [0, 2π)
    """
    rng = np.random.RandomState(seed)
    assert n_genes % K == 0, f"n_genes ({n_genes}) must be divisible by K ({K})"
    genes_per_phase = n_genes // K

    t = np.arange(n_timepoints, dtype=np.float64)

    # Continuous phase angle (wraps every cycle_period)
    phase_angle = (2 * np.pi * t / cycle_period) % (2 * np.pi)  # [T]

    # Discrete state labels from phase angle
    true_states = np.floor(phase_angle / (2 * np.pi / K)).astype(int)
    true_states = np.clip(true_states, 0, K - 1)

    # Damping factor (synchronization loss over time)
    damping = np.exp(-t / (3 * cycle_period))  # τ = 3 cycles ≈ 66h

    # Generate gene expression
    x = np.zeros((n_timepoints, n_genes))
    for k in range(K):
        # Phase offset for this gene group
        phase_offset = 2 * np.pi * k / K
        for g in range(genes_per_phase):
            col = k * genes_per_phase + g
            # Base signal: cosine peaking at assigned phase
            signal = damping * np.cos(phase_angle - phase_offset - g * 0.05)
            # Add gene-specific amplitude variation
            amplitude = 0.5 + rng.rand() * 1.0
            x[:, col] = amplitude * signal + rng.randn(n_timepoints) * noise_std

    return x, true_states, phase_angle


# ──────────────────────────────────────────────────────────────────
# CellCycleConfig — domain config dataclass
# ──────────────────────────────────────────────────────────────────

@dataclass
class CellCycleConfig:
    """Cell cycle domain configuration for CG-Mamba (EGCPM).

    Encapsulates all cell-cycle-specific hyperparameters in one place.
    Used by the forecaster orchestrator to configure HMM, encoder, decoder.

    K-flexible: supports K=3 (G2/M merged) and K=4. Use BIC to select.
    """
    # Domain
    domain: str = "cell_cycle"

    # HMM
    K: int = K_CELL_CYCLE                           # 3 or 4 phases
    hmm_covariance: str = "diag"                    # diag for 16-dim marker emission
    hmm_n_iter: int = 100
    hmm_seed: int = 42
    hmm_init_mode: str = "phase_aware"              # phase_aware | random
    collapse_lambda: float = 0.1
    collapse_min_occ: float = 0.05
    mask_epsilon: float = SOFT_MASK_EPSILON          # soft mask ε (default 1e-4)
    sync_method: str = "thy"                         # G8: thy | thy_noc — Exp3/Exp4 π init

    # Emission (ablation)
    emission: EmissionConfig = field(default_factory=EmissionConfig)

    # Data
    n_total_genes: int = 874                        # Whitfield periodic genes
    n_timepoints_train: int = 48                    # Exp3 (double thymidine)
    n_timepoints_test: int = 17                     # Exp4 (thy-noc, cross-experiment)
    cycle_period_hours: float = 22.0                # HeLa cell cycle period

    # Encoder (Mamba backbone)
    d_model: int = 64
    depth: int = 3
    N_warm: int = 4                                 # 8% of L~50 (smaller than ILI's 14)
    V_encoder: int = 874                            # G5: Mamba encoder input dim (full gene set,
                                                    # distinct from HMM marker V = emission.n_emission_features)

    # Fourier seasonality
    fourier_periods: list[float] = field(
        default_factory=lambda: [22.0]              # fundamental period only
    )
    d_season_target: int = 6                        # G7: SeasonalityModule output dim
                                                    # (1 period × sin+cos = 2 Fourier feats → projected to 6)

    # Forecasting — Direction Message v2 §2.5 multi-horizon (5차 review §2.1)
    # h=1  : single-step (immediate fluctuation)
    # h=5  : ~¼ cycle (operational, analog to ILI 4-week horizon)
    # h=11 : ~½ cycle (phase-transition boundary)
    # h=22 dropped from main: with L_win=24, T=48 yields only 3 windows for
    # h=22, statistically unreliable for MAE/WIS reporting. Supplementary
    # qualitative figure (single trajectory) used instead.
    horizons: tuple[int, ...] = (1, 5, 11)
    lookback: int = 24                              # L_win — sliding window length

    # Optimizer LR ratios (Direction Message v2 §2.8, mitigates small-data
    # overfit on state_embeddings). Resolved by CellCycleForecaster.make_optimizer.
    state_embed_lr_ratio: float = 0.02              # state_embed_lr = backbone_lr × this
    weight_decay_state_embed: float = 1e-4          # vs ILI's 0.0 — small data needs WD

    # Small-data regularization (5차 review §2.2: 118K params / 20 windows
    # = 5,900 params/sample, 6,900× ILI's ratio → aggressive regularization)
    dropout: float = 0.1                            # vs ILI's 0.0 baseline
    decoder_hidden: int | None = None               # None → direct Linear (no shared MLP);
                                                    # int → SiLU(Linear)+per-horizon heads
    weight_decay_decoder: float = 1e-3              # vs ILI's 1e-5 — 100× tighter
    base_lr: float = 2e-4                           # vs ILI's 5e-4 — slower learning
    early_stop_patience: int = 7                    # vs ILI's 20 — fast halt on small CV folds

    # Loss function (5차 review R1)
    loss_type: str = "huber"                        # huber | mse | mae
    huber_delta: float = 1.0                        # L2→L1 transition threshold

    # Context module
    use_context: bool = False                       # Phase-Only (Option 3): no external context
    use_dsp: bool = False                           # DSP not applicable to cell cycle

    def __post_init__(self) -> None:
        """Validate fields after construction (P2.4 — symmetric coverage).

        Catches common config mistakes early before they propagate to HMM
        construction or training. Covers all numeric/enum fields in the
        dataclass uniformly: previously only the new (G5/G7/G8) fields were
        validated, leading to asymmetric coverage.

        Includes the K=3 + default-K=4-emission mismatch fail-fast (I3),
        which would otherwise surface only at hmm.fit() time with a
        confusing shape error.
        """
        # Core HMM enums and K
        if self.K not in (3, 4):
            raise ValueError(f"K must be 3 or 4, got {self.K}")
        if self.sync_method not in ("thy", "thy_noc"):
            raise ValueError(
                f"sync_method must be 'thy' or 'thy_noc', got {self.sync_method!r}"
            )
        if self.hmm_covariance not in ("diag", "full"):
            raise ValueError(
                f"hmm_covariance must be 'diag' or 'full', got {self.hmm_covariance!r}"
            )
        if self.hmm_init_mode not in ("phase_aware", "random"):
            raise ValueError(
                f"hmm_init_mode must be 'phase_aware' or 'random', "
                f"got {self.hmm_init_mode!r}"
            )

        # Numeric HMM fields
        if self.hmm_n_iter < 1:
            raise ValueError(f"hmm_n_iter must be >= 1, got {self.hmm_n_iter}")
        if self.collapse_lambda < 0:
            raise ValueError(
                f"collapse_lambda must be >= 0, got {self.collapse_lambda}"
            )
        if not (0.0 < self.collapse_min_occ < 1.0):
            raise ValueError(
                f"collapse_min_occ must be in (0, 1), got {self.collapse_min_occ}"
            )
        if not (0.0 <= self.mask_epsilon < 1.0):
            raise ValueError(
                f"mask_epsilon must be in [0, 1), got {self.mask_epsilon}"
            )

        # Data / encoder / forecasting
        if self.n_total_genes < 1:
            raise ValueError(f"n_total_genes must be >= 1, got {self.n_total_genes}")
        if self.n_timepoints_train < 1:
            raise ValueError(
                f"n_timepoints_train must be >= 1, got {self.n_timepoints_train}"
            )
        if self.n_timepoints_test < 1:
            raise ValueError(
                f"n_timepoints_test must be >= 1, got {self.n_timepoints_test}"
            )
        if self.cycle_period_hours <= 0:
            raise ValueError(
                f"cycle_period_hours must be > 0, got {self.cycle_period_hours}"
            )
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.N_warm < 0:
            raise ValueError(f"N_warm must be >= 0, got {self.N_warm}")
        if self.V_encoder < 1:
            raise ValueError(f"V_encoder must be >= 1, got {self.V_encoder}")
        if self.d_season_target < 1:
            raise ValueError(
                f"d_season_target must be >= 1, got {self.d_season_target}"
            )
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError(
                f"horizons must be a non-empty tuple of positive ints, "
                f"got {self.horizons}"
            )
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self.lookback}")
        # 2.1 sanity: longest horizon must fit within remaining data after lookback.
        # Use the smaller of the two training experiments (Exp3 = 48 timepoints).
        max_horizon = max(self.horizons)
        if self.lookback + max_horizon > self.n_timepoints_train:
            raise ValueError(
                f"lookback ({self.lookback}) + max(horizons) ({max_horizon}) > "
                f"n_timepoints_train ({self.n_timepoints_train}); no valid "
                f"sliding window. Reduce lookback or horizons."
            )
        # 2.8 small-data overfit guards.
        if self.state_embed_lr_ratio <= 0:
            raise ValueError(
                f"state_embed_lr_ratio must be > 0, got {self.state_embed_lr_ratio}"
            )
        if self.weight_decay_state_embed < 0:
            raise ValueError(
                f"weight_decay_state_embed must be >= 0, "
                f"got {self.weight_decay_state_embed}"
            )

        # 5차 review §2.2: small-data regularization field guards.
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.decoder_hidden is not None and self.decoder_hidden < 1:
            raise ValueError(
                f"decoder_hidden must be None or >= 1, got {self.decoder_hidden}"
            )
        if self.weight_decay_decoder < 0:
            raise ValueError(
                f"weight_decay_decoder must be >= 0, got {self.weight_decay_decoder}"
            )
        if self.base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {self.base_lr}")
        if self.early_stop_patience < 1:
            raise ValueError(
                f"early_stop_patience must be >= 1, got {self.early_stop_patience}"
            )

        # 5차 review R1: loss function selection.
        if self.loss_type not in ("huber", "mse", "mae"):
            raise ValueError(
                f"loss_type must be 'huber', 'mse', or 'mae', got {self.loss_type!r}"
            )
        if self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")

        # I3 — Fail fast on K=3 with default K=4 marker emission.
        # Default EmissionConfig() carries N_MARKERS=16 K=4 symbols, which
        # would silently create a K=3 HMM with V=16 features → confusing
        # AssertionError at the first hmm.fit(x_12_features) call. We force
        # explicit acknowledgement via a clear error message instead.
        if (
            self.K == 3
            and self.emission.emission_type == "marker"
            and self.emission.n_emission_features == N_MARKERS
            and tuple(self.emission.marker_gene_symbols) == tuple(MARKER_GENES_FLAT)
        ):
            raise ValueError(
                f"CellCycleConfig(K=3) requires an explicit K=3 EmissionConfig. "
                f"The default emission has {N_MARKERS} K=4 markers, incompatible "
                f"with a 3-state HMM that expects {N_MARKERS_K3} K=3 markers.\n"
                f"Fix:\n"
                f"    from src.models.cell_cycle_hmm import (\n"
                f"        CellCycleConfig, EmissionConfig,\n"
                f"        MARKER_GENES_FLAT_K3, N_MARKERS_K3,\n"
                f"    )\n"
                f"    cfg = CellCycleConfig(\n"
                f"        K=3,\n"
                f"        emission=EmissionConfig(\n"
                f"            n_emission_features=N_MARKERS_K3,\n"
                f"            marker_gene_symbols=list(MARKER_GENES_FLAT_K3),\n"
                f"        ),\n"
                f"    )"
            )

    def build_hmm(self) -> CellCycleHMM:
        """Construct CellCycleHMM from this config.

        Note: `cycle_period` is cast to int because CellCycleHMM uses it for
        integer segmentation in `_init_params`. Non-integer values (e.g.,
        22.5h) lose precision via truncation. A UserWarning is emitted when
        this happens so the caller can choose to round or restate the value.
        """
        cycle_period_int = int(self.cycle_period_hours)
        if cycle_period_int != self.cycle_period_hours:
            import warnings
            warnings.warn(
                f"cycle_period_hours={self.cycle_period_hours} (non-integer) "
                f"truncated to {cycle_period_int} for HMM segmentation. "
                f"Use an integer value to suppress this warning.",
                UserWarning,
            )
        return CellCycleHMM(
            n_states=self.K,
            n_features=self.emission.n_emission_features,
            covariance_type=self.hmm_covariance,
            n_iter=self.hmm_n_iter,
            seed=self.hmm_seed,
            collapse_lambda=self.collapse_lambda,
            collapse_min_occ=self.collapse_min_occ,
            init_mode=self.hmm_init_mode,
            cycle_period=cycle_period_int,
            mask_epsilon=self.mask_epsilon,
            sync_method=self.sync_method,
        )
