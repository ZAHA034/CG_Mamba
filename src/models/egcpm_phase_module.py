"""EGCPMPhaseModule — Entropy-Gated Cyclic Phase Module (cell cycle domain).

Standalone PyTorch module that consumes a precomputed HMM posterior γ and
emits an entropy-modulated phase embedding for the Mamba encoder.

Architectural separation from `PhaseModule` (ILI):
    - PhaseModule (ILI): embeds the HMM internally — takes raw input
      x_raw [B, L, V_raw], runs forward-backward over cached buffers, and
      emits a sigmoid-gated phase embedding.
    - EGCPMPhaseModule (cell cycle): the HMM is fit offline on marker genes
      (CellCycleHMM), posteriors are precomputed, and this module receives
      γ directly. No internal HMM, no sigmoid — linear scaling by an
      entropy-derived confidence factor.

Pipeline:
    γ [B, L, K]
        │
        ├─ H = -Σ_k γ_k log γ_k                        Shannon entropy [B, L]
        ├─ c = clip(1 - H/log K, 0, 1)                 confidence       [B, L]
        ├─ phase_embed = γ @ state_embeddings          phase embed      [B, L, D]
        └─ gate_phase  = c.unsqueeze(-1) · phase_embed                  [B, L, D]

Information-theoretic interpretation:
    Confidence c(t) is *equivalently* the normalized KL-divergence from the
    uniform (maximum-entropy) prior U_K = (1/K, ..., 1/K):

        c(t) = D_KL(γ_t || U_K) / log K
             = (log K - H(γ_t)) / log K
             = 1 - H(γ_t) / log K

    So the gate scales the phase embedding by the *information gain* the
    HMM posterior carries over an uninformative prior. At phase boundaries
    (γ ≈ uniform) the gate vanishes, recovering an unbiased Mamba encoder;
    at stable phase centers (γ ≈ one-hot) the gate transmits the embedding
    in full. The same code is interpretable in either entropy or KL terms.

Reference:
    Direction Message v2 §3.1 (data flow trace) — gate = c · (γ @ E) linear.

Numerical parity:
    The torch entropy formulation `gamma * log(gamma.clamp(min=eps))` is
    mathematically more correct than the NumPy reference
    `cell_cycle_hmm.compute_entropy_confidence` which clips both factors
    (`gamma_safe * log(gamma_safe)`): for γ ≈ 0 entries, the torch form
    correctly contributes 0 (lim x→0⁺ x log x = 0), while the numpy form
    contributes eps · log(eps) ≈ -2.76e-11 per dim. Practical impact: nil,
    but worth noting for downstream cross-checks.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


class EGCPMPhaseModule(nn.Module):
    """Entropy-Gated Cyclic Phase Module for cell cycle domain.

    Args:
        K:                  number of HMM states (3 or 4 — Whitfield/MSigDB)
        d_embed:            phase embedding output dim (= d_model in encoder)
        use_entropy_gating: whether to scale phase_embed by entropy confidence
                            c(t) = 1 - H/log K. True (default) is the EGCPM
                            canonical operation; False is the ablation
                            baseline that returns the plain γ @ E embedding.

    Attributes:
        state_embeddings:   [K, d_embed] learnable, zeros-initialized so the
                            gate is identically zero at t=0 (neutral start,
                            matching PhaseModule R-4 convention).
        _last_entropy:      [B, L] Shannon entropy from the most recent
                            training forward pass; None outside training or
                            when entropy gating is disabled.
        _last_confidence:   [B, L] confidence c from the most recent
                            training forward pass; same nullability rules.
    """

    def __init__(
        self,
        K: int = 4,
        d_embed: int = 64,
        use_entropy_gating: bool = True,
    ):
        super().__init__()
        assert K >= 2, f"K must be >= 2, got {K}"
        assert d_embed >= 1, f"d_embed must be >= 1, got {d_embed}"

        self.K = int(K)
        self.d_embed = int(d_embed)
        self.use_entropy_gating = bool(use_entropy_gating)
        self._log_K = math.log(float(self.K))

        # zeros init → gate = 0 at construction time (no phase bias).
        # Mirrors PhaseModule's R-4 convention (phase_module.py:94).
        self.state_embeddings = nn.Parameter(torch.zeros(self.K, self.d_embed))

        # Monitoring (train-only, mirrors PhaseModule._last_gamma pattern)
        self._last_entropy: Optional[torch.Tensor] = None
        self._last_confidence: Optional[torch.Tensor] = None

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        """Compute the entropy-gated phase embedding.

        Args:
            gamma: [B, L, K] HMM posterior (last dim sums to 1).
                   Provided by the cell cycle forecaster after calling
                   CellCycleHMM.posteriors(x_markers) and adding the batch
                   dimension.

        Returns:
            gate_phase: [B, L, d_embed]
                - if use_entropy_gating: c.unsqueeze(-1) · (γ @ E)
                - else:                  γ @ E              (ablation baseline)
        """
        # 5차 review §2.3 (A.6 fix): RuntimeError (not assert) so the guard
        # survives `python -O` (which strips assertions). Mirrors ILI's
        # PhaseModule.forward convention. Catches the common Step 5 ablation
        # bug where K=3 vs K=4 HMM γ is fed to the wrong forecaster.
        if gamma.dim() != 3 or gamma.shape[-1] != self.K:
            raise RuntimeError(
                f"gamma_window K dimension mismatch: expected [B, L, K={self.K}], "
                f"got {tuple(gamma.shape)}. Did you pass γ from a different "
                f"HMM than the one used in init_from_hmm()?"
            )

        phase_embed = gamma @ self.state_embeddings           # [B, L, d_embed]

        if not self.use_entropy_gating:
            # Ablation baseline: γ @ E without entropy modulation.
            # Clear unconditionally (symmetric with gating path, prevents
            # stale values from a previous training pass surfacing in eval).
            self._last_entropy = None
            self._last_confidence = None
            return phase_embed

        # EGCPM core: Shannon entropy → confidence → linear gate.
        H = -(gamma * torch.log(gamma.clamp(min=1e-12))).sum(dim=-1)    # [B, L]
        c = (1.0 - H / self._log_K).clamp(min=0.0, max=1.0)             # [B, L]

        gate_phase = c.unsqueeze(-1) * phase_embed                       # [B, L, d_embed]

        if self.training:
            self._last_entropy = H.detach()
            self._last_confidence = c.detach()
        else:
            self._last_entropy = None
            self._last_confidence = None

        return gate_phase

    # ──────────────────────────────────────────────────────────────────
    # HMM-informed state-embedding initialization (B-4 symmetry hazard fix)
    # ──────────────────────────────────────────────────────────────────

    def init_from_hmm(
        self,
        fitted_hmm: Any,
        seed: int = 42,
        scale: float = 0.5,
    ) -> None:
        """B-4 symmetry-hazard fix — initialize state_embeddings from HMM identity.

        Default zeros init is symmetric across the K rows of state_embeddings:
        `gate_phase = c · (γ @ E)` becomes 0 for every row at construction,
        and backward gradients are collinear → the K rows can never separate
        during training (verified failure mode in ILI's M1.7 v1/v2 — see
        cg_forecaster.py:148 _init_state_embeddings_from_cache narrative).

        Cell cycle data is far more vulnerable than ILI:
          - ILI:  ~hundreds of thousands of training samples
          - Cell: ~20 sliding windows from Exp3 (T=48)
        With ≥3 orders-of-magnitude less data, the zeros initialization
        cannot recover from symmetry collapse via gradient flow alone.

        Identity construction per state k:
            identity_k = [normalize(μ_k), normalize(diag(Σ_k)), normalize(A[k, :])]
            ∈ ℝ^(2V + K)

        Each statistic encodes one HMM-self property:
          - μ_k       : emission mean — *what data does state k look like*
          - diag(Σ_k) : emission variance — *how concentrated is state k*
          - A[k, :]   : outgoing transitions — *what comes after state k*

        Random projection (2V + K) → d_embed:
            proj ~ N(0, 1 / sqrt(2V + K))   [Johnson-Lindenstrauss]
            state_embeddings = (identity @ proj) · scale

        seed=42 yields deterministic init; pass a different seed for ablation.

        Args:
            fitted_hmm: CellCycleHMM (or any object exposing K, V,
                        means [K, V], covars [K, V] or [K, V, V], A [K, K])
                        whose .fit() has been called.
            seed:       RNG seed for the JL projection (default 42, matches
                        ILI convention).
            scale:      multiplicative attenuation of the init (default 0.5).
                        Lower scale → smaller initial gate magnitude.

        Raises:
            ValueError: if hmm.K ≠ self.K, or hmm not fitted.
        """
        if not getattr(fitted_hmm, "_fitted", False):
            raise ValueError("init_from_hmm requires a fitted CellCycleHMM")
        if fitted_hmm.K != self.K:
            raise ValueError(
                f"K mismatch: hmm.K={fitted_hmm.K}, module.K={self.K}"
            )

        K = self.K
        V = fitted_hmm.V

        # diag(Σ_k): handle 'full' (K, V, V) and 'diag' (K, V) covariance.
        covars = np.asarray(fitted_hmm.covars)
        if covars.ndim == 3:
            diag_cov = np.array([np.diag(covars[k]) for k in range(K)])  # [K, V]
        else:
            diag_cov = covars.copy()                                      # [K, V]

        means = np.asarray(fitted_hmm.means, dtype=np.float64)            # [K, V]
        A = np.asarray(fitted_hmm.A, dtype=np.float64)                    # [K, K]

        def _row_normalize(x: np.ndarray) -> np.ndarray:
            """L2-normalize each row, with zero-row safety."""
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            return x / norms

        identity = np.concatenate(
            [_row_normalize(means), _row_normalize(diag_cov), _row_normalize(A)],
            axis=1,
        )                                                                  # [K, 2V + K]
        d_id = identity.shape[1]

        rng = np.random.RandomState(seed)
        proj = rng.randn(d_id, self.d_embed).astype(np.float64) / np.sqrt(d_id)
        init = (identity @ proj) * scale                                   # [K, d_embed]

        with torch.no_grad():
            self.state_embeddings.copy_(
                torch.tensor(init, dtype=self.state_embeddings.dtype,
                             device=self.state_embeddings.device)
            )

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, d_embed={self.d_embed}, "
            f"use_entropy_gating={self.use_entropy_gating}"
        )
