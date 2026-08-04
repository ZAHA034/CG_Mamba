"""PhaseModule — HMM-embedded phase posterior + emission-aware rollout (v2.0.9 M1.4c).

⚠️  ILI domain only. For cell cycle (cross-domain), use `EGCPMPhaseModule`
    from `src.models.egcpm_phase_module` (γ-input, linear gate, no internal HMM).

PLAN v2.0.9 PATCH 3 (D.4.2) + R-1/4 + S-2/4/5/7 + T-1/2/3 + post-review H1/H2/H3/M2 fixes

Role (v2.0.9 — major redesign vs v2.0.8c):
    v2.0.8c: HMM-agnostic *adapter* — consumed externally-provided γ [B,L,K]
             and emitted gate_phase = γ @ state_embed.
    v2.0.9:  HMM-*embedded* module — takes raw input x_raw [B,L,V_raw], builds
             augmented features [x_t, Δx_t] (Furui 1986 delta-MFCC tradition),
             runs torch-autograd-compatible forward-backward on cached HMM
             buffers, and emits BOTH gate_phase and phase_post as a tuple.

Cached HMM artifacts (6 register_buffer, S-2 정규화 반영):
    _A        [K, K]              transition matrix
    _pi       [K]                  initial distribution
    _means    [K, V_aug]           Gaussian means
    _covs     [K, V_aug, V_aug]    Σ_k + reg_covar·I  ← S-2: 정규화 적용된 covariance
    _cov_inv  [K, V_aug, V_aug]    (_covs)⁻¹
    _log_det  [K]                  log|_covs|
    The reg_covar regularization is folded into _covs at cache-time so that
    `_torch_log_emission*` matches `gaussian_hmm.py:_log_emission` (line 159
    cov_reg = self.covars[k] + self.reg_covar * np.eye(V)) bit-for-bit, including
    the aggressive `10·reg_covar·I` fallback when Cholesky fails (H3 safety net).

Stage 2 entry sequence (T-1 — replaces legacy `freeze_hmm_for_stage2`):
    1. (legacy) freeze_hmm_for_stage2() — NEVER call (GaussianHMM has no params)
    2. model.phase_module._cache_hmm_torch(fitted_hmm, device)   ← mandatory once
    3. optimizer = torch.optim.AdamW(model.parameters(), lr=...)  ← buffers auto-excluded

Stage 3 (default SKIP, T-2):
    phase_module._unfreeze_for_stage3()  → A & μ_k become Parameters via
    standard `register_parameter` API (M2 fix: no internal `_buffers` mutation).
    pi / covars / cov_inv / log_det remain frozen buffers (over-constrained).

API:
    forward(x_raw)                              → (gate_phase, phase_post)
    rollout_gate(gamma_t, x_window, horizon)    → (gate_phase_h, gamma_t_h, confidence)
    rollout(gamma_T, x_window, H)               # internal, emission-aware iterative
    _augment_features(x_raw)                    # [B,L,V_raw] → [B,L-1,V_aug]
    _cache_hmm_torch(fitted_hmm, device)        # NumPy → buffers (S-2 정규화 + H3 fallback)
    _torch_log_emission(obs)                    # per-timestep [B,V_aug] → [B,K]  (rollout 전용)
    _torch_log_emission_batched(x_aug)          # batched [B,L,V_aug] → [B,L,K]   (H2 성능)
    _torch_forward_backward(x_aug)              # [B,L,V_aug] → gamma [B,L,K]     (H1 list-stack)
    _unfreeze_for_stage3()                      # buffer → Parameter              (M2 standard API)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class PhaseModule(nn.Module):
    """v2.0.9 HMM-embedded Phase Dynamics module.

    Construction is decoupled from HMM fitting — instantiate with the
    architectural sizes only, then call `_cache_hmm_torch(fitted_hmm, device)`
    once a Stage-1 GaussianHMM has been fit on augmented features.

    Args:
        V_raw:       raw feature dim (V_aug = 2 · V_raw after augmentation)
        K:           number of HMM states (M1.4b winner: 3, fixed by BIC)
        d_embed:     state-embedding output dim (= d_model in CGForecaster)
        hmm_fitted:  optional GaussianHMM already fit on augmented data.
                     If provided, _cache_hmm_torch is invoked immediately.
        device:      device for buffers (defaults to state_embeddings.device).
    """

    # Class-level constant (New-L3): Gaussian log-normalization term.
    _LOG_2PI: float = math.log(2.0 * math.pi)

    def __init__(
        self,
        V_raw: int = 3,
        K: int = 3,
        d_embed: int = 64,
        hmm_fitted=None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert V_raw >= 1 and K >= 1 and d_embed >= 1
        self.V_raw = int(V_raw)
        self.V_aug = 2 * int(V_raw)
        self.K = int(K)
        self.d_embed = int(d_embed)

        # ── Learnable parameters (v2.1.4 L-1: zeros DECLARATION only — overwritten) ──
        # ⚠️ state_embeddings = zeros(K, d_embed) is DECLARATION ONLY. The actual
        # init used during Stage 2 training is HMM-enriched JL projection
        # (PLAN §3.5 + §16 v2.1.4 CH-1), applied via:
        #     CGForecaster.prepare_for_stage2(fitted_hmm)
        #         → _init_state_embeddings_from_cache (cg_forecaster.py:168-243)
        # If prepare_for_stage2() is NOT called before forward(), CGForecaster
        # raises RuntimeError (cg_forecaster.py:459-463) — silent symmetry-
        # stagnation regression is guarded by construction.
        #
        # The original "zeros init → sigmoid(γ·0)=0.5 neutral gate" rationale is
        # superseded by ERR-C5 (symmetry preservation hazard, PLAN §16 v2.1.4):
        # zeros init causes collinear gradients across all K state rows, so the
        # embeddings never differentiate. HMM-enriched identity + JL projection
        # (per-row norm > 0.1, pairwise |cos| < 0.7) breaks symmetry while
        # remaining low-magnitude (× 0.5 scale = σ ∈ [0.378, 0.622]) to preserve
        # start-near-vanilla behavior.
        #
        # Standalone PhaseModule instantiation (e.g., in unit tests, ablation
        # scripts) MUST either call CGForecaster.prepare_for_stage2(hmm) OR
        # initialize self.state_embeddings manually — never train this module
        # with zeros init.
        self.state_embeddings = nn.Parameter(torch.zeros(self.K, self.d_embed))

        # ── HMM buffers (T-1, T-3; S-2 reg-applied) ──
        # Allocated as zeros, filled by _cache_hmm_torch(fitted_hmm).
        self.register_buffer("_A", torch.zeros(self.K, self.K))
        self.register_buffer("_pi", torch.zeros(self.K))
        self.register_buffer("_means", torch.zeros(self.K, self.V_aug))
        self.register_buffer("_covs", torch.zeros(self.K, self.V_aug, self.V_aug))
        self.register_buffer("_cov_inv", torch.zeros(self.K, self.V_aug, self.V_aug))
        self.register_buffer("_log_det", torch.zeros(self.K))
        self._hmm_cached: bool = False

        # Monitoring (train-only, M1.3 _last_gate convention)
        self._last_gamma: Optional[torch.Tensor] = None

        if hmm_fitted is not None:
            self._cache_hmm_torch(hmm_fitted, device=device)

    # ──────────────────────────────────────────────────────────────
    # Feature augmentation (Furui 1986 delta-MFCC tradition)
    # ──────────────────────────────────────────────────────────────
    def _augment_features(self, x_raw: torch.Tensor) -> torch.Tensor:
        """[B, L, V_raw] → [B, L-1, V_aug=2·V_raw] = concat([x_t, Δx_t])."""
        assert x_raw.dim() == 3 and x_raw.shape[-1] == self.V_raw, (
            f"x_raw shape {tuple(x_raw.shape)} incompatible with V_raw={self.V_raw}"
        )
        assert x_raw.shape[1] >= 2, "Need L ≥ 2 to compute Δx"
        delta = x_raw[:, 1:, :] - x_raw[:, :-1, :]                  # [B, L-1, V_raw]
        x_aug = torch.cat([x_raw[:, 1:, :], delta], dim=-1)         # [B, L-1, V_aug]
        return x_aug

    # ──────────────────────────────────────────────────────────────
    # NumPy → Torch bridge (R-1, T-1, S-2, H3 fallback)
    # ──────────────────────────────────────────────────────────────
    def _cache_hmm_torch(self, fitted_hmm, device: Optional[torch.device] = None) -> None:
        """Copy fitted GaussianHMM artifacts into 6 register_buffers (S-2 정규화 포함).

        The reg_covar regularization is applied here (before storing _covs and
        before computing _cov_inv / _log_det), so that `_torch_log_emission*`
        is numerically consistent with `gaussian_hmm.py:_log_emission` which
        does `cov_reg = self.covars[k] + self.reg_covar * np.eye(V)` per call.

        H3 safety net: if Cholesky fails on Σ_k + reg_covar·I (rare ill-condition),
        apply the same aggressive fallback as gaussian_hmm.py:163-165
        (`cov + 10·reg_covar·I`) so PhaseModule mirrors the fitting-time numerics.

        Args:
            fitted_hmm: an instance of `src.models.gaussian_hmm.GaussianHMM`
                        whose `.fit()` has been called. Required attrs:
                        K, V, A, pi, means, covars, reg_covar, _fitted.
            device:     where to place the buffers (default: state_embeddings.device).
        """
        assert getattr(fitted_hmm, "_fitted", False), \
            "fitted_hmm must have completed .fit() before caching"
        assert fitted_hmm.K == self.K, \
            f"K mismatch: HMM={fitted_hmm.K}, module={self.K}"
        assert fitted_hmm.V == self.V_aug, (
            f"V mismatch: HMM features={fitted_hmm.V}, module V_aug={self.V_aug} "
            f"(expected fitted_hmm.V == 2·V_raw)"
        )
        if device is None:
            device = self.state_embeddings.device

        K, V_aug = self.K, self.V_aug
        reg = float(fitted_hmm.reg_covar)
        eye_V = np.eye(V_aug, dtype=np.float64)

        # S-2: pre-apply regularization. covars_reg[k] = Σ_k + reg·I
        covars_reg = fitted_hmm.covars.astype(np.float64) + reg * eye_V[None, :, :]

        # H3 safety net (mirrors gaussian_hmm.py:160-165 Cholesky+aggressive fallback)
        for k in range(K):
            try:
                np.linalg.cholesky(covars_reg[k])
            except np.linalg.LinAlgError:
                # gaussian_hmm.py:164 path: `cov_reg = cov_reg + 10 * reg · I`.
                # Apply ON TOP of the existing reg·I (matches in-fit numerics).
                covars_reg[k] = covars_reg[k] + 10.0 * reg * eye_V
                np.linalg.cholesky(covars_reg[k])  # must succeed; else raise

        cov_inv = np.linalg.inv(covars_reg)
        sign, log_det = np.linalg.slogdet(covars_reg)
        assert (sign > 0).all() and np.isfinite(log_det).all(), (
            f"Regularized covariance not positive-definite (sign={sign}, "
            f"log_det={log_det}). Check fitted_hmm.covars."
        )

        dtype = self.state_embeddings.dtype
        self._A.data = torch.from_numpy(fitted_hmm.A).to(device=device, dtype=dtype)
        self._pi.data = torch.from_numpy(fitted_hmm.pi).to(device=device, dtype=dtype)
        self._means.data = torch.from_numpy(fitted_hmm.means).to(device=device, dtype=dtype)
        self._covs.data = torch.from_numpy(covars_reg).to(device=device, dtype=dtype)
        self._cov_inv.data = torch.from_numpy(cov_inv).to(device=device, dtype=dtype)
        self._log_det.data = torch.from_numpy(log_det).to(device=device, dtype=dtype)
        self._hmm_cached = True

    # ──────────────────────────────────────────────────────────────
    # Per-timestep log-emission (autograd-compatible, used by rollout)
    # ──────────────────────────────────────────────────────────────
    def _torch_log_emission(self, obs: torch.Tensor) -> torch.Tensor:
        """log N(obs | μ_k, Σ_k_reg) for each state k. Used in rollout per-step.

        Args:
            obs: [B, V_aug] per-timestep observation.
        Returns:
            log_lik: [B, K] per-state log-likelihood.
        """
        assert obs.dim() == 2 and obs.shape[-1] == self.V_aug, (
            f"obs shape {tuple(obs.shape)} expected [B, V_aug={self.V_aug}]"
        )
        # diff[b, k, v] = obs[b, v] - μ[k, v]
        diff = obs.unsqueeze(1) - self._means.unsqueeze(0)               # [B, K, V_aug]
        # mahalanobis: 'bkv, kvw, bkw -> bk'
        maha = torch.einsum("bkv,kvw,bkw->bk", diff, self._cov_inv, diff)  # [B, K]
        log_lik = -0.5 * (self.V_aug * self._LOG_2PI + self._log_det.unsqueeze(0) + maha)
        return log_lik

    # ──────────────────────────────────────────────────────────────
    # Batched log-emission (H2 — used by forward-backward over L)
    # ──────────────────────────────────────────────────────────────
    def _torch_log_emission_batched(self, x_aug: torch.Tensor) -> torch.Tensor:
        """log N(x_aug | μ_k, Σ_k_reg) over a full sequence in one einsum.

        Replaces the Python `for t in range(L)` loop that called per-timestep
        `_torch_log_emission` (post-review H2 performance fix). For L=156,
        B=32, K=3, V_aug=6 this is ~80% faster than the per-step loop while
        remaining numerically identical (same einsum contraction modulo
        order-of-summation differences within float-rounding).

        Args:
            x_aug: [B, L, V_aug] full augmented sequence.
        Returns:
            log_lik: [B, L, K].
        """
        assert x_aug.dim() == 3 and x_aug.shape[-1] == self.V_aug, (
            f"x_aug shape {tuple(x_aug.shape)} expected [B, L, V_aug={self.V_aug}]"
        )
        # diff[b, l, k, v] = x_aug[b, l, v] - μ[k, v]
        diff = x_aug.unsqueeze(2) - self._means[None, None, :, :]        # [B, L, K, V_aug]
        # mahalanobis: 'blkv, kvw, blkw -> blk'
        maha = torch.einsum("blkv,kvw,blkw->blk", diff, self._cov_inv, diff)  # [B, L, K]
        log_lik = -0.5 * (self.V_aug * self._LOG_2PI + self._log_det[None, None, :] + maha)
        return log_lik

    # ──────────────────────────────────────────────────────────────
    # Forward-backward (log-space, autograd-compatible, H1 list-stack)
    # ──────────────────────────────────────────────────────────────
    def _torch_forward_backward(self, x_aug: torch.Tensor) -> torch.Tensor:
        """Vanilla Baum-Welch forward-backward (log-space).

        Returns only γ (not (γ, log_likelihood)) per T-3 specification — Stage 1
        training is performed offline with NumPy EM, so torch log-likelihood
        is unused.

        Post-review H1 fix: replaced `torch.empty + in-place assignment` with
        `list-append + torch.stack`. The previous pattern worked but mixed
        autograd-tracked writes into uninitialized memory, making NaN
        propagation hard to trace.

        Args:
            x_aug: [B, L, V_aug] augmented features.
        Returns:
            gamma: [B, L, K] soft posterior γ_t(k) = P(z_t=k | x_{1:L}).
        """
        if not self._hmm_cached:
            raise RuntimeError(
                "PhaseModule buffers empty. Call _cache_hmm_torch(fitted_hmm) "
                "before forward(). Stage 2 entry sequence (T-1) requires "
                "model.phase_module._cache_hmm_torch(device) once at the start."
            )
        assert x_aug.dim() == 3 and x_aug.shape[-1] == self.V_aug
        assert x_aug.shape[1] >= 1, "Need L ≥ 1 for forward-backward (New-M4 fix)"
        B, L, _ = x_aug.shape

        # Batched emissions (H2): [B, L, K]
        log_emit = self._torch_log_emission_batched(x_aug)

        log_pi = torch.log(self._pi.clamp(min=1e-30))                    # [K]
        log_A = torch.log(self._A.clamp(min=1e-30))                      # [K, K]

        # ── Forward (α_0 = π · b(x_0), α_t[k] = (Σ_j α_{t-1}[j] · A[j,k]) · b(x_t)[k]) ──
        # H1: list-stack pattern (cleaner autograd graph, no in-place into empty memory).
        log_alpha_list = [log_pi.unsqueeze(0) + log_emit[:, 0, :]]       # [B, K]
        for t in range(1, L):
            prev = log_alpha_list[-1].unsqueeze(-1)                       # [B, K_j, 1]
            log_trans = prev + log_A.unsqueeze(0)                         # [B, K_j, K_k]
            log_alpha_list.append(
                torch.logsumexp(log_trans, dim=1) + log_emit[:, t, :]
            )
        log_alpha = torch.stack(log_alpha_list, dim=1)                   # [B, L, K]

        # ── Backward (β_{L-1} = 1, β_t[k] = Σ_j A[k,j] · b(x_{t+1})[j] · β_{t+1}[j]) ──
        # Build right-to-left then reverse, again via list-stack.
        # Initial β_{L-1} = 0 in log-space.
        log_beta_rev = [torch.zeros(B, self.K, device=x_aug.device, dtype=x_aug.dtype)]
        for t in range(L - 2, -1, -1):
            nxt = log_beta_rev[-1]                                       # β_{t+1} [B, K_j]
            emit_beta = log_emit[:, t + 1, :] + nxt                       # [B, K_j]
            log_trans = log_A.unsqueeze(0) + emit_beta.unsqueeze(1)       # [B, K_k, K_j]
            log_beta_rev.append(torch.logsumexp(log_trans, dim=-1))       # [B, K_k]
        # log_beta_rev[0] = β_{L-1}, ..., log_beta_rev[L-1] = β_0  →  reverse
        log_beta = torch.stack(list(reversed(log_beta_rev)), dim=1)      # [B, L, K]

        # ── Posterior ──
        # H-1 fix: replace logsumexp-norm + exp(clamp) + re-norm pattern with
        # torch.softmax. softmax is internally max-stabilized (exp(x − x.max))
        # and renormalizes to exact row-sum=1 — bit-equivalent to the previous
        # three-step pattern but cleaner and avoids the clamp(-700, 0) edge
        # case where extreme log probabilities would skew normalization.
        gamma = torch.softmax(log_alpha + log_beta, dim=-1)
        return gamma

    # ──────────────────────────────────────────────────────────────
    # Emission-aware iterative rollout (PATCH 2 / §3.7)
    # ──────────────────────────────────────────────────────────────
    def rollout(
        self,
        gamma_T: torch.Tensor,
        x_window: torch.Tensor,
        H: int,
    ) -> torch.Tensor:
        """Iterative posterior rollout with emission likelihood reweighting.

        v2.0.9 dynamics-informed rollout (PATCH 2, NOT T^h-only):
            γ̃_s[k] = ( Σ_j γ̃_{s-1}[j] · A[j,k] ) · p(ô_{t+s} | z=k)
            γ̃_s ← γ̃_s / γ̃_s.sum()

        ô_{t+s} is predicted via first-order Taylor expansion using the most
        recent two observations:
            x̂_{t+s}  ≈ x_t + s · Δx_t
            Δ̂x_{t+s} ≈ Δx_t                (constant-velocity assumption)
            ô_{t+s}  = concat([x̂_{t+s}, Δ̂x_{t+s}])  ∈ ℝ^{V_aug}

        Args:
            gamma_T:  [B, K] posterior at the last observed timestep (t).
            x_window: [B, w, V_raw] recent raw observations, w ≥ 2 (used for Δx).
            H:        rollout horizon (≥ 1).
        Returns:
            gamma_rollout: [B, H, K]
        """
        assert gamma_T.dim() == 2 and gamma_T.shape[-1] == self.K
        assert x_window.dim() == 3 and x_window.shape[-1] == self.V_raw
        assert x_window.shape[1] >= 2, "x_window must contain at least 2 timesteps for Δx"
        assert H >= 1

        x_t = x_window[:, -1, :]                                          # [B, V_raw]
        delta_t = x_window[:, -1, :] - x_window[:, -2, :]                  # [B, V_raw]

        out = []
        gamma_prev = gamma_T                                              # [B, K]
        for s in range(1, H + 1):
            x_hat = x_t + s * delta_t                                     # [B, V_raw]
            x_aug_hat = torch.cat([x_hat, delta_t], dim=-1)               # [B, V_aug]
            log_emit = self._torch_log_emission(x_aug_hat)                # [B, K]
            # H-2 fix: stabilize subtract-max + defensive floor.
            # `log_emit - max` ∈ (-∞, 0], exp ∈ (0, 1]. The `.clamp(min=1e-30)`
            # protects the rare degenerate-HMM case where all states give
            # vanishingly small emission probabilities (max subtraction would
            # propagate NaN through exp(nan-nan) if max is also −∞).
            emit = torch.exp(log_emit - log_emit.max(dim=-1, keepdim=True).values)
            emit = emit.clamp(min=1e-30)
            # v2.3.1 NaN guard (M2.4 5/27 — 4-7 seasons partial-failure 진단 결과):
            # ill-conditioned HMM cov (cond ~6e3, eig_min ~1.5e-4) + outlier x (e.g. ILI=14.7σ)
            # → log_emit underflow to -inf for ALL states → max=-inf → -inf-(-inf)=NaN.
            # Fallback: uniform emit (degenerate emission → pure transition propagation).
            emit = torch.where(torch.isnan(emit) | torch.isinf(emit), torch.ones_like(emit), emit)
            gamma_trans = gamma_prev @ self._A                            # [B, K]
            gamma_new = gamma_trans * emit                                # [B, K]
            gamma_sum = gamma_new.sum(dim=-1, keepdim=True).clamp(min=1e-30)
            gamma_new = gamma_new / gamma_sum
            # Defensive: if still NaN (gamma_prev was NaN from a previous step), fallback to gamma_prev
            gamma_new = torch.where(torch.isnan(gamma_new) | torch.isinf(gamma_new), gamma_prev, gamma_new)
            out.append(gamma_new)
            gamma_prev = gamma_new
        return torch.stack(out, dim=1)                                    # [B, H, K]

    # ──────────────────────────────────────────────────────────────
    # rollout_gate — single-timestep gate + posterior + entropy confidence
    # ──────────────────────────────────────────────────────────────
    def rollout_gate(
        self,
        gamma_t: torch.Tensor,
        x_window: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the phase gate at a future horizon h with emission-aware rollout.

        Single-timestep operation (NOT a length-(L-1) sequence): starting from
        gamma_t [B, K] (= phase_post[:, -1, :] in CGForecaster Dataflow Step 6),
        rollout H = horizon steps and return the **last-step** gate / posterior
        / entropy-based confidence. EntropyAwareDecoder uses `confidence` to
        modulate the horizon-h prediction weight (PATCH 9).

        Args:
            gamma_t:  [B, K] starting posterior (e.g., phase_post[:, -1, :])
            x_window: [B, w, V_raw] recent raw observations (w ≥ 2)
            horizon:  forecast horizon h (≥ 1)
        Returns:
            gate_phase_h: [B, d_embed]  = sigmoid(γ̃_h · state_embeddings)
            gamma_t_h:    [B, K]        = γ̃_h
            confidence:   [B]           = 1 - H(γ̃_h) / log(K), ∈ [0, 1]
        """
        assert horizon >= 1
        gamma_rollout = self.rollout(gamma_t, x_window, H=horizon)        # [B, H, K]
        gamma_h = gamma_rollout[:, -1, :]                                  # [B, K]
        gate_phase_h = torch.sigmoid(gamma_h @ self.state_embeddings)      # [B, d_embed]
        # Entropy-based confidence: H_norm = -Σ γ log γ / log K, conf = 1 - H_norm
        eps = 1e-12
        H_ent = -(gamma_h * torch.log(gamma_h.clamp(min=eps))).sum(dim=-1)  # [B]
        log_K = math.log(float(self.K))
        confidence = (1.0 - H_ent / log_K).clamp(min=0.0, max=1.0)         # [B]
        return gate_phase_h, gamma_h, confidence

    # ──────────────────────────────────────────────────────────────
    # Stage 3 selective unfreeze (T-2, M2 — standard register_parameter API)
    # ──────────────────────────────────────────────────────────────
    def _unfreeze_for_stage3(self) -> int:
        """Convert _A and _means from buffers to Parameters for joint fine-tune.

        Per PATCH 11 (D.5.3), only A (transition) and μ_k (means) become
        trainable in Stage 3. π / Σ / cov_inv / log_det remain frozen — they
        are over-constrained (simplex, positive-definite) and typically
        regress under joint training.

        Post-review M2 fix: uses standard nn.Module API (`delattr` +
        `register_parameter`) instead of direct `_buffers` dict mutation, which
        is more robust to PyTorch internal changes.

        Idempotent: re-calling has no effect after the first conversion.

        ⚠️ Stage 3 entry sequence (caller responsibility, post-review New-M6):
            n_unfrozen = phase_module._unfreeze_for_stage3()
            # MANDATORY: rebuild optimizer to include the new Parameters.
            # The previously-built optimizer holds a snapshot of `model.parameters()`
            # at construction time; it does NOT auto-discover newly registered
            # Parameters. Failing to rebuild causes silent training failure:
            # `_A.grad` and `_means.grad` are populated but never applied.
            optimizer = torch.optim.AdamW(model.parameters(), lr=...)

        Returns:
            number of parameters now trainable (count of A and means entries).
        """
        if isinstance(self._A, nn.Parameter):
            return self._A.numel() + self._means.numel()

        A_val = self._A.detach().clone()
        means_val = self._means.detach().clone()
        # `delattr` routes through nn.Module.__delattr__ which correctly removes
        # the entry from `_buffers`. `register_parameter` is the canonical API
        # for adding a new Parameter under a given name.
        delattr(self, "_A")
        delattr(self, "_means")
        self.register_parameter("_A", nn.Parameter(A_val))
        self.register_parameter("_means", nn.Parameter(means_val))
        return self._A.numel() + self._means.numel()

    # ──────────────────────────────────────────────────────────────
    # forward — public entry point
    # ──────────────────────────────────────────────────────────────
    def forward(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_raw [B, L, V_raw] → (gate_phase [B, L-1, d_embed], phase_post [B, L-1, K]).

        Pipeline:
            x_aug      = _augment_features(x_raw)                # [B, L-1, V_aug]
            phase_post = _torch_forward_backward(x_aug)          # [B, L-1, K]
            gate_phase = sigmoid(phase_post @ state_embeddings)  # [B, L-1, d_embed]  (S-4)

        Note: L-1 alignment is intentional. CGForecaster (PATCH 10) handles
        x truncation (`x_truncated = x[:, 1:]`) to match this output length.
        """
        if not self._hmm_cached:
            raise RuntimeError(
                "PhaseModule.forward() called before _cache_hmm_torch(fitted_hmm). "
                "Stage 2 entry sequence (T-1): cache HMM, then build optimizer."
            )
        x_aug = self._augment_features(x_raw)                              # [B, L-1, V_aug]
        phase_post = self._torch_forward_backward(x_aug)                   # [B, L-1, K]
        gate_phase = torch.sigmoid(phase_post @ self.state_embeddings)     # [B, L-1, d_embed]

        if self.training:
            self._last_gamma = phase_post.detach()
        else:
            self._last_gamma = None
        return gate_phase, phase_post

    def extra_repr(self) -> str:
        return (
            f"V_raw={self.V_raw}, V_aug={self.V_aug}, K={self.K}, "
            f"d_embed={self.d_embed}, hmm_cached={self._hmm_cached}"
        )
