"""CGForecaster — end-to-end CG-Mamba forecaster (M1.6, v2.1.1).

PLAN v2.0.9 PATCH 10 (D.4.6) active spec + post-review M-1/M-4 fixes
+ v2.1.1 refactor (A-1 decoder pure tensor + C-1 return_intermediates + C-2 param helpers
+ D-1 confidence-per-horizon + D-2 phase transition KL).

Integrates four sub-modules:
    1. PhaseModule         (M1.4c): augmented [x_t, Δx_t] HMM → (gate_phase, phase_post)
    2. EnvModule           (M1.5):  env → gate_env autoencoder
    3. CGMambaEncoder      (M1.6):  context-gated Mamba stack
    4. EntropyAwareDecoder (M1.6):  emission-aware horizon-loop residual head (A-1 pure tensor)

Dataflow (PATCH 10 / D.4.6, R-3 deduplicated):
    [Step 1] PhaseModule(x[:, :, :V_hmm_raw])          → gate_phase [B,L-1,D], phase_post [B,L-1,K]
                                                           ↑ B1: drop num_patients (EB-2 multicoll.)
    [Step 2] EnvModule(env)                             → gate_env [B,L,D]
    [Step 3] (R-3 dedup: gate_phase는 PhaseModule 내부에서 sigmoid 적용 완료)
    [Step 4] x_truncated      = x[:, 1:, :]                  # [B, L-1, V_input=4]
             env_truncated_g  = gate_env[:, 1:, :]            # [B, L-1, D]   ← L-1 alignment
             context_vec      = gate_phase * env_truncated_g  # [B, L-1, D]   ← AND composition
    [Step 5] CGMambaEncoder(x_truncated, context_vec)   → fused [B, L-1, D]
             # M-1: phase_init = phase_post[:, 0, :] 는 M1.7 prefix_proj 통합 시 사용 예정.
             #      현재 M1.6 scope에서는 미사용 (linter F841 회피 — 추출 생략).
    [Step 6] gamma_last       = phase_post[:, -1, :]                          # [B, K]
             x_window         = x_phase[:, -W:, :]                             # [B, W, V_raw]
             # A-1: rollout을 CGForecaster가 직접 호출하여 결과를 텐서로 decoder에 전달
             gamma_all        = phase_module.rollout(gamma_last, x_window,
                                                       H=decoder.max_horizon)  # [B, max_h, K]
             last_value_norm  = x[:, -1, 0]                                    # [B]   R-2 normalized
             predictions      = decoder(fused, last_value_norm,
                                          gamma_all, phase_module.state_embeddings)
                                                           → [B, len(horizons)]

A-1 (v2.1.1 refactor): EntropyAwareDecoder는 phase_module 인자를 받지 않고 순수 텐서
(gamma_all, state_embeddings)만 받음. CGForecaster가 rollout + state_embeddings 추출
책임을 갖는다. 효과:
    - torch.jit.script / torch.compile 호환성 ↑
    - decoder 단위 테스트가 phase_module fixture 없이 텐서만으로 가능
    - PyTorch convention 부합 (nn.Module이 다른 nn.Module을 forward arg로 받지 않음)

C-1 + D-1 + D-2 (v2.1.1 intermediates): `forward(x, env, return_intermediates=True)`로
호출하면 (predictions, intermediates_dict) 반환. intermediates_dict는:
    - gate_phase, phase_post, context_vec, gamma_last, gamma_all, fused
    - eff_gate_per_horizon (D-1): [B, len(horizons)] eff_gate, eval mode에서도 접근
    - confidence_per_horizon (D-1): [B, len(horizons)] phase confidence
    - phase_transition_kl (D-2): [B] KL(γ_0 || γ_{H-1}) — outbreak transition probability proxy

R-2 (PLAN v2.0.9): `last_value_normalized` is in z-score space (ili_weighted_pct's
normalization). External caller (M1.7 trainer / inference pipeline) is responsible
for the inverse transform when reporting raw %wILI predictions.

B1 (V_input=4 → V_hmm_raw=3): WeeklyDataset.x_main has 4 features (ili_weighted_pct,
total_ili_count, num_providers, num_patients). PhaseModule uses only the first
V_hmm_raw=3 (drops num_patients per EB-2 multicollinearity r=0.952).

Stage 2 entry sequence (M-4 fix — mandatory):
    >>> model = CGForecaster(cfg)
    >>> hmm = load_fitted_hmm(Path('runs/m1_4_phase_dynamics_main/.../seed42/'))
    >>> model.prepare_for_stage2(hmm)           # HMM cache + Env decoder freeze (필수 묶음)
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=...)

Stage 3 (default SKIP, T-2):
    >>> if cfg.stage3_enabled:
    ...     n_unfrozen = model.phase_module._unfreeze_for_stage3()
    ...     # MANDATORY: rebuild optimizer to include newly trainable _A, _means
    ...     optimizer = torch.optim.AdamW(model.parameters(), lr=...)
"""
from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn

from src.models.backbone import CGMambaEncoder
from src.models.entropy_decoder import EntropyAwareDecoder
from src.models.env_module import EnvModule
from src.models.phase_module import PhaseModule
from src.utils.config import CGMambaConfig, ILI_TARGET_IDX


class CGForecaster(nn.Module):
    """End-to-end CG-Mamba forecaster (M1.6, v2.1.1 PATCH 10).

    Args:
        cfg: CGMambaConfig — provides V_hmm_raw, K_phase, d_model, d_inner, n_layers,
             gate_rank, gate_bias_init, env_input_dim, env_hidden_dim, horizons,
             rollout_window, main_input_dim, dropout, stage3_enabled.

    Forward signature:
        x   [B, L, main_input_dim=4]
        env [B, L, env_input_dim=2]
        return_intermediates: bool = False
        → predictions [B, len(cfg.horizons)] in normalized space (R-2)
        → (predictions, intermediates_dict) if return_intermediates=True
    """

    def __init__(self, cfg: CGMambaConfig):
        super().__init__()
        self.cfg = cfg
        # Defensive shape contracts at construction (fail fast on cfg mismatches)
        if cfg.main_input_dim < cfg.V_hmm_raw:
            raise RuntimeError(
                f"main_input_dim={cfg.main_input_dim} < V_hmm_raw={cfg.V_hmm_raw} — "
                f"PhaseModule expects to slice x[:, :, :V_hmm_raw]"
            )
        if cfg.rollout_window < 2:
            raise RuntimeError(
                f"rollout_window={cfg.rollout_window} < 2 — "
                f"Δx computation requires at least 2 timesteps"
            )

        # ── 4 sub-modules (S-7 instantiation pattern) ──
        self.phase_module = PhaseModule(
            V_raw=cfg.V_hmm_raw,
            K=cfg.K_phase,
            d_embed=cfg.d_model,
        )
        self.env_module = EnvModule(cfg)
        self.encoder = CGMambaEncoder(cfg)
        self.decoder = EntropyAwareDecoder(
            d_model=cfg.d_model,
            horizons=cfg.horizons,
            K=cfg.K_phase,
            gate_init=-1.1,
        )

    # ──────────────────────────────────────────────────────────────────
    # Stage 2 entry — mandatory bundle (M-4 fix)
    # ──────────────────────────────────────────────────────────────────
    def prepare_for_stage2(self, fitted_hmm) -> None:
        """Mandatory Stage 2 entry sequence (T-1 + PLAN §3.5 Env freeze).

        Must be called by the trainer BEFORE building the Stage 2 optimizer:
          1. `phase_module._cache_hmm_torch(fitted_hmm)` — caches HMM artifacts as
             register_buffers (T-1). Without this call, `forward()` raises
             RuntimeError on the first invocation.
          2. `env_module.freeze_decoder_for_stage2()` — freezes the EnvModule's
             reconstruction decoder (Stage 1 aux only; PLAN §3.5).

        Failing to call this method causes one of two silent-bug paths:
          - PhaseModule.forward raises RuntimeError → caught early ✓
          - EnvModule.decoder stays trainable → contaminates Stage 2 optimizer
            with auxiliary recon params (no error, but wrong gradient flow).

        v2.1.4 state_embeddings HMM-informed init (PLAN R-4 deviation):
          The default zeros init for state_embeddings preserves symmetry across
          the K rows — `gate_phase = sigmoid(gamma @ E)` becomes 0.5 for all
          timesteps, and gradient updates collinear-ize the K rows (verified
          numerically: Step 8 v1/v2 eff_gate std stagnant at 0.051 despite 10x
          LR scaling). Rationale + math: see M1_7_STATE_EMBED_FIX.md §1.2.

          Fix: enriched HMM identity (means, diag(cov), transition row) →
          deterministic random projection (seed=42) → D-dim embedding. Breaks
          symmetry across all D dims while preserving HMM-encoded phase
          identity. Trainable param count unchanged (192).

          Source: cached buffers (_means / _covs / _A) populated by the
          preceding `_cache_hmm_torch()` call. This routes the S-2
          regularization (Σ_k + reg_covar·I) into the init covariance, keeping
          the init exactly consistent with runtime emission likelihood.
        """
        self.phase_module._cache_hmm_torch(fitted_hmm)
        self.env_module.freeze_decoder_for_stage2()
        self._init_state_embeddings_from_cache()

    def _init_state_embeddings_from_cache(self, force: bool = False) -> None:
        """HMM enriched identity + random-projection init for state_embeddings.

        WARNING: resets state_embeddings to HMM-derived init. Must be called
        exactly once, via prepare_for_stage2(), BEFORE optimizer construction.
        Re-invocation overwrites trained values.

        v2.1.7 M-1 guard: subsequent calls are no-op unless force=True. This
        prevents silent regression when a checkpoint-resume path accidentally
        triggers prepare_for_stage2() twice (which would reset trained
        state_embeddings to HMM init and leave Adam's m/v stale).

        ⚠️ Note: `_state_embed_initialized` is a plain Python attribute (not a
        buffer/Parameter), so it does NOT persist across `load_state_dict`. A
        process that loads a checkpoint and re-calls `prepare_for_stage2(hmm)`
        will see the flag as False and re-initialize. Callers should either
        skip `prepare_for_stage2` after restoring a trained ckpt, or pass
        `force=False` (default) and rely on the guard once it has fired in
        the current process.

        Identity construction (per state k, 2·V_aug + K dims):
          [normalize(mu_k), normalize(diag(Sigma_k)), normalize(A[k, :])]
        Each statistic encodes a state-self property:
          - mu_k:        mean emission in augmented [x_t, Δx_t] space
          - diag(Σ_k):   per-feature variance (regularized by reg_covar via S-2)
          - A[k, :]:     outgoing transition tendency (self-loop + transitions)

        Source: cached buffers from `_cache_hmm_torch()`, which apply the S-2
        regularization (Σ_k + reg_covar·I). Using cached buffers ensures the
        init covariance matches the runtime emission likelihood exactly, and
        keeps everything on the model's device (no host↔device round-trip).

        Random projection (2·V_aug + K) → D with entries N(0, 1/V_id),
        deterministic via torch.Generator(seed=42). Element variance ≈ 1, ×
        scale=0.5 gives sigmoid input std 0.5 → gate_phase ∈ [0.378, 0.622]
        (non-saturating, sufficient phase differentiation).

        Step-count sanity (Step 8 v3 baseline, BS=32, 200 epochs):
          676 train windows / 32 batch ≈ 22 steps/epoch * 200 ≈ 4400 steps.
          LR=1e-5 → directional upper-bound displacement ~4.4e-2, relative ~5-15%.
          context_embed group wd=0.0 (optimizer.py) → no shrinkage interference.
        """
        if getattr(self, "_state_embed_initialized", False) and not force:
            print("[StateEmbed Init] SKIP — already initialized (use force=True to override).")
            return
        with torch.no_grad():
            # (0) Source from cached buffers (S-2 regularized, device-resident)
            mu = self.phase_module._means                       # [K, V_aug]
            sigma_sq = self.phase_module._covs.diagonal(        # [K, V_aug]
                dim1=-2, dim2=-1
            )
            trans = self.phase_module._A                        # [K, K]
            K = mu.shape[0]
            D = self.phase_module.state_embeddings.shape[1]
            device = mu.device

            # (1) Normalize each statistic block independently (decouples scale)
            def _normalize(t: torch.Tensor) -> torch.Tensor:
                c = t - t.mean(dim=0, keepdim=True)
                return c / c.std(dim=0, keepdim=True).clamp(min=1e-8)

            # (2) Concat → enriched identity [K, 2·V_aug + K]
            identity = torch.cat([
                _normalize(mu),                                 # [K, V_aug]
                _normalize(sigma_sq),                           # [K, V_aug]
                _normalize(trans),                              # [K, K]
            ], dim=-1)

            # (3) Deterministic random projection (V_id → D), then → device
            V_id = identity.shape[1]
            gen = torch.Generator().manual_seed(42)            # CPU generator
            proj = (torch.randn(V_id, D, generator=gen)
                    / math.sqrt(V_id)).to(device=device, dtype=identity.dtype)

            # (4) Inject (scale=0.5 → sigmoid std 0.5 → gate ∈ [0.38, 0.62])
            scale = 0.5
            self.phase_module.state_embeddings.data.copy_(
                (identity @ proj) * scale
            )

        # Symmetry-break verification (v2.1.7-B B-4: K-generalized, signed cos)
        # K=3 default reproduces 3 pairs (0,1)(0,2)(1,2); K=4 → 6 pairs;
        # K=5 → 10 pairs. Signed cos < 0.7 catches near-identical states;
        # anti-aligned (cos < 0) is mathematically distinguishable — gamma-weighted
        # sums of anti-aligned vectors still depend on gamma orientation.
        # (Initial v2.1.7-B used |cos| < 0.7 which over-restricted low-data
        # regimes like M2.4 3-5 seasons where HMM produces anti-aligned states.)
        E = self.phase_module.state_embeddings.data
        K_embed = E.shape[0]
        norms = E.norm(dim=1).tolist()
        cos_pairs: list[tuple[int, int, float]] = []
        for i in range(K_embed):
            for j in range(i + 1, K_embed):
                cos_ij = torch.cosine_similarity(E[i:i+1], E[j:j+1]).item()
                cos_pairs.append((i, j, cos_ij))
        cos_str = " ".join(f"({i},{j})={c:.4f}" for i, j, c in cos_pairs)
        print(
            f"[StateEmbed Init] shape={tuple(E.shape)}  "
            f"norms=[{', '.join(f'{n:.3f}' for n in norms)}]  "
            f"cos: {cos_str}"
        )
        violations = [(i, j, c) for i, j, c in cos_pairs if c >= 0.7]
        assert not violations, (
            f"Symmetry not broken — {len(violations)} pair(s) with cos ≥ 0.7: "
            f"{', '.join(f'({i},{j})={c:.4f}' for i, j, c in violations)}. "
            f"HMM statistics may be degenerate (K states near-identical). "
            f"Inspect hmm.means / hmm.covars / hmm.A."
        )
        self._state_embed_initialized = True

    # ──────────────────────────────────────────────────────────────────
    # Parameter budget helpers (C-2 fix)
    # ──────────────────────────────────────────────────────────────────
    def n_trainable_params(self) -> int:
        """Total trainable parameter count (PLAN §3.0 budget verification)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_group_summary(self) -> dict[str, int]:
        """Stage 2 4-param-group budget breakdown (PLAN §3.0 / §5.1 D.5.2).

        Returns a dict with per-group param counts plus 'total'. Use for:
          - regression guard: `assert summary['total'] == 115389` in tests
          - M1.7 optimizer construction: 4-group LR split (gate_proj / decoder_gate /
            context_embed / backbone) referenced against these counts
          - paper Table verification against PLAN §3.0 budget (~117K target)

        Note: 'encoder_backbone' includes input_proj + n_layers × Mamba blocks +
        RMSNorms, but excludes the per-layer gate_proj (counted separately).
        """
        state_emb = self.phase_module.state_embeddings.numel()
        gate_proj = sum(
            p.numel()
            for layer in self.encoder.layers
            for p in layer.gate_proj.parameters()
        )
        encoder_total = sum(p.numel() for p in self.encoder.parameters())
        encoder_backbone = encoder_total - gate_proj
        decoder = sum(p.numel() for p in self.decoder.parameters())
        env_encoder = self.env_module.encoder_param_count()
        env_decoder = self.env_module.decoder_param_count()  # Stage 1 aux only
        # HMM buffers (frozen, not counted in trainable budget)
        hmm_buffer_count = sum(
            b.numel() for b in self.phase_module.buffers()
        )

        # Total trainable (excludes Env decoder if frozen, includes if not)
        env_dec_train = sum(
            p.numel()
            for p in self.env_module.decoder.parameters()
            if p.requires_grad
        )
        trainable_total = (
            state_emb + gate_proj + encoder_backbone + decoder + env_encoder + env_dec_train
        )
        return {
            "phase_state_embed": state_emb,
            "gate_proj": gate_proj,
            "encoder_backbone": encoder_backbone,
            "decoder": decoder,
            "env_encoder": env_encoder,
            "env_decoder_trainable": env_dec_train,
            "env_decoder_total": env_decoder,
            "hmm_buffers_frozen": hmm_buffer_count,
            "trainable_total": trainable_total,
        }

    # ──────────────────────────────────────────────────────────────────
    # forward — Dataflow Step 1~6 (PATCH 10)
    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,                          # [B, L, main_input_dim]
        env: torch.Tensor,                        # [B, L, env_input_dim]
        return_intermediates: bool = False,       # C-1: 논문 figure / W&B logging용
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """Per-Dataflow Step 1~6 (PATCH 10). See module docstring.

        Args:
            x:   [B, L, main_input_dim]
            env: [B, L, env_input_dim]
            return_intermediates: if True, return (predictions, dict of intermediate tensors).
                                  intermediates dict keys:
                                    - gate_phase [B, L-1, D]            Phase gate
                                    - phase_post [B, L-1, K]            HMM soft posteriors
                                    - context_vec [B, L-1, D]           AND composition
                                    - gamma_last [B, K]                 last-step posterior
                                    - gamma_all [B, max_horizon, K]     rollout posteriors
                                    - fused [B, L-1, D]                 encoder output
                                    - eff_gate_per_horizon [B, len(h)]  effective gate per h
                                    - confidence_per_horizon [B, len(h)] phase confidence per h
                                    - phase_transition_kl [B]           KL(γ_0 || γ_{H-1}), D-2
        Returns:
            predictions [B, len(horizons)] if return_intermediates=False
            (predictions, intermediates) tuple if return_intermediates=True
        """
        # ── Defensive shape validation (RuntimeError, survives `python -O`) ──
        if x.dim() != 3 or x.shape[-1] != self.cfg.main_input_dim:
            raise RuntimeError(
                f"x expected [B, L, main_input_dim={self.cfg.main_input_dim}], "
                f"got {tuple(x.shape)}"
            )
        if env.dim() != 3 or env.shape[-1] != self.cfg.env_input_dim:
            raise RuntimeError(
                f"env expected [B, L, env_input_dim={self.cfg.env_input_dim}], "
                f"got {tuple(env.shape)}"
            )
        if x.shape[:2] != env.shape[:2]:
            raise RuntimeError(
                f"x and env batch/seq mismatch: x {tuple(x.shape[:2])} vs "
                f"env {tuple(env.shape[:2])}"
            )
        if x.shape[1] < 2:
            raise RuntimeError(
                f"L={x.shape[1]} < 2 — PhaseModule augmentation requires L ≥ 2"
            )

        # Step 1: PhaseModule — uses first V_hmm_raw channels (B1)
        x_phase = x[:, :, :self.cfg.V_hmm_raw]                         # [B, L, V_raw]
        gate_phase, phase_post = self.phase_module(x_phase)            # [B,L-1,D], [B,L-1,K]

        # Step 2: EnvModule
        gate_env = self.env_module(env)                                # [B, L, D]

        # Step 3: (R-3 dedup — gate_phase는 PhaseModule이 sigmoid 적용 완료)

        # Step 4: L-1 alignment + AND composition
        x_truncated = x[:, 1:, :]                                       # [B, L-1, V_input]
        env_truncated_g = gate_env[:, 1:, :]                            # [B, L-1, D]
        context_vec = gate_phase * env_truncated_g                      # [B, L-1, D]

        # Step 5: CGMambaEncoder
        # M-1: phase_init = phase_post[:, 0, :] 는 M1.7 prefix_proj 통합 시 사용 예정.
        #      현재 M1.6 scope에서는 미사용이므로 변수 추출 생략 (linter F841 회피).
        fused = self.encoder(x_truncated, context_vec=context_vec)     # [B, L-1, D]

        # Step 6: EntropyAwareDecoder (A-1: pure tensor I/O)
        gamma_last = phase_post[:, -1, :]                              # [B, K]
        # x_window: handle L < rollout_window edge case (L-3 test #13)
        W = min(self.cfg.rollout_window, x_phase.shape[1])
        x_window = x_phase[:, -W:, :]                                  # [B, W, V_raw]
        last_value_normalized = x[:, -1, ILI_TARGET_IDX]               # [B]  (R-2 normalized; v2.1.7-B B-3 constant)

        # A-1: pre-compute rollout here (instead of inside decoder), pass tensors to decoder.
        gamma_all = self.phase_module.rollout(                          # [B, max_horizon, K]
            gamma_last, x_window, H=self.decoder.max_horizon,
        )

        predictions = self.decoder(
            encoder_out=fused,
            last_value_normalized=last_value_normalized,
            gamma_all=gamma_all,
            state_embeddings=self.phase_module.state_embeddings,
        )                                                              # [B, len(horizons)]

        # C-1 + D-1 + D-2: intermediate tensors for figures / W&B / clinical signals
        if return_intermediates:
            intermediates = self._compute_intermediates(
                gate_phase=gate_phase,
                phase_post=phase_post,
                context_vec=context_vec,
                gamma_last=gamma_last,
                gamma_all=gamma_all,
                fused=fused,
            )
            return predictions, intermediates

        return predictions

    def _compute_intermediates(
        self,
        gate_phase: torch.Tensor,
        phase_post: torch.Tensor,
        context_vec: torch.Tensor,
        gamma_last: torch.Tensor,
        gamma_all: torch.Tensor,
        fused: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build intermediates dict for return_intermediates=True (C-1 + D-1 + D-2).

        Computes per-horizon confidence (D-1) and phase transition KL (D-2) from
        gamma_all directly — no second decoder pass, no extra rollout.
        """
        # D-1: confidence_per_horizon — eval mode에서도 사용 가능 (decoder._last_* 는 train only)
        eps = 1e-12
        confidences = []
        eff_gates = []
        for i, h in enumerate(self.decoder.horizons):
            gamma_h = gamma_all[:, h - 1, :]                                   # [B, K]
            conf_h = EntropyAwareDecoder._compute_confidence(
                gamma_h, self.decoder.K
            )                                                                   # [B]
            confidences.append(conf_h)
            # Reconstruct eff_gate (mirror EntropyAwareDecoder.forward LOGIC-1)
            gate_phase_h = torch.sigmoid(
                gamma_h @ self.phase_module.state_embeddings
            )                                                                   # [B, D]
            gate_strength_h = gate_phase_h.mean(dim=-1)                        # [B]
            eff_gate_h = conf_h * gate_strength_h + (1.0 - conf_h) * 1.0       # [B]
            eff_gates.append(eff_gate_h)

        # D-2: phase transition KL — γ_first vs γ_last in rollout
        # KL(γ_0 || γ_{H-1}) = Σ γ_0 · log(γ_0 / γ_{H-1})
        gamma_first = gamma_all[:, 0, :]                                       # [B, K]
        gamma_lastH = gamma_all[:, -1, :]                                      # [B, K]
        kl_div = (
            gamma_first
            * (
                torch.log(gamma_first.clamp(min=eps))
                - torch.log(gamma_lastH.clamp(min=eps))
            )
        ).sum(dim=-1)                                                          # [B]

        return {
            "gate_phase": gate_phase,
            "phase_post": phase_post,
            "context_vec": context_vec,
            "gamma_last": gamma_last,
            "gamma_all": gamma_all,
            "fused": fused,
            "eff_gate_per_horizon": torch.stack(eff_gates, dim=-1),            # [B, len(horizons)]
            "confidence_per_horizon": torch.stack(confidences, dim=-1),         # [B, len(horizons)]
            "phase_transition_kl": kl_div,                                      # [B]
        }

    def extra_repr(self) -> str:
        return (
            f"V_input={self.cfg.main_input_dim}, V_hmm_raw={self.cfg.V_hmm_raw}, "
            f"K={self.cfg.K_phase}, D={self.cfg.d_model}, "
            f"horizons={self.cfg.horizons}, rollout_W={self.cfg.rollout_window}"
        )
