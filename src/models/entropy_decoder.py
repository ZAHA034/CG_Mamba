"""EntropyAwareDecoder — emission-aware horizon forecasting (M1.6, v2.1.1).

PLAN v2.0.9 PATCH 9 active spec (D.4.5.v2.0.9) + post-review H-1/M-3 fixes
+ v2.1.1 refactor A-1/A-2 (pure tensor I/O + LOGIC-1 docstring 보강).

Role:
    Per-horizon residual correction over `last_value_normalized` with
    entropy-confidence-gated context modulation. Operates on the encoder's
    final-timestep representation (h_last) and the pre-computed rollout
    posteriors `gamma_all`.

Architecture (LOGIC-1):
    h_last = encoder_out[:, -1, :]                       # [B, D]
    correction = Linear(D, len(horizons))(h_last)         # [B, len(horizons)]
    alpha = sigmoid(gate)                                 # scalar, init ≈ 0.25 (gate_init=-1.1)
    for i, h in enumerate(horizons):
        gamma_h         = gamma_all[:, h-1, :]                              # [B, K]
        gate_phase_h    = sigmoid(gamma_h @ state_embeddings)               # [B, D]
        conf_h          = 1 - H(γ_h) / log(K)                               # [B] in [0,1]
        gate_strength_h = gate_phase_h.mean(-1)                             # [B] ∈ (0, 1)
        eff_gate_h      = conf_h * gate_strength_h + (1-conf_h) * 1.0       # [B]
        pred_h          = last_value_normalized + alpha * correction[:,i] * eff_gate_h
    return stack(preds, dim=-1)                          # [B, len(horizons)]

LOGIC-1 (eff_gate 설계 의도 — A-2 보강 docstring):
    eff_gate is **phase-selective MODULATION**, not correction-MAGNITUDE attenuation.

      conf_h = 1 (peaked γ → 모델이 phase k에 강한 확신):
          eff_gate_h = gate_strength_h ∈ (0, 1)
          correction이 phase k에 conditioned된 patterns (state_embeddings@gamma_h)에
          따라 shape됨. 일견 correction이 "감쇠"되는 것처럼 보이지만, 사실은
          phase가 correction의 방향(direction)을 조절하는 것.

      conf_h = 0 (uniform γ → 모델이 어느 phase인지 모름):
          eff_gate_h = 1.0
          phase 정보는 사용 불가하므로 우회. naive (phase-agnostic) correction.

    핵심: α scalar (별도 학습 parameter)가 correction의 절대 크기를 제어하고,
    eff_gate는 phase가 correction에 미치는 영향력의 SELECTIVITY를 결정한다.
    "확실할 때 더 강한 correction"이 아니라 "확실할 때 phase가 correction을 shape".
    JBHI reviewer가 흔히 묻는 "왜 eff_gate < 1 when confident?"에 대한 답이 이것.

    수학적 해석:
        pred_h = last_value + α · correction · eff_gate
              = last_value + α · correction · [conf · gate_strength + (1-conf) · 1]
              = last_value + α · correction · 1                    (uniform γ, conf=0)
              = last_value + α · correction · gate_strength        (peaked γ, conf=1)
    즉 confident 시 phase-modulated correction, uncertain 시 phase-agnostic correction.

Post-review fix H-1 (O(H²) → O(H)):
    Original spec called `phase_module.rollout_gate(horizon=h)` for each h ∈ 1..H, and
    rollout_gate internally calls rollout(H=h) which iterates 1..h. Total emission
    compute = 1+2+...+H = O(H²) (10 emissions for H=4).
    v2.1.0 fixed: caller (CGForecaster) calls `rollout(H=max_horizon)` ONCE → [B, max_h, K],
    passes the full tensor to this decoder. Decoder computes confidence + sigmoid
    directly → O(H) = 4 emissions. ~60% compute savings at H=4.

Post-review fix M-3 (proj dim = len(horizons)):
    For ragged horizons (e.g., (1,2,4,8)), max(horizons)=8 leaves indices 3/5/6/7 as
    dead weights. Fixed: `Linear(d_model, len(horizons))` with `enumerate(horizons)`.

v2.1.1 refactor A-1 (pure tensor I/O):
    Original v2.1.0 spec: `forward(..., phase_module, gamma_last, x_window)` —
    decoder took a PhaseModule reference, called rollout() internally, and accessed
    state_embeddings. This violated PyTorch convention (nn.Module receiving another
    nn.Module as a forward arg), broke torch.jit.script compatibility, and required
    a PhaseModule fixture for every decoder unit test.
    v2.1.1: caller pre-computes `gamma_all = phase_module.rollout(...)` and passes
    it (plus `state_embeddings` tensor) to the decoder. forward is now pure
    tensor-in / tensor-out, scriptable, and unit-testable without mocking.

R-2 (PLAN v2.0.9): `last_value_normalized` is in the normalized space (z-score for
ili_weighted_pct in WeeklyDataset). External caller is responsible for the inverse
transform when reporting raw %wILI predictions.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class EntropyAwareDecoder(nn.Module):
    """v2.1.1 PATCH 9 active spec — emission-aware horizon-loop decoder (pure tensor I/O).

    Args:
        d_model:   encoder output dim (= cfg.d_model). default 64.
        horizons:  forecast horizons in weeks (CDC FluSight: (1, 2, 3, 4)).
                   Ragged grids OK (e.g., (1, 2, 4, 8)).
        K:         HMM state count, for entropy normalization. default 3.
        gate_init: initial sigmoid logit for alpha (default -1.1 → α ≈ 0.25
                   soft correction start, M1.3 near-identity philosophy).
    """

    def __init__(
        self,
        d_model: int = 64,
        horizons: tuple[int, ...] = (1, 2, 3, 4),
        K: int = 3,
        gate_init: float = -1.1,
    ):
        super().__init__()
        assert d_model >= 1, f"d_model={d_model} must be ≥ 1"
        assert K >= 1, f"K={K} must be ≥ 1"
        assert len(horizons) >= 1 and all(int(h) >= 1 for h in horizons), (
            f"horizons must be a non-empty tuple of positive ints, got {horizons}"
        )

        self.horizons = tuple(int(h) for h in horizons)
        self.max_horizon = max(self.horizons)
        self.K = int(K)
        self.d_model = int(d_model)

        # M-3 fix: proj dim = len(horizons). Each horizon gets exactly one slot.
        self.proj = nn.Linear(d_model, len(self.horizons))
        # gate_init=-1.1 → sigmoid(-1.1) ≈ 0.2507 (soft correction at start)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # Monitoring cache (train-mode only, eval-mode None — mirrors
        # ContextGatedMambaBlock._last_gate + PhaseModule._last_gamma).
        # For eval-mode access, CGForecaster.forward(..., return_intermediates=True).
        self._last_eff_gate: Optional[torch.Tensor] = None
        self._last_confidence: Optional[torch.Tensor] = None

    @staticmethod
    def _compute_confidence(gamma_h: torch.Tensor, K: int) -> torch.Tensor:
        """Entropy-based confidence ∈ [0, 1] for a single-step posterior.

        Mirrors PhaseModule.rollout_gate's confidence formula:
            confidence = 1 - H(γ) / log(K)
        Peaked γ (one state ≈ 1) → H ≈ 0 → confidence ≈ 1.
        Uniform γ (all 1/K)      → H = log(K) → confidence = 0.

        Args:
            gamma_h: [B, K] posterior, rows sum to 1.
            K:       state count (passed in to avoid module attribute lookup).
        Returns:
            [B] confidence values clamped to [0, 1].
        """
        eps = 1e-12
        H_ent = -(gamma_h * torch.log(gamma_h.clamp(min=eps))).sum(dim=-1)  # [B]
        log_K = math.log(float(K))
        return (1.0 - H_ent / log_K).clamp(min=0.0, max=1.0)

    def forward(
        self,
        encoder_out: torch.Tensor,                  # [B, L', D]   from CGMambaEncoder
        last_value_normalized: torch.Tensor,        # [B]            R-2 normalized space
        gamma_all: torch.Tensor,                    # [B, max_horizon, K]  A-1: pre-computed by caller
        state_embeddings: torch.Tensor,             # [K, D]          A-1: pure tensor (no module ref)
    ) -> torch.Tensor:
        """Compute per-horizon predictions (pure tensor I/O, A-1 refactor).

        Args:
            encoder_out:          [B, L', D]            from CGMambaEncoder
            last_value_normalized:[B]                    z-score space (R-2)
            gamma_all:            [B, max_horizon, K]   pre-computed rollout posteriors.
                                                          Caller (CGForecaster) computes via
                                                          `phase_module.rollout(gamma_last, x_window, H=max_horizon)`.
            state_embeddings:     [K, D]                phase state embedding tensor.
                                                          Caller passes `phase_module.state_embeddings`.

        Returns:
            predictions: [B, len(horizons)] in normalized space (caller denormalizes).
        """
        # ── Input shape validation (RuntimeError, survives `python -O`) ──
        if encoder_out.dim() != 3:
            raise RuntimeError(
                f"encoder_out expected 3D [B, L, D], got shape {tuple(encoder_out.shape)}"
            )
        if encoder_out.shape[-1] != self.d_model:
            raise RuntimeError(
                f"encoder_out last dim {encoder_out.shape[-1]} != d_model={self.d_model}"
            )
        if last_value_normalized.dim() != 1:
            raise RuntimeError(
                f"last_value_normalized expected 1D [B], "
                f"got shape {tuple(last_value_normalized.shape)}"
            )
        if gamma_all.dim() != 3 or gamma_all.shape[-1] != self.K:
            raise RuntimeError(
                f"gamma_all expected [B, max_horizon, K={self.K}], "
                f"got shape {tuple(gamma_all.shape)}"
            )
        if gamma_all.shape[1] < self.max_horizon:
            raise RuntimeError(
                f"gamma_all rollout length {gamma_all.shape[1]} < "
                f"required max_horizon={self.max_horizon}"
            )
        if state_embeddings.dim() != 2 or state_embeddings.shape != (self.K, self.d_model):
            raise RuntimeError(
                f"state_embeddings expected [K={self.K}, D={self.d_model}], "
                f"got shape {tuple(state_embeddings.shape)}"
            )

        B = encoder_out.shape[0]
        if last_value_normalized.shape[0] != B or gamma_all.shape[0] != B:
            raise RuntimeError(
                f"Batch size mismatch: encoder_out B={B}, "
                f"last_value B={last_value_normalized.shape[0]}, "
                f"gamma_all B={gamma_all.shape[0]}"
            )

        # ── Core computation ──
        h_last = encoder_out[:, -1, :]              # [B, D]
        correction = self.proj(h_last)              # [B, len(horizons)]
        alpha = torch.sigmoid(self.gate)            # scalar ≈ 0.25 init

        preds = []
        eff_gates = []
        confidences = []
        for i, h in enumerate(self.horizons):
            gamma_h = gamma_all[:, h - 1, :]                                       # [B, K]
            gate_phase_h = torch.sigmoid(gamma_h @ state_embeddings)               # [B, D]
            conf_h = self._compute_confidence(gamma_h, self.K)                     # [B]
            gate_strength_h = gate_phase_h.mean(dim=-1)                            # [B]
            eff_gate_h = conf_h * gate_strength_h + (1.0 - conf_h) * 1.0           # [B]  LOGIC-1
            eff_gates.append(eff_gate_h)
            confidences.append(conf_h)
            preds.append(
                last_value_normalized + alpha * correction[:, i] * eff_gate_h
            )                                                                       # [B]

        # Monitoring (train-mode only, detached, eval → None).
        # For eval-mode intermediate access, use CGForecaster.forward(return_intermediates=True).
        if self.training:
            self._last_eff_gate = torch.stack(eff_gates, dim=-1).detach()           # [B, len(horizons)]
            self._last_confidence = torch.stack(confidences, dim=-1).detach()       # [B, len(horizons)]
        else:
            self._last_eff_gate = None
            self._last_confidence = None

        return torch.stack(preds, dim=-1)                                          # [B, len(horizons)]

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, horizons={self.horizons}, "
            f"max_horizon={self.max_horizon}, K={self.K}"
        )
