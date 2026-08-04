"""ContextGatedMambaBlock — pre-conv1d unified gate with low-rank projection.

M1.3 구현. PLAN v2.0.8a §3.2 + §3.8 + D.4.1 + D.4.4 canonical spec.

Architecture:
    context_vec [B, L, D=64]
        │
        ▼
    gate_proj (Sequential):
        Linear(D=64, r=8) + SiLU + Linear(r=8, ED=128)    ← low-rank default
        Linear(D=64, ED=128)                              ← full-rank ablation (gate_rank >= ED)
        │
        ▼ sigmoid
    gate [B, L, ED=128]
        │
        ▼ (pre-conv1d injection point in CGMambaBlock)
    CGMambaBlock(x, gate=gate)

Column modulation equivalence (§12.5):
    x_proj(gate ⊙ x_inner) = x_proj · diag(gate) · x_inner
    → context-conditional column selection of effective conv1d ∘ x_proj weight.

Init (§3.8, v2.0.7 A-3):
    Last linear: weight × 0.01, bias = gate_bias_init (default 2.0)
    → sigmoid(2.0) ≈ 0.8808 near-identity at init.
    weight × 0.01 (not = 0) preserves context gradient direction at first step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cg_mamba_block import CGMambaBlock
from src.utils.config import CGMambaConfig


class ContextGatedMambaBlock(nn.Module):
    """Mamba block with per-layer context gate projection (M1.3).

    Wraps M1.2's `CGMambaBlock` (composition, not inheritance). Inner Mamba logic
    is untouched — only adds `gate_proj` and routes the computed gate through
    `CGMambaBlock.forward(x, gate=...)`.

    Args:
        cfg:            CGMambaConfig (M1.2). Provides D, ED, N, R, r, gate_bias_init.
        gate_rank:      override cfg.gate_rank (None = use cfg). For ablation A1 r ∈ {4,8,16,full}.
        gate_bias_init: override cfg.gate_bias_init (None = use cfg). For ablation A-3 sensitivity.
        context_dim:    override input feature dim of gate_proj (None = use cfg.d_model).

    Forward:
        x:           [B, L, D]   pre-normed hidden state (residual is caller's responsibility)
        context_vec: [B, L, D]   gate_phase ⊙ gate_env, or None for vanilla path (ablation/M1.2 호환)

    Returns:
        y:           [B, L, D]
    """

    def __init__(
        self,
        cfg: CGMambaConfig,
        gate_rank: int | None = None,
        gate_bias_init: float | None = None,
        context_dim: int | None = None,
        disable_gate: bool = False,
    ):
        """
        F7 fix (PLAN D.4.1 spec compliance): added `disable_gate` constructor flag.
        Two equivalent vanilla paths now coexist:
          - `disable_gate=True` (PLAN spec, ablation-time decision)
          - `context_vec=None`  (runtime decision, M1.3 original)
        Both bypass gate_proj and call self.mamba(x, gate=None).
        """
        super().__init__()
        self.cfg = cfg
        ED = cfg.d_inner
        self.disable_gate = disable_gate

        # Resolve overrides (override > cfg default)
        self.gate_rank = gate_rank if gate_rank is not None else cfg.gate_rank
        bias_init = gate_bias_init if gate_bias_init is not None else cfg.gate_bias_init
        ctx_dim = context_dim if context_dim is not None else cfg.d_model

        # ── Vanilla Mamba core (re-uses M1.2 CGMambaBlock) ──
        # gate=None path = vanilla Mamba (bit-identical to M1.2)
        self.mamba = CGMambaBlock(cfg)

        # ── Gate projection (M1.3 new) ──
        # Low-rank (default): D → r → ED   via Sequential(Linear, SiLU, Linear)
        # Full-rank ablation: D → ED       via single Linear (gate_rank >= ED triggers)
        # PLAN §3.2: per-layer params = 1,672 for r=8
        #   = (D·r + r) + (r·ED + ED)  = (64·8+8) + (8·128+128) = 520 + 1,152
        if self.gate_rank >= ED:
            # Full-rank ablation (PLAN §7.4 A1): single Linear(D, ED)
            self.gate_proj = nn.Sequential(
                nn.Linear(ctx_dim, ED),                       # 8,320 params for D=64
            )
        else:
            # Low-rank bottleneck (default r=8)
            self.gate_proj = nn.Sequential(
                nn.Linear(ctx_dim, self.gate_rank),           # D → r
                nn.SiLU(),
                nn.Linear(self.gate_rank, ED),                # r → ED
            )

        # ── Near-identity init (§3.8, v2.0.7 A-3) ──
        self._init_gate_near_identity(bias_init)

        # ── Monitoring cache (Patch #29 Phase 1.5 pattern) ──
        self._last_gate: torch.Tensor | None = None

    def _init_gate_near_identity(self, bias_init: float) -> None:
        """Last Linear: weight × 0.01, bias = bias_init.

        At init step 0:
          h    = SiLU(W1 @ ctx + b1)             # non-zero in general
          gate = sigmoid(0.01·W2 @ h + b2_init)
               ≈ sigmoid(b2_init)                 # since 0.01·W2 ≈ 0
          ∴ initial gate ≈ sigmoid(2.0) ≈ 0.8808 (near-identity, 12% attenuation)

        Difference from CM-Mamba Patch #29 (weight = 0 exactly):
          - CM-Mamba: γ = 1.0 exactly at init → vanilla Mamba exactly
          - CG-Mamba: gate ≈ 0.88 at init → near-identity + context gradient
                      direction preserved from first gradient step.
        """
        last_linear = self.gate_proj[-1]
        with torch.no_grad():
            last_linear.weight.mul_(0.01)
            last_linear.bias.fill_(bias_init)

    def forward(
        self,
        x: torch.Tensor,                              # [B, L, D]
        context_vec: torch.Tensor | None = None,      # [B, L, D] or None
    ) -> torch.Tensor:                                # [B, L, D]
        """Forward pass with optional context gating.

        - context_vec is None    →  vanilla Mamba path (ablation/M1.2 호환)
        - context_vec [B, L, D]  →  gate = sigmoid(gate_proj(context_vec)),
                                    then CGMambaBlock(x, gate=gate)
        """
        if self.disable_gate or context_vec is None:
            # Vanilla path: either constructed with disable_gate=True (PLAN D.4.1
            # spec, A1 ablation) OR caller passes context_vec=None at runtime.
            # Both routes bit-identical to underlying CGMambaBlock(gate=None).
            self._last_gate = None
            return self.mamba(x, gate=None)

        # Shape sanity — verify context_vec matches x in batch+sequence so any
        # mismatch is reported HERE (not downstream as a gate-shape error in
        # CGMambaBlock, which is harder to trace back to the actual cause).
        B_x, L_x, _ = x.shape
        assert context_vec.shape[:2] == (B_x, L_x), (
            f"context_vec batch/seq {tuple(context_vec.shape[:2])} != "
            f"x {(B_x, L_x)} — caller passed mismatched tensors")
        ctx_dim_expected = self.gate_proj[0].in_features
        assert context_vec.shape[-1] == ctx_dim_expected, (
            f"context_vec last dim {context_vec.shape[-1]} != "
            f"gate_proj in_features {ctx_dim_expected}")

        # Compute per-position gate
        gate = torch.sigmoid(self.gate_proj(context_vec))     # [B, L, ED]

        # Cache for W&B monitoring + paper figures (PLAN §3.9, §6.5).
        # L4: skip cache in eval mode — saves memory + avoids stale state during
        # inference. Detach in train mode so gradient graph isn't held in cache.
        self._last_gate = gate.detach() if self.training else None

        return self.mamba(x, gate=gate)
