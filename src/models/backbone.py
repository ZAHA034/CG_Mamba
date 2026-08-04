"""CG-Mamba encoders — CGMambaBackbone (M1.2 vanilla) + CGMambaEncoder (M1.6 통합).

Two parallel encoder classes coexist:

  CGMambaBackbone (M1.2 vanilla):
    n_layers × CGMambaBlock with optional per-layer `gates: list[Tensor]`.
    Used for vanilla baseline + sanity tests + ablation `disable_gate=True`.

  CGMambaEncoder (M1.6, v2.0.9, 2026-05-19):
    n_layers × ContextGatedMambaBlock with shared `context_vec: [B, L, D]`.
    Each layer computes its own gate internally via gate_proj (M1.3 convention).
    Used in CGForecaster end-to-end forward path (PATCH 10 / D.4.6).

Shared parts (both encoders):
    input_proj(main_input_dim → d_model) + (n_layers + 1) × RMSNorm + dropout.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.cg_mamba_block import CGMambaBlock
from src.models.context_gate import ContextGatedMambaBlock
from src.models.rmsnorm import RMSNorm
from src.utils.config import CGMambaConfig


class CGMambaBackbone(nn.Module):
    """Input MLP → n_layers × CGMambaBlock (+ residual + RMSNorm) → final RMSNorm.

    F6 fix (v2.0.8b D.4.4 spec): RMSNorm replaces LayerNorm — matches PLAN's
    canonical encoder spec. Effect on M1.2 param count:
      LayerNorm × (n_layers + 1) = 2D × 4 = 512   (old)
      RMSNorm   × (n_layers + 1) =  D × 4 = 256   (new)
    M1.2 vanilla total: 108,033 → 107,777 (depth=3, n_layers=3, D=64).

    Forward:
        x      [B, L, main_input_dim]      — standardized ILI features
        gates  list of n_layers × [B, L, ED] | None  — per-layer pre-conv1d gate
    Output:
        h_seq  [B, L, D]                    — encoded sequence
    """

    def __init__(self, cfg: CGMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.main_input_dim, cfg.d_model)
        self.layers = nn.ModuleList([CGMambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norms = nn.ModuleList([RMSNorm(cfg.d_model)
                                    for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                                  # [B, L, main_input_dim]
        gates: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:                                    # [B, L, D]
        h = self.input_proj(x)                            # [B, L, D]
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            gate_i = gates[i] if gates is not None else None
            h_pre = norm(h)
            h_layer = layer(h_pre, gate=gate_i)
            h = h + self.dropout(h_layer)                 # residual
        h = self.final_norm(h)
        return h


class OneStepRegressionHead(nn.Module):
    """Minimal head for M1.2 sanity: take last timestep → predict y_{t+1} target."""

    def __init__(self, cfg: CGMambaConfig):
        super().__init__()
        self.linear = nn.Linear(cfg.d_model, 1)

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        # h_seq: [B, L, D]
        h_last = h_seq[:, -1, :]                           # [B, D]
        y = self.linear(h_last).squeeze(-1)                # [B]
        return y


class M1_2_VanillaCGMamba(nn.Module):
    """End-to-end model for M1.2 sanity check."""

    def __init__(self, cfg: CGMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = CGMambaBackbone(cfg)
        self.head = OneStepRegressionHead(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# CGMambaEncoder (M1.6, v2.0.9) — Full integration encoder
# ─────────────────────────────────────────────────────────────────────────────


class CGMambaEncoder(nn.Module):
    """Context-gated Mamba encoder — n_layers × ContextGatedMambaBlock (M1.6).

    Replaces the M1.2 `CGMambaBackbone`'s `gates: list[Tensor]` pattern with a
    single shared `context_vec: Tensor` (M1.3 `ContextGatedMambaBlock` convention).
    Each layer computes its own gate internally from `context_vec` via gate_proj.

    Architecture:
        input_proj(main_input_dim → D)
            ↓
        for layer in n_layers:
            h_pre = norm(h)
            h_layer = ContextGatedMambaBlock(h_pre, context_vec=context_vec)
            h = h + dropout(h_layer)                   ← residual
            ↓
        final_norm

    Forward:
        x:           [B, L, main_input_dim]   standardized ILI features
        context_vec: [B, L, D] | None         shared gate input (M1.3 signature)

    Output:
        h_seq:       [B, L, D]                encoded sequence
    """

    def __init__(self, cfg: CGMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.main_input_dim, cfg.d_model)
        self.layers = nn.ModuleList(
            [ContextGatedMambaBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norms = nn.ModuleList(
            [RMSNorm(cfg.d_model) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                                  # [B, L, main_input_dim]
        context_vec: Optional[torch.Tensor] = None,       # [B, L, D] | None
    ) -> torch.Tensor:                                    # [B, L, D]
        if context_vec is not None:
            # Shape sanity — fail fast with explicit message before downstream.
            # RuntimeError (not assert) survives `python -O`, matching PhaseModule (New-L5).
            B_x, L_x, _ = x.shape
            if context_vec.shape[:2] != (B_x, L_x):
                raise RuntimeError(
                    f"context_vec batch/seq {tuple(context_vec.shape[:2])} != "
                    f"x {(B_x, L_x)} — caller passed mismatched tensors"
                )
            if context_vec.shape[-1] != self.cfg.d_model:
                raise RuntimeError(
                    f"context_vec last dim {context_vec.shape[-1]} != "
                    f"d_model={self.cfg.d_model}"
                )

        h = self.input_proj(x)                            # [B, L, D]
        for layer, norm in zip(self.layers, self.norms):
            h_pre = norm(h)
            h_layer = layer(h_pre, context_vec=context_vec)   # M1.3 signature
            h = h + self.dropout(h_layer)                 # residual
        return self.final_norm(h)

    def extra_repr(self) -> str:
        return (
            f"main_input_dim={self.cfg.main_input_dim}, d_model={self.cfg.d_model}, "
            f"n_layers={self.cfg.n_layers}"
        )
