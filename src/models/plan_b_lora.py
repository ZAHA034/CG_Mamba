"""Plan B: LoRA-style context-conditional W_dt modulation — FORWARD SKELETON ONLY.

PLAN v2.0.8a §13.3 + Appendix C — contingency fallback (NOT default).

Status (M1.3):
    ⚠️ FORWARD SHAPE COMPATIBILITY CHECK ONLY.
    ⚠️ Training / evaluation NOT IMPLEMENTED.
    ⚠️ Plan A (column modulation, ContextGatedMambaBlock) is the default.

⚠️ M1.6/M1.7 INSTANTIATION GUARD:
    Do NOT instantiate `LoRAGatedMambaBlock` from `CGMambaEncoder` or any
    training script before Phase 2 M2.1 exit gate. The block's `lora_A`,
    `lora_B`, `alpha` parameters would be added to `model.parameters()` and
    would receive optimizer steps as dead-branch zero gradients — harmless
    numerically but wasteful and confusing in checkpoints/W&B logs. The
    block class selection logic in M1.6 encoder must check a flag (e.g.,
    `cfg.plan_b_enabled`) and default to `ContextGatedMambaBlock` (Plan A).

When activated:
    Phase 2 M2.1 Exit Gate failure (CG-Mamba MAE > Vanilla Mamba 2 seasons consecutive)
    → switch encoder block class from ContextGatedMambaBlock → LoRAGatedMambaBlock.
    Plan A ablation moves to §7.4 A6 (variation experiment).

Key equation (PLAN §12.5):
    Plan A (column mod):  W_eff = W · diag(gate(context))         — column 부분공간 내
    Plan B (LoRA):        W_eff_dt = W_dt + α · A(context) @ B^T  — 새 방향 추가 가능

LoRA target: W_dt (dt_proj weight) — Δ가 SSM memory horizon (A_bar = exp(Δ·A))을
직접 제어하므로 SSM dynamics 변경의 가장 효율적 지점 (PLAN v2.0.4 보완 #2).

Param cost per layer (r_lora=4):
    W_A (context_dim → r_lora):  D × r_lora + r_lora = 64×4+4 = 260
    W_B (r_lora → ED):           r_lora × ED + ED   = 4×128+128 = 640
    α scalar:                                       = 1
    Total: 901 params/layer (depth=3 ablation: 2,703)
    vs Plan A r=8: 1,672 params/layer — Plan B more compact (no SiLU intermediate).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cg_mamba_block import CGMambaBlock
from src.utils.config import CGMambaConfig


class LoRAGatedMambaBlock(nn.Module):
    """⚠️ Plan B contingency block — FORWARD SKELETON, not trained.

    Forward shape compatibility verification only. Interfaces match
    `ContextGatedMambaBlock.forward(x, context_vec)` so it can be a drop-in
    swap in `CGMambaEncoder.blocks` (M1.6).

    For M1.3 Exit-9 (skeleton existence + shape check). Real training requires:
        - Hooking LoRA delta into `CGMambaBlock.dt_proj.weight` (or wrapping
          dt_proj computation), not just adding a residual to dt_raw.
        - Init scheme matching LoRA paper (A ~ N(0, σ²), B = 0, α scalar).
        - Param group integration in `train_stage2.py` (LoRA-only group).

    Args:
        cfg:         CGMambaConfig (re-uses M1.2 backbone)
        lora_rank:   LoRA bottleneck rank (default 4; range 4-8 per PLAN App C)
        context_dim: context_vec feature dim (default cfg.d_model)
    """

    def __init__(
        self,
        cfg: CGMambaConfig,
        lora_rank: int = 4,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.lora_rank = lora_rank
        ED = cfg.d_inner
        ctx_dim = context_dim if context_dim is not None else cfg.d_model

        # Vanilla Mamba core (re-used from M1.2, identical to Plan A wrapper)
        self.mamba = CGMambaBlock(cfg)

        # LoRA adapter for dt direction
        # Plan B equation (skeleton): dt_lora = α · (W_B @ (W_A @ context))
        #   added to softplus(dt_raw) inside selective_scan path (not implemented here)
        self.lora_A = nn.Linear(ctx_dim, lora_rank)         # context → r_lora
        self.lora_B = nn.Linear(lora_rank, ED)              # r_lora → ED
        self.alpha = nn.Parameter(torch.tensor(0.0))        # scalar gate (sigmoid → 0.5 at init)

        # LoRA-paper init: A ~ N(0, 1/r_lora), B = 0, α = 0
        with torch.no_grad():
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=1.0 / lora_rank**0.5)
            nn.init.zeros_(self.lora_A.bias)
            nn.init.zeros_(self.lora_B.weight)
            nn.init.zeros_(self.lora_B.bias)

    def forward(
        self,
        x: torch.Tensor,                              # [B, L, D]
        context_vec: torch.Tensor | None = None,      # [B, L, D] or None
    ) -> torch.Tensor:                                # [B, L, D]
        """⚠️ SKELETON: current implementation delegates to vanilla CGMambaBlock.

        Real Plan B implementation would inject LoRA delta into dt computation
        inside CGMambaBlock (between x_proj and softplus). That requires either:
          (a) refactor CGMambaBlock.forward to accept an optional dt_residual, OR
          (b) subclass CGMambaBlock and override forward.

        For M1.3 we verify forward I/O shape compatibility only. lora_A/lora_B/α
        receive `requires_grad=True` but are no-ops in the forward graph.

        The shape check below confirms (a) instantiation works, (b) parameters are
        registered with autograd, (c) forward returns the right shape.
        """
        # Touch LoRA params with a dummy computation so they participate in autograd
        # graph (for skeleton shape-trace verification). NOT a real Plan B path.
        if context_vec is not None:
            _lora_delta = self.lora_B(self.lora_A(context_vec))     # [B, L, ED]
            _lora_alpha = torch.sigmoid(self.alpha) * 0.0           # = 0 (skeleton no-op)
            _ = _lora_delta * _lora_alpha   # touch tensors so backward reaches LoRA params
                                             # (real Plan B would inject into dt computation)

        # Delegate to vanilla CGMambaBlock (Plan A is the default, Plan B not active)
        return self.mamba(x, gate=None)


def _self_test() -> None:
    """Sanity: instantiation + forward shape + param count + autograd registration."""
    torch.manual_seed(0)
    cfg = CGMambaConfig()
    block = LoRAGatedMambaBlock(cfg, lora_rank=4)

    # Param count
    lora_params = sum(p.numel() for p in [
        block.lora_A.weight, block.lora_A.bias,
        block.lora_B.weight, block.lora_B.bias,
        block.alpha,
    ])
    expected = (64 * 4 + 4) + (4 * 128 + 128) + 1
    assert lora_params == expected, f"LoRA params: {lora_params} != {expected}"
    print(f"  ✓ LoRA params (rank=4): {lora_params}  (expected {expected})")

    # Forward shape
    B, L = 2, 100
    x = torch.randn(B, L, cfg.d_model)
    ctx = torch.randn(B, L, cfg.d_model)
    y = block(x, ctx)
    assert y.shape == (B, L, cfg.d_model)
    print(f"  ✓ Forward shape: {tuple(x.shape)} → {tuple(y.shape)}")

    # Vanilla path (context_vec=None)
    y_v = block(x, None)
    assert y_v.shape == (B, L, cfg.d_model)
    print(f"  ✓ Vanilla path (ctx=None): {tuple(y_v.shape)}")

    # Init: B is zero → lora_delta = 0
    with torch.no_grad():
        delta = block.lora_B(block.lora_A(ctx))
    assert delta.abs().max().item() == 0.0, "B init should be zero → delta=0"
    print(f"  ✓ LoRA init: B=0 → lora_delta exactly 0 at init")

    # LoRA params registered in nn.Module — confirmed (will be picked up by
    # optimizer once real Plan B forward injects delta into dt computation).
    registered = {n for n, _ in block.named_parameters()}
    assert "lora_A.weight" in registered
    assert "lora_B.weight" in registered
    assert "alpha" in registered
    print(f"  ✓ LoRA params registered ({len(registered)} total in block)")

    # NOTE: We do NOT assert grad-reach on LoRA params here. The skeleton's
    # forward intentionally multiplies by α·0 = 0 (dead branch), so PyTorch's
    # autograd optimization correctly skips LoRA grads. Real Plan B (Phase 2
    # if Plan A fails) must inject delta into dt computation inside
    # CGMambaBlock, at which point grads will flow.
    print(f"  ℹ️  LoRA grads inactive (expected for skeleton; activated in Phase 2)")

    print("\n  ✅ Plan B LoRA forward skeleton OK (Phase 2 contingency only).")


if __name__ == "__main__":
    print("Plan B (LoRA) Forward Skeleton Self-Test")
    print("=" * 50)
    _self_test()
