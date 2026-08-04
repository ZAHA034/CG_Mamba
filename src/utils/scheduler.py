"""Warm-γ 3-Phase LR scheduler for CG-Mamba Stage 2 (M1.7).

PLAN v2.0.9 §5.1 D.5.2 (line 3026-3036) + CM-Mamba v8.0.2 §3.8 inheritance.

Schedule (epoch-indexed, 0-based):
    [0 .. warmup-1]                gate_proj-only warmup: 0.5 → 1.0 linear
                                   (M1.7 Finding 1 fix: PLAN-literal 0.5 start)
    [warmup .. P1-1]               Phase 1: base LR (per group)
    [P1 .. P1+TR-1]                Transition: linear 1.0 → phase2_start_ratio
                                   (M1.7 Finding 2 fix: epoch-(P1+TR-1) endpoint)
    [P1+TR .. total-1]             Phase 2: cosine annealing
                                   phase2_start_ratio → η_min / base_lr

Per-group multiplier behavior:
    gate_proj    : warmup 0.5→1.0 ramp applied (only group with warmup)
    decoder_gate : NO warmup (starts at Phase 1 base immediately)
    context_embed: NO warmup
    backbone     : NO warmup

Continuity verification (M1.7 Finding 1 + 2):
    epoch 0:  gate_proj=0.5 (warmup start),  others=1.0
    epoch 1:  gate_proj=1.0 (warmup end),    others=1.0
    epoch 2:  gate_proj=1.0 (Phase 1),       others=1.0    ← warmup→Phase 1 continuous
    epoch P1-1=9:  all groups=1.0 (Phase 1 end)
    epoch P1=10:   transition progress=0/(TR-1)=0 → multiplier=1.0  ← Phase 1→transition continuous
    epoch P1+TR-1=19:  progress=(TR-1)/(TR-1)=1 → multiplier=phase2_start_ratio (0.5)
    epoch P1+TR=20:    cosine progress=0/cosine_total=0 → cos(0)=1
                       multiplier = min_ratio + (phase2_start_ratio - min_ratio) * 1
                                  = phase2_start_ratio = 0.5
                       ← transition→cosine continuous

phase2_start_ratio = 0.5 rationale (D-7):
    PLAN D.5.2 only states "Linear interpolation to Phase 2 start" without
    specifying the exact ratio. CM-Mamba v8.0.2 §3.8 Warm-γ design halves the
    gate_proj 20× gap during transition. 0.5 is the conservative default.
    Future sensitivity ablation: {0.3, 0.5, 0.7}.
"""
from __future__ import annotations

import math
from typing import Callable

import torch


def _make_lr_lambda(
    group_name: str,
    base_lr: float,
    P1: int,
    TR: int,
    warmup: int,
    total_epochs: int,
    eta_min: float,
    phase2_start_ratio: float,
) -> Callable[[int], float]:
    """Build the per-group lambda function for LambdaLR.

    Returns a function `fn(epoch) -> multiplier` that AdamW multiplies by
    `base_lr` each epoch.
    """
    min_ratio = eta_min / base_lr

    def fn(epoch: int) -> float:
        # (C-1 fix, v2.1.7 H-1 doc update) context_embed override: schedule
        # multiplier is always 1.0 (LR held constant at base_lr). PLAN D.5.2
        # spec is 1e-6 (near-freeze), but v2.1.3 ablation raised the
        # optimizer base_lr to 1e-5 (see optimizer.py:121). Either way the
        # spec-intended behaviour is "no annealing": with base_lr = 1e-5 and
        # eta_min = 1e-6, the unguarded schedule would drift context_embed
        # 1e-5 → ~0.5e-5 → ~1e-6 (10× shrink) during Transition/Phase 2,
        # contrary to design. Short-circuit to 1.0 preserves a stable phase
        # basis regardless of whichever base_lr is active.
        if group_name == "context_embed":
            return 1.0

        # (a) Warmup (gate_proj only, epochs [0, warmup-1])
        # M1.7 Finding 1 fix: PLAN-literal 0.5 → 1.0 endpoints.
        # Formula: 0.5 + 0.5 * epoch / max(warmup-1, 1)
        #   epoch=0       → 0.5
        #   epoch=warmup-1 → 1.0  (last warmup step)
        #   epoch=warmup   → 1.0  (Phase 1 start, continuous)
        if group_name == "gate_proj" and epoch < warmup:
            denom = max(warmup - 1, 1)
            return 0.5 + 0.5 * epoch / denom

        # (b) Phase 1 (all groups, epochs [warmup .. P1-1] for gate_proj,
        #     [0 .. P1-1] for others)
        if epoch < P1:
            return 1.0

        # (c) Transition (epochs [P1, P1+TR-1])
        # M1.7 Finding 2 fix: progress = (epoch - P1) / (TR - 1)
        #   epoch=P1         → progress=0  → multiplier=1.0  (Phase 1 continuous)
        #   epoch=P1+TR-1   → progress=1  → multiplier=phase2_start_ratio
        if epoch < P1 + TR:
            denom_tr = max(TR - 1, 1)
            progress = (epoch - P1) / denom_tr
            return 1.0 - progress * (1.0 - phase2_start_ratio)

        # (d) Phase 2: cosine annealing (epochs [P1+TR, total-1])
        # Starts at phase2_start_ratio, decays to min_ratio (=eta_min/base_lr)
        cosine_epoch = epoch - (P1 + TR)
        cosine_total = max(total_epochs - (P1 + TR), 1)
        cosine_progress = min(cosine_epoch / cosine_total, 1.0)
        return max(
            min_ratio,
            min_ratio + (phase2_start_ratio - min_ratio)
            * 0.5 * (1.0 + math.cos(math.pi * cosine_progress)),
        )

    return fn


def build_warm_gamma_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    P1: int = 10,
    TR: int = 10,
    warmup: int = 2,
    eta_min: float = 1e-6,
    phase2_start_ratio: float = 0.5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the Warm-γ 3-Phase scheduler.

    Args:
        optimizer:           AdamW with named param groups (each dict has 'name', 'lr').
        total_epochs:        full training duration (PLAN D.5.2 default 200).
        P1:                  Phase 1 length (epochs at base LR).
        TR:                  Transition length.
        warmup:              gate_proj-only warmup length (only the gate_proj group
                              experiences the 0.5 → 1.0 ramp).
        eta_min:             Phase 2 cosine annealing floor (PLAN D.5.2).
        phase2_start_ratio:  base LR multiplier at the start of Phase 2 (D-7).

    Returns:
        torch.optim.lr_scheduler.LambdaLR (call scheduler.step() once per epoch).
    """
    # Misuse guard: total_epochs must accommodate at least one full Phase 1 +
    # Transition + at least one Cosine epoch. Catches the common mistake of
    # passing Stage 1 / Stage 3 epoch count where Stage 2's value was expected.
    assert total_epochs > P1 + TR, (
        f"total_epochs={total_epochs} must be > P1+TR={P1+TR}. "
        f"Got total_epochs={total_epochs}, P1={P1}, TR={TR}. "
        f"Did you pass cfg.n_epochs (Stage 1) or args.epochs (Stage 3) "
        f"instead of cfg.stage2_n_epochs?"
    )

    lambdas = []
    for group in optimizer.param_groups:
        name = group.get("name", "backbone")
        base_lr = group["lr"]
        lambdas.append(
            _make_lr_lambda(
                group_name=name,
                base_lr=base_lr,
                P1=P1,
                TR=TR,
                warmup=warmup,
                total_epochs=total_epochs,
                eta_min=eta_min,
                phase2_start_ratio=phase2_start_ratio,
            )
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
