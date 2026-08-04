"""Stage 2 4-param-group AdamW builder (M1.7, PLAN D.5.2).

PLAN v2.0.9 §5.1 D.5.2 (line 2997-3023) spec:
    Group         LR                       WD     Role
    ─────────────────────────────────────────────────────────────────────
    gate_proj     cfg.stage2_gate_lr        1e-3   context → gate mapping (20× backbone)
    decoder_gate  cfg.stage2_backbone_lr    0.0    EntropyAwareDecoder.gate (α scalar)
    context_embed 1e-6                      0.0    state_embeddings [K, D] near-freeze
    backbone      cfg.stage2_backbone_lr    0.01   encoder/decoder.proj/env.encoder/norms

Critical safety guards:
    - ERR-C2 (PLAN D.5.2): every trainable parameter must be assigned to exactly
      one group. Implemented via assert on set equality between named_params
      and union of group-assigned names.
    - PLAN line 3022: backbone weight_decay=0.01 (NOT cfg.weight_decay=1e-5,
      which is Stage 1 default). We use cfg.stage2_backbone_wd to make this
      explicit and prevent silent 1000× regularization mismatch.

Public API:
    build_stage2_optimizer(model, cfg) -> torch.optim.AdamW
"""
from __future__ import annotations

import torch

from src.models.cg_forecaster import CGForecaster
from src.utils.config import CGMambaConfig


def build_stage2_optimizer(
    model: CGForecaster,
    cfg: CGMambaConfig,
) -> torch.optim.AdamW:
    """Build the Stage 2 AdamW optimizer with 4 param groups (PLAN D.5.2).

    Args:
        model: CGForecaster with `prepare_for_stage2(hmm)` already called.
               (Env decoder must be frozen; HMM is frozen as register_buffer.)
        cfg:   CGMambaConfig providing stage2_gate_lr, stage2_backbone_lr,
               stage2_backbone_wd.

    Returns:
        torch.optim.AdamW with 4 named param groups (`name` key).

    Raises:
        RuntimeError: if a trainable parameter is unassigned to any group or
                      assigned to multiple groups (ERR-C2 violation).
    """
    named_params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    # ── Group classification (name-pattern match, PLAN D.5.2 line 893-906) ──
    # NOTE: CGForecaster's `named_parameters` produces names like:
    #   "decoder.gate"            ← EntropyAwareDecoder.gate (α scalar)
    #   "decoder.proj.weight/bias" ← EntropyAwareDecoder.proj  (backbone group)
    #   "encoder.layers.{i}.gate_proj.{j}.weight/bias" ← per-layer gate_proj
    #   "phase_module.state_embeddings"
    # decoder.gate name match: exact (not endswith .decoder.gate; the dot prefix
    # would require a deeper hierarchy, which CGForecaster.decoder.gate does NOT have).
    gate_proj_names = [n for n in named_params if "gate_proj" in n]
    decoder_gate_names = [n for n in named_params if n == "decoder.gate"]
    context_embed_names = [n for n in named_params if "state_embeddings" in n]
    backbone_names = [
        n for n in named_params
        if n not in gate_proj_names
        and n not in decoder_gate_names
        and n not in context_embed_names
    ]

    # ── ERR-C2: every trainable param assigned exactly once ──
    all_assigned = (
        set(gate_proj_names) | set(decoder_gate_names)
        | set(context_embed_names) | set(backbone_names)
    )
    all_trainable = set(named_params.keys())
    unassigned = all_trainable - all_assigned
    over_assigned = all_assigned - all_trainable
    if unassigned or over_assigned:
        raise RuntimeError(
            f"ERR-C2 param group mismatch:\n"
            f"  unassigned (in named_params but in no group): {unassigned}\n"
            f"  over-assigned (in some group but not in named_params): {over_assigned}\n"
            f"  gate_proj_names ({len(gate_proj_names)}): {gate_proj_names}\n"
            f"  decoder_gate_names ({len(decoder_gate_names)}): {decoder_gate_names}\n"
            f"  context_embed_names ({len(context_embed_names)}): {context_embed_names}\n"
            f"  backbone_names ({len(backbone_names)}): {backbone_names[:5]}..."
        )

    # ── Verify no duplicates (set equality already implies no dup, but verify
    # the explicit total count too for defense-in-depth) ──
    expected_total = (
        len(gate_proj_names) + len(decoder_gate_names)
        + len(context_embed_names) + len(backbone_names)
    )
    if expected_total != len(all_trainable):
        raise RuntimeError(
            f"ERR-C2 duplicate assignment: sum of group sizes={expected_total} "
            f"!= unique trainable count={len(all_trainable)}"
        )

    # ── Build param_groups (PLAN D.5.2 LR / WD values) ──
    param_groups = [
        {
            "name": "gate_proj",
            "params": [named_params[n] for n in gate_proj_names],
            "lr": cfg.stage2_gate_lr,
            "weight_decay": cfg.stage2_gate_wd,    # v2.1.7 H-2: surfaced from hardcode (PLAN D.5.2)
        },
        {
            "name": "decoder_gate",
            "params": [named_params[n] for n in decoder_gate_names],
            "lr": cfg.stage2_backbone_lr,
            "weight_decay": 0.0,
        },
        {
            "name": "context_embed",
            "params": [named_params[n] for n in context_embed_names],
            # v2.1.3 ablation: PLAN D.5.2 spec is 1e-6 (near-freeze), but Step 8
            # full run showed state_embeddings barely moved (Δ ~1e-4 over 63 ep).
            # 1e-5 lets phase identity flex without removing the stability the
            # near-freeze provides. Revert to 1e-6 if Check 4 std stays < 0.10.
            "lr": 1e-5,
            "weight_decay": 0.0,
        },
        {
            "name": "backbone",
            "params": [named_params[n] for n in backbone_names],
            "lr": cfg.stage2_backbone_lr,
            "weight_decay": cfg.stage2_backbone_wd,
        },
    ]

    return torch.optim.AdamW(param_groups)
