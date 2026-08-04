"""Phase 5 fairness audit — retrain CG-Mamba Stage 3 with no early stopping.

Saves last-epoch weights (no val-based selection) for all 5 seeds.
Output: runs/m1_8_stage3_train/hpo_p2_..._s{seed}_lastep/last.pt

Does NOT overwrite original best.pt — M2.3/M2.4 results unaffected.

Run: python3 scripts/phase_5_retrain_lastep.py
"""
from __future__ import annotations

import dataclasses
import math
import sys
from argparse import Namespace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.utils.config import CGMambaConfig
from scripts.m1_8_stage3_train import stage3_train

# M2.1 top1 cell hyperparameters (from m2_1_final_topk.py)
GATE_LR = 1e-3
BACKBONE_LR = 1e-4
LOOKBACK = 104
OTHER_LR_BASE = 1e-4
HMM_RATIO = 0.01
SE_RATIO = 0.01
ENV_RATIO = 0.001

SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS = 30   # max epochs (matches original config)
PATIENCE = 0    # NO early stopping → full epochs run, last.pt is true last


def build_cfg(seed: int) -> CGMambaConfig:
    return dataclasses.replace(
        CGMambaConfig(),
        seed=seed,
        lookback=LOOKBACK,
        stage2_gate_lr=GATE_LR,
        stage2_backbone_lr=BACKBONE_LR,
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * HMM_RATIO,
        stage3_state_embed_lr=OTHER_LR_BASE * SE_RATIO,
        stage3_env_lr=OTHER_LR_BASE * ENV_RATIO,
    )


def main():
    base_tag = f"hpo_p2_g{GATE_LR:.0e}_b{BACKBONE_LR:.0e}_lb{LOOKBACK}_h{HMM_RATIO}_se{SE_RATIO}_e{ENV_RATIO}"
    for seed in SEEDS:
        run_name = f"{base_tag}_s{seed}_lastep"
        out_dir = _ROOT / "runs" / "m1_8_stage3_train" / run_name
        if (out_dir / "last.pt").exists():
            print(f"[skip] seed={seed}: last.pt already exists at {out_dir}")
            continue

        stage2_dir = _ROOT / "runs" / "m1_7_train" / f"hpo_p1_g{GATE_LR:.0e}_b{BACKBONE_LR:.0e}_lb{LOOKBACK}_s{seed}"
        hmm_dir = _ROOT / "runs" / "m1_4_phase_dynamics_main" / f"V_raw3_regcov5e-03_K3_seed{seed}"
        env_ckpt = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"

        if not (stage2_dir / "best.pt").exists():
            print(f"[FAIL] seed={seed}: stage2 best.pt not found at {stage2_dir}")
            continue
        if not hmm_dir.exists():
            print(f"[FAIL] seed={seed}: hmm dir not found at {hmm_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"Retraining seed={seed} → {run_name}")
        print(f"  patience=0 (no early stop), epochs={N_EPOCHS}")
        print(f"{'='*70}")

        cfg = build_cfg(seed)
        args = Namespace(
            smoke=False,
            epochs=N_EPOCHS,
            patience=PATIENCE,
            batch_size=64,
            stage2_dir=str(stage2_dir),
            hmm_dir=str(hmm_dir),
            env_encoder_ckpt=str(env_ckpt),
            run_name=run_name,
        )
        try:
            stage3_train(cfg, args)
        except Exception as e:
            print(f"[FAIL] seed={seed}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
