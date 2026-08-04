"""scripts/e1_final_train.py — (α.2) E1 final-train (paper headline)
================================================================================
2 configs × 5 seeds × 원 full train (200140-201839, 868 rows) 학습.

LOCKED γ.6 final preprocessing (held-out 201840+ 와 분리되어 leak-free):
  - data CSV: data/processed/ili_env_weekly_split.csv (원, split='train' = full)
  - scaler: data/processed/normalization_params.json (원 train fit)
  - HMM: runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}  ← D.1 fix
         (paper ablation_retrain.py:74 mirror: per-seed HMM, not seed42-only)
  - env: runs/m1_7_env_pretrain_final/env_encoder_d{D}.pt (α.1, 원 train env fit)

Configs (E1 4차 HPO winner 확정, 2026-06-18):
  - n2_d128 (E1 winner, val_total=0.3030±0.0136, 1/9 grid)
  - n3_d64  (paper baseline + efficiency alt, val_total=0.3161±0.0071, 4/9)

10 trainings total. 평가는 별도 (e1_final_eval.py).

Protocol (paper ablation_retrain `full` 5-seed Table I mirror, manifest 확정):
  - Stage 2: 200 ep, patience=30 (cfg.stage2_patience)
  - Stage 3: 10 ep, patience=0   ← D.3 fix (paper manifest n_epochs_run=10)

CLI:
  python scripts/e1_final_train.py            # 10 runs sequential, GPU 0
"""
from __future__ import annotations
import dataclasses
import gc
import json
import sys
import time
from argparse import Namespace
from itertools import product
from pathlib import Path
import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.utils.config import CGMambaConfig
import scripts.m1_7_train as m1_7
import scripts.m1_8_stage3_train as m1_8

# === LOCKED final-train paths ===
FINAL_CSV       = _ROOT / "data/processed/ili_env_weekly_split.csv"
FINAL_NORM_JSON = _ROOT / "data/processed/normalization_params.json"
# D.1 fix (2026-06-18 audit): per-seed HMM template (paper ablation_retrain mirror).
# evidence: paper best.pt phase_module._A hash 가 seed 마다 다름 (seed42=a9f9.., s123=7188.., s456=0ed2..)
FINAL_HMM_TPL   = _ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}"
FINAL_ENV_TPL   = _ROOT / "runs/m1_7_env_pretrain_final/env_encoder_d{d_model}.pt"
OUT_DIR         = _ROOT / "runs/e1_final"

# === Configs — E1 4차 HPO winner 확정 (2026-06-18 20:42 종료)
# γ.3 selection (val_total ascending, paper m1_9 mirror): n2_d128 1위 / n3_d64 4위
CONFIGS = [
    ("n2_d128", 2, 128),   # E1 winner (val_total=0.3030±0.0136, 1/9 grid)
    ("n3_d64",  3,  64),   # paper baseline + efficiency alternative (val_total=0.3161±0.0071, 4/9)
]
SEEDS = [42, 123, 456, 789, 1024]
BATCH_SIZE = 32

# γ.5 — paper winner HP override (ablation_retrain.py:65-72 그대로)
CG_TOP1_HP = {
    "gate_lr":               1e-3,
    "backbone_lr":           1e-4,
    "lookback":              104,
    "hmm_lr_ratio":          0.01,
    "state_embed_lr_ratio":  0.01,
    "env_lr_ratio":          0.001,
}
OTHER_LR_BASE = 1e-4

# γ.6 — Stage 2 + Stage 3 protocol (paper ablation_retrain `full` 5-seed Table I mirror)
# D.3 fix (2026-06-18 audit): STAGE3_EPOCHS 30 → 10 (manifest 확정)
# evidence: paper runs/m1_8_stage3_train/ablation_retrain_full_s{42..1024}_stage3/final_metrics.json
#          모두 n_epochs_configured=10, n_epochs_run=10, patience=0 (5 seeds 동일).
# 주의: e1_hpo (selection) 는 paper m1_9 mirror (patience=10, epochs=30). 본 e1_final 만 10/0.
STAGE2_EPOCHS = 200
STAGE3_EPOCHS = 10     # D.3 fix: paper ablation_retrain manifest 일치 (was 30)
STAGE3_PATIENCE = 0    # paper Table I headline protocol


def build_cfg(n_layers: int, d_model: int, seed: int) -> CGMambaConfig:
    """γ.5 — CG_TOP1_HP 9개 HP override (paper baseline 과 같은 env)."""
    hp = CG_TOP1_HP
    return dataclasses.replace(
        CGMambaConfig(),
        seed=seed,
        n_layers=n_layers,
        d_model=d_model,
        data_csv=FINAL_CSV,
        norm_json=FINAL_NORM_JSON,
        dropout=0.0,
        lookback=hp["lookback"],
        stage2_gate_lr=hp["gate_lr"],
        stage2_backbone_lr=hp["backbone_lr"],
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * hp["hmm_lr_ratio"],
        stage3_state_embed_lr=OTHER_LR_BASE * hp["state_embed_lr_ratio"],
        stage3_env_lr=OTHER_LR_BASE * hp["env_lr_ratio"],
    )


def build_stage2_args(run_name: str, d_model: int, seed: int) -> Namespace:
    env_ckpt = Path(str(FINAL_ENV_TPL).format(d_model=d_model))
    assert env_ckpt.exists(), f"missing final env ckpt: {env_ckpt}"
    hmm_dir = Path(str(FINAL_HMM_TPL).format(seed=seed))
    assert hmm_dir.exists(), f"missing per-seed HMM ckpt: {hmm_dir}"
    return Namespace(
        smoke=False, epochs=STAGE2_EPOCHS, batch_size=BATCH_SIZE,
        hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(env_ckpt),
        wandb_mode="disabled",
        run_name=run_name,
    )


def build_stage3_args(run_name: str, stage2_dir: Path, d_model: int, seed: int) -> Namespace:
    env_ckpt = Path(str(FINAL_ENV_TPL).format(d_model=d_model))
    hmm_dir = Path(str(FINAL_HMM_TPL).format(seed=seed))
    return Namespace(
        smoke=False, epochs=STAGE3_EPOCHS, patience=STAGE3_PATIENCE,
        batch_size=BATCH_SIZE,
        stage2_dir=str(stage2_dir),
        hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(env_ckpt),
        run_name=run_name,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'#'*80}")
    print(f"# E1 FINAL-TRAIN — 2 configs × 5 seeds = 10 runs")
    print(f"# device={device}")
    print(f"# CSV={FINAL_CSV.name}  norm={FINAL_NORM_JSON.name}")
    print(f"# HMM=per-seed (V_raw3_regcov5e-03_K3_seed{{seed}}) ← D.1 fix")
    print(f"# Stage 3 protocol: epochs={STAGE3_EPOCHS}, patience={STAGE3_PATIENCE} ← D.3 fix")
    print(f"{'#'*80}", flush=True)

    t_total = time.time()
    summary_records = []

    for config_id, n_layers, d_model in CONFIGS:
        for seed in SEEDS:
            run_name = f"e1_final_{config_id}_s{seed}"
            stage3_run_name = f"{run_name}_stage3"
            print(f"\n{'='*80}")
            print(f"[FINAL] {config_id} seed={seed} Stage 2 ({STAGE2_EPOCHS}) + Stage 3 ({STAGE3_EPOCHS})")
            print(f"{'='*80}", flush=True)
            cfg = build_cfg(n_layers, d_model, seed)

            # --- Stage 2 ---
            stage2_args = build_stage2_args(run_name, d_model, seed)
            t0 = time.time()
            stage2_final = m1_7.train(cfg, stage2_args)
            stage2_sec = time.time() - t0
            stage2_val = stage2_final.get('best_val_total', float('nan'))
            print(f"  Stage 2: {stage2_sec:.1f}s  best_val_total={stage2_val:.4f}", flush=True)

            # --- Stage 3 (paper headline ckpt protocol) ---
            stage2_dir = _ROOT / "runs/m1_7_train" / run_name
            stage3_args = build_stage3_args(stage3_run_name, stage2_dir, d_model, seed)
            print(f"\n[FINAL] {config_id} seed={seed} === Stage 3 ===", flush=True)
            t1 = time.time()
            stage3_final = m1_8.stage3_train(cfg, stage3_args)
            stage3_sec = time.time() - t1
            stage3_val = stage3_final.get('best_val_total', float('nan'))
            print(f"  Stage 3: {stage3_sec:.1f}s  best_val_total={stage3_val:.4f}  "
                  f"(Δ vs Stage 2: {stage3_val - stage2_val:+.4f})", flush=True)

            # per-run summary 즉시 저장 (증분, crash 내성)
            rec = dict(
                config_id=config_id, n_layers=n_layers, d_model=d_model, seed=seed,
                stage2_sec=stage2_sec, stage3_sec=stage3_sec,
                stage2_metrics=stage2_final, stage3_metrics=stage3_final,
                stage2_ckpt=str(_ROOT / "runs/m1_7_train" / run_name / "best.pt"),
                stage3_ckpt=str(_ROOT / "runs/m1_8_stage3_train" / stage3_run_name / "best.pt"),
                final_val_total=stage3_val,
            )
            summary_records.append(rec)
            with open(OUT_DIR / f"train_summary_{config_id}_s{seed}.json", "w") as f:
                json.dump(rec, f, indent=2, default=str)

            # GPU 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()

    elapsed_h = (time.time() - t_total) / 3600
    print(f"\n{'#'*80}")
    print(f"# E1 FINAL-TRAIN DONE — {elapsed_h:.2f}h elapsed")
    print(f"{'#'*80}", flush=True)

    with open(OUT_DIR / "train_summary_all.json", "w") as f:
        json.dump(dict(
            locked_constants=dict(
                configs=[c[0] for c in CONFIGS], seeds=SEEDS,
                stage2_epochs=STAGE2_EPOCHS, stage3_epochs=STAGE3_EPOCHS,
                stage3_patience=STAGE3_PATIENCE, batch_size=BATCH_SIZE,
                cg_top1_hp=CG_TOP1_HP, other_lr_base=OTHER_LR_BASE,
                final_csv=str(FINAL_CSV.relative_to(_ROOT)),
                final_norm=str(FINAL_NORM_JSON.relative_to(_ROOT)),
                final_hmm_tpl=str(FINAL_HMM_TPL.relative_to(_ROOT)),  # D.1 fix: per-seed template
            ),
            elapsed_hours=elapsed_h,
            runs=summary_records,
        ), f, indent=2, default=str)
    print(f"  saved: {OUT_DIR / 'train_summary_all.json'}")
    print(f"\n  next: python scripts/e1_final_eval.py  (held-out + test_strict + PC2-a 재확인)")


if __name__ == "__main__":
    main()
