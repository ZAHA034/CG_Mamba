"""scripts/e1_final_smoke.py — D.1+D.3 fix 동작 검증 (1 run)

n3_d64 × seed=42 × Stage 2 (200 ep) + Stage 3 (10 ep, patience=0, per-seed HMM)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

import scripts.e1_final_train as e1f
import scripts.m1_7_train as m1_7
import scripts.m1_8_stage3_train as m1_8


def main():
    n_layers, d_model, seed = 3, 64, 42
    cfg = e1f.build_cfg(n_layers, d_model, seed)
    run_name = "e1_final_smoke_n3_d64_s42"
    stage3_run_name = f"{run_name}_stage3"

    print(f"=== E1 FINAL SMOKE — D.1+D.3 fix verification ===")
    print(f"  cfg: n_layers={cfg.n_layers}, d_model={cfg.d_model}, lookback={cfg.lookback}")
    print(f"  Stage 2: epochs={e1f.STAGE2_EPOCHS}")
    print(f"  Stage 3: epochs={e1f.STAGE3_EPOCHS}, patience={e1f.STAGE3_PATIENCE}")

    # Stage 2
    stage2_args = e1f.build_stage2_args(run_name, d_model, seed)
    print(f"  Stage 2 hmm_dir: {stage2_args.hmm_dir}")
    print(f"  Stage 2 env_ckpt: {stage2_args.env_encoder_ckpt}")
    t0 = time.time()
    s2 = m1_7.train(cfg, stage2_args)
    s2_sec = time.time() - t0
    print(f"\n  STAGE 2 DONE: {s2_sec:.1f}s  best_val_total={s2['best_val_total']:.4f}", flush=True)

    # Stage 3
    stage2_dir = _ROOT / "runs/m1_7_train" / run_name
    stage3_args = e1f.build_stage3_args(stage3_run_name, stage2_dir, d_model, seed)
    print(f"\n  Stage 3 hmm_dir: {stage3_args.hmm_dir}")
    print(f"  Stage 3 epochs={stage3_args.epochs}, patience={stage3_args.patience}")
    t1 = time.time()
    s3 = m1_8.stage3_train(cfg, stage3_args)
    s3_sec = time.time() - t1
    print(f"\n  STAGE 3 DONE: {s3_sec:.1f}s  best_val_total={s3['best_val_total']:.4f}", flush=True)

    # Verify D.1 + D.3 fix
    print(f"\n=== D.1 + D.3 fix verification ===")
    assert s3.get("n_epochs_configured") == 10, f"D.3 BUG: n_epochs_configured={s3.get('n_epochs_configured')} (want 10)"
    assert s3.get("n_epochs_run") == 10, f"D.3 BUG: n_epochs_run={s3.get('n_epochs_run')} (want 10)"
    assert s3.get("patience") == 0, f"D.3 BUG: patience={s3.get('patience')} (want 0)"
    assert "seed42" in stage2_args.hmm_dir, f"D.1 BUG: seed=42 dispatched to {stage2_args.hmm_dir}"
    print(f"  ✓ D.3: n_epochs_configured={s3['n_epochs_configured']}  n_epochs_run={s3['n_epochs_run']}  patience={s3['patience']}")
    print(f"  ✓ D.1: hmm_dir = {Path(stage2_args.hmm_dir).name} (per-seed dispatched)")
    print(f"\n=== SMOKE PASS — ready for final-train launch (10 runs) ===")


if __name__ == "__main__":
    main()
