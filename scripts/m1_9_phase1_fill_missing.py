"""Fill missing Phase 1 Stage 2 checkpoints (v2.1.7-A++ E enabler).

Phase 1 originally trained 3 bases × 3 seeds = 9 ckpts.
But Stage B (E refinement) requires 5 seeds [42, 123, 456, 789, 1024] per base.

Missing combos (auto-detected by this script):
  - Base 1 (g=2e-03, b=1e-04, lb=104) × seeds {789, 1024}
  - Base 3 (g=1e-03, b=1e-04, lb=104) × seeds {789, 1024}
  - Base 2 (g=5e-04, b=1e-04, lb=156) already has all 5 seeds (no work needed)

Pattern: mirrors m1_9_hpo_phase1.py `_run_one_cell` exactly (Stage 2 from scratch,
200-epoch cap, val_total best-tracking, dropout=0.0, full cfg).

Run
---
  CUDA_VISIBLE_DEVICES=1 python3 scripts/m1_9_phase1_fill_missing.py

Output
------
  runs/m1_7_train/hpo_p1_g{gate}_b{back}_lb{lb}_s{seed}/best.pt + history.json + ...
"""
from __future__ import annotations

import dataclasses
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

from src.utils.config import CGMambaConfig                                  # noqa: E402
from scripts.m1_7_train import train as stage2_train                       # noqa: E402


# Phase 1 base × seeds requirement (matches m1_9_hpo_phase1.py BASES + Stage B FINAL_SEEDS=5)
REQUIRED_BASES = [
    {"gate_lr": 5e-4, "backbone_lr": 1e-4, "lookback": 156},
    {"gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104},
    {"gate_lr": 2e-3, "backbone_lr": 1e-4, "lookback": 104},
]
REQUIRED_SEEDS = [42, 123, 456, 789, 1024]

HMM_DIR_TEMPLATE = str(_ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}")
ENV_CKPT = _ROOT / "runs/m1_7_env_pretrain/env_encoder.pt"
P1_ROOT = _ROOT / "runs/m1_7_train"
LOG_PATH = _ROOT / "runs/baseline_grid_logs/phase1_fill_missing.log"


def _cell_run_name(base: dict, seed: int) -> str:
    return f"hpo_p1_g{base['gate_lr']:.0e}_b{base['backbone_lr']:.0e}_lb{base['lookback']}_s{seed}"


def _find_missing() -> list[tuple[dict, int]]:
    missing = []
    for base in REQUIRED_BASES:
        for seed in REQUIRED_SEEDS:
            run_name = _cell_run_name(base, seed)
            best_pt = P1_ROOT / run_name / "best.pt"
            if not best_pt.exists():
                missing.append((base, seed))
    return missing


def _build_cell_cfg(base: dict, seed: int) -> CGMambaConfig:
    return dataclasses.replace(
        CGMambaConfig(),
        stage2_gate_lr=base["gate_lr"],
        stage2_backbone_lr=base["backbone_lr"],
        lookback=base["lookback"],
        seed=seed,
    )


def _build_cell_args(run_name: str, seed: int) -> SimpleNamespace:
    hmm_dir = Path(HMM_DIR_TEMPLATE.format(seed=seed))
    return SimpleNamespace(
        smoke=False,
        epochs=None,                                # use cfg.stage2_n_epochs default (200)
        batch_size=32,
        hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(ENV_CKPT),
        wandb_mode="disabled",
        run_name=run_name,
    )


def _train_one(base: dict, seed: int, log) -> dict:
    run_name = _cell_run_name(base, seed)
    print(f"\n{'='*60}")
    print(f"[START] {run_name}  (gate_lr={base['gate_lr']:.0e}, backbone_lr={base['backbone_lr']:.0e}, "
          f"lookback={base['lookback']}, seed={seed})")
    print('='*60)
    log.write(f"[{datetime.now():%H:%M:%S}] START {run_name}\n"); log.flush()

    hmm_dir = Path(HMM_DIR_TEMPLATE.format(seed=seed))
    if not hmm_dir.exists():
        msg = f"❌ HMM ckpt missing: {hmm_dir} — cannot proceed"
        print(msg); log.write(f"[{datetime.now():%H:%M:%S}] {msg}\n"); log.flush()
        return {"run_name": run_name, "ok": False, "error": "hmm_missing"}

    cfg = _build_cell_cfg(base, seed)
    args = _build_cell_args(run_name, seed)

    t0 = datetime.now()
    try:
        final = stage2_train(cfg, args)
        ok = True
        err = None
    except Exception as e:
        final = {"best_val_total": float("inf"), "best_epoch": -1,
                 "elapsed_sec": (datetime.now() - t0).total_seconds()}
        ok = False
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    elapsed = (datetime.now() - t0).total_seconds()

    status = "OK" if ok else "FAILED"
    print(f"\n[{status}] {run_name}  elapsed={elapsed:.1f}s  best_val={final.get('best_val_total', float('inf')):.4f}")
    log.write(f"[{datetime.now():%H:%M:%S}] {status} {run_name}  elapsed={elapsed:.1f}s  "
              f"best_val={final.get('best_val_total', float('inf')):.4f}  "
              f"{('err='+err) if err else ''}\n"); log.flush()
    return {"run_name": run_name, "ok": ok, "elapsed_sec": elapsed, **final, "error": err}


def main() -> int:
    missing = _find_missing()
    print(f"=== Phase 1 fill missing — {len(missing)} ckpts to train ===")
    for base, seed in missing:
        print(f"  {_cell_run_name(base, seed)}")
    if not missing:
        print("Nothing to do. All required Phase 1 ckpts exist.")
        return 0

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as log:
        log.write(f"\n=== Phase 1 fill missing START at {datetime.now()} ===\n")
        log.write(f"  Missing: {len(missing)} ckpts\n")
        log.flush()

        results = []
        for i, (base, seed) in enumerate(missing, 1):
            print(f"\n[{i}/{len(missing)}]")
            r = _train_one(base, seed, log)
            results.append(r)

        log.write(f"\n=== Phase 1 fill missing END at {datetime.now()} ===\n")
        ok_count = sum(1 for r in results if r['ok'])
        log.write(f"  Completed: {ok_count}/{len(results)}\n")

    print(f"\n=== SUMMARY ===")
    print(f"  Completed: {sum(1 for r in results if r['ok'])}/{len(results)}")
    for r in results:
        status = "✓" if r['ok'] else "✗"
        print(f"  {status} {r['run_name']}: best_val={r.get('best_val_total', float('inf')):.4f}  "
              f"elapsed={r.get('elapsed_sec', 0):.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
