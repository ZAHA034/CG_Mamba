"""M2.1 Final — Option β: Base 2 global top-K cells × 5 seeds × Full Stage 2+3.

Background (v2.1.7-A++, 2026-05-26):
After E HPO (Stage B 5-seed) eval on (val, test_full, test_strict), we observed:
- Global top-10 by 5-seed val_avg ALL came from Base 2 (g=1e-3, lb=104).
- Option α (per-base top-1) gives diversity but mixes inferior bases.
- Option β (global top-K from Base 2) selects uniformly best ratio variants.

Selection (v2.1.7-A++ Option β):
  Top-3 cells by mean val_avg (all Base 2):
    #1 g1e-03_lb104_h0.01_se0.01_e0.001   val=0.3461, tS=0.3875
    #2 g1e-03_lb104_h0.01_se0.001_e0.001  val=0.3467, tS=0.3854
    #3 g1e-03_lb104_h0.001_se0.1_e0.1     val=0.3469, tS=0.3774  ← best tS

Each cell: Full Stage 2 (200 epochs cap) + Stage 3 (30 epochs cap) × 5 seeds.
  → 3 × 5 = 15 runs × ~30min/run = ~7.5h on a free GPU.

Strategy:
  - Reuses scripts/m2_1_final.py train logic by temporarily overwriting
    hpo_winner.json then restoring. Each cell outputs to its own dir.
  - Sequential (one cell at a time) for memory safety.

Output
------
  runs/m2_1_final_topk/cg_mamba_top{1,2,3}/seed{42,123,456,789,1024}/
  runs/m2_1_final_topk/cg_mamba_top{1,2,3}/final_summary.json
  runs/m2_1_final_topk/master_summary.json   (aggregate across 3 cells)

Run
---
  CUDA_VISIBLE_DEVICES=<gpu> python3 scripts/m2_1_final_topk.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]


SEEDS = [42, 123, 456, 789, 1024]   # Match LSTM/PatchTST/DLinear/iTransformer seed list

# Option β cells (Base 2 global top-3 by 5-seed mean val_avg)
TOPK_CELLS = [
    {
        "rank": 1,
        "cell": "hpo_p2_g1e-03_b1e-04_lb104_h0.01_se0.01_e0.001",
        "base_gate_lr": 0.001,
        "base_backbone_lr": 0.0001,
        "base_lookback": 104,
        "hmm_lr_ratio": 0.01,
        "state_embed_lr_ratio": 0.01,
        "env_lr_ratio": 0.001,
        "stage_b_mean_val_avg": 0.3461,
        "stage_b_mean_test_strict_avg": 0.3875,
        "out_subdir": "cg_mamba_top1",
    },
    {
        "rank": 2,
        "cell": "hpo_p2_g1e-03_b1e-04_lb104_h0.01_se0.001_e0.001",
        "base_gate_lr": 0.001,
        "base_backbone_lr": 0.0001,
        "base_lookback": 104,
        "hmm_lr_ratio": 0.01,
        "state_embed_lr_ratio": 0.001,
        "env_lr_ratio": 0.001,
        "stage_b_mean_val_avg": 0.3467,
        "stage_b_mean_test_strict_avg": 0.3854,
        "out_subdir": "cg_mamba_top2",
    },
    {
        "rank": 3,
        "cell": "hpo_p2_g1e-03_b1e-04_lb104_h0.001_se0.1_e0.1",
        "base_gate_lr": 0.001,
        "base_backbone_lr": 0.0001,
        "base_lookback": 104,
        "hmm_lr_ratio": 0.001,
        "state_embed_lr_ratio": 0.1,
        "env_lr_ratio": 0.1,
        "stage_b_mean_val_avg": 0.3469,
        "stage_b_mean_test_strict_avg": 0.3774,
        "out_subdir": "cg_mamba_top3",
    },
]


HPO_WINNER_JSON = _ROOT / "runs/m1_9_hpo_phase2/hpo_winner.json"
M2_1_OUT_ROOT = _ROOT / "runs/m2_1_final/cg_mamba_winner"
TOPK_OUT_ROOT = _ROOT / "runs/m2_1_final_topk"
M2_1_SCRIPT = _ROOT / "scripts/m2_1_final.py"


def _build_winner_blob(cell_cfg: dict) -> dict:
    """Build hpo_winner.json compatible blob from our cell spec."""
    return {
        "phase2_winner": {
            "cell": cell_cfg["cell"],
            "base_gate_lr": cell_cfg["base_gate_lr"],
            "base_backbone_lr": cell_cfg["base_backbone_lr"],
            "base_lookback": cell_cfg["base_lookback"],
            "base_phase1_test_mse": float("nan"),
            "hmm_lr_ratio": cell_cfg["hmm_lr_ratio"],
            "state_embed_lr_ratio": cell_cfg["state_embed_lr_ratio"],
            "env_lr_ratio": cell_cfg["env_lr_ratio"],
            "n_seeds": len(SEEDS),
            "mean_stage3_best_val": cell_cfg["stage_b_mean_val_avg"],
            "std_stage3_best_val": 0.0,
            "mean_stage3_test_mse": float("nan"),
            "std_stage3_test_mse": float("nan"),
            "mean_stage3_test_mase": float("nan"),
            "rollback_count": 0,
            "mean_kappa": 1.0,
            "seeds": SEEDS,
            "_provenance": "m2_1_final_topk (Option β, v2.1.7-A++)",
        },
        "phase2_top5_final": [],
    }


def _swap_winner_json(cell_cfg: dict) -> Path | None:
    """Backup current hpo_winner.json and replace with this cell's blob.
    Returns backup path (None if no original existed)."""
    backup_path = None
    if HPO_WINNER_JSON.exists():
        backup_path = HPO_WINNER_JSON.with_suffix(".json.topk_backup")
        shutil.copy2(HPO_WINNER_JSON, backup_path)
    HPO_WINNER_JSON.write_text(json.dumps(_build_winner_blob(cell_cfg), indent=2))
    return backup_path


def _restore_winner_json(backup_path: Path | None) -> None:
    if backup_path and backup_path.exists():
        shutil.move(backup_path, HPO_WINNER_JSON)


def _move_outputs_to_subdir(target_subdir: Path) -> None:
    """m2_1_final.py outputs to runs/m2_1_final/cg_mamba_winner/.
    Move that to our topk subdir so next cell doesn't overwrite.
    """
    if not M2_1_OUT_ROOT.exists():
        print(f"  ⚠ m2_1_final output dir not found: {M2_1_OUT_ROOT}")
        return
    target_subdir.parent.mkdir(parents=True, exist_ok=True)
    if target_subdir.exists():
        shutil.rmtree(target_subdir)
    shutil.move(str(M2_1_OUT_ROOT), str(target_subdir))


def run_one_cell(cell_cfg: dict, log_path: Path) -> dict:
    """Run m2_1_final.py for one cell × 5 seeds, then move outputs to subdir."""
    target_subdir = TOPK_OUT_ROOT / cell_cfg["out_subdir"]
    if target_subdir.exists() and (target_subdir / "final_summary.json").exists():
        # Resume: skip if already done
        summary = json.loads((target_subdir / "final_summary.json").read_text())
        if summary.get("n_seeds") == len(SEEDS):
            print(f"  ⏭  SKIP {cell_cfg['cell']} (already complete)")
            return {"cell": cell_cfg["cell"], "status": "skipped",
                    "summary_path": str(target_subdir / "final_summary.json")}

    # Clean stale m2_1_final output (might be leftover from another cell)
    if M2_1_OUT_ROOT.exists():
        shutil.rmtree(M2_1_OUT_ROOT)

    print(f"\n{'='*60}")
    print(f"[CELL rank {cell_cfg['rank']}] {cell_cfg['cell']}")
    print(f"  Stage B mean val={cell_cfg['stage_b_mean_val_avg']:.4f}, "
          f"tS={cell_cfg['stage_b_mean_test_strict_avg']:.4f}")
    print(f"  Output: {target_subdir.relative_to(_ROOT)}")
    print('='*60)

    backup = _swap_winner_json(cell_cfg)
    try:
        with log_path.open("a") as log:
            log.write(f"\n[{datetime.now():%H:%M:%S}] START {cell_cfg['cell']}\n")
            log.flush()
            t0 = datetime.now()
            result = subprocess.run(
                [sys.executable, "-u", str(M2_1_SCRIPT)],
                stdout=log, stderr=subprocess.STDOUT, cwd=_ROOT, check=False,
            )
            elapsed = (datetime.now() - t0).total_seconds()
            log.write(f"[{datetime.now():%H:%M:%S}] END rc={result.returncode} elapsed={elapsed:.0f}s\n")
            log.flush()
    finally:
        _restore_winner_json(backup)

    # Move outputs
    if M2_1_OUT_ROOT.exists():
        _move_outputs_to_subdir(target_subdir)

    status = "ok" if result.returncode == 0 else f"failed (rc={result.returncode})"
    print(f"  [{status}] elapsed={elapsed:.0f}s")
    return {"cell": cell_cfg["cell"], "status": status, "elapsed_sec": elapsed,
            "summary_path": str(target_subdir / "final_summary.json")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of cells (for debugging)")
    args = ap.parse_args()

    TOPK_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = _ROOT / "runs/baseline_grid_logs/m2_1_topk.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cells = TOPK_CELLS if args.limit is None else TOPK_CELLS[:args.limit]
    print(f"[M2.1 topk] {len(cells)} cells × {len(SEEDS)} seeds = {len(cells)*len(SEEDS)} runs total")
    print(f"  Seeds: {SEEDS}")

    with log_path.open("a") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"=== M2.1 topk START at {datetime.now()} ===\n")
        log.write(f"  {len(cells)} cells × {len(SEEDS)} seeds = {len(cells)*len(SEEDS)} runs\n")
        log.write(f"{'='*60}\n")

    results = []
    for cell_cfg in cells:
        r = run_one_cell(cell_cfg, log_path)
        results.append(r)

    # Write master summary
    master = {
        "protocol": "Option β — Base 2 global top-3 from E HPO Stage B 5-seed eval",
        "selected_at": "2026-05-26",
        "seeds": SEEDS,
        "cells": cells,
        "run_results": results,
    }
    (TOPK_OUT_ROOT / "master_summary.json").write_text(json.dumps(master, indent=2, default=str))
    print(f"\n=== ALL DONE ===")
    print(f"  Master summary: {(TOPK_OUT_ROOT / 'master_summary.json').relative_to(_ROOT)}")
    for r in results:
        print(f"  {r['status']:<20} {r['cell']}")
    return 0 if all(r["status"] in ("ok", "skipped") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
