"""M1.9 HPO Phase 1 — Stage 2 hyperparameter grid search.

Grid (PLAN §5.3, sweep candidates after user decision; v2.1.x lookback reduced):
    stage2_gate_lr      ∈ {5e-4, 1e-3, 2e-3}   (3 levels)
    stage2_backbone_lr  ∈ {1e-5, 5e-5, 1e-4}   (3 levels)
    lookback            ∈ {104, 156}           (2 levels; 260 dropped, see L65)
    seed                ∈ {42, 123, 456}       (3 seeds)
    → 3·3·2 cells × 3 seeds = 54 runs

Excluded (user decision, justified):
    - stage1_lr:       HMM EM is closed-form (LR-invariant) + EnvMLP autoencoder
                       reaches 97% MSE drop at default 1e-3 (50ep). Reusing the
                       single existing Env ckpt for all cells.
    - dropout:         Deferred to Phase 2 (combined with Stage 3 fine-tune HPO).
    - depth, d_model:  Deferred to M2.5 architecture ablation.
    - gate_bias_init:  M1.3 ablation already confirmed default=2.0.

Stage 1 reuse (time saving ~24min total):
    - HMM ckpts: existing per-seed `runs/m1_4_phase_dynamics_main/
      V_raw3_regcov5e-03_K3_seed{42,123,456}/` (M-step converges identically
      regardless of H-3 fix, so reuse is safe).
    - Env ckpt:  single `runs/m1_7_env_pretrain/env_encoder.pt` (ILI-blind →
      seed-invariant within EnvMLP convergence bound).

Output:
    runs/m1_7_train/hpo_p1_<run_name>/   # per cell × seed (m1_7_train default dir)
        best.pt, history.json, final_metrics.json
    runs/m1_9_hpo_phase1/                # HPO-level aggregation only
        hpo_summary.csv                 # all 54 runs flat
        hpo_summary.json                # nested by (gate_lr, backbone_lr, lookback)
        hpo_winner.json                 # best cell (mean test_mse over 3 seeds)
        progress.log                    # tail-friendly progress
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[1]
if str(_CG_MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CG_MAMBA_ROOT))

from src.utils.config import CGMambaConfig                                      # noqa: E402
from scripts.m1_7_train import train as stage2_train                             # noqa: E402


# ───────────────────────────────────────────────────────────────────
# HP grids
# ───────────────────────────────────────────────────────────────────
GATE_LR_GRID = [5e-4, 1e-3, 2e-3]
BACKBONE_LR_GRID = [1e-5, 5e-5, 1e-4]
# v2.1.x revision (2026-05-21): lookback=260 removed after ETA review.
# 5 years of weekly ILI history (260w) does not provide meaningfully more signal
# than 3 years (156w) for 1-4 week ahead forecast, and lb=260 cells were
# projected to add ~12h to wall-clock. User decision: keep {104, 156} only.
LOOKBACK_GRID = [104, 156]
SEEDS = [42, 123, 456]

# Stage 1 ckpt paths (per seed for HMM, single for Env)
HMM_DIR_TEMPLATE = (
    _CG_MAMBA_ROOT / "runs" / "m1_4_phase_dynamics_main"
    / "V_raw3_regcov5e-03_K3_seed{seed}"
)
ENV_CKPT = _CG_MAMBA_ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"
HPO_ROOT = _CG_MAMBA_ROOT / "runs" / "m1_9_hpo_phase1"


def _check_stage1_ckpts() -> None:
    """Verify all required Stage 1 ckpts exist before launching 81 runs."""
    missing = []
    for seed in SEEDS:
        hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))
        if not (hmm_dir / "hmm_params.npz").exists():
            missing.append(f"  HMM seed={seed}: {hmm_dir / 'hmm_params.npz'}")
    if not ENV_CKPT.exists():
        missing.append(f"  Env: {ENV_CKPT}")
    if missing:
        raise FileNotFoundError(
            "Stage 1 ckpts missing — cannot run HPO Phase 1:\n"
            + "\n".join(missing)
            + "\nRun m1_4_phase_dynamics_main.py (multi-seed) and m1_7_env_pretrain.py first."
        )


def _cell_run_name(gate_lr: float, backbone_lr: float, lookback: int, seed: int) -> str:
    # `hpo_p1_` prefix so m1_7_train output dir (`runs/m1_7_train/hpo_p1_<name>`)
    # is easy to filter/glob from non-HPO runs.
    return f"hpo_p1_g{gate_lr:.0e}_b{backbone_lr:.0e}_lb{lookback}_s{seed}"


def _build_cell_cfg(gate_lr: float, backbone_lr: float, lookback: int, seed: int) -> CGMambaConfig:
    """Build a frozen CGMambaConfig for one HPO cell × seed."""
    return dataclasses.replace(
        CGMambaConfig(),
        stage2_gate_lr=gate_lr,
        stage2_backbone_lr=backbone_lr,
        lookback=lookback,
        seed=seed,
    )


def _build_cell_args(run_name: str, seed: int, batch_size: int) -> SimpleNamespace:
    """Build args namespace matching m1_7_train.py CLI signature.

    `run_name` becomes `runs/m1_7_train/<run_name>/` per m1_7_train.py:162.
    We embed `hpo_p1_` prefix in the name itself rather than a `..` path
    traversal (which would break `out_root.relative_to(_CG_MAMBA_ROOT)`).
    """
    hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))
    return SimpleNamespace(
        smoke=False,
        epochs=None,                            # use cfg.stage2_n_epochs default (200)
        batch_size=batch_size,
        hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(ENV_CKPT),
        wandb_mode="disabled",
        run_name=run_name,
    )


def _run_one_cell(gate_lr, backbone_lr, lookback, seed, batch_size, progress_log) -> dict:
    """Train one (cfg, seed) cell and return its final_metrics dict + cell HP info."""
    run_name = _cell_run_name(gate_lr, backbone_lr, lookback, seed)
    cfg = _build_cell_cfg(gate_lr, backbone_lr, lookback, seed)
    args = _build_cell_args(run_name, seed, batch_size)

    t0 = datetime.now()
    progress_log.write(f"[{t0:%H:%M:%S}] START {run_name}\n")
    progress_log.flush()
    try:
        final = stage2_train(cfg, args)
        ok = True
        err = None
    except Exception as e:
        final = {"best_val_total": float("inf"), "test_mse": float("nan"),
                 "test_mase": float("nan"), "n_epochs_run": 0,
                 "elapsed_sec": (datetime.now() - t0).total_seconds(),
                 "best_epoch": -1}
        ok = False
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()
    status = "OK" if ok else "FAIL"
    progress_log.write(
        f"[{t1:%H:%M:%S}] {status:4s} {run_name}  "
        f"elapsed={elapsed:.1f}s  test_mse={final.get('test_mse', float('nan')):.4f}\n"
    )
    if not ok:
        progress_log.write(f"  ERROR: {err}\n")
    progress_log.flush()

    return {
        "cell": f"hpo_p1_g{gate_lr:.0e}_b{backbone_lr:.0e}_lb{lookback}",
        "run_name": run_name,
        "gate_lr": gate_lr,
        "backbone_lr": backbone_lr,
        "lookback": lookback,
        "seed": seed,
        "ok": ok,
        "best_epoch": final.get("best_epoch", -1),
        "best_val_total": final.get("best_val_total", float("inf")),
        "best_val_mse": final.get("best_val_mse", float("nan")),
        "best_val_mase": final.get("best_val_mase", float("nan")),
        "test_total": final.get("test_total", float("nan")),
        "test_mse": final.get("test_mse", float("nan")),
        "test_mase": final.get("test_mase", float("nan")),
        "elapsed_sec": elapsed,
        "n_epochs_run": final.get("n_epochs_run", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M1.9 HPO Phase 1 — Stage 2 grid")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells with existing final_metrics.json")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny smoke: 1×1×1 cell × 1 seed only (sanity)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N runs (debugging)")
    args = parser.parse_args()

    _check_stage1_ckpts()
    HPO_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Build run list ──
    if args.smoke:
        cells = [(1e-3, 5e-5, 156, 42)]    # 1 cell × 1 seed
    else:
        cells = [(g, b, l, s)
                 for g in GATE_LR_GRID
                 for b in BACKBONE_LR_GRID
                 for l in LOOKBACK_GRID
                 for s in SEEDS]
    total = len(cells) if not args.limit else min(len(cells), args.limit)
    print(f"[HPO Phase 1] {total} runs queued "
          f"({len(GATE_LR_GRID)}×{len(BACKBONE_LR_GRID)}×{len(LOOKBACK_GRID)} cells × {len(SEEDS)} seeds)")
    print(f"  Stage 1 reuse: HMM per-seed + Env single ckpt")
    print(f"  Output: {HPO_ROOT.relative_to(_CG_MAMBA_ROOT)}/")
    print(f"  Resume: {args.resume}")

    # ── Execute ──
    results = []
    progress_path = HPO_ROOT / "progress.log"
    with progress_path.open("a") as progress_log:
        progress_log.write(f"\n=== M1.9 HPO Phase 1 start {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        progress_log.write(f"Total runs: {total}\n\n")
        progress_log.flush()
        for i, (gate_lr, backbone_lr, lookback, seed) in enumerate(cells[:total]):
            run_name = _cell_run_name(gate_lr, backbone_lr, lookback, seed)
            cell_dir = _CG_MAMBA_ROOT / "runs" / "m1_7_train" / run_name
            final_metrics_path = cell_dir / "final_metrics.json"

            if args.resume and final_metrics_path.exists():
                # Resume: load existing result
                final = json.loads(final_metrics_path.read_text())
                results.append({
                    "cell": f"hpo_p1_g{gate_lr:.0e}_b{backbone_lr:.0e}_lb{lookback}",
                    "run_name": run_name,
                    "gate_lr": gate_lr, "backbone_lr": backbone_lr,
                    "lookback": lookback, "seed": seed, "ok": True,
                    "best_epoch": final.get("best_epoch", -1),
                    "best_val_total": final.get("best_val_total", float("inf")),
                    "best_val_mse": final.get("best_val_mse", float("nan")),
                    "best_val_mase": final.get("best_val_mase", float("nan")),
                    "test_total": final.get("test_total", float("nan")),
                    "test_mse": final.get("test_mse", float("nan")),
                    "test_mase": final.get("test_mase", float("nan")),
                    "elapsed_sec": final.get("elapsed_sec", 0.0),
                    "n_epochs_run": final.get("n_epochs_run", 0),
                })
                print(f"  [{i+1:3d}/{total}] SKIP (resume) {run_name}  test_mse={final.get('test_mse', float('nan')):.4f}")
                continue

            print(f"  [{i+1:3d}/{total}] RUN  {run_name}")
            r = _run_one_cell(gate_lr, backbone_lr, lookback, seed, args.batch_size, progress_log)
            results.append(r)

            # Append to CSV after each run (incremental save in case of crash)
            _save_summary(results)

    # ── Final aggregation ──
    summary = _aggregate(results)
    _save_winner(summary)

    print()
    print("=== HPO Phase 1 summary ===")
    print(f"  Completed: {sum(1 for r in results if r['ok'])}/{total}")
    print(f"  Failed:    {sum(1 for r in results if not r['ok'])}")
    if summary["cells"]:
        winner = summary["cells"][0]
        print(f"  Winner cell: gate_lr={winner['gate_lr']:.0e}  "
              f"backbone_lr={winner['backbone_lr']:.0e}  lookback={winner['lookback']}")
        print(f"               mean test_mse={winner['mean_test_mse']:.4f} "
              f"(±{winner['std_test_mse']:.4f})")
        print(f"               default v3 baseline: test_mse=0.2908")
    return 0


def _save_summary(results: list[dict]) -> None:
    """Write hpo_summary.csv after each run (incremental, crash-safe)."""
    csv_path = HPO_ROOT / "hpo_summary.csv"
    fields = ["cell", "run_name", "gate_lr", "backbone_lr", "lookback", "seed", "ok",
              "best_epoch", "best_val_total", "best_val_mse", "best_val_mase",
              "test_total", "test_mse", "test_mase", "elapsed_sec", "n_epochs_run"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _aggregate(results: list[dict]) -> dict:
    """Aggregate by cell (mean+std over seeds) and sort by mean test_mse."""
    by_cell: dict[str, list[dict]] = {}
    for r in results:
        if not r["ok"]:
            continue
        cell_key = f"hpo_p1_g{r['gate_lr']:.0e}_b{r['backbone_lr']:.0e}_lb{r['lookback']}"
        by_cell.setdefault(cell_key, []).append(r)

    cell_summary = []
    for cell_key, rows in by_cell.items():
        test_mses = [r["test_mse"] for r in rows]
        val_totals = [r["best_val_total"] for r in rows]
        cell_summary.append({
            "cell": cell_key,
            "gate_lr": rows[0]["gate_lr"],
            "backbone_lr": rows[0]["backbone_lr"],
            "lookback": rows[0]["lookback"],
            "n_seeds": len(rows),
            "mean_test_mse": float(np.mean(test_mses)),
            "std_test_mse": float(np.std(test_mses)),
            "mean_val_total": float(np.mean(val_totals)),
            "std_val_total": float(np.std(val_totals)),
            "mean_test_mase": float(np.mean([r["test_mase"] for r in rows])),
            "mean_best_epoch": float(np.mean([r["best_epoch"] for r in rows])),
            "seeds": [r["seed"] for r in rows],
        })
    # Sort by mean test_mse ascending (lower is better)
    cell_summary.sort(key=lambda c: c["mean_test_mse"])

    summary = {"cells": cell_summary,
               "n_total_runs": len(results),
               "n_ok": sum(1 for r in results if r["ok"])}
    with (HPO_ROOT / "hpo_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _save_winner(summary: dict) -> None:
    if not summary["cells"]:
        return
    winner = summary["cells"][0]
    with (HPO_ROOT / "hpo_winner.json").open("w") as f:
        json.dump({
            "winner_cell": winner,
            "top5": summary["cells"][:5],
            "instructions": (
                "Next: run m1_9_hpo_phase2.py with this winner cfg as base, "
                "sweeping dropout × context_lr_ratio."
            ),
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
