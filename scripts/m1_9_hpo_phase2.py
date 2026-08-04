"""M1.9 HPO Phase 2 (v2.1.7-A) — Stage 3 4-group LR grid search.

Depends on: Phase 1 winner (`runs/m1_9_hpo_phase1/hpo_winner.json`).

3-axis grid (each ratio = LR / stage3_other_lr=1e-4):
    hmm_lr_ratio         ∈ {0.001, 0.01, 0.1, 0.5}  (4 levels — _A / _means)
    state_embed_lr_ratio ∈ {0.001, 0.01, 0.1, 0.5}  (4 levels — state_embeddings)
    env_lr_ratio         ∈ {0.001, 0.01, 0.1, 0.5}  (4 levels — env_module.encoder)

Two-stage HPO Phase 2 (mode):
    --mode grid   (Stage A): 4×4×4=64 cells × Phase 1 top-3 base × 1 seed = 192 runs
                              Selection by mean stage3_best_val
    --mode final  (Stage B): top-5 cells × 5 seeds = 25 runs
                              CG-Mamba-style 5-seed protocol matching LSTM/PatchTST/iTransformer

Stage 1 + Stage 2 reuse:
    - HMM ckpts: existing per-seed (Phase 1 reuse)
    - Env ckpt: single (Phase 1 reuse)
    - Stage 2 ckpts: `runs/m1_7_train/hpo_p1_g{gate}_b{bb}_lb{lb}_s{seed}/best.pt`
      (top-3 Phase 1 base cells)

Output:
    runs/m1_8_stage3_train/hpo_p2_<run_name>/   (Stage 3 per cell × seed)
    runs/m1_9_hpo_phase2/
        hpo_summary.csv          # all runs flat
        hpo_summary.json         # aggregated by cell (cross-seed mean ± std)
        hpo_winner.json          # phase2 winner cell + final_cfg
        hpo_top5.json            # top-5 cells (for Stage B input)
        progress.log

History (v2.1.7 / v2.1.7-A — bug + redesign):
    v2.1.7 C-1: previously 1-axis (context_lr_ratio) sweep was a no-op due to
                monkey-patch ineffective bug. Fixed by cfg fields + (model, cfg)
                signature. PLAN body 정정.
    v2.1.7-A:   redesigned as 3-axis sweep (hmm / state_embed / env) over 4-group
                Stage 3 optimizer. PLAN §5.3 variable name "context_lr_ratio" →
                3 explicit ratios. Phase 1 top-3 base coverage retained.
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
from scripts.m1_8_stage3_train import stage3_train                               # noqa: E402


# v2.1.7-A+ Option F: 5-level log-uniform full range (cross-axis coverage)
# LR mapping (× OTHER_LR_BASE=1e-4):
#   0.001 → 1e-7 (near-freeze) | 0.01 → 1e-6 | 0.1 → 1e-5 (conservative)
#   1.0   → 1e-4 (active)      | 10  → 1e-3 (aggressive)
# Captures cross-axis combinations (e.g., hmm aggressive × env freeze) that
# disjoint Stage A (small box) + Option C (large box) would have missed.
HMM_LR_RATIO_GRID         = [0.001, 0.01, 0.1, 1.0, 10.0]
STATE_EMBED_LR_RATIO_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
ENV_LR_RATIO_GRID         = [0.001, 0.01, 0.1, 1.0, 10.0]

GRID_SEED = 42                    # Stage A: 1-seed (cell selection by val_total)
FINAL_SEEDS = [42, 123, 456, 789, 1024]   # Stage B: 5-seed final (matches LSTM/PatchTST/iTrans)
SEEDS = [42, 123, 456]            # legacy 3-seed (Phase 1 retained)
OTHER_LR_BASE = 1e-4              # Stage 3 encoder_decoder LR base; all ratios are × this
TOP_K_BASE = 3                    # Phase 1 top-K bases swept in Stage A
TOP_N_FINAL = 5                   # legacy: flat top-N (kept for backward-compat output naming)
PER_BASE_TOP_K = 2                # v2.1.7-A+ : Stage B selects per-base top-K (diversification
                                  #             to mitigate val/test rank divergence)

PHASE1_WINNER_PATH = _CG_MAMBA_ROOT / "runs" / "m1_9_hpo_phase1" / "hpo_winner.json"
HMM_DIR_TEMPLATE = (
    _CG_MAMBA_ROOT / "runs" / "m1_4_phase_dynamics_main"
    / "V_raw3_regcov5e-03_K3_seed{seed}"
)
ENV_CKPT = _CG_MAMBA_ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"
HPO_ROOT = _CG_MAMBA_ROOT / "runs" / "m1_9_hpo_phase2"


def _load_phase1_top_n(n: int = 3) -> list[dict]:
    """Load Phase 1 top-N cells (sorted by mean test_mse ascending)."""
    if not PHASE1_WINNER_PATH.exists():
        raise FileNotFoundError(
            f"Phase 1 winner not found: {PHASE1_WINNER_PATH}. "
            f"Run m1_9_hpo_phase1.py first."
        )
    d = json.loads(PHASE1_WINNER_PATH.read_text())
    top = d.get("top5") or d.get("top_5") or [d["winner_cell"]]
    return top[:n]


def _phase1_stage2_dir(base: dict, seed: int) -> Path:
    """Locate Phase 1 base cell's Stage 2 ckpt directory for a given seed."""
    gate = base["gate_lr"]
    bb = base["backbone_lr"]
    lb = base["lookback"]
    run_name = f"hpo_p1_g{gate:.0e}_b{bb:.0e}_lb{lb}_s{seed}"
    return _CG_MAMBA_ROOT / "runs" / "m1_7_train" / run_name


def _base_cell_tag(base: dict) -> str:
    """Short identifier for a Phase 1 base cell (avoids exponent in dir names)."""
    return f"g{base['gate_lr']:.0e}_b{base['backbone_lr']:.0e}_lb{base['lookback']}"


def _cell_run_name(base: dict, hmm_r: float, se_r: float, env_r: float, seed: int) -> str:
    """Run name encoding all 3 ratios (v2.1.7-A 3-axis grid)."""
    return (f"hpo_p2_{_base_cell_tag(base)}"
            f"_h{hmm_r}_se{se_r}_e{env_r}_s{seed}")


def _build_cfg(
    base: dict, seed: int,
    hmm_ratio: float = 0.1,
    state_embed_ratio: float = 0.1,
    env_ratio: float = 1.0,
) -> CGMambaConfig:
    """Build cfg matching a Phase 1 base cell + 3 Stage 3 LR ratios (v2.1.7-A).

    v2.1.7-A 4-group split: each of hmm / state_embed / env / encoder_decoder
    has its own LR. Sweep ratios are × OTHER_LR_BASE (the encoder_decoder base LR).
    """
    return dataclasses.replace(
        CGMambaConfig(),
        stage2_gate_lr=base["gate_lr"],
        stage2_backbone_lr=base["backbone_lr"],
        lookback=base["lookback"],
        seed=seed,
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * hmm_ratio,
        stage3_state_embed_lr=OTHER_LR_BASE * state_embed_ratio,
        stage3_env_lr=OTHER_LR_BASE * env_ratio,
        # dropout=0.0 default — Stage 2 re-training NOT needed
    )


def _cell_tag(base: dict, hmm_r: float, se_r: float, env_r: float) -> str:
    """Cell-level tag (no seed). Used for aggregation across seeds."""
    return f"hpo_p2_{_base_cell_tag(base)}_h{hmm_r}_se{se_r}_e{env_r}"


def _run_one(base, hmm_r, se_r, env_r, seed, batch_size, progress_log) -> dict:
    """Run Stage 3 for one (base, hmm_r, state_embed_r, env_r, seed) tuple.

    v2.1.7-A: 3-axis sweep over 4-group Stage 3 optimizer.
    Stage 2 ckpt at runs/m1_7_train/hpo_p1_<base_tag>_s<seed>/best.pt is reused.
    """
    run_name = _cell_run_name(base, hmm_r, se_r, env_r, seed)
    cfg = _build_cfg(base, seed,
                     hmm_ratio=hmm_r, state_embed_ratio=se_r, env_ratio=env_r)
    hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))
    stage2_dir = _phase1_stage2_dir(base, seed)

    base_result = {
        "cell": _cell_tag(base, hmm_r, se_r, env_r),
        "run_name": run_name,
        "base_gate_lr": base["gate_lr"],
        "base_backbone_lr": base["backbone_lr"],
        "base_lookback": base["lookback"],
        "base_phase1_test_mse": base.get("mean_test_mse", float("nan")),
        "hmm_lr_ratio": hmm_r,
        "state_embed_lr_ratio": se_r,
        "env_lr_ratio": env_r,
        "seed": seed,
    }

    # Verify Stage 2 ckpt exists
    if not (stage2_dir / "best.pt").exists():
        progress_log.write(
            f"[{datetime.now():%H:%M:%S}] SKIP {run_name}  "
            f"Phase 1 Stage 2 ckpt missing: {stage2_dir / 'best.pt'}\n"
        )
        progress_log.flush()
        return {**base_result, "ok": False,
                "stage3_test_mse": float("nan"), "stage3_test_mase": float("nan"),
                "stage3_best_val": float("nan"), "rollback_triggered": None,
                "final_kappa": float("nan"), "elapsed_sec": 0.0,
                "error": f"Phase 1 ckpt missing: {stage2_dir}"}

    t0 = datetime.now()
    progress_log.write(
        f"[{t0:%H:%M:%S}] START {run_name}  "
        f"h_lr={OTHER_LR_BASE*hmm_r:.0e}, se_lr={OTHER_LR_BASE*se_r:.0e}, "
        f"env_lr={OTHER_LR_BASE*env_r:.0e}, other_lr={OTHER_LR_BASE:.0e}\n"
    )
    progress_log.flush()

    s3_args = SimpleNamespace(
        smoke=False, epochs=30, patience=10,
        batch_size=batch_size,
        stage2_dir=str(stage2_dir), hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(ENV_CKPT),
        run_name=run_name,
    )

    s3_final, err = None, None
    try:
        s3_final = stage3_train(cfg, s3_args)
        ok = True
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()
    progress_log.write(
        f"[{t1:%H:%M:%S}] {'OK' if ok else 'FAIL':4s} {run_name}  elapsed={elapsed:.1f}s  "
        f"test_mse={(s3_final.get('test_mse', float('nan')) if s3_final else float('nan')):.4f}\n"
    )
    if not ok:
        progress_log.write(f"  ERROR: {err}\n")
    progress_log.flush()

    return {**base_result, "ok": ok,
            "stage3_best_val": s3_final.get("best_val_total", float("nan")) if s3_final else float("nan"),
            "stage3_test_mse": s3_final.get("test_mse", float("nan")) if s3_final else float("nan"),
            "stage3_test_mase": s3_final.get("test_mase", float("nan")) if s3_final else float("nan"),
            "rollback_triggered": s3_final.get("rollback_triggered", None) if s3_final else None,
            "final_kappa": s3_final.get("final_kappa", float("nan")) if s3_final else float("nan"),
            "elapsed_sec": elapsed}


def _stage_a_iter(bases):
    """Yield (base, hmm_r, se_r, env_r, seed) for Stage A (1-seed × top-K base × 64 cells)."""
    for b in bases:
        for hr in HMM_LR_RATIO_GRID:
            for sr in STATE_EMBED_LR_RATIO_GRID:
                for er in ENV_LR_RATIO_GRID:
                    yield (b, hr, sr, er, GRID_SEED)


def _resume_result(cell_dir: Path, base, hmm_r, se_r, env_r, seed) -> dict | None:
    """Reconstruct result dict from existing final_metrics.json (resume mode)."""
    fm = cell_dir / "final_metrics.json"
    if not fm.exists():
        return None
    f = json.loads(fm.read_text())
    return {
        "cell": _cell_tag(base, hmm_r, se_r, env_r),
        "run_name": _cell_run_name(base, hmm_r, se_r, env_r, seed),
        "base_gate_lr": base["gate_lr"], "base_backbone_lr": base["backbone_lr"],
        "base_lookback": base["lookback"],
        "base_phase1_test_mse": base.get("mean_test_mse", float("nan")),
        "hmm_lr_ratio": hmm_r, "state_embed_lr_ratio": se_r, "env_lr_ratio": env_r,
        "seed": seed, "ok": True,
        "stage3_best_val": f.get("best_val_total", float("nan")),
        "stage3_test_mse": f.get("test_mse", float("nan")),
        "stage3_test_mase": f.get("test_mase", float("nan")),
        "rollback_triggered": f.get("rollback_triggered"),
        "final_kappa": f.get("final_kappa", float("nan")),
        "elapsed_sec": f.get("elapsed_sec", 0.0),
    }


def _run_stage_a(args) -> int:
    """Stage A: 4×4×4 = 64 cells × Phase 1 top-K base × 1 seed.

    Selection: val_total (= stage3_best_val) ascending.
    Saves: hpo_summary.{csv,json}, hpo_top5.json (input for Stage B).
    """
    bases = _load_phase1_top_n(TOP_K_BASE)
    cells = list(_stage_a_iter(bases))
    if args.smoke:
        cells = cells[:2]
    total = len(cells) if not args.limit else min(len(cells), args.limit)

    print(f"[HPO Phase 2 Stage A] grid search (1-seed)")
    for i, b in enumerate(bases, 1):
        print(f"  base #{i}: gate_lr={b['gate_lr']:.0e}  backbone_lr={b['backbone_lr']:.0e}  "
              f"lookback={b['lookback']}  phase1_mean_test_mse={b['mean_test_mse']:.4f}")
    print(f"  Grid: {len(bases)} bases × "
          f"{len(HMM_LR_RATIO_GRID)}×{len(STATE_EMBED_LR_RATIO_GRID)}×{len(ENV_LR_RATIO_GRID)} ratios "
          f"× 1 seed (GRID_SEED={GRID_SEED}) = {len(cells)} runs (~{len(cells)*3}min)")

    HPO_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    with (HPO_ROOT / "progress.log").open("a") as progress_log:
        progress_log.write(f"\n=== M1.9 HPO Phase 2 Stage A (4×4×4 × top-{len(bases)} × 1-seed) "
                           f"start {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        for i, b in enumerate(bases, 1):
            progress_log.write(f"  base #{i}: {b}\n")
        progress_log.write(f"Total: {total}\n\n")
        progress_log.flush()
        for i, (base, hr, sr, er, seed) in enumerate(cells[:total]):
            run_name = _cell_run_name(base, hr, sr, er, seed)
            cell_dir = _CG_MAMBA_ROOT / "runs" / "m1_8_stage3_train" / run_name
            if args.resume:
                cached = _resume_result(cell_dir, base, hr, sr, er, seed)
                if cached is not None:
                    results.append(cached)
                    print(f"  [{i+1:3d}/{total}] SKIP {run_name}  "
                          f"test_mse={cached['stage3_test_mse']:.4f}")
                    continue
            print(f"  [{i+1:3d}/{total}] RUN  {run_name}")
            r = _run_one(base, hr, sr, er, seed, args.batch_size, progress_log)
            results.append(r)
            _save_summary_stage_a(results)

    summary = _aggregate_stage_a(results)
    _save_top_n_for_stage_b(summary, bases)
    print()
    print("=== HPO Phase 2 Stage A summary ===")
    print(f"  Completed: {sum(1 for r in results if r['ok'])}/{total}")
    print(f"  Failed:    {sum(1 for r in results if not r['ok'])}")
    print(f"  Selection criterion: val_total (= stage3_best_val) ascending")
    print(f"  Top-{TOP_N_FINAL} cells saved to hpo_top5.json for Stage B input")
    for i, c in enumerate(summary["cells"][:TOP_N_FINAL], 1):
        print(f"  #{i}: {c['cell']}  best_val={c['stage3_best_val']:.4f}  test_mse={c['stage3_test_mse']:.4f}")
    return 0


def _run_stage_b(args) -> int:
    """Stage B: top-N cells (from Stage A) × 5 seeds = 25 runs.

    Selection: val_total mean across 5 seeds (paper protocol, matches LSTM/PatchTST/iTrans).
    Saves: hpo_summary_final.json, hpo_winner.json.
    """
    top_path = HPO_ROOT / "hpo_top5.json"
    if not top_path.exists():
        raise FileNotFoundError(
            f"{top_path} not found. Run --mode grid first to produce top-N cells."
        )
    top_cells = json.loads(top_path.read_text())["cells"]
    bases_lookup = {(c["base_gate_lr"], c["base_backbone_lr"], c["base_lookback"]):
                    {"gate_lr": c["base_gate_lr"], "backbone_lr": c["base_backbone_lr"],
                     "lookback": c["base_lookback"],
                     "mean_test_mse": c.get("base_phase1_test_mse", float("nan"))}
                    for c in top_cells}

    n_seeds = len(FINAL_SEEDS)
    total = len(top_cells) * n_seeds
    print(f"[HPO Phase 2 Stage B] 5-seed final over top-{len(top_cells)} cells from Stage A")
    for i, c in enumerate(top_cells, 1):
        print(f"  #{i}: {c['cell']}  Stage A val_total={c['stage3_best_val']:.4f}")
    print(f"  Total runs: {len(top_cells)} cells × {n_seeds} seeds = {total}")

    HPO_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    with (HPO_ROOT / "progress.log").open("a") as progress_log:
        progress_log.write(f"\n=== M1.9 HPO Phase 2 Stage B (top-{len(top_cells)} × {n_seeds}-seed) "
                           f"start {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        progress_log.flush()
        i = 0
        for c in top_cells:
            base = bases_lookup[(c["base_gate_lr"], c["base_backbone_lr"], c["base_lookback"])]
            hr, sr, er = c["hmm_lr_ratio"], c["state_embed_lr_ratio"], c["env_lr_ratio"]
            for seed in FINAL_SEEDS:
                i += 1
                run_name = _cell_run_name(base, hr, sr, er, seed)
                cell_dir = _CG_MAMBA_ROOT / "runs" / "m1_8_stage3_train" / run_name
                if args.resume:
                    cached = _resume_result(cell_dir, base, hr, sr, er, seed)
                    if cached is not None:
                        results.append(cached)
                        print(f"  [{i:2d}/{total}] SKIP {run_name}  "
                              f"test_mse={cached['stage3_test_mse']:.4f}")
                        continue
                print(f"  [{i:2d}/{total}] RUN  {run_name}")
                r = _run_one(base, hr, sr, er, seed, args.batch_size, progress_log)
                results.append(r)
                _save_summary_stage_b(results)

    summary = _aggregate_stage_b(results)
    _save_winner_final(summary)
    print()
    print("=== HPO Phase 2 Stage B summary ===")
    print(f"  Completed: {sum(1 for r in results if r['ok'])}/{total}")
    if summary["cells"]:
        w = summary["cells"][0]
        print(f"  Final winner: {w['cell']}")
        print(f"    val_total mean = {w['mean_stage3_best_val']:.4f} ± {w['std_stage3_best_val']:.4f}")
        print(f"    test_mse mean  = {w['mean_stage3_test_mse']:.4f} ± {w['std_stage3_test_mse']:.4f}")
        print(f"    n_seeds = {w['n_seeds']} ({FINAL_SEEDS})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="M1.9 HPO Phase 2 v2.1.7-A — 4-group Stage 3 grid")
    parser.add_argument("--mode", choices=["grid", "final"], required=True,
                        help="grid=Stage A (192 runs 1-seed); final=Stage B (top-N × 5 seeds)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "grid":
        return _run_stage_a(args)
    else:
        return _run_stage_b(args)


# ───────────────────────── Save / aggregate helpers ─────────────────────────

def _save_summary_stage_a(results: list[dict]) -> None:
    csv_path = HPO_ROOT / "hpo_summary.csv"
    if not results:
        return
    # v2.1.7-A++ fix: union of all keys across all results (failed cells may include
    # 'error' field absent from successful cells). extrasaction='ignore' as backup.
    fields = sorted(set().union(*(r.keys() for r in results)))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _save_summary_stage_b(results: list[dict]) -> None:
    csv_path = HPO_ROOT / "hpo_summary_final.csv"
    if not results:
        return
    # v2.1.7-A++ fix: union of all keys across all results (failed cells may include
    # 'error' field absent from successful cells). extrasaction='ignore' as backup.
    fields = sorted(set().union(*(r.keys() for r in results)))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _aggregate_stage_a(results: list[dict]) -> dict:
    """Stage A is 1-seed per cell, so cell-level summary = single result.
    Sort by stage3_best_val (val_total) ascending — paper-defensible (no test leakage).
    """
    cells = []
    for r in results:
        if not r["ok"]:
            continue
        cells.append({
            "cell": r["cell"],
            "base_gate_lr": r["base_gate_lr"], "base_backbone_lr": r["base_backbone_lr"],
            "base_lookback": r["base_lookback"],
            "base_phase1_test_mse": r["base_phase1_test_mse"],
            "hmm_lr_ratio": r["hmm_lr_ratio"],
            "state_embed_lr_ratio": r["state_embed_lr_ratio"],
            "env_lr_ratio": r["env_lr_ratio"],
            "stage3_best_val": r["stage3_best_val"],
            "stage3_test_mse": r["stage3_test_mse"],
            "stage3_test_mase": r["stage3_test_mase"],
            "rollback_triggered": r["rollback_triggered"],
            "final_kappa": r["final_kappa"],
            "elapsed_sec": r["elapsed_sec"],
        })
    cells.sort(key=lambda c: c["stage3_best_val"])
    summary = {"cells": cells, "n_total_runs": len(results),
               "stage": "A", "selection": "stage3_best_val ascending"}
    with (HPO_ROOT / "hpo_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _save_top_n_for_stage_b(summary: dict, phase1_bases: list[dict]) -> None:
    """Stage B input selection (v2.1.7-A+):

    Per-base top-K=2 cells (val_total ascending within each base). Diversification
    mitigates val/test rank divergence — we observed corr(val, test) ∈ {−0.94,
    −0.30, +0.27} across the 3 Phase 1 bases, with the strongest negative
    correlation on lb=156 (val winner = test loser pattern). Flat top-N selection
    would lock all top cells into a single base, missing diversification value.

    File name stays `hpo_top5.json` for backward-compat with m2_1_final.py loader.
    """
    cells = summary["cells"]
    by_base: dict[str, list[dict]] = {}
    for c in cells:
        base_tag = f"g{c['base_gate_lr']:.0e}_b{c['base_backbone_lr']:.0e}_lb{int(c['base_lookback'])}"
        by_base.setdefault(base_tag, []).append(c)
    # Each base's cells are already sorted (parent cells list is val_total ascending)
    selected = []
    for base_tag, base_cells in by_base.items():
        selected.extend(base_cells[:PER_BASE_TOP_K])
    # Re-sort the union by val_total ascending for display
    selected.sort(key=lambda c: c["stage3_best_val"])
    with (HPO_ROOT / "hpo_top5.json").open("w") as f:
        json.dump({"cells": selected,
                   "phase1_top_bases": phase1_bases,
                   "selection_metric": f"per-base top-{PER_BASE_TOP_K} by stage3_best_val",
                   "per_base_k": PER_BASE_TOP_K,
                   "n_selected": len(selected),
                   "diversification_rationale": (
                       "val/test rank divergence observed (per-base corr(val,test) "
                       "ranged −0.94 to +0.27 across 3 Phase 1 bases). Per-base "
                       "top-K=2 selection ensures Stage B covers multiple bases."
                   )}, f, indent=2)


def _aggregate_stage_b(results: list[dict]) -> dict:
    """Aggregate by cell (across 5 seeds), sort by mean val_total ascending."""
    by_cell: dict[str, list[dict]] = {}
    for r in results:
        if not r["ok"]:
            continue
        by_cell.setdefault(r["cell"], []).append(r)
    cell_summary = []
    for ck, rows in by_cell.items():
        bv = [r["stage3_best_val"] for r in rows]
        ms = [r["stage3_test_mse"] for r in rows]
        cell_summary.append({
            "cell": ck,
            "base_gate_lr": rows[0]["base_gate_lr"], "base_backbone_lr": rows[0]["base_backbone_lr"],
            "base_lookback": rows[0]["base_lookback"],
            "base_phase1_test_mse": rows[0]["base_phase1_test_mse"],
            "hmm_lr_ratio": rows[0]["hmm_lr_ratio"],
            "state_embed_lr_ratio": rows[0]["state_embed_lr_ratio"],
            "env_lr_ratio": rows[0]["env_lr_ratio"],
            "n_seeds": len(rows),
            "mean_stage3_best_val": float(np.mean(bv)),
            "std_stage3_best_val": float(np.std(bv)),
            "mean_stage3_test_mse": float(np.mean(ms)),
            "std_stage3_test_mse": float(np.std(ms)),
            "mean_stage3_test_mase": float(np.mean([r["stage3_test_mase"] for r in rows])),
            "rollback_count": sum(1 for r in rows if r["rollback_triggered"]),
            "mean_kappa": float(np.mean([r["final_kappa"] for r in rows])),
            "seeds": [r["seed"] for r in rows],
        })
    cell_summary.sort(key=lambda c: c["mean_stage3_best_val"])
    summary = {"cells": cell_summary, "n_total_runs": len(results),
               "stage": "B", "selection": "mean_stage3_best_val ascending across 5 seeds"}
    with (HPO_ROOT / "hpo_summary_final.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _save_winner_final(summary: dict) -> None:
    """v2.1.7-A schema: 3 ratios + 4 LRs in final_cfg."""
    if not summary["cells"]:
        return
    w = summary["cells"][0]
    with (HPO_ROOT / "hpo_winner.json").open("w") as f:
        json.dump({
            "phase2_winner": w,
            "phase2_top5_final": summary["cells"][:5],
            "final_cfg": {
                "gate_lr": w["base_gate_lr"],
                "backbone_lr": w["base_backbone_lr"],
                "lookback": w["base_lookback"],
                "dropout": 0.0,
                "hmm_lr_ratio": w["hmm_lr_ratio"],
                "state_embed_lr_ratio": w["state_embed_lr_ratio"],
                "env_lr_ratio": w["env_lr_ratio"],
                "stage3_hmm_lr": OTHER_LR_BASE * w["hmm_lr_ratio"],
                "stage3_state_embed_lr": OTHER_LR_BASE * w["state_embed_lr_ratio"],
                "stage3_env_lr": OTHER_LR_BASE * w["env_lr_ratio"],
                "stage3_other_lr": OTHER_LR_BASE,
            },
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
