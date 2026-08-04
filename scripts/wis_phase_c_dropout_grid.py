"""WIS Phase C — symmetric 3-model dropout grid retrain (PLAN §16 J.4 β option).

Drives 45 runs sequential on 1 GPU:
    LSTM (1 cell)          × dropout ∈ {0.1, 0.2, 0.3} × 5 seeds = 15 runs  (~5 min)
    Vanilla Mamba (1 cell) × 3 dropouts × 5 seeds                = 15 runs  (~21 min)
    CG-Mamba top1 (S2+S3)  × 3 dropouts × 5 seeds                = 15 runs  (~2.1h)
                                                                  ────────────────
    Total                                                        = 45 runs  (~2.5h)

β option: CG-Mamba 의 Stage 2 + Stage 3 모두 새 dropout 으로 처음부터 재학습.
Stage 1 (HMM) 은 dropout 무관 + deterministic → 기존 ckpt 재사용.

Output layout:
    runs/wis_phase_c/
    ├── manifest.json                     ← single source of truth for Phase D
    ├── lstm/d{0.1,0.2,0.3}/seed{...}/    ← train_one_run out_dir 직접
    ├── vanilla_mamba/d{...}/seed{...}/   ← same
    └── cg_mamba/d{...}/seed{...}/        ← stage2_link + stage3_link (manifest only)
        (실제 ckpt 는 runs/m1_7_train/wis_phase_c_cg_mamba_*/와
         runs/m1_8_stage3_train/wis_phase_c_cg_mamba_*/에 저장)

Top1 cell HP (from M2.1):
    base_gate_lr=1e-3, base_backbone_lr=1e-4, lookback=104
    hmm_lr_ratio=0.01, state_embed_lr_ratio=0.01, env_lr_ratio=0.001
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# LSTM + Vanilla Mamba (1-stage trainers)
from scripts.run_lstm_weekly import train_one_run as lstm_train_one_run  # noqa: E402
from scripts.run_vanilla_mamba_weekly import train_one_run as vanilla_train_one_run  # noqa: E402

# CG-Mamba multi-stage
from scripts.m1_7_train import train as stage2_train                       # noqa: E402
from scripts.m1_8_stage3_train import stage3_train                         # noqa: E402
from src.utils.config import CGMambaConfig                                 # noqa: E402

# ─── Constants ─────────────────────────────────────────────────────────────
DROPOUTS = (0.1, 0.2, 0.3)
SEEDS = (42, 123, 456, 789, 1024)

OUT_ROOT = _ROOT / "runs" / "wis_phase_c"
MANIFEST_PATH = OUT_ROOT / "manifest.json"
LOG_PATH = OUT_ROOT / "phase_c.log"

CSV_PATH = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data" / "processed" / "normalization_params.json"

HMM_DIR_TEMPLATE = (
    _ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed{seed}"
)
ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"
OTHER_LR_BASE = 1e-4

# Top1 cell HP (PLAN §16 J.4, M2.1 winner)
CG_MAMBA_TOP1_HP = {
    "gate_lr": 1e-3,
    "backbone_lr": 1e-4,
    "lookback": 104,
    "hmm_lr_ratio": 0.01,
    "state_embed_lr_ratio": 0.01,
    "env_lr_ratio": 0.001,
}

# Winner cells from M2.3 (Pattern A)
LSTM_HP = {
    "lookback": 104,
    "pred_len": 4,
    "enc_in": 6,
    "hidden": 256,
    "num_layers": 2,
    "lr": 5e-4,
    "batch_size": 16,
    "epochs": 100,
    "patience": 20,
    # dropout: overridden per run
}

VANILLA_MAMBA_HP = {
    "seq_len": 104,
    "pred_len": 4,
    "enc_in": 6,
    "d_model": 64,
    "n_layers": 3,
    "d_state": 16,
    "dt_rank": 16,
    "expand": 2,
    "lr": 5e-4,
    "batch_size": 32,
    "epochs": 200,
    "patience": 20,
    # dropout: overridden per run
}


# ─── Helpers ───────────────────────────────────────────────────────────────


def _log(msg: str, log_fh) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"runs": []}


def _save_manifest(manifest: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _record_run(manifest: dict, entry: dict) -> None:
    # Dedup by (model, dropout, seed)
    key = (entry["model"], entry["dropout"], entry["seed"])
    manifest["runs"] = [r for r in manifest["runs"]
                        if (r["model"], r["dropout"], r["seed"]) != key]
    manifest["runs"].append(entry)
    _save_manifest(manifest)


def _has_completed(manifest: dict, model: str, dropout: float, seed: int) -> bool:
    return any(
        r["model"] == model and r["dropout"] == dropout and r["seed"] == seed
        and r.get("status") == "ok"
        for r in manifest["runs"]
    )


# ─── Per-model launchers ───────────────────────────────────────────────────


def _run_lstm(dropout: float, seed: int, device: str, log_fh) -> dict:
    cfg = {**LSTM_HP, "dropout": dropout}
    out_dir = OUT_ROOT / "lstm" / f"d{dropout}" / f"seed{seed}"
    t0 = time.time()
    r = lstm_train_one_run(
        cfg=cfg, seed=seed, csv_path=CSV_PATH, norm_path=NORM_PATH,
        device=device, out_dir=out_dir,
        wandb_enabled=False,
        wandb_run_name=f"wis_phase_c_lstm_d{dropout}_s{seed}",
    )
    elapsed = time.time() - t0
    return {
        "model": "lstm", "dropout": dropout, "seed": seed,
        "status": "ok",
        "out_dir": str(out_dir.relative_to(_ROOT)),
        "ckpt": str((out_dir / "lstm_best.pt").relative_to(_ROOT)),
        "val_mae_h1": r.get("best_val_mae_h1", float("nan")),
        "best_epoch": r.get("best_epoch"),
        "elapsed_sec": elapsed,
    }


def _run_vanilla_mamba(dropout: float, seed: int, device: str, log_fh) -> dict:
    cfg = {**VANILLA_MAMBA_HP, "dropout": dropout}
    out_dir = OUT_ROOT / "vanilla_mamba" / f"d{dropout}" / f"seed{seed}"
    t0 = time.time()
    r = vanilla_train_one_run(
        cfg=cfg, seed=seed, csv_path=CSV_PATH, norm_path=NORM_PATH,
        device=device, out_dir=out_dir,
        wandb_enabled=False,
        wandb_run_name=f"wis_phase_c_vanilla_d{dropout}_s{seed}",
    )
    elapsed = time.time() - t0
    return {
        "model": "vanilla_mamba", "dropout": dropout, "seed": seed,
        "status": "ok",
        "out_dir": str(out_dir.relative_to(_ROOT)),
        "ckpt": str((out_dir / "vanilla_mamba_best.pt").relative_to(_ROOT)),
        "val_mae_h1": r.get("best_val_mae_h1", float("nan")),
        "best_epoch": r.get("best_epoch"),
        "elapsed_sec": elapsed,
    }


def _build_cg_mamba_cfg(dropout: float, seed: int) -> CGMambaConfig:
    """CGMambaConfig from top1 cell HP + new dropout (PLAN J.4 β option)."""
    hp = CG_MAMBA_TOP1_HP
    return dataclasses.replace(
        CGMambaConfig(),
        seed=seed,
        dropout=dropout,                                # ← NEW
        lookback=hp["lookback"],
        stage2_gate_lr=hp["gate_lr"],
        stage2_backbone_lr=hp["backbone_lr"],
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * hp["hmm_lr_ratio"],
        stage3_state_embed_lr=OTHER_LR_BASE * hp["state_embed_lr_ratio"],
        stage3_env_lr=OTHER_LR_BASE * hp["env_lr_ratio"],
    )


def _run_cg_mamba(dropout: float, seed: int, device: str, log_fh,
                  s2_batch_size: int = 32, s3_batch_size: int = 32,
                  s3_epochs: int = 30, s3_patience: int = 10) -> dict:
    """CG-Mamba Stage 2 (new dropout) → Stage 3 (new dropout)."""
    cfg = _build_cg_mamba_cfg(dropout, seed)
    hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))

    s2_name = f"wis_phase_c_cg_mamba_d{dropout}_s{seed}_stage2"
    s3_name = f"wis_phase_c_cg_mamba_d{dropout}_s{seed}_stage3"

    # ─── Stage 2 ───
    _log(f"  [CG-Mamba d={dropout} s={seed}] Stage 2 start", log_fh)
    t_s2_0 = time.time()
    s2_args = SimpleNamespace(
        smoke=False, epochs=None, batch_size=s2_batch_size,
        hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(ENV_CKPT),
        wandb_mode="disabled",
        run_name=s2_name,
    )
    s2_final = stage2_train(cfg, s2_args)
    s2_elapsed = time.time() - t_s2_0
    s2_dir = _ROOT / "runs" / "m1_7_train" / s2_name
    s2_best = s2_dir / "best.pt"
    if not s2_best.exists():
        raise RuntimeError(f"Stage 2 best.pt not produced at {s2_best}")
    _log(f"  [CG-Mamba d={dropout} s={seed}] Stage 2 done elapsed={s2_elapsed:.1f}s  "
         f"val_total={s2_final.get('best_val_total', float('nan')):.4f}", log_fh)

    # ─── Stage 3 ───
    _log(f"  [CG-Mamba d={dropout} s={seed}] Stage 3 start", log_fh)
    t_s3_0 = time.time()
    s3_args = SimpleNamespace(
        smoke=False, epochs=s3_epochs, patience=s3_patience,
        batch_size=s3_batch_size,
        stage2_dir=str(s2_dir),
        hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(ENV_CKPT),
        run_name=s3_name,
    )
    s3_final = stage3_train(cfg, s3_args)
    s3_elapsed = time.time() - t_s3_0
    s3_dir = _ROOT / "runs" / "m1_8_stage3_train" / s3_name
    s3_best = s3_dir / "best.pt"
    if not s3_best.exists():
        raise RuntimeError(f"Stage 3 best.pt not produced at {s3_best}")
    _log(f"  [CG-Mamba d={dropout} s={seed}] Stage 3 done elapsed={s3_elapsed:.1f}s  "
         f"val_total={s3_final.get('best_val_total', float('nan')):.4f}  "
         f"test_mse={s3_final.get('test_mse', float('nan')):.4f}", log_fh)

    # Also drop a thin pointer file under wis_phase_c/cg_mamba/d{d}/seed{s}/
    pointer_dir = OUT_ROOT / "cg_mamba" / f"d{dropout}" / f"seed{seed}"
    pointer_dir.mkdir(parents=True, exist_ok=True)
    (pointer_dir / "manifest_entry.json").write_text(json.dumps({
        "stage2_dir": str(s2_dir.relative_to(_ROOT)),
        "stage3_dir": str(s3_dir.relative_to(_ROOT)),
        "stage2_best": str(s2_best.relative_to(_ROOT)),
        "stage3_best": str(s3_best.relative_to(_ROOT)),
        "dropout": dropout, "seed": seed,
    }, indent=2))

    return {
        "model": "cg_mamba", "dropout": dropout, "seed": seed,
        "status": "ok",
        "stage2_dir": str(s2_dir.relative_to(_ROOT)),
        "stage3_dir": str(s3_dir.relative_to(_ROOT)),
        "stage2_best": str(s2_best.relative_to(_ROOT)),
        "stage3_best": str(s3_best.relative_to(_ROOT)),
        "stage2_val_total": s2_final.get("best_val_total", float("nan")),
        "stage3_val_total": s3_final.get("best_val_total", float("nan")),
        "stage3_test_mse": s3_final.get("test_mse", float("nan")),
        "stage2_elapsed_sec": s2_elapsed,
        "stage3_elapsed_sec": s3_elapsed,
        "elapsed_sec": s2_elapsed + s3_elapsed,
    }


# ─── Main loop ─────────────────────────────────────────────────────────────


MODEL_RUNNERS = {
    "lstm": _run_lstm,
    "vanilla_mamba": _run_vanilla_mamba,
    "cg_mamba": _run_cg_mamba,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1", help="GPU device (default cuda:1)")
    ap.add_argument(
        "--models", nargs="+",
        default=["lstm", "vanilla_mamba", "cg_mamba"],
        choices=["lstm", "vanilla_mamba", "cg_mamba"],
        help="Subset of models to run (default: all 3)",
    )
    ap.add_argument(
        "--dropouts", type=float, nargs="+", default=list(DROPOUTS),
        help="Dropout values to grid (default 0.1 0.2 0.3)",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS),
        help="Seeds (default 42 123 456 789 1024)",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan only, do not train")
    ap.add_argument("--resume", action="store_true",
                    help="Skip runs already marked 'ok' in manifest.json")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    log_fh = open(LOG_PATH, "a")
    _log(f"=== Phase C dropout grid launch  device={args.device}  "
         f"models={args.models}  dropouts={args.dropouts}  seeds={args.seeds} ===", log_fh)

    # Build plan
    plan = []
    for model in args.models:
        for d in args.dropouts:
            for s in args.seeds:
                if args.resume and _has_completed(manifest, model, d, s):
                    _log(f"  SKIP {model} d={d} s={s}  (resume — already ok)", log_fh)
                    continue
                plan.append((model, d, s))
    _log(f"Plan: {len(plan)} runs to execute", log_fh)
    for model, d, s in plan:
        _log(f"  • {model:14s} dropout={d}  seed={s}", log_fh)
    if args.dry_run:
        _log("--dry-run set, exiting without training.", log_fh)
        log_fh.close()
        return 0

    total_t0 = time.time()
    ok_count, fail_count = 0, 0
    for i, (model, d, s) in enumerate(plan, 1):
        _log(f"\n[{i}/{len(plan)}] {model} dropout={d} seed={s} ─────────────────", log_fh)
        runner = MODEL_RUNNERS[model]
        try:
            entry = runner(d, s, args.device, log_fh)
            entry["timestamp"] = datetime.now().isoformat()
            _record_run(manifest, entry)
            ok_count += 1
            _log(f"  ✓ {model} d={d} s={s}  elapsed={entry['elapsed_sec']:.1f}s", log_fh)
        except Exception as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            entry = {
                "model": model, "dropout": d, "seed": s,
                "status": "fail",
                "error": err,
                "timestamp": datetime.now().isoformat(),
            }
            _record_run(manifest, entry)
            fail_count += 1
            _log(f"  ✗ FAIL {model} d={d} s={s}\n{err}", log_fh)

    total_elapsed = time.time() - total_t0
    _log(f"\n=== Phase C done  ok={ok_count}  fail={fail_count}  "
         f"total elapsed={total_elapsed:.1f}s ({total_elapsed/60:.1f} min) ===", log_fh)
    log_fh.close()
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
