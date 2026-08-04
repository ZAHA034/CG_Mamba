"""LSTM weekly grid search driver for CG-Mamba M2.2 (v2.0.8b EB-7).

Modes:
  --mode grid    : 72 configs × 1 seed, val_MAE @ h=1 선정. (default)
  --mode final   : 5-seed × 4-horizon for given config (--config-json)
  --mode smoke   : 1 config × 1 seed × 5 epochs (sanity)

Output:
  runs/lstm_grid/h{H}_l{L}_lr{LR}_bs{BS}/
    ├── lstm_best.pt
    └── results.json
  runs/lstm_grid/grid_summary.csv (집계, 72 rows)
  runs/lstm_final/seed{S}/...
  runs/lstm_final/final_summary.json

W&B (v2.0.8c — CG-Mamba section CM-Mamba와 명확히 분리):
  entity  = hjs40111-personal (CM-Mamba와 공유, 같은 workspace)
  project = cg-mamba-jbhi      (CM-Mamba `cm-mamba-jbhi-ablations`와 분리)
  group   = lstm_{mode}_v2.0.8b  (smoke/grid/final per group)
  tags    = ["lstm", "weekly", "baseline", "v2.0.8b", mode]
  name    = {config_dirname}_seed{S}

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_lstm_weekly.py --mode smoke
  CUDA_VISIBLE_DEVICES=0 python scripts/run_lstm_weekly.py --mode grid
  CUDA_VISIBLE_DEVICES=0 python scripts/run_lstm_weekly.py --mode grid --no-wandb
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# W&B (optional, --no-wandb로 disable 가능)
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Project root → sys.path
_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[1]
sys.path.insert(0, str(_CG_MAMBA_ROOT / "src"))

from baselines.lstm import LSTMForecaster, build_lstm_loaders  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Spec (PLAN v2.0.8b §5.3 EB-6, §7.1 EB-7)
# ---------------------------------------------------------------------------
GRID = {
    "hidden": [64, 128, 192, 256],
    "num_layers": [1, 2, 3],
    "lr": [5e-4, 1e-3, 2e-3],
    "batch_size": [16, 32],
}
FIXED = {
    "lookback": 104,
    "pred_len": 4,
    "dropout": 0.0,
    "epochs": 100,
    "patience": 20,
    "enc_in": 6,
}
GRID_SEED = 1  # single seed for grid phase
FINAL_SEEDS = [42, 123, 456, 789, 1024]  # 5-seed for top-1% configs
TIE_BREAK_PCT = 0.01  # top-1% within best val_MAE

# W&B settings (v2.0.8c — CG-Mamba section 명확 분리)
WANDB_ENTITY = "hjs40111-personal"
WANDB_PROJECT = "cg-mamba-jbhi"
WANDB_BASE_TAGS = ["lstm", "weekly", "baseline", "v2.0.8b"]


def init_wandb(
    enabled: bool,
    mode: str,
    cfg: dict,
    seed: int,
    run_name: str,
    extra_tags: list[str] = None,
) -> "wandb.sdk.wandb_run.Run | None":
    """Initialize W&B run if enabled. Group/tags differentiate CG-Mamba from CM-Mamba.

    Args:
        enabled:    --no-wandb로 disable 가능
        mode:       'smoke' | 'grid' | 'final'
        cfg:        config dict (saved as wandb.config)
        seed:       seed value (saved as wandb.config.seed)
        run_name:   unique run identifier
        extra_tags: additional tags (e.g., ['top1pct'])

    Returns:
        wandb run object or None if disabled.
    """
    if not enabled or not _WANDB_AVAILABLE:
        return None

    tags = WANDB_BASE_TAGS + [mode]
    if extra_tags:
        tags.extend(extra_tags)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=f"lstm_{mode}_v2.0.8b",
        name=run_name,
        tags=tags,
        config={**cfg, "seed": seed, "phase_1_section": "M2.2_LSTM"},
        reinit=True,  # 같은 process에서 여러 run 실행 가능 (grid loop)
    )
    # Patch #23 (CM-Mamba 패턴): epoch을 x-axis로 명시
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    return run


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grid_iter():
    """Yield all 72 config dicts."""
    keys = list(GRID.keys())
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        cfg.update(FIXED)
        yield cfg


def config_dirname(cfg: dict) -> str:
    return (
        f"h{cfg['hidden']}_l{cfg['num_layers']}"
        f"_lr{cfg['lr']:.0e}_bs{cfg['batch_size']}"
    )


def train_one_run(
    cfg: dict,
    seed: int,
    csv_path: Path,
    norm_path: Path,
    device: str,
    out_dir: Path,
    epochs_override: int = None,
    wandb_enabled: bool = True,
    wandb_mode: str = "grid",
    wandb_run_name: str = None,
    wandb_extra_tags: list[str] = None,
) -> dict:
    """단일 LSTM 학습. Returns results dict.

    W&B (v2.0.8c): per-epoch metrics logged to W&B (entity=hjs40111-personal,
    project=cg-mamba-jbhi, group=lstm_{mode}_v2.0.8b).
    """
    set_seeds(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = epochs_override if epochs_override else cfg["epochs"]

    train_loader, val_loader, meta = build_lstm_loaders(
        csv_path=csv_path,
        norm_path=norm_path,
        lookback=cfg["lookback"],
        pred_len=cfg["pred_len"],
        batch_size=cfg["batch_size"],
    )

    model = LSTMForecaster(
        enc_in=cfg["enc_in"],
        hidden=cfg["hidden"],
        num_layers=cfg["num_layers"],
        pred_len=cfg["pred_len"],
        dropout=cfg["dropout"],
    ).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()

    # W&B init (per-run)
    if wandb_run_name is None:
        wandb_run_name = f"{config_dirname(cfg)}_seed{seed}"
    wandb_run = init_wandb(
        enabled=wandb_enabled,
        mode=wandb_mode,
        cfg=cfg,
        seed=seed,
        run_name=wandb_run_name,
        extra_tags=wandb_extra_tags,
    )
    if wandb_run is not None:
        wandb_run.summary["param_count"] = param_count
        wandb_run.summary["n_train_windows"] = meta["n_train_windows"]
        wandb_run.summary["n_val_windows"] = meta["n_val_windows"]
        wandb_run.summary["target_mean"] = meta["target_mean"]
        wandb_run.summary["target_std"] = meta["target_std"]

    best_val_mae_h1 = float("inf")
    best_per_horizon = None
    best_epoch = 0
    patience_counter = 0
    history = []
    t0 = time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)  # [B, H=4]
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        # Validation (per-horizon MAE in z-scored space, then denorm to raw)
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                all_pred.append(pred.cpu())
                all_true.append(y.cpu())
        if not all_pred:
            raise RuntimeError("Empty val_loader — check lookback vs val window count")
        preds_z = torch.cat(all_pred)  # [N, 4]
        trues_z = torch.cat(all_true)  # [N, 4]

        # Denorm to raw scale
        ts, tm = meta["target_std"], meta["target_mean"]
        preds_raw = preds_z * ts + tm
        trues_raw = trues_z * ts + tm

        per_h_mae = (preds_raw - trues_raw).abs().mean(dim=0).numpy()  # [4]
        val_mae_h1 = float(per_h_mae[0])

        train_loss = float(np.mean(tr_losses))
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae_h1": val_mae_h1,
            "val_mae_per_horizon": per_h_mae.tolist(),
        }
        history.append(epoch_record)

        # W&B per-epoch log (v2.0.8c)
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae_h1": val_mae_h1,
                "val_mae_h2": float(per_h_mae[1]),
                "val_mae_h3": float(per_h_mae[2]),
                "val_mae_h4": float(per_h_mae[3]),
                "val_mae_avg": float(per_h_mae.mean()),
            })

        if val_mae_h1 < best_val_mae_h1:
            best_val_mae_h1 = val_mae_h1
            best_per_horizon = per_h_mae.tolist()
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "lstm_best.pt")
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            break

    elapsed = time() - t0
    results = {
        "config": {k: v for k, v in cfg.items()},
        "seed": seed,
        "param_count": param_count,
        "best_val_mae_h1": best_val_mae_h1,
        "best_val_mae_per_horizon": best_per_horizon,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "elapsed_sec": elapsed,
        "n_train_windows": meta["n_train_windows"],
        "n_val_windows": meta["n_val_windows"],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # W&B summary + finalize
    if wandb_run is not None:
        wandb_run.summary["best_val_mae_h1"] = best_val_mae_h1
        wandb_run.summary["best_val_mae_h2"] = best_per_horizon[1]
        wandb_run.summary["best_val_mae_h3"] = best_per_horizon[2]
        wandb_run.summary["best_val_mae_h4"] = best_per_horizon[3]
        wandb_run.summary["best_val_mae_avg"] = float(np.mean(best_per_horizon))
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["epochs_trained"] = epoch
        wandb_run.summary["elapsed_sec"] = elapsed
        # Save lstm_best.pt as W&B artifact (optional but useful)
        try:
            artifact = wandb.Artifact(f"lstm_best_{wandb_run_name}", type="model")
            artifact.add_file(str(out_dir / "lstm_best.pt"))
            wandb_run.log_artifact(artifact)
        except Exception as e:
            print(f"  [W&B] artifact upload skipped: {e}")
        wandb_run.finish()

    return results


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_smoke(
    csv_path: Path, norm_path: Path, device: str, out_root: Path,
    wandb_enabled: bool = True,
) -> None:
    """1 config × 1 seed × 5 epochs sanity test."""
    cfg = {"hidden": 128, "num_layers": 2, "lr": 1e-3, "batch_size": 16, **FIXED}
    smoke_dir = out_root / "lstm_smoke"
    print(f"[SMOKE] config={config_dirname(cfg)} on {device}")
    r = train_one_run(
        cfg, GRID_SEED, csv_path, norm_path, device, smoke_dir,
        epochs_override=5,
        wandb_enabled=wandb_enabled, wandb_mode="smoke",
    )
    print(f"[SMOKE] DONE: val_mae_h1={r['best_val_mae_h1']:.4f}, "
          f"params={r['param_count']:,}, elapsed={r['elapsed_sec']:.1f}s")


def run_grid(
    csv_path: Path, norm_path: Path, device: str, out_root: Path,
    wandb_enabled: bool = True,
) -> None:
    """72 configs × 1 seed."""
    grid_root = out_root / "lstm_grid"
    grid_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_configs = list(grid_iter())
    print(f"[GRID] total {len(all_configs)} configs on {device}, seed={GRID_SEED}")

    for i, cfg in enumerate(all_configs, 1):
        cname = config_dirname(cfg)
        cdir = grid_root / cname
        if (cdir / "results.json").exists():
            print(f"[GRID] {i}/{len(all_configs)} {cname} — SKIP (already done)")
            with open(cdir / "results.json") as f:
                r = json.load(f)
        else:
            print(f"[GRID] {i}/{len(all_configs)} {cname}")
            r = train_one_run(
                cfg, GRID_SEED, csv_path, norm_path, device, cdir,
                wandb_enabled=wandb_enabled, wandb_mode="grid",
            )
            print(f"  → val_mae_h1={r['best_val_mae_h1']:.4f} "
                  f"params={r['param_count']:,} ep={r['best_epoch']}")
            # GPU 메모리 정리 (config 간 leak 방지, v2.0.8b defensive)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary_rows.append({
            "config_dir": cname,
            "hidden": cfg["hidden"],
            "num_layers": cfg["num_layers"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "val_mae_h1": r["best_val_mae_h1"],
            "val_mae_h2": r["best_val_mae_per_horizon"][1],
            "val_mae_h3": r["best_val_mae_per_horizon"][2],
            "val_mae_h4": r["best_val_mae_per_horizon"][3],
            "params": r["param_count"],
            "best_epoch": r["best_epoch"],
            "elapsed_sec": r["elapsed_sec"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("val_mae_h1")
    summary_df.to_csv(grid_root / "grid_summary.csv", index=False)
    best = summary_df.iloc[0]
    tie_break_max = best["val_mae_h1"] * (1 + TIE_BREAK_PCT)
    tie_break = summary_df[summary_df["val_mae_h1"] <= tie_break_max]
    print(f"\n[GRID DONE] best val_mae_h1={best['val_mae_h1']:.4f} → {best['config_dir']}")
    print(f"  Top-1% (≤ {tie_break_max:.4f}): {len(tie_break)} configs")
    print(f"  Saved: {grid_root / 'grid_summary.csv'}")


def run_final(
    csv_path: Path, norm_path: Path, device: str, out_root: Path, config_json: Path,
    wandb_enabled: bool = True,
) -> None:
    """Top-1% configs → 5-seed × 4-horizon.

    config_json은 grid run의 `results.json` 경로:
      {"config": {...}, "seed": 1, "best_val_mae_h1": ..., ...}
    → cfg = results["config"] 로 추출 (Bug #1 fix, v2.0.8b).
    """
    with open(config_json) as f:
        results_blob = json.load(f)
    # results.json의 "config" key가 실제 학습 spec
    if "config" in results_blob:
        cfg = results_blob["config"]
    else:
        # backward compat: 만약 flat config dict가 직접 전달된 경우
        cfg = results_blob
    final_root = out_root / "lstm_final" / config_dirname(cfg)
    final_root.mkdir(parents=True, exist_ok=True)
    all_seeds = []
    for s in FINAL_SEEDS:
        seed_dir = final_root / f"seed{s}"
        print(f"[FINAL] seed={s} on {device}")
        r = train_one_run(
            cfg, s, csv_path, norm_path, device, seed_dir,
            wandb_enabled=wandb_enabled, wandb_mode="final",
            wandb_extra_tags=["top1pct"],
        )
        all_seeds.append(r)
        print(f"  → h1={r['best_val_mae_h1']:.4f}")

    # Aggregate
    per_h = np.array([r["best_val_mae_per_horizon"] for r in all_seeds])  # [5, 4]
    summary = {
        "config": cfg,
        "n_seeds": len(FINAL_SEEDS),
        "seeds": FINAL_SEEDS,
        "mae_mean_per_horizon": per_h.mean(axis=0).tolist(),
        "mae_std_per_horizon": per_h.std(axis=0).tolist(),
        "mae_per_seed_per_horizon": per_h.tolist(),
    }
    with open(final_root / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[FINAL DONE] mean MAE per horizon: {summary['mae_mean_per_horizon']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LSTM weekly grid search (v2.0.8b/c)")
    ap.add_argument("--mode", choices=["smoke", "grid", "final"], default="grid")
    ap.add_argument("--csv", default="data/processed/ili_env_weekly_split.csv")
    ap.add_argument("--norm", default="data/processed/normalization_params.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config-json", default=None,
                    help="(final mode) path to results.json of best grid config")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging (default: enabled, v2.0.8c)")
    args = ap.parse_args()

    csv_path = _CG_MAMBA_ROOT / args.csv
    norm_path = _CG_MAMBA_ROOT / args.norm
    out_root = _CG_MAMBA_ROOT / args.out
    wandb_enabled = (not args.no_wandb) and _WANDB_AVAILABLE

    print(f"[LSTM v2.0.8b] mode={args.mode}, device={args.device}")
    print(f"  csv={csv_path}\n  norm={norm_path}\n  out={out_root}")
    print(f"  wandb={'ENABLED' if wandb_enabled else 'DISABLED'} "
          f"(entity={WANDB_ENTITY}, project={WANDB_PROJECT})")

    if args.mode == "smoke":
        run_smoke(csv_path, norm_path, args.device, out_root, wandb_enabled=wandb_enabled)
    elif args.mode == "grid":
        run_grid(csv_path, norm_path, args.device, out_root, wandb_enabled=wandb_enabled)
    elif args.mode == "final":
        if not args.config_json:
            raise SystemExit("--config-json required for final mode")
        run_final(csv_path, norm_path, args.device, out_root, Path(args.config_json),
                  wandb_enabled=wandb_enabled)


if __name__ == "__main__":
    main()
