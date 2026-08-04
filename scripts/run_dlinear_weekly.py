"""DLinear weekly grid search driver for CG-Mamba (v2.1.7-A++ baseline expansion).

Mirrors scripts/run_patchtst_weekly.py (Pattern A — identical protocol for fair
comparison):
  --mode grid    : 12 configs × 1 seed, val_MAE @ h=1 selection
  --mode final   : 5-seed × 4-horizon for given config (--config-json)
  --mode smoke   : 1 config × 1 seed × 5 epochs (sanity)

Spec (PLAN §7.1 DLinear baseline, identical Pattern A as PatchTST/iTransformer):
  GRID  = moving_avg × individual × lr = 2 × 2 × 3 = 12 configs
  FIXED = seq_len=104, pred_len=4, enc_in=6, batch_size=16,
          epochs=100, patience=20

Output:
  runs/dlinear_grid/ma{moving_avg}_ind{individual}_lr{lr}/
    ├── dlinear_best.pt
    └── results.json
  runs/dlinear_grid/grid_summary.csv
  runs/dlinear_final/{config_dirname}/seed{S}/...

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_dlinear_weekly.py --mode smoke
  CUDA_VISIBLE_DEVICES=0 python scripts/run_dlinear_weekly.py --mode grid
  CUDA_VISIBLE_DEVICES=0 python scripts/run_dlinear_weekly.py --mode final \
      --config-json runs/dlinear_grid/<best>/results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[1]
sys.path.insert(0, str(_CG_MAMBA_ROOT / "src"))

from baselines.dlinear import DLinearForecaster      # type: ignore  # noqa: E402
from baselines.lstm import build_lstm_loaders         # type: ignore  # noqa: E402


GRID = {
    "moving_avg": [13, 25],            # series_decomp kernel size (Autoformer convention)
    "individual": [False, True],       # channel-independent linears (Zeng 2023)
    "lr": [2e-3, 1e-3, 5e-4],
}
FIXED = {
    "seq_len": 104,
    "pred_len": 4,
    "enc_in": 6,
    "batch_size": 16,
    "epochs": 100,
    "patience": 20,
}
GRID_SEED = 1
FINAL_SEEDS = [42, 123, 456, 789, 1024]
TIE_BREAK_PCT = 0.01

WANDB_ENTITY = "hjs40111-personal"
WANDB_PROJECT = "cg-mamba-jbhi"
WANDB_BASE_TAGS = ["dlinear", "weekly", "baseline", "v2.1.7-A++"]


def init_wandb(enabled, mode, cfg, seed, run_name, extra_tags=None):
    if not enabled or not _WANDB_AVAILABLE:
        return None
    tags = list(WANDB_BASE_TAGS) + [mode]
    if extra_tags:
        tags.extend(extra_tags)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=f"dlinear_{mode}_v2.1.7-A++",
        name=run_name,
        tags=tags,
        config={**cfg, "seed": seed, "phase": "M2.6_DLinear"},
        reinit=True,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    return run


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grid_iter():
    keys = list(GRID.keys())
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        cfg.update(FIXED)
        yield cfg


def config_dirname(cfg: dict) -> str:
    ind = "T" if cfg["individual"] else "F"
    return f"ma{cfg['moving_avg']}_ind{ind}_lr{cfg['lr']:.0e}"


def build_model(cfg: dict) -> DLinearForecaster:
    return DLinearForecaster(
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        enc_in=cfg["enc_in"],
        moving_avg=cfg["moving_avg"],
        individual=cfg["individual"],
    )


def train_one_run(
    cfg: dict, seed: int, csv_path: Path, norm_path: Path,
    device: str, out_dir: Path,
    epochs_override: int | None = None,
    wandb_enabled: bool = True, wandb_mode: str = "grid",
    wandb_run_name: str | None = None, wandb_extra_tags: list[str] | None = None,
) -> dict:
    set_seeds(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = epochs_override if epochs_override else cfg["epochs"]

    train_loader, val_loader, meta = build_lstm_loaders(
        csv_path=csv_path, norm_path=norm_path,
        lookback=cfg["seq_len"], pred_len=cfg["pred_len"], batch_size=cfg["batch_size"],
    )

    model = build_model(cfg).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()

    if wandb_run_name is None:
        wandb_run_name = f"{config_dirname(cfg)}_seed{seed}"
    wandb_run = init_wandb(
        enabled=wandb_enabled, mode=wandb_mode, cfg=cfg,
        seed=seed, run_name=wandb_run_name, extra_tags=wandb_extra_tags,
    )
    if wandb_run is not None:
        wandb_run.summary["param_count"] = param_count
        wandb_run.summary["n_train_windows"] = meta["n_train_windows"]
        wandb_run.summary["n_val_windows"] = meta["n_val_windows"]

    best_val_mae_h1 = float("inf")
    best_per_horizon = None
    best_epoch = 0
    patience_counter = 0
    t0 = time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                all_pred.append(pred.cpu()); all_true.append(y.cpu())
        if not all_pred:
            raise RuntimeError("Empty val_loader")
        preds_z = torch.cat(all_pred); trues_z = torch.cat(all_true)
        ts, tm = meta["target_std"], meta["target_mean"]
        preds_raw = preds_z * ts + tm; trues_raw = trues_z * ts + tm
        per_h_mae = (preds_raw - trues_raw).abs().mean(dim=0).numpy()
        val_mae_h1 = float(per_h_mae[0])
        train_loss = float(np.mean(tr_losses))

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch, "train_loss": train_loss,
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
            torch.save(model.state_dict(), out_dir / "dlinear_best.pt")
        else:
            patience_counter += 1
        if patience_counter >= cfg["patience"]:
            break

    elapsed = time() - t0
    results = {
        "config": {k: v for k, v in cfg.items()},
        "seed": seed, "param_count": param_count,
        "best_val_mae_h1": best_val_mae_h1,
        "best_val_mae_per_horizon": best_per_horizon,
        "best_epoch": best_epoch, "epochs_trained": epoch,
        "elapsed_sec": elapsed,
        "n_train_windows": meta["n_train_windows"], "n_val_windows": meta["n_val_windows"],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if wandb_run is not None:
        wandb_run.summary["best_val_mae_h1"] = best_val_mae_h1
        wandb_run.summary["best_val_mae_avg"] = float(np.mean(best_per_horizon))
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["elapsed_sec"] = elapsed
        wandb_run.finish()
    return results


def run_smoke(csv_path, norm_path, device, out_root, wandb_enabled=True):
    cfg = {"moving_avg": 25, "individual": False, "lr": 1e-3, **FIXED}
    smoke_dir = out_root / "dlinear_smoke"
    r = train_one_run(cfg, GRID_SEED, csv_path, norm_path, device, smoke_dir,
                      epochs_override=5, wandb_enabled=wandb_enabled, wandb_mode="smoke")
    print(f"[SMOKE] val_mae_h1={r['best_val_mae_h1']:.4f}, params={r['param_count']:,}, "
          f"elapsed={r['elapsed_sec']:.1f}s")


def run_grid(csv_path, norm_path, device, out_root, wandb_enabled=True):
    grid_root = out_root / "dlinear_grid"
    grid_root.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = list(grid_iter())
    for i, cfg in enumerate(cells, 1):
        cname = config_dirname(cfg)
        cdir = grid_root / cname
        if (cdir / "results.json").exists():
            r = json.loads((cdir / "results.json").read_text())
            print(f"[GRID {i}/{len(cells)}] SKIP {cname}")
        else:
            print(f"[GRID {i}/{len(cells)}] RUN  {cname}")
            r = train_one_run(cfg, GRID_SEED, csv_path, norm_path, device, cdir,
                              wandb_enabled=wandb_enabled, wandb_mode="grid")
            print(f"  -> val_mae_h1={r['best_val_mae_h1']:.4f} params={r['param_count']:,} ep={r['best_epoch']}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append({"config_dir": cname,
                     "moving_avg": cfg["moving_avg"], "individual": cfg["individual"], "lr": cfg["lr"],
                     "val_mae_h1": r["best_val_mae_h1"],
                     "val_mae_h2": r["best_val_mae_per_horizon"][1],
                     "val_mae_h3": r["best_val_mae_per_horizon"][2],
                     "val_mae_h4": r["best_val_mae_per_horizon"][3],
                     "params": r["param_count"], "best_epoch": r["best_epoch"],
                     "elapsed_sec": r["elapsed_sec"]})
    df = pd.DataFrame(rows).sort_values("val_mae_h1")
    df.to_csv(grid_root / "grid_summary.csv", index=False)
    best = df.iloc[0]
    tie_max = best["val_mae_h1"] * (1 + TIE_BREAK_PCT)
    print(f"\n[GRID DONE] best val_mae_h1={best['val_mae_h1']:.4f} -> {best['config_dir']}")
    print(f"  Top-1% (<= {tie_max:.4f}): {len(df[df['val_mae_h1']<=tie_max])} configs")


def run_final(csv_path, norm_path, device, out_root, config_json, wandb_enabled=True):
    with open(config_json) as f:
        cfg = json.load(f)["config"]
    final_root = out_root / "dlinear_final" / config_dirname(cfg)
    final_root.mkdir(parents=True, exist_ok=True)
    all_seeds = []
    for s in FINAL_SEEDS:
        sd = final_root / f"seed{s}"
        print(f"[FINAL] seed={s}")
        r = train_one_run(cfg, s, csv_path, norm_path, device, sd,
                          wandb_enabled=wandb_enabled, wandb_mode="final",
                          wandb_extra_tags=["top1pct"])
        all_seeds.append(r)
        print(f"  -> h1={r['best_val_mae_h1']:.4f}")
    per_h = np.array([r["best_val_mae_per_horizon"] for r in all_seeds])
    summary = {"config": cfg, "n_seeds": len(FINAL_SEEDS), "seeds": FINAL_SEEDS,
               "mae_mean_per_horizon": per_h.mean(axis=0).tolist(),
               "mae_std_per_horizon": per_h.std(axis=0).tolist(),
               "mae_per_seed_per_horizon": per_h.tolist()}
    with open(final_root / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[FINAL DONE] mean MAE per horizon: {summary['mae_mean_per_horizon']}")


def main():
    ap = argparse.ArgumentParser(description="DLinear weekly baseline (v2.1.7-A++)")
    ap.add_argument("--mode", choices=["smoke", "grid", "final", "auto"], default="grid")
    ap.add_argument("--csv", default="data/processed/ili_env_weekly_split.csv")
    ap.add_argument("--norm", default="data/processed/normalization_params.json")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config-json", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    csv_path = _CG_MAMBA_ROOT / args.csv
    norm_path = _CG_MAMBA_ROOT / args.norm
    out_root = _CG_MAMBA_ROOT / args.out
    wandb_enabled = (not args.no_wandb) and _WANDB_AVAILABLE
    print(f"[DLinear v2.1.7-A++] mode={args.mode}, device={args.device}, wandb={'ON' if wandb_enabled else 'OFF'}")

    if args.mode == "smoke":
        run_smoke(csv_path, norm_path, args.device, out_root, wandb_enabled=wandb_enabled)
    elif args.mode == "grid":
        run_grid(csv_path, norm_path, args.device, out_root, wandb_enabled=wandb_enabled)
    elif args.mode == "final":
        if not args.config_json:
            raise SystemExit("--config-json required for final mode")
        run_final(csv_path, norm_path, args.device, out_root, Path(args.config_json),
                  wandb_enabled=wandb_enabled)
    elif args.mode == "auto":
        # grid then final on top-1
        run_grid(csv_path, norm_path, args.device, out_root, wandb_enabled=wandb_enabled)
        grid_root = out_root / "dlinear_grid"
        df = pd.read_csv(grid_root / "grid_summary.csv").sort_values("val_mae_h1").reset_index(drop=True)
        best_cfg_dir = df.iloc[0]["config_dir"]
        best_results = grid_root / best_cfg_dir / "results.json"
        run_final(csv_path, norm_path, args.device, out_root, best_results,
                  wandb_enabled=wandb_enabled)


if __name__ == "__main__":
    main()
