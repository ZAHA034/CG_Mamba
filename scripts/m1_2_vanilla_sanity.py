"""M1.2 sanity: train vanilla CG-Mamba backbone (no gate) on ILI 1-step.

Per PLAN v2.0.7 §9.1 W1 M1.2 exit criteria:
  - Vanilla Mamba MAE < 0.5 (sanity on val %wILI scale)
  - CUDA pre-computed mode decision: pure-PyTorch fallback chosen (see report)

Usage:
    python -m scripts.m1_2_vanilla_sanity
    python -m scripts.m1_2_vanilla_sanity --epochs 30 --lookback 104
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.loader import (
    WeeklyDataset, collate_dict, load_dataset_csv, load_norm_params,
)
from src.models.backbone import M1_2_VanillaCGMamba
from src.utils.config import CGMambaConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate(
    model: nn.Module, loader: DataLoader, norm_target: dict, device: str,
) -> dict:
    """Return dict with MAE/RMSE on un-standardized %wILI scale."""
    model.eval()
    mu, sigma = norm_target["mean"], norm_target["std"]
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_raw = batch["y_raw"].cpu().numpy()
            yhat_std = model(x).cpu().numpy()
            yhat_raw = yhat_std * sigma + mu
            ys.append(y_raw)
            preds.append(yhat_raw)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    mae = float(np.abs(ys - preds).mean())
    rmse = float(np.sqrt(((ys - preds) ** 2).mean()))
    return {"mae": mae, "rmse": rmse,
            "y_mean": float(ys.mean()), "yhat_mean": float(preds.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lookback", type=int, default=104)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_mae", type=float, default=0.5,
                    help="Sanity threshold for val MAE (PLAN v2.0.7 §9.1).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = CGMambaConfig(
        lookback=args.lookback, batch_size=args.batch_size, lr=args.lr,
        n_epochs=args.epochs, seed=args.seed, use_gate=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[M1.2 sanity] {cfg.summary()}")
    print(f"[M1.2 sanity] device={device}, epochs={args.epochs}, horizon={args.horizon}")
    print()

    # Data
    df = load_dataset_csv(cfg.data_csv)
    norm = load_norm_params(cfg.norm_json)
    norm_target = norm["ili_weighted_pct"]

    ds_train = WeeklyDataset(df, "train", cfg.lookback, args.horizon, norm)
    ds_val = WeeklyDataset(df, "val", cfg.lookback, args.horizon, norm)
    ds_test = WeeklyDataset(df, "test", cfg.lookback, args.horizon, norm)
    print(f"[M1.2 sanity] windows: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate_dict, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_dict, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                         collate_fn=collate_dict, num_workers=0)

    # Model
    model = M1_2_VanillaCGMamba(cfg).to(device)
    n_params = model.n_params()
    print(f"[M1.2 sanity] model params: {n_params:,}")

    # Baseline: persistence (y_{t+1} = y_t). Naive sanity floor.
    def persistence_mae(loader, mu, sigma):
        ys, preds = [], []
        for batch in loader:
            last_z = batch["x"][:, -1, 0].numpy()       # ili_weighted_pct standardized
            yhat = last_z * sigma + mu
            ys.append(batch["y_raw"].numpy())
            preds.append(yhat)
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        return float(np.abs(ys - preds).mean())

    pers_val = persistence_mae(dl_val, norm_target["mean"], norm_target["std"])
    pers_test = persistence_mae(dl_test, norm_target["mean"], norm_target["std"])
    print(f"[M1.2 sanity] persistence baseline (y_t -> y_{{t+1}}): val MAE={pers_val:.4f}, test MAE={pers_test:.4f}")
    print()

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    best_val = float("inf")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for batch in dl_train:
            x = batch["x"].to(device)
            y_z = batch["y"].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y_z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_losses.append(loss.item())

        tr_mse = float(np.mean(tr_losses))
        val = evaluate(model, dl_val, norm_target, device)
        history.append({"epoch": ep, "train_mse": tr_mse, **val})

        marker = ""
        if val["mae"] < best_val:
            best_val = val["mae"]
            marker = " *"

        if ep <= 5 or ep % 5 == 0 or ep == args.epochs:
            print(f"  [ep {ep:>3}] train_mse={tr_mse:.4f} | "
                  f"val MAE={val['mae']:.4f}, RMSE={val['rmse']:.4f}"
                  f"{marker}  (elapsed {(time.time()-t0)/60:.1f}min)")

    test = evaluate(model, dl_test, norm_target, device)
    elapsed = (time.time() - t0) / 60

    print()
    print("="*70)
    print(f"[M1.2 sanity] FINAL  ({elapsed:.1f} min, {args.epochs} epochs)")
    print(f"  model params:    {n_params:,}")
    print(f"  persistence:     val MAE={pers_val:.4f},  test MAE={pers_test:.4f}")
    print(f"  best val MAE:    {best_val:.4f}")
    print(f"  final val MAE:   {val['mae']:.4f}")
    print(f"  final test MAE:  {test['mae']:.4f}")
    print(f"  target val MAE:  < {args.target_mae:.4f}")
    print(f"  EXIT CRITERION:  {'PASS ✅' if best_val < args.target_mae else 'FAIL ❌'}")
    print("="*70)

    # Save history
    out_dir = REPO_ROOT / "runs" / "m1_2_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"history_L{args.lookback}_h{args.horizon}_ep{args.epochs}.json"
    with open(out_json, "w") as f:
        json.dump({
            "config": cfg.summary(),
            "args": vars(args),
            "n_params": n_params,
            "persistence_val_mae": pers_val,
            "persistence_test_mae": pers_test,
            "best_val_mae": best_val,
            "final_val": val,
            "final_test": test,
            "elapsed_min": elapsed,
            "history": history,
        }, f, indent=2)
    print(f"  history saved: {out_json.relative_to(REPO_ROOT)}")

    return 0 if best_val < args.target_mae else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
