"""M2.4 Data efficiency — NN baselines (DLinear, LSTM, Vanilla Mamba) × 7 variants × 5 seeds.

Approach:
  1. Pre-filter CSV per N_seasons (mark out-of-window train rows as "_excluded")
  2. Call existing train_one_run() with filtered CSV + custom out_dir
  3. Output: runs/m2_4_data_efficiency/{baseline}/seasons_{N}/seed{s}/

Train period variants (PLAN §16 I, Option B):
  {17, 13, 10, 7, 5, 4, 3} seasons; train end fixed at W39-2018.

Per baseline × 7 × 5 = 35 runs each. Total ~50 min on GPU 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.run_lstm_weekly import train_one_run as lstm_train_one_run
from scripts.run_vanilla_mamba_weekly import train_one_run as vanilla_train_one_run
from scripts.run_patchtst_weekly import train_one_run as patchtst_train_one_run
from scripts.run_epideep_weekly import train_one_run as epideep_train_one_run

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_ROOT = _ROOT / "runs" / "m2_4_data_efficiency"
FILTERED_CSV_DIR = OUT_ROOT / "_filtered_csvs"

TRAIN_PERIODS = [
    ("17_seasons_full", 200240, 17),
    ("13_seasons",      200540, 13),
    ("10_seasons",      200840, 10),
    ( "7_seasons",      201140,  7),
    ( "5_seasons",      201340,  5),
    ( "4_seasons",      201440,  4),
    ( "3_seasons",      201540,  3),
]
SEEDS = (42, 123, 456, 789, 1024)

# Baseline winners (from M2.3 Pattern A HPO)
LSTM_HP = {
    "lookback": 104, "pred_len": 4, "enc_in": 6,
    "hidden": 256, "num_layers": 2, "lr": 5e-4, "batch_size": 16,
    "epochs": 100, "patience": 20, "dropout": 0.0,
}
VANILLA_HP = {
    "seq_len": 104, "pred_len": 4, "enc_in": 6,
    "d_model": 64, "n_layers": 3, "d_state": 16, "dt_rank": 16, "expand": 2,
    "lr": 5e-4, "batch_size": 32, "epochs": 200, "patience": 20, "dropout": 0.0,
}
# PatchTST winner config from runs/patchtst_final/pl16_dm64_lr5e-04/seed42/results.json
# (M2.3 main table baseline — re-used for data efficiency to avoid scope creep)
PATCHTST_HP = {
    "seq_len": 104, "pred_len": 4, "enc_in": 6,
    "d_model": 64, "n_heads": 4, "e_layers": 2, "d_ff_ratio": 2,
    "patch_len": 16, "stride_ratio": 0.5, "dropout": 0.1,
    "lr": 5e-4, "batch_size": 16, "epochs": 100, "patience": 20,
}
# EpiDeep winner config from runs/epideep_final/de128_eh64_lr2e-03/seed42/results.json
# (Adhikari et al. KDD 2019; epidemic-specific DL representative)
EPIDEEP_HP = {
    "seq_len": 104, "pred_len": 4, "enc_in": 6,
    "d_emb": 128, "encoder_hidden": 64, "decoder_hidden": 128,
    "alignment_weight": 0.0, "target_only": False, "dropout": 0.1,
    "lr": 2e-3, "batch_size": 16, "epochs": 100, "patience": 20,
}


def generate_filtered_csvs():
    """Pre-generate filtered CSVs per train period — STRICT interpretation.

    v2: drops earlier rows ENTIRELY (not just marks) so lookback windows
    cannot see pre-train_min data. Consistent with SARIMA's strict protocol.
    """
    FILTERED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    for label, train_min, n in TRAIN_PERIODS:
        out_path = FILTERED_CSV_DIR / f"ili_env_weekly_split_{label}.csv"
        if out_path.exists():
            continue
        # STRICT: drop train rows with epiweek < train_min (NOT mark as excluded)
        # val and test rows untouched.
        # Effect: lookback windows in train cannot access pre-train_min data.
        mask_drop = (df["split"] == "train") & (df["epiweek"] < train_min)
        df_filtered = df[~mask_drop].copy().sort_values("epiweek").reset_index(drop=True)
        n_dropped = mask_drop.sum()
        n_remaining = ((df_filtered["split"] == "train")).sum()
        df_filtered.to_csv(out_path, index=False)
        print(f"  [{label}] train >= {train_min}: {n_remaining} obs (DROPPED {n_dropped} pre-train rows — STRICT)")
    return {label: FILTERED_CSV_DIR / f"ili_env_weekly_split_{label}.csv"
            for label, _, _ in TRAIN_PERIODS}


def run_lstm_m24(filtered_csv, label, seed, device):
    cfg = LSTM_HP.copy()
    out_dir = OUT_ROOT / "lstm" / f"seasons_{label}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = lstm_train_one_run(
        cfg=cfg, seed=seed, csv_path=filtered_csv, norm_path=NORM_PATH,
        device=device, out_dir=out_dir,
        wandb_enabled=False,
        wandb_run_name=f"m2_4_lstm_{label}_s{seed}",
    )
    return {"elapsed": time.time() - t0, **r}


def run_vanilla_m24(filtered_csv, label, seed, device):
    cfg = VANILLA_HP.copy()
    out_dir = OUT_ROOT / "vanilla_mamba" / f"seasons_{label}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = vanilla_train_one_run(
        cfg=cfg, seed=seed, csv_path=filtered_csv, norm_path=NORM_PATH,
        device=device, out_dir=out_dir,
        wandb_enabled=False,
        wandb_run_name=f"m2_4_vanilla_{label}_s{seed}",
    )
    return {"elapsed": time.time() - t0, **r}


def run_patchtst_m24(filtered_csv, label, seed, device):
    """PatchTST M2.4 data efficiency — uses M2.3 winner config (pl16_dm64_lr5e-04).
    Mirrors LSTM/Vanilla pattern."""
    cfg = PATCHTST_HP.copy()
    out_dir = OUT_ROOT / "patchtst" / f"seasons_{label}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = patchtst_train_one_run(
        cfg=cfg, seed=seed, csv_path=filtered_csv, norm_path=NORM_PATH,
        device=device, out_dir=out_dir,
        wandb_enabled=False,
        wandb_run_name=f"m2_4_patchtst_{label}_s{seed}",
    )
    return {"elapsed": time.time() - t0, **r}


def run_epideep_m24(filtered_csv, label, seed, device):
    """EpiDeep M2.4 data efficiency — uses M2.3 winner config (de128_eh64_lr2e-03).
    Epidemic-specific DL representative [Adhikari et al., KDD 2019]."""
    cfg = EPIDEEP_HP.copy()
    out_dir = OUT_ROOT / "epideep" / f"seasons_{label}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = epideep_train_one_run(
        cfg=cfg, seed=seed, csv_path=filtered_csv, norm_path=NORM_PATH,
        device=device, out_dir=out_dir,
        wandb_enabled=False,
        wandb_run_name=f"m2_4_epideep_{label}_s{seed}",
    )
    return {"elapsed": time.time() - t0, **r}


def run_dlinear_m24(filtered_csv, label, seed, device):
    """DLinear: use TSLib-adapted training.

    Note: DLinear training is fast (3s/run). We invoke run_dlinear_weekly's
    train_one_run if available, else implement minimal version.
    """
    # Use baseline_test_eval approach — DLinear is simple linear model
    # For M2.4, we run a quick mini-training within this script.
    from baselines.dlinear import DLinearForecaster
    from baselines.lstm import WeeklyMultiHorizonDataset
    from torch.utils.data import DataLoader
    from src.data.loader import load_norm_params

    norm = load_norm_params(NORM_PATH)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    # Load original DLinear winner cfg
    src_cfg = json.load(open(_ROOT / "runs/dlinear_final/ma13_indF_lr2e-03/seed42/results.json"))["config"]
    cfg = src_cfg.copy()

    out_dir = OUT_ROOT / "dlinear" / f"seasons_{label}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed); np.random.seed(seed)
    df = pd.read_csv(filtered_csv)
    train_ds = WeeklyMultiHorizonDataset(df, "train", norm,
                                          lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    val_ds = WeeklyMultiHorizonDataset(df, "val", norm,
                                        lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    if len(train_ds) == 0:
        return {"elapsed": 0, "error": "empty train set"}

    model = DLinearForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        moving_avg=cfg["moving_avg"], individual=cfg["individual"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = torch.nn.MSELoss()

    t0 = time.time()
    best_val = float("inf"); best_epoch = -1; patience_ctr = 0
    epochs = cfg.get("epochs", 100); patience = cfg.get("patience", 20)
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                val_losses.append(criterion(model(x), y).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss; best_epoch = epoch; patience_ctr = 0
            torch.save(model.state_dict(), out_dir / "dlinear_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience: break

    results = {"config": cfg, "best_val_mse": best_val, "best_epoch": best_epoch,
               "elapsed_sec": time.time() - t0, "label": label, "seed": seed,
               "n_train_windows": len(train_ds), "n_val_windows": len(val_ds)}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return {"elapsed": results["elapsed_sec"], **results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baselines", nargs="+", default=["dlinear", "lstm", "vanilla_mamba"],
                    choices=["dlinear", "lstm", "vanilla_mamba", "patchtst", "epideep"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--variants", nargs="+", default=[lp[0] for lp in TRAIN_PERIODS])
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Generating filtered CSVs per train period...")
    filtered_csvs = generate_filtered_csvs()
    print(f"  CSVs ready: {len(filtered_csvs)} variants\n")

    runners = {"lstm": run_lstm_m24, "vanilla_mamba": run_vanilla_m24,
               "dlinear": run_dlinear_m24, "patchtst": run_patchtst_m24,
               "epideep": run_epideep_m24}
    summary = []
    for baseline in args.baselines:
        runner = runners[baseline]
        print(f"\n=== {baseline.upper()} × {len(args.variants)} variants × {len(args.seeds)} seeds ===")
        for label in args.variants:
            csv_p = filtered_csvs[label]
            for seed in args.seeds:
                t0 = time.time()
                try:
                    r = runner(csv_p, label, seed, args.device)
                    elapsed = r.get("elapsed", time.time() - t0)
                    val_metric = r.get("best_val_mae_h1") or r.get("best_val_mse") or "?"
                    summary.append({"baseline": baseline, "label": label, "seed": seed,
                                   "elapsed_sec": elapsed, "val_metric": val_metric, "ok": True})
                    print(f"  ✓ {baseline:14s} {label:18s} seed={seed:4d}  "
                          f"val={val_metric}  elapsed={elapsed:.1f}s")
                except Exception as e:
                    import traceback
                    summary.append({"baseline": baseline, "label": label, "seed": seed,
                                   "elapsed_sec": time.time() - t0, "error": str(e), "ok": False})
                    print(f"  ✗ {baseline} {label} seed={seed} FAIL: {type(e).__name__}: {e}")

    out_path = OUT_ROOT / "nn_baselines_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path.relative_to(_ROOT)}")
    print(f"Total runs: {sum(1 for s in summary if s['ok'])} OK / {sum(1 for s in summary if not s['ok'])} FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
