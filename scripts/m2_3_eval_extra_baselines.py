"""Re-evaluate M2.3 5-seed final ckpts for 6 baselines that lacked per-horizon × split MAE:
  PatchTST, iTransformer, TimesNet, N-BEATS, EpiDeep (each: 5-seed ckpt)
  + Persistence (deterministic, no ckpt)

Use full data CSV (data/processed/ili_env_weekly_split.csv).
Per-horizon × {test_full, test_strict} × 5-seed mean ± std.
"""
from __future__ import annotations
import json, sys, warnings, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset
from baselines.patchtst import PatchTSTForecaster
from baselines.itransformer import ITransformerForecaster
from baselines.timesnet import TimesNetForecaster
from baselines.nbeats import NBeatsForecaster
from baselines.epideep import EpiDeepForecaster
from src.data.loader import load_norm_params

CSV = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])
HORIZONS = [1, 2, 3, 4]
SEEDS = [42, 123, 456, 789, 1024]
TS_BOUNDARY = 202240

# Winner cfgs (per M2.3 master CSV / §5.4 HP master table)
WINNERS = {
    "patchtst": ("runs/patchtst_final/pl16_dm128_lr5e-04", "patchtst_best.pt"),
    "itransformer": ("runs/itransformer_final/dm128_el2_lr5e-04", "itransformer_best.pt"),
    "timesnet": ("runs/timesnet_final/d64_el2_lr1e-03", "timesnet_best.pt"),
    "nbeats": ("runs/nbeats_final/nb24_h512_lr5e-04", "nbeats_best.pt"),
    "epideep": ("runs/epideep_final/de128_eh64_lr2e-03", "epideep_best.pt"),
}

CLASSES = {
    "patchtst": PatchTSTForecaster,
    "itransformer": ITransformerForecaster,
    "timesnet": TimesNetForecaster,
    "nbeats": NBeatsForecaster,
    "epideep": EpiDeepForecaster,
}


def _ts_idx(eps_h1: np.ndarray) -> np.ndarray:
    return np.where(eps_h1 >= TS_BOUNDARY)[0]


def _eval_one(model, ds, device):
    model.eval().to(device)
    preds = np.zeros((len(ds), 4))
    ys = np.zeros((len(ds), 4))
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            preds[i] = pred[0].cpu().numpy()
            ys[i] = y.numpy()
    preds_raw = preds * TARGET_STD + TARGET_MEAN
    ys_raw = ys * TARGET_STD + TARGET_MEAN
    abs_err = np.abs(preds_raw - ys_raw)
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    ts_i = _ts_idx(eps_h1)
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        out[f"tF_mae_h{h}"] = float(abs_err[:, h_idx].mean())
        out[f"tS_mae_h{h}"] = float(abs_err[ts_i, h_idx].mean()) if len(ts_i) > 0 else np.nan
    out["n_full"] = int(len(abs_err))
    out["n_strict"] = int(len(ts_i))
    return out


def evaluate_nn(base: str, seed: int, device: str) -> dict:
    cfg_dir, ckpt_name = WINNERS[base]
    p = Path(cfg_dir) / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    # Strip extra config keys not in __init__
    Cls = CLASSES[base]
    init_params = Cls.__init__.__code__.co_varnames[:Cls.__init__.__code__.co_argcount]
    kwargs = {k: v for k, v in cfg.items() if k in init_params}
    # Reconstruct d_ff from d_ff_ratio (some configs store ratio not absolute)
    if "d_ff_ratio" in cfg and "d_ff" not in cfg and "d_ff" in init_params:
        kwargs["d_ff"] = int(cfg["d_model"] * cfg["d_ff_ratio"])
    if "stride_ratio" in cfg and "stride" not in cfg and "stride" in init_params:
        kwargs["stride"] = int(cfg["patch_len"] * cfg["stride_ratio"])
    model = Cls(**kwargs)
    ckpt = torch.load(p / ckpt_name, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    df = pd.read_csv(CSV)
    ds = WeeklyMultiHorizonDataset(df, "test", NORM,
                                    lookback=cfg.get("seq_len", cfg.get("lookback", 104)),
                                    pred_len=cfg.get("pred_len", 4))
    return _eval_one(model, ds, device)


def evaluate_persistence() -> dict:
    """y_hat_{t+h} = y_t (last observed). Apply to test windows."""
    df = pd.read_csv(CSV)
    # Use same window definition (gap-aware)
    ds = WeeklyMultiHorizonDataset(df, "test", NORM, lookback=104, pred_len=4)
    preds = np.zeros((len(ds), 4))
    ys = np.zeros((len(ds), 4))
    for i in range(len(ds)):
        x, y = ds[i]  # x [104, 6], y [4] z-scored
        # Persistence: y_hat = x_last (col 0 = z-scored ili)
        last_ili = float(x[-1, 0])
        preds[i] = [last_ili] * 4
        ys[i] = y.numpy()
    preds_raw = preds * TARGET_STD + TARGET_MEAN
    ys_raw = ys * TARGET_STD + TARGET_MEAN
    abs_err = np.abs(preds_raw - ys_raw)
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    ts_i = _ts_idx(eps_h1)
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        out[f"tF_mae_h{h}"] = float(abs_err[:, h_idx].mean())
        out[f"tS_mae_h{h}"] = float(abs_err[ts_i, h_idx].mean())
    out["n_full"] = int(len(abs_err))
    out["n_strict"] = int(len(ts_i))
    return out


def main(device="cuda:1"):
    if not torch.cuda.is_available(): device = "cpu"
    rows = []
    print(f"Device: {device}", flush=True)
    # NN baselines × 5 seeds
    for base in WINNERS:
        for seed in SEEDS:
            try:
                r = evaluate_nn(base, seed, device)
                r.update({"baseline": base, "seed": seed})
                rows.append(r)
                print(f"  ✓ {base:<14} s={seed}  tS_h1={r.get('tS_mae_h1', np.nan):.4f}", flush=True)
            except Exception as e:
                import traceback
                print(f"  ✗ {base} s={seed}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                rows.append({"baseline": base, "seed": seed, "error": str(e)})
    # Persistence (deterministic, single eval)
    try:
        r = evaluate_persistence()
        r.update({"baseline": "persistence", "seed": -1})
        rows.append(r)
        print(f"  ✓ persistence    tS_h1={r['tS_mae_h1']:.4f}", flush=True)
    except Exception as e:
        print(f"  ✗ persistence: {e}", flush=True)
    df = pd.DataFrame(rows)
    out_csv = _ROOT / "runs" / "m2_3_extra_baselines_per_h_split.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  rows={len(df)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    main(args.device)
