"""§18 Phase 3 Extras — DLinear + EpiDeep regional inference.

Companion to scripts/phase_3_region_eval.py. Runs the two additional DL baselines
(DLinear, EpiDeep) on the 10 HHS regions × 5 seeds without touching the existing
runs/phase_3_region_eval.csv (which holds LSTM/Vanilla Mamba/PatchTST/CG-Mamba).

Output:
  runs/phase_3_region_eval_extras.csv

Downstream:
  scripts/phase_3_region_figure.py unions phase_3_region_eval.csv +
  phase_3_region_eval_extras.csv to produce the 6-baseline Fig 3.

Design note (namespace isolation): This script is self-contained and does NOT
import from scripts/phase_3_region_eval.py. The canonical script transitively
loads baselines/patchtst.py which injects /Time-Series-Library on sys.path
and resolves `models.*` / `utils.*` / `layers.*` against the TSLIB packages.
Because CG_Mamba/src/{models,utils} share names with TSLIB packages, importing
the canonical module first leaves stale sys.modules entries that cause
ModuleNotFoundError for TSLIB internals. We inline build_region_df + _eval_one
here to avoid loading PatchTST entirely.

Usage:
  python3 scripts/phase_3_region_eval_extras.py --device cuda:1
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset
from src.baselines.dlinear import DLinearForecaster
from src.baselines.epideep import EpiDeepForecaster
from src.data.loader import load_norm_params

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240
REGIONS = [f"hhs{i}" for i in range(1, 11)]


def build_region_df(region: str) -> pd.DataFrame:
    """Build region-specific CSV with national env + split column.

    Inlined from scripts/phase_3_region_eval.py to avoid transitive PatchTST/
    TSLIB namespace conflicts during cross-script import.
    """
    region_csv = _ROOT / "data" / "raw" / "cdc_ilinet" / "_phase3_phase6_fetch" / f"{region}_full.csv"
    df_r = pd.read_csv(region_csv)
    df_r["epiweek"] = df_r["year"].astype(int) * 100 + df_r["week"].astype(int)
    from epiweeks import Week
    df_r["date"] = df_r.apply(lambda r: Week(int(r["year"]), int(r["week"])).startdate().isoformat(), axis=1)

    env = pd.read_csv(_ROOT / "data" / "processed" / "env_national_weekly.csv")
    env_cols = ["epiweek", "temperature_c", "specific_humidity_g_per_kg"]
    df_merged = df_r.merge(env[env_cols], on="epiweek", how="inner")

    split = pd.read_csv(_ROOT / "data" / "processed" / "ili_env_weekly_split.csv")
    df_merged = df_merged.merge(split[["epiweek", "split"]], on="epiweek", how="inner")

    df_merged["n_stations_available"] = 10
    df_merged["weight_sum_raw"] = 1.0
    return df_merged


def _ts_idx(eps_h1: np.ndarray) -> np.ndarray:
    return np.where(eps_h1 >= TS_BOUNDARY)[0]


def _eval_one(model, ds, device):
    """Standard NN eval: returns per-h MAE for tF and tS. Inlined from canonical script."""
    model.eval().to(device)
    n = len(ds)
    if n == 0:
        return ({f"tF_h{h}": float("nan") for h in HORIZONS}
                | {f"tS_h{h}": float("nan") for h in HORIZONS}
                | {"n_full": 0, "n_strict": 0})
    preds = np.zeros((n, 4))
    ys = np.zeros((n, 4))
    with torch.no_grad():
        for i in range(n):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            preds[i] = model(x)[0].cpu().numpy()
            ys[i] = y.numpy()
    preds_raw = preds * TARGET_STD + TARGET_MEAN
    ys_raw = ys * TARGET_STD + TARGET_MEAN
    abs_err = np.abs(preds_raw - ys_raw)
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    ts_i = _ts_idx(eps_h1)
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        out[f"tF_h{h}"] = float(abs_err[:, h_idx].mean())
        out[f"tS_h{h}"] = float(abs_err[ts_i, h_idx].mean()) if len(ts_i) > 0 else float("nan")
    out["n_full"] = int(n)
    out["n_strict"] = int(len(ts_i))
    return out


def eval_dlinear_region(region_df, seed, device):
    p = _ROOT / "runs/dlinear_final/ma13_indF_lr2e-03" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "dlinear_best.pt", map_location=device, weights_only=True)
    model = DLinearForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        moving_avg=cfg["moving_avg"], individual=cfg["individual"],
    )
    model.load_state_dict(ckpt)
    ds = WeeklyMultiHorizonDataset(region_df, "test", NORM,
                                     lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    return _eval_one(model, ds, device)


def eval_epideep_region(region_df, seed, device):
    p = _ROOT / "runs/epideep_final/de128_eh64_lr2e-03" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "epideep_best.pt", map_location=device, weights_only=True)
    model = EpiDeepForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
        decoder_hidden=cfg["decoder_hidden"],
        alignment_weight=cfg.get("alignment_weight", 0.0),
        dropout=cfg.get("dropout", 0.1),
        target_only=cfg.get("target_only", False),
    )
    model.load_state_dict(ckpt)
    ds = WeeklyMultiHorizonDataset(region_df, "test", NORM,
                                     lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    return _eval_one(model, ds, device)


def main(device="cuda:1"):
    if not torch.cuda.is_available(): device = "cpu"
    rows = []
    print(f"Device: {device}", flush=True)

    for region in REGIONS:
        print(f"\n=== {region} ===", flush=True)
        region_df = build_region_df(region)
        print(f"  Region df: {len(region_df)} rows", flush=True)

        for base_name, eval_fn in [("dlinear", eval_dlinear_region),
                                     ("epideep", eval_epideep_region)]:
            for seed in (42, 123, 456, 789, 1024):
                try:
                    r = eval_fn(region_df, seed, device)
                    r.update({"region": region, "baseline": base_name, "seed": seed})
                    rows.append(r)
                    print(f"  OK {base_name:<10} s={seed}  tS_h1={r.get('tS_h1', float('nan')):.4f}", flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  FAIL {base_name} s={seed}: {type(e).__name__}: {e}", flush=True)
                    rows.append({"region": region, "baseline": base_name, "seed": seed, "error": str(e)})

    df_out = pd.DataFrame(rows)
    out_csv = _ROOT / "runs" / "phase_3_region_eval_extras.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  rows={len(df_out)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    main(args.device)
