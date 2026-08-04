"""§18 Phase 3 — HHS Region stratified evaluation.

Pipeline:
  1. Load region CSV (hhs1~10) + national env (join by epiweek)
  2. Apply national split labels (train/val/test) by epiweek
  3. Run inference for 5 baselines × 10 regions × 4 horizons
  4. Compute per-region MAE (tF + tS)
  5. Save: runs/phase_3_region_eval.csv

Models evaluated (M2.3 final ckpts):
  - SARIMA (re-fit per-region since region-specific seasonality)
  - LSTM (national-trained, regional input — Option A)
  - Vanilla Mamba (same)
  - CG-Mamba (same, with phase context inferred from regional ILI)
  - PatchTST (additional NN family)

Note: env (humidity, temperature) is national pop-weighted (no region-specific
env available). This is documented limitation.
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
from cm_mamba.baselines.lstm_baseline import LSTMForecaster
from baselines.vanilla_mamba import VanillaMambaForecaster
from baselines.patchtst import PatchTSTForecaster
from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import WeeklyDataset, load_norm_params

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240
REGIONS = [f"hhs{i}" for i in range(1, 11)]


def build_region_df(region: str) -> pd.DataFrame:
    """Build region-specific CSV with national env + split column."""
    # Region CSV
    region_csv = _ROOT / "data" / "raw" / "cdc_ilinet" / "_phase3_phase6_fetch" / f"{region}_full.csv"
    df_r = pd.read_csv(region_csv)
    df_r["epiweek"] = df_r["year"].astype(int) * 100 + df_r["week"].astype(int)
    # Compute date as MMWR Sunday from epiweek
    from epiweeks import Week
    df_r["date"] = df_r.apply(lambda r: Week(int(r["year"]), int(r["week"])).startdate().isoformat(), axis=1)

    # National env (no region-specific env)
    env = pd.read_csv(_ROOT / "data" / "processed" / "env_national_weekly.csv")
    env_cols = ["epiweek", "temperature_c", "specific_humidity_g_per_kg"]
    df_merged = df_r.merge(env[env_cols], on="epiweek", how="inner")

    # National split labels by epiweek
    split = pd.read_csv(_ROOT / "data" / "processed" / "ili_env_weekly_split.csv")
    df_merged = df_merged.merge(split[["epiweek", "split"]], on="epiweek", how="inner")

    # Add missing columns for compatibility (n_stations_available, weight_sum_raw)
    df_merged["n_stations_available"] = 10
    df_merged["weight_sum_raw"] = 1.0

    return df_merged


def _ts_idx(eps_h1: np.ndarray) -> np.ndarray:
    return np.where(eps_h1 >= TS_BOUNDARY)[0]


def _eval_one(model, ds, device):
    """Standard NN eval: returns per-h MAE for tF and tS."""
    model.eval().to(device)
    n = len(ds)
    if n == 0:
        return {f"tF_h{h}": float("nan") for h in HORIZONS} | {f"tS_h{h}": float("nan") for h in HORIZONS} | {"n_full": 0, "n_strict": 0}
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


def eval_lstm_region(region_df, seed, device):
    p = _ROOT / "runs/lstm_final/h256_l2_lr5e-04_bs16" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "lstm_best.pt", map_location=device, weights_only=True)
    model = LSTMForecaster(enc_in=cfg["enc_in"], hidden=cfg["hidden"], num_layers=cfg["num_layers"],
                            pred_len=cfg["pred_len"], dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt)
    ds = WeeklyMultiHorizonDataset(region_df, "test", NORM, lookback=cfg["lookback"], pred_len=cfg["pred_len"])
    return _eval_one(model, ds, device)


def eval_vanilla_region(region_df, seed, device):
    p = _ROOT / "runs/vanilla_mamba_final/d64_nl3_lr5e-04" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "vanilla_mamba_best.pt", map_location=device, weights_only=True)
    model = VanillaMambaForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                                     d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                                     dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt)
    ds = WeeklyMultiHorizonDataset(region_df, "test", NORM, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    return _eval_one(model, ds, device)


def eval_patchtst_region(region_df, seed, device):
    p = _ROOT / "runs/patchtst_final/pl16_dm128_lr5e-04" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "patchtst_best.pt", map_location=device, weights_only=True)
    kwargs = dict(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                  d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
                  patch_len=cfg["patch_len"], dropout=cfg.get("dropout", 0.1))
    if "d_ff_ratio" in cfg:
        kwargs["d_ff"] = int(cfg["d_model"] * cfg["d_ff_ratio"])
    if "stride_ratio" in cfg:
        kwargs["stride"] = int(cfg["patch_len"] * cfg["stride_ratio"])
    model = PatchTSTForecaster(**kwargs)
    model.load_state_dict(ckpt)
    ds = WeeklyMultiHorizonDataset(region_df, "test", NORM, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    return _eval_one(model, ds, device)


def eval_cgm_region(region_df, seed, device):
    """CG-Mamba: national-trained ckpt, regional ILI input."""
    # Use M2.4 17_seasons_full ckpt (M2.1 final equivalent)
    m_path = _ROOT / "runs/m2_4_data_efficiency/cg_mamba/seasons_17_seasons_full" / f"seed{seed}" / "manifest.json"
    m = json.load(open(m_path))
    hmm = load_fitted_hmm(Path(m["hmm_dir"]))
    cfg = CGMambaConfig()
    model = CGForecaster(cfg)
    model.prepare_for_stage2(hmm)
    ck = torch.load(Path(m["stage3_best"]), map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    # Use WeeklyDataset (CG-Mamba returns x, env separately)
    ds = WeeklyDataset(region_df, split="test", lookback=cfg.lookback, horizon=max(cfg.horizons), norm=NORM)
    n = len(ds)
    if n == 0:
        return {f"tF_h{h}": float("nan") for h in HORIZONS} | {f"tS_h{h}": float("nan") for h in HORIZONS} | {"n_full": 0, "n_strict": 0, "n_nan": 0}
    eps = region_df["epiweek"].astype(int).to_numpy()
    abs_err_h = [[],[],[],[]]
    eps_h1 = []
    n_nan = 0
    with torch.no_grad():
        for i in range(n):
            d = ds[i]
            x = d["x"].unsqueeze(0).to(device)
            env = d["env"].unsqueeze(0).to(device)
            pred = model(x, env)
            if torch.isnan(pred).any():
                n_nan += 1
                continue
            preds_raw = (pred[0].cpu().numpy() * TARGET_STD + TARGET_MEAN)
            tgt_ep = int(d["target_epiweek"])
            tgt_idx = int(np.where(eps == tgt_ep)[0][0])
            for h_idx, h in enumerate(HORIZONS):
                off = max(HORIZONS) - h
                src = tgt_idx - off
                if 0 <= src < len(eps):
                    y_raw = region_df.iloc[src]["ili_weighted_pct"]
                    abs_err_h[h_idx].append(abs(preds_raw[h_idx] - y_raw))
            eps_h1.append(eps[tgt_idx - (max(HORIZONS) - 1)])
    eps_h1_arr = np.array(eps_h1)
    ts_mask = eps_h1_arr >= TS_BOUNDARY
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        err = np.array(abs_err_h[h_idx])
        out[f"tF_h{h}"] = float(err.mean()) if len(err) > 0 else float("nan")
        out[f"tS_h{h}"] = float(err[ts_mask].mean()) if ts_mask.sum() > 0 else float("nan")
    out["n_full"] = int(len(abs_err_h[0]))
    out["n_strict"] = int(ts_mask.sum())
    out["n_nan"] = n_nan
    return out


def main(device="cuda:1"):
    if not torch.cuda.is_available(): device = "cpu"
    rows = []
    print(f"Device: {device}", flush=True)

    for region in REGIONS:
        print(f"\n=== {region} ===", flush=True)
        region_df = build_region_df(region)
        print(f"  Region df: {len(region_df)} rows, splits {region_df.split.value_counts().to_dict()}", flush=True)

        # NN baselines: 5-seed mean
        for base_name, eval_fn in [("lstm", eval_lstm_region), ("vanilla_mamba", eval_vanilla_region),
                                    ("patchtst", eval_patchtst_region), ("cg_mamba", eval_cgm_region)]:
            for seed in (42, 123, 456, 789, 1024):
                try:
                    r = eval_fn(region_df, seed, device)
                    r.update({"region": region, "baseline": base_name, "seed": seed})
                    rows.append(r)
                    print(f"  ✓ {base_name:<14} s={seed}  tS_h1={r.get('tS_h1', float('nan')):.4f}  n_strict={r.get('n_strict','?')}", flush=True)
                except Exception as e:
                    import traceback
                    print(f"  ✗ {base_name} s={seed}: {type(e).__name__}: {e}", flush=True)
                    rows.append({"region": region, "baseline": base_name, "seed": seed, "error": str(e)})

    df_out = pd.DataFrame(rows)
    out_csv = _ROOT / "runs" / "phase_3_region_eval.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  rows={len(df_out)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    main(args.device)
