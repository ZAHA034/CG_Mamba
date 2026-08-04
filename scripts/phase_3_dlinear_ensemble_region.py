"""§18 Phase 3 — DLinear 5-seed ensemble Gaussian WIS per-region.

M2.3a J.8 Template 2 protocol for DLinear:
  - 5 seeds × deterministic forward → mean + std → Gaussian quantiles → WIS

Output: runs/phase_3_dlinear_ensemble_region.csv
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
_TSLIB = _ROOT.parent / "Time-Series-Library"
# Time-Series-Library FIRST + force-import models to cache before src/models shadows
sys.path.insert(0, str(_TSLIB))
import models  # noqa — force-cache Time-Series-Library/models as sys.modules['models']
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset
from baselines.dlinear import DLinearForecaster
from src.data.loader import load_norm_params
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES


def build_region_df(region: str) -> pd.DataFrame:
    """Inline copy of scripts.phase_3_region_eval.build_region_df (avoid patchtst import)."""
    from epiweeks import Week
    region_csv = _ROOT / "data" / "raw" / "cdc_ilinet" / "_phase3_phase6_fetch" / f"{region}_full.csv"
    df_r = pd.read_csv(region_csv)
    df_r["epiweek"] = df_r["year"].astype(int) * 100 + df_r["week"].astype(int)
    df_r["date"] = df_r.apply(lambda r: Week(int(r["year"]), int(r["week"])).startdate().isoformat(), axis=1)
    env = pd.read_csv(_ROOT / "data" / "processed" / "env_national_weekly.csv")
    df_merged = df_r.merge(env[["epiweek","temperature_c","specific_humidity_g_per_kg"]], on="epiweek", how="inner")
    split = pd.read_csv(_ROOT / "data" / "processed" / "ili_env_weekly_split.csv")
    df_merged = df_merged.merge(split[["epiweek","split"]], on="epiweek", how="inner")
    df_merged["n_stations_available"] = 10
    df_merged["weight_sum_raw"] = 1.0
    return df_merged

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240
REGIONS = [f"hhs{i}" for i in range(1, 11)]
SEEDS = [42, 123, 456, 789, 1024]


def eval_dlinear_ensemble(region, device="cpu"):
    region_df = build_region_df(region)
    # Forward 5 seeds
    preds_per_seed = []
    y_collect = None
    ds_obj = None
    for seed in SEEDS:
        p = _ROOT / "runs/dlinear_final/ma13_indF_lr2e-03" / f"seed{seed}"
        r = json.load(open(p / "results.json"))
        cfg = r["config"]
        ckpt = torch.load(p / "dlinear_best.pt", map_location=device, weights_only=True)
        model = DLinearForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
                                    enc_in=cfg["enc_in"], moving_avg=cfg["moving_avg"],
                                    individual=cfg["individual"])
        model.load_state_dict(ckpt)
        model.eval().to(device)
        ds = WeeklyMultiHorizonDataset(region_df, "test", NORM,
                                        lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
        if ds_obj is None: ds_obj = ds
        preds = np.zeros((len(ds), 4))
        ys = np.zeros((len(ds), 4))
        with torch.no_grad():
            for i in range(len(ds)):
                x, y = ds[i]
                x = x.unsqueeze(0).to(device)
                preds[i] = model(x)[0].cpu().numpy()
                ys[i] = y.numpy()
        preds_per_seed.append(preds)
        if y_collect is None: y_collect = ys
    preds_per_seed = np.stack(preds_per_seed, axis=0)  # [5, N, H]
    n = preds_per_seed.shape[1]

    # Ensemble Gaussian: mean + std across seeds
    mu = preds_per_seed.mean(axis=0)
    sig = preds_per_seed.std(axis=0, ddof=1)
    sig = np.maximum(sig, 1e-6)
    # Sample 100 synthetic predictions for WIS quantiles
    rng = np.random.RandomState(42)
    samples = rng.normal(loc=mu[None, :, :], scale=sig[None, :, :], size=(100, n, 4))
    # Convert to raw
    samples_raw = samples * TARGET_STD + TARGET_MEAN
    y_raw = y_collect * TARGET_STD + TARGET_MEAN
    eps = ds_obj.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds_obj.window_ends + 1]
    ts_mask = eps_h1 >= TS_BOUNDARY

    out = {"region": region, "baseline": "dlinear_ensemble",
           "n_full": n, "n_strict": int(ts_mask.sum())}
    for h_idx, h in enumerate(HORIZONS):
        qf = {q: np.quantile(samples_raw[:, :, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
        out[f"tF_wis_h{h}"] = float(wis(y_raw[:, h_idx], qf).mean())
        out[f"tF_cov95_h{h}"] = float(coverage(y_raw[:, h_idx], qf, alpha=0.05))
        if ts_mask.sum() > 0:
            qf_ts = {q: np.quantile(samples_raw[:, ts_mask, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
            out[f"tS_wis_h{h}"] = float(wis(y_raw[ts_mask, h_idx], qf_ts).mean())
            out[f"tS_cov95_h{h}"] = float(coverage(y_raw[ts_mask, h_idx], qf_ts, alpha=0.05))
    return out


def main():
    rows = []
    for region in REGIONS:
        try:
            r = eval_dlinear_ensemble(region)
            rows.append(r)
            print(f"  ✓ {region}  tS_wis_h1={r['tS_wis_h1']:.4f}  cov95={r['tS_cov95_h1']:.3f}", flush=True)
        except Exception as e:
            print(f"  ✗ {region}: {e}", flush=True)
            rows.append({"region": region, "baseline": "dlinear_ensemble", "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(_ROOT / "runs" / "phase_3_dlinear_ensemble_region.csv", index=False)
    print(f"\nSaved: {len(df)} rows")


if __name__ == "__main__":
    main()
