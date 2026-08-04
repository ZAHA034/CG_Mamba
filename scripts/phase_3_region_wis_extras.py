"""§18 Phase 3 Extras WIS — EpiDeep MC Dropout + DLinear ensemble Gaussian.

Companion to:
  - scripts/phase_3_region_wis.py (LSTM/Vanilla Mamba/PatchTST MC Dropout)
  - scripts/phase_3_cgm_method_f_region.py (CG-Mamba Method F)
  - scripts/phase_3_sarima_wis_region.py (SARIMA Kalman parametric)

Method-specific UQ choices (matching §IV.6 calibration table):
  - EpiDeep:  MC Dropout d=0.1 (n=100 samples)
  - DLinear:  5-seed ensemble Gaussian (μ, σ over 5 deterministic predictions)

Outputs:
  runs/phase_3_region_wis_extras.csv
    EpiDeep:  50 rows (10 regions × 5 seeds)
    DLinear:  10 rows (10 regions, single ensemble per region, seed=-1)

Design notes:
  - Self-contained: does NOT import scripts.phase_3_region_wis to avoid
    transitive PatchTST/TSLIB namespace collisions (see extras eval doc).
  - WIS/coverage helpers reused from src.eval.wis (no TSLIB deps).
  - Uses identical build_region_df / split protocol / 23 FluSight quantile
    levels as canonical script for evaluation-pipeline parity.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset
from src.baselines.dlinear import DLinearForecaster
from src.baselines.epideep import EpiDeepForecaster
from src.data.loader import load_norm_params
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES
from src.eval.quantile_predictions import _dropout_train_mode

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240
REGIONS = [f"hhs{i}" for i in range(1, 11)]
SEEDS = [42, 123, 456, 789, 1024]
N_MC_SAMPLES = 100
EPIDEEP_DROPOUT = 0.1

DLINEAR_DIR = _ROOT / "runs/dlinear_final/ma13_indF_lr2e-03"
EPIDEEP_DIR = _ROOT / "runs/epideep_final/de128_eh64_lr2e-03"


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (inlined to avoid cross-script TSLIB namespace pollution)
# ──────────────────────────────────────────────────────────────────────────────
def build_region_df(region: str) -> pd.DataFrame:
    region_csv = _ROOT / "data" / "raw" / "cdc_ilinet" / "_phase3_phase6_fetch" / f"{region}_full.csv"
    df_r = pd.read_csv(region_csv)
    df_r["epiweek"] = df_r["year"].astype(int) * 100 + df_r["week"].astype(int)
    from epiweeks import Week
    df_r["date"] = df_r.apply(lambda r: Week(int(r["year"]), int(r["week"])).startdate().isoformat(), axis=1)
    env = pd.read_csv(_ROOT / "data" / "processed" / "env_national_weekly.csv")
    df_merged = df_r.merge(env[["epiweek", "temperature_c", "specific_humidity_g_per_kg"]],
                            on="epiweek", how="inner")
    split = pd.read_csv(_ROOT / "data" / "processed" / "ili_env_weekly_split.csv")
    df_merged = df_merged.merge(split[["epiweek", "split"]], on="epiweek", how="inner")
    df_merged["n_stations_available"] = 10
    df_merged["weight_sum_raw"] = 1.0
    return df_merged


def _ts_idx(eps_h1: np.ndarray) -> np.ndarray:
    return np.where(eps_h1 >= TS_BOUNDARY)[0]


def _wis_cov_per_h(samples, y_raw, ts_idx, out):
    """samples: [S, N, H], y_raw: [N, H]."""
    for h_idx, h in enumerate(HORIZONS):
        qf = {q: np.quantile(samples[:, :, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
        out[f"tF_wis_h{h}"]   = float(wis(y_raw[:, h_idx], qf).mean())
        out[f"tF_cov95_h{h}"] = float(coverage(y_raw[:, h_idx], qf, alpha=0.05))
        if len(ts_idx) > 0:
            qf_ts = {q: np.quantile(samples[:, ts_idx, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
            out[f"tS_wis_h{h}"]   = float(wis(y_raw[ts_idx, h_idx], qf_ts).mean())
            out[f"tS_cov95_h{h}"] = float(coverage(y_raw[ts_idx, h_idx], qf_ts, alpha=0.05))


def _gaussian_quantiles_to_samples(mu: np.ndarray, sigma: np.ndarray, n_samples: int) -> np.ndarray:
    """For Gaussian-distributed forecasts, materialize n_samples samples per (N, H).

    mu:    [N, H]
    sigma: [N, H]  (positive)
    Returns: samples [n_samples, N, H]
    """
    rng = np.random.default_rng(seed=20260529)
    z = rng.standard_normal((n_samples, *mu.shape))
    return mu[None, :, :] + z * sigma[None, :, :]


# ──────────────────────────────────────────────────────────────────────────────
# EpiDeep MC Dropout
# ──────────────────────────────────────────────────────────────────────────────
def eval_epideep_wis(region_df, seed, device):
    p = EPIDEEP_DIR / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "epideep_best.pt", map_location=device, weights_only=True)
    model = EpiDeepForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
        decoder_hidden=cfg["decoder_hidden"],
        alignment_weight=cfg.get("alignment_weight", 0.0),
        dropout=EPIDEEP_DROPOUT,            # force d=0.1 for MC eval
        target_only=cfg.get("target_only", False),
    )
    model.load_state_dict(ckpt)
    # Force dropout layers to active rate (in case ckpt has dropout=0)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = EPIDEEP_DROPOUT

    ds = WeeklyMultiHorizonDataset(region_df, "test", NORM,
                                     lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    model.eval().to(device)

    all_samples = []
    y_collect = None
    with _dropout_train_mode(model):
        with torch.no_grad():
            for _ in range(N_MC_SAMPLES):
                preds, ys = [], []
                for x, y in loader:
                    x = x.to(device)
                    preds.append(model(x).cpu().numpy())
                    ys.append(y.numpy())
                preds = np.concatenate(preds, axis=0)
                ys = np.concatenate(ys, axis=0)
                all_samples.append(preds)
                if y_collect is None:
                    y_collect = ys

    samples = np.stack(all_samples, axis=0) * TARGET_STD + TARGET_MEAN
    y_raw = y_collect * TARGET_STD + TARGET_MEAN
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    ts_idx = _ts_idx(eps_h1)
    out = {"baseline": "epideep", "seed": seed,
           "n_full": len(y_raw), "n_strict": len(ts_idx),
           "dropout": EPIDEEP_DROPOUT}
    _wis_cov_per_h(samples, y_raw, ts_idx, out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DLinear 5-seed ensemble Gaussian
# ──────────────────────────────────────────────────────────────────────────────
def _dlinear_predict_one_seed(seed, region_df, device):
    """Returns preds_z [N, H], ys_z [N, H], eps_h1 [N]."""
    p = DLINEAR_DIR / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "dlinear_best.pt", map_location=device, weights_only=True)
    model = DLinearForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        moving_avg=cfg["moving_avg"], individual=cfg["individual"],
    )
    model.load_state_dict(ckpt)
    model.eval().to(device)
    ds = WeeklyMultiHorizonDataset(region_df, "test", NORM,
                                     lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    n = len(ds)
    preds = np.zeros((n, 4))
    ys = np.zeros((n, 4))
    with torch.no_grad():
        for i in range(n):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            preds[i] = model(x)[0].cpu().numpy()
            ys[i] = y.numpy()
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    return preds, ys, eps_h1


def eval_dlinear_ensemble_gaussian_wis(region_df, device):
    """DLinear ensemble Gaussian: μ, σ over 5 deterministic seed predictions.

    Returns one row per region (seed=-1 to denote ensemble).
    """
    per_seed_preds = []
    ys_ref = None
    eps_ref = None
    for seed in SEEDS:
        preds_z, ys_z, eps_h1 = _dlinear_predict_one_seed(seed, region_df, device)
        per_seed_preds.append(preds_z)
        if ys_ref is None:
            ys_ref = ys_z
            eps_ref = eps_h1
    per_seed_preds = np.stack(per_seed_preds, axis=0)   # [5, N, H]
    mu_z = per_seed_preds.mean(axis=0)                   # [N, H]
    sigma_z = per_seed_preds.std(axis=0, ddof=1)         # [N, H]   ddof=1 → sample std
    # Denormalize to raw scale
    mu_raw    = mu_z    * TARGET_STD + TARGET_MEAN
    sigma_raw = sigma_z * TARGET_STD                     # std scales linearly
    y_raw     = ys_ref  * TARGET_STD + TARGET_MEAN
    # Materialize Gaussian samples for unified WIS/cov pipeline
    samples = _gaussian_quantiles_to_samples(mu_raw, sigma_raw, N_MC_SAMPLES)
    ts_idx = _ts_idx(eps_ref)
    out = {"baseline": "dlinear_ensemble_gauss", "seed": -1,
           "n_full": len(y_raw), "n_strict": len(ts_idx),
           "dropout": float("nan")}
    _wis_cov_per_h(samples, y_raw, ts_idx, out)
    return out


def main(device="cuda:1"):
    if not torch.cuda.is_available(): device = "cpu"
    rows = []
    print(f"Device: {device}", flush=True)

    for region in REGIONS:
        print(f"\n=== {region} ===", flush=True)
        region_df = build_region_df(region)

        # EpiDeep MC Dropout (5 seeds)
        for seed in SEEDS:
            try:
                r = eval_epideep_wis(region_df, seed, device)
                r["region"] = region
                rows.append(r)
                print(f"  OK epideep    s={seed:>4}  tS_wis_h1={r.get('tS_wis_h1', float('nan')):.4f}  cov95={r.get('tS_cov95_h1', float('nan')):.3f}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL epideep s={seed}: {type(e).__name__}: {e}", flush=True)
                rows.append({"region": region, "baseline": "epideep", "seed": seed, "error": str(e)})

        # DLinear 5-seed ensemble Gaussian (one row per region)
        try:
            r = eval_dlinear_ensemble_gaussian_wis(region_df, device)
            r["region"] = region
            rows.append(r)
            print(f"  OK dlinear_ensemble  tS_wis_h1={r.get('tS_wis_h1', float('nan')):.4f}  cov95={r.get('tS_cov95_h1', float('nan')):.3f}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL dlinear_ensemble: {type(e).__name__}: {e}", flush=True)
            rows.append({"region": region, "baseline": "dlinear_ensemble_gauss", "seed": -1, "error": str(e)})

    df_out = pd.DataFrame(rows)
    out_csv = _ROOT / "runs" / "phase_3_region_wis_extras.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  rows={len(df_out)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    main(args.device)
