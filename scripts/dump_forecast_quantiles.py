"""Dump per-origin predictive quantile functions for the decision-simulation.

Reuses existing HEADLINE checkpoints (no training). For each (model, region, target_ep, horizon)
produces a shared tau-grid quantile function + y_true (raw wILI), aligned across models on
(region, target_ep, horizon) so decisions are scored on identical targets.

Predictive distribution per model = 5-seed pooled samples -> empirical quantiles:
  CG-Mamba : per seed N(mu, sqrt(s2_total)) [APMD], 200 draws x5 seeds = 1000
  MC bases : per seed 100 MC-Dropout draws x5 = 500 (lstm/vanilla/patchtst/epideep)
  DLinear  : 5-seed ensemble Gaussian -> 1000 draws (matches headline UQ)

Scope = regional headline (test_strict >= 202240). USAGE:
  python scripts/dump_forecast_quantiles.py --regions hhs1        # smoke (1 region)
  python scripts/dump_forecast_quantiles.py                       # all 10 regions
Output: runs/decision_sim/forecast_quantiles.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import run_rolling_origin_baselines as B   # make_model, build_region_df, HEADLINE, DROPOUT_MC, N_MC, SEEDS
import e1_final_eval as E                   # CG-Mamba APMD forward (proven in Stage-1)
from torch.utils.data import DataLoader

OUT = _ROOT / "runs" / "decision_sim"
TAUS = np.round(np.arange(0.005, 1.0, 0.005), 4)
TEST_FIRST = 202240
SEEDS = B.SEEDS
REGIONS = B.REGIONS
HORIZONS = [1, 2, 3, 4]
RNG = np.random.default_rng(20260724)


def _grid_from_samples(samples_1d):
    return np.quantile(samples_1d, TAUS)


def cgm_region(region, device):
    """5-seed APMD-Gaussian pooled samples per (target_ep, horizon)."""
    per = {}   # (target_ep, h) -> list of sample arrays
    y_of = {}
    for seed in SEEDS:
        df = E.collect_regional_predictions("n3_d64", 3, 64, seed, region, device)
        df = df[df.target_ep >= TEST_FIRST]
        for _, r in df.iterrows():
            key = (int(r.target_ep), int(r.horizon))
            s = RNG.normal(r.mu, np.sqrt(max(r.s2_total, 1e-12)), 200)
            per.setdefault(key, []).append(s)
            y_of[key] = float(r.y_true)
    rows = []
    for key, slist in per.items():
        pooled = np.concatenate(slist)
        rows.append(("cg_mamba", region, key[0], key[1], y_of[key], _grid_from_samples(pooled)))
    return rows


def _mc_region(base, region, device):
    """5-seed pooled MC-Dropout samples per (target_ep, horizon)."""
    _, ckpt_name, _ = B.HEADLINE[base]
    norm = B.load_norm_params(B.CANON_NORM)
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    rdf = B.build_region_df(region, B.CANON_SPLIT)
    per = {}; y_of = {}; teps = None
    for seed in SEEDS:
        cell = _ROOT / B.HEADLINE[base][0] / f"seed{seed}"
        cfg, ckpt = B._load_cfg_ckpt(cell, ckpt_name, device)
        model = B.make_model(base, cfg, B.DROPOUT_MC[base]); model.load_state_dict(ckpt)
        seq_len = cfg.get("lookback") or cfg.get("seq_len")
        ds = B.WeeklyMultiHorizonDataset(rdf, "test", norm, lookback=seq_len, pred_len=cfg["pred_len"])
        model.eval().to(device)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout): m.p = B.DROPOUT_MC[base]
            elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)): m.dropout = B.DROPOUT_MC[base]
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        torch.manual_seed(seed)
        eps = ds.df["epiweek"].astype(int).to_numpy()
        tgt = {h: eps[ds.window_ends + h] for h in HORIZONS}       # target ep per horizon
        from src.eval.quantile_predictions import _dropout_train_mode
        samp = []
        with _dropout_train_mode(model):
            with torch.no_grad():
                for _ in range(B.N_MC):
                    preds, ys = [], []
                    for x, y in loader:
                        preds.append(model(x.to(device)).cpu().numpy()); ys.append(y.numpy())
                    samp.append(np.concatenate(preds, 0))
                    if teps is None: y_all = np.concatenate(ys, 0)
        samp = np.stack(samp, 0) * tstd + tmean                    # [100, N, H]
        y_raw = y_all * tstd + tmean
        for h_idx, h in enumerate(HORIZONS):
            for i in range(samp.shape[1]):
                key = (int(tgt[h][i]), h)
                per.setdefault(key, []).append(samp[:, i, h_idx])
                y_of[key] = float(y_raw[i, h_idx])
        teps = True
    rows = []
    for key, slist in per.items():
        if key[0] < TEST_FIRST: continue
        rows.append((base, region, key[0], key[1], y_of[key], _grid_from_samples(np.concatenate(slist))))
    return rows


def dlinear_region(region, device):
    """5-seed ensemble Gaussian -> pooled samples per (target_ep, horizon)."""
    norm = B.load_norm_params(B.CANON_NORM)
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    rdf = B.build_region_df(region, B.CANON_SPLIT)
    per_seed = []; y_ref = None; tgt = None
    for seed in SEEDS:
        cell = _ROOT / B.HEADLINE["dlinear"][0] / f"seed{seed}"
        cfg, ckpt = B._load_cfg_ckpt(cell, "dlinear_best.pt", device)
        model = B.DLinearForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                                    moving_avg=cfg["moving_avg"], individual=cfg["individual"])
        model.load_state_dict(ckpt); model.eval().to(device)
        ds = B.WeeklyMultiHorizonDataset(rdf, "test", norm, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
        n = len(ds); preds = np.zeros((n, 4)); ys = np.zeros((n, 4))
        with torch.no_grad():
            for i in range(n):
                x, y = ds[i]; preds[i] = model(x.unsqueeze(0).to(device))[0].cpu().numpy(); ys[i] = y.numpy()
        per_seed.append(preds)
        if y_ref is None:
            y_ref = ys
            eps = ds.df["epiweek"].astype(int).to_numpy()
            tgt = {h: eps[ds.window_ends + h] for h in HORIZONS}
    ps = np.stack(per_seed, 0)
    mu = ps.mean(0) * tstd + tmean; sigma = ps.std(0, ddof=1) * tstd; y_raw = y_ref * tstd + tmean
    rows = []
    for h_idx, h in enumerate(HORIZONS):
        for i in range(mu.shape[0]):
            tep = int(tgt[h][i])
            if tep < TEST_FIRST: continue
            samp = RNG.normal(mu[i, h_idx], max(sigma[i, h_idx], 1e-9), 1000)
            rows.append(("dlinear_ensemble_gauss", region, tep, h, float(y_raw[i, h_idx]),
                         _grid_from_samples(samp)))
    return rows


def _national_df():
    df = pd.read_csv(B.CANON_SPLIT)
    if "n_stations_available" not in df.columns:
        df["n_stations_available"] = 10
    if "weight_sum_raw" not in df.columns:
        df["weight_sum_raw"] = 1.0
    return df


def cgm_national(device):
    """5-seed APMD-Gaussian pooled samples per (target_ep, horizon) — national series."""
    per, y_of = {}, {}
    for seed in SEEDS:
        df = E.collect_national_predictions("n3_d64", 3, 64, seed, "test", device)
        df = df[df.target_ep >= TEST_FIRST]
        for _, r in df.iterrows():
            key = (int(r.target_ep), int(r.horizon))
            per.setdefault(key, []).append(RNG.normal(r.mu, np.sqrt(max(r.s2_total, 1e-12)), 200))
            y_of[key] = float(r.y_true)
    return [("cg_mamba", "national", k[0], k[1], y_of[k], _grid_from_samples(np.concatenate(v)))
            for k, v in per.items()]


def _mc_national(base, device):
    _, ckpt_name, _ = B.HEADLINE[base]
    norm = B.load_norm_params(B.CANON_NORM)
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    rdf = _national_df()
    per, y_of = {}, {}
    for seed in SEEDS:
        cfg, ckpt = B._load_cfg_ckpt(_ROOT / B.HEADLINE[base][0] / f"seed{seed}", ckpt_name, device)
        model = B.make_model(base, cfg, B.DROPOUT_MC[base]); model.load_state_dict(ckpt)
        seq_len = cfg.get("lookback") or cfg.get("seq_len")
        ds = B.WeeklyMultiHorizonDataset(rdf, "test", norm, lookback=seq_len, pred_len=cfg["pred_len"])
        model.eval().to(device)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout): m.p = B.DROPOUT_MC[base]
            elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)): m.dropout = B.DROPOUT_MC[base]
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        torch.manual_seed(seed)
        eps = ds.df["epiweek"].astype(int).to_numpy()
        tgt = {h: eps[ds.window_ends + h] for h in HORIZONS}
        from src.eval.quantile_predictions import _dropout_train_mode
        samp = []; y_all = None
        with _dropout_train_mode(model):
            with torch.no_grad():
                for _ in range(B.N_MC):
                    preds, ys = [], []
                    for x, y in loader:
                        preds.append(model(x.to(device)).cpu().numpy()); ys.append(y.numpy())
                    samp.append(np.concatenate(preds, 0))
                    if y_all is None: y_all = np.concatenate(ys, 0)
        samp = np.stack(samp, 0) * tstd + tmean; y_raw = y_all * tstd + tmean
        for h_idx, h in enumerate(HORIZONS):
            for i in range(samp.shape[1]):
                key = (int(tgt[h][i]), h)
                per.setdefault(key, []).append(samp[:, i, h_idx]); y_of[key] = float(y_raw[i, h_idx])
    return [(base, "national", k[0], k[1], y_of[k], _grid_from_samples(np.concatenate(v)))
            for k, v in per.items() if k[0] >= TEST_FIRST]


def dlinear_national(device):
    norm = B.load_norm_params(B.CANON_NORM)
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    rdf = _national_df()
    per_seed, y_ref, tgt = [], None, None
    for seed in SEEDS:
        cfg, ckpt = B._load_cfg_ckpt(_ROOT / B.HEADLINE["dlinear"][0] / f"seed{seed}", "dlinear_best.pt", device)
        model = B.DLinearForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                                    moving_avg=cfg["moving_avg"], individual=cfg["individual"])
        model.load_state_dict(ckpt); model.eval().to(device)
        ds = B.WeeklyMultiHorizonDataset(rdf, "test", norm, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
        n = len(ds); preds = np.zeros((n, 4)); ys = np.zeros((n, 4))
        with torch.no_grad():
            for i in range(n):
                x, y = ds[i]; preds[i] = model(x.unsqueeze(0).to(device))[0].cpu().numpy(); ys[i] = y.numpy()
        per_seed.append(preds)
        if y_ref is None:
            y_ref = ys; eps = ds.df["epiweek"].astype(int).to_numpy()
            tgt = {h: eps[ds.window_ends + h] for h in HORIZONS}
    ps = np.stack(per_seed, 0); mu = ps.mean(0) * tstd + tmean; sigma = ps.std(0, ddof=1) * tstd
    y_raw = y_ref * tstd + tmean; rows = []
    for h_idx, h in enumerate(HORIZONS):
        for i in range(mu.shape[0]):
            tep = int(tgt[h][i])
            if tep < TEST_FIRST: continue
            rows.append(("dlinear_ensemble_gauss", "national", tep, h, float(y_raw[i, h_idx]),
                         _grid_from_samples(RNG.normal(mu[i, h_idx], max(sigma[i, h_idx], 1e-9), 1000))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--national", action="store_true", help="national scope instead of regions")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    if args.national:
        print("[dump] national ...", flush=True)
        all_rows += cgm_national(args.device)
        for base in B.MC_BASES:
            all_rows += _mc_national(base, args.device)
        all_rows += dlinear_national(args.device)
        print(f"  national: rows={len(all_rows)}")
    else:
        for region in args.regions:
            print(f"[dump] {region} ...", flush=True)
            all_rows += cgm_region(region, args.device)
            for base in B.MC_BASES:
                all_rows += _mc_region(base, region, args.device)
            all_rows += dlinear_region(region, args.device)
            print(f"  {region}: cumulative rows={len(all_rows)}")

    qcols = [f"q{t:.4f}" for t in TAUS]
    recs = []
    for model, scope, tep, h, y, grid in all_rows:
        d = {"model": model, "scope": scope, "target_ep": tep, "horizon": h, "y_true": y}
        d.update({qcols[i]: grid[i] for i in range(len(TAUS))})
        recs.append(d)
    df = pd.DataFrame(recs)
    if args.national:
        out = OUT / "forecast_quantiles_national.parquet"
    elif len(args.regions) == 10:
        out = OUT / "forecast_quantiles.parquet"
    else:
        out = OUT / f"forecast_quantiles_{'_'.join(args.regions)}.parquet"
    df.to_parquet(out, index=False)
    print(f"\n[dump] {len(df)} rows, models={sorted(df.model.unique())} -> {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
