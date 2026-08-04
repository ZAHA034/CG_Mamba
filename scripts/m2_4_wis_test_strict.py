"""M2.4 unified WIS evaluation — Protocol-specific UQ + Split Conformal (2-track).

Track 1 — Protocol-specific UQ (M2.3 §II.4):
  SARIMA:         Kalman parametric (rolling get_forecast + se_mean → Gaussian Q)
  DLinear:        5-seed ensemble Gaussian
  LSTM:           MC Dropout p=0.3 n=100
  Vanilla Mamba:  MC Dropout p=0.2 n=100
  PatchTST:       MC Dropout p=0.1 n=100
  CG-Mamba:       Method F (HMM-derived calibrated PI)

Track 2 — Split Conformal (Vovk 2005, Romano 2019):
  All baselines:  val residual empirical quantiles, finite-sample corrected

Output:
  runs/m2_4_data_efficiency/m2_4_wis_protocol.csv   (Track 1)
  runs/m2_4_data_efficiency/m2_4_wis_conformal.csv  (Track 2)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset
from cm_mamba.baselines.lstm_baseline import LSTMForecaster
from baselines.vanilla_mamba import VanillaMambaForecaster
from baselines.dlinear import DLinearForecaster
from baselines.patchtst import PatchTSTForecaster
from baselines.epideep import EpiDeepForecaster
from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import WeeklyDataset, MultiHorizonDataset, collate_dict, load_norm_params
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES
from src.eval.quantile_predictions import _dropout_train_mode
from src.eval.hmm_interval import (
    compute_decomposition, method_f_predict_quantiles,
)
from baselines.sarima import build_segment_arrays, fit_sarimax, assert_continuous

ROOT_RUNS = _ROOT / "runs" / "m2_4_data_efficiency"
VARIANTS = ["3_seasons", "4_seasons", "5_seasons", "7_seasons",
            "10_seasons", "13_seasons", "17_seasons_full"]
SEEDS = [42, 123, 456, 789, 1024]
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])

DROPOUT_MC = {"lstm": 0.3, "vanilla_mamba": 0.2, "patchtst": 0.1, "epideep": 0.1}
N_MC_SAMPLES = 100


# ═════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═════════════════════════════════════════════════════════════════════════════

def _ts_split_idx(eps_h1: np.ndarray) -> np.ndarray:
    return np.where(eps_h1 >= TS_BOUNDARY)[0]


def _wis_and_cov_per_h(quantile_forecasts: dict, y: np.ndarray,
                        prefix: str, out: dict) -> None:
    """Compute WIS + Cov95 per horizon from quantile forecasts.
    quantile_forecasts: {q: array [N, H]}, y: [N, H] raw scale."""
    for h_idx, h in enumerate(HORIZONS):
        qf = {q: quantile_forecasts[q][:, h_idx] for q in quantile_forecasts}
        out[f"{prefix}_wis_h{h}"] = float(wis(y[:, h_idx], qf).mean())
        out[f"{prefix}_cov95_h{h}"] = float(coverage(y[:, h_idx], qf, alpha=0.05))


def _load_nn_model(baseline: str, variant: str, seed: int, device: str,
                    dropout_override: float | None = None):
    """Load NN baseline model + config. Returns (model, config_dict)."""
    p = ROOT_RUNS / baseline / f"seasons_{variant}" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt_name = f"{baseline}_best.pt"
    ckpt = torch.load(p / ckpt_name, map_location=device, weights_only=True)
    dr = dropout_override if dropout_override is not None else 0.0

    if baseline == "lstm":
        model = LSTMForecaster(
            enc_in=cfg["enc_in"], hidden=cfg["hidden"],
            num_layers=cfg["num_layers"], pred_len=cfg["pred_len"],
            dropout=dr,
        )
    elif baseline == "vanilla_mamba":
        model = VanillaMambaForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
            enc_in=cfg["enc_in"], d_model=cfg["d_model"],
            n_layers=cfg["n_layers"], d_state=cfg["d_state"],
            dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=dr,
        )
    elif baseline == "patchtst":
        stride = max(1, int(cfg["patch_len"] * cfg["stride_ratio"]))
        model = PatchTSTForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
            enc_in=cfg["enc_in"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
            d_ff=cfg["d_ff_ratio"] * cfg["d_model"],
            patch_len=cfg["patch_len"], stride=stride, dropout=dr,
        )
    elif baseline == "dlinear":
        model = DLinearForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
            enc_in=cfg["enc_in"], moving_avg=cfg["moving_avg"],
            individual=cfg["individual"],
        )
    elif baseline == "epideep":
        model = EpiDeepForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
            decoder_hidden=cfg["decoder_hidden"],
            alignment_weight=cfg.get("alignment_weight", 0.0),
            dropout=dr,
            target_only=cfg.get("target_only", False),
        )
    else:
        raise ValueError(f"Unknown NN baseline: {baseline}")

    model.load_state_dict(ckpt, strict=False)
    model.eval().to(device)
    return model, cfg


def _nn_forward_deterministic(model, ds, device):
    """Single deterministic forward → (preds_zscored [N,H], y_zscored [N,H])."""
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds_all, ys_all = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            preds_all.append(pred.cpu().numpy())
            ys_all.append(y.numpy())
    return np.concatenate(preds_all, 0), np.concatenate(ys_all, 0)


def _get_nn_dataset(variant: str, split: str, cfg: dict):
    csv = ROOT_RUNS / "_filtered_csvs" / f"ili_env_weekly_split_{variant}.csv"
    df = pd.read_csv(csv)
    seq_len = cfg.get("lookback") or cfg.get("seq_len")
    return WeeklyMultiHorizonDataset(df, split, NORM,
                                      lookback=seq_len, pred_len=cfg["pred_len"]), df


def _get_epiweeks_h1(ds, df):
    eps = df["epiweek"].astype(int).to_numpy()
    return eps[ds.window_ends + 1]


# ═════════════════════════════════════════════════════════════════════════════
# Track 1: Protocol-specific UQ
# ═════════════════════════════════════════════════════════════════════════════

def _mc_dropout_quantiles(model, ds, device, dropout_rate):
    """MC Dropout n=100 batched → quantile forecasts {q: [N,H]} raw scale + y_raw."""
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout_rate
        elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            m.dropout = dropout_rate

    all_samples = []
    y_collect = None
    with _dropout_train_mode(model):
        with torch.no_grad():
            for _ in range(N_MC_SAMPLES):
                preds, ys = [], []
                for x, y in loader:
                    preds.append(model(x.to(device)).cpu().numpy())
                    ys.append(y.numpy())
                all_samples.append(np.concatenate(preds, 0))
                if y_collect is None:
                    y_collect = np.concatenate(ys, 0)
    samples = np.stack(all_samples, 0)  # [S, N, H]
    samples_raw = samples * TARGET_STD + TARGET_MEAN
    y_raw = y_collect * TARGET_STD + TARGET_MEAN
    qf = {q: np.quantile(samples_raw, q, axis=0) for q in REQUIRED_QUANTILES}
    return qf, y_raw


def protocol_nn_mc(baseline: str, variant: str, seed: int, device: str) -> dict:
    """LSTM / Vanilla / PatchTST protocol WIS via MC Dropout."""
    dr = DROPOUT_MC[baseline]
    model, cfg = _load_nn_model(baseline, variant, seed, device, dropout_override=dr)
    test_ds, df = _get_nn_dataset(variant, "test", cfg)
    eps_h1 = _get_epiweeks_h1(test_ds, df)
    qf, y_raw = _mc_dropout_quantiles(model, test_ds, device, dr)
    ts_idx = _ts_split_idx(eps_h1)
    out = {"baseline": baseline, "variant": variant, "seed": seed,
           "uq_method": f"mc_dropout_p{dr}", "n_full": len(y_raw),
           "n_strict": len(ts_idx)}
    _wis_and_cov_per_h(qf, y_raw, "test_full", out)
    if len(ts_idx) > 0:
        qf_ts = {q: qf[q][ts_idx] for q in qf}
        _wis_and_cov_per_h(qf_ts, y_raw[ts_idx], "test_strict", out)
    return out


def protocol_dlinear_ensemble(variant: str, device: str) -> dict:
    """DLinear: 5-seed ensemble Gaussian quantiles."""
    preds_seeds = []
    y_raw_ref = None
    eps_h1_ref = None
    for seed in SEEDS:
        model, cfg = _load_nn_model("dlinear", variant, seed, device)
        test_ds, df = _get_nn_dataset(variant, "test", cfg)
        preds_z, ys_z = _nn_forward_deterministic(model, test_ds, device)
        preds_seeds.append(preds_z)
        if y_raw_ref is None:
            y_raw_ref = ys_z * TARGET_STD + TARGET_MEAN
            eps_h1_ref = _get_epiweeks_h1(test_ds, df)
    member_preds_raw = np.stack(preds_seeds, 0) * TARGET_STD + TARGET_MEAN  # [5,N,H]
    mu = member_preds_raw.mean(axis=0)
    var = member_preds_raw.var(axis=0, ddof=1)
    from scipy.stats import norm
    qf = {q: mu + norm.ppf(q) * np.sqrt(np.maximum(var, 1e-12))
           for q in REQUIRED_QUANTILES}
    ts_idx = _ts_split_idx(eps_h1_ref)
    out = {"baseline": "dlinear", "variant": variant, "seed": -1,
           "uq_method": "ensemble_gaussian_5seed",
           "n_full": len(y_raw_ref), "n_strict": len(ts_idx)}
    _wis_and_cov_per_h(qf, y_raw_ref, "test_full", out)
    if len(ts_idx) > 0:
        qf_ts = {q: qf[q][ts_idx] for q in qf}
        _wis_and_cov_per_h(qf_ts, y_raw_ref[ts_idx], "test_strict", out)
    return out


def protocol_cgm_method_f(variant: str, seed: int, device: str) -> dict:
    """CG-Mamba: Method F (HMM-derived calibrated PI)."""
    import dataclasses
    m_path = ROOT_RUNS / "cg_mamba" / f"seasons_{variant}" / f"seed{seed}" / "manifest.json"
    m = json.load(open(m_path))
    hmm_dir = _ROOT / m["hmm_dir"] if not Path(m["hmm_dir"]).is_absolute() else Path(m["hmm_dir"])
    hmm = load_fitted_hmm(hmm_dir)
    cfg = CGMambaConfig()
    model = CGForecaster(cfg).to(device)
    model.prepare_for_stage2(hmm)
    ckpt_path = _ROOT / m["stage3_best"] if not Path(m["stage3_best"]).is_absolute() else Path(m["stage3_best"])
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval()

    pm = model.phase_module
    mu_k_ili = pm._means[:, 0].cpu().numpy()
    sigma2_k_ili = pm._covs[:, 0, 0].cpu().numpy()
    sigma2_k_ili = np.maximum(sigma2_k_ili, 1e-6)  # R2: ill-conditioning guard

    csv_path = _ROOT / m["csv_used"] if not Path(m["csv_used"]).is_absolute() else Path(m["csv_used"])
    df = pd.read_csv(csv_path)

    def _make_loader(split):
        ds = MultiHorizonDataset(df, split, cfg.lookback, tuple(cfg.horizons), NORM)
        return DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                          collate_fn=collate_dict), ds

    @torch.no_grad()
    def _forward(loader):
        mus, gammas, ys, eps_list = [], [], [], []
        for batch in loader:
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            preds, intermediates = model(x, env, return_intermediates=True)
            mus.append(preds.cpu().numpy())
            gammas.append(intermediates["gamma_all"].cpu().numpy())
            ys.append(batch["y"].cpu().numpy())
            if "target_epiweeks" in batch:
                eps_list.extend(batch["target_epiweeks"])
        return (np.concatenate(mus, 0), np.concatenate(gammas, 0),
                np.concatenate(ys, 0), eps_list)

    def _sanitize_gamma(gamma: np.ndarray) -> np.ndarray:
        """Fix overflow/NaN in gamma_all from rollout numerical instability.
        Re-normalize each [K] slice to valid probability simplex."""
        bad = np.isnan(gamma) | np.isinf(gamma) | (np.abs(gamma) > 1e6)
        if bad.any():
            N, H, K = gamma.shape
            for n in range(N):
                for h in range(H):
                    if bad[n, h].any():
                        gamma[n, h, :] = 1.0 / K  # uniform fallback
            gamma = np.maximum(gamma, 0.0)
            sums = gamma.sum(axis=-1, keepdims=True)
            sums = np.maximum(sums, 1e-30)
            gamma = gamma / sums
        return gamma

    val_loader, _ = _make_loader("val")
    test_loader, test_ds = _make_loader("test")
    mu_val, gamma_val, y_val, _ = _forward(val_loader)
    mu_test, gamma_test, y_test, eps_test = _forward(test_loader)
    gamma_val = _sanitize_gamma(gamma_val)
    gamma_test = _sanitize_gamma(gamma_test)

    q_raw, meta = method_f_predict_quantiles(
        mu_CGM_test=mu_test, gamma_all_test=gamma_test,
        mu_CGM_val=mu_val, gamma_all_val=gamma_val,
        y_val=y_val,
        mu_k_ili=mu_k_ili, sigma2_k_ili=sigma2_k_ili,
        target_mean=TARGET_MEAN, target_std=TARGET_STD,
    )
    y_raw = y_test * TARGET_STD + TARGET_MEAN

    eps_h1 = np.array([ep[0] if isinstance(ep, (list, np.ndarray)) else ep
                        for ep in eps_test]) if eps_test else np.zeros(len(y_raw))
    ts_idx = _ts_split_idx(eps_h1)

    out = {"baseline": "cg_mamba", "variant": variant, "seed": seed,
           "uq_method": "method_f", "n_full": len(y_raw), "n_strict": len(ts_idx),
           "method_f_mode": meta["quantile_mode"],
           "method_f_s_per_h": meta["s_per_h"]}
    _wis_and_cov_per_h(q_raw, y_raw, "test_full", out)
    if len(ts_idx) > 0:
        q_ts = {q: q_raw[q][ts_idx] for q in q_raw}
        _wis_and_cov_per_h(q_ts, y_raw[ts_idx], "test_strict", out)
    return out


def protocol_sarima_kalman(variant: str, device: str) -> dict:
    """SARIMA: refit with stored order → rolling get_forecast + se_mean → WIS."""
    from scipy.stats import norm as scipy_norm

    sarima_json = ROOT_RUNS / "sarima" / f"seasons_{variant}.json"
    meta = json.load(open(sarima_json))
    order = tuple(meta["selected_order"])
    seasonal_order = tuple(meta["seasonal_order"])

    csv = ROOT_RUNS / "_filtered_csvs" / f"ili_env_weekly_split_{variant}.csv"
    df = pd.read_csv(csv)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    val_df = df[df["split"] == "val"]

    target_col = "ili_weighted_pct"
    exog_cols = [c for c in df.columns if c not in
                 ["epiweek", "date", "split", target_col,
                  "n_stations_available", "weight_sum_raw"]]

    y_tr = train_df[target_col].to_numpy(dtype=np.float64)
    X_tr = train_df[exog_cols].to_numpy(dtype=np.float64)
    y_va = val_df[target_col].to_numpy(dtype=np.float64)
    X_va = val_df[exog_cols].to_numpy(dtype=np.float64)
    y_te = test_df[target_col].to_numpy(dtype=np.float64)
    X_te = test_df[exog_cols].to_numpy(dtype=np.float64)
    ep_te = test_df["epiweek"].astype(int).to_numpy()

    y_train_val = np.concatenate([y_tr, y_va])
    X_train_val = np.vstack([X_tr, X_va])

    res = fit_sarimax(y_train_val, X_train_val, order, seasonal_order)

    H = max(HORIZONS)
    preds_per_h = {h: [] for h in HORIZONS}
    current_res = res
    N_te = len(y_te)

    for t in range(N_te):
        steps = min(H, N_te - t)
        if steps == 0:
            break
        future_exog = X_te[t:t + steps]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_obj = current_res.get_forecast(steps=steps, exog=future_exog)
            fc_mean = np.asarray(fc_obj.predicted_mean, dtype=np.float64)
            fc_se = np.asarray(fc_obj.se_mean, dtype=np.float64)
            fc_se = np.maximum(fc_se, 1e-8)

        for h in HORIZONS:
            ti = t + h - 1
            if ti >= N_te:
                continue
            preds_per_h[h].append({
                "target_ep": int(ep_te[ti]),
                "y_true": float(y_te[ti]),
                "y_pred": float(fc_mean[h - 1]),
                "y_se": float(fc_se[h - 1]),
            })

        if t + 1 < N_te:
            new_exog = X_te[t:t + 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_res = current_res.append([y_te[t]], exog=new_exog, refit=False)

    out = {"baseline": "sarima", "variant": variant, "seed": -1,
           "uq_method": "kalman_parametric",
           "n_full": 0, "n_strict": 0}

    for h in HORIZONS:
        recs = preds_per_h[h]
        if not recs:
            continue
        y_arr = np.array([r["y_true"] for r in recs])
        mu_arr = np.array([r["y_pred"] for r in recs])
        se_arr = np.array([r["y_se"] for r in recs])
        ep_arr = np.array([r["target_ep"] for r in recs])

        qf_full = {q: mu_arr + scipy_norm.ppf(q) * se_arr for q in REQUIRED_QUANTILES}
        out[f"test_full_wis_h{h}"] = float(wis(y_arr, qf_full).mean())
        out[f"test_full_cov95_h{h}"] = float(coverage(y_arr, qf_full, alpha=0.05))
        out["n_full"] = max(out["n_full"], len(y_arr))

        ts_mask = ep_arr >= TS_BOUNDARY
        if ts_mask.sum() > 0:
            qf_ts = {q: qf_full[q][ts_mask] for q in qf_full}
            out[f"test_strict_wis_h{h}"] = float(wis(y_arr[ts_mask], qf_ts).mean())
            out[f"test_strict_cov95_h{h}"] = float(coverage(y_arr[ts_mask], qf_ts, alpha=0.05))
            out["n_strict"] = max(out["n_strict"], int(ts_mask.sum()))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Track 2: Split Conformal (all baselines)
# ═════════════════════════════════════════════════════════════════════════════

def _conformal_quantiles(pred_test_raw: np.ndarray,
                          residuals_val: np.ndarray) -> dict:
    """Split conformal: pred + quantile(signed residuals, finite-sample corrected)."""
    N_val, H = residuals_val.shape
    out = {}
    for q in REQUIRED_QUANTILES:
        q_adj = min(1.0, q * (N_val + 1) / N_val) if q <= 0.5 \
                else max(0.0, 1 - (1 - q) * (N_val + 1) / N_val)
        offset_h = np.array([np.quantile(residuals_val[:, h], q_adj) for h in range(H)])
        out[q] = pred_test_raw + offset_h[None, :]
    return out


def conformal_nn(baseline: str, variant: str, seed: int, device: str) -> dict:
    """Conformal WIS for LSTM / Vanilla / PatchTST / DLinear (single seed)."""
    model, cfg = _load_nn_model(baseline, variant, seed, device, dropout_override=0.0)
    val_ds, df_v = _get_nn_dataset(variant, "val", cfg)
    test_ds, df_t = _get_nn_dataset(variant, "test", cfg)
    pred_val_z, y_val_z = _nn_forward_deterministic(model, val_ds, device)
    pred_test_z, y_test_z = _nn_forward_deterministic(model, test_ds, device)
    pred_val_raw = pred_val_z * TARGET_STD + TARGET_MEAN
    y_val_raw = y_val_z * TARGET_STD + TARGET_MEAN
    pred_test_raw = pred_test_z * TARGET_STD + TARGET_MEAN
    y_test_raw = y_test_z * TARGET_STD + TARGET_MEAN

    residuals = y_val_raw - pred_val_raw
    qf = _conformal_quantiles(pred_test_raw, residuals)
    eps_h1 = _get_epiweeks_h1(test_ds, df_t)
    ts_idx = _ts_split_idx(eps_h1)

    out = {"baseline": baseline, "variant": variant, "seed": seed,
           "uq_method": "conformal", "n_full": len(y_test_raw),
           "n_strict": len(ts_idx)}
    _wis_and_cov_per_h(qf, y_test_raw, "test_full", out)
    if len(ts_idx) > 0:
        qf_ts = {q: qf[q][ts_idx] for q in qf}
        _wis_and_cov_per_h(qf_ts, y_test_raw[ts_idx], "test_strict", out)
    return out


def conformal_dlinear_ensemble(variant: str, device: str) -> dict:
    """Conformal WIS for DLinear: 5-seed mean as point prediction."""
    preds_val_seeds, preds_test_seeds = [], []
    y_val_ref, y_test_ref, eps_h1_ref = None, None, None
    for seed in SEEDS:
        model, cfg = _load_nn_model("dlinear", variant, seed, device)
        val_ds, df_v = _get_nn_dataset(variant, "val", cfg)
        test_ds, df_t = _get_nn_dataset(variant, "test", cfg)
        pv, yv = _nn_forward_deterministic(model, val_ds, device)
        pt, yt = _nn_forward_deterministic(model, test_ds, device)
        preds_val_seeds.append(pv)
        preds_test_seeds.append(pt)
        if y_val_ref is None:
            y_val_ref = yv * TARGET_STD + TARGET_MEAN
            y_test_ref = yt * TARGET_STD + TARGET_MEAN
            eps_h1_ref = _get_epiweeks_h1(test_ds, df_t)
    pred_val = np.mean(preds_val_seeds, 0) * TARGET_STD + TARGET_MEAN
    pred_test = np.mean(preds_test_seeds, 0) * TARGET_STD + TARGET_MEAN
    residuals = y_val_ref - pred_val
    qf = _conformal_quantiles(pred_test, residuals)
    ts_idx = _ts_split_idx(eps_h1_ref)
    out = {"baseline": "dlinear", "variant": variant, "seed": -1,
           "uq_method": "conformal", "n_full": len(y_test_ref),
           "n_strict": len(ts_idx)}
    _wis_and_cov_per_h(qf, y_test_ref, "test_full", out)
    if len(ts_idx) > 0:
        qf_ts = {q: qf[q][ts_idx] for q in qf}
        _wis_and_cov_per_h(qf_ts, y_test_ref[ts_idx], "test_strict", out)
    return out


def conformal_cgm(variant: str, seed: int, device: str) -> dict:
    """Conformal WIS for CG-Mamba: deterministic forward, val residuals."""
    import dataclasses
    m_path = ROOT_RUNS / "cg_mamba" / f"seasons_{variant}" / f"seed{seed}" / "manifest.json"
    m = json.load(open(m_path))
    hmm_dir = _ROOT / m["hmm_dir"] if not Path(m["hmm_dir"]).is_absolute() else Path(m["hmm_dir"])
    hmm = load_fitted_hmm(hmm_dir)
    cfg = CGMambaConfig()
    model = CGForecaster(cfg).to(device)
    model.prepare_for_stage2(hmm)
    ckpt_path = _ROOT / m["stage3_best"] if not Path(m["stage3_best"]).is_absolute() else Path(m["stage3_best"])
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval()

    csv_path = _ROOT / m["csv_used"] if not Path(m["csv_used"]).is_absolute() else Path(m["csv_used"])
    df = pd.read_csv(csv_path)

    def _make_loader(split):
        ds = MultiHorizonDataset(df, split, cfg.lookback, tuple(cfg.horizons), NORM)
        return DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                          collate_fn=collate_dict), ds

    @torch.no_grad()
    def _forward_point(loader):
        mus, ys, eps_list = [], [], []
        for batch in loader:
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            preds = model(x, env)
            mus.append(preds.cpu().numpy())
            ys.append(batch["y"].cpu().numpy())
            if "target_epiweeks" in batch:
                eps_list.extend(batch["target_epiweeks"])
        return np.concatenate(mus, 0), np.concatenate(ys, 0), eps_list

    val_loader, _ = _make_loader("val")
    test_loader, _ = _make_loader("test")
    mu_val, y_val, _ = _forward_point(val_loader)
    mu_test, y_test, eps_test = _forward_point(test_loader)

    pred_val_raw = mu_val * TARGET_STD + TARGET_MEAN
    y_val_raw = y_val * TARGET_STD + TARGET_MEAN
    pred_test_raw = mu_test * TARGET_STD + TARGET_MEAN
    y_test_raw = y_test * TARGET_STD + TARGET_MEAN

    residuals = y_val_raw - pred_val_raw
    qf = _conformal_quantiles(pred_test_raw, residuals)

    eps_h1 = np.array([ep[0] if isinstance(ep, (list, np.ndarray)) else ep
                        for ep in eps_test]) if eps_test else np.zeros(len(y_test_raw))
    ts_idx = _ts_split_idx(eps_h1)

    out = {"baseline": "cg_mamba", "variant": variant, "seed": seed,
           "uq_method": "conformal", "n_full": len(y_test_raw),
           "n_strict": len(ts_idx)}
    _wis_and_cov_per_h(qf, y_test_raw, "test_full", out)
    if len(ts_idx) > 0:
        qf_ts = {q: qf[q][ts_idx] for q in qf}
        _wis_and_cov_per_h(qf_ts, y_test_raw[ts_idx], "test_strict", out)
    return out


def conformal_sarima(variant: str, device: str) -> dict:
    """Conformal WIS for SARIMA: refit → deterministic point → val residual quantiles."""
    from scipy.stats import norm as scipy_norm

    sarima_json = ROOT_RUNS / "sarima" / f"seasons_{variant}.json"
    meta = json.load(open(sarima_json))
    order = tuple(meta["selected_order"])
    seasonal_order = tuple(meta["seasonal_order"])

    csv = ROOT_RUNS / "_filtered_csvs" / f"ili_env_weekly_split_{variant}.csv"
    df = pd.read_csv(csv)

    target_col = "ili_weighted_pct"
    exog_cols = [c for c in df.columns if c not in
                 ["epiweek", "date", "split", target_col,
                  "n_stations_available", "weight_sum_raw"]]

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    y_tr = train_df[target_col].to_numpy(dtype=np.float64)
    X_tr = train_df[exog_cols].to_numpy(dtype=np.float64)
    y_va = val_df[target_col].to_numpy(dtype=np.float64)
    X_va = val_df[exog_cols].to_numpy(dtype=np.float64)
    y_te = test_df[target_col].to_numpy(dtype=np.float64)
    X_te = test_df[exog_cols].to_numpy(dtype=np.float64)
    ep_te = test_df["epiweek"].astype(int).to_numpy()

    def _rolling_point(res, y_seg, X_seg):
        """Rolling point forecast through segment, returning [N, H] arrays."""
        N = len(y_seg)
        H = max(HORIZONS)
        preds = np.full((N, len(HORIZONS)), np.nan)
        current_res = res
        for t in range(N):
            steps = min(H, N - t)
            if steps == 0:
                break
            future_exog = X_seg[t:t + steps]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = np.asarray(current_res.forecast(steps=steps, exog=future_exog),
                                dtype=np.float64)
            for h_idx, h in enumerate(HORIZONS):
                ti = t + h - 1
                if ti < N:
                    preds[ti, h_idx] = float(fc[h - 1])
            if t + 1 < N:
                new_exog = X_seg[t:t + 1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    current_res = current_res.append([y_seg[t]], exog=new_exog, refit=False)
        return preds

    # Val rolling (fit on train only)
    res_train = fit_sarimax(y_tr, X_tr, order, seasonal_order)
    val_preds = _rolling_point(res_train, y_va, X_va)

    # Test rolling (fit on train+val)
    y_tv = np.concatenate([y_tr, y_va])
    X_tv = np.vstack([X_tr, X_va])
    res_tv = fit_sarimax(y_tv, X_tv, order, seasonal_order)
    test_preds = _rolling_point(res_tv, y_te, X_te)

    # Build [N, H] arrays for val/test, drop rows with NaN
    y_val_mat = np.column_stack([y_va] * len(HORIZONS))
    y_test_mat = np.column_stack([y_te] * len(HORIZONS))

    # Per-horizon conformal: use only rows where prediction exists
    # Simplification: take residuals per-horizon, apply conformal per-horizon
    out = {"baseline": "sarima", "variant": variant, "seed": -1,
           "uq_method": "conformal", "n_full": 0, "n_strict": 0}

    for h_idx, h in enumerate(HORIZONS):
        val_valid = ~np.isnan(val_preds[:, h_idx])
        test_valid = ~np.isnan(test_preds[:, h_idx])
        if val_valid.sum() == 0 or test_valid.sum() == 0:
            continue
        residuals_h = y_va[val_valid] - val_preds[val_valid, h_idx]
        N_val_h = len(residuals_h)
        pred_test_h = test_preds[test_valid, h_idx]
        y_test_h = y_te[test_valid]
        ep_test_h = ep_te[test_valid]

        qf_h = {}
        for q in REQUIRED_QUANTILES:
            q_adj = min(1.0, q * (N_val_h + 1) / N_val_h) if q <= 0.5 \
                    else max(0.0, 1 - (1 - q) * (N_val_h + 1) / N_val_h)
            offset = np.quantile(residuals_h, q_adj)
            qf_h[q] = pred_test_h + offset

        out[f"test_full_wis_h{h}"] = float(wis(y_test_h, qf_h).mean())
        out[f"test_full_cov95_h{h}"] = float(coverage(y_test_h, qf_h, alpha=0.05))
        out["n_full"] = max(out["n_full"], len(y_test_h))

        ts_mask = ep_test_h >= TS_BOUNDARY
        if ts_mask.sum() > 0:
            qf_ts = {q: qf_h[q][ts_mask] for q in qf_h}
            out[f"test_strict_wis_h{h}"] = float(wis(y_test_h[ts_mask], qf_ts).mean())
            out[f"test_strict_cov95_h{h}"] = float(coverage(y_test_h[ts_mask], qf_ts, alpha=0.05))
            out["n_strict"] = max(out["n_strict"], int(ts_mask.sum()))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Main driver
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--track", choices=["protocol", "conformal", "both"], default="both",
                    help="Which track(s) to run")
    ap.add_argument("--baselines", nargs="+",
                    default=["sarima", "dlinear", "lstm", "vanilla_mamba",
                             "patchtst", "epideep", "cg_mamba"])
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Subset of variants (default: all 7)")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing CSV instead of overwriting")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    variants = args.variants if args.variants else VARIANTS
    baselines = args.baselines
    print(f"Device: {device}", flush=True)
    print(f"Variants: {variants}", flush=True)
    print(f"Baselines: {baselines}", flush=True)
    print(f"Track: {args.track}", flush=True)

    # ── Track 1: Protocol-specific UQ ──
    if args.track in ("protocol", "both"):
        print("\n" + "=" * 70)
        print("TRACK 1: Protocol-specific UQ")
        print("=" * 70)
        rows_p = []
        for v in variants:
            print(f"\n--- {v} ---", flush=True)
            t0 = time.time()

            if "sarima" in baselines:
                try:
                    r = protocol_sarima_kalman(v, device)
                    rows_p.append(r)
                    print(f"  ✓ sarima (kalman)    tS_wis_h1={r.get('test_strict_wis_h1','?')}", flush=True)
                except Exception as e:
                    print(f"  ✗ sarima: {e}", flush=True)

            if "dlinear" in baselines:
                try:
                    r = protocol_dlinear_ensemble(v, device)
                    rows_p.append(r)
                    print(f"  ✓ dlinear (ensemble) tS_wis_h1={r.get('test_strict_wis_h1','?'):.4f}", flush=True)
                except Exception as e:
                    print(f"  ✗ dlinear: {e}", flush=True)

            for seed in SEEDS:
                for base in ["lstm", "vanilla_mamba", "patchtst", "epideep"]:
                    if base not in baselines:
                        continue
                    try:
                        r = protocol_nn_mc(base, v, seed, device)
                        rows_p.append(r)
                    except Exception as e:
                        print(f"  ✗ {base} s={seed}: {e}", flush=True)
                        rows_p.append({"baseline": base, "variant": v,
                                        "seed": seed, "error": str(e)})

                if "cg_mamba" in baselines:
                    try:
                        r = protocol_cgm_method_f(v, seed, device)
                        rows_p.append(r)
                    except Exception as e:
                        print(f"  ✗ cg_mamba s={seed}: {e}", flush=True)
                        rows_p.append({"baseline": "cg_mamba", "variant": v,
                                        "seed": seed, "error": str(e)})

            elapsed = time.time() - t0
            n_rows_v = sum(1 for r in rows_p if r.get("variant") == v and "error" not in r)
            print(f"  [{v}] {n_rows_v} rows, {elapsed:.0f}s", flush=True)

        df_p = pd.DataFrame(rows_p)
        out_p = ROOT_RUNS / "m2_4_wis_protocol.csv"
        if args.append and out_p.exists():
            existing = pd.read_csv(out_p)
            df_p = pd.concat([existing, df_p], ignore_index=True)
        df_p.to_csv(out_p, index=False)
        print(f"\n[Track 1] Saved: {out_p}  rows={len(df_p)}", flush=True)

    # ── Track 2: Split Conformal ──
    if args.track in ("conformal", "both"):
        print("\n" + "=" * 70)
        print("TRACK 2: Split Conformal")
        print("=" * 70)
        rows_c = []
        for v in variants:
            print(f"\n--- {v} ---", flush=True)
            t0 = time.time()

            if "sarima" in baselines:
                try:
                    r = conformal_sarima(v, device)
                    rows_c.append(r)
                    print(f"  ✓ sarima (conformal) tS_wis_h1={r.get('test_strict_wis_h1','?')}", flush=True)
                except Exception as e:
                    print(f"  ✗ sarima conformal: {e}", flush=True)

            if "dlinear" in baselines:
                try:
                    r = conformal_dlinear_ensemble(v, device)
                    rows_c.append(r)
                    print(f"  ✓ dlinear (conformal) tS_wis_h1={r.get('test_strict_wis_h1','?'):.4f}", flush=True)
                except Exception as e:
                    print(f"  ✗ dlinear conformal: {e}", flush=True)

            for seed in SEEDS:
                for base in ["lstm", "vanilla_mamba", "patchtst", "epideep"]:
                    if base not in baselines:
                        continue
                    try:
                        r = conformal_nn(base, v, seed, device)
                        rows_c.append(r)
                    except Exception as e:
                        print(f"  ✗ {base} conformal s={seed}: {e}", flush=True)

                if "cg_mamba" in baselines:
                    try:
                        r = conformal_cgm(v, seed, device)
                        rows_c.append(r)
                    except Exception as e:
                        print(f"  ✗ cg_mamba conformal s={seed}: {e}", flush=True)

            elapsed = time.time() - t0
            n_rows_v = sum(1 for r in rows_c if r.get("variant") == v)
            print(f"  [{v}] {n_rows_v} rows, {elapsed:.0f}s", flush=True)

        df_c = pd.DataFrame(rows_c)
        out_c = ROOT_RUNS / "m2_4_wis_conformal.csv"
        if args.append and out_c.exists():
            existing = pd.read_csv(out_c)
            df_c = pd.concat([existing, df_c], ignore_index=True)
        df_c.to_csv(out_c, index=False)
        print(f"\n[Track 2] Saved: {out_c}  rows={len(df_c)}", flush=True)


if __name__ == "__main__":
    main()
