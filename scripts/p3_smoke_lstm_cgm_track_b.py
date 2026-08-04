"""Track B LOCKED smoke — LSTM + CGM_APMD under uniform Split-Conformal CQR.

Reference LOCK: paper/track_b_sub_pre_registration.md (LOCKED 2026-06-21).
Parent LOCK:    project_cgmamba_pc012_locked (2026-06-12 v2).

Purpose
-------
This is the *smoke* pipeline test for Track B (uniform CQR), exercising hard-stop
gate (a) [smoke inversion / NS]. Per LOCK §6(a), an inversion / NS result is NOT
a STOP — it is logged + reported as the verdict and the full evaluation proceeds.

Pipeline
--------
1. Audit ckpt path verification (LOCK hard-stop (c)) — STOP if mismatch.
2. Build national in-sample residual pool:
     - LSTM:  100 MC-Dropout forward passes over val split → empirical quantiles
              at FluSight 23 taus on national val series.
     - CGM:   single deterministic forward over val split → Gaussian quantiles
              from (mu, sigma2_total = within + between).
3. Build regional test_strict (epiweek >= 202240) predictions, per region:
     - LSTM:  100 MC-Dropout forward passes → empirical quantiles at 23 taus.
     - CGM:   deterministic forward → Gaussian quantiles from (mu, sigma2_total).
4. Apply uniform CQR-symmetric Split Conformal via
   `src.eval.wis_standard.quantiles_conformal_cqr` per (baseline, region, horizon)
   using the national val pool. Single routine, no per-baseline wrapper (LOCK §3).
5. Native + Track B WIS / Cov95 via Bracher 2021 (`src.eval.wis_standard.wis,
   coverage`) per (baseline, region, horizon).
6. Aggregation order = per-region per-horizon -> mean over regions -> mean over
   horizons (LOCK §5).
7. Emit hard-stop gate (a) verdict + per-horizon Cov95/WIS for both baselines,
   native vs Track B.

Outputs
-------
- runs/track_b_smoke/lstm_cgm_track_b_results.json   (native + track_b + per_horizon)
- runs/track_b_smoke/lstm_cgm_track_b_per_cell.parquet (per region x horizon x baseline)

Hard-stop enforcement
---------------------
(a) Smoke inversion / NS: PRINT + EXIT 0 with verdict; PROCEED to full eval.
(b) NaN / inf in conformal radius: STOP, exit 1.
(c) ckpt path mismatch vs audit-PASS Section IV.2: STOP, exit 1.
(d) Per-region Cov95 outside [0.5, 1.0]: STOP, exit 1.

Usage
-----
    python scripts/p3_smoke_lstm_cgm_track_b.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# ---- Single source of truth: src/eval/wis_standard ------------------------
from src.eval.wis_standard import (
    REQUIRED_QUANTILES,
    ALPHA_LEVELS,
    INTERVAL_PAIRS,
    FLUSIGHT_23,
    wis,
    coverage,
    quantiles_from_gaussian,
    quantiles_from_samples,
    quantiles_conformal_cqr,
)

from src.data.loader import load_norm_params, WeeklyDataset
from src.utils.config import CGMambaConfig
from src.utils.checkpoints import load_fitted_hmm
from src.models.cg_forecaster import CGForecaster
from src.baselines.lstm import WeeklyMultiHorizonDataset
from cm_mamba.baselines.lstm_baseline import LSTMForecaster
from src.eval.hmm_interval import compute_decomposition
from src.eval.quantile_predictions import _dropout_train_mode
from scripts.phase_3_region_eval import build_region_df


# ============================================================================
# LOCK constants (LOCK §4 base-quantile; LOCK §10 frozen constants)
# ============================================================================
HORIZONS = (1, 2, 3, 4)
TS_BOUNDARY = 202240
REGIONS = tuple(f"hhs{i}" for i in range(1, 11))
SEEDS = (42, 123, 456, 789, 1024)

# Section IV.2 audit-PASS ckpt locations
LSTM_RUN_DIR = _ROOT / "runs" / "lstm_final" / "h256_l2_lr5e-04_bs16"
LSTM_SEED_FOR_SMOKE = 42  # smoke = single-seed LSTM dry-run
LSTM_DROPOUT_MC = 0.3      # LOCK + audit: MC-Dropout d=0.3 at eval

CGM_MANIFEST_DIR = (
    _ROOT / "runs" / "m2_4_data_efficiency" / "cg_mamba" / "seasons_17_seasons_full"
)
CGM_SEED_FOR_SMOKE = 42

N_MC_SAMPLES = 100  # LOCK §10 n_samples_MC

# Source of normalization (parent-lock national-train scaler)
NORM_PATH = _ROOT / "data" / "processed" / "normalization_params.json"
SPLIT_CSV = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"

# Output
OUT_DIR = _ROOT / "runs" / "track_b_smoke"
OUT_JSON = OUT_DIR / "lstm_cgm_track_b_results.json"
OUT_PARQUET = OUT_DIR / "lstm_cgm_track_b_per_cell.parquet"

COV95_BAND = (0.5, 1.0)  # LOCK hard-stop (d)


# ============================================================================
# Hard-stop (c) — ckpt audit
# ============================================================================
def verify_audit_ckpts() -> dict[str, str]:
    """LOCK hard-stop (c): byte-existence audit of Section IV.2 ckpts.

    No retraining, no "closest available". Just verifies the canonical paths
    that produced Table IV (and phase_3 regional CSVs) exist.
    """
    paths = {}
    # LSTM
    p_lstm_results = LSTM_RUN_DIR / f"seed{LSTM_SEED_FOR_SMOKE}" / "results.json"
    p_lstm_ckpt = LSTM_RUN_DIR / f"seed{LSTM_SEED_FOR_SMOKE}" / "lstm_best.pt"
    if not p_lstm_results.exists() or not p_lstm_ckpt.exists():
        print(f"[HARD-STOP c] LSTM Section IV.2 ckpt missing:\n  {p_lstm_results}\n  {p_lstm_ckpt}",
              flush=True)
        sys.exit(1)
    paths["lstm_results_json"] = str(p_lstm_results)
    paths["lstm_ckpt"] = str(p_lstm_ckpt)

    # CGM_APMD
    p_cgm_manifest = CGM_MANIFEST_DIR / f"seed{CGM_SEED_FOR_SMOKE}" / "manifest.json"
    if not p_cgm_manifest.exists():
        print(f"[HARD-STOP c] CGM manifest missing: {p_cgm_manifest}", flush=True)
        sys.exit(1)
    manifest = json.load(open(p_cgm_manifest))
    p_cgm_stage3 = _ROOT / manifest["stage3_best"]
    p_cgm_hmm = _ROOT / manifest["hmm_dir"]
    if not p_cgm_stage3.exists():
        print(f"[HARD-STOP c] CGM stage3 ckpt missing: {p_cgm_stage3}", flush=True)
        sys.exit(1)
    if not p_cgm_hmm.exists():
        print(f"[HARD-STOP c] CGM HMM dir missing: {p_cgm_hmm}", flush=True)
        sys.exit(1)
    paths["cgm_manifest"] = str(p_cgm_manifest)
    paths["cgm_stage3_best"] = str(p_cgm_stage3)
    paths["cgm_hmm_dir"] = str(p_cgm_hmm)
    return paths


# ============================================================================
# Data + window helpers
# ============================================================================
def _load_norm():
    return load_norm_params(NORM_PATH)


def _national_df() -> pd.DataFrame:
    """National ili_env_weekly_split CSV used for in-sample residual pool."""
    df = pd.read_csv(SPLIT_CSV)
    return df


# ----------- LSTM I/O -----------
def _load_lstm_model(device: str, dropout_eval: float) -> tuple[torch.nn.Module, dict]:
    p = LSTM_RUN_DIR / f"seed{LSTM_SEED_FOR_SMOKE}"
    cfg = json.load(open(p / "results.json"))["config"]
    sd = torch.load(p / "lstm_best.pt", map_location=device, weights_only=True)
    model = LSTMForecaster(
        enc_in=cfg["enc_in"], hidden=cfg["hidden"],
        num_layers=cfg["num_layers"], pred_len=cfg["pred_len"],
        dropout=dropout_eval,
    )
    model.load_state_dict(sd)
    # Force MC-Dropout rate per LOCK + audit (d=0.3)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout_eval
        elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            m.dropout = dropout_eval
    return model.to(device), cfg


def _lstm_mc_samples(
    model: torch.nn.Module, ds, device: str, norm: dict,
    n_samples: int = N_MC_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (samples_raw [S,N,H], y_raw [N,H], eps_h1 [N])."""
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    all_samples = []
    y_collect = None
    with _dropout_train_mode(model):
        with torch.no_grad():
            for _ in range(n_samples):
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
    samples = np.stack(all_samples, axis=0) * target_std + target_mean  # [S,N,H]
    y_raw = y_collect * target_std + target_mean
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    return samples, y_raw, eps_h1


# ----------- CGM I/O -----------
def _load_cgm_model(device: str) -> tuple[torch.nn.Module, CGMambaConfig, object]:
    p_manifest = CGM_MANIFEST_DIR / f"seed{CGM_SEED_FOR_SMOKE}" / "manifest.json"
    m = json.load(open(p_manifest))
    hmm = load_fitted_hmm(Path(m["hmm_dir"]))
    cfg = CGMambaConfig()
    model = CGForecaster(cfg)
    model.prepare_for_stage2(hmm)
    ck = torch.load(Path(m["stage3_best"]), map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model, cfg, hmm


def _cgm_decomp_forward(
    model: torch.nn.Module, cfg: CGMambaConfig, hmm, ds, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward CGM deterministically; return (mu_z, sigma2_total_z, y_z, eps_h1).

    All arrays in z-scored space. The Gaussian base quantile is built in z-space
    then converted to raw scale outside (same routine for val & test).
    """
    # HMM emission stats (z-scored ILI dim = idx 0 of V_aug)
    means = hmm.means
    covars = hmm.covars
    mu_k_ili = means[:, 0]
    if covars.ndim == 3:
        sigma2_k_ili = np.array([covars[k, 0, 0] for k in range(covars.shape[0])])
    else:
        sigma2_k_ili = covars[:, 0]

    n = len(ds)
    if n == 0:
        return (np.zeros((0, 4)), np.zeros((0, 4)), np.zeros((0, 4)),
                np.zeros((0,), dtype=np.int64))
    eps_arr = ds.df["epiweek"].astype(int).to_numpy()
    target_mean = float(ds.norm["ili_weighted_pct"]["mean"]) if hasattr(ds, "norm") else None
    # We will get y from ds.df directly (z-space)
    norm = load_norm_params(NORM_PATH)
    ili_p = norm["ili_weighted_pct"]
    target_z_full = (ds.df["ili_weighted_pct"].to_numpy() - ili_p["mean"]) / ili_p["std"]

    mu = np.zeros((n, 4))
    gamma_all = np.zeros((n, 4, 3))
    y_z = np.zeros((n, 4))
    eps_h1 = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)

    with torch.no_grad():
        for i in range(n):
            d = ds[i]
            x = d["x"].unsqueeze(0).to(device)
            env = d["env"].unsqueeze(0).to(device)
            pred, inter = model(x, env, return_intermediates=True)
            if torch.isnan(pred).any():
                valid[i] = False
                continue
            mu[i] = pred[0].cpu().numpy()
            gamma_all[i] = inter["gamma_all"][0].cpu().numpy()
            tgt_ep = int(d["target_epiweek"])
            tgt_idx = int(np.where(eps_arr == tgt_ep)[0][0])
            for h_idx, h in enumerate(HORIZONS):
                src = tgt_idx - (max(HORIZONS) - h)
                if 0 <= src < len(eps_arr):
                    y_z[i, h_idx] = target_z_full[src]
            eps_h1[i] = eps_arr[tgt_idx - (max(HORIZONS) - 1)]

    mu = mu[valid]
    gamma_all = gamma_all[valid]
    y_z = y_z[valid]
    eps_h1 = eps_h1[valid]
    decomp = compute_decomposition(mu, gamma_all, mu_k_ili, sigma2_k_ili)
    return decomp.mu_CGM, decomp.sigma2_total, y_z, eps_h1


# ============================================================================
# National val residual / base-quantile pool (LOCK §2 — option (a))
# ============================================================================
def build_lstm_val_base_quantiles(
    model, device, norm,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """National val LSTM base quantiles in RAW scale + y_val_raw [N_val, H]."""
    df = _national_df()
    val_ds = WeeklyMultiHorizonDataset(
        df, "val", norm, lookback=104, pred_len=max(HORIZONS),
    )
    samples, y_val_raw, _ = _lstm_mc_samples(model, val_ds, device, norm)
    # samples: [S, N, H]. Build empirical quantiles per (N, H).
    qf = quantiles_from_samples(np.transpose(samples, (1, 2, 0)), taus=FLUSIGHT_23, axis=-1)
    # qf keys: tau -> [N, H]
    return qf, y_val_raw


def build_cgm_val_base_quantiles(
    model, cfg, hmm, device, norm,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """National val CGM base quantiles (Gaussian from mu, sigma2_total)."""
    df = _national_df()
    val_ds = WeeklyDataset(
        df, split="val", lookback=cfg.lookback,
        horizon=max(cfg.horizons), norm=norm,
    )
    mu_z, sig2_z, y_z, _ = _cgm_decomp_forward(model, cfg, hmm, val_ds, device)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    mu_raw = mu_z * target_std + target_mean
    sig2_raw = sig2_z * (target_std ** 2)
    y_val_raw = y_z * target_std + target_mean
    qf = quantiles_from_gaussian(mu_raw, sig2_raw, taus=FLUSIGHT_23)
    # quantiles_from_gaussian on shape [N,H] returns dict tau -> [N,H]
    return qf, y_val_raw


# ============================================================================
# Regional test_strict base quantiles
# ============================================================================
def build_lstm_region_test_quantiles(
    model, device, norm, region: str,
) -> tuple[dict[float, np.ndarray], np.ndarray, np.ndarray]:
    """Returns (qf_test [tau -> N_strict, H], y_test_strict_raw [N_strict, H], eps_h1)."""
    region_df = build_region_df(region)
    cfg_full = json.load(open(LSTM_RUN_DIR / f"seed{LSTM_SEED_FOR_SMOKE}" / "results.json"))["config"]
    test_ds = WeeklyMultiHorizonDataset(
        region_df, "test", norm,
        lookback=cfg_full["lookback"], pred_len=cfg_full["pred_len"],
    )
    samples, y_raw, eps_h1 = _lstm_mc_samples(model, test_ds, device, norm)
    ts_idx = np.where(eps_h1 >= TS_BOUNDARY)[0]
    if len(ts_idx) == 0:
        raise RuntimeError(f"LSTM region={region}: empty test_strict")
    samples_ts = samples[:, ts_idx, :]
    y_ts = y_raw[ts_idx]
    qf = quantiles_from_samples(np.transpose(samples_ts, (1, 2, 0)), taus=FLUSIGHT_23, axis=-1)
    return qf, y_ts, eps_h1[ts_idx]


def build_cgm_region_test_quantiles(
    model, cfg, hmm, device, norm, region: str,
) -> tuple[dict[float, np.ndarray], np.ndarray, np.ndarray]:
    region_df = build_region_df(region)
    test_ds = WeeklyDataset(
        region_df, split="test", lookback=cfg.lookback,
        horizon=max(cfg.horizons), norm=norm,
    )
    mu_z, sig2_z, y_z, eps_h1 = _cgm_decomp_forward(model, cfg, hmm, test_ds, device)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    mu_raw = mu_z * target_std + target_mean
    sig2_raw = sig2_z * (target_std ** 2)
    y_raw = y_z * target_std + target_mean
    ts_idx = np.where(eps_h1 >= TS_BOUNDARY)[0]
    if len(ts_idx) == 0:
        raise RuntimeError(f"CGM region={region}: empty test_strict")
    mu_ts = mu_raw[ts_idx]
    sig2_ts = sig2_raw[ts_idx]
    y_ts = y_raw[ts_idx]
    qf = quantiles_from_gaussian(mu_ts, sig2_ts, taus=FLUSIGHT_23)
    return qf, y_ts, eps_h1[ts_idx]


# ============================================================================
# CQR conformal with HARD-STOP (b) [NaN/inf radius] inline
# ============================================================================
def conformal_cqr_with_check(
    base_val: dict[float, np.ndarray], base_test: dict[float, np.ndarray],
    y_val: np.ndarray, label: str, h_idx: int,
) -> dict[float, np.ndarray]:
    """Apply CQR-symmetric Split Conformal per horizon h_idx.

    LOCK §3: single routine = quantiles_conformal_cqr. We slice val + test to
    horizon h_idx (1-D) for the routine, then return per-horizon calibrated qf.
    Hard-stop (b) checks per-tau radius implicitly: if y_val contains nan/inf
    or any tau yields a non-finite radius, STOP.
    """
    val_h = {float(t): np.asarray(base_val[t][:, h_idx]) for t in FLUSIGHT_23}
    test_h = {float(t): np.asarray(base_test[t][:, h_idx]) for t in FLUSIGHT_23}
    y_h = np.asarray(y_val[:, h_idx], dtype=np.float64)

    # Pre-check
    for t in FLUSIGHT_23:
        if not (np.all(np.isfinite(val_h[float(t)])) and
                np.all(np.isfinite(test_h[float(t)]))):
            print(f"[HARD-STOP b] {label} h={h_idx+1}: non-finite base quantile at tau={t}",
                  flush=True)
            sys.exit(1)
    if not np.all(np.isfinite(y_h)):
        print(f"[HARD-STOP b] {label} h={h_idx+1}: non-finite y_val", flush=True)
        sys.exit(1)

    out = quantiles_conformal_cqr(
        base_quantiles_val=val_h,
        base_quantiles_test=test_h,
        y_val=y_h,
        taus=FLUSIGHT_23,
    )
    # Post-check
    for t in FLUSIGHT_23:
        v = out[float(t)]
        if not np.all(np.isfinite(v)):
            print(f"[HARD-STOP b] {label} h={h_idx+1}: non-finite calibrated qf at tau={t}",
                  flush=True)
            sys.exit(1)
    return out


# ============================================================================
# Scoring (LOCK §5 Bracher 2021 + aggregation)
# ============================================================================
def score_per_cell(
    qf: dict[float, np.ndarray], y: np.ndarray,
) -> tuple[float, float]:
    """Return (wis_mean, cov95) over the (region, horizon) cell."""
    wis_vals = wis(y, qf)
    cov95 = coverage(y, qf, alpha=0.05)
    return float(np.mean(wis_vals)), float(cov95)


# ============================================================================
# Main
# ============================================================================
def main(device: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not torch.cuda.is_available():
        device = "cpu"
    print(f"[smoke] device={device}", flush=True)

    # ----- HARD-STOP (c): ckpt audit -----
    ckpt_paths = verify_audit_ckpts()
    print("[audit] Section IV.2 ckpts verified:", flush=True)
    for k, v in ckpt_paths.items():
        print(f"  {k}: {v}", flush=True)

    norm = _load_norm()

    # ----- Models -----
    print("[load] LSTM (seed=42, dropout_eval=0.3)...", flush=True)
    lstm_model, _ = _load_lstm_model(device, dropout_eval=LSTM_DROPOUT_MC)
    print("[load] CGM_APMD (seed=42, manifest path)...", flush=True)
    cgm_model, cgm_cfg, cgm_hmm = _load_cgm_model(device)

    # ----- National val base quantiles (residual pool for CQR) -----
    print("[val] LSTM national MC-Dropout (n=100) over val...", flush=True)
    lstm_val_qf, lstm_y_val = build_lstm_val_base_quantiles(lstm_model, device, norm)
    print(f"  n_val={lstm_y_val.shape[0]}  H={lstm_y_val.shape[1]}", flush=True)
    print("[val] CGM_APMD national Gaussian over val...", flush=True)
    cgm_val_qf, cgm_y_val = build_cgm_val_base_quantiles(cgm_model, cgm_cfg, cgm_hmm, device, norm)
    print(f"  n_val={cgm_y_val.shape[0]}  H={cgm_y_val.shape[1]}", flush=True)

    # n_cal<10 void check (parent LOCK rule)
    n_cal_lstm = lstm_y_val.shape[0]
    n_cal_cgm = cgm_y_val.shape[0]
    if n_cal_lstm < 10 or n_cal_cgm < 10:
        print(f"[HARD-STOP n_cal] LSTM n_val={n_cal_lstm}, CGM n_val={n_cal_cgm} — parent LOCK void rule",
              flush=True)
        sys.exit(1)

    # ----- Per-region scoring -----
    per_cell_rows = []
    cov_band_violations: list[dict] = []
    for region in REGIONS:
        print(f"\n=== region={region} ===", flush=True)
        # Native + Track B: LSTM
        try:
            lstm_test_qf, lstm_y_test, _ = build_lstm_region_test_quantiles(
                lstm_model, device, norm, region,
            )
        except Exception as e:
            print(f"  [LSTM {region}] forward error: {type(e).__name__}: {e}", flush=True)
            raise
        # Native + Track B: CGM
        try:
            cgm_test_qf, cgm_y_test, _ = build_cgm_region_test_quantiles(
                cgm_model, cgm_cfg, cgm_hmm, device, norm, region,
            )
        except Exception as e:
            print(f"  [CGM {region}] forward error: {type(e).__name__}: {e}", flush=True)
            raise

        for h_idx, h in enumerate(HORIZONS):
            # ----- Native scoring (Table IV reproduction sanity) -----
            lstm_native_qf_h = {float(t): lstm_test_qf[float(t)][:, h_idx] for t in FLUSIGHT_23}
            cgm_native_qf_h = {float(t): cgm_test_qf[float(t)][:, h_idx] for t in FLUSIGHT_23}
            y_lstm_h = lstm_y_test[:, h_idx]
            y_cgm_h = cgm_y_test[:, h_idx]
            lstm_wis_native, lstm_cov_native = score_per_cell(lstm_native_qf_h, y_lstm_h)
            cgm_wis_native, cgm_cov_native = score_per_cell(cgm_native_qf_h, y_cgm_h)

            # ----- Track B (uniform CQR) -----
            lstm_cqr_qf = conformal_cqr_with_check(
                base_val=lstm_val_qf, base_test=lstm_test_qf,
                y_val=lstm_y_val, label=f"LSTM/{region}", h_idx=h_idx,
            )
            cgm_cqr_qf = conformal_cqr_with_check(
                base_val=cgm_val_qf, base_test=cgm_test_qf,
                y_val=cgm_y_val, label=f"CGM/{region}", h_idx=h_idx,
            )
            lstm_wis_tb, lstm_cov_tb = score_per_cell(lstm_cqr_qf, y_lstm_h)
            cgm_wis_tb, cgm_cov_tb = score_per_cell(cgm_cqr_qf, y_cgm_h)

            # ----- HARD-STOP (d): Cov95 in [0.5, 1.0] -----
            for tag, cov_v in [
                ("LSTM_native", lstm_cov_native), ("CGM_native", cgm_cov_native),
                ("LSTM_track_b", lstm_cov_tb),   ("CGM_track_b", cgm_cov_tb),
            ]:
                if not (COV95_BAND[0] <= cov_v <= COV95_BAND[1]):
                    cov_band_violations.append({
                        "region": region, "h": h, "tag": tag, "cov95": cov_v,
                    })

            per_cell_rows.append({
                "region": region, "h": h,
                "n_test_strict": int(len(y_lstm_h)),
                "lstm_wis_native": lstm_wis_native,
                "lstm_cov95_native": lstm_cov_native,
                "cgm_wis_native": cgm_wis_native,
                "cgm_cov95_native": cgm_cov_native,
                "lstm_wis_track_b": lstm_wis_tb,
                "lstm_cov95_track_b": lstm_cov_tb,
                "cgm_wis_track_b": cgm_wis_tb,
                "cgm_cov95_track_b": cgm_cov_tb,
            })
            print(
                f"  h={h}  n={len(y_lstm_h):3d}  "
                f"LSTM[nat WIS={lstm_wis_native:.4f} Cov95={lstm_cov_native:.3f}]  "
                f"[TrkB WIS={lstm_wis_tb:.4f} Cov95={lstm_cov_tb:.3f}]  | "
                f"CGM[nat WIS={cgm_wis_native:.4f} Cov95={cgm_cov_native:.3f}]  "
                f"[TrkB WIS={cgm_wis_tb:.4f} Cov95={cgm_cov_tb:.3f}]",
                flush=True,
            )

    df_cells = pd.DataFrame(per_cell_rows)
    df_cells.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[save] per-cell parquet: {OUT_PARQUET}", flush=True)

    # ----- HARD-STOP (d) final adjudication -----
    if cov_band_violations:
        print(f"[HARD-STOP d] Per-region Cov95 outside [0.5, 1.0]: "
              f"{len(cov_band_violations)} cell(s):", flush=True)
        for v in cov_band_violations:
            print(f"  region={v['region']} h={v['h']} {v['tag']} cov95={v['cov95']:.4f}",
                  flush=True)
        # Still write partial JSON for debug, then exit 1
        partial = {
            "status": "HARD_STOP_D",
            "violations": cov_band_violations,
            "per_cell_parquet": str(OUT_PARQUET),
        }
        OUT_JSON.write_text(json.dumps(partial, indent=2))
        sys.exit(1)

    # ----- Aggregation: per-region per-horizon -> mean over regions -> mean over horizons -----
    def _agg(metric: str) -> dict:
        per_h = {}
        for h in HORIZONS:
            sub = df_cells[df_cells.h == h]
            per_h[f"h{h}"] = float(sub[metric].mean())
        avg = float(np.mean([per_h[f"h{h}"] for h in HORIZONS]))
        return {"per_horizon": per_h, "avg_over_horizons": avg}

    native_block = {
        "lstm_wis": _agg("lstm_wis_native"),
        "lstm_cov95": _agg("lstm_cov95_native"),
        "cgm_wis": _agg("cgm_wis_native"),
        "cgm_cov95": _agg("cgm_cov95_native"),
    }
    trackb_block = {
        "lstm_wis": _agg("lstm_wis_track_b"),
        "lstm_cov95": _agg("lstm_cov95_track_b"),
        "cgm_wis": _agg("cgm_wis_track_b"),
        "cgm_cov95": _agg("cgm_cov95_track_b"),
    }

    # ----- F3 horizon-collapse: WIS(h=4) - WIS(h=1) -----
    f3_lstm_native = (
        native_block["lstm_wis"]["per_horizon"]["h4"]
        - native_block["lstm_wis"]["per_horizon"]["h1"]
    )
    f3_lstm_trackb = (
        trackb_block["lstm_wis"]["per_horizon"]["h4"]
        - trackb_block["lstm_wis"]["per_horizon"]["h1"]
    )

    # ----- Hard-stop (a) gate: CGM lead = LSTM_WIS - CGM_WIS  (positive = CGM wins) -----
    cgm_lead_native = (
        native_block["lstm_wis"]["avg_over_horizons"]
        - native_block["cgm_wis"]["avg_over_horizons"]
    )
    cgm_lead_trackb = (
        trackb_block["lstm_wis"]["avg_over_horizons"]
        - trackb_block["cgm_wis"]["avg_over_horizons"]
    )

    inversion = (cgm_lead_native > 0) and (cgm_lead_trackb < 0)
    # NS = lead became near-zero (within smoke tolerance ±0.02 WIS) after Track B
    ns = (cgm_lead_native > 0.02) and (abs(cgm_lead_trackb) <= 0.02)

    if inversion:
        verdict = "INVERSION_PROCEED"
    elif ns:
        verdict = "NS_PROCEED"
    elif cgm_lead_trackb > 0.02:
        verdict = "AS-IS PROCEED"
    else:
        verdict = "AS-IS PROCEED"  # default fallback (LOCK §6(a): always proceed)

    hard_stop_a_gate = {
        "cgm_lead_native_wis": cgm_lead_native,
        "cgm_lead_track_b_wis": cgm_lead_trackb,
        "inversion": bool(inversion),
        "ns": bool(ns),
        "verdict": verdict,
        "f3_lstm_horizon_collapse_native": f3_lstm_native,
        "f3_lstm_horizon_collapse_track_b": f3_lstm_trackb,
    }

    # ----- Final JSON -----
    payload = {
        "status": "OK",
        "lock_reference": "paper/track_b_sub_pre_registration.md (LOCKED 2026-06-21)",
        "parent_lock": "project_cgmamba_pc012_locked (2026-06-12 v2)",
        "smoke_target": "LSTM + CGM_APMD (seed=42), uniform Split-Conformal CQR",
        "ckpt_paths": ckpt_paths,
        "lstm_dropout_eval": LSTM_DROPOUT_MC,
        "n_mc_samples": N_MC_SAMPLES,
        "n_cal_lstm": int(n_cal_lstm),
        "n_cal_cgm": int(n_cal_cgm),
        "n_regions": len(REGIONS),
        "horizons": list(HORIZONS),
        "ts_boundary_epiweek": TS_BOUNDARY,
        "native": native_block,
        "track_b": trackb_block,
        "per_horizon_summary": {
            f"h{h}": {
                "lstm_native_wis": native_block["lstm_wis"]["per_horizon"][f"h{h}"],
                "lstm_native_cov95": native_block["lstm_cov95"]["per_horizon"][f"h{h}"],
                "cgm_native_wis": native_block["cgm_wis"]["per_horizon"][f"h{h}"],
                "cgm_native_cov95": native_block["cgm_cov95"]["per_horizon"][f"h{h}"],
                "lstm_track_b_wis": trackb_block["lstm_wis"]["per_horizon"][f"h{h}"],
                "lstm_track_b_cov95": trackb_block["lstm_cov95"]["per_horizon"][f"h{h}"],
                "cgm_track_b_wis": trackb_block["cgm_wis"]["per_horizon"][f"h{h}"],
                "cgm_track_b_cov95": trackb_block["cgm_cov95"]["per_horizon"][f"h{h}"],
            } for h in HORIZONS
        },
        "hard_stop_a_gate": hard_stop_a_gate,
        "elapsed_sec": time.time() - t0,
        "per_cell_parquet": str(OUT_PARQUET),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[save] results JSON: {OUT_JSON}", flush=True)

    # ----- Stdout summary -----
    print("\n" + "=" * 80, flush=True)
    print("Track B smoke SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Native:   LSTM WIS={native_block['lstm_wis']['avg_over_horizons']:.4f}  "
          f"CGM WIS={native_block['cgm_wis']['avg_over_horizons']:.4f}  "
          f"CGM lead={cgm_lead_native:+.4f}", flush=True)
    print(f"Track B:  LSTM WIS={trackb_block['lstm_wis']['avg_over_horizons']:.4f}  "
          f"CGM WIS={trackb_block['cgm_wis']['avg_over_horizons']:.4f}  "
          f"CGM lead={cgm_lead_trackb:+.4f}", flush=True)
    print(f"Inversion: {inversion}   NS: {ns}", flush=True)
    print(f"F3 horizon-collapse (LSTM, h4-h1 WIS): native={f3_lstm_native:+.4f}  "
          f"track_b={f3_lstm_trackb:+.4f}", flush=True)
    print(f"Verdict (LOCK §6(a) — STOP=never, PROCEED always): {verdict}", flush=True)
    print(f"Hard-stop gates: c=PASS, b=PASS, d=PASS, a=PROCEED (no STOP)", flush=True)
    print(f"Elapsed: {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    sys.exit(main(args.device))
