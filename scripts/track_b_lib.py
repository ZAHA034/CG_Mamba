"""scripts/track_b_lib.py — Track B forward library (LOCK §4 base UQ)

Generalized forward functions for full 5-seed × 6-baseline Track B run.
Per LOCK (paper/track_b_sub_pre_registration.md, 2026-06-21):
  - 4 NN baselines (LSTM, VM, PatchTST, EpiDeep): empirical from n=100 MC Dropout
  - CGM (APMD): Gaussian PI from (μ, σ²_total) via HMM phase decomposition (raw, no Method F)
  - DLinear: Gaussian PI from 5-seed ensemble (mean, std) — no MC, deterministic forward

Constructor signatures + ckpt paths sourced from disk-verified evidence:
  - scripts/phase_3_region_eval.py:109-148 (LSTM, VM, PatchTST, CGM)
  - scripts/phase_3_region_eval_extras.py:115-141 (DLinear, EpiDeep)
  - scripts/phase_3_dlinear_ensemble_region.py:60-85 (DLinear 5-seed ensemble)
  - scripts/p3_smoke_lstm_cgm_track_b.py (LSTM + CGM smoke, seed42 — validated)

Used by:
  - scripts/p3_full_track_b_run.py (main full-run dispatcher, wired next turn)
  - scripts/p3_integration_test.py (per-baseline native = Table IV cell verification)

This module does NOT depend on smoke script (avoids module-level constant mutation).
Each function takes `seed` as an explicit parameter.
"""
from __future__ import annotations
import json, sys, warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# Time-Series-Library on sys.path BEFORE patchtst import
_TSLIB = _ROOT.parent / "Time-Series-Library"
if _TSLIB.exists():
    sys.path.insert(0, str(_TSLIB))
    import models  # noqa — force-cache Time-Series-Library/models

from src.eval.wis_standard import (
    FLUSIGHT_23,
    quantiles_from_gaussian,
    quantiles_from_samples,
    quantiles_conformal_cqr,
    wis,
    coverage,
)
from src.data.loader import load_norm_params, WeeklyDataset
from src.utils.config import CGMambaConfig
from src.utils.checkpoints import load_fitted_hmm
from src.models.cg_forecaster import CGForecaster
from src.eval.hmm_interval import compute_decomposition

# Dataset classes
from baselines.lstm import WeeklyMultiHorizonDataset

# Forecaster classes (lazy imports for PatchTST/EpiDeep/etc. to avoid heavy Time-Series-Library cold start)
from cm_mamba.baselines.lstm_baseline import LSTMForecaster
from baselines.vanilla_mamba import VanillaMambaForecaster

# build_region_df: inline copy from scripts.phase_3_region_eval (avoid PatchTST import chain).
# Identical logic; lazy PatchTST import is handled inside _build_patchtst.
def build_region_df(region: str) -> pd.DataFrame:
    """Inline copy of scripts.phase_3_region_eval.build_region_df (avoid patchtst import)."""
    from epiweeks import Week
    region_csv = _ROOT / "data" / "raw" / "cdc_ilinet" / "_phase3_phase6_fetch" / f"{region}_full.csv"
    df_r = pd.read_csv(region_csv)
    df_r["epiweek"] = df_r["year"].astype(int) * 100 + df_r["week"].astype(int)
    df_r["date"] = df_r.apply(lambda r: Week(int(r["year"]), int(r["week"])).startdate().isoformat(), axis=1)
    env = pd.read_csv(_ROOT / "data" / "processed" / "env_national_weekly.csv")
    df_merged = df_r.merge(env[["epiweek", "temperature_c", "specific_humidity_g_per_kg"]],
                            on="epiweek", how="inner")
    split = pd.read_csv(_ROOT / "data" / "processed" / "ili_env_weekly_split.csv")
    df_merged = df_merged.merge(split[["epiweek", "split"]], on="epiweek", how="inner")
    df_merged["n_stations_available"] = 10
    df_merged["weight_sum_raw"] = 1.0
    return df_merged


# ============================================================================
# LOCK constants
# ============================================================================
HORIZONS = (1, 2, 3, 4)
TS_BOUNDARY = 202240
N_MC_SAMPLES = 100
DROPOUT_MC = {"lstm": 0.3, "vanilla_mamba": 0.1, "patchtst": 0.1, "epideep": 0.1}
NORM_PATH = _ROOT / "data" / "processed" / "normalization_params.json"
SPLIT_CSV = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
SEEDS = (42, 123, 456, 789, 1024)

# Per-baseline run directories (disk-evidence locked 2026-06-21)
RUN_DIRS = {
    "lstm":          _ROOT / "runs/lstm_final/h256_l2_lr5e-04_bs16",
    "vanilla_mamba": _ROOT / "runs/vanilla_mamba_final/d64_nl3_lr5e-04",
    "patchtst":      _ROOT / "runs/patchtst_final/pl16_dm128_lr5e-04",
    "epideep":       _ROOT / "runs/epideep_final/de128_eh64_lr2e-03",
    "dlinear":       _ROOT / "runs/dlinear_final/ma13_indF_lr2e-03",
}
CKPT_NAMES = {
    "lstm":          "lstm_best.pt",
    "vanilla_mamba": "vanilla_mamba_best.pt",
    "patchtst":      "patchtst_best.pt",
    "epideep":       "epideep_best.pt",
    "dlinear":       "dlinear_best.pt",
}


# ============================================================================
# Helpers
# ============================================================================
def load_norm():
    return load_norm_params(NORM_PATH)


def national_df() -> pd.DataFrame:
    """National ili_env_weekly_split CSV (in-sample residual pool source)."""
    return pd.read_csv(SPLIT_CSV)


@contextmanager
def _dropout_train_mode_ctx(model):
    """Toggle dropout layers to train mode for MC Dropout sampling.
    Mirrors src/eval/quantile_predictions._dropout_train_mode but local.
    """
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    rnn_layers = [m for m in model.modules()
                  if isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN))]
    model.eval()
    for m in dropout_layers: m.train()
    for m in rnn_layers: m.train()
    try:
        yield model
    finally:
        model.eval()


# ============================================================================
# Per-baseline forecaster builders (constructor signatures from phase_3_region_eval{,_extras}.py)
# ============================================================================
def _build_lstm(cfg, dropout):
    return LSTMForecaster(
        enc_in=cfg["enc_in"], hidden=cfg["hidden"],
        num_layers=cfg["num_layers"], pred_len=cfg["pred_len"],
        dropout=dropout,
    )


def _build_vm(cfg, dropout):
    return VanillaMambaForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg["d_state"],
        dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=dropout,
    )


def _build_patchtst(cfg, dropout):
    from baselines.patchtst import PatchTSTForecaster
    kwargs = dict(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
        patch_len=cfg["patch_len"], dropout=dropout,
    )
    if "d_ff_ratio" in cfg:
        kwargs["d_ff"] = int(cfg["d_model"] * cfg["d_ff_ratio"])
    if "stride_ratio" in cfg:
        kwargs["stride"] = int(cfg["patch_len"] * cfg["stride_ratio"])
    return PatchTSTForecaster(**kwargs)


def _build_epideep(cfg, dropout):
    from src.baselines.epideep import EpiDeepForecaster
    return EpiDeepForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
        decoder_hidden=cfg["decoder_hidden"],
        alignment_weight=cfg.get("alignment_weight", 0.0),
        dropout=dropout,
        target_only=cfg.get("target_only", False),
    )


def _build_dlinear(cfg):
    from src.baselines.dlinear import DLinearForecaster
    return DLinearForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        moving_avg=cfg["moving_avg"], individual=cfg["individual"],
    )


NN_FORECASTER_BUILDERS = {
    "lstm":          _build_lstm,
    "vanilla_mamba": _build_vm,
    "patchtst":      _build_patchtst,
    "epideep":       _build_epideep,
}


# ============================================================================
# NN baseline loader (LSTM/VM/PatchTST/EpiDeep) — seed-parameterized
# ============================================================================
def load_nn_model_seed(baseline: str, seed: int, device: str,
                          dropout_eval: float | None = None) -> tuple[torch.nn.Module, dict]:
    """Load NN baseline ckpt for a given seed, with MC Dropout rate forced.

    Args:
        baseline: one of lstm / vanilla_mamba / patchtst / epideep
        seed: 42 / 123 / 456 / 789 / 1024
        device: cuda:0 / cpu
        dropout_eval: MC Dropout rate (defaults to DROPOUT_MC[baseline] per LOCK §4)

    Returns: (model, cfg dict from results.json)
    """
    if baseline not in NN_FORECASTER_BUILDERS:
        raise ValueError(f"NN baseline must be one of {list(NN_FORECASTER_BUILDERS)}, got {baseline}")
    if dropout_eval is None:
        dropout_eval = DROPOUT_MC[baseline]
    p = RUN_DIRS[baseline] / f"seed{seed}"
    cfg = json.load(open(p / "results.json"))["config"]
    sd = torch.load(p / CKPT_NAMES[baseline], map_location=device, weights_only=True)
    model = NN_FORECASTER_BUILDERS[baseline](cfg, dropout_eval)
    model.load_state_dict(sd)
    # Force MC Dropout rate on all dropout layers
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout_eval
        elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            m.dropout = dropout_eval
    return model.to(device), cfg


def nn_mc_samples(model, ds, device, norm, n_samples: int = N_MC_SAMPLES):
    """Generic MC Dropout n-sample forward.

    Returns: (samples_raw [S, N, H], y_raw [N, H], eps_h1 [N])
    """
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    all_samples = []
    y_collect = None
    with _dropout_train_mode_ctx(model):
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
    samples = np.stack(all_samples, axis=0) * target_std + target_mean
    y_raw = y_collect * target_std + target_mean
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    return samples, y_raw, eps_h1


def nn_dataset_for_baseline(baseline: str, df: pd.DataFrame, split: str,
                                norm: dict, cfg: dict) -> WeeklyMultiHorizonDataset:
    """Build WeeklyMultiHorizonDataset with the lookback key appropriate for the baseline.

    LSTM uses cfg['lookback']; VM/PatchTST/EpiDeep use cfg['seq_len'].
    """
    lookback = cfg.get("lookback", cfg.get("seq_len"))
    pred_len = cfg["pred_len"]
    return WeeklyMultiHorizonDataset(df, split, norm, lookback=lookback, pred_len=pred_len)


# ============================================================================
# CGM (APMD) — seed-parameterized
# ============================================================================
def load_cgm_model_seed(seed: int, device: str):
    """Load CGM seed-parameterized ckpt from m2_4_data_efficiency manifest.

    Returns: (model, cfg=CGMambaConfig(), hmm)
    """
    m_path = (
        _ROOT / "runs/m2_4_data_efficiency/cg_mamba/seasons_17_seasons_full"
        / f"seed{seed}" / "manifest.json"
    )
    if not m_path.exists():
        raise FileNotFoundError(f"CGM manifest missing for seed={seed}: {m_path}")
    m = json.load(open(m_path))
    hmm = load_fitted_hmm(_ROOT / m["hmm_dir"])
    cfg = CGMambaConfig()
    model = CGForecaster(cfg)
    model.prepare_for_stage2(hmm)
    ck = torch.load(_ROOT / m["stage3_best"], map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model, cfg, hmm


def cgm_decomp_forward(model, cfg, hmm, ds, device):
    """CGM APMD decomposition over a WeeklyDataset.

    Returns: (mu_z [N,H], sigma2_total_z [N,H], y_z [N,H], eps_h1 [N])
    All arrays in z-space; caller denormalizes to raw ILI %.

    Note: This mirrors smoke._cgm_decomp_forward (validated). Implementation is identical;
    duplicated here to avoid module-level constant mutation in smoke.
    """
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


def cgm_dataset(df, split: str, cfg, norm) -> WeeklyDataset:
    """CGM-specific dataset (returns x, env dict per row)."""
    return WeeklyDataset(
        df, split=split, lookback=cfg.lookback,
        horizon=max(cfg.horizons), norm=norm,
    )


# ============================================================================
# DLinear 5-seed ensemble — non-MC, deterministic forward across seeds
# ============================================================================
def dlinear_ensemble_forward(ds, device):
    """5-seed deterministic ensemble forward over a WeeklyMultiHorizonDataset.

    Per LOCK §4 + phase_3_dlinear_ensemble_region.py pattern:
      - Load each of 5 seeds (42, 123, 456, 789, 1024)
      - Deterministic forward (model.eval(), no MC)
      - mean + std across 5 seeds → Gaussian PI

    Returns: (mu_raw [N, H], sigma_raw [N, H], y_raw [N, H], eps_h1 [N])
    """
    norm = load_norm()
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    preds_per_seed = []
    y_collect = None
    for seed in SEEDS:
        p = RUN_DIRS["dlinear"] / f"seed{seed}"
        cfg = json.load(open(p / "results.json"))["config"]
        ckpt = torch.load(p / CKPT_NAMES["dlinear"], map_location=device, weights_only=True)
        model = _build_dlinear(cfg)
        model.load_state_dict(ckpt)
        model.eval().to(device)
        n = len(ds)
        preds = np.zeros((n, 4))
        ys = np.zeros((n, 4))
        with torch.no_grad():
            for i in range(n):
                x, y = ds[i]
                x = x.unsqueeze(0).to(device)
                preds[i] = model(x)[0].cpu().numpy()
                ys[i] = y.numpy()
        preds_per_seed.append(preds)
        if y_collect is None:
            y_collect = ys
    preds = np.stack(preds_per_seed, axis=0)  # [5, N, H]
    mu = preds.mean(axis=0) * target_std + target_mean
    sig = preds.std(axis=0, ddof=1) * target_std
    sig = np.maximum(sig, 1e-6)
    y_raw = y_collect * target_std + target_mean
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    return mu, sig, y_raw, eps_h1


def dlinear_dataset(df, split: str, norm: dict, cfg_ref: dict) -> WeeklyMultiHorizonDataset:
    """DLinear dataset uses cfg_ref['seq_len']/cfg_ref['pred_len']."""
    return WeeklyMultiHorizonDataset(
        df, split, norm,
        lookback=cfg_ref["seq_len"], pred_len=cfg_ref["pred_len"],
    )


# ============================================================================
# Smoke-validated sanity at import time (cheap)
# ============================================================================
def sanity_imports():
    """Cheap import-time sanity: ensures all forecaster classes can be imported.
    Call once at module entry to catch missing baselines early.
    """
    # NN
    _ = LSTMForecaster
    _ = VanillaMambaForecaster
    try:
        from baselines.patchtst import PatchTSTForecaster
    except ImportError as e:
        raise ImportError(f"PatchTSTForecaster import failed (Time-Series-Library): {e}")
    try:
        from src.baselines.epideep import EpiDeepForecaster
    except ImportError as e:
        raise ImportError(f"EpiDeepForecaster import failed: {e}")
    try:
        from src.baselines.dlinear import DLinearForecaster
    except ImportError as e:
        raise ImportError(f"DLinearForecaster import failed: {e}")
    return True


# ============================================================================
# Section IV: Track B build_quantile + conformal + score (LOCK §2-§5)
# ============================================================================
# Mirrors scripts/p3_smoke_lstm_cgm_track_b.py:
#   build_lstm_val_base_quantiles / build_cgm_val_base_quantiles
#   build_lstm_region_test_quantiles / build_cgm_region_test_quantiles
#   conformal_cqr_with_check / score_per_cell
# Extended to all 6 baselines (LSTM, VM, PatchTST, EpiDeep, CGM, DLinear).
#
# Shapes (RAW ILI % scale):
#   qf_val  : dict[float, np.ndarray[N_val, H]]
#   y_val   : np.ndarray[N_val, H]
#   qf_test : dict[float, np.ndarray[N_strict, H]]
#   y_test  : np.ndarray[N_strict, H]
#   eps_h1  : np.ndarray[N_strict]  (h=1 epiweek; masked >= TS_BOUNDARY)
#
# DLinear: seed param ignored (uses all 5 seeds internally for ensemble).

# ----- NN helper (LSTM / VM / PatchTST / EpiDeep) ---------------------------
def _nn_val_base_quantiles(
    baseline: str, seed: int, device: str, norm: dict,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    model, cfg = load_nn_model_seed(baseline, seed, device, dropout_eval=DROPOUT_MC[baseline])
    df = national_df()
    val_ds = nn_dataset_for_baseline(baseline, df, "val", norm, cfg)
    samples, y_val_raw, _ = nn_mc_samples(model, val_ds, device, norm)
    # samples: [S, N, H] -> [N, H, S] for sample-axis-last
    qf = quantiles_from_samples(np.transpose(samples, (1, 2, 0)), taus=FLUSIGHT_23, axis=-1)
    del model
    return qf, y_val_raw


def _nn_region_test_quantiles(
    baseline: str, seed: int, device: str, norm: dict, region: str,
) -> tuple[dict[float, np.ndarray], np.ndarray, np.ndarray]:
    model, cfg = load_nn_model_seed(baseline, seed, device, dropout_eval=DROPOUT_MC[baseline])
    region_df = build_region_df(region)
    test_ds = nn_dataset_for_baseline(baseline, region_df, "test", norm, cfg)
    samples, y_raw, eps_h1 = nn_mc_samples(model, test_ds, device, norm)
    ts_idx = np.where(eps_h1 >= TS_BOUNDARY)[0]
    if len(ts_idx) == 0:
        raise RuntimeError(f"{baseline} region={region}: empty test_strict")
    samples_ts = samples[:, ts_idx, :]
    y_ts = y_raw[ts_idx]
    qf = quantiles_from_samples(np.transpose(samples_ts, (1, 2, 0)), taus=FLUSIGHT_23, axis=-1)
    del model
    return qf, y_ts, eps_h1[ts_idx]


# ----- Per-baseline val base quantile builders ------------------------------
def build_lstm_val_base_quantiles(seed: int, device: str, norm: dict):
    """LSTM national val empirical quantiles via MC Dropout (n=100)."""
    return _nn_val_base_quantiles("lstm", seed, device, norm)


def build_vanilla_mamba_val_base_quantiles(seed: int, device: str, norm: dict):
    """VanillaMamba national val empirical quantiles via MC Dropout (n=100)."""
    return _nn_val_base_quantiles("vanilla_mamba", seed, device, norm)


def build_patchtst_val_base_quantiles(seed: int, device: str, norm: dict):
    """PatchTST national val empirical quantiles via MC Dropout (n=100)."""
    return _nn_val_base_quantiles("patchtst", seed, device, norm)


def build_epideep_val_base_quantiles(seed: int, device: str, norm: dict):
    """EpiDeep national val empirical quantiles via MC Dropout (n=100)."""
    return _nn_val_base_quantiles("epideep", seed, device, norm)


def build_cg_mamba_val_base_quantiles(seed: int, device: str, norm: dict):
    """CGM (APMD) national val Gaussian quantiles from (mu, sigma2_total).

    Per LOCK §4: raw (NOT s_per_h calibrated) base quantile is used.
    """
    model, cfg, hmm = load_cgm_model_seed(seed, device)
    df = national_df()
    val_ds = cgm_dataset(df, "val", cfg, norm)
    mu_z, sig2_z, y_z, _ = cgm_decomp_forward(model, cfg, hmm, val_ds, device)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    mu_raw = mu_z * target_std + target_mean
    sig2_raw = sig2_z * (target_std ** 2)
    y_val_raw = y_z * target_std + target_mean
    qf = quantiles_from_gaussian(mu_raw, sig2_raw, taus=FLUSIGHT_23)
    del model
    return qf, y_val_raw


def build_dlinear_val_base_quantiles(seed: int, device: str, norm: dict):
    """DLinear national val Gaussian quantiles from 5-seed ensemble (mean, std).

    seed argument IGNORED — uses all 5 seeds internally for ensemble per LOCK §4.
    """
    df = national_df()
    # Reference cfg from seed42 — all 5 seeds share architecture (5-seed ensemble training)
    p_ref = RUN_DIRS["dlinear"] / "seed42"
    cfg_ref = json.load(open(p_ref / "results.json"))["config"]
    val_ds = dlinear_dataset(df, "val", norm, cfg_ref)
    mu, sig, y_val_raw, _ = dlinear_ensemble_forward(val_ds, device)
    qf = quantiles_from_gaussian(mu, sig ** 2, taus=FLUSIGHT_23)
    return qf, y_val_raw


# ----- Per-baseline region test quantile builders ---------------------------
def build_lstm_region_test_quantiles(seed: int, device: str, norm: dict, region: str):
    """LSTM regional test_strict empirical quantiles via MC Dropout (n=100)."""
    return _nn_region_test_quantiles("lstm", seed, device, norm, region)


def build_vanilla_mamba_region_test_quantiles(seed: int, device: str, norm: dict, region: str):
    """VanillaMamba regional test_strict empirical quantiles via MC Dropout (n=100)."""
    return _nn_region_test_quantiles("vanilla_mamba", seed, device, norm, region)


def build_patchtst_region_test_quantiles(seed: int, device: str, norm: dict, region: str):
    """PatchTST regional test_strict empirical quantiles via MC Dropout (n=100)."""
    return _nn_region_test_quantiles("patchtst", seed, device, norm, region)


def build_epideep_region_test_quantiles(seed: int, device: str, norm: dict, region: str):
    """EpiDeep regional test_strict empirical quantiles via MC Dropout (n=100)."""
    return _nn_region_test_quantiles("epideep", seed, device, norm, region)


def build_cg_mamba_region_test_quantiles(seed: int, device: str, norm: dict, region: str):
    """CGM (APMD) regional test_strict Gaussian quantiles from (mu, sigma2_total).

    Per LOCK §4: raw (NOT s_per_h calibrated) base quantile.
    """
    model, cfg, hmm = load_cgm_model_seed(seed, device)
    region_df = build_region_df(region)
    test_ds = cgm_dataset(region_df, "test", cfg, norm)
    mu_z, sig2_z, y_z, eps_h1 = cgm_decomp_forward(model, cfg, hmm, test_ds, device)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    mu_raw = mu_z * target_std + target_mean
    sig2_raw = sig2_z * (target_std ** 2)
    y_raw = y_z * target_std + target_mean
    ts_idx = np.where(eps_h1 >= TS_BOUNDARY)[0]
    if len(ts_idx) == 0:
        raise RuntimeError(f"cg_mamba region={region}: empty test_strict")
    mu_ts = mu_raw[ts_idx]
    sig2_ts = sig2_raw[ts_idx]
    y_ts = y_raw[ts_idx]
    qf = quantiles_from_gaussian(mu_ts, sig2_ts, taus=FLUSIGHT_23)
    del model
    return qf, y_ts, eps_h1[ts_idx]


def build_dlinear_region_test_quantiles(seed: int, device: str, norm: dict, region: str):
    """DLinear regional test_strict Gaussian quantiles from 5-seed ensemble.

    seed argument IGNORED — uses all 5 seeds internally for ensemble per LOCK §4.
    """
    region_df = build_region_df(region)
    p_ref = RUN_DIRS["dlinear"] / "seed42"
    cfg_ref = json.load(open(p_ref / "results.json"))["config"]
    test_ds = dlinear_dataset(region_df, "test", norm, cfg_ref)
    mu, sig, y_raw, eps_h1 = dlinear_ensemble_forward(test_ds, device)
    ts_idx = np.where(eps_h1 >= TS_BOUNDARY)[0]
    if len(ts_idx) == 0:
        raise RuntimeError(f"dlinear region={region}: empty test_strict")
    mu_ts = mu[ts_idx]
    sig_ts = sig[ts_idx]
    y_ts = y_raw[ts_idx]
    qf = quantiles_from_gaussian(mu_ts, sig_ts ** 2, taus=FLUSIGHT_23)
    return qf, y_ts, eps_h1[ts_idx]


# ----- Conformal CQR per horizon --------------------------------------------
def conformal_cqr_per_h(
    base_val: dict[float, np.ndarray],
    base_test: dict[float, np.ndarray],
    y_val: np.ndarray,
    baseline: str,
    region: str,
    h_idx: int,
    hard_stop_log=None,
) -> dict[float, np.ndarray]:
    """Apply CQR-symmetric Split Conformal at a single horizon h_idx.

    Slice base_val/base_test [N, H] -> per-h [N] (1-D) per tau in FLUSIGHT_23,
    then call src.eval.wis_standard.quantiles_conformal_cqr.

    Hard-stop (b): if any base quantile or y_val is non-finite, RAISE RuntimeError
    with a detailed message (caller may also append to hard_stop_log if provided).

    Returns: dict tau -> [N_strict] (1-D per-h calibrated quantile).
    """
    val_h = {float(t): np.asarray(base_val[float(t)][:, h_idx]) for t in FLUSIGHT_23}
    test_h = {float(t): np.asarray(base_test[float(t)][:, h_idx]) for t in FLUSIGHT_23}
    y_h = np.asarray(y_val[:, h_idx], dtype=np.float64)

    # Pre-check (LOCK hard-stop b)
    for t in FLUSIGHT_23:
        tf = float(t)
        if not np.all(np.isfinite(val_h[tf])):
            msg = (f"[HARD-STOP b] baseline={baseline} region={region} h={h_idx+1}: "
                   f"non-finite base_val quantile at tau={tf}")
            if hard_stop_log is not None:
                hard_stop_log.append(msg)
            raise RuntimeError(msg)
        if not np.all(np.isfinite(test_h[tf])):
            msg = (f"[HARD-STOP b] baseline={baseline} region={region} h={h_idx+1}: "
                   f"non-finite base_test quantile at tau={tf}")
            if hard_stop_log is not None:
                hard_stop_log.append(msg)
            raise RuntimeError(msg)
    if not np.all(np.isfinite(y_h)):
        msg = (f"[HARD-STOP b] baseline={baseline} region={region} h={h_idx+1}: "
               f"non-finite y_val")
        if hard_stop_log is not None:
            hard_stop_log.append(msg)
        raise RuntimeError(msg)

    qf_track_b = quantiles_conformal_cqr(
        base_quantiles_val=val_h,
        base_quantiles_test=test_h,
        y_val=y_h,
        taus=FLUSIGHT_23,
    )
    # Post-check (LOCK hard-stop b)
    for t in FLUSIGHT_23:
        tf = float(t)
        v = qf_track_b[tf]
        if not np.all(np.isfinite(v)):
            msg = (f"[HARD-STOP b] baseline={baseline} region={region} h={h_idx+1}: "
                   f"non-finite calibrated qf at tau={tf}")
            if hard_stop_log is not None:
                hard_stop_log.append(msg)
            raise RuntimeError(msg)
    return qf_track_b


# ----- Score per (region, h) cell -------------------------------------------
def score_per_cell(
    qf: dict[float, np.ndarray],
    y_test: np.ndarray,
    h_idx: int,
    label: str,
) -> dict[str, float]:
    """Score a single (region, h) cell: WIS (Bracher 2021) + Cov95 + MAE.

    Args:
        qf: dict tau -> [N_strict] (1-D per-h calibrated quantile)
        y_test: [N_strict, H] or [N_strict] — sliced at h_idx if 2-D
        h_idx: horizon index (0..3)
        label: free-form tag for error messages (e.g. "lstm/hhs1/h=1")

    Returns: {"wis": float, "cov95": float, "mae": float}
    """
    y = np.asarray(y_test)
    if y.ndim == 2:
        y = y[:, h_idx]
    y = y.astype(np.float64)
    mu_test = np.asarray(qf[0.5], dtype=np.float64)  # median = point forecast
    if mu_test.shape != y.shape:
        raise RuntimeError(
            f"score_per_cell[{label}]: shape mismatch mu={mu_test.shape} y={y.shape}"
        )
    wis_vals = wis(y, qf)
    cov95 = coverage(y, qf, alpha=0.05)
    mae = float(np.mean(np.abs(mu_test - y)))
    return {"wis": float(np.mean(wis_vals)), "cov95": float(cov95), "mae": mae}


if __name__ == "__main__":
    # Self-test: import sanity + state_dict load for ALL 5 baselines (seed42, cpu).
    # Per user condition #2 (2026-06-22): extend beyond LSTM-only to catch constructor
    # signature mismatch (cfg key / param count) BEFORE next-turn build_quantile work.
    #
    # NOTE: EpiDeep path (de128_eh64_lr2e-03) is HARDCODED from disk-evidence (5 scripts +
    # 5 seeds × ckpt exist). However, the AUTHORITATIVE lock for EpiDeep = the upcoming
    # integration test (per user condition #1, 2026-06-22): EpiDeep native cross-region
    # per-h-mean must reproduce Table IV cell (WIS=0.515, Cov95=0.382) within |Δ|<0.005.
    # If integration test FAILS for EpiDeep, the path here is wrong — DO NOT trust the
    # path until that test passes. Other 5 baselines have disk-evidence lock via
    # phase_3_region_eval(.py / _extras.py) directly referencing the same path.
    import traceback
    print("[track_b_lib] import sanity check...")
    sanity_imports()
    print("[track_b_lib] ✓ all forecaster classes importable")
    print()
    n_pass, n_fail = 0, 0
    fail_list = []
    print("[track_b_lib] state_dict load test — 5 baselines, seed=42, cpu, MC rate forced:")
    for baseline in ["lstm", "vanilla_mamba", "patchtst", "epideep"]:
        try:
            model, cfg = load_nn_model_seed(baseline, 42, "cpu",
                                               dropout_eval=DROPOUT_MC[baseline])
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ {baseline:14s}: state_dict OK, n_params={n_params:,}, "
                  f"cfg keys = {list(cfg.keys())[:5]}...")
            n_pass += 1
            del model
        except Exception as e:
            print(f"  ✗ {baseline:14s}: state_dict LOAD FAIL: {type(e).__name__}: {e}")
            fail_list.append((baseline, str(e)))
            n_fail += 1
    # DLinear: 5-seed ensemble, test single-seed load (seed=42)
    try:
        from src.baselines.dlinear import DLinearForecaster
        p = RUN_DIRS["dlinear"] / "seed42"
        cfg = json.load(open(p / "results.json"))["config"]
        ckpt = torch.load(p / CKPT_NAMES["dlinear"], map_location="cpu", weights_only=True)
        model = _build_dlinear(cfg)
        model.load_state_dict(ckpt)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {'dlinear':14s}: state_dict OK (single seed42 of 5-seed ensemble), "
              f"n_params={n_params:,}, cfg keys = {list(cfg.keys())[:5]}...")
        n_pass += 1
    except Exception as e:
        print(f"  ✗ {'dlinear':14s}: state_dict LOAD FAIL: {type(e).__name__}: {e}")
        fail_list.append(("dlinear", str(e)))
        n_fail += 1
    # CGM: seed42 manifest load + APMD prep
    try:
        model, cfg, hmm = load_cgm_model_seed(42, "cpu")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {'cg_mamba':14s}: manifest + HMM + APMD prepared, n_params={n_params:,}")
        n_pass += 1
        del model
    except Exception as e:
        print(f"  ✗ {'cg_mamba':14s}: load FAIL: {type(e).__name__}: {e}")
        fail_list.append(("cg_mamba", str(e)))
        n_fail += 1
    print()
    print(f"[track_b_lib] self-test summary: {n_pass}/6 PASS, {n_fail}/6 FAIL")
    if n_fail > 0:
        print("[track_b_lib] FAIL details (next-turn fix required BEFORE step 1-6):")
        for b, e in fail_list:
            print(f"  {b}: {e[:200]}")
        import sys
        sys.exit(1)
    print("[track_b_lib] ✓ ALL 6 baselines load OK → safe to extend with build_quantiles next turn.")
    print()
    print("[track_b_lib] EpiDeep path lock note: path is from disk-evidence (5 scripts × 5 seeds);")
    print("[track_b_lib]   AUTHORITATIVE lock = integration test (Table IV reproduction). DO NOT")
    print("[track_b_lib]   trust path until EpiDeep native cross-region per-h-mean WIS/Cov95")
    print("[track_b_lib]   reproduces Table IV (0.515 / 0.382) within |Δ|<0.005.")
