"""Conformal Prediction (Option C) — universal coverage-guaranteed UQ.

Per-horizon split conformal (Vovk et al. 2005; Romano et al. 2019).
Method-agnostic: applies to any baseline's point prediction.

For each baseline with point predictions:
  1. Compute signed residuals on val: r_h_i = y_val_h_i - pred_val_h_i
  2. For target quantile q, conformal quantile offset:
        q_alpha_h = quantile(r_h, q · (n+1)/n)        ← finite-sample correction
  3. Test interval: quantile_q_h = pred_test_h + q_alpha_h

Coverage guarantee (Vovk 2005): under exchangeability, P(y ∈ PI_α) ≥ 1-α.

Note: exchangeability between val (pre-COVID) and test_strict (post-COVID) may
be violated. We report empirical coverage to verify guarantee holds.

Output:
  runs/wis_conformal/per_baseline/<name>.json
  runs/wis_conformal/summary_table.csv

Defense: reviewer Attack 2 (architectural privilege) + Attack 3 (MC Dropout
default) 차단 — universal benchmark.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.eval.wis import wis, coverage, REQUIRED_QUANTILES, wis_decomposed
from src.data.loader import (
    load_dataset_csv, load_norm_params, MultiHorizonDataset, collate_dict,
)
from baselines.lstm import WeeklyMultiHorizonDataset

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_DIR = _ROOT / "runs" / "wis_conformal"
COVID_STRICT_START_EPIWEEK = 202240

SEEDS = (42, 123, 456, 789, 1024)


def conformal_quantiles(
    point_pred_test: np.ndarray,    # [N_test, H]
    residuals_val: np.ndarray,      # [N_val, H] signed residuals (y - pred)
) -> dict[float, np.ndarray]:
    """Per-horizon split conformal: quantile_q_h = pred_h + Q_h(q·(n+1)/n).

    Returns dict q -> [N_test, H] calibrated quantiles.
    """
    N_val, H = residuals_val.shape
    out = {}
    # Finite-sample correction (Romano 2019): quantile level (1 + 1/n) × q
    for q in REQUIRED_QUANTILES:
        # Adjusted quantile level (clipped to [0,1])
        q_adj = min(1.0, q * (N_val + 1) / N_val) if q <= 0.5 \
                else max(0.0, 1 - (1 - q) * (N_val + 1) / N_val)
        offset_h = np.array([np.quantile(residuals_val[:, h], q_adj)
                              for h in range(H)])     # [H]
        out[q] = point_pred_test + offset_h[None, :]
    return out


def score_split(quantiles_raw, y_raw):
    N, H = y_raw.shape
    wis_per_h, disp, under, over = [], [], [], []
    for h in range(H):
        qf_h = {q: quantiles_raw[q][:, h] for q in quantiles_raw}
        y_h = y_raw[:, h]
        w = wis(y_h, qf_h)
        wis_per_h.append(float(w.mean()))
        parts = wis_decomposed(y_h, qf_h)
        disp.append(float(parts["dispersion"].mean()))
        under.append(float(parts["under"].mean()))
        over.append(float(parts["over"].mean()))
    qf_flat = {q: quantiles_raw[q].reshape(-1) for q in quantiles_raw}
    y_flat = y_raw.reshape(-1)
    return {
        "n": int(N),
        "wis_per_horizon": wis_per_h,
        "wis_avg": float(np.mean(wis_per_h)),
        "wis_decomposed": {
            "dispersion_per_horizon": disp,
            "under_per_horizon": under,
            "over_per_horizon": over,
            "dispersion_avg": float(np.mean(disp)),
            "under_avg": float(np.mean(under)),
            "over_avg": float(np.mean(over)),
        },
        "coverage_50": coverage(y_flat, qf_flat, alpha=0.5),
        "coverage_95": coverage(y_flat, qf_flat, alpha=0.05),
    }


# ─── Baseline point-prediction extractors ────────────────────────────────────


def _mask_df(df, split_name, epi_min):
    if epi_min is None:
        return df
    sub = df.copy()
    sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min),
            "split"] = "_excluded"
    return sub


@torch.no_grad()
def _forward_baseline(model, loader, target_mean, target_std, device,
                     is_cg_mamba=False):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        if is_cg_mamba:
            x = batch["x"].to(device); env = batch["env"].to(device); y = batch["y"]
            pred = model(x, env)
        else:
            x, y = batch
            x = x.to(device); y = y.to(device)
            pred = model(x)
        preds.append(pred.cpu().numpy())
        ys.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    return preds * target_std + target_mean, ys * target_std + target_mean


def _build_loader(df, split_name, lookback, pred_len, norm, epi_min, batch_size=32,
                  is_cg_mamba=False, horizons=(1, 2, 3, 4)):
    ds_df = _mask_df(df, split_name, epi_min)
    if is_cg_mamba:
        ds = MultiHorizonDataset(ds_df, split_name, lookback, horizons, norm)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, collate_fn=collate_dict)
    else:
        ds = WeeklyMultiHorizonDataset(ds_df, split_name, norm,
                                       lookback=lookback, pred_len=pred_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_ensemble_mean_preds(baseline, cfg_name, ckpt_file, build_fn,
                            df, norm, device, split_name, epi_min):
    """5-seed ensemble: returns mean predictions [N, H] raw."""
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    cfg_root = _ROOT / "runs" / f"{baseline}_final" / cfg_name
    cfg = json.load(open(cfg_root / "seed42" / "results.json"))["config"]
    lookback = cfg.get("seq_len", cfg.get("lookback", 104))
    pred_len = cfg["pred_len"]
    loader = _build_loader(df, split_name, lookback, pred_len, norm, epi_min)

    preds_per_seed = []
    y_raw = None
    for seed in SEEDS:
        ckpt = cfg_root / f"seed{seed}" / ckpt_file
        if not ckpt.exists():
            continue
        model = build_fn(cfg).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=True)
        p, y = _forward_baseline(model, loader, target_mean, target_std, device)
        preds_per_seed.append(p)
        y_raw = y
    arr = np.stack(preds_per_seed, axis=0)
    return arr.mean(axis=0), y_raw


def get_phase_c_mean_preds(model_key, dropout, df, norm, device, split_name, epi_min):
    """Phase C trained model: deterministic forward (eval mode) → mean prediction."""
    import dataclasses as dc
    from src.models.cg_forecaster import CGForecaster
    from src.utils.config import CGMambaConfig
    from src.utils.checkpoints import load_fitted_hmm
    from baselines.vanilla_mamba import VanillaMambaForecaster
    sys.path.insert(0, str(_ROOT.parent / "CM_Mamba"))
    from cm_mamba.baselines.lstm_baseline import LSTMForecaster

    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    CG_TOP1_HP = {"gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104,
                  "hmm_lr_ratio": 0.01, "state_embed_lr_ratio": 0.01,
                  "env_lr_ratio": 0.001}
    OTHER_LR_BASE = 1e-4
    HMM_DIR_TEMPLATE = (
        _ROOT / "runs" / "m1_4_phase_dynamics_main"
        / "V_raw3_regcov5e-03_K3_seed{seed}"
    )
    ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"

    LSTM_HP = {"lookback": 104, "pred_len": 4, "enc_in": 6,
               "hidden": 256, "num_layers": 2, "lr": 5e-4, "batch_size": 16,
               "epochs": 100, "patience": 20}
    VAN_HP = {"seq_len": 104, "pred_len": 4, "enc_in": 6, "d_model": 64,
              "n_layers": 3, "d_state": 16, "dt_rank": 16, "expand": 2,
              "lr": 5e-4, "batch_size": 32, "epochs": 200, "patience": 20}

    def _build_phase_c_model(seed):
        if model_key == "lstm":
            return LSTMForecaster(enc_in=LSTM_HP["enc_in"], hidden=LSTM_HP["hidden"],
                                  num_layers=LSTM_HP["num_layers"],
                                  pred_len=LSTM_HP["pred_len"], dropout=dropout).to(device)
        elif model_key == "vanilla_mamba":
            return VanillaMambaForecaster(
                seq_len=VAN_HP["seq_len"], pred_len=VAN_HP["pred_len"],
                enc_in=VAN_HP["enc_in"], d_model=VAN_HP["d_model"],
                n_layers=VAN_HP["n_layers"], d_state=VAN_HP["d_state"],
                dt_rank=VAN_HP["dt_rank"], expand=VAN_HP["expand"],
                dropout=dropout,
            ).to(device)
        else:  # cg_mamba
            cfg = dc.replace(
                CGMambaConfig(),
                seed=seed, dropout=dropout, lookback=CG_TOP1_HP["lookback"],
                stage2_gate_lr=CG_TOP1_HP["gate_lr"],
                stage2_backbone_lr=CG_TOP1_HP["backbone_lr"],
                stage3_other_lr=OTHER_LR_BASE,
                stage3_hmm_lr=OTHER_LR_BASE * CG_TOP1_HP["hmm_lr_ratio"],
                stage3_state_embed_lr=OTHER_LR_BASE * CG_TOP1_HP["state_embed_lr_ratio"],
                stage3_env_lr=OTHER_LR_BASE * CG_TOP1_HP["env_lr_ratio"],
            )
            m = CGForecaster(cfg).to(device)
            hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))
            hmm = load_fitted_hmm(hmm_dir)
            m.prepare_for_stage2(hmm)
            if ENV_CKPT.exists():
                state = torch.load(ENV_CKPT, map_location=device, weights_only=True)
                m.env_module.encoder.load_state_dict(state)
            return m

    # Ckpt paths
    def _ckpt_path(seed):
        if model_key == "lstm":
            return _ROOT / "runs/wis_phase_c/lstm" / f"d{dropout}" / f"seed{seed}" / "lstm_best.pt"
        elif model_key == "vanilla_mamba":
            return _ROOT / "runs/wis_phase_c/vanilla_mamba" / f"d{dropout}" / f"seed{seed}" / "vanilla_mamba_best.pt"
        else:
            return _ROOT / "runs/m1_8_stage3_train" / f"wis_phase_c_cg_mamba_d{dropout}_s{seed}_stage3" / "best.pt"

    lookback = CG_TOP1_HP["lookback"] if model_key == "cg_mamba" else (
        LSTM_HP["lookback"] if model_key == "lstm" else VAN_HP["seq_len"])
    is_cgm = (model_key == "cg_mamba")
    loader = _build_loader(df, split_name, lookback, 4, norm, epi_min,
                          is_cg_mamba=is_cgm)

    preds_per_seed = []
    y_raw = None
    for seed in SEEDS:
        ckpt_p = _ckpt_path(seed)
        if not ckpt_p.exists():
            continue
        model = _build_phase_c_model(seed)
        sd = torch.load(ckpt_p, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=True)
        p, y = _forward_baseline(model, loader, target_mean, target_std, device,
                                 is_cg_mamba=is_cgm)
        preds_per_seed.append(p)
        y_raw = y
    return np.stack(preds_per_seed, axis=0).mean(axis=0), y_raw


def get_m21_cg_mamba_mean_preds(df, norm, device, split_name, epi_min):
    """M2.1 CG-Mamba top1 cell, dropout=0.0 ckpts."""
    import dataclasses as dc
    from src.models.cg_forecaster import CGForecaster
    from src.utils.config import CGMambaConfig
    from src.utils.checkpoints import load_fitted_hmm

    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    CG_TOP1_HP = {"gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104,
                  "hmm_lr_ratio": 0.01, "state_embed_lr_ratio": 0.01,
                  "env_lr_ratio": 0.001}
    OTHER_LR_BASE = 1e-4
    HMM_DIR_TEMPLATE = (_ROOT / "runs" / "m1_4_phase_dynamics_main"
                        / "V_raw3_regcov5e-03_K3_seed{seed}")
    ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"
    CG_M21_CKPT_TEMPLATE = (_ROOT / "runs" / "m1_8_stage3_train" /
        "hpo_p2_g1e-03_b1e-04_lb104_h0.01_se0.01_e0.001_s{seed}" / "best.pt")

    loader = _build_loader(df, split_name, 104, 4, norm, epi_min, is_cg_mamba=True)
    preds_per_seed = []
    y_raw = None
    for seed in SEEDS:
        cfg = dc.replace(
            CGMambaConfig(),
            seed=seed, dropout=0.0, lookback=104,
            stage2_gate_lr=CG_TOP1_HP["gate_lr"],
            stage2_backbone_lr=CG_TOP1_HP["backbone_lr"],
            stage3_other_lr=OTHER_LR_BASE,
            stage3_hmm_lr=OTHER_LR_BASE * CG_TOP1_HP["hmm_lr_ratio"],
            stage3_state_embed_lr=OTHER_LR_BASE * CG_TOP1_HP["state_embed_lr_ratio"],
            stage3_env_lr=OTHER_LR_BASE * CG_TOP1_HP["env_lr_ratio"],
        )
        m = CGForecaster(cfg).to(device)
        hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))
        hmm = load_fitted_hmm(hmm_dir)
        m.prepare_for_stage2(hmm)
        if ENV_CKPT.exists():
            state = torch.load(ENV_CKPT, map_location=device, weights_only=True)
            m.env_module.encoder.load_state_dict(state)
        ckpt_p = Path(str(CG_M21_CKPT_TEMPLATE).format(seed=seed))
        sd = torch.load(ckpt_p, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        m.load_state_dict(sd, strict=True)
        p, y = _forward_baseline(m, loader, target_mean, target_std, device, is_cg_mamba=True)
        preds_per_seed.append(p)
        y_raw = y
    return np.stack(preds_per_seed, axis=0).mean(axis=0), y_raw


def get_sarima_mean_preds(df, split_name, epi_min):
    """SARIMA: re-fit + rolling forecast → point predictions per (target_ep, h).

    Returns (mu_raw [N, H], y_raw [N, H]). N matches NN baseline n=149 by
    intersecting target_eps across horizons.
    """
    import warnings
    from baselines.sarima import build_segment_arrays, fit_sarimax, is_consecutive_epiweek

    TRAIN_FIRST = 200240; TRAIN_LAST = 201839
    VAL_FIRST = 201840;   VAL_LAST = 202010
    TEST_FIRST = 202040;  TEST_LAST = 202535
    horizons = (1, 2, 3, 4)

    # Reuse selected order from runs/baselines/sarima.json
    sarima_blob = json.load(open(_ROOT / "runs/baselines/sarima.json"))
    order = tuple(sarima_blob["selected_order"]["order"])
    seasonal_order = tuple(sarima_blob["selected_order"]["seasonal_order"])

    y_tr, X_tr, ep_tr, _ = build_segment_arrays(df, TRAIN_FIRST, TRAIN_LAST)
    y_va, X_va, ep_va, _ = build_segment_arrays(df, VAL_FIRST, VAL_LAST)
    y_te, X_te, ep_te, _ = build_segment_arrays(df, TEST_FIRST, TEST_LAST)

    # Fit on appropriate base segment
    if split_name == "val":
        res = fit_sarimax(y_tr, X_tr, order, seasonal_order)
        y_seg = y_va; X_seg = X_va; ep_seg = ep_va
        prev_ep_last = int(ep_tr[-1])
    else:  # test (both test_full and test_strict come from same test rolling)
        y_trva = np.concatenate([y_tr, y_va])
        X_trva = np.concatenate([X_tr, X_va])
        res = fit_sarimax(y_trva, X_trva, order, seasonal_order)
        y_seg = y_te; X_seg = X_te; ep_seg = ep_te
        prev_ep_last = int(ep_va[-1])

    # Rolling forecast (mean only)
    H = max(horizons)
    N = len(y_seg)
    preds_by_h = {h: [] for h in horizons}
    current_res = res
    for t in range(N):
        steps = min(H, N - t)
        if steps == 0: break
        future_exog = X_seg[t:t + steps]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = np.asarray(current_res.forecast(steps=steps, exog=future_exog),
                           dtype=np.float64)
        for h in horizons:
            target_idx = t + h - 1
            if target_idx >= N: continue
            if t == 0 and not is_consecutive_epiweek(prev_ep_last, int(ep_seg[0])):
                continue
            preds_by_h[h].append({
                "target_ep": int(ep_seg[target_idx]),
                "y": float(y_seg[target_idx]),
                "mean": float(fc[h - 1]),
            })
        if t + 1 < N:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_res = current_res.append([y_seg[t]], exog=X_seg[t:t+1], refit=False)

    # Apply split-level epiweek filter
    if epi_min is not None:
        for h in horizons:
            preds_by_h[h] = [p for p in preds_by_h[h] if p["target_ep"] >= epi_min]
    if split_name == "test" and epi_min is None:
        # test_full uses all of test segment
        pass

    # Intersect target_eps across horizons
    by_ep = {h: {p["target_ep"]: p for p in preds_by_h[h]} for h in horizons}
    common_eps = sorted(set.intersection(*[set(by_ep[h].keys()) for h in horizons]))
    if not common_eps:
        return None, None
    mu = np.array([[by_ep[h][ep]["mean"] for h in horizons] for ep in common_eps])
    y = np.array([[by_ep[h][ep]["y"] for h in horizons] for ep in common_eps])
    return mu, y


def get_persistence_mean_preds(df, norm, split_name, epi_min):
    """Persistence: y_{t+h} = y_t. No model load needed."""
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    loader = _build_loader(df, split_name, 104, 4, norm, epi_min)
    preds, ys = [], []
    for x, y in loader:
        last = x[:, -1, 0].numpy()
        rep = np.repeat(last[:, None], 4, axis=1)
        preds.append(rep)
        ys.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    return preds * target_std + target_mean, ys * target_std + target_mean


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baselines", nargs="+", default=["all"],
                    help="Subset (or 'all'): persistence, dlinear, nbeats, "
                         "patchtst, itransformer, timesnet, epideep, "
                         "lstm_phc, vanilla_phc, cg_mamba_m21")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_baseline").mkdir(exist_ok=True)
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)

    # Baseline configurations
    from baselines.dlinear import DLinearForecaster
    from baselines.nbeats import NBeatsForecaster
    from baselines.patchtst import PatchTSTForecaster
    from baselines.itransformer import ITransformerForecaster
    from baselines.timesnet import TimesNetForecaster
    from baselines.epideep import EpiDeepForecaster

    PHASE_B_BASELINES = {
        "dlinear": ("ma13_indF_lr2e-03", "dlinear_best.pt", lambda cfg: DLinearForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            moving_avg=cfg["moving_avg"], individual=cfg["individual"])),
        "nbeats": ("nb24_h512_lr5e-04", "nbeats_best.pt", lambda cfg: NBeatsForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            hidden=cfg["hidden"], n_blocks=cfg["n_blocks"], n_layers=cfg["n_layers"],
            target_only=cfg.get("target_only", False))),
        "patchtst": ("pl16_dm64_lr5e-04", "patchtst_best.pt", lambda cfg: PatchTSTForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
            d_ff=cfg["d_ff_ratio"] * cfg["d_model"], patch_len=cfg["patch_len"],
            stride=max(1, int(cfg["patch_len"] * cfg["stride_ratio"])),
            dropout=cfg["dropout"])),
        "itransformer": ("dm256_el4_lr5e-04", "itransformer_best.pt",
                         lambda cfg: ITransformerForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
            d_ff=cfg["d_ff_ratio"] * cfg["d_model"], dropout=cfg["dropout"],
            embed=cfg.get("embed", "timeF"), freq=cfg.get("freq", "w"))),
        "timesnet": ("d64_el2_lr1e-03", "timesnet_best.pt", lambda cfg: TimesNetForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], d_ff=cfg["d_ff"], e_layers=cfg["e_layers"],
            top_k=cfg["top_k"], num_kernels=cfg["num_kernels"], dropout=cfg["dropout"])),
        "epideep": ("de128_eh64_lr2e-03", "epideep_best.pt", lambda cfg: EpiDeepForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
            decoder_hidden=cfg["decoder_hidden"], alignment_weight=cfg["alignment_weight"],
            dropout=cfg["dropout"], target_only=cfg.get("target_only", False))),
    }

    all_baseline_names = (list(PHASE_B_BASELINES.keys()) +
                          ["sarima", "persistence", "lstm_phc", "vanilla_phc", "cg_mamba_m21"])
    if "all" in args.baselines:
        baselines_to_run = all_baseline_names
    else:
        baselines_to_run = args.baselines

    results = {}
    SPLITS = [("val", None), ("test_full", None),
              ("test_strict", COVID_STRICT_START_EPIWEEK)]

    for b in baselines_to_run:
        print(f"\n=== {b.upper()} — Conformal ===")
        # Get val residuals first
        if b in PHASE_B_BASELINES:
            cfg_name, ckpt_file, build_fn = PHASE_B_BASELINES[b]
            try:
                val_mu, val_y = get_ensemble_mean_preds(b, cfg_name, ckpt_file, build_fn,
                                                        df, norm, args.device, "val", None)
            except Exception as e:
                print(f"  SKIP — {type(e).__name__}: {e}")
                continue
        elif b == "sarima":
            val_mu, val_y = get_sarima_mean_preds(df, "val", None)
            if val_mu is None:
                print(f"  SKIP — no val data"); continue
        elif b == "persistence":
            val_mu, val_y = get_persistence_mean_preds(df, norm, "val", None)
        elif b == "lstm_phc":
            val_mu, val_y = get_phase_c_mean_preds("lstm", 0.1, df, norm, args.device,
                                                   "val", None)
        elif b == "vanilla_phc":
            val_mu, val_y = get_phase_c_mean_preds("vanilla_mamba", 0.1, df, norm,
                                                   args.device, "val", None)
        elif b == "cg_mamba_m21":
            val_mu, val_y = get_m21_cg_mamba_mean_preds(df, norm, args.device, "val", None)
        else:
            print(f"  SKIP — unknown baseline")
            continue

        val_residuals = val_y - val_mu                    # [N_val, H] signed
        print(f"  val residuals: shape={val_residuals.shape}  "
              f"std per h: {[f'{val_residuals[:,h].std():.3f}' for h in range(val_residuals.shape[1])]}")

        # Apply conformal to each split
        b_result = {"baseline": b, "uq": "split_conformal_per_horizon",
                    "val_residuals_stats": {
                        "n_val": int(val_residuals.shape[0]),
                        "std_per_h": [float(val_residuals[:, h].std()) for h in range(val_residuals.shape[1])],
                        "mean_per_h": [float(val_residuals[:, h].mean()) for h in range(val_residuals.shape[1])],
                    },
                    "splits": {}}
        for split_label, epi_min in SPLITS:
            split_name = "val" if split_label == "val" else "test"
            try:
                if b in PHASE_B_BASELINES:
                    cfg_name, ckpt_file, build_fn = PHASE_B_BASELINES[b]
                    test_mu, test_y = get_ensemble_mean_preds(
                        b, cfg_name, ckpt_file, build_fn,
                        df, norm, args.device, split_name, epi_min)
                elif b == "sarima":
                    test_mu, test_y = get_sarima_mean_preds(df, split_name, epi_min)
                elif b == "persistence":
                    test_mu, test_y = get_persistence_mean_preds(
                        df, norm, split_name, epi_min)
                elif b == "lstm_phc":
                    test_mu, test_y = get_phase_c_mean_preds(
                        "lstm", 0.1, df, norm, args.device, split_name, epi_min)
                elif b == "vanilla_phc":
                    test_mu, test_y = get_phase_c_mean_preds(
                        "vanilla_mamba", 0.1, df, norm, args.device, split_name, epi_min)
                elif b == "cg_mamba_m21":
                    test_mu, test_y = get_m21_cg_mamba_mean_preds(
                        df, norm, args.device, split_name, epi_min)
            except Exception as e:
                print(f"  [{split_label}] SKIP — {type(e).__name__}: {e}")
                continue
            qf = conformal_quantiles(test_mu, val_residuals)
            score = score_split(qf, test_y)
            b_result["splits"][split_label] = score
            print(f"  [{split_label:11s}] WIS={score['wis_avg']:.4f}  "
                  f"cov50={score['coverage_50']:.3f}  cov95={score['coverage_95']:.3f}")

        results[b] = b_result
        out_path = OUT_DIR / "per_baseline" / f"{b}.json"
        out_path.write_text(json.dumps(b_result, indent=2))

    # Summary CSV
    print("\n" + "=" * 90)
    print("Conformal Prediction Summary (test_strict, paper main)")
    print("=" * 90)
    print(f"{'Baseline':<18s} {'WIS (conformal)':>16s} {'cov50':>8s} {'cov95':>8s}")
    print("-" * 90)
    rows_for_csv = []
    sorted_b = sorted(results.items(), key=lambda kv: kv[1]["splits"].get("test_strict", {}).get("wis_avg", 999))
    for b, r in sorted_b:
        if "test_strict" not in r["splits"]:
            continue
        ts = r["splits"]["test_strict"]
        print(f"{b:<18s} {ts['wis_avg']:>16.4f} {ts['coverage_50']:>8.3f} {ts['coverage_95']:>8.3f}")
        rows_for_csv.append({
            "baseline": b, "wis_avg": ts["wis_avg"],
            "wis_h1": ts["wis_per_horizon"][0], "wis_h2": ts["wis_per_horizon"][1],
            "wis_h3": ts["wis_per_horizon"][2], "wis_h4": ts["wis_per_horizon"][3],
            "cov50": ts["coverage_50"], "cov95": ts["coverage_95"],
        })

    with (OUT_DIR / "summary_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
        w.writeheader(); w.writerows(rows_for_csv)
    print(f"\nSaved: {(OUT_DIR / 'summary_table.csv').relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
