"""WIS Phase C evaluation — deterministic MAE + MC Dropout WIS for Phase C ckpts.

For each (model, dropout, seed) ∈ Phase C grid:
  1. Build model from ckpt (LSTM/Vanilla Mamba: direct; CG-Mamba: needs HMM ckpt)
  2. Deterministic forward (eval mode) → per-horizon MAE on val/test_full/test_strict
  3. MC Dropout forward (n=100, dropout layers train mode) → quantile forecasts
     → WIS per horizon on val/test_full/test_strict
  4. Decomposed WIS (dispersion / under / over) + coverage @ 50/95

Outputs:
  runs/wis_phase_c_eval/manifest.json     ← per-(model,dropout,seed) raw metrics
  runs/wis_phase_c_eval/mae_summary.csv   ← 5-seed aggregated MAE per (model, dropout, split)
  runs/wis_phase_c_eval/wis_summary.csv   ← 5-seed aggregated WIS per (model, dropout, split)
  runs/wis_phase_c_eval/winner_selection.json
      Per model: val-MAE-optimal dropout AND val-WIS-optimal dropout (separate
      selection axes — may differ. Paper main reports val-WIS-optimal dropout's
      test_strict WIS per PLAN J.4 protocol.)

GPU required (~5-10 min total on GPU 1 isolation via CUDA_VISIBLE_DEVICES=1).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset                       # noqa: E402
from baselines.vanilla_mamba import VanillaMambaForecaster                 # noqa: E402

import sys as _sys
_PARENT = _ROOT.parent
_sys.path.insert(0, str(_PARENT / "CM_Mamba"))
from cm_mamba.baselines.lstm_baseline import LSTMForecaster                # noqa: E402

from src.models.cg_forecaster import CGForecaster                          # noqa: E402
from src.utils.config import CGMambaConfig                                 # noqa: E402
from src.utils.checkpoints import load_fitted_hmm                          # noqa: E402

from src.data.loader import (                                              # noqa: E402
    load_dataset_csv, load_norm_params, MultiHorizonDataset, collate_dict,
)
from src.eval.wis import wis, wis_decomposed, coverage, REQUIRED_QUANTILES  # noqa: E402
from src.eval.quantile_predictions import _dropout_train_mode              # noqa: E402

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_ROOT = _ROOT / "runs" / "wis_phase_c_eval"
PHASE_C_MANIFEST = _ROOT / "runs" / "wis_phase_c" / "manifest.json"

COVID_STRICT_START_EPIWEEK = 202240

DROPOUTS = (0.1, 0.2, 0.3)
SEEDS = (42, 123, 456, 789, 1024)

# CG-Mamba top1 cell HP (matches Phase C training; see wis_phase_c_dropout_grid.py)
CG_MAMBA_TOP1_HP = {
    "gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104,
    "hmm_lr_ratio": 0.01, "state_embed_lr_ratio": 0.01, "env_lr_ratio": 0.001,
}
OTHER_LR_BASE = 1e-4
HMM_DIR_TEMPLATE = (
    _ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed{seed}"
)
ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"

LSTM_HP = {"lookback": 104, "pred_len": 4, "enc_in": 6,
           "hidden": 256, "num_layers": 2, "lr": 5e-4, "batch_size": 16,
           "epochs": 100, "patience": 20}
VANILLA_HP = {"seq_len": 104, "pred_len": 4, "enc_in": 6,
              "d_model": 64, "n_layers": 3, "d_state": 16, "dt_rank": 16,
              "expand": 2, "lr": 5e-4, "batch_size": 32,
              "epochs": 200, "patience": 20}


# ─── Helpers ────────────────────────────────────────────────────────────────


def _mask_df(df, split_name, epi_min):
    """Apply COVID-strict epi_min mask: exclude split rows < epi_min."""
    if epi_min is None:
        return df
    sub = df.copy()
    sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min),
            "split"] = "_excluded"
    return sub


def _build_loader_baselines(df, split_name, lookback, pred_len, norm,
                            epi_min=None, batch_size=32):
    """For LSTM/Vanilla Mamba — WeeklyMultiHorizonDataset, returns (x, y) tuples."""
    ds_df = _mask_df(df, split_name, epi_min)
    ds = WeeklyMultiHorizonDataset(ds_df, split_name, norm,
                                   lookback=lookback, pred_len=pred_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), len(ds)


def _build_loader_cg_mamba(df, split_name, lookback, norm,
                           epi_min=None, batch_size=32, horizons=(1, 2, 3, 4)):
    """For CG-Mamba — MultiHorizonDataset, returns dict {x, env, y}."""
    ds_df = _mask_df(df, split_name, epi_min)
    ds = MultiHorizonDataset(ds_df, split_name, lookback, horizons, norm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_dict), len(ds)


@torch.no_grad()
def _deterministic_forward(model, loader, target_mean, target_std, device, is_cg_mamba: bool):
    """Returns (preds_raw [N, H], y_raw [N, H])."""
    model.eval()
    preds, ys = [], []
    for batch in loader:
        if is_cg_mamba:
            x = batch["x"].to(device); env = batch["env"].to(device); y = batch["y"].to(device)
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


@torch.no_grad()
def _mc_dropout_forward(model, loader, n_samples, target_mean, target_std, device,
                        is_cg_mamba: bool):
    """Returns (samples_raw [S, N, H], y_raw [N, H])."""
    model.eval()
    all_samples = []
    y_collect = None
    with _dropout_train_mode(model):
        for _ in range(n_samples):
            preds_per_batch, ys_per_batch = [], []
            for batch in loader:
                if is_cg_mamba:
                    x = batch["x"].to(device); env = batch["env"].to(device); y = batch["y"].to(device)
                    pred = model(x, env)
                else:
                    x, y = batch
                    x = x.to(device); y = y.to(device)
                    pred = model(x)
                preds_per_batch.append(pred.cpu().numpy())
                ys_per_batch.append(y.cpu().numpy())
            preds_all = np.concatenate(preds_per_batch, axis=0)
            ys_all = np.concatenate(ys_per_batch, axis=0)
            all_samples.append(preds_all)
            if y_collect is None:
                y_collect = ys_all
    samples = np.stack(all_samples, axis=0) * target_std + target_mean
    y_raw = y_collect * target_std + target_mean
    return samples, y_raw


def _per_horizon_mae(preds_raw: np.ndarray, y_raw: np.ndarray) -> list[float]:
    return [float(np.abs(preds_raw[:, h] - y_raw[:, h]).mean())
            for h in range(preds_raw.shape[1])]


def _samples_to_wis(samples_raw: np.ndarray, y_raw: np.ndarray) -> dict:
    """[S, N, H] samples → per-horizon WIS + decomp + coverage."""
    qf = {q: np.quantile(samples_raw, q, axis=0) for q in REQUIRED_QUANTILES}
    H = y_raw.shape[1]
    wis_per_h, disp_per_h, under_per_h, over_per_h = [], [], [], []
    for h in range(H):
        qf_h = {q: qf[q][:, h] for q in qf}
        y_h = y_raw[:, h]
        wis_per_h.append(float(wis(y_h, qf_h).mean()))
        parts = wis_decomposed(y_h, qf_h)
        disp_per_h.append(float(parts["dispersion"].mean()))
        under_per_h.append(float(parts["under"].mean()))
        over_per_h.append(float(parts["over"].mean()))
    qf_flat = {q: qf[q].reshape(-1) for q in qf}
    y_flat = y_raw.reshape(-1)
    return {
        "wis_per_horizon": wis_per_h,
        "wis_avg": float(np.mean(wis_per_h)),
        "dispersion_per_horizon": disp_per_h,
        "under_per_horizon": under_per_h,
        "over_per_horizon": over_per_h,
        "coverage_50": coverage(y_flat, qf_flat, alpha=0.5),
        "coverage_95": coverage(y_flat, qf_flat, alpha=0.05),
    }


# ─── Model builders ────────────────────────────────────────────────────────


def _build_lstm(dropout: float, device: str):
    return LSTMForecaster(
        enc_in=LSTM_HP["enc_in"], hidden=LSTM_HP["hidden"],
        num_layers=LSTM_HP["num_layers"], pred_len=LSTM_HP["pred_len"],
        dropout=dropout,
    ).to(device)


def _build_vanilla(dropout: float, device: str):
    return VanillaMambaForecaster(
        seq_len=VANILLA_HP["seq_len"], pred_len=VANILLA_HP["pred_len"],
        enc_in=VANILLA_HP["enc_in"], d_model=VANILLA_HP["d_model"],
        n_layers=VANILLA_HP["n_layers"], d_state=VANILLA_HP["d_state"],
        dt_rank=VANILLA_HP["dt_rank"], expand=VANILLA_HP["expand"],
        dropout=dropout,
    ).to(device)


def _build_cg_mamba(dropout: float, seed: int, device: str):
    """Build CGForecaster with proper Stage 2 prep (HMM cache + state_embed init)."""
    hp = CG_MAMBA_TOP1_HP
    cfg = dataclasses.replace(
        CGMambaConfig(),
        seed=seed, dropout=dropout, lookback=hp["lookback"],
        stage2_gate_lr=hp["gate_lr"], stage2_backbone_lr=hp["backbone_lr"],
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * hp["hmm_lr_ratio"],
        stage3_state_embed_lr=OTHER_LR_BASE * hp["state_embed_lr_ratio"],
        stage3_env_lr=OTHER_LR_BASE * hp["env_lr_ratio"],
    )
    model = CGForecaster(cfg).to(device)
    hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))
    hmm = load_fitted_hmm(hmm_dir)
    model.prepare_for_stage2(hmm)
    if ENV_CKPT.exists():
        state = torch.load(ENV_CKPT, map_location=device, weights_only=True)
        model.env_module.encoder.load_state_dict(state)
    return model


# ─── Per-model evaluation ──────────────────────────────────────────────────


def _ckpt_path(model: str, dropout: float, seed: int) -> Path | None:
    if model == "lstm":
        p = _ROOT / "runs/wis_phase_c/lstm" / f"d{dropout}" / f"seed{seed}" / "lstm_best.pt"
    elif model == "vanilla_mamba":
        p = _ROOT / "runs/wis_phase_c/vanilla_mamba" / f"d{dropout}" / f"seed{seed}" / "vanilla_mamba_best.pt"
    elif model == "cg_mamba":
        p = _ROOT / "runs/m1_8_stage3_train" / f"wis_phase_c_cg_mamba_d{dropout}_s{seed}_stage3" / "best.pt"
    else:
        return None
    return p if p.exists() else None


def _evaluate_ckpt(model_name: str, dropout: float, seed: int,
                   n_samples: int, df, norm, device: str) -> dict:
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    if model_name == "lstm":
        lookback = LSTM_HP["lookback"]; pred_len = LSTM_HP["pred_len"]
        model = _build_lstm(dropout, device)
    elif model_name == "vanilla_mamba":
        lookback = VANILLA_HP["seq_len"]; pred_len = VANILLA_HP["pred_len"]
        model = _build_vanilla(dropout, device)
    elif model_name == "cg_mamba":
        lookback = CG_MAMBA_TOP1_HP["lookback"]; pred_len = 4
        model = _build_cg_mamba(dropout, seed, device)
    else:
        raise ValueError(model_name)

    ckpt_path = _ckpt_path(model_name, dropout, seed)
    if ckpt_path is None:
        raise FileNotFoundError(f"ckpt missing: {model_name} d={dropout} s={seed}")
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)

    is_cg_mamba = (model_name == "cg_mamba")
    splits = {}
    for split_label, epi_min in [("val", None),
                                  ("test_full", None),
                                  ("test_strict", COVID_STRICT_START_EPIWEEK)]:
        split_name = "val" if split_label == "val" else "test"
        if is_cg_mamba:
            loader, n = _build_loader_cg_mamba(
                df, split_name, lookback, norm,
                epi_min=epi_min if split_label == "test_strict" else None,
            )
        else:
            loader, n = _build_loader_baselines(
                df, split_name, lookback, pred_len, norm,
                epi_min=epi_min if split_label == "test_strict" else None,
            )
        # Deterministic MAE
        preds_raw, y_raw = _deterministic_forward(
            model, loader, target_mean, target_std, device, is_cg_mamba,
        )
        mae_per_h = _per_horizon_mae(preds_raw, y_raw)
        # MC Dropout WIS
        samples, y_raw2 = _mc_dropout_forward(
            model, loader, n_samples, target_mean, target_std, device, is_cg_mamba,
        )
        wis_results = _samples_to_wis(samples, y_raw2)
        splits[split_label] = {
            "n": int(n),
            "mae_per_horizon": mae_per_h,
            "mae_avg": float(np.mean(mae_per_h)),
            **wis_results,
        }
    return {"model": model_name, "dropout": dropout, "seed": seed, "splits": splits}


# ─── Aggregation ───────────────────────────────────────────────────────────


def _aggregate(per_ckpt_results: list[dict]) -> dict:
    """5-seed mean ± std per (model, dropout, split). Per-model winner selection."""
    by = defaultdict(list)
    for r in per_ckpt_results:
        by[(r["model"], r["dropout"])].append(r)
    agg = {}
    splits = ("val", "test_full", "test_strict")
    for (model, d), runs in by.items():
        per_split = {}
        for sp in splits:
            mae_avg_arr = np.array([r["splits"][sp]["mae_avg"] for r in runs])
            wis_avg_arr = np.array([r["splits"][sp]["wis_avg"] for r in runs])
            cov50 = np.array([r["splits"][sp]["coverage_50"] for r in runs])
            cov95 = np.array([r["splits"][sp]["coverage_95"] for r in runs])
            n_val = runs[0]["splits"][sp]["n"]
            per_split[sp] = {
                "n": int(n_val),
                "mae_avg_mean": float(mae_avg_arr.mean()),
                "mae_avg_std": float(mae_avg_arr.std(ddof=1)),
                "wis_avg_mean": float(wis_avg_arr.mean()),
                "wis_avg_std": float(wis_avg_arr.std(ddof=1)),
                "coverage_50_mean": float(cov50.mean()),
                "coverage_95_mean": float(cov95.mean()),
            }
        agg[(model, d)] = per_split
    return agg


def _select_winners(agg: dict) -> dict:
    """Per model: val-MAE-optimal and val-WIS-optimal dropout selection."""
    winners = {}
    for model in ("lstm", "vanilla_mamba", "cg_mamba"):
        by_d_mae = {d: agg[(model, d)]["val"]["mae_avg_mean"]
                    for d in DROPOUTS if (model, d) in agg}
        by_d_wis = {d: agg[(model, d)]["val"]["wis_avg_mean"]
                    for d in DROPOUTS if (model, d) in agg}
        if not by_d_mae:
            continue
        mae_winner = min(by_d_mae, key=by_d_mae.get)
        wis_winner = min(by_d_wis, key=by_d_wis.get)
        winners[model] = {
            "val_mae_optimal_dropout": mae_winner,
            "val_mae_optimal_value": by_d_mae[mae_winner],
            "val_wis_optimal_dropout": wis_winner,
            "val_wis_optimal_value": by_d_wis[wis_winner],
            "val_mae_all": by_d_mae,
            "val_wis_all": by_d_wis,
            # Paper main reporting: WIS-optimal dropout's test_strict
            "paper_main_test_strict_wis_avg": agg[(model, wis_winner)]["test_strict"]["wis_avg_mean"],
            "paper_main_test_strict_wis_std": agg[(model, wis_winner)]["test_strict"]["wis_avg_std"],
            "paper_main_test_strict_mae_avg": agg[(model, wis_winner)]["test_strict"]["mae_avg_mean"],
        }
    return winners


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--models", nargs="+",
                    default=["lstm", "vanilla_mamba", "cg_mamba"],
                    choices=["lstm", "vanilla_mamba", "cg_mamba"])
    ap.add_argument("--dropouts", type=float, nargs="+", default=list(DROPOUTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)

    plan = [(m, d, s) for m in args.models for d in args.dropouts for s in args.seeds]
    print(f"Plan: {len(plan)} ckpts to evaluate  n_samples={args.n_samples}  device={args.device}")

    per_ckpt = []
    total_t0 = time.time()
    for i, (m, d, s) in enumerate(plan, 1):
        t0 = time.time()
        try:
            r = _evaluate_ckpt(m, d, s, args.n_samples, df, norm, args.device)
            per_ckpt.append(r)
            el = time.time() - t0
            sp = r["splits"]
            print(f"[{i:2d}/{len(plan)}] {m:13s} d={d} s={s:4d} "
                  f"val: MAE={sp['val']['mae_avg']:.4f} WIS={sp['val']['wis_avg']:.4f}  "
                  f"tF: MAE={sp['test_full']['mae_avg']:.4f} WIS={sp['test_full']['wis_avg']:.4f}  "
                  f"tS: MAE={sp['test_strict']['mae_avg']:.4f} WIS={sp['test_strict']['wis_avg']:.4f}  "
                  f"({el:.1f}s)")
        except Exception as e:
            import traceback
            print(f"[{i:2d}/{len(plan)}] {m} d={d} s={s} FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

    # Save raw per-ckpt manifest
    (OUT_ROOT / "manifest.json").write_text(json.dumps(per_ckpt, indent=2))

    # Aggregate + winner selection
    agg = _aggregate(per_ckpt)
    winners = _select_winners(agg)

    # ── Output: MAE summary CSV ──
    mae_rows = []
    for (model, d), per_split in sorted(agg.items()):
        for sp in ("val", "test_full", "test_strict"):
            mae_rows.append({
                "model": model, "dropout": d, "split": sp,
                "n": per_split[sp]["n"],
                "mae_mean": per_split[sp]["mae_avg_mean"],
                "mae_std": per_split[sp]["mae_avg_std"],
            })
    with (OUT_ROOT / "mae_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mae_rows[0].keys()))
        w.writeheader(); w.writerows(mae_rows)

    # ── Output: WIS summary CSV ──
    wis_rows = []
    for (model, d), per_split in sorted(agg.items()):
        for sp in ("val", "test_full", "test_strict"):
            wis_rows.append({
                "model": model, "dropout": d, "split": sp,
                "n": per_split[sp]["n"],
                "wis_mean": per_split[sp]["wis_avg_mean"],
                "wis_std": per_split[sp]["wis_avg_std"],
                "cov50": per_split[sp]["coverage_50_mean"],
                "cov95": per_split[sp]["coverage_95_mean"],
            })
    with (OUT_ROOT / "wis_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wis_rows[0].keys()))
        w.writeheader(); w.writerows(wis_rows)

    # Winners
    (OUT_ROOT / "winner_selection.json").write_text(json.dumps(winners, indent=2))

    total_el = time.time() - total_t0
    print(f"\n=== Phase C eval done — total {total_el/60:.1f} min ===")
    print(f"Saved: mae_summary.csv  wis_summary.csv  winner_selection.json")
    print()
    print("=" * 100)
    print("Per-model val-WIS-optimal dropout + test_strict report (paper main)")
    print("=" * 100)
    for model, w in winners.items():
        print(f"  {model:14s}  val-MAE-opt d={w['val_mae_optimal_dropout']}  "
              f"val-WIS-opt d={w['val_wis_optimal_dropout']}  "
              f"→ test_strict WIS = {w['paper_main_test_strict_wis_avg']:.4f}"
              f" ± {w['paper_main_test_strict_wis_std']:.4f}"
              f"  (MAE = {w['paper_main_test_strict_mae_avg']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
