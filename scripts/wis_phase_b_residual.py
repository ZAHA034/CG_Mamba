"""WIS Phase B group 2 — residual / ensemble Gaussian quantile baselines (PLAN J.3).

Computes WIS for 3 no-retrain baselines:
  - Persistence: y_{t+h} = y_t (no ckpt). Val h-specific residual → empirical quantile.
  - DLinear:     5-seed ensemble Gaussian fit per (obs, horizon).
  - N-BEATS:     5-seed ensemble Gaussian fit per (obs, horizon).

Splits: val, test_full, test_strict (COVID-strict ≥ W40-2022 per PLAN §M2.1).

Output:
  runs/wis_phase_b/persistence/wis_results.json
  runs/wis_phase_b/dlinear/wis_results.json
  runs/wis_phase_b/nbeats/wis_results.json

Each JSON has the shape:
  {
    "baseline": "<name>",
    "cfg_name": "<winner cfg>",
    "splits": {
      "val": {
        "n": int,
        "wis_per_horizon": [h1, h2, h3, h4],
        "wis_avg": float,
        "wis_decomposed": {"dispersion": float, "under": float, "over": float},
        "coverage_50": float,  // 50% PI empirical coverage
        "coverage_95": float,
      },
      "test_full": {...},
      "test_strict": {...}
    }
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset                       # noqa: E402
from baselines.dlinear import DLinearForecaster                            # noqa: E402
from baselines.nbeats import NBeatsForecaster                              # noqa: E402

from src.data.loader import load_dataset_csv, load_norm_params             # noqa: E402
from src.eval.wis import wis, wis_decomposed, coverage                     # noqa: E402
from src.eval.quantile_predictions import (                                # noqa: E402
    residual_quantiles_h_specific,
    ensemble_gaussian_quantiles,
)

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_ROOT = _ROOT / "runs" / "wis_phase_b"
COVID_STRICT_START_EPIWEEK = 202240

SEEDS = (42, 123, 456, 789, 1024)


# ─── Data helpers ──────────────────────────────────────────────────────────


def _build_loader(df, split_name, lookback, pred_len, norm,
                  epi_min=None, batch_size=32):
    """Build DataLoader for a split with optional epiweek mask (COVID-strict)."""
    if epi_min is not None:
        sub = df.copy()
        sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min),
                "split"] = "_excluded"
        ds_df = sub
    else:
        ds_df = df
    ds = WeeklyMultiHorizonDataset(ds_df, split_name, norm,
                                   lookback=lookback, pred_len=pred_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), len(ds)


@torch.no_grad()
def _get_point_predictions_raw(model, loader, target_mean, target_std, device):
    """Forward pass → [N, H] raw-scale predictions and ground-truth."""
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        pred = model(x)
        preds.append(pred.cpu().numpy())
        ys.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    preds_raw = preds * target_std + target_mean
    ys_raw = ys * target_std + target_mean
    return preds_raw, ys_raw


# ─── Persistence ───────────────────────────────────────────────────────────


def _persistence_predictions(loader, target_mean: float, target_std: float,
                             pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    """y_{t+h} = y_t prediction for h=1..pred_len.

    The dataset returns (x, y) with x[:, -1, 0] = y_t (target at last input step,
    z-scored). Persistence forecast = x[:, -1, 0] repeated H times.
    Returns: (preds_raw [N, H], ys_raw [N, H]).
    """
    preds, ys = [], []
    for x, y in loader:
        last = x[:, -1, 0].numpy()                          # [B], z-scored target
        rep = np.repeat(last[:, None], pred_len, axis=1)    # [B, H]
        preds.append(rep)
        ys.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    preds_raw = preds * target_std + target_mean
    ys_raw = ys * target_std + target_mean
    return preds_raw, ys_raw


# ─── WIS scoring ───────────────────────────────────────────────────────────


def _score_split(quantile_forecasts: dict, y_true: np.ndarray) -> dict:
    """Per-horizon and averaged WIS + coverage + decomposition."""
    N, H = y_true.shape
    wis_per_h = []
    disp_per_h, under_per_h, over_per_h = [], [], []
    for h in range(H):
        qf_h = {q: quantile_forecasts[q][:, h] for q in quantile_forecasts}
        y_h = y_true[:, h]
        w = wis(y_h, qf_h)
        wis_per_h.append(float(w.mean()))
        parts = wis_decomposed(y_h, qf_h)
        disp_per_h.append(float(parts["dispersion"].mean()))
        under_per_h.append(float(parts["under"].mean()))
        over_per_h.append(float(parts["over"].mean()))

    # Coverage over all horizons (flatten obs × H)
    qf_flat = {q: quantile_forecasts[q].reshape(-1) for q in quantile_forecasts}
    y_flat = y_true.reshape(-1)
    cov50 = coverage(y_flat, qf_flat, alpha=0.5)
    cov95 = coverage(y_flat, qf_flat, alpha=0.05)

    return {
        "n": int(N),
        "wis_per_horizon": wis_per_h,
        "wis_avg": float(np.mean(wis_per_h)),
        "wis_decomposed": {
            "dispersion_per_horizon": disp_per_h,
            "under_per_horizon": under_per_h,
            "over_per_horizon": over_per_h,
            "dispersion_avg": float(np.mean(disp_per_h)),
            "under_avg": float(np.mean(under_per_h)),
            "over_avg": float(np.mean(over_per_h)),
        },
        "coverage_50": cov50,
        "coverage_95": cov95,
    }


# ─── Baseline runners ──────────────────────────────────────────────────────


def run_persistence(df, norm, device: str, lookback: int = 104, pred_len: int = 4) -> dict:
    """Persistence: y_{t+h} = y_t. Val h-specific residual → empirical quantile."""
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    # Val predictions + residuals (per horizon)
    val_loader, n_val = _build_loader(df, "val", lookback, pred_len, norm)
    val_preds, val_y = _persistence_predictions(val_loader, target_mean, target_std, pred_len)
    val_residuals = val_y - val_preds                          # [N, H]
    val_residuals_per_h = [val_residuals[:, h] for h in range(pred_len)]

    # h-specific residual quantile forecasts on each split
    results = {"baseline": "persistence", "cfg_name": "y_{t+h}=y_t", "splits": {}}

    for split_label, epi_min in [("val", None),
                                  ("test_full", None),
                                  ("test_strict", COVID_STRICT_START_EPIWEEK)]:
        split_name = "val" if split_label == "val" else "test"
        loader, n = _build_loader(df, split_name, lookback, pred_len, norm,
                                  epi_min=epi_min if split_label == "test_strict" else None)
        preds, y_true = _persistence_predictions(loader, target_mean, target_std, pred_len)
        qf = residual_quantiles_h_specific(preds, val_residuals_per_h)
        results["splits"][split_label] = _score_split(qf, y_true)
        print(f"  [persistence] {split_label:11s} n={n}  "
              f"WIS_avg={results['splits'][split_label]['wis_avg']:.4f}  "
              f"cov50={results['splits'][split_label]['coverage_50']:.3f}  "
              f"cov95={results['splits'][split_label]['coverage_95']:.3f}")
    return results


def run_ensemble_baseline(
    df, norm, device: str, baseline_name: str, model_dir_root: Path,
    cfg_name: str, ckpt_file: str, build_fn,
) -> dict:
    """Ensemble Gaussian fit baseline (DLinear, N-BEATS).

    5 seeds → 5 predictions per (obs, horizon) → Gaussian fit → 23-quantile.
    """
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    cfg_dir = model_dir_root / cfg_name
    # Read cfg from seed42 results.json
    res = json.load(open(cfg_dir / "seed42" / "results.json"))
    cfg = res["config"]
    lookback = cfg.get("seq_len", cfg.get("lookback", 104))
    pred_len = cfg["pred_len"]

    # Build loaders ONCE per split (same across seeds since data is fixed)
    splits_data = {}
    for split_label, epi_min in [("val", None),
                                  ("test_full", None),
                                  ("test_strict", COVID_STRICT_START_EPIWEEK)]:
        split_name = "val" if split_label == "val" else "test"
        loader, n = _build_loader(
            df, split_name, lookback, pred_len, norm,
            epi_min=epi_min if split_label == "test_strict" else None,
        )
        splits_data[split_label] = {"loader": loader, "n": n}

    # Run inference per seed, accumulate predictions
    per_split_preds: dict[str, list] = {k: [] for k in splits_data}
    y_per_split: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        ckpt_path = cfg_dir / f"seed{seed}" / ckpt_file
        if not ckpt_path.exists():
            print(f"  [{baseline_name} seed={seed}] SKIP — missing {ckpt_path}")
            continue
        model = build_fn(cfg).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=True)
        for split_label, dat in splits_data.items():
            preds_raw, y_raw = _get_point_predictions_raw(
                model, dat["loader"], target_mean, target_std, device,
            )
            per_split_preds[split_label].append(preds_raw)
            if split_label not in y_per_split:
                y_per_split[split_label] = y_raw   # same across seeds
        print(f"  [{baseline_name} seed={seed}] inference done")

    # Ensemble Gaussian fit + WIS
    results = {"baseline": baseline_name, "cfg_name": cfg_name, "splits": {},
               "n_seeds": len(per_split_preds["val"])}
    for split_label, preds_list in per_split_preds.items():
        members = np.stack(preds_list, axis=0)        # [S, N, H]
        qf = ensemble_gaussian_quantiles(members)
        y_true = y_per_split[split_label]
        results["splits"][split_label] = _score_split(qf, y_true)
        n = splits_data[split_label]["n"]
        print(f"  [{baseline_name}] {split_label:11s} n={n}  "
              f"WIS_avg={results['splits'][split_label]['wis_avg']:.4f}  "
              f"cov50={results['splits'][split_label]['coverage_50']:.3f}  "
              f"cov95={results['splits'][split_label]['coverage_95']:.3f}")
    return results


def _build_dlinear(cfg):
    return DLinearForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        moving_avg=cfg["moving_avg"], individual=cfg["individual"],
    )


def _build_nbeats(cfg):
    return NBeatsForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        hidden=cfg["hidden"], n_blocks=cfg["n_blocks"], n_layers=cfg["n_layers"],
        target_only=cfg.get("target_only", False),
    )


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", nargs="+",
                    default=["persistence", "dlinear", "nbeats"],
                    choices=["persistence", "dlinear", "nbeats"])
    ap.add_argument("--device", default="cpu",
                    help="Inference device. CPU is fine for these small models "
                    "(default cpu so Phase C on GPU 1 is undisturbed)")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)

    if "persistence" in args.baselines:
        print("\n=== Persistence ===")
        out = run_persistence(df, norm, args.device)
        out_path = OUT_ROOT / "persistence" / "wis_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  Saved: {out_path.relative_to(_ROOT)}")

    if "dlinear" in args.baselines:
        print("\n=== DLinear (5-seed ensemble Gaussian) ===")
        out = run_ensemble_baseline(
            df, norm, args.device, "dlinear",
            _ROOT / "runs" / "dlinear_final",
            "ma13_indF_lr2e-03", "dlinear_best.pt", _build_dlinear,
        )
        out_path = OUT_ROOT / "dlinear" / "wis_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  Saved: {out_path.relative_to(_ROOT)}")

    if "nbeats" in args.baselines:
        print("\n=== N-BEATS (5-seed ensemble Gaussian) ===")
        out = run_ensemble_baseline(
            df, norm, args.device, "nbeats",
            _ROOT / "runs" / "nbeats_final",
            "nb24_h512_lr5e-04", "nbeats_best.pt", _build_nbeats,
        )
        out_path = OUT_ROOT / "nbeats" / "wis_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  Saved: {out_path.relative_to(_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
