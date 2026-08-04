"""Eval-only utility: for every baseline final ckpt (LSTM/PatchTST/iTransformer),
measure per-horizon raw MAE on (val, test_full, test_strict_w/o_COVID) splits.

v2.1.7-A+ : Mirror of m1_9_eval_covid_mask.py for CG-Mamba Stage A — applied to
the 3 baselines' 5-seed finals so M2.3 Main Results Table can report all 4 model
families on a unified (val, test_full, test_strict) per-horizon basis.

Output: runs/baselines_test_eval.csv
        Columns: model, cfg_name, seed,
                 val_h1..h4, val_avg, n_val,
                 test_full_h1..h4, test_full_avg, n_test_full,
                 test_strict_h1..h4, test_strict_avg, n_test_strict
"""
from __future__ import annotations

import csv
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

from baselines.lstm import WeeklyMultiHorizonDataset, LSTM_FEATURE_COLS  # noqa: E402
from baselines.patchtst import PatchTSTForecaster                       # noqa: E402
from baselines.itransformer import ITransformerForecaster               # noqa: E402
from baselines.dlinear import DLinearForecaster                         # noqa: E402

from src.data.loader import load_dataset_csv, load_norm_params          # noqa: E402

# CM_Mamba import path: same one baselines/lstm.py uses
import sys as _sys
_PARENT = _ROOT.parent
_sys.path.insert(0, str(_PARENT / "CM_Mamba"))
from cm_mamba.baselines.lstm_baseline import LSTMForecaster              # noqa: E402

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_CSV = _ROOT / "runs/baselines_test_eval.csv"

COVID_STRICT_START_EPIWEEK = 202240   # W40-2022 — strict cutoff

# Baseline configurations (winners from Phase 1 / grid)
BASELINES = [
    ("lstm",         "h256_l2_lr5e-04_bs16",  "lstm_best.pt"),
    ("patchtst",     "pl16_dm64_lr5e-04",     "patchtst_best.pt"),
    ("itransformer", "dm256_el4_lr5e-04",     "itransformer_best.pt"),
    ("dlinear",      "ma13_indF_lr2e-03",     "dlinear_best.pt"),
    ("timesnet",     "d64_el2_lr1e-03",       "timesnet_best.pt"),
]
SEEDS = [42, 123, 456, 789, 1024]


def _build_model(model_name: str, cfg: dict):
    if model_name == "lstm":
        return LSTMForecaster(
            enc_in=cfg["enc_in"],
            hidden=cfg["hidden"],
            num_layers=cfg["num_layers"],
            pred_len=cfg["pred_len"],
            dropout=cfg.get("dropout", 0.0),
        )
    elif model_name == "patchtst":
        stride = max(1, int(cfg["patch_len"] * cfg["stride_ratio"]))
        return PatchTSTForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
            d_ff=cfg["d_ff_ratio"] * cfg["d_model"],
            patch_len=cfg["patch_len"], stride=stride,
            dropout=cfg["dropout"],
        )
    elif model_name == "itransformer":
        return ITransformerForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
            d_ff=cfg["d_ff_ratio"] * cfg["d_model"],
            dropout=cfg["dropout"],
            embed=cfg.get("embed", "timeF"), freq=cfg.get("freq", "w"),
        )
    elif model_name == "dlinear":
        return DLinearForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            moving_avg=cfg["moving_avg"], individual=cfg["individual"],
        )
    elif model_name == "timesnet":
        from baselines.timesnet import TimesNetForecaster  # type: ignore
        return TimesNetForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], d_ff=cfg["d_ff"], e_layers=cfg["e_layers"],
            top_k=cfg["top_k"], num_kernels=cfg["num_kernels"], dropout=cfg["dropout"],
        )
    elif model_name == "nbeats":
        from baselines.nbeats import NBeatsForecaster  # type: ignore
        return NBeatsForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            hidden=cfg["hidden"], n_blocks=cfg["n_blocks"], n_layers=cfg["n_layers"],
            target_only=cfg.get("target_only", False),
        )
    elif model_name == "epideep":
        from baselines.epideep import EpiDeepForecaster  # type: ignore
        return EpiDeepForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
            decoder_hidden=cfg["decoder_hidden"],
            alignment_weight=cfg["alignment_weight"],
            dropout=cfg["dropout"],
            target_only=cfg.get("target_only", False),
        )
    elif model_name == "vanilla_mamba":
        from baselines.vanilla_mamba import VanillaMambaForecaster  # type: ignore
        return VanillaMambaForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            d_state=cfg.get("d_state", 16),
            dt_rank=cfg.get("dt_rank", 16),
            expand=cfg.get("expand", 2),
            dropout=cfg.get("dropout", 0.0),
        )
    raise ValueError(model_name)


def _build_split_loader(df, split_name, lookback, pred_len, norm,
                        epi_min=None, epi_max=None, batch_size=32):
    """Build DataLoader for a split with optional epiweek mask (for COVID-strict)."""
    if epi_min is not None or epi_max is not None:
        sub = df.copy()
        if epi_min is not None:
            sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min), "split"] = "_excluded"
        if epi_max is not None:
            sub.loc[(sub["split"] == split_name) & (sub["epiweek"] > epi_max), "split"] = "_excluded"
        ds_df = sub
    else:
        ds_df = df
    ds = WeeklyMultiHorizonDataset(ds_df, split_name, norm, lookback=lookback, pred_len=pred_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), len(ds)


def _per_horizon_raw_mae(model, loader, target_mean: float, target_std: float, device):
    model.eval()
    per_h_sum = None
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            pred_raw = pred * target_std + target_mean
            y_raw = y * target_std + target_mean
            per_h = (pred_raw - y_raw).abs().sum(dim=0)
            per_h_sum = per_h if per_h_sum is None else per_h_sum + per_h
            n += y.size(0)
    return (per_h_sum / max(n, 1)).cpu().tolist(), n


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra-baseline", action="append", default=[],
                    help="Format 'model:cfg_dir:ckpt_file' — extends BASELINES list. Repeatable.")
    ap.add_argument("--out", default=str(OUT_CSV),
                    help="Output CSV path (default: runs/baselines_test_eval.csv)")
    args = ap.parse_args()

    for spec in args.extra_baseline:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--extra-baseline must be 'model:cfg_dir:ckpt_file', got {spec!r}")
        BASELINES.append(tuple(parts))
        print(f"[+] added baseline: {parts[0]} cfg={parts[1]} ckpt={parts[2]}")

    out_csv_path = Path(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    rows = []
    for model_name, cfg_name, ckpt_file in BASELINES:
        family_dir = _ROOT / "runs" / f"{model_name}_final" / cfg_name
        if not family_dir.exists():
            print(f"  [{model_name}] dir missing: {family_dir}")
            continue
        for seed in SEEDS:
            seed_dir = family_dir / f"seed{seed}"
            ckpt = seed_dir / ckpt_file
            res_json = seed_dir / "results.json"
            if not ckpt.exists():
                print(f"  [{model_name} seed={seed}] SKIP — {ckpt} missing")
                continue
            cfg = json.loads(res_json.read_text())["config"]

            model = _build_model(model_name, cfg).to(device)
            sd = torch.load(ckpt, map_location=device, weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            model.load_state_dict(sd, strict=True)

            lookback = cfg.get("seq_len", cfg.get("lookback", 104))
            pred_len = cfg["pred_len"]

            val_loader, n_val = _build_split_loader(df, "val", lookback, pred_len, norm)
            test_full_loader, n_test_full = _build_split_loader(df, "test", lookback, pred_len, norm)
            test_strict_loader, n_test_strict = _build_split_loader(
                df, "test", lookback, pred_len, norm,
                epi_min=COVID_STRICT_START_EPIWEEK,
            )

            val_mae, _ = _per_horizon_raw_mae(model, val_loader, target_mean, target_std, device)
            test_full_mae, _ = _per_horizon_raw_mae(model, test_full_loader, target_mean, target_std, device)
            test_strict_mae, _ = _per_horizon_raw_mae(model, test_strict_loader, target_mean, target_std, device)

            row = {
                "model": model_name,
                "cfg_name": cfg_name,
                "seed": seed,
                **{f"val_h{h}": v for h, v in zip([1,2,3,4], val_mae)},
                "val_avg": float(np.mean(val_mae)),
                "n_val": n_val,
                **{f"test_full_h{h}": v for h, v in zip([1,2,3,4], test_full_mae)},
                "test_full_avg": float(np.mean(test_full_mae)),
                "n_test_full": n_test_full,
                **{f"test_strict_h{h}": v for h, v in zip([1,2,3,4], test_strict_mae)},
                "test_strict_avg": float(np.mean(test_strict_mae)),
                "n_test_strict": n_test_strict,
            }
            rows.append(row)
            print(f"  [{model_name} seed={seed}] val_avg={row['val_avg']:.4f}  "
                  f"test_full_avg={row['test_full_avg']:.4f}  "
                  f"test_strict_avg={row['test_strict_avg']:.4f}")

    if rows:
        with out_csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved: {out_csv_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
