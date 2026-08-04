"""Eval-only utility: for every HPO Phase 2 Stage A ckpt, measure per-horizon
raw MAE on (val, test_full, test_strict_w/o_COVID) splits.

Output: runs/m1_9_hpo_phase2/eval_covid_mask.csv (192 rows)
        Columns: cell, base, hmm_lr_ratio, state_embed_lr_ratio, env_lr_ratio,
                 val_h1, val_h2, val_h3, val_h4, val_avg,
                 test_full_h1..h4, test_full_avg,
                 test_strict_h1..h4, test_strict_avg

v2.1.7-A+ : COVID strict mask = exclude 2020-21 + 2021-22 (W40-2020 ~ W39-2022).
            Test strict = W40-2022 ~ W35-2025 (152 rows).
            Justification: 두 시즌 모두 ili_weighted_pct std anomalously low
            (0.39, 0.70 vs pre-COVID baseline ~1.6).

Re-uses model+ckpt loading from m2_1_final._eval_per_horizon_raw_mae pattern.
NO training — eval only. ETA ~10min for 192 cells.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

from src.utils.config import CGMambaConfig
from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.data.loader import MultiHorizonDataset, load_dataset_csv, load_norm_params

OTHER_LR_BASE = 1e-4
CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
HMM_DIR_TMPL = str(_ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}")

HPO_ROOT = _ROOT / "runs/m1_9_hpo_phase2"
STAGE3_ROOT = _ROOT / "runs/m1_8_stage3_train"
OUT_CSV = HPO_ROOT / "eval_covid_mask.csv"

COVID_STRICT_START_EPIWEEK = 202240   # W40-2022 — strict cutoff (exclude 2020-21 + 2021-22)


def _build_cfg_from_row(row) -> CGMambaConfig:
    return dataclasses.replace(
        CGMambaConfig(),
        stage2_gate_lr=float(row["base_gate_lr"]),
        stage2_backbone_lr=float(row["base_backbone_lr"]),
        lookback=int(row["base_lookback"]),
        seed=int(row["seed"]),
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * float(row["hmm_lr_ratio"]),
        stage3_state_embed_lr=OTHER_LR_BASE * float(row["state_embed_lr_ratio"]),
        stage3_env_lr=OTHER_LR_BASE * float(row["env_lr_ratio"]),
    )


def _per_horizon_raw_mae(model, loader, target_mean: float, target_std: float, device):
    model.eval()
    per_h_sum = None
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            y = batch["y"].to(device)
            pred = model(x, env)
            pred_raw = pred * target_std + target_mean
            y_raw = y * target_std + target_mean
            per_h = (pred_raw - y_raw).abs().sum(dim=0)
            per_h_sum = per_h if per_h_sum is None else per_h_sum + per_h
            n += y.size(0)
    return (per_h_sum / max(n, 1)).cpu().tolist(), n


def _build_split_loader(df, split_name, cfg, norm, target_mean, target_std,
                        epi_min=None, epi_max=None):
    """Build a loader over a filtered subset of df (for COVID strict mask).

    To exclude COVID seasons, we filter `df` by epiweek BEFORE building
    MultiHorizonDataset. Note: gap-aware loader will naturally drop windows
    that straddle the boundary.
    """
    if epi_min is not None or epi_max is not None:
        # filter, keeping a buffer of `lookback` weeks before the cut (predictors only)
        if epi_min is not None:
            # keep `lookback` rows before epi_min for lookback continuity
            keep_idx = (df["epiweek"] >= epi_min) | (df["split"] != split_name)
            # to allow lookback to span before, keep predecessor rows of same df
            # MultiHorizonDataset only emits windows where target rows ∈ split_name
            sub = df.copy()
            sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min), "split"] = "covid_excluded"
        if epi_max is not None:
            sub.loc[(sub["split"] == split_name) & (sub["epiweek"] > epi_max), "split"] = "covid_excluded"
        ds_df = sub
    else:
        ds_df = df
    horizons = tuple(cfg.horizons)
    ds = MultiHorizonDataset(ds_df, split_name, cfg.lookback, horizons, norm)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    return loader, len(ds)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Eval Stage A / Stage B ckpts on (val, test_full, test_strict).")
    ap.add_argument("--summary-csv", default=str(HPO_ROOT / "hpo_summary.csv"),
                    help="Input summary CSV (hpo_summary.csv for Stage A or hpo_summary_final.csv for Stage B)")
    ap.add_argument("--out-csv", default=str(OUT_CSV),
                    help="Output CSV path (default: eval_covid_mask.csv)")
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    out_csv_path = Path(args.out_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"{summary_csv} missing.")

    # Load all Stage A rows
    import pandas as pd
    df_hpo = pd.read_csv(summary_csv)
    df_hpo = df_hpo[df_hpo["ok"] == True].reset_index(drop=True)
    print(f"Eval over {len(df_hpo)} Stage A cells")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_data = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    out_rows = []
    for i, row in df_hpo.iterrows():
        seed = int(row["seed"])
        cfg = _build_cfg_from_row(row)
        s3_dir = STAGE3_ROOT / row["run_name"]
        ckpt_path = s3_dir / "best.pt"
        if not ckpt_path.exists():
            print(f"  [{i+1:3d}/{len(df_hpo)}] SKIP {row['run_name']} (best.pt missing)")
            continue

        hmm = load_fitted_hmm(Path(HMM_DIR_TMPL.format(seed=seed)))
        model = CGForecaster(cfg)
        model.prepare_for_stage2(hmm)
        # Unfreeze _A/_means so state_dict keys match Stage 3 ckpt
        model.phase_module._unfreeze_for_stage3()

        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict):
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            elif "model" in sd:
                sd = sd["model"]
        model.load_state_dict(sd, strict=True)
        model = model.to(device).eval()

        # Build 3 loaders: val / test_full / test_strict (epiweek >= 202240)
        val_loader, n_val = _build_split_loader(df_data, "val", cfg, norm, target_mean, target_std)
        test_full_loader, n_test_full = _build_split_loader(df_data, "test", cfg, norm, target_mean, target_std)
        test_strict_loader, n_test_strict = _build_split_loader(
            df_data, "test", cfg, norm, target_mean, target_std,
            epi_min=COVID_STRICT_START_EPIWEEK,
        )

        val_mae, _ = _per_horizon_raw_mae(model, val_loader, target_mean, target_std, device)
        test_full_mae, _ = _per_horizon_raw_mae(model, test_full_loader, target_mean, target_std, device)
        test_strict_mae, _ = _per_horizon_raw_mae(model, test_strict_loader, target_mean, target_std, device)

        out_rows.append({
            "cell": row["cell"],
            "run_name": row["run_name"],
            "base_gate_lr": row["base_gate_lr"],
            "base_backbone_lr": row["base_backbone_lr"],
            "base_lookback": row["base_lookback"],
            "hmm_lr_ratio": row["hmm_lr_ratio"],
            "state_embed_lr_ratio": row["state_embed_lr_ratio"],
            "env_lr_ratio": row["env_lr_ratio"],
            "seed": seed,
            "stage3_best_val": row["stage3_best_val"],     # original val_total (loss = mse + 0.3*mase, z-scored)
            "stage3_test_mse": row["stage3_test_mse"],     # original test mse (z-scored)
            **{f"val_h{h}": v for h, v in zip([1,2,3,4], val_mae)},
            "val_avg": float(np.mean(val_mae)),
            "n_val": n_val,
            **{f"test_full_h{h}": v for h, v in zip([1,2,3,4], test_full_mae)},
            "test_full_avg": float(np.mean(test_full_mae)),
            "n_test_full": n_test_full,
            **{f"test_strict_h{h}": v for h, v in zip([1,2,3,4], test_strict_mae)},
            "test_strict_avg": float(np.mean(test_strict_mae)),
            "n_test_strict": n_test_strict,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(df_hpo)}] {row['run_name']}  "
                  f"val_avg={out_rows[-1]['val_avg']:.4f}  "
                  f"test_full_avg={out_rows[-1]['test_full_avg']:.4f}  "
                  f"test_strict_avg={out_rows[-1]['test_strict_avg']:.4f}")

    # Save
    if out_rows:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with out_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nSaved: {out_csv_path} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
