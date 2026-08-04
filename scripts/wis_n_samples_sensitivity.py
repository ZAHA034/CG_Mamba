"""n_samples sensitivity for MC Dropout (PLAN J.7 Q2).

Re-runs CG-Mamba MC Dropout (Phase C d=0.1, top1 cell, 5 seeds) at
n_samples ∈ {50, 100, 200}. Verifies WIS stability across n.

Cheap on GPU 1 (~10-20 min total).
"""
from __future__ import annotations

import argparse
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

from src.models.cg_forecaster import CGForecaster
from src.utils.config import CGMambaConfig
from src.utils.checkpoints import load_fitted_hmm
from src.data.loader import (
    load_dataset_csv, load_norm_params, MultiHorizonDataset, collate_dict,
)
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES
from src.eval.quantile_predictions import _dropout_train_mode

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_DIR = _ROOT / "runs" / "wis_n_samples_sensitivity"
COVID_STRICT_START_EPIWEEK = 202240
SEEDS = (42, 123, 456, 789, 1024)

CG_TOP1_HP = {"gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104,
              "hmm_lr_ratio": 0.01, "state_embed_lr_ratio": 0.01, "env_lr_ratio": 0.001}
OTHER_LR_BASE = 1e-4
HMM_DIR_TEMPLATE = _ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed{seed}"
ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"
PHASE_C_TEMPLATE = _ROOT / "runs" / "m1_8_stage3_train" / "wis_phase_c_cg_mamba_d{d}_s{seed}_stage3" / "best.pt"


def build_cg_mamba(dropout, seed, device):
    cfg = dataclasses.replace(
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
    m.prepare_for_stage2(load_fitted_hmm(hmm_dir))
    if ENV_CKPT.exists():
        state = torch.load(ENV_CKPT, map_location=device, weights_only=True)
        m.env_module.encoder.load_state_dict(state)
    ckpt_p = Path(str(PHASE_C_TEMPLATE).format(d=dropout, seed=seed))
    sd = torch.load(ckpt_p, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    m.load_state_dict(sd, strict=True)
    return m


@torch.no_grad()
def mc_inference(model, loader, n, target_mean, target_std, device):
    model.eval()
    all_s, y_collect = [], None
    with _dropout_train_mode(model):
        for _ in range(n):
            preds, ys = [], []
            for batch in loader:
                x = batch["x"].to(device); env = batch["env"].to(device); y = batch["y"]
                preds.append(model(x, env).cpu().numpy())
                ys.append(y.cpu().numpy())
            all_s.append(np.concatenate(preds))
            if y_collect is None:
                y_collect = np.concatenate(ys)
    samples = np.stack(all_s) * target_std + target_mean
    y_raw = y_collect * target_std + target_mean
    return samples, y_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-list", type=int, nargs="+", default=[50, 100, 200])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    # test_strict loader
    sub = df.copy()
    sub.loc[(sub["split"] == "test") & (sub["epiweek"] < COVID_STRICT_START_EPIWEEK),
            "split"] = "_excluded"
    ds = MultiHorizonDataset(sub, "test", 104, (1, 2, 3, 4), norm)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_dict)

    results = {n: [] for n in args.n_list}
    for n in args.n_list:
        for seed in SEEDS:
            model = build_cg_mamba(0.1, seed, args.device)
            samples, y = mc_inference(model, loader, n, target_mean, target_std, args.device)
            qf = {q: np.quantile(samples, q, axis=0) for q in REQUIRED_QUANTILES}
            wis_per_h = []
            for h in range(4):
                qh = {q: qf[q][:, h] for q in qf}
                wis_per_h.append(float(wis(y[:, h], qh).mean()))
            qf_flat = {q: qf[q].reshape(-1) for q in qf}
            y_flat = y.reshape(-1)
            wis_avg = float(np.mean(wis_per_h))
            c95 = coverage(y_flat, qf_flat, alpha=0.05)
            results[n].append({"seed": seed, "wis_avg": wis_avg, "cov95": c95,
                              "wis_per_horizon": wis_per_h})
            print(f"  n={n:>3d}  seed={seed:>4d}  WIS={wis_avg:.4f}  cov95={c95:.3f}")

    # Aggregate
    summary = {}
    for n, runs in results.items():
        wis_arr = np.array([r["wis_avg"] for r in runs])
        cov_arr = np.array([r["cov95"] for r in runs])
        summary[n] = {
            "wis_avg_mean": float(wis_arr.mean()),
            "wis_avg_std": float(wis_arr.std(ddof=1)),
            "cov95_mean": float(cov_arr.mean()),
        }

    out = {
        "baseline": "cg_mamba_mc_dropout_phase_c_d0.1",
        "split": "test_strict",
        "n_samples_grid": args.n_list,
        "per_seed": {str(n): runs for n, runs in results.items()},
        "summary": {str(n): s for n, s in summary.items()},
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))

    print("\n=== n_samples sensitivity summary (test_strict) ===")
    print(f"{'n':>4s}  {'WIS mean':>10s}  {'± std':>8s}  {'cov95':>8s}")
    for n in args.n_list:
        s = summary[n]
        print(f"{n:>4d}  {s['wis_avg_mean']:>10.4f}  {s['wis_avg_std']:>8.4f}  {s['cov95_mean']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
