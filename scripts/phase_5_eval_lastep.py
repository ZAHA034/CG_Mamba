"""Phase 5 fairness audit — re-evaluate with last-epoch weights (no val leak).

Loads last.pt (saved by phase_5_retrain_lastep.py) and reruns Phase 5
fair evaluation: train-only s_h calibration + common epiweeks with
FluSightNetwork.

Output: runs/phase_5_flusight/cgm_retro_lastep_2018_2019.csv
        runs/phase_5_flusight/cgm_retro_lastep_summary.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from epiweeks import Week

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import MultiHorizonDataset, collate_dict, load_norm_params
from src.eval.hmm_interval import compute_decomposition, calibrate_scale_quantile_matching, construct_quantiles
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES

NORM = load_norm_params(_ROOT / "data/processed/normalization_params.json")
TM = float(NORM["ili_weighted_pct"]["mean"])
TS = float(NORM["ili_weighted_pct"]["std"])

CKPT_TMPL = _ROOT / "runs/m1_8_stage3_train/hpo_p2_g1e-03_b1e-04_lb104_h0.01_se0.01_e0.001_s{seed}_lastep/last.pt"
HMM_TMPL = _ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}"
ENV_CKPT = _ROOT / "runs/m1_7_env_pretrain/env_encoder.pt"

SEEDS = [42, 123, 456, 789, 1024]
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def sanitize_gamma(gamma):
    bad = np.isnan(gamma) | np.isinf(gamma) | (np.abs(gamma) > 1e6)
    if bad.any():
        K = gamma.shape[-1]
        gamma[bad] = 1.0 / K
        gamma = np.maximum(gamma, 0.0)
        gamma = gamma / np.maximum(gamma.sum(axis=-1, keepdims=True), 1e-30)
    return gamma


@torch.no_grad()
def forward_collect(model, loader):
    mus, gammas, ys = [], [], []
    model.eval()
    for batch in loader:
        preds, inter = model(batch["x"].to(device), batch["env"].to(device), return_intermediates=True)
        mus.append(preds.cpu().numpy())
        gammas.append(inter["gamma_all"].cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
    return (np.concatenate(mus, 0), sanitize_gamma(np.concatenate(gammas, 0)),
            np.concatenate(ys, 0))


def main():
    df = pd.read_csv(_ROOT / "data/processed/ili_env_weekly_split.csv")
    teams = pd.read_csv(_ROOT / "runs/phase_5_flusight/team_wis_2018_2019.csv")

    fsn_eps = {h: set(teams[(teams["team"] == "FluSightNetwork") & (teams["target_h"] == h)]["target_epiweek"])
               for h in [1, 2, 3, 4]}

    all_rows = []
    cfg = CGMambaConfig()
    cfg = type(cfg)(**{**cfg.__dict__, "lookback": 104})

    for seed in SEEDS:
        print(f"\n=== seed={seed} ===", flush=True)
        model = CGForecaster(cfg).to(device)
        hmm = load_fitted_hmm(Path(str(HMM_TMPL).format(seed=seed)))
        model.prepare_for_stage2(hmm)
        if ENV_CKPT.exists():
            model.env_module.encoder.load_state_dict(
                torch.load(str(ENV_CKPT), map_location=device, weights_only=True))
        ckpt_path = Path(str(CKPT_TMPL).format(seed=seed))
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=True)
        model.eval()

        pm = model.phase_module
        mu_k = pm._means[:, 0].cpu().numpy()
        sig2_k = np.maximum(pm._covs[:, 0, 0].cpu().numpy(), 1e-6)

        # train s_h calibration
        train_ds = MultiHorizonDataset(df, "train", cfg.lookback, tuple(cfg.horizons), NORM)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_dict)
        mu_tr, gamma_tr, y_tr = forward_collect(model, train_loader)
        decomp_tr = compute_decomposition(mu_tr, gamma_tr, mu_k, sig2_k)
        s_per_h = calibrate_scale_quantile_matching(y_tr, decomp_tr)

        # val forecast (= 2018-2019)
        val_ds = MultiHorizonDataset(df, "val", cfg.lookback, tuple(cfg.horizons), NORM)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_dict)
        mu_val, gamma_val, y_val = forward_collect(model, val_loader)
        decomp_val = compute_decomposition(mu_val, gamma_val, mu_k, sig2_k)
        q_z, mode = construct_quantiles(decomp_val, gamma_val, mu_k, sig2_k, s_per_h)
        q_raw = {q: arr * TS + TM for q, arr in q_z.items()}

        # Build target_epiweeks
        all_target_eps = []
        for i in range(len(val_ds)):
            s = val_ds[i]
            all_target_eps.append(s.get("target_epiweeks", None))

        y_raw = y_val * TS + TM
        for i in range(len(y_raw)):
            if all_target_eps[i] is None:
                continue
            eps_arr = all_target_eps[i]
            for h_idx, h in enumerate([1, 2, 3, 4]):
                if h_idx >= len(eps_arr):
                    continue
                target_ep = int(eps_arr[h_idx])
                if target_ep not in fsn_eps[h]:
                    continue
                y_true = y_raw[i, h_idx]
                qf = {q: np.array([q_raw[q][i, h_idx]]) for q in REQUIRED_QUANTILES}
                if any(np.isnan(v[0]) for v in qf.values()) or np.isnan(y_true):
                    continue
                w = float(wis(np.array([y_true]), qf).mean())
                c = float(coverage(np.array([y_true]), qf, alpha=0.05))
                all_rows.append({"seed": seed, "target_epiweek": target_ep, "target_h": h,
                                 "y_true": y_true, "wis": w, "cov95": c})

        n = len([r for r in all_rows if r["seed"] == seed])
        wm = np.mean([r["wis"] for r in all_rows if r["seed"] == seed])
        print(f"  {n} forecasts, WIS_mean={wm:.4f}, s_per_h={s_per_h.tolist()}")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(_ROOT / "runs/phase_5_flusight/cgm_retro_lastep_2018_2019.csv", index=False)

    overall_wis = float(result_df["wis"].mean())
    overall_cov = float(result_df["cov95"].mean())

    print(f"\n=== CG-Mamba LAST-EPOCH Retrospective (2018-2019) ===")
    for h in [1, 2, 3, 4]:
        sub = result_df[result_df["target_h"] == h]
        print(f"  h={h}: WIS={sub['wis'].mean():.4f}  Cov95={sub['cov95'].mean():.3f}  n={len(sub)}")
    print(f"  Overall: WIS={overall_wis:.4f}  Cov95={overall_cov:.3f}")

    rank_df = pd.read_csv(_ROOT / "runs/phase_5_flusight/team_wis_ranking_2018_2019.csv")
    better = int((rank_df["wis_mean"] < overall_wis).sum())
    print(f"\n  Rank: #{better+1} of {len(rank_df)+1}")

    summary = {
        "method": "CG-Mamba Method F LAST-EPOCH (no val leak)",
        "calibration": "train split (2002-2018)",
        "epiweeks": "common with FluSightNetwork",
        "overall_wis": overall_wis,
        "overall_cov95": overall_cov,
        "rank": better + 1,
        "total": int(len(rank_df) + 1),
        "per_horizon": {int(h): {"wis": float(result_df[result_df["target_h"] == h]["wis"].mean()),
                                  "cov95": float(result_df[result_df["target_h"] == h]["cov95"].mean())}
                        for h in [1, 2, 3, 4]},
    }
    with open(_ROOT / "runs/phase_5_flusight/cgm_retro_lastep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
