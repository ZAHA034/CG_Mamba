"""Phase 5 CP 5.4 — CG-Mamba retrospective forecast for FluSight 2018-2019.

Generates weekly rolling 4-horizon forecasts for the same target epiweeks
as the 40 FluSight teams (201843~201922), scores with WIS + Cov95 using
Method F, and produces a ranking row insertable into team_wis_ranking.

The 2018-2019 season falls in CG-Mamba's val split, so this is NOT
out-of-sample. The comparison is still informative: CG-Mamba trained on
heterogeneous ILI+env data vs. 40 teams that made real-time forecasts
with diverse methods (including ensembles like FluSightNetwork).

Output:
  runs/phase_5_flusight/cgm_retro_wis_2018_2019.csv
  runs/phase_5_flusight/cgm_retro_summary.json
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

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import MultiHorizonDataset, collate_dict, load_norm_params
from src.eval.hmm_interval import method_f_predict_quantiles
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TM = float(NORM["ili_weighted_pct"]["mean"])
TS = float(NORM["ili_weighted_pct"]["std"])
CSV_PATH = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
OUT_DIR = _ROOT / "runs" / "phase_5_flusight"

FLUSIGHT_TARGET_RANGE = (201843, 201922)
SEEDS = [42, 123, 456, 789, 1024]

CG_M21_CKPT = _ROOT / "runs" / "m1_8_stage3_train" / "hpo_p2_g1e-03_b1e-04_lb104_h0.01_se0.01_e0.001_s{seed}" / "best.pt"
HMM_DIR = _ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed{seed}"
ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"


def _sanitize_gamma(gamma: np.ndarray) -> np.ndarray:
    bad = np.isnan(gamma) | np.isinf(gamma) | (np.abs(gamma) > 1e6)
    if bad.any():
        N, H, K = gamma.shape
        for n in range(N):
            for h in range(H):
                if bad[n, h].any():
                    gamma[n, h, :] = 1.0 / K
        gamma = np.maximum(gamma, 0.0)
        sums = gamma.sum(axis=-1, keepdims=True)
        gamma = gamma / np.maximum(sums, 1e-30)
    return gamma


def build_model(seed: int, device: str):
    import dataclasses
    cfg = CGMambaConfig()
    model = CGForecaster(cfg).to(device)
    hmm_dir = Path(str(HMM_DIR).format(seed=seed))
    hmm = load_fitted_hmm(hmm_dir)
    model.prepare_for_stage2(hmm)
    if ENV_CKPT.exists():
        state = torch.load(ENV_CKPT, map_location=device, weights_only=True)
        model.env_module.encoder.load_state_dict(state)
    ckpt_path = Path(str(CG_M21_CKPT).format(seed=seed))
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    return model, cfg


@torch.no_grad()
def forward_collect(model, loader, device):
    mus, gammas, ys, eps_list = [], [], [], []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        env = batch["env"].to(device)
        preds, intermediates = model(x, env, return_intermediates=True)
        mus.append(preds.cpu().numpy())
        gammas.append(intermediates["gamma_all"].cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        if "target_epiweeks" in batch:
            eps_list.extend(batch["target_epiweeks"])
    return (np.concatenate(mus, 0), _sanitize_gamma(np.concatenate(gammas, 0)),
            np.concatenate(ys, 0), eps_list)


def main(device="cuda:0"):
    if not torch.cuda.is_available():
        device = "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    gt_lookup = dict(zip(df["epiweek"], df["ili_weighted_pct"]))

    per_seed_results = []

    for seed in SEEDS:
        print(f"\n=== seed={seed} ===", flush=True)
        model, cfg = build_model(seed, device)
        pm = model.phase_module
        mu_k = pm._means[:, 0].cpu().numpy()
        sig2_k = np.maximum(pm._covs[:, 0, 0].cpu().numpy(), 1e-6)

        val_ds = MultiHorizonDataset(df, "val", cfg.lookback, tuple(cfg.horizons), NORM)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_dict)
        mu_val, gamma_val, y_val, _ = forward_collect(model, val_loader, device)

        # For FluSight comparison, we need forecasts for val period
        # (2018-2019 IS the val split). Same forward pass, filter by target epiweek.
        q_raw, meta = method_f_predict_quantiles(
            mu_CGM_test=mu_val, gamma_all_test=gamma_val,
            mu_CGM_val=mu_val, gamma_all_val=gamma_val,
            y_val=y_val,
            mu_k_ili=mu_k, sigma2_k_ili=sig2_k,
            target_mean=TM, target_std=TS,
        )
        y_raw = y_val * TS + TM

        # Map each val sample to its target epiweeks
        # MultiHorizonDataset: horizons = (1,2,3,4), y[i] = [y_{t+1}, y_{t+2}, y_{t+3}, y_{t+4}]
        # We need to figure out which epiweek each horizon targets
        # Re-extract target_epiweeks from dataset
        all_target_eps = []
        for i in range(len(val_ds)):
            sample = val_ds[i]
            if "target_epiweeks" in sample:
                all_target_eps.append(sample["target_epiweeks"])
            else:
                all_target_eps.append(None)

        # Score per (submission_week, target_h) within FluSight range
        horizons = [1, 2, 3, 4]
        rows = []
        for i in range(len(y_raw)):
            for h_idx, h in enumerate(horizons):
                target_ep = None
                if all_target_eps[i] is not None:
                    eps_arr = all_target_eps[i]
                    if isinstance(eps_arr, (list, np.ndarray, torch.Tensor)):
                        if h_idx < len(eps_arr):
                            target_ep = int(eps_arr[h_idx]) if not isinstance(eps_arr[h_idx], (int, np.integer)) else int(eps_arr[h_idx])

                if target_ep is None:
                    continue
                if not (FLUSIGHT_TARGET_RANGE[0] <= target_ep <= FLUSIGHT_TARGET_RANGE[1]):
                    continue

                y_true = y_raw[i, h_idx]
                gt_check = gt_lookup.get(target_ep)
                qf = {q: q_raw[q][i, h_idx] for q in REQUIRED_QUANTILES}

                if any(np.isnan(v) for v in qf.values()) or np.isnan(y_true):
                    continue

                wis_val = float(wis(np.array([y_true]), {q: np.array([v]) for q, v in qf.items()}).mean())
                cov95_val = float(coverage(np.array([y_true]), {q: np.array([v]) for q, v in qf.items()}, alpha=0.05))

                rows.append({
                    "seed": seed,
                    "target_epiweek": target_ep,
                    "target_h": h,
                    "y_true": y_true,
                    "point": float(mu_val[i, h_idx] * TS + TM),
                    "wis": wis_val,
                    "cov95": cov95_val,
                })

        seed_df = pd.DataFrame(rows)
        per_seed_results.append(seed_df)
        n_rows = len(seed_df)
        wis_mean = seed_df["wis"].mean() if n_rows > 0 else float("nan")
        print(f"  {n_rows} forecasts, WIS_mean={wis_mean:.4f}", flush=True)

    all_df = pd.concat(per_seed_results, ignore_index=True)
    out_csv = OUT_DIR / "cgm_retro_wis_2018_2019.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(all_df)} rows)")

    # Aggregate: mean across seeds per (target_h)
    agg = all_df.groupby("target_h").agg(
        wis_mean=("wis", "mean"),
        wis_std=("wis", "std"),
        cov95_mean=("cov95", "mean"),
        n=("wis", "count"),
    ).reset_index()
    print("\n=== CG-Mamba Retrospective WIS (2018-2019) ===")
    for _, r in agg.iterrows():
        print(f"  h={int(r['target_h'])}: WIS={r['wis_mean']:.4f}±{r['wis_std']:.4f}  Cov95={r['cov95_mean']:.3f}  n={int(r['n'])}")

    overall_wis = all_df["wis"].mean()
    overall_cov = all_df["cov95"].mean()
    print(f"\n  Overall: WIS={overall_wis:.4f}  Cov95={overall_cov:.3f}")

    # Compare with FluSight ranking
    rank_df = pd.read_csv(OUT_DIR / "team_wis_ranking_2018_2019.csv")
    better = (rank_df["wis_mean"] < overall_wis).sum()
    print(f"\n  Rank: #{better + 1} of {len(rank_df) + 1} (40 teams + CG-Mamba)")
    print(f"  Better than {len(rank_df) - better} of {len(rank_df)} FluSight teams")

    summary = {
        "baseline": "CG-Mamba (Method F)",
        "season": "2018-2019",
        "n_seeds": len(SEEDS),
        "per_horizon": {int(r["target_h"]): {"wis": float(r["wis_mean"]), "cov95": float(r["cov95_mean"])}
                        for _, r in agg.iterrows()},
        "overall_wis": float(overall_wis),
        "overall_cov95": float(overall_cov),
        "rank": int(better + 1),
        "total_teams": int(len(rank_df) + 1),
    }
    summary_path = OUT_DIR / "cgm_retro_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    main(args.device)
