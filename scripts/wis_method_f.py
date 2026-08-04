"""WIS measurement via Method F (HMM-Derived Calibrated Intervals).

For each M2.1 top1 cell ckpt (5 seeds):
  1. Forward val/test_full/test_strict → μ_CGM + gamma_all
  2. Method F decomposition (3-component)
  3. Per-horizon calibration s_h from val
  4. Test quantiles → WIS

5 seeds × 3 splits × 23 quantiles. Single-cell only (top1 = global #1).
Single-cell WIS asymmetry handled by paper §IV.X footnote (PLAN J.8 Template 6).

Output:
  runs/wis_method_f/wis_results.json     ← per-seed + aggregated
  runs/wis_method_f/decomposition_temporal.csv  ← for §V.X figure
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
from src.eval.wis import wis, wis_decomposed, coverage
from src.eval.hmm_interval import (
    compute_decomposition, method_f_predict_quantiles,
)

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_DIR = _ROOT / "runs" / "wis_method_f"
COVID_STRICT_START_EPIWEEK = 202240

CG_TOP1_HP = {
    "gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104,
    "hmm_lr_ratio": 0.01, "state_embed_lr_ratio": 0.01, "env_lr_ratio": 0.001,
}
OTHER_LR_BASE = 1e-4
HMM_DIR_TEMPLATE = _ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed{seed}"
ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"
CG_M21_CKPT_TEMPLATE = (
    _ROOT / "runs" / "m1_8_stage3_train" /
    "hpo_p2_g1e-03_b1e-04_lb104_h0.01_se0.01_e0.001_s{seed}" / "best.pt"
)
SEEDS = (42, 123, 456, 789, 1024)


def build_model(seed: int, device: str):
    hp = CG_TOP1_HP
    cfg = dataclasses.replace(
        CGMambaConfig(),
        seed=seed, dropout=0.0, lookback=hp["lookback"],
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
    ckpt_path = Path(str(CG_M21_CKPT_TEMPLATE).format(seed=seed))
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    return model, cfg


def _mask_df(df, split_name, epi_min):
    if epi_min is None:
        return df
    sub = df.copy()
    sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min),
            "split"] = "_excluded"
    return sub


@torch.no_grad()
def forward_collect(model, loader, device):
    """Returns (mu_CGM [N, H] z-scored, gamma_all [N, H, K], y_z [N, H], target_eps list)."""
    model.eval()
    mus, gammas, ys, eps_list = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device); env = batch["env"].to(device)
        y = batch["y"].cpu().numpy()
        preds, intermediates = model(x, env, return_intermediates=True)
        mus.append(preds.cpu().numpy())
        gammas.append(intermediates["gamma_all"].cpu().numpy())
        ys.append(y)
        if "target_epiweeks" in batch:
            eps_list.extend(batch["target_epiweeks"])
    mus = np.concatenate(mus, axis=0)
    gammas = np.concatenate(gammas, axis=0)
    ys = np.concatenate(ys, axis=0)
    return mus, gammas, ys, eps_list


def score_split(quantiles_raw: dict[float, np.ndarray], y_raw: np.ndarray) -> dict:
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--calibration", default="quantile_matching",
                    choices=["quantile_matching", "simple_ratio"])
    ap.add_argument("--mode", default="auto", choices=["auto", "gaussian", "mixture"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    per_seed_results = {}
    decomp_temporal_rows = []   # for §V.X figure

    for seed in args.seeds:
        print(f"\n=== seed={seed} ===")
        model, cfg = build_model(seed, args.device)
        pm = model.phase_module
        mu_k_ili = pm._means[:, 0].cpu().numpy()
        sigma2_k_ili = pm._covs[:, 0, 0].cpu().numpy()

        # Build all three split loaders
        loaders = {}
        for split_label, epi_min in [("val", None),
                                      ("test_full", None),
                                      ("test_strict", COVID_STRICT_START_EPIWEEK)]:
            split_name = "val" if split_label == "val" else "test"
            ds_df = _mask_df(df, split_name, epi_min)
            ds = MultiHorizonDataset(ds_df, split_name, cfg.lookback,
                                     tuple(cfg.horizons), norm)
            loaders[split_label] = DataLoader(ds, batch_size=32, shuffle=False,
                                              num_workers=0, collate_fn=collate_dict)

        # Forward all
        forwards = {}
        for sp, loader in loaders.items():
            mu_z, gamma_all, y_z, eps = forward_collect(model, loader, args.device)
            forwards[sp] = {"mu_z": mu_z, "gamma_all": gamma_all,
                            "y_z": y_z, "eps": eps}
            print(f"  {sp:11s} n={len(mu_z)}")

        # Method F: calibrate on val, predict on each split
        # For each test split, calibrate using val, apply to test
        seed_result = {"seed": seed, "mu_k_ili": mu_k_ili.tolist(),
                       "sigma_k_ili": np.sqrt(sigma2_k_ili).tolist(),
                       "splits": {}}
        for sp in ["val", "test_full", "test_strict"]:
            quantiles_raw, meta = method_f_predict_quantiles(
                mu_CGM_test=forwards[sp]["mu_z"],
                gamma_all_test=forwards[sp]["gamma_all"],
                mu_CGM_val=forwards["val"]["mu_z"],
                gamma_all_val=forwards["val"]["gamma_all"],
                y_val=forwards["val"]["y_z"],
                mu_k_ili=mu_k_ili,
                sigma2_k_ili=sigma2_k_ili,
                target_mean=target_mean, target_std=target_std,
                calibration=args.calibration, mode=args.mode,
            )
            y_raw = forwards[sp]["y_z"] * target_std + target_mean
            score = score_split(quantiles_raw, y_raw)
            score["calibration_meta"] = meta
            seed_result["splits"][sp] = score
            print(f"  [{sp:11s}] WIS_avg={score['wis_avg']:.4f}  "
                  f"cov50={score['coverage_50']:.3f}  "
                  f"cov95={score['coverage_95']:.3f}  "
                  f"mode={meta['quantile_mode']}  "
                  f"s_per_h={[f'{x:.2f}' for x in meta['s_per_h']]}")

        # Save decomposition for temporal figure (test_strict, ALL seeds — for Figure 5 representative-seed selection)
        decomp_ts = compute_decomposition(
            forwards["test_strict"]["mu_z"],
            forwards["test_strict"]["gamma_all"],
            mu_k_ili, sigma2_k_ili,
        )
        for n in range(len(decomp_ts.mu_CGM)):
            for h_idx in range(decomp_ts.mu_CGM.shape[1]):
                ep = forwards["test_strict"]["eps"][n][h_idx] if forwards["test_strict"]["eps"] else -1
                decomp_temporal_rows.append({
                    "seed": seed,
                    "sample_idx": n, "horizon": h_idx + 1,
                    "target_ep": int(ep),
                    "mu_CGM_raw": float(decomp_ts.mu_CGM[n, h_idx] * target_std + target_mean),
                    "mu_HMM_raw": float(decomp_ts.mu_HMM[n, h_idx] * target_std + target_mean),
                    "y_raw": float(forwards["test_strict"]["y_z"][n, h_idx] * target_std + target_mean),
                    "sigma2_within": float(decomp_ts.sigma2_within[n, h_idx]),
                    "sigma2_between_HMM": float(decomp_ts.sigma2_between_HMM[n, h_idx]),
                    "bias_sq": float(decomp_ts.bias_sq[n, h_idx]),
                    "sigma2_total": float(decomp_ts.sigma2_total[n, h_idx]),
                })

        per_seed_results[seed] = seed_result

    # Aggregate across seeds
    print("\n=== Aggregated (5 seeds) ===")
    aggregated = {}
    for sp in ["val", "test_full", "test_strict"]:
        wis_per_h_arr = np.array([per_seed_results[s]["splits"][sp]["wis_per_horizon"]
                                  for s in args.seeds])
        wis_avg_arr = np.array([per_seed_results[s]["splits"][sp]["wis_avg"]
                                for s in args.seeds])
        cov50_arr = np.array([per_seed_results[s]["splits"][sp]["coverage_50"]
                              for s in args.seeds])
        cov95_arr = np.array([per_seed_results[s]["splits"][sp]["coverage_95"]
                              for s in args.seeds])
        n = per_seed_results[args.seeds[0]]["splits"][sp]["n"]
        aggregated[sp] = {
            "n": n,
            "wis_per_horizon_mean": wis_per_h_arr.mean(axis=0).tolist(),
            "wis_per_horizon_std": wis_per_h_arr.std(axis=0, ddof=1).tolist(),
            "wis_avg_mean": float(wis_avg_arr.mean()),
            "wis_avg_std": float(wis_avg_arr.std(ddof=1)),
            "coverage_50_mean": float(cov50_arr.mean()),
            "coverage_95_mean": float(cov95_arr.mean()),
        }
        a = aggregated[sp]
        print(f"  {sp:11s} WIS_avg={a['wis_avg_mean']:.4f}±{a['wis_avg_std']:.4f}  "
              f"cov50={a['coverage_50_mean']:.3f}  cov95={a['coverage_95_mean']:.3f}")

    # Save
    out = {
        "baseline": "cg_mamba_method_f",
        "method": "HMM-Derived Calibrated Intervals (Method F)",
        "ckpt_source": "M2.1 top1 cell (dropout=0.0) — MAE-optimal model preserved",
        "calibration": args.calibration,
        "mode": args.mode,
        "per_seed": {str(s): per_seed_results[s] for s in args.seeds},
        "aggregated": aggregated,
    }
    out_path = OUT_DIR / "wis_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.relative_to(_ROOT)}")

    # Temporal decomposition CSV for §V.X figure (multi-seed)
    if decomp_temporal_rows:
        import csv
        csv_path = OUT_DIR / "decomposition_temporal_5seed.csv"
        fields = list(decomp_temporal_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(decomp_temporal_rows)
        print(f"Saved: {csv_path.relative_to(_ROOT)} ({len(decomp_temporal_rows)} rows, {len(set(r['seed'] for r in decomp_temporal_rows))} seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
