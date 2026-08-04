"""Ablation A3 — emission-aware rollout ON/OFF (post-hoc, no retrain).

Three conditions, all using the same trained M2.1 top1 checkpoints:
  A3-full:       Emission-aware rollout (current architecture)
  A3-transition: Transition-only rollout (T^h, no emission reweighting)
  A3-uniform:    Uniform gamma (conf=0 → eff_gate=1.0, no phase in decoder)

Because LOGIC-1 handles uniform gamma by design (conf=0 → phase-agnostic fallback),
post-hoc ablation is valid without retraining.

For each condition × 5 seeds:
  - MAE per horizon (test_strict)
  - Method F WIS + Cov95
  - Conformal WIS + Cov95 (for reference)

Output: runs/ablation_a3/ablation_a3_results.json
        runs/ablation_a3/ablation_a3_summary.csv
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.models.cg_forecaster import CGForecaster
from src.utils.config import CGMambaConfig
from src.utils.checkpoints import load_fitted_hmm
from src.data.loader import (
    load_dataset_csv, load_norm_params, MultiHorizonDataset, collate_dict,
)
from src.eval.wis import wis, coverage
from src.eval.hmm_interval import compute_decomposition, method_f_predict_quantiles

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_DIR = _ROOT / "runs" / "ablation_a3"
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
    sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min), "split"] = "_excluded"
    return sub


def transition_only_rollout(gamma_T: torch.Tensor, A: torch.Tensor, H: int) -> torch.Tensor:
    """T^h rollout without emission reweighting."""
    out = []
    gamma_prev = gamma_T
    for _ in range(H):
        gamma_new = gamma_prev @ A
        gamma_new = gamma_new / gamma_new.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        out.append(gamma_new)
        gamma_prev = gamma_new
    return torch.stack(out, dim=1)


def uniform_rollout(gamma_T: torch.Tensor, H: int, K: int) -> torch.Tensor:
    """Uniform gamma for all horizons — removes all phase info from decoder."""
    B = gamma_T.shape[0]
    uniform = torch.ones(B, K, device=gamma_T.device, dtype=gamma_T.dtype) / K
    return uniform.unsqueeze(1).expand(B, H, K).contiguous()


@torch.no_grad()
def forward_with_ablation(model, loader, device, mode="full"):
    """Forward pass with ablated rollout.

    mode: "full" | "transition" | "uniform"
    """
    model.eval()
    mus_list, gammas_list, ys_list, eps_list = [], [], [], []

    for batch in loader:
        x = batch["x"].to(device)
        env = batch["env"].to(device)
        y = batch["y"].cpu().numpy()

        # Run full forward up to encoder, then manually compute decoder
        # We need to intercept gamma_all before the decoder
        pm = model.phase_module

        # Steps 1-5 of CGForecaster.forward (encoder path)
        x_phase = x[:, :, :model.cfg.V_hmm_raw]
        gate_phase, phase_post = pm(x_phase)
        gate_env = model.env_module(env)
        x_truncated = x[:, 1:, :]
        env_truncated_g = gate_env[:, 1:, :]
        context_vec = gate_phase * env_truncated_g
        fused = model.encoder(x_truncated, context_vec=context_vec)

        # Step 6: compute gamma_all based on ablation mode
        gamma_last = phase_post[:, -1, :]
        W = min(model.cfg.rollout_window, x_phase.shape[1])
        x_window = x_phase[:, -W:, :]
        last_value_normalized = x[:, -1, 0]  # ILI_TARGET_IDX = 0

        max_h = model.decoder.max_horizon
        K = pm.K

        if mode == "full":
            gamma_all = pm.rollout(gamma_last, x_window, H=max_h)
        elif mode == "transition":
            gamma_all = transition_only_rollout(gamma_last, pm._A, max_h)
        elif mode == "uniform":
            gamma_all = uniform_rollout(gamma_last, max_h, K)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        predictions = model.decoder(
            encoder_out=fused,
            last_value_normalized=last_value_normalized,
            gamma_all=gamma_all,
            state_embeddings=pm.state_embeddings,
        )

        mus_list.append(predictions.cpu().numpy())
        gammas_list.append(gamma_all.cpu().numpy())
        ys_list.append(y)
        if "target_epiweeks" in batch:
            eps_list.extend(batch["target_epiweeks"])

    mus = np.concatenate(mus_list, axis=0)
    gammas = np.concatenate(gammas_list, axis=0)
    ys = np.concatenate(ys_list, axis=0)
    return mus, gammas, ys, eps_list


def compute_mae(mu_z, y_z, target_std):
    """MAE in raw ILI % per horizon."""
    err = np.abs(mu_z - y_z) * target_std
    return err.mean(axis=0)


def compute_wis_cov(quantiles_raw, y_raw):
    """WIS and Cov95 per horizon."""
    N, H = y_raw.shape
    wis_h, cov_h = [], []
    for h in range(H):
        qf_h = {q: quantiles_raw[q][:, h] for q in quantiles_raw}
        y_h = y_raw[:, h]
        wis_h.append(float(wis(y_h, qf_h).mean()))
        cov_h.append(coverage(y_h, qf_h, alpha=0.05))
    return wis_h, cov_h


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    modes = ["full", "transition", "uniform"]
    all_results = {m: [] for m in modes}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        model, cfg = build_model(seed, args.device)
        pm = model.phase_module
        mu_k_ili = pm._means[:, 0].cpu().numpy()
        sigma2_k_ili = pm._covs[:, 0, 0].cpu().numpy()

        # Build val + test_strict loaders
        loaders = {}
        for split_label, epi_min in [("val", None), ("test_strict", COVID_STRICT_START_EPIWEEK)]:
            split_name = "val" if split_label == "val" else "test"
            ds_df = _mask_df(df, split_name, epi_min)
            ds = MultiHorizonDataset(ds_df, split_name, cfg.lookback,
                                     tuple(cfg.horizons), norm)
            loaders[split_label] = DataLoader(ds, batch_size=32, shuffle=False,
                                              num_workers=0, collate_fn=collate_dict)

        for mode in modes:
            print(f"\n  --- Mode: {mode} ---")

            # Forward val (for calibration)
            val_mu_z, val_gamma, val_y_z, _ = forward_with_ablation(
                model, loaders["val"], args.device, mode=mode
            )
            # Forward test_strict
            test_mu_z, test_gamma, test_y_z, test_eps = forward_with_ablation(
                model, loaders["test_strict"], args.device, mode=mode
            )

            # MAE (raw)
            mae_per_h = compute_mae(test_mu_z, test_y_z, target_std)
            mae_avg = float(mae_per_h.mean())
            print(f"    MAE: {' '.join(f'h{i+1}={v:.4f}' for i, v in enumerate(mae_per_h))} avg={mae_avg:.4f}")

            # Method F: end-to-end (z-scored inputs, raw quantile output)
            test_y_raw = test_y_z * target_std + target_mean

            quantiles_test_raw, mf_meta = method_f_predict_quantiles(
                mu_CGM_test=test_mu_z,
                gamma_all_test=test_gamma,
                mu_CGM_val=val_mu_z,
                gamma_all_val=val_gamma,
                y_val=val_y_z,
                mu_k_ili=mu_k_ili,
                sigma2_k_ili=sigma2_k_ili,
                target_mean=target_mean,
                target_std=target_std,
                mode="gaussian",
            )

            wis_per_h, cov_per_h = compute_wis_cov(quantiles_test_raw, test_y_raw)
            wis_avg = float(np.mean(wis_per_h))
            cov_avg = float(np.mean(cov_per_h))
            print(f"    WIS: {' '.join(f'h{i+1}={v:.4f}' for i, v in enumerate(wis_per_h))} avg={wis_avg:.4f}")
            print(f"    Cov95: {' '.join(f'h{i+1}={v:.3f}' for i, v in enumerate(cov_per_h))} avg={cov_avg:.3f}")

            result = {
                "seed": seed,
                "mode": mode,
                "n_test": len(test_mu_z),
                "mae_per_h": mae_per_h.tolist(),
                "mae_avg": mae_avg,
                "wis_per_h": wis_per_h,
                "wis_avg": wis_avg,
                "cov95_per_h": cov_per_h,
                "cov95_avg": cov_avg,
            }
            all_results[mode].append(result)

    # Aggregate
    print(f"\n{'='*60}")
    print("SUMMARY (5-seed mean ± std)")
    print(f"{'='*60}")

    summary_rows = []
    for mode in modes:
        seeds_data = all_results[mode]
        mae_arr = np.array([r["mae_avg"] for r in seeds_data])
        wis_arr = np.array([r["wis_avg"] for r in seeds_data])
        cov_arr = np.array([r["cov95_avg"] for r in seeds_data])

        mae_per_h_arr = np.array([r["mae_per_h"] for r in seeds_data])
        wis_per_h_arr = np.array([r["wis_per_h"] for r in seeds_data])
        cov_per_h_arr = np.array([r["cov95_per_h"] for r in seeds_data])

        print(f"\n  {mode:12s}: MAE={mae_arr.mean():.4f}±{mae_arr.std():.4f}  "
              f"WIS={wis_arr.mean():.4f}±{wis_arr.std():.4f}  "
              f"Cov95={cov_arr.mean():.3f}±{cov_arr.std():.3f}")
        for h in range(4):
            print(f"    h={h+1}: MAE={mae_per_h_arr[:,h].mean():.4f}±{mae_per_h_arr[:,h].std():.4f}  "
                  f"WIS={wis_per_h_arr[:,h].mean():.4f}±{wis_per_h_arr[:,h].std():.4f}  "
                  f"Cov95={cov_per_h_arr[:,h].mean():.3f}±{cov_per_h_arr[:,h].std():.3f}")

        summary_rows.append({
            "mode": mode,
            "mae_avg_mean": float(mae_arr.mean()),
            "mae_avg_std": float(mae_arr.std()),
            "wis_avg_mean": float(wis_arr.mean()),
            "wis_avg_std": float(wis_arr.std()),
            "cov95_avg_mean": float(cov_arr.mean()),
            "cov95_avg_std": float(cov_arr.std()),
            **{f"mae_h{h+1}_mean": float(mae_per_h_arr[:,h].mean()) for h in range(4)},
            **{f"wis_h{h+1}_mean": float(wis_per_h_arr[:,h].mean()) for h in range(4)},
            **{f"cov95_h{h+1}_mean": float(cov_per_h_arr[:,h].mean()) for h in range(4)},
        })

    # Deltas
    full_mae = summary_rows[0]["mae_avg_mean"]
    full_wis = summary_rows[0]["wis_avg_mean"]
    full_cov = summary_rows[0]["cov95_avg_mean"]
    print(f"\n  Deltas (vs full):")
    for row in summary_rows[1:]:
        d_mae = (row["mae_avg_mean"] - full_mae) / full_mae * 100
        d_wis = (row["wis_avg_mean"] - full_wis) / full_wis * 100
        d_cov = row["cov95_avg_mean"] - full_cov
        print(f"    {row['mode']:12s}: ΔMAE={d_mae:+.1f}%  ΔWIS={d_wis:+.1f}%  ΔCov95={d_cov:+.3f}")

    # Save
    with open(OUT_DIR / "ablation_a3_results.json", "w") as f:
        json.dump({"per_seed": all_results, "summary": summary_rows}, f, indent=2)

    import csv
    with open(OUT_DIR / "ablation_a3_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nSaved: {OUT_DIR / 'ablation_a3_results.json'}")
    print(f"Saved: {OUT_DIR / 'ablation_a3_summary.csv'}")


if __name__ == "__main__":
    main()
