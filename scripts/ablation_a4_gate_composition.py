"""Ablation A4 — Gate composition in encoder (post-hoc, no retrain).

Tests how context_vec composition affects performance:
  A4-and:        gate_phase ⊙ gate_env (current, AND composition)
  A4-phase-only: gate_phase only (env contribution removed)
  A4-env-only:   gate_env only (phase contribution in encoder removed)
  A4-none:       context_vec=None (vanilla Mamba encoder path)
  A4-none+A3-uniform: No phase ANYWHERE (pure vanilla — strongest ablation)

All conditions use the same trained M2.1 checkpoints. The encoder's
ContextGatedMambaBlock was designed with context_vec=None fallback
(disable_gate path, bit-identical to vanilla Mamba).

Output: runs/ablation_a4/ablation_a4_results.json
        runs/ablation_a4/ablation_a4_summary.csv
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
from src.eval.hmm_interval import method_f_predict_quantiles

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_DIR = _ROOT / "runs" / "ablation_a4"
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


def uniform_rollout(gamma_T: torch.Tensor, H: int, K: int) -> torch.Tensor:
    B = gamma_T.shape[0]
    uniform = torch.ones(B, K, device=gamma_T.device, dtype=gamma_T.dtype) / K
    return uniform.unsqueeze(1).expand(B, H, K).contiguous()


@torch.no_grad()
def forward_with_ablation(model, loader, device, encoder_mode="and", decoder_mode="full"):
    """Forward pass with ablated gate composition.

    encoder_mode: "and" | "phase_only" | "env_only" | "none"
    decoder_mode: "full" | "uniform"
    """
    model.eval()
    mus_list, gammas_list, ys_list = [], [], []

    for batch in loader:
        x = batch["x"].to(device)
        env = batch["env"].to(device)
        y = batch["y"].cpu().numpy()

        pm = model.phase_module

        # Step 1: PhaseModule
        x_phase = x[:, :, :model.cfg.V_hmm_raw]
        gate_phase, phase_post = pm(x_phase)

        # Step 2: EnvModule
        gate_env = model.env_module(env)

        # Step 4: compose context_vec based on encoder_mode
        x_truncated = x[:, 1:, :]
        env_truncated_g = gate_env[:, 1:, :]

        if encoder_mode == "and":
            context_vec = gate_phase * env_truncated_g
        elif encoder_mode == "phase_only":
            context_vec = gate_phase
        elif encoder_mode == "env_only":
            context_vec = env_truncated_g
        elif encoder_mode == "none":
            context_vec = None
        else:
            raise ValueError(f"Unknown encoder_mode: {encoder_mode}")

        # Step 5: Encoder
        fused = model.encoder(x_truncated, context_vec=context_vec)

        # Step 6: Decoder
        gamma_last = phase_post[:, -1, :]
        W = min(model.cfg.rollback_window if hasattr(model.cfg, 'rollback_window') else model.cfg.rollout_window, x_phase.shape[1])
        x_window = x_phase[:, -W:, :]
        last_value_normalized = x[:, -1, 0]

        max_h = model.decoder.max_horizon
        K = pm.K

        if decoder_mode == "full":
            gamma_all = pm.rollout(gamma_last, x_window, H=max_h)
        elif decoder_mode == "uniform":
            gamma_all = uniform_rollout(gamma_last, max_h, K)
        else:
            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

        predictions = model.decoder(
            encoder_out=fused,
            last_value_normalized=last_value_normalized,
            gamma_all=gamma_all,
            state_embeddings=pm.state_embeddings,
        )

        mus_list.append(predictions.cpu().numpy())
        gammas_list.append(gamma_all.cpu().numpy())
        ys_list.append(y)

    return (np.concatenate(mus_list, 0),
            np.concatenate(gammas_list, 0),
            np.concatenate(ys_list, 0))


def compute_mae(mu_z, y_z, target_std):
    return (np.abs(mu_z - y_z) * target_std).mean(axis=0)


def compute_wis_cov(quantiles_raw, y_raw):
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

    # Ablation conditions: (name, encoder_mode, decoder_mode)
    conditions = [
        ("A4-and",          "and",        "full"),      # current architecture
        ("A4-phase-only",   "phase_only", "full"),      # no env in encoder
        ("A4-env-only",     "env_only",   "full"),      # no phase in encoder
        ("A4-none",         "none",       "full"),      # vanilla encoder, phase decoder
        ("A4-none+uniform", "none",       "uniform"),   # fully vanilla (no phase anywhere)
    ]

    all_results = {name: [] for name, _, _ in conditions}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        model, cfg = build_model(seed, args.device)
        pm = model.phase_module
        mu_k_ili = pm._means[:, 0].cpu().numpy()
        sigma2_k_ili = pm._covs[:, 0, 0].cpu().numpy()

        loaders = {}
        for split_label, epi_min in [("val", None), ("test_strict", COVID_STRICT_START_EPIWEEK)]:
            split_name = "val" if split_label == "val" else "test"
            ds_df = _mask_df(df, split_name, epi_min)
            ds = MultiHorizonDataset(ds_df, split_name, cfg.lookback,
                                     tuple(cfg.horizons), norm)
            loaders[split_label] = DataLoader(ds, batch_size=32, shuffle=False,
                                              num_workers=0, collate_fn=collate_dict)

        for cond_name, enc_mode, dec_mode in conditions:
            print(f"\n  --- {cond_name} (enc={enc_mode}, dec={dec_mode}) ---")

            val_mu_z, val_gamma, val_y_z = forward_with_ablation(
                model, loaders["val"], args.device,
                encoder_mode=enc_mode, decoder_mode=dec_mode,
            )
            test_mu_z, test_gamma, test_y_z = forward_with_ablation(
                model, loaders["test_strict"], args.device,
                encoder_mode=enc_mode, decoder_mode=dec_mode,
            )

            mae_per_h = compute_mae(test_mu_z, test_y_z, target_std)
            mae_avg = float(mae_per_h.mean())
            print(f"    MAE: {' '.join(f'h{i+1}={v:.4f}' for i, v in enumerate(mae_per_h))} avg={mae_avg:.4f}")

            test_y_raw = test_y_z * target_std + target_mean

            quantiles_test_raw, _ = method_f_predict_quantiles(
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
            print(f"    Cov: {' '.join(f'h{i+1}={v:.3f}' for i, v in enumerate(cov_per_h))} avg={cov_avg:.3f}")

            all_results[cond_name].append({
                "seed": seed, "condition": cond_name,
                "mae_per_h": mae_per_h.tolist(), "mae_avg": mae_avg,
                "wis_per_h": wis_per_h, "wis_avg": wis_avg,
                "cov95_per_h": cov_per_h, "cov95_avg": cov_avg,
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (5-seed mean ± std)")
    print(f"{'='*60}")

    summary_rows = []
    for cond_name, _, _ in conditions:
        data = all_results[cond_name]
        mae_arr = np.array([r["mae_avg"] for r in data])
        wis_arr = np.array([r["wis_avg"] for r in data])
        cov_arr = np.array([r["cov95_avg"] for r in data])
        mae_h = np.array([r["mae_per_h"] for r in data])
        wis_h = np.array([r["wis_per_h"] for r in data])
        cov_h = np.array([r["cov95_per_h"] for r in data])

        print(f"\n  {cond_name:20s}: MAE={mae_arr.mean():.4f}±{mae_arr.std():.4f}  "
              f"WIS={wis_arr.mean():.4f}±{wis_arr.std():.4f}  "
              f"Cov95={cov_arr.mean():.3f}±{cov_arr.std():.3f}")
        for h in range(4):
            print(f"    h={h+1}: MAE={mae_h[:,h].mean():.4f}±{mae_h[:,h].std():.4f}  "
                  f"WIS={wis_h[:,h].mean():.4f}±{wis_h[:,h].std():.4f}  "
                  f"Cov95={cov_h[:,h].mean():.3f}±{cov_h[:,h].std():.3f}")

        summary_rows.append({
            "condition": cond_name,
            "mae_avg_mean": float(mae_arr.mean()), "mae_avg_std": float(mae_arr.std()),
            "wis_avg_mean": float(wis_arr.mean()), "wis_avg_std": float(wis_arr.std()),
            "cov95_avg_mean": float(cov_arr.mean()), "cov95_avg_std": float(cov_arr.std()),
            **{f"mae_h{h+1}_mean": float(mae_h[:,h].mean()) for h in range(4)},
            **{f"wis_h{h+1}_mean": float(wis_h[:,h].mean()) for h in range(4)},
            **{f"cov95_h{h+1}_mean": float(cov_h[:,h].mean()) for h in range(4)},
        })

    # Deltas vs A4-and
    ref = summary_rows[0]
    print(f"\n  Deltas (vs A4-and):")
    for row in summary_rows[1:]:
        d_mae = (row["mae_avg_mean"] - ref["mae_avg_mean"]) / ref["mae_avg_mean"] * 100
        d_wis = (row["wis_avg_mean"] - ref["wis_avg_mean"]) / ref["wis_avg_mean"] * 100
        d_cov = row["cov95_avg_mean"] - ref["cov95_avg_mean"]
        print(f"    {row['condition']:20s}: ΔMAE={d_mae:+.1f}%  ΔWIS={d_wis:+.1f}%  ΔCov95={d_cov:+.3f}")

    with open(OUT_DIR / "ablation_a4_results.json", "w") as f:
        json.dump({"per_seed": all_results, "summary": summary_rows}, f, indent=2)

    import csv
    with open(OUT_DIR / "ablation_a4_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nSaved: {OUT_DIR / 'ablation_a4_results.json'}")
    print(f"Saved: {OUT_DIR / 'ablation_a4_summary.csv'}")


if __name__ == "__main__":
    main()
