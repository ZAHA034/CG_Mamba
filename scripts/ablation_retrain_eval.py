"""Ablation Retrain Evaluation — compute MAE/WIS/Cov95 for retrained ckpts.

Evaluates the 3 from-scratch retrained ablation configs from ablation_retrain.py
on val + test_strict, using each subclass's built-in forward (the ablation is
applied during evaluation exactly as it was during training).

Inputs:
  runs/m1_8_stage3_train/ablation_retrain_<config>_s<seed>_stage3/best.pt
  runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed<seed>/  (HMM ckpts)

Outputs:
  runs/ablation_retrain/ablation_retrain_summary.csv   (per (config, seed) row)
  runs/ablation_retrain/ablation_retrain_aggregate.csv (5-seed mean±std per config)
  runs/ablation_retrain/ablation_retrain_results.json  (per-seed + aggregate)

CLI:
  python scripts/ablation_retrain_eval.py --device cuda:0
  python scripts/ablation_retrain_eval.py --device cuda:0 --configs no_env uniform_rollout
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse subclasses + config builder from the retrain script
from scripts.ablation_retrain import (  # noqa: E402
    NoEnvCGForecaster, NoPhaseGateCGForecaster, NoEncGatesCGForecaster, UniformRolloutCGForecaster,
    build_frozen_hpo_cfg, HMM_DIR_TEMPLATE, ENV_CKPT, SEEDS, ABLATIONS, OUT_ROOT,
)
from src.models.cg_forecaster import CGForecaster  # noqa: E402  (for "full" baseline eval)
from src.data.loader import (  # noqa: E402
    MultiHorizonDataset, collate_dict, load_dataset_csv, load_norm_params,
)
from src.eval.wis import wis, coverage  # noqa: E402
from src.eval.hmm_interval import method_f_predict_quantiles  # noqa: E402
from src.utils.checkpoints import load_fitted_hmm  # noqa: E402

CSV_PATH = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data" / "processed" / "normalization_params.json"
COVID_STRICT_START_EPIWEEK = 202240   # W40-2022

SUBCLASS_REGISTRY = {
    "no_env":          NoEnvCGForecaster,
    "no_phase":        NoPhaseGateCGForecaster,
    "no_encgates":     NoEncGatesCGForecaster,
    "uniform_rollout": UniformRolloutCGForecaster,
    "full":            CGForecaster,   # baseline retrain (no forward override) — harness-confound check
}


def _mask_df(df, split_name, epi_min):
    if epi_min is None:
        return df
    sub = df.copy()
    sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min), "split"] = "_excluded"
    return sub


def build_loaders(cfg, df, norm):
    """Build val + test_strict loaders (test_strict = test split with epi >= 202240)."""
    loaders = {}
    for split_label, epi_min in [("val", None), ("test_strict", COVID_STRICT_START_EPIWEEK)]:
        split_name = "val" if split_label == "val" else "test"
        ds_df = _mask_df(df, split_name, epi_min)
        ds = MultiHorizonDataset(ds_df, split_name, cfg.lookback, tuple(cfg.horizons), norm)
        loaders[split_label] = DataLoader(ds, batch_size=32, shuffle=False,
                                          num_workers=0, collate_fn=collate_dict)
    return loaders


def load_retrained_model(ablation: str, seed: int, device: str):
    """Build subclass model, load Stage 1 HMM + Stage 3 ckpt."""
    subclass = SUBCLASS_REGISTRY[ablation]
    cfg = build_frozen_hpo_cfg(seed)
    model = subclass(cfg).to(device)

    # Load HMM (Stage 1 ckpt — frozen during retrain, reused across all configs/seeds)
    hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))
    hmm = load_fitted_hmm(hmm_dir)
    model.prepare_for_stage2(hmm)

    # Load env encoder pretrain (for state_dict compatibility — bypassed in no_env)
    if ENV_CKPT.exists():
        env_state = torch.load(ENV_CKPT, map_location=device, weights_only=True)
        model.env_module.encoder.load_state_dict(env_state)

    # Load Stage 3 fine-tuned ckpt
    stage3_dir = _ROOT / "runs" / "m1_8_stage3_train" / f"ablation_retrain_{ablation}_s{seed}_stage3"
    stage3_best = stage3_dir / "best.pt"
    if not stage3_best.exists():
        raise FileNotFoundError(f"Missing Stage 3 ckpt: {stage3_best}")

    sd = torch.load(stage3_best, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    return model, cfg


@torch.no_grad()
def forward_subclass(model, loader, device):
    """Forward with the subclass's built-in ablated forward.

    Returns (mu_z [N, H], gamma_all [N, H, K], y_z [N, H]).
    """
    model.eval()
    mus, gammas, ys = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        env = batch["env"].to(device)
        y = batch["y"].cpu().numpy()
        pred, inter = model(x, env, return_intermediates=True)
        mus.append(pred.cpu().numpy())
        gammas.append(inter["gamma_all"].cpu().numpy())
        ys.append(y)
    return (np.concatenate(mus, 0),
            np.concatenate(gammas, 0),
            np.concatenate(ys, 0))


def compute_mae_raw(mu_z, y_z, target_std):
    """MAE in raw ILI %wILI units (z-score residual × target_std)."""
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


def evaluate_one(ablation: str, seed: int, device: str, df, norm,
                  target_mean: float, target_std: float) -> dict:
    """Evaluate one (ablation, seed) on val + test_strict."""
    model, cfg = load_retrained_model(ablation, seed, device)
    loaders = build_loaders(cfg, df, norm)

    # Method F needs HMM emission stats (after Stage 3 fine-tune)
    pm = model.phase_module
    mu_k_ili = pm._means[:, 0].cpu().numpy()
    sigma2_k_ili = pm._covs[:, 0, 0].cpu().numpy()

    # Forward both splits
    val_mu_z, val_gamma, val_y_z = forward_subclass(model, loaders["val"], device)
    test_mu_z, test_gamma, test_y_z = forward_subclass(model, loaders["test_strict"], device)

    # MAE (raw units) on test_strict
    mae_per_h = compute_mae_raw(test_mu_z, test_y_z, target_std)
    mae_avg = float(mae_per_h.mean())

    # Method F UQ → WIS + Cov95
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

    return {
        "ablation": ablation, "seed": seed,
        "n_test_strict": int(test_y_z.shape[0]),
        "mae_per_h": mae_per_h.tolist(), "mae_avg": mae_avg,
        "wis_per_h": wis_per_h, "wis_avg": float(np.mean(wis_per_h)),
        "cov95_per_h": cov_per_h, "cov95_avg": float(np.mean(cov_per_h)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--configs", type=str, nargs="+", default=list(ABLATIONS),
                        help=f"Configs to evaluate (default: {list(ABLATIONS)})")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out-dir", type=str, default=str(OUT_ROOT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    print(f"[ablation_retrain_eval] Evaluating {len(args.configs)} configs × {len(args.seeds)} seeds = "
          f"{len(args.configs) * len(args.seeds)} ckpts on {args.device}")
    print(f"  configs: {args.configs}")
    print(f"  seeds:   {args.seeds}")
    print(f"  test_strict cutoff: epiweek >= {COVID_STRICT_START_EPIWEEK}")

    per_seed_results = {c: [] for c in args.configs}
    per_seed_rows = []

    n_done, n_fail = 0, 0
    for ablation in args.configs:
        for seed in args.seeds:
            print(f"\n--- {ablation} / seed={seed} ---")
            try:
                r = evaluate_one(ablation, seed, args.device, df, norm,
                                  target_mean, target_std)
                per_seed_results[ablation].append(r)
                print(f"  MAE: " + " ".join(f"h{i+1}={v:.4f}" for i, v in enumerate(r['mae_per_h']))
                      + f" avg={r['mae_avg']:.4f}")
                print(f"  WIS: " + " ".join(f"h{i+1}={v:.4f}" for i, v in enumerate(r['wis_per_h']))
                      + f" avg={r['wis_avg']:.4f}")
                print(f"  Cov95: " + " ".join(f"h{i+1}={v:.3f}" for i, v in enumerate(r['cov95_per_h']))
                      + f" avg={r['cov95_avg']:.3f}")
                per_seed_rows.append({
                    "ablation": ablation, "seed": seed,
                    "n_test_strict": r["n_test_strict"],
                    **{f"mae_h{h+1}": r["mae_per_h"][h] for h in range(4)},
                    "mae_avg": r["mae_avg"],
                    **{f"wis_h{h+1}": r["wis_per_h"][h] for h in range(4)},
                    "wis_avg": r["wis_avg"],
                    **{f"cov95_h{h+1}": r["cov95_per_h"][h] for h in range(4)},
                    "cov95_avg": r["cov95_avg"],
                })
                n_done += 1
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                n_fail += 1
            except Exception as e:
                print(f"  FAIL: {e}")
                import traceback; traceback.print_exc()
                n_fail += 1

    # Aggregate per ablation (5-seed mean±std)
    print(f"\n{'=' * 60}")
    print(f"AGGREGATE (5-seed mean±std)")
    print(f"{'=' * 60}")
    aggregate_rows = []
    for ablation in args.configs:
        data = per_seed_results[ablation]
        if not data:
            print(f"  {ablation}: no completed runs — skipped")
            continue
        mae_arr = np.array([r["mae_avg"] for r in data])
        wis_arr = np.array([r["wis_avg"] for r in data])
        cov_arr = np.array([r["cov95_avg"] for r in data])
        mae_h = np.array([r["mae_per_h"] for r in data])
        wis_h = np.array([r["wis_per_h"] for r in data])
        cov_h = np.array([r["cov95_per_h"] for r in data])

        print(f"\n  {ablation:18s}: MAE={mae_arr.mean():.4f}±{mae_arr.std():.4f}  "
              f"WIS={wis_arr.mean():.4f}±{wis_arr.std():.4f}  "
              f"Cov95={cov_arr.mean():.3f}±{cov_arr.std():.3f}  (n={len(data)} seeds)")
        for h in range(4):
            print(f"    h={h+1}: MAE={mae_h[:,h].mean():.4f}±{mae_h[:,h].std():.4f}  "
                  f"WIS={wis_h[:,h].mean():.4f}±{wis_h[:,h].std():.4f}  "
                  f"Cov95={cov_h[:,h].mean():.3f}±{cov_h[:,h].std():.3f}")

        aggregate_rows.append({
            "ablation": ablation, "n_seeds": len(data),
            "mae_avg_mean": float(mae_arr.mean()), "mae_avg_std": float(mae_arr.std()),
            "wis_avg_mean": float(wis_arr.mean()), "wis_avg_std": float(wis_arr.std()),
            "cov95_avg_mean": float(cov_arr.mean()), "cov95_avg_std": float(cov_arr.std()),
            **{f"mae_h{h+1}_mean": float(mae_h[:,h].mean()) for h in range(4)},
            **{f"mae_h{h+1}_std": float(mae_h[:,h].std()) for h in range(4)},
            **{f"wis_h{h+1}_mean": float(wis_h[:,h].mean()) for h in range(4)},
            **{f"wis_h{h+1}_std": float(wis_h[:,h].std()) for h in range(4)},
            **{f"cov95_h{h+1}_mean": float(cov_h[:,h].mean()) for h in range(4)},
            **{f"cov95_h{h+1}_std": float(cov_h[:,h].std()) for h in range(4)},
        })

    # Write outputs
    summary_csv = out_dir / "ablation_retrain_summary.csv"
    aggregate_csv = out_dir / "ablation_retrain_aggregate.csv"
    results_json = out_dir / "ablation_retrain_results.json"

    if per_seed_rows:
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_seed_rows[0].keys())
            w.writeheader()
            w.writerows(per_seed_rows)
    if aggregate_rows:
        with open(aggregate_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
            w.writeheader()
            w.writerows(aggregate_rows)
    with open(results_json, "w") as f:
        json.dump({"per_seed": per_seed_results, "aggregate": aggregate_rows}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE: {n_done} evaluated, {n_fail} skipped/failed")
    print(f"  per-seed CSV:  {summary_csv}")
    print(f"  aggregate CSV: {aggregate_csv}")
    print(f"  results JSON:  {results_json}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
