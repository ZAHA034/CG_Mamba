"""Method F sanity check — HMM-derived calibrated intervals viability.

Phase 1 of Method F implementation (PLAN J.12 + clinical interpretability).
Decision gate before full implementation: 6 metrics + 3-component
decomposition temporal alignment check.

Pass criteria (all 3):
  Cond 1: sig2_between/total ratio varies meaningfully (range > 0.3 across val)
  Cond 2: sig2_between peaks align with known phase transitions
          (W40 season start, W5 peak)
  Cond 3: bias² spikes during anomaly periods (small in normal weeks)

Output:
  runs/method_f_sanity/sanity_report.json
  runs/method_f_sanity/temporal_decomposition.csv
"""
from __future__ import annotations

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
sys.path.insert(0, str(_ROOT / "src"))

from src.models.cg_forecaster import CGForecaster
from src.utils.config import CGMambaConfig
from src.utils.checkpoints import load_fitted_hmm
from src.data.loader import (
    load_dataset_csv, load_norm_params, MultiHorizonDataset, collate_dict,
)

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_DIR = _ROOT / "runs" / "method_f_sanity"

# M2.1 top1 cell HP (matching wis_phase_c_dropout_grid.py)
CG_TOP1_HP = {
    "gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104,
    "hmm_lr_ratio": 0.01, "state_embed_lr_ratio": 0.01, "env_lr_ratio": 0.001,
}
OTHER_LR_BASE = 1e-4
HMM_DIR_TEMPLATE = _ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed{seed}"
ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"

# M2.1 top1 cell Stage 3 ckpt location
CG_M21_CKPT_TEMPLATE = (
    _ROOT / "runs" / "m1_8_stage3_train" /
    "hpo_p2_g1e-03_b1e-04_lb104_h0.01_se0.01_e0.001_s{seed}" / "best.pt"
)


def build_cg_mamba_m21(seed: int, device: str):
    """Build CGForecaster matching M2.1 top1 cell + load Stage 3 ckpt."""
    hp = CG_TOP1_HP
    cfg = dataclasses.replace(
        CGMambaConfig(),
        seed=seed, dropout=0.0,  # M2.1 trained at dropout=0
        lookback=hp["lookback"],
        stage2_gate_lr=hp["gate_lr"],
        stage2_backbone_lr=hp["backbone_lr"],
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
    if not ckpt_path.exists():
        raise FileNotFoundError(f"M2.1 ckpt missing: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    return model, cfg


@torch.no_grad()
def compute_method_f_components(model, loader, device, ili_dim_in_aug=0):
    """For each val sample, compute mu_CGM + 3-component decomposition per horizon.

    Returns (per_sample_records, hmm_stats):
      per_sample_records: list of dict per (sample_idx, horizon)
      hmm_stats: dict with mu_k_ili, sig_k_ili (verification)
    """
    pm = model.phase_module
    mu_k_ili = pm._means[:, ili_dim_in_aug].cpu().numpy()                # [K]
    sig2_k_ili = pm._covs[:, ili_dim_in_aug, ili_dim_in_aug].cpu().numpy()  # [K]
    K = len(mu_k_ili)

    records = []
    model.eval()
    sample_idx = 0
    for batch in loader:
        x = batch["x"].to(device); env = batch["env"].to(device)
        y = batch["y"].cpu().numpy()                                       # [B, H]
        preds, intermediates = model(x, env, return_intermediates=True)
        preds = preds.cpu().numpy()                                        # [B, H]
        gamma_all = intermediates["gamma_all"].cpu().numpy()              # [B, max_H, K]
        target_eps = batch.get("target_epiweeks")                          # list of lists

        B, H = preds.shape
        for b in range(B):
            for h_idx in range(H):
                gamma_h = gamma_all[b, h_idx, :]                               # [K]
                mu_CGM = preds[b, h_idx]
                y_h = y[b, h_idx]
                mu_HMM = float((gamma_h * mu_k_ili).sum())
                sig2_within = float((gamma_h * sig2_k_ili).sum())
                sig2_between_HMM = float((gamma_h * (mu_k_ili - mu_HMM)**2).sum())
                bias_sq = float((mu_HMM - mu_CGM)**2)
                sig2_total = sig2_within + sig2_between_HMM + bias_sq
                ep = (target_eps[b][h_idx] if target_eps and len(target_eps) > b
                       and len(target_eps[b]) > h_idx else -1)
                records.append({
                    "sample_idx": sample_idx + b,
                    "horizon": h_idx + 1,
                    "target_ep": int(ep),
                    "y_true_z": float(y_h),
                    "mu_CGM_z": float(mu_CGM),
                    "mu_HMM_z": mu_HMM,
                    "sigma2_within": sig2_within,
                    "sigma2_between_HMM": sig2_between_HMM,
                    "bias_sq": bias_sq,
                    "sigma2_total": sig2_total,
                    "gamma_h": gamma_h.tolist(),
                    "residual_z": float(y_h - mu_CGM),
                })
        sample_idx += B
    return records, {"mu_k_ili_z": mu_k_ili.tolist(),
                     "sigma_k_ili_z": np.sqrt(sig2_k_ili).tolist(), "K": K}


def evaluate_conditions(records, hmm_stats, norm):
    """Apply 3 go/no-go conditions + extract 6 metrics."""
    arr = lambda key: np.array([r[key] for r in records])
    sig2_within = arr("sigma2_within")
    sig2_between = arr("sigma2_between_HMM")
    bias_sq = arr("bias_sq")
    sig2_total = arr("sigma2_total")
    residuals = arr("residual_z")
    target_eps = arr("target_ep")

    sig_total = np.sqrt(sig2_total + 1e-12)
    ratio_between_total = sig2_between / (sig2_total + 1e-12)
    ratio_bias_total = bias_sq / (sig2_total + 1e-12)
    ratio_within_total = sig2_within / (sig2_total + 1e-12)

    # ─── Metric 1: phase mean spread ───
    mu_k = np.array(hmm_stats["mu_k_ili_z"])
    phase_spread = float(mu_k.max() - mu_k.min())

    # ─── Metric 2: per-phase std ───
    sig_k = np.array(hmm_stats["sigma_k_ili_z"])

    # ─── Metric 3-5: σ² component ranges ───
    sig2_within_stats = {"min": float(sig2_within.min()), "max": float(sig2_within.max()),
                       "mean": float(sig2_within.mean()), "std": float(sig2_within.std())}
    sig2_between_stats = {"min": float(sig2_between.min()), "max": float(sig2_between.max()),
                        "mean": float(sig2_between.mean()), "std": float(sig2_between.std())}
    bias_sq_stats = {"min": float(bias_sq.min()), "max": float(bias_sq.max()),
                     "mean": float(bias_sq.mean()), "std": float(bias_sq.std())}

    # ─── Metric 6: σ_between / sig_total ratio temporal variability ───
    ratio_mean = float(ratio_between_total.mean())
    ratio_min = float(ratio_between_total.min())
    ratio_max = float(ratio_between_total.max())
    ratio_range = ratio_max - ratio_min
    ratio_std = float(ratio_between_total.std())

    # ─── Metric 7: calibration scale s estimate ───
    abs_res = np.abs(residuals)
    s_estimate = float(abs_res.std() / (sig_total.std() + 1e-12))
    # alternative quantile-matching s: targets cov95
    s_cov95 = float(np.quantile(abs_res / (sig_total + 1e-12), 0.95) / 1.96)

    # ─── Cond 1: temporal variability ───
    cond1_pass = bool(ratio_range > 0.30)
    cond1_score = ratio_range

    # ─── Cond 2: temporal alignment (rough check) ───
    # Group records by target_ep, compute mean sig2_between per epiweek
    by_ep = {}
    for r in records:
        ep = r["target_ep"]
        by_ep.setdefault(ep, []).append(r["sigma2_between_HMM"])
    ep_to_between = {ep: float(np.mean(v)) for ep, v in by_ep.items() if ep > 0}
    sorted_eps = sorted(ep_to_between.keys())

    # Look for peaks at "transition" epiweeks (rough heuristic):
    # ep % 100 → mmwr week. Transitions ~ W40-W42 (season start), W4-W6 (peak), W20-W22 (decline)
    def ep_to_week(ep):
        return ep % 100

    transition_weeks = list(range(38, 44)) + list(range(2, 8)) + list(range(18, 24))
    stable_weeks = list(range(24, 38)) + list(range(8, 18))   # summer + mid-season

    transition_betweens = [v for ep, v in ep_to_between.items()
                            if ep_to_week(ep) in transition_weeks]
    stable_betweens = [v for ep, v in ep_to_between.items()
                       if ep_to_week(ep) in stable_weeks]
    if transition_betweens and stable_betweens:
        trans_mean = float(np.mean(transition_betweens))
        stable_mean = float(np.mean(stable_betweens))
        align_ratio = trans_mean / (stable_mean + 1e-12)
        cond2_pass = bool(align_ratio > 1.5)  # transition > 1.5× stable
        cond2_score = align_ratio
    else:
        cond2_pass = False
        cond2_score = 0.0
        trans_mean, stable_mean = 0.0, 0.0

    # ─── Cond 3: bias² anomaly behavior ───
    # Quantile spike check: top 5% of bias² vs median
    bias_p95 = float(np.quantile(bias_sq, 0.95))
    bias_median = float(np.median(bias_sq))
    bias_p99 = float(np.quantile(bias_sq, 0.99))
    bias_spike_ratio = bias_p95 / (bias_median + 1e-12)
    cond3_pass = bool(bias_spike_ratio > 3.0)  # 95th percentile > 3× median
    cond3_score = bias_spike_ratio

    return {
        "metric_1_phase_mean_spread_z": phase_spread,
        "metric_2_per_phase_std_z": sig_k.tolist(),
        "metric_3_sigma2_within_stats": sig2_within_stats,
        "metric_4_sigma2_between_HMM_stats": sig2_between_stats,
        "metric_5_bias_sq_stats": bias_sq_stats,
        "metric_6_between_total_ratio": {
            "mean": ratio_mean, "min": ratio_min, "max": ratio_max,
            "range": ratio_range, "std": ratio_std,
        },
        "metric_7_calibration_scale_s_estimate": {
            "via_std_ratio": s_estimate,
            "via_cov95_quantile": s_cov95,
        },
        "decomposition_summary": {
            "within_fraction_mean": float(ratio_within_total.mean()),
            "between_fraction_mean": ratio_mean,
            "bias_fraction_mean": float(ratio_bias_total.mean()),
        },
        "conditions": {
            "cond1_temporal_variability": {
                "pass": cond1_pass, "score": cond1_score,
                "threshold": 0.30,
                "desc": "sig2_between/total ratio range across val samples",
            },
            "cond2_temporal_alignment": {
                "pass": cond2_pass, "score": cond2_score, "threshold": 1.5,
                "desc": "sig2_between mean ratio: transition weeks vs stable weeks",
                "transition_mean": trans_mean, "stable_mean": stable_mean,
            },
            "cond3_bias_anomaly_spike": {
                "pass": cond3_pass, "score": cond3_score, "threshold": 3.0,
                "desc": "bias² p95 / median ratio",
                "bias_median": bias_median, "bias_p95": bias_p95, "bias_p99": bias_p99,
            },
            "all_pass": cond1_pass and cond2_pass and cond3_pass,
            "any_pass": cond1_pass or cond2_pass or cond3_pass,
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)

    # Build M2.1 ckpt (top1 cell, seed42)
    seed = 42
    print(f"Loading M2.1 top1 cell ckpt seed={seed}")
    model, cfg = build_cg_mamba_m21(seed, device)
    print(f"  cfg.dropout={cfg.dropout} (M2.1 trained at 0)")
    print(f"  cfg.lookback={cfg.lookback}")

    # Build val loader
    val_ds = MultiHorizonDataset(df, "val", cfg.lookback, tuple(cfg.horizons), norm)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=collate_dict)
    print(f"  val dataset windows: {len(val_ds)}")

    # Compute components
    print("\nComputing Method F 3-component decomposition over val...")
    records, hmm_stats = compute_method_f_components(model, val_loader, device)
    print(f"  records: {len(records)} (= {len(val_ds)} samples × {len(cfg.horizons)} horizons)")

    # Save raw decomposition CSV
    import csv
    csv_path = OUT_DIR / "temporal_decomposition.csv"
    fields = ["sample_idx", "horizon", "target_ep", "y_true_z", "mu_CGM_z",
              "mu_HMM_z", "sigma2_within", "sigma2_between_HMM", "bias_sq",
              "sigma2_total", "residual_z"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fields})
    print(f"  Saved: {csv_path.relative_to(_ROOT)}")

    # Evaluate go/no-go conditions
    print("\nEvaluating conditions...")
    report = evaluate_conditions(records, hmm_stats, norm)
    report["hmm_stats"] = hmm_stats
    report["seed"] = seed
    report["n_records"] = len(records)

    report_path = OUT_DIR / "sanity_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Pretty print
    print("\n" + "=" * 90)
    print("Method F Sanity Check — 7 metrics + 3 conditions")
    print("=" * 90)
    print(f"\nHMM stats (z-scored):")
    print(f"  mu_k_ili     : {hmm_stats['mu_k_ili_z']}")
    print(f"  sig_k_ili     : {hmm_stats['sigma_k_ili_z']}")
    print(f"  Phase spread: {report['metric_1_phase_mean_spread_z']:.4f}")

    print(f"\n3-Component decomposition (z-score² space):")
    sig2w = report["metric_3_sigma2_within_stats"]
    sig2b = report["metric_4_sigma2_between_HMM_stats"]
    bs = report["metric_5_bias_sq_stats"]
    print(f"  sig2_within       mean={sig2w['mean']:.4f}  range=[{sig2w['min']:.4f}, {sig2w['max']:.4f}]")
    print(f"  sig2_between_HMM  mean={sig2b['mean']:.4f}  range=[{sig2b['min']:.4f}, {sig2b['max']:.4f}]")
    print(f"  bias²            mean={bs['mean']:.4f}  range=[{bs['min']:.4f}, {bs['max']:.4f}]")

    decomp = report["decomposition_summary"]
    print(f"\nFraction of sig2_total (avg):")
    print(f"  within  : {decomp['within_fraction_mean']:.1%}")
    print(f"  between : {decomp['between_fraction_mean']:.1%}")
    print(f"  bias    : {decomp['bias_fraction_mean']:.1%}")

    r6 = report["metric_6_between_total_ratio"]
    print(f"\nsig2_between/total ratio across val:")
    print(f"  range=[{r6['min']:.3f}, {r6['max']:.3f}] (range size {r6['range']:.3f})")
    print(f"  mean={r6['mean']:.3f} std={r6['std']:.3f}")

    s = report["metric_7_calibration_scale_s_estimate"]
    print(f"\nCalibration scale s estimate:")
    print(f"  via std ratio   : {s['via_std_ratio']:.3f}")
    print(f"  via cov95 quant : {s['via_cov95_quantile']:.3f}")

    conds = report["conditions"]
    print("\n" + "=" * 90)
    print("Go/No-go conditions:")
    print("=" * 90)
    for cname, c in conds.items():
        if cname in ("all_pass", "any_pass"):
            continue
        mark = "✅" if c["pass"] else "❌"
        print(f"  {mark} {cname:32s} score={c['score']:.3f} threshold={c['threshold']:.2f}  "
              f"({c['desc']})")
    print()
    if conds["all_pass"]:
        print("✅ ALL 3 CONDITIONS PASS — Method F + §V.X interpretability subsection FULL GO")
    elif conds["any_pass"]:
        n_pass = sum(1 for k in ("cond1_temporal_variability", "cond2_temporal_alignment",
                                  "cond3_bias_anomaly_spike") if conds[k]["pass"])
        print(f"🟡 PARTIAL — {n_pass}/3 conditions pass")
        print("    → Method F GO + §V.X 'honest paragraph' (Option b) fallback")
    else:
        print("🔴 NO CONDITIONS PASS — Method F fallback")
        print("    → §V.X skip, §III.7 calibrated intervals only")
    print(f"\nSaved: {report_path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
