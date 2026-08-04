#!/usr/bin/env python3
"""Experiment D: RAW-native APMD Cov95 for the 4-config ablation.

The Table VI ablation currently reports Cov95 on the *Scaled* APMD variant
(per-horizon s_h calibration via method_f_predict_quantiles). The paper's
headline coverage numbers (national 0.993, regional 0.954) are the *raw*
APMD (s_h = 1, no calibration data). This script re-evaluates the same
20 ablation checkpoints (4 configs x 5 seeds) under the RAW APMD path so the
ablation's calibration claim is stated on the same variant as the headline.

RAW APMD (matches e1_final_eval.eval_cov95_wis, s_h = 1):
    sigma2_total = compute_decomposition(mu_CGM, gamma_all, mu_k, sigma2_k)
    q_z(p)  = mu_CGM + Phi^{-1}(p) * sqrt(sigma2_total)       # z-scored
    q_raw   = q_z * target_std + target_mean                  # denormalized
    Cov95   = mean_over_N[ q_raw(0.025) <= y_raw <= q_raw(0.975) ], per-h, avg

Inference-only (no re-training). Runs on CPU by default (both GPUs busy with
other users' processes -- no-server-interference rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm as _N

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.ablation_retrain_eval import (
    ABLATIONS, SEEDS, CSV_PATH, NORM_PATH, OUT_ROOT,
    load_retrained_model, build_loaders, forward_subclass,
)
from src.data.loader import load_dataset_csv, load_norm_params
from src.eval.hmm_interval import compute_decomposition


def raw_cov95_per_h(mu_z, gamma_all, mu_k_ili, sigma2_k_ili,
                    y_z, target_mean, target_std):
    """RAW APMD (s_h=1) 95% coverage per horizon.  Returns [H] array."""
    decomp = compute_decomposition(mu_z, gamma_all, mu_k_ili, sigma2_k_ili)
    sigma2_total = decomp.sigma2_total            # [N, H], z-scored
    sd = np.sqrt(np.maximum(sigma2_total, 0.0))   # [N, H]

    z_lo, z_hi = _N.ppf(0.025), _N.ppf(0.975)
    q_lo_z = mu_z + z_lo * sd
    q_hi_z = mu_z + z_hi * sd
    # denormalize
    q_lo = q_lo_z * target_std + target_mean
    q_hi = q_hi_z * target_std + target_mean
    y_raw = y_z * target_std + target_mean

    covered = (y_raw >= q_lo) & (y_raw <= q_hi)   # [N, H]
    return covered.mean(axis=0)                    # [H]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--configs", type=str, nargs="+", default=list(ABLATIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--out", type=str,
                   default=str(Path(OUT_ROOT) / "d_raw_apmd_cov95.json"))
    args = p.parse_args()

    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    print(f"[D] RAW APMD Cov95 | {len(args.configs)} configs x {len(args.seeds)} seeds "
          f"on {args.device}")

    per_seed = {c: [] for c in args.configs}
    for cfg_name in args.configs:
        for seed in args.seeds:
            model, cfg = load_retrained_model(cfg_name, seed, args.device)
            loaders = build_loaders(cfg, df, norm)
            pm = model.phase_module
            mu_k = pm._means[:, 0].cpu().numpy()
            s2_k = pm._covs[:, 0, 0].cpu().numpy()
            mu_z, gamma, y_z = forward_subclass(model, loaders["test_strict"], args.device)
            cov_h = raw_cov95_per_h(mu_z, gamma, mu_k, s2_k, y_z, target_mean, target_std)
            per_seed[cfg_name].append(cov_h)
            print(f"  {cfg_name:16s} seed={seed}  "
                  + " ".join(f"h{i+1}={v:.3f}" for i, v in enumerate(cov_h))
                  + f"  avg={cov_h.mean():.3f}")

    # aggregate: mean over seeds of per-seed avg-over-horizon
    summary = {}
    for cfg_name in args.configs:
        arr = np.stack(per_seed[cfg_name])          # [S, H]
        seed_avgs = arr.mean(axis=1)                # [S]
        summary[cfg_name] = {
            "cov95_avg_mean": float(seed_avgs.mean()),
            "cov95_avg_std": float(seed_avgs.std(ddof=1)) if len(seed_avgs) > 1 else 0.0,
            "cov95_per_h_mean": arr.mean(axis=0).tolist(),
            "n_seeds": int(arr.shape[0]),
        }

    full_key = "full" if "full" in summary else list(summary)[0]
    full_mean = summary[full_key]["cov95_avg_mean"]

    print("\n=== RAW APMD Cov95 (mean +/- std over seeds), Delta vs Full ===")
    for cfg_name in args.configs:
        s = summary[cfg_name]
        d = s["cov95_avg_mean"] - full_mean
        dstr = "  (baseline)" if cfg_name == full_key else f"  Delta={d:+.3f}"
        print(f"  {cfg_name:16s} {s['cov95_avg_mean']:.3f} +/- {s['cov95_avg_std']:.3f}"
              f"  (n={s['n_seeds']}){dstr}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "variant": "raw_apmd_s1",
        "full_key": full_key,
        "summary": summary,
        "delta_vs_full": {c: summary[c]["cov95_avg_mean"] - full_mean
                          for c in args.configs},
    }, indent=2))
    print(f"\n[D] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
