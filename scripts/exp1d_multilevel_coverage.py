"""Multi-level empirical coverage (50/80/90/95) for CG-Mamba raw native APMD,
per HHS region, test_strict, at each horizon (focus h=4). Gates the fan-chart band design:
drawing 50/80/95 nested bands = visually claiming 3-level calibration, so measure it first.

Coverage is scale-invariant under the affine denorm, so computed in z-space directly:
  lo = mu_z + Phi^{-1}(a/2)*sigma_z ; hi = mu_z + Phi^{-1}(1-a/2)*sigma_z ; cov = mean(lo<=y<=hi).
Reuses the verified track_b_lib harness (cgm_decomp_forward). 5 seeds, per-seed then mean.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))
import torch
from track_b_lib import (HORIZONS, TS_BOUNDARY, SEEDS, load_cgm_model_seed,
                         cgm_dataset, cgm_decomp_forward, build_region_df, load_norm)

REGIONS = [f"hhs{i}" for i in range(1, 11)]
LEVELS = {"50": 0.50, "80": 0.80, "90": 0.90, "95": 0.95}   # central PI levels
OUT = _ROOT / "runs" / "exp1b_learned_variance" / "multilevel_coverage.json"


def cov_at_levels(mu_z, sig2_z, y_z):
    """mu_z/sig2_z/y_z: [N] at one horizon. Returns dict level->coverage."""
    sig = np.sqrt(np.maximum(sig2_z, 1e-12))
    out = {}
    for name, L in LEVELS.items():
        a = 1.0 - L
        lo = mu_z + norm.ppf(a / 2) * sig
        hi = mu_z + norm.ppf(1 - a / 2) * sig
        out[name] = float(np.mean((y_z >= lo) & (y_z <= hi)))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_p = load_norm()
    # per (region, seed, horizon) coverage
    # models loaded per seed once, evaluated on all regions
    per = {r: {h: {lvl: [] for lvl in LEVELS} for h in HORIZONS} for r in REGIONS}
    for seed in SEEDS:
        model, cfg, hmm = load_cgm_model_seed(seed, device)
        for r in REGIONS:
            rdf = build_region_df(r)
            ds = cgm_dataset(rdf, "test", cfg, norm_p)
            mu_z, sig2_z, y_z, eps_h1 = cgm_decomp_forward(model, cfg, hmm, ds, device)
            idx = np.where(eps_h1 >= TS_BOUNDARY)[0]
            for h_idx, h in enumerate(HORIZONS):
                cov = cov_at_levels(mu_z[idx, h_idx], sig2_z[idx, h_idx], y_z[idx, h_idx])
                for lvl in LEVELS:
                    per[r][h][lvl].append(cov[lvl])
        del model
        # incremental partial dump (survive interruption / GPU eviction)
        (OUT.with_suffix(".partial.json")).write_text(json.dumps(
            {r: {h: {lvl: per[r][h][lvl] for lvl in LEVELS} for h in HORIZONS} for r in REGIONS}, indent=1))
        print(f"[seed {seed}] done (partial saved)", flush=True)

    # aggregate: per-region mean over seeds
    region_cov = {r: {h: {lvl: float(np.mean(per[r][h][lvl])) for lvl in LEVELS} for h in HORIZONS} for r in REGIONS}
    # 10-region aggregate mean per (h, level)
    agg = {h: {lvl: float(np.mean([region_cov[r][h][lvl] for r in REGIONS])) for lvl in LEVELS} for h in HORIZONS}
    # median region by |Cov95 - 0.95| at h=4
    h4 = 4
    dev = {r: abs(region_cov[r][h4]["95"] - 0.95) for r in REGIONS}
    order = sorted(REGIONS, key=lambda r: dev[r])
    median_region = order[len(order) // 2]  # median of 10 -> index 5 (upper-median)

    result = {"levels": LEVELS, "aggregate_by_h": agg, "region_by_h": region_cov,
              "h4_abs_dev_cov95": dev, "median_region_h4": median_region,
              "median_region_rule": "region with median |Cov95-0.95| at h=4"}
    OUT.write_text(json.dumps(result, indent=2))

    print("\n=== 10-region AGGREGATE coverage (achieved vs nominal) ===")
    for h in HORIZONS:
        line = "  ".join(f"{lvl}%:{agg[h][lvl]:.3f}" for lvl in LEVELS)
        print(f"  h={h}:  {line}   (nominal 0.50/0.80/0.90/0.95)")
    print(f"\n=== median region (|Cov95-0.95| median) at h=4: {median_region} ===")
    print(f"  its h4 coverage: " + "  ".join(f"{lvl}%:{region_cov[median_region][h4][lvl]:.3f}" for lvl in LEVELS))
    print(f"\n  per-region |Cov95-0.95| @h4 (sorted): " + ", ".join(f"{r}:{dev[r]:.3f}" for r in order))
    print(f"\n[exp1d] wrote {OUT}")


if __name__ == "__main__":
    main()
