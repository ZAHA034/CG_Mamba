"""Block-bootstrap 95% CI for the headline regional Cov95 (0.954).

Addresses the stats reviewer's "no valid CI on the coverage estimate": units are
autocorrelated across weeks AND spatially correlated across the 10 regions. Design
(pinned, not a hypothesis test -> no pre-reg lock needed): a moving BLOCK bootstrap on
the shared calendar-week axis, resampling contiguous L-week blocks JOINTLY across all 10
regions (a sampled block contributes every region/seed/horizon cell in those weeks) ->
handles autocorrelation (blocks) + cross-region correlation (joint) simultaneously.
Season-level clustering is impossible (test_strict spans 3 seasons -> 3 clusters).

Per-cell coverage is regenerated WITH epiweek via the verified canonical APMD forward
(apmd_residuals.csv dropped the time axis); point estimate must reproduce ~0.954.
Block length pinned L=6 (midpoint of the 4-8 wk design); L=4,8 reported as robustness.
Bootstrap RNG seed pinned (0) for reproducibility. B=2000.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))
import torch
from track_b_lib import (HORIZONS, TS_BOUNDARY, SEEDS, build_region_df,
                         load_cgm_model_seed, cgm_dataset, cgm_decomp_forward, load_norm)

REGIONS = [f"hhs{i}" for i in range(1, 11)]


def collect_cells(device):
    """Per-cell (region, seed, week, horizon) coverage on test_strict, z-space APMD."""
    norm = load_norm()
    weeks, covs = [], []
    for seed in SEEDS:
        model, cfg, hmm = load_cgm_model_seed(seed, device)
        for r in REGIONS:
            ds = cgm_dataset(build_region_df(r), "test", cfg, norm)
            mu, s2, y, eps = cgm_decomp_forward(model, cfg, hmm, ds, device)
            keep = eps >= TS_BOUNDARY
            for hi in range(len(HORIZONS)):
                c = (np.abs(y[keep, hi] - mu[keep, hi]) <= 1.96 * np.sqrt(s2[keep, hi])).astype(float)
                covs.append(c); weeks.append(eps[keep])
    week = np.concatenate(weeks); cov = np.concatenate(covs)
    return week, cov


def block_bootstrap(week, cov, L, B=2000, rng=None):
    """Moving-block bootstrap on the shared week axis (joint over all regions/seeds/h)."""
    rng = rng or np.random.default_rng(0)
    uniq = np.sort(np.unique(week))
    W = len(uniq)
    # contiguous L-week blocks (moving), indexed by start position in uniq
    starts = np.arange(0, max(1, W - L + 1))
    n_blocks = int(np.ceil(W / L))
    # precompute cell-mask per week for speed
    week_to_idx = {w: np.where(week == w)[0] for w in uniq}
    point = float(cov.mean())
    reps = np.empty(B)
    for b in range(B):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        picks = []
        for s in chosen:
            for w in uniq[s:s + L]:
                picks.append(week_to_idx[w])
        idx = np.concatenate(picks)
        reps[b] = cov[idx].mean()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi), float(reps.std())


def main():
    import json
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = _ROOT / "runs" / "apmd_diagnostic" / "bootstrap_cells.npz"
    if cache.exists():
        z = np.load(cache); week, cov = z["week"], z["cov"]
        print(f"[cache] loaded {len(cov)} cells from {cache.name}")
    else:
        print(f"[bootstrap] device={device} — running canonical APMD forward (5 seeds x 10 regions)")
        week, cov = collect_cells(device)
        np.savez(cache, week=week, cov=cov)
        print(f"[cache] saved {len(cov)} cells")
    print(f"[bootstrap] cells={len(cov)}  point Cov95={cov.mean():.4f} (should ~0.954)  weeks={len(np.unique(week))}")
    out = {"point_cov95": float(cov.mean()), "n_cells": int(len(cov)),
           "n_weeks": int(len(np.unique(week))), "B": 2000, "CI": {}}
    for L in (6, 4, 8):
        pt, lo, hi, sd = block_bootstrap(week, cov, L, B=2000, rng=np.random.default_rng(0))
        tag = "PRIMARY" if L == 6 else "robust "
        print(f"[{tag}] L={L}wk: Cov95={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (bootstrap SD {sd:.4f})")
        out["CI"][f"L{L}wk"] = {"point": pt, "ci_lo": lo, "ci_hi": hi, "boot_sd": sd}
    (_ROOT / "runs" / "apmd_diagnostic" / "bootstrap_ci_result.json").write_text(json.dumps(out, indent=2))
    print("WROTE runs/apmd_diagnostic/bootstrap_ci_result.json")


if __name__ == "__main__":
    main()
