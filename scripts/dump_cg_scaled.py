"""Dump Scaled-APMD (cg_scaled) predictive quantiles for the decision-simulation.

Scaled-APMD = SAME CG-Mamba forward (mu, s2_total) as raw APMD, times a per-horizon scalar
s_h fit on the VALIDATION split (leakage-free, grid-search quantile matching -- the paper's
recipe). One forward pass; raw and scaled differ only by the s_h multiplier.

CAVEAT (must state): unlike raw APMD, the scaled mode REQUIRES validation calibration data.

Output: runs/decision_sim/forecast_quantiles_cg_scaled.parquet  (model='cg_scaled')
Result-blind: reported regardless of whether scaled helps or hurts the decision cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as spn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))

import e1_final_eval as E
import regime_shift_drivers as rsd

OUT = _ROOT / "runs" / "decision_sim"
TAUS = np.round(np.arange(0.005, 1.0, 0.005), 4)
TEST_FIRST = 202240
SEEDS = [42, 123, 456, 789, 1024]
REGIONS = [f"hhs{i}" for i in range(1, 11)]
HORIZONS = [1, 2, 3, 4]
RNG = np.random.default_rng(20260725)
S_GRID = np.concatenate([np.linspace(0.01, 0.5, 20), np.linspace(0.5, 3.0, 30), np.linspace(3.0, 30.0, 15)])
QLEVELS = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)


def _forward(scope, seed, split, device):
    """n3_d64 forward on (scope, split) -> raw df_pred (target_ep, horizon, mu, s2_total, y_true)."""
    model, hmm, cfg = E.load_final_model("n3_d64", 3, 64, seed, device)
    norm = E.load_norm_params(E.FINAL_NORM_JSON)
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    mu_k = hmm.means[:, 0].astype(np.float64); s2_k = hmm.covars[:, 0, 0].astype(np.float64)
    if scope == "national":
        df = E.load_dataset_csv(E.FINAL_CSV)
    else:
        df = rsd._build_region_df(scope)
    ds = E.MultiHorizonDataset(df, split=split, lookback=cfg.lookback,
                               horizons=tuple(cfg.horizons), norm=norm)
    dfp = E._forward_dataset(model, ds, device)
    dfp = E._decompose_apmd(dfp, mu_k, s2_k, tmean, tstd)
    del model, hmm
    return dfp


def _fit_s_h(val):
    """Grid-search quantile matching per horizon on validation (raw units)."""
    s_h = {}
    for h in HORIZONS:
        v = val[val.horizon == h]
        mu = v.mu.to_numpy(); s2 = v.s2_total.to_numpy(); y = v.y_true.to_numpy()
        best_s, best_loss = 1.0, np.inf
        for s in S_GRID:
            sig = np.sqrt(s * s2 + 1e-12)
            loss = sum((float((y <= mu + spn.ppf(q) * sig).mean()) - q) ** 2 for q in QLEVELS)
            if loss < best_loss:
                best_loss, best_s = loss, float(s)
        s_h[h] = best_s
    return s_h


def cg_scaled_scope(scope, device):
    per, y_of, s_h_seeds = {}, {}, []
    for seed in SEEDS:
        val = _forward(scope, seed, "val", device)
        s_h = _fit_s_h(val)
        s_h_seeds.append(s_h)
        test = _forward(scope, seed, "test", device)
        test = test[test.target_ep >= TEST_FIRST]
        for _, r in test.iterrows():
            h = int(r.horizon); key = (int(r.target_ep), h)
            sig = np.sqrt(max(s_h[h] * r.s2_total, 1e-12))
            per.setdefault(key, []).append(RNG.normal(r.mu, sig, 200))
            y_of[key] = float(r.y_true)
    rows = [("cg_scaled", scope, k[0], k[1], y_of[k], np.quantile(np.concatenate(v), TAUS))
            for k, v in per.items()]
    mean_sh = {h: float(np.mean([s[h] for s in s_h_seeds])) for h in HORIZONS}
    return rows, mean_sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, sh_log = [], {}
    for scope in ["national"] + REGIONS:
        rows, mean_sh = cg_scaled_scope(scope, args.device)
        all_rows += rows; sh_log[scope] = mean_sh
        print(f"[cg_scaled] {scope}: {len(rows)} rows, mean s_h(val)={ {h: round(v,2) for h,v in mean_sh.items()} }", flush=True)
    qcols = [f"q{t:.4f}" for t in TAUS]
    recs = []
    for model, scope, tep, h, y, grid in all_rows:
        d = {"model": model, "scope": scope, "target_ep": tep, "horizon": h, "y_true": y}
        d.update({qcols[i]: grid[i] for i in range(len(TAUS))})
        recs.append(d)
    pd.DataFrame(recs).to_parquet(OUT / "forecast_quantiles_cg_scaled.parquet", index=False)
    import json
    (OUT / "cg_scaled_s_h.json").write_text(json.dumps(sh_log, indent=2))
    print(f"\n[cg_scaled] {len(recs)} rows -> forecast_quantiles_cg_scaled.parquet  (s_h<1 => tighten)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
