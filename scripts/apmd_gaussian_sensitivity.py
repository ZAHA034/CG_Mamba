"""A1 — APMD Gaussian-approximation sensitivity on the per-sample r>0.3 cells.

LOCKED design: CGM_v2_paper/PREREG_apmd_gaussian_sensitivity.md (2026-08-04).

Quantifies the effect the manuscript (§142-152) explicitly leaves unevaluated: for the
per-sample r = sigma2_between/sigma2_total > 0.3 cells (~3.6% of national test_strict),
Gaussian-approximated because the global switch is on the dataset mean, what is the change in
WIS / Cov95 versus the exact mixture-CDF inversion (Eq. 147)?

Reuses ONLY the actual code paths (no re-implementation of APMD/WIS):
  - forward + per-seed HMM loading:  e1_final_eval.load_final_model / _forward_dataset
  - decomposition + quantiles:       src.eval.hmm_interval.{compute_decomposition,
                                        gaussian_quantiles, mixture_quantile_one}
  - scoring:                         src.eval.wis.{wis, coverage}

Raw native (s_h = 1) throughout — the headline UQ. Consistency gate: the all-Gaussian arm must
reproduce the paper's national headline (WIS 0.399, Cov95 0.993) within tolerance before any delta
is reported.

Run:  python scripts/apmd_gaussian_sensitivity.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import torch

from e1_final_eval import load_final_model, _forward_dataset, FINAL_CSV, FINAL_NORM_JSON
from src.data.loader import load_dataset_csv, load_norm_params, MultiHorizonDataset
from src.eval.hmm_interval import (
    HMMDecomposition, compute_decomposition, gaussian_quantiles, mixture_quantile_one,
)
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES

# LOCKED constants (mirror e1_final_eval headline)
CONFIG = ("n3_d64", 3, 64)        # paper headline config (115,389 params); gate confirms via 0.399/0.993
SEEDS = [42, 123, 456, 789, 1024]
HORIZONS = [1, 2, 3, 4]
TEST_STRICT_START = 202240
R_THRESHOLD = 0.3                 # manuscript's existing threshold — NOT re-tuned
GATE_WIS, GATE_COV = 0.399, 0.993
GATE_TOL_WIS, GATE_TOL_COV = 0.005, 0.010
OUT = _ROOT / "runs" / "apmd_gaussian_sensitivity"


def _pivot_seed(df_pred, target_mean, target_std, mu_k_z, s2_k_z):
    """Long (target_ep,horizon) -> [N,H] z-space arrays + decomposition + per-cell r."""
    df = df_pred[df_pred.target_ep >= TEST_STRICT_START].copy()
    eps = sorted(df.target_ep.unique())
    ep_idx = {e: i for i, e in enumerate(eps)}
    N, H, K = len(eps), len(HORIZONS), len(mu_k_z)
    mu_z = np.full((N, H), np.nan)
    gamma = np.full((N, H, K), np.nan)
    y_z = np.full((N, H), np.nan)
    for _, r in df.iterrows():
        i = ep_idx[int(r.target_ep)]
        h = int(r.horizon) - 1
        mu_z[i, h] = r.mu_z
        gamma[i, h, :] = np.array(r.gamma_h)
        y_z[i, h] = r.y_z
    keep = ~np.isnan(mu_z).any(axis=1)          # complete-horizon origins only
    mu_z, gamma, y_z = mu_z[keep], gamma[keep], y_z[keep]
    decomp = compute_decomposition(mu_z, gamma, mu_k_z, s2_k_z)
    r_cell = decomp.sigma2_between_HMM / (decomp.sigma2_total + 1e-12)   # [N,H]
    return decomp, gamma, y_z, r_cell, target_mean, target_std


def _quantiles_two_arms(decomp, gamma, mu_k_z, s2_k_z, r_cell, mean, std):
    """Returns raw-scale gaussian dict and spliced(mixture on r>0.3) dict; both s=1."""
    N, H = decomp.mu_CGM.shape
    s_ones = np.ones(H)
    g_z = gaussian_quantiles(decomp, s_ones)                      # dict q->[N,H] z-space
    spliced_z = {q: g_z[q].copy() for q in REQUIRED_QUANTILES}
    sigma_k = np.sqrt(s2_k_z)
    mu_shift = decomp.mu_CGM - decomp.mu_HMM                       # recenter (s=1)
    hit = np.argwhere(r_cell > R_THRESHOLD)
    for (n, h) in hit:
        g_h = gamma[n, h, :]
        for q in REQUIRED_QUANTILES:
            y_mix = mixture_quantile_one(q, mu_k_z, sigma_k, g_h)  # actual code fn
            if np.isnan(y_mix):
                continue                                          # keep gaussian on bracket-fail
            spliced_z[q][n, h] = mu_shift[n, h] + y_mix
    to_raw = lambda d: {q: (d[q] * std + mean).ravel() for q in REQUIRED_QUANTILES}
    return to_raw(g_z), to_raw(spliced_z), hit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cid, nl, dm = CONFIG
    norm = load_norm_params(FINAL_NORM_JSON)
    mean = float(norm["ili_weighted_pct"]["mean"]); std = float(norm["ili_weighted_pct"]["std"])

    pool = {"y": [], "gauss": {q: [] for q in REQUIRED_QUANTILES},
            "splice": {q: [] for q in REQUIRED_QUANTILES}, "r": [], "h": []}
    n_hit = 0
    for seed in SEEDS:
        model, hmm, cfg = load_final_model(cid, nl, dm, seed, device)
        mu_k_z = hmm.means[:, 0].astype(np.float64)
        s2_k_z = hmm.covars[:, 0, 0].astype(np.float64)
        df_full = load_dataset_csv(FINAL_CSV)
        ds = MultiHorizonDataset(df_full, split="test", lookback=cfg.lookback,
                                 horizons=tuple(cfg.horizons), norm=norm)
        df_pred = _forward_dataset(model, ds, device)
        decomp, gamma, y_z, r_cell, _, _ = _pivot_seed(df_pred, mean, std, mu_k_z, s2_k_z)
        g_raw, s_raw, hit = _quantiles_two_arms(decomp, gamma, mu_k_z, s2_k_z, r_cell, mean, std)
        y_raw = (y_z * std + mean).ravel()
        pool["y"].append(y_raw)
        for q in REQUIRED_QUANTILES:
            pool["gauss"][q].append(g_raw[q]); pool["splice"][q].append(s_raw[q])
        pool["r"].append(r_cell.ravel())
        pool["h"].append(np.tile(np.array(HORIZONS), decomp.mu_CGM.shape[0]))
        n_hit += len(hit)
        del model, hmm

    y = np.concatenate(pool["y"])
    gq = {q: np.concatenate(pool["gauss"][q]) for q in REQUIRED_QUANTILES}
    sq = {q: np.concatenate(pool["splice"][q]) for q in REQUIRED_QUANTILES}
    rr = np.concatenate(pool["r"]); hh = np.concatenate(pool["h"])
    mask = rr > R_THRESHOLD

    def score(qd, sel=None):
        if sel is None:
            return float(wis(y, qd).mean()), coverage(y, qd, 0.05)
        ys = y[sel]; qs = {q: qd[q][sel] for q in REQUIRED_QUANTILES}
        return float(wis(ys, qs).mean()), coverage(ys, qs, 0.05)

    gate_w, gate_c = score(gq)               # all-gaussian = headline reproduction
    gate_ok = (abs(gate_w - GATE_WIS) <= GATE_TOL_WIS) and (abs(gate_c - GATE_COV) <= GATE_TOL_COV)

    res = {
        "config": cid, "n_cells": int(len(y)),
        "n_hit_r_gt_0.3": int(n_hit), "frac_hit": float(mask.mean()),
        "hit_by_horizon": {int(h): int(((hh == h) & mask).sum()) for h in HORIZONS},
        "max_r": float(rr.max()),
        "consistency_gate": {"gaussian_wis": gate_w, "gaussian_cov95": gate_c,
                             "target_wis": GATE_WIS, "target_cov95": GATE_COV, "PASS": bool(gate_ok)},
    }
    if gate_ok:
        gw_all, gc_all = gate_w, gate_c
        sw_all, sc_all = score(sq)                       # aggregate, mixture spliced on r>0.3
        gw_sub, gc_sub = score(gq, mask)                 # affected subset, gaussian
        sw_sub, sc_sub = score(sq, mask)                 # affected subset, mixture
        res["aggregate"] = {"gaussian": {"wis": gw_all, "cov95": gc_all},
                            "spliced":  {"wis": sw_all, "cov95": sc_all},
                            "delta_wis": sw_all - gw_all, "delta_cov95": sc_all - gc_all}
        res["affected_subset"] = {"n": int(mask.sum()),
                                  "gaussian": {"wis": gw_sub, "cov95": gc_sub},
                                  "mixture":  {"wis": sw_sub, "cov95": sc_sub},
                                  "delta_wis": sw_sub - gw_sub, "delta_cov95": sc_sub - gc_sub}
    else:
        res["HALT"] = "consistency gate FAILED — baseline does not reproduce headline; deltas not reported"

    (OUT / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
