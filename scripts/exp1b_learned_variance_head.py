"""Experiment ①(b-primary): μ-frozen learned-variance head vs RAW native APMD.

Pre-registration (LOCKED, numbers-blind 2026-08-05):
    CGM_v2_paper/PREREG_nll_quantile_head_ablation.md

Question (the true APMD ablation): does APMD's structural HMM-emission variance
beat a LEARNED variance head on the SAME CG-Mamba backbone and the SAME frozen
point forecast μ_CGM?

Design (per LOCK):
  - Load a trained CG-Mamba (stage3_best) per seed. μ = its point forecast, FROZEN.
  - Train ONLY a logvar head g(h_last) → log σ²_z on TRAIN-split residuals (z-space),
    Gaussian-NLL, μ detached. NO val/test residuals touch the σ value (LOCK 2 A);
    early-stop uses a TRAIN-INTERNAL holdout only.
  - Eval identically to raw native APMD: swap σ² source only. σ²_z = exp(logvar)
    replaces APMD's sigma2_total; μ_CGM, denorm, Gaussian-quantile construction,
    test_strict filter, WIS/Cov95 scoring all IDENTICAL (reuses track_b_lib).
  - Comparison target = RAW native APMD (s=1), reproduced in the same run as a
    harness-integrity self-check (must land near the published regional 0.954).

σ-head numerical guard (LOCK 2, declared BEFORE any number): logvar clamped to
[LOGVAR_MIN, LOGVAR_MAX] = [-10.0, 5.0] → σ² ∈ [4.5e-5, 148] (z-space). Generous;
prevents σ→0/∞, not a calibration knob.

Outputs: runs/exp1b_learned_variance/result_seed{seed}.json (per-region, per-horizon
Cov95/WIS/MAE for APMD and learned), + aggregate printed. Decision metric =
native regional h1-4 avg Cov95 (learned) vs near-nominal band [0.90, 1.00].

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/exp1b_learned_variance_head.py --smoke
    CUDA_VISIBLE_DEVICES=1 python scripts/exp1b_learned_variance_head.py --seeds 42 123 456 789 1024
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import track_b_lib as tb
from track_b_lib import (
    HORIZONS, TS_BOUNDARY, SEEDS,
    load_cgm_model_seed, cgm_dataset, cgm_decomp_forward,
    national_df, build_region_df, load_norm,
    score_per_cell,
)
from src.eval.wis_standard import quantiles_from_gaussian, FLUSIGHT_23
from src.data.loader import load_norm_params
from src.eval.hmm_interval import compute_decomposition

REGIONS = [f"hhs{i}" for i in range(1, 11)]
LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0   # LOCK numerical guard (z-space log σ²)
OUT_DIR = _ROOT / "runs" / "exp1b_learned_variance"


# ---------------------------------------------------------------------------
# Forward that mirrors cgm_decomp_forward EXACTLY + also returns h_last.
# Verified bit-identical to the canonical function (see _verify_equivalence).
# ---------------------------------------------------------------------------
def cgm_forward_with_hlast(model, cfg, hmm, ds, device):
    """Copy of track_b_lib.cgm_decomp_forward + capture h_last = fused[:, -1, :].

    Returns (mu_z [M,4], sig2_total_z [M,4], y_z [M,4], eps_h1 [M], h_last [M,D]).
    M = number of VALID (non-NaN pred) windows, same filtering as canonical.
    """
    means = hmm.means
    covars = hmm.covars
    mu_k_ili = means[:, 0]
    if covars.ndim == 3:
        sigma2_k_ili = np.array([covars[k, 0, 0] for k in range(covars.shape[0])])
    else:
        sigma2_k_ili = covars[:, 0]

    n = len(ds)
    if n == 0:
        D = cfg.d_model
        return (np.zeros((0, 4)), np.zeros((0, 4)), np.zeros((0, 4)),
                np.zeros((0,), dtype=np.int64), np.zeros((0, D)))
    eps_arr = ds.df["epiweek"].astype(int).to_numpy()
    norm = load_norm_params(tb.NORM_PATH)
    ili_p = norm["ili_weighted_pct"]
    target_z_full = (ds.df["ili_weighted_pct"].to_numpy() - ili_p["mean"]) / ili_p["std"]

    D = cfg.d_model
    mu = np.zeros((n, 4))
    gamma_all = np.zeros((n, 4, 3))
    y_z = np.zeros((n, 4))
    eps_h1 = np.zeros(n, dtype=np.int64)
    h_last = np.zeros((n, D))
    valid = np.ones(n, dtype=bool)

    with torch.no_grad():
        for i in range(n):
            d = ds[i]
            x = d["x"].unsqueeze(0).to(device)
            env = d["env"].unsqueeze(0).to(device)
            pred, inter = model(x, env, return_intermediates=True)
            if torch.isnan(pred).any():
                valid[i] = False
                continue
            mu[i] = pred[0].cpu().numpy()
            gamma_all[i] = inter["gamma_all"][0].cpu().numpy()
            h_last[i] = inter["fused"][0, -1, :].cpu().numpy()   # <-- the ONLY addition
            tgt_ep = int(d["target_epiweek"])
            tgt_idx = int(np.where(eps_arr == tgt_ep)[0][0])
            for h_idx, h in enumerate(HORIZONS):
                src = tgt_idx - (max(HORIZONS) - h)
                if 0 <= src < len(eps_arr):
                    y_z[i, h_idx] = target_z_full[src]
            eps_h1[i] = eps_arr[tgt_idx - (max(HORIZONS) - 1)]

    mu = mu[valid]; gamma_all = gamma_all[valid]; y_z = y_z[valid]
    eps_h1 = eps_h1[valid]; h_last = h_last[valid]
    decomp = compute_decomposition(mu, gamma_all, mu_k_ili, sigma2_k_ili)
    return decomp.mu_CGM, decomp.sigma2_total, y_z, eps_h1, h_last


def _verify_equivalence(model, cfg, hmm, ds, device):
    """Assert cgm_forward_with_hlast matches canonical cgm_decomp_forward bit-for-bit."""
    mu_a, s2_a, y_a, e_a = cgm_decomp_forward(model, cfg, hmm, ds, device)
    mu_b, s2_b, y_b, e_b, _ = cgm_forward_with_hlast(model, cfg, hmm, ds, device)
    assert np.array_equal(mu_a, mu_b), "mu_z mismatch vs canonical"
    assert np.array_equal(s2_a, s2_b), "sigma2_total mismatch vs canonical"
    assert np.array_equal(y_a, y_b), "y_z mismatch vs canonical"
    assert np.array_equal(e_a, e_b), "eps_h1 mismatch vs canonical"
    return len(mu_a)


# ---------------------------------------------------------------------------
# Learned log-variance head + Gaussian-NLL (z-space, μ frozen)
# ---------------------------------------------------------------------------
class LogVarHead(nn.Module):
    def __init__(self, d_model: int, n_h: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_h)

    def forward(self, h_last):                      # [B, D] -> [B, n_h] log σ²_z
        return self.fc(h_last).clamp(LOGVAR_MIN, LOGVAR_MAX)


def gaussian_nll(logvar, resid):                    # z-space, μ frozen (resid const)
    # 0.5 * (logvar + resid^2 / exp(logvar))  (+const dropped)
    return 0.5 * (logvar + resid ** 2 / torch.exp(logvar))


def train_logvar_head(h_last, resid_z, d_model, seed, device,
                      max_epochs=300, patience=30, lr=1e-3, holdout_frac=0.2):
    """Train σ head on TRAIN residuals only; early-stop on a TRAIN-INTERNAL holdout.

    h_last: [N, D] (frozen features), resid_z: [N, 4] (y_z - μ_z, z-space).
    Returns trained LogVarHead (eval mode) + training log dict.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    N = h_last.shape[0]
    perm = np.random.RandomState(seed).permutation(N)
    n_hold = max(1, int(round(N * holdout_frac)))
    hold_idx, tr_idx = perm[:n_hold], perm[n_hold:]

    H = torch.tensor(h_last, dtype=torch.float32, device=device)
    R = torch.tensor(resid_z, dtype=torch.float32, device=device)
    H_tr, R_tr = H[tr_idx], R[tr_idx]
    H_ho, R_ho = H[hold_idx], R[hold_idx]

    head = LogVarHead(d_model, resid_z.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best_ho, best_state, best_ep, since = float("inf"), None, -1, 0
    for ep in range(max_epochs):
        head.train()
        opt.zero_grad()
        loss = gaussian_nll(head(H_tr), R_tr).mean()
        loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            ho = gaussian_nll(head(H_ho), R_ho).mean().item()
        if ho < best_ho - 1e-6:
            best_ho, best_state, best_ep, since = ho, {k: v.detach().clone() for k, v in head.state_dict().items()}, ep, 0
        else:
            since += 1
            if since >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, {"N_train": int(len(tr_idx)), "N_holdout": int(len(hold_idx)),
                  "best_epoch": best_ep, "best_holdout_nll": best_ho,
                  "stopped_epoch": ep}


# ---------------------------------------------------------------------------
# Scoring: build quantiles for APMD and learned, identical pipeline
# ---------------------------------------------------------------------------
def score_split(mu_z, sig2_apmd_z, y_z, eps_h1, h_last, head, norm, device,
                label, test_strict=True):
    """Return per-horizon dict for APMD and learned on one (region/national) split."""
    tmean = float(norm["ili_weighted_pct"]["mean"])
    tstd = float(norm["ili_weighted_pct"]["std"])

    if test_strict:
        idx = np.where(eps_h1 >= TS_BOUNDARY)[0]
    else:
        idx = np.arange(len(eps_h1))
    if len(idx) == 0:
        raise RuntimeError(f"{label}: empty split")

    mu_raw = (mu_z * tstd + tmean)[idx]
    y_raw = (y_z * tstd + tmean)[idx]

    # APMD raw native: sig2_raw = sig2_total_z * std^2
    sig2_apmd_raw = (sig2_apmd_z * tstd ** 2)[idx]

    # Learned: sig2_z = exp(logvar(h_last)) -> raw
    with torch.no_grad():
        logvar = head(torch.tensor(h_last[idx], dtype=torch.float32, device=device))
        sig2_learned_z = torch.exp(logvar).cpu().numpy()
    sig2_learned_raw = sig2_learned_z * tstd ** 2

    out = {"n": int(len(idx)), "apmd": {}, "learned": {}}
    for name, sig2 in [("apmd", sig2_apmd_raw), ("learned", sig2_learned_raw)]:
        qf = quantiles_from_gaussian(mu_raw, sig2, taus=FLUSIGHT_23)  # dict tau->[N,4]
        for h_idx, h in enumerate(HORIZONS):
            qf_h = {float(t): qf[float(t)][:, h_idx] for t in FLUSIGHT_23}
            cell = score_per_cell(qf_h, y_raw, h_idx, f"{label}/{name}/h{h}")
            out[name][f"h{h}"] = cell
    return out


def run_seed(seed, device, smoke=False, sigma_fit_split="train"):
    """A+ : fit σ head on `sigma_fit_split` ('train'=in-sample=b-primary, or 'val'=held-out).
    Everything else (head, recipe, μ frozen, eval) identical → single-variable = fit-data source."""
    norm = load_norm()
    model, cfg, hmm = load_cgm_model_seed(seed, device)
    d_model = cfg.d_model

    # ---- national train & val forward (mu, y, h_last) ----
    tr_ds = cgm_dataset(national_df(), "train", cfg, norm)
    va_ds = cgm_dataset(national_df(), "val", cfg, norm)
    mu_tr, s2_tr, y_tr, e_tr, hl_tr = cgm_forward_with_hlast(model, cfg, hmm, tr_ds, device)
    mu_va, s2_va, y_va, e_va, hl_va = cgm_forward_with_hlast(model, cfg, hmm, va_ds, device)

    # ---- integrity self-check: my forward == canonical cgm_decomp_forward (on val) ----
    mu_c, s2_c, y_c, e_c = cgm_decomp_forward(model, cfg, hmm, va_ds, device)
    assert (np.array_equal(mu_va, mu_c) and np.array_equal(s2_va, s2_c)
            and np.array_equal(y_va, y_c) and np.array_equal(e_va, e_c)), "forward != canonical"
    print(f"[seed {seed}] equivalence vs canonical OK (N_val={len(mu_va)}, N_train={len(mu_tr)})")

    resid_tr = y_tr - mu_tr
    resid_va = y_va - mu_va
    # in-sample bias magnitude: per-horizon residual std, in-sample(train) vs held-out(val)
    resid_std = {"train_per_h": [float(v) for v in resid_tr.std(axis=0)],
                 "val_per_h":   [float(v) for v in resid_va.std(axis=0)]}

    # ---- fit σ head on the CHOSEN split's residuals (the ONLY thing that varies) ----
    if sigma_fit_split == "train":
        hl_fit, resid_fit = hl_tr, resid_tr
    elif sigma_fit_split == "val":
        hl_fit, resid_fit = hl_va, resid_va
    else:
        raise ValueError(f"sigma_fit_split must be train|val, got {sigma_fit_split}")
    head, tlog = train_logvar_head(hl_fit, resid_fit, d_model, seed, device)
    print(f"[seed {seed}] σ-head fit on '{sigma_fit_split}' residuals: {tlog}")

    # ---- eval: 10 regions (test_strict) ----
    regions = REGIONS[:2] if smoke else REGIONS
    per_region = {}
    for r in regions:
        rdf = build_region_df(r)
        test_ds = cgm_dataset(rdf, "test", cfg, norm)
        mu_z, s2_z, y_z, eps_h1, hlast = cgm_forward_with_hlast(model, cfg, hmm, test_ds, device)
        per_region[r] = score_split(mu_z, s2_z, y_z, eps_h1, hlast, head, norm, device,
                                    label=f"{seed}/{r}", test_strict=True)

    # ---- national test (in-distribution) ----
    nat_ds = cgm_dataset(national_df(), "test", cfg, norm)
    mu_z, s2_z, y_z, eps_h1, hlast = cgm_forward_with_hlast(model, cfg, hmm, nat_ds, device)
    national = score_split(mu_z, s2_z, y_z, eps_h1, hlast, head, norm, device,
                           label=f"{seed}/national", test_strict=True)

    result = {"seed": seed, "sigma_fit_split": sigma_fit_split, "train_log": tlog,
              "resid_std": resid_std, "regions": per_region,
              "national": national, "logvar_clamp": [LOGVAR_MIN, LOGVAR_MAX]}
    return result


def aggregate(results):
    """Regional h1-4 avg Cov95/WIS for APMD and learned, averaged over regions then seeds."""
    def region_avg(res, model):  # per-seed: mean over regions of (h1-4 mean)
        per_h = {f"h{h}": [] for h in HORIZONS}
        cov_all, wis_all = [], []
        for r, d in res["regions"].items():
            hs_cov = [d[model][f"h{h}"]["cov95"] for h in HORIZONS]
            hs_wis = [d[model][f"h{h}"]["wis"] for h in HORIZONS]
            for h in HORIZONS:
                per_h[f"h{h}"].append(d[model][f"h{h}"]["cov95"])
            cov_all.append(np.mean(hs_cov)); wis_all.append(np.mean(hs_wis))
        return (float(np.mean(cov_all)), float(np.mean(wis_all)),
                {k: float(np.mean(v)) for k, v in per_h.items()})

    summary = {"per_seed": [], "logvar_clamp": [LOGVAR_MIN, LOGVAR_MAX]}
    agg = {m: {"cov": [], "wis": [], "perh": {f"h{h}": [] for h in HORIZONS}} for m in ["apmd", "learned"]}
    for res in results:
        row = {"seed": res["seed"]}
        for m in ["apmd", "learned"]:
            cov, wis_, perh = region_avg(res, m)
            row[m] = {"regional_cov95_h1_4avg": cov, "regional_wis_h1_4avg": wis_, "per_h_cov95": perh}
            agg[m]["cov"].append(cov); agg[m]["wis"].append(wis_)
            for h in HORIZONS: agg[m]["perh"][f"h{h}"].append(perh[f"h{h}"])
        summary["per_seed"].append(row)
    summary["mean_over_seeds"] = {
        m: {"regional_cov95_h1_4avg": float(np.mean(agg[m]["cov"])),
            "regional_cov95_std": float(np.std(agg[m]["cov"])),
            "regional_wis_h1_4avg": float(np.mean(agg[m]["wis"])),
            "per_h_cov95": {k: float(np.mean(v)) for k, v in agg[m]["perh"].items()}}
        for m in ["apmd", "learned"]}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true", help="seed42, 2 regions only")
    ap.add_argument("--sigma_fit_split", choices=["train", "val"], default="train",
                    help="A+ : residual source for the σ head. train=in-sample=b-primary; val=held-out.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [42] if args.smoke else args.seeds
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # tag: b-primary (train fit) keeps the original filenames; val fit is tagged so it never overwrites
    tag = ("_fit" + args.sigma_fit_split if args.sigma_fit_split != "train" else "") \
          + ("_smoke" if args.smoke else "")
    print(f"[exp1b] device={device} seeds={seeds} smoke={args.smoke} sigma_fit_split={args.sigma_fit_split}")

    results = []
    for seed in seeds:
        res = run_seed(seed, device, smoke=args.smoke, sigma_fit_split=args.sigma_fit_split)
        results.append(res)
        (OUT_DIR / f"result_seed{seed}{tag}.json").write_text(json.dumps(res, indent=2))
        # quick per-seed readout (regional + national)
        for scope, key in [("regional", "regions"), ("national", "national")]:
            for m in ["apmd", "learned"]:
                if scope == "regional":
                    v = np.mean([np.mean([res["regions"][r][m][f"h{h}"]["cov95"] for h in HORIZONS]) for r in res["regions"]])
                else:
                    v = np.mean([res["national"][m][f"h{h}"]["cov95"] for h in HORIZONS])
                print(f"[seed {seed}] {scope:8s} {m:8s} Cov95 h1-4 avg = {v:.4f}")

    summary = aggregate(results)
    summary["sigma_fit_split"] = args.sigma_fit_split
    print("\n=== AGGREGATE regional (mean over seeds) ===")
    print(json.dumps(summary["mean_over_seeds"], indent=2))
    (OUT_DIR / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[exp1b] wrote {OUT_DIR} (tag='{tag}')")


if __name__ == "__main__":
    main()
