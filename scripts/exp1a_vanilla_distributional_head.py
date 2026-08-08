"""Experiment ①(a, amended) — μ-frozen learned-variance head on the STRONGEST
independent backbone (Vanilla Mamba), + two pinned train-holdout temperatures.

Pre-registration LOCKED (numbers-blind, git 3544a77):
    runs/apmd_diagnostic/PREREG_fairness_baseline_transfer.md

Question (fairness / backbone isolation): does the μ-frozen residual-fit Gaussian-NLL
variance recipe — byte-identical to the already-run CG b-primary — also under-cover
under zero-shot regional transfer when placed on Vanilla Mamba, and does it survive a
variance-matched (RMS) or a quantile-matched (q95) train-holdout temperature?

Design pins honored (see pre-reg §3):
  (pin 1) checkpoint-MAE assert MIRRORS the production path baseline_test_eval.py
          (_build_split_loader / _per_horizon_raw_mae copied verbatim below);
          per-seed match vs runs/baselines_test_eval.csv (tol 1e-3) + 5-seed mean 0.435±0.005.
  (pin 2) the σ-head recipe + 80/20 holdout split are IMPORTED verbatim from exp1b
          (LogVarHead, gaussian_nll, train_logvar_head) — no re-implementation.
  (pin 3) national z-space throughout: fit on national norm params, transfer to regions
          with the SAME national params (zero-shot); NEVER per-region renormalization.
  (pin 4) N_train/N_holdout are NOT forced to exp1b's 541/135 — Vanilla uses the
          WeeklyMultiHorizonDataset window path; whatever N results is reported as-is.

Decision metric (pre-reg §4): Vanilla native regional h1-4 avg Cov95 (zero-shot) for
raw AND both temperatures. CONFIRM iff all three ≤ 0.920; any NEAR [0.924,0.976] → THREAT.
Also re-scores CG b-primary under both temperatures (pre-reg §3.2). Reported regardless of direction.

Usage:
    python scripts/exp1a_vanilla_distributional_head.py --smoke            # seed42, 2 regions
    python scripts/exp1a_vanilla_distributional_head.py --seeds 42 123 456 789 1024
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

# (pin 2) verbatim σ-head recipe + split + CG forward from the already-run b-primary
from exp1b_learned_variance_head import (
    LogVarHead, gaussian_nll, train_logvar_head, cgm_forward_with_hlast, LOGVAR_MIN, LOGVAR_MAX,
)
# harness + scoring (identical to b-primary)
import track_b_lib as tb
from track_b_lib import (
    HORIZONS, TS_BOUNDARY, SEEDS, build_region_df, load_cgm_model_seed, cgm_dataset,
    national_df, score_per_cell,
)
from src.eval.wis_standard import quantiles_from_gaussian, FLUSIGHT_23
from src.data.loader import load_dataset_csv, load_norm_params
# (pin 1) production model + dataset
from baselines.vanilla_mamba import VanillaMambaForecaster
from baselines.lstm import WeeklyMultiHorizonDataset

REGIONS = [f"hhs{i}" for i in range(1, 11)]
VANILLA_DIR = _ROOT / "runs" / "vanilla_mamba_final" / "d64_nl3_lr5e-04"
REC_CSV = _ROOT / "runs" / "baselines_test_eval.csv"
CSV_PATH = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data" / "processed" / "normalization_params.json"
OUT_DIR = _ROOT / "runs" / "exp1a_vanilla_distributional"
COVID_STRICT_START_EPIWEEK = 202240
MAE_TOL, MEAN_TARGET, MEAN_TOL = 1e-3, 0.435, 5e-3


# ---------------------------------------------------------------------------
# (pin 1) VERBATIM copies from baseline_test_eval.py — the production MAE path
# ---------------------------------------------------------------------------
def _build_split_loader(df, split_name, lookback, pred_len, norm,
                        epi_min=None, epi_max=None, batch_size=32):
    if epi_min is not None or epi_max is not None:
        sub = df.copy()
        if epi_min is not None:
            sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min), "split"] = "_excluded"
        if epi_max is not None:
            sub.loc[(sub["split"] == split_name) & (sub["epiweek"] > epi_max), "split"] = "_excluded"
        ds_df = sub
    else:
        ds_df = df
    ds = WeeklyMultiHorizonDataset(ds_df, split_name, norm, lookback=lookback, pred_len=pred_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), len(ds)


def _per_horizon_raw_mae(model, loader, target_mean: float, target_std: float, device):
    model.eval()
    per_h_sum = None
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            pred_raw = pred * target_std + target_mean
            y_raw = y * target_std + target_mean
            per_h = (pred_raw - y_raw).abs().sum(dim=0)
            per_h_sum = per_h if per_h_sum is None else per_h_sum + per_h
            n += y.size(0)
    return (per_h_sum / max(n, 1)).cpu().tolist(), n


# ---------------------------------------------------------------------------
# Vanilla load (production construction) + forward capturing h_last
# ---------------------------------------------------------------------------
def load_vanilla(seed, device):
    seed_dir = VANILLA_DIR / f"seed{seed}"
    cfg = json.loads((seed_dir / "results.json").read_text())["config"]
    model = VanillaMambaForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg.get("d_state", 16),
        dt_rank=cfg.get("dt_rank", 16), expand=cfg.get("expand", 2), dropout=cfg.get("dropout", 0.0),
    )
    sd = torch.load(seed_dir / "vanilla_mamba_best.pt", map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model, cfg


def vanilla_forward_with_hlast(model, ds, device):
    """Return (mu_z [N,4], y_z [N,4], eps_h1 [N], h_last [N,D]) in z-space.
    Replicates VanillaMambaForecaster.forward and asserts bit-identity vs model(x)."""
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    mu, hl, y = [], [], []
    model.eval()
    with torch.no_grad():
        for x, yb in loader:
            x = x.to(device)
            h = model.backbone(x, gates=None)     # [B,L,D]
            h_last = h[:, -1, :]                   # [B,D]
            m = model.head(h_last)                 # [B,H]
            assert torch.equal(m, model(x)), "vanilla forward-with-hlast != model(x)"
            mu.append(m.cpu().numpy()); hl.append(h_last.cpu().numpy()); y.append(yb.numpy())
    mu = np.concatenate(mu); hl = np.concatenate(hl); y = np.concatenate(y)
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    return mu, y, eps_h1, hl


def checkpoint_mae_assert(model, cfg, seed, norm, device):
    """(pin 1) per-seed national test_strict MAE reproduces the record within 1e-3."""
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    df = load_dataset_csv(CSV_PATH)
    lookback, pred_len = cfg.get("seq_len", 104), cfg["pred_len"]
    loader, n = _build_split_loader(df, "test", lookback, pred_len, norm,
                                    epi_min=COVID_STRICT_START_EPIWEEK)
    mae_perh, _ = _per_horizon_raw_mae(model, loader, tmean, tstd, device)
    mae_avg = float(np.mean(mae_perh))
    import pandas as pd
    rec = pd.read_csv(REC_CSV)
    row = rec[(rec.model == "vanilla_mamba") & (rec.cfg_name == "d64_nl3_lr5e-04") & (rec.seed == seed)]
    assert len(row) == 1, f"no record row for seed {seed}"
    rec_avg = float(row.test_strict_avg.iloc[0])
    assert abs(mae_avg - rec_avg) <= MAE_TOL, \
        f"seed {seed}: national test_strict MAE {mae_avg:.4f} != record {rec_avg:.4f} (tol {MAE_TOL}) -> DEGRADED CKPT"
    return mae_avg, rec_avg


# ---------------------------------------------------------------------------
# temperatures (pin 2: same deterministic 80/20 holdout as train_logvar_head)
# ---------------------------------------------------------------------------
def compute_temperatures(head, hl_fit, resid_fit, seed, device):
    """s_h^RMS = RMS(z_holdout,h); s_h^q95 = Q95(|z_holdout,h|)/1.96, per horizon.
    Holdout indices are re-derived with the IDENTICAL rule train_logvar_head uses
    (np.random.RandomState(seed).permutation, holdout_frac=0.2)."""
    N = hl_fit.shape[0]
    perm = np.random.RandomState(seed).permutation(N)
    n_hold = max(1, int(round(N * 0.2)))
    hold = perm[:n_hold]
    with torch.no_grad():
        logvar = head(torch.tensor(hl_fit[hold], dtype=torch.float32, device=device)).cpu().numpy()
    sig_hold = np.sqrt(np.exp(logvar))               # [n_hold, 4] z-space σ
    z = resid_fit[hold] / sig_hold                    # standardized holdout residual
    s_rms = np.sqrt(np.mean(z ** 2, axis=0))          # [4]
    s_q95 = np.quantile(np.abs(z), 0.95, axis=0) / 1.96   # [4]
    return {"rms": s_rms.tolist(), "q95": s_q95.tolist(), "n_holdout": int(n_hold)}


def score_learned(mu_z, y_z, eps_h1, h_last, head, norm, device, s_scale=None, test_strict=True):
    """Per-horizon Cov95/WIS for the learned head (raw or σ scaled by per-h s_scale)."""
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    idx = np.where(eps_h1 >= TS_BOUNDARY)[0] if test_strict else np.arange(len(eps_h1))
    mu_raw = (mu_z * tstd + tmean)[idx]; y_raw = (y_z * tstd + tmean)[idx]
    with torch.no_grad():
        sig2_z = torch.exp(head(torch.tensor(h_last[idx], dtype=torch.float32, device=device))).cpu().numpy()
    sig2_raw = sig2_z * tstd ** 2                     # [Nsel,4]
    if s_scale is not None:
        sig2_raw = sig2_raw * (np.asarray(s_scale)[None, :] ** 2)
    qf = quantiles_from_gaussian(mu_raw, sig2_raw, taus=FLUSIGHT_23)
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        qf_h = {float(t): qf[float(t)][:, h_idx] for t in FLUSIGHT_23}
        out[f"h{h}"] = score_per_cell(qf_h, y_raw, h_idx, f"h{h}")
    return out, int(len(idx))


def cov_h1_4avg(scored):
    return float(np.mean([scored[f"h{h}"]["cov95"] for h in HORIZONS]))


# ---------------------------------------------------------------------------
# per-seed runners
# ---------------------------------------------------------------------------
def run_seed(seed, device, norm, smoke=False):
    regions = REGIONS[:2] if smoke else REGIONS
    res = {"seed": seed, "vanilla": {}, "cg_bprimary": {}}

    # ===== VANILLA backbone (the decision arm) =====
    vmodel, vcfg = load_vanilla(seed, device)
    mae_avg, rec_avg = checkpoint_mae_assert(vmodel, vcfg, seed, norm, device)
    res["vanilla"]["ckpt_mae_national_test_strict"] = {"computed": mae_avg, "record": rec_avg}
    print(f"[seed {seed}] vanilla ckpt MAE {mae_avg:.4f} vs record {rec_avg:.4f} (tol {MAE_TOL}) OK")

    lb, pl = vcfg.get("seq_len", 104), vcfg["pred_len"]
    df_nat = load_dataset_csv(CSV_PATH)
    tr_ds = WeeklyMultiHorizonDataset(df_nat, "train", norm, lookback=lb, pred_len=pl)
    mu_tr, y_tr, _, hl_tr = vanilla_forward_with_hlast(vmodel, tr_ds, device)
    resid_tr = y_tr - mu_tr
    head, tlog = train_logvar_head(hl_tr, resid_tr, vcfg["d_model"], seed, device)
    temps = compute_temperatures(head, hl_tr, resid_tr, seed, device)
    res["vanilla"]["train_log"] = tlog; res["vanilla"]["temperatures"] = temps
    print(f"[seed {seed}] vanilla σ-head: N_train={tlog['N_train']} N_hold={tlog['N_holdout']} "
          f"s_rms={[round(v,3) for v in temps['rms']]} s_q95={[round(v,3) for v in temps['q95']]}")

    per_region = {}
    for r in regions:
        rds = WeeklyMultiHorizonDataset(build_region_df(r), "test", norm, lookback=lb, pred_len=pl)
        mu_z, y_z, eps_h1, hl = vanilla_forward_with_hlast(vmodel, rds, device)
        cell = {}
        for variant, s in [("raw", None), ("rms", temps["rms"]), ("q95", temps["q95"])]:
            cell[variant], _ = score_learned(mu_z, y_z, eps_h1, hl, head, norm, device, s_scale=s)
        per_region[r] = cell
    res["vanilla"]["regions"] = per_region

    # national in-distribution (report both directions, per LOCK anti-hiding)
    nat_ds = WeeklyMultiHorizonDataset(df_nat, "test", norm, lookback=lb, pred_len=pl)
    mu_z, y_z, eps_h1, hl = vanilla_forward_with_hlast(vmodel, nat_ds, device)
    res["vanilla"]["national"] = {v: score_learned(mu_z, y_z, eps_h1, hl, head, norm, device,
                                                    s_scale=s)[0]
                                  for v, s in [("raw", None), ("rms", temps["rms"]), ("q95", temps["q95"])]}

    # ===== CG b-primary re-score under the two temperatures (pre-reg §3.2) =====
    cmodel, ccfg, hmm = load_cgm_model_seed(seed, device)
    ctr = cgm_dataset(national_df(), "train", ccfg, norm)
    cmu_tr, _, cy_tr, _, chl_tr = cgm_forward_with_hlast(cmodel, ccfg, hmm, ctr, device)
    cresid_tr = cy_tr - cmu_tr
    chead, ctlog = train_logvar_head(chl_tr, cresid_tr, ccfg.d_model, seed, device)
    ctemps = compute_temperatures(chead, chl_tr, cresid_tr, seed, device)
    res["cg_bprimary"]["train_log"] = ctlog; res["cg_bprimary"]["temperatures"] = ctemps
    cg_reg = {}
    for r in regions:
        crds = cgm_dataset(build_region_df(r), "test", ccfg, norm)
        cmu_z, _, cy_z, ceps, chl = cgm_forward_with_hlast(cmodel, ccfg, hmm, crds, device)
        cell = {}
        for variant, s in [("raw", None), ("rms", ctemps["rms"]), ("q95", ctemps["q95"])]:
            cell[variant], _ = score_learned(cmu_z, cy_z, ceps, chl, chead, norm, device, s_scale=s)
        cg_reg[r] = cell
    res["cg_bprimary"]["regions"] = cg_reg
    return res


def aggregate(results):
    def reg_avg(res, backbone, variant):
        per_seed = []
        for r in results:
            regs = r[backbone]["regions"]
            per_seed.append(np.mean([cov_h1_4avg(regs[rg][variant]) for rg in regs]))
        return float(np.mean(per_seed)), float(np.std(per_seed)), [round(float(v), 4) for v in per_seed]
    summary = {}
    for backbone in ["vanilla", "cg_bprimary"]:
        summary[backbone] = {}
        for variant in ["raw", "rms", "q95"]:
            m, s, ps = reg_avg(results, backbone, variant)
            summary[backbone][variant] = {"regional_cov95_h1_4avg": m, "std": s, "per_seed": ps}
    return summary


def decide(summary):
    """pre-reg §4: bands on Vanilla regional Cov95 for raw + both temps."""
    def band(x):
        if x <= 0.920: return "CONF"
        if x < 0.924:  return "GAP"
        if x <= 0.976: return "NEAR"
        return "OVER"
    vals = {v: summary["vanilla"][v]["regional_cov95_h1_4avg"] for v in ["raw", "rms", "q95"]}
    bands = {v: band(x) for v, x in vals.items()}
    bset = set(bands.values())
    if "NEAR" in bset:            branch = "THREAT"
    elif "OVER" in bset:          branch = "BOUNDARY"
    elif bset == {"CONF"}:        branch = "CONFIRM"
    else:                          branch = "PARTIAL"
    return {"vanilla_regional_cov95": vals, "bands": bands, "BRANCH": branch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [42] if args.smoke else args.seeds
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    norm = load_norm_params(NORM_PATH)
    print(f"[exp1a] device={device} seeds={seeds} smoke={args.smoke}")

    results = []
    for seed in seeds:
        res = run_seed(seed, device, norm, smoke=args.smoke)
        results.append(res)
        tag = "_smoke" if args.smoke else ""
        (OUT_DIR / f"result_seed{seed}{tag}.json").write_text(json.dumps(res, indent=2))
        for bk in ["vanilla", "cg_bprimary"]:
            for v in ["raw", "rms", "q95"]:
                cov = np.mean([cov_h1_4avg(res[bk]["regions"][rg][v]) for rg in res[bk]["regions"]])
                print(f"[seed {seed}] {bk:12s} {v:4s} regional Cov95 h1-4avg = {cov:.4f}")

    summary = aggregate(results)
    if not args.smoke:
        summary["decision"] = decide(summary)
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\n=== AGGREGATE ==="); print(json.dumps(summary, indent=2))
    else:
        (OUT_DIR / "summary_smoke.json").write_text(json.dumps(summary, indent=2))
        print("\n=== SMOKE AGGREGATE ==="); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
