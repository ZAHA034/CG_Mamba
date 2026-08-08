"""Experiment ①(a) JOINT arm — Vanilla Mamba + distributional head trained END-TO-END.

Pre-registration LOCKED + amended: runs/apmd_diagnostic/PREREG_fairness_baseline_transfer.md
  §3.1 (joint deferred w/ trigger), A2.2 (design pinned), A2.2-refinement (val-WIS selection, git a59f604).
Triggered by A2.1 (hollow-CONFIRM gate fired: Vanilla μ-frozen in-dist 5-seed mean 0.892 < 0.924).

Question (the RESIDUAL fairness attack): does a PROPERLY-TRAINED distributional baseline
(μ+σ learned jointly, selected on a distributional criterion) transfer zero-shot to regions,
where the μ-frozen head did not?

Honest scope (A2.2): joint changes TWO variables at once (backbone AND training protocol) →
NOT the 1-variable isolation of exp1a; it is the direct answer to "would a properly trained
distributional baseline transfer." Reported separately, never blended with the μ-frozen result.

Faithful replication of the original Vanilla training (run_vanilla_mamba_weekly.py):
  same build_lstm_loaders, Adam(lr from cfg), epochs=200, patience=20, batch=32, same 5 seeds,
  same backbone+μ-head architecture. ONLY: (i) loss MSE→Gaussian-NLL on a μ+logσ² head;
  (ii) selection val-MAE@h1 → val-WIS (A2.2-refinement — fairer, restores parent-lock arm(a)).

Reports (A2.2): joint national in-dist Cov95 (A2.1-analogue gate), joint zero-shot regional
transfer Cov95 (§4 bands), and the joint model's national MAE (μ-drift duty). Outputs to
runs/exp1a_joint_vanilla/.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts")); sys.path.insert(0, str(_ROOT / "src"))

from track_b_lib import HORIZONS, TS_BOUNDARY, SEEDS, build_region_df, score_per_cell
from src.eval.wis_standard import quantiles_from_gaussian, FLUSIGHT_23
from src.data.loader import load_dataset_csv, load_norm_params
from baselines.vanilla_mamba import VanillaMambaForecaster
from baselines.lstm import WeeklyMultiHorizonDataset, build_lstm_loaders

REGIONS = [f"hhs{i}" for i in range(1, 11)]
VANILLA_DIR = _ROOT / "runs" / "vanilla_mamba_final" / "d64_nl3_lr5e-04"
CSV_PATH = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data" / "processed" / "normalization_params.json"
OUT_DIR = _ROOT / "runs" / "exp1a_joint_vanilla"
LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0     # same numerical guard as exp1b
VANILLA_POINT_MAE = 0.435               # published point-model national test_strict MAE (μ-drift reference)


def set_seeds(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class JointVanillaMamba(nn.Module):
    """Vanilla backbone + μ head (reused) + parallel logσ² head; trained jointly."""
    def __init__(self, cfg):
        super().__init__()
        self.base = VanillaMambaForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg["d_state"],
            dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=cfg["dropout"])
        self.logvar_head = nn.Linear(cfg["d_model"], cfg["pred_len"])

    def forward(self, x):
        h = self.base.backbone(x, gates=None)          # [B,L,D]
        h_last = h[:, -1, :]                            # [B,D]
        mu = self.base.head(h_last)                     # [B,H]
        logvar = self.logvar_head(h_last).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


def gaussian_nll(mu, logvar, y):
    return (0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar))).mean()


def _score_split(mu_z, lv_z, y_z, eps_h1, norm, test_strict):
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    idx = np.where(eps_h1 >= TS_BOUNDARY)[0] if test_strict else np.arange(len(eps_h1))
    mu_raw = (mu_z * tstd + tmean)[idx]; y_raw = (y_z * tstd + tmean)[idx]
    sig2_raw = (np.exp(lv_z) * tstd ** 2)[idx]
    qf = quantiles_from_gaussian(mu_raw, sig2_raw, taus=FLUSIGHT_23)
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        qf_h = {float(t): qf[float(t)][:, h_idx] for t in FLUSIGHT_23}
        out[f"h{h}"] = score_per_cell(qf_h, y_raw, h_idx, f"h{h}")
    return out


def _forward_ds(model, ds, device):
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    mus, lvs, ys = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            mu, lv = model(x.to(device))
            mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy()); ys.append(y.numpy())
    mu = np.concatenate(mus); lv = np.concatenate(lvs); y = np.concatenate(ys)
    eps = ds.df["epiweek"].astype(int).to_numpy()
    return mu, lv, y, eps[ds.window_ends + 1]


def _val_wis(model, val_loader, norm, device):
    """Selection metric (A2.2-refinement): mean WIS over horizons on all val windows."""
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    mus, lvs, ys = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            mu, lv = model(x.to(device))
            mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy()); ys.append(y.numpy())
    mu = np.concatenate(mus); lv = np.concatenate(lvs); y = np.concatenate(ys)
    mu_raw = mu * tstd + tmean; y_raw = y * tstd + tmean; sig2_raw = np.exp(lv) * tstd ** 2
    qf = quantiles_from_gaussian(mu_raw, sig2_raw, taus=FLUSIGHT_23)
    wis = [score_per_cell({float(t): qf[float(t)][:, hi] for t in FLUSIGHT_23}, y_raw, hi, f"h{h}")["wis"]
           for hi, h in enumerate(HORIZONS)]
    return float(np.mean(wis))


def train_joint(cfg, seed, device):
    set_seeds(seed)
    train_loader, val_loader, meta = build_lstm_loaders(
        csv_path=CSV_PATH, norm_path=NORM_PATH,
        lookback=cfg["seq_len"], pred_len=cfg["pred_len"], batch_size=cfg["batch_size"])
    norm = load_norm_params(NORM_PATH)
    model = JointVanillaMamba(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best_wis, best_state, best_ep, since = float("inf"), None, 0, 0
    t0 = time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mu, lv = model(x)
            loss = gaussian_nll(mu, lv, y)
            opt.zero_grad(); loss.backward(); opt.step()
        vw = _val_wis(model, val_loader, norm, device)
        if vw < best_wis - 1e-9:
            best_wis, best_state, best_ep, since = vw, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            since += 1
            if since >= cfg["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    return model, {"best_val_wis": best_wis, "best_epoch": best_ep, "epochs_trained": epoch,
                   "elapsed_sec": time() - t0, "n_train": meta["n_train_windows"], "n_val": meta["n_val_windows"]}


def cov_avg(scored):
    return float(np.mean([scored[f"h{h}"]["cov95"] for h in HORIZONS]))


def mae_avg(scored):
    return float(np.mean([scored[f"h{h}"]["mae"] for h in HORIZONS]))


def run_seed(seed, device, norm, smoke=False):
    cfg = json.loads((VANILLA_DIR / f"seed{seed}" / "results.json").read_text())["config"]
    if smoke:
        cfg = {**cfg, "epochs": 5, "patience": 5}
    model, tlog = train_joint(cfg, seed, device)
    print(f"[seed {seed}] joint trained: val-WIS {tlog['best_val_wis']:.4f} @ep{tlog['best_epoch']} "
          f"({tlog['epochs_trained']} ep, {tlog['elapsed_sec']:.0f}s)")
    df_nat = load_dataset_csv(CSV_PATH)
    lb, pl = cfg["seq_len"], cfg["pred_len"]

    # national in-dist (A2.1-analogue gate + μ-drift MAE)
    nat_ds = WeeklyMultiHorizonDataset(df_nat, "test", norm, lookback=lb, pred_len=pl)
    mu, lv, y, eps = _forward_ds(model, nat_ds, device)
    nat = _score_split(mu, lv, y, eps, norm, test_strict=True)
    indist_cov, joint_mae = cov_avg(nat), mae_avg(nat)

    # regional zero-shot transfer (§4 decision)
    regions = REGIONS[:2] if smoke else REGIONS
    per_region = {}
    for r in regions:
        rds = WeeklyMultiHorizonDataset(build_region_df(r), "test", norm, lookback=lb, pred_len=pl)
        mu, lv, y, eps = _forward_ds(model, rds, device)
        per_region[r] = _score_split(mu, lv, y, eps, norm, test_strict=True)
    transfer_cov = float(np.mean([cov_avg(per_region[r]) for r in per_region]))
    print(f"[seed {seed}] joint national in-dist Cov95={indist_cov:.4f} MAE={joint_mae:.4f} "
          f"(pt-model {VANILLA_POINT_MAE}) | transfer Cov95={transfer_cov:.4f}")
    return {"seed": seed, "train_log": tlog, "national_indist": nat, "regions": per_region,
            "indist_cov95": indist_cov, "joint_national_mae": joint_mae, "transfer_cov95": transfer_cov}


def decide(indist_mean, transfer_mean):
    def band(x):
        if x <= 0.920: return "CONF"
        if x < 0.924:  return "GAP"
        if x <= 0.976: return "NEAR"
        return "OVER"
    tb = band(transfer_mean)
    if indist_mean < 0.924:
        branch = "JOINT_FAILS_INDIST"
    elif tb == "NEAR":
        branch = "THREAT"
    elif tb == "OVER":
        branch = "BOUNDARY"
    elif tb == "CONF":
        branch = "CONFIRM"          # joint in-dist calibrated AND fails transfer -> strongest closure
    else:
        branch = "PARTIAL"
    return {"joint_indist_cov95_mean": indist_mean, "joint_transfer_cov95_mean": transfer_mean,
            "transfer_band": tb, "BRANCH": branch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [42] if args.smoke else args.seeds
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    norm = load_norm_params(NORM_PATH)
    print(f"[exp1a-joint] device={device} seeds={seeds} smoke={args.smoke}")
    results = []
    for seed in seeds:
        res = run_seed(seed, device, norm, smoke=args.smoke)
        results.append(res)
        tag = "_smoke" if args.smoke else ""
        (OUT_DIR / f"result_seed{seed}{tag}.json").write_text(json.dumps(res, indent=2))

    indist = [r["indist_cov95"] for r in results]
    transfer = [r["transfer_cov95"] for r in results]
    mae = [r["joint_national_mae"] for r in results]
    summary = {
        "joint_indist_cov95": {"mean": float(np.mean(indist)), "std": float(np.std(indist)),
                               "per_seed": [round(x, 4) for x in indist]},
        "joint_transfer_cov95": {"mean": float(np.mean(transfer)), "std": float(np.std(transfer)),
                                 "per_seed": [round(x, 4) for x in transfer]},
        "joint_national_mae": {"mean": float(np.mean(mae)), "std": float(np.std(mae)),
                               "per_seed": [round(x, 4) for x in mae], "point_model_mae": VANILLA_POINT_MAE},
    }
    if not args.smoke:
        summary["decision"] = decide(summary["joint_indist_cov95"]["mean"], summary["joint_transfer_cov95"]["mean"])
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    else:
        (OUT_DIR / "summary_smoke.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ==="); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
