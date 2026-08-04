"""Student-t df=4 sensitivity for ensemble baselines (DLinear, N-BEATS).

Re-applies their 5-seed ensemble predictions with Student-t instead of Gaussian
quantiles (PLAN J.6 sensitivity). Heavy-tail alternative.

Output: runs/wis_student_t/wis_results.json
"""
from __future__ import annotations

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

from baselines.lstm import WeeklyMultiHorizonDataset
from baselines.dlinear import DLinearForecaster
from baselines.nbeats import NBeatsForecaster

from src.data.loader import load_dataset_csv, load_norm_params
from src.eval.wis import wis, wis_decomposed, coverage
from src.eval.quantile_predictions import ensemble_student_t_quantiles

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_DIR = _ROOT / "runs" / "wis_student_t"
COVID_STRICT_START_EPIWEEK = 202240
SEEDS = (42, 123, 456, 789, 1024)


@torch.no_grad()
def get_member_preds_raw(model, loader, target_mean, target_std, device):
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        preds.append(model(x).cpu().numpy())
        ys.append(y.cpu().numpy())
    return (np.concatenate(preds) * target_std + target_mean,
            np.concatenate(ys) * target_std + target_mean)


def _mask(df, sp, ep_min):
    if ep_min is None: return df
    s = df.copy()
    s.loc[(s["split"] == sp) & (s["epiweek"] < ep_min), "split"] = "_excluded"
    return s


def _build_loader(df, split, lookback, pred_len, norm, ep_min):
    s = _mask(df, split, ep_min)
    ds = WeeklyMultiHorizonDataset(s, split, norm, lookback=lookback, pred_len=pred_len)
    return DataLoader(ds, batch_size=32, shuffle=False)


def score_split(qf, y):
    N, H = y.shape
    wis_h, disp, under, over = [], [], [], []
    for h in range(H):
        qh = {q: qf[q][:, h] for q in qf}
        yh = y[:, h]
        wis_h.append(float(wis(yh, qh).mean()))
        p = wis_decomposed(yh, qh)
        disp.append(float(p["dispersion"].mean()))
        under.append(float(p["under"].mean()))
        over.append(float(p["over"].mean()))
    qf_flat = {q: qf[q].reshape(-1) for q in qf}
    y_flat = y.reshape(-1)
    return {
        "n": int(N), "wis_per_horizon": wis_h, "wis_avg": float(np.mean(wis_h)),
        "dispersion_avg": float(np.mean(disp)),
        "under_avg": float(np.mean(under)),
        "over_avg": float(np.mean(over)),
        "coverage_50": coverage(y_flat, qf_flat, alpha=0.5),
        "coverage_95": coverage(y_flat, qf_flat, alpha=0.05),
    }


def run_baseline(b, cfg_name, ckpt_file, build_fn, df, norm, device):
    cfg = json.load(open(_ROOT / "runs" / f"{b}_final" / cfg_name / "seed42" / "results.json"))["config"]
    lookback = cfg.get("seq_len", cfg.get("lookback", 104))
    pred_len = cfg["pred_len"]
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    splits_loaders = {}
    for sp_lbl, ep_min in [("val", None), ("test_full", None),
                            ("test_strict", COVID_STRICT_START_EPIWEEK)]:
        sp_name = "val" if sp_lbl == "val" else "test"
        splits_loaders[sp_lbl] = _build_loader(df, sp_name, lookback, pred_len, norm, ep_min)

    member_preds = {sp: [] for sp in splits_loaders}
    y_raws = {}
    for seed in SEEDS:
        ckpt = _ROOT / "runs" / f"{b}_final" / cfg_name / f"seed{seed}" / ckpt_file
        if not ckpt.exists(): continue
        model = build_fn(cfg).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=True)
        for sp, loader in splits_loaders.items():
            p, y = get_member_preds_raw(model, loader, target_mean, target_std, device)
            member_preds[sp].append(p)
            y_raws[sp] = y

    results = {"baseline": b, "uq": "ensemble_student_t_df4", "splits": {}}
    for sp in splits_loaders:
        members = np.stack(member_preds[sp], axis=0)
        qf_t = ensemble_student_t_quantiles(members, df=4)
        score = score_split(qf_t, y_raws[sp])
        results["splits"][sp] = score
        print(f"  [{b} {sp:11s}] Student-t df=4 WIS={score['wis_avg']:.4f}  "
              f"cov95={score['coverage_95']:.3f}")
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)

    for b, cfg, ckpt, fn in [
        ("dlinear", "ma13_indF_lr2e-03", "dlinear_best.pt",
         lambda c: DLinearForecaster(seq_len=c["seq_len"], pred_len=c["pred_len"],
                                     enc_in=c["enc_in"], moving_avg=c["moving_avg"],
                                     individual=c["individual"])),
        ("nbeats", "nb24_h512_lr5e-04", "nbeats_best.pt",
         lambda c: NBeatsForecaster(seq_len=c["seq_len"], pred_len=c["pred_len"],
                                     enc_in=c["enc_in"], hidden=c["hidden"],
                                     n_blocks=c["n_blocks"], n_layers=c["n_layers"],
                                     target_only=c.get("target_only", False))),
    ]:
        print(f"\n=== {b.upper()} — Student-t df=4 ===")
        r = run_baseline(b, cfg, ckpt, fn, df, norm, device)
        (OUT_DIR / f"{b}.json").write_text(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
