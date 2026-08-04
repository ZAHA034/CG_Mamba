"""§18 Phase 3 — Conformal WIS per-region (5 NN baselines, SARIMA separate).

Split conformal (Vovk 2005, Romano 2019): method-agnostic point-forecast UQ.
All 5 baselines get SAME conformal procedure → fair comparison.

Per region × baseline:
  1. Val forward → point predictions → signed residuals (y - pred)
  2. Per-horizon finite-sample corrected quantile offsets
  3. Test prediction + offsets → conformal quantiles → WIS + cov95

SARIMA conformal deferred to SARIMA WIS completion (needs val predictions).

Output: runs/phase_3_conformal_region.csv
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
_TSLIB = _ROOT.parent / "Time-Series-Library"
sys.path.insert(0, str(_TSLIB))
import models  # force-cache TSLib models
import layers  # force-cache TSLib layers (utils.masking dep)
import utils   # force-cache TSLib utils
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset
from cm_mamba.baselines.lstm_baseline import LSTMForecaster
from baselines.vanilla_mamba import VanillaMambaForecaster
from baselines.patchtst import PatchTSTForecaster
from baselines.dlinear import DLinearForecaster
from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import WeeklyDataset, load_norm_params
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TM = float(NORM["ili_weighted_pct"]["mean"])
TS = float(NORM["ili_weighted_pct"]["std"])
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240
SEEDS = [42, 123, 456, 789, 1024]
REGIONS = [f"hhs{i}" for i in range(1, 11)]


def build_region_df(region):
    from epiweeks import Week
    df_r = pd.read_csv(_ROOT / f"data/raw/cdc_ilinet/_phase3_phase6_fetch/{region}_full.csv")
    df_r["epiweek"] = df_r["year"].astype(int)*100 + df_r["week"].astype(int)
    df_r["date"] = df_r.apply(lambda r: Week(int(r["year"]),int(r["week"])).startdate().isoformat(), axis=1)
    env = pd.read_csv(_ROOT / "data/processed/env_national_weekly.csv")
    df_m = df_r.merge(env[["epiweek","temperature_c","specific_humidity_g_per_kg"]], on="epiweek", how="inner")
    split = pd.read_csv(_ROOT / "data/processed/ili_env_weekly_split.csv")
    df_m = df_m.merge(split[["epiweek","split"]], on="epiweek", how="inner")
    df_m["n_stations_available"] = 10; df_m["weight_sum_raw"] = 1.0
    return df_m


def conformal_quantiles(pred_test, residuals_val):
    """Split conformal: pred + quantile(residuals, q·(n+1)/n)."""
    N_val, H = residuals_val.shape
    out = {}
    for q in REQUIRED_QUANTILES:
        q_adj = min(1.0, q * (N_val + 1) / N_val) if q <= 0.5 \
                else max(0.0, 1 - (1 - q) * (N_val + 1) / N_val)
        offset_h = np.array([np.quantile(residuals_val[:, h], q_adj) for h in range(H)])
        out[q] = pred_test + offset_h[None, :]
    return out


def _forward_nn(model, ds, device):
    """Single deterministic forward → (preds_raw [N,4], ys_raw [N,4])."""
    model.eval().to(device)
    n = len(ds)
    preds = np.zeros((n, 4)); ys = np.zeros((n, 4))
    with torch.no_grad():
        for i in range(n):
            x, y = ds[i]
            preds[i] = model(x.unsqueeze(0).to(device))[0].cpu().numpy()
            ys[i] = y.numpy()
    return preds * TS + TM, ys * TS + TM


def _forward_cgm(model, ds, device, region_df):
    """CG-Mamba single forward → (preds_raw [N,4], ys_raw [N,4])."""
    model.eval().to(device)
    n = len(ds)
    eps = region_df["epiweek"].astype(int).to_numpy()
    preds = np.zeros((n, 4)); ys = np.zeros((n, 4))
    valid = np.ones(n, dtype=bool)
    with torch.no_grad():
        for i in range(n):
            d = ds[i]
            x = d["x"].unsqueeze(0).to(device)
            env = d["env"].unsqueeze(0).to(device)
            pred = model(x, env)
            if torch.isnan(pred).any():
                valid[i] = False; continue
            preds[i] = pred[0].cpu().numpy()
            tgt_ep = int(d["target_epiweek"])
            tgt_idx = int(np.where(eps == tgt_ep)[0][0])
            for h_idx, h in enumerate(HORIZONS):
                src = tgt_idx - (max(HORIZONS) - h)
                if 0 <= src < len(eps):
                    ys[i, h_idx] = (region_df.iloc[src]["ili_weighted_pct"] - TM) / TS
    preds_raw = preds[valid] * TS + TM
    ys_raw = ys[valid] * TS + TM
    return preds_raw, ys_raw, valid


def compute_conformal_wis(pred_val_raw, y_val_raw, pred_test_raw, y_test_raw, eps_h1_test):
    """Compute conformal WIS for tF + tS."""
    residuals = y_val_raw - pred_val_raw  # [N_val, H]
    qf = conformal_quantiles(pred_test_raw, residuals)
    ts_mask = eps_h1_test >= TS_BOUNDARY
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        qf_h = {q: qf[q][:, h_idx] for q in qf}
        out[f"tF_wis_h{h}"] = float(wis(y_test_raw[:, h_idx], qf_h).mean())
        out[f"tF_cov95_h{h}"] = float(coverage(y_test_raw[:, h_idx], qf_h, alpha=0.05))
        if ts_mask.sum() > 0:
            qf_ts = {q: qf[q][ts_mask, h_idx] for q in qf}
            out[f"tS_wis_h{h}"] = float(wis(y_test_raw[ts_mask, h_idx], qf_ts).mean())
            out[f"tS_cov95_h{h}"] = float(coverage(y_test_raw[ts_mask, h_idx], qf_ts, alpha=0.05))
    out["n_full"] = len(y_test_raw)
    out["n_strict"] = int(ts_mask.sum())
    return out


def eval_nn_conformal(base, region_df, seed, device):
    """NN baseline conformal WIS (single deterministic forward, NO dropout)."""
    configs = {
        "lstm": ("runs/lstm_final/h256_l2_lr5e-04_bs16", "lstm_best.pt"),
        "vanilla_mamba": ("runs/vanilla_mamba_final/d64_nl3_lr5e-04", "vanilla_mamba_best.pt"),
        "patchtst": ("runs/patchtst_final/pl16_dm128_lr5e-04", "patchtst_best.pt"),
    }
    cfg_dir, ckpt_name = configs[base]
    p = _ROOT / cfg_dir / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / ckpt_name, map_location=device, weights_only=True)

    if base == "lstm":
        model = LSTMForecaster(enc_in=cfg["enc_in"], hidden=cfg["hidden"],
                                num_layers=cfg["num_layers"], pred_len=cfg["pred_len"], dropout=0.0)
    elif base == "vanilla_mamba":
        model = VanillaMambaForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
                                        enc_in=cfg["enc_in"], d_model=cfg["d_model"],
                                        n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                                        dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=0.0)
    elif base == "patchtst":
        kwargs = dict(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                      d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
                      patch_len=cfg["patch_len"], dropout=0.0)
        if "d_ff_ratio" in cfg: kwargs["d_ff"] = int(cfg["d_model"] * cfg["d_ff_ratio"])
        if "stride_ratio" in cfg: kwargs["stride"] = int(cfg["patch_len"] * cfg["stride_ratio"])
        model = PatchTSTForecaster(**kwargs)
    model.load_state_dict(ckpt, strict=False)

    seq_len = cfg.get("lookback") or cfg.get("seq_len")
    val_ds = WeeklyMultiHorizonDataset(region_df, "val", NORM, lookback=seq_len, pred_len=cfg["pred_len"])
    test_ds = WeeklyMultiHorizonDataset(region_df, "test", NORM, lookback=seq_len, pred_len=cfg["pred_len"])

    pred_val, y_val = _forward_nn(model, val_ds, device)
    pred_test, y_test = _forward_nn(model, test_ds, device)

    eps = test_ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[test_ds.window_ends + 1]
    return compute_conformal_wis(pred_val, y_val, pred_test, y_test, eps_h1)


def eval_dlinear_conformal(region_df, device):
    """DLinear conformal: 5-seed mean as point prediction."""
    preds_val_seeds, preds_test_seeds = [], []
    y_val_raw, y_test_raw = None, None
    for seed in SEEDS:
        r = json.load(open(_ROOT / f"runs/dlinear_final/ma13_indF_lr2e-03/seed{seed}/results.json"))
        cfg = r["config"]
        ckpt = torch.load(_ROOT / f"runs/dlinear_final/ma13_indF_lr2e-03/seed{seed}/dlinear_best.pt",
                           map_location=device, weights_only=True)
        model = DLinearForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
                                    enc_in=cfg["enc_in"], moving_avg=cfg["moving_avg"],
                                    individual=cfg["individual"])
        model.load_state_dict(ckpt)
        val_ds = WeeklyMultiHorizonDataset(region_df, "val", NORM, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
        test_ds = WeeklyMultiHorizonDataset(region_df, "test", NORM, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
        pv, yv = _forward_nn(model, val_ds, device)
        pt, yt = _forward_nn(model, test_ds, device)
        preds_val_seeds.append(pv); preds_test_seeds.append(pt)
        if y_val_raw is None: y_val_raw, y_test_raw = yv, yt
    pred_val = np.mean(preds_val_seeds, axis=0)
    pred_test = np.mean(preds_test_seeds, axis=0)
    eps = test_ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[test_ds.window_ends + 1]
    return compute_conformal_wis(pred_val, y_val_raw, pred_test, y_test_raw, eps_h1)


def eval_cgm_conformal(region_df, seed, device):
    """CG-Mamba conformal: deterministic point prediction."""
    m = json.load(open(_ROOT / f"runs/m2_4_data_efficiency/cg_mamba/seasons_17_seasons_full/seed{seed}/manifest.json"))
    hmm = load_fitted_hmm(Path(m["hmm_dir"]))
    cfg = CGMambaConfig()
    model = CGForecaster(cfg)
    model.prepare_for_stage2(hmm)
    ck = torch.load(Path(m["stage3_best"]), map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck.get("state_dict", ck)), strict=False)

    val_ds = WeeklyDataset(region_df, split="val", lookback=cfg.lookback, horizon=max(cfg.horizons), norm=NORM)
    test_ds = WeeklyDataset(region_df, split="test", lookback=cfg.lookback, horizon=max(cfg.horizons), norm=NORM)

    pv, yv, valid_v = _forward_cgm(model, val_ds, device, region_df)
    pt, yt, valid_t = _forward_cgm(model, test_ds, device, region_df)

    eps = region_df["epiweek"].astype(int).to_numpy()
    # Reconstruct eps_h1 for test
    test_eps_h1 = []
    for i in range(len(test_ds)):
        if not valid_t[i]: continue
        d = test_ds[i]
        tgt_ep = int(d["target_epiweek"])
        tgt_idx = int(np.where(eps == tgt_ep)[0][0])
        test_eps_h1.append(eps[tgt_idx - (max(HORIZONS) - 1)])
    return compute_conformal_wis(pv, yv, pt, yt, np.array(test_eps_h1))


def main(device="cuda:0"):
    if not torch.cuda.is_available(): device = "cpu"
    rows = []
    print(f"Device: {device}", flush=True)
    for region in REGIONS:
        print(f"\n=== {region} ===", flush=True)
        region_df = build_region_df(region)

        # DLinear (single aggregate, no per-seed)
        try:
            r = eval_dlinear_conformal(region_df, device)
            r.update({"region": region, "baseline": "dlinear", "seed": -1})
            rows.append(r)
            print(f"  ✓ dlinear      tS_wis_h1={r.get('tS_wis_h1','?'):.4f}  cov={r.get('tS_cov95_h1','?'):.3f}", flush=True)
        except Exception as e:
            print(f"  ✗ dlinear: {e}", flush=True)

        # NN baselines × seeds
        for base in ["lstm", "vanilla_mamba", "patchtst"]:
            for seed in SEEDS:
                try:
                    r = eval_nn_conformal(base, region_df, seed, device)
                    r.update({"region": region, "baseline": base, "seed": seed})
                    rows.append(r)
                except Exception as e:
                    print(f"  ✗ {base} s={seed}: {e}", flush=True)
            # Print last seed summary
            last = [x for x in rows if x.get("region")==region and x.get("baseline")==base]
            if last:
                avg_wis = np.mean([x.get("tS_wis_h1", np.nan) for x in last[-5:]])
                print(f"  ✓ {base:<14} tS_wis_h1={avg_wis:.4f} (5-seed mean)", flush=True)

        # CG-Mamba × seeds
        for seed in SEEDS:
            try:
                r = eval_cgm_conformal(region_df, seed, device)
                r.update({"region": region, "baseline": "cg_mamba", "seed": seed})
                rows.append(r)
            except Exception as e:
                print(f"  ✗ cg_mamba s={seed}: {e}", flush=True)
        last_cgm = [x for x in rows if x.get("region")==region and x.get("baseline")=="cg_mamba"]
        if last_cgm:
            avg_wis = np.mean([x.get("tS_wis_h1", np.nan) for x in last_cgm[-5:]])
            print(f"  ✓ cg_mamba     tS_wis_h1={avg_wis:.4f} (5-seed mean)", flush=True)

    df = pd.DataFrame(rows)
    out = _ROOT / "runs" / "phase_3_conformal_region.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  rows={len(df)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    main(ap.parse_args().device)
