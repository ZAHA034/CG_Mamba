"""§18 Phase 3 — Region × NN baselines × WIS evaluation.

MC Dropout (n=100) for LSTM, Vanilla Mamba, PatchTST, CG-Mamba.
SARIMA WIS measured separately (parametric Gaussian, Kalman variance).

Output: runs/phase_3_region_wis.csv (200 rows: 4 baselines × 10 regions × 5 seeds)
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset
from cm_mamba.baselines.lstm_baseline import LSTMForecaster
from baselines.vanilla_mamba import VanillaMambaForecaster
from baselines.patchtst import PatchTSTForecaster
from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import WeeklyDataset, load_norm_params
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES
from src.eval.quantile_predictions import _dropout_train_mode

from scripts.phase_3_region_eval import build_region_df

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240
REGIONS = [f"hhs{i}" for i in range(1, 11)]
DROPOUT_MC = {"lstm": 0.3, "vanilla_mamba": 0.1, "cg_mamba": 0.1, "patchtst": 0.1}
N_MC_SAMPLES = 100
SEEDS = [42, 123, 456, 789, 1024]


def _ts_idx(eps_h1):
    return np.where(eps_h1 >= TS_BOUNDARY)[0]


def _wis_cov_per_h(samples, y_raw, ts_idx, out, prefix_tf, prefix_ts):
    """Compute WIS + cov95 per horizon. samples [S,N,H], y_raw [N,H]."""
    for h_idx, h in enumerate(HORIZONS):
        qf = {q: np.quantile(samples[:, :, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
        out[f"{prefix_tf}_wis_h{h}"] = float(wis(y_raw[:, h_idx], qf).mean())
        out[f"{prefix_tf}_cov95_h{h}"] = float(coverage(y_raw[:, h_idx], qf, alpha=0.05))
        if len(ts_idx) > 0:
            qf_ts = {q: np.quantile(samples[:, ts_idx, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
            out[f"{prefix_ts}_wis_h{h}"] = float(wis(y_raw[ts_idx, h_idx], qf_ts).mean())
            out[f"{prefix_ts}_cov95_h{h}"] = float(coverage(y_raw[ts_idx, h_idx], qf_ts, alpha=0.05))


def _mc_samples_nn(model, ds, device, dropout_rate, n_samples=N_MC_SAMPLES):
    """NN MC Dropout sampling. Returns samples_raw [S,N,H], y_raw [N,H], eps_h1 [N]."""
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    model.eval().to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout_rate
        elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            m.dropout = dropout_rate
    all_samples = []
    y_collect = None
    with _dropout_train_mode(model):
        with torch.no_grad():
            for _ in range(n_samples):
                preds, ys = [], []
                for x, y in loader:
                    x = x.to(device)
                    preds.append(model(x).cpu().numpy())
                    ys.append(y.numpy())
                preds = np.concatenate(preds, axis=0)
                ys = np.concatenate(ys, axis=0)
                all_samples.append(preds)
                if y_collect is None: y_collect = ys
    samples = np.stack(all_samples, axis=0) * TARGET_STD + TARGET_MEAN
    y_raw = y_collect * TARGET_STD + TARGET_MEAN
    eps = ds.df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps[ds.window_ends + 1]
    return samples, y_raw, eps_h1


def eval_nn_wis(base, region_df, seed, device):
    """LSTM / Vanilla Mamba / PatchTST WIS via MC Dropout."""
    base_dirs = {
        "lstm": ("runs/lstm_final/h256_l2_lr5e-04_bs16", "lstm_best.pt", LSTMForecaster),
        "vanilla_mamba": ("runs/vanilla_mamba_final/d64_nl3_lr5e-04", "vanilla_mamba_best.pt", VanillaMambaForecaster),
        "patchtst": ("runs/patchtst_final/pl16_dm128_lr5e-04", "patchtst_best.pt", PatchTSTForecaster),
    }
    cfg_dir, ckpt_name, ModelCls = base_dirs[base]
    p = _ROOT / cfg_dir / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / ckpt_name, map_location=device, weights_only=True)

    if base == "lstm":
        model = ModelCls(enc_in=cfg["enc_in"], hidden=cfg["hidden"], num_layers=cfg["num_layers"],
                          pred_len=cfg["pred_len"], dropout=DROPOUT_MC["lstm"])
    elif base == "vanilla_mamba":
        model = ModelCls(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                          d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                          dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=DROPOUT_MC["vanilla_mamba"])
    elif base == "patchtst":
        kwargs = dict(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                      d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
                      patch_len=cfg["patch_len"], dropout=DROPOUT_MC["patchtst"])
        if "d_ff_ratio" in cfg:
            kwargs["d_ff"] = int(cfg["d_model"] * cfg["d_ff_ratio"])
        if "stride_ratio" in cfg:
            kwargs["stride"] = int(cfg["patch_len"] * cfg["stride_ratio"])
        model = ModelCls(**kwargs)
    model.load_state_dict(ckpt)
    seq_len = cfg.get("lookback") or cfg.get("seq_len")
    ds = WeeklyMultiHorizonDataset(region_df, "test", NORM, lookback=seq_len, pred_len=cfg["pred_len"])
    samples, y_raw, eps_h1 = _mc_samples_nn(model, ds, device, DROPOUT_MC[base])
    ts_idx = _ts_idx(eps_h1)
    out = {"baseline": base, "seed": seed, "n_full": len(y_raw), "n_strict": len(ts_idx), "dropout": DROPOUT_MC[base]}
    _wis_cov_per_h(samples, y_raw, ts_idx, out, "tF", "tS")
    return out


def eval_cgm_wis(region_df, seed, device):
    """CG-Mamba WIS via MC Dropout."""
    m_path = _ROOT / "runs/m2_4_data_efficiency/cg_mamba/seasons_17_seasons_full" / f"seed{seed}" / "manifest.json"
    m = json.load(open(m_path))
    hmm = load_fitted_hmm(Path(m["hmm_dir"]))
    cfg = CGMambaConfig()
    model = CGForecaster(cfg)
    model.prepare_for_stage2(hmm)
    ck = torch.load(Path(m["stage3_best"]), map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = DROPOUT_MC["cg_mamba"]
    model.eval().to(device)

    ds = WeeklyDataset(region_df, split="test", lookback=cfg.lookback,
                        horizon=max(cfg.horizons), norm=NORM)
    n = len(ds)
    if n == 0:
        return {"baseline": "cg_mamba", "seed": seed, "error": "empty test set"}
    eps_arr = region_df["epiweek"].astype(int).to_numpy()
    # MC sampling
    all_preds_samples = np.zeros((N_MC_SAMPLES, n, 4))
    eps_h1 = np.zeros(n, dtype=np.int64)
    y_norm = np.zeros((n, 4))
    valid_mask = np.ones(n, dtype=bool)
    with _dropout_train_mode(model):
        with torch.no_grad():
            for s_idx in range(N_MC_SAMPLES):
                for i in range(n):
                    d = ds[i]
                    x = d["x"].unsqueeze(0).to(device)
                    env = d["env"].unsqueeze(0).to(device)
                    pred = model(x, env)
                    if torch.isnan(pred).any():
                        valid_mask[i] = False
                        continue
                    all_preds_samples[s_idx, i] = pred[0].cpu().numpy()
                    if s_idx == 0:
                        tgt_ep = int(d["target_epiweek"])
                        tgt_idx = int(np.where(eps_arr == tgt_ep)[0][0])
                        for h_idx, h in enumerate(HORIZONS):
                            off = max(HORIZONS) - h
                            src = tgt_idx - off
                            if 0 <= src < len(eps_arr):
                                y_norm[i, h_idx] = (region_df.iloc[src]["ili_weighted_pct"] - TARGET_MEAN) / TARGET_STD
                        eps_h1[i] = eps_arr[tgt_idx - (max(HORIZONS) - 1)]
    # Filter valid
    samples = all_preds_samples[:, valid_mask] * TARGET_STD + TARGET_MEAN
    y_raw = y_norm[valid_mask] * TARGET_STD + TARGET_MEAN
    eps_h1_v = eps_h1[valid_mask]
    ts_idx = _ts_idx(eps_h1_v)
    out = {"baseline": "cg_mamba", "seed": seed,
           "n_full": int(valid_mask.sum()), "n_strict": len(ts_idx),
           "n_nan": int((~valid_mask).sum()), "dropout": DROPOUT_MC["cg_mamba"]}
    _wis_cov_per_h(samples, y_raw, ts_idx, out, "tF", "tS")
    return out


def main(device="cuda:1"):
    if not torch.cuda.is_available(): device = "cpu"
    rows = []
    print(f"Device: {device}", flush=True)
    for region in REGIONS:
        print(f"\n=== {region} ===", flush=True)
        region_df = build_region_df(region)
        # CG-Mamba excluded — Method F is best UQ (phase_3_cgm_method_f_region.csv)
        for base in ["lstm", "vanilla_mamba", "patchtst"]:
            for seed in SEEDS:
                try:
                    if base == "cg_mamba":
                        r = eval_cgm_wis(region_df, seed, device)
                    else:
                        r = eval_nn_wis(base, region_df, seed, device)
                    r["region"] = region
                    rows.append(r)
                    tS_wis_h1 = r.get("tS_wis_h1", float("nan"))
                    tS_cov_h1 = r.get("tS_cov95_h1", float("nan"))
                    print(f"  ✓ {base:<14} s={seed}  tS_wis_h1={tS_wis_h1:.4f}  cov95={tS_cov_h1:.3f}", flush=True)
                except Exception as e:
                    import traceback
                    print(f"  ✗ {base} s={seed}: {type(e).__name__}: {e}", flush=True)
                    rows.append({"region": region, "baseline": base, "seed": seed, "error": str(e)})

    df_out = pd.DataFrame(rows)
    out_csv = _ROOT / "runs" / "phase_3_region_wis.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  rows={len(df_out)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    main(args.device)
