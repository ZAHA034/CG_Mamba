"""M2.4 unified test_strict evaluation across all baselines + horizons.

Re-runs inference on each baseline's saved ckpt for:
  - test (full, 257 windows)
  - test_strict (152 windows = target_epiweek >= 202240)
across 4 horizons {1,2,3,4} × 5 seeds × 7 variants.

Output: runs/m2_4_data_efficiency/m2_4_test_strict_all_baselines.csv

NOTE: SARIMA results already have per-horizon test_strict in its own json files
(loaded directly, not re-inferred). CG-Mamba uses v2.3.1 patched phase_module.
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
from baselines.dlinear import DLinearForecaster
from baselines.patchtst import PatchTSTForecaster
from baselines.epideep import EpiDeepForecaster
from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import WeeklyDataset, load_norm_params

ROOT_RUNS = _ROOT / "runs" / "m2_4_data_efficiency"
VARIANTS = ["3_seasons", "4_seasons", "5_seasons", "7_seasons", "10_seasons", "13_seasons", "17_seasons_full"]
SEEDS = [42, 123, 456, 789, 1024]
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240  # test_strict: target_epiweek >= 202240
NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])


def _split_indices(ds: WeeklyMultiHorizonDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (all_test_idx, ts_idx, target_eps_h1) for the dataset.

    test_strict = target_epiweek_h1 >= 202240 (PLAN §4.1 v2.1.7-A++).
    """
    eps = ds.df["epiweek"].astype(int).to_numpy()
    target_eps_h1 = eps[ds.window_ends + 1]  # epiweek at h=1
    all_idx = np.arange(len(ds))
    ts_mask = target_eps_h1 >= TS_BOUNDARY
    return all_idx, all_idx[ts_mask], target_eps_h1


def _eval_loop(model, ds, device, denorm: bool = True) -> dict:
    """Inference all windows → per-horizon MAE for test_full / test_strict.

    Returns dict: {test_full_mae_h1..4, test_strict_mae_h1..4, n_full, n_strict}.
    """
    model.eval()
    model.to(device)
    all_preds = np.zeros((len(ds), 4), dtype=np.float64)
    all_ys = np.zeros((len(ds), 4), dtype=np.float64)
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            all_preds[i] = pred[0].cpu().numpy()
            all_ys[i] = y.numpy()
    if denorm:
        all_preds = all_preds * TARGET_STD + TARGET_MEAN
        all_ys = all_ys * TARGET_STD + TARGET_MEAN
    _, ts_idx, _ = _split_indices(ds)
    abs_err = np.abs(all_preds - all_ys)
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        out[f"test_full_mae_h{h}"] = float(abs_err[:, h_idx].mean())
        if len(ts_idx) > 0:
            out[f"test_strict_mae_h{h}"] = float(abs_err[ts_idx, h_idx].mean())
        else:
            out[f"test_strict_mae_h{h}"] = float("nan")
    out["n_full"] = int(len(ds))
    out["n_strict"] = int(len(ts_idx))
    return out


def _cg_eval_loop(model, ds_cg, device, denorm: bool = True) -> dict:
    """CG-Mamba inference (separate dataloader with x_main + env)."""
    model.eval()
    model.to(device)
    n = len(ds_cg)
    target_eps = np.zeros(n, dtype=np.int64)
    # CG WeeklyDataset returns single h=max, not multi-horizon. Need patched dataset.
    # Use WeeklyMultiHorizonDataset for ys (h=1..4) + manual x_main/env split.
    # Workaround: call CG forecaster which outputs [B, len(horizons)=4]
    # but the loader.WeeklyDataset returns scalar y at h=max.
    # → For multi-horizon MAE on CG-Mamba, build target arrays from df directly.
    df = ds_cg.df.reset_index(drop=True)
    eps = df["epiweek"].astype(int).to_numpy()
    target_z = (df["ili_weighted_pct"].to_numpy() - TARGET_MEAN) / TARGET_STD
    all_preds = np.zeros((n, 4), dtype=np.float64)
    all_ys = np.zeros((n, 4), dtype=np.float64)
    valid_mask = np.ones(n, dtype=bool)
    end_idxs = []
    with torch.no_grad():
        for i in range(n):
            d = ds_cg[i]
            x = d["x"].unsqueeze(0).to(device)
            env = d["env"].unsqueeze(0).to(device)
            pred = model(x, env)
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                valid_mask[i] = False
                continue
            all_preds[i] = pred[0].cpu().numpy()
            # Reconstruct h=1..4 target via target_epiweek from loader
            tgt_ep = int(d["target_epiweek"])
            # Find idx of target_ep in df, then h=1..4 are tgt_ep - (max_h - h)
            tgt_idx = int(np.where(eps == tgt_ep)[0][0])
            # The loader's target_epiweek = epiweek at h=max=4.
            # So h=1 target = tgt_ep - 3 (3 weeks before), etc.
            for h_idx, h in enumerate(HORIZONS):
                offset = max(HORIZONS) - h  # h=1 → offset=3, h=4 → offset=0
                src = tgt_idx - offset
                if src >= 0 and src < len(eps):
                    all_ys[i, h_idx] = target_z[src]
                else:
                    valid_mask[i] = False
                    break
            end_idxs.append(tgt_idx)
    if denorm:
        all_preds = all_preds * TARGET_STD + TARGET_MEAN
        all_ys = all_ys * TARGET_STD + TARGET_MEAN
    # test_strict subset: target_epiweek_h1 >= 202240
    # target_epiweek_h1 = tgt_ep - (max_h - 1) = tgt_ep - 3
    # Easier: use array of h=1 target epweek
    target_eps_h1 = np.array([
        eps[end_idxs[i_local] - (max(HORIZONS) - 1)] if valid_mask[i] else 0
        for i_local, i in enumerate([j for j in range(n) if valid_mask[j]])
    ])
    # Re-compute on valid:
    valid = np.where(valid_mask)[0]
    if len(valid) != len(end_idxs):
        # Mismatch — re-iterate cleanly
        v_preds, v_ys, v_eps = [], [], []
        for vi, i in enumerate(range(n)):
            if not valid_mask[i]: continue
            v_preds.append(all_preds[i])
            v_ys.append(all_ys[i])
            v_eps.append(eps[end_idxs[len(v_eps)] - (max(HORIZONS) - 1)])
        all_preds, all_ys = np.array(v_preds), np.array(v_ys)
        target_eps_h1 = np.array(v_eps)
    else:
        all_preds = all_preds[valid]
        all_ys = all_ys[valid]
    abs_err = np.abs(all_preds - all_ys)
    ts_mask = target_eps_h1 >= TS_BOUNDARY
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        out[f"test_full_mae_h{h}"] = float(abs_err[:, h_idx].mean()) if len(abs_err) > 0 else float("nan")
        if ts_mask.sum() > 0:
            out[f"test_strict_mae_h{h}"] = float(abs_err[ts_mask, h_idx].mean())
        else:
            out[f"test_strict_mae_h{h}"] = float("nan")
    out["n_full"] = int(len(abs_err))
    out["n_strict"] = int(ts_mask.sum())
    out["n_nan"] = int((~valid_mask).sum())
    return out


def evaluate_lstm(variant: str, seed: int, device: str) -> dict:
    p = ROOT_RUNS / "lstm" / f"seasons_{variant}" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "lstm_best.pt", map_location=device, weights_only=True)
    model = LSTMForecaster(
        enc_in=cfg["enc_in"], hidden=cfg["hidden"], num_layers=cfg["num_layers"],
        pred_len=cfg["pred_len"], dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt)
    csv = ROOT_RUNS / "_filtered_csvs" / f"ili_env_weekly_split_{variant}.csv"
    df = pd.read_csv(csv)
    ds = WeeklyMultiHorizonDataset(df, "test", NORM, lookback=cfg["lookback"], pred_len=cfg["pred_len"])
    return _eval_loop(model, ds, device)


def evaluate_vanilla(variant: str, seed: int, device: str) -> dict:
    p = ROOT_RUNS / "vanilla_mamba" / f"seasons_{variant}" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "vanilla_mamba_best.pt", map_location=device, weights_only=True)
    model = VanillaMambaForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg["d_state"],
        dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt)
    csv = ROOT_RUNS / "_filtered_csvs" / f"ili_env_weekly_split_{variant}.csv"
    df = pd.read_csv(csv)
    ds = WeeklyMultiHorizonDataset(df, "test", NORM, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    return _eval_loop(model, ds, device)


def evaluate_patchtst(variant: str, seed: int, device: str) -> dict:
    """PatchTST per-h MAE on test_full + test_strict.

    Loads M2.4 PatchTST ckpt (M2.3 winner config pl16_dm64_lr5e-04 trained per
    variant by scripts/m2_4_nn_baselines.py with --baselines patchtst).
    Point forecast only — MC Dropout uncertainty handled separately.
    """
    p = ROOT_RUNS / "patchtst" / f"seasons_{variant}" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "patchtst_best.pt", map_location=device, weights_only=True)
    stride = max(1, int(cfg["patch_len"] * cfg["stride_ratio"]))
    model = PatchTSTForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
        d_ff=cfg["d_ff_ratio"] * cfg["d_model"],
        patch_len=cfg["patch_len"], stride=stride, dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt)
    csv = ROOT_RUNS / "_filtered_csvs" / f"ili_env_weekly_split_{variant}.csv"
    df = pd.read_csv(csv)
    ds = WeeklyMultiHorizonDataset(df, "test", NORM,
                                    lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    return _eval_loop(model, ds, device)


def evaluate_dlinear(variant: str, seed: int, device: str) -> dict:
    p = ROOT_RUNS / "dlinear" / f"seasons_{variant}" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "dlinear_best.pt", map_location=device, weights_only=True)
    model = DLinearForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        moving_avg=cfg["moving_avg"], individual=cfg["individual"],
    )
    model.load_state_dict(ckpt)
    csv = ROOT_RUNS / "_filtered_csvs" / f"ili_env_weekly_split_{variant}.csv"
    df = pd.read_csv(csv)
    ds = WeeklyMultiHorizonDataset(df, "test", NORM, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    return _eval_loop(model, ds, device)


def evaluate_epideep(variant: str, seed: int, device: str) -> dict:
    """EpiDeep per-h MAE on test_full + test_strict.
    Loads M2.4 EpiDeep ckpt (M2.3 winner config de128_eh64_lr2e-03 trained per
    variant by scripts/m2_4_nn_baselines.py with --baselines epideep)."""
    p = ROOT_RUNS / "epideep" / f"seasons_{variant}" / f"seed{seed}"
    r = json.load(open(p / "results.json"))
    cfg = r["config"]
    ckpt = torch.load(p / "epideep_best.pt", map_location=device, weights_only=True)
    model = EpiDeepForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
        decoder_hidden=cfg["decoder_hidden"],
        alignment_weight=cfg.get("alignment_weight", 0.0),
        dropout=cfg.get("dropout", 0.1),
        target_only=cfg.get("target_only", False),
    )
    model.load_state_dict(ckpt)
    csv = ROOT_RUNS / "_filtered_csvs" / f"ili_env_weekly_split_{variant}.csv"
    df = pd.read_csv(csv)
    ds = WeeklyMultiHorizonDataset(df, "test", NORM, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
    return _eval_loop(model, ds, device)


def evaluate_cgm(variant: str, seed: int, device: str) -> dict:
    m_path = ROOT_RUNS / "cg_mamba" / f"seasons_{variant}" / f"seed{seed}" / "manifest.json"
    m = json.load(open(m_path))
    hmm = load_fitted_hmm(Path(m["hmm_dir"]))
    cfg = CGMambaConfig()
    model = CGForecaster(cfg)
    model.prepare_for_stage2(hmm)
    ck = torch.load(Path(m["stage3_best"]), map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    csv_path = m["csv_used"]
    df = pd.read_csv(csv_path)
    # Use WeeklyDataset for CG-Mamba (returns x, env)
    ds = WeeklyDataset(df, split="test", lookback=cfg.lookback, horizon=max(cfg.horizons), norm=NORM)
    return _cg_eval_loop(model, ds, device)


def evaluate_sarima(variant: str) -> dict:
    """SARIMA results already saved per-horizon — just extract."""
    p = ROOT_RUNS / "sarima" / f"seasons_{variant}.json"
    r = json.load(open(p))
    if r.get("status") != "OK":
        return {f"test_full_mae_h{h}": float("nan") for h in HORIZONS}
    test = r["results"]["test"]
    ts = r["results"]["test_strict"]
    out = {}
    for h in HORIZONS:
        out[f"test_full_mae_h{h}"] = float(test[str(h)]["mae"])
        out[f"test_strict_mae_h{h}"] = float(ts[str(h)]["mae"])
    out["n_full"] = int(test["1"]["n"])
    out["n_strict"] = int(ts["1"]["n"])
    return out


def main(device: str = "cuda:0", baselines: tuple[str, ...] = (
    "sarima", "dlinear", "lstm", "vanilla_mamba", "patchtst", "epideep", "cg_mamba")):
    """Eval selected baselines. By default writes full CSV (overwrites).
    If only PatchTST is selected (single-baseline mode), APPENDS to existing
    CSV instead of overwriting — preserves the 147 rows from prior runs.
    """
    if not torch.cuda.is_available():
        device = "cpu"
    rows = []
    print(f"Device: {device}, baselines: {baselines}")

    nn_baselines = [
        ("dlinear", evaluate_dlinear),
        ("lstm", evaluate_lstm),
        ("vanilla_mamba", evaluate_vanilla),
        ("patchtst", evaluate_patchtst),
        ("epideep", evaluate_epideep),
        ("cg_mamba", evaluate_cgm),
    ]

    for v in VARIANTS:
        if "sarima" in baselines:
            try:
                r = evaluate_sarima(v)
                r.update({"baseline": "sarima", "variant": v, "seed": -1})
                rows.append(r)
                print(f"  ✓ sarima {v:<18}  tS_h1={r.get('test_strict_mae_h1', 'nan'):.4f}")
            except Exception as e:
                print(f"  ✗ sarima {v}: {e}")

        for seed in SEEDS:
            for base, fn in nn_baselines:
                if base not in baselines: continue
                try:
                    r = fn(v, seed, device)
                    r.update({"baseline": base, "variant": v, "seed": seed})
                    rows.append(r)
                    ts_h1 = r.get('test_strict_mae_h1', float('nan'))
                    print(f"  ✓ {base:<14} {v:<18} s={seed}  tS_h1={ts_h1:.4f}  n_strict={r.get('n_strict','?')}" +
                          (f"  nan={r['n_nan']}" if 'n_nan' in r else ""))
                except Exception as e:
                    print(f"  ✗ {base} {v} s={seed}: {type(e).__name__}: {e}")
                    rows.append({"baseline": base, "variant": v, "seed": seed, "error": str(e)})

    new_df = pd.DataFrame(rows)
    out_csv = ROOT_RUNS / "m2_4_test_strict_all_baselines.csv"

    # Append mode: if subset of baselines selected AND CSV exists, merge
    if set(baselines) < {"sarima", "dlinear", "lstm", "vanilla_mamba", "patchtst", "cg_mamba"} \
       and out_csv.exists():
        existing = pd.read_csv(out_csv)
        # Drop any pre-existing rows for the same baselines (to avoid dup)
        existing = existing[~existing["baseline"].isin(baselines)]
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged.to_csv(out_csv, index=False)
        print(f"\nAppended → {out_csv}  rows={len(merged)} (existing {len(existing)} + new {len(new_df)})")
    else:
        new_df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}  rows={len(new_df)}")


if __name__ == "__main__":
    import argparse
    ALL_BASELINES = ("sarima", "dlinear", "lstm", "vanilla_mamba", "patchtst", "cg_mamba")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baselines", nargs="+", default=list(ALL_BASELINES),
                    choices=ALL_BASELINES)
    args = ap.parse_args()
    main(args.device, baselines=tuple(args.baselines))
