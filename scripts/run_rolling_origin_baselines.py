"""ROLLING-ORIGIN driver -- BASELINE side (pre-registered, result-blind, leakage-safe).

Retrains the 5 DL baselines per cutoff with the HEADLINE configs (the fairness-critical
ones the headline regional dominance was computed against) and re-scores their NATIVE
regional UQ per cutoff:
  - LSTM / Vanilla Mamba / PatchTST / EpiDeep : MC-Dropout (100 samples)
  - DLinear                                   : 5-seed ensemble Gaussian (seeded, deterministic)

CONFIG TRAP (locked): the headline regional baselines use the base_dirs configs below --
  patchtst = pl16_dm128 (d_model=128), NOT m2_4's dm64. Rolling retrains with these EXACT
  configs so the comparison is like-for-like with the headline.

Leakage-safe: each baseline is re-fit on the cutoff's train with the cutoff normalization
(train_one_run takes explicit csv_path + norm_path). Eval uses cutoff norm + cutoff split +
cutoff test-window (test_first).

Bug-guard: --regress re-scores the EXISTING headline checkpoints on the canonical split and
must reproduce the stored headline baseline Cov95 (phase_3_region_wis.csv + extras +
dlinear_ensemble_region.csv). NOTE: MC-Dropout is stochastic, so MC baselines reproduce only
within MC error (~<=0.03); DLinear (seeded Gaussian) reproduces near-exactly.

USAGE:
  python scripts/run_rolling_origin_baselines.py --regress                 # eval regression (no train)
  python scripts/run_rolling_origin_baselines.py --cutoffs 2022 --baselines lstm   # shakedown
  python scripts/run_rolling_origin_baselines.py                           # full baseline side
Resumable: existing retrained checkpoints are reused.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
# non-invasive shim: satisfy TSLib's transitive `import reformer_pytorch` (PatchTST uses only
# FullAttention, never LSHSelfAttention) without touching the shared conda env. See _rolling_shims/.
sys.path.insert(0, str(_ROOT / "scripts" / "_rolling_shims"))

from baselines.lstm import WeeklyMultiHorizonDataset
from cm_mamba.baselines.lstm_baseline import LSTMForecaster
from baselines.vanilla_mamba import VanillaMambaForecaster
from baselines.patchtst import PatchTSTForecaster
from src.baselines.epideep import EpiDeepForecaster
from src.baselines.dlinear import DLinearForecaster
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES
from src.eval.quantile_predictions import _dropout_train_mode
from src.data.loader import load_norm_params

from scripts.run_lstm_weekly import train_one_run as lstm_train
from scripts.run_vanilla_mamba_weekly import train_one_run as vanilla_train
from scripts.run_patchtst_weekly import train_one_run as patchtst_train
from scripts.run_epideep_weekly import train_one_run as epideep_train
from scripts.run_dlinear_weekly import train_one_run as dlinear_train

ROLL_ROOT = _ROOT / "runs" / "rolling_origin"
BASE_OUT = ROLL_ROOT / "baselines"
CUTOFFS = [2015, 2016, 2017, 2018, 2022, 2023, 2024]
SEEDS = [42, 123, 456, 789, 1024]
REGIONS = [f"hhs{i}" for i in range(1, 11)]
HORIZONS = [1, 2, 3, 4]
N_MC = 100
DLINEAR_GAUSS_SEED = 20260529
DROPOUT_MC = {"lstm": 0.3, "vanilla_mamba": 0.1, "patchtst": 0.1, "epideep": 0.1}
MC_BASES = ["lstm", "vanilla_mamba", "patchtst", "epideep"]

# HEADLINE configs (fairness-critical; base_dirs from phase_3_region_wis.py + extras)
HEADLINE = {
    "lstm":          ("runs/lstm_final/h256_l2_lr5e-04_bs16", "lstm_best.pt", lstm_train),
    "vanilla_mamba": ("runs/vanilla_mamba_final/d64_nl3_lr5e-04", "vanilla_mamba_best.pt", vanilla_train),
    "patchtst":      ("runs/patchtst_final/pl16_dm128_lr5e-04", "patchtst_best.pt", patchtst_train),
    "epideep":       ("runs/epideep_final/de128_eh64_lr2e-03", "epideep_best.pt", epideep_train),
    "dlinear":       ("runs/dlinear_final/ma13_indF_lr2e-03", "dlinear_best.pt", dlinear_train),
}
# stored headline baseline regional Cov95 (this session, for --regress aggregate check)
HEADLINE_COV = {"lstm": 0.513, "vanilla_mamba": 0.571, "patchtst": 0.695,
                "dlinear_ensemble_gauss": 0.286, "epideep": 0.382}
CANON_SPLIT = _ROOT / "data/processed/ili_env_weekly_split.csv"
CANON_NORM = _ROOT / "data/processed/normalization_params.json"
CANON_TEST_FIRST = 202240


def build_region_df(region, split_csv):
    """Replicate phase_3_region_eval.build_region_df with a PARAMETERIZED split source."""
    from epiweeks import Week
    df_r = pd.read_csv(_ROOT / "data/raw/cdc_ilinet/_phase3_phase6_fetch" / f"{region}_full.csv")
    df_r["epiweek"] = df_r["year"].astype(int) * 100 + df_r["week"].astype(int)
    df_r["date"] = df_r.apply(lambda r: Week(int(r["year"]), int(r["week"])).startdate().isoformat(), axis=1)
    env = pd.read_csv(_ROOT / "data/processed/env_national_weekly.csv")
    df = df_r.merge(env[["epiweek", "temperature_c", "specific_humidity_g_per_kg"]], on="epiweek", how="inner")
    split = pd.read_csv(split_csv)
    df = df.merge(split[["epiweek", "split"]], on="epiweek", how="inner")
    df["n_stations_available"] = 10
    df["weight_sum_raw"] = 1.0
    return df


def make_model(base, cfg, dropout):
    if base == "lstm":
        return LSTMForecaster(enc_in=cfg["enc_in"], hidden=cfg["hidden"], num_layers=cfg["num_layers"],
                              pred_len=cfg["pred_len"], dropout=dropout)
    if base == "vanilla_mamba":
        return VanillaMambaForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                                      d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                                      dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=dropout)
    if base == "patchtst":
        kw = dict(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                  d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
                  patch_len=cfg["patch_len"], dropout=dropout)
        if "d_ff_ratio" in cfg:
            kw["d_ff"] = int(cfg["d_model"] * cfg["d_ff_ratio"])
        if "stride_ratio" in cfg:
            kw["stride"] = int(cfg["patch_len"] * cfg["stride_ratio"])
        return PatchTSTForecaster(**kw)
    if base == "epideep":
        return EpiDeepForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                                 d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
                                 decoder_hidden=cfg["decoder_hidden"],
                                 alignment_weight=cfg.get("alignment_weight", 0.0),
                                 dropout=dropout, target_only=cfg.get("target_only", False))
    raise ValueError(base)


def _load_cfg_ckpt(cell_dir, ckpt_name, device):
    cfg = json.load(open(cell_dir / "results.json"))["config"]
    ckpt = torch.load(cell_dir / ckpt_name, map_location=device, weights_only=True)
    return cfg, ckpt


def _per_h_cov(samples, y_raw, ts_idx):
    """samples [S,N,H] raw, y_raw [N,H] raw -> per-horizon tS cov95/wis on ts_idx."""
    rec = {}
    for h_idx, h in enumerate(HORIZONS):
        qf = {q: np.quantile(samples[:, ts_idx, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
        rec[f"tS_cov95_h{h}"] = float(coverage(y_raw[ts_idx, h_idx], qf, alpha=0.05))
        rec[f"tS_wis_h{h}"] = float(wis(y_raw[ts_idx, h_idx], qf).mean())
    return rec


def eval_mc_baseline(base, cell_dir_fn, region_df, seed, norm, test_first, device):
    """MC-Dropout native UQ (lstm/vanilla/patchtst/epideep) for one (region, seed)."""
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    _, ckpt_name, _ = HEADLINE[base]
    cfg, ckpt = _load_cfg_ckpt(cell_dir_fn(seed), ckpt_name, device)
    model = make_model(base, cfg, DROPOUT_MC[base])
    model.load_state_dict(ckpt)
    seq_len = cfg.get("lookback") or cfg.get("seq_len")
    ds = WeeklyMultiHorizonDataset(region_df, "test", norm, lookback=seq_len, pred_len=cfg["pred_len"])
    model.eval().to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = DROPOUT_MC[base]
        elif isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            m.dropout = DROPOUT_MC[base]
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    torch.manual_seed(seed)                                  # determinism for MC masks
    all_s, y_collect = [], None
    with _dropout_train_mode(model):
        with torch.no_grad():
            for _ in range(N_MC):
                preds, ys = [], []
                for x, y in loader:
                    preds.append(model(x.to(device)).cpu().numpy()); ys.append(y.numpy())
                all_s.append(np.concatenate(preds, axis=0))
                if y_collect is None: y_collect = np.concatenate(ys, axis=0)
    samples = np.stack(all_s, axis=0) * tstd + tmean
    y_raw = y_collect * tstd + tmean
    eps_h1 = ds.df["epiweek"].astype(int).to_numpy()[ds.window_ends + 1]
    ts_idx = np.where(eps_h1 >= test_first)[0]
    return {"baseline": base, "seed": seed, "region": region_df.attrs.get("region", "?"),
            "n_strict": int(len(ts_idx)), **_per_h_cov(samples, y_raw, ts_idx)}


def eval_dlinear_ensemble(cell_dir_fn, region_df, norm, test_first, device):
    """DLinear 5-seed ensemble Gaussian (deterministic, seeded) for one region."""
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    per_seed, ys_ref, eps_ref = [], None, None
    for seed in SEEDS:
        cfg, ckpt = _load_cfg_ckpt(cell_dir_fn(seed), "dlinear_best.pt", device)
        model = DLinearForecaster(seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                                  moving_avg=cfg["moving_avg"], individual=cfg["individual"])
        model.load_state_dict(ckpt); model.eval().to(device)
        ds = WeeklyMultiHorizonDataset(region_df, "test", norm, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
        n = len(ds); preds = np.zeros((n, 4)); ys = np.zeros((n, 4))
        with torch.no_grad():
            for i in range(n):
                x, y = ds[i]
                preds[i] = model(x.unsqueeze(0).to(device))[0].cpu().numpy(); ys[i] = y.numpy()
        per_seed.append(preds)
        if ys_ref is None:
            ys_ref = ys
            eps_ref = ds.df["epiweek"].astype(int).to_numpy()[ds.window_ends + 1]
    ps = np.stack(per_seed, axis=0)                          # [5,N,H]
    mu = ps.mean(axis=0) * tstd + tmean
    sigma = ps.std(axis=0, ddof=1) * tstd
    y_raw = ys_ref * tstd + tmean
    rng = np.random.default_rng(seed=DLINEAR_GAUSS_SEED)
    samples = mu[None] + rng.standard_normal((N_MC, *mu.shape)) * sigma[None]
    ts_idx = np.where(eps_ref >= test_first)[0]
    return {"baseline": "dlinear_ensemble_gauss", "seed": -1, "region": region_df.attrs.get("region", "?"),
            "n_strict": int(len(ts_idx)), **_per_h_cov(samples, y_raw, ts_idx)}


def eval_cutoff(bases, split_csv, norm_json, test_first, cell_dir_fns, device, tag):
    """Evaluate given baselines over all regions. cell_dir_fns[base] -> (lambda seed: dir)."""
    norm = load_norm_params(Path(norm_json))
    rows = []
    for region in REGIONS:
        rdf = build_region_df(region, split_csv)
        rdf.attrs["region"] = region
        for base in bases:
            if base == "dlinear":
                rows.append(eval_dlinear_ensemble(cell_dir_fns["dlinear"], rdf, norm, test_first, device))
            else:
                for seed in SEEDS:
                    rows.append(eval_mc_baseline(base, cell_dir_fns[base], rdf, seed, norm, test_first, device))
        print(f"  [{tag}] {region} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- retrain
def retrain_cutoff(base, Y, cut_csv, cut_norm, device):
    hd_dir, ckpt_name, train_fn = HEADLINE[base]
    hp = json.load(open(_ROOT / hd_dir / "seed42/results.json"))["config"]   # headline HP
    for seed in SEEDS:
        out_dir = BASE_OUT / base / f"cut{Y}" / f"seed{seed}"
        if (out_dir / ckpt_name).exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        train_fn(cfg=hp.copy(), seed=seed, csv_path=cut_csv, norm_path=cut_norm,
                 device=device, out_dir=out_dir, wandb_enabled=False,
                 wandb_run_name=f"roll{Y}_{base}_s{seed}")
        if not (out_dir / ckpt_name).exists():
            raise RuntimeError(f"{base} cut{Y} s{seed}: no ckpt at {out_dir}")
        print(f"  [train] {base} cut{Y} s{seed} ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------- regression
def regression(device):
    print("[regress] re-score HEADLINE baseline checkpoints on canonical split...")
    cell = {b: (lambda s, b=b: _ROOT / HEADLINE[b][0] / f"seed{s}") for b in HEADLINE}
    df = eval_cutoff(list(HEADLINE.keys()), CANON_SPLIT, CANON_NORM, CANON_TEST_FIRST, cell, device, "regress")
    df["cov_avg"] = df[[f"tS_cov95_h{h}" for h in HORIZONS]].mean(axis=1)
    print(f"\n{'baseline':<24}{'recomputed':>12}{'headline':>10}{'|Δ|':>8}")
    ok = True
    for b, hcov in HEADLINE_COV.items():
        sub = df[df.baseline == b]
        if len(sub) == 0:
            print(f"  {b:<22} MISSING"); ok = False; continue
        rc = sub.groupby("region").cov_avg.mean().mean()
        d = abs(rc - hcov)
        tol = 0.02 if b.startswith("dlinear") else 0.04     # dlinear seeded; MC ~0.04
        flag = "PASS" if d <= tol else "FAIL"
        if d > tol: ok = False
        print(f"  {b:<24}{rc:>12.3f}{hcov:>10.3f}{d:>8.3f}  {flag}(tol {tol})")
    print(f"\n[regress] {'PASS: baseline eval reproduces headline (within MC error) -> safe for rolling' if ok else 'FAIL: investigate before rolling'}")
    (ROLL_ROOT / "baseline_regression.csv").write_text(df.to_csv(index=False))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=CUTOFFS)
    ap.add_argument("--baselines", nargs="+", default=["lstm", "vanilla_mamba", "patchtst", "epideep", "dlinear"])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.regress:
        return 0 if regression(args.device) else 2

    all_rows = []
    for Y in args.cutoffs:
        cut_dir = ROLL_ROOT / f"cut{Y}"
        cut_csv = cut_dir / "ili_env_weekly_split.csv"
        cut_norm = cut_dir / "normalization_params.json"
        test_first = Y * 100 + 40
        print(f"\n{'='*64}\n[baselines] cut{Y} test={Y}-{Y+1} (>= {test_first})\n{'='*64}")
        for base in args.baselines:
            retrain_cutoff(base, Y, cut_csv, cut_norm, args.device)
        cell = {b: (lambda s, b=b, Y=Y: BASE_OUT / b / f"cut{Y}" / f"seed{s}") for b in args.baselines}
        df = eval_cutoff(args.baselines, cut_csv, cut_norm, test_first, cell, args.device, f"cut{Y}")
        df["cutoff"] = Y; df["test_season"] = f"{Y}-{Y+1}"
        all_rows.append(df)
        pd.concat(all_rows, ignore_index=True).to_csv(ROLL_ROOT / "baseline_regional_results.csv", index=False)
    out = ROLL_ROOT / "baseline_regional_results.csv"
    pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
    print(f"\n[baselines] done -> {out.relative_to(_ROOT)}")
    print("[baselines] NEXT: merge with CG side + apply pre-registered verdict table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
