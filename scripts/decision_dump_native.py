"""Decision-value native predictive dump + consistency gate.

Produces runs/decision_native/native_predictive.parquet with, per
(model, seed, region, eps_h1, horizon): y_true, mu, sigma and the 23
FluSight quantile columns (src.eval.wis.REQUIRED_QUANTILES).

Faithfully REUSES the canonical inference paths (does NOT re-implement models):
  - lstm / vanilla_mamba / patchtst : eval_nn_wis loading + _mc_samples_nn
        (MC Dropout n=100; rates lstm 0.3, vanilla_mamba 0.1, patchtst 0.1)
        [scripts/phase_3_region_wis.py]
  - epideep                         : eval_epideep_wis (MC Dropout d=0.1, n=100)
        [scripts/phase_3_region_wis_extras.py]
  - dlinear                         : _dlinear_predict_one_seed x5 seeds ->
        ensemble mean mu, sample-std sigma (ddof=1); Gaussian native UQ.
        Quantiles = mu + Phi^{-1}(q)*sigma. seed = -1.
  - cg_mamba                        : READ runs/regime_shift/per_origin_forecasts.parquet
        (NOT re-run). sigma = sqrt(s2_total); quantiles = mu + Phi^{-1}(q)*sigma.

For MC-sampled models the 23 quantiles come from np.quantile(samples, q, axis=0)
EXACTLY as the canonical WIS uses; mu = samples.mean, sigma = samples.std.

NAMESPACE COLLISION: patchtst (TSLIB) collides with dlinear/epideep when imported
in the same process. Run model GROUPS in separate processes via --group:
  {nn: lstm,vanilla_mamba,patchtst}  {extras: epideep,dlinear}  {cg: cg_mamba}
Each group writes a temp parquet; --merge concatenates + runs the consistency gate.

Usage:
  python scripts/decision_dump_native.py --group nn     --device cuda:0
  python scripts/decision_dump_native.py --group extras --device cuda:0
  python scripts/decision_dump_native.py --group cg
  python scripts/decision_dump_native.py --merge
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240
REGIONS = [f"hhs{i}" for i in range(1, 11)]
SEEDS = [42, 123, 456, 789, 1024]
N_MC_SAMPLES = 100
DROPOUT_MC = {"lstm": 0.3, "vanilla_mamba": 0.1, "patchtst": 0.1}
EPIDEEP_DROPOUT = 0.1
MC_SEED = 20260802

OUT_DIR = _ROOT / "runs" / "decision_native"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from src.eval.wis import REQUIRED_QUANTILES  # noqa: E402

QCOLS = [f"q{q:.3f}" for q in REQUIRED_QUANTILES]
QMAP = list(zip(REQUIRED_QUANTILES, QCOLS))


def _seed_mc():
    import torch
    torch.manual_seed(MC_SEED)
    np.random.seed(MC_SEED)


def _rows_from_samples(model_name, seed, region, samples, y_raw, eps_h1):
    """samples [S,N,H] raw, y_raw [N,H] raw, eps_h1 [N] -> long-format rows list."""
    from scipy.stats import mstats  # noqa: F401  (ensure scipy present)
    N, H = y_raw.shape
    mu = samples.mean(axis=0)          # [N,H]
    sigma = samples.std(axis=0)        # [N,H] (population std, matches np default)
    quant = {q: np.quantile(samples, q, axis=0) for q in REQUIRED_QUANTILES}  # each [N,H]
    recs = []
    for h_idx, h in enumerate(HORIZONS):
        base = {
            "model": model_name, "seed": seed, "region": region,
            "horizon": h,
        }
        block = pd.DataFrame({
            "model": model_name, "seed": seed, "region": region,
            "eps_h1": eps_h1.astype(np.int64), "horizon": h,
            "y_true": y_raw[:, h_idx], "mu": mu[:, h_idx], "sigma": sigma[:, h_idx],
        })
        for q, col in QMAP:
            block[col] = quant[q][:, h_idx]
        recs.append(block)
    return pd.concat(recs, ignore_index=True)


def _rows_from_gaussian(model_name, seed, region, mu, sigma, y_raw, eps_h1):
    """Analytic Gaussian quantiles: q = mu + Phi^{-1}(q)*sigma. All [N,H]."""
    from scipy.stats import norm
    recs = []
    for h_idx, h in enumerate(HORIZONS):
        block = pd.DataFrame({
            "model": model_name, "seed": seed, "region": region,
            "eps_h1": eps_h1.astype(np.int64), "horizon": h,
            "y_true": y_raw[:, h_idx], "mu": mu[:, h_idx], "sigma": sigma[:, h_idx],
        })
        for q, col in QMAP:
            block[col] = mu[:, h_idx] + norm.ppf(q) * sigma[:, h_idx]
        recs.append(block)
    return pd.concat(recs, ignore_index=True)


# ─────────────────────────── NN group (lstm / vanilla_mamba / patchtst) ────────
def run_nn(device):
    import torch
    from scripts.phase_3_region_wis import (
        eval_nn_wis as _unused,  # noqa: F401 (keep import path parity)
    )
    # Reuse the canonical loader + sampler directly.
    from scripts.phase_3_region_wis import _mc_samples_nn, NORM, DROPOUT_MC as CANON_DROP
    from scripts.phase_3_region_eval import build_region_df
    from baselines.lstm import WeeklyMultiHorizonDataset
    from cm_mamba.baselines.lstm_baseline import LSTMForecaster
    from baselines.vanilla_mamba import VanillaMambaForecaster
    from baselines.patchtst import PatchTSTForecaster

    if not torch.cuda.is_available():
        device = "cpu"
    base_dirs = {
        "lstm": ("runs/lstm_final/h256_l2_lr5e-04_bs16", "lstm_best.pt", LSTMForecaster),
        "vanilla_mamba": ("runs/vanilla_mamba_final/d64_nl3_lr5e-04", "vanilla_mamba_best.pt", VanillaMambaForecaster),
        "patchtst": ("runs/patchtst_final/pl16_dm128_lr5e-04", "patchtst_best.pt", PatchTSTForecaster),
    }
    all_rows = []
    for region in REGIONS:
        region_df = build_region_df(region)
        for base in ["lstm", "vanilla_mamba", "patchtst"]:
            cfg_dir, ckpt_name, ModelCls = base_dirs[base]
            for seed in SEEDS:
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
                else:  # patchtst
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
                _seed_mc()
                samples, y_raw, eps_h1 = _mc_samples_nn(model, ds, device, DROPOUT_MC[base])
                all_rows.append(_rows_from_samples(base, seed, region, samples, y_raw, eps_h1))
                print(f"  OK {base:<14} {region} s={seed}  N={len(y_raw)}", flush=True)
    out = pd.concat(all_rows, ignore_index=True)
    dst = OUT_DIR / "_tmp_nn.parquet"
    out.to_parquet(dst, index=False)
    print(f"\nSaved {dst}  rows={len(out)}", flush=True)


# ─────────────────────────── extras group (epideep / dlinear) ─────────────────
def run_extras(device):
    import torch
    from scripts.phase_3_region_wis_extras import (
        eval_epideep_wis, _dlinear_predict_one_seed, build_region_df,
        NORM, TARGET_MEAN, TARGET_STD, EPIDEEP_DIR, DLINEAR_DIR,
    )
    from baselines.lstm import WeeklyMultiHorizonDataset
    from src.baselines.epideep import EpiDeepForecaster
    from src.eval.quantile_predictions import _dropout_train_mode
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        device = "cpu"
    all_rows = []
    for region in REGIONS:
        region_df = build_region_df(region)

        # ---- EpiDeep MC Dropout (5 seeds), inlined to grab raw samples ----
        for seed in SEEDS:
            p = EPIDEEP_DIR / f"seed{seed}"
            r = json.load(open(p / "results.json"))
            cfg = r["config"]
            ckpt = torch.load(p / "epideep_best.pt", map_location=device, weights_only=True)
            model = EpiDeepForecaster(
                seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
                d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
                decoder_hidden=cfg["decoder_hidden"],
                alignment_weight=cfg.get("alignment_weight", 0.0),
                dropout=EPIDEEP_DROPOUT, target_only=cfg.get("target_only", False),
            )
            model.load_state_dict(ckpt)
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = EPIDEEP_DROPOUT
            ds = WeeklyMultiHorizonDataset(region_df, "test", NORM, lookback=cfg["seq_len"], pred_len=cfg["pred_len"])
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
            model.eval().to(device)
            _seed_mc()
            all_samples, y_collect = [], None
            with _dropout_train_mode(model):
                with torch.no_grad():
                    for _ in range(N_MC_SAMPLES):
                        preds, ys = [], []
                        for x, y in loader:
                            x = x.to(device)
                            preds.append(model(x).cpu().numpy())
                            ys.append(y.numpy())
                        preds = np.concatenate(preds, axis=0)
                        ys = np.concatenate(ys, axis=0)
                        all_samples.append(preds)
                        if y_collect is None:
                            y_collect = ys
            samples = np.stack(all_samples, axis=0) * TARGET_STD + TARGET_MEAN
            y_raw = y_collect * TARGET_STD + TARGET_MEAN
            eps = ds.df["epiweek"].astype(int).to_numpy()
            eps_h1 = eps[ds.window_ends + 1]
            all_rows.append(_rows_from_samples("epideep", seed, region, samples, y_raw, eps_h1))
            print(f"  OK epideep {region} s={seed}  N={len(y_raw)}", flush=True)

        # ---- DLinear 5-seed ensemble Gaussian (seed=-1) ----
        per_seed, ys_ref, eps_ref = [], None, None
        for seed in SEEDS:
            preds_z, ys_z, eps_h1 = _dlinear_predict_one_seed(seed, region_df, device)
            per_seed.append(preds_z)
            if ys_ref is None:
                ys_ref, eps_ref = ys_z, eps_h1
        per_seed = np.stack(per_seed, axis=0)          # [5,N,H]
        mu_z = per_seed.mean(axis=0)
        sigma_z = per_seed.std(axis=0, ddof=1)         # sample std, ddof=1 (canonical)
        mu_raw = mu_z * TARGET_STD + TARGET_MEAN
        sigma_raw = sigma_z * TARGET_STD
        y_raw = ys_ref * TARGET_STD + TARGET_MEAN
        all_rows.append(_rows_from_gaussian("dlinear", -1, region, mu_raw, sigma_raw, y_raw, eps_ref))
        print(f"  OK dlinear {region} ensemble  N={len(y_raw)}", flush=True)

    out = pd.concat(all_rows, ignore_index=True)
    dst = OUT_DIR / "_tmp_extras.parquet"
    out.to_parquet(dst, index=False)
    print(f"\nSaved {dst}  rows={len(out)}", flush=True)


# ─────────────────────────── cg group (read regime_shift) ─────────────────────
def run_cg():
    src = _ROOT / "runs" / "regime_shift" / "per_origin_forecasts.parquet"
    df = pd.read_parquet(src)
    # regime_shift only stores 'val' and 'test_strict' for cg — keep test only.
    cg = df[(df.model == "cg_mamba") & (df.region.isin(REGIONS))
            & (df.split == "test_strict")].copy()
    # sigma from s2_total (Gaussian native APMD); overwrite parquet 'sigma' to be safe
    cg["sigma"] = np.sqrt(cg["s2_total"].to_numpy())
    # eps_h1 = target_ep at horizon 1 for each (region, seed, origin_ep)
    h1 = (cg[cg.horizon == 1][["region", "seed", "origin_ep", "target_ep"]]
          .rename(columns={"target_ep": "eps_h1"}))
    cg = cg.merge(h1, on=["region", "seed", "origin_ep"], how="left")
    from scipy.stats import norm
    rows = pd.DataFrame({
        "model": "cg_mamba", "seed": cg["seed"].astype(int), "region": cg["region"],
        "eps_h1": cg["eps_h1"].astype(np.int64), "horizon": cg["horizon"].astype(int),
        "y_true": cg["y_true"].to_numpy(), "mu": cg["mu"].to_numpy(), "sigma": cg["sigma"].to_numpy(),
    })
    mu = cg["mu"].to_numpy(); sigma = cg["sigma"].to_numpy()
    for q, col in QMAP:
        rows[col] = mu + norm.ppf(q) * sigma
    dst = OUT_DIR / "_tmp_cg.parquet"
    rows.to_parquet(dst, index=False)
    print(f"Saved {dst}  rows={len(rows)}  (regime_shift split={sorted(cg['split'].unique())})", flush=True)


# ─────────────────────────── merge + consistency gate ────────────────────────
def _agg_from_dump(sub):
    """sub = dumped rows for one (model,region,seed), test_strict only.
    Returns (wis_h14_mean, cov95_h14_mean)."""
    from src.eval.wis import wis, coverage
    wl, cl = [], []
    for h in HORIZONS:
        g = sub[sub.horizon == h]
        if len(g) == 0:
            return np.nan, np.nan
        qf = {q: g[col].to_numpy() for q, col in QMAP}
        y = g["y_true"].to_numpy()
        wl.append(float(wis(y, qf).mean()))
        cl.append(float(coverage(y, qf, alpha=0.05)))
    return float(np.mean(wl)), float(np.mean(cl))


def merge_and_gate():
    parts = []
    for g in ["nn", "extras", "cg"]:
        f = OUT_DIR / f"_tmp_{g}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
            print(f"  loaded {f.name}  rows={len(parts[-1])}", flush=True)
        else:
            print(f"  MISSING {f.name} (group not run)", flush=True)
    full = pd.concat(parts, ignore_index=True)
    col_order = ["model", "seed", "region", "eps_h1", "horizon", "y_true", "mu", "sigma"] + QCOLS
    full = full[col_order]
    dst = OUT_DIR / "native_predictive.parquet"
    full.to_parquet(dst, index=False)
    print(f"\nSaved {dst}  rows={len(full)}  cols={len(full.columns)}", flush=True)
    print(f"models: {sorted(full.model.unique())}", flush=True)

    # ---- consistency gate vs canonical aggregates (test_strict) ----
    canon_wis = pd.read_csv(_ROOT / "runs" / "phase_3_region_wis.csv")
    canon_ext = pd.read_csv(_ROOT / "runs" / "phase_3_region_wis_extras.csv")
    canon_cg = pd.read_csv(_ROOT / "runs" / "phase_3_cgm_method_f_region.csv")

    def canon_row(cdf, base, region, seed):
        m = cdf[(cdf.baseline == base) & (cdf.region == region) & (cdf.seed == seed)]
        if len(m) == 0:
            return None
        row = m.iloc[0]
        w = np.mean([row[f"tS_wis_h{h}"] for h in HORIZONS])
        c = np.mean([row[f"tS_cov95_h{h}"] for h in HORIZONS])
        return float(w), float(c)

    canon_src = {
        "lstm": (canon_wis, "lstm"), "vanilla_mamba": (canon_wis, "vanilla_mamba"),
        "patchtst": (canon_wis, "patchtst"),
        "epideep": (canon_ext, "epideep"), "dlinear": (canon_ext, "dlinear_ensemble_gauss"),
        "cg_mamba": (canon_cg, "cg_mamba_method_F"),
    }

    print("\n=== CONSISTENCY GATE (dumped test_strict vs canonical) ===", flush=True)
    print(f"{'model':<15}{'n_cell':>7}{'WIS mean|d|':>13}{'WIS max|d|':>12}"
          f"{'Cov mean|d|':>13}{'Cov max|d|':>12}", flush=True)
    gate_rows = []
    for model, (cdf, base) in canon_src.items():
        sub_m = full[full.model == model]
        strict = sub_m[sub_m.eps_h1 >= TS_BOUNDARY]
        seeds = sorted(strict.seed.unique())
        wdev, cdev = [], []
        for region in REGIONS:
            for seed in seeds:
                cell = strict[(strict.region == region) & (strict.seed == seed)]
                if len(cell) == 0:
                    continue
                w_d, c_d = _agg_from_dump(cell)
                cr = canon_row(cdf, base, region, seed)
                if cr is None or np.isnan(w_d):
                    continue
                w_c, c_c = cr
                wdev.append(abs(w_d - w_c))
                cdev.append(abs(c_d - c_c))
                gate_rows.append({"model": model, "region": region, "seed": seed,
                                  "wis_dump": w_d, "wis_canon": w_c,
                                  "cov_dump": c_d, "cov_canon": c_c})
        wdev = np.array(wdev); cdev = np.array(cdev)
        print(f"{model:<15}{len(wdev):>7}{wdev.mean():>13.5f}{wdev.max():>12.5f}"
              f"{cdev.mean():>13.5f}{cdev.max():>12.5f}", flush=True)
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(OUT_DIR / "consistency_gate_detail.csv", index=False)
    print(f"\nGate detail: {OUT_DIR / 'consistency_gate_detail.csv'}  ({len(gate_df)} cells)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["nn", "extras", "cg"])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.group == "nn":
        run_nn(args.device)
    elif args.group == "extras":
        run_extras(args.device)
    elif args.group == "cg":
        run_cg()
    elif args.merge:
        merge_and_gate()
    else:
        ap.error("specify --group {nn,extras,cg} or --merge")


if __name__ == "__main__":
    main()
