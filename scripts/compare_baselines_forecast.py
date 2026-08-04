"""Multi-baseline forecast comparison for Figure 5 — seed=42 illustrative.

Generates quantile predictions for 4 DL baselines (Vanilla Mamba MC d=0.1, DLinear ensemble Gaussian,
PatchTST MC d=0.1, EpiDeep MC d=0.1) on test_strict at seed=42, then renders a 5-row × 4-col figure
(rows: CG-Mamba [Method F, loaded from existing CSV] + 4 baselines; cols: h=1,2,3,4).

CGM data: runs/wis_method_f/decomposition_temporal_5seed.csv (seed=42 rows)
Output:
  runs/compare_baselines/baseline_predictions_seed42.csv  ← per-baseline μ + 95% PI band
  notebooks/figures/forecast_compare/forecast_compare.{pdf,png}
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# GPU 1번
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.vanilla_mamba import VanillaMambaForecaster                     # noqa: E402
from baselines.patchtst import PatchTSTForecaster                              # noqa: E402
from baselines.epideep import EpiDeepForecaster                                # noqa: E402
from baselines.dlinear import DLinearForecaster                                # noqa: E402

from src.data.loader import load_dataset_csv, load_norm_params                # noqa: E402
from baselines.lstm import WeeklyMultiHorizonDataset                           # noqa: E402
from torch.utils.data import DataLoader                                        # noqa: E402

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_CSV = _ROOT / "runs/compare_baselines/baseline_predictions_seed42.csv"
OUT_FIG = _ROOT / "notebooks/figures/forecast_compare/forecast_compare"
COVID_STRICT_START = 202240
SEED = 42
N_MC = 100
HORIZONS = [1, 2, 3, 4]
PRED_LEN = 4

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # CUDA_VISIBLE_DEVICES=1 → cuda:0


def _epiweek_to_date(ep: int) -> pd.Timestamp:
    y, w = ep // 100, ep % 100
    jan4 = pd.Timestamp(year=y, month=1, day=4)
    iso1 = jan4 - pd.Timedelta(days=jan4.isoweekday() % 7)
    return iso1 + pd.Timedelta(weeks=w - 1)


def _enable_dropout_only(model):
    """Set eval but keep Dropout layers in train mode (for MC Dropout)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    return model


def _mc_forward(model, loader, n_samples: int):
    """Returns samples [n_samples, n, H], y [n, H]."""
    samples_all = []
    y_all = None
    for s in range(n_samples):
        preds_s, y_s = [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(DEVICE)
                pred = model(x)
                preds_s.append(pred.cpu().numpy())
                if s == 0:
                    y_s.append(y.numpy())
        samples_all.append(np.concatenate(preds_s, axis=0))
        if s == 0:
            y_all = np.concatenate(y_s, axis=0)
    return np.stack(samples_all, axis=0), y_all


def _det_forward(model, loader):
    """Deterministic forward — returns preds [n, H], y [n, H]."""
    preds, ys = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(DEVICE)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0)


def _build_loader(df, norm, lookback):
    ds = WeeklyMultiHorizonDataset(df, "test", norm, lookback=lookback, pred_len=PRED_LEN)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    # eps_h1 from window_ends
    eps_arr = df["epiweek"].astype(int).to_numpy()
    eps_h1 = eps_arr[ds.window_ends + 1]  # target epiweek at h=1
    return loader, eps_h1


def _build_model(baseline: str, cfg: dict, dropout: float):
    if baseline == "vanilla_mamba":
        return VanillaMambaForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_state=cfg["d_state"],
            dt_rank=cfg["dt_rank"], expand=cfg["expand"], dropout=dropout,
        )
    elif baseline == "patchtst":
        stride = max(1, int(cfg["patch_len"] * cfg["stride_ratio"]))
        return PatchTSTForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
            d_ff=int(cfg["d_ff_ratio"] * cfg["d_model"]),
            patch_len=cfg["patch_len"], stride=stride, dropout=dropout,
        )
    elif baseline == "epideep":
        return EpiDeepForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
            decoder_hidden=cfg["decoder_hidden"],
            alignment_weight=cfg["alignment_weight"],
            dropout=dropout,
            target_only=cfg.get("target_only", False),
        )
    elif baseline == "dlinear":
        return DLinearForecaster(
            seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
            moving_avg=cfg["moving_avg"], individual=cfg["individual"],
        )
    raise ValueError(baseline)


BASELINES = [
    ("vanilla_mamba", "d64_nl3_lr5e-04", "vanilla_mamba_best.pt", 0.1, "mc"),
    ("dlinear",       "ma13_indF_lr2e-03", "dlinear_best.pt",     0.0, "ensemble"),  # 5-seed ensemble (no MC)
    ("patchtst",      "pl16_dm128_lr5e-04", "patchtst_best.pt",   0.1, "mc"),
    ("epideep",       "de128_eh64_lr2e-03", "epideep_best.pt",    0.1, "mc"),
]


def main():
    print(f"Device: {DEVICE}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    norm = load_norm_params(NORM_PATH)
    tmean = float(norm["ili_weighted_pct"]["mean"])
    tstd = float(norm["ili_weighted_pct"]["std"])
    df = load_dataset_csv(CSV_PATH)
    # Mask out covid-excluded rows for test_strict (eps >= 202240 only kept as 'test')
    # The dataset uses df.split filter; we additionally filter test rows where eps >= COVID_STRICT_START
    df = df.copy()
    eps_int = df["epiweek"].astype(int)
    # rows that are 'test' but eps < COVID_STRICT_START → exclude (covid-period not strict)
    excl_mask = (df["split"] == "test") & (eps_int < COVID_STRICT_START)
    df.loc[excl_mask, "split"] = "_covid_excluded"

    records = []  # rows for unified CSV

    # === DL baselines ===
    for baseline, cfg_name, ckpt_name, dropout, uq_mode in BASELINES:
        print(f"\n--- {baseline} ({uq_mode}, dropout={dropout}) ---")
        if uq_mode == "ensemble":
            # 5-seed ensemble for DLinear
            seeds = [42, 123, 456, 789, 1024]
            ensemble_preds = []
            eps_h1_arr = None
            for s in seeds:
                ckpt_dir = _ROOT / f"runs/{baseline}_final/{cfg_name}/seed{s}"
                cfg = json.loads((ckpt_dir / "results.json").read_text())["config"]
                model = _build_model(baseline, cfg, dropout=0.0).to(DEVICE)
                ckpt = torch.load(ckpt_dir / ckpt_name, map_location=DEVICE, weights_only=True)
                model.load_state_dict(ckpt)
                loader, eps_h1_arr = _build_loader(df, norm, cfg["seq_len"])
                preds, y = _det_forward(model, loader)
                ensemble_preds.append(preds)
                if s == 42:
                    y_42 = y
            ensemble = np.stack(ensemble_preds, axis=0)
            mu_z = ensemble.mean(axis=0)
            sigma_z = ensemble.std(axis=0, ddof=1)
            mu = mu_z * tstd + tmean
            sigma = sigma_z * tstd
            y_raw = y_42 * tstd + tmean
            eps_arr = eps_h1_arr
        else:
            # Single seed=42, MC Dropout
            ckpt_dir = _ROOT / f"runs/{baseline}_final/{cfg_name}/seed42"
            cfg = json.loads((ckpt_dir / "results.json").read_text())["config"]
            model = _build_model(baseline, cfg, dropout=dropout).to(DEVICE)
            ckpt = torch.load(ckpt_dir / ckpt_name, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt)
            _enable_dropout_only(model)
            loader, eps_h1_arr = _build_loader(df, norm, cfg["seq_len"])
            samples_z, y_z = _mc_forward(model, loader, N_MC)
            mu_z = samples_z.mean(axis=0)
            sigma_z = samples_z.std(axis=0, ddof=1)
            mu = mu_z * tstd + tmean
            sigma = sigma_z * tstd
            y_raw = y_z * tstd + tmean
            eps_arr = eps_h1_arr

        # eps_arr is h=1 target epiweek (window_ends + 1). For h>=2, target_ep = eps_h1 + (h-1).
        # All test_strict because we already masked covid in df.
        print(f"  n_windows={len(eps_arr)}")
        for n in range(len(mu)):
            ep_h1 = int(eps_arr[n])
            for h_idx, h in enumerate(HORIZONS):
                ep_h = ep_h1 + (h - 1)
                # naive epiweek arithmetic — handle year boundary
                yr, wk = ep_h1 // 100, ep_h1 % 100 + (h - 1)
                while wk > 52:
                    wk -= 52
                    yr += 1
                ep_h = yr * 100 + wk
                records.append({
                    "baseline": baseline,
                    "horizon": h,
                    "target_ep": ep_h,
                    "mu": float(mu[n, h_idx]),
                    "sigma": float(sigma[n, h_idx]),
                    "y_true": float(y_raw[n, h_idx]),
                })

    # === Save unified CSV ===
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV.relative_to(_ROOT)} ({len(out_df)} rows)")

    # === Load CGM Method F from existing CSV ===
    cgm = pd.read_csv(_ROOT / "runs/wis_method_f/decomposition_temporal_5seed.csv")
    cgm = cgm[cgm.seed == 42].copy()
    cgm["mu"] = cgm.mu_CGM_raw
    cgm["sigma"] = np.sqrt(cgm.sigma2_total)
    cgm["y_true"] = cgm.y_raw
    cgm["baseline"] = "cg_mamba"
    cgm = cgm[["baseline", "horizon", "target_ep", "mu", "sigma", "y_true"]]

    all_df = pd.concat([cgm, out_df], ignore_index=True)
    all_df["lo95"] = all_df.mu - 1.96 * all_df.sigma
    all_df["hi95"] = all_df.mu + 1.96 * all_df.sigma
    all_df["date"] = all_df.target_ep.apply(_epiweek_to_date)

    # === Render 5x4 figure ===
    baselines_order = ["cg_mamba", "vanilla_mamba", "dlinear", "patchtst", "epideep"]
    baseline_labels = {
        "cg_mamba": "CG-Mamba (Method F)",
        "vanilla_mamba": "Vanilla Mamba (MC d=0.1)",
        "dlinear": "DLinear (5-seed ensemble)",
        "patchtst": "PatchTST (MC d=0.1)",
        "epideep": "EpiDeep (MC d=0.1)",
    }
    baseline_colors = {
        "cg_mamba": "#1f77b4",      # blue
        "vanilla_mamba": "#ff7f0e", # orange
        "dlinear": "#2ca02c",       # green
        "patchtst": "#d62728",      # red
        "epideep": "#9467bd",       # purple
    }

    fig, axes = plt.subplots(5, 4, figsize=(16, 14), sharex=True, sharey=True)
    for row_i, base in enumerate(baselines_order):
        sub_b = all_df[all_df.baseline == base]
        col = baseline_colors[base]
        label = baseline_labels[base]
        for col_i, h in enumerate(HORIZONS):
            ax = axes[row_i, col_i]
            sub = sub_b[sub_b.horizon == h].sort_values("date").reset_index(drop=True)
            inside = ((sub.y_true >= sub.lo95) & (sub.y_true <= sub.hi95)).mean()
            mae = (sub.y_true - sub.mu).abs().mean()
            ax.fill_between(sub.date, sub.lo95, sub.hi95, color=col, alpha=0.20)
            ax.plot(sub.date, sub.mu, color=col, linewidth=1.3)
            ax.plot(sub.date, sub.y_true, color="black", linewidth=1.0, alpha=0.85)
            ax.axvspan(pd.Timestamp("2024-08-04"), pd.Timestamp("2025-08-31"),
                       color="orange", alpha=0.05, zorder=0)
            ax.set_title(f"Cov95={inside:.3f}, MAE={mae:.3f}", fontsize=9, loc="left")
            if col_i == 0:
                ax.set_ylabel(f"{label}\nWeighted ILI (%)", fontsize=9)
            if row_i == 0:
                ax.text(0.5, 1.18, f"h = {h}", transform=ax.transAxes,
                        ha="center", fontsize=11, fontweight="bold")
            if row_i == 4:
                ax.set_xlabel("Date (MMWR week)", fontsize=9)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Figure 5 — Multi-baseline qualitative forecast on test_strict (seed=42, illustrative)\n"
        "Shaded bands: 95% prediction intervals from each baseline's architecturally-natural UQ method.\n"
        "Rows: CG-Mamba (Method F) + 4 DL baselines. Columns: forecast horizons h=1..4.\n"
        "Empirical Cov95 per panel = fraction of ground truth within band (for this seed only; 5-seed aggregate in Table I).",
        fontsize=11, y=0.998,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT_FIG}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_FIG}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {OUT_FIG.relative_to(_ROOT)}.pdf + .png")

    # === Summary per (baseline, horizon) ===
    print("\n=== Per-(baseline, horizon) Cov95 + MAE (test_strict, seed=42) ===")
    summary_rows = []
    for base in baselines_order:
        sub_b = all_df[all_df.baseline == base]
        row = [baseline_labels[base]]
        for h in HORIZONS:
            sub = sub_b[sub_b.horizon == h]
            cov = ((sub.y_true >= sub.lo95) & (sub.y_true <= sub.hi95)).mean()
            mae = (sub.y_true - sub.mu).abs().mean()
            row.append(f"Cov95={cov:.3f}, MAE={mae:.3f}")
        summary_rows.append(row)
        print(f"  {row[0]:<35s} | " + " | ".join(row[1:]))


if __name__ == "__main__":
    main()
