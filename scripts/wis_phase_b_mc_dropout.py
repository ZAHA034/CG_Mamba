"""WIS Phase B group 3 — MC Dropout for 4 Tier-3 NN baselines (PLAN J.3).

These baselines already have dropout > 0 (all dropout=0.1 — PLAN J.5):
  - PatchTST     (pl16_dm64_lr5e-04, dm=64, n_heads=4, el=2)
  - iTransformer (dm256_el4_lr5e-04, dm=256, n_heads=4, el=4)
  - TimesNet     (d64_el2_lr1e-03,   dm=64, el=2, top_k=5, num_kernels=6)
  - EpiDeep      (de128_eh64_lr2e-03, d_emb=128, enc=64, dec=128)

→ MC Dropout (Gal & Ghahramani 2016) at existing dropout rate, n=100 samples.
   dropout_layers_only=True (preserves BN/LN running stats).

Per-seed WIS: 5 seeds × per-split WIS → mean ± std.
Output: runs/wis_phase_b/{baseline}/wis_results.json

GPU required. Recommended to run AFTER Phase C completes (so GPU 1 is free).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.lstm import WeeklyMultiHorizonDataset                       # noqa: E402
from baselines.patchtst import PatchTSTForecaster                          # noqa: E402
from baselines.itransformer import ITransformerForecaster                  # noqa: E402

from src.data.loader import load_dataset_csv, load_norm_params             # noqa: E402
from src.eval.wis import wis, wis_decomposed, coverage, REQUIRED_QUANTILES  # noqa: E402
from src.eval.quantile_predictions import _dropout_train_mode              # noqa: E402

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"
OUT_ROOT = _ROOT / "runs" / "wis_phase_b"
COVID_STRICT_START_EPIWEEK = 202240

SEEDS = (42, 123, 456, 789, 1024)


# ─── Data helpers ──────────────────────────────────────────────────────────


def _build_loader(df, split_name, lookback, pred_len, norm,
                  epi_min=None, batch_size=32):
    if epi_min is not None:
        sub = df.copy()
        sub.loc[(sub["split"] == split_name) & (sub["epiweek"] < epi_min),
                "split"] = "_excluded"
        ds_df = sub
    else:
        ds_df = df
    ds = WeeklyMultiHorizonDataset(ds_df, split_name, norm,
                                   lookback=lookback, pred_len=pred_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), len(ds)


@torch.no_grad()
def _mc_inference(model, loader, n_samples: int,
                  target_mean: float, target_std: float, device: str):
    """Run n MC dropout passes per batch → (samples [S, N, H], y_raw [N, H])."""
    model.eval()
    all_samples = []
    y_collect = None
    with _dropout_train_mode(model):
        for _ in range(n_samples):
            preds_per_batch = []
            ys_per_batch = []
            for x, y in loader:
                x = x.to(device); y = y.to(device)
                pred = model(x)                        # [B, H]
                preds_per_batch.append(pred.cpu().numpy())
                ys_per_batch.append(y.cpu().numpy())
            preds_all = np.concatenate(preds_per_batch, axis=0)   # [N, H]
            ys_all = np.concatenate(ys_per_batch, axis=0)
            all_samples.append(preds_all)
            if y_collect is None:
                y_collect = ys_all
    samples = np.stack(all_samples, axis=0)                       # [S, N, H]
    samples_raw = samples * target_std + target_mean
    y_raw = y_collect * target_std + target_mean
    return samples_raw, y_raw


def _samples_to_quantiles(samples_raw: np.ndarray) -> dict:
    """[S, N, H] samples → dict q → [N, H] quantiles."""
    out = {}
    for q in REQUIRED_QUANTILES:
        out[q] = np.quantile(samples_raw, q, axis=0)
    return out


def _score_split(quantile_forecasts: dict, y_true: np.ndarray) -> dict:
    N, H = y_true.shape
    wis_per_h, disp_per_h, under_per_h, over_per_h = [], [], [], []
    for h in range(H):
        qf_h = {q: quantile_forecasts[q][:, h] for q in quantile_forecasts}
        y_h = y_true[:, h]
        w = wis(y_h, qf_h)
        wis_per_h.append(float(w.mean()))
        parts = wis_decomposed(y_h, qf_h)
        disp_per_h.append(float(parts["dispersion"].mean()))
        under_per_h.append(float(parts["under"].mean()))
        over_per_h.append(float(parts["over"].mean()))
    qf_flat = {q: quantile_forecasts[q].reshape(-1) for q in quantile_forecasts}
    y_flat = y_true.reshape(-1)
    return {
        "n": int(N),
        "wis_per_horizon": wis_per_h,
        "wis_avg": float(np.mean(wis_per_h)),
        "wis_decomposed": {
            "dispersion_per_horizon": disp_per_h,
            "under_per_horizon": under_per_h,
            "over_per_horizon": over_per_h,
            "dispersion_avg": float(np.mean(disp_per_h)),
            "under_avg": float(np.mean(under_per_h)),
            "over_avg": float(np.mean(over_per_h)),
        },
        "coverage_50": coverage(y_flat, qf_flat, alpha=0.5),
        "coverage_95": coverage(y_flat, qf_flat, alpha=0.05),
    }


# ─── Model builders ────────────────────────────────────────────────────────


def _build_patchtst(cfg):
    stride = max(1, int(cfg["patch_len"] * cfg["stride_ratio"]))
    return PatchTSTForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
        d_ff=cfg["d_ff_ratio"] * cfg["d_model"],
        patch_len=cfg["patch_len"], stride=stride, dropout=cfg["dropout"],
    )


def _build_itransformer(cfg):
    return ITransformerForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], e_layers=cfg["e_layers"],
        d_ff=cfg["d_ff_ratio"] * cfg["d_model"], dropout=cfg["dropout"],
        embed=cfg.get("embed", "timeF"), freq=cfg.get("freq", "w"),
    )


def _build_timesnet(cfg):
    from baselines.timesnet import TimesNetForecaster  # noqa: E402
    return TimesNetForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_model=cfg["d_model"], d_ff=cfg["d_ff"], e_layers=cfg["e_layers"],
        top_k=cfg["top_k"], num_kernels=cfg["num_kernels"], dropout=cfg["dropout"],
    )


def _build_epideep(cfg):
    from baselines.epideep import EpiDeepForecaster  # noqa: E402
    return EpiDeepForecaster(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"], enc_in=cfg["enc_in"],
        d_emb=cfg["d_emb"], encoder_hidden=cfg["encoder_hidden"],
        decoder_hidden=cfg["decoder_hidden"],
        alignment_weight=cfg["alignment_weight"],
        dropout=cfg["dropout"],
        target_only=cfg.get("target_only", False),
    )


BASELINES = [
    ("patchtst",     "pl16_dm64_lr5e-04",     "patchtst_best.pt",     _build_patchtst),
    ("itransformer", "dm256_el4_lr5e-04",     "itransformer_best.pt", _build_itransformer),
    ("timesnet",     "d64_el2_lr1e-03",       "timesnet_best.pt",     _build_timesnet),
    ("epideep",      "de128_eh64_lr2e-03",    "epideep_best.pt",      _build_epideep),
]


def _aggregate_seeds(per_seed_splits: dict) -> dict:
    """Mean ± std of WIS_avg across seeds for each split."""
    splits = list(next(iter(per_seed_splits.values())).keys())
    agg = {}
    for sp in splits:
        wis_per_h_all = np.array([per_seed_splits[s][sp]["wis_per_horizon"]
                                  for s in per_seed_splits])         # [n_seed, H]
        wis_avg_all = np.array([per_seed_splits[s][sp]["wis_avg"]
                                for s in per_seed_splits])
        cov50 = np.array([per_seed_splits[s][sp]["coverage_50"] for s in per_seed_splits])
        cov95 = np.array([per_seed_splits[s][sp]["coverage_95"] for s in per_seed_splits])
        n = per_seed_splits[next(iter(per_seed_splits))][sp]["n"]
        agg[sp] = {
            "n": int(n),
            "wis_per_horizon_mean": wis_per_h_all.mean(axis=0).tolist(),
            "wis_per_horizon_std": wis_per_h_all.std(axis=0, ddof=1).tolist(),
            "wis_avg_mean": float(wis_avg_all.mean()),
            "wis_avg_std": float(wis_avg_all.std(ddof=1)),
            "coverage_50_mean": float(cov50.mean()),
            "coverage_95_mean": float(cov95.mean()),
        }
    return agg


def run_baseline(df, norm, device: str, baseline: str, cfg_name: str,
                 ckpt_file: str, build_fn, n_samples: int) -> dict:
    cfg_root = _ROOT / "runs" / f"{baseline}_final" / cfg_name
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])
    cfg = json.loads((cfg_root / "seed42" / "results.json").read_text())["config"]
    lookback = cfg.get("seq_len", cfg.get("lookback", 104))
    pred_len = cfg["pred_len"]
    dropout_used = cfg.get("dropout", float("nan"))

    splits_data = {}
    for split_label, epi_min in [("val", None),
                                  ("test_full", None),
                                  ("test_strict", COVID_STRICT_START_EPIWEEK)]:
        split_name = "val" if split_label == "val" else "test"
        loader, n = _build_loader(
            df, split_name, lookback, pred_len, norm,
            epi_min=epi_min if split_label == "test_strict" else None,
        )
        splits_data[split_label] = {"loader": loader, "n": n}

    per_seed = {}
    for seed in SEEDS:
        ckpt = cfg_root / f"seed{seed}" / ckpt_file
        if not ckpt.exists():
            print(f"  [{baseline} seed={seed}] SKIP — {ckpt} missing")
            continue
        model = build_fn(cfg).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=True)

        seed_splits = {}
        for split_label, dat in splits_data.items():
            t0 = time.time()
            samples, y_true = _mc_inference(
                model, dat["loader"], n_samples,
                target_mean, target_std, device,
            )
            qf = _samples_to_quantiles(samples)
            s = _score_split(qf, y_true)
            elapsed = time.time() - t0
            seed_splits[split_label] = s
            print(f"  [{baseline} seed={seed:4d} {split_label:11s}] "
                  f"WIS_avg={s['wis_avg']:.4f}  cov50={s['coverage_50']:.3f}  "
                  f"cov95={s['coverage_95']:.3f}  ({elapsed:.1f}s)")
        per_seed[seed] = seed_splits

    agg = _aggregate_seeds(per_seed)
    return {
        "baseline": baseline,
        "cfg_name": cfg_name,
        "dropout_used": dropout_used,
        "mc_samples": n_samples,
        "per_seed": {str(s): per_seed[s] for s in per_seed},
        "aggregated": agg,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", nargs="+",
                    default=["patchtst", "itransformer", "timesnet", "epideep"],
                    choices=["patchtst", "itransformer", "timesnet", "epideep"])
    ap.add_argument("--device", default="cuda:1",
                    help="GPU device (default cuda:1; ensure Phase C is done)")
    ap.add_argument("--n-samples", type=int, default=100,
                    help="MC Dropout sample count (PLAN Q2: n=100 default)")
    args = ap.parse_args()

    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)

    runner_map = {b: (cfg_name, ckpt, build) for b, cfg_name, ckpt, build in BASELINES}

    for baseline in args.baselines:
        if baseline not in runner_map:
            print(f"[!] unknown baseline {baseline}, skipping")
            continue
        cfg_name, ckpt_file, build_fn = runner_map[baseline]
        print(f"\n=== {baseline.upper()} ({cfg_name}) — MC Dropout n={args.n_samples} ===")
        try:
            out = run_baseline(df, norm, args.device, baseline,
                               cfg_name, ckpt_file, build_fn, args.n_samples)
            out_path = OUT_ROOT / baseline / "wis_results.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2))
            print(f"  Saved: {out_path.relative_to(_ROOT)}")
        except Exception as e:
            print(f"  [ERROR] {baseline}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
