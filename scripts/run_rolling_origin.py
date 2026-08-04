"""ROLLING-ORIGIN driver -- CG side (pre-registered, result-blind, leakage-safe).

Per cutoff Y (pre-registered set in build_rolling_splits.py), HEADLINE-IDENTICAL protocol
(only the split/norm roll; every HP bit-identical to e1_final_train.py n3_d64):
  env pretrain (once/cutoff) -> per seed: HMM retrain -> Stage2(200ep/pat30) -> Stage3(10ep/pat0)
  -> native-APMD per-horizon regional Cov95 (the eval PROVEN in Stage-1).

Leakage-safe: env encoder AND HMM are re-fit on THIS cutoff's train (never the canonical
2001-2018 fit, which would leak future into pre-COVID cutoffs). Normalization = cutoff norm.

Native scoring: s_per_h is NEVER applied (raw APMD) -- the IV-F Scaled trap is absent.

Bug-guard: --regress runs the SAME parameterized eval on the CANONICAL split + headline
checkpoints and must reproduce Stage-1's stored 50-row CSV (max|Δ|<0.01). Only if that
passes do rolling numbers mean anything.

USAGE:
  python scripts/run_rolling_origin.py --regress                     # eval regression (NO training)
  python scripts/run_rolling_origin.py --cutoffs 2022 --seeds 42     # shakedown (1 CG cell)
  python scripts/run_rolling_origin.py                               # full CG side (7x5)
Resumable: existing env/HMM/stage3 checkpoints are reused, not recomputed.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import e1_final_eval as E                       # reuse forward/decompose/native-cov (Stage-1 proven)
import regime_shift_drivers as rsd              # _build_region_df (+ SPLIT_CSV/NORM_JSON globals)
import scripts.m1_7_env_pretrain as env_pretrain_module
from scripts.m1_7_train import train as stage2_train
from scripts.m1_8_stage3_train import stage3_train
from scripts.m1_4_phase_dynamics_search import RAW_COLS_V3, featurize_raw, augment_features
from src.models.gaussian_hmm import GaussianHMM
from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import load_norm_params, MultiHorizonDataset

ROLL_ROOT = _ROOT / "runs" / "rolling_origin"
CUTOFFS = [2015, 2016, 2017, 2018, 2022, 2023, 2024]
SEEDS = [42, 123, 456, 789, 1024]
REGIONS = [f"hhs{i}" for i in range(1, 11)]
HORIZONS = [1, 2, 3, 4]

# HP bit-identical to headline e1_final_train.py n3_d64
STAGE2_EPOCHS, STAGE3_EPOCHS, STAGE3_PATIENCE, BATCH = 200, 10, 0, 32
ENV_EPOCHS = 100
_HP = dict(gate=1e-3, backbone=1e-4, other=1e-4, hmm=1e-6, embed=1e-6, env=1e-7, lookback=104)

# canonical (for --regress) -- headline paths
CANON_CSV = _ROOT / "data/processed/ili_env_weekly_split.csv"
CANON_NORM = _ROOT / "data/processed/normalization_params.json"
CANON_HMM_TPL = _ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}"
CANON_S3_TPL = _ROOT / "runs/m1_8_stage3_train/e1_final_n3_d64_s{seed}_stage3/best.pt"
CANON_TEST_FIRST = 202240
STORED_CANON = _ROOT / "runs/e1_final/n3_d64_regional_perhorizon_raw.csv"


def build_cfg(seed, cut_csv, cut_norm) -> CGMambaConfig:
    """TRAINING cfg -- headline HP incl. lookback=104 (e1_final_train.py)."""
    return dataclasses.replace(
        CGMambaConfig(), seed=seed, n_layers=3, d_model=64, dropout=0.0,
        lookback=_HP["lookback"],
        stage2_gate_lr=_HP["gate"], stage2_backbone_lr=_HP["backbone"],
        stage3_other_lr=_HP["other"], stage3_hmm_lr=_HP["hmm"],
        stage3_state_embed_lr=_HP["embed"], stage3_env_lr=_HP["env"],
        data_csv=Path(cut_csv), norm_json=Path(cut_norm),
    )


def build_eval_cfg(seed, cut_csv, cut_norm) -> CGMambaConfig:
    """EVAL cfg -- MIRRORS e1_final_eval.load_final_model EXACTLY (does NOT override
    lookback, so it uses CGMambaConfig default lookback=156). The headline regional eval
    used 156 at scoring time even though training used 104; to reproduce the headline
    result (Stage-1 proved this), rolling eval must match. Verified by --regress."""
    return dataclasses.replace(
        CGMambaConfig(), seed=seed, n_layers=3, d_model=64,
        data_csv=Path(cut_csv), norm_json=Path(cut_norm),
    )


# ---------------------------------------------------------------- train stages
def pretrain_env_cutoff(Y, cut_csv, cut_norm):
    dst = ROLL_ROOT / f"cut{Y}" / "env_encoder_d64.pt"
    if dst.exists():
        return dst
    cfg = dataclasses.replace(CGMambaConfig(), d_model=64,
                              data_csv=Path(cut_csv), norm_json=Path(cut_norm))
    env_pretrain_module.pretrain_env(cfg, SimpleNamespace(smoke=False, epochs=ENV_EPOCHS, log_every=10))
    src = _ROOT / "runs/m1_7_env_pretrain/env_encoder.pt"
    if not src.exists():
        raise RuntimeError(f"env pretrain produced no ckpt: {src}")
    shutil.move(str(src), str(dst))
    print(f"  [env] cut{Y} -> {dst.relative_to(_ROOT)}")
    return dst


def retrain_hmm_cutoff(Y, seed, cut_csv, cut_norm):
    """M1.4 protocol HMM re-fit on cutoff train, with CUTOFF norm (fixes the hardcode)."""
    out_dir = ROLL_ROOT / f"cut{Y}" / "hmm" / f"seed{seed}"
    if (out_dir / "hmm_params.npz").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    norm = load_norm_params(Path(cut_norm))                 # <-- cutoff norm, NOT canonical
    df = pd.read_csv(cut_csv)
    train_df = df[df["split"] == "train"].sort_values("epiweek").reset_index(drop=True)
    x_raw = featurize_raw(train_df, norm, RAW_COLS_V3)
    x_aug = augment_features(x_raw)
    best_ll, best = -float("inf"), None
    for init in range(seed, seed + 5):
        hmm = GaussianHMM(n_states=3, n_features=6, covariance_type="full",
                          reg_covar=5e-3, n_iter=100, tol=1e-4, seed=init)
        hmm.fit(x_aug)
        ll = hmm.log_likelihood(x_aug)
        if ll > best_ll:
            best_ll, best = ll, hmm
    np.savez(out_dir / "hmm_params.npz", A=best.A, pi=best.pi, means=best.means,
             covars=best.covars, V=6, K=3, reg_covar=5e-3, seed=seed,
             covariance_type="full", n_iter_run=getattr(best, "n_iter_run", 100),
             final_ll=float(best_ll))
    np.save(out_dir / "viterbi_path.npy", best.viterbi(x_aug))
    np.save(out_dir / "gamma.npy", best.posteriors(x_aug))
    (out_dir / "diagnostics.json").write_text(json.dumps({
        "cutoff": Y, "seed": seed, "V_raw": 3, "V_aug": 6, "K": 3, "reg_covar": 5e-3,
        "n_init": 5, "final_ll": float(best_ll), "n_train_obs": int(len(train_df)),
        "norm_used": str(Path(cut_norm).relative_to(_ROOT)),
        "note": "rolling-origin STRICT: HMM re-fit on cutoff train with cutoff norm",
    }, indent=2))
    return out_dir


def train_cg_cell(Y, seed, cut_csv, cut_norm, env_ckpt, hmm_dir, device):
    rn = f"roll{Y}_n3_d64_s{seed}"
    s3_best = _ROOT / "runs/m1_8_stage3_train" / f"{rn}_stage3" / "best.pt"
    if s3_best.exists():
        print(f"  [cg] cut{Y} s{seed}: stage3 exists, reuse")
        return s3_best
    cfg = build_cfg(seed, cut_csv, cut_norm)
    t0 = time.time()
    stage2_train(cfg, SimpleNamespace(
        smoke=False, epochs=STAGE2_EPOCHS, batch_size=BATCH,
        hmm_dir=str(hmm_dir), env_encoder_ckpt=str(env_ckpt),
        wandb_mode="disabled", run_name=rn))
    s2_dir = _ROOT / "runs/m1_7_train" / rn
    if not (s2_dir / "best.pt").exists():
        raise RuntimeError(f"stage2 best.pt missing: {s2_dir}")
    stage3_train(cfg, SimpleNamespace(
        smoke=False, epochs=STAGE3_EPOCHS, patience=STAGE3_PATIENCE, batch_size=BATCH,
        stage2_dir=str(s2_dir), hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(env_ckpt), run_name=f"{rn}_stage3"))
    if not s3_best.exists():
        raise RuntimeError(f"stage3 best.pt missing: {s3_best}")
    print(f"  [cg] cut{Y} s{seed}: trained ({time.time()-t0:.0f}s)")
    return s3_best


# ---------------------------------------------------------------- native eval
def eval_regional(s3_best, hmm_dir, cut_csv, cut_norm, test_first, seed, region, device):
    """Native-APMD per-horizon regional Cov95/WIS (identical logic to Stage-1).

    Uses build_eval_cfg (mirrors load_final_model, lookback=156) -- NOT the training cfg."""
    cfg = build_eval_cfg(seed, cut_csv, cut_norm)
    model = CGForecaster(cfg)
    hmm = load_fitted_hmm(Path(hmm_dir))
    model.prepare_for_stage2(hmm)
    ck = torch.load(Path(s3_best), map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck.get("state_dict", ck)), strict=False)
    model.eval().to(device)

    norm = load_norm_params(Path(cut_norm))
    tmean = float(norm["ili_weighted_pct"]["mean"])
    tstd = float(norm["ili_weighted_pct"]["std"])
    mu_k = hmm.means[:, 0].astype(np.float64)
    s2_k = hmm.covars[:, 0, 0].astype(np.float64)

    old_split, old_norm = rsd.SPLIT_CSV, rsd.NORM_JSON     # cutoff split labels for regional df
    rsd.SPLIT_CSV, rsd.NORM_JSON = Path(cut_csv), Path(cut_norm)
    try:
        df_reg = rsd._build_region_df(region)
    finally:
        rsd.SPLIT_CSV, rsd.NORM_JSON = old_split, old_norm

    ds = MultiHorizonDataset(df_reg, split="test", lookback=cfg.lookback,
                             horizons=tuple(cfg.horizons), norm=norm)
    df_pred = E._forward_dataset(model, ds, device)
    df_pred = E._decompose_apmd(df_pred, mu_k, s2_k, tmean, tstd)
    df_pred = df_pred[df_pred.target_ep >= test_first].reset_index(drop=True)
    del model, hmm
    gc.collect(); torch.cuda.empty_cache()

    rec = {}
    for h in HORIZONS:
        d = df_pred[df_pred.horizon == h]
        cov, wis = E.eval_cov95_wis(d.mu.to_numpy(), d.s2_total.to_numpy(), d.y_true.to_numpy())
        rec[f"tS_cov95_h{h}"] = float(cov)
        rec[f"tS_wis_h{h}"] = float(wis)
    rec["n_test_origins"] = int(df_pred.target_ep.nunique())
    return rec


def regression_canonical(device):
    """Parameterized eval on canonical split + headline ckpts -> must == Stage-1 stored CSV."""
    print("[regress] parameterized native eval on CANONICAL (no training)...")
    rows = []
    for seed in SEEDS:
        hmm_dir = Path(str(CANON_HMM_TPL).format(seed=seed))
        s3_best = Path(str(CANON_S3_TPL).format(seed=seed))
        for region in REGIONS:
            rec = eval_regional(s3_best, hmm_dir, CANON_CSV, CANON_NORM,
                                CANON_TEST_FIRST, seed, region, device)
            rows.append({"baseline": "cg_mamba", "seed": seed, "region": region, **rec})
    df = pd.DataFrame(rows)
    stored = pd.read_csv(STORED_CANON)
    cols = [f"tS_cov95_h{h}" for h in HORIZONS]
    m = df[["seed", "region"] + cols].merge(stored[["seed", "region"] + cols],
                                            on=["seed", "region"], suffixes=("_new", "_old"))
    max_diff = max(float((m[f"{c}_new"] - m[f"{c}_old"]).abs().max()) for c in cols)
    ok = (len(m) == len(stored)) and (max_diff < 0.01)
    print(f"[regress] n_rows {len(m)}/{len(stored)}  max|Δ| {max_diff:.4f}  -> "
          f"{'PASS: parameterized eval is bug-free, safe for rolling' if ok else 'FAIL: HALT, do NOT trust rolling'}")
    return ok


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regress", action="store_true", help="canonical eval regression only (no training)")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=CUTOFFS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.regress:
        return 0 if regression_canonical(args.device) else 2

    ROLL_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for Y in args.cutoffs:
        cut_dir = ROLL_ROOT / f"cut{Y}"
        cut_csv = cut_dir / "ili_env_weekly_split.csv"
        cut_norm = cut_dir / "normalization_params.json"
        if not cut_csv.exists():
            raise RuntimeError(f"cut{Y} split missing -- run build_rolling_splits.py first")
        test_first = Y * 100 + 40
        print(f"\n{'='*64}\n[rolling] cut{Y}  test={Y}-{Y+1} (>= {test_first})\n{'='*64}")
        env_ckpt = pretrain_env_cutoff(Y, cut_csv, cut_norm)
        for seed in args.seeds:
            hmm_dir = retrain_hmm_cutoff(Y, seed, cut_csv, cut_norm)
            s3_best = train_cg_cell(Y, seed, cut_csv, cut_norm, env_ckpt, hmm_dir, args.device)
            for region in REGIONS:
                rec = eval_regional(s3_best, hmm_dir, cut_csv, cut_norm, test_first,
                                    seed, region, args.device)
                rows.append({"cutoff": Y, "test_season": f"{Y}-{Y+1}", "seed": seed,
                             "region": region, **rec})
            cov = np.mean([rows[-1][f"tS_cov95_h{h}"] for h in HORIZONS])
            print(f"  [eval] cut{Y} s{seed}: last-region cov_avg~{cov:.3f} "
                  f"({len(REGIONS)} regions done)")
        pd.DataFrame(rows).to_csv(ROLL_ROOT / "cg_regional_results.csv", index=False)

    out = ROLL_ROOT / "cg_regional_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[rolling] CG side done: {len(rows)} rows -> {out.relative_to(_ROOT)}")
    print("[rolling] NEXT: baseline side + apply pre-registered verdict table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
