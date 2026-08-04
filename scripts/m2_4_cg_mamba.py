"""M2.4 Data efficiency — CG-Mamba × 7 train period variants × 5 seeds.

CG-Mamba multi-stage retraining per variant:
  - Stage 1 (HMM): SHARED across variants (HMM trained on full data — paper note
    as simplification; HMM provides phase structure which is stable across
    train sizes; Stage 2/3 retraining captures the data efficiency effect)
  - Stage 2 (SSM): RETRAIN per (variant, seed) with filtered CSV
  - Stage 3 (Joint): RETRAIN per (variant, seed) on top of new Stage 2

35 cells × (Stage 2 ~7 min + Stage 3 ~1.4 min) ≈ ~5h GPU.

Uses M2.1 top1 cell HP (PLAN §16 J.4).

Output: runs/m2_4_data_efficiency/cg_mamba/seasons_{N}/seed{s}/
  ├── stage2_dir → symlink to runs/m1_7_train/m2_4_cg_mamba_{N}_s{seed}/
  └── stage3_dir → symlink to runs/m1_8_stage3_train/m2_4_cg_mamba_{N}_s{seed}/
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.m1_7_train import train as stage2_train
from scripts.m1_8_stage3_train import stage3_train
from src.utils.config import CGMambaConfig

OUT_ROOT = _ROOT / "runs" / "m2_4_data_efficiency"
FILTERED_CSV_DIR = OUT_ROOT / "_filtered_csvs"

TRAIN_PERIODS = [
    ("17_seasons_full", 200240, 17),
    ("13_seasons",      200540, 13),
    ("10_seasons",      200840, 10),
    ( "7_seasons",      201140,  7),
    ( "5_seasons",      201340,  5),
    ( "4_seasons",      201440,  4),
    ( "3_seasons",      201540,  3),
]
SEEDS = (42, 123, 456, 789, 1024)

# M2.1 top1 cell HP (PLAN §16 J.4)
CG_TOP1_HP = {
    "gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104,
    "hmm_lr_ratio": 0.01, "state_embed_lr_ratio": 0.01, "env_lr_ratio": 0.001,
}
OTHER_LR_BASE = 1e-4
HMM_DIR_TEMPLATE = (_ROOT / "runs" / "m1_4_phase_dynamics_main"
                    / "V_raw3_regcov5e-03_K3_seed{seed}")
ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"


def generate_filtered_csvs():
    """STRICT CSV filter (v2) — drop earlier train rows entirely.

    Shared with m2_4_nn_baselines.py. Consistent with SARIMA strict protocol.
    """
    FILTERED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    src = _ROOT / "data/processed/ili_env_weekly_split.csv"
    df = pd.read_csv(src)
    csv_paths = {}
    for label, train_min, n in TRAIN_PERIODS:
        out_p = FILTERED_CSV_DIR / f"ili_env_weekly_split_{label}.csv"
        if not out_p.exists():
            mask_drop = (df["split"] == "train") & (df["epiweek"] < train_min)
            df_f = df[~mask_drop].copy().sort_values("epiweek").reset_index(drop=True)
            df_f.to_csv(out_p, index=False)
        csv_paths[label] = out_p
    return csv_paths


def retrain_hmm_for_variant(label, seed, csv_path):
    """Re-train Phase Dynamics HMM on filtered CSV (M2.4 STRICT).

    Uses M1.4 protocol: V_raw=3, V_aug=6, K=3, reg_covar=5e-3, n_init=5.
    Returns dir matching M1.4 layout (hmm_params.npz, viterbi_path.npy,
    gamma.npy, diagnostics.json) consumable by load_fitted_hmm.
    """
    from src.models.gaussian_hmm import GaussianHMM
    from scripts.m1_4_phase_dynamics_search import (
        RAW_COLS_V3, featurize_raw, augment_features,
    )
    from src.data.loader import load_norm_params

    out_dir = OUT_ROOT / "cg_mamba_hmm" / f"seasons_{label}" / f"seed{seed}"
    if (out_dir / "hmm_params.npz").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    norm = load_norm_params(_ROOT / "data/processed/normalization_params.json")
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"].sort_values("epiweek").reset_index(drop=True)
    x_raw = featurize_raw(train_df, norm, RAW_COLS_V3)                    # [L, V_raw=3]
    x_aug = augment_features(x_raw)                                       # [L-1, V_aug=6]

    # n_init=5 inits, pick best LL (matches M1.4 protocol)
    best_ll = -float("inf"); best_hmm = None; best_init = -1
    for init_seed in range(seed, seed + 5):
        hmm = GaussianHMM(n_states=3, n_features=6, covariance_type="full",
                          reg_covar=5e-3, n_iter=100, tol=1e-4, seed=init_seed)
        hmm.fit(x_aug)
        ll = hmm.log_likelihood(x_aug)
        if ll > best_ll:
            best_ll = ll; best_hmm = hmm; best_init = init_seed

    viterbi = best_hmm.viterbi(x_aug)
    gamma = best_hmm.posteriors(x_aug)

    np.savez(out_dir / "hmm_params.npz",
             A=best_hmm.A, pi=best_hmm.pi,
             means=best_hmm.means, covars=best_hmm.covars,
             V=6, K=3, reg_covar=5e-3, seed=seed,
             covariance_type="full",
             n_iter_run=getattr(best_hmm, "n_iter_run", 100),
             final_ll=float(best_ll))
    np.save(out_dir / "viterbi_path.npy", viterbi)
    np.save(out_dir / "gamma.npy", gamma)
    (out_dir / "diagnostics.json").write_text(json.dumps({
        "V_raw": 3, "V_aug": 6, "K": 3, "seed": seed, "reg_covar": 5e-3,
        "n_init": 5, "best_init_seed": best_init, "final_ll": float(best_ll),
        "n_train_obs": len(train_df), "n_aug_obs": len(x_aug),
        "csv_used": str(csv_path.relative_to(_ROOT)),
        "note": "M2.4 STRICT — HMM retrained per (variant, seed)",
    }, indent=2))
    return out_dir


def run_one(label, seed, device, csv_path):
    """Run CG-Mamba M2.4 STRICT:
       1. HMM retrain on filtered CSV (per variant, seed)
       2. Stage 2 retrain
       3. Stage 3 retrain
    """
    hp = CG_TOP1_HP
    cfg = dataclasses.replace(
        CGMambaConfig(),
        seed=seed, dropout=0.0,
        lookback=hp["lookback"],
        stage2_gate_lr=hp["gate_lr"],
        stage2_backbone_lr=hp["backbone_lr"],
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * hp["hmm_lr_ratio"],
        stage3_state_embed_lr=OTHER_LR_BASE * hp["state_embed_lr_ratio"],
        stage3_env_lr=OTHER_LR_BASE * hp["env_lr_ratio"],
        data_csv=csv_path,
    )

    # ── Stage 1: HMM retrain per variant (STRICT) ──
    t_hmm = time.time()
    hmm_dir = retrain_hmm_for_variant(label, seed, csv_path)
    hmm_elapsed = time.time() - t_hmm
    print(f"  HMM retrained: {hmm_dir.relative_to(_ROOT)} ({hmm_elapsed:.1f}s)")

    s2_name = f"m2_4_cg_mamba_{label}_s{seed}_stage2"
    s3_name = f"m2_4_cg_mamba_{label}_s{seed}_stage3"

    # Stage 2
    t_s2 = time.time()
    s2_args = SimpleNamespace(
        smoke=False, epochs=None, batch_size=32,
        hmm_dir=str(hmm_dir), env_encoder_ckpt=str(ENV_CKPT),
        wandb_mode="disabled", run_name=s2_name,
    )
    s2_final = stage2_train(cfg, s2_args)
    s2_elapsed = time.time() - t_s2
    s2_dir = _ROOT / "runs" / "m1_7_train" / s2_name
    s2_best = s2_dir / "best.pt"
    if not s2_best.exists():
        raise RuntimeError(f"Stage 2 best.pt missing: {s2_best}")

    # Stage 3
    t_s3 = time.time()
    s3_args = SimpleNamespace(
        smoke=False, epochs=30, patience=10, batch_size=32,
        stage2_dir=str(s2_dir), hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(ENV_CKPT), run_name=s3_name,
    )
    s3_final = stage3_train(cfg, s3_args)
    s3_elapsed = time.time() - t_s3
    s3_dir = _ROOT / "runs" / "m1_8_stage3_train" / s3_name
    s3_best = s3_dir / "best.pt"
    if not s3_best.exists():
        raise RuntimeError(f"Stage 3 best.pt missing: {s3_best}")

    # Symlink to M2.4 location
    m24_dir = OUT_ROOT / "cg_mamba" / f"seasons_{label}" / f"seed{seed}"
    m24_dir.mkdir(parents=True, exist_ok=True)
    (m24_dir / "manifest.json").write_text(json.dumps({
        "label": label, "seed": seed,
        "hmm_dir": str(hmm_dir.relative_to(_ROOT)),
        "stage2_dir": str(s2_dir.relative_to(_ROOT)),
        "stage3_dir": str(s3_dir.relative_to(_ROOT)),
        "stage2_best": str(s2_best.relative_to(_ROOT)),
        "stage3_best": str(s3_best.relative_to(_ROOT)),
        "stage2_val_total": s2_final.get("best_val_total", float("nan")),
        "stage3_val_total": s3_final.get("best_val_total", float("nan")),
        "stage3_test_mse": s3_final.get("test_mse", float("nan")),
        "hmm_elapsed_sec": hmm_elapsed,
        "stage2_elapsed_sec": s2_elapsed,
        "stage3_elapsed_sec": s3_elapsed,
        "elapsed_sec": hmm_elapsed + s2_elapsed + s3_elapsed,
        "csv_used": str(csv_path.relative_to(_ROOT)),
        "note": "M2.4 STRICT — CSV drops earlier rows; HMM retrained per (variant, seed); env_encoder shared (pretrained on full env data, deemed acceptable as env features stable across years)",
    }, indent=2))

    return {"label": label, "seed": seed,
            "elapsed_sec": hmm_elapsed + s2_elapsed + s3_elapsed,
            "stage3_test_mse": s3_final.get("test_mse"),
            "stage3_val_total": s3_final.get("best_val_total"),
            "hmm_elapsed_sec": hmm_elapsed,
            "ok": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--variants", nargs="+", default=[lp[0] for lp in TRAIN_PERIODS])
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells with existing manifest.json")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Generating filtered CSVs...")
    csv_paths = generate_filtered_csvs()

    plan = [(label, seed) for label in args.variants for seed in args.seeds]
    print(f"Plan: {len(plan)} cells (CG-Mamba × {len(args.variants)} variants × {len(args.seeds)} seeds)")

    summary = []
    total_t0 = time.time()
    for i, (label, seed) in enumerate(plan, 1):
        m24_manifest = OUT_ROOT / "cg_mamba" / f"seasons_{label}" / f"seed{seed}" / "manifest.json"
        if args.resume and m24_manifest.exists():
            print(f"[{i}/{len(plan)}] SKIP {label} s={seed} (resume)")
            continue
        print(f"\n[{i}/{len(plan)}] CG-Mamba {label} s={seed}")
        try:
            r = run_one(label, seed, args.device, csv_paths[label])
            summary.append(r)
            print(f"  ✓ ok, elapsed={r['elapsed_sec']:.1f}s  "
                  f"val_total={r.get('stage3_val_total', '?')}  "
                  f"test_mse={r.get('stage3_test_mse', '?')}")
        except Exception as e:
            import traceback
            summary.append({"label": label, "seed": seed, "ok": False, "error": str(e)})
            print(f"  ✗ FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()

    total = time.time() - total_t0
    out_path = OUT_ROOT / "cg_mamba_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path.relative_to(_ROOT)}")
    print(f"Total: {total/60:.1f} min  ({sum(1 for s in summary if s.get('ok'))} OK / "
          f"{sum(1 for s in summary if not s.get('ok'))} FAIL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
