"""scripts/e1_hpo.py — E1 design-split HPO launcher (γ LOCKED)
================================================================================
9 configs (K=3 × n_layers∈{2,3,4} × d_model∈{32,64,128}) × 5 seeds = 45 runs

LOCKED γ (수정 금지):
  γ.1 — design-train: ep≤201539 ∩ orig 'train' (712 rows). design-val: 201540-201839 (156 rows)
  γ.2 — grid 45 runs (K=3 고정, κ_min=0.9459 통과)
  γ.3 — calibration-제약 WIS selection, Cov95 5-seed pooled, fallback [0.85,0.99] → FAIL
  γ.4 — design-train HMM (runs/m1_4_design_split/V_raw3_regcov5e-03_K3_seed42)
  γ.5 — design-train scaler (normalization_params_design_train.json) + selection 분산 = raw HMM only
  γ.6 — held-out off-season guard (post-E1, 본 launcher 무관)

평가 (design-val 위, MultiHorizonDataset windows):
  - 별도 inference pass: model(x, env, return_intermediates=True) → gamma_all
  - σ²_total = σ²_within + σ²_between (APMD, raw HMM emission, s_h 없음)
  - target_epiweek ≤ 201839 belt+suspenders filter (loader leak 차단)
  - 단일 seed 기대 n ≈ 612 (153 windows × 4 horizons). 624/300/75 시 abort signal
  - 5-seed pooled n ≈ 3060 (612 × 5). pair-level pooling (NOT seed-Cov95 평균)

Selection:
  config* = argmin WIS s.t. pooled Cov95 ∈ [0.90, 0.96]
  fallback: → [0.85, 0.99] → 둘 다 0 → FAIL (floor-full-negative)

증분 저장: run 마다 즉시 runs/e1_hpo/run_{config_id}_seed{s}/metrics.json
메모리 정리: 각 run 후 del model/hmm/ckpt + torch.cuda.empty_cache + gc.collect

CLI:
  python scripts/e1_hpo.py --smoke        # 1 config × 1 seed × 5 epochs (~5분, smoke check)
  python scripts/e1_hpo.py                # full 45 runs (~4-7h GPU)
"""
from __future__ import annotations
import argparse
import dataclasses
import gc
import json
import sys
import time
from argparse import Namespace
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as sp_norm

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import load_norm_params, load_dataset_csv, MultiHorizonDataset
from torch.utils.data import DataLoader

import scripts.m1_7_train as m1_7        # in-process Stage 2 train()
import scripts.m1_8_stage3_train as m1_8  # in-process Stage 3 stage3_train()

# ============================================================================
# LOCKED PATHS + γ CONSTANTS
# ============================================================================
DESIGN_CSV       = _ROOT / "data/processed/ili_env_weekly_split_design.csv"
DESIGN_NORM_JSON = _ROOT / "data/processed/normalization_params_design_train.json"
DESIGN_HMM_DIR   = _ROOT / "runs/m1_4_design_split/V_raw3_regcov5e-03_K3_seed42"
# γ.5 채널 4 — env encoder design-train pretrain (d_model 별 3개, EnvModule.encoder L2 가 d_model 의존)
DESIGN_ENV_CKPT_TPL = _ROOT / "runs/m1_7_env_pretrain_design/env_encoder_d{d_model}.pt"

# γ.5 — paper winner HP override (ablation_retrain.py:65-72 그대로)
CG_TOP1_HP = {
    "gate_lr":               1e-3,
    "backbone_lr":           1e-4,
    "lookback":              104,
    "hmm_lr_ratio":          0.01,
    "state_embed_lr_ratio":  0.01,
    "env_lr_ratio":          0.001,
}
OTHER_LR_BASE = 1e-4

# γ.6 — Stage 2 + Stage 3 protocol (paper m1_9_hpo_phase2 일치)
STAGE2_EPOCHS = 200
STAGE3_EPOCHS = 30
STAGE3_PATIENCE = 10

DESIGN_VAL_END_EPIWEEK = 201839                      # belt+suspenders filter target_ep ≤ 이값

PC2A_FLUSIGHT_23 = np.array([
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 0.975, 0.99
])
_LO_IDX = int(np.where(np.isclose(PC2A_FLUSIGHT_23, 0.025))[0][0])
_HI_IDX = int(np.where(np.isclose(PC2A_FLUSIGHT_23, 0.975))[0][0])

# Grid (γ.2: K=3 고정)
GRID_N_LAYERS = [2, 3, 4]
GRID_D_MODEL  = [32, 64, 128]
SEEDS         = [42, 123, 456, 789, 1024]
N_EPOCHS      = 50                                   # cfg.stage2_n_epochs default
BATCH_SIZE    = 32

# Selection bar
CAL_BAND_PRIMARY  = (0.90, 0.96)
CAL_BAND_FALLBACK = (0.85, 0.99)

# Smoke expected window count
SMOKE_EXPECTED_N = 612                               # 153 windows × 4 horizons (single seed)
SMOKE_TOLERANCE  = 12                                # ±2% tolerance band [600, 624]
ABORT_HARD_LIMITS = {612 + 12: "boundary leak (target_ep > 201839)",
                      75 * 4:    "원 val 75-row leak",
                      75:        "원 val 75-row leak"}

OUT_DIR = _ROOT / "runs/e1_hpo"


# ============================================================================
# Config / args builders (ablation_retrain.py:239-267 pattern)
# ============================================================================
def build_cfg(n_layers: int, d_model: int, seed: int) -> CGMambaConfig:
    """γ.5 — ablation_retrain.build_frozen_hpo_cfg 패턴 적용 (CG_TOP1_HP 9개 HP override).
    paper baseline 과 *line-by-line 같은 HP environment* 보장. (n_layers, d_model) 만 새로 선택."""
    hp = CG_TOP1_HP
    return dataclasses.replace(
        CGMambaConfig(),
        seed=seed,
        n_layers=n_layers,
        d_model=d_model,
        data_csv=DESIGN_CSV,
        norm_json=DESIGN_NORM_JSON,
        # CG_TOP1_HP override (paper winner 그대로)
        dropout=0.0,
        lookback=hp["lookback"],                                              # 104 (default 156 아님)
        stage2_gate_lr=hp["gate_lr"],                                          # 1e-3
        stage2_backbone_lr=hp["backbone_lr"],                                  # 1e-4 (default 5e-5 아님)
        stage3_other_lr=OTHER_LR_BASE,                                         # 1e-4
        stage3_hmm_lr=OTHER_LR_BASE * hp["hmm_lr_ratio"],                      # 1e-6
        stage3_state_embed_lr=OTHER_LR_BASE * hp["state_embed_lr_ratio"],      # 1e-6
        stage3_env_lr=OTHER_LR_BASE * hp["env_lr_ratio"],                      # 1e-7
    )


def build_stage2_args(run_name: str, d_model: int, smoke: bool = False,
                       epochs: int | None = None) -> Namespace:
    """Stage 2 (m1_7.train) args. γ.5 채널 4: design-train env per d_model."""
    env_ckpt = Path(str(DESIGN_ENV_CKPT_TPL).format(d_model=d_model))
    assert env_ckpt.exists(), f"missing design-train env ckpt: {env_ckpt}"
    return Namespace(
        smoke=smoke,
        epochs=(epochs if epochs is not None else (5 if smoke else STAGE2_EPOCHS)),
        batch_size=BATCH_SIZE,
        hmm_dir=str(DESIGN_HMM_DIR),
        env_encoder_ckpt=str(env_ckpt),
        wandb_mode="disabled",
        run_name=run_name,
    )


def build_stage3_args(run_name: str, stage2_dir: Path, d_model: int,
                       smoke: bool = False, epochs: int | None = None,
                       patience: int | None = None) -> Namespace:
    """Stage 3 (m1_8.stage3_train) args. paper m1_9_hpo_phase2 protocol 일치."""
    env_ckpt = Path(str(DESIGN_ENV_CKPT_TPL).format(d_model=d_model))
    return Namespace(
        smoke=smoke,
        epochs=(epochs if epochs is not None else (3 if smoke else STAGE3_EPOCHS)),
        patience=(patience if patience is not None else STAGE3_PATIENCE),
        batch_size=BATCH_SIZE,
        stage2_dir=str(stage2_dir),
        hmm_dir=str(DESIGN_HMM_DIR),
        env_encoder_ckpt=str(env_ckpt),
        run_name=run_name,
    )


# Back-compat alias (legacy callsite, will be removed if no longer used)
def build_args(run_name: str, d_model: int, smoke: bool = False,
                epochs: int | None = None) -> Namespace:
    return build_stage2_args(run_name, d_model, smoke=smoke, epochs=epochs)


# ============================================================================
# Inference + APMD decomposition (raw HMM emission, s_h 없음 — γ.5)
# ============================================================================
def design_val_inference(cfg: CGMambaConfig, ckpt_path: Path,
                          hmm_dir: Path, device: str) -> dict:
    """학습 후 design-val 윈도우 위 model forward (return_intermediates=True) →
       per-(origin, horizon) mu (raw wILI) + σ²_total (raw wILI²) + target_ep.

    LOCKED γ.5: σ²_total = σ²_within + σ²_between, raw HMM emission,
       *post-hoc s_h 없음* (selection 단계 단일 lock).
    """
    df = load_dataset_csv(cfg.data_csv)
    norm = load_norm_params(cfg.norm_json)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std  = float(norm["ili_weighted_pct"]["std"])

    # design-val dataset (split='val' on design-cut CSV = 201540-201839)
    val_ds = MultiHorizonDataset(df, split="val", lookback=cfg.lookback,
                                   horizons=tuple(cfg.horizons), norm=norm)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load model
    hmm = load_fitted_hmm(hmm_dir)
    mu_k_zili = hmm.means[:, 0].astype(np.float64)          # [K] (z-ili emission mean)
    s2_k_zili = hmm.covars[:, 0, 0].astype(np.float64)      # [K] (z-ili diag variance)

    model = CGForecaster(cfg)
    model.prepare_for_stage2(hmm)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    rows = []
    H_list = list(cfg.horizons)
    H = len(H_list)
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            y_z = batch["y"].cpu().numpy()                  # [B, H] z-scored
            target_eps = batch["target_epiweeks"]           # list[H] of list[B] OR torch
            # Convert target_eps to numpy [B, H]
            if isinstance(target_eps, torch.Tensor):
                tep = target_eps.cpu().numpy()
            elif isinstance(target_eps, (list, tuple)):
                arr = []
                for h_eps in target_eps:
                    arr.append(h_eps.cpu().numpy() if isinstance(h_eps, torch.Tensor) else np.asarray(h_eps))
                tep = np.stack(arr, axis=-1) if arr[0].ndim == 1 else np.array(arr).T
            else:
                tep = np.array(target_eps)
            pred, inter = model(x, env, return_intermediates=True)
            mu_z = pred.cpu().numpy()                        # [B, H]
            gamma_all = inter["gamma_all"].cpu().numpy()     # [B, max_h, K]
            B = mu_z.shape[0]
            for b in range(B):
                for hi, h in enumerate(H_list):
                    g_h = gamma_all[b, h - 1, :]                 # [K]
                    mu_hmm_z = float((g_h * mu_k_zili).sum())
                    sw_z = float((g_h * s2_k_zili).sum())
                    sb_z = float((g_h * (mu_k_zili - mu_hmm_z) ** 2).sum())
                    st_z = max(sw_z + sb_z, 1e-12)
                    # Denormalize (raw wILI units)
                    mu_raw = float(mu_z[b, hi] * target_std + target_mean)
                    y_raw  = float(y_z[b, hi] * target_std + target_mean)
                    st_raw = st_z * (target_std ** 2)
                    rows.append(dict(
                        target_ep=int(tep[b, hi]),
                        horizon=int(h),
                        mu=mu_raw,
                        s2_total=st_raw,
                        y_true=y_raw,
                    ))

    # === belt+suspenders filter: target_ep ≤ 201839 ===
    df_pred = pd.DataFrame(rows)
    n_before = len(df_pred)
    df_pred = df_pred[df_pred.target_ep <= DESIGN_VAL_END_EPIWEEK].reset_index(drop=True)
    n_after = len(df_pred)
    n_dropped = n_before - n_after

    # Cleanup
    del model, hmm, ck, sd
    torch.cuda.empty_cache()
    gc.collect()

    return dict(predictions=df_pred, n_before_filter=n_before,
                n_after_filter=n_after, n_dropped=n_dropped)


# ============================================================================
# Cov95 + WIS (raw HMM 분산, s_h 없음)
# ============================================================================
def cov95_wis(df_pred: pd.DataFrame) -> tuple[float, float]:
    """phase-anchored raw PI = μ + Φ⁻¹(τ) · √s²_total. Returns (cov95, wis_mean).

    T5 (2026-06-21): redirect dataframe wrapper to wis_standard single source of truth.
    """
    from src.eval.wis_standard import cov95_wis_from_gaussian
    return cov95_wis_from_gaussian(df_pred.mu.to_numpy(),
                                     df_pred.s2_total.to_numpy(),
                                     df_pred.y_true.to_numpy())


# ============================================================================
# Per-run orchestration
# ============================================================================
def run_one(n_layers: int, d_model: int, seed: int, smoke: bool,
             device: str) -> dict:
    """γ.6 — Stage 2 + Stage 3 protocol (paper m1_9_hpo_phase2 일치).
       Selection criterion = stage3 best_val_total (= MSE + 0.3·MASE) ascending."""
    config_id = f"n{n_layers}_d{d_model}"
    run_name = f"e1_{config_id}_s{seed}"
    stage3_run_name = f"{run_name}_stage3"
    cfg = build_cfg(n_layers, d_model, seed)
    stage2_args = build_stage2_args(run_name, d_model=d_model, smoke=smoke)

    print(f"\n{'='*80}\n[E1] config={config_id} seed={seed} "
          f"stage2_epochs={stage2_args.epochs} smoke={smoke}\n{'='*80}", flush=True)
    t0 = time.time()

    # --- Stage 2 in-process train ---
    stage2_final = m1_7.train(cfg, stage2_args)
    stage2_sec = time.time() - t0
    stage2_val = stage2_final.get('best_val_total', float('nan'))
    print(f"  Stage 2: {stage2_sec:.1f}s  best_val_total={stage2_val:.4f}", flush=True)

    # --- Stage 3 in-process train (m1_8.stage3_train) ---
    stage2_dir = _ROOT / "runs/m1_7_train" / run_name
    stage3_args = build_stage3_args(stage3_run_name, stage2_dir, d_model=d_model, smoke=smoke)
    print(f"\n[E1] === {config_id}/s{seed} === Stage 3 (epochs={stage3_args.epochs}) ===", flush=True)
    t1 = time.time()
    stage3_final = m1_8.stage3_train(cfg, stage3_args)
    stage3_sec = time.time() - t1
    stage3_val = stage3_final.get('best_val_total', float('nan'))
    print(f"  Stage 3: {stage3_sec:.1f}s  best_val_total={stage3_val:.4f}  "
          f"(Δ vs Stage 2: {stage3_val - stage2_val:+.4f})", flush=True)

    # --- Design-val inference (Stage 3 best.pt 사용) ---
    ckpt_path = _ROOT / "runs/m1_8_stage3_train" / stage3_run_name / "best.pt"
    assert ckpt_path.exists(), f"Stage 3 best.pt missing: {ckpt_path}"
    t2 = time.time()
    infer = design_val_inference(cfg, ckpt_path, DESIGN_HMM_DIR, device)
    df_pred = infer["predictions"]
    infer_sec = time.time() - t2
    print(f"  inference: {infer_sec:.1f}s  n_eval={infer['n_after_filter']} "
          f"(n_dropped boundary={infer['n_dropped']})", flush=True)

    # --- Cov95 + WIS (raw HMM 분산, *diagnostic*; selection 은 val_total) ---
    cov, wis = cov95_wis(df_pred)
    print(f"  diagnostic: Cov95={cov:.4f}  WIS={wis:.4f}  (selection metric = val_total={stage3_val:.4f})",
          flush=True)

    # --- Save per-run JSON (증분 저장) ---
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    per_run = dict(
        config=dict(n_layers=n_layers, d_model=d_model, seed=seed,
                     stage2_epochs=stage2_args.epochs,
                     stage3_epochs=stage3_args.epochs, smoke=smoke),
        timings=dict(stage2_sec=stage2_sec, stage3_sec=stage3_sec, infer_sec=infer_sec),
        stage2_metrics=stage2_final,
        stage3_metrics=stage3_final,
        eval=dict(n_eval=int(len(df_pred)), n_dropped_boundary=int(infer["n_dropped"]),
                   cov95_diagnostic=cov, wis_diagnostic=wis,
                   selection_metric_val_total=stage3_val),
    )
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(per_run, f, indent=2, default=str)
    df_pred.to_parquet(run_dir / "design_val_predictions.parquet", index=False)
    print(f"  saved: {run_dir}/metrics.json + design_val_predictions.parquet", flush=True)

    return per_run


# ============================================================================
# 5-seed pooling — val_total mean (paper m1_9_hpo_phase2 selection metric)
# + Cov95/WIS diagnostic (selection 무관, raw HMM 진단용)
# ============================================================================
def pool_5_seeds(config_runs: list[dict], config_id: str) -> dict:
    """γ.3 정정 — selection metric = mean stage3_best_val_total (paper m1_9 일치).
       Cov95/WIS 는 diagnostic only."""
    # Selection metric: 5-seed mean of stage3 best_val_total (점예측 loss, ascending)
    val_totals = [r["eval"]["selection_metric_val_total"] for r in config_runs]
    val_total_mean = float(np.mean(val_totals))
    val_total_std = float(np.std(val_totals, ddof=1)) if len(val_totals) > 1 else 0.0

    # Diagnostic: Cov95/WIS pooled
    df_pool = pd.concat([
        pd.read_parquet(OUT_DIR / f"e1_{config_id}_s{r['config']['seed']}/design_val_predictions.parquet")
        for r in config_runs
    ], ignore_index=True)
    cov, wis = cov95_wis(df_pool)

    return dict(
        config_id=config_id,
        n_layers=config_runs[0]["config"]["n_layers"],
        d_model=config_runs[0]["config"]["d_model"],
        n_seeds=len(config_runs),
        # Selection metric (paper m1_9 일치)
        val_total_mean=val_total_mean,
        val_total_std=val_total_std,
        # Diagnostic only
        n_pooled=int(len(df_pool)),
        cov95_pooled_diagnostic=cov,
        wis_pooled_diagnostic=wis,
        per_seed=[dict(seed=r["config"]["seed"],
                        val_total=r["eval"]["selection_metric_val_total"],
                        cov95=r["eval"]["cov95_diagnostic"],
                        wis=r["eval"]["wis_diagnostic"],
                        n_eval=r["eval"]["n_eval"])
                   for r in config_runs],
    )


def select_by_val_total(pooled: list[dict]) -> dict:
    """γ.3 정정 — argmin val_total_mean ascending (paper m1_9_hpo_phase2 일치).
       band/fallback 폐기 — val_total 은 scalar metric, 단순 ranking."""
    if not pooled:
        return dict(status="E1_FAIL_NO_CONFIGS", winner=None, all_ranked=[])
    ranked = sorted(pooled, key=lambda p: p["val_total_mean"])
    return dict(
        status="OK",
        selection_metric="mean_stage3_best_val_total (= MSE + 0.3·MASE), ascending",
        winner=ranked[0],
        all_ranked=ranked,
    )


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="E1 design-split HPO launcher")
    parser.add_argument("--smoke", action="store_true",
                         help="1 config × 1 seed × 5 epochs, smoke check (n=612 검증)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Smoke: 1 config (n_layers=3, d_model=64 = paper winner) × 1 seed (42) × 짧은 Stage 2+3 ===
    if args.smoke:
        print(f"\n{'#'*80}\n# E1 SMOKE TEST — n3_d64 × seed42 × short Stage 2+3 (paper HP env)\n"
              f"# Verify: CG_TOP1_HP override + Stage 3 호출 + val_total selection metric + design-cut split\n"
              f"{'#'*80}", flush=True)
        run = run_one(n_layers=3, d_model=64, seed=42, smoke=True, device=device)
        n = run["eval"]["n_eval"]
        val_total = run["eval"]["selection_metric_val_total"]
        print(f"\n=== SMOKE GATE ===")
        print(f"  n_eval (design-val) = {n}")
        if abs(n - SMOKE_EXPECTED_N) <= SMOKE_TOLERANCE:
            print(f"  ✓ n_eval PASS  (expected ≈ {SMOKE_EXPECTED_N} ± {SMOKE_TOLERANCE})")
        else:
            print(f"  ✗ n_eval ABORT  (expected ≈ {SMOKE_EXPECTED_N}, got {n})")
            print(f"    n=624 → MultiHorizonDataset boundary leak (held-out 침범)")
            print(f"    n≈300 또는 75 → 원 val 75-row CSV (design-cut CSV 누락)")
            sys.exit(1)
        print(f"  ✓ selection metric (stage3 best_val_total) = {val_total:.4f}  [paper m1_9 일치]")
        print(f"  ✓ paper HP override 작동: cfg.lookback=104, backbone_lr=1e-4")
        print(f"  ✓ Stage 3 ckpt 위치: runs/m1_8_stage3_train/e1_n3_d64_s42_stage3/best.pt")
        print(f"\n  smoke PASS → full launch 가능: python scripts/e1_hpo.py")
        return

    # === Full: 9 configs × 5 seeds = 45 runs ===
    print(f"\n{'#'*80}\n# E1 FULL HPO — 9 configs × 5 seeds = 45 runs\n{'#'*80}", flush=True)
    print(f"# device={device}  design-CSV={DESIGN_CSV.name}  design-HMM={DESIGN_HMM_DIR.name}")
    all_runs = {}
    t_total = time.time()
    for n_layers, d_model in product(GRID_N_LAYERS, GRID_D_MODEL):
        config_id = f"n{n_layers}_d{d_model}"
        config_runs = []
        for seed in SEEDS:
            per_run = run_one(n_layers, d_model, seed, smoke=False, device=device)
            config_runs.append(per_run)
        all_runs[config_id] = config_runs
        # 즉시 config-level pooled 저장
        pooled = pool_5_seeds(config_runs, config_id)
        with open(OUT_DIR / f"{config_id}_pooled.json", "w") as f:
            json.dump(pooled, f, indent=2, default=str)
        print(f"\n[E1 CONFIG DONE] {config_id}: "
              f"val_total={pooled['val_total_mean']:.4f}±{pooled['val_total_std']:.4f}  "
              f"diag Cov95={pooled['cov95_pooled_diagnostic']:.4f}  "
              f"diag WIS={pooled['wis_pooled_diagnostic']:.4f}  n={pooled['n_pooled']}", flush=True)

    elapsed_h = (time.time() - t_total) / 3600
    print(f"\n{'#'*80}\n# E1 ALL RUNS DONE — {elapsed_h:.2f}h elapsed\n{'#'*80}", flush=True)

    # === γ.3 정정 — val_total ascending selection (paper m1_9_hpo_phase2 일치) ===
    pooled_list = []
    for config_id in [f"n{l}_d{d}" for l, d in product(GRID_N_LAYERS, GRID_D_MODEL)]:
        with open(OUT_DIR / f"{config_id}_pooled.json") as f:
            pooled_list.append(json.load(f))

    selection = select_by_val_total(pooled_list)

    print(f"\n=== γ.3 SELECTION RESULT (paper m1_9_hpo_phase2 일치, val_total ascending) ===")
    print(f"  status: {selection['status']}")
    print(f"  selection metric: {selection.get('selection_metric', 'val_total ascending')}")
    if selection["winner"]:
        w = selection["winner"]
        print(f"  WINNER: config={w['config_id']}  "
              f"(n_layers={w['n_layers']}, d_model={w['d_model']})  "
              f"val_total={w['val_total_mean']:.4f}±{w['val_total_std']:.4f}  "
              f"[diag Cov95={w['cov95_pooled_diagnostic']:.4f}, WIS={w['wis_pooled_diagnostic']:.4f}]")

    # All configs table (sorted by val_total ascending)
    print(f"\n=== ALL 9 CONFIGS (sorted by val_total ascending — selection ranking) ===")
    print(f"  {'rank':>4s}  {'config_id':10s}  {'val_total':>11s}  {'±std':>7s}  "
          f"{'[diag Cov95':>12s}  {'WIS]':>7s}")
    for i, p in enumerate(selection["all_ranked"], 1):
        marker = " ← WINNER" if i == 1 else ""
        print(f"  {i:>4d}  {p['config_id']:10s}  {p['val_total_mean']:>11.4f}  "
              f"{p['val_total_std']:>7.4f}  {p['cov95_pooled_diagnostic']:>12.4f}  "
              f"{p['wis_pooled_diagnostic']:>7.4f}{marker}")

    # Final aggregate
    aggregate = dict(
        locked_constants=dict(
            grid_n_layers=GRID_N_LAYERS, grid_d_model=GRID_D_MODEL,
            k_phase_fixed=3, seeds=SEEDS,
            stage2_epochs=STAGE2_EPOCHS, stage3_epochs=STAGE3_EPOCHS,
            stage3_patience=STAGE3_PATIENCE,
            selection_metric="mean_stage3_best_val_total_ascending",
            cg_top1_hp=CG_TOP1_HP,
            other_lr_base=OTHER_LR_BASE,
            design_csv=str(DESIGN_CSV.relative_to(_ROOT)),
            design_norm_json=str(DESIGN_NORM_JSON.relative_to(_ROOT)),
            design_hmm_dir=str(DESIGN_HMM_DIR.relative_to(_ROOT)),
            design_val_end_epiweek=DESIGN_VAL_END_EPIWEEK,
        ),
        elapsed_hours=elapsed_h,
        pooled_per_config=pooled_list,
        selection=selection,
    )
    with open(OUT_DIR / "e1_results.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"\n  saved: {OUT_DIR / 'e1_results.json'}")


if __name__ == "__main__":
    main()
