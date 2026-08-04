"""M2.1 Final — CG-Mamba HPO winner cfg × 5-seed × per-horizon raw val MAE.

Reuses existing seeds {42, 123, 456} ckpts (HMM + Stage 2 + Stage 3).
Trains seeds {789, 1024} from scratch (Stage 1 HMM -> Stage 2 -> Stage 3).
Then evaluates per-horizon raw val MAE for all 5 seeds.

Output (LSTM-style for M2.3 comparability):
  runs/m2_1_final/cg_mamba_winner/
    ├── seed{S}/per_horizon_mae.json     # per-seed raw MAE [h=1..4]
    └── final_summary.json               # 5-seed aggregate (mean ± std)

HPO winner cfg (from runs/m1_9_hpo_phase2/hpo_winner.json, v2.1.7-A schema):
  base       : gate_lr, backbone_lr, lookback (Phase 1 winner)
  Stage 3    : hmm_lr_ratio, state_embed_lr_ratio, env_lr_ratio (Phase 2 winner)
  Each LR    : stage3_<group>_lr = OTHER_LR_BASE × <group>_lr_ratio
  Stage 2    : 200 ep + Warm-γ scheduler
  Stage 3    : 30 ep + patience=10, 4-group optimizer (v2.1.7-A)

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/m2_1_final.py
  CUDA_VISIBLE_DEVICES=0 python scripts/m2_1_final.py --seeds 789 1024  (subset)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from time import time

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import CGMambaConfig                              # noqa: E402
from src.models.cg_forecaster import CGForecaster                       # noqa: E402
from src.utils.checkpoints import load_fitted_hmm                       # noqa: E402
from src.data.loader import (                                            # noqa: E402
    MultiHorizonDataset,
    load_dataset_csv,
    load_norm_params,
)
from torch.utils.data import DataLoader                                  # noqa: E402

from scripts.m1_7_train import train as stage2_train                    # noqa: E402
from scripts.m1_8_stage3_train import stage3_train                      # noqa: E402


# ---------------------------------------------------------------------------
# HPO winner cfg — loaded dynamically from hpo_winner.json (v2.1.7 C-2 fix)
# ---------------------------------------------------------------------------
HPO_WINNER_JSON = _ROOT / "runs/m1_9_hpo_phase2/hpo_winner.json"
OTHER_LR_BASE = 1e-4   # Stage 3 base LR for "other" group; hmm_lr = OTHER_LR_BASE * ctx_ratio
ALL_SEEDS = [42, 123, 456, 789, 1024]


def _load_winner_cfg() -> dict:
    """Load HPO winner from hpo_winner.json (v2.1.7-A 4-group schema).

    Backward-compatible with v2.1.7 1-axis (context_lr_ratio) schema:
    - If `final_cfg.hmm_lr_ratio` present → v2.1.7-A (4-group).
    - Otherwise fall back to legacy `context_lr_ratio` (used as hmm_lr_ratio).
    """
    if not HPO_WINNER_JSON.exists():
        raise FileNotFoundError(
            f"hpo_winner.json not found at {HPO_WINNER_JSON}. "
            "Run HPO Phase 2 first (scripts/m1_9_hpo_phase2.py --mode grid --mode final)."
        )
    blob = json.loads(HPO_WINNER_JSON.read_text())
    w = blob.get("phase2_winner") or blob.get("winner") or {}
    fc = blob.get("final_cfg") or {}

    result = {
        "gate_lr": float(w.get("base_gate_lr", fc.get("gate_lr"))),
        "backbone_lr": float(w.get("base_backbone_lr", fc.get("backbone_lr"))),
        "lookback": int(w.get("base_lookback", fc.get("lookback"))),
        "dropout": float(fc.get("dropout", 0.0)),
    }
    if "hmm_lr_ratio" in fc:
        # v2.1.7-A 4-group schema
        result.update({
            "schema": "v2.1.7-A",
            "hmm_lr_ratio":         float(fc["hmm_lr_ratio"]),
            "state_embed_lr_ratio": float(fc["state_embed_lr_ratio"]),
            "env_lr_ratio":         float(fc["env_lr_ratio"]),
        })
    else:
        # Legacy v2.1.7 (context_lr_ratio = hmm_lr_ratio only)
        cr = float(w.get("context_lr_ratio", fc.get("context_lr_ratio", 0.1)))
        result.update({
            "schema": "v2.1.7-legacy",
            "hmm_lr_ratio":         cr,
            "state_embed_lr_ratio": 0.1,  # Stage 2 default 1e-5 (× 1e-4 = 1e-5)
            "env_lr_ratio":         1.0,  # Stage 3 other_lr default 1e-4
        })
    return result


# v2.1.7 C-2 import-time concern: lazy load via cached getter (import doesn't
# crash if HPO not yet run; main() is the actual entry that requires it).
_WINNER_CACHE: dict | None = None


def _winner() -> dict:
    global _WINNER_CACHE
    if _WINNER_CACHE is None:
        _WINNER_CACHE = _load_winner_cfg()
    return _WINNER_CACHE


def _winner_stage2_tag() -> str:
    w = _winner()
    return f"hpo_p1_g{w['gate_lr']:.0e}_b{w['backbone_lr']:.0e}_lb{w['lookback']}"


def _winner_stage3_tag() -> str:
    """v2.1.7-A 4-group tag: base + 3 ratios."""
    w = _winner()
    return (f"hpo_p2_g{w['gate_lr']:.0e}_b{w['backbone_lr']:.0e}_lb{w['lookback']}"
            f"_h{w['hmm_lr_ratio']}_se{w['state_embed_lr_ratio']}_e{w['env_lr_ratio']}")


HMM_DIR_TMPL = str(_ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}")
ENV_CKPT = _ROOT / "runs/m1_7_env_pretrain/env_encoder.pt"
OUT_ROOT = _ROOT / "runs/m2_1_final/cg_mamba_winner"


def _stage2_dir_for(seed: int) -> str:
    """Stage 2 ckpt dir (lazy — depends on HPO winner; not resolvable at import time)."""
    return str(_ROOT / "runs/m1_7_train" / (_winner_stage2_tag() + f"_s{seed}"))


def _stage3_dir_for(seed: int) -> str:
    return str(_ROOT / "runs/m1_8_stage3_train" / (_winner_stage3_tag() + f"_s{seed}"))

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
NORM_PATH = _ROOT / "data/processed/normalization_params.json"


# ---------------------------------------------------------------------------
def _build_cfg(seed: int) -> CGMambaConfig:
    """Build cfg with HPO winner + Stage 3 LR fields (v2.1.7 C-1 fix).

    Stage 3 LRs propagate via cfg.stage3_hmm_lr / cfg.stage3_other_lr
    (formerly monkey-patched, which was a no-op — see C-1 changelog).
    """
    w = _winner()
    return dataclasses.replace(
        CGMambaConfig(),
        stage2_gate_lr=w["gate_lr"],
        stage2_backbone_lr=w["backbone_lr"],
        lookback=w["lookback"],
        dropout=w["dropout"],
        seed=seed,
        # v2.1.7-A 4-group: all 4 Stage 3 LRs from winner cfg
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * w["hmm_lr_ratio"],
        stage3_state_embed_lr=OTHER_LR_BASE * w["state_embed_lr_ratio"],
        stage3_env_lr=OTHER_LR_BASE * w["env_lr_ratio"],
    )


def _ensure_hmm(seed: int) -> None:
    hmm_dir = Path(HMM_DIR_TMPL.format(seed=seed))
    if (hmm_dir / "hmm_params.npz").exists():
        print(f"  [HMM seed={seed}] SKIP (exists)")
        return
    print(f"  [HMM seed={seed}] training via m1_4 subprocess...")
    cmd = [sys.executable, "-u", str(_ROOT / "scripts/m1_4_phase_dynamics_main.py"),
           "--seeds", str(seed)]
    t0 = time()
    rc = subprocess.run(cmd, cwd=str(_ROOT)).returncode
    if rc != 0 or not (hmm_dir / "hmm_params.npz").exists():
        raise RuntimeError(f"HMM training failed for seed={seed} (rc={rc})")
    print(f"  [HMM seed={seed}] done in {time()-t0:.1f}s")


def _ensure_stage2(seed: int) -> None:
    s2_dir = Path(_stage2_dir_for(seed))
    if (s2_dir / "best.pt").exists():
        print(f"  [Stage2 seed={seed}] SKIP (exists)")
        return
    print(f"  [Stage2 seed={seed}] training (200ep, Warm-γ)...")
    cfg = _build_cfg(seed)
    args = SimpleNamespace(
        smoke=False,
        epochs=None,                          # cfg.stage2_n_epochs default (200)
        batch_size=32,
        hmm_dir=HMM_DIR_TMPL.format(seed=seed),
        env_encoder_ckpt=str(ENV_CKPT),
        wandb_mode="disabled",
        run_name=f"{_winner_stage2_tag()}_s{seed}",
    )
    t0 = time()
    final = stage2_train(cfg, args)
    if not (s2_dir / "best.pt").exists():
        raise RuntimeError(f"Stage 2 produced no best.pt for seed={seed}")
    print(f"  [Stage2 seed={seed}] done in {time()-t0:.1f}s  "
          f"best_val_total={final.get('best_val_total', float('nan')):.4f}")


def _ensure_stage3(seed: int) -> None:
    s3_dir = Path(_stage3_dir_for(seed))
    if (s3_dir / "best.pt").exists():
        print(f"  [Stage3 seed={seed}] SKIP (exists)")
        return
    _w = _winner()
    print(f"  [Stage3 seed={seed}] training (30ep + patience=10, "
          f"hmm_r={_w['hmm_lr_ratio']}, se_r={_w['state_embed_lr_ratio']}, "
          f"env_r={_w['env_lr_ratio']})...")
    cfg = _build_cfg(seed)
    s2_dir = _stage2_dir_for(seed)
    args = SimpleNamespace(
        smoke=False,
        epochs=30,
        patience=10,
        batch_size=32,
        stage2_dir=s2_dir,
        hmm_dir=HMM_DIR_TMPL.format(seed=seed),
        env_encoder_ckpt=str(ENV_CKPT),
        run_name=f"{_winner_stage3_tag()}_s{seed}",
    )
    # v2.1.7 C-1 fix: ctx_ratio now in cfg.stage3_hmm_lr (set by _build_cfg above);
    # _build_stage3_optimizer reads it directly. Monkey-patch removed (was no-op).
    t0 = time()
    final = stage3_train(cfg, args)
    if not (s3_dir / "best.pt").exists():
        raise RuntimeError(f"Stage 3 produced no best.pt for seed={seed}")
    print(f"  [Stage3 seed={seed}] done in {time()-t0:.1f}s  "
          f"test_mse={final.get('test_mse', float('nan')):.4f}")


# ---------------------------------------------------------------------------
def _eval_per_horizon_raw_mae(seed: int) -> tuple[list[float], int]:
    """Load Stage 3 best.pt, run val_loader, return raw per-horizon MAE list (h=1..4)."""
    cfg = _build_cfg(seed)
    s3_dir = Path(_stage3_dir_for(seed))
    hmm_dir = Path(HMM_DIR_TMPL.format(seed=seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_dataset_csv(CSV_PATH)
    norm = load_norm_params(NORM_PATH)
    horizons = tuple(cfg.horizons)
    val_ds = MultiHorizonDataset(df, "val", cfg.lookback, horizons, norm)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std = float(norm["ili_weighted_pct"]["std"])

    hmm = load_fitted_hmm(hmm_dir)
    model = CGForecaster(cfg)
    # Mandatory M-4: load HMM + env into PhaseModule before forward
    model.prepare_for_stage2(hmm)
    # Load env encoder (Stage 3 best.pt includes both backbone + env_encoder)
    sd = torch.load(s3_dir / "best.pt", map_location=device)
    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "model" in sd:
            sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    per_h_abs_sum = None
    n_total = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            y = batch["y"].to(device)                            # [B, H] z-scored
            pred = model(x, env)                                 # [B, H] z-scored
            # Denormalize both to raw scale
            pred_raw = pred * target_std + target_mean
            y_raw = y * target_std + target_mean
            B = y.size(0)
            per_h = (pred_raw - y_raw).abs().sum(dim=0)          # [H]
            if per_h_abs_sum is None:
                per_h_abs_sum = per_h
            else:
                per_h_abs_sum = per_h_abs_sum + per_h
            n_total += B

    per_h_mae = (per_h_abs_sum / max(n_total, 1)).cpu().tolist()
    return per_h_mae, n_total


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="M2.1 Final — CG-Mamba 5-seed final eval")
    parser.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS,
                        help="Seeds to ensure + evaluate (default 5 seeds)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; only evaluate per-horizon MAE")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    w = _winner()
    print(f"[M2.1 Final] CG-Mamba HPO winner × {len(args.seeds)} seeds")
    print(f"  cfg: gate_lr={w['gate_lr']}, bb_lr={w['backbone_lr']}, "
          f"lookback={w['lookback']}, dropout={w['dropout']}\n"
          f"        Stage 3 ratios: hmm={w['hmm_lr_ratio']}  "
          f"state_embed={w['state_embed_lr_ratio']}  env={w['env_lr_ratio']}  "
          f"(schema={w.get('schema', 'v2.1.7-legacy')})")
    print(f"  seeds: {args.seeds}")
    print(f"  output: {OUT_ROOT.relative_to(_ROOT)}/")
    print()

    per_seed_results = []
    for seed in args.seeds:
        seed_dir = OUT_ROOT / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== seed {seed} ===")

        if not args.eval_only:
            _ensure_hmm(seed)
            _ensure_stage2(seed)
            _ensure_stage3(seed)

        t0 = time()
        per_h_mae, n_val = _eval_per_horizon_raw_mae(seed)
        print(f"  [Eval seed={seed}] per_horizon_raw_MAE: "
              f"h=1:{per_h_mae[0]:.4f}  h=2:{per_h_mae[1]:.4f}  "
              f"h=3:{per_h_mae[2]:.4f}  h=4:{per_h_mae[3]:.4f}  "
              f"(n_val={n_val}, {time()-t0:.1f}s)")

        with (seed_dir / "per_horizon_mae.json").open("w") as f:
            json.dump({"seed": seed, "per_horizon_mae": per_h_mae,
                       "n_val_windows": n_val}, f, indent=2)
        per_seed_results.append({"seed": seed, "per_horizon_mae": per_h_mae})

    # Aggregate -> LSTM-style final_summary.json
    # v2.1.7-A++ fix: scan ALL persisted seed dirs (not just current --seeds), so
    # partial reruns (e.g. `--seeds 789`) do not overwrite a prior multi-seed aggregate.
    on_disk = {}
    for sd in sorted(OUT_ROOT.glob("seed*")):
        f = sd / "per_horizon_mae.json"
        if f.exists():
            obj = json.loads(f.read_text())
            on_disk[int(obj["seed"])] = obj["per_horizon_mae"]
    aggregated_seeds = sorted(on_disk.keys())
    per_h_array = np.array([on_disk[s] for s in aggregated_seeds])  # [N, H]
    summary = {
        "config": {
            "name": "CG-Mamba HPO winner",
            "gate_lr": w["gate_lr"],
            "backbone_lr": w["backbone_lr"],
            "lookback": w["lookback"],
            "hmm_lr_ratio":         w["hmm_lr_ratio"],
            "state_embed_lr_ratio": w["state_embed_lr_ratio"],
            "env_lr_ratio":         w["env_lr_ratio"],
            "dropout": w["dropout"],
            "stage2_epochs": 200,
            "stage3_epochs": 30,
            "stage3_patience": 10,
        },
        "n_seeds": len(aggregated_seeds),
        "seeds": aggregated_seeds,
        "mae_mean_per_horizon": per_h_array.mean(axis=0).tolist(),
        "mae_std_per_horizon": per_h_array.std(axis=0).tolist(),
        "mae_per_seed_per_horizon": per_h_array.tolist(),
    }
    summary_path = OUT_ROOT / "final_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[M2.1 Final DONE]")
    print(f"  mean MAE per horizon (h=1..4): "
          f"{[f'{v:.4f}±{s:.3f}' for v, s in zip(summary['mae_mean_per_horizon'], summary['mae_std_per_horizon'])]}")
    print(f"  saved: {summary_path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
