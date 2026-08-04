"""M1.7 Step 6 — Stage 2 trainer (CG-Mamba composite loss + Warm-γ schedule).

PLAN v2.0.9 §5.1 D.5.2 (active spec):
    - Loss:        L = MSE + 0.3·MASE          (lambda_mase=0.3, λ_sparse=0)
    - Loader:      MultiHorizonDataset, horizons=(1,2,3,4), batch_size=32
    - Optimizer:   AdamW, 4 param groups (gate_proj / decoder_gate / context_embed / backbone)
    - LR sched:    Warm-γ 3-Phase (warmup→Phase 1→Transition→Cosine), n_epochs=200, patience=30
    - Grad clip:   1.0
    - HMM init:    cached buffer (T-1), from m1_4_phase_dynamics_main Stage 1 output
    - Env init:    from m1_7_env_pretrain `env_encoder.pt` (if provided)

CLI:
    python scripts/m1_7_train.py --smoke                 # 5 epochs, log every epoch
    python scripts/m1_7_train.py                          # PLAN default 200ep
    python scripts/m1_7_train.py --epochs 50 --batch-size 16
    python scripts/m1_7_train.py --wandb-mode online

Output:
    runs/m1_7_train/<run_name>/
        best.pt              # best-val checkpoint (model state_dict + epoch + metrics)
        history.json         # per-epoch train/val MSE+MASE+total + per-group LR
        config.json          # CGMambaConfig snapshot
        final_metrics.json   # test MSE/MASE at best-val epoch
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[1]
if str(_CG_MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CG_MAMBA_ROOT))

from src.data.loader import (  # noqa: E402
    MultiHorizonDataset,
    collate_dict,
    load_dataset_csv,
    load_norm_params,
)
from src.models.cg_forecaster import CGForecaster  # noqa: E402
from src.models.env_module import EnvModule  # noqa: E402 (only for type hint)
from src.utils.checkpoints import load_fitted_hmm  # noqa: E402
from src.utils.config import CGMambaConfig  # noqa: E402
from src.utils.losses import (  # noqa: E402
    cg_mamba_loss,
    compute_seasonal_mae,
    mase_loss,
)
from src.utils.optimizer import build_stage2_optimizer  # noqa: E402
from src.utils.scheduler import build_warm_gamma_scheduler  # noqa: E402


_DEFAULT_HMM_DIR = (
    _CG_MAMBA_ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed42"
)
_DEFAULT_ENV_CKPT = _CG_MAMBA_ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"


# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────
def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_loaders(cfg: CGMambaConfig, batch_size: int, seed: int):
    df = load_dataset_csv(cfg.data_csv)
    norm = load_norm_params(cfg.norm_json)
    horizons = tuple(cfg.horizons)

    train_ds = MultiHorizonDataset(df, "train", cfg.lookback, horizons, norm)
    val_ds = MultiHorizonDataset(df, "val", cfg.lookback, horizons, norm)
    test_ds = MultiHorizonDataset(df, "test", cfg.lookback, horizons, norm)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_dict, generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_dict, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_dict, drop_last=False,
    )
    return df, norm, train_loader, val_loader, test_loader


def _evaluate(model, loader, seasonal_mae, device, lambda_mase=0.3):
    """Run loss on a loader (eval mode, no_grad).

    Returns aggregate {mse, mase, total} plus per-horizon MAE for direct
    inspection of rollout degradation (Step 8 analysis check #3).
    """
    model.eval()
    n_total = 0
    mse_sum = 0.0
    mase_sum = 0.0
    per_h_abs_sum = None       # [H] running sum of |pred - target|
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            y = batch["y"].to(device)
            pred = model(x, env)
            B = y.size(0)
            mse = torch.nn.functional.mse_loss(pred, y, reduction="mean").item()
            mase = mase_loss(pred, y, seasonal_mae).item()
            mse_sum += mse * B
            mase_sum += mase * B
            # Per-horizon MAE in normalized space — sum, divide at end
            per_h_batch = (pred - y).abs().sum(dim=0)              # [H]
            if per_h_abs_sum is None:
                per_h_abs_sum = per_h_batch
            else:
                per_h_abs_sum = per_h_abs_sum + per_h_batch
            n_total += B
    mse_mean = mse_sum / max(n_total, 1)
    mase_mean = mase_sum / max(n_total, 1)
    per_h_mae = (per_h_abs_sum / max(n_total, 1)).cpu().tolist() if per_h_abs_sum is not None else []
    return {
        "mse": mse_mean,
        "mase": mase_mean,
        "total": mse_mean + lambda_mase * mase_mean,
        "per_horizon_mae": per_h_mae,           # list[float], len = len(horizons)
    }


def _train_one_epoch(model, loader, optimizer, seasonal_mae, device,
                     grad_clip=1.0, lambda_mase=0.3):
    model.train()
    n_total = 0
    loss_sum = 0.0
    mse_sum = 0.0
    mase_sum = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        env = batch["env"].to(device)
        y = batch["y"].to(device)
        pred = model(x, env)
        loss = cg_mamba_loss(pred, y, seasonal_mae, lambda_mase=lambda_mase)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        B = y.size(0)
        loss_sum += loss.item() * B
        mse_sum += torch.nn.functional.mse_loss(pred.detach(), y).item() * B
        mase_sum += mase_loss(pred.detach(), y, seasonal_mae).item() * B
        n_total += B
    return {
        "total": loss_sum / max(n_total, 1),
        "mse": mse_sum / max(n_total, 1),
        "mase": mase_sum / max(n_total, 1),
    }


def _gate_diagnostics(model, val_loader, device, n_batches=1):
    """Sample intermediates from first n_batches of val for monitoring.

    Two metric groups:
      (1) Decoder/rollout side (eff_gate, confidence, KL):
            from `return_intermediates=True` → EntropyAwareDecoder outputs.
            `eff_gate_std` detects state_embeddings differentiation
            (Step 8 v3 analysis: stagnant std with improving test_mse → see (2)).
      (2) Backbone side (gate_proj weight norm + per-layer gate_i stats):
            from ContextGatedMambaBlock._last_gate cache. Tests the
            "indirect HMM-init path" hypothesis — HMM init → distinct
            context_vec → gate_proj learns phase signal → gate_i becomes
            phase-differentiated.

    Eval-mode caveat: ContextGatedMambaBlock._last_gate is skipped in eval
    (L4 memory opt). We briefly switch to train mode to populate the cache.
    cfg.dropout=0.0 + RMSNorm (no running stats) → train mode is safe under
    no_grad.
    """
    was_training = model.training
    model.train()                                          # populate _last_gate
    eff_gate_means, eff_gate_stds = [], []
    conf_means, conf_stds = [], []
    kl_means = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            _, inter = model(x, env, return_intermediates=True)
            eff_gate_means.append(float(inter["eff_gate_per_horizon"].mean()))
            eff_gate_stds.append(float(inter["eff_gate_per_horizon"].std()))
            conf_means.append(float(inter["confidence_per_horizon"].mean()))
            conf_stds.append(float(inter["confidence_per_horizon"].std()))
            kl_means.append(float(inter["phase_transition_kl"].mean()))

    # (2) Backbone gate metrics — verifies CE-1 / "indirect path" hypothesis.
    gate_proj_norms = []
    gate_i_means = []
    gate_i_stds = []
    for layer in model.encoder.layers:
        # gate_proj is Sequential([Linear(D, r), SiLU, Linear(r, ED) ×0.01]).
        # The last Linear's weight norm tracks the phase-amplification capacity
        # (×0.01 init → grows as gate_proj learns to amplify context_vec).
        last_linear = layer.gate_proj[-1]
        gate_proj_norms.append(float(last_linear.weight.norm().item()))
        if layer._last_gate is not None:
            gate_i_means.append(float(layer._last_gate.mean().item()))
            gate_i_stds.append(float(layer._last_gate.std().item()))
        else:
            # C-2 fix: json.dump rejects NaN/Inf. Use None (→ JSON null) so the
            # 200-epoch run's history.json saves without ValueError even if
            # _last_gate cache is unexpectedly absent (e.g., layer fast-path,
            # disable_gate=True, or train→eval toggle race).
            gate_i_means.append(None)
            gate_i_stds.append(None)

    if not was_training:
        model.eval()

    return {
        # (1) Decoder/rollout side
        "eff_gate_mean": float(np.mean(eff_gate_means)) if eff_gate_means else float("nan"),
        "eff_gate_std": float(np.mean(eff_gate_stds)) if eff_gate_stds else float("nan"),
        "confidence_mean": float(np.mean(conf_means)) if conf_means else float("nan"),
        "confidence_std": float(np.mean(conf_stds)) if conf_stds else float("nan"),
        "phase_transition_kl_mean": float(np.mean(kl_means)) if kl_means else float("nan"),
        # (2) Backbone side (per-layer lists)
        "gate_proj_norms": gate_proj_norms,
        "gate_i_means": gate_i_means,
        "gate_i_stds": gate_i_stds,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────
def train(cfg: CGMambaConfig, args) -> dict:
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = _CG_MAMBA_ROOT / "runs" / "m1_7_train" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    df, norm, train_loader, val_loader, test_loader = _build_loaders(
        cfg, batch_size=args.batch_size, seed=cfg.seed,
    )
    n_train_windows = len(train_loader.dataset)
    n_val_windows = len(val_loader.dataset)
    n_test_windows = len(test_loader.dataset)
    print(f"[M1.7 train] device={device}, cfg={cfg.summary()}")
    print(f"  windows: train={n_train_windows}, val={n_val_windows}, test={n_test_windows}")
    print(f"  batch_size={args.batch_size}, horizons={cfg.horizons}, lookback={cfg.lookback}")

    # ── Seasonal MAE (MASE denominator), pre-computed on train ──
    seasonal_mae = compute_seasonal_mae(df, norm).to(device)
    print(f"  seasonal MAE (train, normalized): {seasonal_mae.item():.6f}")

    # ── Model ──
    model = CGForecaster(cfg).to(device)

    # Env encoder pretrain weights (optional)
    if args.env_encoder_ckpt and Path(args.env_encoder_ckpt).exists():
        state = torch.load(args.env_encoder_ckpt, map_location=device, weights_only=True)
        model.env_module.encoder.load_state_dict(state)
        print(f"  loaded env encoder from {args.env_encoder_ckpt}")
    else:
        print(f"  env encoder: random init (no checkpoint)")

    # Fitted HMM → buffer cache + Env decoder freeze (mandatory T-1 + M-4)
    hmm = load_fitted_hmm(Path(args.hmm_dir))
    model.prepare_for_stage2(hmm)
    print(f"  loaded HMM from {args.hmm_dir} (K={hmm.K}, V={hmm.V}, "
          f"reg_covar={hmm.reg_covar:.0e})")

    # Param count summary
    pg_summary = model.param_group_summary()
    print(f"  trainable params: {pg_summary['trainable_total']:,}")

    # ── Optimizer + Scheduler ──
    optimizer = build_stage2_optimizer(model, cfg)
    n_epochs = args.epochs if args.epochs is not None else cfg.stage2_n_epochs

    # Smoke mode shortens schedule but keeps relative shape (warmup=1, P1=1, TR=1).
    # For real runs (>= cfg.stage2_n_epochs), use PLAN D.5.2 defaults.
    if args.smoke:
        scheduler = build_warm_gamma_scheduler(
            optimizer, total_epochs=n_epochs, P1=1, TR=1, warmup=1,
        )
    else:
        scheduler = build_warm_gamma_scheduler(
            optimizer, total_epochs=n_epochs,
        )

    # ── W&B ──
    use_wandb = args.wandb_mode != "disabled"
    if use_wandb:
        import wandb
        wandb.init(
            project="cg-mamba-jbhi",
            group="cg_forecaster_v2.1.2_stage2",
            name=args.run_name,
            mode=args.wandb_mode,
            config={
                "stage": "stage2",
                "n_epochs": n_epochs,
                "batch_size": args.batch_size,
                "lookback": cfg.lookback,
                "horizons": list(cfg.horizons),
                "seed": cfg.seed,
                "smoke": args.smoke,
                "hmm_dir": str(args.hmm_dir),
                "env_encoder_ckpt": str(args.env_encoder_ckpt) if args.env_encoder_ckpt else None,
            },
        )

    # ── Training loop ──
    history = {
        "train": {"total": [], "mse": [], "mase": []},
        "val": {"total": [], "mse": [], "mase": [], "per_horizon_mae": []},
        "lr": {n: [] for n in [g["name"] for g in optimizer.param_groups]},
        "diag": [],   # eff_gate + confidence + KL per epoch
    }
    best_val_total = math.inf
    best_epoch = -1
    epochs_since_improvement = 0
    patience = cfg.stage2_patience if not args.smoke else max(2, n_epochs)

    t0 = time()
    log_every = 1 if args.smoke else max(1, n_epochs // 20)
    for epoch in range(n_epochs):
        tr = _train_one_epoch(
            model, train_loader, optimizer, seasonal_mae, device,
            grad_clip=cfg.grad_clip, lambda_mase=0.3,
        )
        va = _evaluate(model, val_loader, seasonal_mae, device, lambda_mase=0.3)
        scheduler.step()

        # Record
        for k in ("total", "mse", "mase"):
            history["train"][k].append(tr[k])
            history["val"][k].append(va[k])
        history["val"]["per_horizon_mae"].append(va["per_horizon_mae"])
        for g in optimizer.param_groups:
            history["lr"][g["name"]].append(g["lr"])
        diag = _gate_diagnostics(model, val_loader, device, n_batches=1)
        history["diag"].append(diag)

        # Best tracking + early stopping
        improved = va["total"] < best_val_total
        if improved:
            best_val_total = va["total"]
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_total": va["total"],
                    "val_mse": va["mse"],
                    "val_mase": va["mase"],
                },
                out_root / "best.pt",
            )
        else:
            epochs_since_improvement += 1

        # W&B
        if use_wandb:
            log = {
                "epoch": epoch,
                "train/total": tr["total"], "train/mse": tr["mse"], "train/mase": tr["mase"],
                "val/total": va["total"], "val/mse": va["mse"], "val/mase": va["mase"],
                "val/eff_gate_mean": diag["eff_gate_mean"],
                "val/confidence_mean": diag["confidence_mean"],
                "val/phase_transition_kl": diag["phase_transition_kl_mean"],
                "best/val_total": best_val_total,
                "best/epoch": best_epoch,
            }
            for n in history["lr"]:
                log[f"lr/{n}"] = history["lr"][n][-1]
            wandb.log(log)

        if (epoch + 1) % log_every == 0 or epoch == 0 or improved:
            lr_str = ", ".join(
                f"{n[:8]}={history['lr'][n][-1]:.2e}" for n in history["lr"]
            )
            tag = " *best" if improved else ""
            per_h_str = "/".join(f"{m:.3f}" for m in va["per_horizon_mae"])
            # Backbone metrics — layer means (per-layer lists in history.json)
            # C-2 fix: gate_i_stds may contain None (eval-mode skip cache).
            # Filter None before np.mean — log NaN sentinel for visibility.
            gp_norm_mean = float(np.mean(diag["gate_proj_norms"]))
            _gi_valid = [v for v in diag["gate_i_stds"] if v is not None]
            gi_std_mean = float(np.mean(_gi_valid)) if _gi_valid else float("nan")
            print(
                f"  ep {epoch+1:3d}/{n_epochs}  "
                f"train={tr['total']:.4f}  val={va['total']:.4f} "
                f"(mse={va['mse']:.4f}, mase={va['mase']:.4f})  "
                f"per_h_mae=[{per_h_str}]  "
                f"eff_gate={diag['eff_gate_mean']:.3f}±{diag['eff_gate_std']:.3f}  "
                f"conf={diag['confidence_mean']:.3f}±{diag['confidence_std']:.3f}  "
                f"gp_norm={gp_norm_mean:.3f}  gi_std={gi_std_mean:.4f}  "
                f"[{lr_str}]{tag}"
            )

        if epochs_since_improvement >= patience:
            print(f"  early stop at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    elapsed = time() - t0

    # ── Final test eval at best-val checkpoint ──
    ckpt = torch.load(out_root / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test = _evaluate(model, test_loader, seasonal_mae, device, lambda_mase=0.3)
    print()
    print("=== M1.7 Stage 2 training summary ===")
    print(f"  best_epoch:   {best_epoch + 1}  (0-indexed {best_epoch})")
    print(f"  best_val:     total={best_val_total:.4f}  "
          f"mse={ckpt['val_mse']:.4f}  mase={ckpt['val_mase']:.4f}")
    print(f"  test (best):  total={test['total']:.4f}  "
          f"mse={test['mse']:.4f}  mase={test['mase']:.4f}")
    print(f"  elapsed:      {elapsed:.1f}s  ({n_train_windows} train windows × "
          f"{epoch + 1} epochs)")

    # ── Persist artifacts ──
    with open(out_root / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    config_dict = {k: v for k, v in cfg.__dict__.items()
                   if not k.startswith("_") and not callable(v)}
    # Path → str (json-serializable)
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in config_dict.items()}
    # tuple → list
    config_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in config_dict.items()}
    with open(out_root / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_total": best_val_total,
        "best_val_mse": ckpt["val_mse"],
        "best_val_mase": ckpt["val_mase"],
        "test_total": test["total"],
        "test_mse": test["mse"],
        "test_mase": test["mase"],
        "seasonal_mae": float(seasonal_mae.item()),
        "n_epochs_run": epoch + 1,
        "elapsed_sec": elapsed,
    }
    with open(out_root / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved: {out_root.relative_to(_CG_MAMBA_ROOT)}/{{best.pt, history.json, config.json, final_metrics.json}}")

    if use_wandb:
        import wandb
        wandb.finish()

    return final_metrics


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="M1.7 Stage 2 trainer")
    parser.add_argument("--smoke", action="store_true",
                        help="5 epochs, batch=8, P1=TR=warmup=1, log every step")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg.stage2_n_epochs (default 200)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-step batch size (default 32)")
    parser.add_argument("--hmm-dir", type=str, default=str(_DEFAULT_HMM_DIR),
                        help="Stage 1 HMM checkpoint directory")
    parser.add_argument("--env-encoder-ckpt", type=str, default=str(_DEFAULT_ENV_CKPT),
                        help="m1_7_env_pretrain encoder checkpoint (.pt)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default="disabled", help="W&B logging mode (default disabled)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (default auto from timestamp + smoke flag)")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 5
        args.batch_size = 8
    if args.run_name is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"smoke_{ts}" if args.smoke else f"stage2_{ts}"

    cfg = CGMambaConfig()
    final = train(cfg, args)
    return 0 if not math.isnan(final["best_val_total"]) else 1


if __name__ == "__main__":
    sys.exit(main())
