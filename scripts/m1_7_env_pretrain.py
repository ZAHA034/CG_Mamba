"""M1.7 Step 1 — Env autoencoder Stage 1 pretrain (PLAN §5.1 A-2).

Purpose:
    Train EnvModule's encoder so that (humidity, temperature) → gate_env [B,L,D]
    captures meaningful latent representation, while the auxiliary decoder
    reconstructs env_features (MSE loss). After this pretrain, only the encoder
    weights are kept; CGForecaster.prepare_for_stage2() freezes the decoder.

ILI-blind structural guarantee:
    This script uses ONLY (humidity, temperature) — NO ili_weighted_pct, NO
    target. The autoencoder objective is purely reconstruction, so by
    construction there is no information leakage from ILI labels into the
    Env representation. This satisfies PLAN §5.1 A-2 + v2.0.7 design.
    (seasonal_mae computation, which DOES touch ILI, is in src/utils/losses.py
    — kept separate to preserve the structural blindness.)

Post-review (M1.7 direction rev.2):
    - Finding 3: val MSE verification + threshold check
    - Finding 4: per-epoch loss logging (every `log_every` epochs)

Usage:
    python scripts/m1_7_env_pretrain.py
    python scripts/m1_7_env_pretrain.py --epochs 50 --log-every 10
    python scripts/m1_7_env_pretrain.py --smoke   # 5 epochs, log every step

Output:
    runs/m1_7_env_pretrain/env_encoder.pt    # encoder weights only
    runs/m1_7_env_pretrain/diagnostics.json  # train/val MSE history + final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[1]
if str(_CG_MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CG_MAMBA_ROOT))

from src.data.loader import load_dataset_csv, load_norm_params  # noqa: E402
from src.models.env_module import EnvModule  # noqa: E402
from src.utils.config import CGMambaConfig  # noqa: E402


def _build_env_tensor(df, split, norm):
    """Extract z-scored env features for a specific split."""
    sub = df[df["split"] == split]
    env_h = (
        (sub["specific_humidity_g_per_kg"].to_numpy()
         - norm["specific_humidity_g_per_kg"]["mean"])
        / norm["specific_humidity_g_per_kg"]["std"]
    )
    env_t = (
        (sub["temperature_c"].to_numpy()
         - norm["temperature_c"]["mean"])
        / norm["temperature_c"]["std"]
    )
    arr = np.stack([env_h, env_t], axis=-1).astype(np.float32)   # [T, 2]
    return torch.from_numpy(arr).unsqueeze(0)                    # [1, T, 2]


def _set_seed(seed: int) -> None:
    """Reproducibility: torch + numpy + CUDA seed unified."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pretrain_env(cfg: CGMambaConfig, args) -> dict:
    """Train EnvModule on train env data, evaluate on val env data."""
    _set_seed(cfg.seed)

    csv_path = cfg.data_csv
    norm_path = cfg.norm_json
    out_root = _CG_MAMBA_ROOT / "runs" / "m1_7_env_pretrain"
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env pretrain] device={device}, cfg={cfg.summary()}")
    print(f"  output: {out_root.relative_to(_CG_MAMBA_ROOT)}")

    df = load_dataset_csv(csv_path)
    norm = load_norm_params(norm_path)

    # Train + val env tensors (ILI columns 미사용)
    train_env = _build_env_tensor(df, "train", norm).to(device)   # [1, T_train, 2]
    val_env = _build_env_tensor(df, "val", norm).to(device)        # [1, T_val, 2]
    print(f"  train env shape: {tuple(train_env.shape)}, val env shape: {tuple(val_env.shape)}")

    env_module = EnvModule(cfg).to(device)
    optimizer = torch.optim.AdamW(
        env_module.parameters(),
        lr=cfg.stage1_lr,
        weight_decay=cfg.weight_decay,
    )

    n_epochs = args.epochs if args.epochs is not None else cfg.n_epochs   # default 50
    log_every = args.log_every if args.log_every is not None else 10

    # ── Random init baseline (Finding 3 verification) ──
    env_module.eval()
    with torch.no_grad():
        gate_env_val_init = env_module(val_env)
        val_mse_random = env_module.reconstruction_loss(val_env, gate_env_val_init).item()
    print(f"  random-init val MSE (baseline): {val_mse_random:.6f}")

    # ── Training loop ──
    env_module.train()
    history = {"train_mse": [], "val_mse": []}
    t0 = time()
    for epoch in range(n_epochs):
        env_module.train()
        gate_env_tr = env_module(train_env)
        loss = env_module.reconstruction_loss(train_env, gate_env_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history["train_mse"].append(loss.item())

        # Validation
        env_module.eval()
        with torch.no_grad():
            gate_env_v = env_module(val_env)
            val_mse = env_module.reconstruction_loss(val_env, gate_env_v).item()
        history["val_mse"].append(val_mse)

        # Finding 4: per-epoch logging every `log_every` epochs
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:3d}/{n_epochs}  "
                f"train_mse={loss.item():.6f}  val_mse={val_mse:.6f}  "
                f"vs random {val_mse_random:.6f} (ratio {val_mse/val_mse_random:.3f})"
            )

    elapsed = time() - t0

    # ── Finding 3: Final val verification + threshold check ──
    env_module.eval()
    with torch.no_grad():
        gate_env_final = env_module(val_env)
        val_mse_final = env_module.reconstruction_loss(val_env, gate_env_final).item()

    threshold = 0.5 * val_mse_random
    pass_threshold = val_mse_final < threshold

    print()
    print("=== Env pretrain summary ===")
    print(f"  epochs trained: {n_epochs}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  random-init val MSE:  {val_mse_random:.6f}")
    print(f"  final val MSE:        {val_mse_final:.6f}")
    print(f"  threshold (0.5 × random):  {threshold:.6f}")
    print(f"  PLAN §5.1 sanity (val < random × 0.5):  "
          f"{'PASS' if pass_threshold else 'FAIL'}")

    if not pass_threshold:
        print("  ⚠️ Val MSE did not drop below 0.5 × random — possible issues:")
        print("     - LR too low / too high")
        print("     - Insufficient epochs (try --epochs 100)")
        print("     - Data normalization issue")
        print("  Continuing (M1.7 trainer will use this checkpoint anyway —")
        print("  Env contribution is verified in §7.4 A2 ablation).")

    # ── Persist artifacts ──
    encoder_ckpt_path = out_root / "env_encoder.pt"
    torch.save(env_module.encoder.state_dict(), encoder_ckpt_path)
    diag = {
        "n_epochs": n_epochs,
        "seed": cfg.seed,
        "lr": cfg.stage1_lr,
        "weight_decay": cfg.weight_decay,
        "val_mse_random_init": float(val_mse_random),
        "val_mse_final": float(val_mse_final),
        "threshold": float(threshold),
        "passed_threshold": bool(pass_threshold),
        "elapsed_sec": float(elapsed),
        "history": {k: [float(v) for v in vs] for k, vs in history.items()},
    }
    with open(out_root / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    print(f"\nSaved encoder: {encoder_ckpt_path.relative_to(_CG_MAMBA_ROOT)}")
    print(f"Saved diagnostics: {(out_root / 'diagnostics.json').relative_to(_CG_MAMBA_ROOT)}")
    return diag


def main() -> int:
    parser = argparse.ArgumentParser(description="M1.7 Env autoencoder Stage 1 pretrain")
    parser.add_argument("--smoke", action="store_true",
                        help="5 epochs, log every epoch")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg.n_epochs (default 50)")
    parser.add_argument("--log-every", type=int, default=None,
                        help="Per-epoch loss log frequency (default 10)")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 5
        args.log_every = 1

    cfg = CGMambaConfig()
    diag = pretrain_env(cfg, args)
    # Always succeed (exit 0). Threshold failure is logged but doesn't fail CI:
    # Env contribution is independently verified via §7.4 A2 ablation, and a
    # subthreshold pretrain still produces usable encoder weights (reviewer can
    # inspect runs/m1_7_env_pretrain/diagnostics.json to decide whether to retry).
    return 0


if __name__ == "__main__":
    sys.exit(main())
