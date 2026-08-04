"""M1.8 Stage 3 — HMM Joint Fine-tune (PLAN v2.0.9 PATCH 11 / D.5.3).

Spec (PLAN §5.1 D.5.3):
    - Stage 2 best.pt에서 모델 가중치 + HMM cache 복원
    - PhaseModule._A (transition), _means (Gaussian means)를 buffer → Parameter 전환
        (π, Σ, cov_inv, log_det는 over-constrained로 buffer 유지)
    - LR 분리: encoder/decoder = 1e-4, HMM (_A, _means) = 1e-5
    - 10 epoch fine-tune
    - κ monitoring: state flipping rate > 30% (= cohens_kappa_aligned < 0.7) 또는
      dead state 발생 시 rollback (직전 epoch state 복원 + 학습 중단)

Mandatory entry sequence (PATCH 11 + T-2 + New-M6):
    1. model = CGForecaster(cfg); model.prepare_for_stage2(hmm)
    2. model.load_state_dict(stage2_best.pt)
    3. initial Viterbi 저장 (κ rollback 기준점)
    4. n_unfrozen = model.phase_module._unfreeze_for_stage3()
    5. optimizer 재구성 (분리 LR)  ← 필수, model.parameters() snapshot 갱신
    6. 10ep train loop with κ check

CLI:
    python scripts/m1_8_stage3_train.py --smoke           # 3 epoch
    python scripts/m1_8_stage3_train.py                    # PLAN D.5.3 default 10ep
    python scripts/m1_8_stage3_train.py --stage2-dir runs/m1_7_train/stage2_full_v3_hmm_init

Output:
    runs/m1_8_stage3_train/<run_name>/
        best.pt              # best-val checkpoint
        history.json         # per-epoch loss + κ + dead state status
        config.json
        final_metrics.json   # Stage 2 vs Stage 3 비교
        rollback.json        # rollback 발동 시 사유 기록
"""
from __future__ import annotations

import argparse
import copy
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
from src.utils.checkpoints import load_fitted_hmm  # noqa: E402
from src.utils.config import CGMambaConfig  # noqa: E402
from src.utils.losses import cg_mamba_loss, compute_seasonal_mae, mase_loss  # noqa: E402
from src.utils.metrics import cohens_kappa_aligned, state_occupancy  # noqa: E402


_DEFAULT_STAGE2_DIR = _CG_MAMBA_ROOT / "runs" / "m1_7_train" / "stage2_full_v3_hmm_init"
_DEFAULT_HMM_DIR = (
    _CG_MAMBA_ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed42"
)
_DEFAULT_ENV_CKPT = _CG_MAMBA_ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"

# Stage 3 κ rollback thresholds (PLAN D.5.3)
KAPPA_ROLLBACK_THRESHOLD = 0.7      # state flipping rate < 30% (κ_aligned ≥ 0.7)
DEAD_STATE_THRESHOLD = 0.05         # any state occupancy < 5% → dead


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_loaders(cfg, batch_size, seed):
    df = load_dataset_csv(cfg.data_csv)
    norm = load_norm_params(cfg.norm_json)
    horizons = tuple(cfg.horizons)
    train_ds = MultiHorizonDataset(df, "train", cfg.lookback, horizons, norm)
    val_ds = MultiHorizonDataset(df, "val", cfg.lookback, horizons, norm)
    test_ds = MultiHorizonDataset(df, "test", cfg.lookback, horizons, norm)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_dict, generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_dict, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_dict, drop_last=False)
    return df, norm, train_loader, val_loader, test_loader


def _compute_global_viterbi(model, train_dataset, device, batch_size=32) -> np.ndarray:
    """Stage 3 κ rollback 기준: 전체 train data에 대한 Viterbi state assignment.

    각 window의 phase_post.argmax(-1)를 누적 → state flipping 측정 용.
    L-1 alignment 고려 (PhaseModule output은 L-1 length).

    Deterministic ordering (shuffle=False): κ는 sequence alignment 기반이므로
    매 호출마다 같은 timestep 순서로 누적되어야 비교 가능. train_loader (shuffle=True)
    그대로 쓰면 매번 다른 batch 순서로 누적되어 같은 model에서도 κ가 chance level로
    나타남 (버그).
    """
    deterministic_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_dict, drop_last=False,
    )
    model.eval()
    all_states = []
    with torch.no_grad():
        for batch in deterministic_loader:
            x = batch["x"].to(device)
            # PhaseModule이 x[:, :, :V_hmm_raw]를 받음 (CGForecaster Step 1 dataflow)
            x_phase = x[:, :, :model.cfg.V_hmm_raw]
            _, phase_post = model.phase_module(x_phase)   # [B, L-1, K]
            argmax = phase_post.argmax(dim=-1)             # [B, L-1]
            all_states.append(argmax.cpu().numpy().flatten())
    return np.concatenate(all_states)                       # [N_total]


def _build_stage3_optimizer(model, cfg):
    """Stage 3 optimizer with 4-group bijection (v2.1.7-A).

    Groups (PLAN D.5.3):
        hmm             : phase_module._A, phase_module._means       LR = cfg.stage3_hmm_lr
        state_embed     : phase_module.state_embeddings              LR = cfg.stage3_state_embed_lr
        env             : env_module.encoder.*  (decoder frozen)     LR = cfg.stage3_env_lr
        encoder_decoder : backbone + gate_proj + outer decoder + ... LR = cfg.stage3_other_lr

    Stage 3 specific groups (hmm / state_embed / env) are isolated so that
    HPO Phase 2 can sweep each LR independently as a ratio × stage3_other_lr.

    v2.1.7 C-1 history: original `_build_stage3_optimizer(model, *, hmm_lr, other_lr)`
    had explicit-kwargs binding that silently overrode HPO monkey-patches → degenerate
    sweep. Fixed by (a) signature `(model, cfg)` mirroring build_stage2_optimizer,
    and (b) 4-group split so each LR has its own cfg field.

    C-3 fix: full set-based ERR-C2 bijection (every trainable param ↔ exactly one group).
    """
    # Build expected name sets
    expected_hmm_names = {"phase_module._A", "phase_module._means"}
    expected_state_embed_names = {"phase_module.state_embeddings"}

    # Bucket trainable params into 4 groups
    hmm_params, state_embed_params, env_params, other_params = [], [], [], []
    hmm_names_seen, se_names_seen, env_names_seen, other_names = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in expected_hmm_names:
            hmm_params.append(p); hmm_names_seen.append(n)
        elif n in expected_state_embed_names:
            state_embed_params.append(p); se_names_seen.append(n)
        elif n.startswith("env_module."):
            env_params.append(p); env_names_seen.append(n)
        else:
            other_params.append(p); other_names.append(n)

    # ── ERR-C2: 4-group bijection check ──
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assigned = set(hmm_names_seen) | set(se_names_seen) | set(env_names_seen) | set(other_names)
    missing_hmm = expected_hmm_names - set(hmm_names_seen)
    missing_se  = expected_state_embed_names - set(se_names_seen)
    unassigned = trainable - assigned
    over_assigned = assigned - trainable

    if missing_hmm:
        raise RuntimeError(
            f"Stage 3 ERR-C2: HMM params not unfrozen — {missing_hmm}. "
            f"Did _unfreeze_for_stage3() run? Trainable: {trainable}"
        )
    if missing_se:
        raise RuntimeError(
            f"Stage 3 ERR-C2: state_embed param missing — {missing_se}. "
            f"Trainable: {trainable}"
        )
    if unassigned or over_assigned:
        raise RuntimeError(
            f"Stage 3 ERR-C2 mismatch:\n"
            f"  unassigned (trainable but in no group): {unassigned}\n"
            f"  over-assigned (in group but not trainable): {over_assigned}"
        )
    total_grouped = len(hmm_params) + len(state_embed_params) + len(env_params) + len(other_params)
    if total_grouped != len(trainable):
        raise RuntimeError(
            f"Stage 3 ERR-C2 duplicate: group total {total_grouped} ≠ trainable {len(trainable)}"
        )
    assert len(hmm_params) == 2, f"Expected 2 HMM params (_A, _means), got {len(hmm_params)}"
    assert len(state_embed_params) == 1, f"Expected 1 state_embed param, got {len(state_embed_params)}"
    if len(env_params) == 0:
        # env_module decoder frozen in Stage 2; encoder should remain trainable.
        # If env_params is empty something is off — likely env was over-frozen.
        raise RuntimeError(
            "Stage 3 ERR-C2: env group is empty — env_module.encoder should be trainable. "
            "Check freeze_decoder_for_stage2() did not over-freeze."
        )

    return torch.optim.AdamW([
        {"name": "encoder_decoder", "params": other_params,       "lr": cfg.stage3_other_lr},
        {"name": "env",             "params": env_params,         "lr": cfg.stage3_env_lr},
        {"name": "state_embed",     "params": state_embed_params, "lr": cfg.stage3_state_embed_lr},
        {"name": "hmm",             "params": hmm_params,         "lr": cfg.stage3_hmm_lr},
    ]), {"n_other": len(other_params), "n_env": len(env_params),
         "n_state_embed": 1, "n_hmm": 2}


def _train_one_epoch(model, loader, optimizer, seasonal_mae, device,
                     grad_clip=1.0, lambda_mase=0.3):
    model.train()
    n_total = 0
    loss_sum = 0.0
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
        n_total += B
    return loss_sum / max(n_total, 1)


def _evaluate(model, loader, seasonal_mae, device, lambda_mase=0.3):
    model.eval()
    n_total = 0
    mse_sum = 0.0
    mase_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            y = batch["y"].to(device)
            pred = model(x, env)
            B = y.size(0)
            mse_sum += torch.nn.functional.mse_loss(pred, y).item() * B
            mase_sum += mase_loss(pred, y, seasonal_mae).item() * B
            n_total += B
    mse_mean = mse_sum / max(n_total, 1)
    mase_mean = mase_sum / max(n_total, 1)
    return {"mse": mse_mean, "mase": mase_mean,
            "total": mse_mean + lambda_mase * mase_mean}


def stage3_train(cfg, args) -> dict:
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = _CG_MAMBA_ROOT / "runs" / "m1_8_stage3_train" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    # ── 1. Data + HMM + base model ──
    df, norm, train_loader, val_loader, test_loader = _build_loaders(
        cfg, args.batch_size, cfg.seed
    )
    n_train = len(train_loader.dataset)
    print(f"[M1.8 Stage 3] device={device}")
    print(f"  windows: train={n_train}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    seasonal_mae = compute_seasonal_mae(df, norm).to(device)

    model = CGForecaster(cfg).to(device)
    if args.env_encoder_ckpt and Path(args.env_encoder_ckpt).exists():
        state = torch.load(args.env_encoder_ckpt, map_location=device, weights_only=True)
        model.env_module.encoder.load_state_dict(state)
    hmm = load_fitted_hmm(Path(args.hmm_dir))
    model.prepare_for_stage2(hmm)
    print(f"  HMM loaded: K={hmm.K}, V={hmm.V}, reg_covar={hmm.reg_covar:.0e}")

    # ── 2. Load Stage 2 best.pt ──
    stage2_ckpt = Path(args.stage2_dir) / "best.pt"
    assert stage2_ckpt.exists(), f"Stage 2 best.pt not found: {stage2_ckpt}"
    ckpt = torch.load(stage2_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Stage 2 ckpt loaded from {args.stage2_dir} (best_epoch={ckpt['epoch']+1}, val_total={ckpt['val_total']:.4f})")
    stage2_val_total = ckpt["val_total"]

    # ── 3. Initial Viterbi (κ rollback baseline) ──
    initial_viterbi = _compute_global_viterbi(model, train_loader.dataset, device, batch_size=args.batch_size)
    initial_occ = state_occupancy(initial_viterbi, K=cfg.K_phase)
    print(f"  initial Viterbi: occupancy={[f'{o:.3f}' for o in initial_occ]}, "
          f"n_states_used={len(np.unique(initial_viterbi))}")

    # ── 4. Stage 3 unfreeze ──
    n_unfrozen = model.phase_module._unfreeze_for_stage3()
    pre_unfreeze_total = sum(p.numel() for p in model.parameters() if p.requires_grad) - n_unfrozen
    print(f"  Stage 3 unfreeze: _A + _means = {n_unfrozen} new trainable params "
          f"(total trainable: {pre_unfreeze_total} → {pre_unfreeze_total + n_unfrozen})")

    # ── 5. Rebuild optimizer (v2.1.7-A: 4-group split) ──
    optimizer, group_counts = _build_stage3_optimizer(model, cfg)
    print(f"  Stage 3 optimizer (4-group): "
          f"encoder_decoder({group_counts['n_other']}, LR={cfg.stage3_other_lr:.0e}) + "
          f"env({group_counts['n_env']}, LR={cfg.stage3_env_lr:.0e}) + "
          f"state_embed({group_counts['n_state_embed']}, LR={cfg.stage3_state_embed_lr:.0e}) + "
          f"hmm({group_counts['n_hmm']}, LR={cfg.stage3_hmm_lr:.0e})")

    n_epochs = 3 if args.smoke else args.epochs
    print(f"  Training: {n_epochs} epoch (Stage 2 baseline val_total={stage2_val_total:.4f})\n")

    # ── 6. Training loop with κ monitoring + optional early-stop patience ──
    history = {"train_total": [], "val_total": [], "val_mse": [], "val_mase": [],
               "kappa_vs_initial": [], "occupancy": [], "dead_state": []}
    best_val_total = math.inf
    best_epoch = -1
    best_state_dict = None
    rollback = None
    early_stop = None
    # patience: 0 (default) → no early stop, run full n_epochs
    patience = getattr(args, "patience", 0)
    epochs_since_improvement = 0

    prev_state_dict = copy.deepcopy(model.state_dict())  # epoch 시작 시점 backup

    t0 = time()
    for epoch in range(n_epochs):
        tr_loss = _train_one_epoch(model, train_loader, optimizer, seasonal_mae, device,
                                    grad_clip=cfg.grad_clip)
        va = _evaluate(model, val_loader, seasonal_mae, device)

        # κ monitoring
        cur_viterbi = _compute_global_viterbi(model, train_loader.dataset, device, batch_size=args.batch_size)
        cur_occ = state_occupancy(cur_viterbi, K=cfg.K_phase)
        kappa = cohens_kappa_aligned(initial_viterbi, cur_viterbi, K=cfg.K_phase)
        dead_state = any(o < DEAD_STATE_THRESHOLD for o in cur_occ)

        history["train_total"].append(tr_loss)
        history["val_total"].append(va["total"])
        history["val_mse"].append(va["mse"])
        history["val_mase"].append(va["mase"])
        history["kappa_vs_initial"].append(float(kappa))
        history["occupancy"].append([float(o) for o in cur_occ])
        history["dead_state"].append(bool(dead_state))

        improved = va["total"] < best_val_total
        if improved:
            best_val_total = va["total"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        tag = " *best" if improved else ""

        print(f"  ep {epoch+1:2d}/{n_epochs}  "
              f"train={tr_loss:.4f}  val={va['total']:.4f} (mse={va['mse']:.4f}, mase={va['mase']:.4f})  "
              f"κ={kappa:.4f}  occ={[f'{o:.3f}' for o in cur_occ]}  "
              f"dead={dead_state}{tag}")

        # Rollback check (HMM safety guard)
        if kappa < KAPPA_ROLLBACK_THRESHOLD or dead_state:
            reason = []
            if kappa < KAPPA_ROLLBACK_THRESHOLD:
                reason.append(f"κ={kappa:.4f} < {KAPPA_ROLLBACK_THRESHOLD}")
            if dead_state:
                reason.append(f"dead state (min occ={min(cur_occ):.4f})")
            print(f"  ⚠ ROLLBACK trigger: {', '.join(reason)}")
            print(f"     restoring state from epoch {epoch} → halting training")
            model.load_state_dict(prev_state_dict)
            rollback = {"epoch": epoch + 1, "reason": reason,
                        "kappa": float(kappa),
                        "occupancy": [float(o) for o in cur_occ]}
            break

        # Early stopping (patience > 0)
        if patience > 0 and epochs_since_improvement >= patience:
            print(f"  ⏹ EARLY STOP at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs, best at ep {best_epoch+1})")
            early_stop = {"epoch": epoch + 1, "best_epoch": best_epoch + 1,
                          "no_improve_count": epochs_since_improvement}
            break

        # Snapshot for next iteration's potential rollback
        prev_state_dict = copy.deepcopy(model.state_dict())

    elapsed = time() - t0

    # ── 7a. Save LAST epoch weights BEFORE loading best (for leakage-free eval) ──
    last_epoch_idx = len(history["val_total"]) - 1
    last_val_total = history["val_total"][-1] if history["val_total"] else math.inf
    torch.save({
        "epoch": last_epoch_idx,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "val_total": last_val_total,
        "stage2_val_total": stage2_val_total,
        "rollback": rollback,
        "note": "last-epoch weights (no val-based selection)",
    }, out_root / "last.pt")

    # ── 7. Final eval at best ──
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test = _evaluate(model, test_loader, seasonal_mae, device)

    # ── 8. Save ──
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "val_total": best_val_total,
        "stage2_val_total": stage2_val_total,
        "rollback": rollback,
    }, out_root / "best.pt")
    with open(out_root / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_root / "rollback.json", "w") as f:
        json.dump({"rollback_triggered": rollback is not None,
                   "details": rollback}, f, indent=2)
    final = {
        "best_epoch": best_epoch,
        "best_val_total": best_val_total,
        "stage2_val_total": stage2_val_total,
        "delta_val_total": best_val_total - stage2_val_total,
        "test_mse": test["mse"], "test_mase": test["mase"], "test_total": test["total"],
        "rollback_triggered": rollback is not None,
        "early_stop_triggered": early_stop is not None,
        "early_stop_details": early_stop,
        "patience": patience,
        "n_epochs_configured": n_epochs,
        "n_epochs_run": len(history["train_total"]),
        "elapsed_sec": float(elapsed),
        "final_kappa": history["kappa_vs_initial"][-1] if history["kappa_vs_initial"] else None,
    }
    with open(out_root / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)

    # ── Summary ──
    print("\n=== Stage 3 summary ===")
    print(f"  Stage 2 baseline val_total: {stage2_val_total:.4f}")
    print(f"  Stage 3 best val_total:     {best_val_total:.4f} (epoch {best_epoch+1})")
    print(f"  Δval_total:                 {best_val_total - stage2_val_total:+.4f}  "
          f"({'개선' if best_val_total < stage2_val_total else '악화/동등'})")
    print(f"  test (best):                mse={test['mse']:.4f}  mase={test['mase']:.4f}")
    print(f"  final κ vs initial:         {history['kappa_vs_initial'][-1]:.4f}")
    print(f"  rollback:                   {'TRIGGERED' if rollback else 'No'}")
    print(f"  elapsed:                    {elapsed:.1f}s")
    print(f"\nSaved: {out_root.relative_to(_CG_MAMBA_ROOT)}/")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="M1.8 Stage 3 — HMM Joint Fine-tune")
    parser.add_argument("--smoke", action="store_true", help="3 epoch sanity")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Stage 3 epochs (PLAN D.5.3 default 10; raise + patience for HPO)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stop patience on val_total (0 = disabled, run full epochs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage2-dir", type=str, default=str(_DEFAULT_STAGE2_DIR),
                        help="Stage 2 run directory with best.pt")
    parser.add_argument("--hmm-dir", type=str, default=str(_DEFAULT_HMM_DIR))
    parser.add_argument("--env-encoder-ckpt", type=str, default=str(_DEFAULT_ENV_CKPT))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"smoke_{ts}" if args.smoke else f"stage3_{ts}"

    cfg = CGMambaConfig()
    final = stage3_train(cfg, args)
    return 0 if final["best_val_total"] < math.inf else 1


if __name__ == "__main__":
    sys.exit(main())
