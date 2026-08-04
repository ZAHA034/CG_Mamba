"""Ablation Retrain — from-scratch utility ablations (4 configs × 5 seeds).

Trains 4 architectural ablations FROM SCRATCH (no post-hoc, no inference-only
gate disable) to measure component UTILITY for §IV.7 C7 causal claim.

Configs:
  no_env           : gate_env replaced by ones BEFORE training
                       → context_vec = gate_phase only.
                       → EnvModule remains in model but is bypassed.
  no_phase         : gate_phase replaced by ones BEFORE training (clean single
                       component; phase-side mirror of no_env)
                       → context_vec = gate_env only.
                       → HMM PhaseModule still active in decoder rollout & UQ;
                         ONLY the encoder phase-gate on Mamba selectivity is
                         bypassed. Retrained counterpart of the post-hoc
                         "-PhaseModule (enc gate) only" diagnostic.
  no_encgates      : context_vec=None passed to encoder during training
                       → encoder vanilla path (disable_gate semantics).
                       → HMM/PhaseModule still active in decoder rollout & UQ.
                       → NOT equivalent to Vanilla Mamba (Table I): rollout retained.
  uniform_rollout  : phase_module.rollout returns uniform 1/K BEFORE training
                       → decoder receives phase-blind gamma_all.
                       → Encoder gate composition unchanged.

HPO freeze (Reviewer concern 6a):
  All configs use the same HPO as Full CG-Mamba M2.1 winner:
    gate_lr=1e-3, backbone_lr=1e-4, lookback=104,
    hmm_lr_ratio=0.01, state_embed_lr_ratio=0.01, env_lr_ratio=0.001
  HMM Stage 1 ckpts reused per seed (m1_4_phase_dynamics_main/...).
  Env Stage 1 ckpt reused (m1_7_env_pretrain/env_encoder.pt) — except no_env.

Output:
  runs/m1_7_train/ablation_retrain_<config>_s<seed>_stage2/
  runs/m1_8_stage3_train/ablation_retrain_<config>_s<seed>_stage3/
  runs/ablation_retrain/manifest.json
  runs/ablation_retrain/<config>_progress.log

CLI:
  python scripts/ablation_retrain.py --launch-all --device cuda:0
  python scripts/ablation_retrain.py --ablation no_env --seed 42 --device cuda:0

This file does NOT modify any core source file. CGForecaster.forward is overridden
in subclasses; the m1_7_train / m1_8_stage3_train modules are monkey-patched
at module level before their main training functions are called.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─── Core imports (CGForecaster is subclassed, NOT modified) ───
from src.models.cg_forecaster import CGForecaster  # noqa: E402
from src.utils.config import CGMambaConfig, ILI_TARGET_IDX  # noqa: E402

# HPO winner (from runs/ablation_a4 — proven full CG-Mamba config)
CG_TOP1_HP = {
    "gate_lr":               1e-3,
    "backbone_lr":           1e-4,
    "lookback":              104,
    "hmm_lr_ratio":          0.01,
    "state_embed_lr_ratio":  0.01,
    "env_lr_ratio":          0.001,
}
OTHER_LR_BASE = 1e-4
HMM_DIR_TEMPLATE = _ROOT / "runs" / "m1_4_phase_dynamics_main" / "V_raw3_regcov5e-03_K3_seed{seed}"
ENV_CKPT = _ROOT / "runs" / "m1_7_env_pretrain" / "env_encoder.pt"

SEEDS = (42, 123, 456, 789, 1024)
ABLATIONS = ("no_env", "no_phase", "no_encgates", "uniform_rollout", "full")

OUT_ROOT = _ROOT / "runs" / "ablation_retrain"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Ablation subclasses — override forward() with surgical modifications
# ──────────────────────────────────────────────────────────────────────────
def _validate_forward_inputs(self, x: torch.Tensor, env: torch.Tensor) -> None:
    """Lift from parent forward for defensive shape checks."""
    if x.dim() != 3 or x.shape[-1] != self.cfg.main_input_dim:
        raise RuntimeError(
            f"x expected [B, L, main_input_dim={self.cfg.main_input_dim}], got {tuple(x.shape)}"
        )
    if env.dim() != 3 or env.shape[-1] != self.cfg.env_input_dim:
        raise RuntimeError(
            f"env expected [B, L, env_input_dim={self.cfg.env_input_dim}], got {tuple(env.shape)}"
        )
    if x.shape[:2] != env.shape[:2]:
        raise RuntimeError(f"x/env batch-seq mismatch: x {tuple(x.shape[:2])} vs env {tuple(env.shape[:2])}")
    if x.shape[1] < 2:
        raise RuntimeError(f"L={x.shape[1]} < 2 — PhaseModule augmentation requires L >= 2")


class NoEnvCGForecaster(CGForecaster):
    """Ablation: gate_env replaced by ones during training.

    Effect: context_vec = gate_phase * ones = gate_phase.
    EnvModule remains in the model graph (so checkpoint loading is compatible)
    but its output is bypassed. Its gradients are still computed but have no
    downstream effect, so weights drift slowly with optimizer noise — acceptable
    for utility measurement.
    """
    def forward(self, x, env, return_intermediates=False):
        _validate_forward_inputs(self, x, env)
        # Step 1: PhaseModule
        x_phase = x[:, :, :self.cfg.V_hmm_raw]
        gate_phase, phase_post = self.phase_module(x_phase)        # [B,L-1,D], [B,L-1,K]
        # Step 2 (ABLATED): gate_env := ones (env_module bypassed)
        B, L, _ = x.shape
        d_model = self.cfg.d_model
        gate_env = torch.ones(B, L, d_model, device=x.device, dtype=gate_phase.dtype)
        # Step 4: AND composition → context_vec = gate_phase (since gate_env=1)
        x_truncated = x[:, 1:, :]
        env_truncated_g = gate_env[:, 1:, :]
        context_vec = gate_phase * env_truncated_g                  # == gate_phase
        # Step 5: Encoder
        fused = self.encoder(x_truncated, context_vec=context_vec)
        # Step 6: Decoder (HMM rollout & UQ unchanged)
        gamma_last = phase_post[:, -1, :]
        W = min(self.cfg.rollout_window, x_phase.shape[1])
        x_window = x_phase[:, -W:, :]
        last_value_normalized = x[:, -1, ILI_TARGET_IDX]
        gamma_all = self.phase_module.rollout(gamma_last, x_window, H=self.decoder.max_horizon)
        predictions = self.decoder(
            encoder_out=fused, last_value_normalized=last_value_normalized,
            gamma_all=gamma_all, state_embeddings=self.phase_module.state_embeddings,
        )
        if return_intermediates:
            return predictions, self._compute_intermediates(
                gate_phase=gate_phase, phase_post=phase_post, context_vec=context_vec,
                gamma_last=gamma_last, gamma_all=gamma_all, fused=fused,
            )
        return predictions


class NoPhaseGateCGForecaster(CGForecaster):
    """Ablation: gate_phase replaced by ones during training (clean single-component).

    Effect: context_vec = ones * gate_env = gate_env.
    The HMM PhaseModule remains fully active — its posterior still drives the
    decoder rollout and APMD UQ (gamma_all); ONLY the encoder phase-gate that
    modulates Mamba selectivity is bypassed. This isolates the PhaseModule's
    encoder-gate contribution as a clean single-component removal (the retrained
    counterpart of the post-hoc "-PhaseModule (enc gate) only" diagnostic), and
    is the phase-side mirror of NoEnvCGForecaster.
    """
    def forward(self, x, env, return_intermediates=False):
        _validate_forward_inputs(self, x, env)
        # Step 1: PhaseModule (still active — posterior needed for decoder rollout & UQ)
        x_phase = x[:, :, :self.cfg.V_hmm_raw]
        gate_phase_real, phase_post = self.phase_module(x_phase)    # [B,L-1,D], [B,L-1,K]
        # Step 1b (ABLATED): gate_phase := ones (encoder phase-gate bypassed)
        gate_phase = torch.ones_like(gate_phase_real)
        # Step 2: EnvModule (active)
        gate_env = self.env_module(env)
        # Step 4: AND composition → context_vec = gate_env (since gate_phase=1)
        x_truncated = x[:, 1:, :]
        env_truncated_g = gate_env[:, 1:, :]
        context_vec = gate_phase * env_truncated_g                  # == gate_env truncated
        # Step 5: Encoder
        fused = self.encoder(x_truncated, context_vec=context_vec)
        # Step 6: Decoder (HMM rollout & UQ unchanged — uses REAL phase_post)
        gamma_last = phase_post[:, -1, :]
        W = min(self.cfg.rollout_window, x_phase.shape[1])
        x_window = x_phase[:, -W:, :]
        last_value_normalized = x[:, -1, ILI_TARGET_IDX]
        gamma_all = self.phase_module.rollout(gamma_last, x_window, H=self.decoder.max_horizon)
        predictions = self.decoder(
            encoder_out=fused, last_value_normalized=last_value_normalized,
            gamma_all=gamma_all, state_embeddings=self.phase_module.state_embeddings,
        )
        if return_intermediates:
            return predictions, self._compute_intermediates(
                gate_phase=gate_phase, phase_post=phase_post, context_vec=context_vec,
                gamma_last=gamma_last, gamma_all=gamma_all, fused=fused,
            )
        return predictions


class NoEncGatesCGForecaster(CGForecaster):
    """Ablation: context_vec=None passed to encoder (vanilla path).

    Effect: Encoder runs ContextGatedMambaBlock with disable_gate semantics
    (bit-identical to vanilla Mamba per test_context_gated_mamba.py:48).
    HMM/PhaseModule remains active in decoder rollout & gamma_all-based UQ.

    LABELING NOTE: This is NOT equivalent to Vanilla Mamba (Table I) because
    HMM phase context still flows through the decoder. Call this "CG-Mamba w/o
    Encoder Gates" — NOT "Vanilla".
    """
    def forward(self, x, env, return_intermediates=False):
        _validate_forward_inputs(self, x, env)
        # Step 1: PhaseModule (still active — needed for decoder rollout)
        x_phase = x[:, :, :self.cfg.V_hmm_raw]
        gate_phase, phase_post = self.phase_module(x_phase)
        # Step 2: EnvModule (still active — needed for intermediate diagnostics)
        gate_env = self.env_module(env)
        # Step 4 (ABLATED): context_vec := None
        x_truncated = x[:, 1:, :]
        env_truncated_g = gate_env[:, 1:, :]
        context_vec_diag = gate_phase * env_truncated_g     # kept for diagnostics only
        # Step 5: Encoder with context_vec=None (vanilla path)
        fused = self.encoder(x_truncated, context_vec=None)
        # Step 6: Decoder (HMM rollout unchanged)
        gamma_last = phase_post[:, -1, :]
        W = min(self.cfg.rollout_window, x_phase.shape[1])
        x_window = x_phase[:, -W:, :]
        last_value_normalized = x[:, -1, ILI_TARGET_IDX]
        gamma_all = self.phase_module.rollout(gamma_last, x_window, H=self.decoder.max_horizon)
        predictions = self.decoder(
            encoder_out=fused, last_value_normalized=last_value_normalized,
            gamma_all=gamma_all, state_embeddings=self.phase_module.state_embeddings,
        )
        if return_intermediates:
            return predictions, self._compute_intermediates(
                gate_phase=gate_phase, phase_post=phase_post, context_vec=context_vec_diag,
                gamma_last=gamma_last, gamma_all=gamma_all, fused=fused,
            )
        return predictions


class UniformRolloutCGForecaster(CGForecaster):
    """Ablation: phase_module.rollout returns uniform 1/K gamma_all.

    Effect: Decoder receives phase-blind multi-horizon prior.
    Encoder gates (gate_phase * gate_env) remain unchanged → context_vec normal.
    LOGIC-1 confidence-based eff_gate fallback handles uniform γ by design.

    LABELING: "CG-Mamba w/ uniform decoder rollout". Encoder gates intact.
    """
    def forward(self, x, env, return_intermediates=False):
        _validate_forward_inputs(self, x, env)
        # Steps 1, 2, 4, 5: unchanged
        x_phase = x[:, :, :self.cfg.V_hmm_raw]
        gate_phase, phase_post = self.phase_module(x_phase)
        gate_env = self.env_module(env)
        x_truncated = x[:, 1:, :]
        env_truncated_g = gate_env[:, 1:, :]
        context_vec = gate_phase * env_truncated_g
        fused = self.encoder(x_truncated, context_vec=context_vec)
        # Step 6 (ABLATED): gamma_all := uniform 1/K
        gamma_last = phase_post[:, -1, :]
        H_max = self.decoder.max_horizon
        K = self.cfg.K_phase
        B = x.shape[0]
        gamma_all = torch.full(
            (B, H_max, K), 1.0 / K,
            device=x.device, dtype=gamma_last.dtype,
        )
        last_value_normalized = x[:, -1, ILI_TARGET_IDX]
        predictions = self.decoder(
            encoder_out=fused, last_value_normalized=last_value_normalized,
            gamma_all=gamma_all, state_embeddings=self.phase_module.state_embeddings,
        )
        if return_intermediates:
            return predictions, self._compute_intermediates(
                gate_phase=gate_phase, phase_post=phase_post, context_vec=context_vec,
                gamma_last=gamma_last, gamma_all=gamma_all, fused=fused,
            )
        return predictions


_SUBCLASS_REGISTRY = {
    "no_env":          NoEnvCGForecaster,
    "no_phase":        NoPhaseGateCGForecaster,
    "no_encgates":     NoEncGatesCGForecaster,
    "uniform_rollout": UniformRolloutCGForecaster,
    "full":            CGForecaster,   # baseline retrain (no forward override) — for harness-confound check
}


# ──────────────────────────────────────────────────────────────────────────
# Helpers: cfg + args builders
# ──────────────────────────────────────────────────────────────────────────
def build_frozen_hpo_cfg(seed: int) -> CGMambaConfig:
    """Build cfg with HPO frozen to Full CG-Mamba M2.1 winner."""
    hp = CG_TOP1_HP
    return dataclasses.replace(
        CGMambaConfig(),
        seed=seed,
        dropout=0.0,
        lookback=hp["lookback"],
        stage2_gate_lr=hp["gate_lr"],
        stage2_backbone_lr=hp["backbone_lr"],
        stage3_other_lr=OTHER_LR_BASE,
        stage3_hmm_lr=OTHER_LR_BASE * hp["hmm_lr_ratio"],
        stage3_state_embed_lr=OTHER_LR_BASE * hp["state_embed_lr_ratio"],
        stage3_env_lr=OTHER_LR_BASE * hp["env_lr_ratio"],
    )


def make_stage2_args(run_name: str, hmm_dir: Path, env_ckpt: Path, epochs: int,
                      batch_size: int, smoke: bool = False) -> Namespace:
    """Build the Namespace expected by scripts.m1_7_train.train()."""
    return Namespace(
        smoke=smoke,
        epochs=epochs,
        batch_size=batch_size,
        hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(env_ckpt) if env_ckpt else None,
        wandb_mode="disabled",
        run_name=run_name,
    )


def make_stage3_args(run_name: str, stage2_dir: Path, hmm_dir: Path, env_ckpt: Path,
                     epochs: int, patience: int, batch_size: int, smoke: bool = False) -> Namespace:
    """Build the Namespace expected by scripts.m1_8_stage3_train.stage3_train()."""
    return Namespace(
        smoke=smoke,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        stage2_dir=str(stage2_dir),
        hmm_dir=str(hmm_dir),
        env_encoder_ckpt=str(env_ckpt) if env_ckpt else None,
        run_name=run_name,
    )


# ──────────────────────────────────────────────────────────────────────────
# Monkey-patch driver
# ──────────────────────────────────────────────────────────────────────────
def patch_and_train_one(ablation: str, seed: int, device: str,
                         stage2_epochs: int, stage3_epochs: int, batch_size: int,
                         smoke: bool = False) -> dict:
    """Patch CGForecaster symbol → run Stage 2 + Stage 3 → return summary."""
    subclass = _SUBCLASS_REGISTRY[ablation]
    cfg = build_frozen_hpo_cfg(seed)
    hmm_dir = Path(str(HMM_DIR_TEMPLATE).format(seed=seed))
    if not hmm_dir.exists():
        raise FileNotFoundError(f"Missing HMM Stage 1 ckpt for seed={seed}: {hmm_dir}")

    # Env ckpt is always loaded for compatibility with m1_7/m1_8 (which expect a path).
    # For `no_env` the loaded weights are harmless: the EnvModule's output is bypassed
    # in NoEnvCGForecaster.forward (gate_env=ones), so loaded weights produce no
    # downstream effect during training or evaluation.
    env_ckpt = ENV_CKPT if ENV_CKPT.exists() else None

    # Hoist run_names + ckpt paths for resume check (Patch A)
    stage2_run_name = f"ablation_retrain_{ablation}_s{seed}_stage2"
    stage3_run_name = f"ablation_retrain_{ablation}_s{seed}_stage3"
    stage2_dir = _ROOT / "runs" / "m1_7_train" / stage2_run_name
    stage3_dir = _ROOT / "runs" / "m1_8_stage3_train" / stage3_run_name
    stage2_best = stage2_dir / "best.pt"
    stage3_best = stage3_dir / "best.pt"
    stage2_metrics_p = stage2_dir / "final_metrics.json"
    stage3_metrics_p = stage3_dir / "final_metrics.json"

    # PATCH A — Resume: skip if both Stage 2 + Stage 3 already complete
    if (stage2_best.exists() and stage3_best.exists()
        and stage2_metrics_p.exists() and stage3_metrics_p.exists()):
        try:
            s2 = json.loads(stage2_metrics_p.read_text())
            s3 = json.loads(stage3_metrics_p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: resume-skip aborted (metrics unreadable): {e} — will retrain")
        else:
            print(f"[ablation_retrain] === {ablation} / seed={seed} === RESUME-SKIP ===")
            print(f"  Both Stage 2+3 best.pt + final_metrics.json found, skipping retrain.")
            print(f"  Stage 2 best_val_total = {s2.get('best_val_total', float('nan')):.4f}")
            print(f"  Stage 3 best_val_total = {s3.get('best_val_total', float('nan')):.4f}")
            return {
                "ablation": ablation, "seed": seed, "device": device,
                "stage2_run_name": stage2_run_name, "stage3_run_name": stage3_run_name,
                "stage2_best_val_total": s2.get("best_val_total"),
                "stage3_best_val_total": s3.get("best_val_total"),
                "stage2_elapsed_sec": 0.0, "stage3_elapsed_sec": 0.0,
                "stage2_best_path": str(stage2_best),
                "stage3_best_path": str(stage3_best),
                "resumed": True,
            }

    # Pin device for this run
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    # ── Stage 2 ──
    stage2_args = make_stage2_args(stage2_run_name, hmm_dir, env_ckpt,
                                    epochs=stage2_epochs, batch_size=batch_size,
                                    smoke=smoke)
    print(f"[ablation_retrain] === {ablation} / seed={seed} === Stage 2 ===")
    print(f"  device={device}")
    print(f"  cfg overrides: lookback={cfg.lookback}, gate_lr={cfg.stage2_gate_lr:.1e}, "
          f"backbone_lr={cfg.stage2_backbone_lr:.1e}, seed={cfg.seed}")
    print(f"  HMM dir: {hmm_dir.relative_to(_ROOT)}")
    print(f"  Env ckpt: {env_ckpt}")
    # PATCH C — UniformRollout asymmetry notice (for paper caption traceability)
    if ablation == "uniform_rollout":
        print(f"  NOTE: UniformRollout — phase_module.rollout() is bypassed (gamma_all=1/K).")
        print(f"        Stage 3 fine-tunes PhaseModule._A via ENCODER gradients only;")
        print(f"        decoder rollout receives uniform 1/K throughout Stage 2+3 training.")
        print(f"        This asymmetry is intentional and must be acknowledged in paper caption.")
    elif ablation == "no_encgates":
        print(f"  NOTE: NoEncGates — encoder receives context_vec=None (vanilla path).")
        print(f"        HMM/PhaseModule remains active in DECODER rollout & Method F UQ.")
        print(f"        NOT equivalent to Vanilla Mamba (Table I): label as 'CG-Mamba w/o EncGates'.")
    elif ablation == "no_env":
        print(f"  NOTE: NoEnv — gate_env forced to ones (env_module output bypassed in encoder).")
        print(f"        EnvModule weights remain in graph but receive zero gradient downstream.")

    # Lazy import + monkey-patch symbol in m1_7_train module
    import scripts.m1_7_train as m1_7
    m1_7.CGForecaster = subclass
    t0 = time.time()
    stage2_final = m1_7.train(cfg, stage2_args)
    stage2_elapsed = time.time() - t0
    print(f"  Stage 2 done: best_val_total={stage2_final.get('best_val_total', float('nan')):.4f}  "
          f"elapsed={stage2_elapsed:.1f}s")

    # ── Stage 3 ──
    if not stage2_best.exists():
        raise FileNotFoundError(f"Stage 2 best.pt missing for {ablation}/s{seed} at {stage2_dir}")

    stage3_args = make_stage3_args(stage3_run_name, stage2_dir, hmm_dir, env_ckpt,
                                    epochs=stage3_epochs, patience=0, batch_size=batch_size,
                                    smoke=smoke)
    print(f"[ablation_retrain] === {ablation} / seed={seed} === Stage 3 ===")
    print(f"  Stage 2 dir: {stage2_dir.relative_to(_ROOT)}")

    import scripts.m1_8_stage3_train as m1_8
    m1_8.CGForecaster = subclass
    t1 = time.time()
    stage3_final = m1_8.stage3_train(cfg, stage3_args)
    stage3_elapsed = time.time() - t1
    print(f"  Stage 3 done: best_val_total={stage3_final.get('best_val_total', float('nan')):.4f}  "
          f"elapsed={stage3_elapsed:.1f}s")

    return {
        "ablation": ablation, "seed": seed, "device": device,
        "stage2_run_name": stage2_run_name, "stage3_run_name": stage3_run_name,
        "stage2_best_val_total": stage2_final.get("best_val_total"),
        "stage3_best_val_total": stage3_final.get("best_val_total"),
        "stage2_elapsed_sec": stage2_elapsed,
        "stage3_elapsed_sec": stage3_elapsed,
        "stage2_best_path": str(stage2_best),
        "stage3_best_path": str(stage3_best),
        "resumed": False,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation retraining (utility, from-scratch).")
    parser.add_argument("--ablation", choices=list(ABLATIONS),
                        help="Single ablation to run. If omitted with --launch-all, all 3 run.")
    parser.add_argument("--seed", type=int, help="Single seed (must be in SEEDS).")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: all 5).")
    parser.add_argument("--launch-all", action="store_true",
                        help="Run all 3 ablations × all 5 seeds sequentially (15 runs).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device string, e.g. 'cuda:0' (default).")
    parser.add_argument("--stage2-epochs", type=int, default=200,
                        help="Stage 2 epochs (default 200, matches CGMambaConfig default).")
    parser.add_argument("--stage3-epochs", type=int, default=10,
                        help="Stage 3 epochs (default 10).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true",
                        help="Forward smoke=True to m1_7/m1_8 (forces 5 ep + bs=8 + warmup=P1=TR=1).")
    parser.add_argument("--manifest", type=str,
                        default=str(OUT_ROOT / "manifest.json"),
                        help="Append run summaries to this JSON manifest.")
    args = parser.parse_args()
    # Resolve manifest path to absolute to avoid relative_to errors
    args.manifest = str(Path(args.manifest).resolve())

    # Resolve work-list
    seeds = (
        tuple(int(s) for s in args.seeds.split(","))
        if args.seeds else
        (args.seed,) if args.seed is not None else SEEDS
    )
    ablations = (args.ablation,) if args.ablation else ABLATIONS
    if not args.launch_all and (args.ablation is None or args.seed is None) and args.seeds is None:
        parser.error("Specify --ablation+--seed, --ablation+--seeds, or --launch-all.")

    print(f"[ablation_retrain] Launching {len(ablations)} ablation(s) × {len(seeds)} seed(s) = "
          f"{len(ablations) * len(seeds)} runs on {args.device}")
    print(f"  ablations: {ablations}")
    print(f"  seeds:     {seeds}")
    print(f"  HPO frozen to CG_TOP1_HP (lookback=104, gate_lr=1e-3, backbone_lr=1e-4)")

    # Load existing manifest (append-only)
    manifest_path = Path(args.manifest)
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            print(f"  WARN: existing manifest unreadable, will overwrite")

    summaries = []
    n_done = 0
    n_fail = 0
    t_start = time.time()
    for ablation in ablations:
        for seed in seeds:
            print(f"\n{'=' * 70}\nRUN {n_done+n_fail+1}/{len(ablations)*len(seeds)}: "
                  f"ablation={ablation}  seed={seed}\n{'=' * 70}")
            try:
                summary = patch_and_train_one(
                    ablation=ablation, seed=seed, device=args.device,
                    stage2_epochs=args.stage2_epochs,
                    stage3_epochs=args.stage3_epochs,
                    batch_size=args.batch_size,
                    smoke=args.smoke,
                )
                summaries.append(summary)
                # PATCH B — Manifest dedupe: replace existing (ablation, seed) entry
                manifest = [
                    m for m in manifest
                    if (m.get("ablation"), m.get("seed")) != (ablation, seed)
                ]
                manifest.append(summary)
                manifest_path.write_text(json.dumps(manifest, indent=2))
                n_done += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                summaries.append({"ablation": ablation, "seed": seed, "error": str(e)})
                n_fail += 1
                # Continue to next run (don't abort entire batch)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"[ablation_retrain] BATCH COMPLETE")
    print(f"  done={n_done}  failed={n_fail}  total={n_done + n_fail}")
    print(f"  elapsed: {elapsed/60:.1f} min ({elapsed:.0f} sec)")
    try:
        print(f"  manifest: {manifest_path.relative_to(_ROOT)}")
    except ValueError:
        print(f"  manifest: {manifest_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
