"""DEPRECATED in v2.0.9 — retained for paper §7.4 ablation B reproduction. Not active in v2.0.9 main path.

HMM Stage 1 grid driver for CG-Mamba M1.4 (v2.0.8c).

Modes:
  --mode grid    : K ∈ {3,4,5} × seeds ∈ {42,123,456} = 9 runs. (default)
  --mode smoke   : K=3, seed=42, 5 epochs (sanity)

Train data (v2.0.8c — was 2-segment in v2.0.8b):
  seg2-only (2002-W40 ~ 2018-W39, 835 rows). seg1 (200140 ~ 200220, 33 rows)
  excluded for system-wide consistency with LSTM/CG-Mamba sliding-window
  models that auto-exclude seg1 because L=104+ > seg1 length. Same effective
  training data across all forecasting models in §7.1 comparison.

Output:
  runs/hmm_stage1/K{k}_V{v}_seed{s}/
    ├── hmm_stage1.pt
    ├── diagnostics.json (per-epoch κ, dead state, σ check)
    └── viterbi_path.npy
  runs/hmm_stage1/hmm_summary.csv (9 rows × [K, V, seed, final_kappa, dead, ...])

V=3 fallback policy (PLAN §3.4/§5.1 EB-2/EB-3):
  V=4 default 학습 → 모든 K에서 fallback trigger 발동 시 V=3 재학습.
  본 driver는 V=4 1차만 실행하고, fallback 판정은 --check-fallback 옵션으로 별도 수행.

Run:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_hmm_stage1.py --mode smoke --device cuda:0
  CUDA_VISIBLE_DEVICES=1 python scripts/run_hmm_stage1.py --mode grid --device cuda:0
  CUDA_VISIBLE_DEVICES=1 python scripts/run_hmm_stage1.py --mode grid --V 3 --device cuda:0  # fallback
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch

# Project root → sys.path (CG_Mamba/ root for `src.x.y` absolute imports)
_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[2]  # legacy/ moved: scripts/legacy/run_hmm_stage1.py → CG_Mamba/
if str(_CG_MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CG_MAMBA_ROOT))

from src.models.legacy.hmm_stage1 import (  # noqa: E402  (v2.0.9: moved to legacy)
    NeuralSwitchingVARHMM,
    HMM_V4_COLS,
    HMM_V3_FALLBACK_COLS,
    prepare_hmm_train,           # v2.0.8c (was prepare_hmm_train_segments)
    forward_train,                # v2.0.8c (was forward_two_segments)
    init_hmm,
    get_param_groups,
)
from src.utils.metrics import (  # noqa: E402
    cohens_kappa_binary,
    cohens_kappa_aligned,            # v2.0.8c F: cross-seed reproducibility κ
    state_occupancy,
    is_dead_state,
    is_sigma_collapse,
    fallback_trigger,
    DEAD_STATE_THRESHOLD,
    KAPPA_FAIL_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Spec (PLAN v2.0.8c §3.4 EB-2, §5.1 EB-3)
# ---------------------------------------------------------------------------
K_GRID = [3, 4, 5]
SEED_GRID = [42, 123, 456]
DEFAULT_EPOCHS = 100
EARLY_PASS_KAPPA = 0.60  # κ ≥ 0.60 도달 시 early-pass


def train_one_hmm(
    K: int,
    seed: int,
    feature_cols: list[str],
    csv_path: Path,
    norm_path: Path,
    device: str,
    out_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = 1e-3,
    log_every: int = 5,
) -> dict:
    """단일 HMM 학습 (gap-aware 2-segment + κ monitoring).

    Returns:
        dict with final κ, occupancy, sigma_collapse, history
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    seg, ili_raw = prepare_hmm_train(
        csv_path=csv_path,
        norm_path=norm_path,
        feature_cols=feature_cols,
    )
    seg = seg.to(device)
    V = seg.shape[-1]

    # Init (KMeans warm-start + μ0 freeze) — seg2-only (v2.0.8c)
    train_combined = seg.squeeze(0).cpu().numpy()  # [835, V]
    model = init_hmm(V=V, K=K, seed=seed, train_combined=train_combined)
    model = model.to(device)

    param_groups = get_param_groups(model, lr=lr)
    optimizer = torch.optim.AdamW(param_groups)

    history = []
    final_kappa = None
    final_occupancy = None
    final_sigma_collapse = None
    early_passed = False
    t0 = time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        phase_post, nll, _ = forward_train(model, seg)   # v2.0.8c: single-segment
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Diagnostics every `log_every` epochs (and final)
        is_log_epoch = (epoch % log_every == 0) or (epoch == 1) or (epoch == epochs)
        if is_log_epoch:
            with torch.no_grad():
                viterbi_path = model.viterbi(seg).squeeze(0).cpu().numpy()  # [L=835]
                occ = state_occupancy(viterbi_path, K)
                kappa = cohens_kappa_binary(viterbi_path, ili_raw)
                sigma_collapse = is_sigma_collapse(model.log_sigma)
                dead = is_dead_state(occ)

                history.append({
                    "epoch": epoch,
                    "nll": float(nll.item()),
                    "kappa": float(kappa),
                    "occupancy": occ.tolist(),
                    "dead_state": dead,
                    "sigma_collapse": sigma_collapse,
                    "log_sigma": model.log_sigma.detach().cpu().numpy().tolist(),
                })

                if kappa >= EARLY_PASS_KAPPA and not early_passed:
                    early_passed = True

                final_kappa = float(kappa)
                final_occupancy = occ.tolist()
                final_sigma_collapse = bool(sigma_collapse)
                final_viterbi = viterbi_path

    elapsed = time() - t0

    # Save artifacts
    torch.save(model.state_dict(), out_dir / "hmm_stage1.pt")
    np.save(out_dir / "viterbi_path.npy", final_viterbi)

    diagnostics = {
        "K": K,
        "V": V,
        "seed": seed,
        "feature_cols": feature_cols,
        "epochs_trained": epoch,
        "final_kappa": final_kappa,
        "final_occupancy": final_occupancy,
        "final_dead_state": bool(is_dead_state(np.asarray(final_occupancy))),
        "final_sigma_collapse": final_sigma_collapse,
        "early_pass_kappa_0p60": early_passed,
        "elapsed_sec": elapsed,
        "L_train": int(seg.shape[1]),   # v2.0.8c: single-segment (was L1+L2)
        "train_start_epiweek": 200240,   # post-CDC-gap start
        "history": history,
    }
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    return diagnostics


def run_smoke(csv_path: Path, norm_path: Path, device: str, out_root: Path) -> None:
    """K=3, seed=42, 5 epochs sanity."""
    smoke_dir = out_root / "hmm_smoke" / "K3_V4_seed42"
    print(f"[SMOKE] K=3 V=4 seed=42 on {device} (5 epochs)")
    r = train_one_hmm(
        K=3, seed=42, feature_cols=HMM_V4_COLS,
        csv_path=csv_path, norm_path=norm_path,
        device=device, out_dir=smoke_dir, epochs=5,
    )
    print(f"[SMOKE] DONE: final κ={r['final_kappa']:.4f}, "
          f"occ={[f'{o:.2f}' for o in r['final_occupancy']]}, "
          f"dead={r['final_dead_state']}, σ_collapse={r['final_sigma_collapse']}, "
          f"elapsed={r['elapsed_sec']:.1f}s")


def run_grid(
    csv_path: Path, norm_path: Path, device: str, out_root: Path, V: int = 4
) -> None:
    """K ∈ {3,4,5} × seeds ∈ {42,123,456} grid."""
    if V == 4:
        cols = HMM_V4_COLS
    elif V == 3:
        cols = HMM_V3_FALLBACK_COLS
    else:
        raise ValueError(f"V must be 3 or 4, got {V}")

    grid_root = out_root / "hmm_stage1"
    grid_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    n_total = len(K_GRID) * len(SEED_GRID)
    print(f"[GRID] {n_total} runs on {device} (V={V}, cols={cols})")

    # v2.0.8c F: collect per-(K, seed) Viterbi paths for cross-seed κ
    viterbi_paths: dict[int, dict[int, "np.ndarray"]] = {K: {} for K in K_GRID}

    for i, (K, seed) in enumerate(
        [(k, s) for k in K_GRID for s in SEED_GRID], start=1
    ):
        run_name = f"K{K}_V{V}_seed{seed}"
        run_dir = grid_root / run_name
        if (run_dir / "diagnostics.json").exists():
            print(f"[GRID] {i}/{n_total} {run_name} — SKIP")
            with open(run_dir / "diagnostics.json") as f:
                r = json.load(f)
        else:
            print(f"[GRID] {i}/{n_total} {run_name}")
            r = train_one_hmm(
                K=K, seed=seed, feature_cols=cols,
                csv_path=csv_path, norm_path=norm_path,
                device=device, out_dir=run_dir,
            )
            print(f"  → final κ={r['final_kappa']:.4f}, "
                  f"dead={r['final_dead_state']}, σ={r['final_sigma_collapse']}, "
                  f"ep={r['epochs_trained']}")

        # Load viterbi_path for cross-seed κ (saved by train_one_hmm)
        viterbi_paths[K][seed] = np.load(run_dir / "viterbi_path.npy")

        summary_rows.append({
            "K": K,
            "V": V,
            "seed": seed,
            "final_kappa": r["final_kappa"],
            "dead_state": r["final_dead_state"],
            "sigma_collapse": r["final_sigma_collapse"],
            "min_occupancy": min(r["final_occupancy"]),
            "max_occupancy": max(r["final_occupancy"]),
            "early_pass": r["early_pass_kappa_0p60"],
            "elapsed_sec": r["elapsed_sec"],
        })

    # ── v2.0.8c F: Cross-seed reproducibility κ ──
    # Pairwise aligned κ across SEED_GRID (3 seeds → 3 pairs per K).
    # Reported for paper §5.1 reproducibility table; not used in fallback trigger
    # (PLAN EB-3 still uses binary κ vs CDC epi truth).
    print("\n[CROSS-SEED κ] Pairwise aligned κ (paper §5.1 reproducibility)")
    cross_seed_stats: dict[int, dict] = {}
    seeds = list(SEED_GRID)
    for K in K_GRID:
        pairs = []
        for i_s in range(len(seeds)):
            for j_s in range(i_s + 1, len(seeds)):
                s1, s2 = seeds[i_s], seeds[j_s]
                kappa_ij = cohens_kappa_aligned(
                    viterbi_paths[K][s1], viterbi_paths[K][s2], K=K
                )
                pairs.append({"seeds": [s1, s2], "kappa": float(kappa_ij)})
                print(f"  K={K}  κ(seed {s1} vs {s2}) = {kappa_ij:.4f}")
        kappas_only = [p["kappa"] for p in pairs]
        cross_seed_stats[K] = {
            "pairs": pairs,
            "kappa_min": float(min(kappas_only)),
            "kappa_mean": float(np.mean(kappas_only)),
        }
        print(f"  K={K}  cross-seed κ_min={cross_seed_stats[K]['kappa_min']:.4f}, "
              f"κ_mean={cross_seed_stats[K]['kappa_mean']:.4f}")

    # Merge cross-seed κ into summary rows (per K — same value broadcast to 3 seeds)
    for row in summary_rows:
        K = row["K"]
        row["cross_seed_kappa_min"] = cross_seed_stats[K]["kappa_min"]
        row["cross_seed_kappa_mean"] = cross_seed_stats[K]["kappa_mean"]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(grid_root / "hmm_summary.csv", index=False)
    print(f"\n[GRID DONE] Saved: {grid_root / 'hmm_summary.csv'}")

    # Save cross-seed κ to separate JSON for paper §5.1 table
    cross_seed_json = grid_root / "cross_seed_kappa.json"
    with open(cross_seed_json, "w") as f:
        json.dump({"V": V, "K_grid": list(K_GRID), "seeds": seeds,
                   "per_K": cross_seed_stats}, f, indent=2)
    print(f"[CROSS-SEED κ] Saved: {cross_seed_json}")

    # Per-K fallback judgment (EB-3: binary κ — cross-seed κ is supplemental)
    print("\n[FALLBACK CHECK] Per-K aggregate (EB-3: binary κ trigger)")
    for K in K_GRID:
        sub = summary_df[summary_df["K"] == K]
        trig = fallback_trigger(
            final_kappas=sub["final_kappa"].tolist(),
            dead_states=sub["dead_state"].tolist(),
            sigma_collapses=sub["sigma_collapse"].tolist(),
        )
        kappas_str = ", ".join(f"{k:.3f}" for k in trig["metrics"]["final_kappas"])
        cs_min = cross_seed_stats[K]["kappa_min"]
        print(f"  K={K}: binary κ=[{kappas_str}], cross-seed κ_min={cs_min:.3f}, "
              f"fallback={trig['triggered']} ({trig['reason']})")


def main():
    ap = argparse.ArgumentParser(description="HMM Stage 1 grid (v2.0.8c)")
    ap.add_argument("--mode", choices=["smoke", "grid"], default="grid")
    ap.add_argument("--V", type=int, choices=[3, 4], default=4)
    ap.add_argument("--csv", default="data/processed/ili_env_weekly_split.csv")
    ap.add_argument("--norm", default="data/processed/normalization_params.json")
    ap.add_argument("--out", default="runs")
    # CUDA_VISIBLE_DEVICES is the recommended way to pin a specific GPU.
    # Default to cuda:0 (the first VISIBLE device), avoiding the trap where
    # CUDA_VISIBLE_DEVICES=1 + --device cuda:1 fails with invalid ordinal.
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    csv_path = _CG_MAMBA_ROOT / args.csv
    norm_path = _CG_MAMBA_ROOT / args.norm
    out_root = _CG_MAMBA_ROOT / args.out

    print(f"[HMM v2.0.8c] mode={args.mode}, V={args.V}, device={args.device}")
    print(f"  csv={csv_path}\n  norm={norm_path}\n  out={out_root}")

    if args.mode == "smoke":
        run_smoke(csv_path, norm_path, args.device, out_root)
    elif args.mode == "grid":
        run_grid(csv_path, norm_path, args.device, out_root, V=args.V)


if __name__ == "__main__":
    main()
