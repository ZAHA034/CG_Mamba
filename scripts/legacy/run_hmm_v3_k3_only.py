"""DEPRECATED in v2.0.9 — retained for paper §7.4 ablation B reproduction. Not active in v2.0.9 main path.

Targeted run: NSVARHMM K=3, V=3 fallback, 3 seeds (PLAN EB-3 narrow check).

Purpose:
  v2.0.8c Step H에서 K=3 V=4 cross-seed κ=0.974 매우 안정이지만 σ collapse
  100% 발동 → V=3 fallback이 σ collapse를 해결하는지 직접 확인. K=4/K=5는
  GaussianHMM ablation에서 V=3에서도 unstable로 확인되어 제외.

3 runs only:
  K=3, V=3 (drop num_patients per EB-2), seeds=(42, 123, 456)

Output: runs/hmm_stage1_v3_k3_only/K3_V3_seed{s}/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch

_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[2]  # legacy/ moved: scripts/legacy/run_hmm_v3_k3_only.py → CG_Mamba/
if str(_CG_MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_CG_MAMBA_ROOT))

from scripts.legacy.run_hmm_stage1 import train_one_hmm  # v2.0.9: moved to legacy
from src.models.legacy.hmm_stage1 import HMM_V3_FALLBACK_COLS  # v2.0.9: moved to legacy
from src.utils.metrics import cohens_kappa_aligned, fallback_trigger


def main() -> int:
    csv_path = _CG_MAMBA_ROOT / "data" / "processed" / "ili_env_weekly_split.csv"
    norm_path = _CG_MAMBA_ROOT / "data" / "processed" / "normalization_params.json"
    out_root = _CG_MAMBA_ROOT / "runs" / "hmm_stage1_v3_k3_only"
    out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[V=3 K=3 narrow] device={device}")
    print(f"  cols (V=3): {HMM_V3_FALLBACK_COLS}")
    print(f"  output: {out_root.relative_to(_CG_MAMBA_ROOT)}")
    print()

    SEEDS = [42, 123, 456]
    K = 3
    V = 3

    summary_rows = []
    viterbi_paths = {}

    for i, seed in enumerate(SEEDS, start=1):
        run_name = f"K{K}_V{V}_seed{seed}"
        run_dir = out_root / run_name
        print(f"[{i}/3] {run_name}")
        r = train_one_hmm(
            K=K, seed=seed, feature_cols=HMM_V3_FALLBACK_COLS,
            csv_path=csv_path, norm_path=norm_path,
            device=device, out_dir=run_dir,
        )
        print(f"  → final κ={r['final_kappa']:.4f}, dead={r['final_dead_state']}, "
              f"σ_collapse={r['final_sigma_collapse']}, ep={r['epochs_trained']}, "
              f"occ={[f'{o:.3f}' for o in r['final_occupancy']]}")
        viterbi_paths[seed] = np.load(run_dir / "viterbi_path.npy")
        summary_rows.append({
            "K": K, "V": V, "seed": seed,
            "final_kappa": r["final_kappa"],
            "dead_state": r["final_dead_state"],
            "sigma_collapse": r["final_sigma_collapse"],
            "min_occupancy": min(r["final_occupancy"]),
            "max_occupancy": max(r["final_occupancy"]),
            "early_pass": r["early_pass_kappa_0p60"],
            "elapsed_sec": r["elapsed_sec"],
        })

    # Cross-seed pairwise aligned κ
    print()
    print("[CROSS-SEED κ] Pairwise aligned (paper §5.1)")
    pairs = []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            s1, s2 = SEEDS[i], SEEDS[j]
            kappa_ij = cohens_kappa_aligned(viterbi_paths[s1], viterbi_paths[s2], K=K)
            pairs.append({"seeds": [s1, s2], "kappa": float(kappa_ij)})
            print(f"  K=3 V=3  κ(seed {s1} vs {s2}) = {kappa_ij:.4f}")
    kappas = [p["kappa"] for p in pairs]
    cs_min = min(kappas)
    cs_mean = float(np.mean(kappas))
    print(f"  K=3 V=3  cross-seed κ_min={cs_min:.4f}, κ_mean={cs_mean:.4f}")

    # Fallback trigger check
    print()
    print("[FALLBACK CHECK] EB-3")
    trig = fallback_trigger(
        final_kappas=[r["final_kappa"] for r in summary_rows],
        dead_states=[r["dead_state"] for r in summary_rows],
        sigma_collapses=[r["sigma_collapse"] for r in summary_rows],
    )
    binary_kappas_str = ", ".join(f"{r['final_kappa']:.3f}" for r in summary_rows)
    print(f"  K=3 V=3: binary κ=[{binary_kappas_str}]")
    print(f"           cross-seed κ_min={cs_min:.3f}")
    print(f"           fallback={trig['triggered']} ({trig['reason']})")

    # Save
    pd.DataFrame(summary_rows).to_csv(out_root / "summary.csv", index=False)
    with open(out_root / "cross_seed_kappa.json", "w") as f:
        json.dump({"K": K, "V": V, "seeds": SEEDS, "pairs": pairs,
                   "kappa_min": cs_min, "kappa_mean": cs_mean}, f, indent=2)
    print(f"\nSaved: summary.csv + cross_seed_kappa.json under {out_root.relative_to(_CG_MAMBA_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
