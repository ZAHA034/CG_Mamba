"""ROLLING-ORIGIN STAGE 1 -- native-APMD regional eval SANITY GATE (GPU-light, no training).

Purpose (pre-registered unconditional rule #4):
  Before spending ~210 training runs, PROVE that our per-horizon native-APMD regional
  coverage code reproduces the paper headline. We do this by re-scoring the EXISTING
  headline n3_d64 checkpoints (no retraining) with our own native eval and checking it
  matches the stored canonical numbers.

Method (reuses the EXACT headline pipeline unchanged -- this is the point):
  - e1_final_eval.collect_regional_predictions("n3_d64", 3, 64, seed, region) forwards the
    headline Stage-3 checkpoint on regional test_strict and returns native APMD columns
    (mu, s2_total, y_true) with NO s_per_h scaling (the IV-F Scaled trap is absent here).
  - We add PER-HORIZON Cov95/WIS via the headline's own eval_cov95_wis (native quantiles).

Two gates (must BOTH pass before trusting any rolling number):
  GATE-A (aggregate): per-horizon Cov95 (mean over 10 regions x 5 seeds) must reproduce
                      0.998 / 0.970 / 0.939 / 0.910 (avg 0.954) within +/- 0.01.
  GATE-B (row-level): our recomputed tS_cov95_h* must match the stored canonical file
                      runs/e1_final/n3_d64_regional_perhorizon_raw.csv row-by-row
                      (max abs diff < 0.01) -- proving our eval IS the generator's logic.

Config note (canonical, locked): headline regional config = n3_d64 (3 layers, D=64) --
  matches method.tex (D=64, 3 ContextGatedMambaBlock layers) and the stored CSV name.
  e1_final_eval.py's HEADLINE_CONFIG_ID="n2_d128" is an earlier pre-registration framing;
  the PAPER's regional 0.954 claim rests on n3_d64, so rolling uses n3_d64.

GPU: 5 seeds x 10 regions = 50 forward passes, NO training. Writes only to runs/rolling_origin/.

USAGE:
    python scripts/rolling_origin_stage1_sanity.py            # all seeds x regions
    python scripts/rolling_origin_stage1_sanity.py --quick    # seed 42 only (10 fwd, smoke)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import e1_final_eval as E   # reuse headline pipeline verbatim

CONFIG_ID, N_LAYERS, D_MODEL = "n3_d64", 3, 64      # paper headline regional config
STORED_CANON = _ROOT / "runs/e1_final/n3_d64_regional_perhorizon_raw.csv"
OUT_DIR = _ROOT / "runs/rolling_origin"

# pre-registered sanity target (computed from canonical CSV, this session)
TARGET_PERH = {1: 0.998, 2: 0.970, 3: 0.939, 4: 0.910}
TARGET_AVG = 0.954
BAND = 0.01


def compute_perhorizon(seeds, regions, device):
    rows = []
    for seed in seeds:
        for region in regions:
            df = E.collect_regional_predictions(CONFIG_ID, N_LAYERS, D_MODEL, seed, region, device)
            rec = {"baseline": "cg_mamba", "seed": int(seed), "region": region}
            for h in E.HORIZONS:
                d = df[df.horizon == h]
                # NATIVE coverage: reuse the headline's own eval (no s_per_h)
                cov_h, wis_h = E.eval_cov95_wis(d.mu.to_numpy(), d.s2_total.to_numpy(),
                                                d.y_true.to_numpy())
                rec[f"tS_cov95_h{h}"] = float(cov_h)
                rec[f"tS_wis_h{h}"] = float(wis_h)
            rows.append(rec)
            print(f"  [{region:>6} s{seed:<4}] "
                  + " ".join(f"h{h}={rec[f'tS_cov95_h{h}']:.3f}" for h in E.HORIZONS))
    return pd.DataFrame(rows)


def gate_a(df) -> tuple[bool, dict]:
    """Aggregate per-horizon Cov95 (mean over regions, then seeds -- headline aggregation)."""
    perh = {}
    for h in E.HORIZONS:
        # groupby region -> mean over seeds, then mean over regions (balanced == flat mean)
        perh[h] = float(df.groupby("region")[f"tS_cov95_h{h}"].mean().mean())
    avg = float(np.mean(list(perh.values())))
    ok = all(abs(perh[h] - TARGET_PERH[h]) <= BAND for h in E.HORIZONS) and abs(avg - TARGET_AVG) <= BAND
    return ok, {"per_horizon": perh, "avg": avg,
                "target_per_horizon": TARGET_PERH, "target_avg": TARGET_AVG,
                "max_abs_dev": max(abs(perh[h] - TARGET_PERH[h]) for h in E.HORIZONS)}


def gate_b(df) -> tuple[bool, dict]:
    """Row-level match vs stored canonical CSV (strongest: proves eval == generator)."""
    if not STORED_CANON.exists():
        return False, {"error": f"stored canonical not found: {STORED_CANON}"}
    stored = pd.read_csv(STORED_CANON)
    cols = [f"tS_cov95_h{h}" for h in E.HORIZONS]
    key = ["seed", "region"]
    m = df[key + cols].merge(stored[key + cols], on=key, suffixes=("_new", "_old"))
    if len(m) != len(stored):
        return False, {"error": f"row mismatch: recomputed {len(m)} vs stored {len(stored)}"}
    diffs = {c: float((m[f"{c}_new"] - m[f"{c}_old"]).abs().max()) for c in cols}
    max_diff = max(diffs.values())
    return (max_diff < 0.01), {"max_abs_diff": max_diff, "per_col_max_diff": diffs, "n_rows": len(m)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="seed 42 only (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seeds = [42] if args.quick else E.SEEDS
    regions = E.REGIONS_TRANSFER
    device = args.device
    print(f"[stage1] config={CONFIG_ID} seeds={seeds} regions={len(regions)} device={device}")
    print(f"[stage1] NO training -- forwarding existing headline checkpoints only.\n")

    df = compute_perhorizon(seeds, regions, device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / ("stage1_sanity_quick.csv" if args.quick else "stage1_sanity_recomputed.csv")
    df.to_csv(out_csv, index=False)

    a_ok, a = gate_a(df)
    print(f"\n=== GATE-A (aggregate per-horizon Cov95 vs headline) ===")
    for h in E.HORIZONS:
        print(f"  h{h}: recomputed {a['per_horizon'][h]:.3f}  target {TARGET_PERH[h]:.3f}  "
              f"dev {abs(a['per_horizon'][h]-TARGET_PERH[h]):.3f}")
    print(f"  avg {a['avg']:.4f} (target {TARGET_AVG})  max_dev {a['max_abs_dev']:.3f}  "
          f"-> {'PASS' if a_ok else 'FAIL'}")

    if args.quick:
        print("\n[stage1] --quick: GATE-B (row-level, needs all seeds) skipped.")
        print(f"[stage1] wrote {out_csv.relative_to(_ROOT)}")
        return 0 if a_ok else 2

    b_ok, b = gate_b(df)
    print(f"\n=== GATE-B (row-level vs stored canonical CSV) ===")
    if "error" in b:
        print(f"  ERROR: {b['error']}")
    else:
        print(f"  n_rows {b['n_rows']}  max_abs_diff {b['max_abs_diff']:.4f}  "
              f"-> {'PASS' if b_ok else 'FAIL'}")
        for c, d in b["per_col_max_diff"].items():
            print(f"    {c}: max|Δ|={d:.4f}")

    verdict = a_ok and b_ok
    (OUT_DIR / "stage1_sanity_verdict.json").write_text(json.dumps({
        "config": CONFIG_ID, "gate_a": a, "gate_a_pass": a_ok,
        "gate_b": b, "gate_b_pass": b_ok, "overall_pass": verdict,
        "meaning": "if PASS, native-APMD regional eval reproduces the headline -> "
                   "safe to extend to rolling cutoffs; if FAIL, driver bug -> HALT & investigate.",
    }, indent=2))
    print(f"\n{'='*60}\n[stage1] OVERALL: {'PASS -- native eval proven, proceed to rolling' if verdict else 'FAIL -- HALT, investigate driver (NOT a result)'}")
    print(f"[stage1] wrote {out_csv.relative_to(_ROOT)} + verdict json")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
