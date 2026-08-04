"""scripts/p3_integration_test.py — Track B per-baseline NATIVE integration test.

PURPOSE (user condition #1, 2026-06-22)
---------------------------------------
For each of the 6 Track B baselines, reproduce a SEED=42 NATIVE base-UQ cell
over the 10 HHS regions. This script is a FORWARD-CORRECTNESS GATE: it verifies
that the (region_test base-quantile builder + per-cell scoring) pipeline at
seed=42 reproduces a known disk-evidence cell within a tolerance set per the
baseline's stochastic noise floor.

SINGLE-SEED vs 5-SEED TABLE IV RECONCILIATION
---------------------------------------------
- This script: seed=42 only. Targets are the seed=42 disk artifacts
  (phase_3_region_wis*.csv) re-aggregated to cross-region per-h-mean. This is
  a forward-correctness gate, NOT the Table IV (5-seed mean) reproduction.
- Table IV (5-seed mean) reproduction lives in ``scripts/p3_full_track_b_run.py``
  (the full run): 5 seeds × 10 regions × native UQ, mean-aggregated to compare
  against Table IV. That full run carries the AUTHORITATIVE LOCK for EpiDeep
  ckpt path designation per user condition #1 (2026-06-22) — not this script.
- The MC-Dropout baselines (LSTM/VM/PatchTST/EpiDeep, n_mc=100) have an
  irreducible per-cell MC variance even at fixed seed because the MC RNG
  differs between our re-eval library and the disk run. Per-baseline tolerance
  therefore tracks the theoretical MC SE envelope (~2σ; see TOLERANCES).
- DLinear and CGM are deterministic given seed/calibration (DLinear: 5-seed
  Gaussian ensemble, seed arg ignored internally; CGM: deterministic + fixed
  RAW APMD smoke calibration), so they keep the tight |Δ|<0.005 envelope.

Pipeline (per baseline, seed=42)
--------------------------------
1. For each of 10 HHS regions hhs1..hhs10:
     - Build region_test base quantiles via
       ``track_b_lib.build_<baseline>_region_test_quantiles(42, device, norm, region)``.
     - NATIVE only (no CQR): for each h in {1,2,3,4}, slice base quantile dict
       to 1-D per-h then ``score_per_cell``.
2. Aggregate: per-region per-h -> mean over 10 regions per h -> mean over 4 horizons
   -> single (WIS, Cov95) per baseline. (LOCK §5 aggregation order.)
3. Compare aggregated (WIS, Cov95) to per-baseline seed=42 disk-artifact target
   (or RAW APMD smoke target for CG-Mamba). |delta| < per-baseline tolerance
   -> PASS, else FAIL.

HARD-STOP CAVEAT for CG-Mamba
-----------------------------
Track B uses RAW APMD (LOCK §4 raw, no s_per_h calibration). Therefore for
CG-Mamba the target is the 2026-06-21 RAW APMD smoke reproduction
(WIS=0.4045, Cov95=0.952 h-mean), NOT the Method-F Table IV cell (0.368/0.930).
Hardcoded RAW target for CGM, not Method-F. Note: 0.998 is h=1 only (smoke
parquet); h-mean across h=1..4 is (0.998+0.970+0.936+0.905)/4 = 0.9520.

Tolerance basis (corrected 2026-06-23 — tS_* strict targets)
------------------------------------------------------------
Diagnosed via mc_variance_probe.py (N=5 reruns × 4 NN baselines): probe σ ≈
0.0005 (WIS) and ≈ 0.0015-0.0020 (Cov95). 100-MC-sample mean has variance ≈
per-sample-σ / sqrt(100) ≈ tiny. Earlier "systematic Δ" vs disk was NOT MC
noise; it was a target-column mismatch:
  - phase_3_region_wis*.csv has both `tF_*` (full test, n_full=254/region)
    and `tS_*` (strict, eps_h1 >= 202240, n_strict=149/region) columns.
  - The previous TARGETS were extracted from `tF_*` (full, n=254).
  - The builder `build_<x>_region_test_quantiles` returns the strict subset
    (n=149) — its returned `n_test_strict` matches `tS_*` exactly.
  - Cell mismatch → consistent 0.03-0.06 Δ across all 4 NN baselines, NOT a
    forward bug. Re-pointing targets at `tS_*` reproduces builder output
    within probe σ (typically |Δ| < 0.005 WIS, < 0.010 Cov95).
Tight tolerance is therefore correct; the previous loose 2σ envelope was
needed only because of the wrong-column mismatch.

Targets (seed=42 disk-artifact cells, tS_* STRICT, n=149/region)
----------------------------------------------------------------
- cg_mamba:      wis=0.4045, cov95=0.9520  tol=0.005/0.005
                 (2026-06-21 RAW APMD smoke parquet, n_test_strict=149.
                  Deterministic; bit-identical reproduction expected.)
- lstm:          wis=0.3854, cov95=0.5592  tol=0.005/0.010
                 (phase_3_region_wis.csv tS_* seed=42 cross-region h-mean.)
- vanilla_mamba: wis=0.4568, cov95=0.6012  tol=0.005/0.010
                 (phase_3_region_wis.csv tS_* seed=42 cross-region h-mean.)
- patchtst:      wis=0.4604, cov95=0.6025  tol=0.005/0.010
                 (phase_3_region_wis.csv tS_* seed=42 cross-region h-mean.)
- dlinear:       wis=0.509,  cov95=0.286   tol=0.005/0.015
                 (Table IV row = 5-seed Gaussian ensemble. Deterministic;
                  Cov95 tol widened to 0.015 for cuda nondeterminism allowance.)
- epideep:       wis=0.5430, cov95=0.3883  tol=0.005/0.010
                 (phase_3_region_wis_extras.csv tS_* seed=42 cross-region h-mean.)

NOTE: Table IV (5-seed mean) reproduction is the responsibility of
``scripts/p3_full_track_b_run.py`` summary.json -> table_iv_reproduction_check.
The EpiDeep AUTHORITATIVE LOCK per user condition #1 (2026-06-22) is on the
full run, not this seed=42 forward-correctness gate.

Output
------
``runs/track_b_integration/results.json`` with structure:
    {
      "baselines": {
        "<baseline>": {
          "target": {"wis": float, "cov95": float},
          "actual": {"wis": float, "cov95": float},
          "delta": {"wis": float, "cov95": float},
          "per_horizon": {"h1": {...}, ..., "h4": {...}},
          "per_region": [{"region": "hhs1", "h": 1, "wis": float, "cov95": float, ...}, ...],
          "verdict": "PASS" | "FAIL",
          "notes": "..."   # CGM: RAW APMD note
        }, ...
      },
      "overall_verdict": "PASS" | "FAIL",
      "fail_list": ["<baseline>", ...]
    }

Exit code: 0 iff overall_verdict == "PASS" (i.e. fail_list empty), else 1.

CLI
---
    python3 scripts/p3_integration_test.py --device cuda:0
    python3 scripts/p3_integration_test.py --baseline lstm --device cuda:0
    python3 scripts/p3_integration_test.py --baseline epideep --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# Time-Series-Library is auto-added by track_b_lib import
import scripts.track_b_lib as tbl
from scripts.track_b_lib import (
    FLUSIGHT_23,
    HORIZONS,
    score_per_cell,
    load_norm,
)


# ============================================================================
# Constants
# ============================================================================
REGIONS = tuple(f"hhs{i}" for i in range(1, 11))
SEED = 42  # single-seed sanity (user condition #1, 2026-06-22)

# Per-baseline tolerance for |delta| envelope.
# Evidence basis (2026-06-23 MC-instance probe, N=5 reruns × 4 NN baselines):
#   probe σ(WIS) ≈ 0.0005 ; σ(Cov95) ≈ 0.0015-0.0020 across all 4 NN baselines.
# Builder is near-deterministic across reruns at seed=42 (100-MC-sample mean
# variance ≈ per-sample-σ / sqrt(100) ≈ 0.0005). Targets are matched to
# `tS_*` (strict, eps_h1 >= 202240, n=149/region) — the SAME subset the builder
# returns. Earlier `tF_*` (full, n=254) targets were the wrong cell; that mismatch
# (NOT MC noise) drove the previous loose-tol envelope.
# DLinear/CGM deterministic given fixed calibration; tight envelope.
TOLERANCES = {
    "cg_mamba":      {"wis": 0.005, "cov95": 0.005},
    "lstm":          {"wis": 0.005, "cov95": 0.010},
    "vanilla_mamba": {"wis": 0.005, "cov95": 0.010},
    "patchtst":      {"wis": 0.005, "cov95": 0.010},
    "dlinear":       {"wis": 0.005, "cov95": 0.015},
    "epideep":       {"wis": 0.005, "cov95": 0.010},
}

# Per-baseline seed=42 disk-artifact targets (NOT Table IV 5-seed mean; Table IV
# reproduction = scripts/p3_full_track_b_run.py summary.json table_iv_reproduction_check)
TARGETS = {
    "cg_mamba":      {"wis": 0.4045, "cov95": 0.9520,
                       "notes": "RAW APMD (LOCK §4 raw, no s_per_h). 2026-06-21 smoke reproduction h-mean across h=1..4: (0.998+0.970+0.936+0.905)/4 = 0.9520; NOT h=1 only (0.998). NOT Method-F Table IV (0.368/0.930). Deterministic -> tight tol."},
    "lstm":          {"wis": 0.3854, "cov95": 0.5592,
                       "notes": "phase_3_region_wis.csv seed=42 cross-region 10-HHS mean of tS_wis_h{1..4}/tS_cov95_h{1..4} (STRICT, eps_h1>=202240, n=149/region), then h-mean. Builder also returns strict subset (n=149) so columns match. tF_* (full, n=254) gives 0.3779/0.5186 — wrong cell."},
    "vanilla_mamba": {"wis": 0.4568, "cov95": 0.6012,
                       "notes": "phase_3_region_wis.csv seed=42 STRICT cross-region 10-HHS mean (tS_*, n=149)."},
    "patchtst":      {"wis": 0.4604, "cov95": 0.6025,
                       "notes": "phase_3_region_wis.csv seed=42 STRICT cross-region 10-HHS mean (tS_*, n=149)."},
    "dlinear":       {"wis": 0.509,  "cov95": 0.286,
                       "notes": "Table IV row = 5-seed Gaussian ensemble (seed arg ignored internally). Deterministic -> tight tol."},
    "epideep":       {"wis": 0.5430, "cov95": 0.3883,
                       "notes": "phase_3_region_wis_extras.csv seed=42 STRICT cross-region 10-HHS mean (tS_*, n=149). AUTHORITATIVE LOCK for EpiDeep ckpt path = full-run Table IV reproduction, NOT this seed=42 gate."},
}

# Dispatch table for region_test base-quantile builders
BUILDERS = {
    "cg_mamba":      tbl.build_cg_mamba_region_test_quantiles,
    "lstm":          tbl.build_lstm_region_test_quantiles,
    "vanilla_mamba": tbl.build_vanilla_mamba_region_test_quantiles,
    "patchtst":      tbl.build_patchtst_region_test_quantiles,
    "dlinear":       tbl.build_dlinear_region_test_quantiles,
    "epideep":       tbl.build_epideep_region_test_quantiles,
}

ORDER = ("cg_mamba", "lstm", "vanilla_mamba", "patchtst", "dlinear", "epideep")

OUT_DIR = _ROOT / "runs" / "track_b_integration"
OUT_JSON = OUT_DIR / "results.json"


# ============================================================================
# Per-baseline NATIVE evaluation
# ============================================================================
def eval_baseline_native(baseline: str, device: str, norm: dict) -> dict:
    """Run NATIVE-only cross-region per-h evaluation for a single baseline.

    Returns a dict with per_region cells, per_horizon means, aggregated
    (WIS, Cov95), target, delta, verdict, and notes.
    """
    builder = BUILDERS[baseline]
    target = TARGETS[baseline]
    tol = TOLERANCES[baseline]
    print(f"\n=== baseline={baseline} (seed={SEED}, tol WIS={tol['wis']}, Cov95={tol['cov95']}) ===", flush=True)
    t0 = time.time()

    per_region_rows: list[dict] = []
    for region in REGIONS:
        try:
            qf_region, y_region, _eps = builder(SEED, device, norm, region)
        except Exception as e:
            print(f"  [{baseline}/{region}] builder FAIL: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
            raise

        # qf_region: dict tau -> [N_strict, H]  (region_test builder convention)
        # y_region:  [N_strict, H]
        for h_idx, h in enumerate(HORIZONS):
            qf_h = {float(t): np.asarray(qf_region[float(t)][:, h_idx]) for t in FLUSIGHT_23}
            cell = score_per_cell(qf_h, y_region, h_idx, f"{baseline}/{region}/h={h}")
            per_region_rows.append({
                "region": region,
                "h": h,
                "n_test_strict": int(y_region.shape[0]),
                "wis": cell["wis"],
                "cov95": cell["cov95"],
                "mae": cell["mae"],
            })
            print(
                f"  {region} h={h}  n={y_region.shape[0]:3d}  "
                f"WIS={cell['wis']:.4f}  Cov95={cell['cov95']:.3f}  MAE={cell['mae']:.4f}",
                flush=True,
            )

    # Aggregation: per-region per-h -> mean over regions per h -> mean over horizons
    per_horizon = {}
    for h in HORIZONS:
        sub = [r for r in per_region_rows if r["h"] == h]
        wis_h = float(np.mean([r["wis"] for r in sub]))
        cov_h = float(np.mean([r["cov95"] for r in sub]))
        mae_h = float(np.mean([r["mae"] for r in sub]))
        per_horizon[f"h{h}"] = {"wis": wis_h, "cov95": cov_h, "mae": mae_h,
                                  "n_regions": len(sub)}

    actual_wis = float(np.mean([per_horizon[f"h{h}"]["wis"] for h in HORIZONS]))
    actual_cov = float(np.mean([per_horizon[f"h{h}"]["cov95"] for h in HORIZONS]))
    actual_mae = float(np.mean([per_horizon[f"h{h}"]["mae"] for h in HORIZONS]))

    delta_wis = actual_wis - target["wis"]
    delta_cov = actual_cov - target["cov95"]

    pass_wis = abs(delta_wis) < tol["wis"]
    pass_cov = abs(delta_cov) < tol["cov95"]
    verdict = "PASS" if (pass_wis and pass_cov) else "FAIL"

    elapsed = time.time() - t0
    print(f"  -> aggregated: WIS={actual_wis:.4f} (target {target['wis']:.4f}, Delta={delta_wis:+.4f}, tol={tol['wis']})  "
          f"Cov95={actual_cov:.3f} (target {target['cov95']:.3f}, Delta={delta_cov:+.4f}, tol={tol['cov95']})  "
          f"=> {verdict}  [elapsed {elapsed:.1f}s]",
          flush=True)

    return {
        "target": {"wis": target["wis"], "cov95": target["cov95"]},
        "actual": {"wis": actual_wis, "cov95": actual_cov, "mae": actual_mae},
        "delta": {"wis": delta_wis, "cov95": delta_cov},
        "delta_tol": {"wis": tol["wis"], "cov95": tol["cov95"]},
        "pass_wis": bool(pass_wis),
        "pass_cov95": bool(pass_cov),
        "per_horizon": per_horizon,
        "per_region": per_region_rows,
        "verdict": verdict,
        "notes": target["notes"],
        "elapsed_sec": elapsed,
        "seed": SEED,
        "n_regions": len(REGIONS),
    }


# ============================================================================
# Main
# ============================================================================
def main(device: str, baseline_filter: str | None) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print(f"[warn] CUDA not available, falling back to cpu", flush=True)
        device = "cpu"

    print(f"[integration] device={device}  seed={SEED}  regions={len(REGIONS)}  "
          f"per-baseline tolerances (WIS/Cov95):", flush=True)
    for b in ORDER:
        t = TOLERANCES[b]
        print(f"    {b:14s}: WIS<{t['wis']}  Cov95<{t['cov95']}", flush=True)
    print("[integration] Forward-correctness gate at seed=42. "
          "EpiDeep AUTHORITATIVE LOCK = full-run Table IV reproduction "
          "(p3_full_track_b_run.py), NOT this gate.", flush=True)

    norm = load_norm()

    if baseline_filter is not None:
        if baseline_filter not in BUILDERS:
            print(f"[error] unknown --baseline {baseline_filter!r}; "
                  f"valid: {list(BUILDERS.keys())}", flush=True)
            return 2
        baselines = [baseline_filter]
    else:
        baselines = list(ORDER)

    results: dict[str, dict] = {}
    fail_list: list[str] = []
    t_global = time.time()

    for baseline in baselines:
        try:
            res = eval_baseline_native(baseline, device, norm)
        except Exception as e:
            print(f"[integration] {baseline} CRASH: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
            res = {
                "target": {"wis": TARGETS[baseline]["wis"],
                            "cov95": TARGETS[baseline]["cov95"]},
                "actual": None,
                "delta": None,
                "verdict": "FAIL",
                "error": f"{type(e).__name__}: {e}",
                "notes": TARGETS[baseline]["notes"],
                "seed": SEED,
            }
        results[baseline] = res
        if res.get("verdict") != "PASS":
            fail_list.append(baseline)

    overall = "PASS" if not fail_list else "FAIL"
    payload = {
        "purpose": "Per-baseline NATIVE UQ seed=42 forward-correctness gate vs disk-artifact cells. Table IV (5-seed) reproduction = p3_full_track_b_run.py summary.json table_iv_reproduction_check.",
        "seed": SEED,
        "regions": list(REGIONS),
        "horizons": list(HORIZONS),
        "delta_tol_per_baseline": {b: TOLERANCES[b] for b in ORDER},
        "device": device,
        "lock_reference": "paper/track_b_sub_pre_registration.md (LOCKED 2026-06-21)",
        "parent_lock": "project_cgmamba_pc012_locked (2026-06-12 v2)",
        "cg_mamba_caveat": "Track B uses RAW APMD (LOCK §4 raw, no s_per_h). CGM target = 2026-06-21 RAW smoke reproduction (0.4045/0.998), NOT Method-F Table IV (0.368/0.930).",
        "epideep_authoritative_note": "EpiDeep AUTHORITATIVE LOCK lives on the full-run Table IV (5-seed) reproduction in scripts/p3_full_track_b_run.py, NOT this seed=42 forward-correctness gate. EpiDeep PASS here only certifies the (region_test builder + scoring) pipeline at seed=42.",
        "tolerance_rationale": "MC-Dropout baselines (LSTM/VM/PatchTST/EpiDeep, n_mc=100, RNG mismatch between re-eval lib and disk): WIS<0.015 Cov95<0.020 (~2sigma MC SE envelope). DLinear (5-seed Gaussian ensemble) + CGM (deterministic + fixed RAW APMD smoke calibration): WIS<0.005 Cov95<0.005.",
        "baselines": results,
        "overall_verdict": overall,
        "fail_list": fail_list,
        "elapsed_sec_total": time.time() - t_global,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[save] {OUT_JSON}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("INTEGRATION TEST SUMMARY", flush=True)
    print("=" * 80, flush=True)
    for b in baselines:
        r = results[b]
        if r.get("actual") is not None:
            tol_b = TOLERANCES[b]
            print(f"  {b:14s}: WIS {r['actual']['wis']:.4f} (tgt {r['target']['wis']:.4f}, Δ{r['delta']['wis']:+.4f}, tol<{tol_b['wis']})  "
                  f"Cov95 {r['actual']['cov95']:.3f} (tgt {r['target']['cov95']:.3f}, Δ{r['delta']['cov95']:+.4f}, tol<{tol_b['cov95']})  "
                  f"=> {r['verdict']}",
                  flush=True)
        else:
            print(f"  {b:14s}: CRASH -> FAIL ({r.get('error','?')})", flush=True)
    print(f"\nOverall verdict: {overall}", flush=True)
    if fail_list:
        print(f"FAIL list: {fail_list}", flush=True)
        if "epideep" in fail_list:
            print("[NOTE] EpiDeep FAILED the seed=42 forward-correctness gate. "
                  "The AUTHORITATIVE LOCK is on the full-run Table IV (5-seed) "
                  "reproduction, not this gate, but a seed=42 failure here is a "
                  "strong signal that the (region_test builder + scoring) pipeline "
                  "diverges from the disk artifact and warrants investigation "
                  "before the full run.",
                  flush=True)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0",
                    help="torch device (e.g. cuda:0, cpu).")
    ap.add_argument("--baseline", default=None,
                    choices=list(BUILDERS.keys()),
                    help="Run a single baseline (default: all 6).")
    args = ap.parse_args()
    sys.exit(main(args.device, args.baseline))
