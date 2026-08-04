"""scripts/p3_mc_variance_probe.py — measure MC-instance variance directly.

CONTEXT (2026-06-23)
--------------------
Integration test (p3_integration_test.py) at seed=42 found:
  - VM       Δ_wis = +0.058  (largest)
  - PatchTST Δ_cov = +0.059
  - EpiDeep  Δ_cov = +0.053
  - LSTM     Δ_wis = +0.008  (small, safest)
vs targets from phase_3_region_wis.csv seed=42 (single instance on disk).

Tolerance applied (WIS 0.06, Cov95 0.12) was derived from `phase_3_region_wis.csv`
5-seed *training-seed* spread — that captures `model_seed × MC` *combined* variance.
The integration test however compares *same model (seed=42) × different MC RNG only*,
so the relevant noise is **MC-instance variance** — likely smaller than training-seed σ.

This probe measures MC-instance σ directly: each MC NN baseline is forwarded N times
at seed=42 with no RNG seeding (matching phase_3 _mc_samples_nn behavior — different
torch RNG state per Python invocation). The per-baseline (WIS, Cov95) σ_MC across N
reruns is the true noise floor that any single Δ must fit inside.

PROTOCOL
--------
For each of {lstm, vanilla_mamba, patchtst, epideep}:
  - N=3 reruns (sequential, GPU 0-bound)
  - Each rerun:
      for region in 10 HHS:
          build_<baseline>_region_test_quantiles(seed=42, ...)
          per-h score → cross-region per-h-mean → h-mean → (WIS_r, Cov95_r)
  - Compute σ(WIS), σ(Cov95) over N reruns
  - Bound: max |Δ_integration - sample_mean| / σ_MC = z-score
      |z| < 2  → MC noise envelope, forward verified
      |z| > 3  → Δ is real forward difference → bug to investigate

CGM and DLinear excluded (deterministic; integration result is final).

OUTPUT
------
runs/track_b_integration/mc_variance_probe.json:
  {
    "<baseline>": {
      "reruns": [{"wis": ..., "cov95": ...}, ...N],
      "mean": {"wis": ..., "cov95": ...},
      "std":  {"wis": ..., "cov95": ...},
      "spread": {"wis": max-min, "cov95": max-min},
      "integration_target": {"wis": ..., "cov95": ...},   # disk seed=42
      "integration_actual": {"wis": ..., "cov95": ...},   # our seed=42 v2 result
      "delta_vs_target":    {"wis": ..., "cov95": ...},
      "z_score_vs_probe_mean": {"wis": ..., "cov95": ...},  # actual vs probe mean / probe σ
      "verdict": "MC_NOISE" | "MARGINAL" | "BUG_SUSPECT"
    },
    ...
  }

CLI
---
    python3 scripts/p3_mc_variance_probe.py --device cuda:0 --n_reruns 3
    python3 scripts/p3_mc_variance_probe.py --baselines vanilla_mamba --n_reruns 5
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch
import scripts.track_b_lib as tbl
from scripts.track_b_lib import FLUSIGHT_23, HORIZONS, score_per_cell, load_norm


REGIONS = tuple(f"hhs{i}" for i in range(1, 11))
SEED = 42

# Disk seed=42 cross-region per-h-mean (from phase_3_region_wis*.csv)
DISK_TARGETS = {
    "lstm":          {"wis": 0.3779, "cov95": 0.5186},
    "vanilla_mamba": {"wis": 0.4001, "cov95": 0.6365},
    "patchtst":      {"wis": 0.4483, "cov95": 0.5463},
    "epideep":       {"wis": 0.5722, "cov95": 0.3336},
}

# Our integration test v2 (2026-06-23) actuals (same seed=42, different MC RNG instance)
INTEGRATION_V2_ACTUAL = {
    "lstm":          {"wis": 0.3856, "cov95": 0.557},
    "vanilla_mamba": {"wis": 0.4580, "cov95": 0.601},
    "patchtst":      {"wis": 0.4603, "cov95": 0.605},
    "epideep":       {"wis": 0.5427, "cov95": 0.387},
}

BUILDERS = {
    "lstm":          tbl.build_lstm_region_test_quantiles,
    "vanilla_mamba": tbl.build_vanilla_mamba_region_test_quantiles,
    "patchtst":      tbl.build_patchtst_region_test_quantiles,
    "epideep":       tbl.build_epideep_region_test_quantiles,
}

OUT_JSON = _ROOT / "runs" / "track_b_integration" / "mc_variance_probe.json"


def eval_one_rerun(baseline: str, device: str, norm: dict, rerun_idx: int) -> dict:
    builder = BUILDERS[baseline]
    per_region_rows = []
    t0 = time.time()
    for region in REGIONS:
        qf_region, y_region, _eps = builder(SEED, device, norm, region)
        for h_idx, h in enumerate(HORIZONS):
            qf_h = {float(t): np.asarray(qf_region[float(t)][:, h_idx]) for t in FLUSIGHT_23}
            cell = score_per_cell(qf_h, y_region, h_idx, f"{baseline}/{region}/h={h}/r={rerun_idx}")
            per_region_rows.append({"region": region, "h": h,
                                     "wis": cell["wis"], "cov95": cell["cov95"]})
    per_h = {}
    for h in HORIZONS:
        sub = [r for r in per_region_rows if r["h"] == h]
        per_h[f"h{h}"] = {
            "wis": float(np.mean([r["wis"] for r in sub])),
            "cov95": float(np.mean([r["cov95"] for r in sub])),
        }
    wis_agg = float(np.mean([per_h[f"h{h}"]["wis"] for h in HORIZONS]))
    cov_agg = float(np.mean([per_h[f"h{h}"]["cov95"] for h in HORIZONS]))
    elapsed = time.time() - t0
    print(f"    rerun {rerun_idx}: WIS={wis_agg:.4f}  Cov95={cov_agg:.4f}  [{elapsed:.1f}s]", flush=True)
    return {"wis": wis_agg, "cov95": cov_agg, "elapsed_sec": elapsed}


def main(device: str, baselines: list, n_reruns: int) -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print(f"[warn] CUDA not available, falling back to cpu", flush=True)
        device = "cpu"

    print(f"[probe] MC-instance variance probe", flush=True)
    print(f"  device={device}  baselines={baselines}  n_reruns={n_reruns}", flush=True)
    print(f"  seed={SEED} (single weights, MC RNG unseeded → varies per rerun)", flush=True)
    norm = load_norm()

    results = {}
    t_global = time.time()
    for baseline in baselines:
        print(f"\n=== {baseline} ===", flush=True)
        reruns = []
        for r in range(n_reruns):
            reruns.append(eval_one_rerun(baseline, device, norm, r))

        wis_vals = [r["wis"] for r in reruns]
        cov_vals = [r["cov95"] for r in reruns]
        mean = {"wis": float(np.mean(wis_vals)), "cov95": float(np.mean(cov_vals))}
        std = {"wis": float(np.std(wis_vals, ddof=1)) if n_reruns > 1 else 0.0,
               "cov95": float(np.std(cov_vals, ddof=1)) if n_reruns > 1 else 0.0}
        spread = {"wis": float(max(wis_vals) - min(wis_vals)),
                   "cov95": float(max(cov_vals) - min(cov_vals))}

        target = DISK_TARGETS[baseline]
        actual = INTEGRATION_V2_ACTUAL[baseline]
        delta_vs_target = {"wis": actual["wis"] - target["wis"],
                            "cov95": actual["cov95"] - target["cov95"]}
        # z-score of integration_actual vs probe_mean, using probe σ as noise scale
        # (small σ → integration result is itself an extreme draw → suspect)
        def _z(actual_v, mean_v, sd):
            if sd == 0: return None
            return float((actual_v - mean_v) / sd)
        z = {"wis": _z(actual["wis"], mean["wis"], std["wis"]),
             "cov95": _z(actual["cov95"], mean["cov95"], std["cov95"])}

        # Verdict:
        # - MC_NOISE  : both |z| < 2 AND probe spread covers |delta_vs_target|
        # - MARGINAL  : 2 ≤ |z| < 3 OR spread < |delta_vs_target| but within 2× of it
        # - BUG_SUSPECT: |z| ≥ 3 OR spread < |delta_vs_target| / 2
        def _classify(d, sp, zv):
            # If only 1 rerun, fall back to: |delta| < disk-sigma envelope (cant compute z)
            if zv is None:
                return "INSUFFICIENT_N"
            if abs(zv) < 2 and sp >= abs(d):
                return "MC_NOISE"
            if abs(zv) < 3 and sp >= abs(d) / 2:
                return "MARGINAL"
            return "BUG_SUSPECT"

        verdict_wis = _classify(delta_vs_target["wis"], spread["wis"], z["wis"])
        verdict_cov = _classify(delta_vs_target["cov95"], spread["cov95"], z["cov95"])
        # Combined verdict: worst of the two metrics
        order = {"MC_NOISE": 0, "MARGINAL": 1, "BUG_SUSPECT": 2, "INSUFFICIENT_N": 3}
        verdict = max([verdict_wis, verdict_cov], key=lambda v: order.get(v, 99))

        print(f"  → probe: WIS mean={mean['wis']:.4f} σ={std['wis']:.4f} spread={spread['wis']:.4f}", flush=True)
        print(f"           Cov95 mean={mean['cov95']:.4f} σ={std['cov95']:.4f} spread={spread['cov95']:.4f}", flush=True)
        print(f"  → integration v2 vs probe mean: ", flush=True)
        print(f"           WIS z={z['wis']:.2f} (Δ_vs_disk={delta_vs_target['wis']:+.4f})", flush=True)
        print(f"           Cov95 z={z['cov95']:.2f} (Δ_vs_disk={delta_vs_target['cov95']:+.4f})", flush=True)
        print(f"  → verdict: WIS={verdict_wis}  Cov95={verdict_cov}  COMBINED={verdict}", flush=True)

        results[baseline] = {
            "n_reruns": n_reruns,
            "reruns": reruns,
            "mean": mean,
            "std": std,
            "spread": spread,
            "disk_target": target,
            "integration_v2_actual": actual,
            "delta_vs_disk_target": delta_vs_target,
            "z_vs_probe_mean": z,
            "verdict_wis": verdict_wis,
            "verdict_cov95": verdict_cov,
            "verdict": verdict,
        }

    elapsed_total = time.time() - t_global

    # Overall: if any baseline BUG_SUSPECT → block full run
    fail_list = [b for b, r in results.items() if r["verdict"] == "BUG_SUSPECT"]
    marginal_list = [b for b, r in results.items() if r["verdict"] == "MARGINAL"]
    overall_verdict = "BUG_SUSPECT" if fail_list else ("MARGINAL" if marginal_list else "MC_NOISE")

    payload = {
        "purpose": "Direct MC-instance variance probe for 4 MC-NN baselines (LSTM/VM/PatchTST/EpiDeep) at seed=42. Replaces training-seed σ basis (wrong noise model) with same-model MC-RNG-only variance.",
        "seed": SEED,
        "n_reruns": n_reruns,
        "regions": list(REGIONS),
        "horizons": list(HORIZONS),
        "rationale": "Integration test compares same seed=42 weights × different MC RNG instance, so MC-instance σ (not training-seed σ) is the correct noise scale for the |Δ| envelope.",
        "verdict_legend": {
            "MC_NOISE": "|z|<2 AND probe spread ≥ |Δ vs disk target| → forward verified",
            "MARGINAL": "2≤|z|<3 OR spread within 2× of |Δ| → likely noise but watch full-run 5-seed reproduction",
            "BUG_SUSPECT": "|z|≥3 OR spread < |Δ|/2 → real forward difference, investigate before full run",
        },
        "baselines": results,
        "overall_verdict": overall_verdict,
        "bug_suspect_list": fail_list,
        "marginal_list": marginal_list,
        "elapsed_sec_total": elapsed_total,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[save] {OUT_JSON}", flush=True)
    print(f"\n{'='*70}", flush=True)
    print(f"OVERALL VERDICT: {overall_verdict}", flush=True)
    if fail_list:
        print(f"BUG_SUSPECT: {fail_list}  → STOP full-run; investigate forward", flush=True)
    if marginal_list:
        print(f"MARGINAL: {marginal_list}  → proceed to full-run; watch 5-seed Table IV", flush=True)
    print(f"{'='*70}", flush=True)
    return 0 if overall_verdict != "BUG_SUSPECT" else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baselines", default="lstm,vanilla_mamba,patchtst,epideep",
                    help="Comma-separated baselines (default: all 4 MC-NN).")
    ap.add_argument("--n_reruns", type=int, default=3,
                    help="N reruns per baseline (default 3, recommend 5 for VM).")
    args = ap.parse_args()
    bl = [b.strip() for b in args.baselines.split(",") if b.strip()]
    invalid = [b for b in bl if b not in BUILDERS]
    if invalid:
        print(f"[error] unknown baseline(s): {invalid}; valid: {list(BUILDERS)}", flush=True)
        sys.exit(2)
    sys.exit(main(args.device, bl, args.n_reruns))
