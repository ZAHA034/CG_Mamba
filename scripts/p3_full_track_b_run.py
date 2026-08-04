"""scripts/p3_full_track_b_run.py — Track B full 5-baseline run (LOCKED)

Per /A.I_DATA/jbnu/JeongHa/CG_Mamba/paper/track_b_sub_pre_registration.md (LOCKED 2026-06-21):
- 6 baselines: CGM (APMD), LSTM, Vanilla Mamba, PatchTST, DLinear, EpiDeep
- 5 seeds: 42, 123, 456, 789, 1024
- 10 HHS regions × 4 horizons
- raw base UQ + uniform Split-Conformal CQR via src/eval/wis_standard.quantiles_conformal_cqr
- per-region per-horizon aggregation, then mean over regions, then mean over horizons (LOCK §5)
- hard-stop b/c/d STOP inline; (a) PRINT-not-STOP per LOCK §6
- as-is rule per LOCK §7
- output: per-seed per-region per-h parquet + JSON summary

ckpt designations (disk-evidence locked, see 2026-06-21 audit):
  CGM       : runs/m1_8_stage3_train/m2_4_cg_mamba_17_seasons_full_s{seed}_stage3/best.pt  (via manifest)
  LSTM      : runs/lstm_final/h256_l2_lr5e-04_bs16/seed{seed}/lstm_best.pt  (force MC Dropout d=0.3)
  Vanilla M : runs/vanilla_mamba_final/d64_nl3_lr5e-04/seed{seed}/vanilla_mamba_best.pt  (force d=0.1 MC injection — β)
  PatchTST  : runs/patchtst_final/pl16_dm128_lr5e-04/seed{seed}/patchtst_best.pt  (274K, paper Table I)
  DLinear   : runs/dlinear_final/ma13_indF_lr2e-03/seed{seed}/dlinear_best.pt  (5-seed ensemble Gaussian, no MC)
  EpiDeep   : runs/epideep_final/<dim_lr_cfg>/seed{seed}/epideep_best.pt  (force d=0.1 MC injection)

Output:
  runs/track_b_full/per_cell.parquet          (baseline, seed, region, h, native_*, track_b_*)
  runs/track_b_full/summary.json              (per-baseline cross-region per-h-mean WIS/Cov95/MAE, native + track_b)
  runs/track_b_full/hard_stop_log.json        (all (b/c/d) triggers per LOCK §6)
  runs/track_b_full/p3_full_track_b_run.log
"""
from __future__ import annotations
import argparse, gc, json, subprocess, sys, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

# Add Time-Series-Library for PatchTST/DLinear/iTransformer/TimesNet imports if needed
_TSLIB = _ROOT.parent / "Time-Series-Library"
if _TSLIB.exists():
    sys.path.insert(0, str(_TSLIB))

# wis_standard = single source of truth (T5)
from src.eval.wis_standard import (
    FLUSIGHT_23, REQUIRED_QUANTILES, ALPHA_LEVELS,
    quantiles_from_gaussian, quantiles_from_samples,
    quantiles_conformal_cqr, wis, coverage,
)
from src.data.loader import load_norm_params

# Track B forward library — per-baseline val + region-test base-quantile builders
# (LSTM / VanillaMamba / PatchTST / EpiDeep MC Dropout, CGM APMD Gaussian,
#  DLinear 5-seed-ensemble Gaussian), plus conformal CQR per-h and per-cell scoring.
# Note: lib uses `build_cg_mamba_*` (not `build_cgm_*`); we alias on import for spec clarity.
from scripts.track_b_lib import (
    N_MC_SAMPLES, DROPOUT_MC as _LIB_DROPOUT_MC, load_norm,
    build_lstm_val_base_quantiles, build_lstm_region_test_quantiles,
    build_vanilla_mamba_val_base_quantiles, build_vanilla_mamba_region_test_quantiles,
    build_patchtst_val_base_quantiles, build_patchtst_region_test_quantiles,
    build_epideep_val_base_quantiles, build_epideep_region_test_quantiles,
    build_dlinear_val_base_quantiles, build_dlinear_region_test_quantiles,
    build_cg_mamba_val_base_quantiles as build_cgm_val_base_quantiles,
    build_cg_mamba_region_test_quantiles as build_cgm_region_test_quantiles,
    conformal_cqr_per_h, score_per_cell,
)

# === LOCKed constants (from track_b_sub_pre_registration.md) ===
SEEDS = [42, 123, 456, 789, 1024]
REGIONS = [f"hhs{i}" for i in range(1, 11)]
HORIZONS = [1, 2, 3, 4]
TEST_STRICT_START_EPIWEEK = 202240
TRAIN_END_EPIWEEK = 201839
DROPOUT_MC = {"lstm": 0.3, "vanilla_mamba": 0.1, "patchtst": 0.1, "epideep": 0.1}
MC_N_SAMPLES = 100
OUT_DIR = _ROOT / "runs" / "track_b_full"
NORM_PARAMS_PATH = _ROOT / "data" / "processed" / "normalization_params.json"

# === ckpt path templates (disk-evidence locked 2026-06-21) ===
CKPT_PATHS = {
    "cg_mamba": {
        "manifest": _ROOT / "runs/m2_4_data_efficiency/cg_mamba/seasons_17_seasons_full/seed{seed}/manifest.json",
    },
    "lstm": {
        "ckpt": _ROOT / "runs/lstm_final/h256_l2_lr5e-04_bs16/seed{seed}/lstm_best.pt",
        "config": _ROOT / "runs/lstm_final/h256_l2_lr5e-04_bs16/seed{seed}/results.json",
    },
    "vanilla_mamba": {
        "ckpt": _ROOT / "runs/vanilla_mamba_final/d64_nl3_lr5e-04/seed{seed}/vanilla_mamba_best.pt",
        "config": _ROOT / "runs/vanilla_mamba_final/d64_nl3_lr5e-04/seed{seed}/results.json",
    },
    "patchtst": {
        "ckpt": _ROOT / "runs/patchtst_final/pl16_dm128_lr5e-04/seed{seed}/patchtst_best.pt",
        "config": _ROOT / "runs/patchtst_final/pl16_dm128_lr5e-04/seed{seed}/results.json",
    },
    "dlinear": {
        "ckpt": _ROOT / "runs/dlinear_final/ma13_indF_lr2e-03/seed{seed}/dlinear_best.pt",
        "config": _ROOT / "runs/dlinear_final/ma13_indF_lr2e-03/seed{seed}/results.json",
    },
    "epideep": {
        # Disk-evidence locked 2026-06-21:
        #   - scripts/m2_3_eval_extra_baselines.py:42 → ("runs/epideep_final/de128_eh64_lr2e-03", "epideep_best.pt")
        #   - scripts/phase_3_region_wis_extras.py:55 → EPIDEEP_DIR = de128_eh64_lr2e-03
        #   - scripts/phase_3_region_eval_extras.py:126 → same
        #   - scripts/compare_baselines_forecast.py:149 → ("epideep", "de128_eh64_lr2e-03", "epideep_best.pt", 0.1, "mc")
        #   - 5 seeds all exist on disk under this config
        # Per LOCK §6(c) "no closest-available substitution": this is the ONE path; no glob.
        "ckpt": _ROOT / "runs/epideep_final/de128_eh64_lr2e-03/seed{seed}/epideep_best.pt",
        "config": _ROOT / "runs/epideep_final/de128_eh64_lr2e-03/seed{seed}/results.json",
    },
}

# === Hard-stop (d) classification (per smoke 2026-06-21 investigation) ===
# Known-GENUINE pattern: CGM raw APMD over-dispersed at h=1 in low-signal regions.
#   - LOCK §6(d) "debug" satisfied by smoke investigation (mechanism = CQR-symmetric
#     on val_in_PI=1.000 → strictly negative scores → strongly negative radius → over-correct
#     on low-signal regions). Not a methodology swap candidate per LOCK §7.
# Per-condition #2 (user 2026-06-21): full run classification must distinguish this
# known-genuine from UNEXPECTED (d) triggers (other baseline / other horizon / other region)
# which warrant bug-vs-genuine investigation before §V usage.
KNOWN_GENUINE_D_PATTERN = {
    # (baseline, h, regions): expected GENUINE per smoke
    ("cg_mamba", 1): {"hhs1", "hhs7", "hhs10"},   # low-signal regions per smoke 2026-06-21
}


def classify_hard_stop_d(baseline: str, h: int, region: str) -> str:
    """Returns 'KNOWN_GENUINE' (proceed, log) or 'UNEXPECTED' (full run completes,
    but verdict requires bug-vs-genuine investigation before §V).
    Per user condition #2 (2026-06-21).
    """
    key = (baseline, h)
    if key in KNOWN_GENUINE_D_PATTERN and region in KNOWN_GENUINE_D_PATTERN[key]:
        return "KNOWN_GENUINE"
    return "UNEXPECTED"


# === Integration test targets (Table IV cell, seed42 cross-region per-h-mean) ===
# Per user condition #3 (2026-06-21): each baseline native must reproduce Table IV cell
# (seed42 cross-region per-h-mean WIS+Cov95) within |Δ| < 0.005 BEFORE full run launch.
INTEGRATION_TEST_TARGETS_SEED42 = {
    # baseline: {"wis": Table IV WIS, "cov95": Table IV Cov95, "source": disk artifact}
    "cg_mamba":      {"wis": 0.3679, "cov95": 0.9295, "source": "phase_3_cgm_method_f_region.csv seed=42 mean"},
    "lstm":          {"wis": 0.4156, "cov95": 0.5131, "source": "phase_3_region_wis.py seed=42 mean (smoke reproduced 0.001 tolerance)"},
    "vanilla_mamba": {"wis": 0.463,  "cov95": 0.571,  "source": "phase_3_region_wis.py seed=42 mean (Table IV row, 5-seed mean; seed42 alone TBD)"},
    "patchtst":      {"wis": 0.423,  "cov95": 0.695,  "source": "phase_3_region_wis.py seed=42 mean (Table IV row, 5-seed mean; seed42 alone TBD)"},
    "dlinear":       {"wis": 0.509,  "cov95": 0.286,  "source": "phase_3_dlinear_ensemble_region.csv (ensemble = 5-seed; seed42 N/A; ensemble = target)"},
    "epideep":       {"wis": 0.515,  "cov95": 0.382,  "source": "phase_3_region_wis_extras.csv seed=42 mean (Table IV row, 5-seed mean; seed42 alone TBD)"},
}
INTEGRATION_TOLERANCE = 0.005  # |Δ| < 0.005 = PASS (Table IV 3-decimal rounding)


# === Table IV (5-seed mean) reproduction targets — full-run summary check ===
# Per user condition #1 (2026-06-22): EpiDeep AUTHORITATIVE LOCK is reproduction
# of the Table IV (5-seed mean) cell, |Δ|<0.005 on both WIS and Cov95. The other
# 5 baselines have disk-evidence ckpt locks already; their reproduction here is
# a consistency check, but EpiDeep here is the authoritative ckpt-path verdict.
#
# CGM caveat: Table IV cell (0.368/0.930) is the Method-F (per-h calibrated)
# result; Track B uses RAW APMD (LOCK §4 raw quantile, no s_per_h). So CGM
# cannot match Method-F directly. We disclose the Method-F target for reference,
# but the CGM consistency check compares Track B's 5-seed RAW APMD mean to the
# seed=42 RAW APMD smoke reproduction (0.4045/0.998) — across-seed consistency,
# NOT Table IV reproduction.
TABLE_IV_TARGETS_5SEED_MEAN = {
    "cg_mamba": {
        "method_f_table_iv": {"wis": 0.368, "cov95": 0.930,
                              "notes": "Method-F (per-h calibrated). NOT comparable to Track B RAW APMD."},
        "raw_apmd_smoke_seed42": {"wis": 0.4045, "cov95": 0.998,
                                  "notes": "2026-06-21 RAW APMD seed=42 smoke. Track B consistency check target (across-seed)."},
    },
    "lstm":          {"wis": 0.4156, "cov95": 0.5131,
                       "notes": "Table IV 5-seed mean."},
    "vanilla_mamba": {"wis": 0.463,  "cov95": 0.571,
                       "notes": "Table IV 5-seed mean."},
    "patchtst":      {"wis": 0.423,  "cov95": 0.695,
                       "notes": "Table IV 5-seed mean."},
    "dlinear":       {"wis": 0.509,  "cov95": 0.286,
                       "notes": "Table IV 5-seed Gaussian ensemble."},
    "epideep":       {"wis": 0.515,  "cov95": 0.382,
                       "notes": "Table IV 5-seed mean. AUTHORITATIVE LOCK for ckpt path runs/epideep_final/de128_eh64_lr2e-03 per user condition #1, 2026-06-22."},
}
TABLE_IV_TOLERANCE = 0.005  # |Δ| < 0.005 = PASS (Table IV 3-decimal rounding)


def table_iv_reproduction_check(df: pd.DataFrame) -> dict:
    """Compute per-baseline 5-seed-mean NATIVE cross-region per-h-mean WIS+Cov95
    and compare to Table IV (5-seed mean) targets.

    Aggregation order (LOCK §5):
        per-cell -> mean over 10 regions per (seed, h) -> mean over 4 horizons
        per seed -> mean over 5 seeds.

    Returns per-baseline dict with measured / target / delta / verdict.
    EpiDeep verdict is the AUTHORITATIVE LOCK (user condition #1, 2026-06-22).
    CGM: compares to RAW APMD seed=42 smoke (across-seed consistency), discloses
    Method-F Table IV mismatch.
    """
    out = {}
    baselines_present = sorted(df["baseline"].unique().tolist())
    for b in baselines_present:
        sub = df[df["baseline"] == b]
        # Per-seed per-h cross-region mean, then per-seed mean over h, then 5-seed mean.
        by_seed_h = sub.groupby(["seed", "h"]).agg(
            wis=("native_wis", "mean"),
            cov95=("native_cov95", "mean"),
        ).reset_index()
        by_seed = by_seed_h.groupby("seed").agg(
            wis=("wis", "mean"),
            cov95=("cov95", "mean"),
        ).reset_index()
        wis_5seed_mean = float(by_seed["wis"].mean())
        cov_5seed_mean = float(by_seed["cov95"].mean())
        n_seeds = int(len(by_seed))

        if b == "cg_mamba":
            tgt = TABLE_IV_TARGETS_5SEED_MEAN["cg_mamba"]
            method_f = tgt["method_f_table_iv"]
            raw_smoke = tgt["raw_apmd_smoke_seed42"]
            delta_method_f_wis = wis_5seed_mean - method_f["wis"]
            delta_method_f_cov = cov_5seed_mean - method_f["cov95"]
            delta_raw_wis = wis_5seed_mean - raw_smoke["wis"]
            delta_raw_cov = cov_5seed_mean - raw_smoke["cov95"]
            pass_raw = (abs(delta_raw_wis) < TABLE_IV_TOLERANCE and
                        abs(delta_raw_cov) < TABLE_IV_TOLERANCE)
            out[b] = {
                "measured_5seed_mean": {"wis": wis_5seed_mean, "cov95": cov_5seed_mean,
                                         "n_seeds": n_seeds},
                "table_iv_method_f": {
                    "target": method_f,
                    "delta": {"wis": delta_method_f_wis, "cov95": delta_method_f_cov},
                    "verdict": "EXPECTED_MISMATCH",
                    "notes": "Track B uses RAW APMD (LOCK §4), NOT Method-F. "
                             "Mismatch with 0.368/0.930 is expected and disclosed; "
                             "not a CGM failure mode.",
                },
                "raw_apmd_consistency": {
                    "target": raw_smoke,
                    "delta": {"wis": delta_raw_wis, "cov95": delta_raw_cov},
                    "tolerance": TABLE_IV_TOLERANCE,
                    "verdict": "PASS" if pass_raw else "FAIL",
                    "notes": "Across-seed (5-seed) consistency vs seed=42 RAW APMD "
                             "smoke. CGM is deterministic given fixed calibration; "
                             "small variation across seeds tests training-side stability.",
                },
                "authoritative": False,
            }
            continue

        tgt = TABLE_IV_TARGETS_5SEED_MEAN[b]
        delta_wis = wis_5seed_mean - tgt["wis"]
        delta_cov = cov_5seed_mean - tgt["cov95"]
        passed = (abs(delta_wis) < TABLE_IV_TOLERANCE and
                  abs(delta_cov) < TABLE_IV_TOLERANCE)
        verdict = "PASS" if passed else "FAIL"
        entry = {
            "measured_5seed_mean": {"wis": wis_5seed_mean, "cov95": cov_5seed_mean,
                                     "n_seeds": n_seeds},
            "target": {"wis": tgt["wis"], "cov95": tgt["cov95"]},
            "delta": {"wis": delta_wis, "cov95": delta_cov},
            "tolerance": TABLE_IV_TOLERANCE,
            "verdict": verdict,
            "notes": tgt["notes"],
            "authoritative": (b == "epideep"),
        }
        if b == "epideep":
            entry["lock_status"] = (
                "AUTHORITATIVE_LOCK_VERIFIED" if passed
                else "AUTHORITATIVE_LOCK_FAILED"
            )
            entry["lock_note"] = (
                "Per user condition #1 (2026-06-22): EpiDeep ckpt path "
                "runs/epideep_final/de128_eh64_lr2e-03 is AUTHORITATIVELY locked "
                "iff this 5-seed-mean reproduction matches Table IV (0.515/0.382) "
                f"within |Δ|<{TABLE_IV_TOLERANCE}. {'Locked.' if passed else 'NOT locked — ckpt path UNVERIFIED, downstream MUST NOT trust EpiDeep until this passes.'}"
            )
        out[b] = entry
    return out


REGION_CSV_TEMPLATE = _ROOT / "data" / "raw" / "cdc_ilinet" / "_phase3_phase6_fetch" / "{region}_full.csv"


# ============================================================================
# 1. FAIL-FAST PRE-FLIGHT — assert all 6 × 5 × (ckpt + cfg) + 10 regions exist
# ============================================================================
def preflight():
    """Pre-flight: assert all required artifacts exist before any forward.
    Per (B) review: fail-fast in <1 min to avoid 3-5h wasted on missing artifact.
    """
    t0 = time.time()
    missing = []

    # CGM: manifest must exist per seed; manifest references stage3 ckpt + hmm_dir
    for seed in SEEDS:
        manifest_p = Path(str(CKPT_PATHS["cg_mamba"]["manifest"]).format(seed=seed))
        if not manifest_p.exists():
            missing.append(f"CGM seed={seed} manifest: {manifest_p}")
            continue
        m = json.loads(manifest_p.read_text())
        # manifest['stage3_best'] is the ckpt; manifest['hmm_dir'] is HMM
        if not Path(m["stage3_best"]).exists():
            missing.append(f"CGM seed={seed} stage3_best: {m['stage3_best']}")
        if not Path(m["hmm_dir"]).exists():
            missing.append(f"CGM seed={seed} hmm_dir: {m['hmm_dir']}")

    # LSTM, VM, PatchTST, DLinear: ckpt + config per seed
    for baseline in ["lstm", "vanilla_mamba", "patchtst", "dlinear"]:
        for seed in SEEDS:
            ckpt_p = Path(str(CKPT_PATHS[baseline]["ckpt"]).format(seed=seed))
            cfg_p = Path(str(CKPT_PATHS[baseline]["config"]).format(seed=seed))
            if not ckpt_p.exists():
                missing.append(f"{baseline} seed={seed} ckpt: {ckpt_p}")
            if not cfg_p.exists():
                missing.append(f"{baseline} seed={seed} config: {cfg_p}")

    # EpiDeep: evidence-locked single path per user condition #1 (no glob)
    for seed in SEEDS:
        ckpt_p = Path(str(CKPT_PATHS["epideep"]["ckpt"]).format(seed=seed))
        cfg_p = Path(str(CKPT_PATHS["epideep"]["config"]).format(seed=seed))
        if not ckpt_p.exists():
            missing.append(f"epideep seed={seed} ckpt: {ckpt_p}")
        if not cfg_p.exists():
            missing.append(f"epideep seed={seed} config: {cfg_p}")

    # National scaler
    if not NORM_PARAMS_PATH.exists():
        missing.append(f"national scaler: {NORM_PARAMS_PATH}")

    # Region CSVs
    for region in REGIONS:
        rp = Path(str(REGION_CSV_TEMPLATE).format(region=region))
        if not rp.exists():
            missing.append(f"region CSV {region}: {rp}")

    # wis_standard module accessible
    try:
        _test_qf = quantiles_from_gaussian(np.array([1.0]), np.array([0.25]))
        _test_w = wis(np.array([1.3]), _test_qf)
        assert abs(float(_test_w[0]) - 0.16789357902474250) < 1e-12, "wis_standard worked-example mismatch"
    except Exception as e:
        missing.append(f"wis_standard module sanity: {e}")

    elapsed = time.time() - t0
    print(f"[pre-flight] {elapsed:.2f}s elapsed; {len(missing)} missing artifacts")
    if missing:
        print("[pre-flight] FAIL — listing all missing (hard-stop c per LOCK §6):")
        for m in missing[:30]:
            print(f"  ✗ {m}")
        if len(missing) > 30:
            print(f"  ... and {len(missing)-30} more")
        raise RuntimeError(f"pre-flight FAIL: {len(missing)} missing artifacts")
    print(f"[pre-flight] PASS: 6 baselines × {len(SEEDS)} seeds × {len(REGIONS)} regions × "
          f"{len(HORIZONS)} horizons = {6*len(SEEDS)*len(REGIONS)*len(HORIZONS)} cells expected")
    return {}  # epideep discovered → now evidence-locked path; no dynamic dict needed


# ============================================================================
# Integration test — per user condition #3: per-baseline native = Table IV cell
# ============================================================================
def run_integration_test(baseline: str, device: str) -> dict:
    """Run seed42 × 10 regions × native UQ (no CQR) → cross-region per-h-mean
    WIS+Cov95 → compare to INTEGRATION_TEST_TARGETS_SEED42[baseline].

    PASS condition: |Δ| < INTEGRATION_TOLERANCE on both WIS and Cov95.
    FAIL → baseline forward has a bug; full run BLOCKED for that baseline.

    Returns dict per baseline: {wis_measured, cov95_measured, wis_target, cov95_target,
                                  wis_delta, cov95_delta, verdict}.

    Implementation: subprocess to scripts/p3_integration_test.py with --baseline <name>;
    the test script is responsible for the actual seed42 × 10-region × native forward and
    writes a per-baseline results.json next to itself. We parse that JSON on exit code 0.
    Any non-zero exit code or missing results.json is propagated as RuntimeError (caller
    treats as FAIL and BLOCKS the full run per LOCK §3 + user condition #3).
    """
    test_script = _ROOT / "scripts" / "p3_integration_test.py"
    results_path = OUT_DIR / f"integration_test_{baseline}.json"
    # Best-effort clean of stale results so we never silently parse a previous run.
    if results_path.exists():
        results_path.unlink()

    cmd = [
        sys.executable, str(test_script),
        "--baseline", baseline,
        "--device", device,
        "--results-path", str(results_path),
    ]
    print(f"    [subprocess] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        # Indent child stdout for readability inside the parent log.
        for line in proc.stdout.rstrip().splitlines():
            print(f"      | {line}")
    if proc.returncode != 0:
        if proc.stderr:
            for line in proc.stderr.rstrip().splitlines():
                print(f"      ! {line}")
        raise RuntimeError(
            f"integration test subprocess FAILED for {baseline} "
            f"(exit={proc.returncode}); see stderr above"
        )
    if not results_path.exists():
        raise RuntimeError(
            f"integration test for {baseline} exited 0 but did not write "
            f"{results_path}; cannot verify Table IV cell"
        )
    with results_path.open() as f:
        result = json.load(f)
    # Sanity: required fields present
    required = {"wis_measured", "cov95_measured", "wis_target", "cov95_target",
                "wis_delta", "cov95_delta"}
    missing = required - set(result.keys())
    if missing:
        raise RuntimeError(
            f"integration test results.json for {baseline} missing fields: {sorted(missing)}"
        )
    return result


# ============================================================================
# 2. PER-BASELINE BASE-UQ ROUTING — paper-protocol forward + base quantiles
# ============================================================================
# Dispatcher dicts map baseline name -> track_b_lib val/region-test builder.
# Each VAL builder:  (seed, device, norm)            -> (qf_val:dict[tau,[N,H]],  y_val:[N,H])
# Each TEST builder: (seed, device, norm, region)    -> (qf_test:dict[tau,[N,H]], y_test:[N,H], eps_h1:[N])
# DLinear ignores `seed` internally (5-seed ensemble Gaussian); CGM uses APMD Gaussian
# from (μ, σ²_total) on raw scale (LOCK §4: NO s_per_h calibration applied at base level).
BUILD_VAL = {
    "cg_mamba":      build_cgm_val_base_quantiles,
    "lstm":          build_lstm_val_base_quantiles,
    "vanilla_mamba": build_vanilla_mamba_val_base_quantiles,
    "patchtst":      build_patchtst_val_base_quantiles,
    "dlinear":       build_dlinear_val_base_quantiles,
    "epideep":       build_epideep_val_base_quantiles,
}

BUILD_TEST = {
    "cg_mamba":      build_cgm_region_test_quantiles,
    "lstm":          build_lstm_region_test_quantiles,
    "vanilla_mamba": build_vanilla_mamba_region_test_quantiles,
    "patchtst":      build_patchtst_region_test_quantiles,
    "dlinear":       build_dlinear_region_test_quantiles,
    "epideep":       build_epideep_region_test_quantiles,
}


# ============================================================================
# 3. HARD-STOP CHECKS — (a) PRINT, (b/c/d) STOP per LOCK §6
# ============================================================================
def hard_stop_check_b_nan(qf_test: dict, baseline: str, seed: int, region: str, h: int):
    """(b) NaN/inf in conformal radius or quantiles → STOP."""
    for tau, q in qf_test.items():
        if not np.all(np.isfinite(q)):
            raise RuntimeError(
                f"HARD-STOP (b): NaN/inf in conformal quantile {tau} for "
                f"{baseline}/seed={seed}/region={region}/h={h}. "
                f"NO silent filtering, NO NaN replacement per LOCK §6(b)."
            )


def hard_stop_check_d_cov(cov95: float, baseline: str, seed: int, region: str, h: int,
                            stop_log: list):
    """(d) Per-region Cov95 outside [0.5, 1.0] — classification + log (no raise).

    Per user condition #2 (2026-06-21): full-run (d) triggers must be CLASSIFIED into:
      - KNOWN_GENUINE (smoke-confirmed mechanism, CGM × h=1 × hhs1/7/10): log + proceed
      - UNEXPECTED   (other): log + GATE the final §V verdict (full run completes, but
                              results.json verdict = 'BUG_VS_GENUINE_INVESTIGATION_NEEDED')

    LOCK §6(d) "debug" interpretation: smoke investigation already covered KNOWN_GENUINE;
    UNEXPECTED triggers require post-run investigation BEFORE §V usage. Run does NOT stop
    mid-flight (preserves the 3-5h work), but verdict gates downstream.
    """
    if cov95 < 0.5 or cov95 > 1.0:
        classification = classify_hard_stop_d(baseline, h, region)
        stop_log.append({
            "rule": "d",
            "baseline": baseline, "seed": seed,
            "region": region, "h": h,
            "cov95": float(cov95),
            "classification": classification,
        })
        marker = "✓" if classification == "KNOWN_GENUINE" else "⚠"
        print(f"  [hard-stop d {classification}] {marker} {baseline}/s{seed}/{region}/h={h} "
              f"cov95={cov95:.4f} outside [0.5, 1.0]")


def summarize_hard_stop_d_for_verdict(stop_log: list) -> dict:
    """Per user condition #2: gate the verdict on UNEXPECTED (d) count.
    Returns {known_genuine: int, unexpected: int, verdict: str, unexpected_list: [...]}.
    """
    d_entries = [e for e in stop_log if e.get("rule") == "d"]
    known = [e for e in d_entries if e.get("classification") == "KNOWN_GENUINE"]
    unexpected = [e for e in d_entries if e.get("classification") == "UNEXPECTED"]
    if len(unexpected) == 0:
        verdict = "CLEAN_PROCEED_TO_V"
    else:
        verdict = "BUG_VS_GENUINE_INVESTIGATION_NEEDED"
    return {
        "known_genuine_count": len(known),
        "unexpected_count": len(unexpected),
        "verdict": verdict,
        "unexpected_list": unexpected[:50],  # cap for brevity
        "known_genuine_list": known[:50],
    }


# ============================================================================
# 4. OUTPUT SCHEMA — per-seed per-region per-h
# ============================================================================
# parquet columns (canonical):
#   baseline:str, seed:int, region:str, h:int
#   n_strict:int
#   native_wis:float, native_cov95:float, native_mae:float
#   track_b_wis:float, track_b_cov95:float, track_b_mae:float
#   cqr_radius_h:float (diagnostic)
#
# JSON summary structure (per user condition #3 — per-seed breakdown for h1_over_shrink):
#   {
#     "per_baseline": {
#       "<baseline>": {
#         "native":  {"wis_mean": ..., "wis_std": ..., "cov95_mean": ..., "mae_mean": ..., ...},
#         "track_b": {"wis_mean": ..., "wis_std": ..., "cov95_mean": ..., ...,
#                       "cgm_lead_mean": ..., "cgm_lead_std": ..., "cgm_lead_per_seed": [...]},
#         "per_horizon": {h=1: {...}, h=2: {...}, ...}
#       }
#     },
#     "cross_baseline": {
#       "cgm_vs_each_baseline_lead":
#         {"lstm": {"native_per_seed": [...], "track_b_per_seed": [...], ...}, ...},
#       "f3_horizon_collapse":
#         {"native_LSTM_cov95_h_1_to_4_per_seed": [[s42],[s123],...],
#          "track_b_LSTM_cov95_h_1_to_4_per_seed": [...]},
#       "h1_over_shrink_by_region":
#         {"hhs1": {"native_cov95_per_seed": [s42,...,s1024],
#                    "track_b_cov95_per_seed": [s42,...,s1024],
#                    "consistency_across_seeds_below_0.5": int (count of seeds below 0.5)},
#          ...}    # ← per user condition #3: per-seed breakdown for genuine-vs-seed42-specific judgment
#     },
#     "hard_stop_log_summary":
#       {"b_count": 0, "c_count": 0,
#        "d_known_genuine_count": ..., "d_unexpected_count": ...,
#        "verdict": "CLEAN_PROCEED_TO_V" | "BUG_VS_GENUINE_INVESTIGATION_NEEDED"}    # ← gate on UNEXPECTED (d) per condition #2
#   }


# ============================================================================
# Main loop — fully iterative; saves per-baseline per-seed incrementally
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true",
                          help="Run pre-flight only; do not start any forward")
    parser.add_argument("--integration-test", action="store_true",
                          help="Run per-baseline native-reproduction test (seed42 × 10 regions × native UQ "
                               "→ compare cross-region per-h-mean WIS+Cov95 to Table IV cell, "
                               "|Δ|<0.005 PASS). Per user condition #3 (2026-06-21): full run BLOCKED until "
                               "each new baseline (VM/PatchTST/DLinear/EpiDeep) passes; LSTM/CGM already "
                               "verified by smoke 2026-06-21.")
    parser.add_argument("--baselines", nargs="+",
                          default=["cg_mamba", "lstm", "vanilla_mamba", "patchtst", "dlinear", "epideep"],
                          help="Subset of baselines (default all 6)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("# Track B full 5-baseline run (LOCKED per track_b_sub_pre_registration.md)")
    print("=" * 80)
    epideep_paths = preflight()
    if args.preflight_only:
        print("[--preflight-only] returning")
        return 0

    # ── Integration test gate (user condition #3) ──
    if args.integration_test:
        print(f"\n{'='*80}\n# Integration test: per-baseline native = Table IV cell verification\n{'='*80}")
        ALL_PASS = True
        integration_results = {}
        for baseline in args.baselines:
            target = INTEGRATION_TEST_TARGETS_SEED42.get(baseline)
            if target is None:
                print(f"  [skip] {baseline}: no Table IV target")
                continue
            print(f"\n  [{baseline}] seed42 × 10 regions × native UQ — target WIS={target['wis']:.3f} Cov95={target['cov95']:.3f}")
            try:
                result = run_integration_test(baseline, args.device)
                pass_wis = abs(result["wis_delta"]) < INTEGRATION_TOLERANCE
                pass_cov95 = abs(result["cov95_delta"]) < INTEGRATION_TOLERANCE
                verdict = "PASS" if (pass_wis and pass_cov95) else "FAIL"
                if not (pass_wis and pass_cov95):
                    ALL_PASS = False
                print(f"  → measured WIS={result['wis_measured']:.4f} (Δ={result['wis_delta']:+.4f}) "
                      f"Cov95={result['cov95_measured']:.4f} (Δ={result['cov95_delta']:+.4f}) — {verdict}")
                integration_results[baseline] = result
            except NotImplementedError as e:
                print(f"  → [pending implementation] {e}")
                integration_results[baseline] = {"verdict": "PENDING"}
                ALL_PASS = False
        with (OUT_DIR / "integration_test_results.json").open("w") as f:
            json.dump(integration_results, f, indent=2, default=str)
        print(f"\n[integration test] {'ALL PASS — full run authorized' if ALL_PASS else 'FAIL or PENDING — full run BLOCKED'}")
        return 0 if ALL_PASS else 1

    # Per-cell rows (parquet target)
    per_cell_rows = []
    hard_stop_log = []
    t_total = time.time()

    # National scaler — shared across every baseline / seed / region.
    norm = load_norm()

    for baseline in args.baselines:
        print(f"\n{'='*80}\n=== baseline = {baseline} ===\n{'='*80}", flush=True)
        if baseline not in BUILD_VAL or baseline not in BUILD_TEST:
            raise RuntimeError(
                f"unknown baseline '{baseline}'; choose from {sorted(BUILD_VAL.keys())}"
            )

        for seed in SEEDS:
            t_seed = time.time()
            print(f"\n  --- {baseline} / seed={seed} ---", flush=True)

            # ── (A) Build national VAL base quantiles ONCE per (baseline, seed) ──
            # qf_val: dict[tau -> np.ndarray[N_val, H]]; y_val: np.ndarray[N_val, H]
            try:
                qf_val, y_val = BUILD_VAL[baseline](seed, args.device, norm)
            except Exception as e:
                print(f"    [HARD-STOP] VAL forward {baseline}/s{seed}: {e}")
                traceback.print_exc()
                raise

            for region in REGIONS:
                try:
                    # ── (B) Build region TEST base quantiles ──
                    qf_test, y_test, eps_h1 = BUILD_TEST[baseline](
                        seed, args.device, norm, region
                    )
                    n_strict = int(y_test.shape[0])

                    for h_idx, h in enumerate(HORIZONS):
                        # Native cell (no CQR): slice base quantiles at this h, score.
                        qf_nat_h = {float(t): np.asarray(qf_test[float(t)][:, h_idx])
                                    for t in FLUSIGHT_23}
                        nat = score_per_cell(
                            qf_nat_h, y_test, h_idx,
                            label=f"native/{baseline}/s{seed}/{region}/h={h}",
                        )

                        # Track B (CQR-symmetric Split Conformal) — per-h calibration.
                        # conformal_cqr_per_h raises on non-finite (hard-stop b).
                        qf_track_b_h = conformal_cqr_per_h(
                            base_val=qf_val,
                            base_test=qf_test,
                            y_val=y_val,
                            baseline=baseline,
                            region=region,
                            h_idx=h_idx,
                            hard_stop_log=hard_stop_log,
                        )
                        trk = score_per_cell(
                            qf_track_b_h, y_test, h_idx,
                            label=f"track_b/{baseline}/s{seed}/{region}/h={h}",
                        )

                        # CQR radius diagnostic: (q_0.975 - q_0.025) post-CQR mean width.
                        cqr_width = float(np.mean(
                            qf_track_b_h[0.975] - qf_track_b_h[0.025]
                        ))

                        # (d) Cov95 outside [0.5, 1.0] — classify + log (no raise).
                        hard_stop_check_d_cov(
                            trk["cov95"], baseline, seed, region, h, hard_stop_log,
                        )

                        per_cell_rows.append({
                            "baseline": baseline, "seed": int(seed),
                            "region": region, "h": int(h),
                            "n_strict": n_strict,
                            "native_wis": nat["wis"],
                            "native_cov95": nat["cov95"],
                            "native_mae": nat["mae"],
                            "track_b_wis": trk["wis"],
                            "track_b_cov95": trk["cov95"],
                            "track_b_mae": trk["mae"],
                            "cqr_radius_h": cqr_width,
                        })

                    print(f"    [{region}] N_strict={n_strict} cells={len(HORIZONS)} ✓",
                          flush=True)
                except Exception as e:
                    print(f"    [HARD-STOP] {region}: {e}")
                    traceback.print_exc()
                    raise

            print(f"  --- {baseline}/seed={seed} elapsed={time.time()-t_seed:.1f}s ---",
                  flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Save per-cell parquet ────────────────────────────────────────────────
    df = pd.DataFrame(per_cell_rows)
    df.to_parquet(OUT_DIR / "per_cell.parquet", index=False)
    print(f"\n[save] per-cell parquet: {OUT_DIR / 'per_cell.parquet'}  rows={len(df)}")

    # ── Save hard-stop log ───────────────────────────────────────────────────
    with (OUT_DIR / "hard_stop_log.json").open("w") as f:
        json.dump(hard_stop_log, f, indent=2, default=str)
    print(f"[save] hard-stop log: {OUT_DIR / 'hard_stop_log.json'}  entries={len(hard_stop_log)}")

    # ── Build summary.json ───────────────────────────────────────────────────
    summary = build_summary(df, hard_stop_log)
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] summary: {OUT_DIR / 'summary.json'}")

    print(f"\n[done] total elapsed {(time.time()-t_total)/60:.1f} min")
    return 0


# ============================================================================
# 5. SUMMARY AGGREGATION — per_cell.parquet -> summary.json
# ============================================================================
def _agg_native_trackb(sub: pd.DataFrame) -> dict:
    """Aggregate over a sub-DataFrame: cross-region per-h-mean then mean over h.

    LOCK §5 aggregation order: per-cell → mean over regions per (seed, h) →
    mean over h per seed → mean ± std over seeds.
    """
    out = {}
    for tag in ("native", "track_b"):
        # Per-seed per-h cross-region mean (10 regions averaged).
        by_seed_h = sub.groupby(["seed", "h"]).agg(
            wis=(f"{tag}_wis", "mean"),
            cov95=(f"{tag}_cov95", "mean"),
            mae=(f"{tag}_mae", "mean"),
        ).reset_index()
        # Per-seed mean over h (4 horizons averaged).
        by_seed = by_seed_h.groupby("seed").agg(
            wis=("wis", "mean"),
            cov95=("cov95", "mean"),
            mae=("mae", "mean"),
        ).reset_index()
        out[tag] = {
            "wis_mean": float(by_seed["wis"].mean()),
            "wis_std": float(by_seed["wis"].std(ddof=1)) if len(by_seed) > 1 else 0.0,
            "wis_per_seed": by_seed.set_index("seed")["wis"].to_dict(),
            "cov95_mean": float(by_seed["cov95"].mean()),
            "cov95_std": float(by_seed["cov95"].std(ddof=1)) if len(by_seed) > 1 else 0.0,
            "cov95_per_seed": by_seed.set_index("seed")["cov95"].to_dict(),
            "mae_mean": float(by_seed["mae"].mean()),
            "mae_std": float(by_seed["mae"].std(ddof=1)) if len(by_seed) > 1 else 0.0,
        }
        # Per-horizon breakdown (cross-region per-seed then mean over seeds).
        per_h = {}
        for h, grp in by_seed_h.groupby("h"):
            per_h[int(h)] = {
                "wis_mean": float(grp["wis"].mean()),
                "wis_std": float(grp["wis"].std(ddof=1)) if len(grp) > 1 else 0.0,
                "cov95_mean": float(grp["cov95"].mean()),
                "cov95_std": float(grp["cov95"].std(ddof=1)) if len(grp) > 1 else 0.0,
                "mae_mean": float(grp["mae"].mean()),
                "wis_per_seed": grp.set_index("seed")["wis"].to_dict(),
                "cov95_per_seed": grp.set_index("seed")["cov95"].to_dict(),
            }
        out[tag]["per_horizon"] = per_h
    return out


def build_summary(df: pd.DataFrame, hard_stop_log: list) -> dict:
    """Assemble the full summary.json per the schema in §4 of this script."""
    summary = {"per_baseline": {}, "cross_baseline": {}}

    # Per-baseline cross-region per-h-mean WIS/Cov95/MAE (native + track_b).
    baselines_present = sorted(df["baseline"].unique().tolist())
    for b in baselines_present:
        summary["per_baseline"][b] = _agg_native_trackb(df[df["baseline"] == b])

    # ── Cross-baseline: CGM-vs-each lead (Δ WIS, positive = CGM wins) ────────
    cross = {}
    cgm_lead = {}
    if "cg_mamba" in baselines_present:
        cgm = df[df["baseline"] == "cg_mamba"]
        # Per-seed CGM cross-region per-h mean WIS (native + track_b).
        cgm_by_seed = {}
        for tag in ("native", "track_b"):
            tmp = cgm.groupby(["seed", "h"])[f"{tag}_wis"].mean().reset_index()
            tmp = tmp.groupby("seed")[f"{tag}_wis"].mean()
            cgm_by_seed[tag] = tmp  # Series indexed by seed
        for b in baselines_present:
            if b == "cg_mamba":
                continue
            sub = df[df["baseline"] == b]
            entry = {}
            for tag in ("native", "track_b"):
                tmp = sub.groupby(["seed", "h"])[f"{tag}_wis"].mean().reset_index()
                tmp = tmp.groupby("seed")[f"{tag}_wis"].mean()
                # Align on common seeds
                common = sorted(set(cgm_by_seed[tag].index) & set(tmp.index))
                lead = [float(tmp.loc[s] - cgm_by_seed[tag].loc[s]) for s in common]
                entry[f"{tag}_per_seed"] = {int(s): v for s, v in zip(common, lead)}
                entry[f"{tag}_mean"] = float(np.mean(lead)) if lead else float("nan")
                entry[f"{tag}_std"] = float(np.std(lead, ddof=1)) if len(lead) > 1 else 0.0
            cgm_lead[b] = entry
    cross["cgm_vs_each_baseline_lead"] = cgm_lead

    # ── f3_horizon_collapse: LSTM per-seed Cov95 across h=1..4 ───────────────
    f3 = {}
    if "lstm" in baselines_present:
        sub = df[df["baseline"] == "lstm"]
        for tag in ("native", "track_b"):
            # per-seed list of [cov95@h=1, cov95@h=2, cov95@h=3, cov95@h=4]
            # (region averaged within each (seed, h))
            grp = sub.groupby(["seed", "h"])[f"{tag}_cov95"].mean().reset_index()
            per_seed = []
            for seed in sorted(grp["seed"].unique()):
                row = grp[grp["seed"] == seed].sort_values("h")[f"{tag}_cov95"].tolist()
                per_seed.append([float(x) for x in row])
            f3[f"{tag}_LSTM_cov95_h_1_to_4_per_seed"] = per_seed
    cross["f3_horizon_collapse"] = f3

    # ── h1_over_shrink_by_region: per-region per-seed Cov95 at h=1, all baselines ──
    # Aggregated PER BASELINE (key = baseline -> region -> per-seed series).
    h1_dict = {}
    h1 = df[df["h"] == 1]
    for b in baselines_present:
        sub = h1[h1["baseline"] == b]
        by_region = {}
        for region, grp in sub.groupby("region"):
            grp_sorted = grp.sort_values("seed")
            seeds_list = grp_sorted["seed"].tolist()
            nat_cov = grp_sorted["native_cov95"].tolist()
            trk_cov = grp_sorted["track_b_cov95"].tolist()
            n_below_nat = int(sum(1 for v in nat_cov if v < 0.5))
            n_below_trk = int(sum(1 for v in trk_cov if v < 0.5))
            by_region[region] = {
                "seeds": [int(s) for s in seeds_list],
                "native_cov95_per_seed": [float(v) for v in nat_cov],
                "track_b_cov95_per_seed": [float(v) for v in trk_cov],
                "native_consistency_across_seeds_below_0.5": n_below_nat,
                "track_b_consistency_across_seeds_below_0.5": n_below_trk,
            }
        h1_dict[b] = by_region
    cross["h1_over_shrink_by_region"] = h1_dict

    summary["cross_baseline"] = cross

    # ── Hard-stop log summary (condition #2 verdict gate) ────────────────────
    d_summary = summarize_hard_stop_d_for_verdict(hard_stop_log)
    b_count = sum(1 for e in hard_stop_log if isinstance(e, dict) and e.get("rule") == "b")
    c_count = sum(1 for e in hard_stop_log if isinstance(e, dict) and e.get("rule") == "c")
    summary["hard_stop_log_summary"] = {
        "b_count": b_count,
        "c_count": c_count,
        "d_known_genuine_count": d_summary["known_genuine_count"],
        "d_unexpected_count": d_summary["unexpected_count"],
        "verdict": d_summary["verdict"],
        "d_unexpected_list": d_summary["unexpected_list"],
        "d_known_genuine_list": d_summary["known_genuine_list"],
    }

    # ── Table IV reproduction check (user condition #1, 2026-06-22) ──────────
    # 5-seed-mean native cross-region per-h-mean WIS/Cov95 vs Table IV per
    # baseline. EpiDeep verdict here = AUTHORITATIVE LOCK for ckpt path
    # runs/epideep_final/de128_eh64_lr2e-03. CGM: discloses Method-F mismatch,
    # checks across-seed RAW APMD consistency vs seed=42 smoke.
    summary["table_iv_reproduction_check"] = table_iv_reproduction_check(df)
    return summary


if __name__ == "__main__":
    sys.exit(main())
