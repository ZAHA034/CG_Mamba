"""scripts/p4_cold_start_sweep.py — Cold-Start calibration-data-scarcity sweep.

LOCK: paper/cold_start_pre_registration.md (LOCKED 2026-06-23, with append §11).
PARENT LOCK: paper/track_b_sub_pre_registration.md (LOCKED 2026-06-21).

DESIGN (LOCK §3)
----------------
For each (baseline, seed, region):
  1. Forward national-train val pool → (qf_val, y_val). Cached per (baseline, seed).
  2. Forward test_strict region → (qf_test, y_test, eps_h1). Sort all by eps_h1.
  3. Pre-compute STALE-conformal radius per h via CQR(qf_val, y_val) → for stale on
     same eval-idx as fresh (fair-eval comparison).
  4. For K ∈ {5, 10, 20, 40}:
       [first_K]    cal_idx = first K (time-ordered),   eval_idx = remaining
       [random_K]   B=20 subsamples × random K cells,    eval_idx = remaining (per b)
     Per (K, mode, subsample_id, h):
       fresh_q  = CQR(cal=(test_sorted[cal_idx, h], y[cal_idx, h]), apply to test[eval_idx, h])
       native_q = test_sorted[eval_idx, h]   # no CQR
       stale_q  = stale_h_calibrated[eval_idx]
       Score WIS + Cov95 on each, on the SAME eval_idx (fair-eval).
  5. Output per_cell parquet with (baseline, seed, region, h, K, mode, subsample_id,
     n_cal, n_eval, fresh_wis, fresh_cov95, native_wis, native_cov95, stale_wis,
     stale_cov95).

LOCK §3.4 — Only CGM-raw-APMD is genuinely n_cal-independent. Other baselines'
native MC-Dropout intervals are under-calibrated; reported for completeness only.

LOCK §5 HARD-STOPS
------------------
  (a) Cov95 ∉ [0, 1] OR NaN → STOP
  (b) Conformal radius NaN/inf → STOP (via wis_standard internal checks)
  (c) split disjoint verify: cal_idx ∩ eval_idx = ∅ AND (cal_idx ∪ eval_idx) = [0..148]
  (d) Smoke flag: LSTM full n_cal → Track B match (WIS 0.3676, Cov95 0.8738) |Δ|<0.005

CLI
---
    python3 scripts/p4_cold_start_sweep.py --device cuda:0
    python3 scripts/p4_cold_start_sweep.py --smoke --device cuda:0       # LSTM only, K=full sanity
    python3 scripts/p4_cold_start_sweep.py --baselines lstm,cg_mamba --device cuda:0
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch
import scripts.track_b_lib as tbl
from scripts.track_b_lib import (
    FLUSIGHT_23, HORIZONS, score_per_cell, load_norm,
    quantiles_conformal_cqr,
)


# ============================================================================
# Constants (LOCK §9)
# ============================================================================
REGIONS = tuple(f"hhs{i}" for i in range(1, 11))
SEEDS = (42, 123, 456, 789, 1024)
K_VALUES = (5, 10, 20, 40)
B_SUBSAMPLES = 20
N_TEST_STRICT_EXPECTED = 149

BASELINES_ALL = ("cg_mamba", "lstm", "vanilla_mamba", "patchtst", "dlinear", "epideep")

BUILD_VAL = {
    "cg_mamba":      tbl.build_cg_mamba_val_base_quantiles,
    "lstm":          tbl.build_lstm_val_base_quantiles,
    "vanilla_mamba": tbl.build_vanilla_mamba_val_base_quantiles,
    "patchtst":      tbl.build_patchtst_val_base_quantiles,
    "dlinear":       tbl.build_dlinear_val_base_quantiles,
    "epideep":       tbl.build_epideep_val_base_quantiles,
}

BUILD_TEST = {
    "cg_mamba":      tbl.build_cg_mamba_region_test_quantiles,
    "lstm":          tbl.build_lstm_region_test_quantiles,
    "vanilla_mamba": tbl.build_vanilla_mamba_region_test_quantiles,
    "patchtst":      tbl.build_patchtst_region_test_quantiles,
    "dlinear":       tbl.build_dlinear_region_test_quantiles,
    "epideep":       tbl.build_epideep_region_test_quantiles,
}

OUT_DIR = _ROOT / "runs" / "cold_start"


# ============================================================================
# Helpers
# ============================================================================
def _rng_for(baseline, seed, region, K, b):
    """Deterministic RNG for random_K subsample b. Reproducible."""
    key = f"{baseline}|{seed}|{region}|{K}|{b}".encode()
    seed_int = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.default_rng(seed_int)


def _slice_quantile_dict_per_h(qf_dict, h_idx):
    """qf_dict: dict {τ → [N, H]} → dict {τ → [N]} for given horizon."""
    return {tau: np.asarray(qf_dict[tau])[:, h_idx] for tau in qf_dict}


def _slice_quantile_dict_indices(qf_per_h, idx):
    """qf_per_h: dict {τ → [N]} → dict {τ → [len(idx)]}."""
    return {tau: qf_per_h[tau][idx] for tau in qf_per_h}


def _verify_split_disjoint(cal_idx, eval_idx, n_total, label):
    """LOCK §5 (c) — split disjoint hard-stop."""
    cal_set = set(int(x) for x in cal_idx)
    eval_set = set(int(x) for x in eval_idx)
    if cal_set & eval_set:
        raise RuntimeError(f"[{label}] HARD-STOP (c): cal/eval overlap, intersection={cal_set & eval_set}")
    if len(cal_set) + len(eval_set) != n_total:
        raise RuntimeError(f"[{label}] HARD-STOP (c): cal+eval != total, "
                           f"cal={len(cal_set)} eval={len(eval_set)} total={n_total}")


def _verify_cov95(cov, label):
    """LOCK §5 (a) — Cov95 plausibility hard-stop."""
    if np.isnan(cov) or cov < 0.0 or cov > 1.0:
        raise RuntimeError(f"[{label}] HARD-STOP (a): Cov95={cov} ∉ [0,1] or NaN")


# ============================================================================
# Per-baseline-seed evaluation
# ============================================================================
def process_baseline_seed(baseline: str, seed: int, device: str, norm: dict,
                            rows_out: list, t_start: float):
    """One (baseline, seed): forward val + 10 regions of test, sweep K + mode + b."""
    print(f"\n  --- {baseline}/seed={seed} ---", flush=True)
    t0 = time.time()

    # ----- Forward national VAL once per (baseline, seed) for stale-CQR cache -----
    try:
        qf_val, y_val = BUILD_VAL[baseline](seed, device, norm)
    except Exception as e:
        print(f"    [HARD-STOP] VAL forward {baseline}/s{seed}: {e}", flush=True)
        raise

    # Per-region loop
    for region in REGIONS:
        # ----- Forward TEST_STRICT for this region -----
        try:
            qf_test, y_test, eps_h1 = BUILD_TEST[baseline](seed, device, norm, region)
        except Exception as e:
            print(f"    [HARD-STOP] TEST forward {baseline}/s{seed}/{region}: {e}", flush=True)
            raise

        n_test = y_test.shape[0]
        if n_test != N_TEST_STRICT_EXPECTED:
            print(f"    [warn] {baseline}/s{seed}/{region}: n_test={n_test} != expected {N_TEST_STRICT_EXPECTED}", flush=True)

        # Sort by eps_h1 ascending (time order, for first_K determinism)
        order = np.argsort(eps_h1)
        qf_test_sorted = {tau: np.asarray(qf_test[tau])[order] for tau in qf_test}
        y_test_sorted = np.asarray(y_test)[order]
        eps_h1_sorted = eps_h1[order]

        # ----- Pre-compute STALE CQR per h once -----
        # stale_test_calibrated_per_h[h_idx] = dict {τ → [n_test]}
        stale_test_calibrated_per_h = {}
        for h_idx in range(len(HORIZONS)):
            qf_val_h = _slice_quantile_dict_per_h(qf_val, h_idx)
            y_val_h = np.asarray(y_val)[:, h_idx]
            qf_test_h = _slice_quantile_dict_per_h(qf_test_sorted, h_idx)
            stale_test_calibrated_per_h[h_idx] = quantiles_conformal_cqr(
                qf_val_h, qf_test_h, y_val_h, alpha_target=0.05, taus=FLUSIGHT_23,
            )

        # ----- Sweep K × {first, random} × h -----
        for K in K_VALUES:
            for mode, get_indices in _mode_iter(K, n_test, baseline, seed, region):
                # get_indices yields (subsample_id, cal_idx, eval_idx)
                for sub_id, cal_idx, eval_idx in get_indices:
                    label = f"{baseline}/s{seed}/{region}/K{K}/{mode}/b{sub_id}"
                    _verify_split_disjoint(cal_idx, eval_idx, n_test, label)

                    for h_idx, h in enumerate(HORIZONS):
                        # Slice base quantiles per h
                        qf_test_h = _slice_quantile_dict_per_h(qf_test_sorted, h_idx)
                        cal_base_q = _slice_quantile_dict_indices(qf_test_h, cal_idx)
                        eval_base_q = _slice_quantile_dict_indices(qf_test_h, eval_idx)
                        cal_y = y_test_sorted[cal_idx, h_idx]
                        eval_y = y_test_sorted[eval_idx, h_idx]

                        # ----- FRESH conformal -----
                        try:
                            fresh_calibrated_q = quantiles_conformal_cqr(
                                cal_base_q, eval_base_q, cal_y,
                                alpha_target=0.05, taus=FLUSIGHT_23,
                            )
                            fresh = score_per_cell(fresh_calibrated_q, eval_y, h_idx=0,
                                                     label=f"{label}/h={h}/fresh")
                            _verify_cov95(fresh["cov95"], f"{label}/h={h}/fresh")
                        except Exception as e:
                            print(f"    [{label}/h={h}] FRESH CQR fail: {e}", flush=True)
                            raise

                        # ----- NATIVE (no CQR; meaningful only for CGM per LOCK §3.4) -----
                        native = score_per_cell(eval_base_q, eval_y, h_idx=0,
                                                  label=f"{label}/h={h}/native")
                        _verify_cov95(native["cov95"], f"{label}/h={h}/native")

                        # ----- STALE on same eval_idx -----
                        stale_eval_q = _slice_quantile_dict_indices(
                            stale_test_calibrated_per_h[h_idx], eval_idx,
                        )
                        stale = score_per_cell(stale_eval_q, eval_y, h_idx=0,
                                                 label=f"{label}/h={h}/stale")
                        _verify_cov95(stale["cov95"], f"{label}/h={h}/stale")

                        rows_out.append({
                            "baseline": baseline, "seed": seed, "region": region, "h": h,
                            "K": K, "mode": mode, "subsample_id": sub_id,
                            "n_cal": len(cal_idx), "n_eval": len(eval_idx),
                            "fresh_wis": fresh["wis"], "fresh_cov95": fresh["cov95"], "fresh_mae": fresh["mae"],
                            "native_wis": native["wis"], "native_cov95": native["cov95"], "native_mae": native["mae"],
                            "stale_wis": stale["wis"], "stale_cov95": stale["cov95"], "stale_mae": stale["mae"],
                        })

        elapsed = time.time() - t0
        global_elapsed = time.time() - t_start
        n_rows = len(rows_out)
        print(f"    [{baseline}/s{seed}/{region}] done — region_elapsed={elapsed:.1f}s, "
              f"total_elapsed={global_elapsed:.0f}s, accumulated_rows={n_rows}", flush=True)

    print(f"  --- {baseline}/s{seed} done [{time.time()-t0:.1f}s] ---", flush=True)


def _mode_iter(K, n_test, baseline, seed, region):
    """Yield (mode_name, indices_generator) — generator yields (sub_id, cal_idx, eval_idx)."""
    # First_K
    def first_gen():
        cal_idx = np.arange(K)
        eval_idx = np.arange(K, n_test)
        yield None, cal_idx, eval_idx
    yield "first", first_gen()

    # Random_K × B
    def random_gen():
        for b in range(B_SUBSAMPLES):
            rng = _rng_for(baseline, seed, region, K, b)
            cal_idx = np.sort(rng.choice(n_test, K, replace=False))
            eval_idx = np.setdiff1d(np.arange(n_test), cal_idx)
            yield b, cal_idx, eval_idx
    yield "random", random_gen()


# ============================================================================
# Smoke (LOCK §5 (d) + §11.5 — LSTM 'full' n_cal ≈ Track B 0.3676/0.8738)
# ============================================================================
def smoke(device: str):
    """LSTM stale-conformal sanity: directly verify Track B parquet matches LOCK §11.5.
    NO re-forward; reads runs/track_b_full/per_cell.parquet directly.
    """
    print("[smoke] LOCK §11.5 hard-stop (d): LSTM stale (= Track B track_b_*) check", flush=True)
    p = pd.read_parquet(_ROOT / "runs/track_b_full/per_cell.parquet")
    sub = p[p.baseline == "lstm"]
    per_seed_wis, per_seed_cov = [], []
    for s in SEEDS:
        ss = sub[sub.seed == s]
        per_h_wis = [ss[ss.h == h]["track_b_wis"].mean() for h in HORIZONS]
        per_h_cov = [ss[ss.h == h]["track_b_cov95"].mean() for h in HORIZONS]
        per_seed_wis.append(np.mean(per_h_wis))
        per_seed_cov.append(np.mean(per_h_cov))
    measured_wis = float(np.mean(per_seed_wis))
    measured_cov = float(np.mean(per_seed_cov))
    target_wis, target_cov = 0.3676, 0.8738
    dw, dc = abs(measured_wis - target_wis), abs(measured_cov - target_cov)
    passed = (dw < 0.005) and (dc < 0.005)
    print(f"  measured WIS={measured_wis:.4f} (target {target_wis}, |Δ|={dw:.4f})", flush=True)
    print(f"  measured Cov95={measured_cov:.4f} (target {target_cov}, |Δ|={dc:.4f})", flush=True)
    print(f"  smoke (d): {'PASS' if passed else 'FAIL'}", flush=True)

    # Split disjoint unit test
    print(f"\n[smoke] split disjoint unit test", flush=True)
    n_test = N_TEST_STRICT_EXPECTED
    for K in K_VALUES:
        cal = np.arange(K)
        ev = np.arange(K, n_test)
        try:
            _verify_split_disjoint(cal, ev, n_test, f"unit-first/K{K}")
            print(f"  first_K={K}: disjoint ✓", flush=True)
        except RuntimeError as e:
            print(f"  first_K={K}: FAIL: {e}", flush=True); passed = False
        for b in range(2):
            rng = _rng_for("test", 42, "hhs1", K, b)
            cal = np.sort(rng.choice(n_test, K, replace=False))
            ev = np.setdiff1d(np.arange(n_test), cal)
            try:
                _verify_split_disjoint(cal, ev, n_test, f"unit-random/K{K}/b{b}")
                print(f"  random_K={K} b={b}: disjoint ✓", flush=True)
            except RuntimeError as e:
                print(f"  random_K={K} b={b}: FAIL: {e}", flush=True); passed = False

    # Sample LSTM K=40 first single (region, seed) — verify fresh-CQR code path runs and produces sane values
    print(f"\n[smoke] sample LSTM K=40 first single (seed=42, hhs1) — sanity range check", flush=True)
    norm = load_norm()
    rows = []
    process_baseline_seed_subset("lstm", 42, ["hhs1"], [40], ["first"], device, norm, rows, time.time())
    if not rows:
        print(f"  no rows produced — FAIL", flush=True); passed = False
    else:
        df = pd.DataFrame(rows)
        print(f"  rows produced: {len(df)}", flush=True)
        print(f"  fresh_wis: min={df.fresh_wis.min():.4f} max={df.fresh_wis.max():.4f}", flush=True)
        print(f"  fresh_cov95: min={df.fresh_cov95.min():.4f} max={df.fresh_cov95.max():.4f}", flush=True)
        if not ((df.fresh_wis > 0.05).all() and (df.fresh_wis < 2.0).all()):
            print(f"  WIS out of sanity range [0.05, 2.0] — FAIL", flush=True); passed = False
        if not ((df.fresh_cov95 >= 0).all() and (df.fresh_cov95 <= 1).all()):
            print(f"  Cov95 out of [0,1] — FAIL", flush=True); passed = False
        print(f"  sample sanity: {'PASS' if passed else 'FAIL'}", flush=True)

    return 0 if passed else 1


def process_baseline_seed_subset(baseline, seed, regions, K_list, modes, device, norm,
                                   rows_out, t_start):
    """Subset variant of process_baseline_seed for smoke. Calls main routine with limits."""
    # For smoke simplicity, replicate the inner loop with constrained K and modes.
    qf_val, y_val = BUILD_VAL[baseline](seed, device, norm)
    for region in regions:
        qf_test, y_test, eps_h1 = BUILD_TEST[baseline](seed, device, norm, region)
        n_test = y_test.shape[0]
        order = np.argsort(eps_h1)
        qf_test_sorted = {tau: np.asarray(qf_test[tau])[order] for tau in qf_test}
        y_test_sorted = np.asarray(y_test)[order]
        # Stale per h
        stale_test_calibrated_per_h = {}
        for h_idx in range(len(HORIZONS)):
            qf_val_h = _slice_quantile_dict_per_h(qf_val, h_idx)
            y_val_h = np.asarray(y_val)[:, h_idx]
            qf_test_h = _slice_quantile_dict_per_h(qf_test_sorted, h_idx)
            stale_test_calibrated_per_h[h_idx] = quantiles_conformal_cqr(
                qf_val_h, qf_test_h, y_val_h, alpha_target=0.05, taus=FLUSIGHT_23,
            )
        for K in K_list:
            for mode in modes:
                if mode == "first":
                    cal_idx = np.arange(K); eval_idx = np.arange(K, n_test); sub_id = None
                else:
                    rng = _rng_for(baseline, seed, region, K, 0)
                    cal_idx = np.sort(rng.choice(n_test, K, replace=False))
                    eval_idx = np.setdiff1d(np.arange(n_test), cal_idx); sub_id = 0
                _verify_split_disjoint(cal_idx, eval_idx, n_test, f"smoke-{baseline}-{region}-K{K}")
                for h_idx, h in enumerate(HORIZONS):
                    qf_test_h = _slice_quantile_dict_per_h(qf_test_sorted, h_idx)
                    cal_base_q = _slice_quantile_dict_indices(qf_test_h, cal_idx)
                    eval_base_q = _slice_quantile_dict_indices(qf_test_h, eval_idx)
                    cal_y = y_test_sorted[cal_idx, h_idx]
                    eval_y = y_test_sorted[eval_idx, h_idx]
                    fresh_q = quantiles_conformal_cqr(cal_base_q, eval_base_q, cal_y,
                                                        alpha_target=0.05, taus=FLUSIGHT_23)
                    fresh = score_per_cell(fresh_q, eval_y, h_idx=0, label=f"smk/{baseline}/{region}/h={h}")
                    native = score_per_cell(eval_base_q, eval_y, h_idx=0, label=f"smk-native/{region}/h={h}")
                    stale_eval_q = _slice_quantile_dict_indices(stale_test_calibrated_per_h[h_idx], eval_idx)
                    stale = score_per_cell(stale_eval_q, eval_y, h_idx=0, label=f"smk-stale/{region}/h={h}")
                    rows_out.append({
                        "baseline": baseline, "seed": seed, "region": region, "h": h,
                        "K": K, "mode": mode, "subsample_id": sub_id,
                        "n_cal": len(cal_idx), "n_eval": len(eval_idx),
                        "fresh_wis": fresh["wis"], "fresh_cov95": fresh["cov95"], "fresh_mae": fresh["mae"],
                        "native_wis": native["wis"], "native_cov95": native["cov95"], "native_mae": native["mae"],
                        "stale_wis": stale["wis"], "stale_cov95": stale["cov95"], "stale_mae": stale["mae"],
                    })


# ============================================================================
# Main
# ============================================================================
def main(device: str, baselines: list, smoke_only: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print(f"[warn] CUDA not available, falling back to cpu", flush=True)
        device = "cpu"

    if smoke_only:
        return smoke(device)

    print(f"[cold-start] device={device}", flush=True)
    print(f"  baselines={baselines}", flush=True)
    print(f"  seeds={SEEDS}, regions={len(REGIONS)}, K_values={K_VALUES}", flush=True)
    print(f"  B_subsamples={B_SUBSAMPLES}", flush=True)
    print(f"  LOCK: paper/cold_start_pre_registration.md (LOCKED 2026-06-23, append §11)", flush=True)

    norm = load_norm()
    rows_out = []
    t_start = time.time()

    for baseline in baselines:
        if baseline not in BUILD_VAL:
            raise RuntimeError(f"unknown baseline '{baseline}'; choose from {list(BUILD_VAL.keys())}")
        print(f"\n{'='*80}\n=== baseline = {baseline} ===\n{'='*80}", flush=True)
        for seed in SEEDS:
            process_baseline_seed(baseline, seed, device, norm, rows_out, t_start)

    out_parquet = OUT_DIR / "per_cell.parquet"
    df = pd.DataFrame(rows_out)
    df.to_parquet(out_parquet, index=False)
    print(f"\n[save] per_cell: {out_parquet}  rows={len(df)}", flush=True)
    print(f"[done] total elapsed {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baselines", default=",".join(BASELINES_ALL),
                    help="Comma-separated baselines (default: all 6).")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke: LOCK §11.5 hard-stop (d) verify only (no full sweep).")
    args = ap.parse_args()
    bl = [b.strip() for b in args.baselines.split(",") if b.strip()]
    sys.exit(main(args.device, bl, args.smoke))
