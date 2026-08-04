"""scripts/p5_interpretability_extract.py — σ² components re-extraction.

LOCK: paper/interpretability_pre_registration.md (v2.2 FROZEN, append §14.1).

Per §14.1: deterministic re-execution of existing CG-Mamba forward (5 seeds × 10 regions
× 4 horizons × 149 weeks) for the SOLE purpose of extracting per-cell σ² decomposition
components computed by the existing `cgm_decomp_forward` code path. Same locked
checkpoints, same locked seeds, same code — produces byte-identical predictions
(Track B v4 integration test: |Δ|=0.0000 CGM bit-identical).

Outputs:
  runs/interpretability/sigma_components.parquet — per (seed, region, h, week)
    columns: seed, region, h, eps_h1, mu_cgm, sigma2_within, sigma2_between, sigma2_total,
             bias_sq, gamma_all_0, gamma_all_1, gamma_all_2, y_z, y_raw

3-gate reproduction verification (§14.1):
  (i)   Aggregate gate: WIS/Cov95/MAE per (seed,region,h) == Track B native_* within 1e-6
  (ii)  Decomposition identity gate:
          sigma2_total == sigma2_within + sigma2_between (|Δ| < 1e-6)
          AND bias_sq >= 0 for all cells
  (iii) σ²_total → native_cov95 reproduction:
          reconstruct 95% PI from (mu_cgm, sigma2_total) → per-cell coverage
          → LOCK §5 aggregation (per-cell → region-mean per (seed,h) → h-mean → 5-seed mean)
          == Track B regional native_cov95 (0.9548) within 1e-6
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch
from scipy.stats import norm as scipy_norm
import scripts.track_b_lib as tbl
from scripts.track_b_lib import (
    load_norm, build_region_df, load_cgm_model_seed, cgm_dataset,
    HORIZONS, NORM_PATH,
)
from src.eval.hmm_interval import compute_decomposition
from src.data.loader import load_norm_params

# Locked constants (per Track B parent LOCK §1 + cold-start parent LOCK §9)
SEEDS = (42, 123, 456, 789, 1024)
REGIONS = tuple(f"hhs{i}" for i in range(1, 11))


OUT_DIR = _ROOT / "runs" / "interpretability"
OUT_PARQUET = OUT_DIR / "sigma_components.parquet"
TRACK_B_PARQUET = _ROOT / "runs" / "track_b_full" / "per_cell.parquet"

# LOCK §14.1 (iii) [as corrected by §14.2 append 2026-06-27]:
# Target is computed dynamically from `runs/track_b_full/per_cell.parquet` `native_cov95`
# column via LOCK §5 aggregation order (per-cell → cross-region per (seed,h) → h-mean →
# 5-seed mean). Reference value at LOCK time: 0.9547651006711408. The 4-digit display
# value "0.9548" in §14.1 was a clerical transcription error; the full-precision value
# is used for the gate per §14.2.
GATE_TOLERANCE = 1e-6


def extract_for_seed_region(seed, region, device, norm):
    """Per (seed, region): run CGM forward, return per-cell decomposition + y."""
    model, cfg, hmm = load_cgm_model_seed(seed, device)

    means = hmm.means
    covars = hmm.covars
    mu_k_ili = means[:, 0]
    if covars.ndim == 3:
        sigma2_k_ili = np.array([covars[k, 0, 0] for k in range(covars.shape[0])])
    else:
        sigma2_k_ili = covars[:, 0]

    region_df = build_region_df(region)
    ds = cgm_dataset(region_df, "test", cfg, norm)
    n = len(ds)
    if n == 0:
        return None

    eps_arr = region_df["epiweek"].astype(int).to_numpy()
    norm_p = load_norm_params(NORM_PATH)
    ili_p = norm_p["ili_weighted_pct"]
    target_mean = float(ili_p["mean"])
    target_std = float(ili_p["std"])
    target_z_full = (region_df["ili_weighted_pct"].to_numpy() - target_mean) / target_std

    mu = np.zeros((n, 4))
    gamma_all = np.zeros((n, 4, 3))
    y_z = np.zeros((n, 4))
    eps_h1 = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)

    with torch.no_grad():
        for i in range(n):
            d = ds[i]
            x = d["x"].unsqueeze(0).to(device)
            env = d["env"].unsqueeze(0).to(device)
            pred, inter = model(x, env, return_intermediates=True)
            if torch.isnan(pred).any():
                valid[i] = False
                continue
            mu[i] = pred[0].cpu().numpy()
            gamma_all[i] = inter["gamma_all"][0].cpu().numpy()
            tgt_ep = int(d["target_epiweek"])
            tgt_idx = int(np.where(eps_arr == tgt_ep)[0][0])
            for h_idx, h in enumerate(HORIZONS):
                src = tgt_idx - (max(HORIZONS) - h)
                if 0 <= src < len(eps_arr):
                    y_z[i, h_idx] = target_z_full[src]
            eps_h1[i] = eps_arr[tgt_idx - (max(HORIZONS) - 1)]

    mu = mu[valid]
    gamma_all = gamma_all[valid]
    y_z = y_z[valid]
    eps_h1 = eps_h1[valid]

    decomp = compute_decomposition(mu, gamma_all, mu_k_ili, sigma2_k_ili)
    y_raw = y_z * target_std + target_mean

    # Restrict to test_strict (eps_h1 >= 202240) per LOCK §3 + Track B parent
    TS_BOUNDARY = 202240
    ts_mask = eps_h1 >= TS_BOUNDARY

    return {
        "n_strict": int(ts_mask.sum()),
        "eps_h1": eps_h1[ts_mask],
        "mu_cgm_z": decomp.mu_CGM[ts_mask],
        "mu_hmm_z": decomp.mu_HMM[ts_mask],
        "sigma2_within_z": decomp.sigma2_within[ts_mask],
        "sigma2_between_z": decomp.sigma2_between_HMM[ts_mask],
        "sigma2_total_z": decomp.sigma2_total[ts_mask],
        "bias_sq_z": decomp.bias_sq[ts_mask],
        "gamma_all": gamma_all[ts_mask],
        "y_z": y_z[ts_mask],
        "y_raw": y_raw[ts_mask],
        "target_mean": target_mean,
        "target_std": target_std,
    }


def gate_check(rows_df, track_b_df):
    """3-gate reproduction verification per §14.1."""
    print("\n" + "=" * 70, flush=True)
    print("§14.1 3-GATE REPRODUCTION VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    # === Gate (ii): Decomposition identity ===
    print("\n[Gate (ii)] σ²_total == σ²_within + σ²_between AND bias² ≥ 0", flush=True)
    delta = (rows_df["sigma2_total_z"]
             - rows_df["sigma2_within_z"]
             - rows_df["sigma2_between_z"])
    max_abs_delta = float(delta.abs().max())
    bias_min = float(rows_df["bias_sq_z"].min())
    gate_ii_identity = max_abs_delta < GATE_TOLERANCE
    gate_ii_bias_nn = bias_min >= -GATE_TOLERANCE  # allow tiny float noise
    print(f"  max |σ²_total − (σ²_within + σ²_between)| = {max_abs_delta:.3e} (tol {GATE_TOLERANCE})", flush=True)
    print(f"  min bias² = {bias_min:.3e} (must be ≥ 0)", flush=True)
    print(f"  Gate (ii): identity={'PASS' if gate_ii_identity else 'FAIL'}, "
          f"bias²≥0={'PASS' if gate_ii_bias_nn else 'FAIL'}", flush=True)

    # === Gate (i): Aggregate WIS/Cov95/MAE match Track B ===
    # Reconstruct quantiles from (mu_cgm_z, sigma2_total_z) — raw native APMD = Gaussian PI
    # Aggregate to per (seed, region, h) cells.
    print("\n[Gate (i)] Aggregate WIS/Cov95/MAE per (seed,region,h) == Track B native_* within 1e-6", flush=True)

    # Build per-cell stats per (seed, region, h)
    agg_rows = []
    for (seed, region, h), grp in rows_df.groupby(["seed", "region", "h"]):
        sigma_z = np.sqrt(grp["sigma2_total_z"].to_numpy())
        mu_z = grp["mu_cgm_z"].to_numpy()
        y_z = grp["y_z"].to_numpy()
        target_std = grp["target_std"].iloc[0]
        target_mean = grp["target_mean"].iloc[0]

        # Raw native APMD = Gaussian(mu, sigma_total) on z, denormalize for raw scale
        mu_raw = mu_z * target_std + target_mean
        sigma_raw = sigma_z * target_std
        y_raw = grp["y_raw"].to_numpy()

        # MAE
        mae = float(np.mean(np.abs(mu_raw - y_raw)))

        # 95% PI: μ ± 1.96·σ
        lo95 = mu_raw + scipy_norm.ppf(0.025) * sigma_raw
        hi95 = mu_raw + scipy_norm.ppf(0.975) * sigma_raw
        cov95 = float(np.mean((y_raw >= lo95) & (y_raw <= hi95)))

        # WIS: 23 FluSight quantiles
        from src.eval.wis_standard import FLUSIGHT_23, wis
        qf = {}
        for tau in FLUSIGHT_23:
            qf[float(tau)] = mu_raw + scipy_norm.ppf(float(tau)) * sigma_raw
        wis_val = float(np.mean(wis(y_raw, qf)))

        agg_rows.append({
            "seed": seed, "region": region, "h": h,
            "wis": wis_val, "cov95": cov95, "mae": mae,
        })

    agg_df = pd.DataFrame(agg_rows)

    # Match to Track B per_cell.parquet native_* columns
    tb = track_b_df[track_b_df.baseline == "cg_mamba"].copy()
    tb = tb[["seed", "region", "h", "native_wis", "native_cov95", "native_mae"]]
    merged = agg_df.merge(tb, on=["seed", "region", "h"], how="inner")
    merged["delta_wis"] = (merged["wis"] - merged["native_wis"]).abs()
    merged["delta_cov"] = (merged["cov95"] - merged["native_cov95"]).abs()
    merged["delta_mae"] = (merged["mae"] - merged["native_mae"]).abs()
    max_dwis = float(merged["delta_wis"].max())
    max_dcov = float(merged["delta_cov"].max())
    max_dmae = float(merged["delta_mae"].max())
    n_cells = len(merged)
    gate_i = (max_dwis < GATE_TOLERANCE) and (max_dcov < GATE_TOLERANCE) and (max_dmae < GATE_TOLERANCE)
    print(f"  cells compared: {n_cells} (expected: 5 seeds × 10 regions × 4 horizons = 200)", flush=True)
    print(f"  max |Δ WIS| = {max_dwis:.3e} (tol {GATE_TOLERANCE})", flush=True)
    print(f"  max |Δ Cov95| = {max_dcov:.3e} (tol {GATE_TOLERANCE})", flush=True)
    print(f"  max |Δ MAE| = {max_dmae:.3e} (tol {GATE_TOLERANCE})", flush=True)
    print(f"  Gate (i): {'PASS' if gate_i else 'FAIL'}", flush=True)

    # === Gate (iii): σ²_total → reconstructed Cov95 == Track B regional native_cov95 (parquet full-precision) ===
    # Per §14.2 correction: target computed dynamically from Track B parquet, not from
    # 4-digit display value. Aggregation order identical for reproduced and target
    # (LOCK §5: per-cell → cross-region per (seed,h) → h-mean → 5-seed mean).
    tb_cgm_native = track_b_df[track_b_df.baseline == "cg_mamba"]
    per_seed_h_target = tb_cgm_native.groupby(["seed", "h"])["native_cov95"].mean().reset_index()
    per_seed_target = per_seed_h_target.groupby("seed")["native_cov95"].mean()
    target_full = float(per_seed_target.mean())

    per_seed_h_repro = agg_df.groupby(["seed", "h"])["cov95"].mean().reset_index()
    per_seed_repro = per_seed_h_repro.groupby("seed")["cov95"].mean()
    reproduced = float(per_seed_repro.mean())
    delta_target = abs(reproduced - target_full)
    gate_iii = delta_target < GATE_TOLERANCE

    print(f"\n[Gate (iii)] σ²_total → reconstructed Cov95 == Track B parquet native_cov95 (full-precision, §14.2)", flush=True)
    print(f"  reconstructed (5-seed-mean h-mean cross-region) = {reproduced:.16f}", flush=True)
    print(f"  target (parquet, same aggregation)              = {target_full:.16f}", flush=True)
    print(f"  |Δ| = {delta_target:.3e} (tol {GATE_TOLERANCE})", flush=True)
    print(f"  Gate (iii): {'PASS' if gate_iii else 'FAIL'}", flush=True)

    all_pass = gate_i and gate_ii_identity and gate_ii_bias_nn and gate_iii
    print("\n" + "-" * 70, flush=True)
    print(f"§14.1 3-GATE OVERALL: {'ALL PASS' if all_pass else 'FAIL'}", flush=True)
    print("-" * 70, flush=True)
    return all_pass


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[extract] device={device}", flush=True)
    print(f"[extract] seeds={SEEDS}, regions={len(REGIONS)}, horizons={HORIZONS}", flush=True)
    print(f"[extract] output: {OUT_PARQUET}", flush=True)

    norm = load_norm()
    rows = []
    t_start = time.time()

    for seed in SEEDS:
        for region in REGIONS:
            print(f"  [{seed}/{region}] forward...", flush=True)
            data = extract_for_seed_region(seed, region, device, norm)
            if data is None:
                print(f"    empty test set; skipping", flush=True)
                continue
            n = data["n_strict"]
            for i in range(n):
                for h_idx, h in enumerate(HORIZONS):
                    rows.append({
                        "seed": seed,
                        "region": region,
                        "h": h,
                        "eps_h1": int(data["eps_h1"][i]),
                        "week_idx": i,
                        "mu_cgm_z": float(data["mu_cgm_z"][i, h_idx]),
                        "mu_hmm_z": float(data["mu_hmm_z"][i, h_idx]),
                        "sigma2_within_z": float(data["sigma2_within_z"][i, h_idx]),
                        "sigma2_between_z": float(data["sigma2_between_z"][i, h_idx]),
                        "sigma2_total_z": float(data["sigma2_total_z"][i, h_idx]),
                        "bias_sq_z": float(data["bias_sq_z"][i, h_idx]),
                        "gamma_all_0": float(data["gamma_all"][i, h_idx, 0]),
                        "gamma_all_1": float(data["gamma_all"][i, h_idx, 1]),
                        "gamma_all_2": float(data["gamma_all"][i, h_idx, 2]),
                        "y_z": float(data["y_z"][i, h_idx]),
                        "y_raw": float(data["y_raw"][i, h_idx]),
                        "target_mean": data["target_mean"],
                        "target_std": data["target_std"],
                    })
        elapsed = time.time() - t_start
        print(f"  seed={seed} done; total rows={len(rows)}, elapsed={elapsed/60:.1f}min", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n[save] writing {len(df)} rows to {OUT_PARQUET}", flush=True)
    df.to_parquet(OUT_PARQUET, index=False)

    # Run 3-gate verification
    track_b_df = pd.read_parquet(TRACK_B_PARQUET)
    all_pass = gate_check(df, track_b_df)

    if not all_pass:
        print("\n[STOP] §14.1 3-gate FAIL → analysis blocked. Debug before proceeding.", flush=True)
        return 1
    else:
        print(f"\n[OK] §14.1 3-gate ALL PASS. σ² components ready for analysis.", flush=True)
        print(f"  Total elapsed: {(time.time()-t_start)/60:.1f}min", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
