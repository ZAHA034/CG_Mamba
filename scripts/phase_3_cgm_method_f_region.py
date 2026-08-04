"""§18 Phase 3 — CG-Mamba Method F per-region WIS (★ critical novelty).

For each region × seed:
  1. Forward CG-Mamba on regional val + test → μ_CGM + gamma_all
  2. Method F 3-component decomposition (regional)
  3. Per-horizon calibration s_h from regional val (grid search quantile-matching)
  4. Test (full + strict) quantiles → WIS + Cov95

Output: runs/phase_3_cgm_method_f_region.csv
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.models.cg_forecaster import CGForecaster
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import WeeklyDataset, load_norm_params
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES
from src.eval.hmm_interval import (
    compute_decomposition, calibrate_scale_quantile_matching,
)
from scripts.phase_3_region_eval import build_region_df

NORM = load_norm_params(_ROOT / "data" / "processed" / "normalization_params.json")
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD = float(NORM["ili_weighted_pct"]["std"])
HORIZONS = [1, 2, 3, 4]
TS_BOUNDARY = 202240
REGIONS = [f"hhs{i}" for i in range(1, 11)]
SEEDS = [42, 123, 456, 789, 1024]


def _forward_collect(model, ds, device):
    """Forward all windows in dataset, collect mu_CGM + gamma_all + y + eps_h1.

    Returns: mu_CGM [N, 4] (z-scored), gamma_all [N, 4, K], y_norm [N, 4], eps_h1 [N].
    """
    n = len(ds)
    eps_arr = ds.df["epiweek"].astype(int).to_numpy()
    target_z = (ds.df["ili_weighted_pct"].to_numpy() - TARGET_MEAN) / TARGET_STD
    mu_CGM = np.zeros((n, 4))
    gamma_all = np.zeros((n, 4, 3))
    y_norm = np.zeros((n, 4))
    eps_h1 = np.zeros(n, dtype=np.int64)
    valid_mask = np.ones(n, dtype=bool)

    model.eval().to(device)
    with torch.no_grad():
        for i in range(n):
            d = ds[i]
            x = d["x"].unsqueeze(0).to(device)
            env = d["env"].unsqueeze(0).to(device)
            pred, inter = model(x, env, return_intermediates=True)
            if torch.isnan(pred).any():
                valid_mask[i] = False
                continue
            mu_CGM[i] = pred[0].cpu().numpy()
            gamma_all[i] = inter["gamma_all"][0].cpu().numpy()  # [H, K]
            tgt_ep = int(d["target_epiweek"])
            tgt_idx = int(np.where(eps_arr == tgt_ep)[0][0])
            for h_idx, h in enumerate(HORIZONS):
                src = tgt_idx - (max(HORIZONS) - h)
                if 0 <= src < len(eps_arr):
                    y_norm[i, h_idx] = target_z[src]
            eps_h1[i] = eps_arr[tgt_idx - (max(HORIZONS) - 1)]
    return mu_CGM[valid_mask], gamma_all[valid_mask], y_norm[valid_mask], eps_h1[valid_mask]


def _wis_cov(samples_z, y_z, ts_mask):
    """Compute WIS + Cov95 per horizon. samples [S, N, H] z-scored, y [N, H] z-scored.

    Convert to raw scale for reporting.
    """
    samples = samples_z * TARGET_STD + TARGET_MEAN
    y = y_z * TARGET_STD + TARGET_MEAN
    out = {}
    for h_idx, h in enumerate(HORIZONS):
        qf = {q: np.quantile(samples[:, :, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
        out[f"tF_wis_h{h}"] = float(wis(y[:, h_idx], qf).mean())
        out[f"tF_cov95_h{h}"] = float(coverage(y[:, h_idx], qf, alpha=0.05))
        if ts_mask.sum() > 0:
            qf_ts = {q: np.quantile(samples[:, ts_mask, h_idx], q, axis=0) for q in REQUIRED_QUANTILES}
            out[f"tS_wis_h{h}"] = float(wis(y[ts_mask, h_idx], qf_ts).mean())
            out[f"tS_cov95_h{h}"] = float(coverage(y[ts_mask, h_idx], qf_ts, alpha=0.05))
    return out


def _samples_from_decomp(mu_CGM, sigma2_total, s_per_h, n_samples=100, seed=42):
    """Sample n synthetic predictions from calibrated Gaussian.

    samples_h ~ N(mu_CGM, sqrt(s_h * sigma2_total))
    """
    rng = np.random.RandomState(seed)
    N, H = mu_CGM.shape
    samples = np.zeros((n_samples, N, H))
    for h in range(H):
        sigma_h = np.sqrt(s_per_h[h] * sigma2_total[:, h] + 1e-12)
        samples[:, :, h] = rng.normal(loc=mu_CGM[None, :, h], scale=sigma_h[None, :], size=(n_samples, N))
    return samples


def eval_region_method_f(region, seed, device):
    """Compute Method F WIS for one (region, seed)."""
    region_df = build_region_df(region)

    # Load CG-Mamba M2.1 ckpt
    m_path = _ROOT / "runs/m2_4_data_efficiency/cg_mamba/seasons_17_seasons_full" / f"seed{seed}" / "manifest.json"
    m = json.load(open(m_path))
    hmm = load_fitted_hmm(Path(m["hmm_dir"]))
    cfg = CGMambaConfig()
    model = CGForecaster(cfg)
    model.prepare_for_stage2(hmm)
    ck = torch.load(Path(m["stage3_best"]), map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)

    # HMM emission stats (z-scored). Use stored means/covars from HMM checkpoint
    # HMM trained on V_aug=6 augmented features (x + Δx, 3+3 dims). Index 0 of V_aug
    # is the standardized ILI dim (matches target).
    # Phase emission means/covars in z-scored space
    means = hmm.means  # [K, V_aug=6]
    covars = hmm.covars  # [K, V_aug, V_aug] or [K, V_aug] depending on cov_type
    mu_k_ili = means[:, 0]  # [K], z-scored ILI mean per phase
    if covars.ndim == 3:
        sigma2_k_ili = np.array([covars[k, 0, 0] for k in range(covars.shape[0])])
    else:
        sigma2_k_ili = covars[:, 0]
    # All z-scored space

    # Val forward → calibrate s_h
    val_ds = WeeklyDataset(region_df, split="val", lookback=cfg.lookback,
                            horizon=max(cfg.horizons), norm=NORM)
    if len(val_ds) < 4:
        raise ValueError(f"{region} val too small: {len(val_ds)}")
    mu_val, gamma_val, y_val_z, _ = _forward_collect(model, val_ds, device)
    decomp_val = compute_decomposition(mu_val, gamma_val, mu_k_ili, sigma2_k_ili)
    s_per_h = calibrate_scale_quantile_matching(y_val_z, decomp_val)

    # Test forward → decomp → calibrated samples → WIS
    test_ds = WeeklyDataset(region_df, split="test", lookback=cfg.lookback,
                             horizon=max(cfg.horizons), norm=NORM)
    mu_test, gamma_test, y_test_z, eps_h1 = _forward_collect(model, test_ds, device)
    decomp_test = compute_decomposition(mu_test, gamma_test, mu_k_ili, sigma2_k_ili)

    # Sample from calibrated Gaussian
    samples_z = _samples_from_decomp(decomp_test.mu_CGM, decomp_test.sigma2_total, s_per_h)

    # WIS + Cov
    ts_mask = eps_h1 >= TS_BOUNDARY
    out = {"region": region, "baseline": "cg_mamba_method_F", "seed": seed,
           "n_full": len(eps_h1), "n_strict": int(ts_mask.sum()),
           "s_per_h": s_per_h.tolist()}
    out.update(_wis_cov(samples_z, y_test_z, ts_mask))
    return out


def main(device="cuda:0"):
    if not torch.cuda.is_available(): device = "cpu"
    rows = []
    print(f"Device: {device}", flush=True)
    for region in REGIONS:
        print(f"\n=== {region} ===", flush=True)
        for seed in SEEDS:
            try:
                r = eval_region_method_f(region, seed, device)
                rows.append(r)
                tS = r.get('tS_wis_h1', float('nan'))
                cov = r.get('tS_cov95_h1', float('nan'))
                print(f"  ✓ {region} s={seed}  tS_wis_h1={tS:.4f}  cov95={cov:.3f}  s_per_h={[f'{x:.2f}' for x in r['s_per_h']]}", flush=True)
            except Exception as e:
                import traceback
                print(f"  ✗ {region} s={seed}: {type(e).__name__}: {e}", flush=True)
                rows.append({"region": region, "baseline": "cg_mamba_method_F", "seed": seed, "error": str(e)})

    df_out = pd.DataFrame(rows)
    out_csv = _ROOT / "runs" / "phase_3_cgm_method_f_region.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  rows={len(df_out)}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    main(args.device)
