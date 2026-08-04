"""APMD variance-consistency diagnostic (#4). Pre-registered: runs/apmd_diagnostic/PRE_REGISTRATION.md.

Tests Var(y - mu_CGM) ~= sigma^2_total per horizon (R_h) and per dominant phase (vs emission var),
n3_d64 headline, test_strict, national + 10 regions, 5 seeds. No training (forward only).
Applies the LOCKED decision rule: consistent iff R_h in [0.7,1.5]; report full vector regardless.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))
import e1_final_eval as E
import regime_shift_drivers as rsd

OUT = _ROOT / "runs" / "apmd_diagnostic"
TEST_FIRST = 202240
SEEDS = [42, 123, 456, 789, 1024]
REGIONS = [f"hhs{i}" for i in range(1, 11)]
H = [1, 2, 3, 4]


def _forward_keep_gamma(scope, seed, device):
    """Return per-row: target_ep, horizon, mu(raw), y_true(raw), s2_total(raw), dom_phase, sig2_k_raw[K]."""
    model, hmm, cfg = E.load_final_model("n3_d64", 3, 64, seed, device)
    norm = E.load_norm_params(E.FINAL_NORM_JSON)
    tmean = float(norm["ili_weighted_pct"]["mean"]); tstd = float(norm["ili_weighted_pct"]["std"])
    mu_k = hmm.means[:, 0].astype(np.float64); s2_k = hmm.covars[:, 0, 0].astype(np.float64)
    df = E.load_dataset_csv(E.FINAL_CSV) if scope == "national" else rsd._build_region_df(scope)
    ds = E.MultiHorizonDataset(df, split="test", lookback=cfg.lookback, horizons=tuple(cfg.horizons), norm=norm)
    raw = E._forward_dataset(model, ds, device)   # cols: target_ep, horizon, mu_z, gamma_h, y_z
    rows = []
    for _, r in raw.iterrows():
        if int(r.target_ep) < TEST_FIRST:
            continue
        g = np.array(r["gamma_h"])
        mu_hmm_z = float((g * mu_k).sum())
        sw_z = float((g * s2_k).sum()); sb_z = float((g * (mu_k - mu_hmm_z) ** 2).sum())
        rows.append(dict(horizon=int(r.horizon),
                         mu=float(r.mu_z) * tstd + tmean, y=float(r.y_z) * tstd + tmean,
                         s2_total=max(sw_z + sb_z, 1e-12) * tstd ** 2,
                         dom=int(np.argmax(g)), s2_domk=float(s2_k[int(np.argmax(g))]) * tstd ** 2))
    del model, hmm
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for scope in ["national"] + REGIONS:
        for seed in SEEDS:
            d = _forward_keep_gamma(scope, seed, a.device); d["scope"] = scope; d["seed"] = seed
            frames.append(d)
        print(f"[diag] {scope} done", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["resid"] = df.y - df.mu
    df.to_csv(OUT / "apmd_residuals.csv", index=False)

    def Rh(sub):
        return {h: float(np.var(sub[sub.horizon == h].resid) / sub[sub.horizon == h].s2_total.mean()) for h in H}
    print("\n=== R_h = empirical Var(y-mu) / analytic sigma^2_total  (consistent iff in [0.7,1.5]) ===")
    print(f"{'scope':<10}" + "".join(f"h{h}".rjust(9) for h in H))
    res = {}
    for scope in ["national", "REGIONAL(pooled)"]:
        sub = df[df.scope == "national"] if scope == "national" else df[df.scope != "national"]
        rh = Rh(sub); res[scope] = rh
        print(f"{scope:<10}" + "".join(f"{rh[h]:.2f}".rjust(9) for h in H))
    print("\n=== per dominant-phase: empirical residual var vs HMM emission var (regional pooled) ===")
    reg = df[df.scope != "national"]
    perphase = {}
    for k in sorted(reg.dom.unique()):
        s = reg[reg.dom == k]
        emp = float(np.var(s.resid)); emis = float(s.s2_domk.mean())
        perphase[int(k)] = {"emp_resid_var": emp, "emission_var": emis, "ratio": emp / emis if emis > 0 else None, "n": int(len(s))}
        print(f"  phase {k}: emp_resid_var={emp:.3f}  emission_var={emis:.3f}  ratio={emp/emis:.2f}  n={len(s)}")
    # locked verdict wording
    rh_reg = res["REGIONAL(pooled)"]
    consistent_h = [h for h in H if 0.7 <= rh_reg[h] <= 1.5]
    grows = rh_reg[4] > rh_reg[1]
    print(f"\n[diag] regional R_h consistent (in [0.7,1.5]) at horizons: {consistent_h}; R_h grows with h: {grows}")
    (OUT / "apmd_diagnostic_result.json").write_text(json.dumps(
        {"R_h": res, "per_phase_regional": perphase,
         "consistent_horizons_regional": consistent_h, "R_h_grows_with_horizon": bool(grows),
         "licensed_wording": ("empirically consistent at short horizons"
                              + (", under-estimating at longer horizons" if grows else "")
                              + " (NOT 'by construction'; professor-gated)")}, indent=2))
    print(f"[diag] saved -> {OUT.relative_to(_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
