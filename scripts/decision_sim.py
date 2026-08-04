"""Decision-simulation harness (newsvendor / critical-fractile) — pre-registered, result-blind.

Reads a forecast-quantile dump and scores DECISIONS, not forecasts:
  1. calibration-of-decision: realized shortfall rate at fixed tau (must ~= 1-tau if calibrated)
  2. cost-optimal decision: realized total cost over a Cu/Co ratio sweep (rho), tau*=rho/(rho+1)

Dump schema (parquet): rows = (model, scope, origin_ep, horizon, y_true) + quantile columns
named 'q{tau:.4f}' on a shared tau grid. Any tau is obtained by per-row linear interpolation.

USAGE:
  python scripts/decision_sim.py --selftest                 # validate newsvendor math (no data)
  python scripts/decision_sim.py --dump runs/decision_sim/forecast_quantiles.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "runs" / "decision_sim"
TAU_FIXED = [0.80, 0.90, 0.95, 0.975]
RHOS = np.unique(np.round(np.logspace(0, 2, 25), 3))          # Cu/Co in [1,100], log-spaced


def _q_cols(df):
    cols = [c for c in df.columns if c.startswith("q") and c[1:].replace(".", "").isdigit()]
    taus = np.array([float(c[1:]) for c in cols])
    order = np.argsort(taus)
    return [cols[i] for i in order], taus[order]


def _quantile_at(Q, grid, tau):
    """Per-row linear interpolation of the stored quantile function at scalar tau. Q:[N,G]."""
    return np.array([np.interp(tau, grid, Q[i]) for i in range(Q.shape[0])])


def realized_shortfall(y, Q, grid, tau):
    C = _quantile_at(Q, grid, tau)
    return float(np.mean(y > C)), C                            # shortfall = truth exceeds reserve


def cost_at(y, Q, grid, rho):
    tau = rho / (rho + 1.0)
    C = _quantile_at(Q, grid, tau)
    short = np.maximum(0.0, y - C)
    idle = np.maximum(0.0, C - y)
    return float(rho * short.sum() + idle.sum())               # Co=1, Cu=rho


def run(dump_path):
    df = pd.read_parquet(dump_path)
    qc, grid = _q_cols(df)
    scopes = sorted(df.scope.unique())
    models = sorted(df.model.unique())
    print(f"[decision] {len(df)} rows | models={models} | scopes={len(scopes)} | tau-grid={len(grid)}")

    # ---- 1. calibration-of-decision diagnostic (pooled over scopes+horizons) ----
    print("\n=== realized shortfall rate at fixed tau (calibrated => ~= 1-tau) ===")
    print(f"{'model':<24}" + "".join(f"t={t}->{1-t:.3f}".rjust(14) for t in TAU_FIXED))
    diag = {}
    for m in models:
        s = df[df.model == m]
        y = s.y_true.to_numpy(); Q = s[qc].to_numpy()
        row = []
        for t in TAU_FIXED:
            sr, _ = realized_shortfall(y, Q, grid, t)
            row.append(sr)
        diag[m] = row
        print(f"{m:<24}" + "".join(f"{sr:.3f}".rjust(14) for sr in row))

    # ---- 2. cost-optimal decision: rho-curve per scope-family ----
    print("\n=== realized total cost vs rho (Cu/Co); lower=better; per scope ===")
    results = []
    for scope in scopes:
        for m in models:
            s = df[(df.model == m) & (df.scope == scope)]
            if len(s) == 0:
                continue
            y = s.y_true.to_numpy(); Q = s[qc].to_numpy()
            for rho in RHOS:
                results.append({"scope": scope, "model": m, "rho": float(rho),
                                "cost": cost_at(y, Q, grid, rho)})
    res = pd.DataFrame(results)

    # winner per (scope, rho) + decision-dominance summary
    print(f"\n--- cost winner per rho (pooled 'national' + 'regional-mean') ---")
    # aggregate regional scopes into a mean cost per model per rho
    res["family"] = res.scope.apply(lambda s: "national" if s == "national" else "regional")
    agg = res.groupby(["family", "model", "rho"]).cost.mean().reset_index()
    for fam in sorted(agg.family.unique()):
        print(f"\n  [{fam}]  rho -> cost-minimizing model")
        a = agg[agg.family == fam]
        cg_wins = []
        for rho in RHOS:
            ar = a[a.rho == rho].sort_values("cost")
            best = ar.iloc[0]["model"]
            cg_cost = ar[ar.model == "cg_mamba"].cost
            cg_rank = (ar.model.values == "cg_mamba").argmax() + 1 if "cg_mamba" in ar.model.values else -1
            if best == "cg_mamba":
                cg_wins.append(rho)
            if rho in (RHOS[0], RHOS[len(RHOS)//2], RHOS[-1]):
                print(f"    rho={rho:>7.2f}: best={best:<22} (CG rank {cg_rank})")
        if cg_wins:
            print(f"    -> CG-Mamba cost-optimal for rho in [{min(cg_wins):.2f}, {max(cg_wins):.2f}] "
                  f"({len(cg_wins)}/{len(RHOS)} rho points)")
        else:
            print(f"    -> CG-Mamba NEVER cost-optimal in this family")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "decision_cost_curve.csv", index=False)
    pd.DataFrame(diag, index=[f"tau{t}" for t in TAU_FIXED]).T.to_csv(OUT / "decision_shortfall_diag.csv")
    print(f"\n[decision] saved cost curve + shortfall diagnostic -> {OUT.relative_to(_ROOT)}/")
    return 0


def selftest():
    """Validate newsvendor math: calibrated => shortfall ~= 1-tau; overconfident => overshoots;
    calibrated cheaper at high rho."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.standard_normal(n)                                 # true demand ~ N(0,1)
    grid = np.round(np.arange(0.005, 1.0, 0.005), 4)
    from scipy.stats import norm as spn
    z = spn.ppf(grid)
    # calibrated forecaster: predictive N(0,1) (correct); overconfident: N(0,0.5) (too narrow)
    Q_cal = np.tile(0 + 1.0 * z, (n, 1))
    Q_over = np.tile(0 + 0.5 * z, (n, 1))
    print("=== SELFTEST (synthetic; y~N(0,1)) ===")
    print(f"{'tau':>6}{'1-tau':>9}{'calibrated':>13}{'overconfident':>15}")
    ok = True
    for t in TAU_FIXED:
        sc, _ = realized_shortfall(y, Q_cal, grid, t)
        so, _ = realized_shortfall(y, Q_over, grid, t)
        print(f"{t:>6}{1-t:>9.3f}{sc:>13.3f}{so:>15.3f}")
        if abs(sc - (1 - t)) > 0.02:            # calibrated must hit 1-tau
            ok = False
        if so <= sc:                            # overconfident must overshoot shortfall
            ok = False
    # cost at high rho: calibrated should be cheaper
    c_cal = cost_at(y, Q_cal, grid, 50.0)
    c_over = cost_at(y, Q_over, grid, 50.0)
    print(f"\n  cost@rho=50: calibrated={c_cal:.1f}  overconfident={c_over:.1f}  "
          f"-> {'calibrated cheaper (correct)' if c_cal < c_over else 'FAIL'}")
    if c_cal >= c_over:
        ok = False
    print(f"\n[selftest] {'PASS: newsvendor math validated' if ok else 'FAIL: harness bug'}")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump", type=str, default=str(OUT / "forecast_quantiles.parquet"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not Path(args.dump).exists():
        print(f"ERROR: dump not found: {args.dump} (run the dumper first)", file=sys.stderr)
        return 1
    return run(args.dump)


if __name__ == "__main__":
    sys.exit(main())
