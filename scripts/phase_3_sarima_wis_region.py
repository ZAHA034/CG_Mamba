"""§18 Phase 3 — SARIMA region-specific parametric WIS.

Reruns auto_arima + rolling forecast WITH forecast variance (Kalman SE) → WIS.
Extends phase_3_sarima_region.py with variance extraction.

Output: runs/phase_3_sarima_wis_region.json
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
import numpy as np
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.sarima import (
    build_segment_arrays, auto_select_order, fit_sarimax,
    SARIMA_M_WEEKLY, is_consecutive_epiweek,
)
from src.eval.wis import wis, coverage, REQUIRED_QUANTILES
from scripts.phase_3_region_eval import build_region_df

REGIONS = [f"hhs{i}" for i in range(1, 11)]
TRAIN_FIRST, TRAIN_LAST = 200140, 201839
VAL_FIRST, VAL_LAST = 201840, 202010
TEST_FIRST, TEST_LAST = 202040, 202535
TEST_STRICT_FIRST = 202240
HORIZONS = [1, 2, 3, 4]


def rolling_forecast_with_variance(res, y_seg, exog_seg, eps_seg, boundary_ep, horizons=(1,2,3,4)):
    """Rolling forecast + Kalman SE per step. Returns {h: [{..., y_se}]}."""
    H = max(horizons)
    N = len(y_seg)
    preds = {h: [] for h in horizons}
    current_res = res
    for t in range(N):
        steps = min(H, N - t)
        if steps == 0: break
        future_exog = exog_seg[t:t+steps] if exog_seg is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_obj = current_res.get_forecast(steps=steps, exog=future_exog)
            fc_mean = np.asarray(fc_obj.predicted_mean, dtype=np.float64)
            fc_se = np.asarray(fc_obj.se_mean, dtype=np.float64)
        origin_ep = int(boundary_ep) if t == 0 else int(eps_seg[t-1])
        for h in horizons:
            ti = t + h - 1
            if ti >= N: continue
            if t == 0 and boundary_ep is not None:
                if not is_consecutive_epiweek(int(boundary_ep), int(eps_seg[0])): continue
            preds[h].append({
                "origin_ep": origin_ep, "target_ep": int(eps_seg[ti]),
                "y_true": float(y_seg[ti]), "y_pred": float(fc_mean[h-1]),
                "y_se": float(fc_se[h-1]),
            })
        if t+1 < N:
            new_exog = exog_seg[t:t+1] if exog_seg is not None else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_res = current_res.append([y_seg[t]], exog=new_exog, refit=False)
    return preds


def parametric_wis_from_preds(preds_dict, ep_filter_min=None):
    """Compute WIS from Gaussian(mu, se) predictions."""
    out = {}
    for h in HORIZONS:
        records = preds_dict.get(h, [])
        if ep_filter_min is not None:
            records = [r for r in records if r['target_ep'] >= ep_filter_min]
        if not records:
            out[f"wis_h{h}"] = float('nan')
            out[f"cov95_h{h}"] = float('nan')
            continue
        y_true = np.array([r['y_true'] for r in records])
        y_pred = np.array([r['y_pred'] for r in records])
        y_se = np.array([r['y_se'] for r in records])
        y_se = np.maximum(y_se, 1e-8)
        # Gaussian quantile forecasts
        qf = {}
        for q in REQUIRED_QUANTILES:
            z = scipy_norm.ppf(q)
            qf[q] = y_pred + z * y_se
        wis_val = float(wis(y_true, qf).mean())
        cov_val = float(coverage(y_true, qf, alpha=0.05))
        out[f"wis_h{h}"] = wis_val
        out[f"cov95_h{h}"] = cov_val
        out[f"mae_h{h}"] = float(np.abs(y_pred - y_true).mean())
    return out


def run_region(region):
    print(f"\n=== {region} ===", flush=True)
    df = build_region_df(region)
    y_tr, X_tr, ep_tr, _ = build_segment_arrays(df, TRAIN_FIRST, TRAIN_LAST)
    y_va, X_va, ep_va, _ = build_segment_arrays(df, VAL_FIRST, VAL_LAST)
    y_te, X_te, ep_te, _ = build_segment_arrays(df, TEST_FIRST, TEST_LAST)
    print(f"  segments: train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}", flush=True)

    print(f"  auto_arima ...", flush=True)
    t0 = time.time()
    auto_model = auto_select_order(y_tr, X_tr, m=SARIMA_M_WEEKLY,
                                     max_p=4, max_q=4, max_P=2, max_Q=2,
                                     information_criterion="aicc", trace=False)
    order = tuple(int(x) for x in auto_model.order)
    seasonal_order = tuple(int(x) for x in auto_model.seasonal_order)
    print(f"  order={order} seasonal={seasonal_order} ({time.time()-t0:.0f}s)", flush=True)

    # Fit train, forecast val
    res_train = fit_sarimax(y_tr, X_tr, order, seasonal_order)
    val_preds = rolling_forecast_with_variance(res_train, y_va, X_va, ep_va, int(ep_tr[-1]))

    # Refit train+val, forecast test
    y_trva = np.concatenate([y_tr, y_va])
    X_trva = np.vstack([X_tr, X_va])
    res_trva = fit_sarimax(y_trva, X_trva, order, seasonal_order)
    test_preds = rolling_forecast_with_variance(res_trva, y_te, X_te, ep_te, int(ep_va[-1]))

    # WIS: test_full + test_strict
    test_full_wis = parametric_wis_from_preds(test_preds)
    test_strict_wis = parametric_wis_from_preds(test_preds, ep_filter_min=TEST_STRICT_FIRST)

    return {
        "region": region, "order": list(order), "seasonal_order": list(seasonal_order),
        "test_full": test_full_wis, "test_strict": test_strict_wis,
    }


def main():
    results = {}
    for region in REGIONS:
        try:
            r = run_region(region)
            results[region] = r
            ts = r['test_strict']
            print(f"  ✓ {region}  tS_wis_h1={ts.get('wis_h1','?'):.4f}  cov95={ts.get('cov95_h1','?'):.3f}  mae_h1={ts.get('mae_h1','?'):.4f}", flush=True)
        except Exception as e:
            import traceback
            print(f"  ✗ {region}: {e}", flush=True)
            traceback.print_exc()
            results[region] = {"error": str(e)}
    out = _ROOT / "runs" / "phase_3_sarima_wis_region.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
