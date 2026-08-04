"""§18 Phase 3 — SARIMA region-specific fit + inference.

Note (protocol heterogeneity, explicit disclosure):
  NN baselines use national-trained ckpts with regional inputs (Option A).
  SARIMA is a classical statistical model with no reusable weights, so we
  perform region-specific auto_arima + fit + rolling forecast per region
  (Option B for SARIMA only). This is the standard SARIMA protocol.

Output: runs/phase_3_sarima_region.json
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.sarima import (
    build_segment_arrays, auto_select_order, fit_sarimax,
    rolling_forecast_segment, SARIMA_M_WEEKLY, is_consecutive_epiweek,
)
from scripts.phase_3_region_eval import build_region_df

REGIONS = [f"hhs{i}" for i in range(1, 11)]
TRAIN_FIRST = 200140  # W40-2002 (post pre-2002 gap)
TRAIN_LAST = 201839
VAL_FIRST, VAL_LAST = 201840, 202010
TEST_FIRST, TEST_LAST = 202040, 202535
TEST_STRICT_FIRST = 202240
HORIZONS = [1, 2, 3, 4]


def mae_per_h(preds_dict, ep_filter_min=None):
    """preds_dict: {h: [{'target_ep', 'y_true', 'y_pred', ...}]}"""
    out = {}
    for h in HORIZONS:
        records = preds_dict.get(h, [])
        if ep_filter_min is not None:
            records = [r for r in records if r['target_ep'] >= ep_filter_min]
        if records:
            out[h] = float(np.mean([abs(r['y_pred'] - r['y_true']) for r in records]))
        else:
            out[h] = float('nan')
    return out


def run_region(region: str):
    print(f"\n=== {region} ===", flush=True)
    df = build_region_df(region)
    # Build segments
    y_tr, X_tr, ep_tr, _ = build_segment_arrays(df, TRAIN_FIRST, TRAIN_LAST)
    y_va, X_va, ep_va, _ = build_segment_arrays(df, VAL_FIRST, VAL_LAST)
    y_te, X_te, ep_te, _ = build_segment_arrays(df, TEST_FIRST, TEST_LAST)
    print(f"  segments: train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}", flush=True)

    # Order selection on train
    print(f"  auto_arima order selection ...", flush=True)
    t0 = time.time()
    auto_model = auto_select_order(y_tr, X_tr, m=SARIMA_M_WEEKLY,
                                     max_p=4, max_q=4, max_P=2, max_Q=2,
                                     information_criterion="aicc", trace=False)
    order = tuple(int(x) for x in auto_model.order)
    seasonal_order = tuple(int(x) for x in auto_model.seasonal_order)
    print(f"  selected: order={order} seasonal={seasonal_order} (elapsed {time.time()-t0:.1f}s)", flush=True)

    # Fit on train, rolling val
    res_train = fit_sarimax(y_tr, X_tr, order, seasonal_order)
    val_preds = rolling_forecast_segment(
        res_pre_segment=res_train, y_seg=y_va, exog_seg=X_va, eps_seg=ep_va,
        boundary_origin_ep=int(ep_tr[-1]), horizons=HORIZONS,
    )

    # Refit on train+val, rolling test (boundary crosses COVID gap — refit on train+val only,
    # do NOT include covid_excluded rows in segment)
    y_trva = np.concatenate([y_tr, y_va])
    X_trva = np.vstack([X_tr, X_va])
    res_trva = fit_sarimax(y_trva, X_trva, order, seasonal_order)
    test_preds = rolling_forecast_segment(
        res_pre_segment=res_trva, y_seg=y_te, exog_seg=X_te, eps_seg=ep_te,
        boundary_origin_ep=int(ep_va[-1]), horizons=HORIZONS,
    )

    val_mae = mae_per_h(val_preds)
    test_full_mae = mae_per_h(test_preds)
    test_strict_mae = mae_per_h(test_preds, ep_filter_min=TEST_STRICT_FIRST)

    # Count test_strict records (h=1)
    n_strict = sum(1 for r in test_preds.get(1, []) if r['target_ep'] >= TEST_STRICT_FIRST)

    return {
        "region": region,
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "n_train": len(y_tr),
        "n_val": len(y_va),
        "n_test_full": len(y_te),
        "n_test_strict": n_strict,
        "val_mae": val_mae,
        "test_full_mae": test_full_mae,
        "test_strict_mae": test_strict_mae,
    }


def main():
    out = {}
    for region in REGIONS:
        try:
            r = run_region(region)
            out[region] = r
            print(f"  ✓ {region}  tS_h1={r['test_strict_mae'].get(1, 'nan'):.4f}  "
                  f"tF_h1={r['test_full_mae'].get(1, 'nan'):.4f}", flush=True)
        except Exception as e:
            import traceback
            print(f"  ✗ {region}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            out[region] = {"error": str(e)}
    out_path = _ROOT / "runs" / "phase_3_sarima_region.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
