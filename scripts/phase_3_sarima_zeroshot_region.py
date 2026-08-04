"""National SARIMAX applied ZERO-SHOT to the 10 HHS regions (amortized protocol,
matching CG-Mamba): fit ONE national SARIMAX, then apply its estimated parameters
to each region's series WITHOUT per-region refit (statsmodels .apply, refit=False).

Contrast: phase_3_sarima_wis_region.py refits auto_arima per region (favorable to
SARIMAX). This is the fair amortized-vs-amortized comparison for IV-G --- one
fitted model reused across all regions, exactly like CG-Mamba's zero-shot protocol
(region-specific ILI endog, national pop-weighted env exog).

Output: runs/phase_3_sarima_zeroshot_region.json
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.sarima import build_segment_arrays, fit_sarimax  # noqa: E402
from scripts.phase_3_region_eval import build_region_df  # noqa: E402
from scripts.phase_3_sarima_wis_region import (  # noqa: E402
    rolling_forecast_with_variance, parametric_wis_from_preds,
    REGIONS, TRAIN_FIRST, TRAIN_LAST, VAL_FIRST, VAL_LAST,
    TEST_FIRST, TEST_LAST, TEST_STRICT_FIRST, HORIZONS,
)

NAT_ORDER = (3, 0, 0)               # national auto_arima selection (runs/baselines/sarima.json)
NAT_SEASONAL = (1, 0, 0, 52)
NAT_CSV = _ROOT / "data" / "processed" / "ili_env_weekly_split.csv"


def fit_national():
    df = pd.read_csv(NAT_CSV)
    y_tr, X_tr, ep_tr, _ = build_segment_arrays(df, TRAIN_FIRST, TRAIN_LAST)
    y_va, X_va, ep_va, _ = build_segment_arrays(df, VAL_FIRST, VAL_LAST)
    y_trva = np.concatenate([y_tr, y_va])
    X_trva = np.vstack([X_tr, X_va])
    res = fit_sarimax(y_trva, X_trva, NAT_ORDER, NAT_SEASONAL)
    print(f"National SARIMAX fit: order={NAT_ORDER} seasonal={NAT_SEASONAL} "
          f"(n_trva={len(y_trva)})", flush=True)
    return res


def run_region_zeroshot(region, res_national):
    df = build_region_df(region)
    y_tr, X_tr, ep_tr, _ = build_segment_arrays(df, TRAIN_FIRST, TRAIN_LAST)
    y_va, X_va, ep_va, _ = build_segment_arrays(df, VAL_FIRST, VAL_LAST)
    y_te, X_te, ep_te, _ = build_segment_arrays(df, TEST_FIRST, TEST_LAST)
    y_trva = np.concatenate([y_tr, y_va])
    X_trva = np.vstack([X_tr, X_va])
    # Apply national parameters to this region's history (no re-estimation).
    res_region = res_national.apply(y_trva, exog=X_trva, refit=False)
    test_preds = rolling_forecast_with_variance(res_region, y_te, X_te, ep_te, int(ep_va[-1]))
    return {
        "region": region,
        "test_strict": parametric_wis_from_preds(test_preds, ep_filter_min=TEST_STRICT_FIRST),
        "test_full": parametric_wis_from_preds(test_preds),
    }


def main():
    res_nat = fit_national()
    results = {}
    for region in REGIONS:
        try:
            r = run_region_zeroshot(region, res_nat)
            results[region] = r
            ts = r["test_strict"]
            wis_avg = float(np.mean([ts[f"wis_h{h}"] for h in HORIZONS]))
            cov_avg = float(np.mean([ts[f"cov95_h{h}"] for h in HORIZONS]))
            print(f"  ✓ {region}: test_strict WIS(h1-4)={wis_avg:.3f} Cov95={cov_avg:.3f}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[region] = {"error": str(e)}
    out = _ROOT / "runs" / "phase_3_sarima_zeroshot_region.json"
    out.write_text(json.dumps(results, indent=2, default=str))

    wiss, covs = [], []
    for region in REGIONS:
        ts = results[region].get("test_strict", {})
        if ts and "wis_h1" in ts and not np.isnan(ts["wis_h1"]):
            wiss.append(np.mean([ts[f"wis_h{h}"] for h in HORIZONS]))
            covs.append(np.mean([ts[f"cov95_h{h}"] for h in HORIZONS]))
    wiss, covs = np.array(wiss), np.array(covs)
    print(f"\n== National SARIMAX ZERO-SHOT to regions (n={len(wiss)} regions) ==")
    print(f"  WIS   = {wiss.mean():.3f} +/- {wiss.std(ddof=1):.3f}")
    print(f"  Cov95 = {covs.mean():.3f} +/- {covs.std(ddof=1):.3f}")
    print(f"  compare: per-region-refit SARIMAX 0.301+/-0.060 / 0.916+/-0.031")
    print(f"           CG-Mamba amortized zero-shot 0.393 / 0.954+/-0.026")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
