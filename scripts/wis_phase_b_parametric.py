"""WIS Phase B group 1 — SARIMA parametric Gaussian (PLAN J.3, J.7 Q3).

Reuses the auto_arima-selected order from runs/baselines/sarima.json
(no new auto_arima run — that took 32 min). Re-fits SARIMAX with the
same order on train, then rolls through val + test extracting Kalman
forecast mean AND variance per horizon. Parametric Gaussian quantile:
    q(level) = mean + sqrt(variance) * Φ^{-1}(level)

Output: runs/wis_phase_b/sarima/wis_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from baselines.sarima import (                                              # noqa: E402
    build_segment_arrays, fit_sarimax, is_consecutive_epiweek,
)
from src.data.loader import load_dataset_csv                                 # noqa: E402
from src.eval.wis import wis, wis_decomposed, coverage, REQUIRED_QUANTILES   # noqa: E402
from src.eval.quantile_predictions import parametric_gaussian_quantiles      # noqa: E402

CSV_PATH = _ROOT / "data/processed/ili_env_weekly_split.csv"
SARIMA_JSON = _ROOT / "runs/baselines/sarima.json"
OUT_DIR = _ROOT / "runs/wis_phase_b/sarima"

# Split boundaries (match scripts/run_sarima_weekly.py:111-116)
TRAIN_POST_GAP_FIRST = 200240
TRAIN_LAST           = 201839
VAL_FIRST            = 201840
VAL_LAST             = 202010
TEST_FIRST           = 202040
TEST_LAST            = 202535
TEST_STRICT_FIRST    = 202240


def _rolling_segment_with_variance(
    res_pre_segment, y_seg: np.ndarray, exog_seg: np.ndarray,
    eps_seg: np.ndarray, boundary_origin_ep: int | None,
    horizons=(1, 2, 3, 4),
):
    """Mirror of baselines.sarima.rolling_forecast_segment but extracts
    forecast mean AND variance via get_forecast() instead of forecast().
    Returns dict[h] = list of {target_ep, target_idx, y_true, mean, variance}.
    """
    H = max(horizons)
    N = len(y_seg)
    preds = {h: [] for h in horizons}

    current_res = res_pre_segment
    for t in range(N):
        steps = min(H, N - t)
        if steps == 0:
            break
        future_exog = exog_seg[t:t + steps] if exog_seg is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_obj = current_res.get_forecast(steps=steps, exog=future_exog)
            mean = np.asarray(fc_obj.predicted_mean, dtype=np.float64)
            var = np.asarray(fc_obj.var_pred_mean, dtype=np.float64)

        for h in horizons:
            target_idx = t + h - 1
            if target_idx >= N:
                continue
            if t == 0 and boundary_origin_ep is not None:
                if not is_consecutive_epiweek(int(boundary_origin_ep), int(eps_seg[0])):
                    continue
            preds[h].append({
                "target_ep": int(eps_seg[target_idx]),
                "target_idx": int(target_idx),
                "y_true": float(y_seg[target_idx]),
                "mean": float(mean[h - 1]),
                "variance": float(var[h - 1]),
            })

        if t + 1 < N:
            new_exog = exog_seg[t:t + 1] if exog_seg is not None else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_res = current_res.append(
                    [y_seg[t]], exog=new_exog, refit=False,
                )
    return preds


def _records_to_arrays(records_per_h: dict, split: str, horizons=(1, 2, 3, 4)):
    """Filter records by split label and align across horizons by target_ep.

    A target_ep belongs to `split` if VAL_FIRST..VAL_LAST (val) or
    TEST_FIRST..TEST_LAST (test_full) or TEST_STRICT_FIRST..TEST_LAST (test_strict).
    Returns (mean [N, H], var [N, H], y [N, H]) with N = min count across h.
    """
    lo, hi = {"val": (VAL_FIRST, VAL_LAST),
              "test_full": (TEST_FIRST, TEST_LAST),
              "test_strict": (TEST_STRICT_FIRST, TEST_LAST)}[split]
    # Filter per horizon
    by_ep = {h: {r["target_ep"]: r for r in records_per_h[h]
                  if lo <= r["target_ep"] <= hi} for h in horizons}
    # Intersect target_eps across horizons (only obs with all h available)
    common_eps = sorted(set.intersection(*[set(by_ep[h].keys()) for h in horizons]))
    if not common_eps:
        return None
    means = np.array([[by_ep[h][ep]["mean"] for h in horizons] for ep in common_eps])
    vars_ = np.array([[by_ep[h][ep]["variance"] for h in horizons] for ep in common_eps])
    ys = np.array([[by_ep[h][ep]["y_true"] for h in horizons] for ep in common_eps])
    return means, vars_, ys


def _score_split(quantile_forecasts: dict, y_true: np.ndarray) -> dict:
    N, H = y_true.shape
    wis_per_h, disp_per_h, under_per_h, over_per_h = [], [], [], []
    for h in range(H):
        qf_h = {q: quantile_forecasts[q][:, h] for q in quantile_forecasts}
        y_h = y_true[:, h]
        w = wis(y_h, qf_h)
        wis_per_h.append(float(w.mean()))
        parts = wis_decomposed(y_h, qf_h)
        disp_per_h.append(float(parts["dispersion"].mean()))
        under_per_h.append(float(parts["under"].mean()))
        over_per_h.append(float(parts["over"].mean()))

    qf_flat = {q: quantile_forecasts[q].reshape(-1) for q in quantile_forecasts}
    y_flat = y_true.reshape(-1)
    cov50 = coverage(y_flat, qf_flat, alpha=0.5)
    cov95 = coverage(y_flat, qf_flat, alpha=0.05)

    return {
        "n": int(N),
        "wis_per_horizon": wis_per_h,
        "wis_avg": float(np.mean(wis_per_h)),
        "wis_decomposed": {
            "dispersion_per_horizon": disp_per_h,
            "under_per_horizon": under_per_h,
            "over_per_horizon": over_per_h,
            "dispersion_avg": float(np.mean(disp_per_h)),
            "under_avg": float(np.mean(under_per_h)),
            "over_avg": float(np.mean(over_per_h)),
        },
        "coverage_50": cov50,
        "coverage_95": cov95,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load selected order from previous run
    if not SARIMA_JSON.exists():
        raise SystemExit(f"Missing {SARIMA_JSON} — run scripts/run_sarima_weekly.py first")
    saved = json.loads(SARIMA_JSON.read_text())
    order = tuple(saved["selected_order"]["order"])
    seasonal_order = tuple(saved["selected_order"]["seasonal_order"])
    use_exog = saved.get("use_exog", True)
    print(f"[SARIMA-WIS] reusing order={order} seasonal={seasonal_order} exog={use_exog}")

    df = load_dataset_csv(CSV_PATH)
    y_tr, X_tr, ep_tr, _ = build_segment_arrays(df, TRAIN_POST_GAP_FIRST, TRAIN_LAST)
    y_va, X_va, ep_va, _ = build_segment_arrays(df, VAL_FIRST, VAL_LAST)
    y_te, X_te, ep_te, _ = build_segment_arrays(df, TEST_FIRST, TEST_LAST)

    X_tr_use = X_tr if use_exog else None
    X_va_use = X_va if use_exog else None
    X_te_use = X_te if use_exog else None

    # Fit on train, roll through val
    print("[1/3] Fit SARIMAX on train  ...", end=" ", flush=True)
    t0 = time.time()
    res_train = fit_sarimax(y_tr, X_tr_use, order, seasonal_order)
    print(f"done ({time.time() - t0:.1f}s)")

    print("[2/3] Rolling forecast through val (with variance) ...", end=" ", flush=True)
    t0 = time.time()
    val_preds = _rolling_segment_with_variance(
        res_train, y_va, X_va_use, ep_va,
        boundary_origin_ep=int(ep_tr[-1]), horizons=tuple(args.horizons),
    )
    print(f"done ({time.time() - t0:.1f}s)")

    # Refit on train+val, roll through test
    print("[3/3] Refit on train+val + roll through test (with variance) ...", end=" ", flush=True)
    t0 = time.time()
    y_trva = np.concatenate([y_tr, y_va])
    X_trva = np.concatenate([X_tr, X_va]) if use_exog else None
    res_trva = fit_sarimax(y_trva, X_trva, order, seasonal_order)
    test_preds = _rolling_segment_with_variance(
        res_trva, y_te, X_te_use, ep_te,
        boundary_origin_ep=int(ep_va[-1]), horizons=tuple(args.horizons),
    )
    print(f"done ({time.time() - t0:.1f}s)")

    # Score each split via parametric Gaussian quantile
    out = {"baseline": "sarima",
           "cfg_name": f"SARIMAX{order}x{seasonal_order}",
           "use_exog": use_exog,
           "splits": {}}

    for split_label, records_src in [("val", val_preds),
                                      ("test_full", test_preds),
                                      ("test_strict", test_preds)]:
        arrs = _records_to_arrays(records_src, split_label, tuple(args.horizons))
        if arrs is None:
            print(f"  [sarima {split_label:11s}] no records, skipping")
            continue
        means, vars_, ys = arrs
        qf = parametric_gaussian_quantiles(means, vars_)
        out["splits"][split_label] = _score_split(qf, ys)
        s = out["splits"][split_label]
        print(f"  [sarima] {split_label:11s} n={s['n']}  "
              f"WIS_avg={s['wis_avg']:.4f}  "
              f"cov50={s['coverage_50']:.3f}  cov95={s['coverage_95']:.3f}")

    out_path = OUT_DIR / "wis_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
