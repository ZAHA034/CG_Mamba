"""M2.4 Data efficiency — CPU baselines (Persistence + SARIMA) × train periods.

Train period variants (each ends at W39-2018, val/test stay the same):
  17 seasons (full)  : W40-2002 ~ W39-2018  (200240 ~ 201839, 832 obs)
  10 seasons         : W40-2008 ~ W39-2018  (200840 ~ 201839, ~520 obs)
   5 seasons         : W40-2013 ~ W39-2018  (201340 ~ 201839, ~260 obs)
   3 seasons         : W40-2015 ~ W39-2018  (201540 ~ 201839, ~156 obs)

Val:  W40-2018 ~ W10-2020  (unchanged)
Test: W40-2020 ~ W35-2025  (test_full)
Test_strict: W40-2022 ~ W35-2025 (paper main)

Output
------
  runs/m2_4_data_efficiency/sarima/seasons{17,10,5,3}.json
  runs/m2_4_data_efficiency/persistence/all_periods.json   (train-invariant note)
  runs/m2_4_data_efficiency/m2_4_summary.csv

Notes
-----
- Persistence is train-period invariant (no learning), reported once with note.
- SARIMA: re-runs auto_arima order selection per train period.
- Pure CPU, no GPU contention. Safe to run parallel to GPU jobs.
- Uses src.baselines.sarima functions directly (NO modification to existing scripts).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from baselines.sarima import (  # noqa: E402
    SARIMA_M_WEEKLY,
    assert_continuous,
    auto_select_order,
    build_segment_arrays,
    fit_sarimax,
    is_consecutive_epiweek,
    mae_rmse,
    rolling_forecast_segment,
)


# Segment boundaries (matches existing SARIMA full run)
VAL_FIRST            = 201840
VAL_LAST             = 202010
TEST_FIRST           = 202040
TEST_LAST            = 202535
TEST_STRICT_FIRST    = 202240
TRAIN_LAST           = 201839

# M2.4 train period variants
TRAIN_PERIODS = [
    ("17_seasons_full", 200240, 17),  # W40-2002 → W39-2018
    ("13_seasons",      200540, 13),  # W40-2005 → W39-2018  (v2.1.7-A++ Option B)
    ("10_seasons",      200840, 10),  # W40-2008 → W39-2018
    ( "7_seasons",      201140,  7),  # W40-2011 → W39-2018  (v2.1.7-A++ Option B)
    ( "5_seasons",      201340,  5),  # W40-2013 → W39-2018
    ( "4_seasons",      201440,  4),  # W40-2014 → W39-2018  (v2.1.7-A++ Option B)
    ( "3_seasons",      201540,  3),  # W40-2015 → W39-2018
]


def split_target_filter(records, split_label):
    if split_label == "val":
        lo, hi = VAL_FIRST, VAL_LAST
    elif split_label == "test":
        lo, hi = TEST_FIRST, TEST_LAST
    elif split_label == "test_strict":
        lo, hi = TEST_STRICT_FIRST, TEST_LAST
    else:
        raise ValueError(split_label)
    return [r for r in records if lo <= r["target_ep"] <= hi]


def run_one_sarima_train_period(df, train_first_ep, label, n_seasons, out_dir,
                                  horizons=(1, 2, 3, 4), max_p=4, max_q=4,
                                  max_P=2, max_Q=2, m=SARIMA_M_WEEKLY,
                                  ic="aicc", no_exog=False):
    """SARIMA with a specific train-period truncation."""
    print(f"\n{'='*60}")
    print(f"[SARIMA M2.4] {label}  (train_first_ep={train_first_ep}, n_seasons={n_seasons})")
    print('='*60)

    y_tr, X_tr, ep_tr, _ = build_segment_arrays(df, train_first_ep, TRAIN_LAST)
    y_va, X_va, ep_va, _ = build_segment_arrays(df, VAL_FIRST, VAL_LAST)
    y_te, X_te, ep_te, _ = build_segment_arrays(df, TEST_FIRST, TEST_LAST)
    assert_continuous(ep_tr, f"train[{label}]")
    assert_continuous(ep_va, "val")
    assert_continuous(ep_te, "test")
    print(f"  train n={len(y_tr)} ({ep_tr[0]}~{ep_tr[-1]})")
    print(f"  val   n={len(y_va)} ({ep_va[0]}~{ep_va[-1]})")
    print(f"  test  n={len(y_te)} ({ep_te[0]}~{ep_te[-1]})")
    if not is_consecutive_epiweek(int(ep_tr[-1]), int(ep_va[0])):
        raise RuntimeError(
            f"Boundary train->val not consecutive: {int(ep_tr[-1])} -> {int(ep_va[0])}"
        )

    X_tr_use         = None if no_exog else X_tr
    X_va_use         = None if no_exog else X_va
    X_te_use         = None if no_exog else X_te
    X_tr_plus_va_use = None if no_exog else np.vstack([X_tr, X_va])
    y_tr_plus_va     = np.concatenate([y_tr, y_va])

    # Order selection
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto_model = auto_select_order(
                y_tr, X_tr_use,
                m=m, max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                information_criterion=ic, trace=False,
            )
    except Exception as exc:
        print(f"  ❌ auto_arima FAILED: {exc!r}")
        return {"label": label, "n_seasons": n_seasons, "status": "FAILED_ORDER_SELECTION",
                "error": repr(exc), "train_first_epiweek": train_first_ep, "n_train_obs": len(y_tr)}
    elapsed_select = time.time() - t0
    order = tuple(int(x) for x in auto_model.order)
    seasonal_order = tuple(int(x) for x in auto_model.seasonal_order)
    aic = float(auto_model.aic()); aicc = float(auto_model.aicc()); bic = float(auto_model.bic())
    print(f"  selected: order={order}, seasonal={seasonal_order}, "
          f"AICc={aicc:.2f}, BIC={bic:.2f}, t={elapsed_select:.1f}s")

    # Val rolling
    t1 = time.time()
    try:
        res_train = fit_sarimax(y_tr, X_tr_use, order, seasonal_order)
        val_preds = rolling_forecast_segment(
            res_pre_segment=res_train, y_seg=y_va, exog_seg=X_va_use, eps_seg=ep_va,
            boundary_origin_ep=int(ep_tr[-1]), horizons=horizons,
        )
    except Exception as exc:
        print(f"  ❌ val rolling FAILED: {exc!r}")
        return {"label": label, "n_seasons": n_seasons, "status": "FAILED_VAL_ROLLING",
                "error": repr(exc), "selected_order": list(order),
                "seasonal_order": list(seasonal_order)}
    elapsed_val = time.time() - t1

    # Test rolling (refit on train+val)
    t2 = time.time()
    try:
        res_train_val = fit_sarimax(y_tr_plus_va, X_tr_plus_va_use, order, seasonal_order)
        test_preds = rolling_forecast_segment(
            res_pre_segment=res_train_val, y_seg=y_te, exog_seg=X_te_use, eps_seg=ep_te,
            boundary_origin_ep=int(ep_va[-1]), horizons=horizons,
        )
    except Exception as exc:
        print(f"  ❌ test rolling FAILED: {exc!r}")
        return {"label": label, "n_seasons": n_seasons, "status": "FAILED_TEST_ROLLING",
                "error": repr(exc), "selected_order": list(order),
                "seasonal_order": list(seasonal_order)}
    elapsed_test = time.time() - t2

    # Aggregate per split
    results = {}
    for split in ["val", "test", "test_strict"]:
        results[split] = {}
        for h in horizons:
            src = val_preds if split == "val" else test_preds
            recs = split_target_filter(src.get(h, []), split)
            results[split][str(h)] = mae_rmse(recs)

    out = {
        "label": label, "n_seasons": n_seasons,
        "status": "OK",
        "train_first_epiweek": train_first_ep,
        "n_train_obs": int(len(y_tr)),
        "selected_order": list(order),
        "seasonal_order": list(seasonal_order),
        "selection_metrics": {"aic": aic, "aicc": aicc, "bic": bic},
        "elapsed_sec": {"auto_arima": elapsed_select, "val": elapsed_val, "test": elapsed_test},
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seasons_{label}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  saved: {out_path.relative_to(_REPO_ROOT)}")
    # Print per-horizon summary for test_strict
    ts = results["test_strict"]
    print(f"  test_strict MAE: h1={ts['1']['mae']:.4f} h2={ts['2']['mae']:.4f} "
          f"h3={ts['3']['mae']:.4f} h4={ts['4']['mae']:.4f}")
    return out


def run_persistence_invariance_note(df, out_dir):
    """Persistence is train-period invariant — single eval suffices."""
    from baselines.persistence import persistence_pairs
    print(f"\n{'='*60}")
    print("[Persistence M2.4] (train-period invariant — no learning)")
    print('='*60)

    horizons = [1, 2, 3, 4]
    results = {}
    for split in ["val", "test", "test_strict"]:
        results[split] = {}
        for h in horizons:
            preds, targets = persistence_pairs(df, split, h)
            err = preds - targets
            if len(preds):
                results[split][str(h)] = {
                    "mae": float(np.abs(err).mean()),
                    "rmse": float(np.sqrt((err ** 2).mean())),
                    "n": int(len(preds)),
                }
            else:
                results[split][str(h)] = {"mae": float("nan"), "rmse": float("nan"), "n": 0}

    out = {
        "method": "persistence: y_hat_{t+h} = y_t",
        "note": ("Persistence does NOT use training data — it is a per-prediction baseline "
                 "based solely on the last observation. Therefore M2.4 train-period ablation "
                 "yields identical results for all train_period variants."),
        "results_constant_for_all_train_periods": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_periods.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  saved: {out_path.relative_to(_REPO_ROOT)}")
    ts = results["test_strict"]
    print(f"  test_strict MAE (constant): h1={ts['1']['mae']:.4f} h2={ts['2']['mae']:.4f} "
          f"h3={ts['3']['mae']:.4f} h4={ts['4']['mae']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/processed/ili_env_weekly_split.csv")
    ap.add_argument("--out-root", default="runs/m2_4_data_efficiency")
    ap.add_argument("--skip-sarima", action="store_true")
    ap.add_argument("--skip-persistence", action="store_true")
    ap.add_argument("--sarima-max-p", type=int, default=4)
    ap.add_argument("--sarima-max-q", type=int, default=4)
    args = ap.parse_args()

    csv_path = _REPO_ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    out_root = _REPO_ROOT / args.out_root if not Path(args.out_root).is_absolute() else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[M2.4 baselines] dataset: {csv_path.relative_to(_REPO_ROOT)}")
    print(f"[M2.4 baselines] out_root: {out_root.relative_to(_REPO_ROOT)}")
    df = pd.read_csv(csv_path)

    # Persistence (single run, train-invariant)
    if not args.skip_persistence:
        run_persistence_invariance_note(df, out_root / "persistence")

    # SARIMA per train period
    sarima_results = []
    if not args.skip_sarima:
        sarima_dir = out_root / "sarima"
        for label, train_first_ep, n_seasons in TRAIN_PERIODS:
            r = run_one_sarima_train_period(
                df, train_first_ep, label, n_seasons, sarima_dir,
                max_p=args.sarima_max_p, max_q=args.sarima_max_q,
            )
            sarima_results.append(r)

    # Aggregate CSV summary
    rows = []
    if not args.skip_persistence:
        pp = json.load(open(out_root / "persistence/all_periods.json"))
        r = pp["results_constant_for_all_train_periods"]
        for label, _, n in TRAIN_PERIODS:
            rows.append({
                "model": "persistence", "train_period": label, "n_seasons": n,
                "status": "OK (invariant)",
                **{f"val_h{h}_mae": r["val"][str(h)]["mae"] for h in [1,2,3,4]},
                **{f"test_h{h}_mae": r["test"][str(h)]["mae"] for h in [1,2,3,4]},
                **{f"tS_h{h}_mae": r["test_strict"][str(h)]["mae"] for h in [1,2,3,4]},
            })
    if not args.skip_sarima:
        for sr in sarima_results:
            row = {"model": "sarima", "train_period": sr["label"], "n_seasons": sr["n_seasons"],
                   "status": sr["status"]}
            if sr["status"] == "OK":
                for h in [1, 2, 3, 4]:
                    row[f"val_h{h}_mae"]    = sr["results"]["val"][str(h)]["mae"]
                    row[f"test_h{h}_mae"]   = sr["results"]["test"][str(h)]["mae"]
                    row[f"tS_h{h}_mae"]     = sr["results"]["test_strict"][str(h)]["mae"]
            rows.append(row)
    df_summary = pd.DataFrame(rows)
    summary_path = out_root / "m2_4_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path.relative_to(_REPO_ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
