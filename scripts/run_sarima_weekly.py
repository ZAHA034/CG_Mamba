"""SARIMA(X) weekly baseline driver for CG-Mamba (v2.1.7-A++).

Single-cfg protocol (no outer grid):
  1. auto_arima stepwise AICc on continuous train segment (W40-2002 ~ W39-2018)
     - V=5 exog (matches NN baselines minus target)
     - Internal candidate fits ~25-40 models (Hyndman & Khandakar 2008)
  2. Refit SARIMAX with selected order on (segment A).
     Rolling-origin forecast through val (gap-aware) via Kalman .append().
  3. Refit SARIMAX with selected order on (segment A + val).
     Rolling-origin forecast through test (post-COVID).
  4. Compute MAE/RMSE per horizon for val / test (full) / test_strict.

Run
---
  CPU-only (deterministic, MLE-based, seed-invariant):
    python3 scripts/run_sarima_weekly.py
    python3 scripts/run_sarima_weekly.py --trace          # verbose auto_arima
    python3 scripts/run_sarima_weekly.py --no-exog        # pure SARIMA (ablation)
    python3 scripts/run_sarima_weekly.py --max-p 5 --max-q 5  # wider search

Output
------
  runs/baselines/sarima.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from baselines.sarima import (  # noqa: E402
    SARIMA_M_WEEKLY,
    assert_continuous,
    auto_select_order,
    build_segment_arrays,
    fit_sarimax,
    mae_rmse,
    rolling_forecast_segment,
)


WANDB_ENTITY = "hjs40111-personal"
WANDB_PROJECT = "cg-mamba-jbhi"
WANDB_TAGS = ["sarima", "weekly", "baseline", "v2.1.7-A++"]


def log_summary_to_wandb(payload: dict, run_name: str = "sarima_weekly") -> None:
    """Create a summary-only W&B run from the JSON payload that run_sarima_weekly.py emits."""
    if not _WANDB_AVAILABLE:
        print("[sarima] W&B not available, skipping log.")
        return
    cfg = {
        "method": payload.get("method"),
        "search_bounds": payload.get("search_bounds"),
        "fit_sizes": payload.get("fit_sizes"),
        "use_exog": payload.get("use_exog"),
        "source_csv": payload.get("source_csv"),
        "phase": "M2.6_SARIMA",
        "deterministic": True,
    }
    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT,
        group="sarima_v2.1.7-A++",
        name=run_name,
        tags=WANDB_TAGS,
        config=cfg,
        reinit=True,
    )
    sel = payload.get("selected_order", {})
    order = sel.get("order", [None] * 3)
    sorder = sel.get("seasonal_order", [None] * 4)
    run.summary["selected_p"] = order[0]
    run.summary["selected_d"] = order[1]
    run.summary["selected_q"] = order[2]
    run.summary["selected_P"] = sorder[0]
    run.summary["selected_D"] = sorder[1]
    run.summary["selected_Q"] = sorder[2]
    run.summary["selected_m"] = sorder[3]
    for k, v in payload.get("selection_metrics", {}).items():
        run.summary[k] = v
    for k, v in payload.get("elapsed_sec", {}).items():
        run.summary[f"elapsed_sec_{k}"] = v
    for split, hres in payload.get("results", {}).items():
        for h, m in hres.items():
            if m.get("n", 0) > 0:
                run.summary[f"{split}_h{h}_mae"] = m["mae"]
                run.summary[f"{split}_h{h}_rmse"] = m["rmse"]
                run.summary[f"{split}_h{h}_n"] = m["n"]
                run.summary[f"{split}_h{h}_target_mean_raw"] = m["target_mean_raw"]
                run.summary[f"{split}_h{h}_target_std_raw"] = m["target_std_raw"]
    run.finish()
    print("[sarima] W&B summary logged.")


# Segment boundaries (from split_boundaries.json + data inspection)
TRAIN_POST_GAP_FIRST = 200240   # W40-2002 (first epiweek after pre-gap)
TRAIN_LAST           = 201839   # W39-2018
VAL_FIRST            = 201840   # W40-2018
VAL_LAST             = 202010   # W10-2020
TEST_FIRST           = 202040   # W40-2020
TEST_LAST            = 202535   # W35-2025
TEST_STRICT_FIRST    = 202240   # W40-2022 (v2.1.7-A++ strict mask)


def split_target_filter(records: list[dict], split_label: str) -> list[dict]:
    """Filter prediction records by target split semantics.

    split_label:
      - "val"         : target_ep in [VAL_FIRST, VAL_LAST]
      - "test"        : target_ep in [TEST_FIRST, TEST_LAST]
      - "test_strict" : target_ep in [TEST_STRICT_FIRST, TEST_LAST]
    """
    if split_label == "val":
        lo, hi = VAL_FIRST, VAL_LAST
    elif split_label == "test":
        lo, hi = TEST_FIRST, TEST_LAST
    elif split_label == "test_strict":
        lo, hi = TEST_STRICT_FIRST, TEST_LAST
    else:
        raise ValueError(split_label)
    return [r for r in records if lo <= r["target_ep"] <= hi]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/processed/ili_env_weekly_split.csv")
    ap.add_argument("--out", default="runs/baselines/sarima.json")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--m", type=int, default=SARIMA_M_WEEKLY,
                    help="Seasonal period (default 52 = weekly yearly seasonality).")
    ap.add_argument("--max-p", type=int, default=4)
    ap.add_argument("--max-q", type=int, default=4)
    ap.add_argument("--max-P", type=int, default=2)
    ap.add_argument("--max-Q", type=int, default=2)
    ap.add_argument("--information-criterion", default="aicc",
                    choices=["aic", "aicc", "bic", "hqic"])
    ap.add_argument("--no-exog", action="store_true",
                    help="Pure univariate SARIMA (no environmental regressors).")
    ap.add_argument("--trace", action="store_true",
                    help="Verbose auto_arima stepwise trace.")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Skip W&B summary logging.")
    ap.add_argument("--wandb-from-json", default=None,
                    help="Skip computation: load existing sarima.json and only log "
                         "to W&B. Use after the long auto_arima run finishes.")
    args = ap.parse_args()

    # Retro-log mode: load existing JSON and only log to W&B
    if args.wandb_from_json:
        path = Path(args.wandb_from_json)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        payload = json.load(open(path))
        print(f"[sarima] retro-logging from {path.relative_to(_REPO_ROOT)} to W&B")
        log_summary_to_wandb(payload)
        return 0

    csv_path = (_REPO_ROOT / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    out_path = (_REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[sarima] dataset: {csv_path.relative_to(_REPO_ROOT)}")
    df = pd.read_csv(csv_path)
    print(f"  ({len(df)} rows, epiweek {df['epiweek'].min()}..{df['epiweek'].max()})")

    # --------- 1) Build segments ---------
    print("\n[1/4] Building continuous segments")
    y_tr, X_tr, ep_tr, _ = build_segment_arrays(df, TRAIN_POST_GAP_FIRST, TRAIN_LAST)
    y_va, X_va, ep_va, _ = build_segment_arrays(df, VAL_FIRST, VAL_LAST)
    y_te, X_te, ep_te, _ = build_segment_arrays(df, TEST_FIRST, TEST_LAST)
    assert_continuous(ep_tr, "train_post_gap")
    assert_continuous(ep_va, "val")
    assert_continuous(ep_te, "test")
    print(f"  train_post_gap: n={len(y_tr)}, epiweek {ep_tr[0]}..{ep_tr[-1]}")
    print(f"  val           : n={len(y_va)}, epiweek {ep_va[0]}..{ep_va[-1]}")
    print(f"  test          : n={len(y_te)}, epiweek {ep_te[0]}..{ep_te[-1]}")
    n_test_strict = int(((ep_te >= TEST_STRICT_FIRST) & (ep_te <= TEST_LAST)).sum())
    print(f"  test_strict   : n={n_test_strict} (subset of test, >= {TEST_STRICT_FIRST})")

    # Continuity at train_post_gap -> val boundary (W39-2018 -> W40-2018)
    from baselines.sarima import is_consecutive_epiweek
    if not is_consecutive_epiweek(int(ep_tr[-1]), int(ep_va[0])):
        raise RuntimeError(
            f"Boundary train->val not consecutive: {int(ep_tr[-1])} -> {int(ep_va[0])}"
        )

    X_tr_use = None if args.no_exog else X_tr
    X_va_use = None if args.no_exog else X_va
    X_te_use = None if args.no_exog else X_te
    X_tr_plus_va_use = None if args.no_exog else np.vstack([X_tr, X_va])
    y_tr_plus_va = np.concatenate([y_tr, y_va])

    # --------- 2) Order selection (auto_arima on train) ---------
    print(f"\n[2/4] auto_arima order selection (train only, n={len(y_tr)}, "
          f"exog={'V=5' if not args.no_exog else 'none'}, m={args.m}, "
          f"max_p={args.max_p}, max_q={args.max_q}, "
          f"max_P={args.max_P}, max_Q={args.max_Q}, ic={args.information_criterion})")
    t0 = time.time()
    auto_model = auto_select_order(
        y_tr, X_tr_use,
        m=args.m,
        max_p=args.max_p, max_q=args.max_q,
        max_P=args.max_P, max_Q=args.max_Q,
        information_criterion=args.information_criterion,
        trace=args.trace,
    )
    elapsed_select = time.time() - t0
    order = tuple(int(x) for x in auto_model.order)
    seasonal_order = tuple(int(x) for x in auto_model.seasonal_order)
    aic = float(auto_model.aic())
    aicc = float(auto_model.aicc())
    bic = float(auto_model.bic())
    print(f"  selected: order={order}, seasonal_order={seasonal_order}")
    print(f"  AIC={aic:.2f}, AICc={aicc:.2f}, BIC={bic:.2f}")
    print(f"  selection elapsed: {elapsed_select:.1f}s")

    # --------- 3) Val rolling-origin forecast ---------
    print(f"\n[3/4] Val rolling-origin forecast (refit SARIMAX on train only)")
    t1 = time.time()
    res_train = fit_sarimax(y_tr, X_tr_use, order, seasonal_order)
    val_preds = rolling_forecast_segment(
        res_pre_segment=res_train,
        y_seg=y_va,
        exog_seg=X_va_use,
        eps_seg=ep_va,
        boundary_origin_ep=int(ep_tr[-1]),
        horizons=args.horizons,
    )
    elapsed_val = time.time() - t1
    print(f"  val rolling elapsed: {elapsed_val:.1f}s")

    # --------- 4) Test rolling-origin forecast (refit on train+val) ---------
    print(f"\n[4/4] Test rolling-origin forecast (refit SARIMAX on train+val)")
    t2 = time.time()
    res_train_val = fit_sarimax(y_tr_plus_va, X_tr_plus_va_use, order, seasonal_order)
    # boundary_origin_ep for test = last val epiweek (202010); chain to test[0]
    # (202040) is the COVID gap and will be excluded by gap-aware check inside
    # rolling_forecast_segment.
    test_preds = rolling_forecast_segment(
        res_pre_segment=res_train_val,
        y_seg=y_te,
        exog_seg=X_te_use,
        eps_seg=ep_te,
        boundary_origin_ep=int(ep_va[-1]),
        horizons=args.horizons,
    )
    elapsed_test = time.time() - t2
    print(f"  test rolling elapsed: {elapsed_test:.1f}s")

    # --------- Aggregate metrics ---------
    print("\n[METRICS]")
    print(f"{'Split':<16} {'h':>3} {'n_pairs':>8} {'y_mean':>8} {'y_std':>7} "
          f"{'MAE':>8} {'RMSE':>8} {'MAE/y_std':>10}")
    print("-" * 78)
    results: dict = {}
    for split in ["val", "test", "test_strict"]:
        results[split] = {}
        for h in args.horizons:
            src = val_preds if split == "val" else test_preds
            recs = split_target_filter(src.get(h, []), split)
            m = mae_rmse(recs)
            results[split][str(h)] = m
            if m["n"] > 0:
                print(f"{split:<16} {h:>3} {m['n']:>8} "
                      f"{m['target_mean_raw']:>8.4f} {m['target_std_raw']:>7.4f} "
                      f"{m['mae']:>8.4f} {m['rmse']:>8.4f} "
                      f"{m['mae']/m['target_std_raw']:>10.3f}")
            else:
                print(f"{split:<16} {h:>3} {0:>8}      n/a     n/a      n/a      n/a       n/a")

    out = {
        "method": (
            "SARIMAX + auto.arima (Hyndman-Khandakar 2008) stepwise "
            f"{args.information_criterion.upper()} search; "
            f"{'V=5 exog' if not args.no_exog else 'pure univariate'}; "
            "Kalman .append() rolling-origin (no per-origin refit); "
            "refit boundary at val->test (COVID gap)."
        ),
        "selected_order": {
            "order": list(order),
            "seasonal_order": list(seasonal_order),
        },
        "selection_metrics": {"aic": aic, "aicc": aicc, "bic": bic},
        "search_bounds": {
            "max_p": args.max_p, "max_q": args.max_q,
            "max_P": args.max_P, "max_Q": args.max_Q,
            "m": args.m,
            "information_criterion": args.information_criterion,
        },
        "fit_sizes": {
            "n_fit_train_only": int(len(y_tr)),
            "n_fit_train_plus_val": int(len(y_tr) + len(y_va)),
            "epiweek_train_first": int(ep_tr[0]),
            "epiweek_train_last": int(ep_tr[-1]),
            "epiweek_val_first": int(ep_va[0]),
            "epiweek_val_last": int(ep_va[-1]),
            "epiweek_test_first": int(ep_te[0]),
            "epiweek_test_last": int(ep_te[-1]),
            "test_strict_first": TEST_STRICT_FIRST,
        },
        "elapsed_sec": {
            "auto_arima": round(elapsed_select, 2),
            "val_rolling": round(elapsed_val, 2),
            "test_rolling": round(elapsed_test, 2),
        },
        "source_csv": str(csv_path.relative_to(_REPO_ROOT)),
        "use_exog": not args.no_exog,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {out_path.relative_to(_REPO_ROOT)}")

    if not args.no_wandb:
        log_summary_to_wandb(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
