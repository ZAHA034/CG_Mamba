"""SARIMA(X) weekly baseline for CG-Mamba (v2.1.7-A++ baseline expansion).

References
----------
- Hyndman, R.J. & Khandakar, Y. (2008). "Automatic Time Series Forecasting:
  The forecast Package for R." *Journal of Statistical Software*, 27(3). 1-22.
  https://doi.org/10.18637/jss.v027.i03
  [auto.arima stepwise AICc algorithm]
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). OTexts. Ch. 9 (ARIMA), Ch. 10 (Dynamic regression).
- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series
  Analysis: Forecasting and Control* (5th ed.). Wiley.
- Implementation: pmdarima 2.1.1 (Python port of forecast::auto.arima) +
  statsmodels.tsa.statespace.SARIMAX.

Protocol (paper-mandatory for ILI baseline, v2.1.7-A++ §7.1)
-----------------------------------------------------------
- Order selection: stepwise AICc search on continuous train segment
  (W40-2002 to W39-2018 = 832 obs, post pre-2002 gap).
- Search bounds: max_p=4, max_q=4, max_P=2, max_Q=2, m=52 (yearly weekly
  seasonality), d/D auto-determined via KPSS / Canova-Hansen tests.
- Tuning-effort equivalence with NN baselines: auto_arima internally fits
  ~25-40 candidate models under stepwise search (Hyndman-Khandakar 2008 §3),
  matching Pattern A grid sizes (12-17 cfg × 1 seed) of LSTM/PatchTST/DLinear.
- SARIMAX with V=5 environmental exogenous variables (matches NN V=6 input
  ex-target; log1p for count features mirrors LSTM_FEATURE_COLS preprocessing).
- Multi-step forecasts via Kalman-filtered state propagation (statsmodels
  SARIMAXResults.forecast); state advanced one step at a time across val/test
  via .append(refit=False) — analogous to "rolling-origin evaluation" but
  without per-origin MLE re-estimation (Hyndman & Athanasopoulos 2021 §5.10).
- Refit boundary at val→test (after 29-week COVID gap W11-2020 ~ W39-2020)
  to avoid Kalman state degradation across long unobserved interval.

Gap-aware semantics
-------------------
- Train gap (W21-2002 ~ W39-2002, 19 weeks) is excluded by starting from
  W40-2002 — only 32 weeks of pre-gap data are dropped (< 1 season).
- COVID gap is handled by re-MLE refit at the val→test boundary.
- Per-prediction consecutive-epiweek chain check follows persistence.py
  / loader.py conventions (loader.is_consecutive_epiweek).

Output (saved by driver, mirrors persistence.json)
--------------------------------------------------
{
  "method": "SARIMAX + auto.arima (Hyndman-Khandakar 2008) stepwise AICc",
  "selected_order": {"order": [p, d, q], "seasonal_order": [P, D, Q, 52]},
  "aicc": float, "aic": float, "bic": float,
  "n_fit_train_only": 832, "n_fit_train_plus_val": 907,
  "results": {
    "val": {h: {mae, rmse, n, ...}, ...},
    "test": {h: {...}, ...},
    "test_strict": {h: {...}, ...}
  }
}
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd


SARIMA_TARGET_COL = "ili_weighted_pct"
# V=5 exogenous (matches V=6 NN baselines minus target, preserving log1p on counts)
SARIMA_EXOG_COLS = [
    "total_ili_count",      # log1p
    "num_providers",        # log1p
    "num_patients",         # log1p
    "temperature_c",        # raw
    "specific_humidity_g_per_kg",  # raw
]
SARIMA_M_WEEKLY = 52


def is_consecutive_epiweek(prev_ep: int, curr_ep: int) -> bool:
    py, pw = prev_ep // 100, prev_ep % 100
    cy, cw = curr_ep // 100, curr_ep % 100
    if py == cy:
        return cw == pw + 1
    if cy == py + 1 and cw == 1 and pw in (52, 53):
        return True
    return False


def build_segment_arrays(
    df: pd.DataFrame,
    epiweek_first: int,
    epiweek_last: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice df to a continuous segment [epiweek_first, epiweek_last] and
    return (y_target, exog [N, V=5], epiweeks, split_labels).

    Caller verifies continuity of the slice via the returned epiweeks.
    """
    sub = (
        df[(df["epiweek"] >= epiweek_first) & (df["epiweek"] <= epiweek_last)]
        .sort_values("epiweek")
        .reset_index(drop=True)
    )
    y = sub[SARIMA_TARGET_COL].to_numpy(dtype=np.float64)
    exog = np.column_stack([
        np.log1p(sub["total_ili_count"].to_numpy(dtype=np.float64)),
        np.log1p(sub["num_providers"].to_numpy(dtype=np.float64)),
        np.log1p(sub["num_patients"].to_numpy(dtype=np.float64)),
        sub["temperature_c"].to_numpy(dtype=np.float64),
        sub["specific_humidity_g_per_kg"].to_numpy(dtype=np.float64),
    ]).astype(np.float64)
    eps = sub["epiweek"].to_numpy(dtype=np.int64)
    splits = sub["split"].to_numpy()
    return y, exog, eps, splits


def assert_continuous(eps: np.ndarray, label: str) -> None:
    for i in range(1, len(eps)):
        if not is_consecutive_epiweek(int(eps[i - 1]), int(eps[i])):
            raise ValueError(
                f"[{label}] non-consecutive epiweek at idx {i}: "
                f"{int(eps[i-1])} -> {int(eps[i])}"
            )


def auto_select_order(
    y_train: np.ndarray,
    exog_train: np.ndarray | None,
    m: int = SARIMA_M_WEEKLY,
    max_p: int = 4, max_q: int = 4,
    max_P: int = 2, max_Q: int = 2,
    information_criterion: str = "aicc",
    trace: bool = False,
):
    """Run pmdarima.auto_arima stepwise search and return the fitted ARIMA
    object (whose .order, .seasonal_order, .aicc(), etc. are read by the
    driver).
    """
    import pmdarima as pm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(
            y_train,
            X=exog_train,
            seasonal=True,
            m=m,
            max_p=max_p, max_q=max_q,
            max_P=max_P, max_Q=max_Q,
            d=None, D=None,            # let KPSS / Canova-Hansen tests decide
            stepwise=True,
            suppress_warnings=True,
            information_criterion=information_criterion,
            error_action="ignore",
            trace=trace,
        )
    return model


def fit_sarimax(
    y: np.ndarray,
    exog: np.ndarray | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
):
    """Fit statsmodels SARIMAX with given (selected) order. Returns SARIMAXResults."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=200)
    return res


def rolling_forecast_segment(
    res_pre_segment,
    y_seg: np.ndarray,
    exog_seg: np.ndarray | None,
    eps_seg: np.ndarray,
    boundary_origin_ep: int | None,
    horizons: Iterable[int] = (1, 2, 3, 4),
) -> dict[int, list[dict]]:
    """Rolling-origin forecast through a continuous segment of length N.

    At iteration t in [0, N):
      - State has been advanced through (pre-segment data + y_seg[:t]).
      - Forecast h=1..H steps ahead -> targets y_seg[t..t+H-1].
      - For h=1 at t=0, origin is `boundary_origin_ep` (last observed epiweek
        of the prior segment); chain consecutiveness w.r.t. eps_seg[0] is
        checked. Within-segment chains are continuous by construction.
      - Then `.append([y_seg[t]], refit=False)` to update state for t+1.

    Returns {h: [{target_idx, target_ep, y_true, y_pred, origin_ep}, ...]}.
    Predictions whose target_idx >= N are skipped (end-of-segment trim).
    """
    horizons = tuple(horizons)
    H = max(horizons)
    N = len(y_seg)
    preds: dict[int, list[dict]] = {h: [] for h in horizons}

    current_res = res_pre_segment
    for t in range(N):
        steps = min(H, N - t)
        if steps == 0:
            break
        future_exog = exog_seg[t:t + steps] if exog_seg is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = np.asarray(
                current_res.forecast(steps=steps, exog=future_exog),
                dtype=np.float64,
            )

        # Origin epiweek (the last observed point before forecasting)
        if t == 0:
            origin_ep = int(boundary_origin_ep) if boundary_origin_ep is not None else -1
        else:
            origin_ep = int(eps_seg[t - 1])

        for h in horizons:
            target_idx = t + h - 1
            if target_idx >= N:
                continue
            # Gap-aware chain check: at t=0 verify origin_ep -> eps_seg[0] consecutive
            if t == 0 and boundary_origin_ep is not None:
                if not is_consecutive_epiweek(int(boundary_origin_ep), int(eps_seg[0])):
                    continue
            # Within-segment chain is continuous by construction (caller asserted)
            preds[h].append({
                "origin_ep": origin_ep,
                "target_idx": int(target_idx),
                "target_ep": int(eps_seg[target_idx]),
                "y_true": float(y_seg[target_idx]),
                "y_pred": float(fc[h - 1]),
            })

        # Advance state by one observation (Kalman update, no MLE refit)
        if t + 1 < N:
            new_exog = exog_seg[t:t + 1] if exog_seg is not None else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                current_res = current_res.append(
                    [y_seg[t]], exog=new_exog, refit=False,
                )

    return preds


def mae_rmse(records: list[dict]) -> dict:
    if not records:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    err = np.asarray([r["y_pred"] - r["y_true"] for r in records], dtype=np.float64)
    yt = np.asarray([r["y_true"] for r in records], dtype=np.float64)
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "n": int(len(records)),
        "target_mean_raw": float(yt.mean()),
        "target_std_raw": float(yt.std(ddof=0)),
    }
