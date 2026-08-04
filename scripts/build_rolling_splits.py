"""Rolling-origin (expanding-window) split generator for the T1 robustness experiment.

PURPOSE (pre-registered, result-blind):
  Test whether CG-Mamba's regional native-calibration dominance (closer to nominal
  |Cov95-0.95| than every DL baseline, in every HHS region) replicates across
  MULTIPLE train/test season cutoffs -- not just the single 2022-25 window that
  the headline uses. This directly answers the "single fixed split" reject-trigger.

DESIGN (mirrors src/data/build_splits.py exactly; only the boundaries roll):
  For each pre-registered test season Y (active season = W40-Y .. W20-(Y+1)):
    - test  = [Y40, (Y+1)20]           (active flu season, the decision-relevant window)
    - val   = [(Y-1)40, Y20]           (prior active season)
    - train = [200140, (Y-1)39]        (everything before val, contiguous for lookback)
    - COVID hole [202011, 202039] excluded from EVERY split (same as headline).
  Normalization (StandardScaler on ili_weighted_pct, temperature_c,
  specific_humidity_g_per_kg) is RE-FIT on THAT cutoff's train ONLY -> no future
  leakage of normalization stats (this is why we cannot reuse the global
  normalization_params.json).

PRE-REGISTERED CUTOFF SET (fixed BEFORE any training; do not add/drop by result):
  Test seasons: 2015-16, 2016-17, 2017-18, 2018-19  (pre-COVID; clean pre-COVID train)
              + 2022-23, 2023-24, 2024-25            (post-COVID; = headline era)
  Excluded: 2019-20, 2020-21, 2021-22 (COVID-disrupted; matches headline exclusion).
  => 7 expanding-window cutoffs spanning a decade. Train grows with Y (expanding).

OUTPUT (one dir per cutoff, nothing overwritten in data/processed/):
  runs/rolling_origin/cut{Y}/
    ili_env_weekly_split.csv     # 'split' column re-assigned for this cutoff
    normalization_params.json    # re-fit on THIS cutoff's train only
    split_boundaries.json        # boundary spec + row counts
  runs/rolling_origin/cutoffs_manifest.json

USAGE (GPU 0 -- pure data prep):
    python scripts/build_rolling_splits.py
    python scripts/build_rolling_splits.py --cutoffs 2015 2018 2022   # subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = _ROOT / "data" / "processed" / "ili_env_weekly.csv"
OUT_ROOT = _ROOT / "runs" / "rolling_origin"

# ---- pre-registered cutoff set (test-season start year Y) ----
CUTOFFS = [2015, 2016, 2017, 2018, 2022, 2023, 2024]

# ---- mirror build_splits.py exactly ----
SCALER_COLUMNS = ["ili_weighted_pct", "temperature_c", "specific_humidity_g_per_kg"]
COVID_HOLE = (202011, 202039)   # excluded from every split (identical to headline)


def season_windows(Y: int) -> dict:
    """Boundaries for test-season Y.

    HEADLINE-MATCHED (pre-reg decision 2, option i): test spans the FULL season
    W40-Y .. W39-(Y+1) INCLUDING summer weeks -- identical evaluation method to the
    headline test_strict (all weeks, not active-season-only), so this measures
    ROBUSTNESS of the headline result, not a different (self-selected) window.
    """
    test_first, test_last = Y * 100 + 40, (Y + 1) * 100 + 39
    val_first, val_last = (Y - 1) * 100 + 40, Y * 100 + 39
    train_first, train_last = 200140, (Y - 1) * 100 + 39
    return {
        "train": (train_first, train_last),
        "val":   (val_first, val_last),
        "test":  (test_first, test_last),
    }


def assign_split(epiweek: int, win: dict) -> str:
    if COVID_HOLE[0] <= epiweek <= COVID_HOLE[1]:
        return "covid_excluded"
    for name in ("train", "val", "test"):
        lo, hi = win[name]
        if lo <= epiweek <= hi:
            return name
    return "out_of_range"          # summer gaps / weeks after test -> unused


def build_one(df_full: pd.DataFrame, Y: int) -> dict:
    win = season_windows(Y)
    df = df_full.copy()
    df["split"] = df["epiweek"].apply(lambda e: assign_split(int(e), win))

    out_dir = OUT_ROOT / f"cut{Y}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- scaler: fit on THIS cutoff's train only (population std, ddof=0) ---
    train = df[df["split"] == "train"]
    if len(train) == 0:
        raise RuntimeError(f"cut{Y}: empty train split")
    norm_params = {}
    for col in SCALER_COLUMNS:
        mean = float(train[col].mean())
        std = float(train[col].std(ddof=0))
        norm_params[col] = {
            "mean": mean, "std": std, "fit_on": "train",
            "fit_n_rows": int(len(train)),
            "fit_epiweek_first": int(train["epiweek"].min()),
            "fit_epiweek_last": int(train["epiweek"].max()),
        }

    counts = df["split"].value_counts().to_dict()

    # --- write (NEVER touches data/processed/) ---
    out_csv = out_dir / "ili_env_weekly_split.csv"
    df.to_csv(out_csv, index=False)
    (out_dir / "normalization_params.json").write_text(json.dumps({
        "_schema": "normalization_params_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "method": "StandardScaler (z = (x-mean)/std, population std ddof=0)",
        "fit_on": f"cut{Y} train split only -- no val/test leakage",
        "cutoff_test_season": f"{Y}-{Y+1}",
        "params": norm_params,
    }, indent=2))
    (out_dir / "split_boundaries.json").write_text(json.dumps({
        "_schema": "rolling_split_boundaries_v1",
        "cutoff_test_season": f"{Y}-{Y+1}",
        "windows": {k: {"epiweek_first": v[0], "epiweek_last": v[1]}
                    for k, v in win.items()},
        "covid_hole_excluded": {"epiweek_first": COVID_HOLE[0], "epiweek_last": COVID_HOLE[1]},
        "row_counts": {k: int(counts.get(k, 0))
                       for k in ("train", "val", "test", "covid_excluded", "out_of_range")},
    }, indent=2))

    return {
        "Y": Y, "test_season": f"{Y}-{Y+1}",
        "train": [int(train["epiweek"].min()), int(train["epiweek"].max()), int(counts.get("train", 0))],
        "val":   [win["val"][0], win["val"][1], int(counts.get("val", 0))],
        "test":  [win["test"][0], win["test"][1], int(counts.get("test", 0))],
        "csv_sha256": hashlib.sha256(out_csv.read_bytes()).hexdigest()[:16],
        "dir": str(out_dir.relative_to(_ROOT)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoffs", type=int, nargs="+", default=CUTOFFS)
    args = ap.parse_args()

    if not IN_PATH.exists():
        print(f"ERROR: {IN_PATH} not found", file=sys.stderr)
        return 1
    df_full = pd.read_csv(IN_PATH)
    print(f"[rolling] input {IN_PATH.relative_to(_ROOT)}: {len(df_full)} rows, "
          f"epiweek {df_full['epiweek'].min()}->{df_full['epiweek'].max()}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = []
    print(f"\n{'test':>9} {'train (first->last, n)':>32} {'val n':>7} {'test n':>7}")
    for Y in args.cutoffs:
        info = build_one(df_full, Y)
        manifest.append(info)
        print(f"{info['test_season']:>9} "
              f"{info['train'][0]}->{info['train'][1]} ({info['train'][2]:>3})   "
              f"{info['val'][2]:>7} {info['test'][2]:>7}")

    (OUT_ROOT / "cutoffs_manifest.json").write_text(json.dumps({
        "_schema": "rolling_cutoffs_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "pre_registered": True,
        "cutoffs": args.cutoffs,
        "covid_excluded_test_seasons": ["2019-20", "2020-21", "2021-22"],
        "note": "Expanding-window rolling-origin. Normalization re-fit per cutoff train (no leakage). "
                "Nothing in data/processed/ is modified.",
        "PREREG_covid_asymmetry": "post-COVID cutoffs (2022-24) include the suppressed-flu 2020-21 & "
                "2021-22 seasons in TRAINING (natural expanding window); pre-COVID cutoffs (2015-18) do "
                "not. Cross-cutoff differences are therefore NOT attributed solely to test-period shift "
                "(training composition also differs). Only the COVID hole 202011-202039 is excluded from all splits.",
        "PREREG_eval_window": "test = FULL season incl. summer weeks = same all-weeks method as headline "
                "test_strict; chosen so this is a robustness check of the headline, not a different window.",
        "PREREG_verdict_table": {
            "primary_endpoint": "per cutoff: is CG-Mamba closest-to-nominal (min |Cov95-0.95|, h=1-4 avg) in ALL 10 HHS regions?",
            "baseline_set": "vs the 5 DL baselines ONLY: LSTM, Vanilla Mamba, PatchTST, DLinear-ensemble, EpiDeep. SARIMAX EXCLUDED (not native UQ). This set is fixed here; may NOT be narrowed/widened by result.",
            "tie_rule": "strict delta=-1 (CG min in all 10) counts any margin; ADDITIONALLY record per cutoff the number of the 10 regions with a CLEAR margin (CG |dev| at least 0.02 below the best baseline's |dev| -- 0.02 chosen < wILI revision noise 0.034 so sub-noise 0.001-level 'wins' are not counted as clear). A '10/10' composed mostly of <0.02 margins is flagged noise-fragile, not strong replication.",
            "scoring": "CG scored as RAW NATIVE APMD (s_per_h NOT applied) -- identical to headline; NOT the Scaled variant (repeating the IV-F Scaled/native trap is forbidden). Driver asserts this.",
            "robust": ">=6/7 cutoffs replicate strict 10/10 -> may claim 'calibration dominance replicates across forecast origins'",
            "partial": "4-5/7 -> 'replicates in most origins; weaker in [named cutoffs]' (conditional)",
            "failed": "<=3/7 -> rolling-origin WEAKENS the headline; retract the robustness claim; report as-is",
            "baseline_wins": "any cutoff where a baseline beats CG in some region -> disclose which origin & which region",
            "secondary_recorded": "per cutoff ALSO record WIS and MAE (national + regional) -- reported as secondary (no claim), stored for result-blind completeness so a 'calibration robust but WIS breaks at cutoff X' pattern cannot be hidden.",
            "sanity_gate_quantitative": "driver on the CANONICAL headline split (train 2001-2018, test 2022-25) MUST reproduce regional Cov95 within 0.954 +/- 0.01 AND per-horizon ~0.998 -> 0.910; outside this band => treat as DRIVER BUG, HALT and investigate (NOT a scientific result).",
            "unconditional": ["report result regardless of direction",
                              "no rule change after computing",
                              "if <=3/7, it goes in the paper anyway",
                              "pass the quantitative sanity gate before trusting any rolling number"],
        },
        "manifest": manifest,
    }, indent=2))
    print(f"\n[rolling] wrote {len(manifest)} cutoffs -> {OUT_ROOT.relative_to(_ROOT)}/")
    print("[rolling] GPU 0 (data prep only). Nothing in data/processed/ touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
