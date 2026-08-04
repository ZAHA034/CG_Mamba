"""Persistence baseline on the weekly ILI dataset.

Persistence prediction:  ŷ_{t+h} = y_t  (use the most recent observed %wILI as
the forecast for h weeks ahead).

This module recomputes the baseline directly from `ili_env_weekly_split.csv`
(not via the training DataLoader) so the number is reproducible without any
PyTorch dependency, and can be cross-checked against the inline `persistence_mae`
function in `scripts/m1_2_vanilla_sanity.py`.

Gap-aware: a pair (t, t+h) is included ONLY if epiweeks are strictly
consecutive across the full chain t, t+1, ..., t+h. The single train gap
(200220 -> 200240) is therefore excluded.

Usage:
    python -m src.baselines.persistence
    python -m src.baselines.persistence --horizons 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "data" / "processed" / "ili_env_weekly_split.csv"

WANDB_ENTITY = "hjs40111-personal"
WANDB_PROJECT = "cg-mamba-jbhi"
WANDB_TAGS = ["persistence", "weekly", "baseline", "v2.1.7-A++"]


def is_consecutive_epiweek(prev_ep: int, curr_ep: int) -> bool:
    py, pw = prev_ep // 100, prev_ep % 100
    cy, cw = curr_ep // 100, curr_ep % 100
    if py == cy:
        return cw == pw + 1
    if cy == py + 1 and cw == 1 and pw in (52, 53):
        return True
    return False


def persistence_pairs(
    df: pd.DataFrame, target_split: str, horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Yield (y_predict, y_target) arrays where:
      - target row (r+h) is in `target_split`
      - the epiweek chain r -> r+1 -> ... -> r+h is strictly consecutive.

    The predictor row r may be in any split (cross-split lookback per PLAN §4.2).
    For `target_split='test_post_covid'` (DEPRECATED v2.0.4), we filter to epiweek >= 202140.
    For `target_split='test_strict'` (v2.1.7-A++), we filter to epiweek >= 202240
    (W40-2022, excluding 2020-21 + 2021-22 anomalous seasons; PLAN §4.1).
    """
    df = df.sort_values("epiweek").reset_index(drop=True)
    eps = df["epiweek"].to_numpy()
    splits = df["split"].to_numpy()
    y = df["ili_weighted_pct"].to_numpy()
    N = len(df)

    preds, targets = [], []
    for r in range(N - horizon):
        target_idx = r + horizon
        # Filter target by split
        if target_split == "test_post_covid":
            # DEPRECATED v2.0.4 spec (kept for backward-compat)
            if splits[target_idx] != "test" or eps[target_idx] < 202140:
                continue
        elif target_split == "test_strict":
            # v2.1.7-A++ : exclude 2020-21 + 2021-22 anomalous seasons
            if splits[target_idx] != "test" or eps[target_idx] < 202240:
                continue
        else:
            if splits[target_idx] != target_split:
                continue
        # Verify chain r -> r+h is consecutive
        ok = True
        for j in range(r, target_idx):
            if not is_consecutive_epiweek(int(eps[j]), int(eps[j + 1])):
                ok = False
                break
        if not ok:
            continue
        preds.append(y[r])
        targets.append(y[target_idx])

    return np.asarray(preds), np.asarray(targets)


def mae_rmse(preds: np.ndarray, targets: np.ndarray) -> dict:
    if len(preds) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    err = preds - targets
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "n": int(len(preds)),
        "target_mean_raw": float(targets.mean()),
        "target_std_raw": float(targets.std(ddof=0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--csv", type=str, default=str(CSV_PATH))
    ap.add_argument("--save", type=str, default=None,
                    help="Optional path to save the JSON table.")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Skip W&B summary logging.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[persistence] dataset: {Path(args.csv).relative_to(REPO_ROOT)}  "
          f"({len(df)} rows, epiweek {df['epiweek'].min()}..{df['epiweek'].max()})")
    print()

    results: dict = {}
    splits_to_eval = ["train", "val", "test", "test_post_covid", "test_strict"]
    print(f"{'Split':<16} {'h':>3} {'n_pairs':>8} {'y_mean':>8} {'y_std':>7} "
          f"{'MAE':>8} {'RMSE':>8} {'MAE/y_std':>10}")
    print("-" * 78)
    for split in splits_to_eval:
        results[split] = {}
        for h in args.horizons:
            preds, targets = persistence_pairs(df, split, h)
            m = mae_rmse(preds, targets)
            results[split][h] = m
            if m["n"] > 0:
                print(f"{split:<16} {h:>3} {m['n']:>8} "
                      f"{m['target_mean_raw']:>8.4f} {m['target_std_raw']:>7.4f} "
                      f"{m['mae']:>8.4f} {m['rmse']:>8.4f} "
                      f"{m['mae']/m['target_std_raw']:>10.3f}")
            else:
                print(f"{split:<16} {h:>3} {0:>8}      n/a     n/a      n/a      n/a       n/a")

    if args.save:
        out_path = Path(args.save)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "source_csv": str(Path(args.csv).resolve().relative_to(REPO_ROOT)),
                "method": "persistence: y_hat_{t+h} = y_t (gap-aware consecutive chain)",
                "results": results,
            }, f, indent=2)
        print(f"\nsaved: {out_path.relative_to(REPO_ROOT)}")

    wandb_enabled = (not args.no_wandb) and _WANDB_AVAILABLE
    if wandb_enabled:
        run = wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT,
            group="persistence_v2.1.7-A++",
            name="persistence_weekly",
            tags=WANDB_TAGS,
            config={
                "method": "persistence: y_hat_{t+h} = y_t",
                "horizons": args.horizons,
                "splits_evaluated": splits_to_eval,
                "source_csv": str(Path(args.csv).resolve().relative_to(REPO_ROOT)),
                "phase": "M2.6_Persistence",
                "deterministic": True,
            },
            reinit=True,
        )
        for split, hres in results.items():
            for h, m in hres.items():
                if m["n"] > 0:
                    run.summary[f"{split}_h{h}_mae"] = m["mae"]
                    run.summary[f"{split}_h{h}_rmse"] = m["rmse"]
                    run.summary[f"{split}_h{h}_n"] = m["n"]
                    run.summary[f"{split}_h{h}_target_mean_raw"] = m["target_mean_raw"]
                    run.summary[f"{split}_h{h}_target_std_raw"] = m["target_std_raw"]
        run.finish()
        print("[persistence] W&B summary logged.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
