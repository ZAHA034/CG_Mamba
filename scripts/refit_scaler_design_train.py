"""refit_scaler_design_train.py — γ.5 scaler 재적합 (누설 채널 #3 차단)
================================================================================
원 normalization_params.json: train 17 시즌 (200140-201839) fit → design-val 통계 누설.

E1 사전등록 γ.5:
  - design-train scaler: split=='train' AND epiweek <= 201539 (W39-2015)
  - 원 convention 유지: StandardScaler (z-score), std = population (ddof=0)
  - 출력: data/processed/normalization_params_design_train.json

검증:
  - 동일 코드 path 로 full train (원 범위) 재계산 → 원 JSON 과 bit-level 일치 확인
"""
from __future__ import annotations
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
SPLIT_CSV = _ROOT / "data/processed/ili_env_weekly_split.csv"
ORIG_NORM = _ROOT / "data/processed/normalization_params.json"
OUT_PATH = _ROOT / "data/processed/normalization_params_design_train.json"

DESIGN_TRAIN_EPIWEEK_MAX = 201539           # W39-2015 (γ.5 cut)
SCALER_COLS = ["ili_weighted_pct", "temperature_c", "specific_humidity_g_per_kg"]


def fit_scaler(df: pd.DataFrame, scope_label: str) -> dict:
    params = {}
    for col in SCALER_COLS:
        arr = df[col].to_numpy()
        params[col] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "fit_on": scope_label,
            "fit_n_rows": int(len(df)),
            "fit_epiweek_first": int(df["epiweek"].min()),
            "fit_epiweek_last": int(df["epiweek"].max()),
        }
    return params


def main():
    df = pd.read_csv(SPLIT_CSV)

    # === Verification: full train 재계산 vs 원 JSON bit-level 일치 ===
    print("=== Convention 검증: full train 재계산 vs 원 normalization_params.json ===")
    full_train = df[df["split"] == "train"]
    new_full = fit_scaler(full_train, "train")
    with open(ORIG_NORM) as f:
        orig = json.load(f)["params"]
    all_match = True
    for col in SCALER_COLS:
        m_match = abs(new_full[col]["mean"] - orig[col]["mean"]) < 1e-12
        s_match = abs(new_full[col]["std"] - orig[col]["std"]) < 1e-12
        flag = "✓" if (m_match and s_match) else "✗ MISMATCH"
        print(f"  {col:30s}  mean: new={new_full[col]['mean']:.8f} orig={orig[col]['mean']:.8f}  {flag}")
        if not (m_match and s_match):
            all_match = False
    assert all_match, "convention 불일치 — 원 JSON 과 bit-level 일치 안 함"
    print("  ✓ ddof=0 (population std) convention 검증 통과")

    # === design-train scaler 재적합 ===
    print(f"\n=== design-train scaler (γ.5, train ∩ epiweek ≤ {DESIGN_TRAIN_EPIWEEK_MAX}) ===")
    design_train = df[(df["split"] == "train") &
                       (df["epiweek"] <= DESIGN_TRAIN_EPIWEEK_MAX)]
    print(f"  rows: {len(design_train)} (원 train {len(full_train)} - design-val 156 rows)")
    print(f"  epiweek range: [{int(design_train.epiweek.min())}, {int(design_train.epiweek.max())}]")

    new_design = fit_scaler(design_train, "design_train")
    for col in SCALER_COLS:
        d_mean = new_design[col]["mean"] - orig[col]["mean"]
        d_std = new_design[col]["std"] - orig[col]["std"]
        print(f"  {col:30s}  mean={new_design[col]['mean']:.6f} (Δ {d_mean:+.4f})  "
              f"std={new_design[col]['std']:.6f} (Δ {d_std:+.4f})")

    out = {
        "_schema": "normalization_params_v1",
        "built_at": datetime.datetime.utcnow().isoformat() + "Z",
        "method": "StandardScaler (z-score: (x - mean) / std), std = population std (ddof=0)",
        "fit_on": ("design_train (E1 사전등록 γ.5): split=='train' AND epiweek <= "
                    f"{DESIGN_TRAIN_EPIWEEK_MAX}. 누설 채널 차단."),
        "input_dataset": "data/processed/ili_env_weekly.csv",
        "params": new_design,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved: {OUT_PATH}")
    return out


if __name__ == "__main__":
    main()
