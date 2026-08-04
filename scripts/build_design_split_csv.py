"""build_design_split_csv.py — γ.1 design-cut CSV 생성 (re-label only, NO drop)
================================================================================
목적: m1_7_train.py / WeeklyDataset 이 무수정으로 design-train 학습 + design-val 평가
       하도록 split 컬럼만 재라벨. design-train rows 는 살려둠 (lookback 경계 넘기 허용).

LOCKED γ.1 mapping:
  - 원 split=='train' AND epiweek ≤ 201539  → 'train' (design-train, 712 rows)
  - 201540 ≤ epiweek ≤ 201839              → 'val'   (design-val, 156 rows)
  - 201840 ≤ epiweek                       → 'excluded' (held-out + test, E1 무접촉)
  - 그 외 (covid_excluded 등)              → 'excluded'

검증:
  - design-train + design-val = 원 train 868 rows (완전 분할 확인)
  - 원 row 수 보존 (drop 0)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
INPUT_CSV = _ROOT / "data/processed/ili_env_weekly_split.csv"
OUTPUT_CSV = _ROOT / "data/processed/ili_env_weekly_split_design.csv"

DESIGN_TRAIN_END = 201539
DESIGN_VAL_START = 201540
DESIGN_VAL_END = 201839


def relabel(row) -> str:
    ep = int(row["epiweek"])
    orig = row["split"]
    if orig == "train" and ep <= DESIGN_TRAIN_END:
        return "train"
    if DESIGN_VAL_START <= ep <= DESIGN_VAL_END:
        return "val"
    return "excluded"


def main():
    df = pd.read_csv(INPUT_CSV)
    n_orig = len(df)
    df["split_orig"] = df["split"]
    df["split"] = df.apply(relabel, axis=1)

    # === 검증 ===
    counts = df["split"].value_counts().to_dict()
    counts_orig = df["split_orig"].value_counts().to_dict()
    print(f"=== Design-cut CSV ({OUTPUT_CSV.name}) ===")
    print(f"  input rows:  {n_orig}")
    print(f"  output rows: {len(df)}  (drop 0 확인: {n_orig == len(df)})")
    print(f"\n  원 split distribution:    {counts_orig}")
    print(f"  design split distribution: {counts}")

    # γ.1 LOCKED expected counts
    expected_train_design = 712
    expected_val_design = 156

    n_train_design = counts.get("train", 0)
    n_val_design = counts.get("val", 0)

    print(f"\n  γ.1 검증:")
    print(f"    design-train (re-label='train'): {n_train_design}  (expected {expected_train_design})  {'✓' if n_train_design == expected_train_design else '✗'}")
    print(f"    design-val   (re-label='val'):   {n_val_design}  (expected {expected_val_design})  {'✓' if n_val_design == expected_val_design else '✗'}")
    assert n_train_design == expected_train_design, "design-train row count mismatch"
    assert n_val_design == expected_val_design, "design-val row count mismatch"
    assert n_train_design + n_val_design == 868, "design-train + design-val ≠ 원 train 868"

    # design-val epiweek range 검증
    dv = df[df["split"] == "val"]
    dv_min, dv_max = int(dv.epiweek.min()), int(dv.epiweek.max())
    assert dv_min == DESIGN_VAL_START and dv_max == DESIGN_VAL_END, (
        f"design-val epiweek mismatch: [{dv_min}, {dv_max}]")
    print(f"    design-val epiweek range: [{dv_min}..{dv_max}]  ✓")

    # design-train epiweek range 검증
    dt = df[df["split"] == "train"]
    dt_max = int(dt.epiweek.max())
    assert dt_max <= DESIGN_TRAIN_END, (
        f"design-train max epiweek {dt_max} > {DESIGN_TRAIN_END} (γ.1 위반)")
    print(f"    design-train epiweek max: {dt_max} ≤ {DESIGN_TRAIN_END}  ✓")

    # excluded 검증 (FluSight 2018-19 + COVID + test 모두 excluded)
    ex = df[df["split"] == "excluded"]
    ex_min = int(ex.epiweek.min())
    print(f"    excluded (held-out + COVID + test): n={len(ex)}, epiweek_min={ex_min}")
    assert ex_min >= DESIGN_VAL_END + 1, "excluded 가 design-val 안에 침범"
    assert ex_min == 201840, f"excluded 시작점 {ex_min} ≠ 201840 (원 val 시작)"

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
