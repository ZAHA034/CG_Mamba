"""Render M2.4 7-variant enhanced tables (Params + tS MAE + Protocol WIS/Cov + Conformal WIS/Cov).

Merges:
  - runs/m2_4_data_efficiency/m2_4_test_strict_all_baselines.csv (point MAE)
  - runs/m2_4_data_efficiency/m2_4_wis_protocol.csv    (Protocol-specific UQ)
  - runs/m2_4_data_efficiency/m2_4_wis_conformal.csv   (Split Conformal)

Output: markdown tables per variant, IEEE TABLE I format.

Usage:
    python3 scripts/m2_4_render_tables.py [--out OUT.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[1]
RUNS = _ROOT / "runs" / "m2_4_data_efficiency"
MAE_CSV = RUNS / "m2_4_test_strict_all_baselines.csv"
PROTO_CSV = RUNS / "m2_4_wis_protocol.csv"
CONF_CSV = RUNS / "m2_4_wis_conformal.csv"

PARAMS = {
    "sarima":        "n/a (stat)",
    "dlinear":       "840",
    "lstm":          "797,700",
    "vanilla_mamba": "108,033",
    "cg_mamba":      "115,389",
    "patchtst":      "71,428",
    "epideep":       "62,340",
}
NAME = {
    "sarima":        "SARIMA",
    "dlinear":       "DLinear",
    "lstm":          "LSTM",
    "vanilla_mamba": "Vanilla Mamba",
    "cg_mamba":      "CG-Mamba (ours)",
    "patchtst":      "PatchTST",
    "epideep":       "EpiDeep",
}
# Display order: SARIMA first (statistical reference), then DL baselines
# (CG-Mamba first as our contribution, then NN family, EpiDeep epidemic-specific DL)
ORDER = ["sarima", "cg_mamba", "vanilla_mamba", "patchtst", "lstm", "epideep", "dlinear"]
VARIANT_LABEL = {
    "3_seasons": "3 seasons (n_train=156 rows)",
    "4_seasons": "4 seasons (n_train=209 rows)",
    "5_seasons": "5 seasons (n_train=261 rows)",
    "7_seasons": "7 seasons (n_train=365 rows)",
    "10_seasons": "10 seasons (n_train=522 rows)",
    "13_seasons": "13 seasons (n_train=678 rows)",
    "17_seasons_full": "17 seasons / full (n_train=835 rows)",
}
VARIANT_ORDER = ["3_seasons", "4_seasons", "5_seasons", "7_seasons",
                 "10_seasons", "13_seasons", "17_seasons_full"]


def _fmt(mean: float | None, std: float | None, deterministic: bool = False) -> str:
    """Format mean±std, or single value, or em-dash if missing."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    if deterministic or std is None or np.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def _agg(df: pd.DataFrame, baseline: str, variant: str, col: str) -> tuple:
    """Return (mean, std) for baseline×variant×col across seeds. NaN if absent."""
    sub = df[(df["baseline"] == baseline) & (df["variant"] == variant)]
    if len(sub) == 0 or col not in sub.columns:
        return (np.nan, np.nan)
    vals = sub[col].dropna()
    if len(vals) == 0:
        return (np.nan, np.nan)
    if len(vals) == 1:   # deterministic (SARIMA) or ensemble (DLinear)
        return (float(vals.iloc[0]), np.nan)
    return (float(vals.mean()), float(vals.std()))


def _add_avg_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row avg columns for WIS and Cov."""
    for prefix in ["test_full", "test_strict"]:
        wis_cols = [f"{prefix}_wis_h{h}" for h in [1,2,3,4]]
        cov_cols = [f"{prefix}_cov95_h{h}" for h in [1,2,3,4]]
        if all(c in df.columns for c in wis_cols):
            df[f"{prefix}_wis_avg"] = df[wis_cols].mean(axis=1)
        if all(c in df.columns for c in cov_cols):
            df[f"{prefix}_cov95_avg"] = df[cov_cols].mean(axis=1)
    return df


def render(out_path: Path | None = None) -> str:
    """Build markdown for all 7 variants. Returns full string."""
    if not MAE_CSV.exists():
        raise FileNotFoundError(f"Missing {MAE_CSV}")

    mae_df = pd.read_csv(MAE_CSV)
    mae_df["test_full_mae_avg"] = mae_df[[f"test_full_mae_h{h}" for h in [1,2,3,4]]].mean(axis=1)
    mae_df["test_strict_mae_avg"] = mae_df[[f"test_strict_mae_h{h}" for h in [1,2,3,4]]].mean(axis=1)

    proto_df = _add_avg_cols(pd.read_csv(PROTO_CSV)) if PROTO_CSV.exists() else pd.DataFrame()
    conf_df = _add_avg_cols(pd.read_csv(CONF_CSV)) if CONF_CSV.exists() else pd.DataFrame()
    has_proto = len(proto_df) > 0 and "test_strict_wis_h1" in proto_df.columns
    has_conf = len(conf_df) > 0 and "test_strict_wis_h1" in conf_df.columns

    UQ_LABELS = {
        "sarima": "Kalman", "dlinear": "Ens.Gauss", "lstm": "MC-Drop",
        "vanilla_mamba": "MC-Drop", "patchtst": "MC-Drop", "cg_mamba": "Method F",
        "epideep": "MC-Drop",
    }

    out = []
    out.append("# M2.4 STRICT Data Efficiency — Enhanced Tables\n")
    out.append("Test: tS = test_strict (W40-2022 ~ W35-2025, post-COVID, n=149 windows after lookback edge).\n")
    out.append("Training data variants: 7 (3/4/5/7/10/13/17 seasons).\n")
    out.append("Seeds: 5 (42/123/456/789/1024) for DL baselines; SARIMA deterministic (n=1); DLinear reports 5-seed ensemble (n=1 row).\n")
    out.append("\n**Protocol UQ** (each baseline's native uncertainty method):\n")
    out.append("- SARIMA: Kalman parametric\n")
    out.append("- DLinear: 5-seed ensemble Gaussian\n")
    out.append("- LSTM: MC Dropout p=0.3, n=100\n")
    out.append("- Vanilla Mamba: MC Dropout p=0.2, n=100\n")
    out.append("- PatchTST: MC Dropout p=0.1, n=100\n")
    out.append("- EpiDeep: MC Dropout p=0.1, n=100\n")
    out.append("- CG-Mamba: Method F (HMM-derived calibrated PI)\n")
    out.append("\n**Conformal UQ** (uniform sharpness comparator, supplementary):\n")
    out.append("- All: Split conformal prediction with validation residuals, finite-sample corrected (Vovk 2005, Romano 2019)\n")
    out.append("- *Note*: Cov95 under conformal is ≥0.95 by algorithmic construction (theoretical guarantee), so only **conformal WIS** is reported as a sharpness comparator. Conformal Cov95 is not a model-quality metric.\n")
    out.append("\n**Columns**: `pWIS`/`pCov95` = Protocol track (model's native UQ); `cWIS` = Conformal track (post-hoc sharpness under uniform coverage).\n")
    out.append("\n---\n")

    for v in VARIANT_ORDER:
        out.append(f"\n## {VARIANT_LABEL[v]}\n")
        hdr = ["Model", "Params", "UQ"]
        hdr += [f"MAE h{h}" for h in [1,2,3,4]] + ["MAE avg"]
        if has_proto:
            hdr += [f"pWIS h{h}" for h in [1,2,3,4]] + ["pWIS avg", "pCov95"]
        if has_conf:
            hdr += ["cWIS avg"]   # Conformal Cov95 omitted — guaranteed ≥0.95 by construction
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "|".join(["---"] + ["---:"] * (len(hdr)-1)) + "|")

        for b in ORDER:
            det = (b == "sarima")
            row = [f"**{NAME[b]}**", PARAMS[b], UQ_LABELS[b]]
            for c in [f"test_strict_mae_h{h}" for h in [1,2,3,4]] + ["test_strict_mae_avg"]:
                m, s = _agg(mae_df, b, v, c)
                row.append(_fmt(m, s, det))
            if has_proto:
                for c in [f"test_strict_wis_h{h}" for h in [1,2,3,4]] + ["test_strict_wis_avg"]:
                    m, s = _agg(proto_df, b, v, c)
                    row.append(_fmt(m, s, det or (b == "dlinear")))
                m, s = _agg(proto_df, b, v, "test_strict_cov95_avg")
                row.append(_fmt(m, s, det or (b == "dlinear")))
            if has_conf:
                m, s = _agg(conf_df, b, v, "test_strict_wis_avg")
                row.append(_fmt(m, s, det or (b == "dlinear")))
            out.append("| " + " | ".join(row) + " |")

    # ─── Aggregate summary appendix ───
    out.append("\n---\n")
    out.append("\n## Aggregate Summary — h1-h4 averaged across all data sizes\n")

    # Build h1-h4 avg summary table for both tracks
    for track_name, track_df, track_has in [
        ("Protocol-specific WIS", proto_df, has_proto),
        ("Conformal WIS", conf_df, has_conf),
    ]:
        if not track_has:
            continue
        out.append(f"\n### {track_name} (h1-h4 average per variant × baseline)\n")
        hdr = ["Variant"] + [NAME[b] for b in ORDER]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|")
        for v in VARIANT_ORDER:
            row = [VARIANT_LABEL[v]]
            for b in ORDER:
                m, s = _agg(track_df, b, v, "test_strict_wis_avg")
                det = (b == "sarima") or (b == "dlinear")
                row.append(_fmt(m, s, det))
            out.append("| " + " | ".join(row) + " |")

    # Cov95 summary — Protocol only (Conformal Cov95 is trivially ≥0.95 by construction)
    if has_proto:
        out.append(f"\n### Protocol Cov95 (h1-h4 average per variant × baseline; target=0.95)\n")
        out.append("*Conformal Cov95 omitted: split conformal guarantees Cov95 ≥ 1−α by algorithmic construction, so the value reflects whether the calibration algorithm ran correctly — not model quality. The model-quality metric under conformal is sharpness (WIS), reported above.*\n")
        hdr = ["Variant"] + [NAME[b] for b in ORDER]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|")
        for v in VARIANT_ORDER:
            row = [VARIANT_LABEL[v]]
            for b in ORDER:
                m, s = _agg(proto_df, b, v, "test_strict_cov95_avg")
                det = (b == "sarima") or (b == "dlinear")
                row.append(_fmt(m, s, det))
            out.append("| " + " | ".join(row) + " |")

    out.append("\n---\n")
    out.append("\n## Provenance\n")
    out.append("- Point MAE source: `runs/m2_4_data_efficiency/m2_4_test_strict_all_baselines.csv` (217 rows = 7 SARIMA + 35 × 6 DL baselines)\n")
    out.append("- Protocol WIS source: `runs/m2_4_data_efficiency/m2_4_wis_protocol.csv` (189 rows; 0 NaN)\n")
    out.append("- Conformal WIS source: `runs/m2_4_data_efficiency/m2_4_wis_conformal.csv` (189 rows; 0 NaN)\n")
    out.append("- CG-Mamba ckpts: `runs/m2_4_data_efficiency/cg_mamba/seasons_{variant}/seed{seed}/manifest.json` (35)\n")
    out.append("- DL baseline ckpts: `runs/m2_4_data_efficiency/{baseline}/seasons_{variant}/seed{seed}/{baseline}_best.pt` (35 × 5 = 175)\n")
    out.append("- SARIMA: `runs/m2_4_data_efficiency/sarima/seasons_{variant}.json` (7)\n")
    out.append("- COVID masking boundary: `TS_BOUNDARY = 202240` (W40-2022); 105 weeks (2020-21 + 2021-22 seasons) excluded\n")

    md = "\n".join(out)
    if out_path:
        out_path.write_text(md)
        print(f"Saved: {out_path}")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=RUNS / "m2_4_enhanced_tables.md")
    args = ap.parse_args()
    md = render(args.out)
    print(md[:2000] + "\n...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
