"""Regional test_strict: (1) WIS decomposition (dispersion/under/over) per model,
(2) regional average MAE per model. Both from canonical artifacts, gate-checked.

Reuses src.eval.wis (no re-implementation). For the honest 'sharpness bought with
coverage' quantification and the M2 regional-MAE completeness line.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "src"))
from src.eval.wis import wis, wis_decomposed, coverage, REQUIRED_QUANTILES

TEST_STRICT = 202240
QCOLS = {q: f"q{q:.3f}" for q in REQUIRED_QUANTILES}

# ---- (A) Regional avg MAE per model (canonical phase_3 tS_h1..4) ----
ev = pd.concat([pd.read_csv(_ROOT/"runs/phase_3_region_eval.csv"),
                pd.read_csv(_ROOT/"runs/phase_3_region_eval_extras.csv")], ignore_index=True)
tS = ["tS_h1", "tS_h2", "tS_h3", "tS_h4"]
mae = ev.groupby("baseline")[tS].mean()
mae["avg_h1_4"] = mae[tS].mean(axis=1)
print("=== (A) Regional test_strict MAE per model (mean over 10 regions x 5 seeds) ===")
print(mae.round(4).to_string())
print(f"[gate] CG h1={mae.loc['cg_mamba','tS_h1']:.4f} vs DLinear h1={mae.loc['dlinear','tS_h1']:.4f}"
      f"  (DLinear lead {100*(mae.loc['cg_mamba','tS_h1']-mae.loc['dlinear','tS_h1'])/mae.loc['cg_mamba','tS_h1']:.1f}%)")

# ---- CG canonical regional WIS/Cov95 (gate reference, e1_final) ----
cg = pd.read_csv(_ROOT/"runs/e1_final/n3_d64_regional_perhorizon_raw.csv")
cg_wis_canon = cg[[f"tS_wis_h{h}" for h in (1,2,3,4)]].mean().mean()
cg_cov_canon = cg[[f"tS_cov95_h{h}" for h in (1,2,3,4)]].mean().mean()
print(f"\n[CG canonical regional] WIS={cg_wis_canon:.4f}  Cov95={cg_cov_canon:.4f}  (paper 0.393 / 0.954)")

# ---- (B) WIS decomposition per model (native_predictive, test_strict) ----
df = pd.read_parquet(_ROOT/"runs/decision_native/native_predictive.parquet")
df = df[df.eps_h1 >= TEST_STRICT]
print(f"\n=== (B) WIS decomposition per model, regional test_strict (native predictive dump) ===")
print(f"{'model':14s} {'WIS':>7s} {'Cov95':>7s} | {'disp':>7s} {'under':>7s} {'over':>7s} | disp% under% over%")
rows = {}
for m, g in df.groupby("model"):
    y = g.y_true.to_numpy()
    qd = {q: g[QCOLS[q]].to_numpy() for q in REQUIRED_QUANTILES}
    w = float(wis(y, qd).mean())
    c = coverage(y, qd, 0.05)
    dec = wis_decomposed(y, qd)
    disp, und, ov = float(dec["disp" if "disp" in dec else "dispersion"].mean()), float(dec["under"].mean()), float(dec["over"].mean())
    tot = disp + und + ov
    rows[m] = dict(wis=w, cov=c, disp=disp, under=und, over=ov)
    print(f"{m:14s} {w:7.4f} {c:7.4f} | {disp:7.4f} {und:7.4f} {ov:7.4f} | "
          f"{100*disp/tot:4.1f} {100*und/tot:4.1f} {100*ov/tot:4.1f}")

cg_g = rows["cg_mamba"]
gate = abs(cg_g["wis"] - cg_wis_canon) <= 0.01 and abs(cg_g["cov"] - cg_cov_canon) <= 0.01
print(f"\n[gate] native CG WIS={cg_g['wis']:.4f} (canon {cg_wis_canon:.4f}), "
      f"Cov95={cg_g['cov']:.4f} (canon {cg_cov_canon:.4f}) -> {'PASS' if gate else 'FAIL'}")
