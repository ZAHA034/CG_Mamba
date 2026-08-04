"""Fig-POS: gap-filling Pareto (amortized ∩ natively-calibrated).
2 panels (National | Regional). x=WIS (lower better), y=|Cov95-0.95| (lower better).
Marker: star = single amortized model (CG-Mamba + DL baselines);
        square = per-series classical refit (SARIMAX, Persistence).
Frontier through the non-dominated set {SARIMAX, CG-Mamba}.
All coordinates are audit-verified (2026-07-04): national from Table I/III,
regional CG-Mamba from raw re-eval, regional baselines computed here from the
same phase_3 sources that feed Table IV / the combined regional figure.
Output: runs/fig_pos_pareto.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = Path("/A.I_DATA/jbnu/JeongHa/CG_Mamba")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType (IEEE)
})
CGM_C, DL_C, CLS_C = "#1a9850", "#8c8c8c", "#2166ac"

# ---- NATIONAL (audit-verified: Table I method-specific Cov95 + WIS) ----
# (name, WIS, Cov95, kind)  kind: 'cgm'|'dl'|'classical'
NAT = [
    ("CG-Mamba", 0.399, 0.993, "cgm"),
    ("SARIMAX",  0.218, 0.888, "classical"),
    ("Persistence", 0.422, 0.878, "classical"),
    ("PatchTST", 0.368, 0.698, "dl"),
    ("Vanilla Mamba", 0.372, 0.378, "dl"),
    ("EpiDeep", 0.394, 0.377, "dl"),
    ("LSTM", 0.466, 0.335, "dl"),
    ("DLinear", 0.441, 0.289, "dl"),
    ("iTransformer", 0.521, 0.270, "dl"),
    ("N-BEATS", 0.487, 0.272, "dl"),
    ("TimesNet", 0.597, 0.225, "dl"),
]

# ---- REGIONAL baselines: cross-region mean from phase_3 sources (verifiable) ----
HOR = (1, 2, 3, 4)
def xregion(csv, names):
    df = pd.read_csv(_ROOT / csv)
    out = {}
    for b in names:
        s = df[df.baseline == b]
        if not len(s):
            continue
        wis = np.mean([s[[f"tS_wis_h{h}" for h in HOR]].mean(axis=1).groupby(s.region).mean().mean()])
        cov = np.mean([s[[f"tS_cov95_h{h}" for h in HOR]].mean(axis=1).groupby(s.region).mean().mean()])
        out[b] = (float(wis), float(cov))
    return out
reg = {}
reg.update(xregion("runs/phase_3_region_wis.csv", ["lstm", "patchtst", "vanilla_mamba"]))
reg.update(xregion("runs/phase_3_region_wis_extras.csv", ["dlinear_ensemble_gauss", "epideep"]))
# CG-Mamba regional RAW (audit re-eval CSV) + SARIMAX regional
cgmr = pd.read_csv(_ROOT / "runs/e1_final/n3_d64_regional_perhorizon_raw.csv")
cgm_wis = cgmr[[f"tS_wis_h{h}" for h in HOR]].mean(axis=1).groupby(cgmr.region).mean().mean()
cgm_cov = cgmr[[f"tS_cov95_h{h}" for h in HOR]].mean(axis=1).groupby(cgmr.region).mean().mean()
LAB = {"lstm": "LSTM", "patchtst": "PatchTST", "vanilla_mamba": "Vanilla Mamba",
       "dlinear_ensemble_gauss": "DLinear", "epideep": "EpiDeep"}
REG = [("CG-Mamba", float(cgm_wis), float(cgm_cov), "cgm"),
       ("SARIMAX", 0.301, 0.916, "classical")]
for b, (w, c) in reg.items():
    REG.append((LAB[b], w, c, "dl"))

def pareto_front(pts):
    """Non-dominated set minimizing both x=WIS and y=|dev|; return sorted by x."""
    P = sorted(pts, key=lambda t: t[0])
    front, best = [], float("inf")
    for x, y, name in P:
        if y < best - 1e-9:
            front.append((x, y, name)); best = y
    return front

# label offsets (dx, dy, ha) to avoid overlap in the crowded clusters
OFFS_NAT = {
    "Vanilla Mamba": (-0.008, 0.000, "right"), "EpiDeep": (0.008, 0.000, "left"),
    "DLinear": (-0.008, 0.006, "right"), "N-BEATS": (0.000, 0.026, "center"),
    "iTransformer": (0.009, -0.010, "left"), "TimesNet": (0.009, 0.000, "left"),
    "LSTM": (0.009, 0.000, "left"), "PatchTST": (0.009, 0.000, "left"),
    "CG-Mamba": (0.010, -0.006, "left"), "SARIMAX": (-0.010, 0.010, "right"),
    "Persistence": (0.009, 0.000, "left"),
}
OFFS_REG = {
    "DLinear": (-0.009, 0.006, "right"), "EpiDeep": (-0.009, -0.010, "right"),
    "LSTM": (0.009, 0.006, "left"), "Vanilla Mamba": (-0.009, 0.000, "right"),
    "PatchTST": (0.009, -0.008, "left"), "CG-Mamba": (0.011, -0.004, "left"),
    "SARIMAX": (-0.010, 0.012, "right"),
}

def draw(ax, data, title, offs, xlim):
    for name, wis, cov, kind in data:
        dev = abs(cov - 0.95)
        mk = "s" if kind == "classical" else "*"
        c = CGM_C if kind == "cgm" else (CLS_C if kind == "classical" else DL_C)
        sz = 340 if kind == "cgm" else (150 if kind == "classical" else 120)
        ax.scatter(wis, dev, marker=mk, s=sz, c=c, edgecolors="black",
                   linewidths=1.2 if kind == "cgm" else 0.6, zorder=6 if kind == "cgm" else 4,
                   alpha=0.95 if kind != "dl" else 0.8)
        dx, dy, ha = offs.get(name, (0.007, 0.0, "left"))
        ax.annotate(name, (wis, dev), (wis + dx, dev + dy),
                    fontsize=8.5 if kind == "cgm" else 7.5,
                    fontweight="bold" if kind == "cgm" else "normal",
                    color="black", ha=ha, va="center", zorder=7)
    # Pareto frontier (lower-left)
    front = pareto_front([(w, abs(c - 0.95), n) for n, w, c, k in data])
    fx = [p[0] for p in front]; fy = [p[1] for p in front]
    ax.plot(fx, fy, "--", color="#444444", lw=1.3, zorder=3, alpha=0.8)
    ax.fill_between([0, max(fx)], [0, 0], [min(fy), min(fy)], color="#1a9850", alpha=0.05, zorder=0)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.text(0.98, 0.02, "ideal\n(sharp + calibrated)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, style="italic", color="#1a7a3a")
    ax.set_xlabel("WIS  (sharpness — lower better)")
    ax.set_ylabel("|Cov95 $-$ 0.95|  (miscalibration — lower better)")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_ylim(-0.03, 0.80)
    ax.set_xlim(*xlim)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
draw(axes[0], NAT, "National (single U.S. series)", OFFS_NAT, (0.15, 0.66))
draw(axes[1], REG, "Regional (10 HHS, zero-shot transfer)", OFFS_REG, (0.24, 0.585))
handles = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor=CGM_C, markeredgecolor="k", markersize=17, label="CG-Mamba (ours)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor=DL_C, markeredgecolor="k", markersize=12, label="DL baseline (single amortized)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=CLS_C, markeredgecolor="k", markersize=11, label="Classical (per-series refit)"),
    Line2D([0], [0], ls="--", color="#444444", label="Pareto frontier"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf", "png"):
    fig.savefig(_ROOT / f"runs/fig_pos_pareto.{ext}", dpi=300, bbox_inches="tight")
print("saved runs/fig_pos_pareto.{pdf,png}")
print(f"\nNational CG-Mamba: WIS 0.399 |dev| 0.043")
print(f"Regional CG-Mamba (raw re-eval): WIS {cgm_wis:.3f} Cov95 {cgm_cov:.3f} |dev| {abs(cgm_cov-0.95):.3f}")
print("Regional baselines (cross-region mean, from phase_3 sources):")
for n, w, c, k in REG:
    if k == "dl": print(f"  {n:14} WIS {w:.3f}  Cov95 {c:.3f}  |dev| {abs(c-0.95):.3f}")
