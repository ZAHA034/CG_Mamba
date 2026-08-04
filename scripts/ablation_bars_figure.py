"""IV-H component-ablation figure: 3-panel bars (dMAE, dWIS, dCov95).

Each panel shows the paired change (ablated - full) for the three from-scratch
ablations. Significant (95% bootstrap CI excludes 0) bars are full-opacity with a
star; non-significant bars are faded --- so each metric's single dominant driver
"lights up". Reads runs/ablation_retrain/bootstrap_ci.json (matched-env re-run).

Okabe-Ito palette, Nimbus Sans, pdf.fonttype=42 (IEEE-compliant, matches Fig 3-5).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.6,
})

ROOT = Path("/A.I_DATA/jbnu/JeongHa/CG_Mamba")
OUT_DIR = ROOT / "CGM_v2_paper" / "figures"
res = json.load(open(ROOT / "runs/ablation_retrain/bootstrap_ci.json"))["results"]

# component order + Okabe-Ito colors (consistent across panels)
COMPS = [("no_env", "$-$Env", "#E69F00"),
         ("no_phase", "$-$Phase", "#56B4E9"),
         ("uniform_rollout", "$-$Roll", "#009E73")]
METRICS = [("mae_avg", r"$\Delta$MAE", "point acc."),
           ("wis_avg", r"$\Delta$WIS", "sharpness"),
           ("cov95_avg", r"$\Delta$Cov95", "calibration")]

fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.95))
for ax, (mkey, mlab, msub) in zip(axes, METRICS):
    xs = range(len(COMPS))
    for i, (ckey, clab, col) in enumerate(COMPS):
        v = res[ckey][mkey]
        mean, lo, hi = v["mean"], v["ci_low"], v["ci_high"]
        sig = v["ci_excludes_zero"]
        yerr = [[mean - lo], [hi - mean]]
        ax.bar(i, mean, width=0.68, color=col, alpha=1.0 if sig else 0.32,
               edgecolor="#222222", linewidth=0.5, zorder=3)
        ax.errorbar(i, mean, yerr=yerr, fmt="none", ecolor="#222222",
                    elinewidth=0.7, capsize=1.8, zorder=4)
        if sig:
            top = hi if mean >= 0 else lo
            va = "bottom" if mean >= 0 else "top"
            off = (hi - lo) * 0.12 * (1 if mean >= 0 else -1)
            ax.annotate(f"{mean:+.3f}*", (i, top + off), ha="center", va=va,
                        fontsize=5.6, fontweight="bold", zorder=5)
    ax.axhline(0, color="#555555", lw=0.6, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([c[1] for c in COMPS], fontsize=6.2)
    ax.set_title(f"{mlab}\n({msub})", fontsize=7.0, pad=3)
    ax.tick_params(axis="y", labelsize=5.8, length=2)
    ax.margins(y=0.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(w_pad=0.6)
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "ablation_bars.pdf", dpi=300, bbox_inches="tight")
png = "/tmp/claude-1000/-A-I-DATA-jbnu-JeongHa/abe32db0-df35-4793-91c0-c7f3f9dbd6b1/scratchpad/ablation_bars.png"
fig.savefig(png, dpi=300, bbox_inches="tight")
print("saved:", OUT_DIR / "ablation_bars.pdf", "and", png)
