"""K-Selection Figure — supports §III.3 (HMM state count selection).

Combines three sources of evidence:
  Panel A: BIC vs K  (lower is better in absolute fit terms — NOT used for selection)
  Panel B: kappa_min vs K (seed stability — the selection criterion)
  Panel C: dead-state seed count vs K (degeneracy indicator)

Data sources:
  runs/m1_4_ablation_gaussian_hmm/k_search_results.json (V_raw=3, K=3/4/5, 3-seed BIC + kappa)
  runs/m1_4_phase_dynamics/winner_ranking.json  (V_raw=3, K=3/4/5, dead-state flag, 3-seed)
  runs/k3_paper_seeds_kappa.json (K=3 5-seed kappa=1.0 confirmation)

Output:
  notebooks/figures/k_selection/k_selection.pdf + .png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
SEARCH = _ROOT / "runs" / "m1_4_ablation_gaussian_hmm" / "k_search_results.json"
WINNER = _ROOT / "runs" / "m1_4_phase_dynamics" / "winner_ranking.json"
K3_5SEED = _ROOT / "runs" / "k3_paper_seeds_kappa.json"
OUT_DIR = _ROOT / "notebooks" / "figures" / "k_selection"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    search = json.load(open(SEARCH))
    winner = json.load(open(WINNER))
    k3_5seed = json.load(open(K3_5SEED))

    v3 = search["v3"]
    Ks = sorted(int(k) for k in v3["per_K"])

    bic_by_K = {}
    kappa_by_K = {}
    for K in Ks:
        kdata = v3["per_K"][str(K)]
        bic_by_K[K] = kdata["bics"]
        kappa_by_K[K] = [p["kappa"] for p in kdata["kappas_pairwise"]]

    dead_by_K = {}
    for cell in winner:
        if cell["cell"].startswith("V_raw3_"):
            K = int(cell["cell"].split("_K")[1].split("_")[0])
            dead_by_K[K] = cell["any_dead"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel A: BIC
    ax = axes[0]
    bic_means = [np.mean(bic_by_K[K]) for K in Ks]
    bic_stds = [np.std(bic_by_K[K]) for K in Ks]
    ax.errorbar(Ks, bic_means, yerr=bic_stds, fmt="o-", color="#444",
                capsize=4, linewidth=1.6, markersize=8)
    for K, mean in zip(Ks, bic_means):
        ax.annotate(f"{mean:.0f}", (K, mean), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, color="#333")
    ax.set_xticks(Ks)
    ax.set_xlabel("K (number of HMM states)", fontsize=10)
    ax.set_ylabel("BIC (lower = better fit)", fontsize=10)
    ax.set_title("(A) BIC vs K — fit improves monotonically;\n"
                 "BIC alone does not identify the right K",
                 fontsize=10)
    ax.grid(alpha=0.3)

    # Panel B: kappa_min (stability)
    ax = axes[1]
    bar_x = np.arange(len(Ks))
    kappa_mins = [min(kappa_by_K[K]) for K in Ks]
    kappa_means = [np.mean(kappa_by_K[K]) for K in Ks]

    colors = ["#2ca02c" if K == 3 else "#d62728" for K in Ks]
    bars = ax.bar(bar_x, kappa_mins, color=colors, alpha=0.7,
                   edgecolor="black", linewidth=0.8,
                   label="kappa_min (3-seed search)")
    for x, K, kmin, kmean in zip(bar_x, Ks, kappa_mins, kappa_means):
        ax.text(x, kmin + 0.02, f"min={kmin:.3f}\nmean={kmean:.3f}",
                ha="center", fontsize=8)
        if K == 3:
            ax.text(x, 0.04, f"5-seed κ_min={k3_5seed['kappa_min']:.3f}",
                    ha="center", fontsize=8, color="#0a5d0a", weight="bold")

    ax.axhline(0.9, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label="Stability threshold (κ ≥ 0.9)")
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f"K={K}" for K in Ks])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Pairwise Cohen's κ across seeds", fontsize=10)
    ax.set_title("(B) Seed stability vs K — only K=3 is reliably reproducible\n"
                 "(5-seed re-run confirms κ=1.000)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: dead-state degeneracy flag
    ax = axes[2]
    dead_flags = [1 if dead_by_K.get(K, False) else 0 for K in Ks]
    bar_x = np.arange(len(Ks))
    colors = ["#d62728" if d else "#2ca02c" for d in dead_flags]
    ax.bar(bar_x, [1] * len(Ks), color=colors, alpha=0.7,
            edgecolor="black", linewidth=0.8)
    for x, K, d in zip(bar_x, Ks, dead_flags):
        msg = "DEAD-STATE\n(degenerate EM)" if d else "no dead states\n(all K used)"
        ax.text(x, 0.5, msg, ha="center", va="center", fontsize=9,
                weight="bold", color="white")
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f"K={K}" for K in Ks])
    ax.set_yticks([])
    ax.set_title("(C) Posterior degeneracy — K=4 collapses to <K\n"
                 "effective states in main run (winner_ranking.json)",
                 fontsize=10)

    fig.suptitle(
        "Figure 6 — K-Selection Diagnostics (V_raw=3, Gaussian HMM, reg_cov=5e-3)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf = OUT_DIR / "k_selection.pdf"
    png = OUT_DIR / "k_selection.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"       {png}")


if __name__ == "__main__":
    main()
