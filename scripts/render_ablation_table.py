"""Unified Ablation Table — combines A3 (phase rollout) + A4 (gate composition).

Reads:
  runs/ablation_a3/ablation_a3_summary.csv  (modes: full / transition / uniform)
  runs/ablation_a4/ablation_a4_summary.csv  (conditions: A4-and / phase-only / env-only / none / none+uniform)

Produces 7-row unified ablation table presenting:
  - Full CG-Mamba (baseline)
  - Component removal (3 rows): -EnvModule, -PhaseModule, -Both encoder gates
  - Rollout perturbation (2 rows): hard transition, uniform
  - Fully vanilla (none+uniform) — strongest ablation

Writes:
  notebooks/figures/ablation/ablation_table.pdf
  notebooks/figures/ablation/ablation_table.png
  runs/ablation_table.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
A3_CSV = _ROOT / "runs" / "ablation_a3" / "ablation_a3_summary.csv"
A4_CSV = _ROOT / "runs" / "ablation_a4" / "ablation_a4_summary.csv"
OUT_DIR = _ROOT / "notebooks" / "figures" / "ablation"
MD_OUT = _ROOT / "runs" / "ablation_table.md"


def fmt(mean: float, std: float | None = None, digits: int = 3) -> str:
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def build_rows() -> list[dict]:
    a3 = pd.read_csv(A3_CSV).set_index("mode")
    a4 = pd.read_csv(A4_CSV).set_index("condition")

    full = a4.loc["A4-and"]              # == a3.loc["full"]
    env_off = a4.loc["A4-phase-only"]    # phase gate kept, env removed
    phase_off = a4.loc["A4-env-only"]    # env kept, phase encoder gate removed
    both_off = a4.loc["A4-none"]         # both encoder gates removed, rollout intact
    rollout_hard = a3.loc["transition"]
    rollout_uni = a3.loc["uniform"]
    vanilla = a4.loc["A4-none+uniform"]

    rows = [
        {
            "label": "Full CG-Mamba",
            "phase_gate": "Y", "env_gate": "Y", "rollout": "emission-aware",
            "row": full, "marker": "*",
        },
        {
            "label": "  - EnvModule",
            "phase_gate": "Y", "env_gate": "–", "rollout": "emission-aware",
            "row": env_off, "marker": "",
        },
        {
            "label": "  - PhaseModule (encoder gate)",
            "phase_gate": "–", "env_gate": "Y", "rollout": "emission-aware",
            "row": phase_off, "marker": "",
        },
        {
            "label": "  - Both encoder gates",
            "phase_gate": "–", "env_gate": "–", "rollout": "emission-aware",
            "row": both_off, "marker": "",
        },
        {
            "label": "Rollout: hard transition",
            "phase_gate": "Y", "env_gate": "Y", "rollout": "hard transition",
            "row": rollout_hard, "marker": "",
        },
        {
            "label": "Rollout: uniform prior",
            "phase_gate": "Y", "env_gate": "Y", "rollout": "uniform",
            "row": rollout_uni, "marker": "",
        },
        {
            "label": "Fully vanilla (no phase anywhere)",
            "phase_gate": "–", "env_gate": "–", "rollout": "uniform",
            "row": vanilla, "marker": "",
        },
    ]
    return rows


def render_pdf(rows: list[dict], out_pdf: Path, out_png: Path) -> None:
    full = rows[0]["row"]
    full_mae = full["mae_avg_mean"]
    full_wis = full["wis_avg_mean"]
    full_cov = full["cov95_avg_mean"]

    header = [
        "Variant",
        "Phase\ngate",
        "Env\ngate",
        "Rollout",
        "MAE (avg)\nmean±std",
        "WIS (avg)\nmean±std",
        "Cov95 (avg)\nmean±std",
        "ΔMAE\nvs full",
    ]
    table_data = []
    cell_colors = []
    for r in rows:
        row = r["row"]
        mae = row["mae_avg_mean"]
        mae_std = row["mae_avg_std"]
        wis_v = row["wis_avg_mean"]
        wis_std = row["wis_avg_std"]
        cov_v = row["cov95_avg_mean"]
        cov_std = row["cov95_avg_std"]
        d_mae = mae - full_mae
        d_mae_str = f"{d_mae:+.3f}" if abs(d_mae) > 1e-4 else "0.000"

        table_data.append([
            r["label"],
            r["phase_gate"],
            r["env_gate"],
            r["rollout"],
            fmt(mae, mae_std),
            fmt(wis_v, wis_std),
            fmt(cov_v, cov_std),
            d_mae_str,
        ])
        # Color baseline row
        if r["marker"] == "*":
            cell_colors.append(["#fff2cc"] * len(header))
        else:
            cell_colors.append(["white"] * len(header))

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_data, colLabels=header, cellLoc="center", loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.9)

    # Header style
    for j in range(len(header)):
        table[(0, j)].set_facecolor("#dceaf2")
        table[(0, j)].set_text_props(weight="bold")
    # Left-align Variant column
    for i in range(1, len(rows) + 1):
        table[(i, 0)].set_text_props(ha="left")

    fig.suptitle(
        "Table III — Ablation Study (5-seed mean±std on test_S, h=1..4 average)",
        fontsize=12, y=0.97, weight="bold",
    )
    caption = (
        "Baseline highlighted in yellow. Encoder gates removed at inference via the disable_gate\n"
        "path (no re-training; bit-identical to vanilla Mamba). Rollout-only rows keep both encoder\n"
        "gates active but perturb the multi-horizon phase prior. \"Fully vanilla\" removes phase\n"
        "context from encoder AND rollout (uniform prior + no context_vec)."
    )
    fig.text(0.5, 0.06, caption, ha="center", va="top", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"       {out_png}")


def render_markdown(rows: list[dict], out_md: Path) -> None:
    full = rows[0]["row"]
    full_mae = full["mae_avg_mean"]

    lines = [
        "# Table III — Ablation Study",
        "",
        "5-seed mean±std on test_S, MAE/WIS/Cov95 averaged over h=1..4.",
        "",
        "| Variant | Phase gate | Env gate | Rollout | MAE (avg) | WIS (avg) | Cov95 (avg) | ΔMAE vs full |",
        "|---|:---:|:---:|---|---|---|---|---|",
    ]
    for r in rows:
        row = r["row"]
        mae = row["mae_avg_mean"]
        mae_std = row["mae_avg_std"]
        wis_v = row["wis_avg_mean"]
        wis_std = row["wis_avg_std"]
        cov_v = row["cov95_avg_mean"]
        cov_std = row["cov95_avg_std"]
        d_mae = mae - full_mae
        d_mae_str = f"{d_mae:+.3f}" if abs(d_mae) > 1e-4 else "0.000"
        label = r["label"]
        if r["marker"] == "*":
            label = f"**{label.strip()}**"
        lines.append(
            f"| {label} | {r['phase_gate']} | {r['env_gate']} | {r['rollout']} | "
            f"{fmt(mae, mae_std)} | {fmt(wis_v, wis_std)} | {fmt(cov_v, cov_std)} | {d_mae_str} |"
        )

    lines += [
        "",
        "**Notes.** Baseline (full CG-Mamba) shown in bold. "
        "Encoder gates disabled at inference via the `disable_gate` path "
        "(no re-training; bit-identical to vanilla Mamba). "
        "Rollout-only rows keep both encoder gates active but perturb the multi-horizon phase prior. "
        "\"Fully vanilla\" removes phase context from encoder AND rollout.",
        "",
        "**Provenance.**",
        "- A3 (rollout) modes: `runs/ablation_a3/ablation_a3_summary.csv`",
        "- A4 (gate composition) conditions: `runs/ablation_a4/ablation_a4_summary.csv`",
        "- Both share the baseline row (full / A4-and) with identical metrics: "
        f"MAE={full_mae:.3f}, WIS={full['wis_avg_mean']:.3f}, "
        f"Cov95={full['cov95_avg_mean']:.3f}.",
    ]
    out_md.write_text("\n".join(lines))
    print(f"Saved: {out_md}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    render_pdf(rows, OUT_DIR / "ablation_table.pdf", OUT_DIR / "ablation_table.png")
    render_markdown(rows, MD_OUT)


if __name__ == "__main__":
    main()
