"""CG-Mamba architecture block diagram (v3 — publication quality).

Design principles (following Mamba 2024, PatchTST 2023, Transformer original):
- Strict horizontal pipeline (Input → Modules → Gates → Context → Encoder → Decoder → Output)
- Stage 1 / Stage 2 grouped via subtle background bands
- Tap-style input branching (no bypass-arrow crossings)
- Right-angle arrow routing (orthogonal)
- Tensor shapes annotated inside boxes (consistent location)
- Minimal arrow labels (only at logical transitions)

Output: runs/figures/architecture_diagram.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "runs" / "figures"

# ═══════════════════════════════════════════════════════════════
# COLOR PALETTE (muted academic)
# ═══════════════════════════════════════════════════════════════
COLOR_INPUT = "#e6eef8"; COLOR_INPUT_EDGE = "#3a5d8e"
COLOR_PHASE = "#fbe5cc"; COLOR_PHASE_EDGE = "#c66f2e"
COLOR_ENV = "#dceacb"; COLOR_ENV_EDGE = "#4f7a30"
COLOR_GATE = "#f3cdd9"; COLOR_GATE_EDGE = "#9a3a64"
COLOR_BACKBONE = "#dbceec"; COLOR_BACKBONE_EDGE = "#5d3a9c"
COLOR_DECODER = "#fff0b3"; COLOR_DECODER_EDGE = "#947300"
COLOR_OUTPUT = "#c8e2e2"; COLOR_OUTPUT_EDGE = "#2f7373"
COLOR_FROZEN = "#888888"

# Stage background shading
COLOR_STAGE1_BG = "#fff8f0"  # very light cream for Stage 1
COLOR_STAGE2_BG = "#f0f0fa"  # very light lavender for Stage 2


def box(ax, x, y, w, h, title, sub=None,
        color=COLOR_INPUT, edge=COLOR_INPUT_EDGE,
        frozen=False, title_size=10, sub_size=7.5):
    """Rounded rectangle with title + subtitle inside."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.10",
        linewidth=1.8,
        edgecolor=edge if not frozen else COLOR_FROZEN,
        facecolor=color,
        linestyle="--" if frozen else "-",
        zorder=3,
    )
    ax.add_patch(rect)
    cx = x + w/2
    if sub:
        ax.text(cx, y + h*0.66, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", zorder=4)
        ax.text(cx, y + h*0.28, sub, ha="center", va="center",
                fontsize=sub_size, style="italic", color="#222", zorder=4)
    else:
        ax.text(cx, y + h/2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", zorder=4)
    if frozen:
        ax.text(x + w - 0.08, y + h - 0.08, "❄",
                ha="right", va="top", fontsize=12, color=COLOR_FROZEN, zorder=5)


def arrow(ax, start, end, color="#1a1a1a", lw=1.5, head_size=18, zorder=2):
    """Straight directed arrow (orthogonal segments handled separately)."""
    a = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=head_size,
        linewidth=lw,
        color=color,
        zorder=zorder,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)


def line(ax, start, end, color="#1a1a1a", lw=1.5, zorder=2):
    """Plain line segment (no arrowhead) — for orthogonal routing."""
    ax.plot([start[0], end[0]], [start[1], end[1]],
            color=color, linewidth=lw, zorder=zorder, solid_capstyle="round")


def orthogonal_arrow(ax, start, end, via, color="#1a1a1a", lw=1.5, head_size=18):
    """L-shaped arrow: start → via → end (right-angle routing)."""
    line(ax, start, via, color=color, lw=lw)
    arrow(ax, via, end, color=color, lw=lw, head_size=head_size)


def stage_band(ax, x0, y0, w, h, color, edge, label, label_pos="bottom-left"):
    """Subtle background band marking a stage. Label placed in non-overlapping corner."""
    rect = Rectangle((x0, y0), w, h,
                     linewidth=1.0, edgecolor=edge, facecolor=color,
                     linestyle=":", alpha=0.45, zorder=1)
    ax.add_patch(rect)
    if label_pos == "bottom-left":
        ly = y0 + 0.15
    elif label_pos == "bottom-right":
        ly = y0 + 0.15
    else:
        ly = y0 + h - 0.18
    if "right" in label_pos:
        lx = x0 + w - 0.12
        ha = "right"
    else:
        lx = x0 + 0.12
        ha = "left"
    ax.text(lx, ly, label,
            ha=ha, va="center", fontsize=9.5, fontweight="bold",
            color=edge, zorder=6,
            bbox=dict(facecolor="white", edgecolor=edge,
                      alpha=0.95, pad=3, boxstyle="round,pad=0.25"))


def main():
    fig, ax = plt.subplots(figsize=(15.5, 8.5))

    # ═══════════════════════════════════════════════════════════
    # GRID (columns and rows)
    # ═══════════════════════════════════════════════════════════
    # Columns (left x-coord of each module column):
    # C1=0.4 (inputs)
    # C2=2.5 (Stage 1: PhaseModule / EnvModule)
    # C3=4.8 (Stage 2 gates: Phase gate / state_embed / Env embedding)
    # C4=7.1 (context_vec)
    # C5=9.0 (CGMambaBlock × 3)
    # C6=11.3 (EntropyAware Decoder)
    # C7=13.5 (Output ŷ)
    #
    # Rows (y-coordinates):
    # y=6.8: emission-aware rollout (top branch)
    # y=4.7: main horizontal pipeline (encoder/decoder/output row)
    # y=3.6: gate row (Phase gate, context_vec, Env embedding all on this center)
    # y=2.0: env input row
    # y=4.7: x input row (same as main pipeline)

    # ═══════════════════════════════════════════════════════════
    # STAGE BACKGROUND BANDS
    # ═══════════════════════════════════════════════════════════
    # Stage 1 band: covers PhaseModule + EnvModule columns only (excludes top rollout row)
    stage_band(ax, 2.3, 1.4, 2.2, 4.5, COLOR_STAGE1_BG, "#aaaaaa",
               "Stage 1  (frozen)", label_pos="bottom-left")
    # Stage 2 band: covers gates + context + Mamba + Decoder (excludes top rollout row)
    stage_band(ax, 4.7, 1.4, 8.5, 4.5, COLOR_STAGE2_BG, "#aaaaaa",
               "Stage 2  (end-to-end trained)", label_pos="bottom-right")

    # ═══════════════════════════════════════════════════════════
    # MAIN HORIZONTAL PIPELINE (y=4.4-5.0 centered at 4.7)
    # ═══════════════════════════════════════════════════════════

    # --- Inputs (column 1) ---
    box(ax, 0.4, 4.4, 1.7, 0.8,
        "x  (main)", "[B, L, 4]:  wILI, count,\nproviders, patients",
        COLOR_INPUT, COLOR_INPUT_EDGE)
    box(ax, 0.4, 2.0, 1.7, 0.8,
        "env", "[B, L, 2]:\nhumidity, temperature",
        COLOR_INPUT, COLOR_INPUT_EDGE)

    # --- Stage 1 modules (column 2) ---
    box(ax, 2.5, 4.4, 1.9, 0.8,
        "PhaseModule",
        "Gaussian HMM (K=3)\nforward-backward",
        COLOR_PHASE, COLOR_PHASE_EDGE, frozen=True)
    box(ax, 2.5, 2.0, 1.9, 0.8,
        "EnvModule",
        "MLP (2→32→64)\n+ aux decoder ❄",
        COLOR_ENV, COLOR_ENV_EDGE, frozen=True)

    # --- Stage 2 gates (column 3) ---
    box(ax, 4.8, 4.4, 1.9, 0.8,
        "Phase gate",
        "σ(γ_t · E)\n[B, L−1, D]",
        COLOR_GATE, COLOR_GATE_EDGE)
    # state_embeddings — placed ABOVE Phase gate (inline with gate row, distinct visual layer)
    box(ax, 4.8, 5.35, 1.9, 0.55,
        "state_embeddings  E", "[K, D] = 192 params",
        "#ffffff", COLOR_PHASE_EDGE,
        title_size=8.5, sub_size=7)
    box(ax, 4.8, 2.0, 1.9, 0.8,
        "Env embedding",
        "MLP(env)\n[B, L, D]",
        COLOR_GATE, COLOR_GATE_EDGE)

    # --- context_vec (column 4) — aligned with CGMambaBlock (main row) ---
    box(ax, 7.1, 4.1, 1.7, 1.3,
        "context_vec",
        "= gate_phase ⊙ env\n[B, L−1, D]\n(phase-modulated)",
        COLOR_GATE, COLOR_GATE_EDGE,
        title_size=10, sub_size=7.5)

    # --- CGMambaBlock × 3 (column 5) ---
    box(ax, 9.0, 4.1, 2.1, 1.3,
        "CGMambaBlock × 3",
        "depth-3 stack\n107,712 + 5,016 params\ngated W_Δ, W_B, W_C",
        COLOR_BACKBONE, COLOR_BACKBONE_EDGE)

    # --- EntropyAwareDecoder (column 6) ---
    box(ax, 11.4, 4.1, 1.9, 1.3,
        "EntropyAware\nDecoder",
        "LOGIC-1:\neff_gate = c·gate + (1−c)\n261 params",
        COLOR_DECODER, COLOR_DECODER_EDGE)

    # --- Output ŷ (column 7) ---
    box(ax, 13.6, 4.35, 0.85, 0.85,
        "ŷ", "[B, H]\nh = 1..4",
        COLOR_OUTPUT, COLOR_OUTPUT_EDGE,
        title_size=13, sub_size=7.5)

    # ═══════════════════════════════════════════════════════════
    # TOP BRANCH: emission-aware rollout (y=6.4-7.0)
    # ═══════════════════════════════════════════════════════════
    box(ax, 4.8, 6.4, 2.4, 0.7,
        "Emission-aware rollout",
        "γ̃_h = norm(γ̃_{h−1} · A) ⊙ p(x̂_h | state)",
        COLOR_PHASE, COLOR_PHASE_EDGE,
        title_size=10, sub_size=7.5)

    box(ax, 7.5, 6.4, 1.4, 0.7,
        "gamma_all", "[B, H, K]",
        "#ffffff", COLOR_PHASE_EDGE,
        title_size=9, sub_size=7)

    # ═══════════════════════════════════════════════════════════
    # ARROWS (all orthogonal / horizontal)
    # ═══════════════════════════════════════════════════════════
    AR_MAIN = "#1a1a1a"
    AR_PHASE = "#a8541e"  # darker phase color for branch
    AR_ENV = "#3d5e23"    # darker env color
    AR_DASH = "#888"

    # === Main horizontal pipeline (x → ... → ŷ) ===
    # x → PhaseModule
    arrow(ax, (2.1, 4.80), (2.5, 4.80), AR_MAIN)
    # PhaseModule → Phase gate (horizontal)
    arrow(ax, (4.4, 4.80), (4.8, 4.80), AR_MAIN)
    # state_embeddings → Phase gate (vertical down, directly above Phase gate now)
    arrow(ax, (5.75, 5.35), (5.75, 5.22), AR_PHASE, lw=1.2)

    # Phase gate → context_vec (single horizontal arrow, both aligned at y=4.80)
    arrow(ax, (6.70, 4.80), (7.10, 4.80), AR_MAIN)

    # context_vec → CGMambaBlock (single clean horizontal arrow at y=4.75)
    arrow(ax, (8.80, 4.75), (9.00, 4.75), AR_MAIN)

    # CGMambaBlock → Decoder
    arrow(ax, (11.10, 4.75), (11.40, 4.75), AR_MAIN)
    ax.text(11.25, 5.00, "encoder_out",
            ha="center", va="center", fontsize=7.5, style="italic", color="#444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
            zorder=5)

    # Decoder → ŷ
    arrow(ax, (13.30, 4.75), (13.60, 4.75), AR_MAIN)

    # === x → CGMambaBlock direct (tap-style: route in y=3.5 gap, enter Mamba from below) ===
    # Clean routing in the gap between env row (top y=2.8) and context_vec bottom (y=4.1)
    line(ax, (1.25, 4.40), (1.25, 3.50), AR_DASH, lw=1.0)
    line(ax, (1.25, 3.50), (10.05, 3.50), AR_DASH, lw=1.0)
    arrow(ax, (10.05, 3.50), (10.05, 4.10), AR_DASH, lw=1.0)
    ax.text(4.80, 3.65, "x[:, 1:, :]  (sequence input to Mamba)",
            ha="center", va="center", fontsize=7.5, style="italic", color="#666",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
            zorder=5)

    # === env branch (bottom) ===
    arrow(ax, (2.1, 2.40), (2.5, 2.40), AR_ENV)
    arrow(ax, (4.4, 2.40), (4.8, 2.40), AR_ENV)
    # Env embedding → context_vec bottom (right-angle: right then up, entering context_vec from below)
    line(ax, (6.70, 2.40), (7.95, 2.40), AR_ENV)
    arrow(ax, (7.95, 2.40), (7.95, 4.10), AR_ENV)

    # === Top branch: emission-aware rollout ===
    # PhaseModule top output → rollout (up then right)
    line(ax, (3.45, 5.20), (3.45, 6.75), AR_PHASE)
    arrow(ax, (3.45, 6.75), (4.80, 6.75), AR_PHASE)
    ax.text(3.20, 6.00, "γ_last,\nx_window",
            ha="right", va="center", fontsize=7, style="italic", color="#666",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
            zorder=5)

    # Rollout → gamma_all
    arrow(ax, (7.20, 6.75), (7.50, 6.75), AR_PHASE)

    # gamma_all → Decoder (right-angle: right then down)
    line(ax, (8.90, 6.75), (12.35, 6.75), AR_PHASE)
    arrow(ax, (12.35, 6.75), (12.35, 5.40), AR_PHASE)
    ax.text(10.60, 6.92, "gamma_all + state_embeddings  →  per-horizon eff_gate",
            ha="center", va="center", fontsize=7.5, style="italic", color="#444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
            zorder=5)

    # ═══════════════════════════════════════════════════════════
    # TITLE — removed for LaTeX integration (caption handled by \caption{})
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    # LEGEND
    # ═══════════════════════════════════════════════════════════
    legend = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=11,
               markerfacecolor=COLOR_PHASE, markeredgecolor=COLOR_PHASE_EDGE,
               label="Phase / HMM"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=11,
               markerfacecolor=COLOR_ENV, markeredgecolor=COLOR_ENV_EDGE,
               label="Environmental"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=11,
               markerfacecolor=COLOR_GATE, markeredgecolor=COLOR_GATE_EDGE,
               label="Context gate"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=11,
               markerfacecolor=COLOR_BACKBONE, markeredgecolor=COLOR_BACKBONE_EDGE,
               label="Mamba backbone"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=11,
               markerfacecolor=COLOR_DECODER, markeredgecolor=COLOR_DECODER_EDGE,
               label="Decoder"),
        Line2D([0], [0], linestyle="--", linewidth=1.5,
               color=COLOR_FROZEN, label="❄ Frozen module"),
    ]
    ax.legend(handles=legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.06),
              fontsize=9, ncol=6, framealpha=0.95,
              columnspacing=1.4, handletextpad=0.6)

    # ═══════════════════════════════════════════════════════════
    # FINAL FORMATTING
    # ═══════════════════════════════════════════════════════════
    ax.set_xlim(0, 15)
    ax.set_ylim(1.0, 7.4)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "architecture_diagram.pdf"
    png_path = OUT_DIR / "architecture_diagram.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path.relative_to(_ROOT)}")
    print(f"Saved: {png_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
