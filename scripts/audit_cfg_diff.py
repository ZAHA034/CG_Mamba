"""scripts/audit_cfg_diff.py — 4-way config + wrapper orchestration audit
================================================================================
Goal: 단 1회 체계적 spec-audit. 4 sources line-by-line diff →
      mismatch 0 → final-train 진입; mismatch ≥1 → fix → smoke → 재확인.

4 sources:
  1. m1_9_hpo_phase2._build_cfg          (paper HPO Phase 2 mirror — Stage 3 only)
  2. ablation_retrain.build_frozen_hpo_cfg (paper final-train mirror — Stage 2+3)
  3. e1_hpo.build_cfg                     (corrected E1 HPO — Stage 2+3)
  4. e1_final_train.build_cfg             (corrected E1 final-train — Stage 2+3)

3 sections:
  A. CGMambaConfig field 4-way diff (categorized: must-match / expected-differ)
  B. Stage 2/3 args (Namespace) 4-way diff (build_stage2_args / build_stage3_args)
  C. Wrapper orchestration 4-way comparison (7 dims, manual structured)

Output: runs/_audit/cfg_diff_2026-06-18.md (mismatch hilite, BUG = red must-match)

CLI:
  python scripts/audit_cfg_diff.py
"""
from __future__ import annotations
import dataclasses
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

from src.utils.config import CGMambaConfig

import scripts.m1_9_hpo_phase2 as m1_9
import scripts.ablation_retrain as ar
import scripts.e1_hpo as e1h
import scripts.e1_final_train as e1f


OUT_DIR = _ROOT / "runs" / "_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "cfg_diff_2026-06-18.md"


# ─────────────────────────── Field categorization ───────────────────────────
# MUST_MATCH: mismatch = BUG (paper baseline unfairness)
# EXPECTED_DIFFER: mismatch = OK (design vs final / E1 grid sweep / etc.)
# NOTE_DIFFER: mismatch = needs disclosure but not BUG
FIELD_CATEGORY = {
    # Backbone — architecture-internal (MUST match across all)
    "d_state":            ("MUST_MATCH", "SSM state dim N"),
    "d_conv":             ("MUST_MATCH", "causal conv1d kernel"),
    "expand":             ("MUST_MATCH", "ED = expand × D"),
    "dt_rank":            ("MUST_MATCH", "dt projection rank"),
    "gate_rank":          ("MUST_MATCH", "low-rank gate bottleneck"),
    "gate_bias_init":     ("MUST_MATCH", "sigmoid(2.0) ≈ 0.88 near-identity"),
    "use_gate":           ("MUST_MATCH", "encoder block class selector"),

    # HMM (PhaseModule) — architecture-internal
    "n_states":           ("MUST_MATCH", "HMM K"),
    "hmm_seeds":          ("MUST_MATCH", "Stage 1 ckpt seed tuple"),
    "V_hmm_raw":          ("MUST_MATCH", "raw feature dim (3)"),
    "K_phase":            ("MUST_MATCH", "K=3 fixed"),
    "hmm_reg_covar":      ("MUST_MATCH", "5e-3 winner"),
    "hmm_n_init":         ("MUST_MATCH", "EM multi-start"),
    "stage3_enabled":     ("MUST_MATCH", "Stage 3 selective unfreeze flag"),

    # Rollout / Env / Decoder — architecture-internal
    "rollout_window":     ("MUST_MATCH", "PhaseModule.rollout W"),
    "env_input_dim":      ("MUST_MATCH", "[sh, t] = 2"),
    "env_hidden_dim":     ("MUST_MATCH", "env autoencoder MLP hidden"),
    "horizons":           ("MUST_MATCH", "CDC FluSight (1,2,3,4)"),
    "n_warm":             ("MUST_MATCH", "prefix injection length"),
    "main_input_dim":     ("MUST_MATCH", "backbone input dim"),

    # Training — protocol (MUST match paper)
    "batch_size":         ("MUST_MATCH", "32"),
    "weight_decay":       ("MUST_MATCH", "1e-5"),
    "grad_clip":          ("MUST_MATCH", "1.0"),
    "dropout":            ("MUST_MATCH", "0.0"),
    "stage1_lr":          ("MUST_MATCH", "Stage 1 HMM/Env lr"),
    "stage2_n_epochs":    ("MUST_MATCH", "200"),
    "stage2_backbone_wd": ("MUST_MATCH", "0.01"),
    "stage2_gate_wd":     ("MUST_MATCH", "1e-3"),
    "stage2_patience":    ("MUST_MATCH", "30"),
    "lr":                 ("NOTE_DIFFER", "legacy single-lr (Stage 1 / vanilla)"),
    "n_epochs":           ("NOTE_DIFFER", "legacy Stage 1 epochs (Stage 2 = stage2_n_epochs)"),

    # LRs — paper winner HP (MUST match within paper-comparable runs)
    "stage2_gate_lr":          ("MUST_MATCH", "paper winner 1e-3"),
    "stage2_backbone_lr":      ("MUST_MATCH", "paper winner 1e-4"),
    "stage3_hmm_lr":           ("MUST_MATCH", "1e-6 (= 1e-4 × 0.01)"),
    "stage3_state_embed_lr":   ("MUST_MATCH", "1e-6"),
    "stage3_env_lr":           ("MUST_MATCH", "1e-7"),
    "stage3_other_lr":         ("MUST_MATCH", "1e-4"),

    # Lookback — paper winner HP
    "lookback":           ("MUST_MATCH", "paper winner 104 (default 156)"),

    # E1 sweep dimensions (EXPECTED_DIFFER)
    "n_layers":           ("EXPECTED_DIFFER", "E1 sweeps {2,3,4}; paper fixed 3"),
    "d_model":            ("EXPECTED_DIFFER", "E1 sweeps {32,64,128}; paper fixed 64"),

    # Per-seed / per-run
    "seed":               ("EXPECTED_DIFFER", "run-level seed selection"),

    # Paths — design vs full (EXPECTED_DIFFER for E1, but verify)
    "data_csv":           ("EXPECTED_DIFFER", "design.csv (E1 HPO) / full.csv (E1 final / paper)"),
    "norm_json":          ("EXPECTED_DIFFER", "design norm / full norm"),
    "boundaries_json":    ("NOTE_DIFFER", "split boundaries (rarely consumed at runtime)"),
}


# ─────────────────────────── Build 4 cfgs ───────────────────────────
def build_4_cfgs() -> dict[str, CGMambaConfig]:
    """Sample point: seed=42, paper baseline (n_layers=3, d_model=64).
       m1_9 base = CG_TOP1_HP-equivalent fake winner dict.
    """
    seed = 42
    n_layers, d_model = 3, 64

    fake_base = {
        "gate_lr":      1e-3,
        "backbone_lr":  1e-4,
        "lookback":     104,
    }
    cfg_m1_9 = m1_9._build_cfg(
        fake_base, seed,
        hmm_ratio=0.01, state_embed_ratio=0.01, env_ratio=0.001,
    )
    cfg_ar = ar.build_frozen_hpo_cfg(seed)
    cfg_e1h = e1h.build_cfg(n_layers, d_model, seed)
    cfg_e1f = e1f.build_cfg(n_layers, d_model, seed)
    return {
        "m1_9 (paper HPO)":               cfg_m1_9,
        "ablation_retrain (paper final)": cfg_ar,
        "e1_hpo (corrected HPO)":         cfg_e1h,
        "e1_final_train (corrected fin)": cfg_e1f,
    }


def build_4_args() -> dict[str, dict[str, Namespace]]:
    """Stage 2 + Stage 3 args from each wrapper. m1_9 has Stage 3 only."""
    seed = 42
    n_layers, d_model = 3, 64

    # m1_9 builds args inline in _run_one. Reconstruct equivalently.
    fake_base = {"gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104}
    m1_9_run_name = m1_9._cell_run_name(fake_base, 0.01, 0.01, 0.001, seed)
    m1_9_stage2_dir = m1_9._phase1_stage2_dir(fake_base, seed)
    m1_9_hmm_dir = Path(str(m1_9.HMM_DIR_TEMPLATE).format(seed=seed))
    m1_9_stage3 = SimpleNamespace(
        smoke=False, epochs=30, patience=10,
        batch_size=32,
        stage2_dir=str(m1_9_stage2_dir), hmm_dir=str(m1_9_hmm_dir),
        env_encoder_ckpt=str(m1_9.ENV_CKPT),
        run_name=m1_9_run_name,
    )

    ar_run_name = "ablation_retrain_full_s42_stage2"
    ar_stage2_dir = _ROOT / "runs" / "m1_7_train" / ar_run_name
    ar_hmm_dir = Path(str(ar.HMM_DIR_TEMPLATE).format(seed=seed))
    ar_env_ckpt = ar.ENV_CKPT
    ar_stage2 = ar.make_stage2_args(ar_run_name, ar_hmm_dir, ar_env_ckpt,
                                     epochs=200, batch_size=32, smoke=False)
    ar_stage3 = ar.make_stage3_args(
        "ablation_retrain_full_s42_stage3", ar_stage2_dir, ar_hmm_dir, ar_env_ckpt,
        epochs=10, patience=0, batch_size=32, smoke=False,
    )

    e1h_run_name = f"e1_n{n_layers}_d{d_model}_s{seed}"
    e1h_stage2_dir = _ROOT / "runs/m1_7_train" / e1h_run_name
    e1h_stage2 = e1h.build_stage2_args(e1h_run_name, d_model)
    e1h_stage3 = e1h.build_stage3_args(f"{e1h_run_name}_stage3", e1h_stage2_dir, d_model)

    e1f_run_name = f"e1_final_n{n_layers}_d{d_model}_s{seed}"
    e1f_stage2_dir = _ROOT / "runs/m1_7_train" / e1f_run_name
    e1f_stage2 = e1f.build_stage2_args(e1f_run_name, d_model, seed)        # D.1 fix: seed added
    e1f_stage3 = e1f.build_stage3_args(f"{e1f_run_name}_stage3", e1f_stage2_dir, d_model, seed)

    return {
        "m1_9 (paper HPO)":               {"stage2": None,       "stage3": m1_9_stage3},
        "ablation_retrain (paper final)": {"stage2": ar_stage2,  "stage3": ar_stage3},
        "e1_hpo (corrected HPO)":         {"stage2": e1h_stage2, "stage3": e1h_stage3},
        "e1_final_train (corrected fin)": {"stage2": e1f_stage2, "stage3": e1f_stage3},
    }


# ─────────────────────────── Render ───────────────────────────
def _fmt_val(v) -> str:
    """Compact rendering for table cells."""
    if v is None:
        return "—"
    if isinstance(v, Path):
        try:
            return str(v.relative_to(_ROOT))
        except ValueError:
            return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-2 or abs(v) >= 1e4:
            return f"{v:.2e}"
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def render_field_diff(cfgs: dict[str, CGMambaConfig]) -> tuple[str, int, int, list[str]]:
    """Returns (markdown_table, n_must_match_bugs, n_expected_differs, bug_lines)."""
    sources = list(cfgs.keys())
    fields = [f.name for f in dataclasses.fields(CGMambaConfig)]
    fields_sorted = sorted(fields, key=lambda n: (
        # Category order: MUST_MATCH first, then NOTE, then EXPECTED
        {"MUST_MATCH": 0, "NOTE_DIFFER": 1, "EXPECTED_DIFFER": 2}.get(
            FIELD_CATEGORY.get(n, ("MUST_MATCH",))[0], 0),
        n,
    ))

    rows = []
    n_bugs = 0
    n_expected = 0
    bug_lines = []
    for fname in fields_sorted:
        vals = [getattr(c, fname) for c in cfgs.values()]
        # Comparison: stringify (handles Path/tuple/etc.)
        sv = [_fmt_val(v) for v in vals]
        all_same = len(set(sv)) == 1
        category, note = FIELD_CATEGORY.get(fname, ("MUST_MATCH", ""))

        if all_same:
            status = "✓"
        else:
            if category == "MUST_MATCH":
                status = "🔴 **BUG**"
                n_bugs += 1
                bug_lines.append(f"  - `{fname}` ({note}): {sv}")
            elif category == "NOTE_DIFFER":
                status = "🟡 note"
            else:
                status = "🟢 OK"
                n_expected += 1

        cat_short = {"MUST_MATCH": "must", "NOTE_DIFFER": "note", "EXPECTED_DIFFER": "exp"}[category]
        row = f"| `{fname}` | `{cat_short}` | {sv[0]} | {sv[1]} | {sv[2]} | {sv[3]} | {status} |"
        rows.append(row)

    header = (
        "| field | cat | " + " | ".join(sources) + " | status |\n"
        "|---|---|" + "---|" * (len(sources) + 1) + "\n"
    )
    table = header + "\n".join(rows)
    return table, n_bugs, n_expected, bug_lines


def render_args_diff(args_4: dict[str, dict[str, Namespace]]) -> str:
    """Compare Stage 2 + Stage 3 Namespace args."""
    sources = list(args_4.keys())
    out = []

    out.append("\n### B.1 Stage 2 args (`make_stage2_args` / `build_stage2_args`)\n")
    s2_fields = ["smoke", "epochs", "batch_size", "hmm_dir", "env_encoder_ckpt",
                 "wandb_mode", "run_name"]
    out.append("| field | " + " | ".join(sources) + " |")
    out.append("|---|" + "---|" * len(sources))
    for fname in s2_fields:
        vals = []
        for s in sources:
            ns = args_4[s]["stage2"]
            vals.append(_fmt_val(getattr(ns, fname, None)) if ns else "(no Stage 2)")
        out.append(f"| `{fname}` | " + " | ".join(vals) + " |")

    out.append("\n### B.2 Stage 3 args (`make_stage3_args` / `build_stage3_args`)\n")
    s3_fields = ["smoke", "epochs", "patience", "batch_size", "stage2_dir",
                 "hmm_dir", "env_encoder_ckpt", "run_name"]
    out.append("| field | " + " | ".join(sources) + " |")
    out.append("|---|" + "---|" * len(sources))
    for fname in s3_fields:
        vals = []
        for s in sources:
            ns = args_4[s]["stage3"]
            vals.append(_fmt_val(getattr(ns, fname, None)) if ns else "—")
        out.append(f"| `{fname}` | " + " | ".join(vals) + " |")

    return "\n".join(out) + "\n"


# ─────────────────────────── Wrapper orchestration table (manual structured) ───
WRAPPER_DIMS = [
    # (dim, m1_9, ablation_retrain, e1_hpo, e1_final_train)
    ("Stage 2 call?",
     "No (reuses Phase 1 ckpt at hpo_p1_*_s{seed}/best.pt)",
     "Yes (m1_7.train, 200 ep)",
     "Yes (m1_7.train, 200 ep)",
     "Yes (m1_7.train, 200 ep)"),
    ("Stage 3 call?",
     "Yes (m1_8.stage3_train, 30 ep, patience=10)",
     "Yes (m1_8.stage3_train, manifest 확정 10 ep, patience=0)",
     "Yes (m1_8.stage3_train, 30 ep, patience=10)",
     "Yes (m1_8.stage3_train, **10 ep ← D.3 fix**, patience=0)"),
    ("HMM ckpt scheme",
     "per-seed: `m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}`",
     "per-seed: `m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}`",
     "**seed42 only**: `m1_4_design_split/V_raw3_regcov5e-03_K3_seed42` (γ.4 design lock)",
     "**per-seed ← D.1 fix**: `m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}`"),
    ("Env ckpt scheme",
     "single: `m1_7_env_pretrain/env_encoder.pt`",
     "single: `m1_7_env_pretrain/env_encoder.pt`",
     "**per d_model**: `m1_7_env_pretrain_design/env_encoder_d{d_model}.pt`",
     "**per d_model**: `m1_7_env_pretrain_final/env_encoder_d{d_model}.pt`"),
    ("CGForecaster class swap (monkey-patch)?",
     "No (uses default CGForecaster)",
     "Yes (`m1_7.CGForecaster = subclass` for {no_env, no_encgates, uniform_rollout, full})",
     "No (default CGForecaster)",
     "No (default CGForecaster)"),
    ("CSV / norm path",
     "default (`ili_env_weekly_split.csv` / `normalization_params.json`)",
     "default (`ili_env_weekly_split.csv` / `normalization_params.json`)",
     "**design** (`ili_env_weekly_split_design.csv` / `normalization_params_design_train.json`)",
     "explicit-but-same-as-default (`ili_env_weekly_split.csv` / `normalization_params.json`)"),
    ("Post-train eval?",
     "Inside m1_8.stage3_train (test_mse / test_mase saved to final_metrics.json)",
     "Inside m1_8.stage3_train (same)",
     "Inline: design_val_inference + Cov95/WIS diagnostic + parquet",
     "**None** (separate e1_final_eval.py for held-out + test_strict)"),
]


def render_wrapper_table() -> str:
    sources = ["m1_9 (paper HPO)", "ablation_retrain (paper final)",
               "e1_hpo (corrected HPO)", "e1_final_train (corrected fin)"]
    out = ["| dim | " + " | ".join(sources) + " |",
           "|---|" + "---|" * len(sources)]
    for row in WRAPPER_DIMS:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


# ─────────────────────────── Main ───────────────────────────
def main():
    cfgs = build_4_cfgs()
    args_4 = build_4_args()

    table, n_bugs, n_exp, bug_lines = render_field_diff(cfgs)
    args_table = render_args_diff(args_4)
    wrapper_table = render_wrapper_table()

    # Sanity check (must-match list 누락 detect)
    declared = set(FIELD_CATEGORY.keys())
    actual = {f.name for f in dataclasses.fields(CGMambaConfig)}
    missing_in_category = sorted(actual - declared)
    extra_in_category = sorted(declared - actual)

    md = []
    md.append("# E1 corrected vs paper — 4-way config + wrapper audit\n")
    md.append(f"Date: 2026-06-18  |  Sample: seed=42, n_layers=3, d_model=64 (paper baseline cell)\n")
    md.append("\n## Summary\n")
    md.append(f"- **MUST_MATCH BUGs**: **{n_bugs}** (red)")
    md.append(f"- EXPECTED_DIFFER OKs: {n_exp} (green)")
    if bug_lines:
        md.append("\n**BUG list (immediate fix required)**:")
        md.extend(bug_lines)
    if missing_in_category:
        md.append(f"\n**Categorization missing** (defaulted to MUST_MATCH): {missing_in_category}")
    if extra_in_category:
        md.append(f"\n**Categorization stale** (no longer in dataclass): {extra_in_category}")

    md.append("\n## A. CGMambaConfig field 4-way diff\n")
    md.append("Categories: `must`=MUST_MATCH (mismatch=BUG) | `note`=NOTE_DIFFER (disclose) "
              "| `exp`=EXPECTED_DIFFER (E1 sweep / design vs full)\n")
    md.append(table)

    md.append("\n## B. Stage 2 + Stage 3 args (Namespace) diff\n")
    md.append(args_table)

    md.append("\n## C. Wrapper orchestration 4-way comparison\n")
    md.append(wrapper_table)

    md.append("\n## D. Risk-zone interpretation\n")
    md.append("1. **HMM ckpt scheme** — paper uses per-seed (5 HMM ckpts; each Stage 3 seed sees its own HMM); "
              "we use **seed42-only** HMM for both E1 HPO and E1 final. This is intentional for design-cut "
              "HMM re-fit (γ.4: design-train HMM was re-fit with seed=42 only), but `e1_final_train.py` "
              "also uses seed42-only HMM from `m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed42` — "
              "this **breaks paper-final fairness**. Required check: was per-seed HMM the intended final spec? "
              "If yes → CATCH (C-extra): final-train must point to per-seed HMM template.\n")
    md.append("2. **Env ckpt scheme** — paper uses a single env_encoder.pt (d_model-agnostic); we use "
              "per-d_model env_encoder_d{D}.pt. EnvModule.encoder L2's output dim is d_model "
              "(env_input_dim → env_hidden_dim → d_model projection), so paper's single ckpt is "
              "only valid for the Phase 1 winner d_model. For our d_model sweep, per-d_model is **necessary**. "
              "EXPECTED_DIFFER for E1 HPO. For E1 final-train: winner=n2_d128 → d128 env ckpt at "
              "`runs/m1_7_env_pretrain_final/env_encoder_d128.pt` (verified exists).\n")
    md.append("3. **Stage 3 epochs (10 vs 30)** — paper m1_9 uses 30, ablation_retrain CLI defaults to 10. "
              "Paper Table I headline ckpt source: `ablation_retrain` (the `full` ablation row trained from "
              "scratch with stage3_epochs=10). Our e1_final_train uses 30 (paper m1_9 / m2_1_final mirror). "
              "Required check: which paper script produced Table I `Full CG-Mamba` numbers? If "
              "ablation_retrain → e1_final_train STAGE3_EPOCHS should be 10, not 30.\n")
    md.append("4. **Stage 3 patience (10 vs 0)** — by design: e1_hpo=10 (paper m1_9 mirror, "
              "early-stop OK for HPO selection); e1_final_train=0 (paper ablation_retrain mirror, "
              "full 30 ep for final headline ckpt). EXPECTED, but tied to #3 — if paper headline = "
              "ablation_retrain with epochs=10, then patience=0 + epochs=10 is the right mirror.\n")
    md.append("5. **Post-train eval inside vs outside wrapper** — paper m1_9 / ablation_retrain do "
              "eval inside `m1_8.stage3_train` (test_mse / test_mase saved). e1_hpo does inline "
              "design-val inference; e1_final_train delegates to separate e1_final_eval.py. "
              "Functional: same backbone, different orchestration. EXPECTED.\n")

    md.append("\n## E. Decision\n")
    if n_bugs == 0:
        md.append("- ✅ **A (field diff)**: 0 must-match BUGs. paper baseline 과 같은 HP environment.\n")
    else:
        md.append(f"- 🔴 **A (field diff)**: {n_bugs} must-match BUG → fix required before final-train.\n")
    md.append("- ⚠️  **C/D (wrapper + risk-zone)**: 2 latent issues require user decision before final-train:\n")
    md.append("    - **D.1**: e1_final_train HMM = seed42 only (vs paper per-seed). Either (i) accept as "
              "design choice + disclose, or (ii) switch to per-seed template.\n")
    md.append("    - **D.3**: e1_final_train STAGE3_EPOCHS=30 (vs paper ablation_retrain CLI default 10). "
              "Either (i) keep 30 (paper m1_9 mirror), or (ii) switch to 10 (ablation_retrain mirror).\n")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(md))
    print(f"WROTE: {OUT_PATH.relative_to(_ROOT)}")
    print(f"  must-match BUGs:      {n_bugs}")
    print(f"  expected-differ OKs:  {n_exp}")
    print(f"  missing in category:  {missing_in_category}")
    print(f"  extra in category:    {extra_in_category}")
    if bug_lines:
        print("\nBUG list:")
        for b in bug_lines:
            print(b)


if __name__ == "__main__":
    main()
