# Release Manifest — CG-Mamba code repository

Working notes for turning this working directory into the public code release cited by the paper.
Not part of the published paper; you may delete it before or after pushing.

Prepared 2026-08-03. The `.gitignore` and `README.md` in this directory are already release-ready.

---

## 1. What ships vs. what stays out

**Included (tracked):** `src/` (models, data pipeline, eval, baselines, utils, tests), `scripts/`
(reproduction drivers — scratch/superseded/figure-only scripts are individually noted below),
`README.md`, `LICENSE`, `requirements.txt`, `config/credentials.template.json`, the data
provenance/manifest files, the pre-registrations, and a **curated set of small canonical result
artifacts** under `runs/` that back every table/figure (see §3).

**Excluded (via `.gitignore`):** `external/` (2.3 GB), `wandb/`, `logs/`, `outputs/`,
`__pycache__/`, `.pytest_cache/`, raw + processed bulk data, model checkpoints (`runs/**/*.pt`,
~4 GB → Zenodo), and the internal/superseded docs `CG_Mamba_PLAN.md`, `docs/`, `paper/`,
`latex_submission/`, `notebooks/`, `CGM_v2_paper/` (paper LaTeX).

## 2. Author decisions (defaults chosen — override if you prefer)

- **`CGM_v2_paper/` (paper LaTeX): excluded from the code repo.** Convention is a code-only repo;
  the two pre-registrations under it are force-added (§3). If you instead want the LaTeX in the repo,
  remove the `CGM_v2_paper/` line from `.gitignore` — but then also delete `latex_submission/`
  (superseded) so reviewers don't read the wrong tree, and keep `FEEDBACK_RESPONSE.md` out.
- **`notebooks/` and `docs/`: excluded** (internal). Re-add individual notebooks only after clearing
  their outputs if you want them public.
- **Model checkpoints: not in git** (largest exceeds GitHub's 100 MB cap). Upload `runs/**/*.pt` to
  **Zenodo** and put the DOI in the paper's Code Availability statement + `README.md`.

## 3. Canonical small artifacts — MUST be force-added

`runs/` and `data/processed/*.parquet` are fully git-ignored, so a bare `git add` **silently drops
the result artifacts the paper's tables come from.** Force-add them explicitly:

```bash
# from the repo root, after `git init`
while read -r f; do
  [ -e "$f" ] && git add -f "$f" || echo "MISSING (check): $f"
done <<'EOF'
runs/m2_1_final_topk/master_summary.json
runs/m2_4_data_efficiency/m2_4_test_strict_all_baselines.csv
runs/m2_4_data_efficiency/m2_4_summary.csv
runs/m2_3_extra_baselines_per_h_split.csv
runs/master_3column_wis_table.csv
runs/master_wis_table.csv
runs/bootstrap_ci_summary.json
runs/baselines/sarima.json
runs/baselines/persistence.json
runs/phase_3_region_eval.csv
runs/phase_3_region_wis.csv
runs/phase_3_region_eval_extras.csv
runs/phase_3_region_wis_extras.csv
runs/phase_3_sarima_wis_region.json
runs/phase_3_sarima_zeroshot_region.json
runs/phase_3_wilcoxon_region.json
runs/phase_3_conformal_region.csv
runs/e1_final/n3_d64_regional_perhorizon_raw.csv
runs/e1_final/e1_final_eval.json
runs/rolling_origin/cutoffs_manifest.json
runs/rolling_origin/verdict.json
runs/rolling_origin/stage1_sanity_verdict.json
runs/track_b_full/summary.json
runs/track_b_full/per_cell.parquet
runs/ablation_retrain/ablation_retrain_results.json
runs/ablation_retrain/bootstrap_ci.json
runs/wis_method_f/wis_results.json
runs/wis_method_f/decomposition_temporal.csv
runs/apmd_diagnostic/apmd_diagnostic_result.json
runs/apmd_diagnostic/PRE_REGISTRATION.md
runs/k_selection_kappa_ari.json
runs/interpretability/main_analysis_locked.json
runs/phase_5_flusight/cgm_retro_fair_summary.json
runs/phase_d/wilcoxon_results.json
runs/regime_shift/per_origin_forecasts.parquet
runs/cold_start/per_cell.parquet
data/PROVENANCE.json
data/processed/split_boundaries.json
data/processed/normalization_params.json
data/processed/ili_env_weekly_MANIFEST.json
data/geo/us_states.geojson
E1_design_split_pre_registration.md
CGM_v2_paper/PREREG_decision_value.md
CGM_v2_paper/PREREG_provisioning_value.md
EOF
```

Any line printed as `MISSING` means the artifact isn't on disk under that path — regenerate it (§ README
Reproduction) or correct the path before publishing. A few parquet files (`per_origin_forecasts.parquet`,
`per_cell.parquet`) may exceed 50 MB; if so, move them to Zenodo alongside the checkpoints and drop them
from this list.

## 4. Reproduction

See `README.md` → **Reproduction**. Every step there was checked to reference a script that exists on
disk (verified 2026-08-03). The paper-artifact → source-map table in the README lists which script and
which canonical file back each table/figure.

## 5. Fix before pushing (do NOT auto-apply — author review)

- **`scripts/p3_table_i_verification.py`** reads the stale `latex_submission/` tree; repoint it to
  `CGM_v2_paper/` (tolerance 0.002) or drop it from the release.
- **Manuscript "promise #13" (condition-number analysis / pre-safeguard failure characterization):**
  there is no dedicated script or small artifact for it. The numerical/κ guard lives in
  `scripts/m1_8_stage3_train.py` + root `kappa_recheck.py`. Either (a) ship a small condition-number
  sweep script, or (b) soften the manuscript sentence to describe the guard that exists. **This is a
  paper edit — decide before submission; it is not applied here.**
- **Root one-off scripts** `pc0_measurement.py`, `pc2_a_measurement.py` are unclassified; keep or remove
  at your discretion. `regime_shift_experiment.py`, `regime_shift_drivers.py`, `kappa_recheck.py` are
  kept (referenced by the reproduction path / numerical guard).

## 6. Create and push

```bash
cd /A.I_DATA/jbnu/JeongHa/CG_Mamba
git init && git add . && git checkout -b main   # .gitignore handles the excludes
# then run the force-add block in §3 for the canonical artifacts
git commit -m "CG-Mamba: reproduction code for IEEE JBHI submission"
# create an EMPTY GitHub repo, then:
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

## 7. After the URL exists — fill 3 placeholders in the paper

- `CGM_v2_paper/master.tex` line ~60: `\thanks{This work was supported by [FUNDING TODO]...}` → funding text (advisor).
- `CGM_v2_paper/master.tex` line ~63: `\thanks{Source code: [CODE-REPO TODO].}` → the GitHub URL.
- `CGM_v2_paper/tex/statements.tex`: `available at \texttt{[GITHUB\_URL]}` → the same URL (+ Zenodo DOI on acceptance).
