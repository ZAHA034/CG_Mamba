# CG-Mamba: A Selective State-Space Model with Native, Recalibration-Free Prediction Intervals for Influenza-Like Illness Forecasting

Reference implementation and reproduction code for the paper

> **CG-Mamba: A Selective State-Space Model with Native, Recalibration-Free Prediction Intervals for Influenza-Like Illness Forecasting**
> JeongHa Park and Jaehyuk Cho (corresponding), Jeonbuk National University.
> Submitted to *IEEE Journal of Biomedical and Health Informatics* (2026).

CG-Mamba is a compact (~115K trainable-parameter) selective state-space (Mamba) forecaster whose
selectivity is gated by a **frozen 3-state Gaussian-HMM epidemic-phase posterior**, and which emits
**native analytic prediction intervals** via **Analytic Phase-Mixture Decomposition (APMD)** — no
post-hoc recalibration step. It is evaluated on U.S. CDC ILINet weighted ILI (wILI), nationally and
across the 10 HHS regions under zero-shot national-to-regional transfer.

**Scope (honest).** CG-Mamba is *not* the most accurate or sharpest forecaster: a per-series classical
SARIMAX leads on point accuracy (MAE 0.347 vs 0.397) and aggregate interval score. Its contribution is
**the closest-to-nominal *native* 95% interval coverage among the deep-learning forecasters evaluated**
(regional Cov95 0.954; national 0.993, over-covering in the safe direction), at DL-best national MAE
(0.397), with peak-regime and long-horizon under-coverage disclosed in the paper. We position CG-Mamba
as a *complement* to classical models, not a replacement. See the paper's Table I / Section IV for the
full results — this README does not restate the tables to avoid drift.

> **Naming note:** the APMD interval is implemented in code under its historical name **"Method F"**
> (`src/eval/hmm_interval.py`, `scripts/wis_method_f.py`). APMD (paper) = Method F (code).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies (see `requirements.txt`): `numpy`, `pandas`, `torch>=2.0`, `scikit-learn`, `einops`,
`epiweeks`, `requests`, `pytest`. The selective scan uses the `mamba-ssm` CUDA kernel when available and
falls back to a pure-PyTorch implementation (`src/models/ssm_scan.py`) otherwise, so a GPU is recommended
but not required. Tests: `pytest src/tests`.

## Repository layout

```
src/
  models/      CG-Mamba: backbone, context_gate (ContextGatedMambaBlock), phase_module (frozen HMM),
               env_module, entropy_decoder, heteroscedastic_head, gaussian_hmm, ssm_scan (CPU fallback)
  data/        Public-data pipeline: download_cdc_fluview, download_noaa_isd, parse_isd,
               build_env_weekly, build_merged_weekly, build_splits, validate_dataset, loader
  eval/        wis / wis_standard (WIS + Cov95), hmm_interval (APMD / "Method F"), reliability, bootstrap
  baselines/   persistence, sarima, lstm, dlinear, nbeats, timesnet, patchtst, itransformer,
               vanilla_mamba, epideep (all from-scratch, faithful)
  utils/       config (CGMambaConfig), losses (MSE + 0.3*MASE), optimizer, scheduler, checkpoints
  tests/       pytest unit tests
scripts/       reproduction drivers (training, evaluation, tables, figures) — see Reproduction below
data/          pipeline manifests + provenance (raw/processed bulk is git-ignored; PROVENANCE.json,
               MANIFEST.json, split_boundaries.json, normalization_params.json are tracked)
runs/          small canonical result artifacts backing the paper's tables/figures are tracked;
               model checkpoints and bulk outputs are git-ignored (see "Large artifacts")
config/        credentials.template.json (optional API tokens; real credentials git-ignored)
```

Pre-registration records included in the release:
`E1_design_split_pre_registration.md`, `CGM_v2_paper/PREREG_decision_value.md`,
`CGM_v2_paper/PREREG_provisioning_value.md`, `runs/rolling_origin/cutoffs_manifest.json`,
`runs/apmd_diagnostic/PRE_REGISTRATION.md`, `runs/interpretability/*_locked.json`.

## Data

All data are public. The pipeline downloads and rebuilds everything from source; only provenance and
manifests are tracked in git (raw/processed data are large and git-ignored).

- **CDC FluView** national + regional weekly wILI, via the CMU Delphi Epidata API (`download_cdc_fluview`).
- **NOAA NCEI ISD** hourly temperature/dew-point for 10 US-MSA stations → population-weighted national
  weekly humidity/temperature (`download_noaa_isd` → `parse_isd` → `build_env_weekly`).

```bash
python -m src.data.download_cdc_fluview
python -m src.data.download_noaa_isd
python -m src.data.parse_isd
python -m src.data.build_env_weekly
python -m src.data.build_merged_weekly
python -m src.data.build_splits          # train/val/test/covid_excluded + train-only StandardScaler
python -m src.data.validate_dataset      # 9 integrity checks (MMWR alignment, checksums vs manifest)
```

## Reproduction

The headline pipeline (steps 1–3) must run before the table/figure scripts. Each stage writes to `runs/`.

```bash
# 1. CG-Mamba training pipeline (env pretrain -> frozen K=3 HMM -> Stage-2 SSM -> Stage-3 joint fine-tune -> eval)
python scripts/m1_7_env_pretrain.py
python scripts/m1_4_phase_dynamics_main.py
python scripts/e1_final_train.py
python scripts/m1_8_stage3_train.py
python scripts/e1_final_eval.py

# 2. National baselines (5 seeds each)
for b in dlinear lstm vanilla_mamba patchtst nbeats timesnet itransformer epideep sarima; do
  python scripts/run_${b}_weekly.py
done

# 3. Table I  (national MAE + WIS/Cov95)
python scripts/m2_4_eval_test_strict.py
python scripts/m2_3_eval_extra_baselines.py
python scripts/m2_4_wis_test_strict.py
python scripts/m2_4_render_tables.py
python scripts/p3_table_i_verification.py         # row-by-row re-derivation vs the .tex

# 4. Section IV-classical  (SARIMAX: per-series-refit + amortized zero-shot)
python scripts/phase_3_sarima_region.py
python scripts/phase_3_sarima_wis_region.py
python scripts/phase_3_sarima_zeroshot_region.py

# 5. Fig 3  (zero-shot regional Cov95 maps + Holm-Bonferroni significance)
python scripts/phase_3_region_eval.py
python scripts/phase_3_region_wis.py
python scripts/phase_3_wilcoxon_region.py

# 6. Table II  (pre-registered rolling-origin robustness, 7 expanding-window origins)
python scripts/build_rolling_splits.py            # writes runs/rolling_origin/cutoffs_manifest.json
python scripts/rolling_origin_stage1_sanity.py    # bit-identical reproduction gate
python scripts/run_rolling_origin.py
python scripts/run_rolling_origin_baselines.py
python scripts/verdict_rolling_origin.py

# 7. Table III  (uniform split-conformal CQR recalibration)
python scripts/p3_full_track_b_run.py
python scripts/p3_integration_test.py

# 8. Table IV  (from-scratch component ablation + paired-bootstrap CIs)
python scripts/ablation_retrain.py
python scripts/ablation_retrain_eval.py
python scripts/ablation_retrain_bootstrap_ci.py
python scripts/render_ablation_table_v2.py

# 9. Section IV-6  (APMD decomposition + variance-consistency validation)
python scripts/wis_method_f.py
python scripts/apmd_variance_diagnostic.py

# 10. Section IV-5  (data-efficiency sweep, 3..17 seasons)
python scripts/m2_4_cg_mamba.py
python scripts/m2_4_nn_baselines.py
python scripts/m2_4_efficiency_figure.py

# 11. Pre-registered interpretability NULL analysis (sigma^2_between)
python scripts/p5_interpretability_extract.py
python scripts/p5_step4_main.py
```

The canonical NATIONAL 148-origin forecast parquet (used by the APMD diagnostics) is produced by the
top-level `regime_shift_experiment.py` / `regime_shift_drivers.py`.

## Paper artifact → source map

| Paper item | Reproduction script(s) | Canonical artifact |
|---|---|---|
| Table I (national) | `m2_4_eval_test_strict.py`, `m2_4_wis_test_strict.py` | `runs/m2_1_final_topk/master_summary.json`, `runs/master_3column_wis_table.csv` |
| §IV-classical SARIMAX | `run_sarima_weekly.py`, `phase_3_sarima_*` | `runs/baselines/sarima.json`, `runs/phase_3_sarima_wis_region.json`, `runs/phase_3_sarima_zeroshot_region.json` |
| Fig 3 (regional maps) | `phase_3_region_eval.py`, `phase_3_wilcoxon_region.py` | `runs/e1_final/n3_d64_regional_perhorizon_raw.csv`, `runs/phase_3_wilcoxon_region.json` |
| Table II (rolling-origin) | `build_rolling_splits.py`, `run_rolling_origin*.py`, `verdict_rolling_origin.py` | `runs/rolling_origin/cutoffs_manifest.json`, `runs/rolling_origin/verdict.json` |
| Table III (conformal) | `p3_full_track_b_run.py` | `runs/track_b_full/summary.json`, `runs/track_b_full/per_cell.parquet` |
| Table IV (ablation) | `ablation_retrain*.py` | `runs/ablation_retrain/ablation_retrain_results.json`, `bootstrap_ci.json` |
| §IV-6 APMD | `wis_method_f.py`, `apmd_variance_diagnostic.py` | `runs/wis_method_f/decomposition_temporal.csv`, `runs/apmd_diagnostic/apmd_diagnostic_result.json` |
| §IV-5 data efficiency | `m2_4_cg_mamba.py`, `m2_4_nn_baselines.py` | `runs/m2_4_data_efficiency/m2_4_summary.csv` |
| K=3 selection | `m1_4_phase_dynamics_search.py` | `runs/k_selection_kappa_ari.json` |
| Interpretability NULL | `p5_interpretability_extract.py`, `p5_step4_main.py` | `runs/interpretability/main_analysis_locked.json` |

## Large artifacts

Model checkpoints (`runs/**/*.pt`, ~4 GB; some files exceed GitHub's 100 MB cap) are **not** in git.
They are archived on Zenodo (DOI added on acceptance); the small canonical result artifacts above are
tracked so the paper's numbers can be inspected without re-running training.

## Citation

```bibtex
@article{park2026cgmamba,
  author  = {Park, JeongHa and Cho, Jaehyuk},
  title   = {{CG-Mamba}: A Selective State-Space Model with Native, Recalibration-Free Prediction Intervals
             for Influenza-Like Illness Forecasting},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

## License

[MIT](LICENSE) © 2026 JeongHa Park, Jeonbuk National University.
