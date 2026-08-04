# Pre-Registration — Decision-Value of CG-Mamba's Native Calibration (Cost-Loss / Threshold-Crossing)

**Status:** LOCKED on 2026-08-02, before any decision-value computation.
**Rule:** Everything below is fixed prior to running. No metric, threshold, cost-loss ratio, stratum, or verdict criterion may be added or changed after seeing results (per the project honesty discipline: no self-serving post-hoc selection). If the design proves infeasible, we abandon the experiment rather than re-specify it.

---

## 1. Motivation and research question

The paper's contribution is that CG-Mamba attains the closest-to-nominal **native** interval calibration among amortized-deployable forecasters. Calibration is desirable because reliable predictive probabilities yield better cost-sensitive decisions (Richardson 2000). We test whether this calibration advantage **translates into operational decision value** for a concrete public-health task.

**Primary question (confirmatory).** Among models deployable as a single amortized instance (no per-location refitting, no recalibration set), does CG-Mamba's native predictive distribution yield **higher relative economic value** for a threshold-crossing decision than the deep-learning baselines' native uncertainty?

**Directional hypothesis.** CG-Mamba's value-score curve dominates the best deployable DL baseline over the pre-specified cost-loss range. (We do NOT hypothesize superiority over per-series SARIMAX; see §8.)

## 2. Prior work and differentiation (verified 2026-08-02)

- **Cost-loss / relative economic value framework** — Murphy (1977, classical origin; cite to be finalized), **Richardson (2000)** [QJRMS 126:649–667, verified], Wilks (2001, to verify). Standard binary-event value framework parameterized by the cost-loss ratio C/L.
- **Methodological authority = peer-reviewed.** The value framework rests on **Richardson (2000)** [QJRMS, peer-reviewed, verified] and the classical cost-loss origins (Murphy 1977; to verify) --- NOT on any preprint.
- **Concurrent preprint (differentiation only, not authority)** — **Gerlee, Lundh, Saxne Jöud & Thorén (2026)** [arXiv 2601.05921; still preprint as of 2026-08, no journal version] applies a cost-loss Value Score to FluSight influenza peak-intensity forecasts. **It does not evaluate calibration or deep-learning/amortized models.** We cite it as concurrent related work (to show awareness and differentiation), clearly flagged as a preprint; our claims do not depend on it.
- **Our differentiation.** Gerlee et al. ask *does forecasting beat climatology* on FluSight peak intensity. We instead use the same cost-loss/value lens to ask *whether the native-calibration advantage of one amortized DL model beats deployable DL alternatives on a threshold decision* — a model-comparison, calibration-value question. We cite Gerlee et al. as the framework precedent and scope our claim narrowly.

## 3. Data and models (LOCKED)

- **Evaluation window:** `test_strict` (W40-2022 – W35-2025, 152 weeks post-COVID), the paper's primary window.
- **Scale:** 10 HHS regions (primary) and national (secondary), raw wILI.
- **Predictive distributions:** each model's **NATIVE** predictive law, from the **same source as Table I** (`runs/wis_phase_b/<model>/` per-model native quantile forecasts / native per-cell). **NOT `forecast_quantiles_all.parquet`** (verified to use a non-native UQ for baselines — the previously-found consistency trap). CG-Mamba = raw native APMD.
- **Models — primary comparison (amortized-deployable):** CG-Mamba vs the 5 DL baselines used in the regional panel (LSTM, Vanilla Mamba, PatchTST, DLinear-ensemble, EpiDeep), each under its native UQ.
- **Reference (not in the primary claim):** per-series SARIMAX and amortized SARIMAX. Persistence = floor.

## 4. Decision task (LOCKED)

At each forecast origin and horizon h ∈ {1,2,3,4}, a decision-maker chooses whether to **prepare** (cost C) against wILI exceeding an activity threshold τ at the target week. If they do not prepare and wILI exceeds τ, they incur loss L (L > C).
- The model supplies **p = P(y_{target} > τ)** from its native predictive distribution.
- Optimal cost-loss action: **prepare iff p > C/L** (the standard cost-loss rule).
- Reference forecast = climatology: the training-period (2001–2018) empirical crossing frequency for that (region, week-of-season).

## 5. Threshold τ (LOCKED — externally defined, no post-hoc choice)

To avoid self-serving threshold selection, τ is fixed to **region-specific percentiles of the 2001–2018 training-period wILI distribution**, reported at **two pre-specified levels: the 85th and 95th percentiles** (both reported; neither cherry-picked). Rationale: reproducible, external to the test data, avoids CDC-baseline vintage issues. (A CDC-published-baseline variant may be added only as a clearly-labeled secondary robustness check, never replacing the locked primary.)

## 6. Metric (LOCKED)

- **Primary:** Relative Economic Value V(C/L) (Richardson 2000):
  V = (E_clim − E_forecast) / (E_clim − E_perfect), computed at **every** cost-loss ratio C/L ∈ (0,1) on a fixed grid (the full value curve — no single ratio is selected). V=1 perfect, V=0 = climatology, V<0 = worse than climatology.
  - Reported summaries: max_V, the C/L range where V>0, and the value curve itself, per (model, region, horizon, τ-level).
- **Secondary (descriptive):** threshold-crossing sensitivity, specificity, and detection lead-time at the cost-loss-optimal operating point, per model.
- **Uncertainty:** 5-seed spread for the DL/CG models (as elsewhere in the paper); no new significance test is introduced beyond the paper's existing conventions.

## 7. Stratification (LOCKED — origin-side only)

- Overall (all origins).
- **By origin-side intensity regime** (the canonical-ordered frozen-HMM phase at the forecast origin — low vs high emission-variance state, per `feedback_stratification_labelswitch_audit`). Outcome-conditional (observed-y) stratification is **prohibited** for any value claim (may appear only as an explicitly-labeled operational side-view). This honestly exposes peak-regime behavior, where CG mildly under-covers.

## 8. Pre-registered verdict (LOCKED kill/keep)

Let Δ = CG-Mamba's V minus the **best** deployable DL baseline's V, evaluated per (region, horizon, τ-level) over the C/L grid.

- **KEEP** (claim decision-value, scoped to amortized-deployable): CG-Mamba's value curve is ≥ the best deployable DL baseline in **≥ 70% of (region, horizon, τ-level) cells** across the C/L range where any model has V>0, **and** CG's pooled max_V ≥ the best-DL pooled max_V. Margin must exceed a pre-declared noise floor = the 5-seed std of V.
- **PARTIAL** (report as suggestive, no headline claim): 50–70% of cells.
- **KILL** (no decision-value claim; report as a limitation): < 50% of cells, or CG worse at high-intensity regime specifically.
- **SARIMAX:** reported as a reference curve. If per-series SARIMAX exceeds CG, the claim remains scoped to "among amortized-deployable models" — this is expected and is not a failure.

## 9. Honesty commitments (LOCKED)

1. Locked before any computation; this file is the record.
2. **No selective / self-serving reporting.** A KILL outcome means we simply **omit** the decision-value section --- a null explored internally carries no obligation to publish, and the paper stands on its calibration contribution. The binding rules are: (i) **no re-specification** of this design after seeing results (that is the p-hacking the lock prevents); (ii) **if** a decision-value result is included, the **full** pre-registered analysis is reported **without cherry-picking** cells/thresholds/ratios; (iii) we **never claim** positive decision-value when the pre-registered verdict is KILL/PARTIAL.
3. No post-hoc addition/removal of thresholds, ratios, metrics, or strata.
4. **Peak / high-intensity regime behavior reported explicitly** (this is where CG's disclosed under-coverage lives; it is the decision-critical regime).
5. Claim scope = amortized-deployable only; per-series SARIMAX comparison reported honestly.
6. Native-UQ source consistent with Table I (§3); any inconsistency halts the experiment.

## 10. What we will NOT claim
- Not superiority over per-series SARIMAX on decision value.
- Not a measured clinical/patient-outcome benefit (this is a retrospective decision-simulation on surveillance data, not a prospective clinical study).
- Not that native intervals are "usable as decision thresholds" in general — only the specific, cost-loss-scoped result that survives §8.

## 11. Feasibility notes (no new training)
Reuses saved native predictive distributions (no retraining, no GPU). Requires only: (a) confirm the native quantiles exist per model in the Table-I source; (b) build τ from training wILI percentiles; (c) compute P(cross), V(C/L), and the verdict. If the native predictive distributions for baselines are not recoverable consistently with Table I, the experiment is abandoned (not run on an inconsistent source).

---

## Citations to add on write-up (verification status)
- Richardson (2000), QJRMS 126(563):649–667, DOI 10.1002/qj.49712656313 — **verified (Wiley, ADS 2000QJRMS.126..649R)**.
- Gerlee, Lundh, Saxne Jöud, Thorén (2026), arXiv 2601.05921 — **verified (arXiv abstract, 2026-01-09)**.
- Murphy (1977), Wilks (2001) — classical cost-loss origins; **verify exact cite before adding**.
- Existing in bib: reich2019flusight, bracher2021wis, gneiting2007scoring, lutz2019applying.

---

## RESULTS (executed 2026-08-02, after lock; no re-specification)

**Consistency gate (§11): PASS.** Dumped native predictives reproduce the paper's regional native Cov95 within ≤0.001 (dlinear +0.010, analytic-vs-100-sample Gaussian; explained). CG per-horizon 0.998→0.970→0.938→0.907 matches the paper's disclosed drift and the e1_final canonical. (A subagent initially mis-compared CG against the val-calibrated `method_f` variant, 0.930; corrected against the paper's raw-native headline 0.954.)

**Verdict: KILL.** CG-Mamba's native calibration does NOT translate into superior threshold-crossing decision value.
- Mean area-under-V (Relative Economic Value, 80 cells): cg 0.514, lstm 0.567, patchtst 0.550, vanilla 0.519, epideep 0.454, dlinear 0.330. CG is mid-pack (below lstm/patchtst/vanilla; clearly beats only dlinear).
- CG ≥ strongest single DL (lstm): **31% of cells** (KEEP required 70%).
- High-intensity origin regime (the decision-critical, surge regime): CG **worse** (29%) — the §8 explicit KILL condition.
- Robust: same KILL under (a) the CG-favorable single-DL operationalization (vs the harsher envelope 2%), (b) the CG-favorable "prepare iff p>α" rule (rewards calibration), (c) Gaussianizing all predictives (31%), (d) both τ levels.

**Mechanism (honest):** for a binary threshold action, sharper-but-under-covering predictives make more decisive correct decisions; CG's calibrated-but-wider intervals give less decisive P(y>τ). Calibration improves interval scoring (WIS/coverage), not binary decision value, in this setting.

**Action per §9:** OMIT the decision-value section; make no decision-value claim. The paper stands on its calibration contribution (unchanged, 14p). This retroactively validates removing "usable as decision thresholds" from §V-A earlier. No sub-region (e.g., low-α surge-averse) is carved out as a claim — that would be the self-serving move the lock forbids. No tuning was applied; the design was executed as locked and the result went against the hypothesis.
