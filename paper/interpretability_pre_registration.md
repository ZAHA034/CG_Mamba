# Interpretability ($\sigma^2_{\text{between}}$) Sub-Pre-Registration (LOCKED v2.1, append-only)

**Status:** LOCKED v2.2 FROZEN 2026-06-27 (final pre-analysis lock; replaces v2.1 same-date draft after independent 6-lens adversarial verify and 1 micro-fix integration; no further in-place edits to §1-13 — change control §14 append-only)
**Parents:**
- `paper/track_b_sub_pre_registration.md` (LOCKED 2026-06-21)
- `paper/cold_start_pre_registration.md` (LOCKED 2026-06-23, with append §11)
- `project_cgmamba_pc012_locked` (CG-Mamba PC0/PC1/PC2 v2 LOCKED, 2026-06-12)

**Relationship:** APPEND-ONLY EXTENSION of parent locks. Does NOT replace, supersede, weaken, or reframe any parent lock. Pre-registers ONE additional analysis ($\sigma^2_{\text{between}}$ as candidate epidemic-phase-ambiguity signal) on EXISTING Track B per-cell APMD outputs (no new training, no new forward passes).

**As-is rule (user-confirmed 2026-06-27):** Results reported exactly as computed under the locked pipeline. STRONG → paper §12 framing template; MARGINAL → Discussion only, no headline; FORBIDDEN → drop from paper headline with §9 one-line disclosure. No post-hoc redefinition, no salvage reframing.

**v2.1 → v2.2 changelog (1 micro-fix from independent 6-lens adversarial verify):**
- **Regression engine version + RNG seeds + §8 (g) separation detection pinned**: workflow lens identified that `statsmodels` version, Newton tolerance, RNG seeds, and §8 (g) separation-detection mechanism remained unpinned. §13 appended with 5 rows; §8 (g) rewritten with explicit trigger condition. All other constants and §1-12 unchanged.

**v2 → v2.1 changelog (3 critical pre-freeze fixes integrated, all from independent statistical-rigor review):**
- **BIG fix (in-sample bias)**: ΔR² is structurally non-negative when M1 nests M0; ΔAUC has small-sample optimistic bias. v2's "block-bootstrap CI excludes zero" alone is a false-positive machine on this geometry. v2.1 makes the **block-conditional permutation null** the *primary significance test* (in-sample bias cancels symmetrically under the null), with block-bootstrap BCa CI demoted to *secondary effect-size descriptive*. See §5.2, §2 (a)+(b).
- **Regression engine freeze**: AUC implementation alone (sklearn) does not pin the M0/M1 *fit*; statsmodels MLE (unpenalized) vs sklearn LogisticRegression (L2 default) yield different coefficients → different ΔAUC. v2.1 pins **statsmodels.Logit (unpenalized, Newton solver, max_iter=100)** and **statsmodels.OLS (unpenalized)**. See §13.
- **log1p conservativity note**: log1p compresses the very transition-week spikes we are trying to detect → null-biased. We pre-disclose: a FORBIDDEN outcome under log1p does NOT entitle a "no signal" claim. See §13 (with explicit interpretation guard).

**v1 → v2 changelog (kept for traceability, 4 user catches from outline review):**
- Fix #1: H1 disambiguated to nested logistic (M0: σ²_total only; M1: σ²_total + σ²_between; ΔAUC = AUC(M1) − AUC(M0)). ρ_between demoted to secondary descriptive.
- Fix #2: STRONG reachability restored via seed-pooled primary + 5/5 sign-consistency robustness gate (replaced 4/5 independent Bonferroni).
- Fix #3: H2 target single-locked to h=1 |y − μ|; §7 extended with symmetric H2 wILI confound control.
- Fix #4: §2 (a) discloses H1-onset/peak as positive-class-rare (under-powered); H1-turning identified as primary-powered subtest. m=4 retained.

---

## 1. SCOPE (frozen)

**In scope.** Evaluate whether the analytically derived $\sigma^2_{\text{between}}$ component of APMD carries information that is (i) specifically elevated at pre-defined epidemic transition points incremental beyond $\sigma^2_{\text{total}}$ AND (ii) improves prediction of transition-associated h=1 absolute error beyond a $\sigma^2_{\text{total}}$-only baseline.

**Out of scope.** Clinical decision-utility, threshold-based real-time triggers, cross-domain demonstration, re-training/new-forward.

---

## 2. PRE-REGISTERED HYPOTHESES (✅/⚠/❌, frozen BEFORE seeing results)

### Primary family ($m = 4$, Bonferroni $\alpha_{\text{Bonf}} = 0.0125$)

For each H1 subtest, operationalization is a **nested logistic regression** (Fix #1):
- **M0 (baseline)**: $\mathrm{logit}(\text{transition}_{\,t,r,s}) = \beta_0 + \beta_1 \cdot \mathrm{log1p}(\sigma^2_{\text{total},\,t,r,s})$
- **M1 (extended)**: $\mathrm{logit}(\text{transition}_{\,t,r,s}) = \beta_0 + \beta_1 \cdot \mathrm{log1p}(\sigma^2_{\text{total},\,t,r,s}) + \beta_2 \cdot \mathrm{log1p}(\sigma^2_{\text{between},\,t,r,s})$
- Test statistic: $\Delta\text{AUC} = \mathrm{AUC}(M1) - \mathrm{AUC}(M0)$, seed-pooled (§5).

The three H1 subtests differ only in the transition label (§3):
- **H1-onset**: positives = locked onset weeks. *Disclosure*: positives are season-rare ($\leq 30$ across 30 blocks); under-powered by design.
- **H1-peak**: positives = locked peak weeks. *Disclosure*: same under-power.
- **H1-turning**: positives = locked turning-point weeks. **Primary-powered subtest** (positives are O(10) per block).

The fourth primary test:
- **H2-transition-error**: nested **linear regression** of *transition-associated h=1 absolute error* on variance components:
  - M0: $|y - \mu_{\text{CGM}}|_{h=1,\,\text{transition-assoc}} = \beta_0 + \beta_1 \cdot \mathrm{log1p}(\sigma^2_{\text{total}})$
  - M1: same + $\beta_2 \cdot \mathrm{log1p}(\sigma^2_{\text{between}})$
  - Test statistic: $\Delta R^2 = R^2(M1) - R^2(M0)$.
  - "Transition-associated" = within $\pm 2$ weeks of any transition point in (onset ∪ peak ∪ turning).
  - Single error target — h=1 absolute error only (Fix #3).

### ✅ STRONG claim allowed IFF (ALL of):

- **(a)** At least one H1 subtest: seed-pooled observed $\Delta\text{AUC} \geq 0.05$, **block-conditional permutation $p_{\text{Bonf}} < 0.0125$** (BIG fix), AND block-bootstrap BCa 95% CI excludes zero (effect-size confirmation).
- **(b)** H2-transition-error: seed-pooled observed $\Delta R^2 \geq 0.05$, **block-conditional permutation $p_{\text{Bonf}} < 0.0125$**, AND block-bootstrap BCa 95% CI excludes zero.
- **(c)** **5/5 seed sign-consistency robustness**: in each of the 5 training seeds analyzed separately, the point estimates of $\Delta\text{AUC}$ (passing subtest) and $\Delta R^2$ are BOTH $\geq 0$.
- **(d)** **wILI-intensity-controlled effect** (Fix #3): both (a) and (b) remain $\geq 0.03$ after wILI-level covariate adjustment (§7), each with its own permutation $p_{\text{Bonf}} < 0.0125$.

### ⚠ MARGINAL (Discussion only, no headline):
- (a) and (b) effects in $[0.03, 0.05)$ with permutation $p_{\text{Bonf}} < 0.0125$; OR
- 4/5 sign-consistency; OR
- wILI-adjusted effect in $[0.03, 0.05)$.

### ❌ FORBIDDEN (DROP, §9 one-line disclosure):
- All four primary tests: $\Delta < 0.03$ AND permutation $p_{\text{Bonf}} \geq 0.0125$.
- $\leq 3/5$ seed sign-consistency.
- wILI-adjusted effects all below 0.03 (intensity-tracking artifact).

---

## 3. TRANSITION-POINT DEFINITION (frozen — algorithm, parameters, list)

Computed from **regional wILI series** (10 HHS regions), Track B's test-strict window (W40-2022 -- W35-2025).

### 3.1 Onset
For each (region, season), onset week = first week $t$ with wILI$(t) \geq$ CDC-published baseline$(s) + 1.0$ pp AND wILI$(t+1)$ also clears the baseline. If no such week, the (region, season) contributes 0 onset positives.

### 3.2 Peak
For each (region, season), peak week = $\arg\max_t \text{wILI}_{r,s}(t)$ within season window (W40 through W20 of the following year).

### 3.3 Turning points
Series smoothed via **Savitzky-Golay, window $w = 5$, polynomial order $p = 2$** (frozen). Turning-point weeks are first-derivative sign-change weeks.

### 3.4 Exhaustiveness hard-stop
Combined transition coverage $\leq 35\%$ per (region, season) block. Coverage $> 35\%$ in any block → STOP.

### 3.5 List freeze
Computed once from locked wILI series; saved to `runs/interpretability/transition_points_locked.json`. Subsequent re-computation must match byte-identically; mismatch → hard-stop.

---

## 4. ANALYSIS PIPELINE (frozen)

### 4.1 Inputs (reused, no re-forward)
- `runs/track_b_full/per_cell.parquet`: per (baseline, seed, region, h) cell with $\sigma^2_{\text{within}}, \sigma^2_{\text{between}}, \sigma^2_{\text{total}}, \mu_{\text{CGM}}, y$.
- Restrict to baseline `cg_mamba`. 5 seeds × 10 regions × 4 horizons × 149 evaluation weeks.

### 4.2 Primary statistic
$\Delta\text{AUC}$ / $\Delta R^2$ from nested M0/M1 fits (§2). The ratio $\rho_{\text{between}} = \sigma^2_{\text{between}}/(\sigma^2_{\text{total}}+\epsilon)$ is reported only as **secondary descriptive** (per-class mean/variance), never as a primary test statistic.

### 4.3 Aggregation
Primary tests at $h = 1$ only. $h \in \{2,3,4\}$ secondary descriptive, NOT in $m = 4$ family.

---

## 5. RESAMPLING UNIT + INFERENCE (frozen — BIG-fix + Fix #2 + autocorrelation)

Three independent threats addressed in this section:
1. **Temporal autocorrelation**: week-level pseudo-replication → block resampling on (season, region).
2. **Seed-level over-Bonferroni**: → seed-pooled primary + 5/5 sign-consistency robustness.
3. **In-sample bias of ΔR²/ΔAUC** (BIG fix added in v2.1): ΔR² is structurally $\geq 0$ when M1 nests M0 → noise σ²_between alone passes "ΔR² ≥ 0.05" if measured in-sample; ΔAUC has small-sample optimistic bias for the same reason. Bootstrap CI that uses in-sample ΔAUC inherits the bias. Resolved by **block-conditional permutation null as primary significance test**, with bootstrap CI demoted to secondary effect-size descriptive.

### 5.1 Block unit
A block is a **(season, region) tuple**. Evaluation window covers 3 seasons × 10 regions = **30 blocks**. All 5 seeds enter the procedures *together* (seed is an additional factor inside each block, not a resampling unit).

### 5.2 Primary inference: block-conditional permutation null + block bootstrap CI (BIG-fix)

The primary inference is a **block-conditional permutation null** combined with a **block-bootstrap CI** for effect-size descriptive purposes.

#### 5.2.1 PRIMARY: block-conditional permutation null (significance)

For each $b = 1, \ldots, B_{\text{perm}} = 1000$:
1. Within each (season, region) block, *randomly permute* the $\sigma^2_{\text{between}}$ time series across weeks. This breaks the alignment between $\sigma^2_{\text{between}}$ and transitions WITHIN block while preserving:
   - The temporal autocorrelation of $\sigma^2_{\text{between}}$ itself (permuting within block, not across blocks).
   - The temporal structure of all other variables ($\sigma^2_{\text{total}}$, wILI, transition labels, $y$, $\mu_{\text{CGM}}$).
2. Refit M0 and M1 on the permuted data.
3. Compute $\Delta\text{AUC}_{b}^{\text{perm}}$ (for H1-*) or $\Delta R^2_{b}^{\text{perm}}$ (for H2) on the SAME permuted data (in-sample, matching observed condition — so in-sample bias cancels symmetrically).

The observed statistic $\Delta\text{AUC}_{\text{obs}}$ (or $\Delta R^2_{\text{obs}}$) is computed on the unpermuted data using the same in-sample fit.

**Two-sided permutation p-value**: $p_{\text{perm}} = \frac{1 + \sum_{b=1}^{B_{\text{perm}}} \mathbb{1}[|\Delta_{b}^{\text{perm}}| \geq |\Delta_{\text{obs}}|]}{B_{\text{perm}} + 1}$.

**Bonferroni correction**: $p_{\text{Bonf}} = \min(1, p_{\text{perm}} \cdot 4)$.

**Why this is the right null**: ΔR² is $\geq 0$ for ANY nesting (including pure-noise σ²_between additions), so any "ΔR² > 0" test against zero is anti-conservative. The permutation null computes the distribution of ΔR² under "no real signal" but with the *same in-sample optimism*. Significance arises iff observed ΔR² exceeds what the in-sample-optimism-only null produces.

#### 5.2.2 SECONDARY: block bootstrap BCa CI (effect-size descriptive)

For each $b = 1, \ldots, B_{\text{boot}} = 1000$:
1. Sample 30 blocks WITH replacement. Refit M0/M1 on resampled in-bag data.
2. Compute in-sample $\Delta\text{AUC}_b^{\text{boot}}$ on the resampled set.

**BCa percentile interval** at $\alpha = 0.05$ (uncorrected) and $\alpha_{\text{Bonf}} = 0.0125$. The CI is reported alongside the permutation p-value to quantify effect-size uncertainty. The CI excluding zero is **NOT** the primary significance criterion (§2 (a)+(b) require both permutation $p_{\text{Bonf}} < 0.0125$ AND CI excluding zero).

### 5.3 Multi-seed robustness gate (Fix #2 — 5/5 sign-consistency)

Separately, for each of the 5 individual seeds:
- Fit M0 and M1 on that seed's data only (no block resampling).
- Record the sign of $\Delta\text{AUC}$ (or $\Delta R^2$) point estimate.
- **5/5 sign-consistency** = all 5 single-seed point estimates have the SAME sign as the seed-pooled point estimate.
- This is the §2 (c) gate — robustness, NOT independent significance.

### 5.4 Hard-stops on inference
- Block count $< 25$ → STOP.
- Any single block contributes $> 20\%$ of within-block variance → tier-downgrade (STRONG → MARGINAL).
- Permutation null with empty support (degenerate) → STOP.
- BCa percentile fails (extreme tail mass) → fall back to basic percentile and disclose.

### 5.5 Rationale (this resampling unit addresses ALL three threats simultaneously)
- Week-level pseudo-replication killed by block resampling on (season, region).
- Seed-level over-Bonferroni killed by seed-pooling.
- In-sample bias of ΔR²/ΔAUC killed by permutation null (BIG-fix).
- Robustness against seed-noise preserved by 5/5 sign-consistency (direction-only).

---

## 6. MULTIPLICITY ADJUSTMENT (frozen)

Primary family: $m = 4$ = {H1-onset, H1-peak, H1-turning, H2-transition-error}. Bonferroni $\alpha_{\text{Bonf}} = 0.0125$. Each primary test reports BOTH uncorrected and Bonferroni-corrected permutation $p$; Bonferroni-corrected is the headline. Holm-Bonferroni reported in supplementary.

Excluded from primary family (descriptive secondary, NOT multiplicity-adjusted): per-horizon decomposition ($h = 2, 3, 4$), per-region heterogeneity, per-season heterogeneity, univariate $\rho_{\text{between}}$ class-mean comparisons.

---

## 7. CONFOUNDER CONTROL — wILI intensity (frozen, both H1 and H2; Fix #3)

To rule out the "$\sigma^2_{\text{between}}$ is just tracking wILI level" challenge.

### 7.1 H1 wILI adjustment
- M0_adj: $\mathrm{logit}(\text{transition}) \sim \mathrm{log1p}(\sigma^2_{\text{total}}) + \mathrm{log1p}(\text{wILI level})$
- M1_adj: same + $\mathrm{log1p}(\sigma^2_{\text{between}})$
- Acceptance: $\Delta\text{AUC}_{\text{adj}} \geq 0.03$ AND block-conditional permutation $p_{\text{Bonf}} < 0.0125$ AND block-bootstrap BCa CI excludes zero (each H1 subtest separately).

### 7.2 H2 wILI adjustment
- M0_adj: $|y - \mu|_{h=1,\text{transition-assoc}} \sim \mathrm{log1p}(\sigma^2_{\text{total}}) + \mathrm{log1p}(\text{wILI level})$
- M1_adj: same + $\mathrm{log1p}(\sigma^2_{\text{between}})$
- Acceptance: $\Delta R^2_{\text{adj}} \geq 0.03$, same dual permutation+CI requirements.

### 7.3 Confounded-outcome rule
If ALL four primary adjusted effects drop below 0.03, FORBIDDEN downgrade (intensity-tracking artifact).

Note: the unadjusted §2 (a)+(b) results are PRIMARY; the wILI-adjusted (d) is an additional acceptance gate.

---

## 8. HARD-STOPS

- **(a)** Transition coverage $> 35\%$ per block → STOP (§3.4).
- **(b)** Block count $< 25$ → STOP (§5.4).
- **(c)** Single block contributes $> 20\%$ within-block variance → tier-downgrade.
- **(d)** $\leq 3/5$ seed sign-consistency → FORBIDDEN downgrade (§2 ❌).
- **(e)** NaN/inf in any test statistic → STOP, no silent filtering.
- **(f)** Locked transition-point JSON byte-mismatch → STOP (§3.5).
- **(g)** Logistic regression non-convergence on M1 — trigger condition: `statsmodels.tools.sm_exceptions.PerfectSeparationError` raised OR `mle_retvals['converged'] == False` after Newton solver with tol=$10^{-8}$ — flag, downgrade tier; if affects $\geq 2$ of 4 primary tests, STOP.
- **(h)** Permutation null degenerate (variance = 0 across $B_{\text{perm}}$) → STOP, debug (§5.4).

No methodology swap. No post-hoc redefinition of transition points. No re-tuning of Savitzky-Golay $(w, p)$. No removal of the wILI confounder gate. No expansion of multiplicity family. No promotion of secondary $\rho_{\text{between}}$ statistics. **No swap of permutation null for bootstrap CI as the primary significance test (BIG-fix lock).**

---

## 9. AS-IS REPORTING (verbatim, user-confirmed 2026-06-27)

> Results are reported exactly as computed under the locked pipeline. STRONG outcomes enter the paper as a *candidate phase-ambiguity signal* finding (§12 framing); MARGINAL outcomes are reported in Discussion only, no headline; FORBIDDEN outcomes are dropped from the paper headline with a one-line disclosure ("We investigated whether $\sigma^2_{\text{between}}$ carries information beyond $\sigma^2_{\text{total}}$ at epidemic transition points under a pre-registered nested-logistic protocol with block-conditional permutation null; the analysis did not meet our pre-specified significance criteria and is not pursued further in this paper"). The log1p input transform is conservative (it compresses transition-week spikes); a FORBIDDEN outcome under log1p is NOT a "no signal" claim — it is a "below-pre-registered-threshold under our locked operationalization" disclosure (§13 conservativity note). No post-hoc redefinition, no salvage reframing.

---

## 10. PRE-FLIGHT CHECKS (required BEFORE primary analysis)

- **(P1)** HMM phase-posterior sanity: $\gamma_{\text{all}}$ per (region, season) confirms the 3-state HMM separates seasonal regimes; uniform/degenerate posterior → STOP.
- **(P2)** Parquet column verification: $\sigma^2_{\text{within}}, \sigma^2_{\text{between}}, \sigma^2_{\text{total}}, \mu_{\text{CGM}}, y$ non-null across 5 seeds × 10 regions × 4 horizons.
- **(P3)** Transition-point exhaustiveness pre-check: coverage $\leq 35\%$ per block.
- **(P4)** Synthetic-null block-bootstrap power sanity: under a Gaussian-noise null on synthetic data matched in size + autocorrelation, BCa block-bootstrap CIs cover zero $\geq 93\%$ of the time.
- **(P5)** Smoothing reproducibility: re-derive turning points with locked $(w=5, p=2)$ Savitzky-Golay; byte-identical match to saved JSON.
- **(P6)** Logistic specification check: M0 and M1 formulas saved as a single locked Python statsmodels formula string; assert identical at run time.
- **(P7, BIG-fix calibration check)** Permutation-null calibration: under a synthetic null (random shuffle of $\sigma^2_{\text{between}}$ relative to outcome, no real signal), confirm permutation $p$-value distribution is approximately uniform on $[0, 1]$ over 100 synthetic-null simulations. Validates the permutation test is properly calibrated and not anti- or conservatively biased.

---

## 11. RISK ACKNOWLEDGMENT (explicit, LOCK v2.1 time)

User confirmed (2026-06-27) the following risks at LOCK v2.1 time:

1. **STRONG outcome probability** (re-estimated for v2.1 with permutation null as primary): ~15-30% STRONG, ~30-45% MARGINAL, ~30-45% FORBIDDEN. The permutation null is more conservative than the v2 bootstrap-CI primary; this is intentional (eliminates in-sample false positives).
2. **FORBIDDEN outcome is acceptable**: paper headline unaffected; native-calibration spine survives.
3. **STRONG outcome framing limit**: even on success, framing is *"candidate phase-ambiguity signal"* with clinical-utility validation as future work.
4. **Additive scope**: analysis is purely additive. STRONG/MARGINAL adds $\sim 1$ page; FORBIDDEN keeps the paper at 16 pages with a single-line disclosure.
5. **log1p conservativity**: log1p compresses transition-week σ²_between spikes; a FORBIDDEN outcome under log1p is null-biased and should NOT be over-interpreted as "no signal exists". Reported transparently (§9, §13).
6. **Regression engine is frozen** to statsmodels.Logit (unpenalized MLE, Newton, max_iter=100) and statsmodels.OLS (unpenalized). Reproducibility requires running with these exact estimators; sklearn LogisticRegression (L2 default) produces different coefficients.
7. **No post-freeze tuning**: §13 constants are immutable post-freeze except via §14 append-only blocks.

---

## 12. SUCCESS-CASE FRAMING (verbatim, locked BEFORE analysis)

If STRONG:

> "Beyond the native-calibration contribution, we observe that the analytically derived between-phase variance component $\sigma^2_{\text{between}}$ carries information not present in the total predictive variance $\sigma^2_{\text{total}}$ alone. Under a pre-registered nested-logistic-regression protocol (M0: $\mathrm{log1p}(\sigma^2_{\text{total}})$; M1: + $\mathrm{log1p}(\sigma^2_{\text{between}})$) with block-conditional permutation null as the primary significance test (in-sample bias controlled symmetrically) and seed-pooled block-bootstrap BCa CI for effect-size, $\sigma^2_{\text{between}}$ improves the AUC for discriminating epidemic transition weeks by $\Delta\text{AUC} = \{\text{value}\}$ over a $\sigma^2_{\text{total}}$-only baseline (permutation $p_{\text{Bonf}} = \{p\}$ across $m = 4$ family, BCa 95% CI $[\{lo\}, \{hi\}]$), with the effect remaining $\Delta\text{AUC} = \{\text{value\_adj}\}$ after controlling for regional wILI level and direction-consistent across all 5 training seeds. This positions $\sigma^2_{\text{between}}$ as a candidate phase-ambiguity signal; clinical-utility validation through forward-looking deployment is identified as priority follow-up (Section \ref{sec:v_e})."

No deviation from this template language without explicit append below §14.

---

## 13. PRE-REGISTERED CONSTANTS (frozen — immutable post-freeze)

| Key | Value | Rationale |
|---|---|---|
| primary_test_specification | Nested logistic (H1) / linear (H2); ΔAUC / ΔR² | Fix #1 |
| analysis_unit_of_resampling | block (season × region), seeds pooled | Fix #2 + autocorrelation |
| block_count_expected | 30 (3 seasons × 10 regions) | Track B test window |
| robustness_gate | 5/5 seed sign-consistency | Fix #2 |
| **primary_significance_test** | **block-conditional permutation null** ($B_{\text{perm}} = 1000$) | **BIG-fix: in-sample bias control** |
| **secondary_effect_size_test** | **block bootstrap BCa CI** ($B_{\text{boot}} = 1000$) | **effect-size descriptive (NOT primary significance)** |
| **logistic_estimator** | **`statsmodels.Logit`, unpenalized MLE, Newton solver, max_iter=100** | **regression engine freeze (v2.1)** |
| **linear_estimator** | **`statsmodels.OLS`, unpenalized** | **regression engine freeze (v2.1)** |
| **wILI_covariate_form** | **log1p(regional_per_week_wILI)** | heavy-tailed → log; identical §7.1/§7.2 |
| **variance_input_scaling** | **log1p(σ²_total), log1p(σ²_between)** | symmetric log; entering M0/M1 |
| **log1p_conservativity_note** | log1p compresses transition spikes → null-biased; FORBIDDEN under log1p ≠ "no signal" | §9, §11 interpretation guard |
| smoothing_filter | Savitzky-Golay | §3.3 |
| smoothing_window | $w = 5$ | frozen |
| smoothing_order | $p = 2$ | frozen |
| **cdc_baseline_vintage** | **CDC FluView baselines on or before 2025-09-01 (URL/SHA pinned in code repo)** | vintage freeze |
| **auc_implementation** | **`sklearn.metrics.roc_auc_score` (default tie handling)** | standard |
| **season_window** | **W40_year through W20_year+1** | CDC epidemic-week convention; used for onset/peak definitions |
| onset_threshold | CDC baseline + 1.0 pp | §3.1 |
| transition_coverage_limit | $\leq 35\%$ per (season, region) | §3.4 |
| primary_horizon | $h = 1$ | operational relevance |
| h2_error_target | $|y - \mu_{\text{CGM}}|_{h=1}$ (single) | Fix #3 |
| transition_associated_window | $\pm 2$ weeks of any transition point | H2 |
| baseline_for_increment | $\sigma^2_{\text{total}}$ only (M0) | Fix #1 |
| confound_covariate | $\mathrm{log1p}(\text{wILI level})$ regional per-week | §7 |
| h1_powered_subtest | H1-turning (others under-powered, disclosed) | Fix #4 |
| primary_family_size | $m = 4$ | H1-onset/peak/turning + H2 |
| alpha_uncorrected | 0.05 | |
| alpha_bonferroni | 0.0125 | $0.05 / 4$ |
| forbidden_threshold | $\Delta < 0.03$ AND permutation $p_{\text{Bonf}} \geq 0.0125$ | §2 ❌ |
| marginal_threshold | $0.03 \leq \Delta < 0.05$ OR 4/5 sign | §2 ⚠ |
| strong_threshold | $\Delta \geq 0.05$ + permutation $p_{\text{Bonf}} < 0.0125$ + BCa CI excludes 0 + 5/5 sign + wILI-adjusted $\geq 0.03$ | §2 ✅ |
| outputs | `runs/interpretability/{transition_points_locked.json, primary_results.json, summary.md}` | reproducibility |
| **statsmodels_version** | **>=0.14.0,<0.15** (exact pin via `requirements.txt` SHA in code repo) | regression engine version freeze (v2.2) |
| **logistic_solver_convergence** | **Newton solver, tol=$10^{-8}$, no method fallback** | convergence parameter freeze (v2.2) |
| **logistic_separation_detection** | `statsmodels.tools.sm_exceptions.PerfectSeparationError` raised OR `mle_retvals['converged'] == False` | explicit §8 (g) trigger mechanism (v2.2) |
| **permutation_rng_seed** | **20260627** | reproducible permutation null (v2.2) |
| **bootstrap_rng_seed** | **20260628** | reproducible bootstrap BCa CI (v2.2) |
| reframe_lock | No new hypotheses, no new transition defs, no new resampling units, no salvage, no swap of permutation for bootstrap as primary | as-is enforced |

---

## 14. CHANGE CONTROL

Identical to parent locks:
1. Explicit user confirmation referencing this file by path.
2. New dated append-only block below §14 (no in-place edits to §1-13).
3. Re-verification that no clause conflicts with parent locks.

End of LOCKED v2.2 FROZEN $\sigma^2_{\text{between}}$ interpretability sub-pre-registration (§1-14).

---

## 14.1 APPEND — Pre-analysis σ² extraction clarification (2026-06-27)

Pre-flight P2 verification (post-freeze) revealed that `runs/track_b_full/per_cell.parquet` stores only aggregate metrics (WIS, Cov95, MAE per region/h/seed) but NOT the per-cell σ²-decomposition components ($\sigma^2_{\text{within}}, \sigma^2_{\text{between}}, \sigma^2_{\text{total}}, \text{bias}^2, \mu_{\text{CGM}}$) needed by §4.1 of the primary analysis.

**Clarification of §1 out-of-scope "new forward passes":** a deterministic re-execution of the existing CG-Mamba forward (5 seeds × 10 regions × 149 weeks × 4 horizons) for the SOLE purpose of (i) extracting per-cell σ²-decomposition components computed by the existing `cgm_decomp_forward` code path (`scripts/track_b_lib.py`, the same function that produced Track B's WIS/Cov95/MAE) and (ii) saving them to `runs/interpretability/sigma_components.parquet`, with the SAME locked CGM checkpoints (`runs/m2_4_data_efficiency/.../seed{seed}/manifest.json` per Track B parent §10), SAME locked seeds {42, 123, 456, 789, 1024}, and SAME code paths used for Track B, is NOT a new forward pass in the LOCK §1 sense. The re-extraction produces byte-identical model predictions (already verified in Track B v4 integration test: |Δ|=0.0000 CGM bit-identical) and adds only saved variance-component columns. No re-training, no architectural modification, no measurement redefinition. Reproducibility is preserved (same seeds, same code, same checkpoints).

**Three-gate reproduction verification (ALL must pass before primary analysis):**

(i) **Aggregate gate**: per (seed, region, h), the re-extracted Cov95/WIS/MAE aggregates must match Track B's `per_cell.parquet` `native_cov95`/`native_wis`/`native_mae` columns within $|\Delta| < 10^{-6}$ (float64 numerical noise tolerance).

(ii) **Decomposition identity gate**: per (seed, region, h, week), the extracted components must satisfy $\sigma^2_{\text{total}} = \sigma^2_{\text{within}} + \sigma^2_{\text{between}}$ within $|\Delta| < 10^{-6}$ AND $\text{bias}^2 \geq 0$ for all cells (deterministic offset, separate non-negativity sanity). Per code design (`src/eval/hmm_interval.py:86-95`, mirroring method.tex eq. 6.3-6.4): bias² is excluded from σ²_total because it is a deterministic refinement offset, not a random uncertainty component; including it would conflate calibration semantics. This identity verifies the decomposition is self-consistent independent of any external reference.

(iii) **σ²_total → native_cov95 reproduction gate**: for each (seed, region, h), reconstruct a 95% prediction interval from the re-extracted ($\mu_{\text{CGM}}, \sigma^2_{\text{total}}$) per cell, compute the per-cell coverage of the held-out $y$, aggregate via the locked LOCK §5 order (per-cell → cross-region per-h-mean → h-mean → 5-seed mean), and verify the reproduced regional native Cov95 equals the locked value **0.9548** (Track B parent verified) within $|\Delta| < 10^{-6}$. This pins σ²_total to a frozen quantity and validates the entire decomposition pipeline.

**Any gate failure → STOP**, do not proceed to primary analysis; debug the re-extraction without proceeding to permutation testing.

The clarification has zero impact on §2 (hypotheses), §5 (inference), §13 (constants); it only makes explicit the data-plumbing path from existing forward to the σ² components required by the primary analysis. Anti-over-claim discipline is unchanged.

End of append §14.1.

---

## 14.2 APPEND — Gate (iii) target precision spec-bug correction (2026-06-27)

**Bug**: §14.1 (iii) wording cites target "0.9548" (4-digit rounded display value) with tolerance $|\Delta| < 10^{-6}$. These two are mathematically incompatible: a 4-digit rounded target carries $\pm 5 \times 10^{-5}$ implicit precision noise, which exceeds the stated $10^{-6}$ tolerance by 50× regardless of computational fidelity. The mismatch was present in the spec BEFORE re-extraction (it is a clerical transcription error, not a result-adjusted change).

**Independent proof of bit-identical reproduction (Gate i)**: Gate (i) verified that the re-extracted per (seed, region, h) aggregate Cov95 values match Track B's `per_cell.parquet` `native_cov95` column with max $|\Delta| = 0.00$ (exact float64 equality across all 200 cells). Reproducibility of the underlying $\sigma^2_{\text{total}}$ computation is therefore *independently established* by Gate (i), independent of Gate (iii).

**Correction**: §14.1 (iii) target is corrected from the literal display value "0.9548" to **the full-precision float64 value computed from `runs/track_b_full/per_cell.parquet`** under the `native_cov95` column, aggregated via the locked LOCK §5 order (per-cell → cross-region per (seed, h) → h-mean → 5-seed mean). The literal value (computed once at LOCK time and recorded here for audit traceability) is **0.9547651006711408**. The tolerance $|\Delta| < 10^{-6}$ is unchanged. The re-extraction script computes the target dynamically from the parquet at run time so that any future legitimate update of the parquet (under §14-style append-only protocol) keeps the gate self-consistent.

**Bright-line scope of this correction (not a precedent)**: this spec-bug correction is permitted only because it satisfies all four criteria:
- **(a)** the corrected clause is a *plumbing/reproduction gate* (Gate iii), not a §2 substantive hypothesis threshold;
- **(b)** the bit-identical reproduction it intends to verify is *independently established* by Gate (i) (max $|\Delta| = 0.00$);
- **(c)** the precision spec was *self-contradictory before* re-extraction (4-digit rounded target + $10^{-6}$ tolerance are mathematically incompatible regardless of computational outcome);
- **(d)** the corrected target value is *not adjusted to the observed result* — it is the full-precision form of the same locked Track B quantity originally intended.

If ANY future spec issue fails ANY of (a)–(d) — particularly any §2 substantive hypothesis threshold ($\Delta\text{AUC} \geq 0.05$, $\Delta R^2 \geq 0.05$, $\alpha_{\text{Bonf}} = 0.0125$, $5/5$ sign-consistency, §7 wILI-adjusted $\geq 0.03$ gate) — the answer is unconditionally STOP per §9 as-is rule. §2 substantive thresholds are immutable post-result; this §14.2 correction does NOT create a precedent for relaxation.

End of append §14.2.

---

## 14.3 APPEND — Provenance + scipy/savgol pin for STEP 2 (2026-06-27)

**Provenance (user-confirmed 2026-06-27)**: All wILI series used for transition-point computation MUST be derived from `runs/interpretability/sigma_components.parquet`'s `y_raw` column (or its z-scored counterpart, equivalently). No fresh CDC pull. Rationale: ILINet is subject to post-hoc revisions; using the same y the model was evaluated on is the only way to keep transition points and σ²_between on the same timeline.

- **peak / turning points**: computed directly from per-(region, week) y (z-space or raw, both scale-invariant for argmax / derivative sign).
- **onset**: requires raw-% comparison against CDC baseline. y_raw column already provides the inverse-transformed series.
- **External constants only**: CDC FluView baseline thresholds per season, vintage ≤ 2025-09-01, recorded in `runs/interpretability/cdc_baselines_locked.json` (with source URL + retrieval date).

**Additional §13 constants (P5 byte-identity)**:

| Key | Value | Rationale |
|---|---|---|
| **scipy_version** | **>=1.11,<1.14** (exact pin via requirements.txt SHA) | Savitzky-Golay implementation byte-identity (P5) |
| **savgol_mode** | **`mode='interp'`** (scipy default) | edge handling frozen for P5 reproducibility |

**P3 strict no-loosen**: if combined transition coverage > 35% per (region, season) block per §3.4, STOP per §8 (a); refine only by *tightening* (e.g., raising onset threshold from +1.0 pp to +1.5 pp, or restricting turning to higher-derivative-magnitude weeks), via §14-style append-only block. Improvising loosened thresholds is forbidden.

End of append §14.3.

---

## 14.4 APPEND — H1-onset NOT EVALUABLE (regional baseline unavailable; CDC prohibition on national-uniform) (2026-06-28)

**Trigger.** Per §3.1, `onset_week` requires CDC-published *regional* baselines per (region, season). Sourced-value sweep (rule: 추측 0 / 부분 fetch + 추측 혼합 금지) yielded:

- **2023-24**: 11/11 obtained (national 2.9% + R1–R10), byte-identical across 4 Wayback snapshots of CDC FluView Overview.
- **2022-23**: only national 2.5% (regional 10/10 missing).
- **2024-25**: only national 3.0% (regional 10/10 missing).

Exhausted sources (all dead): `cdcepi/FluSight-forecasts/wILI_Baseline.csv` (pre-pandemic only, stops at 2019-20); `cdcepi/Flusight-forecast-hub` (target switched to hospital admissions, no wILI baseline file); Wayback Machine 2022–2026 snapshots of `cdc.gov/fluview/overview/index.html` and `cdc.gov/flu/weekly/overview.htm` (no 2022-23 snapshot available; 2024-11-27 snapshot still showed 2023-24 baselines — CDC delayed page update); CDC weekly archives (`weeklyarchives2022-2023/Week40.htm` 404); Delphi Epidata `fluview_meta` endpoint (table metadata only, no baselines). User-side institutional/GUI access to FluView Interactive: confirmed unavailable (user decision B, 2026-06-28).

**CDC explicit prohibition** (FluView Overview, verbatim from 4 Wayback snapshots cross-checked):

> "Due to the wide variability in regional level data, it is not appropriate to apply the national baseline to regional data."

→ National-uniform substitution (national value applied to all 10 HHS regions) is **forbidden by CDC methodology**. No §3.1 deviation permissible for onset under this constraint.

**Decision.** **H1-onset = NOT EVALUABLE.** Recorded as the string `"not_evaluable"` in result artifacts (NOT a p-value, NOT a 1.0 placeholder).

**Multiplicity preserved (conservative; anti-self-serving).** Primary family size $m = 4$ RETAINED. Bonferroni $\alpha_{\text{Bonf}} = 0.0125$ unchanged. Self-serving multiplicity relaxation ($m{=}4 \to m{=}3$) is forbidden; the three evaluable primary tests {H1-peak, H1-turning, H2-transition-error} face the same Bonferroni threshold as if H1-onset had run. Dropping H1-onset from the family count would loosen $\alpha$ to $0.0167$ in our own favor and is therefore prohibited.

**Refinement #1 — STRONG verdict criterion (§2(a)) updated.** §2(a)'s "at least one of {H1-onset, H1-peak, H1-turning}" now reads "at least one of {H1-peak, H1-turning}". H1-onset cannot contribute to the STRONG verdict.

**Refinement #2 — P3 coverage gate basis updated.** §3.4 P3 coverage gate evaluated on (peak $\cup$ turning) ONLY (onset excluded from the union). The $\leq 35\%$ per (region, season) threshold itself is unchanged (non-loosening); the union shrinks correspondingly. (onset positives are $\leq 30$ across 1490 cells per Fix #4 disclosure — coverage shift expected negligible, neutral; not a multiplicity-effective relaxation since the threshold itself is fixed.)

**§3.4 H2 transition-associated window correspondingly updated**: window = (peak $\cup$ turning) $\pm 2$ weeks (onset removed from the union $\Rightarrow$ sample-size reduction $\Rightarrow$ power reduction = anti-self-serving).

**Supplementary record (NOT used in any primary test).** 2023-24 regional baselines (10/10) preserved in `runs/interpretability/cdc_baselines_sourced.json` with source URLs (Wayback snapshot URLs) + verbatim CDC quotes. This record exists for audit traceability; it is **not consulted** by STEP 2 transition-point computation or STEP 4 main analysis under this LOCK.

**§V-D paper disclosure (Methods + Limitations).**

- *Methods (Interpretability section)*: "Regional ILINet baselines for the 2022-23 and 2024-25 seasons were not available from CDC sources at the time of analysis. CDC methodology explicitly prohibits applying the national baseline to regional data. We therefore recorded H1-onset as not evaluable and retained the primary family size m=4 conservatively (Bonferroni α=0.0125)."
- *Limitations (§V-D)*: "H1-onset transition was not evaluated due to regional baseline unavailability for 2 of 3 evaluation seasons. The primary-powered subtest (H1-turning) remains evaluable; multiplicity family size m=4 retained conservatively to forbid self-serving α relaxation."

**Bright-line 4 criteria check** (per §14.2 audit protocol, applied to this §14.4 itself):

| Criterion | Status |
|---|---|
| (a) Plumbing/reproduction gate (not §2 substantive threshold) | ✓ — onset evaluability is a data-availability constraint, not a threshold tune. §2 thresholds ($\Delta\text{AUC} \geq 0.05$, $\Delta R^2 \geq 0.05$, $\alpha_{\text{Bonf}} = 0.0125$, 5/5 sign, §7 $\geq 0.03$) untouched. |
| (b) Independent proof (not result-adjusted) | ✓ — server fetched all CDC sources BEFORE any STEP 2 / STEP 4 results were computed. JSON record (`cdc_baselines_sourced.json`) dated 2026-06-28, stamped before main analysis. |
| (c) Pre-existing self-contradiction | ✓ — §3.1 (requires regional CDC baselines) + observed CDC prohibition on national-uniform = pre-existing wall, not a result-driven retreat. |
| (d) Not result-adjusted | ✓ — main analysis not yet run. Decision made strictly before STEP 4. |

End of append §14.4.
