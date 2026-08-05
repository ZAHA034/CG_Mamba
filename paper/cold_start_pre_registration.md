# Cold-Start Sub-Pre-Registration (LOCKED, append-only extension)

**Status:** LOCKED 2026-06-23
**Parent:** `paper/track_b_sub_pre_registration.md` (LOCKED 2026-06-21)
**Grandparent:** `project_cgmamba_pc012_locked` (CG-Mamba PC0/PC1/PC2 v2 LOCKED, 2026-06-12)
**Relationship:** APPEND-ONLY EXTENSION of Track B sub-pre-registration. Does NOT replace, supersede, weaken, or reframe Track B or PC0/PC1/PC2. Pre-registers ONE additional analysis (cold-start cal-data-scarcity sweep) over the existing Track B baseline set + Track B results (reused as stale-conformal reference). If any clause here conflicts with the parent locks, parent locks win and the conflicting clause is null.
**As-is rule (user-confirmed 2026-06-23):** Results reported exactly as computed. Success of cold-start does NOT mean CGM winning; success means *measurement-based decision* (JBHI 강행 / 하향 / pivot) regardless of outcome direction.

---

## 1. SCOPE (frozen)

Post-deployment cold-start UQ in epidemic time series forecasting:
- **In scope**: post-deployment, small-n_cal regime where calibration data accumulates progressively after a model is already trained and deployed.
- **Out of scope**: training-time data scarcity. Model retraining NOT in scope (CGM HMM fit + dual-mamba weights frozen at Track B values).

**JBHI clinical anchor**: weekly epidemic surveillance, calibration window 5-40 weeks (test_strict prefix); 30-week flu season → 17%-133% of season. n_cal ≥ 10 (~2.5 months) treated as "realistic deployment" anchor; n_cal < 5 treated as "unrealistically small" anchor.

---

## 2. PRE-REGISTERED CLAIMS (✅/❌ criteria, frozen BEFORE seeing results)

### Primary hypothesis (H)
n_cal 감소 → conformal-baseline calibration 불안정/악화. CGM raw APMD (n_cal-independent) 안정. → CGM relative advantage 확대 at small n_cal.

### ✅ STRONG claim allowed IFF (ALL of):
- **(a)** CGM-raw-APMD `|Cov95 − 0.95|` < `best fresh-conformal-baseline` `|Cov95 − 0.95|` at n_cal ∈ {5, 10, 20, 40}.
- **(b)** Crossover at n_cal ≥ 10 (i.e., CGM-raw beats best fresh-conformal at *realistic* deployment-relevant n_cal).
- **(c)** Fresh conformal baseline finite-sample std (B=20 random subsamples) blows up at small n_cal (evidence for instability).
- **(d)** CGM-raw-APMD better than **stale-conformal** reference (= Track B `track_b_*` columns, no re-forward). Beating only the fresh-small straw-man does NOT count.
- **(e)** Bonferroni-corrected p < 0.05 (5 n_cal × 5 conformal baselines = 25 comparisons → α/25 = 0.002).

### ⚠ WEAK / convenience claim allowed IFF:
- CGM-raw-APMD `≈` stale-conformal (NS after Bonferroni) AND > fresh-small at n_cal ≥ 10.
- Framing: "no-calibration-step convenience matches stale-conformal performance"; performance superiority NOT claimed.

### ❌ STRONG claim FORBIDDEN IFF (ANY of):
- Crossover only at n_cal < 5.
- Fresh-small-conformal catches up to CGM-raw at small n_cal.
- Bonferroni-corrected NS.
- CGM-raw NS-vs or worse-than stale-conformal (Track B reference).

### Pre-registered outcome space (3 rows)

| Outcome | Meaning | Prior probability | Direction |
|---|---|---|---|
| CGM-raw > stale-conformal at n_cal ≥ 10 (Bonferroni) | True performance lead | ~15% | ✅ JBHI strong case |
| CGM-raw ≈ stale-conformal, > fresh-small at n_cal ≥ 10 | Convenience-only advantage | ~55% | ⚠ JBHI thin; venue downgrade likely |
| Crossover at n_cal < 5 OR Bonferroni NS | Practical advantage marginal | ~30% | 🔴 Pivot |

Priors anchored to Track B finding: 3/5 baselines significant CGM lead retained, 2/5 NS under uniform CQR.

---

## 3. DESIGN (frozen)

### 3.1 n_cal sweep
n_cal ∈ {5, 10, 20, 40, full≈75} (full = national-train residual pool size; matches Track B parent LOCK §2 (a)).

### 3.2 Fresh calibration extraction — TWO modes per K
For each (region, seed, baseline, K ∈ {5, 10, 20, 40}):

- **[first_K]** — test_strict (eps_h1 ≥ 202240, n=149/region) 의 첫 K cells, 시간순. Deterministic, single sample per (region, seed). Realistic deployment scenario. CONFOUNDS: small-n + temporal-shift.

- **[random_K]** — test_strict 전체에서 random K cells without replacement × B=20 independent subsamples. Same-distribution sampling. Controls for temporal-shift; isolates small-n only.

- **Evaluation**: test_strict 의 *remaining* cells (149 − K for first_K; per-subsample remaining for random_K).

### 3.3 Conformal layer
LOCK §3 parent unchanged: CQR-symmetric, finite-sample corrected, 23 FluSight quantiles, single uniform routine `src/eval/wis_standard.quantiles_conformal_cqr`. NO per-baseline radius scaling, NO Mondrian, NO non-symmetric CQR.

### 3.4 Native baselines (n_cal-independent)
Only CGM-raw-APMD genuinely n_cal-independent. NN-baseline native MC-Dropout intervals exist but are severely under-calibrated (Track B native Cov95 = 0.30-0.70 vs 0.95 nominal); reported for completeness only.

### 3.5 Stale-conformal reference (reused, no re-forward)
`runs/track_b_full/per_cell.parquet` `track_b_*` columns = stale-conformal baseline (LOCK §2 (a) parent: "national-train residual reuse" = exactly stale, pre-deployment static calibration). Drawn as horizontal reference lines per baseline in plots.

### 3.6 Aggregation order (LOCK §5 parent, unchanged)
per-cell → mean over 10 regions per (seed, h) → mean over 4 horizons → mean over 5 seeds.

For B=20 random subsamples: extra mean across B at the cell level before region aggregation (giving per-cell finite-sample mean; cell-level std reported separately).

### 3.7 Metrics
- **Primary**: `|Cov95 − 0.95|` per (n_cal, baseline)
- **Secondary**: WIS per (n_cal, baseline)
- **Stability**: σ(`|Cov95 − 0.95|`) across B=20 random subsamples at each n_cal (finite-sample MSE of CQR estimator — NOT "deployment instability")
- **Heterogeneity**: per-region and per-horizon CGM lead disaggregated

---

## 4. ATTRIBUTION SPLIT (frozen)

For each K ∈ {5, 10, 20, 40}:

```
Δ_deployment(K) = mean[|Cov95 − 0.95| @ first_K]
                − mean[|Cov95 − 0.95| @ random_K]
                = temporal-shift contribution
                  (random_K controls small-n; difference = exchangeability violation)
```

### Reporting structure
- **(Headline)** first_K → realistic deployment effect (small-n + temporal-shift combined)
- **(Attribution)** random_K → same-distribution finite-sample effect (small-n only, B=20 std characterized)
- **(Diagnosis)** Δ_deployment = magnitude of CQR exchangeability assumption violation in epidemic time series

### Direction-of-finding decision rule
If |Δ_deployment| dominates total CGM advantage (i.e., |Δ_deployment| ≥ 0.5 × CGM-lead) → finding reframed to "**conformal exchangeability assumption broken in epidemic time series**". This is a separately publishable direction, NOT a salvage reframe of the original claim.

---

## 5. HARD-STOP RULES

- **(a)** Cov95 ∉ [0, 1] OR NaN → STOP, debug split/aggregation. No clipping.
- **(b)** Conformal radius NaN / inf → STOP, debug n_cal extraction. No silent filtering, no fallback to native.
- **(c)** test_strict split overlap (first_K ∩ random_K ∩ evaluation NOT disjoint per region/seed) → STOP.
- **(d)** LSTM full n_cal smoke ≠ Track B LSTM (WIS 0.3676, Cov95 0.8738) within |Δ| < 0.005 → STOP, debug stale-reference correspondence.

No methodology swap, no post-hoc tolerance widening, no baseline addition, no n_cal grid modification after launch.

---

## 6. AS-IS REPORTING (verbatim, user-confirmed 2026-06-23)

> Results reported exactly as computed. The realistic ceiling is convenience-only (CGM ties stale-conformal); this is the ~55% prior outcome. If CGM-raw genuinely beats stale-conformal at n_cal ≥ 10 with Bonferroni significance, that is the JBHI-strong headline. If CGM-raw ties stale-conformal, the publication value is the *measurement* itself: it converts the venue decision (JBHI / downgrade / pivot) from guess to data. NO post-hoc strong-version recovery, NO straw-man comparison ("we only meant fresh-small"), NO reframing of the convenience claim into a performance claim.

### Attribution outcome
If attribution split shows temporal-shift dominant → the headline becomes "conformal exchangeability violation in epidemic forecasting" (separate finding direction). If small-n dominant → original claim holds. Either way, reported as-found.

---

## 7. RISK ACKNOWLEDGMENT (explicit, LOCK time)

User confirmed (2026-06-23) the following risks at LOCK time:

1. **Paper bet**: paper primary contribution is bet on this experiment's outcome.
2. **Realistic ceiling = convenience**: probability >50% that landing is "CGM ties stale-conformal" — Track B's 2/5 NS baselines are the prior.
3. **JBHI strong case unlikely**: CGM-raw > stale-conformal with Bonferroni p < 0.05 at n_cal ≥ 10 is the minority outcome.
4. **NS/thin acceptance**: as-is reporting includes the NS outcome; no salvage reframe.
5. **Direction reframe possibility**: attribution split may reveal exchangeability-violation as the true finding; this is honest, not a reframe.
6. **Experiment value = decision**: this is the last cheap gate to convert venue choice from guess to measurement. NS/thin outcome is itself a publishable-quality finding ("native UQ convenience without performance advantage at realistic n_cal").

---

## 8. PRE-FLIGHT CHECKS (required BEFORE sweep launch)

1. **LSTM full n_cal smoke**: Run cold-start pipeline at K = full (n_cal ≈ 75, national-train residual pool) for LSTM only. Check against Track B `track_b_*` LSTM values (WIS = 0.3676, Cov95 = 0.8738). |Δ| < 0.005 required, else hard-stop (d).
2. **Split disjoint verify**: For each (region, seed, K), assert (first_K) ∩ (random_K subsample b) = ∅ AND (first_K ∪ random_K_b) ∩ (evaluation_b) = ∅ for b ∈ {1..20}.
3. **Aggregation order verify**: per-cell → region-mean → horizon-mean → seed-mean matches parent LOCK §5. Unit-test on synthetic input.
4. **Stale reference correctness**: confirm `runs/track_b_full/per_cell.parquet` `track_b_*` columns are the parent-lock-compliant stale baseline (LOCK §2 (a) national-train residual reuse).

---

## 9. PRE-REGISTERED CONSTANTS

| Key | Value | Rationale |
|---|---|---|
| n_cal_values | {5, 10, 20, 40, full≈75} | Sweep range covering "unrealistic" → "realistic" → "Track B full" |
| B_subsamples | 20 | Finite-sample MSE estimation of random_K CQR |
| n_seeds | 5 (42, 123, 456, 789, 1024) | Parent lock |
| n_regions | 10 (hhs1..hhs10) | Parent lock |
| n_horizons | 4 (h=1..4) | Parent lock |
| test_set | test_strict (eps_h1 ≥ 202240, n=149/region) | Parent lock |
| baseline_set | {CGM-APMD, LSTM, VM, PatchTST, DLinear, EpiDeep} | Parent lock §1 |
| stale_reference | Track B full-run track_b_* columns, NO re-forward | LOCK §2 (a) parent |
| primary_metric | \|Cov95 − 0.95\| | User condition |
| secondary_metric | WIS | User condition |
| multi_comparison | Bonferroni primary (α=0.05/25=0.002), uncorrected disclosed as secondary | Statistical rigor + power transparency |
| heterogeneity_checks | per-region, per-horizon CGM lead disaggregation | Convenient-skepticism rule |
| attribution_split | first_K vs random_K | Temporal-shift attribution required |
| crossover_realistic_anchor | n_cal ≥ 10 (~2.5 months weekly surveillance) | Clinical anchor |
| reframe_lock | No new baselines, no metric swaps, no n_cal grid changes, no salvage | As-is enforced |

---

## 10. CHANGE CONTROL

Identical to parent lock §11:
1. Explicit user confirmation referencing this file by path.
2. New dated append-only block appended below Section 10 (no in-place edits to Sections 1-9).
3. Re-verification that no clause conflicts with parent locks (Track B + PC0/PC1/PC2).

End of LOCKED cold-start sub-pre-registration (sections 1-10).

---

## 11. APPEND BLOCK — 2026-06-23 PROPER-SCORE DISCIPLINE TIGHTENING (user-confirmed)

**User catch (2026-06-23)**: §2 strong-claim criteria (a) and (d) defined `|Cov95 − 0.95|` only; this admits *over-dispersion gaming* ("wide intervals → good coverage" without sharpness). Project-wide discipline carried through PC0/PC1/PC2 + Track B = *coverage AT competitive proper score*, NOT coverage alone. Track B evidence: CGM-raw |Cov95−0.95|=0.005 vs LSTM-stale=0.076 (Cov95 → CGM wins), but WIS CGM-raw 0.391 vs LSTM-stale 0.368 (WIS → LSTM-stale sharper). Cov95-only criterion would call this "CGM honestly wins" when it's "CGM wider intervals."

### 11.1 §2 ✅ STRONG claim — ADDITIONAL WIS conditions (additive, does NOT weaken existing)

- **(a) UNCHANGED**: CGM-raw-APMD `|Cov95 − 0.95|` < best fresh-conformal-baseline at n_cal ∈ {5, 10, 20, 40}.
- **(a') NEW**: AND CGM-raw-APMD WIS *NS-or-better* vs that same best fresh-conformal-baseline. NS-or-better defined as:
  - Paired t-test 5-seed, two-tailed → p ≥ 0.05 (NS), OR
  - p < 0.05 AND CGM-raw-APMD mean WIS ≤ baseline mean WIS (CGM-better).
  - Bonferroni-corrected (α = 0.05/25 = 0.002).
- **(d) UNCHANGED**: CGM-raw-APMD `|Cov95 − 0.95|` < stale-conformal reference.
- **(d') NEW**: AND CGM-raw-APMD WIS NS-or-better than stale-conformal (same protocol as (a')).

### 11.2 New ❌ FORBIDDEN condition

- CGM-raw-APMD wins on `|Cov95 − 0.95|` but is significantly *worse* on WIS (Bonferroni-corrected paired t-test p < 0.05 AND CGM mean > baseline mean) → strong claim FORBIDDEN. This is the over-dispersion artifact.

### 11.3 §4 ATTRIBUTION clarification — exchangeability reframe is NOT a CGM contribution

**User catch (2026-06-23)**: §4's "separately publishable" exchangeability-violation finding (if temporal-shift dominant) is a finding about *conformal's weakness*, NOT about CGM. Adding to the lock:

- If `|Δ_deployment|` dominates the headline, finding is reported as "**conformal exchangeability assumption broken in epidemic forecasting**" — SEPARATE finding direction.
- It does NOT count toward CGM's contribution.
- No salvage reframing: a conformal-side finding cannot upgrade a CGM-side claim.
- Decision rule: if attribution split dominates, the paper's CGM-side contribution is *downgraded* (the cold-start CGM advantage is mostly mediated by conformal's failure, not CGM's strength), not upgraded by the orthogonal conformal finding.

### 11.4 Updated outcome space (replaces §2 table)

| Outcome | New criterion | Prior | Direction |
|---|---|---|---|
| CGM-raw > stale-conformal on Cov95 AND WIS NS-or-better at n_cal ≥ 10 (Bonferroni) | Honest: coverage + sharpness | ~15% | ✅ JBHI strong |
| CGM-raw > stale Cov95 but significantly worse WIS | Over-dispersion artifact | <5% (Track B prior) | 🔴 Strong claim FORBIDDEN |
| CGM-raw ≈ stale on both Cov95 and WIS (NS) | Convenience-only | ~55% | ⚠ JBHI thin |
| Crossover at n_cal < 5 OR Bonferroni NS on either metric | Marginal | ~25% | 🔴 Pivot |

(Sum = 100%; over-dispersion artifact moved from "stealth strong" to its own forbidden row, prior reallocated.)

### 11.5 Pre-flight strengthening — hard-stop (d) now requires BOTH metrics

- **§5 (d) AMENDED**: LSTM full n_cal smoke ≠ Track B LSTM on EITHER (WIS = 0.3676 within |Δ| < 0.005) OR (Cov95 = 0.8738 within |Δ| < 0.005) → STOP. (Previously only Cov95 was named; WIS now required.)

### 11.6 Statistical reporting requirement

For each (n_cal, baseline) pair in the sweep, the report MUST present:
1. `|Cov95 − 0.95|` mean ± 5-seed std + Bonferroni-corrected paired t-test vs CGM-raw.
2. WIS mean ± 5-seed std + Bonferroni-corrected paired t-test vs CGM-raw.
3. JOINT verdict: STRONG / FORBIDDEN-by-over-dispersion / CONVENIENCE / NS.
4. Region heterogeneity (per-HHS lead disaggregation).
5. Horizon heterogeneity (h=1..4 lead disaggregation).
6. Attribution split (Δ_deployment vs random_K only).

### 11.7 Conflict check vs parent locks

- Parent Track B sub-pre-reg §3 (CQR-symmetric uniform routine) — UNCHANGED.
- Parent Track B sub-pre-reg §5 (Bracher 2021 WIS) — REINFORCED (WIS now co-primary, not auxiliary).
- Parent PC0/PC1/PC2 — UNCHANGED.

No clause in this append conflicts with parent locks.

### 11.8 Confirmation

User confirmed this proper-score discipline tightening at LOCK time (2026-06-23) before sweep launch. As-is rule (§6) extended to cover the new (a')/(d') conditions and the over-dispersion forbidden row.

End of 2026-06-23 append block.

End of LOCKED cold-start sub-pre-registration (with append §11).
