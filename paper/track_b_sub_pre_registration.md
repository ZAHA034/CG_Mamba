# Track B Sub-Pre-Registration (LOCKED, append-only)

**Status:** LOCKED 2026-06-21
**Parent:** `project_cgmamba_pc012_locked` (CG-Mamba PC0/PC1/PC2 v2 LOCKED, 2026-06-12)
**Relationship:** This document is an APPEND-ONLY EXTENSION to the parent lock. It does NOT replace, supersede, weaken, or reframe PC0/PC1/PC2. It pre-registers ONE additional analysis (Track B: uniform Split-Conformal UQ harmonization) over the EXISTING Table IV baseline set. If any clause here conflicts with the parent lock, the parent lock wins and the conflicting clause is null.
**As-is rule (user-confirmed, beta path):** Success of Track B means *architecture-vs-UQ-method separation plus honest reporting*. Success does NOT mean CGM winning. Results are reported exactly as computed.

---

## 1. SCOPE (frozen baseline set)

Exactly **6 baselines** are evaluated under Track B, identical to Table IV of the manuscript:

1. **CG-Mamba APMD** (ours)
2. **LSTM**
3. **Vanilla Mamba**
4. **PatchTST**
5. **DLinear**
6. **EpiDeep**

**ABSOLUTE EXCLUSIONS (no exceptions):**
- iTransformer — NOT added (would be a reframe of the locked scope).
- TimesNet — NOT added (would be a reframe of the locked scope).
- N-BEATS — NOT added (would be a reframe of the locked scope).
- SARIMAX is referenced only as a classical anchor and is NOT in the conformal comparison set.

Any future request to "add just one more baseline" is by definition a reframe and is rejected by this lock.

---

## 2. CALIBRATION WINDOW (committed choice)

**Committed choice: (a) national-train residual quantile reuse.**

Rationale: Option (a) preserves the locked PC0/PC1/PC2 *national-trained, regional-inference* protocol verbatim. Residuals used for Split-Conformal calibration are drawn from the same national-training residual pool that the base models were fit against, with the held-out regional inference windows untouched. Option (b) (a separate 201840–202010 cal window with 75 origins x 4 horizons) would introduce a second data-partition convention into the locked protocol; rejected.

**Operational definition of the cal pool:**
- Source = national-train residuals over the in-sample national window already declared in the parent lock.
- Per-region calibration uses the parent-lock `n_cal<10 void` rule (region-horizon cells with fewer than 10 valid residuals are voided, not back-filled).
- No re-fitting, no re-splitting, no re-weighting.

---

## 3. CONFORMITY SCORE (CQR-symmetric, finite-sample corrected)

For each FluSight quantile level `tau` (23 levels), the symmetric CQR conformity score is:

```
s_i = max( base_q_{alpha/2}(x_i) - y_i ,  y_i - base_q_{1 - alpha/2}(x_i) )
```

where `base_q` is the baseline-specific base quantile predictor (Section 4 below) and `alpha` is tied to the target FluSight nominal coverage of that quantile pair.

- **Finite-sample radius:** `q_hat = ceil( (n_cal + 1) * (1 - alpha) ) / n_cal`-th order statistic of `{s_i}`. If `q_hat` exceeds 1, the cell is voided per the parent lock.
- **Symmetric application:** identical radius applied to lower and upper quantile of each FluSight pair.
- **Uniform implementation:** all 6 baselines use the SAME routine: `src/eval/wis_standard.quantiles_conformal_cqr`. No baseline-specific wrapper, no per-baseline radius scaling.
- Applied uniformly across all **23 FluSight quantiles**.

---

## 4. BASE QUANTILE per BASELINE (frozen)

| Baseline | Base quantile source |
|---|---|
| CG-Mamba APMD | Gaussian PI from `(mu, sigma2_total)`; quantile = `mu + Phi^{-1}(tau) * sqrt(sigma2_total)` |
| LSTM | Empirical quantiles from **n=100 MC-Dropout** forward passes |
| Vanilla Mamba | Empirical quantiles from **n=100 MC-Dropout** forward passes |
| PatchTST | Empirical quantiles from **n=100 MC-Dropout** forward passes |
| EpiDeep | Empirical quantiles from **n=100 MC-Dropout** forward passes |
| DLinear | Gaussian PI from ensemble `(mean, std)` over the locked ensemble seeds |

All six are then CQR-conformalized per Section 3. No baseline uses bare native quantiles in the Track B comparison.

---

## 5. AGGREGATION (Bracher 2021 WIS, locked)

- **Per-cell metric:** standard WIS over the 23 FluSight quantiles via `src/eval/wis_standard`.
- **Aggregation order:** per-region per-horizon WIS computed first, then mean across regions, then mean across horizons.
- Coverage diagnostics: per-region Cov50 and Cov95 reported at the same granularity.
- No region weighting, no horizon weighting, no top-region trimming.

---

## 6. HARD-STOP RULES

The following are STOP-the-experiment triggers. None of them permit a methodology swap.

- **(a) Smoke inversion / NS:** If the smoke LSTM run shows the CGM WIS lead inverting or becoming non-significant (NS), document the result and proceed to the full evaluation as planned. **NO dropout-rate retuning. NO alpha reselection. NO swap to Mondrian or weighted conformal. NO swap to non-symmetric CQR.**
- **(b) NaN / inf in conformal radius:** STOP, debug the source (cal pool size, score overflow, ckpt corruption). **No silent filtering, no NaN replacement, no fallback to native quantiles.**
- **(c) Baseline ckpt mismatch with Section IV.2:** STOP, escalate to user. **No alternate ckpt, no retraining, no "closest available" substitution.**
- **(d) Per-region Cov95 outside the [0.5, 1.0] plausible band:** STOP, debug the cal pool and the base quantile mapping for that baseline. No truncation, no clipping of the reported number.

---

## 7. AS-IS REPORTING RULE (verbatim, user-confirmed)

> Track B results are reported exactly as computed. The F3 horizon-collapse pattern (LSTM 0.68 to 0.39) and the Cov95 +23 to +64 percentage-point gap are EXPECTED to substantially shrink under uniform conformal; this is the truth-test of the Track B analysis. If the gap shrinks to a narrow defensible WIS-sharpness lead, that is the headline. If the gap inverts, that is the headline. No post-hoc strong-version recovery, no "but if we also..." rescue, no reframing of CGM's claim from UQ-superiority to architecture-superiority *after* seeing the numbers.

Architecture-vs-UQ-method separation is the *analytical goal*; CGM winning is NOT the success criterion.

---

## 8. PUBLICATION FORMAT (both tables required)

- **Table IV (native UQ, retained):** the original Table IV is published as-is, with an explicit caveat citing **Foong et al. 2019** on MC-Dropout OOD miscalibration. This caveat names MC-Dropout-based intervals (LSTM, Vanilla Mamba, PatchTST, EpiDeep) as susceptible to OOD overconfidence/underconfidence.
- **Table IV-prime (uniform Split-Conformal, new):** all 6 baselines re-scored under the Section 3 protocol. Same metric stack (WIS, Cov50, Cov95). Same aggregation order.
- **Section V narrative:** the comparison is framed as *architecture under the same UQ*, NOT as "compact SSM holds". The locked narrative anchor from the parent lock is preserved; no narrative drift to a UQ-method claim and no drift to an architectural-superiority claim.

---

## 9. RELATIONSHIP TO PARENT LOCK

This document is an **append-only extension** of `project_cgmamba_pc012_locked` (2026-06-12 v2). It:

- Does NOT replace PC0/PC1/PC2.
- Does NOT alter the 5 locked operational constants (cluster bootstrap, adjacency-2H, n_cal<10 void, sigma_k diag/Viterbi, narrative lock).
- Does NOT change the baseline set, the regional-inference protocol, or the WIS implementation.
- ADDS one analysis (Track B uniform-conformal Table IV-prime) with hard-stop rules and a verbatim as-is reporting rule.

Per `feedback_lock_state_no_reframe`, no further self-initiated extensions, sensitivity sweeps, or "one more" additions will be proposed after this round. Operational-detail sub-pre-registration ends here.

---

## 10. PRE-REGISTERED CONSTANTS (frozen)

| Key | Value | Rationale |
|---|---|---|
| baseline_set | {CGM-APMD, LSTM, VanillaMamba, PatchTST, DLinear, EpiDeep} | Mirrors Table IV exactly. |
| excluded_baselines | {iTransformer, TimesNet, N-BEATS} | Adding them = reframe; forbidden by lock. |
| cal_window_choice | (a) national-train residual reuse | Preserves national-train / regional-infer protocol verbatim. |
| alpha_levels | 23 FluSight quantile levels (paired around medians) | Matches the FluSight quantile grid used in WIS. |
| n_samples_MC | 100 | Matches Section IV.2 MC-Dropout draw count for LSTM/VMamba/PatchTST/EpiDeep. |
| dlinear_uq | Gaussian PI from ensemble (mean, std) | Matches DLinear's locked ensemble seed set; no MC-Dropout. |
| cgm_uq | Gaussian PI from (mu, sigma2_total) | Native APMD output; no MC sampling. |
| conformity_score | CQR-symmetric, finite-sample corrected | Same routine across all 6 baselines. |
| conformal_routine | src/eval/wis_standard.quantiles_conformal_cqr | Single code path; no per-baseline wrapper. |
| wis_routine | src/eval/wis_standard (Bracher 2021) | Matches parent lock metric stack. |
| aggregation_order | per-region per-horizon -> mean over regions -> mean over horizons | Same as parent lock reporting order. |
| n_cal_void_threshold | <10 (inherited from parent lock) | Parent lock rule; not relitigated. |
| scaler_source | parent-lock national-train scaler | No re-fit, no per-region rescale. |
| ckpt_source | Section IV.2 checkpoints, byte-identical | Hard-stop (c) covers mismatch. |
| nan_inf_policy | STOP + debug, no silent filtering | Hard-stop (b). |
| cov95_plausible_band | [0.5, 1.0] | Hard-stop (d). |
| smoke_target | LSTM, full Track B pipeline dry-run | Trigger for hard-stop (a) check. |
| publication_format | Table IV (native, with Foong 2019 caveat) + Table IV-prime (uniform conformal) | Both required; not either-or. |
| narrative_frame | "architecture under same UQ", not "compact SSM holds" | Parent-lock narrative anchor preserved. |
| reframe_lock | No new baselines, no methodology swaps, no post-hoc rescue | Enforces as-is rule. |

---

## 11. CHANGE CONTROL

Any change to this document requires:
1. Explicit user confirmation referencing this file by path.
2. A new dated append-only block appended below (no in-place edits to Sections 1–10).
3. Re-verification that no clause conflicts with `project_cgmamba_pc012_locked`.

End of LOCKED Track B sub-pre-registration.
