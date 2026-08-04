# Pre-Registration — Provisioning Value of CG-Mamba's Native Calibration (Newsvendor / Native-Quantile)

**Status:** LOCKED on 2026-08-02, before any provisioning-value computation.
**Rule:** Everything below is fixed prior to running. No cost ratio, quantile, metric, stratum, or verdict criterion may be added/changed after seeing results. If infeasible, we abandon rather than re-specify.
**Companion:** This is the SINGLE mechanistically-motivated follow-up to the binary-threshold decision-value experiment (`PREREG_decision_value.md`, verdict KILL). It is NOT iterative metric-shopping: the binary experiment's root-cause analysis (CG has best discrimination AUC 0.972 and best *potential* value, but its calibrated-wide native probabilities are un-decisive under a recalibratable binary threshold) points specifically to decisions that (a) consume the interval *width* and (b) use the *native* quantile without recalibration. If THIS also fails, we stop — no further favorable-experiment search.

---

## 1. Motivation and research question

The binary-threshold experiment showed calibration does not help a decision that only needs discrimination + a recalibratable threshold. This experiment tests the complementary case the mechanism predicts CG should win: a decision whose ACTION IS the model's native quantile, so interval calibration is consumed directly.

**Primary question (confirmatory).** Among amortized-deployable models, when a decision-maker provisions resources at each model's NATIVE predictive quantile (no recalibration), does CG-Mamba incur lower expected newsvendor (asymmetric provisioning) cost than the deep-learning baselines?

**Directional hypothesis.** CG-Mamba's native-quantile provisioning value dominates the best deployable DL baseline, concentrated in the high-critical-ratio (surge-averse) regime where under-provisioning is costly — because CG's near-nominal (safe-direction) upper quantiles are reliable while the under-covering baselines' upper quantiles are too low. (We do NOT hypothesize superiority over per-series SARIMAX.)

## 2. Prior work and differentiation (verified 2026-08-02)

- **Quantile = optimal action under asymmetric (newsvendor) loss** — **Gneiting (2011)** [*Int. J. Forecasting* 27(2):197–207, peer-reviewed, VERIFIED]: the α-quantile is the optimal point forecast under piecewise-linear (newsvendor/pinball) loss with critical ratio α. This is the theoretical anchor: a *calibrated* model's stated α-quantile is the cost-optimal order; a miscalibrated one's is not.
- **Value-score framework** — **Richardson (2000)** [QJRMS, verified]. Newsvendor model: classical (Arrow–Harris–Marschak 1951; textbook — cite to finalize).
- **Differentiation.** The binary threshold experiment (recalibratable) nullified calibration's value; this experiment isolates the case Gneiting (2011) formalizes — the native quantile IS the action, so calibration is consumed. Novelty vs the paper's WIS: WIS *averages* pinball loss over all quantile levels (where CG is DL-competitive); this isolates the *upper-tail / surge* provisioning cost and translates it into an operational decision, under the realistic no-recalibration deployment.

## 3. Data and models (LOCKED)

- Reuse `runs/decision_native/native_predictive.parquet` (consistency gate PASSED 2026-08-02: reproduces paper regional native Cov95 within ≤0.001; dlinear +0.010). **No new inference for the native (A) analysis.**
- `test_strict`, 10 HHS regions, horizons 1–4, raw wILI. Native quantiles per model (23 FluSight levels + mu/sigma). CG-Mamba = raw native APMD.
- Primary comparison (amortized-deployable): CG-Mamba vs {LSTM, Vanilla Mamba, PatchTST, DLinear-ensemble, EpiDeep}. Reference (not in claim): SARIMAX (per-series, amortized), Persistence (floor).

## 4. Decision task — newsvendor (LOCKED)

At each (origin, horizon h, region), the decision-maker provisions an order level equal to the model's native **critical-ratio quantile** `q_CR` (interpolated from the 23 native quantiles). Realized cost for observed `y`:
  `C = c_u · max(y − order, 0) + c_o · max(order − y, 0)`, with `c_u + c_o = 1`, critical ratio `CR = c_u`.
(Equivalently the pinball loss at level CR; by Gneiting 2011 the true CR-quantile minimizes E[C].)
- Reference (climatology): provision at the training-period (2001–2018) seasonal `CR`-quantile per (region, week-of-season).

## 5. Cost ratios / quantile levels (LOCKED — full grid, no cherry-pick)

Report the **entire** `CR ∈ {0.05, 0.10, …, 0.95}` grid (provision at native `q_CR`). The **high-CR (≥0.7) surge-averse regime is flagged a priori as the operationally-realistic public-health case**, but ALL CR are reported; no single CR is selected for the verdict. Two τ-independent — provisioning is on the continuous wILI level, not a threshold, so §Prereg-decision's τ does not apply here.

## 6. Metric (LOCKED)

- **Primary:** Provisioning Value Score `PVS(CR) = 1 − C_model(CR)/C_clim(CR)` per (model, region, horizon, CR). >0 = beats seasonal climatology; report the full CR-curve and pooled means.
- **Secondary (descriptive):** raw expected cost; realized service level (fraction `y ≤ order`) vs nominal `CR` per model — this directly exposes that only a calibrated model delivers the promised service level natively.
- 5-seed mean for CG/DL (as in the paper).

## 7. TWO-SIDED analysis (LOCKED — the honesty core)

Both are computed and BOTH reported regardless of outcome:
- **(A) NATIVE-quantile provisioning** (order at the model's stated `q_CR`) — primary; the paper's no-recalibration deployment. Mechanism predicts CG wins at high CR.
- **(B) RECALIBRATED-quantile provisioning** — each model's quantile function is empirically recalibrated on the **validation split** (a monotone remap so val coverage matches nominal), then applied to test; order at the recalibrated `q_CR`. Mechanism predicts the CG advantage **shrinks** (parallels the binary experiment's optimal-threshold result). (B) requires a validation-split native dump; if infeasible, (B) is represented by the binary experiment's already-established recalibration-nullification result, reported alongside.

The two sides DELIMIT the claim: native calibration's value is precisely that it **eliminates the recalibration step** for interval-consuming decisions.

## 8. Stratification (LOCKED — origin-side only)
Overall + by origin-side intensity regime (canonical-ordered frozen-HMM phase; per `feedback_stratification_labelswitch_audit`). Outcome-conditional stratification prohibited for any claim.

## 9. Pre-registered verdict (LOCKED)

- **KEEP** (scoped claim: *native calibration lowers provisioning cost without recalibration*): CG native-PVS ≥ the best deployable DL baseline's native-PVS in **≥70% of (region, horizon, CR-band) cells** over the reported CR grid, **and** the advantage is concentrated in the high-CR (≥0.7) regime as the mechanism predicts, **and** the recalibrated (B) counterpart is reported honestly (a shrinking advantage there is expected and is the boundary of the claim, not a failure).
- **PARTIAL** (suggestive, no headline): 50–70%.
- **KILL** (omit; no provisioning-value claim): <50%, or CG not better even at high CR.
- SARIMAX: reference curve; per-series SARIMAX beating CG keeps the claim scoped to amortized-deployable.

## 10. Honesty commitments (LOCKED)

1. Locked before computation; this file is the record.
2. **Report native (A) AND recalibrated (B), AND alongside the binary-threshold KILL** — the full two-sided story. One-sided reporting is prohibited.
3. No post-hoc CR/quantile/metric/stratum selection; no re-specification after results.
4. Scope narrowly: "provisioning decisions consuming native intervals." No generalization to "decision value" (the binary KILL disproves that).
5. **Disclose that the favorability derives from CG's already-claimed safe-direction over-coverage** — not a new inflated mechanism.
6. If KILL: omit; paper unchanged (14p). If KEEP: any added text presents BOTH experiments (binary KILL + provisioning KEEP) together.
7. **Anti-fishing:** this is the ONE mechanistically-motivated follow-up; if it fails we stop searching for favorable framings.

## 11. What we will NOT claim
- Not superiority over per-series SARIMAX.
- Not general "decision value" — only native-interval provisioning, scoped, two-sided.
- Not clinical/patient outcomes (retrospective provisioning simulation on surveillance data).

## 12. Feasibility
(A) reuses the existing test-split dump — pure analysis, no GPU. (B) needs a validation-split native dump (same pipeline, val split) OR falls back to the binary experiment's recalibration result. Climatology from training wILI (already built). If (A)'s native quantiles are inconsistent with Table I → already gated PASS, so proceed.

## Citations (verification status)
- Gneiting (2011), IJF 27(2):197–207 — **verified**. · Richardson (2000), QJRMS 126:649–667 — **verified**. · Arrow–Harris–Marschak (1951) newsvendor — textbook, cite to finalize. · Existing: gneiting2007scoring, bracher2021wis, lutz2019applying.

### §12 implementation notes (recorded 2026-08-02, before running; setup re-check)
Two implementation details fixed at the setup re-check (not re-specifications — they do not touch the metric, cost ratios, or verdict; both are CG-neutral-to-unfavorable):
1. **Origin alignment.** cg_mamba has 148 strict origins/cell vs the DL models' 149 (the known regime_shift off-by-one). The CG-vs-DL comparison uses the **intersection of origins** present for all models, per (region, horizon).
2. **Non-negative orders.** wILI (and provisioned resources) cannot be negative; the wide Gaussian lower quantiles go negative. Orders are clipped `order = max(q_CR, 0)`, applied uniformly. This mildly RAISES low-CR orders → more overage for the wider CG at low CR → if anything CG-unfavorable there.
Provisioning climatology = seasonal (region, MMWR-week) CR-quantile of 2001–2018 training wILI (distinct from the binary experiment's crossing-frequency climatology). Note: the CG-vs-DL PVS ranking is climatology-independent (common reference cancels), so climatology choice affects only absolute PVS, not the verdict.

---

## RESULTS (executed 2026-08-02, after lock; no re-specification)

**(A) Native-quantile provisioning — verdict: NOT KEEP (near-miss).**
- Mechanism CONFIRMED as predicted: CG-Mamba has the highest mean PVS in the high-CR (≥0.7, surge-averse) regime (0.668 vs lstm 0.580, all DL lower) and the lowest at low-CR (0.210) — its safe-direction over-coverage helps when under-provisioning is costly and hurts when over-provisioning is costly.
- Pre-registered verdict metric (CG ≥ strongest single DL [lstm] by cell-count): ALL-CR 40%, high-CR≥0.7 **68%**, low-CR 12%. **Does NOT clear the locked 70% KEEP bar under any reading** (overall 40%; high-CR 68%<70%). CG beats 4/5 DL at high-CR (73–88%) but falls just short vs the single toughest (lstm, 68%).
- **Action: OMIT; no provisioning-value claim.** Not re-scoped to high-CR-only (goalpost-moving forbidden), not switched to mean-PVS (which would have favored CG — post-hoc metric change forbidden). The locked cell-count-70% bar was not met.
- **(B) recalibrated: moot** — (A) did not clear KEEP, and recalibration only shrinks CG's edge (established by the binary experiment). Not run.

**Combined conclusion (both experiments):** CG's native calibration is real for interval quality but does NOT support a claimable decision-value advantage — binary threshold (KILL, discrimination-only + recalibratable) and provisioning (near-miss, real but sub-threshold surge-averse signal). The paper's honest scope (claim calibration, not decision superiority) is empirically vindicated. Anti-fishing clause invoked: no further favorable-framing search.
