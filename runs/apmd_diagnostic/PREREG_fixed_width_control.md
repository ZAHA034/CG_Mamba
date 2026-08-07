# PRE-REGISTRATION (DRAFT) — Fixed-width control for APMD (§5)

**Status:** DRAFT, written BEFORE any scoring. No-retrain reanalysis (uses existing forecasts μ_CGM, realized y,
phase posterior, and frozen HMM emission variances). LOCK on author approval; scoring runs only after commit.

## 0. What we already knew before this pre-reg (transparency; the risk here is confirmation-biased scoring, not margin-shopping)
A 10-minute CV pre-check was run on `runs/apmd_diagnostic/apmd_residuals.csv` (regional). Results, which are the
basis for the prediction below:
- **CV of the APMD interval half-width = 52.6%** (per-horizon 51–54%): the width is NOT constant.
- The width distribution is **effectively bimodal**: 10/50/90th percentiles = [0.324, 1.381, 1.381], and
  √0.105 = 0.324, √1.907 = 1.381 — i.e. the width **switches between the low- and high-variance emission states**.
- Emission variances σ²_k = {0.105, 0.179, 1.907} (**18× spread** = the convex-combination ceiling); dominant-state
  fractions ≈ {high-var 0.52, 0.28, 0.20}. Median = 90th percentile = high-var peak ⇒ >half of samples sit at the
  wide regime.

Because the branch is now *predicted*, the guarded risk is reading an ambiguous result favorably. §4 fixes that.

## 1. Hypothesis (pre-specified)
The APMD interval is a **two-regime width switch** — ≈0.32 (low-activity weeks) vs ≈1.38 (epidemic weeks) —
selected by the phase posterior. A single fixed width cannot reproduce this and will **over-cover low-activity
weeks and under-cover epidemic (high-variance) weeks**.

## 2. Quantitative prediction (pre-specified, before scoring)
Fixed-width variant 1 uses the stationary-weighted emission scale √(Σ_k π_k σ²_k) ≈
√(0.52·1.907 + 0.28·0.105 + 0.20·0.179) ≈ √1.05 ≈ **1.03**. Relative to APMD it is **~25% narrower in
high-variance weeks** (1.03 vs 1.38) and **~3.2× wider in low-variance weeks** (1.03 vs 0.32). Since APMD already covers only 0.94 (h=1) / 0.89 (h=4) in the high-variance state, and 0.89 corresponds to ±1.60σ
of the realized error, a 25%-narrower interval (±1.20σ) is predicted to give **≈0.77 at h=4** — a ~0.12 gap, far
above the 0.026 threshold (§4). High-variance weeks = epidemic surge = the operationally costliest cells to
under-cover; so if confirmed, the result is an **operational** finding (fixed width under-covers the surge), not
merely statistical.

## 3. Design (no retrain)
- Same point forecast μ_CGM. Two fixed-width controls: **(V1)** √(Σ_k π_k σ²_k) (stationary HMM scale);
  **(V2)** the global training-residual 95% quantile. Both are frozen constants (generative/residual-sourced,
  no per-location calibration data — the deployability property is identical to APMD's under either control).
  **V1 and V2 answer different questions:** V1 (same frozen HMM) tests whether the γ-weighting is needed *given*
  the HMM — a **phase-necessity** test; V2 (residual-fit constant) tests whether a generative-sourced scale is
  needed at all — a **generative-source / transfer** test, and the **stronger competitor** (a residual-quantile
  constant is the best fixed width a practitioner would actually deploy). Beating V1 alone invites "you picked a
  weak constant"; beating V2 blocks that.
  **V2 consistency check:** V2 is residual-fit, and §IV-D reports a residual-fit learned head under-covers on
  zero-shot regional transfer (0.868 vs 0.954); V2 should likewise under-cover regionally. If V2 instead transfers
  well, that tensions with §IV-D and must be investigated, not glossed.
- Build all 23 FluSight quantiles as $\mu_{\text{CGM}} + \Phi^{-1}(q)\cdot(\text{constant scale})$ at the same
  levels used for APMD; score APMD vs V1, V2.
- **High-variance cell definition (fixed, not invented):** the forecast-origin dominant phase, per §IV-C2 verbatim
  — "Conditioning on the frozen HMM's dominant phase at the forecast origin ($\arg\max\gamma_h$; states ordered by
  emission variance)"; the high-variance cell = the highest-emission-variance state (the post-COVID dominant
  regime, 59% of regional test-weeks).
- **Primary decision metric (single): the SIGNED Cov95 gap in the high-variance cells, per horizon** — the
  prediction is *under-coverage* by the fixed width (§4). Aggregate coverage is expected ~indistinguishable
  (shared σ²_k); the decision is made in this stratum only.
- **Corroborating evidence only (NOT in the verdict):** aggregate WIS + dispersion/miss-penalty decomposition;
  horizon- and per-region-stratified Cov95/WIS. Reported for mechanism consistency, never entering the decision.
- **Inference unit / multiplicity:** the high-variance-cell Cov95 is a proportion over the high-variance-stratum
  forecast cells (pooled over regions; autocorrelated), reported descriptively with **no significance test and no
  multiplicity adjustment** (estimates, not tests), consistent with the paper's stance.

## 4. Decision rule / falsification (single metric; locked BEFORE scoring)
Decision is on the **signed high-variance-cell Cov95 gap** alone — WIS decomposition is corroboration, not part of
the verdict (this closes the mixed-outcome hole). Threshold = **0.026** (the ablation's SESOI_Cov95 = cross-region
Cov95 SD): the SAME threshold paper-wide, avoiding threshold-shopping — and it is the *harder* bar here (a lower
one would only favour us). **Boundary case:** the verdict is on distance *below* nominal in the under-covering direction; any high-variance
cell where APMD itself over-covers is excluded from the verdict and reported separately (unlikely — high-variance
cells all under-cover, 0.94/0.89 — but stated for completeness). Three mutually exclusive outcomes on the signed gap:
- **APMD-WINS (predicted):** fixed-width high-variance-cell Cov95 is **≥ 0.026 further BELOW nominal** than APMD's
  (fixed under-covers the surge cells more). → confirm the *relative* claim; keep the phase-mixture structure; add
  the two-regime narrative + the width-driver clarification (width varies via γ-weighted σ²_within, CV 52%;
  σ²_between's 2.5% is a separate, small term — rebutting the "static/decorative width" reading). WIS miss-penalty
  reported as corroboration only.
- **TIE / SIMPLIFY:** fixed-width high-variance-cell Cov95 **within 0.026 of APMD's** → record prediction FAILED;
  retract "the phase machinery is operationally necessary"; simplify to the 3-scalar constant scale and re-narrate
  ("a frozen 3-state HMM's stationary emission variance as a constant interval scale attains zero-shot regional
  Cov95 0.954 while all deep UQ sit at 0.29–0.70"). Deployability survives unchanged.
- **WRONG-SIGN (mechanism refuted):** fixed-width **over-covers** the high-variance cells (even if |dev| is large)
  → the operational prediction (fixed under-covers the surge) is FALSE; record as refuted; do NOT claim a win from
  a wrong-direction deviation.

**Winning-branch narrative (pre-committed — most important):** even on a win the numbers are APMD ≈0.89 vs fixed
≈0.77 — *both* under-cover the surge. The claim is therefore strictly **relative** (phase-adaptivity *reduces*
surge under-coverage), **not absolute** (surge coverage is adequate — it is not). The residual absolute
under-coverage (0.89 at high-variance h=4) is exactly what a **horizon-adaptive native scale** would fix; this
result is the direct motivation for that extension, not a claim the interval suffices in surges.

## 5. Honest limitations to state regardless of branch
- Width ≤ max σ_k = 1.381 (convex-combination ceiling): **the interval cannot exceed the emission variance of the
  widest fitted phase; unprecedented-severity seasons fall outside the construction's range.**
- The width adapts ~4× across **phase** but −1% across **horizon**: the former is the mechanism, the latter the
  disclosed limitation → horizon-adaptive scale is the obvious next step, not an excuse.

## 6. Artifact / timestamp
Committed to the git-tracked `runs/` area before scoring; git history is the forking-path / confirmation-bias
artifact (CV pre-check result recorded in §0 is what we knew going in).
