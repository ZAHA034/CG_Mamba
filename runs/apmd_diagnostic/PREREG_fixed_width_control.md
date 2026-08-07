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
high-variance weeks** (1.03 vs 1.38) and **~3.2× wider in low-variance weeks** (1.03 vs 0.32). Since APMD already
covers only 0.94 (h=1) / 0.89 (h=4) in the high-variance state, a 25%-narrower fixed width there is predicted to
fall **below ~0.85 at h=4**. High-variance weeks = epidemic surge = the operationally costliest cells to
under-cover; so if confirmed, the result is an **operational** win (surge under-coverage), not merely statistical.

## 3. Design (no retrain)
- Same point forecast μ_CGM. Two fixed-width controls: **(V1)** √(Σ_k π_k σ²_k) (stationary HMM scale);
  **(V2)** the global training-residual 95% quantile. Both are frozen constants (generative/residual-sourced,
  no per-location calibration data — so the deployability property is identical to APMD's under either control).
- Build Gaussian intervals/quantiles from μ_CGM ± the constant scale; score APMD vs V1, V2.
- **Primary metric: Cov95 in the high-variance-phase cells** (per-horizon within it). **Secondary:** aggregate
  WIS + its dispersion/miss-penalty decomposition; horizon-stratified and per-region Cov95/WIS.
- Aggregate coverage is expected to be ~indistinguishable (all share σ²_k); the test is decided in the
  stratified cells, per §0.

## 4. Decision rule / falsification (both branches locked BEFORE scoring)
- **Branch APMD-WINS (predicted):** if, in the high-variance cells, fixed-width Cov95 is **≥ 0.02 further from
  nominal than APMD's** (0.02 = the paper's revision-noise floor) AND its WIS miss-penalty there is higher →
  confirm "phase-adaptivity avoids surge under-coverage (operationally meaningful)." Keep the phase-mixture
  structure; add the two-regime narrative and the width-driver clarification (width varies via γ-weighted
  σ²_within, CV 52%; σ²_between's 2.5% is a separate, small term — rebutting the "static/decorative width"
  reading).
- **Branch PREDICTION-FAILED / TIE:** if fixed-width Cov95 in the high-variance cells is **within 0.02 of APMD's**
  and WIS is indistinguishable → record the prediction as **FAILED**; **retract** the claim that the phase-mixture
  machinery is operationally necessary; **simplify** the method and re-narrate ("a frozen 3-state HMM's
  stationary emission variance as a constant interval scale attains zero-shot regional Cov95 0.954 while all deep
  UQ methods sit at 0.29–0.70"). The deployability contribution survives unchanged in this branch.

## 5. Honest limitations to state regardless of branch
- Width ≤ max σ_k = 1.381 (convex-combination ceiling): **the interval cannot exceed the emission variance of the
  widest fitted phase; unprecedented-severity seasons fall outside the construction's range.**
- The width adapts ~4× across **phase** but −1% across **horizon**: the former is the mechanism, the latter the
  disclosed limitation → horizon-adaptive scale is the obvious next step, not an excuse.

## 6. Artifact / timestamp
Committed to the git-tracked `runs/` area before scoring; git history is the forking-path / confirmation-bias
artifact (CV pre-check result recorded in §0 is what we knew going in).
