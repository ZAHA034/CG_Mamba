# PRE-REGISTRATION (DRAFT — pending author confirmation) — Table IV ablation reanalysis

**Status:** DRAFT written BEFORE any confidence-interval computation. Point estimates are already
visible in the compiled manuscript (ΔMAE −0.003, ΔWIS −0.000, ΔCov95 +0.004; −Rollout ΔCov95 −0.036;
−Env ΔMAE +0.161; Full CG-Mamba MAE 0.392 vs Table I headline 0.397). **The CI half-widths have NOT been
computed.** This rule governs the *claim* drawn from the (unseen) CIs. LOCK upon author approval; no CI is
computed until locked.

## 0. Unconditional reporting change (holds in every branch)
Table IV switches from the bold = "95% bootstrap CI excludes 0" dichotomy to **effect size + 90% interval
(paired t, df = 4)** for every Δ, with the **5 per-seed paired differences shown** (caption/footnote) and
the **reference-model seed SD** reported (Full CG-Mamba MAE 0.392 ± SD). Percentile bootstrap is dropped
(poor CI coverage at n = 5). Paired iff Full and each variant share the seed stream — to be confirmed and
stated in the caption; if unpaired, paired analysis is still valid but conservative.

## 1. SESOI — justified by rank / conclusion preservation (INTERNAL), not by trust in the metric
An effect is **material** iff it is large enough to change a stated conclusion of *this* paper. This is a
statement about internal decision-sensitivity and is logically independent of whether the (vintage-conditional)
MAE is externally trustworthy — MAE is de-headlined for external-validity reasons; the SESOI below uses its
*magnitude* only as an internal threshold.
- **SESOI_MAE = 0.038** — the national DL-best MAE margin (CG 0.397 vs nearest 0.435). A gate effect ≥ this
  would forfeit CG's DL-best-MAE status.
- **SESOI_WIS = 0.023** *(candidate — CONFIRM before lock)* — the regional DL-family WIS lead (CG 0.393 vs
  LSTM 0.416). Note: the ablation Δ is national test_strict while the protected WIS conclusion is regional;
  importing the (smaller) regional margin as the SESOI is the **conservative** choice (harder to claim
  equivalence). Confirm the intended WIS conclusion to protect.
- **SESOI_Cov95 = 0.02** *(candidate — CONFIRM)* — the coverage change that would alter the "near-nominal"
  characterization. (Revision-noise scale 0.034 is an alternative external anchor; 0.02 is the tighter,
  conservative choice.)

## 2. Per-metric decision, using the 90% CI of the ablation Δ (= effect of REMOVING the component)
Evaluate ΔMAE and ΔWIS **independently** (per-metric split — a narrow ΔMAE CI does not license a claim about
ΔWIS):
- **Branch E (Equivalence):** 90% CI ⊂ (−SESOI, +SESOI) → claim "the gate's marginal effect on [metric] is
  bounded below the decision-relevant scale (SESOI)"; the *no-contribution* statement stands as a bounded
  equivalence result.
- **Branch I (Inconclusive):** CI includes 0 but extends beyond ±SESOI → cannot bound. Demote the [metric]
  null to a **descriptive observation** (report point estimate + CI, no equivalence claim); rely on the
  CI-independent protocol-offset anchor (§4). MAY trigger an **ablation-only** seed increase, explicitly
  labelled "robustness; primary inference (region/origin) unchanged."
- **Branch R (Resolved):** CI excludes 0 →
  - **R1 (resolved-immaterial):** CI ⊂ (−SESOI, +SESOI) → statistically resolved yet below the decision scale
    → report as a small effect size **explicitly labelled immaterial**; NOT contribution-reversing, and (for
    Cov95) NOT a "harm" claim.
  - **R2 (material):** CI extends beyond ±SESOI → the gate materially affects [metric] → **Contribution 3 is
    revised** to report a gate contribution on [metric]; narrative updated.

## 3. ΔCov95 (the +0.004) — handled under the same branches with SESOI_Cov95
The visible +0.004 is **pre-committed to Branch R1 (resolved-immaterial)**: it is expected to be statistically
resolved (coverage is near seed-invariant because the frozen K=3 HMM is reproducible, κ_min = 1.000) yet far
below SESOI_Cov95. We will report it as a small effect below the decision scale, **not** as evidence the gate
harms calibration. (If, contrary to expectation, |CI| reaches beyond SESOI_Cov95 → R2 for Cov95.)

## 4. CI-independent anchor (stated regardless of branch)
The gate effect (|ΔMAE| = 0.003) is smaller than the offset between the headline model and its from-scratch
replica (0.397 → 0.392 = 0.005), i.e., **below the scale at which the paper's own protocol variation moves the
metric** (framed as a protocol/replication offset, not a clean reproducibility floor — the two means come from
different training protocols).

## 5. Contribution 3 restructure (holds in ALL branches)
The contribution's substance is the two large, unambiguous effects — **Env → accuracy (ΔMAE +0.161)** and
**Rollout → calibration (ΔCov95 −0.036)**. The gate's null is a subordinate, bounded observation, not the
contribution's basis. No branch removes Contribution 3; branches change only how the gate's null is stated.

## 6. Naming (#4)
Decided after the table is recomposed. The effect-size table + bounded-null makes the text-handling defense
(gate = structural posterior→selectivity path; marginal effect bounded below the decision scale; retained for
architectural completeness) sufficient; a rename is optional, not required.

## 7. Artifact / timestamp
This file is written before any CI computation. If the paper directory is under git, commit before computing;
otherwise the file mtime + this PREREG convention + the session transcript are the timestamped artifact.
