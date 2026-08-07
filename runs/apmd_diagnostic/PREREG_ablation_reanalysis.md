# PRE-REGISTRATION (LOCKED 2026-08-07) — Table IV ablation reanalysis

**Status:** LOCKED 2026-08-07, BEFORE any gate-row confidence-interval computation (git-tracked; the draft and
this lock are separate commits — git history is the forking-path artifact). Point estimates were already visible
in the compiled manuscript (ΔMAE −0.003, ΔWIS −0.000, ΔCov95 +0.004; −Rollout ΔCov95 −0.036 [−0.055,−0.019];
−Env ΔMAE +0.161 [+0.109,+0.241]; Full CG-Mamba MAE 0.392 vs Table I headline 0.397); **the gate-row CI
half-widths have NOT been computed.** This rule governs the *claim* drawn from the (unseen) CIs. All SESOI
values (§1) are pre-existing manuscript quantities, not derived from the ablation CIs.

## 0. Unconditional reporting change (holds in every branch)
Table IV switches from the bold = "95% bootstrap CI excludes 0" dichotomy to **effect size + 90% interval
(paired t, df = 4)** for every Δ, with the **5 per-seed paired differences shown** (caption/footnote) and
the **reference-model seed SD** reported (Full CG-Mamba MAE 0.392 ± SD). Percentile bootstrap is dropped
(poor CI coverage at n = 5). Paired iff Full and each variant share the seed stream — to be confirmed and
stated in the caption; if unpaired, paired analysis is still valid but conservative.

**Multiplicity:** we report estimates with intervals rather than conducting hypothesis tests; across the
3 metrics × 3 variants no multiplicity adjustment is applied, and no claim rests on a dichotomous significance
decision.

## 1. SESOI — the paper's own revealed thresholds (single principle)
Every SESOI is **the smallest effect this paper itself treats as a claimable advantage** (its *revealed
threshold*). Rationale: calling a gate effect *below* this a "measurable contribution" would apply a stricter
standard to the ablation than the paper applies to its own headline claims (internal inconsistency), while a
looser threshold would be self-serving. The threshold is fixed by the manuscript → **zero selection freedom**
(no margin-shopping). This is internal decision-sensitivity, logically independent of whether the
(vintage-conditional) MAE is *externally* trustworthy.
- **SESOI_MAE = 0.038** — the paper's revealed accuracy threshold: CG's DL-best-MAE claim rests on exactly this
  margin (CG 0.397 vs nearest Vanilla Mamba 0.435; Finding f:national_mae, which itself calls this "the
  DL-family rank, not a resolved statistical separation").
- **SESOI_WIS = 0.023** — the paper's *only* claimed WIS advantage: the regional lead (CG 0.393 vs LSTM 0.416,
  Finding f:region_wis). CG has no national WIS lead (0.399 vs PatchTST 0.368), so there is no other WIS
  conclusion to protect. WIS is scope-invariant in units and the national/regional bands overlap
  (≈0.37–0.60 / 0.39–0.52), so the threshold applies to the (national) ablation Δ.
- **SESOI_Cov95 = 0.026** — the model's reported cross-region Cov95 SD (Findings f:region_cov /
  f:region_stability; 0.954 ± 0.026): a coverage shift smaller than the model's own cross-region variability
  changes no coverage characterization. Units are a 0–1 coverage ratio, matching ΔCov95. (Revision noise 0.034
  is *rejected*: it is in wILI percentage points, dimensionally inconsistent with a coverage ratio.)

Because the gate point estimates are ≤ 0 (ΔMAE −0.003 = an *improvement* on removal), a Branch-E result is
stated as **"the gate's marginal effect is bounded below the scale this paper accepts as a [metric]
contribution,"** not as "the gate has no effect."

## 2. Per-metric decision, using the 90% CI of the ablation Δ (= effect of REMOVING the component)
Evaluate ΔMAE and ΔWIS **independently** (per-metric split — a narrow ΔMAE CI does not license a claim about
ΔWIS):

**Burden of proof (boundary handling).** An immateriality/equivalence claim (Branch E, Branch R1) requires the
90% CI to lie *entirely* within (−SESOI, +SESOI). Any CI that reaches or crosses a ±SESOI boundary is treated as
extending beyond it → Branch I (if it includes 0) or Branch R2 (if it excludes 0). Partial overlap is never
immaterial — the burden is on the immateriality claim.
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

**Large effects (−Env, −Rollout) — reported as effect+CI and described comparatively, NOT gated by E/I/R.**
−Env is cleanly material: ΔMAE CI [+0.109, +0.241] and ΔWIS CI [+0.083, +0.153] lie *entirely* beyond their
SESOIs. −Rollout is the sole, directionally-certain calibration driver: ΔCov95 CI [−0.055, −0.019] excludes 0
and is the only component that moves Cov95 (−Env −0.002 NS, −gate +0.004); it is material at the point estimate
(0.036 > 0.026) **but its CI reaches into the sub-SESOI band** (inner bound 0.019 < 0.026, an n=5 width
property). It will be described precisely as such — sole calibration driver, point-material — **not** as
"cleanly material." (This straddle is the live case that motivates the burden-of-proof rule above.)

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
The contribution's substance is the localization onto two effects — **Env → accuracy** (ΔMAE +0.161, CI
entirely beyond SESOI: cleanly material) and **Rollout → calibration** (ΔCov95 −0.036, the sole Cov95 driver;
point-material, CI straddling SESOI_Cov95 per §2). The gate's null is a subordinate, bounded observation, not
the contribution's basis. No branch removes Contribution 3; branches change only how the gate's null is stated.

## 6. Naming (#4)
Decided after the table is recomposed. The effect-size table + bounded-null makes the text-handling defense
(gate = structural posterior→selectivity path; marginal effect bounded below the decision scale; retained for
architectural completeness) sufficient; a rename is optional, not required.

## 7. Artifact / timestamp
This file lives in the git-tracked `runs/` area (the `CGM_v2_paper/` LaTeX directory is gitignored by author
choice, so pre-regs needing a git timestamp belong here alongside `PRE_REGISTRATION.md`). The DRAFT was
committed before any CI computation; this LOCKED revision is a subsequent commit. Git history is the
timestamped forking-path artifact. (Pre-existing pre-regs under `CGM_v2_paper/` are NOT git-tracked — a
separate hygiene item if git timestamps are wanted for them.)
