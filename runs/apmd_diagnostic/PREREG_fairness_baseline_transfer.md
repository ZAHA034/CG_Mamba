# PRE-REGISTRATION — Fairness baseline: does the μ-frozen residual-fit variance recipe fail zero-shot transfer on the STRONGEST independent backbone, and survive temperature corrections?

**Status:** LOCK candidate (all three final revisions applied). Frozen BEFORE any Vanilla-head or temperature number
is observed. On git-commit this file becomes the forking-path artifact; heads are fit + scored only after.

**AMENDMENT (audit trail, not protocol-shopping):** this **amends** locked
`CGM_v2_paper/PREREG_nll_quantile_head_ablation.md` **arm (a)** — changing its protocol from **joint end-to-end**
(μ+σ learned) to **μ-frozen** (Vanilla point forecast frozen, σ-head only). The amendment is made **before any arm
(a) number exists** (arm (a) has never been run); the **sole rationale is single-variable backbone isolation** —
swapping ONLY the backbone from the already-run b-primary, so no head-design or μ-training degree of freedom
confounds the comparison. The superseded joint protocol is **not deleted** but explicitly DEFERRED with a trigger
(§3.1).

---

## 0. Why this experiment, and the confirmation-bias hazard (declared up front)

**The reviewer attack it answers (convergent, 8-agent memoryless reject-review):** "A learned Gaussian-NLL variance
head *beats* APMD in-distribution on both calibration and sharpness (national 0.945/0.277 vs 0.993/0.399); APMD's
only edge is one zero-shot transfer number (0.868 vs 0.954), which is confounded." Fairness question: **is the
learned head's transfer-failure a property of CG-Mamba's own backbone (we crippled it), or does it generalize to a
standard, strong, independent baseline — and does it survive the scale corrections a practitioner would apply?**

**We have a stake in the outcome** (unlike the fixed-width control). Our prediction is transfer-failure; the
paper's kernel lives or dies here. Confirmation-bias risk is MAXIMAL. §3–§4 remove design/scoring discretion; §0.1
records what we know so no result is read favorably after the fact.

### 0.1 What we already know (numbers-blind context; basis for the prediction)
- **b-primary (CG backbone, DONE, LOCK-3 ELEVATE fired):** learned μ-frozen σ-head, train-residual fit, zero-shot
  regional Cov95 **0.868** vs APMD 0.955; in-distribution national learned **0.945**/0.277 vs APMD 0.993/0.399.
  Source: `runs/exp1b_learned_variance/summary.json`.

- **Confound-1 is a LIVE attack, and CONFIRM is not safe while it stands.** A reviewer computes: regional 0.868 for
  a nominal ±1.96σ interval implies half-width factor f with 2Φ(1.96f)−1 = 0.868 → **f = 0.768**, i.e. σ_pred is
  ~**23% too small**. The manuscript itself writes *"in-sample residuals run 16–49% smaller than held-out"*
  (`results_main.tex`). The ranges overlap, so the reviewer's move is exact: *"correct σ by the authors' own
  numbers and 0.868 rises toward nominal — the whole transfer-failure may be an in-sample residual-shrinkage
  artifact."* This must be foreclosed, not deferred.

- **The val-fit probe does NOT foreclose it (downgraded).** Fitting σ on **val (out-of-sample) residuals** gives
  learned regional **0.750** (`summary_fitval.json`) — lower, so the *direction* is robust to residual sourcing.
  But the probe is **contaminated** and cannot resolve confound-1: the σ-head is fit on ~75 samples (per-seed swings
  0.68–0.85, std 0.064), and val is **pre-COVID** while test is **post-COVID** (a distribution shift confounding the
  probe). It licenses only "direction robust," NOT "confound-1 resolved" or "train-fit is less favorable."

- **The REAL answer to confound-1 (pre-registered prediction + basis):** an in-sample residual shrinkage is
  ~**uniform** — it should depress national and regional coverage by the same factor. But the SAME σ-head yields
  national in-distribution f = 0.979 (Cov95 **0.945**, σ only ~2% small) versus regional f = 0.768 (~23% small). A
  **2%-vs-23% asymmetry is not shrinkage** (which early-stop already absorbed in-distribution) — it is
  **distribution shift** national→regional, exactly what APMD's frozen generative buffer covers and a residual-fit
  conditional variance does not. §3.2 tests this with TWO train-holdout temperatures. **Honest split of what we
  predict:** the **variance-matched (RMS)** temperature is predicted to give s_h ≈ national-calibrated ≈ 1.0–1.02 and
  NOT rescue regional (0.868 → ≈0.875); the **quantile-matched (q95)** temperature outcome is **genuinely uncertain**
  — because the residuals are leptokurtic (kurtosis ≈ 11, `discussion.tex`), Q95(|z|)/1.96 > 1, and depending on the
  national-holdout tail weight it could lift regional from CONF-side up to NEAR (worked estimate in §2). We do not
  pre-decide it; §4 requires BOTH temperatures to fail for CONFIRM.

- **Confound-2 (generative-vs-residual SOURCE) is OUT OF SCOPE.** This experiment isolates the **backbone** (and
  scale corrections), not the variance SOURCE. **No mechanism-source claim is made or moved by this run;** the
  source claim remains RETRACTED per the b-primary §7.1-CORRECTION. (Enforced in §4.1: CONFIRM must not attribute
  the result to "the generative scale.")

## 1. Hypothesis (pre-specified, scoped)
The **μ-frozen residual-fit Gaussian-NLL variance recipe** under-covers under zero-shot national→regional transfer
on **both** the strongest independent amortized backbone (Vanilla Mamba) **and** CG-Mamba; a **variance-matched**
train-holdout temperature does not rescue it (predicted s ≈ 1); whether a **quantile-matched** temperature rescues
it is **left open and tested** (§3.2, §4). Claim scope is the **recipe**, never "learned heads in general" (§3.1).
Falsifiable: if the Vanilla recipe — raw or under either temperature — transfers to near-nominal, the
transfer-advantage is not unique to APMD.

## 2. Quantitative prediction (before scoring)
- Vanilla learned regional Cov95 (raw) ≈ [0.80, 0.90], under-covering, ≥0.026 below CG's 0.954.
- **RMS temperature: no rescue predicted** (s ≈ 1.0–1.02 from the national-calibrated holdout; regional ≈0.875).
- **q95 temperature: outcome uncertain** — with leptokurtic holdout z, s^q95 ≈ 1.15 gives regional eff factor
  0.768×1.15 = 0.883 → Cov95 ≈ 2Φ(1.731)−1 ≈ **0.917** (just below the 0.920 CONF-edge); s^q95 ≈ 1.20 → **0.928** →
  NEAR → THREAT. So the q95 arm can genuinely flip the branch; that is the point of running it.

Stated only to pre-commit; the decision (§4) is on the realized numbers.

## 3. Design + DEGREES-OF-FREEDOM LOCK (every knob fixed, no post-hoc search)
- **Backbone (single, fixed):** Vanilla Mamba = the paper's baseline `CGMambaBackbone, use_gate=False`, config
  `d64_nl3_lr5e-04`, checkpoints `runs/vanilla_mamba_final/d64_nl3_lr5e-04/seed{42,123,456,789,1024}`. Rationale
  locked: nearest architecture (same backbone class, gate off) + strongest DL MAE baseline → blocks "weak opponent."
  No other baseline added or swapped.
- **(c) Checkpoint-integrity assert (BEFORE any head fit) — corrected to avoid spurious ABORT:** 0.435 is the
  **5-seed mean** of national `test_strict_avg`, NOT a per-seed value (per-seed spread is ~±0.03). So assert in TWO
  parts against the stored evaluation record `runs/baselines_test_eval.csv`
  (rows `model=vanilla_mamba, cfg_name=d64_nl3_lr5e-04`):
  1. **Per-seed:** the loaded checkpoint's national test_strict MAE (h1–4 avg) matches that seed's recorded
     `test_strict_avg` within tol **1e-3**. Reference values (verified present, 2026-08-08):
     seed42 0.4620, seed123 0.4068, seed456 0.4231, seed789 0.4183, seed1024 0.4655.
  2. **Mean:** the 5-seed mean equals the published **0.435 ± 0.005** (record mean = 0.4351 ✓).
  Either assert failing → ABORT (blocks the "degraded checkpoint" attack) without spurious per-seed failures.
- **Head (identical to b-primary — no redesign):** `LogVarHead = nn.Linear(d_model=64 → 4)` on `h_last =
  fused[:,-1,:]`; Gaussian-NLL z-space; μ = Vanilla's own **frozen** point forecast; logvar clamp `[-10,5]`;
  early-stop on a **train-internal 80/20 holdout only** (no val/test in the σ value). Byte-for-byte the b-primary
  recipe, backbone swapped → blocks "you designed the baseline's head unfavorably." Forward-consistency self-check
  required (same integrity assert exp1b uses).
- **Residual source (fixed):** **train split**, native, no calibration data — identical to b-primary and fair vs
  APMD's no-calibration-data property.
- **HPO (frozen):** the Vanilla checkpoints' existing config; no new search, no coverage tuning.
- **Seeds (fixed):** {42,123,456,789,1024}; none dropped; non-convergence reported, not rescued.
- **Eval (identical harness):** `test_strict` (epiweek ≥ 202240), 23 FluSight quantiles, WIS+Cov95 via `track_b_lib`,
  10 HHS regions zero-shot + national z-score. Comparison target = **RAW native APMD** (0.954), never Scaled APMD.
- **Reporting (anti-hiding):** per-horizon + per-region + national, in-distribution AND zero-shot, for Vanilla-learned
  (raw + both temperatures) AND APMD; reported regardless of direction.

### 3.1 Joint end-to-end arm — explicit DEFER + trigger (not "optional")
The superseded joint protocol (μ+σ learned end-to-end) is DEFERRED. **Acknowledged residual attack surface:** a
reviewer may say "a *jointly trained* distributional model would transfer where a μ-frozen head does not." We do not
pre-empt that; we **scope every claim sentence to the μ-frozen residual-fit recipe** (§4.1) and set a trigger:
**run the joint arm iff (THREAT fires) OR (a reviewer requests it).** Under the joint arm the same 3-way rule (§4)
applies, decided on its own numbers.

### 3.2 Temperature-scaled arms (NEW — TWO formulas pinned now to forbid knob-shopping; both backbones)
On the **train-internal 80/20 holdout** used for early-stop (national; no val/test → the no-calibration-data
property is preserved), with standardized holdout residuals z_h = resid_holdout,h / σ_pred_holdout,h, compute **two
per-horizon scalar temperatures** and score a variant for each (σ'_h = s_h · σ_pred,h; everything else identical):
- **Variance-matched (RMS):** `s_h^RMS = RMS(z_holdout,h)` — matches predictive variance (z' unit-variance).
- **Quantile-matched (q95):** `s_h^q95 = Q95(|z_holdout,h|) / 1.96` — targets 95% coverage directly; **same
  philosophy as the manuscript's own Scaled APMD** (in-paper consistency), and under leptokurtic residuals it is the
  formula that FAVORS the learned head (Q95(|z|) > 1.96 → wider). Including it forecloses the reviewer's
  "you used the variance-matched temperature because it favors you" rebuttal.

Both formulas are fixed here; no alternative temperature is tried. Applied to **Vanilla AND a re-score of b-primary
(CG backbone)** — both reported. Seconds/minutes; no retrain.

## 4. Decision rule — 3-WAY, bands LOCKED before numbers (exact, non-overlapping)
**Decision metric:** learned **native regional h1–4 avg Cov95, zero-shot**, for **raw AND both temperature variants
(RMS, q95)**. **τ = 0.026** (paper-wide SESOI_Cov95; not re-chosen). d_CG = |0.954−0.95| = 0.004. Per-arm band of a
value x: `CONF-side x ≤ 0.920 | GAP 0.920 < x < 0.924 | NEAR 0.924 ≤ x ≤ 0.976 | OVER x > 0.976`.
(0.920 = 0.95 − (d_CG+τ); [0.924, 0.976] = 0.95 ± τ.)

| Branch | Condition (Vanilla, zero-shot regional Cov95) | Meaning |
|---|---|---|
| **CONFIRM** (predicted) | **raw AND every temperature variant (RMS, q95) CONF-side** (all ≤ 0.920) | μ-frozen recipe under-covers on both backbones AND survives BOTH temperature corrections → inflation/shrinkage rebuttal foreclosed |
| **THREAT** | **raw or any temperature variant NEAR** (∈ [0.924, 0.976]) | a standard recipe (or a coverage-targeting temperature) transfers near-nominal → drop transfer-uniqueness, adopt "among" |
| **PARTIAL** | any variant GAP and not THREAT/BOUNDARY (e.g. some variant ∈ (0.920,0.924)) | directional under-coverage within margin → report directional, no resolved claim |
| **BOUNDARY** | raw or any variant OVER (> 0.976) and not THREAT | over-covers far → reported separately, NOT counted as CONFIRM |

Precedence when arms disagree: **THREAT > BOUNDARY > PARTIAL > CONFIRM** (any NEAR ⇒ THREAT; else any OVER ⇒
BOUNDARY; else any GAP ⇒ PARTIAL; else all CONF-side ⇒ CONFIRM). No band overlaps; every (raw, RMS, q95) triple
maps to exactly one branch. The branch is not reversible post-hoc.

## 4.1 Per-branch headline sentence, PRE-WRITTEN + SCOPE-GUARDED (numbers filled after scoring)
- **CONFIRM:** "A μ-frozen residual-fit Gaussian-NLL variance head — the identical recipe — under-covers under
  zero-shot regional transfer on both the strongest independent baseline (Vanilla Mamba, Cov95 [X]) and CG-Mamba
  (0.868), and **neither a variance-matched nor a quantile-matched train-holdout temperature rescues it**
  ([X_RMS]/[X_q95]); recalibration-free near-nominal transfer is therefore **not an artifact of CG-Mamba's backbone
  and not recoverable by rescaling the learned variance.**" — sentence STOPS here; **no attribution to "the
  generative scale"** (that is confound-2, out of scope).
- **THREAT:** "A μ-frozen residual-fit head (or its train-holdout temperature correction) on a standard amortized
  baseline (Vanilla Mamba) attains near-nominal zero-shot regional coverage (Cov95 [X]); recalibration-free
  multi-region calibration is therefore achievable without APMD, so CG-Mamba is *among* the amortized forecasters
  that reach it — its distinctive properties are the analytic, phase-decomposable interval and the tightest
  cross-region/-horizon coverage stability, not uniqueness of native transfer calibration."
- **PARTIAL:** "A μ-frozen head on Vanilla Mamba is partially competitive on zero-shot transfer (Cov95 [X],
  directionally under-covering but within the 0.026 margin of CG-Mamba); CG-Mamba's transfer-calibration advantage
  is reported as directional, not resolved."

## 5. Honest limitations (stated regardless of branch)
- Isolates **backbone + scale corrections**, not the variance **SOURCE** (confound-2); no mechanism-source claim.
- Scope is the **μ-frozen residual-fit recipe**; the **joint end-to-end** arm is deferred (§3.1) — a residual attack
  surface we name rather than hide.
- Confound-1 is addressed by the two temperature arms (§3.2, the q95 one deliberately favoring the learned head under
  the paper's own leptokurtosis) + the national-vs-regional 2%/23% asymmetry argument (§0.1), NOT by the contaminated
  val-fit probe.
- Seeds = optimization variance, not forecasting observations; inference unit = regions; descriptive stats + the
  10-region sign test only, no manufactured significance.

## 6. Artifact / timestamp
Git-committed to tracked `runs/apmd_diagnostic/` BEFORE scoring; git history is the forking-path / bias artifact.
New script `scripts/exp1a_vanilla_distributional_head.py` (mirrors `exp1b_learned_variance_head.py`: Vanilla loader +
two-part checkpoint-MAE assert against `runs/baselines_test_eval.csv`, μ-frozen σ-head, RMS + q95 temperature
variants, forward-integrity self-check); re-score b-primary (CG) under both temperatures in the same run; outputs to
`runs/exp1a_vanilla_distributional/`.

---

## AMENDMENT 2 (2026-08-08) — hollow-CONFIRM gate + joint-arm design pin

**Transparency:** written AFTER observing the SMOKE run (seed 42 only: Vanilla μ-frozen head national **in-dist
Cov95 = 0.738** vs CG 0.945; Vanilla s^RMS/s^q95 all < 1) but BEFORE the full 5-seed locked decision numbers exist.
Like the q95 addition, this amendment moves ONLY in the **burden-increasing** direction — it adds a CONFIRM
disqualification and adds the joint arm; it never loosens a locked band or the §4 branch computation. The threshold
below reuses a locked bound → **zero new numeric degrees of freedom**.

**Interpretation reframe (from the smoke, stated now so it is not post-hoc):** Vanilla's in-dist failure is NOT
merely "the experiment got weaker." The stronger reading: the identical recipe on an independent backbone is worse
*in-distribution*, so CG's **0.868 was the recipe's BEST case** — transfer failed on the one backbone where the
recipe is even in-dist-calibrated (CG, 0.945). Role split, fixed now: the clean "a well-calibrated learned head
fails to transfer" evidence is **CG b-primary itself** (0.945→0.868); the **Vanilla arm** is the supporting
evidence that the recipe is **backbone-fragile and that we did not pick a head-favorable backbone**. Together they
answer the original attack ("you only tried your own backbone"); the stronger residual attack ("a *properly trained*
distributional baseline would transfer") is closable only by the joint arm (A2.2).

### A2.1 hollow-CONFIRM gate (interpretation gate on §4 — NOT a change to the band math)
The §4 branch is computed exactly as locked. ADDITIONAL one-directional condition on the CONFIRM *interpretation*:
- Even when §4 computes BRANCH=CONFIRM, the §4.1 CONFIRM sentence is **licensed only if Vanilla raw national
  in-distribution Cov95 (5-seed mean) ≥ 0.924** (= the locked NEAR lower bound 0.95−τ; natural definition "a head
  qualifies to carry a transfer test iff it is itself within τ of nominal in-distribution"; no new number invented).
- If that in-dist Cov95 < 0.924 (smoke ⇒ near-certain to fire): the §4.1 CONFIRM sentence is **replaced** by the
  two-axis narrative (reframe above) and the **joint arm (A2.2) auto-triggers**. This gate can only DOWNGRADE a
  CONFIRM to the two-axis narrative; it can never upgrade any branch.

### A2.2 joint end-to-end arm — DESIGN PINNED (triggers per §3.1 THREAT, or per A2.1)
- **Design:** the Vanilla Mamba architecture + a parallel LogVar head, μ and σ trained **jointly** end-to-end under
  Gaussian-NLL. Loss is swapped (point-MSE → Gaussian-NLL on a μ+logσ² head) AND the model-selection / early-stop
  metric switches from **val-MAE@h1 → val-WIS** (the distributional analogue). All remaining config identical to the
  original Vanilla point model (Adam, lr, epochs=200, patience=20, batch=32, data, same 5 seeds). Cost: 5 retrains of
  the ~108K-param model.
  **A2.2-refinement (2026-08-08, pre-joint-result, burden-INCREASING):** the original A2.2 shorthand "only the loss
  is swapped" would, taken literally, keep val-MAE@h1 selection — which selects the *distributional* baseline on
  *point* accuracy, crippling it in our favor (self-serving). val-WIS selection is the FAIRER, harder-for-us choice
  (the distributional baseline is selected on a distributional criterion) and it **restores the parent lock
  `PREREG_nll_quantile_head_ablation.md` arm (a)** spec ("selection switches from val-MAE@h1 to val-WIS"), so it
  introduces **zero new degrees of freedom**. μ is thus selected for WIS, not MAE → μ-drift is expected and reported
  (below).
- **Honest role (must be stated in the manuscript):** joint changes TWO variables at once (backbone AND training
  protocol) → it is **NOT** the 1-variable isolation experiment. Its role is the direct answer to the residual
  attack *"does a properly trained distributional baseline transfer?"* — never blended with the μ-frozen result.
- **μ-drift reporting duty:** joint training changes μ. Report the jointly-trained model's **national MAE** alongside
  — the point-accuracy cost of the NLL objective is itself informative; hiding it is an attack surface.
- **Verdict:** the SAME §4 bands on joint's zero-shot regional transfer Cov95 (raw). Pre-written branch sentences:
  - **joint CONF-side (≤0.920) AND joint in-dist ≥0.924:** "Even a jointly-trained distributional head,
    in-distribution-calibrated on the strongest baseline (Cov95 [X_indist]), under-covers under zero-shot regional
    transfer (Cov95 [X]); recalibration-free near-nominal transfer is not achieved by a properly-trained learned
    distributional baseline either."
  - **joint NEAR [0.924,0.976]:** "A jointly-trained distributional head on Vanilla Mamba attains near-nominal
    zero-shot transfer (Cov95 [X]); recalibration-free multi-region calibration IS achievable by a properly-trained
    learned baseline, so CG-Mamba is *among* such models" (uniqueness dropped, 'among' repositioning as §4.1 THREAT).
  - **joint FAILS in-dist (national Cov95 < 0.924):** "No learned-variance path we evaluated — μ-frozen or jointly
    trained — achieved in-distribution calibration on the independent baseline; this supports the backbone-fragility
    of learned residual-fit UQ, while leaving open, as a stated limitation, that some untried recipe might."
  - PARTIAL / BOUNDARY as in §4.

### A2.3 temperature-<1 interpretation (reporting note, decided in advance)
The smoke shows Vanilla s^RMS, s^q95 all < 1 (holdout σ looked too large), so the temperature "correction" *worsens*
Vanilla transfer (0.738 → 0.644 / 0.609). This is the pinned formula operating honestly. Fixed meaning: a
train-internal holdout does not reflect the test-era residual scale, so holdout-based recalibration back-fires —
itself an ADDITIONAL failure mode of the residual-fit recipe, NOT a defect of this experiment's design.

**Artifact:** committed to tracked `runs/apmd_diagnostic/` as an amendment BEFORE the full 5-seed numbers exist; the
joint-arm script `scripts/exp1a_joint_vanilla_head.py` is written and run only when A2.1/§3.1 triggers.

---

## 7. RESULTS (appended 2026-08-08, after the full 5-seed run; numbers-blind boundary = git 3544a77 lock + 70e8506 amend)

**Harness integrity:** the two-part checkpoint-MAE assert (§3(c)) passed for ALL 5 seeds (summary.json only writes if
every assert passes; seed42 computed 0.46202 vs record 0.46202, 1e-7). CG b-primary raw reproduces exp1b
bit-faithfully (5-seed regional 0.868; per-seed 0.832/0.825/0.919/0.919/0.844). `vanilla_forward_with_hlast` asserted
equal to `model(x)`.

**§4 BRANCH (Vanilla zero-shot regional Cov95 h1–4 avg, 5-seed mean):** raw **0.789** / RMS **0.754** / q95 **0.736**
— all ≤ 0.920 (CONF-side), no NEAR → **BRANCH = CONFIRM** (as computed by locked `decide()`).

**A2.1 hollow-CONFIRM gate:** Vanilla raw NATIONAL in-dist Cov95 (5-seed mean) = **0.8919 < 0.924** → **GATE FIRES**
→ the §4.1 CONFIRM sentence is NOT licensed → the **two-axis narrative is adopted** and the **joint arm (A2.2)
auto-triggers**.

**Per-seed structure (reported per LOCK anti-aggregation; it STRENGTHENS the two-axis reading):**
| seed | Vanilla in-dist (national raw) | Vanilla transfer (regional raw) |
|---|---|---|
| 42   | 0.738 | 0.724 |
| 123  | 0.956 | 0.798 |
| 456  | 0.973 | 0.821 |
| 789  | 0.958 | 0.807 |
| 1024 | 0.834 | 0.797 |

→ On the **3 seeds where Vanilla IS in-dist-calibrated** (123/456/789, ≥0.924) the identical recipe STILL
under-covers transfer (~0.80) — extending the clean "a well-calibrated learned head fails to transfer" evidence
BEYOND CG b-primary. The **2 low-in-dist seeds** (42/1024) show the recipe is backbone-fragile. The 5-seed mean
(0.892) is dragged below the 0.924 gate by those two → gate fires on the locked mean rule (applied as locked; the
per-seed split is reported, not used to re-slice the decision).

**Temperatures foreclose the inflation/shrinkage rebuttal on BOTH backbones (the confound-1 answer):** neither a
variance-matched (RMS) nor a coverage-targeting (q95) train-holdout temperature lifts transfer to nominal — Vanilla
*worsens* (0.789 → 0.754 / 0.736; s<1, per A2.3), CG b-primary stays under (0.868 → RMS 0.877 / q95 0.864, both
< 0.920). The q95 arm — pre-registered as the one that could flip to THREAT under leptokurtosis — did **not** flip.

**Adopted (numbers-blind wording now instantiated):** the two-axis narrative (A2.1). No manuscript edit is made until
the joint arm (A2.2) completes and the user directs the reflection. b-primary's mechanism-source claim remains
RETRACTED (confound-2 out of scope).

**Artifacts:** `runs/exp1a_vanilla_distributional/{summary.json, result_seed{42,123,456,789,1024}.json, full_run.log}`;
`scripts/exp1a_vanilla_distributional_head.py`.
