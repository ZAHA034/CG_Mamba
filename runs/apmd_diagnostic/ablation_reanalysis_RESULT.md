# Table IV ablation reanalysis — RESULT (executes PREREG_ablation_reanalysis.md, LOCKED 31c72d9 / AMEND 7780f93)

Source: `runs/ablation_retrain/ablation_retrain_results.json` (per_seed, 5 seeds 42/123/456/789/1024,
seed-matched paired). Method per §0: paired-t, df=4, 90% two-sided (t=2.131847). Cov95 = Scaled-APMD variant.
Validation: Full means MAE 0.392 / WIS 0.297 / Cov95 0.886 — exact match to Table IV. **Reference noise floor:
Full per-seed MAE 0.392 ± 0.020 (SD).**

## Branch verdicts (pre-registered)
| row | Δ | mean | 90% t-CI | SESOI | branch |
|---|---|---|---|---|---|
| −Phase gate | ΔMAE | −0.0025 | [−0.0209, +0.0159] | 0.038 | **E** (bounded equivalence) |
| −Phase gate | ΔWIS | −0.0003 | [−0.0090, +0.0083] | 0.023 | **E** (bounded equivalence) |
| −Phase gate | ΔCov95 | +0.0037 | [+0.0006, +0.0068] | 0.026 | **R1** (resolved-immaterial, 90%-only) |
| −Env | ΔMAE | +0.1609 | [+0.0752, +0.2465] | 0.038 | **R2** (material, clean) |
| −Env | ΔWIS | +0.1128 | [+0.0695, +0.1561] | 0.023 | **R2** (material, clean) |
| −Rollout | ΔCov95 | −0.0356 | [−0.0583, −0.0128] | 0.026 | **R2** (material, straddles SESOI) |

Raw per-seed differences (variant − full), order [42,123,456,789,1024]:
- gate ΔMAE: −0.0038, +0.0212, +0.0077, −0.0308, −0.0069
- gate ΔWIS: −0.0030, +0.0101, +0.0027, −0.0142, +0.0027
- gate ΔCov95: 0.0000, +0.0084, +0.0034, +0.0050, +0.0017  → **= 0, 5, 2, 3, 1 of 596 cells** (149×4)
- −Env ΔMAE: +0.1187, +0.3169, +0.1140, +0.0979, +0.1568
- −Env ΔWIS: +0.0784, +0.1889, +0.0976, +0.0809, +0.1181
- −Rollout ΔCov95: −0.0185, −0.0738, −0.0134, −0.0403, −0.0319

## Outcome vs pre-reg
- Gate accuracy null → **Branch E** (NOT the feared Branch I; hw 0.018 < 0.038). "NS" upgraded to bounded equivalence.
- Gate ΔCov95 +0.004 → **R1** exactly as pre-committed; report as resolved-but-immaterial (0–5 of 596 cells), NOT harm.
- −Env cleanly material (accuracy/sharpness driver).
- −Rollout ΔCov95 CI **excludes 0** → §5 conditionality NOT triggered; calibration-driver claim **holds**. But CI
  inner bound −0.0128 < SESOI 0.026 → **straddles** (recomputation deepened the straddle vs bootstrap −0.019;
  bootstrap was hiding marginality). Describe as sole driver, point-material, CI reaches into sub-SD band — not
  "cleanly material."
- **Contribution 3 survives intact** (no demotion, no reversal); gate null strengthened to equivalence.

## Sensitivity / honesty disclosures (to appear as footnotes/caption)
1. **seed-123 sensitivity (−Rollout).** Excluding seed 123: −Rollout ΔCov95 mean −0.036 → −0.026 (n=4 90% CI
   [−0.041, −0.012]); direction and 0-exclusion retained → material verdict holds, but the effect size rests
   substantially on one seed. **NOTE:** seed 123 is *not* an outlier in the Full reference runs (Full MAE
   42:0.386/123:0.372/456:0.405/789:0.420/1024:0.377 — the most-variable Full seed is 789); the large seed-123
   differences come from the ablated variants' training variability at n=5, not a Full-run outlier. State the
   sensitivity factually, without a Full-outlier attribution.
2. **gate ΔCov95 immateriality.** 0–5 of 596 test cells (per-seed 0/5/2/3/1); mean +0.004 resolved at 90%
   (CI includes 0 at 95%), below the model's own cross-region coverage SD (0.026).
3. **expected-effect vs realization (gate).** The equivalence bound is on the expected effect over training
   randomness; individual seed ΔMAE spans −0.031 to +0.021 (spread 0.052 > SESOI), transparent in per-seed values.
4. **−Env normality.** One of five paired differences (+0.317, seed 123) is ~2× the others; the t-interval assumes
   approximate normality, which the printed per-seed values let readers assess. Verdict (material) is robust to this.
