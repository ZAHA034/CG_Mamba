# Fixed-width control — RESULT (executes PREREG_fixed_width_control.md, LOCK e74a555)

Source: `runs/apmd_diagnostic/apmd_residuals.csv` (regional). No retrain. High-variance cell = forecast-origin
dominant phase = highest-emission-variance state (σ²=1.907; 59% of regional cells), per §IV-C2. Decision metric
(locked): signed high-variance-cell Cov95 gap, under-covering direction, threshold 0.026; WIS/miss = corroboration.

## Scales
- V1 (stationary HMM scale, √mean σ²_dom): sd 1.082, half-width 2.121. [pre-reg estimate was ~1.03; actual 1.08 → V1 wider than estimated]
- V2 (national residual 95th-pct constant; residual-fit → transfer test; the stronger competitor): half-width 1.704.

## High-variance (surge) cells — Cov95
| h | APMD | V1 | V2 | gap V1 (APMD−V1) | gap V2 |
|---|---|---|---|---|---|
| h=3 | 0.921 | 0.875 | 0.832 | +0.046 | +0.089 |
| h=4 | 0.891 | 0.835 | 0.796 | +0.055 | +0.094 |
| all | 0.943 | 0.908 | 0.875 | +0.035 | +0.068 |
- Boundary: h=1 (APMD 0.998) and h=2 (APMD 0.960) — APMD over-covers → EXCLUDED from the under-coverage verdict (reported separately: at short horizons both APMD and the constants over-cover, APMD less so).
- 2-regime (low-var cells): APMD 0.970 vs V1 0.999 / V2 0.997 (constants over-cover low-activity; APMD adapts).
- Aggregate (corroboration): APMD 0.954 / V1 0.945 / V2 0.925 — near-indistinguishable; the effect is decided in the stratified surge cells, not in aggregate (as pre-registered).
- Miss-penalty (corroboration, high-var): APMD total miss 0.057 vs V1 0.092 (V1 more misses, esp. y>upper 0.067 vs 0.042).

## VERDICT (locked §4): APMD-WINS (predicted branch confirmed)
In the under-covering surge cells (h=3, h=4), APMD under-covers materially less than BOTH constants — gap ≥ 0.026 in every case, correct sign (both constants below nominal and more below than APMD). Beating V2 (the residual-fit constant, the strongest fixed width) by MORE (0.089–0.094) than V1 blocks the "weak constant" rebuttal; V2's poor transfer (aggregate 0.925) is consistent with §IV-D's residual-fit-doesn't-transfer finding.

**Mechanism claim retained + strengthened:** the phase posterior's γ-weighting of the 18×-spread emission variances materially reduces surge under-coverage vs any constant scale. This is the width-driver clarification (width varies via γ-weighted σ²_within, CV 52%; σ²_between's 2.5% is a separate small term) — rebutting the "static/decorative width" reading.

## Winning-branch narrative (locked; applied to prevent over-sell)
Even on this win, APMD is 0.891 at high-variance h=4 — still under-covering the surge. The claim is strictly
**relative** (phase-adaptivity *reduces* surge under-coverage: −0.055 vs V1, −0.094 vs V2), **not absolute** (0.891
is not adequate). The residual absolute under-coverage at long horizons is exactly what a **horizon-adaptive native
scale** would fix — this result is the direct motivation for that extension, not a claim the interval suffices in surges.

## Honest caveats
- V1 sd actual 1.082 > pre-reg estimate 1.03 → V1 under-covers less than the σ-calc predicted (0.835 vs ~0.77 at h4); verdict unchanged (gap still ≥ 0.026) but the margin is smaller than the rough prediction.
- Coverage proportions over autocorrelated stratum cells (pooled over regions), reported descriptively; no test, no multiplicity adjustment (estimates, not tests).
