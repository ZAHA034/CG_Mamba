# Pre-Registration — APMD Gaussian-Approximation Sensitivity (r>0.3 cells)

**Status:** LOCKED on 2026-08-04, before any computation.
**Rule:** Cell definition, predictive law, metrics, consistency gate, and reporting rule below are fixed
prior to running. No subset/metric/threshold re-selection after seeing results. Report regardless of
outcome (two-sided). If infeasible on the cached artifacts, regenerate via the actual eval pipeline; do
not substitute an inline re-implementation of APMD.

## 1. Motivation (fills a gap the manuscript states explicitly)

`method.tex` §142–152: APMD's Gaussian/mixture switch is evaluated on the **dataset-mean** ratio
`r = mean_{n,h}(σ²_between/σ²_total)`. Here `r = 0.03 ≤ 0.3`, so the **Gaussian form (Eq. 144) is used
for every sample**. The manuscript discloses that **3.6% of national test_strict sample-horizons have
per-sample r > 0.3 (up to 0.66 at h=4) and are Gaussian-approximated rather than routed to the
mixture-CDF inversion (Eq. 147), "whose per-sample effect we do not separately evaluate."**

This pre-registration evaluates exactly that per-sample effect: how much does Gaussian-approximating the
r>0.3 cells (instead of the exact mixture-CDF quantile) change WIS and Cov95?

## 2. Data / source (LOCKED)

- **National test_strict**, horizons h=1–4, raw native APMD — the paper's **headline** UQ (Cov95 0.993,
  WIS 0.399 nationally).
- Per-cell APMD components required: `μ_CGM[n,h]`, `γ_all[n,h,k]` (K=3 phase posterior), HMM emission
  `μ_k`, `σ²_k` (z-scored), target denorm `(mean, std)`, realized `y[n,h]` (raw). Reuse the canonical
  national APMD source used for the headline (`runs/regime_shift/per_origin_forecasts.parquet` /
  `e1_final` national eval). **No retraining.** If components are not fully cached, regenerate them with
  the actual `scripts/e1_final_eval.py` / `regime_shift_experiment.py` path (deterministic).

## 3. Predictive law (LOCKED — raw native, s=1)

The headline uses **no post-hoc scaling**, so `s_h = 1` for both arms. Both arms are computed with the
**actual code** in `src/eval/hmm_interval.py` — no re-implementation:

- **Baseline arm (what the paper reports):** `construct_quantiles(..., mode="gaussian")` →
  `gaussian_quantiles` (Eq. 144) for all cells.
- **Alternative arm:** `construct_quantiles(..., mode="mixture")` →
  `mixture_quantiles_per_sample` (Eq. 147, brentq inversion of `F_mix(y)=Σ_k γ_k Φ((y−μ_k)/σ_k)`,
  recentred by `μ_CGM−μ_HMM`) **applied only to the r>0.3 cells**; all other cells keep the Gaussian
  quantiles. (This isolates the 3.6% cells; splicing, not a global mode switch.)

Both produce the full 23 FluSight levels, denormalized to raw wILI.

## 4. Affected-cell definition (LOCKED)

Affected subset = sample-horizons with **per-sample `r_{n,h} = σ²_between[n,h] / σ²_total[n,h] > 0.3`**
(threshold is the manuscript's existing 0.3 — NOT re-tuned). Report the exact count and fraction
(expected ≈3.6%) and the per-horizon distribution (expected concentration at h=4, max r≈0.66).

## 5. Metrics (LOCKED — via `src/eval/wis`, 23 levels, same as the paper)

1. **On the affected subset (r>0.3 cells):** mean WIS and Cov95 under Gaussian vs mixture; report
   `ΔWIS = WIS_mix − WIS_gauss` and `ΔCov95 = Cov95_mix − Cov95_gauss`.
2. **Aggregate national test_strict:** overall WIS and Cov95 for (a) all-Gaussian (headline) vs
   (b) mixture-spliced-for-r>0.3-cells; report `ΔWIS_agg`, `ΔCov95_agg` (diluted by the 3.6%).
3. **Per-horizon** h=1–4 breakdown of both.

## 6. Consistency gate (LOCKED)

The all-Gaussian arm MUST reproduce the paper's headline national metrics within tolerance
**|ΔWIS| ≤ 0.005 and |ΔCov95| ≤ 0.010** (targets: WIS 0.399, Cov95 0.993). If not, the source/pipeline is
inconsistent with the headline → **halt and do not report deltas** (per
`feedback_automation_misses_ground_truth`; the deltas are only trustworthy if the baseline reproduces).

## 7. Reporting rule (LOCKED — fixed before results)

- Report §5.1–5.3 in full regardless of sign/magnitude. No cherry-picking of subset/metric/horizon; no
  threshold re-selection.
- **If small** (pre-set: `ΔWIS_agg < 0.005` AND `|ΔCov95_agg| < 0.005`): the approximation is validated →
  add a 1–2 sentence quantified bound to the manuscript (near §152 / §IV-6), full detail to the code repo.
- **If large:** report honestly as a caveat (and note that per-sample routing would remove it); do NOT
  hide it and do NOT add a favorable-only summary. The manuscript sentence, if added, states the measured
  effect verbatim.
- The added manuscript sentence (if any) must not increase the page count past 14 (attach to an existing
  sentence; detail to repo).

## 8. What this does NOT do

- No change to the headline UQ (raw native Gaussian form stays the reported method; this only *quantifies*
  the approximation it already makes).
- No re-tuning of the 0.3 threshold, no new calibration, no scaling.

## RESULTS (executed 2026-08-04, after lock; no re-specification)

**Consistency gate: PASS.** All-Gaussian arm reproduces the headline national metrics: WIS $0.40014$
(target $0.399$, |Δ|$=0.0011\le0.005$), Cov95 $0.99295$ (target $0.993$, within $0.010$). Deltas trusted.
Artifact: `runs/apmd_gaussian_sensitivity/result.json`.

**Affected cells reproduce the manuscript's disclosure:** $110/2980 = 3.69\%$ of national test_strict
sample-horizons have per-sample $r>0.3$ (manuscript states $3.6\%$); max $r=0.660$ (states $0.66$);
concentrated at long horizon (h1/h2/h3/h4 = 10/25/30/45).

**Effect of routing the r>0.3 cells through the exact mixture-CDF (Eq. 147) instead of the Gaussian form:**
- **Aggregate national test_strict:** Gaussian WIS $0.40014$ / Cov95 $0.99295$ → mixture-spliced WIS
  $0.40138$ / Cov95 $0.99295$. **ΔWIS $=+0.0012$, ΔCov95 $=0.000$.**
- **Affected 110-cell subset:** Gaussian WIS $0.3173$ / Cov95 $1.000$ → mixture WIS $0.3509$ / Cov95
  $1.000$. **ΔWIS $=+0.0336$, ΔCov95 $=0.000$** (the exact mixture is, if anything, marginally
  *worse*-scoring at identical, complete coverage).

**Verdict (per locked §7):** SMALL ($\Delta$WIS_agg $0.0012<0.005$ and $|\Delta$Cov95_agg$|=0.000<0.005$)
→ the dataset-mean Gaussian switch is empirically validated; the approximation neither harms coverage
nor is favorable-biased (it is marginally better-scoring than the exact mixture on the affected cells).
Action: replace the manuscript's "we do not separately evaluate" clause (§152) with the measured bound;
full per-cell analysis in the released code. No headline change.
