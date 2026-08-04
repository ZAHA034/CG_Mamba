# APMD variance-consistency diagnostic — PRE-REGISTRATION (locked before results)

**Purpose (#4):** test the load-bearing assumption behind APMD's validity —
Var(y − μ_CGM) ≈ σ²_total (the neural forecast's error variance is captured by the
frozen-HMM phase-mixture total variance) — per horizon and per phase.

**Computed (raw wILI units), n3_d64 headline model, test_strict, national + 10 regions, 5 seeds:**
- per sample: μ_CGM, σ²_total, σ²_within, y_true, dominant phase k* = argmax γ_t.
- per horizon h: R_h = Var(y − μ_CGM)_h / mean(σ²_total)_h  (empirical / analytic).
- per phase k: Var(y − μ_CGM | k*=k)  vs  HMM emission variance σ²_k.

**Decision rule (LOCKED — applied regardless of outcome):**
- The assumption is **empirically consistent** at horizon h iff R_h ∈ [0.7, 1.5].
- If R_h rises above 1.5 as h grows → report as **"consistent at short horizons, under-estimating
  at longer horizons"** (predicted by the horizon-static σ²_total; corroborated by s_h(h) growth).
- If R_h < 0.7 anywhere → APMD **over-estimates** there (over-coverage), report as such.
- **Wording licensed by this diagnostic:** "empirically consistent / degrades with horizon" ONLY.
  It does **NOT** license "calibrated by construction" — that phrasing is a separate, professor-gated
  change (part of the #4 cluster). The diagnostic only tells us which empirical wording is honest.
- Report the full R_h vector and per-phase table regardless of direction. No threshold is moved
  after seeing results.

**Sanity:** national CGM over-covers (Cov95 0.993) → expect R_h < 1 at short h nationally
(σ²_total too large); regionally near-nominal → expect R_h ≈ 1 at short h, rising with h.
