# CG-Mamba — Data Pipeline (Phase 1 M1.1)

National weekly **ILI + environment** dataset for the IEEE JBHI submission.
Outputs a single MMWR-aligned dataset with train/val/test boundary metadata, fitted
with a train-only StandardScaler. No tokens or accounts required.

PLAN reference: `CG_Mamba_PLAN.md` v2.0.5 §4 Data.

---

## Layout

```
data/
  raw/
    cdc_ilinet/
      national_weekly.csv         # CDC FluView (Delphi API)
      MANIFEST.json
    noaa_isd/
      {isd_id}/{year}.csv         # 10 stations × 25 years = 250 files (~1.8 GB)
      MANIFEST.json
  interim/
    noaa_daily/
      {isd_id}_daily.parquet      # per-station daily aggregates
  processed/
    env_national_weekly.csv       # NOAA pop-weighted national weekly
    env_weekly_MANIFEST.json
    ili_env_weekly.csv            # CDC ⨝ NOAA on epiweek (final 1,229 rows)
    ili_env_weekly_MANIFEST.json
    ili_env_weekly_split.csv      # + 'split' column (train/val/covid_excluded/test)
    split_boundaries.json         # epiweek boundaries spec
    normalization_params.json     # train-only mean/std per column
  PROVENANCE.json                 # ← top-level data provenance, citations, license
```

---

## Sources

| Source | Records | License | Auth | Citation |
|---|---|---|---|---|
| CDC FluView ILINet (national) | 1,233 weekly | Public domain (17 USC §105) | none | Delphi Epidata API |
| NOAA NCEI ISD hourly | 10 stations × 25 yr | Public domain (17 USC §105) | none | Smith, Lott & Vose (2011) |
| US Census 2020 (MSA pop) | 10 MSAs | Public domain | none | US Census Bureau (2021) |

See `PROVENANCE.json` for full station table, formula citations, and pipeline detail.

---

## How to rebuild from scratch

Requires Python 3.11+, `pandas`, `numpy`, `pyarrow`, `requests`, `epiweeks`.

```bash
# 0. (optional) credentials — default pipeline does NOT need any
cp config/credentials.template.json config/credentials.json   # leave placeholders

# 1. Download CDC FluView national weekly ILI (~100 KB)
python -m src.data.download_cdc_fluview

# 2. Download NOAA NCEI ISD hourly  (~1.8 GB, ~10 min @ 5 workers)
python -m src.data.download_noaa_isd

# 3. Parse ISD hourly -> daily (TMP+DEW, QC filter, Bolton 1980 q_g)
python -m src.data.parse_isd

# 4. Daily -> MMWR weekly, population-weighted national mean
python -m src.data.build_env_weekly

# 5. Merge CDC ILI + env weekly on epiweek
python -m src.data.build_merged_weekly

# 6. Assign split labels + fit train-only scaler
python -m src.data.build_splits

# 7. Validate end-to-end (9 tests)
python -m src.data.validate_dataset
```

---

## Final dataset (`ili_env_weekly_split.csv`)

| Column | Type | Source | Role |
|---|---|---|---|
| `date` | str (YYYY-MM-DD) | CDC = NOAA (verified equal) | MMWR Sunday |
| `year`, `week`, `epiweek` | int | both | join key |
| `ili_weighted_pct` | float | CDC FluView | **target** (%wILI) |
| `ili_unweighted_pct` | float | CDC | auxiliary |
| `total_ili_count` | int | CDC | auxiliary |
| `num_providers` | int | CDC | denominator |
| `num_patients` | int | CDC | denominator |
| `temperature_c` | float | NOAA pop-weighted | **predictor** |
| `specific_humidity_g_per_kg` | float | NOAA pop-weighted (Bolton 1980) | **predictor** |
| `n_stations_available` | int | NOAA diagnostic | always 10 in current build |
| `weight_sum_raw` | float | NOAA diagnostic | always 1.0 in current build |
| `split` | str | `build_splits.py` | `train` / `val` / `covid_excluded` / `test` |

| Stat | Train | Val | Test | COVID-excl |
|---|---|---|---|---|
| Rows | 868 | 75 | 257 | 29 |
| Epiweek | 200140 – 201839 | 201840 – 202010 | 202040 – 202535 | 202011 – 202039 |
| Seasons | 16 full + 2001-02 partial | 2018-19 + 2019-20 truncate | 4 full + 2024-25 partial | — |

---

## Known limitations (see `PROVENANCE.json` → `known_limitations`)

### CDC 2002 summer reporting gap
CDC FluView (via Delphi API) does **not report ILI for W21 ~ W39 of 2002** (19
weeks missing). This is a **one-off anomaly** — year-round reporting is
consistent from 2003 onward.

- Train = 868 rows = **16 full seasons (2002-03 ~ 2017-18)** + **2001-02 partial
  (33 weeks: W40-2001 ~ W20-2002)**. 2001-02 is the only partial training season.
- **Exactly 1 epiweek gap in train**: `200220 → 200240` (W20-2002 → W40-2002,
  19-week jump).
- **Loader (M1.3) implication**: one cross-gap sliding window — either skip
  across the gap or apply a begin-offset at W40-2002. Single gap → simple.
- `validate_dataset.py` T10 asserts the train gap set equals exactly
  `[(200220, 200240)]` and that val/test/covid_excluded are contiguous.

### NOAA NCEI publish lag for 2025
NCEI ISD lags real time by **~6-9 months** for QC + aggregation. As of 2026-05-17
the latest hourly observation across all 10 stations is **2025-08-27**.

- Test = 257 rows = 4 full seasons (2020-21 has W53 → 53 weeks) + **2024-25 partial (48 of 52 weeks)**
- W36-2025 ~ W39-2025 (4 CDC rows) drop in the inner-join because NOAA hasn't published yet
- **Rebuild plan**: pipeline is idempotent — once NCEI publishes more 2025 data,
  re-run `download_noaa_isd → parse_isd → build_env_weekly → build_merged_weekly → build_splits`
  to pick up the additional weeks. Suggested cadence: re-check every 2-3 months.

---

## Split design (PLAN v2.0.5 §4)

**Single dataset + boundary metadata** (not 3 physical files):
- Loaders filter on `split` column or use the epiweek ranges from
  `split_boundaries.json`.
- Lookback windows at val/test boundaries read **predictors only** from
  preceding rows — labels are NOT used.

**COVID handling:**
- Val truncated at MMWR 2020-W10 (just before the pandemic disruption).
- `covid_excluded` (W11-2020 ~ W39-2020) is held out — neither val nor test.
- Test starts at W40-2020. Reporting tables show two rows:
  - **Test full** — all 257 test rows (includes COVID-era 2020-21 season).
  - **Test w/o COVID** — subset of 204 rows in `[202140, 202535]` (4 partial seasons 2021-22 ~ 2024-25).

**Scaler:** `StandardScaler` (mean, population std) fit on **train only**.
Saved to `normalization_params.json` with `fit_n_rows = 868` (verified by
`validate_dataset.py` T8 — recomputed mean/std must match saved params).

---

## Credentials

The default pipeline works with **zero tokens** (all sources are public-domain
bulk HTTPS).

Optional tokens (for higher rate limits / future use) are documented in
`docs/CREDENTIALS.md`. Schema is `config/credentials.template.json`; real
`credentials.json` is gitignored.

---

## Validation status

All 9 tests in `src/data/validate_dataset.py` pass:

```
✅ T1 MMWR Sunday alignment between sources
✅ T2 no duplicate / monotonic epiweeks
✅ T3 no NaN in target + env predictors
✅ T4 plausible value ranges (ILI %, T °C, q g/kg)
✅ T5 split boundaries non-overlapping + complete
✅ T6 row counts sum (868 + 75 + 29 + 257 = 1229)
✅ T7 test_post_covid ⊂ test
✅ T8 scaler train-only fit verification
✅ T9 SHA256 vs manifests (env weekly, merged)
```
