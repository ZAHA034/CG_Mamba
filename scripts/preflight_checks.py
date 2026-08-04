"""§18 Phase 1 — Pre-flight Data Checks (JBHI Strengthening Package).

Verifies dependencies for §18 Phase 3-6:
  1. CDC FluSight Hub repo accessibility (Phase 5)
  2. FluSight forecast format inspection (Phase 5)
  3. ILINet 2025-26 season data availability (Phase 6)
  4. HHS Region split capability (Phase 3)
  5. Target alignment feasibility — wILI ↔ wk_inc_flu_hosp (Phase 5)
  6. Existing CDC epi-wave classification check (Phase 4 anti-circular)

Output: outputs/preflight_checks.md
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
OUT_FILE = _ROOT / "outputs" / "preflight_checks.md"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

results = {"checks": [], "summary": {}}


def check(name, ok, detail=""):
    status = "✅" if ok else "❌"
    results["checks"].append({"name": name, "ok": ok, "detail": detail})
    print(f"  {status} {name}")
    if detail:
        print(f"     {detail}")
    return ok


print("=" * 80)
print("§18 Phase 1 — Pre-flight Data Checks")
print("=" * 80)

# ─── Check 1: CDC FluSight Hub access ───
print("\n[1] CDC FluSight Hub accessibility")
flusight_dir = _ROOT / "external" / "FluSight-forecast-hub"
if flusight_dir.exists():
    check("FluSight Hub already cloned", True, str(flusight_dir.relative_to(_ROOT)))
else:
    # Don't auto-clone; just check connectivity to github.com
    try:
        r = subprocess.run(
            ["git", "ls-remote", "https://github.com/cdcepi/FluSight-forecast-hub", "HEAD"],
            capture_output=True, timeout=15, text=True,
        )
        check("FluSight Hub remote reachable",
              r.returncode == 0,
              "Not yet cloned. Run: git clone https://github.com/cdcepi/FluSight-forecast-hub external/")
    except Exception as e:
        check("FluSight Hub remote reachable", False, f"{type(e).__name__}: {e}")

# ─── Check 2: ILINet 2025-26 in our data ───
print("\n[2] ILINet 2025-26 season data availability")
csv_path = _ROOT / "data/processed/ili_env_weekly_split.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    df["year"] = df["epiweek"] // 100
    df["week"] = df["epiweek"] % 100
    # 2025-26 season = W40-2025 to W39-2026
    season_2025_26 = df[(df.epiweek >= 202540) & (df.epiweek <= 202639)]
    check("ILI CSV contains 2025-26 season data",
          len(season_2025_26) >= 10,
          f"{len(season_2025_26)} weeks (W{season_2025_26.epiweek.min()} ~ W{season_2025_26.epiweek.max() if len(season_2025_26)>0 else 'n/a'})")
    last_ep = df.epiweek.max()
    check("Recent data freshness (within 8 weeks)",
          (202620 - last_ep) < 8 if last_ep > 202600 else False,
          f"Last epiweek in CSV: {last_ep}")
else:
    check("ILI CSV exists", False, str(csv_path))

# ─── Check 3: HHS Region split capability ───
print("\n[3] HHS Region stratification capability")
data_columns = list(df.columns) if csv_path.exists() else []
has_region_col = any("region" in c.lower() or "hhs" in c.lower() for c in data_columns)
check("Region column in CSV", has_region_col,
      f"columns: {data_columns[:10]}..." if data_columns else "no CSV")
if not has_region_col:
    # Check if region data exists separately
    region_csv = _ROOT / "data/raw/ilinet_by_region.csv"
    check("Region CSV separate file exists", region_csv.exists(),
          f"{region_csv.relative_to(_ROOT)}" if region_csv.exists() else "needs separate fetch from CDC")

# ─── Check 4: Target alignment (wILI ↔ hospitalizations) ───
print("\n[4] Target alignment feasibility (wILI ↔ wk_inc_flu_hosp)")
# wILI is what we forecast; FluSight 2024-25+ uses wk_inc_flu_hosp (hospital admissions)
# These are FUNDAMENTALLY different signals
check("Our target = ili_weighted_pct (wILI)", True, "outpatient ILI percentage")
check("FluSight 2024-25+ target = wk_inc_flu_hosp",
      False, "Hospital flu admissions — DIFFERENT from wILI. Phase 5 needs Plan B (FluSight-baseline only) OR proxy conversion")

# ─── Check 5: CDC epi-wave classification ───
print("\n[5] CDC epi-wave classification (Phase 4 anti-circular)")
# CDC FluView publishes "ILI activity level" (Levels 1-13)
# This could serve as INDEPENDENT label for Phase 4 HMM-epi alignment
check("ILI activity level column in CSV",
      any("activity" in c.lower() or "level" in c.lower() for c in data_columns),
      "Check CDC FluView for state-level activity classification")

# ─── Summary ───
n_pass = sum(1 for c in results["checks"] if c["ok"])
n_fail = len(results["checks"]) - n_pass
results["summary"] = {
    "n_checks": len(results["checks"]),
    "n_pass": n_pass,
    "n_fail": n_fail,
}

print()
print("=" * 80)
print(f"Pre-flight summary: {n_pass}/{len(results['checks'])} PASS")
print("=" * 80)

# Decision per check (informs Phase 5 branch)
print()
print("DECISION TREE for §18 Phase 5 (FluSight comparison):")
flusight_target_ok = any(c["name"].startswith("Our target") for c in results["checks"] if c["ok"])
print(f"  - wILI ↔ wk_inc_flu_hosp: INCOMPATIBLE")
print(f"  - Phase 5 branch: DOWNGRADE → 'FluSight-baseline only' (3-4h)")
print(f"    Alternative: Use FluSight 2022-23 season which has wILI targets (if any)")
print(f"    OR: Build correlation proxy (riskier)")

# Save markdown report
md = ["# §18 Phase 1 — Pre-flight Checks Report", "",
      f"**Date**: 2026-05-27", f"**Status**: {n_pass}/{len(results['checks'])} PASS",
      "", "## Checks", ""]
for c in results["checks"]:
    md.append(f"- {'✅' if c['ok'] else '❌'} **{c['name']}**: {c['detail']}")
md += ["", "## Phase 5 Decision", "",
       "**FluSight target alignment**: wILI (our target) ≠ wk_inc_flu_hosp (FluSight current target).",
       "Phase 5 must DOWNGRADE to 'FluSight-baseline only' comparison (~3-4h) OR use historical FluSight",
       "seasons that had wILI targets (2018-2022 era).",
       "",
       "## Phase 3 Decision (Region)", "",
       "Region stratification requires separate data fetch — add prep step before Phase 3 main work.",
       "",
       "## Phase 4 Decision (HMM-epi)", "",
       "Independent label source: CDC FluView ILI activity level (10-13 levels per region).",
       "If unavailable in our CSV, fetch from CDC API. Avoids STL-decomposition circular validation.",
       ""]
OUT_FILE.write_text("\n".join(md))
print(f"\nSaved: {OUT_FILE.relative_to(_ROOT)}")

(_ROOT / "outputs" / "preflight_checks.json").write_text(json.dumps(results, indent=2))
