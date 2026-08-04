#!/usr/bin/env python3
"""
Verify the rolling-origin table (label: tab:rolling) and the surrounding text
claims in Section IV against the canonical on-disk artifacts.

Manuscript values are PARSED from the LaTeX source, never transcribed by hand.
Disk values are recomputed from the raw per-(cutoff, region, seed) records using
the aggregation the table caption states: "10-region, 5-seed mean of the h=1-4
Cov95".

Canonical sources
-----------------
  runs/rolling_origin/cg_regional_results.csv        CG-Mamba, 7 x 10 x 5 rows
  runs/rolling_origin/baseline_regional_results.csv  5 DL baselines
  runs/rolling_origin/verdict.json                   pre-registered win counts

Any cell whose disk source cannot be located is reported NOT_EVALUABLE rather
than silently passed.

Usage:
    python scripts/verify_table_rolling.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "CGM_v2_paper" / "tex" / "results_main.tex"
RO = ROOT / "runs" / "rolling_origin"

TOL = 0.002          # 3-decimal rounding tolerance, as in the Table I check
NOMINAL = 0.95
HCOLS = [f"tS_cov95_h{h}" for h in (1, 2, 3, 4)]


# --------------------------------------------------------------------------
# 1. Parse the manuscript
# --------------------------------------------------------------------------
def parse_rolling_table(tex: str) -> tuple[list[dict], dict]:
    """Return (per-origin rows, footer dict) parsed from tab:rolling."""
    block = re.search(r"\\label\{tab:rolling\}(.*?)\\end\{tabular\}", tex, re.S)
    if not block:
        raise SystemExit("tab:rolling not found in the manuscript")
    body = block.group(1)

    rows = []
    for m in re.finditer(
        r"(\d{4})--(\d{2})\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*"
        r"(?:\\textbf\{)?(\d+)\}?\s*&\s*(?:\\textbf\{)?(\d+)\}?",
        body,
    ):
        rows.append({
            "cutoff": m.group(1),
            "cov95": float(m.group(3)),
            "dev": float(m.group(4)),
            "strict": int(m.group(5)),
            "clear": int(m.group(6)),
        })

    footer = {}
    fm = re.search(r"mean Cov95\s*&\s*([\d.]+)", body)
    if fm:
        footer["mean_cov95"] = float(fm.group(1))
    vm = re.search(r"strict\s*(\d+)/(\d+),\s*clear\s*(\d+)/(\d+)", body)
    if vm:
        footer["strict_n"], footer["n_origins"] = int(vm.group(1)), int(vm.group(2))
        footer["clear_n"] = int(vm.group(3))
    return rows, footer


def parse_text_claims(tex: str) -> dict:
    """Pull the two numeric claims made in the body text about this table."""
    out = {}
    m = re.search(r"average Cov95 stays within \$([\d.]+)\$ of nominal", tex)
    if m:
        out["cg_max_dev"] = float(m.group(1))
    m = re.search(r"every DL baseline is \$([\d.]+)\$--\$([\d.]+)\$ from nominal", tex)
    if m:
        out["baseline_dev_range"] = (float(m.group(1)), float(m.group(2)))
    return out


# --------------------------------------------------------------------------
# 2. Recompute from disk
# --------------------------------------------------------------------------
def _cell_mean(row: dict) -> float:
    """h=1-4 average Cov95 for one (cutoff, region, seed) record."""
    return sum(float(row[c]) for c in HCOLS) / len(HCOLS)


def disk_cg_by_cutoff() -> dict[str, float]:
    rows = list(csv.DictReader(open(RO / "cg_regional_results.csv")))
    acc: dict[str, list[float]] = {}
    for r in rows:
        acc.setdefault(r["cutoff"], []).append(_cell_mean(r))
    return {k: sum(v) / len(v) for k, v in acc.items()}, rows


def disk_baselines_by_cutoff() -> dict[tuple[str, str], float]:
    rows = list(csv.DictReader(open(RO / "baseline_regional_results.csv")))
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        acc.setdefault((r["baseline"], r["cutoff"]), []).append(_cell_mean(r))
    return {k: sum(v) / len(v) for k, v in acc.items()}, rows


# --------------------------------------------------------------------------
def main() -> int:
    for p in (TEX, RO / "cg_regional_results.csv",
              RO / "baseline_regional_results.csv", RO / "verdict.json"):
        if not p.exists():
            print(f"NOT_EVALUABLE: missing {p}")
            return 1

    tex = TEX.read_text()
    rows, footer = parse_rolling_table(tex)
    claims = parse_text_claims(tex)
    cg, cg_rows = disk_cg_by_cutoff()
    base, base_rows = disk_baselines_by_cutoff()
    verdict = json.loads((RO / "verdict.json").read_text())

    print("=" * 88)
    print("tab:rolling verification -- manuscript (parsed) vs canonical artifacts")
    print(f"CG rows: {len(cg_rows)}   baseline rows: {len(base_rows)}   "
          f"tolerance |Delta| < {TOL}")
    print("=" * 88)
    print(f"{'origin':<10}{'field':<10}{'paper':>10}{'disk':>12}{'|Delta|':>10}  verdict")
    print("-" * 88)

    npass = nfail = nna = 0

    def check(origin, field, paper, disk, tol=TOL):
        nonlocal npass, nfail, nna
        if disk is None:
            print(f"{origin:<10}{field:<10}{paper:>10}{'--':>12}{'--':>10}  NOT_EVALUABLE")
            nna += 1
            return
        d = abs(paper - disk)
        ok = d < tol
        print(f"{origin:<10}{field:<10}{paper:>10.3f}{disk:>12.5f}{d:>10.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
        npass += ok
        nfail += (not ok)

    for r in rows:
        c = r["cutoff"]
        dv = cg.get(c)
        check(c, "Cov95", r["cov95"], dv)
        check(c, "|dev|", r["dev"], None if dv is None else abs(dv - NOMINAL))
        # win counts come from the pre-registered verdict file
        pc = verdict.get("per_cutoff_cg_wins", {}).get(c, {})
        for key, name in (("strict", "best/10"), ("clear", "clear/10")):
            if key in pc:
                ok = r[key] == pc[key]
                print(f"{c:<10}{name:<10}{r[key]:>10}{pc[key]:>12}{'':>10}  "
                      f"{'PASS' if ok else 'FAIL'}")
                npass += ok
                nfail += (not ok)
            else:
                print(f"{c:<10}{name:<10}{r[key]:>10}{'--':>12}{'--':>10}  NOT_EVALUABLE")
                nna += 1

    print("-" * 88)
    if "mean_cov95" in footer:
        check("mean", "Cov95", footer["mean_cov95"],
              sum(cg.values()) / len(cg))
    if "strict_n" in footer:
        ok = footer["strict_n"] == verdict.get("n_strict_10of10")
        print(f"{'footer':<10}{'strict':<10}{footer['strict_n']:>10}"
              f"{verdict.get('n_strict_10of10'):>12}{'':>10}  {'PASS' if ok else 'FAIL'}")
        npass += ok; nfail += (not ok)
    if "clear_n" in footer:
        disk_clear = sum(1 for v in verdict["per_cutoff_cg_wins"].values()
                         if v["clear"] == 10)
        ok = footer["clear_n"] == disk_clear
        print(f"{'footer':<10}{'clear':<10}{footer['clear_n']:>10}{disk_clear:>12}"
              f"{'':>10}  {'PASS' if ok else 'FAIL'}")
        npass += ok; nfail += (not ok)

    # ---- body-text claims -------------------------------------------------
    print("\nBody-text claims:")
    if "cg_max_dev" in claims:
        worst = max(abs(v - NOMINAL) for v in cg.values())
        ok = worst <= claims["cg_max_dev"] + 1e-9
        print(f"  CG within {claims['cg_max_dev']} of nominal at every origin: "
              f"worst |dev| on disk = {worst:.5f} -> {'PASS' if ok else 'FAIL'}")
        npass += ok; nfail += (not ok)
    else:
        print("  CG max-deviation claim not found in text -> NOT_EVALUABLE"); nna += 1

    if "baseline_dev_range" in claims:
        lo_c, hi_c = claims["baseline_dev_range"]
        devs = {k: abs(v - NOMINAL) for k, v in base.items()}
        lo_d, hi_d = min(devs.values()), max(devs.values())
        ok = (round(lo_d, 2) >= round(lo_c, 2) - 0.005 and
              round(hi_d, 2) <= round(hi_c, 2) + 0.005)
        print(f"  every DL baseline {lo_c}-{hi_c} from nominal: "
              f"disk range = {lo_d:.4f}-{hi_d:.4f} -> {'PASS' if ok else 'FAIL'}")
        print(f"      min at {min(devs, key=devs.get)}, max at {max(devs, key=devs.get)}")
        npass += ok; nfail += (not ok)
    else:
        print("  baseline range claim not found in text -> NOT_EVALUABLE"); nna += 1

    print(f"\nSUMMARY: PASS={npass}  FAIL={nfail}  NOT_EVALUABLE={nna}")
    print("VERDICT:", "PASS" if nfail == 0 and nna == 0 else
                      ("FAIL" if nfail else "PARTIAL (gaps, no contradictions)"))
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
