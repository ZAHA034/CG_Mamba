#!/usr/bin/env python3
"""
Verify the component-ablation table (label: tab:ablation) and the body-text
bootstrap confidence intervals against the canonical on-disk artifacts.

Manuscript values are PARSED from the LaTeX source, never transcribed by hand.

Checks
------
  1. every cell of tab:ablation (mean metric and paired Delta vs Full)
  2. the bold convention the caption states -- "Bold Delta: 95% bootstrap CI
     excludes zero" -- against ci_excludes_zero on disk, in both directions
  3. the CI endpoints quoted in the body text

Canonical sources
-----------------
  runs/ablation_retrain/ablation_retrain_results.json   5-seed per-ablation means
  runs/ablation_retrain/bootstrap_ci.json               paired deltas, 10k bootstrap

Cells whose disk source cannot be located are reported NOT_EVALUABLE.

Usage:
    python scripts/verify_table_ablation.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "CGM_v2_paper" / "tex" / "results_main.tex"
AB = ROOT / "runs" / "ablation_retrain"

TOL = 0.002

# manuscript row label -> key in the artifacts
ROWS = {
    "EnvModule": "no_env",
    "Phase gate": "no_phase",
    "Rollout": "uniform_rollout",
}
METRICS = ["mae_avg", "wis_avg", "cov95_avg"]


def parse_ablation_table(tex: str):
    """Return (full_row, {ablation: {metric: (value, delta, is_bold)}})."""
    block = re.search(r"\\label\{tab:ablation\}(.*?)\\end\{tabular\*\}", tex, re.S)
    if not block:
        raise SystemExit("tab:ablation not found in the manuscript")
    body = block.group(1)

    full = None
    m = re.search(r"Full CG-Mamba \(ref\)((?:\s*&\s*\$[-\d.]+\$){3})", body)
    if m:
        full = [float(x) for x in re.findall(r"\$([-\d.]+)\$", m.group(1))]

    out = {}
    for label, key in ROWS.items():
        # capture the whole LaTeX row: from the label to the end-of-row marker.
        # NB: must not stop at the first backslash -- \mathbf{} appears mid-row.
        rm = re.search(rf"\$-\$ {re.escape(label)}[^\n]*?\\\\", body)
        if not rm:
            out[key] = None            # reported as NOT_EVALUABLE downstream
            continue
        cells = []
        for cell in rm.group(0).split("&")[1:]:
            vm = re.search(r"\$([\d.]+)\$", cell)
            dm = re.search(r"\(\s*\$?(\\mathbf\{)?\$?([+\-][\d.]+)", cell)
            if vm and dm:
                cells.append((float(vm.group(1)), float(dm.group(2)),
                              bool(dm.group(1))))
        out[key] = cells if len(cells) == len(METRICS) else None
    return full, out


def main() -> int:
    for p in (TEX, AB / "ablation_retrain_results.json", AB / "bootstrap_ci.json"):
        if not p.exists():
            print(f"NOT_EVALUABLE: missing {p}")
            return 1

    tex = TEX.read_text()
    full_paper, rows_paper = parse_ablation_table(tex)

    res = json.loads((AB / "ablation_retrain_results.json").read_text())
    agg = {a["ablation"]: a for a in res["aggregate"]}
    boot = json.loads((AB / "bootstrap_ci.json").read_text())["results"]

    npass = nfail = nna = 0

    def check(row, field, paper, disk, tol=TOL):
        nonlocal npass, nfail, nna
        if disk is None:
            print(f"{row:<20}{field:<12}{paper:>10}{'--':>12}{'--':>10}  NOT_EVALUABLE")
            nna += 1
            return
        d = abs(paper - disk)
        ok = d < tol
        print(f"{row:<20}{field:<12}{paper:>10.3f}{disk:>12.5f}{d:>10.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
        npass += ok; nfail += (not ok)

    print("=" * 90)
    print("tab:ablation verification -- manuscript (parsed) vs canonical artifacts")
    print(f"n_boot = {json.loads((AB / 'bootstrap_ci.json').read_text())['n_boot']}"
          f"   tolerance |Delta| < {TOL}")
    print("=" * 90)
    print(f"{'row':<20}{'field':<12}{'paper':>10}{'disk':>12}{'|Delta|':>10}  verdict")
    print("-" * 90)

    if full_paper:
        for metric, paper in zip(METRICS, full_paper):
            check("Full (ref)", metric, paper, agg["full"].get(metric + "_mean"))
    else:
        print("Full row not parsed -> NOT_EVALUABLE"); nna += 1

    bold_expect = {}
    for key in ROWS.values():
        cells = rows_paper.get(key)
        if not cells:
            print(f"{key:<20}{'ALL':<12}{'--':>10}{'--':>12}{'--':>10}  "
                  f"NOT_EVALUABLE (row not parsed from LaTeX)")
            nna += len(METRICS) * 2
            continue
        for metric, (val, delta, is_bold) in zip(METRICS, cells):
            check(key, metric, val, agg[key].get(metric + "_mean"))
            check(key, "d" + metric, delta, boot.get(key, {}).get(metric, {}).get("mean"))
            bold_expect[(key, metric)] = is_bold

    # ---- bold convention, both directions ---------------------------------
    print("-" * 90)
    print("Bold convention (caption: bold Delta <=> 95% bootstrap CI excludes zero):")
    for (key, metric), is_bold in bold_expect.items():
        disk_excl = boot.get(key, {}).get(metric, {}).get("ci_excludes_zero")
        if disk_excl is None:
            print(f"  {key:<18}{metric:<12} bold={is_bold}  disk=--  NOT_EVALUABLE"); nna += 1
            continue
        ok = (is_bold == disk_excl)
        print(f"  {key:<18}{metric:<12} bold={str(is_bold):<6} "
              f"ci_excludes_zero={str(disk_excl):<6} {'PASS' if ok else 'FAIL'}")
        npass += ok; nfail += (not ok)

    # ---- body-text CI endpoints -------------------------------------------
    print("-" * 90)
    print("Body-text CI endpoints:")
    text_cis = re.findall(
        r"\$\\Delta\$(MAE|WIS|Cov95) \$([+\-][\d.]+)\$ \[\$?([+\-][\d.]+)\$?, ?\$?([+\-][\d.]+)\$?\]",
        tex.replace("$-$", "-"))
    pairs = {"MAE": "mae_avg", "WIS": "wis_avg", "Cov95": "cov95_avg"}
    if not text_cis:
        print("  no CI triplets parsed from the text -> NOT_EVALUABLE"); nna += 1
    for name, mean_s, lo_s, hi_s in text_cis:
        metric = pairs[name]
        # the text quotes -Env for MAE/WIS and -Rollout for Cov95
        key = "uniform_rollout" if metric == "cov95_avg" else "no_env"
        d = boot[key][metric]
        for label, paper, disk in (("mean", float(mean_s), d["mean"]),
                                   ("ci_low", float(lo_s), d["ci_low"]),
                                   ("ci_high", float(hi_s), d["ci_high"])):
            check(f"{key}/{name}", label, paper, disk)

    print(f"\nSUMMARY: PASS={npass}  FAIL={nfail}  NOT_EVALUABLE={nna}")
    print("VERDICT:", "PASS" if nfail == 0 and nna == 0 else
                      ("FAIL" if nfail else "PARTIAL (gaps, no contradictions)"))
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
