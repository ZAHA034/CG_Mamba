#!/usr/bin/env python3
"""
P3 Table I verification: extract Table I (tab:national) values from
CGM_v2_paper/tex/results_main.tex and compare against on-disk artifacts.

Author-side check. The manuscript LaTeX source is not part of the public code
repository, so this script reports that and exits cleanly when it is absent.

Goal: close file-version-mismatch risk identified in P3 audit. Extraction was
HIGH-confidence on results_main.tex, but the paper may have been edited after
the Table I numbers were finalized; we re-derive every number directly from
disk and report PASS/FAIL row-by-row.

Tolerance: |Delta| < 0.002 (rounding tolerance at 3 decimals).
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX  = ROOT / "CGM_v2_paper/tex/results_main.tex"
RUNS = ROOT / "runs"

TOL = 0.002  # 3-decimal rounding tolerance

# ---------------------------------------------------------------------------
# 1. Parse Table I from results_main.tex
# ---------------------------------------------------------------------------
def parse_table_i(tex_path: Path):
    """Return ordered list of dicts with keys: model, params, mae, wis, cov95."""
    text = tex_path.read_text()
    # Locate the tab:national block
    m = re.search(r"\\label\{tab:national\}(.*?)\\end\{table\}", text, re.DOTALL)
    if not m:
        raise RuntimeError("tab:national not found")
    block = m.group(1)
    # Each data row pattern: "& <Model> & <Params> & <MAE> & <WIS> & <Cov95> \\"
    # Strip \textbf{} and \underline{} wrappers.
    def strip_fmt(s):
        s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\underline\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{[^}]*(\\shortstack(\[[^\]]*\])?\{[^}]*\})?[^}]*\}", "", s)
        return s.strip()

    rows = []
    # Split on rows ending with \\ then \hline boundaries
    for raw in block.split("\\\\"):
        if "&" not in raw:
            continue
        line = strip_fmt(raw)
        # remove any leftover multirow/shortstack tokens
        line = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", line)
        line = re.sub(r"\\shortstack\[[^\]]*\]\{[^}]*\}", "", line)
        # remove the leading "Group &" if present and any stray braces
        cells = [c.strip().strip("{}").strip() for c in line.split("&")]
        # Drop empty leading cells from multirow stripping
        # Expected cells = either 6 (Group,Model,P,MAE,WIS,Cov95) or 5 (Model,P,MAE,WIS,Cov95)
        # Filter known header / separator
        if not cells:
            continue
        # Discard obvious header
        joined = " ".join(cells).lower()
        if "model" in joined and "mae" in joined and "wis" in joined:
            continue
        # Heuristic pick last 5 cells = Model, Params, MAE, WIS, Cov95
        if len(cells) >= 5:
            model, params, mae, wis, cov = cells[-5], cells[-4], cells[-3], cells[-2], cells[-1]
        else:
            continue
        try:
            mae_v = float(mae)
            wis_v = float(wis)
            cov_v = float(cov)
        except ValueError:
            continue
        rows.append({
            "model":  model.strip(),
            "params": params.strip(),
            "mae":    mae_v,
            "wis":    wis_v,
            "cov95":  cov_v,
        })
    return rows

# ---------------------------------------------------------------------------
# 2. Read disk artifacts
# ---------------------------------------------------------------------------
def read_wis_test_strict(path: Path):
    """Return (wis_avg, cov95) from a wis_phase_b/<model>/wis_results.json file."""
    with open(path) as f:
        d = json.load(f)
    s = d["splits"]["test_strict"]
    return s["wis_avg"], s["coverage_95"]

def read_master_wis_table():
    """Return dict name -> (wis_avg, cov95) from master_wis_table.json."""
    with open(RUNS / "master_wis_table.json") as f:
        rows = json.load(f)
    return {r["name"]: (r["avg"], r["cov95"]) for r in rows}

def read_mae_m2_4_full_17seasons():
    """Return dict baseline -> mean(test_strict mae avg) over seeds at variant=17_seasons_full."""
    agg = defaultdict(list)
    path = RUNS / "m2_4_data_efficiency" / "m2_4_test_strict_all_baselines.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["variant"] == "17_seasons_full":
                v = sum(float(r[f"test_strict_mae_h{i}"]) for i in (1, 2, 3, 4)) / 4
                agg[r["baseline"]].append(v)
    return {k: (sum(v) / len(v), len(v)) for k, v in agg.items()}

def read_mae_m2_3_extra():
    """Return dict baseline -> mean(test_strict mae avg) from m2_3_extra_baselines_per_h_split.csv."""
    agg = defaultdict(list)
    path = RUNS / "m2_3_extra_baselines_per_h_split.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            v = sum(float(r[f"tS_mae_h{i}"]) for i in (1, 2, 3, 4)) / 4
            agg[r["baseline"]].append(v)
    return {k: (sum(v) / len(v), len(v)) for k, v in agg.items()}

# ---------------------------------------------------------------------------
# 3. Build disk-side row for each model
# ---------------------------------------------------------------------------
def build_disk_table():
    """Return dict model -> dict(mae, wis, cov95, sources). None where missing."""
    mae_m2_4   = read_mae_m2_4_full_17seasons()
    mae_m2_3e  = read_mae_m2_3_extra()
    master     = read_master_wis_table()

    # Source priority is per-model: re-evaluated "extra baselines" run
    # (m2_3_extra) supersedes the original m2_4 17_seasons_full run for the
    # five baselines that were re-evaluated under the corrected MAE protocol
    # (patchtst, epideep, itransformer, timesnet, nbeats, persistence). The
    # other five (sarima, dlinear, lstm, vanilla_mamba, cg_mamba) are sourced
    # from m2_4 17_seasons_full.
    M2_3_EXTRA_FIRST = {"patchtst", "epideep", "itransformer", "timesnet",
                        "nbeats", "persistence"}
    def pick_mae(*keys):
        for k in keys:
            if k in M2_3_EXTRA_FIRST:
                if k in mae_m2_3e:
                    return mae_m2_3e[k][0], f"m2_3_extra:{k}(n={mae_m2_3e[k][1]})"
                if k in mae_m2_4:
                    return mae_m2_4[k][0], f"m2_4_17seasons:{k}(n={mae_m2_4[k][1]})"
            else:
                if k in mae_m2_4:
                    return mae_m2_4[k][0], f"m2_4_17seasons:{k}(n={mae_m2_4[k][1]})"
                if k in mae_m2_3e:
                    return mae_m2_3e[k][0], f"m2_3_extra:{k}(n={mae_m2_3e[k][1]})"
        return None, None

    def pick_wis(*keys, prefer_wis_phase_b_dir=None):
        # First try master_wis_table.json
        for k in keys:
            if k in master:
                return master[k][0], master[k][1], f"master_wis_table:{k}"
        # fallback: wis_phase_b/<dir>/wis_results.json
        if prefer_wis_phase_b_dir:
            p = RUNS / "wis_phase_b" / prefer_wis_phase_b_dir / "wis_results.json"
            if p.exists():
                w, c = read_wis_test_strict(p)
                return w, c, f"wis_phase_b/{prefer_wis_phase_b_dir}/wis_results.json"
        return None, None, None

    out = {}

    # SARIMAX
    mae, mae_src = pick_mae("sarima")
    wis, cov, wis_src = pick_wis("sarima", prefer_wis_phase_b_dir="sarima")
    out["SARIMAX"] = dict(mae=mae, wis=wis, cov95=cov,
                          sources=dict(mae=mae_src, wis_cov=wis_src))

    # Persistence
    mae, mae_src = pick_mae("persistence")
    wis, cov, wis_src = pick_wis("persistence", prefer_wis_phase_b_dir="persistence")
    out["Persistence"] = dict(mae=mae, wis=wis, cov95=cov,
                              sources=dict(mae=mae_src, wis_cov=wis_src))

    # CG-Mamba (APMD)
    mae, mae_src = pick_mae("cg_mamba")
    wis, cov, wis_src = pick_wis("cg_mamba_method_F")
    out["CG-Mamba (APMD)"] = dict(mae=mae, wis=wis, cov95=cov,
                                   sources=dict(mae=mae_src, wis_cov=wis_src))

    # Vanilla Mamba (method-specific = MC@d0.1)
    mae, mae_src = pick_mae("vanilla_mamba")
    wis, cov, wis_src = pick_wis("vanilla_mamba_MC@d0.1", "vanilla_mamba")
    out["Vanilla Mamba"] = dict(mae=mae, wis=wis, cov95=cov,
                                 sources=dict(mae=mae_src, wis_cov=wis_src))

    # LSTM (method-specific = MC@d0.3)
    mae, mae_src = pick_mae("lstm")
    wis, cov, wis_src = pick_wis("lstm_MC@d0.3", "lstm")
    out["LSTM"] = dict(mae=mae, wis=wis, cov95=cov,
                       sources=dict(mae=mae_src, wis_cov=wis_src))

    # PatchTST
    mae, mae_src = pick_mae("patchtst")
    wis, cov, wis_src = pick_wis("patchtst", prefer_wis_phase_b_dir="patchtst")
    out["PatchTST"] = dict(mae=mae, wis=wis, cov95=cov,
                           sources=dict(mae=mae_src, wis_cov=wis_src))

    # EpiDeep
    mae, mae_src = pick_mae("epideep")
    wis, cov, wis_src = pick_wis("epideep", prefer_wis_phase_b_dir="epideep")
    out["EpiDeep"] = dict(mae=mae, wis=wis, cov95=cov,
                          sources=dict(mae=mae_src, wis_cov=wis_src))

    # DLinear
    mae, mae_src = pick_mae("dlinear")
    wis, cov, wis_src = pick_wis("dlinear", prefer_wis_phase_b_dir="dlinear")
    out["DLinear"] = dict(mae=mae, wis=wis, cov95=cov,
                          sources=dict(mae=mae_src, wis_cov=wis_src))

    # iTransformer
    mae, mae_src = pick_mae("itransformer")
    wis, cov, wis_src = pick_wis("itransformer", prefer_wis_phase_b_dir="itransformer")
    out["iTransformer"] = dict(mae=mae, wis=wis, cov95=cov,
                                sources=dict(mae=mae_src, wis_cov=wis_src))

    # N-BEATS
    mae, mae_src = pick_mae("nbeats")
    wis, cov, wis_src = pick_wis("nbeats", prefer_wis_phase_b_dir="nbeats")
    out["N-BEATS"] = dict(mae=mae, wis=wis, cov95=cov,
                          sources=dict(mae=mae_src, wis_cov=wis_src))

    # TimesNet
    mae, mae_src = pick_mae("timesnet")
    wis, cov, wis_src = pick_wis("timesnet", prefer_wis_phase_b_dir="timesnet")
    out["TimesNet"] = dict(mae=mae, wis=wis, cov95=cov,
                           sources=dict(mae=mae_src, wis_cov=wis_src))

    return out

# ---------------------------------------------------------------------------
# 4. Compare and print
# ---------------------------------------------------------------------------
def main():
    if not TEX.exists():
        print(f"Manuscript LaTeX source not found: {TEX}")
        print("This is an author-side check against the paper source, which is not")
        print("shipped in the public code repository. Nothing to verify; exiting.")
        return 0

    tex_rows = parse_table_i(TEX)
    disk     = build_disk_table()

    print("=" * 98)
    print("P3 Table I verification: LaTeX (tab:national) vs disk artifacts")
    print(f"Tolerance |Delta| < {TOL} (rounding at 3 decimals)")
    print("=" * 98)
    print(f"\n[1] Parsed {len(tex_rows)} rows from {TEX.relative_to(ROOT)}:")
    for r in tex_rows:
        print(f"    {r['model']:<22} params={r['params']:<8} "
              f"MAE={r['mae']:.3f}  WIS={r['wis']:.3f}  Cov95={r['cov95']:.3f}")

    print(f"\n[2] Per-row comparison vs disk:")
    hdr = f"{'Model':<22} {'Metric':<6} {'LaTeX':>8} {'Disk':>10} {'|Delta|':>9} {'Verdict':>8}  Source"
    print(hdr)
    print("-" * len(hdr))

    n_pass = 0
    n_fail = 0
    n_miss = 0
    discrepancies = []

    for r in tex_rows:
        name = r["model"]
        if name not in disk:
            print(f"{name:<22} {'ALL':<6} {'':>8} {'NO MATCH':>10} {'':>9} {'GAP':>8}  (no disk source mapped)")
            n_miss += 3
            discrepancies.append(f"{name}: no disk source mapped")
            continue
        d = disk[name]
        for metric in ("mae", "wis", "cov95"):
            tex_v = r[metric]
            disk_v = d[metric]
            src = d["sources"]["mae"] if metric == "mae" else d["sources"]["wis_cov"]
            if disk_v is None:
                print(f"{name:<22} {metric.upper():<6} {tex_v:>8.3f} {'MISSING':>10} {'':>9} {'GAP':>8}  -")
                n_miss += 1
                discrepancies.append(f"{name}/{metric}: disk artifact missing")
                continue
            delta = abs(tex_v - disk_v)
            ok = delta < TOL
            verdict = "PASS" if ok else "FAIL"
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                discrepancies.append(
                    f"{name}/{metric}: LaTeX={tex_v:.3f} disk={disk_v:.6f} "
                    f"|Delta|={delta:.4f} src={src}"
                )
            print(f"{name:<22} {metric.upper():<6} {tex_v:>8.3f} {disk_v:>10.4f} "
                  f"{delta:>9.4f} {verdict:>8}  {src}")

    print("-" * len(hdr))
    print(f"\n[3] SUMMARY:  PASS={n_pass}  FAIL={n_fail}  MISSING={n_miss}")
    if n_fail == 0 and n_miss == 0:
        print("    -> VERDICT: ALL PASS (LaTeX Table I matches disk artifacts).")
        rc = 0
    elif n_fail == 0:
        print("    -> VERDICT: PARTIAL (no contradictions, but some disk artifacts missing).")
        rc = 0
    else:
        print("    -> VERDICT: FAIL (LaTeX disagrees with disk on at least one cell).")
        rc = 1

    if discrepancies:
        print("\n[4] Discrepancies / gaps:")
        for d in discrepancies:
            print(f"    - {d}")

    print("\n[5] Source-mapping note (per P3 audit):")
    print("    - MAE values for DL baselines (cg_mamba, lstm, vanilla_mamba, dlinear,")
    print("      patchtst, epideep) come from m2_4 17_seasons_full and m2_3_extra")
    print("      (summary-only across 5 seeds, no per-prediction parquet).")
    print("    - SARIMAX MAE from m2_4 17_seasons_full (deterministic, single point).")
    print("    - WIS/Cov95 from master_wis_table.json (aggregated) and wis_phase_b/<m>/")
    print("      wis_results.json (per-baseline). CG-Mamba WIS/Cov95 is ONLY in")
    print("      master_wis_table.json; there is no runs/wis_phase_b/cg_mamba/ directory")
    print("      => calibrator cannot be re-fit from per-prediction artifacts (Track B blocker).")

    sys.exit(rc)

if __name__ == "__main__":
    main()
