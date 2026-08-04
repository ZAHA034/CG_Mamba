"""ROLLING-ORIGIN verdict — apply the PRE-REGISTERED table to CG + baseline results.

Reads (produced by the two drivers):
  runs/rolling_origin/cg_regional_results.csv        (cutoff, seed, region, tS_cov95_h*, tS_wis_h*)
  runs/rolling_origin/baseline_regional_results.csv  (cutoff, baseline, seed, region, tS_cov95_h*, tS_wis_h*)

PRE-REGISTERED endpoint (verbatim, locked in cutoffs_manifest.json BEFORE any result):
  Per cutoff: is CG-Mamba closest-to-nominal (min |Cov95-0.95|, h1-4 avg) in ALL 10 HHS
  regions vs the 5 DL baselines (lstm, vanilla_mamba, patchtst, dlinear_ensemble_gauss,
  epideep)? SARIMAX excluded (not native UQ).
  tie_rule: strict delta=-1 counts any margin; ALSO count regions with a CLEAR margin
  (CG |dev| >= 0.02 below the best baseline's |dev|).
  Verdict: >=6/7 cutoffs strict 10/10 = ROBUST; 4-5 = PARTIAL; <=3 = FAILED (retract claim).
  Result-blind: report ALL cutoffs; no post-hoc rule change; <=3/7 still goes in the paper.
  Secondary (recorded, no claim): WIS per cutoff.

This script CHANGES NOTHING about the rule — it only tallies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
ROLL = _ROOT / "runs" / "rolling_origin"
H = [1, 2, 3, 4]
DL5 = ["lstm", "vanilla_mamba", "patchtst", "dlinear_ensemble_gauss", "epideep"]
REGIONS = [f"hhs{i}" for i in range(1, 11)]
CLEAR = 0.02


def _cov_avg(df):
    df = df.copy()
    df["cov_avg"] = df[[f"tS_cov95_h{h}" for h in H]].mean(axis=1)
    df["wis_avg"] = df[[f"tS_wis_h{h}" for h in H]].mean(axis=1)
    return df


def main() -> int:
    cg = _cov_avg(pd.read_csv(ROLL / "cg_regional_results.csv"))
    bl = _cov_avg(pd.read_csv(ROLL / "baseline_regional_results.csv"))
    cutoffs = sorted(cg.cutoff.unique())

    # per (cutoff, region): CG cov_avg (mean over seeds) + best-baseline |dev|
    per_cutoff = {}
    strict_wins = {}
    for Y in cutoffs:
        cgY = cg[cg.cutoff == Y].groupby("region").cov_avg.mean()   # mean over seeds
        blY = bl[bl.cutoff == Y]
        # each baseline per-region cov_avg (mean over seeds / ensemble)
        base_reg = {b: blY[blY.baseline == b].groupby("region").cov_avg.mean() for b in DL5}
        rows, cg_win, clear_win = [], 0, 0
        for reg in REGIONS:
            cgd = abs(cgY.get(reg, np.nan) - 0.95)
            bdevs = {b: abs(base_reg[b].get(reg, np.nan) - 0.95) for b in DL5}
            best_b = min(bdevs, key=bdevs.get)
            best_bd = bdevs[best_b]
            win = cgd < best_bd
            clear = (best_bd - cgd) >= CLEAR
            cg_win += int(win); clear_win += int(win and clear)
            rows.append(dict(region=reg, cg_cov=float(cgY.get(reg, np.nan)), cg_dev=float(cgd),
                             best_baseline=best_b, best_base_dev=float(best_bd),
                             margin=float(best_bd - cgd), cg_wins=bool(win), clear=bool(clear)))
        per_cutoff[Y] = pd.DataFrame(rows)
        strict_wins[Y] = (cg_win, clear_win)

    # ---- report ----
    print("=" * 74)
    print("ROLLING-ORIGIN VERDICT (pre-registered; CG vs 5 DL baselines, native UQ)")
    print("=" * 74)
    print(f"\n{'cutoff':<9}{'CG wins /10':>13}{'clear(>=.02)':>14}{'CG avgCov':>11}{'worst DL avgCov':>17}")
    n_strict10 = 0
    for Y in cutoffs:
        cw, clw = strict_wins[Y]
        if cw == 10:
            n_strict10 += 1
        cgavg = cg[cg.cutoff == Y].cov_avg.mean()
        dl_avg = bl[bl.cutoff == Y].groupby("baseline").cov_avg.mean()
        worst = dl_avg.min()
        print(f"{str(Y)+'-'+str(Y+1)[2:]:<9}{cw:>10}/10{clw:>12}/10{cgavg:>11.3f}{worst:>17.3f}")

    verdict = ("ROBUST" if n_strict10 >= 6 else "PARTIAL" if n_strict10 >= 4 else "FAILED")
    print(f"\n  cutoffs with strict 10/10 = {n_strict10}/7  ->  VERDICT: {verdict}")
    claim = {"ROBUST": "may claim 'calibration dominance replicates across forecast origins'",
             "PARTIAL": "replicates in MOST origins; name the weaker cutoffs (conditional claim)",
             "FAILED": "rolling-origin WEAKENS the headline; RETRACT robustness claim; report as-is"}[verdict]
    print(f"  -> {claim}")

    # any cutoff where a baseline beats CG in some region -> disclose
    print("\n--- cutoffs where CG loses >=1 region (disclose which origin & region) ---")
    any_loss = False
    for Y in cutoffs:
        losses = per_cutoff[Y][~per_cutoff[Y].cg_wins]
        if len(losses):
            any_loss = True
            for _, r in losses.iterrows():
                print(f"  {Y}-{str(Y+1)[2:]} {r.region}: CG |dev|={r.cg_dev:.3f} vs {r.best_baseline} "
                      f"|dev|={r.best_base_dev:.3f} (CG worse by {-r.margin:.3f})")
    if not any_loss:
        print("  (none — CG closest-to-nominal in all 10 regions at every cutoff)")

    # secondary: WIS (recorded, no claim)
    print("\n--- SECONDARY (recorded, no claim): mean WIS per cutoff ---")
    print(f"{'cutoff':<9}{'CG':>9}" + "".join(f"{b[:8]:>10}" for b in DL5))
    for Y in cutoffs:
        cgw = cg[cg.cutoff == Y].wis_avg.mean()
        blw = bl[bl.cutoff == Y].groupby("baseline").wis_avg.mean()
        print(f"{str(Y)+'-'+str(Y+1)[2:]:<9}{cgw:>9.3f}" + "".join(f"{blw.get(b, np.nan):>10.3f}" for b in DL5))

    # save
    out = {"n_strict_10of10": n_strict10, "verdict": verdict, "claim": claim,
           "per_cutoff_cg_wins": {int(Y): {"strict": int(strict_wins[Y][0]),
                                           "clear": int(strict_wins[Y][1])} for Y in cutoffs},
           "prereg": "verdict table locked in cutoffs_manifest.json before results; unchanged here"}
    (ROLL / "verdict.json").write_text(json.dumps(out, indent=2))
    for Y in cutoffs:
        per_cutoff[Y].to_csv(ROLL / f"verdict_detail_cut{Y}.csv", index=False)
    print(f"\n  saved: runs/rolling_origin/verdict.json + per-cutoff detail CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
