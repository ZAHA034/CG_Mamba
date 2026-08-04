"""M1.9 HPO chain wrapper — wait for Phase 1, then auto-launch Phase 2.

Polls runs/m1_9_hpo_phase1/hpo_summary.csv until 54 data rows are present,
then launches Phase 2 (Stage 3 ctx_ratio sweep over Phase 1 top-3 bases).

Usage:
    python3 scripts/m1_9_hpo_chain.py
    python3 scripts/m1_9_hpo_chain.py --top-n 3 --check-interval 60

The wrapper itself is cheap (sleeps + file stat). Phase 1 must be running
under a separate background task; the wrapper does NOT spawn Phase 1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_CG_MAMBA_ROOT = _THIS_FILE.parents[1]

PHASE1_CSV = _CG_MAMBA_ROOT / "runs" / "m1_9_hpo_phase1" / "hpo_summary.csv"
PHASE1_WINNER_JSON = _CG_MAMBA_ROOT / "runs" / "m1_9_hpo_phase1" / "hpo_winner.json"
PHASE2_SCRIPT = _CG_MAMBA_ROOT / "scripts" / "m1_9_hpo_phase2.py"
CHAIN_LOG = _CG_MAMBA_ROOT / "runs" / "m1_9_hpo_phase1_logs" / "chain.log"


def _count_data_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open() as f:
        return max(0, sum(1 for _ in f) - 1)   # minus header


def _is_phase1_finalized() -> bool:
    """Phase 1 considered done when winner.json is fresher than smoke-era timestamp
    AND has top5 with at least 3 entries (top-N=3 requirement)."""
    if not PHASE1_WINNER_JSON.exists():
        return False
    try:
        d = json.loads(PHASE1_WINNER_JSON.read_text())
    except Exception:
        return False
    top = d.get("top5") or d.get("top_5") or []
    return len(top) >= 3


def main() -> int:
    parser = argparse.ArgumentParser(description="HPO chain: wait Phase 1 → launch Phase 2")
    parser.add_argument("--target-runs", type=int, default=54,
                        help="Number of Phase 1 runs to wait for (default 54)")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Phase 2 top-N base cells (default 3)")
    parser.add_argument("--check-interval", type=int, default=60,
                        help="Polling interval in seconds (default 60)")
    parser.add_argument("--max-wait-hours", type=float, default=10.0,
                        help="Max wait time before bailing out (default 10h)")
    parser.add_argument("--phase2-batch-size", type=int, default=32)
    args = parser.parse_args()

    CHAIN_LOG.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    log_lines = [f"\n=== HPO chain wrapper start {start:%Y-%m-%d %H:%M:%S} ==="]
    log_lines.append(f"  target_runs={args.target_runs}, top_n={args.top_n}, "
                     f"check_interval={args.check_interval}s")
    print("\n".join(log_lines), flush=True)
    with CHAIN_LOG.open("a") as f:
        f.write("\n".join(log_lines) + "\n")

    # ── Polling loop ──
    max_wait_sec = args.max_wait_hours * 3600
    elapsed_max_reached = False
    while True:
        n_done = _count_data_rows(PHASE1_CSV)
        elapsed = (datetime.now() - start).total_seconds()
        if n_done >= args.target_runs and _is_phase1_finalized():
            msg = f"[{datetime.now():%H:%M:%S}] Phase 1 DONE: {n_done}/{args.target_runs} runs, winner.json finalized"
            print(msg, flush=True)
            with CHAIN_LOG.open("a") as f:
                f.write(msg + "\n")
            break
        if elapsed > max_wait_sec:
            msg = f"[{datetime.now():%H:%M:%S}] TIMEOUT after {args.max_wait_hours}h, Phase 1 incomplete ({n_done}/{args.target_runs})"
            print(msg, flush=True)
            with CHAIN_LOG.open("a") as f:
                f.write(msg + "\n")
            elapsed_max_reached = True
            break

        msg = (f"[{datetime.now():%H:%M:%S}] Phase 1 progress: {n_done}/{args.target_runs} runs  "
               f"(elapsed wrapper time: {elapsed/60:.1f}min). Sleeping {args.check_interval}s...")
        print(msg, flush=True)
        with CHAIN_LOG.open("a") as f:
            f.write(msg + "\n")
        time.sleep(args.check_interval)

    if elapsed_max_reached:
        return 1

    # ── Launch Phase 2 ──
    cmd = [sys.executable, "-u", str(PHASE2_SCRIPT),
           "--top-n", str(args.top_n),
           "--batch-size", str(args.phase2_batch_size),
           "--resume"]
    msg = f"\n=== Launching HPO Phase 2 at {datetime.now():%Y-%m-%d %H:%M:%S} ===\n  cmd: {' '.join(cmd)}"
    print(msg, flush=True)
    with CHAIN_LOG.open("a") as f:
        f.write(msg + "\n")

    rc = subprocess.run(cmd).returncode

    end_msg = f"\n=== HPO chain DONE at {datetime.now():%Y-%m-%d %H:%M:%S}, phase2 rc={rc} ==="
    print(end_msg, flush=True)
    with CHAIN_LOG.open("a") as f:
        f.write(end_msg + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
