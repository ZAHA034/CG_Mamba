"""M1.3 bias_init sensitivity smoke test.

PLAN v2.0.8a §3.8 + §9.1 W2 — bias_init ∈ {0.0, 1.0, 2.0, 3.0} 5-epoch synthetic
training stability check. **Not** a full ILI training run — only block-level
forward+backward+optimizer.step() loop with random tensors.

Stability criteria (revised — synthetic random target has no learnable signal):
  1. No NaN/Inf in loss across 5 epochs.
  2. No loss spike: max(losses) - min(losses) < 0.5.
  3. Gate diversification: gate.std at ep5 > gate.std at ep1 (context starts
     to differentiate per-position gates as W2 drifts from ~0).
  4. Default 2.0 must satisfy all three; if any fails, retreat to 3.0.

Note: A "loss decrease" criterion would be meaningless here because the target
is i.i.d. random — no signal to learn. Real learning curves require ILI data
(M1.7 full integration).

Usage:
    python -m scripts.m1_3_bias_init_smoke
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from src.models.context_gate import ContextGatedMambaBlock
from src.utils.config import CGMambaConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def smoke_test(bias_init: float, epochs: int = 5, B: int = 4, L: int = 104,
               seed: int = 42) -> dict:
    """Run 5-epoch synthetic forward+backward+step loop. Return diagnostic dict."""
    torch.manual_seed(seed)
    cfg = CGMambaConfig()
    block = ContextGatedMambaBlock(cfg, gate_bias_init=bias_init)
    opt = torch.optim.AdamW(block.parameters(), lr=1e-3)

    losses: list[float] = []
    grad_norms: list[float] = []
    gate_means: list[float] = []
    gate_stds: list[float] = []
    stable = True

    # Snapshot initial gate
    with torch.no_grad():
        ctx_init = torch.randn(B, L, cfg.d_model)
        g_init = torch.sigmoid(block.gate_proj(ctx_init))

    for ep in range(epochs):
        # Synthetic batch (different per epoch but seeded for reproducibility)
        torch.manual_seed(seed + ep)
        x = torch.randn(B, L, cfg.d_model)
        ctx = torch.randn(B, L, cfg.d_model)
        target = torch.randn(B, L, cfg.d_model)

        y = block(x, ctx)
        loss = (y - target).pow(2).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            stable = False
            losses.append(float("nan"))
            break

        losses.append(loss.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0).item()
        grad_norms.append(gn)
        opt.step()

        # Track gate distribution post-step
        with torch.no_grad():
            g = block._last_gate
            gate_means.append(g.mean().item())
            gate_stds.append(g.std().item())

    if len(losses) > 1:
        monotone_decrease = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    else:
        monotone_decrease = False

    return {
        "bias_init": bias_init,
        "sigmoid_bias": torch.sigmoid(torch.tensor(bias_init)).item(),
        "init_gate_mean": float(g_init.mean()),
        "init_gate_std": float(g_init.std()),
        "stable": stable,
        "losses": losses,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_delta": (losses[-1] - losses[0]) if losses else None,
        "monotone_decrease": monotone_decrease,
        "grad_norms": grad_norms,
        "gate_means": gate_means,
        "gate_stds": gate_stds,
    }


def main() -> int:
    bias_grid = [0.0, 1.0, 2.0, 3.0]
    results: dict = {}

    print("=" * 78)
    print("M1.3 bias_init sensitivity smoke test (PLAN §3.8 A-3)")
    print("=" * 78)
    print(f"{'bias_init':>10} {'σ(bias)':>9} {'init_gate':>10} {'L_first':>10} "
          f"{'L_last':>10} {'ΔL':>10} {'monot↓':>7} {'stable':>7}")
    print("-" * 78)

    for b in bias_grid:
        r = smoke_test(b)
        results[str(b)] = r
        print(f"{b:>10.1f} {r['sigmoid_bias']:>9.4f} {r['init_gate_mean']:>10.4f} "
              f"{r['loss_first']:>10.4f} {r['loss_last']:>10.4f} "
              f"{r['loss_delta']:>+10.4f} {str(r['monotone_decrease']):>7} "
              f"{str(r['stable']):>7}")

    print()
    print("Per bias_init: gate_means trajectory (epochs 1..5)")
    for b in bias_grid:
        r = results[str(b)]
        means = " → ".join(f"{m:.4f}" for m in r["gate_means"])
        stds = " → ".join(f"{s:.4f}" for s in r["gate_stds"])
        print(f"  bias={b}:  mean: {means}")
        print(f"             std:  {stds}")

    # Save JSON
    out_dir = REPO_ROOT / "runs" / "m1_3_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_init_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path.relative_to(REPO_ROOT)}")

    # PLAN §9.1 W2 exit criteria (revised): "5-epoch no NaN/Inf, no loss spike".
    # Note: synthetic random target → loss ~ const ≈ Var(x)+Var(target). Loss
    # decrease isn't a meaningful criterion here (no signal to learn). The right
    # check is *no instability* — block forward+backward+step run cleanly.
    SPIKE_THRESHOLD = 0.5    # |Δloss| > 0.5 across 5 epochs flags instability
    all_stable = all(r["stable"] for r in results.values())
    spike_check = {b: (max(r["losses"]) - min(r["losses"]) < SPIKE_THRESHOLD
                       if r["stable"] else False)
                   for b, r in results.items()}
    all_no_spike = all(spike_check.values())
    # Gate diversification: std should grow over epochs (context has some effect)
    gate_diversifying = {b: (r["gate_stds"][-1] > r["gate_stds"][0]
                             if r["stable"] else False)
                         for b, r in results.items()}
    all_diversifying = all(gate_diversifying.values())

    print()
    print("=" * 78)
    print("Decision (PLAN §9.1 W2 exit criteria, revised):")
    print(f"  All 4 bias_init stable (no NaN/Inf):  {all_stable}")
    print(f"  All 4 no loss spike (|Δ| < 0.5):       {all_no_spike}")
    print(f"  All 4 gate std diversifying (std↑):    {all_diversifying}")
    for b in bias_grid:
        r = results[str(b)]
        std_ratio = r["gate_stds"][-1] / r["gate_stds"][0] if r["gate_stds"][0] > 0 else 0
        print(f"    bias={b}: spike Δ={max(r['losses'])-min(r['losses']):.4f}, "
              f"std×{std_ratio:.2f} (ep1→ep5)")

    ok = all_stable and all_no_spike and all_diversifying
    if ok:
        print(f"  ✅ Default 2.0 CONFIRMED (full integration M1.7로 진행)")
    else:
        print(f"  ❌ Instability detected — investigate")
    print("=" * 78)
    print("Note: synthetic random target → loss decrease not meaningful here.")
    print("      Meaningful learning curves require real ILI data (M1.7).")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
