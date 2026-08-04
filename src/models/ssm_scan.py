"""Pure-PyTorch selective scan for CG-Mamba.

PLAN v2.0.7 §3.2 — fallback option when mamba-ssm CUDA kernel is unavailable.
For weekly L ≤ 260 the O(L·D·N) sequential scan is fast enough on GPU (~ms).

Discretization (Mamba paper, eq. 4):
    A_bar_t = exp(dt_t · A)                       [ED, N]
    B_bar_t = dt_t · B_t                          [B, ED, N]
    h_t     = A_bar_t · h_{t-1} + B_bar_t · x_t   [B, ED, N]
    y_t     = (C_t · h_t).sum(-1) + D · x_t        [B, ED]

Performance NOTE (L1, M1.3 review):
    Sequential O(L) Python loop. For L≤260 (PLAN lookback grid max) this is
    sub-millisecond per layer on GPU. If M1.7 profiling shows scan to be the
    bottleneck, consider `torch.jit.script` on the inner loop or rewriting
    as a parallel scan (Blelloch). Premature now — defer to post-M1.7.

Inputs match mamba-ssm `selective_scan_fn` shape convention:
    x       [B, ED, L]   — post-conv1d activated input (or pre-conv1d if A-1)
    dt      [B, ED, L]   — already softplus-ed and (optionally) gated
    A       [ED, N]      — log-parameterized state matrix: A = -exp(A_log)
    B       [B, N, L]    — input matrix (time-varying, from x_proj)
    C       [B, N, L]    — output matrix (time-varying, from x_proj)
    D       [ED]         — skip connection (Mamba's z-modulated residual)
    z       [B, ED, L]   — gate from in_proj split (silu(z) * y)

Output:
    y       [B, ED, L]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def selective_scan_torch(
    x: torch.Tensor,        # [B, ED, L]
    dt: torch.Tensor,       # [B, ED, L]   (already softplus + γ applied)
    A: torch.Tensor,        # [ED, N]
    B: torch.Tensor,        # [B, N, L]
    C: torch.Tensor,        # [B, N, L]
    D: torch.Tensor,        # [ED]
    z: torch.Tensor | None = None,    # [B, ED, L]
) -> torch.Tensor:          # [B, ED, L]
    """Pure-PyTorch selective scan. See module docstring."""
    batch, ed, seqlen = x.shape
    n = A.shape[1]
    assert A.shape == (ed, n), f"A shape {tuple(A.shape)} != ({ed}, {n})"
    assert B.shape == (batch, n, seqlen)
    assert C.shape == (batch, n, seqlen)
    assert dt.shape == (batch, ed, seqlen)

    # Discretize once over all timesteps (vectorize over t)
    # dA_bar: [B, ED, N, L]  = exp(dt[:,:,None,:] * A[None,:,:,None])
    dt_exp = dt.unsqueeze(2)                                    # [B, ED, 1, L]
    A_exp = A.unsqueeze(0).unsqueeze(-1)                        # [1, ED, N, 1]
    dA_bar = torch.exp(dt_exp * A_exp)                          # [B, ED, N, L]

    # dB_bar · x : [B, ED, N, L]  =  dt[:,:,None,:] · B[:,None,:,:] · x[:,:,None,:]
    B_exp = B.unsqueeze(1)                                      # [B, 1, N, L]
    x_exp = x.unsqueeze(2)                                      # [B, ED, 1, L]
    dB_bar_x = dt_exp * B_exp * x_exp                           # [B, ED, N, L]

    # Sequential scan over L
    h = x.new_zeros(batch, ed, n)
    ys = []
    for t in range(seqlen):
        h = dA_bar[..., t] * h + dB_bar_x[..., t]               # [B, ED, N]
        # y_t = (C_t · h_t).sum(over N)  →  [B, ED]
        y_t = (C[:, :, t].unsqueeze(1) * h).sum(dim=-1)         # [B, ED]
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)                                 # [B, ED, L]
    # Skip connection
    y = y + D.view(1, ed, 1) * x                                # [B, ED, L]
    # Gate (silu(z) * y, Mamba's z-modulated output)
    if z is not None:
        y = y * F.silu(z)
    return y
