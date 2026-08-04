"""CG-Mamba single block — vanilla Mamba forward with optional pre-conv1d gate hook.

Wiring (PLAN v2.0.7 A-1):
    x [B,L,D]
      ─in_proj─→ xz [B,L,2*ED]
                  ├─split─→ x_inner [B,L,ED]
                  └────────→ z       [B,L,ED]
              ┌─→ (M1.3 hook) x_inner = gate ⊙ x_inner   ← PRE-conv1d, raw
              ↓
            conv1d (causal, k=4) → activated x_inner [B,L,ED]
              ↓
            x_proj → split (dt_raw, B, C) of dim (dt_rank, N, N)
              ↓
            dt_proj(dt_raw) → dt [B,L,ED]
              ↓
            dt = softplus(dt + dt_proj.bias)
              ↓
            selective_scan_torch(x_inner, dt, A, B, C, D, z) → y [B,L,ED]
              ↓
            out_proj → y [B,L,D]

For M1.2 (use_gate=False), the gate hook is skipped → wiring is bit-identical to
vanilla Mamba-1 slow path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.ssm_scan import selective_scan_torch
from src.utils.config import CGMambaConfig


class CGMambaBlock(nn.Module):
    """One Mamba-1 layer with an optional pre-conv1d unified gate hook.

    Constructor mirrors mamba-ssm Mamba module signature (a subset).
    Parameter init follows Mamba paper §A.2.
    """

    def __init__(self, cfg: CGMambaConfig):
        super().__init__()
        self.cfg = cfg
        D, ED, N = cfg.d_model, cfg.d_inner, cfg.d_state
        R = cfg.dt_rank
        K = cfg.d_conv

        # in_proj: D → 2·ED  (x and z)
        self.in_proj = nn.Linear(D, 2 * ED, bias=False)

        # Causal conv1d, depthwise, kernel K=4
        self.conv1d = nn.Conv1d(
            in_channels=ED, out_channels=ED, kernel_size=K, padding=K - 1,
            groups=ED, bias=True,
        )

        # x_proj: ED → (R + 2N)  — produces dt_raw, B_t, C_t
        self.x_proj = nn.Linear(ED, R + 2 * N, bias=False)

        # dt_proj: R → ED  (low-rank dt parameterization)
        self.dt_proj = nn.Linear(R, ED, bias=True)
        # Mamba paper dt init: random log-uniform init for dt_proj.bias such that
        # softplus(bias) ∈ [dt_min, dt_max]. Here we use dt_min=1e-3, dt_max=1e-1.
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(ED) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
            ).clamp(min=1e-4)
            # inverse softplus
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj.bias._no_reinit = True

        # A_log: parameterize A = -exp(A_log) so A is always negative
        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(ED, 1)   # [ED, N]
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(ED))

        # out_proj: ED → D
        self.out_proj = nn.Linear(ED, D, bias=False)

    def forward(
        self,
        x: torch.Tensor,                       # [B, L, D]
        gate: torch.Tensor | None = None,      # [B, L, ED]  (M1.3 hook)
    ) -> torch.Tensor:                         # [B, L, D]
        B_size, L, D = x.shape
        ED = self.cfg.d_inner
        N = self.cfg.d_state
        R = self.cfg.dt_rank

        # 1. in_proj + split
        xz = self.in_proj(x)                             # [B, L, 2*ED]
        x_inner, z = xz.chunk(2, dim=-1)                 # each [B, L, ED]

        # 2. Pre-conv1d gate hook (M1.3 / v2.0.7 A-1). M1.2: gate is None → skip.
        if gate is not None:
            assert gate.shape == (B_size, L, ED), (
                f"gate {tuple(gate.shape)} != (B={B_size}, L={L}, ED={ED})")
            x_inner = x_inner * gate

        # 3. Causal conv1d (channel-last → channel-first → channel-last)
        x_inner = rearrange(x_inner, "b l d -> b d l")
        x_inner = self.conv1d(x_inner)[..., :L]          # crop padding tail → [B, ED, L]
        x_inner = F.silu(x_inner)

        # 4. x_proj on flattened batch×L
        x_dbl = self.x_proj(rearrange(x_inner, "b d l -> (b l) d"))   # [B*L, R+2N]
        dt_raw, B_t, C_t = torch.split(x_dbl, [R, N, N], dim=-1)

        # 5. dt projection + softplus + bias
        dt = self.dt_proj.weight @ dt_raw.t()                          # [ED, B*L]
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        dt = dt + self.dt_proj.bias.float().unsqueeze(0).unsqueeze(-1)
        dt = F.softplus(dt)                                            # [B, ED, L]

        # 6. Reshape B, C
        B_t = rearrange(B_t, "(b l) n -> b n l", l=L).contiguous()     # [B, N, L]
        C_t = rearrange(C_t, "(b l) n -> b n l", l=L).contiguous()     # [B, N, L]

        # 7. Selective scan
        A = -torch.exp(self.A_log.float())                             # [ED, N]
        y = selective_scan_torch(
            x=x_inner, dt=dt, A=A, B=B_t, C=C_t, D=self.D,
            z=rearrange(z, "b l d -> b d l"),
        )                                                              # [B, ED, L]

        # 8. out_proj
        y = rearrange(y, "b d l -> b l d")
        y = self.out_proj(y)                                           # [B, L, D]
        return y
