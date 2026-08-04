"""RMSNorm — Root Mean Square layer normalization (Zhang & Sennrich 2019).

PLAN v2.0.8b D.4.4 / D.4.6 / §3.0 budget: encoder uses RMSNorm (not LayerNorm).

vs LayerNorm:
  LayerNorm(D): 2D params (weight + bias)  →  64×2 = 128 per norm
  RMSNorm(D):   D  params (weight only)    →  64    = 64  per norm

Differences:
  - LayerNorm centers (subtracts mean) and scales by std + has bias.
  - RMSNorm only scales by RMS (no centering, no bias).
  - Faster, fewer params, no qualitative loss in transformer-style architectures
    (per Zhang & Sennrich 2019 and confirmed in CM-Mamba ablation).

PyTorch 2.x added `torch.nn.functional.rms_norm`; we implement a small wrapper
class for clarity + control over epsilon + state_dict naming.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS layer norm: y = x / RMS(x) * weight, where RMS(x) = sqrt(mean(x^2) + eps).

    Args:
        dim: feature dim (last axis).
        eps: numerical stability epsilon (default 1e-6 per Mamba / LLaMA convention).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        # Per-sample RMS over last axis (Mamba / LLaMA convention)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
