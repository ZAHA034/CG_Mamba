"""EnvModule — env [B,L,2] → gate_env [B,L,D] autoencoder MLP (M1.5).

PLAN v2.0.9 §3.5 + v2.0.7 A-2 + v2.0.8b EB-4

Role: Encode z-scored env features (humidity, temperature) into a D-dim
gate_env embedding, paired with gate_phase (from PhaseModule, M1.4c v2.0.9)
via element-wise product to form `context_vec` for ContextGatedMambaBlock
(M1.3). M1.6 (CGForecaster) handles the L-1 alignment between PhaseModule's
length-(L-1) output and EnvModule's length-L output (`env_truncated =
env[:, 1:, :]` per PATCH 10 Dataflow).

Architecture:
    env [B, L, V_env=2]
        ↓
    EnvEncoder:   Linear(2, 32) → SiLU → Linear(32, D=64)
        ↓
    gate_env [B, L, D=64]
        │
        ├── (M1.6) context_vec = gate_phase ⊙ gate_env_truncated[:, 1:, :]
        │
        ↓
    EnvDecoder:   Linear(D=64, 32) → SiLU → Linear(32, 2)   ← Stage 1 aux only
        ↓
    env_recon [B, L, 2]  ← MSE reconstruction loss target

Init philosophy — NORMAL (NOT near-zero):
    v2.0.9 update: PhaseModule.state_embeddings (S-5 rename from state_embed)
    is initialized to exactly zero (R-4 zeros init, replacing the v2.0.8c
    std=0.02 randn). Consequently gate_phase = sigmoid(phase_post @ 0) =
    sigmoid(0) = **exactly 0.5** at init — a uniform multiplicative scalar
    rather than near-zero.

    Why EnvModule still needs normal (NOT near-zero) init:
        context_vec = gate_phase ⊙ gate_env (M1.6)
                    = 0.5 · gate_env                        (v2.0.9 at init)
        If gate_env were ALSO near-zero, context_vec ≈ 0.5·0 = 0 → no
        contextual signal flows into ContextGatedMambaBlock, and the
        gradient ∂L/∂state_embeddings (which depends on gate_env via the
        chain rule) collapses to zero magnitude. Normal init keeps
        gate_env ~ O(1) → context_vec ~ 0.5·O(1) = O(1) → healthy
        gradient signal to state_embeddings during Stage 2/3.

    Meanwhile, ContextGatedMambaBlock's gate_proj init (weight×0.01, bias=2.0)
    independently guarantees gate ≈ sigmoid(2.0) ≈ 0.88 (near-identity)
    REGARDLESS of context_vec magnitude. So EnvModule's normal init does NOT
    break the near-identity property — only the gradient health is improved.

Param budget (V=2, H=32, D=64):
    Encoder:  (2×32 + 32) + (32×64 + 64) = 96 + 2,112 = 2,208  ← main budget
    Decoder:  (64×32 + 32) + (32×2 + 2)  = 2,080 + 66 = 2,146  ← Stage 1 only
    Total:    4,354

    EnvDecoder는 Stage 1 reconstruction loss 전용. Stage 2/3에서는 encoder만
    사용 (gate_env 산출). decoder는 inference path에 미포함이므로 §3.0 main
    budget ~117K 에 미포함.

Data characteristics (M2.2 분석):
    Pearson r(humidity, temperature) = 0.9658  → near-collinear, eff. dim ≈ 1
    Both correlate with ILI (r ≈ -0.6).
    Hidden=32 is generous for 2D input — allows nonlinear envxILI mapping
    beyond linear correlation (PLAN §3.5 v2.0.8b EB-4 physical justification).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import CGMambaConfig


class EnvModule(nn.Module):
    """Autoencoder MLP mapping env features → gate_env embedding.

    Forward:
        env [B, L, V_env=2] → gate_env [B, L, D=64]

    Auxiliary (Stage 1 only):
        decode(gate_env) → env_recon [B, L, 2]   for MSE recon loss

    The encoder output `gate_env` is element-wise multiplied with `gate_phase`
    from PhaseModule (M1.4) to form `context_vec` for ContextGatedMambaBlock
    (M1.3). Decoder is auxiliary and excluded from inference path (Stage 2/3).

    Args:
        cfg: CGMambaConfig providing env_input_dim (V=2), env_hidden_dim (H=32),
             d_model (D=64).
    """

    def __init__(self, cfg: CGMambaConfig):
        super().__init__()

        # Fail-fast config validation (before allocating layers).
        self._validate_config(cfg)

        self.cfg = cfg

        V = cfg.env_input_dim    # 2
        H = cfg.env_hidden_dim   # 32
        D = cfg.d_model          # 64

        # ── Encoder: V → H → D (V_env → gate_env) ──
        self.encoder = nn.Sequential(
            nn.Linear(V, H),     # 2 → 32
            nn.SiLU(),
            nn.Linear(H, D),     # 32 → 64
        )

        # ── Decoder: D → H → V (gate_env → env_recon, Stage 1 aux) ──
        self.decoder = nn.Sequential(
            nn.Linear(D, H),     # 64 → 32
            nn.SiLU(),
            nn.Linear(H, V),     # 32 → 2
        )

        # NOTE: NO explicit init override. PyTorch's default Kaiming-uniform
        # (default for nn.Linear) gives O(1) output scale → healthy gradient
        # chain to PhaseModule.state_embeddings (v2.0.9 S-5; see module docstring).

        # Monitoring cache (W&B / paper figures). Same pattern as
        # ContextGatedMambaBlock._last_gate + PhaseModule._last_gamma:
        # train-only, detached, eval-mode = None.
        self._last_env: torch.Tensor | None = None

    @staticmethod
    def _validate_config(cfg: CGMambaConfig) -> None:
        """Sanity-check config values at construction time."""
        assert cfg.env_input_dim >= 1, (
            f"env_input_dim={cfg.env_input_dim} must be ≥ 1")
        assert cfg.env_hidden_dim >= cfg.env_input_dim, (
            f"env_hidden_dim={cfg.env_hidden_dim} < env_input_dim={cfg.env_input_dim} "
            f"— bottleneck narrower than input defeats autoencoder purpose")
        assert cfg.d_model >= cfg.env_hidden_dim, (
            f"d_model={cfg.d_model} < env_hidden_dim={cfg.env_hidden_dim} "
            f"— encoder hidden must not exceed output dim (inverted bottleneck)")

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        """Encode env features → gate_env embedding.

        Args:
            env: [B, L, V_env] z-scored environmental features
                 (V_env=2: [specific_humidity, temperature]).

        Returns:
            gate_env: [B, L, D] embedding for element-wise product with gate_phase.

        Raises:
            ValueError: if env has wrong shape (not 3D, or last dim ≠ V_env).
        """
        # Shape validation — fail with explicit message before cryptic torch error.
        if env.dim() != 3:
            raise ValueError(
                f"EnvModule expects 3D input [B, L, V_env], got {env.dim()}D "
                f"shape {tuple(env.shape)}")
        if env.shape[-1] != self.cfg.env_input_dim:
            raise ValueError(
                f"EnvModule input last dim {env.shape[-1]} != "
                f"env_input_dim={self.cfg.env_input_dim}")

        gate_env = self.encoder(env)   # [B, L, D]

        # Cache for monitoring (train-mode only, detached). Mirrors M1.3 / M1.4
        # _last_gate / _last_gamma patterns. Eval mode → None to avoid stale
        # state leakage during inference.
        self._last_env = gate_env.detach() if self.training else None

        return gate_env

    def decode(self, gate_env: torch.Tensor) -> torch.Tensor:
        """Decode gate_env → env_recon (auxiliary reconstruction).

        Used only in Stage 1 for autoencoder reconstruction loss. Stage 2/3
        inference does NOT call decode.

        Args:
            gate_env: [B, L, D] encoder output.

        Returns:
            env_recon: [B, L, V_env] reconstructed env features.

        Raises:
            ValueError: if gate_env has wrong shape.
        """
        if gate_env.dim() != 3:
            raise ValueError(
                f"decode expects 3D input [B, L, D], got {gate_env.dim()}D "
                f"shape {tuple(gate_env.shape)}")
        if gate_env.shape[-1] != self.cfg.d_model:
            raise ValueError(
                f"decode input last dim {gate_env.shape[-1]} != "
                f"d_model={self.cfg.d_model}")
        return self.decoder(gate_env)   # [B, L, V_env]

    def reconstruction_loss(
        self,
        env: torch.Tensor,
        gate_env: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE reconstruction loss (Stage 1 auxiliary objective).

        Convenience method: if gate_env is None, runs forward() internally.
        This avoids double-forward when caller needs both gate_env (for
        context_vec) and recon_loss (for Stage 1 backward).

        Args:
            env:      [B, L, V_env] original z-scored env features.
            gate_env: [B, L, D] pre-computed encoder output (optional —
                      if None, calls self.forward(env)).

        Returns:
            Scalar MSE loss tensor.
        """
        if gate_env is None:
            gate_env = self.forward(env)
        env_recon = self.decode(gate_env)
        return F.mse_loss(env_recon, env)

    # ── Parameter group helpers (M1.7 optimizer construction 용) ──

    def encoder_parameters(self):
        """Iterator for encoder params.

        Use in M1.7 Stage 2/3 optimizer group construction — encoder만 학습
        대상이고 decoder는 Stage 1에서만 사용된다 (freeze_decoder_for_stage2
        호출 전제). 외부에서 `env_module.encoder.parameters()` 직접 접근 대신
        본 method를 사용하여 encapsulation 유지.
        """
        return self.encoder.parameters()

    def decoder_parameters(self):
        """Iterator for decoder params (Stage 1 reconstruction loss 학습 전용).

        Stage 2/3 진입 시 freeze_decoder_for_stage2()로 frozen 처리되므로,
        이 method가 반환하는 params도 requires_grad=False 상태가 된다.
        """
        return self.decoder.parameters()

    def encoder_param_count(self) -> int:
        """Encoder-only param count (entry into main CG-Mamba budget)."""
        return sum(p.numel() for p in self.encoder.parameters())

    def decoder_param_count(self) -> int:
        """Decoder-only param count (Stage 1 aux, NOT in main budget)."""
        return sum(p.numel() for p in self.decoder.parameters())

    # ── Stage 2/3 transition: decoder freeze (PLAN §5.1 spec) ──

    def freeze_decoder_for_stage2(self) -> int:
        """Freeze decoder params for Stage 2/3 (PLAN §3.5 + §5.1 spec).

        EnvModule의 decoder는 Stage 1 reconstruction loss 전용. Stage 2부터는
        encoder의 gate_env만 사용되고 decoder는 inference path에서 호출되지
        않는다. optimizer에 잘못 포함될 위험 + budget overrun 방지를 위해
        명시적 freeze.

        Pattern mirrors `hmm_stage1.freeze_hmm_for_stage2()` (M1.4):
            - already-frozen params는 카운트에서 제외 ("이 호출이 새로
              freeze한 수"를 반환)
            - post-condition assert: encoder는 여전히 trainable 보장

        Returns:
            Number of scalar params frozen by this call (sum of `.numel()`
            for decoder params with requires_grad=True before the call;
            2,146 on first invocation, 0 on idempotent re-call). Returns the
            numel sum, NOT the parameter-tensor count (Linear weight + bias
            are two tensors but report their full scalar size).
        """
        n_frozen = 0
        for p in self.decoder.parameters():
            if p.requires_grad:
                p.requires_grad = False
                n_frozen += p.numel()                # C-1 fix: numel sum

        # Post-condition 1: ALL decoder params now frozen (numel sum = 0)
        decoder_trainable = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )
        assert decoder_trainable == 0, (
            f"freeze_decoder_for_stage2 failed: {decoder_trainable} decoder "
            f"scalar params still trainable. Stage 2 invariant violated (PLAN §3.5)."
        )

        # Post-condition 2: Encoder MUST remain trainable (Stage 2 학습 대상)
        encoder_trainable = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        assert encoder_trainable > 0, (
            f"freeze_decoder_for_stage2 invariant violated: encoder has 0 "
            f"trainable scalar params. EnvModule encoder must remain trainable "
            f"in Stage 2 (gate_env produces context_vec via gradient signal)."
        )

        return n_frozen

    def extra_repr(self) -> str:
        return (
            f"V_env={self.cfg.env_input_dim}, H={self.cfg.env_hidden_dim}, "
            f"D={self.cfg.d_model}"
        )
