"""Central configuration for CG-Mamba.

Per PLAN v2.0.8b §3.5 (누락 보완 #3): D/ED/N/K/r/V are single-source-of-truth
constants. No hardcoding in module-level code.

Default values match PLAN v2.0.8b:
  - §3.0 budget table (depth=3 weekly, ~117K trainable target)
  - §5.3 HP table (lookback=156, dropout=0.0, 3 LR fields)
  - §3.4 (HMM V=4 default + V=3 fallback, EB-2)
  - §5.1 (HMM 3 K × 3 seeds = 9 ckpts, EB-3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────── Module-level constants (not in CGMambaConfig dataclass) ───────────────
# These constants are not training HPs — they describe fixed data layout.
# Kept outside frozen dataclass so updates do not require ckpt re-instantiation
# and changes are visible to all loaders without config plumbing.

ILI_TARGET_IDX: int = 0
"""Position of ili_weighted_pct in main_input feature dimension (v2.1.7-B B-3).

All NN baselines (LSTM, Vanilla Mamba, PatchTST, iTransformer, TimesNet, DLinear,
EpiDeep, N-BEATS) and CG-Mamba assume target is feature 0 in V=6 layout:
    [ili_weighted_pct, log1p(total_ili_count), log1p(num_providers),
     log1p(num_patients), temperature_c, specific_humidity_g_per_kg]

Used by CGForecaster.forward (cg_forecaster.py:397) for last-value extraction
and by all baselines' TARGET_IDX constants. Changing this constant requires
data pipeline + all baseline model rewiring — handle as breaking change.
"""


@dataclass(frozen=True)
class CGMambaConfig:
    # ─────────────── Backbone (Mamba) ───────────────
    d_model: int = 64                  # D  — hidden dim
    d_state: int = 16                  # N  — SSM state dim
    d_conv:  int = 4                   # causal conv1d kernel
    expand:  int = 2                   # → d_inner = expand * d_model = 128 (ED)
    dt_rank: int = 16                  # R  — dt projection rank (v2.0.7 A-5 explicit override)
    n_layers: int = 3                  # depth=3 weekly default (v2.0.8 M1.2); ablation [2,3,4]

    # ─────────────── Gate (M1.3) ───────────────
    gate_rank: int = 8                 # r  — low-rank bottleneck (PLAN §3.2)
    gate_bias_init: float = 2.0        # v2.0.7 A-3: sigmoid(2.0) ≈ 0.88 near-identity init
    # use_gate selects backbone block class in M1.6 encoder integration:
    #   False → CGMambaBlock (M1.2 vanilla baseline + ablation `disable_gate`)
    #   True  → ContextGatedMambaBlock (M1.3, default for full CG-Mamba runs)
    # Not consumed in M1.2/M1.3 modules directly — read by M1.6 CGMambaEncoder.
    use_gate: bool = False

    # ─────────────── HMM (M1.4, v2.0.8b EB-2/EB-3) ───────────────
    n_states: int = 3                  # K  — HMM states (M1.4c v2.0.9: K=3 fixed)
    # hmm_input_dim removed (v2.1.7 H-3): legacy V=4 default conflicted with
    # active V_raw=3 / V_aug=6 path (M1.4c). All HMM code uses V_hmm_raw below.
    # 9 Stage-1 ckpts: 3 K × 3 seeds (PLAN v2.0.8b EB-3, expanded from K=4-only 3-seed)
    hmm_seeds: tuple[int, ...] = (42, 123, 456)

    # ─────────────── Phase Dynamics HMM (v2.0.9, M1.4c — L-3 / S-7 / T-2) ───────────────
    # PhaseModule(V_raw=cfg.V_hmm_raw, K=cfg.K_phase, d_embed=cfg.d_model) per S-7.
    # Augmented feature space V_aug = 2 · V_hmm_raw (Furui 1986 [x_t, Δx_t]).
    # M1.4b winner (V=3 × K=3 × reg_covar=5e-3 × n_init=5 → cross-seed κ_min=1.000).
    V_hmm_raw: int = 3                 # raw feature dim before augmentation (PLAN §3.4)
    K_phase: int = 3                   # K=3 fixed by BIC penalty analysis (Task #86)
    hmm_reg_covar: float = 5e-3        # M1.4b sweep winner (vs 1e-3/1e-2)
    hmm_n_init: int = 5                # multi-start count for Stage 1 EM
    # Stage 3 selective unfreeze flag (T-2). If True, the trainer is expected to
    # call `phase_module._unfreeze_for_stage3()` and rebuild the optimizer
    # (see PhaseModule docstring for the mandatory entry sequence).
    stage3_enabled: bool = False

    # ─────────────── M1.6 rollout (v2.0.9 PATCH 2 — L-2 forward-compat) ───────────────
    # PATCH 2 rollout — Δx 계산 + Taylor extrapolation용 trailing window. W ≥ 2 필수.
    # 현재 PhaseModule.rollout()은 마지막 2 timestep만 실제 사용 (x_t, Δx_t).
    # W=5는 forward-compat: 향후 v2.1.x Idea B (Phase Entropy → Δt 변조) 또는
    # higher-order Taylor 확장 (constant-acceleration: 3 timestep) 시 필요.
    rollout_window: int = 5

    # ─────────────── Env (M1.5) ───────────────
    env_input_dim: int = 2             # [specific_humidity_g_per_kg, temperature_c]
    env_hidden_dim: int = 32           # v2.0.7 A-2 autoencoder MLP hidden

    # ─────────────── Decoder (M1.7) ───────────────
    horizons: tuple[int, ...] = (1, 2, 3, 4)   # weeks ahead (CDC FluSight)
    # n_warm (prefix injection length in WEEKS, v2.0.8 weekly-native; was 14 days):
    #   prefix_proj: Linear(K_phase, d_model * n_warm) per PLAN D.4.4
    n_warm: int = 4

    # ─────────────── Time series ───────────────
    lookback: int = 156                # weeks (PLAN §5.3 default; HP grid [104, 156, 260])
    main_input_dim: int = 4            # backbone input: same 4 features as HMM V=4

    # ─────────────── Training ───────────────
    batch_size: int = 32
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    n_epochs: int = 50
    seed: int = 42
    dropout: float = 0.0               # v2.0.8b §5.3 default (baseline에 임의 dropout 도입 회피)

    # ─── Learning rates (v2.0.8 H-3, 3 fields for 4-param-group split) ───
    # M1.7 Stage 2 4-param-group split uses these distinct LRs:
    #   gate_proj   group: stage2_gate_lr      (Phase 1 start, cosine → 1e-6)
    #   decoder_gate group: stage2_backbone_lr  (Phase 1 start, cosine → 1e-6)
    #   context_embed (state_embeddings, near-freeze): 1e-6
    #   backbone     group: stage2_backbone_lr  (Phase 1 start, cosine → 1e-6)
    # Stage 1 (HMM ELBO + Env autoencoder) uses stage1_lr.
    stage1_lr: float = 1e-3            # HMM / Env autoencoder
    stage2_gate_lr: float = 1e-3       # gate_proj group (20× backbone)
    stage2_backbone_lr: float = 5e-5   # backbone + decoder_gate groups
                                       # (context_embed is hardcoded 1e-6 near-freeze,
                                       #  NOT driven by this field — see PLAN D.5.2 line 3021)

    # ─── Stage 2 training schedule (M1.7, v2.1.2 — PLAN D.5.2 active spec) ───
    # PLAN D.5.2 (line 2997-3036) requires distinct values from Stage 1 defaults.
    # We keep `n_epochs=50` and `weight_decay=1e-5` for Stage 1 / M1.2 backward compat,
    # and introduce Stage 2-specific fields here.
    stage2_n_epochs: int = 200         # PLAN D.5.2 line 3036: "Total 200 epoch"
    stage2_backbone_wd: float = 0.01   # PLAN D.5.2 line 3022: backbone group weight_decay=0.01
    stage2_gate_wd: float = 1e-3       # PLAN D.5.2: gate_proj group weight_decay (v2.1.7 H-2: surfaced from optimizer.py hardcode)
    stage2_patience: int = 30          # PLAN D.5.2 line 3036: early stopping patience=30

    # ─── Stage 3 (joint HMM fine-tune) LR fields (v2.1.7 C-1 + v2.1.7-A 4-group) ───
    # Previously these were hardcoded in m1_8_stage3_train._build_stage3_optimizer(),
    # and m1_9_hpo_phase2 / m2_1_final attempted to override via monkey-patch — but
    # the explicit kwargs at the call site bound ahead of the patched defaults,
    # silently discarding ctx_ratio sweep values (CRITICAL bug: HPO Phase 2 was
    # a degenerate sweep). Now first-class cfg fields with explicit signature.
    #
    # v2.1.7-A: 4-group split (hmm / state_embed / env / encoder_decoder). HPO Phase 2
    # now sweeps 3 independent LR ratios over Phase 1 top-3 base cells.
    stage3_hmm_lr: float = 1e-5         # _A / _means (HMM transition + emission means)
    stage3_state_embed_lr: float = 1e-5 # phase_module.state_embeddings (Stage 2 v2.1.3 default 일관)
    stage3_env_lr: float = 1e-4         # env_module.encoder.* (decoder is frozen from Stage 2)
    stage3_other_lr: float = 1e-4       # backbone / gate / outer decoder (everything else)
    # HPO Phase 2 sweeps each as ratio × stage3_other_lr base.

    # Backward-compat: M1.2 sanity script + M1.3 smoke use `lr` as single value.
    # Equivalent to `stage1_lr` when training Stage 1 / vanilla baselines.
    lr: float = 1e-3

    # ─────────────── Paths ───────────────
    data_csv: Path = field(default_factory=lambda:
        REPO_ROOT / "data" / "processed" / "ili_env_weekly_split.csv")
    norm_json: Path = field(default_factory=lambda:
        REPO_ROOT / "data" / "processed" / "normalization_params.json")
    boundaries_json: Path = field(default_factory=lambda:
        REPO_ROOT / "data" / "processed" / "split_boundaries.json")

    @property
    def d_inner(self) -> int:
        """ED = expand * D."""
        return self.expand * self.d_model

    def summary(self) -> str:
        return (
            f"CGMambaConfig(D={self.d_model}, ED={self.d_inner}, N={self.d_state}, "
            f"dt_rank={self.dt_rank}, K={self.n_states}, "
            f"V_hmm_raw={self.V_hmm_raw}, K_phase={self.K_phase}, "
            f"reg_covar={self.hmm_reg_covar:.0e}, n_init={self.hmm_n_init}, "
            f"rollout_W={self.rollout_window}, "
            f"stage3={'ON' if self.stage3_enabled else 'OFF'}, "
            f"r={self.gate_rank}, depth={self.n_layers}, lookback={self.lookback}, "
            f"horizons={self.horizons}, dropout={self.dropout}, "
            f"gate={'ON' if self.use_gate else 'OFF'})"
        )


if __name__ == "__main__":
    cfg = CGMambaConfig()
    print(cfg.summary())
    print(f"  data_csv:        {cfg.data_csv.relative_to(REPO_ROOT)}")
    print(f"  norm_json:       {cfg.norm_json.relative_to(REPO_ROOT)}")
    print(f"  boundaries_json: {cfg.boundaries_json.relative_to(REPO_ROOT)}")
    print(f"  hmm_seeds:       {cfg.hmm_seeds}")
    print(f"  LR (Stage 1 / Stage 2 gate / Stage 2 backbone): "
          f"{cfg.stage1_lr:g} / {cfg.stage2_gate_lr:g} / {cfg.stage2_backbone_lr:g}")
