"""src/models/heteroscedastic_head.py — E3 learned calibration head (γ.4 재해석)
================================================================================
Replaces eq.13 post-hoc s_h grid-search with *learned end-to-end* calibration:

    σ²_cal(t, h) = exp(a_h) · σ²_within(t, h) + exp(b_h) · σ²_between(t, h)

where a_h, b_h are learned per-horizon scalars (initialized at 0 → identity →
raw HMM 분산). Optimized by WIS-based loss on calibration set.

LOCKED γ.4 재해석 (E1 결과 후 변경):
  - Original threshold: "transfer Cov95 ≥ 0.92 보존 → learned head 채택, < 0.92 → s_h 유지"
  - New (E1 raw=over-cover 0.98 발견 후): "raw 분산은 *over-conservative* → head 의
    역할은 '0.92 보존' 이 아니라 *over (0.98) → nominal (0.95) 로 당기되 transfer
    안 깨기'"
  - 즉 head 가 *under-cover* 가 아니라 *over-coverage 완화* 역할

paper §V-A reposition (γ.7 disclosure 5):
  - raw 분산 = intrinsic over-conservative (post-hoc tuning 인공물 아님)
  - E3 head = "learned end-to-end calibration" — reviewer #2 의 "eq.13 s_h =
    conformal 과 다를 바 없다" 비판을 *"post-hoc 을 learned 로 교체"* 로 정면 응답

용도:
  - α.4: held-out + test_strict 각 raw + calibrated 둘 다 보고
  - E1 winner (n4_d128) 위 적용
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as sp_norm

FLUSIGHT_23 = np.array([
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 0.975, 0.99
])


# ============================================================================
class HeteroHead(nn.Module):
    """Per-horizon learned scale on APMD variance decomposition.

    σ²_cal(t, h) = α_h · σ²_within(t, h) + β_h · σ²_between(t, h)
    where α_h = exp(log_alpha[h]),  β_h = exp(log_beta[h])
    init: log_alpha = log_beta = 0 → α = β = 1 → σ²_cal = σ²_total = raw (identity)

    Trainable: 2H scalars (H=4 → 8 params). Very small, generalize-friendly.
    """

    def __init__(self, n_horizons: int = 4):
        super().__init__()
        self.H = n_horizons
        self.log_alpha = nn.Parameter(torch.zeros(n_horizons))
        self.log_beta = nn.Parameter(torch.zeros(n_horizons))

    def forward(self, s2_within: torch.Tensor, s2_between: torch.Tensor) -> torch.Tensor:
        """s2_within, s2_between: [N, H]. Returns σ²_cal [N, H]."""
        alpha = torch.exp(self.log_alpha)[None, :]
        beta = torch.exp(self.log_beta)[None, :]
        return alpha * s2_within + beta * s2_between

    def get_params(self) -> dict:
        return dict(
            alpha=torch.exp(self.log_alpha).detach().cpu().numpy().tolist(),
            beta=torch.exp(self.log_beta).detach().cpu().numpy().tolist(),
        )


# ============================================================================
# Loss + quantile construction
# ============================================================================
def gaussian_quantiles(mu: torch.Tensor, sigma2: torch.Tensor,
                        taus: torch.Tensor) -> torch.Tensor:
    """μ + Φ⁻¹(τ) · √σ². mu/sigma2: [N, H], taus: [Q]. Returns [N, H, Q]."""
    sigma = torch.sqrt(torch.clamp(sigma2, min=1e-12))
    return mu[..., None] + sigma[..., None] * taus[None, None, :]


def wis_loss_23(mu: torch.Tensor, sigma2: torch.Tensor, y: torch.Tensor,
                 flusight_z: torch.Tensor, flusight_tau: torch.Tensor) -> torch.Tensor:
    """23-quantile WIS, mean. mu/sigma2/y: [N, H]. Returns scalar."""
    sigma = torch.sqrt(torch.clamp(sigma2, min=1e-12))
    Q = mu[..., None] + sigma[..., None] * flusight_z[None, None, :]      # [N, H, Q]
    y_b = y[..., None]                                                      # [N, H, 1]
    pinball = torch.where(
        y_b >= Q,
        flusight_tau[None, None, :] * (y_b - Q),
        (1 - flusight_tau[None, None, :]) * (Q - y_b),
    )
    return 2.0 * pinball.mean()


# ============================================================================
def fit_hetero_head(mu_cal: np.ndarray, s2_within_cal: np.ndarray,
                     s2_between_cal: np.ndarray, y_cal: np.ndarray,
                     n_horizons: int = 4, epochs: int = 500,
                     lr: float = 1e-2, verbose: bool = False) -> HeteroHead:
    """Fit head on calibration set by WIS loss. Numpy inputs [N, H].

    LOCKED:
      - init: α = β = 1 (identity, raw HMM)
      - optimizer: Adam lr=0.01, epochs=500 (very small head, converges fast)
      - loss: 23-quantile WIS (FluSight protocol)
      - calibration set: γ.4 의 design-val 또는 held-out subset (caller 결정)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu = torch.tensor(mu_cal, dtype=torch.float32, device=device)
    sw = torch.tensor(s2_within_cal, dtype=torch.float32, device=device)
    sb = torch.tensor(s2_between_cal, dtype=torch.float32, device=device)
    y = torch.tensor(y_cal, dtype=torch.float32, device=device)
    z = torch.tensor(sp_norm.ppf(FLUSIGHT_23), dtype=torch.float32, device=device)
    tau = torch.tensor(FLUSIGHT_23, dtype=torch.float32, device=device)

    head = HeteroHead(n_horizons=n_horizons).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    init_loss = None
    for ep in range(epochs):
        optimizer.zero_grad()
        s2_cal = head(sw, sb)
        loss = wis_loss_23(mu, s2_cal, y, z, tau)
        if ep == 0:
            init_loss = loss.item()
        loss.backward()
        optimizer.step()
        if verbose and (ep + 1) % 100 == 0:
            params = head.get_params()
            print(f"  ep {ep+1}/{epochs}  WIS={loss.item():.4f}  "
                  f"α={[f'{a:.3f}' for a in params['alpha']]}  "
                  f"β={[f'{b:.3f}' for b in params['beta']]}")
    if verbose:
        final_loss = loss.item()
        print(f"  fit done: init WIS={init_loss:.4f} → final WIS={final_loss:.4f}")
    return head


def apply_hetero_head(head: HeteroHead, s2_within: np.ndarray,
                       s2_between: np.ndarray) -> np.ndarray:
    """Apply fit head to new data (held-out / test_strict). Returns σ²_cal [N, H]."""
    device = next(head.parameters()).device
    sw = torch.tensor(s2_within, dtype=torch.float32, device=device)
    sb = torch.tensor(s2_between, dtype=torch.float32, device=device)
    with torch.no_grad():
        s2_cal = head(sw, sb)
    return s2_cal.cpu().numpy()


# ============================================================================
# Cov95 + WIS evaluation (raw vs calibrated dual)
# ============================================================================
def eval_cov95_wis(mu: np.ndarray, s2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Returns (Cov95, WIS) over all (origin, horizon) pairs.

    T5 (2026-06-21): wrapper around src.eval.wis_standard.cov95_wis_from_gaussian
    — single source of truth. Numerical result is bit-identical to the previous
    inline implementation (Bracher 2021 Eq.(1) ≡ Eq.(4) ≡ 2·pinball.mean over 23
    FluSight quantiles, verified at 16-digit precision by
    src.tests.test_wis_standard.test_worked_example_*).
    """
    from src.eval.wis_standard import cov95_wis_from_gaussian
    return cov95_wis_from_gaussian(mu, s2, y)


def dual_report(mu: np.ndarray, s2_within: np.ndarray, s2_between: np.ndarray,
                y: np.ndarray, head: HeteroHead | None = None) -> dict:
    """Return both raw + calibrated (Cov95, WIS) for transparent §V-A disclosure."""
    s2_raw = s2_within + s2_between
    cov_raw, wis_raw = eval_cov95_wis(mu, s2_raw, y)
    out = dict(raw=dict(cov95=cov_raw, wis=wis_raw))
    if head is not None:
        s2_cal = apply_hetero_head(head, s2_within, s2_between)
        cov_cal, wis_cal = eval_cov95_wis(mu, s2_cal, y)
        out["calibrated"] = dict(cov95=cov_cal, wis=wis_cal,
                                   alpha=head.get_params()["alpha"],
                                   beta=head.get_params()["beta"])
    return out
