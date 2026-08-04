"""GaussianHMM — simpler HMM baseline for CG-Mamba §7.4 ablation.

**Status (M1.4)**: §7.4 ablation candidate, NOT the M1.4 main path.

The M1.4 main path uses `NeuralSwitchingVARHMM` (legacy CM-Mamba, GRU-based
neural switching + VAR emissions) via `src/models/hmm_stage1.py`. This module
provides a simpler i.i.d. Gaussian-emission HMM with Baum-Welch EM for paper
§7.4 ablation comparison:

Emission models contrasted:
    GaussianHMM:             p(x_t | z_t=k) = 𝒩(x_t; μ_k, Σ_k)
    NeuralSwitchingVARHMM:   p(x_t | z_t=k, x_{t-1}, h_{t-1}) =
                              𝒩(x_t; f_k(x_{t-1}, h_{t-1}), σ_k²)

ILI time-series has strong autocorrelation (ρ₁ ≈ 0.85-0.95), so the AR
emission *should* produce sharper posteriors. But for the CG-Mamba role —
coarse phase separation into K∈{3,4,5} regimes (baseline / onset / peak /
decline) — mean-level separation may dominate. The ablation directly tests
which dominates downstream MAE.

Paper §7.4 narrative outcomes:
  - Similar MAE → "phase separation is the contribution, HMM complexity
                   is secondary" (generality of CG-Mamba)
  - GaussianHMM significantly worse → "temporal-aware phase detection
                                       justifies NeuralSwitchingVARHMM"

Pipeline (used by scripts/m1_4_ablation_gaussian_hmm_search.py):
    1. Offline: GaussianHMM.fit(x_train) — EM on training ILI features (numpy)
    2. Offline: GaussianHMM.posteriors(x_full) — γ [T, K] for full series
    3. Online (M1.6): PhaseModule.forward(γ_batch) → gate_phase [B, L, D]

Cross-seed κ is provided by `src/utils/metrics.py:cohens_kappa_aligned`
(shared with main path; this module does not redefine κ).
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Numerics — local helper (no scipy dependency)
# ─────────────────────────────────────────────────────────────────

def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Numerically stable logsumexp.

    Args:
        a:    input array
        axis: reduction axis (None = flatten all)
    Returns:
        scalar if axis is None, else array with axis squeezed
    """
    if axis is None:
        c = np.max(a)
        if not np.isfinite(c):
            return float(c)
        return float(c + np.log(np.sum(np.exp(a - c))))
    else:
        c = np.max(a, axis=axis, keepdims=True)
        c_safe = np.where(np.isfinite(c), c, 0.0)
        result = c_safe + np.log(np.sum(np.exp(a - c_safe), axis=axis, keepdims=True))
        # Restore -inf where c was -inf
        result = np.where(np.isfinite(c), result, c)
        return np.squeeze(result, axis=axis)


# ─────────────────────────────────────────────────────────────────
# GaussianHMM — EM fitting + forward-backward (numpy, offline)
# ─────────────────────────────────────────────────────────────────

class GaussianHMM:
    """Gaussian-emission HMM with Baum-Welch EM (numpy, offline).

    Stationary transition matrix. Non-stationarity is captured by
    regime-switching states (PLAN v2.0.8b §3.3).

    All forward-backward computations in log-space for numerical stability.

    Args:
        n_states:        K — number of hidden states (search: K ∈ {3,4,5})
        n_features:      V — observation dimension (default 4)
        covariance_type: 'full' (V×V per state) or 'diag' (V per state)
        reg_covar:       regularization added to Σ diagonal
        n_iter:          max EM iterations
        tol:             log-likelihood convergence tolerance
        seed:            random seed for parameter initialization
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 4,
        covariance_type: str = "full",
        reg_covar: float = 1e-3,
        n_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
    ):
        self.K = n_states
        self.V = n_features
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

        # Parameters (set in fit())
        self.pi: np.ndarray | None = None       # [K]
        self.A: np.ndarray | None = None        # [K, K]
        self.means: np.ndarray | None = None    # [K, V]
        self.covars: np.ndarray | None = None   # [K, V, V] or [K, V]

        self.n_iter_run: int = 0
        self.ll_history: list[float] = []
        self._fitted = False

    # ── Initialization ────────────────────────────────────────────

    def _init_params(self, x: np.ndarray) -> None:
        """K-means-like initialization for EM warm start."""
        rng = np.random.RandomState(self.seed)
        T, V = x.shape
        K = self.K

        # π: uniform
        self.pi = np.full(K, 1.0 / K)

        # A: diagonal-dominant (0.9 self-transition, epidemic phases are sticky)
        off_diag = 0.1 / max(K - 1, 1)
        self.A = np.full((K, K), off_diag)
        np.fill_diagonal(self.A, 0.9)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Means: K random data points (spread initialization)
        idx = rng.choice(T, size=min(K, T), replace=False)
        self.means = x[idx].copy()

        # Covars: data variance (identity-scaled).
        # H-3 fix: do NOT add `+ self.reg_covar` here. The init covariance is
        # raw data variance; the emission likelihood (`_log_emission`, line 159
        # `cov_reg = self.covars[k] + self.reg_covar * np.eye(V)`) is the
        # single source of truth for regularization. M-step (line 277-279)
        # recomputes covars from posterior statistics without reg, so any init
        # reg leaks out after the first EM iteration anyway. Adding reg here
        # caused the first-step emission to use `var + 2·reg` (transiently),
        # which deviates from the reported reg_covar value (e.g., 5e-3 in
        # PLAN but effectively 1e-2 on iteration 0). After this fix, effective
        # reg = reg_covar everywhere (paper claim matches implementation).
        data_var = np.var(x, axis=0)
        if self.covariance_type == "full":
            self.covars = np.array([np.diag(data_var) for _ in range(K)])
        else:
            self.covars = np.tile(data_var, (K, 1))

    # ── Log emissions ─────────────────────────────────────────────

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        """log N(x_t | μ_k, Σ_k) → [T, K]."""
        T, V = x.shape
        K = self.K
        log_prob = np.empty((T, K))

        for k in range(K):
            diff = x - self.means[k]                               # [T, V]

            if self.covariance_type == "full":
                cov_reg = self.covars[k] + self.reg_covar * np.eye(V)
                try:
                    L = np.linalg.cholesky(cov_reg)
                except np.linalg.LinAlgError:
                    # Aggressive fallback regularization
                    cov_reg = cov_reg + 10 * self.reg_covar * np.eye(V)
                    L = np.linalg.cholesky(cov_reg)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                z = np.linalg.solve(L, diff.T)                    # [V, T]
                maha = np.sum(z ** 2, axis=0)                     # [T]
            else:  # diag
                var = self.covars[k] + self.reg_covar
                log_det = np.sum(np.log(var))
                maha = np.sum(diff ** 2 / var, axis=1)

            log_prob[:, k] = -0.5 * (V * np.log(2 * np.pi) + log_det + maha)

        return log_prob

    # ── Forward-backward (vectorized over K) ──────────────────────

    def _forward(self, log_emit: np.ndarray) -> np.ndarray:
        """Forward algorithm (log-space). [T, K] → log_α [T, K]."""
        T, K = log_emit.shape
        log_alpha = np.empty((T, K))
        log_A = np.log(np.maximum(self.A, 1e-300))

        log_alpha[0] = np.log(np.maximum(self.pi, 1e-300)) + log_emit[0]

        for t in range(1, T):
            # temp[j, k] = log_α[t-1, j] + log_A[j, k]  → [K, K]
            temp = log_alpha[t - 1, :, None] + log_A
            log_alpha[t] = _logsumexp(temp, axis=0) + log_emit[t]

        return log_alpha

    def _backward(self, log_emit: np.ndarray) -> np.ndarray:
        """Backward algorithm (log-space). [T, K] → log_β [T, K]."""
        T, K = log_emit.shape
        log_beta = np.empty((T, K))
        log_A = np.log(np.maximum(self.A, 1e-300))

        log_beta[T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            # temp[k, j] = log_A[k,j] + log_emit[t+1,j] + log_β[t+1,j]  → [K, K]
            temp = log_A + log_emit[t + 1][None, :] + log_beta[t + 1][None, :]
            log_beta[t] = _logsumexp(temp, axis=1)

        return log_beta

    # ── E-step ────────────────────────────────────────────────────

    def _e_step(self, x: np.ndarray):
        """Compute γ [T, K], ξ [T-1, K, K], log-likelihood."""
        T = x.shape[0]
        K = self.K

        log_emit = self._log_emission(x)
        log_alpha = self._forward(log_emit)
        log_beta = self._backward(log_emit)

        # γ
        log_gamma = log_alpha + log_beta                           # [T, K]
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(np.clip(log_gamma, -700, 0))
        gamma /= gamma.sum(axis=1, keepdims=True)                 # re-normalize

        # ξ
        log_A = np.log(np.maximum(self.A, 1e-300))
        xi = np.empty((T - 1, K, K))
        for t in range(T - 1):
            log_xi_t = (log_alpha[t, :, None]                     # [K, 1]
                        + log_A                                    # [K, K]
                        + log_emit[t + 1, None, :]                 # [1, K]
                        + log_beta[t + 1, None, :])                # [1, K]
            log_xi_t -= _logsumexp(log_xi_t)                       # normalize
            xi[t] = np.exp(np.clip(log_xi_t, -700, 0))

        # log-likelihood = logsumexp(log_α[T-1])
        ll = float(_logsumexp(log_alpha[-1]))

        return gamma, xi, ll

    # ── M-step ────────────────────────────────────────────────────

    def _m_step(self, x: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """Update π, A, μ, Σ from sufficient statistics."""
        T, V = x.shape
        K = self.K

        # π = γ[0] (normalized)
        self.pi = gamma[0].copy()
        self.pi = np.maximum(self.pi, 1e-300)
        self.pi /= self.pi.sum()

        # A[j, k] = Σ_t ξ[t,j,k] / Σ_t γ[t,j]
        xi_sum = xi.sum(axis=0)                                    # [K, K]
        denom = gamma[:-1].sum(axis=0)                             # [K]
        for j in range(K):
            if denom[j] > 1e-300:
                self.A[j] = xi_sum[j] / denom[j]
            else:
                self.A[j] = 1.0 / K
        self.A = np.maximum(self.A, 1e-300)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # μ_k, Σ_k
        for k in range(K):
            g_k = gamma[:, k]                                      # [T]
            n_k = g_k.sum()
            if n_k < 1e-300:
                continue

            # μ_k = Σ_t γ[t,k] · x_t / Σ_t γ[t,k]
            self.means[k] = (g_k[:, None] * x).sum(axis=0) / n_k

            # Σ_k = Σ_t γ[t,k] · (x_t - μ_k)(x_t - μ_k)^T / Σ_t γ[t,k]
            diff = x - self.means[k]                               # [T, V]
            if self.covariance_type == "full":
                self.covars[k] = (diff * g_k[:, None]).T @ diff / n_k
            else:
                self.covars[k] = (g_k[:, None] * diff ** 2).sum(axis=0) / n_k

    # ── Public API ────────────────────────────────────────────────

    def fit(self, x: np.ndarray) -> "GaussianHMM":
        """Baum-Welch EM algorithm.

        Args:
            x: [T, V] observation sequence (single contiguous segment)
        Returns:
            self (fitted)
        """
        assert x.ndim == 2 and x.shape[1] == self.V, \
            f"Expected [T, {self.V}], got {x.shape}"
        assert x.shape[0] >= self.K, \
            f"T={x.shape[0]} < K={self.K}: not enough data for {self.K} states"

        self._init_params(x)
        self.ll_history = []
        prev_ll = -np.inf

        for i in range(self.n_iter):
            gamma, xi, ll = self._e_step(x)
            self.ll_history.append(ll)

            if abs(ll - prev_ll) < self.tol:
                self.n_iter_run = i + 1
                break
            prev_ll = ll
            self._m_step(x, gamma, xi)
        else:
            self.n_iter_run = self.n_iter

        self._fitted = True
        return self

    def posteriors(self, x: np.ndarray) -> np.ndarray:
        """Soft posteriors γ[T, K] = P(z_t = k | x_{1:T}).

        Can be called on any sequence (train/val/test) after fitting.
        For gapped time series, call per contiguous segment separately.
        """
        assert self._fitted, "Call fit() first"
        assert x.ndim == 2 and x.shape[1] == self.V

        log_emit = self._log_emission(x)
        log_alpha = self._forward(log_emit)
        log_beta = self._backward(log_emit)

        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(np.clip(log_gamma, -700, 0))
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        """Most-likely state sequence [T] (integer). Used for Cohen's κ."""
        assert self._fitted
        assert x.ndim == 2 and x.shape[1] == self.V

        log_emit = self._log_emission(x)
        T, K = log_emit.shape
        log_A = np.log(np.maximum(self.A, 1e-300))

        # Viterbi DP
        V_mat = np.empty((T, K))
        bp = np.empty((T, K), dtype=int)

        V_mat[0] = np.log(np.maximum(self.pi, 1e-300)) + log_emit[0]

        for t in range(1, T):
            trans = V_mat[t - 1, :, None] + log_A                 # [K_from, K_to]
            bp[t] = trans.argmax(axis=0)                           # [K_to]
            V_mat[t] = trans.max(axis=0) + log_emit[t]

        # Backtrace
        states = np.empty(T, dtype=int)
        states[-1] = V_mat[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = bp[t + 1, states[t + 1]]
        return states

    def log_likelihood(self, x: np.ndarray) -> float:
        """Total log-likelihood P(x_{1:T} | model)."""
        assert self._fitted
        log_emit = self._log_emission(x)
        log_alpha = self._forward(log_emit)
        return float(_logsumexp(log_alpha[-1]))

    def bic(self, x: np.ndarray) -> float:
        """Bayesian Information Criterion. BIC = -2·LL + n_params·log(T).

        Lower BIC is better (penalizes model complexity).
        """
        ll = self.log_likelihood(x)
        T = x.shape[0]
        return -2 * ll + self._n_free_params() * np.log(T)

    def _n_free_params(self) -> int:
        """Number of free (estimable) parameters."""
        K, V = self.K, self.V
        n_pi = K - 1                               # initial probs (sum=1 constraint)
        n_A = K * (K - 1)                           # transition rows (each sums to 1)
        n_mu = K * V                                # means
        if self.covariance_type == "full":
            n_cov = K * V * (V + 1) // 2            # symmetric covariance matrices
        else:
            n_cov = K * V                            # diagonal variances
        return n_pi + n_A + n_mu + n_cov

    def dead_states(self, x: np.ndarray, threshold: float = 0.05) -> list[int]:
        """States with mean posterior mass < threshold.

        Dead states indicate over-parameterization (K too large) or
        poor initialization. PLAN §3.7: dead state → K candidate rejected.
        """
        gamma = self.posteriors(x)
        mean_post = gamma.mean(axis=0)                             # [K]
        return [int(k) for k in range(self.K) if mean_post[k] < threshold]

    # ── Serialization ─────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        """Serialize fitted HMM parameters to JSON-compatible dict."""
        return {
            "K": self.K,
            "V": self.V,
            "covariance_type": self.covariance_type,
            "reg_covar": self.reg_covar,
            "seed": self.seed,
            "n_iter_run": self.n_iter_run,
            "ll_history": self.ll_history,
            "pi": self.pi.tolist() if self.pi is not None else None,
            "A": self.A.tolist() if self.A is not None else None,
            "means": self.means.tolist() if self.means is not None else None,
            "covars": self.covars.tolist() if self.covars is not None else None,
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> "GaussianHMM":
        """Reconstruct fitted HMM from serialized dict."""
        hmm = cls(
            n_states=d["K"],
            n_features=d["V"],
            covariance_type=d["covariance_type"],
            reg_covar=d["reg_covar"],
            seed=d["seed"],
        )
        hmm.pi = np.array(d["pi"])
        hmm.A = np.array(d["A"])
        hmm.means = np.array(d["means"])
        hmm.covars = np.array(d["covars"])
        hmm.n_iter_run = d["n_iter_run"]
        hmm.ll_history = d.get("ll_history", [])
        hmm._fitted = True
        return hmm


# ─────────────────────────────────────────────────────────────────
# Synthetic data generator (for testing / smoke)
# ─────────────────────────────────────────────────────────────────

def generate_synthetic_hmm_data(
    K: int = 3,
    V: int = 4,
    T: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic HMM data for testing.

    Creates well-separated Gaussian clusters with sticky transitions.

    Returns:
        x:      [T, V] observations
        states: [T] true hidden states
    """
    rng = np.random.RandomState(seed)

    # Transition matrix (sticky)
    A = np.full((K, K), 0.05 / max(K - 1, 1))
    np.fill_diagonal(A, 0.95)
    A /= A.sum(axis=1, keepdims=True)

    # Well-separated means
    means = rng.randn(K, V) * 3.0

    # Isotropic covariances
    covars = np.array([np.eye(V) * (0.5 + 0.5 * k) for k in range(K)])

    # Generate state sequence
    pi = np.full(K, 1.0 / K)
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])

    # Generate observations
    x = np.empty((T, V))
    for t in range(T):
        x[t] = rng.multivariate_normal(means[states[t]], covars[states[t]])

    return x, states
