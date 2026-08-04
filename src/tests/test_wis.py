"""WIS infrastructure regression tests (PLAN J.10).

Bracher 2021 정의 일치 확인:
    1. interval_score 단일 PI 정의 (eq. 1) — 4 케이스 (in/below/above + width)
    2. wis() interval-form (eq. 3) vs wis_via_quantile_loss() pinball-form (§2.4)
       두 식이 동일 값을 산출 (≤ 1e-10 차이) — properscoring 부재 시 self-cross-check
    3. wis_decomposed 합계가 wis() 와 일치
    4. residual_quantiles_h_specific h-별 분리 (h=1 좁음, h=4 넓음)
    5. parametric_gaussian_quantiles z-score correctness
    6. ensemble_gaussian_quantiles 5-seed sample size 처리
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.wis import (
    REQUIRED_QUANTILES,
    ALPHA_LEVELS,
    INTERVAL_PAIRS,
    interval_score,
    wis,
    wis_decomposed,
    wis_via_quantile_loss,
    quantile_loss,
    coverage,
)
from src.eval.quantile_predictions import (
    residual_quantiles_h_specific,
    parametric_gaussian_quantiles,
    ensemble_gaussian_quantiles,
    ensemble_student_t_quantiles,
)


# ---------- Bracher 2021 eq. (1) interval_score ----------


def test_interval_score_inside():
    """y inside [l, u] → IS = width."""
    y = np.array([5.0])
    lo = np.array([3.0])
    hi = np.array([8.0])
    s = interval_score(y, lo, hi, alpha=0.1)
    assert np.allclose(s, [5.0])


def test_interval_score_below():
    """y < l → IS = width + (2/α)(l - y)."""
    y = np.array([1.0])
    lo = np.array([3.0])
    hi = np.array([8.0])
    alpha = 0.1
    expected = (8 - 3) + (2 / alpha) * (3 - 1)   # 5 + 40 = 45
    s = interval_score(y, lo, hi, alpha)
    assert np.allclose(s, [expected])


def test_interval_score_above():
    """y > u → IS = width + (2/α)(y - u)."""
    y = np.array([10.0])
    lo = np.array([3.0])
    hi = np.array([8.0])
    alpha = 0.1
    expected = (8 - 3) + (2 / alpha) * (10 - 8)  # 5 + 40 = 45
    s = interval_score(y, lo, hi, alpha)
    assert np.allclose(s, [expected])


def test_interval_score_vectorized():
    y = np.array([5.0, 1.0, 10.0])
    lo = np.array([3.0, 3.0, 3.0])
    hi = np.array([8.0, 8.0, 8.0])
    alpha = 0.1
    s = interval_score(y, lo, hi, alpha)
    expected = np.array([5.0, 5 + 40, 5 + 40])
    assert np.allclose(s, expected)


# ---------- Bracher 2021 eq. (3) wis ----------


def _make_symmetric_forecast(median: float, halfwidths: list[float]) -> dict[float, np.ndarray]:
    """Build a symmetric 23-quantile forecast around `median`.

    halfwidths: length 11 list; halfwidths[k] = half-width of the (1-α_k) PI.
    Median q=0.5 is centered on `median`.
    """
    assert len(halfwidths) == len(ALPHA_LEVELS)
    qf: dict[float, np.ndarray] = {0.5: np.array([median])}
    for hw, (q_lo, q_hi) in zip(halfwidths, INTERVAL_PAIRS):
        qf[q_lo] = np.array([median - hw])
        qf[q_hi] = np.array([median + hw])
    return qf


def test_wis_perfect_point_forecast():
    """All 23 quantiles equal to y → WIS = 0."""
    y = np.array([7.0])
    qf = {q: np.array([7.0]) for q in REQUIRED_QUANTILES}
    s = wis(y, qf)
    assert np.allclose(s, [0.0])


def test_wis_interval_form_equals_quantile_loss_form():
    """Bracher 2021 §2.4: WIS via interval-form ≡ (2/(2K+1)) * Σ_q QL_q.

    Cross-check that catches indexing/weight bugs in wis() vs wis_via_quantile_loss().
    """
    rng = np.random.default_rng(42)
    N = 50
    y = rng.normal(0, 1, size=N)
    # Random forecasts but order-preserving (q_{0.01} < q_{0.025} < ... < q_{0.99})
    base = rng.normal(0, 1, size=N)
    spread = rng.uniform(0.5, 2.0, size=N)
    z = np.array([-2.326, -1.96, -1.645, -1.282, -1.036, -0.842, -0.674,
                  -0.524, -0.385, -0.253, -0.126,
                  0.0,
                  0.126, 0.253, 0.385, 0.524, 0.674, 0.842, 1.036, 1.282,
                  1.645, 1.96, 2.326])
    qf = {q: base + spread * z[i] for i, q in enumerate(REQUIRED_QUANTILES)}

    interval_form = wis(y, qf)
    pinball_form = wis_via_quantile_loss(y, qf)
    assert np.allclose(interval_form, pinball_form, atol=1e-10), (
        f"Two WIS formulas disagree. max diff = "
        f"{np.max(np.abs(interval_form - pinball_form))}"
    )


def test_wis_decomposed_sum_equals_wis():
    rng = np.random.default_rng(123)
    N = 30
    y = rng.normal(0, 1, size=N)
    base = rng.normal(0, 1, size=N)
    z = np.array([-2.326, -1.96, -1.645, -1.282, -1.036, -0.842, -0.674,
                  -0.524, -0.385, -0.253, -0.126,
                  0.0,
                  0.126, 0.253, 0.385, 0.524, 0.674, 0.842, 1.036, 1.282,
                  1.645, 1.96, 2.326])
    qf = {q: base + z[i] for i, q in enumerate(REQUIRED_QUANTILES)}

    direct = wis(y, qf)
    parts = wis_decomposed(y, qf)
    assert np.allclose(parts["total"], direct, atol=1e-12)
    assert np.allclose(parts["dispersion"] + parts["under"] + parts["over"],
                       direct, atol=1e-12)


def test_wis_missing_quantile_raises():
    y = np.array([0.0])
    qf = {q: np.array([0.0]) for q in REQUIRED_QUANTILES if q != 0.5}
    with pytest.raises(ValueError, match="missing"):
        wis(y, qf)


def test_wis_symmetric_above_median():
    """Sanity: symmetric forecast centered at 0, y=0 → WIS = average half-width.

    For a perfectly centered y at the median, IS_α = width = 2*halfwidth for
    all α. WIS = (1/(K+0.5)) * [0 + Σ_k (α_k/2) * 2*hw_k]
              = (1/(K+0.5)) * Σ_k α_k * hw_k
    """
    median = 0.0
    halfwidths = [1.0] * len(ALPHA_LEVELS)
    qf = _make_symmetric_forecast(median, halfwidths)
    y = np.array([0.0])
    s = wis(y, qf)
    K = len(ALPHA_LEVELS)
    expected = sum(alpha * hw for alpha, hw in zip(ALPHA_LEVELS, halfwidths)) / (K + 0.5)
    assert np.allclose(s, [expected])


def test_quantile_loss_basic():
    """QL_q(y, q_hat) — manual checks for q=0.5 (= 0.5 * |y - q_hat|)."""
    y = np.array([1.0, 5.0])
    q_hat = np.array([3.0, 3.0])
    ql = quantile_loss(y, q_hat, q=0.5)
    # y=1, q_hat=3 → (1-3)*(0.5 - 1) = -2 * -0.5 = 1.0
    # y=5, q_hat=3 → (5-3)*(0.5 - 0) = 2 * 0.5 = 1.0
    assert np.allclose(ql, [1.0, 1.0])


def test_coverage_basic():
    qf = {0.05: np.array([0.0, 0.0]), 0.95: np.array([1.0, 1.0])}
    # one inside, one above
    y = np.array([0.5, 2.0])
    cov = coverage(y, qf, alpha=0.1)
    assert cov == 0.5


# ---------- quantile_predictions ----------


def test_residual_quantiles_h_specific_shape():
    """h-별 residual 분리: h=1과 h=4가 다른 quantile."""
    point_preds = np.zeros((10, 4))
    val_residuals_per_h = [
        np.array([-0.1, 0.0, 0.1]),       # h=1 narrow
        np.array([-0.2, 0.0, 0.2]),
        np.array([-0.3, 0.0, 0.3]),
        np.array([-0.5, 0.0, 0.5]),       # h=4 wide
    ]
    qf = residual_quantiles_h_specific(point_preds, val_residuals_per_h)
    # q=0.5 should be median residual = 0.0 everywhere
    assert np.allclose(qf[0.5], 0.0)
    # q=0.99 widens with horizon
    q99 = qf[0.99]
    assert q99[0, 0] < q99[0, 3], "h=4 q=0.99 should exceed h=1 q=0.99"


def test_residual_quantiles_h_specific_length_mismatch():
    point_preds = np.zeros((5, 4))
    val_residuals_per_h = [np.array([0.0]), np.array([0.0])]  # only 2, need 4
    with pytest.raises(ValueError, match="length"):
        residual_quantiles_h_specific(point_preds, val_residuals_per_h)


def test_parametric_gaussian_quantiles_z_scores():
    """N(0,1) → q=0.5 is 0, q=0.975 ≈ 1.96."""
    mean = np.zeros((1, 1))
    var = np.ones((1, 1))
    qf = parametric_gaussian_quantiles(mean, var)
    assert np.allclose(qf[0.5], 0.0)
    assert np.allclose(qf[0.975], 1.959964, atol=1e-3)
    assert np.allclose(qf[0.025], -1.959964, atol=1e-3)


def test_parametric_gaussian_zero_variance():
    """Zero variance → all quantiles at the mean."""
    mean = np.full((1, 1), 5.0)
    var = np.zeros((1, 1))
    qf = parametric_gaussian_quantiles(mean, var)
    for q in REQUIRED_QUANTILES:
        assert np.allclose(qf[q], 5.0)


def test_ensemble_gaussian_quantiles_basic():
    """5 members at the same value → zero std → all quantiles at the mean."""
    member_preds = np.ones((5, 3, 4))
    qf = ensemble_gaussian_quantiles(member_preds)
    assert np.allclose(qf[0.5], 1.0)
    assert np.allclose(qf[0.99], 1.0)
    assert np.allclose(qf[0.01], 1.0)


def test_ensemble_gaussian_recovers_mean_and_std():
    """Members ~ N(2, 0.5²) → q=0.5 ≈ 2.0, q=0.975 ≈ 2 + 1.96*0.5."""
    rng = np.random.default_rng(7)
    members = rng.normal(2.0, 0.5, size=(5000, 1, 1))
    qf = ensemble_gaussian_quantiles(members)
    assert np.allclose(qf[0.5], 2.0, atol=0.05)
    assert np.allclose(qf[0.975], 2.0 + 1.96 * 0.5, atol=0.05)


def test_ensemble_student_t_wider_than_gaussian():
    """Student-t df=4 has heavier tails than Gaussian at fixed scale."""
    members = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(5, 1, 1)
    g = ensemble_gaussian_quantiles(members)
    t = ensemble_student_t_quantiles(members, df=4)
    assert float(t[0.99].item()) > float(g[0.99].item())
    assert float(t[0.01].item()) < float(g[0.01].item())
