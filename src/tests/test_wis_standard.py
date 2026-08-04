"""src/tests/test_wis_standard.py — worked-example unit verification for T5.

Tests the field-standard WIS arithmetic at bit-level precision, and verifies
that the four PI constructors (Gaussian / sample / Method F / CQR) all flow
into the same Bracher 2021 scoring.

Worked example anchor (measured directly, three forms agree at ~3e-17):
    μ=1.0, σ=0.5, y=1.3 (Gaussian, FluSight 23-quantile)
    → WIS = 0.16789357902474250 (Bracher Eq.1 ≡ Eq.4 ≡ manual 2·pinball.mean()).

Note: Workflow A synthesis reported 0.167894 (rounded 6 d.p., off by ~4e-7
from this 16-digit anchor). 16-digit anchor is the canonical drift target.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.wis_standard import (
    FLUSIGHT_23,
    ALPHA_LEVELS,
    INTERVAL_PAIRS,
    REQUIRED_QUANTILES,
    quantiles_from_gaussian,
    quantiles_from_samples,
    calibrate_s_h,
    quantiles_method_f_calibrated,
    quantiles_conformal_cqr,
    cov95_wis_from_gaussian,
    coverage,
    wis,
    wis_via_quantile_loss,
)


# Worked-example anchor — must NOT drift across refactors.
# 16-digit measured anchor; three forms agree at ~3e-17.
EXPECTED_WIS = 0.16789357902474250


@pytest.fixture
def gaussian_worked_example():
    """μ=1.0, σ=0.5, y=1.3, FluSight 23 quantiles."""
    mu = np.array([1.0])
    sigma2 = np.array([0.25])
    y = np.array([1.3])
    return mu, sigma2, y


def test_gaussian_quantiles_shape_and_keys():
    mu = np.zeros(3); sigma2 = np.ones(3)
    qf = quantiles_from_gaussian(mu, sigma2)
    assert set(qf.keys()) == {float(t) for t in FLUSIGHT_23}
    for q in qf.values():
        assert q.shape == (3,)


def test_worked_example_gaussian_interval_form(gaussian_worked_example):
    """Bracher Eq.(1) interval form — primary form."""
    mu, sigma2, y = gaussian_worked_example
    qf = quantiles_from_gaussian(mu, sigma2)
    wis_value = float(wis(y, qf)[0])
    assert wis_value == pytest.approx(EXPECTED_WIS, abs=1e-12), (
        f"interval-form WIS={wis_value:.16f} drifted from anchor {EXPECTED_WIS}"
    )


def test_worked_example_gaussian_pinball_form(gaussian_worked_example):
    """Bracher Eq.(4) pinball aggregation — must equal Eq.(1)."""
    mu, sigma2, y = gaussian_worked_example
    qf = quantiles_from_gaussian(mu, sigma2)
    wis_via_pinball = float(wis_via_quantile_loss(y, qf)[0])
    assert wis_via_pinball == pytest.approx(EXPECTED_WIS, abs=1e-12)


def test_worked_example_legacy_wrapper(gaussian_worked_example):
    """cov95_wis_from_gaussian must reproduce the worked-example WIS."""
    mu, sigma2, y = gaussian_worked_example
    cov, wis_value = cov95_wis_from_gaussian(mu, sigma2, y)
    assert wis_value == pytest.approx(EXPECTED_WIS, abs=1e-12)


def test_coverage_inside_pi(gaussian_worked_example):
    """y=1.3 inside μ ± 1.96σ = [0.02, 1.98] → covered."""
    mu, sigma2, y = gaussian_worked_example
    qf = quantiles_from_gaussian(mu, sigma2)
    cov = coverage(y, qf, alpha=0.05)
    assert cov == 1.0


def test_samples_to_quantiles_recovers_gaussian():
    """Empirical-sample PI with many samples → matches Gaussian PI."""
    rng = np.random.default_rng(0)
    n = 1
    s = 200000
    samples = rng.normal(loc=1.0, scale=0.5, size=(n, s))
    qf_samp = quantiles_from_samples(samples, axis=-1)
    # Compare to analytic Gaussian
    qf_gauss = quantiles_from_gaussian(np.array([1.0]), np.array([0.25]))
    for tau in [0.025, 0.5, 0.975]:
        # tail-quantile sampling noise: std ~ sqrt(τ(1-τ)/S)/φ(z(τ)) · σ ≈ 0.003
        assert qf_samp[tau][0] == pytest.approx(qf_gauss[tau][0], abs=5e-3), (
            f"τ={tau}: samples={qf_samp[tau][0]} vs gauss={qf_gauss[tau][0]}"
        )


def test_method_f_recovers_gaussian_when_s_equals_one():
    """quantiles_method_f_calibrated(s_h=1) ≡ quantiles_from_gaussian."""
    mu = np.array([[1.0, 1.5], [0.5, 2.0]])
    sigma2 = np.array([[0.25, 0.36], [0.16, 0.49]])
    s_h_unit = np.ones(2)
    qf_method_f = quantiles_method_f_calibrated(mu, sigma2, s_h_unit)
    qf_gauss = quantiles_from_gaussian(mu, sigma2)
    for tau in FLUSIGHT_23:
        np.testing.assert_allclose(qf_method_f[float(tau)], qf_gauss[float(tau)],
                                    atol=1e-12)


def test_method_f_shrinks_when_s_less_than_one():
    """s_h < 1 should shrink PI width (the paper Method F mechanism)."""
    mu = np.zeros((10, 4))
    sigma2 = np.ones((10, 4))
    s_h = np.array([0.1, 0.2, 0.3, 0.4])
    qf = quantiles_method_f_calibrated(mu, sigma2, s_h)
    for h in range(4):
        width = qf[0.975][:, h] - qf[0.025][:, h]
        # PI width ∝ sqrt(s_h) → smaller s_h → narrower PI
        assert width[0] < 2 * 1.96, f"horizon {h}: width={width[0]} did not shrink"


def test_cqr_returns_all_23_quantiles():
    """quantiles_conformal_cqr must populate all 23 FluSight quantiles."""
    rng = np.random.default_rng(0)
    n_val, n_test = 100, 50
    mu_val = rng.normal(size=n_val)
    sigma2_val = np.ones(n_val) * 0.25
    y_val = mu_val + rng.normal(scale=0.5, size=n_val)
    base_val = quantiles_from_gaussian(mu_val, sigma2_val)
    mu_test = rng.normal(size=n_test)
    base_test = quantiles_from_gaussian(mu_test, np.ones(n_test) * 0.25)
    qf_cqr = quantiles_conformal_cqr(base_val, base_test, y_val)
    # CQR populates the 22 non-median quantiles + median
    expected_keys = set(round(a/2, 4) for a in ALPHA_LEVELS) | \
                     set(round(1-a/2, 4) for a in ALPHA_LEVELS) | {0.5}
    assert set(qf_cqr.keys()) >= expected_keys
    for q in qf_cqr.values():
        assert q.shape == (n_test,)


def test_cqr_achieves_target_coverage_on_iid_data():
    """CQR has finite-sample-corrected (1-α) coverage on exchangeable data."""
    rng = np.random.default_rng(42)
    n_val, n_test = 500, 500
    mu_val = rng.normal(loc=0.0, scale=1.0, size=n_val)
    y_val = mu_val + rng.normal(loc=0.0, scale=0.5, size=n_val)
    base_val = quantiles_from_gaussian(mu_val, np.ones(n_val) * 0.25)
    mu_test = rng.normal(loc=0.0, scale=1.0, size=n_test)
    y_test = mu_test + rng.normal(loc=0.0, scale=0.5, size=n_test)
    base_test = quantiles_from_gaussian(mu_test, np.ones(n_test) * 0.25)
    qf_cqr = quantiles_conformal_cqr(base_val, base_test, y_val)
    cov = coverage(y_test, qf_cqr, alpha=0.05)
    # Target 95% ± finite-sample wiggle; should be within ~0.95-0.97
    assert 0.93 <= cov <= 0.99, f"CQR coverage {cov} out of expected range"


def test_calibrate_s_h_recovers_unit_scale_when_gaussian():
    """If val data is truly Gaussian(μ, σ²), s_h should be ≈ 1.0."""
    rng = np.random.default_rng(0)
    N, H = 500, 4
    mu = rng.normal(size=(N, H))
    sigma2 = np.ones((N, H)) * 0.25
    y = mu + rng.normal(scale=0.5, size=(N, H))
    s_h = calibrate_s_h(mu, sigma2, y)
    np.testing.assert_allclose(s_h, np.ones(H), atol=0.5,
                                err_msg=f"s_h drift from 1 on Gaussian data: {s_h}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
