"""src/eval/bootstrap.py — region-cluster bootstrap utility (γ.5 채널 #1 lock)
================================================================================
용도:
  - PC0, PC2-a, kappa_recheck, e1_hpo 의 CI 통합 (ad-hoc 중복 제거)
  - paper §IV Table III/IV/V 의 모든 CI (reviewer #5 의 40/40 sign test
    pseudoreplication 가짜 통계 대체)
  - region-cluster bootstrap: 10 HHS region 을 *cluster 단위* resample
    (region-week naive 아님 — synchronized epidemics 라 region 독립 아님)

LOCKED protocol (γ.5):
  - cluster = region (per-region 단위), 10 HHS = 10 cluster
  - resample regions WITH replacement, B=1000 default
  - percentile 95% CI default
  - explicit seed (default 42, 재현성)

함수:
  cluster_bootstrap_mean(values_per_cluster, B, seed, level)
    → (mean, lo, hi)
  cluster_bootstrap_delta(values_a, values_b, B, seed, level)
    → (mean_delta, lo, hi)  # paired by cluster name
"""
from __future__ import annotations
from typing import Mapping
import numpy as np

DEFAULT_B = 1000
DEFAULT_SEED = 42
DEFAULT_LEVEL = 0.95


def _resample(values: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    out = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        out[b] = float(values[idx].mean())
    return out


def cluster_bootstrap_mean(
    per_cluster: Mapping[str, float] | np.ndarray,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> tuple[float, float, float]:
    """cluster 단위 (default: 10 HHS region) resample 후 mean 의 percentile CI.

    Args:
        per_cluster: {region_name: scalar} dict 또는 [n_cluster] ndarray
        B: bootstrap replicate 수
        seed: PRNG seed
        level: CI 신뢰수준 (0.95 → 95% CI)
    Returns:
        (point_mean, ci_lo, ci_hi)
    """
    if isinstance(per_cluster, Mapping):
        values = np.array(list(per_cluster.values()), dtype=np.float64)
    else:
        values = np.asarray(per_cluster, dtype=np.float64)
    assert values.ndim == 1, f"expected 1-D, got {values.shape}"
    assert len(values) >= 2, f"too few clusters: {len(values)}"
    rng = np.random.default_rng(seed)
    boot = _resample(values, B, rng)
    alpha = (1 - level) / 2
    return (float(values.mean()),
            float(np.percentile(boot, 100 * alpha)),
            float(np.percentile(boot, 100 * (1 - alpha))))


def cluster_bootstrap_delta(
    per_cluster_a: Mapping[str, float],
    per_cluster_b: Mapping[str, float],
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> tuple[float, float, float]:
    """Δ_cluster = a − b paired by cluster, then bootstrap mean(Δ).

    Args:
        per_cluster_a, per_cluster_b: {region: scalar} dicts. 키 정렬 일치 필요.
    Returns:
        (mean_delta, ci_lo, ci_hi)
    """
    common = sorted(set(per_cluster_a) & set(per_cluster_b))
    assert len(common) >= 2, f"too few paired clusters: {len(common)}"
    delta = np.array([per_cluster_a[k] - per_cluster_b[k] for k in common],
                      dtype=np.float64)
    return cluster_bootstrap_mean(delta, B=B, seed=seed, level=level)


def excludes_zero(ci_lo: float, ci_hi: float) -> bool:
    """CI 가 0 을 제외하는지 (= significant difference 신호)."""
    return (ci_lo > 0) or (ci_hi < 0)
