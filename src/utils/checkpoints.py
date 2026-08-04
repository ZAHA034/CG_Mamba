"""Stage 1 → Stage 2 checkpoint loaders (M1.6, v2.0.9).

Public API:
    load_fitted_hmm(run_dir) -> GaussianHMM
        Reconstruct a fitted GaussianHMM instance from a Stage 1 main run's
        `hmm_params.npz` (C1 metadata: A, pi, means, covars, reg_covar, K, V,
        covariance_type, n_iter_run, final_ll). The returned instance is
        ready for PhaseModule._cache_hmm_torch caching (T-1 entry sequence).

Stage 1 → Stage 2 entry sequence (T-1, PLAN v2.0.9 §5.1 D.5.2):
    >>> from pathlib import Path
    >>> from src.utils.checkpoints import load_fitted_hmm
    >>> from src.models import CGForecaster
    >>> from src.utils.config import CGMambaConfig
    >>>
    >>> cfg = CGMambaConfig()
    >>> model = CGForecaster(cfg)
    >>> hmm = load_fitted_hmm(Path('runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed42'))
    >>> model.prepare_for_stage2(hmm)        # M-4: HMM cache + Env decoder freeze
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=...)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.gaussian_hmm import GaussianHMM


def load_fitted_hmm(run_dir: Path) -> GaussianHMM:
    """Reconstruct a fitted GaussianHMM from a Stage 1 `hmm_params.npz`.

    The npz must contain the full C1 metadata produced by
    `scripts/m1_4_phase_dynamics_main.py` / `m1_4_phase_dynamics_search.py`:
        A, pi, means, covars, reg_covar, K, V, covariance_type, n_iter_run, final_ll.

    Returns a GaussianHMM instance with `._fitted = True`, ready for
    `PhaseModule._cache_hmm_torch(fitted_hmm)` (T-1 Stage 2 entry).

    Args:
        run_dir: Path to a single Stage 1 seed directory containing
                 `hmm_params.npz` (e.g., `runs/m1_4_phase_dynamics_main/
                 V_raw3_regcov5e-03_K3_seed42/`).

    Returns:
        Fitted GaussianHMM with all artifacts populated.

    Raises:
        FileNotFoundError: if `hmm_params.npz` does not exist in `run_dir`.
        KeyError: if any required metadata key is missing from the npz.
    """
    npz_path = Path(run_dir) / "hmm_params.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"hmm_params.npz not found at {npz_path}. "
            f"Stage 1 main run output is required (see "
            f"scripts/m1_4_phase_dynamics_main.py)."
        )
    d = np.load(npz_path, allow_pickle=False)

    hmm = GaussianHMM(
        n_states=int(d["K"]),
        n_features=int(d["V"]),
        covariance_type=str(d["covariance_type"]),
        reg_covar=float(d["reg_covar"]),
    )
    hmm.A = d["A"].astype(np.float64)
    hmm.pi = d["pi"].astype(np.float64)
    hmm.means = d["means"].astype(np.float64)
    hmm.covars = d["covars"].astype(np.float64)
    hmm.n_iter_run = int(d["n_iter_run"])
    hmm._fitted = True
    return hmm
