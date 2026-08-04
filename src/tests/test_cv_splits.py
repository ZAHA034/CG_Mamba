"""Tests for src/utils/cv_splits.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.cv_splits import kfold_window_indices, sliding_window_splits


# ──────────────────────────────────────────────────────────────────
# sliding_window_splits
# ──────────────────────────────────────────────────────────────────

class TestSlidingWindowSplits:

    def test_exp3_default(self):
        """T=48 (Exp3), L_win=24, h=5 → 20 windows per Direction Message 2.1."""
        splits = sliding_window_splits(T=48, L_win=24, horizon=5)
        assert len(splits) == 20

    def test_exp3_lwin_20(self):
        """T=48, L_win=20, h=5 → 24 windows."""
        splits = sliding_window_splits(T=48, L_win=20, horizon=5)
        assert len(splits) == 24

    def test_exp3_lwin_30(self):
        """T=48, L_win=30, h=5 → 14 windows."""
        splits = sliding_window_splits(T=48, L_win=30, horizon=5)
        assert len(splits) == 14

    def test_window_shape_invariants(self):
        """Each window: t_end_x - t_start = L_win, t_end_y - t_end_x = horizon."""
        L_win, h = 24, 5
        for t_start, t_end_x, t_end_y in sliding_window_splits(T=48, L_win=L_win, horizon=h):
            assert t_end_x - t_start == L_win
            assert t_end_y - t_end_x == h

    def test_first_window_starts_at_zero(self):
        splits = sliding_window_splits(T=48, L_win=24, horizon=5)
        assert splits[0] == (0, 24, 29)

    def test_last_window_ends_at_T(self):
        T = 48
        splits = sliding_window_splits(T=T, L_win=24, horizon=5, stride=1)
        assert splits[-1][2] == T

    def test_stride(self):
        """stride=2 halves the number of windows."""
        s1 = sliding_window_splits(T=48, L_win=24, horizon=5, stride=1)
        s2 = sliding_window_splits(T=48, L_win=24, horizon=5, stride=2)
        # 20 / 2 = 10 (ceil), depends on exact arithmetic
        assert len(s2) == (len(s1) + 1) // 2

    def test_insufficient_T_raises(self):
        """T < L_win + horizon → ValueError."""
        with pytest.raises(ValueError, match="no valid sliding window"):
            sliding_window_splits(T=20, L_win=24, horizon=5)

    def test_invalid_L_win(self):
        with pytest.raises(ValueError, match="L_win"):
            sliding_window_splits(T=48, L_win=0, horizon=5)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            sliding_window_splits(T=48, L_win=24, horizon=0)

    def test_invalid_stride(self):
        with pytest.raises(ValueError, match="stride"):
            sliding_window_splits(T=48, L_win=24, horizon=5, stride=0)


# ──────────────────────────────────────────────────────────────────
# kfold_window_indices
# ──────────────────────────────────────────────────────────────────

class TestKFoldWindowIndices:

    def test_5fold_on_20_windows(self):
        """20 windows / 5 folds = 4 val per fold."""
        folds = kfold_window_indices(20, n_folds=5)
        assert len(folds) == 5
        for tr, va in folds:
            assert len(va) == 4
            assert len(tr) == 16

    def test_disjoint_train_val(self):
        folds = kfold_window_indices(20, n_folds=5)
        for tr, va in folds:
            assert set(tr.tolist()).isdisjoint(va.tolist())

    def test_complete_coverage(self):
        """Across all folds, every window appears in exactly one val set."""
        folds = kfold_window_indices(20, n_folds=5)
        all_val = np.concatenate([va for _, va in folds])
        assert sorted(all_val.tolist()) == list(range(20))

    def test_contiguous_folds(self):
        """Default contiguous=True: val indices are consecutive."""
        folds = kfold_window_indices(20, n_folds=5)
        for _, va in folds:
            # consecutive integers
            assert np.all(np.diff(va) == 1)

    def test_round_robin_mode(self):
        """contiguous=False uses round-robin."""
        folds = kfold_window_indices(20, n_folds=5, contiguous=False)
        # Fold 0 → indices 0, 5, 10, 15
        assert folds[0][1].tolist() == [0, 5, 10, 15]

    def test_uneven_split(self):
        """n_windows % n_folds != 0 → first folds get extra elements."""
        folds = kfold_window_indices(22, n_folds=5)
        val_sizes = [len(va) for _, va in folds]
        assert val_sizes == [5, 5, 4, 4, 4]   # 22 = 5+5+4+4+4

    def test_n_folds_too_high(self):
        """n_folds > n_windows raises."""
        with pytest.raises(ValueError, match="n_windows"):
            kfold_window_indices(3, n_folds=5)

    def test_n_folds_below_2(self):
        with pytest.raises(ValueError, match="n_folds"):
            kfold_window_indices(20, n_folds=1)
