"""Tests for src/data/preprocess_whitfield.py.

Coverage:
    - parse_sample_title: regex robustness
    - load_series_matrix: shape + 'null' → NaN
    - split_by_experiment: Exp3=48, Exp4=17
    - normalize_expression: drop / fill / z-score invariants
    - extract_probe_metadata: SOFT file parsing
    - preprocess_whitfield: end-to-end smoke + output schema + reproducibility

A few tests depend on the actual GSE3497 data at the default raw_dir. They
skip cleanly if the data is absent, so the module remains testable in CI
environments without the dataset.
"""
from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.preprocess_whitfield import (
    DEFAULT_RAW_DIR,
    EXP_TO_SYNC,
    ExperimentSplit,
    WhitfieldPreprocessConfig,
    align_to_train_features,
    extract_probe_metadata,
    load_preprocessed,
    load_series_matrix,
    normalize_expression,
    parse_sample_title,
    preprocess_whitfield,
    split_by_experiment,
)


# ──────────────────────────────────────────────────────────────────
# Data availability gate
# ──────────────────────────────────────────────────────────────────

WHITFIELD_AVAILABLE = (
    DEFAULT_RAW_DIR.exists()
    and (DEFAULT_RAW_DIR / "GSE3497-GPL3001_series_matrix.txt.gz").exists()
)
requires_whitfield = pytest.mark.skipif(
    not WHITFIELD_AVAILABLE,
    reason=f"Whitfield raw data not available at {DEFAULT_RAW_DIR}",
)


# ──────────────────────────────────────────────────────────────────
# parse_sample_title
# ──────────────────────────────────────────────────────────────────

class TestParseSampleTitle:

    def test_thy_noc_format(self):
        exp, tp = parse_sample_title("26h Thy-Noc, Exp4")
        assert exp == "Exp4"
        assert tp == 26.0

    def test_double_thymidine_format(self):
        exp, tp = parse_sample_title("0 hr Hela Double Thymidine, exp3")
        assert exp == "Exp3"
        assert tp == 0.0

    def test_decimal_timepoint(self):
        exp, tp = parse_sample_title("2.5 hr X, Exp2")
        assert exp == "Exp2"
        assert tp == 2.5

    def test_lower_then_upper_exp(self):
        """Caps-insensitive on 'exp' token."""
        exp, _ = parse_sample_title("36hr Thy, exp3")
        assert exp == "Exp3"

    def test_extra_quotes_stripped(self):
        exp, tp = parse_sample_title('"""4 hr Hela DT, exp3"""')
        assert exp == "Exp3"
        assert tp == 4.0

    def test_missing_time_returns_none(self):
        exp, tp = parse_sample_title("Reference channel, exp3")
        assert exp == "Exp3"
        assert tp is None

    def test_missing_exp_returns_none(self):
        exp, tp = parse_sample_title("26h biological replicate")
        assert exp is None
        assert tp == 26.0

    def test_empty_string(self):
        exp, tp = parse_sample_title("")
        assert exp is None
        assert tp is None


# ──────────────────────────────────────────────────────────────────
# normalize_expression
# ──────────────────────────────────────────────────────────────────

class TestNormalizeExpression:

    def test_drops_probes_above_threshold(self):
        T, V = 10, 5
        x = np.ones((T, V), dtype=np.float32)
        # Column 0: all NaN → drop. Column 1: 60% NaN → drop @ thr=0.5
        x[:, 0] = np.nan
        x[:6, 1] = np.nan
        # Column 2: 30% NaN → keep
        x[:3, 2] = np.nan
        x_clean, kept = normalize_expression(
            x, drop_nan_threshold=0.5, fill_strategy="zero", normalize=False,
        )
        assert set(kept.tolist()) == {2, 3, 4}
        assert x_clean.shape == (T, 3)

    def test_z_score_unit_variance(self):
        rng = np.random.RandomState(0)
        x = rng.randn(20, 5).astype(np.float32) * 3.0 + 10.0
        x_norm, _ = normalize_expression(
            x, drop_nan_threshold=0.5, fill_strategy="zero", normalize=True,
        )
        np.testing.assert_allclose(x_norm.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(x_norm.std(axis=0), 1.0, atol=1e-5)

    def test_no_nan_in_output(self):
        x = np.full((10, 4), np.nan, dtype=np.float32)
        x[0, 0] = 1.0
        x[5, 0] = 2.0
        x_clean, _ = normalize_expression(
            x, drop_nan_threshold=0.99, fill_strategy="interpolate", normalize=False,
        )
        assert not np.isnan(x_clean).any()

    def test_constant_probe_avoids_div_by_zero(self):
        x = np.ones((10, 3), dtype=np.float32)
        # All probes constant → std=0 → guard kicks in
        x_norm, _ = normalize_expression(
            x, drop_nan_threshold=0.5, fill_strategy="zero", normalize=True,
        )
        # Should be zero-mean (mean subtracted) with no NaN
        assert not np.isnan(x_norm).any()
        np.testing.assert_allclose(x_norm, 0.0, atol=1e-6)

    def test_interpolate_fills_internal_nan(self):
        x = np.array([
            [1.0, 1.0],
            [2.0, np.nan],
            [3.0, 3.0],
        ], dtype=np.float32)
        x_clean, _ = normalize_expression(
            x, drop_nan_threshold=0.5, fill_strategy="interpolate", normalize=False,
        )
        # Interior NaN should be linearly interpolated to 2.0
        np.testing.assert_allclose(x_clean[1, 1], 2.0, atol=1e-6)

    def test_interpolate_extends_edge_nan(self):
        x = np.array([
            [np.nan],
            [1.0],
            [2.0],
        ], dtype=np.float32)
        x_clean, _ = normalize_expression(
            x, drop_nan_threshold=0.5, fill_strategy="interpolate", normalize=False,
        )
        # Leading NaN should be edge-extended to the first valid value (1.0)
        assert x_clean[0, 0] == 1.0

    def test_invalid_fill_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown fill_strategy"):
            normalize_expression(np.zeros((3, 2)), fill_strategy="bogus")


# ──────────────────────────────────────────────────────────────────
# split_by_experiment
# ──────────────────────────────────────────────────────────────────

class TestSplitByExperiment:

    def test_sorts_by_timepoint(self):
        gsm_ids = ["GSM1", "GSM2", "GSM3"]
        titles = {
            "GSM1": "10 hr X, Exp3",
            "GSM2": "0 hr X, Exp3",
            "GSM3": "5 hr X, Exp3",
        }
        splits = split_by_experiment(gsm_ids, titles)
        assert "Exp3" in splits
        s = splits["Exp3"]
        assert s.timepoints_hours == [0.0, 5.0, 10.0]
        assert s.gsm_ids == ["GSM2", "GSM3", "GSM1"]
        assert s.column_indices == [1, 2, 0]   # original col indices

    def test_assigns_sync_method(self):
        gsm_ids = ["GSM1", "GSM2"]
        titles = {
            "GSM1": "0 hr DT, Exp3",
            "GSM2": "26h Thy-Noc, Exp4",
        }
        splits = split_by_experiment(gsm_ids, titles)
        assert splits["Exp3"].sync_method == "thy"
        assert splits["Exp4"].sync_method == "thy_noc"

    def test_skips_unparseable(self):
        gsm_ids = ["GSM1", "GSM2"]
        titles = {
            "GSM1": "0 hr, Exp3",
            "GSM2": "reference channel",   # no exp, no time
        }
        splits = split_by_experiment(gsm_ids, titles)
        assert len(splits["Exp3"].gsm_ids) == 1

    def test_canonical_exp_to_sync_dict(self):
        """Canonical mapping covers all 5 Whitfield experiments."""
        for exp in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]:
            assert exp in EXP_TO_SYNC


# ──────────────────────────────────────────────────────────────────
# load_series_matrix (synthetic mini-file)
# ──────────────────────────────────────────────────────────────────

class TestLoadSeriesMatrix:

    def _write_mini_series_matrix(self, tmp_path: Path) -> Path:
        """Build a tiny but format-correct .txt.gz fixture."""
        content = (
            "!Series_title\t\"toy\"\n"
            "!Sample_title\t\"0h thy, Exp3\"\t\"2h thy, Exp3\"\t\"26h Thy-Noc, Exp4\"\n"
            "!series_matrix_table_begin\n"
            '"ID_REF"\t"GSM_A"\t"GSM_B"\t"GSM_C"\n'
            "1\t0.5\t-0.3\tnull\n"
            "2\t1.1\t0.0\t-1.5\n"
            "3\tnull\tnull\t2.0\n"
            "!series_matrix_table_end\n"
        )
        path = tmp_path / "mini_series.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def test_shape_and_null_handling(self, tmp_path):
        path = self._write_mini_series_matrix(tmp_path)
        expr, gsm_ids, probe_ids, titles = load_series_matrix(path)
        assert expr.shape == (3, 3)               # 3 probes × 3 samples
        assert gsm_ids == ["GSM_A", "GSM_B", "GSM_C"]
        assert probe_ids == ["1", "2", "3"]
        # 'null' → NaN
        assert np.isnan(expr[0, 2])
        assert np.isnan(expr[2, 0])
        assert np.isnan(expr[2, 1])
        # Normal values preserved
        np.testing.assert_allclose(expr[0, 0], 0.5, atol=1e-6)
        np.testing.assert_allclose(expr[1, 2], -1.5, atol=1e-6)

    def test_sample_titles_mapped(self, tmp_path):
        path = self._write_mini_series_matrix(tmp_path)
        _, _, _, titles = load_series_matrix(path)
        assert titles["GSM_A"] == "0h thy, Exp3"
        assert titles["GSM_C"] == "26h Thy-Noc, Exp4"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Series matrix not found"):
            load_series_matrix(tmp_path / "nonexistent.txt.gz")


# ──────────────────────────────────────────────────────────────────
# extract_probe_metadata (synthetic mini SOFT)
# ──────────────────────────────────────────────────────────────────

class TestExtractProbeMetadata:

    def _write_mini_soft(self, tmp_path: Path) -> Path:
        """Tiny SOFT fixture with one target platform and noise platforms."""
        content = (
            "^DATABASE = GeoMiniDB\n"
            "^PLATFORM = GPL_OTHER\n"
            "!platform_table_begin\n"
            "ID\tSPOT_ID\tGB_LIST\n"
            "X\tIGNORE\tNOTHING\n"
            "!platform_table_end\n"
            "^PLATFORM = GPL3001\n"
            "!platform_table_begin\n"
            "ID\tSPOT_ID\tGB_LIST\tPOLYMER\tTYPE\n"
            "101\tIMAGE:1234\tAA111,AA222\tDNA\tcDNA_clone\n"
            "102\tIMAGE:5678\tBB333\tDNA\tcDNA_clone\n"
            "!platform_table_end\n"
        )
        path = tmp_path / "mini.soft.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def test_extracts_target_platform_only(self, tmp_path):
        path = self._write_mini_soft(tmp_path)
        meta = extract_probe_metadata(path, platform="GPL3001")
        assert set(meta.keys()) == {"101", "102"}
        assert meta["101"]["image_clone"] == "IMAGE:1234"
        assert meta["101"]["gb_accessions"] == ["AA111", "AA222"]
        assert meta["102"]["gb_accessions"] == ["BB333"]

    def test_missing_file_returns_empty(self, tmp_path):
        meta = extract_probe_metadata(tmp_path / "nonexistent.soft.gz")
        assert meta == {}

    def test_missing_platform_returns_empty(self, tmp_path):
        path = self._write_mini_soft(tmp_path)
        meta = extract_probe_metadata(path, platform="GPL_DOESNT_EXIST")
        assert meta == {}


# ──────────────────────────────────────────────────────────────────
# WhitfieldPreprocessConfig validation
# ──────────────────────────────────────────────────────────────────

class TestConfig:

    def test_defaults(self):
        cfg = WhitfieldPreprocessConfig()
        assert cfg.platform == "GPL3001"
        assert cfg.drop_nan_threshold == 0.5
        assert cfg.fill_nan_strategy == "interpolate"
        assert cfg.normalize is True

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="drop_nan_threshold"):
            WhitfieldPreprocessConfig(drop_nan_threshold=1.5)

    def test_invalid_fill_strategy_raises(self):
        with pytest.raises(ValueError, match="fill_nan_strategy"):
            WhitfieldPreprocessConfig(fill_nan_strategy="bogus")


# ──────────────────────────────────────────────────────────────────
# End-to-end pipeline (requires actual Whitfield data)
# ──────────────────────────────────────────────────────────────────

@requires_whitfield
class TestEndToEnd:

    def test_smoke_run(self, tmp_path):
        cfg = WhitfieldPreprocessConfig(
            output_dir=tmp_path,
            extract_probe_metadata=False,   # skip 344MB SOFT decompression for speed
        )
        summary = preprocess_whitfield(cfg)

        # Top-level shape
        assert summary["n_probes"] == 44160
        assert summary["n_samples"] == 65
        assert set(summary["experiments"]) == {"Exp3", "Exp4"}

        # Per-experiment counts (the project's primary cross-domain contract)
        exp3 = summary["per_experiment"]["Exp3"]
        exp4 = summary["per_experiment"]["Exp4"]
        assert exp3["n_samples"] == 48
        assert exp3["sync_method"] == "thy"
        assert exp4["n_samples"] == 17
        assert exp4["sync_method"] == "thy_noc"

        # Output files exist
        for path_str in summary["output_paths"].values():
            assert Path(path_str).exists(), f"Missing output: {path_str}"

    def test_output_shapes(self, tmp_path):
        cfg = WhitfieldPreprocessConfig(
            output_dir=tmp_path,
            extract_probe_metadata=False,
        )
        preprocess_whitfield(cfg)

        # Load and check shapes
        exp3 = load_preprocessed(tmp_path, exp_id="Exp3")
        exp4 = load_preprocessed(tmp_path, exp_id="Exp4")
        assert exp3["x"].ndim == 2 and exp3["x"].shape[0] == 48
        assert exp4["x"].ndim == 2 and exp4["x"].shape[0] == 17
        # Each experiment kept the same probes it normalizes against
        assert exp3["x"].shape[1] == len(exp3["kept_probe_indices"])
        assert exp4["x"].shape[1] == len(exp4["kept_probe_indices"])

    def test_no_nan_in_outputs(self, tmp_path):
        cfg = WhitfieldPreprocessConfig(
            output_dir=tmp_path,
            extract_probe_metadata=False,
        )
        preprocess_whitfield(cfg)
        for exp_id in ["Exp3", "Exp4"]:
            data = load_preprocessed(tmp_path, exp_id=exp_id)
            assert not np.isnan(data["x"]).any(), f"NaN in {exp_id} output"

    def test_z_score_normalization(self, tmp_path):
        cfg = WhitfieldPreprocessConfig(
            output_dir=tmp_path,
            extract_probe_metadata=False,
        )
        preprocess_whitfield(cfg)
        for exp_id in ["Exp3", "Exp4"]:
            data = load_preprocessed(tmp_path, exp_id=exp_id)
            x = data["x"]
            # Per-probe mean ≈ 0, std ≈ 1 (constant probes get std=1 fallback)
            np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-4)
            # std could be 0 for constant probes that survived; allow
            stds = x.std(axis=0)
            assert ((stds < 1e-3) | (np.abs(stds - 1.0) < 1e-3)).all()

    def test_reproducibility(self, tmp_path):
        cfg = WhitfieldPreprocessConfig(
            output_dir=tmp_path / "run1",
            extract_probe_metadata=False,
        )
        preprocess_whitfield(cfg)
        cfg2 = WhitfieldPreprocessConfig(
            output_dir=tmp_path / "run2",
            extract_probe_metadata=False,
        )
        preprocess_whitfield(cfg2)
        a = load_preprocessed(tmp_path / "run1", exp_id="Exp3")
        b = load_preprocessed(tmp_path / "run2", exp_id="Exp3")
        np.testing.assert_array_equal(a["x"], b["x"])
        np.testing.assert_array_equal(a["timepoints_hours"], b["timepoints_hours"])

    def test_metadata_json_schema(self, tmp_path):
        cfg = WhitfieldPreprocessConfig(
            output_dir=tmp_path,
            extract_probe_metadata=False,
        )
        preprocess_whitfield(cfg)
        sm = json.loads((tmp_path / "sample_metadata.json").read_text())
        assert sm["platform"] == "GPL3001"
        assert sm["n_samples_total"] == 65
        assert sm["experiments"]["Exp3"]["sync_method"] == "thy"
        assert sm["experiments"]["Exp4"]["sync_method"] == "thy_noc"

    def test_load_preprocessed_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_preprocessed(tmp_path, exp_id="Exp3")


# ──────────────────────────────────────────────────────────────────
# align_to_train_features (P1.1)
# ──────────────────────────────────────────────────────────────────

class TestAlignToTrainFeatures:
    """Cross-experiment probe alignment for train→test transfer."""

    def test_perfect_overlap(self):
        """Identical kept_indices → bijection (no missing features)."""
        train_kept = np.array([0, 1, 2, 5, 8])
        test_kept = np.array([0, 1, 2, 5, 8])
        # Pick features at train-local positions 1 and 3 (orig probes 1 and 5)
        feature_idx = [1, 3]
        tr, te = align_to_train_features(train_kept, test_kept, feature_idx)
        assert tr == [1, 3]
        assert te == [1, 3]   # same positions because train_kept == test_kept

    def test_offset_overlap(self):
        """Different kept_indices but feature probes present in both → remap."""
        train_kept = np.array([0, 2, 4, 6, 8])    # train kept evens
        test_kept = np.array([2, 4, 6, 8])        # test dropped probe 0
        # Feature: train-local idx 2 → original probe 4
        feature_idx = [2]
        tr, te = align_to_train_features(train_kept, test_kept, feature_idx)
        assert tr == [2]
        assert te == [1]   # probe 4 is at position 1 in test_kept

    def test_partial_drop_with_warning(self):
        """Some features missing in test → drop + UserWarning."""
        train_kept = np.array([0, 1, 2, 3, 4])
        test_kept = np.array([0, 2, 4])   # test dropped probes 1 and 3
        # Features pointing at train-local 1 (probe 1, missing) and 2 (probe 2, present)
        feature_idx = [1, 2]
        with pytest.warns(UserWarning, match="no surviving probe"):
            tr, te = align_to_train_features(train_kept, test_kept, feature_idx)
        assert tr == [2]     # only the present one
        assert te == [1]     # probe 2 at position 1 in test_kept

    def test_zero_overlap(self):
        """No common probes → empty result with warning."""
        train_kept = np.array([0, 1, 2])
        test_kept = np.array([10, 11, 12])
        feature_idx = [0, 1]
        with pytest.warns(UserWarning, match="2 of 2 features"):
            tr, te = align_to_train_features(train_kept, test_kept, feature_idx)
        assert tr == []
        assert te == []

    def test_empty_features(self):
        """Empty feature list → empty result, no warning."""
        train_kept = np.array([0, 1, 2])
        test_kept = np.array([0, 1, 2])
        tr, te = align_to_train_features(train_kept, test_kept, [])
        assert tr == []
        assert te == []

    def test_out_of_bounds_raises(self):
        """Feature index outside [0, V_train) raises ValueError."""
        train_kept = np.array([0, 1, 2])
        test_kept = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="train_local_feature_indices"):
            align_to_train_features(train_kept, test_kept, [5])
        with pytest.raises(ValueError, match="train_local_feature_indices"):
            align_to_train_features(train_kept, test_kept, [-1])

    def test_preserves_input_order(self):
        """Output order matches input feature_idx order (not probe ID order)."""
        train_kept = np.array([10, 20, 30, 40])
        test_kept = np.array([10, 20, 30, 40])
        feature_idx = [3, 0, 2]   # arbitrary ordering
        tr, te = align_to_train_features(train_kept, test_kept, feature_idx)
        assert tr == [3, 0, 2]
        assert te == [3, 0, 2]

    def test_accepts_numpy_array(self):
        """Numpy array input for feature_idx works (not just list)."""
        train_kept = np.array([0, 1, 2])
        test_kept = np.array([0, 1, 2])
        tr, te = align_to_train_features(train_kept, test_kept, np.array([0, 1]))
        assert tr == [0, 1]
        assert te == [0, 1]


@requires_whitfield
class TestAlignmentOnRealData:
    """Verify alignment helper works on the actual Exp3/Exp4 preprocessed data."""

    def test_exp3_to_exp4_alignment(self, tmp_path):
        """Real-data smoke: top-100 variance features in Exp3 should mostly
        survive into Exp4 (high probe overlap, ~99.5%)."""
        cfg = WhitfieldPreprocessConfig(
            output_dir=tmp_path, extract_probe_metadata=False,
        )
        preprocess_whitfield(cfg)
        exp3 = load_preprocessed(tmp_path, "Exp3")
        exp4 = load_preprocessed(tmp_path, "Exp4")

        # Top-100 highest-variance positions in x_train
        var = exp3["x"].var(axis=0)
        top100 = np.argsort(var)[::-1][:100].tolist()

        # Most or all should align (Exp4 keeps 44051/44160; Exp3 keeps 43981)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            tr, te = align_to_train_features(
                exp3["kept_probe_indices"],
                exp4["kept_probe_indices"],
                top100,
            )
        # Sanity: at least 80% align (in practice >99%)
        assert len(tr) >= 80
        assert len(tr) == len(te)
        # Verify probe IDs match position-wise
        for t_idx, e_idx in zip(tr, te):
            assert (
                exp3["kept_probe_indices"][t_idx]
                == exp4["kept_probe_indices"][e_idx]
            )


# ──────────────────────────────────────────────────────────────────
# Skip notice
# ──────────────────────────────────────────────────────────────────

def test_data_availability_notice():
    """Emit a clear note if Whitfield data is missing (CI environments)."""
    if not WHITFIELD_AVAILABLE:
        pytest.skip(
            f"Whitfield raw data not available at {DEFAULT_RAW_DIR}. "
            f"Set raw_dir explicitly in WhitfieldPreprocessConfig or "
            f"populate the default location with the GSE3497 series matrix."
        )
    # If data IS available, this test is a no-op (just confirms the gate).
    assert WHITFIELD_AVAILABLE
