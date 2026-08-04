"""Whitfield 2002 (GEO GSE3497) preprocessing for CG-Mamba cell cycle.

Converts the raw GEO Series matrix (GPL3001 cDNA microarray) into model-ready
expression matrices for the two cell-synchronization protocols used in the
cell cycle experiments:

    Exp3 (double thymidine, "thy"):       48 timepoints, training default
    Exp4 (thymidine-nocodazole, "thy_noc"): 17 timepoints, cross-experiment test

Pipeline:
    1. Locate `GSE3497-{platform}_series_matrix.txt.gz` in `raw_dir`.
    2. Parse `!Sample_title` headers → assign each GSM column to (experiment,
       timepoint_hours) via robust regex.
    3. Parse `!series_matrix_table_begin` ... `_end` block → [probes × samples]
       float matrix. 'null' literals become NaN.
    4. Drop probes whose NaN fraction exceeds `drop_nan_threshold`.
    5. Linearly interpolate remaining NaN along the time axis per-experiment
       (with edge-extension fallback for boundary NaNs).
    6. z-score normalize each surviving probe across timepoints (per experiment,
       independent — Exp3 statistics never leak into Exp4 normalization).
    7. Optionally extract probe metadata (IMAGE clone, GB accessions) from the
       SOFT family file. This is best-effort: if the SOFT file is missing,
       probe IDs are returned as a plain integer list.
    8. Persist outputs as `.npz` + `.json` in `output_dir`.

Tier scope (v2.2.0 §17.8):
    Tier 1+2 implemented here. Gene-symbol mapping (Tier 3, requires NCBI
    eutils or external lookup table) is deferred — variance/random/latent
    emission modes work without symbols; the 'marker' emission ablation
    awaits the symbol mapping pipeline.

Reference:
    Whitfield ML et al. (2002) Mol. Biol. Cell 13(6):1977-2000.
    GEO GSE3497, GPL3001 platform (24K cDNA, late-2001 Stanford print run).
"""
from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

# Default raw data location.
# Priority: $WHITFIELD_RAW_DIR env var > sibling CM_Mamba mirror > final fallback.
# The CM_Mamba mirror is the canonical location per
# project_q1_crossdomain_benchmarks (Q1 cross-domain benchmark set), assumed
# to be a sibling directory of CG_Mamba in the JeongHa workspace.
_ENV_RAW_DIR = os.environ.get("WHITFIELD_RAW_DIR", "").strip()
_SIBLING_FALLBACK = (
    Path(__file__).resolve().parents[2]      # CG_Mamba/
    .parent                                   # JeongHa/
    / "CM_Mamba" / "data" / "raw" / "whitfield2002"
)
DEFAULT_RAW_DIR = Path(_ENV_RAW_DIR) if _ENV_RAW_DIR else _SIBLING_FALLBACK
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "whitfield2002"


@dataclass
class WhitfieldPreprocessConfig:
    """Configuration for the Whitfield 2002 preprocessing pipeline.

    Args:
        raw_dir:            directory containing GSE3497-{platform}_series_matrix.txt.gz
                            and (optionally) GSE3497_family.soft.gz.
        output_dir:         where to write x_exp3.npz / x_exp4.npz / *.json.
        platform:           GEO platform accession (default GPL3001 — the 24K
                            cDNA array used for thy & thy-noc time courses).
        drop_nan_threshold: fraction of NaN per probe above which the probe is
                            dropped before any imputation (default 0.5).
        fill_nan_strategy:  imputation method for remaining NaN values.
                            'interpolate' (default) = linear-along-time +
                                                      edge-extend boundaries.
                            'zero'                  = replace with 0.0 (post-
                                                      normalization safe).
                            'median'                = replace with per-probe
                                                      median across timepoints.
        normalize:          z-score each surviving probe across timepoints
                            (per experiment, no cross-experiment leakage).
        extract_probe_metadata: read the GPL3001 platform table from the SOFT
                            family file (slow, large file decompression).
                            False yields a stub metadata file with probe IDs only.
    """
    raw_dir: Path = DEFAULT_RAW_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    platform: str = "GPL3001"
    drop_nan_threshold: float = 0.5
    fill_nan_strategy: str = "interpolate"
    normalize: bool = True
    extract_probe_metadata: bool = True

    def __post_init__(self) -> None:
        # Coerce path-likes
        self.raw_dir = Path(self.raw_dir)
        self.output_dir = Path(self.output_dir)
        if not (0.0 <= self.drop_nan_threshold <= 1.0):
            raise ValueError(
                f"drop_nan_threshold must be in [0, 1], got {self.drop_nan_threshold}"
            )
        if self.fill_nan_strategy not in ("interpolate", "zero", "median"):
            raise ValueError(
                f"fill_nan_strategy must be one of "
                f"('interpolate', 'zero', 'median'), got {self.fill_nan_strategy!r}"
            )


# ──────────────────────────────────────────────────────────────────
# Sample title parsing
# ──────────────────────────────────────────────────────────────────

# Tolerant patterns (case-insensitive). Real titles seen:
#   "26h Thy-Noc, Exp4"             → (Exp4, 26.0)
#   "0 hr Hela Double Thymidine, exp3"  → (Exp3, 0.0)
_EXP_PATTERN = re.compile(r"[Ee]xp\s*(\d+)")
_TIME_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*h(?:r|ours?)?\b", re.IGNORECASE)


def parse_sample_title(title: str) -> tuple[str | None, float | None]:
    """Extract (experiment_id, timepoint_hours) from a sample title string.

    The series matrix uses inconsistent free-text titles. This parser is
    deliberately permissive: it succeeds whenever an "expN" token AND a
    "<number>h(r)?" token are both present, regardless of casing or order.

    Args:
        title: free-text Sample_title field, possibly wrapped in extra quotes.

    Returns:
        (experiment_id, timepoint_hours) where experiment_id is "ExpN" (canonical
        case) and timepoint_hours is a float. Returns (None, None) on
        unparseable titles, with the caller responsible for the error policy.
    """
    s = title.strip().strip('"').strip("'").strip()
    exp_match = _EXP_PATTERN.search(s)
    time_match = _TIME_PATTERN.search(s)
    exp = f"Exp{exp_match.group(1)}" if exp_match else None
    tp = float(time_match.group(1)) if time_match else None
    return exp, tp


# ──────────────────────────────────────────────────────────────────
# Series matrix parser
# ──────────────────────────────────────────────────────────────────

def load_series_matrix(
    path: Path,
) -> tuple[np.ndarray, list[str], list[str], dict[str, str]]:
    """Load a GEO Series matrix file (.txt.gz).

    Parses three sections:
      - Header lines beginning with '!' (metadata).
      - '!Sample_title' line (one per GSM column, tab-delimited).
      - '!series_matrix_table_begin' ... '_end' block (probe × sample expression).

    Args:
        path: path to a `GSE*_series_matrix.txt.gz` file.

    Returns:
        expr:        [n_probes, n_samples] float32 array, 'null' → NaN.
        gsm_ids:     [n_samples] GSM identifiers (table header row).
        probe_ids:   [n_probes] probe IDs (table ID_REF column, as strings).
        sample_titles: dict mapping GSM ID → free-text Sample_title.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the expected `_table_begin` marker is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Series matrix not found: {path}")

    sample_titles_raw: list[str] = []
    gsm_ids: list[str] = []
    probe_ids: list[str] = []
    data_rows: list[list[float]] = []

    in_table = False
    seen_header = False

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue

            if line.startswith("!Sample_title"):
                # Format: !Sample_title\t"title1"\t"title2"\t...
                sample_titles_raw = line.split("\t")[1:]
                continue

            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                seen_header = False
                continue

            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue

            if not in_table:
                continue  # other ! metadata — currently ignored

            tokens = line.split("\t")
            if not seen_header:
                # First in-table line is the column header: "ID_REF"\tGSM...
                gsm_ids = [t.strip().strip('"') for t in tokens[1:]]
                seen_header = True
                continue

            # Data row
            probe_ids.append(tokens[0].strip().strip('"'))
            row = []
            for tok in tokens[1:]:
                tok = tok.strip().strip('"')
                if tok == "" or tok.lower() == "null":
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(tok))
                    except ValueError:
                        row.append(np.nan)
            data_rows.append(row)

    if not data_rows:
        raise ValueError(
            f"No expression rows parsed from {path}. "
            f"Expected '!series_matrix_table_begin' block."
        )

    expr = np.asarray(data_rows, dtype=np.float32)
    # Align sample titles to GSM IDs (positional, per GEO convention).
    sample_titles: dict[str, str] = {}
    for gsm, raw in zip(gsm_ids, sample_titles_raw):
        sample_titles[gsm] = raw.strip().strip('"').strip()

    return expr, gsm_ids, probe_ids, sample_titles


# ──────────────────────────────────────────────────────────────────
# Probe metadata (Tier 2 — IMAGE clone, GB accession)
# ──────────────────────────────────────────────────────────────────

def extract_probe_metadata(
    soft_path: Path,
    platform: str = "GPL3001",
) -> dict[str, dict[str, Any]]:
    """Extract per-probe annotation from the SOFT family record.

    Scans for the `^PLATFORM = {platform}` block and the subsequent
    `!platform_table_begin` ... `_end` table.

    Args:
        soft_path: path to `GSE3497_family.soft.gz`.
        platform:  GPL accession to extract (default GPL3001).

    Returns:
        Dict probe_id (str) → {image_clone, gb_accessions, polymer, type}.
        Empty dict if the SOFT file is missing or the platform is not found.
    """
    if not soft_path.exists():
        return {}

    out: dict[str, dict[str, Any]] = {}
    target_marker = f"^PLATFORM = {platform}"
    in_target_platform = False
    in_table = False
    header_cols: list[str] = []

    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^PLATFORM"):
                in_target_platform = line.strip() == target_marker
                in_table = False
                continue
            if not in_target_platform:
                continue
            if line.startswith("!platform_table_begin"):
                in_table = True
                header_cols = []
                continue
            if line.startswith("!platform_table_end"):
                in_table = False
                in_target_platform = False  # next platform will reset
                continue
            if not in_table:
                continue
            tokens = line.split("\t")
            if not header_cols:
                header_cols = [t.strip() for t in tokens]
                continue
            row = dict(zip(header_cols, [t.strip() for t in tokens]))
            probe_id = row.get("ID", "")
            if not probe_id:
                continue
            gb_list_raw = row.get("GB_LIST", "")
            gb_accessions = (
                [g.strip() for g in gb_list_raw.split(",") if g.strip()]
                if gb_list_raw else []
            )
            out[probe_id] = {
                "image_clone": row.get("SPOT_ID", ""),
                "gb_accessions": gb_accessions,
                "polymer": row.get("POLYMER", ""),
                "type": row.get("TYPE", ""),
            }
    return out


# ──────────────────────────────────────────────────────────────────
# Experiment splitting
# ──────────────────────────────────────────────────────────────────

@dataclass
class ExperimentSplit:
    """One experiment's samples sorted by timepoint."""
    exp_id: str                          # e.g., "Exp3"
    gsm_ids: list[str]                   # GSM IDs in time order
    timepoints_hours: list[float]        # sorted ascending
    column_indices: list[int]            # original column indices into expr
    sync_method: str                     # "thy" or "thy_noc"


# Canonical mapping from Whitfield experiment label → CellCycleHMM sync_method.
EXP_TO_SYNC: dict[str, str] = {
    "Exp1": "thy",       # earlier thymidine-only time course
    "Exp2": "thy",       # second thymidine-only time course
    "Exp3": "thy",       # double thymidine, 48 timepoints (PRIMARY training set)
    "Exp4": "thy_noc",   # thymidine + nocodazole, 17 timepoints (cross-domain test)
    "Exp5": "thy_noc",   # mitotic shake-off / extended thy-noc (auxiliary)
}


def split_by_experiment(
    gsm_ids: list[str],
    sample_titles: dict[str, str],
) -> dict[str, ExperimentSplit]:
    """Group samples by experiment and sort each group by timepoint.

    Args:
        gsm_ids:        column order of the expression matrix.
        sample_titles:  GSM → free-text title (from load_series_matrix).

    Returns:
        Dict exp_id → ExperimentSplit. Samples with unparseable titles are
        silently dropped (with a warning logged via the returned summary).
    """
    by_exp: dict[str, list[tuple[int, str, float]]] = {}
    for col_idx, gsm in enumerate(gsm_ids):
        title = sample_titles.get(gsm, "")
        exp, tp = parse_sample_title(title)
        if exp is None or tp is None:
            continue
        by_exp.setdefault(exp, []).append((col_idx, gsm, tp))

    out: dict[str, ExperimentSplit] = {}
    for exp_id, entries in by_exp.items():
        entries.sort(key=lambda e: e[2])
        out[exp_id] = ExperimentSplit(
            exp_id=exp_id,
            gsm_ids=[e[1] for e in entries],
            timepoints_hours=[e[2] for e in entries],
            column_indices=[e[0] for e in entries],
            sync_method=EXP_TO_SYNC.get(exp_id, "thy"),
        )
    return out


# ──────────────────────────────────────────────────────────────────
# Normalization (NaN handling + z-score)
# ──────────────────────────────────────────────────────────────────

def _interpolate_nan_along_time(x: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values along axis 0 (time), per column.

    Boundary NaNs are filled by edge-extension (nearest non-NaN value).
    """
    out = x.astype(np.float32, copy=True)
    T, V = out.shape
    for v in range(V):
        col = out[:, v]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        if nan_mask.all():
            # Entire column is NaN — should have been dropped earlier;
            # zero-fill as a safe fallback.
            out[:, v] = 0.0
            continue
        valid_idx = np.flatnonzero(~nan_mask)
        nan_idx = np.flatnonzero(nan_mask)
        col[nan_idx] = np.interp(
            nan_idx, valid_idx, col[valid_idx],
            left=col[valid_idx[0]], right=col[valid_idx[-1]],
        )
        out[:, v] = col
    return out


def normalize_expression(
    expr: np.ndarray,
    drop_nan_threshold: float = 0.5,
    fill_strategy: str = "interpolate",
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop bad probes, fill remaining NaN, z-score normalize per probe.

    Args:
        expr:               [T, n_probes] expression matrix (NaN-tolerant).
        drop_nan_threshold: drop columns whose NaN fraction exceeds this.
        fill_strategy:      'interpolate' (linear-along-time + edge-extend) |
                            'zero' (replace with 0) | 'median' (per-probe).
        normalize:          z-score (mean 0, std 1) per surviving probe.

    Returns:
        x_clean:        [T, n_kept] normalized matrix, no NaN.
        kept_columns:   [n_kept] indices of probes retained from `expr`.
    """
    assert expr.ndim == 2, f"Expected [T, V], got shape {expr.shape}"
    T, V = expr.shape
    nan_frac = np.isnan(expr).mean(axis=0)              # [V]
    keep = nan_frac <= drop_nan_threshold
    kept_columns = np.flatnonzero(keep)
    x = expr[:, kept_columns].astype(np.float32, copy=True)

    if fill_strategy == "interpolate":
        x = _interpolate_nan_along_time(x)
    elif fill_strategy == "zero":
        x = np.nan_to_num(x, nan=0.0)
    elif fill_strategy == "median":
        for v in range(x.shape[1]):
            col = x[:, v]
            mask = np.isnan(col)
            if mask.any() and not mask.all():
                col[mask] = float(np.nanmedian(col))
            elif mask.all():
                col[:] = 0.0
            x[:, v] = col
    else:
        raise ValueError(f"Unknown fill_strategy {fill_strategy!r}")

    # Final safety: any residual NaN → 0 (should not occur).
    x = np.nan_to_num(x, nan=0.0)

    if normalize:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)            # avoid /0 for constant probes
        x = (x - mean) / std
        x = x.astype(np.float32, copy=False)

    return x, kept_columns


# ──────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────

def preprocess_whitfield(
    config: WhitfieldPreprocessConfig | None = None,
) -> dict[str, Any]:
    """Run the full pipeline and write outputs to `config.output_dir`.

    Args:
        config: pipeline configuration. None = use all defaults.

    Returns:
        Summary dict with shapes, paths, and per-experiment metadata.
    """
    if config is None:
        config = WhitfieldPreprocessConfig()

    series_path = config.raw_dir / f"GSE3497-{config.platform}_series_matrix.txt.gz"
    soft_path = config.raw_dir / "GSE3497_family.soft.gz"

    # 1) Parse series matrix
    expr_all, gsm_ids, probe_ids, sample_titles = load_series_matrix(series_path)
    n_probes, n_samples = expr_all.shape

    # 2) Split by experiment
    splits = split_by_experiment(gsm_ids, sample_titles)

    # 3) Per-experiment normalization (Exp3 / Exp4 do NOT leak into each other).
    config.output_dir.mkdir(parents=True, exist_ok=True)
    per_exp_summary: dict[str, dict[str, Any]] = {}
    union_kept: set[int] = set()
    exp_kept_cols: dict[str, np.ndarray] = {}
    exp_x: dict[str, np.ndarray] = {}

    for exp_id, split in splits.items():
        # Slice: rows=probes (axis 1 here will be probes after transpose; but
        # expr_all is [n_probes, n_samples], so x_exp = expr[:, columns].T → [T, V_probes]
        x_exp = expr_all[:, split.column_indices].T            # [T, n_probes]
        x_clean, kept_cols = normalize_expression(
            x_exp,
            drop_nan_threshold=config.drop_nan_threshold,
            fill_strategy=config.fill_nan_strategy,
            normalize=config.normalize,
        )
        exp_x[exp_id] = x_clean
        exp_kept_cols[exp_id] = kept_cols
        union_kept.update(kept_cols.tolist())
        per_exp_summary[exp_id] = {
            "n_samples": len(split.gsm_ids),
            "n_probes_kept": int(x_clean.shape[1]),
            "sync_method": split.sync_method,
            "gsm_ids": split.gsm_ids,
            "timepoints_hours": split.timepoints_hours,
            "kept_probe_indices": kept_cols.tolist(),
        }

    # 4) Persist per-experiment npz
    paths: dict[str, str] = {}
    for exp_id, x in exp_x.items():
        out_path = config.output_dir / f"x_{exp_id.lower()}.npz"
        np.savez(
            out_path,
            x=x.astype(np.float32, copy=False),
            kept_probe_indices=exp_kept_cols[exp_id].astype(np.int32),
            timepoints_hours=np.asarray(
                per_exp_summary[exp_id]["timepoints_hours"], dtype=np.float32
            ),
        )
        paths[exp_id] = str(out_path)

    # 5) Probe metadata (Tier 2, best-effort)
    probe_meta_path = config.output_dir / "probe_metadata.json"
    if config.extract_probe_metadata and soft_path.exists():
        platform_meta = extract_probe_metadata(soft_path, platform=config.platform)
    else:
        platform_meta = {}
    probe_metadata = {
        "n_probes_total": n_probes,
        "probe_ids": probe_ids,
        "annotations": platform_meta,                          # may be empty if not extracted
        "platform": config.platform,
        "note": (
            "GPL3001 cDNA microarray. SPOT_ID = IMAGE clone identifier "
            "(no direct HGNC symbol mapping; Tier 3 mapping deferred — "
            "PLAN §17.8)."
        ),
    }
    with open(probe_meta_path, "w", encoding="utf-8") as fh:
        json.dump(probe_metadata, fh, indent=2)

    # 6) Sample metadata
    sample_meta_path = config.output_dir / "sample_metadata.json"
    sample_metadata = {
        "platform": config.platform,
        "n_samples_total": n_samples,
        "experiments": {
            exp_id: {k: v for k, v in info.items() if k != "kept_probe_indices"}
            for exp_id, info in per_exp_summary.items()
        },
    }
    with open(sample_meta_path, "w", encoding="utf-8") as fh:
        json.dump(sample_metadata, fh, indent=2)

    return {
        "n_probes": n_probes,
        "n_samples": n_samples,
        "experiments": list(splits.keys()),
        "per_experiment": per_exp_summary,
        "output_paths": {
            **paths,
            "probe_metadata": str(probe_meta_path),
            "sample_metadata": str(sample_meta_path),
        },
    }


# ──────────────────────────────────────────────────────────────────
# Loader (for downstream forecaster use)
# ──────────────────────────────────────────────────────────────────

def load_preprocessed(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    exp_id: str = "Exp3",
) -> dict[str, np.ndarray]:
    """Load a previously-preprocessed experiment.

    Args:
        output_dir: directory containing x_{exp_id.lower()}.npz.
        exp_id:     'Exp3' or 'Exp4' (or any preprocessed experiment).

    Returns:
        dict with keys 'x', 'kept_probe_indices', 'timepoints_hours'.
    """
    output_dir = Path(output_dir)
    path = output_dir / f"x_{exp_id.lower()}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed {exp_id} not found at {path}. "
            f"Run preprocess_whitfield(config) first."
        )
    with np.load(path) as data:
        return {
            "x": data["x"],
            "kept_probe_indices": data["kept_probe_indices"],
            "timepoints_hours": data["timepoints_hours"],
        }


# ──────────────────────────────────────────────────────────────────
# Cross-experiment feature alignment (P1.1)
# ──────────────────────────────────────────────────────────────────

def align_to_train_features(
    train_kept_indices: np.ndarray,
    test_kept_indices: np.ndarray,
    train_local_feature_indices: list[int] | np.ndarray,
) -> tuple[list[int], list[int]]:
    """Map train-local feature indices → test-local indices via original probe IDs.

    Each experiment drops a different subset of probes during NaN filtering
    (Exp3 keeps ~43,981 of 44,160 probes, Exp4 keeps ~44,051). A feature
    selected from the train experiment (e.g., the 16 highest-variance probes
    in Exp3) must be re-located in the test experiment via the *original*
    probe ID — not the train-local position — to ensure that the same
    biological probe is being used for inference on Exp4.

    Pipeline:
        x_train [T_train, V_train]  ← columns indexed by train_kept_indices
        x_test  [T_test,  V_test ]  ← columns indexed by test_kept_indices

        train_local_feature_indices ⊆ [0, V_train)
            │
            │  position-in-x_train → original probe ID
            ▼
        original probe IDs of selected features
            │
            │  lookup in test_kept_indices
            ▼
        test_local positions, with features absent from test dropped.

    Args:
        train_kept_indices:           [V_train] indices into the original
                                       probe array surviving in the train set
                                       (= `kept_probe_indices` from x_exp3.npz).
        test_kept_indices:            [V_test]  same for the test set.
        train_local_feature_indices:  positions in x_train identifying the
                                       features to align (e.g., output of
                                       select_emission_features called on Exp3).

    Returns:
        train_aligned: list of column indices into x_train. Same length as
                       test_aligned. Some entries from
                       train_local_feature_indices may be missing if the
                       corresponding probe was dropped during Exp4 NaN
                       filtering.
        test_aligned:  matching column indices into x_test. Paired
                       element-wise with train_aligned (same probe ID at the
                       same position).

    Notes:
        - When a feature has no surviving probe in the test set, both lists
          drop that entry and a UserWarning is emitted listing the missing
          original probe IDs.
        - If ALL features are dropped, both lists are empty (caller
          responsibility to handle).

    Example:
        >>> exp3 = load_preprocessed(out_dir, "Exp3")
        >>> exp4 = load_preprocessed(out_dir, "Exp4")
        >>> # Selected 16 markers in Exp3 (positions in x_train)
        >>> _, markers_in_x_train = select_emission_features(
        ...     exp3["x"], probe_id_strings_train, config.emission)
        >>> tr, te = align_to_train_features(
        ...     exp3["kept_probe_indices"], exp4["kept_probe_indices"],
        ...     markers_in_x_train)
        >>> x_train_markers = exp3["x"][:, tr]   # [48, 16 or fewer]
        >>> x_test_markers  = exp4["x"][:, te]   # [17, 16 or fewer], aligned
    """
    train_kept = np.asarray(train_kept_indices, dtype=np.int64)
    test_kept = np.asarray(test_kept_indices, dtype=np.int64)
    feature_idx = np.asarray(train_local_feature_indices, dtype=np.int64)

    if feature_idx.size > 0:
        if feature_idx.min() < 0 or feature_idx.max() >= train_kept.size:
            raise ValueError(
                f"train_local_feature_indices must be in [0, V_train={train_kept.size}); "
                f"got min={int(feature_idx.min())}, max={int(feature_idx.max())}"
            )

    # Build test-probe → test-local lookup
    test_to_local: dict[int, int] = {int(orig): i for i, orig in enumerate(test_kept)}

    train_aligned: list[int] = []
    test_aligned: list[int] = []
    missing_probes: list[int] = []
    for local_idx in feature_idx.tolist():
        orig_probe = int(train_kept[local_idx])
        if orig_probe in test_to_local:
            train_aligned.append(int(local_idx))
            test_aligned.append(test_to_local[orig_probe])
        else:
            missing_probes.append(orig_probe)

    if missing_probes:
        import warnings
        head = missing_probes[:5]
        tail = "..." if len(missing_probes) > 5 else ""
        warnings.warn(
            f"align_to_train_features: {len(missing_probes)} of "
            f"{len(feature_idx)} features have no surviving probe in the "
            f"test experiment; dropped. First few original probe indices: "
            f"{head}{tail}",
            UserWarning,
        )

    return train_aligned, test_aligned
