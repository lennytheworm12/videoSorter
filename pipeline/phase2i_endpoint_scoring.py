"""Phase 2I Universal Dependencies/syntactic Feature Set C ablation.

Phase 2I runs exactly two new cells -- ``logistic_C`` and ``lightgbm_C`` --
on the frozen Phase 2H grouped leave-one-window-out benchmark.  Feature Set C
is the frozen Phase 2H Feature Set B plus deterministic UD/syntactic features
computed from a local, CPU-only Stanza parse of each Bronze window
(``pipeline.phase2i_syntax``).

The published Phase 2H run-1 artifact is the baseline of record: it is
hash-verified (archive SHA-256, aggregate content SHA-256, benchmark content
hash) and its frozen B metrics/predictions are reused for deltas, error
taxonomy comparison, overlap-cluster diagnostics, and derivation of the seven
universally-missed gold endpoints.  No A/B cell is retrained.

Constraints preserved from Phase 2H:
* identical grouped leave-one-window-out folds, fixed .5 threshold, seed;
* every scaler/vectorizer/categorical syntax vocabulary fitted on training
  windows only; held-out OOV audited;
* logistic config and LightGBM config are exactly the Phase 2H B configs;
* syntax features are learned features only -- never hard endpoint rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import lightgbm
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.phase2g_endpoint_recovery import BENCHMARK_CONTENT_SHA256
from pipeline.phase2g_silver import canonical_sha256
from pipeline.phase2h_endpoint_scoring import (
    CELLS,
    DENSE_A_FEATURES,
    DENSE_B_EXTRA_FEATURES,
    DROP,
    ERROR_CODES,
    ERROR_PRECEDENCE,
    GOLD_RANK_HIGH_K,
    KEEP,
    KEEP_THRESHOLD,
    LGBM_CONFIG,
    LOGISTIC_CONFIG,
    RANK_KS,
    SEED,
    SPARSE_CONFIG,
    VECTORIZER_PARAMS,
    CandidateRow,
    CellPreprocessor,
    Phase2HCVError,
    Phase2HError,
    _aggregate_top,
    _held_out_token_types,
    _metric,
    _oov_audit,
    _pooled_metrics,
    _window_metrics,
    balanced_class_weights,
    build_dataset,
    classify_all_errors,
    compute_rankings,
    error_taxonomy_counts,
    extract_dense_features,
    extract_sparse_inputs,
    load_benchmark,
    validate_benchmark_mapping,
    validate_dataset,
)
from pipeline.phase2i_syntax import (
    BOUNDARY_AMBIGUOUS,
    BOUNDARY_EXACT,
    BOUNDARY_PARTIAL,
    BOUNDARY_STATUSES,
    BOUNDARY_TOKEN_ALIGNED,
    BOUNDARY_UNALIGNED,
    DENSE_C_EXTRA_FEATURES,
    LOCKED_ASSETS_MANIFEST_SHA256,
    MAX_VALUES_PER_GROUP,
    PIPELINE_VERSION,
    STANZA_PACKAGE,
    STANZA_PROCESSORS,
    STANZA_VERSION,
    SYNTAX_GROUPS,
    UdParse,
    CandidateSyntax,
    SyntaxEncoder,
    Phase2ISyntaxError,
    compute_candidate_syntax,
    dense_c_matrix,
    feature_schema_c,
    is_sha256_hex,
    parse_definition,
    parse_window_text,
    syntax_groups_from_records,
    verify_assets_provenance,
    _load_json_strict,
    _symlink_ancestor_problems,
)


RUN_VERSION = "phase2i-syntax-features-v1"
DEFAULT_PARSER_ASSETS = (
    Path(__file__).resolve().parents[1] / "data" / "phase2i_assets"
)
DEFAULT_SOURCE_REPOSITORY = Path(__file__).resolve().parents[1]
CELLS_C = ("logistic_C", "lightgbm_C")
MODEL_NAMES_C = ("logistic", "lightgbm")
BASELINE_CELLS = ("logistic_A", "logistic_B", "lightgbm_A", "lightgbm_B")
BASELINE_B_CELLS = ("logistic_B", "lightgbm_B")
LOCKED_WINDOW_IDS = (
    "mid-push-prevents-side-collapse",
    "push-poke-wave-crash",
    "sweeper-limits-mid-play",
    "unwarded-bush-hook-risk",
    "wave-reset-after-kill",
)

PHASE2H_RUN1_ARCHIVE_SHA256 = (
    "22aaab162f6122691f577bc95746a0b7b1da9834706766b746a29737a5e46380"
)
PHASE2H_RUN1_AGGREGATE_SHA256 = (
    "3a890de5f429056bae9d9932ce7f1985d9315e20655b4239be5acee4174edee2"
)

# Frozen audit record: the seven Phase 2H gold endpoints selected DROP by all
# four baseline cells.  Derivation is always performed from the baseline
# window-table predictions; this tuple is only the historical validation lock.
UNIVERSALLY_MISSED_LOCK = (
    (
        "transcript:z5IXabhMLzQ:w0001-afc35185fd6a:mee69f08b652c63baa001",
        "enemy team cannot punish you guys going four for Gwen",
    ),
    (
        "transcript:uAdWuLPYn-0:w0001-fb3205f09aad:m3e540d6c5b309162c084",
        "the sweeper should be used around mid",
    ),
    (
        "transcript:uAdWuLPYn-0:w0001-fb3205f09aad:mb32742ed25712c6ba6f7",
        "around mid",
    ),
    (
        "transcript:uAdWuLPYn-0:w0001-8c53593b7d6d:m653eec8153b29d0679ef",
        "you guys hard lose level one",
    ),
    (
        "transcript:uAdWuLPYn-0:w0001-8c53593b7d6d:mddf88231dbb94e21a19a",
        "win level one",
    ),
    (
        "transcript:3nKrtwpZ6sQ:w0001-bc9b368f0c97:m1355fb438dfe24cd4e0f",
        "run into Tower and just die",
    ),
    (
        "transcript:3nKrtwpZ6sQ:w0001-bc9b368f0c97:mfb235e6d91b18d3cd580",
        "pull the wave up again",
    ),
)


class Phase2IError(ValueError):
    """Base error for Phase 2I contract violations."""


class Phase2IBaselineError(Phase2IError):
    """Published Phase 2H baseline verification failed."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_tar(archive: Path, destination: Path) -> None:
    """Extract the SHA-locked baseline archive with a path-traversal guard
    (defense in depth; the archive SHA is already pinned).  Only regular
    files and directories are accepted; links, devices, FIFOs, and any member
    whose resolved path escapes the destination are rejected."""
    destination = Path(destination).resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            member_path = Path(member.name)
            if not (member.isfile() or member.isdir()):
                raise Phase2IBaselineError(
                    "unsafe member type in baseline archive: "
                    f"{member.name!r}",
                )
            if (
                member.name.startswith("/")
                or ".." in member_path.parts
                or not member.name
                or "\\" in member.name
                or not (destination / member.name).resolve().is_relative_to(
                    destination,
                )
            ):
                raise Phase2IBaselineError(
                    "unsafe member path in baseline archive: "
                    f"{member.name!r}",
                )
        handle.extractall(destination)


def load_phase2h_baseline(
    archive_path: str | Path,
) -> dict[str, Any]:
    """Load and hash-verify the published Phase 2H run-1 baseline of record.

    Returns a mapping whose ``tempdir`` entry keeps the extracted artifact
    alive; callers must keep the returned mapping referenced and close it
    with :func:`close_phase2h_baseline` when done.
    """
    archive = Path(archive_path)
    if not archive.is_file():
        raise Phase2IBaselineError(
            f"Phase 2H baseline archive missing: {archive}",
        )
    actual_archive = _file_sha256(archive)
    if actual_archive != PHASE2H_RUN1_ARCHIVE_SHA256:
        raise Phase2IBaselineError(
            "Phase 2H run-1 archive SHA-256 "
            f"{actual_archive} != locked {PHASE2H_RUN1_ARCHIVE_SHA256}",
        )
    temporary = tempfile.TemporaryDirectory(prefix="phase2i-baseline-")
    extracted = Path(temporary.name)
    try:
        _extract_tar(archive, extracted)
        aggregate_path = extracted / "phase2h-endpoint-scoring.json"
        if not aggregate_path.is_file():
            raise Phase2IBaselineError(
                "Phase 2H archive has no phase2h-endpoint-scoring.json",
            )
        aggregate = json.loads(
            aggregate_path.read_text(encoding="utf-8"),
        )
        inner = {
            key: value for key, value in aggregate.items()
            if key != "content_sha256"
        }
        if aggregate.get("content_sha256") != canonical_sha256(inner):
            raise Phase2IBaselineError(
                "Phase 2H aggregate content_sha256 does not self-verify",
            )
        if aggregate["content_sha256"] != PHASE2H_RUN1_AGGREGATE_SHA256:
            raise Phase2IBaselineError(
                "Phase 2H run-1 aggregate SHA-256 "
                f"{aggregate['content_sha256']} != locked "
                f"{PHASE2H_RUN1_AGGREGATE_SHA256}",
            )
        locked_benchmark = aggregate.get("input_hashes", {}).get(
            "benchmark_content_sha256",
        )
        if locked_benchmark != BENCHMARK_CONTENT_SHA256:
            raise Phase2IBaselineError(
                "Phase 2H baseline benchmark lock "
                f"{locked_benchmark!r} != {BENCHMARK_CONTENT_SHA256}",
            )
        window_tables: dict[str, dict[str, Any]] = {}
        window_dir = extracted / "windows"
        for path in sorted(window_dir.glob("*.json")):
            window_tables[path.stem] = json.loads(
                path.read_text(encoding="utf-8"),
            )
        if set(window_tables) != {
            "mid-push-prevents-side-collapse",
            "push-poke-wave-crash",
            "sweeper-limits-mid-play",
            "unwarded-bush-hook-risk",
            "wave-reset-after-kill",
        }:
            raise Phase2IBaselineError(
                "Phase 2H baseline window set is not the locked benchmark",
            )
    except Exception:
        temporary.cleanup()
        raise
    return {
        "aggregate": aggregate,
        "window_tables": window_tables,
        "tempdir": temporary,
    }


def close_phase2h_baseline(baseline: Mapping[str, Any]) -> None:
    tempdir = baseline.get("tempdir")
    if isinstance(tempdir, tempfile.TemporaryDirectory):
        tempdir.cleanup()


def baseline_rankings_from_tables(
    baseline_window_tables: Mapping[str, Mapping[str, Any]],
    *,
    cells: Sequence[str] = BASELINE_CELLS,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Convert baseline window tables into the rankings shape used by
    Phase 2H metric/diagnostic helpers."""
    rankings: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for window_id, table in baseline_window_tables.items():
        cell_rankings: dict[str, dict[str, dict[str, Any]]] = {}
        for cell in cells:
            entries: dict[str, dict[str, Any]] = {}
            for candidate in table["candidates"]:
                prediction = candidate["predictions"].get(cell)
                if prediction is None:
                    raise Phase2IBaselineError(
                        f"baseline window {window_id} has no {cell} "
                        "prediction",
                    )
                entries[candidate["candidate_id"]] = {
                    "score": prediction["score"],
                    "rank": prediction["rank"],
                    "selected": prediction["selected"],
                }
            cell_rankings[cell] = entries
        rankings[window_id] = cell_rankings
    return rankings


def validate_cv_folds_match_baseline(
    folds: Sequence[Mapping[str, Any]],
    baseline_folds: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Exact fold-by-fold comparison against the archived Phase 2H folds.

    The Phase 2I folds must be identical to the frozen Phase 2H leave-one-
    window-out folds, including train/test window ids, candidate counts,
    positive counts, and class weights; five distinct LOO splits are not
    sufficient.
    """
    problems: list[str] = []
    if len(folds) != len(baseline_folds):
        return [
            f"fold count {len(folds)} != archived Phase 2H "
            f"{len(baseline_folds)}",
        ]
    compared_keys = (
        "fold_index",
        "train_window_ids",
        "test_window_id",
        "train_candidate_count",
        "train_positive_count",
        "train_negative_count",
        "test_candidate_count",
        "test_positive_count",
        "class_weights",
    )
    for index, (fold, baseline) in enumerate(zip(folds, baseline_folds)):
        for key in compared_keys:
            if not _strict_equal(fold.get(key), baseline.get(key)):
                problems.append(
                    f"fold {index} {key} {fold.get(key)!r} != archived "
                    f"{baseline.get(key)!r}",
                )
    return problems


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def derive_universally_missed(
    baseline_window_tables: Mapping[str, Mapping[str, Any]],
    *,
    cells: Sequence[str] = BASELINE_CELLS,
) -> list[dict[str, Any]]:
    """Robustly derive gold endpoints selected DROP by every baseline cell
    from the baseline window-table predictions (never a hardcoded list)."""
    entries: list[dict[str, Any]] = []
    for window_id in sorted(baseline_window_tables):
        table = baseline_window_tables[window_id]
        for candidate in table["candidates"]:
            if candidate["label"] != KEEP:
                continue
            predictions = candidate["predictions"]
            if not all(
                predictions.get(cell, {}).get("selected") == DROP
                for cell in cells
            ):
                continue
            entries.append({
                "window_id": window_id,
                "candidate_id": candidate["candidate_id"],
                "alias": candidate["alias"],
                "text": candidate["text"],
                "predictions": {
                    cell: {
                        "score": predictions[cell]["score"],
                        "rank": predictions[cell]["rank"],
                        "selected": predictions[cell]["selected"],
                    }
                    for cell in cells
                },
            })
    entries.sort(key=lambda entry: (entry["window_id"], entry["candidate_id"]))
    return entries


def validate_universally_missed(
    entries: Sequence[Mapping[str, Any]],
    *,
    expected: Sequence[tuple[str, str]] = UNIVERSALLY_MISSED_LOCK,
) -> list[str]:
    """Validate the derived universally-missed set against the frozen audit
    lock (exact candidate IDs and surface texts)."""
    derived = [
        (entry["candidate_id"], entry["text"]) for entry in entries
    ]
    problems: list[str] = []
    if len(derived) != len(expected):
        problems.append(
            f"derived universally-missed count {len(derived)} != "
            f"expected {len(expected)}",
        )
    expected_by_id = {candidate_id: text for candidate_id, text in expected}
    derived_by_id = {candidate_id: text for candidate_id, text in derived}
    for candidate_id in sorted(set(expected_by_id) | set(derived_by_id)):
        if candidate_id not in expected_by_id:
            problems.append(
                f"unexpected universally-missed candidate {candidate_id!r}",
            )
            continue
        if candidate_id not in derived_by_id:
            problems.append(
                f"missing universally-missed candidate {candidate_id!r}",
            )
            continue
        if derived_by_id[candidate_id] != expected_by_id[candidate_id]:
            problems.append(
                f"universally-missed candidate {candidate_id!r} text "
                "differs from the frozen audit record",
            )
    return problems


class CellPreprocessorC:
    """Feature Set C preprocessing fitted on training windows only.

    Dense C features are scaled with a training-fitted ``StandardScaler``;
    the Phase 2H sparse B block (word n-grams + boundary flags) keeps the
    exact Phase 2H vectorizer; the syntax categorical block is one-hot
    encoded by a train-fitted :class:`SyntaxEncoder`.  No held-out value ever
    contributes a column.
    """

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.vectorizer: CountVectorizer | None = None
        self.vocab_terms: list[str] = []
        self.syntax_encoder: SyntaxEncoder | None = None

    def fit(
        self,
        dense: np.ndarray,
        texts: Sequence[str],
        boundaries: Sequence[tuple[str, str, str]],
        syntax_records: Sequence[Mapping[str, Sequence[str]]],
    ) -> "CellPreprocessorC":
        self.scaler = StandardScaler().fit(dense)
        try:
            self.vectorizer = CountVectorizer(**VECTORIZER_PARAMS).fit(texts)
            self.vocab_terms = [
                term
                for term, _ in sorted(
                    self.vectorizer.vocabulary_.items(),
                    key=lambda item: item[1],
                )
            ]
        except ValueError:
            self.vectorizer = None
            self.vocab_terms = []
        self.syntax_encoder = SyntaxEncoder().fit(syntax_records)
        return self

    def transform(
        self,
        dense: np.ndarray,
        texts: Sequence[str],
        boundaries: Sequence[tuple[str, str, str]],
        syntax_records: Sequence[Mapping[str, Sequence[str]]],
    ) -> sp.csr_matrix:
        if self.scaler is None or self.syntax_encoder is None:
            raise Phase2HError(
                "CellPreprocessorC must be fitted before transform",
            )
        scaled = self.scaler.transform(dense)
        inherited_dense_count = len(DENSE_A_FEATURES) + len(
            DENSE_B_EXTRA_FEATURES,
        )
        if scaled.shape[1] != (
            inherited_dense_count + len(DENSE_C_EXTRA_FEATURES)
        ):
            raise Phase2HError(
                "Feature Set C dense matrix does not match the frozen B "
                "plus syntax schema",
            )
        if self.vectorizer is None:
            ngrams = sp.csr_matrix((len(dense), 0), dtype=np.float64)
        else:
            ngrams = self.vectorizer.transform(texts)
        boundary_flags = _boundary_sparse_c(boundaries, self.vocab_terms)
        syntax_sparse = self.syntax_encoder.transform(syntax_records)
        # Keep the complete, exact Phase 2H B block as a contiguous prefix.
        # This makes the one-lever ablation directly auditable: every column
        # after the prefix is new syntax evidence.
        inherited_b = sp.hstack(
            [
                scaled[:, :inherited_dense_count],
                ngrams,
                boundary_flags,
            ],
            format="csr",
        )
        syntax = sp.hstack(
            [scaled[:, inherited_dense_count:], syntax_sparse],
            format="csr",
        )
        return sp.hstack([inherited_b, syntax], format="csr")

    def feature_names(self, dense_names: Sequence[str]) -> list[str]:
        if self.syntax_encoder is None:
            raise Phase2HError("CellPreprocessorC must be fitted first")
        ngram_names = [f"ngram={term}" for term in self.vocab_terms]
        boundary_names = [
            f"{role}={term}"
            for role in SPARSE_CONFIG["boundary_roles"]
            for term in self.vocab_terms
        ]
        inherited_dense_count = len(DENSE_A_FEATURES) + len(
            DENSE_B_EXTRA_FEATURES,
        )
        dense_names = list(dense_names)
        if len(dense_names) != (
            inherited_dense_count + len(DENSE_C_EXTRA_FEATURES)
        ):
            raise Phase2HError(
                "Feature Set C names do not match the frozen B plus syntax "
                "schema",
            )
        return (
            dense_names[:inherited_dense_count]
            + ngram_names
            + boundary_names
            + dense_names[inherited_dense_count:]
            + self.syntax_encoder.feature_names()
        )


def _boundary_sparse_c(
    boundaries: Sequence[tuple[str, str, str]],
    vocab_terms: Sequence[str],
) -> sp.csr_matrix:
    """Replica of Phase 2H boundary-flag encoding (kept local to Phase 2I so
    Phase 2H internals are not modified)."""
    vocab_index = {term: index for index, term in enumerate(vocab_terms)}
    n = len(boundaries)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_index, (first, last, head) in enumerate(boundaries):
        for offset, token in enumerate((first, last, head)):
            column = vocab_index.get(token)
            if column is None:
                continue
            rows.append(row_index)
            cols.append(offset * len(vocab_terms) + column)
            data.append(1.0)
    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n, len(vocab_terms) * 3),
        dtype=np.float64,
    )


def make_model_c(cell: str, y_train: np.ndarray) -> Any:
    """Fixed Phase 2I models: exact Phase 2H Logistic B / LightGBM B configs."""
    if cell not in CELLS_C:
        raise Phase2HError(f"unknown Phase 2I model cell {cell!r}")
    model_name, _ = cell.split("_")
    if model_name == "logistic":
        return LogisticRegression(**LOGISTIC_CONFIG)
    weights = balanced_class_weights(y_train)
    return lightgbm.LGBMClassifier(
        **{**LGBM_CONFIG, "class_weight": weights},
    )


def build_candidate_syntax(
    dataset: Mapping[str, Any],
    parses: Mapping[str, UdParse],
) -> dict[str, list[CandidateSyntax]]:
    """Compute per-window syntax records with an exact parse/window set
    contract: every dataset window must have exactly one parse and the parse
    ``window_id`` must equal the dataset window key."""
    dataset_windows = set(dataset["windows"])
    parse_windows = set(parses)
    if parse_windows != dataset_windows:
        missing = sorted(dataset_windows - parse_windows)
        extra = sorted(parse_windows - dataset_windows)
        raise Phase2IError(
            "parse/window set mismatch: "
            + (f"missing parses {missing}; " if missing else "")
            + (f"extra parses {extra}" if extra else ""),
        )
    records: dict[str, list[CandidateSyntax]] = {}
    for window_id in sorted(dataset["windows"]):
        parse = parses[window_id]
        if parse.window_id != window_id:
            raise Phase2IError(
                f"parse.window_id {parse.window_id!r} != dataset window "
                f"{window_id!r}",
            )
        if parse.text != dataset["windows"][window_id]["bronze_text"]:
            raise Phase2IError(
                f"parse text mismatch for window {window_id}",
            )
        records[window_id] = [
            compute_candidate_syntax(parse, row)
            for row in dataset["windows"][window_id]["rows"]
        ]
    return records


def _train_score_diagnostics(
    model: Any,
    x_train: sp.csr_matrix,
    y_train: np.ndarray,
) -> dict[str, Any]:
    probabilities = model.predict_proba(x_train)[:, 1]
    predicted = probabilities >= KEEP_THRESHOLD
    return {
        "candidate_count": int(len(y_train)),
        "positive_count": int(y_train.sum()),
        "negative_count": int(len(y_train) - y_train.sum()),
        "predicted_keep_count": int(predicted.sum()),
        "average_precision": _safe_average_precision_c(
            y_train, probabilities,
        ),
        "roc_auc": _safe_roc_auc_c(y_train, probabilities),
    }


def _safe_average_precision_c(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if int(labels.sum()) == 0 or int(len(labels) - labels.sum()) == 0:
        return None
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores))


def _safe_roc_auc_c(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if int(labels.sum()) == 0 or int(len(labels) - labels.sum()) == 0:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def run_cv_c(
    dataset: Mapping[str, Any],
    parses: Mapping[str, UdParse],
    *,
    cells: Sequence[str] = CELLS_C,
    verbose: bool = False,
) -> dict[str, Any]:
    """Grouped leave-one-window-out CV for Feature Set C cells."""
    cells = tuple(cells)
    if len(set(cells)) != len(cells):
        raise Phase2HError(f"duplicate Phase 2I cells requested: {cells}")
    for cell in cells:
        if cell not in CELLS_C:
            raise Phase2HError(f"unknown Phase 2I cell {cell!r}")
    window_ids = sorted(dataset["windows"])
    if len(window_ids) < 2:
        raise Phase2HCVError(
            "leave-one-window-out CV requires at least two windows",
        )
    records_by_window = build_candidate_syntax(dataset, parses)
    records_all = [
        record
        for window_id in window_ids
        for record in records_by_window[window_id]
    ]
    dense_a_all, dense_b_extra_all = extract_dense_features(
        dataset, window_ids,
    )
    dense_b_all = np.hstack([dense_a_all, dense_b_extra_all])
    dense_c_all = np.hstack([
        dense_b_all,
        dense_c_matrix(records_all),
    ])
    texts_all, boundaries_all = extract_sparse_inputs(dataset, window_ids)
    syntax_records_all = syntax_groups_from_records(records_all)
    labels_all = np.array([
        1 if row.label == KEEP else 0
        for window_id in window_ids
        for row in dataset["windows"][window_id]["rows"]
    ], dtype=np.int64)
    counts = [
        len(dataset["windows"][window_id]["rows"])
        for window_id in window_ids
    ]
    offsets = np.cumsum([0] + counts)
    slices = np.array([
        (offsets[index], offsets[index + 1])
        for index in range(len(window_ids))
    ], dtype=np.int64)
    folds: list[dict[str, Any]] = []
    oof_scores: dict[str, dict[str, dict[str, float]]] = {
        window_id: {} for window_id in window_ids
    }
    fit_scope: dict[str, dict[int, dict[str, Any]]] = {
        cell: {} for cell in cells
    }
    fitted_models: dict[str, dict[int, tuple[Any, list[str]]]] = {
        cell: {} for cell in cells
    }
    for fold_index, (test_start, test_end) in enumerate(slices):
        test_window_id = window_ids[fold_index]
        train_window_ids = [
            window_id for index, window_id in enumerate(window_ids)
            if index != fold_index
        ]
        train_slices = np.concatenate([
            np.arange(start, stop)
            for index, (start, stop) in enumerate(slices)
            if index != fold_index
        ])
        test_slices = np.arange(test_start, test_end)
        y_train = labels_all[train_slices]
        if int(y_train.sum()) == 0 or int(len(y_train) - y_train.sum()) == 0:
            raise Phase2HCVError(
                f"fold {fold_index} (holdout {test_window_id!r}) lacks one "
                "class in training rows",
            )
        train_dense = dense_c_all[train_slices]
        test_dense = dense_c_all[test_slices]
        train_texts = [texts_all[index] for index in train_slices]
        test_texts = [texts_all[index] for index in test_slices]
        train_boundaries = [boundaries_all[index] for index in train_slices]
        test_boundaries = [boundaries_all[index] for index in test_slices]
        train_syntax = [syntax_records_all[index] for index in train_slices]
        test_syntax = [syntax_records_all[index] for index in test_slices]
        test_token_types = _held_out_token_types(test_texts)
        weights = balanced_class_weights(y_train)
        test_rows = dataset["windows"][test_window_id]["rows"]
        folds.append({
            "fold_index": fold_index,
            "train_window_ids": train_window_ids,
            "test_window_id": test_window_id,
            "train_candidate_count": int(len(train_slices)),
            "train_positive_count": int(y_train.sum()),
            "train_negative_count": int(len(train_slices) - y_train.sum()),
            "test_candidate_count": int(len(test_slices)),
            "test_positive_count": int(labels_all[test_slices].sum()),
            "class_weights": {
                KEEP: float(weights[1]),
                DROP: float(weights[0]),
            },
        })
        for cell in cells:
            if verbose:
                print(
                    f"[phase2i] fold {fold_index} {cell}: "
                    f"train={len(train_slices)} test={len(test_slices)}",
                    file=sys.stderr,
                )
            preprocessor = CellPreprocessorC()
            preprocessor.fit(
                train_dense, train_texts, train_boundaries, train_syntax,
            )
            x_train = preprocessor.transform(
                train_dense, train_texts, train_boundaries, train_syntax,
            )
            x_test = preprocessor.transform(
                test_dense, test_texts, test_boundaries, test_syntax,
            )
            model = make_model_c(cell, y_train)
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            for index, row in enumerate(test_rows):
                oof_scores[test_window_id].setdefault(row.candidate_id, {})[
                    cell
                ] = float(probabilities[index])
            dense_names = (
                list(DENSE_A_FEATURES)
                + list(DENSE_B_EXTRA_FEATURES)
                + list(DENSE_C_EXTRA_FEATURES)
            )
            names = preprocessor.feature_names(dense_names)
            syntax_encoder = preprocessor.syntax_encoder
            if syntax_encoder is None:
                raise Phase2HError("syntax encoder missing after fit")
            fit_scope[cell][fold_index] = {
                "fold_index": fold_index,
                "train_window_ids": train_window_ids,
                "test_window_id": test_window_id,
                "fit_scope": "training windows only",
                "train_candidate_count": int(len(y_train)),
                "train_positive_count": int(y_train.sum()),
                "train_negative_count": int(len(y_train) - y_train.sum()),
                "class_weights": {
                    KEEP: float(weights[1]),
                    DROP: float(weights[0]),
                },
                "scaler": {
                    "fit_on": "training_rows_only",
                    "feature_count": int(len(preprocessor.scaler.mean_)),
                    "mean_sha256": canonical_sha256(
                        [float(value) for value in preprocessor.scaler.mean_],
                    ),
                    "scale_sha256": canonical_sha256(
                        [float(value) for value in preprocessor.scaler.scale_],
                    ),
                },
                "vectorizer": {
                    "fit_on": "training_rows_only",
                    "params": dict(VECTORIZER_PARAMS),
                    "vocabulary_size": len(preprocessor.vocab_terms),
                    "vocabulary_sha256": canonical_sha256(
                        list(preprocessor.vocab_terms),
                    ),
                    **_oov_audit(
                        test_token_types, preprocessor.vocab_terms,
                    ),
                },
                "syntax_encoder": {
                    "fit_on": "training_rows_only",
                    "params": {
                        "max_values_per_group": (
                            syntax_encoder.max_values_per_group
                        ),
                    },
                    "vocabulary_sizes": {
                        group: len(syntax_encoder.vocabulary[group])
                        for group in SYNTAX_GROUPS
                    },
                    "vocabulary_sha256": syntax_encoder.vocabulary_sha256(),
                    **syntax_encoder.oov_audit(test_syntax),
                },
                "model_config": _model_config_snapshot_c(model),
                "feature_names_sha256": canonical_sha256(names),
                "train_diagnostics": _train_score_diagnostics(
                    model, x_train, y_train,
                ),
            }
            fitted_models[cell][fold_index] = (model, names)
    return {
        "folds": folds,
        "fit_scope": fit_scope,
        "oof_scores": oof_scores,
        "fitted_models": fitted_models,
        "candidate_syntax": records_by_window,
    }


def _model_config_snapshot_c(model: Any) -> dict[str, Any]:
    if isinstance(model, LogisticRegression):
        return {"family": "logistic", "params": dict(LOGISTIC_CONFIG)}
    params = model.get_params()
    class_weight = params.get("class_weight")
    if not isinstance(class_weight, dict):
        raise Phase2IError("LightGBM class_weight missing from fitted model")
    return {
        "family": "lightgbm",
        "params": {
            **dict(LGBM_CONFIG),
            "class_weight": {
                KEEP: float(class_weight[1]),
                DROP: float(class_weight[0]),
            },
        },
    }


def compute_cell_metrics_c(
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    cell: str,
) -> dict[str, Any]:
    """Candidate-level pooled OOF metrics plus per-fold metrics for a C cell."""
    if cell not in CELLS_C:
        raise Phase2HError(f"unknown Phase 2I cell {cell!r}")
    window_ids = sorted(dataset["windows"])
    per_fold: dict[str, dict[str, Any]] = {}
    labels: list[int] = []
    scores: list[float] = []
    for window_id in window_ids:
        rows = dataset["windows"][window_id]["rows"]
        per_fold[window_id] = _window_metrics(
            window_id, rows, rankings, cell,
        )
        for row in rows:
            entry = rankings[window_id][cell][row.candidate_id]
            labels.append(1 if row.label == KEEP else 0)
            scores.append(entry["score"])
    return _pooled_metrics(labels, scores, per_fold, window_ids)


def build_deltas(
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    c_metrics: Mapping[str, Mapping[str, Any]],
    *,
    cells: Sequence[str] = CELLS_C,
) -> dict[str, dict[str, Any]]:
    """Explicit C-vs-frozen-B metric deltas per B/C cell pair."""
    deltas: dict[str, dict[str, Any]] = {}
    pairs = {
        "logistic_C": "logistic_B",
        "lightgbm_C": "lightgbm_B",
    }
    for c_cell in cells:
        b_cell = pairs[c_cell]
        baseline = baseline_metrics[b_cell]
        current = c_metrics[c_cell]
        absolute: dict[str, Any] = {}
        delta_values: dict[str, Any] = {}
        for key in (
            "precision", "recall", "f1", "average_precision", "roc_auc",
        ):
            baseline_value = baseline[key].get("value", baseline[key].get("rate"))
            current_value = current[key].get("value", current[key].get("rate"))
            absolute[key] = {
                "baseline": baseline_value,
                "phase2i": current_value,
            }
            delta_values[key] = _numeric_delta(
                baseline_value, current_value,
            )
        absolute["selected"] = {
            "baseline": baseline["selected"],
            "phase2i": current["selected"],
        }
        delta_values["selected"] = _numeric_delta(
            baseline["selected"], current["selected"],
        )
        recall_at_k: dict[str, Any] = {}
        precision_at_k: dict[str, Any] = {}
        for k in RANK_KS:
            baseline_rate = baseline["recall_at_k"][str(k)]["rate"]
            current_rate = current["recall_at_k"][str(k)]["rate"]
            recall_at_k[str(k)] = {
                "baseline": baseline_rate,
                "phase2i": current_rate,
                "delta": _numeric_delta(baseline_rate, current_rate),
            }
            baseline_precision = baseline["precision_at_k"][str(k)]["rate"]
            current_precision = current["precision_at_k"][str(k)]["rate"]
            precision_at_k[str(k)] = {
                "baseline": baseline_precision,
                "phase2i": current_precision,
                "delta": _numeric_delta(
                    baseline_precision, current_precision,
                ),
            }
        absolute["recall_at_k"] = {
            str(k): {
                "baseline": baseline["recall_at_k"][str(k)]["rate"],
                "phase2i": current["recall_at_k"][str(k)]["rate"],
            }
            for k in RANK_KS
        }
        delta_values["recall_at_k"] = recall_at_k
        absolute["precision_at_k"] = {
            str(k): {
                "baseline": baseline["precision_at_k"][str(k)]["rate"],
                "phase2i": current["precision_at_k"][str(k)]["rate"],
            }
            for k in RANK_KS
        }
        delta_values["precision_at_k"] = precision_at_k
        absolute["gold_rank"] = {
            "baseline_median": baseline["gold_rank"]["median"],
            "phase2i_median": current["gold_rank"]["median"],
            "baseline_mean": baseline["gold_rank"]["mean"],
            "phase2i_mean": current["gold_rank"]["mean"],
        }
        delta_values["gold_rank"] = {
            "delta_median": _numeric_delta(
                baseline["gold_rank"]["median"],
                current["gold_rank"]["median"],
            ),
            "delta_mean": _numeric_delta(
                baseline["gold_rank"]["mean"],
                current["gold_rank"]["mean"],
            ),
        }
        deltas[c_cell] = {
            "baseline_cell": b_cell,
            "absolute": absolute,
            "delta": delta_values,
        }
    return deltas


def _numeric_delta(baseline: Any, current: Any) -> float | None:
    if baseline is None or current is None:
        return None
    return float(current) - float(baseline)


_EXACT_GOLD = "EXACT_GOLD"
_CONTAINED_SHORTER = "CONTAINED_SHORTER"
_CONTAINING_LONGER = "CONTAINING_LONGER"
_OTHER_OVERLAPPING = "OTHER_OVERLAPPING"
_OVERLAP_CLASSES = (
    _EXACT_GOLD,
    _CONTAINED_SHORTER,
    _CONTAINING_LONGER,
    _OTHER_OVERLAPPING,
)


def overlap_cluster_syntax_diagnostics(
    dataset: Mapping[str, Any],
    rankings_c: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    rankings_b: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    records_by_window: Mapping[str, Sequence[CandidateSyntax]],
    *,
    cells: Sequence[str] = CELLS_C,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-gold-positive overlap-cluster diagnostics.

    Every overlapping cluster member is classified as EXACT_GOLD,
    CONTAINED_SHORTER, CONTAINING_LONGER, or OTHER_OVERLAPPING with syntax
    attributes and B/C ranks attached; whether containing/contained
    distractors outrank the exact gold is reported per C cell.  No NMS or
    pruning is performed; this is diagnostic only.
    """
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for window_id in sorted(dataset["windows"]):
        window = dataset["windows"][window_id]
        rows = window["rows"]
        records = {
            record.candidate_id: record
            for record in records_by_window[window_id]
        }
        rank_b_by_cell = {
            cell: {
                candidate_id: entry["rank"]
                for candidate_id, entry in rankings_b[window_id][cell].items()
            }
            for cell in BASELINE_B_CELLS
        }
        per_window: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not row.is_gold_positive:
                continue
            cluster = [
                other for other in rows
                if other.start < row.end and other.end > row.start
            ]
            record = records[row.candidate_id]
            gold_rank_c = {
                cell: rankings_c[window_id][cell][row.candidate_id]["rank"]
                for cell in cells
            }
            gold_rank_b = {
                cell: rank_b_by_cell[cell.replace("_C", "_B")][
                    row.candidate_id
                ]
                for cell in cells
            }
            members: list[dict[str, Any]] = []
            classification_counts = {
                cls: 0 for cls in _OVERLAP_CLASSES
            }
            containing_outranks = {cell: False for cell in cells}
            contained_outranks = {cell: False for cell in cells}
            for other in cluster:
                if other.candidate_id == row.candidate_id:
                    relation_class = _EXACT_GOLD
                elif (
                    other.start <= row.start
                    and other.end >= row.end
                ):
                    relation_class = _CONTAINING_LONGER
                elif (
                    other.start >= row.start
                    and other.end <= row.end
                ):
                    relation_class = _CONTAINED_SHORTER
                else:
                    relation_class = _OTHER_OVERLAPPING
                classification_counts[relation_class] += 1
                other_record = records[other.candidate_id]
                outranks_exact = {
                    cell: (
                        rankings_c[window_id][cell][other.candidate_id][
                            "rank"
                        ]
                        < gold_rank_c[cell]
                    )
                    for cell in cells
                }
                members.append({
                    "candidate_id": other.candidate_id,
                    "text": other.text,
                    "relation_class": relation_class,
                    "syntax": _syntax_summary(other_record),
                    "ranks": {
                        cell: {
                            "phase2i": (
                                rankings_c[window_id][cell][
                                    other.candidate_id
                                ]["rank"]
                            ),
                            "baseline": rank_b_by_cell[
                                cell.replace("_C", "_B")
                            ][other.candidate_id],
                        }
                        for cell in cells
                    },
                    "outranks_exact_gold": outranks_exact,
                })
                for cell in cells:
                    if (
                        relation_class == _CONTAINING_LONGER
                        and outranks_exact[cell]
                    ):
                        containing_outranks[cell] = True
                    if (
                        relation_class == _CONTAINED_SHORTER
                        and outranks_exact[cell]
                    ):
                        contained_outranks[cell] = True
            cluster_rank_c = {
                cell: 1 + sum(
                    1 for other in cluster
                    if rankings_c[window_id][cell][other.candidate_id][
                        "rank"
                    ] < gold_rank_c[cell]
                )
                for cell in cells
            }
            cluster_rank_b = {
                cell: 1 + sum(
                    1 for other in cluster
                    if rank_b_by_cell[cell.replace("_C", "_B")][
                        other.candidate_id
                    ] < gold_rank_b[cell]
                )
                for cell in cells
            }
            per_cell: dict[str, dict[str, Any]] = {}
            for cell in cells:
                per_cell[cell] = {
                    "gold_rank_baseline": gold_rank_b[cell],
                    "gold_rank_phase2i": gold_rank_c[cell],
                    "cluster_rank_baseline": cluster_rank_b[cell],
                    "cluster_rank_phase2i": cluster_rank_c[cell],
                    "cluster_rank_delta": (
                        cluster_rank_c[cell] - cluster_rank_b[cell]
                    ),
                    "containing_distractor_outranks": (
                        containing_outranks[cell]
                    ),
                    "contained_distractor_outranks": (
                        contained_outranks[cell]
                    ),
                    "outranking_count": sum(
                        1 for member in members
                        if member["outranks_exact_gold"][cell]
                    ),
                    "outranking_boundary_status_counts": _count_boundary(
                        [
                            records[member["candidate_id"]]
                            for member in members
                            if member["outranks_exact_gold"][cell]
                        ],
                    ),
                    "outranking_contains_root_count": sum(
                        1 for member in members
                        if member["outranks_exact_gold"][cell]
                        and _record_contains_root(
                            records[member["candidate_id"]],
                        )
                    ),
                }
            per_window[row.candidate_id] = {
                "window_id": window_id,
                "candidate_id": row.candidate_id,
                "text": row.text,
                "syntax": _syntax_summary(record),
                "cluster_size": len(cluster),
                "members": sorted(
                    members, key=lambda member: member["candidate_id"],
                ),
                "classification_counts": classification_counts,
                "cells": per_cell,
            }
        output[window_id] = per_window
    return output


def _count_boundary(records: Sequence[CandidateSyntax]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.boundary_status] = (
            counts.get(record.boundary_status, 0) + 1
        )
    return counts


def _syntax_summary(record: CandidateSyntax) -> dict[str, Any]:
    head = record.head_word
    root = record.root_word
    return {
        "boundary_status": record.boundary_status,
        "multi_token": record.multi_token,
        "spans_multiple_sentences": record.spans_multiple_sentences,
        "ambiguity": list(record.ambiguity),
        "contains_root": _record_contains_root(record),
        "head": {
            "word_id": record.candidate_head_id,
            "lemma": head.lemma if head else None,
            "upos": head.upos if head else None,
            "xpos": head.xpos if head else None,
            "deprel": head.deprel if head else None,
            "dependent_count": record.head_dependent_count,
            "child_deprels": list(record.head_child_deprels),
            "relation_context": list(record.head_relation_context),
        } if head else None,
        "root": {
            "word_id": record.candidate_root_id,
            "lemma": root.lemma if root else None,
            "upos": root.upos if root else None,
        } if root else None,
        "root_count": len(record.root_ids),
        "multiple_roots": record.multiple_roots,
        "clause_count": len(record.clause_root_ids),
        "multiple_clause_roots": len(record.clause_root_ids) > 1,
        "finite_verb_count": len(record.finite_verb_ids),
        "predicate_count": len(record.syntactic_predicate_ids),
        "aux_count": len(record.aux_ids),
        "modal_count": len(record.modal_ids),
        "neg_count": len(record.neg_ids),
        "mark_count": len(record.mark_ids),
        "crossing_arcs": record.crossing_arc_count,
        "external_governors": len(record.external_governor_ids),
        "external_governor_exists": bool(record.external_governor_ids),
        "external_governor_lemmas": list(record.external_governor_lemmas),
        "external_governor_uposes": list(record.external_governor_uposes),
        "external_governor_deprels": list(record.external_governor_deprels),
        "subtree_size": record.subtree_size,
        "subtree_intersection_count": record.subtree_intersection_count,
        "subtree_word_fraction": (
            record.subtree_intersection_count / len(record.word_ids)
            if record.word_ids else 0.0
        ),
        "subtree_fraction": (
            record.subtree_intersection_count / record.subtree_size
            if record.subtree_size else 0.0
        ),
        "subtree_exact": record.subtree_exact,
        "span_connected": record.span_connected,
        "relations": list(record.relations),
        "internal_rel_context": list(record.internal_relation_context),
        "crossing_incoming_rel_context": list(
            record.crossing_incoming_relation_context,
        ),
        "crossing_outgoing_rel_context": list(
            record.crossing_outgoing_relation_context,
        ),
        "predicate_argument_context": list(
            record.predicate_argument_context,
        ),
        "predicate_internal_argument": record.predicate_internal_argument,
        "predicate_external_argument": record.predicate_external_argument,
        "predicate_internal_subject": record.predicate_internal_subject,
        "predicate_external_subject": record.predicate_external_subject,
        "predicate_internal_object": record.predicate_internal_object,
        "predicate_external_object": record.predicate_external_object,
        "predicate_internal_oblique": record.predicate_internal_oblique,
        "predicate_external_oblique": record.predicate_external_oblique,
        "predicate_internal_complement": (
            record.predicate_internal_complement
        ),
        "predicate_external_complement": (
            record.predicate_external_complement
        ),
        "predicate_internal_aux": record.predicate_internal_aux,
        "predicate_external_aux": record.predicate_external_aux,
        "predicate_internal_modifier": record.predicate_internal_modifier,
        "predicate_external_modifier": record.predicate_external_modifier,
        "predicate_internal_neg": record.predicate_internal_neg,
        "predicate_external_neg": record.predicate_external_neg,
        "predicate_internal_mark": record.predicate_internal_mark,
        "predicate_external_mark": record.predicate_external_mark,
    }


def _record_contains_root(record: CandidateSyntax) -> bool:
    return any(
        word is not None and word.head == 0
        for word in (record.head_word, record.root_word)
    )


def error_taxonomy_b_vs_c(
    baseline_aggregate: Mapping[str, Any],
    errors_c: Mapping[str, Mapping[str, Mapping[str, str | None]]],
    dataset: Mapping[str, Any],
    *,
    cells: Sequence[str] = CELLS_C,
) -> dict[str, dict[str, Any]]:
    baseline_counts = {
        cell: baseline_aggregate["error_taxonomy"][cell]
        for cell in BASELINE_B_CELLS
    }
    c_counts = error_taxonomy_counts(errors_c, dataset, cells=cells)
    comparison: dict[str, dict[str, Any]] = {}
    pairs = {
        "logistic_C": "logistic_B",
        "lightgbm_C": "lightgbm_B",
    }
    for c_cell in cells:
        b_cell = pairs[c_cell]
        baseline = baseline_counts[b_cell]
        current = c_counts[c_cell]
        codes: dict[str, Any] = {}
        for code in sorted(set(ERROR_CODES) | set(baseline["codes"])):
            codes[code] = {
                "baseline": baseline["codes"].get(code, 0),
                "phase2i": current["codes"].get(code, 0),
                "delta": (
                    current["codes"].get(code, 0)
                    - baseline["codes"].get(code, 0)
                ),
            }
        comparison[c_cell] = {
            "baseline_cell": b_cell,
            "correct": {
                "baseline": baseline["correct"],
                "phase2i": current["correct"],
                "delta": current["correct"] - baseline["correct"],
            },
            "codes": codes,
        }
    return comparison


def _token_key(sentence_id: int, token_id: int) -> str:
    return f"s{sentence_id}:t{token_id}"


def _word_key(sentence_id: int, word_id: int) -> str:
    return f"s{sentence_id}:w{word_id}"


def _objective_parser_error_reasons(
    record: CandidateSyntax,
    parse: UdParse,
) -> tuple[str, ...]:
    """Objective parser/alignment errors only.

    Per spec, only UNALIGNED (no intersecting parser token) and
    TOKEN_SURFACE_MISMATCH (candidate surface text missing from the aligned
    token span) are automatically safe parser/alignment errors.  AMBIGUOUS
    and multi-sentence spans are diagnostics, not automatic parser blame.
    """
    reasons: list[str] = []
    if record.boundary_status == BOUNDARY_UNALIGNED:
        reasons.append("UNALIGNED")
    if record.token_ids:
        token_by_id = {
            _token_key(sentence.sentence_id, token.token_id): token
            for sentence in parse.sentences
            for token in sentence.tokens
        }
        first = token_by_id[record.token_ids[0]]
        last = token_by_id[record.token_ids[-1]]
        span_text = parse.text[first.start_char:last.end_char]
        if record.text not in span_text:
            reasons.append("TOKEN_SURFACE_MISMATCH")
    return tuple(sorted(set(reasons)))


def _diagnostic_only_flags(record: CandidateSyntax) -> tuple[str, ...]:
    """Alignment observations that are not automatically parser errors."""
    flags: list[str] = []
    if record.boundary_status == BOUNDARY_AMBIGUOUS:
        flags.append("AMBIGUOUS_BOUNDARY")
    if record.spans_multiple_sentences:
        flags.append("SPANS_MULTIPLE_SENTENCES")
    if "MULTIWORD_BOUNDARY_CUT" in record.ambiguity:
        flags.append("MULTIWORD_BOUNDARY_CUT")
    if "MULTIPLE_CANDIDATE_HEADS" in record.ambiguity:
        flags.append("MULTIPLE_CANDIDATE_HEADS")
    return tuple(sorted(set(flags)))


def _explain_parser_error(
    record: CandidateSyntax,
    parse: UdParse,
    reasons: Sequence[str],
) -> dict[str, Any]:
    token_by_id = {
        _token_key(sentence.sentence_id, token.token_id): token
        for sentence in parse.sentences
        for token in sentence.tokens
    }
    word_by_id = {
        _word_key(sentence.sentence_id, word.word_id): word
        for sentence in parse.sentences
        for word in sentence.words
    }
    aligned_tokens = [token_by_id[key] for key in record.token_ids]
    first = aligned_tokens[0] if aligned_tokens else None
    last = aligned_tokens[-1] if aligned_tokens else None
    span_text = (
        parse.text[first.start_char:last.end_char]
        if first is not None and last is not None else ""
    )
    return {
        "window_id": record.window_id,
        "candidate_id": record.candidate_id,
        "text": record.text,
        "start": record.start,
        "end": record.end,
        "expected_text_slice": record.text,
        "aligned_token_span": {
            "start": first.start_char if first is not None else None,
            "end": last.end_char if last is not None else None,
            "text": span_text,
        },
        "reasons": list(reasons),
        "boundary_status": record.boundary_status,
        "ambiguity": list(record.ambiguity),
        "spans_multiple_sentences": record.spans_multiple_sentences,
        "tokens": [
            {
                "token_id": _token_key(
                    token.sentence_id, token.token_id,
                ),
                "text": token.text,
                "start_char": token.start_char,
                "end_char": token.end_char,
                "multiword": token.multiword,
                "word_ids": list(token.word_ids),
            }
            for token in aligned_tokens
        ],
        "words": [
            {
                "word_id": _word_key(word.sentence_id, word.word_id),
                "text": word.text,
                "lemma": word.lemma,
                "upos": word.upos,
                "xpos": word.xpos,
                "head": word.head,
                "deprel": word.deprel,
                "offset_kind": word.offset_kind,
            }
            for key, word in sorted(word_by_id.items())
            if key in record.word_ids
        ],
        "explanation": (
            "the candidate surface text is not represented by the aligned "
            "parser token span and/or no parser token intersects the "
            "candidate span; this is an objective alignment/parser error, "
            "not a claim about gold syntax"
        ),
    }


def parser_error_diagnostics(
    dataset: Mapping[str, Any],
    parses: Mapping[str, UdParse],
    records_by_window: Mapping[str, Sequence[CandidateSyntax]],
) -> dict[str, Any]:
    """Parser/alignment diagnostics.

    Objective parser errors are limited to UNALIGNED and
    TOKEN_SURFACE_MISMATCH (with full source/candidate/parser token/
    dependency/alignment explanations).  AMBIGUOUS boundaries and
    multi-sentence spans are reported as diagnostics only and never
    automatically blamed on the parser.
    """
    per_window: dict[str, dict[str, Any]] = {}
    total_counts: dict[str, int] = {}
    objective_errors: list[dict[str, Any]] = []
    diagnostic_only_total = 0
    for window_id in sorted(dataset["windows"]):
        parse = parses[window_id]
        counts = {status: 0 for status in BOUNDARY_STATUSES}
        window_errors: list[dict[str, Any]] = []
        window_diagnostic: list[dict[str, Any]] = []
        for record in records_by_window[window_id]:
            counts[record.boundary_status] += 1
            reasons = _objective_parser_error_reasons(record, parse)
            if reasons:
                window_errors.append(_explain_parser_error(
                    record, parse, reasons,
                ))
                continue
            flags = _diagnostic_only_flags(record)
            if flags:
                diagnostic_only_total += 1
                window_diagnostic.append({
                    "candidate_id": record.candidate_id,
                    "text": record.text,
                    "flags": list(flags),
                    "boundary_status": record.boundary_status,
                    "ambiguity": list(record.ambiguity),
                    "spans_multiple_sentences": (
                        record.spans_multiple_sentences
                    ),
                })
        objective_errors.extend(window_errors)
        for status, count in counts.items():
            total_counts[status] = total_counts.get(status, 0) + count
        per_window[window_id] = {
            "boundary_status_counts": counts,
            "objective_parser_error_count": len(window_errors),
            "objective_parser_errors": window_errors,
            "diagnostic_only_count": len(window_diagnostic),
            "diagnostic_only": window_diagnostic,
        }
    return {
        "definition": (
            "parser/alignment diagnostics; objective parser errors are only "
            "UNALIGNED or TOKEN_SURFACE_MISMATCH; AMBIGUOUS_BOUNDARY and "
            "SPANS_MULTIPLE_SENTENCES are diagnostic-only flags, never "
            "automatic parser blame; diagnostics only, never endpoint rules"
        ),
        "boundary_status_counts_total": total_counts,
        "objective_parser_error_total": len(objective_errors),
        "diagnostic_only_total": diagnostic_only_total,
        "per_window": per_window,
    }


def classify_all_errors_c(
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    parses: Mapping[str, UdParse],
    records_by_window: Mapping[str, Sequence[CandidateSyntax]],
    *,
    cells: Sequence[str] = CELLS_C,
) -> dict[str, dict[str, dict[str, str | None]]]:
    """Phase 2H error taxonomy plus activated PARSER_FEATURE_ERROR.

    ``PARSER_FEATURE_ERROR`` is assigned only to an otherwise-misclassified
    **DROP-row false positive** that carries a genuine objective parser/
    alignment error (UNALIGNED or TOKEN_SURFACE_MISMATCH).  KEEP-row false
    negatives always preserve their Phase 2H gold-rank code
    (``GOLD_RANKED_HIGH_THRESHOLD_MISS``/``GOLD_RANKED_LOW``), even when the
    row also has parser/alignment issues; those issues are recorded in
    parser diagnostics instead.  Ambiguous boundaries and multi-sentence
    spans are never automatic parser blame.
    """
    errors = classify_all_errors(dataset, rankings, cells=cells)
    parser_error_by_window: dict[str, set[str]] = {
        window_id: set() for window_id in dataset["windows"]
    }
    label_by_window: dict[str, dict[str, str]] = {}
    for window_id in sorted(dataset["windows"]):
        parse = parses[window_id]
        label_by_window[window_id] = {
            row.candidate_id: row.label
            for row in dataset["windows"][window_id]["rows"]
        }
        for record in records_by_window[window_id]:
            if _objective_parser_error_reasons(record, parse):
                parser_error_by_window[window_id].add(
                    record.candidate_id,
                )
    for cell in cells:
        for window_id in sorted(dataset["windows"]):
            for candidate_id in parser_error_by_window[window_id]:
                code = errors[cell][window_id][candidate_id]
                if code is not None:
                    if label_by_window[window_id][candidate_id] == DROP:
                        errors[cell][window_id][candidate_id] = (
                            "PARSER_FEATURE_ERROR"
                        )
    return errors


def universally_missed_analysis(
    baseline_window_tables: Mapping[str, Mapping[str, Any]],
    rankings_c: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    dataset: Mapping[str, Any],
    records_by_window: Mapping[str, Sequence[CandidateSyntax]],
    parses: Mapping[str, UdParse],
    *,
    cells: Sequence[str] = CELLS_C,
) -> dict[str, Any]:
    """Seven-universally-missed endpoint analysis.

    The endpoint set is always derived from the baseline window-table
    predictions (never only a hardcoded list), then validated against the
    frozen audit lock.  Each entry reports Phase 2H B and Phase 2I C ranks/
    scores, syntax summary, parser/alignment quality, B-to-C movement, and
    an evidence-based ``failure_appears_unrelated_to_syntax`` field.
    """
    entries = derive_universally_missed(baseline_window_tables)
    validation_problems = validate_universally_missed(entries)
    records_index = {
        window_id: {
            record.candidate_id: record
            for record in records_by_window[window_id]
        }
        for window_id in records_by_window
    }
    analysis: list[dict[str, Any]] = []
    for entry in entries:
        window_id = entry["window_id"]
        candidate_id = entry["candidate_id"]
        record = records_index[window_id][candidate_id]
        parse = parses[window_id]
        parser_flags = _objective_parser_error_reasons(record, parse)
        movement: dict[str, dict[str, Any]] = {}
        for c_cell in cells:
            b_cell = c_cell.replace("_C", "_B")
            baseline = entry["predictions"][b_cell]
            current = rankings_c[window_id][c_cell][candidate_id]
            delta_rank = current["rank"] - baseline["rank"]
            movement[c_cell] = {
                "direction": (
                    "up" if delta_rank < 0
                    else "down" if delta_rank > 0 else "tied"
                ),
                "rank_delta": delta_rank,
                "baseline_rank": baseline["rank"],
                "baseline_score": baseline["score"],
                "baseline_selected": baseline["selected"],
                "phase2i_rank": current["rank"],
                "phase2i_score": current["score"],
                "phase2i_selected": current["selected"],
            }
        unrelated = _failure_unrelated_evidence(
            record, parser_flags, movement,
        )
        analysis.append({
            "window_id": window_id,
            "candidate_id": candidate_id,
            "alias": entry["alias"],
            "text": entry["text"],
            "baseline_predictions": entry["predictions"],
            "phase2i_movement": movement,
            "syntax": _syntax_summary(record),
            "parser_alignment_quality": {
                "boundary_status": record.boundary_status,
                "ambiguity": list(record.ambiguity),
                "spans_multiple_sentences": record.spans_multiple_sentences,
                "diagnostic_flags": list(_diagnostic_only_flags(record)),
                "objective_parser_errors": list(parser_flags),
            },
            "failure_appears_unrelated_to_syntax": unrelated["value"],
            "failure_unrelated_judgment_available": unrelated[
                "judgment_available"
            ],
            "failure_unrelated_evidence": unrelated["evidence"],
            "failure_unrelated_definition": unrelated["definition"],
        })
    return {
        "definition": (
            "derived Phase 2H gold endpoints selected DROP by every "
            "baseline cell, with B/C ranks/scores, syntax summary, parser/"
            "alignment quality, B-to-C movement, and an evidence-based "
            "syntax-unrelatedness judgment"
        ),
        "derived_from_baseline_tables": True,
        "validation_problems": validation_problems,
        "validated": not validation_problems,
        "lock_sha256": canonical_sha256(UNIVERSALLY_MISSED_LOCK),
        "entries": analysis,
    }


def _failure_unrelated_evidence(
    record: CandidateSyntax,
    parser_flags: Sequence[str],
    movement: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence: list[str] = []
    aligned = record.boundary_status in (
        BOUNDARY_EXACT, BOUNDARY_TOKEN_ALIGNED,
    )
    evidence.append(
        "candidate boundary is exactly/token aligned with parser tokens"
        if aligned
        else f"candidate boundary is {record.boundary_status}",
    )
    evidence.append(
        "candidate lies within a single parsed sentence"
        if not record.spans_multiple_sentences
        else "candidate spans multiple parsed sentences",
    )
    evidence.append(
        "no alignment ambiguity flagged"
        if not record.ambiguity
        else "alignment ambiguity flagged",
    )
    evidence.append(
        "no objective parser/alignment error"
        if not parser_flags
        else "objective parser/alignment error present",
    )
    directions = {
        cell: str(values["direction"])
        for cell, values in movement.items()
    }
    evidence.append(
        "B-to-C rank movement: "
        + ", ".join(
            f"{cell}={direction}"
            for cell, direction in sorted(directions.items())
        ),
    )
    structurally_auditable = bool(
        aligned
        and not record.spans_multiple_sentences
        and not record.ambiguity
        and not parser_flags
    )
    if not structurally_auditable:
        value = False
    else:
        # This field is deliberately conservative: it only says that the
        # *current bounded syntax feature set* appears unrelated when a clean
        # parse supplied no upward rank movement in either C cell. It is not
        # a claim that syntax in general cannot explain the endpoint.
        value = all(
            direction != "up" for direction in directions.values()
        )
    return {
        "value": value,
        "judgment_available": structurally_auditable,
        "evidence": evidence,
        "definition": (
            "true only when parse/alignment is clean and neither C cell "
            "moves the endpoint upward; false when at least one C cell "
            "moves it upward or when parse/alignment is not clean enough to "
            "judge; judgment_available distinguishes those false cases"
        ),
    }


def syntax_logistic_coefficients(
    fitted_models: Mapping[str, Mapping[int, tuple[Any, list[str]]]],
    *,
    cells: Sequence[str] = ("logistic_C",),
) -> dict[str, Any]:
    """Syntax-feature-only logistic coefficients per fold and aggregate."""
    output: dict[str, Any] = {}
    for cell in cells:
        if cell != "logistic_C":
            continue
        per_fold: dict[str, dict[str, float]] = {}
        positive: list[tuple[str, float]] = []
        negative: list[tuple[str, float]] = []
        for fold_index, (model, names) in fitted_models[cell].items():
            coefficients = model.coef_[0]
            syntax = {
                str(names[index]): float(coefficients[index])
                for index in range(len(names))
                if str(names[index]).startswith("syntax:")
                or str(names[index]) in DENSE_C_EXTRA_FEATURES
            }
            per_fold[str(fold_index)] = syntax
            for name, value in syntax.items():
                if value >= 0:
                    positive.append((name, value))
                else:
                    negative.append((name, value))
        output[cell] = {
            "kind": "logistic_syntax_coefficients",
            "per_fold": per_fold,
            "aggregate_top_positive": _aggregate_top(positive, 15),
            "aggregate_top_negative": _aggregate_top(
                negative, 15, ascending=True,
            ),
            "syntax_feature_count": len(set(
                name for name, _ in positive + negative
            )),
        }
    return output


def syntax_vs_inherited_importance(
    fitted_models: Mapping[str, Mapping[int, tuple[Any, list[str]]]],
    *,
    cells: Sequence[str] = ("lightgbm_C",),
) -> dict[str, Any]:
    """LightGBM gain importance split: syntax vs inherited Phase 2H B.

    The frozen Phase 2H LightGBM config fixes ``importance_type=gain``, so
    ``model.feature_importances_`` is gain importance by construction.
    Every positive/nonzero per-feature gain is persisted in the complete
    ``importances`` list for each fold; ``top_importances`` is only the
    presentation-level top-20 prefix, and the aggregate lists are computed
    from the complete per-fold data rather than the displayed prefix.
    """
    if LGBM_CONFIG.get("importance_type") != "gain":
        raise Phase2IError(
            "syntax_vs_inherited_importance requires the frozen LightGBM "
            f"importance_type=gain; got "
            f"{LGBM_CONFIG.get('importance_type')!r}",
        )
    output: dict[str, Any] = {}
    for cell in cells:
        if cell != "lightgbm_C":
            continue
        per_fold: dict[str, dict[str, Any]] = {}
        syntax_totals: list[tuple[str, float]] = []
        inherited_totals: list[tuple[str, float]] = []
        for fold_index, (model, names) in fitted_models[cell].items():
            importance = model.feature_importances_
            items: list[tuple[str, float]] = []
            for index in range(len(names)):
                name = str(names[index])
                value = float(importance[index])
                if value <= 0:
                    continue
                items.append((name, value))
            items.sort(key=lambda item: (-item[1], item[0]))
            syntax_gain = 0.0
            inherited_gain = 0.0
            for name, value in items:
                if name.startswith("syntax:") or name in DENSE_C_EXTRA_FEATURES:
                    syntax_gain += value
                else:
                    inherited_gain += value
            total = syntax_gain + inherited_gain
            per_fold[str(fold_index)] = {
                "syntax_gain": syntax_gain,
                "inherited_gain": inherited_gain,
                "syntax_share": (
                    syntax_gain / total if total > 0 else None
                ),
                "importances": [
                    [name, value] for name, value in items
                ],
                "top_importances": [
                    [name, value] for name, value in items[:20]
                ],
            }
            syntax_totals.extend([
                (name, value) for name, value in items
                if name.startswith("syntax:") or name in DENSE_C_EXTRA_FEATURES
            ])
            inherited_totals.extend([
                (name, value) for name, value in items
                if not (
                    name.startswith("syntax:")
                    or name in DENSE_C_EXTRA_FEATURES
                )
            ])
        output[cell] = {
            "kind": "lightgbm_gain_importance_syntax_vs_inherited",
            "per_fold": per_fold,
            "aggregate_syntax_top": _aggregate_top(syntax_totals, 15),
            "aggregate_inherited_top": _aggregate_top(inherited_totals, 10),
        }
    return output


def training_vs_held_out_diagnostics(
    fit_scope: Mapping[str, Mapping[int, Mapping[str, Any]]],
    metrics_c: Mapping[str, Mapping[str, Any]],
    *,
    cells: Sequence[str] = CELLS_C,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for cell in cells:
        per_fold: dict[str, dict[str, Any]] = {}
        for fold_index in sorted(fit_scope[cell]):
            scope = fit_scope[cell][fold_index]
            test_window_id = scope["test_window_id"]
            test_metrics = metrics_c[cell]["per_fold"][test_window_id]
            per_fold[str(fold_index)] = {
                "test_window_id": test_window_id,
                "train": {
                    "candidate_count": scope["train_candidate_count"],
                    "positive_count": scope["train_positive_count"],
                    "negative_count": scope["train_negative_count"],
                    "average_precision": scope["train_diagnostics"][
                        "average_precision"
                    ],
                    "roc_auc": scope["train_diagnostics"]["roc_auc"],
                    "predicted_keep_count": scope["train_diagnostics"][
                        "predicted_keep_count"
                    ],
                },
                "held_out": {
                    "candidate_count": test_metrics["candidate_count"],
                    "positive_count": test_metrics["label_keep_count"],
                    "negative_count": test_metrics["label_drop_count"],
                    "average_precision": test_metrics["average_precision"][
                        "value"
                    ],
                    "roc_auc": test_metrics["roc_auc"]["value"],
                    "predicted_keep_count": test_metrics["selected"],
                },
                "b_token_oov_count": scope["vectorizer"][
                    "test_oov_token_type_count"
                ],
                "syntax_oov_count": scope["syntax_encoder"][
                    "oov_value_count"
                ],
            }
        output[cell] = {"per_fold": per_fold}
    return output


def run_experiment_c(
    benchmark: Mapping[str, Any] | str | Path,
    *,
    cells: Sequence[str] = CELLS_C,
    assets_dir: str | Path,
    baseline_archive: str | Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Full offline Phase 2I experiment: dataset, local parse, C CV, metrics,
    deltas vs frozen B, diagnostics."""
    if isinstance(benchmark, (str, Path)):
        benchmark = load_benchmark(benchmark)
    else:
        validate_benchmark_mapping(benchmark)
    dataset = build_dataset(benchmark)
    baseline = load_phase2h_baseline(baseline_archive)
    try:
        immutability = validate_early_immutability(
            dataset, baseline["window_tables"],
        )
        if immutability["problems"]:
            raise Phase2IError(
                "Phase 2H baseline immutability audit failed before parse/"
                "training: "
                + "; ".join(immutability["problems"][:5]),
            )
        parses = {
            window_id: parse_window_text(
                window["bronze_text"], window_id,
                assets_dir=assets_dir, verbose=verbose,
            )
            for window_id, window in sorted(dataset["windows"].items())
        }
        cv = run_cv_c(dataset, parses, cells=cells, verbose=verbose)
        fold_problems = validate_cv_folds_match_baseline(
            cv["folds"], baseline["aggregate"]["folds"],
        )
        if fold_problems:
            raise Phase2IError(
                "Phase 2I CV folds differ from the archived Phase 2H folds: "
                + "; ".join(fold_problems),
            )
        rankings = compute_rankings(dataset, cv["oof_scores"], cells=cells)
        errors = classify_all_errors_c(
            dataset, rankings, parses, cv["candidate_syntax"], cells=cells,
        )
        metrics = {
            cell: compute_cell_metrics_c(dataset, rankings, cell)
            for cell in cells
        }
        baseline_metrics = {
            cell: baseline["aggregate"]["metrics"][cell]
            for cell in BASELINE_B_CELLS
        }
        deltas = build_deltas(baseline_metrics, metrics, cells=cells)
        universally_missed = derive_universally_missed(
            baseline["window_tables"],
        )
        universally_missed_problems = validate_universally_missed(
            universally_missed,
        )
        missed_analysis = universally_missed_analysis(
            baseline["window_tables"], rankings, dataset,
            cv["candidate_syntax"], parses, cells=cells,
        )
        baseline_rankings = baseline_rankings_from_tables(
            baseline["window_tables"],
        )
        overlap = overlap_cluster_syntax_diagnostics(
            dataset, rankings, baseline_rankings,
            cv["candidate_syntax"], cells=cells,
        )
        parser_diagnostics = parser_error_diagnostics(
            dataset, parses, cv["candidate_syntax"],
        )
        syntax_coefficients = syntax_logistic_coefficients(
            cv["fitted_models"], cells=cells,
        )
        importance_split = syntax_vs_inherited_importance(
            cv["fitted_models"], cells=cells,
        )
        training_held_out = training_vs_held_out_diagnostics(
            cv["fit_scope"], metrics, cells=cells,
        )
        error_taxonomy = error_taxonomy_b_vs_c(
            baseline["aggregate"], errors, dataset, cells=cells,
        )
    finally:
        close_phase2h_baseline(baseline)
    return {
        "dataset": dataset,
        "parses": parses,
        "baseline_immutability_audit": immutability,
        "folds": cv["folds"],
        "fit_scope": cv["fit_scope"],
        "oof_scores": cv["oof_scores"],
        "fitted_models": cv["fitted_models"],
        "fold_match_validation": {
            "validated": not fold_problems,
            "problems": fold_problems,
            "compared_against": "archived Phase 2H run-1 folds",
        },
        "rankings": rankings,
        "errors": errors,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "deltas": deltas,
        "universally_missed": universally_missed,
        "universally_missed_problems": universally_missed_problems,
        "universally_missed_analysis": missed_analysis,
        "overlap_cluster_syntax_diagnostics": overlap,
        "parser_diagnostics": parser_diagnostics,
        "syntax_coefficients": syntax_coefficients,
        "syntax_vs_inherited_importance": importance_split,
        "training_vs_held_out": training_held_out,
        "error_taxonomy_b_vs_c": error_taxonomy,
        "candidate_syntax": cv["candidate_syntax"],
        "cells": list(cells),
    }


def _definition_c(cells: Sequence[str] = CELLS_C) -> dict[str, Any]:
    return {
        "run_version": RUN_VERSION,
        "task": (
            "Phase 2I Universal Dependencies/syntactic Feature Set C ablation "
            "over the frozen Phase 2H candidate-level KEEP/DROP benchmark"
        ),
        "keep_threshold": KEEP_THRESHOLD,
        "seed": SEED,
        "cells": list(cells),
        "model_configs": {
            "logistic": dict(LOGISTIC_CONFIG),
            "lightgbm": {
                key: value for key, value in LGBM_CONFIG.items()
                if key != "class_weight"
            },
        },
        "feature_schema": feature_schema_c(),
        "fold_design": {
            "kind": "grouped_leave_one_window_out",
            "group_key": "window",
            "windows": "locked five-case Phase 2F/2G benchmark",
            "tie_breaking": (
                "descending score; deterministic candidate catalog order "
                "(ascending start/end)"
            ),
            "rank_base": 1,
            "fold_guard": (
                "never train/test a fold whose training windows have no "
                "positive examples"
            ),
        },
        "error_taxonomy": list(ERROR_PRECEDENCE),
        "gold_rank_high_k": GOLD_RANK_HIGH_K,
        "rank_ks": list(RANK_KS),
        "parse_definition": parse_definition(),
        "baseline_of_record": {
            "phase2h_run1_archive_sha256": PHASE2H_RUN1_ARCHIVE_SHA256,
            "phase2h_run1_aggregate_sha256": PHASE2H_RUN1_AGGREGATE_SHA256,
            "baseline_b_cells": list(BASELINE_B_CELLS),
            "note": (
                "frozen Phase 2H B metrics/predictions are loaded and "
                "hash-verified; A/B are never retrained"
            ),
        },
        "no_llm": True,
        "no_generative_endpoint_model": True,
        "syntax_is_learned_only": True,
    }


def _dependency_versions_c() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for name in (
        "scikit-learn", "lightgbm", "numpy", "scipy", "stanza", "torch",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


_BASELINE_CANDIDATE_FIELDS = (
    "case_id", "window_id", "candidate_id", "alias", "start", "end",
    "absolute_start", "absolute_end", "text", "segment_ids",
    "segment_bounds", "type_hints", "label", "excluded",
    "ambiguity_state", "gold_mention_ids", "gold_node_types",
)


def _validate_baseline_window_table(
    window: Mapping[str, Any],
    baseline_table: Mapping[str, Any],
) -> list[str]:
    """Compare every frozen provenance field and candidate record against the
    Phase 2H baseline before any Phase 2I syntax/C prediction is added."""
    problems: list[str] = []
    for key in (
        "case_id", "window_id", "bronze_text_sha256", "catalog_sha256",
        "candidate_generator_version",
    ):
        if baseline_table.get(key) != window.get(key):
            problems.append(
                f"baseline table {key} {baseline_table.get(key)!r} != "
                f"frozen dataset {window.get(key)!r}",
            )
    baseline_candidates = baseline_table.get("candidates")
    rows = window["rows"]
    if not isinstance(baseline_candidates, list):
        problems.append("baseline table candidates must be a list")
    elif baseline_table.get("candidate_count") != len(baseline_candidates):
        problems.append(
            f"baseline candidate_count {baseline_table.get('candidate_count')} "
            f"!= list length {len(baseline_candidates)}",
        )
    if not isinstance(baseline_candidates, list) or len(
        baseline_candidates,
    ) != len(rows):
        if isinstance(baseline_candidates, list):
            problems.append(
                f"baseline candidate count {len(baseline_candidates)} != "
                f"frozen dataset {len(rows)}",
            )
        return problems
    if window.get("bronze_text_sha256") != hashlib.sha256(
        window.get("bronze_text", "").encode("utf-8"),
    ).hexdigest():
        problems.append(
            "frozen dataset bronze_text_sha256 does not hash its "
            "bronze_text",
        )
    for index, (row, candidate) in enumerate(zip(rows, baseline_candidates)):
        for field in _BASELINE_CANDIDATE_FIELDS:
            if candidate.get(field) != _json_value(getattr(row, field)):
                problems.append(
                    f"baseline candidate {index} ({row.candidate_id}) "
                    f"{field} differs from the frozen dataset",
                )
        predictions = candidate.get("predictions")
        if not isinstance(predictions, Mapping):
            problems.append(
                f"baseline candidate {row.candidate_id} has no predictions",
            )
        else:
            for cell in BASELINE_CELLS:
                prediction = predictions.get(cell)
                if (
                    not isinstance(prediction, Mapping)
                    or not {"score", "rank", "selected"} <= set(prediction)
                ):
                    problems.append(
                        f"baseline candidate {row.candidate_id} {cell} "
                        "prediction is malformed",
                    )
    return problems


def validate_early_immutability(
    dataset: Mapping[str, Any],
    baseline_window_tables: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare the frozen Phase 2H dataset to the archived Phase 2H window
    tables before any parsing or training happens.

    Every window-level provenance field and every candidate, in exact catalog
    order, is compared for ID/span/absolute offsets/text/labels/exclusions/
    ambiguity/gold metadata; the archived generator/catalog/Bronze locks are
    also cross-checked.  This audit is the first contract checked by
    :func:`run_experiment_c` and is recorded in the aggregate artifact.
    """
    locked_window_ids = sorted(dataset["windows"])
    problems: list[str] = []
    if set(baseline_window_tables) != set(locked_window_ids):
        problems.append(
            "archived Phase 2H window set does not match the frozen "
            "five-case benchmark",
        )
    windows_compared: dict[str, dict[str, Any]] = {}
    for window_id in locked_window_ids:
        if window_id not in baseline_window_tables:
            problems.append(f"missing baseline table for {window_id}")
            continue
        window_problems = _validate_baseline_window_table(
            dataset["windows"][window_id],
            baseline_window_tables[window_id],
        )
        problems.extend(window_problems)
        windows_compared[window_id] = {
            "candidate_count": len(dataset["windows"][window_id]["rows"]),
            "problems": window_problems,
        }
    return {
        "definition": (
            "frozen Phase 2H dataset vs archived Phase 2H run-1 window "
            "tables, compared in exact candidate order for all window "
            "provenance fields and candidate ID/span/absolute offsets/"
            "text/label/exclusion/ambiguity/gold metadata, plus generator/"
            "catalog/Bronze locks; checked before parsing or training"
        ),
        "window_count": len(locked_window_ids),
        "candidate_count": sum(
            len(dataset["windows"][window_id]["rows"])
            for window_id in locked_window_ids
        ),
        "positive_count": sum(
            1
            for window_id in locked_window_ids
            for row in dataset["windows"][window_id]["rows"]
            if row.label == KEEP
        ),
        "validated": not problems,
        "problems": problems,
        "problems_sha256": canonical_sha256(problems),
        "windows_compared": windows_compared,
    }


def _raise_baseline_mismatch(problems: Sequence[str]) -> None:
    if problems:
        raise Phase2IError(
            "Phase 2H baseline window table does not match the frozen "
            "dataset: " + "; ".join(list(problems)[:5]),
        )


def _parser_tokens_for_record(
    parse: UdParse,
    record: CandidateSyntax,
) -> list[dict[str, Any]]:
    token_by_id = {
        _token_key(sentence.sentence_id, token.token_id): token
        for sentence in parse.sentences
        for token in sentence.tokens
    }
    return [
        {
            "token_id": key,
            "text": token.text,
            "start_char": token.start_char,
            "end_char": token.end_char,
            "multiword": token.multiword,
            "word_ids": list(token.word_ids),
        }
        for key, token in token_by_id.items()
        if key in record.token_ids
    ]


def _parser_words_for_record(
    parse: UdParse,
    record: CandidateSyntax,
) -> list[dict[str, Any]]:
    word_by_id = {
        _word_key(sentence.sentence_id, word.word_id): word
        for sentence in parse.sentences
        for word in sentence.words
    }
    return [
        {
            "word_id": key,
            "text": word.text,
            "lemma": word.lemma,
            "upos": word.upos,
            "xpos": word.xpos,
            "head": word.head,
            "deprel": word.deprel,
            "offset_kind": word.offset_kind,
        }
        for key, word in word_by_id.items()
        if key in record.word_ids
    ]


def _candidate_syntax_projection(
    parse: UdParse,
    record: CandidateSyntax,
) -> dict[str, Any]:
    """Canonical complete projection of one persisted candidate syntax
    object from the normalized parse plus recomputed candidate evidence.

    Publication (:func:`build_phase2i_window_table`) and acceptance
    verification (:func:`_verify_candidate_syntax_and_diagnostics`) share
    this single serializer, so every persisted syntax field is compared
    recursively against independently recomputed evidence rather than only
    against the stored evidence hash.
    """
    head = record.head_word
    root = record.root_word
    return {
        "boundary_status": record.boundary_status,
        "start_aligned": record.start_aligned,
        "end_aligned": record.end_aligned,
        "multi_token": record.multi_token,
        "spans_multiple_sentences": record.spans_multiple_sentences,
        "ambiguity": list(record.ambiguity),
        "sentence_ids": list(record.sentence_ids),
        "token_ids": list(record.token_ids),
        "word_ids": list(record.word_ids),
        "candidate_head": record.candidate_head_id,
        "candidate_root": record.candidate_root_id,
        "head": {
            "lemma": head.lemma if head else None,
            "upos": head.upos if head else None,
            "xpos": head.xpos if head else None,
            "deprel": head.deprel if head else None,
            "feats": head.feats if head else None,
            "dependent_count": record.head_dependent_count,
            "child_deprels": list(record.head_child_deprels),
            "relation_context": list(record.head_relation_context),
        } if head else None,
        "root": {
            "lemma": root.lemma if root else None,
            "upos": root.upos if root else None,
        } if root else None,
        "word_depths": dict(zip(
            record.word_ids, record.candidate_depth_values,
        )),
        "root_ids": list(record.root_ids),
        "multiple_roots": record.multiple_roots,
        "syntactic_predicates": list(record.syntactic_predicate_ids),
        "boundary_heads": list(record.boundary_head_ids),
        "external_governors": list(record.external_governor_ids),
        "external_governor_lemmas": list(record.external_governor_lemmas),
        "external_governor_uposes": list(record.external_governor_uposes),
        "external_governor_deprels": list(record.external_governor_deprels),
        "crossing_arcs": record.crossing_arc_count,
        "subtree_size": record.subtree_size,
        "subtree_intersection_count": record.subtree_intersection_count,
        "subtree_word_fraction": (
            record.subtree_intersection_count / len(record.word_ids)
            if record.word_ids else 0.0
        ),
        "subtree_fraction": (
            record.subtree_intersection_count / record.subtree_size
            if record.subtree_size else 0.0
        ),
        "subtree_exact": record.subtree_exact,
        "span_connected": record.span_connected,
        "clause_roots": list(record.clause_root_ids),
        "multiple_clause_roots": len(record.clause_root_ids) > 1,
        "finite_verbs": list(record.finite_verb_ids),
        "aux": list(record.aux_ids),
        "modals": list(record.modal_ids),
        "negations": list(record.neg_ids),
        "marks": list(record.mark_ids),
        "cases": list(record.case_ids),
        "relations": list(record.relations),
        "internal_rel_context": list(record.internal_relation_context),
        "crossing_incoming_rel_context": list(
            record.crossing_incoming_relation_context,
        ),
        "crossing_outgoing_rel_context": list(
            record.crossing_outgoing_relation_context,
        ),
        "predicate_argument_context": list(record.predicate_argument_context),
        "scope_internal": list(record.scope_internal_ids),
        "scope_external": list(record.scope_external_ids),
        "pronouns": list(record.pronoun_ids),
        "actions": list(record.action_ids),
        "predicate_argument_internal": record.predicate_internal_argument,
        "predicate_argument_external": record.predicate_external_argument,
        "predicate_internal_subject": record.predicate_internal_subject,
        "predicate_external_subject": record.predicate_external_subject,
        "predicate_internal_object": record.predicate_internal_object,
        "predicate_external_object": record.predicate_external_object,
        "predicate_internal_oblique": record.predicate_internal_oblique,
        "predicate_external_oblique": record.predicate_external_oblique,
        "predicate_internal_complement": (
            record.predicate_internal_complement
        ),
        "predicate_external_complement": (
            record.predicate_external_complement
        ),
        "predicate_internal_aux": record.predicate_internal_aux,
        "predicate_external_aux": record.predicate_external_aux,
        "predicate_internal_modifier": record.predicate_internal_modifier,
        "predicate_external_modifier": record.predicate_external_modifier,
        "predicate_internal_neg": record.predicate_internal_neg,
        "predicate_external_neg": record.predicate_external_neg,
        "predicate_internal_mark": record.predicate_internal_mark,
        "predicate_external_mark": record.predicate_external_mark,
        "aux_internal": record.aux_internal,
        "aux_external": record.aux_external,
        "neg_internal": record.neg_internal,
        "neg_external": record.neg_external,
        "mark_internal": record.mark_internal,
        "mark_external": record.mark_external,
        "case_internal": record.case_internal,
        "case_external": record.case_external,
        "scope_governor_inside": bool(record.scope_internal_ids),
        "scope_governor_outside": bool(record.scope_external_ids),
        "pronoun_inside_governor": record.pronoun_inside_governor,
        "pronoun_outside_governor": record.pronoun_outside_governor,
        "action_internal_argument": record.action_internal_argument,
        "action_external_argument": record.action_external_argument,
        "action_complement": record.action_complement,
        "action_modifier": record.action_modifier,
        "groups": [
            {"group": name, "values": list(values)}
            for name, values in record.groups
        ],
        "parser_tokens": _parser_tokens_for_record(parse, record),
        "parser_words": _parser_words_for_record(parse, record),
        "evidence_sha256": record.evidence_sha256(),
    }


def build_phase2i_window_table(
    dataset: Mapping[str, Any],
    rankings_c: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    errors_c: Mapping[str, Mapping[str, Mapping[str, str | None]]],
    baseline_table: Mapping[str, Any],
    parse: UdParse,
    records: Sequence[CandidateSyntax],
    window_id: str,
    *,
    cells: Sequence[str] = CELLS_C,
) -> dict[str, Any]:
    window = dataset["windows"][window_id]
    _raise_baseline_mismatch(
        _validate_baseline_window_table(window, baseline_table),
    )
    records_by_id = {record.candidate_id: record for record in records}
    baseline_candidates = baseline_table["candidates"]
    candidates: list[dict[str, Any]] = []
    for row, baseline_candidate in zip(
        window["rows"], baseline_candidates,
    ):
        record = records_by_id.get(row.candidate_id)
        if record is None:
            raise Phase2IError(
                f"syntax records miss candidate {row.candidate_id}",
            )
        if (
            record.window_id != window["window_id"]
            or record.candidate_id != row.candidate_id
            or record.start != row.start
            or record.end != row.end
            or record.text != row.text
        ):
            raise Phase2IError(
                f"syntax record identity mismatch for {row.candidate_id}",
            )
        candidates.append({
            "case_id": row.case_id,
            "window_id": row.window_id,
            "candidate_id": row.candidate_id,
            "alias": row.alias,
            "start": row.start,
            "end": row.end,
            "absolute_start": row.absolute_start,
            "absolute_end": row.absolute_end,
            "text": row.text,
            "segment_ids": list(row.segment_ids),
            "segment_bounds": [
                [bound_start, bound_end]
                for bound_start, bound_end in row.segment_bounds
            ],
            "type_hints": list(row.type_hints),
            "label": row.label,
            "excluded": row.excluded,
            "ambiguity_state": row.ambiguity_state,
            "gold_mention_ids": list(row.gold_mention_ids),
            "gold_node_types": list(row.gold_node_types),
            "predictions": {
                **{
                    cell: {
                        **rankings_c[window_id][cell][row.candidate_id],
                        "error_code": errors_c[cell][window_id][
                            row.candidate_id
                        ],
                    }
                    for cell in cells
                },
                **{
                    cell: baseline_candidate["predictions"][cell]
                    for cell in BASELINE_CELLS
                },
            },
            "syntax": _candidate_syntax_projection(parse, record),
            "parse_sha256": parse.parse_sha256,
        })
    return {
        "case_id": window["case_id"],
        "window_id": window["window_id"],
        "bronze_text_sha256": window["bronze_text_sha256"],
        "catalog_sha256": window["catalog_sha256"],
        "candidate_generator_version": window["candidate_generator_version"],
        "candidate_count": len(candidates),
        "parse_sha256": parse.parse_sha256,
        "candidates": candidates,
    }


def build_aggregate_c(
    benchmark_path: Path,
    result: Mapping[str, Any],
    *,
    repo: Path,
    created_at: str | None = None,
    window_table_hashes: Mapping[str, str] | None = None,
    assets_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_path)
    benchmark_file_sha256 = hashlib.sha256(
        benchmark_path.read_bytes(),
    ).hexdigest()
    definition = _definition_c(cells=result.get("cells", CELLS_C))
    commit, dirty = _git_state_c(repo)
    dataset_summary = validate_dataset(result["dataset"])
    parse_hashes = {
        window_id: result["parses"][window_id].parse_sha256
        for window_id in sorted(result["dataset"]["windows"])
    }
    if window_table_hashes is None:
        raise Phase2IError(
            "window_table_hashes are required to build the aggregate",
        )
    table_hashes = {
        window_id: {
            "candidate_table_sha256": window_table_hashes[window_id],
        }
        for window_id in sorted(result["dataset"]["windows"])
    }
    inner = {
        "run_version": RUN_VERSION,
        "created_at": created_at or datetime.now(
            timezone.utc,
        ).isoformat().replace("+00:00", "Z"),
        "git_commit": commit,
        "repository_dirty": dirty,
        "definition": definition,
        "definition_sha256": canonical_sha256(definition),
        "input_hashes": {
            "benchmark_content_sha256": BENCHMARK_CONTENT_SHA256,
            "benchmark_file_sha256": benchmark_file_sha256,
            "phase2h_run1_archive_sha256": PHASE2H_RUN1_ARCHIVE_SHA256,
            "phase2h_run1_aggregate_sha256": PHASE2H_RUN1_AGGREGATE_SHA256,
        },
        "benchmark_content_sha256": BENCHMARK_CONTENT_SHA256,
        "dataset_summary": dataset_summary,
        "baseline_immutability_audit": result["baseline_immutability_audit"],
        "folds": result["folds"],
        "fold_match_validation": result["fold_match_validation"],
        "fit_scope": result["fit_scope"],
        "metrics": result["metrics"],
        "baseline_metrics": result["baseline_metrics"],
        "deltas": result["deltas"],
        "universally_missed": result["universally_missed"],
        "universally_missed_validation": {
            "problems": result["universally_missed_problems"],
            "validated": not result["universally_missed_problems"],
            "lock_sha256": canonical_sha256(UNIVERSALLY_MISSED_LOCK),
        },
        "universally_missed_analysis": (
            result["universally_missed_analysis"]
        ),
        "error_taxonomy_b_vs_c": result["error_taxonomy_b_vs_c"],
        "overlap_cluster_syntax_diagnostics": (
            result["overlap_cluster_syntax_diagnostics"]
        ),
        "parser_diagnostics": result["parser_diagnostics"],
        "syntax_coefficients": result["syntax_coefficients"],
        "syntax_vs_inherited_importance": (
            result["syntax_vs_inherited_importance"]
        ),
        "training_vs_held_out": result["training_vs_held_out"],
        "parse_hashes": parse_hashes,
        "assets_provenance": assets_provenance or {},
        "dependencies": _dependency_versions_c(),
        "window_tables": table_hashes,
    }
    return {"content_sha256": canonical_sha256(inner), **inner}


def _git_state_c(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip())
    return commit, dirty


def publish_phase2i_artifact(
    output: Path,
    aggregate: Mapping[str, Any],
    window_tables: Mapping[str, Mapping[str, Any]],
    parse_tables: Mapping[str, Mapping[str, Any]],
    *,
    benchmark_path: str | Path,
    baseline_archive: str | Path,
    assets_dir: str | Path = DEFAULT_PARSER_ASSETS,
) -> Path:
    """Atomically publish the immutable Phase 2I artifact.

    The complete temporary artifact is self-verified with the fail-closed
    acceptance verifier before the atomic ``os.replace``; a verification
    failure leaves no output directory behind.
    """
    path_problems = _symlink_ancestor_problems(output)
    if path_problems:
        raise ValueError(
            "artifact output path is unsafe: " + "; ".join(path_problems),
        )
    output = Path(os.path.abspath(os.fspath(output)))
    if output.exists():
        raise ValueError(
            "output directory already exists; Phase 2I artifacts are "
            "immutable",
        )
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    path_problems = _symlink_ancestor_problems(output)
    if path_problems:
        raise ValueError(
            "artifact output path became unsafe: "
            + "; ".join(path_problems),
        )
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=parent))
    files: list[Path] = []
    try:
        aggregate_path = temporary / "phase2i-syntax-features.json"
        aggregate_path.write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files.append(aggregate_path)
        window_dir = temporary / "windows"
        for window_id, table in sorted(window_tables.items()):
            window_path = window_dir / f"{window_id}.json"
            window_path.parent.mkdir(parents=True, exist_ok=True)
            window_path.write_text(
                json.dumps(table, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            files.append(window_path)
        parser_dir = temporary / "parser"
        for window_id, parse_table in sorted(parse_tables.items()):
            parse_path = parser_dir / f"{window_id}.json"
            parse_path.parent.mkdir(parents=True, exist_ok=True)
            parse_path.write_text(
                json.dumps(parse_table, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            files.append(parse_path)
        manifest = {
            "files": [
                {
                    "path": str(path.relative_to(temporary)),
                    "file_sha256": hashlib.sha256(
                        path.read_bytes(),
                    ).hexdigest(),
                }
                for path in sorted(
                    files, key=lambda item: str(item.relative_to(temporary)),
                )
            ],
        }
        manifest_path = temporary / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files.append(manifest_path)
        verification_problems = _verify_phase2i_artifact(
            temporary,
            benchmark_path=benchmark_path,
            baseline_archive=baseline_archive,
            assets_dir=assets_dir,
        )
        if verification_problems:
            raise Phase2IError(
                "published Phase 2I artifact failed its own acceptance "
                "verification: " + "; ".join(
                    verification_problems[:50],
                ),
            )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


_REQUIRED_AGGREGATE_KEYS = frozenset({
    "run_version", "created_at", "git_commit", "repository_dirty",
    "definition", "definition_sha256", "input_hashes",
    "benchmark_content_sha256", "dataset_summary",
    "baseline_immutability_audit", "folds", "fold_match_validation",
    "fit_scope", "metrics", "baseline_metrics", "deltas",
    "universally_missed", "universally_missed_validation",
    "universally_missed_analysis", "error_taxonomy_b_vs_c",
    "overlap_cluster_syntax_diagnostics", "parser_diagnostics",
    "syntax_coefficients", "syntax_vs_inherited_importance",
    "training_vs_held_out", "parse_hashes", "assets_provenance",
    "dependencies", "window_tables",
})

_INPUT_HASH_KEYS = frozenset({
    "benchmark_content_sha256", "benchmark_file_sha256",
    "phase2h_run1_archive_sha256", "phase2h_run1_aggregate_sha256",
})

_DEPENDENCY_KEYS = frozenset({
    "python", "scikit-learn", "lightgbm", "numpy", "scipy", "stanza",
    "torch",
})

_WINDOW_TABLE_KEYS = frozenset({
    "case_id", "window_id", "bronze_text_sha256", "catalog_sha256",
    "candidate_generator_version", "candidate_count", "parse_sha256",
    "candidates",
})

_CANDIDATE_KEYS = frozenset({
    "case_id", "window_id", "candidate_id", "alias", "start", "end",
    "absolute_start", "absolute_end", "text", "segment_ids",
    "segment_bounds", "type_hints", "label", "excluded",
    "ambiguity_state", "gold_mention_ids", "gold_node_types",
    "predictions", "syntax", "parse_sha256",
})

_PREDICTION_KEYS = frozenset({"score", "rank", "selected", "error_code"})

_FOLD_KEYS = frozenset({
    "fold_index", "train_window_ids", "test_window_id",
    "train_candidate_count", "train_positive_count",
    "train_negative_count", "test_candidate_count",
    "test_positive_count", "class_weights",
})

_FOLD_MATCH_VALIDATION_KEYS = frozenset({
    "validated", "problems", "compared_against",
})

_IMMUTABILITY_KEYS = frozenset({
    "definition", "window_count", "candidate_count", "positive_count",
    "validated", "problems", "problems_sha256", "windows_compared",
})

_FIT_SCOPE_RECORD_KEYS = frozenset({
    "fold_index", "train_window_ids", "test_window_id", "fit_scope",
    "train_candidate_count", "train_positive_count",
    "train_negative_count", "class_weights", "scaler", "vectorizer",
    "syntax_encoder", "model_config", "feature_names_sha256",
    "train_diagnostics",
})

_SCALER_KEYS = frozenset({
    "fit_on", "feature_count", "mean_sha256", "scale_sha256",
})

_VECTORIZER_KEYS = frozenset({
    "fit_on", "params", "vocabulary_size", "vocabulary_sha256",
    "oov_definition", "test_oov_token_types",
    "test_oov_token_type_count", "test_oov_token_types_sha256",
})

_SYNTAX_ENCODER_KEYS = frozenset({
    "fit_on", "params", "vocabulary_sizes", "vocabulary_sha256",
    "oov_definition", "per_group", "oov_value_count", "oov_sha256",
})

_TRAIN_DIAGNOSTICS_KEYS = frozenset({
    "candidate_count", "positive_count", "negative_count",
    "predicted_keep_count", "average_precision", "roc_auc",
})

_TRAIN_DIAGNOSTICS_INT_KEYS = frozenset({
    "candidate_count", "positive_count", "negative_count",
    "predicted_keep_count",
})

_TRAIN_DIAGNOSTICS_FLOAT_KEYS = frozenset({
    "average_precision", "roc_auc",
})

_MODEL_CONFIG_KEYS = frozenset({"family", "params"})

_UNIVERSALLY_MISSED_VALIDATION_KEYS = frozenset({
    "problems", "validated", "lock_sha256",
})

_ASSETS_PROVENANCE_KEYS = frozenset({
    "verified", "problems", "schema_version", "stanza_version",
    "package", "processors", "manifest_sha256", "files",
})

_SYNTAX_TABLE_KEYS = frozenset({
    "boundary_status", "start_aligned", "end_aligned", "multi_token",
    "spans_multiple_sentences", "ambiguity", "sentence_ids",
    "token_ids", "word_ids", "candidate_head", "candidate_root",
    "head", "root", "word_depths", "root_ids", "multiple_roots",
    "syntactic_predicates", "boundary_heads", "external_governors",
    "external_governor_lemmas", "external_governor_uposes",
    "external_governor_deprels", "crossing_arcs", "subtree_size",
    "subtree_intersection_count", "subtree_word_fraction",
    "subtree_fraction", "subtree_exact", "span_connected",
    "clause_roots", "multiple_clause_roots", "finite_verbs", "aux",
    "modals", "negations", "marks", "cases", "relations",
    "internal_rel_context", "crossing_incoming_rel_context",
    "crossing_outgoing_rel_context", "predicate_argument_context",
    "scope_internal", "scope_external", "pronouns", "actions",
    "predicate_argument_internal", "predicate_argument_external",
    "predicate_internal_subject", "predicate_external_subject",
    "predicate_internal_object", "predicate_external_object",
    "predicate_internal_oblique", "predicate_external_oblique",
    "predicate_internal_complement", "predicate_external_complement",
    "predicate_internal_aux", "predicate_external_aux",
    "predicate_internal_modifier", "predicate_external_modifier",
    "predicate_internal_neg", "predicate_external_neg",
    "predicate_internal_mark", "predicate_external_mark",
    "aux_internal", "aux_external", "neg_internal", "neg_external",
    "mark_internal", "mark_external", "case_internal",
    "case_external", "scope_governor_inside",
    "scope_governor_outside", "pronoun_inside_governor",
    "pronoun_outside_governor", "action_internal_argument",
    "action_external_argument", "action_complement", "action_modifier",
    "groups", "parser_tokens", "parser_words", "evidence_sha256",
})

_ALL_PREDICTION_CELLS = CELLS_C + BASELINE_CELLS

_BENCHMARK_VERIFY_CACHE: dict[str, Mapping[str, Any]] = {}
_BASELINE_VERIFY_CACHE: dict[str, dict[str, Any]] = {}
_PARSE_VERIFY_CACHE: dict[str, UdParse] = {}
_STANZA_REPLAY_CACHE: dict[str, UdParse] = {}
_MODEL_REFIT_OUTPUTS_CACHE: dict[str, dict[str, Any]] = {}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _json_equal(left: object, right: object) -> bool:
    """Exact structural equality for deterministic recomputation.

    Phase 2I locks dependency versions and seeds, so a tolerance would only
    permit authenticated artifact values to be changed after the run.
    """
    return _strict_equal(left, right)


def _strict_equal(left: object, right: object) -> bool:
    """Exact recursive structural equality for frozen/locked values.

    Types must match exactly (``True != 1`` and ``1 != 1.0``), mappings are
    compared by key set, and lists are compared elementwise.  Floats compare
    exactly, so this must only be used for values whose bit patterns are
    fixed by JSON round-tripping of identical deterministic computations --
    never for independently recomputed floating aggregates.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_equal(item_left, item_right)
            for item_left, item_right in zip(left, right)
        )
    return left == right


def _cached_benchmark_mapping(path: Path) -> Mapping[str, Any]:
    digest = _file_sha256(path)
    cached = _BENCHMARK_VERIFY_CACHE.get(digest)
    if cached is None:
        cached = load_benchmark(path)
        _BENCHMARK_VERIFY_CACHE[digest] = cached
    return cached


def _cached_baseline(path: Path) -> dict[str, Any]:
    digest = _file_sha256(path)
    cached = _BASELINE_VERIFY_CACHE.get(digest)
    if cached is None:
        cached = load_phase2h_baseline(path)
        _BASELINE_VERIFY_CACHE[digest] = cached
    return cached


def _verify_artifact_manifest(
    directory: Path,
    problems: list[str],
) -> bool:
    ancestor_problems = _symlink_ancestor_problems(directory)
    if ancestor_problems:
        problems.append(
            f"{directory}: artifact path has a symlinked ancestor or root: "
            + "; ".join(ancestor_problems),
        )
        return False
    manifest_path = directory / "MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        problems.append(f"{directory}: MANIFEST.json is missing or a symlink")
        return False
    try:
        manifest = _load_json_strict(manifest_path)
    except (OSError, Phase2ISyntaxError) as error:
        problems.append(f"{directory}: MANIFEST.json is invalid JSON: {error}")
        return False
    if not isinstance(manifest, Mapping):
        problems.append(f"{directory}: MANIFEST must be a JSON object")
        return False
    if set(manifest) != {"files"}:
        problems.append(
            f"{directory}: MANIFEST top-level key set is not canonical",
        )
        return False
    listed_paths: set[str] = set()
    raw_files = manifest.get("files")
    valid = True
    if not isinstance(raw_files, list):
        problems.append(f"{directory}: MANIFEST files must be a list")
        return False
    for entry in raw_files:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "file_sha256"}
        ):
            problems.append(f"{directory}: manifest entry is malformed")
            valid = False
            continue
        relative = entry.get("path")
        expected = entry.get("file_sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or relative.startswith("./")
            or "\\" in relative
            or ".." in Path(relative).parts
            or relative != str(Path(relative))
            or not isinstance(expected, str)
            or not is_sha256_hex(expected)
        ):
            problems.append(
                f"{directory}: manifest entry path/hash is malformed",
            )
            valid = False
            continue
        if relative in listed_paths:
            problems.append(f"{directory}: manifest path {relative} duplicated")
            valid = False
            continue
        listed_paths.add(relative)
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            problems.append(f"{directory}: manifest lists missing {relative}")
            valid = False
            continue
        actual = _file_sha256(path)
        if actual != expected:
            problems.append(
                f"{directory}: {relative} sha256 {actual} != {expected}",
            )
            valid = False
    on_disk = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }
    symlinks = [
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        problems.append(
            f"{directory}: artifact contains symlinks {sorted(symlinks)}",
        )
        valid = False
    expected_files = listed_paths | {"MANIFEST.json"}
    if on_disk != expected_files:
        unlisted = sorted(on_disk - expected_files)
        missing = sorted(expected_files - on_disk)
        problems.append(
            f"{directory}: MANIFEST has "
            + (f"unlisted files {unlisted}; " if unlisted else "")
            + (f"missing files {missing}" if missing else ""),
        )
        valid = False
    expected_manifest_files = {
        "phase2i-syntax-features.json",
        *{f"windows/{wid}.json" for wid in LOCKED_WINDOW_IDS},
        *{f"parser/{wid}.json" for wid in LOCKED_WINDOW_IDS},
    }
    if listed_paths != expected_manifest_files:
        problems.append(
            f"{directory}: MANIFEST file set is not exactly one aggregate "
            "plus five window tables plus five parser tables",
        )
        valid = False
    if isinstance(raw_files, list) and all(
        isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
        for entry in raw_files
    ):
        paths_in_order = [entry["path"] for entry in raw_files]
        if paths_in_order != sorted(paths_in_order):
            problems.append(
                f"{directory}: MANIFEST entries are not in canonical path "
                "order",
            )
            valid = False
    return valid


def _verify_aggregate_header(
    aggregate: Mapping[str, Any],
    directory: Path,
    problems: list[str],
) -> None:
    if set(aggregate) != _REQUIRED_AGGREGATE_KEYS | {"content_sha256"}:
        problems.append(
            f"{directory}: aggregate top-level key set is not the complete "
            "frozen contract",
        )
    inner = {
        key: value for key, value in aggregate.items()
        if key != "content_sha256"
    }
    if aggregate.get("content_sha256") != canonical_sha256(inner):
        problems.append(f"{directory}: aggregate content_sha256 invalid")
    if aggregate.get("run_version") != RUN_VERSION:
        problems.append(f"{directory}: aggregate run_version is not {RUN_VERSION}")
    created_at = aggregate.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        problems.append(f"{directory}: aggregate created_at missing")
    git_commit = aggregate.get("git_commit")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        problems.append(
            f"{directory}: aggregate git_commit is not a Git object ID",
        )
    else:
        try:
            object_type = subprocess.run(
                ["git", "cat-file", "-t", git_commit],
                cwd=DEFAULT_SOURCE_REPOSITORY,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            problems.append(
                f"{directory}: aggregate git_commit is not available in "
                f"the source repository: {error}",
            )
        else:
            if object_type != "commit":
                problems.append(
                    f"{directory}: aggregate git_commit resolves to "
                    f"{object_type!r}, not a commit",
                )
    if aggregate.get("repository_dirty") is not False:
        problems.append(f"{directory}: artifact was not published from a clean tree")

    definition = aggregate.get("definition")
    if not isinstance(definition, Mapping):
        problems.append(f"{directory}: aggregate definition missing")
    else:
        if aggregate.get("definition_sha256") != canonical_sha256(definition):
            problems.append(f"{directory}: aggregate definition_sha256 invalid")
        if not _strict_equal(definition, _definition_c()):
            problems.append(f"{directory}: definition is not the frozen contract")

    input_hashes = aggregate.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or set(
        input_hashes,
    ) != _INPUT_HASH_KEYS:
        problems.append(f"{directory}: input_hashes missing or malformed")
    else:
        if input_hashes.get("benchmark_content_sha256") != (
            BENCHMARK_CONTENT_SHA256
        ):
            problems.append(f"{directory}: benchmark content hash differs")
        if input_hashes.get("phase2h_run1_archive_sha256") != (
            PHASE2H_RUN1_ARCHIVE_SHA256
        ):
            problems.append(f"{directory}: Phase 2H archive hash differs")
        if input_hashes.get("phase2h_run1_aggregate_sha256") != (
            PHASE2H_RUN1_AGGREGATE_SHA256
        ):
            problems.append(f"{directory}: Phase 2H aggregate hash differs")
        if not is_sha256_hex(input_hashes.get("benchmark_file_sha256", "")):
            problems.append(f"{directory}: benchmark file hash malformed")
    if aggregate.get("benchmark_content_sha256") != BENCHMARK_CONTENT_SHA256:
        problems.append(f"{directory}: benchmark_content_sha256 differs")

    dependencies = aggregate.get("dependencies")
    if not isinstance(dependencies, Mapping) or set(
        dependencies,
    ) != _DEPENDENCY_KEYS:
        problems.append(f"{directory}: dependencies missing or malformed")
    elif not _strict_equal(dependencies, _dependency_versions_c()):
        problems.append(
            f"{directory}: dependency versions differ from the verifying "
            "runtime",
        )

    window_tables_lock = aggregate.get("window_tables")
    if not isinstance(window_tables_lock, Mapping) or set(
        window_tables_lock,
    ) != set(LOCKED_WINDOW_IDS):
        problems.append(f"{directory}: window table lock set is not the five windows")
    else:
        for window_id in LOCKED_WINDOW_IDS:
            info = window_tables_lock.get(window_id)
            if not isinstance(info, Mapping) or set(info) != {
                "candidate_table_sha256",
            } or not is_sha256_hex(
                info.get("candidate_table_sha256", ""),
            ):
                problems.append(
                    f"{directory}: window {window_id} table lock malformed",
                )
    parse_hashes_lock = aggregate.get("parse_hashes")
    if not isinstance(parse_hashes_lock, Mapping) or set(
        parse_hashes_lock,
    ) != set(LOCKED_WINDOW_IDS):
        problems.append(f"{directory}: parse hash set is not the five windows")
    else:
        for window_id in LOCKED_WINDOW_IDS:
            if not is_sha256_hex(parse_hashes_lock.get(window_id, "")):
                problems.append(f"{directory}: parse {window_id} hash malformed")

    provenance = aggregate.get("assets_provenance")
    if not isinstance(provenance, Mapping) or set(
        provenance,
    ) != _ASSETS_PROVENANCE_KEYS:
        problems.append(f"{directory}: assets provenance missing or malformed")
    else:
        if provenance.get("verified") is not True or provenance.get("problems"):
            problems.append(f"{directory}: assets provenance is not verified")
        if provenance.get("schema_version") != "phase2i-parser-assets-v1":
            problems.append(f"{directory}: assets schema version differs")
        if provenance.get("manifest_sha256") != LOCKED_ASSETS_MANIFEST_SHA256:
            problems.append(f"{directory}: assets manifest is not the locked value")
        if provenance.get("stanza_version") != STANZA_VERSION:
            problems.append(f"{directory}: assets stanza version differs")
        if provenance.get("package") != STANZA_PACKAGE:
            problems.append(f"{directory}: assets package differs")
        if list(provenance.get("processors") or []) != list(
            STANZA_PROCESSORS,
        ):
            problems.append(f"{directory}: assets processors differ")
        files = provenance.get("files")
        if not isinstance(files, list) or len(files) != 9:
            problems.append(f"{directory}: assets files are not the locked nine")
        else:
            clean_files = []
            seen_assets: set[str] = set()
            for entry in files:
                if (
                    not isinstance(entry, Mapping)
                    or set(entry) != {"path", "sha256"}
                    or not isinstance(entry.get("path"), str)
                    or not is_sha256_hex(entry.get("sha256", ""))
                ):
                    problems.append(f"{directory}: assets file entry malformed")
                    continue
                if entry["path"] in seen_assets:
                    problems.append(
                        f"{directory}: assets file {entry['path']} duplicated",
                    )
                seen_assets.add(entry["path"])
                clean_files.append(
                    {"path": entry["path"], "sha256": entry["sha256"]},
                )
            if canonical_sha256(clean_files) != provenance.get(
                "manifest_sha256",
            ):
                problems.append(
                    f"{directory}: assets file manifest does not self-verify",
                )

    fold_validation = aggregate.get("fold_match_validation")
    if (
        not isinstance(fold_validation, Mapping)
        or set(fold_validation) != _FOLD_MATCH_VALIDATION_KEYS
        or fold_validation.get("validated") is not True
        or fold_validation.get("problems") != []
        or fold_validation.get("compared_against")
        != "archived Phase 2H run-1 folds"
    ):
        problems.append(f"{directory}: fold match validation is not clean")


def _load_artifact_parses(
    directory: Path,
    aggregate: Mapping[str, Any],
    assets_dir: str | Path,
) -> tuple[dict[str, UdParse | None], list[str]]:
    problems: list[str] = []
    parses: dict[str, UdParse | None] = {}
    provenance = verify_assets_provenance(assets_dir)
    if not provenance.get("verified"):
        problems.append(
            f"{directory}: locked parser assets cannot be verified for "
            "Stanza replay: " + "; ".join(
                provenance.get("problems") or ["unknown provenance error"],
            ),
        )
    parse_hashes_lock = aggregate.get("parse_hashes")
    for window_id in LOCKED_WINDOW_IDS:
        path = directory / "parser" / f"{window_id}.json"
        try:
            digest = _file_sha256(path)
            parse_table = _load_json_strict(path)
        except (OSError, Phase2ISyntaxError) as error:
            problems.append(
                f"{directory}: parse {window_id}.json unreadable: {error}",
            )
            parses[window_id] = None
            continue
        if not isinstance(parse_table, Mapping):
            problems.append(f"{directory}: parse {window_id} must be an object")
            parses[window_id] = None
            continue
        expected = (
            parse_hashes_lock.get(window_id)
            if isinstance(parse_hashes_lock, Mapping) else None
        )
        if parse_table.get("parse_sha256") != expected:
            problems.append(
                f"{directory}: parse {window_id} aggregate sha256 differs",
            )
        cached_parse = _PARSE_VERIFY_CACHE.get(digest)
        if cached_parse is None:
            try:
                parse = UdParse.from_dict(parse_table)
            except Phase2ISyntaxError as error:
                problems.append(
                    f"{directory}: parse {window_id} does not canonically "
                    f"self-verify: {error}",
                )
                parses[window_id] = None
                continue
            if path.read_text(encoding="utf-8") != (
                json.dumps(
                    parse.to_dict(), indent=2, ensure_ascii=False,
                ) + "\n"
            ):
                problems.append(
                    f"{directory}: parse {window_id} raw JSON field order "
                    "is not canonical",
                )
            _PARSE_VERIFY_CACHE[digest] = parse
        else:
            parse = cached_parse
        if parse.window_id != window_id:
            problems.append(f"{directory}: parse {window_id} window_id differs")
        if parse.parser != "stanza":
            problems.append(f"{directory}: parse {window_id} is not real stanza")
        if provenance.get("verified"):
            replay_key = canonical_sha256({
                "window_id": window_id,
                "text_sha256": parse.text_sha256,
                "assets_manifest_sha256": provenance.get("manifest_sha256"),
                "dependencies": _dependency_versions_c(),
                "parse_definition": parse_definition(),
            })
            replay = _STANZA_REPLAY_CACHE.get(replay_key)
            if replay is None:
                try:
                    replay = parse_window_text(
                        parse.text,
                        window_id,
                        assets_dir=assets_dir,
                    )
                except Phase2ISyntaxError as error:
                    problems.append(
                        f"{directory}: parse {window_id} locked Stanza "
                        f"replay failed: {error}",
                    )
                else:
                    _STANZA_REPLAY_CACHE[replay_key] = replay
            if replay is not None and not _strict_equal(
                parse.to_dict(), replay.to_dict(),
            ):
                problems.append(
                    f"{directory}: parse {window_id} differs from locked "
                    "Stanza replay",
                )
        parses[window_id] = parse
    return parses, problems


def _load_artifact_tables(
    directory: Path,
    aggregate: Mapping[str, Any],
    parses: Mapping[str, UdParse | None],
) -> tuple[dict[str, dict[str, Any] | None], list[str]]:
    problems: list[str] = []
    tables: dict[str, dict[str, Any] | None] = {}
    window_tables_lock = aggregate.get("window_tables")
    for window_id in LOCKED_WINDOW_IDS:
        path = directory / "windows" / f"{window_id}.json"
        try:
            table = _load_json_strict(path)
        except (OSError, Phase2ISyntaxError) as error:
            problems.append(
                f"{directory}: window {window_id}.json unreadable: {error}",
            )
            tables[window_id] = None
            continue
        if not isinstance(table, Mapping):
            problems.append(f"{directory}: window {window_id} table invalid")
            tables[window_id] = None
            continue
        if set(table) != _WINDOW_TABLE_KEYS:
            problems.append(
                f"{directory}: window {window_id} table key set malformed",
            )
        info = (
            window_tables_lock.get(window_id)
            if isinstance(window_tables_lock, Mapping) else None
        )
        expected = (
            info.get("candidate_table_sha256")
            if isinstance(info, Mapping) else None
        )
        if canonical_sha256(table) != expected:
            problems.append(
                f"{directory}: window {window_id} table hash mismatch",
            )
        if table.get("case_id") != window_id:
            problems.append(f"{directory}: window {window_id} case_id differs")
        parse = parses.get(window_id)
        if parse is not None:
            if table.get("parse_sha256") != parse.parse_sha256:
                problems.append(
                    f"{directory}: window {window_id} parse hash differs",
                )
            if table.get("bronze_text_sha256") != parse.text_sha256:
                problems.append(
                    f"{directory}: window {window_id} Bronze text hash "
                    "differs from parse text hash",
                )
        tables[window_id] = table
    return tables, problems


def _reconstruct_dataset_and_rankings(
    tables: Mapping[str, Mapping[str, Any] | None],
    parses: Mapping[str, UdParse | None],
    directory: Path,
) -> tuple[
    dict[str, Any] | None,
    dict[str, dict[str, dict[str, dict[str, Any]]]] | None,
    dict[str, dict[str, dict[str, dict[str, Any]]]] | None,
    list[str],
]:
    """Reconstruct CandidateRows and rankings from window-table contents."""
    problems: list[str] = []
    dataset: dict[str, Any] = {"windows": {}}
    rankings_c: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    rankings_b: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    seen_candidate_ids: set[str] = set()
    total_keep = 0
    total_drop = 0
    all_oof_c: dict[str, dict[str, dict[str, float]]] = {}
    all_oof_b: dict[str, dict[str, dict[str, float]]] = {}
    for window_id in LOCKED_WINDOW_IDS:
        table = tables.get(window_id)
        parse = parses.get(window_id)
        if table is None or parse is None:
            continue
        candidates = table.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            problems.append(
                f"{directory}: window {window_id} candidates invalid",
            )
            continue
        if table.get("candidate_count") != len(candidates):
            problems.append(
                f"{directory}: window {window_id} candidate count differs",
            )
        rows: list[CandidateRow] = []
        window_valid = True
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                problems.append(
                    f"{directory}: window {window_id} candidate malformed",
                )
                window_valid = False
                continue
            if set(candidate) != _CANDIDATE_KEYS:
                problems.append(
                    f"{directory}: window {window_id} candidate key set "
                    "malformed",
                )
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                problems.append(
                    f"{directory}: window {window_id} candidate id invalid",
                )
                window_valid = False
                continue
            if candidate_id in seen_candidate_ids:
                problems.append(
                    f"{directory}: candidate {candidate_id} duplicated",
                )
            seen_candidate_ids.add(candidate_id)
            source_window_id = table.get("window_id")
            if (
                not isinstance(source_window_id, str)
                or not candidate_id.startswith(source_window_id + ":m")
            ):
                problems.append(
                    f"{directory}: candidate {candidate_id} is not bound to "
                    "its source window",
                )
            if (
                candidate.get("case_id") != window_id
                or candidate.get("window_id") != source_window_id
            ):
                problems.append(
                    f"{directory}: candidate {candidate_id} identity differs",
                )
            if candidate.get("parse_sha256") != parse.parse_sha256:
                problems.append(
                    f"{directory}: candidate {candidate_id} parse hash "
                    "mismatch",
                )
            start = candidate.get("start")
            end = candidate.get("end")
            text = candidate.get("text")
            absolute_start = candidate.get("absolute_start")
            absolute_end = candidate.get("absolute_end")
            if (
                not _is_int(start) or not _is_int(end)
                or not _is_int(absolute_start) or not _is_int(absolute_end)
                or not isinstance(text, str)
                or not 0 <= start < end <= len(parse.text)
                or parse.text[start:end] != text
                or absolute_end - absolute_start != end - start
            ):
                problems.append(
                    f"{directory}: candidate {candidate_id} frozen benchmark "
                    "Bronze slice/offsets mismatch",
                )
            alias = candidate.get("alias")
            segment_ids = candidate.get("segment_ids")
            segment_bounds = candidate.get("segment_bounds")
            type_hints = candidate.get("type_hints")
            if (
                not isinstance(alias, str)
                or not isinstance(segment_ids, list)
                or not segment_ids
                or not all(
                    isinstance(item, str) and item for item in segment_ids
                )
                or not isinstance(segment_bounds, list)
                or not segment_bounds
                or not all(
                    isinstance(item, list) and len(item) == 2
                    and _is_int(item[0]) and _is_int(item[1])
                    for item in segment_bounds
                )
                or not isinstance(type_hints, list)
                or not type_hints
                or not all(isinstance(item, str) for item in type_hints)
            ):
                problems.append(
                    f"{directory}: candidate {candidate_id} provenance "
                    "fields malformed",
                )
            label = candidate.get("label")
            if label == KEEP:
                total_keep += 1
            elif label == DROP:
                total_drop += 1
            else:
                problems.append(
                    f"{directory}: candidate {candidate_id} label invalid",
                )
                window_valid = False
                continue
            if candidate.get("excluded") is not False:
                problems.append(
                    f"{directory}: candidate {candidate_id} is excluded",
                )
            if candidate.get("ambiguity_state") != "NONE":
                problems.append(
                    f"{directory}: candidate {candidate_id} ambiguity state "
                    "differs",
                )
            gold_mention_ids = candidate.get("gold_mention_ids")
            gold_node_types = candidate.get("gold_node_types")
            if (
                not isinstance(gold_mention_ids, list)
                or not all(isinstance(item, str) for item in gold_mention_ids)
                or not isinstance(gold_node_types, list)
                or not all(isinstance(item, str) for item in gold_node_types)
                or bool(gold_mention_ids) != (label == KEEP)
            ):
                problems.append(
                    f"{directory}: candidate {candidate_id} gold metadata "
                    "malformed",
                )
            predictions = candidate.get("predictions")
            if not isinstance(predictions, Mapping) or set(
                predictions,
            ) != set(_ALL_PREDICTION_CELLS):
                problems.append(
                    f"{directory}: candidate {candidate_id} predictions "
                    "malformed",
                )
                window_valid = False
                continue
            valid_predictions = True
            for cell in _ALL_PREDICTION_CELLS:
                prediction = predictions.get(cell)
                if not isinstance(prediction, Mapping) or set(
                    prediction,
                ) != _PREDICTION_KEYS:
                    problems.append(
                        f"{directory}: candidate {candidate_id} {cell} "
                        "prediction malformed",
                    )
                    valid_predictions = False
                    continue
                score = prediction.get("score")
                rank = prediction.get("rank")
                selected = prediction.get("selected")
                error_code = prediction.get("error_code")
                if (
                    not _is_real_number(score)
                    or not 0.0 <= float(score) <= 1.0
                    or not math.isfinite(float(score))
                    or not _is_int(rank)
                    or not 1 <= rank <= len(candidates)
                    or selected not in (KEEP, DROP)
                    or (
                        selected == KEEP
                    ) != (float(score) >= KEEP_THRESHOLD)
                    or (
                        error_code is not None
                        and error_code not in ERROR_CODES
                    )
                ):
                    problems.append(
                        f"{directory}: candidate {candidate_id} {cell} "
                        "score/rank/selection/error contract violated",
                    )
                    valid_predictions = False
            if not valid_predictions:
                window_valid = False
                continue
            rows.append(CandidateRow(
                case_id=candidate["case_id"],
                window_id=candidate["window_id"],
                candidate_id=candidate_id,
                alias=alias,
                start=start,
                end=end,
                absolute_start=absolute_start,
                absolute_end=absolute_end,
                text=text,
                segment_ids=tuple(segment_ids),
                segment_bounds=tuple(
                    (bound[0], bound[1]) for bound in segment_bounds
                ),
                type_hints=tuple(type_hints),
                source_kind="transcript",
                is_gold_positive=label == KEEP,
                label=label,
                excluded=candidate["excluded"],
                ambiguity_state=candidate["ambiguity_state"],
                gold_mention_ids=tuple(gold_mention_ids),
                gold_node_types=tuple(gold_node_types),
            ))
        gold_spans = tuple(sorted({
            (row.start, row.end) for row in rows if row.label == KEEP
        }))
        dataset["windows"][window_id] = {
            "case_id": window_id,
            "window_id": table.get("window_id"),
            "bronze_text": parse.text,
            "bronze_text_sha256": table.get("bronze_text_sha256"),
            "catalog_sha256": table.get("catalog_sha256"),
            "candidate_generator_version": table.get(
                "candidate_generator_version",
            ),
            "gold_spans": gold_spans,
            "rows": tuple(rows),
        }
        if not window_valid or len(rows) != len(candidates):
            continue
        all_oof_c[window_id] = {
            row.candidate_id: {
                cell: float(candidates[index]["predictions"][cell]["score"])
                for cell in CELLS_C
            }
            for index, row in enumerate(rows)
        }
        all_oof_b[window_id] = {
            row.candidate_id: {
                cell: float(candidates[index]["predictions"][cell]["score"])
                for cell in BASELINE_CELLS
            }
            for index, row in enumerate(rows)
        }
    if len(dataset["windows"]) != 5:
        problems.append(f"{directory}: dataset reconstruction is incomplete")
        return None, None, None, problems
    if len(all_oof_c) == 5 and len(all_oof_b) == 5:
        expected_c = compute_rankings(
            dataset, all_oof_c, cells=CELLS_C,
        )
        expected_b = compute_rankings(
            dataset, all_oof_b, cells=BASELINE_CELLS,
        )
        rankings_c = expected_c
        rankings_b = expected_b
        for window_id in LOCKED_WINDOW_IDS:
            table = tables.get(window_id)
            window = dataset["windows"].get(window_id)
            if table is None or window is None:
                continue
            candidates = table.get("candidates")
            rows = window["rows"]
            if (
                not isinstance(candidates, list)
                or len(candidates) != len(rows)
            ):
                continue
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping):
                    continue
                candidate_id = rows[index].candidate_id
                predictions = candidate.get("predictions")
                if not isinstance(predictions, Mapping):
                    continue
                for cell in CELLS_C:
                    stored = predictions.get(cell)
                    expected = expected_c[window_id][cell][candidate_id]
                    stored_ranking = {
                        key: stored.get(key)
                        for key in ("score", "rank", "selected")
                    } if isinstance(stored, Mapping) else stored
                    if not _json_equal(stored_ranking, expected):
                        problems.append(
                            f"{directory}: candidate {candidate_id} {cell} "
                            "rank/score/selection differs from recomputed "
                            "rankings",
                        )
                for cell in BASELINE_CELLS:
                    stored = predictions.get(cell)
                    expected = expected_b[window_id][cell][candidate_id]
                    stored_ranking = {
                        key: stored.get(key)
                        for key in ("score", "rank", "selected")
                    } if isinstance(stored, Mapping) else stored
                    if not _json_equal(stored_ranking, expected):
                        problems.append(
                            f"{directory}: candidate {candidate_id} {cell} "
                            "baseline rank/score/selection differs from "
                            "recomputed rankings",
                        )
    if total_keep != 33 or total_drop != 16591:
        problems.append(
            f"{directory}: label totals {total_keep}/33 KEEP and "
            f"{total_drop}/16591 DROP differ",
        )
    return dataset, rankings_c, rankings_b, problems


def _cross_check_benchmark(
    dataset: Mapping[str, Any],
    tables: Mapping[str, Mapping[str, Any] | None],
    benchmark: Mapping[str, Any],
    directory: Path,
    problems: list[str],
) -> Mapping[str, Any]:
    frozen_dataset = build_dataset(benchmark)
    for window_id in LOCKED_WINDOW_IDS:
        table = tables.get(window_id)
        frozen = frozen_dataset["windows"].get(window_id)
        if table is None or frozen is None:
            continue
        for key in (
            "case_id", "window_id", "bronze_text_sha256", "catalog_sha256",
            "candidate_generator_version",
        ):
            if table.get(key) != frozen.get(key):
                problems.append(
                    f"{directory}: window {window_id} {key} differs from "
                    "the frozen benchmark",
                )
        candidates = table.get("candidates")
        if not isinstance(candidates, list):
            continue
        if len(candidates) != len(frozen["rows"]):
            problems.append(
                f"{directory}: window {window_id} candidate count/order "
                "differs from the frozen benchmark",
            )
            continue
        for index, (candidate, row) in enumerate(
            zip(candidates, frozen["rows"]),
        ):
            if not isinstance(candidate, Mapping):
                continue
            for field in _BASELINE_CANDIDATE_FIELDS:
                if candidate.get(field) != _json_value(getattr(row, field)):
                    problems.append(
                        f"{directory}: candidate {index} "
                        f"({candidate.get('candidate_id')}) {field} "
                        "differs from the frozen benchmark",
                    )
    return frozen_dataset


def _verify_dataset_summary(
    aggregate: Mapping[str, Any],
    dataset: Mapping[str, Any],
    frozen_dataset: Mapping[str, Any] | None,
    problems: list[str],
) -> None:
    summary = aggregate.get("dataset_summary")
    if not isinstance(summary, Mapping):
        problems.append("dataset_summary missing")
        return
    if frozen_dataset is not None:
        expected = validate_dataset(frozen_dataset)
    else:
        per_window: dict[str, dict[str, Any]] = {}
        for window_id in LOCKED_WINDOW_IDS:
            window = dataset["windows"].get(window_id)
            if window is None:
                continue
            rows = window["rows"]
            positive_count = sum(
                1 for row in rows if row.label == KEEP
            )
            per_window[window_id] = {
                "candidate_count": len(rows),
                "positive_count": positive_count,
                "candidate_coverage": _metric(
                    positive_count, positive_count,
                ),
            }
        expected = {
            "window_count": len(dataset["windows"]),
            "candidate_count": sum(
                item["candidate_count"] for item in per_window.values()
            ),
            "positive_count": sum(
                item["positive_count"] for item in per_window.values()
            ),
            "per_window": per_window,
        }
    if not _json_equal(summary, expected):
        problems.append(
            "dataset_summary differs from the frozen/recomputed dataset",
        )


def _verify_immutability_audit(
    aggregate: Mapping[str, Any],
    dataset: Mapping[str, Any],
    frozen_dataset: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    problems: list[str],
) -> None:
    audit = aggregate.get("baseline_immutability_audit")
    if not isinstance(audit, Mapping) or set(audit) != _IMMUTABILITY_KEYS:
        problems.append("baseline immutability audit missing or malformed")
        return
    if frozen_dataset is not None and baseline is not None:
        expected = validate_early_immutability(
            frozen_dataset, baseline["window_tables"],
        )
        if not _json_equal(audit, expected):
            problems.append(
                "baseline immutability audit differs from frozen "
                "recomputation",
            )
        return
    if audit.get("validated") is not True or audit.get("problems"):
        problems.append("baseline immutability audit is not clean")
    if audit.get("problems_sha256") != canonical_sha256(
        audit.get("problems") if isinstance(audit.get("problems"), list)
        else [],
    ):
        problems.append("baseline immutability problems hash invalid")
    if audit.get("window_count") != 5:
        problems.append("baseline immutability window count is not five")
    if audit.get("candidate_count") != 16624:
        problems.append("baseline immutability candidate count is not 16624")
    if audit.get("positive_count") != 33:
        problems.append("baseline immutability positive count is not 33")
    windows_compared = audit.get("windows_compared")
    expected_compared = {
        window_id: {
            "candidate_count": len(dataset["windows"][window_id]["rows"]),
            "problems": [],
        }
        for window_id in LOCKED_WINDOW_IDS
    }
    if not _json_equal(windows_compared, expected_compared):
        problems.append("baseline immutability windows_compared differs")


def _verify_folds(
    aggregate: Mapping[str, Any],
    dataset: Mapping[str, Any],
    archived_folds: Sequence[Mapping[str, Any]] | None,
    problems: list[str],
) -> list[Mapping[str, Any]] | None:
    folds = aggregate.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        problems.append("folds are not the exact five LOO splits")
        return None
    counts = {
        window_id: len(dataset["windows"][window_id]["rows"])
        for window_id in LOCKED_WINDOW_IDS
    }
    positives = {
        window_id: sum(
            1 for row in dataset["windows"][window_id]["rows"]
            if row.label == KEEP
        )
        for window_id in LOCKED_WINDOW_IDS
    }
    usable = True
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or set(fold) != _FOLD_KEYS:
            problems.append(f"fold {index} malformed")
            usable = False
            continue
        test_window = fold.get("test_window_id")
        if not isinstance(test_window, str):
            problems.append(f"fold {index} test window id is not a string")
            usable = False
            continue
        train_windows = fold.get("train_window_ids")
        expected_train = [
            window_id for window_id in LOCKED_WINDOW_IDS
            if window_id != test_window
        ]
        if test_window != LOCKED_WINDOW_IDS[index]:
            problems.append(
                f"fold {index} test window is not the exact archived "
                "sequence",
            )
        if not isinstance(train_windows, list) or train_windows != (
            expected_train
        ):
            problems.append(f"fold {index} train windows differ")
        if fold.get("fold_index") != index:
            problems.append(f"fold {index} fold_index differs")
        train_candidate_count = sum(counts[wid] for wid in expected_train)
        train_positive_count = sum(positives[wid] for wid in expected_train)
        expected_counts = {
            "train_candidate_count": train_candidate_count,
            "train_positive_count": train_positive_count,
            "train_negative_count": (
                train_candidate_count - train_positive_count
            ),
            "test_candidate_count": counts.get(test_window),
            "test_positive_count": positives.get(test_window),
        }
        for key, expected in expected_counts.items():
            if fold.get(key) != expected:
                problems.append(f"fold {index} {key} differs")
        class_weights = fold.get("class_weights")
        if not isinstance(class_weights, Mapping) or set(class_weights) != {
            KEEP, DROP,
        }:
            problems.append(f"fold {index} class weights malformed")
        elif (
            train_positive_count > 0
            and train_candidate_count > train_positive_count
        ):
            expected_weights = {
                KEEP: train_candidate_count / (2 * train_positive_count),
                DROP: train_candidate_count / (
                    2 * (train_candidate_count - train_positive_count)
                ),
            }
            if not _strict_equal(class_weights, expected_weights):
                problems.append(f"fold {index} class weights differ")
    if usable and archived_folds is not None and all(
        isinstance(fold, Mapping) for fold in folds
    ):
        problems.extend(
            validate_cv_folds_match_baseline(folds, archived_folds),
        )
    return folds if usable else None


def _recompute_fit_preprocessing(
    dataset: Mapping[str, Any],
    records_by_window: Mapping[str, Sequence[CandidateSyntax]],
    fold: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Independently recompute fold preprocessing from training windows only.

    Rebuilds the numerical scaler, lexical CountVectorizer vocabulary/config,
    syntax categorical vocabulary/encodings, feature names/hashes, and the
    held-out lexical/syntax OOV audits from the fold's four training windows.
    The training and held-out C matrices, labels, and held-out candidate order
    are also returned so the verifier can refit each frozen cell model and
    reproduce every persisted held-out probability.  The preprocessor is fit
    on training windows only; transforming the held-out window does not leak
    held-out statistics.  Both C cells use identical preprocessing, so this
    is computed once per fold.  Returns ``None`` for malformed inputs so
    verification fails closed instead of raising.
    """
    train_window_ids = fold.get("train_window_ids")
    test_window_id = fold.get("test_window_id")
    fold_index = fold.get("fold_index")
    if (
        not _is_int(fold_index)
        or not isinstance(test_window_id, str)
        or not isinstance(train_window_ids, list)
        or len(train_window_ids) != 4
        or not all(isinstance(item, str) for item in train_window_ids)
    ):
        return None
    windows = dataset.get("windows")
    if not isinstance(windows, Mapping):
        return None
    train_records: list[CandidateSyntax] = []
    for window_id in train_window_ids:
        window = windows.get(window_id)
        records = records_by_window.get(window_id)
        if not isinstance(window, Mapping) or not records:
            return None
        rows = window.get("rows")
        if not isinstance(rows, (list, tuple)) or len(rows) != len(records):
            return None
        train_records.extend(records)
    test_window = windows.get(test_window_id)
    test_records = records_by_window.get(test_window_id)
    if not isinstance(test_window, Mapping) or not test_records:
        return None
    test_rows = test_window.get("rows")
    if not isinstance(test_rows, (list, tuple)) or len(test_rows) != len(
        test_records,
    ):
        return None

    dense_a, dense_b_extra = extract_dense_features(dataset, train_window_ids)
    dense_b = np.hstack([dense_a, dense_b_extra])
    dense_c = np.hstack([dense_b, dense_c_matrix(train_records)])
    texts, boundaries = extract_sparse_inputs(dataset, train_window_ids)
    syntax_train = syntax_groups_from_records(train_records)
    labels_train = np.array([
        1 if row.label == KEEP else 0
        for window_id in train_window_ids
        for row in windows[window_id]["rows"]
    ], dtype=np.int64)
    if (
        len(labels_train) != len(train_records)
        or int(labels_train.sum()) == 0
        or int(len(labels_train) - labels_train.sum()) == 0
    ):
        return None
    preprocessor = CellPreprocessorC().fit(
        dense_c, texts, boundaries, syntax_train,
    )
    scaler = preprocessor.scaler
    encoder = preprocessor.syntax_encoder
    if scaler is None or encoder is None:
        return None
    x_train = preprocessor.transform(
        dense_c, texts, boundaries, syntax_train,
    )
    if x_train.shape[0] != len(labels_train):
        return None
    recomputed_weights = balanced_class_weights(labels_train)
    class_weights = fold.get("class_weights")
    if not isinstance(class_weights, Mapping) or not _strict_equal(
        class_weights,
        {
            KEEP: float(recomputed_weights[1]),
            DROP: float(recomputed_weights[0]),
        },
    ):
        return None
    test_dense_a, test_dense_b_extra = extract_dense_features(
        dataset, [test_window_id],
    )
    test_dense_b = np.hstack([test_dense_a, test_dense_b_extra])
    test_dense_c = np.hstack([
        test_dense_b,
        dense_c_matrix(test_records),
    ])
    test_texts, test_boundaries = extract_sparse_inputs(
        dataset, [test_window_id],
    )
    test_token_types = _held_out_token_types(test_texts)
    test_syntax = syntax_groups_from_records(test_records)
    x_test = preprocessor.transform(
        test_dense_c, test_texts, test_boundaries, test_syntax,
    )
    test_candidate_ids = [row.candidate_id for row in test_rows]
    if (
        x_test.shape[0] != len(test_rows)
        or len(test_candidate_ids) != len(set(test_candidate_ids))
    ):
        return None
    dense_names = (
        list(DENSE_A_FEATURES)
        + list(DENSE_B_EXTRA_FEATURES)
        + list(DENSE_C_EXTRA_FEATURES)
    )
    names = preprocessor.feature_names(dense_names)
    preprocessing = {
        "fold_index": fold_index,
        "train_window_ids": list(train_window_ids),
        "test_window_id": test_window_id,
        "fit_scope": "training windows only",
        "train_candidate_count": fold.get("train_candidate_count"),
        "train_positive_count": fold.get("train_positive_count"),
        "train_negative_count": fold.get("train_negative_count"),
        "class_weights": dict(class_weights),
        "scaler": {
            "fit_on": "training_rows_only",
            "feature_count": int(len(scaler.mean_)),
            "mean_sha256": canonical_sha256([
                float(value) for value in scaler.mean_
            ]),
            "scale_sha256": canonical_sha256([
                float(value) for value in scaler.scale_
            ]),
        },
        "vectorizer": {
            "fit_on": "training_rows_only",
            "params": _json_value(dict(VECTORIZER_PARAMS)),
            "vocabulary_size": len(preprocessor.vocab_terms),
            "vocabulary_sha256": canonical_sha256(
                list(preprocessor.vocab_terms),
            ),
            **_oov_audit(test_token_types, preprocessor.vocab_terms),
        },
        "syntax_encoder": {
            "fit_on": "training_rows_only",
            "params": {
                "max_values_per_group": encoder.max_values_per_group,
            },
            "vocabulary_sizes": {
                group: len(encoder.vocabulary[group])
                for group in SYNTAX_GROUPS
            },
            "vocabulary_sha256": encoder.vocabulary_sha256(),
            **encoder.oov_audit(test_syntax),
        },
        "feature_names_sha256": canonical_sha256(names),
    }
    return {
        "preprocessing": preprocessing,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": labels_train,
        "test_window_id": test_window_id,
        "test_candidate_ids": test_candidate_ids,
        "feature_names": names,
        "class_weights": {
            KEEP: float(recomputed_weights[1]),
            DROP: float(recomputed_weights[0]),
        },
    }


def _validate_train_diagnostics(
    diagnostics: Any,
    *,
    y_train: np.ndarray,
    fold_index: Any,
    cell: str,
) -> list[str]:
    """Validate the types, finiteness, ranges, and label-count invariants of
    one persisted training-overfitting diagnostic record."""
    label = f"fit_scope {cell} fold {fold_index} train diagnostics"
    if not isinstance(diagnostics, Mapping) or set(
        diagnostics,
    ) != _TRAIN_DIAGNOSTICS_KEYS:
        return [f"{label} malformed"]
    problems: list[str] = []
    candidate_count = len(y_train)
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - y_train.sum())
    expected_counts = {
        "candidate_count": candidate_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }
    for key, expected in expected_counts.items():
        value = diagnostics.get(key)
        if not _is_int(value) or value != expected:
            problems.append(
                f"{label} {key} is not the training label count",
            )
    predicted_keep_count = diagnostics.get("predicted_keep_count")
    if (
        not _is_int(predicted_keep_count)
        or not 0 <= predicted_keep_count <= candidate_count
    ):
        problems.append(
            f"{label} predicted_keep_count is out of the legal range",
        )
    for key in sorted(_TRAIN_DIAGNOSTICS_FLOAT_KEYS):
        value = diagnostics.get(key)
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            problems.append(
                f"{label} {key} is not a finite float in [0, 1]",
            )
    return problems


def _fit_recomputed_model_outputs(
    cell: str,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Refit one frozen cell and reproduce train diagnostics and test scores.

    The model sees only the independently reconstructed training matrix and
    labels during fitting.  The held-out matrix is transformed by that same
    train-fitted preprocessor and used only for ``predict_proba``.
    """
    model = make_model_c(cell, training["y_train"])
    model.fit(training["x_train"], training["y_train"])
    held_out_scores = model.predict_proba(training["x_test"])[:, 1]
    if (
        len(held_out_scores) != len(training["test_candidate_ids"])
        or not np.all(np.isfinite(held_out_scores))
        or not np.all((held_out_scores >= 0.0) & (held_out_scores <= 1.0))
    ):
        raise Phase2IError(
            f"{cell} independent refit produced invalid held-out scores",
        )
    return {
        "train_diagnostics": _train_score_diagnostics(
            model, training["x_train"], training["y_train"],
        ),
        "held_out_scores": [float(score) for score in held_out_scores],
        "model": model,
        "feature_names": list(training["feature_names"]),
    }


def _model_refit_outputs_cache_key(
    cell: str,
    training: Mapping[str, Any],
) -> str | None:
    """Content-address a recomputed model/diagnostic result by every semantic
    input used to produce it.

    The key includes both exact sparse matrices, label bytes, held-out window
    and candidate order, cell/frozen configs/seed/dependency versions, and
    validated class weights.  It is intentionally not derived from any
    untrusted artifact hash.  Models themselves are never serialized.
    """
    x_train = training.get("x_train")
    x_test = training.get("x_test")
    y_train = training.get("y_train")
    test_window_id = training.get("test_window_id")
    test_candidate_ids = training.get("test_candidate_ids")
    feature_names = training.get("feature_names")
    if (
        not isinstance(x_train, sp.spmatrix)
        or not isinstance(x_test, sp.spmatrix)
        or not isinstance(y_train, np.ndarray)
        or y_train.dtype != np.int64
        or not isinstance(test_window_id, str)
        or not isinstance(test_candidate_ids, list)
        or not all(isinstance(item, str) for item in test_candidate_ids)
        or not isinstance(feature_names, list)
        or not all(isinstance(item, str) for item in feature_names)
        or x_train.shape[1] != len(feature_names)
        or x_test.shape[1] != len(feature_names)
        or x_test.shape[0] != len(test_candidate_ids)
    ):
        return None
    try:
        def sparse_matrix_sha256(matrix: sp.spmatrix) -> str:
            csr = matrix.tocsr(copy=False)
            matrix_hash = hashlib.sha256()
            matrix_hash.update(csr.indptr.tobytes())
            matrix_hash.update(csr.indices.tobytes())
            matrix_hash.update(csr.data.tobytes())
            matrix_hash.update(
                np.asarray(csr.shape, dtype=np.int64).tobytes(),
            )
            matrix_hash.update(str(csr.dtype).encode("utf-8"))
            return matrix_hash.hexdigest()

        labels_hash = hashlib.sha256(y_train.tobytes()).hexdigest()
        semantic = {
            "cell": cell,
            "keep_threshold": KEEP_THRESHOLD,
            "seed": SEED,
            "x_train_sha256": sparse_matrix_sha256(x_train),
            "x_test_sha256": sparse_matrix_sha256(x_test),
            "y_train_sha256": labels_hash,
            "test_window_id": test_window_id,
            "test_candidate_ids": test_candidate_ids,
            "feature_names": feature_names,
            "class_weights": training.get("class_weights"),
            "logistic_config": _json_value(dict(LOGISTIC_CONFIG)),
            "lightgbm_config": _json_value({
                key: value for key, value in LGBM_CONFIG.items()
                if key != "class_weight"
            }),
            "dependencies": _dependency_versions_c(),
        }
        return canonical_sha256(semantic)
    except (AttributeError, TypeError, ValueError):
        return None


def _cached_fit_recomputed_model_outputs(
    cell: str,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the frozen cell once per complete content-addressed fold input."""
    cache_key = _model_refit_outputs_cache_key(cell, training)
    if cache_key is not None:
        cached = _MODEL_REFIT_OUTPUTS_CACHE.get(cache_key)
        if cached is not None:
            return cached
    outputs = _fit_recomputed_model_outputs(cell, training)
    if cache_key is not None and not _validate_train_diagnostics(
        outputs["train_diagnostics"],
        y_train=training["y_train"],
        fold_index="recomputed",
        cell=cell,
    ):
        _MODEL_REFIT_OUTPUTS_CACHE[cache_key] = outputs
    return outputs


def _expected_fit_scope_record(
    preprocessing: Mapping[str, Any],
    fold: Mapping[str, Any],
    cell: str,
) -> dict[str, Any]:
    """Compose the expected full fit-scope record from shared fold
    preprocessing plus the cell-specific frozen model config snapshot."""
    class_weights = fold.get("class_weights")
    if cell == "logistic_C":
        model_config = {
            "family": "logistic",
            "params": dict(LOGISTIC_CONFIG),
        }
    else:
        model_config = {
            "family": "lightgbm",
            "params": {
                **dict(LGBM_CONFIG),
                "class_weight": (
                    dict(class_weights)
                    if isinstance(class_weights, Mapping) else {}
                ),
            },
        }
    return {
        **preprocessing,
        "model_config": _json_value(model_config),
    }


def _verify_fit_scope(
    aggregate: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]] | None,
    dataset: Mapping[str, Any],
    records_by_window: Mapping[str, Sequence[CandidateSyntax]],
    problems: list[str],
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[int, dict[str, Any] | None],
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[int, tuple[Any, list[str]]]],
]:
    fit_scope = aggregate.get("fit_scope")
    if not isinstance(fit_scope, Mapping) or set(fit_scope) != set(CELLS_C):
        problems.append("fit_scope missing or does not cover both C cells")
        return (
            {cell: {} for cell in CELLS_C},
            {},
            {cell: {} for cell in CELLS_C},
            {cell: {} for cell in CELLS_C},
        )
    folds_by_index: dict[int, Mapping[str, Any]] = {}
    if folds is not None:
        for fold in folds:
            if isinstance(fold, Mapping) and _is_int(fold.get("fold_index")):
                folds_by_index[fold["fold_index"]] = fold
    training_by_fold: dict[int, dict[str, Any] | None] = {}
    verified_train_diagnostics: dict[str, dict[int, dict[str, Any]]] = {
        cell: {} for cell in CELLS_C
    }
    verified_held_out_scores: dict[
        str, dict[str, dict[str, float]]
    ] = {cell: {} for cell in CELLS_C}
    verified_fitted_models: dict[
        str, dict[int, tuple[Any, list[str]]]
    ] = {cell: {} for cell in CELLS_C}
    for cell in CELLS_C:
        scopes = fit_scope.get(cell)
        if not isinstance(scopes, Mapping) or set(scopes) != {
            "0", "1", "2", "3", "4",
        }:
            problems.append(f"fit_scope {cell} is not the five folds")
            continue
        for key, scope in scopes.items():
            if not isinstance(scope, Mapping) or set(
                scope,
            ) != _FIT_SCOPE_RECORD_KEYS:
                problems.append(f"fit_scope {cell} fold {key} malformed")
                continue
            fold_index = scope.get("fold_index")
            if not _is_int(fold_index) or fold_index != int(key):
                problems.append(f"fit_scope {cell} fold {key} index differs")
                continue
            fold = folds_by_index.get(fold_index)
            if fold is not None:
                if (
                    scope.get("train_window_ids")
                    != fold.get("train_window_ids")
                    or scope.get("test_window_id")
                    != fold.get("test_window_id")
                ):
                    problems.append(
                        f"fit_scope {cell} fold {key} windows differ",
                    )
                for count_key in (
                    "train_candidate_count", "train_positive_count",
                    "train_negative_count",
                ):
                    if scope.get(count_key) != fold.get(count_key):
                        problems.append(
                            f"fit_scope {cell} fold {key} {count_key} differs",
                        )
                if not _strict_equal(
                    scope.get("class_weights"), fold.get("class_weights"),
                ):
                    problems.append(
                        f"fit_scope {cell} fold {key} class weights differ",
                    )
            if scope.get("fit_scope") != "training windows only":
                problems.append(
                    f"fit_scope {cell} fold {key} scope is not train-only",
                )
            scaler = scope.get("scaler")
            if not isinstance(scaler, Mapping) or set(scaler) != _SCALER_KEYS:
                problems.append(f"fit_scope {cell} fold {key} scaler malformed")
            elif (
                scaler.get("fit_on") != "training_rows_only"
                or not _is_int(scaler.get("feature_count"))
                or not scaler.get("feature_count") > 0
                or not is_sha256_hex(scaler.get("mean_sha256", ""))
                or not is_sha256_hex(scaler.get("scale_sha256", ""))
            ):
                problems.append(f"fit_scope {cell} fold {key} scaler differs")
            vectorizer = scope.get("vectorizer")
            if not isinstance(vectorizer, Mapping) or set(
                vectorizer,
            ) != _VECTORIZER_KEYS:
                problems.append(
                    f"fit_scope {cell} fold {key} vectorizer malformed",
                )
            elif (
                vectorizer.get("fit_on") != "training_rows_only"
                or not _strict_equal(
                    vectorizer.get("params"),
                    _json_value(dict(VECTORIZER_PARAMS)),
                )
                or not _is_int(vectorizer.get("vocabulary_size"))
                or not is_sha256_hex(
                    vectorizer.get("vocabulary_sha256", ""),
                )
                or not isinstance(
                    vectorizer.get("test_oov_token_types"), list,
                )
                or not _is_int(
                    vectorizer.get("test_oov_token_type_count"),
                )
                or vectorizer.get("test_oov_token_type_count")
                != len(vectorizer.get("test_oov_token_types"))
                or vectorizer.get("test_oov_token_types_sha256")
                != canonical_sha256(
                    vectorizer.get("test_oov_token_types"),
                )
            ):
                problems.append(
                    f"fit_scope {cell} fold {key} vectorizer differs",
                )
            syntax_encoder = scope.get("syntax_encoder")
            if not isinstance(syntax_encoder, Mapping) or set(
                syntax_encoder,
            ) != _SYNTAX_ENCODER_KEYS:
                problems.append(
                    f"fit_scope {cell} fold {key} syntax_encoder malformed",
                )
            elif (
                syntax_encoder.get("fit_on") != "training_rows_only"
                or not _strict_equal(
                    syntax_encoder.get("params"),
                    {"max_values_per_group": MAX_VALUES_PER_GROUP},
                )
                or not isinstance(
                    syntax_encoder.get("vocabulary_sizes"), Mapping,
                )
                or set(syntax_encoder.get("vocabulary_sizes") or {})
                != set(SYNTAX_GROUPS)
                or not all(
                    _is_int(value)
                    for value in (
                        syntax_encoder.get("vocabulary_sizes") or {}
                    ).values()
                )
                or not is_sha256_hex(
                    syntax_encoder.get("vocabulary_sha256", ""),
                )
                or not isinstance(syntax_encoder.get("per_group"), Mapping)
                or set(syntax_encoder.get("per_group") or {})
                != set(SYNTAX_GROUPS)
                or not _is_int(syntax_encoder.get("oov_value_count"))
                or syntax_encoder.get("oov_value_count")
                != sum(
                    len(values) for values in (
                        syntax_encoder.get("per_group") or {}
                    ).values()
                )
                or syntax_encoder.get("oov_sha256") != canonical_sha256(
                    syntax_encoder.get("per_group"),
                )
            ):
                problems.append(
                    f"fit_scope {cell} fold {key} syntax_encoder differs",
                )
            model_config = scope.get("model_config")
            if not isinstance(model_config, Mapping) or set(
                model_config,
            ) != _MODEL_CONFIG_KEYS:
                problems.append(
                    f"fit_scope {cell} fold {key} model_config malformed",
                )
            else:
                family = model_config.get("family")
                params = model_config.get("params")
                expected_family = (
                    "logistic" if cell == "logistic_C" else "lightgbm"
                )
                if family != expected_family:
                    problems.append(
                        f"fit_scope {cell} fold {key} model family differs",
                    )
                if family == "logistic":
                    if not _strict_equal(
                        params, _json_value(dict(LOGISTIC_CONFIG)),
                    ):
                        problems.append(
                            f"fit_scope {cell} fold {key} logistic config "
                            "differs",
                        )
                else:
                    locked_lgbm = {
                        key: value for key, value in LGBM_CONFIG.items()
                        if key != "class_weight"
                    }
                    if not isinstance(params, Mapping) or not _strict_equal(
                        {
                            key: value for key, value in params.items()
                            if key != "class_weight"
                        },
                        locked_lgbm,
                    ):
                        problems.append(
                            f"fit_scope {cell} fold {key} lightgbm config "
                            "differs",
                        )
                    elif fold is not None and not _strict_equal(
                        params.get("class_weight"),
                        fold.get("class_weights"),
                    ):
                        problems.append(
                            f"fit_scope {cell} fold {key} lightgbm class "
                            "weights differ",
                        )
            if not is_sha256_hex(scope.get("feature_names_sha256", "")):
                problems.append(
                    f"fit_scope {cell} fold {key} feature names hash malformed",
                )
            if fold is not None:
                fold_index = fold.get("fold_index")
                if fold_index not in training_by_fold:
                    try:
                        training_by_fold[fold_index] = (
                            _recompute_fit_preprocessing(
                                dataset, records_by_window, fold,
                            )
                        )
                    except Exception as error:
                        problems.append(
                            f"fit_scope {cell} fold {key} preprocessing "
                            f"recomputation failed: {error}",
                        )
                        training_by_fold[fold_index] = None
                training = training_by_fold[fold_index]
                if training is not None:
                    expected = _expected_fit_scope_record(
                        training["preprocessing"], fold, cell,
                    )
                    for field, expected_value in expected.items():
                        if not _strict_equal(
                            scope.get(field), expected_value,
                        ):
                            problems.append(
                                f"fit_scope {cell} fold {key} {field} "
                                "differs from train-only recomputation",
                            )

                train_diagnostics = scope.get("train_diagnostics")
                if training is None:
                    problems.append(
                        f"fit_scope {cell} fold {key} train diagnostics "
                        "cannot be independently verified",
                    )
                    continue
                validation_problems = _validate_train_diagnostics(
                    train_diagnostics,
                    y_train=training["y_train"],
                    fold_index=key,
                    cell=cell,
                )
                problems.extend(validation_problems)
                if validation_problems:
                    continue
                try:
                    recomputed_outputs = (
                        _cached_fit_recomputed_model_outputs(cell, training)
                    )
                except Exception as error:
                    problems.append(
                        f"fit_scope {cell} fold {key} train diagnostics "
                        f"independent refit failed: {error}",
                    )
                    continue
                recomputed_diagnostics = recomputed_outputs[
                    "train_diagnostics"
                ]
                verified_fitted_models[cell][fold_index] = (
                    recomputed_outputs["model"],
                    recomputed_outputs["feature_names"],
                )
                if not _json_equal(train_diagnostics, recomputed_diagnostics):
                    problems.append(
                        f"fit_scope {cell} fold {key} train diagnostics "
                        "differ from the training-only refit",
                    )
                    continue
                verified_train_diagnostics[cell][fold_index] = (
                    recomputed_diagnostics
                )
                verified_held_out_scores[cell][training["test_window_id"]] = {
                    candidate_id: score
                    for candidate_id, score in zip(
                        training["test_candidate_ids"],
                        recomputed_outputs["held_out_scores"],
                    )
                }

    return (
        verified_train_diagnostics,
        training_by_fold,
        verified_held_out_scores,
        verified_fitted_models,
    )


def _verify_held_out_scores(
    rankings_c: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    expected_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    problems: list[str],
) -> None:
    """Compare every persisted C score with a training-only model refit."""
    for cell in CELLS_C:
        cell_scores = expected_scores.get(cell)
        if not isinstance(cell_scores, Mapping) or set(cell_scores) != set(
            LOCKED_WINDOW_IDS,
        ):
            problems.append(
                f"{cell} held-out scores do not cover the five folds",
            )
            continue
        for window_id in LOCKED_WINDOW_IDS:
            window_rankings = rankings_c.get(window_id, {}).get(cell)
            window_scores = cell_scores.get(window_id)
            if (
                not isinstance(window_rankings, Mapping)
                or not isinstance(window_scores, Mapping)
                or set(window_rankings) != set(window_scores)
            ):
                problems.append(
                    f"{cell} held-out scores for {window_id} have a "
                    "candidate-set mismatch",
                )
                continue
            for candidate_id, expected in window_scores.items():
                stored = window_rankings[candidate_id].get("score")
                if (
                    type(stored) is not float
                    or stored != expected
                ):
                    problems.append(
                        f"candidate {candidate_id} {cell} held-out score "
                        "differs from the training-only refit",
                    )


def _recompute_baseline_metrics(
    dataset: Mapping[str, Any],
    rankings_b: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
) -> dict[str, Mapping[str, Any]]:
    window_ids = sorted(dataset["windows"])
    output: dict[str, Mapping[str, Any]] = {}
    for cell in BASELINE_CELLS:
        per_fold = {
            window_id: _window_metrics(
                window_id, dataset["windows"][window_id]["rows"],
                rankings_b, cell,
            )
            for window_id in window_ids
        }
        labels: list[int] = []
        scores: list[float] = []
        for window_id in window_ids:
            for row in dataset["windows"][window_id]["rows"]:
                labels.append(1 if row.label == KEEP else 0)
                scores.append(
                    rankings_b[window_id][cell][row.candidate_id]["score"],
                )
        output[cell] = _pooled_metrics(labels, scores, per_fold, window_ids)
    return output


def _verify_metrics_and_deltas(
    aggregate: Mapping[str, Any],
    dataset: Mapping[str, Any],
    rankings_c: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    rankings_b: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    archive_aggregate: Mapping[str, Any] | None,
    problems: list[str],
) -> dict[str, Mapping[str, Any]] | None:
    metrics = aggregate.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(CELLS_C):
        problems.append("metrics missing or do not cover both C cells")
        return None
    recomputed_c: dict[str, Mapping[str, Any]] = {}
    for cell in CELLS_C:
        expected = compute_cell_metrics_c(dataset, rankings_c, cell)
        stored = metrics.get(cell)
        if not isinstance(stored, Mapping) or not _json_equal(
            stored, expected,
        ):
            problems.append(
                f"metrics {cell} differ from recomputed predictions",
            )
        recomputed_c[cell] = expected
    baseline_metrics = aggregate.get("baseline_metrics")
    if not isinstance(baseline_metrics, Mapping) or set(
        baseline_metrics,
    ) != set(BASELINE_B_CELLS):
        problems.append("baseline_metrics missing or malformed")
        return recomputed_c
    recomputed_b = _recompute_baseline_metrics(dataset, rankings_b)
    for cell in BASELINE_B_CELLS:
        if not _json_equal(baseline_metrics.get(cell), recomputed_b[cell]):
            problems.append(
                f"baseline_metrics {cell} differ from window-table "
                "recomputation",
            )
        if archive_aggregate is not None:
            archived = archive_aggregate.get("metrics", {}).get(cell)
            if not _strict_equal(baseline_metrics.get(cell), archived):
                problems.append(
                    f"baseline_metrics {cell} differ from archived Phase 2H",
                )
    deltas = aggregate.get("deltas")
    if not isinstance(deltas, Mapping) or set(deltas) != set(CELLS_C):
        problems.append("deltas missing or malformed")
        return recomputed_c
    expected_deltas = build_deltas(
        baseline_metrics, recomputed_c, cells=CELLS_C,
    )
    for cell in CELLS_C:
        if not _json_equal(deltas.get(cell), expected_deltas[cell]):
            problems.append(f"deltas {cell} differ from recomputed values")
    return recomputed_c


def _baseline_tables_from_window_tables(
    tables: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for window_id in LOCKED_WINDOW_IDS:
        table = tables.get(window_id)
        candidates = table.get("candidates") if table is not None else []
        if not isinstance(candidates, list):
            output[window_id] = {"candidates": []}
            continue
        output[window_id] = {
            "candidates": [
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "alias": candidate.get("alias"),
                    "text": candidate.get("text"),
                    "label": candidate.get("label"),
                    "predictions": {
                        cell: candidate.get("predictions", {}).get(cell)
                        for cell in BASELINE_CELLS
                    },
                }
                for candidate in candidates
                if isinstance(candidate, Mapping)
            ],
        }
    return output


def _verify_universally_missed(
    aggregate: Mapping[str, Any],
    tables: Mapping[str, Mapping[str, Any] | None],
    problems: list[str],
) -> None:
    baseline_tables = _baseline_tables_from_window_tables(tables)
    derived = derive_universally_missed(baseline_tables)
    problems.extend(validate_universally_missed(derived))
    stored = aggregate.get("universally_missed")
    if not isinstance(stored, list) or not _json_equal(stored, derived):
        problems.append("universally_missed differs from its derivation")
    validation = aggregate.get("universally_missed_validation")
    if (
        not isinstance(validation, Mapping)
        or set(validation) != _UNIVERSALLY_MISSED_VALIDATION_KEYS
        or validation.get("validated") is not True
        or validation.get("problems") != []
        or validation.get("lock_sha256")
        != canonical_sha256(UNIVERSALLY_MISSED_LOCK)
    ):
        problems.append("universally_missed_validation is not clean")


def _training_vs_held_out_from_verified_diagnostics(
    training_by_fold: Mapping[int, Mapping[str, Any] | None],
    verified_train_diagnostics: Mapping[
        str, Mapping[int, Mapping[str, Any]]
    ],
    metrics_c: Mapping[str, Mapping[str, Any]],
    *,
    cells: Sequence[str] = CELLS_C,
) -> dict[str, dict[str, Any]] | None:
    """Build training-vs-held-out diagnostics exclusively from values that
    were independently recomputed during verification.

    The persisted ``fit_scope`` is deliberately not consulted here: training
    counts, OOV counts, and score diagnostics come from the reconstructed
    train-only matrix/model, and held-out metrics come from the recomputed
    candidate rankings.
    """
    output: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(verified_train_diagnostics.get(cell), Mapping):
            return None
        per_fold: dict[str, dict[str, Any]] = {}
        for fold_index in range(5):
            training = training_by_fold.get(fold_index)
            diagnostics = verified_train_diagnostics.get(cell, {}).get(
                fold_index,
            )
            if not isinstance(training, Mapping) or not isinstance(
                diagnostics, Mapping,
            ):
                return None
            preprocessing = training.get("preprocessing")
            if not isinstance(preprocessing, Mapping):
                return None
            test_window_id = preprocessing.get("test_window_id")
            if not isinstance(test_window_id, str):
                return None
            test_metrics = metrics_c[cell]["per_fold"].get(test_window_id)
            if not isinstance(test_metrics, Mapping):
                return None
            vectorizer = preprocessing.get("vectorizer")
            syntax_encoder = preprocessing.get("syntax_encoder")
            if not isinstance(vectorizer, Mapping) or not isinstance(
                syntax_encoder, Mapping,
            ):
                return None
            per_fold[str(fold_index)] = {
                "test_window_id": test_window_id,
                "train": {
                    "candidate_count": diagnostics["candidate_count"],
                    "positive_count": diagnostics["positive_count"],
                    "negative_count": diagnostics["negative_count"],
                    "average_precision": diagnostics["average_precision"],
                    "roc_auc": diagnostics["roc_auc"],
                    "predicted_keep_count": diagnostics[
                        "predicted_keep_count"
                    ],
                },
                "held_out": {
                    "candidate_count": test_metrics["candidate_count"],
                    "positive_count": test_metrics["label_keep_count"],
                    "negative_count": test_metrics["label_drop_count"],
                    "average_precision": test_metrics["average_precision"][
                        "value"
                    ],
                    "roc_auc": test_metrics["roc_auc"]["value"],
                    "predicted_keep_count": test_metrics["selected"],
                },
                "b_token_oov_count": vectorizer[
                    "test_oov_token_type_count"
                ],
                "syntax_oov_count": syntax_encoder["oov_value_count"],
            }
        output[cell] = {"per_fold": per_fold}
    return output


def _verify_training_held_out(
    aggregate: Mapping[str, Any],
    training_by_fold: Mapping[int, Mapping[str, Any] | None],
    verified_train_diagnostics: Mapping[
        str, Mapping[int, Mapping[str, Any]]
    ],
    recomputed_metrics_c: Mapping[str, Mapping[str, Any]] | None,
    problems: list[str],
) -> None:
    stored = aggregate.get("training_vs_held_out")
    if (
        not isinstance(stored, Mapping)
        or recomputed_metrics_c is None
    ):
        problems.append("training_vs_held_out missing or not verifiable")
        return
    if set(stored) != set(CELLS_C):
        problems.append("training_vs_held_out does not cover both C cells")
        return
    try:
        expected = _training_vs_held_out_from_verified_diagnostics(
            training_by_fold,
            verified_train_diagnostics,
            recomputed_metrics_c,
            cells=CELLS_C,
        )
    except Exception as error:
        problems.append(
            f"training_vs_held_out recomputation failed: {error}",
        )
        return
    if expected is None:
        problems.append(
            "training_vs_held_out cannot be recomputed from verified "
            "training diagnostics",
        )
        return
    for cell in CELLS_C:
        if not _json_equal(stored.get(cell), expected.get(cell)):
            problems.append(
                f"training_vs_held_out {cell} differs from recomputed "
                "values",
            )


def _verify_candidate_syntax_and_diagnostics(
    aggregate: Mapping[str, Any],
    dataset: Mapping[str, Any],
    parses: Mapping[str, UdParse | None],
    tables: Mapping[str, Mapping[str, Any] | None],
    rankings_c: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    rankings_b: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    records_by_window: Mapping[str, Sequence[CandidateSyntax]],
    baseline: Mapping[str, Any] | None,
    problems: list[str],
) -> None:
    for window_id in LOCKED_WINDOW_IDS:
        parse = parses.get(window_id)
        table = tables.get(window_id)
        window = dataset["windows"].get(window_id)
        records = records_by_window.get(window_id)
        if parse is None or table is None or window is None:
            continue
        candidates = table.get("candidates")
        rows = window["rows"]
        if not isinstance(candidates, list) or len(candidates) != len(rows):
            continue
        if records is None or len(records) != len(rows):
            problems.append(
                f"candidate syntax record count differs for {window_id}",
            )
            continue
        for row, candidate, record in zip(rows, candidates, records):
            if not isinstance(candidate, Mapping):
                continue
            stored_syntax = candidate.get("syntax")
            if not isinstance(stored_syntax, Mapping) or set(
                stored_syntax,
            ) != _SYNTAX_TABLE_KEYS:
                problems.append(
                    f"candidate {row.candidate_id} syntax key set malformed",
                )
                continue
            if stored_syntax.get("evidence_sha256") != (
                record.evidence_sha256()
            ):
                problems.append(
                    f"candidate {row.candidate_id} syntax evidence does "
                    "not recompute from the parse",
                )
            expected_syntax = _candidate_syntax_projection(parse, record)
            for key in sorted(_SYNTAX_TABLE_KEYS - {"evidence_sha256"}):
                if not _strict_equal(
                    stored_syntax.get(key), expected_syntax.get(key),
                ):
                    problems.append(
                        f"candidate {row.candidate_id} syntax field "
                        f"{key} differs from recomputed evidence",
                    )
    if len(records_by_window) != 5:
        problems.append("candidate syntax recomputation is incomplete")
        return

    expected_parser = parser_error_diagnostics(
        dataset, parses, records_by_window,
    )
    if not _json_equal(aggregate.get("parser_diagnostics"), expected_parser):
        problems.append("parser_diagnostics differ from recomputation")

    expected_overlap = overlap_cluster_syntax_diagnostics(
        dataset, rankings_c, rankings_b, records_by_window, cells=CELLS_C,
    )
    if not _json_equal(
        aggregate.get("overlap_cluster_syntax_diagnostics"),
        expected_overlap,
    ):
        problems.append(
            "overlap_cluster_syntax_diagnostics differ from recomputation",
        )

    errors_c = classify_all_errors_c(
        dataset, rankings_c, parses, records_by_window, cells=CELLS_C,
    )
    errors_b = classify_all_errors(
        dataset, rankings_b, cells=BASELINE_CELLS,
    )
    for window_id in LOCKED_WINDOW_IDS:
        table = tables.get(window_id)
        window = dataset["windows"].get(window_id)
        if table is None or window is None:
            continue
        candidates = table.get("candidates")
        if not isinstance(candidates, list):
            continue
        for row, candidate in zip(window["rows"], candidates):
            if not isinstance(candidate, Mapping):
                continue
            stored_predictions = candidate.get("predictions")
            if not isinstance(stored_predictions, Mapping):
                continue
            for cell in CELLS_C:
                expected_code = errors_c[cell][window_id][
                    row.candidate_id
                ]
                if stored_predictions.get(cell, {}).get(
                    "error_code",
                ) != expected_code:
                    problems.append(
                        f"candidate {row.candidate_id} {cell} error_code "
                        "differs from recomputed taxonomy",
                    )
            for cell in BASELINE_CELLS:
                expected_code = errors_b[cell][window_id][
                    row.candidate_id
                ]
                if stored_predictions.get(cell, {}).get(
                    "error_code",
                ) != expected_code:
                    problems.append(
                        f"candidate {row.candidate_id} {cell} baseline "
                        "error_code differs from recomputed taxonomy",
                    )

    b_taxonomy = error_taxonomy_counts(
        errors_b, dataset, cells=BASELINE_B_CELLS,
    )
    if baseline is not None:
        archived_taxonomy = baseline["aggregate"].get("error_taxonomy")
        archived_b_taxonomy = (
            {
                cell: archived_taxonomy.get(cell)
                for cell in BASELINE_B_CELLS
            }
            if isinstance(archived_taxonomy, Mapping)
            else archived_taxonomy
        )
        if not _strict_equal(archived_b_taxonomy, b_taxonomy):
            problems.append(
                "archived Phase 2H error taxonomy does not match baseline "
                "window-table recomputation",
            )
    expected_taxonomy = error_taxonomy_b_vs_c(
        {"error_taxonomy": b_taxonomy}, errors_c, dataset, cells=CELLS_C,
    )
    if not _json_equal(
        aggregate.get("error_taxonomy_b_vs_c"), expected_taxonomy,
    ):
        problems.append("error_taxonomy_b_vs_c differs from recomputation")

    baseline_tables = _baseline_tables_from_window_tables(tables)
    expected_analysis = universally_missed_analysis(
        baseline_tables, rankings_c, dataset, records_by_window, parses,
        cells=CELLS_C,
    )
    if not _json_equal(
        aggregate.get("universally_missed_analysis"), expected_analysis,
    ):
        problems.append(
            "universally_missed_analysis differs from recomputation",
        )


def _verify_explainability(
    aggregate: Mapping[str, Any],
    problems: list[str],
    *,
    expected_coefficients: Mapping[str, Any] | None = None,
    expected_importance: Mapping[str, Any] | None = None,
) -> None:
    coefficients = aggregate.get("syntax_coefficients")
    if not isinstance(coefficients, Mapping) or set(coefficients) != {
        "logistic_C",
    }:
        problems.append("syntax_coefficients missing or malformed")
    else:
        entry = coefficients.get("logistic_C")
        if not isinstance(entry, Mapping) or set(entry) != {
            "kind", "per_fold", "aggregate_top_positive",
            "aggregate_top_negative", "syntax_feature_count",
        }:
            problems.append("syntax_coefficients entry malformed")
        else:
            if entry.get("kind") != "logistic_syntax_coefficients":
                problems.append("syntax_coefficients kind differs")
            per_fold = entry.get("per_fold")
            if not isinstance(per_fold, Mapping) or set(per_fold) != {
                "0", "1", "2", "3", "4",
            }:
                problems.append("syntax_coefficients per_fold malformed")
            else:
                positive: list[tuple[str, float]] = []
                negative: list[tuple[str, float]] = []
                names: set[str] = set()
                for fold, coeffs in per_fold.items():
                    if not isinstance(coeffs, Mapping):
                        problems.append(
                            f"syntax_coefficients fold {fold} malformed",
                        )
                        continue
                    for name, value in coeffs.items():
                        if (
                            not isinstance(name, str)
                            or not _is_real_number(value)
                        ):
                            problems.append(
                                f"syntax_coefficients fold {fold} entry "
                                "malformed",
                            )
                            continue
                        names.add(name)
                        if float(value) >= 0:
                            positive.append((name, float(value)))
                        else:
                            negative.append((name, float(value)))
                if not _json_equal(
                    entry.get("aggregate_top_positive"),
                    _aggregate_top(positive, 15),
                ):
                    problems.append(
                        "syntax_coefficients aggregate positive differs",
                    )
                if not _json_equal(
                    entry.get("aggregate_top_negative"),
                    _aggregate_top(negative, 15, ascending=True),
                ):
                    problems.append(
                        "syntax_coefficients aggregate negative differs",
                    )
                if entry.get("syntax_feature_count") != len(names):
                    problems.append(
                        "syntax_coefficients feature count differs",
                    )
    if expected_coefficients is not None and not _json_equal(
        coefficients, expected_coefficients,
    ):
        problems.append(
            "syntax_coefficients differ from the training-only model refits",
        )

    importance = aggregate.get("syntax_vs_inherited_importance")
    if not isinstance(importance, Mapping) or set(importance) != {
        "lightgbm_C",
    }:
        problems.append(
            "syntax_vs_inherited_importance missing or malformed",
        )
    else:
        entry = importance.get("lightgbm_C")
        if not isinstance(entry, Mapping) or set(entry) != {
            "kind", "per_fold", "aggregate_syntax_top",
            "aggregate_inherited_top",
        }:
            problems.append("syntax_vs_inherited_importance malformed")
        else:
            if entry.get("kind") != (
                "lightgbm_gain_importance_syntax_vs_inherited"
            ):
                problems.append("syntax_vs_inherited_importance kind differs")
            per_fold = entry.get("per_fold")
            if not isinstance(per_fold, Mapping) or set(per_fold) != {
                "0", "1", "2", "3", "4",
            }:
                problems.append(
                    "syntax_vs_inherited_importance per_fold malformed",
                )
            else:
                syntax_items: list[tuple[str, float]] = []
                inherited_items: list[tuple[str, float]] = []
                for fold, fold_data in per_fold.items():
                    if not isinstance(fold_data, Mapping) or set(
                        fold_data,
                    ) != {
                        "syntax_gain", "inherited_gain", "syntax_share",
                        "importances", "top_importances",
                    }:
                        problems.append(
                            "syntax_vs_inherited_importance fold "
                            f"{fold} malformed",
                        )
                        continue
                    importances = fold_data.get("importances")
                    top = fold_data.get("top_importances")
                    if not isinstance(importances, list) or not isinstance(
                        top, list,
                    ):
                        problems.append(
                            f"syntax_vs_inherited_importance fold {fold} "
                            "importances/top_importances malformed",
                        )
                        continue
                    valid_importances = True
                    previous: float | None = None
                    for item in importances:
                        if (
                            not isinstance(item, list)
                            or len(item) != 2
                            or not isinstance(item[0], str)
                            or not _is_real_number(item[1])
                            or not float(item[1]) > 0
                        ):
                            problems.append(
                                f"syntax_vs_inherited_importance fold "
                                f"{fold} importance entry malformed",
                            )
                            valid_importances = False
                            continue
                        if previous is not None and float(item[1]) > (
                            previous + 1e-12
                        ):
                            problems.append(
                                f"syntax_vs_inherited_importance fold "
                                f"{fold} importance ordering invalid",
                            )
                            valid_importances = False
                        previous = float(item[1])
                    if not valid_importances:
                        continue
                    if (
                        len(top) != min(20, len(importances))
                        or top != importances[:len(top)]
                    ):
                        problems.append(
                            f"syntax_vs_inherited_importance fold {fold} "
                            "top_importances is not the complete-list prefix",
                        )
                    recomputed_syntax_gain = 0.0
                    recomputed_inherited_gain = 0.0
                    for name, value in (
                        (item[0], float(item[1])) for item in importances
                    ):
                        if (
                            name.startswith("syntax:")
                            or name in DENSE_C_EXTRA_FEATURES
                        ):
                            recomputed_syntax_gain += value
                        else:
                            recomputed_inherited_gain += value
                    syntax_gain = fold_data.get("syntax_gain")
                    inherited_gain = fold_data.get("inherited_gain")
                    if (
                        not _is_real_number(syntax_gain)
                        or not _is_real_number(inherited_gain)
                        or float(syntax_gain) < 0
                        or float(inherited_gain) < 0
                        or not math.isclose(
                            float(syntax_gain), recomputed_syntax_gain,
                            rel_tol=1e-9, abs_tol=1e-12,
                        )
                        or not math.isclose(
                            float(inherited_gain),
                            recomputed_inherited_gain,
                            rel_tol=1e-9, abs_tol=1e-12,
                        )
                    ):
                        problems.append(
                            "syntax_vs_inherited_importance fold "
                            f"{fold} gains malformed",
                        )
                        continue
                    total = (
                        recomputed_syntax_gain + recomputed_inherited_gain
                    )
                    share = fold_data.get("syntax_share")
                    if total > 0:
                        if not _is_real_number(share) or not math.isclose(
                            float(share), recomputed_syntax_gain / total,
                            rel_tol=1e-9, abs_tol=1e-12,
                        ):
                            problems.append(
                                "syntax_vs_inherited_importance fold "
                                f"{fold} share differs",
                            )
                    elif share is not None:
                        problems.append(
                            "syntax_vs_inherited_importance fold "
                            f"{fold} share should be null",
                        )
                    for item in importances:
                        if (
                            item[0].startswith("syntax:")
                            or item[0] in DENSE_C_EXTRA_FEATURES
                        ):
                            syntax_items.append(
                                (item[0], float(item[1])),
                            )
                        else:
                            inherited_items.append(
                                (item[0], float(item[1])),
                            )
                if not _json_equal(
                    entry.get("aggregate_syntax_top"),
                    _aggregate_top(syntax_items, 15),
                ):
                    problems.append(
                        "syntax_vs_inherited_importance aggregate syntax "
                        "top differs",
                    )
                if not _json_equal(
                    entry.get("aggregate_inherited_top"),
                    _aggregate_top(inherited_items, 10),
                ):
                    problems.append(
                        "syntax_vs_inherited_importance aggregate inherited "
                        "top differs",
                    )
    if expected_importance is not None and not _json_equal(
        importance, expected_importance,
    ):
        problems.append(
            "syntax_vs_inherited_importance differs from the training-only "
            "model refits",
        )


def _recompute_artifact_candidate_syntax(
    dataset: Mapping[str, Any],
    parses: Mapping[str, UdParse | None],
    directory: Path,
    problems: list[str],
) -> dict[str, list[CandidateSyntax]]:
    """Recompute candidate syntax evidence once per window.

    The same deterministic pass feeds both fit-scope reconstruction and the
    per-candidate persisted-syntax comparison, avoiding duplicate expensive
    recomputation.  A window whose parse cannot recompute its candidate
    evidence fails closed with a problem and is omitted from the result.
    """
    records_by_window: dict[str, list[CandidateSyntax]] = {}
    for window_id in LOCKED_WINDOW_IDS:
        parse = parses.get(window_id)
        window = dataset["windows"].get(window_id)
        if parse is None or window is None:
            continue
        rows = window["rows"]
        try:
            records_by_window[window_id] = [
                compute_candidate_syntax(parse, row) for row in rows
            ]
        except Phase2ISyntaxError as error:
            problems.append(
                f"{directory}: window {window_id} candidate syntax "
                f"recomputation failed: {error}",
            )
    return records_by_window


def _verify_phase2i_artifact_body(
    directory: Path,
    *,
    benchmark_path: str | Path | None,
    baseline_archive: str | Path | None,
    assets_dir: str | Path,
) -> list[str]:
    problems: list[str] = []
    manifest_ok = _verify_artifact_manifest(directory, problems)
    aggregate_path = directory / "phase2i-syntax-features.json"
    aggregate: Mapping[str, Any] | None = None
    if aggregate_path.is_symlink() or not aggregate_path.is_file():
        problems.append(f"{directory}: aggregate JSON is missing or a symlink")
    else:
        try:
            raw = _load_json_strict(aggregate_path)
        except (OSError, Phase2ISyntaxError) as error:
            problems.append(f"{directory}: aggregate JSON is invalid: {error}")
        else:
            if not isinstance(raw, Mapping):
                problems.append(f"{directory}: aggregate must be a JSON object")
            else:
                aggregate = raw
    if aggregate is None or not manifest_ok:
        return problems

    _verify_aggregate_header(aggregate, directory, problems)

    parses, parse_problems = _load_artifact_parses(
        directory, aggregate, assets_dir,
    )
    problems.extend(parse_problems)
    tables, table_problems = _load_artifact_tables(
        directory, aggregate, parses,
    )
    problems.extend(table_problems)
    if any(parses.get(wid) is None for wid in LOCKED_WINDOW_IDS) or any(
        tables.get(wid) is None for wid in LOCKED_WINDOW_IDS
    ):
        return problems

    # Cross-check the frozen Phase 2H predictions before reconstructing
    # rankings.  A tampered baseline score can invalidate many ranks at once;
    # reporting the direct archive mismatch first keeps the root cause visible
    # even when reconstruction must subsequently fail closed.
    baseline = None
    archive_aggregate = None
    archived_folds = None
    if baseline_archive is not None:
        if aggregate.get("input_hashes", {}).get(
            "phase2h_run1_archive_sha256",
        ) != _file_sha256(Path(baseline_archive)):
            problems.append(
                f"{directory}: baseline archive hash does not match the "
                "artifact lock",
            )
        baseline = _cached_baseline(Path(baseline_archive))
        archive_aggregate = baseline["aggregate"]
        archived_folds = archive_aggregate.get("folds")
        archived_tables = baseline["window_tables"]
        for window_id in LOCKED_WINDOW_IDS:
            table = tables.get(window_id)
            archived = archived_tables.get(window_id)
            if table is None or archived is None:
                continue
            stored_candidates = table.get("candidates")
            archived_candidates = archived.get("candidates")
            if (
                not isinstance(stored_candidates, list)
                or not isinstance(archived_candidates, list)
            ):
                continue
            if len(stored_candidates) != len(archived_candidates):
                problems.append(
                    f"{directory}: window {window_id} baseline candidate "
                    "count differs from the archive",
                )
                continue
            for index, (stored, archived) in enumerate(zip(
                stored_candidates, archived_candidates,
            )):
                if (
                    not isinstance(stored, Mapping)
                    or not isinstance(archived, Mapping)
                ):
                    continue
                for cell in BASELINE_CELLS:
                    stored_prediction = stored.get("predictions", {}).get(
                        cell,
                    )
                    archived_prediction = archived.get(
                        "predictions", {},
                    ).get(cell)
                    if not _strict_equal(
                        stored_prediction, archived_prediction,
                    ):
                        problems.append(
                            f"{directory}: candidate {index} "
                            f"({stored.get('candidate_id')}) {cell} baseline "
                            "predictions differ from the archive",
                        )

    dataset, rankings_c, rankings_b, row_problems = (
        _reconstruct_dataset_and_rankings(tables, parses, directory)
    )
    problems.extend(row_problems)
    if (
        row_problems
        or dataset is None
        or rankings_c is None
        or rankings_b is None
    ):
        return problems

    benchmark = None
    frozen_dataset = None
    if benchmark_path is not None:
        benchmark = _cached_benchmark_mapping(Path(benchmark_path))
        if aggregate.get("input_hashes", {}).get(
            "benchmark_file_sha256",
        ) != _file_sha256(Path(benchmark_path)):
            problems.append(
                f"{directory}: benchmark file hash does not match the "
                "artifact lock",
            )
        frozen_dataset = _cross_check_benchmark(
            dataset, tables, benchmark, directory, problems,
        )

    _verify_dataset_summary(aggregate, dataset, frozen_dataset, problems)
    _verify_immutability_audit(
        aggregate, dataset, frozen_dataset, baseline, problems,
    )
    folds = _verify_folds(aggregate, dataset, archived_folds, problems)
    records_by_window: dict[str, list[CandidateSyntax]] = {}
    verified_train_diagnostics: dict[
        str, dict[int, dict[str, Any]]
    ] = {cell: {} for cell in CELLS_C}
    training_by_fold: dict[int, dict[str, Any] | None] = {}
    verified_held_out_scores: dict[
        str, dict[str, dict[str, float]]
    ] = {cell: {} for cell in CELLS_C}
    verified_fitted_models: dict[
        str, dict[int, tuple[Any, list[str]]]
    ] = {cell: {} for cell in CELLS_C}
    if not problems:
        records_by_window = _recompute_artifact_candidate_syntax(
            dataset, parses, directory, problems,
        )
        (
            verified_train_diagnostics,
            training_by_fold,
            verified_held_out_scores,
            verified_fitted_models,
        ) = _verify_fit_scope(
            aggregate, folds, dataset, records_by_window, problems,
        )
        _verify_held_out_scores(
            rankings_c, verified_held_out_scores, problems,
        )
    expected_coefficients = syntax_logistic_coefficients(
        verified_fitted_models, cells=("logistic_C",),
    )
    expected_importance = syntax_vs_inherited_importance(
        verified_fitted_models, cells=("lightgbm_C",),
    )
    _verify_explainability(
        aggregate,
        problems,
        expected_coefficients=expected_coefficients,
        expected_importance=expected_importance,
    )
    recomputed_metrics_c = _verify_metrics_and_deltas(
        aggregate, dataset, rankings_c, rankings_b, archive_aggregate,
        problems,
    )
    _verify_universally_missed(aggregate, tables, problems)
    _verify_training_held_out(
        aggregate, training_by_fold, verified_train_diagnostics,
        recomputed_metrics_c, problems,
    )
    if not problems:
        _verify_candidate_syntax_and_diagnostics(
            aggregate, dataset, parses, tables, rankings_c, rankings_b,
            records_by_window, baseline, problems,
        )
    return problems


def _verify_phase2i_artifact(
    directory: Path,
    *,
    benchmark_path: str | Path | None = None,
    baseline_archive: str | Path | None = None,
    assets_dir: str | Path = DEFAULT_PARSER_ASSETS,
) -> list[str]:
    """Fail-closed verification of the complete Phase 2I artifact contract.

    The verifier is deterministic, never raises on malformed untrusted
    artifact structures, and recomputes the derivable semantics -- threshold
    selections, ranks, per-window/pooled/ranking metrics, confusion and
    selected counts, B-vs-C deltas, the seven universally-missed endpoints,
    error taxonomy, overlap/parser diagnostics, explainability summaries,
    training-vs-held-out sections, every persisted candidate syntax field
    (recursively, against independently recomputed parse evidence), and the
    complete train-only fit scope (scaler, lexical vectorizer vocabulary/
    config, syntax categorical vocabulary/encodings, feature names/hashes,
    and held-out OOV audits) -- from the per-candidate predictions, parse
    tables, and frozen candidate rows.  When ``benchmark_path``/
    ``baseline_archive`` are supplied it additionally cross-checks candidate
    IDs/order, exact offsets, catalog/generator metadata, gold metadata, and
    every baseline prediction/fold/metric against the frozen files.
    """
    directory = Path(directory)
    problems: list[str] = []
    try:
        problems.extend(_verify_phase2i_artifact_body(
            directory,
            benchmark_path=benchmark_path,
            baseline_archive=baseline_archive,
            assets_dir=assets_dir,
        ))
    except Exception as error:
        problems.append(
            f"{directory}: verifier crashed on untrusted artifact "
            f"structure ({type(error).__name__}: {error})",
        )
    return problems


def compare_phase2i_artifacts(
    left: Path,
    right: Path,
    *,
    benchmark_path: str | Path,
    baseline_archive: str | Path,
    assets_dir: str | Path = DEFAULT_PARSER_ASSETS,
) -> list[str]:
    """Compare two clean Phase 2I reruns.

    Both artifacts are first fully verified by the strict acceptance
    verifier (including recomputed semantics and frozen benchmark/baseline
    cross-checks); only after both pass are deterministic contents compared.
    ``created_at`` (and the dependent ``content_sha256``) are excluded by
    design because each run records its own UTC timestamp.
    """
    differences: list[str] = []
    for label, path in (("left", left), ("right", right)):
        problems = _verify_phase2i_artifact(
            Path(path),
            benchmark_path=benchmark_path,
            baseline_archive=baseline_archive,
            assets_dir=assets_dir,
        )
        if problems:
            differences.append(
                f"{label} artifact failed strict acceptance verification",
            )
            differences.extend(f"  - {problem}" for problem in problems)
    if differences:
        return differences
    left_body = _load_json_strict(
        Path(left) / "phase2i-syntax-features.json",
    )
    right_body = _load_json_strict(
        Path(right) / "phase2i-syntax-features.json",
    )
    left_inner = {
        key: value for key, value in left_body.items()
        if key not in {"created_at", "content_sha256"}
    }
    right_inner = {
        key: value for key, value in right_body.items()
        if key not in {"created_at", "content_sha256"}
    }
    for key in sorted(set(left_inner) | set(right_inner)):
        if left_inner.get(key) != right_inner.get(key):
            differences.append(f"{key} differs")
    window_ids = sorted(LOCKED_WINDOW_IDS)
    for window_id in window_ids:
        left_path = Path(left) / "windows" / f"{window_id}.json"
        right_path = Path(right) / "windows" / f"{window_id}.json"
        if not left_path.is_file() or not right_path.is_file():
            differences.append(f"window table {window_id} missing on a side")
            continue
        if hashlib.sha256(left_path.read_bytes()).hexdigest() != (
            hashlib.sha256(right_path.read_bytes()).hexdigest()
        ):
            differences.append(
                f"window table {window_id} file content differs",
            )
    for window_id in window_ids:
        left_path = Path(left) / "parser" / f"{window_id}.json"
        right_path = Path(right) / "parser" / f"{window_id}.json"
        if not left_path.is_file() or not right_path.is_file():
            differences.append(f"parse {window_id} missing on a side")
            continue
        if hashlib.sha256(left_path.read_bytes()).hexdigest() != (
            hashlib.sha256(right_path.read_bytes()).hexdigest()
        ):
            differences.append(
                f"parse {window_id} file content differs",
            )
    try:
        left_manifest = _load_json_strict(Path(left) / "MANIFEST.json")
        right_manifest = _load_json_strict(Path(right) / "MANIFEST.json")
        for manifest in (left_manifest, right_manifest):
            for entry in manifest["files"]:
                if entry["path"] == "phase2i-syntax-features.json":
                    entry["file_sha256"] = "<created_at-dependent>"
    except (OSError, Phase2ISyntaxError, KeyError, TypeError) as error:
        differences.append(f"manifest comparison failed: {error}")
    else:
        if not _strict_equal(left_manifest, right_manifest):
            differences.append("canonical MANIFEST structure differs")
    return differences


__all__ = [
    "BASELINE_B_CELLS",
    "BASELINE_CELLS",
    "CELLS_C",
    "LOCKED_WINDOW_IDS",
    "MODEL_NAMES_C",
    "PHASE2H_RUN1_AGGREGATE_SHA256",
    "PHASE2H_RUN1_ARCHIVE_SHA256",
    "PIPELINE_VERSION",
    "RUN_VERSION",
    "UNIVERSALLY_MISSED_LOCK",
    "CellPreprocessorC",
    "Phase2IBaselineError",
    "Phase2IError",
    "baseline_rankings_from_tables",
    "build_aggregate_c",
    "build_candidate_syntax",
    "build_deltas",
    "build_phase2i_window_table",
    "classify_all_errors_c",
    "close_phase2h_baseline",
    "compare_phase2i_artifacts",
    "compute_cell_metrics_c",
    "derive_universally_missed",
    "error_taxonomy_b_vs_c",
    "load_phase2h_baseline",
    "make_model_c",
    "overlap_cluster_syntax_diagnostics",
    "parser_error_diagnostics",
    "publish_phase2i_artifact",
    "run_cv_c",
    "run_experiment_c",
    "syntax_logistic_coefficients",
    "syntax_vs_inherited_importance",
    "training_vs_held_out_diagnostics",
    "universally_missed_analysis",
    "validate_universally_missed",
    "validate_early_immutability",
    "validate_cv_folds_match_baseline",
    "verify_assets_provenance",
]
