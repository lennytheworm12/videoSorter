"""Phase 2H discriminative semantic endpoint scoring (offline, deterministic).

Phase 2H is a candidate-level binary KEEP/DROP endpoint scorer over the exact
exhaustive source-grounded candidate universe produced by the frozen Phase 2F
candidate generator (``pipeline.semantic_mentions.generate_mention_candidates``).
It never calls an LLM/API, never constructs semantic edges or downstream
graphs, and never changes Phase 2F candidate generation or the locked Phase 2G
gold labels.

The experiment compares four fixed cells on identical grouped
leave-one-window-out folds across the locked five-case Phase 2F/2G benchmark:

* ``logistic_A``  -- class-weighted L2 logistic regression on geometry/provenance features;
* ``logistic_B``  -- logistic regression on A plus bounded lexical/cue features;
* ``lightgbm_A``  -- conservative deterministic LightGBM on A;
* ``lightgbm_B``  -- LightGBM on A plus bounded lexical/cue features.

Every preprocessing statistic (scaler, sparse vocabulary, class weights, model
fit) is computed from training windows only, and the fixed preregistered KEEP
threshold is 0.5.  The B-cell fit-scope audit reports the actual held-out
token types absent from the fitted vocabulary (count, sorted list, and a
deterministic SHA-256) so the reported fit scope is truthful.  No
source/window/case IDs, question text, mention IDs, gold roles/types, or
label-derived values ever enter the predictive features; they are retained
only as provenance.  Optional UD-syntax integration is explicitly out of scope
for this first implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import lightgbm
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.phase2g_endpoint_recovery import (
    BENCHMARK_CONTENT_SHA256,
    build_case_experiment,
    load_benchmark,
    validate_experiment_coverage,
)
from pipeline.phase2g_silver import canonical_sha256
from pipeline.semantic_mentions import (
    MENTION_CATALOG_VERSION,
    NODE_TYPES,
    _ABILITY_RESOURCES,
    _ACTIONS,
    _CONDITION_MARKERS,
    _EVENTS,
    _MODALS,
    _NEGATIONS,
    _PRONOUNS,
    _TIME_MARKERS,
    generate_mention_candidates,
)
from pipeline.semantic_source import BronzeSource, window_from_exact_span


RUN_VERSION = "phase2h-endpoint-scoring-v1"
FEATURE_SCHEMA_VERSION = "phase2h-features-v1"
KEEP_THRESHOLD = 0.5
SEED = 20260817
GOLD_RANK_HIGH_K = 5
RANK_KS = (1, 3, 5, 10)
MODEL_NAMES = ("logistic", "lightgbm")
FEATURE_SETS = ("A", "B")
CELLS = tuple(
    f"{model}_{feature_set}"
    for model in MODEL_NAMES
    for feature_set in FEATURE_SETS
)

KEEP = "KEEP"
DROP = "DROP"

ERROR_CODES = frozenset({
    "OVERLAPPING_LONGER_SPAN",
    "OVERLAPPING_SHORTER_FRAGMENT",
    "PRONOUN_DISTRACTOR",
    "GENERIC_ACTION_DISTRACTOR",
    "GENERIC_ENTITY_DISTRACTOR",
    "DISCOURSE_FILLER",
    "WRONG_CUE_PRIOR",
    "SOURCE_POSITION_BIAS",
    "PARSER_FEATURE_ERROR",
    "GOLD_RANKED_HIGH_THRESHOLD_MISS",
    "GOLD_RANKED_LOW",
    "OTHER",
})

# Documented first-failure precedence for false positives; each false positive
# receives exactly one classification.
ERROR_PRECEDENCE = (
    "PARSER_FEATURE_ERROR",
    "OVERLAPPING_LONGER_SPAN",
    "OVERLAPPING_SHORTER_FRAGMENT",
    "PRONOUN_DISTRACTOR",
    "DISCOURSE_FILLER",
    "GENERIC_ACTION_DISTRACTOR",
    "GENERIC_ENTITY_DISTRACTOR",
    "WRONG_CUE_PRIOR",
    "SOURCE_POSITION_BIAS",
    "OTHER",
)

_WORD = re.compile(r"[^\W_][\w'’%-]*", re.UNICODE)
_PUNCTUATION_AFTER = frozenset(".!?;:)]\"'’")
_PUNCTUATION_BEFORE = frozenset(".!?;:,)]\"'’")

# Bounded lexical cue sets used by feature set B.  The pronoun, ability-alias,
# action, event, time, condition, negation, and modal lists are the exact
# frozen cue lists owned by the Phase 2F candidate generator
# (``pipeline.semantic_mentions``) so feature evidence stays aligned with
# generator type hints.  The remaining sets are Phase 2H-local deterministic
# bounded lists.
PRONOUN_TOKENS = frozenset(_PRONOUNS)
ABILITY_ALIAS_TOKENS = frozenset(_ABILITY_RESOURCES)
ACTION_TOKENS = frozenset(_ACTIONS)
EVENT_TOKENS = frozenset(_EVENTS)
TEMPORAL_TOKENS = frozenset(_TIME_MARKERS)
CONDITIONAL_TOKENS = frozenset(_CONDITION_MARKERS)
MODAL_TOKENS = frozenset(_MODALS)
NEGATION_TOKENS = frozenset(_NEGATIONS)
CAUSAL_TOKENS = frozenset(
    "because so thus therefore since hence consequently as".split(),
)
FILLER_TOKENS = frozenset(
    "like um uh yeah yep right okay ok well actually basically literally "
    "kind sort maybe perhaps just really very".split(),
)
DISFLUENCY_BIGRAMS = frozenset({
    "you know", "i mean", "kind of", "sort of", "i think", "i guess",
})
FUNCTION_WORDS = frozenset(
    "the a an and or but nor for so yet if unless while when where which who "
    "whom whose that this these those there here it its they them their theirs "
    "i me my mine we us our ours you your yours he him his she her hers "
    "to of in on at by with from into onto upon about against between through "
    "during before after above below under over is are was were be been being "
    "am do does did done have has had having will would shall should can could "
    "may might must not no n't never neither nor without like um uh yeah right "
    "okay ok well actually basically literally kind sort maybe perhaps just "
    "really very".split(),
)

DENSE_A_FEATURES = (
    "char_len",
    "token_count",
    "normalized_start",
    "normalized_end",
    "normalized_length",
    "segment_count",
    "starts_at_segment_boundary",
    "ends_at_segment_boundary",
    "starts_after_punctuation",
    "ends_before_punctuation",
    "has_internal_punctuation",
    "overlap_count",
    "contains_count",
    "contained_by_count",
    "max_overlap_fraction",
    "type_hint_count",
    "hint_ENTITY",
    "hint_ABILITY_OR_RESOURCE",
    "hint_ACTION",
    "hint_EVENT",
    "hint_STATE",
    "hint_TIME",
    "hint_LOCATION_OR_SPACE",
    "hint_QUANTITY",
    "has_digit",
    "has_percent",
    "source_kind_transcript",
)

# Column order of the eight per-hint flags in DENSE_A_FEATURES.  ``OUTCOME``
# hints remain available provenance but are not separate columns in the
# bounded v1 schema; ``type_hint_count`` still counts them.
HINT_FEATURE_COLUMNS = (
    "ENTITY",
    "ABILITY_OR_RESOURCE",
    "ACTION",
    "EVENT",
    "STATE",
    "TIME",
    "LOCATION_OR_SPACE",
    "QUANTITY",
)

DENSE_B_EXTRA_FEATURES = (
    "starts_with_uppercase",
    "all_caps_word",
    "ends_with_ing",
    "ends_with_ed",
    "ends_with_s",
    "has_apostrophe_s",
    "has_hyphen",
    "is_single_token",
    "cue_pronoun",
    "cue_ability_alias",
    "cue_action",
    "cue_event",
    "cue_modal",
    "cue_negation",
    "cue_conditional",
    "cue_causal",
    "cue_temporal",
    "cue_filler",
    "cue_disfluency",
    "has_quote",
)

VECTORIZER_PARAMS = {
    "analyzer": "word",
    "token_pattern": r"[^\W_][\w'’%-]*",
    "ngram_range": (1, 2),
    "binary": True,
    "lowercase": True,
    "max_features": 1500,
    "min_df": 2,
}

SPARSE_CONFIG = {
    "kind": "word_ngrams_1_2_plus_boundary_token_flags",
    "vectorizer": {
        "analyzer": "word",
        "token_pattern": r"[^\W_][\w'’%-]*",
        "ngram_range": [1, 2],
        "binary": True,
        "lowercase": True,
        "max_features": 1500,
        "min_df": 2,
    },
    "boundary_roles": ["first", "last", "head"],
}

LOGISTIC_CONFIG = {
    "penalty": "l2",
    "solver": "lbfgs",
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 300,
    "tol": 1e-4,
    "random_state": SEED,
}

LGBM_CONFIG = {
    "n_estimators": 120,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 20,
    "min_child_weight": 1.0,
    "learning_rate": 0.1,
    "reg_alpha": 1e-3,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": 2,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
    "importance_type": "gain",
}


class Phase2HError(ValueError):
    """Base error for Phase 2H contract violations."""


class Phase2HCoverageError(Phase2HError):
    """Deterministic candidate/dataset coverage validation failed."""


class Phase2HCVError(Phase2HError):
    """Grouped CV could not be trained or scored under the contract."""


@dataclass(frozen=True)
class CandidateRow:
    """One deterministic candidate row with retained provenance.

    Predictive features never include the provenance fields below (case/window
    IDs, candidate IDs, offsets are excluded from feature vectors; only
    normalized/non-identifying derived values are used).
    """

    case_id: str
    window_id: str
    candidate_id: str
    alias: str
    start: int
    end: int
    absolute_start: int
    absolute_end: int
    text: str
    segment_ids: tuple[str, ...]
    segment_bounds: tuple[tuple[int, int], ...]
    type_hints: tuple[str, ...]
    source_kind: str
    is_gold_positive: bool
    label: str
    excluded: bool
    ambiguity_state: str
    gold_mention_ids: tuple[str, ...]
    gold_node_types: tuple[str, ...]


def _metric(hits: int, denominator: int) -> dict[str, Any]:
    return {
        "hit_count": int(hits),
        "denominator": int(denominator),
        "rate": hits / denominator if denominator else None,
    }


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group() for match in _WORD.finditer(text))


def _lower_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group().lower() for match in _WORD.finditer(text))


def _head_token(tokens: Sequence[str]) -> str:
    lowered = [token.lower() for token in tokens]
    for token in reversed(lowered):
        if token not in FUNCTION_WORDS:
            return token
    return lowered[-1] if lowered else ""


def build_dataset(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    """Build the one-row-per-candidate Phase 2H dataset from the locked
    five-case benchmark.

    Positive iff the exact ``(start, end)`` equals any existing acceptable gold
    endpoint span under the Phase 2G contract.  Multiple gold mentions/roles on
    the same exact candidate produce one row with all applicable mention IDs
    and node types retained as provenance.  Nonpositive, nonexcluded rows are
    DROP.  Inclusion/ambiguity/exclusion state is always explicitly
    represented.
    """
    experiments = {
        case["id"]: build_case_experiment(case) for case in benchmark["cases"]
    }
    validate_experiment_coverage(experiments)
    windows: dict[str, dict[str, Any]] = {}
    for case in benchmark["cases"]:
        experiment = experiments[case["id"]]
        source = BronzeSource(case["source_id"], case["source_text"])
        window = window_from_exact_span(source, 0, len(source.text))
        candidates = generate_mention_candidates(window)
        candidate_by_id = {item.candidate_id: item for item in candidates}
        gold_spans: set[tuple[int, int]] = set()
        span_mentions: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for task in experiment["endpoint_tasks"]:
            for span in task["gold_spans"]:
                span_tuple = tuple(span)
                gold_spans.add(span_tuple)
                span_mentions.setdefault(span_tuple, []).append({
                    "mention_id": task["gold_mention_id"],
                    "node_types": list(task["gold_node_types"]),
                })
        rows: list[CandidateRow] = []
        for record in experiment["catalog"]:
            candidate = candidate_by_id.get(record["candidate_id"])
            if candidate is None:
                raise Phase2HCoverageError(
                    f"candidate {record['candidate_id']!r} was not reproduced "
                    "by the frozen generator",
                )
            span = (record["start"], record["end"])
            is_positive = span in gold_spans
            mention_records = span_mentions.get(span, [])
            rows.append(CandidateRow(
                case_id=experiment["case_id"],
                window_id=record["window_id"],
                candidate_id=record["candidate_id"],
                alias=record["alias"],
                start=record["start"],
                end=record["end"],
                absolute_start=record["absolute_start"],
                absolute_end=record["absolute_end"],
                text=record["text"],
                segment_ids=tuple(record["segment_ids"]),
                segment_bounds=tuple(
                    (segment.start, segment.end) for segment in window.segments
                    if segment.segment_id in record["segment_ids"]
                ),
                type_hints=candidate.type_hints,
                source_kind=window.source_kind,
                is_gold_positive=is_positive,
                label=KEEP if is_positive else DROP,
                excluded=False,
                ambiguity_state="NONE",
                gold_mention_ids=tuple(
                    item["mention_id"] for item in mention_records
                ),
                gold_node_types=tuple(
                    node_type
                    for item in mention_records
                    for node_type in item["node_types"]
                ),
            ))
        window_id = experiment["catalog"][0]["window_id"]
        windows[experiment["case_id"]] = {
            "case_id": experiment["case_id"],
            "window_id": window_id,
            "bronze_text": experiment["bronze_text"],
            "bronze_text_sha256": experiment["bronze_text_sha256"],
            "catalog_sha256": experiment["catalog_sha256"],
            "candidate_generator_version": MENTION_CATALOG_VERSION,
            "gold_spans": tuple(sorted(gold_spans)),
            "rows": tuple(rows),
        }
    dataset = {"windows": windows}
    validate_dataset(dataset)
    return dataset


def validate_dataset(
    dataset: Mapping[str, Any], *, expected_positive_count: int = 33,
) -> dict[str, Any]:
    """Validate the Phase 2H dataset contract (33/33, uniqueness, labels)."""
    if (
        isinstance(expected_positive_count, bool)
        or not isinstance(expected_positive_count, int)
        or expected_positive_count <= 0
    ):
        raise Phase2HCoverageError("expected positive count must be positive")
    windows = dataset.get("windows")
    if not isinstance(windows, Mapping) or not windows:
        raise Phase2HCoverageError("dataset must contain at least one window")
    window_ids = sorted(windows)
    total_candidates = 0
    total_positives = 0
    per_window: dict[str, dict[str, Any]] = {}
    for window_id in window_ids:
        window = windows[window_id]
        rows = window.get("rows")
        if not isinstance(rows, tuple) or not rows:
            raise Phase2HCoverageError(
                f"window {window_id} rows must be a nonempty tuple",
            )
        bronze_text = window.get("bronze_text")
        if not isinstance(bronze_text, str):
            raise Phase2HCoverageError(f"window {window_id} bronze text missing")
        candidate_ids: set[str] = set()
        spans: set[tuple[int, int]] = set()
        positive_count = 0
        for row in rows:
            if not isinstance(row, CandidateRow):
                raise Phase2HCoverageError("rows must be CandidateRow objects")
            if row.candidate_id in candidate_ids:
                raise Phase2HCoverageError(
                    f"duplicate candidate row {row.candidate_id!r}",
                )
            if (row.start, row.end) in spans:
                raise Phase2HCoverageError(
                    f"duplicate candidate span in window {window_id}",
                )
            candidate_ids.add(row.candidate_id)
            spans.add((row.start, row.end))
            source_window_id = window.get("window_id")
            if (
                not isinstance(source_window_id, str)
                or not row.candidate_id.startswith(source_window_id + ":m")
            ):
                raise Phase2HCoverageError(
                    f"candidate {row.candidate_id!r} is not bound to its window",
                )
            if (
                not isinstance(row.start, int) or not isinstance(row.end, int)
                or not 0 <= row.start < row.end <= len(bronze_text)
            ):
                raise Phase2HCoverageError("candidate offsets are invalid")
            if bronze_text[row.start:row.end] != row.text:
                raise Phase2HCoverageError(
                    "candidate text is not the exact bronze slice",
                )
            if not row.segment_ids or any(
                not isinstance(item, str) or not item for item in row.segment_ids
            ):
                raise Phase2HCoverageError(
                    "candidate segment provenance must be a nonempty id list",
                )
            if not row.segment_bounds or any(
                not isinstance(item, tuple) or len(item) != 2
                for item in row.segment_bounds
            ):
                raise Phase2HCoverageError(
                    "candidate segment bounds must be a nonempty offset list",
                )
            if not row.type_hints or not set(row.type_hints) <= NODE_TYPES:
                raise Phase2HCoverageError("candidate type hints are invalid")
            if row.is_gold_positive != (row.label == KEEP):
                raise Phase2HCoverageError("candidate label is inconsistent")
            if row.is_gold_positive != bool(row.gold_mention_ids):
                raise Phase2HCoverageError(
                    "gold positive must retain all mention IDs",
                )
            if row.excluded:
                raise Phase2HCoverageError(
                    "the locked Phase 2G fixture defines no exclusions",
                )
            if row.ambiguity_state != "NONE":
                raise Phase2HCoverageError(
                    "the locked Phase 2G fixture defines no ambiguous rows",
                )
            if row.is_gold_positive:
                positive_count += 1
        expected_positives = len(window.get("gold_spans", ()))
        if positive_count != expected_positives:
            raise Phase2HCoverageError(
                f"window {window_id} positive rows {positive_count} do not "
                f"match gold span count {expected_positives}",
            )
        total_candidates += len(rows)
        total_positives += positive_count
        per_window[window_id] = {
            "candidate_count": len(rows),
            "positive_count": positive_count,
            "candidate_coverage": _metric(positive_count, expected_positives),
        }
    if total_positives != expected_positive_count:
        raise Phase2HCoverageError(
            f"gold endpoint coverage is {total_positives}/"
            f"{expected_positive_count}, expected "
            f"{expected_positive_count}/{expected_positive_count}",
        )
    summary = {
        "window_count": len(window_ids),
        "candidate_count": total_candidates,
        "positive_count": total_positives,
        "per_window": per_window,
    }
    return summary


def _interval_stats(
    rows: Sequence[CandidateRow], window_text: str,
) -> np.ndarray:
    """Dense geometry/provenance feature matrix (feature set A)."""
    n = len(rows)
    text_len = len(window_text)
    starts = np.array([row.start for row in rows], dtype=np.float64)
    ends = np.array([row.end for row in rows], dtype=np.float64)
    lengths = ends - starts
    if n == 0:
        return np.zeros((0, len(DENSE_A_FEATURES)), dtype=np.float64)
    overlap = (
        (starts[:, None] < ends[None, :]) & (ends[:, None] > starts[None, :])
    )
    np.fill_diagonal(overlap, False)
    overlap_count = overlap.sum(axis=1)
    contains = (
        (starts[:, None] <= starts[None, :])
        & (ends[:, None] >= ends[None, :])
        & (
            (starts[:, None] != starts[None, :])
            | (ends[:, None] != ends[None, :])
        )
    )
    contains_count = contains.sum(axis=1)
    contained_by = contains.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pairwise = np.minimum(ends[:, None], ends[None, :]) - np.maximum(
            starts[:, None], starts[None, :],
        )
        pairwise[pairwise < 0] = 0
        np.fill_diagonal(pairwise, 0)
        max_overlap_fraction = pairwise.max(axis=1) / lengths
        max_overlap_fraction = np.nan_to_num(
            max_overlap_fraction, nan=0.0, posinf=0.0, neginf=0.0,
        )
    hint_flags = np.zeros((n, len(HINT_FEATURE_COLUMNS)), dtype=np.float64)
    for index, row in enumerate(rows):
        for hint in row.type_hints:
            if hint in HINT_FEATURE_COLUMNS:
                hint_flags[index, HINT_FEATURE_COLUMNS.index(hint)] = 1.0
    matrix = np.zeros((n, len(DENSE_A_FEATURES)), dtype=np.float64)
    matrix[:, 0] = lengths
    matrix[:, 1] = np.array([len(_tokens(row.text)) for row in rows])
    matrix[:, 2] = starts / text_len
    matrix[:, 3] = ends / text_len
    matrix[:, 4] = lengths / text_len
    matrix[:, 5] = np.array([len(row.segment_ids) for row in rows])
    matrix[:, 6] = np.array([
        1.0 if any(row.start == bound_start for bound_start, _ in row.segment_bounds)
        else 0.0
        for row in rows
    ])
    matrix[:, 7] = np.array([
        1.0 if any(row.end == bound_end for _, bound_end in row.segment_bounds)
        else 0.0
        for row in rows
    ])
    for index, row in enumerate(rows):
        before = _previous_nonspace(window_text, row.start)
        after = _next_nonspace(window_text, row.end)
        matrix[index, 8] = 1.0 if before in _PUNCTUATION_AFTER else 0.0
        matrix[index, 9] = 1.0 if after in _PUNCTUATION_BEFORE else 0.0
        matrix[index, 10] = 1.0 if _has_internal_punctuation(row.text) else 0.0
        matrix[index, 15] = len(row.type_hints)
        matrix[index, 16:16 + len(HINT_FEATURE_COLUMNS)] = hint_flags[index]
        matrix[index, 24] = 1.0 if re.search(r"\d", row.text) else 0.0
        matrix[index, 25] = 1.0 if re.search(r"\d+(?:\.\d+)?%", row.text) else 0.0
        matrix[index, 26] = 1.0 if row.source_kind == "transcript" else 0.0
    matrix[:, 12] = contains_count.astype(np.float64)
    matrix[:, 13] = contained_by.astype(np.float64)
    matrix[:, 11] = overlap_count.astype(np.float64)
    matrix[:, 14] = max_overlap_fraction
    return matrix


def _previous_nonspace(
    text: str, start: int,
) -> str:
    position = start - 1
    while position >= 0:
        character = text[position]
        if not character.isspace():
            return character
        position -= 1
    return ""


def _next_nonspace(
    text: str, end: int,
) -> str:
    position = end
    while position < len(text):
        character = text[position]
        if not character.isspace():
            return character
        position += 1
    return ""


def _has_internal_punctuation(text: str) -> bool:
    return bool(re.search(r"[,;:.!?—–-]", text))


def _surface_cue_matrix(rows: Sequence[CandidateRow]) -> np.ndarray:
    """Dense surface/cue feature matrix (feature set B extras)."""
    n = len(rows)
    matrix = np.zeros((n, len(DENSE_B_EXTRA_FEATURES)), dtype=np.float64)
    for index, row in enumerate(rows):
        tokens = _tokens(row.text)
        lowered = [token.lower() for token in tokens]
        matrix[index, 0] = 1.0 if tokens and tokens[0][:1].isupper() else 0.0
        matrix[index, 1] = 1.0 if tokens and all(
            token.isalpha() and token == token.upper() and token.lower() != token
            for token in tokens
        ) else 0.0
        matrix[index, 2] = 1.0 if lowered and lowered[-1].endswith("ing") else 0.0
        matrix[index, 3] = 1.0 if lowered and lowered[-1].endswith("ed") else 0.0
        matrix[index, 4] = 1.0 if lowered and lowered[-1].endswith("s") else 0.0
        matrix[index, 5] = 1.0 if re.search(r"['’]s\b", row.text.lower()) else 0.0
        matrix[index, 6] = 1.0 if "-" in row.text else 0.0
        matrix[index, 7] = 1.0 if len(tokens) == 1 else 0.0
        matrix[index, 8] = 1.0 if set(lowered) & PRONOUN_TOKENS else 0.0
        matrix[index, 9] = 1.0 if set(lowered) & ABILITY_ALIAS_TOKENS else 0.0
        matrix[index, 10] = 1.0 if set(lowered) & ACTION_TOKENS else 0.0
        matrix[index, 11] = 1.0 if set(lowered) & EVENT_TOKENS else 0.0
        matrix[index, 12] = 1.0 if set(lowered) & MODAL_TOKENS else 0.0
        matrix[index, 13] = 1.0 if set(lowered) & NEGATION_TOKENS else 0.0
        matrix[index, 14] = 1.0 if set(lowered) & CONDITIONAL_TOKENS else 0.0
        matrix[index, 15] = 1.0 if set(lowered) & CAUSAL_TOKENS else 0.0
        matrix[index, 16] = 1.0 if set(lowered) & TEMPORAL_TOKENS else 0.0
        matrix[index, 17] = 1.0 if set(lowered) & FILLER_TOKENS else 0.0
        matrix[index, 18] = 1.0 if _disfluency(row.text) else 0.0
        matrix[index, 19] = 1.0 if '"' in row.text or "'" in row.text else 0.0
    return matrix


def _disfluency(text: str) -> bool:
    lowered = " ".join(_lower_tokens(text))
    if any(bigram in lowered for bigram in DISFLUENCY_BIGRAMS):
        return True
    return bool(set(_lower_tokens(text)) & FILLER_TOKENS)


def extract_dense_features(
    dataset: Mapping[str, Any], window_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dense A, dense B-extra) for rows in the given windows."""
    chunks_a: list[np.ndarray] = []
    chunks_b: list[np.ndarray] = []
    for window_id in window_ids:
        window = dataset["windows"][window_id]
        rows = window["rows"]
        chunks_a.append(_interval_stats(rows, window["bronze_text"]))
        chunks_b.append(_surface_cue_matrix(rows))
    return np.vstack(chunks_a), np.vstack(chunks_b)


def extract_sparse_inputs(
    dataset: Mapping[str, Any], window_ids: Sequence[str],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (candidate texts, boundary token triples) for sparse features."""
    texts: list[str] = []
    boundaries: list[tuple[str, str, str]] = []
    for window_id in window_ids:
        for row in dataset["windows"][window_id]["rows"]:
            tokens = _tokens(row.text)
            lowered = [token.lower() for token in tokens]
            first = lowered[0] if lowered else ""
            last = lowered[-1] if lowered else ""
            head = _head_token(tokens)
            texts.append(row.text)
            boundaries.append((first, last, head))
    return texts, boundaries


def _held_out_token_types(texts: Sequence[str]) -> set[str]:
    """Distinct lowercased token types in held-out texts.

    Uses the exact sparse-pipeline tokenizer/lowercasing semantics
    (``_WORD`` with ``[^\\W_][\\w'’%-]*`` plus lowercase), which matches
    ``CountVectorizer`` with ``lowercase=True`` and the same token pattern.
    Boundary tokens (first/last/head) are derived from these same lowercased
    text tokens, so they are covered by this set; an OOV boundary token simply
    emits no boundary flag.
    """
    return {
        match.group().lower()
        for text in texts
        for match in _WORD.finditer(text)
    }


def _oov_audit(
    test_token_types: set[str],
    vocab_terms: Sequence[str],
) -> dict[str, Any]:
    """Audit of held-out token types absent from the fitted vocabulary."""
    vocabulary = set(vocab_terms)
    oov_types = sorted(test_token_types - vocabulary)
    return {
        "oov_definition": (
            "distinct lowercased token types (unigrams) in held-out texts "
            "absent from the fitted training-only vocabulary; same "
            "tokenizer/lowercasing semantics as the sparse features"
        ),
        "test_oov_token_types": oov_types,
        "test_oov_token_type_count": len(oov_types),
        "test_oov_token_types_sha256": canonical_sha256(oov_types),
    }


def feature_schema() -> dict[str, Any]:
    """Versioned, testable feature schema (no identifiers, no gold-derived
    values; vocabularies/scalers are fit on training rows only)."""
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "feature_set_A": {
            "label": "geometry/provenance",
            "dense_features": list(DENSE_A_FEATURES),
            "note": (
                "non-identifying geometry, segment/punctuation boundaries, "
                "overlap/containment statistics, and candidate generator "
                "type-hint evidence"
            ),
        },
        "feature_set_B": {
            "label": "A plus bounded lexical/cue features",
            "dense_extras": list(DENSE_B_EXTRA_FEATURES),
            "sparse": SPARSE_CONFIG,
            "boundary_token_definition": (
                "first=first token, last=last token, head=last non-function "
                "token (fallback last token)"
            ),
        },
        "fit_scope": "training windows only",
        "prohibited_features": [
            "case ids", "source ids", "window ids", "candidate ids",
            "mention ids", "question text", "gold roles/types",
            "label-derived values", "DeepSeek labels/predictions",
        ],
    }


def _boundary_sparse(
    boundaries: Sequence[tuple[str, str, str]],
    vocab_terms: Sequence[str],
) -> sp.csr_matrix:
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


class CellPreprocessor:
    """Preprocessing fitted on training windows only."""

    def __init__(self, feature_set: str) -> None:
        if feature_set not in FEATURE_SETS:
            raise Phase2HError(f"unknown feature set {feature_set!r}")
        self.feature_set = feature_set
        self.scaler: StandardScaler | None = None
        self.vectorizer: CountVectorizer | None = None
        self.vocab_terms: list[str] = []

    def fit(
        self,
        dense: np.ndarray,
        texts: Sequence[str],
        boundaries: Sequence[tuple[str, str, str]],
    ) -> "CellPreprocessor":
        self.scaler = StandardScaler().fit(dense)
        if self.feature_set == "B":
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
                # min_df pruning can legitimately remove every term on a tiny
                # training corpus; the sparse block then contributes no
                # columns and the cell reduces to the scaled dense features.
                self.vectorizer = None
                self.vocab_terms = []
        return self

    def transform(
        self,
        dense: np.ndarray,
        texts: Sequence[str],
        boundaries: Sequence[tuple[str, str, str]],
    ) -> np.ndarray | sp.csr_matrix:
        if self.scaler is None:
            raise Phase2HError("preprocessor must be fitted before transform")
        scaled = self.scaler.transform(dense)
        if self.feature_set == "A":
            return scaled
        if self.vectorizer is None:
            ngrams = sp.csr_matrix((len(dense), 0), dtype=np.float64)
        else:
            ngrams = self.vectorizer.transform(texts)
        boundary_flags = _boundary_sparse(boundaries, self.vocab_terms)
        return sp.hstack([scaled, ngrams, boundary_flags], format="csr")

    def feature_names(self, dense_names: Sequence[str]) -> list[str]:
        if self.feature_set == "A":
            return list(dense_names)
        ngram_names = [f"ngram={term}" for term in self.vocab_terms]
        boundary_names = [
            f"{role}={term}"
            for role in SPARSE_CONFIG["boundary_roles"]
            for term in self.vocab_terms
        ]
        return list(dense_names) + ngram_names + boundary_names


def balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced weights from training labels only (equivalents of sklearn's
    ``balanced`` rule: ``n / (2 * n_class)``)."""
    n = len(y)
    positive = int(y.sum())
    negative = n - positive
    if positive == 0 or negative == 0:
        raise Phase2HCVError(
            "balanced class weighting requires both classes in training rows",
        )
    return {
        1: n / (2 * positive),
        0: n / (2 * negative),
    }


def make_model(cell: str, y_train: np.ndarray) -> Any:
    """Create the fixed model for a cell (no tuning)."""
    if cell not in CELLS:
        raise Phase2HError(f"unknown model cell {cell!r}")
    model_name, _ = cell.split("_")
    if model_name == "logistic":
        return LogisticRegression(**LOGISTIC_CONFIG)
    weights = balanced_class_weights(y_train)
    return lightgbm.LGBMClassifier(
        **{**LGBM_CONFIG, "class_weight": weights},
    )


def _fold_slices(window_ids: Sequence[str], dataset: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    counts = [len(dataset["windows"][wid]["rows"]) for wid in window_ids]
    offsets = np.cumsum([0] + counts)
    return np.array([
        (offsets[index], offsets[index + 1]) for index in range(len(window_ids))
    ], dtype=np.int64)


def run_cv(
    dataset: Mapping[str, Any],
    *,
    cells: Sequence[str] = CELLS,
    verbose: bool = False,
) -> dict[str, Any]:
    """Grouped leave-one-window-out CV over the locked five-case benchmark.

    Every model cell sees identical folds.  All preprocessing/vectorizers/
    scalers/class weights/training statistics are fit on training windows only.
    Raises :class:`Phase2HCVError` if a training fold lacks either class and
    :class:`Phase2HError` if the same cell is requested more than once.
    """
    cells = tuple(cells)
    if len(set(cells)) != len(cells):
        raise Phase2HError(
            f"duplicate model cells requested: {', '.join(cells)}",
        )
    for cell in cells:
        if cell not in CELLS:
            raise Phase2HError(f"unknown model cell {cell!r}")
    window_ids = sorted(dataset["windows"])
    if len(window_ids) < 2:
        raise Phase2HCVError(
            "leave-one-window-out CV requires at least two windows",
        )
    dense_a_all, dense_b_extra_all = extract_dense_features(dataset, window_ids)
    dense_b_all = np.hstack([dense_a_all, dense_b_extra_all])
    texts_all, boundaries_all = extract_sparse_inputs(dataset, window_ids)
    labels_all = np.array([
        1 if row.label == KEEP else 0
        for wid in window_ids
        for row in dataset["windows"][wid]["rows"]
    ], dtype=np.int64)
    slices = _fold_slices(window_ids, dataset)
    folds: list[dict[str, Any]] = []
    oof_scores: dict[str, dict[str, dict[str, float]]] = {
        wid: {} for wid in window_ids
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
            wid for index, wid in enumerate(window_ids)
            if index != fold_index
        ]
        train_slices = np.concatenate([
            np.arange(start, stop)
            for index, (start, stop) in enumerate(slices)
            if index != fold_index
        ])
        test_slices = np.arange(test_start, test_end)
        y_train = labels_all[train_slices]
        train_positive_count = int(y_train.sum())
        train_negative_count = int(len(y_train) - train_positive_count)
        if train_positive_count == 0:
            raise Phase2HCVError(
                f"fold {fold_index} (holdout {test_window_id!r}) has no "
                "positive training examples; the contract forbids training "
                "or testing this fold",
            )
        if train_negative_count == 0:
            raise Phase2HCVError(
                f"fold {fold_index} (holdout {test_window_id!r}) has no "
                "negative training examples; balanced class weights require "
                "both classes in the training rows",
            )
        train_dense_a = dense_a_all[train_slices]
        test_dense_a = dense_a_all[test_slices]
        train_dense_b = dense_b_all[train_slices]
        test_dense_b = dense_b_all[test_slices]
        train_texts = [texts_all[index] for index in train_slices]
        test_texts = [texts_all[index] for index in test_slices]
        train_boundaries = [boundaries_all[index] for index in train_slices]
        test_boundaries = [boundaries_all[index] for index in test_slices]
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
            model_name, feature_set = cell.split("_")
            if verbose:
                print(
                    f"[phase2h] fold {fold_index} {cell}: "
                    f"train={len(train_slices)} test={len(test_slices)}",
                    file=sys.stderr,
                )
            dense_train = train_dense_a if feature_set == "A" else train_dense_b
            dense_test = test_dense_a if feature_set == "A" else test_dense_b
            preprocessor = CellPreprocessor(feature_set)
            preprocessor.fit(
                dense_train, train_texts, train_boundaries,
            )
            x_train = preprocessor.transform(
                dense_train, train_texts, train_boundaries,
            )
            x_test = preprocessor.transform(
                dense_test, test_texts, test_boundaries,
            )
            model = make_model(cell, y_train)
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            for index, row in enumerate(test_rows):
                oof_scores[test_window_id].setdefault(row.candidate_id, {})[
                    cell
                ] = float(probabilities[index])
            dense_names = (
                list(DENSE_A_FEATURES)
                if feature_set == "A"
                else list(DENSE_A_FEATURES) + list(DENSE_B_EXTRA_FEATURES)
            )
            names = preprocessor.feature_names(dense_names)
            oov_audit = None
            if feature_set == "B":
                oov_audit = _oov_audit(
                    test_token_types, preprocessor.vocab_terms,
                )
            fit_scope[cell][fold_index] = _fit_scope_record(
                fold_index=fold_index,
                train_window_ids=train_window_ids,
                test_window_id=test_window_id,
                y_train=y_train,
                weights=weights,
                preprocessor=preprocessor,
                feature_names=names,
                model=model,
                cell=cell,
                oov_audit=oov_audit,
            )
            fitted_models[cell][fold_index] = (model, names)
    return {
        "folds": folds,
        "fit_scope": fit_scope,
        "oof_scores": oof_scores,
        "fitted_models": fitted_models,
    }


def _fit_scope_record(
    *,
    fold_index: int,
    train_window_ids: list[str],
    test_window_id: str,
    y_train: np.ndarray,
    weights: dict[int, float],
    preprocessor: CellPreprocessor,
    feature_names: list[str],
    model: Any,
    cell: str,
    oov_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scaler = preprocessor.scaler
    if scaler is None:
        raise Phase2HError("scaler was not fitted")
    mean = [float(value) for value in scaler.mean_]
    scale = [float(value) for value in scaler.scale_]
    record: dict[str, Any] = {
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
            "feature_count": int(len(mean)),
            "mean": mean,
            "scale": scale,
            "mean_sha256": canonical_sha256(mean),
            "scale_sha256": canonical_sha256(scale),
        },
        "model_config": _model_config_snapshot(model),
    }
    if preprocessor.feature_set == "B":
        vocabulary = list(preprocessor.vocab_terms)
        if oov_audit is None:
            raise Phase2HError(
                "B-cell fit-scope records require a held-out OOV audit",
            )
        record["vectorizer"] = {
            "fit_on": "training_rows_only",
            "vocabulary_terms_origin": "training_rows_only",
            "params": dict(VECTORIZER_PARAMS),
            "vocabulary_size": len(vocabulary),
            "vocabulary_sha256": canonical_sha256(vocabulary),
        }
        record["vectorizer"].update(oov_audit)
    record["feature_names_sha256"] = canonical_sha256(feature_names)
    return record


def _model_config_snapshot(model: Any) -> dict[str, Any]:
    if isinstance(model, LogisticRegression):
        return {"family": "logistic", "params": dict(LOGISTIC_CONFIG)}
    params = model.get_params()
    if "class_weight" in params and isinstance(params["class_weight"], dict):
        params["class_weight"] = {
            KEEP: float(params["class_weight"][1]),
            DROP: float(params["class_weight"][0]),
        }
    return {"family": "lightgbm", "params": params}


def compute_rankings(
    dataset: Mapping[str, Any],
    oof_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    cells: Sequence[str] = CELLS,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Rank candidates within each held-out window by descending score with
    deterministic candidate-order tie breaking; 1-based ranks."""
    rankings: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for window_id in sorted(dataset["windows"]):
        rows = dataset["windows"][window_id]["rows"]
        window_rankings: dict[str, dict[str, dict[str, Any]]] = {}
        for cell in cells:
            scores = [
                oof_scores[window_id][row.candidate_id][cell]
                for row in rows
            ]
            order = sorted(
                range(len(rows)),
                key=lambda index: (-scores[index], index),
            )
            cell_rankings: dict[str, dict[str, Any]] = {}
            for rank, index in enumerate(order, 1):
                row = rows[index]
                cell_rankings[row.candidate_id] = {
                    "score": scores[index],
                    "rank": rank,
                    "selected": KEEP if scores[index] >= KEEP_THRESHOLD else DROP,
                }
            window_rankings[cell] = cell_rankings
        rankings[window_id] = window_rankings
    return rankings


def _window_metrics(
    window_id: str,
    rows: Sequence[CandidateRow],
    rankings: Mapping[str, Mapping[str, dict[str, Any]]],
    cell: str,
) -> dict[str, Any]:
    labels = np.array([1 if row.label == KEEP else 0 for row in rows])
    scores = np.array([
        rankings[window_id][cell][row.candidate_id]["score"] for row in rows
    ])
    selected = scores >= KEEP_THRESHOLD
    predicted = selected.astype(int)
    positive_count = int(labels.sum())
    keep_count = int(predicted.sum())
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    recall = _metric(tp, positive_count)
    precision = _metric(tp, keep_count)
    f1 = (
        (2 * tp / (2 * tp + fp + fn))
        if (2 * tp + fp + fn) else None
    )
    ap = _score_metric(_safe_average_precision(labels, scores), positive_count)
    auc = _score_metric(_safe_roc_auc(labels, scores), len(labels))
    prevalence = positive_count / len(labels) if labels.size else 0.0
    order = sorted(
        range(len(rows)),
        key=lambda index: (-scores[index], index),
    )
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    recall_at_k: dict[str, dict[str, Any]] = {}
    precision_at_k: dict[str, dict[str, Any]] = {}
    for k in RANK_KS:
        top = order[:k]
        hits = int(sum(1 for index in top if labels[index] == 1))
        recall_at_k[str(k)] = _metric(hits, positive_count)
        precision_at_k[str(k)] = _metric(hits, min(k, len(rows)))
    gold_ranks = [
        ranks[index] for index in range(len(rows)) if labels[index] == 1
    ]
    gold_rank_stats = _rank_stats(gold_ranks)
    overlap = _overlap_diagnostics(rows, rankings[window_id][cell])
    return {
        "window_id": window_id,
        "candidate_count": len(rows),
        "label_keep_count": positive_count,
        "label_drop_count": int(len(labels) - positive_count),
        "predicted_keep_count": keep_count,
        "predicted_drop_count": int(len(labels) - keep_count),
        "selected": keep_count,
        "prevalence": prevalence,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "precision": precision,
        "recall": recall,
        "f1": {"value": f1},
        "average_precision": ap,
        "roc_auc": auc,
        "all_drop_baseline": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        "all_keep_baseline": {
            "precision": prevalence,
            "recall": 1.0,
            "f1": (
                (2 * prevalence / (1 + prevalence))
                if prevalence > 0 else 0.0
            ),
        },
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "gold_rank": gold_rank_stats,
        "overlap_diagnostics": overlap,
    }


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if int(labels.sum()) == 0 or int(len(labels) - labels.sum()) == 0:
        return None
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores))


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if int(labels.sum()) == 0 or int(len(labels) - labels.sum()) == 0:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def _score_metric(value: float | None, denominator: int) -> dict[str, Any]:
    return {"value": value, "denominator": int(denominator)}


def _rank_stats(ranks: list[int]) -> dict[str, Any]:
    if not ranks:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(ranks),
        "mean": float(np.mean(ranks)),
        "median": float(np.median(ranks)),
    }


def _overlap_diagnostics(
    rows: Sequence[CandidateRow],
    cell_rankings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rank_by_id = {
        candidate_id: entry["rank"]
        for candidate_id, entry in cell_rankings.items()
    }
    entries: list[dict[str, Any]] = []
    containing_total = 0
    contained_total = 0
    cluster_ranks: list[int] = []
    cluster_sizes: list[int] = []
    for row in rows:
        if not row.is_gold_positive:
            continue
        cluster = [
            other for other in rows
            if other.start < row.end and other.end > row.start
        ]
        containing = [
            other for other in rows
            if other.start < row.start and other.end > row.end
        ]
        contained = [
            other for other in rows
            if other.start > row.start and other.end < row.end
        ]
        gold_rank = rank_by_id[row.candidate_id]
        cluster_rank = 1 + sum(
            1 for other in cluster
            if rank_by_id[other.candidate_id] < gold_rank
        )
        containing_outranking = sum(
            1 for other in containing
            if rank_by_id[other.candidate_id] < gold_rank
        )
        contained_outranking = sum(
            1 for other in contained
            if rank_by_id[other.candidate_id] < gold_rank
        )
        containing_total += containing_outranking
        contained_total += contained_outranking
        cluster_ranks.append(cluster_rank)
        cluster_sizes.append(len(cluster))
        entries.append({
            "candidate_id": row.candidate_id,
            "gold_rank": gold_rank,
            "overlap_cluster_size": len(cluster),
            "overlap_cluster_rank": cluster_rank,
            "containing_distractors_outranking": containing_outranking,
            "contained_distractors_outranking": contained_outranking,
        })
    return {
        "gold_positive_count": len(entries),
        "containing_distractors_outranking_total": containing_total,
        "contained_distractors_outranking_total": contained_total,
        "overlap_cluster_rank": _rank_stats(cluster_ranks),
        "overlap_cluster_size": _rank_stats(cluster_sizes),
        "per_gold_positive": entries,
    }


def compute_cell_metrics(
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    cell: str,
) -> dict[str, Any]:
    """Candidate-level pooled OOF metrics plus per-fold metrics for one cell."""
    if cell not in CELLS:
        raise Phase2HError(f"unknown model cell {cell!r}")
    window_ids = sorted(dataset["windows"])
    per_fold: dict[str, dict[str, Any]] = {}
    labels: list[int] = []
    scores: list[float] = []
    for window_id in window_ids:
        rows = dataset["windows"][window_id]["rows"]
        per_fold[window_id] = _window_metrics(window_id, rows, rankings, cell)
        for row in rows:
            entry = rankings[window_id][cell][row.candidate_id]
            labels.append(1 if row.label == KEEP else 0)
            scores.append(entry["score"])
    return _pooled_metrics(labels, scores, per_fold, window_ids)


def _pooled_metrics(
    labels: list[int],
    scores: list[float],
    per_fold: Mapping[str, Mapping[str, Any]],
    window_ids: Sequence[str],
) -> dict[str, Any]:
    labels_array = np.array(labels, dtype=np.int64)
    scores_array = np.array(scores, dtype=np.float64)
    selected = scores_array >= KEEP_THRESHOLD
    predicted = selected.astype(int)
    positive_count = int(labels_array.sum())
    keep_count = int(predicted.sum())
    tp = int(((predicted == 1) & (labels_array == 1)).sum())
    fp = int(((predicted == 1) & (labels_array == 0)).sum())
    tn = int(((predicted == 0) & (labels_array == 0)).sum())
    fn = int(((predicted == 0) & (labels_array == 1)).sum())
    prevalence = positive_count / len(labels_array) if labels_array.size else 0.0
    recall_at_k: dict[str, dict[str, Any]] = {}
    precision_at_k: dict[str, dict[str, Any]] = {}
    for k in RANK_KS:
        recall_hits = sum(
            per_fold[wid]["recall_at_k"][str(k)]["hit_count"]
            for wid in window_ids
        )
        recall_denominator = sum(
            per_fold[wid]["recall_at_k"][str(k)]["denominator"]
            for wid in window_ids
        )
        precision_hits = sum(
            per_fold[wid]["precision_at_k"][str(k)]["hit_count"]
            for wid in window_ids
        )
        precision_denominator = sum(
            per_fold[wid]["precision_at_k"][str(k)]["denominator"]
            for wid in window_ids
        )
        recall_at_k[str(k)] = _metric(recall_hits, recall_denominator)
        precision_at_k[str(k)] = _metric(precision_hits, precision_denominator)
    gold_ranks = [
        rank
        for wid in window_ids
        for rank in per_fold[wid]["overlap_diagnostics"]["per_gold_positive"]
    ]
    gold_rank_values = [item["gold_rank"] for item in gold_ranks]
    cluster_ranks = [item["overlap_cluster_rank"] for item in gold_ranks]
    return {
        "candidate_count": len(labels_array),
        "label_keep_count": positive_count,
        "label_drop_count": int(len(labels_array) - positive_count),
        "predicted_keep_count": keep_count,
        "predicted_drop_count": int(len(labels_array) - keep_count),
        "selected": keep_count,
        "prevalence": prevalence,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "precision": _metric(tp, keep_count),
        "recall": _metric(tp, positive_count),
        "f1": {
            "value": (
                (2 * tp / (2 * tp + fp + fn))
                if (2 * tp + fp + fn) else None
            ),
        },
        "average_precision": _score_metric(
            _safe_average_precision(labels_array, scores_array),
            positive_count,
        ),
        "roc_auc": _score_metric(
            _safe_roc_auc(labels_array, scores_array),
            len(labels_array),
        ),
        "all_drop_baseline": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        "all_keep_baseline": {
            "precision": prevalence,
            "recall": 1.0,
            "f1": (
                (2 * prevalence / (1 + prevalence))
                if prevalence > 0 else 0.0
            ),
        },
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "gold_rank": _rank_stats(gold_rank_values),
        "overlap_cluster_rank": _rank_stats(cluster_ranks),
        "per_fold": per_fold,
    }


def classify_candidate_error(
    row: CandidateRow,
    *,
    window_gold_spans: Sequence[tuple[int, int]],
    window_rows: Sequence[CandidateRow],
    predicted_label: str,
    rank: int,
    window_text_len: int,
) -> str | None:
    """Deterministic diagnostic error taxonomy (never feeds training).

    Correct predictions return ``None``.  False negatives return
    ``GOLD_RANKED_HIGH_THRESHOLD_MISS`` (gold ranked within the top
    ``GOLD_RANK_HIGH_K`` but below the fixed threshold) or ``GOLD_RANKED_LOW``.
    False positives use the documented precedence in ``ERROR_PRECEDENCE``.
    """
    if predicted_label not in (KEEP, DROP):
        raise Phase2HError(f"unknown predicted label {predicted_label!r}")
    if row.label == predicted_label:
        return None
    if row.label == KEEP:
        if rank <= min(GOLD_RANK_HIGH_K, len(window_rows)):
            return "GOLD_RANKED_HIGH_THRESHOLD_MISS"
        return "GOLD_RANKED_LOW"
    lowered = set(_lower_tokens(row.text))
    start, end = row.start, row.end
    # ``PARSER_FEATURE_ERROR`` remains in the taxonomy for a future Feature
    # Set C integration, but Phase 2H runs no parser and carries no parser
    # evidence state on CandidateRow, so it is never assigned here.
    # Inclusive-boundary span containment: LONGER_SPAN when the gold span is
    # fully contained by the candidate with at least one boundary strictly
    # extended; SHORTER_FRAGMENT symmetrically when the candidate is fully
    # contained by the gold span.  An exact same-span candidate matches
    # neither and falls through to the remaining codes.
    if any(
        gold_start >= start and gold_end <= end
        and (gold_start > start or gold_end < end)
        for gold_start, gold_end in window_gold_spans
    ):
        return "OVERLAPPING_LONGER_SPAN"
    if any(
        gold_start <= start and gold_end >= end
        and (gold_start < start or gold_end > end)
        for gold_start, gold_end in window_gold_spans
    ):
        return "OVERLAPPING_SHORTER_FRAGMENT"
    if lowered and lowered <= PRONOUN_TOKENS:
        return "PRONOUN_DISTRACTOR"
    if _disfluency(row.text):
        return "DISCOURSE_FILLER"
    if lowered & ACTION_TOKENS:
        return "GENERIC_ACTION_DISTRACTOR"
    if "ENTITY" in row.type_hints and len(_tokens(row.text)) <= 2:
        return "GENERIC_ENTITY_DISTRACTOR"
    if lowered & (MODAL_TOKENS | NEGATION_TOKENS | CONDITIONAL_TOKENS | CAUSAL_TOKENS | TEMPORAL_TOKENS):
        return "WRONG_CUE_PRIOR"
    if window_text_len and (
        start / window_text_len < 0.05 or end / window_text_len > 0.95
    ):
        return "SOURCE_POSITION_BIAS"
    return "OTHER"


def classify_all_errors(
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    *,
    cells: Sequence[str] = CELLS,
) -> dict[str, dict[str, dict[str, str | None]]]:
    errors: dict[str, dict[str, dict[str, str | None]]] = {}
    for cell in cells:
        errors[cell] = {}
        for window_id in sorted(dataset["windows"]):
            window = dataset["windows"][window_id]
            rows = window["rows"]
            cell_errors: dict[str, str | None] = {}
            for row in rows:
                entry = rankings[window_id][cell][row.candidate_id]
                cell_errors[row.candidate_id] = classify_candidate_error(
                    row,
                    window_gold_spans=window["gold_spans"],
                    window_rows=rows,
                    predicted_label=entry["selected"],
                    rank=entry["rank"],
                    window_text_len=len(window["bronze_text"]),
                )
            errors[cell][window_id] = cell_errors
    return errors


def error_taxonomy_counts(
    errors: Mapping[str, Mapping[str, Mapping[str, str | None]]],
    dataset: Mapping[str, Any],
    *,
    cells: Sequence[str] = CELLS,
) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for cell in cells:
        tally = {code: 0 for code in ERROR_CODES}
        correct = 0
        for window_id in sorted(dataset["windows"]):
            for candidate_id in errors[cell][window_id]:
                code = errors[cell][window_id][candidate_id]
                if code is None:
                    correct += 1
                else:
                    tally[code] += 1
        counts[cell] = {"correct": correct, "codes": tally}
    return counts


def strongest_features(
    models: Mapping[str, Mapping[int, tuple[Any, list[str]]]],
    *,
    cells: Sequence[str] = CELLS,
) -> dict[str, dict[str, Any]]:
    """Strongest positive/negative logistic coefficients and LightGBM gain
    importances from the fitted per-fold models (identical folds), aggregated
    across folds as the mean signed coefficient (positive ranked descending,
    negative ranked ascending) and the mean nonnegative gain respectively;
    ``folds_seen`` reports how many folds produced each feature.  Models
    themselves are never persisted."""
    output: dict[str, dict[str, Any]] = {}
    for cell in cells:
        model_name, _ = cell.split("_")
        per_fold: dict[str, dict[str, Any]] = {}
        positive: list[tuple[str, float]] = []
        negative: list[tuple[str, float]] = []
        gains: list[tuple[str, float]] = []
        for fold_index, (model, names) in models[cell].items():
            if model_name == "logistic":
                coefficients = model.coef_[0]
                order = np.argsort(coefficients)
                top_negative = [
                    (str(names[index]), float(coefficients[index]))
                    for index in order[:10]
                ]
                top_positive = [
                    (str(names[index]), float(coefficients[index]))
                    for index in order[-10:][::-1]
                ]
                positive.extend(top_positive)
                negative.extend(top_negative)
                per_fold[str(fold_index)] = {
                    "top_negative": top_negative,
                    "top_positive": top_positive,
                }
            else:
                importance = model.feature_importances_
                order = np.argsort(importance)[::-1][:20]
                top = [
                    (str(names[index]), float(importance[index]))
                    for index in order
                ]
                gains.extend(top)
                per_fold[str(fold_index)] = {"top_importances": top}
        if model_name == "logistic":
            output[cell] = {
                "kind": "logistic_coefficients",
                "per_fold": per_fold,
                "aggregate_top_positive": _aggregate_top(positive, 10),
                "aggregate_top_negative": _aggregate_top(
                    negative, 10, ascending=True,
                ),
            }
        else:
            output[cell] = {
                "kind": "lightgbm_gain_importance",
                "per_fold": per_fold,
                "aggregate_top_importances": _aggregate_top(gains, 20),
            }
    return output


def _aggregate_top(
    items: list[tuple[str, float]], limit: int, *, ascending: bool = False,
) -> list[dict[str, Any]]:
    """Rank features by mean value across folds (ties by feature name).

    Signed-sum ranking would favor fold frequency, so aggregation uses the
    mean: logistic positive coefficients descending, negative coefficients
    ascending (most negative first), LightGBM nonnegative gains descending.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, value in items:
        totals[name] = totals.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1
    means = {
        name: total / counts[name] for name, total in totals.items()
    }
    ranked = sorted(
        means.items(),
        key=(
            lambda item: (item[1], item[0])
            if ascending else (-item[1], item[0])
        ),
    )[:limit]
    return [
        {"feature": name, "mean": value, "folds_seen": counts[name]}
        for name, value in ranked
    ]


def validate_benchmark_mapping(benchmark: Mapping[str, Any]) -> None:
    """Verify a raw benchmark mapping against the canonical content hash and
    the trusted Phase 2F/2G lock and envelope.

    The canonical hash is recomputed over every field except
    ``content_sha256`` and must equal the declared value and the preregistered
    ``BENCHMARK_CONTENT_SHA256``; the mapping must also be the five-case
    ``LEGACY_FAILURE`` split.
    """
    if not isinstance(benchmark, Mapping):
        raise Phase2HError("benchmark must be a Mapping")
    if (
        benchmark.get("split") != "LEGACY_FAILURE"
        or not isinstance(benchmark.get("cases"), list)
        or len(benchmark["cases"]) != 5
    ):
        raise Phase2HError(
            "benchmark is not the five-case LEGACY_FAILURE split",
        )
    inner = {
        key: value for key, value in benchmark.items()
        if key != "content_sha256"
    }
    computed = canonical_sha256(inner)
    declared = benchmark.get("content_sha256")
    if declared != computed:
        raise Phase2HError(
            "benchmark content hash does not self-verify: "
            f"declared {declared!r} != recomputed {computed!r}",
        )
    if computed != BENCHMARK_CONTENT_SHA256:
        raise Phase2HError(
            "benchmark content does not match the preregistered lock",
        )


def run_experiment(
    benchmark: Mapping[str, Any] | str | Path,
    *,
    cells: Sequence[str] = CELLS,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full offline Phase 2H experiment (dataset, CV, metrics)."""
    if isinstance(benchmark, (str, Path)):
        benchmark = load_benchmark(benchmark)
    else:
        validate_benchmark_mapping(benchmark)
    dataset = build_dataset(benchmark)
    cv = run_cv(dataset, cells=cells, verbose=verbose)
    rankings = compute_rankings(dataset, cv["oof_scores"], cells=cells)
    errors = classify_all_errors(dataset, rankings, cells=cells)
    metrics = {
        cell: compute_cell_metrics(dataset, rankings, cell) for cell in cells
    }
    strongest = strongest_features(cv["fitted_models"], cells=cells)
    return {
        "dataset": dataset,
        "folds": cv["folds"],
        "fit_scope": cv["fit_scope"],
        "rankings": rankings,
        "errors": errors,
        "metrics": metrics,
        "strongest_features": strongest,
        "cells": list(cells),
    }


def _git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip())
    return commit, dirty


def _definition() -> dict[str, Any]:
    return {
        "run_version": RUN_VERSION,
        "task": (
            "candidate-level binary KEEP/DROP semantic endpoint scoring over "
            "the exhaustive source-grounded Phase 2F candidate universe"
        ),
        "keep_threshold": KEEP_THRESHOLD,
        "seed": SEED,
        "cells": list(CELLS),
        "model_configs": {
            "logistic": LOGISTIC_CONFIG,
            "lightgbm": {
                key: value for key, value in LGBM_CONFIG.items()
                if key != "class_weight"
            },
        },
        "feature_schema": feature_schema(),
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
        "metric_definitions": {
            "pooled": (
                "candidate-level out-of-fold metrics over every candidate "
                "scored when its window was held out"
            ),
            "recall_at_k": (
                "gold positives in top K / gold positives in that window; "
                "pooled with explicit summed denominators"
            ),
            "precision_at_k": (
                "gold positives in top K / min(K, candidate count); pooled "
                "with explicit summed denominators"
            ),
            "gold_rank": "1-based candidate rank inside the held-out window",
            "baselines": "all-DROP and all-KEEP predictions",
        },
        "no_llm": True,
        "no_semantic_edges": True,
        "optional_ud_syntax": "not part of this first implementation",
        "phase2g_clean_run_2_historical_metadata": {
            "purpose": "fixed report metadata only; never used as labels",
            "raw": {"recall": "8/33", "precision": "8/397"},
            "mechanical": {"recall": "4/33", "precision": "4/311"},
            "resolved": {"recall": "10/33", "precision": "10/765"},
        },
    }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for name in ("scikit-learn", "lightgbm", "numpy", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def build_window_table(
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]],
    errors: Mapping[str, Mapping[str, Mapping[str, str | None]]],
    window_id: str,
    *,
    cells: Sequence[str] = CELLS,
) -> dict[str, Any]:
    window = dataset["windows"][window_id]
    candidates: list[dict[str, Any]] = []
    for row in window["rows"]:
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
                cell: {
                    **rankings[window_id][cell][row.candidate_id],
                    "error_code": errors[cell][window_id][row.candidate_id],
                }
                for cell in cells
            },
        })
    return {
        "case_id": window["case_id"],
        "window_id": window["window_id"],
        "bronze_text_sha256": window["bronze_text_sha256"],
        "catalog_sha256": window["catalog_sha256"],
        "candidate_generator_version": window["candidate_generator_version"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_aggregate(
    benchmark_path: Path,
    result: Mapping[str, Any],
    *,
    repo: Path,
    created_at: str | None = None,
    window_table_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_path)
    benchmark_file_sha256 = hashlib.sha256(
        benchmark_path.read_bytes(),
    ).hexdigest()
    definition = _definition()
    commit, dirty = _git_state(repo)
    dataset_summary = validate_dataset(result["dataset"])
    catalog_hashes = {
        window_id: window["catalog_sha256"]
        for window_id, window in result["dataset"]["windows"].items()
    }
    errors = result["errors"]
    strongest = result.get("strongest_features")
    if window_table_hashes is None:
        raise Phase2HError(
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
        "created_at": created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": commit,
        "repository_dirty": dirty,
        "definition": definition,
        "definition_sha256": canonical_sha256(definition),
        "input_hashes": {
            "benchmark_content_sha256": BENCHMARK_CONTENT_SHA256,
            "benchmark_file_sha256": benchmark_file_sha256,
        },
        "benchmark_content_sha256": BENCHMARK_CONTENT_SHA256,
        "candidate_generator": {
            "version": MENTION_CATALOG_VERSION,
            "catalog_hashes": catalog_hashes,
        },
        "dataset_summary": dataset_summary,
        "folds": result["folds"],
        "fit_scope": result["fit_scope"],
        "metrics": result["metrics"],
        "error_taxonomy": error_taxonomy_counts(
            errors, result["dataset"], cells=result["cells"],
        ),
        "strongest_features": strongest or {},
        "dependencies": _dependency_versions(),
        "window_tables": table_hashes,
    }
    return {"content_sha256": canonical_sha256(inner), **inner}


def publish_artifact(
    output: Path,
    aggregate: Mapping[str, Any],
    window_tables: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Atomically publish the immutable Phase 2H artifact."""
    output = Path(output)
    if output.exists():
        raise ValueError("output directory already exists; artifacts are immutable")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=parent))
    files: list[Path] = []
    try:
        aggregate_path = temporary / "phase2h-endpoint-scoring.json"
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
        manifest = {
            "files": [
                {
                    "path": str(path.relative_to(temporary)),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(files, key=lambda item: str(item.relative_to(temporary)))
            ],
        }
        manifest_path = temporary / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files.append(manifest_path)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _load_aggregate(directory: Path) -> Mapping[str, Any]:
    return json.loads(
        (Path(directory) / "phase2h-endpoint-scoring.json").read_text(encoding="utf-8"),
    )


def _verify_artifact_files(directory: Path) -> list[str]:
    """Verify the MANIFEST and every listed file's bytes for one artifact."""
    directory = Path(directory)
    manifest_path = directory / "MANIFEST.json"
    if not manifest_path.is_file():
        return [f"{directory}: MANIFEST.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return [f"{directory}: MANIFEST.json has no files list"]
    problems: list[str] = []
    for entry in entries:
        relative = entry.get("path")
        expected = entry.get("file_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            problems.append(f"{directory}: MANIFEST.json file entry is invalid")
            continue
        path = directory / relative
        if not path.is_file():
            problems.append(f"{directory}: MANIFEST lists missing file {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(
                f"{directory}: file {relative} sha256 {actual} != "
                f"manifest {expected}",
            )
    return problems


def _verify_aggregate_integrity(
    directory: Path, body: Mapping[str, Any],
) -> list[str]:
    """Verify the aggregate self-hash and the window files against the
    window-table locks recorded inside the aggregate."""
    directory = Path(directory)
    problems: list[str] = []
    inner = {
        key: value for key, value in body.items() if key != "content_sha256"
    }
    if body.get("content_sha256") != canonical_sha256(inner):
        problems.append(
            f"{directory}: aggregate content_sha256 does not self-verify",
        )
    window_dir = directory / "windows"
    for window_id, info in (body.get("window_tables") or {}).items():
        window_path = window_dir / f"{window_id}.json"
        if not window_path.is_file():
            problems.append(
                f"{directory}: window table {window_id}.json is missing",
            )
            continue
        table = json.loads(window_path.read_text(encoding="utf-8"))
        expected = (info or {}).get("candidate_table_sha256")
        actual = canonical_sha256(table)
        if actual != expected:
            problems.append(
                f"{directory}: window table {window_id} content sha256 "
                f"{actual} != aggregate lock {expected}",
            )
    return problems


def _window_file_hashes(directory: Path) -> dict[str, str]:
    directory = Path(directory)
    window_dir = directory / "windows"
    if not window_dir.is_dir():
        return {}
    return {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(window_dir.glob("*.json"))
    }


def compare_artifacts(left: Path, right: Path) -> list[str]:
    """Compare two clean reruns while allowing timestamps to differ.

    Each artifact is first verified against its own MANIFEST (the aggregate
    and every window file's actual bytes) and against the aggregate's
    self-hash and window-table locks; then deterministic fields -- including
    ``git_commit`` and ``repository_dirty`` -- and the deterministic window
    file bytes are compared.  Any tampering yields a clear difference entry.
    """
    left = Path(left)
    right = Path(right)
    differences: list[str] = []
    differences.extend(_verify_artifact_files(left))
    differences.extend(_verify_artifact_files(right))
    try:
        left_body = _load_aggregate(left)
    except (OSError, ValueError) as error:
        left_body = {}
        differences.append(f"left: aggregate unreadable ({error})")
    try:
        right_body = _load_aggregate(right)
    except (OSError, ValueError) as error:
        right_body = {}
        differences.append(f"right: aggregate unreadable ({error})")
    differences.extend(_verify_aggregate_integrity(left, left_body))
    differences.extend(_verify_aggregate_integrity(right, right_body))
    for key in (
        "run_version", "git_commit", "repository_dirty", "definition_sha256",
        "input_hashes",
        "benchmark_content_sha256", "candidate_generator",
        "dataset_summary", "folds", "fit_scope", "metrics",
        "error_taxonomy", "strongest_features", "window_tables",
        "dependencies",
    ):
        if left_body.get(key) != right_body.get(key):
            differences.append(f"{key} differs")
    left_window_hashes = _window_file_hashes(left)
    right_window_hashes = _window_file_hashes(right)
    for window_id in sorted(
        set(left_window_hashes) | set(right_window_hashes),
    ):
        if left_window_hashes.get(window_id) != right_window_hashes.get(window_id):
            differences.append(
                f"window table file {window_id}.json bytes differ",
            )
    return differences
