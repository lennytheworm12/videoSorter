"""Phase 2K gate-locked paired downstream rerun machinery.

This module runs the existing Phase 2F generative semantic compiler and the
existing Phase 2H discriminative endpoint scorer over the exact same 30
Phase 2K windows under RAW_BRONZE vs CONTEXTUAL_POLISH, but only after every
Phase 2K human/audit/alignment gate passes:

- the Phase 2K output directory must be a deep-validated live build whose
  finalized human review gate recomputes to PASSED and whose completed
  transformation audit validates (reused from
  :mod:`pipeline.phase2k_downstream_alignment`);
- the finalized alignment packet must recompute to RELEASE_GATE REVIEWED with
  all 30 windows and all 311 endpoint IDs and must match the current Phase 2K
  state exactly;
- the alignment summary must be the deterministic summary of that packet.

No model/scorer call happens before those gates pass.  The live path emits
immutable canonical-hash envelopes (preflight input contract, generative
raw/polished artifacts, discriminative raw/polished all-cell artifacts, and a
comparison-input evidence artifact) into a caller-selected directory that
must not already exist; the directory is built in a temporary sibling and
atomically renamed only after complete validation, so a partial live/provider
failure never publishes a complete result directory.

The module is downstream only: it never performs mechanical cleanup or
semantic extraction, never edits Phase 2J/Phase 2K artifacts, and never
infers a human closeout decision, diagnosis, or note.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from pipeline.phase2k_contextual_reconstruction import (
    ROOT,
    canonical_sha256,
    load_json_strict,
    text_sha256,
)
import pipeline.phase2k_downstream_alignment as alignment
from pipeline.phase2k_downstream_alignment import (
    TARGET_COUNT,
    TARGET_WINDOW_COUNT,
    load_alignment_inputs,
    validate_alignment_summary,
    validate_downstream_alignment_packet,
)
from pipeline.phase2k_downstream_comparison import (
    DISCRIMINATIVE_ARCHITECTURE_FAMILY,
    DOWNSTREAM_DIAGNOSIS_VALUES,
    FINAL_CLOSEOUT_STATUSES,
    GENERATIVE_ARCHITECTURE_FAMILY,
    POLISHED_INPUT_REPRESENTATION,
    RAW_INPUT_REPRESENTATION,
    build_downstream_comparison,
    validate_downstream_comparison,
)
from pipeline.phase2h_endpoint_scoring import (
    CELLS,
    DROP,
    FEATURE_SCHEMA_VERSION,
    KEEP,
    KEEP_THRESHOLD,
    RUN_VERSION,
    SEED,
    CandidateRow,
    compute_rankings as phase2h_compute_rankings,
    run_cv as phase2h_run_cv,
    _window_metrics as phase2h_window_metrics,
)
from pipeline.semantic_compiler import (
    SemanticCompilerConfig,
    compile_source_semantic_ir,
)
from pipeline.semantic_ir_artifact import (
    SemanticRunArtifact,
    build_semantic_run_artifact,
)
from pipeline.semantic_mentions import (
    NODE_TYPES,
    generate_mention_candidates,
)
from pipeline.semantic_source import BronzeSource, window_from_exact_span


PREFLIGHT_SCHEMA_VERSION = "phase2k-downstream-rerun-preflight-contract-v1"
GENERATIVE_ARTIFACT_SCHEMA_VERSION = (
    "phase2k-downstream-rerun-generative-artifact-v1"
)
DISCRIMINATIVE_ARTIFACT_SCHEMA_VERSION = (
    "phase2k-downstream-rerun-discriminative-artifact-v1"
)
COMPARISON_INPUT_SCHEMA_VERSION = "phase2k-downstream-rerun-comparison-input-v1"

INPUT_ADAPTER_VERSION = "phase2k-downstream-input-adapter-v1"
SEMANTIC_TARGET_CONTRACT_VERSION = "phase2k-semantic-target-contract-v1"
EVALUATION_CONTRACT_VERSION = "phase2k-downstream-evaluation-contract-v1"

DEFAULT_PRIMARY_CELL = "logistic_B"

ARTIFACT_FILENAMES = {
    "preflight": "preflight-input-contract.json",
    "generative_raw": "generative-raw.json",
    "generative_polished": "generative-polished.json",
    "discriminative_raw": "discriminative-raw.json",
    "discriminative_polished": "discriminative-polished.json",
    "comparison_input": "comparison-input.json",
}

_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")

_ROW_KEYS = (
    "window_id",
    "target_count",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "output_count",
    "provenance_valid_count",
    "abstained",
    "output_sha256",
)

EVALUATION_CONTRACT = {
    "version": EVALUATION_CONTRACT_VERSION,
    "matching": (
        "exact local span; deterministic one-target-at-most-one-TP; a second "
        "node/candidate matching the alternatives of one target is FP"
    ),
    "node_type": (
        "required when the target node_type is non-null; wildcard when null"
    ),
    "provenance": (
        "an output counts as provenance-valid only when its exact local text "
        "equals the window slice"
    ),
    "discriminative": (
        "KEEP if score >= 0.5; no tuning; all four fixed Phase 2H cells run, "
        "comparison-v2 rows use only the declared primary cell"
    ),
}
EVALUATION_CONTRACT_SHA256 = canonical_sha256(EVALUATION_CONTRACT)


# ---------------------------------------------------------------------------
# Compiler execution identity
# ---------------------------------------------------------------------------


def _normalize_alias_tuple(values: Iterable[str], label: str) -> tuple[str, ...]:
    """Deterministic canonical alias tuple (same rule as the compiler)."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} aliases must be an iterable of strings")
    items = tuple(values)
    if any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in items
    ):
        raise ValueError(f"{label} aliases must be non-empty trimmed strings")
    return tuple(sorted(set(items), key=lambda item: (item.casefold(), item)))


def build_compiler_execution_sha256(
    config: SemanticCompilerConfig,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
) -> str:
    """Hash of the exact compiler execution descriptor, aliases included.

    The descriptor includes the full frozen config plus the exact sorted
    entity/ability aliases, so a resealed artifact whose alias set differs
    from the preflight identity can never validate.
    """
    return canonical_sha256({
        **asdict(config),
        "entity_aliases": list(
            _normalize_alias_tuple(entity_aliases, "entity"),
        ),
        "ability_aliases": list(
            _normalize_alias_tuple(ability_aliases, "ability"),
        ),
    })


# ---------------------------------------------------------------------------
# Small strict helpers
# ---------------------------------------------------------------------------


def _require_exact_keys(value: object, expected: Iterable[str], label: str) -> None:
    expected_set = frozenset(expected)
    if not isinstance(value, Mapping) or set(value) != expected_set:
        missing = (
            sorted(expected_set - set(value)) if isinstance(value, Mapping) else []
        )
        extra = sorted(set(value) - expected_set) if isinstance(value, Mapping) else []
        detail = (
            f"missing={missing} extra={extra}"
            if isinstance(value, Mapping)
            else "not an object"
        )
        raise ValueError(f"{label} key set is invalid: {detail}")


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _require_nonempty_string(value: object, label: str) -> str:
    text = _require_string(value, label)
    if not text.strip():
        raise ValueError(f"{label} must be non-empty")
    return text


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_enum(value: object, options: Iterable[str], label: str) -> str:
    text = _require_string(value, label)
    if text not in frozenset(options):
        raise ValueError(f"{label} has invalid value {text!r}")
    return text


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_hex64(value: object, label: str) -> str:
    text = _require_string(value, label)
    if _HEX64.fullmatch(text) is None:
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return text


def _validate_recomputed_content_hash(obj: Mapping[str, Any], *, label: str) -> None:
    _require_hex64(obj.get("content_sha256"), f"{label} content_sha256")
    expected = canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })
    if obj["content_sha256"] != expected:
        raise ValueError(f"{label} content_sha256 does not match canonical content")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _write_exact(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


# ---------------------------------------------------------------------------
# Gate-locked input loading
# ---------------------------------------------------------------------------


def load_rerun_inputs(
    *,
    phase2k_dir: Path,
    alignment_packet_path: Path,
    alignment_summary_path: Path,
    reviewed_packet_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    """Deeply validate every Phase 2K/2J/alignment gate the rerun consumes.

    Reuses the alignment module's source-bound loading/validation, then
    additionally requires the finalized alignment packet (release gate
    REVIEWED, exact 30 windows / 311 endpoints, current-source bindings) and
    its deterministic summary.  No provider/scorer is called here.
    """
    inputs = load_alignment_inputs(
        phase2k_dir=phase2k_dir,
        reviewed_packet_path=reviewed_packet_path,
        coverage_path=coverage_path,
    )
    packet = load_json_strict(
        alignment_packet_path, label="phase2k downstream alignment packet",
    )
    summary = load_json_strict(
        alignment_summary_path, label="phase2k downstream alignment summary",
    )
    validated_packet = validate_downstream_alignment_packet(
        packet, require_blank=False, bindings=inputs,
    )
    validated_summary = validate_alignment_summary(
        summary, finalized=validated_packet,
    )
    if validated_summary["alignment_packet_sha256"] != validated_packet[
        "content_sha256"
    ]:
        raise ValueError(
            "phase2k alignment summary is not bound to the alignment packet",
        )
    by_window: dict[str, list[dict[str, Any]]] = {}
    for item in validated_packet["items"]:
        by_window.setdefault(item["window_id"], []).append(item)
    if len(by_window) != TARGET_WINDOW_COUNT or any(
        len(items) == 0 for items in by_window.values()
    ):
        raise ValueError(
            f"phase2k alignment packet must span exactly {TARGET_WINDOW_COUNT} "
            "windows",
        )
    return {
        **inputs,
        "alignment_packet_path": alignment_packet_path,
        "alignment_summary_path": alignment_summary_path,
        "alignment_packet": validated_packet,
        "alignment_summary": validated_summary,
        "items_by_window": by_window,
    }


# ---------------------------------------------------------------------------
# Exact source-bound representation adapters
# ---------------------------------------------------------------------------


def _phase2j_source_id_by_window(inputs: Mapping[str, Any]) -> dict[str, str]:
    by_window: dict[str, str] = {}
    for record in inputs["reviewed_packet"]["records"]:
        by_window[record["window_id"]] = record["upstream_source_id"]
    return by_window


def _record_by_window(
    inputs: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    by_window: dict[str, Mapping[str, Any]] = {}
    for record in inputs["records_obj"]["records"]:
        if record["record_type"] == "D":
            by_window[record["window_id"]] = record
    return by_window


def _window_texts(inputs: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Exact sealed texts per window: raw Bronze and contextual polish.

    The D clean transcript is the reconstructed Pass 1 text and may differ
    from Bronze by design; it must exactly match the D record's own sealed
    reconstruction subobject (content and hash).  The D target block must
    also exactly match the A raw-Bronze target block and the Phase 2J
    reviewed packet's source/window identity (source group, absolute source
    span), binding the adapter to the exact source/window/target identity.
    """
    records = inputs["records_obj"]["records"]
    a_by_window = {
        record["window_id"]: record for record in records
        if record["record_type"] == "A"
    }
    d_by_window = _record_by_window(inputs)
    reviewed_by_window = {
        record["window_id"]: record
        for record in inputs["reviewed_packet"]["records"]
    }
    texts: dict[str, dict[str, str]] = {}
    for window_id, d_record in d_by_window.items():
        target = d_record["target"]
        d_content = d_record["content"]
        raw_text = target["text"]
        a_record = a_by_window.get(window_id)
        if a_record is None:
            raise ValueError(
                f"phase2k A record is missing for window {window_id}",
            )
        a_content = a_record["content"]
        if d_record.get("window_id") != window_id or a_record.get(
            "window_id",
        ) != window_id:
            raise ValueError(
                f"phase2k A/D record window identity is invalid for "
                f"{window_id}",
            )
        if a_record.get("target") != target:
            raise ValueError(
                f"phase2k D target does not exactly match the A target for "
                f"{window_id}",
            )
        if target.get("window_id") != window_id:
            raise ValueError(
                f"phase2k A/D target window binding is invalid for "
                f"{window_id}",
            )
        reviewed = reviewed_by_window[window_id]
        if (
            target.get("source_group_id") != reviewed.get("source_group_id")
            or target.get("source_absolute_start")
            != reviewed.get("upstream_start")
            or target.get("source_absolute_end")
            != reviewed.get("upstream_end")
        ):
            raise ValueError(
                f"phase2k A/D target source binding does not match the "
                f"Phase 2J reviewed packet for {window_id}",
            )
        if a_content.get("kind") != "raw_bronze":
            raise ValueError(f"phase2k A record is not raw Bronze for {window_id}")
        if a_content.get("text") != raw_text:
            raise ValueError(
                f"phase2k A content text does not match the D target for "
                f"{window_id}",
            )
        if a_content.get("text_sha256") != text_sha256(raw_text):
            raise ValueError(f"phase2k A text hash is invalid for {window_id}")
        if target.get("text_sha256") != text_sha256(raw_text):
            raise ValueError(
                f"phase2k D target text hash is invalid for {window_id}",
            )
        if d_content.get("generation_status") != "GENERATED":
            raise ValueError(
                f"phase2k D record is not GENERATED for {window_id}",
            )
        reconstruction = d_content.get("reconstruction")
        if not isinstance(reconstruction, Mapping):
            raise ValueError(
                f"phase2k D record is missing its reconstruction artifact for "
                f"{window_id}",
            )
        clean_text = d_content.get("clean_target_transcript")
        if not isinstance(clean_text, str) or not clean_text:
            raise ValueError(
                f"phase2k D clean transcript is invalid for {window_id}",
            )
        expected_clean_hash = text_sha256(clean_text)
        if reconstruction.get("clean_target_transcript") != clean_text:
            raise ValueError(
                f"phase2k D clean transcript does not match its sealed "
                f"reconstruction for {window_id}",
            )
        _require_hex64(
            d_content.get("clean_target_transcript_sha256"),
            f"phase2k D clean transcript hash for {window_id}",
        )
        _require_hex64(
            reconstruction.get("clean_target_transcript_sha256"),
            f"phase2k D reconstruction hash for {window_id}",
        )
        if d_content.get("clean_target_transcript_sha256") != expected_clean_hash:
            raise ValueError(
                f"phase2k D clean transcript hash is invalid for {window_id}",
            )
        if reconstruction.get(
            "clean_target_transcript_sha256",
        ) != expected_clean_hash:
            raise ValueError(
                f"phase2k D reconstruction hash is invalid for {window_id}",
            )
        semantic_polish = d_content.get("semantic_polish")
        if not isinstance(semantic_polish, Mapping):
            raise ValueError(f"phase2k D semantic polish is missing for {window_id}")
        polished_text = semantic_polish.get("polished_text")
        if not isinstance(polished_text, str) or not polished_text:
            raise ValueError(
                f"phase2k D semantic polish text is invalid for {window_id}",
            )
        texts[window_id] = {
            "raw": raw_text,
            "polished": polished_text,
        }
    return texts


def build_input_adapters(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic RAW_BRONZE / CONTEXTUAL_POLISH window adapters.

    Both adapters are built with :func:`BronzeSource` +
    :func:`window_from_exact_span` exactly like the Phase 2J coverage module,
    so the adapter windows are deterministic and source-bound.  The raw
    adapter text is the sealed D/A Bronze target text; the polished adapter
    text is the sealed D ``semantic_polish.polished_text``.  Raw and polished
    adapter content hashes always differ.
    """
    texts = _window_texts(inputs)
    source_ids = _phase2j_source_id_by_window(inputs)
    validated_packet = inputs["alignment_packet"]
    raw_descriptors: list[dict[str, Any]] = []
    polished_descriptors: list[dict[str, Any]] = []
    by_window: dict[str, dict[str, Any]] = {}
    for window_id in sorted(texts):
        raw_text = texts[window_id]["raw"]
        polished_text = texts[window_id]["polished"]
        upstream_source_id = source_ids[window_id]
        raw_source = BronzeSource(
            f"transcript:{upstream_source_id}", raw_text,
        )
        raw_window = window_from_exact_span(raw_source, 0, len(raw_text))
        polished_source_id = f"polished:{window_id}"
        polished_source = BronzeSource(polished_source_id, polished_text)
        polished_window = window_from_exact_span(
            polished_source, 0, len(polished_text),
        )
        raw_descriptor = {
            "adapter_version": INPUT_ADAPTER_VERSION,
            "representation": RAW_INPUT_REPRESENTATION,
            "phase2k_window_id": window_id,
            "window_id": raw_window.window_id,
            "source_id": raw_source.source_id,
            "source_kind": raw_source.source_kind,
            "text_sha256": text_sha256(raw_text),
            "span": [raw_window.source_start, raw_window.source_end],
        }
        polished_descriptor = {
            "adapter_version": INPUT_ADAPTER_VERSION,
            "representation": POLISHED_INPUT_REPRESENTATION,
            "phase2k_window_id": window_id,
            "window_id": polished_window.window_id,
            "source_id": polished_source.source_id,
            "source_kind": polished_source.source_kind,
            "text_sha256": text_sha256(polished_text),
            "span": [polished_window.source_start, polished_window.source_end],
        }
        # Verify every accepted target span against the exact adapter text.
        for item in validated_packet["items"]:
            if item["window_id"] != window_id:
                continue
            bronze = item["bronze_target"]
            raw_span = (bronze["evaluation_start"], bronze["evaluation_end"])
            if raw_window.text[raw_span[0]:raw_span[1]] != bronze[
                "evaluation_text"
            ]:
                raise ValueError(
                    f"alignment item {item['alignment_id']} raw evaluation "
                    "span is not an exact slice of the Bronze window",
                )
            for span in item["decision"]["polished_spans"]:
                if polished_window.text[span["start"]:span["end"]] != span["text"]:
                    raise ValueError(
                        f"alignment item {item['alignment_id']} polished span "
                        "is not an exact slice of the polished window",
                    )
        raw_descriptors.append(raw_descriptor)
        polished_descriptors.append(polished_descriptor)
        by_window[window_id] = {
            "raw": {"window": raw_window, "descriptor": raw_descriptor},
            "polished": {"window": polished_window, "descriptor": polished_descriptor},
        }
    raw_sha = canonical_sha256(raw_descriptors)
    polished_sha = canonical_sha256(polished_descriptors)
    if raw_sha == polished_sha:
        raise ValueError("raw and polished input adapter hashes must differ")
    return {
        "raw": {
            "adapter_sha256": raw_sha,
            "windows": {
                descriptor["phase2k_window_id"]: descriptor
                for descriptor in raw_descriptors
            },
        },
        "polished": {
            "adapter_sha256": polished_sha,
            "windows": {
                descriptor["phase2k_window_id"]: descriptor
                for descriptor in polished_descriptors
            },
        },
        "by_window": by_window,
    }


def _adapter_window_dict(window: Any) -> dict[str, Any]:
    """Canonical dict form of a semantic source window (artifact schema)."""
    return {
        "window_id": window.window_id,
        "source_id": window.source_id,
        "source_kind": window.source_kind,
        "source_start": window.source_start,
        "source_end": window.source_end,
        "text": window.text,
        "source_content_sha256": window.source_content_sha256,
        "source_provenance_sha256": window.source_provenance_sha256,
        "source_context_sha256": window.source_context_sha256,
        "speaker": window.speaker,
        "start_ms": window.start_ms,
        "end_ms": window.end_ms,
        "metadata": [list(item) for item in window.metadata],
        "segments": [
            {
                "segment_id": segment.segment_id,
                "window_id": segment.window_id,
                "kind": segment.kind,
                "start": segment.start,
                "end": segment.end,
                "absolute_start": segment.absolute_start,
                "absolute_end": segment.absolute_end,
                "source_text": segment.source_text,
                "version": segment.version,
            }
            for segment in window.segments
        ],
        "version": window.version,
    }


# ---------------------------------------------------------------------------
# Dataset/semantic-target bindings
# ---------------------------------------------------------------------------


def build_dataset_binding(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Exact Phase 2K dataset binding used by the v2 comparison envelope."""
    return {
        "phase2k_records_sha256": _require_hex64(
            inputs["records_obj"].get("content_sha256"),
            "phase2k records content hash",
        ),
        "finalized_human_packet_sha256": _require_hex64(
            inputs["finalized_packet"].get("content_sha256"),
            "finalized human packet content hash",
        ),
        "human_summary_sha256": canonical_sha256(inputs["human_summary"]),
        "completed_transformation_audit_sha256": _require_hex64(
            inputs["completed_audit"].get("content_sha256"),
            "completed transformation audit content hash",
        ),
        "window_ids_sha256": canonical_sha256(inputs["window_ids"]),
        "window_count": len(inputs["window_ids"]),
        "human_review_gate_status": "PASSED",
    }


def build_semantic_target_contract(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic semantic target contract bound to all 311 items."""
    items = inputs["alignment_packet"]["items"]
    target_items = [
        {
            "alignment_id": item["alignment_id"],
            "window_id": item["window_id"],
            "endpoint_id": item["endpoint_id"],
            "node_type": item["node_type"],
            "bronze_target": item["bronze_target"],
            "decision": item["decision"],
        }
        for item in items
    ]
    return {
        "contract_version": SEMANTIC_TARGET_CONTRACT_VERSION,
        "contract_sha256": canonical_sha256(target_items),
        "target_count": len(items),
        "boundary_rule": (
            "phase2k-target-boundary-rule-v1-phase2j-terminal-punctuation "
            "(263 unchanged, 48 corrected)"
        ),
    }


def _scorer_config_descriptor(primary_cell: str) -> dict[str, Any]:
    _require_enum(primary_cell, CELLS, "primary cell")
    return {
        "run_version": RUN_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "keep_threshold": KEEP_THRESHOLD,
        "seed": SEED,
        "cells": list(CELLS),
        "primary_cell": primary_cell,
    }


def build_scorer_config_sha256(primary_cell: str) -> str:
    return canonical_sha256(_scorer_config_descriptor(primary_cell))


# ---------------------------------------------------------------------------
# Deterministic target evaluation (one-target-at-most-one-TP)
# ---------------------------------------------------------------------------


def _representation_targets(
    inputs: Mapping[str, Any],
    representation: str,
) -> dict[str, list[dict[str, Any]]]:
    """Per-window ordered targets with accepted exact local spans."""
    targets_by_window: dict[str, list[dict[str, Any]]] = {}
    for item in inputs["alignment_packet"]["items"]:
        if representation == RAW_INPUT_REPRESENTATION:
            bronze = item["bronze_target"]
            accepted = [(bronze["evaluation_start"], bronze["evaluation_end"])]
        else:
            state = item["decision"]["state"]
            accepted = (
                [(span["start"], span["end"]) for span in item["decision"]["polished_spans"]]
                if state in ("ALIGNED", "MULTIPLE_CANDIDATES")
                else []
            )
        targets_by_window.setdefault(item["window_id"], []).append({
            "alignment_id": item["alignment_id"],
            "endpoint_id": item["endpoint_id"],
            "node_type": item["node_type"],
            "accepted_spans": accepted,
        })
    return targets_by_window


def evaluate_targets(
    *,
    window_id: str,
    targets: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    window_text: str,
    require_exact_node_type: bool = True,
) -> dict[str, Any]:
    """Deterministic one-target-at-most-one-TP evaluation over exact spans.

    ``outputs`` is an ordered sequence of
    ``{"output_id", "span": (start, end), "text", "node_type"}``.  A target
    matches the first unconsumed output whose span is one of its accepted
    spans and whose node type equals the target node type when the target
    node type is non-null (wildcard when null).  With
    ``require_exact_node_type=True`` (generative family) an output whose
    ``node_type`` is ``None`` never wildcards a typed target; with
    ``require_exact_node_type=False`` (discriminative family) matching is
    span-only and node type is ignored entirely.  Extra outputs (including a
    second output matching another alternative of one target) are FP.
    """
    target_count = len(targets)
    consumed: set[int] = set()
    per_target: list[dict[str, Any]] = []
    for target in targets:
        matched_index: int | None = None
        for index, output in enumerate(outputs):
            if index in consumed:
                continue
            span_matches = output["span"] in target["accepted_spans"]
            type_matches = _target_type_matches(
                target["node_type"],
                output["node_type"],
                require_exact_node_type=require_exact_node_type,
            )
            if span_matches and type_matches:
                matched_index = index
                break
        if matched_index is not None:
            consumed.add(matched_index)
        per_target.append({
            "alignment_id": target["alignment_id"],
            "accepted_spans": [
                [start, end] for start, end in target["accepted_spans"]
            ],
            "node_type": target["node_type"],
            "matched_output_id": (
                outputs[matched_index]["output_id"]
                if matched_index is not None
                else None
            ),
            "tp": matched_index is not None,
        })
    true_positive = len(consumed)
    output_count = len(outputs)
    false_positive = output_count - true_positive
    false_negative = target_count - true_positive
    provenance_valid = sum(
        1
        for output in outputs
        if (
            output["span"][0] >= 0
            and output["span"][1] <= len(window_text)
            and output["text"] == window_text[
                output["span"][0]:output["span"][1]
            ]
        )
    )
    output_sha256 = canonical_sha256([
        output["output_id"] for output in outputs
    ])
    row = {
        "window_id": window_id,
        "target_count": target_count,
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "output_count": output_count,
        "provenance_valid_count": provenance_valid,
        "abstained": output_count == 0,
        "output_sha256": output_sha256,
    }
    validate_row(row, expected_window_id=window_id)
    return {
        "row": row,
        "per_target": per_target,
        "matched_output_ids": sorted(
            outputs[index]["output_id"] for index in consumed
        ),
        "output_ids": [output["output_id"] for output in outputs],
    }


def _target_type_matches(
    target_node_type: str | None,
    output_node_type: str | None,
    *,
    require_exact_node_type: bool,
) -> bool:
    if not require_exact_node_type:
        return True
    if target_node_type is None:
        return True
    return output_node_type == target_node_type


def validate_row(row: object, *, expected_window_id: str) -> dict[str, Any]:
    _require_exact_keys(row, _ROW_KEYS, "phase2k downstream rerun row")
    window_id = _require_nonempty_string(row["window_id"], "row window_id")
    if window_id != expected_window_id:
        raise ValueError("row window_id does not match the dataset window")
    target_count = _require_int(
        row["target_count"], "row target_count", minimum=0,
    )
    true_positive = _require_int(
        row["true_positive_count"], "row true_positive_count", minimum=0,
    )
    false_positive = _require_int(
        row["false_positive_count"], "row false_positive_count", minimum=0,
    )
    false_negative = _require_int(
        row["false_negative_count"], "row false_negative_count", minimum=0,
    )
    output_count = _require_int(
        row["output_count"], "row output_count", minimum=0,
    )
    provenance_valid = _require_int(
        row["provenance_valid_count"],
        "row provenance_valid_count",
        minimum=0,
    )
    _require_bool(row["abstained"], "row abstained")
    _require_hex64(row["output_sha256"], "row output_sha256")
    if true_positive + false_negative != target_count:
        raise ValueError("row true_positive + false_negative must equal target_count")
    if true_positive + false_positive != output_count:
        raise ValueError("row true_positive + false_positive must equal output_count")
    if provenance_valid > output_count:
        raise ValueError("row provenance_valid_count must not exceed output_count")
    if row["abstained"] != (output_count == 0):
        raise ValueError("row abstained must be true iff output_count == 0")
    return dict(row)


# ---------------------------------------------------------------------------
# Preflight / input contract
# ---------------------------------------------------------------------------


def _preflight_payload(
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    config: SemanticCompilerConfig,
    primary_cell: str,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
) -> dict[str, Any]:
    """Unsealed preflight payload with no predictions or result rows."""
    entity_aliases = _normalize_alias_tuple(entity_aliases, "entity")
    ability_aliases = _normalize_alias_tuple(ability_aliases, "ability")
    dataset_binding = build_dataset_binding(inputs)
    contract = build_semantic_target_contract(inputs)
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "purpose": (
            "Gate-locked Phase 2K downstream paired-rerun input contract.  "
            "Carries exact dataset/semantic-target bindings, deterministic "
            "raw/polished input adapters, and the declared compiler/scorer "
            "configuration; contains no predictions, model results, or "
            "fabricated rows."
        ),
        "predictions": False,
        "dataset_binding": dataset_binding,
        "semantic_target_contract": contract,
        "compiler": {
            "config_sha256": canonical_sha256(asdict(config)),
            "execution_sha256": build_compiler_execution_sha256(
                config,
                entity_aliases=entity_aliases,
                ability_aliases=ability_aliases,
            ),
            "config": asdict(config),
            "entity_aliases": list(entity_aliases),
            "ability_aliases": list(ability_aliases),
        },
        "discriminative": _scorer_config_descriptor(primary_cell),
        "adapters": {
            "raw": {
                "adapter_sha256": adapters["raw"]["adapter_sha256"],
                "window_count": len(adapters["raw"]["windows"]),
                "windows": adapters["raw"]["windows"],
            },
            "polished": {
                "adapter_sha256": adapters["polished"]["adapter_sha256"],
                "window_count": len(adapters["polished"]["windows"]),
                "windows": adapters["polished"]["windows"],
            },
        },
        "gates": {
            "human_review_gate_status": "PASSED",
            "alignment_release_gate": "REVIEWED",
            "alignment_packet_sha256": inputs["alignment_packet"]["content_sha256"],
            "alignment_summary_sha256": canonical_sha256(inputs["alignment_summary"]),
            "target_count": TARGET_COUNT,
            "window_count": TARGET_WINDOW_COUNT,
        },
    }


def build_preflight_contract(
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    config: SemanticCompilerConfig,
    primary_cell: str,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
) -> dict[str, Any]:
    """Sealed input contract with no predictions or result rows."""
    payload = _preflight_payload(
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
    )
    envelope = {"content_sha256": canonical_sha256(payload), **payload}
    validate_preflight_contract(
        envelope,
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
    )
    return envelope


def validate_preflight_contract(
    value: object,
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    config: SemanticCompilerConfig,
    primary_cell: str,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("phase2k rerun preflight contract must be an object")
    _require_exact_keys(
        value,
        (
            "schema_version",
            "content_sha256",
            "purpose",
            "predictions",
            "dataset_binding",
            "semantic_target_contract",
            "compiler",
            "discriminative",
            "adapters",
            "gates",
        ),
        "phase2k rerun preflight contract",
    )
    if value["schema_version"] != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("phase2k rerun preflight schema version is invalid")
    _validate_recomputed_content_hash(
        value, label="phase2k rerun preflight contract",
    )
    _require_bool(value["predictions"], "preflight predictions")
    if value["predictions"]:
        raise ValueError("preflight input contract must carry no predictions")
    payload = _preflight_payload(
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
    )
    expected = {"content_sha256": canonical_sha256(payload), **payload}
    if value != expected:
        raise ValueError("preflight input contract does not match its sources")
    return dict(value)


# ---------------------------------------------------------------------------
# Generative (Phase 2F) artifacts
# ---------------------------------------------------------------------------


def _serialized_run_nodes(run_artifact_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    run = run_artifact_payload.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("semantic run artifact is missing its run payload")
    nodes = run.get("mention_nodes")
    if not isinstance(nodes, list):
        raise ValueError("semantic run artifact mention_nodes are invalid")
    outputs: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("semantic run node is malformed")
        span = node.get("source_span")
        if not isinstance(span, Mapping):
            raise ValueError("semantic run node span is malformed")
        outputs.append({
            "output_id": node.get("node_id"),
            "span": (span.get("local_start"), span.get("local_end")),
            "text": span.get("text"),
            "node_type": node.get("node_type"),
        })
    return outputs


def _evaluate_generative_windows(
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    representation: str,
    window_run_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    targets_by_window = _representation_targets(inputs, representation)
    windows: list[dict[str, Any]] = []
    for phase2k_window_id in sorted(targets_by_window):
        adapter = adapters["by_window"][phase2k_window_id][
            "raw" if representation == RAW_INPUT_REPRESENTATION else "polished"
        ]
        window_text = adapter["window"].text
        run_artifact = window_run_artifacts[phase2k_window_id]
        outputs = _serialized_run_nodes(run_artifact)
        evaluated = evaluate_targets(
            window_id=phase2k_window_id,
            targets=targets_by_window[phase2k_window_id],
            outputs=outputs,
            window_text=window_text,
            require_exact_node_type=True,
        )
        windows.append({
            "phase2k_window_id": phase2k_window_id,
            "adapter_window_id": adapter["window"].window_id,
            "target_count": len(targets_by_window[phase2k_window_id]),
            "row": evaluated["row"],
            "per_target": evaluated["per_target"],
            "matched_output_ids": evaluated["matched_output_ids"],
            "output_ids": evaluated["output_ids"],
        })
    return {"windows": windows}


def build_generative_artifacts(
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    config: SemanticCompilerConfig,
    chat: Callable[..., str],
    created_at: str,
    git_commit: str,
    repository_dirty: bool,
    compiler: Callable[..., Any] = compile_source_semantic_ir,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the Phase 2F compiler on both representations and seal artifacts.

    Returns ``(raw_artifact, polished_artifact, raw_evidence, polished_evidence)``.
    Every window run is preserved through the existing Phase 2F semantic run
    artifact serialization, including full typed run evidence and raw provider
    lineage.
    """
    entity_aliases = _normalize_alias_tuple(entity_aliases, "entity")
    ability_aliases = _normalize_alias_tuple(ability_aliases, "ability")
    artifacts: dict[str, dict[str, Any]] = {}
    evidences: dict[str, dict[str, Any]] = {}
    for representation, adapter_key in (
        (RAW_INPUT_REPRESENTATION, "raw"),
        (POLISHED_INPUT_REPRESENTATION, "polished"),
    ):
        window_runs: dict[str, Any] = {}
        for phase2k_window_id in sorted(adapters["by_window"]):
            adapter = adapters["by_window"][phase2k_window_id][adapter_key]
            run = compiler(
                adapter["window"],
                chat,
                config=config,
                entity_aliases=entity_aliases,
                ability_aliases=ability_aliases,
            )
            if not hasattr(run, "validate"):
                raise ValueError("phase2k compiler must return a typed run")
            run.validate()
            provider_failures = [
                failure for failure in run.failures
                if failure.code == "PROVIDER_FAILURE"
            ]
            if provider_failures:
                raise ValueError(
                    "phase2k generative provider failure aborts the whole "
                    f"downstream rerun for {phase2k_window_id}: "
                    f"{provider_failures[0].detail}",
                )
            run_artifact = build_semantic_run_artifact(
                run,
                git_commit=git_commit,
                repository_dirty=repository_dirty,
                created_at=created_at,
                input_hashes={
                    "phase2k_records_sha256": inputs["records_obj"][
                        "content_sha256"
                    ],
                    "phase2k_alignment_packet_sha256": inputs["alignment_packet"][
                        "content_sha256"
                    ],
                    "phase2k_input_adapter_sha256": adapters[adapter_key][
                        "adapter_sha256"
                    ],
                },
            )
            window_runs[phase2k_window_id] = run_artifact.payload
        evidence = _evaluate_generative_windows(
            inputs, adapters, representation, window_runs,
        )
        payload = {
            "schema_version": GENERATIVE_ARTIFACT_SCHEMA_VERSION,
            "architecture_family": GENERATIVE_ARCHITECTURE_FAMILY,
            "input_representation": representation,
            "input_adapter_sha256": adapters[adapter_key]["adapter_sha256"],
            "compiler_config_sha256": build_compiler_execution_sha256(
                config,
                entity_aliases=entity_aliases,
                ability_aliases=ability_aliases,
            ),
            "compiler_config": asdict(config),
            "entity_aliases": list(entity_aliases),
            "ability_aliases": list(ability_aliases),
            "created_at": created_at,
            "git_commit": git_commit,
            "repository_dirty": repository_dirty,
            "dataset_binding": build_dataset_binding(inputs),
            "semantic_target_contract_sha256": build_semantic_target_contract(
                inputs,
            )["contract_sha256"],
            "evaluation_contract_sha256": EVALUATION_CONTRACT_SHA256,
            "windows": evidence["windows"],
            "run_artifacts": window_runs,
        }
        artifact = {"content_sha256": canonical_sha256(payload), **payload}
        artifacts[representation] = artifact
        evidences[representation] = evidence
    return (
        artifacts[RAW_INPUT_REPRESENTATION],
        artifacts[POLISHED_INPUT_REPRESENTATION],
        evidences[RAW_INPUT_REPRESENTATION],
        evidences[POLISHED_INPUT_REPRESENTATION],
    )


# ---------------------------------------------------------------------------
# Discriminative (Phase 2H) artifacts
# ---------------------------------------------------------------------------


def _accepted_spans_by_window(
    inputs: Mapping[str, Any],
    representation: str,
) -> dict[str, dict[tuple[int, int], list[str]]]:
    accepted: dict[str, dict[tuple[int, int], list[str]]] = {}
    for window_id, targets in _representation_targets(
        inputs, representation,
    ).items():
        window_spans: dict[tuple[int, int], list[str]] = {}
        for target in targets:
            for span in target["accepted_spans"]:
                window_spans.setdefault(tuple(span), []).append(
                    target["alignment_id"],
                )
        accepted[window_id] = window_spans
    return accepted


def build_candidate_dataset(
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    representation: str,
) -> dict[str, Any]:
    """Custom CandidateRow dataset over the frozen Phase 2F candidates.

    Candidate training labels are KEEP for every accepted alternative exact
    span and DROP otherwise.  Provenance is exact by construction: rows are
    generated from the frozen ``generate_mention_candidates`` API over the
    deterministic adapter windows.
    """
    adapter_key = "raw" if representation == RAW_INPUT_REPRESENTATION else "polished"
    accepted = _accepted_spans_by_window(inputs, representation)
    windows: dict[str, dict[str, Any]] = {}
    for phase2k_window_id in sorted(accepted):
        adapter = adapters["by_window"][phase2k_window_id][adapter_key]
        window = adapter["window"]
        candidates = generate_mention_candidates(window)
        window_accepted = accepted[phase2k_window_id]
        window_targets = _representation_targets(
            inputs, representation,
        )[phase2k_window_id]
        node_type_by_span: dict[tuple[int, int], list[str]] = {}
        for item in window_targets:
            for item_span in item["accepted_spans"]:
                if item["node_type"] is not None:
                    node_type_by_span.setdefault(tuple(item_span), []).append(
                        item["node_type"],
                    )
        rows: list[CandidateRow] = []
        width = max(3, len(str(len(candidates))))
        for index, candidate in enumerate(candidates, 1):
            span = (candidate.start, candidate.end)
            gold_ids = window_accepted.get(span, ())
            is_positive = bool(gold_ids)
            node_types = sorted(set(node_type_by_span.get(span, ())))
            rows.append(CandidateRow(
                case_id=phase2k_window_id,
                window_id=window.window_id,
                candidate_id=candidate.candidate_id,
                alias=f"c{index:0{width}d}",
                start=candidate.start,
                end=candidate.end,
                absolute_start=candidate.absolute_start,
                absolute_end=candidate.absolute_end,
                text=candidate.source_text,
                segment_ids=candidate.segment_ids,
                segment_bounds=tuple(
                    (segment.start, segment.end)
                    for segment in window.segments
                    if segment.segment_id in candidate.segment_ids
                ),
                type_hints=candidate.type_hints,
                source_kind=window.source_kind,
                is_gold_positive=is_positive,
                label=KEEP if is_positive else DROP,
                excluded=False,
                ambiguity_state="NONE",
                gold_mention_ids=tuple(sorted(gold_ids)),
                gold_node_types=node_types,
            ))
        windows[window.window_id] = {
            "window_id": window.window_id,
            "phase2k_window_id": phase2k_window_id,
            "bronze_text": window.text,
            "accepted_spans": {
                f"{start}:{end}": ids
                for (start, end), ids in sorted(window_accepted.items())
            },
            "rows": tuple(rows),
        }
    dataset = {"windows": windows}
    _validate_candidate_dataset(dataset, accepted)
    return dataset


def _validate_candidate_dataset(
    dataset: Mapping[str, Any],
    accepted: Mapping[str, Mapping[tuple[int, int], list[str]]],
) -> None:
    windows = dataset["windows"]
    if not isinstance(windows, Mapping) or not windows:
        raise ValueError("phase2k rerun candidate dataset is empty")
    for window in windows.values():
        rows = window["rows"]
        if not isinstance(rows, tuple) or not rows:
            raise ValueError("phase2k rerun candidate window has no rows")
        bronze_text = window["bronze_text"]
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, CandidateRow):
                raise ValueError("phase2k rerun candidate rows must be CandidateRow")
            if row.candidate_id in seen:
                raise ValueError("phase2k rerun candidate IDs must be unique")
            seen.add(row.candidate_id)
            if not 0 <= row.start < row.end <= len(bronze_text):
                raise ValueError("phase2k rerun candidate offsets are invalid")
            if bronze_text[row.start:row.end] != row.text:
                raise ValueError("phase2k rerun candidate text is not an exact slice")
            if row.is_gold_positive != (row.label == KEEP):
                raise ValueError("phase2k rerun candidate label is inconsistent")
            if row.is_gold_positive != bool(row.gold_mention_ids):
                raise ValueError(
                    "phase2k rerun positive candidate must retain mention IDs",
                )
        candidate_spans = {(row.start, row.end) for row in rows}
        window_accepted = accepted[window["phase2k_window_id"]]
        expected_positive = sum(
            1 for span in window_accepted if span in candidate_spans
        )
        actual_positive = sum(1 for row in rows if row.is_gold_positive)
        if actual_positive != expected_positive:
            raise ValueError(
                "phase2k rerun candidate positive count does not match the "
                "accepted spans",
            )
        positive_spans = {
            (row.start, row.end) for row in rows if row.is_gold_positive
        }
        if any(span not in window_accepted for span in positive_spans):
            raise ValueError(
                "phase2k rerun positive candidate spans are not accepted spans",
            )


def _evaluate_discriminative_windows(
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Any],
    representation: str,
    primary_cell: str,
) -> dict[str, Any]:
    targets_by_window = _representation_targets(inputs, representation)
    windows: list[dict[str, Any]] = []
    for phase2k_window_id in sorted(targets_by_window):
        adapter = adapters["by_window"][phase2k_window_id][
            "raw" if representation == RAW_INPUT_REPRESENTATION else "polished"
        ]
        adapter_window_id = adapter["window"].window_id
        rows = dataset["windows"][adapter_window_id]["rows"]
        window_rankings = rankings[adapter_window_id]
        outputs: list[dict[str, Any]] = []
        cell_scores: dict[str, dict[str, dict[str, Any]]] = {}
        cell_metrics: dict[str, dict[str, Any]] = {}
        for cell in CELLS:
            cell_scores[cell] = {
                row.candidate_id: dict(window_rankings[cell][row.candidate_id])
                for row in rows
            }
            cell_metrics[cell] = phase2h_window_metrics(
                adapter_window_id, rows, rankings, cell,
            )
        for row in rows:
            entry = window_rankings[primary_cell][row.candidate_id]
            if entry["selected"] == KEEP:
                outputs.append({
                    "output_id": row.candidate_id,
                    "span": (row.start, row.end),
                    "text": row.text,
                    "node_type": None,
                })
        evaluated = evaluate_targets(
            window_id=phase2k_window_id,
            targets=targets_by_window[phase2k_window_id],
            outputs=outputs,
            window_text=adapter["window"].text,
            require_exact_node_type=False,
        )
        windows.append({
            "phase2k_window_id": phase2k_window_id,
            "adapter_window_id": adapter_window_id,
            "candidate_count": len(rows),
            "target_count": len(targets_by_window[phase2k_window_id]),
            "row": evaluated["row"],
            "per_target": evaluated["per_target"],
            "matched_output_ids": evaluated["matched_output_ids"],
            "output_ids": evaluated["output_ids"],
            "cell_scores": cell_scores,
            "cell_metrics": cell_metrics,
        })
    return {"windows": windows}


def replay_discriminative_evidence(
    *,
    dataset: Mapping[str, Any],
    run_cv_fn: Callable[..., Any] | None = None,
    compute_rankings_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Deterministically replay the frozen Phase 2H CV and rankings.

    Production defaults resolve to the frozen real Phase 2H functions;
    tests inject cheap deterministic stand-ins.  The returned ``cv`` and
    ``rankings`` must reproduce the sealed artifact's folds, fit scope,
    per-cell scores, and per-cell metrics exactly.
    """
    if run_cv_fn is None:
        run_cv_fn = phase2h_run_cv
    if compute_rankings_fn is None:
        compute_rankings_fn = phase2h_compute_rankings
    cv = run_cv_fn(dataset, cells=CELLS)
    if not isinstance(cv, Mapping) or not isinstance(
        cv.get("oof_scores"), Mapping,
    ):
        raise ValueError("phase2k run_cv must return oof_scores")
    fit_scope = cv.get("fit_scope")
    if isinstance(fit_scope, Mapping):
        normalized_fit_scope: dict[Any, Any] = {}
        for cell, records in fit_scope.items():
            if isinstance(records, Mapping):
                normalized_fit_scope[cell] = {
                    str(fold_index): record
                    for fold_index, record in records.items()
                }
            else:
                normalized_fit_scope[cell] = records
        cv = {**cv, "fit_scope": normalized_fit_scope}
    rankings = compute_rankings_fn(
        dataset, cv["oof_scores"], cells=CELLS,
    )
    return {"cv": cv, "rankings": rankings}


def build_discriminative_artifacts(
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    created_at: str,
    primary_cell: str = DEFAULT_PRIMARY_CELL,
    run_cv_fn: Callable[..., Any] | None = None,
    compute_rankings_fn: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the Phase 2H scorer on both representations and seal artifacts.

    Runs the same grouped leave-one-window-out folds over all four fixed
    Phase 2H cells with the existing ``run_cv``/``compute_rankings`` APIs and
    ``KEEP_THRESHOLD``; no tuning.  Every cell's scores are preserved as
    supplementary evidence while comparison-v2 rows use only the declared
    primary cell.
    """
    artifacts: dict[str, dict[str, Any]] = {}
    for representation, adapter_key in (
        (RAW_INPUT_REPRESENTATION, "raw"),
        (POLISHED_INPUT_REPRESENTATION, "polished"),
    ):
        dataset = build_candidate_dataset(
            inputs=inputs, adapters=adapters, representation=representation,
        )
        replayed = replay_discriminative_evidence(
            dataset=dataset,
            run_cv_fn=run_cv_fn,
            compute_rankings_fn=compute_rankings_fn,
        )
        cv = replayed["cv"]
        rankings = replayed["rankings"]
        evidence = _evaluate_discriminative_windows(
            inputs=inputs,
            adapters=adapters,
            dataset=dataset,
            rankings=rankings,
            representation=representation,
            primary_cell=primary_cell,
        )
        payload = {
            "schema_version": DISCRIMINATIVE_ARTIFACT_SCHEMA_VERSION,
            "architecture_family": DISCRIMINATIVE_ARCHITECTURE_FAMILY,
            "input_representation": representation,
            "input_adapter_sha256": adapters[adapter_key]["adapter_sha256"],
            "scorer_config_sha256": build_scorer_config_sha256(primary_cell),
            "scorer_config": _scorer_config_descriptor(primary_cell),
            "created_at": created_at,
            "dataset_binding": build_dataset_binding(inputs),
            "semantic_target_contract_sha256": build_semantic_target_contract(
                inputs,
            )["contract_sha256"],
            "evaluation_contract_sha256": EVALUATION_CONTRACT_SHA256,
            "primary_cell": primary_cell,
            "cells": list(CELLS),
            "folds": cv["folds"],
            "fit_scope": cv["fit_scope"],
            "windows": evidence["windows"],
        }
        artifact = {"content_sha256": canonical_sha256(payload), **payload}
        artifacts[representation] = artifact
    return artifacts[RAW_INPUT_REPRESENTATION], artifacts[
        POLISHED_INPUT_REPRESENTATION
    ]


# ---------------------------------------------------------------------------
# Comparison-input evidence artifact
# ---------------------------------------------------------------------------


def _comparison_input_payload(
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    config: SemanticCompilerConfig,
    primary_cell: str,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
    generative_raw: Mapping[str, Any],
    generative_polished: Mapping[str, Any],
    discriminative_raw: Mapping[str, Any],
    discriminative_polished: Mapping[str, Any],
) -> dict[str, Any]:
    """Unsealed comparison-input payload with the two v2 builder blocks."""
    contract = build_semantic_target_contract(inputs)
    dataset_binding = build_dataset_binding(inputs)
    entity_aliases = _normalize_alias_tuple(entity_aliases, "entity")
    ability_aliases = _normalize_alias_tuple(ability_aliases, "ability")

    def _arch(
        *,
        family: str,
        config_sha256: str,
        raw_cell: Mapping[str, Any],
        polished_cell: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "family": family,
            "semantic_contract_sha256": contract["contract_sha256"],
            "model_or_scorer_config_sha256": config_sha256,
            "evaluation_contract_sha256": EVALUATION_CONTRACT_SHA256,
            "raw_input_adapter_sha256": adapters["raw"]["adapter_sha256"],
            "polished_input_adapter_sha256": adapters["polished"][
                "adapter_sha256"
            ],
            "raw": raw_cell,
            "polished": polished_cell,
        }

    def _cell(
        representation: str,
        artifact: Mapping[str, Any],
    ) -> dict[str, Any]:
        rows = [window["row"] for window in artifact["windows"]]
        return {
            "input_representation": representation,
            "output_artifact_sha256": artifact["content_sha256"],
            "rows": rows,
        }

    architectures = {
        "generative": _arch(
            family=GENERATIVE_ARCHITECTURE_FAMILY,
            config_sha256=build_compiler_execution_sha256(
                config,
                entity_aliases=entity_aliases,
                ability_aliases=ability_aliases,
            ),
            raw_cell=_cell(
                RAW_INPUT_REPRESENTATION, generative_raw,
            ),
            polished_cell=_cell(
                POLISHED_INPUT_REPRESENTATION, generative_polished,
            ),
        ),
        "discriminative": _arch(
            family=DISCRIMINATIVE_ARCHITECTURE_FAMILY,
            config_sha256=build_scorer_config_sha256(primary_cell),
            raw_cell=_cell(
                RAW_INPUT_REPRESENTATION, discriminative_raw,
            ),
            polished_cell=_cell(
                POLISHED_INPUT_REPRESENTATION, discriminative_polished,
            ),
        ),
    }
    return {
        "schema_version": COMPARISON_INPUT_SCHEMA_VERSION,
        "purpose": (
            "Phase 2K downstream comparison-input evidence artifact.  Carries "
            "the two v2 architecture builder blocks with exact dataset/"
            "semantic-target bindings and measured rows.  Contains no "
            "decision, diagnosis, or note; those are supplied by the human "
            "closeout finalizer only."
        ),
        "dataset_binding": dataset_binding,
        "semantic_target_contract": contract,
        "primary_cell": primary_cell,
        "architectures": architectures,
        "artifact_files": dict(ARTIFACT_FILENAMES),
    }


def build_comparison_input(
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    config: SemanticCompilerConfig,
    primary_cell: str,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
    generative_raw: Mapping[str, Any],
    generative_polished: Mapping[str, Any],
    discriminative_raw: Mapping[str, Any],
    discriminative_polished: Mapping[str, Any],
) -> dict[str, Any]:
    """Sealed evidence artifact with the two v2 architecture builder blocks.

    Carries exact dataset/semantic-target bindings and rows, but no inferred
    decision, diagnosis, or note.  ``output_artifact_sha256`` values bind the
    actual canonical artifact envelope content hashes.
    """
    payload = _comparison_input_payload(
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
        generative_raw=generative_raw,
        generative_polished=generative_polished,
        discriminative_raw=discriminative_raw,
        discriminative_polished=discriminative_polished,
    )
    envelope = {"content_sha256": canonical_sha256(payload), **payload}
    validate_comparison_input(
        envelope,
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
        generative_raw=generative_raw,
        generative_polished=generative_polished,
        discriminative_raw=discriminative_raw,
        discriminative_polished=discriminative_polished,
    )
    return envelope


def validate_comparison_input(
    value: object,
    *,
    inputs: Mapping[str, Any],
    adapters: Mapping[str, Any],
    config: SemanticCompilerConfig,
    primary_cell: str,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
    generative_raw: Mapping[str, Any],
    generative_polished: Mapping[str, Any],
    discriminative_raw: Mapping[str, Any],
    discriminative_polished: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("phase2k comparison-input artifact must be an object")
    _require_exact_keys(
        value,
        (
            "schema_version",
            "content_sha256",
            "purpose",
            "dataset_binding",
            "semantic_target_contract",
            "primary_cell",
            "architectures",
            "artifact_files",
        ),
        "phase2k comparison-input artifact",
    )
    if value["schema_version"] != COMPARISON_INPUT_SCHEMA_VERSION:
        raise ValueError("phase2k comparison-input schema version is invalid")
    _validate_recomputed_content_hash(
        value, label="phase2k comparison-input artifact",
    )
    _require_exact_keys(
        value["architectures"], ("generative", "discriminative"),
        "comparison-input architectures",
    )
    payload = _comparison_input_payload(
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
        generative_raw=generative_raw,
        generative_polished=generative_polished,
        discriminative_raw=discriminative_raw,
        discriminative_polished=discriminative_polished,
    )
    expected = {"content_sha256": canonical_sha256(payload), **payload}
    if value != expected:
        raise ValueError("comparison-input artifact does not match its sources")
    if any(
        key in value for key in ("decision", "diagnosis", "note")
    ):
        raise ValueError("comparison-input artifact must not carry closeout fields")
    return dict(value)


# ---------------------------------------------------------------------------
# Full run and atomic publication
# ---------------------------------------------------------------------------


def run_phase2k_downstream_rerun(
    *,
    phase2k_dir: Path,
    alignment_packet_path: Path,
    alignment_summary_path: Path,
    reviewed_packet_path: Path,
    coverage_path: Path,
    output: Path,
    config: SemanticCompilerConfig,
    chat: Callable[..., str],
    primary_cell: str = DEFAULT_PRIMARY_CELL,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
    compiler: Callable[..., Any] = compile_source_semantic_ir,
    run_cv_fn: Callable[..., Any] | None = None,
    compute_rankings_fn: Callable[..., Any] | None = None,
    created_at: str | None = None,
    git_commit: str | None = None,
    repository_dirty: bool | None = None,
) -> Path:
    """Run the full gate-locked paired rerun and atomically publish output.

    Every gate is validated before the first compiler/scorer call.  All
    artifacts are written into a temporary sibling directory, fully validated,
    and only then atomically renamed to ``output``.  Any failure (including a
    partial live/provider failure) leaves no result directory at ``output``.
    """
    if output.exists():
        raise ValueError(
            f"output directory already exists; refusing to overwrite: {output}",
        )
    if created_at is None:
        created_at = _now_iso()
    if git_commit is None or repository_dirty is None:
        commit, dirty = _git_state(ROOT)
        if git_commit is None:
            git_commit = commit
        if repository_dirty is None:
            repository_dirty = dirty
    if _COMMIT.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a full lowercase revision")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=parent))
    try:
        _run_phase2k_downstream_rerun_into(
            temporary,
            phase2k_dir=phase2k_dir,
            alignment_packet_path=alignment_packet_path,
            alignment_summary_path=alignment_summary_path,
            reviewed_packet_path=reviewed_packet_path,
            coverage_path=coverage_path,
            config=config,
            chat=chat,
            primary_cell=primary_cell,
            entity_aliases=entity_aliases,
            ability_aliases=ability_aliases,
            compiler=compiler,
            run_cv_fn=run_cv_fn,
            compute_rankings_fn=compute_rankings_fn,
            created_at=created_at,
            git_commit=git_commit,
            repository_dirty=repository_dirty,
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _run_phase2k_downstream_rerun_into(
    staging: Path,
    *,
    phase2k_dir: Path,
    alignment_packet_path: Path,
    alignment_summary_path: Path,
    reviewed_packet_path: Path,
    coverage_path: Path,
    config: SemanticCompilerConfig,
    chat: Callable[..., str],
    primary_cell: str,
    entity_aliases: Iterable[str],
    ability_aliases: Iterable[str],
    compiler: Callable[..., Any],
    run_cv_fn: Callable[..., Any] | None,
    compute_rankings_fn: Callable[..., Any] | None,
    created_at: str,
    git_commit: str,
    repository_dirty: bool,
) -> None:
    """Build, write, and fail-closed validate the rerun into ``staging``."""
    inputs = load_rerun_inputs(
        phase2k_dir=phase2k_dir,
        alignment_packet_path=alignment_packet_path,
        alignment_summary_path=alignment_summary_path,
        reviewed_packet_path=reviewed_packet_path,
        coverage_path=coverage_path,
    )
    adapters = build_input_adapters(inputs)
    preflight = build_preflight_contract(
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
    )
    generative_raw, generative_polished, _, _ = build_generative_artifacts(
        inputs=inputs,
        adapters=adapters,
        config=config,
        chat=chat,
        created_at=created_at,
        git_commit=git_commit,
        repository_dirty=repository_dirty,
        compiler=compiler,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
    )
    discriminative_raw, discriminative_polished = build_discriminative_artifacts(
        inputs=inputs,
        adapters=adapters,
        created_at=created_at,
        primary_cell=primary_cell,
        run_cv_fn=run_cv_fn,
        compute_rankings_fn=compute_rankings_fn,
    )
    comparison_input = build_comparison_input(
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
        generative_raw=generative_raw,
        generative_polished=generative_polished,
        discriminative_raw=discriminative_raw,
        discriminative_polished=discriminative_polished,
    )
    files = {
        ARTIFACT_FILENAMES["preflight"]: preflight,
        ARTIFACT_FILENAMES["generative_raw"]: generative_raw,
        ARTIFACT_FILENAMES["generative_polished"]: generative_polished,
        ARTIFACT_FILENAMES["discriminative_raw"]: discriminative_raw,
        ARTIFACT_FILENAMES["discriminative_polished"]: discriminative_polished,
        ARTIFACT_FILENAMES["comparison_input"]: comparison_input,
    }
    for filename, body in files.items():
        _write_exact(staging / filename, _canonical_json(body) + "\n")
    validate_rerun_evidence(
        staging,
        phase2k_dir=phase2k_dir,
        alignment_packet_path=alignment_packet_path,
        alignment_summary_path=alignment_summary_path,
        reviewed_packet_path=reviewed_packet_path,
        coverage_path=coverage_path,
        run_cv_fn=run_cv_fn,
        compute_rankings_fn=compute_rankings_fn,
    )


# ---------------------------------------------------------------------------
# Source-bound evidence validation
# ---------------------------------------------------------------------------


def _evidence_context(
    *,
    evidence_dir: Path,
    phase2k_dir: Path,
    alignment_packet_path: Path,
    alignment_summary_path: Path,
    reviewed_packet_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    inputs = load_rerun_inputs(
        phase2k_dir=phase2k_dir,
        alignment_packet_path=alignment_packet_path,
        alignment_summary_path=alignment_summary_path,
        reviewed_packet_path=reviewed_packet_path,
        coverage_path=coverage_path,
    )
    preflight = load_json_strict(
        evidence_dir / ARTIFACT_FILENAMES["preflight"],
        label="preflight input contract",
    )
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("preflight input contract schema version is invalid")
    config_raw = preflight.get("compiler", {}).get("config")
    if not isinstance(config_raw, Mapping):
        raise ValueError("preflight input contract is missing the compiler config")
    config = SemanticCompilerConfig(**config_raw)
    primary_cell = preflight.get("discriminative", {}).get("primary_cell")
    if not isinstance(primary_cell, str) or primary_cell not in CELLS:
        raise ValueError("preflight input contract has an invalid primary cell")
    entity_aliases = tuple(preflight.get("compiler", {}).get(
        "entity_aliases", [],
    ))
    ability_aliases = tuple(preflight.get("compiler", {}).get(
        "ability_aliases", [],
    ))
    adapters = build_input_adapters(inputs)
    validate_preflight_contract(
        preflight,
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=primary_cell,
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
    )
    return {
        "inputs": inputs,
        "adapters": adapters,
        "config": config,
        "primary_cell": primary_cell,
        "entity_aliases": entity_aliases,
        "ability_aliases": ability_aliases,
        "preflight": preflight,
    }


def _validate_generative_artifact_file(
    context: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    representation: str,
) -> None:
    adapter_key = "raw" if representation == RAW_INPUT_REPRESENTATION else "polished"
    _require_exact_keys(
        artifact,
        (
            "schema_version",
            "content_sha256",
            "architecture_family",
            "input_representation",
            "input_adapter_sha256",
            "compiler_config_sha256",
            "compiler_config",
            "entity_aliases",
            "ability_aliases",
            "created_at",
            "git_commit",
            "repository_dirty",
            "dataset_binding",
            "semantic_target_contract_sha256",
            "evaluation_contract_sha256",
            "windows",
            "run_artifacts",
        ),
        "phase2k generative artifact",
    )
    if artifact["schema_version"] != GENERATIVE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("phase2k generative artifact schema version is invalid")
    _validate_recomputed_content_hash(
        artifact, label="phase2k generative artifact",
    )
    if artifact["architecture_family"] != GENERATIVE_ARCHITECTURE_FAMILY:
        raise ValueError("phase2k generative artifact family is invalid")
    if artifact["input_representation"] != representation:
        raise ValueError("phase2k generative artifact representation is swapped")
    inputs = context["inputs"]
    adapters = context["adapters"]
    config = context["config"]
    expected_adapter = adapters[adapter_key]["adapter_sha256"]
    if artifact["input_adapter_sha256"] != expected_adapter:
        raise ValueError(
            "phase2k generative artifact input adapter does not match its sources",
        )
    expected_entity_aliases = list(context["entity_aliases"])
    expected_ability_aliases = list(context["ability_aliases"])
    if artifact["entity_aliases"] != expected_entity_aliases:
        raise ValueError("phase2k generative artifact entity aliases were changed")
    if artifact["ability_aliases"] != expected_ability_aliases:
        raise ValueError("phase2k generative artifact ability aliases were changed")
    expected_execution_sha256 = build_compiler_execution_sha256(
        config,
        entity_aliases=expected_entity_aliases,
        ability_aliases=expected_ability_aliases,
    )
    if artifact["compiler_config_sha256"] != expected_execution_sha256:
        raise ValueError(
            "phase2k generative artifact execution identity does not match "
            "the preflight contract",
        )
    if artifact["compiler_config_sha256"] != context["preflight"]["compiler"][
        "execution_sha256"
    ]:
        raise ValueError(
            "phase2k generative artifact execution identity does not match "
            "the preflight execution hash",
        )
    if artifact["compiler_config"] != asdict(config):
        raise ValueError(
            "phase2k generative artifact compiler config body does not match",
        )
    if artifact["dataset_binding"] != build_dataset_binding(inputs):
        raise ValueError("phase2k generative artifact dataset binding is stale")
    contract_sha = build_semantic_target_contract(inputs)["contract_sha256"]
    if artifact["semantic_target_contract_sha256"] != contract_sha:
        raise ValueError("phase2k generative artifact target contract is stale")
    if artifact["evaluation_contract_sha256"] != EVALUATION_CONTRACT_SHA256:
        raise ValueError("phase2k generative artifact evaluation contract is stale")
    if not _COMMIT.fullmatch(artifact["git_commit"]):
        raise ValueError("phase2k generative artifact git commit is invalid")
    run_artifacts = artifact["run_artifacts"]
    if not isinstance(run_artifacts, Mapping):
        raise ValueError("phase2k generative run artifacts are invalid")
    windows = artifact["windows"]
    if not isinstance(windows, list):
        raise ValueError("phase2k generative windows are invalid")
    targets_by_window = _representation_targets(inputs, representation)
    expected_windows = sorted(targets_by_window)
    if [window["phase2k_window_id"] for window in windows] != expected_windows:
        raise ValueError("phase2k generative window order is invalid")
    _WINDOW_KEYS = (
        "phase2k_window_id",
        "adapter_window_id",
        "target_count",
        "row",
        "per_target",
        "matched_output_ids",
        "output_ids",
    )
    if set(run_artifacts) != set(expected_windows):
        raise ValueError(
            "phase2k generative run artifacts do not cover every window",
        )
    for window in windows:
        _require_exact_keys(
            window, _WINDOW_KEYS, "phase2k generative window",
        )
        phase2k_window_id = window["phase2k_window_id"]
        payload = run_artifacts.get(phase2k_window_id)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"phase2k generative run artifact is missing for "
                f"{phase2k_window_id}",
            )
        restored = SemanticRunArtifact.from_json(_canonical_json(payload))
        if restored.content_sha256 != payload.get("content_sha256"):
            raise ValueError(
                f"phase2k generative run artifact hash is invalid for "
                f"{phase2k_window_id}",
            )
        adapter = adapters["by_window"][phase2k_window_id][adapter_key]
        expected_window_dict = _adapter_window_dict(adapter["window"])
        restored_window = restored.payload["run"].get("window")
        if restored_window != expected_window_dict:
            raise ValueError(
                "phase2k generative run artifact window/source does not "
                f"match the bound adapter for {phase2k_window_id}",
            )
        expected_input_hashes = {
            "bronze_source_content_sha256": adapter["window"].source_content_sha256,
            "bronze_source_provenance_sha256": adapter["window"].source_provenance_sha256,
            "source_window_sha256": canonical_sha256(expected_window_dict),
            "phase2k_records_sha256": inputs["records_obj"]["content_sha256"],
            "phase2k_alignment_packet_sha256": inputs["alignment_packet"][
                "content_sha256"
            ],
            "phase2k_input_adapter_sha256": expected_adapter,
        }
        if restored.payload["input_hashes"] != expected_input_hashes:
            raise ValueError(
                "phase2k generative run artifact input hashes do not match "
                f"its sources for {phase2k_window_id}",
            )
        if (
            restored.payload.get("git_commit") != artifact["git_commit"]
            or restored.payload.get("repository_dirty")
            != artifact["repository_dirty"]
            or restored.payload.get("created_at") != artifact["created_at"]
        ):
            raise ValueError(
                "phase2k generative run artifact lineage does not match its "
                f"envelope for {phase2k_window_id}",
            )
        run_body = restored.payload["run"]
        if run_body.get("entity_aliases") != expected_entity_aliases or (
            run_body.get("ability_aliases") != expected_ability_aliases
        ):
            raise ValueError(
                "phase2k generative run artifact aliases do not match the "
                f"envelope for {phase2k_window_id}",
            )
        failures = run_body.get("failures")
        if isinstance(failures, list) and any(
            isinstance(failure, Mapping)
            and failure.get("code") == "PROVIDER_FAILURE"
            for failure in failures
        ):
            raise ValueError(
                "phase2k generative run artifact contains a typed "
                f"PROVIDER_FAILURE for {phase2k_window_id}",
            )
        window_text = adapter["window"].text
        outputs = _serialized_run_nodes(payload)
        recomputed = evaluate_targets(
            window_id=phase2k_window_id,
            targets=targets_by_window[phase2k_window_id],
            outputs=outputs,
            window_text=window_text,
            require_exact_node_type=True,
        )
        if recomputed["row"] != window["row"]:
            raise ValueError(
                f"phase2k generative rows do not recompute for "
                f"{phase2k_window_id}",
            )
        if recomputed["per_target"] != window["per_target"]:
            raise ValueError(
                f"phase2k generative matching does not recompute for "
                f"{phase2k_window_id}",
            )
        if recomputed["matched_output_ids"] != window["matched_output_ids"]:
            raise ValueError(
                f"phase2k generative matched outputs do not recompute for "
                f"{phase2k_window_id}",
            )
        validate_row(window["row"], expected_window_id=phase2k_window_id)
    if sum(window["row"]["target_count"] for window in windows) != TARGET_COUNT:
        raise ValueError("phase2k generative target counts do not sum to 311")


_FOLD_KEYS = (
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
_FIT_SCOPE_BASE_KEYS = (
    "fold_index",
    "train_window_ids",
    "test_window_id",
    "fit_scope",
    "train_candidate_count",
    "train_positive_count",
    "train_negative_count",
    "class_weights",
    "scaler",
    "model_config",
    "feature_names_sha256",
)
_CELL_SCORE_KEYS = ("score", "rank", "selected")
_CELL_METRIC_KEYS = (
    "window_id",
    "candidate_count",
    "label_keep_count",
    "label_drop_count",
    "predicted_keep_count",
    "predicted_drop_count",
    "selected",
    "prevalence",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "roc_auc",
    "all_drop_baseline",
    "all_keep_baseline",
    "recall_at_k",
    "precision_at_k",
    "gold_rank",
    "overlap_diagnostics",
)
_DISCRIMINATIVE_WINDOW_KEYS = (
    "phase2k_window_id",
    "adapter_window_id",
    "candidate_count",
    "target_count",
    "row",
    "per_target",
    "matched_output_ids",
    "output_ids",
    "cell_scores",
    "cell_metrics",
)


def _validate_folds_structure(folds: object) -> None:
    _require_list(folds, "phase2k discriminative folds")
    for index, fold in enumerate(folds):
        label = f"phase2k discriminative fold {index}"
        _require_exact_keys(fold, _FOLD_KEYS, label)
        _require_int(fold["fold_index"], f"{label} fold_index", minimum=0)
        train_window_ids = _require_list(
            fold["train_window_ids"], f"{label} train_window_ids",
        )
        if any(
            not isinstance(item, str) or not item.strip()
            for item in train_window_ids
        ):
            raise ValueError(f"{label} train_window_ids are invalid")
        _require_nonempty_string(
            fold["test_window_id"], f"{label} test_window_id",
        )
        for key in (
            "train_candidate_count",
            "train_positive_count",
            "train_negative_count",
            "test_candidate_count",
            "test_positive_count",
        ):
            _require_int(fold[key], f"{label} {key}", minimum=0)
        _require_exact_keys(fold["class_weights"], (KEEP, DROP), f"{label} class_weights")
        for weight in fold["class_weights"].values():
            _require_number(weight, f"{label} class weight")


def _validate_fit_scope_structure(fit_scope: object, folds: object) -> None:
    label = "phase2k discriminative fit_scope"
    _require_exact_keys(fit_scope, CELLS, label)
    fold_indices = {
        str(fold["fold_index"]) for fold in _require_list(folds, "folds")
    }
    for cell in CELLS:
        cell_label = f"{label} {cell}"
        records_by_index = fit_scope[cell]
        _require_exact_keys(records_by_index, fold_indices, cell_label)
        feature_set = cell.split("_")[1]
        expected_keys = (
            _FIT_SCOPE_BASE_KEYS
            if feature_set == "A"
            else _FIT_SCOPE_BASE_KEYS + ("vectorizer",)
        )
        for index, record in records_by_index.items():
            record_label = f"{cell_label} fold {index}"
            _require_exact_keys(record, expected_keys, record_label)
            _require_int(
                record["fold_index"], f"{record_label} fold_index",
                minimum=0,
            )
            train_window_ids = _require_list(
                record["train_window_ids"], f"{record_label} train_window_ids",
            )
            if any(
                not isinstance(item, str) or not item.strip()
                for item in train_window_ids
            ):
                raise ValueError(f"{record_label} train_window_ids are invalid")
            _require_nonempty_string(
                record["test_window_id"], f"{record_label} test_window_id",
            )
            _require_enum(
                record["fit_scope"], ("training windows only",),
                f"{record_label} fit_scope",
            )
            for key in (
                "train_candidate_count",
                "train_positive_count",
                "train_negative_count",
            ):
                _require_int(record[key], f"{record_label} {key}", minimum=0)
            _require_exact_keys(
                record["class_weights"], (KEEP, DROP),
                f"{record_label} class_weights",
            )
            for weight in record["class_weights"].values():
                _require_number(weight, f"{record_label} class weight")
            _require_hex64(
                record["feature_names_sha256"],
                f"{record_label} feature_names_sha256",
            )


def _validate_discriminative_artifact_file(
    context: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    representation: str,
    run_cv_fn: Callable[..., Any] | None = None,
    compute_rankings_fn: Callable[..., Any] | None = None,
) -> None:
    adapter_key = "raw" if representation == RAW_INPUT_REPRESENTATION else "polished"
    _require_exact_keys(
        artifact,
        (
            "schema_version",
            "content_sha256",
            "architecture_family",
            "input_representation",
            "input_adapter_sha256",
            "scorer_config_sha256",
            "scorer_config",
            "created_at",
            "dataset_binding",
            "semantic_target_contract_sha256",
            "evaluation_contract_sha256",
            "primary_cell",
            "cells",
            "folds",
            "fit_scope",
            "windows",
        ),
        "phase2k discriminative artifact",
    )
    if artifact["schema_version"] != DISCRIMINATIVE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("phase2k discriminative artifact schema version is invalid")
    _validate_recomputed_content_hash(
        artifact, label="phase2k discriminative artifact",
    )
    if artifact["architecture_family"] != DISCRIMINATIVE_ARCHITECTURE_FAMILY:
        raise ValueError("phase2k discriminative artifact family is invalid")
    if artifact["input_representation"] != representation:
        raise ValueError("phase2k discriminative artifact representation is swapped")
    inputs = context["inputs"]
    adapters = context["adapters"]
    primary_cell = context["primary_cell"]
    expected_adapter = adapters[adapter_key]["adapter_sha256"]
    if artifact["input_adapter_sha256"] != expected_adapter:
        raise ValueError(
            "phase2k discriminative artifact input adapter does not match its "
            "sources",
        )
    if artifact["primary_cell"] != primary_cell:
        raise ValueError("phase2k discriminative primary cell was changed")
    if artifact["cells"] != list(CELLS):
        raise ValueError("phase2k discriminative cells were changed")
    if artifact["scorer_config_sha256"] != build_scorer_config_sha256(
        primary_cell,
    ):
        raise ValueError("phase2k discriminative scorer config was changed")
    if artifact["scorer_config"] != _scorer_config_descriptor(primary_cell):
        raise ValueError("phase2k discriminative scorer config body was changed")
    if artifact["dataset_binding"] != build_dataset_binding(inputs):
        raise ValueError("phase2k discriminative dataset binding is stale")
    contract_sha = build_semantic_target_contract(inputs)["contract_sha256"]
    if artifact["semantic_target_contract_sha256"] != contract_sha:
        raise ValueError("phase2k discriminative target contract is stale")
    if artifact["evaluation_contract_sha256"] != EVALUATION_CONTRACT_SHA256:
        raise ValueError("phase2k discriminative evaluation contract is stale")
    _validate_folds_structure(artifact["folds"])
    _validate_fit_scope_structure(artifact["fit_scope"], artifact["folds"])
    dataset = build_candidate_dataset(
        inputs=inputs, adapters=adapters, representation=representation,
    )
    replayed = replay_discriminative_evidence(
        dataset=dataset,
        run_cv_fn=run_cv_fn,
        compute_rankings_fn=compute_rankings_fn,
    )
    cv = replayed["cv"]
    rankings = replayed["rankings"]
    if canonical_sha256(artifact["folds"]) != canonical_sha256(cv["folds"]):
        raise ValueError(
            "phase2k discriminative folds do not reproduce from the bound "
            "dataset",
        )
    if canonical_sha256(artifact["fit_scope"]) != canonical_sha256(
        cv["fit_scope"],
    ):
        raise ValueError(
            "phase2k discriminative fit_scope does not reproduce from the "
            "bound dataset",
        )
    targets_by_window = _representation_targets(inputs, representation)
    expected_windows = sorted(targets_by_window)
    windows = artifact["windows"]
    if not isinstance(windows, list):
        raise ValueError("phase2k discriminative windows are invalid")
    if [window["phase2k_window_id"] for window in windows] != expected_windows:
        raise ValueError("phase2k discriminative window order is invalid")
    for window in windows:
        _require_exact_keys(
            window,
            _DISCRIMINATIVE_WINDOW_KEYS,
            "phase2k discriminative window",
        )
        phase2k_window_id = window["phase2k_window_id"]
        adapter = adapters["by_window"][phase2k_window_id][adapter_key]
        adapter_window_id = adapter["window"].window_id
        rows = dataset["windows"][adapter_window_id]["rows"]
        if window["candidate_count"] != len(rows):
            raise ValueError(
                f"phase2k discriminative candidate count does not recompute "
                f"for {phase2k_window_id}",
            )
        cell_scores = window["cell_scores"]
        if not isinstance(cell_scores, Mapping) or set(cell_scores) != set(CELLS):
            raise ValueError("phase2k discriminative cell scores are incomplete")
        window_rankings = rankings[adapter_window_id]
        selected: list[dict[str, Any]] = []
        for cell in CELLS:
            scores = cell_scores[cell]
            cell_label = f"phase2k discriminative cell_scores {cell}"
            candidate_ids = {row.candidate_id for row in rows}
            _require_exact_keys(scores, candidate_ids, cell_label)
            for candidate_id, entry in scores.items():
                entry_label = f"{cell_label} {candidate_id}"
                _require_exact_keys(entry, _CELL_SCORE_KEYS, entry_label)
                _require_number(entry["score"], f"{entry_label} score")
                _require_int(entry["rank"], f"{entry_label} rank", minimum=1)
                _require_enum(
                    entry["selected"], (KEEP, DROP),
                    f"{entry_label} selected",
                )
                if entry["selected"] != (
                    KEEP if entry["score"] >= KEEP_THRESHOLD else DROP
                ):
                    raise ValueError(f"{entry_label} selected is inconsistent")
            if scores != window_rankings[cell]:
                raise ValueError(
                    f"phase2k discriminative cell scores do not reproduce "
                    f"for {cell} in {phase2k_window_id}",
                )
            cell_metrics = window["cell_metrics"]
            if not isinstance(cell_metrics, Mapping) or set(
                cell_metrics,
            ) != set(CELLS):
                raise ValueError("phase2k discriminative cell metrics are incomplete")
            metric = cell_metrics[cell]
            metric_label = f"phase2k discriminative cell_metrics {cell}"
            _require_exact_keys(metric, _CELL_METRIC_KEYS, metric_label)
            if metric["window_id"] != adapter_window_id:
                raise ValueError(f"{metric_label} window_id is invalid")
            expected_metric = phase2h_window_metrics(
                adapter_window_id, rows, rankings, cell,
            )
            if metric != expected_metric:
                raise ValueError(
                    f"phase2k discriminative cell metrics do not reproduce "
                    f"for {cell} in {phase2k_window_id}",
                )
        for row in rows:
            entry = window_rankings[primary_cell][row.candidate_id]
            if entry["selected"] == KEEP:
                selected.append({
                    "output_id": row.candidate_id,
                    "span": (row.start, row.end),
                    "text": row.text,
                    "node_type": None,
                })
        recomputed = evaluate_targets(
            window_id=phase2k_window_id,
            targets=targets_by_window[phase2k_window_id],
            outputs=selected,
            window_text=adapter["window"].text,
            require_exact_node_type=False,
        )
        if recomputed["row"] != window["row"]:
            raise ValueError(
                f"phase2k discriminative rows do not recompute for "
                f"{phase2k_window_id}",
            )
        if recomputed["per_target"] != window["per_target"]:
            raise ValueError(
                f"phase2k discriminative matching does not recompute for "
                f"{phase2k_window_id}",
            )
        if recomputed["matched_output_ids"] != window["matched_output_ids"]:
            raise ValueError(
                f"phase2k discriminative matched outputs do not recompute for "
                f"{phase2k_window_id}",
            )
        validate_row(window["row"], expected_window_id=phase2k_window_id)
    if sum(window["row"]["target_count"] for window in windows) != TARGET_COUNT:
        raise ValueError("phase2k discriminative target counts do not sum to 311")


def validate_rerun_evidence(
    evidence_dir: Path,
    *,
    phase2k_dir: Path,
    alignment_packet_path: Path,
    alignment_summary_path: Path,
    reviewed_packet_path: Path,
    coverage_path: Path,
    run_cv_fn: Callable[..., Any] | None = None,
    compute_rankings_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Reload and fail-closed validate a published rerun evidence directory.

    Recomputes hashes, rows, configs, and adapters from the current Phase 2K
    sources, replays the bound Phase 2H folds/fit_scope/scores/metrics with
    the same (injectable) frozen Phase 2H functions used to build the
    artifacts, and rejects tampering, swapped raw/polished artifacts, changed
    primary scorer, stale gates, and target mismatch.
    """
    context = _evidence_context(
        evidence_dir=evidence_dir,
        phase2k_dir=phase2k_dir,
        alignment_packet_path=alignment_packet_path,
        alignment_summary_path=alignment_summary_path,
        reviewed_packet_path=reviewed_packet_path,
        coverage_path=coverage_path,
    )
    for filename in ARTIFACT_FILENAMES.values():
        if not (evidence_dir / filename).is_file():
            raise ValueError(f"phase2k rerun evidence is missing: {filename}")
    generative_raw = load_json_strict(
        evidence_dir / ARTIFACT_FILENAMES["generative_raw"],
        label="generative raw artifact",
    )
    generative_polished = load_json_strict(
        evidence_dir / ARTIFACT_FILENAMES["generative_polished"],
        label="generative polished artifact",
    )
    discriminative_raw = load_json_strict(
        evidence_dir / ARTIFACT_FILENAMES["discriminative_raw"],
        label="discriminative raw artifact",
    )
    discriminative_polished = load_json_strict(
        evidence_dir / ARTIFACT_FILENAMES["discriminative_polished"],
        label="discriminative polished artifact",
    )
    comparison_input = load_json_strict(
        evidence_dir / ARTIFACT_FILENAMES["comparison_input"],
        label="comparison-input artifact",
    )
    _validate_generative_artifact_file(
        context, generative_raw, representation=RAW_INPUT_REPRESENTATION,
    )
    _validate_generative_artifact_file(
        context, generative_polished, representation=POLISHED_INPUT_REPRESENTATION,
    )
    _validate_discriminative_artifact_file(
        context,
        discriminative_raw,
        representation=RAW_INPUT_REPRESENTATION,
        run_cv_fn=run_cv_fn,
        compute_rankings_fn=compute_rankings_fn,
    )
    _validate_discriminative_artifact_file(
        context,
        discriminative_polished,
        representation=POLISHED_INPUT_REPRESENTATION,
        run_cv_fn=run_cv_fn,
        compute_rankings_fn=compute_rankings_fn,
    )
    validate_comparison_input(
        comparison_input,
        inputs=context["inputs"],
        adapters=context["adapters"],
        config=context["config"],
        primary_cell=context["primary_cell"],
        generative_raw=generative_raw,
        generative_polished=generative_polished,
        discriminative_raw=discriminative_raw,
        discriminative_polished=discriminative_polished,
    )
    return {
        "context": context,
        "preflight": context["preflight"],
        "generative_raw": generative_raw,
        "generative_polished": generative_polished,
        "discriminative_raw": discriminative_raw,
        "discriminative_polished": discriminative_polished,
        "comparison_input": comparison_input,
    }


# ---------------------------------------------------------------------------
# Explicit human closeout finalizer
# ---------------------------------------------------------------------------


def finalize_phase2k_downstream_rerun(
    *,
    evidence_dir: Path,
    phase2k_dir: Path,
    alignment_packet_path: Path,
    alignment_summary_path: Path,
    reviewed_packet_path: Path,
    coverage_path: Path,
    decision: str,
    diagnosis: str,
    note: str,
    run_cv_fn: Callable[..., Any] | None = None,
    compute_rankings_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Finalize the v2 comparison from validated evidence and human fields.

    The decision/diagnosis/note are supplied by the human caller; they are
    never inferred.  The emitted comparison is validated fail-closed against
    the exact Phase 2K records, finalized human packet, human review summary,
    and completed transformation audit.
    """
    _require_enum(decision, FINAL_CLOSEOUT_STATUSES, "closeout decision")
    _require_enum(diagnosis, DOWNSTREAM_DIAGNOSIS_VALUES, "closeout diagnosis")
    _require_nonempty_string(note, "closeout note")
    validated = validate_rerun_evidence(
        evidence_dir,
        phase2k_dir=phase2k_dir,
        alignment_packet_path=alignment_packet_path,
        alignment_summary_path=alignment_summary_path,
        reviewed_packet_path=reviewed_packet_path,
        coverage_path=coverage_path,
        run_cv_fn=run_cv_fn,
        compute_rankings_fn=compute_rankings_fn,
    )
    comparison_input = validated["comparison_input"]
    inputs = validated["context"]["inputs"]
    comparison = build_downstream_comparison(
        dataset_binding=comparison_input["dataset_binding"],
        semantic_target_contract=comparison_input["semantic_target_contract"],
        architectures=comparison_input["architectures"],
        decision=decision,
        diagnosis=diagnosis,
        note=note,
    )
    validate_downstream_comparison(
        comparison,
        label="phase2k downstream comparison",
        records_obj=inputs["records_obj"],
        finalized_packet=inputs["finalized_packet"],
        human_summary=inputs["human_summary"],
        completed_audit=inputs["completed_audit"],
    )
    return comparison
