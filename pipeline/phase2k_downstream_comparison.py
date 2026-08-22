"""Phase 2K downstream-comparison v2 contract (isolated, stdlib-only).

The v1 downstream comparison only carried ``schema_version``,
``comparison_complete``, ``decision``, and ``note``, so a finalize step could
falsely close Phase 2K without any measured generative/discriminative
evidence.  This module replaces that declaration with the v2 comparison: one
canonical hash envelope that binds the exact Phase 2K dataset/human-review
artifacts, carries architecture-specific raw/polished per-window rows, and
recomputes every metric and raw-vs-polished delta from those rows.

The module is intentionally import-free of the Phase 2K core module so the
core can re-export these names without a circular import.  The canonical hash
implementation is byte-for-byte identical to ``canonical_sha256`` in
``pipeline.phase2k_contextual_reconstruction``.

This module never runs models, never fabricates rows, and never invents a
metric-to-diagnosis threshold.  The decision/diagnosis remain human empirical
interpretation, now bound to exact measured evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping


DOWNSTREAM_COMPARISON_SCHEMA_VERSION = "phase2k-downstream-comparison-v2"

FINAL_CLOSEOUT_STATUSES = frozenset({
    "CONTEXTUAL_POLISH_VALIDATED",
    "CONTEXT_ALONE_SUFFICIENT",
    "POLISH_UNSAFE_OVER_RECONSTRUCTING",
    "NO_MATERIAL_REPRESENTATION_GAIN",
    "INCONCLUSIVE",
})

GENERATIVE_ARCHITECTURE_FAMILY = "PHASE2F_GENERATIVE_SEMANTIC_IR"
DISCRIMINATIVE_ARCHITECTURE_FAMILY = "PHASE2H_DISCRIMINATIVE_ENDPOINT_SCORING"
RAW_INPUT_REPRESENTATION = "RAW_BRONZE"
POLISHED_INPUT_REPRESENTATION = "CONTEXTUAL_POLISH"

DOWNSTREAM_DIAGNOSIS_VALUES = frozenset({
    # Required Phase 2K paired-comparison interpretations.  These values name
    # the observable result before any narrower causal subtype is inferred.
    "RAW_REPRESENTATION_BOTTLENECK",
    "GENERATIVE_FAILURE_SUBSTANTIALLY_LOSSY_INPUT",
    "INPUT_QUALITY_AND_GENERATIVE_SPARSE_DISCRIMINATION_BOTTLENECK",
    "CONTEXTUAL_POLISH_NOT_EXPLANATORY",
    "DOWNSTREAM_SEMANTIC_EXTRACTION_FAILURE_BOUNDARY",
    # Optional subtyping when the human A/B/C/D and radius evidence licenses
    # a more specific interpretation.
    "CONTEXT_TRUNCATION_BOTTLENECK",
    "ASR_STRUCTURE_BOTTLENECK",
    "GENERATION_TASK_FORMULATION_BOTTLENECK",
    "MIXED",
    "INCONCLUSIVE",
})

COMPARISON_METRIC_NAMES = (
    "precision",
    "recall",
    "f1",
    "unsupported_rate",
    "provenance_valid_rate",
    "abstention_rate",
)

_HEX64 = re.compile(r"[0-9a-f]{64}")

_TOP_LEVEL_KEYS = (
    "schema_version",
    "content_sha256",
    "comparison_complete",
    "dataset_binding",
    "semantic_target_contract",
    "architectures",
    "decision",
    "diagnosis",
    "note",
)
_DATASET_BINDING_KEYS = (
    "phase2k_records_sha256",
    "finalized_human_packet_sha256",
    "human_summary_sha256",
    "completed_transformation_audit_sha256",
    "window_ids_sha256",
    "window_count",
    "human_review_gate_status",
)
_SEMANTIC_TARGET_CONTRACT_KEYS = (
    "contract_version",
    "contract_sha256",
    "target_count",
    "boundary_rule",
)
_ARCHITECTURE_KEYS = (
    "family",
    "semantic_contract_sha256",
    "model_or_scorer_config_sha256",
    "evaluation_contract_sha256",
    "raw_input_adapter_sha256",
    "polished_input_adapter_sha256",
    "raw",
    "polished",
    "deltas",
)
_BUILDER_ARCHITECTURE_KEYS = (
    "family",
    "semantic_contract_sha256",
    "model_or_scorer_config_sha256",
    "evaluation_contract_sha256",
    "raw_input_adapter_sha256",
    "polished_input_adapter_sha256",
    "raw",
    "polished",
)
_CELL_KEYS = ("input_representation", "output_artifact_sha256", "rows", "metrics")
_BUILDER_CELL_KEYS = ("input_representation", "output_artifact_sha256", "rows")
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
_METRIC_KEYS = ("numerator", "denominator", "rate")


def canonical_sha256(value: object) -> str:
    """Canonical content hash identical to the repository convention."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


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


def _require_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(
        value,
    ):
        raise ValueError(f"{label} must be a finite number")
    return value


def _safe_float(value: float) -> float:
    return round(float(value), 4)


def _validate_recomputed_content_hash(obj: Mapping[str, Any], *, label: str) -> None:
    _require_hex64(obj.get("content_sha256"), f"{label} content_sha256")
    expected = canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })
    if obj["content_sha256"] != expected:
        raise ValueError(f"{label} content_sha256 does not match canonical content")


def _records_window_ids(records_obj: Mapping[str, Any]) -> list[str]:
    """Sorted unique dataset window IDs from the reconstruction records."""
    records = records_obj.get("records")
    if not isinstance(records, list):
        raise ValueError("phase2k records file must contain a records list")
    window_ids: list[str] = []
    for record in records:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("window_id"), str)
            or not record["window_id"]
        ):
            raise ValueError("phase2k records window identities are invalid")
        window_ids.append(record["window_id"])
    return sorted(set(window_ids))


def _recomputed_cell_metrics(
    rows: list[Mapping[str, Any]],
    *,
    window_count: int,
) -> dict[str, dict[str, Any]]:
    """Deterministic cell metrics recomputed from exact per-window rows."""
    true_positive = sum(
        _require_int(row["true_positive_count"], "row true_positive_count")
        for row in rows
    )
    false_positive = sum(
        _require_int(row["false_positive_count"], "row false_positive_count")
        for row in rows
    )
    false_negative = sum(
        _require_int(row["false_negative_count"], "row false_negative_count")
        for row in rows
    )
    target_sum = sum(
        _require_int(row["target_count"], "row target_count") for row in rows
    )
    output_sum = sum(
        _require_int(row["output_count"], "row output_count") for row in rows
    )
    provenance_valid_sum = sum(
        _require_int(
            row["provenance_valid_count"], "row provenance_valid_count",
        )
        for row in rows
    )
    abstained_count = sum(1 for row in rows if row["abstained"])

    def metric(numerator: int, denominator: int) -> dict[str, Any]:
        if denominator == 0:
            return {"numerator": numerator, "denominator": denominator, "rate": None}
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": _safe_float(numerator / denominator),
        }

    return {
        "precision": metric(true_positive, true_positive + false_positive),
        "recall": metric(true_positive, target_sum),
        "f1": metric(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        "unsupported_rate": metric(false_positive, output_sum),
        "provenance_valid_rate": metric(provenance_valid_sum, output_sum),
        "abstention_rate": metric(abstained_count, window_count),
    }


def _recomputed_deltas(
    raw_metrics: Mapping[str, Mapping[str, Any]],
    polished_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for name in COMPARISON_METRIC_NAMES:
        raw_rate = raw_metrics[name]["rate"]
        polished_rate = polished_metrics[name]["rate"]
        if raw_rate is None or polished_rate is None:
            deltas[name] = None
        else:
            deltas[name] = _safe_float(polished_rate - raw_rate)
    return deltas


def _validate_row(
    row: object,
    *,
    label: str,
    expected_window_id: str,
) -> None:
    _require_exact_keys(row, _ROW_KEYS, label)
    window_id = _require_nonempty_string(row["window_id"], f"{label} window_id")
    if window_id != expected_window_id:
        raise ValueError(
            f"{label} window_id does not match the dataset window order",
        )
    target_count = _require_int(
        row["target_count"], f"{label} target_count", minimum=0,
    )
    true_positive = _require_int(
        row["true_positive_count"], f"{label} true_positive_count", minimum=0,
    )
    false_positive = _require_int(
        row["false_positive_count"], f"{label} false_positive_count", minimum=0,
    )
    false_negative = _require_int(
        row["false_negative_count"], f"{label} false_negative_count", minimum=0,
    )
    output_count = _require_int(
        row["output_count"], f"{label} output_count", minimum=0,
    )
    provenance_valid_count = _require_int(
        row["provenance_valid_count"],
        f"{label} provenance_valid_count",
        minimum=0,
    )
    _require_bool(row["abstained"], f"{label} abstained")
    _require_hex64(row["output_sha256"], f"{label} output_sha256")
    if true_positive + false_negative != target_count:
        raise ValueError(
            f"{label} true_positive + false_negative must equal target_count",
        )
    if true_positive + false_positive != output_count:
        raise ValueError(
            f"{label} true_positive + false_positive must equal output_count",
        )
    if provenance_valid_count > output_count:
        raise ValueError(
            f"{label} provenance_valid_count must not exceed output_count",
        )


def _validate_cell_rows(
    cell: object,
    *,
    label: str,
    expected_representation: str,
    expected_window_ids: list[str],
) -> list[dict[str, Any]]:
    _require_exact_keys(cell, _CELL_KEYS, label)
    representation = _require_string(
        cell["input_representation"], f"{label} input_representation",
    )
    if representation != expected_representation:
        raise ValueError(f"{label} input_representation is invalid")
    _require_hex64(cell["output_artifact_sha256"], f"{label} output_artifact_sha256")
    rows = _require_list(cell["rows"], f"{label} rows")
    if len(rows) != len(expected_window_ids):
        raise ValueError(
            f"{label} row count does not match the dataset window count",
        )
    for index, (row, window_id) in enumerate(
        zip(rows, expected_window_ids),
    ):
        _validate_row(
            row,
            label=f"{label} rows[{index}]",
            expected_window_id=window_id,
        )
    return rows


def _validate_metrics(
    value: object,
    *,
    rows: list[Mapping[str, Any]],
    window_count: int,
    label: str,
) -> None:
    _require_exact_keys(value, COMPARISON_METRIC_NAMES, label)
    expected = _recomputed_cell_metrics(rows, window_count=window_count)
    for name in COMPARISON_METRIC_NAMES:
        metric = value[name]
        if not isinstance(metric, Mapping):
            raise ValueError(f"{label} {name} must be an object")
        _require_exact_keys(metric, _METRIC_KEYS, f"{label} {name}")
        numerator = _require_int(
            metric["numerator"], f"{label} {name} numerator", minimum=0,
        )
        denominator = _require_int(
            metric["denominator"], f"{label} {name} denominator", minimum=0,
        )
        expected_metric = expected[name]
        if (
            numerator != expected_metric["numerator"]
            or denominator != expected_metric["denominator"]
        ):
            raise ValueError(
                f"{label} {name} numerator/denominator does not match rows",
            )
        rate = metric["rate"]
        expected_rate = expected_metric["rate"]
        if expected_rate is None:
            if rate is not None:
                raise ValueError(
                    f"{label} {name} rate must be null for a zero denominator",
                )
            continue
        _require_finite_float(rate, f"{label} {name} rate")
        if rate != expected_rate:
            raise ValueError(f"{label} {name} rate does not match its counts")


def _validate_deltas(
    value: object,
    *,
    raw_metrics: Mapping[str, Mapping[str, Any]],
    polished_metrics: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    _require_exact_keys(value, COMPARISON_METRIC_NAMES, label)
    expected = _recomputed_deltas(raw_metrics, polished_metrics)
    for name in COMPARISON_METRIC_NAMES:
        delta = value[name]
        if expected[name] is None:
            if delta is not None:
                raise ValueError(
                    f"{label} deltas.{name} must be null when a source rate is null",
                )
            continue
        _require_finite_float(delta, f"{label} deltas.{name}")
        if delta != expected[name]:
            raise ValueError(
                f"{label} deltas.{name} does not match raw/polished metrics",
            )


def build_downstream_comparison(
    *,
    dataset_binding: Mapping[str, Any],
    semantic_target_contract: Mapping[str, Any],
    architectures: Mapping[str, Mapping[str, Any]],
    decision: str,
    diagnosis: str,
    note: str,
) -> dict[str, Any]:
    """Assemble a sealed v2 comparison envelope from already-measured rows.

    Rows must be supplied by the architecture-specific rerun artifacts; this
    helper never fabricates them.  Cell metrics and raw-vs-polished deltas are
    recomputed deterministically, then the envelope is sealed with the
    canonical content hash.  Full fail-closed validation (including dataset
    bindings) is performed separately by ``validate_downstream_comparison``.
    """
    _require_exact_keys(
        architectures, ("generative", "discriminative"), "architectures",
    )
    built_architectures: dict[str, Any] = {}
    for arch_name, family in (
        ("generative", GENERATIVE_ARCHITECTURE_FAMILY),
        ("discriminative", DISCRIMINATIVE_ARCHITECTURE_FAMILY),
    ):
        arch = architectures[arch_name]
        _require_exact_keys(
            arch, _BUILDER_ARCHITECTURE_KEYS, f"architectures.{arch_name}",
        )
        if arch["family"] != family:
            raise ValueError(f"architectures.{arch_name} family is invalid")
        for key in (
            "semantic_contract_sha256",
            "model_or_scorer_config_sha256",
            "evaluation_contract_sha256",
            "raw_input_adapter_sha256",
            "polished_input_adapter_sha256",
        ):
            _require_hex64(arch[key], f"architectures.{arch_name}.{key}")
        if arch["polished_input_adapter_sha256"] == arch["raw_input_adapter_sha256"]:
            raise ValueError(
                f"architectures.{arch_name} raw/polished input adapters must differ",
            )
        raw = arch["raw"]
        polished = arch["polished"]
        if not isinstance(raw, Mapping) or not isinstance(polished, Mapping):
            raise ValueError(f"architectures.{arch_name} cells must be objects")
        _require_exact_keys(
            raw, _BUILDER_CELL_KEYS, f"architectures.{arch_name}.raw",
        )
        _require_exact_keys(
            polished,
            _BUILDER_CELL_KEYS,
            f"architectures.{arch_name}.polished",
        )
        if raw["input_representation"] != RAW_INPUT_REPRESENTATION:
            raise ValueError(
                f"architectures.{arch_name}.raw input_representation is invalid",
            )
        if polished["input_representation"] != POLISHED_INPUT_REPRESENTATION:
            raise ValueError(
                f"architectures.{arch_name}.polished input_representation is invalid",
            )
        raw_rows = _require_list(
            raw["rows"], f"architectures.{arch_name}.raw rows",
        )
        polished_rows = _require_list(
            polished["rows"], f"architectures.{arch_name}.polished rows",
        )
        if len(raw_rows) != len(polished_rows):
            raise ValueError(
                f"architectures.{arch_name} raw/polished row counts must match",
            )
        window_count = len(raw_rows)
        if window_count == 0:
            raise ValueError("architectures cells must contain at least one row")
        raw_cell = dict(raw)
        raw_cell["metrics"] = _recomputed_cell_metrics(
            raw_rows, window_count=window_count,
        )
        polished_cell = dict(polished)
        polished_cell["metrics"] = _recomputed_cell_metrics(
            polished_rows, window_count=window_count,
        )
        built_architectures[arch_name] = {
            "family": arch["family"],
            "semantic_contract_sha256": arch["semantic_contract_sha256"],
            "model_or_scorer_config_sha256": arch["model_or_scorer_config_sha256"],
            "evaluation_contract_sha256": arch["evaluation_contract_sha256"],
            "raw_input_adapter_sha256": arch["raw_input_adapter_sha256"],
            "polished_input_adapter_sha256": arch["polished_input_adapter_sha256"],
            "raw": raw_cell,
            "polished": polished_cell,
            "deltas": _recomputed_deltas(
                raw_cell["metrics"], polished_cell["metrics"],
            ),
        }
    comparison = {
        "schema_version": DOWNSTREAM_COMPARISON_SCHEMA_VERSION,
        "comparison_complete": True,
        "dataset_binding": dict(dataset_binding),
        "semantic_target_contract": dict(semantic_target_contract),
        "architectures": built_architectures,
        "decision": decision,
        "diagnosis": diagnosis,
        "note": note,
    }
    return {
        "content_sha256": canonical_sha256(comparison),
        **comparison,
    }


def validate_downstream_comparison(
    value: object,
    *,
    label: str,
    records_obj: Mapping[str, Any],
    finalized_packet: Mapping[str, Any],
    human_summary: Mapping[str, Any],
    completed_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a v2 downstream comparison fail-closed against Phase 2K state.

    ``records_obj``/``finalized_packet``/``human_summary`` are the exact
    generated Phase 2K objects; ``completed_audit`` is the validated completed
    transformation audit for live builds and ``None`` for no-provider builds.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    _require_exact_keys(value, _TOP_LEVEL_KEYS, label)
    if value["schema_version"] != DOWNSTREAM_COMPARISON_SCHEMA_VERSION:
        raise ValueError(f"{label} schema version is invalid")
    _validate_recomputed_content_hash(value, label=label)
    _require_bool(value["comparison_complete"], f"{label} comparison_complete")
    if not value["comparison_complete"]:
        raise ValueError(f"{label} must be marked comparison_complete")

    binding = value["dataset_binding"]
    _require_exact_keys(binding, _DATASET_BINDING_KEYS, f"{label} dataset_binding")
    _require_hex64(
        binding["phase2k_records_sha256"], f"{label} dataset_binding phase2k_records_sha256",
    )
    if binding["phase2k_records_sha256"] != records_obj.get("content_sha256"):
        raise ValueError(
            f"{label} dataset_binding does not match the reconstruction records",
        )
    _require_hex64(
        binding["finalized_human_packet_sha256"],
        f"{label} dataset_binding finalized_human_packet_sha256",
    )
    if binding["finalized_human_packet_sha256"] != finalized_packet.get(
        "content_sha256",
    ):
        raise ValueError(
            f"{label} dataset_binding does not match the finalized human packet",
        )
    _require_hex64(
        binding["human_summary_sha256"], f"{label} dataset_binding human_summary_sha256",
    )
    if binding["human_summary_sha256"] != canonical_sha256(human_summary):
        raise ValueError(
            f"{label} dataset_binding does not match the human review summary",
        )
    if completed_audit is None:
        if binding["completed_transformation_audit_sha256"] is not None:
            raise ValueError(
                f"{label} completed_transformation_audit_sha256 must be null "
                "without a transformation audit",
            )
    else:
        _require_hex64(
            binding["completed_transformation_audit_sha256"],
            f"{label} dataset_binding completed_transformation_audit_sha256",
        )
        if binding["completed_transformation_audit_sha256"] != completed_audit.get(
            "content_sha256",
        ):
            raise ValueError(
                f"{label} dataset_binding does not match the completed "
                "transformation audit",
            )
    window_ids = _records_window_ids(records_obj)
    window_count = _require_int(
        binding["window_count"], f"{label} dataset_binding window_count", minimum=1,
    )
    if window_count != len(window_ids):
        raise ValueError(
            f"{label} dataset_binding window_count does not match the dataset",
        )
    _require_hex64(
        binding["window_ids_sha256"], f"{label} dataset_binding window_ids_sha256",
    )
    if binding["window_ids_sha256"] != canonical_sha256(window_ids):
        raise ValueError(
            f"{label} dataset_binding window_ids_sha256 does not match the dataset",
        )
    human_gate = human_summary.get("review_gate", {}).get("status")
    if binding["human_review_gate_status"] != "PASSED" or human_gate != "PASSED":
        raise ValueError(
            f"{label} requires human_review_gate_status PASSED",
        )

    contract = value["semantic_target_contract"]
    _require_exact_keys(
        contract, _SEMANTIC_TARGET_CONTRACT_KEYS, f"{label} semantic_target_contract",
    )
    _require_nonempty_string(
        contract["contract_version"], f"{label} semantic_target_contract contract_version",
    )
    _require_hex64(
        contract["contract_sha256"], f"{label} semantic_target_contract contract_sha256",
    )
    target_count = _require_int(
        contract["target_count"],
        f"{label} semantic_target_contract target_count",
        minimum=1,
    )
    _require_nonempty_string(
        contract["boundary_rule"], f"{label} semantic_target_contract boundary_rule",
    )

    architectures = value["architectures"]
    _require_exact_keys(
        architectures, ("generative", "discriminative"), f"{label} architectures",
    )
    shared_target_counts: list[int] | None = None
    for arch_name, family in (
        ("generative", GENERATIVE_ARCHITECTURE_FAMILY),
        ("discriminative", DISCRIMINATIVE_ARCHITECTURE_FAMILY),
    ):
        arch = architectures[arch_name]
        _require_exact_keys(arch, _ARCHITECTURE_KEYS, f"{label} architectures.{arch_name}")
        if arch["family"] != family:
            raise ValueError(
                f"{label} architectures.{arch_name} family is invalid",
            )
        for key in (
            "semantic_contract_sha256",
            "model_or_scorer_config_sha256",
            "evaluation_contract_sha256",
            "raw_input_adapter_sha256",
            "polished_input_adapter_sha256",
        ):
            _require_hex64(
                arch[key], f"{label} architectures.{arch_name}.{key}",
            )
        if arch["polished_input_adapter_sha256"] == arch["raw_input_adapter_sha256"]:
            raise ValueError(
                f"{label} architectures.{arch_name} raw/polished input "
                "adapters must differ",
            )
        for cell_name, representation in (
            ("raw", RAW_INPUT_REPRESENTATION),
            ("polished", POLISHED_INPUT_REPRESENTATION),
        ):
            cell = arch[cell_name]
            rows = _validate_cell_rows(
                cell,
                label=f"{label} architectures.{arch_name}.{cell_name}",
                expected_representation=representation,
                expected_window_ids=window_ids,
            )
            target_counts = [row["target_count"] for row in rows]
            if shared_target_counts is None:
                shared_target_counts = target_counts
            elif target_counts != shared_target_counts:
                raise ValueError(
                    f"{label} per-window target_count must be identical across "
                    "every raw/polished cell and architecture",
                )
            _validate_metrics(
                cell["metrics"],
                rows=rows,
                window_count=window_count,
                label=f"{label} architectures.{arch_name}.{cell_name} metrics",
            )
        _validate_deltas(
            arch["deltas"],
            raw_metrics=arch["raw"]["metrics"],
            polished_metrics=arch["polished"]["metrics"],
            label=f"{label} architectures.{arch_name}",
        )

    if shared_target_counts is None:
        raise ValueError(f"{label} architectures are empty")
    if sum(shared_target_counts) != target_count:
        raise ValueError(
            f"{label} summed target_count does not match the semantic target contract",
        )

    decision = _require_enum(
        value["decision"], FINAL_CLOSEOUT_STATUSES, f"{label} decision",
    )
    diagnosis = _require_enum(
        value["diagnosis"], DOWNSTREAM_DIAGNOSIS_VALUES, f"{label} diagnosis",
    )
    note = _require_nonempty_string(value["note"], f"{label} note")
    return dict(value)
