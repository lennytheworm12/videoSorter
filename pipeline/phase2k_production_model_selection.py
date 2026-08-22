"""Phase 2K production-model selection over the frozen full-context benchmark.

This module extends the completed Phase 2K full-transcript context ablation
with three new conditions over the SAME frozen 10-target benchmark, the SAME
Condition B inputs (full transcript + marked target + metadata + vocabulary +
byte-identical extraction instructions), and the SAME grounding/scoring
contract:

  P   DeepSeek V4 Pro, one independent full-context extraction per target.
  F   DeepSeek V4 Flash, one independent full-context extraction per target.
  FV  DeepSeek V4 Flash Best-of-5 plus a selection-only Flash verifier:
      five genuinely independent generator calls per target followed by one
      verifier call that MUST select exactly one of the five existing
      candidates.  The verifier may not merge, rewrite, repair, or extend any
      candidate, and may not produce a sixth answer.

The frozen 0x Alpha full-context result (109/110 strict successes) remains the
capability baseline; the settled isolated-vs-full question is never reopened.

This module contains no model calls and uses only the Python standard
library.  Live execution happens through the companion script via the OpenCode
CLI, exactly like the original ablation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.phase2j_context_ablation import (
    _require_exact_keys,
    _require_nonempty_string,
    canonical_sha256,
    text_sha256,
)
from pipeline.phase2k_full_transcript_ablation import (
    ArtifactError,
    CONDITION_CODES,
    DEFAULT_OUTPUT_DIR,
    INTERMEDIATE_SCHEMA_VERSION,
    REVIEW_FIELDS,
    SEMANTIC_FIELDS,
)

# ---------------------------------------------------------------------------
# Versions, models, configuration
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "phase2k-production-model-selection-v1"
RUN_SCHEMA_VERSION = "phase2k-production-model-selection-run-v1"
VERIFIER_PAYLOAD_SCHEMA_VERSION = (
    "phase2k-production-model-verifier-payload-v1"
)
VERIFIER_RESPONSE_SCHEMA_VERSION = (
    "phase2k-production-model-verifier-response-v1"
)
CONDITION_SUMMARY_SCHEMA_VERSION = (
    "phase2k-production-model-condition-summary-v1"
)
SELECTION_REPORT_SCHEMA_VERSION = (
    "phase2k-production-model-selection-report-v1"
)

BASELINE_MODEL = "opencode-go/ox-alpha-free"
MODEL_PRO = "opencode-go/deepseek-v4-pro"
MODEL_FLASH = "opencode-go/deepseek-v4-flash"

# One configuration per model, frozen before any live benchmark call: default
# transport behavior only, no thinking/variant flags, no tuning afterwards.
CONDITION_MODELS = {
    "P": MODEL_PRO,
    "F": MODEL_FLASH,
    "FV": MODEL_FLASH,
}
VERIFIER_MODEL = MODEL_FLASH
CANDIDATE_COUNT = 5
CALLS_PER_TARGET = {"P": 1, "F": 1, "FV": CANDIDATE_COUNT + 1}

NEW_CONDITION_CODES = ("P", "F", "FV")
ALL_CONDITION_CODES = ("OX",) + NEW_CONDITION_CODES

VERIFIER_ORDER_SALT = "phase2k-verifier-candidate-order-v1"

DEFAULT_PROD_SEL_DIR = DEFAULT_OUTPUT_DIR / "prod_sel_v1"

# Substantive fields for the verifier-usefulness gate.
SUBSTANTIVE_FIELDS = (
    "actors_entities",
    "reference_bindings",
    "abilities_resources",
    "events_actions",
    "states",
    "conditions",
    "explicit_relationships",
)

KEY_GATE_FIELDS = (
    "actors_entities",
    "reference_bindings",
    "abilities_resources",
    "explicit_relationships",
)


class SelectionError(ValueError):
    """Raised when a production-model selection artifact fails validation."""


def candidate_call_id(case_id: str, index: int) -> str:
    return f"{case_id}/fv/candidate_{index}"


def verifier_call_id(case_id: str) -> str:
    return f"{case_id}/fv/verifier"


# ---------------------------------------------------------------------------
# Deterministic candidate ordering (auditable positional-bias control)
# ---------------------------------------------------------------------------


def deterministic_candidate_order(case_id: str) -> list[str]:
    """Shuffled candidate_1..candidate_5 ids, deterministic per target."""
    ids = [f"candidate_{i}" for i in range(1, CANDIDATE_COUNT + 1)]
    seed = canonical_sha256([VERIFIER_ORDER_SALT, case_id])
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


# ---------------------------------------------------------------------------
# Verifier payload construction
# ---------------------------------------------------------------------------


def build_verifier_payload(
    *,
    payload_b: Mapping[str, Any],
    candidate_responses: Mapping[str, Mapping[str, Any]],
    candidate_order: Iterable[str],
) -> dict[str, Any]:
    """Build the model-visible verifier payload.

    ``candidate_responses`` maps candidate id -> validated intermediate
    extraction response.  Candidates are presented in the supplied
    deterministic order; nothing about sibling cases or the baseline is
    exposed.
    """
    order = list(candidate_order)
    if sorted(order) != [
        f"candidate_{i}" for i in range(1, CANDIDATE_COUNT + 1)
    ]:
        raise SelectionError("verifier payload requires candidates 1..5")
    presented = []
    for candidate_id in order:
        response = candidate_responses[candidate_id]
        validate_candidate_response_binding(
            response,
            case_id=payload_b["case_id"],
            payload=payload_b,
        )
        presented.append({
            "candidate_id": candidate_id,
            "extraction": dict(response),
        })
    verifier_payload = {
        "schema_version": VERIFIER_PAYLOAD_SCHEMA_VERSION,
        "case_id": payload_b["case_id"],
        "task": (
            "You are a strict selection-only verifier.  Five independent "
            "source-grounded semantic extractions of the SAME target passage "
            "are supplied below in a numbered presentation order.  Select "
            "exactly ONE candidate that best preserves source-supported "
            "semantics.  You must NOT merge candidates, rewrite or repair "
            "any candidate, add new semantic claims, or produce your own "
            "extraction.  Prefer correct abstention and honestly unresolved "
            "references over plausible but unsupported League inference."
        ),
        "target": dict(payload_b["target"]),
        "transcript": payload_b["transcript"],
        "target_char_start": payload_b["target_char_start"],
        "target_char_end": payload_b["target_char_end"],
        "metadata": dict(payload_b["metadata"]),
        "metadata_fields_supplied": list(
            payload_b["metadata_fields_supplied"],
        ),
        "vocabulary_sha256": payload_b["vocabulary_sha256"],
        "extraction_instructions_sha256": payload_b["instructions_sha256"],
        "extraction_schema_version": INTERMEDIATE_SCHEMA_VERSION,
        "candidates": presented,
        "selection_contract": {
            "response_keys": [
                "schema_version",
                "case_id",
                "selected_candidate_id",
                "rationale",
            ],
            "optional_keys": ["criteria_scores"],
            "selected_candidate_id": (
                "exactly one of the supplied candidate_id values; any other "
                "value is invalid"
            ),
            "criteria_scores": (
                "optional object mapping candidate_id to an object of "
                "integer criterion scores such as source_faithfulness, "
                "entity_correctness, reference_binding, ability_ownership, "
                "event_correctness, uncertainty_preservation, "
                "evidence_grounding"
            ),
            "rationale": "non-empty brief justification",
        },
    }
    envelope = {
        "verifier_payload": verifier_payload,
        "content_sha256": "",
    }
    envelope["content_sha256"] = canonical_sha256(
        {k: v for k, v in envelope.items() if k != "content_sha256"},
    )
    return envelope


def validate_candidate_response_binding(
    response: Mapping[str, Any],
    *,
    case_id: str,
    payload: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        response,
        (
            "schema_version", "case_id", "condition", "payload_sha256",
            "instructions_sha256", "fields",
        ),
        "verifier candidate response",
    )
    if response["case_id"] != case_id:
        raise SelectionError("candidate response case_id mismatch")
    if response["payload_sha256"] != payload["content_sha256"]:
        raise SelectionError(
            "candidate response is not bound to the frozen Condition B payload",
        )


# ---------------------------------------------------------------------------
# Verifier response validation (selection-only contract)
# ---------------------------------------------------------------------------


def validate_verifier_response(
    response: Mapping[str, Any],
    *,
    case_id: str,
    candidate_order: Iterable[str],
) -> str:
    """Strictly validate a verifier response; return the selected candidate id."""
    allowed_optional = {"criteria_scores"}
    required = {
        "schema_version", "case_id", "selected_candidate_id", "rationale",
    }
    keys = set(response)
    if not required <= keys:
        raise SelectionError(
            f"verifier response missing required keys: "
            f"{sorted(required - keys)}",
        )
    extra = keys - required - allowed_optional
    if extra:
        raise SelectionError(
            f"verifier response has unexpected keys: {sorted(extra)}",
        )
    if response["schema_version"] != VERIFIER_RESPONSE_SCHEMA_VERSION:
        raise SelectionError("verifier response schema version is invalid")
    if response["case_id"] != case_id:
        raise SelectionError("verifier response case_id is invalid")
    selected = response["selected_candidate_id"]
    if not isinstance(selected, str):
        raise SelectionError("verifier selected_candidate_id must be a string")
    order = list(candidate_order)
    if selected not in order:
        raise SelectionError(
            f"verifier selected {selected!r} which is not one of the "
            f"presented candidates {order}; synthesized answers are forbidden",
        )
    rationale = response["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise SelectionError("verifier rationale must be a non-empty string")
    if "criteria_scores" in keys:
        scores = response["criteria_scores"]
        if not isinstance(scores, dict):
            raise SelectionError("verifier criteria_scores must be an object")
        for candidate_id, entry in scores.items():
            if candidate_id not in order:
                raise SelectionError(
                    f"criteria_scores references unknown candidate "
                    f"{candidate_id!r}",
                )
            if not isinstance(entry, dict):
                raise SelectionError(
                    "criteria_scores entries must be objects",
                )
    return selected


# ---------------------------------------------------------------------------
# Selection integrity
# ---------------------------------------------------------------------------


def check_selection_integrity(
    *,
    final_output: Mapping[str, Any],
    candidate_outputs: Mapping[str, Mapping[str, Any]],
    selected_candidate_id: str,
) -> None:
    """The FV final output must equal the selected candidate exactly."""
    if selected_candidate_id not in candidate_outputs:
        raise SelectionError(
            f"selected candidate {selected_candidate_id!r} has no output",
        )
    selected = candidate_outputs[selected_candidate_id]
    if canonical_sha256(final_output) != canonical_sha256(selected):
        raise SelectionError(
            "FV final output differs from the selected candidate; merging "
            "or rewriting is forbidden",
        )


# ---------------------------------------------------------------------------
# Frozen-contract scoring aggregation and gates
# ---------------------------------------------------------------------------


def _strict_success(entry: Mapping[str, Any]) -> bool:
    return (
        entry["correctness"] in ("CORRECT", "ABSENT_CORRECTLY")
        and entry["unsupported_inference"] == "NONE"
        and entry["source_grounding"] in ("GROUNDED", "NOT_APPLICABLE")
    )


def compute_condition_metrics(
    reviews: Mapping[str, Mapping[str, Any]],
    *,
    condition: str,
    case_ids: Iterable[str],
) -> dict[str, Any]:
    """Aggregate frozen-contract review scores for one condition.

    ``reviews`` is keyed ``{case_id}:{condition}:{field}`` with entries
    carrying correctness / unsupported_inference / source_grounding.
    """
    case_ids = list(case_ids)
    expected = {
        f"{case_id}:{condition}:{field}"
        for case_id in case_ids
        for field in REVIEW_FIELDS
    }
    if set(reviews) != expected:
        missing = sorted(expected - set(reviews))
        extra = sorted(set(reviews) - expected)
        raise SelectionError(
            f"condition {condition} reviews incomplete: "
            f"missing={missing[:5]} extra={extra[:5]}",
        )

    def strict(case_id: str, field: str) -> bool:
        return _strict_success(reviews[f"{case_id}:{condition}:{field}"])

    per_field = {}
    for field in REVIEW_FIELDS:
        successes = [strict(case_id, field) for case_id in case_ids]
        per_field[field] = {
            "successes": sum(successes),
            "total": len(case_ids),
        }

    per_target = {}
    for case_id in case_ids:
        per_target[case_id] = {
            "successes": sum(strict(case_id, f) for f in REVIEW_FIELDS),
            "total": len(REVIEW_FIELDS),
        }

    unsupported_fields = 0
    major_unsupported = 0
    grounding_failures = 0
    unresolved_preserving = 0
    for key, entry in reviews.items():
        if entry["unsupported_inference"] != "NONE":
            unsupported_fields += 1
        if entry["unsupported_inference"] == "MAJOR":
            major_unsupported += 1
        if entry["source_grounding"] == "UNGROUNDED":
            grounding_failures += 1
        if (
            entry["correctness"] == "CORRECT"
            and key.split(":")[-1] == "uncertainty_unresolved"
        ):
            unresolved_preserving += 1

    total_strict = sum(t["successes"] for t in per_target.values())
    metrics = {
        "condition": condition,
        "total_strict_successes": total_strict,
        "total_judgments": len(reviews),
        "per_field": per_field,
        "per_target": per_target,
        "unsupported_field_count": unsupported_fields,
        "major_unsupported_count": major_unsupported,
        "grounding_failure_count": grounding_failures,
        "unresolved_field_successes": unresolved_preserving,
    }
    return metrics


def evaluate_condition_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen PASS / CONDITIONAL PASS / FAIL gates."""
    total = metrics["total_strict_successes"]
    key = {
        field: metrics["per_field"][field]["successes"]
        for field in KEY_GATE_FIELDS
    }
    major = metrics["major_unsupported_count"]
    grounding_failures = metrics["grounding_failure_count"]
    unsupported_fields = metrics["unsupported_field_count"]

    hard_fail_reasons = []
    if major > 0:
        hard_fail_reasons.append("major unsupported claim present")
    if grounding_failures > 0:
        hard_fail_reasons.append("grounding failure indicates broken source discipline")

    pass_gate = (
        total >= 104
        and all(v >= 9 for v in key.values())
        and major == 0
        and grounding_failures == 0
        and unsupported_fields <= 1
    )
    conditional_gate = (
        total >= 99
        and all(v >= 8 for v in key.values())
        and major == 0
        and grounding_failures == 0
        and unsupported_fields <= 2
    )
    if hard_fail_reasons:
        outcome = "FAIL"
    elif pass_gate:
        outcome = "PASS"
    elif conditional_gate:
        outcome = "CONDITIONAL_PASS"
    else:
        outcome = "FAIL"
    return {
        "outcome": outcome,
        "hard_fail_reasons": hard_fail_reasons,
        "gate_checks": {
            "pass_total_ge_104": total >= 104,
            "pass_key_fields_ge_9": all(v >= 9 for v in key.values()),
            "conditional_total_ge_99": total >= 99,
            "conditional_key_fields_ge_8": all(v >= 8 for v in key.values()),
            "no_major_unsupported": major == 0,
            "no_grounding_failures": grounding_failures == 0,
            "pass_unsupported_le_1": unsupported_fields <= 1,
            "conditional_unsupported_le_2": unsupported_fields <= 2,
        },
    }


def evaluate_verifier_usefulness(
    *,
    flash_metrics: Mapping[str, Any],
    flash_verifier_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Frozen gate deciding whether Best-of-5 + verifier is justified."""
    delta_total = (
        flash_verifier_metrics["total_strict_successes"]
        - flash_metrics["total_strict_successes"]
    )
    improved_targets = 0
    worsened_targets = 0
    case_ids = sorted(flash_metrics["per_target"])
    for case_id in case_ids:
        f_t = flash_metrics["per_target"][case_id]["successes"]
        fv_t = flash_verifier_metrics["per_target"][case_id]["successes"]
        if fv_t > f_t:
            improved_targets += 1
        elif fv_t < f_t:
            worsened_targets += 1
    improved_substantive = [
        field
        for field in SUBSTANTIVE_FIELDS
        if flash_verifier_metrics["per_field"][field]["successes"]
        > flash_metrics["per_field"][field]["successes"]
    ]
    major_delta = (
        flash_verifier_metrics["major_unsupported_count"]
        - flash_metrics["major_unsupported_count"]
    )
    grounding_delta = (
        flash_verifier_metrics["grounding_failure_count"]
        - flash_metrics["grounding_failure_count"]
    )
    checks = {
        "strict_improvement_ge_3": delta_total >= 3,
        "improves_ge_2_targets": improved_targets >= 2,
        "worsens_le_1_target": worsened_targets <= 1,
        "improves_substantive_field": bool(improved_substantive),
        "no_major_unsupported_increase": major_delta <= 0,
        "no_grounding_failure_increase": grounding_delta <= 0,
    }
    useful = all(checks.values())
    return {
        "delta_strict_successes": delta_total,
        "improved_targets": improved_targets,
        "worsened_targets": worsened_targets,
        "improved_substantive_fields": improved_substantive,
        "checks": checks,
        "decision": (
            "VERIFIER_SCALING_USEFUL" if useful
            else "VERIFIER_SCALING_NOT_JUSTIFIED"
        ),
    }


PRODUCTION_RECOMMENDATIONS = {
    "F": "V4_FLASH_SINGLE_PASS_PROMOTED",
    "FV": "V4_FLASH_VERIFIER_PROMOTED",
    "P": "V4_PRO_PROMOTED",
}


def select_production_model(
    *,
    gates: Mapping[str, Mapping[str, Any]],
    cost_per_target_seconds: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """Choose the cheapest reliable passing condition.

    Quality gates override nominal model preference; among passing
    conditions, actual measured cost per target decides, with the documented
    heuristic order (Flash single-pass, then Flash+verifier, then Pro) as the
    tie-break when cost evidence is unavailable or equal.
    """
    preference = ["F", "FV", "P"]
    passing = [c for c in preference if gates[c]["outcome"] == "PASS"]
    conditional = [
        c for c in preference if gates[c]["outcome"] == "CONDITIONAL_PASS"
    ]
    if passing:
        chosen_pool = passing
        basis = "PASS"
    elif conditional:
        chosen_pool = conditional
        basis = "CONDITIONAL_PASS (not automatically promoted; errors require review)"
    else:
        return {
            "recommendation": "NO_DEEPSEEK_CONFIGURATION_MEETS_PRODUCTION_GATE",
            "basis": "no condition reached even the conditional gate",
            "cost_comparison_used": False,
        }
    costs = cost_per_target_seconds or {}
    available = {
        c: costs.get(c)
        for c in chosen_pool
        if isinstance(costs.get(c), (int, float)) and costs[c] > 0
    }
    if len(available) == len(chosen_pool) and len(chosen_pool) > 1:
        cheapest = min(chosen_pool, key=lambda c: costs[c])
        cost_used = True
    else:
        cheapest = chosen_pool[0]
        cost_used = False
    return {
        "recommendation": PRODUCTION_RECOMMENDATIONS[cheapest],
        "cheapest_measured_condition": cheapest if cost_used else None,
        "preference_first_passing_condition": chosen_pool[0],
        "basis": basis,
        "cost_comparison_used": cost_used,
    }
