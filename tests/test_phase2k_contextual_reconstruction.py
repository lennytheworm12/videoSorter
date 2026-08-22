"""Focused tests for the isolated Phase 2K contextual-reconstruction core."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import pipeline.phase2k_contextual_reconstruction as p2k_module
from pipeline.phase2k_contextual_reconstruction import (
    CONFIG_VERSION,
    HUMAN_SCORE_FIELDS,
    INFERENCE_CONFIG_VERSION,
    LEAGUE_VOCABULARY_SCHEMA_VERSION,
    MECHANICAL_CORRECTION_PROMPT_VERSION,
    MECHANICAL_MAX_CORRECTIONS,
    MECHANICAL_PROMPT_VERSION,
    MECHANICAL_UNCERTAINTY_CAP,
    MECHANICAL_RESPONSE_SCHEMA_VERSION,
    NO_PROVIDER_INFERENCE_CONFIG,
    NOT_APPLICABLE,
    OUTPUT_FILENAMES,
    PIPELINE_VERSION,
    POLISH_CORRECTION_PROMPT_VERSION,
    POLISH_MAX_CORRECTIONS,
    POLISH_PROMPT_VERSION,
    POLISH_SUPPORT_MODES,
    POLISH_RESPONSE_SCHEMA_VERSION,
    PROVIDER_MAX_CORRECTIONS,
    ProviderCorrectionExhausted,
    RECONSTRUCTION_CORRECTION_PROMPT_VERSION,
    RECONSTRUCTION_MAX_CORRECTIONS,
    RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
    RECORDS_SCHEMA_VERSION,
    REVIEW_GATE_SPEC,
    SLOT_KEYS,
    SUFFICIENCY_NORMALIZED_SCHEMA_VERSION,
    SUFFICIENCY_RESPONSE_SCHEMA_VERSION,
    TEXT_RESTORATION_TASK_KIND,
    CONTEXTUAL_REPAIR_TYPES,
    _cache_key,
    _reconstruction_operation_ids,
    _bind_evidence_quotes,
    apply_mechanical_repairs,
    _bind_repair_proposals,
    _validate_reconstruction_raw_compact,
    build_mechanical_prompt,
    build_mechanical_provenance,
    build_human_review_packet,
    build_phase2k_outputs,
    build_polish_correction_prompt,
    build_reconstruction_correction_prompt,
    cache_load,
    cache_store,
    canonical_sha256,
    call_provider,
    detect_champion_alias_hints,
    evaluate_review_gate,
    import_completed_human_reviews,
    inference_config_hash,
    load_json_strict,
    normalize_path_locator,
    normalize_sufficiency_compact_response,
    parse_provider_json,
    retrieve_context,
    run_adaptive_diagnostics,
    run_mechanical_cleanup,
    run_polish,
    run_reconstruction,
    run_sufficiency_diagnostic,
    summarize_human_reviews,
    validate_context,
    validate_human_review_packet,
    validate_inference_config,
    validate_output_directory,
    validate_polish_response,
    validate_sufficiency_response,
    _envelope,
    text_sha256,
)
from tests._phase2k_helpers import (
    CHAMPIONS,
    GENERIC_TRANSCRIPT,
    build_fixture,
    make_selected,
)


TEST_LIVE_INFERENCE_CONFIG = {
    "provider": "test-backend",
    "model": "test-model",
    "endpoint": "https://example.test/endpoint",
    "temperature": 0.0,
    "max_tokens": 8192,
    "thinking": "disabled",
    "purpose": "phase2k-test-live",
}


def _selected(
    transcript: str,
    *,
    champion: str = "Lux",
    target: str | None = None,
) -> dict[str, Any]:
    if target is None:
        target = transcript
    start = transcript.index(target)
    return make_selected(
        "testvid",
        transcript,
        start,
        start + len(target),
        index=1,
        champion=champion,
        role="mid",
        video_title=f"Video {champion}",
    )


def _slots(
    *,
    decision: str,
    champion: str = "Lux",
    unresolved: str | None = None,
) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    for key in SLOT_KEYS:
        if unresolved == key:
            slots[key] = {
                "status": "UNKNOWN",
                "candidates": [],
                "confidence": "LOW",
                "evidence_spans": [],
            }
        else:
            slots[key] = {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "NONE",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            }
    if decision == "SUFFICIENT":
        return slots
    if unresolved is None:
        slots["pronouns"] = {
            "status": "AMBIGUOUS",
            "candidates": [{
                "candidate": "Lux",
                "confidence": "MEDIUM",
                "evidence_spans": [],
            }],
            "confidence": "MEDIUM",
            "evidence_spans": [],
        }
    return slots


def _compact_slots_from_normalized(
    normalized_slots: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert normalized slots (evidence_spans) to compact evidence_quotes."""
    compact: dict[str, Any] = {}
    for key, slot in normalized_slots.items():
        candidates = []
        for candidate in slot.get("candidates", []):
            candidates.append({
                "candidate": candidate["candidate"],
                "confidence": candidate["confidence"],
                "evidence_quotes": [
                    span["text"]
                    for span in candidate.get("evidence_spans", [])
                ],
            })
        compact[key] = {
            "status": slot["status"],
            "candidates": candidates,
            "confidence": slot["confidence"],
            "evidence_quotes": [
                span["text"]
                for span in slot.get("evidence_spans", [])
            ],
        }
    return compact


def _compact_slots(
    *,
    decision: str,
    champion: str = "Lux",
    unresolved: str | None = None,
) -> dict[str, Any]:
    """Normalized-shape slots converted to the compact provider v2 shape."""
    return _compact_slots_from_normalized(
        _slots(decision=decision, unresolved=unresolved),
    )


def _sufficiency_raw(
    decision: str,
    *,
    champion: str = "Lux",
    unresolved: str | None = None,
    conflicts: list[dict[str, Any]] | None = None,
    slots: Mapping[str, Any] | None = None,
) -> str:
    return json.dumps({
        "schema_version": SUFFICIENCY_RESPONSE_SCHEMA_VERSION,
        "decision": decision,
        "slots": _compact_slots(
            decision=decision, champion=champion, unresolved=unresolved,
        ) if slots is None else _compact_slots_from_normalized(slots),
        "metadata_conflicts": conflicts or [],
        "rationale": "test rationale",
    })


def _compact_sufficiency_raw(
    decision: str,
    *,
    slots: Mapping[str, Any] | None = None,
    conflicts: list[dict[str, Any]] | None = None,
    rationale: str = "test rationale",
    schema_version: str | None = None,
) -> str:
    """Emit a compact v2 sufficiency raw with full slot-shape control."""
    return json.dumps({
        "schema_version": schema_version or SUFFICIENCY_RESPONSE_SCHEMA_VERSION,
        "decision": decision,
        "slots": _compact_slots_from_normalized(
            slots if slots is not None else _slots(decision=decision),
        ),
        "metadata_conflicts": conflicts or [],
        "rationale": rationale,
    })


def _mechanical_raw(
    selected: Mapping[str, Any],
    *,
    repairs: list[dict[str, Any]] | None = None,
    uncertainties: list[dict[str, Any]] | None = None,
    clean_text: str | None = None,
    rationale: str = "test",
) -> str:
    """Compact v3 provider response for one window.

    ``repairs``/``uncertainties`` may be given in the old full-format test
    shape; they are converted to compact proposals before emission.  When
    ``clean_text`` is omitted it is computed by deterministic binding so the
    emitted raw response is always internally consistent.
    """
    bronze = selected["source_text"]
    repair_proposals = [_repair_proposal(item) for item in (repairs or [])]
    uncertainty_proposals = [
        _uncertainty_proposal(item) for item in (uncertainties or [])
    ]
    if clean_text is None:
        bound = _bind_repair_proposals(
            repair_proposals, bronze_text=bronze, selected=selected,
        )
        clean_text = apply_mechanical_repairs(bronze, bound)
    return json.dumps({
        "schema_version": MECHANICAL_RESPONSE_SCHEMA_VERSION,
        "clean_text": clean_text,
        "repairs": repair_proposals,
        "uncertainties": uncertainty_proposals,
        "rationale": rationale,
    })


def _repair_proposal(repair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "original_text": repair["original_text"],
        "replacement": repair["replacement"],
        "repair_type": repair["repair_type"],
        "confidence": repair["confidence"],
        "rationale": repair["rationale"],
    }


def _uncertainty_proposal(uncertainty: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "surface_text": uncertainty["text"],
        "uncertainty_type": uncertainty["uncertainty_type"],
        "alternatives": uncertainty["alternatives"],
        "note": uncertainty["note"],
    }


def _compact_mechanical_raw(
    selected: Mapping[str, Any],
    *,
    clean_text: str,
    repairs: list[dict[str, Any]] | None = None,
    uncertainties: list[dict[str, Any]] | None = None,
    rationale: str = "test",
) -> str:
    """Emit a compact v3 provider response with proposal-level control."""
    return json.dumps({
        "schema_version": MECHANICAL_RESPONSE_SCHEMA_VERSION,
        "clean_text": clean_text,
        "repairs": repairs or [],
        "uncertainties": uncertainties or [],
        "rationale": rationale,
    })


def _uncertainty(
    selected: Mapping[str, Any],
    *,
    local_start: int,
    local_end: int,
    uncertainty_type: str = "ASR_ALTERNATIVES",
    alternatives: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = selected["source_text"][local_start:local_end]
    return {
        "uncertainty_id": f"p2k:test:unc:{local_start}",
        "target_local_start": local_start,
        "target_local_end": local_end,
        "source_absolute_start": selected["upstream_start"] + local_start,
        "source_absolute_end": selected["upstream_start"] + local_end,
        "text": text,
        "uncertainty_type": uncertainty_type,
        "alternatives": alternatives or [
            {"text": text, "confidence": "LOW"},
        ],
        "evidence": [
            {
                "target_local_start": local_start,
                "target_local_end": local_end,
                "text": text,
            },
        ],
        "note": "ambiguous without an identity/ownership decision",
    }


def _reconstruction_raw(
    *,
    cleaned: str,
    bronze: str,
    base_offset: int,
    selected: Mapping[str, Any],
    contextual_repairs: list[dict[str, Any]] | None = None,
    bindings_override: list[dict[str, Any]] | None = None,
    unresolved_alternatives: list[dict[str, Any]] | None = None,
) -> str:
    if bindings_override is not None:
        bindings = bindings_override
    else:
        bindings = []
        for match in re.finditer(r"\bYou\b", bronze):
            local_start, local_end = match.start(), match.end()
            bindings.append({
                "slot": "pronouns",
                "mention_text": match.group(),
                "resolved_candidate": selected["metadata"]["champion"],
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": [],
                "alternatives": [],
                "metadata_contributed": True,
                "rationale": "test binding",
            })
    return json.dumps({
        "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
        "clean_target_transcript": cleaned,
        "contextual_repairs": contextual_repairs or [],
        "bindings": bindings,
        "unresolved_alternatives": unresolved_alternatives or [],
        "rationale": "test",
    })


def _bronze_span(
    selected: Mapping[str, Any],
    *,
    local_start: int,
    local_end: int,
) -> dict[str, Any]:
    return {
        "target_local_start": local_start,
        "target_local_end": local_end,
        "source_absolute_start": selected["upstream_start"] + local_start,
        "source_absolute_end": selected["upstream_start"] + local_end,
        "text": selected["source_text"][local_start:local_end],
    }


def _polish_raw(
    selected: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    *,
    statements: list[dict[str, Any]] | None = None,
    unsupported: list[dict[str, Any]] | None = None,
) -> str:
    if statements is None:
        operation_ids = [
            item["repair_id"] for item in reconstruction["contextual_repairs"]
        ] + [
            item["binding_id"] for item in reconstruction["bindings"]
        ]
        if operation_ids:
            statements = [{
                "text": reconstruction["clean_target_transcript"],
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": [selected["source_text"]],
                "reconstruction_operation_ids": operation_ids,
                "support_mode": "RECONSTRUCTION_DERIVED",
                "unchanged_source_quote": None,
            }]
        else:
            statements = [{
                "text": reconstruction["clean_target_transcript"],
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": [selected["source_text"]],
                "reconstruction_operation_ids": [],
                "support_mode": "UNCHANGED_EXACT",
                "unchanged_source_quote": selected["source_text"],
            }]
    return json.dumps({
        "schema_version": POLISH_RESPONSE_SCHEMA_VERSION,
        "statements": statements,
        "unsupported_claims": unsupported or [],
        "rationale": "test",
    })


class CountingChat:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected provider call")
        return self.responses.pop(0)


def _live_factory_chat() -> Any:
    """Deterministic live-mode chat producing one clean run per window."""
    def chat(system: str, user: str) -> str:
        payload = json.loads(user)
        task = payload.get("task")
        if task == "mechanical_cleanup":
            flat_metadata = {
                key: payload["metadata"][key]["value"]
                for key in ("champion", "role", "video_title")
                if key in payload["metadata"]
            }
            return _mechanical_raw({
                **payload["target"],
                "source_text": payload["target"]["bronze_text"],
                "upstream_start": payload["target"]["upstream_start"],
                "upstream_end": payload["target"]["upstream_end"],
                "upstream_content_sha256": payload["target"]["upstream_content_sha256"],
                "canonical_record_sha256": payload["target"]["canonical_record_sha256"],
                "metadata": flat_metadata,
            })
        if task == "semantic_sufficiency":
            champion = (
                payload["metadata"].get("champion", {}).get("value", "Lux")
            )
            return _sufficiency_raw(
                "SUFFICIENT", champion=champion,
            )
        if task == "reconstruction":
            bronze = payload["target"]["bronze_text"]
            selected = {
                "source_text": bronze,
                "upstream_start": payload["target"]["upstream_start"],
                "upstream_end": payload["target"]["upstream_end"],
                "upstream_content_sha256": payload["target"]["upstream_content_sha256"],
                "window_id": payload["target"]["window_id"],
                "source_group_id": payload["target"]["source_group_id"],
                "canonical_record_sha256": payload["target"]["canonical_record_sha256"],
                "metadata": {
                    key: payload["metadata"][key]["value"]
                    for key in ("champion", "role", "video_title")
                    if key in payload["metadata"]
                },
            }
            return _reconstruction_raw(
                cleaned=bronze,
                bronze=bronze,
                base_offset=payload["target"]["upstream_start"],
                selected=selected,
            )
        if task == "semantic_polish":
            # The payload carries the validated reconstruction subobject.
            reconstruction = payload["reconstruction"]
            selected = {
                "source_text": payload["target"]["bronze_text"],
                "upstream_start": payload["target"]["upstream_start"],
                "upstream_end": payload["target"]["upstream_end"],
                "upstream_content_sha256": payload["target"]["upstream_content_sha256"],
                "window_id": payload["target"]["window_id"],
                "source_group_id": payload["target"]["source_group_id"],
                "canonical_record_sha256": payload["target"]["canonical_record_sha256"],
                "metadata": {
                    key: payload["metadata"][key]["value"]
                    for key in ("champion", "role", "video_title")
                    if key in payload["metadata"]
                },
            }
            return _polish_raw(selected, reconstruction)
        raise AssertionError(f"unknown task {task}")
    return chat


class Phase2KCoreTests(unittest.TestCase):
    def test_bronze_immutability_and_exact_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            manifest_bytes = manifest_path.read_bytes()
            output = root / "phase2k"
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="no_provider",
            )
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
            records = load_json_strict(
                result["paths"]["records"], label="records",
            )
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            for record in records["records"]:
                target = record["target"]
                row = connection.execute(
                    "SELECT transcription FROM videos WHERE video_id = ?",
                    (target["source_group_id"].removeprefix("video:"),),
                ).fetchone()
                full = row[0]
                self.assertEqual(
                    full[target["source_absolute_start"]:target["source_absolute_end"]],
                    target["text"],
                )
            connection.close()
            self.assertEqual(result["window_count"], 30)

    def test_source_ordering_and_no_cross_video_retrieval(self):
        text_a = "One sentence here. Another sentence here. Target sentence. Tail sentence."
        text_b = "Unrelated video sentence. Another unrelated sentence."
        target = "Target sentence."
        start = text_a.index(target)
        end = start + len(target)
        segments_a = retrieve_context(
            text_a,
            source_group_id="video:aaa",
            window_id="w:aaa",
            target_start=start,
            target_end=end,
            bronze_text=target,
            previous_segments=5,
            following_segments=5,
            radius_label="r10",
        )
        for segment in segments_a["segments"]:
            if segment["kind"] == "target":
                self.assertTrue(segment["segment_id"].startswith("target:"))
            else:
                self.assertTrue(segment["segment_id"].startswith("seg:video:aaa:"))
        validate_context(segments_a, text_a)
        self.assertTrue(all(
            text_b.find(segment["text"]) == -1 or not segment["text"].strip()
            for segment in segments_a["segments"]
        ))
        offsets = [
            segment["source_absolute_start"] for segment in segments_a["segments"]
        ]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(segments_a["segments"][-1]["source_absolute_end"], len(text_a))
        self.assertEqual(segments_a["previous_stop_reason"], "SOURCE_BOUNDARY")
        self.assertEqual(segments_a["following_stop_reason"], "SOURCE_BOUNDARY")

    def test_target_only_backward_forward_both_and_hard_cap(self):
        text = (
            "s1 one. s1 two. s1 three. s1 four. s1 five. s1 six. "
            "TARGET here. s1 seven. s1 eight. s1 nine. s1 ten."
        )
        target = "TARGET here."
        start = text.index(target)
        end = start + len(target)

        target_only = retrieve_context(
            text, source_group_id="video:v", window_id="w",
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=0, following_segments=0, radius_label="target_only",
        )
        self.assertEqual([item["kind"] for item in target_only["segments"]], ["target"])
        self.assertEqual(target_only["stop_reason"], "TARGET_ONLY")
        self.assertEqual(target_only["actual"]["total_tokens"], 2)

        backward = retrieve_context(
            text, source_group_id="video:v", window_id="w",
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=3, following_segments=0, radius_label="r5",
        )
        self.assertEqual(backward["actual"]["previous_segments"], 3)
        self.assertEqual(backward["actual"]["following_segments"], 0)
        self.assertTrue(all(
            item["kind"] in {"previous", "target", "target_context"}
            for item in backward["segments"]
        ))
        self.assertFalse(any(
            item["kind"] == "following" for item in backward["segments"]
        ))

        forward = retrieve_context(
            text, source_group_id="video:v", window_id="w",
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=0, following_segments=3, radius_label="r5",
        )
        self.assertEqual(forward["actual"]["previous_segments"], 0)
        self.assertEqual(forward["actual"]["following_segments"], 3)
        self.assertFalse(any(
            item["kind"] == "previous" for item in forward["segments"]
        ))

        both = retrieve_context(
            text, source_group_id="video:v", window_id="w",
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=3, following_segments=4, radius_label="r10",
        )
        self.assertEqual(both["actual"]["previous_segments"], 3)
        self.assertEqual(both["actual"]["following_segments"], 4)
        validate_context(both, text)

        capped = retrieve_context(
            text, source_group_id="video:v", window_id="w",
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=50, following_segments=50, radius_label="r10",
            segment_cap=2,
        )
        self.assertEqual(capped["actual"]["previous_segments"], 2)
        self.assertEqual(capped["actual"]["following_segments"], 2)
        self.assertEqual(capped["previous_stop_reason"], "HARD_SEGMENT_CAP")
        self.assertEqual(capped["following_stop_reason"], "HARD_SEGMENT_CAP")

    def test_metadata_conflicts_stay_explicit(self):
        text = "He hit R. Lux is on your team. Viktor is the enemy."
        target = "He hit R."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text, source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=2, following_segments=2, radius_label="r5",
        )
        conflicts = [{
            "field": "champion_identities",
            "metadata_value": "Viktor",
            "context_evidence": "Lux is on your team. Viktor is the enemy.",
            "note": "metadata champion conflicts with the target pronoun evidence",
        }]
        raw = _sufficiency_raw(
            "NEED_MORE_FOLLOWING_CONTEXT",
            champion="Viktor",
            unresolved="champion_identities",
            conflicts=conflicts,
        )
        parsed = normalize_sufficiency_compact_response(
            json.loads(raw),
            transcript=text,
            context=context,
            at_max_context=False,
        )
        self.assertEqual(parsed["metadata_conflicts"], conflicts)

    def test_repair_span_provenance_and_deterministic_application(self):
        selected = _selected("he used W to escape. wed on the wave.")
        base = selected["upstream_start"]
        repairs = [
            {
                "repair_id": "r1",
                "target_local_start": 0,
                "target_local_end": 2,
                "original_text": "he",
                "replacement": "He",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "evidence_spans": [
                    {"target_local_start": 0, "target_local_end": 2, "text": "he"},
                ],
                "rationale": "sentence start",
            }
        ]
        cleaned = apply_mechanical_repairs(selected["source_text"], repairs)
        self.assertEqual(cleaned, "He used W to escape. wed on the wave.")
        chat = CountingChat([_mechanical_raw(selected, repairs=repairs)])
        result = run_mechanical_cleanup(
            selected, chat=chat, config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["repair_count"], 1)
        self.assertEqual(result["repairs"][0]["source_absolute_start"], base + 0)
        self.assertEqual(result["repairs"][0]["source_absolute_end"], base + 2)
        self.assertEqual(
            result["mechanical_cleaned_text"],
            apply_mechanical_repairs(
                selected["source_text"], result["repairs"],
            ),
        )
        # Zero edits are allowed and must round-trip.
        zero = run_mechanical_cleanup(
            selected,
            chat=CountingChat([_mechanical_raw(selected)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(zero["mechanical_cleaned_text"], selected["source_text"])
        # Overlapping repairs are rejected.
        overlap = [
            {
                "repair_id": "a",
                "target_local_start": 0,
                "target_local_end": 4,
                "original_text": "he u",
                "replacement": "xx",
                "repair_type": "SPELLING",
                "confidence": "HIGH",
                "evidence_spans": [],
                "rationale": "x",
            },
            {
                "repair_id": "b",
                "target_local_start": 2,
                "target_local_end": 5,
                "original_text": " us",
                "replacement": "yy",
                "repair_type": "SPELLING",
                "confidence": "HIGH",
                "evidence_spans": [],
                "rationale": "y",
            },
        ]
        with self.assertRaises(ValueError):
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    _mechanical_raw(
                    selected,
                    repairs=overlap,
                    clean_text="xxyyed W to escape. wed on the wave.",
                    ),
                    _mechanical_raw(
                    selected,
                    repairs=overlap,
                    clean_text="xxyyed W to escape. wed on the wave.",
                    ),
                    _mechanical_raw(
                    selected,
                    repairs=overlap,
                    clean_text="xxyyed W to escape. wed on the wave.",
                    ),
                    _mechanical_raw(
                    selected,
                    repairs=overlap,
                    clean_text="xxyyed W to escape. wed on the wave.",
                    ),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        # Ownership/entity resolution repair types are never mechanical.
        forbidden = dict(repairs[0])
        forbidden["repair_type"] = "ABILITY_OWNERSHIP_RESOLUTION"
        with self.assertRaises(ValueError):
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    _mechanical_raw(selected, repairs=[forbidden]),
                    _mechanical_raw(selected, repairs=[forbidden]),
                    _mechanical_raw(selected, repairs=[forbidden]),
                    _mechanical_raw(selected, repairs=[forbidden]),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )

    def test_hs_remains_unresolved_and_wed_contract_allows_w_d(self):
        text = "here you are dead and have to flash if HS one more"
        selected = _selected(text, champion="Viktor")
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([_mechanical_raw(selected)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["mechanical_cleaned_text"], text)
        self.assertEqual(result["repairs"], [])

        # A high-confidence mechanical ASR/domain repair "wed" -> "W'd" is
        # allowed because the local phrase + domain metadata disambiguate the
        # intended form without resolving who owns the ability.
        wed_text = "maybe he wed on the wave"
        wed = _selected(wed_text, champion="Viktor")
        start = wed_text.index("wed")
        repairs = [{
            "repair_id": "p2k:test:wed",
            "target_local_start": start,
            "target_local_end": start + 3,
            "original_text": "wed",
            "replacement": "W'd",
            "repair_type": "ASR_HOMOPHONE",
            "confidence": "HIGH",
            "evidence_spans": [
                {
                    "target_local_start": start + 4,
                    "target_local_end": len(wed_text),
                    "text": "on the wave",
                },
            ],
            "rationale": "domain ability key plus reduced auxiliary",
        }]
        wed_result = run_mechanical_cleanup(
            wed,
            chat=CountingChat([_mechanical_raw(wed, repairs=repairs)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            wed_result["mechanical_cleaned_text"],
            "maybe he W'd on the wave",
        )
        self.assertEqual(wed_result["repair_count"], 1)
        self.assertEqual(wed_result["repairs"][0]["repair_type"], "ASR_HOMOPHONE")
        # The owner of W remains unresolved: ownership/entity/pronoun repair
        # types are still forbidden mechanically.
        for forbidden_type in (
            "ENTITY_RESOLUTION",
            "PRONOUN_RESOLUTION",
            "ABILITY_OWNERSHIP_RESOLUTION",
            "REFERENT_RESOLUTION",
        ):
            forbidden = dict(repairs[0])
            forbidden["repair_type"] = forbidden_type
            with self.assertRaises(ValueError):
                run_mechanical_cleanup(
                    wed,
                    chat=CountingChat([
                        _mechanical_raw(wed, repairs=[forbidden]),
                        _mechanical_raw(wed, repairs=[forbidden]),
                        _mechanical_raw(wed, repairs=[forbidden]),
                        _mechanical_raw(wed, repairs=[forbidden]),
                    ]),
                    config_hash=canonical_sha256({"v": 1}),
                )
        # The prompt encodes the adversarial contract: HS stays unresolved
        # when ambiguous while wed -> W'd is permitted.
        system, _ = build_mechanical_prompt(wed)
        self.assertIn("wed on the wave", system)
        self.assertIn("W'd", system)
        self.assertIn("HS", system)
        self.assertIn("must remain unchanged", system)

    def test_ability_letter_ordinary_word_is_not_forced(self):
        text = "The letter W is silent in this word. He said it twice."
        selected = _selected(text, champion="Lux")
        mechanical = run_mechanical_cleanup(
            selected,
            chat=CountingChat([_mechanical_raw(selected)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(mechanical["mechanical_cleaned_text"], text)
        context = retrieve_context(
            text, source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=0, target_end=len(text), bronze_text=text,
            previous_segments=1, following_segments=1, radius_label="r2",
        )
        raw = _sufficiency_raw("SUFFICIENT", champion="Lux")
        parsed = normalize_sufficiency_compact_response(
            json.loads(raw),
            transcript=text,
            context=context,
            at_max_context=False,
        )
        self.assertEqual(parsed["slots"]["ability_ownership"]["status"], "RESOLVED")

    def test_adaptive_loop_backward_forward_both_and_max(self):
        text = (
            "Earlier Lux used her ultimate. Earlier Lux moved to mid. "
            "He hit R. Later Viktor died. Later the wave reset."
        )
        target = "He hit R."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Viktor", target=target)

        backward = run_adaptive_diagnostics(
            selected,
            transcript=text,
            mechanical_cleaned_text=text,
            chat=CountingChat([
                _sufficiency_raw("NEED_MORE_PREVIOUS_CONTEXT", unresolved="pronouns"),
                _sufficiency_raw("SUFFICIENT"),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual([a["stage"] for a in backward[0]], ["r1", "r2"])
        self.assertEqual(backward[1]["decision"], "SUFFICIENT")
        self.assertEqual(
            backward[0][1]["context"]["requested"],
            {"previous_segments": 5, "following_segments": 2},
        )

        forward = run_adaptive_diagnostics(
            selected,
            transcript=text,
            mechanical_cleaned_text=text,
            chat=CountingChat([
                _sufficiency_raw("NEED_MORE_FOLLOWING_CONTEXT", unresolved="pronouns"),
                _sufficiency_raw("SUFFICIENT"),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            forward[0][1]["context"]["requested"],
            {"previous_segments": 2, "following_segments": 5},
        )

        both = run_adaptive_diagnostics(
            selected,
            transcript=text,
            mechanical_cleaned_text=text,
            chat=CountingChat([
                _sufficiency_raw("NEED_BOTH", unresolved="pronouns"),
                _sufficiency_raw("SUFFICIENT"),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            both[0][1]["context"]["requested"],
            {"previous_segments": 5, "following_segments": 5},
        )

        maximum = run_adaptive_diagnostics(
            selected,
            transcript=text,
            mechanical_cleaned_text=text,
            chat=CountingChat([
                _sufficiency_raw("NEED_MORE_FOLLOWING_CONTEXT", unresolved="pronouns"),
                _sufficiency_raw("NEED_MORE_PREVIOUS_CONTEXT", unresolved="pronouns"),
                _sufficiency_raw("NEED_BOTH", unresolved="pronouns"),
                _sufficiency_raw(
                    "MAX_CONTEXT_BUT_UNRESOLVED", unresolved="pronouns",
                ),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            [a["stage"] for a in maximum[0]],
            ["r1", "r2", "r3", "r4_bounded_local_episode"],
        )
        self.assertEqual(maximum[1]["decision"], "MAX_CONTEXT_BUT_UNRESOLVED")
        self.assertTrue(maximum[1]["at_max_context"])

    def test_later_context_resolves(self):
        text = (
            "Two champions fought. Ahri used W first. "
            "He used W again. Viktor also used W. "
            "The replay shows Viktor's W hit the wave."
        )
        target = "He used W again."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Viktor", target=target)
        attempts, final = run_adaptive_diagnostics(
            selected,
            transcript=text,
            mechanical_cleaned_text=text,
            chat=CountingChat([
                _sufficiency_raw(
                    "NEED_MORE_FOLLOWING_CONTEXT",
                    unresolved="ability_ownership",
                ),
                _sufficiency_raw("SUFFICIENT"),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(len(attempts), 2)
        self.assertEqual(final["decision"], "SUFFICIENT")

    def test_he_two_candidates_and_binding_provenance(self):
        text = "Lux and Ahri fought. He hit R. Viktor watched from base."
        target = "He hit R."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text, source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=1, following_segments=2, radius_label="r5",
        )
        target_segment_id = next(
            item["segment_id"] for item in context["segments"]
            if item["kind"] == "target"
        )
        raw = _sufficiency_raw(
            "NEED_MORE_FOLLOWING_CONTEXT",
            champion="Viktor",
            unresolved="pronouns",
        )
        parsed = normalize_sufficiency_compact_response(
            json.loads(raw),
            transcript=text,
            context=context,
            at_max_context=False,
        )
        self.assertEqual(parsed["slots"]["pronouns"]["status"], "UNKNOWN")

        # Reconstruction with an explicit binding and exact evidence.
        final_slots = _slots(decision="SUFFICIENT", champion="Viktor")
        final_slots["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "Viktor",
                "confidence": "HIGH",
                "evidence_spans": [{
                    "segment_id": target_segment_id,
                    "source_absolute_start": start + 17,
                    "source_absolute_end": start + 24,
                    "text": "watched",
                }],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": final_slots,
                    "metadata_conflicts": [],
                },
            },
        }
        cleaned = text
        binding = {
            "slot": "pronouns",
            "mention_text": "He",
            "resolved_candidate": "Viktor",
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_quotes": ["watched"],
            "alternatives": [],
            "metadata_contributed": True,
            "rationale": "later context resolves to Viktor",
        }
        raw_rec = _reconstruction_raw(
            cleaned=target,
            bronze=target,
            base_offset=start,
            selected=selected,
            bindings_override=[binding],
        )
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=cleaned,
            final_diagnostic=diagnostic,
            chat=CountingChat([raw_rec]),
            config_hash=canonical_sha256({"v": 1}),
        )
        binding = reconstruction["bindings"][0]
        self.assertEqual(binding["mention"]["text"], "He")
        self.assertEqual(binding["resolved_candidate"], "Viktor")
        self.assertEqual(binding["evidence_spans"][0]["text"], "watched")
        self.assertTrue(binding["metadata_contributed"])

    def test_multiple_candidates_and_ambiguous_statuses(self):
        text = "Lux or Ahri could have used W. He hit R. Later Ahri died."
        target = "He hit R."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Ahri", target=target)
        context = retrieve_context(
            text, source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=1, following_segments=2, radius_label="r10",
        )
        slots = _slots(
            decision="NEED_MORE_FOLLOWING_CONTEXT",
            unresolved="pronouns",
        )
        slots["pronouns"] = {
            "status": "MULTIPLE_CANDIDATES",
            "candidates": [
                {
                    "candidate": "Lux",
                    "confidence": "MEDIUM",
                    "evidence_spans": [{
                        "segment_id": next(
                            item["segment_id"] for item in context["segments"]
                            if item["kind"] == "previous"
                        ),
                        "source_absolute_start": 0,
                        "source_absolute_end": 8,
                        "text": "Lux or A",
                    }],
                },
                {
                    "candidate": "Ahri",
                    "confidence": "MEDIUM",
                    "evidence_spans": [],
                },
            ],
            "confidence": "MEDIUM",
            "evidence_spans": [],
        }
        raw = _compact_sufficiency_raw(
            "NEED_MORE_FOLLOWING_CONTEXT",
            slots=slots,
            rationale="he has two candidates: Lux and Ahri",
        )
        parsed = normalize_sufficiency_compact_response(
            json.loads(raw),
            transcript=text,
            context=context,
            at_max_context=False,
        )
        self.assertEqual(
            parsed["slots"]["pronouns"]["status"],
            "MULTIPLE_CANDIDATES",
        )
        self.assertEqual(
            [c["candidate"] for c in parsed["slots"]["pronouns"]["candidates"]],
            ["Lux", "Ahri"],
        )

        final_slots = _slots(
            decision="MAX_CONTEXT_BUT_UNRESOLVED",
            unresolved="pronouns",
        )
        final_slots["pronouns"] = {
            "status": "AMBIGUOUS",
            "candidates": [{
                "candidate": "Lux",
                "confidence": "LOW",
                "evidence_spans": [],
            }],
            "confidence": "LOW",
            "evidence_spans": [],
        }
        diagnostic = {
            "decision": "MAX_CONTEXT_BUT_UNRESOLVED",
            "response": {
                "parsed": {
                    "slots": final_slots,
                    "metadata_conflicts": [],
                },
            },
        }
        binding = {
            "slot": "pronouns",
            "mention_text": "He",
            "resolved_candidate": "AMBIGUOUS",
            "resolved_status": "AMBIGUOUS",
            "confidence": "LOW",
            "evidence_quotes": [],
            "alternatives": [{
                "candidate": "Lux",
                "evidence_quotes": ["Lux"],
                "note": "candidate remains ambiguous",
            }],
            "metadata_contributed": False,
            "rationale": "ambiguous without resolving evidence",
        }
        raw_rec = _reconstruction_raw(
            cleaned=target,
            bronze=target,
            base_offset=start,
            selected=selected,
            bindings_override=[binding],
        )
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=diagnostic,
            chat=CountingChat([raw_rec]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["bindings"][0]["resolved_status"],
            "AMBIGUOUS",
        )

    def test_unresolved_statuses_propagate_to_bindings(self):
        text = "He hit R. No other context is available."
        target = "He hit R."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Lux", target=target)
        context = retrieve_context(
            text, source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start, target_end=end, bronze_text=target,
            previous_segments=10, following_segments=10, radius_label="r10",
        )
        final_slots = _slots(
            decision="MAX_CONTEXT_BUT_UNRESOLVED", unresolved="pronouns",
        )
        final_slots["pronouns"] = {
            "status": "CONTEXT_INSUFFICIENT",
            "candidates": [],
            "confidence": "LOW",
            "evidence_spans": [],
        }
        diagnostic = {
            "decision": "MAX_CONTEXT_BUT_UNRESOLVED",
            "response": {
                "parsed": {
                    "slots": final_slots,
                    "metadata_conflicts": [],
                },
            },
        }
        binding = {
            "slot": "pronouns",
            "mention_text": "He",
            "resolved_candidate": "CONTEXT_INSUFFICIENT",
            "resolved_status": "CONTEXT_INSUFFICIENT",
            "confidence": "LOW",
            "evidence_quotes": [],
            "alternatives": [],
            "metadata_contributed": False,
            "rationale": "no resolving evidence exists",
        }
        raw_rec = _reconstruction_raw(
            cleaned=target,
            bronze=target,
            base_offset=start,
            selected=selected,
            bindings_override=[binding],
        )
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=diagnostic,
            chat=CountingChat([raw_rec]),
            config_hash=canonical_sha256({"v": 1}),
        )
        binding = reconstruction["bindings"][0]
        self.assertEqual(binding["resolved_status"], "CONTEXT_INSUFFICIENT")
        polish = run_polish(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            reconstruction=reconstruction,
            chat=CountingChat([_polish_raw(
                selected,
                reconstruction,
                unsupported=[{
                    "claim": "He is Lux",
                    "reason": "AMBIGUOUS_OWNERSHIP",
                    "note": "not licensed by evidence",
                }],
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(polish["counts"]["unsupported_claim_count"], 1)
        self.assertGreater(polish["counts"]["unsupported_rate"], 0.0)

    def test_forbidden_abstractions_are_rejected(self):
        selected = _selected("He hit R.", champion="Lux")
        context = retrieve_context(
            "He hit R. More context.",
            source_group_id="video:testvid",
            window_id="w",
            target_start=0,
            target_end=len("He hit R."),
            bronze_text="He hit R.",
            previous_segments=1,
            following_segments=1,
            radius_label="r2",
        )
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": _slots(decision="SUFFICIENT"),
                    "metadata_conflicts": [],
                },
            },
        }
        target_segment_id = next(
            item["segment_id"] for item in context["segments"]
            if item["kind"] == "target"
        )
        repair = {
            "original_text": selected["source_text"],
            "replacement": "the conversion of the wave is ideal",
            "repair_type": "DOMAIN_SPELLING",
            "confidence": "HIGH",
            "evidence_quotes": [selected["source_text"]],
            "rationale": "test",
        }
        raw_rec = _reconstruction_raw(
            cleaned="the conversion of the wave is ideal",
            bronze=selected["source_text"],
            base_offset=0,
            selected=selected,
            contextual_repairs=[repair],
        )
        with self.assertRaises(ValueError):
            run_reconstruction(
                selected,
                transcript="He hit R. More context.",
                context=context,
                mechanical_cleaned_text="He hit R.",
                final_diagnostic=diagnostic,
                chat=CountingChat([raw_rec, raw_rec, raw_rec, raw_rec]),
                config_hash=canonical_sha256({"v": 1}),
            )

    def test_unlicensed_entities_are_rejected(self):
        selected = _selected("He hit R.", champion="Lux")
        context = retrieve_context(
            "He hit R.",
            source_group_id="video:testvid",
            window_id="w",
            target_start=0,
            target_end=len("He hit R."),
            bronze_text="He hit R.",
            previous_segments=0,
            following_segments=0,
            radius_label="target_only",
        )
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": _slots(decision="SUFFICIENT"),
                    "metadata_conflicts": [],
                },
            },
        }
        target_segment_id = next(
            item["segment_id"] for item in context["segments"]
            if item["kind"] == "target"
        )
        repair = {
            "original_text": selected["source_text"],
            "replacement": "Zed appears and steals the kill",
            "repair_type": "DOMAIN_SPELLING",
            "confidence": "HIGH",
            "evidence_quotes": [selected["source_text"]],
            "rationale": "test",
        }
        raw_rec = _reconstruction_raw(
            cleaned="Zed appears and steals the kill",
            bronze=selected["source_text"],
            base_offset=0,
            selected=selected,
            contextual_repairs=[repair],
        )
        with self.assertRaises(ValueError):
            run_reconstruction(
                selected,
                transcript="He hit R.",
                context=context,
                mechanical_cleaned_text="He hit R.",
                final_diagnostic=diagnostic,
                chat=CountingChat([raw_rec, raw_rec, raw_rec, raw_rec]),
                config_hash=canonical_sha256({"v": 1}),
            )

    def test_stable_canonical_hashes_and_prompt_config_recording(self):
        first = canonical_sha256({"b": 1, "a": [2, 3], "c": "x"})
        second = canonical_sha256({"c": "x", "a": [2, 3], "b": 1})
        self.assertEqual(first, second)
        selected = _selected("He hit R.", champion="Lux")
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([_mechanical_raw(selected)]),
            config_hash=canonical_sha256({"cfg": 1}),
            inference_config=TEST_LIVE_INFERENCE_CONFIG,
        )
        self.assertIn("prompt_hash", result["model_call"])
        self.assertIn("prompt_version", result["model_call"])
        self.assertIn("raw_response_sha256", result["model_call"])
        self.assertEqual(
            result["model_call"]["inference_config"],
            TEST_LIVE_INFERENCE_CONFIG,
        )
        self.assertEqual(
            result["model_call"]["inference_config_hash"],
            inference_config_hash(TEST_LIVE_INFERENCE_CONFIG),
        )
        self.assertEqual(
            result["model_call"]["inference_config_version"],
            INFERENCE_CONFIG_VERSION,
        )
        self.assertIn("pipeline_version", result)

    def test_response_cache_reproducibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            calls: list[str] = []

            def chat(system: str, user: str) -> str:
                calls.append(user)
                return '{"schema_version": "x", "ok": true}'

            first = call_provider(
                chat,
                system="s",
                user="u1",
                schema_version="phase2k-test-v1",
                prompt_version="p1",
                config_hash=canonical_sha256({"c": 1}),
                label="cache test",
                cache_dir=cache_dir,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            second = call_provider(
                chat,
                system="s",
                user="u1",
                schema_version="phase2k-test-v1",
                prompt_version="p1",
                config_hash=canonical_sha256({"c": 1}),
                label="cache test",
                cache_dir=cache_dir,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            self.assertEqual(first["source"], "provider")
            self.assertEqual(second["source"], "cache")
            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                first["inference_config_hash"],
                inference_config_hash(TEST_LIVE_INFERENCE_CONFIG),
            )
            self.assertEqual(first["inference_config"], TEST_LIVE_INFERENCE_CONFIG)
            third = call_provider(
                chat,
                system="s",
                user="u2",
                schema_version="phase2k-test-v1",
                prompt_version="p1",
                config_hash=canonical_sha256({"c": 1}),
                label="cache test",
                cache_dir=cache_dir,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            self.assertEqual(third["source"], "provider")
            self.assertNotEqual(third["cache_key"], first["cache_key"])
            self.assertEqual(len(calls), 2)
            # Changing any inference-config component must invalidate the
            # cache even when the prompt is identical.
            variants = (
                {"temperature": 0.5},
                {"model": "other-model"},
                {"thinking": "enabled"},
                {"max_tokens": 2048},
            )
            seen_keys = {first["cache_key"]}
            for variant in variants:
                changed = dict(TEST_LIVE_INFERENCE_CONFIG)
                changed.update(variant)
                result = call_provider(
                    chat,
                    system="s",
                    user="u1",
                    schema_version="phase2k-test-v1",
                    prompt_version="p1",
                    config_hash=canonical_sha256({"c": 1}),
                    label="cache test",
                    cache_dir=cache_dir,
                    inference_config=changed,
                )
                self.assertEqual(result["source"], "provider")
                self.assertNotIn(result["cache_key"], seen_keys)
                seen_keys.add(result["cache_key"])
            self.assertEqual(len(calls), 1 + 1 + len(variants))
            # Bumping the response schema or prompt version must also
            # invalidate otherwise-identical cached responses.
            bumped_schema = call_provider(
                chat,
                system="s",
                user="u1",
                schema_version="phase2k-test-v2",
                prompt_version="p1",
                config_hash=canonical_sha256({"c": 1}),
                label="cache test",
                cache_dir=cache_dir,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            bumped_prompt = call_provider(
                chat,
                system="s",
                user="u1",
                schema_version="phase2k-test-v1",
                prompt_version="p2",
                config_hash=canonical_sha256({"c": 1}),
                label="cache test",
                cache_dir=cache_dir,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            self.assertEqual(bumped_schema["source"], "provider")
            self.assertEqual(bumped_prompt["source"], "provider")
            self.assertNotEqual(bumped_schema["cache_key"], first["cache_key"])
            self.assertNotEqual(bumped_prompt["cache_key"], first["cache_key"])
            self.assertEqual(len(calls), 1 + 1 + len(variants) + 2)
            entry = load_json_strict(
                cache_dir / f"{first['cache_key']}.json",
                label="cache entry",
            )
            self.assertEqual(entry["inference_config_hash"], first["inference_config_hash"])
            self.assertEqual(entry["inference_config"], TEST_LIVE_INFERENCE_CONFIG)

    def test_inference_config_sealing_and_secret_rejection(self):
        # Required keys are enforced fail-closed.
        with self.assertRaises(ValueError):
            validate_inference_config({"provider": "x"}, label="t")
        with self.assertRaises(ValueError):
            validate_inference_config(
                {
                    "provider": "x",
                    "model": "m",
                    "endpoint": None,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "thinking": "maybe",
                    "purpose": "p",
                },
                label="t",
            )
        # Secret-bearing fields are rejected at any nesting depth.
        with self.assertRaises(ValueError):
            validate_inference_config(
                {**TEST_LIVE_INFERENCE_CONFIG, "api_key": "sk-test"},
                label="t",
            )
        with self.assertRaises(ValueError):
            validate_inference_config(
                {
                    **TEST_LIVE_INFERENCE_CONFIG,
                    "credentials": {"password": "hunter2"},
                },
                label="t",
            )
        # The hash is order-independent and secret-free snapshots round-trip.
        self.assertEqual(
            inference_config_hash(TEST_LIVE_INFERENCE_CONFIG),
            inference_config_hash(dict(reversed(list(
                TEST_LIVE_INFERENCE_CONFIG.items(),
            )))),
        )
        self.assertEqual(
            validate_inference_config(
                TEST_LIVE_INFERENCE_CONFIG, label="t",
            ),
            TEST_LIVE_INFERENCE_CONFIG,
        )
        # The explicit no-provider snapshot seals provider "none".
        self.assertEqual(NO_PROVIDER_INFERENCE_CONFIG["provider"], "none")
        self.assertEqual(
            inference_config_hash(NO_PROVIDER_INFERENCE_CONFIG),
            canonical_sha256(NO_PROVIDER_INFERENCE_CONFIG),
        )

    def test_build_seals_inference_config_and_live_requires_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            with self.assertRaises(ValueError):
                build_phase2k_outputs(
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                    doc_path=None,
                    output_dir=root / "phase2k-live",
                    mode="live",
                    chat=_live_factory_chat(),
                )  # live mode requires a sealed inference config
            with self.assertRaises(ValueError):
                build_phase2k_outputs(
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                    doc_path=None,
                    output_dir=root / "phase2k-bad-no-provider",
                    mode="no_provider",
                    inference_config=TEST_LIVE_INFERENCE_CONFIG,
                )  # no-provider must seal the explicit no-provider snapshot
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=root / "phase2k",
                mode="no_provider",
            )
            records = load_json_strict(
                result["paths"]["records"], label="records",
            )
            summary = load_json_strict(
                result["paths"]["build_summary"], label="build summary",
            )
            self.assertEqual(
                records["inference_config"],
                NO_PROVIDER_INFERENCE_CONFIG,
            )
            self.assertEqual(
                records["inference_config_hash"],
                inference_config_hash(NO_PROVIDER_INFERENCE_CONFIG),
            )
            self.assertEqual(
                records["inference_config_version"],
                INFERENCE_CONFIG_VERSION,
            )
            self.assertEqual(
                summary["inference_config_hash"],
                records["inference_config_hash"],
            )
            self.assertEqual(
                summary["inference_config"],
                records["inference_config"],
            )
            self.assertEqual(summary["config_hash"], records["config_hash"])

    def test_live_validation_rejects_inference_config_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "phase2k-live"
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="live",
                chat=_live_factory_chat(),
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            validated = validate_output_directory(
                output_dir=output,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )
            self.assertEqual(
                validated["inference_config_hash"],
                inference_config_hash(TEST_LIVE_INFERENCE_CONFIG),
            )

            records_path = result["paths"]["records"]
            summary_path = result["paths"]["build_summary"]
            pristine_records = load_json_strict(records_path, label="records")
            pristine_summary = load_json_strict(summary_path, label="summary")
            attempt_path = sorted((output / "attempts").rglob("r1.json"))[0]

            def rewrite(path: Path, value: Mapping[str, Any]) -> None:
                path.write_text(
                    json.dumps(value, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )

            def revalidate() -> None:
                validate_output_directory(
                    output_dir=output,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                )

            # A tampered model-call snapshot/hash inside records is rejected.
            records = load_json_strict(
                records_path, label="records",
            )
            tampered_record = None
            for record in records["records"]:
                if record["record_type"] == "B":
                    record["content"]["model_call"]["inference_config_hash"] = (
                        "0" * 64
                    )
                    tampered_record = record
                    break
            self.assertIsNotNone(tampered_record)
            records = _envelope({
                key: value for key, value in records.items()
                if key != "content_sha256"
            })
            rewrite(records_path, records)
            with self.assertRaises(ValueError):
                revalidate()
            rewrite(records_path, pristine_records)

            # A tampered attempt model-call hash is rejected.
            attempt = load_json_strict(attempt_path, label="attempt")
            pristine_attempt = json.loads(json.dumps(attempt))
            attempt["model_call"]["inference_config_hash"] = "0" * 64
            rewrite(attempt_path, attempt)
            with self.assertRaises(ValueError):
                revalidate()
            rewrite(attempt_path, pristine_attempt)

            # A tampered top-level records snapshot is rejected.
            records = load_json_strict(
                records_path, label="records",
            )
            records["inference_config"] = {
                **records["inference_config"],
                "temperature": 0.5,
            }
            records = _envelope({
                key: value for key, value in records.items()
                if key != "content_sha256"
            })
            rewrite(records_path, records)
            with self.assertRaises(ValueError):
                revalidate()
            rewrite(records_path, pristine_records)

            # A build summary that disagrees with records is rejected.
            summary = load_json_strict(
                summary_path, label="build summary",
            )
            summary["inference_config_hash"] = "1" * 64
            summary = _envelope({
                key: value for key, value in summary.items()
                if key != "content_sha256"
            })
            rewrite(summary_path, summary)
            with self.assertRaises(ValueError):
                revalidate()

            # Restoring the artifacts makes validation pass again.
            rewrite(summary_path, pristine_summary)
            revalidate()

    def test_provider_json_parsing_fail_closed(self):
        self.assertEqual(parse_provider_json('{"a": [1, 2]}', label="t"), {"a": [1, 2]})
        self.assertEqual(
            parse_provider_json('```json\n{"a": 1}\n```', label="t"),
            {"a": 1},
        )
        for bad in (
            '{"a": 1, "a": 2}',
            '{"a": NaN}',
            '{"a": Infinity}',
            '{"a": 1.5}',
            '[1, 2]',
            'text before\n{"a": 1}\ntext after',
        ):
            with self.assertRaises(ValueError):
                parse_provider_json(bad, label="t")

    def test_mechanical_deterministic_binding_seals_real_wrong_offset_sample(self):
        # Regression fixture based on the real 30-window live run: the model
        # emitted needless uncertainties with wrong character offsets for a
        # window whose clean_text was already identical to Bronze.  The
        # compact schema carries only exact quoted surfaces, and the harness
        # deterministically binds/seals exact slices, absolute offsets, and
        # full provenance while preserving the raw response verbatim.
        bronze = (
            ">> okay, >> just press B. And if somehow now Diana makes them "
            "10 health, then of course press ult and cancel your base. But "
            "don't do this because now you're jailed forever again because "
            "now if you stay this wave then obviously they can look to dive"
        )
        selected = _selected(bronze)
        raw_response = _compact_mechanical_raw(
            selected,
            clean_text=bronze,
            uncertainties=[
                {
                    "surface_text": ">>",
                    "uncertainty_type": "PUNCTUATION_UNCERTAIN",
                    "alternatives": [
                        {"text": ">>", "confidence": "HIGH"},
                        {"text": ">", "confidence": "LOW"},
                    ],
                    "note": "speaker or turn marker punctuation",
                },
                {
                    "surface_text": ">>",
                    "uncertainty_type": "PUNCTUATION_UNCERTAIN",
                    "alternatives": [
                        {"text": ">>", "confidence": "HIGH"},
                        {"text": ">", "confidence": "LOW"},
                    ],
                    "note": "same marker later in the target",
                },
                {
                    "surface_text": "Diana",
                    "uncertainty_type": "ASR_ALTERNATIVES",
                    "alternatives": [
                        {"text": "Diana", "confidence": "HIGH"},
                        {"text": "Dianna", "confidence": "LOW"},
                    ],
                    "note": "champion spelling surface",
                },
                {
                    "surface_text": "health",
                    "uncertainty_type": "ASR_ALTERNATIVES",
                    "alternatives": [
                        {"text": "health", "confidence": "HIGH"},
                        {"text": "helth", "confidence": "LOW"},
                    ],
                    "note": "clear standard word",
                },
                {
                    "surface_text": "ult",
                    "uncertainty_type": "DOMAIN_TOKEN_UNCERTAIN",
                    "alternatives": [
                        {"text": "ult", "confidence": "HIGH"},
                        {"text": "R", "confidence": "LOW"},
                    ],
                    "note": "ultimate shorthand surface",
                },
                {
                    "surface_text": "base",
                    "uncertainty_type": "DOMAIN_TOKEN_UNCERTAIN",
                    "alternatives": [
                        {"text": "base", "confidence": "HIGH"},
                        {"text": "bass", "confidence": "LOW"},
                    ],
                    "note": "recall-to-base surface",
                },
                {
                    "surface_text": "jailed",
                    "uncertainty_type": "ASR_ALTERNATIVES",
                    "alternatives": [
                        {"text": "jailed", "confidence": "HIGH"},
                        {"text": "jail", "confidence": "LOW"},
                    ],
                    "note": "clear in context",
                },
                {
                    "surface_text": "wave",
                    "uncertainty_type": "DOMAIN_TOKEN_UNCERTAIN",
                    "alternatives": [
                        {"text": "wave", "confidence": "HIGH"},
                        {"text": "way", "confidence": "LOW"},
                    ],
                    "note": "minion wave surface",
                },
            ],
            rationale=(
                "No repairs; the original live sample carried wrong "
                "offsets and manufactured uncertainties for clear words."
            ),
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([raw_response]),
            config_hash=canonical_sha256({"v": 1}),
        )
        base = selected["upstream_start"]
        self.assertEqual(result["mechanical_cleaned_text"], bronze)
        self.assertEqual(result["repairs"], [])
        self.assertEqual(len(result["uncertainties"]), 8)
        for uncertainty in result["uncertainties"]:
            local_start = uncertainty["target_local_start"]
            local_end = uncertainty["target_local_end"]
            self.assertEqual(
                uncertainty["text"],
                bronze[local_start:local_end],
            )
            self.assertEqual(
                uncertainty["source_absolute_start"],
                base + local_start,
            )
            self.assertEqual(
                uncertainty["source_absolute_end"],
                base + local_end,
            )
            self.assertEqual(
                uncertainty["evidence"][0]["text"],
                bronze[local_start:local_end],
            )
        # Both ">>" surfaces bind to the two distinct speaker markers in
        # left-to-right order, not to the model's wrong offsets.
        self.assertEqual(
            [item["text"] for item in result["uncertainties"]],
            [">>", ">>", "Diana", "health", "ult", "base", "jailed", "wave"],
        )
        self.assertEqual(
            result["uncertainties"][1]["target_local_start"],
            bronze.index(">>", 1),
        )
        # Full provenance is sealed deterministically even though the raw
        # response carried no lineage fields.
        provenance = result["provenance"]
        self.assertEqual(provenance["task_kind"], TEXT_RESTORATION_TASK_KIND)
        self.assertEqual(provenance["target"]["bronze_text"], bronze)
        self.assertEqual(
            provenance["target"]["bronze_text_sha256"],
            text_sha256(bronze),
        )
        self.assertEqual(provenance["prompt_version"], MECHANICAL_PROMPT_VERSION)
        self.assertEqual(
            provenance["schema_version"],
            MECHANICAL_RESPONSE_SCHEMA_VERSION,
        )
        # The raw response is preserved content-addressed and the validated
        # proposals are retained verbatim as the audit trail.
        self.assertEqual(
            result["model_call"]["raw_response_sha256"],
            text_sha256(raw_response),
        )
        self.assertEqual(result["raw_proposals"]["clean_text"], bronze)
        self.assertEqual(
            len(result["raw_proposals"]["uncertainties"]),
            8,
        )
        self.assertEqual(
            result["uncertainties"][0]["proposal_index"],
            result["raw_proposals"]["uncertainties"].index(
                result["raw_proposals"]["uncertainties"][0],
            ),
        )

    def test_mechanical_missing_lineage_is_sealed_and_semantic_fails(self):
        selected = _selected("He hit R.", champion="Lux")
        raw = _compact_mechanical_raw(
            selected,
            clean_text=selected["source_text"],
        )
        # The provider response omits task_kind/target/hashes/metadata
        # entirely; deterministic sealing must still produce full lineage.
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([raw]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            result["provenance"]["task_kind"],
            TEXT_RESTORATION_TASK_KIND,
        )
        self.assertEqual(
            result["provenance"]["target"]["window_id"],
            selected["window_id"],
        )
        self.assertEqual(
            result["provenance"]["input_metadata"]["champion"]["value"],
            "Lux",
        )
        self.assertEqual(result["provenance"]["rationale"], "test")
        # Semantic structural keys anywhere still fail closed, including
        # inside repair proposals and nested uncertainty objects.
        for semantic_field in (
            "entities", "champion_binding", "ability_owner", "events",
        ):
            bad = json.loads(raw)
            bad["repairs"] = [{
                "original_text": "He",
                "replacement": "H",
                "repair_type": "SPELLING",
                "confidence": "HIGH",
                "rationale": "prohibited semantic field",
                semantic_field: "prohibited",
            }]
            with self.assertRaises(ProviderCorrectionExhausted) as caught:
                run_mechanical_cleanup(
                    selected,
                    chat=CountingChat([
                        json.dumps(bad), json.dumps(bad), json.dumps(bad),
                        json.dumps(bad),
                    ]),
                    config_hash=canonical_sha256({"v": 1}),
                )
            self.assertIn(
                "semantic extraction field",
                caught.exception.attempts[0]["error"],
            )
        nested = json.loads(raw)
        nested["uncertainties"] = [{
            "surface_text": "He",
            "uncertainty_type": "ASR_ALTERNATIVES",
            "alternatives": [{"text": "he", "confidence": "LOW"}],
            "note": "prohibited nested semantic field",
            "champion_binding": {"entity": "Lux"},
        }]
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    json.dumps(nested), json.dumps(nested), json.dumps(nested),
                    json.dumps(nested),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "semantic extraction field",
            caught.exception.attempts[0]["error"],
        )

    def test_mechanical_clear_words_need_no_uncertainties_and_cap(self):
        selected = _selected(
            ">> okay, >> press B. Diana has health and can press ult.",
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([_mechanical_raw(selected)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        # Clear standard words, intentional speaker markers, and correct
        # domain tokens are never required to be listed as uncertainties.
        self.assertEqual(result["uncertainties"], [])

        surfaces = [">>", "okay", "press", "B", "Diana", "has", "health", "can"]
        proposals = [
            {
                "surface_text": surface,
                "uncertainty_type": "ASR_ALTERNATIVES",
                "alternatives": [
                    {"text": surface, "confidence": "HIGH"},
                    {"text": surface + "?", "confidence": "LOW"},
                ],
                "note": "clear word listed only to exercise the cap",
            }
            for surface in surfaces
        ]
        at_cap = _compact_mechanical_raw(
            selected,
            clean_text=selected["source_text"],
            uncertainties=proposals,
        )
        at_cap_result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([at_cap]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(len(at_cap_result["uncertainties"]), 8)

        over_cap = _compact_mechanical_raw(
            selected,
            clean_text=selected["source_text"],
            uncertainties=proposals + [{
                "surface_text": "ult",
                "uncertainty_type": "ASR_ALTERNATIVES",
                "alternatives": [
                    {"text": "ult", "confidence": "HIGH"},
                ],
                "note": "ninth proposal exceeds the cap",
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([over_cap, over_cap, over_cap, over_cap]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn("cap", caught.exception.attempts[0]["error"])

    def test_mechanical_clean_text_unrepresented_edit_fails_closed(self):
        selected = _selected("he used W.")
        raw = _compact_mechanical_raw(
            selected,
            clean_text="He used W.",
            rationale="clean_text changes without any repair proposal",
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([raw, raw, raw, raw]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertEqual(
            len(caught.exception.attempts), MECHANICAL_MAX_CORRECTIONS + 1,
        )
        self.assertTrue(all(
            attempt["status"] == "FAILED"
            for attempt in caught.exception.attempts
        ))
        self.assertIn("clean_text must equal", caught.exception.attempts[0]["error"])
        self.assertEqual(
            [attempt["attempt_index"] for attempt in caught.exception.attempts],
            [0, 1, 2, 3],
        )

    def test_mechanical_ambiguous_unbindable_overlapping_fail_closed(self):
        selected = _selected("he used W. he escaped.")
        config_hash = canonical_sha256({"v": 1})

        # A quoted original that never appears in Bronze cannot be bound.
        unbindable = _compact_mechanical_raw(
            selected,
            clean_text=selected["source_text"],
            repairs=[{
                "original_text": "xyz not in bronze",
                "replacement": "xyz",
                "repair_type": "SPELLING",
                "confidence": "HIGH",
                "rationale": "not an exact quote",
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    unbindable, unbindable, unbindable, unbindable,
                ]),
                config_hash=config_hash,
            )
        self.assertIn(
            "cannot be bound deterministically",
            caught.exception.attempts[0]["error"],
        )

        # A repeated original with a single proposal is ambiguous: ordered
        # binding takes the first occurrence, so clean_text cannot match.
        ambiguous = _compact_mechanical_raw(
            selected,
            clean_text="he used W. He escaped.",
            repairs=[{
                "original_text": "he",
                "replacement": "He",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "rationale": "second occurrence intended",
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted):
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    ambiguous, ambiguous, ambiguous, ambiguous,
                ]),
                config_hash=config_hash,
            )

        # Overlapping proposals fail closed.
        overlapping = _compact_mechanical_raw(
            selected,
            clean_text="xxyyed W. he escaped.",
            repairs=[
                {
                    "original_text": "he us",
                    "replacement": "xx",
                    "repair_type": "SPELLING",
                    "confidence": "HIGH",
                    "rationale": "first span",
                },
                {
                    "original_text": "used",
                    "replacement": "yy",
                    "repair_type": "SPELLING",
                    "confidence": "HIGH",
                    "rationale": "second overlapping span",
                },
            ],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    overlapping, overlapping, overlapping, overlapping,
                ]),
                config_hash=config_hash,
            )
        self.assertIn("must not overlap", caught.exception.attempts[0]["error"])

    def test_mechanical_truncated_provider_json_fails_closed(self):
        selected = _selected("He hit R.")
        raw = _compact_mechanical_raw(
            selected,
            clean_text=selected["source_text"],
        )
        truncated = raw[: len(raw) // 2]
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    truncated, truncated, truncated, truncated,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn("JSON", caught.exception.attempts[0]["error"])

    def test_abcd_exact_target_invariant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "phase2k"
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="no_provider",
            )
            records = load_json_strict(
                result["paths"]["records"], label="records",
            )
            by_window: dict[str, list[dict[str, Any]]] = {}
            for record in records["records"]:
                by_window.setdefault(record["window_id"], []).append(record)
            for window_id, window_records in by_window.items():
                targets = {record["target"]["text"] for record in window_records}
                offsets = {
                    (
                        record["target"]["source_absolute_start"],
                        record["target"]["source_absolute_end"],
                    )
                    for record in window_records
                }
                hashes = {
                    record["target"]["text_sha256"] for record in window_records
                }
                self.assertEqual(len(targets), 1)
                self.assertEqual(len(offsets), 1)
                self.assertEqual(len(hashes), 1)
                by_type = {
                    record["record_type"]: record for record in window_records
                }
                self.assertEqual(by_type["A"]["content"]["text"], by_type["B"]["content"]["text"])
                d_content = by_type["D"]["content"]
                self.assertEqual(
                    d_content["clean_target_transcript"],
                    apply_mechanical_repairs(
                        by_type["A"]["content"]["text"],
                        d_content["contextual_repairs"],
                    ),
                )
                self.assertEqual(d_content["generation_status"], "NOT_GENERATED")
                self.assertTrue(d_content["is_placeholder"])
                self.assertEqual(d_content["contextual_repairs"], [])
            radius_labels = {
                entry["radius"] for entry in records["context_radius_entries"]
            }
            self.assertEqual(
                radius_labels,
                {"target_only", "r2", "r5", "r10", "bounded_local_episode"},
            )
            self.assertEqual(len(records["context_radius_entries"]), 30 * 5)

    def test_human_review_packet_blank_and_mapping_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=root / "phase2k",
                mode="no_provider",
            )
            packet = load_json_strict(
                result["paths"]["human_packet"], label="human packet",
            )
            mapping = load_json_strict(
                result["paths"]["human_mapping"], label="mapping",
            )
            self.assertEqual(len(packet["review_items"]), 30 * (3 + 5))
            for item in packet["review_items"]:
                self.assertTrue(all(value is None for value in item["scores"].values()))
                self.assertIsNone(item["reviewer"])
                self.assertIsNone(item["completed_at"])
                # Official packet is strictly blind: no condition/radius/
                # record identity anywhere.
                self.assertNotIn("condition_code", item)
                self.assertNotIn("record_type", item)
                self.assertNotIn("radius", item)
                self.assertNotIn("provenance", item)
                presentation = item["presentation"]
                self.assertEqual(
                    presentation["schema_version"],
                    "phase2k-review-presentation-v2",
                )
                self.assertTrue(presentation["sections"])
                self.assertTrue(all(
                    section["text"].strip() for section in presentation["sections"]
                ))
                self.assertIn(
                    "primary",
                    {section["id"] for section in presentation["sections"]},
                )
                label = item["blinded_label"]
                self.assertEqual(mapping["labels"][label]["window_id"], item["window_id"])
                mapped = mapping["labels"][label]
                self.assertIn(mapped["condition_code"], ("A", "B", "C"))
                self.assertEqual(
                    mapped["target_text_sha256"],
                    item["presentation"]["target_sha256"],
                )
                self.assertTrue(mapped["record_id"] or mapped["entry_id"])
                self.assertTrue(mapped["record_sha256"] or mapped["entry_sha256"])
            self.assertNotIn(
                "D",
                {mapping["labels"][item["blinded_label"]]["condition_code"]
                 for item in packet["review_items"]},
            )
            self.assertEqual(
                packet["blinding"]["mapping_sha256"],
                mapping["content_sha256"],
            )

    def test_incomplete_human_review_rejected_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=root / "phase2k",
                mode="no_provider",
            )
            packet = load_json_strict(
                result["paths"]["human_packet"], label="human packet",
            )
            item_ids = [item["review_item_id"] for item in packet["review_items"]]
            incomplete = {
                item_id: {
                    "scores": {
                        field: 3 if field != "causality" else None
                        for field in HUMAN_SCORE_FIELDS
                    },
                    "reviewer": "test-human",
                    "completed_at": "2026-08-19T00:00:00.000Z",
                    "notes": [],
                }
                for item_id in item_ids
            }
            with self.assertRaises(ValueError):
                import_completed_human_reviews(
                    packet,
                    incomplete,
                    reviewer="test-human",
                    completed_at="2026-08-19T00:00:00.000Z",
                )
            complete = {
                item_id: {
                    "scores": {
                        field: (index % 6)
                        for index, field in enumerate(HUMAN_SCORE_FIELDS)
                    },
                    "reviewer": "test-human",
                    "completed_at": "2026-08-19T00:00:00.000Z",
                    "notes": [],
                }
                for item_id in item_ids
            }
            finalized = import_completed_human_reviews(
                packet,
                complete,
                reviewer="test-human",
                completed_at="2026-08-19T00:00:00.000Z",
            )
            records_obj = load_json_strict(
                result["paths"]["records"], label="records",
            )
            mapping = load_json_strict(
                result["paths"]["human_mapping"], label="mapping",
            )
            summary = summarize_human_reviews(
                finalized, mapping=mapping, records_file=records_obj,
            )
            self.assertEqual(
                summary["overall"]["item_count"], len(item_ids),
            )
            self.assertIn("by_condition", summary)
            self.assertIn("by_radius", summary)
            self.assertIn("r2", summary["by_radius"])
            self.assertIn("review_gate", summary)
            self.assertIn(
                summary["review_gate"]["status"], ("PASSED", "FAILED"),
            )
            self.assertTrue(summary["review_gate"]["evaluated"])
            self.assertIn("asr_repair_correctness", summary["review_gate"]["metrics"])
            self.assertIn("d_semantic_recoverability", summary["review_gate"]["metrics"])
            for code in ("A", "B", "C"):
                self.assertIn("entity_ability_completeness", summary["by_condition"][code])
            # Official artifact remains blank.
            official = load_json_strict(
                result["paths"]["human_packet"], label="official packet",
            )
            for item in official["review_items"]:
                self.assertTrue(all(value is None for value in item["scores"].values()))

    def test_validate_output_directory_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "phase2k"
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="no_provider",
            )
            validated = validate_output_directory(
                output_dir=output,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )
            self.assertEqual(
                validated["records_sha256"], result["records_sha256"],
            )
            with self.assertRaises(ValueError):
                build_phase2k_outputs(
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                    doc_path=None,
                    output_dir=output,
                    mode="no_provider",
                )

    def test_path_invariant_frozen_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            doc_path = root / "replication.md"
            doc_path.write_text("frozen doc", encoding="utf-8")
            self.assertEqual(
                normalize_path_locator(Path("data/phase2j/x.json").resolve()),
                "data/phase2j/x.json",
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                relative = build_phase2k_outputs(
                    manifest_path=Path("window-selection-manifest-v1.json"),
                    packet_path=Path("reviewed-endpoint-annotation-packet-v1.json"),
                    db_path=Path("videos.db"),
                    doc_path=Path("replication.md"),
                    output_dir=root / "phase2k-rel",
                    mode="no_provider",
                )
                absolute = build_phase2k_outputs(
                    manifest_path=manifest_path.resolve(),
                    packet_path=packet_path.resolve(),
                    db_path=db_path.resolve(),
                    doc_path=doc_path.resolve(),
                    output_dir=root / "phase2k-abs",
                    mode="no_provider",
                )
            finally:
                os.chdir(old_cwd)
            self.assertEqual(
                relative["frozen_manifest_sha256"],
                absolute["frozen_manifest_sha256"],
            )
            self.assertEqual(relative["records_sha256"], absolute["records_sha256"])
            self.assertEqual(
                relative["human_packet_sha256"], absolute["human_packet_sha256"],
            )
            # Validate-only accepts the other path spelling of the same files.
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                validated = validate_output_directory(
                    output_dir=root / "phase2k-abs",
                    manifest_path=Path("window-selection-manifest-v1.json"),
                    packet_path=Path("reviewed-endpoint-annotation-packet-v1.json"),
                    db_path=Path("videos.db"),
                )
            finally:
                os.chdir(old_cwd)
            self.assertEqual(
                validated["frozen_input_manifest_sha256"],
                absolute["frozen_manifest_sha256"],
            )

    def test_contextual_reconstruction_repairs_licensed_by_bindings(self):
        text = "Viktor is the player. He hit R. The replay shows it."
        target = "He hit R."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=end,
            bronze_text=target,
            previous_segments=1,
            following_segments=2,
            radius_label="r5",
        )
        target_segment_id = next(
            item["segment_id"] for item in context["segments"]
            if item["kind"] == "target"
        )
        evidence = {
            "segment_id": target_segment_id,
            "source_absolute_start": start,
            "source_absolute_end": start + 2,
            "text": "He",
        }
        repair = {
            "original_text": "He",
            "replacement": "Viktor",
            "repair_type": "PRONOUN_RESOLUTION",
            "confidence": "HIGH",
            "evidence_quotes": ["He"],
            "rationale": "context names Viktor as the player",
        }
        binding = {
            "slot": "pronouns",
            "mention_text": "He",
            "resolved_candidate": "Viktor",
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_quotes": ["He"],
            "alternatives": [],
            "metadata_contributed": True,
            "rationale": "licensed by context",
        }
        final_slots = _slots(decision="SUFFICIENT", champion="Viktor")
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {"slots": final_slots, "metadata_conflicts": []},
            },
        }
        raw = _reconstruction_raw(
            cleaned="Viktor hit R.",
            bronze=target,
            base_offset=start,
            selected=selected,
            contextual_repairs=[repair],
            bindings_override=[binding],
        )
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=diagnostic,
            chat=CountingChat([raw]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(reconstruction["clean_target_transcript"], "Viktor hit R.")
        # D is contextually reconstructed and is NOT forced to equal B.
        self.assertNotEqual(reconstruction["clean_target_transcript"], target)
        self.assertEqual(reconstruction["counts"]["contextual_repair_count"], 1)
        self.assertEqual(reconstruction["counts"]["resolution_repair_count"], 1)
        self.assertEqual(
            reconstruction["contextual_repairs"][0]["repair_type"],
            "PRONOUN_RESOLUTION",
        )
        self.assertEqual(
            reconstruction["contextual_repairs"][0]["evidence_spans"][0]["text"],
            "He",
        )

    def test_contextual_resolution_requires_resolved_licensed_binding(self):
        text = "Viktor is the player. He hit R. The replay shows it."
        target = "He hit R."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=end,
            bronze_text=target,
            previous_segments=1,
            following_segments=2,
            radius_label="r5",
        )
        target_segment_id = next(
            item["segment_id"] for item in context["segments"]
            if item["kind"] == "target"
        )
        evidence = {
            "segment_id": target_segment_id,
            "source_absolute_start": start,
            "source_absolute_end": start + 2,
            "text": "He",
        }
        repair = {
            "original_text": "He",
            "replacement": "Viktor",
            "repair_type": "PRONOUN_RESOLUTION",
            "confidence": "HIGH",
            "evidence_quotes": ["He"],
            "rationale": "context names Viktor as the player",
        }
        final_slots = _slots(decision="SUFFICIENT", champion="Viktor")
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {"slots": final_slots, "metadata_conflicts": []},
            },
        }

        def attempt(raw: str) -> None:
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=target,
                final_diagnostic=diagnostic,
                chat=CountingChat([raw, raw, raw, raw]),
                config_hash=canonical_sha256({"v": 1}),
            )

        # No binding over the mention span: the resolution is unlicensed.
        unlicensed = _reconstruction_raw(
            cleaned="Viktor hit R.",
            bronze=target,
            base_offset=start,
            selected=selected,
            contextual_repairs=[repair],
            bindings_override=[],
        )
        with self.assertRaises(ValueError):
            attempt(unlicensed)
        # Unresolved binding (AMBIGUOUS) must not license a rewrite.
        ambiguous_binding = {
            "slot": "pronouns",
            "mention_text": "He",
            "resolved_candidate": "AMBIGUOUS",
            "resolved_status": "AMBIGUOUS",
            "confidence": "LOW",
            "evidence_quotes": [],
            "alternatives": [],
            "metadata_contributed": False,
            "rationale": "not resolvable",
        }
        ambiguous_slots = dict(final_slots)
        ambiguous_slots["pronouns"] = {
            "status": "AMBIGUOUS",
            "candidates": [{
                "candidate": "Viktor",
                "confidence": "LOW",
                "evidence_spans": [],
            }],
            "confidence": "LOW",
            "evidence_spans": [],
        }
        diagnostic["response"]["parsed"]["slots"] = ambiguous_slots
        unresolved_raw = _reconstruction_raw(
            cleaned="Viktor hit R.",
            bronze=target,
            base_offset=start,
            selected=selected,
            contextual_repairs=[repair],
            bindings_override=[ambiguous_binding],
        )
        with self.assertRaises(ValueError):
            attempt(unresolved_raw)

    def test_official_packet_blind_and_deterministic_shuffle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)

            def build(output: Path) -> dict[str, Any]:
                return build_phase2k_outputs(
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                    doc_path=None,
                    output_dir=output,
                    mode="no_provider",
                )

            result = build(root / "phase2k")
            packet = load_json_strict(
                result["paths"]["human_packet"], label="human packet",
            )
            mapping = load_json_strict(
                result["paths"]["human_mapping"], label="mapping",
            )

            def walk(value: object, *, skip: frozenset[str]) -> None:
                if isinstance(value, Mapping):
                    for key, item in value.items():
                        self.assertNotIn(key, {
                            "condition_code", "record_type", "radius",
                            "radius_label", "stage", "stage_label",
                            "record_id", "entry_id", "provenance",
                            "generation_status",
                        })
                        if key in skip:
                            continue
                        walk(item, skip=skip)
                elif isinstance(value, list):
                    for item in value:
                        walk(item, skip=skip)
                elif isinstance(value, str):
                    self.assertNotIn(value, {
                        "A", "B", "C", "D", "target_only", "r2", "r5",
                        "r10", "bounded_local_episode",
                        "raw_bronze", "mechanical_clean", "enlarged_context",
                        "reconstruction", "NOT_GENERATED", "GENERATED",
                    })
                    self.assertNotIn("p2k:rec:", value)
                    self.assertNotIn("p2k:radius:", value)

            walk(packet, skip=frozenset({"text", "notes"}))
            for item in packet["review_items"]:
                self.assertTrue(item["presentation"]["sections"])
                self.assertTrue(all(
                    section["text"].strip()
                    for section in item["presentation"]["sections"]
                ))
            # Presentation order is a deterministic seeded shuffle, not the
            # canonical A/B/C/radius ordering.
            result_two = build(root / "phase2k-two")
            packet_two = load_json_strict(
                result_two["paths"]["human_packet"], label="human packet two",
            )
            self.assertEqual(packet, packet_two)
            canonical = (
                ("A", None), ("B", None), ("C", None),
                ("C", "target_only"), ("C", "r2"), ("C", "r5"),
                ("C", "r10"), ("C", "bounded_local_episode"),
            )
            orders = []
            for window_id in sorted({
                item["window_id"] for item in packet["review_items"]
            }):
                sequence = tuple(
                    (
                        mapping["labels"][item["blinded_label"]][
                            "condition_code"
                        ],
                        mapping["labels"][item["blinded_label"]]["radius"],
                    )
                    for item in packet["review_items"]
                    if item["window_id"] == window_id
                )
                orders.append(sequence)
            self.assertFalse(all(order == canonical for order in orders))

    def test_packet_validator_rejects_leaked_condition_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=root / "phase2k",
                mode="no_provider",
            )
            packet = load_json_strict(
                result["paths"]["human_packet"], label="human packet",
            )
            validate_human_review_packet(packet, require_blank=True)
            for field in ("condition_code", "record_type", "radius"):
                leaked = json.loads(json.dumps(packet))
                leaked["review_items"][0][field] = "A"
                with self.assertRaises(ValueError):
                    validate_human_review_packet(leaked, require_blank=True)

    def test_review_gate_preregistered_thresholds_and_not_applicable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=root / "phase2k-live",
                mode="live",
                chat=_live_factory_chat(),
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            packet = load_json_strict(
                result["paths"]["human_packet"], label="human packet",
            )
            mapping = load_json_strict(
                result["paths"]["human_mapping"], label="mapping",
            )
            records_obj = load_json_strict(
                result["paths"]["records"], label="records",
            )

            def make_reviews(
                *,
                fail_unsupported: bool = False,
                not_applicable_meaning: bool = False,
            ) -> dict[str, Any]:
                reviews: dict[str, Any] = {}
                for item in packet["review_items"]:
                    condition = mapping["labels"][item["blinded_label"]][
                        "condition_code"
                    ]
                    scores: dict[str, Any] = {}
                    for field in HUMAN_SCORE_FIELDS:
                        if condition == "D":
                            if fail_unsupported and field == "unsupported_invention":
                                scores[field] = 5
                            elif field in (
                                "unsupported_invention", "remaining_ambiguity",
                            ):
                                scores[field] = 0
                            elif (
                                not_applicable_meaning
                                and field == "meaning_preservation"
                            ):
                                scores[field] = NOT_APPLICABLE
                            else:
                                scores[field] = 5
                        elif field == "asr_repair_correctness":
                            scores[field] = NOT_APPLICABLE
                        elif field in ("unsupported_invention", "remaining_ambiguity"):
                            scores[field] = 0
                        else:
                            scores[field] = 4
                    reviews[item["review_item_id"]] = {
                        "scores": scores,
                        "reviewer": "human",
                        "completed_at": "2026-08-19T00:00:00.000Z",
                        "notes": [],
                    }
                return reviews

            finalized = import_completed_human_reviews(
                packet,
                make_reviews(),
                reviewer="human",
                completed_at="2026-08-19T00:00:00.000Z",
            )
            summary = summarize_human_reviews(
                finalized, mapping=mapping, records_file=records_obj,
            )
            gate = summary["review_gate"]
            self.assertEqual(gate["status"], "PASSED")
            self.assertTrue(gate["evaluated"])
            self.assertEqual(summary["by_condition"]["D"]["item_count"], 30)
            self.assertEqual(summary["by_radius"]["r2"]["item_count"], 30)
            self.assertEqual(summary["by_radius"]["target_only"]["item_count"], 30)
            self.assertEqual(gate["metrics"]["d_semantic_recoverability"], 5.0)
            self.assertEqual(
                gate["metrics"]["d_over_a_semantic_recoverability_gain"], 1.0,
            )
            self.assertEqual(gate["metrics"]["d_meaning_preservation"], 5.0)
            self.assertEqual(gate["metrics"]["d_unsupported_invention"], 0.0)
            self.assertEqual(
                gate["metrics"]["asr_repair_correctness_applicable"], 30,
            )
            self.assertEqual(
                summary["by_condition"]["A"]["applicable_counts"][
                    "asr_repair_correctness"
                ]["applicable"],
                0,
            )
            self.assertEqual(
                summary["by_condition"]["A"]["applicable_counts"][
                    "asr_repair_correctness"
                ]["not_applicable"],
                30,
            )
            self.assertEqual(gate["thresholds"]["d_semantic_recoverability_min"], 4.0)
            self.assertEqual(
                gate["thresholds"]["d_over_a_semantic_recoverability_gain_min"],
                0.75,
            )

            finalized_fail = import_completed_human_reviews(
                packet,
                make_reviews(fail_unsupported=True),
                reviewer="human",
                completed_at="2026-08-19T00:00:00.000Z",
            )
            gate_fail = summarize_human_reviews(
                finalized_fail, mapping=mapping, records_file=records_obj,
            )["review_gate"]
            self.assertEqual(gate_fail["status"], "FAILED")
            self.assertTrue(any(
                reason["criterion"] == "d_unsupported_invention"
                and not reason["passed"]
                for reason in gate_fail["reasons"]
            ))

            finalized_na = import_completed_human_reviews(
                packet,
                make_reviews(not_applicable_meaning=True),
                reviewer="human",
                completed_at="2026-08-19T00:00:00.000Z",
            )
            gate_na = summarize_human_reviews(
                finalized_na, mapping=mapping, records_file=records_obj,
            )["review_gate"]
            self.assertEqual(gate_na["status"], "FAILED")
            self.assertTrue(any(
                reason["criterion"] == "d_meaning_preservation_applicable"
                and not reason["passed"]
                for reason in gate_na["reasons"]
            ))

    def test_mechanical_correction_succeeds_and_retains_all_attempts(self):
        selected = _selected("he used W.")
        bad = _compact_mechanical_raw(
            selected,
            clean_text="He used W.",
            rationale="unrepresented edit with no repair proposal",
        )
        good = _mechanical_raw(
            selected,
            repairs=[{
                "original_text": "he",
                "replacement": "He",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "rationale": "sentence start",
            }],
        )
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "raw_responses"
            result = run_mechanical_cleanup(
                selected,
                chat=CountingChat([bad, good]),
                config_hash=canonical_sha256({"v": 1}),
                raw_response_dir=raw_dir,
            )
            self.assertEqual(result["mechanical_cleaned_text"], "He used W.")
            self.assertEqual(len(result["attempts"]), 2)
            self.assertEqual(result["attempts"][0]["status"], "FAILED")
            self.assertEqual(result["attempts"][0]["attempt_index"], 0)
            self.assertIn(
                "clean_text must equal",
                result["attempts"][0]["error"],
            )
            self.assertEqual(result["attempts"][1]["status"], "OK")
            self.assertEqual(result["attempts"][1]["attempt_index"], 1)
            self.assertEqual(
                result["model_call"],
                result["attempts"][1]["model_call"],
            )
            for attempt in result["attempts"]:
                self.assertRegex(
                    attempt["model_call"]["raw_response_sha256"],
                    r"[0-9a-f]{64}",
                )
                self.assertIsNotNone(attempt["model_call"]["raw_response_path"])
                raw_file = raw_dir / attempt["model_call"]["raw_response_path"]
                self.assertTrue(raw_file.is_file())

    def test_mechanical_correction_succeeds_for_overlap_and_noop(self):
        selected = _selected("he used W.")
        config_hash = canonical_sha256({"v": 1})

        overlapping = _compact_mechanical_raw(
            selected,
            clean_text="xxyyed W.",
            repairs=[
                {
                    "original_text": "he us",
                    "replacement": "xx",
                    "repair_type": "SPELLING",
                    "confidence": "HIGH",
                    "rationale": "first span",
                },
                {
                    "original_text": "used",
                    "replacement": "yy",
                    "repair_type": "SPELLING",
                    "confidence": "HIGH",
                    "rationale": "overlapping span",
                },
            ],
        )
        good = _mechanical_raw(
            selected,
            repairs=[{
                "original_text": "he",
                "replacement": "He",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "rationale": "sentence start",
            }],
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([overlapping, good]),
            config_hash=config_hash,
        )
        self.assertEqual(result["mechanical_cleaned_text"], "He used W.")
        self.assertEqual([a["status"] for a in result["attempts"]], ["FAILED", "OK"])
        self.assertIn("must not overlap", result["attempts"][0]["error"])

        noop = _compact_mechanical_raw(
            selected,
            clean_text=selected["source_text"],
            repairs=[{
                "original_text": "he",
                "replacement": "he",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "rationale": "no-op placeholder",
            }],
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([noop, good]),
            config_hash=config_hash,
        )
        self.assertEqual(result["mechanical_cleaned_text"], "He used W.")
        self.assertEqual([a["status"] for a in result["attempts"]], ["FAILED", "OK"])
        self.assertIn(
            "replacement must differ from original_text",
            result["attempts"][0]["error"],
        )

    def test_mechanical_lexical_priority_rationale_is_accepted(self):
        bronze = (
            "because you don't have mid pryo for example starting raptor "
            "The downside is they spawn faster."
        )
        selected = _selected(bronze)
        raw = _compact_mechanical_raw(
            selected,
            clean_text=(
                "Because you don't have mid prio for example starting raptor "
                "The downside is they spawn faster."
            ),
            repairs=[{
                "original_text": "because",
                "replacement": "Because",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "rationale": "Sentence start capitalization.",
            }, {
                "original_text": "pryo",
                "replacement": "prio",
                "repair_type": "ASR_HOMOPHONE",
                "confidence": "HIGH",
                "rationale": (
                    "Context-free ASR correction: 'pryo' is a common "
                    "mishearing of 'prio' (priority), a standard League term."
                ),
            }],
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([raw]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            [repair["replacement"] for repair in result["repairs"]],
            ["Because", "prio"],
        )
        self.assertIn("priority", result["raw_proposals"]["repairs"][1]["rationale"])

        # The narrow rationale allowance never permits semantic structural
        # fields; they still fail closed after bounded retries.
        bad = json.loads(raw)
        bad["repairs"][1]["entities"] = ["Lux"]
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([json.dumps(bad)] * 4),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "semantic extraction field",
            caught.exception.attempts[0]["error"],
        )

        # A semantic endpoint list disguised as rationale is still rejected.
        endpoint_list = json.loads(raw)
        endpoint_list["rationale"] = (
            "entities: Lux; events: He used R; bindings: Lux owns R"
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([json.dumps(endpoint_list)] * 4),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "semantic endpoint/extraction list",
            caught.exception.attempts[0]["error"],
        )

    def test_sufficiency_compact_quotes_bind_to_exact_offsets(self):
        text = "Lux used W. He hit R. He escaped."
        target = "He hit R."
        start = text.index(target)
        end = start + len(target)
        selected = _selected(text, champion="Lux", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=end,
            bronze_text=target,
            previous_segments=1,
            following_segments=1,
            radius_label="r2",
        )
        previous_segment_id = next(
            item["segment_id"] for item in context["segments"]
            if item["kind"] == "previous"
        )
        target_segment_id = next(
            item["segment_id"] for item in context["segments"]
            if item["kind"] == "target"
        )
        slots = _slots(decision="SUFFICIENT")
        slots["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "Lux",
                "confidence": "HIGH",
                "evidence_spans": [{
                    "segment_id": previous_segment_id,
                    "source_absolute_start": 0,
                    "source_absolute_end": len("Lux used W."),
                    "text": "Lux used W",
                }],
            }],
            "confidence": "HIGH",
            "evidence_spans": [{
                "segment_id": target_segment_id,
                "source_absolute_start": start,
                "source_absolute_end": end,
                "text": target,
            }],
        }
        raw = _compact_sufficiency_raw("SUFFICIENT", slots=slots)
        parsed = normalize_sufficiency_compact_response(
            json.loads(raw),
            transcript=text,
            context=context,
            at_max_context=False,
        )
        pronoun = parsed["slots"]["pronouns"]
        self.assertEqual(pronoun["status"], "RESOLVED")
        self.assertEqual(
            pronoun["candidates"][0]["evidence_spans"],
            [{
                "segment_id": previous_segment_id,
                "source_absolute_start": text.index("Lux used W"),
                "source_absolute_end": text.index("Lux used W") + len("Lux used W"),
                "text": "Lux used W",
            }],
        )
        self.assertEqual(
            pronoun["evidence_spans"],
            [{
                "segment_id": target_segment_id,
                "source_absolute_start": start,
                "source_absolute_end": end,
                "text": target,
            }],
        )
        self.assertEqual(set(parsed["slots"]), set(SLOT_KEYS))

        # A quote that is absent, or repeated in context but supplied once,
        # fails deterministic binding.
        absent = json.loads(raw)
        absent["slots"]["pronouns"]["candidates"][0]["evidence_quotes"] = [
            "not in the transcript",
        ]
        with self.assertRaises(ValueError) as caught:
            normalize_sufficiency_compact_response(
                absent,
                transcript=text,
                context=context,
                at_max_context=False,
            )
        self.assertIn("absent", str(caught.exception))

        ambiguous = json.loads(raw)
        ambiguous["slots"]["pronouns"]["candidates"][0]["evidence_quotes"] = ["He"]
        with self.assertRaises(ValueError) as caught:
            normalize_sufficiency_compact_response(
                ambiguous,
                transcript=text,
                context=context,
                at_max_context=False,
            )
        self.assertIn("ambiguous", str(caught.exception))

    def test_sufficiency_correction_succeeds_and_exhausts_fail_closed(self):
        text = "Lux used W. He hit R."
        target = "He hit R."
        start = text.index(target)
        selected = _selected(text, champion="Lux", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r2",
        )
        good = _sufficiency_raw("SUFFICIENT")

        wrong_keys = json.dumps({
            "decision": "SUFFICIENT",
            "slot_analysis": {"pronouns": {"status": "RESOLVED"}},
            "decision_reasoning": "wrong envelope",
        })
        outcome = run_sufficiency_diagnostic(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=target,
            at_max_context=False,
            stage_label="r1",
            chat=CountingChat([wrong_keys, good]),
            config_hash=canonical_sha256({"v": 1}),
        )
        attempts = outcome["attempts"]
        self.assertEqual([a["status"] for a in attempts], ["FAILED", "OK"])
        self.assertEqual(attempts[0]["attempt_index"], 0)
        self.assertEqual(attempts[1]["attempt_index"], 1)
        self.assertIn("key set is invalid", attempts[0]["error"])
        self.assertEqual(outcome["final_attempt"]["decision"], "SUFFICIENT")
        self.assertEqual(
            outcome["final_attempt"]["response"]["parsed"]["decision"],
            "SUFFICIENT",
        )

        float_bad = json.loads(good)
        float_bad["slots"]["pronouns"]["confidence"] = 0.95
        outcome = run_sufficiency_diagnostic(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=target,
            at_max_context=False,
            stage_label="r1",
            chat=CountingChat([json.dumps(float_bad), good]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual([a["status"] for a in outcome["attempts"]], ["FAILED", "OK"])
        self.assertIn("floating", outcome["attempts"][0]["error"])

        malformed = good[: len(good) // 2]
        outcome = run_sufficiency_diagnostic(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=target,
            at_max_context=False,
            stage_label="r1",
            chat=CountingChat([malformed, good]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual([a["status"] for a in outcome["attempts"]], ["FAILED", "OK"])
        self.assertIn("JSON", outcome["attempts"][0]["error"])

        # Exhausted retries fail closed and preserve every attempt + error.
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_sufficiency_diagnostic(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=target,
                at_max_context=False,
                stage_label="r1",
                chat=CountingChat([wrong_keys, wrong_keys, wrong_keys]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertEqual(
            len(caught.exception.attempts),
            PROVIDER_MAX_CORRECTIONS + 1,
        )
        self.assertTrue(all(
            attempt["status"] == "FAILED"
            for attempt in caught.exception.attempts
        ))
        self.assertTrue(all(
            "key set is invalid" in attempt["error"]
            for attempt in caught.exception.attempts
        ))
        for position, attempt in enumerate(caught.exception.attempts):
            self.assertEqual(attempt["attempt_index"], position)
            self.assertIsNone(attempt["decision"])

    def test_correction_cache_keys_bind_attempt_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            selected = _selected("he used W.")
            bad = _compact_mechanical_raw(
                selected,
                clean_text="He used W.",
                rationale="unrepresented edit",
            )
            good = _mechanical_raw(
                selected,
                repairs=[{
                    "original_text": "he",
                    "replacement": "He",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "rationale": "sentence start",
                }],
            )
            result = run_mechanical_cleanup(
                selected,
                chat=CountingChat([bad, good]),
                config_hash=canonical_sha256({"v": 1}),
                cache_dir=cache_dir,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            keys = {
                attempt["model_call"]["cache_key"]
                for attempt in result["attempts"]
            }
            self.assertEqual(len(keys), 2)
            entries = sorted(
                (cache_dir / name).name for name in cache_dir.iterdir()
            )
            self.assertEqual(len(entries), 2)
            by_index = {
                load_json_strict(cache_dir / name, label="cache")[
                    "attempt_index"
                ]
                for name in entries
            }
            self.assertEqual(by_index, {0, 1})

    def test_failure_artifacts_retain_attempt_history_and_raw_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            good = _live_factory_chat()

            def chat(system: str, user: str) -> str:
                payload = json.loads(user)
                if payload.get("task", "").startswith("mechanical"):
                    selected = {
                        "source_text": payload["target"]["bronze_text"],
                        "upstream_start": payload["target"]["upstream_start"],
                        "upstream_end": payload["target"]["upstream_end"],
                        "upstream_content_sha256": payload["target"][
                            "upstream_content_sha256"
                        ],
                        "canonical_record_sha256": payload["target"][
                            "canonical_record_sha256"
                        ],
                        "window_id": payload["target"]["window_id"],
                        "source_group_id": payload["target"][
                            "source_group_id"
                        ],
                        "metadata": {
                            key: payload["metadata"][key]["value"]
                            for key in ("champion", "role", "video_title")
                            if key in payload["metadata"]
                        },
                    }
                    return _compact_mechanical_raw(
                        selected,
                        clean_text=selected["source_text"],
                        repairs=[{
                            "original_text": "the",
                            "replacement": "the",
                            "repair_type": "SPELLING",
                            "confidence": "HIGH",
                            "rationale": (
                                f"no-op for {payload['target']['window_id']}"
                            ),
                        }],
                    )
                return good(system, user)

            output = root / "phase2k-live"
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="live",
                chat=chat,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            self.assertEqual(result["window_failure_count"], 30)
            summary = load_json_strict(
                result["paths"]["build_summary"], label="summary",
            )
            self.assertEqual(summary["window_failure_count"], 30)
            # Every failed window preserves the ordered attempt history and
            # the exact final failure; every retry raw file is referenced.
            failures = sorted((output / "attempts").rglob("failure.json"))
            self.assertEqual(len(failures), 30)
            for failure_path in failures:
                failure = load_json_strict(
                    failure_path, label="failure",
                )
                self.assertEqual(failure["status"], "FAILED")
                self.assertEqual(failure["stage"], "mechanical_cleanup")
                self.assertEqual(
                    failure["attempt_count"], MECHANICAL_MAX_CORRECTIONS + 1,
                )
                self.assertEqual(
                    len(failure["attempts"]), MECHANICAL_MAX_CORRECTIONS + 1,
                )
                self.assertTrue(all(
                    attempt["status"] == "FAILED"
                    for attempt in failure["attempts"]
                ))
                self.assertIn(
                    "replacement must differ",
                    failure["attempts"][0]["error"],
                )
                self.assertIn(
                    failure["attempts"][-1]["error"],
                    failure["error"],
                )
            raw_files = sorted((output / "raw_responses").glob("*.txt"))
            # Content addressing dedupes the four identical attempts within
            # one window; every retry raw file is still referenced.
            self.assertEqual(len(raw_files), 30)
            # The full output validates: every retry raw file is
            # content-addressed and referenced by the failure artifacts.
            validate_output_directory(
                output_dir=output,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )

    def test_raw_response_validation_covers_retry_raw_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)

            def chat(system: str, user: str) -> str:
                payload = json.loads(user)
                if payload.get("task", "").startswith("mechanical"):
                    selected = {
                        "source_text": payload["target"]["bronze_text"],
                        "upstream_start": payload["target"]["upstream_start"],
                        "upstream_end": payload["target"]["upstream_end"],
                        "upstream_content_sha256": payload["target"][
                            "upstream_content_sha256"
                        ],
                        "canonical_record_sha256": payload["target"][
                            "canonical_record_sha256"
                        ],
                        "window_id": payload["target"]["window_id"],
                        "source_group_id": payload["target"][
                            "source_group_id"
                        ],
                        "metadata": {
                            key: payload["metadata"][key]["value"]
                            for key in ("champion", "role", "video_title")
                            if key in payload["metadata"]
                        },
                    }
                    if payload.get("task") == "mechanical_cleanup":
                        return _compact_mechanical_raw(
                            selected,
                            clean_text="He used W.",
                            rationale="unrepresented edit",
                        )
                    return _mechanical_raw(selected)
                return _live_factory_chat()(system, user)

            output = root / "phase2k-live"
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="live",
                chat=chat,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            records = load_json_strict(
                result["paths"]["records"], label="records",
            )
            b_record = next(
                record for record in records["records"]
                if record["record_type"] == "B"
            )
            attempts = b_record["content"]["attempts"]
            self.assertEqual([a["status"] for a in attempts], ["FAILED", "OK"])
            retry_raw_name = attempts[0]["model_call"]["raw_response_path"]
            retry_raw = output / "raw_responses" / retry_raw_name
            pristine = retry_raw.read_text(encoding="utf-8")

            def revalidate() -> None:
                validate_output_directory(
                    output_dir=output,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                )

            retry_raw.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                revalidate()
            retry_raw.write_text(pristine, encoding="utf-8")

            retry_raw.unlink()
            with self.assertRaises(ValueError):
                revalidate()
            retry_raw.write_text(pristine, encoding="utf-8")

            orphan = output / "raw_responses" / ("0" * 64 + ".txt")
            orphan.write_text("orphan", encoding="utf-8")
            with self.assertRaises(ValueError):
                revalidate()
            orphan.unlink()
            revalidate()

    def test_adaptive_loop_keeps_invariants_with_corrections(self):
        text = (
            "Earlier Lux used her ultimate. Earlier Lux moved to mid. "
            "He hit R. Later Viktor died. Later the wave reset."
        )
        target = "He hit R."
        start = text.index(target)
        selected = _selected(text, champion="Viktor", target=target)
        wrong_keys = json.dumps({
            "decision": "NEED_MORE_PREVIOUS_CONTEXT",
            "slot_analysis": {"pronouns": {"status": "UNKNOWN"}},
        })
        backward = run_adaptive_diagnostics(
            selected,
            transcript=text,
            mechanical_cleaned_text=text,
            chat=CountingChat([
                wrong_keys,
                _sufficiency_raw("NEED_MORE_PREVIOUS_CONTEXT", unresolved="pronouns"),
                _sufficiency_raw("SUFFICIENT"),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        attempts, final = backward
        self.assertEqual(
            [a["stage"] for a in attempts],
            ["r1", "r1", "r2"],
        )
        self.assertEqual(
            [a["status"] for a in attempts],
            ["FAILED", "OK", "OK"],
        )
        self.assertEqual(final["decision"], "SUFFICIENT")
        self.assertEqual(final["stage"], "r2")
        self.assertEqual(
            attempts[2]["context"]["requested"],
            {"previous_segments": 5, "following_segments": 2},
        )

        # Max-context invariant is unchanged: NEED_MORE_* at max is invalid
        # and the deterministic loop still stops at MAX_CONTEXT_BUT_UNRESOLVED.
        maximum = run_adaptive_diagnostics(
            selected,
            transcript=text,
            mechanical_cleaned_text=text,
            chat=CountingChat([
                _sufficiency_raw("NEED_MORE_FOLLOWING_CONTEXT", unresolved="pronouns"),
                _sufficiency_raw("NEED_MORE_PREVIOUS_CONTEXT", unresolved="pronouns"),
                _sufficiency_raw("NEED_BOTH", unresolved="pronouns"),
                _sufficiency_raw(
                    "MAX_CONTEXT_BUT_UNRESOLVED", unresolved="pronouns",
                ),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            [a["stage"] for a in maximum[0]],
            ["r1", "r2", "r3", "r4_bounded_local_episode"],
        )
        self.assertEqual(maximum[1]["decision"], "MAX_CONTEXT_BUT_UNRESOLVED")

    def test_sufficiency_correction_integration_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            good = _live_factory_chat()
            wrong_keys = json.dumps({
                "decision": "SUFFICIENT",
                "slot_analysis": {"pronouns": {"status": "RESOLVED"}},
                "decision_reasoning": "wrong envelope",
            })

            def chat(system: str, user: str) -> str:
                payload = json.loads(user)
                if payload.get("task", "").startswith("mechanical"):
                    selected = {
                        "source_text": payload["target"]["bronze_text"],
                        "upstream_start": payload["target"]["upstream_start"],
                        "upstream_end": payload["target"]["upstream_end"],
                        "upstream_content_sha256": payload["target"][
                            "upstream_content_sha256"
                        ],
                        "canonical_record_sha256": payload["target"][
                            "canonical_record_sha256"
                        ],
                        "window_id": payload["target"]["window_id"],
                        "source_group_id": payload["target"][
                            "source_group_id"
                        ],
                        "metadata": {
                            key: payload["metadata"][key]["value"]
                            for key in ("champion", "role", "video_title")
                            if key in payload["metadata"]
                        },
                    }
                    return _mechanical_raw(selected)
                if payload.get("task") == "semantic_sufficiency":
                    return wrong_keys
                if payload.get("task") == "semantic_sufficiency_correction":
                    return _sufficiency_raw("SUFFICIENT")
                return good(system, user)

            output = root / "phase2k-live"
            result = build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="live",
                chat=chat,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            self.assertEqual(result["window_failure_count"], 0)
            attempt_files = sorted((output / "attempts").rglob("*.json"))
            self.assertTrue(any(
                path.name == "r1.json" for path in attempt_files
            ))
            self.assertTrue(any(
                path.name == "r1.attempt-1.json" for path in attempt_files
            ))
            base_path = next(
                path for path in attempt_files if path.name == "r1.json"
            )
            retry_path = next(
                path for path in attempt_files
                if path.name == "r1.attempt-1.json"
            )
            base_attempt = load_json_strict(base_path, label="base attempt")
            corrected = load_json_strict(retry_path, label="corrected attempt")
            self.assertEqual(base_attempt["status"], "FAILED")
            self.assertIn("key set is invalid", base_attempt["error"])
            self.assertEqual(corrected["status"], "OK")
            self.assertEqual(corrected["decision"], "SUFFICIENT")
            self.assertEqual(corrected["attempt_index"], 1)
            self.assertEqual(base_attempt["attempt_index"], 0)
            validate_output_directory(
                output_dir=output,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )
            # Tamper/missing/orphan checks cover sufficiency retry raw files
            # referenced by both the failed base attempt and the correction.
            retry_raw = output / "raw_responses" / (
                corrected["model_call"]["raw_response_path"]
            )
            pristine = retry_raw.read_text(encoding="utf-8")

            def revalidate() -> None:
                validate_output_directory(
                    output_dir=output,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                )

            retry_raw.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                revalidate()
            retry_raw.write_text(pristine, encoding="utf-8")
            retry_raw.unlink()
            with self.assertRaises(ValueError):
                revalidate()
            retry_raw.write_text(pristine, encoding="utf-8")
            orphan = output / "raw_responses" / ("1" * 64 + ".txt")
            orphan.write_text("orphan", encoding="utf-8")
            with self.assertRaises(ValueError):
                revalidate()
            orphan.unlink()
            revalidate()


class Phase2KReconstructionPolishHardeningTests(unittest.TestCase):
    def _recon_setup(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, int]:
        text = "Viktor is the player. He hit R. The replay shows it."
        target = "He hit R."
        start = text.index(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=2,
            radius_label="r5",
        )
        slots = _slots(decision="SUFFICIENT", champion="Viktor")
        slots["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "Viktor",
                "confidence": "HIGH",
                "evidence_spans": [],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {"slots": slots, "metadata_conflicts": []},
            },
        }
        return selected, context, diagnostic, target, start

    def _valid_compact(self) -> dict[str, Any]:
        return {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "Viktor hit R.",
            "contextual_repairs": [{
                "original_text": "He",
                "replacement": "Viktor",
                "repair_type": "PRONOUN_RESOLUTION",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "rationale": "context names Viktor as the player",
            }],
            "bindings": [{
                "slot": "pronouns",
                "mention_text": "He",
                "resolved_candidate": "Viktor",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "alternatives": [{
                    "candidate": "Viktor",
                    "evidence_quotes": ["Viktor is the player"],
                    "note": "licensed by context",
                }],
                "metadata_contributed": False,
                "rationale": "context names Viktor as the player",
            }],
            "unresolved_alternatives": [{
                "slot": "discourse_refs",
                "mention_text": "R.",
                "alternatives": [{
                    "candidate": "the ultimate",
                    "confidence": "LOW",
                    "evidence_quotes": ["The replay shows it."],
                }],
                "evidence_quotes": ["The replay shows it."],
                "note": "ambiguous ability reference",
            }],
            "rationale": "context names Viktor as the player",
        }

    def test_reconstruction_compact_happy_path_seals_everything(self):
        selected, context, diagnostic, target, start = self._recon_setup()
        compact = self._valid_compact()
        diagnostic["decision"] = "MAX_CONTEXT_BUT_UNRESOLVED"
        diagnostic["response"]["parsed"]["slots"]["discourse_refs"] = {
            "status": "CONTEXT_INSUFFICIENT",
            "candidates": [],
            "confidence": "LOW",
            "evidence_spans": [],
        }
        raw = json.dumps(compact)
        reconstruction = run_reconstruction(
            selected,
            transcript="Viktor is the player. He hit R. The replay shows it.",
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=diagnostic,
            chat=CountingChat([raw]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"], "Viktor hit R.",
        )
        repair = reconstruction["contextual_repairs"][0]
        self.assertEqual(repair["repair_id"], "p2k:ctx:r0001")
        self.assertEqual(repair["target_local_start"], 0)
        self.assertEqual(repair["target_local_end"], 2)
        self.assertEqual(repair["source_absolute_start"], start)
        self.assertEqual(repair["source_absolute_end"], start + 2)
        self.assertEqual(
            repair["evidence_spans"][0]["text"], "Viktor is the player",
        )
        binding = reconstruction["bindings"][0]
        self.assertEqual(binding["binding_id"], "p2k:ctx:b0001")
        self.assertEqual(binding["mention"]["text"], "He")
        self.assertEqual(binding["resolved_candidate"], "Viktor")
        self.assertEqual(
            binding["evidence_spans"][0]["text"], "Viktor is the player",
        )
        unresolved = reconstruction["unresolved_alternatives"][0]
        self.assertEqual(unresolved["unresolved_id"], "p2k:ctx:u0001")
        self.assertEqual(unresolved["mention"]["text"], "R.")
        provenance = reconstruction["provenance"]
        self.assertEqual(
            provenance["task_kind"], "CONTEXTUAL_RECONSTRUCTION",
        )
        self.assertEqual(provenance["target"]["window_id"], selected["window_id"])
        self.assertEqual(provenance["target"]["bronze_text"], target)
        self.assertEqual(provenance["rationale"], compact["rationale"])
        self.assertEqual(
            provenance["schema_version"], RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
        )
        self.assertEqual(reconstruction["raw_compact"], compact)
        self.assertEqual(len(reconstruction["attempts"]), 1)
        self.assertEqual(reconstruction["attempts"][0]["status"], "OK")
        self.assertEqual(
            len(reconstruction["attempts"][0]["model_call"]["raw_response_sha256"]),
            64,
        )

    def test_reconstruction_provider_schema_rejects_offsets_ids_extra_keys(self):
        selected, context, diagnostic, _target, _start = self._recon_setup()
        base = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "He hit R.",
            "contextual_repairs": [],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "test",
        }
        top_provenance = dict(base)
        top_provenance["provenance"] = {"task_kind": "CONTEXTUAL_RECONSTRUCTION"}
        repair_offsets = dict(base)
        repair_offsets["contextual_repairs"] = [{
            "original_text": "He",
            "replacement": "Viktor",
            "repair_type": "PRONOUN_RESOLUTION",
            "confidence": "HIGH",
            "evidence_quotes": ["He"],
            "rationale": "test",
            "target_local_start": 0,
            "source_absolute_start": 100,
        }]
        binding_id = dict(base)
        binding_id["bindings"] = [{
            "slot": "pronouns",
            "mention_text": "He",
            "resolved_candidate": "Viktor",
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_quotes": [],
            "alternatives": [],
            "metadata_contributed": True,
            "rationale": "test",
            "binding_id": "provider-supplied-id",
        }]
        for label, case in (
            ("provenance echo", top_provenance),
            ("repair offsets", repair_offsets),
            ("binding id", binding_id),
        ):
            raw = json.dumps(case)
            with self.assertRaises(ProviderCorrectionExhausted) as caught:
                run_reconstruction(
                    selected,
                    transcript="Viktor is the player. He hit R. The replay shows it.",
                    context=context,
                    mechanical_cleaned_text="He hit R.",
                    final_diagnostic=diagnostic,
                    chat=CountingChat([raw, raw, raw, raw]),
                    config_hash=canonical_sha256({"v": 1}),
                )
            self.assertIn(
                "key set", caught.exception.attempts[0]["error"], msg=label,
            )

    def test_reconstruction_corrections_succeed_after_each_failure_type(self):
        selected, context, diagnostic, _target, _start = self._recon_setup()
        valid = self._valid_compact()
        valid_raw = json.dumps(valid)
        malformed = "{definitely not json"
        missing = dict(valid)
        del missing["rationale"]
        unrepresented = dict(valid)
        unrepresented["contextual_repairs"] = []
        noop = dict(valid)
        noop["contextual_repairs"] = [{
            "original_text": "He",
            "replacement": "He",
            "repair_type": "PRONOUN_RESOLUTION",
            "confidence": "HIGH",
            "evidence_quotes": ["Viktor is the player"],
            "rationale": "no-op",
        }]
        absent = dict(valid)
        absent["contextual_repairs"] = [{
            "original_text": "not-present",
            "replacement": "Viktor",
            "repair_type": "PRONOUN_RESOLUTION",
            "confidence": "HIGH",
            "evidence_quotes": ["Viktor is the player"],
            "rationale": "absent quote",
        }]
        overlap = dict(valid)
        overlap["contextual_repairs"] = [
            {
                "original_text": "He hit",
                "replacement": "Viktor hit",
                "repair_type": "PRONOUN_RESOLUTION",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "rationale": "overlap a",
            },
            {
                "original_text": "hit R.",
                "replacement": "attacked.",
                "repair_type": "DOMAIN_SPELLING",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "rationale": "overlap b",
            },
        ]
        cases = (
            ("malformed json", malformed),
            ("missing keys", json.dumps(missing)),
            ("unrepresented edit", json.dumps(unrepresented)),
            ("no-op", json.dumps(noop)),
            ("absent quote", json.dumps(absent)),
            ("overlap", json.dumps(overlap)),
        )
        for label, bad_raw in cases:
            reconstruction = run_reconstruction(
                selected,
                transcript="Viktor is the player. He hit R. The replay shows it.",
                context=context,
                mechanical_cleaned_text="He hit R.",
                final_diagnostic=diagnostic,
                chat=CountingChat([bad_raw, valid_raw]),
                config_hash=canonical_sha256({"v": 1}),
            )
            self.assertEqual(
                reconstruction["clean_target_transcript"],
                "Viktor hit R.",
                msg=label,
            )
            self.assertEqual(len(reconstruction["attempts"]), 2, msg=label)
            self.assertEqual(
                reconstruction["attempts"][0]["status"], "FAILED", msg=label,
            )
            self.assertTrue(
                reconstruction["attempts"][0]["error"], msg=label,
            )

    def test_reconstruction_exhaustion_preserves_all_four_attempts(self):
        selected, context, diagnostic, _target, _start = self._recon_setup()
        malformed = "{bad json"
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript="Viktor is the player. He hit R. The replay shows it.",
                context=context,
                mechanical_cleaned_text="He hit R.",
                final_diagnostic=diagnostic,
                chat=CountingChat([
                    malformed, malformed, malformed, malformed,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertEqual(
            len(caught.exception.attempts), RECONSTRUCTION_MAX_CORRECTIONS + 1,
        )
        self.assertEqual(RECONSTRUCTION_MAX_CORRECTIONS + 1, 4)
        self.assertEqual(
            [attempt["attempt_index"] for attempt in caught.exception.attempts],
            [0, 1, 2, 3],
        )
        self.assertTrue(all(
            attempt["status"] == "FAILED" for attempt in caught.exception.attempts
        ))
        self.assertEqual(
            [attempt["attempt_kind"] for attempt in caught.exception.attempts],
            ["base", "correction:1", "correction:2", "correction:3"],
        )
        self.assertTrue(all(
            attempt["error"] for attempt in caught.exception.attempts
        ))

    def test_resolution_repair_requires_final_diagnostic_license(self):
        selected, context, diagnostic, _target, _start = self._recon_setup()
        selected["metadata"]["champion"] = "Lux"
        diagnostic["response"]["parsed"]["slots"]["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "Ahri",
                "confidence": "HIGH",
                "evidence_spans": [],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        compact = self._valid_compact()
        raw = json.dumps(compact)
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript="Viktor is the player. He hit R. The replay shows it.",
                context=context,
                mechanical_cleaned_text="He hit R.",
                final_diagnostic=diagnostic,
                chat=CountingChat([raw, raw, raw, raw]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "not licensed", caught.exception.attempts[0]["error"],
        )

    def test_unresolved_mention_is_never_rewritten(self):
        text = "Viktor is the player. He hit R."
        target = "He hit R."
        start = text.index(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r5",
        )
        slots = _slots(decision="SUFFICIENT", champion="Viktor")
        slots["pronouns"] = {
            "status": "UNKNOWN",
            "candidates": [],
            "confidence": "LOW",
            "evidence_spans": [],
        }
        diagnostic = {
            "decision": "MAX_CONTEXT_BUT_UNRESOLVED",
            "response": {
                "parsed": {"slots": slots, "metadata_conflicts": []},
            },
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "Viktor hit R.",
            "contextual_repairs": [{
                "original_text": "He",
                "replacement": "Viktor",
                "repair_type": "PRONOUN_RESOLUTION",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "rationale": "tries to rewrite an unresolved mention",
            }],
            "bindings": [{
                "slot": "pronouns",
                "mention_text": "He",
                "resolved_candidate": "UNKNOWN",
                "resolved_status": "UNKNOWN",
                "confidence": "LOW",
                "evidence_quotes": [],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "unresolved",
            }],
            "unresolved_alternatives": [],
            "rationale": "test",
        }
        raw = json.dumps(compact)
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=target,
                final_diagnostic=diagnostic,
                chat=CountingChat([raw, raw, raw, raw]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "not licensed by a RESOLVED binding",
            caught.exception.attempts[0]["error"],
        )

    def test_literal_bronze_abstraction_words_survive_and_new_ones_fail(self):
        text = "Push priority now, apply pressure. He hit R."
        target = "He hit R."
        start = text.index(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r5",
        )
        slots = _slots(decision="SUFFICIENT", champion="Viktor")
        slots["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "Viktor",
                "confidence": "HIGH",
                "evidence_spans": [],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {"slots": slots, "metadata_conflicts": []},
            },
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "Viktor hit R.",
            "contextual_repairs": [{
                "original_text": "He",
                "replacement": "Viktor",
                "repair_type": "PRONOUN_RESOLUTION",
                "confidence": "HIGH",
                "evidence_quotes": ["Push priority now, apply pressure."],
                "rationale": "priority and pressure are literal Bronze words",
            }],
            "bindings": [{
                "slot": "pronouns",
                "mention_text": "He",
                "resolved_candidate": "Viktor",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": [],
                "alternatives": [],
                "metadata_contributed": True,
                "rationale": "licensed",
            }],
            "unresolved_alternatives": [],
            "rationale": "priority and pressure stay source-faithful",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=diagnostic,
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(reconstruction["clean_target_transcript"], "Viktor hit R.")

        invented = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "tempo hit R.",
            "contextual_repairs": [{
                "original_text": "He",
                "replacement": "tempo",
                "repair_type": "DOMAIN_SPELLING",
                "confidence": "HIGH",
                "evidence_quotes": ["He"],
                "rationale": "new strategy abstraction",
            }],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "invented tempo",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=target,
                final_diagnostic=diagnostic,
                chat=CountingChat([
                    json.dumps(invented),
                    json.dumps(invented),
                    json.dumps(invented),
                    json.dumps(invented),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "not licensed by exact source evidence",
            caught.exception.attempts[0]["error"],
        )

    def test_polish_support_modes_and_zero_operation_paraphrase(self):
        selected = _selected("He hit R.", champion="Lux")
        text = "He hit R."
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=0,
            target_end=len(text),
            bronze_text=text,
            previous_segments=0,
            following_segments=0,
            radius_label="target_only",
        )
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": _slots(decision="SUFFICIENT", champion="Lux"),
                    "metadata_conflicts": [],
                },
            },
        }
        no_op_reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=diagnostic,
            chat=CountingChat([_reconstruction_raw(
                cleaned=text,
                bronze=text,
                base_offset=0,
                selected=selected,
                bindings_override=[],
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            _reconstruction_operation_ids(no_op_reconstruction), set(),
        )

        def polish_with(statements: list[dict[str, Any]]) -> dict[str, Any]:
            return run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                reconstruction=no_op_reconstruction,
                chat=CountingChat([_polish_raw(
                    selected,
                    no_op_reconstruction,
                    statements=statements,
                )]),
                config_hash=canonical_sha256({"v": 1}),
            )

        unchanged = polish_with([{
            "text": text,
            "modality_preserved": True,
            "negation_preserved": True,
            "uncertainty_preserved": True,
            "evidence_quotes": [text],
            "reconstruction_operation_ids": [],
            "support_mode": "UNCHANGED_EXACT",
            "unchanged_source_quote": text,
        }])
        self.assertEqual(
            unchanged["statements"][0]["support_mode"], "UNCHANGED_EXACT",
        )
        self.assertEqual(
            unchanged["statements"][0]["unchanged_source_quote"]["text"], text,
        )
        self.assertEqual(
            unchanged["statements"][0]["statement_id"], "p2k:stmt:s0001",
        )

        paraphrase = polish_with([{
            "text": "Lux hits R",
            "modality_preserved": True,
            "negation_preserved": True,
            "uncertainty_preserved": True,
            "evidence_quotes": [text],
            "reconstruction_operation_ids": [],
            "support_mode": "EVIDENCE_PARAPHRASE",
            "unchanged_source_quote": None,
        }])
        self.assertEqual(
            paraphrase["statements"][0]["support_mode"], "EVIDENCE_PARAPHRASE",
        )
        self.assertEqual(
            paraphrase["statements"][0]["reconstruction_operation_ids"], [],
        )

        bad_unchanged = _polish_raw(
            selected,
            no_op_reconstruction,
            statements=[{
                "text": "Lux hits R",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": [text],
                "reconstruction_operation_ids": [],
                "support_mode": "UNCHANGED_EXACT",
                "unchanged_source_quote": text,
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted):
            run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                reconstruction=no_op_reconstruction,
                chat=CountingChat([
                    bad_unchanged, bad_unchanged, bad_unchanged, bad_unchanged,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )

        bad_derived = _polish_raw(
            selected,
            no_op_reconstruction,
            statements=[{
                "text": "Lux hits R",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": [text],
                "reconstruction_operation_ids": [],
                "support_mode": "RECONSTRUCTION_DERIVED",
                "unchanged_source_quote": None,
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted):
            run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                reconstruction=no_op_reconstruction,
                chat=CountingChat([
                    bad_derived, bad_derived, bad_derived, bad_derived,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )

    def test_polish_quote_binding_ids_corrections_and_unsupported_strict(self):
        selected = _selected("You push the wave now. It is safe.", champion="Lux")
        text = "You push the wave now. It is safe."
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=0,
            target_end=len(text),
            bronze_text=text,
            previous_segments=0,
            following_segments=0,
            radius_label="target_only",
        )
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": _slots(decision="SUFFICIENT", champion="Lux"),
                    "metadata_conflicts": [],
                },
            },
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=diagnostic,
            chat=CountingChat([_reconstruction_raw(
                cleaned=text,
                bronze=text,
                base_offset=0,
                selected=selected,
                bindings_override=[],
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        valid = _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "You push the wave now, it is safe.",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["You push the wave now."],
                "reconstruction_operation_ids": [],
                "support_mode": "EVIDENCE_PARAPHRASE",
                "unchanged_source_quote": None,
            }],
        )
        polish = run_polish(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            reconstruction=reconstruction,
            chat=CountingChat(["{bad json", valid]),
            config_hash=canonical_sha256({"v": 1}),
        )
        statement = polish["statements"][0]
        self.assertEqual(statement["statement_id"], "p2k:stmt:s0001")
        self.assertEqual(
            statement["evidence_spans"][0]["text"], "You push the wave now.",
        )
        self.assertEqual(statement["evidence_spans"][0]["target_local_start"], 0)
        self.assertEqual(
            statement["evidence_spans"][0]["source_absolute_start"],
            selected["upstream_start"],
        )
        self.assertEqual(len(polish["attempts"]), 2)

        bad_reason = _polish_raw(
            selected,
            reconstruction,
            statements=[],
            unsupported=[{
                "claim": "unsupported claim",
                "reason": "NOT_A_REAL_REASON",
                "note": "bad reason",
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                reconstruction=reconstruction,
                chat=CountingChat([
                    bad_reason, bad_reason, bad_reason, bad_reason,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "unsupported claim", caught.exception.attempts[0]["error"],
        )

    def test_reconstruction_polish_retry_raw_files_and_audit_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = (
                "Coach Viktor explains the play. "
                "You should push the wave now. "
                "Your opponent is low on mana."
            )
            transcripts = {
                f"s{index:02d}": (
                    transcript if index == 1 else
                    GENERIC_TRANSCRIPT.format(
                        champion=CHAMPIONS[index % len(CHAMPIONS)],
                    )
                )
                for index in range(1, 31)
            }
            manifest_path, packet_path, db_path = build_fixture(
                root, transcripts=transcripts,
            )
            recon_calls = [0]
            polish_calls = [0]

            def _selected_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
                target = payload["target"]
                return {
                    "source_text": target["bronze_text"],
                    "upstream_start": target["upstream_start"],
                    "upstream_end": target["upstream_end"],
                    "upstream_content_sha256": target["upstream_content_sha256"],
                    "window_id": target["window_id"],
                    "source_group_id": target["source_group_id"],
                    "canonical_record_sha256": target["canonical_record_sha256"],
                    "metadata": {
                        key: payload["metadata"][key]["value"]
                        for key in ("champion", "role", "video_title")
                        if key in payload["metadata"]
                    },
                }

            def chat(system: str, user: str) -> str:
                payload = json.loads(user)
                task = payload.get("task")
                if task == "mechanical_cleanup":
                    return _mechanical_raw(_selected_from_payload(payload))
                if task == "semantic_sufficiency":
                    return _sufficiency_raw(
                        "SUFFICIENT",
                        champion=payload["metadata"].get("champion", {}).get(
                            "value", "Lux",
                        ),
                    )
                if task in ("reconstruction", "reconstruction_correction"):
                    recon_calls[0] += 1
                    if recon_calls[0] == 1:
                        return "{broken reconstruction json"
                    selected = _selected_from_payload(payload)
                    return _reconstruction_raw(
                        cleaned=selected["source_text"],
                        bronze=selected["source_text"],
                        base_offset=selected["upstream_start"],
                        selected=selected,
                    )
                if task in ("semantic_polish", "semantic_polish_correction"):
                    polish_calls[0] += 1
                    if polish_calls[0] == 1:
                        return "{broken polish json"
                    selected = _selected_from_payload(payload)
                    return _polish_raw(
                        selected,
                        payload["reconstruction"],
                    )
                raise AssertionError(f"unexpected task {task}")

            output = root / "live"
            build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="live",
                chat=chat,
                inference_config=TEST_LIVE_INFERENCE_CONFIG,
            )
            validate_output_directory(
                output_dir=output,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )
            records = load_json_strict(
                output / OUTPUT_FILENAMES["records"], label="records",
            )
            d_content = next(
                record["content"]
                for record in records["records"]
                if record["record_type"] == "D"
            )
            self.assertEqual(
                len(d_content["reconstruction"]["attempts"]), 2,
            )
            self.assertEqual(
                len(d_content["semantic_polish"]["attempts"]), 2,
            )
            self.assertEqual(
                d_content["reconstruction"]["attempts"][0]["status"], "FAILED",
            )
            self.assertEqual(
                d_content["semantic_polish"]["attempts"][0]["status"], "FAILED",
            )
            self.assertIn("schema_version", d_content["reconstruction"]["raw_compact"])
            self.assertIn("schema_version", d_content["semantic_polish"]["raw_compact"])
            audit = load_json_strict(
                output / OUTPUT_FILENAMES["transformation_audit"], label="audit",
            )
            statement_ops = audit["window_audits"][0]["operations"][
                "polished_statements"
            ]
            self.assertEqual(len(statement_ops), 1)
            self.assertEqual(
                statement_ops[0]["support_mode"], "RECONSTRUCTION_DERIVED",
            )
            self.assertIsNone(statement_ops[0]["unchanged_source_quote"])

            def revalidate() -> None:
                validate_output_directory(
                    output_dir=output,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                )

            recon_retry = d_content["reconstruction"]["attempts"][0][
                "model_call"
            ]["raw_response_path"]
            recon_file = output / "raw_responses" / recon_retry
            pristine = recon_file.read_text(encoding="utf-8")
            recon_file.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                revalidate()
            recon_file.write_text(pristine, encoding="utf-8")
            recon_file.unlink()
            with self.assertRaises(ValueError):
                revalidate()
            recon_file.write_text(pristine, encoding="utf-8")

            polish_retry = d_content["semantic_polish"]["attempts"][0][
                "model_call"
            ]["raw_response_path"]
            polish_file = output / "raw_responses" / polish_retry
            pristine = polish_file.read_text(encoding="utf-8")
            polish_file.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                revalidate()
            polish_file.write_text(pristine, encoding="utf-8")

            orphan = output / "raw_responses" / ("2" * 64 + ".txt")
            orphan.write_text("orphan", encoding="utf-8")
            with self.assertRaises(ValueError):
                revalidate()
            orphan.unlink()
            revalidate()


class Phase2KLiveRobustnessFixTests(unittest.TestCase):
    """Regression tests for the Phase 2K v6 live-run repair fixes."""

    def _context(
        self,
        transcript: str,
        target: str,
        *,
        previous: int = 1,
        following: int = 0,
    ) -> dict[str, Any]:
        start = transcript.index(target)
        selected = _selected(transcript, target=target)
        return retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=previous,
            following_segments=following,
            radius_label="r5",
        )

    def _diagnostic(
        self,
        *,
        champion: str = "Lux",
        slots: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _slots(decision="SUFFICIENT", champion=champion)
        if slots:
            resolved.update(slots)
        return {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": resolved,
                    "metadata_conflicts": [],
                },
            },
        }

    def test_entity_licensing_accepts_casefolded_bronze_names_contractions_and_punctuation(self):
        transcript = "for this play draven is strong but I'm sure it's fine okay"
        selected = _selected(transcript, champion="Lux")
        context = self._context(transcript, transcript, previous=0)
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": (
                "For this play Draven is strong but I'm sure it's fine Okay,"
            ),
            "contextual_repairs": [
                {
                    "original_text": "for",
                    "replacement": "For",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "evidence_quotes": ["for this play draven is strong"],
                    "rationale": "sentence start",
                },
                {
                    "original_text": "draven",
                    "replacement": "Draven",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "evidence_quotes": ["draven is strong"],
                    "rationale": "champion name present in Bronze",
                },
                {
                    "original_text": "okay",
                    "replacement": "Okay,",
                    "repair_type": "PUNCTUATION",
                    "confidence": "HIGH",
                    "evidence_quotes": ["fine okay"],
                    "rationale": "turn marker punctuation",
                },
            ],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "source-faithful capitalization fixes",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=transcript,
            context=context,
            mechanical_cleaned_text=transcript,
            final_diagnostic=self._diagnostic(),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"],
            "For this play Draven is strong but I'm sure it's fine Okay,",
        )

    def test_entity_licensing_accepts_names_licensed_only_by_evidence(self):
        transcript = "Nami is the support. he is strong."
        target = "he is strong."
        selected = _selected(transcript, champion="Lux", target=target)
        context = self._context(transcript, target, previous=1)
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "Nami is strong.",
            "contextual_repairs": [{
                "original_text": "he",
                "replacement": "Nami",
                "repair_type": "DOMAIN_SPELLING",
                "confidence": "HIGH",
                "evidence_quotes": ["Nami is the support."],
                "rationale": "context names Nami as the support",
            }],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "evidence-licensed name",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=transcript,
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=self._diagnostic(),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(reconstruction["clean_target_transcript"], "Nami is strong.")

    def test_polish_entity_licensing_bronze_casefold_and_new_name_rejection(self):
        text = "nami should push the wave now"
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=self._diagnostic(),
            chat=CountingChat([_reconstruction_raw(
                cleaned=text,
                bronze=text,
                base_offset=0,
                selected=selected,
                bindings_override=[],
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        valid = _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "Nami should push the wave now",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["nami should push the wave now"],
                "reconstruction_operation_ids": [],
                "support_mode": "EVIDENCE_PARAPHRASE",
                "unchanged_source_quote": None,
            }],
        )
        polish = run_polish(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            reconstruction=reconstruction,
            chat=CountingChat([valid]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            polish["statements"][0]["text"], "Nami should push the wave now",
        )
        invented = _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "Zed should push the wave now",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["nami should push the wave now"],
                "reconstruction_operation_ids": [],
                "support_mode": "EVIDENCE_PARAPHRASE",
                "unchanged_source_quote": None,
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                reconstruction=reconstruction,
                chat=CountingChat([
                    invented, invented, invented, invented,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "unlicensed named entity",
            caught.exception.attempts[0]["error"],
        )

        apostrophized_name = _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "K'Sante should push the wave now",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["nami should push the wave now"],
                "reconstruction_operation_ids": [],
                "support_mode": "EVIDENCE_PARAPHRASE",
                "unchanged_source_quote": None,
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                reconstruction=reconstruction,
                chat=CountingChat([
                    apostrophized_name,
                    apostrophized_name,
                    apostrophized_name,
                    apostrophized_name,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "K'Sante",
            caught.exception.attempts[0]["error"],
        )

    def test_normalized_mention_binding_maps_case_variants_and_stores_exact_slice(self):
        text = "draven is strong but this darus is not."
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [
                    {
                        "candidate": "Draven",
                        "confidence": "HIGH",
                        "evidence_spans": [],
                    },
                    {
                        "candidate": "Darius",
                        "confidence": "HIGH",
                        "evidence_spans": [],
                    },
                ],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": text,
            "contextual_repairs": [],
            "bindings": [
                {
                    "slot": "champion_identities",
                    "mention_text": "Draven",
                    "resolved_candidate": "Draven",
                    "resolved_status": "RESOLVED",
                    "confidence": "HIGH",
                    "evidence_quotes": [],
                    "alternatives": [],
                    "metadata_contributed": False,
                    "rationale": "case variant of Bronze draven",
                },
                {
                    "slot": "champion_identities",
                    "mention_text": "This darus",
                    "resolved_candidate": "Darius",
                    "resolved_status": "RESOLVED",
                    "confidence": "HIGH",
                    "evidence_quotes": [],
                    "alternatives": [],
                    "metadata_contributed": False,
                    "rationale": "case variant of Bronze this darus",
                },
            ],
            "unresolved_alternatives": [],
            "rationale": "case-variant mentions bind to exact Bronze spans",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=self._diagnostic(slots=slots),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        draven_mention = reconstruction["bindings"][0]["mention"]
        self.assertEqual(draven_mention["text"], "draven")
        self.assertEqual(
            draven_mention["target_local_start"], text.index("draven"),
        )
        self.assertEqual(
            draven_mention["target_local_end"],
            text.index("draven") + len("draven"),
        )
        self.assertEqual(
            reconstruction["raw_compact"]["bindings"][0]["mention_text"],
            "Draven",
        )
        darus_mention = reconstruction["bindings"][1]["mention"]
        self.assertEqual(darus_mention["text"], "this darus")

    def test_normalized_evidence_quote_binding_maps_punctuation_and_case(self):
        transcript = "Yeah and now you don't win. he should stop."
        target = "he should stop."
        selected = _selected(transcript, champion="Lux", target=target)
        context = self._context(transcript, target, previous=1)
        slots = {
            "pronouns": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "You",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        quote = "Yeah, and now you don't win."
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "You should stop.",
            "contextual_repairs": [{
                "original_text": "he",
                "replacement": "You",
                "repair_type": "PRONOUN_RESOLUTION",
                "confidence": "HIGH",
                "evidence_quotes": [quote],
                "rationale": "context names the addressee",
            }],
            "bindings": [{
                "slot": "pronouns",
                "mention_text": "he",
                "resolved_candidate": "You",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": [quote],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "licensed by context",
            }],
            "unresolved_alternatives": [],
            "rationale": "punctuation-cleaned long quote maps uniquely",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=transcript,
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=self._diagnostic(slots=slots),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        repair_evidence = reconstruction["contextual_repairs"][0][
            "evidence_spans"
        ][0]
        self.assertEqual(repair_evidence["text"], "Yeah and now you don't win.")
        self.assertEqual(repair_evidence["source_absolute_start"], 0)
        self.assertEqual(
            reconstruction["raw_compact"]["contextual_repairs"][0][
                "evidence_quotes"
            ][0],
            quote,
        )
        binding_evidence = reconstruction["bindings"][0]["evidence_spans"][0]
        self.assertEqual(binding_evidence["text"], "Yeah and now you don't win.")

    def test_normalized_binding_rejects_lexical_and_apostrophe_changes(self):
        # "W'd" must never normalize-match Bronze "wed".
        text = "maybe he wed on the wave"
        selected = _selected(text, champion="Viktor")
        context = self._context(text, text, previous=0)
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": text,
            "contextual_repairs": [],
            "bindings": [{
                "slot": "ability_ownership",
                "mention_text": "W'd",
                "resolved_candidate": "NONE",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": [],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "apostrophe contraction is not wed",
            }],
            "unresolved_alternatives": [],
            "rationale": "must fail closed",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                final_diagnostic=self._diagnostic(champion="Viktor"),
                chat=CountingChat([
                    json.dumps(compact), json.dumps(compact),
                    json.dumps(compact), json.dumps(compact),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "absent from the supplied source",
            caught.exception.attempts[0]["error"],
        )

        # "prio" must never normalize-match context evidence "pryo".
        transcript = "mid pryo matters. he should stay."
        target = "he should stay."
        selected = _selected(transcript, champion="Viktor", target=target)
        context = self._context(transcript, target, previous=1)
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "They should stay.",
            "contextual_repairs": [{
                "original_text": "he",
                "replacement": "They",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "evidence_quotes": ["mid prio matters"],
                "rationale": "typo changes never normalized-match",
            }],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "must fail closed",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=transcript,
                context=context,
                mechanical_cleaned_text=target,
                final_diagnostic=self._diagnostic(champion="Viktor"),
                chat=CountingChat([
                    json.dumps(compact), json.dumps(compact),
                    json.dumps(compact), json.dumps(compact),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "absent from the supplied source",
            caught.exception.attempts[0]["error"],
        )

    def test_single_binding_repeated_surface_binds_first_occurrence(self):
        # A single proposal for a repeated surface is now valid: bindings
        # bind left-to-right per semantic assertion group and fewer
        # proposals than total source occurrences are permitted.
        text = "Camille Camille is strong."
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Camille",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": text,
            "contextual_repairs": [],
            "bindings": [{
                "slot": "champion_identities",
                "mention_text": "Camille",
                "resolved_candidate": "Camille",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": [],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "first occurrence of the repeated surface",
            }],
            "unresolved_alternatives": [],
            "rationale": "single binding binds deterministically",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=self._diagnostic(slots=slots),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        mention = reconstruction["bindings"][0]["mention"]
        self.assertEqual(mention["text"], "Camille")
        self.assertEqual(
            mention["target_local_start"], text.index("Camille"),
        )
        self.assertEqual(
            mention["target_local_end"],
            text.index("Camille") + len("Camille"),
        )

    def test_same_semantic_group_over_supply_fails_closed(self):
        # More proposals than occurrences inside ONE semantic assertion
        # group (same mention_text + slot + candidate + status) still
        # cannot be bound deterministically.
        text = "Camille Camille is strong."
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Camille",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        binding = {
            "slot": "champion_identities",
            "mention_text": "Camille",
            "resolved_candidate": "Camille",
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_quotes": [],
            "alternatives": [],
            "metadata_contributed": False,
                "rationale": "over-supplied semantic group",
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": text,
            "contextual_repairs": [],
            "bindings": [binding, binding, binding],
            "unresolved_alternatives": [],
            "rationale": "must fail closed",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                final_diagnostic=self._diagnostic(slots=slots),
                chat=CountingChat([
                    json.dumps(compact), json.dumps(compact),
                    json.dumps(compact), json.dumps(compact),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "cannot be bound deterministically",
            caught.exception.attempts[0]["error"],
        )

    def test_grouped_binding_different_slots_share_first_occurrence(self):
        # Live pool:jDx: 3 source "you" occurrences but four bindings in
        # different slots for the same addressed-player mention.  Each
        # semantic assertion group binds left-to-right and different
        # slots/assertions share the first deterministic occurrence.
        text = (
            "you could land ticks when you pick up champions but you "
            "should not"
        )
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        slots = _slots(decision="SUFFICIENT", champion="Lux")
        for key in (
            "ability_ownership",
            "champion_identities",
            "principal_actors",
            "pronouns",
        ):
            slots[key] = {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Mel",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            }
        binding = {
            "slot": None,
            "mention_text": "you",
            "resolved_candidate": "Mel",
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_quotes": ["you could land ticks"],
            "alternatives": [],
            "metadata_contributed": False,
            "rationale": "addressed player",
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": text,
            "contextual_repairs": [],
            "bindings": [
                {**binding, "slot": "ability_ownership"},
                {**binding, "slot": "champion_identities"},
                {**binding, "slot": "principal_actors"},
                {**binding, "slot": "pronouns"},
            ],
            "unresolved_alternatives": [],
            "rationale": "different slots share the same mention span",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=self._diagnostic(slots=slots),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(len(reconstruction["bindings"]), 4)
        first = reconstruction["bindings"][0]["mention"]
        for binding in reconstruction["bindings"][1:]:
            self.assertEqual(
                binding["mention"]["target_local_start"],
                first["target_local_start"],
            )
            self.assertEqual(
                binding["mention"]["target_local_end"],
                first["target_local_end"],
            )
        self.assertEqual(first["text"], "you")

    def test_grouped_binding_fewer_proposals_than_occurrences(self):
        # Live pool:n2: 5 source "she" occurrences but only three bindings
        # in different slots.  Fewer same-group proposals than total
        # occurrences is valid; each group binds to the first occurrence.
        text = (
            "she farms then she loses the wave she uses Q and she uses E "
            "but she has no mana"
        )
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        slots = _slots(decision="SUFFICIENT", champion="Lux")
        for key in ("pronouns", "champion_identities", "principal_actors"):
            slots[key] = {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "enemy mid laner",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            }
        binding = {
            "slot": None,
            "mention_text": "she",
            "resolved_candidate": "enemy mid laner",
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_quotes": ["she farms then she loses the wave"],
            "alternatives": [],
            "metadata_contributed": False,
            "rationale": "opponent in lane",
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": text,
            "contextual_repairs": [],
            "bindings": [
                {**binding, "slot": "pronouns"},
                {**binding, "slot": "champion_identities"},
                {**binding, "slot": "principal_actors"},
            ],
            "unresolved_alternatives": [],
            "rationale": "fewer bindings than occurrences is valid",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=self._diagnostic(slots=slots),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(len(reconstruction["bindings"]), 3)
        first = reconstruction["bindings"][0]["mention"]
        for binding in reconstruction["bindings"][1:]:
            self.assertEqual(
                binding["mention"]["target_local_start"],
                first["target_local_start"],
            )
        self.assertEqual(first["text"], "she")

    def test_entity_resolution_repair_licensed_across_repeated_surfaces(self):
        # Live pool:wTx: 4 "atrox" occurrences are repaired but only one
        # RESOLVED entity binding is supplied.  One entity binding licenses
        # repeated identical canonicalization repairs via strict surface
        # normalization of mention/candidate vs original/replacement.
        text = (
            "look at atrox atrox has 11 CS after this if atrox T is back "
            "here do you really think atrox with clo armor will win"
        )
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Aatrox",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        clean = text.replace("atrox", "Aatrox")
        repair = {
            "original_text": "atrox",
            "replacement": "Aatrox",
            "repair_type": "ENTITY_RESOLUTION",
            "confidence": "HIGH",
            "evidence_quotes": [
                "look at atrox atrox has 11 CS after this if atrox T is "
                "back here do you really think atrox with clo armor will win"
            ],
            "rationale": "Aatrox is the named champion",
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": [repair, repair, repair, repair],
            "bindings": [{
                "slot": "champion_identities",
                "mention_text": "atrox",
                "resolved_candidate": "Aatrox",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": [
                    "look at atrox atrox has 11 CS after this if atrox T "
                    "is back here do you really think atrox with clo armor "
                    "will win"
                ],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "atrox is Aatrox",
            }],
            "unresolved_alternatives": [],
            "rationale": "one entity binding licenses repeated repairs",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=self._diagnostic(slots=slots),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"], clean,
        )
        self.assertEqual(
            reconstruction["counts"]["contextual_repair_count"], 4,
        )
        self.assertEqual(
            reconstruction["counts"]["resolution_repair_count"], 4,
        )

    def test_entity_resolution_repair_rejects_different_candidate(self):
        # A binding resolved to a different candidate must not license the
        # canonicalization repair, even on repeated surfaces.
        text = (
            "look at atrox atrox has 11 CS after this if atrox T is back "
            "here do you really think atrox with clo armor will win"
        )
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Darius",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        repair = {
            "original_text": "atrox",
            "replacement": "Aatrox",
            "repair_type": "ENTITY_RESOLUTION",
            "confidence": "HIGH",
            "evidence_quotes": [
                "look at atrox atrox has 11 CS after this if atrox T is "
                "back here do you really think atrox with clo armor will win"
            ],
            "rationale": "wrong candidate",
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": text.replace("atrox", "Aatrox"),
            "contextual_repairs": [repair, repair, repair, repair],
            "bindings": [{
                "slot": "champion_identities",
                "mention_text": "atrox",
                "resolved_candidate": "Darius",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": [
                    "look at atrox atrox has 11 CS after this if atrox T "
                    "is back here do you really think atrox with clo armor "
                    "will win"
                ],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "different candidate",
            }],
            "unresolved_alternatives": [],
            "rationale": "must fail closed",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                final_diagnostic=self._diagnostic(slots=slots),
                chat=CountingChat([
                    json.dumps(compact), json.dumps(compact),
                    json.dumps(compact), json.dumps(compact),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "is not licensed by a RESOLVED binding",
            caught.exception.attempts[0]["error"],
        )

    def test_pronoun_repair_not_licensed_by_entity_binding_elsewhere(self):
        # Pronoun resolution stays exact-span licensed: an entity binding
        # on a different span cannot license a pronoun rewrite.
        text = (
            "look at atrox atrox has 11 CS after this if he is back here "
            "do you really think atrox with clo armor will win"
        )
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Aatrox",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        evidence = (
            "look at atrox atrox has 11 CS after this if he is back here "
            "do you really think atrox with clo armor will win"
        )
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": text.replace(
                "atrox", "Aatrox",
            ).replace("he is back here", "Aatrox is back here"),
            "contextual_repairs": [
                {
                    "original_text": "he",
                    "replacement": "Aatrox",
                    "repair_type": "PRONOUN_RESOLUTION",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "pronoun rewrite",
                },
                {
                    "original_text": "atrox",
                    "replacement": "Aatrox",
                    "repair_type": "ENTITY_RESOLUTION",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "Aatrox canonicalization",
                },
                {
                    "original_text": "atrox",
                    "replacement": "Aatrox",
                    "repair_type": "ENTITY_RESOLUTION",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "Aatrox canonicalization",
                },
                {
                    "original_text": "atrox",
                    "replacement": "Aatrox",
                    "repair_type": "ENTITY_RESOLUTION",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "Aatrox canonicalization",
                },
            ],
            "bindings": [{
                "slot": "champion_identities",
                "mention_text": "atrox",
                "resolved_candidate": "Aatrox",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": [evidence],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "atrox is Aatrox",
            }],
            "unresolved_alternatives": [],
            "rationale": "must fail closed",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                final_diagnostic=self._diagnostic(slots=slots),
                chat=CountingChat([
                    json.dumps(compact), json.dumps(compact),
                    json.dumps(compact), json.dumps(compact),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "has no binding over the same mention span",
            caught.exception.attempts[0]["error"],
        )

    def test_whitespace_repair_normalizes_nbsp_and_seals_exact_slice(self):
        # Live pool:AOxq: Bronze uses NBSP inside "[<NBSP>__<NBSP>]"; the
        # provider can only write regular spaces.  A contextual WHITESPACE
        # proposal with raw original_text == replacement is mapped through
        # surface normalization to the exact NBSP Bronze slice; the sealed
        # repair stores the NBSP slice and the raw compact is preserved.
        bronze = (
            "she heals more than [\u00a0__\u00a0] window and "
            "[\u00a0__\u00a0] is fine"
        )
        clean = (
            "she heals more than [ __ ] window and [ __ ] is fine"
        )
        selected = _selected(bronze, champion="Lux")
        context = self._context(bronze, bronze, previous=0)
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": [
                {
                    "original_text": "[ __ ]",
                    "replacement": "[ __ ]",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": [
                        "she heals more than [\u00a0__\u00a0] window",
                    ],
                    "rationale": "normalize non-breaking spaces",
                },
                {
                    "original_text": "[ __ ]",
                    "replacement": "[ __ ]",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": [
                        "she heals more than [\u00a0__\u00a0] window",
                    ],
                    "rationale": "normalize non-breaking spaces",
                },
            ],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "whitespace normalization",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=bronze,
            context=context,
            mechanical_cleaned_text=bronze,
            final_diagnostic=self._diagnostic(),
            chat=CountingChat([json.dumps(compact)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"], clean,
        )
        repairs = reconstruction["contextual_repairs"]
        self.assertEqual(len(repairs), 2)
        for repair in repairs:
            self.assertEqual(repair["repair_type"], "WHITESPACE")
            self.assertEqual(repair["original_text"], "[\u00a0__\u00a0]")
            self.assertEqual(repair["replacement"], "[ __ ]")
        self.assertEqual(
            reconstruction["raw_compact"]["contextual_repairs"][0][
                "original_text"
            ],
            "[ __ ]",
        )
        self.assertEqual(
            reconstruction["raw_compact"]["contextual_repairs"][1][
                "original_text"
            ],
            "[ __ ]",
        )

    def test_whitespace_true_noop_fails_closed(self):
        # When the exact regular-space quote already exists in Bronze, a
        # WHITESPACE original==replacement proposal is a true no-op and
        # must be rejected.
        bronze = "she heals more than [ __ ] window"
        selected = _selected(bronze, champion="Lux")
        context = self._context(bronze, bronze, previous=0)
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": bronze,
            "contextual_repairs": [{
                "original_text": "[ __ ]",
                "replacement": "[ __ ]",
                "repair_type": "WHITESPACE",
                "confidence": "HIGH",
                "evidence_quotes": ["she heals more than [ __ ] window"],
                "rationale": "no-op",
            }],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "must fail closed",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=bronze,
                context=context,
                mechanical_cleaned_text=bronze,
                final_diagnostic=self._diagnostic(),
                chat=CountingChat([
                    json.dumps(compact), json.dumps(compact),
                    json.dumps(compact), json.dumps(compact),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "true no-op",
            caught.exception.attempts[0]["error"],
        )

    def test_reconstruction_diff_suggestion_full_word_and_second_response_succeeds(self):
        # Live pool:GKci: the char-level diff 'e' -> 'E' must expand to the
        # full word with a concrete suggestion (original_text="exhaust"
        # replacement="Exhaust") so the corrected response returns the
        # explicit capitalization repair.
        text = "Play with Flash because like you're going exhaust Smite"
        selected = _selected(text, champion="Lux")
        context = self._context(text, text, previous=0)
        clean = text.replace("exhaust", "Exhaust")
        good = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": [{
                "original_text": "exhaust",
                "replacement": "Exhaust",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "evidence_quotes": [text],
                "rationale": "spell name capitalization",
            }],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "capitalize Exhaust",
        }
        bad = dict(good)
        bad["contextual_repairs"] = []
        captured: dict[str, Any] = {}

        def chat(system: str, user: str) -> str:
            payload = json.loads(user)
            if payload.get("task") == "reconstruction_correction":
                captured["system"] = system
                captured["validator_error"] = payload["validator_error"]
                return json.dumps(good)
            return json.dumps(bad)

        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=self._diagnostic(),
            chat=chat,
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"], clean,
        )
        self.assertEqual(len(reconstruction["attempts"]), 2)
        self.assertIn(
            'suggest original_text="exhaust" replacement="Exhaust"',
            captured["validator_error"],
        )
        self.assertIn("full_word", captured["validator_error"])
        self.assertIn(
            "diagnostic suggestions",
            captured["system"],
        )

    def test_mechanical_diff_feedback_feeds_correction_prompt_and_succeeds(self):
        selected = _selected("he used W.")
        bad = _compact_mechanical_raw(
            selected,
            clean_text="He used W. now",
            rationale="unrepresented trailing edit",
        )
        good = _mechanical_raw(
            selected,
            repairs=[{
                "original_text": "he",
                "replacement": "He",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "rationale": "sentence start",
            }],
        )
        captured: dict[str, Any] = {}

        def chat(system: str, user: str) -> str:
            payload = json.loads(user)
            if payload.get("task") == "mechanical_cleanup_correction":
                captured["system"] = system
                captured["validator_error"] = payload["validator_error"]
                return good
            return bad

        result = run_mechanical_cleanup(
            selected,
            chat=chat,
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["mechanical_cleaned_text"], "He used W.")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["status"], "FAILED")
        self.assertIn("ordered_non_equal_changes", captured["validator_error"])
        self.assertIn("applied=", captured["validator_error"])
        self.assertIn("requested=", captured["validator_error"])
        self.assertIn("replace", captured["validator_error"])
        self.assertIn("insert", captured["validator_error"])
        self.assertIn("diff", captured["system"].casefold())
        self.assertIn(
            "full non-empty replacement span",
            captured["system"],
        )

    def test_reconstruction_diff_feedback_feeds_correction_prompt_and_succeeds(self):
        text = "Viktor is the player. He hit R. The replay shows it."
        target = "He hit R."
        start = text.index(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=2,
            radius_label="r5",
        )
        slots = _slots(decision="SUFFICIENT", champion="Viktor")
        slots["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "Viktor",
                "confidence": "HIGH",
                "evidence_spans": [],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        diagnostic = self._diagnostic(champion="Viktor", slots=slots)
        good = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "Viktor hit R.",
            "contextual_repairs": [{
                "original_text": "He",
                "replacement": "Viktor",
                "repair_type": "PRONOUN_RESOLUTION",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "rationale": "context names Viktor as the player",
            }],
            "bindings": [{
                "slot": "pronouns",
                "mention_text": "He",
                "resolved_candidate": "Viktor",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "context names Viktor as the player",
            }],
            "unresolved_alternatives": [],
            "rationale": "context names Viktor as the player",
        }
        bad = dict(good)
        bad["clean_target_transcript"] = "Viktor hit R. extra"
        captured: dict[str, Any] = {}

        def chat(system: str, user: str) -> str:
            payload = json.loads(user)
            if payload.get("task") == "reconstruction_correction":
                captured["system"] = system
                captured["validator_error"] = payload["validator_error"]
                return json.dumps(good)
            return json.dumps(bad)

        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=diagnostic,
            chat=chat,
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"], "Viktor hit R.",
        )
        self.assertEqual(len(reconstruction["attempts"]), 2)
        self.assertIn(
            "ordered_non_equal_changes", captured["validator_error"],
        )
        self.assertIn("applied=", captured["validator_error"])
        self.assertIn("requested=", captured["validator_error"])
        self.assertIn("insert", captured["validator_error"])
        self.assertIn("diff", captured["system"].casefold())
        self.assertIn(
            "full non-empty replacement span",
            captured["system"],
        )

    def test_mechanical_diff_feedback_exhaustion_remains_fail_closed(self):
        selected = _selected("he used W.")
        bad = _compact_mechanical_raw(
            selected,
            clean_text="He used W. now",
            rationale="unrepresented trailing edit",
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([bad, bad, bad, bad]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertEqual(
            len(caught.exception.attempts), MECHANICAL_MAX_CORRECTIONS + 1,
        )
        self.assertIn(
            "ordered_non_equal_changes",
            caught.exception.attempts[0]["error"],
        )
        self.assertIn(
            "ordered_non_equal_changes",
            caught.exception.attempts[2]["error"],
        )


class Phase2KGlobalAssignmentContractTests(unittest.TestCase):
    """Focused regression coverage for the Phase 2K bounded global repair
    assignment, whitespace-only WHITESPACE binding, narrow NONE-sentinel
    omission, composed reference licensing, and repeated-correction
    contracts."""

    def _run(
        self,
        bronze: str,
        clean: str,
        *,
        repairs: list[dict[str, Any]] | None = None,
        bindings: list[dict[str, Any]] | None = None,
        slots: Mapping[str, Any] | None = None,
        chat: Any = None,
    ) -> dict[str, Any]:
        selected = _selected(bronze, champion="Lux")
        context = retrieve_context(
            bronze,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=0,
            target_end=len(bronze),
            bronze_text=bronze,
            previous_segments=0,
            following_segments=0,
            radius_label="r5",
        )
        diagnostic_slots = _slots(decision="SUFFICIENT")
        if slots:
            diagnostic_slots.update(slots)
        final_diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": diagnostic_slots,
                    "metadata_conflicts": [],
                },
            },
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": repairs or [],
            "bindings": bindings or [],
            "unresolved_alternatives": [],
            "rationale": "test",
        }
        raw = json.dumps(compact)
        if chat is None:
            chat = CountingChat([raw, raw, raw, raw])
        return run_reconstruction(
            selected,
            transcript=bronze,
            context=context,
            mechanical_cleaned_text=bronze,
            final_diagnostic=final_diagnostic,
            chat=chat,
            config_hash=canonical_sha256({"v": 1}),
        )

    def _binding(
        self,
        *,
        slot: str,
        mention: str,
        candidate: str,
        status: str = "RESOLVED",
    ) -> dict[str, Any]:
        return {
            "slot": slot,
            "mention_text": mention,
            "resolved_candidate": candidate,
            "resolved_status": status,
            "confidence": "HIGH",
            "evidence_quotes": [],
            "alternatives": [],
            "metadata_contributed": False,
            "rationale": "test binding",
        }

    def test_whitespace_only_nbsp_binds_left_to_right_with_exact_evidence(self):
        bronze = "she heals more than [\u00a0__\u00a0] window and it is fine"
        clean = "she heals more than [ __ ] window and it is fine"
        evidence = "she heals more than [\u00a0__\u00a0] window"
        reconstruction = self._run(
            bronze,
            clean,
            repairs=[
                {
                    "original_text": "\u00a0",
                    "replacement": " ",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "normalize NBSP",
                },
                {
                    "original_text": "\u00a0",
                    "replacement": " ",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "normalize NBSP",
                },
            ],
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"], clean,
        )
        repairs = reconstruction["contextual_repairs"]
        self.assertEqual(len(repairs), 2)
        nbsp_positions = [
            index for index, char in enumerate(bronze) if char == "\u00a0"
        ]
        for repair, expected_start in zip(repairs, nbsp_positions):
            self.assertEqual(repair["repair_type"], "WHITESPACE")
            self.assertEqual(repair["original_text"], "\u00a0")
            self.assertEqual(repair["replacement"], " ")
            self.assertEqual(repair["target_local_start"], expected_start)
            self.assertEqual(repair["target_local_end"], expected_start + 1)
            self.assertEqual(repair["evidence_spans"][0]["text"], evidence)

    def test_whitespace_only_true_noop_and_nonwhitespace_rejected(self):
        # A whitespace-only proposal whose replacement equals its original
        # has no valid Bronze span and is a true no-op.
        bronze = "heals"
        noop = {
            "original_text": "\u00a0",
            "replacement": "\u00a0",
            "repair_type": "WHITESPACE",
            "confidence": "HIGH",
            "evidence_quotes": [bronze],
            "rationale": "true no-op",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(bronze, bronze, repairs=[noop])
        self.assertIn(
            "true no-op",
            caught.exception.attempts[0]["error"],
        )
        # A WHITESPACE repair that changes non-whitespace characters is
        # never bindable.
        bronze = "she heals more than [\u00a0__\u00a0] window"
        non_whitespace = {
            "original_text": "\u00a0",
            "replacement": "x",
            "repair_type": "WHITESPACE",
            "confidence": "HIGH",
            "evidence_quotes": [bronze],
            "rationale": "not a whitespace change",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                bronze.replace("\u00a0", "x"),
                repairs=[non_whitespace],
            )
        self.assertIn(
            "cannot be bound to any exact, surface-normalized, or "
            "explicit-whitespace Bronze span",
            caught.exception.attempts[0]["error"],
        )

    def test_whitespace_repair_requires_exact_evidence(self):
        # Live pool:AOxq old correction payload: the evidence quote
        # "[ <NBSP>__<NBSP> ]" with regular spaces around the NBSPs is
        # genuinely absent from the supplied context (the Bronze slice is
        # "[<NBSP>__<NBSP>]"), so it must stay strict and fail with
        # actionable quote guidance rather than being coerced.
        bronze = "she heals more than [\u00a0__\u00a0] window"
        clean = bronze.replace("\u00a0", " ")
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": [
                {
                    "original_text": "\u00a0",
                    "replacement": " ",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": ["[ \u00a0__\u00a0 ]"],
                    "rationale": "normalize NBSP",
                },
                {
                    "original_text": "\u00a0",
                    "replacement": " ",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": ["[ \u00a0__\u00a0 ]"],
                    "rationale": "normalize NBSP",
                },
            ],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "must stay strict",
        }
        raw = json.dumps(compact)
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                clean,
                chat=CountingChat([raw, raw, raw, raw]),
            )
        self.assertIn(
            "absent from the supplied source",
            caught.exception.attempts[0]["error"],
        )
        self.assertIn(
            "Quote an exact contiguous context span",
            caught.exception.attempts[0]["error"],
        )

    def test_global_assignment_picks_intended_e_in_uses_e(self):
        bronze = "Viktor uses e and then e again."
        clean = "Viktor uses E and then e again."
        reconstruction = self._run(
            bronze,
            clean,
            repairs=[{
                "original_text": "e",
                "replacement": "E",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "evidence_quotes": [bronze],
                "rationale": "spell E",
            }],
        )
        repair = reconstruction["contextual_repairs"][0]
        intended = bronze.index("uses e") + len("uses ")
        self.assertEqual(repair["target_local_start"], intended)
        self.assertEqual(repair["target_local_end"], intended + 1)
        self.assertEqual(
            bronze[repair["target_local_start"]:repair["target_local_end"]],
            "e",
        )

    def test_global_assignment_requires_every_proposal(self):
        # Two identical e->E proposals against a single e occurrence must
        # fail closed (duplicate guidance), never silently drop one.
        bronze = "uses e"
        repair = {
            "original_text": "e",
            "replacement": "E",
            "repair_type": "CAPITALIZATION",
            "confidence": "HIGH",
            "evidence_quotes": [bronze],
            "rationale": "x",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                "uses E",
                repairs=[repair, dict(repair)],
            )
        self.assertIn(
            "do not drop or duplicate repairs",
            caught.exception.attempts[0]["error"],
        )

    def test_ambiguous_global_assignment_fails_closed(self):
        # Two proposals with identical original/replacement but different
        # repair types form different groups with interchangeable spans:
        # multiple selections reproduce the clean transcript, so the
        # assignment must fail closed as ambiguous.
        bronze = "e e"
        repair = {
            "original_text": "e",
            "replacement": "E",
            "confidence": "HIGH",
            "evidence_quotes": [bronze],
            "rationale": "x",
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                "E E",
                repairs=[
                    {**repair, "repair_type": "CAPITALIZATION"},
                    {**repair, "repair_type": "DOMAIN_SPELLING"},
                ],
            )
        self.assertIn(
            "ambiguous",
            caught.exception.attempts[0]["error"],
        )

    def test_overlapping_repairs_fail_with_merge_guidance(self):
        bronze = "He hit R."
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                "Viktor attacked R.",
                repairs=[
                    {
                        "original_text": "He hit",
                        "replacement": "Viktor attacked",
                        "repair_type": "PRONOUN_RESOLUTION",
                        "confidence": "HIGH",
                        "evidence_quotes": [bronze],
                        "rationale": "overlap a",
                    },
                    {
                        "original_text": "hit R.",
                        "replacement": "attacked R.",
                        "repair_type": "CONTEXTUAL_ASR",
                        "confidence": "HIGH",
                        "evidence_quotes": [bronze],
                        "rationale": "overlap b",
                    },
                ],
            )
        self.assertIn(
            "merge overlapping edits into one exact Bronze span",
            caught.exception.attempts[0]["error"],
        )
        self.assertIn(
            "do not drop or duplicate repairs",
            caught.exception.attempts[0]["error"],
        )

    def test_redundant_unbindable_repair_fails_with_remove_guidance(self):
        # Live pool:GKci: "Eyeball" -> "eyeball" cannot bind when Bronze
        # already holds the lowercase form and clean matches Bronze; the
        # error must tell the provider to remove the entire repair.
        bronze = "we take eyeball"
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                bronze,
                repairs=[{
                    "original_text": "Eyeball",
                    "replacement": "eyeball",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "evidence_quotes": [bronze],
                    "rationale": "redundant capitalization",
                }],
            )
        self.assertIn(
            "remove this entire repair",
            caught.exception.attempts[0]["error"],
        )

    def test_equation_rhs_licensing_with_mention_lhs_match(self):
        # A diagnostic candidate "she/her = enemy mid laner; you/your =
        # Veigar player" licenses mention "she" through the equation LHS and
        # the resolved referent through the principal/champion candidate.
        bronze = "she uses e right we go"
        clean = "she uses E right, we go"
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "enemy mid laner",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
            "pronouns": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": (
                        "she/her = enemy mid laner; you/your = Veigar player"
                    ),
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        reconstruction = self._run(
            bronze,
            clean,
            repairs=[
                {
                    "original_text": "e",
                    "replacement": "E",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "evidence_quotes": [bronze],
                    "rationale": "spell E",
                },
                {
                    "original_text": "right we",
                    "replacement": "right, we",
                    "repair_type": "PUNCTUATION",
                    "confidence": "HIGH",
                    "evidence_quotes": [bronze],
                    "rationale": "comma",
                },
            ],
            bindings=[self._binding(
                slot="pronouns",
                mention="she",
                candidate="enemy mid laner",
            )],
            slots=slots,
        )
        binding = reconstruction["bindings"][0]
        self.assertEqual(binding["resolved_candidate"], "enemy mid laner")
        self.assertEqual(binding["mention"]["text"], "she")

    def test_composed_pronoun_licensing_and_rejections(self):
        # Live pool:oA4C-style: pronoun/reference bindings validate when the
        # mention is licensed by its own slot and the referent is licensed
        # by an entity/principal candidate.
        bronze = "Lucian and rakan fight. he hits him. raan is strong."
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [
                    {
                        "candidate": "Lucian",
                        "confidence": "HIGH",
                        "evidence_spans": [],
                    },
                    {
                        "candidate": "rakan",
                        "confidence": "HIGH",
                        "evidence_spans": [],
                    },
                ],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
            "pronouns": {
                "status": "RESOLVED",
                "candidates": [
                    {
                        "candidate": "he",
                        "confidence": "HIGH",
                        "evidence_spans": [],
                    },
                    {
                        "candidate": "him",
                        "confidence": "HIGH",
                        "evidence_spans": [],
                    },
                ],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
            "unresolved_asr": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "raan",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        reconstruction = self._run(
            bronze,
            bronze,
            bindings=[
                self._binding(
                    slot="pronouns", mention="he", candidate="Lucian",
                ),
                self._binding(
                    slot="pronouns", mention="him", candidate="rakan",
                ),
                self._binding(
                    slot="unresolved_asr", mention="raan", candidate="rakan",
                ),
            ],
            slots=slots,
        )
        self.assertEqual(len(reconstruction["bindings"]), 3)

        # Own-slot mention license missing: candidate "they" cannot license
        # the "he" mention, so composed licensing must reject.
        missing_mention = dict(slots)
        missing_mention["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "they",
                "confidence": "HIGH",
                "evidence_spans": [],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                bronze,
                bindings=[self._binding(
                    slot="pronouns", mention="he", candidate="Lucian",
                )],
                slots=missing_mention,
            )
        self.assertIn(
            "not licensed",
            caught.exception.attempts[0]["error"],
        )

        # Referent license missing: entity slot only knows Lucian, so
        # "he" -> "Rakan" has no licensed referent.
        missing_referent = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Lucian",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
            "pronouns": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "he",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                bronze,
                bindings=[self._binding(
                    slot="pronouns", mention="he", candidate="Rakan",
                )],
                slots=missing_referent,
            )
        self.assertIn(
            "not licensed",
            caught.exception.attempts[0]["error"],
        )

    def test_n2_we_binding_unlicensed_but_span_assignment_proven(self):
        # Live pool:n2: the e->E repair must bind the intended "uses e"
        # occurrence independently of the invalid semantic "we" binding.
        bronze = "she uses e right we go"
        clean = "she uses E right, we go"
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "enemy mid laner",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
            "pronouns": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": (
                        "she/her = enemy mid laner; you/your = Veigar player"
                    ),
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        repairs = [
            {
                "original_text": "e",
                "replacement": "E",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "evidence_quotes": [bronze],
                "rationale": "spell E",
            },
            {
                "original_text": "right we",
                "replacement": "right, we",
                "repair_type": "PUNCTUATION",
                "confidence": "HIGH",
                "evidence_quotes": [bronze],
                "rationale": "comma",
            },
        ]
        reconstruction = self._run(
            bronze,
            clean,
            repairs=repairs,
            bindings=[self._binding(
                slot="pronouns",
                mention="she",
                candidate="enemy mid laner",
            )],
            slots=slots,
        )
        e_repairs = [
            repair for repair in reconstruction["contextual_repairs"]
            if repair["original_text"] == "e"
        ]
        self.assertEqual(len(e_repairs), 1)
        intended = bronze.index("uses e") + len("uses ")
        self.assertEqual(e_repairs[0]["target_local_start"], intended)
        self.assertEqual(
            bronze[e_repairs[0]["target_local_start"]:
                   e_repairs[0]["target_local_end"]],
            "e",
        )

        # The same payload with the unlicensed "we" -> Veigar player binding
        # must be rejected: "we" is not licensed by the equation LHS.
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                clean,
                repairs=repairs,
                bindings=[
                    self._binding(
                        slot="pronouns",
                        mention="she",
                        candidate="enemy mid laner",
                    ),
                    self._binding(
                        slot="pronouns",
                        mention="we",
                        candidate="Veigar player",
                    ),
                ],
                slots=slots,
            )
        self.assertIn(
            "not licensed",
            caught.exception.attempts[0]["error"],
        )

    def test_none_sentinel_omission_and_raw_compact_audit(self):
        bronze = "He hit R."
        reconstruction = self._run(
            bronze,
            bronze,
            bindings=[
                self._binding(
                    slot="pronouns",
                    mention="NONE",
                    candidate="NONE",
                ),
                self._binding(
                    slot="pronouns",
                    mention="He",
                    candidate="NONE",
                ),
            ],
        )
        self.assertEqual(reconstruction["bindings"], [])
        self.assertEqual(reconstruction["omitted_binding_count"], 2)
        self.assertEqual(len(reconstruction["raw_compact"]["bindings"]), 2)
        _validate_reconstruction_raw_compact(
            reconstruction["raw_compact"],
            reconstruction=reconstruction,
            label="unit test",
        )

    def test_source_absent_binding_rejected_with_removal_guidance(self):
        # Live pool:jDx: "your queue" is absent from Bronze and must be
        # rejected with explicit remove-entire-binding guidance.
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                "He hit R.",
                "He hit R.",
                bindings=[self._binding(
                    slot="ability_ownership",
                    mention="your queue",
                    candidate="Mel",
                )],
            )
        self.assertIn(
            "absent from the supplied source",
            caught.exception.attempts[0]["error"],
        )
        self.assertIn(
            "Remove this entire binding",
            caught.exception.attempts[0]["error"],
        )

    def test_repeated_identical_correction_prompt_is_materially_stronger(self):
        text = "Viktor is the player. He hit R. The replay shows it."
        target = "He hit R."
        start = text.index(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r5",
        )
        slots = _slots(decision="SUFFICIENT", champion="Viktor")
        slots["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "Viktor",
                "confidence": "HIGH",
                "evidence_spans": [],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        final_diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {"slots": slots, "metadata_conflicts": []},
            },
        }
        bad = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "Viktor hit R. extra",
            "contextual_repairs": [],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "bad",
        }
        bad_raw = json.dumps(bad)
        captured: list[dict[str, Any]] = []

        def chat(system: str, user: str) -> str:
            payload = json.loads(user)
            if payload.get("task") == "reconstruction_correction":
                captured.append({"system": system})
            return bad_raw

        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=target,
                final_diagnostic=final_diagnostic,
                chat=chat,
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertEqual(
            len(caught.exception.attempts), RECONSTRUCTION_MAX_CORRECTIONS + 1,
        )
        self.assertEqual(len(captured), RECONSTRUCTION_MAX_CORRECTIONS)
        first, second, third = captured
        self.assertNotIn("byte-identical", first["system"])
        self.assertIn("byte-identical", second["system"])
        self.assertIn("materially different", second["system"])
        self.assertIn("byte-identical", third["system"])
        self.assertIn(
            "Identical repeat responses are rejected",
            second["system"],
        )
        self.assertNotEqual(
            caught.exception.attempts[1]["model_call"]["prompt_hash"],
            caught.exception.attempts[2]["model_call"]["prompt_hash"],
        )
        self.assertEqual(
            caught.exception.attempts[1]["attempt_kind"], "correction:1",
        )
        self.assertEqual(
            caught.exception.attempts[2]["attempt_kind"], "correction:2",
        )
        self.assertEqual(
            caught.exception.attempts[3]["attempt_kind"], "correction:3",
        )
        self.assertEqual(
            [attempt["status"] for attempt in caught.exception.attempts],
            ["FAILED", "FAILED", "FAILED", "FAILED"],
        )


class Phase2KFinalThreeLiveFixTests(unittest.TestCase):
    """Focused regression tests for the final three live #6 failures:

    - AOxq: absent evidence quotes get a bounded exact-slice suggestion
      (diagnostic only; the malformed quote still fails closed).
    - jDx: source-absent binding mentions aggregate into one bounded error.
    - wTx: narrow exact-span composite ENTITY_RESOLUTION license.
    """

    def _run(
        self,
        bronze: str,
        clean: str,
        *,
        repairs: list[dict[str, Any]] | None = None,
        bindings: list[dict[str, Any]] | None = None,
        slots: Mapping[str, Any] | None = None,
        chat: Any = None,
    ) -> dict[str, Any]:
        selected = _selected(bronze, champion="Lux")
        context = retrieve_context(
            bronze,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=0,
            target_end=len(bronze),
            bronze_text=bronze,
            previous_segments=0,
            following_segments=0,
            radius_label="r5",
        )
        diagnostic_slots = _slots(decision="SUFFICIENT")
        if slots:
            diagnostic_slots.update(slots)
        final_diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": diagnostic_slots,
                    "metadata_conflicts": [],
                },
            },
        }
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": repairs or [],
            "bindings": bindings or [],
            "unresolved_alternatives": [],
            "rationale": "test",
        }
        raw = json.dumps(compact)
        if chat is None:
            chat = CountingChat([raw, raw, raw, raw])
        return run_reconstruction(
            selected,
            transcript=bronze,
            context=context,
            mechanical_cleaned_text=bronze,
            final_diagnostic=final_diagnostic,
            chat=chat,
            config_hash=canonical_sha256({"v": 1}),
        )

    def _binding(
        self,
        *,
        slot: str,
        mention: str,
        candidate: str,
        status: str = "RESOLVED",
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "slot": slot,
            "mention_text": mention,
            "resolved_candidate": candidate,
            "resolved_status": status,
            "confidence": "HIGH",
            "evidence_quotes": evidence or [],
            "alternatives": [],
            "metadata_contributed": False,
            "rationale": "test binding",
        }

    def _repair(
        self,
        *,
        original: str,
        replacement: str,
        repair_type: str,
        evidence: list[str],
    ) -> dict[str, Any]:
        return {
            "original_text": original,
            "replacement": replacement,
            "repair_type": repair_type,
            "confidence": "HIGH",
            "evidence_quotes": evidence,
            "rationale": "test repair",
        }

    def test_absent_evidence_quote_suggests_exact_slice_and_stays_strict(self):
        # Live pool:AOxq shape: the provider quoted "[ <NBSP>__<NBSP> ]"
        # (regular spaces around the NBSPs) while the exact source slices
        # are "[<NBSP>__<NBSP>]".  The whitespace skeleton maps to exactly
        # one distinct exact slice, so the error suggests it verbatim, but
        # the malformed quote is never accepted or coerced.
        bronze = "she heals more than [\u00a0__\u00a0] and [\u00a0__\u00a0] window"
        clean = bronze.replace("\u00a0", " ")
        malformed = "[ \u00a0__\u00a0 ]"
        repairs = [
            self._repair(
                original="\u00a0",
                replacement=" ",
                repair_type="WHITESPACE",
                evidence=[malformed],
            ),
            self._repair(
                original="\u00a0",
                replacement=" ",
                repair_type="WHITESPACE",
                evidence=[malformed],
            ),
            self._repair(
                original="\u00a0",
                replacement=" ",
                repair_type="WHITESPACE",
                evidence=[malformed],
            ),
            self._repair(
                original="\u00a0",
                replacement=" ",
                repair_type="WHITESPACE",
                evidence=[malformed],
            ),
        ]
        raw = json.dumps({
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": repairs,
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "test",
        })
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(bronze, clean, chat=CountingChat([raw, raw, raw, raw]))
        error = caught.exception.attempts[0]["error"]
        self.assertIn("absent from the supplied source", error)
        self.assertIn("Suggested exact replacement evidence_quote", error)
        self.assertIn(repr("[\u00a0__\u00a0]"), error)
        self.assertIn("Quote it verbatim", error)
        self.assertIn("repeat it once per intended occurrence", error)
        self.assertIn(
            "Quote an exact contiguous context span", error,
        )

    def test_absent_evidence_quote_no_suggestion_when_ambiguous_or_no_skeleton(self):
        # Two distinct source slices share the same whitespace skeleton:
        # the intended exact slice is ambiguous, so no suggestion is
        # emitted and the response still fails closed.
        bronze = "she heals more than [\u00a0__\u00a0] and [ __ ] window"
        clean = bronze.replace("\u00a0", " ")
        repairs = [
            self._repair(
                original="\u00a0",
                replacement=" ",
                repair_type="WHITESPACE",
                evidence=["[ \u00a0__\u00a0 ]"],
            ),
            self._repair(
                original="\u00a0",
                replacement=" ",
                repair_type="WHITESPACE",
                evidence=["[ \u00a0__\u00a0 ]"],
            ),
        ]
        raw = json.dumps({
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": repairs,
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "test",
        })
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(bronze, clean, chat=CountingChat([raw, raw, raw, raw]))
        error = caught.exception.attempts[0]["error"]
        self.assertIn("absent from the supplied source", error)
        self.assertNotIn("Suggested exact replacement evidence_quote", error)

        # No whitespace skeleton match at all: no suggestion.
        repairs = [
            self._repair(
                original="\u00a0",
                replacement=" ",
                repair_type="WHITESPACE",
                evidence=["[ __ ]"],
            ),
        ]
        raw = json.dumps({
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "plain text with a bracket mask",
            "contextual_repairs": repairs,
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "test",
        })
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                "plain text with\u00a0a bracket mask",
                "plain text with a bracket mask",
                chat=CountingChat([raw, raw, raw, raw]),
            )
        error = caught.exception.attempts[0]["error"]
        self.assertIn("absent from the supplied source", error)
        self.assertNotIn("Suggested exact replacement evidence_quote", error)

    def test_source_absent_bindings_aggregated_into_one_error(self):
        # Live pool:jDx shape: every unique absent binding mention is
        # reported in one bounded error with explicit removal of each
        # listed binding, while a present binding is unaffected.
        bronze = "He hit R."
        bindings = [
            self._binding(
                slot="pronouns", mention="He", candidate="He",
            ),
            self._binding(
                slot="champion_identities",
                mention="Synindra",
                candidate="Synindra",
            ),
            self._binding(
                slot="ability_ownership",
                mention="your Q",
                candidate="Mel's Q",
            ),
            self._binding(
                slot="pronouns", mention="him", candidate="Synindra",
            ),
            self._binding(
                slot="condition",
                mention="if you leave now",
                candidate="if you leave now",
            ),
        ]
        raw = json.dumps({
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": bronze,
            "contextual_repairs": [],
            "bindings": bindings,
            "unresolved_alternatives": [],
            "rationale": "test",
        })
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                bronze,
                bindings=bindings,
                chat=CountingChat([raw, raw, raw, raw]),
            )
        error = caught.exception.attempts[0]["error"]
        for absent in ("Synindra", "your Q", "him", "if you leave now"):
            self.assertIn(absent, error)
        self.assertIn("Remove each of these entire bindings", error)
        self.assertIn("never normalized", error)
        self.assertNotIn("'He'", error)

    def test_single_source_absent_binding_keeps_existing_guidance(self):
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                "He hit R.",
                "He hit R.",
                bindings=[self._binding(
                    slot="ability_ownership",
                    mention="your queue",
                    candidate="Mel",
                )],
            )
        self.assertIn(
            "absent from the supplied source",
            caught.exception.attempts[0]["error"],
        )
        self.assertIn(
            "Remove this entire binding",
            caught.exception.attempts[0]["error"],
        )

    def test_aggregate_absent_error_does_not_affect_valid_grouped_bindings(self):
        bronze = "you are strong and you are fine."
        slots = {
            "pronouns": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "You",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
            "principal_actors": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "You",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        reconstruction = self._run(
            bronze,
            bronze,
            bindings=[
                self._binding(
                    slot="pronouns", mention="you", candidate="You",
                ),
                self._binding(
                    slot="principal_actors",
                    mention="you",
                    candidate="You",
                ),
            ],
            slots=slots,
        )
        self.assertEqual(len(reconstruction["bindings"]), 2)
        for binding in reconstruction["bindings"]:
            self.assertEqual(binding["mention"]["text"], "you")

    def test_exact_span_composite_entity_license_accepts_this_darus(self):
        # Live pool:wTx shape: exact-span "this darus" -> "this Darius"
        # with candidate Darius is licensed by the composite rule.
        bronze = "this darus will be strong."
        clean = "this Darius will be strong."
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Darius",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        reconstruction = self._run(
            bronze,
            clean,
            repairs=[self._repair(
                original="this darus",
                replacement="this Darius",
                repair_type="ENTITY_RESOLUTION",
                evidence=["this darus"],
            )],
            bindings=[self._binding(
                slot="champion_identities",
                mention="this darus",
                candidate="Darius",
                evidence=["this darus"],
            )],
            slots=slots,
        )
        repair = reconstruction["contextual_repairs"][0]
        self.assertEqual(repair["replacement"], "this Darius")
        self.assertEqual(
            reconstruction["clean_target_transcript"], clean,
        )

    def test_exact_span_composite_entity_license_rejects_broad_rewrites(self):
        cases = [
            # Broad rewrite: candidate Darius is present but surrounding
            # tokens are not preserved and extra words are added.
            {
                "clean": "Darius dominates lane will be strong.",
                "replacement": "Darius dominates lane",
                "candidate": "Darius",
            },
            # Substring-inside-token candidate is never a complete token.
            {
                "clean": "this Darius will be strong.",
                "replacement": "this Darius",
                "candidate": "Dari",
            },
            # Wrong candidate.
            {
                "clean": "this Darius will be strong.",
                "replacement": "this Darius",
                "candidate": "Aatrox",
            },
            # Changed surrounding word.
            {
                "clean": "the Darius will be strong.",
                "replacement": "the Darius",
                "candidate": "Darius",
            },
        ]
        for case in cases:
            with self.subTest(case=case):
                bronze = "this darus will be strong."
                clean = case["clean"]
                slots = {
                    "champion_identities": {
                        "status": "RESOLVED",
                        "candidates": [{
                            "candidate": case["candidate"],
                            "confidence": "HIGH",
                            "evidence_spans": [],
                        }],
                        "confidence": "HIGH",
                        "evidence_spans": [],
                    },
                }
                with self.assertRaises(ProviderCorrectionExhausted) as caught:
                    self._run(
                        bronze,
                        clean,
                        repairs=[self._repair(
                            original="this darus",
                            replacement=case["replacement"],
                            repair_type="ENTITY_RESOLUTION",
                            evidence=["this darus"],
                        )],
                        bindings=[self._binding(
                            slot="champion_identities",
                            mention="this darus",
                            candidate=case["candidate"],
                            evidence=["this darus"],
                        )],
                        slots=slots,
                    )
                error = caught.exception.attempts[0]["error"]
                self.assertIn(
                    "is not licensed by a RESOLVED binding", error,
                )
                self.assertIn("exact-span composite", error)

    def test_exact_span_composite_entity_license_requires_exact_mention_span(self):
        # The binding mention span ("darus") does not exactly equal the
        # repair span ("this darus"), so the composite license never fires.
        bronze = "this darus will be strong."
        clean = "this Darius will be strong."
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Darius",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                clean,
                repairs=[self._repair(
                    original="this darus",
                    replacement="this Darius",
                    repair_type="ENTITY_RESOLUTION",
                    evidence=["this darus"],
                )],
                bindings=[self._binding(
                    slot="champion_identities",
                    mention="darus",
                    candidate="Darius",
                    evidence=["this darus"],
                )],
                slots=slots,
            )
        error = caught.exception.attempts[0]["error"]
        self.assertIn("is not licensed by a RESOLVED binding", error)
        self.assertIn("exact-span composite", error)

    def test_entity_repair_failure_text_distinguishes_license_paths(self):
        bronze = "this darus will be strong."
        clean = "this Darius will be strong."
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Aatrox",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._run(
                bronze,
                clean,
                repairs=[self._repair(
                    original="this darus",
                    replacement="this Darius",
                    repair_type="ENTITY_RESOLUTION",
                    evidence=["this darus"],
                )],
                bindings=[self._binding(
                    slot="champion_identities",
                    mention="this darus",
                    candidate="Aatrox",
                    evidence=["this darus"],
                )],
                slots=slots,
            )
        error = caught.exception.attempts[0]["error"]
        self.assertIn(
            "full-surface/repeated canonicalization", error,
        )
        self.assertIn("exact-span composite", error)


class Phase2KPhase2KLiveV9FixTests(unittest.TestCase):
    """Regression tests for the four live #7 failures (AOxq/ZYIP/n2/w9).

    - AOxq: repair-anchored exact evidence binding accepts an exact quote
      that occurs in multiple context occurrences when exactly one exact
      occurrence contains the repair span.
    - ZYIP: FILLER deletions may use an empty replacement with a non-empty
      exact original_text and evidence; no other contextual repair type may.
    - n2: mechanical cleanup allows one initial + three corrections and
      keeps the exact live progression (overlap, missing cap+comma, missing
      comma, complete) inside four attempts.
    - w9: semantic polish allows one initial + three corrections, suggests
      the exact Bronze evidence quote for repaired text absent from Bronze,
      and accepts a RECONSTRUCTION_DERIVED fourth payload.
    """

    def _context(
        self,
        transcript: str,
        target: str,
        *,
        previous: int = 1,
        following: int = 0,
    ) -> dict[str, Any]:
        start = transcript.index(target)
        selected = _selected(transcript, target=target)
        return retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=previous,
            following_segments=following,
            radius_label="r5",
        )

    def _diagnostic(
        self,
        *,
        champion: str = "Lux",
        slots: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _slots(decision="SUFFICIENT", champion=champion)
        if slots:
            resolved.update(slots)
        return {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": resolved,
                    "metadata_conflicts": [],
                },
            },
        }

    def test_repair_anchored_duplicate_exact_quote_binds_containing_occurrence(
        self,
    ):
        # Live pool:AOxq correction:2 replay: the exact evidence quote
        # "[<NBSP>__<NBSP>]" occurs in both a neighboring segment and the
        # target segment.  Each NBSP->space WHITESPACE repair supplies the
        # quote once; the all-context occurrence-count rule would reject the
        # one-quote evidence list, but the repair-anchored rule accepts it
        # because exactly one exact occurrence contains each repair's
        # source-absolute span.
        neighbor = "So we should only accept the [\u00a0__\u00a0] window "
        target = (
            "I think flash is broken on Camille So I would go flash TP and "
            "[\u00a0__\u00a0] window"
        )
        transcript = neighbor + target
        selected = _selected(transcript, champion="Fiora", target=target)
        start = transcript.index(target)
        context = retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r5",
        )
        clean = (
            target
            .replace("Camille So", "Camille. So")
            .replace("\u00a0", " ")
        )
        raw = json.dumps({
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": [
                {
                    "original_text": "Camille So",
                    "replacement": "Camille. So",
                    "repair_type": "PUNCTUATION",
                    "confidence": "HIGH",
                    "evidence_quotes": [
                        "I think flash is broken on Camille So I would go "
                        "flash TP",
                    ],
                    "rationale": "missing period after Camille",
                },
                {
                    "original_text": "\u00a0",
                    "replacement": " ",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": ["[\u00a0__\u00a0]"],
                    "rationale": "NBSP to space",
                },
                {
                    "original_text": "\u00a0",
                    "replacement": " ",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": ["[\u00a0__\u00a0]"],
                    "rationale": "NBSP to space",
                },
            ],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "punctuation plus whitespace normalization",
        })
        reconstruction = run_reconstruction(
            selected,
            transcript=transcript,
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=self._diagnostic(champion="Fiora"),
            chat=CountingChat([raw]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(reconstruction["clean_target_transcript"], clean)
        self.assertEqual(len(reconstruction["attempts"]), 1)
        target_mask_start = target.index("[\u00a0__\u00a0]")
        target_mask_end = target_mask_start + len("[\u00a0__\u00a0]")
        whitespace = [
            repair for repair in reconstruction["contextual_repairs"]
            if repair["repair_type"] == "WHITESPACE"
        ]
        self.assertEqual(len(whitespace), 2)
        for repair in whitespace:
            self.assertEqual(
                repair["evidence_spans"][0]["text"], "[\u00a0__\u00a0]",
            )
            self.assertEqual(
                repair["evidence_spans"][0]["source_absolute_start"],
                start + target_mask_start,
            )
            self.assertEqual(
                repair["evidence_spans"][0]["source_absolute_end"],
                start + target_mask_end,
            )

    def test_repair_anchored_zero_containing_occurrence_fails_closed(self):
        # The exact quote occurs in a neighbor and in the target, but NO
        # occurrence contains the repair span (the repair edits the word
        # "Window"): the existing fail-closed evidence error is preserved.
        neighbor = "mask [\u00a0__\u00a0] then "
        target = "Window [\u00a0__\u00a0]"
        transcript = neighbor + target
        selected = _selected(transcript, champion="Lux", target=target)
        start = transcript.index(target)
        context = retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r5",
        )
        compact = {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": target.replace("Window", "window"),
            "contextual_repairs": [{
                "original_text": "Window",
                "replacement": "window",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "evidence_quotes": ["[\u00a0__\u00a0]"],
                "rationale": "no containing occurrence",
            }],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "must stay fail-closed",
        }
        raw = json.dumps(compact)
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=transcript,
                context=context,
                mechanical_cleaned_text=target,
                final_diagnostic=self._diagnostic(),
                chat=CountingChat([raw, raw, raw, raw]),
                config_hash=canonical_sha256({"v": 1}),
            )
        error = caught.exception.attempts[0]["error"]
        self.assertIn(
            "absent from the supplied source", error,
        )

    def test_repair_anchored_multiple_containing_occurrences_fail_closed(
        self,
    ):
        # Defensive branch: even when crafted overlapping exact occurrences
        # both contain the repair span, the anchor rescue never fires and
        # the deterministic ambiguity error is preserved.
        from unittest import mock

        import pipeline.phase2k_contextual_reconstruction as p2k

        context = {
            "segments": [{
                "segment_id": "seg:test:00001",
                "text": "abab",
                "source_absolute_start": 0,
            }],
        }
        with mock.patch.object(
            p2k, "_occurrences", side_effect=lambda text, needle: (
                [0, 1] if needle == "abc" else []
            ),
        ):
            with self.assertRaises(ValueError) as caught:
                _bind_evidence_quotes(
                    ["abc"],
                    context=context,
                    label="contextual repair proposal 1",
                    allow_normalized=True,
                    anchor_span=(1, 3),
                )
        self.assertIn(
            "absent from the supplied source", str(caught.exception),
        )

    def test_repair_anchored_malformed_quote_still_fails_with_suggestion(
        self,
    ):
        # Live pool:AOxq correction:1: the malformed "[ <NBSP>__<NBSP> ]"
        # quote stays strict and the error suggests the exact Bronze slice
        # verbatim; it is never normalized or coerced.
        neighbor = "So we should only accept the [\u00a0__\u00a0] window "
        target = (
            "I think flash is broken on Camille So I would go flash TP and "
            "[\u00a0__\u00a0] window"
        )
        transcript = neighbor + target
        selected = _selected(transcript, champion="Fiora", target=target)
        start = transcript.index(target)
        context = retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r5",
        )
        clean = (
            target
            .replace("Camille So", "Camille. So")
            .replace("\u00a0", " ")
        )
        raw = json.dumps({
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": [
                {
                    "original_text": "Camille So",
                    "replacement": "Camille. So",
                    "repair_type": "PUNCTUATION",
                    "confidence": "HIGH",
                    "evidence_quotes": [
                        "I think flash is broken on Camille So I would go "
                        "flash TP",
                    ],
                    "rationale": "missing period after Camille",
                },
                {
                    "original_text": "\u00a0",
                    "replacement": " ",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": ["[ \u00a0__\u00a0 ]"],
                    "rationale": "malformed quote",
                },
                {
                    "original_text": "\u00a0",
                    "replacement": " ",
                    "repair_type": "WHITESPACE",
                    "confidence": "HIGH",
                    "evidence_quotes": ["[ \u00a0__\u00a0 ]"],
                    "rationale": "malformed quote",
                },
            ],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "punctuation plus whitespace normalization",
        })
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_reconstruction(
                selected,
                transcript=transcript,
                context=context,
                mechanical_cleaned_text=target,
                final_diagnostic=self._diagnostic(champion="Fiora"),
                chat=CountingChat([raw, raw, raw, raw]),
                config_hash=canonical_sha256({"v": 1}),
            )
        error = caught.exception.attempts[0]["error"]
        self.assertIn("absent from the supplied source", error)
        self.assertIn("Suggested exact replacement evidence_quote", error)
        self.assertIn(repr("[\u00a0__\u00a0]"), error)
        self.assertIn("Quote it verbatim", error)

    def test_filler_empty_deletion_validates_and_reproduces_clean(self):
        # Live pool:ZYIP correction:2 shape: FILLER may delete a filler
        # plus its leading whitespace with an empty replacement; the clean
        # text has no double spaces and the sealed repair/raw compact stay
        # intact.
        target = (
            "yeah and now you don't win but the one way that you can win "
            "is if you do get push so if there's an angle to get push then "
            "you could get it then you could win MH but you're not meant "
            "to win this you know but"
        )
        transcript = (
            "a jna sometimes it's good every [\u00a0__\u00a0] game " + target
        )
        selected = _selected(transcript, champion="Varus", target=target)
        start = transcript.index(target)
        context = retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r5",
        )
        clean = (
            target
            .replace("yeah", "Yeah", 1)
            .replace(" win MH but", " win but")
        )
        evidence = (
            "Yeah and now you don't win but the one way that you can win "
            "is if you do get push so if there's an angle to get push then "
            "you could get it then you could win MH but you're not meant "
            "to win this you know but"
        )
        raw = json.dumps({
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": [
                {
                    "original_text": "yeah",
                    "replacement": "Yeah",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "Sentence-initial capitalization.",
                },
                {
                    "original_text": " MH",
                    "replacement": "",
                    "repair_type": "FILLER",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": (
                        "Non-lexical filler 'MH' removed, including the "
                        "preceding space."
                    ),
                },
            ],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": "filler deletion plus capitalization",
        })
        reconstruction = run_reconstruction(
            selected,
            transcript=transcript,
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=self._diagnostic(champion="Varus"),
            chat=CountingChat([raw]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(reconstruction["clean_target_transcript"], clean)
        self.assertEqual(len(reconstruction["attempts"]), 1)
        filler = next(
            repair for repair in reconstruction["contextual_repairs"]
            if repair["repair_type"] == "FILLER"
        )
        self.assertEqual(filler["original_text"], " MH")
        self.assertEqual(filler["replacement"], "")
        self.assertEqual(
            target[filler["target_local_start"]:filler["target_local_end"]],
            " MH",
        )
        self.assertTrue(filler["evidence_spans"])
        # The raw compact preserves the provider's empty replacement and the
        # deterministic application reproduces clean exactly.
        self.assertEqual(
            reconstruction["raw_compact"]["contextual_repairs"][1][
                "replacement"
            ],
            "",
        )

    def test_filler_empty_replacement_rejected_for_every_other_type(self):
        target = "you could win MH but"
        selected = _selected(target, champion="Varus")
        context = self._context(target, target, previous=0)
        for repair_type in CONTEXTUAL_REPAIR_TYPES:
            if repair_type == "FILLER":
                continue
            compact = {
                "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
                "clean_target_transcript": target.replace(" MH", ""),
                "contextual_repairs": [{
                    "original_text": " MH",
                    "replacement": "",
                    "repair_type": repair_type,
                    "confidence": "HIGH",
                    "evidence_quotes": [target],
                    "rationale": "generic deletion attempt",
                }],
                "bindings": [],
                "unresolved_alternatives": [],
                "rationale": "must fail closed",
            }
            raw = json.dumps(compact)
            with self.assertRaises(ProviderCorrectionExhausted) as caught:
                run_reconstruction(
                    selected,
                    transcript=target,
                    context=context,
                    mechanical_cleaned_text=target,
                    final_diagnostic=self._diagnostic(champion="Varus"),
                    chat=CountingChat([raw, raw, raw, raw]),
                    config_hash=canonical_sha256({"v": 1}),
                )
            self.assertIn(
                "replacement must be non-empty",
                caught.exception.attempts[0]["error"],
                msg=repair_type,
            )

    def test_filler_replay_zyip_base_attempt_validates(self):
        # Exact live pool:ZYIP base payload: the only invalid field was the
        # empty FILLER replacement; with the FILLER-only exception the base
        # attempt validates on its own.
        target = (
            "yeah and now you don't win but the one way that you can win "
            "is if you do get push so if there's an angle to get push then "
            "you could get it then you could win MH but you're not meant "
            "to win this you know but"
        )
        transcript = (
            "a jna sometimes it's good every [\u00a0__\u00a0] game " + target
        )
        selected = _selected(transcript, champion="Varus", target=target)
        start = transcript.index(target)
        context = retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=0,
            radius_label="r5",
        )
        clean = (
            target
            .replace("yeah", "Yeah", 1)
            .replace(" win MH but", " win but")
        )
        evidence = (
            "Yeah and now you don't win but the one way that you can win "
            "is if you do get push so if there's an angle to get push then "
            "you could get it then you could win MH but you're not meant "
            "to win this you know but"
        )
        raw = json.dumps({
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": clean,
            "contextual_repairs": [
                {
                    "original_text": "yeah",
                    "replacement": "Yeah",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "Sentence-initial capitalization.",
                },
                {
                    "original_text": " MH",
                    "replacement": "",
                    "repair_type": "FILLER",
                    "confidence": "HIGH",
                    "evidence_quotes": [evidence],
                    "rationale": "Non-lexical filler 'MH' removed.",
                },
            ],
            "bindings": [],
            "unresolved_alternatives": [],
            "rationale": (
                "The target transcript contains only a filler 'MH' and a "
                "capitalization issue."
            ),
        })
        reconstruction = run_reconstruction(
            selected,
            transcript=transcript,
            context=context,
            mechanical_cleaned_text=target,
            final_diagnostic=self._diagnostic(champion="Varus"),
            chat=CountingChat([raw]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(reconstruction["clean_target_transcript"], clean)
        self.assertEqual(len(reconstruction["attempts"]), 1)

    def test_mechanical_four_attempt_live_progression_succeeds(self):
        # Live pool:n2 progression: base overlap failure, correction:1
        # missing capitalization + comma, correction:2 missing the comma,
        # then a complete valid repair response on correction:3.  The
        # mechanical stage allows one initial + three corrections and every
        # correction receives the prior raw response and the latest
        # actionable diff.
        bronze = (
            "then does she get to farm? No, no, she loses the whole wave. "
            "Then yes, yes so that's why you should run at her now that she "
            "uses Q because if she uses E right, we then you lose like 100 "
            "HP 150 but then she has no"
        )
        clean = bronze.replace("then does", "Then does").replace(
            "yes so", "yes, so", 1,
        )
        selected = _selected(bronze)
        overlap = _compact_mechanical_raw(
            selected,
            clean_text=clean,
            repairs=[
                {
                    "original_text": "then does",
                    "replacement": "Then does",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "rationale": "overlap a",
                },
                {
                    "original_text": "does she",
                    "replacement": "did she",
                    "repair_type": "SPELLING",
                    "confidence": "HIGH",
                    "rationale": "overlap b",
                },
            ],
        )
        capitalization_only = _compact_mechanical_raw(
            selected,
            clean_text=clean,
            repairs=[{
                "original_text": "then does",
                "replacement": "Then does",
                "repair_type": "CAPITALIZATION",
                "confidence": "HIGH",
                "rationale": "capitalization only",
            }],
        )
        missing_both = _compact_mechanical_raw(
            selected,
            clean_text=clean,
            rationale="neither capitalization nor comma represented",
        )
        complete = _compact_mechanical_raw(
            selected,
            clean_text=clean,
            repairs=[
                {
                    "original_text": "then does",
                    "replacement": "Then does",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "rationale": "sentence start",
                },
                {
                    "original_text": "yes so",
                    "replacement": "yes, so",
                    "repair_type": "PUNCTUATION",
                    "confidence": "HIGH",
                    "rationale": "missing comma",
                },
            ],
        )
        captured: list[tuple[str, str, str]] = []

        def chat(system: str, user: str) -> str:
            payload = json.loads(user)
            if payload.get("task") == "mechanical_cleanup_correction":
                captured.append(
                    (system, user, payload["validator_error"]),
                )
            if len(captured) == 0:
                return overlap
            if len(captured) == 1:
                return missing_both
            if len(captured) == 2:
                return capitalization_only
            return complete

        result = run_mechanical_cleanup(
            selected,
            chat=chat,
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["mechanical_cleaned_text"], clean)
        self.assertEqual(len(result["attempts"]), 4)
        self.assertEqual(
            [attempt["status"] for attempt in result["attempts"]],
            ["FAILED", "FAILED", "FAILED", "OK"],
        )
        self.assertEqual(
            [attempt["attempt_kind"] for attempt in result["attempts"]],
            ["base", "correction:1", "correction:2", "correction:3"],
        )
        self.assertIn("must not overlap", result["attempts"][0]["error"])
        self.assertIn("ordered_non_equal_changes", result["attempts"][1]["error"])
        self.assertIn("ordered_non_equal_changes", result["attempts"][2]["error"])
        self.assertEqual(len(captured), 3)
        # correction:3 receives the prior raw response (embedded in the user
        # payload) and the latest actionable comma diff.
        self.assertIn("prior_raw_response", captured[2][1])
        self.assertIn("at most three corrections", captured[2][0])
        self.assertIn("applied=", captured[2][2])

    def test_mechanical_exhaustion_after_four_attempts(self):
        selected = _selected("he used W.")
        bad = _compact_mechanical_raw(
            selected,
            clean_text="He used W. now",
            rationale="unrepresented trailing edit",
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([bad, bad, bad, bad]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertEqual(
            len(caught.exception.attempts), MECHANICAL_MAX_CORRECTIONS + 1,
        )
        self.assertEqual(
            [attempt["attempt_kind"] for attempt in caught.exception.attempts],
            ["base", "correction:1", "correction:2", "correction:3"],
        )
        self.assertTrue(all(
            attempt["status"] == "FAILED"
            for attempt in caught.exception.attempts
        ))

    def test_polish_four_attempt_live_progression_succeeds(self):
        # Live pool:w9 progression: base absent repaired evidence quote
        # (with an exact Bronze suggestion), malformed correction, wrong
        # UNCHANGED_EXACT correction, then a valid RECONSTRUCTION_DERIVED
        # response on correction:3.  Four attempts with raw lineage and
        # success.
        text = (
            "swap if Camille absolutely needs it. But you should not want "
            "to swap, you know."
        )
        clean = text.replace("swap", "Swap", 1)
        selected = _selected(text, champion="Fizz")
        context = self._context(text, text, previous=0)
        diagnostic = self._diagnostic(champion="Fizz")
        repair = {
            "original_text": "swap",
            "replacement": "Swap",
            "repair_type": "CAPITALIZATION",
            "confidence": "HIGH",
            "evidence_quotes": [text],
            "rationale": "sentence start",
        }
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=diagnostic,
            chat=CountingChat([_reconstruction_raw(
                cleaned=clean,
                bronze=text,
                base_offset=0,
                selected=selected,
                contextual_repairs=[repair],
                bindings_override=[],
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        operation_id = reconstruction["contextual_repairs"][0]["repair_id"]

        base = _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "Swap if Camille absolutely needs it.",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["Swap if Camille absolutely needs it."],
                "reconstruction_operation_ids": [operation_id],
                "support_mode": "RECONSTRUCTION_DERIVED",
                "unchanged_source_quote": None,
            }],
        )
        malformed = "{_schema=== not json"
        wrong_unchanged = _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "Swap if Camille absolutely needs it.",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["swap if Camille absolutely needs it."],
                "reconstruction_operation_ids": [operation_id],
                "support_mode": "UNCHANGED_EXACT",
                "unchanged_source_quote": "swap if Camille absolutely needs it.",
            }],
        )
        valid = _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "Swap if Camille absolutely needs it.",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["swap if Camille absolutely needs it."],
                "reconstruction_operation_ids": [operation_id],
                "support_mode": "RECONSTRUCTION_DERIVED",
                "unchanged_source_quote": None,
            }],
        )
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "raw_responses"
            polish = run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                reconstruction=reconstruction,
                chat=CountingChat([base, malformed, wrong_unchanged, valid]),
                config_hash=canonical_sha256({"v": 1}),
                raw_response_dir=raw_dir,
            )
            self.assertEqual(len(polish["attempts"]), 4)
            self.assertEqual(
                [attempt["status"] for attempt in polish["attempts"]],
                ["FAILED", "FAILED", "FAILED", "OK"],
            )
            self.assertEqual(
                [attempt["attempt_kind"] for attempt in polish["attempts"]],
                ["base", "correction:1", "correction:2", "correction:3"],
            )
            base_error = polish["attempts"][0]["error"]
            self.assertIn("absent from Bronze", base_error)
            self.assertIn("Suggested exact replacement evidence_quote", base_error)
            self.assertIn(
                repr("swap if Camille absolutely needs it."), base_error,
            )
            self.assertIn(
                "text must exactly equal its unchanged_source_quote",
                polish["attempts"][2]["error"],
            )
            statement = polish["statements"][0]
            self.assertEqual(statement["support_mode"], "RECONSTRUCTION_DERIVED")
            self.assertEqual(
                statement["evidence_spans"][0]["text"],
                "swap if Camille absolutely needs it.",
            )
            self.assertIsNone(statement["unchanged_source_quote"])
            self.assertEqual(
                statement["reconstruction_operation_ids"], [operation_id],
            )
            for attempt in polish["attempts"]:
                raw_file = raw_dir / attempt["model_call"]["raw_response_path"]
                self.assertTrue(raw_file.is_file())
                self.assertRegex(
                    attempt["model_call"]["raw_response_sha256"],
                    r"[0-9a-f]{64}",
                )

    def test_polish_exhaustion_after_four_attempts_fail_closed(self):
        text = "swap if Camille absolutely needs it."
        selected = _selected(text, champion="Fizz")
        context = self._context(text, text, previous=0)
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=self._diagnostic(champion="Fizz"),
            chat=CountingChat([_reconstruction_raw(
                cleaned=text.replace("swap", "Swap", 1),
                bronze=text,
                base_offset=0,
                selected=selected,
                contextual_repairs=[{
                    "original_text": "swap",
                    "replacement": "Swap",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "evidence_quotes": [text],
                    "rationale": "sentence start",
                }],
                bindings_override=[],
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        malformed = "{bad json"
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=text,
                reconstruction=reconstruction,
                chat=CountingChat([
                    malformed, malformed, malformed, malformed,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertEqual(
            len(caught.exception.attempts), POLISH_MAX_CORRECTIONS + 1,
        )
        self.assertTrue(all(
            attempt["status"] == "FAILED"
            for attempt in caught.exception.attempts
        ))

    def test_polish_absent_evidence_suggestion_requires_unique_normalized_slice(
        self,
    ):
        # When the normalized fallback maps to zero or multiple distinct
        # Bronze slices there is no suggestion; the quote still fails.
        from pipeline.phase2k_contextual_reconstruction import _bind_bronze_quotes

        with self.assertRaises(ValueError) as caught:
            _bind_bronze_quotes(
                ["nope"],
                bronze_text="Nothing here matches.",
                base_offset=0,
                label="polish statement 1",
            )
        self.assertIn("absent from Bronze", str(caught.exception))
        self.assertNotIn("Suggested exact replacement", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            _bind_bronze_quotes(
                ["nami"],
                bronze_text="nami nami",
                base_offset=0,
                label="polish statement 1",
            )
        self.assertIn("ambiguous", str(caught.exception))
        self.assertNotIn("Suggested exact replacement", str(caught.exception))


class Phase2KReconstructionCorrectionV14HardeningTests(unittest.TestCase):
    """Phase 2K v14 reconstruction-correction hardening.

    - Correction prompt v7: every evidence_quotes field at every nesting
      level must be a JSON array, and RESOLVED resolved_candidate must copy
      one complete licensed diagnostic candidate byte-for-byte.
    - Reconstruction allows one initial attempt plus at most three
      corrections (four total calls); other stages keep their budgets.
    - Correction cache keys change through the v7 prompt version and cached
      v7 outputs are reused without mutation.
    """

    def _recon_setup(self):
        text = "Viktor is the player. He hit R. The replay shows it."
        target = "He hit R."
        start = text.index(target)
        selected = _selected(text, champion="Viktor", target=target)
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=1,
            following_segments=2,
            radius_label="r5",
        )
        slots = _slots(decision="SUFFICIENT", champion="Viktor")
        slots["pronouns"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": "Viktor",
                "confidence": "HIGH",
                "evidence_spans": [],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {"slots": slots, "metadata_conflicts": []},
            },
        }
        return selected, context, diagnostic, target

    def _valid_compact(self) -> dict[str, Any]:
        return {
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "clean_target_transcript": "Viktor hit R.",
            "contextual_repairs": [{
                "original_text": "He",
                "replacement": "Viktor",
                "repair_type": "PRONOUN_RESOLUTION",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "rationale": "context names Viktor as the player",
            }],
            "bindings": [{
                "slot": "pronouns",
                "mention_text": "He",
                "resolved_candidate": "Viktor",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": ["Viktor is the player"],
                "alternatives": [],
                "metadata_contributed": False,
                "rationale": "context names Viktor as the player",
            }],
            "unresolved_alternatives": [],
            "rationale": "context names Viktor as the player",
        }

    def test_correction_prompt_v7_array_only_and_exact_candidate_rules(self):
        selected, context, diagnostic, target = self._recon_setup()
        system, user = build_reconstruction_correction_prompt(
            selected,
            context,
            target,
            diagnostic,
            prior_raw="{}",
            error=ValueError(
                "contextual repair proposal 2 evidence_quotes must be a list",
            ),
        )
        payload = json.loads(user)
        self.assertEqual(
            payload["correction_prompt_version"],
            RECONSTRUCTION_CORRECTION_PROMPT_VERSION,
        )
        self.assertEqual(
            RECONSTRUCTION_CORRECTION_PROMPT_VERSION,
            "phase2k-reconstruction-correction-prompt-v7",
        )
        self.assertIn("JSON array of exact quote strings", system)
        self.assertIn("byte-for-byte", system)
        self.assertIn("never a scalar string, object, or null", system)
        self.assertIn("parentheticals", system)
        self.assertIn("NBSP/Unicode whitespace", system)
        rules = payload["correction_rules"]
        self.assertIsInstance(rules, list)
        self.assertTrue(rules)
        self.assertTrue(any(
            "JSON array of exact quote strings" in rule for rule in rules
        ))
        self.assertTrue(any(
            "byte-for-byte" in rule for rule in rules
        ))
        self.assertTrue(any(
            "WHITESPACE" in rule for rule in rules
        ))

    def test_scalar_evidence_quotes_fail_then_array_correction_succeeds(self):
        selected, context, diagnostic, _target = self._recon_setup()
        valid = self._valid_compact()
        bad = dict(valid)
        bad["contextual_repairs"] = [{
            **valid["contextual_repairs"][0],
            "evidence_quotes": "Viktor is the player",
        }]
        reconstruction = run_reconstruction(
            selected,
            transcript="Viktor is the player. He hit R. The replay shows it.",
            context=context,
            mechanical_cleaned_text="He hit R.",
            final_diagnostic=diagnostic,
            chat=CountingChat([json.dumps(bad), json.dumps(valid)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"], "Viktor hit R.",
        )
        self.assertEqual(len(reconstruction["attempts"]), 2)
        self.assertEqual(reconstruction["attempts"][0]["status"], "FAILED")
        self.assertIn(
            "evidence_quotes must be a list",
            reconstruction["attempts"][0]["error"],
        )
        self.assertEqual(reconstruction["attempts"][1]["status"], "OK")

    def test_shortened_case_changed_candidate_fails_then_exact_allowed_succeeds(
        self,
    ):
        # Live pool:n2 shape: the provider shortened and title-cased the
        # licensed candidate, stripping its parenthetical qualifier.  The
        # shortened value must fail; the exact full allowed candidate must
        # be copied byte-for-byte.
        text = "she uses e right we go"
        exact_candidate = "enemy mid laner (champion not named in transcript)"
        selected = _selected(text, champion="Lux")
        context = retrieve_context(
            text,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=0,
            target_end=len(text),
            bronze_text=text,
            previous_segments=0,
            following_segments=0,
            radius_label="target_only",
        )
        slots = _slots(decision="SUFFICIENT", champion="Lux")
        slots["champion_identities"] = {
            "status": "RESOLVED",
            "candidates": [{
                "candidate": exact_candidate,
                "confidence": "HIGH",
                "evidence_spans": [],
            }],
            "confidence": "HIGH",
            "evidence_spans": [],
        }
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {"slots": slots, "metadata_conflicts": []},
            },
        }

        def compact(candidate: str) -> str:
            return json.dumps({
                "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
                "clean_target_transcript": text,
                "contextual_repairs": [],
                "bindings": [{
                    "slot": "champion_identities",
                    "mention_text": "she",
                    "resolved_candidate": candidate,
                    "resolved_status": "RESOLVED",
                    "confidence": "HIGH",
                    "evidence_quotes": [],
                    "alternatives": [],
                    "metadata_contributed": False,
                    "rationale": "test candidate copy",
                }],
                "unresolved_alternatives": [],
                "rationale": "test candidate copy",
            })

        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=text,
            final_diagnostic=diagnostic,
            chat=CountingChat([
                compact("Enemy mid laner"),
                compact(exact_candidate),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(len(reconstruction["attempts"]), 2)
        error = reconstruction["attempts"][0]["error"]
        self.assertIn("not licensed", error)
        self.assertIn(repr(exact_candidate), error)
        self.assertEqual(
            reconstruction["bindings"][0]["resolved_candidate"],
            exact_candidate,
        )

    def test_reconstruction_succeeds_on_correction_three_with_stage_budget(self):
        selected, context, diagnostic, _target = self._recon_setup()
        valid = self._valid_compact()
        bad1 = dict(valid)
        bad1["contextual_repairs"] = [{
            **valid["contextual_repairs"][0],
            "evidence_quotes": "Viktor is the player",
        }]
        bad2 = dict(valid)
        del bad2["rationale"]
        bad3 = dict(valid)
        bad3["clean_target_transcript"] = "Viktor hit R. extra"
        reconstruction = run_reconstruction(
            selected,
            transcript="Viktor is the player. He hit R. The replay shows it.",
            context=context,
            mechanical_cleaned_text="He hit R.",
            final_diagnostic=diagnostic,
            chat=CountingChat([
                json.dumps(bad1),
                json.dumps(bad2),
                json.dumps(bad3),
                json.dumps(valid),
            ]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(
            reconstruction["clean_target_transcript"], "Viktor hit R.",
        )
        self.assertEqual(
            [attempt["attempt_kind"] for attempt in reconstruction["attempts"]],
            ["base", "correction:1", "correction:2", "correction:3"],
        )
        self.assertEqual(
            len(reconstruction["attempts"]), RECONSTRUCTION_MAX_CORRECTIONS + 1,
        )
        self.assertEqual(RECONSTRUCTION_MAX_CORRECTIONS + 1, 4)

        # The generic/sufficiency correction budget is unchanged.
        suff_selected = _selected("he used W.")
        suff_context = retrieve_context(
            "he used W.",
            source_group_id=suff_selected["source_group_id"],
            window_id=suff_selected["window_id"],
            target_start=0,
            target_end=len("he used W."),
            bronze_text="he used W.",
            previous_segments=0,
            following_segments=0,
            radius_label="target_only",
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_sufficiency_diagnostic(
                suff_selected,
                transcript="he used W.",
                context=suff_context,
                mechanical_cleaned_text="He used W.",
                at_max_context=False,
                stage_label="r1",
                chat=CountingChat(["{bad", "{bad", "{bad"]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertEqual(
            len(caught.exception.attempts), PROVIDER_MAX_CORRECTIONS + 1,
        )
        self.assertEqual(PROVIDER_MAX_CORRECTIONS + 1, 3)
        self.assertEqual(
            [attempt["attempt_kind"] for attempt in caught.exception.attempts],
            ["base", "correction:1", "correction:2"],
        )

    def test_correction_cache_keys_bind_v7_prompt_version_and_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            selected, context, diagnostic, _target = self._recon_setup()
            valid = self._valid_compact()
            bad = dict(valid)
            bad["contextual_repairs"] = [{
                **valid["contextual_repairs"][0],
                "evidence_quotes": "Viktor is the player",
            }]
            first = run_reconstruction(
                selected,
                transcript="Viktor is the player. He hit R. The replay shows it.",
                context=context,
                mechanical_cleaned_text="He hit R.",
                final_diagnostic=diagnostic,
                chat=CountingChat([json.dumps(bad), json.dumps(valid)]),
                config_hash=canonical_sha256({"v": 1}),
                cache_dir=cache_dir,
            )
            second = run_reconstruction(
                selected,
                transcript="Viktor is the player. He hit R. The replay shows it.",
                context=context,
                mechanical_cleaned_text="He hit R.",
                final_diagnostic=diagnostic,
                chat=CountingChat([json.dumps(bad), json.dumps(valid)]),
                config_hash=canonical_sha256({"v": 1}),
                cache_dir=cache_dir,
            )
            first_correction = first["attempts"][1]["model_call"]
            second_correction = second["attempts"][1]["model_call"]
            self.assertEqual(
                first_correction["prompt_version"],
                RECONSTRUCTION_CORRECTION_PROMPT_VERSION,
            )
            self.assertNotEqual(
                first["attempts"][0]["model_call"]["cache_key"],
                first_correction["cache_key"],
            )
            self.assertEqual(
                first_correction["cache_key"], second_correction["cache_key"],
            )
            self.assertEqual(
                second["attempts"][0]["model_call"]["source"], "cache",
            )
            self.assertEqual(
                second["attempts"][1]["model_call"]["source"], "cache",
            )


class Phase2KLiveV15FixTests(unittest.TestCase):
    """Focused regression tests for the Phase 2K v15 live repair.

    - Context-only binding proposals (a mention with zero target-Bronze
      matches but at least one exact/surface-normalized match in the
      supplied ordered context, for example the live ``enemy mid``
      mention) are conservatively omitted from normalized target bindings,
      preserved verbatim in ``raw_compact``, and counted in
      ``omitted_binding_count``; mentions absent from both target and
      context still fail with remove-entire-binding guidance, and
      target-present bindings are unaffected.
    - UNCHANGED_EXACT polish text mismatches now report the exact actual
      text and exact unchanged quote plus explicit source-exact /
      RECONSTRUCTION_DERIVED repair actions, and the polish correction
      prompt bumps to v3 with coherent retry/cache provenance.
    """

    def _context(
        self,
        transcript: str,
        target: str,
        *,
        previous: int = 1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        start = transcript.index(target)
        selected = _selected(transcript, target=target)
        context = retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=start,
            target_end=start + len(target),
            bronze_text=target,
            previous_segments=previous,
            following_segments=0,
            radius_label="r5",
        )
        return selected, context

    def _binding(
        self,
        *,
        slot: str,
        mention: str,
        candidate: str,
    ) -> dict[str, Any]:
        return {
            "slot": slot,
            "mention_text": mention,
            "resolved_candidate": candidate,
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_quotes": [],
            "alternatives": [],
            "metadata_contributed": False,
            "rationale": "test binding",
        }

    def _diagnostic(
        self,
        *,
        slots: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        diagnostic_slots = _slots(decision="SUFFICIENT")
        if slots:
            diagnostic_slots.update(slots)
        return {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": diagnostic_slots,
                    "metadata_conflicts": [],
                },
            },
        }

    def _reconstruction(
        self,
        selected: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        transcript: str,
        clean: str,
        bindings: list[dict[str, Any]] | None = None,
        slots: Mapping[str, Any] | None = None,
        chat: Any = None,
    ) -> dict[str, Any]:
        raw = _reconstruction_raw(
            cleaned=clean,
            bronze=selected["source_text"],
            base_offset=selected["upstream_start"],
            selected=selected,
            bindings_override=bindings or [],
        )
        if chat is None:
            chat = CountingChat([raw, raw, raw, raw])
        return run_reconstruction(
            selected,
            transcript=transcript,
            context=context,
            mechanical_cleaned_text=clean,
            final_diagnostic=self._diagnostic(slots=slots),
            chat=chat,
            config_hash=canonical_sha256({"v": 1}),
        )

    def _polish_setup(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        bronze = "keep laning. I can lane."
        selected = _selected(bronze, champion="Lux")
        context = retrieve_context(
            bronze,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=0,
            target_end=len(bronze),
            bronze_text=bronze,
            previous_segments=0,
            following_segments=0,
            radius_label="target_only",
        )
        reconstruction = run_reconstruction(
            selected,
            transcript=bronze,
            context=context,
            mechanical_cleaned_text=bronze,
            final_diagnostic=self._diagnostic(),
            chat=CountingChat([_reconstruction_raw(
                cleaned=bronze,
                bronze=bronze,
                base_offset=0,
                selected=selected,
                bindings_override=[],
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        return selected, context, reconstruction

    def _mismatched_polish_raw(
        self,
        selected: Mapping[str, Any],
        reconstruction: Mapping[str, Any],
    ) -> str:
        return _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "Keep laning.",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["keep laning."],
                "reconstruction_operation_ids": [],
                "support_mode": "UNCHANGED_EXACT",
                "unchanged_source_quote": "keep laning.",
            }],
        )

    def _exact_polish_raw(
        self,
        selected: Mapping[str, Any],
        reconstruction: Mapping[str, Any],
    ) -> str:
        return _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": "keep laning.",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["keep laning."],
                "reconstruction_operation_ids": [],
                "support_mode": "UNCHANGED_EXACT",
                "unchanged_source_quote": "keep laning.",
            }],
        )

    def test_context_only_enemy_mid_binding_omitted_and_audited(self):
        # Live pool:n2RuZ0vwkE4 shape: the provider bound "enemy mid", which
        # occurs only in the surrounding context segment, never in the
        # target Bronze.  It must be omitted from the normalized target
        # bindings, preserved verbatim in raw_compact, and counted, while a
        # target-present "she" binding stays untouched.
        transcript = (
            "enemy mid makes a mistake right he just used spell "
            "you're allowed to walk up because "
            "Then does she get to farm? No, no, she loses the whole wave."
        )
        target = "Then does she get to farm? No, no, she loses the whole wave."
        selected, context = self._context(transcript, target, previous=1)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "enemy mid laner",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        reconstruction = self._reconstruction(
            selected,
            context,
            transcript=transcript,
            clean=target,
            bindings=[
                self._binding(
                    slot="champion_identities",
                    mention="enemy mid",
                    candidate="enemy mid laner",
                ),
                self._binding(
                    slot="champion_identities",
                    mention="she",
                    candidate="enemy mid laner",
                ),
            ],
            slots=slots,
        )
        self.assertEqual(reconstruction["omitted_binding_count"], 1)
        self.assertEqual(len(reconstruction["bindings"]), 1)
        self.assertEqual(
            reconstruction["bindings"][0]["mention"]["text"], "she",
        )
        self.assertNotIn(
            "enemy mid",
            [binding["mention"]["text"] for binding in reconstruction["bindings"]],
        )
        raw_mentions = [
            proposal["mention_text"]
            for proposal in reconstruction["raw_compact"]["bindings"]
        ]
        self.assertIn("enemy mid", raw_mentions)
        self.assertIn("she", raw_mentions)
        self.assertEqual(
            len(reconstruction["raw_compact"]["bindings"]),
            len(reconstruction["bindings"])
            + reconstruction["omitted_binding_count"],
        )
        _validate_reconstruction_raw_compact(
            reconstruction["raw_compact"],
            reconstruction=reconstruction,
            label="unit test",
        )

    def test_raw_compact_count_tampering_detected(self):
        transcript = (
            "enemy mid makes a mistake right he just used spell "
            "Then does she get to farm? No, no, she loses the whole wave."
        )
        target = "Then does she get to farm? No, no, she loses the whole wave."
        selected, context = self._context(transcript, target, previous=1)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "enemy mid laner",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        reconstruction = self._reconstruction(
            selected,
            context,
            transcript=transcript,
            clean=target,
            bindings=[self._binding(
                slot="champion_identities",
                mention="enemy mid",
                candidate="enemy mid laner",
            )],
            slots=slots,
        )
        self.assertEqual(reconstruction["omitted_binding_count"], 1)
        self.assertEqual(len(reconstruction["raw_compact"]["bindings"]), 1)
        self.assertEqual(reconstruction["bindings"], [])

        # Dropping the preserved raw proposal breaks the documented count
        # invariant (normalized + omitted == raw).
        dropped = {
            **reconstruction["raw_compact"],
            "bindings": reconstruction["raw_compact"]["bindings"][:-1],
        }
        with self.assertRaises(ValueError) as caught:
            _validate_reconstruction_raw_compact(
                dropped,
                reconstruction=reconstruction,
                label="unit test",
            )
        self.assertIn(
            "raw_compact bindings count is inconsistent",
            str(caught.exception),
        )

        # Zeroing the omission count also breaks the invariant.
        tampered = dict(reconstruction)
        tampered["omitted_binding_count"] = 0
        with self.assertRaises(ValueError) as caught:
            _validate_reconstruction_raw_compact(
                reconstruction["raw_compact"],
                reconstruction=tampered,
                label="unit test",
            )
        self.assertIn(
            "raw_compact bindings count is inconsistent",
            str(caught.exception),
        )

    def test_absent_from_both_target_and_context_still_fails(self):
        # "your queue" is absent from the target Bronze AND from the
        # supplied surrounding context, so it is not context-only and must
        # keep exhausting corrections with remove-entire-binding guidance.
        transcript = "Viktor is the player. He hit R."
        target = "He hit R."
        selected, context = self._context(transcript, target, previous=1)
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            self._reconstruction(
                selected,
                context,
                transcript=transcript,
                clean=target,
                bindings=[self._binding(
                    slot="ability_ownership",
                    mention="your queue",
                    candidate="Mel",
                )],
            )
        error = caught.exception.attempts[0]["error"]
        self.assertIn("absent from the supplied source", error)
        self.assertIn("Remove this entire binding", error)

    def test_target_present_binding_behavior_unchanged(self):
        transcript = "enemy mid makes a mistake. Lux hit R."
        target = "Lux hit R."
        selected, context = self._context(transcript, target, previous=1)
        slots = {
            "champion_identities": {
                "status": "RESOLVED",
                "candidates": [{
                    "candidate": "Lux",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                }],
                "confidence": "HIGH",
                "evidence_spans": [],
            },
        }
        reconstruction = self._reconstruction(
            selected,
            context,
            transcript=transcript,
            clean=target,
            bindings=[
                self._binding(
                    slot="champion_identities",
                    mention="Lux",
                    candidate="Lux",
                ),
                self._binding(
                    slot="champion_identities",
                    mention="lux",
                    candidate="Lux",
                ),
            ],
            slots=slots,
        )
        self.assertEqual(reconstruction["omitted_binding_count"], 0)
        self.assertEqual(len(reconstruction["bindings"]), 2)
        self.assertEqual(
            [binding["mention"]["text"] for binding in reconstruction["bindings"]],
            ["Lux", "Lux"],
        )
        self.assertEqual(
            len(reconstruction["raw_compact"]["bindings"]), 2,
        )

    def test_polish_unchanged_exact_mismatch_error_has_exact_data_and_actions(
        self,
    ):
        selected, context, reconstruction = self._polish_setup()
        mismatch = self._mismatched_polish_raw(selected, reconstruction)
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_polish(
                selected,
                transcript="keep laning. I can lane.",
                context=context,
                mechanical_cleaned_text="keep laning. I can lane.",
                reconstruction=reconstruction,
                chat=CountingChat([mismatch, mismatch, mismatch, mismatch]),
                config_hash=canonical_sha256({"v": 1}),
            )
        error = caught.exception.attempts[0]["error"]
        self.assertIn("UNCHANGED_EXACT text mismatch", error)
        self.assertIn(json.dumps("Keep laning."), error)
        self.assertIn(json.dumps("keep laning."), error)
        self.assertIn(
            "text must exactly equal its unchanged_source_quote", error,
        )
        self.assertIn("byte-exactly equal", error)
        self.assertIn("RECONSTRUCTION_DERIVED", error)
        self.assertIn("unchanged_source_quote null", error)
        self.assertIn(
            "reconstruction_operation_id supporting the change", error,
        )

    def test_polish_correction_prompt_v3_guidance_and_cache_provenance(self):
        selected, context, reconstruction = self._polish_setup()
        mismatch = self._mismatched_polish_raw(selected, reconstruction)
        exact = self._exact_polish_raw(selected, reconstruction)

        system, user = build_polish_correction_prompt(
            selected,
            context,
            "keep laning. I can lane.",
            reconstruction,
            prior_raw=mismatch,
            error=ValueError(
                "polish statement 1 UNCHANGED_EXACT text mismatch: actual "
                'text is "Keep laning." but unchanged_source_quote is '
                '"keep laning."',
            ),
        )
        payload = json.loads(user)
        self.assertEqual(
            payload["correction_prompt_version"],
            POLISH_CORRECTION_PROMPT_VERSION,
        )
        self.assertEqual(
            POLISH_CORRECTION_PROMPT_VERSION,
            "phase2k-semantic-polish-correction-prompt-v3",
        )
        self.assertIn("byte-exactly equal", system)
        self.assertIn("copy the exact quoted value verbatim", system)
        self.assertIn("RECONSTRUCTION_DERIVED", system)
        self.assertIn("unchanged_source_quote null", system)
        self.assertIn("never relabel repaired text as UNCHANGED_EXACT", system)

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            first = run_polish(
                selected,
                transcript="keep laning. I can lane.",
                context=context,
                mechanical_cleaned_text="keep laning. I can lane.",
                reconstruction=reconstruction,
                chat=CountingChat([mismatch, exact]),
                config_hash=canonical_sha256({"v": 1}),
                cache_dir=cache_dir,
            )
            second = run_polish(
                selected,
                transcript="keep laning. I can lane.",
                context=context,
                mechanical_cleaned_text="keep laning. I can lane.",
                reconstruction=reconstruction,
                chat=CountingChat([mismatch, exact]),
                config_hash=canonical_sha256({"v": 1}),
                cache_dir=cache_dir,
            )
        first_base = first["attempts"][0]["model_call"]
        first_correction = first["attempts"][1]["model_call"]
        second_base = second["attempts"][0]["model_call"]
        second_correction = second["attempts"][1]["model_call"]
        self.assertEqual(first_base["prompt_version"], POLISH_PROMPT_VERSION)
        self.assertEqual(
            first_correction["prompt_version"],
            POLISH_CORRECTION_PROMPT_VERSION,
        )
        self.assertNotEqual(first_base["cache_key"], first_correction["cache_key"])
        self.assertEqual(first_base["cache_key"], second_base["cache_key"])
        self.assertEqual(
            first_correction["cache_key"], second_correction["cache_key"],
        )
        self.assertEqual(second_base["source"], "cache")
        self.assertEqual(second_correction["source"], "cache")
        self.assertEqual(len(first["attempts"]), 2)
        self.assertEqual(
            [attempt["status"] for attempt in first["attempts"]],
            ["FAILED", "OK"],
        )


class Phase2KChampionLexicalV16Tests(unittest.TestCase):
    """Phase 2K v16 champion-name lexical normalization upgrade tests.

    The v2 lexical vocabulary drives deterministic exact word-boundary
    champion-spelling hints (direct, guarded, metadata-licensed, never).
    Every eligible hint must become an explicit DOMAIN_SPELLING repair with
    exact Bronze/source spans; omissions, wrong replacements, and extra
    unlicensed champion repairs fail closed and feed the correction flow.
    Common words (like/then/when/ward/well), grocery kale, fishing pike,
    ordinary rise/rice, and signature "sig" never become champion
    corrections; Soie stays non-mandatory; version/cache/config lineage
    changes and tampered vocabulary fail closed.
    """

    def _hint_repair(
        self,
        selected: Mapping[str, Any],
        hint: Mapping[str, Any],
        *,
        rationale: str = "listed lexical hint",
    ) -> dict[str, Any]:
        return {
            "original_text": hint["text"],
            "replacement": hint["canonical"],
            "repair_type": "DOMAIN_SPELLING",
            "confidence": "HIGH",
            "rationale": rationale,
        }

    def _hint_repairs(
        self,
        selected: Mapping[str, Any],
        hints: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            self._hint_repair(selected, hint, rationale=f"hint {index + 1}")
            for index, hint in enumerate(hints)
        ]

    def test_versus_kale_hint_required_correction_and_exact_provenance(self):
        bronze = "versus Kale top"
        selected = _selected(bronze)
        config_hash = canonical_sha256({"v": 1})

        # The prompt carries the deterministic hint list plus the v2 rules.
        _system, user = build_mechanical_prompt(selected)
        payload = json.loads(user)
        self.assertEqual(
            payload["lexical_vocabulary"]["schema_version"],
            "phase2k-league-lexical-vocabulary-v2",
        )
        self.assertEqual(len(payload["lexical_hints"]), 1)
        self.assertEqual(payload["lexical_hints"][0]["surface_text"], "Kale")
        self.assertEqual(payload["lexical_hints"][0]["canonical"], "Kayle")
        self.assertEqual(
            payload["lexical_hints"][0]["rule_category"], "direct",
        )
        self.assertIn(
            "champion_alias_rules",
            payload["lexical_vocabulary"],
        )

        hints = detect_champion_alias_hints(bronze, selected)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["text"], "Kale")
        self.assertEqual(hints[0]["canonical"], "Kayle")
        self.assertEqual(hints[0]["rule_category"], "direct")
        self.assertEqual(
            hints[0]["target_local_start"],
            bronze.index("Kale"),
        )

        # A provider response that omits the repair is rejected; the
        # correction flow then succeeds with the explicit repair.
        omitting = _compact_mechanical_raw(selected, clean_text=bronze)
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([omitting, omitting, omitting, omitting]),
                config_hash=config_hash,
            )
        error = caught.exception.attempts[0]["error"]
        self.assertIn("must repair every eligible champion-spelling hint", error)
        self.assertIn("Kale", error)
        self.assertIn("Kayle", error)
        self.assertIn("DOMAIN_SPELLING", error)

        good = _mechanical_raw(
            selected,
            repairs=self._hint_repairs(selected, hints),
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([omitting, good]),
            config_hash=config_hash,
        )
        self.assertEqual(
            [attempt["status"] for attempt in result["attempts"]],
            ["FAILED", "OK"],
        )
        self.assertEqual(result["mechanical_cleaned_text"], "versus Kayle top")
        self.assertEqual(result["repair_count"], 1)
        repair = result["repairs"][0]
        self.assertEqual(repair["repair_type"], "DOMAIN_SPELLING")
        self.assertEqual(repair["original_text"], "Kale")
        self.assertEqual(repair["replacement"], "Kayle")
        local_start = bronze.index("Kale")
        self.assertEqual(repair["target_local_start"], local_start)
        self.assertEqual(repair["target_local_end"], local_start + 4)
        self.assertEqual(
            repair["source_absolute_start"],
            selected["upstream_start"] + local_start,
        )
        self.assertEqual(
            repair["source_absolute_end"],
            selected["upstream_start"] + local_start + 4,
        )
        self.assertEqual(
            repair["evidence_spans"][0]["text"], "Kale",
        )
        self.assertEqual(result["provenance"]["prompt_version"], (
            "phase2k-mechanical-cleanup-prompt-v4"
        ))
        self.assertEqual(
            result["model_call"]["prompt_version"],
            "phase2k-mechanical-cleanup-correction-prompt-v4",
        )
        self.assertEqual(
            result["attempts"][1]["model_call"]["prompt_version"],
            "phase2k-mechanical-cleanup-correction-prompt-v4",
        )
        self.assertEqual(result["lexical_hints"], hints)
        self.assertEqual(result["lexical_hint_count"], 1)
        # Raw proposals are retained verbatim as the audit trail.
        self.assertEqual(
            result["raw_proposals"]["repairs"],
            json.loads(good)["repairs"],
        )

    def test_direct_aliases_case_insensitive_preserve_canonical(self):
        direct = [
            ("versus Kale", "Kayle"),
            ("BRIER is here", "Briar"),
            ("milo support", "Milio"),
            ("morana top", "Morgana"),
            ("RAAN jungle", "Rakan"),
            ("Atrox mid", "Aatrox"),
            ("talia adc", "Taliyah"),
            ("nocturn jungle", "Nocturne"),
        ]
        for bronze, canonical in direct:
            selected = _selected(bronze)
            hints = detect_champion_alias_hints(bronze, selected)
            self.assertEqual(len(hints), 1, bronze)
            self.assertEqual(hints[0]["canonical"], canonical, bronze)
            self.assertEqual(hints[0]["rule_category"], "direct", bronze)
            result = run_mechanical_cleanup(
                selected,
                chat=CountingChat([_mechanical_raw(
                    selected,
                    repairs=self._hint_repairs(selected, hints),
                )]),
                config_hash=canonical_sha256({"v": 1}),
            )
            self.assertEqual(
                result["repairs"][0]["replacement"], canonical,
            )
            self.assertEqual(
                result["repairs"][0]["repair_type"], "DOMAIN_SPELLING",
            )
            self.assertIn(canonical, result["mechanical_cleaned_text"])

    def test_repeated_eligible_occurrences_all_repaired(self):
        bronze = "Kale versus Kale. Kale top."
        selected = _selected(bronze)
        hints = detect_champion_alias_hints(bronze, selected)
        self.assertEqual(len(hints), 3)
        self.assertEqual(
            [hint["occurrence_index"] for hint in hints],
            [0, 1, 2],
        )

        # Repairing only the first occurrence leaves the other hints missing.
        partial = _mechanical_raw(
            selected,
            repairs=[self._hint_repair(selected, hints[0])],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([partial, partial, partial, partial]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn("hint:0002", caught.exception.attempts[0]["error"])
        self.assertIn("hint:0003", caught.exception.attempts[0]["error"])

        complete = _mechanical_raw(
            selected,
            repairs=self._hint_repairs(selected, hints),
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([complete]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["repair_count"], 3)
        self.assertEqual(
            result["mechanical_cleaned_text"],
            "Kayle versus Kayle. Kayle top.",
        )
        self.assertEqual(
            [repair["target_local_start"] for repair in result["repairs"]],
            [hint["target_local_start"] for hint in hints],
        )
        self.assertTrue(all(
            repair["repair_type"] == "DOMAIN_SPELLING"
            for repair in result["repairs"]
        ))

    def test_darus_metadata_licensed_only_with_darius(self):
        bronze = "versus darus top"

        # With Darius metadata the hint is eligible and mandatory.
        selected = _selected(bronze, champion="Darius")
        hints = detect_champion_alias_hints(bronze, selected)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["canonical"], "Darius")
        self.assertEqual(hints[0]["rule_category"], "metadata_licensed")
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([_mechanical_raw(
                selected, repairs=self._hint_repairs(selected, hints),
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["mechanical_cleaned_text"], "versus Darius top")

        # Without Darius metadata (Varus is the competing lexical
        # possibility) darus is never eligible and never forced.
        for champion in ("Varus", "Lux"):
            no_metadata = _selected(bronze, champion=champion)
            self.assertEqual(
                detect_champion_alias_hints(bronze, no_metadata), [],
            )
            raw = _compact_mechanical_raw(
                no_metadata, clean_text=bronze,
            )
            result = run_mechanical_cleanup(
                no_metadata,
                chat=CountingChat([raw]),
                config_hash=canonical_sha256({"v": 1}),
            )
            self.assertEqual(result["mechanical_cleaned_text"], bronze)
            self.assertEqual(result["repairs"], [])

            invented = _compact_mechanical_raw(
                no_metadata,
                clean_text=bronze,
                repairs=[{
                    "original_text": "darus",
                    "replacement": "Darius",
                    "repair_type": "DOMAIN_SPELLING",
                    "confidence": "HIGH",
                    "rationale": "unlicensed",
                }],
            )
            with self.assertRaises(ProviderCorrectionExhausted) as caught:
                run_mechanical_cleanup(
                    no_metadata,
                    chat=CountingChat([
                        invented, invented, invented, invented,
                    ]),
                    config_hash=canonical_sha256({"v": 1}),
                )
            self.assertIn(
                "not licensed by any eligible champion-spelling hint",
                caught.exception.attempts[0]["error"],
            )

        # The rule is licensed by an exact champion field/value match, not
        # by Darius appearing in an unrelated metadata field such as title.
        wrong_field = _selected(bronze, champion="Varus")
        wrong_field["metadata"]["video_title"] = "Darius matchup guide"
        self.assertEqual(
            detect_champion_alias_hints(bronze, wrong_field), [],
        )

    def test_never_words_never_become_champion_corrections(self):
        never = [
            ("I like this", "like", "Pyke"),
            ("then we push", "then", "Shen"),
            ("when you walk up", "when", "Shen"),
            ("ward the river", "ward", "Bard"),
            ("well played", "well", "Rell"),
        ]
        for bronze, surface, canonical in never:
            selected = _selected(bronze)
            self.assertEqual(
                detect_champion_alias_hints(bronze, selected), [],
            )
            untouched = _compact_mechanical_raw(
                selected, clean_text=bronze,
            )
            result = run_mechanical_cleanup(
                selected,
                chat=CountingChat([untouched]),
                config_hash=canonical_sha256({"v": 1}),
            )
            self.assertEqual(result["repairs"], [])

            invented = _compact_mechanical_raw(
                selected,
                clean_text=bronze,
                repairs=[{
                    "original_text": surface,
                    "replacement": canonical,
                    "repair_type": "DOMAIN_SPELLING",
                    "confidence": "HIGH",
                    "rationale": "unlicensed",
                }],
            )
            with self.assertRaises(ProviderCorrectionExhausted) as caught:
                run_mechanical_cleanup(
                    selected,
                    chat=CountingChat([
                        invented, invented, invented, invented,
                    ]),
                    config_hash=canonical_sha256({"v": 1}),
                )
            self.assertIn(
                "not licensed by any eligible champion-spelling hint",
                caught.exception.attempts[0]["error"],
            )

    def test_soie_remains_non_mandatory_uncertainty_only(self):
        bronze = "Soie is strong in lane"
        selected = _selected(bronze)
        self.assertEqual(detect_champion_alias_hints(bronze, selected), [])

        # No repair is forced; the text stays unchanged.
        untouched = _compact_mechanical_raw(selected, clean_text=bronze)
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([untouched]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["repairs"], [])
        self.assertEqual(result["mechanical_cleaned_text"], bronze)

        # Soie->Zoe is never an automatic repair.
        invented = _compact_mechanical_raw(
            selected,
            clean_text=bronze,
            repairs=[{
                "original_text": "Soie",
                "replacement": "Zoe",
                "repair_type": "DOMAIN_SPELLING",
                "confidence": "HIGH",
                "rationale": "unlicensed",
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([invented, invented, invented, invented]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "not licensed by any eligible champion-spelling hint",
            caught.exception.attempts[0]["error"],
        )

        # A DOMAIN_TOKEN_UNCERTAIN uncertainty with Zoe among the
        # alternatives is the licensed non-mandatory expression.
        start = bronze.index("Soie")
        uncertain = _compact_mechanical_raw(
            selected,
            clean_text=bronze,
            uncertainties=[{
                "surface_text": "Soie",
                "uncertainty_type": "DOMAIN_TOKEN_UNCERTAIN",
                "alternatives": [
                    {"text": "Zoe", "confidence": "MEDIUM"},
                    {"text": "Soie", "confidence": "MEDIUM"},
                ],
                "note": "champion-name surface uncertainty",
            }],
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([uncertain]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["repairs"], [])
        self.assertEqual(result["uncertainties"][0]["text"], "Soie")
        self.assertEqual(
            result["uncertainties"][0]["target_local_start"], start,
        )
        self.assertEqual(
            [item["text"] for item in result["uncertainties"][0]["alternatives"]],
            ["Zoe", "Soie"],
        )

    def test_guarded_pike_ryze_sig_positive_and_negative_corpus(self):
        positives = [
            ("versus pike", "Pyke"),
            ("playing pike", "Pyke"),
            ("pike support", "Pyke"),
            ("playing rise", "Ryze"),
            ("rise mid", "Ryze"),
            ("rice support", "Ryze"),
            ("Draven is stronger than Sig", "Ziggs"),
            ("pick Sig", "Ziggs"),
            ("Sig support", "Ziggs"),
        ]
        for bronze, canonical in positives:
            selected = _selected(bronze)
            hints = detect_champion_alias_hints(bronze, selected)
            self.assertEqual(len(hints), 1, bronze)
            self.assertEqual(hints[0]["canonical"], canonical, bronze)
            self.assertEqual(hints[0]["rule_category"], "guarded", bronze)
            self.assertIsNotNone(hints[0]["syntax_hint"], bronze)
            result = run_mechanical_cleanup(
                selected,
                chat=CountingChat([_mechanical_raw(
                    selected,
                    repairs=self._hint_repairs(selected, hints),
                )]),
                config_hash=canonical_sha256({"v": 1}),
            )
            self.assertEqual(
                result["repairs"][0]["replacement"], canonical,
            )
            self.assertIn(canonical, result["mechanical_cleaned_text"])

        negatives = [
            "we need a fishing pike",
            "the pike is by the river",
            "the sig file",
            "that sig means signature",
            "worse than rice",
            "the price of rice went up",
            "rise and shine",
            "rise to the occasion",
        ]
        for bronze in negatives:
            selected = _selected(bronze)
            self.assertEqual(
                detect_champion_alias_hints(bronze, selected), [],
                bronze,
            )
            result = run_mechanical_cleanup(
                selected,
                chat=CountingChat([_compact_mechanical_raw(
                    selected, clean_text=bronze,
                )]),
                config_hash=canonical_sha256({"v": 1}),
            )
            self.assertEqual(result["repairs"], [], bronze)

    def test_grocery_kale_capitalization_rule_documented(self):
        # Capital-initial Kale is an unconditional direct hint; lowercase
        # grocery "kale" is the documented negative.
        capital = _selected("Kale is strong in lane")
        self.assertEqual(len(detect_champion_alias_hints(
            capital["source_text"], capital,
        )), 1)

        bronze = "grocery kale is healthy"
        selected = _selected(bronze)
        self.assertEqual(detect_champion_alias_hints(bronze, selected), [])
        untouched = _compact_mechanical_raw(selected, clean_text=bronze)
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([untouched]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["repairs"], [])
        self.assertEqual(result["mechanical_cleaned_text"], bronze)

        invented = _compact_mechanical_raw(
            selected,
            clean_text=bronze,
            repairs=[{
                "original_text": "kale",
                "replacement": "Kayle",
                "repair_type": "DOMAIN_SPELLING",
                "confidence": "HIGH",
                "rationale": "unlicensed grocery kale",
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([invented, invented, invented, invented]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "not licensed by any eligible champion-spelling hint",
            caught.exception.attempts[0]["error"],
        )

    def test_contradictory_uncertainty_does_not_force_correction(self):
        # Without Darius metadata, darus is not eligible; a contradictory
        # uncertainty (Darius vs Varus) is retained without any correction.
        bronze = "versus darus top"
        selected = _selected(bronze, champion="Varus")
        start = bronze.index("darus")
        uncertain = _compact_mechanical_raw(
            selected,
            clean_text=bronze,
            uncertainties=[{
                "surface_text": "darus",
                "uncertainty_type": "ASR_ALTERNATIVES",
                "alternatives": [
                    {"text": "Darius", "confidence": "MEDIUM"},
                    {"text": "Varus", "confidence": "MEDIUM"},
                ],
                "note": "competing champion spellings; metadata says Varus",
            }],
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([uncertain]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["repairs"], [])
        self.assertEqual(result["mechanical_cleaned_text"], bronze)
        self.assertEqual(result["uncertainties"][0]["text"], "darus")
        self.assertEqual(
            result["uncertainties"][0]["target_local_start"], start,
        )
        self.assertEqual(
            [item["text"] for item in result["uncertainties"][0]["alternatives"]],
            ["Darius", "Varus"],
        )

    def test_version_lineage_and_cache_key_binding(self):
        self.assertEqual(
            MECHANICAL_PROMPT_VERSION,
            "phase2k-mechanical-cleanup-prompt-v4",
        )
        self.assertEqual(
            MECHANICAL_CORRECTION_PROMPT_VERSION,
            "phase2k-mechanical-cleanup-correction-prompt-v4",
        )
        self.assertEqual(
            MECHANICAL_RESPONSE_SCHEMA_VERSION,
            "phase2k-mechanical-cleanup-response-v3",
        )
        self.assertEqual(
            LEAGUE_VOCABULARY_SCHEMA_VERSION,
            "phase2k-league-lexical-vocabulary-v2",
        )
        self.assertEqual(
            PIPELINE_VERSION,
            "phase2k-contextual-reconstruction-v7",
        )
        self.assertEqual(
            CONFIG_VERSION,
            "phase2k-config-v3",
        )
        self.assertEqual(
            RECORDS_SCHEMA_VERSION,
            "phase2k-reconstruction-records-v7",
        )
        self.assertEqual(
            p2k_module.VOCABULARY_PATH.name,
            "league_lexical_vocabulary_v2.json",
        )
        # Cache keys bind the prompt version, so old v15 (v3) mechanical
        # responses can never be reused for the v4 prompt.
        v3_key = _cache_key(
            prompt_hash="p",
            inference_config_hash="i",
            schema_version=MECHANICAL_RESPONSE_SCHEMA_VERSION,
            prompt_version="phase2k-mechanical-cleanup-prompt-v3",
            attempt_index=0,
            attempt_kind="base",
        )
        v4_key = _cache_key(
            prompt_hash="p",
            inference_config_hash="i",
            schema_version=MECHANICAL_RESPONSE_SCHEMA_VERSION,
            prompt_version=MECHANICAL_PROMPT_VERSION,
            attempt_index=0,
            attempt_kind="base",
        )
        self.assertNotEqual(v3_key, v4_key)

        # A live mechanical run seals the v4 lineage and replays from cache.
        bronze = "versus Kale top"
        selected = _selected(bronze)
        hints = detect_champion_alias_hints(bronze, selected)
        raw = _mechanical_raw(
            selected, repairs=self._hint_repairs(selected, hints),
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            config_hash = canonical_sha256({"v": 1})
            first = run_mechanical_cleanup(
                selected,
                chat=CountingChat([raw]),
                config_hash=config_hash,
                cache_dir=cache_dir,
            )
            second = run_mechanical_cleanup(
                selected,
                chat=CountingChat([raw]),
                config_hash=config_hash,
                cache_dir=cache_dir,
            )
        self.assertEqual(
            first["model_call"]["prompt_version"],
            "phase2k-mechanical-cleanup-prompt-v4",
        )
        self.assertEqual(first["model_call"]["source"], "provider")
        self.assertEqual(second["model_call"]["source"], "cache")
        self.assertEqual(first["model_call"]["cache_key"], second["model_call"]["cache_key"])
        self.assertEqual(first["provenance"]["config_version"], "phase2k-config-v3")

    def test_tampered_vocabulary_fails_closed(self):
        original_path = p2k_module.VOCABULARY_PATH
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_schema = root / "wrong-schema.json"
            wrong_schema.write_text(
                json.dumps(
                    {
                        **load_json_strict(
                            original_path, label="vocabulary",
                        ),
                        "schema_version": "phase2k-league-lexical-vocabulary-v1",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            missing_rules = root / "missing-rules.json"
            vocabulary = load_json_strict(original_path, label="vocabulary")
            vocabulary.pop("champion_alias_rules")
            missing_rules.write_text(
                json.dumps(vocabulary, sort_keys=True),
                encoding="utf-8",
            )
            invalid_rule = root / "invalid-rule.json"
            vocabulary = load_json_strict(original_path, label="vocabulary")
            vocabulary["champion_alias_rules"]["never"].append({
                "alias": "not a word",
                "canonical": "Pyke",
                "policy": "never",
            })
            invalid_rule.write_text(
                json.dumps(vocabulary, sort_keys=True),
                encoding="utf-8",
            )
            try:
                p2k_module.VOCABULARY_PATH = wrong_schema
                with self.assertRaises(ValueError) as caught:
                    p2k_module.load_lexical_vocabulary()
                self.assertIn("schema version is invalid", str(caught.exception))

                p2k_module.VOCABULARY_PATH = missing_rules
                with self.assertRaises(ValueError) as caught:
                    p2k_module.load_lexical_vocabulary()
                self.assertIn("champion_alias_rules", str(caught.exception))

                p2k_module.VOCABULARY_PATH = invalid_rule
                with self.assertRaises(ValueError) as caught:
                    p2k_module.load_lexical_vocabulary()
                self.assertIn("alias must be a non-empty ASCII word", str(caught.exception))
            finally:
                p2k_module.VOCABULARY_PATH = original_path

    def test_semantic_fields_rejected_inside_domain_spelling_repair(self):
        bronze = "versus Kale top"
        selected = _selected(bronze)
        raw = _compact_mechanical_raw(
            selected,
            clean_text="versus Kayle top",
            repairs=[{
                "original_text": "Kale",
                "replacement": "Kayle",
                "repair_type": "DOMAIN_SPELLING",
                "confidence": "HIGH",
                "rationale": "listed lexical hint",
                "champion_binding": {"champion": "Kayle"},
            }],
        )
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    raw, raw, raw, raw,
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "semantic extraction field",
            caught.exception.attempts[0]["error"],
        )


if __name__ == "__main__":
    unittest.main()
