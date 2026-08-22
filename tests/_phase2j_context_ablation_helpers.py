"""Shared synthetic Phase 2J context-ablation fixtures (offline, stdlib-only).

This module is not collected by pytest.  It builds a schema-valid frozen
Phase 2J manifest/reviewed packet, a read-only-style SQLite transcript DB,
exact source-grounded extraction outputs, and human-attested review sets so
the focused test modules stay fast and offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.phase2j_context_ablation import (
    CONDITION_CODES,
    COMPLETED_REVIEWS_SCHEMA_VERSION,
    DIFFICULTY_WEIGHTS,
    OUTPUT_SCHEMA_VERSION,
    OUTPUTS_SCHEMA_VERSION,
    SEMANTIC_FIELDS,
    canonical_sha256,
    text_sha256,
)


ROLES = ("mid", "top", "jungle", "adc", "support")
CHAMPIONS = ("Lux", "Garen", "Ahri", "Jinx", "Lee Sin", "Viktor")
PARTITIONS = ("EXPANDED_DEV", "FROZEN_REPLICATION")

TAG_POOL = [
    "punctuation_poor", "omitted_actor", "pronoun", "multiple_abilities",
    "multiple_champions", "nested_condition", "cause_chain", "uncertainty",
    "contradiction", "implicit_cause", "explicit_cause", "conditional",
    "temporal", "contrast", "advice_explanation", "resource_exchange",
    "wave_reasoning", "multi_sentence",
]

# Deterministic tag sets per synthetic manifest window (30 windows).  The
# scores are intentionally varied so the expected top-10 ranking is stable.
WINDOW_TAG_SETS = [
    ["pronoun", "punctuation_poor", "omitted_actor", "multiple_abilities",
     "nested_condition", "cause_chain", "uncertainty", "contradiction",
     "implicit_cause", "explicit_cause", "conditional", "temporal",
     "contrast", "advice_explanation", "resource_exchange"],
    ["punctuation_poor", "omitted_actor", "pronoun", "multiple_abilities",
     "multiple_champions", "nested_condition", "cause_chain", "uncertainty",
     "contradiction", "implicit_cause", "explicit_cause", "conditional",
     "temporal", "contrast", "advice_explanation"],
    ["pronoun", "punctuation_poor", "multiple_abilities", "nested_condition",
     "cause_chain", "uncertainty", "contradiction", "explicit_cause",
     "conditional", "temporal", "contrast", "resource_exchange"],
    ["omitted_actor", "pronoun", "punctuation_poor", "uncertainty",
     "nested_condition", "cause_chain", "multiple_abilities", "conditional",
     "temporal", "contrast", "advice_explanation", "wave_reasoning",
     "multi_sentence"],
    ["pronoun", "punctuation_poor", "omitted_actor", "nested_condition",
     "uncertainty", "cause_chain", "contradiction", "conditional",
     "temporal", "contrast", "implicit_cause", "explicit_cause",
     "advice_explanation", "wave_reasoning", "multi_sentence"],
    ["punctuation_poor", "pronoun", "uncertainty", "nested_condition",
     "cause_chain", "contradiction", "conditional", "temporal", "contrast",
     "advice_explanation"],
    ["pronoun", "omitted_actor", "punctuation_poor", "uncertainty",
     "cause_chain", "contradiction", "conditional", "temporal", "contrast",
     "advice_explanation", "wave_reasoning", "multi_sentence"],
    ["pronoun", "punctuation_poor", "omitted_actor", "multiple_abilities",
     "nested_condition", "cause_chain", "uncertainty", "contradiction",
     "implicit_cause", "explicit_cause", "conditional", "temporal",
     "contrast", "advice_explanation", "resource_exchange", "multi_sentence"],
    ["pronoun", "punctuation_poor", "omitted_actor", "multiple_champions",
     "nested_condition", "cause_chain", "uncertainty", "contradiction",
     "implicit_cause", "explicit_cause", "conditional", "temporal",
     "contrast", "advice_explanation", "wave_reasoning", "multi_sentence"],
    ["punctuation_poor", "pronoun", "omitted_actor", "uncertainty",
     "cause_chain", "contradiction", "implicit_cause", "explicit_cause",
     "conditional", "temporal", "contrast", "advice_explanation",
     "resource_exchange", "wave_reasoning", "multi_sentence"],
    ["pronoun", "punctuation_poor", "uncertainty", "cause_chain",
     "contradiction", "conditional", "temporal", "contrast",
     "advice_explanation", "resource_exchange"],
    ["pronoun", "omitted_actor", "punctuation_poor", "uncertainty",
     "contradiction", "conditional", "temporal", "contrast",
     "advice_explanation", "multi_sentence"],
    ["pronoun", "punctuation_poor", "uncertainty", "nested_condition",
     "cause_chain", "conditional", "temporal", "contrast",
     "advice_explanation", "wave_reasoning"],
    ["pronoun", "punctuation_poor", "uncertainty", "cause_chain",
     "contradiction", "implicit_cause", "conditional", "temporal",
     "contrast", "advice_explanation", "resource_exchange", "multi_sentence"],
    ["pronoun", "punctuation_poor", "uncertainty", "nested_condition",
     "cause_chain", "conditional", "temporal", "contrast",
     "advice_explanation", "resource_exchange", "wave_reasoning"],
    ["pronoun", "punctuation_poor", "uncertainty", "contradiction",
     "conditional", "temporal", "contrast", "advice_explanation",
     "wave_reasoning", "multi_sentence"],
    ["pronoun", "punctuation_poor", "uncertainty", "cause_chain",
     "conditional", "temporal", "contrast", "resource_exchange"],
    ["pronoun", "punctuation_poor", "uncertainty", "conditional", "temporal",
     "contrast", "advice_explanation", "wave_reasoning"],
    ["pronoun", "punctuation_poor", "uncertainty", "contradiction",
     "conditional", "contrast", "advice_explanation", "multi_sentence"],
    ["pronoun", "punctuation_poor", "uncertainty", "conditional", "temporal",
     "contrast", "advice_explanation"],
    ["pronoun", "punctuation_poor", "uncertainty", "conditional", "contrast",
     "advice_explanation", "wave_reasoning"],
    ["pronoun", "punctuation_poor", "uncertainty", "conditional", "contrast",
     "resource_exchange", "multi_sentence"],
    ["pronoun", "punctuation_poor", "uncertainty", "conditional", "contrast",
     "advice_explanation"],
    ["pronoun", "punctuation_poor", "uncertainty", "conditional", "contrast"],
    ["pronoun", "punctuation_poor", "uncertainty", "conditional"],
    ["pronoun", "punctuation_poor", "uncertainty"],
    ["pronoun", "punctuation_poor"],
    ["pronoun"],
    [],
    ["wave_reasoning"],
]


def _transcript(source_id: str, index: int) -> str:
    words = (
        [f"word{index}-{i}" for i in range(1800)]
        + ["jinx"]
        + [f"tail{index}-{i}" for i in range(400)]
    )
    return " ".join(words)


def token_table(text: str) -> list[dict[str, Any]]:
    return [
        {
            "token_index": index,
            "start": match.start(),
            "end": match.end(),
            "text": match.group(),
        }
        for index, match in enumerate(re.finditer(r"\S+", text))
    ]


def make_manifest(transcripts: Mapping[str, str]) -> dict[str, Any]:
    selected = []
    for index in range(30):
        source_id = f"video{index:02d}"
        transcript = transcripts[source_id]
        start = 100 + index
        end = start + 220
        text = transcript[start:end]
        champion = CHAMPIONS[index % len(CHAMPIONS)]
        role = ROLES[index % len(ROLES)]
        video_title = f"Coach {champion} {role} {index}"
        window_id = f"pool:{source_id}:w{index:05d}-{hashlib.sha256(text.encode()).hexdigest()[:20]}"
        record = {
            "source_group_id": f"video:{source_id}",
            "window_id": window_id,
            "upstream_source_id": source_id,
            "upstream_start": start,
            "upstream_end": end,
            "source_text": text,
            "source_text_sha256": text_sha256(text),
            "upstream_content_sha256": text_sha256(transcript),
            "source_text_char_length": len(text),
            "metadata": {
                "champion": champion,
                "role": role,
                "video_title": video_title,
            },
            "phenomena": list(WINDOW_TAG_SETS[index]),
            "asr_punctuation_band": (
                "PUNCTUATION_POOR"
                if "punctuation_poor" in WINDOW_TAG_SETS[index]
                else "PUNCTUATED"
            ),
            "partition": PARTITIONS[index % len(PARTITIONS)],
            "candidate_generator_version": (
                "phase2f-mention-catalog-v3-cross-segment-ngrams-32"
            ),
            "candidate_count": 10 + index,
            "candidate_catalog_sha256": hashlib.sha256(
                f"catalog{index}".encode(),
            ).hexdigest(),
            "canonical_record_sha256": hashlib.sha256(
                f"canonical{index}".encode(),
            ).hexdigest(),
        }
        selected.append(record)
    manifest = {
        "schema_version": "phase2j-window-selection-manifest-v1",
        "purpose": "synthetic Phase 2J manifest for Phase 2J context-ablation tests",
        "release_gate": "LOCKED",
        "selection_policy": {"synthetic": True},
        "input_hashes": {},
        "legacy_source_exclusions": [],
        "selected": selected,
        "partition_counts": {},
        "diversity_summary": {},
        "candidate_generator_version": (
            "phase2f-mention-catalog-v3-cross-segment-ngrams-32"
        ),
        "checkpoint": "PRE_ANNOTATION_CHECKPOINT",
    }
    manifest = {
        **manifest,
        "content_sha256": canonical_sha256(manifest),
    }
    return manifest


def make_packet(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    records = []
    for index, selected in enumerate(manifest["selected"]):
        text = selected["source_text"]
        records.append({
            "record_index": index,
            "annotation_id": f"p2j:{selected['window_id']}",
            "source_group_id": selected["source_group_id"],
            "window_id": selected["window_id"],
            "upstream_source_id": selected["upstream_source_id"],
            "upstream_start": selected["upstream_start"],
            "upstream_end": selected["upstream_end"],
            "partition": selected["partition"],
            "bronze_text": text,
            "bronze_text_sha256": text_sha256(text),
            "bronze_char_length": len(text),
            "tokens": token_table(text),
            "endpoints": [
                {
                    "endpoint_id": f"p2j:{selected['window_id']}:ep:0001",
                    "bronze_text": text,
                    "char_start": 0,
                    "char_end": len(text),
                    "token_start": 0,
                    "token_end": len(token_table(text)) - 1,
                    "node_type": "ACTION",
                    "ambiguity_state": "NONE",
                    "disposition": "KEEP",
                    "adjudication_requested": False,
                    "pass_provenance": "PASS_B",
                    "notes": "synthetic",
                },
            ],
            "window_status": "REVIEWED",
            "pass_a": {},
            "pass_b": {},
            "ambiguity_controls": {"flagged": False, "notes": []},
            "exclusion_controls": {"flagged": False, "notes": []},
            "reviewer_notes": [],
        })
    packet = {
        "schema_version": "phase2j-endpoint-annotation-packet-v1",
        "purpose": "synthetic Phase 2J reviewed packet for tests",
        "annotation_version": "phase2j-endpoint-annotation-v1",
        "release_gate": "LOCKED",
        "selection_manifest_sha256": manifest["content_sha256"],
        "selection_manifest_schema_version": (
            "phase2j-window-selection-manifest-v1"
        ),
        "candidate_generator_version": (
            "phase2f-mention-catalog-v3-cross-segment-ngrams-32"
        ),
        "candidate_catalog": {},
        "rules": {},
        "records": records,
    }
    return {
        **packet,
        "content_sha256": canonical_sha256(packet),
    }


def make_transcript_db(
    root: Path,
    transcripts: Mapping[str, str],
) -> Path:
    db_path = root / "videos.db"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE videos ("
        "video_id TEXT PRIMARY KEY, video_url TEXT, video_title TEXT, "
        "description TEXT, role TEXT, champion TEXT, rank TEXT, "
        "message_timestamp TEXT, status TEXT, transcription TEXT, "
        "created_at TEXT, source TEXT, game TEXT, subject TEXT, "
        "website_rating TEXT)",
    )
    connection.execute(
        "CREATE TABLE champion_abilities ("
        "champion TEXT, ability_slot TEXT, name TEXT, description TEXT, "
        "cooldown TEXT, range TEXT, cost TEXT, properties TEXT)",
    )
    for index, (source_id, transcript) in enumerate(transcripts.items()):
        champion = CHAMPIONS[index % len(CHAMPIONS)]
        role = ROLES[index % len(ROLES)]
        connection.execute(
            "INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                f"https://www.youtube.com/watch?v={source_id}",
                f"Coach {champion} {role} {index}",
                "",
                role,
                champion,
                None,
                "",
                "analyzed",
                transcript,
                "",
                "youtube",
                "lol",
                "",
                "",
            ),
        )
    for champion in CHAMPIONS:
        for slot in ("P", "Q", "W", "E", "R"):
            connection.execute(
                "INSERT INTO champion_abilities VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    champion,
                    slot,
                    f"{champion} {slot} ability",
                    f"{champion} {slot} description",
                    "10",
                    "500",
                    "60",
                    "{}",
                ),
            )
    connection.commit()
    connection.close()
    return db_path


def expected_selected_case_ids() -> list[str]:
    rows = []
    for index, tags in enumerate(WINDOW_TAG_SETS):
        present = [tag for tag in tags if tag in DIFFICULTY_WEIGHTS]
        rows.append((
            index,
            sum(DIFFICULTY_WEIGHTS[tag] for tag in present),
            len(tags),
        ))
    rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
    return [f"p2ja:case:{rank:04d}" for rank, _ in enumerate(rows[:10], 1)]


def _payload_source(payload: Mapping[str, Any]) -> str:
    if payload["condition"] == "A":
        return payload["target"]["bronze_text"]
    return payload["transcript"]


def make_valid_output(
    payload: Mapping[str, Any],
    *,
    case_id: str,
    condition: str,
) -> dict[str, Any]:
    source = _payload_source(payload)
    first_token = re.search(r"\S+", source)
    if first_token is None:
        raise AssertionError("source has no tokens")
    quote = first_token.group()
    source_range = {
        "char_start": first_token.start(),
        "char_end": first_token.end(),
    }
    reference = {
        "quote": quote,
        "source_range": dict(source_range),
    }
    fields: dict[str, Any] = {}
    for field in SEMANTIC_FIELDS:
        fields[field] = []
    fields["actors"] = [{
        "item_id": f"{case_id}:{condition}:actors:0001",
        "extraction_text": "coach",
        "resolution_status": "literal_explicit",
        "source_references": [dict(reference)],
    }]
    fields["supporting_source_ranges"] = [{
        "item_id": f"{case_id}:{condition}:supporting_source_ranges:0001",
        "extraction_text": "target range",
        "resolution_status": "literal_explicit",
        "source_references": [dict(reference)],
        "source_range": dict(source_range),
    }]
    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": case_id,
        "condition": condition,
        "payload_sha256": payload["content_sha256"],
        "instructions_sha256": payload["instructions_sha256"],
        "fields": fields,
    }
    return {
        **output,
        "content_sha256": canonical_sha256(output),
    }


def make_outputs_bundle(
    payloads_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    cases = []
    for payload_case in payloads_artifact["cases"]:
        case_id = payload_case["case_id"]
        cases.append({
            "case_id": case_id,
            "A": make_valid_output(payload_case["A"], case_id=case_id, condition="A"),
            "B": make_valid_output(payload_case["B"], case_id=case_id, condition="B"),
        })
    bundle = {
        "schema_version": OUTPUTS_SCHEMA_VERSION,
        "purpose": "synthetic validated outputs for tests",
        "release_gate": "LOCKED",
        "payloads_sha256": payloads_artifact["content_sha256"],
        "instructions_sha256": payloads_artifact["instructions_sha256"],
        "cases": cases,
    }
    return {
        **bundle,
        "content_sha256": canonical_sha256(bundle),
    }


def make_completed_reviews(
    packet: Mapping[str, Any],
    *,
    success_by_item: Mapping[str, bool],
    major_by_item: set[str] | None = None,
    reviewer: str = "tester",
    completed_at: str = "2026-08-20T00:00:00Z",
    reviewer_kind: str = "human",
    human_review_attested: bool = True,
    attestation_statement: str = (
        "I attest that I personally reviewed every blinded extraction item "
        "in the packet against the supplied source evidence."
    ),
) -> dict[str, Any]:
    major_by_item = major_by_item or set()
    reviews = {}
    for item in packet["review_items"]:
        review_item_id = item["review_item_id"]
        success = success_by_item.get(review_item_id, False)
        if success:
            correctness = "CORRECT"
            unsupported = "NONE"
            grounding = "GROUNDED"
        else:
            correctness = "INCORRECT"
            unsupported = "MAJOR" if review_item_id in major_by_item else "NONE"
            grounding = "UNGROUNDED"
        reviews[review_item_id] = {
            "correctness": correctness,
            "unsupported_inference": unsupported,
            "source_grounding": grounding,
            "notes": [],
        }
    completed = {
        "schema_version": COMPLETED_REVIEWS_SCHEMA_VERSION,
        "reviewer_kind": reviewer_kind,
        "human_review_attested": human_review_attested,
        "attestation_statement": attestation_statement,
        "reviewer": reviewer,
        "completed_at": completed_at,
        "reviews": reviews,
    }
    return {
        **completed,
        "content_sha256": canonical_sha256(completed),
    }


def build_fixture(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    """Build manifest/packet/DB and return their paths plus transcripts."""
    transcripts = {
        f"video{index:02d}": _transcript(f"video{index:02d}", index)
        for index in range(30)
    }
    manifest = make_manifest(transcripts)
    packet = make_packet(manifest)
    manifest_path = root / "manifest.json"
    packet_path = root / "packet.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    db_path = make_transcript_db(root, transcripts)
    return manifest_path, packet_path, db_path, transcripts
