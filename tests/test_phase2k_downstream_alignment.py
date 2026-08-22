"""Focused tests for the Phase 2K downstream semantic-target alignment."""

from __future__ import annotations

import copy
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

import pipeline.phase2k_downstream_alignment as alignment_module
import scripts.build_phase2k_downstream_alignment as build_cli
import scripts.finalize_phase2k_downstream_alignment as finalize_cli
from pipeline.phase2j_candidate_coverage import (
    MENTION_CATALOG_VERSION,
    load_candidate_coverage,
)
from pipeline.phase2k_contextual_reconstruction import (
    OUTPUT_FILENAMES,
    canonical_sha256,
    file_sha256,
    load_json_strict,
    text_sha256,
    validate_completed_transformation_audits,
    validate_output_directory,
)
from pipeline.phase2k_downstream_alignment import (
    ALIGNMENT_DECISION_STATES,
    ALIGNMENT_PACKET_SCHEMA_VERSION,
    BOUNDARY_RULE_VERSION,
    TARGET_COUNT,
    TARGET_WINDOW_COUNT,
    build_alignment_summary,
    build_boundary_manifest,
    build_downstream_alignment_packet,
    finalize_downstream_alignment_packet,
    validate_alignment_summary,
    validate_downstream_alignment_packet,
    _scan_forbidden_leaks,
)
from pipeline.semantic_mentions import NODE_TYPES
from tests._phase2k_helpers import (
    CHAMPIONS,
    PARTITIONS,
    ROLES,
    make_packet_record,
    make_selected,
    token_table,
)
from tests.test_phase2k_contracts import (
    complete_reviews,
    finalize_live_state,
    passing_reviews,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_REVIEWED_PACKET = ROOT / "data/phase2j/reviewed-endpoint-annotation-packet-v1.json"
REAL_COVERAGE = ROOT / "data/phase2j/candidate-coverage-v1.json"
NULL_NODE_ENDPOINT = (
    "p2j:pool:MjHLNnOPgn8:w00190-ad0cc2adb93f3e63133b:ep:0008"
)

ALIGN_WINDOW_TEXT = (
    "You should push the wave now, then recall, and reset. "
    "Your opponent is low on mana, so wait. "
    "Stay behind the minions, use your cooldowns, and flash out if needed."
)

_NODE_TYPE_CYCLE = (
    "ENTITY", "ABILITY_OR_RESOURCE", "EVENT", "ACTION", "STATE", "OUTCOME",
    "QUANTITY", "TIME", "LOCATION_OR_SPACE",
)


def _span_pool(
    text: str,
    *,
    terminal: str | None = None,
) -> list[tuple[int, int, str]]:
    if terminal is not None:
        spans: list[tuple[int, int, str]] = []
        for position, char in enumerate(text):
            if char != terminal:
                continue
            starts = [
                index
                for index in range(position + 1)
                if index == 0 or text[index - 1].isspace()
            ]
            for start in starts:
                spans.append((start, position + 1, text[start:position + 1]))
        return spans
    spans = []
    tokens = [(match.start(), match.end()) for match in __import__("re").finditer(
        r"\S+", text,
    )]
    for start, end in tokens:
        spans.append((start, end, text[start:end]))
    for width in (2, 3):
        for index in range(len(tokens) - width + 1):
            start = tokens[index][0]
            end = tokens[index + width - 1][1]
            spans.append((start, end, text[start:end]))
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int, str]] = []
    for start, end, span_text in spans:
        if (start, end) in seen:
            continue
        seen.add((start, end))
        unique.append((start, end, span_text))
    return unique


def _token_indices(text: str, start: int, end: int) -> tuple[int, int]:
    tokens = token_table(text)
    token_start = 0
    token_end = 0
    for index, token in enumerate(tokens):
        if token["end"] > start:
            token_start = index
            break
    for index, token in enumerate(tokens):
        if token["start"] < end:
            token_end = index
    return token_start, token_end


def _window_plan(window_index: int) -> tuple[int, int, str | None]:
    """window_index is 1-based; returns (total, missing, terminal)."""
    total = 11 if window_index <= 11 else 10
    if window_index <= 14:
        return total, 2, "."
    if window_index <= 24:
        return total, 2, ","
    return total, 0, None


def _window_endpoints(
    selected: Mapping[str, Any],
    window_index: int,
) -> list[dict[str, Any]]:
    text = selected["source_text"]
    total, missing, terminal = _window_plan(window_index)
    period_pool = _span_pool(text, terminal=".")
    comma_pool = _span_pool(text, terminal=",")
    covered_pool = _span_pool(text)
    endpoints: list[dict[str, Any]] = []
    missing_positions = {total - 2, total - 1} if missing else set()
    pool = period_pool if terminal == "." else comma_pool
    covered_index = (window_index * 5) % max(1, len(covered_pool))
    missing_index = (window_index * 7) % max(1, len(pool))
    for position in range(total):
        is_missing = position in missing_positions
        if is_missing:
            start, end, span_text = pool[missing_index % len(pool)]
            missing_index += 1
        else:
            start, end, span_text = covered_pool[covered_index % len(covered_pool)]
            covered_index += 1
        token_start, token_end = _token_indices(text, start, end)
        node_type = _NODE_TYPE_CYCLE[
            (window_index + position) % len(_NODE_TYPE_CYCLE)
        ]
        if window_index == 1 and position == total - 1:
            node_type = None
        endpoints.append({
            "endpoint_id": (
                f"p2j:{selected['window_id']}:ep:{position + 1:04d}"
            ),
            "bronze_text": span_text,
            "char_start": start,
            "char_end": end,
            "token_start": token_start,
            "token_end": token_end,
            "node_type": node_type,
            "ambiguity_state": "NONE",
            "disposition": "KEEP",
            "adjudication_requested": False,
            "notes": "synthetic Phase 2K alignment test endpoint",
            "pass_provenance": "PASS_A" if position % 2 == 0 else "PASS_B",
        })
    return endpoints


def _rich_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    """30-window Phase 2J fixture with 311 endpoints and a coverage artifact."""
    transcripts = {
        f"s{index:02d}": (
            "Intro sentence with extra tokens. " + ALIGN_WINDOW_TEXT
            + " Outro sentence with more tokens."
        )
        for index in range(1, 31)
    }
    source_ids = sorted(transcripts)
    selected = []
    for index, source_id in enumerate(source_ids, 1):
        transcript = transcripts[source_id]
        champion = CHAMPIONS[index % len(CHAMPIONS)]
        role = ROLES[index % len(ROLES)]
        video_title = f"Video {source_id} {champion}"
        start = transcript.index(ALIGN_WINDOW_TEXT)
        end = start + len(ALIGN_WINDOW_TEXT)
        partition = PARTITIONS[1 if index > 24 else 0]
        selected.append(make_selected(
            source_id,
            transcript,
            start,
            end,
            index=index,
            champion=champion,
            role=role,
            video_title=video_title,
            partition=partition,
        ))
    manifest = {
        "schema_version": "phase2j-window-selection-manifest-v1",
        "purpose": "Synthetic Phase 2K alignment test fixture; not real Phase 2J data.",
        "release_gate": "LOCKED",
        "selection_policy": {
            "seed": "test",
            "target_window_count": len(selected),
            "target_distinct_video_source_groups": len(selected),
            "one_window_per_upstream_source": True,
            "source_group_id_rule": "video:<id>",
            "eligibility": ["synthetic"],
            "diversity_score": {
                "phenomenon_points": 8,
                "phenomenon_below_count": 2,
                "role_points": 4,
                "role_below_count": 2,
                "asr_band_points": 2,
                "asr_band_below_count": 3,
                "unrepresented_champion_points": 1,
                "asr_band_definition": "synthetic",
                "tie_break": "seed",
            },
            "partition": {
                "order": "EXPANDED_DEV first",
                "EXPANDED_DEV": 24,
                "FROZEN_REPLICATION": 6,
            },
            "diversity_preference_statement": "synthetic",
        },
        "input_hashes": {
            "legacy_benchmark_content_sha256": "0" * 64,
            "legacy_benchmark_file_sha256": "0" * 64,
            "legacy_manifest_content_sha256": "0" * 64,
            "legacy_manifest_file_sha256": "0" * 64,
            "pool_content_sha256": "0" * 64,
            "pool_file_sha256": "0" * 64,
        },
        "legacy_source_exclusions": ["legacy:1", "legacy:2", "legacy:3"],
        "selected": selected,
        "partition_counts": {
            "EXPANDED_DEV": sum(
                1 for item in selected if item["partition"] == "EXPANDED_DEV"
            ),
            "FROZEN_REPLICATION": sum(
                1 for item in selected if item["partition"] == "FROZEN_REPLICATION"
            ),
        },
        "diversity_summary": {
            "phenomenon_counts": {"pronoun": len(selected)},
            "role_counts": {role: 0 for role in ROLES},
            "asr_punctuation_band_counts": {
                "PUNCTUATED": 0,
                "PUNCTUATION_POOR": len(selected),
            },
            "champion_counts": {champion: 0 for champion in CHAMPIONS},
            "distinct_champions": len(CHAMPIONS),
            "candidate_count": 7 * len(selected),
        },
        "candidate_generator_version": (
            "phase2f-mention-catalog-v3-cross-segment-ngrams-32"
        ),
        "checkpoint": "PRE_ANNOTATION_CHECKPOINT",
    }
    manifest = {"content_sha256": canonical_sha256(manifest), **manifest}
    records = []
    for index, item in enumerate(selected, 1):
        record = make_packet_record(item, index=index)
        record["endpoints"] = _window_endpoints(item, index)
        records.append(record)
    packet = {
        "schema_version": "phase2j-endpoint-annotation-packet-v1",
        "purpose": "Synthetic Phase 2K alignment test packet; not real Phase 2J data.",
        "annotation_version": "phase2j-endpoint-annotation-v1",
        "release_gate": "LOCKED",
        "selection_manifest_sha256": manifest["content_sha256"],
        "selection_manifest_schema_version": "phase2j-window-selection-manifest-v1",
        "candidate_generator_version": (
            "phase2f-mention-catalog-v3-cross-segment-ngrams-32"
        ),
        "candidate_catalog": {
            "count": 7 * len(selected),
            "per_window": {
                item["window_id"]: {"count": 7, "catalog_sha256": "0" * 64}
                for item in selected
            },
        },
        "rules": {
            "window_statuses": ["REVIEWED"],
            "endpoint_dispositions": ["KEEP"],
            "pass_a": "synthetic",
            "pass_b": "synthetic",
            "in_review_rule": "synthetic",
            "pass_b_requires_pass_a": True,
            "gold_eligibility_rule": "synthetic",
            "overlap_rule": "synthetic",
            "non_keep_rule": "synthetic",
            "reviewer_instructions": "synthetic",
        },
        "records": records,
    }
    packet = {"content_sha256": canonical_sha256(packet), **packet}

    manifest_path = root / "window-selection-manifest-v1.json"
    packet_path = root / "reviewed-endpoint-annotation-packet-v1.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    packet_path.write_text(
        json.dumps(packet, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    coverage_path = root / "candidate-coverage-v1.json"
    coverage_path.write_text(
        json.dumps(
            _coverage_artifact(manifest, packet, manifest_path, packet_path),
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    db_path = root / "videos.db"
    connection = __import__("sqlite3").connect(db_path)
    connection.execute(
        "CREATE TABLE videos ("
        "video_id TEXT PRIMARY KEY, video_url TEXT, video_title TEXT, "
        "description TEXT, role TEXT, champion TEXT, rank TEXT, "
        "message_timestamp TEXT, status TEXT, transcription TEXT, "
        "created_at TEXT, source TEXT, game TEXT, subject TEXT, "
        "website_rating REAL)",
    )
    for index, (source_id, transcript) in enumerate(
        sorted(transcripts.items()), 1,
    ):
        connection.execute(
            "INSERT INTO videos (video_id, video_url, video_title, role, "
            "champion, transcription, game) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                f"https://example.test/{source_id}",
                f"Video {source_id}",
                ROLES[index % len(ROLES)],
                CHAMPIONS[index % len(CHAMPIONS)],
                transcript,
                "lol",
            ),
        )
    connection.commit()
    connection.close()
    return manifest_path, packet_path, db_path, coverage_path


def _coverage_record(
    *,
    endpoint: Mapping[str, Any],
    record: Mapping[str, Any],
    selected: Mapping[str, Any],
    is_missing: bool,
    ordinal: int,
) -> dict[str, Any]:
    window_id = record["window_id"]
    source_id = selected["upstream_source_id"]
    base = {
        "endpoint_id": endpoint["endpoint_id"],
        "window_id": window_id,
        "source_group_id": record["source_group_id"],
        "partition": record["partition"],
        "role": selected["metadata"]["role"],
        "node_type": endpoint["node_type"],
        "char_start": endpoint["char_start"],
        "char_end": endpoint["char_end"],
        "absolute_start": record["upstream_start"] + endpoint["char_start"],
        "absolute_end": record["upstream_start"] + endpoint["char_end"],
        "bronze_text": endpoint["bronze_text"],
    }
    if is_missing:
        candidate = {
            "candidate_id": f"transcript:{source_id}:c:{ordinal:05d}",
            "candidate_alias": f"C{ordinal:04d}",
            "start": endpoint["char_start"],
            "end": endpoint["char_end"] - 1,
            "absolute_start": (
                record["upstream_start"] + endpoint["char_start"]
            ),
            "absolute_end": (
                record["upstream_start"] + endpoint["char_end"] - 1
            ),
            "text": endpoint["bronze_text"][:-1],
        }
        return {
            **base,
            "error_code": "CANDIDATE_GENERATION_MISS",
            "failure_category": "MIXED_BOUNDARY_MISMATCH",
            "overlap_count": 1,
            "overlaps": [candidate],
        }
    return {
        **base,
        "candidate_id": f"transcript:{source_id}:c:{ordinal:05d}",
        "candidate_alias": f"C{ordinal:04d}",
        "candidate_window_id": f"transcript:{source_id}:w00001",
        "candidate_segment_ids": [f"transcript:{source_id}:s001"],
        "candidate_catalog_sha256": "ab" * 32,
        "candidate_generator_version": MENTION_CATALOG_VERSION,
    }


def _coverage_artifact(
    manifest: Mapping[str, Any],
    packet: Mapping[str, Any],
    manifest_path: Path,
    packet_path: Path,
) -> dict[str, Any]:
    selected_by_window = {
        item["window_id"]: item for item in manifest["selected"]
    }
    covered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    per_window: dict[str, dict[str, Any]] = {}
    ordinal = 0
    window_gold: dict[str, int] = {}
    window_hit: dict[str, int] = {}
    partition_hit = {key: 0 for key in PARTITIONS}
    partition_denominator = {key: 0 for key in PARTITIONS}
    source_group_hit: dict[str, int] = {}
    source_group_denominator: dict[str, int] = {}
    node_type_hit: dict[str, int] = {}
    node_type_denominator: dict[str, int] = {}
    role_hit: dict[str, int] = {}
    role_denominator: dict[str, int] = {}
    for record_index, record in enumerate(packet["records"], 1):
        window_id = record["window_id"]
        selected = selected_by_window[window_id]
        _, missing_count, _terminal = _window_plan(record_index)
        missing_positions = {
            len(record["endpoints"]) - 2, len(record["endpoints"]) - 1,
        } if missing_count else set()
        for position, endpoint in enumerate(record["endpoints"]):
            is_missing = position in missing_positions
            coverage_record = _coverage_record(
                endpoint=endpoint,
                record=record,
                selected=selected,
                is_missing=is_missing,
                ordinal=ordinal,
            )
            ordinal += 1
            window_gold[window_id] = window_gold.get(window_id, 0) + 1
            partition_denominator[record["partition"]] += 1
            source_group_denominator[record["source_group_id"]] = (
                source_group_denominator.get(record["source_group_id"], 0) + 1
            )
            node_key = (
                endpoint["node_type"]
                if endpoint["node_type"] is not None
                else "null"
            )
            role_key = selected["metadata"]["role"] or "none"
            node_type_denominator[node_key] = (
                node_type_denominator.get(node_key, 0) + 1
            )
            role_denominator[role_key] = role_denominator.get(role_key, 0) + 1
            if is_missing:
                missing.append(coverage_record)
            else:
                covered.append(coverage_record)
                window_hit[window_id] = window_hit.get(window_id, 0) + 1
                partition_hit[record["partition"]] += 1
                source_group_hit[record["source_group_id"]] = (
                    source_group_hit.get(record["source_group_id"], 0) + 1
                )
                node_type_hit[node_key] = node_type_hit.get(node_key, 0) + 1
                role_hit[role_key] = role_hit.get(role_key, 0) + 1
    candidate_total = 0
    for record_index, record in enumerate(packet["records"], 1):
        window_id = record["window_id"]
        gold = window_gold[window_id]
        hit = window_hit.get(window_id, 0)
        candidate_count = 7 + (record_index % 5)
        candidate_total += candidate_count
        per_window[window_id] = {
            "candidate_count": candidate_count,
            "gold_count": gold,
            "hit_count": hit,
            "rate": hit / gold,
        }

    def metric(hit: int, denominator: int) -> dict[str, Any]:
        return {
            "hit_count": int(hit),
            "denominator": int(denominator),
            "rate": hit / denominator if denominator else 0.0,
        }

    body = {
        "schema_version": "phase2j-candidate-coverage-v1",
        "purpose": (
            "Synthetic Phase 2K alignment coverage fixture; discovery "
            "coverage only, no model scoring, predictions, thresholds, "
            "ranks, labels, or error taxonomy."
        ),
        "release_gate": "LOCKED",
        "checkpoint": "CANDIDATE_COVERAGE_GATE",
        "candidate_generator_version": MENTION_CATALOG_VERSION,
        "selection_manifest": {
            "file_sha256": file_sha256(manifest_path),
            "content_sha256": manifest["content_sha256"],
        },
        "reviewed_packet": {
            "file_sha256": file_sha256(packet_path),
            "content_sha256": packet["content_sha256"],
        },
        "scoring_absence": {
            "model_scoring": "ABSENT",
            "model_predictions": "ABSENT",
            "thresholds": "ABSENT",
            "statement": (
                "Synthetic discovery coverage only; no model scoring, "
                "predictions, thresholds, ranks, labels, or error taxonomy."
            ),
        },
        "coverage": {
            "aggregate": metric(len(covered), len(covered) + len(missing)),
            "total_candidates": candidate_total,
            "per_partition": {
                key: metric(partition_hit[key], partition_denominator[key])
                for key in PARTITIONS
            },
            "per_source_group": {
                key: metric(
                    source_group_hit.get(key, 0),
                    source_group_denominator[key],
                )
                for key in sorted(source_group_denominator)
            },
            "per_node_type": {
                key: metric(node_type_hit.get(key, 0), node_type_denominator[key])
                for key in sorted(node_type_denominator)
            },
            "per_role": {
                key: metric(role_hit.get(key, 0), role_denominator[key])
                for key in sorted(role_denominator)
            },
            "per_window": per_window,
            "node_type_key_rule": (
                "node_type values are used as keys; the None node_type is "
                "keyed as 'null'."
            ),
            "role_key_rule": (
                "manifest metadata role values are used as keys; an empty "
                "role is keyed as 'none'."
            ),
        },
        "covered_endpoints": covered,
        "missing_endpoints": missing,
    }
    return {
        "content_sha256": canonical_sha256({
            key: value for key, value in body.items() if key != "content_sha256"
        }),
        **body,
    }


def _shared_state() -> dict[str, Any]:
    if not hasattr(_shared_state, "value"):
        root = Path(tempfile.mkdtemp(prefix="phase2k-align-"))
        manifest_path, packet_path, db_path, coverage_path = _rich_fixture(root)
        state = finalize_live_state(
            root,
            manifest_path=manifest_path,
            packet_path=packet_path,
            db_path=db_path,
        )
        state["coverage_path"] = coverage_path
        state["root"] = root
        _shared_state.value = state
    return _shared_state.value


def _blank_packet() -> dict[str, Any]:
    state = _shared_state()
    return build_downstream_alignment_packet(
        phase2k_dir=state["output"],
        reviewed_packet_path=state["packet_path"],
        coverage_path=state["coverage_path"],
    )


def _default_decisions(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic valid decisions with no cross-target span collisions."""
    decisions: dict[str, Any] = {}
    local_by_window: dict[str, int] = {}
    for item in packet["items"]:
        window_id = item["window_id"]
        local = local_by_window.get(window_id, 0)
        local_by_window[window_id] = local + 1
        text = item["representation"]["polished_text"]
        state = (
            "ABSENT", "ALIGNED", "AMBIGUOUS", "ALIGNED",
            "MULTIPLE_CANDIDATES", "ALIGNED",
        )[local % 6]
        primary_start = 1 + local * 5
        primary = {
            "start": primary_start,
            "end": primary_start + 2,
            "text": text[primary_start:primary_start + 2],
        }
        spans = [primary]
        if state == "ABSENT":
            spans = []
        elif state == "MULTIPLE_CANDIDATES":
            secondary_start = 60 + local * 5
            spans = [
                primary,
                {
                    "start": secondary_start,
                    "end": secondary_start + 2,
                    "text": text[secondary_start:secondary_start + 2],
                },
            ]
        decisions[item["alignment_id"]] = {
            "state": state,
            "polished_spans": spans,
            "reviewer": "human",
            "completed_at": "2026-08-19T00:00:00.000Z",
            "notes": [],
        }
    return decisions


def _mutate_records(
    output_dir: Path,
    *,
    mutation: str,
) -> None:
    records_path = output_dir / OUTPUT_FILENAMES["records"]
    records = load_json_strict(records_path, label="records copy")
    target = None
    for record in records["records"]:
        if record["record_type"] == "D":
            target = record
            break
    content = target["content"]
    if mutation == "not_generated":
        content["generation_status"] = "NOT_GENERATED"
    elif mutation == "placeholder":
        content["is_placeholder"] = True
    elif mutation == "missing_polish":
        content["semantic_polish"] = None
    elif mutation == "stale":
        content["clean_target_transcript"] = content["clean_target_transcript"] + " X"
        content["clean_target_transcript_sha256"] = text_sha256(
            content["clean_target_transcript"],
        )
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    target["canonical_record_sha256"] = canonical_sha256({
        key: value for key, value in target.items()
        if key != "canonical_record_sha256"
    })
    records["content_sha256"] = canonical_sha256({
        key: value for key, value in records.items()
        if key != "content_sha256"
    })
    records_path.write_text(
        json.dumps(records, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )


def _stale_inference_config_version(output_dir: Path) -> None:
    """Make records/build-summary/audits consistently stale on the config
    schema version, rebinding every hash so the output stays internally and
    audit bound.  Only the current-contract deep validation can reject it."""
    records_path = output_dir / OUTPUT_FILENAMES["records"]
    records = load_json_strict(records_path, label="records copy")
    records["inference_config_version"] = "phase2k-inference-config-v1"
    records["content_sha256"] = canonical_sha256({
        key: value for key, value in records.items()
        if key != "content_sha256"
    })
    records_path.write_text(
        json.dumps(records, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )

    audit_path = output_dir / OUTPUT_FILENAMES["transformation_audit"]
    audit = load_json_strict(audit_path, label="audit copy")
    audit["binding"]["records_sha256"] = records["content_sha256"]
    audit["content_sha256"] = canonical_sha256({
        key: value for key, value in audit.items() if key != "content_sha256"
    })
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )

    completed_path = output_dir / OUTPUT_FILENAMES[
        "finalized_transformation_audit"
    ]
    completed = load_json_strict(completed_path, label="completed audit copy")
    completed["binding"]["records_sha256"] = records["content_sha256"]
    completed["content_sha256"] = canonical_sha256({
        key: value for key, value in completed.items()
        if key != "content_sha256"
    })
    completed_path.write_text(
        json.dumps(completed, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_path = output_dir / OUTPUT_FILENAMES["build_summary"]
    summary = load_json_strict(summary_path, label="build summary copy")
    summary["records_sha256"] = records["content_sha256"]
    summary["transformation_audit_sha256"] = audit["content_sha256"]
    summary["inference_config_version"] = "phase2k-inference-config-v1"
    summary["content_sha256"] = canonical_sha256({
        key: value for key, value in summary.items()
        if key != "content_sha256"
    })
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )


class RealArtifactBoundaryTests(unittest.TestCase):
    def test_real_311_boundary_manifest_and_null_node_type(self):
        from pipeline.phase2k_contextual_reconstruction import (
            load_phase2j_reviewed_packet,
        )
        packet = load_phase2j_reviewed_packet(REAL_REVIEWED_PACKET)
        coverage = load_candidate_coverage(REAL_COVERAGE)
        manifest = build_boundary_manifest(packet, coverage)
        self.assertEqual(len(manifest), 311)
        self.assertEqual(len(packet["records"]), 30)
        self.assertEqual(len(coverage["covered_endpoints"]), 263)
        self.assertEqual(len(coverage["missing_endpoints"]), 48)
        statuses = {
            entry["correction_status"] for entry in manifest.values()
        }
        self.assertEqual(statuses, {"UNCHANGED", "TERMINAL_PUNCTUATION_DROPPED"})
        unchanged = sum(
            1 for entry in manifest.values()
            if entry["correction_status"] == "UNCHANGED"
        )
        corrected = sum(
            1 for entry in manifest.values()
            if entry["correction_status"] == "TERMINAL_PUNCTUATION_DROPPED"
        )
        periods = sum(
            1 for entry in manifest.values() if entry["dropped_text"] == "."
        )
        commas = sum(
            1 for entry in manifest.values() if entry["dropped_text"] == ","
        )
        self.assertEqual((unchanged, corrected), (263, 48))
        self.assertEqual((periods, commas), (28, 20))
        null_entry = manifest[NULL_NODE_ENDPOINT]
        self.assertEqual(null_entry["correction_status"], "UNCHANGED")
        self.assertEqual(null_entry["original_text"], "really long range")
        self.assertEqual(null_entry["original_start"], 149)
        self.assertEqual(null_entry["original_end"], 166)
        for endpoint_id, entry in manifest.items():
            if entry["correction_status"] == "TERMINAL_PUNCTUATION_DROPPED":
                self.assertEqual(
                    entry["evaluation_end"], entry["original_end"] - 1,
                )
                self.assertEqual(
                    entry["evaluation_text"], entry["original_text"][:-1],
                )
                self.assertIn(entry["dropped_text"], (".", ","))

    def test_real_artifact_immutability(self):
        from pipeline.phase2k_contextual_reconstruction import (
            load_phase2j_reviewed_packet,
        )
        packet_before = REAL_REVIEWED_PACKET.read_bytes()
        coverage_before = REAL_COVERAGE.read_bytes()
        packet = load_phase2j_reviewed_packet(REAL_REVIEWED_PACKET)
        coverage = load_candidate_coverage(REAL_COVERAGE)
        build_boundary_manifest(packet, coverage)
        self.assertEqual(REAL_REVIEWED_PACKET.read_bytes(), packet_before)
        self.assertEqual(REAL_COVERAGE.read_bytes(), coverage_before)


class AlignmentBuilderTests(unittest.TestCase):
    def test_builder_builds_blank_packet_from_finalized_live_output(self):
        state = _shared_state()
        packet = _blank_packet()
        self.assertEqual(
            packet["schema_version"], ALIGNMENT_PACKET_SCHEMA_VERSION,
        )
        self.assertEqual(packet["release_gate"], "AWAITING_HUMAN_REVIEW")
        self.assertEqual(
            set(packet),
            {
                "schema_version", "content_sha256", "purpose", "release_gate",
                "dataset_binding", "boundary_rule", "items",
            },
        )
        self.assertEqual(len(packet["items"]), 311)
        self.assertEqual(
            packet["dataset_binding"]["target_count"], 311,
        )
        self.assertEqual(
            packet["dataset_binding"]["window_count"], 30,
        )
        self.assertEqual(
            packet["dataset_binding"]["human_review_gate_status"], "PASSED",
        )
        self.assertEqual(
            packet["dataset_binding"]["phase2k_records_sha256"],
            state["records"]["content_sha256"],
        )
        self.assertEqual(
            packet["boundary_rule"]["rule_version"], BOUNDARY_RULE_VERSION,
        )
        self.assertEqual(packet["boundary_rule"]["unchanged_count"], 263)
        self.assertEqual(packet["boundary_rule"]["corrected_count"], 48)
        self.assertEqual(
            packet["boundary_rule"]["dropped_terminal_period_count"], 28,
        )
        self.assertEqual(
            packet["boundary_rule"]["dropped_terminal_comma_count"], 20,
        )
        null_items = [item for item in packet["items"] if item["node_type"] is None]
        self.assertEqual(len(null_items), 1)
        # Representation must equal the sealed D record for every window.
        for item in packet["items"]:
            d_record = next(
                record
                for record in state["records"]["records"]
                if record["window_id"] == item["window_id"]
                and record["record_type"] == "D"
            )
            content = d_record["content"]
            self.assertEqual(
                item["representation"]["clean_target_transcript"],
                content["clean_target_transcript"],
            )
            self.assertEqual(
                item["representation"]["polished_text"],
                content["semantic_polish"]["polished_text"],
            )
            self.assertEqual(
                item["representation"]["polished_text_sha256"],
                text_sha256(content["semantic_polish"]["polished_text"]),
            )
            self.assertEqual(item["decision"], {
                "state": None,
                "polished_spans": [],
                "reviewer": None,
                "completed_at": None,
                "notes": [],
            })
        # Every item carries the exact canonical item key set.
        for item in packet["items"]:
            self.assertEqual(
                set(item),
                {
                    "alignment_id", "window_id", "endpoint_id", "node_type",
                    "bronze_target", "representation", "decision",
                },
            )
            self.assertEqual(
                set(item["bronze_target"]),
                {
                    "original_start", "original_end", "original_text",
                    "source_absolute_start", "source_absolute_end",
                    "evaluation_start", "evaluation_end", "evaluation_text",
                    "correction_status", "dropped_text",
                },
            )
            self.assertEqual(
                set(item["representation"]),
                {
                    "clean_target_transcript",
                    "clean_target_transcript_sha256",
                    "polished_text",
                    "polished_text_sha256",
                },
            )
            self.assertEqual(
                set(item["decision"]),
                {"state", "polished_spans", "reviewer", "completed_at", "notes"},
            )
        # Exact endpoint identity across raw Bronze and polished input.
        endpoint_ids = {item["endpoint_id"] for item in packet["items"]}
        self.assertEqual(len(endpoint_ids), 311)
        from pipeline.phase2k_contextual_reconstruction import (
            load_phase2j_reviewed_packet,
        )
        reviewed = load_phase2j_reviewed_packet(state["packet_path"])
        packet_endpoint_ids = {
            endpoint["endpoint_id"]
            for record in reviewed["records"]
            for endpoint in record["endpoints"]
            if endpoint["disposition"] == "KEEP"
        }
        self.assertEqual(endpoint_ids, packet_endpoint_ids)

    def test_builder_rejects_no_provider_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path, coverage_path = _rich_fixture(root)
            from pipeline.phase2k_contextual_reconstruction import (
                build_phase2k_outputs,
            )
            output = root / "no-provider"
            build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="no_provider",
            )
            with self.assertRaises(ValueError) as caught:
                build_downstream_alignment_packet(
                    phase2k_dir=output,
                    reviewed_packet_path=packet_path,
                    coverage_path=coverage_path,
                )
            self.assertIn("no-provider", str(caught.exception))

    def test_builder_rejects_placeholder_not_generated_missing_polish(self):
        state = _shared_state()
        for mutation, expected in (
            ("not_generated", "not GENERATED"),
            ("placeholder", "placeholder"),
            ("missing_polish", "missing a sealed"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    output = root / "copy"
                    shutil.copytree(state["output"], output)
                    _mutate_records(output, mutation=mutation)
                    with self.assertRaises(ValueError) as caught:
                        build_downstream_alignment_packet(
                            phase2k_dir=output,
                            reviewed_packet_path=state["packet_path"],
                            coverage_path=state["coverage_path"],
                        )
                    self.assertIn(expected, str(caught.exception))

    def test_builder_rejects_stale_records_via_existing_audit_validator(self):
        state = _shared_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "copy"
            shutil.copytree(state["output"], output)
            _mutate_records(output, mutation="stale")
            with self.assertRaises(ValueError) as caught:
                build_downstream_alignment_packet(
                    phase2k_dir=output,
                    reviewed_packet_path=state["packet_path"],
                    coverage_path=state["coverage_path"],
                )
            self.assertIn("transformation audit", str(caught.exception))

    def test_builder_deep_validates_current_contract_before_accepting_inputs(
        self,
    ):
        state = _shared_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "copy"
            shutil.copytree(state["output"], output)
            _stale_inference_config_version(output)

            # The stale output is internally/audit bound: the completed audit
            # still validates against the blank audit and rehashed records.
            records = load_json_strict(
                output / OUTPUT_FILENAMES["records"], label="records copy",
            )
            audit = load_json_strict(
                output / OUTPUT_FILENAMES["transformation_audit"],
                label="audit copy",
            )
            completed = load_json_strict(
                output / OUTPUT_FILENAMES[
                    "finalized_transformation_audit"
                ],
                label="completed audit copy",
            )
            validate_completed_transformation_audits(
                audit, completed, records_obj=records,
            )

            calls: list[dict[str, Any]] = []
            real_validate = validate_output_directory

            def spy(**kwargs: Any) -> dict[str, Any]:
                calls.append(kwargs)
                return real_validate(**kwargs)

            with mock.patch.object(
                alignment_module,
                "validate_output_directory",
                side_effect=spy,
            ):
                with self.assertRaises(ValueError) as caught:
                    build_downstream_alignment_packet(
                        phase2k_dir=output,
                        reviewed_packet_path=state["packet_path"],
                        coverage_path=state["coverage_path"],
                    )
            self.assertIn(
                "inference config version is invalid", str(caught.exception),
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["output_dir"], output)
            self.assertEqual(calls[0]["packet_path"], state["packet_path"])
            self.assertEqual(calls[0]["db_path"], state["db_path"])
            self.assertEqual(
                calls[0]["manifest_path"], state["manifest_path"],
            )

    def test_builder_rejects_missing_failed_or_stale_human_gate(self):
        state = _shared_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "copy"
            shutil.copytree(state["output"], output)
            (output / OUTPUT_FILENAMES["finalized_packet"]).unlink()
            with self.assertRaises(ValueError) as caught:
                build_downstream_alignment_packet(
                    phase2k_dir=output,
                    reviewed_packet_path=state["packet_path"],
                    coverage_path=state["coverage_path"],
                )
            self.assertIn("missing", str(caught.exception))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path, coverage_path = _rich_fixture(root)
            failed = finalize_live_state(
                root,
                reviews_factory=complete_reviews,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )
            with self.assertRaises(ValueError) as caught:
                build_downstream_alignment_packet(
                    phase2k_dir=failed["output"],
                    reviewed_packet_path=packet_path,
                    coverage_path=coverage_path,
                )
            self.assertIn("PASSED", str(caught.exception))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "copy"
            shutil.copytree(state["output"], output)
            summary_path = output / OUTPUT_FILENAMES["human_summary"]
            summary = load_json_strict(summary_path, label="summary copy")
            summary["overall"]["item_count"] = 1
            summary_path.write_text(
                json.dumps(summary, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                build_downstream_alignment_packet(
                    phase2k_dir=output,
                    reviewed_packet_path=state["packet_path"],
                    coverage_path=state["coverage_path"],
                )
            self.assertIn("does not match", str(caught.exception))

    def test_builder_rejects_missing_and_invalid_audit(self):
        state = _shared_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "copy"
            shutil.copytree(state["output"], output)
            (output / OUTPUT_FILENAMES["finalized_transformation_audit"]).unlink()
            with self.assertRaises(ValueError) as caught:
                build_downstream_alignment_packet(
                    phase2k_dir=output,
                    reviewed_packet_path=state["packet_path"],
                    coverage_path=state["coverage_path"],
                )
            self.assertIn("missing", str(caught.exception))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "copy"
            shutil.copytree(state["output"], output)
            audit_path = output / OUTPUT_FILENAMES["finalized_transformation_audit"]
            audit = load_json_strict(audit_path, label="audit copy")
            audit["schema_version"] = "phase2k-transformation-audit-packet-v1"
            audit_path.write_text(
                json.dumps(audit, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                build_downstream_alignment_packet(
                    phase2k_dir=output,
                    reviewed_packet_path=state["packet_path"],
                    coverage_path=state["coverage_path"],
                )
            self.assertIn("schema version", str(caught.exception))

    def test_builder_rejects_coverage_packet_hash_mismatch(self):
        state = _shared_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet_path = root / "reviewed-endpoint-annotation-packet-v1.json"
            shutil.copyfile(state["packet_path"], packet_path)
            coverage_path = root / "candidate-coverage-v1.json"
            coverage = load_json_strict(
                state["coverage_path"], label="coverage copy",
            )
            coverage["reviewed_packet"]["content_sha256"] = "0" * 64
            coverage["content_sha256"] = canonical_sha256({
                key: value for key, value in coverage.items()
                if key != "content_sha256"
            })
            coverage_path.write_text(
                json.dumps(coverage, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                build_downstream_alignment_packet(
                    phase2k_dir=state["output"],
                    reviewed_packet_path=packet_path,
                    coverage_path=coverage_path,
                )
            self.assertIn("content hash", str(caught.exception))

    def test_packet_has_no_prediction_or_semantic_extraction_fields(self):
        packet = _blank_packet()
        _scan_forbidden_leaks(packet, path="packet")

        def collect_keys(value: Any) -> set[str]:
            keys: set[str] = set()
            if isinstance(value, Mapping):
                for key, item in value.items():
                    keys.add(key)
                    keys.update(collect_keys(item))
            elif isinstance(value, list):
                for item in value:
                    keys.update(collect_keys(item))
            return keys

        keys = collect_keys(packet)
        for key in (
            "model_predictions", "model_scoring", "predictions", "score",
            "scores", "probability", "rank", "threshold", "scorer",
            "semantic_claims", "entities", "relations", "claims",
            "semantic_extraction", "architectures", "generative",
            "discriminative",
        ):
            self.assertNotIn(key, keys)

    def test_phase2j_inputs_immutable_and_hash_bound(self):
        state = _shared_state()
        manifest_before = state["manifest_path"].read_bytes()
        packet_before = state["packet_path"].read_bytes()
        coverage_before = state["coverage_path"].read_bytes()
        alignment = _blank_packet()
        self.assertEqual(state["manifest_path"].read_bytes(), manifest_before)
        self.assertEqual(state["packet_path"].read_bytes(), packet_before)
        self.assertEqual(state["coverage_path"].read_bytes(), coverage_before)
        binding = alignment["dataset_binding"]
        coverage = load_candidate_coverage(state["coverage_path"])
        self.assertEqual(
            binding["phase2j_reviewed_packet_sha256"],
            coverage["reviewed_packet"]["content_sha256"],
        )
        self.assertEqual(
            binding["phase2j_coverage_sha256"], coverage["content_sha256"],
        )

    def test_canonical_hash_and_key_set_reject_mutation(self):
        packet = _blank_packet()
        validate_downstream_alignment_packet(packet, require_blank=True)
        tampered = copy.deepcopy(packet)
        tampered["items"][0]["representation"]["polished_text"] = "tampered"
        with self.assertRaises(ValueError) as caught:
            validate_downstream_alignment_packet(tampered, require_blank=True)
        self.assertIn("content_sha256", str(caught.exception))
        extra = copy.deepcopy(packet)
        extra["unexpected"] = True
        with self.assertRaises(ValueError):
            validate_downstream_alignment_packet(extra, require_blank=True)


class AlignmentFinalizeTests(unittest.TestCase):
    def test_finalize_valid_decisions_and_summary_arithmetic(self):
        packet = _blank_packet()
        decisions = _default_decisions(packet)
        finalized = finalize_downstream_alignment_packet(packet, decisions)
        self.assertEqual(finalized["release_gate"], "REVIEWED")
        validate_downstream_alignment_packet(finalized, require_blank=False)
        for item in finalized["items"]:
            self.assertIn(item["decision"]["state"], ALIGNMENT_DECISION_STATES)
            self.assertTrue(item["decision"]["reviewer"])
            self.assertTrue(item["decision"]["completed_at"])
            for span in item["decision"]["polished_spans"]:
                self.assertEqual(
                    span["text"],
                    item["representation"]["polished_text"][
                        span["start"]:span["end"]
                    ],
                )
        summary = build_alignment_summary(finalized)
        validate_alignment_summary(summary, finalized=finalized)
        self.assertEqual(summary["total"], 311)
        self.assertEqual(
            summary["alignment_packet_sha256"], finalized["content_sha256"],
        )
        self.assertEqual(
            sum(entry["count"] for entry in summary["by_state"].values()), 311,
        )
        self.assertEqual(
            sum(entry["count"] for entry in summary["by_node_type"].values()), 311,
        )
        self.assertEqual(
            sum(entry["total"] for entry in summary["by_window"].values()), 311,
        )
        self.assertEqual(len(summary["by_window"]), 30)
        self.assertEqual(summary["boundary_corrections"], {
            "unchanged_count": 263,
            "corrected_count": 48,
            "dropped_terminal_period_count": 28,
            "dropped_terminal_comma_count": 20,
        })
        unresolved = summary["unresolved_targets"]
        expected_ids = sorted(
            item["alignment_id"]
            for item in finalized["items"]
            if item["decision"]["state"] in ("ABSENT", "AMBIGUOUS")
        )
        self.assertEqual(unresolved["count"], len(expected_ids))
        self.assertEqual(sorted(unresolved["alignment_ids"]), expected_ids)
        self.assertTrue(unresolved["count"] > 0)

    def test_validate_returns_complete_envelope_with_content_sha256(self):
        packet = _blank_packet()
        validated = validate_downstream_alignment_packet(
            packet, require_blank=True,
        )
        self.assertEqual(
            set(validated),
            {
                "schema_version", "content_sha256", "purpose", "release_gate",
                "dataset_binding", "boundary_rule", "items",
            },
        )
        self.assertEqual(validated["content_sha256"], packet["content_sha256"])
        finalized = finalize_downstream_alignment_packet(
            packet, _default_decisions(packet),
        )
        validated_finalized = validate_downstream_alignment_packet(
            finalized, require_blank=False,
        )
        self.assertEqual(
            set(validated_finalized),
            {
                "schema_version", "content_sha256", "purpose", "release_gate",
                "dataset_binding", "boundary_rule", "items",
            },
        )
        self.assertEqual(
            validated_finalized["content_sha256"],
            finalized["content_sha256"],
        )

    def _single_decision(self, packet: Mapping[str, Any]) -> dict[str, Any]:
        decisions = _default_decisions(packet)
        item = packet["items"][0]
        decisions[item["alignment_id"]] = {
            "state": "ALIGNED",
            "polished_spans": [{
                "start": 2,
                "end": 4,
                "text": item["representation"]["polished_text"][2:4],
            }],
            "reviewer": "human",
            "completed_at": "2026-08-19T00:00:00.000Z",
            "notes": [],
        }
        return decisions

    def test_finalize_rejects_invalid_states_and_cardinality(self):
        packet = _blank_packet()
        for state in ("ALIGNED", "ABSENT", "AMBIGUOUS", "MULTIPLE_CANDIDATES"):
            decisions = self._single_decision(packet)
            item_id = packet["items"][0]["alignment_id"]
            decisions[item_id]["state"] = state
            if state == "ABSENT":
                decisions[item_id]["polished_spans"] = []
            elif state == "AMBIGUOUS":
                decisions[item_id]["polished_spans"] = []
            elif state == "MULTIPLE_CANDIDATES":
                text = packet["items"][0]["representation"]["polished_text"]
                decisions[item_id]["polished_spans"] = [
                    {"start": 2, "end": 4, "text": text[2:4]},
                    {"start": 10, "end": 12, "text": text[10:12]},
                ]
            if state == "ALIGNED":
                finalize_downstream_alignment_packet(packet, decisions)
        for state, spans in (
            ("ALIGNED", []),
            ("ABSENT", [{"start": 2, "end": 4, "text": "xx"}]),
            ("MULTIPLE_CANDIDATES", [{"start": 2, "end": 4, "text": "xx"}]),
            ("NOT_A_STATE", []),
        ):
            with self.subTest(state=state):
                decisions = self._single_decision(packet)
                item_id = packet["items"][0]["alignment_id"]
                text = packet["items"][0]["representation"]["polished_text"]
                if spans:
                    spans = [
                        {"start": 2, "end": 4, "text": text[2:4]},
                    ] if len(spans) == 1 else spans
                decisions[item_id]["state"] = state
                decisions[item_id]["polished_spans"] = spans
                with self.assertRaises(ValueError):
                    finalize_downstream_alignment_packet(packet, decisions)

    def test_finalize_rejects_invalid_spans_and_duplicates(self):
        packet = _blank_packet()
        text = packet["items"][0]["representation"]["polished_text"]
        cases = [
            {"start": True, "end": 4, "text": text[True:4]},
            {"start": 2, "end": True, "text": text[2:True]},
            {"start": 4, "end": 4, "text": ""},
            {"start": 2, "end": 4, "text": "WRONG"},
            {"start": 2, "end": len(text) + 1, "text": text[2:] + "x"},
            {"start": -1, "end": 4, "text": text[-1:4]},
        ]
        for span in cases:
            with self.subTest(span=span):
                decisions = self._single_decision(packet)
                item_id = packet["items"][0]["alignment_id"]
                decisions[item_id]["polished_spans"] = [span]
                with self.assertRaises(ValueError):
                    finalize_downstream_alignment_packet(packet, decisions)
        decisions = self._single_decision(packet)
        item_id = packet["items"][0]["alignment_id"]
        decisions[item_id]["state"] = "MULTIPLE_CANDIDATES"
        decisions[item_id]["polished_spans"] = [
            {"start": 2, "end": 4, "text": text[2:4]},
            {"start": 2, "end": 4, "text": text[2:4]},
        ]
        with self.assertRaises(ValueError) as caught:
            finalize_downstream_alignment_packet(packet, decisions)
        self.assertIn("unique", str(caught.exception))

    def test_finalize_rejects_partial_decision_maps(self):
        packet = _blank_packet()
        decisions = _default_decisions(packet)
        missing_key = next(iter(decisions))
        del decisions[missing_key]
        with self.assertRaises(ValueError) as caught:
            finalize_downstream_alignment_packet(packet, decisions)
        self.assertIn("missing", str(caught.exception))
        decisions = _default_decisions(packet)
        decisions["p2k:align:extra"] = {
            "state": "ABSENT",
            "polished_spans": [],
            "reviewer": "human",
            "completed_at": "2026-08-19T00:00:00.000Z",
            "notes": [],
        }
        with self.assertRaises(ValueError) as caught:
            finalize_downstream_alignment_packet(packet, decisions)
        self.assertIn("extra", str(caught.exception))

    def test_finalize_rejects_cross_target_duplicate_spans(self):
        packet = _blank_packet()
        decisions = _default_decisions(packet)
        window_items = [
            item for item in packet["items"]
            if item["window_id"] == packet["items"][0]["window_id"]
        ]
        first, second = window_items[0], window_items[1]
        text = first["representation"]["polished_text"]
        shared_span = {"start": 2, "end": 4, "text": text[2:4]}
        decisions[first["alignment_id"]] = {
            "state": "ALIGNED",
            "polished_spans": [dict(shared_span)],
            "reviewer": "human",
            "completed_at": "2026-08-19T00:00:00.000Z",
            "notes": [],
        }
        decisions[second["alignment_id"]] = {
            "state": "ALIGNED",
            "polished_spans": [dict(shared_span)],
            "reviewer": "human",
            "completed_at": "2026-08-19T00:00:00.000Z",
            "notes": [],
        }
        with self.assertRaises(ValueError) as caught:
            finalize_downstream_alignment_packet(packet, decisions)
        self.assertIn("cross-target duplicate", str(caught.exception))

    def test_finalize_preserves_source_content_exactly(self):
        packet = _blank_packet()
        decisions = _default_decisions(packet)
        finalized = finalize_downstream_alignment_packet(packet, decisions)
        for blank_item, finalized_item in zip(
            packet["items"], finalized["items"],
        ):
            self.assertEqual(
                finalized_item["bronze_target"], blank_item["bronze_target"],
            )
            self.assertEqual(
                finalized_item["representation"], blank_item["representation"],
            )
            self.assertEqual(
                finalized_item["node_type"], blank_item["node_type"],
            )
        tampered = copy.deepcopy(finalized)
        tampered["items"][0]["bronze_target"]["original_text"] = "mutated"
        with self.assertRaises(ValueError):
            validate_downstream_alignment_packet(tampered, require_blank=False)


class AlignmentCliTests(unittest.TestCase):
    def test_build_cli_no_overwrite_and_validate_only(self):
        state = _shared_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "alignment.json"
            code = build_cli.main([
                "--phase2k-dir", str(state["output"]),
                "--reviewed-packet", str(state["packet_path"]),
                "--coverage", str(state["coverage_path"]),
                "--output", str(output),
            ])
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            with self.assertRaises(SystemExit):
                build_cli.main([
                    "--phase2k-dir", str(state["output"]),
                    "--reviewed-packet", str(state["packet_path"]),
                    "--coverage", str(state["coverage_path"]),
                    "--output", str(output),
                ])
            code = build_cli.main([
                "--phase2k-dir", str(state["output"]),
                "--reviewed-packet", str(state["packet_path"]),
                "--coverage", str(state["coverage_path"]),
                "--output", str(output),
                "--validate-only",
            ])
            self.assertEqual(code, 0)
            bad = root / "bad.json"
            bad.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
            code = build_cli.main([
                "--phase2k-dir", str(state["output"]),
                "--reviewed-packet", str(state["packet_path"]),
                "--coverage", str(state["coverage_path"]),
                "--output", str(bad),
                "--validate-only",
            ])
            self.assertEqual(code, 1)

    def test_finalize_cli_writes_packet_and_summary_without_overwrite(self):
        state = _shared_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blank = root / "blank.json"
            blank.write_text(
                json.dumps(_blank_packet(), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            decisions_path = root / "decisions.json"
            decisions_path.write_text(
                json.dumps(_default_decisions(_blank_packet()), sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            finalized_path = root / "finalized.json"
            summary_path = root / "summary.json"
            code = finalize_cli.main([
                "--phase2k-dir", str(state["output"]),
                "--reviewed-packet", str(state["packet_path"]),
                "--coverage", str(state["coverage_path"]),
                "--packet", str(blank),
                "--decisions", str(decisions_path),
                "--output", str(finalized_path),
                "--summary", str(summary_path),
            ])
            self.assertEqual(code, 0)
            self.assertTrue(finalized_path.is_file())
            self.assertTrue(summary_path.is_file())
            with self.assertRaises(SystemExit):
                finalize_cli.main([
                    "--phase2k-dir", str(state["output"]),
                    "--reviewed-packet", str(state["packet_path"]),
                    "--coverage", str(state["coverage_path"]),
                    "--packet", str(blank),
                    "--decisions", str(decisions_path),
                    "--output", str(finalized_path),
                    "--summary", str(summary_path),
                ])
            partial = root / "partial.json"
            decisions = _default_decisions(_blank_packet())
            del decisions[next(iter(decisions))]
            partial.write_text(
                json.dumps(decisions, sort_keys=True) + "\n", encoding="utf-8",
            )
            code = finalize_cli.main([
                "--phase2k-dir", str(state["output"]),
                "--reviewed-packet", str(state["packet_path"]),
                "--coverage", str(state["coverage_path"]),
                "--packet", str(blank),
                "--decisions", str(partial),
                "--output", str(root / "finalized2.json"),
                "--summary", str(root / "summary2.json"),
            ])
            self.assertEqual(code, 1)

    def test_finalize_cli_rejects_canonical_forged_packet_with_bindings(self):
        state = _shared_state()
        forged_text = "Forged polished display text. " * 12
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decisions = _default_decisions(_blank_packet())
            cases = (
                ("dataset binding", "dataset_binding does not match its inputs"),
                ("display/source", "polished text must equal the sealed D"),
            )
            for label, expected in cases:
                with self.subTest(label=label):
                    slug = label.replace(" ", "-").replace("/", "-")
                    packet = copy.deepcopy(_blank_packet())
                    if label == "dataset binding":
                        packet["dataset_binding"]["phase2k_records_sha256"] = (
                            "11" * 32
                        )
                    else:
                        item = packet["items"][0]
                        item["representation"]["polished_text"] = forged_text
                        item["representation"]["polished_text_sha256"] = (
                            text_sha256(forged_text)
                        )
                    packet["content_sha256"] = canonical_sha256({
                        key: value for key, value in packet.items()
                        if key != "content_sha256"
                    })
                    # The forgery is canonical and self-consistently valid
                    # without source bindings.
                    validate_downstream_alignment_packet(
                        packet, require_blank=True,
                    )
                    blank = root / f"blank-{slug}.json"
                    blank.write_text(
                        json.dumps(packet, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    decisions_path = root / f"decisions-{slug}.json"
                    decisions_path.write_text(
                        json.dumps(decisions, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        code = finalize_cli.main([
                            "--phase2k-dir", str(state["output"]),
                            "--reviewed-packet", str(state["packet_path"]),
                            "--coverage", str(state["coverage_path"]),
                            "--packet", str(blank),
                            "--decisions", str(decisions_path),
                            "--output", str(root / f"out-{slug}.json"),
                            "--summary", str(root / f"sum-{slug}.json"),
                        ])
                    self.assertEqual(code, 1)
                    self.assertIn(expected, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
