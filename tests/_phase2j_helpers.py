"""Shared synthetic Phase 2J fixtures for the focused test modules.

This module is not collected by pytest (its name does not match
``test_*.py``).  It builds schema-valid retained pools, legacy manifests, and
legacy benchmarks using only stdlib dependencies so the Phase 2J tests stay
fast and offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.semantic_ir_pool import (
    POOL_PHENOMENA,
    detect_pool_phenomena,
    validate_semantic_window_pool,
)
from pipeline.phase2j_source_selection import canonical_sha256
from pipeline.phase2j_annotation_packet import (
    ANNOTATION_VERSION,
    PACKET_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
HUMAN_SESSION_SCHEMA_VERSION = "phase2j-review-session-v1"
SOL_REVIEW_SCHEMA_VERSION = "phase2j-sol-parallel-review-v1"

# Reuse the four corpus texts from the existing pool test; together they cover
# every frozen lexical phenomenon (verified by tests/test_semantic_ir_pool.py).
BASE_TEXTS = (
    (
        "If Lux misses Q and when Ahri uses W, you should push two waves "
        "before dragon because mana is lower and therefore they cannot contest. "
        "Push river instead, but maybe do not wait but move behind tower now."
    ),
    (
        "You should hold the lane and keep the tower safe since the enemy is "
        "dangerous nearby while your team prepares the next careful play."
    ),
    (
        "Jinx has a long attack range and deals steady damage to targets in the "
        "bottom lane during ordinary team fights around the map."
    ),
    (
        "push the wave move river hold vision save flash track cooldown take "
        "space respect range keep health use wards avoid danger and look for "
        "angles"
    ),
)
ROLES = ("mid", "top", "jungle", "adc", "support")
CHAMPIONS = ("Lux", "Garen", "Ahri", "Jinx", "Lee Sin")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_pool(
    source_ids: Iterable[str],
    *,
    champion_names: tuple[str, ...] = ("Lux", "Ahri", "Garen"),
) -> dict[str, Any]:
    """Build a schema-valid retained pool with one window per source."""
    sources = list(source_ids)
    windows = []
    for index, source_id in enumerate(sorted(sources), 1):
        text = BASE_TEXTS[index % len(BASE_TEXTS)]
        phenomena = list(detect_pool_phenomena(text, champion_names))
        identity = hashlib.sha256(
            f"{source_id}:0:{len(text)}:{text}".encode("utf-8"),
        ).hexdigest()[:20]
        role = ROLES[index % len(ROLES)]
        champion = CHAMPIONS[index % len(CHAMPIONS)]
        windows.append({
            "pool_index": index,
            "window_id": f"pool:{source_id}:w00001-{identity}",
            "source_id": f"transcript:{source_id}",
            "source_kind": "transcript",
            "upstream_source_id": source_id,
            "upstream_start": 0,
            "upstream_end": len(text),
            "token_offset": 0,
            "source_window_ordinal": 1,
            "source_text": text,
            "source_text_sha256": _sha256_text(text),
            "upstream_content_sha256": _sha256_text(text),
            "phenomena": phenomena,
            "metadata": {
                "video_title": f"Video {source_id}",
                "role": role,
                "champion": champion,
            },
        })
    phenomenon_counts = {key: 0 for key in POOL_PHENOMENA}
    for window in windows:
        for phenomenon in window["phenomena"]:
            phenomenon_counts[phenomenon] += 1
    inner = {
        "schema_version": "phase2f-semantic-window-pool-v1",
        "purpose": "Synthetic retained pool for Phase 2J tests; not gold.",
        "selection_policy": {
            "target_count": len(windows),
            "target_words": 48,
            "stride_words": 40,
            "minimum_per_phenomenon": 1,
            "one_window_per_upstream_source": True,
            "champion_names": list(sorted(set(champion_names))),
            "excluded_phase2b_sources": [],
            "excluded_phase2d_sources": [],
        },
        "input_hashes": {
            "database_sha256": "0" * 64,
            "frozen_fixture_sha256": "1" * 64,
            "development_fixture_sha256": "2" * 64,
        },
        "phenomenon_counts": dict(sorted(phenomenon_counts.items())),
        "windows": windows,
    }
    pool = {"content_sha256": canonical_sha256(inner), **inner}
    validate_semantic_window_pool(pool)
    return pool


def write_pool(path: Path, source_ids: Iterable[str], **kwargs: Any) -> Mapping[str, Any]:
    pool = build_pool(source_ids, **kwargs)
    path.write_text(
        json.dumps(pool, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return pool


def build_legacy_manifest(
    cases: Iterable[tuple[str, str, str, str]],
) -> dict[str, Any]:
    """Build a schema-valid legacy manifest from (case_id, source, upstream, text)."""
    windows = []
    for case_id, source_id, upstream_id, text in cases:
        windows.append({
            "case_id": case_id,
            "phenomena": ["legacy_failure_regression"],
            "source_id": source_id,
            "source_text": text,
            "source_text_sha256": _sha256_text(text),
            "upstream_content_sha256": _sha256_text(text),
            "upstream_end": len(text),
            "upstream_source_id": upstream_id,
            "upstream_start": 0,
        })
    inner = {
        "schema_version": "phase2f-legacy-five-source-manifest-v1",
        "purpose": "Synthetic legacy five-window manifest for Phase 2J tests.",
        "input_hashes": {
            "database_sha256": "3" * 64,
            "phase2d_fixture_sha256": "4" * 64,
            "phase2e_artifact_file_sha256": "5" * 64,
            "phase2e_artifact_inner_sha256": "6" * 64,
        },
        "windows": windows,
    }
    return {"content_sha256": canonical_sha256(inner), **inner}


def write_legacy_manifest(
    path: Path, cases: Iterable[tuple[str, str, str, str]],
) -> Mapping[str, Any]:
    manifest = build_legacy_manifest(cases)
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_legacy_benchmark(
    manifest: Mapping[str, Any], case_ids: Iterable[str],
) -> dict[str, Any]:
    cases = [
        {
            "id": case_id,
            "exhaustive": False,
            "edges": [],
            "mentions": [],
            "qualifiers": [],
            "questions": [],
            "references": [],
            "phenomena": ["legacy_failure_regression", "legacy_phase2e_failure"],
        }
        for case_id in case_ids
    ]
    inner = {
        "schema_version": "phase2f-semantic-benchmark-v1",
        "purpose": "Synthetic legacy benchmark for Phase 2J tests.",
        "split": "LEGACY_FAILURE",
        "pool_manifest_sha256": manifest["content_sha256"],
        "cases": cases,
    }
    return {"content_sha256": canonical_sha256(inner), **inner}


def write_legacy_benchmark(
    path: Path, manifest: Mapping[str, Any], case_ids: Iterable[str],
) -> Mapping[str, Any]:
    benchmark = build_legacy_benchmark(manifest, case_ids)
    path.write_text(
        json.dumps(benchmark, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return benchmark


def default_sources(count: int = 40) -> list[str]:
    return [f"src-{index:02d}" for index in range(1, count + 1)]


def write_standard_phase2j_inputs(
    root: Path,
    *,
    legacy_sources: tuple[str, ...] = ("legacy-1", "legacy-2"),
) -> tuple[Path, Path, Path, Mapping[str, Any], Mapping[str, Any]]:
    """Write pool + legacy manifest + benchmark; return their paths and values."""
    sources = default_sources(40) + list(legacy_sources)
    pool_path = root / "pool.json"
    pool = write_pool(pool_path, sources)
    cases = [
        (f"legacy-case-{index}", f"transcript:{source}", source, BASE_TEXTS[0])
        for index, source in enumerate(legacy_sources, 1)
    ]
    # The locked benchmark needs five cases; the remaining three reference
    # sources that never appear in the retained pool.
    extra = [
        (f"legacy-case-{index}", f"transcript:outside-{index}", f"outside-{index}", BASE_TEXTS[1])
        for index in range(3, 6)
    ]
    cases = cases + extra
    manifest_path = root / "legacy-manifest.json"
    legacy_manifest = write_legacy_manifest(manifest_path, cases)
    benchmark_path = root / "legacy-benchmark.json"
    legacy_benchmark = write_legacy_benchmark(
        benchmark_path, legacy_manifest, [case[0] for case in cases],
    )
    return pool_path, manifest_path, benchmark_path, legacy_manifest, legacy_benchmark


def rehash_record(record: Mapping[str, Any]) -> dict[str, Any]:
    inner = {
        key: value for key, value in record.items() if key != "canonical_record_sha256"
    }
    return {**dict(record), "canonical_record_sha256": canonical_sha256(inner)}


def rehash_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    inner = {key: value for key, value in manifest.items() if key != "content_sha256"}
    return {**dict(manifest), "content_sha256": canonical_sha256(inner)}


def rehash_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    inner = {key: value for key, value in packet.items() if key != "content_sha256"}
    return {**dict(packet), "content_sha256": canonical_sha256(inner)}


def _default_human_spans(record: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    if len(record["tokens"]) >= 5:
        return [(0, 0, "ENTITY"), (2, 3, "ACTION")]
    return [(0, 0, "ENTITY")]


def _default_sol_spans(record: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    count = len(record["tokens"])
    if count >= 8:
        return [(4, 4, "TIME"), (5, 5, "LOCATION_OR_SPACE"), (6, 6, "STATE")]
    if count >= 4:
        return [(4, 4, "TIME")] if count == 5 else [(4, 4, "TIME"), (5, 5, "STATE")]
    return []


def _endpoint_dict(
    record: Mapping[str, Any],
    *,
    index: int,
    token_start: int,
    token_end: int,
    node_type: str,
    prefix: str,
) -> dict[str, Any]:
    tokens = record["tokens"]
    char_start = tokens[token_start]["start"]
    char_end = tokens[token_end]["end"]
    return {
        "endpoint_id": f"{prefix}:{record['window_id']}:ep:{str(index).zfill(4)}",
        "exact_bronze_text": record["bronze_text"][char_start:char_end],
        "char_start": char_start,
        "char_end": char_end,
        "token_start": token_start,
        "token_end": token_end,
        "node_type": node_type,
    }


def build_human_session(
    packet: Mapping[str, Any],
    *,
    spans_for_window: Any = None,
    reviewer_name: str = "test-reviewer",
) -> dict[str, Any]:
    """Build a schema-valid synthetic Pass A human session over the packet."""
    records = []
    for index, record in enumerate(packet["records"], 1):
        spans = spans_for_window(record) if spans_for_window else _default_human_spans(record)
        endpoints = [
            {
                **_endpoint_dict(
                    record,
                    index=endpoint_index,
                    token_start=token_start,
                    token_end=token_end,
                    node_type=node_type,
                    prefix="p2j:review",
                ),
                "ambiguity_state": "NONE",
                "disposition": "KEEP",
                "pass_provenance": "PASS_A",
                "human_accepted": True,
                "created_sequence": endpoint_index,
            }
            for endpoint_index, (token_start, token_end, node_type) in enumerate(spans, 1)
        ]
        records.append({
            "record_index": index,
            "window_id": record["window_id"],
            "source_group_id": record["source_group_id"],
            "bronze_text": record["bronze_text"],
            "bronze_text_sha256": record["bronze_text_sha256"],
            "bronze_char_length": record["bronze_char_length"],
            "tokens": record["tokens"],
            "endpoints": endpoints,
            "window_status": "IN_REVIEW",
            "outcome": "CLEAN",
            "note": "",
            "reviewer_name": reviewer_name,
            "completed_at": "2026-08-18",
            "pass_a_complete": True,
        })
    return {
        "schema_version": HUMAN_SESSION_SCHEMA_VERSION,
        "annotation_version": ANNOTATION_VERSION,
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "packet_sha256": packet["content_sha256"],
        "exported_at": "2026-08-18T00:00:00Z",
        "records": records,
    }


def write_human_session(
    path: Path,
    packet: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    session = build_human_session(packet, **kwargs)
    path.write_text(
        json.dumps(session, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return session


def build_sol_review(
    packet: Mapping[str, Any],
    *,
    spans_for_window: Any = None,
) -> dict[str, Any]:
    """Build a schema-valid synthetic sealed Sol parallel review over the packet."""
    records = []
    for index, record in enumerate(packet["records"], 1):
        spans = spans_for_window(record) if spans_for_window else _default_sol_spans(record)
        proposed = []
        for endpoint_index, (token_start, token_end, node_type) in enumerate(spans, 1):
            tokens = record["tokens"]
            char_start = tokens[token_start]["start"]
            char_end = tokens[token_end]["end"]
            proposed.append({
                "ambiguity_state": "NONE",
                "concise_rationale": "synthetic Sol second opinion for tests",
                "exact_bronze_text": record["bronze_text"][char_start:char_end],
                "node_type": node_type,
                "pass_provenance": "SOL_PARALLEL_NON_GOLD",
                "token_end": token_end,
                "token_start": token_start,
            })
        records.append({
            "record_index": index,
            "window_id": record["window_id"],
            "bronze_text_sha256": record["bronze_text_sha256"],
            "window_ambiguity_notes": [],
            "omission_audit_notes": [],
            "proposed_endpoints": proposed,
        })
    inner = {
        "schema_version": SOL_REVIEW_SCHEMA_VERSION,
        "blank_packet_sha256": packet["content_sha256"],
        "selection_manifest_sha256": packet["selection_manifest_sha256"],
        "purpose": "SEALED NAVIGATION/AUDIT ONLY; NOT GOLD",
        "reasoning_effort": "high",
        "reviewer_model": "synthetic-sol-test",
        "visibility_gate": "SEALED_UNTIL_HUMAN_PASS_A_COMPLETE",
        "records": records,
    }
    return {"content_sha256": canonical_sha256(inner), **inner}


def write_sol_review(
    path: Path,
    packet: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    review = build_sol_review(packet, **kwargs)
    path.write_text(
        json.dumps(review, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return review
