"""Shared synthetic Phase 2K fixtures for the focused test modules.

This module is not collected by pytest.  It builds schema-valid Phase 2J
frozen-input manifests/reviewed packets plus a read-only-style SQLite
transcript DB using only the stdlib, so Phase 2K tests stay fast and offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from pipeline.phase2k_contextual_reconstruction import canonical_sha256, text_sha256


ROLES = ("mid", "top", "jungle", "adc", "support")
CHAMPIONS = ("Lux", "Garen", "Ahri", "Jinx", "Lee Sin", "Viktor")
PARTITIONS = ("EXPANDED_DEV", "FROZEN_REPLICATION")

GENERIC_TRANSCRIPT = (
    "Coach " + "{champion}" + " explains the play. "
    "You should push the wave now. "
    "Your opponent is low on mana. "
    "Stay behind the minions and wait for the gank. "
    "When they walk up, use your cooldowns and flash out if needed. "
    "That is the correct play for this lane state."
)


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


def make_selected(
    source_id: str,
    transcript: str,
    start: int,
    end: int,
    *,
    index: int,
    champion: str,
    role: str,
    video_title: str,
    partition: str = "EXPANDED_DEV",
) -> dict[str, Any]:
    text = transcript[start:end]
    identity = hashlib.sha256(
        f"{source_id}:{start}:{end}:{text}".encode("utf-8"),
    ).hexdigest()[:20]
    window_id = f"pool:{source_id}:w00001-{identity}"
    selected = {
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
        "phenomena": ["pronoun", "punctuation_poor"],
        "asr_punctuation_band": "PUNCTUATION_POOR",
        "partition": partition,
        "candidate_generator_version": "phase2f-mention-catalog-v3-cross-segment-ngrams-32",
        "candidate_count": 7,
        "candidate_catalog_sha256": "0" * 64,
        "canonical_record_sha256": "1" * 64,
    }
    selected["canonical_record_sha256"] = canonical_sha256(selected)
    return selected


def make_packet_record(
    selected: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    text = selected["source_text"]
    return {
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
                "notes": "synthetic Phase 2K test endpoint",
                "pass_provenance": "PASS_B",
            }
        ],
        "window_status": "REVIEWED",
        "pass_a": {
            "status": "COMPLETE",
            "reviewer": "test",
            "completed_at": "2026-08-19",
            "notes": [],
            "endpoint_count": 1,
        },
        "pass_b": {
            "status": "COMPLETE",
            "reviewer": "test",
            "completed_at": "2026-08-19T00:00:00.000Z",
            "notes": [],
            "audit_checks": {
                "boundaries": True,
                "omissions": True,
                "roles": True,
                "duplicates": True,
                "ambiguity": True,
            },
        },
        "ambiguity_controls": {"flagged": False, "notes": []},
        "exclusion_controls": {"flagged": False, "notes": []},
        "reviewer_notes": [],
    }


def build_fixture(
    root: Path,
    *,
    transcripts: Mapping[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    """Write synthetic Phase 2J inputs and a transcript DB; return paths."""
    if transcripts is None:
        transcripts = {}
        for index in range(1, 31):
            champion = CHAMPIONS[index % len(CHAMPIONS)]
            source_id = f"s{index:02d}"
            text = GENERIC_TRANSCRIPT.format(champion=champion)
            transcripts[source_id] = text
    source_ids = sorted(transcripts)
    selected = []
    for index, source_id in enumerate(source_ids, 1):
        transcript = transcripts[source_id]
        champion = CHAMPIONS[index % len(CHAMPIONS)]
        role = ROLES[index % len(ROLES)]
        video_title = f"Video {source_id} {champion}"
        start = transcript.index("You should push the wave now.")
        end = start + len("You should push the wave now.")
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
        "purpose": "Synthetic Phase 2K test fixture; not real Phase 2J data.",
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
            "EXPANDED_DEV": sum(1 for item in selected if item["partition"] == "EXPANDED_DEV"),
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
        "candidate_generator_version": "phase2f-mention-catalog-v3-cross-segment-ngrams-32",
        "checkpoint": "PRE_ANNOTATION_CHECKPOINT",
    }
    manifest = {"content_sha256": canonical_sha256(manifest), **manifest}
    records = [
        make_packet_record(item, index=index)
        for index, item in enumerate(selected, 1)
    ]
    packet = {
        "schema_version": "phase2j-endpoint-annotation-packet-v1",
        "purpose": "Synthetic Phase 2K test packet; not real Phase 2J data.",
        "annotation_version": "phase2j-endpoint-annotation-v1",
        "release_gate": "LOCKED",
        "selection_manifest_sha256": manifest["content_sha256"],
        "selection_manifest_schema_version": "phase2j-window-selection-manifest-v1",
        "candidate_generator_version": "phase2f-mention-catalog-v3-cross-segment-ngrams-32",
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

    db_path = root / "videos.db"
    connection = sqlite3.connect(db_path)
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
    return manifest_path, packet_path, db_path

