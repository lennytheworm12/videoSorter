"""Shared synthetic Phase 2K full-transcript ablation fixtures (offline).

Not collected by pytest.  Builds a schema-valid frozen Phase 2J manifest, a
SQLite transcript DB with descriptions and champion abilities, a lexical
vocabulary v2 snapshot, and validated A/B payloads so the focused Phase 2K
test module stays fast and offline.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from pipeline.phase2j_context_ablation import (
    canonical_sha256,
    text_sha256,
)
from pipeline.phase2k_full_transcript_ablation import (
    INTERMEDIATE_SCHEMA_VERSION,
    SEMANTIC_FIELDS,
    build_case_vocabulary,
    build_condition_payloads,
    build_extraction_instructions,
    select_phase2k_cases,
)
from tests._phase2j_context_ablation_helpers import (
    WINDOW_TAG_SETS,
    make_manifest,
)


def make_lexical_vocabulary() -> dict[str, Any]:
    vocabulary = {
        "schema_version": "phase2k-league-lexical-vocabulary-v2",
        "purpose": "synthetic lexical snapshot for tests",
        "ability_keys": ["Q", "W", "E", "R", "D", "F"],
        "summoner_spells": ["flash", "ignite"],
        "basic_domain_tokens": ["wave", "lane", "jungle"],
        "champion_alias_rules": {"Karfus": "Karthus"},
    }
    return vocabulary


def _transcript(source_id: str, index: int) -> str:
    words = (
        [f"word{index}-{i}" for i in range(1800)]
        + ["jinx"]
        + [f"tail{index}-{i}" for i in range(400)]
    )
    return " ".join(words)


def make_transcript_db_2k(
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
    champions = ("Lux", "Garen", "Ahri", "Jinx", "Lee Sin", "Viktor")
    roles = ("mid", "top", "jungle", "adc", "support")
    for index, (source_id, transcript) in enumerate(transcripts.items()):
        champion = champions[index % len(champions)]
        role = roles[index % len(roles)]
        connection.execute(
            "INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                f"https://www.youtube.com/watch?v={source_id}",
                f"Coach {champion} {role} {index}",
                f"Synthetic description {index} for {champion} {role}.",
                role,
                champion,
                None if index % 2 else "emerald",
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
    for champion in champions:
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


def build_phase2k_fixture(root: Path) -> dict[str, Any]:
    """Build manifest/DB/vocabulary files plus derived frozen artifacts."""
    transcripts = {
        f"video{index:02d}": _transcript(f"video{index:02d}", index)
        for index in range(30)
    }
    manifest = make_manifest(transcripts)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    db_path = make_transcript_db_2k(root, transcripts)
    vocabulary = make_lexical_vocabulary()
    vocabulary_path = root / "league_lexical_vocabulary_v2.json"
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")

    from pipeline.phase2j_context_ablation import (
        champion_abilities_for_transcript,
        load_lexical_vocabulary,
        open_transcript_db,
    )

    connection = open_transcript_db(db_path)
    try:
        from pipeline.phase2k_full_transcript_ablation import fetch_source_rows

        source_rows = fetch_source_rows(connection, manifest["selected"])
        cases = select_phase2k_cases(manifest, source_rows=source_rows)
        instructions = build_extraction_instructions()
        lexical = load_lexical_vocabulary(vocabulary_path)
        vocabulary_by_case = {}
        for case in cases:
            row = source_rows[case["upstream_source_id"]]
            champion_data = champion_abilities_for_transcript(
                connection,
                metadata_champion=row["champion"],
                transcript=row["transcript"],
                video_id=row["video_id"],
            )
            vocabulary_by_case[case["case_id"]] = build_case_vocabulary(
                case_id=case["case_id"],
                lexical_vocabulary=lexical,
                champion_data=champion_data,
            )
        payload_cases, provenance_by_case = build_condition_payloads(
            cases=cases,
            source_rows=source_rows,
            vocabulary_by_case=vocabulary_by_case,
            instructions=instructions,
        )
    finally:
        connection.close()
    return {
        "transcripts": transcripts,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "db_path": db_path,
        "vocabulary": vocabulary,
        "vocabulary_path": vocabulary_path,
        "cases": cases,
        "instructions": instructions,
        "lexical_vocabulary": lexical,
        "payload_cases": payload_cases,
        "provenance_by_case": provenance_by_case,
    }


def _payload_source(payload: Mapping[str, Any]) -> str:
    if payload["condition"] == "A":
        return payload["target"]["bronze_text"]
    return payload["transcript"]


def make_valid_intermediate_response(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a schema-valid intermediate response bound to the payload."""
    source = _payload_source(payload)
    first_token = re.search(r"\S+", source)
    quote_a = first_token.group()
    occurrence_a = 0
    fields: dict[str, Any] = {field: [] for field in SEMANTIC_FIELDS}
    reference = {"quote": quote_a, "occurrence_index": occurrence_a}
    fields["actors_entities"] = [{
        "extraction_text": "coach",
        "resolution_status": "literal_explicit",
        "source_references": [dict(reference)],
    }]
    fields["reference_bindings"] = [{
        "extraction_text": "he binds to the opponent",
        "resolution_status": "unresolved",
        "source_references": [dict(reference)],
    }]
    fields["explicit_relationships"] = [{
        "extraction_text": "coach USES flash",
        "relation_type": "USES",
        "resolution_status": "context_resolved",
        "source_references": [dict(reference)],
    }]
    fields["uncertainty_unresolved"] = []
    fields["supporting_source_spans"] = [{
        "extraction_text": "target span",
        "resolution_status": "literal_explicit",
        "source_references": [dict(reference)],
    }]
    response = {
        "schema_version": INTERMEDIATE_SCHEMA_VERSION,
        "case_id": payload["case_id"],
        "condition": payload["condition"],
        "payload_sha256": payload["content_sha256"],
        "instructions_sha256": payload["instructions_sha256"],
        "fields": fields,
    }
    return response


def make_reviews(
    review_packet: Mapping[str, Any],
    *,
    reviewer_kind: str = "agent",
    b_success_fields: set[str] | None = None,
    a_success_fields: set[str] | None = None,
) -> dict[str, Any]:
    b_success_fields = b_success_fields or set()
    a_success_fields = a_success_fields or set()

    def entry(success: bool) -> dict[str, Any]:
        if success:
            return {
                "correctness": "CORRECT",
                "unsupported_inference": "NONE",
                "source_grounding": "GROUNDED",
                "notes": [],
            }
        return {
            "correctness": "INCORRECT",
            "unsupported_inference": "NONE",
            "source_grounding": "GROUNDED",
            "notes": [],
        }

    reviews = {}
    for case in review_packet["cases"]:
        case_id = case["case_id"]
        for field in review_packet["review_fields"]:
            reviews[f"{case_id}:A:{field}"] = entry(
                f"{case_id}:A:{field}" in a_success_fields,
            )
            reviews[f"{case_id}:B:{field}"] = entry(
                f"{case_id}:B:{field}" in b_success_fields,
            )
    completed = {
        "schema_version": "phase2k-full-transcript-ablation-completed-reviews-v1",
        "reviewer_kind": reviewer_kind,
        "reviewer_identity": "test-reviewer",
        "completed_at": "2026-08-22T00:00:00Z",
        "reviews": reviews,
    }
    if reviewer_kind == "agent":
        completed["agent_scoping_statement"] = (
            "Test scoring produced by an automated reviewer."
        )
    else:
        completed["human_review_attested"] = True
        completed["attestation_statement"] = "I reviewed every item."
    return {
        **completed,
        "content_sha256": canonical_sha256(completed),
    }


def expected_selected_window_ids() -> list[str]:
    rows = []
    for index, tags in enumerate(WINDOW_TAG_SETS):
        from pipeline.phase2j_context_ablation import DIFFICULTY_WEIGHTS

        present = [tag for tag in tags if tag in DIFFICULTY_WEIGHTS]
        rows.append((
            index,
            sum(DIFFICULTY_WEIGHTS[tag] for tag in present),
            len(tags),
        ))
    rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
    return [f"video{index:02d}" for index, _, _, in rows[:10]]
