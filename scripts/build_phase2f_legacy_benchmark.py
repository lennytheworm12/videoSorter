#!/usr/bin/env python3
"""Build the reviewed Phase 2E five-case regression benchmark for Phase 2F.

The case-specific values in this file are gold annotations, never compiler
heuristics. Source text, offsets, and case eligibility are reconstructed from
the immutable Phase 2E artifact, Phase 2D fixture, and primary bronze DB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from pipeline.semantic_ir_evaluation import (
    BENCHMARK_SCHEMA_VERSION, load_semantic_benchmark,
)
from pipeline.semantic_mentions import generate_mention_candidates
from pipeline.semantic_source import BronzeSource, window_from_exact_span


LEGACY_MANIFEST_VERSION = "phase2f-legacy-five-source-manifest-v1"


ANNOTATIONS: Mapping[str, Mapping[str, Any]] = {
    "wave-reset-after-kill": {
        "mentions": (
            ("kill_actor", "ENTITY", "you", "when you kill him here"),
            ("die_actor", "ENTITY", "you", "you should just run into Tower"),
            ("kill", "EVENT", "kill him", "when you kill him here"),
            ("die", "ACTION", "run into Tower and just die", None),
            ("pull", "ACTION", "pull the wave up again", None),
        ),
        "edges": (
            ("actor_kill", "kill_actor", "kill", ("ACTOR",)),
            ("actor_die", "die_actor", "die", ("ACTOR",)),
            ("kill_condition", "kill", "die", ("CONDITION",)),
            ("die_purpose", "die", "pull", ("PURPOSE",)),
        ),
        "qualifiers": (
            ("kill_when", "kill", "conditionality", "CONDITIONAL", "when", "when you kill him here"),
        ),
        "references": (
            ("kill_actor_unresolved", "kill_actor", "INSUFFICIENT_EVIDENCE", ()),
            ("die_actor_unresolved", "die_actor", "INSUFFICIENT_EVIDENCE", ()),
        ),
    },
    "push-poke-wave-crash": {
        "mentions": (
            ("pressure_actor", "ENTITY", "you", "you really want to make sure you keep poking"),
            ("crash_actor", "ENTITY", "you", "and you crash like this"),
            ("lane", ("ACTION", "STATE"), "playing Push and poke lanes", None),
            ("crash", ("ACTION", "EVENT"), "you crash like this", None),
            ("pressure", "ACTION", "keep poking and pushing and hitting creeps", None),
        ),
        "edges": (
            ("actor_pressure", "pressure_actor", "pressure", ("ACTOR",)),
            ("actor_crash", "crash_actor", "crash", ("ACTOR",)),
            ("lane_condition", "lane", "pressure", ("CONDITION",)),
            ("crash_condition", "crash", "pressure", ("CONDITION",)),
        ),
        "qualifiers": (
            ("lane_when", "lane", "conditionality", "CONDITIONAL", "when", "when you're playing Push and poke lanes"),
        ),
        "references": (
            ("pressure_actor_unresolved", "pressure_actor", "INSUFFICIENT_EVIDENCE", ()),
            ("crash_actor_unresolved", "crash_actor", "INSUFFICIENT_EVIDENCE", ()),
        ),
    },
    "sweeper-limits-mid-play": {
        "mentions": (
            ("sweeper", "ABILITY_OR_RESOURCE", "sweeper", None),
            ("use", "ACTION", "the sweeper should be used around mid", None),
            ("vision", "ABILITY_OR_RESOURCE", "their Vision on Mid", None),
            ("remove_vision", "ACTION", "remove their Vision on Mid", None),
            ("play_ability", "ABILITY_OR_RESOURCE", "their ability to play on Mid", None),
            ("remove_play", "ACTION", "remove their ability to play on Mid", None),
            ("location", "LOCATION_OR_SPACE", "around mid", "the sweeper should be used around mid"),
        ),
        "edges": (
            ("sweeper_object", "sweeper", "use", ("OBJECT",)),
            ("vision_purpose", "use", "remove_vision", ("PURPOSE",)),
            ("vision_object", "vision", "remove_vision", ("OBJECT",)),
            ("play_purpose", "use", "remove_play", ("PURPOSE",)),
            ("play_object", "play_ability", "remove_play", ("OBJECT",)),
            ("mid_modifies", "location", "use", ("MODIFIES",)),
        ),
        "qualifiers": (
            ("use_should", "use", "modality", "OBLIGATORY", "should", "the sweeper should be used around mid"),
        ),
    },
    "mid-push-prevents-side-collapse": {
        "mentions": (
            ("allies", "ENTITY", "you guys", "you guys going four for Gwen"),
            ("enemy", "ENTITY", "enemy team", None),
            ("push", ("ACTION", "OUTCOME", "STATE"), "deep mid push", "if you guys got deep mid push"),
            ("rotate", "ACTION", "you guys going four for Gwen", None),
            ("cannot_punish", ("OUTCOME", "STATE"), "enemy team cannot punish you guys going four for Gwen", None),
        ),
        "edges": (
            ("allies_rotate", "allies", "rotate", ("ACTOR",)),
            ("enemy_experiences", "enemy", "cannot_punish", ("EXPERIENCER",)),
            ("push_condition", "push", "cannot_punish", ("CONDITION",)),
            ("safety_enables", "cannot_punish", "rotate", ("ENABLES",)),
        ),
        "qualifiers": (
            ("push_if", "push", "conditionality", "CONDITIONAL", "if", "if you guys got deep mid push"),
            ("punish_negative", "cannot_punish", "polarity", "NEGATIVE", "cannot", "enemy team cannot punish you guys going four for Gwen"),
        ),
        "references": (
            ("allies_unresolved", "allies", "INSUFFICIENT_EVIDENCE", ()),
        ),
    },
    "unwarded-bush-hook-risk": {
        "mentions": (
            ("allies", "ENTITY", "you guys", "you guys hard lose level one"),
            ("walker", "ENTITY", "you", "you don't walk like this"),
            ("walk", "ACTION", "walk like this", "you don't walk like this"),
            ("lose", ("OUTCOME", "STATE"), "you guys hard lose level one", None),
            ("win_condition", ("EVENT", "STATE"), "get hooked and land double Q on them", None),
            ("hooked", ("EVENT", "STATE"), "get hooked", None),
            ("double_q", ("ACTION", "EVENT"), "land double Q on them", None),
            ("win", ("OUTCOME", "STATE"), "win level one", None),
            ("ward_actor", "ENTITY", "you", "you are able to Ward"),
            ("ward", "ACTION", "Ward", "able to Ward without stepping into hook"),
            ("step_hook", ("ACTION", "EVENT"), "stepping into hook", None),
        ),
        "edges": (
            ("walker_walk", "walker", "walk", ("ACTOR",)),
            ("allies_lose", "allies", "lose", ("ACTOR",)),
            ("lose_prevents", "lose", "walk", ("PREVENTS",)),
            ("conjunctive_win_condition", "win_condition", "win", ("CONDITION",)),
            ("ward_actor_edge", "ward_actor", "ward", ("ACTOR",)),
            ("avoid_hook_condition", "step_hook", "ward", ("CONDITION",)),
        ),
        "qualifiers": (
            ("walk_negative", "walk", "polarity", "NEGATIVE", "don't", "you don't walk like this"),
            ("win_only", "win", "restriction", "EXCLUSIVE", "only", "the only way you guys can win level one"),
            ("win_if", "win_condition", "conditionality", "CONDITIONAL", "if", "if you get hooked and land double Q on them"),
            ("ward_can", "ward", "modality", "POSSIBLE", "can", "I mean you can if you are able to Ward"),
            ("avoid_hook_negative", "step_hook", "polarity", "NEGATIVE", "without", "without stepping into hook"),
        ),
        "references": (
            ("allies_unresolved", "allies", "INSUFFICIENT_EVIDENCE", ()),
            ("walker_unresolved", "walker", "INSUFFICIENT_EVIDENCE", ()),
            ("ward_actor_unresolved", "ward_actor", "INSUFFICIENT_EVIDENCE", ()),
        ),
    },
}


_MENTION_DIMENSION = {
    "ENTITY": "entity_recovery", "ABILITY_OR_RESOURCE": "ability_resource_recovery",
    "EVENT": "event_recovery", "ACTION": "action_recovery",
    "STATE": "state_outcome_recovery", "OUTCOME": "state_outcome_recovery",
    "QUANTITY": "quantity", "TIME": "temporal_edges",
    "LOCATION_OR_SPACE": "location_or_space",
}
_EDGE_DIMENSION = {
    "ACTOR": "actor_target_roles", "TARGET": "actor_target_roles",
    "OBJECT": "actor_target_roles", "EXPERIENCER": "actor_target_roles",
    "CAUSES": "causal_edges", "ENABLES": "causal_edges", "PREVENTS": "causal_edges",
    "REQUIRES": "causal_edges", "PURPOSE": "causal_edges", "RESULT": "causal_edges",
    "CONDITION": "condition_recovery", "TEMPORAL_BEFORE": "temporal_edges",
    "TEMPORAL_AFTER": "temporal_edges", "TEMPORAL_UNTIL": "temporal_edges",
    "TERMINATES": "temporal_edges", "CONTRASTS_WITH": "contrast",
    "NEGATES": "negation", "MODIFIES": "semantic_completeness",
    "REFERS_TO": "coreference",
}
_QUALIFIER_DIMENSION = {
    "polarity": "negation", "modality": "modality",
    "temporal_scope": "temporal_edges", "conditionality": "condition_recovery",
    "comparative_degree": "comparison", "uncertainty": "uncertainty",
    "restriction": "semantic_completeness",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _located_span(text: str, phrase: str, anchor: str | None) -> list[int]:
    if anchor is None:
        starts = [index for index in range(len(text)) if text.startswith(phrase, index)]
        if len(starts) != 1:
            raise ValueError(f"gold phrase is not unique: {phrase!r}")
        start = starts[0]
    else:
        anchor_start = text.index(anchor)
        relative = anchor.index(phrase)
        start = anchor_start + relative
        if text[start:start + len(phrase)] != phrase:
            raise ValueError("anchored reviewed phrase is not an exact source slice")
    return [start, start + len(phrase)]


def _case(case_id: str, source: Mapping[str, Any], annotation: Mapping[str, Any]) -> dict[str, Any]:
    text = source["source_text"]
    mentions = []
    mention_types = {}
    for mention_id, node_type, phrase, anchor in annotation["mentions"]:
        node_types = (node_type,) if isinstance(node_type, str) else tuple(node_type)
        mentions.append({
            "id": mention_id, "node_types": sorted(node_types),
            "acceptable_spans": [_located_span(text, phrase, anchor)], "critical": True,
        })
        mention_types[mention_id] = node_types
    edges = [{
        "id": edge_id, "source": left, "target": right,
        "edge_types": sorted(edge_types), "critical": True,
    } for edge_id, left, right, edge_types in annotation["edges"]]
    qualifiers = [{
        "id": qualifier_id, "mention": mention_id, "field": field, "value": value,
        "cue_spans": [_located_span(text, cue, anchor)], "critical": True,
    } for qualifier_id, mention_id, field, value, cue, anchor in annotation["qualifiers"]]
    references = [{
        "id": reference_id, "source": mention_id, "status": status,
        "targets": sorted(targets), "critical": True,
    } for reference_id, mention_id, status, targets in annotation.get("references", ())]
    questions = []
    for mention in mentions:
        node_type = mention["node_types"][0]
        questions.append({
            "id": "question-mention-" + mention["id"],
            "prompt": "Which exact source mention expresses " + mention["id"].replace("_", " ") + "?",
            "dimension": _MENTION_DIMENSION[node_type],
            "requires": ["mention:" + mention["id"]], "critical": True,
        })
    for edge in edges:
        questions.append({
            "id": "question-edge-" + edge["id"],
            "prompt": "What source-supported relation expresses " + edge["id"].replace("_", " ") + "?",
            "dimension": _EDGE_DIMENSION[edge["edge_types"][0]],
            "requires": ["edge:" + edge["id"]], "critical": True,
        })
    for qualifier in qualifiers:
        questions.append({
            "id": "question-qualifier-" + qualifier["id"],
            "prompt": "What grounded qualifier expresses " + qualifier["id"].replace("_", " ") + "?",
            "dimension": _QUALIFIER_DIMENSION[qualifier["field"]],
            "requires": ["qualifier:" + qualifier["id"]], "critical": True,
        })
    for reference in references:
        questions.append({
            "id": "question-reference-" + reference["id"],
            "prompt": "What is the supported reference status for " + reference["id"].replace("_", " ") + "?",
            "dimension": "coreference", "requires": ["reference:" + reference["id"]],
            "critical": True,
        })
    return {
        "id": case_id, "split": "LEGACY_FAILURE",
        "source_id": source["source_id"], "source_kind": "transcript",
        "source_text": text, "upstream_source_id": source["upstream_source_id"],
        "upstream_start": source["upstream_start"], "upstream_end": source["upstream_end"],
        "phenomena": sorted(set(source["phenomena"] + ["legacy_phase2e_failure"])),
        "exhaustive": False, "mentions": mentions, "edges": edges,
        "qualifiers": qualifiers, "references": references, "questions": questions,
    }


def build(db: Path, phase2d: Path, phase2e: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    phase2d_body = json.loads(phase2d.read_text(encoding="utf-8"))
    phase2e_body = json.loads(phase2e.read_text(encoding="utf-8"))
    fixture_cases = {item["id"]: item for item in phase2d_body["cases"]}
    artifact_cases = {item["case_id"]: item for item in phase2e_body["cases"]}
    sources = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for case_id in ANNOTATIONS:
            artifact_case = artifact_cases[case_id]
            if artifact_case.get("eligible") is not True:
                raise ValueError("legacy reviewed case is not eligible in the valid artifact")
            catalog = artifact_case["modes"][0]["candidate_catalog"]
            start = min(item["alignment"]["absolute_start"] for item in catalog)
            end = max(item["alignment"]["absolute_end"] for item in catalog)
            video_id = fixture_cases[case_id]["source_video_id"]
            row = connection.execute(
                "SELECT video_id, transcription FROM videos WHERE video_id = ?",
                (video_id,),
            ).fetchone()
            if row is None:
                raise ValueError("legacy source video is absent from primary bronze DB")
            full_text = str(row["transcription"])
            source_text = full_text[start:end]
            source = {
                "case_id": case_id, "source_id": "transcript:" + video_id,
                "upstream_source_id": video_id, "upstream_start": start,
                "upstream_end": end, "source_text": source_text,
                "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                "upstream_content_sha256": hashlib.sha256(full_text.encode()).hexdigest(),
                "phenomena": ["legacy_failure_regression"],
            }
            window = window_from_exact_span(
                BronzeSource(source["source_id"], full_text), start, end,
            )
            catalog_spans = {
                (item.start, item.end) for item in generate_mention_candidates(window)
            }
            reviewed_case = _case(case_id, source, ANNOTATIONS[case_id])
            missing = [
                mention["id"] for mention in reviewed_case["mentions"]
                if not any(tuple(span) in catalog_spans for span in mention["acceptable_spans"])
            ]
            if missing:
                raise ValueError(
                    f"legacy deterministic mention catalog misses {case_id}: {missing!r}",
                )
            sources.append(source)
    manifest_inner = {
        "schema_version": LEGACY_MANIFEST_VERSION,
        "purpose": "Exact bronze source windows for the five eligible Phase 2E architecture failures.",
        "input_hashes": {
            "database_sha256": _sha256(db), "phase2d_fixture_sha256": _sha256(phase2d),
            "phase2e_artifact_file_sha256": _sha256(phase2e),
            "phase2e_artifact_inner_sha256": phase2e_body["content_sha256"],
        },
        "windows": sources,
    }
    manifest = {"content_sha256": _canonical_sha256(manifest_inner), **manifest_inner}
    benchmark_inner = {
        "schema_version": BENCHMARK_SCHEMA_VERSION, "split": "LEGACY_FAILURE",
        "purpose": "Reviewed legacy regression: source semantic facts needed to reconstruct the five Phase 2E mechanisms.",
        "pool_manifest_sha256": manifest["content_sha256"],
        "cases": [_case(item["case_id"], item, ANNOTATIONS[item["case_id"]]) for item in sources],
    }
    benchmark = {"content_sha256": _canonical_sha256(benchmark_inner), **benchmark_inner}
    return manifest, benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--phase2d", type=Path, required=True)
    parser.add_argument("--phase2e-artifact", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    args = parser.parse_args()
    manifest, benchmark = build(args.db, args.phase2d, args.phase2e_artifact)
    for path, value in (
        (args.manifest_output, manifest), (args.benchmark_output, benchmark),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_semantic_benchmark(
        args.benchmark_output, expected_split="LEGACY_FAILURE",
        expected_content_sha256=benchmark["content_sha256"],
        expected_pool_manifest_sha256=manifest["content_sha256"],
    )
    print(json.dumps({
        "manifest_content_sha256": manifest["content_sha256"],
        "benchmark_content_sha256": benchmark["content_sha256"],
        "cases": len(benchmark["cases"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
