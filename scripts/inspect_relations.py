"""Inspect persisted Phase 1/2 relations and their source evidence.

Examples:
  uv run python -m scripts.inspect_relations --db videos.db --champion Thresh
  uv run python -m scripts.inspect_relations --db videos.db --concept access
  uv run python -m scripts.inspect_relations --db videos.db --evidence-id 4807
  uv run python -m scripts.inspect_relations --review-file /tmp/extraction.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def load_relations(
    db_path: Path,
    *,
    champion: str | None = None,
    concept: str | None = None,
    evidence_id: str | None = None,
    minimum_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Load relations with their evidence, keeping derived and raw data explicit."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM strategic_relations WHERE confidence >= ? ORDER BY confidence DESC, id",
            (minimum_confidence,),
        ).fetchall()
        results = []
        for row in rows:
            relation = _decode_relation(dict(row))
            relation["evidence_refs"] = [
                dict(ref)
                for ref in conn.execute(
                    """SELECT source_type, source_id, insight_id, quote
                       FROM strategic_relation_evidence WHERE relation_id = ?
                       ORDER BY source_type, source_id, insight_id""",
                    (relation["id"],),
                )
            ]
            if _matches(relation, champion, concept, evidence_id):
                results.append(relation)
    return results


def load_review_decisions(path: Path) -> list[dict[str, Any]]:
    """Load non-accepted decisions emitted by extract_relations dry-run JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        raise ValueError("review file is not extract_relations JSON")
    return [item for item in payload["decisions"] if item.get("status") != "accepted"]


def _decode_relation(relation: dict[str, Any]) -> dict[str, Any]:
    for field in ("condition_json", "condition_event_json", "effect_json", "alignment_json", "concepts"):
        raw = relation.pop(field, "[]" if field == "alignment_json" else "null" if field == "condition_event_json" else '\"\"')
        relation[field.removesuffix("_json")] = json.loads(raw)
    return relation


def _matches(relation: dict[str, Any], champion: str | None, concept: str | None, evidence_id: str | None) -> bool:
    haystack = " ".join((relation["subject_key"], relation["object_key"])).casefold()
    if champion and champion.casefold() not in haystack:
        return False
    if concept and concept.casefold() not in {item.casefold() for item in relation["concepts"]} | {
        relation["subject_key"].casefold(), relation["object_key"].casefold()
    }:
        return False
    return not evidence_id or any(ref.get("insight_id") == str(evidence_id) for ref in relation["evidence_refs"])


def render_relations(relations: list[dict[str, Any]]) -> str:
    if not relations:
        return "No matching persisted relations."
    lines = []
    for relation in relations:
        condition = f" | condition: {relation['condition']}" if relation["condition"] else ""
        effect = f" | effect: {relation['effect']}" if relation["effect"] else ""
        lines.extend(
            (
                f"{relation['id']} [{relation['data_version']}, {relation['ontology_version']}]",
                f"  {relation['subject_key']} --{relation['relation_type']}--> {relation['object_key']}{condition}{effect}",
                "  condition event: " + json.dumps(relation["condition_event"], sort_keys=True) if relation["condition_event"] else "  condition event: none",
                f"  confidence={relation['confidence']:.2f}; provenance={relation['provenance_type']}; patch={relation['patch_sensitivity']}",
                f"  concepts: {', '.join(relation['concepts'])}",
                "  alignments: " + "; ".join(
                    f"{item['field']}={item['source_text']!r} -> {item['canonical_value']} ({item['mapping_type']})"
                    for item in relation["alignment"]
                ),
                "  evidence: " + "; ".join(
                    f"{ref['source_type']}:{ref['source_id']} insight={ref['insight_id'] or '-'}"
                    for ref in relation["evidence_refs"]
                ),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect provenanced strategic relations")
    parser.add_argument("--db", type=Path, default=Path("videos.db"))
    parser.add_argument("--champion")
    parser.add_argument("--concept")
    parser.add_argument("--evidence-id")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.review_file:
        decisions = load_review_decisions(args.review_file)
        print(json.dumps(decisions, indent=2))
        return
    relations = load_relations(
        args.db, champion=args.champion, concept=args.concept,
        evidence_id=args.evidence_id, minimum_confidence=args.min_confidence,
    )
    print(json.dumps(relations, indent=2) if args.json else render_relations(relations))


if __name__ == "__main__":
    main()
