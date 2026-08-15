"""Run the bounded Phase 2 relation compiler over explicit source insight IDs.

Examples:
  uv run python -m scripts.extract_relations --db videos.db --insight-id 4807
  uv run python -m scripts.extract_relations --db videos.db --insight-id 4807 --apply
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import core.database as database
from core.llm import BACKEND, MODEL, chat
from pipeline.relation_extract import extract_relations, packet_from_insight_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract provenanced strategic relations from explicit insight IDs")
    parser.add_argument("--db", type=Path, default=Path("videos.db"))
    parser.add_argument("--insight-id", action="append", required=True, dest="insight_ids")
    parser.add_argument("--apply", action="store_true", help="Persist accepted relations; default is dry-run")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    packet = packet_from_insight_ids(str(args.db), args.insight_ids)
    decisions = extract_relations(packet, chat, acceptance_threshold=args.threshold)
    payload = {
        "mode": "apply" if args.apply else "dry_run",
        "model": {"backend": BACKEND, "model": MODEL},
        "evidence": [asdict(item) for item in packet.evidence],
        "ability_aliases": dict(packet.ability_aliases),
        "decisions": [_decision_json(item) for item in decisions],
        "persistence_action": "persist accepted relations" if args.apply else "no mutation",
    }
    if args.apply:
        previous = database.DB_PATH
        try:
            database.DB_PATH = args.db
            database.persist_strategic_relations(decisions)
        finally:
            database.DB_PATH = previous
    print(json.dumps(payload, indent=2))
    if args.json_output:
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _decision_json(decision):
    relation = asdict(decision.relation) if decision.relation else None
    return {
        "raw_relation": dict(decision.raw),
        "canonical_relation": relation,
        "status": decision.status,
        "validation_warnings": list(decision.warnings),
        "confidence_components": dict(decision.confidence_components),
    }


if __name__ == "__main__":
    main()
