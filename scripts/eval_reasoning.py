"""Compare evidence-only and structured-context answers for Phase 1.

The harness snapshots ordinary RAG evidence once per case, then runs the same
configured answer model with strategic context disabled and enabled. The
default fixture database is temporary, so this experiment never mutates the
application's source evidence or production strategic cache.

Usage:
    uv run python -m scripts.eval_reasoning --live
    uv run python -m scripts.eval_reasoning --live --case caitlyn-vs-mage
    uv run python -m scripts.eval_reasoning --live --strategic-db /path/to/strategic.db
"""

from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest import mock

import core.database as database
from core.llm import BACKEND, MODEL
from core.strategic_types import load_strategic_fixture
from retrieval import query


@dataclass(frozen=True)
class ReasoningCase:
    id: str
    question: str
    mode: str
    champion_a: str
    champion_b: str | None = None
    allies: tuple[str, ...] = ()
    enemies: tuple[str, ...] = ()


PHASE_1_CASES = (
    ReasoningCase(
        "caitlyn-vs-mage",
        "How should Caitlyn use persistent control against an artillery mage's intermittent spell pressure and resource budget?",
        "general",
        "Caitlyn",
    ),
    ReasoningCase(
        "yunara-thresh-vs-tristana-yuumi",
        "How should Yunara and Thresh play vs Tristana and Yuumi through access versus continuity?",
        "team_matchup",
        "Yunara",
        allies=("Yunara", "Thresh"),
        enemies=("Tristana", "Yuumi"),
    ),
    ReasoningCase(
        "kaisa-conditional-access",
        "When does Kai'Sa have conditional access and concentrated damage, and when does access become a liability?",
        "general",
        "Kai'Sa",
    ),
    ReasoningCase(
        "sylas-hp-joust-second-rotation",
        "How should Sylas evaluate access, the HP joust, and threatened continuation into a second rotation?",
        "general",
        "Sylas",
    ),
)


@contextmanager
def fixture_database():
    """Yield a temporary DB containing only the manual strategic fixture."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "strategic-fixture.db"
        previous = database.DB_PATH
        try:
            database.DB_PATH = path
            database.init_db()
            database.persist_strategic_fixture(
                load_strategic_fixture("data/strategic_fixtures_v0.json")
            )
        finally:
            database.DB_PATH = previous
        yield (str(path),)


def _snapshot_evidence(case: ReasoningCase, top_k: int) -> tuple[dict, dict]:
    if case.mode == "general":
        evidence = query.retrieve(
            case.question,
            champion=case.champion_a,
            top_k=top_k,
        )
        return {"general": evidence}, {"type": "general"}
    if case.mode == "team_matchup":
        if len(case.allies) < 2 or len(case.enemies) < 2:
            raise ValueError("team matchup cases require two allied and two enemy champions")
        allies, enemies = query._retrieve_team_matchup(
            case.question,
            case.allies,
            case.enemies,
            role=None,
            game="lol",
            top_k=top_k,
        )
        return {"allies": allies, "enemies": enemies}, {
            "type": "team_matchup",
            "allies": case.allies,
            "enemies": case.enemies,
        }
    if case.mode not in {"matchup", "synergy"} or not case.champion_b:
        raise ValueError(f"unsupported evaluation case mode: {case.mode}")
    left, right = query.retrieve_duo(
        case.question,
        case.champion_a,
        case.champion_b,
        top_k=top_k,
    )
    return {"left": left, "right": right}, {
        "type": case.mode,
        "a": case.champion_a,
        "b": case.champion_b,
    }


@contextmanager
def _use_evidence_snapshot(snapshot: dict[str, list[dict]], intent: dict):
    if intent["type"] == "general":
        with mock.patch.object(query, "retrieve", return_value=snapshot["general"]):
            yield
        return
    if intent["type"] == "team_matchup":
        with mock.patch.object(query, "detect_intent", return_value=intent), mock.patch.object(
            query,
            "_retrieve_team_matchup",
            return_value=(snapshot["allies"], snapshot["enemies"]),
        ):
            yield
        return
    with mock.patch.object(query, "detect_intent", return_value=intent), mock.patch.object(
        query,
        "retrieve_duo",
        return_value=(snapshot["left"], snapshot["right"]),
    ):
        yield


def evaluate_case(
    case: ReasoningCase,
    *,
    strategic_db_paths: tuple[str, ...],
    top_k: int = query.TOP_K,
    run_model: bool = False,
) -> dict:
    """Build a comparable pair from exactly one RAG evidence snapshot."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if case.mode == "team_matchup" and top_k < 4:
        raise ValueError("team matchup top_k must be at least 4")
    snapshot, intent = _snapshot_evidence(case, top_k)
    result = {
        "case": asdict(case),
        "answer_model": {"backend": BACKEND, "model": MODEL},
        "base_evidence": snapshot,
        "strategic_db_paths": list(strategic_db_paths),
        "baseline_answer": None,
        "structured_answer": None,
    }
    if not run_model:
        return result

    answer_kwargs = {
        "show_sources": False,
        "top_k": top_k,
        "strategic_db_paths": strategic_db_paths,
    }
    if intent["type"] == "general":
        answer_kwargs["champion"] = case.champion_a
    with _use_evidence_snapshot(snapshot, intent):
        result["baseline_answer"] = query.answer(
            case.question,
            include_strategic_context=False,
            **answer_kwargs,
        )
        result["structured_answer"] = query.answer(
            case.question,
            include_strategic_context=True,
            **answer_kwargs,
        )
    return result


def render_comparison(result: dict) -> str:
    case = result["case"]
    header = f"## {case['id']}\n\nQuestion: {case['question']}"
    metadata = (
        f"\n\nModel: {result['answer_model']['backend']} / "
        f"{result['answer_model']['model']}\n"
    )
    evidence_count = _evidence_count(result["base_evidence"])
    evidence = f"Base RAG evidence snapshot: {evidence_count} row(s), shared by both variants.\n"
    baseline = result["baseline_answer"] or "(model not run; use --live)"
    structured = result["structured_answer"] or "(model not run; use --live)"
    return (
        header
        + metadata
        + evidence
        + "\n### Baseline: RAG evidence only\n"
        + baseline
        + "\n\n### Structured: same RAG evidence + derived strategic context\n"
        + structured
    )


def _evidence_count(snapshot: dict) -> int:
    return sum(
        len(rows) if isinstance(rows, list) else sum(len(member_rows) for member_rows in rows.values())
        for rows in snapshot.values()
    )


def _selected_cases(ids: list[str]) -> tuple[ReasoningCase, ...]:
    if not ids:
        return PHASE_1_CASES
    selected = tuple(case for case in PHASE_1_CASES if case.id in ids)
    unknown = sorted(set(ids) - {case.id for case in selected})
    if unknown:
        raise ValueError(f"unknown evaluation case(s): {', '.join(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 strategic reasoning comparison")
    parser.add_argument("--case", action="append", dest="case_ids", help="Case ID; repeatable")
    parser.add_argument("--strategic-db", action="append", dest="strategic_dbs", help="Existing strategic SQLite DB; repeatable")
    parser.add_argument("--live", action="store_true", help="Call the configured answer model")
    parser.add_argument("--top-k", type=_positive_int, default=query.TOP_K)
    parser.add_argument("--json-output", type=Path, help="Write comparable JSON results")
    args = parser.parse_args()

    cases = _selected_cases(args.case_ids or [])
    if args.strategic_dbs:
        paths_context = _static_paths(tuple(args.strategic_dbs))
    else:
        paths_context = fixture_database()
    with paths_context as strategic_paths:
        results = [
            evaluate_case(
                case,
                strategic_db_paths=strategic_paths,
                top_k=args.top_k,
                run_model=args.live,
            )
            for case in cases
        ]
    print("\n\n".join(render_comparison(result) for result in results))
    if args.json_output:
        args.json_output.write_text(json.dumps(results, indent=2) + "\n")


@contextmanager
def _static_paths(paths: tuple[str, ...]):
    yield paths


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


if __name__ == "__main__":
    main()
