#!/usr/bin/env python3
"""Run the preregistered Phase 2F strong-reference semantic-IR gates.

The default mode performs every source/hash/isolation check without calling a
provider. ``--live`` is required for external inference. This runner never
persists production graph state; it writes only reconstructible evaluation
artifacts to a new caller-selected directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Mapping

# Permit both ``python -m scripts.eval_phase2f_semantic_ir`` and direct
# execution from an arbitrary working directory without requiring PYTHONPATH.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.semantic_compiler import (
    SemanticCompilerConfig, compile_source_semantic_ir,
)
from pipeline.semantic_ir_artifact import build_semantic_run_artifact
from pipeline.semantic_ir_evaluation import (
    BENCHMARK_SCHEMA_VERSION, SemanticBenchmark, evaluate_semantic_benchmark,
    load_semantic_benchmark,
)
from pipeline.semantic_ir_pool import load_semantic_window_pool
from pipeline.semantic_source import BronzeSource, window_from_exact_span
from scripts.build_phase2f_legacy_benchmark import build as build_legacy_benchmark


LEGACY_MANIFEST_SHA256 = "cf86dde955f4cbeee091f38aab8293256b0c48f809c969384185a330ee511241"
LEGACY_BENCHMARK_SHA256 = "a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd"
REPRESENTATIVE_POOL_SHA256 = "9b89c6d6c6c8070eba48d6db47254e156c1b2591c1480a60f98a1e8d789491c2"
REFERENCE_MODEL = "deepseek-v4-pro"
REFERENCE_THINKING = "disabled"
REFERENCE_ENDPOINT = "https://api.deepseek.com"
ABILITY_ALIASES = (
    "Q", "W", "E", "R", "ult", "ultimate", "Flash", "Teleport",
    "Ignite", "Exhaust", "Ward", "Sweeper",
)
LEGACY_RUN_VERSION = "phase2f-legacy-strong-reference-run-v1"
LEGACY_GATE_VERSION = "phase2f-legacy-five-gate-v1-strict-complete"
LEGACY_CASE_CHECKSUM_DENOMINATORS = {
    "wave-reset-after-kill": 12,
    "push-poke-wave-crash": 12,
    "sweeper-limits-mid-play": 14,
    "mid-push-prevents-side-collapse": 12,
    "unwarded-bush-hook-risk": 25,
}
LEGACY_CASE_METRIC_DENOMINATORS = {
    "wave-reset-after-kill": (5, 4, 1, 2, 12),
    "push-poke-wave-crash": (5, 4, 1, 2, 12),
    "sweeper-limits-mid-play": (7, 6, 1, 0, 14),
    "mid-push-prevents-side-collapse": (5, 4, 2, 1, 12),
    "unwarded-bush-hook-risk": (11, 6, 5, 3, 25),
}
LEGACY_METRIC_DENOMINATORS = {
    "mention_candidate_coverage": 33,
    "mention_selection_recall": 33,
    "mention_type_recall": 33,
    "edge_pair_coverage": 24,
    "edge_recall": 24,
    "qualifier_candidate_coverage": 10,
    "qualifier_recall": 10,
    "reference_candidate_coverage": 8,
    "reference_recall": 8,
    "semantic_completeness": 75,
    "semantic_checksum": 75,
}
_SAFE_FAILURE_CODES = {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}


def reference_config() -> SemanticCompilerConfig:
    return SemanticCompilerConfig.create(
        REFERENCE_MODEL,
        provider_configuration={
            "provider": "deepseek",
            "endpoint": REFERENCE_ENDPOINT,
            "purpose": "phase2f-representation-viability-reference",
        },
        thinking=REFERENCE_THINKING,
        mention_partition_size=600,
        mention_max_tokens=2048,
        qualifier_max_tokens=512,
        coreference_max_tokens=256,
        edge_max_tokens=256,
        coreference_max_segment_distance=2,
        edge_max_character_distance=600,
        edge_max_segment_distance=2,
    )


def evaluate_legacy_gate(
    evaluation: Mapping[str, Any], *, benchmark: SemanticBenchmark,
    runs: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute, bind, then apply the immutable strict five-case gate."""
    reasons: list[str] = []
    try:
        recomputed = evaluate_semantic_benchmark(
            benchmark, runs, expected_content_sha256=LEGACY_BENCHMARK_SHA256,
        )
    except Exception as exc:
        return {
            "gate_version": LEGACY_GATE_VERSION, "passed": False,
            "reasons": [f"typed run evaluation could not be reconstructed: {type(exc).__name__}"],
            "non_exhaustive_invention_policy": (
                "Unscored extra nodes and edges are reported, not credited as supported or "
                "counted as invention; exhaustive DEV/FROZEN owns invention acceptance."
            ),
        }
    if _canonical_json(evaluation) != _canonical_json(recomputed):
        reasons.append("evaluation does not equal reconstruction from typed retained runs")
    if evaluation.get("benchmark_schema_version") != BENCHMARK_SCHEMA_VERSION \
            or evaluation.get("benchmark_content_sha256") != LEGACY_BENCHMARK_SHA256:
        reasons.append("evaluation is not bound to the locked legacy benchmark")
    if evaluation.get("split") != "LEGACY_FAILURE" or evaluation.get("case_count") != 5:
        reasons.append("evaluation does not exactly contain the five legacy cases")
    for name, denominator in LEGACY_METRIC_DENOMINATORS.items():
        metric = evaluation.get(name)
        if not _exact_perfect_metric(metric, denominator):
            reasons.append(f"{name} is not exactly {denominator}/{denominator}")
    for name in ("source_span_validity", "edge_provenance_traceability"):
        metric = evaluation.get(name)
        if not _exact_perfect_metric(metric, 5):
            reasons.append(f"{name} is not 5/5")
    cases = evaluation.get("cases")
    observed_failures: dict[str, int] = {}
    if not isinstance(cases, list) or {
        item.get("case_id") for item in cases if isinstance(item, Mapping)
    } != set(LEGACY_CASE_CHECKSUM_DENOMINATORS) or len(cases) != 5:
        reasons.append("per-case evidence is incomplete")
    else:
        metric_sums = {
            name: [0, 0] for name in LEGACY_METRIC_DENOMINATORS
        }
        for case in cases:
            if not isinstance(case, Mapping) or case.get("case_id") not in LEGACY_CASE_CHECKSUM_DENOMINATORS:
                reasons.append("per-case evidence is malformed")
                continue
            case_id = case["case_id"]
            mentions, edges, qualifiers, references, facts = \
                LEGACY_CASE_METRIC_DENOMINATORS[case_id]
            expected_case_denominators = {
                "mention_candidate_coverage": mentions,
                "mention_selection_recall": mentions,
                "mention_type_recall": mentions,
                "edge_pair_coverage": edges,
                "edge_recall": edges,
                "qualifier_candidate_coverage": qualifiers,
                "qualifier_recall": qualifiers,
                "reference_candidate_coverage": references,
                "reference_recall": references,
                "semantic_completeness": facts,
                "semantic_checksum": facts,
            }
            for name, denominator in expected_case_denominators.items():
                if not _exact_perfect_metric(case.get(name), denominator):
                    reasons.append(f"{case_id} {name} is not exactly {denominator}/{denominator}")
            checksum = case.get("semantic_checksum")
            if not _exact_perfect_metric(
                checksum, LEGACY_CASE_CHECKSUM_DENOMINATORS[case_id],
            ):
                reasons.append(f"{case_id} semantic checksum has the wrong locked cardinality")
            if case.get("status") not in {"OK", "PARTIAL"}:
                reasons.append(f"{case_id} has no usable semantic graph")
            if case.get("source_span_validity") is not True:
                reasons.append(f"{case_id} has invalid source spans")
            if case.get("edge_provenance_traceability") is not True:
                reasons.append(f"{case_id} has untraceable edge provenance")
            for name in LEGACY_METRIC_DENOMINATORS:
                metric = case.get(name)
                if not _valid_metric(metric):
                    reasons.append(f"{case_id} {name} is malformed")
                    continue
                metric_sums[name][0] += metric["hit_count"]
                metric_sums[name][1] += metric["denominator"]
            questions, recovered = case.get("questions"), case.get("recovered_facts")
            if not isinstance(questions, list) or len(questions) != facts or any(
                not isinstance(item, Mapping)
                or item.get("answerable_from_bronze") is not True
                or item.get("answerable_from_ir") is not True
                or item.get("missing_requirements") != []
                for item in questions
            ):
                reasons.append(f"{case_id} checksum question evidence is incomplete")
            if not isinstance(recovered, list) or len(recovered) != facts \
                    or len(set(recovered)) != facts:
                reasons.append(f"{case_id} recovered fact evidence is incomplete")
            for field in ("dimensions", "critical_dimensions", "mention_families"):
                nested = case.get(field)
                if not isinstance(nested, Mapping) or any(
                    not _perfect_nonzero_or_empty_metric(metric)
                    for metric in nested.values()
                ):
                    reasons.append(f"{case_id} {field} evidence is incomplete")
            case_failures = case.get("failures")
            if not isinstance(case_failures, list):
                reasons.append(f"{case_id} failure evidence is malformed")
                continue
            for failure in case_failures:
                if not isinstance(failure, Mapping) or set(failure) != {
                    "code", "fact_id", "critical", "stage", "detail",
                } or not isinstance(failure.get("code"), str) \
                        or not isinstance(failure.get("detail"), str) \
                        or not failure["detail"] or not isinstance(failure.get("critical"), bool):
                    reasons.append(f"{case_id} failure evidence is malformed")
                    continue
                code = failure["code"]
                observed_failures[code] = observed_failures.get(code, 0) + 1
                if code not in _SAFE_FAILURE_CODES:
                    reasons.append(f"{code} occurred in {case_id}")
                elif failure["detail"] != code or failure.get("stage") not in {
                    "mentions", "qualifiers", "coreference", "edges",
                } or failure.get("critical") is not False \
                        or not isinstance(failure.get("fact_id"), str):
                    reasons.append(f"{case_id} safe abstention evidence is contradictory")
            expected_status = "PARTIAL" if case_failures else "OK"
            if case.get("status") != expected_status:
                reasons.append(f"{case_id} status contradicts retained failures")
            expected_first_loss = case_failures[0]["code"] if case_failures else None
            if case.get("first_loss") not in ({None} if expected_first_loss is None else _SAFE_FAILURE_CODES):
                reasons.append(f"{case_id} first-loss evidence is contradictory")
        for name, (hit, denominator) in metric_sums.items():
            aggregate = evaluation.get(name)
            if isinstance(aggregate, Mapping) and (
                aggregate.get("hit_count"), aggregate.get("denominator")
            ) != (hit, denominator):
                reasons.append(f"{name} does not reconcile with per-case evidence")
    failures = evaluation.get("failure_counts")
    if not isinstance(failures, Mapping):
        reasons.append("failure taxonomy is unavailable")
    else:
        if failures != dict(sorted(observed_failures.items())):
            reasons.append("aggregate failure taxonomy does not reconcile with case evidence")
        for code, count in failures.items():
            if not isinstance(code, str) or isinstance(count, bool) or not isinstance(count, int) \
                    or count <= 0 or code not in _SAFE_FAILURE_CODES:
                reasons.append(f"failure taxonomy contains prohibited or malformed entry: {code}")
        if failures.get("INSUFFICIENT_EVIDENCE", 0) < 8:
            reasons.append("reviewed unresolved references lack retained insufficiency evidence")
    return {
        "gate_version": LEGACY_GATE_VERSION,
        "passed": not reasons,
        "reasons": reasons,
        "non_exhaustive_invention_policy": (
            "Unscored extra nodes and edges are reported, not credited as supported or "
            "counted as invention; exhaustive DEV/FROZEN owns invention acceptance."
        ),
    }


def _valid_metric(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"hit_count", "denominator", "rate"}:
        return False
    hit, denominator, rate = value["hit_count"], value["denominator"], value["rate"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (hit, denominator)) \
            or hit < 0 or denominator < 0 or hit > denominator:
        return False
    expected = hit / denominator if denominator else None
    if rate is not None and (
        isinstance(rate, bool) or not isinstance(rate, (int, float))
    ):
        return False
    return rate == expected


def _exact_perfect_metric(value: object, denominator: int) -> bool:
    if not _valid_metric(value) or value["hit_count"] != denominator \
            or value["denominator"] != denominator:
        return False
    return value["rate"] == (1.0 if denominator else None)


def _perfect_nonzero_or_empty_metric(value: object) -> bool:
    return _valid_metric(value) and (
        (value["denominator"] == 0 and value["hit_count"] == 0 and value["rate"] is None)
        or (
            value["denominator"] > 0 and value["hit_count"] == value["denominator"]
            and value["rate"] == 1.0
        )
    )


def validate_inputs(args: argparse.Namespace) -> tuple[
    Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], SemanticBenchmark,
]:
    manifest = _strict_json_file(args.manifest)
    benchmark_body = _strict_json_file(args.benchmark)
    pool = load_semantic_window_pool(args.pool)
    if manifest.get("content_sha256") != LEGACY_MANIFEST_SHA256:
        raise ValueError("legacy manifest does not match the preregistered content lock")
    if benchmark_body.get("content_sha256") != LEGACY_BENCHMARK_SHA256:
        raise ValueError("legacy benchmark does not match the preregistered content lock")
    if pool.get("content_sha256") != REPRESENTATIVE_POOL_SHA256:
        raise ValueError("representative pool does not match its preregistered content lock")
    rebuilt_manifest, rebuilt_benchmark = build_legacy_benchmark(
        args.db, args.phase2d, args.phase2e_artifact,
    )
    if rebuilt_manifest != manifest or rebuilt_benchmark != benchmark_body:
        raise ValueError("legacy source/benchmark artifacts do not reproduce from immutable inputs")
    benchmark = load_semantic_benchmark(
        args.benchmark,
        expected_split="LEGACY_FAILURE",
        expected_content_sha256=LEGACY_BENCHMARK_SHA256,
        expected_pool_manifest_sha256=LEGACY_MANIFEST_SHA256,
    )
    return manifest, benchmark_body, pool, benchmark


def run_live(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    pool: Mapping[str, Any],
    benchmark: SemanticBenchmark,
) -> tuple[Path, Mapping[str, Any]]:
    repo = Path(__file__).resolve().parents[1]
    try:
        args.output.resolve().relative_to(repo.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("evaluation output must be outside the source repository")
    if args.output.exists():
        raise ValueError("output directory already exists; evaluation artifacts are immutable")
    git_commit, repository_dirty = _git_state(repo)
    if repository_dirty:
        raise ValueError("the preregistered live run requires a clean committed worktree")
    import core.llm as llm
    endpoint = getattr(llm, "_DEEPSEEK_BASE_URL", None)
    if llm.BACKEND != "deepseek" or endpoint != REFERENCE_ENDPOINT:
        raise ValueError(
            "the preregistered reference run requires the official DeepSeek provider endpoint",
        )
    config = reference_config()
    entity_aliases = tuple(pool["selection_policy"]["champion_names"])
    windows = _reconstruct_windows(args.db, benchmark)
    runs = {}
    for index, case in enumerate(benchmark.cases, 1):
        print(f"[phase2f] compiling legacy case {index}/5: {case.case_id}", flush=True)
        runs[case.case_id] = compile_source_semantic_ir(
            windows[case.case_id], llm.chat, config=config,
            entity_aliases=entity_aliases, ability_aliases=ABILITY_ALIASES,
        )
    evaluation = evaluate_semantic_benchmark(
        benchmark, runs, expected_content_sha256=LEGACY_BENCHMARK_SHA256,
    )
    gate = evaluate_legacy_gate(evaluation, benchmark=benchmark, runs=runs)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    common_hashes = {
        "database_sha256": manifest["input_hashes"]["database_sha256"],
        "phase2d_fixture_sha256": manifest["input_hashes"]["phase2d_fixture_sha256"],
        "phase2e_artifact_sha256": manifest["input_hashes"]["phase2e_artifact_file_sha256"],
        "legacy_manifest_sha256": LEGACY_MANIFEST_SHA256,
        "legacy_benchmark_sha256": LEGACY_BENCHMARK_SHA256,
        "representative_pool_sha256": REPRESENTATIVE_POOL_SHA256,
    }
    parent = args.output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=args.output.name + ".tmp-", dir=parent))
    try:
        artifacts = []
        case_metrics = {item["case_id"]: item for item in evaluation["cases"]}
        for case in benchmark.cases:
            artifact = build_semantic_run_artifact(
                runs[case.case_id], git_commit=git_commit,
                repository_dirty=repository_dirty, created_at=created_at,
                input_hashes=common_hashes, evaluation=case_metrics[case.case_id],
            )
            filename = case.case_id + ".semantic-run.json"
            _write_exact(temporary / filename, artifact.to_json())
            artifacts.append({
                "case_id": case.case_id, "file": filename,
                "content_sha256": artifact.content_sha256,
                "file_sha256": artifact.file_sha256,
                "run_status": runs[case.case_id].status,
            })
        inner = {
            "run_version": LEGACY_RUN_VERSION,
            "created_at": created_at,
            "git_commit": git_commit,
            "repository_dirty": repository_dirty,
            "provider": "deepseek", "provider_endpoint": REFERENCE_ENDPOINT,
            "compiler_config": asdict(config),
            "entity_aliases_sha256": _canonical_sha256(list(entity_aliases)),
            "ability_aliases": list(ABILITY_ALIASES),
            "input_hashes": common_hashes,
            "artifacts": artifacts,
            "evaluation": evaluation,
            "gate": gate,
        }
        aggregate = {"content_sha256": _canonical_sha256(inner), **inner}
        _write_exact(temporary / "legacy-evaluation.json", _canonical_json(aggregate))
        os.replace(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return args.output, aggregate


def _reconstruct_windows(
    db: Path, benchmark: SemanticBenchmark,
) -> dict[str, Any]:
    windows = {}
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
        for index, case in enumerate(benchmark.cases, 1):
            row = connection.execute(
                "SELECT transcription FROM videos WHERE video_id = ? AND game = 'lol'",
                (case.upstream_source_id,),
            ).fetchone()
            if row is None or not isinstance(row[0], str):
                raise ValueError("legacy bronze source is absent from the primary database")
            source = BronzeSource(case.source_id, row[0])
            window = window_from_exact_span(
                source, case.upstream_start, case.upstream_end, index=index,
            )
            if window.text != case.source_text:
                raise ValueError("legacy source window does not match reviewed bronze")
            windows[case.case_id] = window
    return windows


def _git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip())
    return commit, dirty


def _strict_json_file(path: Path) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load trusted JSON input: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"trusted JSON input is not an object: {path}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_exact(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--phase2d", type=Path, required=True)
    parser.add_argument("--phase2e-artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest, _, pool, benchmark = validate_inputs(args)
    if not args.live:
        print(_canonical_json({
            "status": "VALIDATED_NO_PROVIDER_CALL", "cases": len(benchmark.cases),
            "legacy_manifest_sha256": LEGACY_MANIFEST_SHA256,
            "legacy_benchmark_sha256": LEGACY_BENCHMARK_SHA256,
            "representative_pool_sha256": REPRESENTATIVE_POOL_SHA256,
            "compiler_config": asdict(reference_config()),
        }))
        return 0
    if args.output is None:
        raise ValueError("--output is required with --live")
    output, aggregate = run_live(args, manifest, pool, benchmark)
    passed = bool(aggregate["gate"]["passed"])
    print(_canonical_json({
        "status": "PASSED" if passed else "FAILED", "output": str(output),
        "content_sha256": aggregate["content_sha256"],
        "gate_passed": passed,
        "gate_reasons": aggregate["gate"]["reasons"],
    }))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
