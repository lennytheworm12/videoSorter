#!/usr/bin/env python3
"""Run the Phase 2G controlled three-condition endpoint-recovery ablation.

Default mode performs the complete no-provider validation: benchmark content
lock, 33/33 reviewed exact endpoint coverage, 8 status tasks, silver fixture
invariants, 15 deterministic requests, and per-case artifact records that carry
the full candidate catalogs (phase-2f ids, local/upstream absolute offsets,
bronze text, segment provenance) plus expected task definitions -- without
calling a provider.  ``--live`` requires ``--output`` (a not-yet-existing
directory outside the repository) and uses the official DeepSeek endpoint with
``deepseek-v4-pro``, thinking disabled, temperature 0, one provider call per
case per condition (15 calls).  ``--compare A B`` checks that a clean rerun
preserves deterministic inputs and score/failure distributions while
timestamps/raw-output hashes may differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2g_endpoint_recovery import (
    CONDITIONS,
    ENDPOINT_RECOVERY_SYSTEM,
    REFERENCE_ENDPOINT,
    REFERENCE_MODEL,
    REFERENCE_THINKING,
    RUN_VERSION,
    assemble_case_record,
    build_aggregate,
    build_case_experiment,
    build_request,
    compare_artifacts,
    load_benchmark,
    publish_artifact,
    run_experiment,
    validate_catalog_records,
    validate_experiment_coverage,
)
from pipeline.phase2g_silver import (
    CONDITIONS as SILVER_CONDITIONS,
    SILVER_FIXTURE_CONTENT_SHA256,
    condition_text,
    load_silver_fixture,
    validate_fixture_against_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"
DEFAULT_SILVER = ROOT / "data/phase2g_silver_v1.json"


def _validate(
    benchmark_path: Path, silver_path: Path, *, verbose: bool = False,
) -> None:
    benchmark = load_benchmark(benchmark_path)
    fixture = load_silver_fixture(silver_path)
    validate_fixture_against_benchmark(benchmark, fixture)
    experiments = {
        case["id"]: build_case_experiment(case) for case in benchmark["cases"]
    }
    coverage = validate_experiment_coverage(experiments)
    status_total = sum(
        experiment["expected_status_count"] for experiment in experiments.values()
    )
    build_requests = lambda: {
        condition: {
            case["id"]: build_request(
                experiments[case["id"]],
                _condition_text(benchmark, case["id"], condition, fixture),
                condition=condition,
            )["request_sha256"]
            for case in benchmark["cases"]
        }
        for condition in CONDITIONS
    }
    requests = build_requests()
    if requests != build_requests():
        raise ValueError("Phase 2G requests are not deterministic across reruns")
    # Every per-case condition artifact record must carry the full catalog and
    # expected task definitions with all validated Bronze-bound fields.
    for case in benchmark["cases"]:
        experiment = experiments[case["id"]]
        validate_catalog_records(
            experiment["catalog"], experiment["bronze_text"],
            experiment["upstream_start"],
        )
        for condition in CONDITIONS:
            text = _condition_text(benchmark, case["id"], condition, fixture)
            request = build_request(experiment, text, condition=condition)
            record = assemble_case_record(
                experiment, case, condition, fixture,
                text=text, request=request, raw_response=None,
                provider_failure=None,
                parse_error="no-provider structural validation",
                parsed=None,
            )
            for field in (
                "catalog", "expected_endpoint_tasks", "expected_status_tasks",
                "input",
            ):
                if field not in record:
                    raise ValueError(
                        f"{condition}/{case['id']} artifact misses {field!r}",
                    )
            for catalog_record in record["catalog"]:
                for field in (
                    "alias", "candidate_id", "start", "end", "absolute_start",
                    "absolute_end", "text", "segment_ids",
                ):
                    if field not in catalog_record:
                        raise ValueError(
                            f"{condition}/{case['id']} catalog misses {field!r}",
                        )
    print(f"[phase2g] no-provider validation passed")
    print(f"[phase2g] benchmark content sha256: {benchmark['content_sha256']}")
    print(f"[phase2g] silver fixture content sha256: {fixture['content_sha256']}")
    print(
        f"[phase2g] endpoint coverage: "
        f"{coverage['candidate_coverage']['hit_count']}/"
        f"{coverage['candidate_coverage']['denominator']}",
    )
    print(
        f"[phase2g] status tasks: {status_total}/8; conditions: "
        f"{', '.join(conditions_label())} ({len(CONDITIONS)}); "
        f"15 deterministic case-level requests",
    )
    print(
        "[phase2g] artifact records carry full catalogs with phase-2f candidate "
        "ids, window-local and upstream absolute offsets, bronze text, and "
        "segment provenance plus expected endpoint/status task definitions",
    )
    if verbose:
        for condition, cases in requests.items():
            print(f"[phase2g] {condition}: {len(cases)} requests, sample sha {cases[next(iter(cases))][:16]}...")


def conditions_label() -> list[str]:
    return list(CONDITIONS)


def _condition_text(benchmark, case_id, condition, fixture) -> str:
    case = next(item for item in benchmark["cases"] if item["id"] == case_id)
    return condition_text(case, condition, fixture)


def _live(args: argparse.Namespace) -> int:
    if args.output is None:
        raise SystemExit("--live requires --output (a new directory outside the repo)")
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("evaluation output must be outside the source repository")
    if output.exists():
        raise SystemExit("output directory already exists; evaluation artifacts are immutable")
    benchmark = load_benchmark(args.benchmark)
    fixture = load_silver_fixture(args.silver)
    import core.llm as llm
    endpoint = getattr(llm, "_DEEPSEEK_BASE_URL", None)
    if getattr(llm, "BACKEND", None) != "deepseek" or endpoint != REFERENCE_ENDPOINT:
        raise SystemExit(
            "the Phase 2G reference run requires the official DeepSeek provider endpoint",
        )
    result = run_experiment(benchmark, fixture, llm.chat)
    aggregate = build_aggregate(
        args.benchmark, args.silver, result, repo=ROOT, provider="deepseek",
    )
    publish_artifact(output, aggregate)
    gate = aggregate["promotion_gate"]
    print(f"[phase2g] published immutable artifact: {output}")
    print(
        f"[phase2g] promotion gate passed={gate['passed']} "
        f"satisfied_conditions={gate['satisfied_conditions']}",
    )
    for condition in CONDITIONS:
        metrics = aggregate["conditions"][condition]["metrics"]
        print(
            f"[phase2g] {condition}: recall={metrics['endpoint_recall']['rate']}, "
            f"precision={metrics['endpoint_precision']['rate']}, "
            f"role={metrics['role_accuracy']['rate']}, "
            f"status={metrics['status_accuracy']['rate']}, "
            f"parse={metrics['parseability']['rate']}",
        )
    return 0


def _compare(args: argparse.Namespace) -> int:
    differences = compare_artifacts(args.compare_left, args.compare_right)
    if differences:
        print("[phase2g] artifacts differ:")
        for difference in differences:
            print("  -", difference)
        return 1
    print("[phase2g] artifacts match on deterministic inputs and score/failure distributions")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare-left", type=Path, dest="compare_left")
    parser.add_argument("--compare-right", type=Path, dest="compare_right")
    args = parser.parse_args(argv)
    if args.compare_left or args.compare_right:
        if not (args.compare_left and args.compare_right):
            raise SystemExit("--compare-left and --compare-right must be used together")
        if args.live or args.output:
            raise SystemExit("--compare cannot be combined with --live/--output")
        return _compare(args)
    if args.live:
        return _live(args)
    if args.output is not None:
        raise SystemExit("--output is only valid with --live")
    _validate(args.benchmark, args.silver, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
