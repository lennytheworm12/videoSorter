#!/usr/bin/env python3
"""Run the offline Phase 2H discriminative semantic endpoint scoring.

Default mode runs the complete offline experiment on the locked five-case
Phase 2F/2G benchmark: dataset contract (33/33), grouped leave-one-window-out
CV for all four cells (logistic A/B, LightGBM A/B), candidate-level pooled and
per-fold metrics, recall/precision@K, gold ranks, overlap diagnostics, error
taxonomy, and strongest features -- without calling any LLM/API.

``--output DIR`` publishes the immutable artifact to a new directory outside
the repository and only from a clean committed tree (``git status`` must be
empty).  ``--compare-left A --compare-right B`` verifies that a clean rerun
preserves deterministic inputs, scores, and metrics while timestamps may
differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2h_endpoint_scoring import (
    CELLS,
    RUN_VERSION,
    build_aggregate,
    build_dataset,
    build_window_table,
    canonical_sha256,
    compare_artifacts,
    load_benchmark,
    publish_artifact,
    run_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"


def _repository_dirty(repo: Path) -> bool:
    return bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip())


def _window_tables_and_hashes(result) -> tuple[dict, dict]:
    tables = {}
    hashes = {}
    for window_id in sorted(result["dataset"]["windows"]):
        table = build_window_table(
            result["dataset"], result["rankings"], result["errors"], window_id,
            cells=result["cells"],
        )
        tables[window_id] = table
        hashes[window_id] = canonical_sha256(table)
    return tables, hashes


def _offline(
    benchmark_path: Path,
    *,
    cells: tuple[str, ...],
    verbose: bool,
) -> dict:
    benchmark = load_benchmark(benchmark_path)
    dataset = build_dataset(benchmark)
    summary = {
        "window_count": len(dataset["windows"]),
        "candidate_count": sum(
            len(window["rows"]) for window in dataset["windows"].values()
        ),
        "positive_count": sum(
            1
            for window in dataset["windows"].values()
            for row in window["rows"]
            if row.is_gold_positive
        ),
    }
    print(f"[phase2h] run_version: {RUN_VERSION}")
    print(f"[phase2h] benchmark content sha256: {benchmark['content_sha256']}")
    print(
        f"[phase2h] dataset: {summary['window_count']} windows, "
        f"{summary['candidate_count']} candidates, "
        f"{summary['positive_count']}/33 gold endpoints",
    )
    print(
        f"[phase2h] cells: {', '.join(cells)}; grouped leave-one-window-out "
        "CV over identical folds; threshold 0.5; offline",
    )
    result = run_experiment(benchmark, cells=cells, verbose=verbose)
    for cell in cells:
        metrics = result["metrics"][cell]
        print(
            f"[phase2h] {cell}: precision={metrics['precision']['rate']}, "
            f"recall={metrics['recall']['rate']}, "
            f"f1={metrics['f1']['value']}, "
            f"auc={metrics['roc_auc']['value']}, "
            f"ap={metrics['average_precision']['value']}, "
            f"gold_rank_median={metrics['gold_rank']['median']}",
        )
    print(
        f"[phase2h] pooled recall@10: "
        f"{result['metrics'][cells[0]]['recall_at_k']['10']['rate']} "
        "(first cell)",
    )
    return result


def _publish(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit(
            "Phase 2H artifact output must be outside the source repository",
        )
    if output.exists():
        raise SystemExit(
            "output directory already exists; Phase 2H artifacts are immutable",
        )
    if _repository_dirty(ROOT):
        raise SystemExit(
            "Phase 2H artifacts may only be published from a clean committed "
            "tree; commit or stash your changes first",
        )
    benchmark = load_benchmark(args.benchmark)
    result = run_experiment(benchmark, cells=args.cells, verbose=args.verbose)
    tables, hashes = _window_tables_and_hashes(result)
    aggregate = build_aggregate(
        args.benchmark, result, repo=ROOT,
        window_table_hashes=hashes,
    )
    if aggregate["repository_dirty"]:
        raise SystemExit("repository became dirty during artifact build")
    publish_artifact(output, aggregate, tables)
    print(f"[phase2h] published immutable artifact: {output}")
    print(f"[phase2h] artifact content sha256: {aggregate['content_sha256']}")
    for cell in args.cells:
        metrics = aggregate["metrics"][cell]
        print(
            f"[phase2h] {cell}: precision={metrics['precision']['rate']}, "
            f"recall={metrics['recall']['rate']}, "
            f"f1={metrics['f1']['value']}",
        )
    return 0


def _compare(args: argparse.Namespace) -> int:
    differences = compare_artifacts(
        args.compare_left, args.compare_right,
    )
    if differences:
        print("[phase2h] artifacts differ:")
        for difference in differences:
            print("  -", difference)
        return 1
    print("[phase2h] artifacts match on deterministic inputs, scores, and metrics")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--cell", action="append", dest="cells", default=None,
        choices=list(CELLS),
        help="restrict to a model cell (repeatable; default: all four)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare-left", type=Path, dest="compare_left")
    parser.add_argument("--compare-right", type=Path, dest="compare_right")
    args = parser.parse_args(argv)
    args.cells = tuple(args.cells) if args.cells else CELLS
    if len(set(args.cells)) != len(args.cells):
        raise SystemExit("duplicate --cell values are not allowed")
    if args.compare_left or args.compare_right:
        if not (args.compare_left and args.compare_right):
            raise SystemExit(
                "--compare-left and --compare-right must be used together",
            )
        if args.output is not None:
            raise SystemExit("--compare cannot be combined with --output")
        return _compare(args)
    if args.output is not None:
        return _publish(args)
    _offline(args.benchmark, cells=args.cells, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
