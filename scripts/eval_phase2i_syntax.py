#!/usr/bin/env python3
"""Run the Phase 2I UD/syntactic Feature Set C ablation (offline).

Default mode runs the complete offline experiment on the locked five-case
benchmark: local CPU Stanza parse of every Bronze window, Feature Set C
extraction, grouped leave-one-window-out CV for ``logistic_C`` and
``lightgbm_C`` (exact Phase 2H B model configs), metrics, explicit deltas vs
the hash-verified frozen Phase 2H B baseline, derived universally-missed
endpoint tracking, overlap-cluster syntax diagnostics, B-vs-C error
taxonomy, parser/alignment diagnostics, syntax coefficients/importances, and
training-vs-held-out diagnostics.  No network and no LLM at evaluation time.

``--output DIR`` publishes an immutable artifact to a new directory outside
the repository and only from a clean committed tree.  ``--compare-left A
--compare-right B`` verifies a deterministic rerun.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2h_endpoint_scoring import canonical_sha256
from pipeline.phase2i_endpoint_scoring import (
    CELLS_C,
    RUN_VERSION,
    build_aggregate_c,
    build_dataset,
    build_phase2i_window_table,
    close_phase2h_baseline,
    compare_phase2i_artifacts,
    load_phase2h_baseline,
    load_benchmark,
    publish_phase2i_artifact,
    run_experiment_c,
)
from pipeline.phase2i_syntax import (
    _symlink_ancestor_problems,
    verify_assets_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"
DEFAULT_ASSETS = ROOT / "data" / "phase2i_assets"
DEFAULT_ARCHIVE = (
    ROOT / "data/phase2h_artifacts/phase2h-endpoint-scoring-run1.tar.gz"
)


def _repository_dirty(repo: Path) -> bool:
    return bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip())


def _print_metrics(result) -> None:
    print(f"[phase2i] baseline of record: {RUN_VERSION} vs Phase 2H B")
    for cell in result["cells"]:
        metrics = result["metrics"][cell]
        b_cell = cell.replace("_C", "_B")
        baseline = result["baseline_metrics"][b_cell]
        deltas = result["deltas"][cell]["delta"]
        print(
            f"[phase2i] {cell}: precision={metrics['precision']['rate']} "
            f"(B {baseline['precision']['rate']}), "
            f"recall={metrics['recall']['rate']} "
            f"(B {baseline['recall']['rate']}), "
            f"f1={metrics['f1']['value']} (B {baseline['f1']['value']}), "
            f"auc={metrics['roc_auc']['value']} "
            f"(B {baseline['roc_auc']['value']}), "
            f"ap={metrics['average_precision']['value']}, "
            f"gold_rank_median={metrics['gold_rank']['median']} "
            f"(B {baseline['gold_rank']['median']}), "
            f"selected={metrics['selected']} (B {baseline['selected']})",
        )
        print(
            f"[phase2i]   delta: precision={deltas['precision']}, "
            f"recall={deltas['recall']}, f1={deltas['f1']}, "
            f"auc={deltas['roc_auc']}, selected={deltas['selected']}, "
            f"recall@10={deltas['recall_at_k']['10']['delta']}, "
            f"median_gold_rank={deltas['gold_rank']['delta_median']}",
        )
    missed = result["universally_missed"]
    print(
        f"[phase2i] universally missed gold endpoints: {len(missed)} "
        f"(problems: {result['universally_missed_problems']})",
    )


def _offline(args: argparse.Namespace) -> int:
    provenance = verify_assets_provenance(args.assets_dir)
    if not provenance["verified"]:
        raise SystemExit(
            "[phase2i] parser assets provenance verification failed: "
            + str(provenance.get("problems") or provenance.get("reason")),
        )
    benchmark = load_benchmark(args.benchmark)
    print(f"[phase2i] run_version: {RUN_VERSION}")
    print(
        f"[phase2i] benchmark content sha256: "
        f"{benchmark['content_sha256']}",
    )
    print(
        f"[phase2i] cells: {', '.join(args.cells)}; grouped "
        "leave-one-window-out CV; threshold 0.5; offline CPU; local assets",
    )
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
    print(
        f"[phase2i] dataset: {summary['window_count']} windows, "
        f"{summary['candidate_count']} candidates, "
        f"{summary['positive_count']}/33 gold endpoints",
    )
    result = run_experiment_c(
        benchmark,
        cells=args.cells,
        assets_dir=args.assets_dir,
        baseline_archive=args.archive,
        verbose=args.verbose,
    )
    _print_metrics(result)
    return 0


def _publish(args: argparse.Namespace) -> int:
    output_problems = _symlink_ancestor_problems(args.output)
    if output_problems:
        raise SystemExit(
            "Phase 2I artifact output path is unsafe: "
            + "; ".join(output_problems),
        )
    output = Path(os.path.abspath(os.fspath(args.output)))
    if tuple(args.cells) != CELLS_C:
        raise SystemExit(
            "Phase 2I publication requires both C cells "
            "(logistic_C and lightgbm_C); --cell subsets are "
            "non-published diagnostics only",
        )
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit(
            "Phase 2I artifact output must be outside the source repository",
        )
    if output.exists():
        raise SystemExit(
            "output directory already exists; Phase 2I artifacts are immutable",
        )
    if _repository_dirty(ROOT):
        raise SystemExit(
            "Phase 2I artifacts may only be published from a clean committed "
            "tree; commit or stash your changes first",
        )
    start_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    provenance = verify_assets_provenance(args.assets_dir)
    if not provenance["verified"]:
        raise SystemExit(
            "parser assets provenance is required for publication: "
            + str(provenance.get("problems") or provenance.get("reason")),
        )
    benchmark = load_benchmark(args.benchmark)
    result = run_experiment_c(
        benchmark,
        cells=args.cells,
        assets_dir=args.assets_dir,
        baseline_archive=args.archive,
        verbose=args.verbose,
    )
    baseline = load_phase2h_baseline(args.archive)
    tables = {}
    hashes = {}
    parse_tables = {}
    try:
        for window_id in sorted(result["dataset"]["windows"]):
            table = build_phase2i_window_table(
                result["dataset"], result["rankings"], result["errors"],
                baseline["window_tables"][window_id],
                result["parses"][window_id],
                result["candidate_syntax"][window_id],
                window_id,
                cells=args.cells,
            )
            tables[window_id] = table
            hashes[window_id] = canonical_sha256(table)
            parse_tables[window_id] = result["parses"][window_id].to_dict()
    finally:
        close_phase2h_baseline(baseline)
    aggregate = build_aggregate_c(
        args.benchmark,
        result,
        repo=ROOT,
        window_table_hashes=hashes,
        assets_provenance=provenance,
    )
    if aggregate["repository_dirty"]:
        raise SystemExit("repository became dirty during artifact build")
    if aggregate["git_commit"] != start_commit:
        raise SystemExit(
            "repository HEAD changed during the Phase 2I artifact build",
        )
    publish_phase2i_artifact(
        output, aggregate, tables, parse_tables,
        benchmark_path=args.benchmark,
        baseline_archive=args.archive,
        assets_dir=args.assets_dir,
    )
    print(f"[phase2i] published immutable artifact: {output}")
    print(f"[phase2i] artifact content sha256: {aggregate['content_sha256']}")
    _print_metrics(result)
    return 0


def _compare(args: argparse.Namespace) -> int:
    differences = compare_phase2i_artifacts(
        args.compare_left, args.compare_right,
        benchmark_path=args.benchmark,
        baseline_archive=args.archive,
        assets_dir=args.assets_dir,
    )
    if differences:
        print("[phase2i] artifacts differ:")
        for difference in differences:
            print("  -", difference)
        return 1
    print(
        "[phase2i] artifacts match on deterministic inputs, scores, metrics, "
        "deltas, diagnostics, assets provenance, and window/parser file "
        "content hashes (created_at timestamps are excluded by design)",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--cell", action="append", dest="cells", default=None,
        choices=list(CELLS_C),
        help="restrict to a Phase 2I cell (repeatable; default: both)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare-left", type=Path, dest="compare_left")
    parser.add_argument("--compare-right", type=Path, dest="compare_right")
    args = parser.parse_args(argv)
    args.cells = tuple(args.cells) if args.cells else CELLS_C
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
    return _offline(args)


if __name__ == "__main__":
    raise SystemExit(main())
