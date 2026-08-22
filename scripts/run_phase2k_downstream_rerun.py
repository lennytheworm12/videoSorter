#!/usr/bin/env python3
"""Run the gate-locked Phase 2K paired downstream rerun (or preflight only).

Every Phase 2K human/audit/alignment gate must pass before any provider call:
the Phase 2K output must be a deep-validated live build with a PASSED
finalized human review gate and validated completed transformation audit, and
the finalized alignment packet/summary must be REVIEWED with all 30 windows
and all 311 endpoint IDs.  The default (no ``--live``) mode performs every
check and prints the sealed preflight/input contract without calling a
provider or fabricating result rows.  ``--live`` additionally compiles the
Phase 2F semantic IR and runs the four Phase 2H cells on both representations
and publishes the immutable canonical-hash evidence directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2k_contextual_reconstruction import ROOT
from pipeline.semantic_ir_pool import load_semantic_window_pool
from pipeline.phase2k_downstream_rerun import (
    DEFAULT_PRIMARY_CELL,
    build_preflight_contract,
    build_input_adapters,
    load_rerun_inputs,
    run_phase2k_downstream_rerun,
)
from pipeline.semantic_compiler import SemanticCompilerConfig


REFERENCE_MODEL = "deepseek-v4-pro"
REFERENCE_THINKING = "disabled"
REFERENCE_ENDPOINT = "https://api.deepseek.com"

DEFAULT_REVIEWED_PACKET = ROOT / "data/phase2j/reviewed-endpoint-annotation-packet-v1.json"
DEFAULT_COVERAGE = ROOT / "data/phase2j/candidate-coverage-v1.json"
DEFAULT_POOL = ROOT / "data/semantic_ir_window_pool_v1.json"

ABILITY_ALIASES = (
    "Q", "W", "E", "R", "ult", "ultimate",
    "Flash", "Teleport", "Ignite", "Exhaust", "Ward", "Sweeper",
)


def reference_config() -> SemanticCompilerConfig:
    return SemanticCompilerConfig.create(
        REFERENCE_MODEL,
        provider_configuration={
            "provider": "deepseek",
            "endpoint": REFERENCE_ENDPOINT,
            "purpose": "phase2k-downstream-paired-rerun",
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


def load_provider() -> Callable[..., str]:
    """Load the chat provider exactly as safely as the Phase 2F eval CLI."""
    import core.llm as llm

    endpoint = getattr(llm, "_DEEPSEEK_BASE_URL", None)
    if llm.BACKEND != "deepseek" or endpoint != REFERENCE_ENDPOINT:
        raise ValueError(
            "the Phase 2K downstream rerun requires the official DeepSeek "
            "provider endpoint",
        )
    return llm.chat


def load_alias_sets() -> tuple[list[str], tuple[str, ...]]:
    """Load/validate the frozen Phase 2F entity/ability alias sets.

    Entity aliases are the exact sorted champion-name set sealed in
    ``data/semantic_ir_window_pool_v1.json``; ability aliases are the fixed
    summoner/ability alias set.  An absent, empty, or invalid champion set
    fails closed instead of silently compiling with no entity aliases.
    """
    pool = load_semantic_window_pool(DEFAULT_POOL)
    policy = pool.get("selection_policy")
    champion_names = (
        policy.get("champion_names") if isinstance(policy, Mapping) else None
    )
    if (
        not isinstance(champion_names, list)
        or not champion_names
        or any(
            not isinstance(alias, str) or not alias.strip()
            for alias in champion_names
        )
    ):
        raise ValueError(
            "semantic window pool champion_names must be a non-empty list of "
            "non-empty strings",
        )
    for alias in ABILITY_ALIASES:
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("ability aliases must be non-empty trimmed strings")
    return list(champion_names), ABILITY_ALIASES


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate-locked Phase 2K paired downstream rerun (preflight by "
            "default; --live requires the official DeepSeek provider)."
        ),
    )
    parser.add_argument(
        "--phase2k-dir", type=Path, required=True,
        help="Finalized live Phase 2K output directory.",
    )
    parser.add_argument(
        "--alignment-packet", type=Path, required=True,
        help="Finalized Phase 2K downstream alignment packet (REVIEWED).",
    )
    parser.add_argument(
        "--alignment-summary", type=Path, required=True,
        help="Phase 2K downstream alignment summary.",
    )
    parser.add_argument(
        "--reviewed-packet", type=Path, default=DEFAULT_REVIEWED_PACKET,
        help="Phase 2J reviewed endpoint annotation packet (immutable input).",
    )
    parser.add_argument(
        "--coverage", type=Path, default=DEFAULT_COVERAGE,
        help="Phase 2J candidate coverage artifact (immutable input).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "New immutable evidence directory; required with --live and must "
            "not already exist."
        ),
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run the paired generative/discriminative rerun and publish evidence.",
    )
    parser.add_argument(
        "--primary-cell", default=DEFAULT_PRIMARY_CELL,
        choices=sorted({
            "logistic_A", "logistic_B", "lightgbm_A", "lightgbm_B",
        }),
        help="Declared Phase 2H primary cell for comparison-v2 rows.",
    )
    args = parser.parse_args(argv)

    for label, path in (
        ("phase2k output directory", args.phase2k_dir),
        ("alignment packet", args.alignment_packet),
        ("alignment summary", args.alignment_summary),
        ("Phase 2J reviewed packet", args.reviewed_packet),
        ("Phase 2J candidate coverage", args.coverage),
    ):
        if not Path(path).exists():
            parser.error(f"{label} does not exist: {path}")

    try:
        config = reference_config()
        entity_aliases, ability_aliases = load_alias_sets()
        if not args.live:
            inputs = load_rerun_inputs(
                phase2k_dir=args.phase2k_dir,
                alignment_packet_path=args.alignment_packet,
                alignment_summary_path=args.alignment_summary,
                reviewed_packet_path=args.reviewed_packet,
                coverage_path=args.coverage,
            )
            adapters = build_input_adapters(inputs)
            preflight = build_preflight_contract(
                inputs=inputs,
                adapters=adapters,
                config=config,
                primary_cell=args.primary_cell,
                entity_aliases=entity_aliases,
                ability_aliases=ability_aliases,
            )
            print(_canonical_json({
                "status": "VALIDATED_NO_PROVIDER_CALL",
                "preflight_content_sha256": preflight["content_sha256"],
                "schema_version": preflight["schema_version"],
                "target_count": preflight["gates"]["target_count"],
                "window_count": preflight["gates"]["window_count"],
                "primary_cell": args.primary_cell,
            }))
            return 0
        if args.output is None:
            parser.error("--output is required with --live")
        try:
            args.output.resolve().relative_to(ROOT.resolve())
        except ValueError:
            pass
        else:
            parser.error("rerun output must be outside the source repository")
        if args.output.exists():
            parser.error(
                f"output directory already exists: {args.output}",
            )
        chat = load_provider()
        output = run_phase2k_downstream_rerun(
            phase2k_dir=args.phase2k_dir,
            alignment_packet_path=args.alignment_packet,
            alignment_summary_path=args.alignment_summary,
            reviewed_packet_path=args.reviewed_packet,
            coverage_path=args.coverage,
            output=args.output,
            config=config,
            chat=chat,
            primary_cell=args.primary_cell,
            entity_aliases=entity_aliases,
            ability_aliases=ability_aliases,
        )
        evidence = load_rerun_evidence_summary(output)
        print(_canonical_json({
            "status": "COMPLETE",
            "output": str(output),
            **evidence,
        }))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            f"[phase2k] downstream rerun failed: {exc}",
            file=sys.stderr,
        )
        return 1


def load_rerun_evidence_summary(output: Path) -> dict[str, str]:
    from pipeline.phase2k_contextual_reconstruction import load_json_strict
    from pipeline.phase2k_downstream_rerun import ARTIFACT_FILENAMES

    preflight = load_json_strict(
        output / ARTIFACT_FILENAMES["preflight"], label="preflight",
    )
    comparison_input = load_json_strict(
        output / ARTIFACT_FILENAMES["comparison_input"],
        label="comparison input",
    )
    return {
        "preflight_content_sha256": preflight["content_sha256"],
        "comparison_input_content_sha256": comparison_input["content_sha256"],
        "primary_cell": comparison_input["primary_cell"],
        "generative_raw_content_sha256": load_json_strict(
            output / ARTIFACT_FILENAMES["generative_raw"],
            label="generative raw",
        )["content_sha256"],
        "generative_polished_content_sha256": load_json_strict(
            output / ARTIFACT_FILENAMES["generative_polished"],
            label="generative polished",
        )["content_sha256"],
        "discriminative_raw_content_sha256": load_json_strict(
            output / ARTIFACT_FILENAMES["discriminative_raw"],
            label="discriminative raw",
        )["content_sha256"],
        "discriminative_polished_content_sha256": load_json_strict(
            output / ARTIFACT_FILENAMES["discriminative_polished"],
            label="discriminative polished",
        )["content_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
