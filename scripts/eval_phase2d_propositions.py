"""Run the Phase 2E span-first Stage A development source-mode ablation.

This command is dry-run only: it reads insights and bronze transcripts, calls
Stage A when ``--live`` is supplied, and writes an inspectable JSON artifact.
It never creates candidate ledger entries or strategic relations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core.llm import BACKEND, MODEL, chat
from pipeline.phase2d_evaluation import (
    DEFAULT_HELD_OUT_FIXTURE,
    evaluate_source_modes,
    load_development_cases,
)
from pipeline.proposition_extract import SPAN_FIRST_PROMPT_VERSION, extract_span_first_propositions
from pipeline.relation_extract import DEEPSEEK_THINKING_MODE, RELATION_FLASH_MODEL, RELATION_PRO_MODEL
from pipeline.source_windows import SourceWindowResolver


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2E span-first Stage A source modes")
    parser.add_argument("--fixture", type=Path, default=Path("data/relation_extraction_phase2d_dev_v0.json"))
    parser.add_argument("--db", type=Path, required=True, help="Read-only SQLite evidence/bronze database")
    parser.add_argument("--live", action="store_true", help="Call the configured relation model; no persistence")
    parser.add_argument("--variant", choices=("flash", "pro"), default="flash")
    parser.add_argument("--model", help="Explicit model ID; overrides --variant")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mode", choices=("insight", "transcript", "combined"), action="append")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("--live is required; this command has no synthetic model mode")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.model is not None:
        args.model = args.model.strip()
        if not args.model:
            parser.error("--model must not be blank")
    if BACKEND != "deepseek" and not args.model:
        parser.error("--variant requires LLM_PROVIDER=deepseek")
    modes = tuple(args.mode or ("insight", "transcript", "combined"))
    if len(modes) != len(set(modes)):
        parser.error("--mode may be supplied at most once per source mode")
    selected = args.model or (RELATION_FLASH_MODEL if args.variant == "flash" else RELATION_PRO_MODEL)
    cases = load_development_cases(args.fixture, held_out_path=DEFAULT_HELD_OUT_FIXTURE)
    result = evaluate_source_modes(
        cases,
        resolver=SourceWindowResolver(str(args.db)),
        extractor=lambda packet: extract_span_first_propositions(
            packet, chat, model=selected, max_tokens=args.max_tokens,
            thinking=DEEPSEEK_THINKING_MODE if BACKEND == "deepseek" else None,
        ),
        modes=modes,  # type: ignore[arg-type]
    )
    result["model"] = {
        "backend": BACKEND,
        "model": selected or MODEL,
        "variant": "custom" if args.model else args.variant,
        "provider": BACKEND,
        "thinking": DEEPSEEK_THINKING_MODE if BACKEND == "deepseek" else None,
        "max_tokens": args.max_tokens,
        "prompt_version": SPAN_FIRST_PROMPT_VERSION,
    }
    result["config"] = {
        "modes": list(modes),
        "fixture": str(args.fixture),
        "held_out_fixture": str(DEFAULT_HELD_OUT_FIXTURE),
        "db": str(args.db),
        "live": True,
    }
    result["fixture"] = str(args.fixture)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    content_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    artifact = {"content_sha256": content_sha256, **result}
    final = json.dumps(artifact, indent=2, sort_keys=True)
    print(final)
    if args.json_output:
        args.json_output.write_text(final + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
