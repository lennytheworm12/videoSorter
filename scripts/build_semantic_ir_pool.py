#!/usr/bin/env python3
"""Build the deterministic Phase 2F representative bronze-window pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.semantic_ir_pool import (
    build_semantic_window_pool, validate_semantic_window_pool,
    verify_semantic_window_pool_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-fixture", type=Path, default=Path("data/relation_extraction_phase2b_v0.json"))
    parser.add_argument("--development-fixture", type=Path, default=Path("data/relation_extraction_phase2d_dev_v0.json"))
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    pool = build_semantic_window_pool(
        args.db, frozen_fixture=args.frozen_fixture,
        development_fixture=args.development_fixture, target_count=args.count,
    )
    validate_semantic_window_pool(pool)
    verify_semantic_window_pool_inputs(
        pool, db_path=args.db, frozen_fixture=args.frozen_fixture,
        development_fixture=args.development_fixture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(pool, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "windows": len(pool["windows"]),
        "content_sha256": pool["content_sha256"],
        "phenomenon_counts": pool["phenomenon_counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
