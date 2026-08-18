"""Shared, process-wide Phase 2I experiment cache for the test suite.

The real offline experiment is expensive (five CPU Stanza parses plus ten
model fits), so every Phase 2I test module shares one cached result per test
process.  This module is not collected by pytest (its name does not match
``test_*.py``).
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"
ASSETS = ROOT / "data" / "phase2i_assets"
ARCHIVE = (
    ROOT / "data/phase2h_artifacts/phase2h-endpoint-scoring-run1.tar.gz"
)

_EXPERIMENT = None


def experiment_result():
    global _EXPERIMENT
    if _EXPERIMENT is None:
        from pipeline.phase2i_endpoint_scoring import run_experiment_c

        _EXPERIMENT = run_experiment_c(
            BENCHMARK,
            assets_dir=ASSETS,
            baseline_archive=ARCHIVE,
        )
    return _EXPERIMENT
