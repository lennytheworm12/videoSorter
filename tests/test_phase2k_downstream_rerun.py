"""Focused provider-free tests for the Phase 2K gate-locked paired rerun."""

from __future__ import annotations

import copy
import io
import json
import shutil
import tempfile
import unittest
import collections.abc
import typing
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest import mock

from pipeline.phase2k_contextual_reconstruction import (
    canonical_sha256,
    text_sha256,
)
from pipeline.phase2k_downstream_alignment import (
    build_alignment_summary,
    build_downstream_alignment_packet,
    finalize_downstream_alignment_packet,
)
from pipeline.phase2k_downstream_comparison import (
    DISCRIMINATIVE_ARCHITECTURE_FAMILY,
    GENERATIVE_ARCHITECTURE_FAMILY,
    POLISHED_INPUT_REPRESENTATION,
    RAW_INPUT_REPRESENTATION,
    validate_downstream_comparison,
)
from pipeline.phase2h_endpoint_scoring import CELLS, DROP, KEEP, KEEP_THRESHOLD
from pipeline.semantic_compiler import (
    SemanticCompilerConfig,
    compile_source_semantic_ir,
)
from pipeline.semantic_ir_artifact import build_semantic_run_artifact
from pipeline.semantic_source import BronzeSource, window_from_exact_span
import pipeline.phase2k_downstream_rerun as rerun
from pipeline.phase2k_downstream_rerun import (
    ARTIFACT_FILENAMES,
    DEFAULT_PRIMARY_CELL,
    build_candidate_dataset,
    build_comparison_input,
    build_discriminative_artifacts,
    build_generative_artifacts,
    build_input_adapters,
    build_preflight_contract,
    evaluate_targets,
    finalize_phase2k_downstream_rerun,
    load_rerun_inputs,
    run_phase2k_downstream_rerun,
    validate_rerun_evidence,
)
from tests.test_phase2k_downstream_alignment import (
    _default_decisions,
    _shared_state as _alignment_shared_state,
)
from tests.test_semantic_compiler import ScriptedSemanticModel


def _config() -> SemanticCompilerConfig:
    return SemanticCompilerConfig.create(
        "reference-pro",
        provider_configuration={"provider": "scripted"},
        mention_partition_size=1000,
    )


def _shared_state() -> dict[str, Any]:
    state = _alignment_shared_state()
    if "finalized_packet" not in state:
        blank = build_downstream_alignment_packet(
            phase2k_dir=state["output"],
            reviewed_packet_path=state["packet_path"],
            coverage_path=state["coverage_path"],
        )
        decisions = _default_decisions(blank)
        finalized = finalize_downstream_alignment_packet(blank, decisions)
        summary = build_alignment_summary(finalized)
        state["blank_packet"] = blank
        state["decisions"] = decisions
        state["finalized_packet"] = finalized
        state["alignment_summary"] = summary
        state["alignment_packet_path"] = state["root"] / "alignment-finalized.json"
        state["alignment_summary_path"] = state["root"] / "alignment-summary.json"
        state["alignment_packet_path"].write_text(
            json.dumps(finalized, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        state["alignment_summary_path"].write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return state


def _rerun_args(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase2k_dir": state["output"],
        "alignment_packet_path": state["alignment_packet_path"],
        "alignment_summary_path": state["alignment_summary_path"],
        "reviewed_packet_path": state["packet_path"],
        "coverage_path": state["coverage_path"],
    }


def _fake_run_cv(dataset: Mapping[str, Any], *, cells=CELLS, verbose=False):
    """Deterministic offline CV stand-in: accepted spans score 0.9 else 0.1.

    Returns one structurally complete fold and matching fit-scope record per
    held-out window so the replay validator exercises the same fold/fit-scope
    reproduction path a real Phase 2H run would, without any model training.
    """
    cells = tuple(cells)
    window_ids = sorted(dataset["windows"])
    oof_scores: dict[str, dict[str, dict[str, float]]] = {}
    folds: list[dict[str, Any]] = []
    fit_scope: dict[str, dict[int, dict[str, Any]]] = {
        cell: {} for cell in cells
    }
    fitted_models: dict[str, dict[int, Any]] = {cell: {} for cell in cells}
    for fold_index, test_window_id in enumerate(window_ids):
        train_window_ids = [
            window_id for window_id in window_ids if window_id != test_window_id
        ]
        test_rows = dataset["windows"][test_window_id]["rows"]
        train_rows = [
            row
            for window_id in train_window_ids
            for row in dataset["windows"][window_id]["rows"]
        ]
        train_positive = sum(1 for row in train_rows if row.label == KEEP)
        test_positive = sum(1 for row in test_rows if row.label == KEEP)
        oof_scores[test_window_id] = {}
        for row in test_rows:
            score = 0.9 if row.label == KEEP else 0.1
            oof_scores[test_window_id][row.candidate_id] = {
                cell: score for cell in cells
            }
        folds.append({
            "fold_index": fold_index,
            "train_window_ids": train_window_ids,
            "test_window_id": test_window_id,
            "train_candidate_count": len(train_rows),
            "train_positive_count": train_positive,
            "train_negative_count": len(train_rows) - train_positive,
            "test_candidate_count": len(test_rows),
            "test_positive_count": test_positive,
            "class_weights": {KEEP: 0.5, DROP: 0.5},
        })
        for cell in cells:
            fit_scope[cell][fold_index] = {
                "fold_index": fold_index,
                "train_window_ids": train_window_ids,
                "test_window_id": test_window_id,
                "fit_scope": "training windows only",
                "train_candidate_count": len(train_rows),
                "train_positive_count": train_positive,
                "train_negative_count": len(train_rows) - train_positive,
                "class_weights": {KEEP: 0.5, DROP: 0.5},
                "scaler": {
                    "fit_on": "training_rows_only",
                    "feature_count": 1,
                    "mean": [0.5],
                    "scale": [1.0],
                    "mean_sha256": canonical_sha256([0.5]),
                    "scale_sha256": canonical_sha256([1.0]),
                },
                "model_config": {"family": "logistic", "params": {}},
                "feature_names_sha256": "0" * 64,
            }
            if cell.split("_")[1] == "B":
                fit_scope[cell][fold_index]["vectorizer"] = {
                    "fit_on": "training_rows_only",
                    "vocabulary_terms_origin": "training_rows_only",
                    "params": {},
                    "vocabulary_size": 0,
                    "vocabulary_sha256": canonical_sha256([]),
                }
    return {
        "oof_scores": oof_scores,
        "folds": folds,
        "fit_scope": fit_scope,
        "fitted_models": fitted_models,
    }


def _fake_compute_rankings(
    dataset: Mapping[str, Any],
    oof_scores: Mapping[str, Any],
    *,
    cells=CELLS,
) -> dict[str, Any]:
    """Deterministic rank stand-in mirroring the real Phase 2H rankings."""
    rankings: dict[str, Any] = {}
    for window_id in sorted(dataset["windows"]):
        rows = dataset["windows"][window_id]["rows"]
        window_rankings: dict[str, Any] = {}
        for cell in cells:
            scores = [
                oof_scores[window_id][row.candidate_id][cell]
                for row in rows
            ]
            order = sorted(
                range(len(rows)),
                key=lambda index: (-scores[index], index),
            )
            cell_rankings: dict[str, Any] = {}
            for rank, index in enumerate(order, 1):
                row = rows[index]
                cell_rankings[row.candidate_id] = {
                    "score": scores[index],
                    "rank": rank,
                    "selected": (
                        KEEP if scores[index] >= KEEP_THRESHOLD else DROP
                    ),
                }
            window_rankings[cell] = cell_rankings
        rankings[window_id] = window_rankings
    return rankings


def _reseal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        key: item for key, item in value.items() if key != "content_sha256"
    }
    return {"content_sha256": canonical_sha256(body), **body}


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, Mapping):
        if key in value:
            return True
        return any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _run_full(
    state: Mapping[str, Any],
    *,
    output: Path,
    entity_aliases: tuple[str, ...] = (),
    ability_aliases: tuple[str, ...] = (),
) -> Path:
    return run_phase2k_downstream_rerun(
        **_rerun_args(state),
        output=output,
        config=_config(),
        chat=ScriptedSemanticModel(),
        entity_aliases=entity_aliases,
        ability_aliases=ability_aliases,
        run_cv_fn=_fake_run_cv,
        created_at="2026-08-19T00:00:00Z",
        git_commit="a" * 40,
        repository_dirty=True,
    )


def _write_json(path: Path, body: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")


def _build_evidence_dir(
    state: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Assemble a complete validated evidence directory without a full run."""
    evidence = state["root"] / name
    if evidence.exists():
        shutil.rmtree(evidence)
    evidence.mkdir()
    inputs = load_rerun_inputs(**_rerun_args(state))
    adapters = build_input_adapters(inputs)
    config = _config()
    created_at = "2026-08-19T00:00:00Z"
    git_commit = "a" * 40
    preflight = build_preflight_contract(
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=DEFAULT_PRIMARY_CELL,
    )
    generative_raw, generative_polished, _, _ = build_generative_artifacts(
        inputs=inputs,
        adapters=adapters,
        config=config,
        chat=ScriptedSemanticModel(),
        created_at=created_at,
        git_commit=git_commit,
        repository_dirty=True,
    )
    discriminative_raw, discriminative_polished = build_discriminative_artifacts(
        inputs=inputs,
        adapters=adapters,
        created_at=created_at,
        primary_cell=DEFAULT_PRIMARY_CELL,
        run_cv_fn=_fake_run_cv,
        compute_rankings_fn=_fake_compute_rankings,
    )
    comparison_input = build_comparison_input(
        inputs=inputs,
        adapters=adapters,
        config=config,
        primary_cell=DEFAULT_PRIMARY_CELL,
        generative_raw=generative_raw,
        generative_polished=generative_polished,
        discriminative_raw=discriminative_raw,
        discriminative_polished=discriminative_polished,
    )
    for filename, body in (
        (ARTIFACT_FILENAMES["preflight"], preflight),
        (ARTIFACT_FILENAMES["generative_raw"], generative_raw),
        (ARTIFACT_FILENAMES["generative_polished"], generative_polished),
        (ARTIFACT_FILENAMES["discriminative_raw"], discriminative_raw),
        (ARTIFACT_FILENAMES["discriminative_polished"], discriminative_polished),
        (ARTIFACT_FILENAMES["comparison_input"], comparison_input),
    ):
        _write_json(evidence / filename, body)
    return evidence, inputs, adapters


def _fast_evidence_state() -> dict[str, Any]:
    if not hasattr(_fast_evidence_state, "value"):
        state = _shared_state()
        pristine, inputs, adapters = _build_evidence_dir(
            state, name="rerun-evidence-fast",
        )
        validate_rerun_evidence(
            pristine,
            **_rerun_args(state),
            run_cv_fn=_fake_run_cv,
            compute_rankings_fn=_fake_compute_rankings,
        )
        _fast_evidence_state.value = {
            "state": state,
            "pristine": pristine,
            "inputs": inputs,
            "adapters": adapters,
        }
    return _fast_evidence_state.value


def _tamper_evidence_file(
    evidence: Path,
    *,
    filename: str,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    path = evidence / ARTIFACT_FILENAMES[filename]
    body = json.loads(path.read_text(encoding="utf-8"))
    mutate(body)
    path.write_text(
        json.dumps(_reseal(body), sort_keys=True) + "\n", encoding="utf-8",
    )
    return path


def _tamper_copy(
    state: Mapping[str, Any],
    pristine: Path,
    *,
    name: str,
) -> Path:
    work = state["root"] / name
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(pristine, work)
    return work


class GateAndPreflightTests(unittest.TestCase):
    def test_gates_block_provider_calls_before_reviewed_alignment(self):
        state = _shared_state()
        blank_path = state["root"] / "alignment-blank.json"
        blank_path.write_text(
            json.dumps(state["blank_packet"], sort_keys=True) + "\n",
            encoding="utf-8",
        )

        def exploding_chat(**kwargs):
            raise AssertionError("provider must not be called before gates pass")

        with self.assertRaisesRegex(ValueError, "release gate"):
            args = dict(_rerun_args(state))
            args["alignment_packet_path"] = blank_path
            run_phase2k_downstream_rerun(
                **args,
                output=state["root"] / "never",
                config=_config(),
                chat=exploding_chat,
                run_cv_fn=_fake_run_cv,
                created_at="2026-08-19T00:00:00Z",
                git_commit="a" * 40,
                repository_dirty=True,
            )
        self.assertFalse((state["root"] / "never").exists())

    def test_stale_alignment_summary_is_rejected(self):
        state = _shared_state()
        stale = copy.deepcopy(state["alignment_summary"])
        stale["total"] = 310
        stale_path = state["root"] / "alignment-summary-stale.json"
        stale_path.write_text(
            json.dumps(stale, sort_keys=True) + "\n", encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "summary"):
            args = dict(_rerun_args(state))
            args["alignment_summary_path"] = stale_path
            load_rerun_inputs(**args)

    def test_no_provider_preflight_cli(self):
        import scripts.run_phase2k_downstream_rerun as cli

        state = _shared_state()
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = cli.main([
                "--phase2k-dir", str(state["output"]),
                "--alignment-packet", str(state["alignment_packet_path"]),
                "--alignment-summary", str(state["alignment_summary_path"]),
                "--reviewed-packet", str(state["packet_path"]),
                "--coverage", str(state["coverage_path"]),
            ])
        self.assertEqual(code, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["status"], "VALIDATED_NO_PROVIDER_CALL")
        self.assertEqual(report["target_count"], 311)
        self.assertEqual(report["window_count"], 30)

    def test_preflight_has_no_predictions_or_result_rows(self):
        state = _shared_state()
        inputs = load_rerun_inputs(**_rerun_args(state))
        adapters = build_input_adapters(inputs)
        preflight = build_preflight_contract(
            inputs=inputs,
            adapters=adapters,
            config=_config(),
            primary_cell=DEFAULT_PRIMARY_CELL,
        )
        self.assertFalse(preflight["predictions"])
        self.assertFalse(_contains_key(preflight, "rows"))
        self.assertFalse(_contains_key(preflight, "scores"))
        self.assertFalse(_contains_key(preflight, "output_count"))
        self.assertEqual(
            preflight["gates"],
            {
                "human_review_gate_status": "PASSED",
                "alignment_release_gate": "REVIEWED",
                "alignment_packet_sha256": state["finalized_packet"][
                    "content_sha256"
                ],
                "alignment_summary_sha256": canonical_sha256(
                    state["alignment_summary"],
                ),
                "target_count": 311,
                "window_count": 30,
            },
        )


class AdapterAndTargetTests(unittest.TestCase):
    def test_exact_311_30_binding(self):
        state = _shared_state()
        inputs = load_rerun_inputs(**_rerun_args(state))
        raw_targets = rerun._representation_targets(
            inputs, RAW_INPUT_REPRESENTATION,
        )
        polished_targets = rerun._representation_targets(
            inputs, POLISHED_INPUT_REPRESENTATION,
        )
        self.assertEqual(len(raw_targets), 30)
        self.assertEqual(len(polished_targets), 30)
        self.assertEqual(
            sum(len(items) for items in raw_targets.values()), 311,
        )
        self.assertEqual(
            sum(len(items) for items in polished_targets.values()), 311,
        )
        for window_id in raw_targets:
            self.assertEqual(
                len(raw_targets[window_id]),
                len(polished_targets[window_id]),
            )

    def test_raw_polished_adapters_are_exact_and_distinct(self):
        state = _shared_state()
        inputs = load_rerun_inputs(**_rerun_args(state))
        adapters = build_input_adapters(inputs)
        self.assertEqual(len(adapters["raw"]["windows"]), 30)
        self.assertEqual(len(adapters["polished"]["windows"]), 30)
        self.assertNotEqual(
            adapters["raw"]["adapter_sha256"],
            adapters["polished"]["adapter_sha256"],
        )
        texts = rerun._window_texts(inputs)
        for window_id, descriptors in (
            (wid, adapters["raw"]["windows"]) for wid in adapters["raw"]["windows"]
        ):
            descriptor = descriptors[window_id]
            self.assertEqual(
                descriptor["text_sha256"],
                text_sha256(texts[window_id]["raw"]),
            )
            self.assertEqual(descriptor["representation"], "RAW_BRONZE")
        for window_id, descriptor in adapters["polished"]["windows"].items():
            self.assertEqual(
                descriptor["text_sha256"],
                text_sha256(texts[window_id]["polished"]),
            )
            self.assertEqual(
                descriptor["representation"], "CONTEXTUAL_POLISH",
            )

    def test_absent_ambiguous_stay_in_denominator(self):
        targets = [
            {
                "alignment_id": "p2k:align:one",
                "endpoint_id": "one",
                "node_type": "ACTION",
                "accepted_spans": [],
            },
            {
                "alignment_id": "p2k:align:two",
                "endpoint_id": "two",
                "node_type": None,
                "accepted_spans": [],
            },
        ]
        evaluated = evaluate_targets(
            window_id="w", targets=targets, outputs=[], window_text="text",
        )
        row = evaluated["row"]
        self.assertEqual(row["target_count"], 2)
        self.assertEqual(row["true_positive_count"], 0)
        self.assertEqual(row["false_negative_count"], 2)
        self.assertEqual(row["output_count"], 0)
        self.assertTrue(row["abstained"])
        self.assertTrue(all(not item["tp"] for item in evaluated["per_target"]))

    def test_multiple_one_tp_and_node_type_matching(self):
        source = BronzeSource("transcript:small", "Lux walks. She waits.")
        window = window_from_exact_span(source, 0, len(source.text))
        run = compile_source_semantic_ir(
            window, ScriptedSemanticModel(), config=_config(),
        )
        outputs = [
            {
                "output_id": node.node_id,
                "span": (node.source_span.local_start, node.source_span.local_end),
                "text": node.source_span.text,
                "node_type": node.node_type.value,
            }
            for node in run.mention_nodes
        ]
        self.assertEqual(len(outputs), 4)
        targets = [
            {
                "alignment_id": "p2k:align:multi",
                "endpoint_id": "multi",
                "node_type": None,
                "accepted_spans": [(0, 3), (4, 9)],
            },
            {
                "alignment_id": "p2k:align:type",
                "endpoint_id": "type",
                "node_type": "ACTION",
                "accepted_spans": [(10, 14)],
            },
        ]
        evaluated = evaluate_targets(
            window_id=window.window_id,
            targets=targets,
            outputs=outputs,
            window_text=window.text,
        )
        row = evaluated["row"]
        # First target: one TP (Lux) and one FP (walks, the extra alternative
        # match); the wrong-type She node is not a match; waits is FP.
        self.assertEqual(row["true_positive_count"], 1)
        self.assertEqual(row["false_negative_count"], 1)
        self.assertEqual(row["output_count"], 4)
        self.assertEqual(row["false_positive_count"], 3)
        self.assertTrue(evaluated["per_target"][0]["tp"])
        self.assertFalse(evaluated["per_target"][1]["tp"])


class ArtifactTests(unittest.TestCase):
    def test_generative_artifacts_preserve_full_run_evidence(self):
        state = _shared_state()
        inputs = load_rerun_inputs(**_rerun_args(state))
        adapters = build_input_adapters(inputs)
        raw, polished, raw_evidence, polished_evidence = build_generative_artifacts(
            inputs=inputs,
            adapters=adapters,
            config=_config(),
            chat=ScriptedSemanticModel(),
            created_at="2026-08-19T00:00:00Z",
            git_commit="a" * 40,
            repository_dirty=True,
        )
        for artifact, representation in (
            (raw, "RAW_BRONZE"),
            (polished, "CONTEXTUAL_POLISH"),
        ):
            self.assertEqual(artifact["schema_version"],
                             rerun.GENERATIVE_ARTIFACT_SCHEMA_VERSION)
            self.assertEqual(artifact["input_representation"], representation)
            self.assertEqual(artifact["architecture_family"],
                             GENERATIVE_ARCHITECTURE_FAMILY)
            self.assertEqual(len(artifact["windows"]), 30)
            first_window = artifact["windows"][0]
            payload = artifact["run_artifacts"][first_window["phase2k_window_id"]]
            run_payload = payload["run"]
            self.assertIn("mention_selection", run_payload)
            partition = run_payload["mention_selection"]["partition_results"][0]
            self.assertIn("raw_output", partition)
            rerun.validate_row(
                first_window["row"],
                expected_window_id=first_window["phase2k_window_id"],
            )

    def test_discriminative_all_cells_and_declared_primary(self):
        state = _shared_state()
        inputs = load_rerun_inputs(**_rerun_args(state))
        adapters = build_input_adapters(inputs)
        raw, polished = build_discriminative_artifacts(
            inputs=inputs,
            adapters=adapters,
            created_at="2026-08-19T00:00:00Z",
            primary_cell=DEFAULT_PRIMARY_CELL,
            run_cv_fn=_fake_run_cv,
        )
        raw_dataset = build_candidate_dataset(
            inputs=inputs, adapters=adapters,
            representation=RAW_INPUT_REPRESENTATION,
        )
        polished_dataset = build_candidate_dataset(
            inputs=inputs, adapters=adapters,
            representation=POLISHED_INPUT_REPRESENTATION,
        )
        for artifact in (raw, polished):
            self.assertEqual(artifact["cells"], list(CELLS))
            self.assertEqual(artifact["primary_cell"], DEFAULT_PRIMARY_CELL)
            self.assertEqual(len(artifact["windows"]), 30)
            dataset = (
                raw_dataset
                if artifact["input_representation"] == "RAW_BRONZE"
                else polished_dataset
            )
            for window in artifact["windows"]:
                self.assertEqual(
                    set(window["cell_scores"]), set(CELLS),
                )
                candidate_ids = {
                    row.candidate_id
                    for row in dataset["windows"][window["adapter_window_id"]]["rows"]
                }
                for cell in CELLS:
                    self.assertEqual(
                        set(window["cell_scores"][cell]),
                        candidate_ids,
                    )

    def test_row_arithmetic_invariants(self):
        state = _shared_state()
        inputs = load_rerun_inputs(**_rerun_args(state))
        adapters = build_input_adapters(inputs)
        raw, polished, _, _ = build_generative_artifacts(
            inputs=inputs,
            adapters=adapters,
            config=_config(),
            chat=ScriptedSemanticModel(),
            created_at="2026-08-19T00:00:00Z",
            git_commit="a" * 40,
            repository_dirty=True,
        )
        for artifact in (raw, polished):
            for window in artifact["windows"]:
                row = window["row"]
                self.assertEqual(
                    row["true_positive_count"]
                    + row["false_negative_count"],
                    row["target_count"],
                )
                self.assertEqual(
                    row["true_positive_count"]
                    + row["false_positive_count"],
                    row["output_count"],
                )
                self.assertLessEqual(
                    row["provenance_valid_count"], row["output_count"],
                )
                self.assertEqual(
                    row["abstained"], row["output_count"] == 0,
                )


class FullRunTests(unittest.TestCase):
    def test_full_run_publishes_validated_evidence_with_bound_hashes(self):
        state = _shared_state()
        output = state["root"] / "rerun-evidence"
        _run_full(state, output=output)
        self.assertTrue(output.is_dir())
        self.assertEqual(
            set(ARTIFACT_FILENAMES.values()),
            {path.name for path in output.iterdir()},
        )
        validated = validate_rerun_evidence(
            output,
            **_rerun_args(state),
            run_cv_fn=_fake_run_cv,
            compute_rankings_fn=_fake_compute_rankings,
        )
        comparison_input = validated["comparison_input"]
        for arch_name, artifact_key in (
            ("generative", "generative_raw"),
            ("generative", "generative_polished"),
            ("discriminative", "discriminative_raw"),
            ("discriminative", "discriminative_polished"),
        ):
            arch = comparison_input["architectures"][arch_name]
            cell = arch["raw" if artifact_key.endswith("raw") else "polished"]
            artifact_file = validated[artifact_key]
            self.assertEqual(
                cell["output_artifact_sha256"],
                artifact_file["content_sha256"],
            )
        # Same per-window target counts across every raw/polished cell.
        raw_counts = [
            row["target_count"]
            for row in comparison_input["architectures"]["generative"]["raw"]["rows"]
        ]
        self.assertEqual(sum(raw_counts), 311)
        for arch in comparison_input["architectures"].values():
            for cell in (arch["raw"], arch["polished"]):
                self.assertEqual(
                    [row["target_count"] for row in cell["rows"]],
                    raw_counts,
                )

    def test_tamper_swap_primary_and_stale_gates_are_rejected(self):
        state = _shared_state()
        pristine = state["root"] / "rerun-evidence-tamper"
        _run_full(state, output=pristine)

        work = state["root"] / "rerun-evidence-tamper-work1"
        shutil.copytree(pristine, work)
        tampered = work / ARTIFACT_FILENAMES["generative_raw"]
        body = json.loads(tampered.read_text(encoding="utf-8"))
        body["windows"][0]["row"]["true_positive_count"] = 1
        tampered.write_text(
            json.dumps(_reseal(body), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "recompute"):
            validate_rerun_evidence(
                work,
                **_rerun_args(state),
                run_cv_fn=_fake_run_cv,
                compute_rankings_fn=_fake_compute_rankings,
            )

        swap_dir = state["root"] / "rerun-evidence-tamper-work2"
        shutil.copytree(pristine, swap_dir)
        raw_file = swap_dir / ARTIFACT_FILENAMES["generative_raw"]
        polished_file = swap_dir / ARTIFACT_FILENAMES["generative_polished"]
        raw_bytes = raw_file.read_bytes()
        polished_bytes = polished_file.read_bytes()
        raw_file.write_bytes(polished_bytes)
        polished_file.write_bytes(raw_bytes)
        with self.assertRaisesRegex(ValueError, "swapped"):
            validate_rerun_evidence(
                swap_dir,
                **_rerun_args(state),
                run_cv_fn=_fake_run_cv,
                compute_rankings_fn=_fake_compute_rankings,
            )

        primary_dir = state["root"] / "rerun-evidence-tamper-work3"
        shutil.copytree(pristine, primary_dir)
        preflight_path = primary_dir / ARTIFACT_FILENAMES["preflight"]
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["discriminative"]["primary_cell"] = "lightgbm_A"
        preflight_path.write_text(
            json.dumps(_reseal(preflight), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "primary cell"):
            validate_rerun_evidence(
                primary_dir,
                **_rerun_args(state),
                run_cv_fn=_fake_run_cv,
                compute_rankings_fn=_fake_compute_rankings,
            )

        stale = copy.deepcopy(state["alignment_summary"])
        stale["total"] = 310
        stale_path = state["root"] / "alignment-summary-stale2.json"
        stale_path.write_text(
            json.dumps(stale, sort_keys=True) + "\n", encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "summary"):
            args = dict(_rerun_args(state))
            args["alignment_summary_path"] = stale_path
            validate_rerun_evidence(
                pristine,
                **args,
                run_cv_fn=_fake_run_cv,
                compute_rankings_fn=_fake_compute_rankings,
            )

    def test_atomic_failure_publishes_nothing(self):
        state = _shared_state()
        output = state["root"] / "rerun-evidence-atomic"
        calls = {"count": 0}

        def failing_compiler(window, chat, *, config, entity_aliases=(), ability_aliases=()):
            calls["count"] += 1
            if calls["count"] >= 2:
                raise RuntimeError("synthetic provider failure")
            return compile_source_semantic_ir(
                window, chat, config=config,
                entity_aliases=entity_aliases,
                ability_aliases=ability_aliases,
            )

        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            run_phase2k_downstream_rerun(
                **_rerun_args(state),
                output=output,
                config=_config(),
                chat=ScriptedSemanticModel(),
                compiler=failing_compiler,
                run_cv_fn=_fake_run_cv,
                created_at="2026-08-19T00:00:00Z",
                git_commit="a" * 40,
                repository_dirty=True,
            )
        self.assertFalse(output.exists())
        leftovers = [
            path for path in state["root"].iterdir()
            if path.name.startswith(output.name + ".tmp-")
        ]
        self.assertEqual(leftovers, [])


class FinalizerTests(unittest.TestCase):
    def test_finalizer_requires_explicit_human_fields(self):
        state = _shared_state()
        output = state["root"] / "rerun-evidence-finalize"
        _run_full(state, output=output)
        comparison = finalize_phase2k_downstream_rerun(
            evidence_dir=output,
            **_rerun_args(state),
            decision="CONTEXTUAL_POLISH_VALIDATED",
            diagnosis="MIXED",
            note="human-verified closeout for the synthetic Phase 2K fixture",
            run_cv_fn=_fake_run_cv,
            compute_rankings_fn=_fake_compute_rankings,
        )
        self.assertEqual(comparison["decision"], "CONTEXTUAL_POLISH_VALIDATED")
        self.assertEqual(comparison["diagnosis"], "MIXED")
        self.assertIn("human-verified closeout", comparison["note"])
        inputs = load_rerun_inputs(**_rerun_args(state))
        validate_downstream_comparison(
            comparison,
            label="comparison",
            records_obj=inputs["records_obj"],
            finalized_packet=inputs["finalized_packet"],
            human_summary=inputs["human_summary"],
            completed_audit=inputs["completed_audit"],
        )

        for decision in ("NOT_A_STATUS", "WAITING_FOR_HUMAN_REVIEW"):
            with self.assertRaises(ValueError):
                finalize_phase2k_downstream_rerun(
                    evidence_dir=output,
                    **_rerun_args(state),
                    decision=decision,
                    diagnosis="MIXED",
                    note="note",
                    run_cv_fn=_fake_run_cv,
                    compute_rankings_fn=_fake_compute_rankings,
                )
        with self.assertRaises(ValueError):
            finalize_phase2k_downstream_rerun(
                evidence_dir=output,
                **_rerun_args(state),
                decision="CONTEXTUAL_POLISH_VALIDATED",
                diagnosis="NOT_A_DIAGNOSIS",
                note="note",
                run_cv_fn=_fake_run_cv,
                compute_rankings_fn=_fake_compute_rankings,
            )
        with self.assertRaises(ValueError):
            finalize_phase2k_downstream_rerun(
                evidence_dir=output,
                **_rerun_args(state),
                decision="CONTEXTUAL_POLISH_VALIDATED",
                diagnosis="MIXED",
                note="   ",
                run_cv_fn=_fake_run_cv,
                compute_rankings_fn=_fake_compute_rankings,
            )

    def test_finalizer_cli_writes_v2_file(self):
        import scripts.finalize_phase2k_downstream_rerun as cli

        state = _shared_state()
        output = state["root"] / "rerun-evidence-finalize-cli"
        _run_full(state, output=output)
        comparison_path = state["root"] / "downstream-comparison-v2.json"
        with mock.patch.object(
            rerun, "phase2h_run_cv", new=_fake_run_cv,
        ), mock.patch.object(
            rerun, "phase2h_compute_rankings", new=_fake_compute_rankings,
        ):
            code = cli.main([
                "--evidence-dir", str(output),
                "--phase2k-dir", str(state["output"]),
                "--alignment-packet", str(state["alignment_packet_path"]),
                "--alignment-summary", str(state["alignment_summary_path"]),
                "--reviewed-packet", str(state["packet_path"]),
                "--coverage", str(state["coverage_path"]),
                "--decision", "CONTEXTUAL_POLISH_VALIDATED",
                "--diagnosis", "MIXED",
                "--note", "explicit human closeout from the finalizer CLI test",
                "--output", str(comparison_path),
            ])
        self.assertEqual(code, 0)
        self.assertTrue(comparison_path.is_file())
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        self.assertEqual(
            comparison["schema_version"], "phase2k-downstream-comparison-v2",
        )
        # The mkstemp transaction leaves no partial/temporary file behind and
        # the output is complete and parseable.
        self.assertEqual(
            [
                path for path in comparison_path.parent.iterdir()
                if path.name.startswith(comparison_path.name + ".tmp-")
            ],
            [],
        )


class DIntegrityTests(unittest.TestCase):
    """Phase 2K D-record integrity: clean/reconstruction hash/text binding and
    exact A/D source-window-target identity."""

    def _synthetic_inputs(
        self,
        *,
        clean_text: str | None = None,
    ) -> tuple[dict[str, Any], str, str]:
        bronze = "Lux walks. She waits."
        polished = "Lux walks. She waits. (contextual polish)"
        if clean_text is None:
            clean_text = "Lux walks. She waits. (reconstructed)"
        window_id = "p2k:win:w-dtest"
        source_group_id = "video:test"
        target = {
            "window_id": window_id,
            "source_group_id": source_group_id,
            "source_absolute_start": 0,
            "source_absolute_end": len(bronze),
            "text": bronze,
            "text_sha256": text_sha256(bronze),
            "char_length": len(bronze),
        }
        a_record = {
            "record_type": "A",
            "window_id": window_id,
            "target": copy.deepcopy(target),
            "content": {
                "kind": "raw_bronze",
                "text": bronze,
                "text_sha256": text_sha256(bronze),
                "char_length": len(bronze),
            },
        }
        d_content = {
            "generation_status": "GENERATED",
            "clean_target_transcript": clean_text,
            "clean_target_transcript_sha256": text_sha256(clean_text),
            "reconstruction": {
                "generation_status": "GENERATED",
                "clean_target_transcript": clean_text,
                "clean_target_transcript_sha256": text_sha256(clean_text),
            },
            "semantic_polish": {"polished_text": polished},
        }
        d_record = {
            "record_type": "D",
            "window_id": window_id,
            "target": copy.deepcopy(target),
            "content": d_content,
        }
        inputs = {
            "records_obj": {"records": [a_record, d_record]},
            "reviewed_packet": {"records": [{
                "window_id": window_id,
                "source_group_id": source_group_id,
                "upstream_start": 0,
                "upstream_end": len(bronze),
            }]},
        }
        return inputs, bronze, polished

    def test_valid_d_clean_transcript_differing_from_bronze_is_accepted(self):
        inputs, bronze, polished = self._synthetic_inputs()
        texts = rerun._window_texts(inputs)
        d_record = inputs["records_obj"]["records"][1]
        self.assertEqual(texts["p2k:win:w-dtest"]["raw"], bronze)
        self.assertEqual(
            texts["p2k:win:w-dtest"]["polished"], polished,
        )
        self.assertNotEqual(
            d_record["content"]["clean_target_transcript"], bronze,
        )

    def test_d_clean_and_reconstruction_hash_text_tampering_is_rejected(self):
        cases = [
            (
                "clean text diverges from sealed reconstruction",
                lambda content: content.__setitem__(
                    "clean_target_transcript", "tampered text",
                ),
                "does not match its sealed reconstruction",
            ),
            (
                "clean transcript hash invalid",
                lambda content: content.__setitem__(
                    "clean_target_transcript_sha256", "0" * 64,
                ),
                "phase2k D clean transcript hash is invalid",
            ),
            (
                "reconstruction text diverges from clean transcript",
                lambda content: content["reconstruction"].__setitem__(
                    "clean_target_transcript", "tampered text",
                ),
                "does not match its sealed reconstruction",
            ),
            (
                "reconstruction hash invalid",
                lambda content: content["reconstruction"].__setitem__(
                    "clean_target_transcript_sha256", "0" * 64,
                ),
                "phase2k D reconstruction hash is invalid",
            ),
        ]
        for label, mutate, message in cases:
            with self.subTest(mutation=label):
                inputs, _, _ = self._synthetic_inputs()
                content = inputs["records_obj"]["records"][1]["content"]
                mutate(content)
                with self.assertRaisesRegex(ValueError, message):
                    rerun._window_texts(inputs)

    def test_exact_a_d_source_window_target_tampering_is_rejected(self):
        def records_of(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            records = inputs["records_obj"]["records"]
            return records[0], records[1]

        cases = [
            (
                "A record window identity",
                lambda inputs: records_of(inputs)[0].__setitem__(
                    "window_id", "p2k:win:other",
                ),
                "phase2k A record is missing",
            ),
            (
                "D record window identity",
                lambda inputs: records_of(inputs)[1].__setitem__(
                    "window_id", "p2k:win:other",
                ),
                "phase2k A record is missing",
            ),
            (
                "A target diverges from D target",
                lambda inputs: records_of(inputs)[0].__setitem__(
                    "target", {"window_id": "p2k:win:other"},
                ),
                "phase2k D target does not exactly match the A target",
            ),
            (
                "A/D target window binding",
                lambda inputs: [record["target"].__setitem__(
                    "window_id", "p2k:win:other",
                ) for record in records_of(inputs)],
                "phase2k A/D target window binding is invalid",
            ),
            (
                "A/D target source binding",
                lambda inputs: [record["target"].__setitem__(
                    "source_group_id", "video:other",
                ) for record in records_of(inputs)],
                "phase2k A/D target source binding does not match",
            ),
            (
                "A content text diverges from target",
                lambda inputs: records_of(inputs)[0]["content"].__setitem__(
                    "text", "tampered bronze",
                ),
                "phase2k A content text does not match the D target",
            ),
            (
                "A text hash invalid",
                lambda inputs: records_of(inputs)[0]["content"].__setitem__(
                    "text_sha256", "0" * 64,
                ),
                "phase2k A text hash is invalid",
            ),
            (
                "D target text hash invalid",
                lambda inputs: [
                    record["target"].__setitem__("text_sha256", "0" * 64)
                    for record in records_of(inputs)
                ],
                "phase2k D target text hash is invalid",
            ),
        ]
        for label, mutate, message in cases:
            with self.subTest(mutation=label):
                inputs, _, _ = self._synthetic_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(ValueError, message):
                    rerun._window_texts(inputs)


class FamilyMatchingTests(unittest.TestCase):
    def test_generative_typed_target_rejects_null_node_type_but_span_only_accepts(self):
        source = BronzeSource("transcript:small", "Lux walks. She waits.")
        window = window_from_exact_span(source, 0, len(source.text))
        run = compile_source_semantic_ir(
            window, ScriptedSemanticModel(), config=_config(),
        )
        outputs = [
            {
                "output_id": node.node_id,
                "span": (node.source_span.local_start, node.source_span.local_end),
                "text": node.source_span.text,
                "node_type": node.node_type.value,
            }
            for node in run.mention_nodes
        ]
        null_type_outputs = [dict(item, node_type=None) for item in outputs]
        node = outputs[0]
        typed_target = {
            "alignment_id": "p2k:align:family",
            "endpoint_id": "family",
            "node_type": next(
                item["node_type"]
                for item in outputs
                if item["node_type"] != node["node_type"]
            ),
            "accepted_spans": [node["span"]],
        }
        generative = evaluate_targets(
            window_id=window.window_id,
            targets=[typed_target],
            outputs=null_type_outputs,
            window_text=window.text,
            require_exact_node_type=True,
        )
        discriminative = evaluate_targets(
            window_id=window.window_id,
            targets=[typed_target],
            outputs=null_type_outputs,
            window_text=window.text,
            require_exact_node_type=False,
        )
        self.assertEqual(generative["row"]["true_positive_count"], 0)
        self.assertEqual(generative["row"]["false_negative_count"], 1)
        self.assertEqual(discriminative["row"]["true_positive_count"], 1)
        self.assertEqual(discriminative["row"]["false_negative_count"], 0)


class ProviderFailureTests(unittest.TestCase):
    def test_real_compiler_exploding_chat_typed_failure_and_atomic_abort(self):
        state = _shared_state()
        source = BronzeSource("transcript:small", "Lux walks. She waits.")
        window = window_from_exact_span(source, 0, len(source.text))

        def exploding_chat(**kwargs):
            raise RuntimeError("synthetic provider outage")

        run = compile_source_semantic_ir(
            window, exploding_chat, config=_config(),
        )
        self.assertTrue(run.failures)
        self.assertTrue(any(
            failure.code == "PROVIDER_FAILURE" for failure in run.failures
        ))

        output = state["root"] / "rerun-evidence-provider"
        with self.assertRaisesRegex(ValueError, "provider failure"):
            run_phase2k_downstream_rerun(
                **_rerun_args(state),
                output=output,
                config=_config(),
                chat=exploding_chat,
                run_cv_fn=_fake_run_cv,
                created_at="2026-08-19T00:00:00Z",
                git_commit="a" * 40,
                repository_dirty=True,
            )
        self.assertFalse(output.exists())
        self.assertEqual(
            [
                path for path in state["root"].iterdir()
                if path.name.startswith(output.name + ".tmp-")
            ],
            [],
        )


class GenerativeValidatorTamperTests(unittest.TestCase):
    def _validate_tampered(
        self,
        mutate: Callable[[dict[str, Any]], None],
        message: str,
        *,
        filename: str = "generative_raw",
    ) -> None:
        fast = _fast_evidence_state()
        state = fast["state"]
        work = _tamper_copy(
            state, fast["pristine"], name="rerun-evidence-gen-tamper",
        )
        _tamper_evidence_file(work, filename=filename, mutate=mutate)
        with self.assertRaisesRegex(ValueError, message):
            validate_rerun_evidence(
                work,
                **_rerun_args(state),
                run_cv_fn=_fake_run_cv,
                compute_rankings_fn=_fake_compute_rankings,
            )

    def test_resealed_alias_and_execution_hash_mismatch_rejected(self):
        cases = [
            (
                "entity aliases",
                lambda body: body.__setitem__("entity_aliases", ["Lux"]),
                "phase2k generative artifact entity aliases were changed",
            ),
            (
                "ability aliases",
                lambda body: body.__setitem__("ability_aliases", ["Q"]),
                "phase2k generative artifact ability aliases were changed",
            ),
            (
                "execution hash",
                lambda body: body.__setitem__(
                    "compiler_config_sha256", "0" * 64,
                ),
                "execution identity does not match the preflight contract",
            ),
        ]
        for label, mutate, message in cases:
            with self.subTest(mutation=label):
                self._validate_tampered(mutate, message)

    def test_run_input_hash_tampering_rejected(self):
        for key in (
            "phase2k_records_sha256",
            "phase2k_alignment_packet_sha256",
            "phase2k_input_adapter_sha256",
        ):
            with self.subTest(input_hash=key):
                def mutate(body: dict[str, Any], key: str = key) -> None:
                    first_window = next(iter(body["run_artifacts"]))
                    payload = body["run_artifacts"][first_window]
                    payload["input_hashes"][key] = "0" * 64
                    body["run_artifacts"][first_window] = _reseal(payload)
                self._validate_tampered(
                    mutate, "input hashes do not match its sources",
                )

    def test_window_source_and_lineage_tampering_rejected(self):
        cases = [
            (
                "window identity",
                lambda body: _tamper_window_identity(body),
                "window ID is not bound to its exact source span and version",
            ),
            (
                "created_at lineage",
                lambda body: _tamper_payload_lineage(
                    body, "created_at", "2026-08-19T01:00:00Z",
                ),
                "lineage does not match its envelope",
            ),
            (
                "git_commit lineage",
                lambda body: _tamper_payload_lineage(
                    body, "git_commit", "b" * 40,
                ),
                "lineage does not match its envelope",
            ),
            (
                "repository_dirty lineage",
                lambda body: _tamper_payload_lineage(
                    body, "repository_dirty", False,
                ),
                "lineage does not match its envelope",
            ),
        ]
        for label, mutate, message in cases:
            with self.subTest(mutation=label):
                self._validate_tampered(mutate, message)

    def test_resealed_run_artifact_with_provider_failure_rejected(self):
        fast = _fast_evidence_state()
        state = fast["state"]
        inputs = fast["inputs"]
        adapters = fast["adapters"]
        first_window = sorted(adapters["by_window"])[0]
        adapter = adapters["by_window"][first_window]["raw"]

        def exploding_chat(**kwargs):
            raise RuntimeError("synthetic provider outage")

        failed = compile_source_semantic_ir(
            adapter["window"], exploding_chat, config=_config(),
        )
        failed_payload = build_semantic_run_artifact(
            failed,
            git_commit="a" * 40,
            repository_dirty=True,
            created_at="2026-08-19T00:00:00Z",
            input_hashes={
                "phase2k_records_sha256": inputs["records_obj"][
                    "content_sha256"
                ],
                "phase2k_alignment_packet_sha256": inputs["alignment_packet"][
                    "content_sha256"
                ],
                "phase2k_input_adapter_sha256": adapters["raw"][
                    "adapter_sha256"
                ],
            },
        ).payload

        def mutate(body: dict[str, Any]) -> None:
            body["run_artifacts"][first_window] = failed_payload

        self._validate_tampered(mutate, "typed PROVIDER_FAILURE")


def _tamper_window_identity(body: dict[str, Any]) -> None:
    first_window = next(iter(body["run_artifacts"]))
    payload = body["run_artifacts"][first_window]
    window_dict = payload["run"]["window"]
    window_dict["window_id"] = window_dict["source_id"] + ":w:tampered"
    payload["input_hashes"]["source_window_sha256"] = canonical_sha256(
        window_dict,
    )
    body["run_artifacts"][first_window] = _reseal(payload)


def _tamper_payload_lineage(
    body: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    first_window = next(iter(body["run_artifacts"]))
    payload = body["run_artifacts"][first_window]
    payload[key] = value
    body["run_artifacts"][first_window] = _reseal(payload)


class DiscriminativeValidatorTamperTests(unittest.TestCase):
    def _validate_tampered(
        self,
        mutate: Callable[[dict[str, Any]], None],
        message: str,
        *,
        filename: str = "discriminative_raw",
    ) -> None:
        fast = _fast_evidence_state()
        state = fast["state"]
        work = _tamper_copy(
            state, fast["pristine"], name="rerun-evidence-disc-tamper",
        )
        _tamper_evidence_file(work, filename=filename, mutate=mutate)
        with self.assertRaisesRegex(ValueError, message):
            validate_rerun_evidence(
                work,
                **_rerun_args(state),
                run_cv_fn=_fake_run_cv,
                compute_rankings_fn=_fake_compute_rankings,
            )

    def test_folds_structure_and_replay_mismatch_rejected(self):
        cases = [
            (
                "structural fold key removed",
                lambda body: body["folds"][0].pop("test_window_id"),
                "fold 0 key set is invalid",
            ),
            (
                "fold replay mismatch",
                lambda body: body["folds"][0].__setitem__(
                    "train_positive_count",
                    body["folds"][0]["train_positive_count"] + 1,
                ),
                "folds do not reproduce",
            ),
        ]
        for label, mutate, message in cases:
            with self.subTest(mutation=label):
                self._validate_tampered(mutate, message)

    def test_fit_scope_replay_mismatch_rejected(self):
        def mutate(body: dict[str, Any]) -> None:
            record = body["fit_scope"]["logistic_A"]["0"]
            record["train_positive_count"] = record[
                "train_positive_count"
            ] + 1

        self._validate_tampered(mutate, "fit_scope does not reproduce")

    def test_cell_score_and_metric_tampering_rejected(self):
        def score_mutate(body: dict[str, Any]) -> None:
            scores = body["windows"][0]["cell_scores"]["logistic_A"]
            candidate_id = next(iter(scores))
            entry = scores[candidate_id]
            entry["score"] = (
                0.95 if entry["score"] >= KEEP_THRESHOLD else 0.05
            )

        self._validate_tampered(score_mutate, "cell scores do not reproduce")

        def metric_mutate(body: dict[str, Any]) -> None:
            metric = body["windows"][0]["cell_metrics"]["logistic_A"]
            current = metric["precision"]["rate"]
            metric["precision"]["rate"] = (
                0.123456 if current != 0.123456 else 0.654321
            )

        self._validate_tampered(metric_mutate, "cell metrics do not reproduce")

    def test_every_cell_remains_structurally_required(self):
        cases = [
            (
                "artifact cells list",
                lambda body: body.__setitem__("cells", list(CELLS)[:3]),
                "phase2k discriminative cells were changed",
            ),
            (
                "fit_scope cell",
                lambda body: body["fit_scope"].pop("logistic_B"),
                "phase2k discriminative fit_scope key set is invalid",
            ),
            (
                "cell_scores cell",
                lambda body: body["windows"][0]["cell_scores"].pop(
                    "logistic_B",
                ),
                "phase2k discriminative cell scores are incomplete",
            ),
            (
                "cell_metrics cell",
                lambda body: body["windows"][0]["cell_metrics"].pop(
                    "logistic_B",
                ),
                "phase2k discriminative cell metrics are incomplete",
            ),
        ]
        for label, mutate, message in cases:
            with self.subTest(mutation=label):
                self._validate_tampered(mutate, message)


class CliAliasContractTests(unittest.TestCase):
    def test_frozen_ability_aliases_and_pool_champion_names(self):
        import scripts.run_phase2k_downstream_rerun as cli

        self.assertEqual(
            cli.ABILITY_ALIASES,
            (
                "Q", "W", "E", "R", "ult", "ultimate",
                "Flash", "Teleport", "Ignite", "Exhaust", "Ward", "Sweeper",
            ),
        )
        entity_aliases, ability_aliases = cli.load_alias_sets()
        self.assertIsInstance(entity_aliases, list)
        self.assertTrue(entity_aliases)
        self.assertTrue(all(
            isinstance(alias, str) and alias.strip()
            for alias in entity_aliases
        ))
        self.assertEqual(ability_aliases, cli.ABILITY_ALIASES)

    def test_alias_sets_fail_closed_on_absent_empty_invalid_champion_names(self):
        import scripts.run_phase2k_downstream_rerun as cli

        cases = [
            ("absent", {"selection_policy": {}}),
            ("empty", {"selection_policy": {"champion_names": []}}),
            (
                "blank entry",
                {"selection_policy": {"champion_names": [""]}},
            ),
            (
                "non-string entry",
                {"selection_policy": {"champion_names": [None]}},
            ),
            (
                "mixed blank entry",
                {"selection_policy": {"champion_names": ["Lux", ""]}},
            ),
        ]
        for label, pool in cases:
            with self.subTest(fail_mode=label):
                with mock.patch.object(
                    cli, "load_semantic_window_pool", return_value=pool,
                ):
                    with self.assertRaisesRegex(ValueError, "champion_names"):
                        cli.load_alias_sets()

    def test_preflight_cli_passes_non_empty_aliases(self):
        import scripts.run_phase2k_downstream_rerun as cli

        state = _shared_state()
        captured: dict[str, Any] = {}

        def fake_preflight(**kwargs):
            captured.update(kwargs)
            return {
                "content_sha256": "0" * 64,
                "schema_version": "x",
                "gates": {"target_count": 311, "window_count": 30},
            }

        with mock.patch.object(
            cli, "build_preflight_contract", side_effect=fake_preflight,
        ):
            code = cli.main([
                "--phase2k-dir", str(state["output"]),
                "--alignment-packet", str(state["alignment_packet_path"]),
                "--alignment-summary", str(state["alignment_summary_path"]),
                "--reviewed-packet", str(state["packet_path"]),
                "--coverage", str(state["coverage_path"]),
            ])
        self.assertEqual(code, 0)
        self.assertTrue(captured["entity_aliases"])
        self.assertEqual(captured["ability_aliases"], cli.ABILITY_ALIASES)

    def test_load_provider_annotation_is_typing_callable(self):
        import scripts.run_phase2k_downstream_rerun as cli

        annotation = typing.get_type_hints(cli.load_provider)["return"]
        self.assertIs(typing.get_origin(annotation), collections.abc.Callable)


if __name__ == "__main__":
    unittest.main()
