import json
from dataclasses import replace
import hashlib
import io
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.phase2g_endpoint_recovery import load_benchmark
from pipeline.phase2g_silver import canonical_sha256
from pipeline.phase2h_endpoint_scoring import (
    DROP,
    KEEP,
    KEEP_THRESHOLD,
    LGBM_CONFIG,
    CandidateRow,
    build_dataset,
    classify_all_errors,
    compute_rankings,
    extract_dense_features,
    extract_sparse_inputs,
)
from pipeline.phase2i_endpoint_scoring import (
    BASELINE_CELLS,
    CELLS_C,
    LOCKED_WINDOW_IDS,
    Phase2IError,
    PHASE2H_RUN1_AGGREGATE_SHA256,
    RUN_VERSION,
    _aggregate_top,
    _SYNTAX_TABLE_KEYS,
    _candidate_syntax_projection,
    _extract_tar,
    _strict_equal,
    _syntax_summary,
    _verify_explainability,
    build_aggregate_c,
    build_candidate_syntax,
    build_phase2i_window_table,
    classify_all_errors_c,
    close_phase2h_baseline,
    compare_phase2i_artifacts,
    load_phase2h_baseline,
    parser_error_diagnostics,
    publish_phase2i_artifact,
    run_experiment_c,
    syntax_vs_inherited_importance,
    universally_missed_analysis,
    validate_cv_folds_match_baseline,
)
from pipeline.phase2i_syntax import (
    BOUNDARY_STATUSES,
    STANZA_PROCESSORS,
    UdParse,
    assets_manifest_sha256,
    compute_candidate_syntax,
    verify_assets_provenance,
)
from tests.test_phase2i_syntax import _fixture_parse
from tests._phase2i_helpers import (
    ARCHIVE as _HELPER_ARCHIVE,
    ASSETS as _HELPER_ASSETS,
    BENCHMARK as _HELPER_BENCHMARK,
    experiment_result as _shared_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_GIT_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
    text=True, capture_output=True,
).stdout.strip()
BENCHMARK = _HELPER_BENCHMARK
ASSETS = _HELPER_ASSETS
ARCHIVE = _HELPER_ARCHIVE


def _experiment_result():
    return _shared_experiment()


class Phase2ICellTests(unittest.TestCase):
    def test_threshold_is_fixed_at_half(self):
        self.assertEqual(KEEP_THRESHOLD, 0.5)

    def test_ranking_selection_uses_fixed_threshold(self):
        benchmark = load_benchmark(BENCHMARK)
        dataset = build_dataset(benchmark)
        oof_scores = {
            window_id: {
                row.candidate_id: {
                    "logistic_C": 0.5 if index == 0 else 0.25,
                }
                for index, row in enumerate(window["rows"])
            }
            for window_id, window in dataset["windows"].items()
        }
        rankings = compute_rankings(
            dataset, oof_scores, cells=("logistic_C",),
        )
        for window_id, window in dataset["windows"].items():
            rows = window["rows"]
            first = rankings[window_id]["logistic_C"][rows[0].candidate_id]
            self.assertEqual(first["selected"], KEEP)
            second = rankings[window_id]["logistic_C"][rows[1].candidate_id]
            self.assertEqual(second["selected"], DROP)
            self.assertEqual(first["score"], 0.5)
            self.assertEqual(second["score"], 0.25)

    def test_cv_uses_identical_grouped_folds_and_train_only_scope(self):
        if not ASSETS.is_dir():
            self.skipTest("parser assets not present")
        result = _experiment_result()
        dataset = result["dataset"]
        folds = result["folds"]
        fit_scope = result["fit_scope"]
        oof_scores = result["oof_scores"]
        self.assertEqual(len(folds), 5)
        fold_windows = [
            (tuple(fold["train_window_ids"]), fold["test_window_id"])
            for fold in folds
        ]
        self.assertEqual(len(set(fold_windows)), 5)
        # OOF scores are keyed window -> candidate -> cell (the Phase 2H
        # compute_rankings contract); the nested candidate entries cover
        # every candidate exactly once.
        self.assertEqual(set(oof_scores), set(dataset["windows"]))
        self.assertEqual(
            sum(len(scores) for scores in oof_scores.values()),
            sum(
                len(dataset["windows"][window_id]["rows"])
                for window_id in dataset["windows"]
            ),
        )
        for window_id, scores in oof_scores.items():
            row_ids = {
                row.candidate_id
                for row in dataset["windows"][window_id]["rows"]
            }
            self.assertEqual(set(scores), row_ids)
            for row in dataset["windows"][window_id]["rows"]:
                self.assertEqual(
                    set(scores[row.candidate_id]), set(CELLS_C),
                )
        for cell in CELLS_C:
            for fold_index in range(5):
                scope = fit_scope[cell][fold_index]
                self.assertEqual(scope["fit_scope"], "training windows only")
                self.assertIn("mean_sha256", scope["scaler"])
                self.assertIn("vocabulary_sha256", scope["vectorizer"])
                self.assertIn(
                    "vocabulary_sha256", scope["syntax_encoder"],
                )
                self.assertIn(
                    "oov_value_count", scope["syntax_encoder"],
                )
                self.assertIn(
                    "test_oov_token_type_count", scope["vectorizer"],
                )
        self.assertTrue(result["fold_match_validation"]["validated"])

    def test_folds_exactly_match_archived_phase2h(self):
        baseline = load_phase2h_baseline(ARCHIVE)
        try:
            archived = baseline["aggregate"]["folds"]
            self.assertEqual(len(archived), 5)
            result = _experiment_result()
            self.assertEqual(
                validate_cv_folds_match_baseline(
                    result["folds"], archived,
                ),
                [],
            )
            for fold, archived_fold in zip(result["folds"], archived):
                self.assertEqual(fold, archived_fold)
            self.assertEqual(
                result["fold_match_validation"]["compared_against"],
                "archived Phase 2H run-1 folds",
            )
            type_tampered = json.loads(json.dumps(result["folds"]))
            type_tampered[0]["fold_index"] = False
            self.assertTrue(
                validate_cv_folds_match_baseline(
                    type_tampered, archived,
                ),
                "type-aware frozen fold comparison accepted False as 0",
            )
        finally:
            close_phase2h_baseline(baseline)

    def test_build_candidate_syntax_requires_exact_parse_set(self):
        benchmark = load_benchmark(BENCHMARK)
        dataset = build_dataset(benchmark)
        parses = _experiment_result()["parses"]
        # Extra parse is rejected.
        with self.assertRaises(Exception):
            build_candidate_syntax(dataset, {
                **parses, "extra-window": parses["mid-push-prevents-side-collapse"],
            })
        # Missing parse is rejected.
        missing = {
            window_id: parse
            for window_id, parse in parses.items()
            if window_id != "mid-push-prevents-side-collapse"
        }
        with self.assertRaises(Exception):
            build_candidate_syntax(dataset, missing)
        # Parse window_id mismatch is rejected.
        tampered = {
            window_id: parse
            for window_id, parse in parses.items()
        }
        first = tampered["mid-push-prevents-side-collapse"]
        import hashlib as _hashlib
        wrong_id = UdParse(
            window_id="other",
            text=first.text,
            text_sha256=_hashlib.sha256(
                first.text.encode("utf-8"),
            ).hexdigest(),
            sentences=first.sentences,
            parser=first.parser,
            parser_version=first.parser_version,
            package=first.package,
            processors=first.processors,
            language=first.language,
            model_assets=first.model_assets,
            assets_manifest_sha256=first.assets_manifest_sha256,
            pipeline_version=first.pipeline_version,
            parse_sha256="",
        )
        wrong_id = UdParse(
            **{
                **wrong_id.__dict__,
                "parse_sha256": canonical_sha256(
                    wrong_id.canonical_serialization(),
                ),
            },
        )
        tampered["mid-push-prevents-side-collapse"] = wrong_id
        with self.assertRaises(Exception):
            build_candidate_syntax(dataset, tampered)


class Phase2IStrictEqualityTests(unittest.TestCase):
    def test_strict_equality_is_type_aware(self):
        self.assertTrue(_strict_equal(True, True))
        self.assertFalse(_strict_equal(True, 1))
        self.assertFalse(_strict_equal(1, 1.0))
        self.assertFalse(_strict_equal([1], [1.0]))
        self.assertTrue(_strict_equal(
            {"a": [1, "x"], "b": None},
            {"b": None, "a": [1, "x"]},
        ))
        self.assertFalse(_strict_equal(
            {"a": 1, "b": 2}, {"a": 1},
        ))


class Phase2IObliquePersistenceTests(unittest.TestCase):
    def test_oblique_roles_are_in_canonical_projection_and_summary(self):
        text = "you reset the wave by hand"
        parse = _fixture_parse(
            text,
            [
                (1, "you", 0, 3, [1]),
                (2, "reset", 4, 9, [2]),
                (3, "the", 10, 13, [3]),
                (4, "wave", 14, 18, [4]),
                (5, "by", 19, 21, [5]),
                (6, "hand", 22, 26, [6]),
            ],
            [
                (1, 1, "you", "you", "PRON", "PRP", 2, "nsubj", 0, 3),
                (
                    2, 2, "reset", "reset", "VERB", "VBP", 0, "root",
                    4, 9, "VerbForm=Fin",
                ),
                (3, 3, "the", "the", "DET", "DT", 4, "det", 10, 13),
                (4, 4, "wave", "wave", "NOUN", "NN", 2, "obj", 14, 18),
                (5, 5, "by", "by", "ADP", "IN", 6, "case", 19, 21),
                (6, 6, "hand", "hand", "NOUN", "NN", 2, "obl", 22, 26),
            ],
        )
        row = type(
            "Row",
            (),
            {
                "window_id": "fixture",
                "candidate_id": "fixture:c1",
                "start": 4,
                "end": 26,
                "text": "reset the wave by hand",
            },
        )()
        record = compute_candidate_syntax(parse, row)

        projection = _candidate_syntax_projection(parse, record)
        self.assertEqual(set(projection), _SYNTAX_TABLE_KEYS)
        self.assertIs(projection["predicate_internal_oblique"], True)
        self.assertIs(projection["predicate_external_oblique"], False)

        summary = _syntax_summary(record)
        self.assertIs(summary["predicate_internal_oblique"], True)
        self.assertIs(summary["predicate_external_oblique"], False)

        # A truncated span that excludes the oblique child keeps the roles
        # separate: the object stays internal and the oblique becomes external.
        truncated = compute_candidate_syntax(
            parse,
            type(
                "Row",
                (),
                {
                    "window_id": "fixture",
                    "candidate_id": "fixture:c2",
                    "start": 4,
                    "end": 18,
                    "text": "reset the wave",
                },
            )(),
        )
        summary = _syntax_summary(truncated)
        self.assertIs(summary["predicate_internal_oblique"], False)
        self.assertIs(summary["predicate_external_oblique"], True)


class _FakeLightGbm:
    """Minimal model stub exposing only the frozen gain-importance vector."""

    def __init__(self, importances):
        self._importances = np.asarray(importances, dtype=np.float64)

    @property
    def feature_importances_(self):
        return self._importances


class Phase2IExplainabilitySyntheticTests(unittest.TestCase):
    """Synthetic importance data independent of the real offline run."""

    @staticmethod
    def _synthetic_fitted_models():
        folds = 5
        per_fold_competitors = 20
        names = [
            f"syntax:f{fold}_{index}"
            for fold in range(folds)
            for index in range(per_fold_competitors)
        ]
        names.append("syntax:tail")
        names.append("ngram=keep")
        fitted: dict[int, tuple[_FakeLightGbm, list[str]]] = {}
        for fold in range(folds):
            values: list[float] = []
            for other in range(folds):
                values.extend([
                    2.0 if other == fold else 0.01
                    for _ in range(per_fold_competitors)
                ])
            values.append(1.0)  # syntax:tail below the 20 fold leaders
            values.append(0.5)  # ngram=keep
            fitted[fold] = (_FakeLightGbm(values), list(names))
        return {"lightgbm_C": fitted}

    def test_importance_aggregate_uses_complete_folds_beyond_top20(self):
        entry = syntax_vs_inherited_importance(
            self._synthetic_fitted_models(),
        )["lightgbm_C"]
        for fold, fold_data in entry["per_fold"].items():
            top_names = [
                name for name, _ in fold_data["top_importances"]
            ]
            self.assertNotIn("syntax:tail", top_names, fold_data)
            self.assertEqual(len(fold_data["top_importances"]), 20)
            self.assertGreaterEqual(len(fold_data["importances"]), 21)
            complete_names = [
                name for name, _ in fold_data["importances"]
            ]
            self.assertIn("syntax:tail", complete_names)
            self.assertEqual(
                fold_data["top_importances"],
                fold_data["importances"][:20],
            )
        syntax_names = [
            item["feature"] for item in entry["aggregate_syntax_top"]
        ]
        self.assertIn("syntax:tail", syntax_names)
        self.assertEqual(
            syntax_names.index("syntax:tail"),
            0,
            entry["aggregate_syntax_top"],
        )
        inherited_names = [
            item["feature"] for item in entry["aggregate_inherited_top"]
        ]
        self.assertIn("ngram=keep", inherited_names)

    def test_explainability_verifier_recomputes_aggregate_from_complete_data(
        self,
    ):
        entry = syntax_vs_inherited_importance(
            self._synthetic_fitted_models(),
        )["lightgbm_C"]
        coeffs_per_fold = {
            str(index): {"syntax:x": 1.0} for index in range(5)
        }
        aggregate = {
            "syntax_coefficients": {
                "logistic_C": {
                    "kind": "logistic_syntax_coefficients",
                    "per_fold": coeffs_per_fold,
                    "aggregate_top_positive": _aggregate_top(
                        [("syntax:x", 1.0)] * 5, 15,
                    ),
                    "aggregate_top_negative": [],
                    "syntax_feature_count": 1,
                },
            },
            "syntax_vs_inherited_importance": {
                "lightgbm_C": entry,
            },
        }
        problems: list[str] = []
        _verify_explainability(aggregate, problems)
        self.assertEqual(problems, [])

        tampered = json.loads(json.dumps(aggregate))
        fold_data = tampered["syntax_vs_inherited_importance"][
            "lightgbm_C"
        ]["per_fold"]["0"]
        tail_index = next(
            index for index, item in enumerate(fold_data["importances"])
            if item[0] == "syntax:tail"
        )
        fold_data["importances"].pop(tail_index)
        problems = []
        _verify_explainability(tampered, problems)
        self.assertTrue(
            any(
                "gains" in problem or "aggregate" in problem
                for problem in problems
            ),
            problems,
        )


class Phase2IExperimentTests(unittest.TestCase):
    def setUp(self):
        if not ASSETS.is_dir():
            self.skipTest("parser assets not present")
        self.result = _experiment_result()

    def test_dataset_counts_are_unchanged(self):
        self.assertEqual(len(self.result["dataset"]["windows"]), 5)
        total = sum(
            len(window["rows"])
            for window in self.result["dataset"]["windows"].values()
        )
        self.assertEqual(total, 16624)
        self.assertEqual(
            sum(
                1
                for window in self.result["dataset"]["windows"].values()
                for row in window["rows"]
                if row.is_gold_positive
            ),
            33,
        )
        self.assertTrue(
            self.result["baseline_immutability_audit"]["validated"],
        )
        self.assertEqual(
            self.result["baseline_immutability_audit"]["problems"], [],
        )

    def test_early_immutability_audit_rejects_tampering(self):
        from pipeline.phase2i_endpoint_scoring import (
            Phase2IError,
            validate_early_immutability,
        )
        from unittest.mock import patch

        benchmark = load_benchmark(BENCHMARK)
        dataset = build_dataset(benchmark)
        baseline = load_phase2h_baseline(ARCHIVE)
        try:
            audit = validate_early_immutability(
                dataset, baseline["window_tables"],
            )
            self.assertTrue(audit["validated"], audit["problems"])
            self.assertEqual(audit["window_count"], 5)
            self.assertEqual(audit["candidate_count"], 16624)
            self.assertEqual(audit["positive_count"], 33)

            tampered = build_dataset(benchmark)
            window_id = sorted(tampered["windows"])[0]
            rows = list(tampered["windows"][window_id]["rows"])
            rows[0] = replace(rows[0], text="tampered")
            tampered["windows"][window_id]["rows"] = tuple(rows)
            tampered_audit = validate_early_immutability(
                tampered, baseline["window_tables"],
            )
            self.assertFalse(tampered_audit["validated"])
            self.assertTrue(tampered_audit["problems"])

            # Enforcement happens before parsing: the immutability failure is
            # raised and the parser is never invoked.
            tampered_tables = json.loads(
                json.dumps(baseline["window_tables"]),
            )
            tampered_tables[window_id]["candidates"][0]["start"] = 999999
            tampered_baseline = {
                **baseline,
                "window_tables": tampered_tables,
            }
            with patch(
                "pipeline.phase2i_endpoint_scoring.load_phase2h_baseline",
                return_value=tampered_baseline,
            ), patch(
                "pipeline.phase2i_endpoint_scoring.parse_window_text",
                side_effect=AssertionError(
                    "parse must not run before the immutability audit",
                ),
            ):
                with self.assertRaises(Phase2IError):
                    run_experiment_c(
                        BENCHMARK,
                        assets_dir=ASSETS,
                        baseline_archive=ARCHIVE,
                    )
        finally:
            close_phase2h_baseline(baseline)

    def test_cells_and_metrics_structure(self):
        self.assertEqual(tuple(self.result["cells"]), CELLS_C)
        for cell in CELLS_C:
            metrics = self.result["metrics"][cell]
            self.assertEqual(metrics["candidate_count"], 16624)
            self.assertEqual(metrics["label_keep_count"], 33)
            self.assertEqual(len(metrics["per_fold"]), 5)
            self.assertIn("precision", metrics)
            self.assertIn("roc_auc", metrics)
            self.assertIn("average_precision", metrics)
            self.assertIn("precision_at_k", metrics)
            self.assertIn("gold_rank", metrics)
            self.assertIsInstance(metrics["gold_rank"]["mean"], float)
            self.assertIsInstance(metrics["gold_rank"]["median"], float)
            for k in ("1", "3", "5", "10"):
                self.assertIn(k, metrics["recall_at_k"])
                self.assertIn(k, metrics["precision_at_k"])
            for window_metrics in metrics["per_fold"].values():
                self.assertIn("precision_at_k", window_metrics)
                self.assertIn("gold_rank", window_metrics)
            self.assertEqual(
                self.result["deltas"][cell]["baseline_cell"],
                cell.replace("_C", "_B"),
            )
            self.assertIn("delta", self.result["deltas"][cell])
            delta = self.result["deltas"][cell]["delta"]
            self.assertIn("precision_at_k", delta)
            self.assertIn("gold_rank", delta)
            self.assertIn("delta_mean", delta["gold_rank"])
            self.assertIn("delta_median", delta["gold_rank"])

    def test_deltas_match_baseline_and_current(self):
        for cell in CELLS_C:
            b_cell = cell.replace("_C", "_B")
            baseline = self.result["baseline_metrics"][b_cell]
            current = self.result["metrics"][cell]
            delta = self.result["deltas"][cell]["delta"]
            self.assertAlmostEqual(
                delta["precision"],
                current["precision"]["rate"]
                - baseline["precision"]["rate"],
            )
            self.assertAlmostEqual(
                delta["recall_at_k"]["10"]["delta"],
                current["recall_at_k"]["10"]["rate"]
                - baseline["recall_at_k"]["10"]["rate"],
            )
            self.assertAlmostEqual(
                delta["precision_at_k"]["10"]["delta"],
                current["precision_at_k"]["10"]["rate"]
                - baseline["precision_at_k"]["10"]["rate"],
            )
            self.assertAlmostEqual(
                delta["gold_rank"]["delta_median"],
                current["gold_rank"]["median"]
                - baseline["gold_rank"]["median"],
            )

    def test_c_preprocessing_inherits_exact_b_block(self):
        from pipeline.phase2h_endpoint_scoring import (
            CellPreprocessor,
            DENSE_A_FEATURES,
            DENSE_B_EXTRA_FEATURES,
        )
        from pipeline.phase2i_endpoint_scoring import CellPreprocessorC
        from pipeline.phase2i_syntax import (
            DENSE_C_EXTRA_FEATURES,
            dense_c_matrix,
            syntax_groups_from_records,
        )

        dataset = self.result["dataset"]
        window_ids = sorted(dataset["windows"])
        dense_a, dense_b_extra = extract_dense_features(dataset, window_ids)
        dense_b = np.hstack([dense_a, dense_b_extra])
        texts, boundaries = extract_sparse_inputs(dataset, window_ids)
        syntax_records = [
            record
            for window_id in window_ids
            for record in self.result["candidate_syntax"][window_id]
        ]
        syntax_groups = syntax_groups_from_records(syntax_records)
        sample = list(range(0, len(dense_b), 40))[:500]
        dense_train = dense_b[sample]
        syntax_dense_train = dense_c_matrix(
            [syntax_records[index] for index in sample],
        )
        dense_c_train = np.hstack([dense_train, syntax_dense_train])
        texts_train = [texts[index] for index in sample]
        boundaries_train = [boundaries[index] for index in sample]
        syntax_train = [syntax_groups[index] for index in sample]
        b_pre = CellPreprocessor("B")
        b_pre.fit(dense_train, texts_train, boundaries_train)
        x_b = b_pre.transform(dense_train, texts_train, boundaries_train)
        c_pre = CellPreprocessorC()
        c_pre.fit(
            dense_c_train, texts_train, boundaries_train, syntax_train,
        )
        x_c = c_pre.transform(
            dense_c_train, texts_train, boundaries_train, syntax_train,
        )
        # The Feature Set C matrix must start with the exact Phase 2H B block
        # under the same training rows; syntax columns are appended only.
        self.assertEqual(x_b.shape[0], x_c.shape[0])
        self.assertLessEqual(x_b.shape[1], x_c.shape[1])
        np.testing.assert_allclose(
            x_c[:, :x_b.shape[1]].toarray(), x_b.toarray(), rtol=1e-12,
        )
        names_b = b_pre.feature_names(
            list(DENSE_A_FEATURES) + list(DENSE_B_EXTRA_FEATURES),
        )
        names_c = c_pre.feature_names(
            list(DENSE_A_FEATURES)
            + list(DENSE_B_EXTRA_FEATURES)
            + list(DENSE_C_EXTRA_FEATURES)
        )
        self.assertEqual(names_c[: len(names_b)], names_b)
        self.assertEqual(
            names_c[len(names_b):],
            list(DENSE_C_EXTRA_FEATURES)
            + c_pre.syntax_encoder.feature_names(),
        )
        self.assertEqual(LGBM_CONFIG["importance_type"], "gain")

    def test_universally_missed_analysis_is_derived_and_complete(self):
        baseline = load_phase2h_baseline(ARCHIVE)
        try:
            analysis = universally_missed_analysis(
                baseline["window_tables"],
                self.result["rankings"],
                self.result["dataset"],
                self.result["candidate_syntax"],
                self.result["parses"],
            )
        finally:
            close_phase2h_baseline(baseline)
        self.assertTrue(analysis["validated"])
        self.assertTrue(analysis["derived_from_baseline_tables"])
        self.assertEqual(len(analysis["entries"]), 7)
        for entry in analysis["entries"]:
            self.assertIn("phase2i_movement", entry)
            self.assertIn("logistic_C", entry["phase2i_movement"])
            self.assertIn("lightgbm_C", entry["phase2i_movement"])
            self.assertIn("syntax", entry)
            self.assertIn("parser_alignment_quality", entry)
            self.assertIn(
                "failure_appears_unrelated_to_syntax", entry,
            )
            self.assertIsInstance(
                entry["failure_appears_unrelated_to_syntax"], bool,
            )
            self.assertIn(
                "failure_unrelated_evidence",
                entry,
            )

    def test_universally_missed_validation_passes(self):
        self.assertEqual(
            self.result["universally_missed_problems"], [],
        )
        self.assertEqual(len(self.result["universally_missed"]), 7)

    def test_parser_diagnostics_are_explicit(self):
        diagnostics = self.result["parser_diagnostics"]
        counts = diagnostics["boundary_status_counts_total"]
        self.assertEqual(
            set(counts), set(BOUNDARY_STATUSES),
        )
        self.assertEqual(sum(counts.values()), 16624)
        self.assertIsInstance(
            diagnostics["objective_parser_error_total"], int,
        )
        self.assertIsInstance(diagnostics["diagnostic_only_total"], int)
        self.assertIn("definition", diagnostics)
        self.assertIn(
            "never automatic parser blame", diagnostics["definition"],
        )

    def test_parser_feature_error_is_activated_for_objective_errors(self):
        from pipeline.phase2h_endpoint_scoring import classify_all_errors

        errors = classify_all_errors_c(
            self.result["dataset"],
            self.result["rankings"],
            self.result["parses"],
            self.result["candidate_syntax"],
        )
        phase2h_errors = classify_all_errors(
            self.result["dataset"], self.result["rankings"],
            cells=CELLS_C,
        )
        counts = {
            cell: {
                code: 0 for code in ("PARSER_FEATURE_ERROR",)
            }
            for cell in CELLS_C
        }
        for cell in CELLS_C:
            for window_id in self.result["dataset"]["windows"]:
                for code in errors[cell][window_id].values():
                    if code == "PARSER_FEATURE_ERROR":
                        counts[cell]["PARSER_FEATURE_ERROR"] += 1
        # Every PARSER_FEATURE_ERROR candidate must carry an objective
        # parser/alignment error (UNALIGNED or TOKEN_SURFACE_MISMATCH).
        diagnostics = self.result["parser_diagnostics"]
        objective_ids = {
            window_id: {
                error["candidate_id"]
                for error in per_window["objective_parser_errors"]
            }
            for window_id, per_window in diagnostics["per_window"].items()
        }
        for cell in CELLS_C:
            for window_id in self.result["dataset"]["windows"]:
                for candidate_id, code in errors[cell][window_id].items():
                    if code == "PARSER_FEATURE_ERROR":
                        self.assertIn(
                            candidate_id, objective_ids[window_id],
                        )
            self.assertGreaterEqual(
                counts[cell]["PARSER_FEATURE_ERROR"], 0,
            )
            # False-negative rank codes are preserved: gold-ranked codes are
            # never relabeled as parser feature errors.
            for window_id in self.result["dataset"]["windows"]:
                for candidate_id, code in errors[cell][window_id].items():
                    plain = phase2h_errors[cell][window_id][candidate_id]
                    if plain in (
                        "GOLD_RANKED_HIGH_THRESHOLD_MISS",
                        "GOLD_RANKED_LOW",
                    ):
                        self.assertEqual(code, plain)
                    elif plain is not None:
                        self.assertIn(
                            code,
                            (plain, "PARSER_FEATURE_ERROR"),
                        )
        # The taxonomy still covers every candidate exactly once.
        for cell in CELLS_C:
            self.assertEqual(
                sum(
                    1
                    for window_id in self.result["dataset"]["windows"]
                    for code in errors[cell][window_id].values()
                ),
                16624,
            )

    def test_parser_taxonomy_injected_fp_fn_and_ambiguous_rules(self):
        cells = ("logistic_C",)

        def row(candidate_id, start, end, text, label):
            return CandidateRow(
                case_id="case",
                window_id="full-window",
                candidate_id=candidate_id,
                alias=candidate_id,
                start=start,
                end=end,
                absolute_start=start,
                absolute_end=end,
                text=text,
                segment_ids=(),
                segment_bounds=(),
                type_hints=(),
                source_kind="transcript",
                is_gold_positive=label == KEEP,
                label=label,
                excluded=False,
                ambiguity_state="NONE",
                gold_mention_ids=(),
                gold_node_types=(),
            )

        # A token-free parse makes every row an objective UNALIGNED error.
        unaligned_text = "x" * 100
        unaligned_keep = row("keep-un", 0, 1, "x", KEEP)
        unaligned_drop = row("drop-un", 1, 2, "x", DROP)
        unaligned_parse = UdParse(
            window_id="unaligned",
            text=unaligned_text,
            text_sha256=hashlib.sha256(
                unaligned_text.encode("utf-8"),
            ).hexdigest(),
            sentences=(),
            parser="fixture",
            parser_version="test",
            package="ewt",
            processors=STANZA_PROCESSORS,
            language="en",
            model_assets=(),
            assets_manifest_sha256=assets_manifest_sha256(()),
            pipeline_version="test",
            parse_sha256="",
        )
        unaligned_parse = UdParse(
            **{
                **unaligned_parse.__dict__,
                "parse_sha256": canonical_sha256(
                    unaligned_parse.canonical_serialization(),
                ),
            },
        )
        unaligned_records = {
            "keep-un": compute_candidate_syntax(
                unaligned_parse, unaligned_keep,
            ),
            "drop-un": compute_candidate_syntax(
                unaligned_parse, unaligned_drop,
            ),
        }

        # An MWT-cut boundary is AMBIGUOUS with tokens present: it must never
        # be automatic parser blame.
        ambiguous_text = "we don't go"
        ambiguous_parse = _fixture_parse(
            ambiguous_text,
            [
                (1, "we", 0, 2, [1]),
                (2, "don't", 3, 8, [2, 3]),
                (4, "go", 9, 11, [4]),
            ],
            [
                (1, 1, "we", "we", "PRON", "PRP", 4, "nsubj", 0, 2),
                (2, 2, "do", "do", "AUX", "VBP", 4, "aux", 3, 8),
                (3, 2, "n't", "not", "PART", "RB", 4, "neg", 3, 8),
                (4, 4, "go", "go", "VERB", "VB", 0, "root", 9, 11),
            ],
        )
        ambiguous_drop = row("drop-mwt", 4, 8, "on't", DROP)
        ambiguous_record = compute_candidate_syntax(
            ambiguous_parse, ambiguous_drop,
        )
        self.assertEqual(
            ambiguous_record.boundary_status,
            "AMBIGUOUS",
        )
        self.assertNotIn("NO_INTERSECTING_TOKENS", ambiguous_record.ambiguity)

        dataset = {
            "windows": {
                "unaligned": {
                    "window_id": "full-window",
                    "bronze_text": unaligned_text,
                    "gold_spans": [(0, 1)],
                    "rows": [unaligned_keep, unaligned_drop],
                },
                "ambiguous": {
                    "window_id": "full-window",
                    "bronze_text": ambiguous_text,
                    "gold_spans": [],
                    "rows": [ambiguous_drop],
                },
            },
        }
        rankings = {
            "unaligned": {
                "logistic_C": {
                    "keep-un": {
                        "score": 0.1, "rank": 99, "selected": DROP,
                    },
                    "drop-un": {
                        "score": 0.9, "rank": 1, "selected": KEEP,
                    },
                },
            },
            "ambiguous": {
                "logistic_C": {
                    "drop-mwt": {
                        "score": 0.9, "rank": 1, "selected": KEEP,
                    },
                },
            },
        }
        parses = {
            "unaligned": unaligned_parse,
            "ambiguous": ambiguous_parse,
        }
        records_by_window = {
            "unaligned": list(unaligned_records.values()),
            "ambiguous": [ambiguous_record],
        }
        plain = classify_all_errors(
            dataset, rankings, cells=cells,
        )
        errors = classify_all_errors_c(
            dataset, rankings, parses, records_by_window, cells=cells,
        )
        # The KEEP false negative keeps its gold-rank code even though it is
        # UNALIGNED; parser diagnostics still record the objective error.
        self.assertNotEqual(
            errors["logistic_C"]["unaligned"]["keep-un"],
            "PARSER_FEATURE_ERROR",
        )
        self.assertEqual(
            errors["logistic_C"]["unaligned"]["keep-un"],
            plain["logistic_C"]["unaligned"]["keep-un"],
        )
        diagnostics = parser_error_diagnostics(
            dataset, parses, records_by_window,
        )
        objective_ids = {
            error["candidate_id"]
            for error in diagnostics["per_window"]["unaligned"][
                "objective_parser_errors"
            ]
        }
        self.assertIn("keep-un", objective_ids)
        # Only the misclassified DROP false positive gets the override.
        self.assertEqual(
            errors["logistic_C"]["unaligned"]["drop-un"],
            "PARSER_FEATURE_ERROR",
        )
        # The AMBIGUOUS false positive is not automatic parser blame.
        self.assertEqual(
            errors["logistic_C"]["ambiguous"]["drop-mwt"],
            plain["logistic_C"]["ambiguous"]["drop-mwt"],
        )
        self.assertNotEqual(
            errors["logistic_C"]["ambiguous"]["drop-mwt"],
            "PARSER_FEATURE_ERROR",
        )

    def test_syntax_coefficients_and_importance_split_present(self):
        coefficients = self.result["syntax_coefficients"]["logistic_C"]
        self.assertEqual(
            coefficients["kind"], "logistic_syntax_coefficients",
        )
        self.assertEqual(len(coefficients["per_fold"]), 5)
        self.assertGreaterEqual(
            coefficients["syntax_feature_count"], 0,
        )
        importance = self.result["syntax_vs_inherited_importance"][
            "lightgbm_C"
        ]
        self.assertEqual(
            importance["kind"],
            "lightgbm_gain_importance_syntax_vs_inherited",
        )
        self.assertEqual(len(importance["per_fold"]), 5)

    def test_training_vs_held_out_diagnostics(self):
        diagnostics = self.result["training_vs_held_out"]
        for cell in CELLS_C:
            self.assertEqual(len(diagnostics[cell]["per_fold"]), 5)
            first = diagnostics[cell]["per_fold"]["0"]
            self.assertIn("train", first)
            self.assertIn("held_out", first)
            self.assertIn("b_token_oov_count", first)
            self.assertIn("syntax_oov_count", first)

    def test_error_taxonomy_b_vs_c_structure(self):
        taxonomy = self.result["error_taxonomy_b_vs_c"]
        for cell in CELLS_C:
            entry = taxonomy[cell]
            self.assertEqual(
                entry["baseline_cell"], cell.replace("_C", "_B"),
            )
            self.assertIn("correct", entry)
            self.assertIn("PARSER_FEATURE_ERROR", entry["codes"])

    def test_parse_artifacts_self_verify(self):
        for window_id, parse in self.result["parses"].items():
            self.assertIsInstance(parse, UdParse)
            self.assertEqual(
                UdParse.from_dict(parse.to_dict()), parse,
            )

    def test_real_output_finiteness_and_clause_root_invariants(self):
        """Benchmark regression check on real EWT output.

        No participle/infinitive/gerund/converb may ever be classified finite
        when a VerbForm feature exists, and no ordinary aux/cop dependent may
        ever appear as a structural clause root across the full 16,624-row
        candidate universe.
        """
        finite_xpos = {"VBD", "VBP", "VBZ", "MD"}
        clause_deps = {"csubj", "ccomp", "advcl", "acl", "xcomp"}
        for window_id, parse in self.result["parses"].items():
            words = {
                f"s{sentence.sentence_id}:w{word.word_id}": word
                for sentence in parse.sentences
                for word in sentence.words
            }
            for record in self.result["candidate_syntax"][window_id]:
                for key in record.finite_verb_ids:
                    word = words[key]
                    feats = {
                        pair.split("=", 1)[0]: pair.split("=", 1)[1]
                        for pair in word.feats.split("|")
                        if "=" in pair
                    }
                    if "VerbForm" in feats:
                        self.assertEqual(
                            feats["VerbForm"], "Fin", word,
                        )
                    else:
                        self.assertIn(word.upos, {"VERB", "AUX"}, word)
                        self.assertIn(word.xpos, finite_xpos, word)
                for key in record.clause_root_ids:
                    word = words[key]
                    family = word.deprel.split(":")[0]
                    if word.head != 0 and family in {"aux", "cop"}:
                        self.fail(
                            f"aux/cop {word.word_id} is a clause root in "
                            f"window {window_id}",
                        )
                    structural = (
                        word.head == 0
                        or family in clause_deps
                        or (
                            word.upos == "VERB"
                            and family == "conj"
                        )
                    )
                    self.assertTrue(structural, word)


class Phase2IArtifactTests(unittest.TestCase):
    def setUp(self):
        if not ASSETS.is_dir():
            self.skipTest("parser assets not present")
        self.result = _experiment_result()

    def _provenance(self):
        provenance = verify_assets_provenance(ASSETS)
        self.assertTrue(provenance["verified"], provenance.get("problems"))
        return provenance

    def _aggregate(self, *, window_table_hashes, created_at="2026-01-01T00:00:00Z"):
        with patch(
            "pipeline.phase2i_endpoint_scoring._git_state_c",
            return_value=(TEST_GIT_COMMIT, False),
        ):
            return build_aggregate_c(
                BENCHMARK,
                self.result,
                repo=ROOT,
                created_at=created_at,
                window_table_hashes=window_table_hashes,
                assets_provenance=self._provenance(),
            )

    def _tables_and_hashes(self, result):
        baseline = load_phase2h_baseline(ARCHIVE)
        try:
            tables = {}
            hashes = {}
            parse_tables = {}
            for window_id in sorted(result["dataset"]["windows"]):
                table = build_phase2i_window_table(
                    result["dataset"], result["rankings"],
                    result["errors"],
                    baseline["window_tables"][window_id],
                    result["parses"][window_id],
                    result["candidate_syntax"][window_id],
                    window_id,
                )
                tables[window_id] = table
                hashes[window_id] = canonical_sha256(table)
                parse_tables[window_id] = (
                    result["parses"][window_id].to_dict()
                )
            return tables, hashes, parse_tables
        finally:
            close_phase2h_baseline(baseline)

    @staticmethod
    def _rehash_artifact(directory):
        """Recompute every outer hash/lock after an inner semantic tamper.

        This simulates an attacker who regenerates ``content_sha256``, the
        per-window candidate-table locks, the parse locks, and the MANIFEST
        file digests after changing inner values, so detection can only come
        from semantic recomputation rather than the outer hash checks.
        """
        root = Path(directory)
        aggregate_path = root / "phase2i-syntax-features.json"
        body = json.loads(aggregate_path.read_text(encoding="utf-8"))
        window_tables = body.get("window_tables")
        if isinstance(window_tables, dict):
            for window_id, info in window_tables.items():
                table = json.loads(
                    (root / "windows" / f"{window_id}.json").read_text(
                        encoding="utf-8",
                    ),
                )
                if isinstance(info, dict):
                    info["candidate_table_sha256"] = canonical_sha256(table)
        parse_hashes = body.get("parse_hashes")
        if isinstance(parse_hashes, dict):
            for window_id in parse_hashes:
                parse_table = json.loads(
                    (root / "parser" / f"{window_id}.json").read_text(
                        encoding="utf-8",
                    ),
                )
                parse_hashes[window_id] = parse_table.get("parse_sha256")
        inner = {
            key: value for key, value in body.items()
            if key != "content_sha256"
        }
        body["content_sha256"] = canonical_sha256(inner)
        aggregate_path.write_text(
            json.dumps(body, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest_path = root / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            entry["file_sha256"] = hashlib.sha256(
                (root / entry["path"]).read_bytes(),
            ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_artifact_is_hash_locked_immutable_and_manifested(self):
        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            aggregate = self._aggregate(window_table_hashes=hashes)
            inner = {
                key: value for key, value in aggregate.items()
                if key != "content_sha256"
            }
            self.assertEqual(
                aggregate["content_sha256"], canonical_sha256(inner),
            )
            self.assertEqual(aggregate["run_version"], RUN_VERSION)
            self.assertEqual(
                aggregate["input_hashes"][
                    "phase2h_run1_aggregate_sha256"
                ],
                PHASE2H_RUN1_AGGREGATE_SHA256,
            )
            self.assertTrue(
                aggregate["baseline_immutability_audit"]["validated"],
            )
            self.assertEqual(
                aggregate["baseline_immutability_audit"]["problems"], [],
            )
            output = Path(tmp) / "phase2i-run"
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                (output / "phase2i-syntax-features.json").exists(),
            )
            self.assertTrue((output / "MANIFEST.json").exists())
            self.assertEqual(len(list((output / "windows").iterdir())), 5)
            self.assertEqual(len(list((output / "parser").iterdir())), 5)
            manifest = json.loads(
                (output / "MANIFEST.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(len(manifest["files"]), 1 + 5 + 5)
            for entry in manifest["files"]:
                path = output / entry["path"]
                self.assertEqual(
                    entry["file_sha256"],
                    __import__("hashlib").sha256(
                        path.read_bytes(),
                    ).hexdigest(),
                )
            with self.assertRaises(ValueError):
                publish_phase2i_artifact(
                    output, aggregate, tables, parse_tables,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )

    def test_compare_artifacts_matches_deterministic_rerun(self):
        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            first = self._aggregate(window_table_hashes=hashes)
            second = self._aggregate(
                window_table_hashes=hashes,
                created_at="2026-08-16T00:00:00Z",
            )
            publish_phase2i_artifact(
                left, first, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            publish_phase2i_artifact(
                right, second, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertEqual(
                compare_phase2i_artifacts(
                    left, right,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )
            # Tamper with a metric value and rehash the outer locks; the
            # comparator must reject the wrong side through semantic
            # recomputation (not merely the outer content hash).
            wrong_dir = Path(tmp) / "wrong"
            shutil.copytree(right, wrong_dir)
            aggregate_path = wrong_dir / "phase2i-syntax-features.json"
            wrong = json.loads(
                aggregate_path.read_text(encoding="utf-8"),
            )
            wrong["metrics"]["logistic_C"]["recall"]["rate"] = 0.1
            aggregate_path.write_text(
                json.dumps(wrong, indent=2) + "\n",
                encoding="utf-8",
            )
            self._rehash_artifact(wrong_dir)
            differences = compare_phase2i_artifacts(
                left, wrong_dir,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "metrics" in item or "verification" in item
                    for item in differences
                ),
                differences,
            )
            # Publication is fail-closed: the same rehashed semantic tamper
            # must be rejected before os.replace and leave no output dir.
            rejected = Path(tmp) / "rejected"
            with self.assertRaises(Exception):
                publish_phase2i_artifact(
                    rejected, wrong, tables, parse_tables,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )
            self.assertFalse(rejected.exists())

    def test_baseline_tampering_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "tampered.tar.gz"
            shutil.copy2(ARCHIVE, tampered)
            data = bytearray(tampered.read_bytes())
            data[100] ^= 0x01
            tampered.write_bytes(bytes(data))
            with self.assertRaises(Exception) as caught:
                load_phase2h_baseline(tampered)
            self.assertIn("SHA-256", str(caught.exception))

    def test_baseline_window_table_immutability_rejects_tampering(self):
        from pipeline.phase2i_endpoint_scoring import Phase2IError

        result = self.result
        baseline = load_phase2h_baseline(ARCHIVE)
        window_id = sorted(result["dataset"]["windows"])[0]
        try:
            baseline_table = baseline["window_tables"][window_id]
            tampered = json.loads(json.dumps(baseline_table))
            # Candidate surface text differs from the frozen dataset.
            tampered["candidates"][0]["text"] = "tampered"
            with self.assertRaises(Phase2IError):
                build_phase2i_window_table(
                    result["dataset"], result["rankings"],
                    result["errors"], tampered,
                    result["parses"][window_id],
                    result["candidate_syntax"][window_id],
                    window_id,
                )
            # Candidate ordering swap must be rejected.
            swapped = json.loads(json.dumps(baseline_table))
            swapped["candidates"][0], swapped["candidates"][1] = (
                swapped["candidates"][1], swapped["candidates"][0],
            )
            with self.assertRaises(Phase2IError):
                build_phase2i_window_table(
                    result["dataset"], result["rankings"],
                    result["errors"], swapped,
                    result["parses"][window_id],
                    result["candidate_syntax"][window_id],
                    window_id,
                )
            # Provenance field tampering must be rejected.
            bad_hash = json.loads(json.dumps(baseline_table))
            bad_hash["bronze_text_sha256"] = "0" * 64
            with self.assertRaises(Phase2IError):
                build_phase2i_window_table(
                    result["dataset"], result["rankings"],
                    result["errors"], bad_hash,
                    result["parses"][window_id],
                    result["candidate_syntax"][window_id],
                    window_id,
                )
            # Missing candidate count must be rejected (no None lookup path).
            truncated = json.loads(json.dumps(baseline_table))
            truncated["candidates"] = truncated["candidates"][:-1]
            with self.assertRaises(Phase2IError):
                build_phase2i_window_table(
                    result["dataset"], result["rankings"],
                    result["errors"], truncated,
                    result["parses"][window_id],
                    result["candidate_syntax"][window_id],
                    window_id,
                )
        finally:
            close_phase2h_baseline(baseline)

    def test_only_requested_cells_are_published(self):
        result = self.result
        baseline = load_phase2h_baseline(ARCHIVE)
        window_id = sorted(result["dataset"]["windows"])[0]
        try:
            baseline_table = baseline["window_tables"][window_id]
            single = build_phase2i_window_table(
                result["dataset"], result["rankings"],
                result["errors"], baseline_table,
                result["parses"][window_id],
                result["candidate_syntax"][window_id],
                window_id,
                cells=("logistic_C",),
            )
            predictions = single["candidates"][0]["predictions"]
            self.assertIn("logistic_C", predictions)
            self.assertNotIn("lightgbm_C", predictions)
            for cell in BASELINE_CELLS:
                self.assertIn(cell, predictions)
            # Official default remains both C cells.
            both = build_phase2i_window_table(
                result["dataset"], result["rankings"],
                result["errors"], baseline_table,
                result["parses"][window_id],
                result["candidate_syntax"][window_id],
                window_id,
            )
            both_predictions = both["candidates"][0]["predictions"]
            self.assertIn("logistic_C", both_predictions)
            self.assertIn("lightgbm_C", both_predictions)
            self.assertEqual(
                set(single["candidates"][0]["syntax"]),
                set(both["candidates"][0]["syntax"]),
            )
        finally:
            close_phase2h_baseline(baseline)

    def test_artifact_verifier_detects_unlisted_and_tampered_parse(self):
        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                left, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            publish_phase2i_artifact(
                right, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertEqual(
                compare_phase2i_artifacts(
                    left, right,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )
            # Unlisted file on one side must be detected.
            (right / "intruder.json").write_text(
                "{}", encoding="utf-8",
            )
            differences = compare_phase2i_artifacts(
                left, right,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("unlisted" in item for item in differences),
            )
            (right / "intruder.json").unlink()
            # A parse file whose canonical hash no longer self-verifies must
            # be detected (not merely its stored parse_sha256 string).
            parse_path = (
                right / "parser" / "mid-push-prevents-side-collapse.json"
            )
            parse_table = json.loads(
                parse_path.read_text(encoding="utf-8"),
            )
            parse_table["parse_sha256"] = "0" * 64
            parse_path.write_text(
                json.dumps(parse_table, indent=2) + "\n",
                encoding="utf-8",
            )
            self._rehash_artifact(right)
            differences = compare_phase2i_artifacts(
                left, right,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("self-verify" in item for item in differences),
            )

    def test_verifier_enforces_full_acceptance_contract(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertEqual(_verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            ), [])

            aggregate_path = output / "phase2i-syntax-features.json"
            original = aggregate_path.read_bytes()

            def mutate(transform):
                body = json.loads(original.decode("utf-8"))
                transform(body)
                aggregate_path.write_text(
                    json.dumps(body, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._rehash_artifact(output)
                problems = _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )
                aggregate_path.write_bytes(original)
                self._rehash_artifact(output)
                return problems

            cases = [
                (
                    lambda body: body["definition"].__setitem__(
                        "keep_threshold", 0.9,
                    ),
                    "frozen contract",
                ),
                (
                    lambda body: body["definition"].__setitem__("seed", 1),
                    "frozen contract",
                ),
                (
                    lambda body: body["definition"].__setitem__(
                        "cells", ["logistic_C"],
                    ),
                    "frozen contract",
                ),
                (
                    lambda body: body.__setitem__("repository_dirty", True),
                    "clean tree",
                ),
                (
                    lambda body: body.__setitem__(
                        "git_commit", "not-a-git-object",
                    ),
                    "Git object ID",
                ),
                (
                    lambda body: body["assets_provenance"].__setitem__(
                        "verified", False,
                    ),
                    "not verified",
                ),
                (
                    lambda body: body["dataset_summary"].__setitem__(
                        "candidate_count", 1,
                    ),
                    "dataset_summary",
                ),
                (
                    lambda body: body["definition"].__setitem__(
                        "task", "tampered",
                    ),
                    "definition_sha256 invalid",
                ),
            ]
            for transform, needle in cases:
                problems = mutate(transform)
                self.assertTrue(
                    any(needle in item for item in problems),
                    (needle, problems),
                )

            # Manifest extras and traversal paths are rejected.
            manifest_path = output / "MANIFEST.json"
            manifest_original = manifest_path.read_bytes()
            body = json.loads(manifest_original.decode("utf-8"))
            body["files"].append({
                "path": "intruder.json",
                "file_sha256": "0" * 64,
            })
            manifest_path.write_text(
                json.dumps(body, indent=2) + "\n", encoding="utf-8",
            )
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(any("MANIFEST" in item for item in problems))
            manifest_path.write_bytes(manifest_original)

            body = json.loads(manifest_original.decode("utf-8"))
            body["files"].append({
                "path": "../evil.json",
                "file_sha256": "0" * 64,
            })
            manifest_path.write_text(
                json.dumps(body, indent=2) + "\n", encoding="utf-8",
            )
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(any("malformed" in item for item in problems))
            manifest_path.write_bytes(manifest_original)

            body = json.loads(manifest_original.decode("utf-8"))
            body["hidden_payload"] = {"unexpected": True}
            manifest_path.write_text(
                json.dumps(body, indent=2) + "\n", encoding="utf-8",
            )
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("top-level key set" in item for item in problems),
                problems,
            )
            manifest_path.write_bytes(manifest_original)

            body = json.loads(manifest_original.decode("utf-8"))
            body["files"][0]["hidden_payload"] = True
            manifest_path.write_text(
                json.dumps(body, indent=2) + "\n", encoding="utf-8",
            )
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("manifest entry is malformed" in item for item in problems),
                problems,
            )
            manifest_path.write_bytes(manifest_original)

            # A parse table with a wrong window_id but a self-consistent
            # hash is caught by the window_id cross-check.
            parse_path = output / "parser" / (
                "mid-push-prevents-side-collapse.json"
            )
            parse_original = parse_path.read_bytes()
            parse_body = json.loads(parse_original.decode("utf-8"))
            parse_body["window_id"] = "wrong-window"
            parse_body["parse_sha256"] = canonical_sha256({
                key: value for key, value in parse_body.items()
                if key != "parse_sha256"
            })
            parse_path.write_text(
                json.dumps(parse_body, indent=2) + "\n",
                encoding="utf-8",
            )
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("window_id differs" in item for item in problems),
                problems,
            )
            parse_path.write_bytes(parse_original)
            self._rehash_artifact(output)

    def test_manifest_rejects_noncanonical_file_order_after_rehash(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            manifest_path = output / "MANIFEST.json"
            original = manifest_path.read_bytes()
            manifest = json.loads(original.decode("utf-8"))
            manifest["files"] = list(reversed(manifest["files"]))
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            # Repair every digest so ordering is the only remaining defect.
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "canonical path order" in item
                    for item in problems
                ),
                problems,
            )
            manifest_path.write_bytes(original)
            self._rehash_artifact(output)
            self.assertEqual(
                _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )

    def test_aggregate_rejects_nonexistent_hex_git_commit_after_rehash(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            aggregate_path = output / "phase2i-syntax-features.json"
            original = aggregate_path.read_bytes()
            body = json.loads(original.decode("utf-8"))
            # Syntactically valid 40-hex SHA-1 that names no Git object.
            body["git_commit"] = "0" * 40
            aggregate_path.write_text(
                json.dumps(body, indent=2) + "\n",
                encoding="utf-8",
            )
            # Repair every affected hash so only object authentication
            # can reject the artifact.
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "git_commit" in item and "not available" in item
                    for item in problems
                ),
                problems,
            )
            aggregate_path.write_bytes(original)
            self._rehash_artifact(output)
            self.assertEqual(
                _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )

    def test_endpoint_json_rejects_duplicate_keys_and_noncanonical_bytes(
        self,
    ):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            window_id = LOCKED_WINDOW_IDS[0]
            window_path = output / "windows" / f"{window_id}.json"
            parse_path = output / "parser" / f"{window_id}.json"
            window_original = window_path.read_bytes()
            parse_original = parse_path.read_bytes()

            # Duplicate first key in a window table; last value wins, so
            # every repaired lock still matches the original semantics.
            table = json.loads(window_original.decode("utf-8"))
            duplicate = json.dumps(table, indent=2).replace(
                "{\n  \"case_id\"",
                "{\n  \"case_id\": \"fabricated\",\n  \"case_id\"",
                1,
            ) + "\n"
            window_path.write_text(duplicate, encoding="utf-8")
            # Noncanonical bytes in a parser table (compact JSON, no
            # trailing newline) while its parse_sha256 stays valid.
            parse_table = json.loads(parse_original.decode("utf-8"))
            parse_path.write_text(
                json.dumps(parse_table), encoding="utf-8",
            )
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "duplicate JSON object key" in item
                    for item in problems
                ),
                problems,
            )
            self.assertTrue(
                any(
                    "canonical project format" in item
                    for item in problems
                ),
                problems,
            )
            window_path.write_bytes(window_original)
            parse_path.write_bytes(parse_original)
            self._rehash_artifact(output)
            self.assertEqual(
                _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )

    def test_compare_rejects_identically_invalid_artifacts(self):
        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                left, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            publish_phase2i_artifact(
                right, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            for side in (left, right):
                path = side / "phase2i-syntax-features.json"
                body = json.loads(path.read_text(encoding="utf-8"))
                body["metrics"]["logistic_C"]["recall"]["rate"] = 0.123
                path.write_text(
                    json.dumps(body, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._rehash_artifact(side)
            differences = compare_phase2i_artifacts(
                left, right,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(differences)
            self.assertTrue(
                any("verification" in item for item in differences),
                differences,
            )

            # Two identically incomplete, self-rehashed artifacts must not
            # compare as valid merely because they are the same shape.
            stripped_left = Path(tmp) / "stripped-left"
            stripped_right = Path(tmp) / "stripped-right"
            shutil.copytree(left, stripped_left)
            shutil.copytree(right, stripped_right)
            for side in (stripped_left, stripped_right):
                path = side / "phase2i-syntax-features.json"
                body = json.loads(path.read_text(encoding="utf-8"))
                del body["syntax_coefficients"]
                path.write_text(
                    json.dumps(body, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._rehash_artifact(side)
            differences = compare_phase2i_artifacts(
                stripped_left, stripped_right,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(differences)

    def test_verifier_rejects_malformed_structures_without_crashing(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            aggregate_path = output / "phase2i-syntax-features.json"
            original = aggregate_path.read_bytes()

            def mutate_and_verify(transform, needle):
                body = json.loads(original.decode("utf-8"))
                transform(body)
                aggregate_path.write_text(
                    json.dumps(body, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._rehash_artifact(output)
                problems = _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )
                aggregate_path.write_bytes(original)
                self._rehash_artifact(output)
                self.assertTrue(
                    any(needle in item for item in problems),
                    (needle, problems),
                )

            # Missing and non-mapping dependencies must be rejected.
            mutate_and_verify(
                lambda body: body.__setitem__("dependencies", "stanza"),
                "dependencies",
            )
            mutate_and_verify(
                lambda body: body["dependencies"].pop("stanza"),
                "dependencies",
            )
            # A non-mapping window_tables value must fail closed, not crash.
            mutate_and_verify(
                lambda body: body.__setitem__("window_tables", 17),
                "window table lock",
            )
            # Unhashable fold window ids must fail closed, not crash.
            mutate_and_verify(
                lambda body: body["folds"][0].__setitem__(
                    "test_window_id",
                    ["mid-push-prevents-side-collapse"],
                ),
                "not a string",
            )
            # Five duplicate valid-looking folds are not the archived
            # sequence; every frozen test window must appear exactly once.
            mutate_and_verify(
                lambda body: body.__setitem__(
                    "folds",
                    [
                        dict(body["folds"][0], fold_index=index)
                        for index in range(5)
                    ],
                ),
                "archived",
            )
            # Missing required top-level sections are rejected even when the
            # artifact is rehashed.
            mutate_and_verify(
                lambda body: body.pop("parser_diagnostics"),
                "top-level key set",
            )

    def test_semantic_tampering_with_rehashed_hashes_is_rejected(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            aggregate_path = output / "phase2i-syntax-features.json"
            original = aggregate_path.read_bytes()

            cases = [
                (
                    lambda body: body["metrics"]["logistic_C"].__setitem__(
                        "selected",
                        body["metrics"]["logistic_C"]["selected"] + 1,
                    ),
                    "metrics",
                ),
                (
                    lambda body: body["deltas"]["logistic_C"]["delta"].__setitem__(
                        "precision", 123.0,
                    ),
                    "deltas",
                ),
                (
                    lambda body: body["universally_missed"].pop(),
                    "universally_missed",
                ),
                (
                    lambda body: body["error_taxonomy_b_vs_c"][
                        "logistic_C"
                    ]["correct"].__setitem__("phase2i", 999),
                    "error_taxonomy",
                ),
                (
                    lambda body: body["parser_diagnostics"].__setitem__(
                        "objective_parser_error_total", 999,
                    ),
                    "parser_diagnostics",
                ),
                (
                    lambda body: body[
                        "overlap_cluster_syntax_diagnostics"
                    ].__setitem__("intruder-window", {}),
                    "overlap_cluster_syntax_diagnostics",
                ),
                (
                    lambda body: body["syntax_coefficients"][
                        "logistic_C"
                    ].__setitem__("syntax_feature_count", 999),
                    "syntax_coefficients",
                ),
                (
                    lambda body: body["syntax_vs_inherited_importance"][
                        "lightgbm_C"
                    ]["per_fold"]["0"].__setitem__("syntax_share", -1.0),
                    "share differs",
                ),
                (
                    lambda body: body["training_vs_held_out"][
                        "logistic_C"
                    ]["per_fold"]["0"]["held_out"].__setitem__(
                        "predicted_keep_count",
                        body["training_vs_held_out"]["logistic_C"][
                            "per_fold"
                        ]["0"]["held_out"]["predicted_keep_count"] + 1,
                    ),
                    "training_vs_held_out",
                ),
            ]
            for transform, needle in cases:
                body = json.loads(original.decode("utf-8"))
                transform(body)
                aggregate_path.write_text(
                    json.dumps(body, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._rehash_artifact(output)
                problems = _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )
                self.assertTrue(
                    any(needle in item for item in problems),
                    (needle, problems),
                )
                aggregate_path.write_bytes(original)
                self._rehash_artifact(output)
                self.assertEqual(
                    _verify_phase2i_artifact(
                        output,
                        benchmark_path=BENCHMARK,
                        baseline_archive=ARCHIVE,
                    ),
                    [],
                )

    def test_window_table_tampering_with_rehashed_locks_is_rejected(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            window_id = LOCKED_WINDOW_IDS[0]
            table_path = output / "windows" / f"{window_id}.json"
            original_table = table_path.read_bytes()

            def verify_table_tamper(transform, needle):
                table = json.loads(original_table.decode("utf-8"))
                transform(table)
                table_path.write_text(
                    json.dumps(table, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._rehash_artifact(output)
                problems = _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )
                table_path.write_bytes(original_table)
                self._rehash_artifact(output)
                self.assertTrue(
                    any(needle in item for item in problems),
                    (needle, problems),
                )

            # A score change that breaks the frozen rank/selection semantics
            # is caught even after all outer locks are recomputed.
            verify_table_tamper(
                lambda table: table["candidates"][0]["predictions"][
                    "logistic_C"
                ].__setitem__(
                    "score",
                    0.0 if table["candidates"][0]["predictions"][
                        "logistic_C"
                    ]["selected"] == KEEP else 1.0,
                ),
                "contract violated",
            )
            # Exact offsets are validated against the frozen benchmark.
            verify_table_tamper(
                lambda table: table["candidates"][0].__setitem__(
                    "start", table["candidates"][0]["start"] + 1,
                ),
                "frozen benchmark",
            )
            # Catalog/generator metadata is validated against the frozen
            # benchmark.
            verify_table_tamper(
                lambda table: table.__setitem__(
                    "candidate_generator_version", "tampered",
                ),
                "frozen benchmark",
            )
            # A baseline B prediction that differs from the archive is
            # rejected even when the artifact rehashes cleanly.
            verify_table_tamper(
                lambda table: table["candidates"][0]["predictions"][
                    "logistic_B"
                ].__setitem__("score", 0.123),
                "baseline predictions differ from the archive",
            )
            # Gold metadata must remain consistent with the KEEP label.
            verify_table_tamper(
                lambda table: table["candidates"][0].__setitem__(
                    "gold_mention_ids", ["tampered-mention"],
                ),
                "gold metadata",
            )

    def test_syntax_field_tampering_with_rehashed_locks_is_rejected(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            window_id = LOCKED_WINDOW_IDS[0]
            table_path = output / "windows" / f"{window_id}.json"
            original_table = table_path.read_bytes()

            def verify_syntax_tamper(transform, field):
                table = json.loads(original_table.decode("utf-8"))
                syntax = table["candidates"][0]["syntax"]
                transform(syntax)
                table_path.write_text(
                    json.dumps(table, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._rehash_artifact(output)
                problems = _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )
                table_path.write_bytes(original_table)
                self._rehash_artifact(output)
                self.assertTrue(
                    any(
                        f"syntax field {field}" in item
                        for item in problems
                    ),
                    (field, problems),
                )

            # Representative fields that were previously only reachable
            # through the stored evidence hash are now compared directly to
            # recomputed parser/candidate evidence, field by field.
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "boundary_status", "tampered",
                ),
                "boundary_status",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "start_aligned", not syntax["start_aligned"],
                ),
                "start_aligned",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "token_ids", ["s1:t999"],
                ),
                "token_ids",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "crossing_arcs", syntax["crossing_arcs"] + 1,
                ),
                "crossing_arcs",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "finite_verbs", syntax["finite_verbs"] + ["s1:w999"],
                ),
                "finite_verbs",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "modals", syntax["modals"] + ["s1:w999"],
                ),
                "modals",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "predicate_internal_oblique",
                    not syntax["predicate_internal_oblique"],
                ),
                "predicate_internal_oblique",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "predicate_external_oblique",
                    not syntax["predicate_external_oblique"],
                ),
                "predicate_external_oblique",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "negations", syntax["negations"] + ["s1:w999"],
                ),
                "negations",
            )
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "clause_roots", syntax["clause_roots"] + ["s1:w999"],
                ),
                "clause_roots",
            )
            # A sub-tolerance mutation of a derived float field was
            # previously tolerated by the tolerant JSON comparator; the
            # canonical recursive projection must reject it exactly.
            verify_syntax_tamper(
                lambda syntax: syntax.__setitem__(
                    "subtree_word_fraction",
                    syntax["subtree_word_fraction"] + 1e-13,
                ),
                "subtree_word_fraction",
            )

    def test_fit_scope_reconstruction_rejects_tampering(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            aggregate_path = output / "phase2i-syntax-features.json"
            original = aggregate_path.read_bytes()

            def mutate_and_verify(transform, needle):
                body = json.loads(original.decode("utf-8"))
                transform(body)
                aggregate_path.write_text(
                    json.dumps(body, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._rehash_artifact(output)
                problems = _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )
                aggregate_path.write_bytes(original)
                self._rehash_artifact(output)
                self.assertTrue(
                    any(needle in item for item in problems),
                    (needle, problems),
                )

            mutate_and_verify(
                lambda body: body["fit_scope"]["logistic_C"]["0"][
                    "scaler"
                ].__setitem__("mean_sha256", "0" * 64),
                "scaler differs from train-only recomputation",
            )
            mutate_and_verify(
                lambda body: body["fit_scope"]["logistic_C"]["0"][
                    "vectorizer"
                ].__setitem__("vocabulary_size", 999),
                "vectorizer differs from train-only recomputation",
            )
            mutate_and_verify(
                lambda body: body["fit_scope"]["logistic_C"]["0"][
                    "syntax_encoder"
                ]["vocabulary_sizes"].__setitem__("head_lemma", 999),
                "syntax_encoder differs from train-only recomputation",
            )
            mutate_and_verify(
                lambda body: body["fit_scope"]["logistic_C"]["0"].__setitem__(
                    "feature_names_sha256", "0" * 64,
                ),
                "feature_names_sha256 differs from train-only recomputation",
            )

            def tamper_vectorizer_oov(body):
                scope = body["fit_scope"]["logistic_C"]["0"]["vectorizer"]
                scope["test_oov_token_types"] = (
                    scope["test_oov_token_types"] + ["tampered-token"]
                )
                scope["test_oov_token_type_count"] = len(
                    scope["test_oov_token_types"],
                )
                scope["test_oov_token_types_sha256"] = canonical_sha256(
                    scope["test_oov_token_types"],
                )
            mutate_and_verify(
                tamper_vectorizer_oov,
                "vectorizer differs from train-only recomputation",
            )

            def tamper_syntax_oov(body):
                scope = body["fit_scope"]["logistic_C"]["0"][
                    "syntax_encoder"
                ]
                scope["per_group"]["head_lemma"] = (
                    scope["per_group"]["head_lemma"] + ["tampered-value"]
                )
                scope["oov_value_count"] = sum(
                    len(values) for values in scope["per_group"].values()
                )
                scope["oov_sha256"] = canonical_sha256(scope["per_group"])
            mutate_and_verify(
                tamper_syntax_oov,
                "syntax_encoder differs from train-only recomputation",
            )

    def test_coordinated_train_diagnostics_and_held_out_tampering_rejected(
        self,
    ):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            aggregate_path = output / "phase2i-syntax-features.json"
            original = aggregate_path.read_bytes()

            body = json.loads(original.decode("utf-8"))
            for cell in CELLS_C:
                for fold_key, scope in body["fit_scope"][cell].items():
                    diagnostics = scope["train_diagnostics"]
                    diagnostics["average_precision"] = 0.123456789
                    diagnostics["roc_auc"] = 0.987654321
                    diagnostics["predicted_keep_count"] = (
                        0 if diagnostics["predicted_keep_count"] != 0 else 1
                    )
                    mirrored = body["training_vs_held_out"][cell][
                        "per_fold"
                    ][fold_key]["train"]
                    mirrored["average_precision"] = diagnostics[
                        "average_precision"
                    ]
                    mirrored["roc_auc"] = diagnostics["roc_auc"]
                    mirrored["predicted_keep_count"] = diagnostics[
                        "predicted_keep_count"
                    ]
            aggregate_path.write_text(
                json.dumps(body, indent=2) + "\n",
                encoding="utf-8",
            )
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "train diagnostics" in item
                    or "training_vs_held_out" in item
                    for item in problems
                ),
                problems,
            )

            aggregate_path.write_bytes(original)
            self._rehash_artifact(output)
            self.assertEqual(
                _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )

    def test_rank_preserving_held_out_score_tampering_is_rejected(self):
        """Persisted probabilities must come from the frozen fold refit.

        This adversarial mutation preserves candidate order, threshold
        selection, metrics, diagnostics, and every outer hash.  Only direct
        candidate-by-candidate score provenance can detect it.
        """
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )

            mutated_ids = {}
            for cell in CELLS_C:
                changed = False
                for window_id in sorted(self.result["dataset"]["windows"]):
                    table_path = output / "windows" / f"{window_id}.json"
                    table = json.loads(table_path.read_text(encoding="utf-8"))
                    ordered = sorted(
                        table["candidates"],
                        key=lambda candidate: (
                            -candidate["predictions"][cell]["score"],
                            candidate["candidate_id"],
                        ),
                    )
                    for index in range(1, len(ordered) - 1):
                        previous_score = ordered[index - 1]["predictions"][
                            cell
                        ]["score"]
                        prediction = ordered[index]["predictions"][cell]
                        score = prediction["score"]
                        next_score = ordered[index + 1]["predictions"][cell][
                            "score"
                        ]
                        mutated = math.nextafter(score, 1.0)
                        if not (
                            previous_score > mutated > next_score
                            and (mutated >= KEEP_THRESHOLD)
                            == (score >= KEEP_THRESHOLD)
                        ):
                            continue
                        prediction["score"] = mutated
                        mutated_ids[cell] = ordered[index]["candidate_id"]
                        table_path.write_text(
                            json.dumps(table, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        changed = True
                        break
                    if changed:
                        break
                self.assertTrue(changed, f"no safe score mutation for {cell}")

            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            for cell, candidate_id in mutated_ids.items():
                self.assertTrue(
                    any(
                        candidate_id in problem
                        and cell in problem
                        and "held-out score differs" in problem
                        for problem in problems
                    ),
                    problems,
                )

    def test_coherent_explainability_fabrication_is_rejected(self):
        """Self-consistent summaries must still match the refitted models."""
        from pipeline.phase2i_endpoint_scoring import (
            _aggregate_top,
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            aggregate_path = output / "phase2i-syntax-features.json"
            body = json.loads(aggregate_path.read_text(encoding="utf-8"))

            coefficient_entry = body["syntax_coefficients"]["logistic_C"]
            positive = []
            negative = []
            coefficient_names = set()
            for coefficients in coefficient_entry["per_fold"].values():
                for name in list(coefficients):
                    coefficients[name] *= 2.0
                    coefficient_names.add(name)
                    target = positive if coefficients[name] >= 0 else negative
                    target.append((name, coefficients[name]))
            coefficient_entry["aggregate_top_positive"] = _aggregate_top(
                positive, 15,
            )
            coefficient_entry["aggregate_top_negative"] = _aggregate_top(
                negative, 15, ascending=True,
            )
            coefficient_entry["syntax_feature_count"] = len(
                coefficient_names,
            )

            importance_entry = body["syntax_vs_inherited_importance"][
                "lightgbm_C"
            ]
            syntax_items = []
            inherited_items = []
            from pipeline.phase2i_syntax import DENSE_C_EXTRA_FEATURES
            for fold_data in importance_entry["per_fold"].values():
                for item in fold_data["importances"]:
                    item[1] *= 2.0
                    target = (
                        syntax_items
                        if item[0].startswith("syntax:")
                        or item[0] in DENSE_C_EXTRA_FEATURES
                        else inherited_items
                    )
                    target.append((item[0], item[1]))
                fold_data["top_importances"] = fold_data["importances"][:20]
                fold_data["syntax_gain"] *= 2.0
                fold_data["inherited_gain"] *= 2.0
            importance_entry["aggregate_syntax_top"] = _aggregate_top(
                syntax_items, 15,
            )
            importance_entry["aggregate_inherited_top"] = _aggregate_top(
                inherited_items, 10,
            )

            aggregate_path.write_text(
                json.dumps(body, indent=2) + "\n", encoding="utf-8",
            )
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "syntax_coefficients differ from the training-only"
                    in problem
                    for problem in problems
                ),
                problems,
            )
            self.assertTrue(
                any(
                    "syntax_vs_inherited_importance differs from the "
                    "training-only" in problem
                    for problem in problems
                ),
                problems,
            )

    def test_artifact_root_symlinks_and_dependency_fabrication_rejected(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )

            alias = root / "run-alias"
            alias.symlink_to(output, target_is_directory=True)
            problems = _verify_phase2i_artifact(
                alias,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("symlinked ancestor or root" in item for item in problems),
                problems,
            )

            hidden_parent = root / "run-alias" / ".." / "run"
            problems = _verify_phase2i_artifact(
                hidden_parent,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("parent-directory traversal" in item for item in problems),
                problems,
            )

            aggregate_path = output / "phase2i-syntax-features.json"
            body = json.loads(aggregate_path.read_text(encoding="utf-8"))
            body["dependencies"]["numpy"] = "fabricated"
            aggregate_path.write_text(
                json.dumps(body, indent=2) + "\n", encoding="utf-8",
            )
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("dependency versions differ" in item for item in problems),
                problems,
            )

    def test_self_hashed_parser_fabrication_differs_from_stanza_replay(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            window_id = "mid-push-prevents-side-collapse"
            parse_path = output / "parser" / f"{window_id}.json"
            parse_table = json.loads(parse_path.read_text(encoding="utf-8"))
            word = parse_table["sentences"][0]["words"][0]
            word["deps"] = "999:obl"
            parse_table["parse_sha256"] = canonical_sha256({
                key: value for key, value in parse_table.items()
                if key != "parse_sha256"
            })
            parse_path.write_text(
                json.dumps(parse_table, indent=2) + "\n", encoding="utf-8",
            )
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("differs from locked Stanza replay" in item for item in problems),
                problems,
            )

    def test_verification_never_reuses_success_after_tree_mutation(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertEqual(
                _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )

            aggregate_path = output / "phase2i-syntax-features.json"
            aggregate_original = aggregate_path.read_bytes()
            manifest_path = output / "MANIFEST.json"
            manifest_original = manifest_path.read_bytes()

            # Byte-only formatting changes still invalidate the exact
            # manifest; a prior successful verification must never mask it.
            aggregate_body = json.loads(
                aggregate_original.decode("utf-8"),
            )
            aggregate_path.write_text(
                json.dumps(aggregate_body, separators=(",", ":")),
                encoding="utf-8",
            )
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(any("sha256" in item for item in problems))
            aggregate_path.write_bytes(aggregate_original)

            # Fields validated by the aggregate header cannot be excluded
            # from a process-wide success decision.
            aggregate_body = json.loads(
                aggregate_original.decode("utf-8"),
            )
            aggregate_body["created_at"] = ""
            aggregate_body["content_sha256"] = "0" * 64
            aggregate_path.write_text(
                json.dumps(
                    aggregate_body, indent=2, ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            manifest_body = json.loads(
                manifest_original.decode("utf-8"),
            )
            for entry in manifest_body["files"]:
                if entry["path"] == "phase2i-syntax-features.json":
                    entry["file_sha256"] = hashlib.sha256(
                        aggregate_path.read_bytes(),
                    ).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest_body, indent=2, ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "content_sha256" in item or "created_at" in item
                    for item in problems
                ),
                problems,
            )
            aggregate_path.write_bytes(aggregate_original)
            manifest_path.write_bytes(manifest_original)

            # Non-file tree entries are also part of every fresh manifest
            # validation and cannot be hidden by an earlier green result.
            unlisted_link = output / "unlisted-directory-link"
            unlisted_link.symlink_to(
                output / "windows", target_is_directory=True,
            )
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any("symlink" in item for item in problems),
                problems,
            )

    def test_malformed_train_diagnostics_reject_without_crashing(self):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            aggregate_path = output / "phase2i-syntax-features.json"
            original = aggregate_path.read_bytes()

            cases = (
                (
                    "average_precision", float("nan"),
                    "non-finite JSON number", False,
                ),
                (
                    "average_precision", float("inf"),
                    "non-finite JSON number", False,
                ),
                ("roc_auc", 1.5, "not a finite float", True),
                (
                    "average_precision", "not-a-float",
                    "not a finite float", True,
                ),
                ("predicted_keep_count", "many", "legal range", True),
            )
            for field, value, needle, require_diagnostic_context in cases:
                with self.subTest(field=field, value=value):
                    body = json.loads(original.decode("utf-8"))
                    for cell in CELLS_C:
                        for scope in body["fit_scope"][cell].values():
                            scope["train_diagnostics"][field] = value
                    aggregate_path.write_text(
                        json.dumps(body, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    self._rehash_artifact(output)
                    problems = _verify_phase2i_artifact(
                        output,
                        benchmark_path=BENCHMARK,
                        baseline_archive=ARCHIVE,
                    )
                    aggregate_path.write_bytes(original)
                    self._rehash_artifact(output)
                    self.assertTrue(
                        any(
                            needle in item
                            and (
                                not require_diagnostic_context
                                or "train diagnostics" in item
                            )
                            for item in problems
                        ),
                        (field, value, problems),
                    )

            self.assertEqual(
                _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )

    def test_sub_tolerance_archived_mutations_are_rejected_after_rehashing(
        self,
    ):
        from pipeline.phase2i_endpoint_scoring import (
            _verify_phase2i_artifact,
        )

        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            aggregate = self._aggregate(window_table_hashes=hashes)
            publish_phase2i_artifact(
                output, aggregate, tables, parse_tables,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            window_id = LOCKED_WINDOW_IDS[0]
            table_path = output / "windows" / f"{window_id}.json"
            original_table = table_path.read_bytes()

            table = json.loads(original_table.decode("utf-8"))
            prediction = table["candidates"][0]["predictions"][
                "logistic_B"
            ]
            prediction["score"] = prediction["score"] + 1e-13
            table_path.write_text(
                json.dumps(table, indent=2) + "\n",
                encoding="utf-8",
            )
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "baseline predictions differ from the archive" in item
                    for item in problems
                ),
                problems,
            )
            table_path.write_bytes(original_table)
            self._rehash_artifact(output)

            aggregate_path = output / "phase2i-syntax-features.json"
            original_aggregate = aggregate_path.read_bytes()
            body = json.loads(original_aggregate.decode("utf-8"))
            body["baseline_metrics"]["logistic_B"]["precision"]["rate"] += (
                1e-13
            )
            aggregate_path.write_text(
                json.dumps(body, indent=2) + "\n",
                encoding="utf-8",
            )
            self._rehash_artifact(output)
            problems = _verify_phase2i_artifact(
                output,
                benchmark_path=BENCHMARK,
                baseline_archive=ARCHIVE,
            )
            self.assertTrue(
                any(
                    "differ from archived Phase 2H" in item
                    for item in problems
                ),
                problems,
            )
            aggregate_path.write_bytes(original_aggregate)
            self._rehash_artifact(output)
            self.assertEqual(
                _verify_phase2i_artifact(
                    output,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                ),
                [],
            )

    def test_publication_self_verifies_before_atomic_replace(self):
        tables, hashes, parse_tables = self._tables_and_hashes(self.result)
        with tempfile.TemporaryDirectory() as tmp:
            aggregate = self._aggregate(window_table_hashes=hashes)
            aggregate = json.loads(json.dumps(aggregate))
            aggregate["metrics"]["lightgbm_C"]["selected"] += 1
            inner = {
                key: value for key, value in aggregate.items()
                if key != "content_sha256"
            }
            aggregate["content_sha256"] = canonical_sha256(inner)
            output = Path(tmp) / "phase2i-run"
            with self.assertRaises(Phase2IError):
                publish_phase2i_artifact(
                    output, aggregate, tables, parse_tables,
                    benchmark_path=BENCHMARK,
                    baseline_archive=ARCHIVE,
                )
            self.assertFalse(output.exists())
            self.assertEqual(
                list(Path(tmp).glob("phase2i-run.tmp-*")), [],
            )

    def test_archive_extraction_rejects_non_regular_and_escaping_members(self):
        from pipeline.phase2i_endpoint_scoring import Phase2IBaselineError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                (tarfile.SYMTYPE, "symlink", "target"),
                (tarfile.LNKTYPE, "hardlink", "payload.txt"),
                (tarfile.FIFOTYPE, "fifo", ""),
                (tarfile.CHRTYPE, "chardev", ""),
                (tarfile.BLKTYPE, "blockdev", ""),
            ]
            for member_type, name, linkname in cases:
                with self.subTest(member_type=member_type):
                    archive = root / f"{name}.tar.gz"
                    with tarfile.open(archive, "w:gz") as handle:
                        payload = tarfile.TarInfo("payload.txt")
                        payload.size = 1
                        payload_bytes = b"x"
                        handle.addfile(payload, io.BytesIO(payload_bytes))
                        member = tarfile.TarInfo(name)
                        member.type = member_type
                        if member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                            member.linkname = linkname
                        if member_type in (tarfile.CHRTYPE, tarfile.BLKTYPE):
                            member.devmajor = 1
                            member.devminor = 3
                        handle.addfile(member)
                    destination = root / name
                    destination.mkdir()
                    with self.assertRaises(Phase2IBaselineError) as caught:
                        _extract_tar(archive, destination)
                    self.assertIn("unsafe member", str(caught.exception))

            archive = root / "traversal.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                member = tarfile.TarInfo("../escape.txt")
                member.size = 1
                handle.addfile(member, io.BytesIO(b"x"))
            destination = root / "traversal"
            destination.mkdir()
            with self.assertRaises(Phase2IBaselineError) as caught:
                _extract_tar(archive, destination)
            self.assertIn("unsafe member", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
