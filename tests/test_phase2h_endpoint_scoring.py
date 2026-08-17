import hashlib
import json
from pathlib import Path
import base64
import pickle
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from pipeline.phase2h_endpoint_scoring import (
    CELLS,
    DENSE_A_FEATURES,
    DENSE_B_EXTRA_FEATURES,
    DROP,
    ERROR_CODES,
    ERROR_PRECEDENCE,
    FEATURE_SCHEMA_VERSION,
    GOLD_RANK_HIGH_K,
    KEEP,
    KEEP_THRESHOLD,
    LOGISTIC_CONFIG,
    VECTORIZER_PARAMS,
    RUN_VERSION,
    SEED,
    CandidateRow,
    CellPreprocessor,
    Phase2HError,
    Phase2HCoverageError,
    Phase2HCVError,
    balanced_class_weights,
    build_aggregate,
    build_dataset,
    build_window_table,
    canonical_sha256,
    classify_all_errors,
    classify_candidate_error,
    compare_artifacts,
    compute_cell_metrics,
    compute_rankings,
    error_taxonomy_counts,
    extract_dense_features,
    extract_sparse_inputs,
    feature_schema,
    load_benchmark,
    publish_artifact,
    run_cv,
    run_experiment,
    strongest_features,
    validate_benchmark_mapping,
    validate_dataset,
    _boundary_sparse,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"


def _row(
    case_id,
    window_id,
    index,
    start,
    end,
    text,
    *,
    label=DROP,
    hints=("ENTITY",),
    mention_ids=(),
    node_types=(),
):
    return CandidateRow(
        case_id=case_id,
        window_id=window_id,
        candidate_id=f"{window_id}:m{index:04d}",
        alias=f"C{index:04d}",
        start=start,
        end=end,
        absolute_start=start,
        absolute_end=end,
        text=text,
        segment_ids=(f"{window_id}:s001",),
        segment_bounds=((0, max(end, 1)),),
        type_hints=tuple(hints),
        source_kind="transcript",
        is_gold_positive=label == KEEP,
        label=label,
        excluded=False,
        ambiguity_state="NONE",
        gold_mention_ids=tuple(mention_ids),
        gold_node_types=tuple(node_types),
    )


def _window(case_id, text, rows):
    return {
        "case_id": case_id,
        "window_id": f"{case_id}:w0001",
        "bronze_text": text,
        "bronze_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "catalog_sha256": "0" * 64,
        "candidate_generator_version": "synthetic-test",
        "gold_spans": tuple(
            sorted(
                (row.start, row.end)
                for row in rows if row.is_gold_positive
            ),
        ),
        "rows": tuple(rows),
    }


def _simple_dataset():
    """Two windows, each with one gold positive and four negatives."""
    window_a_text = "alpha beta gamma delta epsilon"
    window_a_rows = (
        _row("winA", "winA:w0001", 1, 0, 5, "alpha"),
        _row(
            "winA", "winA:w0001", 2, 6, 10, "beta",
            label=KEEP, hints=("ACTION", "ENTITY"),
            mention_ids=("m1",), node_types=("ENTITY",),
        ),
        _row("winA", "winA:w0001", 3, 11, 16, "gamma"),
        _row("winA", "winA:w0001", 4, 17, 22, "delta"),
        _row("winA", "winA:w0001", 5, 23, 30, "epsilon"),
    )
    window_b_text = "zeta eta theta iota kappa"
    window_b_rows = (
        _row(
            "winB", "winB:w0001", 1, 0, 4, "zeta",
            label=KEEP, hints=("TIME",),
            mention_ids=("m1",), node_types=("TIME",),
        ),
        _row("winB", "winB:w0001", 2, 5, 8, "eta"),
        _row("winB", "winB:w0001", 3, 9, 14, "theta"),
        _row("winB", "winB:w0001", 4, 15, 19, "iota"),
        _row("winB", "winB:w0001", 5, 20, 25, "kappa"),
    )
    dataset = {
        "windows": {
            "winA": _window("winA", window_a_text, window_a_rows),
            "winB": _window("winB", window_b_text, window_b_rows),
        },
    }
    validate_dataset(dataset, expected_positive_count=2)
    return dataset


def _rankings_from_scores(dataset, scores_by_window, cell="logistic_A"):
    rankings = {}
    for window_id in sorted(dataset["windows"]):
        rows = dataset["windows"][window_id]["rows"]
        scores = scores_by_window[window_id]
        order = sorted(
            range(len(rows)),
            key=lambda index: (-scores[index], index),
        )
        cell_rankings = {}
        for rank, index in enumerate(order, 1):
            row = rows[index]
            cell_rankings[row.candidate_id] = {
                "score": scores[index],
                "rank": rank,
                "selected": KEEP if scores[index] >= KEEP_THRESHOLD else DROP,
            }
        rankings[window_id] = {cell: cell_rankings}
    return rankings


def _cv_digest(dataset, cells):
    cv = run_cv(dataset, cells=cells)
    payload = {
        "folds": cv["folds"],
        "fit_scope": cv["fit_scope"],
        "oof_scores": cv["oof_scores"],
    }
    return canonical_sha256(payload)


class Phase2HDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_benchmark(BENCHMARK)
        cls.dataset = build_dataset(cls.benchmark)
        cls.summary = validate_dataset(cls.dataset)

    def test_benchmark_coverage_is_33_of_33(self):
        self.assertEqual(self.summary["window_count"], 5)
        self.assertEqual(self.summary["positive_count"], 33)
        self.assertEqual(self.summary["candidate_count"], 16624)
        expected = {
            "wave-reset-after-kill": (3344, 5),
            "push-poke-wave-crash": (3248, 5),
            "sweeper-limits-mid-play": (3344, 7),
            "mid-push-prevents-side-collapse": (3344, 5),
            "unwarded-bush-hook-risk": (3344, 11),
        }
        for case_id, (candidates, positives) in expected.items():
            window = self.dataset["windows"][case_id]
            self.assertEqual(len(window["rows"]), candidates)
            self.assertEqual(
                sum(row.is_gold_positive for row in window["rows"]),
                positives,
            )

    def test_positive_labels_match_exact_gold_spans(self):
        for case_id, window in self.dataset["windows"].items():
            gold_spans = set(window["gold_spans"])
            for row in window["rows"]:
                self.assertEqual(
                    row.label,
                    KEEP if (row.start, row.end) in gold_spans else DROP,
                )
                self.assertEqual(
                    row.is_gold_positive,
                    (row.start, row.end) in gold_spans,
                )

    def test_negatives_are_drop_and_exclusion_state_is_explicit(self):
        for window in self.dataset["windows"].values():
            for row in window["rows"]:
                self.assertIn(row.ambiguity_state, ("NONE", "AMBIGUOUS"))
                self.assertIsInstance(row.excluded, bool)
                self.assertEqual(row.excluded, False)
                if not row.is_gold_positive:
                    self.assertEqual(row.label, DROP)
                    self.assertEqual(row.gold_mention_ids, ())

    def test_provenance_is_retained_without_identifiers_in_features(self):
        for window in self.dataset["windows"].values():
            for row in window["rows"]:
                self.assertTrue(row.candidate_id.startswith(
                    window["window_id"] + ":m",
                ))
                self.assertEqual(
                    window["bronze_text"][row.start:row.end], row.text,
                )
                self.assertTrue(row.segment_ids)
                self.assertTrue(row.type_hints)
                if row.is_gold_positive:
                    self.assertTrue(row.gold_mention_ids)
        dense_a, dense_b = extract_dense_features(
            self.dataset, sorted(self.dataset["windows"]),
        )
        self.assertTrue(np.issubdtype(dense_a.dtype, np.floating))
        self.assertTrue(np.issubdtype(dense_b.dtype, np.floating))
        forbidden = ("id", "case", "window", "mention", "sha", "candidate")
        for name in DENSE_A_FEATURES + DENSE_B_EXTRA_FEATURES:
            lowered_tokens = re.split(r"[^a-z0-9]+", name.lower())
            self.assertFalse(
                any(part in lowered_tokens for part in forbidden),
                f"feature {name!r} looks like an identifier",
            )
        texts, boundaries = extract_sparse_inputs(
            self.dataset, sorted(self.dataset["windows"]),
        )
        self.assertEqual(len(texts), self.summary["candidate_count"])

    def test_same_span_multiple_mentions_produce_one_row(self):
        text = "abcdefghij"
        rows = (
            _row(
                "win", "win:w0001", 1, 2, 5, "cde",
                label=KEEP, mention_ids=("m1", "m2"),
                node_types=("ENTITY", "ACTION"),
            ),
            _row("win", "win:w0001", 2, 6, 8, "gh"),
        )
        dataset = {
            "windows": {
                "win": _window("win", text, rows),
            },
        }
        summary = validate_dataset(dataset, expected_positive_count=1)
        self.assertEqual(summary["positive_count"], 1)
        positive = next(
            row for row in rows if row.is_gold_positive
        )
        self.assertEqual(positive.gold_mention_ids, ("m1", "m2"))
        self.assertEqual(positive.gold_node_types, ("ENTITY", "ACTION"))
        self.assertEqual(len(rows), 2)

    def test_coverage_validation_rejects_missing_gold_rows(self):
        text = "abcdefghij"
        rows = (
            _row(
                "win", "win:w0001", 1, 2, 5, "cde",
                label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
            ),
        )
        dataset = {
            "windows": {
                "win": {
                    **_window("win", text, rows),
                    "gold_spans": ((2, 5), (7, 9)),
                },
            },
        }
        with self.assertRaises(Phase2HCoverageError):
            validate_dataset(dataset, expected_positive_count=2)


class Phase2HFeatureTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _simple_dataset()

    def test_feature_schema_is_versioned_and_complete(self):
        schema = feature_schema()
        self.assertEqual(schema["version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(
            schema["feature_set_A"]["dense_features"],
            list(DENSE_A_FEATURES),
        )
        self.assertEqual(
            schema["feature_set_B"]["dense_extras"],
            list(DENSE_B_EXTRA_FEATURES),
        )
        self.assertEqual(schema["fit_scope"], "training windows only")
        self.assertIn("label-derived", " ".join(schema["prohibited_features"]))

    def test_feature_extraction_is_deterministic_and_missing_free(self):
        window_ids = sorted(self.dataset["windows"])
        first_a, first_b = extract_dense_features(self.dataset, window_ids)
        second_a, second_b = extract_dense_features(self.dataset, window_ids)
        np.testing.assert_array_equal(first_a, second_a)
        np.testing.assert_array_equal(first_b, second_b)
        self.assertEqual(first_a.shape[1], len(DENSE_A_FEATURES))
        self.assertEqual(first_b.shape[1], len(DENSE_B_EXTRA_FEATURES))
        self.assertFalse(np.isnan(first_a).any())
        self.assertFalse(np.isnan(first_b).any())
        self.assertFalse(np.isinf(first_a).any())

    def test_preprocessor_fit_is_isolated_to_training_rows(self):
        # Window "held" contains a token that appears only there.
        window_ids = ["trainA", "trainB", "held"]
        rows_a = (
            _row("trainA", "trainA:w0001", 1, 0, 5, "alpha"),
            _row(
                "trainA", "trainA:w0001", 2, 6, 16, "alphatrain",
                label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row("trainA", "trainA:w0001", 3, 17, 27, "alphatrain"),
        )
        rows_b = (
            _row("trainB", "trainB:w0001", 1, 0, 4, "beta"),
            _row(
                "trainB", "trainB:w0001", 2, 5, 14, "betatrain",
                label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row("trainB", "trainB:w0001", 3, 15, 24, "betatrain"),
        )
        rows_held = (
            _row("held", "held:w0001", 1, 0, 11, "betaholdout"),
            _row(
                "held", "held:w0001", 2, 12, 23, "betaholdout",
                label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row("held", "held:w0001", 3, 24, 35, "betaholdout"),
        )
        dataset = {
            "windows": {
                "trainA": _window("trainA", "alpha alphatrain alphatrain", rows_a),
                "trainB": _window("trainB", "beta betatrain betatrain", rows_b),
                "held": _window("held", "betaholdout betaholdout betaholdout", rows_held),
            },
        }
        validate_dataset(dataset, expected_positive_count=3)
        dense_a, dense_b_extra = extract_dense_features(dataset, window_ids)
        dense_b = np.hstack([dense_a, dense_b_extra])
        texts, boundaries = extract_sparse_inputs(dataset, window_ids)
        train_indices = [0, 1, 2, 3, 4, 5]
        pre = CellPreprocessor("B").fit(
            dense_b[train_indices],
            [texts[i] for i in train_indices],
            [boundaries[i] for i in train_indices],
        )
        self.assertNotIn("betaholdout", set(pre.vocab_terms))
        self.assertIn("alphatrain", set(pre.vocab_terms))
        # Scaler statistics come from training rows only.
        expected_mean = dense_b[train_indices].mean(axis=0)
        np.testing.assert_allclose(pre.scaler.mean_, expected_mean, rtol=1e-12)
        # The held-out-only token must not survive into the transformed matrix.
        transformed = pre.transform(
            dense_b, texts, boundaries,
        )
        names = pre.feature_names(
            list(DENSE_A_FEATURES) + list(DENSE_B_EXTRA_FEATURES),
        )
        self.assertFalse(
            any("betaholdout" in name for name in names),
        )
        self.assertEqual(transformed.shape[0], 9)


class Phase2HCVTests(unittest.TestCase):
    def test_grouped_folds_never_leak_the_test_window(self):
        dataset = _simple_dataset()
        cv = run_cv(dataset, cells=("logistic_A",))
        self.assertEqual(len(cv["folds"]), 2)
        held_out = set()
        for fold in cv["folds"]:
            self.assertNotIn(fold["test_window_id"], fold["train_window_ids"])
            self.assertEqual(len(fold["train_window_ids"]), 1)
            held_out.add(fold["test_window_id"])
        self.assertEqual(held_out, {"winA", "winB"})
        self.assertTrue(all(
            fold["train_positive_count"] >= 1 for fold in cv["folds"]
        ))

    def test_all_four_cells_share_identical_folds_and_fit_scope(self):
        dataset = _simple_dataset()
        cv = run_cv(dataset)
        fold_signature = [
            (fold["train_window_ids"], fold["test_window_id"])
            for fold in cv["folds"]
        ]
        for cell in CELLS:
            self.assertEqual(sorted(cv["fit_scope"][cell]), [0, 1])
            for fold_index in (0, 1):
                record = cv["fit_scope"][cell][fold_index]
                self.assertEqual(
                    (
                        record["train_window_ids"],
                        record["test_window_id"],
                    ),
                    fold_signature[fold_index],
                )

    def test_cv_is_deterministic_across_reruns(self):
        dataset = _simple_dataset()
        first = run_cv(dataset, cells=("logistic_A",))
        second = run_cv(dataset, cells=("logistic_A",))
        self.assertEqual(first["oof_scores"], second["oof_scores"])
        self.assertEqual(first["fit_scope"], second["fit_scope"])

    def test_fold_without_positive_training_examples_fails_clearly(self):
        no_positive = (
            _row("winA", "winA:w0001", 1, 0, 5, "alpha"),
            _row("winA", "winA:w0001", 2, 6, 10, "beta"),
        )
        with_positive = (
            _row(
                "winB", "winB:w0001", 1, 0, 4, "zeta",
                label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row("winB", "winB:w0001", 2, 5, 8, "eta"),
        )
        dataset = {
            "windows": {
                "winA": _window("winA", "alpha beta", no_positive),
                "winB": _window("winB", "zeta eta", with_positive),
            },
        }
        validate_dataset(dataset, expected_positive_count=1)
        with self.assertRaises(Phase2HCVError) as caught:
            run_cv(dataset, cells=("logistic_A",))
        self.assertIn("no positive training examples", str(caught.exception))

    def test_fold_without_negative_training_examples_fails_clearly(self):
        all_positive = (
            _row(
                "winA", "winA:w0001", 1, 0, 5, "alpha",
                label=KEEP, hints=("ENTITY",),
                mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row(
                "winA", "winA:w0001", 2, 6, 10, "beta",
                label=KEEP, hints=("ENTITY",),
                mention_ids=("m2",), node_types=("ENTITY",),
            ),
        )
        mixed = (
            _row(
                "winB", "winB:w0001", 1, 0, 4, "zeta",
                label=KEEP, hints=("ENTITY",),
                mention_ids=("m3",), node_types=("ENTITY",),
            ),
            _row("winB", "winB:w0001", 2, 5, 8, "eta"),
        )
        dataset = {
            "windows": {
                "winA": _window("winA", "alpha beta", all_positive),
                "winB": _window("winB", "zeta eta", mixed),
            },
        }
        validate_dataset(dataset, expected_positive_count=3)
        with self.assertRaises(Phase2HCVError) as caught:
            run_cv(dataset, cells=("logistic_A",))
        self.assertIn("no negative training examples", str(caught.exception))

    def test_duplicate_requested_cells_are_rejected(self):
        dataset = _simple_dataset()
        with self.assertRaises(Phase2HError) as caught:
            run_cv(dataset, cells=("logistic_A", "logistic_A"))
        self.assertIn("duplicate model cells", str(caught.exception))

    def test_class_weights_and_model_configs_are_preregistered(self):
        y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        weights = balanced_class_weights(y)
        self.assertEqual(weights, {1: 6.0, 0: 12 / 22})
        self.assertEqual(LOGISTIC_CONFIG["class_weight"], "balanced")
        self.assertEqual(LOGISTIC_CONFIG["random_state"], SEED)
        dataset = _simple_dataset()
        cv = run_cv(dataset, cells=("logistic_A", "lightgbm_A"))
        for cell in ("logistic_A", "lightgbm_A"):
            record = cv["fit_scope"][cell][0]
            self.assertEqual(record["fit_scope"], "training windows only")
            self.assertEqual(record["class_weights"], {
                KEEP: 5 / (2 * 1),
                DROP: 5 / (2 * 4),
            })
        logistic_config = cv["fit_scope"]["logistic_A"][0]["model_config"]
        self.assertEqual(logistic_config["family"], "logistic")
        self.assertEqual(logistic_config["params"]["class_weight"], "balanced")
        lgbm_config = cv["fit_scope"]["lightgbm_A"][0]["model_config"]
        self.assertEqual(lgbm_config["family"], "lightgbm")
        self.assertEqual(lgbm_config["params"]["class_weight"], {
            KEEP: 5 / 2,
            DROP: 5 / 8,
        })

    def test_fit_scope_proves_training_only_statistics(self):
        dataset = _simple_dataset()
        cv = run_cv(dataset, cells=("logistic_B",))
        for fold_index, fold in enumerate(cv["folds"]):
            record = cv["fit_scope"]["logistic_B"][fold_index]
            self.assertEqual(record["train_window_ids"], fold["train_window_ids"])
            self.assertEqual(record["test_window_id"], fold["test_window_id"])
            self.assertEqual(record["vectorizer"]["fit_on"], "training_rows_only")
            self.assertEqual(
                record["vectorizer"]["vocabulary_terms_origin"],
                "training_rows_only",
            )
            self.assertNotIn("terms_missing_from_train", record["vectorizer"])
            self.assertGreaterEqual(record["vectorizer"]["vocabulary_size"], 0)
            self.assertIn("test_oov_token_type_count", record["vectorizer"])
            self.assertIn("test_oov_token_types_sha256", record["vectorizer"])

    def test_oov_audit_matches_independent_recomputation(self):
        dataset = _simple_dataset()
        cv = run_cv(dataset, cells=("logistic_B",))
        for fold in cv["folds"]:
            train_texts = [
                row.text
                for window_id in fold["train_window_ids"]
                for row in dataset["windows"][window_id]["rows"]
            ]
            test_texts = [
                row.text for row in dataset["windows"][fold["test_window_id"]]["rows"]
            ]
            # Independent replication of the training-only vocabulary.
            try:
                vectorizer = CountVectorizer(**VECTORIZER_PARAMS).fit(train_texts)
                vocabulary = [
                    term
                    for term, _ in sorted(
                        vectorizer.vocabulary_.items(),
                        key=lambda item: item[1],
                    )
                ]
            except ValueError:
                # min_df pruning removed every term on this tiny corpus.
                vocabulary = []
            expected_oov = sorted({
                match.group().lower()
                for text in test_texts
                for match in re.finditer(VECTORIZER_PARAMS["token_pattern"], text)
            } - set(vocabulary))
            audit = cv["fit_scope"]["logistic_B"][fold["fold_index"]]["vectorizer"]
            self.assertEqual(audit["test_oov_token_type_count"], len(expected_oov))
            self.assertEqual(audit["test_oov_token_types"], expected_oov)
            self.assertEqual(
                audit["test_oov_token_types_sha256"],
                canonical_sha256(expected_oov),
            )

    def test_oov_audit_covers_boundary_tokens(self):
        rows_a = (
            _row(
                "winA", "winA:w0001", 1, 0, 9, "the alpha",
                label=KEEP, hints=("ENTITY",),
                mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row("winA", "winA:w0001", 2, 10, 18, "the beta"),
        )
        rows_b = (
            _row(
                "winB", "winB:w0001", 1, 0, 8, "the quux",
                label=KEEP, hints=("ENTITY",),
                mention_ids=("m2",), node_types=("ENTITY",),
            ),
            _row("winB", "winB:w0001", 2, 9, 14, "gamma"),
        )
        dataset = {
            "windows": {
                "winA": _window("winA", "the alpha the beta", rows_a),
                "winB": _window("winB", "the quux gamma", rows_b),
            },
        }
        validate_dataset(dataset, expected_positive_count=2)
        cv = run_cv(dataset, cells=("logistic_B",))
        fold = next(fold for fold in cv["folds"] if fold["test_window_id"] == "winB")
        audit = cv["fit_scope"]["logistic_B"][fold["fold_index"]]["vectorizer"]
        self.assertIn("quux", audit["test_oov_token_types"])
        # Independent boundary replication: "the" appears in both training
        # documents, so only its first-role flag fires; the OOV last/head
        # tokens emit nothing.
        vocabulary = [
            term
            for term, _ in sorted(
                CountVectorizer(**VECTORIZER_PARAMS).fit(
                    [row.text for row in rows_a],
                ).vocabulary_.items(),
                key=lambda item: item[1],
            )
        ]
        self.assertEqual(vocabulary, ["the"])
        _, boundaries = extract_sparse_inputs(dataset, ["winB"])
        flags = _boundary_sparse(boundaries, vocabulary)
        self.assertEqual(flags.sum(), 1.0)


class Phase2HMetricTests(unittest.TestCase):
    def test_thresholding_and_tie_breaking(self):
        dataset = _simple_dataset()
        scores = {
            "winA": [0.1, 0.49, 0.5, 0.9, 0.0],
            "winB": [0.2, 0.3, 0.4, 0.6, 0.51],
        }
        rankings = _rankings_from_scores(dataset, scores)
        for window_id in ("winA", "winB"):
            rows = dataset["windows"][window_id]["rows"]
            for row in rows:
                entry = rankings[window_id]["logistic_A"][row.candidate_id]
                self.assertEqual(
                    entry["selected"],
                    KEEP if entry["score"] >= KEEP_THRESHOLD else DROP,
                )
        # winB row 4 (score 0.6) and row 5 (score 0.51) both selected.
        win_b = rankings["winB"]["logistic_A"]
        rows_b = dataset["windows"]["winB"]["rows"]
        self.assertEqual(
            [win_b[row.candidate_id]["rank"] for row in rows_b],
            [5, 4, 3, 1, 2],
        )

    def test_pr_metrics_confusion_and_baselines(self):
        dataset = _simple_dataset()
        # winA: labels [DROP,KEEP,DROP,DROP,DROP], scores put beta top.
        scores = {
            "winA": [0.9, 0.8, 0.2, 0.1, 0.0],
            "winB": [0.95, 0.05, 0.04, 0.03, 0.02],
        }
        rankings = _rankings_from_scores(dataset, scores)
        metrics = compute_cell_metrics(
            dataset, rankings, "logistic_A",
        )
        pooled = metrics
        self.assertEqual(pooled["label_keep_count"], 2)
        self.assertEqual(pooled["predicted_keep_count"], 3)
        self.assertEqual(pooled["confusion_matrix"], {
            "true_positive": 2,
            "false_positive": 1,
            "true_negative": 7,
            "false_negative": 0,
        })
        self.assertEqual(pooled["precision"], {"hit_count": 2, "denominator": 3, "rate": 2 / 3})
        self.assertEqual(pooled["recall"], {"hit_count": 2, "denominator": 2, "rate": 1.0})
        self.assertEqual(pooled["f1"]["value"], 0.8)
        self.assertEqual(pooled["prevalence"], 0.2)
        self.assertEqual(pooled["all_drop_baseline"], {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        })
        self.assertEqual(pooled["all_keep_baseline"]["recall"], 1.0)
        self.assertEqual(pooled["all_keep_baseline"]["precision"], 0.2)

    def test_recall_precision_at_k_and_gold_ranks(self):
        text = "one two three four five"
        rows = (
            _row("win", "win:w0001", 1, 0, 3, "one"),
            _row(
                "win", "win:w0001", 2, 4, 7, "two",
                label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row("win", "win:w0001", 3, 8, 13, "three"),
            _row(
                "win", "win:w0001", 4, 14, 18, "four",
                label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row("win", "win:w0001", 5, 19, 23, "five"),
        )
        dataset = {"windows": {"win": _window("win", text, rows)}}
        validate_dataset(dataset, expected_positive_count=2)
        # scores: golds at ranks 1 and 3; candidate 5 at rank 2.
        rankings = _rankings_from_scores(
            dataset, {"win": [0.9, 0.85, 0.2, 0.8, 0.1]},
        )
        metrics = compute_cell_metrics(dataset, rankings, "logistic_A")
        self.assertEqual(metrics["recall_at_k"]["1"], {
            "hit_count": 0, "denominator": 2, "rate": 0.0,
        })
        self.assertEqual(metrics["recall_at_k"]["3"], {
            "hit_count": 2, "denominator": 2, "rate": 1.0,
        })
        self.assertEqual(metrics["precision_at_k"]["1"], {
            "hit_count": 0, "denominator": 1, "rate": 0.0,
        })
        self.assertEqual(metrics["precision_at_k"]["3"], {
            "hit_count": 2, "denominator": 3, "rate": 2 / 3,
        })
        self.assertEqual(metrics["gold_rank"]["mean"], 2.5)
        self.assertEqual(metrics["gold_rank"]["median"], 2.5)
        per_fold = metrics["per_fold"]["win"]
        self.assertEqual(per_fold["gold_rank"]["count"], 2)
        self.assertEqual(per_fold["recall_at_k"]["1"]["rate"], 0.0)

    def test_overlap_diagnostics_count_distractors_outranking_gold(self):
        text = "abcdefghijklmnop"
        rows = (
            _row("win", "win:w0001", 1, 0, 2, "ab"),
            _row(
                "win", "win:w0001", 2, 3, 6, "def",
                label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
            ),
            _row("win", "win:w0001", 3, 2, 8, "cdefgh"),
            _row("win", "win:w0001", 4, 4, 5, "e"),
            _row("win", "win:w0001", 5, 9, 14, "jklmn"),
        )
        dataset = {"windows": {"win": _window("win", text, rows)}}
        validate_dataset(dataset, expected_positive_count=1)
        # containing (2,8) score 0.95, contained (4,5) score 0.9, gold 0.85.
        rankings = _rankings_from_scores(
            dataset,
            {"win": [0.1, 0.85, 0.95, 0.9, 0.05]},
        )
        metrics = compute_cell_metrics(dataset, rankings, "logistic_A")
        diagnostics = metrics["overlap_cluster_rank"]
        self.assertEqual(diagnostics["count"], 1)
        self.assertEqual(diagnostics["median"], 3.0)
        entry = metrics["per_fold"]["win"]["overlap_diagnostics"][
            "per_gold_positive"
        ][0]
        self.assertEqual(entry["gold_rank"], 3)
        self.assertEqual(entry["overlap_cluster_size"], 3)
        self.assertEqual(entry["overlap_cluster_rank"], 3)
        self.assertEqual(
            entry["containing_distractors_outranking"], 1,
        )
        self.assertEqual(
            entry["contained_distractors_outranking"], 1,
        )
        aggregate = metrics["per_fold"]["win"]["overlap_diagnostics"]
        self.assertEqual(
            aggregate["containing_distractors_outranking_total"], 1,
        )
        self.assertEqual(
            aggregate["contained_distractors_outranking_total"], 1,
        )


class Phase2HErrorTaxonomyTests(unittest.TestCase):
    def _row(self, **kwargs):
        defaults = dict(
            case_id="win", window_id="win:w0001", index=1,
            start=10, end=14, text="xxxx",
        )
        defaults.update(kwargs)
        return _row(**defaults)

    def test_every_false_positive_code_is_reachable_and_precedence_holds(self):
        gold_spans = [(10, 14)]
        cases = [
            (
                "OVERLAPPING_LONGER_SPAN",
                self._row(index=1, start=5, end=20, text="longer span here",
                          hints=("ENTITY",)),
            ),
            (
                "OVERLAPPING_SHORTER_FRAGMENT",
                self._row(index=2, start=11, end=13, text="xx",
                          hints=("ENTITY",)),
            ),
            (
                "PRONOUN_DISTRACTOR",
                self._row(index=3, start=0, end=3, text="you"),
            ),
            (
                "DISCOURSE_FILLER",
                self._row(index=4, start=30, end=37, text="you know"),
            ),
            (
                "GENERIC_ACTION_DISTRACTOR",
                self._row(index=5, start=40, end=48, text="fight push"),
            ),
            (
                "GENERIC_ENTITY_DISTRACTOR",
                self._row(index=6, start=50, end=55, text="thing"),
            ),
            (
                "WRONG_CUE_PRIOR",
                self._row(
                    index=7, start=60, end=74, text="unless then when",
                    hints=("TIME",),
                ),
            ),
            (
                "SOURCE_POSITION_BIAS",
                self._row(
                    index=8, start=0, end=11, text="some long phrase",
                    hints=("TIME",),
                ),
            ),
            (
                "PARSER_FEATURE_ERROR",
                self._row(index=9, start=70, end=77, text="foo[bar]"),
            ),
        ]
        for expected, row in cases:
            with self.subTest(expected=expected):
                code = classify_candidate_error(
                    row,
                    window_gold_spans=gold_spans,
                    window_rows=(row,),
                    predicted_label=KEEP,
                    rank=1,
                    window_text_len=100,
                )
                self.assertEqual(code, expected)

    def test_false_negative_codes_depend_on_gold_rank(self):
        gold = self._row(
            index=1, start=10, end=14, text="gold span",
            label=KEEP, mention_ids=("m1",), node_types=("ENTITY",),
        )
        high = classify_candidate_error(
            gold,
            window_gold_spans=[(10, 14)],
            window_rows=(gold,) * GOLD_RANK_HIGH_K,
            predicted_label=DROP,
            rank=2,
            window_text_len=100,
        )
        self.assertEqual(high, "GOLD_RANKED_HIGH_THRESHOLD_MISS")
        low = classify_candidate_error(
            gold,
            window_gold_spans=[(10, 14)],
            window_rows=(gold,) * 20,
            predicted_label=DROP,
            rank=20,
            window_text_len=100,
        )
        self.assertEqual(low, "GOLD_RANKED_LOW")
        correct = classify_candidate_error(
            gold,
            window_gold_spans=[(10, 14)],
            window_rows=(gold,),
            predicted_label=KEEP,
            rank=1,
            window_text_len=100,
        )
        self.assertIsNone(correct)

    def test_taxonomy_codes_match_definition(self):
        self.assertEqual(
            set(ERROR_PRECEDENCE)
            | {"GOLD_RANKED_HIGH_THRESHOLD_MISS", "GOLD_RANKED_LOW"},
            ERROR_CODES,
        )
        self.assertEqual(ERROR_PRECEDENCE[0], "PARSER_FEATURE_ERROR")

    def test_overlap_taxonomy_handles_inclusive_same_boundary_extensions(self):
        gold_spans = [(10, 14)]
        cases = [
            (
                "OVERLAPPING_LONGER_SPAN",
                self._row(
                    index=1, start=10, end=20,
                    text="same start extension", hints=("ENTITY",),
                ),
            ),
            (
                "OVERLAPPING_LONGER_SPAN",
                self._row(
                    index=2, start=5, end=14,
                    text="same end extension", hints=("ENTITY",),
                ),
            ),
            (
                "OVERLAPPING_SHORTER_FRAGMENT",
                self._row(
                    index=3, start=10, end=12,
                    text="same start fragment", hints=("ENTITY",),
                ),
            ),
            (
                "OVERLAPPING_SHORTER_FRAGMENT",
                self._row(
                    index=4, start=12, end=14,
                    text="same end fragment", hints=("ENTITY",),
                ),
            ),
        ]
        for expected, row in cases:
            with self.subTest(expected=expected, start=row.start, end=row.end):
                code = classify_candidate_error(
                    row,
                    window_gold_spans=gold_spans,
                    window_rows=(row,),
                    predicted_label=KEEP,
                    rank=1,
                    window_text_len=100,
                )
                self.assertEqual(code, expected)

    def test_exact_same_span_is_not_an_overlap_code(self):
        row = self._row(
            index=1, start=20, end=40, text="some ordinary phrase",
            hints=("ENTITY",),
        )
        code = classify_candidate_error(
            row,
            window_gold_spans=[(20, 40)],
            window_rows=(row,),
            predicted_label=KEEP,
            rank=1,
            window_text_len=100,
        )
        self.assertEqual(code, "OTHER")

    def test_overlap_precedence_collisions(self):
        gold_spans = [(10, 14)]
        containing_pronoun = self._row(
            index=1, start=5, end=20, text="you", hints=("ENTITY",),
        )
        fragment_filler = self._row(
            index=2, start=11, end=13, text="you know", hints=("ENTITY",),
        )
        artifact_overlap = self._row(
            index=3, start=5, end=20, text="foo[bar]", hints=("ENTITY",),
        )
        cases = [
            ("OVERLAPPING_LONGER_SPAN", containing_pronoun),
            ("OVERLAPPING_SHORTER_FRAGMENT", fragment_filler),
            ("PARSER_FEATURE_ERROR", artifact_overlap),
        ]
        for expected, row in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    classify_candidate_error(
                        row,
                        window_gold_spans=gold_spans,
                        window_rows=(row,),
                        predicted_label=KEEP,
                        rank=1,
                        window_text_len=100,
                    ),
                    expected,
                )


class Phase2HFeatureAggregationTests(unittest.TestCase):
    def test_strongest_features_rank_by_mean_across_folds(self):
        from types import SimpleNamespace

        names = [f"p{index:02d}" for index in range(11)] + [
            f"n{index:02d}" for index in range(11)
        ]
        # 11 positives + 11 negatives per fold so the per-fold top-10 slices
        # stay sign-pure.  p07/n07 dominate fold 0 but are near zero in
        # fold 1, where they fall outside the top-10; signed-sum aggregation
        # would rank p02/n02 first (2.0 / -2.0), mean aggregation must not.
        positives = [0.9, 0.1, 1.0, 0.8, 0.7, 0.6, 0.5, 1.9, 0.4, 0.3, 0.2]
        negatives = [
            -0.9, -0.1, -1.0, -0.8, -0.7, -0.6, -0.5, -2.5, -0.4, -0.3, -0.2,
        ]

        def logistic(fold_coefficients):
            return SimpleNamespace(
                coef_=np.array([fold_coefficients], dtype=np.float64),
            )

        fold1_positives = list(positives)
        fold1_positives[7] = 0.01
        fold1_negatives = list(negatives)
        fold1_negatives[7] = -0.01
        logistic_models = {
            "logistic_A": {
                0: (logistic(positives + negatives), names),
                1: (logistic(fold1_positives + fold1_negatives), names),
            },
        }
        result = strongest_features(logistic_models, cells=("logistic_A",))
        positive = result["logistic_A"]["aggregate_top_positive"]
        negative = result["logistic_A"]["aggregate_top_negative"]
        self.assertEqual(positive[0]["feature"], "p07")
        self.assertEqual(positive[0]["mean"], 1.9)
        self.assertEqual(positive[0]["folds_seen"], 1)
        self.assertEqual(positive[1]["feature"], "p02")
        self.assertEqual(positive[1]["mean"], 1.0)
        self.assertEqual(positive[1]["folds_seen"], 2)
        self.assertEqual(negative[0]["feature"], "n07")
        self.assertEqual(negative[0]["mean"], -2.5)
        self.assertEqual(negative[0]["folds_seen"], 1)
        self.assertEqual(negative[1]["feature"], "n02")
        self.assertEqual(negative[1]["mean"], -1.0)

        def lgbm(fold_importances):
            return SimpleNamespace(
                feature_importances_=np.array(
                    fold_importances, dtype=np.float64,
                ),
            )

        # Distinct nonnegative gains: p07 dominates fold 0 but falls outside
        # the top-20 in fold 1 (0.01), so its mean is 1.9 with folds_seen 1;
        # signed-sum aggregation would rank p02 (2.0 over two folds) first.
        gains0 = [
            0.9, 0.1, 1.0, 0.8, 0.7, 0.6, 0.5, 1.9, 0.4, 0.3, 0.2,
            0.95, 0.02, 0.8, 0.6, 0.5, 0.3, 0.7, 0.2, 0.1, 0.05, 0.4,
        ]
        gains1 = [
            0.9, 0.1, 1.0, 0.8, 0.7, 0.6, 0.5, 0.01, 0.4, 0.3, 0.2,
            0.01, 0.02, 0.8, 0.6, 0.5, 0.3, 0.7, 0.2, 0.1, 0.05, 0.4,
        ]
        lgbm_models = {
            "lightgbm_A": {
                0: (lgbm(gains0), names),
                1: (lgbm(gains1), names),
            },
        }
        result = strongest_features(lgbm_models, cells=("lightgbm_A",))
        top = result["lightgbm_A"]["aggregate_top_importances"]
        self.assertEqual(top[0]["feature"], "p07")
        self.assertEqual(top[0]["mean"], 1.9)
        self.assertEqual(top[0]["folds_seen"], 1)
        self.assertEqual(top[1]["feature"], "p02")
        self.assertEqual(top[1]["mean"], 1.0)
        self.assertEqual(top[1]["folds_seen"], 2)


class Phase2HBenchmarkValidationTests(unittest.TestCase):
    def test_benchmark_mapping_self_verifies_against_trusted_lock(self):
        benchmark = load_benchmark(BENCHMARK)
        validate_benchmark_mapping(benchmark)
        tampered = dict(benchmark)
        tampered["extra_key"] = "x"
        with self.assertRaises(Phase2HError) as caught:
            validate_benchmark_mapping(tampered)
        self.assertIn("content hash", str(caught.exception))

    def test_run_experiment_rejects_tampered_mapping_input(self):
        benchmark = load_benchmark(BENCHMARK)
        tampered = dict(benchmark)
        tampered["cases"] = [dict(case) for case in benchmark["cases"]]
        tampered["cases"][0]["source_text"] = "tampered source"
        with self.assertRaises(Phase2HError) as caught:
            run_experiment(tampered, cells=("logistic_A",))
        self.assertIn("content hash", str(caught.exception))

    def test_run_experiment_rejects_wrong_envelope_mapping(self):
        benchmark = load_benchmark(BENCHMARK)
        wrong_split = dict(benchmark)
        wrong_split["split"] = "OTHER"
        with self.assertRaises(Phase2HError) as caught:
            run_experiment(wrong_split, cells=("logistic_A",))
        self.assertIn("LEGACY_FAILURE", str(caught.exception))


class Phase2HCrossProcessTests(unittest.TestCase):
    def test_cross_process_four_cell_determinism_on_synthetic_dataset(self):
        """Four-cell CV determinism across processes on the bounded synthetic
        dataset.

        A cross-process four-cell run of the real 16.6k-candidate benchmark
        would cost roughly twice the already heavy single-process smoke test,
        so this regression runs all four cells on the small synthetic dataset
        in a fresh interpreter and compares a canonical digest of the folds,
        fit scope (including the OOV audit), and OOF scores against the
        in-process run.  Real-benchmark rerun determinism is covered
        single-process by ``test_cv_is_deterministic_across_reruns``.
        """
        dataset = _simple_dataset()
        cells = CELLS
        in_process = _cv_digest(dataset, cells)
        script = (
            "import base64, pickle, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from pipeline.phase2h_endpoint_scoring import canonical_sha256, run_cv\n"
            "dataset = pickle.loads(base64.b64decode(sys.stdin.buffer.read()))\n"
            "cv = run_cv(dataset, cells=tuple(sys.argv[2].split(',')))\n"
            "payload = {'folds': cv['folds'], 'fit_scope': cv['fit_scope'], "
            "'oof_scores': cv['oof_scores']}\n"
            "print(canonical_sha256(payload))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(ROOT), ",".join(cells)],
            input=base64.b64encode(pickle.dumps(dataset)).decode(),
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), in_process)


class Phase2HArtifactTests(unittest.TestCase):
    def test_artifact_is_hash_locked_immutable_and_manifested(self):
        benchmark = load_benchmark(BENCHMARK)
        dataset = build_dataset(benchmark)
        cv = run_cv(dataset, cells=("logistic_A",))
        rankings = compute_rankings(dataset, cv["oof_scores"], cells=("logistic_A",))
        errors = classify_all_errors(dataset, rankings, cells=("logistic_A",))
        metrics = {
            cell: compute_cell_metrics(dataset, rankings, cell)
            for cell in ("logistic_A",)
        }
        result = {
            "dataset": dataset,
            "folds": cv["folds"],
            "fit_scope": cv["fit_scope"],
            "rankings": rankings,
            "errors": errors,
            "metrics": metrics,
            "strongest_features": strongest_features(
                cv["fitted_models"], cells=("logistic_A",),
            ),
            "cells": ["logistic_A"],
        }
        tables = {
            window_id: build_window_table(
                dataset, rankings, errors, window_id, cells=("logistic_A",),
            )
            for window_id in sorted(dataset["windows"])
        }
        hashes = {
            window_id: canonical_sha256(table)
            for window_id, table in tables.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            aggregate = build_aggregate(
                BENCHMARK, result, repo=ROOT, window_table_hashes=hashes,
            )
            inner = {
                key: value for key, value in aggregate.items()
                if key != "content_sha256"
            }
            self.assertEqual(
                aggregate["content_sha256"], canonical_sha256(inner),
            )
            self.assertEqual(aggregate["run_version"], RUN_VERSION)
            self.assertEqual(
                aggregate["input_hashes"]["benchmark_content_sha256"],
                "a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd",
            )
            output = Path(tmp) / "phase2h-run"
            publish_artifact(output, aggregate, tables)
            self.assertTrue(
                (output / "phase2h-endpoint-scoring.json").exists(),
            )
            self.assertTrue((output / "MANIFEST.json").exists())
            self.assertEqual(len(list((output / "windows").iterdir())), 5)
            manifest = json.loads(
                (output / "MANIFEST.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(len(manifest["files"]), 1 + 5)
            for entry in manifest["files"]:
                path = output / entry["path"]
                self.assertEqual(
                    entry["file_sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            with self.assertRaises(ValueError):
                publish_artifact(output, aggregate, tables)
            bad = Path(tmp) / "bad-run"
            with patch(
                "pipeline.phase2h_endpoint_scoring.os.replace",
                side_effect=OSError("boom"),
            ):
                with self.assertRaises(OSError):
                    publish_artifact(bad, aggregate, tables)
            self.assertFalse(bad.exists())

    def test_window_tables_retain_full_provenance_and_predictions(self):
        benchmark = load_benchmark(BENCHMARK)
        dataset = build_dataset(benchmark)
        cv = run_cv(dataset, cells=("logistic_A",))
        rankings = compute_rankings(dataset, cv["oof_scores"], cells=("logistic_A",))
        errors = classify_all_errors(dataset, rankings, cells=("logistic_A",))
        table = build_window_table(
            dataset, rankings, errors, "wave-reset-after-kill",
            cells=("logistic_A",),
        )
        self.assertEqual(table["candidate_count"], 3344)
        positive = next(
            candidate for candidate in table["candidates"]
            if candidate["label"] == KEEP
        )
        self.assertTrue(positive["gold_mention_ids"])
        self.assertTrue(positive["gold_node_types"])
        self.assertIn("logistic_A", positive["predictions"])
        prediction = positive["predictions"]["logistic_A"]
        self.assertIn("score", prediction)
        self.assertIn("rank", prediction)
        self.assertIn("selected", prediction)
        self.assertIn("error_code", prediction)
        row_by_id = {
            row.candidate_id: row
            for row in dataset["windows"]["wave-reset-after-kill"]["rows"]
        }
        for candidate in table["candidates"]:
            for field in (
                "candidate_id", "start", "end", "absolute_start",
                "absolute_end", "text", "segment_ids", "segment_bounds",
                "type_hints",
                "label", "excluded", "ambiguity_state",
            ):
                self.assertIn(field, candidate)
            row = row_by_id[candidate["candidate_id"]]
            self.assertEqual(
                candidate["segment_bounds"],
                [
                    [bound_start, bound_end]
                    for bound_start, bound_end in row.segment_bounds
                ],
            )

    def test_compare_artifacts_verifies_git_state_manifest_and_file_bytes(self):
        benchmark = load_benchmark(BENCHMARK)
        dataset = build_dataset(benchmark)
        cv = run_cv(dataset, cells=("logistic_A",))
        rankings = compute_rankings(
            dataset, cv["oof_scores"], cells=("logistic_A",),
        )
        errors = classify_all_errors(
            dataset, rankings, cells=("logistic_A",),
        )
        metrics = {
            "logistic_A": compute_cell_metrics(
                dataset, rankings, "logistic_A",
            ),
        }
        result = {
            "dataset": dataset,
            "folds": cv["folds"],
            "fit_scope": cv["fit_scope"],
            "rankings": rankings,
            "errors": errors,
            "metrics": metrics,
            "strongest_features": strongest_features(
                cv["fitted_models"], cells=("logistic_A",),
            ),
            "cells": ["logistic_A"],
        }
        tables = {
            window_id: build_window_table(
                dataset, rankings, errors, window_id,
                cells=("logistic_A",),
            )
            for window_id in sorted(dataset["windows"])
        }
        hashes = {
            window_id: canonical_sha256(table)
            for window_id, table in tables.items()
        }

        def build(commit, dirty, created_at):
            with patch(
                "pipeline.phase2h_endpoint_scoring._git_state",
                return_value=(commit, dirty),
            ):
                return build_aggregate(
                    BENCHMARK, result, repo=ROOT,
                    created_at=created_at,
                    window_table_hashes=hashes,
                )

        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            publish_artifact(
                left, build("commit-a", False, "2026-01-01T00:00:00Z"), tables,
            )
            publish_artifact(
                right, build("commit-b", True, "2026-08-16T00:00:00Z"), tables,
            )
            differences = compare_artifacts(left, right)
            self.assertIn("git_commit differs", differences)
            self.assertIn("repository_dirty differs", differences)

            clean = Path(tmp) / "clean"
            publish_artifact(
                clean, build("commit-a", False, "2026-02-02T00:00:00Z"), tables,
            )
            self.assertEqual(compare_artifacts(left, clean), [])

            tampered_window = Path(tmp) / "tampered-window"
            publish_artifact(
                tampered_window,
                build("commit-a", False, "2026-03-03T00:00:00Z"),
                tables,
            )
            window_path = tampered_window / "windows" / "wave-reset-after-kill.json"
            window_path.write_text(
                window_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            differences = compare_artifacts(left, tampered_window)
            self.assertTrue(any(
                "wave-reset-after-kill" in difference
                for difference in differences
            ))

            tampered_aggregate = Path(tmp) / "tampered-aggregate"
            publish_artifact(
                tampered_aggregate,
                build("commit-a", False, "2026-04-04T00:00:00Z"),
                tables,
            )
            aggregate_path = tampered_aggregate / "phase2h-endpoint-scoring.json"
            body = json.loads(aggregate_path.read_text(encoding="utf-8"))
            body["metrics"]["logistic_A"]["recall"]["rate"] = 0.1
            aggregate_path.write_text(
                json.dumps(body, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            differences = compare_artifacts(left, tampered_aggregate)
            self.assertTrue(any(
                "phase2h-endpoint-scoring.json" in difference
                or "content_sha256" in difference
                or "metrics differs" in difference
                for difference in differences
            ))

    def test_error_taxonomy_counts_cover_all_candidates(self):
        dataset = _simple_dataset()
        cv = run_cv(dataset, cells=("logistic_A",))
        rankings = compute_rankings(dataset, cv["oof_scores"], cells=("logistic_A",))
        errors = classify_all_errors(dataset, rankings, cells=("logistic_A",))
        counts = error_taxonomy_counts(errors, dataset, cells=("logistic_A",))
        total = counts["logistic_A"]["correct"] + sum(
            counts["logistic_A"]["codes"].values(),
        )
        self.assertEqual(total, 10)
        self.assertEqual(set(counts["logistic_A"]["codes"]), ERROR_CODES)


class Phase2HIntegrationTests(unittest.TestCase):
    def test_full_four_cell_real_benchmark_smoke(self):
        benchmark = load_benchmark(BENCHMARK)
        dataset = build_dataset(benchmark)
        cv = run_cv(dataset)
        rankings = compute_rankings(dataset, cv["oof_scores"])
        errors = classify_all_errors(dataset, rankings)
        for cell in CELLS:
            metrics = compute_cell_metrics(dataset, rankings, cell)
            self.assertEqual(metrics["candidate_count"], 16624)
            self.assertEqual(metrics["label_keep_count"], 33)
            self.assertEqual(
                sum(
                    fold_metrics["candidate_count"]
                    for fold_metrics in metrics["per_fold"].values()
                ),
                16624,
            )
            self.assertEqual(len(metrics["per_fold"]), 5)
        counts = error_taxonomy_counts(errors, dataset)
        for cell in CELLS:
            self.assertEqual(
                counts[cell]["correct"]
                + sum(counts[cell]["codes"].values()),
                16624,
            )


if __name__ == "__main__":
    unittest.main()
