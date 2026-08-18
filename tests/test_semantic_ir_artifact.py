import copy
import hashlib
import json
import math
import unittest

from pipeline.semantic_compiler import SemanticCompilerConfig, compile_source_semantic_ir
from pipeline.semantic_ir_artifact import SemanticRunArtifact, build_semantic_run_artifact
from pipeline.semantic_qualifiers import QUALIFIER_SYSTEM
from tests.test_semantic_compiler import ScriptedSemanticModel, _window


class SemanticIRArtifactTests(unittest.TestCase):
    @staticmethod
    def _reseal_outer(payload):
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        payload["content_sha256"] = hashlib.sha256(
            json.dumps(
                body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def _run(self, chat=None):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        return compile_source_semantic_ir(_window(), chat or ScriptedSemanticModel(), config=config)

    def _artifact(self, run):
        return build_semantic_run_artifact(
            run, git_commit="a" * 40, repository_dirty=True,
            created_at="2026-08-16T15:30:00-07:00",
            input_hashes={"fixture_sha256": "f" * 64, "database_sha256": "d" * 64},
            evaluation={"semantic_checksum": {"hit_count": 3, "denominator": 4, "rate": 0.75}},
        )

    def test_artifact_retains_complete_run_and_round_trips_deterministically(self):
        run = self._run()
        artifact = self._artifact(run)
        payload = artifact.payload
        self.assertEqual(payload["git_commit"], "a" * 40)
        self.assertIn("source_window_sha256", payload["input_hashes"])
        retained = payload["run"]
        self.assertEqual(len(retained["mention_catalog"]), len(run.mention_catalog))
        self.assertEqual(
            len(retained["mention_selection"]["partition_results"]),
            len(run.mention_selection.partition_results),
        )
        self.assertEqual(len(retained["qualifier_runs"]), len(run.qualifier_runs))
        self.assertEqual(
            len(retained["coreference"]["candidate_sets"]),
            len(run.coreference.candidate_sets),
        )
        self.assertEqual(
            len(retained["edge_classification"]["pairs"]),
            len(run.edge_classification.pairs),
        )
        self.assertTrue(all(
            "raw_output" in item["result"] for item in retained["qualifier_runs"]
        ))
        restored = SemanticRunArtifact.from_json(artifact.to_json())
        self.assertEqual(restored.content_sha256, artifact.content_sha256)
        self.assertEqual(restored.to_json(), artifact.to_json())
        self.assertEqual(restored.file_sha256, artifact.file_sha256)

    def test_content_tamper_duplicate_json_and_bad_revision_fail_closed(self):
        artifact = self._artifact(self._run())
        tampered = copy.deepcopy(dict(artifact.payload))
        tampered["run"]["mention_selection"]["partition_results"][0]["raw_output"] = "{}"
        with self.assertRaisesRegex(ValueError, "content hash"):
            SemanticRunArtifact(tampered)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SemanticRunArtifact.from_json('{"content_sha256":"x","content_sha256":"y"}')
        with self.assertRaisesRegex(ValueError, "git commit"):
            build_semantic_run_artifact(
                self._run(), git_commit="short", repository_dirty=False,
                created_at="2026-08-16T15:30:00Z",
            )

    def test_provider_failure_keeps_catalog_raw_request_and_failure_taxonomy(self):
        model = ScriptedSemanticModel()

        def fail_qualifier(**kwargs):
            if kwargs["system"] == QUALIFIER_SYSTEM:
                raise TimeoutError("offline")
            return model(**kwargs)

        run = self._run(fail_qualifier)
        artifact = self._artifact(run)
        qualifier_runs = artifact.payload["run"]["qualifier_runs"]
        self.assertTrue(qualifier_runs)
        for item in qualifier_runs:
            self.assertEqual(
                [candidate["candidate_id"] for candidate in item["candidates"]],
                item["result"]["candidate_ids"],
            )
            self.assertEqual(item["result"]["raw_output"], "")
            self.assertTrue(item["result"]["failure"].startswith("QualifierProviderError:"))
            self.assertTrue(item["result"]["request_json"])
        self.assertTrue(any(
            item["code"] == "PROVIDER_FAILURE" for item in artifact.payload["run"]["failures"]
        ))

    def test_input_hashes_and_timestamp_are_strict(self):
        run = self._run()
        with self.assertRaisesRegex(ValueError, "input hashes"):
            build_semantic_run_artifact(
                run, git_commit="a" * 40, repository_dirty=False,
                created_at="2026-08-16T15:30:00Z", input_hashes={"fixture": "bad"},
            )
        with self.assertRaisesRegex(ValueError, "timezone"):
            build_semantic_run_artifact(
                run, git_commit="a" * 40, repository_dirty=False,
                created_at="2026-08-16T15:30:00",
            )

    def test_resealed_inner_run_tampering_is_rejected_reconstructively(self):
        artifact = self._artifact(self._run())
        mutations = (
            lambda run: run.__setitem__("integrity_sha256", "0" * 64),
            lambda run: run.__setitem__("config", {}),
            lambda run: run.__setitem__("mention_catalog", []),
            lambda run: run.__setitem__("failures", [{
                "stage": "mentions", "code": "MODEL_PARSE_FAILURE",
                "item_id": None, "detail": "invented",
            }]),
            lambda run: run.__setitem__("status", "NONE"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                payload = copy.deepcopy(dict(artifact.payload))
                mutate(payload["run"])
                self._reseal_outer(payload)
                with self.assertRaises(ValueError):
                    SemanticRunArtifact(payload)

        payload = copy.deepcopy(dict(artifact.payload))
        payload["run"]["mention_selection"]["partition_results"][0]["raw_output"] = "{}"
        self._reseal_outer(payload)
        with self.assertRaises(ValueError):
            SemanticRunArtifact(payload)

    def test_resealed_required_input_hash_tampering_is_rejected(self):
        payload = copy.deepcopy(dict(self._artifact(self._run()).payload))
        payload["input_hashes"]["source_window_sha256"] = "0" * 64
        self._reseal_outer(payload)
        with self.assertRaisesRegex(ValueError, "source input hashes"):
            SemanticRunArtifact(payload)

    def test_nonfinite_evaluation_values_fail_closed(self):
        run = self._run()
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    build_semantic_run_artifact(
                        run, git_commit="a" * 40, repository_dirty=False,
                        created_at="2026-08-16T15:30:00Z",
                        evaluation={"rate": value},
                    )

    def test_file_hash_covers_the_only_supported_serialization(self):
        artifact = self._artifact(self._run())
        encoded = artifact.to_json().encode("utf-8")
        self.assertEqual(artifact.file_sha256, hashlib.sha256(encoded).hexdigest())
        with self.assertRaises(TypeError):
            artifact.to_json(indent=2)

    def test_evaluation_shape_and_typed_canonical_order_fail_closed(self):
        artifact = self._artifact(self._run())
        for invalid in ("not-an-object", [1, 2], 42, True):
            with self.subTest(evaluation=invalid):
                payload = copy.deepcopy(dict(artifact.payload))
                payload["evaluation"] = invalid
                self._reseal_outer(payload)
                with self.assertRaisesRegex(ValueError, "evaluation"):
                    SemanticRunArtifact(payload)

        payload = copy.deepcopy(dict(artifact.payload))
        self.assertGreater(len(payload["run"]["graph"]["nodes"]), 1)
        payload["run"]["graph"]["nodes"].reverse()
        self._reseal_outer(payload)
        with self.assertRaisesRegex(ValueError, "canonical typed form"):
            SemanticRunArtifact(payload)


if __name__ == "__main__":
    unittest.main()
