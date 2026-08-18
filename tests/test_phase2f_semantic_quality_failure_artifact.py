import hashlib
import json
from pathlib import Path
import tarfile
import unittest

from pipeline.semantic_ir_artifact import SemanticRunArtifact


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/phase2f_artifacts/phase2f-legacy-pro-run2.tar.gz"
ARCHIVE_SHA256 = "b17cde9d7dc909c317aac81be08e9ed4860f91231d5568aeb6ee515a1fd67183"
AGGREGATE_FILE_SHA256 = "ad3801a9fc23a23837fe0ad078273a2744fb9640bbd826172d359af4654cf547"
AGGREGATE_CONTENT_SHA256 = "b0a030765217f2dcb52634d31eec171b307541308012945f87864cf7d5697492"
RUN_COMMIT = "b5317c6bd90572e052ab85f399e339c4de83a4e8"


class Phase2FSemanticQualityFailureArtifactTests(unittest.TestCase):
    def test_archived_run_is_reconstructible_model_quality_failure(self):
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), ARCHIVE_SHA256)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            prefix = "phase2f-legacy-pro-run2/"
            aggregate_raw = archive.extractfile(prefix + "legacy-evaluation.json").read()
            self.assertEqual(hashlib.sha256(aggregate_raw).hexdigest(), AGGREGATE_FILE_SHA256)
            aggregate = json.loads(aggregate_raw)
            self.assertEqual(aggregate["content_sha256"], AGGREGATE_CONTENT_SHA256)
            self.assertEqual(aggregate["git_commit"], RUN_COMMIT)
            self.assertFalse(aggregate["repository_dirty"])
            self.assertEqual(aggregate["provider"], "deepseek")
            self.assertEqual(aggregate["provider_endpoint"], "https://api.deepseek.com")
            self.assertFalse(aggregate["gate"]["passed"])
            evaluation = aggregate["evaluation"]
            self.assertEqual(evaluation["mention_candidate_coverage"], {
                "denominator": 33, "hit_count": 33, "rate": 1.0,
            })
            for key, denominator in (
                ("mention_selection_recall", 33), ("mention_type_recall", 33),
                ("edge_recall", 24), ("qualifier_recall", 10),
                ("reference_recall", 8), ("semantic_checksum", 75),
            ):
                self.assertEqual(evaluation[key], {
                    "denominator": denominator, "hit_count": 0, "rate": 0.0,
                })

            mention_calls = qualifier_calls = edge_calls = provider_failures = 0
            indexed = {item["case_id"]: item for item in aggregate["artifacts"]}
            for case in evaluation["cases"]:
                self.assertEqual(case["first_loss"], "MODEL_PARSE_FAILURE")
                raw = archive.extractfile(prefix + indexed[case["case_id"]]["file"]).read()
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), indexed[case["case_id"]]["file_sha256"],
                )
                artifact = SemanticRunArtifact.from_json(raw.decode("utf-8"))
                self.assertEqual(artifact.to_json().encode(), raw)
                run = artifact.payload["run"]
                mention_calls += len(run["mention_selection"]["partition_results"])
                qualifier_calls += len(run["qualifier_runs"])
                edge_calls += len(run["edge_results"])
                provider_failures += sum(
                    failure["code"] == "PROVIDER_FAILURE" for failure in run["failures"]
                )
            self.assertEqual((mention_calls, qualifier_calls, edge_calls), (30, 80, 1839))
            self.assertEqual(provider_failures, 0)


if __name__ == "__main__":
    unittest.main()
