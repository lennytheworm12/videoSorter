import hashlib
import json
from pathlib import Path
import tarfile
import unittest

from pipeline.semantic_ir_artifact import SemanticRunArtifact


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/phase2f_artifacts/phase2f-legacy-pro-run1.tar.gz"
ARCHIVE_SHA256 = "e6c2122a2b91c2b70d9775f2c108c26c82cdfff2f5cea9b3c5f60dbbc4146330"
AGGREGATE_FILE_SHA256 = "68e85a6d9b69265ffe8793b27f0c8a44235857be4b8134a1670c3d48c2d4fe1c"
AGGREGATE_CONTENT_SHA256 = "80be66cc48b9f2e7685a3da2effa3e190cc1cd8cae5aa9c392a6ba1e693c782a"
RUN_COMMIT = "a0feefd50013722c943976a9131eb545f364178c"


class Phase2FProviderFailureArtifactTests(unittest.TestCase):
    def test_archived_run_is_reconstructible_provider_failure_not_quality_evidence(self):
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), ARCHIVE_SHA256)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            prefix = "phase2f-legacy-pro-run1/"
            aggregate_raw = archive.extractfile(prefix + "legacy-evaluation.json").read()
            self.assertEqual(hashlib.sha256(aggregate_raw).hexdigest(), AGGREGATE_FILE_SHA256)
            aggregate = json.loads(aggregate_raw)
            self.assertEqual(aggregate["content_sha256"], AGGREGATE_CONTENT_SHA256)
            self.assertEqual(aggregate["git_commit"], RUN_COMMIT)
            self.assertFalse(aggregate["repository_dirty"])
            self.assertFalse(aggregate["gate"]["passed"])
            self.assertEqual(aggregate["evaluation"]["failure_counts"]["PROVIDER_FAILURE"], 30)
            self.assertEqual(
                aggregate["evaluation"]["mention_candidate_coverage"],
                {"hit_count": 33, "denominator": 33, "rate": 1.0},
            )
            indexed = {item["case_id"]: item for item in aggregate["artifacts"]}
            self.assertEqual(len(indexed), 5)
            for case in aggregate["evaluation"]["cases"]:
                self.assertEqual(case["first_loss"], "PROVIDER_FAILURE")
                member = prefix + indexed[case["case_id"]]["file"]
                raw = archive.extractfile(member).read()
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), indexed[case["case_id"]]["file_sha256"],
                )
                artifact = SemanticRunArtifact.from_json(raw.decode("utf-8"))
                self.assertEqual(artifact.content_sha256, indexed[case["case_id"]]["content_sha256"])
                run = artifact.payload["run"]
                partitions = run["mention_selection"]["partition_results"]
                self.assertEqual(len(partitions), 6)
                self.assertTrue(all(
                    item["failure"] == "MentionProviderError:URLError"
                    and item["raw_output"] == "" and item["parsed_output"] is None
                    for item in partitions
                ))


if __name__ == "__main__":
    unittest.main()
