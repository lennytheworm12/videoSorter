"""Focused tests for Phase 2K production-model selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.phase2j_context_ablation import (
    canonical_sha256,
    file_sha256,
    text_sha256,
)
from pipeline.phase2k_full_transcript_ablation import (
    REVIEW_FIELDS,
    build_extraction_instructions,
    build_payloads_artifact,
    build_selection_artifact,
)
from pipeline.phase2k_production_model_selection import (
    CALLS_PER_TARGET,
    CANDIDATE_COUNT,
    CONDITION_MODELS,
    MODEL_FLASH,
    MODEL_PRO,
    RUN_SCHEMA_VERSION,
    VERIFIER_RESPONSE_SCHEMA_VERSION,
    build_verifier_payload,
    candidate_call_id,
    check_selection_integrity,
    compute_condition_metrics,
    deterministic_candidate_order,
    evaluate_condition_gate,
    evaluate_verifier_usefulness,
    select_production_model,
    validate_verifier_response,
    verifier_call_id,
)
from scripts import run_phase2k_full_transcript_ablation as base_script
from scripts import run_phase2k_production_model_selection as sel_script
from scripts.run_phase2k_full_transcript_ablation import (
    call_prompt_bytes as base_call_prompt_bytes,
)
from tests._phase2k_full_transcript_ablation_helpers import (
    build_phase2k_fixture,
    make_valid_intermediate_response,
)

FIXTURE: dict | None = None


def fixture() -> dict:
    global FIXTURE
    if FIXTURE is None:
        data = build_phase2k_fixture(
            Path(tempfile.mkdtemp(prefix="p2k-prodsel-tests-")),
        )
        selection = build_selection_artifact(
            manifest_path=data["manifest_path"],
            manifest=data["manifest"],
            db_path=data["db_path"],
            cases=data["cases"],
        )
        instructions = build_extraction_instructions()
        payloads = build_payloads_artifact(
            selection=selection,
            instructions=instructions,
            payload_cases=data["payload_cases"],
            provenance_by_case=data["provenance_by_case"],
        )
        out_dir = Path(tempfile.mkdtemp(prefix="p2k-prodsel-out-"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / base_script.SELECTION_FILENAME).write_text(
            json.dumps(selection), encoding="utf-8",
        )
        (out_dir / base_script.INSTRUCTIONS_FILENAME).write_text(
            json.dumps(instructions), encoding="utf-8",
        )
        (out_dir / base_script.PAYLOADS_FILENAME).write_text(
            json.dumps(payloads), encoding="utf-8",
        )
        data.update({
            "selection_artifact": selection,
            "instructions": instructions,
            "payloads_artifact": payloads,
            "output_dir": out_dir,
        })
        FIXTURE = data
    return FIXTURE


def cli_args(**kwargs):
    data = fixture()
    defaults = {
        "output_dir": str(data["output_dir"]),
        "db": str(data["db_path"]),
        "vocabulary": str(data["vocabulary_path"]),
        "run_dir": None,
        "max_workers": 1,
        "retries": 0,
        "force": False,
        "transport": "auto",
    }
    defaults.update(kwargs)
    return mock.Mock(**defaults)


def canned_transport(responses_by_prompt_sha: dict[str, str]):
    def fake_execute(self, *, model, prompt_bytes, mode="auto"):
        sha = text_sha256(prompt_bytes.decode("utf-8"))
        if sha not in responses_by_prompt_sha:
            raise AssertionError("unexpected prompt requested from transport")
        return responses_by_prompt_sha[sha], "opencode_cli"

    return fake_execute


class CannedTransport(sel_script.ModelTransport):
    """Deterministic offline transport returning canned model responses."""

    def __init__(self, generator_responses: dict[str, str]) -> None:
        super().__init__()
        self.generator_responses = generator_responses
        self.calls: list[tuple[str, str]] = []

    def execute(self, *, model, prompt_bytes, mode="auto"):  # noqa: D102
        text = prompt_bytes.decode("utf-8")
        wrapper = sel_script.VERIFIER_WRAPPER_PROMPT
        if text.startswith(wrapper):
            body = json.loads(text[len(wrapper) + 2:])
            case_id = body["verifier_payload"]["case_id"]
            presented = [
                c["candidate_id"]
                for c in body["verifier_payload"]["candidates"]
            ]
            assert len(presented) == CANDIDATE_COUNT
            response = json.dumps({
                "schema_version": VERIFIER_RESPONSE_SCHEMA_VERSION,
                "case_id": case_id,
                "selected_candidate_id": presented[0],
                "rationale": "first presented preserves semantics best",
            })
            self.calls.append(("verifier", case_id))
            return response, "opencode_cli"
        sha = text_sha256(text)
        if sha not in self.generator_responses:
            raise AssertionError("generator prompt not recognized")
        pair = self._pair_for_sha(sha)
        self.calls.append(("generator", pair))
        return self.generator_responses[sha], "opencode_cli"

    def _pair_for_sha(self, sha: str) -> str:
        data = fixture()
        for pair in data["payload_cases"]:
            expected = text_sha256(
                base_call_prompt_bytes(pair["B"]).decode("utf-8"),
            )
            if expected == sha:
                return pair["case_id"]
        raise AssertionError("unknown generator prompt")


class FrozenBenchmarkTests(unittest.TestCase):
    def test_payload_regeneration_is_deterministic(self) -> None:
        data = fixture()
        rebuilt = build_payloads_artifact(
            selection=data["selection_artifact"],
            instructions=data["instructions"],
            payload_cases=data["payload_cases"],
            provenance_by_case=data["provenance_by_case"],
        )
        self.assertEqual(
            canonical_sha256(rebuilt),
            canonical_sha256(data["payloads_artifact"]),
        )

    def test_same_ten_targets_and_bronze_hashes(self) -> None:
        data = fixture()
        cases = data["payloads_artifact"]["cases"]
        self.assertEqual(len(cases), 10)
        for index, pair in enumerate(cases, 1):
            self.assertEqual(pair["case_id"], f"p2k:case:{index:04d}")
            self.assertEqual(
                pair["A"]["target"]["bronze_text_sha256"],
                pair["B"]["target"]["bronze_text_sha256"],
            )
            self.assertNotEqual(
                pair["A"]["content_sha256"],
                pair["B"]["content_sha256"],
            )


class InitAndSeparationTests(unittest.TestCase):
    def _init(self, condition: str, run_dir: Path) -> dict:
        with mock.patch.object(
            base_script,
            "DEFAULT_MANIFEST_PATH",
            fixture()["manifest_path"],
        ):
            sel_script.cmd_init(cli_args(
                condition=condition, run_dir=str(run_dir),
            ))
        return json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8"),
        )

    def test_manifests_bind_distinct_models_and_calls(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="p2k-sep-"))
        manifests = {}
        for condition in ("P", "F", "FV"):
            manifests[condition] = self._init(
                condition, workspace / condition,
            )
        self.assertEqual(manifests["P"]["requested_model"], MODEL_PRO)
        self.assertEqual(manifests["F"]["requested_model"], MODEL_FLASH)
        self.assertEqual(manifests["FV"]["requested_model"], MODEL_FLASH)
        self.assertEqual(len(manifests["P"]["calls"]), 10)
        self.assertEqual(len(manifests["F"]["calls"]), 10)
        self.assertEqual(len(manifests["FV"]["calls"]), 10 * CANDIDATE_COUNT)

    def test_best_of_five_candidate_ids_unique_and_independent(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="p2k-bo5-"))
        manifest = self._init("FV", workspace / "FV")
        by_case: dict[str, list[dict]] = {}
        for call in manifest["calls"]:
            self.assertEqual(call["role"], "generator")
            self.assertIn(call["candidate_id"], [
                f"candidate_{i}" for i in range(1, CANDIDATE_COUNT + 1)
            ])
            by_case.setdefault(call["case_id"], []).append(call)
        self.assertEqual(len(by_case), 10)
        for case_id, calls in by_case.items():
            ids = [call["candidate_id"] for call in calls]
            self.assertEqual(sorted(ids), sorted(set(ids)))
            self.assertEqual(len(ids), CALLS_PER_TARGET["FV"] - 1)
            self.assertEqual(
                len({call["prompt_sha256"] for call in calls}), 1,
                "candidates must share the same frozen input",
            )
        # No candidate leakage: prompts never mention other candidates.
        for call in manifest["calls"][:1]:
            payload_b = next(
                pair["B"] for pair in fixture()["payload_cases"]
                if pair["case_id"] == call["case_id"]
            )
            prompt = base_call_prompt_bytes(payload_b).decode("utf-8")
            self.assertNotIn("candidate_", prompt)

    def test_model_change_invalidates_manifest_reuse(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="p2k-modelchg-"))
        manifest = self._init("P", workspace / "P")
        manifest["requested_model"] = MODEL_FLASH
        (workspace / "P" / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        with mock.patch.object(
            base_script,
            "DEFAULT_MANIFEST_PATH",
            fixture()["manifest_path"],
        ):
            with self.assertRaises(SystemExit):
                sel_script.cmd_init(cli_args(
                    condition="P", run_dir=str(workspace / "P"),
                ))


class VerifierContractTests(unittest.TestCase):
    def test_ordering_is_deterministic_and_permuted(self) -> None:
        order_a = deterministic_candidate_order("p2k:case:0001")
        order_b = deterministic_candidate_order("p2k:case:0001")
        self.assertEqual(order_a, order_b)
        self.assertEqual(sorted(order_a), [
            f"candidate_{i}" for i in range(1, CANDIDATE_COUNT + 1)
        ])
        orders = {
            tuple(deterministic_candidate_order(f"p2k:case:{i:04d}"))
            for i in range(1, 11)
        }
        self.assertGreater(len(orders), 1, "ordering should vary per target")

    def test_verifier_payload_contains_all_candidates_and_context(self) -> None:
        pair = fixture()["payload_cases"][0]
        payload_b = pair["B"]
        responses = {
            f"candidate_{i}": make_valid_intermediate_response(payload_b)
            for i in range(1, CANDIDATE_COUNT + 1)
        }
        order = deterministic_candidate_order(pair["case_id"])
        envelope = build_verifier_payload(
            payload_b=payload_b,
            candidate_responses=responses,
            candidate_order=order,
        )
        inner = envelope["verifier_payload"]
        self.assertEqual(
            [c["candidate_id"] for c in inner["candidates"]], order,
        )
        self.assertEqual(inner["transcript"], payload_b["transcript"])
        self.assertEqual(
            inner["metadata_fields_supplied"],
            payload_b["metadata_fields_supplied"],
        )
        self.assertIn("vocabulary_sha256", inner)

    def test_verifier_response_accepts_valid_selection(self) -> None:
        order = deterministic_candidate_order("p2k:case:0002")
        selected = validate_verifier_response(
            {
                "schema_version": VERIFIER_RESPONSE_SCHEMA_VERSION,
                "case_id": "p2k:case:0002",
                "selected_candidate_id": order[2],
                "rationale": "best grounded",
            },
            case_id="p2k:case:0002",
            candidate_order=order,
        )
        self.assertEqual(selected, order[2])

    def test_verifier_rejects_sixth_answer(self) -> None:
        order = deterministic_candidate_order("p2k:case:0003")
        with self.assertRaises(Exception):
            validate_verifier_response(
                {
                    "schema_version": VERIFIER_RESPONSE_SCHEMA_VERSION,
                    "case_id": "p2k:case:0003",
                    "selected_candidate_id": "candidate_6",
                    "rationale": "synthesized",
                },
                case_id="p2k:case:0003",
                candidate_order=order,
            )

    def test_verifier_rejects_malformed_outputs(self) -> None:
        order = deterministic_candidate_order("p2k:case:0004")
        bad_cases = [
            {"schema_version": "wrong", "case_id": "p2k:case:0004",
             "selected_candidate_id": order[0], "rationale": "x"},
            {"schema_version": VERIFIER_RESPONSE_SCHEMA_VERSION,
             "case_id": "other-case",
             "selected_candidate_id": order[0], "rationale": "x"},
            {"schema_version": VERIFIER_RESPONSE_SCHEMA_VERSION,
             "case_id": "p2k:case:0004",
             "selected_candidate_id": order[0], "rationale": ""},
            {"schema_version": VERIFIER_RESPONSE_SCHEMA_VERSION,
             "case_id": "p2k:case:0004",
             "selected_candidate_id": order[0]},
        ]
        for bad in bad_cases:
            with self.assertRaises(Exception):
                validate_verifier_response(
                    bad,
                    case_id="p2k:case:0004",
                    candidate_order=order,
                )

    def test_selection_integrity_enforced(self) -> None:
        from pipeline.phase2k_full_transcript_ablation import (
            import_intermediate_response,
        )

        pair = fixture()["payload_cases"][0]
        response = make_valid_intermediate_response(pair["B"])
        imported = import_intermediate_response(
            response, case_id=pair["case_id"], condition="B", payload=pair["B"],
        )
        check_selection_integrity(
            final_output=imported,
            candidate_outputs={"candidate_1": imported},
            selected_candidate_id="candidate_1",
        )
        tampered = json.loads(json.dumps(imported))
        tampered["fields"]["actors_entities"] = []
        with self.assertRaises(Exception):
            check_selection_integrity(
                final_output=tampered,
                candidate_outputs={"candidate_1": imported},
                selected_candidate_id="candidate_1",
            )


def make_reviews(condition: str, case_ids: list[str],
                 failures: dict[str, int] | None = None,
                 majors: set[str] | None = None,
                 ungrounded: set[str] | None = None) -> dict:
    failures = failures or {}
    majors = majors or set()
    ungrounded = ungrounded or set()
    reviews: dict[str, dict] = {}
    fail_iter = {case: count for case, count in failures.items()}
    for case_id in case_ids:
        for field in REVIEW_FIELDS:
            key = f"{case_id}:{condition}:{field}"
            remaining = fail_iter.get(case_id, 0)
            if remaining > 0 and field not in (
                "actors_entities",
                "reference_bindings",
                "abilities_resources",
                "explicit_relationships",
            ):
                reviews[key] = {
                    "correctness": "PARTIAL",
                    "unsupported_inference": "NONE",
                    "source_grounding": "GROUNDED",
                }
                fail_iter[case_id] = remaining - 1
                continue
            correctness = "CORRECT"
            unsupported = "MAJOR" if key in majors else "NONE"
            grounding = "UNGROUNDED" if key in ungrounded else "GROUNDED"
            reviews[key] = {
                "correctness": correctness,
                "unsupported_inference": unsupported,
                "source_grounding": grounding,
            }
    # If any failures remain undistributed (all key fields), mark them on
    # supporting_source_spans of the earliest cases.
    for case_id in case_ids:
        while fail_iter.get(case_id, 0) > 0:
            key = f"{case_id}:{condition}:supporting_source_spans"
            reviews[key] = {
                "correctness": "PARTIAL",
                "unsupported_inference": "NONE",
                "source_grounding": "GROUNDED",
            }
            fail_iter[case_id] -= 1
    return reviews


CASE_IDS = [f"p2k:case:{i:04d}" for i in range(1, 11)]


class GateBoundaryTests(unittest.TestCase):
    def _metrics(self, reviews: dict, condition: str = "F") -> dict:
        return compute_condition_metrics(
            reviews, condition=condition, case_ids=CASE_IDS,
        )

    def test_exact_pass_boundary_at_104(self) -> None:
        reviews = make_reviews("F", CASE_IDS, failures={
            case_id: 1 for case_id in CASE_IDS[:6]
        })
        total = sum(
            1 for entry in reviews.values()
            if entry["correctness"] in ("CORRECT", "ABSENT_CORRECTLY")
            and entry["unsupported_inference"] == "NONE"
            and entry["source_grounding"] in ("GROUNDED", "NOT_APPLICABLE")
        )
        self.assertEqual(total, 104)
        gate = evaluate_condition_gate(self._metrics(reviews))
        self.assertEqual(gate["outcome"], "PASS")

    def test_below_pass_boundary_not_pass(self) -> None:
        reviews = make_reviews("F", CASE_IDS, failures={
            case_id: 1 for case_id in CASE_IDS[:7]
        })
        gate = evaluate_condition_gate(self._metrics(reviews))
        self.assertEqual(gate["outcome"], "CONDITIONAL_PASS")

    def test_conditional_boundary_99_vs_fail_98(self) -> None:
        reviews_99 = make_reviews("F", CASE_IDS, failures={
            case_id: 1 for case_id in CASE_IDS
        } | {CASE_IDS[0]: 2})
        gate = evaluate_condition_gate(self._metrics(reviews_99))
        self.assertEqual(gate["outcome"], "CONDITIONAL_PASS")

        reviews_98 = make_reviews("F", CASE_IDS, failures=dict.fromkeys(
            CASE_IDS, 1,
        ))
        for field in ("recommended_advice", "consequences_outcomes"):
            reviews_98[f"{CASE_IDS[0]}:F:{field}"] = {
                "correctness": "PARTIAL",
                "unsupported_inference": "NONE",
                "source_grounding": "GROUNDED",
            }
        gate = evaluate_condition_gate(self._metrics(reviews_98))
        self.assertEqual(gate["outcome"], "FAIL")

    def test_major_unsupported_or_ungrounded_force_fail(self) -> None:
        base = make_reviews("F", CASE_IDS)
        with_major = dict(base)
        with_major[f"{CASE_IDS[0]}:F:states"] = {
            "correctness": "CORRECT",
            "unsupported_inference": "MAJOR",
            "source_grounding": "GROUNDED",
        }
        gate = evaluate_condition_gate(self._metrics(with_major))
        self.assertEqual(gate["outcome"], "FAIL")

        with_ungrounded = dict(base)
        with_ungrounded[f"{CASE_IDS[0]}:F:states"] = {
            "correctness": "CORRECT",
            "unsupported_inference": "NONE",
            "source_grounding": "UNGROUNDED",
        }
        gate = evaluate_condition_gate(self._metrics(with_ungrounded))
        self.assertEqual(gate["outcome"], "FAIL")


class VerifierUsefulnessGateTests(unittest.TestCase):
    def test_useful_when_all_checks_hold(self) -> None:
        f_reviews = make_reviews("F", CASE_IDS, failures=dict.fromkeys(
            CASE_IDS, 1,
        ))
        fv_reviews = make_reviews("FV", CASE_IDS)
        f_metrics = compute_condition_metrics(
            f_reviews, condition="F", case_ids=CASE_IDS,
        )
        fv_metrics = compute_condition_metrics(
            fv_reviews, condition="FV", case_ids=CASE_IDS,
        )
        result = evaluate_verifier_usefulness(
            flash_metrics=f_metrics, flash_verifier_metrics=fv_metrics,
        )
        self.assertEqual(result["decision"], "VERIFIER_SCALING_USEFUL")

    def test_not_justified_on_tiny_gain(self) -> None:
        f_reviews = make_reviews("F", CASE_IDS, failures={
            CASE_IDS[0]: 1, CASE_IDS[1]: 1, CASE_IDS[2]: 1,
        })
        fv_reviews = make_reviews("FV", CASE_IDS, failures={
            CASE_IDS[0]: 1, CASE_IDS[1]: 1,
        })
        f_metrics = compute_condition_metrics(
            f_reviews, condition="F", case_ids=CASE_IDS,
        )
        fv_metrics = compute_condition_metrics(
            fv_reviews, condition="FV", case_ids=CASE_IDS,
        )
        result = evaluate_verifier_usefulness(
            flash_metrics=f_metrics, flash_verifier_metrics=fv_metrics,
        )
        self.assertEqual(result["delta_strict_successes"], 1)
        self.assertEqual(
            result["decision"], "VERIFIER_SCALING_NOT_JUSTIFIED",
        )


class ProductionSelectionTests(unittest.TestCase):
    def test_preference_and_cost_logic(self) -> None:
        passing = {c: {"outcome": "PASS"} for c in ("F", "FV", "P")}
        result = select_production_model(gates=passing)
        self.assertEqual(result["recommendation"], "V4_FLASH_SINGLE_PASS_PROMOTED")
        result = select_production_model(
            gates=passing,
            cost_per_target_seconds={"F": 100.0, "FV": 30.0, "P": 90.0},
        )
        self.assertEqual(result["recommendation"], "V4_FLASH_VERIFIER_PROMOTED")
        none_pass = {c: {"outcome": "FAIL"} for c in ("F", "FV", "P")}
        result = select_production_model(gates=none_pass)
        self.assertEqual(
            result["recommendation"],
            "NO_DEEPSEEK_CONFIGURATION_MEETS_PRODUCTION_GATE",
        )

    def test_conditional_pass_is_never_auto_promoted(self) -> None:
        conditional_only = {
            "F": {"outcome": "FAIL"},
            "FV": {"outcome": "CONDITIONAL_PASS"},
            "P": {"outcome": "CONDITIONAL_PASS"},
        }
        result = select_production_model(
            gates=conditional_only,
            cost_per_target_seconds={"FV": 30.0, "P": 90.0},
        )
        self.assertEqual(
            result["recommendation"],
            "NO_DEEPSEEK_CONFIGURATION_MEETS_PRODUCTION_GATE",
        )
        self.assertEqual(result["conditional_candidates"], ["FV", "P"])


class ResumeFlowTests(unittest.TestCase):
    def test_resume_does_not_rerun_and_appends_single_verifier(self) -> None:
        data = fixture()
        workspace = Path(tempfile.mkdtemp(prefix="p2k-resume-"))
        run_dir = workspace / "FV"
        with mock.patch.object(
            base_script, "DEFAULT_MANIFEST_PATH", data["manifest_path"],
        ):
            sel_script.cmd_init(cli_args(condition="FV", run_dir=str(run_dir)))
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            # Simulate all 50 candidate calls completing with valid raws.
            pairs = {pair["case_id"]: pair for pair in data["payload_cases"]}
            for call in manifest["calls"]:
                payload_b = pairs[call["case_id"]]["B"]
                raw = json.dumps(make_valid_intermediate_response(payload_b))
                raw_file = sel_script.raw_path_for(run_dir, call["call_id"])
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                raw_file.write_text(raw, encoding="utf-8")
                call.update({
                    "status": "completed",
                    "raw_path": str(raw_file.relative_to(run_dir)),
                    "raw_response_sha256": file_sha256(raw_file),
                })
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            original_hashes = {
                call["call_id"]: call["raw_response_sha256"]
                for call in manifest["calls"]
            }

            def execute_impl(self_transport, *, model, prompt_bytes,
                             mode="auto"):
                text = prompt_bytes.decode("utf-8")
                wrapper = sel_script.VERIFIER_WRAPPER_PROMPT
                if text.startswith(wrapper):
                    body = json.loads(text[len(wrapper) + 2:])
                    case_id = body["verifier_payload"]["case_id"]
                    presented = [
                        c["candidate_id"]
                        for c in body["verifier_payload"]["candidates"]
                    ]
                    self.assertEqual(len(presented), CANDIDATE_COUNT)
                    return json.dumps({
                        "schema_version": VERIFIER_RESPONSE_SCHEMA_VERSION,
                        "case_id": case_id,
                        "selected_candidate_id": presented[0],
                        "rationale": "first preserves semantics best",
                    }), "opencode_cli"
                raise AssertionError("generator prompts must not re-execute")

            with mock.patch.object(
                base_script, "DEFAULT_MANIFEST_PATH", data["manifest_path"],
            ), mock.patch.object(
                sel_script.ModelTransport, "execute", execute_impl,
            ):
                sel_script.cmd_run(cli_args(
                    condition="FV", run_dir=str(run_dir), max_workers=1,
                ))

            refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
            verifier_calls = [
                c for c in refreshed["calls"]
                if c.get("role") == "verifier"
            ]
            self.assertEqual(len(verifier_calls), 10)
            self.assertTrue(all(
                c.get("status") == "completed" for c in verifier_calls
            ))
            # Candidate calls were not re-executed: raw hashes unchanged.
            for call in refreshed["calls"]:
                if call.get("candidate_id") is not None:
                    self.assertEqual(
                        file_sha256(run_dir / call["raw_path"]),
                        original_hashes[call["call_id"]],
                    )
            # Candidate presentation order recorded and deterministic.
            for call in verifier_calls:
                self.assertEqual(
                    call["candidate_order"],
                    deterministic_candidate_order(call["case_id"]),
                )


if __name__ == "__main__":
    unittest.main()
