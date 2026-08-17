"""Phase 2G silver representations for the locked Phase 2F legacy benchmark.

This module owns the reviewed ``data/phase2g_silver_v1.json`` fixture and the
deterministic validation of its two non-bronze conditions:

* ``MECHANICAL_SILVER`` -- a reversible linguistic cleanup of the immutable
  bronze text.  Every output fragment is either an exact bronze span kept or
  rewritten, a deletion of an exact bronze span, or an insertion at an exact
  bronze character anchor.  The cleanup is limited to capitalization,
  punctuation/sentence/clause repair, semantically irrelevant filler/duplicate
  removal, and unambiguous ASR spelling corrections.  There is no pronoun
  resolution and no League strategic ontology.
* ``RESOLVED_SILVER`` -- the mechanical representation plus high-confidence
  linguistic reference resolution (singular "you" -> the coached player,
  "you guys" -> the coached player's team, "your" -> the coached player's).
  Every resolution op retains its bronze span, prior bronze text, the
  mechanical text at that span, the resolved text, transformation type,
  confidence, and alternatives.  Ambiguous references stay unchanged with
  alternatives.  No strategic concepts are introduced.

The module never modifies bronze; ``RAW_BRONZE`` is always the benchmark's
exact embedded source text.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping


SILVER_SCHEMA_VERSION = "phase2g-silver-v1"
SILVER_FIXTURE_CONTENT_SHA256 = "4ae3f1bd167f1bebb27ce3d27118833d7c869bb28c4b076d98735182a9fb5a41"
BENCHMARK_CONTENT_SHA256 = "a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd"

RAW_BRONZE = "RAW_BRONZE"
MECHANICAL_SILVER = "MECHANICAL_SILVER"
RESOLVED_SILVER = "RESOLVED_SILVER"
CONDITIONS = (RAW_BRONZE, MECHANICAL_SILVER, RESOLVED_SILVER)

# Pass 1 must not contain League strategic concepts.  This is the audited
# banned list used for silver leakage validation.
FORBIDDEN_STRATEGIC_CONCEPTS = (
    "access", "continuity", "initiative", "tempo", "priority", "conversion",
    "wave obligation", "wave_obligation", "power spike", "win condition",
    "front to back",
)

_FRAGMENT_KINDS = frozenset({"source", "changed", "insertion"})
_RESOLUTION_TYPES = frozenset({"PRONOUN_RESOLUTION", "AMBIGUOUS_RETAINED"})


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _require(value: object, label: str, kind: type) -> None:
    if not isinstance(value, kind):
        raise ValueError(f"{label} must be {kind.__name__}")


class Phase2GSilverError(ValueError):
    """The silver fixture or a derived representation violates an invariant."""


def load_silver_fixture(path: str | Path) -> Mapping[str, Any]:
    """Load and validate the locked silver fixture."""
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, Mapping):
        raise Phase2GSilverError("silver fixture must be a JSON object")
    if body.get("content_sha256") != SILVER_FIXTURE_CONTENT_SHA256:
        raise Phase2GSilverError(
            "silver fixture content does not match the preregistered lock",
        )
    inner = {key: value for key, value in body.items() if key != "content_sha256"}
    if canonical_sha256(inner) != body["content_sha256"]:
        raise Phase2GSilverError("silver fixture content hash does not self-verify")
    if body.get("schema_version") != SILVER_SCHEMA_VERSION:
        raise Phase2GSilverError("silver fixture schema version is unsupported")
    if body.get("benchmark_content_sha256") != BENCHMARK_CONTENT_SHA256:
        raise Phase2GSilverError("silver fixture is not bound to the locked benchmark")
    cases = body.get("cases")
    if not isinstance(cases, Mapping) or len(cases) != 5:
        raise Phase2GSilverError("silver fixture must cover exactly five cases")
    for case_id, case in cases.items():
        _validate_case_record(case_id, case)
    return body


def _validate_case_record(case_id: str, case: Mapping[str, Any]) -> None:
    if not isinstance(case, Mapping):
        raise Phase2GSilverError(f"{case_id}: case record must be an object")
    mechanical = case.get("mechanical")
    resolved = case.get("resolved")
    if not isinstance(mechanical, Mapping) or not isinstance(resolved, Mapping):
        raise Phase2GSilverError(f"{case_id}: mechanical/resolved records required")
    fragments = mechanical.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise Phase2GSilverError(f"{case_id}: mechanical fragments required")
    for fragment in fragments:
        _validate_fragment(case_id, fragment)
    text = mechanical.get("text")
    if not isinstance(text, str):
        raise Phase2GSilverError(f"{case_id}: mechanical text required")
    if mechanical.get("sha256") != hashlib.sha256(text.encode()).hexdigest():
        raise Phase2GSilverError(f"{case_id}: mechanical sha256 does not match text")
    ops = resolved.get("resolution_ops")
    if not isinstance(ops, list) or not ops:
        raise Phase2GSilverError(f"{case_id}: resolution ops required")
    for op in ops:
        _validate_resolution_op(case_id, op)
    resolved_text = resolved.get("text")
    if not isinstance(resolved_text, str):
        raise Phase2GSilverError(f"{case_id}: resolved text required")
    if resolved.get("sha256") != hashlib.sha256(resolved_text.encode()).hexdigest():
        raise Phase2GSilverError(f"{case_id}: resolved sha256 does not match text")


def _validate_fragment(case_id: str, fragment: Mapping[str, Any]) -> None:
    kind = fragment.get("kind")
    if kind not in _FRAGMENT_KINDS:
        raise Phase2GSilverError(f"{case_id}: unknown fragment kind {kind!r}")
    _require(fragment.get("reason"), f"{case_id} fragment reason", str)
    if kind == "insertion":
        anchor = fragment.get("anchor")
        if isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 0:
            raise Phase2GSilverError(f"{case_id}: invalid insertion anchor")
        _require(fragment.get("text"), f"{case_id} insertion text", str)
        return
    start = fragment.get("start")
    end = fragment.get("end")
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or not 0 <= start < end
    ):
        raise Phase2GSilverError(f"{case_id}: invalid fragment span")
    if kind == "changed":
        _require(fragment.get("text"), f"{case_id} changed text", str)


def _validate_resolution_op(case_id: str, op: Mapping[str, Any]) -> None:
    span = op.get("bronze_span")
    if (
        not isinstance(span, list) or len(span) != 2
        or isinstance(span[0], bool) or isinstance(span[1], bool)
        or not isinstance(span[0], int) or not isinstance(span[1], int)
        or not 0 <= span[0] < span[1]
    ):
        raise Phase2GSilverError(f"{case_id}: invalid resolution bronze span")
    for label in ("prior_text", "mechanical_text", "resolved_text"):
        _require(op.get(label), f"{case_id} {label}", str)
    if op.get("transformation_type") not in _RESOLUTION_TYPES:
        raise Phase2GSilverError(f"{case_id}: unknown resolution type")
    confidence = op.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
            or not 0.0 <= float(confidence) <= 1.0:
        raise Phase2GSilverError(f"{case_id}: resolution confidence must be in [0, 1]")
    alternatives = op.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        raise Phase2GSilverError(f"{case_id}: resolution alternatives required")
    for alternative in alternatives:
        if not isinstance(alternative, Mapping) or not isinstance(alternative.get("text"), str):
            raise Phase2GSilverError(f"{case_id}: invalid resolution alternative")
        alt_confidence = alternative.get("confidence")
        if (
            isinstance(alt_confidence, bool)
            or not isinstance(alt_confidence, (int, float))
            or not 0.0 <= float(alt_confidence) <= 1.0
        ):
            raise Phase2GSilverError(f"{case_id}: invalid alternative confidence")


def _render_mechanical(
    bronze: str, fragments: list[Mapping[str, Any]],
) -> tuple[str, list[tuple[int, int, int, int, int]]]:
    """Render mechanical text with whitespace-run collapsing.

    Returns (text, per-non-insertion-fragment intervals
    (bronze_start, bronze_end, mech_start, mech_end, leading_strip)).
    """
    out: list[str] = []
    intervals: list[tuple[int, int, int, int, int]] = []
    pos = 0
    for fragment in fragments:
        kind = fragment["kind"]
        if kind == "insertion":
            text = fragment["text"]
        else:
            text = fragment["text"] if kind == "changed" else bronze[fragment["start"]:fragment["end"]]
        if text:
            if out and out[-1].endswith(" ") and text.startswith(" "):
                text = text[1:]
            text = re.sub(r" {2,}", " ", text)
            if out and out[-1].endswith(" ") and text.startswith(" "):
                text = text[1:]
        if text:
            out.append(text)
            if kind != "insertion":
                strip = 0 if kind == "changed" else max(
                    0, (fragment["end"] - fragment["start"]) - len(text),
                )
                if strip:
                    if bronze[fragment["start"] + strip:fragment["end"]] != text:
                        raise Phase2GSilverError(
                            "unsupported interior collapse in fragment",
                        )
                intervals.append(
                    (fragment["start"], fragment["end"], pos, pos + len(text), strip),
                )
            pos += len(text)
        elif kind != "insertion":
            intervals.append((fragment["start"], fragment["end"], pos, pos, 0))
    return "".join(out), intervals


def _mech_interval(
    intervals: list[tuple[int, int, int, int, int]], start: int, end: int,
) -> tuple[int, int]:
    start_frag = end_frag = None
    for bs, be, ms0, me0, strip in intervals:
        if bs <= start < be:
            start_frag = (bs, be, ms0, me0, strip)
        if bs <= end - 1 < be:
            end_frag = (bs, be, ms0, me0, strip)
    if start_frag is None or end_frag is None:
        raise Phase2GSilverError("resolution span has no mechanical alignment")
    bs, be, ms0, me0, strip = start_frag
    ms = ms0 if start == bs else ms0 + (start - bs) - strip
    bs, be, ms0, me0, strip = end_frag
    me = me0 if end == be else ms0 + (end - bs) - strip
    if not ms < me:
        raise Phase2GSilverError("resolution span maps to an empty mechanical interval")
    return ms, me


def _letters_preserving(old: str, new: str) -> bool:
    def norm(value: str) -> str:
        return "".join(
            char.lower() for char in unicodedata.normalize("NFKC", value)
            if char.isalnum()
        )
    return norm(old) == norm(new)


def banned_concept_hits(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(concept for concept in FORBIDDEN_STRATEGIC_CONCEPTS if concept in lowered)


def validate_silver_fixture(benchmark: Mapping[str, Any]) -> None:
    """Validate the full silver fixture against the locked benchmark."""
    fixture_path = _default_fixture_path()
    fixture = load_silver_fixture(fixture_path)
    validate_fixture_against_benchmark(benchmark, fixture)


def validate_fixture_against_benchmark(
    benchmark: Mapping[str, Any], fixture: Mapping[str, Any],
) -> None:
    """Validate every fixture invariant against the benchmark records."""
    if benchmark.get("content_sha256") != BENCHMARK_CONTENT_SHA256:
        raise Phase2GSilverError("benchmark is not the locked Phase 2F legacy benchmark")
    cases = {item["id"]: item for item in benchmark["cases"]}
    if set(cases) != set(fixture["cases"]):
        raise Phase2GSilverError("silver fixture case set does not match the benchmark")
    for case_id, case in cases.items():
        record = fixture["cases"][case_id]
        bronze = case["source_text"]
        fragments = record["mechanical"]["fragments"]
        non_insertions = [
            fragment for fragment in fragments if fragment["kind"] != "insertion"
        ]
        if not non_insertions:
            raise Phase2GSilverError(f"{case_id}: no source/rewrite fragments")
        if non_insertions[0]["start"] != 0:
            raise Phase2GSilverError(
                f"{case_id}: mechanical fragments must start at bronze offset 0",
            )
        if any(
            left["end"] != right["start"]
            for left, right in zip(non_insertions, non_insertions[1:])
        ):
            raise Phase2GSilverError(
                f"{case_id}: mechanical fragments must cover bronze contiguously",
            )
        if non_insertions[-1]["end"] != len(bronze):
            raise Phase2GSilverError(
                f"{case_id}: mechanical fragments must end at bronze offset {len(bronze)}",
            )
        previous_anchor = -1
        for fragment in fragments:
            if fragment["kind"] == "insertion":
                anchor = fragment["anchor"]
                if anchor < 0 or anchor > len(bronze):
                    raise Phase2GSilverError(
                        f"{case_id}: insertion anchor {anchor} is outside the bronze",
                    )
                if anchor < previous_anchor:
                    raise Phase2GSilverError(
                        f"{case_id}: insertion anchors must be monotonic",
                    )
                previous_anchor = anchor
            else:
                if not 0 <= fragment["start"] < fragment["end"] <= len(bronze):
                    raise Phase2GSilverError(
                        f"{case_id}: fragment span is outside the bronze",
                    )
        mechanical_text, intervals = _render_mechanical(
            bronze, fragments,
        )
        if mechanical_text != record["mechanical"]["text"]:
            raise Phase2GSilverError(f"{case_id}: mechanical text does not reconstruct")
        reverse = "".join(
            bronze[fragment["start"]:fragment["end"]]
            for fragment in fragments
            if fragment["kind"] != "insertion"
        )
        if reverse != bronze:
            raise Phase2GSilverError(f"{case_id}: mechanical cleanup is not reversible")
        hits = banned_concept_hits(mechanical_text)
        if hits:
            raise Phase2GSilverError(
                f"{case_id}: mechanical silver leaks strategic concepts {hits}",
            )
        gold_spans = {
            tuple(span) for mention in case["mentions"] for span in mention["acceptable_spans"]
        }
        for fragment in record["mechanical"]["fragments"]:
            if fragment["kind"] != "changed":
                continue
            if _letters_preserving(bronze[fragment["start"]:fragment["end"]], fragment["text"]):
                continue
            for (gs, ge) in gold_spans:
                if not (fragment["end"] <= gs or fragment["start"] >= ge):
                    raise Phase2GSilverError(
                        f"{case_id}: content-changing edit overlaps a gold span",
                    )
        ops = record["resolved"]["resolution_ops"]
        ordered = sorted(ops, key=lambda op: (op["bronze_span"][0], op["bronze_span"][1]))
        if ordered != ops:
            raise Phase2GSilverError(f"{case_id}: resolution ops must be bronze-sorted")
        for left, right in zip(ops, ops[1:]):
            if left["bronze_span"][1] > right["bronze_span"][0]:
                raise Phase2GSilverError(f"{case_id}: resolution ops overlap")
        resolved_parts = []
        last = 0
        for op in ops:
            s, e = op["bronze_span"]
            if bronze[s:e] != op["prior_text"]:
                raise Phase2GSilverError(f"{case_id}: resolution prior text is not bronze")
            ms, me = _mech_interval(intervals, s, e)
            if mechanical_text[ms:me] != op["mechanical_text"]:
                raise Phase2GSilverError(
                    f"{case_id}: resolution mechanical text does not align",
                )
            resolved_parts.append(mechanical_text[last:ms])
            resolved_parts.append(op["resolved_text"])
            last = me
        resolved_parts.append(mechanical_text[last:])
        resolved_text = "".join(resolved_parts)
        if resolved_text != record["resolved"]["text"]:
            raise Phase2GSilverError(f"{case_id}: resolved text does not reconstruct")
        hits = banned_concept_hits(resolved_text)
        if hits:
            raise Phase2GSilverError(
                f"{case_id}: resolved silver leaks strategic concepts {hits}",
            )


def _default_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "phase2g_silver_v1.json"


def condition_text(
    benchmark_case: Mapping[str, Any], condition: str,
    fixture: Mapping[str, Any] | None = None,
) -> str:
    """Return the deterministic condition representation text for a case."""
    if condition == RAW_BRONZE:
        return benchmark_case["source_text"]
    if condition not in (MECHANICAL_SILVER, RESOLVED_SILVER):
        raise ValueError(f"unknown Phase 2G condition: {condition!r}")
    if fixture is None:
        fixture = load_silver_fixture(_default_fixture_path())
    return fixture["cases"][benchmark_case["id"]][
        "mechanical" if condition == MECHANICAL_SILVER else "resolved"
    ]["text"]


def silver_input_record(
    benchmark_case: Mapping[str, Any], condition: str,
    fixture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the retained input representation for one condition/case.

    Mechanical and resolved silver inputs both retain the underlying bronze-
    anchored fragments/transformations; resolved inputs additionally retain the
    resolution operations.  Raw bronze retains only the immutable text.
    """
    if fixture is None:
        fixture = load_silver_fixture(_default_fixture_path())
    text = condition_text(benchmark_case, condition, fixture)
    record: dict[str, Any] = {
        "condition": condition,
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if condition in (MECHANICAL_SILVER, RESOLVED_SILVER):
        record["fragments"] = fixture["cases"][benchmark_case["id"]]["mechanical"]["fragments"]
        record["transformations"] = [
            {
                "kind": "REWRITE" if fragment["kind"] == "changed" else fragment["kind"].upper(),
                "bronze_span": (
                    [fragment["anchor"], fragment["anchor"]]
                    if fragment["kind"] == "insertion"
                    else [fragment["start"], fragment["end"]]
                ),
                "bronze_text": "" if fragment["kind"] == "insertion"
                else benchmark_case["source_text"][fragment["start"]:fragment["end"]],
                "silver_text": fragment.get("text", ""),
                "reason": fragment["reason"],
            }
            for fragment in fixture["cases"][benchmark_case["id"]]["mechanical"]["fragments"]
        ]
        if condition == RESOLVED_SILVER:
            record["resolution_ops"] = fixture["cases"][benchmark_case["id"]]["resolved"]["resolution_ops"]
    return record
