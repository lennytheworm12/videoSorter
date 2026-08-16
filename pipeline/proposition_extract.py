"""Span-first source-grounded proposition extraction before candidate mapping.

Phase 2E deliberately separates evidence localization, source-semantic slot
recovery, causal-direction classification, and ontology normalization.  Every
source slot is validated against a verified source before deterministic code
may assemble a :class:`GroundedProposition`.  Selected evidence, recovered
slots, and the causal direction survive any later stage failure so artifacts
can diagnose the first loss boundary.  A normalization abstention or failure
never discards a successfully recovered source-semantic frame; malformed or
unsupported model output fails closed and never reaches candidate mapping or
persistence.

Failures carry a deterministic taxonomy.  An ``UnsupportedSourceSlot`` means
model text could not be exactly grounded in the selected evidence, an
``InventedOntologyContent`` means normalization returned an ID outside the
closed ontology, and an ordinary ``ValueError`` means malformed or partial
model output.  A provider exception raised by a model call fails closed at
that stage and is recorded with the ``ProviderCallError`` marker and no raw
output, so it is never conflated with malformed or unsupported model output.
:class:`StageAExtraction` exposes explicit counts
(``unsupported_slot_count`` and ``invented_ontology_count``), an
``invented_ontology_taxonomy`` mapping each invented ID to its occurrence
count, plus the exception class name in artifacts and frame markers, so
evaluators never conflate unsupported source slots, invented closed-ontology
IDs, and malformed output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Callable, Literal, Mapping

from core.ontology import RELATION_TYPES, STRATEGIC_CONCEPTS
from pipeline.relation_extract import GroundedProposition
from pipeline.source_windows import SourceWindow


SourceMode = Literal["insight", "transcript", "combined"]
CausalDirection = Literal[
    "actor_event_causes_effect",
    "effect_causes_actor_event",
    "association_only",
    "temporal_sequence_only",
    "insufficient_causal_claim",
]
_VALID_MODES = frozenset(("insight", "transcript", "combined"))
_VALID_DIRECTIONS = frozenset(
    (
        "actor_event_causes_effect",
        "effect_causes_actor_event",
        "association_only",
        "temporal_sequence_only",
        "insufficient_causal_claim",
    )
)

SPAN_FIRST_PROMPT_VERSION = "phase2e-clause-first-v2"

_CLAUSE_MAX_TOKENS = 32
_CLAUSE_WINDOW_STRIDE = 16
_CLAUSE_CATALOG_LIMIT = 20
_NONSPACE_RE = re.compile(r"\S+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;])\s+")
_DISCOURSE_BOUNDARY_RE = re.compile(
    r"(?i)(?<![\w'’])(?:when|if|because|unless|while|after|before|once|until|"
    r"whenever|although|though|whereas|however|therefore|but|so|then)(?=\s)"
)

EVIDENCE_LOCALIZATION_SYSTEM = """Return JSON only. Select the smallest one
or two supplied candidate IDs that together state one coaching mechanism. A
mechanism contains an action, resource, or state and a supported consequence
or opportunity. Include its trigger or condition when present. Recommendations
such as "do X to achieve Y" contain a mechanism. Select IDs from one source
only. Never quote, paraphrase, generate source text, or provide character
offsets. Do not interpret ontology concepts. Return no selection when no
candidate or linked candidate pair states a causal mechanism."""

SLOT_SYSTEMS = {
    "actor": """Return JSON only. From the selected exact evidence, copy the
source phrase naming the causal actor, resource, state, or player action. It is
the thing or action whose event produces the consequence, not the affected
result. Do not paraphrase or normalize.""",
    "event": """Return JSON only. From the selected exact evidence, copy the
smallest source phrase naming what the causal actor does or what happens to it.
This is the causal action, event, or state change linking the actor to the
consequence. Do not copy the consequence and do not paraphrase or normalize.""",
    "effect": """Return JSON only. From the selected exact evidence, copy the
source phrase naming the supported consequence, result, or opportunity. Do
not copy the causal actor as the effect and do not paraphrase or normalize.""",
    "condition": """Return JSON only. Decide whether the selected mechanism
has a source-stated trigger, qualifier, timing, or prerequisite. If present,
copy its exact source phrase. If absent, return null or the string "NONE".
Never invent a condition and do not turn the consequence into a condition
merely to fill the field.""",
}

DIRECTION_SYSTEM = """Return JSON only. Classify the causal direction of the
supplied source-selected actor, event, effect, and optional condition. Choose
exactly one allowed label. actor_event_causes_effect means the actor/event
produces or enables the effect. effect_causes_actor_event means the proposed
roles are reversed. Association and temporal sequence are not causation.
Choose insufficient_causal_claim when the evidence does not support a causal
claim. Do not rewrite any source field."""

NORMALIZATION_SYSTEM = """Return JSON only. Classify the already recovered
source-semantic fields against the supplied closed ontology. This is a
separate normalization step: do not change source text and do not add facts.
Choose a canonical ID only when the source phrase directly supports it;
otherwise return null. Null is preferred to an uncertain mapping."""

# Retained legacy one-pass contract for parse_grounded_propositions callers
# and older Phase 2D artifacts. Live Stage A uses the decomposed systems.
PROPOSITION_SYSTEM = """Return JSON only. Extract source-grounded causal
propositions from supplied source text. Do not select ontology concepts,
canonical relation types, or canonical entities. Return zero propositions for
advice without a stated or clearly entailed causal mechanism. Each non-null
field must copy one supplied source span exactly, including wording and tense.
Do not paraphrase. The system derives character spans; return only the source
label that contains each exact quoted field. A coaching recommendation is a
causal proposition when the same supplied text states both an action/resource
and its strategic effect (for example, "use X to remove Y"). In that case,
quote the actor/action as subject, the stated causal phrase as predicate, and
the affected result as effect. Do not return zero merely because the source
uses "should", "want to", or "to" rather than "because". Return zero for
advice that lacks a stated or directly entailed mechanism."""


@dataclass(frozen=True)
class PropositionSource:
    kind: Literal["insight", "transcript"]
    text: str


@dataclass(frozen=True)
class PropositionPacket:
    evidence_id: str
    source_video_id: str
    insight_text: str
    mode: SourceMode
    source_window: SourceWindow | None = None

    def validate(self) -> None:
        if not self.evidence_id or not self.source_video_id or not self.insight_text.strip():
            raise ValueError("proposition packet requires insight identity and text")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"unknown source mode: {self.mode}")
        if self.mode in {"transcript", "combined"}:
            if self.source_window is None or not self.source_window.resolved:
                raise ValueError("transcript source mode requires a verified source window")
            if self.source_window.evidence_id != self.evidence_id or self.source_window.source_video_id != self.source_video_id:
                raise ValueError("source window does not belong to proposition evidence")

    def sources(self) -> tuple[PropositionSource, ...]:
        self.validate()
        values = []
        if self.mode in {"insight", "combined"}:
            values.append(PropositionSource("insight", self.insight_text))
        if self.mode in {"transcript", "combined"}:
            assert self.source_window and self.source_window.transcript_window
            values.append(PropositionSource("transcript", self.source_window.transcript_window))
        return tuple(values)

    def prompt(self) -> str:
        """Render the source-only packet used by evidence localization."""
        rendered = "\n\n".join(f"[{source.kind}]\n{source.text}" for source in self.sources())
        source_kinds = [source.kind for source in self.sources()]
        return (
            "EVIDENCE ID: " + self.evidence_id + "\nSOURCE TEXT:\n" + rendered
            + "\n\nAllowed source values: " + json.dumps(source_kinds)
            + '. Return exactly {"source":"<allowed source value>","evidence_spans":["exact source span"]}'
            + ' or {"source":null,"evidence_spans":[]}.'
        )


@dataclass(frozen=True)
class SourceAlignment:
    source_kind: Literal["insight", "transcript"]
    start: int
    end: int
    source_text: str
    absolute_start: int | None = None
    absolute_end: int | None = None


@dataclass(frozen=True)
class ClauseCandidate:
    """A deterministic source-local semantic unit offered by stable ID."""

    candidate_id: str
    alignment: SourceAlignment


@dataclass(frozen=True)
class PropositionAlignment:
    field: Literal["subject", "predicate", "effect", "condition"]
    source_kind: Literal["insight", "transcript"]
    start: int
    end: int
    source_text: str
    absolute_start: int | None = None
    absolute_end: int | None = None


@dataclass(frozen=True)
class SemanticSlot:
    role: Literal["actor", "event", "effect", "condition"]
    alignment: SourceAlignment

    @property
    def text(self) -> str:
        return self.alignment.source_text


@dataclass(frozen=True)
class OntologyNormalization:
    actor_concept: str | None
    event_relation: str | None
    effect_concept: str | None


@dataclass(frozen=True)
class SourceSemanticFrame:
    evidence_spans: tuple[SourceAlignment, ...]
    actor: SemanticSlot
    event: SemanticSlot
    effect: SemanticSlot
    condition: SemanticSlot | None
    causal_direction: CausalDirection
    normalization: OntologyNormalization | None
    normalization_failure: str | None = None


@dataclass(frozen=True)
class ExtractedProposition:
    proposition: GroundedProposition
    alignments: tuple[PropositionAlignment, ...]


@dataclass(frozen=True)
class StageArtifact:
    stage: str
    raw_output: str | None
    parsed_output: Mapping[str, Any] | None
    failure: str | None = None


@dataclass(frozen=True)
class StageAExtraction:
    propositions: tuple[ExtractedProposition, ...]
    frames: tuple[SourceSemanticFrame, ...]
    artifacts: tuple[StageArtifact, ...]
    failure_stage: str | None = None
    unsupported_slot_count: int = 0
    invented_ontology_count: int = 0
    evidence_spans: tuple[SourceAlignment, ...] = ()
    slots: Mapping[str, SemanticSlot | None] = field(default_factory=dict)
    causal_direction: CausalDirection | None = None
    invented_ontology_taxonomy: Mapping[str, int] = field(default_factory=dict)
    candidate_catalog: tuple[ClauseCandidate, ...] = ()

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "prompt_version": SPAN_FIRST_PROMPT_VERSION,
            "candidate_catalog": [asdict(candidate) for candidate in self.candidate_catalog],
            "failure_stage": self.failure_stage,
            "unsupported_slot_count": self.unsupported_slot_count,
            "invented_ontology_count": self.invented_ontology_count,
            "invented_ontology_taxonomy": dict(self.invented_ontology_taxonomy),
            "raw_stage_outputs": [asdict(item) for item in self.artifacts],
            "selected_evidence_spans": [asdict(span) for span in self.evidence_spans],
            "recovered_slots": [
                {"role": role, "slot": asdict(slot) if slot is not None else None}
                for role, slot in self.slots.items()
            ],
            "causal_direction": self.causal_direction,
            "semantic_frames": [asdict(frame) for frame in self.frames],
            "assembled_propositions": [
                {"proposition": asdict(item.proposition), "alignments": [asdict(value) for value in item.alignments]}
                for item in self.propositions
            ],
        }


class UnsupportedSourceSlot(ValueError):
    """A model-selected slot was not uniquely grounded in selected evidence."""


class InventedOntologyContent(ValueError):
    """Normalization returned one or more IDs outside the closed ontology."""

    def __init__(self, message: str, invented: Mapping[str, int]) -> None:
        super().__init__(message)
        self.invented = dict(invented)

    @property
    def count(self) -> int:
        return sum(self.invented.values())


class ProviderCallError(Exception):
    """A model provider call failed before producing raw output.

    This is distinct from parse failures (``ValueError`` and its subclasses):
    the failing :class:`StageArtifact` records no raw output and carries the
    ``ProviderCallError`` marker so evaluators never conflate a provider
    outage with malformed, unsupported, or invented model output.
    """


def extract_span_first_propositions(
    packet: PropositionPacket,
    chat: Callable[..., str],
    *,
    model: str | None = None,
    max_tokens: int = 512,
    thinking: str | None = None,
) -> StageAExtraction:
    """Run Phase 2E's observable, fail-closed Stage A pipeline.

    Selected evidence and every successfully parsed slot are retained even
    when a later stage fails.  A normalization failure retains the recovered
    semantic frame with ``normalization=None`` and a failure marker instead of
    discarding the frame; the stage name in ``failure_stage`` keeps extraction
    failures distinguishable from normalization failures.  A provider
    exception raised by a model call fails closed at the current stage,
    retaining every already completed artifact, selected evidence span,
    recovered slot, and the causal direction when already classified; the
    failing artifact records no raw output with a ``ProviderCallError`` marker.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    packet.validate()
    artifacts: list[StageArtifact] = []
    candidates = enumerate_clause_candidates(packet)

    try:
        raw = _provider_call(
            chat, EVIDENCE_LOCALIZATION_SYSTEM,
            clause_evidence_prompt(packet, candidates), model, max_tokens, thinking,
        )
        spans, parsed = parse_candidate_evidence_selection(raw, candidates, packet)
    except ProviderCallError as exc:
        artifacts.append(StageArtifact("evidence_localization", None, None, type(exc).__name__))
        return StageAExtraction(
            (), (), tuple(artifacts), failure_stage="evidence_localization",
            candidate_catalog=candidates,
        )
    except Exception as exc:
        artifacts.append(StageArtifact("evidence_localization", raw, None, type(exc).__name__))
        return StageAExtraction(
            (), (), tuple(artifacts), failure_stage="evidence_localization",
            unsupported_slot_count=_unsupported_count(exc),
            candidate_catalog=candidates,
        )
    artifacts.append(StageArtifact("evidence_localization", raw, parsed))
    if not spans:
        return StageAExtraction((), (), tuple(artifacts), candidate_catalog=candidates)

    slots: dict[str, SemanticSlot | None] = {}
    for role in ("actor", "event", "effect", "condition"):
        user = _slot_prompt(role, spans, slots)
        try:
            raw = _provider_call(chat, SLOT_SYSTEMS[role], user, model, max_tokens, thinking)
            slot, parsed = parse_semantic_slot(raw, role, spans, packet)
        except ProviderCallError as exc:
            artifacts.append(StageArtifact(role + "_extraction", None, None, type(exc).__name__))
            return StageAExtraction(
                (), (), tuple(artifacts), failure_stage=role + "_extraction",
                evidence_spans=spans, slots=dict(slots),
                candidate_catalog=candidates,
            )
        except Exception as exc:
            artifacts.append(StageArtifact(role + "_extraction", raw, None, type(exc).__name__))
            return StageAExtraction(
                (), (), tuple(artifacts), failure_stage=role + "_extraction",
                unsupported_slot_count=_unsupported_count(exc),
                evidence_spans=spans, slots=dict(slots),
                candidate_catalog=candidates,
            )
        artifacts.append(StageArtifact(role + "_extraction", raw, parsed))
        slots[role] = slot
        if role != "condition" and slot is None:
            return StageAExtraction(
                (), (), tuple(artifacts), failure_stage=role + "_extraction",
                evidence_spans=spans, slots=dict(slots),
                candidate_catalog=candidates,
            )

    actor = slots["actor"]
    event = slots["event"]
    effect = slots["effect"]
    condition = slots["condition"]
    assert actor is not None and event is not None and effect is not None

    try:
        raw = _provider_call(chat, DIRECTION_SYSTEM, _direction_prompt(spans, actor, event, effect, condition), model, max_tokens, thinking)
        direction, parsed = parse_causal_direction(raw)
    except ProviderCallError as exc:
        artifacts.append(StageArtifact("causal_direction", None, None, type(exc).__name__))
        return StageAExtraction(
            (), (), tuple(artifacts), failure_stage="causal_direction",
            evidence_spans=spans, slots=dict(slots),
            candidate_catalog=candidates,
        )
    except Exception as exc:
        artifacts.append(StageArtifact("causal_direction", raw, None, type(exc).__name__))
        return StageAExtraction(
            (), (), tuple(artifacts), failure_stage="causal_direction",
            evidence_spans=spans, slots=dict(slots),
            candidate_catalog=candidates,
        )
    artifacts.append(StageArtifact("causal_direction", raw, parsed))

    try:
        raw = _provider_call(chat, NORMALIZATION_SYSTEM, _normalization_prompt(actor, event, effect), model, max_tokens, thinking)
        normalization, parsed = parse_ontology_normalization(raw)
    except ProviderCallError as exc:
        artifacts.append(StageArtifact("ontology_normalization", None, None, type(exc).__name__))
        frame = SourceSemanticFrame(
            spans, actor, event, effect, condition, direction,
            normalization=None, normalization_failure=type(exc).__name__,
        )
        return StageAExtraction(
            (), (frame,), tuple(artifacts), failure_stage="ontology_normalization",
            evidence_spans=spans, slots=dict(slots), causal_direction=direction,
            candidate_catalog=candidates,
        )
    except Exception as exc:
        artifacts.append(StageArtifact("ontology_normalization", raw, None, type(exc).__name__))
        frame = SourceSemanticFrame(
            spans, actor, event, effect, condition, direction,
            normalization=None, normalization_failure=type(exc).__name__,
        )
        return StageAExtraction(
            (), (frame,), tuple(artifacts), failure_stage="ontology_normalization",
            evidence_spans=spans, slots=dict(slots), causal_direction=direction,
            invented_ontology_count=_invented_count(exc),
            invented_ontology_taxonomy=_invented_taxonomy(exc),
            candidate_catalog=candidates,
        )
    artifacts.append(StageArtifact("ontology_normalization", raw, parsed))

    frame = SourceSemanticFrame(spans, actor, event, effect, condition, direction, normalization)
    proposition = assemble_grounded_proposition(frame, packet.evidence_id)
    propositions = (proposition,) if proposition is not None else ()
    return StageAExtraction(
        propositions, (frame,), tuple(artifacts),
        evidence_spans=spans, slots=dict(slots), causal_direction=direction,
        candidate_catalog=candidates,
    )


def extract_grounded_propositions(
    packet: PropositionPacket,
    chat: Callable[..., str],
    *,
    model: str | None = None,
    max_tokens: int = 512,
    thinking: str | None = None,
) -> tuple[ExtractedProposition, ...]:
    """Compatibility wrapper returning only deterministically assembled output.

    Call :func:`extract_span_first_propositions` directly when stage artifacts
    or the intermediate semantic frame are required.
    """
    return extract_span_first_propositions(
        packet, chat, model=model, max_tokens=max_tokens, thinking=thinking,
    ).propositions


def enumerate_clause_candidates(packet: PropositionPacket) -> tuple[ClauseCandidate, ...]:
    """Build a bounded, deterministic catalog of exact source-local units.

    Sentence and discourse boundaries provide the primary units.  Long
    punctuation-poor transcript regions are covered by overlapping token
    windows, so every offered value remains an exact span without asking the
    model to generate text or offsets.  Candidate IDs are stable for identical
    packets and restart for each source kind.
    """
    packet.validate()
    catalog: list[ClauseCandidate] = []
    for source in packet.sources():
        bounds = {0, len(source.text)}
        bounds.update(match.end() for match in _SENTENCE_BOUNDARY_RE.finditer(source.text))
        bounds.update(match.start() for match in _DISCOURSE_BOUNDARY_RE.finditer(source.text))
        ordered = sorted(bounds)
        spans: list[tuple[int, int]] = []
        for start, end in zip(ordered, ordered[1:]):
            start, end = _trim_bounds(source.text, start, end)
            if start >= end:
                continue
            tokens = list(_NONSPACE_RE.finditer(source.text, start, end))
            if len(tokens) <= _CLAUSE_MAX_TOKENS:
                spans.append((start, end))
                continue
            window_starts = list(range(0, len(tokens) - _CLAUSE_MAX_TOKENS + 1, _CLAUSE_WINDOW_STRIDE))
            final_start = len(tokens) - _CLAUSE_MAX_TOKENS
            if not window_starts or window_starts[-1] != final_start:
                window_starts.append(final_start)
            spans.extend(
                (tokens[index].start(), tokens[index + _CLAUSE_MAX_TOKENS - 1].end())
                for index in window_starts
            )

        if not spans and source.text.strip():
            start, end = _trim_bounds(source.text, 0, len(source.text))
            spans.append((start, end))
        spans = _deduplicate_bounds(spans)
        if len(spans) > _CLAUSE_CATALOG_LIMIT:
            spans = _evenly_bounded(spans, _CLAUSE_CATALOG_LIMIT)
        for index, (start, end) in enumerate(spans, 1):
            alignment = _alignment_from_bounds(
                source.kind, source.text, start, end, packet.source_window,
            )
            catalog.append(ClauseCandidate(f"{source.kind}:c{index:03d}", alignment))
    return tuple(catalog)


def clause_evidence_prompt(
    packet: PropositionPacket, candidates: tuple[ClauseCandidate, ...],
) -> str:
    """Render only deterministic candidate IDs and their exact source text."""
    packet.validate()
    allowed = [source.kind for source in packet.sources()]
    lines = [
        f"[{candidate.candidate_id}] {candidate.alignment.source_text}"
        for candidate in candidates
    ]
    return (
        "EVIDENCE ID: " + packet.evidence_id
        + "\nSOURCE CANDIDATES:\n" + ("\n".join(lines) or "(none)")
        + "\n\nAllowed source values: " + json.dumps(allowed)
        + '. Return exactly {"source":"<allowed source value>","candidate_ids":["<candidate id>"]}'
        + ' using one or two listed IDs, or {"source":null,"candidate_ids":[]}.'
    )


def parse_candidate_evidence_selection(
    raw: str,
    candidates: tuple[ClauseCandidate, ...],
    packet: PropositionPacket,
) -> tuple[tuple[SourceAlignment, ...], Mapping[str, Any]]:
    """Validate ID-only localization and derive all character offsets."""
    body = _json_object(raw, "candidate evidence localizer")
    if set(body) != {"source", "candidate_ids"} or not isinstance(body.get("candidate_ids"), list):
        raise ValueError("candidate evidence localizer requires source and candidate_ids")
    kind = body.get("source")
    candidate_ids = body["candidate_ids"]
    if kind is None and candidate_ids == []:
        return (), body
    if not isinstance(kind, str) or kind not in {source.kind for source in packet.sources()}:
        raise ValueError("candidate evidence selection has an invalid source")
    if not 1 <= len(candidate_ids) <= 2:
        raise ValueError("candidate evidence selection requires one or two IDs")
    if not all(isinstance(value, str) and value for value in candidate_ids):
        raise ValueError("candidate evidence selection IDs must be nonempty strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate evidence selection contains duplicate IDs")
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate catalog contains duplicate IDs")
    try:
        selected = [by_id[value] for value in candidate_ids]
    except KeyError as exc:
        raise ValueError("candidate evidence selection contains an unknown ID") from exc
    if any(candidate.alignment.source_kind != kind for candidate in selected):
        raise ValueError("candidate evidence selection mixes or misdeclares sources")
    spans = tuple(candidate.alignment for candidate in selected)
    return coalesce_selected_evidence(spans, packet), body


def coalesce_selected_evidence(
    spans: tuple[SourceAlignment, ...], packet: PropositionPacket,
) -> tuple[SourceAlignment, ...]:
    """Merge overlapping/touching selected units but preserve real gaps."""
    if not spans:
        return ()
    source_kind = spans[0].source_kind
    if any(span.source_kind != source_kind for span in spans):
        raise ValueError("selected evidence must use one coherent source")
    source_by_kind = {source.kind: source.text for source in packet.sources()}
    if source_kind not in source_by_kind:
        raise ValueError("selected evidence uses a source unavailable to the packet")
    source_text = source_by_kind[source_kind]
    merged: list[SourceAlignment] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if (
            not merged
            or (
                span.start > merged[-1].end
                and source_text[merged[-1].end:span.start].strip()
            )
        ):
            merged.append(span)
            continue
        previous = merged.pop()
        merged.append(_alignment_from_bounds(
            source_kind, source_text, previous.start, max(previous.end, span.end),
            packet.source_window,
        ))
    return tuple(merged)


def parse_evidence_selection(
    raw: str, packet: PropositionPacket,
) -> tuple[tuple[SourceAlignment, ...], Mapping[str, Any]]:
    body = _json_object(raw, "evidence localizer")
    if set(body) != {"source", "evidence_spans"} or not isinstance(body.get("evidence_spans"), list):
        raise ValueError("evidence localizer requires source and evidence_spans")
    kind = body.get("source")
    phrases = body["evidence_spans"]
    if kind is None and phrases == []:
        return (), body
    sources = {item.kind: item.text for item in packet.sources()}
    if kind not in sources or not 1 <= len(phrases) <= 2:
        raise ValueError("evidence selection has invalid source or span count")
    if not all(isinstance(phrase, str) and phrase.strip() for phrase in phrases):
        raise ValueError("evidence selection spans must be nonempty strings")
    spans = tuple(_source_alignment(kind, phrase, sources[kind], packet.source_window) for phrase in phrases)
    if len({(item.start, item.end) for item in spans}) != len(spans):
        raise ValueError("evidence selection contains duplicate spans")
    if _has_nested_spans(spans):
        raise ValueError("evidence selection contains nested non-minimal spans")
    return spans, body


def parse_semantic_slot(
    raw: str,
    role: str,
    evidence_spans: tuple[SourceAlignment, ...],
    packet: PropositionPacket,
) -> tuple[SemanticSlot | None, Mapping[str, Any]]:
    if role not in SLOT_SYSTEMS:
        raise ValueError(f"unknown semantic slot: {role}")
    if not evidence_spans:
        raise ValueError(f"{role} extractor requires selected evidence spans")
    body = _json_object(raw, role + " extractor")
    if set(body) != {role}:
        raise ValueError(f"{role} extractor must return only {role}")
    phrase = body.get(role)
    if role == "condition" and (phrase is None or phrase == "NONE"):
        return None, body
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError(f"{role} extractor requires an exact source phrase")
    source_kind = evidence_spans[0].source_kind
    if any(span.source_kind != source_kind for span in evidence_spans):
        raise ValueError("selected evidence must use one coherent source")
    sources = {item.kind: item.text for item in packet.sources()}
    if source_kind not in sources:
        raise ValueError("selected evidence uses a source unavailable to the packet")
    alignment = _source_alignment_within_spans(
        source_kind, phrase, evidence_spans, sources[source_kind], packet.source_window,
    )
    return SemanticSlot(role, alignment), body  # type: ignore[arg-type]


def parse_causal_direction(raw: str) -> tuple[CausalDirection, Mapping[str, Any]]:
    body = _json_object(raw, "causal direction classifier")
    if set(body) != {"causal_direction"} or body.get("causal_direction") not in _VALID_DIRECTIONS:
        raise ValueError("causal direction classifier returned an invalid label")
    return body["causal_direction"], body  # type: ignore[return-value]


def parse_ontology_normalization(raw: str) -> tuple[OntologyNormalization, Mapping[str, Any]]:
    body = _json_object(raw, "ontology normalizer")
    required = {"actor_concept", "event_relation", "effect_concept"}
    if set(body) != required:
        raise ValueError("ontology normalizer returned an invalid shape")
    actor = body.get("actor_concept")
    event = body.get("event_relation")
    effect = body.get("effect_concept")
    for field, value in (("actor_concept", actor), ("event_relation", event), ("effect_concept", effect)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"ontology normalizer {field} must be a string or null")
    checks = (
        ("actor_concept", "actor concept", actor, STRATEGIC_CONCEPTS),
        ("event_relation", "event relation", event, RELATION_TYPES),
        ("effect_concept", "effect concept", effect, STRATEGIC_CONCEPTS),
    )
    invented_values = [
        (label, value)
        for field, label, value, allowed in checks
        if value is not None and value not in allowed
    ]
    if invented_values:
        invented: dict[str, int] = {}
        for _, value in invented_values:
            invented[str(value)] = invented.get(str(value), 0) + 1
        raise InventedOntologyContent(
            "ontology normalizer invented " + ", ".join(f"{label} {value!r}" for label, value in invented_values),
            invented,
        )
    return OntologyNormalization(actor, event, effect), body


def assemble_grounded_proposition(
    frame: SourceSemanticFrame, evidence_id: str,
) -> ExtractedProposition | None:
    """Assemble source output without allowing a final generative rewrite.

    Only supported forward causality becomes a :class:`GroundedProposition`;
    reversed and non-causal directions abstain.  A malformed frame fails
    closed with a ``ValueError`` rather than emitting ungrounded output.
    """
    if frame.causal_direction != "actor_event_causes_effect":
        return None
    slots = (frame.actor, frame.event, frame.effect) + ((frame.condition,) if frame.condition else ())
    if len({item.alignment.source_kind for item in slots}) != 1:
        raise ValueError("semantic frame slots must use one coherent source")
    if len({span.source_kind for span in frame.evidence_spans}) != 1:
        raise ValueError("semantic frame evidence spans must use one coherent source")
    source_kind = slots[0].alignment.source_kind
    for slot in slots:
        if not any(
            span.source_kind == source_kind
            and span.start <= slot.alignment.start
            and slot.alignment.end <= span.end
            for span in frame.evidence_spans
        ):
            raise ValueError("semantic frame slot lies outside selected evidence spans")
    proposition = GroundedProposition(
        frame.actor.text,
        frame.event.text,
        frame.effect.text,
        frame.condition.text if frame.condition else None,
        (evidence_id,),
    )
    names = ("subject", "predicate", "effect", "condition")
    alignments = tuple(
        PropositionAlignment(name, slot.alignment.source_kind, slot.alignment.start, slot.alignment.end,
                             slot.text, slot.alignment.absolute_start, slot.alignment.absolute_end)
        for name, slot in zip(names, slots)
    )
    return ExtractedProposition(proposition, alignments)


def parse_grounded_propositions(raw: str, packet: PropositionPacket) -> tuple[ExtractedProposition, ...]:
    """Parse legacy one-pass artifacts while retaining strict source checks."""
    packet.validate()
    body = _json_object(raw, "proposition extractor")
    if not isinstance(body.get("propositions"), list):
        raise ValueError("proposition extractor response requires propositions list")
    sources = {item.kind: item.text for item in packet.sources()}
    extracted = []
    for item in body["propositions"]:
        if not isinstance(item, Mapping):
            raise ValueError("propositions must be objects")
        proposition = GroundedProposition.from_dict({
            "subject_source": item.get("subject_source"),
            "predicate_source": item.get("predicate_source"),
            "effect_source": item.get("effect_source"),
            "condition_source": item.get("condition_source"),
            "evidence_ids": [packet.evidence_id],
        })
        grounding = item.get("grounding")
        if not isinstance(grounding, Mapping):
            raise ValueError("grounded proposition requires field grounding")
        alignments = []
        for field, phrase in (("subject", proposition.subject_source), ("predicate", proposition.predicate_source),
                              ("effect", proposition.effect_source), ("condition", proposition.condition_source)):
            raw_alignment = grounding.get(field)
            if phrase is None:
                if raw_alignment is not None:
                    raise ValueError("null condition cannot have grounding")
                continue
            if not isinstance(raw_alignment, Mapping):
                raise ValueError(f"grounded proposition {field} requires grounding")
            kind = raw_alignment.get("source")
            if set(raw_alignment) != {"source"} or kind not in sources:
                raise ValueError(f"grounded proposition {field} has invalid source grounding")
            aligned = _source_alignment(kind, phrase, sources[kind], packet.source_window)
            alignments.append(PropositionAlignment(field, aligned.source_kind, aligned.start, aligned.end,
                                                   aligned.source_text, aligned.absolute_start, aligned.absolute_end))
        if len({item.source_kind for item in alignments}) != 1:
            raise ValueError("grounded proposition fields must use one coherent source")
        extracted.append(ExtractedProposition(proposition, tuple(alignments)))
    return tuple(extracted)


def _chat(
    chat: Callable[..., str], system: str, user: str, model: str | None,
    max_tokens: int, thinking: str | None,
) -> str:
    return chat(system=system, user=user, temperature=0.0, max_tokens=max_tokens, model=model, thinking=thinking)


def _provider_call(
    chat: Callable[..., str], system: str, user: str, model: str | None,
    max_tokens: int, thinking: str | None,
) -> str:
    try:
        return _chat(chat, system, user, model, max_tokens, thinking)
    except Exception as exc:
        raise ProviderCallError("provider call failed before producing raw output") from exc


def _slot_prompt(
    role: str, evidence_spans: tuple[SourceAlignment, ...], slots: Mapping[str, SemanticSlot | None],
) -> str:
    selected = "\n".join(f"[span {index + 1}] {span.source_text}" for index, span in enumerate(evidence_spans))
    prior = "\n".join(f"{name}: {slot.text if slot else 'NONE'}" for name, slot in slots.items()) or "(none)"
    value = 'null or "NONE"' if role == "condition" else '"exact source phrase"'
    return f"SELECTED EVIDENCE:\n{selected}\n\nPREVIOUSLY SELECTED SLOTS:\n{prior}\n\nReturn exactly: {{\"{role}\":{value}}}."


def _direction_prompt(
    spans: tuple[SourceAlignment, ...], actor: SemanticSlot, event: SemanticSlot,
    effect: SemanticSlot, condition: SemanticSlot | None,
) -> str:
    return (
        "SELECTED EVIDENCE:\n" + "\n".join(item.source_text for item in spans)
        + f"\n\nactor: {actor.text}\nevent: {event.text}\neffect: {effect.text}\ncondition: "
        + (condition.text if condition else "NONE")
        + "\nAllowed labels: " + json.dumps(sorted(_VALID_DIRECTIONS))
        + '. Return exactly: {"causal_direction":"<allowed label>"}.'
    )


def _normalization_prompt(actor: SemanticSlot, event: SemanticSlot, effect: SemanticSlot) -> str:
    concepts = {key: value.description for key, value in sorted(STRATEGIC_CONCEPTS.items())}
    return (
        f"SOURCE FRAME:\nactor: {actor.text}\nevent: {event.text}\neffect: {effect.text}\n\n"
        + "Allowed strategic concepts: " + json.dumps(concepts, sort_keys=True)
        + "\nAllowed event relations: " + json.dumps(sorted(RELATION_TYPES))
        + '\nReturn exactly: {"actor_concept":null,"event_relation":null,"effect_concept":null}, replacing null only with a directly supported allowed ID.'
    )


def _json_object(raw: str, stage: str) -> Mapping[str, Any]:
    try:
        body = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{stage} returned malformed JSON") from exc
    if not isinstance(body, Mapping):
        raise ValueError(f"{stage} must return a JSON object")
    return body


def _trim_bounds(source: str, start: int, end: int) -> tuple[int, int]:
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


def _deduplicate_bounds(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    result = []
    for span in sorted(spans):
        if span not in seen:
            seen.add(span)
            result.append(span)
    return result


def _evenly_bounded(
    spans: list[tuple[int, int]], limit: int,
) -> list[tuple[int, int]]:
    """Keep deterministic coverage across an unexpectedly dense source."""
    if len(spans) <= limit:
        return spans
    indexes = [round(index * (len(spans) - 1) / (limit - 1)) for index in range(limit)]
    return [spans[index] for index in dict.fromkeys(indexes)]


def _alignment_from_bounds(
    kind: str,
    source: str,
    start: int,
    end: int,
    source_window: SourceWindow | None,
) -> SourceAlignment:
    if kind not in {"insight", "transcript"} or not 0 <= start < end <= len(source):
        raise ValueError("candidate source alignment has invalid bounds")
    absolute_start = absolute_end = None
    if kind == "transcript":
        if source_window is None or source_window.window_start is None:
            raise ValueError("transcript candidate requires a verified source window")
        absolute_start = source_window.window_start + start
        absolute_end = source_window.window_start + end
    return SourceAlignment(
        kind, start, end, source[start:end], absolute_start, absolute_end,
    )  # type: ignore[arg-type]


def _source_alignment(
    kind: str,
    phrase: str,
    source: str,
    source_window: SourceWindow | None,
    *,
    unsupported: bool = False,
) -> SourceAlignment:
    error = UnsupportedSourceSlot if unsupported else ValueError
    phrase = phrase.strip()
    if not phrase:
        raise error("source phrase must not be empty")
    locations = _exact_locations(source, phrase)
    if len(locations) != 1:
        raise error("source phrase must quote one unambiguous exact source phrase")
    start, end = locations[0]
    absolute_start = absolute_end = None
    if kind == "transcript":
        assert source_window is not None and source_window.window_start is not None
        absolute_start = source_window.window_start + start
        absolute_end = source_window.window_start + end
    return SourceAlignment(kind, start, end, phrase, absolute_start, absolute_end)  # type: ignore[arg-type]


def _source_alignment_within_spans(
    kind: str,
    phrase: str,
    evidence_spans: tuple[SourceAlignment, ...],
    source: str,
    source_window: SourceWindow | None,
) -> SourceAlignment:
    """Ground a phrase to its single occurrence inside the selected evidence.

    Uniqueness is evaluated within the selected spans only, so common text that
    also appears elsewhere in the packet source resolves when exactly one
    selected occurrence exists.  Zero or multiple contained occurrences fail
    closed with the unsupported-slot taxonomy, and overlapping or duplicated
    selected contexts never double-count a single source occurrence.
    """
    phrase = phrase.strip()
    if not phrase:
        raise UnsupportedSourceSlot("source phrase must not be empty")
    contained = [
        (start, end)
        for start, end in _exact_locations(source, phrase)
        if any(span.start <= start and end <= span.end for span in evidence_spans)
    ]
    if not contained:
        raise UnsupportedSourceSlot("source phrase is outside selected evidence spans")
    if len(contained) > 1:
        raise UnsupportedSourceSlot(
            "source phrase must quote one unambiguous exact source phrase within selected evidence spans"
        )
    start, end = contained[0]
    absolute_start = absolute_end = None
    if kind == "transcript":
        assert source_window is not None and source_window.window_start is not None
        absolute_start = source_window.window_start + start
        absolute_end = source_window.window_start + end
    return SourceAlignment(kind, start, end, phrase, absolute_start, absolute_end)  # type: ignore[arg-type]


def _has_nested_spans(spans: tuple[SourceAlignment, ...]) -> bool:
    for index, outer in enumerate(spans):
        for inner in spans[index + 1:]:
            if outer.start <= inner.start and inner.end <= outer.end:
                return True
            if inner.start <= outer.start and outer.end <= inner.end:
                return True
    return False


def _unsupported_count(exc: Exception) -> int:
    return 1 if isinstance(exc, UnsupportedSourceSlot) else 0


def _invented_count(exc: Exception) -> int:
    return exc.count if isinstance(exc, InventedOntologyContent) else 0


def _invented_taxonomy(exc: Exception) -> Mapping[str, int]:
    return exc.invented if isinstance(exc, InventedOntologyContent) else {}


def _exact_locations(source: str, phrase: str) -> tuple[tuple[int, int], ...]:
    """Find byte-identical token-bounded source phrases."""
    positions = []
    start = 0
    while True:
        index = source.find(phrase, start)
        if index < 0:
            return tuple(positions)
        end = index + len(phrase)
        if ((index == 0 or not _token_character(source[index - 1]))
                and (end == len(source) or not _token_character(source[end]))):
            positions.append((index, end))
        start = index + 1


def _token_character(value: str) -> bool:
    return value.isalnum() or value in {"'", "’", "‘", "`", "_"}
