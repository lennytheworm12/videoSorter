"""Phase 2D source-grounded proposition extraction before ontology mapping.

This module intentionally stops before canonical relation selection.  It
preserves propositions that are grounded but unmappable, and validates every
field against either the immutable insight summary, a verified bronze window,
or both depending on the requested source mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Literal, Mapping

from pipeline.relation_extract import GroundedProposition
from pipeline.source_windows import SourceWindow


SourceMode = Literal["insight", "transcript", "combined"]
_VALID_MODES = frozenset(("insight", "transcript", "combined"))

PROPOSITION_SYSTEM = """Return JSON only. Extract source-grounded causal
propositions from supplied source text. Do not select ontology concepts,
canonical relation types, or canonical entities. Return zero propositions for
advice without a stated or clearly entailed causal mechanism. Each non-null
field must copy one supplied source span exactly, including wording and tense.
Do not paraphrase. The system derives character spans; return only the source
label that contains each exact quoted field."""


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
        rendered = "\n\n".join(f"[{source.kind}]\n{source.text}" for source in self.sources())
        source_kinds = [source.kind for source in self.sources()]
        return (
            "EVIDENCE ID: " + self.evidence_id + "\nSOURCE TEXT:\n" + rendered
            + "\n\nAllowed grounding source values: " + json.dumps(source_kinds)
            + ". Return exactly: {\"propositions\":[{\"subject_source\":\"...\",\"predicate_source\":\"...\",\"effect_source\":\"...\",\"condition_source\":null,\"grounding\":{\"subject\":{\"source\":\"<allowed source value>\"},\"predicate\":{\"source\":\"<allowed source value>\"},\"effect\":{\"source\":\"<allowed source value>\"},\"condition\":null}}]}."
            + " Copy every non-null field exactly from the supplied source. Do not invent text or use text outside the supplied sources."
        )


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
class ExtractedProposition:
    proposition: GroundedProposition
    alignments: tuple[PropositionAlignment, ...]


def extract_grounded_propositions(
    packet: PropositionPacket, chat: Callable[..., str], *, model: str | None = None,
    max_tokens: int = 512, thinking: str | None = None,
) -> tuple[ExtractedProposition, ...]:
    """Extract propositions without canonical mapping or persistence."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    raw = chat(
        system=PROPOSITION_SYSTEM, user=packet.prompt(), temperature=0.0,
        max_tokens=max_tokens, model=model, thinking=thinking,
    )
    return parse_grounded_propositions(raw, packet)


def parse_grounded_propositions(raw: str, packet: PropositionPacket) -> tuple[ExtractedProposition, ...]:
    """Parse and verify source fields and their claimed source offsets."""
    packet.validate()
    try:
        body = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("proposition extractor returned malformed JSON") from exc
    if not isinstance(body, Mapping) or not isinstance(body.get("propositions"), list):
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
        for field, phrase in (
            ("subject", proposition.subject_source),
            ("predicate", proposition.predicate_source),
            ("effect", proposition.effect_source),
            ("condition", proposition.condition_source),
        ):
            raw_alignment = grounding.get(field)
            if phrase is None:
                if raw_alignment is not None:
                    raise ValueError("null condition cannot have grounding")
                continue
            alignment = _parse_alignment(field, phrase, raw_alignment, sources, packet.source_window)
            alignments.append(alignment)
        proposition_sources = {item.source_kind for item in alignments}
        if len(proposition_sources) != 1:
            raise ValueError("grounded proposition fields must use one coherent source")
        extracted.append(ExtractedProposition(proposition, tuple(alignments)))
    return tuple(extracted)


def _parse_alignment(
    field: str, phrase: str, raw: object, sources: Mapping[str, str], source_window: SourceWindow | None,
) -> PropositionAlignment:
    if not isinstance(raw, Mapping):
        raise ValueError(f"grounded proposition {field} requires grounding")
    kind = raw.get("source")
    if set(raw) != {"source"} or kind not in sources:
        raise ValueError(f"grounded proposition {field} has invalid source grounding")
    source = sources[kind]
    locations = _exact_locations(source, phrase)
    if len(locations) != 1:
        raise ValueError(f"grounded proposition {field} must quote one unambiguous exact source phrase")
    start, end = locations[0]
    absolute_start = absolute_end = None
    if kind == "transcript":
        assert source_window is not None and source_window.window_start is not None
        absolute_start = source_window.window_start + start
        absolute_end = source_window.window_start + end
    return PropositionAlignment(field, kind, start, end, phrase, absolute_start, absolute_end)


def _exact_locations(source: str, phrase: str) -> tuple[tuple[int, int], ...]:
    """Find byte-identical source phrases; ambiguity cannot become provenance."""
    positions = []
    start = 0
    while True:
        index = source.find(phrase, start)
        if index < 0:
            return tuple(positions)
        end = index + len(phrase)
        if (
            (index == 0 or not _token_character(source[index - 1]))
            and (end == len(source) or not _token_character(source[end]))
        ):
            positions.append((index, end))
        start = index + 1


def _token_character(value: str) -> bool:
    return value.isalnum() or value in {"'", "’", "‘", "`", "_"}
