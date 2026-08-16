"""Immutable, source-preserving semantic intermediate representation.

This module is the Phase 2F boundary between exact bronze text and any later
domain interpretation.  It intentionally models mentions and general semantic
relations rather than propositions or domain concepts.  Every accepted object
is reconstructible from a source window and carries the model decision that
selected it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Mapping

from pipeline.semantic_source import PASS0_VERSION


SCHEMA_VERSION = "semantic-ir-v2"
COMPILER_VERSION = "phase2f-source-semantic-ir-compiler-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NON_DOMAIN_UNITS = frozenset({
    "count", "percent", "milliseconds", "seconds", "minutes", "hours",
    "characters", "tokens", "words", "units",
})


class NodeType(str, Enum):
    ENTITY = "ENTITY"
    ABILITY_OR_RESOURCE = "ABILITY_OR_RESOURCE"
    EVENT = "EVENT"
    ACTION = "ACTION"
    STATE = "STATE"
    OUTCOME = "OUTCOME"
    QUANTITY = "QUANTITY"
    TIME = "TIME"
    LOCATION_OR_SPACE = "LOCATION_OR_SPACE"


class EdgeType(str, Enum):
    ACTOR = "ACTOR"
    TARGET = "TARGET"
    OBJECT = "OBJECT"
    EXPERIENCER = "EXPERIENCER"
    CAUSES = "CAUSES"
    ENABLES = "ENABLES"
    PREVENTS = "PREVENTS"
    REQUIRES = "REQUIRES"
    CONDITION = "CONDITION"
    PURPOSE = "PURPOSE"
    RESULT = "RESULT"
    TEMPORAL_BEFORE = "TEMPORAL_BEFORE"
    TEMPORAL_AFTER = "TEMPORAL_AFTER"
    TEMPORAL_UNTIL = "TEMPORAL_UNTIL"
    TERMINATES = "TERMINATES"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    NEGATES = "NEGATES"
    MODIFIES = "MODIFIES"
    REFERS_TO = "REFERS_TO"


class AmbiguityState(str, Enum):
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class Polarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"


class Modality(str, Enum):
    ASSERTED = "ASSERTED"
    POSSIBLE = "POSSIBLE"
    PROBABLE = "PROBABLE"
    NECESSARY = "NECESSARY"
    OBLIGATORY = "OBLIGATORY"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    UNKNOWN = "UNKNOWN"


class TemporalScope(str, Enum):
    NONE = "NONE"
    PAST = "PAST"
    PRESENT = "PRESENT"
    FUTURE = "FUTURE"
    ONGOING = "ONGOING"
    BOUNDED = "BOUNDED"
    HABITUAL = "HABITUAL"
    UNKNOWN = "UNKNOWN"


class Conditionality(str, Enum):
    UNCONDITIONAL = "UNCONDITIONAL"
    CONDITIONAL = "CONDITIONAL"
    HYPOTHETICAL = "HYPOTHETICAL"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    UNKNOWN = "UNKNOWN"


class ComparativeDegree(str, Enum):
    NONE = "NONE"
    EQUAL = "EQUAL"
    GREATER = "GREATER"
    LESS = "LESS"
    MAXIMUM = "MAXIMUM"
    MINIMUM = "MINIMUM"
    UNKNOWN = "UNKNOWN"


class Uncertainty(str, Enum):
    CERTAIN = "CERTAIN"
    LIKELY = "LIKELY"
    POSSIBLE = "POSSIBLE"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


class QualifierKind(str, Enum):
    POLARITY = "POLARITY"
    MODALITY = "MODALITY"
    TEMPORAL_SCOPE = "TEMPORAL_SCOPE"
    CONDITIONALITY = "CONDITIONALITY"
    COMPARATIVE_DEGREE = "COMPARATIVE_DEGREE"
    UNCERTAINTY = "UNCERTAINTY"


def _require_nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_sha256(value: object) -> str:
    """Return the canonical SHA-256 of a JSON-compatible value or string."""
    payload: object = value if not isinstance(value, str) else {"text": value}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}_{hashlib.sha256(_canonical_bytes(value)).hexdigest()[:24]}"


@dataclass(frozen=True)
class SourceSpan:
    """An exact half-open character span within one immutable source window."""

    source_id: str
    window_id: str
    local_start: int
    local_end: int
    text: str
    absolute_start: int | None = None
    absolute_end: int | None = None
    speaker: str | None = None
    start_timestamp: float | None = None
    end_timestamp: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.source_id, "source_id")
        _require_nonempty(self.window_id, "window_id")
        start = _require_int(self.local_start, "local_start")
        end = _require_int(self.local_end, "local_end")
        if start < 0 or end <= start:
            raise ValueError("source span requires 0 <= local_start < local_end")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("source span text must be non-empty")
        if len(self.text) != end - start:
            raise ValueError("source span text length must equal its local offset width")
        if (self.absolute_start is None) != (self.absolute_end is None):
            raise ValueError("absolute offsets must be supplied together")
        if self.absolute_start is not None:
            absolute_start = _require_int(self.absolute_start, "absolute_start")
            absolute_end = _require_int(self.absolute_end, "absolute_end")
            if absolute_start < 0 or absolute_end <= absolute_start:
                raise ValueError("absolute offsets must be ordered and non-negative")
            if absolute_end - absolute_start != end - start:
                raise ValueError("absolute and local source spans must have equal width")
        if self.speaker is not None:
            _require_nonempty(self.speaker, "speaker")
        if (self.start_timestamp is None) != (self.end_timestamp is None):
            raise ValueError("timestamps must be supplied together")
        if self.start_timestamp is not None:
            for value, label in ((self.start_timestamp, "start_timestamp"), (self.end_timestamp, "end_timestamp")):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{label} must be a finite number")
            if self.start_timestamp < 0 or self.end_timestamp < self.start_timestamp:
                raise ValueError("timestamps must be ordered and non-negative")

    @property
    def start(self) -> int:
        """Compatibility shorthand for ``local_start``."""
        return self.local_start

    @property
    def end(self) -> int:
        """Compatibility shorthand for ``local_end``."""
        return self.local_end

    def validate_against(
        self, source_id: str, window_id: str, window_text: str, *,
        window_source_start: int | None = None, speaker: str | None = None,
        start_timestamp: float | None = None, end_timestamp: float | None = None,
    ) -> None:
        if self.source_id != source_id or self.window_id != window_id:
            raise ValueError("source span belongs to a different source or window")
        if self.local_end > len(window_text):
            raise ValueError("source span falls outside the source window")
        if window_text[self.local_start:self.local_end] != self.text:
            raise ValueError("source span text does not exactly match the source window")
        if window_source_start is not None:
            _require_int(window_source_start, "window_source_start")
            if (self.absolute_start, self.absolute_end) != (
                window_source_start + self.local_start, window_source_start + self.local_end,
            ):
                raise ValueError("source span absolute offsets do not match the bronze window")
            if self.speaker != speaker:
                raise ValueError("source span speaker does not match the bronze window")
            if (self.start_timestamp, self.end_timestamp) != (start_timestamp, end_timestamp):
                raise ValueError("source span timestamps do not match the bronze window")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id, "window_id": self.window_id,
            "local_start": self.local_start, "local_end": self.local_end, "text": self.text,
            "absolute_start": self.absolute_start, "absolute_end": self.absolute_end,
            "speaker": self.speaker, "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceSpan":
        _expect_keys(value, {
            "source_id", "window_id", "local_start", "local_end", "text", "absolute_start",
            "absolute_end", "speaker", "start_timestamp", "end_timestamp",
        }, "source span")
        return cls(**dict(value))


@dataclass(frozen=True)
class GroundedValue:
    """A source phrase plus an optional non-domain numeric normalization."""

    text: str
    span: SourceSpan
    normalized_value: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.text, "grounded value text")
        if self.text != self.span.text:
            raise ValueError("grounded value text must equal its exact source span")
        if self.normalized_value is not None:
            if isinstance(self.normalized_value, bool) or not isinstance(self.normalized_value, (int, float)):
                raise ValueError("normalized_value must be numeric")
            if not math.isfinite(float(self.normalized_value)):
                raise ValueError("normalized_value must be finite")
        if self.unit is not None:
            _require_nonempty(self.unit, "grounded value unit")
            if self.unit not in _NON_DOMAIN_UNITS:
                raise ValueError("grounded value unit must use the closed non-domain unit vocabulary")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "span": self.span.to_dict(),
                "normalized_value": self.normalized_value, "unit": self.unit}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundedValue":
        _expect_keys(value, {"text", "span", "normalized_value", "unit"}, "grounded value")
        if not isinstance(value["text"], str):
            raise ValueError("grounded value text must be a string")
        return cls(value["text"], SourceSpan.from_dict(_mapping(value["span"], "grounded value span")),
                   value["normalized_value"], value["unit"])


@dataclass(frozen=True)
class QualifierCue:
    kind: QualifierKind
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.kind, QualifierKind) or not isinstance(self.span, SourceSpan):
            raise ValueError("qualifier cues require a typed kind and exact source span")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "span": self.span.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QualifierCue":
        _expect_keys(value, {"kind", "span"}, "qualifier cue")
        try:
            return cls(
                QualifierKind(value["kind"]),
                SourceSpan.from_dict(_mapping(value["span"], "qualifier cue span")),
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid qualifier cue") from exc


@dataclass(frozen=True)
class SemanticQualifiers:
    polarity: Polarity = Polarity.UNKNOWN
    negated: bool = False
    modality: Modality = Modality.UNKNOWN
    temporal_scope: TemporalScope = TemporalScope.UNKNOWN
    conditionality: Conditionality = Conditionality.UNKNOWN
    comparative_degree: ComparativeDegree = ComparativeDegree.UNKNOWN
    duration: GroundedValue | None = None
    quantity: GroundedValue | None = None
    uncertainty: Uncertainty = Uncertainty.UNKNOWN
    cues: tuple[QualifierCue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.polarity, Polarity) or not isinstance(self.negated, bool):
            raise ValueError("polarity and negated must use their declared types")
        for value, enum_type, label in (
            (self.modality, Modality, "modality"), (self.temporal_scope, TemporalScope, "temporal_scope"),
            (self.conditionality, Conditionality, "conditionality"),
            (self.comparative_degree, ComparativeDegree, "comparative_degree"),
            (self.uncertainty, Uncertainty, "uncertainty"),
        ):
            if not isinstance(value, enum_type):
                raise ValueError(f"{label} must use its declared enum")
        if self.negated != (self.polarity is Polarity.NEGATIVE):
            raise ValueError("negative polarity and the negated flag must agree")
        if self.duration is not None and not isinstance(self.duration, GroundedValue):
            raise ValueError("duration must be a GroundedValue")
        if self.quantity is not None and not isinstance(self.quantity, GroundedValue):
            raise ValueError("quantity must be a GroundedValue")
        if not isinstance(self.cues, tuple) or any(not isinstance(item, QualifierCue) for item in self.cues):
            raise ValueError("qualifier cues must be an immutable tuple of QualifierCue values")
        if len({json.dumps(item.to_dict(), sort_keys=True) for item in self.cues}) != len(self.cues):
            raise ValueError("qualifier cues must be unique")
        asserted = {
            QualifierKind.POLARITY: self.polarity is not Polarity.UNKNOWN,
            QualifierKind.MODALITY: self.modality is not Modality.UNKNOWN,
            QualifierKind.TEMPORAL_SCOPE: self.temporal_scope is not TemporalScope.UNKNOWN,
            QualifierKind.CONDITIONALITY: self.conditionality is not Conditionality.UNKNOWN,
            QualifierKind.COMPARATIVE_DEGREE: self.comparative_degree is not ComparativeDegree.UNKNOWN,
            QualifierKind.UNCERTAINTY: self.uncertainty is not Uncertainty.UNKNOWN,
        }
        cue_kinds = {cue.kind for cue in self.cues}
        missing = [kind.value for kind, required in asserted.items() if required and kind not in cue_kinds]
        unsupported = [kind.value for kind in cue_kinds if not asserted[kind]]
        if missing or unsupported:
            raise ValueError(f"qualifier cues do not match asserted fields; missing={missing}, unsupported={unsupported}")
        object.__setattr__(self, "cues", tuple(sorted(
            self.cues, key=lambda item: (item.kind.value, item.span.local_start, item.span.local_end, item.span.text),
        )))

    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(item.span for item in self.cues) + tuple(
            item.span for item in (self.duration, self.quantity) if item is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "polarity": self.polarity.value, "negated": self.negated,
            "modality": self.modality.value, "temporal_scope": self.temporal_scope.value,
            "conditionality": self.conditionality.value,
            "comparative_degree": self.comparative_degree.value,
            "duration": self.duration.to_dict() if self.duration else None,
            "quantity": self.quantity.to_dict() if self.quantity else None,
            "uncertainty": self.uncertainty.value,
            "cues": [item.to_dict() for item in self.cues],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticQualifiers":
        keys = {"polarity", "negated", "modality", "temporal_scope", "conditionality",
                "comparative_degree", "duration", "quantity", "uncertainty", "cues"}
        _expect_keys(value, keys, "semantic qualifiers")
        try:
            cues = value["cues"]
            if not isinstance(cues, list):
                raise ValueError("serialized qualifier cues must be a list")
            return cls(
                polarity=Polarity(value["polarity"]), negated=value["negated"],
                modality=Modality(value["modality"]), temporal_scope=TemporalScope(value["temporal_scope"]),
                conditionality=Conditionality(value["conditionality"]),
                comparative_degree=ComparativeDegree(value["comparative_degree"]),
                duration=_grounded_or_none(value["duration"]), quantity=_grounded_or_none(value["quantity"]),
                uncertainty=Uncertainty(value["uncertainty"]),
                cues=tuple(QualifierCue.from_dict(_mapping(item, "qualifier cue")) for item in cues),
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid semantic qualifiers") from exc


@dataclass(frozen=True)
class ModelDecisionProvenance:
    """Digest summary; independent verification requires the retained run artifact."""

    decision_id: str
    model_id: str
    prompt_version: str
    configuration_sha256: str
    input_sha256: str
    output_sha256: str
    candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.decision_id, "decision_id")
        _require_nonempty(self.model_id, "model_id")
        _require_nonempty(self.prompt_version, "prompt_version")
        _require_sha256(self.configuration_sha256, "configuration_sha256")
        _require_sha256(self.input_sha256, "input_sha256")
        _require_sha256(self.output_sha256, "output_sha256")
        if not isinstance(self.candidate_ids, tuple):
            raise ValueError("candidate_ids must be an immutable tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.candidate_ids):
            raise ValueError("candidate_ids must contain non-empty strings")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        object.__setattr__(self, "candidate_ids", tuple(sorted(self.candidate_ids)))

    @classmethod
    def create(
        cls, decision_id: str, model_id: str, prompt_version: str, *, configuration: object,
        model_input: object, model_output: object, candidate_ids: tuple[str, ...] = (),
    ) -> "ModelDecisionProvenance":
        return cls(decision_id, model_id, prompt_version, content_sha256(configuration),
                   content_sha256(model_input), content_sha256(model_output), candidate_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id, "model_id": self.model_id,
            "prompt_version": self.prompt_version, "configuration_sha256": self.configuration_sha256,
            "input_sha256": self.input_sha256, "output_sha256": self.output_sha256,
            "candidate_ids": list(self.candidate_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelDecisionProvenance":
        keys = {"decision_id", "model_id", "prompt_version", "configuration_sha256",
                "input_sha256", "output_sha256", "candidate_ids"}
        _expect_keys(value, keys, "model decision provenance")
        candidates = value["candidate_ids"]
        if not isinstance(candidates, list):
            raise ValueError("serialized candidate_ids must be a list")
        return cls(value["decision_id"], value["model_id"], value["prompt_version"],
                   value["configuration_sha256"], value["input_sha256"], value["output_sha256"],
                   tuple(candidates))


@dataclass(frozen=True)
class SemanticNode:
    node_type: NodeType
    source_span: SourceSpan
    provenance: ModelDecisionProvenance
    qualifiers: SemanticQualifiers = field(default_factory=SemanticQualifiers)
    ambiguity: AmbiguityState = AmbiguityState.NONE
    referent_candidates: tuple[SourceSpan, ...] = ()
    confidence: float = 0.0
    compiler_version: str = COMPILER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.node_type, NodeType):
            raise ValueError("node_type must be a NodeType")
        if not isinstance(self.source_span, SourceSpan) or not isinstance(self.provenance, ModelDecisionProvenance):
            raise ValueError("semantic nodes require a source span and model provenance")
        if not isinstance(self.qualifiers, SemanticQualifiers) or not isinstance(self.ambiguity, AmbiguityState):
            raise ValueError("semantic nodes require typed qualifiers and ambiguity")
        if not isinstance(self.referent_candidates, tuple) or any(
            not isinstance(item, SourceSpan) for item in self.referent_candidates
        ):
            raise ValueError("referent candidates must be immutable exact SourceSpan values")
        ordered_referents = tuple(sorted(
            self.referent_candidates, key=lambda item: (item.local_start, item.local_end, item.text),
        ))
        if len({json.dumps(item.to_dict(), sort_keys=True) for item in ordered_referents}) != len(ordered_referents):
            raise ValueError("referent candidates must be unique")
        object.__setattr__(self, "referent_candidates", ordered_referents)
        if self.ambiguity is AmbiguityState.MULTIPLE_CANDIDATES and len(self.referent_candidates) < 2:
            raise ValueError("MULTIPLE_CANDIDATES requires at least two referent candidates")
        if self.referent_candidates and self.ambiguity not in {
            AmbiguityState.AMBIGUOUS, AmbiguityState.MULTIPLE_CANDIDATES,
        }:
            raise ValueError("referent candidates require an ambiguous state")
        if any(item == self.source_span for item in self.referent_candidates):
            raise ValueError("a semantic node cannot refer to its own exact source mention")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not math.isfinite(float(self.confidence)) or not 0 <= self.confidence <= 1:
            raise ValueError("semantic node confidence must be between zero and one")
        if self.compiler_version != COMPILER_VERSION:
            raise ValueError("semantic node compiler_version is unsupported")

    @property
    def node_id(self) -> str:
        # Identity is source-semantic, not tied to one provider invocation.
        # Provenance remains in the graph content hash and serialized proof.
        return _stable_id("node", self._identity_dict())

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type.value, "source_span": self.source_span.to_dict(),
            "qualifiers": self.qualifiers.to_dict(), "ambiguity": self.ambiguity.value,
        }

    def _content_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type.value, "source_span": self.source_span.to_dict(),
            "provenance": self.provenance.to_dict(), "qualifiers": self.qualifiers.to_dict(),
            "ambiguity": self.ambiguity.value,
            "referent_candidates": [item.to_dict() for item in self.referent_candidates],
            "confidence": self.confidence, "compiler_version": self.compiler_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, **self._content_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticNode":
        keys = {"node_id", "node_type", "source_span", "provenance", "qualifiers",
                "ambiguity", "referent_candidates", "confidence", "compiler_version"}
        _expect_keys(value, keys, "semantic node")
        candidates = value["referent_candidates"]
        if not isinstance(candidates, list):
            raise ValueError("serialized referent candidates must be a list")
        try:
            node = cls(
                NodeType(value["node_type"]), SourceSpan.from_dict(_mapping(value["source_span"], "node span")),
                ModelDecisionProvenance.from_dict(_mapping(value["provenance"], "node provenance")),
                SemanticQualifiers.from_dict(_mapping(value["qualifiers"], "node qualifiers")),
                AmbiguityState(value["ambiguity"]),
                tuple(SourceSpan.from_dict(_mapping(item, "referent candidate span")) for item in candidates),
                value["confidence"], value["compiler_version"],
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid semantic node") from exc
        if value["node_id"] != node.node_id:
            raise ValueError("semantic node ID does not match its content")
        return node


@dataclass(frozen=True)
class SemanticEdge:
    edge_type: EdgeType
    source_node_id: str
    target_node_id: str
    evidence: tuple[SourceSpan, ...]
    provenance: ModelDecisionProvenance
    qualifiers: SemanticQualifiers = field(default_factory=SemanticQualifiers)
    ambiguity: AmbiguityState = AmbiguityState.NONE
    confidence: float = 0.0
    compiler_version: str = COMPILER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.edge_type, EdgeType):
            raise ValueError("edge_type must be an EdgeType")
        _require_nonempty(self.source_node_id, "source_node_id")
        _require_nonempty(self.target_node_id, "target_node_id")
        if self.source_node_id == self.target_node_id:
            raise ValueError("semantic edges cannot be self-referential")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("semantic edges require immutable exact source evidence")
        if any(not isinstance(item, SourceSpan) for item in self.evidence):
            raise ValueError("edge evidence must contain SourceSpan values")
        if not isinstance(self.provenance, ModelDecisionProvenance):
            raise ValueError("semantic edges require model decision provenance")
        if not isinstance(self.qualifiers, SemanticQualifiers) or not isinstance(self.ambiguity, AmbiguityState):
            raise ValueError("semantic edges require typed qualifiers and ambiguity")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not math.isfinite(float(self.confidence)) or not 0 <= self.confidence <= 1:
            raise ValueError("semantic edge confidence must be between zero and one")
        if self.compiler_version != COMPILER_VERSION:
            raise ValueError("semantic edge compiler_version is unsupported")
        ordered = tuple(sorted(self.evidence, key=lambda item: (
            item.source_id, item.window_id, item.local_start, item.local_end, item.text,
        )))
        if len({json.dumps(item.to_dict(), sort_keys=True) for item in ordered}) != len(ordered):
            raise ValueError("edge evidence spans must be unique")
        object.__setattr__(self, "evidence", ordered)

    @property
    def edge_id(self) -> str:
        return _stable_id("edge", self._identity_dict())

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "edge_type": self.edge_type.value, "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "qualifiers": self.qualifiers.to_dict(), "ambiguity": self.ambiguity.value,
        }

    def _content_dict(self) -> dict[str, Any]:
        return {
            "edge_type": self.edge_type.value, "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "provenance": self.provenance.to_dict(), "qualifiers": self.qualifiers.to_dict(),
            "ambiguity": self.ambiguity.value,
            "confidence": self.confidence, "compiler_version": self.compiler_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"edge_id": self.edge_id, **self._content_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticEdge":
        keys = {"edge_id", "edge_type", "source_node_id", "target_node_id", "evidence",
                "provenance", "qualifiers", "ambiguity", "confidence", "compiler_version"}
        _expect_keys(value, keys, "semantic edge")
        evidence = value["evidence"]
        if not isinstance(evidence, list):
            raise ValueError("serialized edge evidence must be a list")
        try:
            edge = cls(
                EdgeType(value["edge_type"]), value["source_node_id"], value["target_node_id"],
                tuple(SourceSpan.from_dict(_mapping(item, "edge evidence span")) for item in evidence),
                ModelDecisionProvenance.from_dict(_mapping(value["provenance"], "edge provenance")),
                SemanticQualifiers.from_dict(_mapping(value["qualifiers"], "edge qualifiers")),
                AmbiguityState(value["ambiguity"]), value["confidence"], value["compiler_version"],
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid semantic edge") from exc
        if value["edge_id"] != edge.edge_id:
            raise ValueError("semantic edge ID does not match its content")
        return edge


def edge_type_supports(edge_type: EdgeType, source: NodeType, target: NodeType) -> bool:
    """Fail-closed structural signatures; these are general semantics, not domain rules."""
    occurrences = {NodeType.EVENT, NodeType.ACTION, NodeType.STATE, NodeType.OUTCOME}
    participants = {NodeType.ENTITY, NodeType.ABILITY_OR_RESOURCE, NodeType.LOCATION_OR_SPACE}
    if edge_type in {EdgeType.ACTOR, EdgeType.EXPERIENCER}:
        return source is NodeType.ENTITY and target in occurrences
    if edge_type in {EdgeType.TARGET, EdgeType.OBJECT}:
        return (source in occurrences and target in participants) or (
            edge_type is EdgeType.OBJECT and source is NodeType.ABILITY_OR_RESOURCE and target in occurrences
        )
    if edge_type in {
        EdgeType.CAUSES, EdgeType.ENABLES, EdgeType.PREVENTS, EdgeType.REQUIRES,
        EdgeType.PURPOSE, EdgeType.RESULT, EdgeType.TERMINATES,
    }:
        return source in occurrences and target in occurrences
    if edge_type in {EdgeType.TEMPORAL_BEFORE, EdgeType.TEMPORAL_AFTER, EdgeType.TEMPORAL_UNTIL}:
        return source in occurrences | {NodeType.TIME} and target in occurrences | {NodeType.TIME}
    if edge_type is EdgeType.CONDITION:
        return source in occurrences | {NodeType.TIME} and target in occurrences
    if edge_type in {EdgeType.NEGATES, EdgeType.MODIFIES}:
        return source in {NodeType.TIME, NodeType.QUANTITY, NodeType.STATE, NodeType.OUTCOME}
    if edge_type is EdgeType.REFERS_TO:
        return True
    if edge_type is EdgeType.CONTRASTS_WITH:
        return source is target or (source in occurrences and target in occurrences)
    return False


@dataclass(frozen=True)
class SemanticGraph:
    """One validated source-local graph and its immutable bronze window."""

    source_id: str
    window_id: str
    source_kind: str
    source_start: int
    source_end: int
    source_text: str
    bronze_source_sha256: str
    source_provenance_sha256: str
    pass0_version: str
    speaker: str | None
    start_timestamp: float | None
    end_timestamp: float | None
    nodes: tuple[SemanticNode, ...]
    edges: tuple[SemanticEdge, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.source_id, "graph source_id")
        _require_nonempty(self.window_id, "graph window_id")
        _require_nonempty(self.source_kind, "graph source_kind")
        _require_nonempty(self.pass0_version, "graph pass0_version")
        _require_sha256(self.bronze_source_sha256, "graph bronze_source_sha256")
        _require_sha256(self.source_provenance_sha256, "graph source_provenance_sha256")
        if self.pass0_version != PASS0_VERSION:
            raise ValueError("graph Pass 0 version is unsupported")
        start = _require_int(self.source_start, "graph source_start")
        end = _require_int(self.source_end, "graph source_end")
        if start < 0 or end <= start:
            raise ValueError("graph source offsets must be ordered and non-negative")
        if not isinstance(self.source_text, str) or not self.source_text:
            raise ValueError("graph source_text must be non-empty")
        if end - start != len(self.source_text):
            raise ValueError("graph source offsets must equal the exact window width")
        suffix_payload = f"{self.source_provenance_sha256}:{start}:{end}:{self.pass0_version}".encode("utf-8")
        suffix = hashlib.sha256(suffix_payload).hexdigest()[:12]
        if re.fullmatch(rf"{re.escape(self.source_id)}:w\d{{4,}}-{suffix}", self.window_id) is None:
            raise ValueError("graph window ID is inconsistent with Pass 0 provenance")
        if self.speaker is not None:
            _require_nonempty(self.speaker, "graph speaker")
        if (self.start_timestamp is None) != (self.end_timestamp is None):
            raise ValueError("graph timestamps must be supplied together")
        if self.start_timestamp is not None:
            for value in (self.start_timestamp, self.end_timestamp):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError("graph timestamps must be finite numbers")
            if self.start_timestamp < 0 or self.end_timestamp < self.start_timestamp:
                raise ValueError("graph timestamps must be ordered and non-negative")
        if not isinstance(self.nodes, tuple) or not isinstance(self.edges, tuple):
            raise ValueError("graph nodes and edges must be immutable tuples")
        if any(not isinstance(item, SemanticNode) for item in self.nodes):
            raise ValueError("graph nodes must contain SemanticNode values")
        if any(not isinstance(item, SemanticEdge) for item in self.edges):
            raise ValueError("graph edges must contain SemanticEdge values")
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda item: item.node_id)))
        object.__setattr__(self, "edges", tuple(sorted(self.edges, key=lambda item: item.edge_id)))
        self.validate()

    def validate(self) -> None:
        node_by_id: dict[str, SemanticNode] = {}
        for node in self.nodes:
            self._validate_span(node.source_span)
            for span in node.qualifiers.spans():
                self._validate_span(span)
            if node.node_id in node_by_id:
                raise ValueError(f"duplicate semantic node ID: {node.node_id}")
            node_by_id[node.node_id] = node
        for node in self.nodes:
            for candidate in node.referent_candidates:
                self._validate_span(candidate)
        edge_ids: set[str] = set()
        for edge in self.edges:
            source = node_by_id.get(edge.source_node_id)
            target = node_by_id.get(edge.target_node_id)
            if source is None or target is None:
                raise ValueError("semantic edge endpoint is absent from the graph")
            if source.source_span.source_id != target.source_span.source_id:
                raise ValueError("cross-source semantic edges are forbidden")
            for span in (*edge.evidence, *edge.qualifiers.spans()):
                self._validate_span(span)
            if not edge_type_supports(edge.edge_type, source.node_type, target.node_type):
                raise ValueError("semantic edge endpoint types are incompatible with its direction")
            evidence_start = min(source.source_span.local_start, target.source_span.local_start)
            evidence_end = max(source.source_span.local_end, target.source_span.local_end)
            if not any(
                span.local_start <= evidence_start and span.local_end >= evidence_end
                for span in edge.evidence
            ):
                raise ValueError("semantic edge evidence must jointly cover both directed endpoints")
            if edge.edge_type is EdgeType.REFERS_TO:
                if source.ambiguity is not AmbiguityState.NONE:
                    raise ValueError("an accepted REFERS_TO edge cannot remain unresolved")
            if edge.edge_id in edge_ids:
                raise ValueError(f"duplicate semantic edge ID: {edge.edge_id}")
            edge_ids.add(edge.edge_id)

    def _validate_span(self, span: SourceSpan) -> None:
        span.validate_against(
            self.source_id, self.window_id, self.source_text,
            window_source_start=self.source_start, speaker=self.speaker,
            start_timestamp=self.start_timestamp, end_timestamp=self.end_timestamp,
        )

    @classmethod
    def from_source_window(
        cls, window: object, nodes: tuple[SemanticNode, ...], edges: tuple[SemanticEdge, ...] = (),
    ) -> "SemanticGraph":
        """Build the graph only after the authoritative Pass 0 contract validates."""
        from pipeline.semantic_source import SemanticSourceWindow

        if not isinstance(window, SemanticSourceWindow):
            raise ValueError("semantic graph requires a SemanticSourceWindow")
        window.validate()
        graph = cls(
            window.source_id, window.window_id, window.source_kind, window.source_start,
            window.source_end, window.text, window.source_content_sha256,
            window.source_provenance_sha256, window.version, window.speaker,
            window.start_ms, window.end_ms, nodes, edges,
        )
        graph.validate_against_source_window(window)
        return graph

    def validate_against_source_window(self, window: object) -> None:
        from pipeline.semantic_source import SemanticSourceWindow

        if not isinstance(window, SemanticSourceWindow):
            raise ValueError("semantic graph requires a SemanticSourceWindow")
        window.validate()
        expected = (
            window.source_id, window.window_id, window.source_kind, window.source_start,
            window.source_end, window.text, window.source_content_sha256,
            window.source_provenance_sha256, window.version, window.speaker,
            window.start_ms, window.end_ms,
        )
        actual = (
            self.source_id, self.window_id, self.source_kind, self.source_start,
            self.source_end, self.source_text, self.bronze_source_sha256,
            self.source_provenance_sha256, self.pass0_version, self.speaker,
            self.start_timestamp, self.end_timestamp,
        )
        if actual != expected:
            raise ValueError("semantic graph source contract does not match Pass 0")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(_canonical_bytes(self._content_dict())).hexdigest()

    def _content_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id, "window_id": self.window_id, "source_kind": self.source_kind,
            "source_start": self.source_start, "source_end": self.source_end,
            "source_text": self.source_text, "bronze_source_sha256": self.bronze_source_sha256,
            "source_provenance_sha256": self.source_provenance_sha256,
            "pass0_version": self.pass0_version, "speaker": self.speaker,
            "start_timestamp": self.start_timestamp, "end_timestamp": self.end_timestamp,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
        }

    def to_artifact(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "content_hash": self.content_hash, **self._content_dict()}

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return json.dumps(self.to_artifact(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return json.dumps(self.to_artifact(), sort_keys=True, ensure_ascii=False, indent=indent)

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> "SemanticGraph":
        keys = {
            "schema_version", "content_hash", "source_id", "window_id", "source_kind",
            "source_start", "source_end", "source_text", "bronze_source_sha256", "pass0_version",
            "source_provenance_sha256",
            "speaker", "start_timestamp", "end_timestamp", "nodes", "edges",
        }
        _expect_keys(value, keys, "semantic graph artifact")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported semantic IR schema version: {value['schema_version']!r}")
        nodes, edges = value["nodes"], value["edges"]
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("serialized graph nodes and edges must be lists")
        graph = cls(
            value["source_id"], value["window_id"], value["source_kind"],
            value["source_start"], value["source_end"], value["source_text"],
            value["bronze_source_sha256"], value["source_provenance_sha256"],
            value["pass0_version"], value["speaker"],
            value["start_timestamp"], value["end_timestamp"],
            tuple(SemanticNode.from_dict(_mapping(item, "serialized node")) for item in nodes),
            tuple(SemanticEdge.from_dict(_mapping(item, "serialized edge")) for item in edges),
        )
        if value["content_hash"] != graph.content_hash:
            raise ValueError("semantic graph content hash does not match its content")
        return graph

    @classmethod
    def from_json(cls, payload: str) -> "SemanticGraph":
        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("semantic graph JSON contains duplicate keys")
                result[key] = value
            return result
        try:
            value = json.loads(payload, object_pairs_hook=unique)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed semantic graph JSON") from exc
        return cls.from_artifact(_mapping(value, "semantic graph artifact"))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _mapping(value, label)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"invalid {label} keys; missing={missing}, extra={extra}")


def _grounded_or_none(value: object) -> GroundedValue | None:
    if value is None:
        return None
    return GroundedValue.from_dict(_mapping(value, "grounded qualifier"))


# Descriptive aliases make the boundary pleasant for callers while retaining
# one canonical implementation and serialized form.
GeneralEdgeType = EdgeType
IRNode = SemanticNode
IREdge = SemanticEdge
IRGraph = SemanticGraph
DecisionProvenance = ModelDecisionProvenance


__all__ = [
    "SCHEMA_VERSION", "COMPILER_VERSION", "NodeType", "EdgeType", "GeneralEdgeType", "AmbiguityState",
    "Polarity", "Modality", "TemporalScope", "Conditionality", "ComparativeDegree",
    "Uncertainty", "QualifierKind", "QualifierCue", "SourceSpan", "GroundedValue", "SemanticQualifiers",
    "ModelDecisionProvenance", "DecisionProvenance", "SemanticNode", "SemanticEdge",
    "SemanticGraph", "IRNode", "IREdge", "IRGraph", "content_sha256", "edge_type_supports",
]
