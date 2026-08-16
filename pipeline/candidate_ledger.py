"""Phase 2D provisional candidate ledger; it never writes compiled relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from core.strategic_types import relation_types_conflict
from pipeline.candidate_generation import CandidateSet, generate_candidates
from pipeline.constrained_mapper import MappingSelection
from pipeline.relation_extract import GroundedProposition


class LedgerStatus(str, Enum):
    TRUSTED = "trusted"
    PROVISIONAL_MAPPED = "provisional_mapped"
    PROVISIONAL_UNMAPPED = "provisional_unmapped"
    CONTRADICTED = "contradicted"
    REJECTED = "rejected"
    NO_RELATION = "no_relation"


@dataclass(frozen=True)
class LedgerEvidence:
    evidence_id: str
    source_video_id: str
    proposition: GroundedProposition
    mapper_confidence: float | None
    model_id: str | None = None


@dataclass
class RelationHypothesis:
    subject_id: str | None
    relation_id: str | None
    object_id: str | None
    condition: str | None
    status: LedgerStatus
    evidence: list[LedgerEvidence] = field(default_factory=list)
    contradiction_keys: set[tuple[str, str, str, str | None]] = field(default_factory=set)
    rejection_reason: str | None = None

    @property
    def independent_video_count(self) -> int:
        return len({item.source_video_id for item in self.evidence})

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.evidence_id for item in self.evidence))

    @property
    def mapping_confidences(self) -> tuple[float, ...]:
        return tuple(item.mapper_confidence for item in self.evidence if item.mapper_confidence is not None)

    def key(self) -> tuple[str | None, str | None, str | None, str | None]:
        return (self.subject_id, self.relation_id, self.object_id, _condition_key(self.condition))


class CandidateLedger:
    """In-memory/serializable aggregation boundary before compiled knowledge."""

    def __init__(self, *, evidence_sources: Mapping[str, str], ability_aliases: Mapping[str, str], top_k_concepts: int = 5, trusted_min_independent_videos: int = 2, trusted_min_confidence: float = .75) -> None:
        if trusted_min_independent_videos < 1 or not 0.0 <= trusted_min_confidence <= 1.0:
            raise ValueError("invalid trust thresholds")
        if not evidence_sources or any(not key or not value for key, value in evidence_sources.items()):
            raise ValueError("ledger requires an immutable evidence source catalog")
        if top_k_concepts <= 0:
            raise ValueError("ledger requires a positive candidate top_k")
        self.trusted_min_independent_videos = trusted_min_independent_videos
        self.trusted_min_confidence = trusted_min_confidence
        self._hypotheses: dict[tuple[str | None, str | None, str | None, str | None], RelationHypothesis] = {}
        self._evidence_sources = dict(evidence_sources)
        self._ability_aliases = tuple(sorted((str(key), str(value)) for key, value in ability_aliases.items()))
        self._top_k_concepts = top_k_concepts

    def record(
        self, proposition: GroundedProposition, selection: MappingSelection, *, evidence_id: str,
        source_video_id: str, candidates: CandidateSet, model_id: str | None = None,
    ) -> RelationHypothesis | None:
        if not evidence_id or not source_video_id or evidence_id not in proposition.evidence_ids:
            raise ValueError("ledger evidence must be provenanced by the proposition")
        if self._evidence_sources.get(evidence_id) != source_video_id:
            raise ValueError("ledger evidence source must be registered and immutable")
        _validate_selection(selection, candidates, proposition, self._ability_aliases, self._top_k_concepts)
        if selection.status == "no_relation":
            key = (None, None, None, _condition_key(" | ".join(("no_relation", proposition.condition_source or "", proposition.effect_source))))
            hypothesis = self._hypotheses.setdefault(key, RelationHypothesis(None, None, None, proposition.condition_source, LedgerStatus.NO_RELATION))
            outcome = LedgerEvidence(evidence_id, source_video_id, proposition, None, model_id)
            if not any(item.evidence_id == evidence_id and item.model_id == model_id for item in hypothesis.evidence):
                hypothesis.evidence.append(outcome)
            return hypothesis
        if selection.status == "unmapped":
            unmapping_context = " | ".join(value for value in (proposition.condition_source, proposition.effect_source) if value)
            key = (None, None, None, _condition_key(unmapping_context))
            hypothesis = self._hypotheses.setdefault(key, RelationHypothesis(None, None, None, proposition.condition_source, LedgerStatus.PROVISIONAL_UNMAPPED))
        else:
            hypothesis = self._mapped_hypothesis(proposition, selection)
        evidence = LedgerEvidence(evidence_id, source_video_id, proposition, selection.confidence, model_id)
        if not any(item.evidence_id == evidence_id and item.model_id == model_id for item in hypothesis.evidence):
            hypothesis.evidence.append(evidence)
        self._refresh_statuses()
        return hypothesis

    def record_rejected(
        self, proposition: GroundedProposition, *, evidence_id: str, source_video_id: str, reason: str,
        model_id: str | None = None,
    ) -> RelationHypothesis:
        """Retain a grounded failed candidate without creating a relation hypothesis."""
        if not evidence_id or not source_video_id or evidence_id not in proposition.evidence_ids or not reason.strip():
            raise ValueError("rejected ledger evidence must be provenanced and include a reason")
        if self._evidence_sources.get(evidence_id) != source_video_id:
            raise ValueError("rejected ledger evidence source must be registered and immutable")
        context = " | ".join(value for value in (proposition.condition_source, proposition.effect_source, reason) if value)
        key = (None, None, None, _condition_key(context))
        hypothesis = self._hypotheses.setdefault(key, RelationHypothesis(None, None, None, proposition.condition_source, LedgerStatus.REJECTED, rejection_reason=reason.strip()))
        evidence = LedgerEvidence(evidence_id, source_video_id, proposition, None, model_id)
        if not any(item.evidence_id == evidence_id and item.model_id == model_id for item in hypothesis.evidence):
            hypothesis.evidence.append(evidence)
        return hypothesis

    def hypotheses(self) -> tuple[RelationHypothesis, ...]:
        return tuple(sorted(self._hypotheses.values(), key=lambda item: (item.status.value, item.key())))

    def _mapped_hypothesis(self, proposition: GroundedProposition, selection: MappingSelection) -> RelationHypothesis:
        if not selection.subject_id or not selection.relation_id or not selection.object_id:
            raise ValueError("mapped ledger selection requires candidate IDs")
        condition = proposition.condition_source if selection.condition_index is not None else None
        key = (selection.subject_id, selection.relation_id, selection.object_id, _condition_key(condition))
        return self._hypotheses.setdefault(key, RelationHypothesis(*key, LedgerStatus.PROVISIONAL_MAPPED))

    def _refresh_statuses(self) -> None:
        values = list(self._hypotheses.values())
        for value in values:
            value.contradiction_keys.clear()
        for index, left in enumerate(values):
            if not left.relation_id:
                continue
            for right in values[index + 1:]:
                if not right.relation_id or left.subject_id != right.subject_id or left.object_id != right.object_id:
                    continue
                if _condition_key(left.condition) != _condition_key(right.condition):
                    continue
                if relation_types_conflict(left.relation_id, right.relation_id):
                    left.contradiction_keys.add(right.key())
                    right.contradiction_keys.add(left.key())
        for value in values:
            if value.contradiction_keys:
                value.status = LedgerStatus.CONTRADICTED
            elif value.status == LedgerStatus.REJECTED:
                continue
            elif value.status == LedgerStatus.NO_RELATION:
                continue
            elif value.subject_id is None:
                value.status = LedgerStatus.PROVISIONAL_UNMAPPED
            elif self._is_trusted(value):
                value.status = LedgerStatus.TRUSTED
            else:
                value.status = LedgerStatus.PROVISIONAL_MAPPED

    def _is_trusted(self, hypothesis: RelationHypothesis) -> bool:
        confidences = hypothesis.mapping_confidences
        return (
            hypothesis.independent_video_count >= self.trusted_min_independent_videos
            and bool(confidences)
            and min(confidences) >= self.trusted_min_confidence
        )


def _condition_key(value: str | None) -> str | None:
    return " ".join(value.lower().split()) if value else None


def _validate_selection(
    selection: MappingSelection, candidates: CandidateSet, proposition: GroundedProposition,
    ledger_aliases: tuple[tuple[str, str], ...], ledger_top_k: int,
) -> None:
    signature = (proposition.subject_source, proposition.predicate_source, proposition.effect_source, proposition.condition_source, proposition.evidence_ids)
    if candidates.proposition_signature != signature:
        raise ValueError("ledger candidate set does not belong to proposition")
    if candidates.ability_aliases != ledger_aliases or candidates.top_k_concepts != ledger_top_k:
        raise ValueError("ledger candidate set does not use approved generation policy")
    expected = generate_candidates(proposition, ability_aliases=dict(ledger_aliases), top_k_concepts=ledger_top_k)
    if candidates != expected:
        raise ValueError("ledger candidate set was not deterministically generated")
    if selection.status not in {"mapped", "unmapped", "no_relation"}:
        raise ValueError("ledger selection has invalid status")
    if selection.status != "mapped":
        if any(item is not None for item in (selection.subject_id, selection.relation_id, selection.object_id, selection.condition_index, selection.confidence)):
            raise ValueError("non-mapped ledger selection must not select candidates")
        return
    if not all(isinstance(item, str) for item in (selection.subject_id, selection.relation_id, selection.object_id)):
        raise ValueError("mapped ledger selection requires candidate IDs")
    if selection.subject_id not in {item.id for item in candidates.subject} or selection.relation_id not in {item.id for item in candidates.relation} or selection.object_id not in {item.id for item in candidates.object}:
        raise ValueError("mapped ledger selection is not present in candidate set")
    if selection.condition_index is not None and (not isinstance(selection.condition_index, int) or isinstance(selection.condition_index, bool) or not 0 <= selection.condition_index < len(candidates.condition)):
        raise ValueError("mapped ledger selection has invalid condition")
    if selection.confidence is not None and (not isinstance(selection.confidence, (int, float)) or isinstance(selection.confidence, bool) or not 0.0 <= float(selection.confidence) <= 1.0):
        raise ValueError("mapped ledger selection has invalid confidence")
