"""Source-local mention candidates and constrained selection for Phase 2F.

Candidates are deliberately broad and ontology-free.  The model selects IDs
and source-semantic node types; deterministic code owns every offset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Iterable, Mapping

from pipeline.semantic_source import SemanticSourceWindow
from pipeline.semantic_ir import (
    AmbiguityState, ModelDecisionProvenance, NodeType, SemanticNode, SemanticQualifiers, SourceSpan,
    COMPILER_VERSION, content_sha256,
)


MENTION_MAX_WORDS = 32
MENTION_CATALOG_VERSION = f"phase2f-mention-catalog-v3-cross-segment-ngrams-{MENTION_MAX_WORDS}"
MENTION_SELECTION_PROMPT_VERSION_LEGACY = "phase2f-mention-selection-v1"
MENTION_SELECTION_PROMPT_VERSION = "phase2f-mention-selection-v2-focal-anchors"
MENTION_MAX_FOCAL_STARTS_PER_REQUEST = 1
MENTION_FOCAL_CONTEXT_PADDING_CHARACTERS = 160
MENTION_SELECTION_SYSTEM_LEGACY = (
    "Return strict JSON only. Recover low-level source semantics, not strategic propositions or "
    "League ontology. Use only supplied candidate IDs and allowed node types. Abstain when uncertain."
)
MENTION_SELECTION_SYSTEM = (
    "Return strict JSON only. Return one object. Recover exact, atomic, source-semantic mentions; do not "
    "synthesize propositions, strategic concepts, or League ontology. Use only supplied local aliases."
)
NODE_TYPES = frozenset({
    "ENTITY", "ABILITY_OR_RESOURCE", "EVENT", "ACTION", "STATE", "OUTCOME",
    "QUANTITY", "TIME", "LOCATION_OR_SPACE",
})
AMBIGUITY_STATES = frozenset({
    "NONE", "UNKNOWN", "AMBIGUOUS", "MULTIPLE_CANDIDATES", "INSUFFICIENT_EVIDENCE",
})
SELECTION_STATUSES = frozenset({"OK", "NONE", "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"})

_WORD = re.compile(r"[^\W_][\w'’%-]*", re.UNICODE)
_PRONOUNS = frozenset("i me my mine we us our ours you your yours he him his she her hers it its they them their theirs this that these those".split())
_ABILITY_RESOURCES = frozenset(
    "q w e r ult ultimate flash teleport tp ignite exhaust heal barrier cleanse smite mana energy health hp cooldown charge charges stack stacks wave vision ward wards sweeper trinket item items".split()
)
_ACTIONS = frozenset(
    "answer attack back base buy catch chase contest crash die dive engage farm fight flank freeze hit hook jump kill move peel portal push recall respect roam rotate run save spend step stop trade walk ward".split()
)
_EVENTS = frozenset(
    "arrive become begins comes ends expires fail fails lands miss missed misses resets returns spent starts used wastes".split()
)
_TIME_MARKERS = frozenset("after before during first later once then until when whenever while".split())
_CONDITION_MARKERS = frozenset("if unless only whenever when while once after before".split())
_NEGATIONS = frozenset("no not never neither nor can't cant cannot couldn't couldnt doesn't doesnt don't dont won't wont unable without".split())
_MODALS = frozenset("can could may might must should usually sometimes would".split())
_LOCATIONS = frozenset("baron base bot bush bushes dragon jungle lane mid river side tower top".split())


@dataclass(frozen=True)
class MentionCandidate:
    candidate_id: str
    window_id: str
    start: int
    end: int
    absolute_start: int
    absolute_end: int
    source_text: str
    type_hints: tuple[str, ...]
    segment_ids: tuple[str, ...]
    version: str = MENTION_CATALOG_VERSION

    def validate(self, window: SemanticSourceWindow) -> None:
        if self.version != MENTION_CATALOG_VERSION:
            raise ValueError("mention candidate version is unsupported")
        if self.window_id != window.window_id:
            raise ValueError("mention candidate belongs to a different window")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (
            self.start, self.end, self.absolute_start, self.absolute_end,
        )):
            raise ValueError("mention candidate offsets must be integers")
        if not 0 <= self.start < self.end <= len(window.text):
            raise ValueError("mention candidate has invalid offsets")
        if self.absolute_start != window.source_start + self.start or self.absolute_end != window.source_start + self.end:
            raise ValueError("mention candidate absolute offsets are inconsistent")
        if window.text[self.start:self.end] != self.source_text:
            raise ValueError("mention candidate text is not an exact source slice")
        if not set(self.type_hints) <= NODE_TYPES:
            raise ValueError("mention candidate contains an unknown type hint")
        if not isinstance(self.type_hints, tuple) or not isinstance(self.segment_ids, tuple):
            raise ValueError("mention candidate hints and segment IDs must be immutable tuples")
        if tuple(sorted(set(self.type_hints))) != self.type_hints:
            raise ValueError("mention candidate type hints must be sorted and unique")
        expected_segments = tuple(sorted(
            segment.segment_id for segment in window.segments
            if segment.start < self.end and self.start < segment.end
        ))
        if self.segment_ids != expected_segments or not self.segment_ids:
            raise ValueError("mention candidate segment provenance is invalid")
        if self.candidate_id != _candidate_id(window.window_id, self.start, self.end, self.source_text):
            raise ValueError("mention candidate ID is not bound to its exact source span")


@dataclass(frozen=True)
class MentionSelection:
    candidate_id: str
    node_type: str
    confidence: float
    ambiguity: str = "NONE"


@dataclass(frozen=True)
class MentionSelectionResult:
    status: str
    mentions: tuple[MentionSelection, ...]
    raw_output: str
    parsed_output: Mapping[str, Any] | None
    failure: str | None = None
    candidate_ids: tuple[str, ...] = ()
    prompt: str = ""
    model_id: str | None = None
    configuration_sha256: str | None = None
    request_json: str = ""


@dataclass(frozen=True)
class MentionCatalogSelectionResult:
    status: str
    catalog: tuple[MentionCandidate, ...]
    partition_results: tuple[MentionSelectionResult, ...]
    mentions: tuple[MentionSelection, ...]
    failures: tuple[str, ...] = ()
    abstentions: tuple[str, ...] = ()


class MentionProviderError(Exception):
    """The provider failed before returning raw mention output."""


def generate_mention_candidates(
    window: SemanticSourceWindow,
    *,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
    max_ngram_words: int = MENTION_MAX_WORDS,
) -> tuple[MentionCandidate, ...]:
    """Generate a high-recall exact-span catalog before model selection."""
    window.validate()
    if isinstance(max_ngram_words, bool) or not isinstance(max_ngram_words, int) or max_ngram_words != MENTION_MAX_WORDS:
        raise ValueError("max_ngram_words is fixed by the mention catalog version")
    aliases = tuple((alias, "ENTITY") for alias in entity_aliases) + tuple(
        (alias, "ABILITY_OR_RESOURCE") for alias in ability_aliases
    )
    spans: dict[tuple[int, int], set[str]] = {}
    segment_membership: dict[tuple[int, int], set[str]] = {}

    for segment in window.segments:
        spans.setdefault((segment.start, segment.end), set()).update(_hints(segment.source_text))
        segment_membership.setdefault((segment.start, segment.end), set()).add(segment.segment_id)
        words = tuple(_WORD.finditer(window.text, segment.start, segment.end))
        for offset in range(len(words)):
            for size in range(1, min(max_ngram_words, len(words) - offset) + 1):
                start, end = words[offset].start(), words[offset + size - 1].end()
                text = window.text[start:end]
                spans.setdefault((start, end), set()).update(_hints(text))
                segment_membership.setdefault((start, end), set()).add(segment.segment_id)

    # Pass 0 segments are stable discourse hints, not semantic boundaries. Add
    # bounded n-grams over the entire window so punctuation-poor fallback or a
    # sentence boundary cannot deterministically erase an explicit mention.
    words = tuple(_WORD.finditer(window.text))
    for offset in range(len(words)):
        for size in range(1, min(max_ngram_words, len(words) - offset) + 1):
            start, end = words[offset].start(), words[offset + size - 1].end()
            text = window.text[start:end]
            spans.setdefault((start, end), set()).update(_hints(text))
            segment_membership.setdefault((start, end), set()).update(
                segment.segment_id for segment in window.segments
                if segment.start < end and start < segment.end
            )

    for alias, node_type in aliases:
        for start, end in _exact_alias_spans(window.text, alias):
            spans.setdefault((start, end), set()).add(node_type)
            segment_membership.setdefault((start, end), set()).update(
                segment.segment_id for segment in window.segments
                if segment.start < end and start < segment.end
            )

    candidates = []
    for (start, end), hints in sorted(spans.items()):
        if not hints:
            # Untyped spans remain available for model typing; the hints are not
            # a deterministic semantic answer.
            hints = set(NODE_TYPES)
        candidate = MentionCandidate(
            candidate_id=_candidate_id(window.window_id, start, end, window.text[start:end]),
            window_id=window.window_id,
            start=start,
            end=end,
            absolute_start=window.source_start + start,
            absolute_end=window.source_start + end,
            source_text=window.text[start:end],
            type_hints=tuple(sorted(hints)),
            segment_ids=tuple(sorted(segment_membership.get((start, end), ()))),
        )
        candidate.validate(window)
        candidates.append(candidate)
    return tuple(candidates)


def partition_candidate_catalog(
    candidates: tuple[MentionCandidate, ...], *, max_candidates: int = 180,
) -> tuple[tuple[MentionCandidate, ...], ...]:
    """Build bounded focal-start requests without splitting a source anchor.

    The exhaustive catalog remains the deterministic coverage oracle.  A model
    request sees exactly one source-start anchor, with every possible
    end boundary for those anchors kept together.  This prevents the old flat
    catalog slices from asking a partial start band to recover the whole source.
    ``max_candidates`` is a request budget, except that a single indivisible
    focal group is retained whole rather than silently dropping candidates.
    """
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if not candidates:
        return ()
    ordered = tuple(sorted(candidates, key=lambda item: (item.start, item.end, item.candidate_id)))
    if ordered != candidates:
        raise ValueError("mention candidates must use deterministic source order")
    groups: list[tuple[MentionCandidate, ...]] = []
    index = 0
    while index < len(candidates):
        start = candidates[index].start
        end = index + 1
        while end < len(candidates) and candidates[end].start == start:
            end += 1
        groups.append(candidates[index:end])
        index = end
    partitions: list[tuple[MentionCandidate, ...]] = []
    current: list[MentionCandidate] = []
    focal_count = 0
    for group in groups:
        if current and (
            focal_count >= MENTION_MAX_FOCAL_STARTS_PER_REQUEST
            or len(current) + len(group) > max_candidates
        ):
            partitions.append(tuple(current))
            current = []
            focal_count = 0
        current.extend(group)
        focal_count += 1
    if current:
        partitions.append(tuple(current))
    return tuple(partitions)


def mention_selection_prompt(
    window: SemanticSourceWindow,
    candidates: tuple[MentionCandidate, ...],
    *,
    prompt_version: str = MENTION_SELECTION_PROMPT_VERSION,
) -> str:
    if prompt_version == MENTION_SELECTION_PROMPT_VERSION_LEGACY:
        return _legacy_mention_selection_prompt(window, candidates)
    if prompt_version != MENTION_SELECTION_PROMPT_VERSION:
        raise ValueError("mention prompt version is unsupported")
    if len({item.start for item in candidates}) > MENTION_MAX_FOCAL_STARTS_PER_REQUEST:
        raise ValueError("mention focal request contains too many source anchors")
    aliases = _candidate_aliases(candidates)
    focus_aliases = {
        start: f"f{index:02d}" for index, start in enumerate(
            sorted({item.start for item in candidates}), 1,
        )
    }
    values = [
        {
            "id": alias,
            "focus": focus_aliases[item.start],
            "start": item.start,
            "end": item.end,
            "text": item.source_text,
        }
        for alias, item in zip(aliases, candidates)
    ]
    focal_offsets = [
        {"focus": focus_aliases[start], "start": start}
        for start in sorted(focus_aliases)
    ]
    context_start = max(
        0, min((item.start for item in candidates), default=0)
        - MENTION_FOCAL_CONTEXT_PADDING_CHARACTERS,
    )
    context_end = min(
        len(window.text), max((item.end for item in candidates), default=0)
        + MENTION_FOCAL_CONTEXT_PADDING_CHARACTERS,
    )
    return (
        f"LOCAL SOURCE CONTEXT [{context_start},{context_end}) IN WINDOW COORDINATES:\n"
        + window.text[context_start:context_end] + "\n\nFOCAL STARTS:\n"
        + json.dumps(focal_offsets, ensure_ascii=False, separators=(",", ":"))
        + "\n\nCANDIDATES:\n"
        + json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        + "\nSelect every explicit atomic semantic mention beginning at the listed focal starts only. "
          "For each focal start, decide whether one or more mentions begin at that exact character. "
          "Select the smallest complete source span for each "
          "distinct mention; do not use a long clause as a proxy for meaning that begins elsewhere. "
          "Nested mentions with the same start are allowed when each independently denotes source "
          "meaning. ENTITY is a minimal referring expression; ABILITY_OR_RESOURCE is a named or "
          "described ability/resource; ACTION is an intentional act; EVENT is an occurrence; STATE is "
          "a condition that holds; OUTCOME is a resulting situation; QUANTITY, TIME, and "
          "LOCATION_OR_SPACE retain their ordinary source meanings. Preserve grammatical pronouns as "
          "ENTITY mentions but do not resolve or rewrite them. Do not include neighboring conditions, "
          "explanations, or unrelated predicates in a selected span. Return local candidate aliases only. "
          "Return exactly {\"status\":\"OK|NONE|UNKNOWN|AMBIGUOUS|INSUFFICIENT_EVIDENCE\","
          "\"mentions\":[{\"candidate_id\":\"...\",\"node_type\":\"...\","
          "\"confidence\":0.0,\"ambiguity\":\"NONE|UNKNOWN|AMBIGUOUS|INSUFFICIENT_EVIDENCE\"}]} "
          "Use status OK iff at least one mention is selected; otherwise use NONE, UNKNOWN, AMBIGUOUS, "
          "or INSUFFICIENT_EVIDENCE with an empty mentions list. Do not classify qualifiers or resolve "
          "references in this pass."
    )


def select_mentions(
    window: SemanticSourceWindow,
    candidates: tuple[MentionCandidate, ...],
    chat: Callable[..., str],
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    thinking: str | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> MentionSelectionResult:
    """Ask a model for ID/type decisions, retaining raw failure evidence."""
    window.validate()
    for candidate in candidates:
        candidate.validate(window)
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("mention selection requires unique candidates")
    prompt = mention_selection_prompt(window, candidates)
    candidate_ids = tuple(item.candidate_id for item in candidates)
    effective_configuration = {
        "caller_configuration": dict(configuration or {}), "temperature": 0.0,
        "max_tokens": max_tokens, "model": model, "thinking": thinking,
        "prompt_version": MENTION_SELECTION_PROMPT_VERSION,
    }
    request = {"system": MENTION_SELECTION_SYSTEM, "user": prompt, **effective_configuration}
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    config_hash = content_sha256(effective_configuration)
    if not candidates:
        return MentionSelectionResult(
            "NONE", (), "", {"status": "NONE", "mentions": []}, candidate_ids=(),
            prompt=prompt, model_id=model, configuration_sha256=config_hash, request_json=request_json,
        )
    try:
        raw = chat(
            system=MENTION_SELECTION_SYSTEM,
            user=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            model=model,
            thinking=thinking,
        )
    except Exception as exc:
        return MentionSelectionResult(
            "INSUFFICIENT_EVIDENCE", (), "", None,
            type(MentionProviderError()).__name__ + ":" + type(exc).__name__,
            candidate_ids, prompt, model, config_hash, request_json,
        )
    try:
        status, selections, body = parse_mention_selection(
            raw, candidates, prompt_version=MENTION_SELECTION_PROMPT_VERSION,
        )
    except Exception as exc:
        retained_raw = raw if isinstance(raw, str) else repr(raw)
        return MentionSelectionResult(
            "INSUFFICIENT_EVIDENCE", (), retained_raw, None, type(exc).__name__,
            candidate_ids, prompt, model, config_hash, request_json,
        )
    return MentionSelectionResult(
        status, selections, raw, body, None, candidate_ids, prompt, model, config_hash, request_json,
    )


def select_mention_catalog(
    window: SemanticSourceWindow,
    catalog: tuple[MentionCandidate, ...],
    chat: Callable[..., str],
    *,
    model: str | None,
    configuration: Mapping[str, Any],
    max_candidates: int = 180,
    max_tokens: int = 2048,
    thinking: str | None = None,
) -> MentionCatalogSelectionResult:
    """Select every catalog partition while retaining partial failures explicitly."""
    window.validate()
    for candidate in catalog:
        candidate.validate(window)
    partitions = partition_candidate_catalog(catalog, max_candidates=max_candidates)
    results = tuple(
        select_mentions(
            window, partition, chat, model=model, configuration=configuration,
            max_tokens=max_tokens, thinking=thinking,
        )
        for partition in partitions
    )
    mentions = tuple(item for result in results for item in result.mentions)
    if len({item.candidate_id for item in mentions}) != len(mentions):
        raise ValueError("partition aggregation produced duplicate mention candidates")
    failures = tuple(
        f"partition:{index}:{result.failure}" for index, result in enumerate(results, 1) if result.failure
    )
    abstentions = tuple(
        f"partition:{index}:{result.status}" for index, result in enumerate(results, 1)
        if result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"} and not result.failure
    )
    if failures or abstentions:
        status = "PARTIAL" if mentions else next(
            (item for item in ("AMBIGUOUS", "UNKNOWN", "INSUFFICIENT_EVIDENCE")
             if any(result.status == item for result in results)),
            "INSUFFICIENT_EVIDENCE",
        )
    elif mentions:
        status = "OK"
    else:
        statuses = {result.status for result in results}
        status = next((item for item in ("AMBIGUOUS", "UNKNOWN", "INSUFFICIENT_EVIDENCE") if item in statuses), "NONE")
    return MentionCatalogSelectionResult(status, catalog, results, mentions, failures, abstentions)


def assemble_semantic_nodes(
    window: SemanticSourceWindow,
    selection: MentionCatalogSelectionResult,
) -> tuple[SemanticNode, ...]:
    """Resolve model-selected IDs to exact spans; never accept model offsets or free text."""
    window.validate()
    if selection.status not in {"OK", "PARTIAL"} and selection.mentions:
        raise ValueError("non-success mention run cannot smuggle semantic nodes")
    _validate_catalog_selection(window, selection)
    by_id = {candidate.candidate_id: candidate for candidate in selection.catalog}
    result_by_candidate = {
        mention.candidate_id: result
        for result in selection.partition_results for mention in result.mentions
    }
    nodes = []
    for mention in selection.mentions:
        candidate = by_id.get(mention.candidate_id)
        result = result_by_candidate.get(mention.candidate_id)
        if candidate is None or result is None:
            raise ValueError("selected mention is not traceable to its candidate partition")
        candidate.validate(window)
        if result.model_id is None or result.configuration_sha256 is None:
            raise ValueError("selected mention lacks effective model/configuration provenance")
        span = SourceSpan(
            window.source_id, window.window_id, candidate.start, candidate.end, candidate.source_text,
            candidate.absolute_start, candidate.absolute_end, window.speaker, window.start_ms, window.end_ms,
        )
        request = _strict_request(result.request_json)
        provenance = ModelDecisionProvenance(
            decision_id=f"{mention.candidate_id}:mention-selection",
            model_id=result.model_id,
            prompt_version=str(request["prompt_version"]),
            configuration_sha256=result.configuration_sha256,
            input_sha256=content_sha256(json.loads(result.request_json)),
            output_sha256=content_sha256(result.raw_output),
            candidate_ids=result.candidate_ids,
        )
        nodes.append(SemanticNode(
            NodeType(mention.node_type), span, provenance, SemanticQualifiers(),
            AmbiguityState(mention.ambiguity), (), mention.confidence,
            COMPILER_VERSION,
        ))
    return tuple(sorted(nodes, key=lambda item: item.node_id))


def _validate_catalog_selection(
    window: SemanticSourceWindow, selection: MentionCatalogSelectionResult,
) -> None:
    catalog_ids = tuple(item.candidate_id for item in selection.catalog)
    if len(set(catalog_ids)) != len(catalog_ids):
        raise ValueError("mention catalog candidate IDs must be unique")
    by_id = {item.candidate_id: item for item in selection.catalog}
    partition_ids = tuple(
        candidate_id for result in selection.partition_results for candidate_id in result.candidate_ids
    )
    if partition_ids != catalog_ids or len(set(partition_ids)) != len(partition_ids):
        raise ValueError("mention partitions must disjointly and exactly cover the catalog")
    flattened = tuple(mention for result in selection.partition_results for mention in result.mentions)
    if selection.mentions != flattened:
        raise ValueError("aggregate mentions contradict retained partition decisions")
    expected_failures = []
    expected_abstentions = []
    for index, result in enumerate(selection.partition_results, 1):
        partition = tuple(by_id[item] for item in result.candidate_ids)
        if not result.request_json or result.model_id is None or result.configuration_sha256 is None:
            raise ValueError("mention partition lacks reconstructible request provenance")
        request = _strict_request(result.request_json)
        prompt_version = request.get("prompt_version")
        expected_system = _mention_selection_system(prompt_version)
        if request.get("system") != expected_system or request.get("user") != result.prompt:
            raise ValueError("mention partition request contradicts its retained prompt")
        if result.prompt != mention_selection_prompt(
            window, partition, prompt_version=str(prompt_version),
        ):
            raise ValueError("mention partition prompt is not reconstructible")
        if prompt_version == MENTION_SELECTION_PROMPT_VERSION:
            if len({item.start for item in partition}) > MENTION_MAX_FOCAL_STARTS_PER_REQUEST:
                raise ValueError("mention focal partition exceeds its versioned anchor bound")
            for candidate in partition:
                if any(
                    other.start == candidate.start and other.candidate_id not in result.candidate_ids
                    for other in selection.catalog
                ):
                    raise ValueError("mention focal partition splits a source-start decision set")
        effective = {
            key: request[key] for key in (
                "caller_configuration", "temperature", "max_tokens", "model", "thinking", "prompt_version",
            )
        }
        if content_sha256(effective) != result.configuration_sha256 or request.get("model") != result.model_id:
            raise ValueError("mention partition effective configuration hash is invalid")
        if result.failure:
            expected_failures.append(f"partition:{index}:{result.failure}")
            if result.status != "INSUFFICIENT_EVIDENCE" or result.mentions or result.parsed_output is not None:
                raise ValueError("failed mention partition cannot retain accepted decisions")
            if result.failure.startswith(f"{MentionProviderError.__name__}:"):
                if result.raw_output != "" or not re.fullmatch(
                    r"MentionProviderError:[A-Za-z_][A-Za-z0-9_]*", result.failure,
                ):
                    raise ValueError("mention provider failure evidence is inconsistent")
            else:
                if not isinstance(result.raw_output, str) or not result.raw_output:
                    raise ValueError("mention model parse failure must retain nonempty raw output")
                try:
                    parse_mention_selection(
                        result.raw_output, partition, prompt_version=str(prompt_version),
                    )
                except Exception as exc:
                    if result.failure != type(exc).__name__:
                        raise ValueError("mention model parse failure taxonomy is inconsistent") from exc
                else:
                    raise ValueError("claimed mention model parse failure reparses successfully")
            continue
        if result.candidate_ids:
            status, mentions, body = parse_mention_selection(
                result.raw_output, partition, prompt_version=str(prompt_version),
            )
            if (status, mentions, body) != (result.status, result.mentions, result.parsed_output):
                raise ValueError("mention partition raw output contradicts its parsed decision")
        elif (result.status, result.mentions, result.parsed_output) != (
            "NONE", (), {"status": "NONE", "mentions": []},
        ):
            raise ValueError("empty mention partition decision is invalid")
        if result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
            expected_abstentions.append(f"partition:{index}:{result.status}")
    if selection.failures != tuple(expected_failures) or selection.abstentions != tuple(expected_abstentions):
        raise ValueError("aggregate failure/abstention evidence is inconsistent")
    if expected_failures or expected_abstentions:
        expected_status = "PARTIAL" if flattened else next(
            (item for item in ("AMBIGUOUS", "UNKNOWN", "INSUFFICIENT_EVIDENCE")
             if any(result.status == item for result in selection.partition_results)),
            "INSUFFICIENT_EVIDENCE",
        )
    elif flattened:
        expected_status = "OK"
    else:
        statuses = {result.status for result in selection.partition_results}
        expected_status = next(
            (item for item in ("AMBIGUOUS", "UNKNOWN", "INSUFFICIENT_EVIDENCE") if item in statuses),
            "NONE",
        )
    if selection.status != expected_status:
        raise ValueError("aggregate mention status contradicts its partition decisions")


def _strict_request(payload: str) -> Mapping[str, Any]:
    body = _strict_object(payload, allow_single_fence=False)
    expected = {
        "system", "user", "caller_configuration", "temperature", "max_tokens",
        "model", "thinking", "prompt_version",
    }
    if set(body) != expected or not isinstance(body.get("caller_configuration"), Mapping):
        raise ValueError("mention request artifact has an invalid shape")
    return body


def parse_mention_selection(
    raw: str,
    candidates: tuple[MentionCandidate, ...],
    *,
    prompt_version: str = MENTION_SELECTION_PROMPT_VERSION,
) -> tuple[str, tuple[MentionSelection, ...], Mapping[str, Any]]:
    if prompt_version not in {
        MENTION_SELECTION_PROMPT_VERSION_LEGACY, MENTION_SELECTION_PROMPT_VERSION,
    }:
        raise ValueError("mention prompt version is unsupported")
    body = _strict_object(
        raw, allow_single_fence=prompt_version == MENTION_SELECTION_PROMPT_VERSION,
    )
    if set(body) != {"status", "mentions"} or body.get("status") not in SELECTION_STATUSES or not isinstance(body.get("mentions"), list):
        raise ValueError("mention selection has an invalid envelope")
    status = str(body["status"])
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("mention candidate IDs must be unique")
    aliases = dict(zip(_candidate_aliases(candidates), candidates))
    selections = []
    seen: set[str] = set()
    for raw_item in body["mentions"]:
        if not isinstance(raw_item, Mapping) or set(raw_item) != {
            "candidate_id", "node_type", "confidence", "ambiguity",
        }:
            raise ValueError("mention selection item has an invalid shape")
        candidate_id = raw_item.get("candidate_id")
        node_type = raw_item.get("node_type")
        confidence = raw_item.get("confidence")
        ambiguity = raw_item.get("ambiguity")
        if not isinstance(candidate_id, str):
            raise ValueError("mention selection contains a non-string candidate ID")
        candidate = (
            aliases.get(candidate_id)
            if prompt_version == MENTION_SELECTION_PROMPT_VERSION
            else by_id.get(candidate_id)
        )
        if candidate is None:
            raise ValueError("mention selection contains an unknown candidate ID")
        if node_type not in NODE_TYPES:
            raise ValueError("mention selection contains an unknown node type")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("mention confidence must be between zero and one")
        if ambiguity not in AMBIGUITY_STATES - {"MULTIPLE_CANDIDATES"}:
            raise ValueError("mention ambiguity state is invalid")
        key = candidate.candidate_id
        if key in seen:
            raise ValueError("mention selection contains a duplicate candidate ID")
        seen.add(key)
        selections.append(MentionSelection(
            candidate.candidate_id, str(node_type), float(confidence), str(ambiguity),
        ))
    if status == "OK" and not selections:
        raise ValueError("OK mention selection requires at least one mention")
    if status != "OK" and selections:
        raise ValueError("non-OK mention selection cannot smuggle accepted mentions")
    return status, tuple(selections), body


def candidate_coverage(
    candidates: tuple[MentionCandidate, ...],
    reviewed: Iterable[tuple[int, int, str]],
    *,
    window: SemanticSourceWindow,
) -> dict[str, dict[str, int | float]]:
    """Report deterministic exact-span coverage separately by mention family."""
    reviewed_values = tuple(reviewed)
    window.validate()
    for candidate in candidates:
        candidate.validate(window)
    by_span = {(item.start, item.end): item for item in candidates}
    counts: dict[str, list[int]] = {
        key: [0, 0] for key in (
            "entity", "ability_resource", "action_event", "state_outcome", "condition_time", "negation",
        )
    }
    for start, end, node_type in reviewed_values:
        if node_type not in NODE_TYPES | {"CONDITION", "NEGATION"}:
            raise ValueError("reviewed mention uses unknown node type")
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start
            or end > len(window.text)
        ):
            raise ValueError("reviewed mention span is invalid")
        bucket = _coverage_bucket(node_type)
        counts.setdefault(bucket, [0, 0])
        counts[bucket][1] += 1
        if (start, end) in by_span:
            counts[bucket][0] += 1
    return {
        key: {"hit_count": hit, "denominator": total, "recall": hit / total if total else 0.0}
        for key, (hit, total) in sorted(counts.items())
    }


def _coverage_bucket(node_type: str) -> str:
    return {
        "ENTITY": "entity",
        "ABILITY_OR_RESOURCE": "ability_resource",
        "ACTION": "action_event",
        "EVENT": "action_event",
        "STATE": "state_outcome",
        "OUTCOME": "state_outcome",
        "TIME": "condition_time",
        "CONDITION": "condition_time",
        "NEGATION": "negation",
    }.get(node_type, node_type.lower())


def _hints(text: str) -> set[str]:
    tokens = [match.group().lower().replace("’", "'") for match in _WORD.finditer(text)]
    values = set(tokens)
    hints: set[str] = set()
    if values & _PRONOUNS or any(token[:1].isupper() for token in text.split()):
        hints.add("ENTITY")
    if values & _ABILITY_RESOURCES or any(len(token) == 1 and token in "qwer" for token in tokens):
        hints.add("ABILITY_OR_RESOURCE")
    if values & _ACTIONS:
        hints.add("ACTION")
    if values & _EVENTS or any(token.endswith(("ed", "ing")) for token in tokens):
        hints.add("EVENT")
    if values & (_NEGATIONS | _MODALS):
        hints.add("STATE")
    if values & _TIME_MARKERS:
        hints.add("TIME")
    if values & _LOCATIONS:
        hints.add("LOCATION_OR_SPACE")
    if any(token.isdigit() or re.fullmatch(r"\d+(?:\.\d+)?%", token) for token in tokens):
        hints.add("QUANTITY")
    if values & _CONDITION_MARKERS:
        hints.update({"TIME", "EVENT", "STATE"})
    return hints


def _exact_alias_spans(text: str, alias: str) -> tuple[tuple[int, int], ...]:
    alias = alias.strip()
    if not alias:
        return ()
    spans = []
    for match in re.finditer(re.escape(alias), text, re.IGNORECASE):
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end():]
        if before and (before.isalnum() or before in "_'’"):
            continue
        if after and (after[0].isalnum() or after[0] == "_"):
            continue
        if after.startswith(("'s", "’s")) or not after or after[0] not in "'’":
            spans.append((match.start(), match.end()))
    return tuple(spans)


def _candidate_id(window_id: str, start: int, end: int, source_text: str) -> str:
    raw = json.dumps(
        [MENTION_CATALOG_VERSION, window_id, start, end, source_text],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return f"{window_id}:m{hashlib.sha256(raw).hexdigest()[:20]}"


def _candidate_aliases(candidates: tuple[MentionCandidate, ...]) -> tuple[str, ...]:
    width = max(3, len(str(len(candidates))))
    return tuple(f"c{index:0{width}d}" for index in range(1, len(candidates) + 1))


def _mention_selection_system(prompt_version: object) -> str:
    if prompt_version == MENTION_SELECTION_PROMPT_VERSION:
        return MENTION_SELECTION_SYSTEM
    if prompt_version == MENTION_SELECTION_PROMPT_VERSION_LEGACY:
        return MENTION_SELECTION_SYSTEM_LEGACY
    raise ValueError("mention prompt version is unsupported")


def _legacy_mention_selection_prompt(
    window: SemanticSourceWindow, candidates: tuple[MentionCandidate, ...],
) -> str:
    values = [
        {"id": item.candidate_id, "text": item.source_text, "type_hints": item.type_hints}
        for item in candidates
    ]
    return (
        "SOURCE WINDOW:\n" + window.text + "\n\nCANDIDATES:\n"
        + json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        + "\nSelect every explicit semantic mention needed to preserve what the source says. "
          "Select IDs only; hints are nonbinding. Do not infer champion identities for pronouns. "
          "Return exactly {\"status\":\"OK|NONE|UNKNOWN|AMBIGUOUS|INSUFFICIENT_EVIDENCE\","
          "\"mentions\":[{\"candidate_id\":\"...\",\"node_type\":\"...\","
          "\"confidence\":0.0,\"ambiguity\":\"NONE|UNKNOWN|AMBIGUOUS|INSUFFICIENT_EVIDENCE\"}]} "
          "Do not classify qualifiers or resolve references in this pass."
    )


def _strict_object(raw: str, *, allow_single_fence: bool) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result
    if not isinstance(raw, str):
        raise ValueError("mention selection output must be a string")
    payload = raw
    if allow_single_fence and raw.startswith("```json\n") and raw.endswith("\n```"):
        payload = raw[len("```json\n"):-len("\n```")]
    try:
        body = json.loads(payload, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("mention selection returned malformed JSON") from exc
    if not isinstance(body, Mapping):
        raise ValueError("mention selection must return a JSON object")
    return body
