"""Deterministic Pass 0 source windows for the Phase 2F semantic compiler.

This module owns source identity, exact offsets, stable segmentation, and
round-trip validation only.  It deliberately contains no proposition, causal,
or League ontology logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import re
from typing import Literal


PASS0_SEGMENT_WORDS = 32
PASS0_VERSION = f"phase2f-pass0-v0-segments-{PASS0_SEGMENT_WORDS}"
_TOKEN = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.!?]+[\"'’”)}\]]*\s+")
_DISCOURSE = re.compile(
    r"\s+(?=(?:because|but|so|then|when|whenever|once|if|unless|until|while|after|before|although|however)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BronzeSource:
    """Immutable bronze text and the metadata that genuinely exists."""

    source_id: str
    text: str
    source_kind: str = "transcript"
    speaker: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*:[^\s:][^\s]*", self.source_id) is None
        ):
            raise ValueError("source_id must be stable and namespaced")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("bronze source text must be nonempty")
        if not isinstance(self.source_kind, str) or not self.source_kind.strip():
            raise ValueError("source_kind must be nonempty")
        if self.speaker is not None and (not isinstance(self.speaker, str) or not self.speaker.strip()):
            raise ValueError("speaker must be a nonempty string when present")
        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError("source timestamps must be both present or both absent")
        if self.start_ms is not None:
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.start_ms, self.end_ms)):
                raise ValueError("source timestamps must be integer milliseconds")
            if self.start_ms < 0 or self.end_ms <= self.start_ms:
                raise ValueError("source timestamps are invalid")
        if not isinstance(self.metadata, tuple):
            raise ValueError("source metadata must be an immutable tuple")
        if any(
            not isinstance(item, tuple) or len(item) != 2
            or not isinstance(item[0], str) or not item[0].strip()
            or not isinstance(item[1], str)
            for item in self.metadata
        ):
            raise ValueError("source metadata must contain nonempty string keys and string values")
        if len({key for key, _ in self.metadata}) != len(self.metadata):
            raise ValueError("source metadata keys must be unique")

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def provenance_sha256(self) -> str:
        payload = {
            "source_id": self.source_id, "text": self.text, "source_kind": self.source_kind,
            "speaker": self.speaker, "start_ms": self.start_ms, "end_ms": self.end_ms,
            "metadata": sorted(self.metadata),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def context_sha256(self) -> str:
        payload = {
            "source_id": self.source_id, "source_kind": self.source_kind, "speaker": self.speaker,
            "start_ms": self.start_ms, "end_ms": self.end_ms,
            "metadata": sorted(self.metadata),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SourceSegment:
    segment_id: str
    window_id: str
    kind: Literal["sentence", "discourse", "fallback"]
    start: int
    end: int
    absolute_start: int
    absolute_end: int
    source_text: str
    version: str = PASS0_VERSION

    def validate(self, window: "SemanticSourceWindow") -> None:
        if not isinstance(self.segment_id, str) or not isinstance(self.window_id, str):
            raise ValueError("segment IDs must be strings")
        if self.kind not in {"sentence", "discourse", "fallback"}:
            raise ValueError("segment kind is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.start, self.end, self.absolute_start, self.absolute_end)
        ):
            raise ValueError("segment offsets must be integers")
        if not isinstance(self.source_text, str) or not self.source_text:
            raise ValueError("segment source text must be nonempty")
        if self.window_id != window.window_id:
            raise ValueError("segment belongs to a different window")
        if self.version != PASS0_VERSION or self.version != window.version:
            raise ValueError("segment version does not match the Pass 0 window")
        if not 0 <= self.start < self.end <= len(window.text):
            raise ValueError("segment has invalid local offsets")
        if self.absolute_start != window.source_start + self.start or self.absolute_end != window.source_start + self.end:
            raise ValueError("segment absolute offsets do not match the window")
        if window.text[self.start:self.end] != self.source_text:
            raise ValueError("segment text does not match exact window source")
        expected = f"{window.window_id}:s{window.segments.index(self) + 1:03d}"
        if self.segment_id != expected:
            raise ValueError("segment ID is not stable for its ordered span")


@dataclass(frozen=True)
class SemanticSourceWindow:
    """Exact, source-local context unit supplied to Pass 1."""

    window_id: str
    source_id: str
    source_kind: str
    source_start: int
    source_end: int
    text: str
    source_content_sha256: str
    source_provenance_sha256: str
    source_context_sha256: str
    speaker: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    segments: tuple[SourceSegment, ...] = field(default_factory=tuple)
    version: str = PASS0_VERSION

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def reconstruct(self) -> str:
        """Return the exact bronze slice; segmentation is never destructive."""
        return self.text

    def validate(self, source: BronzeSource | None = None) -> None:
        if not isinstance(self.source_id, str) or not isinstance(self.source_kind, str):
            raise ValueError("window source identity must use strings")
        if not self.source_kind.strip():
            raise ValueError("window source kind must be nonempty")
        if self.speaker is not None and (not isinstance(self.speaker, str) or not self.speaker.strip()):
            raise ValueError("window speaker must be nonempty when present")
        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError("window timestamps must be supplied together")
        if self.start_ms is not None and (
            any(isinstance(value, bool) or not isinstance(value, int) for value in (self.start_ms, self.end_ms))
            or self.start_ms < 0 or self.end_ms <= self.start_ms
        ):
            raise ValueError("window timestamps must be valid integer milliseconds")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.source_start, self.source_end)):
            raise ValueError("window source offsets must be integers")
        if not isinstance(self.segments, tuple):
            raise ValueError("window segments must be an immutable tuple")
        if not isinstance(self.metadata, tuple):
            raise ValueError("window metadata must be an immutable tuple")
        if any(
            not isinstance(item, tuple) or len(item) != 2
            or not isinstance(item[0], str) or not item[0].strip() or not isinstance(item[1], str)
            for item in self.metadata
        ) or len({key for key, _ in self.metadata}) != len(self.metadata):
            raise ValueError("window metadata is malformed")
        if not isinstance(self.window_id, str) or not self.window_id.startswith(self.source_id + ":w"):
            raise ValueError("window ID must retain the complete source prefix")
        if self.version != PASS0_VERSION:
            raise ValueError("window uses an unsupported Pass 0 version")
        if not 0 <= self.source_start < self.source_end:
            raise ValueError("window has invalid source offsets")
        if self.source_end - self.source_start != len(self.text):
            raise ValueError("window offsets do not match its exact text length")
        if not self.text.strip():
            raise ValueError("window text must be nonempty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_content_sha256):
            raise ValueError("window source hash must be a lowercase SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_provenance_sha256):
            raise ValueError("window source provenance hash must be a lowercase SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_context_sha256):
            raise ValueError("window source context hash must be a lowercase SHA-256 digest")
        context_payload = {
            "source_id": self.source_id, "source_kind": self.source_kind, "speaker": self.speaker,
            "start_ms": self.start_ms, "end_ms": self.end_ms, "metadata": sorted(self.metadata),
        }
        context_hash = hashlib.sha256(json.dumps(
            context_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        if self.source_context_sha256 != context_hash:
            raise ValueError("window context does not match its retained source context hash")
        expected_suffix = _window_identity_suffix(
            self.source_provenance_sha256, self.source_start, self.source_end, self.version,
        )
        expected_id = rf"{re.escape(self.source_id)}:w\d{{4,}}-{expected_suffix}"
        if not isinstance(self.window_id, str) or re.fullmatch(expected_id, self.window_id) is None:
            raise ValueError("window ID is not bound to its exact source span and version")
        if len({segment.segment_id for segment in self.segments}) != len(self.segments):
            raise ValueError("window segment IDs must be unique")
        if not self.segments:
            raise ValueError("window requires deterministic source segments")
        last_end = 0
        for segment in self.segments:
            segment.validate(self)
            if segment.start < last_end:
                raise ValueError("window segments must not overlap")
            if self.text[last_end:segment.start].strip():
                raise ValueError("window segmentation discarded non-whitespace source text")
            last_end = segment.end
        if self.text[last_end:].strip():
            raise ValueError("window segmentation discarded trailing source text")
        base = replace(self, segments=())
        if self.segments != segment_window(base):
            raise ValueError("window segments do not match the versioned deterministic algorithm")
        if source is not None:
            if self.source_id != source.source_id or self.source_kind != source.source_kind:
                raise ValueError("window source identity mismatch")
            if self.source_content_sha256 != source.content_sha256:
                raise ValueError("window source hash mismatch")
            if self.source_provenance_sha256 != source.provenance_sha256:
                raise ValueError("window source provenance hash mismatch")
            if self.source_context_sha256 != source.context_sha256:
                raise ValueError("window source context hash mismatch")
            if self.source_end > len(source.text) or source.text[self.source_start:self.source_end] != self.text:
                raise ValueError("window cannot round-trip to bronze")
            if (self.speaker, self.start_ms, self.end_ms, self.metadata) != (
                source.speaker, source.start_ms, source.end_ms, source.metadata,
            ):
                raise ValueError("window contextual metadata does not match bronze")


def build_context_windows(
    source: BronzeSource,
    *,
    target_words: int = 120,
    overlap_words: int = 24,
) -> tuple[SemanticSourceWindow, ...]:
    """Build deterministic overlapping windows on token boundaries.

    Windowing never fabricates punctuation or timestamps.  When a source has
    timestamps only at document level, those exact bounds remain attached as
    context rather than being falsely interpolated per token.
    """
    if (
        isinstance(target_words, bool) or not isinstance(target_words, int)
        or isinstance(overlap_words, bool) or not isinstance(overlap_words, int)
        or target_words <= 0 or overlap_words < 0 or overlap_words >= target_words
    ):
        raise ValueError("target_words must be positive and exceed overlap_words")
    tokens = tuple(_TOKEN.finditer(source.text))
    if not tokens:
        raise ValueError("bronze source contains no tokens")
    stride = target_words - overlap_words
    bounds: list[tuple[int, int]] = []
    for offset in range(0, len(tokens), stride):
        selected = tokens[offset:offset + target_words]
        if not selected:
            break
        start = selected[0].start()
        end = selected[-1].end()
        if bounds and end <= bounds[-1][1]:
            continue
        bounds.append((start, end))
        if offset + target_words >= len(tokens):
            break
    windows = []
    for index, (start, end) in enumerate(bounds, 1):
        window_id = _window_id(source, start, end, index)
        text = source.text[start:end]
        window = SemanticSourceWindow(
            window_id=window_id,
            source_id=source.source_id,
            source_kind=source.source_kind,
            source_start=start,
            source_end=end,
            text=text,
            source_content_sha256=source.content_sha256,
            source_provenance_sha256=source.provenance_sha256,
            source_context_sha256=source.context_sha256,
            speaker=source.speaker,
            start_ms=source.start_ms,
            end_ms=source.end_ms,
            metadata=source.metadata,
        )
        segments = segment_window(window)
        window = SemanticSourceWindow(**{**window.__dict__, "segments": segments})
        window.validate(source)
        windows.append(window)
    return tuple(windows)


def window_from_exact_span(source: BronzeSource, start: int, end: int, *, index: int = 1) -> SemanticSourceWindow:
    """Build one verified window from explicit deterministic bronze offsets."""
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or not 0 <= start < end <= len(source.text)
    ):
        raise ValueError("explicit window span is invalid")
    if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
        raise ValueError("window index must be a positive integer")
    window = SemanticSourceWindow(
        window_id=_window_id(source, start, end, index),
        source_id=source.source_id,
        source_kind=source.source_kind,
        source_start=start,
        source_end=end,
        text=source.text[start:end],
        source_content_sha256=source.content_sha256,
        source_provenance_sha256=source.provenance_sha256,
        source_context_sha256=source.context_sha256,
        speaker=source.speaker,
        start_ms=source.start_ms,
        end_ms=source.end_ms,
        metadata=source.metadata,
    )
    window = SemanticSourceWindow(**{**window.__dict__, "segments": segment_window(window)})
    window.validate(source)
    return window


def segment_window(
    window: SemanticSourceWindow, *, fallback_words: int = PASS0_SEGMENT_WORDS,
) -> tuple[SourceSegment, ...]:
    """Return stable sentence/discourse units or punctuation-poor fallbacks."""
    if (
        isinstance(fallback_words, bool) or not isinstance(fallback_words, int)
        or fallback_words != PASS0_SEGMENT_WORDS
    ):
        raise ValueError("fallback_words is fixed by the Pass 0 version")
    raw_bounds = _sentence_bounds(window.text)
    expanded: list[tuple[int, int]] = []
    kinds: list[str] = []
    for start, end in raw_bounds:
        discourse = _discourse_bounds(window.text, start, end)
        for discourse_start, discourse_end in discourse:
            bounded = _bounded_span(window.text, discourse_start, discourse_end, fallback_words)
            expanded.extend(bounded)
            if len(bounded) > 1:
                kinds.extend(["fallback"] * len(bounded))
            else:
                kinds.append("sentence" if len(discourse) == 1 else "discourse")
    raw_bounds = expanded
    segments = []
    filtered = []
    for (start, end), kind in zip(raw_bounds, kinds):
        start, end = _trim(window.text, start, end)
        if start >= end:
            continue
        filtered.append((start, end, kind))
    for index, (start, end, kind) in enumerate(filtered, 1):
        segments.append(SourceSegment(
            segment_id=f"{window.window_id}:s{index:03d}",
            window_id=window.window_id,
            kind=kind,  # type: ignore[arg-type]
            start=start,
            end=end,
            absolute_start=window.source_start + start,
            absolute_end=window.source_start + end,
            source_text=window.text[start:end],
        ))
    return tuple(segments)


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        while end > match.start() and text[end - 1].isspace():
            end -= 1
        bounds.append((start, end))
        start = match.end()
    bounds.append((start, len(text)))
    return bounds


def _discourse_bounds(text: str, start: int, end: int) -> list[tuple[int, int]]:
    bounds = []
    cursor = start
    for match in _DISCOURSE.finditer(text, start, end):
        bounds.append((cursor, match.start()))
        cursor = match.end()
    bounds.append((cursor, end))
    return bounds


def _fallback_bounds(text: str, words: int) -> list[tuple[int, int]]:
    tokens = tuple(_TOKEN.finditer(text))
    return [(part[0].start(), part[-1].end()) for offset in range(0, len(tokens), words)
            if (part := tokens[offset:offset + words])]


def _bounded_span(text: str, start: int, end: int, words: int) -> list[tuple[int, int]]:
    tokens = tuple(_TOKEN.finditer(text, start, end))
    if len(tokens) <= words:
        return [(start, end)]
    return [
        (part[0].start(), part[-1].end())
        for offset in range(0, len(tokens), words)
        if (part := tokens[offset:offset + words])
    ]


def _window_identity_suffix(source_hash: str, start: int, end: int, version: str) -> str:
    payload = f"{source_hash}:{start}:{end}:{version}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _window_id(source: BronzeSource, start: int, end: int, index: int) -> str:
    suffix = _window_identity_suffix(source.provenance_sha256, start, end, PASS0_VERSION)
    return f"{source.source_id}:w{index:04d}-{suffix}"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
