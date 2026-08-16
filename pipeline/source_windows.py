"""Resolve immutable insight summaries to local bronze transcript windows.

Phase 2D deliberately searches only the transcript belonging to the insight's
``video_id``.  Insights do not currently retain timestamps or source spans, so
the resolver reports how a window was found and keeps ambiguity visible rather
than claiming an exact source span it cannot prove.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3
from typing import Iterable


DEFAULT_WINDOW_WORDS = 120
DEFAULT_WINDOW_STRIDE = 60
DEFAULT_LEXICAL_THRESHOLD = 0.12
MIN_LEXICAL_OVERLAP_TOKENS = 2
LEXICAL_AMBIGUITY_MARGIN = 0.02


@dataclass(frozen=True)
class TranscriptMatch:
    """One candidate transcript span supporting a source-window resolution."""

    start: int
    end: int
    score: float
    overlap_count: int


@dataclass(frozen=True)
class SourceWindow:
    """A local bronze window paired with the immutable insight summary."""

    evidence_id: str
    source_video_id: str
    insight_text: str
    transcript_window: str | None
    window_start: int | None
    window_end: int | None
    alignment_method: str
    alignment_score: float
    exact_source_spans: tuple[tuple[int, int], ...] = ()
    candidate_spans: tuple[TranscriptMatch, ...] = ()

    @property
    def resolved(self) -> bool:
        """Whether this is safe to use as a verified source alignment.

        Ambiguous and externally supplied-but-unverified windows retain text for
        operator review, but cannot silently become source provenance.
        """
        return self.alignment_method in {"explicit_span", "exact_text", "lexical_window"}


class SourceWindowResolver:
    """Read-only resolver from an insight record to its own bronze transcript."""

    def __init__(
        self,
        db_path: str,
        *,
        window_words: int = DEFAULT_WINDOW_WORDS,
        window_stride: int = DEFAULT_WINDOW_STRIDE,
        lexical_threshold: float = DEFAULT_LEXICAL_THRESHOLD,
    ) -> None:
        if window_words <= 0 or window_stride <= 0:
            raise ValueError("window_words and window_stride must be positive")
        if not 0.0 <= lexical_threshold <= 1.0:
            raise ValueError("lexical_threshold must be between 0 and 1")
        self.db_path = db_path
        self.window_words = window_words
        self.window_stride = window_stride
        self.lexical_threshold = lexical_threshold

    def resolve(
        self,
        evidence_id: str,
        *,
        expected_source_id: str | None = None,
        source_span: tuple[int, int] | None = None,
        source_span_verified: bool = False,
    ) -> SourceWindow:
        """Resolve an insight ID without mutating evidence or bronze text.

        ``source_span`` is supported for callers that have externally retained
        alignment metadata.  Only a caller that marks that metadata verified
        receives an exact alignment. Existing insights have no such column, so
        normal Phase 2D calls progress through exact and lexical resolution.
        """
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT i.id, i.video_id, i.text, v.transcription
                FROM insights AS i
                LEFT JOIN videos AS v ON v.video_id = i.video_id
                WHERE i.id = ?
                """,
                (str(evidence_id),),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown insight ID: {evidence_id}")
        insight = str(row["text"] or "").strip()
        video_id = str(row["video_id"] or "").strip()
        transcript = row["transcription"]
        base = dict(evidence_id=str(row["id"]), source_video_id=video_id, insight_text=insight)
        if expected_source_id is not None and expected_source_id != video_id:
            return SourceWindow(**base, transcript_window=None, window_start=None, window_end=None,
                                alignment_method="source_mismatch", alignment_score=0.0)
        if not isinstance(transcript, str) or not transcript.strip():
            return SourceWindow(**base, transcript_window=None, window_start=None, window_end=None,
                                alignment_method="transcript_missing", alignment_score=0.0)
        if source_span is not None:
            return _span_resolution(base, transcript, source_span, source_span_verified)
        exact_spans = tuple(_find_exact_spans(transcript, insight))
        if len(exact_spans) == 1:
            start, end = exact_spans[0]
            return _window_at_span(base, transcript, start, end, "exact_text", 1.0, exact_spans)
        if len(exact_spans) > 1:
            start, end = exact_spans[0]
            return _window_at_span(base, transcript, start, end, "ambiguous_exact", 1.0, exact_spans)
        candidates = tuple(_lexical_candidates(transcript, insight, self.window_words, self.window_stride))
        if (
            not candidates
            or candidates[0].score < self.lexical_threshold
            or candidates[0].overlap_count < MIN_LEXICAL_OVERLAP_TOKENS
        ):
            return SourceWindow(**base, transcript_window=None, window_start=None, window_end=None,
                                alignment_method="unresolved", alignment_score=(candidates[0].score if candidates else 0.0),
                                candidate_spans=candidates)
        if len(candidates) > 1 and candidates[0].score - candidates[1].score <= LEXICAL_AMBIGUITY_MARGIN:
            return SourceWindow(**base, transcript_window=None, window_start=None, window_end=None,
                                alignment_method="ambiguous_lexical", alignment_score=candidates[0].score,
                                candidate_spans=candidates)
        best = candidates[0]
        return SourceWindow(**base, transcript_window=transcript[best.start:best.end], window_start=best.start,
                            window_end=best.end, alignment_method="lexical_window", alignment_score=best.score,
                            candidate_spans=candidates)


def _span_resolution(
    base: dict[str, str], transcript: str, span: tuple[int, int], verified: bool,
) -> SourceWindow:
    if (
        not isinstance(span, tuple) or len(span) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in span)
    ):
        return SourceWindow(**base, transcript_window=None, window_start=None, window_end=None,
                            alignment_method="invalid_source_span", alignment_score=0.0)
    start, end = span
    if start < 0 or end <= start or end > len(transcript):
        return SourceWindow(**base, transcript_window=None, window_start=None, window_end=None,
                            alignment_method="invalid_source_span", alignment_score=0.0)
    return _window_at_span(
        base, transcript, start, end,
        "explicit_span" if verified else "unverified_external_span",
        1.0 if verified else 0.0,
        ((start, end),),
    )


def _window_at_span(
    base: dict[str, str], transcript: str, start: int, end: int, method: str, score: float,
    exact_spans: tuple[tuple[int, int], ...],
) -> SourceWindow:
    left = transcript.rfind(" ", 0, max(0, start - 500))
    right = transcript.find(" ", min(len(transcript), end + 500))
    window_start = 0 if left < 0 else left + 1
    window_end = len(transcript) if right < 0 else right
    return SourceWindow(**base, transcript_window=transcript[window_start:window_end], window_start=window_start,
                        window_end=window_end, alignment_method=method, alignment_score=score,
                        exact_source_spans=exact_spans)


def _find_exact_spans(transcript: str, insight: str) -> Iterable[tuple[int, int]]:
    if not insight:
        return ()
    pattern = re.compile(re.escape(insight), re.IGNORECASE)
    return tuple((match.start(), match.end()) for match in pattern.finditer(transcript))


def _lexical_candidates(
    transcript: str, insight: str, window_words: int, stride: int,
) -> Iterable[TranscriptMatch]:
    tokens = set(_meaningful_tokens(insight))
    if not tokens:
        return ()
    words = list(re.finditer(r"\S+", transcript))
    if not words:
        return ()
    candidates = []
    for offset in range(0, len(words), stride):
        segment = words[offset:offset + window_words]
        if not segment:
            continue
        window_tokens = set(_meaningful_tokens(" ".join(item.group() for item in segment)))
        overlap = tokens & window_tokens
        if not overlap:
            continue
        # Recall-friendly token coverage with a small precision component.
        score = 0.8 * len(overlap) / len(tokens) + 0.2 * len(overlap) / len(window_tokens)
        candidates.append(TranscriptMatch(segment[0].start(), segment[-1].end(), round(score, 4), len(overlap)))
    return tuple(sorted(candidates, key=lambda item: (-item.score, item.start))[:5])


def _meaningful_tokens(text: str) -> Iterable[str]:
    return (
        token for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) > 2 and token not in {"that", "with", "when", "your", "from", "this", "they", "them", "then", "into", "will", "have", "should", "must", "rather", "than", "being", "where"}
    )
