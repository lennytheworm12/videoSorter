"""Phase 2I normalized Universal Dependencies syntax layer (offline, CPU).

Phase 2I adds deterministic UD/syntactic evidence (Feature Set C) on top of
the frozen Phase 2H benchmark.  This module owns:

* a normalized, auditable UD parse representation (``UdParse``/``UdSentence``/
  ``UdToken``/``UdWord``) that is independent of the Stanza runtime so
  deterministic feature tests can use fixtures;
* reversible projection of Stanza raw token character offsets to the
  immutable Bronze window text (MWT/contraction words are projected onto
  their parent token offsets and flagged);
* per-candidate token/word alignment with an explicit boundary status
  (``EXACT``/``TOKEN_ALIGNED``/``PARTIAL_BOUNDARY``/``UNALIGNED``/
  ``AMBIGUOUS``), multi-token state, candidate head/root, and explicit
  ambiguity records -- alignment never silently guesses;
* deterministic Feature Set C extraction (dense numeric syntax features plus
  train-only categorical syntax vocabulary via :class:`SyntaxEncoder`).

Every syntax feature is a *learned* feature; no hard endpoint KEEP/DROP rules
live here.  The parser itself is invoked only through local assets with
``DownloadMethod.NONE`` (no network inference/download at evaluation time).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from pipeline.phase2g_silver import canonical_sha256
from pipeline.phase2h_endpoint_scoring import (
    ACTION_TOKENS,
    MODAL_TOKENS,
    NEGATION_TOKENS,
    PRONOUN_TOKENS,
)


PIPELINE_VERSION = "phase2i-syntax-v1"
PARSE_SCHEMA_VERSION = "phase2i-ud-parse-v1"
SYNTAX_FEATURE_SCHEMA_VERSION = "phase2i-syntax-features-v1"

STANZA_LANGUAGE = "en"
STANZA_PACKAGE = "ewt"
STANZA_PROCESSORS = ("tokenize", "mwt", "pos", "lemma", "depparse")
STANZA_VERSION = "1.14.0"
# The exact manifest lock of the approved local Stanza 1.14.0 English EWT
# asset set.  Every real acceptance parse must reproduce this manifest from
# its recorded per-file SHA-256 hashes; fixture parses built directly in unit
# tests are exempt from this lock.
LOCKED_ASSETS_MANIFEST_SHA256 = (
    "ee9b1a3a22e29ac0ddafcafbe00ef742c803094014cccd0e1d2a43b3f38ae357"
)

# Deterministic cap on per-group categorical syntax vocabulary size.  The
# vocabulary itself is always fitted from training windows only; this cap
# merely bounds the one-hot matrix (ties broken alphabetically).
MAX_VALUES_PER_GROUP = 300

BOUNDARY_EXACT = "EXACT"
BOUNDARY_TOKEN_ALIGNED = "TOKEN_ALIGNED"
BOUNDARY_PARTIAL = "PARTIAL_BOUNDARY"
BOUNDARY_UNALIGNED = "UNALIGNED"
BOUNDARY_AMBIGUOUS = "AMBIGUOUS"
BOUNDARY_STATUSES = (
    BOUNDARY_EXACT,
    BOUNDARY_TOKEN_ALIGNED,
    BOUNDARY_PARTIAL,
    BOUNDARY_UNALIGNED,
    BOUNDARY_AMBIGUOUS,
)

# The frozen relation families whose local dependency context becomes a
# categorical Feature Set C signal.
RELATION_DEPS = frozenset((
    "nsubj", "csubj", "obj", "iobj", "obl", "advcl", "acl", "xcomp",
    "ccomp", "conj", "aux", "cop", "mark", "neg", "case", "compound",
    "amod", "advmod",
))

SUBJECT_DEPS = frozenset(("nsubj", "csubj"))
OBJECT_DEPS = frozenset(("obj", "iobj"))
OBLIQUE_DEPS = frozenset(("obl",))
COMPLEMENT_DEPS = frozenset(("xcomp", "ccomp", "advcl", "acl"))
CLAUSE_DEPS = frozenset(("csubj", "ccomp", "advcl", "acl", "xcomp"))
SCOPE_DEPS = frozenset((
    "aux", "cop", "mark", "neg", "case", "advmod", "amod",
))
PREDICATE_UPOS = frozenset(("VERB", "AUX"))

# Roles that make a dependency child real syntactic argument/adjunct
# evidence for a predicate, mapped to the categorical predicate-argument
# context value emitted for Feature Set C.
_PREDICATE_CHILD_ROLE_ORDER = (
    ("subject", SUBJECT_DEPS),
    ("object", OBJECT_DEPS),
    ("oblique", OBLIQUE_DEPS),
    ("complement", COMPLEMENT_DEPS),
    ("aux", frozenset(("aux", "cop"))),
    ("modifier", frozenset(("advmod", "amod", "nummod", "appos"))),
    ("neg", frozenset(("neg",))),
    ("mark", frozenset(("mark",))),
)

_SHA256_HEX_CHARS = frozenset("0123456789abcdef")

_FINITE_XPOS = frozenset(("VBD", "VBP", "VBZ", "MD"))
_MODAL_XPOS = frozenset(("MD",))
_MODAL_FEATS = ("VerbType=Mod",)

_PARSE_RECORD_KEYS = frozenset({
    "parse_sha256", "schema_version", "window_id", "text", "text_sha256",
    "sentences", "parser", "parser_version", "package", "processors",
    "language", "model_assets", "assets_manifest_sha256",
    "pipeline_version",
})
_SENTENCE_RECORD_KEYS = frozenset({
    "sentence_id", "start_char", "end_char", "text", "tokens", "words",
})
_TOKEN_RECORD_KEYS = frozenset({
    "token_id", "sentence_id", "text", "start_char", "end_char",
    "multiword", "word_ids",
})
_WORD_RECORD_KEYS = frozenset({
    "word_id", "sentence_id", "token_id", "text", "lemma", "upos",
    "xpos", "feats", "head", "deprel", "deps", "start_char",
    "end_char", "offset_kind",
})


class Phase2ISyntaxError(ValueError):
    """Base error for Phase 2I syntax contract violations."""


class Phase2IParseError(Phase2ISyntaxError):
    """Stanza parsing or parse-asset contract failed."""


def is_sha256_hex(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX_CHARS for character in value)
    )


def _json_value(value: object) -> object:
    """Convert tuples to lists recursively for baseline JSON comparison."""
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _load_json_strict(path: str | Path) -> Any:
    """Load project JSON without duplicate keys or alternate whitespace.

    The writer contract is UTF-8, two-space indentation, no ASCII escaping,
    and one trailing newline.  Object key order remains semantically defined
    by each record schema and is validated separately where it matters.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise Phase2ISyntaxError(
                    f"duplicate JSON object key {key!r} in {path}",
                )
            output[key] = value
        return output

    def reject_nonfinite_constant(value: str) -> None:
        raise Phase2ISyntaxError(
            f"non-finite JSON number {value!r} in {path}",
        )

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise Phase2ISyntaxError(f"invalid JSON in {path}: {error}") from error
    expected = json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    if raw != expected:
        raise Phase2ISyntaxError(
            f"JSON bytes in {path} are not in canonical project format",
        )
    return value


@dataclass(frozen=True)
class UdWord:
    """One UD word (a dependency-tree node).

    Words inside an MWT/contraction token have no independent character
    offsets in Stanza output; they are projected onto their parent token's
    offsets and ``offset_kind`` is ``MWT_PROJECTED``.  Single-token words
    carry ``offset_kind == TOKEN`` with the raw Stanza offsets, which are
    already relative to the Bronze window text.
    """

    word_id: int
    sentence_id: int
    token_id: int
    text: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    start_char: int | None
    end_char: int | None
    offset_kind: str

    @property
    def is_mwt_projected(self) -> bool:
        return self.offset_kind == "MWT_PROJECTED"


@dataclass(frozen=True)
class UdToken:
    """One Stanza surface token with Bronze-relative char offsets."""

    token_id: int
    sentence_id: int
    text: str
    start_char: int | None
    end_char: int | None
    multiword: bool
    word_ids: tuple[int, ...]


@dataclass(frozen=True)
class UdSentence:
    """One UD sentence with its dependency graph."""

    sentence_id: int
    start_char: int
    end_char: int
    text: str
    tokens: tuple[UdToken, ...]
    words: tuple[UdWord, ...]

    def word_by_id(self, word_id: int) -> UdWord:
        for word in self.words:
            if word.word_id == word_id:
                return word
        raise Phase2ISyntaxError(
            f"sentence {self.sentence_id} has no word {word_id}",
        )

    def word_id_map(self) -> dict[int, UdWord]:
        return {word.word_id: word for word in self.words}

    def depth_map(self) -> dict[int, int]:
        """Distance from the sentence root (root depth 0)."""
        by_id = self.word_id_map()
        depths: dict[int, int] = {}
        visiting: set[int] = set()

        def depth(word_id: int) -> int:
            if word_id in depths:
                return depths[word_id]
            word = by_id[word_id]
            if word.head == 0:
                depths[word_id] = 0
                return 0
            if word.head not in by_id:
                raise Phase2ISyntaxError(
                    f"word {word_id} heads missing word {word.head}",
                )
            if word.head in visiting:
                raise Phase2ISyntaxError(
                    f"dependency cycle detected near word {word_id}",
                )
            visiting.add(word_id)
            depths[word_id] = 1 + depth(word.head)
            visiting.discard(word_id)
            return depths[word_id]

        for word_id in by_id:
            depth(word_id)
        return depths

    def children_map(self) -> dict[int, tuple[int, ...]]:
        by_id = self.word_id_map()
        children: dict[int, list[int]] = {word_id: [] for word_id in by_id}
        for word in self.words:
            if word.head in by_id:
                children[word.head].append(word.word_id)
        return {
            word_id: tuple(sorted(items))
            for word_id, items in children.items()
        }

    def subtree(self, word_id: int) -> frozenset[int]:
        """Descendants of ``word_id`` including itself."""
        children = self.children_map()
        result: set[int] = set()
        pending = [word_id]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            pending.extend(children.get(current, ()))
        return frozenset(result)


@dataclass(frozen=True)
class UdParse:
    """Normalized, auditable parse of one Bronze window text.

    Independent of the Stanza runtime: the same structure can be rebuilt from
    its canonical JSON serialization (``from_dict``) for deterministic
    feature tests.  ``parse_sha256`` covers every deterministic field except
    itself, including the model asset hashes.
    """

    window_id: str
    text: str
    text_sha256: str
    sentences: tuple[UdSentence, ...]
    parser: str
    parser_version: str
    package: str
    processors: tuple[str, ...]
    language: str
    model_assets: tuple[tuple[str, str], ...]
    assets_manifest_sha256: str
    pipeline_version: str
    parse_sha256: str

    def canonical_serialization(self) -> dict[str, Any]:
        return {
            "schema_version": PARSE_SCHEMA_VERSION,
            "window_id": self.window_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "sentences": [
                {
                    "sentence_id": sentence.sentence_id,
                    "start_char": sentence.start_char,
                    "end_char": sentence.end_char,
                    "text": sentence.text,
                    "tokens": [
                        {
                            "token_id": token.token_id,
                            "sentence_id": token.sentence_id,
                            "text": token.text,
                            "start_char": token.start_char,
                            "end_char": token.end_char,
                            "multiword": token.multiword,
                            "word_ids": list(token.word_ids),
                        }
                        for token in sentence.tokens
                    ],
                    "words": [
                        {
                            "word_id": word.word_id,
                            "sentence_id": word.sentence_id,
                            "token_id": word.token_id,
                            "text": word.text,
                            "lemma": word.lemma,
                            "upos": word.upos,
                            "xpos": word.xpos,
                            "feats": word.feats,
                            "head": word.head,
                            "deprel": word.deprel,
                            "deps": word.deps,
                            "start_char": word.start_char,
                            "end_char": word.end_char,
                            "offset_kind": word.offset_kind,
                        }
                        for word in sentence.words
                    ],
                }
                for sentence in self.sentences
            ],
            "parser": self.parser,
            "parser_version": self.parser_version,
            "package": self.package,
            "processors": list(self.processors),
            "language": self.language,
            "model_assets": [
                {"path": path, "sha256": digest}
                for path, digest in self.model_assets
            ],
            "assets_manifest_sha256": self.assets_manifest_sha256,
            "pipeline_version": self.pipeline_version,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.canonical_serialization()
        return {"parse_sha256": self.parse_sha256, **body}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UdParse":
        if not isinstance(data, Mapping):
            raise Phase2ISyntaxError("parse record must be a mapping")
        if set(data) != _PARSE_RECORD_KEYS:
            raise Phase2ISyntaxError("parse record key set is not canonical")
        if data.get("schema_version") != PARSE_SCHEMA_VERSION:
            raise Phase2ISyntaxError(
                f"unsupported parse schema {data.get('schema_version')!r}",
            )
        window_id = data.get("window_id")
        text = data.get("text")
        if not isinstance(window_id, str) or not isinstance(text, str):
            raise Phase2ISyntaxError("parse record requires window_id/text")
        text_sha256 = data.get("text_sha256")
        if text_sha256 != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise Phase2ISyntaxError(
                "parse text_sha256 does not match its Bronze text",
            )
        parser = data.get("parser")
        parser_version = data.get("parser_version")
        if not isinstance(parser, str) or not parser:
            raise Phase2ISyntaxError("parse record requires a parser name")
        if not isinstance(parser_version, str) or not parser_version:
            raise Phase2ISyntaxError(
                "parse record requires a non-empty parser_version",
            )
        package = data.get("package")
        if not isinstance(package, str) or not package:
            raise Phase2ISyntaxError("parse record requires a package")
        language = data.get("language")
        if not isinstance(language, str) or not language:
            raise Phase2ISyntaxError("parse record requires a language")
        raw_processors = data.get("processors")
        if (
            not isinstance(raw_processors, list)
            or not raw_processors
            or not all(type(item) is str and item for item in raw_processors)
        ):
            raise Phase2ISyntaxError(
                "parse record processors must be a non-empty string list",
            )
        processors = tuple(raw_processors)
        pipeline_version = data.get("pipeline_version")
        if not isinstance(pipeline_version, str) or not pipeline_version:
            raise Phase2ISyntaxError(
                "parse record requires a non-empty pipeline_version",
            )
        raw_model_assets = data.get("model_assets")
        if not isinstance(raw_model_assets, list):
            raise Phase2ISyntaxError(
                "parse record requires a model_assets list",
            )
        model_assets_list: list[tuple[str, str]] = []
        for item in raw_model_assets:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"path", "sha256"}
                or type(item.get("path")) is not str
                or type(item.get("sha256")) is not str
            ):
                raise Phase2ISyntaxError(
                    "parse record model asset entry is malformed",
                )
            model_assets_list.append((item["path"], item["sha256"]))
        sentences: list[UdSentence] = []
        raw_sentences = data.get("sentences")
        if not isinstance(raw_sentences, list):
            raise Phase2ISyntaxError("parse record requires a sentences list")
        for raw in raw_sentences:
            if not isinstance(raw, Mapping) or set(
                raw,
            ) != _SENTENCE_RECORD_KEYS:
                raise Phase2ISyntaxError("sentence record key set is invalid")
            sentence_id = raw.get("sentence_id")
            start_char = raw.get("start_char")
            end_char = raw.get("end_char")
            if (
                type(sentence_id) is not int
                or type(start_char) is not int
                or type(end_char) is not int
                or not 0 <= start_char < end_char <= len(text)
            ):
                raise Phase2ISyntaxError(
                    f"sentence {sentence_id!r} has invalid Bronze offsets",
                )
            sentence_text = raw.get("text")
            if sentence_text != text[start_char:end_char]:
                raise Phase2ISyntaxError(
                    f"sentence {sentence_id} text is not its Bronze slice",
                )
            tokens: list[UdToken] = []
            raw_tokens = raw.get("tokens")
            if not isinstance(raw_tokens, list):
                raise Phase2ISyntaxError("sentence requires a tokens list")
            for raw_token in raw_tokens:
                if not isinstance(raw_token, Mapping) or set(
                    raw_token,
                ) != _TOKEN_RECORD_KEYS:
                    raise Phase2ISyntaxError(
                        "token record key set is invalid",
                    )
                token_id = raw_token.get("token_id")
                token_sentence_id = raw_token.get("sentence_id")
                token_text = raw_token.get("text")
                token_start = raw_token.get("start_char")
                token_end = raw_token.get("end_char")
                word_ids = raw_token.get("word_ids")
                if (
                    type(token_id) is not int
                    or type(token_sentence_id) is not int
                    or token_sentence_id != sentence_id
                    or type(token_text) is not str
                    or not token_text
                    or type(token_start) is not int
                    or type(token_end) is not int
                    or not 0 <= token_start < token_end <= len(text)
                    or token_text != text[token_start:token_end]
                    or not isinstance(word_ids, list)
                    or not word_ids
                    or not all(type(item) is int for item in word_ids)
                    or type(raw_token.get("multiword")) is not bool
                ):
                    raise Phase2ISyntaxError(
                        f"token {token_id!r} in sentence {sentence_id} "
                        "is malformed or its text is not its Bronze slice",
                    )
                tokens.append(UdToken(
                    token_id=token_id,
                    sentence_id=sentence_id,
                    text=token_text,
                    start_char=token_start,
                    end_char=token_end,
                    multiword=raw_token["multiword"],
                    word_ids=tuple(word_ids),
                ))
            words: list[UdWord] = []
            raw_words = raw.get("words")
            if not isinstance(raw_words, list):
                raise Phase2ISyntaxError("sentence requires a words list")
            for raw_word in raw_words:
                if not isinstance(raw_word, Mapping) or set(
                    raw_word,
                ) != _WORD_RECORD_KEYS:
                    raise Phase2ISyntaxError("word record key set is invalid")
                word_id = raw_word.get("word_id")
                word_sentence_id = raw_word.get("sentence_id")
                token_id = raw_word.get("token_id")
                word_text = raw_word.get("text")
                head = raw_word.get("head")
                if (
                    type(word_id) is not int
                    or type(word_sentence_id) is not int
                    or word_sentence_id != sentence_id
                    or type(token_id) is not int
                    or type(word_text) is not str
                    or not word_text
                    or type(head) is not int
                    or head < 0
                    or not all(
                        type(raw_word.get(key)) is str
                        for key in (
                            "lemma", "upos", "xpos", "feats", "deprel",
                            "deps", "offset_kind",
                        )
                    )
                ):
                    raise Phase2ISyntaxError(
                        f"word {word_id!r} in sentence {sentence_id} "
                        "is malformed",
                    )
                offset_kind = raw_word.get("offset_kind")
                if offset_kind not in ("TOKEN", "MWT_PROJECTED"):
                    raise Phase2ISyntaxError(
                        f"word {word_id!r} has invalid offset_kind "
                        f"{offset_kind!r}",
                    )
                word_start = raw_word.get("start_char")
                word_end = raw_word.get("end_char")
                if (
                    type(word_start) is not int
                    or type(word_end) is not int
                    or not 0 <= word_start < word_end <= len(text)
                ):
                    raise Phase2ISyntaxError(
                        f"word {word_id!r} has invalid character offsets",
                    )
                words.append(UdWord(
                    word_id=word_id,
                    sentence_id=sentence_id,
                    token_id=token_id,
                    text=word_text,
                    lemma=raw_word["lemma"],
                    upos=raw_word["upos"],
                    xpos=raw_word["xpos"],
                    feats=raw_word["feats"],
                    head=head,
                    deprel=raw_word["deprel"],
                    deps=raw_word["deps"],
                    start_char=word_start,
                    end_char=word_end,
                    offset_kind=offset_kind,
                ))
            sentences.append(UdSentence(
                sentence_id=sentence_id,
                start_char=start_char,
                end_char=end_char,
                text=sentence_text,
                tokens=tuple(tokens),
                words=tuple(words),
            ))
        model_assets = tuple(model_assets_list)
        parse = cls(
            window_id=window_id,
            text=text,
            text_sha256=text_sha256,
            sentences=tuple(sentences),
            parser=parser,
            parser_version=parser_version,
            package=package,
            processors=processors,
            language=language,
            model_assets=model_assets,
            assets_manifest_sha256=data.get("assets_manifest_sha256") or "",
            pipeline_version=pipeline_version,
            parse_sha256=data.get("parse_sha256") or "",
        )
        _validate_parse_structure(parse)
        expected = canonical_sha256(parse.canonical_serialization())
        if parse.parse_sha256 != expected:
            raise Phase2ISyntaxError(
                "parse_sha256 does not self-verify",
            )
        if canonical_sha256(data) != canonical_sha256(parse.to_dict()):
            raise Phase2ISyntaxError(
                "raw parse record differs from canonical serialization",
            )
        return parse


def _validate_parse_structure(parse: UdParse) -> None:
    """Verify every deterministic structural invariant of a normalized
    parse: Bronze slices, unique/ordered ids, token/word links, projected
    offsets, dependency tree validity, and parser/model metadata shape.
    Raises :class:`Phase2ISyntaxError` on the first violation."""
    if not parse.window_id:
        raise Phase2ISyntaxError("parse window_id must be non-empty")
    if not parse.text:
        raise Phase2ISyntaxError("parse text must be non-empty")
    if parse.text_sha256 != hashlib.sha256(
        parse.text.encode("utf-8"),
    ).hexdigest():
        raise Phase2ISyntaxError(
            "parse text_sha256 does not match its Bronze text",
        )
    sentence_ids = [sentence.sentence_id for sentence in parse.sentences]
    if sentence_ids != list(range(1, len(sentence_ids) + 1)):
        raise Phase2ISyntaxError(
            "sentence ids must be unique, ordered, and contiguous from 1",
        )
    previous_sentence_end: int | None = None
    token_covered = bytearray(len(parse.text))
    for sentence in parse.sentences:
        if not (
            0 <= sentence.start_char < sentence.end_char <= len(parse.text)
        ):
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} has invalid Bronze offsets",
            )
        if sentence.text != parse.text[
            sentence.start_char:sentence.end_char
        ]:
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} text is not its Bronze "
                "slice",
            )
        if (
            previous_sentence_end is not None
            and sentence.start_char < previous_sentence_end
        ):
            raise Phase2ISyntaxError(
                "sentences overlap or are out of Bronze order",
            )
        previous_sentence_end = sentence.end_char
        if not sentence.tokens:
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} has no tokens",
            )
        if not sentence.words:
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} has no words",
            )
        token_ids = [token.token_id for token in sentence.tokens]
        if len(set(token_ids)) != len(token_ids) or token_ids != sorted(
            token_ids,
        ):
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} token ids are not unique "
                "and ordered",
            )
        previous_token_end: int | None = None
        for token in sentence.tokens:
            if token.sentence_id != sentence.sentence_id:
                raise Phase2ISyntaxError(
                    f"token {token.token_id} sentence_id mismatch",
                )
            if not (
                sentence.start_char
                <= token.start_char
                < token.end_char
                <= sentence.end_char
            ):
                raise Phase2ISyntaxError(
                    f"token {token.token_id} offsets lie outside its "
                    "sentence",
                )
            if token.text != parse.text[
                token.start_char:token.end_char
            ]:
                raise Phase2ISyntaxError(
                    f"token {token.token_id} text is not its Bronze slice",
                )
            token_covered[token.start_char:token.end_char] = b"\x01" * (
                token.end_char - token.start_char
            )
            if (
                previous_token_end is not None
                and token.start_char < previous_token_end
            ):
                raise Phase2ISyntaxError(
                    f"token {token.token_id} overlaps an earlier token",
                )
            previous_token_end = token.end_char
            if (
                not token.word_ids
                or len(set(token.word_ids)) != len(token.word_ids)
            ):
                raise Phase2ISyntaxError(
                    f"token {token.token_id} word_ids are malformed",
                )
            if token.word_ids[0] != token.token_id:
                raise Phase2ISyntaxError(
                    f"token {token.token_id} id is not its first word id",
                )
        word_ids = [word.word_id for word in sentence.words]
        if word_ids != list(range(1, len(word_ids) + 1)):
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} word ids are not "
                "contiguous 1..N",
            )
        token_word_ids = [
            word_id
            for token in sentence.tokens
            for word_id in token.word_ids
        ]
        if token_word_ids != word_ids:
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} token word_ids do not "
                "partition the sentence words",
            )
        token_by_id = {
            token.token_id: token for token in sentence.tokens
        }
        word_id_set = set(word_ids)
        root_count = 0
        for word in sentence.words:
            if word.sentence_id != sentence.sentence_id:
                raise Phase2ISyntaxError(
                    f"word {word.word_id} sentence_id mismatch",
                )
            token = token_by_id.get(word.token_id)
            if token is None:
                raise Phase2ISyntaxError(
                    f"word {word.word_id} references missing token "
                    f"{word.token_id}",
                )
            if word.word_id not in token.word_ids:
                raise Phase2ISyntaxError(
                    f"word {word.word_id} is not listed by its token",
                )
            if word.offset_kind not in ("TOKEN", "MWT_PROJECTED"):
                raise Phase2ISyntaxError(
                    f"word {word.word_id} offset_kind is invalid",
                )
            if word.offset_kind == "TOKEN" and word.text != token.text:
                raise Phase2ISyntaxError(
                    f"word {word.word_id} text disagrees with its "
                    f"non-MWT token {word.token_id}",
                )
            if (
                word.start_char != token.start_char
                or word.end_char != token.end_char
            ):
                raise Phase2ISyntaxError(
                    f"word {word.word_id} offsets disagree with its token "
                    f"{word.token_id}",
                )
            if word.head != 0 and word.head not in word_id_set:
                raise Phase2ISyntaxError(
                    f"word {word.word_id} head {word.head} is not an "
                    "existing word in the sentence",
                )
            if word.head == 0:
                root_count += 1
        if root_count != 1:
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} must have exactly one "
                f"root; found {root_count}",
            )
        try:
            sentence.depth_map()
        except Phase2ISyntaxError as error:
            raise Phase2ISyntaxError(
                f"sentence {sentence.sentence_id} dependency graph is "
                f"invalid: {error}",
            ) from error
    uncovered = [
        index for index, character in enumerate(parse.text)
        if not character.isspace() and not token_covered[index]
    ]
    if uncovered:
        preview = uncovered[:10]
        raise Phase2ISyntaxError(
            "parser tokens do not cover every non-whitespace Bronze "
            f"character; first uncovered offsets: {preview}",
        )
    if not parse.parser or not parse.parser_version:
        raise Phase2ISyntaxError(
            "parser and parser_version must be non-empty",
        )
    if parse.language != STANZA_LANGUAGE:
        raise Phase2ISyntaxError(
            f"parse language {parse.language!r} != {STANZA_LANGUAGE}",
        )
    if parse.package != STANZA_PACKAGE:
        raise Phase2ISyntaxError(
            f"parse package {parse.package!r} != {STANZA_PACKAGE}",
        )
    if (
        not parse.processors
        or not all(
            isinstance(item, str) and item for item in parse.processors
        )
    ):
        raise Phase2ISyntaxError(
            "parse processors must be non-empty strings",
        )
    if set(parse.processors) != set(STANZA_PROCESSORS):
        raise Phase2ISyntaxError(
            "parse processors must exactly match the required contract",
        )
    seen_paths: set[str] = set()
    for path, digest in parse.model_assets:
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise Phase2ISyntaxError(
                f"model asset path {path!r} is invalid",
            )
        if not is_sha256_hex(digest):
            raise Phase2ISyntaxError(
                f"model asset {path} sha256 is malformed",
            )
        if path in seen_paths:
            raise Phase2ISyntaxError(
                f"model asset path {path!r} is duplicated",
            )
        seen_paths.add(path)
    if parse.assets_manifest_sha256 != assets_manifest_sha256(
        parse.model_assets,
    ):
        raise Phase2ISyntaxError(
            "assets_manifest_sha256 does not match model assets",
        )
    if parse.parser == "stanza":
        if parse.parser_version != STANZA_VERSION:
            raise Phase2ISyntaxError(
                f"stanza parse version {parse.parser_version!r} != "
                f"required {STANZA_VERSION}",
            )
        if parse.processors != STANZA_PROCESSORS:
            raise Phase2ISyntaxError(
                "real stanza parse processors do not exactly match the "
                "Phase 2I contract",
            )
        if parse.pipeline_version != PIPELINE_VERSION:
            raise Phase2ISyntaxError(
                "real stanza parse pipeline_version does not match the "
                "Phase 2I contract",
            )
        if not parse.model_assets:
            raise Phase2ISyntaxError(
                "real stanza parses require non-empty model assets",
            )
        if parse.assets_manifest_sha256 != LOCKED_ASSETS_MANIFEST_SHA256:
            raise Phase2ISyntaxError(
                "real stanza parse asset manifest does not match the "
                "locked Phase 2I provenance",
            )
    if not parse.pipeline_version:
        raise Phase2ISyntaxError(
            "pipeline_version must be non-empty",
        )


# Transient files that must never be treated as model assets: a Hugging Face
# download cache inside the assets dir, Python caches, and partial download
# artifacts.  They are excluded from hashing/provenance and from the
# on-disk/listed verification set.
_TRANSIENT_ASSET_DIR_NAMES = frozenset({
    ".hf_cache", "__pycache__", ".git", "tmp",
})
_TRANSIENT_ASSET_SUFFIXES = (".lock", ".tmp", ".part", ".download")


def _lexically_absolute_path(path: str | Path) -> Path:
    """Absolute lexical form of ``path`` without resolving any symlink.

    ``os.path.abspath`` normalizes the supplied spelling against the process
    working directory but never follows a symlink, so evidence about
    symlinked components (for example ``/proc/self/cwd``) is preserved for
    the fail-closed lstat walk below.
    """
    return Path(os.path.abspath(os.fspath(path)))


def _symlink_ancestor_problems(path: str | Path) -> list[str]:
    """lstat every existing component from the filesystem anchor through
    ``path`` and report every symlink encountered, including the final
    component.

    The walk starts at the filesystem root of the *lexical* absolute path,
    so symlinked ancestors (a temp directory link, ``/proc/self/cwd``, or a
    directly symlinked asset root) are rejected before any walk, hash,
    download, or parser load can read through them.  The walk stops at the
    first missing component and never descends through a detected link.
    """
    supplied = Path(os.fspath(path))
    if ".." in supplied.parts:
        return [
            f"path {str(supplied)!r} contains parent-directory traversal",
        ]
    absolute = _lexically_absolute_path(supplied)
    problems: list[str] = []
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current = current / part
        try:
            status = current.lstat()
        except OSError:
            break
        if stat.S_ISLNK(status.st_mode):
            problems.append(
                f"asset path {str(absolute)!r} traverses symlink "
                f"component {str(current)!r}",
            )
            break
        if not stat.S_ISDIR(status.st_mode):
            break
    return problems


def verify_parser_asset_path(path: str | Path) -> dict[str, Any]:
    """Fail-closed lexical symlink preflight for a parser asset path.

    The setup script runs this before any download, hashing, or parser load.
    It lexically absolutizes the supplied path without resolving any
    component and lstat-walks from the filesystem anchor through the final
    component, rejecting any symlink ancestor or a symlinked root.
    Descendant links inside an existing asset tree are rejected later by
    :func:`verify_assets_provenance`.
    """
    problems = _symlink_ancestor_problems(path)
    return {
        "path": str(_lexically_absolute_path(path)),
        "verified": not problems,
        "problems": problems,
        "reason": (
            "a symlink exists in the lexically supplied parser asset path; "
            "setup never resolves or reads through it"
        ) if problems else None,
    }


def _asset_symlink_problems(assets_dir: Path) -> list[str]:
    """Report every symlink at or below ``assets_dir`` without following it.

    This is a fail-closed preflight: neither this walker nor any later asset
    reader may traverse a symlinked parent directory (for example
    ``assets/en -> external``) or read a symlinked file.  ``os.walk`` with
    ``followlinks=False`` lists symlinked directories in ``dirnames`` but
    does not descend into them, so a symlinked parent is still discovered
    and reported.
    """
    problems: list[str] = []
    if assets_dir.is_symlink():
        problems.append("parser assets dir itself is a symlink")
    try:
        walker = os.walk(assets_dir, followlinks=False)
        for current_root, dirnames, filenames in walker:
            current = Path(current_root)
            for name in sorted([*dirnames, *filenames]):
                entry = current / name
                if entry.is_symlink():
                    problems.append(
                        "asset tree contains symlink "
                        f"{str(entry.relative_to(assets_dir))!r}",
                    )
    except OSError as error:  # pragma: no cover - defensive
        problems.append(f"asset tree walk failed: {error}")
    return problems


def _symlink_component_problem(
    assets_dir: Path,
    relative: str,
) -> str | None:
    """Check every traversed component of ``relative`` for symlinks.

    ``Path.is_symlink`` performs an ``lstat`` and never follows the link, so
    checking the accumulated prefix of each component catches both symlinked
    files and symlinked parent directories without reading through them.
    """
    current = assets_dir
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return (
                f"asset path {relative!r} traverses symlink component "
                f"{str(current.relative_to(assets_dir))!r}"
            )
    return None


def _asset_file_paths(assets_dir: Path):
    """Yield regular, non-symlinked model asset files for hashing.

    A path is skipped when any component of it is a symlink so that hashing
    never reads through a link; :func:`verify_assets_provenance` independently
    rejects the tree before this generator is used by a real parse.
    """
    assets_dir = Path(assets_dir)
    for path in sorted(assets_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(assets_dir)
        if _symlink_component_problem(assets_dir, str(relative)):
            continue
        if path.name == "ASSET_PROVENANCE.json":
            continue
        if any(
            part in _TRANSIENT_ASSET_DIR_NAMES for part in relative.parts
        ):
            continue
        if relative.name.endswith(_TRANSIENT_ASSET_SUFFIXES):
            continue
        yield path


def _model_asset_hashes(assets_dir: Path) -> tuple[tuple[str, str], ...]:
    assets_dir = _lexically_absolute_path(assets_dir)
    ancestor_problems = _symlink_ancestor_problems(assets_dir)
    if ancestor_problems:
        raise Phase2IParseError(
            "parser asset path has a symlinked ancestor or root: "
            + "; ".join(ancestor_problems),
        )
    if not assets_dir.is_dir():
        raise Phase2IParseError(f"parser assets dir missing: {assets_dir}")
    if assets_dir.is_symlink():
        raise Phase2IParseError(
            f"parser assets dir is a symlink: {assets_dir}",
        )
    entries: list[tuple[str, str]] = []
    for path in _asset_file_paths(assets_dir):
        relative = str(path.relative_to(assets_dir))
        entries.append((
            relative,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ))
    if not entries:
        raise Phase2IParseError(f"parser assets dir is empty: {assets_dir}")
    return tuple(entries)


def assets_manifest_sha256(model_assets: Sequence[tuple[str, str]]) -> str:
    return canonical_sha256([
        {"path": path, "sha256": digest}
        for path, digest in sorted(model_assets)
    ])


def _stanza_to_ud_parse(
    doc: Any,
    *,
    window_id: str,
    text: str,
    parser_version: str,
    model_assets: Sequence[tuple[str, str]],
    language: str = STANZA_LANGUAGE,
    package: str = STANZA_PACKAGE,
    processors: Sequence[str] = STANZA_PROCESSORS,
) -> UdParse:
    sentences: list[UdSentence] = []
    for sentence_index, stanza_sentence in enumerate(doc.sentences, 1):
        raw_tokens = list(stanza_sentence.tokens)
        if not raw_tokens:
            continue
        token_records: list[UdToken] = []
        word_records: list[UdWord] = []
        for stanza_token in raw_tokens:
            token_id = int(stanza_token.id[0])
            token_start = (
                int(stanza_token.start_char)
                if stanza_token.start_char is not None else None
            )
            token_end = (
                int(stanza_token.end_char)
                if stanza_token.end_char is not None else None
            )
            token_text = str(stanza_token.text)
            expanded = [w for w in stanza_token.words]
            if token_start is None or token_end is None:
                raise Phase2IParseError(
                    f"stanza token {token_text!r} in sentence "
                    f"{sentence_index} lacks character offsets",
                )
            if not (0 <= token_start < token_end <= len(text)):
                raise Phase2IParseError(
                    f"stanza token {token_text!r} has invalid offsets "
                    f"{token_start}:{token_end}",
                )
            if text[token_start:token_end] != token_text:
                raise Phase2IParseError(
                    f"stanza token surface mismatch: parser produced "
                    f"{token_text!r} for Bronze slice "
                    f"{text[token_start:token_end]!r}",
                )
            if not expanded:
                raise Phase2IParseError(
                    f"stanza token {token_text!r} has no words",
                )
            is_multiword = len(expanded) > 1
            word_ids: list[int] = []
            for word in expanded:
                word_id = int(word.id)
                word_ids.append(word_id)
                word_records.append(UdWord(
                    word_id=word_id,
                    sentence_id=sentence_index,
                    token_id=token_id,
                    text=str(word.text),
                    lemma=str(word.lemma or ""),
                    upos=str(word.upos or ""),
                    xpos=str(word.xpos or ""),
                    feats=str(word.feats or ""),
                    head=int(word.head),
                    deprel=str(word.deprel or ""),
                    deps=str(word.deps or ""),
                    start_char=token_start,
                    end_char=token_end,
                    offset_kind=(
                        "MWT_PROJECTED" if is_multiword else "TOKEN"
                    ),
                ))
            token_records.append(UdToken(
                token_id=token_id,
                sentence_id=sentence_index,
                text=token_text,
                start_char=token_start,
                end_char=token_end,
                multiword=is_multiword,
                word_ids=tuple(word_ids),
            ))
        sentence_start = token_records[0].start_char
        sentence_end = token_records[-1].end_char
        if sentence_start is None or sentence_end is None:
            raise Phase2IParseError(
                f"stanza sentence {sentence_index} lacks token offsets",
            )
        sentences.append(UdSentence(
            sentence_id=sentence_index,
            start_char=sentence_start,
            end_char=sentence_end,
            text=text[sentence_start:sentence_end],
            tokens=tuple(token_records),
            words=tuple(word_records),
        ))
    parse = UdParse(
        window_id=window_id,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        sentences=tuple(sentences),
        parser="stanza",
        parser_version=parser_version,
        package=package,
        processors=tuple(processors),
        language=language,
        model_assets=tuple(model_assets),
        assets_manifest_sha256=assets_manifest_sha256(model_assets),
        pipeline_version=PIPELINE_VERSION,
        parse_sha256="",
    )
    try:
        _validate_parse_structure(parse)
    except Phase2ISyntaxError as error:
        raise Phase2IParseError(
            f"stanza parse failed structural validation: {error}",
        ) from error
    if assets_manifest_sha256(parse.model_assets) != (
        LOCKED_ASSETS_MANIFEST_SHA256
    ):
        raise Phase2IParseError(
            "real stanza parse model assets do not match the locked "
            "Phase 2I provenance manifest",
        )
    parse_sha256 = canonical_sha256(parse.canonical_serialization())
    return UdParse(
        **{
            **parse.__dict__,
            "parse_sha256": parse_sha256,
        },
    )


def parse_window_text(
    text: str,
    window_id: str,
    *,
    assets_dir: str | Path,
    verbose: bool = False,
) -> UdParse:
    """Parse one Bronze window with local Stanza assets only (CPU).

    ``DownloadMethod.NONE`` guarantees no network download at evaluation
    time; a missing model raises :class:`Phase2IParseError` instead.
    Determinism is enforced with a single torch CPU thread.
    """
    try:
        import torch
        import stanza
    except ImportError as error:  # pragma: no cover - environment guard
        raise Phase2IParseError(
            "stanza/torch are required for real parsing: " + str(error),
        ) from error
    if stanza.__version__ != STANZA_VERSION:
        raise Phase2IParseError(
            f"stanza {stanza.__version__} != required {STANZA_VERSION}",
        )
    torch.set_num_threads(1)
    assets_dir = _lexically_absolute_path(assets_dir)
    provenance = verify_assets_provenance(assets_dir)
    if not provenance["verified"]:
        raise Phase2IParseError(
            "parser assets provenance verification failed: "
            + "; ".join(
                provenance.get("problems")
                or [provenance.get("reason", "unknown reason")],
            ),
        )
    if provenance.get("manifest_sha256") != LOCKED_ASSETS_MANIFEST_SHA256:
        raise Phase2IParseError(
            "parser assets manifest does not match the locked Phase 2I "
            "provenance",
        )
    locked_files = {
        entry["path"]: entry["sha256"]
        for entry in provenance.get("files", [])
        if isinstance(entry, Mapping)
    }
    model_assets = _model_asset_hashes(assets_dir)
    if dict(model_assets) != locked_files:
        raise Phase2IParseError(
            "real parser model assets do not match the locked provenance "
            "file set/hashes",
        )
    processors = ",".join(STANZA_PROCESSORS)
    try:
        pipeline = stanza.Pipeline(
            STANZA_LANGUAGE,
            model_dir=str(assets_dir),
            processors=processors,
            package=STANZA_PACKAGE,
            use_gpu=False,
            verbose=verbose,
            download_method=stanza.DownloadMethod.NONE,
        )
        doc = pipeline(text)
    except Exception as error:
        raise Phase2IParseError(
            f"stanza parse failed for window {window_id!r}: {error}",
        ) from error
    return _stanza_to_ud_parse(
        doc,
        window_id=window_id,
        text=text,
        parser_version=stanza.__version__,
        model_assets=model_assets,
    )


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        digits = ""
        for character in part:
            if character.isdigit():
                digits += character
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def _intersects(
    a_start: int, a_end: int, b_start: int, b_end: int,
) -> bool:
    return a_start < b_end and b_start < a_end


def _deprel_family(deprel: str) -> str:
    return deprel.split(":")[0]


def _parse_feats(feats: str) -> dict[str, str]:
    """Parse a UD FEATS string into an exact key/value mapping.

    UD FEATS are ``Key=Value|Key=Value``; the empty/underscore value parses to
    an empty mapping.  Unknown or malformed pairs are ignored, but the
    ``VerbForm`` key is treated exactly: its presence is decisive for
    finiteness, so a present-but-empty value never falls through to XPOS.
    """
    if not feats or feats == "_":
        return {}
    parsed: dict[str, str] = {}
    for pair in feats.split("|"):
        key, separator, value = pair.partition("=")
        if key:
            parsed[key] = value if separator else ""
    return parsed


def _is_finite(word: UdWord) -> bool:
    """Finite iff UD ``VerbForm=Fin`` is present, or -- only when no
    ``VerbForm`` key exists at all -- the XPOS fallback marks a finite
    VERB/AUX form (VBD/VBP/VBZ/MD).  Participles, infinitives, gerunds,
    verbal nouns, and converbs are never finite, even when they carry a
    tense feature such as ``Tense=Past`` on a past participle."""
    parsed = _parse_feats(word.feats)
    if "VerbForm" in parsed:
        return parsed["VerbForm"] == "Fin"
    return word.upos in PREDICATE_UPOS and word.xpos in _FINITE_XPOS


def _is_modal(word: UdWord) -> bool:
    if word.xpos in _MODAL_XPOS:
        return True
    if any(marker in word.feats for marker in _MODAL_FEATS):
        return True
    return (
        word.deprel == "aux"
        and word.lemma.lower() in MODAL_TOKENS
    )


def _is_neg(word: UdWord) -> bool:
    if word.deprel == "neg":
        return True
    if "Polarity=Neg" in word.feats:
        return True
    return word.lemma.lower() in NEGATION_TOKENS


def _is_syntactic_predicate(word: UdWord) -> bool:
    """True syntactic predicate: VERB, finite AUX/VERB, clause dependents, or
    the sentence root.  This is intentionally independent of the bounded
    Phase 2F action-token lexicon, which remains a separate signal."""
    if word.upos == "VERB":
        return True
    if _is_finite(word):
        return True
    if _deprel_family(word.deprel) in CLAUSE_DEPS:
        return True
    return word.head == 0


def _is_structural_clause_root(
    word: UdWord,
    words: Mapping[str, UdWord],
) -> bool:
    """Structural UD clause root, independent of bare finiteness.

    A clause root is one of:

    * the sentence root (``head == 0``);
    * a syntactic predicate governing an ``advcl``/``xcomp``/``ccomp``/
      ``acl``/``csubj`` clause;
    * a finite VERB conjunct whose governor is itself a structural clause
      root (justified coordination).

    Ordinary ``aux``/``cop`` dependents are never clause roots even when
    they are finite (``is``/``has``/``will``/``should`` in aux position).
    A root AUX remains structural because the ``head == 0`` case is checked
    before the aux/cop exclusion.
    """
    if word.head == 0:
        return True
    family = _deprel_family(word.deprel)
    if family in ("aux", "cop"):
        return False
    if family in CLAUSE_DEPS and _is_syntactic_predicate(word):
        return True
    if word.upos == "VERB" and _is_finite(word) and family == "conj":
        governor = words.get(_word_key(word.sentence_id, word.head))
        if governor is None:
            return False
        return _is_structural_clause_root(governor, words)
    return False


def _predicate_role(family: str) -> str | None:
    for role, families in _PREDICATE_CHILD_ROLE_ORDER:
        if family in families:
            return role
    return None


@dataclass(frozen=True)
class CandidateSyntax:
    """Deterministic per-candidate syntax evidence and Feature Set C signals.

    Every field is derived from the normalized parse + immutable Bronze
    candidate span; alignment decisions are explicit and never guessed.
    ``groups`` holds the categorical syntax evidence consumed by
    :class:`SyntaxEncoder` (fitted on training windows only).
    """

    window_id: str
    candidate_id: str
    start: int
    end: int
    text: str
    sentence_ids: tuple[int, ...]
    token_ids: tuple[str, ...]
    word_ids: tuple[str, ...]
    boundary_status: str
    multi_token: bool
    start_aligned: bool
    end_aligned: bool
    spans_multiple_sentences: bool
    ambiguity: tuple[str, ...]
    candidate_head_id: str | None
    candidate_root_id: str | None
    head_word: UdWord | None
    root_word: UdWord | None
    candidate_depth_values: tuple[int, ...]
    root_ids: tuple[str, ...]
    multiple_roots: bool
    syntactic_predicate_ids: tuple[str, ...]
    head_dependent_count: int
    head_child_deprels: tuple[str, ...]
    head_relation_context: tuple[str, ...]
    internal_relation_context: tuple[str, ...]
    crossing_incoming_relation_context: tuple[str, ...]
    crossing_outgoing_relation_context: tuple[str, ...]
    external_governor_lemmas: tuple[str, ...]
    external_governor_uposes: tuple[str, ...]
    external_governor_deprels: tuple[str, ...]
    predicate_argument_context: tuple[str, ...]
    boundary_head_ids: tuple[str, ...]
    external_governor_ids: tuple[str, ...]
    crossing_arc_count: int
    subtree_size: int
    subtree_intersection_count: int
    subtree_exact: bool
    span_connected: bool
    clause_root_ids: tuple[str, ...]
    finite_verb_ids: tuple[str, ...]
    aux_ids: tuple[str, ...]
    modal_ids: tuple[str, ...]
    neg_ids: tuple[str, ...]
    mark_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    relations: tuple[str, ...]
    scope_internal_ids: tuple[str, ...]
    scope_external_ids: tuple[str, ...]
    pronoun_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    predicate_internal_argument: bool
    predicate_external_argument: bool
    predicate_internal_subject: bool
    predicate_external_subject: bool
    predicate_internal_object: bool
    predicate_external_object: bool
    predicate_internal_oblique: bool
    predicate_external_oblique: bool
    predicate_internal_complement: bool
    predicate_external_complement: bool
    predicate_internal_aux: bool
    predicate_external_aux: bool
    predicate_internal_modifier: bool
    predicate_external_modifier: bool
    predicate_internal_neg: bool
    predicate_external_neg: bool
    predicate_internal_mark: bool
    predicate_external_mark: bool
    aux_internal: bool
    aux_external: bool
    neg_internal: bool
    neg_external: bool
    mark_internal: bool
    mark_external: bool
    case_internal: bool
    case_external: bool
    pronoun_inside_governor: bool
    pronoun_outside_governor: bool
    action_internal_argument: bool
    action_external_argument: bool
    action_complement: bool
    action_modifier: bool
    groups: tuple[tuple[str, tuple[str, ...]], ...]

    def group_values(self, group: str) -> tuple[str, ...]:
        for name, values in self.groups:
            if name == group:
                return values
        return ()

    def evidence_sha256(self) -> str:
        payload = {
            "window_id": self.window_id,
            "candidate_id": self.candidate_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "sentence_ids": list(self.sentence_ids),
            "token_ids": list(self.token_ids),
            "word_ids": list(self.word_ids),
            "boundary_status": self.boundary_status,
            "multi_token": self.multi_token,
            "start_aligned": self.start_aligned,
            "end_aligned": self.end_aligned,
            "spans_multiple_sentences": self.spans_multiple_sentences,
            "ambiguity": list(self.ambiguity),
            "candidate_head_id": self.candidate_head_id,
            "candidate_root_id": self.candidate_root_id,
            "candidate_depth_values": list(self.candidate_depth_values),
            "root_ids": list(self.root_ids),
            "multiple_roots": self.multiple_roots,
            "syntactic_predicate_ids": list(self.syntactic_predicate_ids),
            "head_dependent_count": self.head_dependent_count,
            "head_child_deprels": list(self.head_child_deprels),
            "head_relation_context": list(self.head_relation_context),
            "internal_relation_context": list(self.internal_relation_context),
            "crossing_incoming_relation_context": list(
                self.crossing_incoming_relation_context,
            ),
            "crossing_outgoing_relation_context": list(
                self.crossing_outgoing_relation_context,
            ),
            "external_governor_lemmas": list(self.external_governor_lemmas),
            "external_governor_uposes": list(self.external_governor_uposes),
            "external_governor_deprels": list(self.external_governor_deprels),
            "predicate_argument_context": list(
                self.predicate_argument_context,
            ),
            "boundary_head_ids": list(self.boundary_head_ids),
            "external_governor_ids": list(self.external_governor_ids),
            "crossing_arc_count": self.crossing_arc_count,
            "subtree_size": self.subtree_size,
            "subtree_intersection_count": self.subtree_intersection_count,
            "subtree_exact": self.subtree_exact,
            "span_connected": self.span_connected,
            "clause_root_ids": list(self.clause_root_ids),
            "finite_verb_ids": list(self.finite_verb_ids),
            "aux_ids": list(self.aux_ids),
            "modal_ids": list(self.modal_ids),
            "neg_ids": list(self.neg_ids),
            "mark_ids": list(self.mark_ids),
            "case_ids": list(self.case_ids),
            "relations": list(self.relations),
            "scope_internal_ids": list(self.scope_internal_ids),
            "scope_external_ids": list(self.scope_external_ids),
            "pronoun_ids": list(self.pronoun_ids),
            "action_ids": list(self.action_ids),
            "predicate_internal_argument": self.predicate_internal_argument,
            "predicate_external_argument": self.predicate_external_argument,
            "predicate_internal_subject": self.predicate_internal_subject,
            "predicate_external_subject": self.predicate_external_subject,
            "predicate_internal_object": self.predicate_internal_object,
            "predicate_external_object": self.predicate_external_object,
            "predicate_internal_oblique": self.predicate_internal_oblique,
            "predicate_external_oblique": self.predicate_external_oblique,
            "predicate_internal_complement": (
                self.predicate_internal_complement
            ),
            "predicate_external_complement": (
                self.predicate_external_complement
            ),
            "predicate_internal_aux": self.predicate_internal_aux,
            "predicate_external_aux": self.predicate_external_aux,
            "predicate_internal_modifier": self.predicate_internal_modifier,
            "predicate_external_modifier": self.predicate_external_modifier,
            "predicate_internal_neg": self.predicate_internal_neg,
            "predicate_external_neg": self.predicate_external_neg,
            "predicate_internal_mark": self.predicate_internal_mark,
            "predicate_external_mark": self.predicate_external_mark,
            "aux_internal": self.aux_internal,
            "aux_external": self.aux_external,
            "neg_internal": self.neg_internal,
            "neg_external": self.neg_external,
            "mark_internal": self.mark_internal,
            "mark_external": self.mark_external,
            "case_internal": self.case_internal,
            "case_external": self.case_external,
            "pronoun_inside_governor": self.pronoun_inside_governor,
            "pronoun_outside_governor": self.pronoun_outside_governor,
            "action_internal_argument": self.action_internal_argument,
            "action_external_argument": self.action_external_argument,
            "action_complement": self.action_complement,
            "action_modifier": self.action_modifier,
            "groups": [
                {"group": name, "values": list(values)}
                for name, values in self.groups
            ],
        }
        return canonical_sha256(payload)


def _sentence_index(parse: UdParse) -> dict[int, UdSentence]:
    return {sentence.sentence_id: sentence for sentence in parse.sentences}


def _word_key(sentence_id: int, word_id: int) -> str:
    return f"s{sentence_id}:w{word_id}"


def _token_key(sentence_id: int, token_id: int) -> str:
    return f"s{sentence_id}:t{token_id}"


def _sentence_depths(sentence: UdSentence) -> dict[int, int]:
    return sentence.depth_map()


def _compute_candidate(
    parse: UdParse,
    *,
    start: int,
    end: int,
    text: str,
    window_id: str,
    candidate_id: str,
) -> CandidateSyntax:
    if not 0 <= start < end <= len(parse.text):
        raise Phase2ISyntaxError(
            f"candidate {candidate_id} offsets {start}:{end} are outside "
            "the Bronze window",
        )
    if parse.text[start:end] != text:
        raise Phase2ISyntaxError(
            f"candidate {candidate_id} text is not its Bronze slice",
        )
    sentences = _sentence_index(parse)
    hit_tokens: list[tuple[int, UdToken]] = []
    for sentence_id, sentence in sentences.items():
        for token in sentence.tokens:
            if (
                token.start_char is None
                or token.end_char is None
            ):
                continue
            if _intersects(start, end, token.start_char, token.end_char):
                hit_tokens.append((sentence_id, token))
    hit_tokens.sort(key=lambda item: (item[0], item[1].token_id))
    sentence_ids = tuple(sorted({sid for sid, _ in hit_tokens}))
    token_ids = tuple(
        _token_key(sid, token.token_id) for sid, token in hit_tokens
    )
    word_ids_by_sentence: dict[int, list[int]] = {}
    for sid, token in hit_tokens:
        word_ids_by_sentence.setdefault(sid, []).extend(token.word_ids)
    word_ids = tuple(
        _word_key(sid, word_id)
        for sid in sorted(word_ids_by_sentence)
        for word_id in sorted(set(word_ids_by_sentence[sid]))
    )
    if not hit_tokens:
        return CandidateSyntax(
            window_id=window_id,
            candidate_id=candidate_id,
            start=start,
            end=end,
            text=text,
            sentence_ids=(),
            token_ids=(),
            word_ids=(),
            boundary_status=BOUNDARY_UNALIGNED,
            multi_token=False,
            start_aligned=False,
            end_aligned=False,
            spans_multiple_sentences=False,
            ambiguity=("NO_INTERSECTING_TOKENS",),
            candidate_head_id=None,
            candidate_root_id=None,
            head_word=None,
            root_word=None,
            candidate_depth_values=(),
            root_ids=(),
            multiple_roots=False,
            syntactic_predicate_ids=(),
            head_dependent_count=0,
            head_child_deprels=(),
            head_relation_context=(),
            internal_relation_context=(),
            crossing_incoming_relation_context=(),
            crossing_outgoing_relation_context=(),
            external_governor_lemmas=(),
            external_governor_uposes=(),
            external_governor_deprels=(),
            predicate_argument_context=(),
            boundary_head_ids=(),
            external_governor_ids=(),
            crossing_arc_count=0,
            subtree_size=0,
            subtree_intersection_count=0,
            subtree_exact=False,
            span_connected=False,
            clause_root_ids=(),
            finite_verb_ids=(),
            aux_ids=(),
            modal_ids=(),
            neg_ids=(),
            mark_ids=(),
            case_ids=(),
            relations=(),
            scope_internal_ids=(),
            scope_external_ids=(),
            pronoun_ids=(),
            action_ids=(),
            predicate_internal_argument=False,
            predicate_external_argument=False,
            predicate_internal_subject=False,
            predicate_external_subject=False,
            predicate_internal_object=False,
            predicate_external_object=False,
            predicate_internal_oblique=False,
            predicate_external_oblique=False,
            predicate_internal_complement=False,
            predicate_external_complement=False,
            predicate_internal_aux=False,
            predicate_external_aux=False,
            predicate_internal_modifier=False,
            predicate_external_modifier=False,
            predicate_internal_neg=False,
            predicate_external_neg=False,
            predicate_internal_mark=False,
            predicate_external_mark=False,
            aux_internal=False,
            aux_external=False,
            neg_internal=False,
            neg_external=False,
            mark_internal=False,
            mark_external=False,
            case_internal=False,
            case_external=False,
            pronoun_inside_governor=False,
            pronoun_outside_governor=False,
            action_internal_argument=False,
            action_external_argument=False,
            action_complement=False,
            action_modifier=False,
            groups=(),
        )
    first_sentence_id, first_token = hit_tokens[0]
    last_sentence_id, last_token = hit_tokens[-1]
    spans_multiple_sentences = len(sentence_ids) > 1
    first_start = first_token.start_char
    last_end = last_token.end_char
    # Ambiguous only when a candidate boundary actually cuts an MWT token:
    # the start boundary falls inside a multiword first token, or the end
    # boundary falls inside a multiword last token.  A fully token-aligned
    # MWT endpoint is ordinary EXACT/TOKEN_ALIGNED evidence, and a partial
    # cut of a single-word token is ordinary PARTIAL_BOUNDARY.
    boundary_cuts_multiword = (
        (first_token.multiword and first_start != start)
        or (last_token.multiword and last_end != end)
    )
    if len(hit_tokens) == 1 and first_start == start and last_end == end:
        boundary_status = BOUNDARY_EXACT
    elif first_start == start and last_end == end:
        boundary_status = BOUNDARY_TOKEN_ALIGNED
    elif boundary_cuts_multiword:
        boundary_status = BOUNDARY_AMBIGUOUS
    else:
        boundary_status = BOUNDARY_PARTIAL
    ambiguity: list[str] = []
    if boundary_status == BOUNDARY_AMBIGUOUS:
        ambiguity.append("MULTIWORD_BOUNDARY_CUT")
    if spans_multiple_sentences:
        ambiguity.append("SPANS_MULTIPLE_SENTENCES")
    multi_token = len(token_ids) > 1

    words: dict[str, UdWord] = {}
    depths: dict[str, int] = {}
    children: dict[str, list[str]] = {}
    for sid in sentence_ids:
        sentence = sentences[sid]
        sentence_depths = _sentence_depths(sentence)
        for word in sentence.words:
            key = _word_key(sid, word.word_id)
            words[key] = word
            depths[key] = sentence_depths[word.word_id]
            children.setdefault(key, [])
        for word in sentence.words:
            if word.head == 0:
                continue
            parent_key = _word_key(sid, word.head)
            if parent_key in words:
                children.setdefault(parent_key, []).append(
                    _word_key(sid, word.word_id),
                )

    candidate_words = {key: words[key] for key in word_ids}
    head_keys = []
    for key, word in candidate_words.items():
        if word.head == 0:
            head_keys.append(key)
            continue
        sid = word.sentence_id
        parent_key = _word_key(sid, word.head)
        if parent_key not in candidate_words:
            head_keys.append(key)
    head_keys = sorted(
        head_keys,
        key=lambda key: (depths.get(key, 0), key),
    )
    if head_keys:
        candidate_head_id = head_keys[0]
        if len(head_keys) > 1:
            ambiguity.append("MULTIPLE_CANDIDATE_HEADS")
    else:
        candidate_head_id = None
    ordered_words = sorted(
        candidate_words.values(),
        key=lambda word: (
            word.start_char if word.start_char is not None else -1,
            word.sentence_id,
            word.word_id,
        ),
    )
    if ordered_words:
        candidate_root_id = min(
            candidate_words,
            key=lambda key: (depths.get(key, 0), key),
        )
        depth_values = [depths[key] for key in word_ids]
        min_depth = min(depths[key] for key in word_ids)
        if sum(1 for key in word_ids if depths[key] == min_depth) > 1:
            ambiguity.append("MULTIPLE_SUBTREE_ROOTS")
    else:
        candidate_root_id = None
        depth_values = []
    head_word = words.get(candidate_head_id) if candidate_head_id else None
    root_word = words.get(candidate_root_id) if candidate_root_id else None

    boundary_head_ids = tuple(
        key for key in candidate_words
        if (
            candidate_words[key].head != 0
            and _word_key(
                candidate_words[key].sentence_id,
                candidate_words[key].head,
            ) not in candidate_words
        )
    )
    external_governor_ids = tuple(sorted({
        _word_key(word.sentence_id, word.head)
        for word in candidate_words.values()
        if word.head != 0
        and _word_key(word.sentence_id, word.head) not in candidate_words
    }))
    crossing_arc_count = len(boundary_head_ids)
    for sentence_id in sentence_ids:
        sentence = sentences[sentence_id]
        for word in sentence.words:
            parent_key = _word_key(sentence_id, word.head)
            if word.head == 0:
                continue
            child_key = _word_key(sentence_id, word.word_id)
            if child_key in candidate_words and parent_key not in candidate_words:
                continue
            if parent_key in candidate_words and child_key not in candidate_words:
                crossing_arc_count += 1

    subtree: set[str] = set()
    if candidate_head_id:
        sid = head_word.sentence_id
        subtree = {
            _word_key(sid, word_id)
            for word_id in sentences[sid].subtree(head_word.word_id)
        }
    subtree_intersection = len(
        set(word_ids) & subtree,
    ) if subtree else 0
    subtree_size = len(subtree)
    subtree_exact = bool(subtree) and set(word_ids) == subtree

    connected = False
    if candidate_words:
        adjacent: dict[str, set[str]] = {
            key: set() for key in candidate_words
        }
        for key, word in candidate_words.items():
            if word.head == 0:
                continue
            parent_key = _word_key(word.sentence_id, word.head)
            if parent_key in candidate_words:
                adjacent[key].add(parent_key)
                adjacent[parent_key].add(key)
        start_key = next(iter(candidate_words))
        seen = {start_key}
        pending = [start_key]
        while pending:
            current = pending.pop()
            for neighbor in adjacent[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        connected = seen == set(candidate_words)

    finite_verb_ids: list[str] = []
    aux_ids: list[str] = []
    modal_ids: list[str] = []
    neg_ids: list[str] = []
    mark_ids: list[str] = []
    case_ids: list[str] = []
    relations: set[str] = set()
    scope_internal: list[str] = []
    scope_external: list[str] = []
    pronoun_ids_list: list[str] = []
    action_ids_list: list[str] = []
    pronoun_inside_governor = False
    pronoun_outside_governor = False
    for key, word in sorted(candidate_words.items()):
        family = _deprel_family(word.deprel)
        relations.add(family)
        if _is_finite(word):
            finite_verb_ids.append(key)
        if family == "aux" or (
            word.upos == "AUX" and family != "cop"
        ):
            aux_ids.append(key)
        if _is_modal(word):
            modal_ids.append(key)
        if _is_neg(word):
            neg_ids.append(key)
        if family == "mark":
            mark_ids.append(key)
        if family == "case":
            case_ids.append(key)
        if family in SCOPE_DEPS:
            parent_key = (
                _word_key(word.sentence_id, word.head)
                if word.head != 0 else None
            )
            if parent_key in candidate_words:
                scope_internal.append(key)
            else:
                scope_external.append(key)
        is_pronoun = (
            word.upos == "PRON"
            or word.lemma.lower() in PRONOUN_TOKENS
        )
        if is_pronoun:
            pronoun_ids_list.append(key)
            parent_key = (
                _word_key(word.sentence_id, word.head)
                if word.head != 0 else None
            )
            if parent_key in candidate_words:
                pronoun_inside_governor = True
            else:
                pronoun_outside_governor = True
        is_action = (
            word.upos in PREDICATE_UPOS
            and (
                word.lemma.lower() in ACTION_TOKENS
                or word.text.lower() in ACTION_TOKENS
            )
        )
        if is_action:
            action_ids_list.append(key)

    clause_root_ids = tuple(sorted({
        key for key in candidate_words
        if _is_structural_clause_root(candidate_words[key], words)
    }))
    root_ids = tuple(sorted({
        key for key in candidate_words
        if candidate_words[key].head == 0
    }))
    multiple_roots = len(root_ids) > 1
    syntactic_predicate_ids = tuple(sorted({
        key for key in candidate_words
        if _is_syntactic_predicate(candidate_words[key])
    }))

    def _argument_evidence(
        predicate_key: str,
    ) -> tuple[bool, bool, bool, bool]:
        predicate = candidate_words[predicate_key]
        internal = False
        external = False
        complement = False
        modifier = False
        for child_key in children.get(predicate_key, ()):
            child = words[child_key]
            family = _deprel_family(child.deprel)
            if family in SUBJECT_DEPS or family in OBJECT_DEPS:
                if child_key in candidate_words:
                    internal = True
                else:
                    external = True
            if family in COMPLEMENT_DEPS and child_key in candidate_words:
                complement = True
            if family in {"advmod", "obl"} and child_key in candidate_words:
                modifier = True
        return internal, external, complement, modifier

    def _predicate_role_evidence() -> dict[str, bool]:
        evidence = {
            f"{role}:{side}": False
            for role, _ in _PREDICATE_CHILD_ROLE_ORDER
            for side in ("internal", "external")
        }
        for predicate_key in syntactic_predicate_ids:
            for child_key in children.get(predicate_key, ()):
                child = words[child_key]
                role = _predicate_role(_deprel_family(child.deprel))
                if role is None:
                    continue
                side = (
                    "internal"
                    if child_key in candidate_words
                    else "external"
                )
                evidence[f"{role}:{side}"] = True
        return evidence

    predicate_role_evidence = _predicate_role_evidence()

    head_dependent_count = 0
    head_child_deprels: set[str] = set()
    if candidate_head_id:
        head_children = children.get(candidate_head_id, ())
        head_dependent_count = len(head_children)
        head_child_deprels = {
            _deprel_family(words[child_key].deprel)
            for child_key in head_children
        }
    head_relation_context: set[str] = set()
    if head_word is not None:
        head_family = _deprel_family(head_word.deprel)
        if head_word.head == 0:
            governor_upos = "ROOT"
        else:
            governor_key = _word_key(
                head_word.sentence_id, head_word.head,
            )
            governor = words.get(governor_key)
            governor_upos = governor.upos if governor else "OUT"
        head_relation_context.add(
            f"{head_family}:{governor_upos}:{head_word.upos}",
        )

    internal_relation_context: set[str] = set()
    crossing_incoming_relation_context: set[str] = set()
    crossing_outgoing_relation_context: set[str] = set()
    external_governor_lemmas: set[str] = set()
    external_governor_uposes: set[str] = set()
    external_governor_deprels: set[str] = set()
    for key, word in candidate_words.items():
        family = _deprel_family(word.deprel)
        if family in RELATION_DEPS and word.head != 0:
            parent_key = _word_key(word.sentence_id, word.head)
            parent = words.get(parent_key)
            if parent_key in candidate_words:
                if parent is not None:
                    internal_relation_context.add(
                        f"{family}:{parent.upos}:{word.upos}",
                    )
            else:
                governor_upos = parent.upos if parent is not None else "OUT"
                crossing_incoming_relation_context.add(
                    f"{family}:{governor_upos}:{word.upos}",
                )
                if parent is not None:
                    if parent.lemma:
                        external_governor_lemmas.add(parent.lemma.lower())
                    if parent.upos:
                        external_governor_uposes.add(parent.upos)
                    if parent.deprel:
                        external_governor_deprels.add(
                            _deprel_family(parent.deprel),
                        )
        for child_key in children.get(key, ()):
            if child_key in candidate_words:
                continue
            child = words[child_key]
            child_family = _deprel_family(child.deprel)
            if child_family not in RELATION_DEPS:
                continue
            crossing_outgoing_relation_context.add(
                f"{child_family}:{word.upos}:{child.upos}",
            )

    predicate_internal = False
    predicate_external = False
    predicate_internal_subject = predicate_role_evidence["subject:internal"]
    predicate_external_subject = predicate_role_evidence["subject:external"]
    predicate_internal_object = predicate_role_evidence["object:internal"]
    predicate_external_object = predicate_role_evidence["object:external"]
    predicate_internal_oblique = predicate_role_evidence["oblique:internal"]
    predicate_external_oblique = predicate_role_evidence["oblique:external"]
    predicate_internal_complement = predicate_role_evidence[
        "complement:internal"
    ]
    predicate_external_complement = predicate_role_evidence[
        "complement:external"
    ]
    predicate_internal_aux = predicate_role_evidence["aux:internal"]
    predicate_external_aux = predicate_role_evidence["aux:external"]
    predicate_internal_modifier = predicate_role_evidence[
        "modifier:internal"
    ]
    predicate_external_modifier = predicate_role_evidence[
        "modifier:external"
    ]
    predicate_internal_neg = predicate_role_evidence["neg:internal"]
    predicate_external_neg = predicate_role_evidence["neg:external"]
    predicate_internal_mark = predicate_role_evidence["mark:internal"]
    predicate_external_mark = predicate_role_evidence["mark:external"]
    predicate_argument_context = tuple(sorted({
        f"{role}:{side}"
        for role, _ in _PREDICATE_CHILD_ROLE_ORDER
        for side in ("internal", "external")
        if predicate_role_evidence[f"{role}:{side}"]
    }))
    aux_internal = False
    aux_external = False
    neg_internal = False
    neg_external = False
    mark_internal = False
    mark_external = False
    case_internal = False
    case_external = False
    action_internal_argument = False
    action_external_argument = False
    action_complement = False
    action_modifier = False
    for key in candidate_words:
        word = candidate_words[key]
        parent_key = (
            _word_key(word.sentence_id, word.head)
            if word.head != 0 else None
        )
        parent_inside = parent_key in candidate_words
        if key in finite_verb_ids or word.head == 0:
            internal, external, _, _ = _argument_evidence(key)
            predicate_internal = predicate_internal or internal
            predicate_external = predicate_external or external
        family = _deprel_family(word.deprel)
        if family == "aux":
            aux_internal = aux_internal or parent_inside
            aux_external = aux_external or not parent_inside
        if family == "neg":
            neg_internal = neg_internal or parent_inside
            neg_external = neg_external or not parent_inside
        if family == "mark":
            mark_internal = mark_internal or parent_inside
            mark_external = mark_external or not parent_inside
        if family == "case":
            case_internal = case_internal or parent_inside
            case_external = case_external or not parent_inside
        if key in action_ids_list:
            internal, external, complement, modifier = _argument_evidence(
                key,
            )
            action_internal_argument = action_internal_argument or internal
            action_external_argument = action_external_argument or external
            action_complement = action_complement or complement
            action_modifier = action_modifier or modifier

    groups: list[tuple[str, tuple[str, ...]]] = []
    groups.append((
        "head_lemma",
        (head_word.lemma.lower(),)
        if head_word and head_word.lemma else (),
    ))
    groups.append((
        "head_upos",
        (head_word.upos,) if head_word and head_word.upos else (),
    ))
    groups.append((
        "head_xpos",
        (head_word.xpos,) if head_word and head_word.xpos else (),
    ))
    groups.append((
        "head_deprel",
        (_deprel_family(head_word.deprel),)
        if head_word and head_word.deprel else (),
    ))
    groups.append((
        "root_upos",
        (root_word.upos,) if root_word and root_word.upos else (),
    ))
    groups.append((
        "first_lemma",
        (ordered_words[0].lemma.lower(),)
        if ordered_words and ordered_words[0].lemma else (),
    ))
    groups.append((
        "last_lemma",
        (ordered_words[-1].lemma.lower(),)
        if ordered_words and ordered_words[-1].lemma else (),
    ))
    groups.append((
        "first_upos",
        (ordered_words[0].upos,) if ordered_words and ordered_words[0].upos else (),
    ))
    groups.append((
        "last_upos",
        (ordered_words[-1].upos,) if ordered_words and ordered_words[-1].upos else (),
    ))
    relation_context: set[str] = set()
    for key, word in candidate_words.items():
        family = _deprel_family(word.deprel)
        if family not in RELATION_DEPS:
            continue
        governor_upos = "ROOT" if word.head == 0 else (
            words.get(_word_key(word.sentence_id, word.head)).upos
            if _word_key(word.sentence_id, word.head) in words else "OUT"
        )
        relation_context.add(
            f"{family}:{governor_upos}:{word.upos}",
        )
    groups.append(("rel_context", tuple(sorted(relation_context))))
    groups.append((
        "head_relation_context",
        tuple(sorted(head_relation_context)),
    ))
    groups.append((
        "head_child_deprels",
        tuple(sorted(head_child_deprels)),
    ))
    groups.append((
        "internal_rel_context",
        tuple(sorted(internal_relation_context)),
    ))
    groups.append((
        "crossing_incoming_rel_context",
        tuple(sorted(crossing_incoming_relation_context)),
    ))
    groups.append((
        "crossing_outgoing_rel_context",
        tuple(sorted(crossing_outgoing_relation_context)),
    ))
    groups.append((
        "external_governor_lemma",
        tuple(sorted(external_governor_lemmas)),
    ))
    groups.append((
        "external_governor_upos",
        tuple(sorted(external_governor_uposes)),
    ))
    groups.append((
        "external_governor_deprel",
        tuple(sorted(external_governor_deprels)),
    ))
    groups.append((
        "predicate_argument_context",
        predicate_argument_context,
    ))
    groups.append((
        "neg_lemma",
        tuple(sorted({
            words[key].lemma.lower() for key in neg_ids
        })),
    ))
    groups.append((
        "aux_lemma",
        tuple(sorted({
            words[key].lemma.lower() for key in aux_ids
        })),
    ))
    groups.append((
        "modal_lemma",
        tuple(sorted({
            words[key].lemma.lower() for key in modal_ids
        })),
    ))
    groups.append((
        "mark_lemma",
        tuple(sorted({
            words[key].lemma.lower() for key in mark_ids
        })),
    ))
    groups.append((
        "case_lemma",
        tuple(sorted({
            words[key].lemma.lower() for key in case_ids
        })),
    ))
    pronoun_governor_upos: set[str] = set()
    pronoun_governor_deprel: set[str] = set()
    for key in pronoun_ids_list:
        word = candidate_words[key]
        if word.head == 0:
            continue
        governor = words.get(_word_key(word.sentence_id, word.head))
        if governor is None:
            continue
        if governor.upos:
            pronoun_governor_upos.add(governor.upos)
        if governor.deprel:
            pronoun_governor_deprel.add(_deprel_family(governor.deprel))
    groups.append(("pronoun_governor_upos", tuple(sorted(pronoun_governor_upos))))
    groups.append((
        "pronoun_governor_deprel",
        tuple(sorted(pronoun_governor_deprel)),
    ))

    return CandidateSyntax(
        window_id=window_id,
        candidate_id=candidate_id,
        start=start,
        end=end,
        text=text,
        sentence_ids=sentence_ids,
        token_ids=token_ids,
        word_ids=word_ids,
        boundary_status=boundary_status,
        multi_token=multi_token,
        start_aligned=first_start == start,
        end_aligned=last_end == end,
        spans_multiple_sentences=spans_multiple_sentences,
        ambiguity=tuple(sorted(set(ambiguity))),
        candidate_head_id=candidate_head_id,
        candidate_root_id=candidate_root_id,
        head_word=head_word,
        root_word=root_word,
        candidate_depth_values=tuple(depth_values),
        root_ids=root_ids,
        multiple_roots=multiple_roots,
        syntactic_predicate_ids=syntactic_predicate_ids,
        head_dependent_count=head_dependent_count,
        head_child_deprels=tuple(sorted(head_child_deprels)),
        head_relation_context=tuple(sorted(head_relation_context)),
        internal_relation_context=tuple(sorted(internal_relation_context)),
        crossing_incoming_relation_context=tuple(sorted(
            crossing_incoming_relation_context,
        )),
        crossing_outgoing_relation_context=tuple(sorted(
            crossing_outgoing_relation_context,
        )),
        external_governor_lemmas=tuple(sorted(external_governor_lemmas)),
        external_governor_uposes=tuple(sorted(external_governor_uposes)),
        external_governor_deprels=tuple(sorted(external_governor_deprels)),
        predicate_argument_context=predicate_argument_context,
        boundary_head_ids=boundary_head_ids,
        external_governor_ids=external_governor_ids,
        crossing_arc_count=crossing_arc_count,
        subtree_size=subtree_size,
        subtree_intersection_count=subtree_intersection,
        subtree_exact=subtree_exact,
        span_connected=connected,
        clause_root_ids=clause_root_ids,
        finite_verb_ids=tuple(finite_verb_ids),
        aux_ids=tuple(aux_ids),
        modal_ids=tuple(modal_ids),
        neg_ids=tuple(neg_ids),
        mark_ids=tuple(mark_ids),
        case_ids=tuple(case_ids),
        relations=tuple(sorted(relations)),
        scope_internal_ids=tuple(scope_internal),
        scope_external_ids=tuple(scope_external),
        pronoun_ids=tuple(pronoun_ids_list),
        action_ids=tuple(action_ids_list),
        predicate_internal_argument=predicate_internal,
        predicate_external_argument=predicate_external,
        predicate_internal_subject=predicate_internal_subject,
        predicate_external_subject=predicate_external_subject,
        predicate_internal_object=predicate_internal_object,
        predicate_external_object=predicate_external_object,
        predicate_internal_oblique=predicate_internal_oblique,
        predicate_external_oblique=predicate_external_oblique,
        predicate_internal_complement=predicate_internal_complement,
        predicate_external_complement=predicate_external_complement,
        predicate_internal_aux=predicate_internal_aux,
        predicate_external_aux=predicate_external_aux,
        predicate_internal_modifier=predicate_internal_modifier,
        predicate_external_modifier=predicate_external_modifier,
        predicate_internal_neg=predicate_internal_neg,
        predicate_external_neg=predicate_external_neg,
        predicate_internal_mark=predicate_internal_mark,
        predicate_external_mark=predicate_external_mark,
        aux_internal=aux_internal,
        aux_external=aux_external,
        neg_internal=neg_internal,
        neg_external=neg_external,
        mark_internal=mark_internal,
        mark_external=mark_external,
        case_internal=case_internal,
        case_external=case_external,
        pronoun_inside_governor=pronoun_inside_governor,
        pronoun_outside_governor=pronoun_outside_governor,
        action_internal_argument=action_internal_argument,
        action_external_argument=action_external_argument,
        action_complement=action_complement,
        action_modifier=action_modifier,
        groups=tuple(groups),
    )


def compute_candidate_syntax(
    parse: UdParse,
    row: Any,
) -> CandidateSyntax:
    """Compute syntax evidence for one Phase 2H ``CandidateRow``."""
    return _compute_candidate(
        parse,
        start=row.start,
        end=row.end,
        text=row.text,
        window_id=row.window_id,
        candidate_id=row.candidate_id,
    )


DENSE_C_EXTRA_FEATURES = (
    "syntax_token_count",
    "syntax_word_count",
    "syntax_start_aligned",
    "syntax_end_aligned",
    "syntax_exact",
    "syntax_token_aligned",
    "syntax_partial_boundary",
    "syntax_unaligned",
    "syntax_ambiguous",
    "syntax_multi_token",
    "syntax_spans_multiple_sentences",
    "syntax_contains_root",
    "syntax_candidate_head_depth",
    "syntax_candidate_max_depth",
    "syntax_candidate_mean_depth",
    "syntax_boundary_heads",
    "syntax_external_governors",
    "syntax_external_governor_exists",
    "syntax_crossing_arcs",
    "syntax_subtree_word_fraction",
    "syntax_subtree_fraction",
    "syntax_subtree_intersection",
    "syntax_subtree_exact",
    "syntax_span_connected",
    "syntax_root_count",
    "syntax_multiple_roots",
    "syntax_multiple_clause_roots",
    "syntax_finite_verb_count",
    "syntax_predicate_count",
    "syntax_has_predicate",
    "syntax_clause_count",
    "syntax_aux_count",
    "syntax_modal_count",
    "syntax_neg_count",
    "syntax_mark_count",
    "syntax_case_count",
    "syntax_head_dependent_count",
    "syntax_has_nsubj",
    "syntax_has_csubj",
    "syntax_has_obj",
    "syntax_has_iobj",
    "syntax_has_obl",
    "syntax_has_advcl",
    "syntax_has_acl",
    "syntax_has_xcomp",
    "syntax_has_ccomp",
    "syntax_has_conj",
    "syntax_has_aux",
    "syntax_has_cop",
    "syntax_has_mark",
    "syntax_has_neg",
    "syntax_has_case",
    "syntax_has_compound",
    "syntax_has_amod",
    "syntax_has_advmod",
    "syntax_has_finite_verb",
    "syntax_has_modal",
    "syntax_predicate_argument_internal",
    "syntax_predicate_argument_external",
    "syntax_predicate_internal_subject",
    "syntax_predicate_external_subject",
    "syntax_predicate_internal_object",
    "syntax_predicate_external_object",
    "syntax_predicate_internal_oblique",
    "syntax_predicate_external_oblique",
    "syntax_predicate_internal_complement",
    "syntax_predicate_external_complement",
    "syntax_predicate_internal_aux",
    "syntax_predicate_external_aux",
    "syntax_predicate_internal_modifier",
    "syntax_predicate_external_modifier",
    "syntax_predicate_internal_neg",
    "syntax_predicate_external_neg",
    "syntax_predicate_internal_mark",
    "syntax_predicate_external_mark",
    "syntax_aux_internal",
    "syntax_aux_external",
    "syntax_neg_internal",
    "syntax_neg_external",
    "syntax_mark_internal",
    "syntax_mark_external",
    "syntax_case_internal",
    "syntax_case_external",
    "syntax_scope_governor_inside",
    "syntax_scope_governor_outside",
    "syntax_pronoun_inside_governor",
    "syntax_pronoun_outside_governor",
    "syntax_action_internal_argument",
    "syntax_action_external_argument",
    "syntax_action_complement",
    "syntax_action_modifier",
)


SYNTAX_GROUPS = (
    "head_lemma",
    "head_upos",
    "head_xpos",
    "head_deprel",
    "head_relation_context",
    "head_child_deprels",
    "root_upos",
    "first_lemma",
    "last_lemma",
    "first_upos",
    "last_upos",
    "rel_context",
    "internal_rel_context",
    "crossing_incoming_rel_context",
    "crossing_outgoing_rel_context",
    "external_governor_lemma",
    "external_governor_upos",
    "external_governor_deprel",
    "neg_lemma",
    "aux_lemma",
    "modal_lemma",
    "mark_lemma",
    "case_lemma",
    "pronoun_governor_upos",
    "pronoun_governor_deprel",
    "predicate_argument_context",
)


def dense_c_matrix(
    records: Sequence[CandidateSyntax],
) -> np.ndarray:
    """Dense numeric Feature Set C matrix aligned to
    ``DENSE_C_EXTRA_FEATURES``."""
    n = len(records)
    matrix = np.zeros((n, len(DENSE_C_EXTRA_FEATURES)), dtype=np.float64)
    columns = {
        name: index for index, name in enumerate(DENSE_C_EXTRA_FEATURES)
    }
    for index, record in enumerate(records):
        head_depth = None
        if record.candidate_head_id and record.candidate_depth_values:
            by_id = dict(zip(record.word_ids, record.candidate_depth_values))
            head_depth = by_id.get(record.candidate_head_id)
        matrix[index, columns["syntax_token_count"]] = len(record.token_ids)
        matrix[index, columns["syntax_word_count"]] = len(record.word_ids)
        matrix[index, columns["syntax_start_aligned"]] = float(
            record.start_aligned
        )
        matrix[index, columns["syntax_end_aligned"]] = float(
            record.end_aligned
        )
        matrix[index, columns["syntax_exact"]] = float(
            record.boundary_status == BOUNDARY_EXACT
        )
        matrix[index, columns["syntax_token_aligned"]] = float(
            record.boundary_status == BOUNDARY_TOKEN_ALIGNED
        )
        matrix[index, columns["syntax_partial_boundary"]] = float(
            record.boundary_status == BOUNDARY_PARTIAL
        )
        matrix[index, columns["syntax_unaligned"]] = float(
            record.boundary_status == BOUNDARY_UNALIGNED
        )
        matrix[index, columns["syntax_ambiguous"]] = float(
            record.boundary_status == BOUNDARY_AMBIGUOUS
        )
        matrix[index, columns["syntax_multi_token"]] = float(
            record.multi_token
        )
        matrix[index, columns["syntax_spans_multiple_sentences"]] = float(
            record.spans_multiple_sentences
        )
        matrix[index, columns["syntax_contains_root"]] = float(
            any(
                word.head == 0
                for word in (record.head_word, record.root_word)
                if word is not None
            )
        )
        matrix[index, columns["syntax_candidate_head_depth"]] = (
            float(head_depth) if head_depth is not None else -1.0
        )
        matrix[index, columns["syntax_candidate_max_depth"]] = (
            float(max(record.candidate_depth_values))
            if record.candidate_depth_values else -1.0
        )
        matrix[index, columns["syntax_candidate_mean_depth"]] = (
            float(np.mean(record.candidate_depth_values))
            if record.candidate_depth_values else 0.0
        )
        matrix[index, columns["syntax_boundary_heads"]] = len(
            record.boundary_head_ids
        )
        matrix[index, columns["syntax_external_governors"]] = len(
            record.external_governor_ids
        )
        matrix[index, columns["syntax_external_governor_exists"]] = float(
            bool(record.external_governor_ids)
        )
        matrix[index, columns["syntax_crossing_arcs"]] = (
            record.crossing_arc_count
        )
        matrix[index, columns["syntax_subtree_word_fraction"]] = (
            record.subtree_intersection_count / len(record.word_ids)
            if record.word_ids else 0.0
        )
        matrix[index, columns["syntax_subtree_fraction"]] = (
            record.subtree_intersection_count / record.subtree_size
            if record.subtree_size else 0.0
        )
        matrix[index, columns["syntax_subtree_intersection"]] = (
            float(record.subtree_intersection_count)
        )
        matrix[index, columns["syntax_subtree_exact"]] = float(
            record.subtree_exact
        )
        matrix[index, columns["syntax_span_connected"]] = float(
            record.span_connected
        )
        matrix[index, columns["syntax_root_count"]] = len(record.root_ids)
        matrix[index, columns["syntax_multiple_roots"]] = float(
            record.multiple_roots
        )
        matrix[index, columns["syntax_multiple_clause_roots"]] = float(
            len(record.clause_root_ids) > 1
        )
        matrix[index, columns["syntax_finite_verb_count"]] = len(
            record.finite_verb_ids
        )
        matrix[index, columns["syntax_predicate_count"]] = len(
            record.syntactic_predicate_ids
        )
        matrix[index, columns["syntax_has_predicate"]] = float(
            bool(record.syntactic_predicate_ids)
        )
        matrix[index, columns["syntax_clause_count"]] = len(
            record.clause_root_ids
        )
        matrix[index, columns["syntax_aux_count"]] = len(record.aux_ids)
        matrix[index, columns["syntax_modal_count"]] = len(record.modal_ids)
        matrix[index, columns["syntax_neg_count"]] = len(record.neg_ids)
        matrix[index, columns["syntax_mark_count"]] = len(record.mark_ids)
        matrix[index, columns["syntax_case_count"]] = len(record.case_ids)
        matrix[index, columns["syntax_head_dependent_count"]] = (
            record.head_dependent_count
        )
        relations = set(record.relations)
        for family in (
            "nsubj", "csubj", "obj", "iobj", "obl", "advcl", "acl",
            "xcomp", "ccomp", "conj", "aux", "cop", "mark", "neg",
            "case", "compound", "amod", "advmod",
        ):
            matrix[index, columns[f"syntax_has_{family}"]] = float(
                family in relations
            )
        matrix[index, columns["syntax_has_finite_verb"]] = float(
            bool(record.finite_verb_ids)
        )
        matrix[index, columns["syntax_has_modal"]] = float(
            bool(record.modal_ids)
        )
        matrix[index, columns["syntax_predicate_argument_internal"]] = float(
            record.predicate_internal_argument
        )
        matrix[index, columns["syntax_predicate_argument_external"]] = float(
            record.predicate_external_argument
        )
        matrix[index, columns["syntax_predicate_internal_subject"]] = float(
            record.predicate_internal_subject
        )
        matrix[index, columns["syntax_predicate_external_subject"]] = float(
            record.predicate_external_subject
        )
        matrix[index, columns["syntax_predicate_internal_object"]] = float(
            record.predicate_internal_object
        )
        matrix[index, columns["syntax_predicate_external_object"]] = float(
            record.predicate_external_object
        )
        matrix[index, columns["syntax_predicate_internal_oblique"]] = float(
            record.predicate_internal_oblique
        )
        matrix[index, columns["syntax_predicate_external_oblique"]] = float(
            record.predicate_external_oblique
        )
        matrix[index, columns["syntax_predicate_internal_complement"]] = float(
            record.predicate_internal_complement
        )
        matrix[index, columns["syntax_predicate_external_complement"]] = float(
            record.predicate_external_complement
        )
        matrix[index, columns["syntax_predicate_internal_aux"]] = float(
            record.predicate_internal_aux
        )
        matrix[index, columns["syntax_predicate_external_aux"]] = float(
            record.predicate_external_aux
        )
        matrix[index, columns["syntax_predicate_internal_modifier"]] = float(
            record.predicate_internal_modifier
        )
        matrix[index, columns["syntax_predicate_external_modifier"]] = float(
            record.predicate_external_modifier
        )
        matrix[index, columns["syntax_predicate_internal_neg"]] = float(
            record.predicate_internal_neg
        )
        matrix[index, columns["syntax_predicate_external_neg"]] = float(
            record.predicate_external_neg
        )
        matrix[index, columns["syntax_predicate_internal_mark"]] = float(
            record.predicate_internal_mark
        )
        matrix[index, columns["syntax_predicate_external_mark"]] = float(
            record.predicate_external_mark
        )
        matrix[index, columns["syntax_aux_internal"]] = float(
            record.aux_internal
        )
        matrix[index, columns["syntax_aux_external"]] = float(
            record.aux_external
        )
        matrix[index, columns["syntax_neg_internal"]] = float(
            record.neg_internal
        )
        matrix[index, columns["syntax_neg_external"]] = float(
            record.neg_external
        )
        matrix[index, columns["syntax_mark_internal"]] = float(
            record.mark_internal
        )
        matrix[index, columns["syntax_mark_external"]] = float(
            record.mark_external
        )
        matrix[index, columns["syntax_case_internal"]] = float(
            record.case_internal
        )
        matrix[index, columns["syntax_case_external"]] = float(
            record.case_external
        )
        matrix[index, columns["syntax_scope_governor_inside"]] = float(
            bool(record.scope_internal_ids)
        )
        matrix[index, columns["syntax_scope_governor_outside"]] = float(
            bool(record.scope_external_ids)
        )
        matrix[index, columns["syntax_pronoun_inside_governor"]] = float(
            record.pronoun_inside_governor
        )
        matrix[index, columns["syntax_pronoun_outside_governor"]] = float(
            record.pronoun_outside_governor
        )
        matrix[index, columns["syntax_action_internal_argument"]] = float(
            record.action_internal_argument
        )
        matrix[index, columns["syntax_action_external_argument"]] = float(
            record.action_external_argument
        )
        matrix[index, columns["syntax_action_complement"]] = float(
            record.action_complement
        )
        matrix[index, columns["syntax_action_modifier"]] = float(
            record.action_modifier
        )
    return matrix


def syntax_groups_from_records(
    records: Sequence[CandidateSyntax],
) -> list[dict[str, tuple[str, ...]]]:
    return [dict(record.groups) for record in records]


class SyntaxEncoder:
    """Categorical syntax vocabulary fitted on training windows only.

    Values are one-hot encoded per :data:`SYNTAX_GROUPS`; values observed
    only in held-out windows produce no column (explicitly audited).  The
    per-group vocabulary is capped at ``max_values_per_group`` by descending
    training frequency (ties alphabetical), which bounds the matrix while
    remaining deterministic.
    """

    def __init__(self, *, max_values_per_group: int = MAX_VALUES_PER_GROUP) -> None:
        if (
            isinstance(max_values_per_group, bool)
            or not isinstance(max_values_per_group, int)
            or max_values_per_group <= 0
        ):
            raise Phase2ISyntaxError(
                "max_values_per_group must be a positive integer",
            )
        self.max_values_per_group = max_values_per_group
        self.vocabulary: dict[str, tuple[str, ...]] = {}
        self._index: dict[str, dict[str, int]] = {}
        self.fitted = False

    def fit(
        self,
        records: Sequence[Mapping[str, Sequence[str]]],
    ) -> "SyntaxEncoder":
        counts: dict[str, dict[str, int]] = {
            group: {} for group in SYNTAX_GROUPS
        }
        for record in records:
            for group in SYNTAX_GROUPS:
                values = record.get(group) or ()
                for value in values:
                    if not isinstance(value, str) or not value:
                        continue
                    counts[group][value] = counts[group].get(value, 0) + 1
        vocabulary: dict[str, tuple[str, ...]] = {}
        for group in SYNTAX_GROUPS:
            ranked = sorted(
                counts[group].items(),
                key=lambda item: (-item[1], item[0]),
            )
            vocabulary[group] = tuple(
                value for value, _ in ranked[: self.max_values_per_group]
            )
        self.vocabulary = vocabulary
        self._index = {
            group: {
                value: index for index, value in enumerate(vocabulary[group])
            }
            for group in SYNTAX_GROUPS
        }
        self.fitted = True
        return self

    def transform(
        self,
        records: Sequence[Mapping[str, Sequence[str]]],
    ) -> sp.csr_matrix:
        if not self.fitted:
            raise Phase2ISyntaxError("SyntaxEncoder must be fitted first")
        column_offsets: dict[str, int] = {}
        offset = 0
        for group in SYNTAX_GROUPS:
            column_offsets[group] = offset
            offset += len(self.vocabulary[group])
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row_index, record in enumerate(records):
            for group in SYNTAX_GROUPS:
                values = record.get(group) or ()
                group_index = self._index[group]
                for value in values:
                    if not isinstance(value, str) or not value:
                        continue
                    column = group_index.get(value)
                    if column is None:
                        continue
                    rows.append(row_index)
                    cols.append(column_offsets[group] + column)
                    data.append(1.0)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(records), offset),
            dtype=np.float64,
        )

    def feature_names(self) -> list[str]:
        if not self.fitted:
            raise Phase2ISyntaxError("SyntaxEncoder must be fitted first")
        names: list[str] = []
        for group in SYNTAX_GROUPS:
            for value in self.vocabulary[group]:
                names.append(f"syntax:{group}={value}")
        return names

    def vocabulary_sha256(self) -> str:
        if not self.fitted:
            raise Phase2ISyntaxError("SyntaxEncoder must be fitted first")
        return canonical_sha256({
            group: list(self.vocabulary[group])
            for group in SYNTAX_GROUPS
        })

    def oov_audit(
        self,
        records: Sequence[Mapping[str, Sequence[str]]],
    ) -> dict[str, Any]:
        """Held-out categorical values absent from the training vocabulary."""
        if not self.fitted:
            raise Phase2ISyntaxError("SyntaxEncoder must be fitted first")
        training = {
            group: set(self.vocabulary[group]) for group in SYNTAX_GROUPS
        }
        held_out: dict[str, set[str]] = {
            group: set() for group in SYNTAX_GROUPS
        }
        for record in records:
            for group in SYNTAX_GROUPS:
                for value in record.get(group) or ():
                    if (
                        isinstance(value, str)
                        and value
                        and value not in training[group]
                    ):
                        held_out[group].add(value)
        oov = {
            group: sorted(values) for group, values in held_out.items()
        }
        return {
            "oov_definition": (
                "distinct held-out categorical syntax values per group "
                "absent from the training-window-only vocabulary; OOV "
                "values emit no feature column"
            ),
            "per_group": oov,
            "oov_value_count": sum(len(values) for values in oov.values()),
            "oov_sha256": canonical_sha256(oov),
        }


def feature_schema_c() -> dict[str, Any]:
    return {
        "version": SYNTAX_FEATURE_SCHEMA_VERSION,
        "feature_set_C": {
            "label": (
                "frozen Phase 2H feature set B plus deterministic "
                "UD/syntactic features"
            ),
            "dense_extras": list(DENSE_C_EXTRA_FEATURES),
            "categorical_groups": list(SYNTAX_GROUPS),
            "categorical_encoder": {
                "kind": "train-window-only one-hot SyntaxEncoder",
                "max_values_per_group": MAX_VALUES_PER_GROUP,
            },
            "boundary_statuses": list(BOUNDARY_STATUSES),
            "relation_context_families": sorted(RELATION_DEPS),
            "notes": (
                "all syntax features are learned features only; no hard "
                "endpoint rules; scaler/vocabularies fitted on training "
                "windows only; alignment ambiguity is explicit and never "
                "silently resolved; relation contexts are exposed distinctly "
                "as head relation, wholly internal, crossing incoming "
                "(candidate token -> external governor), and crossing "
                "outgoing (external token -> candidate head/token) with "
                "external governor lemma/UPOS/deprel; all subtree fractions "
                "are bounded to [0,1]; predicate evidence uses syntactic "
                "VERB/AUX/finite/clause predicates and is never limited to "
                "the bounded action-token lexicon"
            ),
        },
        "fit_scope": "training windows only",
        "prohibited_features": [
            "case ids", "source ids", "window ids", "candidate ids",
            "mention ids", "question text", "gold roles/types",
            "label-derived values", "DeepSeek labels/predictions",
            "generative endpoint predictions",
        ],
    }


def parse_definition() -> dict[str, Any]:
    return {
        "schema_version": PARSE_SCHEMA_VERSION,
        "parser": "stanza",
        "parser_version": STANZA_VERSION,
        "parser_version_constraint": f"=={STANZA_VERSION}",
        "required_runtime": f"stanza=={STANZA_VERSION}",
        "language": STANZA_LANGUAGE,
        "package": STANZA_PACKAGE,
        "processors": list(STANZA_PROCESSORS),
        "locked_assets_manifest_sha256": LOCKED_ASSETS_MANIFEST_SHA256,
        "runtime": "CPU only; torch single thread; no network at evaluation",
        "download_policy": "DownloadMethod.NONE at evaluation; setup script "
                           "is the only network path",
        "offset_projection": (
            "stanza token char offsets are Bronze-relative; MWT expanded "
            "words are projected onto their parent token offsets and "
            "flagged offset_kind=MWT_PROJECTED"
        ),
    }


def verify_assets_provenance(
    assets_dir: str | Path,
    *,
    provenance_path: str | Path | None = None,
    require_locked: bool = True,
) -> dict[str, Any]:
    """Verify local parser assets against ASSET_PROVENANCE.json hashes.

    Rejects malformed/duplicate/extra files, validates the exact schema,
    ``stanza==STANZA_VERSION``, package, and processor contract, and ignores
    transient cache files (``.hf_cache`` etc.) that are not model assets.
    Any symlink at or below ``assets_dir`` -- including a symlinked parent
    directory such as ``assets/en -> external`` -- is rejected before any
    file is read through it.  With ``require_locked`` (the default, used by
    every evaluation path) the manifest SHA must also equal
    :data:`LOCKED_ASSETS_MANIFEST_SHA256`.
    The supplied path is lexically absolutized and every existing ancestor
    component (from the filesystem anchor through the asset root) is
    lstat-checked for symlinks before any walk/hash, so a link in the path
    itself is never resolved away or read through.
    ``require_locked=False`` is for unit-fixture provenance structure checks
    only and is never used by real acceptance.
    """
    assets_dir = _lexically_absolute_path(assets_dir)
    problems: list[str] = []
    problems.extend(_symlink_ancestor_problems(assets_dir))
    if problems:
        return {
            "verified": False,
            "problems": problems,
            "reason": (
                "parser asset path has a symlinked ancestor or root; real "
                "acceptance never resolves or reads through a symlink"
            ),
        }
    provenance_path = _lexically_absolute_path(
        provenance_path or assets_dir / "ASSET_PROVENANCE.json",
    )
    problems.extend(_symlink_ancestor_problems(provenance_path))
    if problems:
        return {
            "verified": False,
            "problems": problems,
            "reason": (
                "parser provenance path has a symlinked ancestor or root; "
                "real acceptance never resolves or reads through a symlink"
            ),
        }
    problems.extend(_asset_symlink_problems(assets_dir))
    if provenance_path.is_symlink():
        problems.append("provenance file is a symlink")
    if problems:
        return {
            "verified": False,
            "problems": problems,
            "reason": (
                "parser assets tree contains symlinks; real acceptance "
                "never reads through a symlink"
            ),
        }
    if not provenance_path.is_file():
        return {
            "verified": False,
            "reason": f"provenance file missing: {provenance_path}",
        }
    try:
        provenance = _load_json_strict(provenance_path)
    except (OSError, Phase2ISyntaxError) as error:
        return {
            "verified": False,
            "reason": f"provenance JSON is invalid: {error}",
        }
    if not isinstance(provenance, Mapping):
        return {
            "verified": False,
            "reason": "provenance must be a JSON object",
        }
    if set(provenance) != {
        "schema_version", "stanza_version", "package", "processors",
        "created_at", "manifest_sha256", "files",
    }:
        problems.append("provenance top-level key set is not canonical")
    if not isinstance(provenance.get("created_at"), str) or not provenance.get(
        "created_at",
    ):
        problems.append("provenance created_at is missing")
    if provenance.get("schema_version") != "phase2i-parser-assets-v1":
        problems.append(
            "provenance schema_version is not phase2i-parser-assets-v1",
        )
    stanza_version = provenance.get("stanza_version")
    if (
        not isinstance(stanza_version, str)
        or stanza_version != STANZA_VERSION
    ):
        problems.append(
            f"provenance stanza_version {stanza_version!r} != "
            f"{STANZA_VERSION}",
        )
    if provenance.get("package") != STANZA_PACKAGE:
        problems.append(
            f"provenance package {provenance.get('package')!r} != "
            f"{STANZA_PACKAGE}",
        )
    if list(provenance.get("processors") or []) != list(STANZA_PROCESSORS):
        problems.append(
            f"provenance processors {provenance.get('processors')!r} != "
            f"{list(STANZA_PROCESSORS)}",
        )
    entries = provenance.get("files")
    if not isinstance(entries, list):
        problems.append("provenance files must be a list")
        entries = []
    seen_paths: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256"}
        ):
            problems.append("provenance file entry is invalid")
            continue
        relative = entry.get("path")
        expected = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            problems.append(
                f"provenance file path {relative!r} is invalid",
            )
            continue
        if not is_sha256_hex(expected):
            problems.append(
                f"provenance file {relative} sha256 is malformed",
            )
            continue
        if relative in seen_paths:
            problems.append(f"provenance file {relative} duplicated")
            continue
        seen_paths.add(relative)
        component_problem = _symlink_component_problem(
            assets_dir, relative,
        )
        if component_problem is not None:
            problems.append(component_problem)
            continue
        path = assets_dir / relative
        if not path.is_file() or path.is_symlink():
            problems.append(f"asset missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"asset {relative} sha256 mismatch")
    canonical_entry_order = sorted(
        (
            entry for entry in entries
            if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
        ),
        key=lambda entry: entry["path"],
    )
    if entries != canonical_entry_order:
        problems.append("provenance files are not in canonical path order")
    on_disk = {
        str(path.relative_to(assets_dir))
        for path in _asset_file_paths(assets_dir)
    }
    for relative in sorted(on_disk - seen_paths):
        problems.append(f"unlisted asset on disk: {relative}")
    manifest_sha256 = canonical_sha256([
        {"path": entry["path"], "sha256": entry["sha256"]}
        for entry in entries
        if isinstance(entry, Mapping)
    ])
    if manifest_sha256 != provenance.get("manifest_sha256"):
        problems.append("asset manifest sha256 does not self-verify")
    if require_locked and manifest_sha256 != LOCKED_ASSETS_MANIFEST_SHA256:
        problems.append(
            "asset manifest sha256 does not match the locked Phase 2I "
            "provenance",
        )
    if require_locked and provenance.get("manifest_sha256") != (
        LOCKED_ASSETS_MANIFEST_SHA256
    ):
        problems.append(
            "provenance manifest_sha256 is not the locked Phase 2I "
            "provenance value",
        )
    return {
        "verified": not problems,
        "problems": problems,
        "schema_version": provenance.get("schema_version"),
        "stanza_version": provenance.get("stanza_version"),
        "package": provenance.get("package"),
        "processors": provenance.get("processors"),
        "manifest_sha256": provenance.get("manifest_sha256"),
        "files": [
            {"path": entry["path"], "sha256": entry["sha256"]}
            for entry in entries
            if isinstance(entry, Mapping)
        ],
    }


def write_asset_provenance(
    assets_dir: str | Path,
    *,
    stanza_version: str,
    package: str = STANZA_PACKAGE,
    processors: Sequence[str] = STANZA_PROCESSORS,
    created_at: str,
) -> Path:
    """Write ASSET_PROVENANCE.json for a populated assets directory."""
    assets_dir = Path(assets_dir)
    entries = [
        {"path": path, "sha256": digest}
        for path, digest in _model_asset_hashes(assets_dir)
    ]
    manifest_sha256 = canonical_sha256(entries)
    provenance = {
        "schema_version": "phase2i-parser-assets-v1",
        "stanza_version": stanza_version,
        "package": package,
        "processors": list(processors),
        "created_at": created_at,
        "manifest_sha256": manifest_sha256,
        "files": entries,
    }
    path = assets_dir / "ASSET_PROVENANCE.json"
    path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_parse_artifact(path: str | Path) -> UdParse:
    """Load and self-verify a serialized parse artifact."""
    data = _load_json_strict(path)
    parse = UdParse.from_dict(data)
    expected = json.dumps(
        parse.to_dict(), indent=2, ensure_ascii=False,
    ) + "\n"
    if Path(path).read_text(encoding="utf-8") != expected:
        raise Phase2ISyntaxError(
            "raw parse JSON does not match canonical field ordering",
        )
    return parse


def parse_artifact_paths(artifacts_dir: str | Path) -> dict[str, Path]:
    artifacts_dir = Path(artifacts_dir)
    return {
        path.stem: path
        for path in sorted(artifacts_dir.glob("*.json"))
    }


__all__ = [
    "BOUNDARY_AMBIGUOUS",
    "BOUNDARY_EXACT",
    "BOUNDARY_PARTIAL",
    "BOUNDARY_STATUSES",
    "BOUNDARY_TOKEN_ALIGNED",
    "BOUNDARY_UNALIGNED",
    "CandidateSyntax",
    "DENSE_C_EXTRA_FEATURES",
    "LOCKED_ASSETS_MANIFEST_SHA256",
    "MAX_VALUES_PER_GROUP",
    "PARSE_SCHEMA_VERSION",
    "PIPELINE_VERSION",
    "RELATION_DEPS",
    "STANZA_LANGUAGE",
    "STANZA_PACKAGE",
    "STANZA_PROCESSORS",
    "STANZA_VERSION",
    "SYNTAX_FEATURE_SCHEMA_VERSION",
    "SYNTAX_GROUPS",
    "SyntaxEncoder",
    "UdParse",
    "UdSentence",
    "UdToken",
    "UdWord",
    "Phase2IParseError",
    "Phase2ISyntaxError",
    "assets_manifest_sha256",
    "compute_candidate_syntax",
    "dense_c_matrix",
    "feature_schema_c",
    "is_sha256_hex",
    "parse_definition",
    "parse_window_text",
    "syntax_groups_from_records",
    "verify_parser_asset_path",
    "verify_assets_provenance",
    "write_asset_provenance",
]
