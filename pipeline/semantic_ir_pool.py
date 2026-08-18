"""Deterministic representative bronze-window pool for Phase 2F review.

The pool is not a gold benchmark. It provides source-exact windows spanning a
declared set of general linguistic phenomena so reviewers can choose DEV and
FROZEN_EVAL cases without searching the corpus opportunistically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping


POOL_SCHEMA_VERSION = "phase2f-semantic-window-pool-v1"
POOL_PHENOMENA = (
    "simple_fact", "direct_advice", "advice_explanation", "explicit_cause",
    "implicit_cause", "conditional", "nested_condition", "temporal",
    "negation", "modality", "comparison", "pronoun", "omitted_actor",
    "multiple_champions", "multiple_abilities", "wave_reasoning",
    "resource_exchange", "cause_chain", "multi_sentence", "contrast",
    "contradiction", "uncertainty", "quantity", "location_or_space",
    "punctuation_poor",
)

_TOKEN = re.compile(r"\S+")
_WORD = re.compile(r"[A-Za-z0-9']+")
_PATTERNS = {
    "direct_advice": re.compile(r"\b(?:you should|you need to|you must|try to|make sure|do not|don't|never|always)\b", re.I),
    "explicit_cause": re.compile(r"\b(?:because|therefore|so that|which means|as a result|causes?|enables?|prevents?|requires?)\b", re.I),
    "conditional": re.compile(r"\b(?:if|unless|when|whenever|only if|provided that)\b", re.I),
    "temporal": re.compile(r"\b(?:before|after|once|until|while|during|then|later|first|comes? back|returns?|expires?|ends?)\b", re.I),
    "negation": re.compile(r"\b(?:no|not|never|cannot|can't|cant|couldn't|couldnt|doesn't|doesnt|don't|dont|won't|wont|without|unable)\b", re.I),
    "modality": re.compile(r"\b(?:can|could|may|might|must|should|would|usually|sometimes|probably|likely)\b", re.I),
    "comparison": re.compile(r"\b(?:more|less|better|worse|higher|lower|faster|slower|stronger|weaker|than|same|equal|most|least)\b", re.I),
    "pronoun": re.compile(r"\b(?:he|him|his|she|her|hers|it|its|they|them|their|this|that|these|those)\b", re.I),
    "omitted_actor": re.compile(r"(?:^|[.!?]\s+)(?:push|walk|move|hold|wait|save|use|take|play|keep|stop|respect|bait|track|avoid|look)\b", re.I),
    "wave_reasoning": re.compile(r"\b(?:wave|waves|minion|creeps|push|freeze|crash|bounce|slow push)\b", re.I),
    "resource_exchange": re.compile(r"\b(?:mana|energy|health|hp|cooldown|charge|stack|gold|resource|ward|sweeper|flash|teleport|ignite|exhaust)\b", re.I),
    "contrast": re.compile(r"\b(?:but|however|although|whereas|rather than|instead)\b", re.I),
    "contradiction": re.compile(r"\b(?:not .{0,45} but|but .{0,45} not|doesn't mean|does not mean|on the other hand)\b", re.I),
    "uncertainty": re.compile(r"\b(?:maybe|might|could|probably|likely|possibly|sometimes|usually|I think|I guess|perhaps)\b", re.I),
    "quantity": re.compile(r"(?:\b\d+(?:\.\d+)?%?\b|\b(?:one|two|three|four|five|more|less|half|double|twice)\b)", re.I),
    "location_or_space": re.compile(r"\b(?:top|mid|bot|lane|river|jungle|bush|tower|base|baron|dragon|side|behind|front|range|distance|angle)\b", re.I),
}


def build_semantic_window_pool(
    db_path: Path,
    *,
    frozen_fixture: Path,
    development_fixture: Path,
    target_count: int = 300,
    target_words: int = 48,
    stride_words: int = 40,
    minimum_per_phenomenon: int = 8,
) -> dict[str, Any]:
    for value, label in (
        (target_count, "target_count"), (target_words, "target_words"),
        (stride_words, "stride_words"), (minimum_per_phenomenon, "minimum_per_phenomenon"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if stride_words > target_words:
        raise ValueError("pool stride cannot exceed target window size")
    frozen_sources = _fixture_source_ids(frozen_fixture)
    development_sources = _fixture_source_ids(development_fixture)
    excluded_sources = frozen_sources | development_sources
    champion_names = _champion_names(db_path)
    champions = _champion_pattern(champion_names)
    candidates = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT video_id, video_title, role, champion, transcription
            FROM videos
            WHERE game = 'lol' AND transcription IS NOT NULL
              AND trim(transcription) <> ''
            ORDER BY video_id
            """
        )
        for row in rows:
            video_id = str(row["video_id"])
            if video_id in excluded_sources:
                continue
            text = str(row["transcription"])
            tokens = tuple(_TOKEN.finditer(text))
            if len(tokens) < 8:
                continue
            per_video: dict[str, dict[str, Any]] = {}
            offsets = range(0, len(tokens), stride_words)
            for ordinal, offset in enumerate(offsets, 1):
                selected = tokens[offset:offset + target_words]
                if len(selected) < min(16, target_words):
                    continue
                start, end = selected[0].start(), selected[-1].end()
                window_text = text[start:end]
                phenomena = detect_pool_phenomena(window_text, champions)
                record = _window_record(
                    video_id, row, text, start, end, offset, ordinal, phenomena,
                )
                # Retain one rich deterministic candidate for each phenomenon
                # plus a general candidate, bounding memory per source.
                keys = phenomena or ("simple_fact",)
                for key in keys:
                    previous = per_video.get(key)
                    if previous is None or _candidate_rank(record) < _candidate_rank(previous):
                        per_video[key] = record
            unique = {item["window_id"]: item for item in per_video.values()}
            candidates.extend(unique.values())
    if len({item["upstream_source_id"] for item in candidates}) < target_count:
        raise ValueError("corpus cannot provide the requested number of source-isolated windows")

    selected: list[dict[str, Any]] = []
    selected_sources: set[str] = set()
    counts = {key: 0 for key in POOL_PHENOMENA}
    by_phenomenon = {
        key: sorted(
            (item for item in candidates if key in item["phenomena"]),
            key=_candidate_rank,
        )
        for key in POOL_PHENOMENA
    }
    for phenomenon in POOL_PHENOMENA:
        for item in by_phenomenon[phenomenon]:
            if counts[phenomenon] >= minimum_per_phenomenon:
                break
            if item["upstream_source_id"] in selected_sources:
                continue
            _select(item, selected, selected_sources, counts)
    for item in sorted(candidates, key=_candidate_rank):
        if len(selected) >= target_count:
            break
        if item["upstream_source_id"] not in selected_sources:
            _select(item, selected, selected_sources, counts)
    if len(selected) != target_count:
        raise ValueError("pool selection did not reach its preregistered size")
    missing = {
        key: count for key, count in counts.items() if count < minimum_per_phenomenon
    }
    if missing:
        raise ValueError("pool phenomenon coverage is insufficient: " + repr(missing))
    selected.sort(key=lambda item: item["window_id"])
    for index, item in enumerate(selected, 1):
        item["pool_index"] = index
    inner = {
        "schema_version": POOL_SCHEMA_VERSION,
        "purpose": "Representative source-exact bronze windows for Phase 2F semantic IR review; not gold labels.",
        "selection_policy": {
            "target_count": target_count, "target_words": target_words,
            "stride_words": stride_words,
            "minimum_per_phenomenon": minimum_per_phenomenon,
            "one_window_per_upstream_source": True,
            "champion_names": list(champion_names),
            "excluded_phase2b_sources": sorted(frozen_sources),
            "excluded_phase2d_sources": sorted(development_sources),
        },
        "input_hashes": {
            "database_sha256": _file_sha256(db_path),
            "frozen_fixture_sha256": _file_sha256(frozen_fixture),
            "development_fixture_sha256": _file_sha256(development_fixture),
        },
        "phenomenon_counts": dict(sorted(counts.items())),
        "windows": selected,
    }
    return {"content_sha256": _canonical_sha256(inner), **inner}


def validate_semantic_window_pool(value: Mapping[str, Any]) -> None:
    expected = {
        "content_sha256", "schema_version", "purpose", "selection_policy",
        "input_hashes", "phenomenon_counts", "windows",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("semantic window pool envelope is invalid")
    if value["schema_version"] != POOL_SCHEMA_VERSION:
        raise ValueError("semantic window pool version is unsupported")
    inner = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["content_sha256"] != _canonical_sha256(inner):
        raise ValueError("semantic window pool content hash is invalid")
    policy, windows = value["selection_policy"], value["windows"]
    expected_policy = {
        "target_count", "target_words", "stride_words", "minimum_per_phenomenon",
        "one_window_per_upstream_source", "champion_names",
        "excluded_phase2b_sources", "excluded_phase2d_sources",
    }
    if not isinstance(policy, Mapping) or set(policy) != expected_policy \
            or not isinstance(windows, list) or len(windows) != policy.get("target_count"):
        raise ValueError("semantic window pool size/policy is invalid")
    for key in ("target_count", "target_words", "stride_words", "minimum_per_phenomenon"):
        if isinstance(policy[key], bool) or not isinstance(policy[key], int) \
                or policy[key] < 1:
            raise ValueError("semantic window pool numeric policy is invalid")
    if policy["stride_words"] > policy["target_words"]:
        raise ValueError("semantic window pool stride exceeds its window size")
    if policy["one_window_per_upstream_source"] is not True \
            or not isinstance(policy["champion_names"], list) \
            or policy["champion_names"] != sorted(set(policy["champion_names"])) \
            or any(not isinstance(item, str) or not item.strip() for item in policy["champion_names"]):
        raise ValueError("semantic window pool source-selection policy is invalid")
    for key in ("excluded_phase2b_sources", "excluded_phase2d_sources"):
        if not isinstance(policy[key], list) or policy[key] != sorted(set(policy[key])) \
                or any(not isinstance(item, str) or not item for item in policy[key]):
            raise ValueError("semantic window pool exclusion policy is invalid")
    if not isinstance(value["input_hashes"], Mapping) \
            or set(value["input_hashes"]) != {
                "database_sha256", "frozen_fixture_sha256", "development_fixture_sha256",
            } or any(
        not _is_sha256(item) for item in value["input_hashes"].values()
    ):
        raise ValueError("semantic window pool input hashes are invalid")
    ids, sources, counts = set(), set(), {key: 0 for key in POOL_PHENOMENA}
    excluded = set(policy.get("excluded_phase2b_sources", ())) | set(
        policy.get("excluded_phase2d_sources", ()),
    )
    for index, item in enumerate(windows, 1):
        _validate_window_record(item)
        if item["phenomena"] != list(detect_pool_phenomena(
            item["source_text"], tuple(policy["champion_names"]),
        )):
            raise ValueError("semantic window pool phenomenon tags are not reproducible")
        if item["token_offset"] != (
            item["source_window_ordinal"] - 1
        ) * policy["stride_words"]:
            raise ValueError("semantic window pool token offset/ordinal is invalid")
        if item["pool_index"] != index or item["window_id"] in ids:
            raise ValueError("semantic window pool IDs/order are invalid")
        if (
            policy["one_window_per_upstream_source"]
            and item["upstream_source_id"] in sources
        ) or item["upstream_source_id"] in excluded:
            raise ValueError("semantic window pool source isolation is invalid")
        ids.add(item["window_id"])
        sources.add(item["upstream_source_id"])
        for phenomenon in item["phenomena"]:
            counts[phenomenon] += 1
    if value["phenomenon_counts"] != dict(sorted(counts.items())):
        raise ValueError("semantic window pool phenomenon counts are invalid")
    if any(count < policy.get("minimum_per_phenomenon", 0) for count in counts.values()):
        raise ValueError("semantic window pool does not satisfy phenomenon coverage")


def verify_semantic_window_pool_inputs(
    value: Mapping[str, Any],
    *,
    db_path: Path,
    frozen_fixture: Path,
    development_fixture: Path,
    reproduce_selection: bool = False,
) -> None:
    """Bind a self-consistent pool back to the immutable corpus and fixtures."""
    validate_semantic_window_pool(value)
    if not isinstance(reproduce_selection, bool):
        raise ValueError("pool reproduction flag must be boolean")
    expected_hashes = {
        "database_sha256": _file_sha256(db_path),
        "frozen_fixture_sha256": _file_sha256(frozen_fixture),
        "development_fixture_sha256": _file_sha256(development_fixture),
    }
    if value["input_hashes"] != expected_hashes:
        raise ValueError("semantic window pool input files do not match retained hashes")
    policy = value["selection_policy"]
    if policy["excluded_phase2b_sources"] != sorted(_fixture_source_ids(frozen_fixture)) \
            or policy["excluded_phase2d_sources"] != sorted(_fixture_source_ids(development_fixture)):
        raise ValueError("semantic window pool exclusions do not match its fixtures")
    if policy["champion_names"] != list(_champion_names(db_path)):
        raise ValueError("semantic window pool champion catalog does not match its corpus")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for item in value["windows"]:
            row = connection.execute(
                """SELECT video_id, video_title, role, champion, transcription
                   FROM videos WHERE video_id = ? AND game = 'lol'""",
                (item["upstream_source_id"],),
            ).fetchone()
            if row is None:
                raise ValueError("semantic window pool source is absent from its corpus")
            text = str(row["transcription"] or "")
            if (
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                != item["upstream_content_sha256"]
                or text[item["upstream_start"]:item["upstream_end"]] != item["source_text"]
                or item["metadata"] != {
                    "video_title": str(row["video_title"] or ""),
                    "role": str(row["role"] or ""),
                    "champion": str(row["champion"] or ""),
                }
            ):
                raise ValueError("semantic window pool record contradicts immutable corpus source")
    if reproduce_selection:
        rebuilt = build_semantic_window_pool(
            db_path, frozen_fixture=frozen_fixture,
            development_fixture=development_fixture,
            target_count=policy["target_count"], target_words=policy["target_words"],
            stride_words=policy["stride_words"],
            minimum_per_phenomenon=policy["minimum_per_phenomenon"],
        )
        if rebuilt != value:
            raise ValueError("semantic window pool is not the deterministic selected corpus pool")


def load_semantic_window_pool(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("semantic window pool JSON is unavailable or malformed") from exc
    validate_semantic_window_pool(value)
    return value


def detect_pool_phenomena(
    text: str, champions: Iterable[str] | re.Pattern[str] = (),
) -> tuple[str, ...]:
    lowered = text.casefold()
    values = {key for key, pattern in _PATTERNS.items() if pattern.search(text)}
    advice = "direct_advice" in values or "omitted_actor" in values
    causal = "explicit_cause" in values
    if advice and causal:
        values.add("advice_explanation")
    if advice and not causal and re.search(r"\b(?:is|are|has|have|can't|cannot|weak|strong|safe|dangerous)\b", text, re.I):
        values.add("implicit_cause")
    if len(_PATTERNS["conditional"].findall(text)) >= 2:
        values.add("nested_condition")
    if len(_PATTERNS["explicit_cause"].findall(text)) >= 2:
        values.add("cause_chain")
    if len(re.findall(r"[.!?]+(?:\s+|$)", text)) >= 2:
        values.add("multi_sentence")
    if not re.search(r"[.!?]", text):
        values.add("punctuation_poor")
    if isinstance(champions, re.Pattern):
        names = {match.group(0).casefold() for match in champions.finditer(text)}
    else:
        names = {
            name.casefold() for name in champions
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.I)
        }
    if len(names) >= 2:
        values.add("multiple_champions")
    ability_tokens = set(re.findall(r"\b(?:q|w|e|r|ult|ultimate|flash|teleport|ignite|exhaust)\b", lowered))
    if len(ability_tokens) >= 2:
        values.add("multiple_abilities")
    if not values & {
        "direct_advice", "explicit_cause", "conditional", "negation", "contrast",
        "uncertainty", "modality",
    } and re.search(r"\b(?:is|are|has|have|does|gives|deals|takes)\b", text, re.I):
        values.add("simple_fact")
    return tuple(sorted(values))


def _window_record(
    video_id: str, row: sqlite3.Row, full_text: str, start: int, end: int,
    token_offset: int, ordinal: int, phenomena: tuple[str, ...],
) -> dict[str, Any]:
    window_text = full_text[start:end]
    identity = hashlib.sha256(
        f"{video_id}:{start}:{end}:{window_text}".encode("utf-8"),
    ).hexdigest()[:20]
    return {
        "window_id": f"pool:{video_id}:w{ordinal:05d}-{identity}",
        "source_id": f"transcript:{video_id}",
        "source_kind": "transcript",
        "upstream_source_id": video_id,
        "upstream_start": start,
        "upstream_end": end,
        "token_offset": token_offset,
        "source_window_ordinal": ordinal,
        "source_text": window_text,
        "source_text_sha256": hashlib.sha256(window_text.encode("utf-8")).hexdigest(),
        "upstream_content_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "phenomena": list(phenomena),
        "metadata": {
            "video_title": str(row["video_title"] or ""),
            "role": str(row["role"] or ""),
            "champion": str(row["champion"] or ""),
        },
    }


def _validate_window_record(item: object) -> None:
    expected = {
        "pool_index", "window_id", "source_id", "source_kind", "upstream_source_id",
        "upstream_start", "upstream_end", "source_text", "source_text_sha256",
        "upstream_content_sha256", "token_offset", "source_window_ordinal",
        "phenomena", "metadata",
    }
    if not isinstance(item, Mapping) or set(item) != expected:
        raise ValueError("semantic window pool record is invalid")
    if any(not isinstance(item[key], str) or not item[key] for key in (
        "window_id", "source_id", "source_kind", "upstream_source_id", "source_text",
    )):
        raise ValueError("semantic window pool source identity/text is invalid")
    start, end = item["upstream_start"], item["upstream_end"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) \
            or start < 0 or end - start != len(item["source_text"]):
        raise ValueError("semantic window pool offsets are invalid")
    for key in ("pool_index", "token_offset", "source_window_ordinal"):
        if isinstance(item[key], bool) or not isinstance(item[key], int) \
                or item[key] < (1 if key != "token_offset" else 0):
            raise ValueError("semantic window pool ordinals are invalid")
    identity = hashlib.sha256(
        f"{item['upstream_source_id']}:{start}:{end}:{item['source_text']}".encode("utf-8"),
    ).hexdigest()[:20]
    if item["window_id"] != (
        f"pool:{item['upstream_source_id']}:w{item['source_window_ordinal']:05d}-{identity}"
    ) or item["source_kind"] != "transcript" \
            or item["source_id"] != "transcript:" + item["upstream_source_id"]:
        raise ValueError("semantic window pool stable source identity is invalid")
    if item["source_text_sha256"] != hashlib.sha256(item["source_text"].encode()).hexdigest() \
            or not _is_sha256(item["upstream_content_sha256"]):
        raise ValueError("semantic window pool source hashes are invalid")
    if not isinstance(item["phenomena"], list) or tuple(item["phenomena"]) != tuple(sorted(set(item["phenomena"]))) \
            or any(value not in POOL_PHENOMENA for value in item["phenomena"]):
        raise ValueError("semantic window pool phenomena are invalid")
    if not isinstance(item["metadata"], Mapping) or set(item["metadata"]) != {
        "video_title", "role", "champion",
    } or any(not isinstance(value, str) for value in item["metadata"].values()):
        raise ValueError("semantic window pool metadata is invalid")


def _select(
    item: dict[str, Any], selected: list[dict[str, Any]], selected_sources: set[str],
    counts: dict[str, int],
) -> None:
    selected.append(dict(item))
    selected_sources.add(item["upstream_source_id"])
    for phenomenon in item["phenomena"]:
        counts[phenomenon] += 1


def _candidate_rank(item: Mapping[str, Any]) -> tuple[int, str]:
    return -len(item["phenomena"]), item["window_id"]


def _champion_names(db_path: Path) -> tuple[str, ...]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT champion FROM videos
            WHERE game = 'lol' AND champion IS NOT NULL AND trim(champion) <> ''
            ORDER BY champion
            """,
        )
        return tuple(str(row[0]) for row in rows if str(row[0]).strip())


def _champion_pattern(names: Iterable[str]) -> re.Pattern[str]:
    alternatives = sorted(
        {re.escape(name.strip()) for name in names if name.strip()},
        key=lambda value: (-len(value), value.casefold()),
    )
    if not alternatives:
        return re.compile(r"(?!)")
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.I)


def _fixture_source_ids(path: Path) -> set[str]:
    body = json.loads(path.read_text(encoding="utf-8"))
    values = set()
    for case in body.get("cases", []):
        source = case.get("source_video_id")
        if isinstance(source, str) and source:
            values.add(_upstream_source_id(source))
        for evidence in case.get("evidence", []):
            source = evidence.get("source_id")
            if isinstance(source, str) and source and not source.startswith("__"):
                values.add(_upstream_source_id(source))
    return values


def _upstream_source_id(value: str) -> str:
    return value.split(":", 1)[1] if value.startswith("transcript:") else value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("semantic window pool JSON contains duplicate keys")
        value[key] = item
    return value


__all__ = [
    "POOL_SCHEMA_VERSION", "POOL_PHENOMENA", "build_semantic_window_pool",
    "validate_semantic_window_pool", "load_semantic_window_pool",
    "verify_semantic_window_pool_inputs", "detect_pool_phenomena",
]
