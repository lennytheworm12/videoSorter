"""Phase 2J pre-annotation source selection (model-blind, source-exact).

Phase 2J selects exactly 30 windows from the retained, independently recorded
Phase 2F pool (``semantic_ir_window_pool_v1.json``) with exactly one window per
distinct upstream video source group.  The selection is deterministic and
model-blind: eligibility and the greedy diversity preference use only pool
source identity, exact offsets/text hashes, ordinary metadata, the frozen
lexical phenomenon tags, and the diagnostic ASR punctuation band derived from
the frozen ``punctuation_poor`` tag.  No model predictions, scores, ranks,
uncertainty, syntax importance, labels, or error taxonomy influence selection,
and none are recorded as annotation-facing content.

The five legacy Phase 2H/2I windows (``semantic_ir_legacy_manifest_v1.json``
bound to ``semantic_ir_legacy_failure_v1.json``) are regression-only and are
excluded even if the retained pool already excludes them.

The frozen Phase 2F mention catalog is regenerated only to bind identity,
count, and hash for each selected window.  It is never scored and never
exposed to reviewers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from pipeline.semantic_ir_pool import (
    POOL_PHENOMENA,
    load_semantic_window_pool,
    validate_semantic_window_pool,
)
from pipeline.semantic_mentions import (
    MENTION_CATALOG_VERSION,
    generate_mention_candidates,
)
from pipeline.semantic_source import BronzeSource, window_from_exact_span


SELECTION_SCHEMA_VERSION = "phase2j-window-selection-manifest-v1"
SELECTION_SEED = "20260817"
TARGET_WINDOW_COUNT = 30
PARTITION_EXPANDED_DEV = "EXPANDED_DEV"
PARTITION_FROZEN_REPLICATION = "FROZEN_REPLICATION"
PARTITION_SIZES = {
    PARTITION_EXPANDED_DEV: 24,
    PARTITION_FROZEN_REPLICATION: 6,
}
PARTITIONS = tuple(sorted(PARTITION_SIZES))
ASR_PUNCTUATION_POOR = "PUNCTUATION_POOR"
ASR_PUNCTUATED = "PUNCTUATED"
ASR_BANDS = (ASR_PUNCTUATION_POOR, ASR_PUNCTUATED)
LEGACY_MANIFEST_SCHEMA_VERSION = "phase2f-legacy-five-source-manifest-v1"
LEGACY_BENCHMARK_SCHEMA_VERSION = "phase2f-semantic-benchmark-v1"
CHECKPOINT = "PRE_ANNOTATION_CHECKPOINT"

# Preregistered greedy diversity preference.  These are diversity preferences,
# not claims of corpus balance; actual distributions are recorded separately.
PHENOMENON_POINTS = 8
PHENOMENON_BELOW_COUNT = 2
ROLE_POINTS = 4
ROLE_BELOW_COUNT = 2
ASR_BAND_POINTS = 2
ASR_BAND_BELOW_COUNT = 3
CHAMPION_POINTS = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_MANIFEST_KEYS = frozenset({
    "content_sha256", "schema_version", "purpose", "release_gate",
    "selection_policy", "input_hashes", "legacy_source_exclusions", "selected",
    "partition_counts", "diversity_summary", "candidate_generator_version",
    "checkpoint",
})
_EXPECTED_POLICY_KEYS = frozenset({
    "seed", "target_window_count", "target_distinct_video_source_groups",
    "one_window_per_upstream_source", "source_group_id_rule",
    "eligibility", "diversity_score", "partition",
    "diversity_preference_statement",
})
_EXPECTED_DIVERSITY_SCORE_KEYS = frozenset({
    "phenomenon_points", "phenomenon_below_count", "role_points",
    "role_below_count", "asr_band_points", "asr_band_below_count",
    "unrepresented_champion_points", "asr_band_definition", "tie_break",
})
_EXPECTED_PARTITION_POLICY_KEYS = frozenset({
    "order", "EXPANDED_DEV", "FROZEN_REPLICATION",
})
_EXPECTED_INPUT_HASH_KEYS = frozenset({
    "pool_file_sha256", "pool_content_sha256",
    "legacy_manifest_file_sha256", "legacy_manifest_content_sha256",
    "legacy_benchmark_file_sha256", "legacy_benchmark_content_sha256",
})
_EXPECTED_SELECTED_KEYS = frozenset({
    "source_group_id", "window_id", "upstream_source_id", "upstream_start",
    "upstream_end", "source_text", "source_text_sha256",
    "upstream_content_sha256", "source_text_char_length", "metadata",
    "phenomena", "asr_punctuation_band", "partition",
    "candidate_generator_version", "candidate_count",
    "candidate_catalog_sha256", "canonical_record_sha256",
})
_EXPECTED_DIVERSITY_SUMMARY_KEYS = frozenset({
    "phenomenon_counts", "role_counts", "asr_punctuation_band_counts",
    "champion_counts", "distinct_champions", "candidate_count",
})


def canonical_sha256(value: object) -> str:
    """Canonical content hash consistent with repository conventions."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_strict(path: Path, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        body = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
        )
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON is unavailable or malformed") from exc
    if not isinstance(body, dict):
        raise ValueError(f"{label} must be a JSON object")
    return body


def load_legacy_manifest(path: Path) -> Mapping[str, Any]:
    """Load and self-verify the locked legacy five-window source manifest."""
    body = _load_json_strict(path, label="legacy source manifest")
    if body.get("schema_version") != LEGACY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("legacy source manifest version is unsupported")
    expected = {"content_sha256", "schema_version", "purpose", "input_hashes", "windows"}
    if set(body) != expected:
        raise ValueError("legacy source manifest envelope is invalid")
    inner = {key: item for key, item in body.items() if key != "content_sha256"}
    if body["content_sha256"] != canonical_sha256(inner):
        raise ValueError("legacy source manifest content hash is invalid")
    windows = body["windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("legacy source manifest must contain windows")
    for window in windows:
        if not isinstance(window, Mapping):
            raise ValueError("legacy source manifest windows must be objects")
        for key in ("case_id", "source_id", "upstream_source_id"):
            if not isinstance(window.get(key), str) or not window[key]:
                raise ValueError("legacy source manifest window identity is invalid")
    return body


def load_legacy_benchmark(path: Path, *, manifest: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Load and self-verify the locked five-case Phase 2H/2I benchmark."""
    body = _load_json_strict(path, label="legacy benchmark")
    if body.get("schema_version") != LEGACY_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("legacy benchmark version is unsupported")
    expected = {
        "content_sha256", "pool_manifest_sha256", "purpose",
        "schema_version", "split", "cases",
    }
    if set(body) != expected:
        raise ValueError("legacy benchmark envelope is invalid")
    inner = {key: item for key, item in body.items() if key != "content_sha256"}
    if body["content_sha256"] != canonical_sha256(inner):
        raise ValueError("legacy benchmark content hash is invalid")
    if body.get("split") != "LEGACY_FAILURE":
        raise ValueError("legacy benchmark split is not the locked legacy failure split")
    cases = body.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("legacy benchmark must contain exactly five cases")
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("id"), str) \
                or not case["id"]:
            raise ValueError("legacy benchmark case identity is invalid")
    if manifest is not None and body.get("pool_manifest_sha256") != manifest.get("content_sha256"):
        raise ValueError("legacy benchmark is not bound to its source manifest")
    return body


def legacy_source_exclusions(
    manifest: Mapping[str, Any], benchmark: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return every upstream/source video ID in the legacy five-window set."""
    if benchmark is not None:
        benchmark_ids = {
            str(case["id"])
            for case in benchmark.get("cases", [])
            if isinstance(case, Mapping) and isinstance(case.get("id"), str)
        }
    else:
        benchmark_ids = set()
    values: set[str] = set()
    for window in manifest.get("windows", []):
        if not isinstance(window, Mapping):
            continue
        if window.get("case_id") in benchmark_ids:
            upstream = window.get("upstream_source_id")
            if isinstance(upstream, str) and upstream:
                values.add(upstream)
        source = window.get("source_id")
        if isinstance(source, str) and source:
            values.add(_strip_source_prefix(source))
        # Defensive: a benchmark case may itself carry source identity.
    for case in benchmark.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        for key in ("source_video_id", "source_id"):
            source = case.get(key)
            if isinstance(source, str) and source:
                values.add(_strip_source_prefix(source))
    return tuple(sorted(values))


def _strip_source_prefix(value: str) -> str:
    return value.split(":", 1)[1] if value.startswith("transcript:") else value


def asr_punctuation_band(phenomena: Iterable[str]) -> str:
    """Diagnostic ASR band from the frozen lexical phenomenon tags only."""
    if "punctuation_poor" in phenomena:
        return ASR_PUNCTUATION_POOR
    return ASR_PUNCTUATED


def marginal_diversity_score(
    record: Mapping[str, Any],
    *,
    phenomenon_counts: Mapping[str, int],
    role_counts: Mapping[str, int],
    asr_band_counts: Mapping[str, int],
    selected_champions: set[str],
) -> int:
    """Preregistered greedy marginal preference; not a model score."""
    score = 0
    for phenomenon in record["phenomena"]:
        if phenomenon_counts.get(phenomenon, 0) < PHENOMENON_BELOW_COUNT:
            score += PHENOMENON_POINTS
    role = record["metadata"]["role"]
    if role and role_counts.get(role, 0) < ROLE_BELOW_COUNT:
        score += ROLE_POINTS
    band = asr_punctuation_band(record["phenomena"])
    if asr_band_counts.get(band, 0) < ASR_BAND_BELOW_COUNT:
        score += ASR_BAND_POINTS
    champion = record["metadata"]["champion"]
    if champion and champion not in selected_champions:
        score += CHAMPION_POINTS
    return int(score)


def _selection_tie_key(record: Mapping[str, Any]) -> tuple[str, str]:
    raw = f"{SELECTION_SEED}:{record['window_id']}".encode("utf-8")
    return (hashlib.sha256(raw).hexdigest(), str(record["window_id"]))


def _partition_tie_key(source_group_id: str, window_id: str) -> tuple[str, str, str]:
    raw = f"{SELECTION_SEED}:{source_group_id}".encode("utf-8")
    return (hashlib.sha256(raw).hexdigest(), source_group_id, window_id)


def select_windows(
    pool: Mapping[str, Any], *, excluded_sources: Iterable[str],
) -> list[dict[str, Any]]:
    """Select exactly 30 windows from 30 distinct video source groups."""
    validate_semantic_window_pool(pool)
    excluded = set(excluded_sources)
    candidates = [
        dict(item)
        for item in pool["windows"]
        if item["upstream_source_id"] not in excluded
    ]
    if len(candidates) < TARGET_WINDOW_COUNT:
        raise ValueError(
            f"pool cannot provide {TARGET_WINDOW_COUNT} eligible source-isolated windows",
        )
    candidates.sort(key=lambda item: (item["upstream_source_id"], item["window_id"]))
    phenomenon_counts = {phenomenon: 0 for phenomenon in POOL_PHENOMENA}
    role_counts: dict[str, int] = {}
    asr_band_counts = {ASR_PUNCTUATION_POOR: 0, ASR_PUNCTUATED: 0}
    selected_champions: set[str] = set()
    selected: list[dict[str, Any]] = []
    while len(selected) < TARGET_WINDOW_COUNT:
        scored = [
            (
                marginal_diversity_score(
                    item,
                    phenomenon_counts=phenomenon_counts,
                    role_counts=role_counts,
                    asr_band_counts=asr_band_counts,
                    selected_champions=selected_champions,
                ),
                _selection_tie_key(item),
                item,
            )
            for item in candidates
        ]
        scored.sort(key=lambda row: (-row[0], row[1][0], row[1][1]))
        best = scored[0][2]
        candidates.remove(best)
        selected.append(best)
        for phenomenon in best["phenomena"]:
            phenomenon_counts[phenomenon] += 1
        role = best["metadata"]["role"]
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1
        band = asr_punctuation_band(best["phenomena"])
        asr_band_counts[band] += 1
        champion = best["metadata"]["champion"]
        if champion:
            selected_champions.add(champion)
    for item in selected:
        item["source_group_id"] = f"video:{item['upstream_source_id']}"
    ordered = sorted(
        selected,
        key=lambda item: _partition_tie_key(
            item["source_group_id"], item["window_id"],
        ),
    )
    for index, item in enumerate(ordered):
        item["partition"] = (
            PARTITION_EXPANDED_DEV
            if index < PARTITION_SIZES[PARTITION_EXPANDED_DEV]
            else PARTITION_FROZEN_REPLICATION
        )
    return ordered


def candidate_catalog_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    """Regenerate the frozen mention catalog solely to bind identity/count/hash.

    The catalog is generated but never scored and never exposed in the
    annotation-facing packet.
    """
    source_id = f"transcript:{record['upstream_source_id']}"
    text = record["source_text"]
    source = BronzeSource(source_id, text)
    window = window_from_exact_span(source, 0, len(text))
    candidates = generate_mention_candidates(window)
    aliases = tuple(f"C{index:04d}" for index in range(1, len(candidates) + 1))
    upstream_start = record["upstream_start"]
    catalog_records = [
        {
            "alias": alias,
            "candidate_id": item.candidate_id,
            "window_id": item.window_id,
            "start": item.start,
            "end": item.end,
            "absolute_start": upstream_start + item.start,
            "absolute_end": upstream_start + item.end,
            "text": item.source_text,
            "segment_ids": list(item.segment_ids),
        }
        for alias, item in zip(aliases, candidates)
    ]
    return {
        "candidate_generator_version": MENTION_CATALOG_VERSION,
        "candidate_count": len(catalog_records),
        "candidate_catalog_sha256": canonical_sha256(catalog_records),
    }


def _diversity_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    phenomenon_counts = {phenomenon: 0 for phenomenon in POOL_PHENOMENA}
    role_counts: dict[str, int] = {}
    asr_band_counts = {ASR_PUNCTUATION_POOR: 0, ASR_PUNCTUATED: 0}
    champion_counts: dict[str, int] = {}
    candidate_count = 0
    for record in records:
        for phenomenon in record["phenomena"]:
            phenomenon_counts[phenomenon] += 1
        role = record["metadata"]["role"]
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1
        band = record["asr_punctuation_band"]
        asr_band_counts[band] += 1
        champion = record["metadata"]["champion"]
        if champion:
            champion_counts[champion] = champion_counts.get(champion, 0) + 1
        candidate_count += record["candidate_count"]
    return {
        "phenomenon_counts": dict(sorted(phenomenon_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "asr_punctuation_band_counts": dict(sorted(asr_band_counts.items())),
        "champion_counts": dict(sorted(champion_counts.items())),
        "distinct_champions": len(champion_counts),
        "candidate_count": int(candidate_count),
    }


def build_selection_manifest(
    *,
    pool: Mapping[str, Any],
    pool_path: Path,
    legacy_manifest: Mapping[str, Any],
    legacy_manifest_path: Path,
    legacy_benchmark: Mapping[str, Any],
    legacy_benchmark_path: Path,
) -> Mapping[str, Any]:
    """Build the deterministic Phase 2J window-selection manifest."""
    validate_semantic_window_pool(pool)
    exclusions = legacy_source_exclusions(legacy_manifest, legacy_benchmark)
    selected = select_windows(pool, excluded_sources=exclusions)
    records = []
    for item in selected:
        binding = candidate_catalog_binding(item)
        record = {
            "source_group_id": item["source_group_id"],
            "window_id": item["window_id"],
            "upstream_source_id": item["upstream_source_id"],
            "upstream_start": item["upstream_start"],
            "upstream_end": item["upstream_end"],
            "source_text": item["source_text"],
            "source_text_sha256": item["source_text_sha256"],
            "upstream_content_sha256": item["upstream_content_sha256"],
            "source_text_char_length": len(item["source_text"]),
            "metadata": dict(item["metadata"]),
            "phenomena": list(item["phenomena"]),
            "asr_punctuation_band": asr_punctuation_band(item["phenomena"]),
            "partition": item["partition"],
            "candidate_generator_version": binding["candidate_generator_version"],
            "candidate_count": binding["candidate_count"],
            "candidate_catalog_sha256": binding["candidate_catalog_sha256"],
        }
        record["canonical_record_sha256"] = canonical_sha256(record)
        records.append(record)
    inner = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "purpose": (
            "Model-blind, source-exact Phase 2J pre-annotation selection: "
            "exactly 30 windows from 30 distinct independently recorded video "
            "source groups; not gold and not labels."
        ),
        "release_gate": "LOCKED",
        "checkpoint": CHECKPOINT,
        "selection_policy": {
            "seed": SELECTION_SEED,
            "target_window_count": TARGET_WINDOW_COUNT,
            "target_distinct_video_source_groups": TARGET_WINDOW_COUNT,
            "one_window_per_upstream_source": True,
            "source_group_id_rule": "source_group_id = 'video:' + upstream_source_id",
            "eligibility": [
                "retained phase2f semantic_ir_window_pool_v1.json only",
                "source group derived exactly from pool window upstream_source_id",
                "exclude every upstream/source video ID in the legacy five-case Phase 2H/2I benchmark",
                "eligibility uses only pool/source identity, exact offsets/text hashes, ordinary metadata, frozen lexical phenomenon tags, and diagnostic ASR metadata",
                "no model predictions, scores, ranks, uncertainty, syntax importance, labels, or error taxonomy",
            ],
            "diversity_score": {
                "phenomenon_points": PHENOMENON_POINTS,
                "phenomenon_below_count": PHENOMENON_BELOW_COUNT,
                "role_points": ROLE_POINTS,
                "role_below_count": ROLE_BELOW_COUNT,
                "asr_band_points": ASR_BAND_POINTS,
                "asr_band_below_count": ASR_BAND_BELOW_COUNT,
                "unrepresented_champion_points": CHAMPION_POINTS,
                "asr_band_definition": (
                    "PUNCTUATION_POOR when the frozen phenomenon tag "
                    "'punctuation_poor' is present, otherwise PUNCTUATED"
                ),
                "tie_break": (
                    "sha256('20260817:' + window_id), then window_id; "
                    "smallest digest first"
                ),
            },
            "partition": {
                "order": (
                    "stable order by sha256('20260817:' + source_group_id), "
                    "then source_group_id, then window_id"
                ),
                PARTITION_EXPANDED_DEV: PARTITION_SIZES[PARTITION_EXPANDED_DEV],
                PARTITION_FROZEN_REPLICATION: PARTITION_SIZES[PARTITION_FROZEN_REPLICATION],
            },
            "diversity_preference_statement": (
                "These greedy diversity preferences are not claims of corpus "
                "balance; the manifest records the actual distributions."
            ),
        },
        "input_hashes": {
            "pool_file_sha256": file_sha256(pool_path),
            "pool_content_sha256": str(pool["content_sha256"]),
            "legacy_manifest_file_sha256": file_sha256(legacy_manifest_path),
            "legacy_manifest_content_sha256": str(legacy_manifest["content_sha256"]),
            "legacy_benchmark_file_sha256": file_sha256(legacy_benchmark_path),
            "legacy_benchmark_content_sha256": str(legacy_benchmark["content_sha256"]),
        },
        "legacy_source_exclusions": list(exclusions),
        "selected": records,
        "partition_counts": dict(PARTITION_SIZES),
        "diversity_summary": _diversity_summary(records),
        "candidate_generator_version": MENTION_CATALOG_VERSION,
    }
    manifest = {"content_sha256": canonical_sha256(inner), **inner}
    validate_selection_manifest(manifest)
    return manifest


def _validate_policy(policy: Mapping[str, Any]) -> None:
    if not isinstance(policy, Mapping) or set(policy) != _EXPECTED_POLICY_KEYS:
        raise ValueError("phase2j selection policy is invalid")
    if policy["seed"] != SELECTION_SEED:
        raise ValueError("phase2j selection seed is not the preregistered seed")
    for key, expected in (
        ("target_window_count", TARGET_WINDOW_COUNT),
        ("target_distinct_video_source_groups", TARGET_WINDOW_COUNT),
    ):
        if policy[key] != expected:
            raise ValueError("phase2j selection target is invalid")
    if policy["one_window_per_upstream_source"] is not True:
        raise ValueError("phase2j source isolation policy is invalid")
    if not isinstance(policy["source_group_id_rule"], str) or not policy["source_group_id_rule"]:
        raise ValueError("phase2j source group rule is invalid")
    eligibility = policy["eligibility"]
    if not isinstance(eligibility, list) or not eligibility \
            or any(not isinstance(item, str) or not item for item in eligibility):
        raise ValueError("phase2j eligibility policy is invalid")
    score = policy["diversity_score"]
    if not isinstance(score, Mapping) or set(score) != _EXPECTED_DIVERSITY_SCORE_KEYS:
        raise ValueError("phase2j diversity score policy is invalid")
    expected_values = {
        "phenomenon_points": PHENOMENON_POINTS,
        "phenomenon_below_count": PHENOMENON_BELOW_COUNT,
        "role_points": ROLE_POINTS,
        "role_below_count": ROLE_BELOW_COUNT,
        "asr_band_points": ASR_BAND_POINTS,
        "asr_band_below_count": ASR_BAND_BELOW_COUNT,
        "unrepresented_champion_points": CHAMPION_POINTS,
    }
    for key, expected in expected_values.items():
        if isinstance(score[key], bool) or score[key] != expected:
            raise ValueError("phase2j diversity score constants are not preregistered")
    for key in ("asr_band_definition", "tie_break"):
        if not isinstance(score[key], str) or not score[key]:
            raise ValueError("phase2j diversity score rule text is invalid")
    partition = policy["partition"]
    if not isinstance(partition, Mapping) or set(partition) != _EXPECTED_PARTITION_POLICY_KEYS:
        raise ValueError("phase2j partition policy is invalid")
    for key, expected in PARTITION_SIZES.items():
        if isinstance(partition[key], bool) or partition[key] != expected:
            raise ValueError("phase2j partition sizes are not preregistered")
    if not isinstance(partition["order"], str) or not partition["order"]:
        raise ValueError("phase2j partition order rule is invalid")
    if not isinstance(policy["diversity_preference_statement"], str) \
            or not policy["diversity_preference_statement"]:
        raise ValueError("phase2j diversity preference statement is invalid")


def _validate_selected_record(
    record: Mapping[str, Any], *,
    seen_ids: set[str], seen_sources: set[str], seen_groups: set[str],
    partition_counts: dict[str, int],
) -> None:
    if not isinstance(record, Mapping) or set(record) != _EXPECTED_SELECTED_KEYS:
        raise ValueError("phase2j selected record is invalid")
    window_id = record["window_id"]
    upstream = record["upstream_source_id"]
    group = record["source_group_id"]
    if not isinstance(window_id, str) or not window_id \
            or not isinstance(upstream, str) or not upstream \
            or not isinstance(group, str) or not group:
        raise ValueError("phase2j selected record identity is invalid")
    if group != f"video:{upstream}":
        raise ValueError("phase2j source_group_id must derive from upstream_source_id")
    if window_id in seen_ids or upstream in seen_sources or group in seen_groups:
        raise ValueError("phase2j selected records contain duplicate source/window identity")
    seen_ids.add(window_id)
    seen_sources.add(upstream)
    seen_groups.add(group)
    start, end = record["upstream_start"], record["upstream_end"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) \
            or start < 0 or end <= start:
        raise ValueError("phase2j upstream offsets are invalid")
    text = record["source_text"]
    if not isinstance(text, str) or not text.strip() or end - start != len(text):
        raise ValueError("phase2j upstream offsets do not match exact source text")
    if record["source_text_sha256"] != hashlib.sha256(text.encode("utf-8")).hexdigest() \
            or not _SHA256.fullmatch(record["source_text_sha256"]):
        raise ValueError("phase2j source text hash is invalid")
    if not _SHA256.fullmatch(record["upstream_content_sha256"]):
        raise ValueError("phase2j upstream content hash is invalid")
    if isinstance(record["source_text_char_length"], bool) \
            or record["source_text_char_length"] != len(text):
        raise ValueError("phase2j source text character length is invalid")
    metadata = record["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "video_title", "role", "champion",
    } or any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError("phase2j selected metadata is invalid")
    phenomena = record["phenomena"]
    if not isinstance(phenomena, list) or tuple(phenomena) != tuple(sorted(set(phenomena))) \
            or any(value not in POOL_PHENOMENA for value in phenomena):
        raise ValueError("phase2j frozen phenomenon tags are invalid")
    expected_band = asr_punctuation_band(phenomena)
    if record["asr_punctuation_band"] != expected_band:
        raise ValueError("phase2j ASR punctuation band is not derived from frozen tags")
    partition = record["partition"]
    if partition not in PARTITION_SIZES:
        raise ValueError("phase2j partition assignment is invalid")
    partition_counts[partition] += 1
    if record["candidate_generator_version"] != MENTION_CATALOG_VERSION:
        raise ValueError("phase2j candidate generator version is unsupported")
    if isinstance(record["candidate_count"], bool) or not isinstance(record["candidate_count"], int) \
            or record["candidate_count"] <= 0:
        raise ValueError("phase2j candidate catalog count is invalid")
    if not _SHA256.fullmatch(record["candidate_catalog_sha256"]):
        raise ValueError("phase2j candidate catalog hash is invalid")
    record_inner = {key: value for key, value in record.items() if key != "canonical_record_sha256"}
    if record["canonical_record_sha256"] != canonical_sha256(record_inner):
        raise ValueError("phase2j canonical record hash is invalid")


def validate_selection_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the Phase 2J manifest envelope and every selected record."""
    if not isinstance(manifest, Mapping) or set(manifest) != _EXPECTED_MANIFEST_KEYS:
        raise ValueError("phase2j selection manifest envelope is invalid")
    if manifest["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise ValueError("phase2j selection manifest version is unsupported")
    if manifest["release_gate"] != "LOCKED":
        raise ValueError("phase2j release gate must remain LOCKED")
    if manifest["checkpoint"] != CHECKPOINT:
        raise ValueError("phase2j checkpoint marker is invalid")
    if not isinstance(manifest["purpose"], str) or not manifest["purpose"]:
        raise ValueError("phase2j manifest purpose is invalid")
    inner = {key: item for key, item in manifest.items() if key != "content_sha256"}
    if manifest["content_sha256"] != canonical_sha256(inner):
        raise ValueError("phase2j selection manifest content hash is invalid")
    _validate_policy(manifest["selection_policy"])
    input_hashes = manifest["input_hashes"]
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != _EXPECTED_INPUT_HASH_KEYS \
            or any(not _SHA256.fullmatch(value) for value in input_hashes.values()):
        raise ValueError("phase2j input hashes are invalid")
    exclusions = manifest["legacy_source_exclusions"]
    if not isinstance(exclusions, list) or exclusions != sorted(set(exclusions)) \
            or any(not isinstance(item, str) or not item for item in exclusions):
        raise ValueError("phase2j legacy source exclusions are invalid")
    records = manifest["selected"]
    if not isinstance(records, list) or len(records) != TARGET_WINDOW_COUNT:
        raise ValueError("phase2j must select exactly 30 windows")
    partition_counts = {key: 0 for key in PARTITION_SIZES}
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_groups: set[str] = set()
    for record in records:
        _validate_selected_record(
            record, seen_ids=seen_ids, seen_sources=seen_sources,
            seen_groups=seen_groups, partition_counts=partition_counts,
        )
    if manifest["partition_counts"] != dict(PARTITION_SIZES) \
            or partition_counts != dict(PARTITION_SIZES):
        raise ValueError("phase2j partition counts are invalid")
    if manifest["candidate_generator_version"] != MENTION_CATALOG_VERSION:
        raise ValueError("phase2j candidate generator version is unsupported")
    summary = manifest["diversity_summary"]
    if not isinstance(summary, Mapping) or set(summary) != _EXPECTED_DIVERSITY_SUMMARY_KEYS:
        raise ValueError("phase2j diversity summary is invalid")
    expected_summary = _diversity_summary(records)
    if summary != expected_summary:
        raise ValueError("phase2j diversity summary does not match selected records")
    if isinstance(summary["candidate_count"], bool) or summary["candidate_count"] <= 0:
        raise ValueError("phase2j candidate total is invalid")


def verify_selection_manifest_catalogs(manifest: Mapping[str, Any]) -> None:
    """Regenerate frozen candidate catalogs and verify bound count/hash."""
    validate_selection_manifest(manifest)
    for record in manifest["selected"]:
        binding = candidate_catalog_binding(record)
        if binding["candidate_generator_version"] != record["candidate_generator_version"] \
                or binding["candidate_count"] != record["candidate_count"] \
                or binding["candidate_catalog_sha256"] != record["candidate_catalog_sha256"]:
            raise ValueError(
                f"phase2j candidate catalog binding is not reproducible for {record['window_id']}",
            )


def verify_selection_manifest_inputs(
    manifest: Mapping[str, Any],
    *,
    pool_path: Path,
    legacy_manifest_path: Path,
    legacy_benchmark_path: Path,
    verify_catalogs: bool = True,
    reproduce_selection: bool = False,
) -> None:
    """Bind the manifest back to the immutable pool and legacy inputs."""
    validate_selection_manifest(manifest)
    if not isinstance(verify_catalogs, bool) or not isinstance(reproduce_selection, bool):
        raise ValueError("phase2j verification flags must be boolean")
    pool = load_semantic_window_pool(pool_path)
    legacy_manifest = load_legacy_manifest(legacy_manifest_path)
    legacy_benchmark = load_legacy_benchmark(
        legacy_benchmark_path, manifest=legacy_manifest,
    )
    expected_hashes = {
        "pool_file_sha256": file_sha256(pool_path),
        "pool_content_sha256": str(pool["content_sha256"]),
        "legacy_manifest_file_sha256": file_sha256(legacy_manifest_path),
        "legacy_manifest_content_sha256": str(legacy_manifest["content_sha256"]),
        "legacy_benchmark_file_sha256": file_sha256(legacy_benchmark_path),
        "legacy_benchmark_content_sha256": str(legacy_benchmark["content_sha256"]),
    }
    if manifest["input_hashes"] != expected_hashes:
        raise ValueError("phase2j input files do not match retained hashes")
    exclusions = legacy_source_exclusions(legacy_manifest, legacy_benchmark)
    if manifest["legacy_source_exclusions"] != list(exclusions):
        raise ValueError("phase2j legacy exclusions do not match the loaded benchmark")
    pool_by_window = {item["window_id"]: item for item in pool["windows"]}
    for record in manifest["selected"]:
        source = pool_by_window.get(record["window_id"])
        if source is None:
            raise ValueError("phase2j selected window is absent from the retained pool")
        if source["source_text"] != record["source_text"] \
                or source["source_text_sha256"] != record["source_text_sha256"] \
                or source["upstream_source_id"] != record["upstream_source_id"] \
                or source["upstream_start"] != record["upstream_start"] \
                or source["upstream_end"] != record["upstream_end"] \
                or source["upstream_content_sha256"] != record["upstream_content_sha256"] \
                or source["metadata"] != record["metadata"] \
                or source["phenomena"] != record["phenomena"]:
            raise ValueError("phase2j selected record contradicts the retained pool")
    if reproduce_selection:
        rebuilt = select_windows(pool, excluded_sources=exclusions)
        rebuilt_identities = [
            (item["source_group_id"], item["window_id"], item["partition"])
            for item in rebuilt
        ]
        manifest_identities = [
            (item["source_group_id"], item["window_id"], item["partition"])
            for item in manifest["selected"]
        ]
        if rebuilt_identities != manifest_identities:
            raise ValueError("phase2j selection is not deterministic")
    if verify_catalogs:
        verify_selection_manifest_catalogs(manifest)


def load_selection_manifest(path: Path) -> Mapping[str, Any]:
    """Strict canonical load with duplicate-key rejection and validation."""
    body = _load_json_strict(path, label="phase2j selection manifest")
    validate_selection_manifest(body)
    return body


__all__ = [
    "ASR_BANDS", "ASR_PUNCTUATED", "ASR_PUNCTUATION_POOR", "CHECKPOINT",
    "PARTITIONS", "PARTITION_EXPANDED_DEV", "PARTITION_FROZEN_REPLICATION",
    "PARTITION_SIZES", "SELECTION_SCHEMA_VERSION", "SELECTION_SEED",
    "TARGET_WINDOW_COUNT", "asr_punctuation_band",
    "build_selection_manifest", "candidate_catalog_binding",
    "canonical_sha256", "file_sha256", "legacy_source_exclusions",
    "load_legacy_benchmark", "load_legacy_manifest",
    "load_selection_manifest", "marginal_diversity_score", "select_windows",
    "validate_selection_manifest", "verify_selection_manifest_catalogs",
    "verify_selection_manifest_inputs",
]
