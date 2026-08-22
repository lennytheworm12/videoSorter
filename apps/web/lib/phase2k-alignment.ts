/**
 * Pure Phase 2K downstream semantic-target alignment review utilities.
 *
 * This module intentionally contains no Node or browser APIs (Web Crypto is
 * injected as a digest function) so it can be unit-tested under Jest and
 * imported safely by the client bundle.
 *
 * Data boundary: the only accepted input is the official blank
 * "phase2k-downstream-alignment-packet-v1" JSON.  The packet is explicitly
 * model/scorer blind: no downstream predictions, model results, scores,
 * scorers, or architectures are ever loaded, displayed, or generated.  The
 * packet is validated strictly in the browser (exact keys, versions, gates,
 * canonical content hash, every SHA-256, boundary rule, blank decisions, and
 * cross-window consistency), a session bound to the packet content hash is
 * kept locally, and a compact decisions map is produced only when every item
 * is explicitly complete.  No decision, reviewer, or timestamp is ever
 * inferred or fabricated.
 */

export const PACKET_SCHEMA_VERSION = "phase2k-downstream-alignment-packet-v1";
export const SESSION_SCHEMA_VERSION = "phase2k-downstream-alignment-session-v1";
export const DECISIONS_FILENAME = "phase2k-downstream-alignment-decisions-v1.json";

export const RELEASE_GATE_AWAITING_HUMAN_REVIEW = "AWAITING_HUMAN_REVIEW";
export const RELEASE_GATE_REVIEWED = "REVIEWED";
export const BOUNDARY_RULE_VERSION =
  "phase2k-target-boundary-rule-v1-phase2j-terminal-punctuation";

export const TARGET_COUNT = 311;
export const TARGET_WINDOW_COUNT = 30;
export const UNCHANGED_ENDPOINT_COUNT = 263;
export const CORRECTED_ENDPOINT_COUNT = 48;
export const MISSING_PERIOD_COUNT = 28;
export const MISSING_COMMA_COUNT = 20;

export const ALIGNMENT_DECISION_STATES = [
  "ALIGNED",
  "ABSENT",
  "AMBIGUOUS",
  "MULTIPLE_CANDIDATES",
] as const;

export const CORRECTION_STATUSES = [
  "UNCHANGED",
  "TERMINAL_PUNCTUATION_DROPPED",
] as const;

export const NODE_TYPES = [
  "ENTITY",
  "ABILITY_OR_RESOURCE",
  "EVENT",
  "ACTION",
  "STATE",
  "OUTCOME",
  "QUANTITY",
  "TIME",
  "LOCATION_OR_SPACE",
] as const;

export type NodeType = (typeof NODE_TYPES)[number] | null;
export type DecisionState = (typeof ALIGNMENT_DECISION_STATES)[number];
export type CorrectionStatus = (typeof CORRECTION_STATUSES)[number];

const PACKET_TOP_LEVEL_KEYS = [
  "schema_version",
  "content_sha256",
  "purpose",
  "release_gate",
  "dataset_binding",
  "boundary_rule",
  "items",
] as const;

const DATASET_BINDING_KEYS = [
  "phase2k_records_sha256",
  "phase2j_reviewed_packet_sha256",
  "phase2j_coverage_sha256",
  "finalized_human_packet_sha256",
  "human_summary_sha256",
  "completed_transformation_audit_sha256",
  "window_ids_sha256",
  "window_count",
  "target_count",
  "human_review_gate_status",
] as const;

const BOUNDARY_RULE_KEYS = [
  "rule_version",
  "unchanged_count",
  "corrected_count",
  "dropped_terminal_period_count",
  "dropped_terminal_comma_count",
  "behavior",
] as const;

const ITEM_KEYS = [
  "alignment_id",
  "window_id",
  "endpoint_id",
  "node_type",
  "bronze_target",
  "representation",
  "decision",
] as const;

const BRONZE_TARGET_KEYS = [
  "original_start",
  "original_end",
  "original_text",
  "source_absolute_start",
  "source_absolute_end",
  "evaluation_start",
  "evaluation_end",
  "evaluation_text",
  "correction_status",
  "dropped_text",
] as const;

const REPRESENTATION_KEYS = [
  "clean_target_transcript",
  "clean_target_transcript_sha256",
  "polished_text",
  "polished_text_sha256",
] as const;

const DECISION_KEYS = [
  "state",
  "polished_spans",
  "reviewer",
  "completed_at",
  "notes",
] as const;

const SESSION_DECISION_KEYS = [...DECISION_KEYS, "complete"] as const;

const SPAN_KEYS = ["start", "end", "text"] as const;

// Exact mirror of the Python contract: keys are matched case-sensitively and
// string values only when they equal one of the forbidden tokens, so
// legitimate purpose/behavior prose ("scorer-blind", "no predictions") never
// trips the scanner.
const FORBIDDEN_ALIGNMENT_KEYS = new Set([
  "model_predictions",
  "model_scoring",
  "predictions",
  "prediction",
  "predicted",
  "predicted_label",
  "predicted_labels",
  "score",
  "scores",
  "probability",
  "probabilities",
  "rank",
  "ranks",
  "ranking",
  "rankings",
  "threshold",
  "thresholds",
  "scorer",
  "scoring",
  "architecture",
  "architectures",
  "semantic_ir",
  "semantic_claims",
  "semantic_extraction",
  "entities",
  "relations",
  "claims",
  "extracted",
  "extractor",
  "generative",
  "discriminative",
  "model_result",
  "model_results",
]);

const FORBIDDEN_ALIGNMENT_VALUES = new Set([
  "PHASE2F",
  "PHASE2H",
  "GENERATIVE",
  "DISCRIMINATIVE",
  "SCORED",
  "PREDICTED",
]);

export type BronzeTarget = {
  original_start: number;
  original_end: number;
  original_text: string;
  source_absolute_start: number;
  source_absolute_end: number;
  evaluation_start: number;
  evaluation_end: number;
  evaluation_text: string;
  correction_status: CorrectionStatus;
  dropped_text: null | "." | ",";
};

export type Representation = {
  clean_target_transcript: string;
  clean_target_transcript_sha256: string;
  polished_text: string;
  polished_text_sha256: string;
};

export type PolishedSpan = {
  start: number;
  end: number;
  text: string;
};

export type BlankDecision = {
  state: null;
  polished_spans: [];
  reviewer: null;
  completed_at: null;
  notes: [];
};

export type PacketItem = {
  alignment_id: string;
  window_id: string;
  endpoint_id: string;
  node_type: NodeType;
  bronze_target: BronzeTarget;
  representation: Representation;
  decision: BlankDecision;
};

export type DatasetBinding = {
  phase2k_records_sha256: string;
  phase2j_reviewed_packet_sha256: string;
  phase2j_coverage_sha256: string;
  finalized_human_packet_sha256: string;
  human_summary_sha256: string;
  completed_transformation_audit_sha256: string;
  window_ids_sha256: string;
  window_count: 30;
  target_count: 311;
  human_review_gate_status: "PASSED";
};

export type BoundaryRule = {
  rule_version: typeof BOUNDARY_RULE_VERSION;
  unchanged_count: 263;
  corrected_count: 48;
  dropped_terminal_period_count: 28;
  dropped_terminal_comma_count: 20;
  behavior: string;
};

export type AlignmentPacket = {
  schema_version: typeof PACKET_SCHEMA_VERSION;
  content_sha256: string;
  purpose: string;
  release_gate: "AWAITING_HUMAN_REVIEW";
  dataset_binding: DatasetBinding;
  boundary_rule: BoundaryRule;
  items: PacketItem[];
};

export type SessionDecision = {
  state: DecisionState | null;
  polished_spans: PolishedSpan[];
  reviewer: string;
  completed_at: string | null;
  notes: string[];
  complete: boolean;
};

export type SessionItem = {
  alignment_id: string;
  window_id: string;
  endpoint_id: string;
  node_type: NodeType;
  bronze_target: BronzeTarget;
  representation: Representation;
  decision: SessionDecision;
};

export type AlignmentSession = {
  schema_version: typeof SESSION_SCHEMA_VERSION;
  packet_schema_version: typeof PACKET_SCHEMA_VERSION;
  packet_sha256: string;
  exported_at: string | null;
  items: SessionItem[];
};

export type ProgressSummary = {
  total: number;
  complete: number;
  /** Fully ready (state/reviewer/cardinality) but not explicitly marked complete. */
  ready: number;
  /** Has at least one review edit but is not ready to complete. */
  in_progress: number;
  untouched: number;
};

export type ValidationResult =
  | { ok: true; session: AlignmentSession }
  | { ok: false; errors: string[] };

export type CompletionResult =
  | { ok: true; session: AlignmentSession }
  | { ok: false; errors: string[] };

export type DecisionsMapEntry = {
  state: DecisionState;
  polished_spans: PolishedSpan[];
  reviewer: string;
  completed_at: string;
  notes: string[];
};

/** Plain object keyed by alignment_id in deterministic packet order. */
export type DecisionsMap = Record<string, DecisionsMapEntry>;

const HEX64 = /^[0-9a-f]{64}$/;
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/;

// ---------------------------------------------------------------------------
// Canonical serialization / hashing (Python `canonical_sha256` semantics)
// ---------------------------------------------------------------------------

export type Sha256Digest = (utf8: Uint8Array) => string | Promise<string>;

/**
 * Recursive object-key-sorted compact JSON serialization matching Python's
 * `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
 * Array order is retained.  The alignment schema only carries strings,
 * integers, booleans, nulls, arrays, and objects, so number formatting
 * matches.
 */
export function canonicalSerialize(value: unknown): string {
  if (value === null || typeof value === "boolean" || typeof value === "number") {
    return JSON.stringify(value);
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalSerialize(item)).join(",")}]`;
  }
  if (isRecordObject(value)) {
    const entries = Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalSerialize(value[key])}`);
    return `{${entries.join(",")}}`;
  }
  throw new TypeError(`cannot canonically serialize ${typeof value}`);
}

export function utf8Bytes(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

export function bytesToHex(bytes: Uint8Array | ArrayBuffer): string {
  const view =
    bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes as ArrayBuffer);
  let hex = "";
  for (const byte of view) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}

export async function defaultDigest(utf8: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new Error("Web Crypto is unavailable in this environment");
  }
  return bytesToHex(await subtle.digest("SHA-256", utf8 as unknown as BufferSource));
}

export async function computeCanonicalSha256(
  value: unknown,
  digest: Sha256Digest = defaultDigest,
): Promise<string> {
  return Promise.resolve(digest(utf8Bytes(canonicalSerialize(value))));
}

/** Canonical content SHA-256 of a packet excluding its content_sha256 key. */
export function canonicalPacketSha256(
  packet: Record<string, unknown>,
  digest: Sha256Digest = defaultDigest,
): Promise<string> {
  const inner: Record<string, unknown> = {};
  for (const key of Object.keys(packet)) {
    if (key !== "content_sha256") {
      inner[key] = packet[key];
    }
  }
  return Promise.resolve(computeCanonicalSha256(inner, digest));
}

/** Ordinary UTF-8 SHA-256 of a text value. */
export function textSha256(
  text: string,
  digest: Sha256Digest = defaultDigest,
): Promise<string> {
  return Promise.resolve(digest(utf8Bytes(text)));
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => actual.includes(key));
}

function arraysEqual(left: readonly unknown[], right: readonly unknown[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function deepEqual(left: unknown, right: unknown): boolean {
  if (left === right) {
    return true;
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    return (
      left.length === right.length &&
      left.every((value, index) => deepEqual(value, right[index]))
    );
  }
  if (isRecordObject(left) && isRecordObject(right)) {
    const leftKeys = Object.keys(left);
    const rightKeys = Object.keys(right);
    return (
      leftKeys.length === rightKeys.length &&
      leftKeys.every(
        (key) =>
          Object.prototype.hasOwnProperty.call(right, key) &&
          deepEqual(left[key], right[key]),
      ) &&
      rightKeys.every((key) => Object.prototype.hasOwnProperty.call(left, key))
    );
  }
  return false;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new Error(`${label} must be a string`);
  }
  return value;
}

function requireNonEmptyString(value: unknown, label: string): string {
  const text = requireString(value, label);
  if (text.trim() === "") {
    throw new Error(`${label} must be a non-empty string`);
  }
  return text;
}

function requireHex64(value: unknown, label: string): string {
  const text = requireString(value, label);
  if (!HEX64.test(text)) {
    throw new Error(`${label} must be a 64-character lowercase hex string`);
  }
  return text;
}

function requireInt(
  value: unknown,
  label: string,
  minimum: number | null = null,
): number {
  if (typeof value === "boolean" || !Number.isInteger(value)) {
    throw new Error(`${label} must be an integer`);
  }
  if (minimum !== null && (value as number) < minimum) {
    throw new Error(`${label} must be >= ${minimum}`);
  }
  return value as number;
}

function requireEnum<T extends string>(
  value: unknown,
  options: readonly T[],
  label: string,
): T {
  const text = requireString(value, label);
  if (!(options as readonly string[]).includes(text)) {
    throw new Error(`${label} has invalid value ${JSON.stringify(text)}`);
  }
  return text as T;
}

/**
 * Recursive conservative leak scan mirroring the Python contract: object
 * keys are always scanned (exact match), and string values are scanned only
 * when they exactly equal a forbidden token.  Free prose such as the packet
 * purpose or boundary behavior never matches and is preserved.
 */
export function findForbiddenLeaks(value: unknown, path = ""): string[] {
  const problems: string[] = [];
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      problems.push(...findForbiddenLeaks(item, `${path}[${index}]`));
    });
    return problems;
  }
  if (!isRecordObject(value)) {
    if (typeof value === "string" && FORBIDDEN_ALIGNMENT_VALUES.has(value)) {
      problems.push(`forbidden value ${JSON.stringify(value)} at ${path || "root"}`);
    }
    return problems;
  }
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_ALIGNMENT_KEYS.has(key)) {
      problems.push(`forbidden field "${key}" at ${path || "root"}`);
    }
    const childPath = path ? `${path}.${key}` : key;
    if (typeof child === "string") {
      if (FORBIDDEN_ALIGNMENT_VALUES.has(child)) {
        problems.push(`forbidden value ${JSON.stringify(child)} at ${childPath}`);
      }
    } else {
      problems.push(...findForbiddenLeaks(child, childPath));
    }
  }
  return problems;
}

/** Exact half-open span validation against the polished text. */
export function validatePolishedSpan(
  span: unknown,
  polishedText: string,
  label: string,
): PolishedSpan {
  if (!isRecordObject(span) || !hasExactKeys(span, SPAN_KEYS)) {
    throw new Error(`${label} must be exactly {start, end, text}`);
  }
  const start = requireInt(span.start, `${label} start`, 0);
  const end = requireInt(span.end, `${label} end`, 0);
  const text = requireString(span.text, `${label} text`);
  if (!(start < end && end <= polishedText.length)) {
    throw new Error(`${label} span is out of bounds for the polished text`);
  }
  if (polishedText.slice(start, end) !== text) {
    throw new Error(`${label} is not the exact half-open slice of the polished text`);
  }
  return { start, end, text };
}

function validateSortedUniqueSpans(
  spans: PolishedSpan[],
  label: string,
): void {
  const pairs = spans.map((span) => `${span.start}:${span.end}`);
  if (new Set(pairs).size !== pairs.length) {
    throw new Error(`${label} polished spans must be unique`);
  }
  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  if (pairs.join(",") !== sorted.map((span) => `${span.start}:${span.end}`).join(",")) {
    throw new Error(`${label} polished spans must be deterministically sorted by (start, end)`);
  }
}

function sanitizeBronzeTarget(raw: unknown, label: string): BronzeTarget {
  if (!isRecordObject(raw) || !hasExactKeys(raw, BRONZE_TARGET_KEYS)) {
    throw new Error(`${label} keys are invalid`);
  }
  const originalStart = requireInt(raw.original_start, `${label} original_start`, 0);
  const originalEnd = requireInt(raw.original_end, `${label} original_end`, 0);
  const originalText = requireString(raw.original_text, `${label} original_text`);
  const sourceAbsoluteStart = requireInt(
    raw.source_absolute_start,
    `${label} source_absolute_start`,
    0,
  );
  const sourceAbsoluteEnd = requireInt(
    raw.source_absolute_end,
    `${label} source_absolute_end`,
    0,
  );
  const evaluationStart = requireInt(
    raw.evaluation_start,
    `${label} evaluation_start`,
    0,
  );
  const evaluationEnd = requireInt(
    raw.evaluation_end,
    `${label} evaluation_end`,
    0,
  );
  const evaluationText = requireString(
    raw.evaluation_text,
    `${label} evaluation_text`,
  );
  if (!(originalStart < originalEnd)) {
    throw new Error(`${label} original span is invalid`);
  }
  if (originalEnd - originalStart !== originalText.length) {
    throw new Error(`${label} original_text is not the exact original slice`);
  }
  if (sourceAbsoluteEnd - sourceAbsoluteStart !== originalEnd - originalStart) {
    throw new Error(`${label} source absolute span is inconsistent`);
  }
  const status = requireEnum(
    raw.correction_status,
    CORRECTION_STATUSES,
    `${label} correction_status`,
  );
  const droppedText = raw.dropped_text;
  if (status === "UNCHANGED") {
    if (droppedText !== null) {
      throw new Error(`${label} UNCHANGED target must have null dropped_text`);
    }
    if (
      evaluationStart !== originalStart ||
      evaluationEnd !== originalEnd ||
      evaluationText !== originalText
    ) {
      throw new Error(`${label} UNCHANGED evaluation span must equal the original`);
    }
  } else {
    if (droppedText !== "." && droppedText !== ",") {
      throw new Error(`${label} dropped_text must be "." or ","`);
    }
    if (
      originalText.length === 0 ||
      originalText.slice(-1) !== droppedText ||
      evaluationStart !== originalStart ||
      evaluationEnd !== originalEnd - 1 ||
      evaluationText !== originalText.slice(0, -1)
    ) {
      throw new Error(
        `${label} corrected evaluation span must drop exactly one terminal punctuation character`,
      );
    }
  }
  return {
    original_start: originalStart,
    original_end: originalEnd,
    original_text: originalText,
    source_absolute_start: sourceAbsoluteStart,
    source_absolute_end: sourceAbsoluteEnd,
    evaluation_start: evaluationStart,
    evaluation_end: evaluationEnd,
    evaluation_text: evaluationText,
    correction_status: status,
    dropped_text: droppedText as null | "." | ",",
  };
}

async function sanitizeRepresentation(
  raw: unknown,
  label: string,
  digest: Sha256Digest,
): Promise<Representation> {
  if (!isRecordObject(raw) || !hasExactKeys(raw, REPRESENTATION_KEYS)) {
    throw new Error(`${label} keys are invalid`);
  }
  const cleanText = requireString(
    raw.clean_target_transcript,
    `${label} clean_target_transcript`,
  );
  const polishedText = requireString(raw.polished_text, `${label} polished_text`);
  const cleanSha = requireHex64(
    raw.clean_target_transcript_sha256,
    `${label} clean_target_transcript_sha256`,
  );
  const polishedSha = requireHex64(
    raw.polished_text_sha256,
    `${label} polished_text_sha256`,
  );
  if ((await textSha256(cleanText, digest)) !== cleanSha) {
    throw new Error(`${label} clean_target_transcript_sha256 does not match the text`);
  }
  if ((await textSha256(polishedText, digest)) !== polishedSha) {
    throw new Error(`${label} polished_text_sha256 does not match the text`);
  }
  return {
    clean_target_transcript: cleanText,
    clean_target_transcript_sha256: cleanSha,
    polished_text: polishedText,
    polished_text_sha256: polishedSha,
  };
}

function sanitizeBlankDecision(raw: unknown, label: string): BlankDecision {
  if (!isRecordObject(raw) || !hasExactKeys(raw, DECISION_KEYS)) {
    throw new Error(`${label} keys are invalid`);
  }
  if (raw.state !== null) {
    throw new Error(`blank ${label} must have a null state`);
  }
  if (!Array.isArray(raw.polished_spans) || raw.polished_spans.length > 0) {
    throw new Error(`blank ${label} must have empty polished spans`);
  }
  if (raw.reviewer !== null || raw.completed_at !== null) {
    throw new Error(`blank ${label} must have null reviewer/completed_at`);
  }
  if (!Array.isArray(raw.notes) || raw.notes.length > 0) {
    throw new Error(`blank ${label} must have empty notes`);
  }
  return {
    state: null,
    polished_spans: [],
    reviewer: null,
    completed_at: null,
    notes: [],
  };
}

/**
 * Strict sanitizer/validator for the official blank packet.  Verifies exact
 * recursive key sets, schema version, release gate, canonical content hash,
 * every text SHA-256, the 311/30 identity counts, stable alignment IDs, the
 * 263/48/28/20 boundary rule with exact correction slice invariants, blank
 * decisions, and cross-window consistency (the dataset binding window hash
 * must match the item window IDs).  Rejects finalized, forged, or
 * model/scorer-leaking packets.
 */
export async function sanitizePacket(
  raw: unknown,
  digest: Sha256Digest = defaultDigest,
): Promise<AlignmentPacket> {
  if (!isRecordObject(raw)) {
    throw new Error("phase2k downstream alignment packet must be a JSON object");
  }
  const leaks = findForbiddenLeaks(raw);
  if (leaks.length > 0) {
    throw new Error(
      `alignment packet contains forbidden model/scorer material: ${leaks.join("; ")}`,
    );
  }
  if (!hasExactKeys(raw, PACKET_TOP_LEVEL_KEYS)) {
    throw new Error("phase2k downstream alignment packet top-level keys are invalid");
  }
  if (raw.schema_version !== PACKET_SCHEMA_VERSION) {
    throw new Error(
      `phase2k downstream alignment packet schema_version must be ${PACKET_SCHEMA_VERSION}`,
    );
  }
  const purpose = requireNonEmptyString(
    raw.purpose,
    "phase2k downstream alignment packet purpose",
  );
  if (raw.release_gate !== RELEASE_GATE_AWAITING_HUMAN_REVIEW) {
    throw new Error(
      "alignment packet release_gate must be AWAITING_HUMAN_REVIEW; finalized packets are rejected",
    );
  }
  const contentSha256 = requireHex64(
    raw.content_sha256,
    "phase2k downstream alignment packet content_sha256",
  );
  const recomputed = await canonicalPacketSha256(raw, digest);
  if (recomputed !== contentSha256) {
    throw new Error(
      "phase2k downstream alignment packet content_sha256 does not match the canonical content",
    );
  }

  const datasetBindingRaw = raw.dataset_binding;
  if (!isRecordObject(datasetBindingRaw) || !hasExactKeys(datasetBindingRaw, DATASET_BINDING_KEYS)) {
    throw new Error("alignment packet dataset_binding keys are invalid");
  }
  const bindingHexKeys = [
    "phase2k_records_sha256",
    "phase2j_reviewed_packet_sha256",
    "phase2j_coverage_sha256",
    "finalized_human_packet_sha256",
    "human_summary_sha256",
    "completed_transformation_audit_sha256",
    "window_ids_sha256",
  ];
  for (const key of bindingHexKeys) {
    requireHex64(datasetBindingRaw[key], `alignment packet dataset_binding.${key}`);
  }
  const windowCount = requireInt(
    datasetBindingRaw.window_count,
    "alignment packet dataset_binding.window_count",
    1,
  );
  if (windowCount !== TARGET_WINDOW_COUNT) {
    throw new Error("alignment packet dataset_binding.window_count must be 30");
  }
  const targetCount = requireInt(
    datasetBindingRaw.target_count,
    "alignment packet dataset_binding.target_count",
    1,
  );
  if (targetCount !== TARGET_COUNT) {
    throw new Error("alignment packet dataset_binding.target_count must be 311");
  }
  if (datasetBindingRaw.human_review_gate_status !== "PASSED") {
    throw new Error("alignment packet dataset_binding.human_review_gate_status must be PASSED");
  }

  const boundaryRaw = raw.boundary_rule;
  if (!isRecordObject(boundaryRaw) || !hasExactKeys(boundaryRaw, BOUNDARY_RULE_KEYS)) {
    throw new Error("alignment packet boundary_rule keys are invalid");
  }
  if (boundaryRaw.rule_version !== BOUNDARY_RULE_VERSION) {
    throw new Error("alignment packet boundary_rule.rule_version is invalid");
  }
  const expectedCounts: Array<[string, number]> = [
    ["unchanged_count", UNCHANGED_ENDPOINT_COUNT],
    ["corrected_count", CORRECTED_ENDPOINT_COUNT],
    ["dropped_terminal_period_count", MISSING_PERIOD_COUNT],
    ["dropped_terminal_comma_count", MISSING_COMMA_COUNT],
  ];
  for (const [key, expected] of expectedCounts) {
    if (requireInt(boundaryRaw[key], `alignment packet boundary_rule.${key}`) !== expected) {
      throw new Error(`alignment packet boundary_rule.${key} is invalid`);
    }
  }
  requireNonEmptyString(boundaryRaw.behavior, "alignment packet boundary_rule.behavior");

  if (!Array.isArray(raw.items)) {
    throw new Error("alignment packet items must be an array");
  }
  if (raw.items.length !== TARGET_COUNT) {
    throw new Error(
      `alignment packet must contain exactly ${TARGET_COUNT} items (found ${raw.items.length})`,
    );
  }

  const seenAlignmentIds = new Set<string>();
  const seenEndpointIds = new Set<string>();
  const items: PacketItem[] = [];
  for (let index = 0; index < raw.items.length; index += 1) {
    const item = raw.items[index];
    const label = `alignment packet items[${index}]`;
    if (!isRecordObject(item) || !hasExactKeys(item, ITEM_KEYS)) {
      throw new Error(`${label} keys are invalid`);
    }
    const alignmentId = requireNonEmptyString(item.alignment_id, `${label} alignment_id`);
    if (!alignmentId.startsWith("p2k:align:")) {
      throw new Error(`${label} alignment_id prefix is invalid`);
    }
    if (seenAlignmentIds.has(alignmentId)) {
      throw new Error(`alignment item IDs must be unique (duplicate ${alignmentId})`);
    }
    seenAlignmentIds.add(alignmentId);
    const windowId = requireNonEmptyString(item.window_id, `${label} window_id`);
    const endpointId = requireNonEmptyString(item.endpoint_id, `${label} endpoint_id`);
    if (seenEndpointIds.has(endpointId)) {
      throw new Error(`endpoint IDs must be unique (duplicate ${endpointId})`);
    }
    seenEndpointIds.add(endpointId);
    if (alignmentId !== `p2k:align:${endpointId}`) {
      throw new Error(`${label} alignment_id must derive from endpoint_id`);
    }
    const nodeType = item.node_type;
    if (nodeType !== null && !(NODE_TYPES as readonly string[]).includes(nodeType as string)) {
      throw new Error(`${label} node_type is invalid`);
    }
    const bronzeTarget = sanitizeBronzeTarget(item.bronze_target, `${label} bronze_target`);
    const representation = await sanitizeRepresentation(
      item.representation,
      `${label} representation`,
      digest,
    );
    const decision = sanitizeBlankDecision(item.decision, `${label} decision`);
    items.push({
      alignment_id: alignmentId,
      window_id: windowId,
      endpoint_id: endpointId,
      node_type: nodeType as NodeType,
      bronze_target: bronzeTarget,
      representation,
      decision,
    });
  }

  const windowIds = [...new Set(items.map((item) => item.window_id))].sort();
  if (windowIds.length !== TARGET_WINDOW_COUNT) {
    throw new Error(
      `alignment items must span exactly ${TARGET_WINDOW_COUNT} windows (found ${windowIds.length})`,
    );
  }
  const expectedWindowHash = await computeCanonicalSha256(windowIds, digest);
  if (datasetBindingRaw.window_ids_sha256 !== expectedWindowHash) {
    throw new Error(
      "alignment packet dataset_binding.window_ids_sha256 does not match the item window IDs",
    );
  }

  let unchanged = 0;
  let corrected = 0;
  let periods = 0;
  let commas = 0;
  for (const item of items) {
    const status = item.bronze_target.correction_status;
    if (status === "UNCHANGED") {
      unchanged += 1;
    } else {
      corrected += 1;
      if (item.bronze_target.dropped_text === ".") {
        periods += 1;
      } else if (item.bronze_target.dropped_text === ",") {
        commas += 1;
      }
    }
  }
  if (
    unchanged !== UNCHANGED_ENDPOINT_COUNT ||
    corrected !== CORRECTED_ENDPOINT_COUNT ||
    periods !== MISSING_PERIOD_COUNT ||
    commas !== MISSING_COMMA_COUNT
  ) {
    throw new Error(
      `alignment item boundary counts are invalid; expected ${UNCHANGED_ENDPOINT_COUNT} unchanged / ${CORRECTED_ENDPOINT_COUNT} corrected with ${MISSING_PERIOD_COUNT} periods and ${MISSING_COMMA_COUNT} commas`,
    );
  }

  return {
    schema_version: PACKET_SCHEMA_VERSION,
    content_sha256: contentSha256,
    purpose,
    release_gate: RELEASE_GATE_AWAITING_HUMAN_REVIEW,
    dataset_binding: {
      ...(datasetBindingRaw as Record<string, unknown>),
    } as unknown as DatasetBinding,
    boundary_rule: {
      ...(boundaryRaw as Record<string, unknown>),
    } as unknown as BoundaryRule,
    items,
  };
}

// ---------------------------------------------------------------------------
// Session helpers
// ---------------------------------------------------------------------------

function findItem(session: AlignmentSession, alignmentId: string): SessionItem {
  const item = session.items.find(
    (candidate) => candidate.alignment_id === alignmentId,
  );
  if (!item) {
    throw new Error(`alignment item not found: ${alignmentId}`);
  }
  return item;
}

function withItem(
  session: AlignmentSession,
  alignmentId: string,
  update: (item: SessionItem) => SessionItem,
): AlignmentSession {
  return {
    ...session,
    items: session.items.map((item) =>
      item.alignment_id === alignmentId ? update(item) : item,
    ),
  };
}

/** Build a fresh review session bound to the packet content hash. */
export function buildSessionFromPacket(packet: AlignmentPacket): AlignmentSession {
  return {
    schema_version: SESSION_SCHEMA_VERSION,
    packet_schema_version: PACKET_SCHEMA_VERSION,
    packet_sha256: packet.content_sha256,
    exported_at: null,
    items: packet.items.map((item) => ({
      alignment_id: item.alignment_id,
      window_id: item.window_id,
      endpoint_id: item.endpoint_id,
      node_type: item.node_type,
      bronze_target: item.bronze_target,
      representation: item.representation,
      decision: {
        state: null,
        polished_spans: [],
        reviewer: "",
        completed_at: null,
        notes: [],
        complete: false,
      },
    })),
  };
}

function decisionCardinalityError(
  state: DecisionState,
  spanCount: number,
): string | null {
  if (state === "ALIGNED" && spanCount === 0) {
    return "ALIGNED requires at least one polished span";
  }
  if (state === "ABSENT" && spanCount > 0) {
    return "ABSENT requires zero polished spans";
  }
  if (state === "MULTIPLE_CANDIDATES" && spanCount < 2) {
    return "MULTIPLE_CANDIDATES requires at least two polished spans";
  }
  return null;
}

/** Missing completion inputs for an item; empty means the item is ready. */
export function itemMissingFields(item: SessionItem): string[] {
  const missing: string[] = [];
  const decision = item.decision;
  if (decision.state === null) {
    missing.push("state");
  }
  if (decision.reviewer.trim() === "") {
    missing.push("reviewer");
  }
  if (decision.state !== null) {
    const cardinality = decisionCardinalityError(
      decision.state,
      decision.polished_spans.length,
    );
    if (cardinality !== null) {
      missing.push(cardinality);
    }
  }
  decision.polished_spans.forEach((span, index) => {
    try {
      validatePolishedSpan(
        span,
        item.representation.polished_text,
        `${item.alignment_id} polished_spans[${index}]`,
      );
    } catch (error) {
      missing.push(error instanceof Error ? error.message : "invalid span");
    }
  });
  try {
    validateSortedUniqueSpans(decision.polished_spans, item.alignment_id);
  } catch (error) {
    missing.push(error instanceof Error ? error.message : "invalid span order");
  }
  return missing;
}

/** Set (or clear with null) the alignment state.  Editing uncompletes. */
export function setItemState(
  session: AlignmentSession,
  alignmentId: string,
  state: DecisionState | null,
): AlignmentSession {
  if (state !== null && !(ALIGNMENT_DECISION_STATES as readonly string[]).includes(state)) {
    throw new Error(`invalid alignment state: ${String(state)}`);
  }
  return withItem(session, alignmentId, (item) => {
    const changed = item.decision.state !== state;
    const polishedSpans =
      state === null || state === "ABSENT" ? [] : item.decision.polished_spans;
    return {
      ...item,
      decision: {
        ...item.decision,
        state,
        polished_spans: polishedSpans,
        complete: changed ? false : item.decision.complete,
        completed_at: changed ? null : item.decision.completed_at,
      },
    };
  });
}

/** Changing the reviewer invalidates an existing explicit completion. */
export function setItemReviewer(
  session: AlignmentSession,
  alignmentId: string,
  reviewer: string,
): AlignmentSession {
  if (typeof reviewer !== "string") {
    throw new Error("reviewer must be a string");
  }
  return withItem(session, alignmentId, (item) => {
    const changed = reviewer !== item.decision.reviewer;
    return {
      ...item,
      decision: {
        ...item.decision,
        reviewer,
        complete: changed ? false : item.decision.complete,
        completed_at: changed ? null : item.decision.completed_at,
      },
    };
  });
}

/** Notes are annotations; any change to a completed item retracts completion. */
export function setItemNotes(
  session: AlignmentSession,
  alignmentId: string,
  notes: string[],
): AlignmentSession {
  if (!Array.isArray(notes) || notes.some((note) => typeof note !== "string")) {
    throw new Error("alignment item notes must be an array of strings");
  }
  return withItem(session, alignmentId, (item) => {
    const changed = !arraysEqual(notes, item.decision.notes);
    return {
      ...item,
      decision: {
        ...item.decision,
        notes: [...notes],
        complete: changed ? false : item.decision.complete,
        completed_at: changed ? null : item.decision.completed_at,
      },
    };
  });
}

/** Convenience for long sessions: apply one reviewer name to every item. */
export function setAllReviewers(
  session: AlignmentSession,
  reviewer: string,
): AlignmentSession {
  if (typeof reviewer !== "string") {
    throw new Error("reviewer must be a string");
  }
  return {
    ...session,
    items: session.items.map((item) => {
      const changed = item.decision.reviewer !== reviewer;
      return {
        ...item,
        decision: {
          ...item.decision,
          reviewer,
          complete: changed ? false : item.decision.complete,
          completed_at: changed ? null : item.decision.completed_at,
        },
      };
    }),
  };
}

/** Add one exact polished span; keeps the span list unique and sorted. */
export function addSpan(
  session: AlignmentSession,
  alignmentId: string,
  span: { start: number; end: number; text: string },
): AlignmentSession {
  const item = findItem(session, alignmentId);
  const state = item.decision.state;
  if (state === null) {
    throw new Error("set an alignment state before adding polished spans");
  }
  if (state === "ABSENT") {
    throw new Error("ABSENT decisions cannot carry polished spans");
  }
  const validated = validatePolishedSpan(
    span,
    item.representation.polished_text,
    `${alignmentId} polished span`,
  );
  const duplicate = item.decision.polished_spans.some(
    (existing) => existing.start === validated.start && existing.end === validated.end,
  );
  if (duplicate) {
    throw new Error(`${alignmentId} already has a span at ${validated.start}:${validated.end}`);
  }
  const polishedSpans = [...item.decision.polished_spans, validated].sort(
    (a, b) => a.start - b.start || a.end - b.end,
  );
  return withItem(session, alignmentId, (current) => ({
    ...current,
    decision: {
      ...current.decision,
      polished_spans: polishedSpans,
      complete: false,
      completed_at: null,
    },
  }));
}

/** Remove one span by list index; editing a completed item uncompletes it. */
export function removeSpan(
  session: AlignmentSession,
  alignmentId: string,
  index: number,
): AlignmentSession {
  const item = findItem(session, alignmentId);
  if (!Number.isInteger(index) || index < 0 || index >= item.decision.polished_spans.length) {
    throw new Error(`${alignmentId} span index ${index} is out of range`);
  }
  return withItem(session, alignmentId, (current) => ({
    ...current,
    decision: {
      ...current.decision,
      polished_spans: current.decision.polished_spans.filter(
        (_, spanIndex) => spanIndex !== index,
      ),
      complete: false,
      completed_at: null,
    },
  }));
}

function crossTargetSpanConflicts(
  session: AlignmentSession,
  targetItem: SessionItem,
): string[] {
  const state = targetItem.decision.state;
  if (state !== "ALIGNED" && state !== "MULTIPLE_CANDIDATES") {
    return [];
  }
  const conflicts: string[] = [];
  for (const candidate of session.items) {
    if (candidate.alignment_id === targetItem.alignment_id || !candidate.decision.complete) {
      continue;
    }
    const candidateState = candidate.decision.state;
    if (
      candidate.window_id !== targetItem.window_id ||
      (candidateState !== "ALIGNED" && candidateState !== "MULTIPLE_CANDIDATES")
    ) {
      continue;
    }
    for (const span of targetItem.decision.polished_spans) {
      const conflict = candidate.decision.polished_spans.some(
        (other) => other.start === span.start && other.end === span.end,
      );
      if (conflict) {
        conflicts.push(
          `span ${span.start}:${span.end} in ${targetItem.window_id} is already assigned to ${candidate.alignment_id}`,
        );
      }
    }
  }
  return [...new Set(conflicts)];
}

/**
 * Explicit completion transition: generates completed_at only here, and only
 * when the state, reviewer, span cardinality, and cross-target span
 * uniqueness all hold.
 */
export function completeItem(
  session: AlignmentSession,
  alignmentId: string,
  completedAt: string,
): CompletionResult {
  const item = findItem(session, alignmentId);
  const missing = itemMissingFields(item);
  if (missing.length > 0) {
    return {
      ok: false,
      errors: missing.map((field) => `missing ${field}`),
    };
  }
  if (typeof completedAt !== "string" || !ISO_TIMESTAMP.test(completedAt)) {
    return {
      ok: false,
      errors: ["completed_at must be a non-empty ISO timestamp"],
    };
  }
  const conflicts = crossTargetSpanConflicts(session, item);
  if (conflicts.length > 0) {
    return { ok: false, errors: conflicts };
  }
  return {
    ok: true,
    session: withItem(session, alignmentId, (current) => ({
      ...current,
      decision: {
        ...current.decision,
        complete: true,
        completed_at: completedAt,
      },
    })),
  };
}

/** Retract an explicit completion; clears the generated timestamp. */
export function uncompleteItem(
  session: AlignmentSession,
  alignmentId: string,
): AlignmentSession {
  return withItem(session, alignmentId, (item) => ({
    ...item,
    decision: {
      ...item.decision,
      complete: false,
      completed_at: null,
    },
  }));
}

export function summarizeProgress(session: AlignmentSession): ProgressSummary {
  const summary: ProgressSummary = {
    total: session.items.length,
    complete: 0,
    ready: 0,
    in_progress: 0,
    untouched: 0,
  };
  for (const item of session.items) {
    if (item.decision.complete) {
      summary.complete += 1;
      continue;
    }
    const missing = itemMissingFields(item);
    if (missing.length === 0) {
      summary.ready += 1;
      continue;
    }
    const decision = item.decision;
    const hasEdits =
      decision.state !== null ||
      decision.reviewer.trim() !== "" ||
      decision.notes.length > 0 ||
      decision.polished_spans.length > 0;
    if (hasEdits) {
      summary.in_progress += 1;
    } else {
      summary.untouched += 1;
    }
  }
  return summary;
}

/** Errors blocking the final decisions export; empty means exportable. */
export function decisionsMapErrors(session: AlignmentSession): string[] {
  const errors: string[] = [];
  if (session.items.length === 0) {
    return ["session has no alignment items"];
  }
  for (const item of session.items) {
    const decision = item.decision;
    if (!decision.complete) {
      errors.push(`${item.alignment_id} is not marked complete`);
      continue;
    }
    if (decision.state === null) {
      errors.push(`${item.alignment_id} has no alignment state`);
    } else if (!(ALIGNMENT_DECISION_STATES as readonly string[]).includes(decision.state)) {
      errors.push(`${item.alignment_id} has an invalid alignment state`);
    }
    if (decision.reviewer.trim() === "") {
      errors.push(`${item.alignment_id} has no reviewer`);
    }
    if (
      typeof decision.completed_at !== "string" ||
      !ISO_TIMESTAMP.test(decision.completed_at)
    ) {
      errors.push(`${item.alignment_id} has no valid completed_at`);
    }
    if (decision.notes.some((note) => typeof note !== "string")) {
      errors.push(`${item.alignment_id} notes must be strings`);
    }
    decision.polished_spans.forEach((span, index) => {
      try {
        validatePolishedSpan(
          span,
          item.representation.polished_text,
          `${item.alignment_id} polished_spans[${index}]`,
        );
      } catch (error) {
        errors.push(error instanceof Error ? error.message : `${item.alignment_id} has an invalid span`);
      }
    });
    try {
      validateSortedUniqueSpans(decision.polished_spans, item.alignment_id);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : `${item.alignment_id} spans are invalid`);
    }
    if (decision.state !== null) {
      const cardinality = decisionCardinalityError(
        decision.state,
        decision.polished_spans.length,
      );
      if (cardinality !== null) {
        errors.push(`${item.alignment_id} ${cardinality}`);
      }
    }
  }

  const assigned = new Map<string, string>();
  for (const item of session.items) {
    if (!item.decision.complete) {
      continue;
    }
    const state = item.decision.state;
    if (state !== "ALIGNED" && state !== "MULTIPLE_CANDIDATES") {
      continue;
    }
    for (const span of item.decision.polished_spans) {
      const key = `${item.window_id}:${span.start}:${span.end}`;
      const previous = assigned.get(key);
      if (previous !== undefined && previous !== item.alignment_id) {
        errors.push(
          `cross-target duplicate polished span ${span.start}:${span.end} in ${item.window_id} is assigned to both ${previous} and ${item.alignment_id}`,
        );
      } else {
        assigned.set(key, item.alignment_id);
      }
    }
  }
  return errors;
}

/**
 * Compact decisions map in the exact finalizer shape, keyed by alignment_id
 * in deterministic packet order.  Refuses (throws) unless every item is
 * explicitly complete and globally valid.
 */
export function buildDecisionsMap(session: AlignmentSession): DecisionsMap {
  const errors = decisionsMapErrors(session);
  if (errors.length > 0) {
    throw new Error(
      `alignment decisions require every item complete and valid: ${errors.slice(0, 3).join("; ")}`,
    );
  }
  const map: DecisionsMap = {};
  for (const item of session.items) {
    map[item.alignment_id] = {
      state: item.decision.state as DecisionState,
      polished_spans: item.decision.polished_spans.map((span) => ({ ...span })),
      reviewer: item.decision.reviewer,
      completed_at: item.decision.completed_at as string,
      notes: [...item.decision.notes],
    };
  }
  return map;
}

/** Session backup export; the only nondeterministic field is the explicit timestamp. */
export function buildSessionExport(
  session: AlignmentSession,
  exportedAt: string,
): AlignmentSession {
  return { ...session, exported_at: exportedAt };
}

/** All exact occurrences of a needle, including overlapping ones, sorted by start. */
export function findExactOccurrences(text: string, needle: string): PolishedSpan[] {
  if (needle === "") {
    return [];
  }
  const spans: PolishedSpan[] = [];
  let index = text.indexOf(needle);
  while (index !== -1) {
    spans.push({ start: index, end: index + needle.length, text: needle });
    index = text.indexOf(needle, index + 1);
  }
  return spans;
}

/**
 * Full backup-import validation: session schema versions, exact packet-hash
 * binding, immutable packet item content, span exactness/ordering, decision
 * cardinality at completion, and cross-target span uniqueness.  No timestamp
 * is fabricated: completed_at is only accepted when the item is explicitly
 * marked complete.
 */
export function validateSessionInput(
  input: unknown,
  packet: AlignmentPacket,
): ValidationResult {
  const errors: string[] = [];
  if (!isRecordObject(input)) {
    return { ok: false, errors: ["session must be a JSON object"] };
  }
  if (input.schema_version !== SESSION_SCHEMA_VERSION) {
    errors.push(`schema_version must be ${SESSION_SCHEMA_VERSION}`);
  }
  if (input.packet_schema_version !== PACKET_SCHEMA_VERSION) {
    errors.push(`packet_schema_version must be ${PACKET_SCHEMA_VERSION}`);
  }
  if (input.packet_sha256 !== packet.content_sha256) {
    errors.push("packet_sha256 does not match the loaded packet content hash");
  }
  if (input.exported_at !== null && typeof input.exported_at !== "string") {
    errors.push("exported_at must be null or a string");
  }
  const leaks = findForbiddenLeaks(input);
  if (leaks.length > 0) {
    errors.push(...leaks.map((problem) => `session ${problem}`));
  }
  if (!Array.isArray(input.items)) {
    errors.push("items must be an array");
    return { ok: false, errors };
  }
  if (input.items.length !== packet.items.length) {
    errors.push(
      `items must contain exactly ${packet.items.length} entries (found ${input.items.length})`,
    );
  }

  const items: SessionItem[] = [];
  input.items.forEach((rawItem, index) => {
    const reference = packet.items[index];
    if (!reference) {
      return;
    }
    const path = `items[${index}]`;
    if (!isRecordObject(rawItem)) {
      errors.push(`${path} must be an object`);
      return;
    }
    if (!hasExactKeys(rawItem, ITEM_KEYS)) {
      errors.push(`${path} keys are invalid`);
    }
    if (rawItem.alignment_id !== reference.alignment_id) {
      errors.push(`${path}.alignment_id does not match the loaded packet`);
    }
    if (rawItem.window_id !== reference.window_id) {
      errors.push(`${path}.window_id does not match the loaded packet`);
    }
    if (rawItem.endpoint_id !== reference.endpoint_id) {
      errors.push(`${path}.endpoint_id does not match the loaded packet`);
    }
    if (rawItem.node_type !== reference.node_type) {
      errors.push(`${path}.node_type does not match the loaded packet`);
    }
    if (!deepEqual(rawItem.bronze_target, reference.bronze_target)) {
      errors.push(`${path}.bronze_target does not match the loaded packet`);
    }
    if (!deepEqual(rawItem.representation, reference.representation)) {
      errors.push(`${path}.representation does not match the loaded packet`);
    }

    const rawDecision = rawItem.decision;
    const decision: SessionDecision = {
      state: null,
      polished_spans: [],
      reviewer: "",
      completed_at: null,
      notes: [],
      complete: false,
    };
    if (!isRecordObject(rawDecision) || !hasExactKeys(rawDecision, SESSION_DECISION_KEYS)) {
      errors.push(`${path}.decision keys are invalid`);
      items.push({
        alignment_id:
          typeof rawItem.alignment_id === "string"
            ? rawItem.alignment_id
            : reference.alignment_id,
        window_id:
          typeof rawItem.window_id === "string" ? rawItem.window_id : reference.window_id,
        endpoint_id:
          typeof rawItem.endpoint_id === "string"
            ? rawItem.endpoint_id
            : reference.endpoint_id,
        node_type: reference.node_type,
        bronze_target: reference.bronze_target,
        representation: reference.representation,
        decision,
      });
      return;
    }

    const state = rawDecision.state;
    if (state !== null && !(ALIGNMENT_DECISION_STATES as readonly string[]).includes(state as string)) {
      errors.push(`${path}.decision.state is invalid`);
    } else {
      decision.state = state as DecisionState | null;
    }

    if (Array.isArray(rawDecision.polished_spans)) {
      rawDecision.polished_spans.forEach((rawSpan, spanIndex) => {
        try {
          decision.polished_spans.push(
            validatePolishedSpan(
              rawSpan,
              reference.representation.polished_text,
              `${path}.decision.polished_spans[${spanIndex}]`,
            ),
          );
        } catch (error) {
          errors.push(
            error instanceof Error ? error.message : `${path}.decision span is invalid`,
          );
        }
      });
    } else {
      errors.push(`${path}.decision.polished_spans must be an array`);
    }
    try {
      validateSortedUniqueSpans(decision.polished_spans, `${path}.decision`);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : `${path}.decision spans are invalid`);
    }

    if (typeof rawDecision.reviewer === "string") {
      decision.reviewer = rawDecision.reviewer;
    } else {
      errors.push(`${path}.decision.reviewer must be a string`);
    }
    if (
      rawDecision.completed_at === null ||
      (typeof rawDecision.completed_at === "string" && ISO_TIMESTAMP.test(rawDecision.completed_at))
    ) {
      decision.completed_at =
        typeof rawDecision.completed_at === "string" ? rawDecision.completed_at : null;
    } else {
      errors.push(`${path}.decision.completed_at must be null or an ISO timestamp`);
    }
    if (
      Array.isArray(rawDecision.notes) &&
      rawDecision.notes.every((note: unknown) => typeof note === "string")
    ) {
      decision.notes = [...(rawDecision.notes as string[])];
    } else {
      errors.push(`${path}.decision.notes must be an array of strings`);
    }
    if (typeof rawDecision.complete === "boolean") {
      decision.complete = rawDecision.complete;
    } else {
      errors.push(`${path}.decision.complete must be a boolean`);
    }

    const stateSet = decision.state !== null;
    const reviewerPresent = decision.reviewer.trim() !== "";
    if (decision.complete) {
      if (decision.completed_at === null) {
        errors.push(`${path} completed_at is required once the item is marked complete`);
      }
      if (!stateSet) {
        errors.push(`${path} completion requires an alignment state`);
      }
      if (!reviewerPresent) {
        errors.push(`${path} completion requires a reviewer`);
      }
      if (decision.state !== null) {
        const cardinality = decisionCardinalityError(
          decision.state,
          decision.polished_spans.length,
        );
        if (cardinality !== null) {
          errors.push(`${path} ${cardinality}`);
        }
      }
    } else if (decision.completed_at !== null) {
      errors.push(
        `${path} completed_at must be null until the item is explicitly marked complete`,
      );
    }
    if (decision.state === null && decision.polished_spans.length > 0) {
      errors.push(`${path} spans require a non-null alignment state`);
    }
    if (decision.state === "ABSENT" && decision.polished_spans.length > 0) {
      errors.push(`${path} ABSENT decisions cannot carry polished spans`);
    }

    items.push({
      alignment_id:
        typeof rawItem.alignment_id === "string"
          ? rawItem.alignment_id
          : reference.alignment_id,
      window_id:
        typeof rawItem.window_id === "string" ? rawItem.window_id : reference.window_id,
      endpoint_id:
        typeof rawItem.endpoint_id === "string"
          ? rawItem.endpoint_id
          : reference.endpoint_id,
      node_type: reference.node_type,
      bronze_target: reference.bronze_target,
      representation: reference.representation,
      decision,
    });
  });

  const assigned = new Map<string, string>();
  for (const item of items) {
    if (!item.decision.complete) {
      continue;
    }
    const state = item.decision.state;
    if (state !== "ALIGNED" && state !== "MULTIPLE_CANDIDATES") {
      continue;
    }
    for (const span of item.decision.polished_spans) {
      const key = `${item.window_id}:${span.start}:${span.end}`;
      const previous = assigned.get(key);
      if (previous !== undefined && previous !== item.alignment_id) {
        errors.push(
          `cross-target duplicate polished span ${span.start}:${span.end} in ${item.window_id} is assigned to both ${previous} and ${item.alignment_id}`,
        );
      } else {
        assigned.set(key, item.alignment_id);
      }
    }
  }

  if (errors.length > 0) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    session: {
      schema_version: SESSION_SCHEMA_VERSION,
      packet_schema_version: PACKET_SCHEMA_VERSION,
      packet_sha256: packet.content_sha256,
      exported_at: typeof input.exported_at === "string" ? input.exported_at : null,
      items,
    },
  };
}
