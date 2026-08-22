/**
 * Pure Phase 2K blinded human semantic-recoverability review utilities.
 *
 * This module intentionally contains no Node or browser APIs so it can be
 * unit-tested under Jest and imported safely by the client bundle.
 *
 * Data boundary: the only accepted input is the official blank
 * "phase2k-human-review-packet-v2" JSON.  Sanitization drops every field the
 * reviewer does not need (including the packet's `blinding` envelope, which
 * names the separately retained mapping artifact), and a conservative
 * recursive scan rejects condition/radius/stage/model/scorer/mapping
 * structure anywhere in the reviewer-facing payload.  The mapping file
 * ("phase2k-human-review-mapping-v2.json") is never loaded, and no score,
 * reviewer identity, or timestamp is ever fabricated.
 */

export const PACKET_SCHEMA_VERSION = "phase2k-human-review-packet-v2";
export const PRESENTATION_SCHEMA_VERSION = "phase2k-review-presentation-v2";
export const SESSION_SCHEMA_VERSION = "phase2k-review-session-v1";

export const SCORE_FIELDS = [
  "coached_actor",
  "opponent_entity",
  "pronouns",
  "ability_ownership",
  "core_action",
  "condition",
  "consequence",
  "causality",
  "standalone_coaching_claim",
  "asr_repair_correctness",
  "entity_binding_correctness",
  "meaning_preservation",
  "unsupported_invention",
  "remaining_ambiguity",
] as const;

export const SCORE_MIN = 0;
export const SCORE_MAX = 5;
export const NOT_APPLICABLE = "NOT_APPLICABLE";

export type ScoreField = (typeof SCORE_FIELDS)[number];
export type ScoreValue = number | typeof NOT_APPLICABLE;
export type RubricDirection = "higher_is_better" | "lower_is_better";

export type RubricEntry = {
  description: string;
  direction: RubricDirection;
  not_applicable_allowed: boolean;
};

export type Rubric = Record<ScoreField, RubricEntry>;

export type PresentationSection = {
  id: "primary" | "supplement";
  text: string;
};

export type Presentation = {
  schema_version: typeof PRESENTATION_SCHEMA_VERSION;
  target_sha256: string;
  displayed_target_sha256?: string;
  sections: PresentationSection[];
};

/** Exactly one sanitized, strictly-blank packet item. */
export type PacketItem = {
  review_item_id: string;
  window_id: string;
  blinded_label: string;
  presentation: Presentation;
  content_sha256: string;
  scores: Record<ScoreField, null>;
  reviewer: null;
  completed_at: null;
  notes: [];
};

/**
 * Sanitized reviewer-facing packet.  `blinding` (and therefore the mapping
 * artifact name/hash) is deliberately absent from this payload.
 */
export type ReviewPacket = {
  schema_version: typeof PACKET_SCHEMA_VERSION;
  purpose: string;
  release_gate: "AWAITING_HUMAN_REVIEW";
  review_items: PacketItem[];
  scoring_fields: readonly ScoreField[];
  score_range: { min: typeof SCORE_MIN; max: typeof SCORE_MAX };
  rubric: Rubric;
  content_sha256: string;
};

export type SessionScore = Record<ScoreField, ScoreValue | null>;

export type SessionItem = {
  review_item_id: string;
  window_id: string;
  blinded_label: string;
  presentation: Presentation;
  content_sha256: string;
  scores: SessionScore;
  reviewer: string;
  completed_at: string | null;
  notes: string[];
  complete: boolean;
};

export type ReviewSession = {
  schema_version: typeof SESSION_SCHEMA_VERSION;
  packet_schema_version: typeof PACKET_SCHEMA_VERSION;
  packet_sha256: string;
  exported_at: string | null;
  items: SessionItem[];
};

export type ProgressSummary = {
  total: number;
  complete: number;
  /** All 14 scores and a reviewer are present, but not explicitly marked complete. */
  ready: number;
  /** Has at least one review edit but is not ready to complete. */
  in_progress: number;
  untouched: number;
};

export type ValidationResult =
  | { ok: true; session: ReviewSession }
  | { ok: false; errors: string[] };

export type CompletionResult =
  | { ok: true; session: ReviewSession }
  | { ok: false; errors: string[] };

export type ReviewsMapEntry = {
  scores: Record<ScoreField, ScoreValue>;
  reviewer: string;
  completed_at: string;
  notes: string[];
};

/** Plain object keyed by review_item_id in deterministic packet order. */
export type ReviewsMap = Record<string, ReviewsMapEntry>;

const HEX64 = /^[0-9a-f]{64}$/;
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/;
const UNBLINDED_ID_PATTERN = /p2k:(rec|radius):/;

const PACKET_TOP_LEVEL_KEYS = [
  "schema_version",
  "purpose",
  "release_gate",
  "blinding",
  "review_items",
  "scoring_fields",
  "score_range",
  "rubric",
  "content_sha256",
] as const;

const PACKET_ITEM_KEYS = [
  "review_item_id",
  "window_id",
  "blinded_label",
  "presentation",
  "content_sha256",
  "scores",
  "reviewer",
  "completed_at",
  "notes",
] as const;

const SESSION_ITEM_KEYS = [
  ...PACKET_ITEM_KEYS,
  "complete",
] as const;

const PRESENTATION_KEYS_WITH_DISPLAYED = [
  "schema_version",
  "target_sha256",
  "displayed_target_sha256",
  "sections",
] as const;

const PRESENTATION_KEYS_BASE = [
  "schema_version",
  "target_sha256",
  "sections",
] as const;

const BLINDING_KEYS = [
  "method",
  "seed",
  "mapping_file",
  "mapping_sha256",
] as const;

/**
 * Structural keys that must never appear in the reviewer-facing payload or
 * session.  Deliberately excludes the official blank-packet fields
 * (`scores`, `scoring_fields`, `score_range`, and the 14 score names), which
 * are legitimate reviewer-facing content.
 */
const FORBIDDEN_STRUCTURAL_KEYS = new Set([
  // condition/radius/stage provenance
  "condition_code",
  "window_condition",
  "record_type",
  "radius",
  "radius_label",
  "radius_entry",
  "stage",
  "stage_label",
  "semantic_stage",
  // unblinded record/entry identity and source coordinates
  "record_id",
  "entry_id",
  "record_sha256",
  "entry_sha256",
  "source_absolute_start",
  "source_absolute_end",
  "context_id",
  "kind",
  "candidate",
  "candidates",
  "candidate_catalog",
  // provenance / mapping artifacts (the mapping file is never loaded)
  "provenance",
  "mapping",
  "mapping_file",
  "mapping_sha256",
  "labels",
  "label_map",
  "condition_map",
  "generation_status",
  "clean_target_transcript",
  "resolved_semantic_paraphrase",
  "paraphrase_text",
  "mechanical_clean",
  "raw_bronze",
  "enlarged_context",
  "target_only",
  "reconstruction",
  // model / scorer / prediction / downstream result material
  "model",
  "model_id",
  "model_name",
  "model_data",
  "model_version",
  "model_prediction",
  "model_predictions",
  "model_score",
  "model_scores",
  "model_result",
  "results",
  "result",
  "scorer",
  "prediction",
  "predictions",
  "predicted",
  "predicted_label",
  "probability",
  "probabilities",
  "logits",
  "confidence",
  // the blinding envelope must not reach the client
  "blinding",
]);

/**
 * Exact structural values that would reveal a condition/radius.  Free-text
 * prose (`text`, `notes`, `description`, `purpose`) is never value-scanned.
 */
const FORBIDDEN_STRUCTURAL_VALUES = new Set([
  "A",
  "B",
  "C",
  "D",
  "RADIUS_ENTRY",
  "raw_bronze",
  "mechanical_clean",
  "enlarged_context",
  "reconstruction",
  "target_only",
  "NOT_GENERATED",
  "GENERATED",
  "r1",
  "r2",
  "r3",
  "r5",
  "r10",
  "bounded_local_episode",
]);

/** Keys whose values are reviewer-facing prose and must not be word-scanned. */
const FREE_TEXT_KEYS = new Set(["text", "notes", "description", "purpose"]);

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => actual.includes(key));
}

function hasSameKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && expected.every((key) => actual.includes(key));
}

function arraysEqual(left: readonly unknown[], right: readonly unknown[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function deepEqual(left: unknown, right: unknown): boolean {
  if (left === right) {
    return true;
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    return (
      left.length === right.length && left.every((value, index) => deepEqual(value, right[index]))
    );
  }
  if (isRecordObject(left) && isRecordObject(right)) {
    const leftKeys = Object.keys(left);
    const rightKeys = Object.keys(right);
    return (
      leftKeys.length === rightKeys.length &&
      leftKeys.every((key) => Object.prototype.hasOwnProperty.call(right, key)) &&
      leftKeys.every((key) => deepEqual(left[key], right[key]))
    );
  }
  return false;
}

function requireNonEmptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function requireHex64(value: unknown, label: string): string {
  if (typeof value !== "string" || !HEX64.test(value)) {
    throw new Error(`${label} must be a 64-character hex string`);
  }
  return value;
}

/**
 * Recursive conservative structural-leak scan.  Keys are always scanned;
 * string values are scanned only outside free-text prose keys.  Returns
 * human-readable paths; an empty result means clean.
 */
export function findForbiddenLeaks(
  value: unknown,
  path = "",
): string[] {
  const problems: string[] = [];
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      problems.push(...findForbiddenLeaks(item, `${path}[${index}]`));
    });
    return problems;
  }
  if (!isRecordObject(value)) {
    if (typeof value === "string" && FORBIDDEN_STRUCTURAL_VALUES.has(value)) {
      problems.push(`forbidden value ${JSON.stringify(value)} at ${path || "root"}`);
    } else if (typeof value === "string" && UNBLINDED_ID_PATTERN.test(value)) {
      problems.push(`unblinded record identity at ${path || "root"}`);
    }
    return problems;
  }
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_STRUCTURAL_KEYS.has(key.toLowerCase())) {
      problems.push(`forbidden field "${key}" at ${path || "root"}`);
    }
    const childPath = path ? `${path}.${key}` : key;
    if (FREE_TEXT_KEYS.has(key)) {
      // Reviewer-facing prose is never value-scanned; only the key is checked.
      continue;
    }
    if (typeof child === "string") {
      if (FORBIDDEN_STRUCTURAL_VALUES.has(child)) {
        problems.push(`forbidden value ${JSON.stringify(child)} at ${childPath}`);
      } else if (UNBLINDED_ID_PATTERN.test(child)) {
        problems.push(`unblinded record identity at ${childPath}`);
      }
    } else {
      problems.push(...findForbiddenLeaks(child, childPath));
    }
  }
  return problems;
}

function sanitizePresentation(raw: unknown): Presentation {
  if (!isRecordObject(raw)) {
    throw new Error("phase2k review item presentation must be an object");
  }
  const hasDisplayed = Object.prototype.hasOwnProperty.call(raw, "displayed_target_sha256");
  const expectedKeys = hasDisplayed ? PRESENTATION_KEYS_WITH_DISPLAYED : PRESENTATION_KEYS_BASE;
  if (!hasExactKeys(raw, expectedKeys)) {
    throw new Error("phase2k review item presentation keys are invalid");
  }
  if (raw.schema_version !== PRESENTATION_SCHEMA_VERSION) {
    throw new Error(
      `phase2k review presentation schema_version must be ${PRESENTATION_SCHEMA_VERSION}`,
    );
  }
  const targetSha256 = requireHex64(raw.target_sha256, "presentation target_sha256");
  const displayedTargetSha256 = hasDisplayed
    ? requireHex64(raw.displayed_target_sha256, "presentation displayed_target_sha256")
    : undefined;
  if (!Array.isArray(raw.sections) || raw.sections.length === 0) {
    throw new Error("phase2k review item presentation requires non-empty sections");
  }
  const seenSectionIds = new Set<string>();
  let hasPrimary = false;
  const sections = raw.sections.map((rawSection, index) => {
    if (!isRecordObject(rawSection) || !hasExactKeys(rawSection, ["id", "text"])) {
      throw new Error(`phase2k presentation section ${index} must be {id, text}`);
    }
    const sectionId = requireNonEmptyString(
      rawSection.id,
      `presentation section ${index} id`,
    ) as "primary" | "supplement";
    if (sectionId !== "primary" && sectionId !== "supplement") {
      throw new Error(`presentation section ${index} id is not neutral`);
    }
    if (seenSectionIds.has(sectionId)) {
      throw new Error(`presentation section ids must be unique (duplicate ${sectionId})`);
    }
    seenSectionIds.add(sectionId);
    if (sectionId === "primary") {
      hasPrimary = true;
    }
    const text = requireNonEmptyString(rawSection.text, `presentation section ${index} text`);
    return { id: sectionId, text };
  });
  if (!hasPrimary) {
    throw new Error("phase2k review item presentation requires primary content");
  }
  return {
    schema_version: PRESENTATION_SCHEMA_VERSION,
    target_sha256: targetSha256,
    ...(displayedTargetSha256 !== undefined
      ? { displayed_target_sha256: displayedTargetSha256 }
      : {}),
    sections,
  };
}

function sanitizeItem(
  raw: unknown,
  index: number,
  seenIds: Set<string>,
  seenLabels: Set<string>,
): PacketItem {
  if (!isRecordObject(raw)) {
    throw new Error(`phase2k review item ${index} must be an object`);
  }
  if (!hasExactKeys(raw, PACKET_ITEM_KEYS)) {
    throw new Error(`phase2k review item ${index} keys are invalid`);
  }
  const reviewItemId = requireNonEmptyString(raw.review_item_id, `review item ${index} id`);
  if (reviewItemId.includes("rec:") || reviewItemId.includes("radius:")) {
    throw new Error(`review item ${index} id encodes unblinded identity`);
  }
  if (seenIds.has(reviewItemId)) {
    throw new Error(`phase2k review item ids must be unique (duplicate ${reviewItemId})`);
  }
  seenIds.add(reviewItemId);
  const blindedLabel = requireNonEmptyString(
    raw.blinded_label,
    `review item ${index} blinded_label`,
  );
  if (seenLabels.has(blindedLabel)) {
    throw new Error(`phase2k review item labels must be unique (duplicate ${blindedLabel})`);
  }
  seenLabels.add(blindedLabel);
  const windowId = requireNonEmptyString(raw.window_id, `review item ${index} window_id`);
  const presentation = sanitizePresentation(raw.presentation);
  const contentSha256 = requireHex64(raw.content_sha256, `review item ${index} content_sha256`);
  if (!isRecordObject(raw.scores) || !hasSameKeys(raw.scores, SCORE_FIELDS)) {
    throw new Error(`review item ${index} scores must contain exactly the official score fields`);
  }
  for (const field of SCORE_FIELDS) {
    if (raw.scores[field] !== null) {
      throw new Error(`official human review packet must remain blank at ${reviewItemId}.${field}`);
    }
  }
  if (raw.reviewer !== null) {
    throw new Error(`blank human review packet cannot be signed at ${reviewItemId}`);
  }
  if (raw.completed_at !== null) {
    throw new Error(`blank human review packet cannot carry a timestamp at ${reviewItemId}`);
  }
  if (!Array.isArray(raw.notes) || raw.notes.length > 0) {
    throw new Error(`blank human review packet notes must be empty at ${reviewItemId}`);
  }
  const scores = Object.fromEntries(SCORE_FIELDS.map((field) => [field, null])) as Record<
    ScoreField,
    null
  >;
  return {
    review_item_id: reviewItemId,
    window_id: windowId,
    blinded_label: blindedLabel,
    presentation,
    content_sha256: contentSha256,
    scores,
    reviewer: null,
    completed_at: null,
    notes: [],
  };
}

/**
 * Strict sanitizer for the official blank packet.  Validates schema/version,
 * hash shapes, score/rubric key sets, blank review fields, unique opaque
 * ids/labels, neutral presentations, and forbidden structural leakage, then
 * returns only the reviewer-facing payload (the `blinding` envelope is
 * dropped before the leak scan and never reaches the client).
 */
export function sanitizePacket(raw: unknown): ReviewPacket {
  if (!isRecordObject(raw)) {
    throw new Error("phase2k human-review packet must be a JSON object");
  }
  // Scan a copy with the blinding envelope removed: the official packet may
  // name the mapping artifact inside `blinding`, but nothing else may.  The
  // scan runs first so forbidden material is never masked by key-set errors.
  const scanTarget = withoutKey(raw, "blinding");
  const leaks = findForbiddenLeaks(scanTarget);
  if (leaks.length > 0) {
    throw new Error(`phase2k packet contains forbidden structural material: ${leaks.join("; ")}`);
  }
  if (!hasExactKeys(raw, PACKET_TOP_LEVEL_KEYS)) {
    throw new Error("phase2k human-review packet top-level keys are invalid");
  }
  if (raw.schema_version !== PACKET_SCHEMA_VERSION) {
    throw new Error(`phase2k packet schema_version must be ${PACKET_SCHEMA_VERSION}`);
  }
  const purpose = requireNonEmptyString(raw.purpose, "phase2k packet purpose");
  if (raw.release_gate !== "AWAITING_HUMAN_REVIEW") {
    throw new Error("phase2k packet release_gate must be AWAITING_HUMAN_REVIEW");
  }
  if (!isRecordObject(raw.blinding) || !hasExactKeys(raw.blinding, BLINDING_KEYS)) {
    throw new Error("phase2k packet blinding envelope is invalid");
  }
  requireNonEmptyString(raw.blinding.method, "phase2k blinding method");
  requireNonEmptyString(raw.blinding.seed, "phase2k blinding seed");
  requireNonEmptyString(raw.blinding.mapping_file, "phase2k blinding mapping_file");
  requireHex64(raw.blinding.mapping_sha256, "phase2k blinding mapping_sha256");
  if (
    !Array.isArray(raw.scoring_fields) ||
    !arraysEqual(raw.scoring_fields, SCORE_FIELDS)
  ) {
    throw new Error("phase2k packet scoring_fields are invalid");
  }
  if (
    !isRecordObject(raw.score_range) ||
    !hasExactKeys(raw.score_range, ["min", "max"]) ||
    raw.score_range.min !== SCORE_MIN ||
    raw.score_range.max !== SCORE_MAX
  ) {
    throw new Error("phase2k packet score_range must be {min: 0, max: 5}");
  }
  if (!isRecordObject(raw.rubric) || !hasSameKeys(raw.rubric, SCORE_FIELDS)) {
    throw new Error("phase2k packet rubric is incomplete");
  }
  const rubric = {} as Rubric;
  for (const field of SCORE_FIELDS) {
    const entry = raw.rubric[field];
    if (!isRecordObject(entry) || !hasExactKeys(entry, ["description", "direction", "not_applicable_allowed"])) {
      throw new Error(`phase2k rubric entry ${field} is invalid`);
    }
    const description = requireNonEmptyString(
      entry.description,
      `phase2k rubric ${field} description`,
    );
    if (entry.direction !== "higher_is_better" && entry.direction !== "lower_is_better") {
      throw new Error(`phase2k rubric ${field} direction is invalid`);
    }
    if (typeof entry.not_applicable_allowed !== "boolean") {
      throw new Error(`phase2k rubric ${field} N/A flag is invalid`);
    }
    rubric[field] = {
      description,
      direction: entry.direction as RubricDirection,
      not_applicable_allowed: entry.not_applicable_allowed as boolean,
    };
  }
  const contentSha256 = requireHex64(raw.content_sha256, "phase2k packet content_sha256");
  if (!Array.isArray(raw.review_items) || raw.review_items.length === 0) {
    throw new Error("phase2k packet review_items must be a non-empty array");
  }
  const seenIds = new Set<string>();
  const seenLabels = new Set<string>();
  const reviewItems = raw.review_items.map((item, index) =>
    sanitizeItem(item, index, seenIds, seenLabels),
  );
  return {
    schema_version: PACKET_SCHEMA_VERSION,
    purpose,
    release_gate: "AWAITING_HUMAN_REVIEW",
    review_items: reviewItems,
    scoring_fields: SCORE_FIELDS,
    score_range: { min: SCORE_MIN, max: SCORE_MAX },
    rubric,
    content_sha256: contentSha256,
  };
}

function withoutKey(value: unknown, keyToRemove: string): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => withoutKey(item, keyToRemove));
  }
  if (isRecordObject(value)) {
    const copy: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value)) {
      if (key === keyToRemove) {
        continue;
      }
      copy[key] = withoutKey(child, keyToRemove);
    }
    return copy;
  }
  return value;
}

function findItem(session: ReviewSession, itemId: string): SessionItem {
  const item = session.items.find((candidate) => candidate.review_item_id === itemId);
  if (!item) {
    throw new Error(`review item not found: ${itemId}`);
  }
  return item;
}

function withItem(
  session: ReviewSession,
  itemId: string,
  update: (item: SessionItem) => SessionItem,
): ReviewSession {
  return {
    ...session,
    items: session.items.map((item) =>
      item.review_item_id === itemId ? update(item) : item,
    ),
  };
}

function blankSessionScores(): SessionScore {
  return Object.fromEntries(SCORE_FIELDS.map((field) => [field, null])) as SessionScore;
}

/** Build a fresh review session bound to the packet content hash. */
export function buildSessionFromPacket(packet: ReviewPacket): ReviewSession {
  return {
    schema_version: SESSION_SCHEMA_VERSION,
    packet_schema_version: PACKET_SCHEMA_VERSION,
    packet_sha256: packet.content_sha256,
    exported_at: null,
    items: packet.review_items.map((item) => ({
      review_item_id: item.review_item_id,
      window_id: item.window_id,
      blinded_label: item.blinded_label,
      presentation: item.presentation,
      content_sha256: item.content_sha256,
      scores: blankSessionScores(),
      reviewer: "",
      completed_at: null,
      notes: [],
      complete: false,
    })),
  };
}

function isValidScoreValue(value: unknown, entry: RubricEntry): boolean {
  if (typeof value === "number") {
    return Number.isInteger(value) && value >= SCORE_MIN && value <= SCORE_MAX;
  }
  return value === NOT_APPLICABLE && entry.not_applicable_allowed;
}

/** Missing completion inputs for an item; empty means the item is ready. */
export function itemMissingFields(item: SessionItem): string[] {
  const missing: string[] = [];
  for (const field of SCORE_FIELDS) {
    if (item.scores[field] === null) {
      missing.push(field);
    }
  }
  if (item.reviewer.trim() === "") {
    missing.push("reviewer");
  }
  return missing;
}

/** Set (or clear with null) one score.  Editing a completed item uncompletes it. */
export function setItemScore(
  session: ReviewSession,
  rubric: Rubric,
  itemId: string,
  field: ScoreField,
  value: ScoreValue | null,
): ReviewSession {
  if (value !== null && !isValidScoreValue(value, rubric[field])) {
    const naNote = rubric[field].not_applicable_allowed
      ? ""
      : " (NOT_APPLICABLE is not allowed for this field)";
    throw new Error(`invalid score for ${field}: ${String(value)}${naNote}`);
  }
  return withItem(session, itemId, (item) => {
    const scores = { ...item.scores, [field]: value } as SessionScore;
    const resetCompletion = item.complete && item.scores[field] !== value;
    return {
      ...item,
      scores,
      complete: resetCompletion ? false : item.complete,
      completed_at: resetCompletion ? null : item.completed_at,
    };
  });
}

/** One note per line.  Notes are annotations and do not uncomplete an item. */
export function setItemNotes(
  session: ReviewSession,
  itemId: string,
  notes: string[],
): ReviewSession {
  if (!Array.isArray(notes) || notes.some((note) => typeof note !== "string")) {
    throw new Error("review item notes must be an array of strings");
  }
  return withItem(session, itemId, (item) => ({ ...item, notes: [...notes] }));
}

/** Changing the reviewer invalidates an existing explicit completion. */
export function setItemReviewer(
  session: ReviewSession,
  itemId: string,
  reviewer: string,
): ReviewSession {
  if (typeof reviewer !== "string") {
    throw new Error("reviewer must be a string");
  }
  return withItem(session, itemId, (item) => {
    const resetCompletion = item.complete && reviewer !== item.reviewer;
    return {
      ...item,
      reviewer,
      complete: resetCompletion ? false : item.complete,
      completed_at: resetCompletion ? null : item.completed_at,
    };
  });
}

/** Convenience for long sessions: apply one reviewer name to every item. */
export function setAllReviewers(
  session: ReviewSession,
  reviewer: string,
): ReviewSession {
  if (typeof reviewer !== "string") {
    throw new Error("reviewer must be a string");
  }
  return {
    ...session,
    items: session.items.map((item) => ({
      ...item,
      reviewer,
      complete: item.complete && reviewer === item.reviewer ? item.complete : false,
      completed_at:
        item.complete && reviewer === item.reviewer ? item.completed_at : null,
    })),
  };
}

/**
 * Explicit completion transition: generates (never invents) completed_at only
 * here, and only when all 14 scores and a reviewer are present.
 */
export function completeItem(
  session: ReviewSession,
  itemId: string,
  completedAt: string,
): CompletionResult {
  const item = findItem(session, itemId);
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
  return {
    ok: true,
    session: withItem(session, itemId, (current) => ({
      ...current,
      complete: true,
      completed_at: completedAt,
    })),
  };
}

/** Retract an explicit completion; clears the generated timestamp. */
export function uncompleteItem(
  session: ReviewSession,
  itemId: string,
): ReviewSession {
  return withItem(session, itemId, (item) => ({
    ...item,
    complete: false,
    completed_at: null,
  }));
}

export function summarizeProgress(session: ReviewSession): ProgressSummary {
  const summary: ProgressSummary = {
    total: session.items.length,
    complete: 0,
    ready: 0,
    in_progress: 0,
    untouched: 0,
  };
  for (const item of session.items) {
    if (item.complete) {
      summary.complete += 1;
      continue;
    }
    const missing = itemMissingFields(item);
    if (missing.length === 0) {
      summary.ready += 1;
      continue;
    }
    const hasEdits =
      item.reviewer.trim() !== "" ||
      item.notes.length > 0 ||
      SCORE_FIELDS.some((field) => item.scores[field] !== null);
    if (hasEdits) {
      summary.in_progress += 1;
    } else {
      summary.untouched += 1;
    }
  }
  return summary;
}

/** Errors blocking the final reviews-map export; empty means exportable. */
export function reviewsMapErrors(
  session: ReviewSession,
  rubric: Rubric,
): string[] {
  const errors: string[] = [];
  if (session.items.length === 0) {
    return ["session has no review items"];
  }
  for (const item of session.items) {
    if (!item.complete) {
      errors.push(`${item.review_item_id} is not marked complete`);
      continue;
    }
    for (const field of SCORE_FIELDS) {
      const value = item.scores[field];
      if (value === null) {
        errors.push(`${item.review_item_id} is missing ${field}`);
      } else if (!isValidScoreValue(value, rubric[field])) {
        errors.push(`${item.review_item_id}.${field} has an invalid score`);
      }
    }
    if (item.reviewer.trim() === "") {
      errors.push(`${item.review_item_id} has no reviewer`);
    }
    if (typeof item.completed_at !== "string" || !ISO_TIMESTAMP.test(item.completed_at)) {
      errors.push(`${item.review_item_id} has no valid completed_at`);
    }
    if (item.notes.some((note) => typeof note !== "string")) {
      errors.push(`${item.review_item_id} notes must be strings`);
    }
  }
  return errors;
}

/**
 * Final reviews map in the exact finalizer shape, keyed by review_item_id in
 * deterministic packet order.  Refuses (throws) unless every item is
 * explicitly complete.
 */
export function buildReviewsMap(
  session: ReviewSession,
  rubric: Rubric,
): ReviewsMap {
  const errors = reviewsMapErrors(session, rubric);
  if (errors.length > 0) {
    throw new Error(
      `reviews map requires every item complete: ${errors.slice(0, 3).join("; ")}`,
    );
  }
  const map: ReviewsMap = {};
  for (const item of session.items) {
    map[item.review_item_id] = {
      scores: { ...item.scores } as Record<ScoreField, ScoreValue>,
      reviewer: item.reviewer,
      completed_at: item.completed_at as string,
      notes: [...item.notes],
    };
  }
  return map;
}

/** Session backup export; the only nondeterministic field is the explicit timestamp. */
export function buildSessionExport(
  session: ReviewSession,
  exportedAt: string,
): ReviewSession {
  return { ...session, exported_at: exportedAt };
}

/**
 * Full backup-import validation: schema versions, exact packet-hash binding,
 * immutable packet item content, allowed score values per rubric, completion
 * invariants, and a forbidden structural-leak scan.
 */
export function validateSessionInput(
  input: unknown,
  packet: ReviewPacket,
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
  if (input.items.length !== packet.review_items.length) {
    errors.push(
      `items must contain exactly ${packet.review_items.length} entries (found ${input.items.length})`,
    );
  }

  const items: SessionItem[] = [];
  input.items.forEach((rawItem, index) => {
    const reference = packet.review_items[index];
    if (!reference) {
      return;
    }
    const path = `items[${index}]`;
    if (!isRecordObject(rawItem)) {
      errors.push(`${path} must be an object`);
      return;
    }
    if (!hasExactKeys(rawItem, SESSION_ITEM_KEYS)) {
      errors.push(`${path} keys are invalid`);
    }
    if (rawItem.review_item_id !== reference.review_item_id) {
      errors.push(`${path}.review_item_id does not match the loaded packet`);
    }
    if (rawItem.window_id !== reference.window_id) {
      errors.push(`${path}.window_id does not match the loaded packet`);
    }
    if (rawItem.blinded_label !== reference.blinded_label) {
      errors.push(`${path}.blinded_label does not match the loaded packet`);
    }
    if (!deepEqual(rawItem.presentation, reference.presentation)) {
      errors.push(`${path}.presentation does not match the loaded packet`);
    }
    if (rawItem.content_sha256 !== reference.content_sha256) {
      errors.push(`${path}.content_sha256 does not match the loaded packet`);
    }

    const scores = {} as SessionScore;
    if (!isRecordObject(rawItem.scores) || !hasSameKeys(rawItem.scores, SCORE_FIELDS)) {
      errors.push(`${path}.scores must contain exactly the official score fields`);
    } else {
      for (const field of SCORE_FIELDS) {
        const value = rawItem.scores[field];
        if (value !== null && !isValidScoreValue(value, packet.rubric[field])) {
          errors.push(`${path}.scores.${field} is invalid`);
        }
        scores[field] = value as ScoreValue | null;
      }
    }
    const reviewer = rawItem.reviewer;
    if (typeof reviewer !== "string") {
      errors.push(`${path}.reviewer must be a string`);
    }
    const completedAt = rawItem.completed_at;
    if (
      completedAt !== null &&
      !(typeof completedAt === "string" && ISO_TIMESTAMP.test(completedAt))
    ) {
      errors.push(`${path}.completed_at must be null or an ISO timestamp`);
    }
    if (
      !Array.isArray(rawItem.notes) ||
      rawItem.notes.some((note: unknown) => typeof note !== "string")
    ) {
      errors.push(`${path}.notes must be an array of strings`);
    }
    const complete = rawItem.complete;
    if (typeof complete !== "boolean") {
      errors.push(`${path}.complete must be a boolean`);
    }

    const scoresComplete = SCORE_FIELDS.every((field) => scores[field] !== null);
    const reviewerPresent = typeof reviewer === "string" && reviewer.trim() !== "";
    if (complete === true) {
      if (completedAt === null) {
        errors.push(`${path} completed_at is required once the item is marked complete`);
      }
      if (!scoresComplete) {
        errors.push(`${path} completion requires all 14 scores`);
      }
      if (!reviewerPresent) {
        errors.push(`${path} completion requires a reviewer`);
      }
    } else if (completedAt !== null) {
      errors.push(`${path} completed_at must be null until the item is explicitly marked complete`);
    }

    items.push({
      review_item_id:
        typeof rawItem.review_item_id === "string"
          ? rawItem.review_item_id
          : reference.review_item_id,
      window_id: typeof rawItem.window_id === "string" ? rawItem.window_id : reference.window_id,
      blinded_label:
        typeof rawItem.blinded_label === "string"
          ? rawItem.blinded_label
          : reference.blinded_label,
      presentation:
        deepEqual(rawItem.presentation, reference.presentation)
          ? reference.presentation
          : (rawItem.presentation as Presentation),
      content_sha256:
        typeof rawItem.content_sha256 === "string"
          ? rawItem.content_sha256
          : reference.content_sha256,
      scores,
      reviewer: typeof reviewer === "string" ? reviewer : "",
      completed_at:
        typeof completedAt === "string" && ISO_TIMESTAMP.test(completedAt) ? completedAt : null,
      notes: Array.isArray(rawItem.notes)
        ? (rawItem.notes as string[]).map((note) => note)
        : [],
      complete: complete === true,
    });
  });

  if (errors.length > 0) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    session: {
      schema_version: SESSION_SCHEMA_VERSION,
      packet_schema_version: PACKET_SCHEMA_VERSION,
      packet_sha256: packet.content_sha256,
      exported_at:
        typeof input.exported_at === "string" ? input.exported_at : null,
      items,
    },
  };
}
