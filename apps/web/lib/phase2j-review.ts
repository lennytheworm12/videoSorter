/**
 * Pure Phase 2J human review-session utilities.
 *
 * This module is shared by the build-time server page and the browser
 * client.  It intentionally contains no Node or browser APIs so it can be
 * unit-tested under Jest and imported safely by the client bundle.
 *
 * The review session is review material, never gold: Pass B remains a later
 * blinded audit and nothing here writes the canonical repository packet.
 */

export const PACKET_SCHEMA_VERSION = "phase2j-endpoint-annotation-packet-v1";
export const ANNOTATION_VERSION = "phase2j-endpoint-annotation-v1";
export const SESSION_SCHEMA_VERSION = "phase2j-review-session-v1";

export const ENDPOINT_TYPES = [
  "ENTITY",
  "ABILITY_OR_RESOURCE",
  "EVENT",
  "ACTION",
  "STATE",
  "OUTCOME",
  "QUANTITY",
  "TIME",
  "LOCATION_OR_SPACE",
  "UNDETERMINED",
] as const;

export type EndpointType = (typeof ENDPOINT_TYPES)[number];

export type ReviewToken = {
  token_index: number;
  text: string;
  start: number;
  end: number;
};

/** Client-facing record: the exact locked Bronze window and nothing else. */
export type ReviewRecord = {
  record_index: number;
  window_id: string;
  source_group_id: string;
  bronze_text: string;
  bronze_text_sha256: string;
  bronze_char_length: number;
  tokens: ReviewToken[];
};

export type ReviewEndpoint = {
  endpoint_id: string;
  exact_bronze_text: string;
  char_start: number;
  char_end: number;
  token_start: number;
  token_end: number;
  node_type: EndpointType;
  ambiguity_state: "NONE";
  disposition: "KEEP";
  pass_provenance: "PASS_A";
  human_accepted: true;
  created_sequence: number;
};

export type WindowOutcome = "CLEAN" | "AMBIGUOUS" | "EXCLUDED";
export type WindowStatus = "UNREVIEWED" | "IN_REVIEW" | "AMBIGUOUS" | "EXCLUDED";

export type SessionRecord = ReviewRecord & {
  endpoints: ReviewEndpoint[];
  window_status: WindowStatus;
  outcome: WindowOutcome;
  note: string;
  reviewer_name: string;
  completed_at: string | null;
  pass_a_complete: boolean;
};

export type ReviewSession = {
  schema_version: typeof SESSION_SCHEMA_VERSION;
  annotation_version: typeof ANNOTATION_VERSION;
  packet_schema_version: typeof PACKET_SCHEMA_VERSION;
  packet_sha256: string;
  exported_at: string | null;
  records: SessionRecord[];
};

export type Phase2JReviewPayload = {
  schema_version: typeof PACKET_SCHEMA_VERSION;
  annotation_version: typeof ANNOTATION_VERSION;
  packet_sha256: string;
  records: ReviewRecord[];
};

export type ProgressSummary = {
  total: number;
  unreviewed: number;
  in_review: number;
  ambiguous: number;
  excluded: number;
  pass_a_complete: number;
  endpoints: number;
};

export type ValidationResult =
  | { ok: true; session: ReviewSession }
  | { ok: false; errors: string[] };

/**
 * Scorer/model-field keys that must never appear anywhere in the packet,
 * matching the repository's recursive annotation-facing validation.
 */
const PACKET_FORBIDDEN_KEYS = new Set([
  "score",
  "scores",
  "probability",
  "probabilities",
  "confidence",
  "rank",
  "ranks",
  "ranked",
  "ranking",
  "rankings",
  "prediction",
  "predictions",
  "predicted",
  "predicted_label",
  "predicted_labels",
  "label",
  "labels",
  "gold_label",
  "gold_labels",
  "syntax_importance",
  "syntax_importances",
  "feature_importance",
  "feature_importances",
  "importance",
  "importances",
  "error_taxonomy",
  "model_suggestion",
  "model_suggestions",
  "suggestion",
  "suggestions",
  "model_id",
  "model_name",
  "model_score",
  "logits",
  "proba",
]);

/**
 * Keys that must never appear in a browser review session: packet-internal
 * fields, privacy fields (partition, champion, role, video title, upstream
 * coordinates), and any scorer/model material.  The session schema uses
 * `packet_sha256`, never `content_sha256`.
 */
export const SESSION_FORBIDDEN_KEYS = new Set([
  ...PACKET_FORBIDDEN_KEYS,
  "model",
  "model_data",
  "partition",
  "candidate",
  "candidates",
  "candidate_catalog",
  "candidate_generator_version",
  "catalog",
  "catalog_sha256",
  "champion",
  "role",
  "video_title",
  "video_title_url",
  "annotation_id",
  "upstream_source_id",
  "upstream_start",
  "upstream_end",
  "ambiguity_controls",
  "exclusion_controls",
  "pass_a",
  "pass_b",
  "release_gate",
  "purpose",
  "rules",
  "content_sha256",
  "selection_manifest_sha256",
  "selection_manifest_schema_version",
  "proposal",
  "proposals",
  "sol",
]);

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => actual.includes(key));
}

/**
 * Recursive scan for forbidden field names (case-insensitive) and floating
 * point values.  Returns human-readable paths; empty means clean.
 */
export function findForbiddenFields(
  value: unknown,
  forbiddenKeys: ReadonlySet<string>,
  path = "",
): string[] {
  const problems: string[] = [];
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      problems.push(...findForbiddenFields(item, forbiddenKeys, `${path}[${index}]`));
    });
    return problems;
  }
  if (!isRecordObject(value)) {
    if (typeof value === "number" && !Number.isInteger(value)) {
      problems.push(`floating point value at ${path || "root"}`);
    }
    return problems;
  }
  for (const [key, child] of Object.entries(value)) {
    if (forbiddenKeys.has(key.toLowerCase())) {
      problems.push(`forbidden field "${key}" at ${path || "root"}`);
    }
    if (typeof child === "number" && !Number.isInteger(child)) {
      problems.push(`floating point value at ${path ? `${path}.` : ""}${key}`);
    }
    problems.push(...findForbiddenFields(child, forbiddenKeys, path ? `${path}.${key}` : key));
  }
  return problems;
}

function sanitizeTokens(rawTokens: unknown, bronzeText: string, windowId: string): ReviewToken[] {
  if (!Array.isArray(rawTokens)) {
    throw new Error(`phase2j token table for window ${windowId} must be an array`);
  }
  const tokens: ReviewToken[] = [];
  let previousEnd = 0;
  rawTokens.forEach((rawToken, index) => {
    if (
      !isRecordObject(rawToken) ||
      !hasExactKeys(rawToken, ["token_index", "text", "start", "end"])
    ) {
      throw new Error(`phase2j token record ${index} in window ${windowId} is invalid`);
    }
    const { token_index, text, start, end } = rawToken;
    if (token_index !== index) {
      throw new Error(`phase2j token indices must be sequential in window ${windowId}`);
    }
    if (
      typeof start !== "number" ||
      !Number.isInteger(start) ||
      typeof end !== "number" ||
      !Number.isInteger(end) ||
      start < 0 ||
      end <= start ||
      end > bronzeText.length
    ) {
      throw new Error(`phase2j token offsets are invalid in window ${windowId}`);
    }
    if (typeof text !== "string" || bronzeText.slice(start, end) !== text) {
      throw new Error(`phase2j token text is not an exact source slice in window ${windowId}`);
    }
    if (index > 0 && start <= previousEnd) {
      throw new Error(`phase2j tokens must be ordered and non-overlapping in window ${windowId}`);
    }
    if (bronzeText.slice(previousEnd, start).trim() !== "") {
      throw new Error(`phase2j token table discarded non-whitespace source text in window ${windowId}`);
    }
    previousEnd = end;
    tokens.push({ token_index: index, text, start, end });
  });
  if (tokens.length > 0 && bronzeText.slice(previousEnd).trim() !== "") {
    throw new Error(`phase2j token table discarded trailing source text in window ${windowId}`);
  }
  return tokens;
}

/**
 * Build-time sanitizer: validates the locked packet, scans it for scorer/model
 * fields, and returns only the exact fields the client is allowed to receive.
 */
export function sanitizePacket(raw: unknown): Phase2JReviewPayload {
  if (!isRecordObject(raw)) {
    throw new Error("phase2j packet must be a JSON object");
  }
  if (raw.schema_version !== PACKET_SCHEMA_VERSION) {
    throw new Error(`phase2j packet schema_version must be ${PACKET_SCHEMA_VERSION}`);
  }
  if (raw.annotation_version !== ANNOTATION_VERSION) {
    throw new Error(`phase2j packet annotation_version must be ${ANNOTATION_VERSION}`);
  }
  const contentSha = raw.content_sha256;
  if (typeof contentSha !== "string" || !/^[0-9a-f]{64}$/.test(contentSha)) {
    throw new Error("phase2j packet content_sha256 must be a 64-character hex string");
  }
  const forbidden = findForbiddenFields(raw, PACKET_FORBIDDEN_KEYS);
  if (forbidden.length > 0) {
    throw new Error(`phase2j packet contains forbidden fields: ${forbidden.join("; ")}`);
  }
  if (!Array.isArray(raw.records)) {
    throw new Error("phase2j packet records must be an array");
  }
  if (raw.records.length === 0) {
    throw new Error("phase2j packet records must not be empty");
  }
  const seenWindowIds = new Set<string>();
  const records = raw.records.map((rawRecord, index) => {
    if (!isRecordObject(rawRecord)) {
      throw new Error(`phase2j record ${index} must be an object`);
    }
    const { record_index, window_id, source_group_id } = rawRecord;
    const bronzeText = rawRecord.bronze_text;
    const bronzeSha = rawRecord.bronze_text_sha256;
    const bronzeLength = rawRecord.bronze_char_length;
    if (record_index !== index + 1) {
      throw new Error(`phase2j records must be ordered with record_index ${index + 1}`);
    }
    if (typeof window_id !== "string" || window_id.length === 0) {
      throw new Error(`phase2j record ${index} window_id must be a non-empty string`);
    }
    if (seenWindowIds.has(window_id)) {
      throw new Error(`phase2j duplicate window_id ${window_id}`);
    }
    seenWindowIds.add(window_id);
    if (typeof source_group_id !== "string" || source_group_id.length === 0) {
      throw new Error(`phase2j record ${index} source_group_id must be a non-empty string`);
    }
    if (typeof bronzeText !== "string" || bronzeText.length === 0) {
      throw new Error(`phase2j record ${index} bronze_text must be a non-empty string`);
    }
    if (typeof bronzeSha !== "string" || !/^[0-9a-f]{64}$/.test(bronzeSha)) {
      throw new Error(`phase2j record ${index} bronze_text_sha256 must be a 64-character hex string`);
    }
    if (bronzeLength !== bronzeText.length) {
      throw new Error(`phase2j record ${index} bronze_char_length does not match bronze_text`);
    }
    return {
      record_index,
      window_id,
      source_group_id,
      bronze_text: bronzeText,
      bronze_text_sha256: bronzeSha,
      bronze_char_length: bronzeLength,
      tokens: sanitizeTokens(rawRecord.tokens, bronzeText, window_id),
    };
  });
  return {
    schema_version: PACKET_SCHEMA_VERSION,
    annotation_version: ANNOTATION_VERSION,
    packet_sha256: contentSha,
    records,
  };
}

/**
 * Snap an arbitrary character range (from a DOM selection) to inclusive whole
 * whitespace-token boundaries.  Reverse selections are normalized.  Returns
 * null for empty or whitespace-only selections.
 */
export function snapCharRangeToTokens(
  tokens: ReviewToken[],
  charStart: number,
  charEnd: number,
): { token_start: number; token_end: number } | null {
  if (
    !Number.isInteger(charStart) ||
    !Number.isInteger(charEnd) ||
    charStart < 0 ||
    charEnd < 0
  ) {
    return null;
  }
  const lo = Math.min(charStart, charEnd);
  const hi = Math.max(charStart, charEnd);
  if (hi <= lo) {
    return null;
  }
  let first = -1;
  let last = -1;
  for (const token of tokens) {
    if (first === -1 && token.end > lo) {
      first = token.token_index;
    }
    if (token.start < hi) {
      last = token.token_index;
    }
  }
  if (first === -1 || last === -1 || first > last) {
    return null;
  }
  return { token_start: first, token_end: last };
}

/**
 * Derive the exact Bronze character span and phrase for a whole-token range.
 */
export function deriveEndpointSpan(
  record: ReviewRecord,
  tokenStart: number,
  tokenEnd: number,
): { char_start: number; char_end: number; exact_bronze_text: string } | null {
  if (
    !Number.isInteger(tokenStart) ||
    !Number.isInteger(tokenEnd) ||
    tokenStart < 0 ||
    tokenEnd >= record.tokens.length ||
    tokenStart > tokenEnd
  ) {
    return null;
  }
  const charStart = record.tokens[tokenStart].start;
  const charEnd = record.tokens[tokenEnd].end;
  return {
    char_start: charStart,
    char_end: charEnd,
    exact_bronze_text: record.bronze_text.slice(charStart, charEnd),
  };
}

export function spansOverlap(
  left: { char_start: number; char_end: number },
  right: { char_start: number; char_end: number },
): boolean {
  return left.char_start < right.char_end && right.char_start < left.char_end;
}

export function findOverlappingEndpoint(
  endpoints: ReviewEndpoint[],
  candidate: { char_start: number; char_end: number },
): ReviewEndpoint | null {
  return endpoints.find((endpoint) => spansOverlap(endpoint, candidate)) ?? null;
}

/**
 * Create a Pass A endpoint with the exact Bronze span.  Deterministic
 * endpoint id: p2j:review:<window_id>:ep:<4-digit sequence>.
 */
export function createEndpoint(
  record: ReviewRecord,
  tokenStart: number,
  tokenEnd: number,
  nodeType: EndpointType,
  sequence: number,
): ReviewEndpoint | null {
  const span = deriveEndpointSpan(record, tokenStart, tokenEnd);
  if (!span) {
    return null;
  }
  return {
    endpoint_id: `p2j:review:${record.window_id}:ep:${String(sequence).padStart(4, "0")}`,
    exact_bronze_text: span.exact_bronze_text,
    char_start: span.char_start,
    char_end: span.char_end,
    token_start: tokenStart,
    token_end: tokenEnd,
    node_type: nodeType,
    ambiguity_state: "NONE",
    disposition: "KEEP",
    pass_provenance: "PASS_A",
    human_accepted: true,
    created_sequence: sequence,
  };
}

/** Allocate a stable sequence without reusing an id after a deletion. */
export function nextEndpointSequence(endpoints: ReviewEndpoint[]): number {
  return endpoints.reduce(
    (maximum, endpoint) => Math.max(maximum, endpoint.created_sequence),
    -1,
  ) + 1;
}

export function addEndpointToWindow(
  state: SessionRecord,
  endpoint: ReviewEndpoint,
): SessionRecord {
  return {
    ...state,
    endpoints: [...state.endpoints, endpoint],
    window_status: state.window_status === "UNREVIEWED" ? "IN_REVIEW" : state.window_status,
  };
}

export function removeEndpointFromWindow(
  state: SessionRecord,
  endpointId: string,
): SessionRecord {
  const endpoints = state.endpoints.filter((endpoint) => endpoint.endpoint_id !== endpointId);
  const backToUnreviewed =
    state.outcome === "CLEAN" &&
    endpoints.length === 0 &&
    state.note.trim() === "" &&
    state.reviewer_name.trim() === "" &&
    state.completed_at === null &&
    !state.pass_a_complete;
  return {
    ...state,
    endpoints,
    window_status: backToUnreviewed ? "UNREVIEWED" : state.window_status,
  };
}

/** Outcome transition.  EXCLUDED clears endpoints; AMBIGUOUS keeps them. */
export function applyOutcome(state: SessionRecord, outcome: WindowOutcome): SessionRecord {
  switch (outcome) {
    case "AMBIGUOUS":
      return {
        ...state,
        outcome: "AMBIGUOUS",
        window_status: "AMBIGUOUS",
        pass_a_complete: false,
      };
    case "EXCLUDED":
      return {
        ...state,
        outcome: "EXCLUDED",
        window_status: "EXCLUDED",
        endpoints: [],
        pass_a_complete: false,
      };
    case "CLEAN":
      return {
        ...state,
        outcome: "CLEAN",
        window_status:
          state.endpoints.length > 0 ||
          state.note.trim() !== "" ||
          state.reviewer_name.trim() !== "" ||
          state.completed_at !== null ||
          state.pass_a_complete
            ? "IN_REVIEW"
            : "UNREVIEWED",
      };
  }
}

/** Sign Pass A (reviewer name + date) or retract it for any review outcome. */
export function markPassAComplete(
  state: SessionRecord,
  reviewerName: string,
  completedAt: string | null,
  complete: boolean,
): SessionRecord {
  const windowStatus: WindowStatus =
    state.outcome === "AMBIGUOUS"
      ? "AMBIGUOUS"
      : state.outcome === "EXCLUDED"
        ? "EXCLUDED"
        : "IN_REVIEW";
  return {
    ...state,
    reviewer_name: reviewerName,
    completed_at: completedAt,
    pass_a_complete: complete,
    window_status: windowStatus,
  };
}

export function buildSessionFromPayload(payload: Phase2JReviewPayload): ReviewSession {
  return {
    schema_version: SESSION_SCHEMA_VERSION,
    annotation_version: ANNOTATION_VERSION,
    packet_schema_version: PACKET_SCHEMA_VERSION,
    packet_sha256: payload.packet_sha256,
    exported_at: null,
    records: payload.records.map(
      (record): SessionRecord => ({
        ...record,
        endpoints: [],
        window_status: "UNREVIEWED",
        outcome: "CLEAN",
        note: "",
        reviewer_name: "",
        completed_at: null,
        pass_a_complete: false,
      }),
    ),
  };
}

/** Export copy; the only nondeterministic field is the explicit timestamp. */
export function buildExportSession(
  session: ReviewSession,
  exportedAt: string,
): ReviewSession {
  return {
    ...session,
    exported_at: exportedAt,
  };
}

export function summarizeProgress(session: ReviewSession): ProgressSummary {
  const summary: ProgressSummary = {
    total: session.records.length,
    unreviewed: 0,
    in_review: 0,
    ambiguous: 0,
    excluded: 0,
    pass_a_complete: 0,
    endpoints: 0,
  };
  for (const record of session.records) {
    summary.endpoints += record.endpoints.length;
    if (record.pass_a_complete) {
      summary.pass_a_complete += 1;
    }
    switch (record.window_status) {
      case "UNREVIEWED":
        summary.unreviewed += 1;
        break;
      case "IN_REVIEW":
        summary.in_review += 1;
        break;
      case "AMBIGUOUS":
        summary.ambiguous += 1;
        break;
      case "EXCLUDED":
        summary.excluded += 1;
        break;
    }
  }
  return summary;
}

function isOneOf<T extends readonly string[]>(value: unknown, allowed: T): value is T[number] {
  return typeof value === "string" && (allowed as readonly string[]).includes(value);
}

function asOneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : fallback;
}

function tokensEqual(left: unknown, right: ReviewToken[]): boolean {
  if (!Array.isArray(left) || left.length !== right.length) {
    return false;
  }
  return left.every((token, index) => {
    if (!isRecordObject(token)) {
      return false;
    }
    const reference = right[index];
    return (
      token.token_index === reference.token_index &&
      token.text === reference.text &&
      token.start === reference.start &&
      token.end === reference.end
    );
  });
}

function requireString(
  value: unknown,
  label: string,
  errors: string[],
): string {
  if (typeof value !== "string") {
    errors.push(`${label} must be a string`);
    return "";
  }
  return value;
}

function requireInt(
  value: unknown,
  label: string,
  errors: string[],
): number | null {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    errors.push(`${label} must be an integer`);
    return null;
  }
  return value;
}

/**
 * Full import validation: schema, packet binding, exact 30 identities and
 * order, endpoint token ranges / exact Bronze slices, no overlap or
 * duplicates, allowed enums, status invariants, and forbidden-field scan.
 */
export function validateSessionInput(
  input: unknown,
  packetSha256: string,
  referenceRecords: ReviewRecord[],
): ValidationResult {
  const errors: string[] = [];
  if (!isRecordObject(input)) {
    return { ok: false, errors: ["session must be a JSON object"] };
  }
  if (input.schema_version !== SESSION_SCHEMA_VERSION) {
    errors.push(`schema_version must be ${SESSION_SCHEMA_VERSION}`);
  }
  if (input.annotation_version !== ANNOTATION_VERSION) {
    errors.push(`annotation_version must be ${ANNOTATION_VERSION}`);
  }
  if (input.packet_schema_version !== PACKET_SCHEMA_VERSION) {
    errors.push(`packet_schema_version must be ${PACKET_SCHEMA_VERSION}`);
  }
  if (input.packet_sha256 !== packetSha256) {
    errors.push("packet_sha256 does not match the locked packet content hash");
  }
  if (input.exported_at !== null && typeof input.exported_at !== "string") {
    errors.push("exported_at must be null or a string");
  }
  const forbidden = findForbiddenFields(input, SESSION_FORBIDDEN_KEYS);
  if (forbidden.length > 0) {
    errors.push(...forbidden.map((problem) => `session ${problem}`));
  }
  if (!Array.isArray(input.records)) {
    errors.push("records must be an array");
    return { ok: false, errors };
  }
  if (input.records.length !== referenceRecords.length) {
    errors.push(
      `records must contain exactly ${referenceRecords.length} windows (found ${input.records.length})`,
    );
  }

  const records: SessionRecord[] = [];
  input.records.forEach((rawRecord, index) => {
    const reference = referenceRecords[index];
    if (!reference) {
      return;
    }
    const path = `records[${index}]`;
    if (!isRecordObject(rawRecord)) {
      errors.push(`${path} must be an object`);
      return;
    }
    if (rawRecord.record_index !== index + 1) {
      errors.push(`${path}.record_index must be ${index + 1}`);
    }
    if (rawRecord.window_id !== reference.window_id) {
      errors.push(`${path}.window_id does not match the locked window (expected ${reference.window_id})`);
    }
    if (rawRecord.source_group_id !== reference.source_group_id) {
      errors.push(`${path}.source_group_id does not match the locked window`);
    }
    if (rawRecord.bronze_text !== reference.bronze_text) {
      errors.push(`${path}.bronze_text does not match the locked window`);
    }
    if (rawRecord.bronze_text_sha256 !== reference.bronze_text_sha256) {
      errors.push(`${path}.bronze_text_sha256 does not match the locked window`);
    }
    if (rawRecord.bronze_char_length !== reference.bronze_char_length) {
      errors.push(`${path}.bronze_char_length does not match the locked window`);
    }
    if (!tokensEqual(rawRecord.tokens, reference.tokens)) {
      errors.push(`${path}.tokens do not match the locked token table`);
    }

    const outcome = rawRecord.outcome;
    const windowStatus = rawRecord.window_status;
    if (!isOneOf(outcome, ["CLEAN", "AMBIGUOUS", "EXCLUDED"])) {
      errors.push(`${path}.outcome must be CLEAN, AMBIGUOUS, or EXCLUDED`);
    }
    if (!isOneOf(windowStatus, ["UNREVIEWED", "IN_REVIEW", "AMBIGUOUS", "EXCLUDED"])) {
      errors.push(`${path}.window_status is invalid`);
    }
    const note = requireString(rawRecord.note, `${path}.note`, errors);
    const reviewerName = requireString(rawRecord.reviewer_name, `${path}.reviewer_name`, errors);
    const completedAt = rawRecord.completed_at;
    if (completedAt !== null && typeof completedAt !== "string") {
      errors.push(`${path}.completed_at must be null or a string`);
    }
    const completedAtValid =
      typeof completedAt === "string" && /^\d{4}-\d{2}-\d{2}/.test(completedAt);
    if (completedAt !== null && !completedAtValid) {
      errors.push(`${path}.completed_at must be an ISO date (YYYY-MM-DD...)`);
    }
    const completedAtValue: string | null = completedAtValid ? completedAt : null;
    const passAComplete = rawRecord.pass_a_complete;
    if (typeof passAComplete !== "boolean") {
      errors.push(`${path}.pass_a_complete must be a boolean`);
    }

    const endpoints: ReviewEndpoint[] = [];
    if (!Array.isArray(rawRecord.endpoints)) {
      errors.push(`${path}.endpoints must be an array`);
    } else {
      const seenIds = new Set<string>();
      rawRecord.endpoints.forEach((rawEndpoint, endpointIndex) => {
        const endpointPath = `${path}.endpoints[${endpointIndex}]`;
        if (!isRecordObject(rawEndpoint)) {
          errors.push(`${endpointPath} must be an object`);
          return;
        }
        const rawEndpointId = rawEndpoint.endpoint_id;
        const endpointId =
          typeof rawEndpointId === "string" && rawEndpointId.startsWith("p2j:review:")
            ? rawEndpointId
            : null;
        if (endpointId === null) {
          errors.push(`${endpointPath}.endpoint_id is invalid`);
        } else if (seenIds.has(endpointId)) {
          errors.push(`${endpointPath}.endpoint_id is duplicated`);
        } else {
          seenIds.add(endpointId);
        }
        const tokenStart = requireInt(rawEndpoint.token_start, `${endpointPath}.token_start`, errors);
        const tokenEnd = requireInt(rawEndpoint.token_end, `${endpointPath}.token_end`, errors);
        const charStart = requireInt(rawEndpoint.char_start, `${endpointPath}.char_start`, errors);
        const charEnd = requireInt(rawEndpoint.char_end, `${endpointPath}.char_end`, errors);
        if (tokenStart === null || tokenEnd === null || charStart === null || charEnd === null) {
          return;
        }
        if (tokenStart < 0 || tokenEnd >= reference.tokens.length || tokenStart > tokenEnd) {
          errors.push(`${endpointPath} token range is out of bounds`);
        }
        if (charStart < 0 || charEnd <= charStart || charEnd > reference.bronze_text.length) {
          errors.push(`${endpointPath} character range is out of bounds`);
        }
        if (
          tokenStart < reference.tokens.length &&
          charStart < reference.bronze_text.length &&
          reference.tokens[tokenStart].start !== charStart
        ) {
          errors.push(`${endpointPath} token_start does not match char_start`);
        }
        if (
          tokenEnd < reference.tokens.length &&
          charEnd <= reference.bronze_text.length &&
          reference.tokens[tokenEnd].end !== charEnd
        ) {
          errors.push(`${endpointPath} token_end does not match char_end`);
        }
        const exactText = requireString(
          rawEndpoint.exact_bronze_text,
          `${endpointPath}.exact_bronze_text`,
          errors,
        );
        if (
          charStart >= 0 &&
          charEnd <= reference.bronze_text.length &&
          reference.bronze_text.slice(charStart, charEnd) !== exactText
        ) {
          errors.push(`${endpointPath}.exact_bronze_text is not an exact Bronze slice`);
        }
        const nodeType = rawEndpoint.node_type;
        if (!isOneOf(nodeType, ENDPOINT_TYPES)) {
          errors.push(`${endpointPath}.node_type is not an allowed endpoint type`);
        }
        if (rawEndpoint.ambiguity_state !== "NONE") {
          errors.push(`${endpointPath}.ambiguity_state must be NONE in this route`);
        }
        if (rawEndpoint.disposition !== "KEEP") {
          errors.push(`${endpointPath}.disposition must be KEEP in this route`);
        }
        if (rawEndpoint.pass_provenance !== "PASS_A") {
          errors.push(`${endpointPath}.pass_provenance must be PASS_A in this route`);
        }
        if (rawEndpoint.human_accepted !== true) {
          errors.push(`${endpointPath}.human_accepted must be true`);
        }
        const createdSequence = requireInt(
          rawEndpoint.created_sequence,
          `${endpointPath}.created_sequence`,
          errors,
        );
        if (
          tokenStart !== null &&
          tokenEnd !== null &&
          charStart !== null &&
          charEnd !== null &&
          nodeType !== undefined &&
          endpointId !== null
        ) {
          endpoints.push({
            endpoint_id: endpointId,
            exact_bronze_text: exactText,
            char_start: charStart,
            char_end: charEnd,
            token_start: tokenStart,
            token_end: tokenEnd,
            node_type: nodeType as EndpointType,
            ambiguity_state: "NONE",
            disposition: "KEEP",
            pass_provenance: "PASS_A",
            human_accepted: true,
            created_sequence: createdSequence ?? 0,
          });
        }
      });
      for (let leftIndex = 0; leftIndex < endpoints.length; leftIndex += 1) {
        for (let rightIndex = leftIndex + 1; rightIndex < endpoints.length; rightIndex += 1) {
          if (spansOverlap(endpoints[leftIndex], endpoints[rightIndex])) {
            errors.push(
              `${path} endpoints overlap: ${endpoints[leftIndex].exact_bronze_text} vs ${endpoints[rightIndex].exact_bronze_text}`,
            );
          }
        }
      }
    }

    const outcomeValue: WindowOutcome = asOneOf(
      outcome,
      ["CLEAN", "AMBIGUOUS", "EXCLUDED"],
      "CLEAN",
    );
    const statusValue: WindowStatus = asOneOf(
      windowStatus,
      ["UNREVIEWED", "IN_REVIEW", "AMBIGUOUS", "EXCLUDED"],
      "UNREVIEWED",
    );
    const passACompleteValue = passAComplete === true;
    if (passACompleteValue && (reviewerName.trim() === "" || completedAtValue === null)) {
      errors.push(`${path} Pass A completion requires reviewer name and completed_at`);
    }
    if (statusValue === "UNREVIEWED") {
      if (outcomeValue !== "CLEAN") {
        errors.push(`${path} UNREVIEWED windows must have outcome CLEAN`);
      }
      if (endpoints.length > 0) {
        errors.push(`${path} UNREVIEWED windows cannot have endpoints`);
      }
      if (note !== "" || reviewerName !== "" || completedAtValue !== null || passACompleteValue) {
        errors.push(`${path} UNREVIEWED windows cannot carry review fields`);
      }
    } else if (statusValue === "IN_REVIEW") {
      if (outcomeValue !== "CLEAN") {
        errors.push(`${path} IN_REVIEW windows must have outcome CLEAN`);
      }
    } else if (statusValue === "AMBIGUOUS") {
      if (outcomeValue !== "AMBIGUOUS") {
        errors.push(`${path} AMBIGUOUS windows must have outcome AMBIGUOUS`);
      }
      if (note.trim() === "") {
        errors.push(`${path} AMBIGUOUS windows require a reviewer note`);
      }
    } else if (statusValue === "EXCLUDED") {
      if (outcomeValue !== "EXCLUDED") {
        errors.push(`${path} EXCLUDED windows must have outcome EXCLUDED`);
      }
      if (note.trim() === "") {
        errors.push(`${path} EXCLUDED windows require a reviewer note`);
      }
      if (endpoints.length > 0) {
        errors.push(`${path} EXCLUDED windows must have no endpoints`);
      }
    }
    records.push({
      ...reference,
      endpoints,
      window_status: statusValue,
      outcome: outcomeValue,
      note,
      reviewer_name: reviewerName,
      completed_at: completedAtValue,
      pass_a_complete: passACompleteValue,
    });
  });

  if (errors.length > 0) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    session: {
      schema_version: SESSION_SCHEMA_VERSION,
      annotation_version: ANNOTATION_VERSION,
      packet_schema_version: PACKET_SCHEMA_VERSION,
      packet_sha256: packetSha256,
      exported_at:
        typeof input.exported_at === "string"
          ? input.exported_at
          : null,
      records,
    },
  };
}
