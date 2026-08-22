/**
 * Pure Phase 2J human-vs-Sol adjudication utilities.
 *
 * This module is shared by the build-time server page and the browser client
 * and intentionally contains no Node or browser APIs so it can be unit-tested
 * under Jest.  It reads only the generated sanitized adjudication packet and
 * produces review material exports.
 *
 * Sol proposals are a second opinion (navigation/audit only), never gold.
 * Nothing here auto-promotes Sol, and the export remains REVIEW MATERIAL until
 * a separately validated canonical import/finalizer runs.
 */

import {
  ENDPOINT_TYPES,
  SESSION_FORBIDDEN_KEYS,
  spansOverlap,
  type EndpointType,
  type ReviewRecord,
  type ReviewToken,
} from "./phase2j-review";

export const ADJUDICATION_PACKET_SCHEMA_VERSION = "phase2j-adjudication-packet-v1";
export const ADJUDICATION_VERSION = "phase2j-adjudication-v1";
export const ADJUDICATION_EXPORT_SCHEMA_VERSION = "phase2j-adjudication-export-v2";
export const ADJUDICATION_STATE_SCHEMA_VERSION = "phase2j-adjudication-state-v1";

/** The five Pass B source-grounded audit attestation checks. */
export const AUDIT_CHECKS = [
  "boundaries",
  "omissions",
  "roles",
  "duplicates",
  "ambiguity",
] as const;

export type AuditCheckKey = (typeof AUDIT_CHECKS)[number];
export type AuditChecks = Record<AuditCheckKey, boolean>;

export type AdjudicationComponentClass =
  | "EXACT_AGREEMENT"
  | "TYPE_DISAGREEMENT"
  | "BOUNDARY_DISAGREEMENT"
  | "SOL_ONLY"
  | "HUMAN_ONLY";

export type AdjudicationHumanEndpoint = {
  endpoint_id: string;
  exact_bronze_text: string;
  char_start: number;
  char_end: number;
  token_start: number;
  token_end: number;
  node_type: EndpointType;
};

export type AdjudicationSolEndpoint = {
  endpoint_id: string;
  exact_bronze_text: string;
  char_start: number;
  char_end: number;
  token_start: number;
  token_end: number;
  node_type: EndpointType | null;
  sol_ambiguity_state: string | null;
  sol_rationale: string | null;
};

export type AdjudicationComponent = {
  component_id: string;
  classification: AdjudicationComponentClass;
  human_endpoint_ids: string[];
  sol_endpoint_ids: string[];
};

export type AdjudicationRecord = ReviewRecord & {
  human_outcome: "CLEAN" | "AMBIGUOUS" | "EXCLUDED";
  human_endpoints: AdjudicationHumanEndpoint[];
  sol_endpoints: AdjudicationSolEndpoint[];
  components: AdjudicationComponent[];
};

export type AdjudicationTotals = {
  windows: number;
  components: number;
  exact_agreements: number;
  type_disagreements: number;
  boundary_disagreements: number;
  sol_only: number;
  human_only: number;
  human_endpoints: number;
  sol_endpoints: number;
};

/** Client-facing sanitized payload; the raw packet's self-hash is renamed. */
export type AdjudicationPayload = {
  schema_version: typeof ADJUDICATION_PACKET_SCHEMA_VERSION;
  adjudication_version: typeof ADJUDICATION_VERSION;
  annotation_version: "phase2j-endpoint-annotation-v1";
  packet_schema_version: "phase2j-endpoint-annotation-packet-v1";
  packet_sha256: string;
  adjudication_packet_sha256: string;
  human_session_sha256: string;
  sol_review_sha256: string;
  totals: AdjudicationTotals;
  records: AdjudicationRecord[];
};

export type ComponentDecision =
  | { kind: "KEEP_HUMAN_SET" }
  | { kind: "KEEP_SOL_SET" }
  | { kind: "DROP" }
  | { kind: "CUSTOM"; token_start: number; token_end: number; node_type: EndpointType };

export type AdjudicationOutcome = "CLEAN" | "AMBIGUOUS" | "EXCLUDED";

export type AdjudicationRecordState = {
  record_index: number;
  window_id: string;
  outcome: AdjudicationOutcome;
  note: string;
  decisions: Record<string, ComponentDecision>;
};

export type AdjudicationState = {
  schema_version: typeof ADJUDICATION_STATE_SCHEMA_VERSION;
  adjudication_packet_sha256: string;
  reviewer_name: string;
  records: AdjudicationRecordState[];
};

export type ResolvedEndpointProvenance = "HUMAN" | "SOL" | "SHARED" | "CUSTOM";

export type ResolvedEndpoint = {
  endpoint_id: string;
  component_id: string;
  exact_bronze_text: string;
  char_start: number;
  char_end: number;
  token_start: number;
  token_end: number;
  node_type: EndpointType;
  provenance_source: ResolvedEndpointProvenance;
};

export type ComponentExportEntry = {
  component_id: string;
  classification: AdjudicationComponentClass;
  decision: ComponentDecision | null;
  resolved_by:
    | "PRE_RESOLVED"
    | "HUMAN_SET"
    | "SOL_SET"
    | "DROP"
    | "CUSTOM"
    | "WINDOW_AMBIGUOUS"
    | "WINDOW_EXCLUDED";
};

export type AdjudicationExportRecord = {
  record_index: number;
  window_id: string;
  outcome: AdjudicationOutcome;
  note: string;
  components: ComponentExportEntry[];
  resolved_endpoints: ResolvedEndpoint[];
};

export type AdjudicationExport = {
  schema_version: typeof ADJUDICATION_EXPORT_SCHEMA_VERSION;
  adjudication_version: typeof ADJUDICATION_VERSION;
  packet_schema_version: "phase2j-endpoint-annotation-packet-v1";
  adjudication_packet_sha256: string;
  packet_sha256: string;
  human_session_sha256: string;
  sol_review_sha256: string;
  status_label: "REVIEW_MATERIAL";
  reviewer_name: string;
  exported_at: string | null;
  audit_checks: AuditChecks;
  records: AdjudicationExportRecord[];
};

export type AdjudicationProgress = {
  windows: number;
  resolved_windows: number;
  components: number;
  resolved_components: number;
  ambiguous: number;
  excluded: number;
};

export type ValidationResult =
  | { ok: true; state: AdjudicationState }
  | { ok: false; errors: string[] };

export type ExportValidationResult =
  | { ok: true; state: AdjudicationState; audit_checks: AuditChecks }
  | { ok: false; errors: string[] };

export type ExportResult =
  | { ok: true; export: AdjudicationExport }
  | { ok: false; errors: string[] };

/** Never-appear keys for adjudication packets and exports (case-insensitive). */
export const ADJUDICATION_FORBIDDEN_KEYS = new Set([
  ...SESSION_FORBIDDEN_KEYS,
  "model",
  "model_data",
  "reviewer_model",
  "reasoning_effort",
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
  "rules",
  "selection_manifest_sha256",
  "selection_manifest_schema_version",
  "proposal",
  "proposals",
  "sol",
  "reviewer_name",
  "reviewed_at",
  "completed_at",
  "exported_at",
]);
// The adjudication packet's canonical self-hash and purpose are sanctioned
// fields; they are not client session or scorer material.
ADJUDICATION_FORBIDDEN_KEYS.delete("content_sha256");
ADJUDICATION_FORBIDDEN_KEYS.delete("purpose");

/** Export/state scans may carry reviewer identity and export timestamps. */
export const ADJUDICATION_EXPORT_FORBIDDEN_KEYS = new Set(
  [...ADJUDICATION_FORBIDDEN_KEYS].filter(
    (key) => key !== "reviewer_name" && key !== "exported_at",
  ),
);

const PACKET_ENVELOPE_KEYS = [
  "content_sha256",
  "schema_version",
  "adjudication_version",
  "annotation_version",
  "packet_schema_version",
  "packet_sha256",
  "human_session_schema_version",
  "human_session_sha256",
  "sol_review_schema_version",
  "sol_review_sha256",
  "visibility_gate",
  "purpose",
  "totals",
  "records",
] as const;

const TOTALS_KEYS = [
  "windows",
  "components",
  "exact_agreements",
  "type_disagreements",
  "boundary_disagreements",
  "sol_only",
  "human_only",
  "human_endpoints",
  "sol_endpoints",
] as const;

const RECORD_KEYS = [
  "record_index",
  "window_id",
  "source_group_id",
  "bronze_text",
  "bronze_text_sha256",
  "bronze_char_length",
  "tokens",
  "human_outcome",
  "human_endpoints",
  "sol_endpoints",
  "components",
] as const;

const HUMAN_ENDPOINT_KEYS = [
  "endpoint_id",
  "exact_bronze_text",
  "char_start",
  "char_end",
  "token_start",
  "token_end",
  "node_type",
] as const;

const SOL_ENDPOINT_KEYS = [
  "endpoint_id",
  "exact_bronze_text",
  "char_start",
  "char_end",
  "token_start",
  "token_end",
  "node_type",
  "sol_ambiguity_state",
  "sol_rationale",
] as const;

const COMPONENT_KEYS = [
  "component_id",
  "classification",
  "human_endpoint_ids",
  "sol_endpoint_ids",
] as const;

const STATE_RECORD_KEYS = [
  "record_index",
  "window_id",
  "outcome",
  "note",
  "decisions",
] as const;

const STATE_ENVELOPE_KEYS = [
  "schema_version",
  "adjudication_packet_sha256",
  "reviewer_name",
  "records",
] as const;

const EXPORT_ENVELOPE_KEYS = [
  "schema_version",
  "adjudication_version",
  "packet_schema_version",
  "adjudication_packet_sha256",
  "packet_sha256",
  "human_session_sha256",
  "sol_review_sha256",
  "status_label",
  "reviewer_name",
  "exported_at",
  "audit_checks",
  "records",
] as const;

const EXPORT_RECORD_KEYS = [
  "record_index",
  "window_id",
  "outcome",
  "note",
  "components",
  "resolved_endpoints",
] as const;

const EXPORT_COMPONENT_KEYS = [
  "component_id",
  "classification",
  "decision",
  "resolved_by",
] as const;

const RESOLVED_ENDPOINT_KEYS = [
  "endpoint_id",
  "component_id",
  "exact_bronze_text",
  "char_start",
  "char_end",
  "token_start",
  "token_end",
  "node_type",
  "provenance_source",
] as const;

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => actual.includes(key));
}

function isOneOf<T extends readonly string[]>(value: unknown, allowed: T): value is T[number] {
  return typeof value === "string" && (allowed as readonly string[]).includes(value);
}

function asOneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : fallback;
}

function requireString(value: unknown, label: string, errors: string[]): string {
  if (typeof value !== "string") {
    errors.push(`${label} must be a string`);
    return "";
  }
  return value;
}

function requireInt(value: unknown, label: string, errors: string[]): number | null {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    errors.push(`${label} must be an integer`);
    return null;
  }
  return value;
}

function requireSha256(value: unknown, label: string, errors: string[]): string {
  const text = requireString(value, label, errors);
  if (text !== "" && !/^[0-9a-f]{64}$/.test(text)) {
    errors.push(`${label} must be a 64-character hex string`);
  }
  return text;
}

/** Runtime audit-checks validation shared by build, import, and autosave. */
function validateAuditChecks(
  value: unknown,
  label: string,
  errors: string[],
): AuditChecks | null {
  if (!isRecordObject(value)) {
    errors.push(`${label} must be an object with the five Pass B audit checks`);
    return null;
  }
  for (const key of AUDIT_CHECKS) {
    if (!(key in value)) {
      errors.push(`${label}.${key} is missing`);
    } else if (typeof value[key] !== "boolean") {
      errors.push(`${label}.${key} must be a boolean`);
    }
  }
  for (const key of Object.keys(value)) {
    if (!(AUDIT_CHECKS as readonly string[]).includes(key)) {
      errors.push(`${label}.${key} is not a recognized Pass B audit check`);
    }
  }
  const complete = AUDIT_CHECKS.every((key) => value[key] === true);
  if (!complete) {
    errors.push(`${label} requires every audit check to be true`);
  }
  return complete &&
    AUDIT_CHECKS.every((key) => typeof value[key] === "boolean") &&
    Object.keys(value).every((key) => (AUDIT_CHECKS as readonly string[]).includes(key))
    ? (value as AuditChecks)
    : null;
}

/** Fresh all-false Pass B audit attestation. */
export function emptyAuditChecks(): AuditChecks {
  return {
    boundaries: false,
    omissions: false,
    roles: false,
    duplicates: false,
    ambiguity: false,
  };
}

export function auditChecksComplete(auditChecks: AuditChecks): boolean {
  return AUDIT_CHECKS.every((key) => auditChecks[key] === true);
}

/** Parse a stored/imported audit-checks object; null means invalid. */
export function parseAuditChecks(value: unknown): AuditChecks | null {
  const errors: string[] = [];
  const parsed = validateAuditChecks(value, "audit_checks", errors);
  return errors.length === 0 ? parsed : null;
}

/** Recursive scan for forbidden field names and floating point values. */
export function findAdjudicationForbiddenFields(
  value: unknown,
  forbiddenKeys: ReadonlySet<string>,
  path = "",
): string[] {
  const problems: string[] = [];
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      problems.push(...findAdjudicationForbiddenFields(item, forbiddenKeys, `${path}[${index}]`));
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
    problems.push(...findAdjudicationForbiddenFields(child, forbiddenKeys, path ? `${path}.${key}` : key));
  }
  return problems;
}

function validateTokenTable(
  rawTokens: unknown,
  bronzeText: string,
  windowId: string,
  errors: string[],
): ReviewToken[] {
  if (!Array.isArray(rawTokens)) {
    errors.push(`token table for window ${windowId} must be an array`);
    return [];
  }
  const tokens: ReviewToken[] = [];
  let previousEnd = 0;
  rawTokens.forEach((rawToken, index) => {
    if (
      !isRecordObject(rawToken) ||
      !hasExactKeys(rawToken, ["token_index", "text", "start", "end"])
    ) {
      errors.push(`token record ${index} in window ${windowId} is invalid`);
      return;
    }
    const tokenIndex = rawToken.token_index;
    const text = rawToken.text;
    const start = rawToken.start;
    const end = rawToken.end;
    if (tokenIndex !== index) {
      errors.push(`token indices must be sequential in window ${windowId}`);
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
      errors.push(`token offsets are invalid in window ${windowId}`);
      return;
    }
    if (typeof text !== "string" || bronzeText.slice(start, end) !== text) {
      errors.push(`token text is not an exact source slice in window ${windowId}`);
      return;
    }
    if (index > 0 && start <= previousEnd) {
      errors.push(`tokens must be ordered and non-overlapping in window ${windowId}`);
    }
    if (bronzeText.slice(previousEnd, start).trim() !== "") {
      errors.push(`token table discarded source text in window ${windowId}`);
    }
    previousEnd = end;
    tokens.push({ token_index: index, text, start, end });
  });
  if (tokens.length > 0 && bronzeText.slice(previousEnd).trim() !== "") {
    errors.push(`token table discarded trailing source text in window ${windowId}`);
  }
  return tokens;
}

function classifyComponent(
  human: AdjudicationHumanEndpoint[],
  sol: AdjudicationSolEndpoint[],
): AdjudicationComponentClass {
  if (human.length > 0 && sol.length > 0) {
    if (human.length === 1 && sol.length === 1) {
      const left = human[0];
      const right = sol[0];
      const sameSpan =
        left.token_start === right.token_start && left.token_end === right.token_end;
      const sameType = left.node_type === right.node_type;
      if (sameSpan && sameType) {
        return "EXACT_AGREEMENT";
      }
      if (sameSpan && !sameType) {
        return "TYPE_DISAGREEMENT";
      }
    }
    return "BOUNDARY_DISAGREEMENT";
  }
  if (sol.length > 0) {
    return "SOL_ONLY";
  }
  return "HUMAN_ONLY";
}

function validateSpanEndpoint(
  endpoint: Record<string, unknown>,
  record: AdjudicationRecord,
  path: string,
  errors: string[],
  allowNullType: boolean,
): AdjudicationHumanEndpoint | AdjudicationSolEndpoint | null {
  const tokenStart = requireInt(endpoint.token_start, `${path}.token_start`, errors);
  const tokenEnd = requireInt(endpoint.token_end, `${path}.token_end`, errors);
  const charStart = requireInt(endpoint.char_start, `${path}.char_start`, errors);
  const charEnd = requireInt(endpoint.char_end, `${path}.char_end`, errors);
  if (tokenStart === null || tokenEnd === null || charStart === null || charEnd === null) {
    return null;
  }
  if (tokenStart < 0 || tokenEnd >= record.tokens.length || tokenStart > tokenEnd) {
    errors.push(`${path} token range is out of bounds`);
  }
  if (charStart < 0 || charEnd <= charStart || charEnd > record.bronze_text.length) {
    errors.push(`${path} character range is out of bounds`);
  }
  if (
    tokenStart < record.tokens.length &&
    record.tokens[tokenStart].start !== charStart
  ) {
    errors.push(`${path} token_start does not match char_start`);
  }
  if (
    tokenEnd < record.tokens.length &&
    record.tokens[tokenEnd].end !== charEnd
  ) {
    errors.push(`${path} token_end does not match char_end`);
  }
  const exactText = requireString(endpoint.exact_bronze_text, `${path}.exact_bronze_text`, errors);
  if (
    charStart >= 0 &&
    charEnd <= record.bronze_text.length &&
    record.bronze_text.slice(charStart, charEnd) !== exactText
  ) {
    errors.push(`${path}.exact_bronze_text is not an exact Bronze slice`);
  }
  const nodeType = endpoint.node_type;
  if (allowNullType) {
    if (nodeType !== null && !isOneOf(nodeType, ENDPOINT_TYPES)) {
      errors.push(`${path}.node_type is not an allowed endpoint type`);
    }
  } else if (!isOneOf(nodeType, ENDPOINT_TYPES)) {
    errors.push(`${path}.node_type is not an allowed endpoint type`);
  }
  return {
    endpoint_id: requireString(endpoint.endpoint_id, `${path}.endpoint_id`, errors),
    exact_bronze_text: exactText,
    char_start: charStart,
    char_end: charEnd,
    token_start: tokenStart,
    token_end: tokenEnd,
    node_type: nodeType as EndpointType | null,
  } as AdjudicationHumanEndpoint | AdjudicationSolEndpoint;
}

function validateNoInternalOverlap(
  endpoints: Array<{ char_start: number; char_end: number; endpoint_id: string }>,
  path: string,
  errors: string[],
): void {
  for (let leftIndex = 0; leftIndex < endpoints.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < endpoints.length; rightIndex += 1) {
      if (spansOverlap(endpoints[leftIndex], endpoints[rightIndex])) {
        errors.push(
          `${path} endpoints overlap: ${endpoints[leftIndex].endpoint_id} vs ${endpoints[rightIndex].endpoint_id}`,
        );
      }
    }
  }
}

function validateComponents(
  record: AdjudicationRecord,
  path: string,
  errors: string[],
): void {
  const humanById = new Map(record.human_endpoints.map((endpoint) => [endpoint.endpoint_id, endpoint]));
  const solById = new Map(record.sol_endpoints.map((endpoint) => [endpoint.endpoint_id, endpoint]));
  const seenComponentIds = new Set<string>();
  const coveredHuman = new Set<string>();
  const coveredSol = new Set<string>();
  record.components.forEach((component, sequence) => {
    const componentPath = `${path}.components[${sequence}]`;
    const expectedId = `p2j:adjudicate:${record.window_id}:c:${String(sequence + 1).padStart(4, "0")}`;
    if (component.component_id !== expectedId) {
      errors.push(`${componentPath}.component_id must be ${expectedId}`);
    }
    if (seenComponentIds.has(component.component_id)) {
      errors.push(`${componentPath}.component_id is duplicated`);
    }
    seenComponentIds.add(component.component_id);
    if (!isOneOf(component.classification, [
      "EXACT_AGREEMENT",
      "TYPE_DISAGREEMENT",
      "BOUNDARY_DISAGREEMENT",
      "SOL_ONLY",
      "HUMAN_ONLY",
    ])) {
      errors.push(`${componentPath}.classification is invalid`);
    }
    const human = component.human_endpoint_ids.map((id) => humanById.get(id));
    const sol = component.sol_endpoint_ids.map((id) => solById.get(id));
    if (human.some((item) => !item)) {
      errors.push(`${componentPath} references an unknown human endpoint`);
    }
    if (sol.some((item) => !item)) {
      errors.push(`${componentPath} references an unknown Sol endpoint`);
    }
    for (const id of component.human_endpoint_ids) {
      if (coveredHuman.has(id)) {
        errors.push(`${componentPath} repeats human endpoint ${id}`);
      }
      coveredHuman.add(id);
    }
    for (const id of component.sol_endpoint_ids) {
      if (coveredSol.has(id)) {
        errors.push(`${componentPath} repeats Sol endpoint ${id}`);
      }
      coveredSol.add(id);
    }
    const expected = classifyComponent(
      human.filter((item): item is AdjudicationHumanEndpoint => Boolean(item)),
      sol.filter((item): item is AdjudicationSolEndpoint => Boolean(item)),
    );
    if (component.classification !== expected) {
      errors.push(`${componentPath}.classification is inconsistent (expected ${expected})`);
    }
    if (
      (component.classification === "EXACT_AGREEMENT" ||
        component.classification === "TYPE_DISAGREEMENT") &&
      (component.human_endpoint_ids.length !== 1 || component.sol_endpoint_ids.length !== 1)
    ) {
      errors.push(`${componentPath} must contain exactly one span per side`);
    }
  });
  for (const endpoint of record.human_endpoints) {
    if (!coveredHuman.has(endpoint.endpoint_id)) {
      errors.push(`${path} human endpoint ${endpoint.endpoint_id} is not covered by a component`);
    }
  }
  for (const endpoint of record.sol_endpoints) {
    if (!coveredSol.has(endpoint.endpoint_id)) {
      errors.push(`${path} Sol endpoint ${endpoint.endpoint_id} is not covered by a component`);
    }
  }
}

function computeTotals(records: AdjudicationRecord[]): AdjudicationTotals {
  const totals: AdjudicationTotals = {
    windows: records.length,
    components: 0,
    exact_agreements: 0,
    type_disagreements: 0,
    boundary_disagreements: 0,
    sol_only: 0,
    human_only: 0,
    human_endpoints: 0,
    sol_endpoints: 0,
  };
  for (const record of records) {
    totals.components += record.components.length;
    totals.human_endpoints += record.human_endpoints.length;
    totals.sol_endpoints += record.sol_endpoints.length;
    for (const component of record.components) {
      switch (component.classification) {
        case "EXACT_AGREEMENT":
          totals.exact_agreements += 1;
          break;
        case "TYPE_DISAGREEMENT":
          totals.type_disagreements += 1;
          break;
        case "BOUNDARY_DISAGREEMENT":
          totals.boundary_disagreements += 1;
          break;
        case "SOL_ONLY":
          totals.sol_only += 1;
          break;
        case "HUMAN_ONLY":
          totals.human_only += 1;
          break;
      }
    }
  }
  return totals;
}

/**
 * Build-time sanitizer: validates the generated adjudication packet and returns
 * only the exact client-facing payload.  The raw packet's canonical self-hash
 * is exposed as `adjudication_packet_sha256`.
 */
export function sanitizeAdjudicationPacket(raw: unknown): AdjudicationPayload {
  const errors: string[] = [];
  if (!isRecordObject(raw)) {
    throw new Error("phase2j adjudication packet must be a JSON object");
  }
  if (!hasExactKeys(raw, PACKET_ENVELOPE_KEYS)) {
    throw new Error("phase2j adjudication packet envelope is invalid");
  }
  if (raw.schema_version !== ADJUDICATION_PACKET_SCHEMA_VERSION) {
    throw new Error(`schema_version must be ${ADJUDICATION_PACKET_SCHEMA_VERSION}`);
  }
  if (raw.adjudication_version !== ADJUDICATION_VERSION) {
    throw new Error(`adjudication_version must be ${ADJUDICATION_VERSION}`);
  }
  if (raw.annotation_version !== "phase2j-endpoint-annotation-v1") {
    throw new Error("phase2j adjudication annotation_version is unsupported");
  }
  if (raw.packet_schema_version !== "phase2j-endpoint-annotation-packet-v1") {
    throw new Error("phase2j adjudication packet_schema_version is unsupported");
  }
  if (raw.visibility_gate !== "SOL_VISIBLE_FOR_ADJUDICATION") {
    throw new Error("phase2j adjudication visibility gate is invalid");
  }
  if (typeof raw.purpose !== "string" || raw.purpose.length === 0) {
    throw new Error("phase2j adjudication purpose must be a non-empty string");
  }
  requireSha256(raw.content_sha256, "content_sha256", errors);
  requireSha256(raw.packet_sha256, "packet_sha256", errors);
  requireSha256(raw.human_session_sha256, "human_session_sha256", errors);
  requireSha256(raw.sol_review_sha256, "sol_review_sha256", errors);
  if (raw.human_session_schema_version !== "phase2j-review-session-v1") {
    errors.push("human_session_schema_version must be phase2j-review-session-v1");
  }
  if (raw.sol_review_schema_version !== "phase2j-sol-parallel-review-v1") {
    errors.push("sol_review_schema_version must be phase2j-sol-parallel-review-v1");
  }
  const forbidden = findAdjudicationForbiddenFields(raw, ADJUDICATION_FORBIDDEN_KEYS);
  if (forbidden.length > 0) {
    errors.push(...forbidden.map((problem) => `packet ${problem}`));
  }
  if (!isRecordObject(raw.totals) || !hasExactKeys(raw.totals, TOTALS_KEYS)) {
    throw new Error("phase2j adjudication totals are invalid");
  }
  if (!Array.isArray(raw.records) || raw.records.length !== 30) {
    throw new Error("phase2j adjudication packet must contain exactly 30 records");
  }

  const records: AdjudicationRecord[] = [];
  const seenWindowIds = new Set<string>();
  raw.records.forEach((rawRecord, index) => {
    const path = `records[${index}]`;
    if (!isRecordObject(rawRecord) || !hasExactKeys(rawRecord, RECORD_KEYS)) {
      errors.push(`${path} is invalid`);
      return;
    }
    if (rawRecord.record_index !== index + 1) {
      errors.push(`${path}.record_index must be ${index + 1}`);
    }
    const windowId = requireString(rawRecord.window_id, `${path}.window_id`, errors);
    if (seenWindowIds.has(windowId)) {
      errors.push(`${path} window_id is duplicated`);
    }
    seenWindowIds.add(windowId);
    requireString(rawRecord.source_group_id, `${path}.source_group_id`, errors);
    const bronzeText = requireString(rawRecord.bronze_text, `${path}.bronze_text`, errors);
    requireSha256(rawRecord.bronze_text_sha256, `${path}.bronze_text_sha256`, errors);
    if (rawRecord.bronze_char_length !== bronzeText.length) {
      errors.push(`${path}.bronze_char_length does not match bronze_text`);
    }
    const tokens = validateTokenTable(rawRecord.tokens, bronzeText, windowId, errors);
    if (!isOneOf(rawRecord.human_outcome, ["CLEAN", "AMBIGUOUS", "EXCLUDED"])) {
      errors.push(`${path}.human_outcome is invalid`);
    }
    const humanEndpoints: AdjudicationHumanEndpoint[] = [];
    const solEndpoints: AdjudicationSolEndpoint[] = [];
    const humanEndpointIds = new Set<string>();
    const solEndpointIds = new Set<string>();
    if (!Array.isArray(rawRecord.human_endpoints)) {
      errors.push(`${path}.human_endpoints must be an array`);
    } else {
      rawRecord.human_endpoints.forEach((rawEndpoint, endpointIndex) => {
        const endpointPath = `${path}.human_endpoints[${endpointIndex}]`;
        if (!isRecordObject(rawEndpoint) || !hasExactKeys(rawEndpoint, HUMAN_ENDPOINT_KEYS)) {
          errors.push(`${endpointPath} is invalid`);
          return;
        }
        const endpoint = validateSpanEndpoint(
          rawEndpoint, { ...recordShim(bronzeText, tokens, windowId), tokens } as AdjudicationRecord,
          endpointPath,
          errors,
          false,
        ) as AdjudicationHumanEndpoint | null;
        if (endpoint) {
          if (humanEndpointIds.has(endpoint.endpoint_id)) {
            errors.push(`${endpointPath}.endpoint_id is duplicated`);
          }
          humanEndpointIds.add(endpoint.endpoint_id);
          humanEndpoints.push(endpoint);
        }
      });
      validateNoInternalOverlap(humanEndpoints, `${path}.human_endpoints`, errors);
    }
    if (!Array.isArray(rawRecord.sol_endpoints)) {
      errors.push(`${path}.sol_endpoints must be an array`);
    } else {
      rawRecord.sol_endpoints.forEach((rawEndpoint, endpointIndex) => {
        const endpointPath = `${path}.sol_endpoints[${endpointIndex}]`;
        if (!isRecordObject(rawEndpoint) || !hasExactKeys(rawEndpoint, SOL_ENDPOINT_KEYS)) {
          errors.push(`${endpointPath} is invalid`);
          return;
        }
        if (
          rawEndpoint.sol_ambiguity_state !== null &&
          typeof rawEndpoint.sol_ambiguity_state !== "string"
        ) {
          errors.push(`${endpointPath}.sol_ambiguity_state must be null or a string`);
        }
        if (
          rawEndpoint.sol_rationale !== null &&
          typeof rawEndpoint.sol_rationale !== "string"
        ) {
          errors.push(`${endpointPath}.sol_rationale must be null or a string`);
        }
        const endpoint = validateSpanEndpoint(
          rawEndpoint, { ...recordShim(bronzeText, tokens, windowId), tokens } as AdjudicationRecord,
          endpointPath,
          errors,
          true,
        ) as AdjudicationSolEndpoint | null;
        if (endpoint) {
          if (solEndpointIds.has(endpoint.endpoint_id)) {
            errors.push(`${endpointPath}.endpoint_id is duplicated`);
          }
          solEndpointIds.add(endpoint.endpoint_id);
          solEndpoints.push({
            ...endpoint,
            sol_ambiguity_state: rawEndpoint.sol_ambiguity_state as string | null,
            sol_rationale: rawEndpoint.sol_rationale as string | null,
          });
        }
      });
      validateNoInternalOverlap(solEndpoints, `${path}.sol_endpoints`, errors);
    }
    const record: AdjudicationRecord = {
      record_index: rawRecord.record_index as number,
      window_id: windowId,
      source_group_id: rawRecord.source_group_id as string,
      bronze_text: bronzeText,
      bronze_text_sha256: rawRecord.bronze_text_sha256 as string,
      bronze_char_length: rawRecord.bronze_char_length as number,
      tokens,
      human_outcome: rawRecord.human_outcome as AdjudicationRecord["human_outcome"],
      human_endpoints: humanEndpoints,
      sol_endpoints: solEndpoints,
      components: rawRecord.components as AdjudicationComponent[],
    };
    if (!Array.isArray(rawRecord.components)) {
      errors.push(`${path}.components must be an array`);
    } else {
      rawRecord.components.forEach((rawComponent, componentIndex) => {
        const componentPath = `${path}.components[${componentIndex}]`;
        if (!isRecordObject(rawComponent) || !hasExactKeys(rawComponent, COMPONENT_KEYS)) {
          errors.push(`${componentPath} is invalid`);
          return;
        }
        if (!Array.isArray(rawComponent.human_endpoint_ids)) {
          errors.push(`${componentPath}.human_endpoint_ids must be an array`);
        }
        if (!Array.isArray(rawComponent.sol_endpoint_ids)) {
          errors.push(`${componentPath}.sol_endpoint_ids must be an array`);
        }
        record.components[componentIndex] = {
          component_id: rawComponent.component_id as string,
          classification: rawComponent.classification as AdjudicationComponentClass,
          human_endpoint_ids: rawComponent.human_endpoint_ids as string[],
          sol_endpoint_ids: rawComponent.sol_endpoint_ids as string[],
        };
      });
      validateComponents(record, path, errors);
    }
    records.push(record);
  });

  const totals = computeTotals(records);
  for (const key of Object.keys(totals) as Array<keyof AdjudicationTotals>) {
    if (raw.totals[key] !== totals[key]) {
      errors.push(`totals.${key} is inconsistent with the records`);
    }
  }
  if (errors.length > 0) {
    throw new Error(`phase2j adjudication packet is invalid: ${errors.slice(0, 8).join("; ")}`);
  }
  return {
    schema_version: ADJUDICATION_PACKET_SCHEMA_VERSION,
    adjudication_version: ADJUDICATION_VERSION,
    annotation_version: "phase2j-endpoint-annotation-v1",
    packet_schema_version: "phase2j-endpoint-annotation-packet-v1",
    packet_sha256: raw.packet_sha256 as string,
    adjudication_packet_sha256: raw.content_sha256 as string,
    human_session_sha256: raw.human_session_sha256 as string,
    sol_review_sha256: raw.sol_review_sha256 as string,
    totals,
    records,
  };
}

function recordShim(
  bronzeText: string,
  tokens: ReviewToken[],
  windowId: string,
): Pick<AdjudicationRecord, "bronze_text" | "tokens" | "window_id"> {
  return { bronze_text: bronzeText, tokens, window_id: windowId };
}

/** Default decisions: exact agreements are pre-resolved; everything else starts unresolved. */
export function buildAdjudicationState(payload: AdjudicationPayload): AdjudicationState {
  return {
    schema_version: ADJUDICATION_STATE_SCHEMA_VERSION,
    adjudication_packet_sha256: payload.adjudication_packet_sha256,
    reviewer_name: "",
    records: payload.records.map((record) => {
      const decisions: Record<string, ComponentDecision> = {};
      for (const component of record.components) {
        if (component.classification === "EXACT_AGREEMENT") {
          decisions[component.component_id] = { kind: "KEEP_HUMAN_SET" };
        }
      }
      return {
        record_index: record.record_index,
        window_id: record.window_id,
        outcome: record.human_outcome,
        note: "",
        decisions,
      };
    }),
  };
}

export function defaultDecisionFor(
  component: AdjudicationComponent,
): ComponentDecision | null {
  return component.classification === "EXACT_AGREEMENT"
    ? { kind: "KEEP_HUMAN_SET" }
    : null;
}

export function isComponentResolved(
  component: AdjudicationComponent,
  decision: ComponentDecision | undefined,
): boolean {
  return decision !== undefined;
}

export function componentDecisionAllowed(
  component: AdjudicationComponent,
  decision: ComponentDecision,
): boolean {
  switch (component.classification) {
    case "EXACT_AGREEMENT":
      return (
        decision.kind === "KEEP_HUMAN_SET" ||
        decision.kind === "DROP" ||
        decision.kind === "CUSTOM"
      );
    case "TYPE_DISAGREEMENT":
    case "BOUNDARY_DISAGREEMENT":
      return (
        decision.kind === "KEEP_HUMAN_SET" ||
        decision.kind === "KEEP_SOL_SET" ||
        decision.kind === "DROP" ||
        decision.kind === "CUSTOM"
      );
    case "SOL_ONLY":
      return (
        decision.kind === "KEEP_SOL_SET" ||
        decision.kind === "DROP" ||
        decision.kind === "CUSTOM"
      );
    case "HUMAN_ONLY":
      return (
        decision.kind === "KEEP_HUMAN_SET" ||
        decision.kind === "DROP" ||
        decision.kind === "CUSTOM"
      );
  }
}

export function windowResolvedByOutcome(stateRecord: AdjudicationRecordState): boolean {
  return stateRecord.outcome !== "CLEAN" && stateRecord.note.trim() !== "";
}

export function isWindowResolved(
  record: AdjudicationRecord,
  stateRecord: AdjudicationRecordState,
): boolean {
  if (windowResolvedByOutcome(stateRecord)) {
    return true;
  }
  if (stateRecord.outcome !== "CLEAN") {
    return false;
  }
  return record.components.every((component) => isComponentResolved(
    component,
    stateRecord.decisions[component.component_id],
  ));
}

export function unresolvedComponents(
  record: AdjudicationRecord,
  stateRecord: AdjudicationRecordState,
): AdjudicationComponent[] {
  return record.components.filter(
    (component) => !isComponentResolved(component, stateRecord.decisions[component.component_id]),
  );
}

export function summarizeAdjudicationProgress(
  payload: AdjudicationPayload,
  state: AdjudicationState,
): AdjudicationProgress {
  const progress: AdjudicationProgress = {
    windows: payload.records.length,
    resolved_windows: 0,
    components: payload.totals.components,
    resolved_components: 0,
    ambiguous: 0,
    excluded: 0,
  };
  payload.records.forEach((record, index) => {
    const stateRecord = state.records[index];
    if (!stateRecord) {
      return;
    }
    if (stateRecord.outcome === "AMBIGUOUS") {
      progress.ambiguous += 1;
    } else if (stateRecord.outcome === "EXCLUDED") {
      progress.excluded += 1;
    }
    if (isWindowResolved(record, stateRecord)) {
      progress.resolved_windows += 1;
    }
    for (const component of record.components) {
      if (isComponentResolved(component, stateRecord.decisions[component.component_id])) {
        progress.resolved_components += 1;
      }
    }
  });
  return progress;
}

/** Explicit per-window action: resolve everything to the human Pass A choices. */
export function keepPassAChoices(
  record: AdjudicationRecord,
  stateRecord: AdjudicationRecordState,
): AdjudicationRecordState {
  const decisions: Record<string, ComponentDecision> = {};
  for (const component of record.components) {
    if (component.classification === "EXACT_AGREEMENT") {
      decisions[component.component_id] = { kind: "KEEP_HUMAN_SET" };
    } else if (component.human_endpoint_ids.length > 0) {
      decisions[component.component_id] = { kind: "KEEP_HUMAN_SET" };
    } else {
      decisions[component.component_id] = { kind: "DROP" };
    }
  }
  return { ...stateRecord, decisions };
}

function customSpanValid(
  record: AdjudicationRecord,
  decision: Extract<ComponentDecision, { kind: "CUSTOM" }>,
): boolean {
  return (
    Number.isInteger(decision.token_start) &&
    Number.isInteger(decision.token_end) &&
    decision.token_start >= 0 &&
    decision.token_end < record.tokens.length &&
    decision.token_start <= decision.token_end
  );
}

function endpointFromComponentChoice(
  record: AdjudicationRecord,
  component: AdjudicationComponent,
  decision: ComponentDecision,
): Array<{ endpoint: AdjudicationHumanEndpoint | AdjudicationSolEndpoint; provenance: ResolvedEndpointProvenance }> {
  switch (decision.kind) {
    case "KEEP_HUMAN_SET": {
      const shared = component.classification === "EXACT_AGREEMENT";
      return component.human_endpoint_ids.flatMap((id) => {
        const endpoint = record.human_endpoints.find((item) => item.endpoint_id === id);
        return endpoint
          ? [{ endpoint, provenance: (shared ? "SHARED" : "HUMAN") as ResolvedEndpointProvenance }]
          : [];
      });
    }
    case "KEEP_SOL_SET":
      return component.sol_endpoint_ids.flatMap((id) => {
        const endpoint = record.sol_endpoints.find((item) => item.endpoint_id === id);
        return endpoint ? [{ endpoint, provenance: "SOL" as const }] : [];
      });
    case "DROP":
      return [];
    case "CUSTOM":
      return [{
        endpoint: {
          endpoint_id: "custom",
          exact_bronze_text: record.bronze_text.slice(
            record.tokens[decision.token_start].start,
            record.tokens[decision.token_end].end,
          ),
          char_start: record.tokens[decision.token_start].start,
          char_end: record.tokens[decision.token_end].end,
          token_start: decision.token_start,
          token_end: decision.token_end,
          node_type: decision.node_type,
        },
        provenance: "CUSTOM" as const,
      }];
  }
}

/**
 * Derive the resolved endpoint set for one window, allocating deterministic
 * endpoint ids.  Returns errors instead of throwing so the client can surface
 * exactly what blocks export.
 */
export function deriveResolvedEndpoints(
  record: AdjudicationRecord,
  stateRecord: AdjudicationRecordState,
): { endpoints: ResolvedEndpoint[]; errors: string[] } {
  const errors: string[] = [];
  if (stateRecord.outcome === "EXCLUDED") {
    return { endpoints: [], errors };
  }
  const candidates: Array<{
    endpoint: AdjudicationHumanEndpoint | AdjudicationSolEndpoint;
    component_id: string;
    provenance: ResolvedEndpointProvenance;
  }> = [];
  for (const component of record.components) {
    const decision = stateRecord.decisions[component.component_id];
    if (stateRecord.outcome === "AMBIGUOUS" && !decision) {
      continue;
    }
    if (!decision) {
      errors.push(`component ${component.component_id} is unresolved`);
      continue;
    }
    if (!componentDecisionAllowed(component, decision)) {
      errors.push(`component ${component.component_id} has an invalid decision for ${component.classification}`);
      continue;
    }
    if (decision.kind === "CUSTOM" && !customSpanValid(record, decision)) {
      errors.push(`component ${component.component_id} has an invalid custom span`);
      continue;
    }
    for (const item of endpointFromComponentChoice(record, component, decision)) {
      if (item.endpoint.node_type === null) {
        errors.push(
          `component ${component.component_id} keeps a Sol endpoint with no type; choose a type first`,
        );
        continue;
      }
      candidates.push({
        endpoint: item.endpoint,
        component_id: component.component_id,
        provenance: item.provenance,
      });
    }
  }
  candidates.sort((left, right) => {
    const leftEndpoint = left.endpoint;
    const rightEndpoint = right.endpoint;
    if (leftEndpoint.char_start !== rightEndpoint.char_start) {
      return leftEndpoint.char_start - rightEndpoint.char_start;
    }
    if (leftEndpoint.char_end !== rightEndpoint.char_end) {
      return leftEndpoint.char_end - rightEndpoint.char_end;
    }
    return left.component_id.localeCompare(right.component_id);
  });
  for (let leftIndex = 0; leftIndex < candidates.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < candidates.length; rightIndex += 1) {
      if (spansOverlap(candidates[leftIndex].endpoint, candidates[rightIndex].endpoint)) {
        errors.push(
          `resolved endpoints overlap: ${candidates[leftIndex].endpoint.exact_bronze_text} vs ${candidates[rightIndex].endpoint.exact_bronze_text}`,
        );
      }
    }
  }
  const endpoints = candidates.map((candidate, index) => ({
    endpoint_id: `p2j:adjudicate:${record.window_id}:ep:${String(index + 1).padStart(4, "0")}`,
    component_id: candidate.component_id,
    exact_bronze_text: candidate.endpoint.exact_bronze_text,
    char_start: candidate.endpoint.char_start,
    char_end: candidate.endpoint.char_end,
    token_start: candidate.endpoint.token_start,
    token_end: candidate.endpoint.token_end,
    node_type: candidate.endpoint.node_type as EndpointType,
    provenance_source: candidate.provenance,
  }));
  return { endpoints, errors };
}

function decisionEntry(
  component: AdjudicationComponent,
  decision: ComponentDecision | undefined,
  stateRecord: AdjudicationRecordState,
): ComponentExportEntry {
  if (stateRecord.outcome === "EXCLUDED") {
    return {
      component_id: component.component_id,
      classification: component.classification,
      decision: null,
      resolved_by: "WINDOW_EXCLUDED",
    };
  }
  if (stateRecord.outcome === "AMBIGUOUS" && !decision) {
    return {
      component_id: component.component_id,
      classification: component.classification,
      decision: null,
      resolved_by: "WINDOW_AMBIGUOUS",
    };
  }
  if (!decision) {
    // Export is blocked by the caller's error list; this placeholder never
    // ships and is only needed to keep the record builder total.
    return {
      component_id: component.component_id,
      classification: component.classification,
      decision: null,
      resolved_by: "WINDOW_AMBIGUOUS",
    };
  }
  if (component.classification === "EXACT_AGREEMENT" && decision.kind === "KEEP_HUMAN_SET") {
    return {
      component_id: component.component_id,
      classification: component.classification,
      decision,
      resolved_by: "PRE_RESOLVED",
    };
  }
  const resolvedByMap: Record<
    "KEEP_HUMAN_SET" | "KEEP_SOL_SET" | "DROP" | "CUSTOM",
    ComponentExportEntry["resolved_by"]
  > = {
    KEEP_HUMAN_SET: "HUMAN_SET",
    KEEP_SOL_SET: "SOL_SET",
    DROP: "DROP",
    CUSTOM: "CUSTOM",
  };
  const resolvedBy = resolvedByMap[decision.kind];
  return {
    component_id: component.component_id,
    classification: component.classification,
    decision,
    resolved_by: resolvedBy,
  };
}

function exportRecordFromState(
  record: AdjudicationRecord,
  stateRecord: AdjudicationRecordState,
): { record: AdjudicationExportRecord; errors: string[] } {
  const errors: string[] = [];
  if (stateRecord.outcome === "CLEAN") {
    for (const component of record.components) {
      if (!stateRecord.decisions[component.component_id]) {
        errors.push(
          `window ${record.window_id} component ${component.component_id} is unresolved`,
        );
      }
    }
  } else if (stateRecord.note.trim() === "") {
    errors.push(`window ${record.window_id} requires a note for ${stateRecord.outcome}`);
  }
  const derived = deriveResolvedEndpoints(record, stateRecord);
  errors.push(...derived.errors);
  const components = record.components.map((component) =>
    decisionEntry(component, stateRecord.decisions[component.component_id], stateRecord),
  );
  return {
    record: {
      record_index: record.record_index,
      window_id: record.window_id,
      outcome: stateRecord.outcome,
      note: stateRecord.note,
      components,
      resolved_endpoints: derived.endpoints,
    },
    errors,
  };
}

/**
 * Build a validated REVIEW MATERIAL export from the current adjudication
 * state.  CLEAN windows must resolve every component; AMBIGUOUS/EXCLUDED
 * windows require a note; EXCLUDED clears endpoints.
 */
export function buildAdjudicationExport(
  payload: AdjudicationPayload,
  state: AdjudicationState,
  reviewerName: string,
  exportedAt: string,
  auditChecks: AuditChecks,
): ExportResult {
  const errors: string[] = [];
  if (reviewerName.trim() === "") {
    errors.push("reviewer_name is required before export");
  }
  const validatedAuditChecks = validateAuditChecks(auditChecks, "audit_checks", errors);
  const records: AdjudicationExportRecord[] = [];
  payload.records.forEach((record, index) => {
    const stateRecord = state.records[index];
    if (!stateRecord) {
      errors.push(`missing adjudication state for record ${index + 1}`);
      return;
    }
    const built = exportRecordFromState(record, stateRecord);
    errors.push(...built.errors);
    records.push(built.record);
  });
  if (errors.length > 0) {
    return { ok: false, errors };
  }
  const exportValue: AdjudicationExport = {
    schema_version: ADJUDICATION_EXPORT_SCHEMA_VERSION,
    adjudication_version: ADJUDICATION_VERSION,
    packet_schema_version: "phase2j-endpoint-annotation-packet-v1",
    adjudication_packet_sha256: payload.adjudication_packet_sha256,
    packet_sha256: payload.packet_sha256,
    human_session_sha256: payload.human_session_sha256,
    sol_review_sha256: payload.sol_review_sha256,
    status_label: "REVIEW_MATERIAL",
    reviewer_name: reviewerName.trim(),
    exported_at: exportedAt,
    audit_checks: validatedAuditChecks as AuditChecks,
    records,
  };
  const validation = validateAdjudicationExport(exportValue, payload);
  if (!validation.ok) {
    return { ok: false, errors: validation.errors };
  }
  return { ok: true, export: exportValue };
}

function validateStateRecord(
  rawRecord: unknown,
  reference: AdjudicationRecord,
  index: number,
  errors: string[],
): AdjudicationRecordState | null {
  const path = `records[${index}]`;
  if (!isRecordObject(rawRecord) || !hasExactKeys(rawRecord, STATE_RECORD_KEYS)) {
    errors.push(`${path} is invalid`);
    return null;
  }
  if (rawRecord.record_index !== index + 1) {
    errors.push(`${path}.record_index must be ${index + 1}`);
  }
  if (rawRecord.window_id !== reference.window_id) {
    errors.push(`${path}.window_id does not match the adjudication packet`);
  }
  if (!isOneOf(rawRecord.outcome, ["CLEAN", "AMBIGUOUS", "EXCLUDED"])) {
    errors.push(`${path}.outcome is invalid`);
  }
  const note = requireString(rawRecord.note, `${path}.note`, errors);
  const outcome = asOneOf(rawRecord.outcome, ["CLEAN", "AMBIGUOUS", "EXCLUDED"], "CLEAN");
  if (outcome !== "CLEAN" && note.trim() === "") {
    errors.push(`${path} requires a note for ${outcome}`);
  }
  const decisions: Record<string, ComponentDecision> = {};
  if (!isRecordObject(rawRecord.decisions)) {
    errors.push(`${path}.decisions must be an object`);
  } else {
    for (const component of reference.components) {
      const rawDecision = rawRecord.decisions[component.component_id];
      if (rawDecision === undefined) {
        continue;
      }
      const decision = parseDecision(rawDecision, `${path}.decisions.${component.component_id}`, errors);
      if (decision && componentDecisionAllowed(component, decision)) {
        decisions[component.component_id] = decision;
      } else if (decision) {
        errors.push(
          `${path}.decisions.${component.component_id} is not allowed for ${component.classification}`,
        );
      }
    }
  }
  if (isRecordObject(rawRecord.decisions)) {
    for (const key of Object.keys(rawRecord.decisions)) {
      if (!reference.components.some((component) => component.component_id === key)) {
        errors.push(`${path}.decisions.${key} is not a component in this window`);
      }
    }
  }
  return {
    record_index: index + 1,
    window_id: reference.window_id,
    outcome,
    note,
    decisions,
  };
}

function parseDecision(
  value: unknown,
  path: string,
  errors: string[],
): ComponentDecision | null {
  if (!isRecordObject(value)) {
    errors.push(`${path} must be an object`);
    return null;
  }
  const kind = value.kind;
  if (!isOneOf(kind, ["KEEP_HUMAN_SET", "KEEP_SOL_SET", "DROP", "CUSTOM"])) {
    errors.push(`${path}.kind is invalid`);
    return null;
  }
  if (kind === "CUSTOM") {
    const tokenStart = requireInt(value.token_start, `${path}.token_start`, errors);
    const tokenEnd = requireInt(value.token_end, `${path}.token_end`, errors);
    const nodeType = value.node_type;
    if (!isOneOf(nodeType, ENDPOINT_TYPES)) {
      errors.push(`${path}.node_type is invalid`);
      return null;
    }
    if (tokenStart === null || tokenEnd === null) {
      return null;
    }
    return { kind: "CUSTOM", token_start: tokenStart, token_end: tokenEnd, node_type: nodeType };
  }
  if (kind === "KEEP_HUMAN_SET") {
    return { kind: "KEEP_HUMAN_SET" };
  }
  if (kind === "KEEP_SOL_SET") {
    return { kind: "KEEP_SOL_SET" };
  }
  return { kind: "DROP" };
}

function validateExportRecord(
  rawRecord: unknown,
  reference: AdjudicationRecord,
  index: number,
  errors: string[],
): AdjudicationExportRecord | null {
  const path = `records[${index}]`;
  if (!isRecordObject(rawRecord) || !hasExactKeys(rawRecord, EXPORT_RECORD_KEYS)) {
    errors.push(`${path} is invalid`);
    return null;
  }
  if (rawRecord.record_index !== index + 1) {
    errors.push(`${path}.record_index must be ${index + 1}`);
  }
  if (rawRecord.window_id !== reference.window_id) {
    errors.push(`${path}.window_id does not match the adjudication packet`);
  }
  const outcome = asOneOf(rawRecord.outcome, ["CLEAN", "AMBIGUOUS", "EXCLUDED"], "CLEAN");
  if (!isOneOf(rawRecord.outcome, ["CLEAN", "AMBIGUOUS", "EXCLUDED"])) {
    errors.push(`${path}.outcome is invalid`);
  }
  const note = requireString(rawRecord.note, `${path}.note`, errors);
  if (outcome !== "CLEAN" && note.trim() === "") {
    errors.push(`${path} requires a note for ${outcome}`);
  }
  const decisions: Record<string, ComponentDecision> = {};
  const components: ComponentExportEntry[] = [];
  if (!Array.isArray(rawRecord.components) ||
      rawRecord.components.length !== reference.components.length) {
    errors.push(`${path}.components must cover every adjudication component`);
  } else {
    rawRecord.components.forEach((rawEntry, componentIndex) => {
      const entryPath = `${path}.components[${componentIndex}]`;
      const referenceComponent = reference.components[componentIndex];
      if (!isRecordObject(rawEntry) || !hasExactKeys(rawEntry, EXPORT_COMPONENT_KEYS)) {
        errors.push(`${entryPath} is invalid`);
        return;
      }
      if (rawEntry.component_id !== referenceComponent.component_id) {
        errors.push(`${entryPath}.component_id does not match the packet`);
      }
      if (rawEntry.classification !== referenceComponent.classification) {
        errors.push(`${entryPath}.classification does not match the packet`);
      }
      const resolvedBy = asOneOf(rawEntry.resolved_by, [
        "PRE_RESOLVED",
        "HUMAN_SET",
        "SOL_SET",
        "DROP",
        "CUSTOM",
        "WINDOW_AMBIGUOUS",
        "WINDOW_EXCLUDED",
      ], "DROP");
      if (!isOneOf(rawEntry.resolved_by, [
        "PRE_RESOLVED",
        "HUMAN_SET",
        "SOL_SET",
        "DROP",
        "CUSTOM",
        "WINDOW_AMBIGUOUS",
        "WINDOW_EXCLUDED",
      ])) {
        errors.push(`${entryPath}.resolved_by is invalid`);
      }
      const rawDecision = rawEntry.decision;
      const decision = rawDecision === null
        ? null
        : parseDecision(rawDecision, `${entryPath}.decision`, errors);
      if (outcome === "CLEAN") {
        if (!decision) {
          errors.push(`${entryPath} is unresolved in a CLEAN window`);
        } else if (!componentDecisionAllowed(referenceComponent, decision)) {
          errors.push(`${entryPath}.decision is not allowed for ${referenceComponent.classification}`);
        }
      }
      if (referenceComponent.classification === "EXACT_AGREEMENT") {
        if (outcome === "EXCLUDED") {
          if (decision !== null || resolvedBy !== "WINDOW_EXCLUDED") {
            errors.push(`${entryPath} excluded windows must clear exact agreements`);
          }
        } else if (decision === null) {
          if (outcome !== "AMBIGUOUS" || resolvedBy !== "WINDOW_AMBIGUOUS") {
            errors.push(
              `${entryPath} exact agreements must be kept, edited, dropped, or marked window-ambiguous`,
            );
          }
        } else {
          const expectedResolvedBy =
            decision.kind === "KEEP_HUMAN_SET" ? "PRE_RESOLVED" : decision.kind;
          if (resolvedBy !== expectedResolvedBy) {
            errors.push(`${entryPath} resolved_by must be ${expectedResolvedBy} for ${decision.kind}`);
          }
        }
      } else if (outcome === "CLEAN" && resolvedBy === "PRE_RESOLVED") {
        errors.push(`${entryPath} only exact agreements may be pre-resolved`);
      }
      if (decision) {
        decisions[referenceComponent.component_id] = decision;
      }
      components.push({
        component_id: referenceComponent.component_id,
        classification: referenceComponent.classification,
        decision,
        resolved_by: resolvedBy,
      });
    });
  }
  const stateRecord: AdjudicationRecordState = {
    record_index: index + 1,
    window_id: reference.window_id,
    outcome,
    note,
    decisions,
  };
  const derived = deriveResolvedEndpoints(reference, stateRecord);
  errors.push(...derived.errors);
  const rawEndpoints = rawRecord.resolved_endpoints;
  if (!Array.isArray(rawEndpoints)) {
    errors.push(`${path}.resolved_endpoints must be an array`);
    return {
      record_index: index + 1,
      window_id: reference.window_id,
      outcome,
      note,
      components,
      resolved_endpoints: [],
    };
  }
  rawEndpoints.forEach((rawEndpoint, endpointIndex) => {
    const endpointPath = `${path}.resolved_endpoints[${endpointIndex}]`;
    if (!isRecordObject(rawEndpoint) || !hasExactKeys(rawEndpoint, RESOLVED_ENDPOINT_KEYS)) {
      errors.push(`${endpointPath} is invalid`);
      return;
    }
    if (!isOneOf(rawEndpoint.node_type, ENDPOINT_TYPES)) {
      errors.push(`${endpointPath}.node_type is invalid`);
    }
    if (!isOneOf(rawEndpoint.provenance_source, ["HUMAN", "SOL", "SHARED", "CUSTOM"])) {
      errors.push(`${endpointPath}.provenance_source is invalid`);
    }
    if (!reference.components.some((component) => component.component_id === rawEndpoint.component_id)) {
      errors.push(`${endpointPath}.component_id does not belong to this window`);
    }
    const tokenStart = requireInt(rawEndpoint.token_start, `${endpointPath}.token_start`, errors);
    const tokenEnd = requireInt(rawEndpoint.token_end, `${endpointPath}.token_end`, errors);
    const charStart = requireInt(rawEndpoint.char_start, `${endpointPath}.char_start`, errors);
    const charEnd = requireInt(rawEndpoint.char_end, `${endpointPath}.char_end`, errors);
    if (
      tokenStart !== null &&
      tokenEnd !== null &&
      tokenStart >= 0 &&
      tokenEnd < reference.tokens.length &&
      reference.tokens[tokenStart].start === charStart &&
      reference.tokens[tokenEnd].end === charEnd
    ) {
      const exact = requireString(rawEndpoint.exact_bronze_text, `${endpointPath}.exact_bronze_text`, errors);
      if (reference.bronze_text.slice(charStart as number, charEnd as number) !== exact) {
        errors.push(`${endpointPath}.exact_bronze_text is not an exact Bronze slice`);
      }
    } else if (tokenStart !== null || tokenEnd !== null || charStart !== null || charEnd !== null) {
      errors.push(`${endpointPath} token/char bounds are inconsistent`);
    }
  });
  const endpointObjects = rawEndpoints.filter(isRecordObject) as ResolvedEndpoint[];
  if (JSON.stringify(derived.endpoints) !== JSON.stringify(endpointObjects)) {
    errors.push(`${path}.resolved_endpoints do not match the derived endpoint set`);
  }
  for (let leftIndex = 0; leftIndex < endpointObjects.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < endpointObjects.length; rightIndex += 1) {
      if (spansOverlap(endpointObjects[leftIndex], endpointObjects[rightIndex])) {
        errors.push(`${path} resolved endpoints overlap`);
      }
    }
  }
  return {
    record_index: index + 1,
    window_id: reference.window_id,
    outcome,
    note,
    components,
    resolved_endpoints: endpointObjects,
  };
}

/** Strict import/autosave validation; returns a normalized adjudication state. */
export function validateAdjudicationState(
  input: unknown,
  payload: AdjudicationPayload,
): ValidationResult {
  const errors: string[] = [];
  if (!isRecordObject(input)) {
    return { ok: false, errors: ["adjudication state must be a JSON object"] };
  }
  if (!hasExactKeys(input, STATE_ENVELOPE_KEYS)) {
    errors.push("adjudication state envelope is invalid");
  }
  if (input.schema_version !== ADJUDICATION_STATE_SCHEMA_VERSION) {
    errors.push(`schema_version must be ${ADJUDICATION_STATE_SCHEMA_VERSION}`);
  }
  if (input.adjudication_packet_sha256 !== payload.adjudication_packet_sha256) {
    errors.push("adjudication_packet_sha256 does not match this adjudication packet");
  }
  requireString(input.reviewer_name, "reviewer_name", errors);
  if (!Array.isArray(input.records) || input.records.length !== payload.records.length) {
    errors.push(`records must contain exactly ${payload.records.length} windows`);
  }
  const records: AdjudicationRecordState[] = [];
  if (Array.isArray(input.records)) {
    input.records.forEach((rawRecord, index) => {
      const stateRecord = validateStateRecord(
        rawRecord, payload.records[index], index, errors,
      );
      if (stateRecord) {
        records.push(stateRecord);
      }
    });
  }
  const forbidden = findAdjudicationForbiddenFields(
    input, ADJUDICATION_EXPORT_FORBIDDEN_KEYS,
  );
  if (forbidden.length > 0) {
    errors.push(...forbidden.map((problem) => `state ${problem}`));
  }
  if (errors.length > 0) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    state: {
      schema_version: ADJUDICATION_STATE_SCHEMA_VERSION,
      adjudication_packet_sha256: payload.adjudication_packet_sha256,
      reviewer_name: input.reviewer_name as string,
      records,
    },
  };
}

/** Strict import validation for a previously exported adjudication file. */
export function validateAdjudicationExport(
  input: unknown,
  payload: AdjudicationPayload,
): ExportValidationResult {
  const errors: string[] = [];
  if (!isRecordObject(input)) {
    return { ok: false, errors: ["adjudication export must be a JSON object"] };
  }
  if (!hasExactKeys(input, EXPORT_ENVELOPE_KEYS)) {
    errors.push("adjudication export envelope is invalid");
  }
  if (input.schema_version !== ADJUDICATION_EXPORT_SCHEMA_VERSION) {
    errors.push(`schema_version must be ${ADJUDICATION_EXPORT_SCHEMA_VERSION}`);
  }
  if (input.adjudication_version !== ADJUDICATION_VERSION) {
    errors.push(`adjudication_version must be ${ADJUDICATION_VERSION}`);
  }
  if (input.packet_schema_version !== "phase2j-endpoint-annotation-packet-v1") {
    errors.push("packet_schema_version is unsupported");
  }
  if (input.adjudication_packet_sha256 !== payload.adjudication_packet_sha256) {
    errors.push("adjudication_packet_sha256 does not match this adjudication packet");
  }
  if (input.packet_sha256 !== payload.packet_sha256) {
    errors.push("packet_sha256 does not match the locked packet");
  }
  if (input.human_session_sha256 !== payload.human_session_sha256) {
    errors.push("human_session_sha256 does not match the adjudication packet");
  }
  if (input.sol_review_sha256 !== payload.sol_review_sha256) {
    errors.push("sol_review_sha256 does not match the adjudication packet");
  }
  if (input.status_label !== "REVIEW_MATERIAL") {
    errors.push("status_label must be REVIEW_MATERIAL");
  }
  const reviewerName = requireString(input.reviewer_name, "reviewer_name", errors);
  if (reviewerName.trim() === "") {
    errors.push("reviewer_name is required");
  }
  const exportedAt = input.exported_at;
  if (exportedAt !== null && typeof exportedAt !== "string") {
    errors.push("exported_at must be null or a string");
  }
  const auditChecks = validateAuditChecks(input.audit_checks, "audit_checks", errors);
  if (!Array.isArray(input.records) || input.records.length !== payload.records.length) {
    errors.push(`records must contain exactly ${payload.records.length} windows`);
  }
  const records: AdjudicationRecordState[] = [];
  if (Array.isArray(input.records)) {
    input.records.forEach((rawRecord, index) => {
      const reference = payload.records[index];
      if (!reference) {
        return;
      }
      const exportRecord = validateExportRecord(rawRecord, reference, index, errors);
      if (exportRecord) {
        const decisions: Record<string, ComponentDecision> = {};
        for (const entry of exportRecord.components) {
          if (entry.decision) {
            decisions[entry.component_id] = entry.decision;
          }
        }
        records.push({
          record_index: exportRecord.record_index,
          window_id: exportRecord.window_id,
          outcome: exportRecord.outcome,
          note: exportRecord.note,
          decisions,
        });
      }
    });
  }
  const forbidden = findAdjudicationForbiddenFields(
    input, ADJUDICATION_EXPORT_FORBIDDEN_KEYS,
  );
  if (forbidden.length > 0) {
    errors.push(...forbidden.map((problem) => `export ${problem}`));
  }
  if (errors.length > 0) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    state: {
      schema_version: ADJUDICATION_STATE_SCHEMA_VERSION,
      adjudication_packet_sha256: payload.adjudication_packet_sha256,
      reviewer_name: input.reviewer_name as string,
      records,
    },
    audit_checks: auditChecks as AuditChecks,
  };
}
