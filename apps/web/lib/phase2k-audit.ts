/**
 * Pure Phase 2K live transformation-audit review utilities.
 *
 * This module is browser-safe (no Node imports) and mirrors the authoritative
 * Python contract in pipeline/phase2k_contextual_reconstruction.py:
 *
 * - `TRANSFORMATION_AUDIT_SCHEMA_VERSION` / `phase2k-transformation-audit-packet-v2`
 * - `COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION` /
 *   `phase2k-completed-transformation-audit-v2`
 * - `build_transformation_audit`, `validate_transformation_audit_packet`,
 *   `_validate_transformation_audit_operation`, and
 *   `validate_completed_transformation_audits`
 *
 * The blank template is imported by the operator from a local file (never a
 * hard-coded server path) and is validated strictly before use.  Sessions are
 * bound to the template content hash plus the records hash, and the completed
 * packet is exported as-is for `scripts/finalize_phase2k_human_review.py`.
 * Nothing here ever fabricates a decision or attestation.
 */

export const TRANSFORMATION_AUDIT_SCHEMA_VERSION =
  "phase2k-transformation-audit-packet-v2";
export const COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION =
  "phase2k-completed-transformation-audit-v2";
export const RELEASE_GATE_AWAITING_REVIEW = "AWAITING_HUMAN_REVIEW";
export const RELEASE_GATE_REVIEWED = "REVIEWED";
export const SESSION_SCHEMA_VERSION = "phase2k-transformation-audit-session-v1";

export const AUDIT_CATEGORIES = [
  "mechanical_repairs",
  "contextual_repairs",
  "entity_bindings",
  "pronoun_bindings",
  "reference_bindings",
  "ability_bindings",
  "polished_statements",
] as const;

export type AuditCategory = (typeof AUDIT_CATEGORIES)[number];

export const AUDIT_OPERATION_KINDS = [
  "MECHANICAL_REPAIR",
  "CONTEXTUAL_REPAIR",
  "ENTITY_BINDING",
  "PRONOUN_BINDING",
  "REFERENCE_BINDING",
  "ABILITY_BINDING",
  "POLISHED_STATEMENT",
] as const;

export type AuditOperationKind = (typeof AUDIT_OPERATION_KINDS)[number];

export const CATEGORY_KINDS: Record<AuditCategory, AuditOperationKind> = {
  mechanical_repairs: "MECHANICAL_REPAIR",
  contextual_repairs: "CONTEXTUAL_REPAIR",
  entity_bindings: "ENTITY_BINDING",
  pronoun_bindings: "PRONOUN_BINDING",
  reference_bindings: "REFERENCE_BINDING",
  ability_bindings: "ABILITY_BINDING",
  polished_statements: "POLISHED_STATEMENT",
};

export const AUDIT_DECISIONS = ["APPROVE", "REJECT", "AMBIGUOUS"] as const;
export type AuditDecision = (typeof AUDIT_DECISIONS)[number];

/** Exact closed error taxonomy from the authoritative Python module. */
export const AUDIT_ERROR_TAXONOMY = [
  "ASR_REPAIR_CORRECT",
  "ASR_REPAIR_WRONG",
  "ASR_REPAIR_UNRESOLVED",
  "ENTITY_BIND_CORRECT",
  "ENTITY_BIND_WRONG",
  "ENTITY_BIND_UNRESOLVED",
  "ABILITY_OWNER_CORRECT",
  "ABILITY_OWNER_WRONG",
  "ABILITY_OWNER_UNRESOLVED",
  "PRONOUN_BIND_WRONG",
  "DISCOURSE_REFERENCE_UNRESOLVED",
  "CONTEXT_TOO_SHORT",
  "CONTEXT_EXPANDED_UNNECESSARILY",
  "UNCERTAINTY_ERASED",
  "NEGATION_CHANGED",
  "MODALITY_CHANGED",
  "CAUSALITY_INVENTED",
  "EVENT_INVENTED",
  "SOURCE_DETAIL_DROPPED",
  "OVERGENERALIZED",
  "OTHER",
] as const;

export const CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"] as const;

export const BINDING_SLOTS = [
  "principal_actors",
  "pronouns",
  "champion_identities",
  "ability_ownership",
  "core_action_event",
  "state_outcome",
  "condition",
  "consequence",
  "temporal_refs",
  "discourse_refs",
  "unresolved_asr",
] as const;

export const BINDING_STATUSES = [
  "RESOLVED",
  "UNKNOWN",
  "AMBIGUOUS",
  "MULTIPLE_CANDIDATES",
  "CONTEXT_INSUFFICIENT",
] as const;

export const POLISH_SUPPORT_MODES = [
  "UNCHANGED_EXACT",
  "EVIDENCE_PARAPHRASE",
  "RECONSTRUCTION_DERIVED",
] as const;

/** The six explicit statement attestations; never defaulted or fabricated. */
export const STATEMENT_ATTESTATION_FIELDS = [
  "supported",
  "uncertainty_preserved",
  "negation_preserved",
  "modality_preserved",
  "causality_invented",
  "source_detail_dropped",
] as const;

export type StatementAttestationField = (typeof STATEMENT_ATTESTATION_FIELDS)[number];

const TOP_LEVEL_KEYS = [
  "content_sha256",
  "schema_version",
  "purpose",
  "release_gate",
  "binding",
  "error_taxonomy",
  "decisions",
  "operation_kinds",
  "operation_map",
  "window_audits",
] as const;

const WINDOW_KEYS = [
  "window_id",
  "bronze_target",
  "operations",
  "first_failure",
  "first_reconstruction_failure",
] as const;

const BRONZE_TARGET_KEYS = [
  "text",
  "text_sha256",
  "source_absolute_start",
  "source_absolute_end",
] as const;

const OPERATION_MAP_KEYS = [
  "operation_id",
  "window_id",
  "category",
  "operation_kind",
  "ordinal",
] as const;

const FAILURE_BLOCK_KEYS = [
  "stage",
  "prompt_version",
  "response_schema_version",
  "error",
  "error_taxonomy",
] as const;

const REPAIR_KEYS = [
  "operation_id",
  "operation_kind",
  "repair_type",
  "confidence",
  "original_text",
  "replacement",
  "evidence_spans",
  "decision",
  "corrected_replacement",
  "error_taxonomy",
] as const;

const BINDING_KEYS = [
  "operation_id",
  "operation_kind",
  "binding_id",
  "slot",
  "mention",
  "resolved_candidate",
  "resolved_status",
  "evidence_spans",
  "human_resolvable_required",
  "decision",
  "error_taxonomy",
] as const;

const STATEMENT_KEYS = [
  "operation_id",
  "operation_kind",
  "statement_id",
  "text",
  "evidence_spans",
  "reconstruction_operation_ids",
  "support_mode",
  "unchanged_source_quote",
  "decision",
  "supported",
  "uncertainty_preserved",
  "negation_preserved",
  "modality_preserved",
  "causality_invented",
  "source_detail_dropped",
  "error_taxonomy",
] as const;

const BRONZE_SPAN_KEYS = [
  "target_local_start",
  "target_local_end",
  "source_absolute_start",
  "source_absolute_end",
  "text",
] as const;

const UNCHANGED_SOURCE_QUOTE_KEYS = BRONZE_SPAN_KEYS;

const SESSION_KEYS = [
  "schema_version",
  "template_sha256",
  "records_sha256",
  "operations",
] as const;

const SESSION_OPERATION_KEYS = [
  "operation_id",
  "decision",
  "corrected_replacement",
  "error_taxonomy",
  "supported",
  "uncertainty_preserved",
  "negation_preserved",
  "modality_preserved",
  "causality_invented",
  "source_detail_dropped",
] as const;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RepairOperation = {
  operation_id: string;
  operation_kind: "MECHANICAL_REPAIR" | "CONTEXTUAL_REPAIR";
  repair_type: string;
  confidence: string;
  original_text: string;
  replacement: string;
  evidence_spans: unknown[];
  decision: null;
  corrected_replacement: null;
  error_taxonomy: null;
};

export type BronzeSpan = {
  target_local_start: number;
  target_local_end: number;
  source_absolute_start: number;
  source_absolute_end: number;
  text: string;
};

export type BindingOperation = {
  operation_id: string;
  operation_kind:
    | "ENTITY_BINDING"
    | "PRONOUN_BINDING"
    | "REFERENCE_BINDING"
    | "ABILITY_BINDING";
  binding_id: string;
  slot: string;
  mention: BronzeSpan;
  resolved_candidate: string;
  resolved_status: string;
  evidence_spans: unknown[];
  human_resolvable_required: boolean;
  decision: null;
  error_taxonomy: null;
};

export type UnchangedSourceQuote = BronzeSpan;

export type StatementOperation = {
  operation_id: string;
  operation_kind: "POLISHED_STATEMENT";
  statement_id: string;
  text: string;
  evidence_spans: unknown[];
  reconstruction_operation_ids: string[];
  support_mode: string;
  unchanged_source_quote: UnchangedSourceQuote | null;
  decision: null;
  supported: null;
  uncertainty_preserved: null;
  negation_preserved: null;
  modality_preserved: null;
  causality_invented: null;
  source_detail_dropped: null;
  error_taxonomy: null;
};

export type AuditOperation = RepairOperation | BindingOperation | StatementOperation;

export type FailureBlock = {
  stage: string;
  prompt_version: string;
  response_schema_version: string;
  error: string;
  error_taxonomy: null;
};

export type BronzeTarget = {
  text: string;
  text_sha256: string;
  source_absolute_start: number;
  source_absolute_end: number;
};

export type WindowAudit = {
  window_id: string;
  bronze_target: BronzeTarget;
  operations: Record<AuditCategory, AuditOperation[]>;
  first_failure: FailureBlock | null;
  first_reconstruction_failure: FailureBlock | null;
};

export type OperationMapEntry = {
  operation_id: string;
  window_id: string;
  category: AuditCategory;
  operation_kind: AuditOperationKind;
  ordinal: number;
};

export type AuditTemplate = {
  content_sha256: string;
  schema_version: typeof TRANSFORMATION_AUDIT_SCHEMA_VERSION;
  purpose: string;
  release_gate: typeof RELEASE_GATE_AWAITING_REVIEW;
  binding: { records_sha256: string };
  error_taxonomy: readonly string[];
  decisions: readonly AuditDecision[];
  operation_kinds: readonly AuditOperationKind[];
  operation_map: Record<string, OperationMapEntry>;
  window_audits: WindowAudit[];
};

export type SessionOperation = {
  operation_id: string;
  decision: AuditDecision | null;
  corrected_replacement: string | null;
  error_taxonomy: string | null;
  supported: boolean | null;
  uncertainty_preserved: boolean | null;
  negation_preserved: boolean | null;
  modality_preserved: boolean | null;
  causality_invented: boolean | null;
  source_detail_dropped: boolean | null;
};

export type AuditSession = {
  schema_version: typeof SESSION_SCHEMA_VERSION;
  template_sha256: string;
  records_sha256: string;
  operations: SessionOperation[];
};

export type CompletedOperation = {
  operation_id: string;
  operation_kind: AuditOperationKind;
  decision: AuditDecision;
  error_taxonomy: string | null;
  corrected_replacement?: string | null;
  supported?: boolean;
  uncertainty_preserved?: boolean;
  negation_preserved?: boolean;
  modality_preserved?: boolean;
  causality_invented?: boolean;
  source_detail_dropped?: boolean;
};

export type CompletedAudit = {
  content_sha256: string;
  schema_version: typeof COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION;
  purpose: string;
  release_gate: typeof RELEASE_GATE_REVIEWED;
  binding: { records_sha256: string };
  error_taxonomy: string[];
  decisions: AuditDecision[];
  operation_kinds: AuditOperationKind[];
  operation_map: Record<string, OperationMapEntry>;
  window_audits: Array<{
    window_id: string;
    bronze_target: BronzeTarget;
    operations: Record<AuditCategory, CompletedOperation[]>;
    first_failure: FailureBlock | null;
    first_reconstruction_failure: FailureBlock | null;
  }>;
};

export type FlatOperation = {
  ordinal: number;
  operation_id: string;
  window_id: string;
  category: AuditCategory;
  operation_kind: AuditOperationKind;
  operation: AuditOperation;
  window: WindowAudit;
};

export type AuditProgress = {
  total: number;
  completed: number;
  remaining: number;
  by_category: Record<AuditCategory, { total: number; completed: number }>;
};

export type TemplateValidationResult =
  | { ok: true; template: AuditTemplate }
  | { ok: false; errors: string[] };

export type SessionValidationResult =
  | { ok: true; session: AuditSession }
  | { ok: false; errors: string[] };

export type CompletedAuditResult =
  | { ok: true; completed: CompletedAudit }
  | { ok: false; errors: string[] };

/** Sync or async SHA-256 digest from UTF-8 bytes to lowercase hex. */
export type Sha256Digest = (utf8: Uint8Array) => string | Promise<string>;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(
  value: Record<string, unknown>,
  keys: readonly string[],
): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => actual.includes(key));
}

function isHex64(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

function isBoolean(value: unknown): value is boolean {
  return typeof value === "boolean";
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

function isOneOf<T extends readonly string[]>(
  value: unknown,
  allowed: T,
): value is T[number] {
  return typeof value === "string" && (allowed as readonly string[]).includes(value);
}

function deepClone<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((item) => deepClone(item)) as unknown as T;
  }
  if (isRecord(value)) {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value)) {
      out[key] = deepClone(value[key]);
    }
    return out as T;
  }
  return value;
}

// ---------------------------------------------------------------------------
// Canonical serialization / hashing (Python `canonical_sha256` semantics)
// ---------------------------------------------------------------------------

/**
 * Recursive object-key-sorted compact JSON serialization matching Python's
 * `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
 * Array order is retained.  The audit schema only carries strings, integers,
 * booleans, nulls, arrays, and objects, so number formatting matches.
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
  if (isRecord(value)) {
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
    bytes instanceof Uint8Array
      ? bytes
      : new Uint8Array(bytes as ArrayBuffer);
  let hex = "";
  for (const byte of view) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}

async function defaultDigest(utf8: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new Error("Web Crypto is unavailable in this environment");
  }
  return bytesToHex(await subtle.digest("SHA-256", utf8 as unknown as BufferSource));
}

/** Canonical content SHA-256 over the whole value (excluding envelope key). */
export async function computeCanonicalSha256(
  value: unknown,
  digest: Sha256Digest = defaultDigest,
): Promise<string> {
  return digest(utf8Bytes(canonicalSerialize(value)));
}

/** Canonical content SHA-256 of a packet excluding its content_sha256 key. */
export function canonicalPacketSha256(
  packet: Record<string, unknown>,
  digest: Sha256Digest,
): Promise<string> {
  const inner: Record<string, unknown> = {};
  for (const key of Object.keys(packet)) {
    if (key !== "content_sha256") {
      inner[key] = packet[key];
    }
  }
  return computeCanonicalSha256(inner, digest);
}

// ---------------------------------------------------------------------------
// Blank template validation
// ---------------------------------------------------------------------------

function validateFailureBlock(
  value: unknown,
  label: string,
  errors: string[],
  expectReconstructionStage: boolean,
): FailureBlock | null {
  if (value === null) {
    return null;
  }
  if (!isRecord(value) || !hasExactKeys(value, FAILURE_BLOCK_KEYS)) {
    errors.push(`${label} must be null or an exact failure block`);
    return null;
  }
  if (!isNonEmptyString(value.stage)) {
    errors.push(`${label}.stage must be a non-empty string`);
  } else if (expectReconstructionStage && value.stage !== "reconstruction") {
    errors.push(`${label}.stage must be "reconstruction"`);
  }
  if (!isNonEmptyString(value.prompt_version)) {
    errors.push(`${label}.prompt_version must be a non-empty string`);
  }
  if (!isNonEmptyString(value.response_schema_version)) {
    errors.push(`${label}.response_schema_version must be a non-empty string`);
  }
  if (typeof value.error !== "string") {
    errors.push(`${label}.error must be a string`);
  }
  if (value.error_taxonomy !== null) {
    errors.push(`${label}.error_taxonomy must be null in a blank audit`);
  }
  if (errors.length === 0) {
    return value as unknown as FailureBlock;
  }
  return null;
}

function validateEvidenceSpans(value: unknown, label: string, errors: string[]): boolean {
  if (!Array.isArray(value) || value.some((span) => !isRecord(span))) {
    errors.push(`${label} must be an array of evidence span objects`);
    return false;
  }
  return true;
}

/**
 * Validate the canonical five-field Bronze span shape and, when the window
 * Bronze target is available, its consistency with that target (local bounds,
 * exact text slice, and source-absolute offsets).  Mirrors the authoritative
 * Python `_validate_bronze_span`.
 */
function validateBronzeSpan(
  value: unknown,
  label: string,
  errors: string[],
  bronzeTarget: BronzeTarget | null,
): boolean {
  if (!isRecord(value) || !hasExactKeys(value, BRONZE_SPAN_KEYS)) {
    errors.push(`${label} must be an object with exactly the five Bronze span fields`);
    return false;
  }
  const localStart = value.target_local_start;
  const localEnd = value.target_local_end;
  const sourceStart = value.source_absolute_start;
  const sourceEnd = value.source_absolute_end;
  const text = value.text;
  let valid = true;
  if (!isInt(localStart) || !isInt(localEnd)) {
    errors.push(`${label}.target_local_start/end must be integers`);
    valid = false;
  }
  if (!isInt(sourceStart) || !isInt(sourceEnd)) {
    errors.push(`${label}.source_absolute_start/end must be integers`);
    valid = false;
  }
  if (typeof text !== "string") {
    errors.push(`${label}.text must be a string`);
    valid = false;
  }
  if (
    bronzeTarget !== null &&
    isInt(localStart) &&
    isInt(localEnd) &&
    isInt(sourceStart) &&
    isInt(sourceEnd)
  ) {
    const start = localStart as number;
    const end = localEnd as number;
    if (start < 0 || end < 0 || start >= end || end > bronzeTarget.text.length) {
      errors.push(`${label} local offsets are invalid against the Bronze target`);
      valid = false;
    } else if (
      typeof text !== "string" ||
      bronzeTarget.text.slice(start, end) !== text
    ) {
      errors.push(`${label}.text is not an exact Bronze target slice`);
      valid = false;
    }
    if (sourceStart !== bronzeTarget.source_absolute_start + start) {
      errors.push(`${label}.source_absolute_start must equal the Bronze target source start plus local start`);
      valid = false;
    }
    if (sourceEnd !== bronzeTarget.source_absolute_start + end) {
      errors.push(`${label}.source_absolute_end must equal the Bronze target source start plus local end`);
      valid = false;
    }
  }
  return valid;
}

function validateOperation(
  value: unknown,
  category: AuditCategory,
  label: string,
  errors: string[],
  bronzeTarget: BronzeTarget | null,
): boolean {
  const kind = CATEGORY_KINDS[category];
  if (!isRecord(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  let keys: readonly string[];
  if (category === "mechanical_repairs" || category === "contextual_repairs") {
    keys = REPAIR_KEYS;
  } else if (category.endsWith("bindings")) {
    keys = BINDING_KEYS;
  } else {
    keys = STATEMENT_KEYS;
  }
  if (!hasExactKeys(value, keys)) {
    errors.push(`${label} key set is invalid`);
    return false;
  }
  if (!isNonEmptyString(value.operation_id)) {
    errors.push(`${label}.operation_id must be a non-empty string`);
  }
  if (value.operation_kind !== kind) {
    errors.push(`${label}.operation_kind must be ${kind}`);
  }
  if (value.decision !== null) {
    errors.push(`${label}.decision must be null in a blank audit`);
  }
  if (value.error_taxonomy !== null) {
    errors.push(`${label}.error_taxonomy must be null in a blank audit`);
  }

  if (category === "mechanical_repairs" || category === "contextual_repairs") {
    if (!isNonEmptyString(value.repair_type)) {
      errors.push(`${label}.repair_type must be a non-empty string`);
    }
    if (!isOneOf(value.confidence, CONFIDENCE_LEVELS)) {
      errors.push(`${label}.confidence must be one of ${CONFIDENCE_LEVELS.join(", ")}`);
    }
    if (typeof value.original_text !== "string") {
      errors.push(`${label}.original_text must be a string`);
    }
    if (typeof value.replacement !== "string") {
      errors.push(`${label}.replacement must be a string`);
    }
    validateEvidenceSpans(value.evidence_spans, `${label}.evidence_spans`, errors);
    if (value.corrected_replacement !== null) {
      errors.push(`${label}.corrected_replacement must be null in a blank audit`);
    }
  } else if (category.endsWith("bindings")) {
    if (!isNonEmptyString(value.binding_id)) {
      errors.push(`${label}.binding_id must be a non-empty string`);
    }
    if (!isOneOf(value.slot, BINDING_SLOTS)) {
      errors.push(`${label}.slot must be a recognized binding slot`);
    }
    validateBronzeSpan(value.mention, `${label}.mention`, errors, bronzeTarget);
    if (typeof value.resolved_candidate !== "string") {
      errors.push(`${label}.resolved_candidate must be a string`);
    }
    if (!isOneOf(value.resolved_status, BINDING_STATUSES)) {
      errors.push(`${label}.resolved_status must be a recognized status`);
    }
    validateEvidenceSpans(value.evidence_spans, `${label}.evidence_spans`, errors);
    if (!isBoolean(value.human_resolvable_required)) {
      errors.push(`${label}.human_resolvable_required must be a boolean`);
    }
  } else {
    if (!isNonEmptyString(value.statement_id)) {
      errors.push(`${label}.statement_id must be a non-empty string`);
    }
    if (!isNonEmptyString(value.text)) {
      errors.push(`${label}.text must be a non-empty string`);
    }
    validateEvidenceSpans(value.evidence_spans, `${label}.evidence_spans`, errors);
    if (!isStringArray(value.reconstruction_operation_ids)) {
      errors.push(`${label}.reconstruction_operation_ids must be an array of strings`);
    }
    if (!isOneOf(value.support_mode, POLISH_SUPPORT_MODES)) {
      errors.push(`${label}.support_mode must be a recognized support mode`);
    }
    if (
      value.unchanged_source_quote !== null &&
      (!isRecord(value.unchanged_source_quote) ||
        !hasExactKeys(value.unchanged_source_quote, UNCHANGED_SOURCE_QUOTE_KEYS) ||
        !isInt(value.unchanged_source_quote.target_local_start) ||
        !isInt(value.unchanged_source_quote.target_local_end) ||
        !isInt(value.unchanged_source_quote.source_absolute_start) ||
        !isInt(value.unchanged_source_quote.source_absolute_end) ||
        typeof value.unchanged_source_quote.text !== "string")
    ) {
      errors.push(`${label}.unchanged_source_quote must be null or an exact quote object`);
    }
    for (const field of STATEMENT_ATTESTATION_FIELDS) {
      if (value[field] !== null) {
        errors.push(`${label}.${field} must be null in a blank audit`);
      }
    }
  }
  return errors.length === 0;
}

/**
 * Strictly validate a blank transformation-audit template.  Completed,
 * malformed, or tampered files are rejected with collected errors.
 */
export function validateBlankTemplate(raw: unknown): TemplateValidationResult {
  const errors: string[] = [];
  if (!isRecord(raw)) {
    return { ok: false, errors: ["phase2k transformation audit must be an object"] };
  }
  if (!hasExactKeys(raw, TOP_LEVEL_KEYS)) {
    return {
      ok: false,
      errors: [
        `phase2k transformation audit top-level key set is invalid; missing=${TOP_LEVEL_KEYS.filter(
          (key) => !(key in raw),
        )} extra=${Object.keys(raw).filter(
          (key) => !(TOP_LEVEL_KEYS as readonly string[]).includes(key),
        )}`,
      ],
    };
  }
  if (raw.schema_version !== TRANSFORMATION_AUDIT_SCHEMA_VERSION) {
    errors.push(
      `schema_version must be ${TRANSFORMATION_AUDIT_SCHEMA_VERSION} (completed or malformed files are rejected)`,
    );
  }
  if (!isHex64(raw.content_sha256)) {
    errors.push("content_sha256 must be a 64-character hex string");
  }
  if (typeof raw.purpose !== "string" || raw.purpose.trim().length === 0) {
    errors.push("purpose must be a non-empty string");
  }
  if (raw.release_gate !== RELEASE_GATE_AWAITING_REVIEW) {
    errors.push(`release_gate must be ${RELEASE_GATE_AWAITING_REVIEW}`);
  }
  const binding = raw.binding;
  if (
    !isRecord(binding) ||
    !hasExactKeys(binding, ["records_sha256"]) ||
    !isHex64(binding.records_sha256)
  ) {
    errors.push("binding must be exactly { records_sha256 } with a 64-character hex value");
  }
  const errorTaxonomy = raw.error_taxonomy as unknown;
  if (
    !Array.isArray(errorTaxonomy) ||
    errorTaxonomy.length !== AUDIT_ERROR_TAXONOMY.length
  ) {
    errors.push("error_taxonomy must be the exact closed taxonomy list");
  } else if (
    !(AUDIT_ERROR_TAXONOMY as readonly string[]).every(
      (value, index) => errorTaxonomy[index] === value,
    )
  ) {
    errors.push("error_taxonomy values must exactly match the authoritative taxonomy");
  }
  const decisions = raw.decisions as unknown;
  if (
    !Array.isArray(decisions) ||
    decisions.length !== AUDIT_DECISIONS.length ||
    !(AUDIT_DECISIONS as readonly string[]).every(
      (value, index) => decisions[index] === value,
    )
  ) {
    errors.push("decisions must be exactly APPROVE, REJECT, AMBIGUOUS");
  }
  const operationKinds = raw.operation_kinds as unknown;
  if (
    !Array.isArray(operationKinds) ||
    operationKinds.length !== AUDIT_OPERATION_KINDS.length ||
    !(AUDIT_OPERATION_KINDS as readonly string[]).every(
      (value, index) => operationKinds[index] === value,
    )
  ) {
    errors.push("operation_kinds must be the exact ordered kind list");
  }

  const operationMap = raw.operation_map;
  const mapEntries: Record<string, OperationMapEntry> = {};
  if (!isRecord(operationMap)) {
    errors.push("operation_map must be an object");
  } else {
    for (const [operationId, entry] of Object.entries(operationMap)) {
      const label = `operation_map[${operationId}]`;
      if (!isRecord(entry) || !hasExactKeys(entry, OPERATION_MAP_KEYS)) {
        errors.push(`${label} key set is invalid`);
        continue;
      }
      if (entry.operation_id !== operationId) {
        errors.push(`${label}.operation_id must match its map key`);
      }
      if (!isNonEmptyString(entry.window_id)) {
        errors.push(`${label}.window_id must be a non-empty string`);
      }
      if (!isOneOf(entry.category, AUDIT_CATEGORIES)) {
        errors.push(`${label}.category must be a recognized category`);
      } else if (
        entry.operation_kind !== CATEGORY_KINDS[entry.category as AuditCategory]
      ) {
        errors.push(`${label}.operation_kind must match its category`);
      }
      if (!isInt(entry.ordinal) || entry.ordinal < 0) {
        errors.push(`${label}.ordinal must be a non-negative integer`);
      }
      if (errors.length === 0) {
        mapEntries[operationId] = entry as unknown as OperationMapEntry;
      }
    }
  }

  const windows: WindowAudit[] = [];
  if (!Array.isArray(raw.window_audits)) {
    errors.push("window_audits must be an array");
  } else {
    for (const [windowIndex, rawWindow] of raw.window_audits.entries()) {
      const label = `window_audits[${windowIndex}]`;
      if (!isRecord(rawWindow) || !hasExactKeys(rawWindow, WINDOW_KEYS)) {
        errors.push(`${label} key set is invalid`);
        continue;
      }
      if (!isNonEmptyString(rawWindow.window_id)) {
        errors.push(`${label}.window_id must be a non-empty string`);
      }
      const target = rawWindow.bronze_target;
      const targetValid =
        isRecord(target) &&
        hasExactKeys(target, BRONZE_TARGET_KEYS) &&
        typeof target.text === "string" &&
        isHex64(target.text_sha256) &&
        isInt(target.source_absolute_start) &&
        isInt(target.source_absolute_end);
      if (!targetValid) {
        errors.push(`${label}.bronze_target must be an exact Bronze target block`);
      }
      const bronzeTarget = targetValid
        ? (target as unknown as BronzeTarget)
        : null;
      const operations = rawWindow.operations;
      if (!isRecord(operations)) {
        errors.push(`${label}.operations must be an object`);
      } else {
        for (const category of AUDIT_CATEGORIES) {
          if (!Array.isArray(operations[category])) {
            errors.push(`${label}.operations.${category} must be an array`);
            continue;
          }
          for (const [index, operation] of (operations[category] as unknown[]).entries()) {
            validateOperation(
              operation,
              category,
              `${label}.operations.${category}[${index}]`,
              errors,
              bronzeTarget,
            );
          }
        }
      }
      const firstFailure = validateFailureBlock(
        rawWindow.first_failure,
        `${label}.first_failure`,
        errors,
        false,
      );
      const reconstructionFailure = validateFailureBlock(
        rawWindow.first_reconstruction_failure,
        `${label}.first_reconstruction_failure`,
        errors,
        true,
      );
      if (errors.length === 0) {
        windows.push({
          window_id: rawWindow.window_id as string,
          bronze_target: target as unknown as BronzeTarget,
          operations: operations as unknown as Record<AuditCategory, AuditOperation[]>,
          first_failure: firstFailure,
          first_reconstruction_failure: reconstructionFailure,
        });
      }
    }
  }

  // Cross-window operation identity, map consistency, and ordinal integrity.
  const seenOperationIds = new Set<string>();
  let traversalOrdinal = 0;
  for (const window of windows) {
    for (const category of AUDIT_CATEGORIES) {
      for (const operation of window.operations[category]) {
        const operationId = operation.operation_id;
        if (seenOperationIds.has(operationId)) {
          errors.push(`operation IDs must be unique; duplicate ${operationId}`);
        }
        seenOperationIds.add(operationId);
        const mapped = mapEntries[operationId];
        if (mapped === undefined) {
          errors.push(`operation_map is missing ${operationId}`);
        } else {
          if (mapped.window_id !== window.window_id) {
            errors.push(`operation_map entry for ${operationId} has the wrong window`);
          }
          if (mapped.category !== category) {
            errors.push(`operation_map entry for ${operationId} has the wrong category`);
          }
          if (mapped.operation_kind !== operation.operation_kind) {
            errors.push(`operation_map entry for ${operationId} has the wrong operation kind`);
          }
          if (mapped.ordinal !== traversalOrdinal) {
            errors.push(
              `operation ordinals must be dense and ordered; ${operationId} expected ${traversalOrdinal}`,
            );
          }
        }
        traversalOrdinal += 1;
      }
    }
  }
  for (const operationId of Object.keys(mapEntries)) {
    if (!seenOperationIds.has(operationId)) {
      errors.push(`operation_map entry ${operationId} does not match any window operation`);
    }
  }

  if (errors.length > 0) {
    return { ok: false, errors };
  }

  return {
    ok: true,
    template: {
      content_sha256: raw.content_sha256 as string,
      schema_version: TRANSFORMATION_AUDIT_SCHEMA_VERSION,
      purpose: raw.purpose as string,
      release_gate: RELEASE_GATE_AWAITING_REVIEW,
      binding: { records_sha256: (raw.binding as { records_sha256: string }).records_sha256 },
      error_taxonomy: [...(raw.error_taxonomy as string[])],
      decisions: [...(raw.decisions as AuditDecision[])],
      operation_kinds: [...(raw.operation_kinds as AuditOperationKind[])],
      operation_map: mapEntries,
      window_audits: windows,
    },
  };
}

/**
 * Recompute the blank template's canonical content hash (Python semantics)
 * and verify it matches the sealed content_sha256.  The template must already
 * pass `validateBlankTemplate`; this adds cryptographic content integrity.
 */
export async function verifyTemplateContentHash(
  template: AuditTemplate,
  digest: Sha256Digest,
): Promise<boolean> {
  return (
    template.content_sha256 ===
    (await canonicalPacketSha256(
      template as unknown as Record<string, unknown>,
      digest,
    ))
  );
}

/**
 * Strict template validation that throws on the first failure; used by the
 * route/tests when a single rejection message is sufficient.
 */
export function sanitizeBlankTemplate(raw: unknown): AuditTemplate {
  const result = validateBlankTemplate(raw);
  if (!result.ok) {
    throw new Error(result.errors.slice(0, 3).join(" "));
  }
  return result.template;
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

/** All operations in canonical ordinal order (operation_map.ordinal). */
export function flattenOperations(template: AuditTemplate): FlatOperation[] {
  const flattened: FlatOperation[] = [];
  for (const window of template.window_audits) {
    for (const category of AUDIT_CATEGORIES) {
      for (const operation of window.operations[category]) {
        const entry = template.operation_map[operation.operation_id];
        flattened.push({
          ordinal: entry.ordinal,
          operation_id: operation.operation_id,
          window_id: window.window_id,
          category,
          operation_kind: operation.operation_kind,
          operation,
          window,
        });
      }
    }
  }
  flattened.sort((left, right) => left.ordinal - right.ordinal);
  return flattened;
}

export function operationCategory(
  template: AuditTemplate,
  operationId: string,
): AuditCategory | null {
  return template.operation_map[operationId]?.category ?? null;
}

export function buildSession(template: AuditTemplate): AuditSession {
  return {
    schema_version: SESSION_SCHEMA_VERSION,
    template_sha256: template.content_sha256,
    records_sha256: template.binding.records_sha256,
    operations: flattenOperations(template).map(({ operation_id }) => ({
      operation_id,
      decision: null,
      corrected_replacement: null,
      error_taxonomy: null,
      supported: null,
      uncertainty_preserved: null,
      negation_preserved: null,
      modality_preserved: null,
      causality_invented: null,
      source_detail_dropped: null,
    })),
  };
}

function validateSessionOperationFields(
  rawOperation: unknown,
  category: AuditCategory,
  label: string,
  errors: string[],
): void {
  if (!isRecord(rawOperation) || !hasExactKeys(rawOperation, SESSION_OPERATION_KEYS)) {
    errors.push(`${label} key set is invalid`);
    return;
  }
  const decision = rawOperation.decision;
  if (decision !== null && !isOneOf(decision, AUDIT_DECISIONS)) {
    errors.push(`${label}.decision must be APPROVE, REJECT, AMBIGUOUS, or null`);
  }
  const taxonomy = rawOperation.error_taxonomy;
  if (taxonomy !== null && !isOneOf(taxonomy, AUDIT_ERROR_TAXONOMY)) {
    errors.push(`${label}.error_taxonomy is not in the closed taxonomy`);
  } else if (taxonomy !== null && decision !== "REJECT") {
    errors.push(`${label}.error_taxonomy is only valid with a REJECT decision`);
  }
  const corrected = rawOperation.corrected_replacement;
  if (
    (corrected !== null && typeof corrected !== "string") ||
    (category.endsWith("bindings") && corrected !== null) ||
    (category === "polished_statements" && corrected !== null)
  ) {
    errors.push(`${label}.corrected_replacement must be null or a string for repairs only`);
  }
  for (const field of STATEMENT_ATTESTATION_FIELDS) {
    const value = rawOperation[field];
    if (value !== null && !isBoolean(value)) {
      errors.push(`${label}.${field} must be a boolean or null`);
    }
    if (category !== "polished_statements" && value !== null) {
      errors.push(`${label}.${field} is only valid for polished statements`);
    }
  }
}

/**
 * Strict session validation: immutable binding to the template content hash and
 * records hash, exact operation identity/order, and per-category field rules.
 */
export function validateSessionInput(
  raw: unknown,
  template: AuditTemplate,
): SessionValidationResult {
  const errors: string[] = [];
  if (!isRecord(raw) || !hasExactKeys(raw, SESSION_KEYS)) {
    return { ok: false, errors: ["audit session envelope is invalid"] };
  }
  if (raw.schema_version !== SESSION_SCHEMA_VERSION) {
    errors.push(`session schema_version must be ${SESSION_SCHEMA_VERSION}`);
  }
  if (raw.template_sha256 !== template.content_sha256) {
    errors.push("session template_sha256 does not match the loaded template");
  }
  if (raw.records_sha256 !== template.binding.records_sha256) {
    errors.push("session records_sha256 does not match the template binding");
  }
  const operations = raw.operations;
  const expected = flattenOperations(template);
  if (!Array.isArray(operations)) {
    errors.push("session operations must be an array");
  } else {
    if (operations.length !== expected.length) {
      errors.push(
        `session operation count ${operations.length} does not match template ${expected.length}`,
      );
    }
    const seenIds = new Set<string>();
    for (const [index, rawOperation] of operations.entries()) {
      const label = `session.operations[${index}]`;
      const expectedId = expected[index]?.operation_id;
      if (!isRecord(rawOperation) || typeof rawOperation.operation_id !== "string") {
        errors.push(`${label}.operation_id must be a string`);
        continue;
      }
      if (rawOperation.operation_id !== expectedId) {
        errors.push(`${label}.operation_id must match the template ordinal order`);
      }
      if (seenIds.has(rawOperation.operation_id)) {
        errors.push(`${label} duplicates operation ${rawOperation.operation_id}`);
      }
      seenIds.add(rawOperation.operation_id);
      const category = expected[index]?.category;
      if (category !== undefined) {
        validateSessionOperationFields(rawOperation, category, label, errors);
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
      template_sha256: raw.template_sha256 as string,
      records_sha256: raw.records_sha256 as string,
      operations: operations as SessionOperation[],
    },
  };
}

function updateSessionOperation(
  session: AuditSession,
  operationId: string,
  update: (operation: SessionOperation) => SessionOperation,
): AuditSession {
  return {
    ...session,
    operations: session.operations.map((operation) =>
      operation.operation_id === operationId ? update(operation) : operation,
    ),
  };
}

/** Set an explicit decision; switching away from REJECT clears the taxonomy. */
export function setSessionDecision(
  session: AuditSession,
  operationId: string,
  decision: AuditDecision | null,
): AuditSession {
  return updateSessionOperation(session, operationId, (operation) => ({
    ...operation,
    decision,
    error_taxonomy: decision === "REJECT" ? operation.error_taxonomy : null,
  }));
}

export function setSessionTaxonomy(
  session: AuditSession,
  operationId: string,
  taxonomy: string | null,
): AuditSession {
  return updateSessionOperation(session, operationId, (operation) => ({
    ...operation,
    error_taxonomy: taxonomy,
  }));
}

export function setSessionCorrection(
  session: AuditSession,
  operationId: string,
  correctedReplacement: string | null,
): AuditSession {
  return updateSessionOperation(session, operationId, (operation) => ({
    ...operation,
    corrected_replacement: correctedReplacement,
  }));
}

export function setSessionAttestation(
  session: AuditSession,
  operationId: string,
  field: StatementAttestationField,
  value: boolean | null,
): AuditSession {
  return updateSessionOperation(session, operationId, (operation) => ({
    ...operation,
    [field]: value,
  }));
}

/** Completion requires a decision and, for REJECT, a taxonomy value. */
export function isOperationComplete(
  template: AuditTemplate,
  session: AuditSession,
  operationId: string,
): boolean {
  const category = operationCategory(template, operationId);
  const operation = session.operations.find((item) => item.operation_id === operationId);
  if (category === null || operation === undefined) {
    return false;
  }
  if (operation.decision === null) {
    return false;
  }
  if (operation.decision === "REJECT" && operation.error_taxonomy === null) {
    return false;
  }
  if (category === "polished_statements") {
    return STATEMENT_ATTESTATION_FIELDS.every(
      (field) => operation[field] === true || operation[field] === false,
    );
  }
  return true;
}

export function summarizeProgress(
  template: AuditTemplate,
  session: AuditSession,
): AuditProgress {
  const byCategory = Object.fromEntries(
    AUDIT_CATEGORIES.map((category) => [category, { total: 0, completed: 0 }]),
  ) as Record<AuditCategory, { total: number; completed: number }>;
  let completed = 0;
  let total = 0;
  for (const item of flattenOperations(template)) {
    total += 1;
    byCategory[item.category].total += 1;
    if (isOperationComplete(template, session, item.operation_id)) {
      completed += 1;
      byCategory[item.category].completed += 1;
    }
  }
  return {
    total,
    completed,
    remaining: total - completed,
    by_category: byCategory,
  };
}

// ---------------------------------------------------------------------------
// Completed audit export
// ---------------------------------------------------------------------------

/**
 * Fill the exact completed packet from the session and recompute the canonical
 * content SHA-256 exactly like Python (`canonical_sha256` over the top-level
 * object excluding `content_sha256`).  Refuses incomplete or invalid sessions;
 * it never fabricates a decision or attestation.
 */
export async function buildCompletedAudit(
  template: AuditTemplate,
  session: AuditSession,
  digest: Sha256Digest,
): Promise<CompletedAuditResult> {
  const sessionResult = validateSessionInput(session, template);
  if (!sessionResult.ok) {
    return { ok: false, errors: sessionResult.errors };
  }
  const errors: string[] = [];
  for (const item of flattenOperations(template)) {
    if (!isOperationComplete(template, session, item.operation_id)) {
      errors.push(
        `operation ${item.operation_id} is incomplete: decision${
          item.category === "polished_statements" ? " and all six attestations" : ""
        } required${item.category === "polished_statements" ? "" : ""}`,
      );
    }
  }
  if (errors.length > 0) {
    return { ok: false, errors };
  }

  const sessionByOperation = new Map(
    session.operations.map((operation) => [operation.operation_id, operation]),
  );
  const completed = deepClone(template) as unknown as CompletedAudit;
  completed.schema_version = COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION;
  completed.release_gate = RELEASE_GATE_REVIEWED;
  for (const window of completed.window_audits) {
    for (const category of AUDIT_CATEGORIES) {
      for (const operation of window.operations[category]) {
        const human = sessionByOperation.get(operation.operation_id);
        if (!human) {
          return { ok: false, errors: [`missing session data for ${operation.operation_id}`] };
        }
        operation.decision = human.decision as CompletedOperation["decision"];
        operation.error_taxonomy = human.error_taxonomy;
        if (category === "mechanical_repairs" || category === "contextual_repairs") {
          operation.corrected_replacement = human.corrected_replacement;
        }
        if (category === "polished_statements") {
          operation.supported = human.supported as boolean;
          operation.uncertainty_preserved = human.uncertainty_preserved as boolean;
          operation.negation_preserved = human.negation_preserved as boolean;
          operation.modality_preserved = human.modality_preserved as boolean;
          operation.causality_invented = human.causality_invented as boolean;
          operation.source_detail_dropped = human.source_detail_dropped as boolean;
        }
      }
    }
  }
  completed.content_sha256 = await canonicalPacketSha256(
    completed as unknown as Record<string, unknown>,
    digest,
  );
  return { ok: true, completed };
}
