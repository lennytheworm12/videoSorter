import { createHash } from "node:crypto";
import {
  AUDIT_CATEGORIES,
  AUDIT_DECISIONS,
  AUDIT_ERROR_TAXONOMY,
  AUDIT_OPERATION_KINDS,
  buildCompletedAudit,
  buildSession,
  canonicalSerialize,
  computeCanonicalSha256,
  COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION,
  flattenOperations,
  isOperationComplete,
  RELEASE_GATE_AWAITING_REVIEW,
  RELEASE_GATE_REVIEWED,
  sanitizeBlankTemplate,
  setSessionAttestation,
  setSessionCorrection,
  setSessionDecision,
  setSessionTaxonomy,
  STATEMENT_ATTESTATION_FIELDS,
  summarizeProgress,
  TRANSFORMATION_AUDIT_SCHEMA_VERSION,
  validateBlankTemplate,
  validateSessionInput,
  verifyTemplateContentHash,
  type AuditCategory,
  type AuditSession,
  type AuditTemplate,
  type BindingOperation,
  type CompletedOperation,
  type Sha256Digest,
} from "./phase2k-audit";

function sha256Hex(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

const digest: Sha256Digest = (bytes) =>
  createHash("sha256").update(bytes).digest("hex");

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function canonicalSha256(value: unknown): string {
  return sha256Hex(stableStringify(value));
}

function repairOperation(
  operationId: string,
  overrides: Record<string, unknown> = {},
) {
  return {
    operation_id: operationId,
    operation_kind: "MECHANICAL_REPAIR",
    repair_type: "ASR_HOMOPHONE",
    confidence: "HIGH",
    original_text: "hit R yeah",
    replacement: "hit R, yeah",
    evidence_spans: [
      {
        segment_id: "seg-1",
        source_absolute_start: 10,
        source_absolute_end: 21,
        text: "hit R yeah",
      },
    ],
    decision: null,
    corrected_replacement: null,
    error_taxonomy: null,
    ...overrides,
  };
}

function contextualRepairOperation(
  operationId: string,
  overrides: Record<string, unknown> = {},
) {
  return repairOperation(operationId, {
    operation_kind: "CONTEXTUAL_REPAIR",
    repair_type: "CONTEXTUAL_ASR",
    ...overrides,
  });
}

function bindingOperation(
  operationId: string,
  overrides: Record<string, unknown> = {},
) {
  return {
    operation_id: operationId,
    operation_kind: "ENTITY_BINDING",
    binding_id: "bind-1",
    slot: "principal_actors",
    mention: {
      target_local_start: 0,
      target_local_end: 9,
      source_absolute_start: 0,
      source_absolute_end: 9,
      text: "the enemy",
    },
    resolved_candidate: "opposing champion",
    resolved_status: "RESOLVED",
    evidence_spans: [
      {
        segment_id: "seg-1",
        source_absolute_start: 0,
        source_absolute_end: 9,
        text: "the enemy",
      },
    ],
    human_resolvable_required: true,
    decision: null,
    error_taxonomy: null,
    ...overrides,
  };
}

function pronounBindingOperation(
  operationId: string,
  overrides: Record<string, unknown> = {},
) {
  return bindingOperation(operationId, {
    operation_kind: "PRONOUN_BINDING",
    binding_id: "bind-pronoun",
    slot: "pronouns",
    ...overrides,
  });
}

function statementOperation(
  operationId: string,
  overrides: Record<string, unknown> = {},
) {
  return {
    operation_id: operationId,
    operation_kind: "POLISHED_STATEMENT",
    statement_id: "stmt-1",
    text: "The enemy backed off after the wave crashed.",
    evidence_spans: [
      {
        segment_id: "seg-2",
        source_absolute_start: 21,
        source_absolute_end: 40,
        text: "after the wave crashed",
      },
    ],
    reconstruction_operation_ids: ["op-bind-1"],
    support_mode: "EVIDENCE_PARAPHRASE",
    unchanged_source_quote: {
      target_local_start: 0,
      target_local_end: 9,
      source_absolute_start: 21,
      source_absolute_end: 30,
      text: "wave crashed",
    },
    decision: null,
    supported: null,
    uncertainty_preserved: null,
    negation_preserved: null,
    modality_preserved: null,
    causality_invented: null,
    source_detail_dropped: null,
    error_taxonomy: null,
    ...overrides,
  };
}

type WindowOperations = Partial<Record<AuditCategory, unknown[]>>;

function makeWindow(
  windowId: string,
  operations: WindowOperations,
  overrides: Record<string, unknown> = {},
) {
  const text = "the enemy backs off";
  return {
    window_id: windowId,
    bronze_target: {
      text,
      text_sha256: sha256Hex(text),
      source_absolute_start: 0,
      source_absolute_end: text.length,
    },
    operations: {
      mechanical_repairs: [],
      contextual_repairs: [],
      entity_bindings: [],
      pronoun_bindings: [],
      reference_bindings: [],
      ability_bindings: [],
      polished_statements: [],
      ...operations,
    },
    first_failure: null,
    first_reconstruction_failure: null,
    ...overrides,
  };
}

function makeTemplate(overrides: Record<string, unknown> = {}): AuditTemplate {
  const windowAudits = [
    makeWindow("w1", {
      mechanical_repairs: [repairOperation("w1::mech::r1")],
      entity_bindings: [bindingOperation("w1::bind::b1")],
      polished_statements: [statementOperation("w1::stmt::s1")],
    }),
    makeWindow("w2", {
      contextual_repairs: [contextualRepairOperation("w2::ctx::r1")],
      pronoun_bindings: [pronounBindingOperation("w2::bind::p1")],
      polished_statements: [statementOperation("w2::stmt::s1")],
    }),
  ];
  const operationMap: Record<string, unknown> = {};
  let ordinal = 0;
  for (const window of windowAudits) {
    for (const category of AUDIT_CATEGORIES) {
      const items = (window.operations[category] ?? []) as unknown[];
      for (const item of items) {
        const operation = item as {
          operation_id: string;
          operation_kind: string;
        };
        operationMap[operation.operation_id] = {
          operation_id: operation.operation_id,
          window_id: window.window_id,
          category,
          operation_kind: operation.operation_kind,
          ordinal,
        };
        ordinal += 1;
      }
    }
  }
  const inner = {
    schema_version: TRANSFORMATION_AUDIT_SCHEMA_VERSION,
    purpose: "Downstream-result-blind Phase 2K transformation audit fixture.",
    release_gate: RELEASE_GATE_AWAITING_REVIEW,
    binding: { records_sha256: "a".repeat(64) },
    error_taxonomy: [...AUDIT_ERROR_TAXONOMY],
    decisions: [...AUDIT_DECISIONS],
    operation_kinds: [...AUDIT_OPERATION_KINDS],
    operation_map: operationMap,
    window_audits: windowAudits,
  };
  return {
    content_sha256: canonicalSha256(inner),
    ...inner,
    ...overrides,
  } as AuditTemplate;
}

function rehashTemplate(
  template: AuditTemplate,
  patch: Record<string, unknown>,
): AuditTemplate {
  const inner: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(template)) {
    if (key !== "content_sha256") {
      inner[key] = value;
    }
  }
  const merged = { ...inner, ...patch };
  return { content_sha256: canonicalSha256(merged), ...merged } as AuditTemplate;
}

function completeSession(template: AuditTemplate, session: AuditSession): AuditSession {
  let next = session;
  for (const item of flattenOperations(template)) {
    next = setSessionDecision(next, item.operation_id, "APPROVE");
    if (item.category === "polished_statements") {
      for (const field of STATEMENT_ATTESTATION_FIELDS) {
        next = setSessionAttestation(
          next,
          item.operation_id,
          field,
          field === "causality_invented" ? false : true,
        );
      }
    }
  }
  return next;
}

describe("blank template sanitization", () => {
  it("accepts a valid blank template and verifies its sealed content hash", async () => {
    const template = makeTemplate();
    const sanitized = sanitizeBlankTemplate(template);
    expect(sanitized.window_audits).toHaveLength(2);
    expect(flattenOperations(sanitized)).toHaveLength(6);
    expect(await verifyTemplateContentHash(sanitized, digest)).toBe(true);
  });

  it("rejects a completed packet (wrong schema) as a blank template", () => {
    const raw = makeTemplate({
      schema_version: COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION,
    });
    const result = validateBlankTemplate(raw);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/schema_version/);
    }
  });

  it("rejects a wrong release gate", () => {
    const raw = makeTemplate({ release_gate: RELEASE_GATE_REVIEWED });
    const result = validateBlankTemplate(raw);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/release_gate/);
    }
  });

  it("rejects malformed content hashes and records binding", () => {
    const badHash = validateBlankTemplate(makeTemplate({ content_sha256: "nope" }));
    expect(badHash.ok).toBe(false);
    if (!badHash.ok) {
      expect(badHash.errors.join(" ")).toMatch(/content_sha256/);
    }
    const badBinding = validateBlankTemplate(
      makeTemplate({ binding: { records_sha256: "short" } }),
    );
    expect(badBinding.ok).toBe(false);
    if (!badBinding.ok) {
      expect(badBinding.errors.join(" ")).toMatch(/binding/);
    }
  });

  it("rejects duplicate operation IDs", () => {
    const window = makeWindow("w1", {
      mechanical_repairs: [
        repairOperation("w1::mech::r1"),
        repairOperation("w1::mech::r1", { repair_type: "SPELLING" }),
      ],
    });
    const raw = makeTemplate({ window_audits: [window] });
    const result = validateBlankTemplate(raw);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/unique/);
    }
  });

  it("rejects broken operation maps and non-dense ordinals", () => {
    const template = makeTemplate();
    const map = {
      ...template.operation_map,
      "w1::mech::r1": {
        ...template.operation_map["w1::mech::r1"],
        ordinal: 99,
      },
    };
    const result = validateBlankTemplate({ ...template, operation_map: map });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/ordinals must be dense and ordered/);
    }
  });

  it("rejects operation_map entries whose category or kind does not match", () => {
    const template = makeTemplate();
    const map = {
      ...template.operation_map,
      "w1::mech::r1": {
        ...template.operation_map["w1::mech::r1"],
        category: "polished_statements",
        operation_kind: "POLISHED_STATEMENT",
      },
    };
    const result = validateBlankTemplate({ ...template, operation_map: map });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/wrong category|operation_kind must match/);
    }
  });

  it("rejects nonblank human fields in a blank template", () => {
    const template = makeTemplate();
    const windows = template.window_audits.map((window) => ({
      ...window,
      operations: {
        ...window.operations,
        mechanical_repairs:
          window.window_id === "w1"
            ? [
                {
                  ...window.operations.mechanical_repairs[0],
                  decision: "APPROVE",
                },
              ]
            : window.operations.mechanical_repairs,
      },
    }));
    const result = validateBlankTemplate({ ...template, window_audits: windows });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/decision must be null/);
    }
  });

  it("rejects extra or missing top-level keys", () => {
    const withExtra = { ...makeTemplate(), extra: true };
    expect(validateBlankTemplate(withExtra).ok).toBe(false);
    const { extra, ...withoutPurpose } = makeTemplate() as Record<string, unknown>;
    delete withoutPurpose.purpose;
    expect(validateBlankTemplate(withoutPurpose).ok).toBe(false);
  });
});

describe("binding mention contract", () => {
  function bindingMention(): Record<string, unknown> {
    return {
      target_local_start: 0,
      target_local_end: 9,
      source_absolute_start: 0,
      source_absolute_end: 9,
      text: "the enemy",
    };
  }

  function templateWithMention(mention: unknown) {
    const template = makeTemplate();
    const windows = template.window_audits.map((window, index) =>
      index === 0
        ? {
            ...window,
            operations: {
              ...window.operations,
              entity_bindings: [
                {
                  ...window.operations.entity_bindings[0],
                  mention,
                },
              ],
            },
          }
        : window,
    );
    return { ...template, window_audits: windows };
  }

  it("accepts a full five-field Bronze span mention", () => {
    const result = validateBlankTemplate(makeTemplate());
    expect(result.ok).toBe(true);
    if (result.ok) {
      const mention = (
        result.template.window_audits[0].operations.entity_bindings[0] as BindingOperation
      ).mention;
      expect(mention).toEqual(bindingMention());
    }
  });

  it("rejects a string mention", () => {
    const result = validateBlankTemplate(templateWithMention("the enemy"));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/mention/);
    }
  });

  it("rejects missing or extra mention fields", () => {
    const missing = templateWithMention({
      target_local_start: 0,
      target_local_end: 9,
      source_absolute_start: 0,
      source_absolute_end: 9,
    });
    expect(validateBlankTemplate(missing).ok).toBe(false);
    const extra = templateWithMention({ ...bindingMention(), extra: true });
    expect(validateBlankTemplate(extra).ok).toBe(false);
  });

  it("rejects boolean and non-integer offsets", () => {
    const booleanOffset = templateWithMention({
      ...bindingMention(),
      target_local_start: true,
    });
    expect(validateBlankTemplate(booleanOffset).ok).toBe(false);
    const floatOffset = templateWithMention({
      ...bindingMention(),
      target_local_end: 9.5,
    });
    expect(validateBlankTemplate(floatOffset).ok).toBe(false);
  });

  it("rejects negative, reversed, and out-of-bounds local offsets", () => {
    const negative = templateWithMention({
      ...bindingMention(),
      target_local_start: -1,
      source_absolute_start: -1,
    });
    expect(validateBlankTemplate(negative).ok).toBe(false);
    const reversed = templateWithMention({
      ...bindingMention(),
      target_local_start: 9,
      target_local_end: 0,
      source_absolute_start: 9,
      source_absolute_end: 0,
    });
    expect(validateBlankTemplate(reversed).ok).toBe(false);
    const outOfBounds = templateWithMention({
      ...bindingMention(),
      target_local_end: 20,
      source_absolute_end: 20,
    });
    expect(validateBlankTemplate(outOfBounds).ok).toBe(false);
  });

  it("rejects local/absolute length disagreement", () => {
    const startMismatch = templateWithMention({
      ...bindingMention(),
      source_absolute_start: 1,
    });
    expect(validateBlankTemplate(startMismatch).ok).toBe(false);
    const endMismatch = templateWithMention({
      ...bindingMention(),
      source_absolute_end: 8,
    });
    expect(validateBlankTemplate(endMismatch).ok).toBe(false);
  });

  it("rejects a text not equal to the exact Bronze target slice", () => {
    const wrongLength = templateWithMention({
      ...bindingMention(),
      text: "the enem",
    });
    expect(validateBlankTemplate(wrongLength).ok).toBe(false);
    const wrongSlice = templateWithMention({
      ...bindingMention(),
      text: "enemy backs",
    });
    expect(validateBlankTemplate(wrongSlice).ok).toBe(false);
  });
});

describe("decision transitions and REJECT taxonomy", () => {
  it("requires a taxonomy value for REJECT", () => {
    const template = makeTemplate();
    const session = buildSession(template);
    const operationId = "w1::mech::r1";
    expect(isOperationComplete(template, session, operationId)).toBe(false);
    const rejected = setSessionDecision(session, operationId, "REJECT");
    expect(isOperationComplete(template, rejected, operationId)).toBe(false);
    const withTaxonomy = setSessionTaxonomy(rejected, operationId, "ASR_REPAIR_WRONG");
    expect(isOperationComplete(template, withTaxonomy, operationId)).toBe(true);
  });

  it("clears the taxonomy when switching away from REJECT", () => {
    const template = makeTemplate();
    const session = buildSession(template);
    const operationId = "w1::mech::r1";
    let next = setSessionDecision(session, operationId, "REJECT");
    next = setSessionTaxonomy(next, operationId, "ASR_REPAIR_WRONG");
    next = setSessionDecision(next, operationId, "APPROVE");
    expect(next.operations[0].error_taxonomy).toBeNull();
  });

  it("rejects a session where taxonomy is set without REJECT", () => {
    const template = makeTemplate();
    const session = buildSession(template);
    const operationId = "w1::mech::r1";
    const tampered = setSessionTaxonomy(
      setSessionDecision(session, operationId, "APPROVE"),
      operationId,
      "OTHER",
    );
    const result = validateSessionInput(tampered, template);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/only valid with a REJECT decision/);
    }
  });
});

describe("statement attestations", () => {
  it("requires all six explicit booleans for completion", () => {
    const template = makeTemplate();
    const session = buildSession(template);
    const operationId = "w1::stmt::s1";
    let next = setSessionDecision(session, operationId, "APPROVE");
    for (const field of STATEMENT_ATTESTATION_FIELDS.slice(0, 5)) {
      expect(isOperationComplete(template, next, operationId)).toBe(false);
      next = setSessionAttestation(next, operationId, field, true);
    }
    expect(isOperationComplete(template, next, operationId)).toBe(false);
    next = setSessionAttestation(next, operationId, "source_detail_dropped", false);
    expect(isOperationComplete(template, next, operationId)).toBe(true);
  });

  it("never defaults or fabricates decisions or attestations", () => {
    const template = makeTemplate();
    const session = buildSession(template);
    expect(session.operations.every((operation) => operation.decision === null)).toBe(
      true,
    );
    expect(
      session.operations.every((operation) =>
        STATEMENT_ATTESTATION_FIELDS.every((field) => operation[field] === null),
      ),
    ).toBe(true);
  });
});

describe("session binding and import validation", () => {
  it("round-trips a session through JSON and validates against the template", () => {
    const template = makeTemplate();
    const session = completeSession(template, buildSession(template));
    const roundTripped = JSON.parse(JSON.stringify(session)) as unknown;
    const result = validateSessionInput(roundTripped, template);
    expect(result.ok).toBe(true);
  });

  it("rejects a session bound to a different template", () => {
    const templateA = makeTemplate();
    const templateB = rehashTemplate(templateA, {
      purpose: "different fixture purpose",
    });
    const session = buildSession(templateA);
    const result = validateSessionInput(session, templateB);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/template_sha256/);
    }
  });

  it("rejects a session with a wrong records hash", () => {
    const template = makeTemplate();
    const session = {
      ...buildSession(template),
      records_sha256: "f".repeat(64),
    };
    const result = validateSessionInput(session, template);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/records_sha256/);
    }
  });

  it("rejects sessions with missing, reordered, or duplicate operations", () => {
    const template = makeTemplate();
    const base = buildSession(template);
    const missing = {
      ...base,
      operations: base.operations.slice(1),
    };
    expect(validateSessionInput(missing, template).ok).toBe(false);
    const reordered = {
      ...base,
      operations: [...base.operations].reverse(),
    };
    expect(validateSessionInput(reordered, template).ok).toBe(false);
    const duplicated = {
      ...base,
      operations: [base.operations[0], ...base.operations],
    };
    expect(validateSessionInput(duplicated, template).ok).toBe(false);
  });
});

describe("progress counts", () => {
  it("counts total, completed, and remaining operations", () => {
    const template = makeTemplate();
    const fresh = buildSession(template);
    expect(summarizeProgress(template, fresh)).toEqual({
      total: 6,
      completed: 0,
      remaining: 6,
      by_category: {
        mechanical_repairs: { total: 1, completed: 0 },
        contextual_repairs: { total: 1, completed: 0 },
        entity_bindings: { total: 1, completed: 0 },
        pronoun_bindings: { total: 1, completed: 0 },
        reference_bindings: { total: 0, completed: 0 },
        ability_bindings: { total: 0, completed: 0 },
        polished_statements: { total: 2, completed: 0 },
      },
    });
    const oneDone = setSessionDecision(fresh, "w1::mech::r1", "APPROVE");
    expect(summarizeProgress(template, oneDone).completed).toBe(1);
    const allDone = completeSession(template, fresh);
    expect(summarizeProgress(template, allDone)).toMatchObject({
      total: 6,
      completed: 6,
      remaining: 0,
    });
  });
});

describe("completed audit export", () => {
  it("refuses export while any operation is incomplete", async () => {
    const template = makeTemplate();
    const session = buildSession(template);
    const result = await buildCompletedAudit(template, session, digest);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/incomplete/);
    }
  });

  it("preserves immutable fields and emits the exact completed schema", async () => {
    const template = makeTemplate();
    let session = buildSession(template);
    session = completeSession(template, session);
    session = setSessionDecision(session, "w1::mech::r1", "REJECT");
    session = setSessionTaxonomy(session, "w1::mech::r1", "ASR_REPAIR_WRONG");
    session = setSessionCorrection(session, "w1::mech::r1", "hit R, yeah.");
    session = setSessionDecision(session, "w1::bind::b1", "REJECT");
    session = setSessionTaxonomy(session, "w1::bind::b1", "ENTITY_BIND_WRONG");

    const result = await buildCompletedAudit(template, session, digest);
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const completed = result.completed;
    expect(completed.schema_version).toBe(COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION);
    expect(completed.release_gate).toBe(RELEASE_GATE_REVIEWED);
    expect(completed.purpose).toBe(template.purpose);
    expect(completed.binding).toEqual(template.binding);
    expect(completed.error_taxonomy).toEqual(template.error_taxonomy);
    expect(completed.decisions).toEqual(template.decisions);
    expect(completed.operation_kinds).toEqual(template.operation_kinds);
    expect(completed.operation_map).toEqual(template.operation_map);
    for (let index = 0; index < template.window_audits.length; index += 1) {
      expect(completed.window_audits[index].window_id).toBe(
        template.window_audits[index].window_id,
      );
      expect(completed.window_audits[index].bronze_target).toEqual(
        template.window_audits[index].bronze_target,
      );
      expect(completed.window_audits[index].first_failure).toEqual(
        template.window_audits[index].first_failure,
      );
      expect(completed.window_audits[index].first_reconstruction_failure).toEqual(
        template.window_audits[index].first_reconstruction_failure,
      );
    }

    const mechanical = completed.window_audits[0].operations
      .mechanical_repairs[0] as CompletedOperation & {
      repair_type: string;
      confidence: string;
      original_text: string;
      replacement: string;
      evidence_spans: unknown[];
    };
    expect(mechanical.decision).toBe("REJECT");
    expect(mechanical.error_taxonomy).toBe("ASR_REPAIR_WRONG");
    expect(mechanical.corrected_replacement).toBe("hit R, yeah.");
    expect(mechanical.original_text).toBe("hit R yeah");
    expect(mechanical.replacement).toBe("hit R, yeah");
    const binding = completed.window_audits[0].operations.entity_bindings[0];
    expect(binding.decision).toBe("REJECT");
    expect(binding.error_taxonomy).toBe("ENTITY_BIND_WRONG");
    const statement = completed.window_audits[0].operations.polished_statements[0];
    expect(statement.supported).toBe(true);
    expect(statement.uncertainty_preserved).toBe(true);
    expect(statement.negation_preserved).toBe(true);
    expect(statement.modality_preserved).toBe(true);
    expect(statement.causality_invented).toBe(false);
    expect(statement.source_detail_dropped).toBe(true);
  });

  it("recomputes content_sha256 exactly like Python canonical hashing", async () => {
    const template = makeTemplate();
    const session = completeSession(template, buildSession(template));
    const result = await buildCompletedAudit(template, session, digest);
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const completed = result.completed;
    const inner: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(completed)) {
      if (key !== "content_sha256") {
        inner[key] = value;
      }
    }
    expect(completed.content_sha256).toBe(canonicalSha256(inner));
    expect(await computeCanonicalSha256(inner, digest)).toBe(canonicalSha256(inner));
  });
});

describe("canonical serialization", () => {
  it("matches Python sort_keys compact separators semantics", () => {
    const value = {
      z: [1, 2, { b: true, a: null }],
      a: { c: "text", b: 0 },
    };
    expect(canonicalSerialize(value)).toBe(
      '{"a":{"b":0,"c":"text"},"z":[1,2,{"a":null,"b":true}]}',
    );
  });

  it("retains array order while sorting object keys", () => {
    expect(canonicalSerialize({ b: [3, 1, 2], a: [] })).toBe('{"a":[],"b":[3,1,2]}');
  });

  it("produces a stable digest across key insertion order", async () => {
    const left = { x: 1, y: { deep: [true, "s"] } };
    const right = { y: { deep: [true, "s"] }, x: 1 };
    expect(canonicalSerialize(left)).toBe(canonicalSerialize(right));
    expect(await computeCanonicalSha256(left, digest)).toBe(
      await computeCanonicalSha256(right, digest),
    );
  });
});

describe("template content integrity", () => {
  it("rejects a template whose sealed hash no longer matches its content", async () => {
    const template = makeTemplate();
    const tampered = {
      ...template,
      content_sha256: "f".repeat(64),
    };
    expect(await verifyTemplateContentHash(tampered, digest)).toBe(false);
  });
});
