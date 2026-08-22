import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import {
  AUDIT_CHECKS,
  ADJUDICATION_EXPORT_FORBIDDEN_KEYS,
  ADJUDICATION_EXPORT_SCHEMA_VERSION,
  ADJUDICATION_PACKET_SCHEMA_VERSION,
  ADJUDICATION_STATE_SCHEMA_VERSION,
  ADJUDICATION_VERSION,
  emptyAuditChecks,
  parseAuditChecks,
  buildAdjudicationExport,
  buildAdjudicationState,
  componentDecisionAllowed,
  deriveResolvedEndpoints,
  findAdjudicationForbiddenFields,
  isComponentResolved,
  isWindowResolved,
  keepPassAChoices,
  sanitizeAdjudicationPacket,
  summarizeAdjudicationProgress,
  unresolvedComponents,
  validateAdjudicationExport,
  validateAdjudicationState,
  type AdjudicationComponent,
  type AdjudicationHumanEndpoint,
  type AdjudicationPayload,
  type AdjudicationRecord,
  type AdjudicationSolEndpoint,
  type AdjudicationState,
  type AdjudicationTotals,
  type AuditChecks,
  type ComponentDecision,
} from "./phase2j-adjudication";
import { ENDPOINT_TYPES } from "./phase2j-review";

function sha256Hex(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function allTrueAuditChecks(): AuditChecks {
  return {
    boundaries: true,
    omissions: true,
    roles: true,
    duplicates: true,
    ambiguity: true,
  };
}

function tokenTable(text: string) {
  const tokens = [];
  const pattern = /\S+/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    tokens.push({
      token_index: tokens.length,
      text: match[0],
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return tokens;
}

const TEXT_A = "one two three four five six seven eight nine ten";
const TEXT_B = "alpha beta gamma delta epsilon zeta eta theta iota kappa";

type ComponentSeed = {
  classification: AdjudicationComponent["classification"];
  human: Array<[number, number, (typeof ENDPOINT_TYPES)[number]]>;
  sol: Array<[number, number, (typeof ENDPOINT_TYPES)[number] | null]>;
};

function makeAdjudicationRecord(
  windowId: string,
  text: string,
  seeds: ComponentSeed[],
): AdjudicationRecord {
  const tokens = tokenTable(text);
  const humanEndpoints: AdjudicationHumanEndpoint[] = [];
  const solEndpoints: AdjudicationSolEndpoint[] = [];
  const components: AdjudicationComponent[] = [];
  let humanSequence = 1;
  let solSequence = 1;
  seeds.forEach((seed, seedIndex) => {
    const humanIds = seed.human.map(([tokenStart, tokenEnd, nodeType]) => {
      const charStart = tokens[tokenStart].start;
      const charEnd = tokens[tokenEnd].end;
      const endpointId = `p2j:review:${windowId}:ep:${String(humanSequence).padStart(4, "0")}`;
      humanSequence += 1;
      humanEndpoints.push({
        endpoint_id: endpointId,
        exact_bronze_text: text.slice(charStart, charEnd),
        char_start: charStart,
        char_end: charEnd,
        token_start: tokenStart,
        token_end: tokenEnd,
        node_type: nodeType,
      });
      return endpointId;
    });
    const solIds = seed.sol.map(([tokenStart, tokenEnd, nodeType]) => {
      const charStart = tokens[tokenStart].start;
      const charEnd = tokens[tokenEnd].end;
      const endpointId = `p2j:sol:${windowId}:ep:${String(solSequence).padStart(4, "0")}`;
      solSequence += 1;
      solEndpoints.push({
        endpoint_id: endpointId,
        exact_bronze_text: text.slice(charStart, charEnd),
        char_start: charStart,
        char_end: charEnd,
        token_start: tokenStart,
        token_end: tokenEnd,
        node_type: nodeType,
        sol_ambiguity_state: "NONE",
        sol_rationale: "synthetic second opinion",
      });
      return endpointId;
    });
    components.push({
      component_id: `p2j:adjudicate:${windowId}:c:${String(seedIndex + 1).padStart(4, "0")}`,
      classification: seed.classification,
      human_endpoint_ids: humanIds,
      sol_endpoint_ids: solIds,
    });
  });
  return {
    record_index: 1,
    window_id: windowId,
    source_group_id: `video:${windowId}`,
    bronze_text: text,
    bronze_text_sha256: sha256Hex(text),
    bronze_char_length: text.length,
    tokens,
    human_outcome: "CLEAN",
    human_endpoints: humanEndpoints,
    sol_endpoints: solEndpoints,
    components,
  };
}

function makePayload(records: AdjudicationRecord[] = []): AdjudicationPayload {
  const totals: AdjudicationTotals = {
    windows: records.length,
    components: records.reduce((sum, record) => sum + record.components.length, 0),
    exact_agreements: records.reduce(
      (sum, record) =>
        sum + record.components.filter((component) => component.classification === "EXACT_AGREEMENT").length,
      0,
    ),
    type_disagreements: records.reduce(
      (sum, record) =>
        sum + record.components.filter((component) => component.classification === "TYPE_DISAGREEMENT").length,
      0,
    ),
    boundary_disagreements: records.reduce(
      (sum, record) =>
        sum + record.components.filter((component) => component.classification === "BOUNDARY_DISAGREEMENT").length,
      0,
    ),
    sol_only: records.reduce(
      (sum, record) =>
        sum + record.components.filter((component) => component.classification === "SOL_ONLY").length,
      0,
    ),
    human_only: records.reduce(
      (sum, record) =>
        sum + record.components.filter((component) => component.classification === "HUMAN_ONLY").length,
      0,
    ),
    human_endpoints: records.reduce((sum, record) => sum + record.human_endpoints.length, 0),
    sol_endpoints: records.reduce((sum, record) => sum + record.sol_endpoints.length, 0),
  };
  const normalizedRecords = records.map((record, index) => ({
    ...record,
    record_index: index + 1,
  }));
  return {
    schema_version: ADJUDICATION_PACKET_SCHEMA_VERSION,
    adjudication_version: ADJUDICATION_VERSION,
    annotation_version: "phase2j-endpoint-annotation-v1",
    packet_schema_version: "phase2j-endpoint-annotation-packet-v1",
    packet_sha256: "a".repeat(64),
    adjudication_packet_sha256: "b".repeat(64),
    human_session_sha256: "c".repeat(64),
    sol_review_sha256: "d".repeat(64),
    totals,
    records: normalizedRecords,
  };
}

function standardSeeds(): ComponentSeed[] {
  return [
    {
      classification: "TYPE_DISAGREEMENT",
      human: [[0, 0, "ENTITY"]],
      sol: [[0, 0, "ACTION"]],
    },
    {
      classification: "BOUNDARY_DISAGREEMENT",
      human: [[2, 3, "ACTION"]],
      sol: [[2, 4, "TIME"]],
    },
    { classification: "SOL_ONLY", human: [], sol: [[6, 6, "STATE"]] },
    { classification: "SOL_ONLY", human: [], sol: [[8, 8, "EVENT"]] },
    {
      classification: "EXACT_AGREEMENT",
      human: [[9, 9, "OUTCOME"]],
      sol: [[9, 9, "OUTCOME"]],
    },
  ];
}

function standardPayload(): AdjudicationPayload {
  return makePayload([
    makeAdjudicationRecord("pool:w1-aaa", TEXT_A, standardSeeds()),
    makeAdjudicationRecord("pool:w2-bbb", TEXT_B, standardSeeds()),
  ]);
}

describe("component decision rules", () => {
  const payload = standardPayload();
  const record = payload.records[0];
  const state = buildAdjudicationState(payload);
  const decisions = state.records[0].decisions;

  it("pre-resolves exact agreements only", () => {
    expect(state.records[0].decisions["p2j:adjudicate:pool:w1-aaa:c:0005"]).toEqual({
      kind: "KEEP_HUMAN_SET",
    });
    expect(state.records[0].decisions["p2j:adjudicate:pool:w1-aaa:c:0001"]).toBeUndefined();
    expect(isComponentResolved(record.components[0], undefined)).toBe(false);
    expect(isComponentResolved(record.components[4], decisions["p2j:adjudicate:pool:w1-aaa:c:0005"])).toBe(true);
  });

  it("enforces the allowed decision matrix", () => {
    const typeComponent = record.components[0];
    const boundaryComponent = record.components[1];
    const solOnly = record.components[2];
    const exact = record.components[4];
    expect(componentDecisionAllowed(typeComponent, { kind: "KEEP_HUMAN_SET" })).toBe(true);
    expect(componentDecisionAllowed(typeComponent, { kind: "KEEP_SOL_SET" })).toBe(true);
    expect(componentDecisionAllowed(typeComponent, { kind: "DROP" })).toBe(true);
    expect(componentDecisionAllowed(typeComponent, { kind: "CUSTOM", token_start: 0, token_end: 0, node_type: "EVENT" })).toBe(true);
    expect(componentDecisionAllowed(boundaryComponent, { kind: "KEEP_HUMAN_SET" })).toBe(true);
    expect(componentDecisionAllowed(solOnly, { kind: "KEEP_HUMAN_SET" })).toBe(false);
    expect(componentDecisionAllowed(solOnly, { kind: "KEEP_SOL_SET" })).toBe(true);
    expect(componentDecisionAllowed(solOnly, { kind: "DROP" })).toBe(true);
    expect(componentDecisionAllowed(exact, { kind: "KEEP_SOL_SET" })).toBe(false);
    expect(componentDecisionAllowed(exact, { kind: "DROP" })).toBe(true);
    expect(componentDecisionAllowed(exact, {
      kind: "CUSTOM",
      token_start: 9,
      token_end: 9,
      node_type: "ACTION",
    })).toBe(true);
    expect(componentDecisionAllowed(exact, { kind: "KEEP_HUMAN_SET" })).toBe(true);
  });

  it("initializes outcomes from the sanitized human outcome", () => {
    const records = [
      makeAdjudicationRecord("pool:w3-ccc", TEXT_A, standardSeeds()),
      makeAdjudicationRecord("pool:w4-ddd", TEXT_B, standardSeeds()),
    ];
    records[0].human_outcome = "AMBIGUOUS";
    records[1].human_outcome = "EXCLUDED";
    const payload = makePayload(records);
    const state = buildAdjudicationState(payload);
    expect(state.records[0].outcome).toBe("AMBIGUOUS");
    expect(state.records[0].note).toBe("");
    expect(state.records[1].outcome).toBe("EXCLUDED");
    // AMBIGUOUS/EXCLUDED remain incomplete until the reviewer supplies a note.
    const blocked = buildAdjudicationExport(
      payload, { ...state, reviewer_name: "Ada" }, "Ada", "2026-08-18T00:00:00Z",
      allTrueAuditChecks(),
    );
    expect(blocked.ok).toBe(false);
    if (!blocked.ok) {
      expect(blocked.errors.some((error) => error.includes("note"))).toBe(true);
    }
  });

  it("tracks window resolution and unresolved components", () => {
    expect(isWindowResolved(record, state.records[0])).toBe(false);
    expect(unresolvedComponents(record, state.records[0])).toHaveLength(4);
    const kept = keepPassAChoices(record, state.records[0]);
    expect(isWindowResolved(record, kept)).toBe(true);
    expect(unresolvedComponents(record, kept)).toHaveLength(0);
    expect(kept.decisions["p2j:adjudicate:pool:w1-aaa:c:0001"]).toEqual({ kind: "KEEP_HUMAN_SET" });
    expect(kept.decisions["p2j:adjudicate:pool:w1-aaa:c:0003"]).toEqual({ kind: "DROP" });
  });
});

describe("deriveResolvedEndpoints", () => {
  const payload = standardPayload();
  const record = payload.records[0];

  it("resolves a full window to exact non-overlapping endpoints", () => {
    const state = buildAdjudicationState(payload);
    const kept = keepPassAChoices(record, state.records[0]);
    const derived = deriveResolvedEndpoints(record, kept);
    expect(derived.errors).toEqual([]);
    expect(derived.endpoints.map((endpoint) => endpoint.provenance_source)).toEqual([
      "HUMAN",
      "HUMAN",
      "SHARED",
    ]);
    expect(derived.endpoints.map((endpoint) => endpoint.exact_bronze_text)).toEqual([
      "one",
      "three four",
      "ten",
    ]);
    expect(derived.endpoints[0].endpoint_id).toBe("p2j:adjudicate:pool:w1-aaa:ep:0001");
  });

  it("accepts Sol alternatives and custom spans", () => {
    const state = buildAdjudicationState(payload);
    const decisions: Record<string, ComponentDecision> = {
      "p2j:adjudicate:pool:w1-aaa:c:0001": { kind: "KEEP_SOL_SET" },
      "p2j:adjudicate:pool:w1-aaa:c:0002": { kind: "DROP" },
      "p2j:adjudicate:pool:w1-aaa:c:0003": {
        kind: "CUSTOM",
        token_start: 6,
        token_end: 6,
        node_type: "QUANTITY",
      },
      "p2j:adjudicate:pool:w1-aaa:c:0004": { kind: "DROP" },
      "p2j:adjudicate:pool:w1-aaa:c:0005": { kind: "KEEP_HUMAN_SET" },
    };
    const derived = deriveResolvedEndpoints(record, { ...state.records[0], decisions });
    expect(derived.errors).toEqual([]);
    expect(derived.endpoints.map((endpoint) => endpoint.exact_bronze_text)).toEqual([
      "one",
      "seven",
      "ten",
    ]);
    expect(derived.endpoints[1].provenance_source).toBe("CUSTOM");
    expect(derived.endpoints[1].node_type).toBe("QUANTITY");
  });

  it("reports overlapping resolved endpoints for a custom span crossing components", () => {
    const state = buildAdjudicationState(payload);
    const decisions: Record<string, ComponentDecision> = {
      "p2j:adjudicate:pool:w1-aaa:c:0001": { kind: "KEEP_HUMAN_SET" },
      "p2j:adjudicate:pool:w1-aaa:c:0002": {
        kind: "CUSTOM",
        token_start: 0,
        token_end: 4,
        node_type: "STATE",
      },
      "p2j:adjudicate:pool:w1-aaa:c:0003": { kind: "DROP" },
      "p2j:adjudicate:pool:w1-aaa:c:0004": { kind: "DROP" },
      "p2j:adjudicate:pool:w1-aaa:c:0005": { kind: "KEEP_HUMAN_SET" },
    };
    const derived = deriveResolvedEndpoints(record, { ...state.records[0], decisions });
    expect(derived.errors.some((error) => error.includes("overlap"))).toBe(true);
  });

  it("rejects keeping a Sol endpoint that has no type", () => {
    const nullTypeRecord = makeAdjudicationRecord("pool:w3-ccc", TEXT_A, [
      { classification: "SOL_ONLY", human: [], sol: [[6, 6, null]] },
    ]);
    const payload = makePayload([nullTypeRecord]);
    const state = buildAdjudicationState(payload);
    const decisions: Record<string, ComponentDecision> = {
      "p2j:adjudicate:pool:w3-ccc:c:0001": { kind: "KEEP_SOL_SET" },
    };
    const derived = deriveResolvedEndpoints(nullTypeRecord, {
      ...state.records[0],
      decisions,
    });
    expect(derived.errors.some((error) => error.includes("no type"))).toBe(true);
  });
});

describe("adjudication export", () => {
  it("builds a valid REVIEW MATERIAL export after full resolution", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const withReviewer = { ...state, reviewer_name: "Ada" };
    const kept = {
      ...withReviewer,
      records: withReviewer.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const result = buildAdjudicationExport(payload, kept, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.export.status_label).toBe("REVIEW_MATERIAL");
    expect(result.export.schema_version).toBe(ADJUDICATION_EXPORT_SCHEMA_VERSION);
    expect(result.export.reviewer_name).toBe("Ada");
    expect(result.export.packet_sha256).toBe("a".repeat(64));
    expect(result.export.records).toHaveLength(2);
    expect(result.export.records[0].resolved_endpoints).toHaveLength(3);
    expect(findAdjudicationForbiddenFields(result.export, ADJUDICATION_EXPORT_FORBIDDEN_KEYS)).toEqual([]);
  });

  it("blocks export while any CLEAN window is unresolved", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const result = buildAdjudicationExport(payload, state, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.errors.some((error) => error.includes("unresolved"))).toBe(true);
  });

  it("requires a note for AMBIGUOUS and EXCLUDED windows", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const ambiguous = {
      ...state,
      records: state.records.map((record, index) =>
        index === 0
          ? { ...record, outcome: "AMBIGUOUS" as const }
          : keepPassAChoices(payload.records[index], record),
      ),
    };
    const blocked = buildAdjudicationExport(payload, ambiguous, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(blocked.ok).toBe(false);
    const withNote = {
      ...ambiguous,
      records: ambiguous.records.map((record, index) =>
        index === 0 ? { ...record, note: "context is genuinely unclear" } : record,
      ),
    };
    const accepted = buildAdjudicationExport(payload, withNote, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(accepted.ok).toBe(true);
    if (!accepted.ok) {
      return;
    }
    expect(accepted.export.records[0].components[0].resolved_by).toBe("WINDOW_AMBIGUOUS");
  });

  it("clears endpoints for EXCLUDED windows and requires a note", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const excluded = {
      ...state,
      records: state.records.map((record, index) =>
        index === 0
          ? { ...record, outcome: "EXCLUDED" as const }
          : keepPassAChoices(payload.records[index], record),
      ),
    };
    expect(buildAdjudicationExport(payload, excluded, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks()).ok).toBe(false);
    const withNote = {
      ...excluded,
      records: excluded.records.map((record, index) =>
        index === 0 ? { ...record, note: "unusable ASR" } : record,
      ),
    };
    const result = buildAdjudicationExport(payload, withNote, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.export.records[0].resolved_endpoints).toEqual([]);
  });

  it("keeps exact agreements pre-resolved by default and records explicit CUSTOM or DROP", () => {
    const payload = standardPayload();
    const exactId = "p2j:adjudicate:pool:w1-aaa:c:0005";
    const state = buildAdjudicationState(payload);
    const kept = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const defaultExport = buildAdjudicationExport(payload, kept, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(defaultExport.ok).toBe(true);
    if (!defaultExport.ok) {
      return;
    }
    const defaultEntry = defaultExport.export.records[0].components.find(
      (entry) => entry.component_id === exactId,
    );
    expect(defaultEntry?.decision).toEqual({ kind: "KEEP_HUMAN_SET" });
    expect(defaultEntry?.resolved_by).toBe("PRE_RESOLVED");

    const dropped = {
      ...kept,
      records: kept.records.map((stateRecord, index) =>
        index === 0
          ? {
              ...stateRecord,
              decisions: { ...stateRecord.decisions, [exactId]: { kind: "DROP" as const } },
            }
          : stateRecord,
      ),
    };
    const dropExport = buildAdjudicationExport(payload, dropped, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(dropExport.ok).toBe(true);
    if (!dropExport.ok) {
      return;
    }
    const dropEntry = dropExport.export.records[0].components.find(
      (entry) => entry.component_id === exactId,
    );
    expect(dropEntry?.decision).toEqual({ kind: "DROP" });
    expect(dropEntry?.resolved_by).toBe("DROP");
    expect(
      dropExport.export.records[0].resolved_endpoints.some(
        (endpoint) => endpoint.exact_bronze_text === "ten",
      ),
    ).toBe(false);

    const custom = {
      ...kept,
      records: kept.records.map((stateRecord, index) =>
        index === 0
          ? {
              ...stateRecord,
              decisions: {
                ...stateRecord.decisions,
                [exactId]: {
                  kind: "CUSTOM" as const,
                  token_start: 5,
                  token_end: 5,
                  node_type: "QUANTITY" as const,
                },
              },
            }
          : stateRecord,
      ),
    };
    const customExport = buildAdjudicationExport(payload, custom, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(customExport.ok).toBe(true);
    if (!customExport.ok) {
      return;
    }
    const customEntry = customExport.export.records[0].components.find(
      (entry) => entry.component_id === exactId,
    );
    expect(customEntry?.decision).toEqual({
      kind: "CUSTOM",
      token_start: 5,
      token_end: 5,
      node_type: "QUANTITY",
    });
    expect(customEntry?.resolved_by).toBe("CUSTOM");
    expect(
      customExport.export.records[0].resolved_endpoints.some(
        (endpoint) =>
          endpoint.provenance_source === "CUSTOM" && endpoint.exact_bronze_text === "six",
      ),
    ).toBe(true);
  });

  it("rejects KEEP_SOL_SET and mismatched resolved_by for exact agreements", () => {
    const payload = standardPayload();
    const exactId = "p2j:adjudicate:pool:w1-aaa:c:0005";
    const state = buildAdjudicationState(payload);
    const invalid = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((stateRecord, index) =>
        index === 0
          ? {
              ...stateRecord,
              decisions: {
                ...stateRecord.decisions,
                [exactId]: { kind: "KEEP_SOL_SET" as const },
              },
            }
          : stateRecord,
      ),
    };
    const stateResult = validateAdjudicationState(invalid, payload);
    expect(stateResult.ok).toBe(false);
    const exportResult = buildAdjudicationExport(payload, invalid, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(exportResult.ok).toBe(false);
    if (!exportResult.ok) {
      expect(exportResult.errors.some((error) => error.includes("invalid decision"))).toBe(true);
    }

    const kept = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const built = buildAdjudicationExport(payload, kept, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }
    const mismatched = JSON.parse(JSON.stringify(built.export)) as typeof built.export;
    const entry = mismatched.records[0].components.find(
      (component) => component.component_id === exactId,
    );
    if (entry) {
      entry.resolved_by = "CUSTOM";
    }
    expect(validateAdjudicationExport(mismatched, payload).ok).toBe(false);
  });

  it("round-trips an export through strict import validation", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const kept = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const built = buildAdjudicationExport(payload, kept, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }
    const validation = validateAdjudicationExport(built.export, payload);
    expect(validation.ok).toBe(true);
    if (!validation.ok) {
      return;
    }
    expect(validation.state.reviewer_name).toBe("Ada");
    expect(validation.state.records[0].decisions).toEqual(kept.records[0].decisions);
    expect(validation.audit_checks).toEqual(allTrueAuditChecks());
  });

  it("rejects tampered imports", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const kept = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const built = buildAdjudicationExport(payload, kept, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }
    const wrongHash = { ...built.export, packet_sha256: "0".repeat(64) };
    expect(validateAdjudicationExport(wrongHash, payload).ok).toBe(false);

    const tamperedEndpoint = JSON.parse(JSON.stringify(built.export)) as typeof built.export;
    tamperedEndpoint.records[0].resolved_endpoints[0].exact_bronze_text = "tampered";
    expect(validateAdjudicationExport(tamperedEndpoint, payload).ok).toBe(false);

    const droppedEntry = JSON.parse(JSON.stringify(built.export)) as typeof built.export;
    droppedEntry.records[0].components = droppedEntry.records[0].components.slice(1);
    expect(validateAdjudicationExport(droppedEntry, payload).ok).toBe(false);
  });

  it("rejects endpoint tampering that keeps derived IDs intact", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const kept = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const built = buildAdjudicationExport(payload, kept, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }
    const clone = (): typeof built.export =>
      JSON.parse(JSON.stringify(built.export)) as typeof built.export;
    const tokens = payload.records[0].tokens;

    const wrongType = clone();
    wrongType.records[0].resolved_endpoints[0].node_type = "ACTION";
    expect(validateAdjudicationExport(wrongType, payload).ok).toBe(false);

    const wrongProvenance = clone();
    wrongProvenance.records[0].resolved_endpoints[0].provenance_source = "SOL";
    expect(validateAdjudicationExport(wrongProvenance, payload).ok).toBe(false);

    const wrongComponent = clone();
    wrongComponent.records[0].resolved_endpoints[0].component_id =
      "p2j:adjudicate:pool:w1-aaa:c:0002";
    expect(validateAdjudicationExport(wrongComponent, payload).ok).toBe(false);

    const wrongSpan = clone();
    wrongSpan.records[0].resolved_endpoints[0].token_start = 5;
    wrongSpan.records[0].resolved_endpoints[0].token_end = 5;
    wrongSpan.records[0].resolved_endpoints[0].char_start = tokens[5].start;
    wrongSpan.records[0].resolved_endpoints[0].char_end = tokens[5].end;
    wrongSpan.records[0].resolved_endpoints[0].exact_bronze_text = tokens[5].text;
    expect(validateAdjudicationExport(wrongSpan, payload).ok).toBe(false);
  });

  it("rejects blank or whitespace reviewer names on import", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const kept = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const built = buildAdjudicationExport(payload, kept, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks());
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }
    expect(validateAdjudicationExport({ ...built.export, reviewer_name: "" }, payload).ok).toBe(false);
    expect(
      validateAdjudicationExport({ ...built.export, reviewer_name: "   " }, payload).ok,
    ).toBe(false);
  });
});

describe("Pass B audit attestation", () => {
  function builtExport() {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const kept = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const result = buildAdjudicationExport(
      payload, kept, "Ada", "2026-08-18T00:00:00Z", allTrueAuditChecks(),
    );
    expect(result.ok).toBe(true);
    if (!result.ok) {
      throw new Error("fixture export failed");
    }
    return { payload, result };
  }

  it("ships the v2 envelope with an exact all-true audit_checks object", () => {
    const { result } = builtExport();
    expect(result.export.schema_version).toBe("phase2j-adjudication-export-v2");
    expect(result.export.audit_checks).toEqual(allTrueAuditChecks());
    expect(AUDIT_CHECKS).toEqual([
      "boundaries",
      "omissions",
      "roles",
      "duplicates",
      "ambiguity",
    ]);
  });

  it("requires all five audit checks to be true before export", () => {
    const { payload, result } = builtExport();
    const state = validateAdjudicationExport(result.export, payload);
    expect(state.ok).toBe(true);
    for (const key of AUDIT_CHECKS) {
      const missing = JSON.parse(JSON.stringify(result.export)) as typeof result.export;
      delete missing.audit_checks[key];
      const blocked = buildAdjudicationExport(
        payload, state.ok ? state.state : buildAdjudicationState(payload),
        "Ada", "2026-08-18T00:00:00Z", missing.audit_checks,
      );
      expect(blocked.ok).toBe(false);
      if (!blocked.ok) {
        expect(blocked.errors.some((error) => error.includes("audit_checks"))).toBe(true);
      }
    }
    const falseCheck = buildAdjudicationExport(
      payload, state.ok ? state.state : buildAdjudicationState(payload),
      "Ada", "2026-08-18T00:00:00Z",
      { ...allTrueAuditChecks(), duplicates: false },
    );
    expect(falseCheck.ok).toBe(false);
    if (!falseCheck.ok) {
      expect(falseCheck.errors.some((error) => error.includes("every audit check"))).toBe(true);
    }
    const extraCheck = buildAdjudicationExport(
      payload, state.ok ? state.state : buildAdjudicationState(payload),
      "Ada", "2026-08-18T00:00:00Z",
      { ...allTrueAuditChecks(), fabricated: true } as unknown as AuditChecks,
    );
    expect(extraCheck.ok).toBe(false);
    if (!extraCheck.ok) {
      expect(extraCheck.errors.some((error) => error.includes("fabricated"))).toBe(true);
    }
    const nonBoolean = buildAdjudicationExport(
      payload, state.ok ? state.state : buildAdjudicationState(payload),
      "Ada", "2026-08-18T00:00:00Z",
      { ...allTrueAuditChecks(), ambiguity: "yes" as unknown as boolean },
    );
    expect(nonBoolean.ok).toBe(false);
  });

  it("rejects missing, false, extra, and non-boolean audit checks on import", () => {
    const { payload, result } = builtExport();
    const clone = (): typeof result.export =>
      JSON.parse(JSON.stringify(result.export)) as typeof result.export;

    const legacyV1 = clone() as unknown as Record<string, unknown>;
    delete legacyV1.audit_checks;
    expect(validateAdjudicationExport(legacyV1, payload).ok).toBe(false);

    const falseImport = clone();
    falseImport.audit_checks.boundaries = false;
    expect(validateAdjudicationExport(falseImport, payload).ok).toBe(false);

    const extraImport = clone();
    extraImport.audit_checks = {
      ...extraImport.audit_checks,
      invented: true,
    } as unknown as AuditChecks;
    expect(validateAdjudicationExport(extraImport, payload).ok).toBe(false);

    const nonBooleanImport = clone();
    nonBooleanImport.audit_checks.omissions = "done" as unknown as boolean;
    expect(validateAdjudicationExport(nonBooleanImport, payload).ok).toBe(false);

    const wrongSchema = clone();
    wrongSchema.schema_version =
      "phase2j-adjudication-export-v1" as unknown as typeof wrongSchema.schema_version;
    expect(validateAdjudicationExport(wrongSchema, payload).ok).toBe(false);
  });

  it("round-trips and parses stored audit checks", () => {
    const { payload, result } = builtExport();
    const validation = validateAdjudicationExport(result.export, payload);
    expect(validation.ok).toBe(true);
    if (!validation.ok) {
      return;
    }
    expect(validation.audit_checks).toEqual(allTrueAuditChecks());
    expect(parseAuditChecks(validation.audit_checks)).toEqual(allTrueAuditChecks());
    expect(parseAuditChecks(emptyAuditChecks())).toBeNull();
    expect(parseAuditChecks({ ...allTrueAuditChecks(), boundaries: false })).toBeNull();
    expect(parseAuditChecks("nonsense")).toBeNull();
  });
});

describe("adjudication state persistence", () => {
  it("round-trips autosave state", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const validation = validateAdjudicationState(
      { ...state, reviewer_name: "Ada" },
      payload,
    );
    expect(validation.ok).toBe(true);
    if (!validation.ok) {
      return;
    }
    expect(validation.state.schema_version).toBe(ADJUDICATION_STATE_SCHEMA_VERSION);
    expect(validation.state.records[0].decisions).toEqual(state.records[0].decisions);
  });

  it("rejects a state bound to a different packet", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const result = validateAdjudicationState(
      { ...state, adjudication_packet_sha256: "f".repeat(64) },
      payload,
    );
    expect(result.ok).toBe(false);
  });

  it("rejects unknown decision keys in a state record", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const tampered = {
      ...state,
      reviewer_name: "Ada",
      records: state.records.map((record, index) =>
        index === 0
          ? {
              ...record,
              decisions: {
                ...record.decisions,
                "p2j:adjudicate:pool:w1-aaa:c:9999": { kind: "DROP" as const },
                "not-a-component": { kind: "KEEP_HUMAN_SET" as const },
              },
            }
          : record,
      ),
    };
    const result = validateAdjudicationState(tampered, payload);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.some((error) => error.includes("c:9999"))).toBe(true);
      expect(result.errors.some((error) => error.includes("not-a-component"))).toBe(true);
    }
  });

  it("keeps the Pass B attestation out of the v1 component autosave state", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const tampered = {
      ...state,
      reviewer_name: "Ada",
      audit_checks: allTrueAuditChecks(),
    };
    expect(validateAdjudicationState(tampered, payload).ok).toBe(false);
    const clean = { ...state, reviewer_name: "Ada" };
    const result = validateAdjudicationState(clean, payload);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.state.records).toEqual(state.records);
    }
  });
});

describe("progress", () => {
  it("counts resolved windows and components", () => {
    const payload = standardPayload();
    const state = buildAdjudicationState(payload);
    const progress = summarizeAdjudicationProgress(payload, state);
    expect(progress.windows).toBe(2);
    expect(progress.components).toBe(10);
    expect(progress.resolved_windows).toBe(0);
    expect(progress.resolved_components).toBe(2);
    const kept = {
      ...state,
      records: state.records.map((stateRecord, index) =>
        keepPassAChoices(payload.records[index], stateRecord),
      ),
    };
    const next = summarizeAdjudicationProgress(payload, kept);
    expect(next.resolved_windows).toBe(2);
    expect(next.resolved_components).toBe(10);
  });
});

describe("sanitizeAdjudicationPacket", () => {
  it("rejects invalid envelopes and forbidden fields", () => {
    const raw = {
      content_sha256: "b".repeat(64),
      schema_version: ADJUDICATION_PACKET_SCHEMA_VERSION,
      adjudication_version: ADJUDICATION_VERSION,
      annotation_version: "phase2j-endpoint-annotation-v1",
      packet_schema_version: "phase2j-endpoint-annotation-packet-v1",
      packet_sha256: "a".repeat(64),
      human_session_schema_version: "phase2j-review-session-v1",
      human_session_sha256: "c".repeat(64),
      sol_review_schema_version: "phase2j-sol-parallel-review-v1",
      sol_review_sha256: "d".repeat(64),
      visibility_gate: "SOL_VISIBLE_FOR_ADJUDICATION",
      purpose: "adjudication deck",
      totals: {},
      records: [],
    };
    expect(() => sanitizeAdjudicationPacket(raw)).toThrow(/totals/);
    expect(() =>
      sanitizeAdjudicationPacket({
        ...raw,
        visibility_gate: "SEALED_UNTIL_HUMAN_PASS_A_COMPLETE",
      }),
    ).toThrow(/visibility gate/);
  });

  it("sanitizes the locked adjudication packet", () => {
    const packetPath = path.resolve(
      process.cwd(),
      "../../data/phase2j/phase2j-adjudication-packet-v1.json",
    );
    const raw = JSON.parse(readFileSync(packetPath, "utf8")) as unknown;
    const payload = sanitizeAdjudicationPacket(raw);
    expect(payload.records).toHaveLength(30);
    expect(payload.totals.components).toBe(326);
    expect(payload.totals.exact_agreements).toBe(49);
    expect(payload.totals.type_disagreements).toBe(16);
    expect(payload.totals.boundary_disagreements).toBe(87);
    expect(payload.totals.sol_only).toBe(174);
    expect(payload.totals.human_only).toBe(0);
    expect(payload.totals.human_endpoints).toBe(166);
    expect(payload.totals.sol_endpoints).toBe(338);
    expect(payload.packet_sha256).toBe(
      "3f766b08696ed512063d999c75877001d77b03db136f8edae78e631e1725c62a",
    );
    expect(payload.adjudication_packet_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(
      findAdjudicationForbiddenFields(payload, ADJUDICATION_EXPORT_FORBIDDEN_KEYS),
    ).toEqual([]);
    for (const record of payload.records) {
      for (const component of record.components) {
        expect(component.human_endpoint_ids.length + component.sol_endpoint_ids.length).toBeGreaterThan(0);
      }
    }
  });
});
