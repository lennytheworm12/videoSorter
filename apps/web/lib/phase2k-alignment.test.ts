import { createHash } from "node:crypto";
import {
  addSpan,
  ALIGNMENT_DECISION_STATES,
  buildDecisionsMap,
  buildSessionExport,
  buildSessionFromPacket,
  canonicalSerialize,
  completeItem,
  computeCanonicalSha256,
  CORRECTION_STATUSES,
  DECISIONS_FILENAME,
  decisionsMapErrors,
  findExactOccurrences,
  itemMissingFields,
  PACKET_SCHEMA_VERSION,
  removeSpan,
  sanitizePacket,
  SESSION_SCHEMA_VERSION,
  setAllReviewers,
  setItemNotes,
  setItemReviewer,
  setItemState,
  summarizeProgress,
  TARGET_COUNT,
  TARGET_WINDOW_COUNT,
  textSha256,
  uncompleteItem,
  validateSessionInput,
  type AlignmentPacket,
  type AlignmentSession,
  type DecisionState,
  type PolishedSpan,
  type Sha256Digest,
} from "./phase2k-alignment";

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

const NODE_TYPE_CYCLE = [
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

function windowItemCounts(): number[] {
  // 21 items in the first window, 10 in the remaining 29 -> 311 total.
  return [21, ...Array<number>(TARGET_WINDOW_COUNT - 1).fill(10)];
}

function buildFixtureItem(
  globalIndex: number,
  windowIndex: number,
  windowId: string,
): Record<string, unknown> {
  const corrected = globalIndex < 48;
  const period = globalIndex < 28;
  const nodeType =
    globalIndex === 57 ? null : NODE_TYPE_CYCLE[globalIndex % NODE_TYPE_CYCLE.length];
  const originalText = corrected
    ? `bronze fragment ${globalIndex}${period ? "." : ","}`
    : `bronze fragment ${globalIndex}`;
  const evaluationText = corrected ? originalText.slice(0, -1) : originalText;
  const originalStart = 100 + globalIndex * 3;
  const originalEnd = originalStart + originalText.length;
  const sourceAbsoluteStart = 5000 + globalIndex * 13;
  const sourceAbsoluteEnd = sourceAbsoluteStart + originalText.length;
  const polishedText =
    `Polished ${globalIndex}: ${originalText} ` +
    "polished content filler ".repeat(16);
  const cleanTranscript = `Clean transcript for ${windowId} item ${globalIndex}.`;
  const endpointId = `p2j:${windowId}:ep:${String(windowIndex + 1).padStart(4, "0")}`;
  return {
    alignment_id: `p2k:align:${endpointId}`,
    window_id: windowId,
    endpoint_id: endpointId,
    node_type: nodeType,
    bronze_target: {
      original_start: originalStart,
      original_end: originalEnd,
      original_text: originalText,
      source_absolute_start: sourceAbsoluteStart,
      source_absolute_end: sourceAbsoluteEnd,
      evaluation_start: originalStart,
      evaluation_end: corrected ? originalEnd - 1 : originalEnd,
      evaluation_text: evaluationText,
      correction_status: corrected
        ? "TERMINAL_PUNCTUATION_DROPPED"
        : "UNCHANGED",
      dropped_text: corrected ? (period ? "." : ",") : null,
    },
    representation: {
      clean_target_transcript: cleanTranscript,
      clean_target_transcript_sha256: sha256Hex(cleanTranscript),
      polished_text: polishedText,
      polished_text_sha256: sha256Hex(polishedText),
    },
    decision: {
      state: null,
      polished_spans: [],
      reviewer: null,
      completed_at: null,
      notes: [],
    },
  };
}

function buildBlankPacket(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  const windowIds = Array.from(
    { length: TARGET_WINDOW_COUNT },
    (_, index) => `pool:w${String(index + 1).padStart(5, "0")}`,
  );
  const counts = windowItemCounts();
  const items: Record<string, unknown>[] = [];
  let globalIndex = 0;
  windowIds.forEach((windowId, windowIndex) => {
    for (let local = 0; local < counts[windowIndex]; local += 1) {
      items.push(buildFixtureItem(globalIndex, local, windowId));
      globalIndex += 1;
    }
  });
  const bindingHex = (label: string) => sha256Hex(`binding:${label}`);
  const body = {
    schema_version: PACKET_SCHEMA_VERSION,
    purpose:
      "Scorer/model-blind Phase 2K downstream semantic-target alignment packet. " +
      "Carries no downstream predictions, model results, scores, or semantic extraction.",
    release_gate: "AWAITING_HUMAN_REVIEW",
    dataset_binding: {
      phase2k_records_sha256: bindingHex("phase2k_records"),
      phase2j_reviewed_packet_sha256: bindingHex("phase2j_reviewed_packet"),
      phase2j_coverage_sha256: bindingHex("phase2j_coverage"),
      finalized_human_packet_sha256: bindingHex("finalized_human_packet"),
      human_summary_sha256: bindingHex("human_summary"),
      completed_transformation_audit_sha256: bindingHex("completed_audit"),
      window_ids_sha256: canonicalSha256(windowIds),
      window_count: TARGET_WINDOW_COUNT,
      target_count: TARGET_COUNT,
      human_review_gate_status: "PASSED",
    },
    boundary_rule: {
      rule_version: "phase2k-target-boundary-rule-v1-phase2j-terminal-punctuation",
      unchanged_count: 263,
      corrected_count: 48,
      dropped_terminal_period_count: 28,
      dropped_terminal_comma_count: 20,
      behavior:
        "The exact 48 Phase 2J candidate-coverage-identified missing endpoints drop exactly " +
        "one terminal '.' or ',' from the evaluation span; the other 263 keep the reviewed span.",
    },
    items,
    ...overrides,
  };
  return {
    content_sha256: canonicalSha256(body),
    ...body,
  };
}

function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** Recompute the canonical content hash after a mutation so the targeted
 * structural rejection is reached instead of the hash-mismatch rejection. */
function mutate(
  raw: Record<string, unknown>,
  mutateFn: (copy: Record<string, unknown>) => void,
): Record<string, unknown> {
  const copy = deepClone(raw);
  mutateFn(copy);
  const { content_sha256: _ignored, ...body } = copy;
  return { content_sha256: canonicalSha256(body), ...body };
}

async function sanitizeBlank(
  raw: Record<string, unknown>,
): Promise<AlignmentPacket> {
  return sanitizePacket(raw, digest);
}

function defaultCompletedSession(
  packet: AlignmentPacket,
  overrides: Partial<Record<string, unknown>> = {},
): AlignmentSession {
  let session = buildSessionFromPacket(packet);
  session = setAllReviewers(session, "tester");
  const localByWindow = new Map<string, number>();
  for (const item of session.items) {
    const local = localByWindow.get(item.window_id) ?? 0;
    localByWindow.set(item.window_id, local + 1);
    const state: DecisionState = (
      ["ABSENT", "ALIGNED", "AMBIGUOUS", "ALIGNED", "MULTIPLE_CANDIDATES", "ALIGNED"] as const
    )[local % 6];
    const polished = item.representation.polished_text;
    const primaryStart = 1 + local * 5;
    const primary: PolishedSpan = {
      start: primaryStart,
      end: primaryStart + 2,
      text: polished.slice(primaryStart, primaryStart + 2),
    };
    let spans: PolishedSpan[] = [primary];
    if (state === "ABSENT") {
      spans = [];
    } else if (state === "MULTIPLE_CANDIDATES") {
      const secondaryStart = 60 + local * 5;
      spans = [
        primary,
        {
          start: secondaryStart,
          end: secondaryStart + 2,
          text: polished.slice(secondaryStart, secondaryStart + 2),
        },
      ];
    }
    session = setItemState(session, item.alignment_id, state);
    for (const span of spans) {
      session = addSpan(session, item.alignment_id, span);
    }
  }
  for (const item of session.items) {
    const result = completeItem(session, item.alignment_id, "2026-08-19T00:00:00.000Z");
    if (!result.ok) {
      throw new Error(`fixture completion failed for ${item.alignment_id}: ${result.errors.join("; ")}`);
    }
    session = result.session;
  }
  return deepClone({ ...session, ...overrides });
}

describe("canonical serialization", () => {
  it("matches Python-style sorted-key compact JSON", () => {
    expect(canonicalSerialize({ b: 2, a: [1, "x", null, true] })).toBe(
      '{"a":[1,"x",null,true],"b":2}',
    );
    expect(canonicalSerialize({ z: "αβ", y: { q: 0 } })).toBe('{"y":{"q":0},"z":"αβ"}');
  });
});

describe("blank packet sanitization", () => {
  it("accepts a valid strict packet and verifies the canonical content hash", async () => {
    const raw = buildBlankPacket();
    const packet = await sanitizeBlank(raw);
    expect(packet.schema_version).toBe(PACKET_SCHEMA_VERSION);
    expect(packet.release_gate).toBe("AWAITING_HUMAN_REVIEW");
    expect(packet.items).toHaveLength(TARGET_COUNT);
    expect(packet.dataset_binding.target_count).toBe(311);
    expect(packet.dataset_binding.window_count).toBe(30);
    expect(packet.dataset_binding.human_review_gate_status).toBe("PASSED");
    const recomputed = await computeCanonicalSha256(
      Object.fromEntries(
        Object.entries(raw).filter(([key]) => key !== "content_sha256"),
      ),
      digest,
    );
    expect(packet.content_sha256).toBe(recomputed);
    expect(await computeCanonicalSha256(
      Object.fromEntries(
        Object.entries(packet).filter(([key]) => key !== "content_sha256"),
      ),
      digest,
    )).toBe(packet.content_sha256);
  });

  it("rejects wrong top-level keys", async () => {
    const extra = mutate(buildBlankPacket(), (copy) => {
      (copy as Record<string, unknown>).forged_field = true;
    });
    await expect(sanitizeBlank(extra)).rejects.toThrow("top-level keys are invalid");
    const missing = mutate(buildBlankPacket(), (copy) => {
      delete copy.items;
    });
    await expect(sanitizeBlank(missing)).rejects.toThrow("top-level keys are invalid");
    const itemExtra = mutate(buildBlankPacket(), (copy) => {
      ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>).extra_field = {};
    });
    await expect(sanitizeBlank(itemExtra)).rejects.toThrow("keys are invalid");
  });

  it("rejects wrong schema version", async () => {
    const raw = mutate(buildBlankPacket(), (copy) => {
      copy.schema_version = "phase2k-downstream-alignment-packet-v2";
    });
    await expect(sanitizeBlank(raw)).rejects.toThrow("schema_version");
  });

  it("rejects a finalized release gate", async () => {
    const raw = mutate(buildBlankPacket(), (copy) => {
      copy.release_gate = "REVIEWED";
    });
    await expect(sanitizeBlank(raw)).rejects.toThrow("AWAITING_HUMAN_REVIEW");
  });

  it("rejects a tampered content hash and non-hex hashes", async () => {
    const raw = buildBlankPacket();
    raw.purpose = "tampered purpose";
    await expect(sanitizeBlank(raw)).rejects.toThrow("content_sha256");
    const badHex = buildBlankPacket({ content_sha256: "not-a-hash" });
    await expect(sanitizeBlank(badHex)).rejects.toThrow("64-character");
  });

  it("rejects wrong item/window/target counts", async () => {
    const short = mutate(buildBlankPacket(), (copy) => {
      copy.items = (copy.items as unknown[]).slice(0, 310);
    });
    await expect(sanitizeBlank(short)).rejects.toThrow("exactly 311 items");

    const bindingWrong = mutate(buildBlankPacket(), (copy) => {
      (copy.dataset_binding as Record<string, unknown>).target_count = 310;
    });
    await expect(sanitizeBlank(bindingWrong)).rejects.toThrow("must be 311");

    const windowsWrong = mutate(buildBlankPacket(), (copy) => {
      (copy.dataset_binding as Record<string, unknown>).window_count = 29;
    });
    await expect(sanitizeBlank(windowsWrong)).rejects.toThrow("must be 30");
  });

  it("rejects wrong window identity and window hash inconsistency", async () => {
    const duplicateWindow = mutate(buildBlankPacket(), (copy) => {
      const items = copy.items as Record<string, unknown>[];
      items[10].window_id = "pool:forged-window";
    });
    await expect(sanitizeBlank(duplicateWindow)).rejects.toThrow("30 windows");

    const staleHash = mutate(buildBlankPacket(), (copy) => {
      (copy.dataset_binding as Record<string, unknown>).window_ids_sha256 = "0".repeat(64);
    });
    await expect(sanitizeBlank(staleHash)).rejects.toThrow("window_ids_sha256");
  });

  it("rejects duplicate endpoint/alignment IDs and broken ID derivation", async () => {
    const duplicateEndpoint = mutate(buildBlankPacket(), (copy) => {
      const items = copy.items as Record<string, unknown>[];
      items[5].endpoint_id = items[0].endpoint_id;
      items[5].alignment_id = `p2k:align:${items[0].endpoint_id}`;
    });
    await expect(sanitizeBlank(duplicateEndpoint)).rejects.toThrow(/unique|derive/);

    const brokenDerivation = mutate(buildBlankPacket(), (copy) => {
      ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>).alignment_id =
        "p2k:align:other";
    });
    await expect(sanitizeBlank(brokenDerivation)).rejects.toThrow("derive from endpoint_id");
  });

  it("rejects representation text-hash mismatches", async () => {
    const cleanWrong = mutate(buildBlankPacket(), (copy) => {
      (
        ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>)
          .representation as Record<string, unknown>
      ).clean_target_transcript = "tampered clean transcript";
    });
    await expect(sanitizeBlank(cleanWrong)).rejects.toThrow("clean_target_transcript_sha256");

    const polishedWrong = mutate(buildBlankPacket(), (copy) => {
      (
        ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>)
          .representation as Record<string, unknown>
      ).polished_text = "tampered polished text";
    });
    await expect(sanitizeBlank(polishedWrong)).rejects.toThrow("polished_text_sha256");
  });

  it("verifies boundary 263/48/28/20 and correction slice invariants", async () => {
    const packet = await sanitizeBlank(buildBlankPacket());
    const unchanged = packet.items.filter(
      (item) => item.bronze_target.correction_status === "UNCHANGED",
    );
    const corrected = packet.items.filter(
      (item) => item.bronze_target.correction_status === "TERMINAL_PUNCTUATION_DROPPED",
    );
    expect(unchanged).toHaveLength(263);
    expect(corrected).toHaveLength(48);
    expect(corrected.filter((item) => item.bronze_target.dropped_text === ".")).toHaveLength(28);
    expect(corrected.filter((item) => item.bronze_target.dropped_text === ",")).toHaveLength(20);
    for (const item of corrected) {
      const target = item.bronze_target;
      expect(target.evaluation_start).toBe(target.original_start);
      expect(target.evaluation_end).toBe(target.original_end - 1);
      expect(target.evaluation_text).toBe(target.original_text.slice(0, -1));
      expect(target.dropped_text).toBe(target.original_text.slice(-1));
      expect(target.source_absolute_end - target.source_absolute_start).toBe(
        target.original_end - target.original_start,
      );
    }
    for (const item of unchanged) {
      const target = item.bronze_target;
      expect(target.dropped_text).toBeNull();
      expect(target.evaluation_text).toBe(target.original_text);
      expect(target.evaluation_end).toBe(target.original_end);
    }
  });

  it("rejects wrong boundary rule counts and tampered correction slices", async () => {
    const wrongRule = mutate(buildBlankPacket(), (copy) => {
      (copy.boundary_rule as Record<string, unknown>).unchanged_count = 264;
    });
    await expect(sanitizeBlank(wrongRule)).rejects.toThrow("unchanged_count");

    const tampered = mutate(buildBlankPacket(), (copy) => {
      const target = (
        ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>)
          .bronze_target as Record<string, unknown>
      );
      target.evaluation_end = (target.evaluation_end as number) + 1;
    });
    await expect(sanitizeBlank(tampered)).rejects.toThrow("corrected evaluation span");
  });

  it("accepts null node type and rejects unknown node types", async () => {
    const packet = await sanitizeBlank(buildBlankPacket());
    const nullItems = packet.items.filter((item) => item.node_type === null);
    expect(nullItems).toHaveLength(1);

    const unknownType = mutate(buildBlankPacket(), (copy) => {
      ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>).node_type =
        "ARCHITECTURE";
    });
    await expect(sanitizeBlank(unknownType)).rejects.toThrow("node_type is invalid");
  });

  it("rejects non-blank decisions", async () => {
    const withState = mutate(buildBlankPacket(), (copy) => {
      (
        ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>)
          .decision as Record<string, unknown>
      ).state = "ALIGNED";
    });
    await expect(sanitizeBlank(withState)).rejects.toThrow("null state");

    const withSpans = mutate(buildBlankPacket(), (copy) => {
      (
        ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>)
          .decision as Record<string, unknown>
      ).polished_spans = [{ start: 0, end: 1, text: "P" }];
    });
    await expect(sanitizeBlank(withSpans)).rejects.toThrow("empty polished spans");

    const withReviewer = mutate(buildBlankPacket(), (copy) => {
      (
        ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>)
          .decision as Record<string, unknown>
      ).reviewer = "someone";
    });
    await expect(sanitizeBlank(withReviewer)).rejects.toThrow("null reviewer");

    const withNotes = mutate(buildBlankPacket(), (copy) => {
      (
        ((copy.items as Record<string, unknown>[])[0] as Record<string, unknown>)
          .decision as Record<string, unknown>
      ).notes = ["x"];
    });
    await expect(sanitizeBlank(withNotes)).rejects.toThrow("empty notes");
  });

  it("rejects forbidden model/scorer/prediction leakage but preserves prose", async () => {
    const packet = await sanitizeBlank(buildBlankPacket());
    expect(packet.purpose).toContain("predictions");
    expect(packet.purpose).toContain("Scorer/model-blind");

    const topLevelScores = buildBlankPacket({ scores: {} });
    await expect(sanitizeBlank(topLevelScores)).rejects.toThrow("forbidden");

    const nested = buildBlankPacket();
    (
      (nested.items as Record<string, unknown>[])[0].representation as Record<string, unknown>
    ).model_predictions = [];
    await expect(sanitizeBlank(nested)).rejects.toThrow("forbidden");

    const forbiddenValue = buildBlankPacket();
    (
      (forbiddenValue.items as Record<string, unknown>[])[0].decision as Record<string, unknown>
    ).notes = ["PREDICTED"];
    await expect(sanitizeBlank(forbiddenValue)).rejects.toThrow("forbidden value");
  });
});

describe("sessions", () => {
  let packet: AlignmentPacket;

  beforeAll(async () => {
    packet = await sanitizeBlank(buildBlankPacket());
  });

  it("builds a blank session bound to the packet content hash with no timestamps", () => {
    const session = buildSessionFromPacket(packet);
    expect(session.schema_version).toBe(SESSION_SCHEMA_VERSION);
    expect(session.packet_schema_version).toBe(PACKET_SCHEMA_VERSION);
    expect(session.packet_sha256).toBe(packet.content_sha256);
    expect(session.exported_at).toBeNull();
    expect(session.items).toHaveLength(311);
    expect(session.items.every((item) => item.decision.state === null)).toBe(true);
    expect(session.items.every((item) => item.decision.completed_at === null)).toBe(true);
    expect(session.items.every((item) => item.decision.complete === false)).toBe(true);
  });

  it("validates session packet binding and immutable packet content on import", () => {
    const session = buildSessionFromPacket(packet);
    const exported = buildSessionExport(session, "2026-08-19T12:00:00.000Z");
    const result = validateSessionInput(exported, packet);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.session.packet_sha256).toBe(packet.content_sha256);
      expect(result.session.exported_at).toBe("2026-08-19T12:00:00.000Z");
    }

    const wrongPacket = deepClone(packet);
    wrongPacket.content_sha256 = "1".repeat(64);
    const wrongBinding = validateSessionInput(exported, wrongPacket);
    expect(wrongBinding.ok).toBe(false);
    if (!wrongBinding.ok) {
      expect(wrongBinding.errors.join(" ")).toContain("packet_sha256");
    }

    const tampered = deepClone(exported);
    tampered.items[3].bronze_target.original_text = "forged";
    const tamperedResult = validateSessionInput(tampered, packet);
    expect(tamperedResult.ok).toBe(false);
    if (!tamperedResult.ok) {
      expect(tamperedResult.errors.join(" ")).toContain("bronze_target");
    }
  });

  it("never fabricates completed_at at import", () => {
    const session = buildSessionFromPacket(packet);
    const forged = deepClone(session);
    forged.items[0].decision.completed_at = "2026-08-19T00:00:00.000Z";
    const result = validateSessionInput(forged, packet);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toContain("explicitly marked complete");
    }

    const incompleteWithTimestamp = deepClone(session);
    incompleteWithTimestamp.items[0].decision.complete = true;
    const missingTimestamp = validateSessionInput(incompleteWithTimestamp, packet);
    expect(missingTimestamp.ok).toBe(false);
    if (!missingTimestamp.ok) {
      expect(missingTimestamp.errors.join(" ")).toContain("completed_at is required");
    }
  });

  it("enforces span exactness, integer/range bounds, sorting, and uniqueness", () => {
    let session = buildSessionFromPacket(packet);
    const item = session.items[0];
    const polished = item.representation.polished_text;
    session = setItemState(session, item.alignment_id, "ALIGNED");

    expect(() =>
      addSpan(session, item.alignment_id, { start: 0, end: 4, text: "WRONG" }),
    ).toThrow("exact half-open slice");
    expect(() =>
      addSpan(session, item.alignment_id, { start: 1.5, end: 4, text: polished.slice(1.5, 4) }),
    ).toThrow("integer");
    expect(() =>
      addSpan(session, item.alignment_id, { start: true as unknown as number, end: 4, text: "" }),
    ).toThrow("integer");
    expect(() =>
      addSpan(session, item.alignment_id, { start: 4, end: 4, text: polished.slice(4, 4) }),
    ).toThrow("out of bounds");
    expect(() =>
      addSpan(session, item.alignment_id, {
        start: polished.length - 1,
        end: polished.length + 1,
        text: polished.slice(polished.length - 1),
      }),
    ).toThrow("out of bounds");

    session = addSpan(session, item.alignment_id, {
      start: 8,
      end: 12,
      text: polished.slice(8, 12),
    });
    session = addSpan(session, item.alignment_id, {
      start: 2,
      end: 5,
      text: polished.slice(2, 5),
    });
    expect(
      session.items[0].decision.polished_spans.map((span) => span.start),
    ).toEqual([2, 8]);
    expect(() =>
      addSpan(session, item.alignment_id, { start: 2, end: 5, text: polished.slice(2, 5) }),
    ).toThrow("already has a span");
  });

  it("requires a state before spans and rejects spans on ABSENT", () => {
    const session = buildSessionFromPacket(packet);
    const item = session.items[0];
    expect(() =>
      addSpan(session, item.alignment_id, { start: 0, end: 2, text: "Po" }),
    ).toThrow("set an alignment state");
    const absent = setItemState(session, item.alignment_id, "ABSENT");
    expect(() =>
      addSpan(absent, item.alignment_id, { start: 0, end: 2, text: "Po" }),
    ).toThrow("ABSENT decisions cannot carry");
    expect(absent.items[0].decision.polished_spans).toEqual([]);
  });

  it("enforces all completion cardinalities", () => {
    let session = buildSessionFromPacket(packet);
    const item = session.items[0];
    session = setItemReviewer(session, item.alignment_id, "tester");

    let result = completeItem(session, item.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toContain("state");
    }

    session = setItemState(session, item.alignment_id, "ALIGNED");
    result = completeItem(session, item.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toContain("at least one polished span");
    }

    const polished = item.representation.polished_text;
    session = addSpan(session, item.alignment_id, {
      start: 1,
      end: 3,
      text: polished.slice(1, 3),
    });
    result = completeItem(session, item.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(result.ok).toBe(true);

    session = uncompleteItem(result.ok ? result.session : session, item.alignment_id);
    session = setItemState(session, item.alignment_id, "AMBIGUOUS");
    result = completeItem(session, item.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(result.ok).toBe(true);

    session = uncompleteItem(result.ok ? result.session : session, item.alignment_id);
    session = setItemState(session, item.alignment_id, "MULTIPLE_CANDIDATES");
    result = completeItem(session, item.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toContain("at least two polished spans");
    }
    session = addSpan(session, item.alignment_id, {
      start: 30,
      end: 33,
      text: polished.slice(30, 33),
    });
    result = completeItem(session, item.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.session.items[0].decision.completed_at).toBe("2026-08-19T00:00:00.000Z");
    }
  });

  it("edits clear completion and completed_at", () => {
    let session = buildSessionFromPacket(packet);
    const item = session.items[0];
    session = setItemReviewer(session, item.alignment_id, "tester");
    session = setItemState(session, item.alignment_id, "AMBIGUOUS");
    const result = completeItem(session, item.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    session = result.session;
    expect(session.items[0].decision.complete).toBe(true);

    session = setItemReviewer(session, item.alignment_id, "other");
    expect(session.items[0].decision.complete).toBe(false);
    expect(session.items[0].decision.completed_at).toBeNull();

    session = setItemReviewer(session, item.alignment_id, "tester");
    const second = completeItem(session, item.alignment_id, "2026-08-19T01:00:00.000Z");
    expect(second.ok).toBe(true);
    session = second.ok ? second.session : session;

    session = setItemState(session, item.alignment_id, "ALIGNED");
    expect(session.items[0].decision.complete).toBe(false);
    expect(session.items[0].decision.completed_at).toBeNull();

    session = setItemState(session, item.alignment_id, "AMBIGUOUS");
    const third = completeItem(session, item.alignment_id, "2026-08-19T02:00:00.000Z");
    session = third.ok ? third.session : session;
    session = setItemNotes(session, item.alignment_id, ["note"]);
    expect(session.items[0].decision.complete).toBe(false);
    expect(session.items[0].decision.completed_at).toBeNull();

    session = setItemNotes(session, item.alignment_id, []);
    const fourth = completeItem(session, item.alignment_id, "2026-08-19T03:00:00.000Z");
    session = fourth.ok ? fourth.session : session;
    const polished = item.representation.polished_text;
    session = setItemState(session, item.alignment_id, "ALIGNED");
    session = addSpan(session, item.alignment_id, {
      start: 2,
      end: 5,
      text: polished.slice(2, 5),
    });
    expect(session.items[0].decision.complete).toBe(false);
    expect(session.items[0].decision.completed_at).toBeNull();

    session = removeSpan(session, item.alignment_id, 0);
    expect(session.items[0].decision.polished_spans).toEqual([]);
    expect(() => removeSpan(session, item.alignment_id, 0)).toThrow("out of range");
  });

  it("rejects the same exact span assigned to two endpoint IDs in a window", () => {
    const session = buildSessionFromPacket(packet);
    const first = session.items[0];
    const second = session.items[1];
    expect(first.window_id).toBe(second.window_id);
    const polished = first.representation.polished_text;
    const span = { start: 0, end: 2, text: polished.slice(0, 2) };

    let working = setAllReviewers(session, "tester");
    working = setItemState(working, first.alignment_id, "ALIGNED");
    working = addSpan(working, first.alignment_id, span);
    const firstComplete = completeItem(working, first.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(firstComplete.ok).toBe(true);
    working = firstComplete.ok ? firstComplete.session : working;

    working = setItemState(working, second.alignment_id, "ALIGNED");
    working = addSpan(working, second.alignment_id, span);
    const secondComplete = completeItem(working, second.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(secondComplete.ok).toBe(false);
    if (!secondComplete.ok) {
      expect(secondComplete.errors.join(" ")).toContain("already assigned");
    }
    working = setItemState(working, second.alignment_id, "ABSENT");
    working = setItemState(working, second.alignment_id, "ALIGNED");
    working = addSpan(working, second.alignment_id, {
      start: 5,
      end: 8,
      text: polished.slice(5, 8),
    });
    const unique = completeItem(working, second.alignment_id, "2026-08-19T00:00:00.000Z");
    expect(unique.ok).toBe(true);
  });

  it("summarizes progress across states", () => {
    let session = buildSessionFromPacket(packet);
    expect(summarizeProgress(session)).toEqual({
      total: 311,
      complete: 0,
      ready: 0,
      in_progress: 0,
      untouched: 311,
    });
    const first = session.items[0];
    session = setItemState(session, first.alignment_id, "AMBIGUOUS");
    session = setItemReviewer(session, first.alignment_id, "tester");
    expect(summarizeProgress(session).ready).toBe(1);
    const result = completeItem(session, first.alignment_id, "2026-08-19T00:00:00.000Z");
    session = result.ok ? result.session : session;
    expect(summarizeProgress(session)).toMatchObject({ complete: 1, ready: 0 });
  });

  it("builds the exact decisions map only when all 311 items are complete", () => {
    let session = buildSessionFromPacket(packet);
    expect(() => buildDecisionsMap(session)).toThrow("every item complete");

    session = defaultCompletedSession(packet);
    expect(decisionsMapErrors(session)).toEqual([]);
    expect(summarizeProgress(session)).toMatchObject({ complete: 311 });
    const map = buildDecisionsMap(session);
    expect(Object.keys(map)).toEqual(packet.items.map((item) => item.alignment_id));
    for (const item of packet.items) {
      const entry = map[item.alignment_id];
      expect(ALIGNMENT_DECISION_STATES).toContain(entry.state);
      expect(entry.reviewer).toBe("tester");
      expect(entry.completed_at).toBe("2026-08-19T00:00:00.000Z");
      expect(Array.isArray(entry.notes)).toBe(true);
      for (const span of entry.polished_spans) {
        expect(item.representation.polished_text.slice(span.start, span.end)).toBe(span.text);
      }
    }
  });

  it("rejects a session import with a cross-target duplicate span", () => {
    const session = defaultCompletedSession(packet);
    const forged = deepClone(session);
    const firstIndex = forged.items.findIndex(
      (item) => item.decision.polished_spans.length > 0,
    );
    const first = forged.items[firstIndex];
    const secondIndex = forged.items.findIndex(
      (item, index) =>
        index !== firstIndex && item.window_id === first.window_id,
    );
    const second = forged.items[secondIndex];
    const span = first.decision.polished_spans[0];
    second.decision.state = "ALIGNED";
    second.decision.polished_spans = [
      {
        start: span.start,
        end: span.end,
        text: second.representation.polished_text.slice(span.start, span.end),
      },
    ];
    second.decision.complete = true;
    const result = validateSessionInput(forged, packet);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toContain("cross-target duplicate");
    }
    expect(decisionsMapErrors(forged as AlignmentSession).join(" ")).toContain(
      "cross-target duplicate",
    );
  });

  it("validates itemMissingFields for each decision shape", () => {
    let session = buildSessionFromPacket(packet);
    const item = session.items[0];
    expect(itemMissingFields(item)).toContain("state");
    session = setItemReviewer(session, item.alignment_id, "tester");
    session = setItemState(session, item.alignment_id, "ALIGNED");
    expect(itemMissingFields(session.items[0]).join(" ")).toContain("at least one polished span");
  });
});

describe("exact-target occurrence helper", () => {
  it("returns unique, multiple, and none behavior", () => {
    expect(findExactOccurrences("the target appears once here", "target")).toEqual([
      { start: 4, end: 10, text: "target" },
    ]);
    expect(findExactOccurrences("abc abc abc", "abc")).toHaveLength(3);
    expect(findExactOccurrences("aaa", "aa")).toEqual([
      { start: 0, end: 2, text: "aa" },
      { start: 1, end: 3, text: "aa" },
    ]);
    expect(findExactOccurrences("nothing here", "absent")).toEqual([]);
    expect(findExactOccurrences("anything", "")).toEqual([]);
    const spans = findExactOccurrences("x.y.x", ".");
    expect(spans.every((span, index) => span.start < span.end)).toBe(true);
    expect(spans.map((span) => span.start)).toEqual([1, 3]);
  });
});

describe("contract constants", () => {
  it("declares the exact export filename", () => {
    expect(DECISIONS_FILENAME).toBe("phase2k-downstream-alignment-decisions-v1.json");
  });

  it("keeps text hashes as ordinary UTF-8 SHA-256", async () => {
    expect(await textSha256("hello", digest)).toBe(sha256Hex("hello"));
    expect(await textSha256("héllo", digest)).toBe(sha256Hex("héllo"));
  });

  it("keeps correction statuses stable", () => {
    expect(CORRECTION_STATUSES).toEqual(["UNCHANGED", "TERMINAL_PUNCTUATION_DROPPED"]);
  });
});
