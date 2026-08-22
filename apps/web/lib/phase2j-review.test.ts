import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import {
  addEndpointToWindow,
  ANNOTATION_VERSION,
  applyOutcome,
  buildExportSession,
  buildSessionFromPayload,
  createEndpoint,
  deriveEndpointSpan,
  ENDPOINT_TYPES,
  findForbiddenFields,
  PACKET_SCHEMA_VERSION,
  markPassAComplete,
  nextEndpointSequence,
  removeEndpointFromWindow,
  sanitizePacket,
  SESSION_FORBIDDEN_KEYS,
  SESSION_SCHEMA_VERSION,
  snapCharRangeToTokens,
  spansOverlap,
  summarizeProgress,
  validateSessionInput,
  type Phase2JReviewPayload,
  type ReviewRecord,
  type ReviewSession,
  type SessionRecord,
} from "./phase2j-review";

function sha256Hex(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

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

const SAMPLE_TEXT = "hit R yeah like this";

function makeRecord(overrides: Partial<ReviewRecord> = {}): ReviewRecord {
  const bronzeText = SAMPLE_TEXT;
  return {
    record_index: 1,
    window_id: "pool:group:w001-abc",
    source_group_id: "video:group",
    bronze_text: bronzeText,
    bronze_text_sha256: sha256Hex(bronzeText),
    bronze_char_length: bronzeText.length,
    tokens: tokenTable(bronzeText),
    ...overrides,
  };
}

function makePacket(records: ReviewRecord[] = [makeRecord()]) {
  const inner = {
    annotation_version: ANNOTATION_VERSION,
    purpose: "test packet",
    records,
    release_gate: "LOCKED",
    rules: {},
    schema_version: PACKET_SCHEMA_VERSION,
  };
  return { content_sha256: canonicalSha256(inner), ...inner };
}

function freshSession(payload: Phase2JReviewPayload): ReviewSession {
  return buildSessionFromPayload(payload);
}

describe("snapCharRangeToTokens", () => {
  const tokens = tokenTable(SAMPLE_TEXT);

  it("snaps a mid-token selection to the whole token", () => {
    expect(snapCharRangeToTokens(tokens, 0, 2)).toEqual({ token_start: 0, token_end: 0 });
    expect(snapCharRangeToTokens(tokens, 2, 6)).toEqual({ token_start: 0, token_end: 1 });
  });

  it("normalizes reverse selections", () => {
    expect(snapCharRangeToTokens(tokens, 5, 4)).toEqual({ token_start: 1, token_end: 1 });
  });

  it("rejects whitespace-only and empty selections", () => {
    expect(snapCharRangeToTokens(tokens, 3, 4)).toBeNull();
    expect(snapCharRangeToTokens(tokens, 4, 4)).toBeNull();
  });

  it("rejects out-of-bounds input", () => {
    expect(snapCharRangeToTokens(tokens, -1, 3)).toBeNull();
    expect(snapCharRangeToTokens(tokens, 0.5, 3)).toBeNull();
  });

  it("snaps a cross-sentence selection to first and last tokens", () => {
    expect(snapCharRangeToTokens(tokens, 0, 100)).toEqual({ token_start: 0, token_end: 4 });
  });
});

describe("deriveEndpointSpan", () => {
  const record = makeRecord();

  it("derives exact char offsets and Bronze slices", () => {
    expect(deriveEndpointSpan(record, 0, 0)).toEqual({
      char_start: 0,
      char_end: 3,
      exact_bronze_text: "hit",
    });
    expect(deriveEndpointSpan(record, 0, 1)).toEqual({
      char_start: 0,
      char_end: 5,
      exact_bronze_text: "hit R",
    });
  });

  it("rejects invalid token ranges", () => {
    expect(deriveEndpointSpan(record, -1, 0)).toBeNull();
    expect(deriveEndpointSpan(record, 1, 0)).toBeNull();
    expect(deriveEndpointSpan(record, 0, 99)).toBeNull();
  });
});

describe("spansOverlap", () => {
  it("treats adjacent spans as non-overlapping", () => {
    expect(
      spansOverlap({ char_start: 0, char_end: 3 }, { char_start: 3, char_end: 5 }),
    ).toBe(false);
  });

  it("detects partial and exact overlap", () => {
    expect(
      spansOverlap({ char_start: 0, char_end: 3 }, { char_start: 2, char_end: 5 }),
    ).toBe(true);
    expect(
      spansOverlap({ char_start: 0, char_end: 3 }, { char_start: 0, char_end: 3 }),
    ).toBe(true);
  });
});

describe("createEndpoint", () => {
  const record = makeRecord();

  it("creates a deterministic Pass A endpoint with exact Bronze fields", () => {
    const endpoint = createEndpoint(record, 0, 1, "ENTITY", 0);
    expect(endpoint).not.toBeNull();
    expect(endpoint?.endpoint_id).toBe(`p2j:review:${record.window_id}:ep:0000`);
    expect(endpoint?.exact_bronze_text).toBe("hit R");
    expect(endpoint?.char_start).toBe(0);
    expect(endpoint?.char_end).toBe(5);
    expect(endpoint?.token_start).toBe(0);
    expect(endpoint?.token_end).toBe(1);
    expect(endpoint?.node_type).toBe("ENTITY");
    expect(endpoint?.ambiguity_state).toBe("NONE");
    expect(endpoint?.disposition).toBe("KEEP");
    expect(endpoint?.pass_provenance).toBe("PASS_A");
    expect(endpoint?.human_accepted).toBe(true);
  });

  it("rejects out-of-bounds token ranges", () => {
    expect(createEndpoint(record, -1, 0, "EVENT", 1)).toBeNull();
    expect(createEndpoint(record, 0, 99, "EVENT", 1)).toBeNull();
  });

  it("does not reuse a sequence after an earlier endpoint is deleted", () => {
    const first = createEndpoint(record, 0, 0, "ACTION", 0)!;
    const second = createEndpoint(record, 1, 1, "ABILITY_OR_RESOURCE", 1)!;
    expect(nextEndpointSequence([second])).toBe(2);
    expect(createEndpoint(record, 2, 2, "EVENT", nextEndpointSequence([second]))?.endpoint_id)
      .toContain(":ep:0002");
    expect(first.endpoint_id).not.toBe(second.endpoint_id);
  });
});

describe("sanitizePacket", () => {
  it("returns only the allowed client fields", () => {
    const packet = makePacket();
    const payload = sanitizePacket(packet);
    expect(payload.schema_version).toBe(PACKET_SCHEMA_VERSION);
    expect(payload.annotation_version).toBe(ANNOTATION_VERSION);
    expect(payload.packet_sha256).toBe(packet.content_sha256);
    expect(payload.records).toHaveLength(1);
    expect(Object.keys(payload.records[0]).sort()).toEqual([
      "bronze_char_length",
      "bronze_text",
      "bronze_text_sha256",
      "record_index",
      "source_group_id",
      "tokens",
      "window_id",
    ]);
    expect(payload.records[0].tokens).toEqual(tokenTable(SAMPLE_TEXT));
    expect(findForbiddenFields(payload, SESSION_FORBIDDEN_KEYS)).toEqual([]);
  });

  it("strips privacy fields such as partition while allowing them in the packet", () => {
    const record = { ...makeRecord(), partition: "EXPANDED_DEV", champion: "secret" };
    const payload = sanitizePacket(makePacket([record]));
    expect(payload.records[0]).not.toHaveProperty("partition");
    expect(payload.records[0]).not.toHaveProperty("champion");
  });

  it("rejects scorer/model fields recursively", () => {
    const packet = makePacket();
    (packet.records[0] as Record<string, unknown>).score = 0.5;
    expect(() => sanitizePacket(packet)).toThrow(/forbidden/);
  });

  it("rejects wrong versions, ordering, and token corruption", () => {
    const wrongVersion = makePacket();
    wrongVersion.schema_version = "phase2j-endpoint-annotation-packet-v9";
    expect(() => sanitizePacket(wrongVersion)).toThrow(/schema_version/);

    const outOfOrder = makePacket([
      { ...makeRecord({ record_index: 2, window_id: "pool:group:w002" }) },
      { ...makeRecord() },
    ]);
    expect(() => sanitizePacket(outOfOrder)).toThrow(/ordered/);

    const corrupt = makePacket();
    corrupt.records[0].tokens = [
      { token_index: 0, text: "hix", start: 0, end: 3 },
    ];
    expect(() => sanitizePacket(corrupt)).toThrow(/exact source slice/);
  });
});

describe("session transitions", () => {
  const payload = sanitizePacket(makePacket());
  const session = freshSession(payload);
  const base = session.records[0];

  it("builds a fresh unreviewed session", () => {
    expect(session.schema_version).toBe(SESSION_SCHEMA_VERSION);
    expect(session.exported_at).toBeNull();
    expect(base.window_status).toBe("UNREVIEWED");
    expect(base.outcome).toBe("CLEAN");
    expect(base.endpoints).toEqual([]);
  });

  it("moves to IN_REVIEW on add and back to UNREVIEWED on removal", () => {
    const endpoint = createEndpoint(payload.records[0], 0, 1, "ENTITY", 0);
    expect(endpoint).not.toBeNull();
    const added = addEndpointToWindow(base, endpoint!);
    expect(added.window_status).toBe("IN_REVIEW");
    expect(added.endpoints).toHaveLength(1);
    const removed = removeEndpointFromWindow(added, endpoint!.endpoint_id);
    expect(removed.window_status).toBe("UNREVIEWED");
    expect(removed.endpoints).toEqual([]);
  });

  it("applies outcome transitions", () => {
    const ambiguous = applyOutcome(base, "AMBIGUOUS");
    expect(ambiguous.window_status).toBe("AMBIGUOUS");
    expect(ambiguous.outcome).toBe("AMBIGUOUS");

    const withEndpoint = addEndpointToWindow(base, createEndpoint(payload.records[0], 0, 0, "TIME", 0)!);
    const excluded = applyOutcome(withEndpoint, "EXCLUDED");
    expect(excluded.window_status).toBe("EXCLUDED");
    expect(excluded.endpoints).toEqual([]);

    expect(applyOutcome(base, "CLEAN").window_status).toBe("UNREVIEWED");
  });

  it("allows a human to sign ambiguous and excluded Pass A outcomes", () => {
    const ambiguous = markPassAComplete(
      { ...applyOutcome(base, "AMBIGUOUS"), note: "unresolved referent" },
      "Reviewer",
      "2026-08-18",
      true,
    );
    expect(ambiguous.window_status).toBe("AMBIGUOUS");
    expect(ambiguous.pass_a_complete).toBe(true);

    const excluded = markPassAComplete(
      { ...applyOutcome(base, "EXCLUDED"), note: "ASR/context loss" },
      "Reviewer",
      "2026-08-18",
      true,
    );
    expect(excluded.window_status).toBe("EXCLUDED");
    expect(excluded.pass_a_complete).toBe(true);
  });
});

describe("validateSessionInput", () => {
  const payload = sanitizePacket(makePacket());
  const reference = payload.records;

  function withEndpoints(
    session: ReviewSession,
    endpoints: NonNullable<ReturnType<typeof createEndpoint>>[],
  ): ReviewSession {
    return {
      ...session,
      records: session.records.map((record, index) =>
        index === 0
          ? {
              ...record,
              endpoints,
              window_status: endpoints.length > 0 ? "IN_REVIEW" : "UNREVIEWED",
            }
          : record,
      ),
    };
  }

  it("accepts a valid exported round trip", () => {
    const session = freshSession(payload);
    const exported = buildExportSession(
      withEndpoints(session, [createEndpoint(reference[0], 0, 1, "ENTITY", 0)!]),
      "2026-08-18T12:00:00.000Z",
    );
    const result = validateSessionInput(
      JSON.parse(JSON.stringify(exported)),
      payload.packet_sha256,
      reference,
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.session.records[0].endpoints).toHaveLength(1);
      expect(result.session.exported_at).toBe("2026-08-18T12:00:00.000Z");
    }
  });

  it("is deterministic apart from the explicit export timestamp", () => {
    const session = freshSession(payload);
    const first = JSON.stringify(buildExportSession(session, "2026-08-18T00:00:00.000Z"));
    const second = JSON.stringify(buildExportSession(session, "2026-08-18T00:00:00.000Z"));
    expect(first).toBe(second);
  });

  it("rejects a packet hash mismatch", () => {
    const result = validateSessionInput(freshSession(payload), "0".repeat(64), reference);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/packet_sha256/);
    }
  });

  it("rejects wrong schema versions", () => {
    const session = freshSession(payload) as unknown as Record<string, unknown>;
    session.schema_version = "phase2j-review-session-v9";
    const result = validateSessionInput(session, payload.packet_sha256, reference);
    expect(result.ok).toBe(false);
  });

  it("rejects identity and ordering mismatches", () => {
    const session = freshSession(payload);
    const moved = {
      ...session,
      records: [
        { ...session.records[0], window_id: "pool:group:different" },
        ...session.records.slice(1),
      ],
    };
    const result = validateSessionInput(moved, payload.packet_sha256, reference);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/window_id/);
    }

    const reordered = {
      ...session,
      records: [
        { ...session.records[0], record_index: 2 },
        ...session.records.slice(1),
      ],
    };
    const reorderedResult = validateSessionInput(reordered, payload.packet_sha256, reference);
    expect(reorderedResult.ok).toBe(false);
  });

  it("rejects overlapping or duplicate endpoints", () => {
    const session = freshSession(payload);
    const overlapping = withEndpoints(session, [
      createEndpoint(reference[0], 0, 1, "ENTITY", 0)!,
      createEndpoint(reference[0], 0, 0, "EVENT", 1)!,
    ]);
    const result = validateSessionInput(overlapping, payload.packet_sha256, reference);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/overlap/);
    }
  });

  it("rejects invalid enums and corrupted Bronze slices", () => {
    const session = freshSession(payload);
    const badType = withEndpoints(session, [
      {
        ...createEndpoint(reference[0], 0, 1, "ENTITY", 0)!,
        node_type: "BOGUS" as never,
      },
    ]);
    expect(validateSessionInput(badType, payload.packet_sha256, reference).ok).toBe(false);

    const badSlice = withEndpoints(session, [
      {
        ...createEndpoint(reference[0], 0, 1, "ENTITY", 0)!,
        exact_bronze_text: "hix R",
      },
    ]);
    expect(validateSessionInput(badSlice, payload.packet_sha256, reference).ok).toBe(false);
  });

  it("rejects EXCLUDED windows with endpoints or without a note", () => {
    const session = freshSession(payload);
    const excludedWithEndpoints: ReviewSession = {
      ...session,
      records: [
        {
          ...session.records[0],
          outcome: "EXCLUDED",
          window_status: "EXCLUDED",
          note: "unusable ASR",
          endpoints: [createEndpoint(reference[0], 0, 1, "ENTITY", 0)!],
        },
      ],
    };
    expect(validateSessionInput(excludedWithEndpoints, payload.packet_sha256, reference).ok).toBe(
      false,
    );

    const excludedNoNote: ReviewSession = {
      ...session,
      records: [
        {
          ...session.records[0],
          outcome: "EXCLUDED",
          window_status: "EXCLUDED",
          note: "",
          endpoints: [],
        },
      ],
    };
    expect(validateSessionInput(excludedNoNote, payload.packet_sha256, reference).ok).toBe(false);
  });

  it("rejects AMBIGUOUS windows without a note and UNREVIEWED windows with fields", () => {
    const session = freshSession(payload);
    const ambiguousNoNote: ReviewSession = {
      ...session,
      records: [
        {
          ...session.records[0],
          outcome: "AMBIGUOUS",
          window_status: "AMBIGUOUS",
          note: "",
          endpoints: [],
        },
      ],
    };
    expect(validateSessionInput(ambiguousNoNote, payload.packet_sha256, reference).ok).toBe(false);

    const unreviewedWithNote: ReviewSession = {
      ...session,
      records: [
        {
          ...session.records[0],
          window_status: "UNREVIEWED",
          note: "nope",
          endpoints: [],
        },
      ],
    };
    expect(validateSessionInput(unreviewedWithNote, payload.packet_sha256, reference).ok).toBe(
      false,
    );
  });

  it("rejects Pass A completion without reviewer name or date", () => {
    const session = freshSession(payload);
    const incomplete: ReviewSession = {
      ...session,
      records: [
        {
          ...session.records[0],
          window_status: "IN_REVIEW",
          reviewer_name: "",
          completed_at: null,
          pass_a_complete: true,
          endpoints: [],
        },
      ],
    };
    const result = validateSessionInput(incomplete, payload.packet_sha256, reference);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/reviewer name and completed_at/);
    }
  });

  it("rejects forbidden fields anywhere in the session", () => {
    const session = freshSession(payload) as unknown as Record<string, unknown>;
    (session.records as Record<string, unknown>[])[0].partition = "EXPANDED_DEV";
    const result = validateSessionInput(session, payload.packet_sha256, reference);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toMatch(/partition/);
    }
  });

  it("rejects extra or missing records", () => {
    const session = freshSession(payload);
    const extra = {
      ...session,
      records: [
        ...session.records,
        { ...session.records[0], record_index: 2, window_id: "pool:group:extra" },
      ],
    };
    const result = validateSessionInput(extra, payload.packet_sha256, reference);
    expect(result.ok).toBe(false);
  });
});

describe("summarizeProgress", () => {
  it("counts statuses and endpoints", () => {
    const payload = sanitizePacket(
      makePacket([
        makeRecord(),
        makeRecord({ record_index: 2, window_id: "pool:group:w002" }),
      ]),
    );
    const session = freshSession(payload);
    const summary = summarizeProgress(session);
    expect(summary.total).toBe(2);
    expect(summary.unreviewed).toBe(2);
    expect(summary.endpoints).toBe(0);

    const withEndpoint: SessionRecord = {
      ...session.records[0],
      endpoints: [createEndpoint(payload.records[0], 0, 0, "STATE", 0)!],
      window_status: "IN_REVIEW",
      reviewer_name: "Reviewer",
      completed_at: "2026-08-18",
      pass_a_complete: true,
    };
    const updated = { ...session, records: [withEndpoint, session.records[1]] };
    const next = summarizeProgress(updated);
    expect(next.in_review).toBe(1);
    expect(next.pass_a_complete).toBe(1);
    expect(next.endpoints).toBe(1);
    expect(next.unreviewed).toBe(1);
  });
});

describe("locked Phase 2J packet", () => {
  const packetPath = path.resolve(
    process.cwd(),
    "../../data/phase2j/endpoint-annotation-packet-v1.json",
  );

  it("sanitizes all 30 locked windows", () => {
    const raw = JSON.parse(readFileSync(packetPath, "utf8")) as unknown;
    const payload = sanitizePacket(raw);
    expect(payload.records).toHaveLength(30);
    expect(payload.records.map((record) => record.record_index)).toEqual(
      Array.from({ length: 30 }, (_, index) => index + 1),
    );
    expect(new Set(payload.records.map((record) => record.window_id)).size).toBe(30);
    expect(findForbiddenFields(payload, SESSION_FORBIDDEN_KEYS)).toEqual([]);
    for (const record of payload.records) {
      expect(record.bronze_char_length).toBe(record.bronze_text.length);
      for (const token of record.tokens) {
        expect(record.bronze_text.slice(token.start, token.end)).toBe(token.text);
      }
    }
  });

  it("round-trips an exported session against the real packet", () => {
    const raw = JSON.parse(readFileSync(packetPath, "utf8")) as unknown;
    const payload = sanitizePacket(raw);
    const session = freshSession(payload);
    const first = createEndpoint(payload.records[0], 0, 1, "ENTITY", 0);
    expect(first).not.toBeNull();
    const updated = {
      ...session,
      records: session.records.map((record, index) =>
        index === 0
          ? {
              ...record,
              endpoints: [first!],
              window_status: "IN_REVIEW" as const,
            }
          : record,
      ),
    };
    const result = validateSessionInput(updated, payload.packet_sha256, payload.records);
    expect(result.ok).toBe(true);
    expect(ENDPOINT_TYPES).toContain("UNDETERMINED");
  });
});
