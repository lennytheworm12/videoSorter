import { createHash } from "node:crypto";
import {
  buildReviewsMap,
  buildSessionExport,
  buildSessionFromPacket,
  completeItem,
  NOT_APPLICABLE,
  PACKET_SCHEMA_VERSION,
  PRESENTATION_SCHEMA_VERSION,
  SCORE_FIELDS,
  sanitizePacket,
  setAllReviewers,
  setItemNotes,
  setItemReviewer,
  setItemScore,
  summarizeProgress,
  uncompleteItem,
  validateSessionInput,
  type Presentation,
  type ReviewPacket,
  type Rubric,
  type SessionItem,
  type SessionScore,
} from "./phase2k-review";

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

const HEX = "0123456789abcdef";
function randomHex(bytes: number): string {
  let out = "";
  for (let index = 0; index < bytes; index += 1) {
    out += HEX[Math.floor(Math.random() * HEX.length)];
  }
  return out;
}

function hex64(seed: string): string {
  return sha256Hex(`fixture:${seed}`);
}

function makeRubric(): Rubric {
  const rubric = {} as Rubric;
  for (const field of SCORE_FIELDS) {
    const lowerIsBetter =
      field === "unsupported_invention" || field === "remaining_ambiguity";
    const naDisallowed =
      field === "standalone_coaching_claim" || field === "unsupported_invention";
    rubric[field] = {
      description: `Rubric description for ${field}.`,
      direction: lowerIsBetter ? "lower_is_better" : "higher_is_better",
      not_applicable_allowed: !naDisallowed,
    };
  }
  return rubric;
}

type PresentationOverride = Partial<Omit<Presentation, "sections" | "schema_version">> & {
  sections?: Array<{ id: string; text: string }>;
  schema_version?: string;
};

function makePresentation(overrides: PresentationOverride = {}): Presentation {
  return {
    schema_version: PRESENTATION_SCHEMA_VERSION,
    target_sha256: hex64("target"),
    displayed_target_sha256: hex64("displayed-target"),
    sections: [{ id: "primary", text: "⟪TARGET⟫hit R here⟪/TARGET⟫" }],
    ...overrides,
  } as Presentation;
}

type FixtureItem = {
  review_item_id: string;
  window_id: string;
  blinded_label: string;
  presentation: Presentation;
  content_sha256: string;
  scores: Record<string, unknown>;
  reviewer: string | null;
  completed_at: string | null;
  notes: string[];
};

function makeItem(index: number, overrides: Partial<FixtureItem> = {}): FixtureItem {
  const label = `BLIND-${index.toString(16).padStart(8, "0")}`;
  const windowId = `pool:group:w${String(index + 1).padStart(3, "0")}-abc`;
  return {
    review_item_id: `p2k:hr:${windowId}:${label}`,
    window_id: windowId,
    blinded_label: label,
    presentation: makePresentation(),
    content_sha256: hex64(`item-${index}`),
    scores: Object.fromEntries(SCORE_FIELDS.map((field) => [field, null])),
    reviewer: null,
    completed_at: null,
    notes: [],
    ...overrides,
  };
}

function makePacket(
  items: FixtureItem[] = [makeItem(0), makeItem(1)],
  overrides: Partial<Record<string, unknown>> = {},
): Record<string, unknown> {
  const inner = {
    schema_version: PACKET_SCHEMA_VERSION,
    purpose: "Blinded Phase 2K human review packet.",
    release_gate: "AWAITING_HUMAN_REVIEW",
    blinding: {
      method: "seeded_random_condition_labels",
      seed: "phase2k-hr-blinding-20260819",
      mapping_file: "phase2k-human-review-mapping-v2.json",
      mapping_sha256: hex64("mapping"),
    },
    review_items: items,
    scoring_fields: [...SCORE_FIELDS],
    score_range: { min: 0, max: 5 },
    rubric: makeRubric(),
    ...overrides,
  };
  const contentSha256 = overrides.content_sha256 ?? canonicalSha256(inner);
  return { content_sha256: contentSha256, ...inner };
}

function sanitizedPacket(items: FixtureItem[] = [makeItem(0), makeItem(1)]): ReviewPacket {
  return sanitizePacket(makePacket(items));
}

function blankScores(): SessionScore {
  return Object.fromEntries(SCORE_FIELDS.map((field) => [field, null])) as SessionScore;
}

function fillScores(item: SessionItem): SessionItem {
  const scores = {} as SessionScore;
  for (const field of SCORE_FIELDS) {
    scores[field] = field === "unsupported_invention" ? 1 : 4;
  }
  return { ...item, scores };
}

const ISO = "2026-08-19T12:00:00.000Z";

describe("sanitizePacket", () => {
  it("accepts a strict official blank packet and emits only reviewer-facing fields", () => {
    const packet = sanitizedPacket();
    expect(packet.schema_version).toBe(PACKET_SCHEMA_VERSION);
    expect(packet.release_gate).toBe("AWAITING_HUMAN_REVIEW");
    expect(packet.score_range).toEqual({ min: 0, max: 5 });
    expect(packet.scoring_fields).toEqual(SCORE_FIELDS);
    expect(packet.content_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(packet.review_items).toHaveLength(2);
    // The blinding envelope (which names the mapping artifact) must not reach the client.
    expect(Object.keys(packet).sort()).toEqual([
      "content_sha256",
      "purpose",
      "release_gate",
      "review_items",
      "rubric",
      "schema_version",
      "score_range",
      "scoring_fields",
    ]);
    for (const item of packet.review_items) {
      expect(item.reviewer).toBeNull();
      expect(item.completed_at).toBeNull();
      expect(item.notes).toEqual([]);
      expect(Object.keys(item.scores).sort()).toEqual([...SCORE_FIELDS].sort());
      expect(Object.values(item.scores).every((value) => value === null)).toBe(true);
      expect(item.presentation.sections.length).toBeGreaterThan(0);
    }
  });

  it("rejects wrong schema, version, gate, key sets, and malformed hashes", () => {
    expect(() => sanitizePacket(makePacket([], { schema_version: "wrong" }))).toThrow(
      /schema_version/,
    );
    expect(() => sanitizePacket(makePacket([], { release_gate: "REVIEWED" }))).toThrow(
      /release_gate/,
    );
    expect(() => sanitizePacket(makePacket([], { score_range: { min: 1, max: 5 } }))).toThrow(
      /score_range/,
    );
    expect(() =>
      sanitizePacket(makePacket([], { scoring_fields: [...SCORE_FIELDS].reverse() })),
    ).toThrow(/scoring_fields/);
    expect(() => sanitizePacket(makePacket([], { content_sha256: "not-a-hash" }))).toThrow(
      /content_sha256/,
    );
    expect(() => sanitizePacket(makePacket([], { rubric: {} }))).toThrow(/rubric/);
    expect(() => sanitizePacket(makePacket([], { blinding: {} }))).toThrow(/blinding/);
  });

  it("rejects condition/radius/mapping/model/downstream structural leakage", () => {
    const withConditionCode = makePacket();
    (withConditionCode.review_items as FixtureItem[])[0] = {
      ...(withConditionCode.review_items as FixtureItem[])[0],
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      condition_code: "A",
    } as FixtureItem;
    expect(() => sanitizePacket(withConditionCode)).toThrow(/forbidden/);

    const withRadius = makePacket();
    (withRadius.review_items as FixtureItem[])[0] = {
      ...(withRadius.review_items as FixtureItem[])[0],
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      radius_label: "r5",
    } as FixtureItem;
    expect(() => sanitizePacket(withRadius)).toThrow(/radius_label/);

    const withMapping = makePacket();
    (withMapping.review_items as FixtureItem[])[0] = {
      ...(withMapping.review_items as FixtureItem[])[0],
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      mapping: { label: "BLIND-x", condition_code: "B" },
    } as FixtureItem;
    expect(() => sanitizePacket(withMapping)).toThrow(/mapping/);

    const withModel = makePacket();
    (withModel.review_items as FixtureItem[])[0] = {
      ...(withModel.review_items as FixtureItem[])[0],
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      model_name: "gpt-x",
      prediction: "raw_bronze",
    } as FixtureItem;
    expect(() => sanitizePacket(withModel)).toThrow(/model_name/);

    // Top-level downstream-result leakage is caught too.
    const withResults = makePacket([], { results: { scorer: "sol" } });
    expect(() => sanitizePacket(withResults)).toThrow(/results/);

    // Forbidden structural values (exact condition/radius tokens) are caught.
    const withValue = makePacket();
    (withValue.review_items as FixtureItem[])[0] = {
      ...(withValue.review_items as FixtureItem[])[0],
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      extra_value: "bounded_local_episode",
    } as FixtureItem;
    expect(() => sanitizePacket(withValue)).toThrow(/bounded_local_episode/);
  });

  it("does not scan free-text presentation prose for words", () => {
    const packet = makePacket([
      makeItem(0, {
        presentation: makePresentation({
          sections: [
            {
              id: "primary",
              text: "the reconstruction includes the radius r5 but this is reviewer-facing prose",
            },
          ],
        }),
      }),
    ]);
    expect(() => sanitizePacket(packet)).not.toThrow();
  });

  it("rejects nonblank official packets (scores, reviewer, timestamps, notes)", () => {
    const scored = makeItem(0, {
      scores: {
        ...Object.fromEntries(SCORE_FIELDS.map((field) => [field, null])),
        core_action: 3,
      },
    });
    expect(() => sanitizePacket(makePacket([scored]))).toThrow(/remain blank/);

    const signed = makeItem(0, { reviewer: "Tester" });
    expect(() => sanitizePacket(makePacket([signed]))).toThrow(/cannot be signed/);

    const timestamped = makeItem(0, { completed_at: ISO });
    expect(() => sanitizePacket(makePacket([timestamped]))).toThrow(/timestamp/);

    const noted = makeItem(0, { notes: ["prefilled"] });
    expect(() => sanitizePacket(makePacket([noted]))).toThrow(/notes/);
  });

  it("rejects duplicate ids/labels and invalid presentations", () => {
    const first = makeItem(0);
    const duplicateId = makeItem(1, { review_item_id: first.review_item_id });
    expect(() => sanitizePacket(makePacket([first, duplicateId]))).toThrow(/unique/);

    const duplicateLabel = makeItem(1, { blinded_label: first.blinded_label });
    expect(() => sanitizePacket(makePacket([first, duplicateLabel]))).toThrow(/unique/);

    const unblindedId = makeItem(0, { review_item_id: "p2k:rec:some-record" });
    expect(() => sanitizePacket(makePacket([unblindedId]))).toThrow(/unblinded/);

    const emptySections = makeItem(0, {
      presentation: makePresentation({ sections: [] }),
    });
    expect(() => sanitizePacket(makePacket([emptySections]))).toThrow(/non-empty sections/);

    const noPrimary = makeItem(0, {
      presentation: makePresentation({
        sections: [{ id: "supplement", text: "only supplement" }],
      }),
    });
    expect(() => sanitizePacket(makePacket([noPrimary]))).toThrow(/primary/);

    const badSectionId = makeItem(0, {
      presentation: makePresentation({
        sections: [{ id: "target", text: "bad id" }],
      }),
    });
    expect(() => sanitizePacket(makePacket([badSectionId]))).toThrow(/neutral/);

    const badHash = makeItem(0, {
      presentation: makePresentation({ target_sha256: "abc" }),
    });
    expect(() => sanitizePacket(makePacket([badHash]))).toThrow(/target_sha256/);

    const wrongVersion = makeItem(0, {
      presentation: makePresentation({ schema_version: "phase2k-review-presentation-v1" }),
    });
    expect(() => sanitizePacket(makePacket([wrongVersion]))).toThrow(/schema_version/);
  });
});

describe("session score transitions", () => {
  it("builds a blank session bound to the packet hash with no fabricated fields", () => {
    const packet = sanitizedPacket();
    const session = buildSessionFromPacket(packet);
    expect(session.packet_sha256).toBe(packet.content_sha256);
    expect(session.exported_at).toBeNull();
    expect(session.items).toHaveLength(2);
    for (const item of session.items) {
      expect(item.scores).toEqual(blankScores());
      expect(item.reviewer).toBe("");
      expect(item.completed_at).toBeNull();
      expect(item.notes).toEqual([]);
      expect(item.complete).toBe(false);
    }
  });

  it("allows N/A only where the rubric allows it", () => {
    const packet = sanitizedPacket();
    const session = buildSessionFromPacket(packet);
    const allowed = setItemScore(
      session,
      packet.rubric,
      session.items[0].review_item_id,
      "coached_actor",
      NOT_APPLICABLE,
    );
    expect(allowed.items[0].scores.coached_actor).toBe(NOT_APPLICABLE);
    expect(() =>
      setItemScore(
        session,
        packet.rubric,
        session.items[0].review_item_id,
        "standalone_coaching_claim",
        NOT_APPLICABLE,
      ),
    ).toThrow(/NOT_APPLICABLE/);
    expect(() =>
      setItemScore(
        session,
        packet.rubric,
        session.items[0].review_item_id,
        "unsupported_invention",
        NOT_APPLICABLE,
      ),
    ).toThrow(/NOT_APPLICABLE/);
  });

  it("accepts integers 0-5 and rejects out-of-range or non-integer values", () => {
    const packet = sanitizedPacket();
    let session = buildSessionFromPacket(packet);
    for (const value of [0, 5, 3]) {
      session = setItemScore(
        session,
        packet.rubric,
        session.items[0].review_item_id,
        "core_action",
        value,
      );
    }
    expect(session.items[0].scores.core_action).toBe(3);
    expect(() =>
      setItemScore(
        session,
        packet.rubric,
        session.items[0].review_item_id,
        "core_action",
        6,
      ),
    ).toThrow(/invalid score/);
    expect(() =>
      setItemScore(
        session,
        packet.rubric,
        session.items[0].review_item_id,
        "core_action",
        1.5,
      ),
    ).toThrow(/invalid score/);
  });

  it("does not fabricate a timestamp or reviewer before explicit completion", () => {
    const packet = sanitizedPacket();
    let session = buildSessionFromPacket(packet);
    session = setItemScore(
      session,
      packet.rubric,
      session.items[0].review_item_id,
      "core_action",
      4,
    );
    session = setItemNotes(session, session.items[0].review_item_id, ["a note"]);
    expect(session.items[0].completed_at).toBeNull();
    expect(session.items[0].complete).toBe(false);
    // Exporting the backup still leaves every field review-only.
    const backup = buildSessionExport(session, ISO);
    expect(backup.items[0].completed_at).toBeNull();
    expect(backup.items[0].reviewer).toBe("");
    expect(backup.exported_at).toBe(ISO);
  });

  it("completion requires all 14 scores and a reviewer, then stamps the explicit timestamp", () => {
    const packet = sanitizedPacket();
    const itemId = packet.review_items[0].review_item_id;
    let session = buildSessionFromPacket(packet);

    session = setItemReviewer(session, itemId, "Tester");
    let result = completeItem(session, itemId, ISO);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toContain("core_action");
      expect(result.errors).toHaveLength(14);
    }

    session = { ...session, items: session.items.map((item) => fillScores(item)) };
    // Reviewer was cleared along with the scores map: completion refuses.
    session = setItemReviewer(session, itemId, "");
    result = completeItem(session, itemId, ISO);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors).toEqual(["missing reviewer"]);
    }

    session = setItemReviewer(session, itemId, "Tester");
    result = completeItem(session, itemId, ISO);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.session.items[0].complete).toBe(true);
      expect(result.session.items[0].completed_at).toBe(ISO);
    }

    // A score edit after explicit completion invalidates the timestamp.
    const edited = setItemScore(
      session,
      packet.rubric,
      itemId,
      "core_action",
      5,
    );
    expect(edited.items[0].complete).toBe(false);
    expect(edited.items[0].completed_at).toBeNull();

    // Uncompleting clears the generated timestamp.
    const undone = uncompleteItem(session, itemId);
    expect(undone.items[0].complete).toBe(false);
    expect(undone.items[0].completed_at).toBeNull();
  });
});

describe("session import validation", () => {
  it("validates packet-hash binding and immutable packet item content", () => {
    const packet = sanitizedPacket();
    const session = buildSessionFromPacket(packet);

    const wrongBinding = { ...session, packet_sha256: randomHex(32) };
    const bindingResult = validateSessionInput(wrongBinding, packet);
    expect(bindingResult.ok).toBe(false);
    if (!bindingResult.ok) {
      expect(bindingResult.errors.join(" ")).toContain("does not match");
    }

    const tampered = {
      ...session,
      items: session.items.map((item, index) =>
        index === 0
          ? {
              ...item,
              presentation: {
                ...item.presentation,
                sections: [{ id: "primary", text: "tampered text" }],
              },
            }
          : item,
      ),
    };
    const tamperResult = validateSessionInput(tampered, packet);
    expect(tamperResult.ok).toBe(false);
    if (!tamperResult.ok) {
      expect(tamperResult.errors.join(" ")).toContain("presentation does not match");
    }

    const relabeled = {
      ...session,
      items: session.items.map((item, index) =>
        index === 0 ? { ...item, blinded_label: "BLIND-ffffffff" } : item,
      ),
    };
    const relabelResult = validateSessionInput(relabeled, packet);
    expect(relabelResult.ok).toBe(false);
    if (!relabelResult.ok) {
      expect(relabelResult.errors.join(" ")).toContain("blinded_label does not match");
    }

    const reordered = { ...session, items: [...session.items].reverse() };
    const orderResult = validateSessionInput(reordered, packet);
    expect(orderResult.ok).toBe(false);
    if (!orderResult.ok) {
      expect(orderResult.errors.join(" ")).toContain("does not match");
    }

    const missingItem = { ...session, items: session.items.slice(0, 1) };
    const countResult = validateSessionInput(missingItem, packet);
    expect(countResult.ok).toBe(false);
    if (!countResult.ok) {
      expect(countResult.errors.join(" ")).toContain("exactly 2 entries");
    }

    const validResult = validateSessionInput(session, packet);
    expect(validResult.ok).toBe(true);
  });

  it("validates allowed score values, reviewer, timestamps, and completion invariants", () => {
    const packet = sanitizedPacket();
    const session = buildSessionFromPacket(packet);

    const badNa = {
      ...session,
      items: [
        {
          ...session.items[0],
          scores: { ...session.items[0].scores, standalone_coaching_claim: NOT_APPLICABLE },
        },
        ...session.items.slice(1),
      ],
    };
    const naResult = validateSessionInput(badNa, packet);
    expect(naResult.ok).toBe(false);
    if (!naResult.ok) {
      expect(naResult.errors.join(" ")).toContain("standalone_coaching_claim is invalid");
    }

    const completedWithoutTimestamp = {
      ...session,
      items: session.items.map((item) => ({ ...fillScores(item), complete: true })),
    };
    const stampResult = validateSessionInput(completedWithoutTimestamp, packet);
    expect(stampResult.ok).toBe(false);
    if (!stampResult.ok) {
      expect(stampResult.errors.join(" ")).toContain("completed_at");
    }

    const timestampWithoutCompletion = {
      ...session,
      items: [{ ...session.items[0], completed_at: ISO }, ...session.items.slice(1)],
    };
    const prematureResult = validateSessionInput(timestampWithoutCompletion, packet);
    expect(prematureResult.ok).toBe(false);
    if (!prematureResult.ok) {
      expect(prematureResult.errors.join(" ")).toContain("until the item is explicitly marked");
    }

    const complete = session.items.map((item) => ({
      ...fillScores(item),
      reviewer: "Tester",
      complete: true,
    }));
    const withStamps = { ...session, items: complete };
    const valid = validateSessionInput(
      { ...withStamps, items: complete.map((item) => ({ ...item, completed_at: ISO })) },
      packet,
    );
    expect(valid.ok).toBe(true);
  });

  it("rejects session leakage of forbidden structural material", () => {
    const packet = sanitizedPacket();
    const session = buildSessionFromPacket(packet);
    const leaked = {
      ...session,
      items: [
        {
          ...session.items[0],
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          model_prediction: "reconstruction",
        } as unknown,
        ...session.items.slice(1),
      ],
    };
    const result = validateSessionInput(leaked, packet);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.join(" ")).toContain("forbidden field");
    }
  });
});

describe("final reviews map", () => {
  function completedSession(): ReturnType<typeof buildSessionFromPacket> {
    const packet = sanitizedPacket();
    let session = buildSessionFromPacket(packet);
    for (const item of session.items) {
      session = setItemReviewer(session, item.review_item_id, "Tester");
      session = setItemNotes(session, item.review_item_id, ["note one", "note two"]);
      for (const field of SCORE_FIELDS) {
        const value = field === "unsupported_invention" ? 1 : 4;
        session = setItemScore(session, packet.rubric, item.review_item_id, field, value);
      }
      const result = completeItem(session, item.review_item_id, ISO);
      if (!result.ok) {
        throw new Error(`test fixture failed: ${result.errors.join("; ")}`);
      }
      session = result.session;
    }
    return session;
  }

  it("refuses while any item is incomplete", () => {
    const packet = sanitizedPacket();
    const session = buildSessionFromPacket(packet);
    expect(() => buildReviewsMap(session, packet.rubric)).toThrow(/every item complete/);
  });

  it("produces the exact finalizer shape with deterministic item order", () => {
    const packet = sanitizedPacket();
    const session = completedSession();
    const map = buildReviewsMap(session, packet.rubric);
    expect(Object.keys(map)).toEqual(packet.review_items.map((item) => item.review_item_id));
    for (const [itemId, entry] of Object.entries(map)) {
      expect(Object.keys(entry).sort()).toEqual([
        "completed_at",
        "notes",
        "reviewer",
        "scores",
      ]);
      expect(entry.reviewer).toBe("Tester");
      expect(entry.completed_at).toBe(ISO);
      expect(entry.notes).toEqual(["note one", "note two"]);
      expect(Object.keys(entry.scores).sort()).toEqual([...SCORE_FIELDS].sort());
      expect(
        Object.values(entry.scores).every(
          (value) => typeof value === "number" && value >= 0 && value <= 5,
        ),
      ).toBe(true);
      expect(map[itemId].scores.unsupported_invention).toBe(1);
    }
  });

  it("exports a usable progress backup before completion", () => {
    const packet = sanitizedPacket();
    let session = buildSessionFromPacket(packet);
    session = setItemScore(
      session,
      packet.rubric,
      session.items[0].review_item_id,
      "core_action",
      4,
    );
    const backup = buildSessionExport(session, ISO);
    const result = validateSessionInput(backup, packet);
    expect(result.ok).toBe(true);
  });
});

describe("progress", () => {
  it("counts complete, ready, in-progress, and untouched items", () => {
    const packet = sanitizedPacket([makeItem(0), makeItem(1), makeItem(2)]);
    let session = buildSessionFromPacket(packet);
    expect(summarizeProgress(session)).toEqual({
      total: 3,
      complete: 0,
      ready: 0,
      in_progress: 0,
      untouched: 3,
    });

    const itemId = session.items[0].review_item_id;
    session = setItemScore(
      session,
      packet.rubric,
      itemId,
      "coached_actor",
      3,
    );
    expect(summarizeProgress(session)).toMatchObject({ in_progress: 1, untouched: 2 });

    // Fill everything but do not explicitly complete -> ready.
    let filled = session;
    for (const item of filled.items) {
      filled = setItemReviewer(filled, item.review_item_id, "Tester");
      for (const field of SCORE_FIELDS) {
        filled = setItemScore(filled, packet.rubric, item.review_item_id, field, 3);
      }
    }
    expect(summarizeProgress(filled)).toMatchObject({ ready: 3, complete: 0 });

    const firstId = filled.items[0].review_item_id;
    const completed = completeItem(filled, firstId, ISO);
    expect(completed.ok).toBe(true);
    if (completed.ok) {
      expect(summarizeProgress(completed.session)).toMatchObject({
        total: 3,
        complete: 1,
        ready: 2,
      });
    }
  });

  it("setAllReviewers applies a reviewer without fabricating timestamps", () => {
    const packet = sanitizedPacket();
    const session = buildSessionFromPacket(packet);
    const updated = setAllReviewers(session, "Tester");
    expect(updated.items.every((item) => item.reviewer === "Tester")).toBe(true);
    expect(updated.items.every((item) => item.completed_at === null)).toBe(true);
    expect(updated.items.every((item) => item.complete === false)).toBe(true);
  });
});
