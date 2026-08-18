import { normalizeQueryResponse, parseSourceLine, splitEmbeddedSources, type QueryResponse } from "./query-response";

function response(answer: string, sources: string[] = []): QueryResponse {
  return {
    answer,
    sources,
    normalized_question: "How do I identify and execute my win condition?",
    metadata: {
      game: "lol",
    },
  };
}

describe("query response source splitting", () => {
  it("leaves answers without embedded sources unchanged", () => {
    const result = normalizeQueryResponse(response("Answer only."));

    expect(result.answer).toBe("Answer only.");
    expect(result.sources).toEqual([]);
  });

  it("moves standard Sources blocks out of the answer", () => {
    const split = splitEmbeddedSources(
      "Answer text.\n\n---\nSources:\n[0.38 | conf 0.81 | src-w 2.13 | supabase] (macro_advice) Row one"
    );

    expect(split.answer).toBe("Answer text.");
    expect(split.sources).toEqual([
      "Sources:",
      "[0.38 | conf 0.81 | src-w 2.13 | supabase] (macro_advice) Row one",
    ]);
  });

  it("moves legacy raw bracketed source rows out of the answer", () => {
    const result = normalizeQueryResponse(
      response(
        "Answer text.\n" +
          "[0.38 | conf 0.81 | src-w 2.13 | supabase] (macro_advice) Identify the win condition.\n" +
          "[0.44 | conf 0.63 | src-w 1.60 | discord] (principles) Always plan."
      )
    );

    expect(result.answer).toBe("Answer text.");
    expect(result.answer).not.toContain("[0.38 | conf");
    expect(result.sources).toHaveLength(2);
    expect(result.sources[0]).toContain("(macro_advice)");
  });

  it("moves raw source rows even when they are appended without a newline", () => {
    const result = normalizeQueryResponse(
      response(
        "Answer text.[0.38 | conf 0.81 | src-w 2.13 | supabase] (macro_advice) Identify the win condition. " +
          "[0.44 | conf 0.63 | src-w 1.60 | discord] (principles) Always plan."
      )
    );

    expect(result.answer).toBe("Answer text.");
    expect(result.sources).toHaveLength(2);
    expect(result.sources[0]).toContain("(macro_advice)");
    expect(result.sources[1]).toContain("(principles)");
  });

  it("strips embedded sources from the answer even when API sources are already present", () => {
    const result = normalizeQueryResponse(
      response(
        "Answer text.\n\n---\nSources:\n[0.38 | conf 0.81] (macro_advice) Embedded row",
        ["[0.50 | conf 0.90] (principles) API row"]
      )
    );

    expect(result.answer).toBe("Answer text.");
    expect(result.sources).toEqual(["[0.50 | conf 0.90] (principles) API row"]);
  });

  it("parses source metrics and categories for evidence rendering", () => {
    const parsed = parseSourceLine("[0.38 | conf 0.81 | src-w 2.13] (macro_advice) Identify the win condition.");

    expect(parsed.metrics).toBe("0.38 | conf 0.81 | src-w 2.13");
    expect(parsed.category).toBe("macro_advice");
    expect(parsed.text).toBe("Identify the win condition.");
  });
});
