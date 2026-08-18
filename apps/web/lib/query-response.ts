export type QueryResponse = {
  answer: string;
  sources: string[];
  normalized_question: string;
  metadata: {
    game: string;
    subject?: string | null;
    role?: string | null;
    reasoning?: string | null;
    backend_label?: string | null;
    backend_quality?: string | null;
    retrieval_mode?: string | null;
    semantic_enabled?: boolean | null;
  };
};

export type ParsedSource = {
  metrics: string | null;
  category: string | null;
  text: string;
};

type SplitResponse = {
  answer: string;
  sources: string[];
};

export function parseSourceLine(line: string): ParsedSource {
  const trimmed = line.trim();
  if (/^sources/i.test(trimmed)) {
    return { metrics: null, category: null, text: trimmed };
  }

  const match = trimmed.match(/^\[(.+?)\]\s+\((.+?)\)\s+(.+)$/);
  if (match) {
    return {
      metrics: match[1],
      category: match[2],
      text: match[3]
    };
  }

  return { metrics: null, category: null, text: trimmed };
}

export function splitEmbeddedSources(answer: string): SplitResponse {
  const markerMatch = answer.match(/\n+\s*---\s*\n\s*(Sources(?:[^\n]*)?)\n/i);
  if (markerMatch?.index !== undefined) {
    const answerText = answer.slice(0, markerMatch.index).trimEnd();
    const sourceText = answer
      .slice(markerMatch.index)
      .replace(/^\s*---\s*\n/, "")
      .trim();
    const sources = sourceText
      .split(/\r?\n/)
      .map((line) => line.trimEnd())
      .filter((line) => line.trim().length > 0);

    return { answer: answerText, sources };
  }

  const rawSourceMatch = answer.match(
    /\[\d+(?:\.\d+)?\s*\|\s*conf\s+[^\]]+\]\s+\([^)]+\)\s+/i
  );
  if (rawSourceMatch?.index === undefined) {
    return { answer, sources: [] };
  }

  const answerText = answer.slice(0, rawSourceMatch.index).trimEnd();
  const sources = answer
    .slice(rawSourceMatch.index)
    .split(/\s*(?=\[\d+(?:\.\d+)?\s*\|\s*conf\s+)/i)
    .map((line) => line.trimEnd())
    .filter((line) => line.trim().length > 0);

  return { answer: answerText, sources };
}

export function normalizeQueryResponse(response: QueryResponse): QueryResponse {
  const split = splitEmbeddedSources(response.answer);
  if (split.sources.length === 0) {
    return response;
  }
  return {
    ...response,
    answer: split.answer,
    sources: response.sources.length > 0 ? response.sources : split.sources,
  };
}
