"use client";

import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import Link from "next/link";
import styles from "../app/phase2j-review/review.module.css";
import {
  addEndpointToWindow,
  applyOutcome,
  buildExportSession,
  buildSessionFromPayload,
  createEndpoint,
  deriveEndpointSpan,
  ENDPOINT_TYPES,
  findOverlappingEndpoint,
  markPassAComplete,
  nextEndpointSequence,
  removeEndpointFromWindow,
  snapCharRangeToTokens,
  summarizeProgress,
  validateSessionInput,
  type EndpointType,
  type Phase2JReviewPayload,
  type ReviewEndpoint,
  type ReviewSession,
  type SessionRecord,
  type WindowOutcome,
} from "../lib/phase2j-review";

const OUTCOMES: readonly WindowOutcome[] = ["CLEAN", "AMBIGUOUS", "EXCLUDED"];

const TYPE_GUIDE: Record<
  EndpointType,
  { label: string; description: string; example: string }
> = {
  ENTITY: {
    label: "Entity",
    description: "A participant, target, object, or unresolved pronoun.",
    example: "he · you · the enemy",
  },
  ABILITY_OR_RESOURCE: {
    label: "Ability / resource",
    description: "A spell, ability, cooldown, health, mana, gold, or similar resource.",
    example: "W · Flash · mana",
  },
  EVENT: {
    label: "Event",
    description: "Something that happens, without framing it as advice or an intended move.",
    example: "the wave crashes · dragon spawns",
  },
  ACTION: {
    label: "Action",
    description: "A performed, avoided, required, or recommended move.",
    example: "push the wave · didn't W",
  },
  STATE: {
    label: "State",
    description: "A condition that is currently true or describes the situation.",
    example: "you are low · W is down",
  },
  OUTCOME: {
    label: "Outcome",
    description: "A consequence or result produced by the situation or action.",
    example: "you are dead · they lose the wave",
  },
  QUANTITY: {
    label: "Quantity",
    description: "An explicit amount, count, level, duration, or measurement.",
    example: "100 HP · two waves",
  },
  TIME: {
    label: "Time",
    description: "A timing point, interval, sequence cue, or temporal condition.",
    example: "after level six · before dragon",
  },
  LOCATION_OR_SPACE: {
    label: "Location / space",
    description: "A place, direction, range, or positional region.",
    example: "under tower · in the river",
  },
  UNDETERMINED: {
    label: "Undetermined",
    description: "The exact meaningful span is clear, but its endpoint type genuinely is not.",
    example: "Use sparingly; add a reviewer note",
  },
};

type PickerState = {
  tokenStart: number;
  tokenEnd: number;
  charStart: number;
  charEnd: number;
  phrase: string;
  position: { top: number; left: number; anchor: "above" | "below" };
};

type Message = { kind: "error" | "info"; text: string };

function charOffsetAt(root: HTMLElement, container: Node, offset: number): number {
  const prefix = document.createRange();
  prefix.selectNodeContents(root);
  prefix.setEnd(container, offset);
  return prefix.toString().length;
}

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function Phase2JReviewClient({ payload }: { payload: Phase2JReviewPayload }) {
  const [session, setSession] = useState<ReviewSession>(() => buildSessionFromPayload(payload));
  const [hydrated, setHydrated] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [picker, setPicker] = useState<PickerState | null>(null);
  const [message, setMessage] = useState<Message | null>(null);
  const [revealed, setRevealed] = useState<ReadonlySet<string>>(new Set());
  const [focusedEndpointId, setFocusedEndpointId] = useState<string | null>(null);
  const [pendingExcluded, setPendingExcluded] = useState(false);
  const [undoStack, setUndoStack] = useState<ReviewSession[]>([]);

  const sourceRef = useRef<HTMLParagraphElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sessionRef = useRef(session);
  const currentIndexRef = useRef(currentIndex);
  const pickerRefState = useRef(picker);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  useEffect(() => {
    pickerRefState.current = picker;
  }, [picker]);

  const storageKey = `phase2j-review-session:v1:${payload.packet_sha256}`;
  const record = session.records[currentIndex];
  const progress = useMemo(() => summarizeProgress(session), [session]);

  // Hydrate the local autosave bound to the exact packet content hash.
  useEffect(() => {
    let restored: ReviewSession | null = null;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw) {
        const parsed: unknown = JSON.parse(raw);
        const result = validateSessionInput(parsed, payload.packet_sha256, payload.records);
        if (result.ok) {
          restored = result.session;
        } else {
          setMessage({
            kind: "error",
            text: `Saved session from this browser was ignored: ${result.errors[0]}`,
          });
        }
      }
    } catch {
      // Corrupt autosave: start fresh; nothing is written to the canonical packet.
    }
    if (restored) {
      setSession(restored);
      setMessage({ kind: "info", text: "Restored your saved session from this browser." });
    }
    setHydrated(true);
  }, [storageKey, payload.packet_sha256, payload.records]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(session));
    } catch {
      // Storage may be unavailable (private mode / quota); review still works.
    }
  }, [session, hydrated, storageKey]);

  const commitRecord = useCallback((nextRecord: SessionRecord, trackUndo = true) => {
    if (trackUndo) {
      setUndoStack((stack) => [...stack.slice(-49), sessionRef.current]);
    }
    const index = currentIndexRef.current;
    setSession({
      ...sessionRef.current,
      records: sessionRef.current.records.map((existing, existingIndex) =>
        existingIndex === index ? nextRecord : existing,
      ),
    });
  }, []);

  const goTo = useCallback((index: number) => {
    const total = sessionRef.current.records.length;
    const next = Math.min(Math.max(index, 0), total - 1);
    setCurrentIndex(next);
    setPicker(null);
    setFocusedEndpointId(null);
    setPendingExcluded(false);
  }, []);

  const endpointByToken = useMemo(() => {
    const map = new Map<number, ReviewEndpoint>();
    for (const endpoint of record.endpoints) {
      for (let tokenIndex = endpoint.token_start; tokenIndex <= endpoint.token_end; tokenIndex += 1) {
        map.set(tokenIndex, endpoint);
      }
    }
    return map;
  }, [record.endpoints]);

  const handleSelection = useCallback(() => {
    const root = sourceRef.current;
    const selection = window.getSelection?.();
    if (
      !root ||
      !selection ||
      selection.isCollapsed ||
      selection.rangeCount === 0 ||
      pickerRefState.current
    ) {
      return;
    }
    const currentRecord = sessionRef.current.records[currentIndexRef.current];
    if (currentRecord.outcome === "EXCLUDED") {
      return;
    }
    const range = selection.getRangeAt(0);
    if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) {
      return;
    }
    const first = charOffsetAt(root, range.startContainer, range.startOffset);
    const second = charOffsetAt(root, range.endContainer, range.endOffset);
    const charStart = Math.min(first, second);
    const charEnd = Math.max(first, second);
    const snapped = snapCharRangeToTokens(currentRecord.tokens, charStart, charEnd);
    if (!snapped) {
      return;
    }
    const span = deriveEndpointSpan(
      currentRecord,
      snapped.token_start,
      snapped.token_end,
    );
    if (!span) {
      return;
    }
    const rect = range.getBoundingClientRect();
    const estimatedPickerHeight = 430;
    const below = rect.bottom + 8 + estimatedPickerHeight <= window.innerHeight;
    const top = below ? rect.bottom + 8 : Math.max(8, rect.top - 8 - estimatedPickerHeight);
    const left = Math.min(
      Math.max(rect.left + rect.width / 2 - 190, 8),
      Math.max(8, window.innerWidth - 408),
    );
    setPicker({
      tokenStart: snapped.token_start,
      tokenEnd: snapped.token_end,
      charStart: span.char_start,
      charEnd: span.char_end,
      phrase: span.exact_bronze_text,
      position: { top, left, anchor: below ? "below" : "above" },
    });
  }, []);

  const handleTouchSelection = useCallback(() => {
    window.setTimeout(() => handleSelection(), 0);
  }, [handleSelection]);

  const closePicker = useCallback(() => {
    setPicker(null);
    window.getSelection()?.removeAllRanges();
  }, []);

  useEffect(() => {
    if (!picker) {
      return;
    }
    const firstButton = pickerRef.current?.querySelector<HTMLButtonElement>("button");
    const timer = window.setTimeout(() => firstButton?.focus(), 0);
    return () => window.clearTimeout(timer);
  }, [picker]);

  const acceptEndpoint = useCallback(
    (nodeType: EndpointType) => {
      if (!picker) {
        return;
      }
      const currentRecord = sessionRef.current.records[currentIndexRef.current];
      if (currentRecord.outcome === "EXCLUDED") {
        setMessage({
          kind: "error",
          text: "EXCLUDED windows cannot contain endpoints. Switch the outcome back to CLEAN first.",
        });
        closePicker();
        return;
      }
      const endpoint = createEndpoint(
        currentRecord,
        picker.tokenStart,
        picker.tokenEnd,
        nodeType,
        nextEndpointSequence(currentRecord.endpoints),
      );
      if (!endpoint) {
        setMessage({ kind: "error", text: "Could not accept that span; the token range is invalid." });
        closePicker();
        return;
      }
      const overlap = findOverlappingEndpoint(currentRecord.endpoints, endpoint);
      if (overlap) {
        setMessage({
          kind: "error",
          text: `Rejected: that span overlaps “${overlap.exact_bronze_text}” (${overlap.node_type}). Exact Bronze spans must not overlap.`,
        });
        closePicker();
        return;
      }
      const updated = addEndpointToWindow(currentRecord, endpoint);
      commitRecord(updated);
      setRevealed((previous) => new Set(previous).add(endpoint.endpoint_id));
      setFocusedEndpointId(endpoint.endpoint_id);
      window.setTimeout(() => {
        setRevealed((previous) => {
          const next = new Set(previous);
          next.delete(endpoint.endpoint_id);
          return next;
        });
      }, 800);
      setMessage({
        kind: "info",
        text: `Accepted “${endpoint.exact_bronze_text}” as ${nodeType}.`,
      });
      closePicker();
    },
    [picker, commitRecord, closePicker],
  );

  const focusEndpoint = useCallback((endpoint: ReviewEndpoint) => {
    setFocusedEndpointId(endpoint.endpoint_id);
    const tokenElement = sourceRef.current?.querySelector<HTMLElement>(
      `[data-token-index="${endpoint.token_start}"]`,
    );
    tokenElement?.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "nearest",
    });
  }, []);

  const removeEndpoint = useCallback(
    (endpointId: string) => {
      const currentRecord = sessionRef.current.records[currentIndexRef.current];
      const endpoint = currentRecord.endpoints.find((item) => item.endpoint_id === endpointId);
      commitRecord(removeEndpointFromWindow(currentRecord, endpointId));
      setFocusedEndpointId(null);
      setMessage({
        kind: "info",
        text: `Removed ${endpoint ? `“${endpoint.exact_bronze_text}”` : "endpoint"}. This change is local and recoverable via undo or a prior export.`,
      });
    },
    [commitRecord],
  );

  const changeOutcome = useCallback(
    (outcome: WindowOutcome) => {
      const currentRecord = sessionRef.current.records[currentIndexRef.current];
      if (outcome === "EXCLUDED" && currentRecord.endpoints.length > 0) {
        setPendingExcluded(true);
        return;
      }
      const updated = applyOutcome(currentRecord, outcome);
      commitRecord(updated);
      setPendingExcluded(false);
      if (outcome !== "CLEAN" && updated.note.trim() === "") {
        setMessage({
          kind: "info",
          text: `${outcome} windows require a reviewer note before they can be exported.`,
        });
      }
    },
    [commitRecord],
  );

  const confirmExcluded = useCallback(() => {
    const currentRecord = sessionRef.current.records[currentIndexRef.current];
    commitRecord(applyOutcome(currentRecord, "EXCLUDED"));
    setPendingExcluded(false);
    setFocusedEndpointId(null);
    setMessage({
      kind: "info",
      text: `Window ${currentRecord.record_index} excluded; its ${currentRecord.endpoints.length} endpoint(s) were cleared locally.`,
    });
  }, [commitRecord]);

  const setNote = useCallback(
    (note: string) => {
      const currentRecord = sessionRef.current.records[currentIndexRef.current];
      commitRecord({ ...currentRecord, note }, false);
    },
    [commitRecord],
  );

  const setReviewer = useCallback(
    (reviewerName: string) => {
      const currentRecord = sessionRef.current.records[currentIndexRef.current];
      commitRecord(
        markPassAComplete(
          currentRecord,
          reviewerName,
          currentRecord.completed_at,
          currentRecord.pass_a_complete,
        ),
        false,
      );
    },
    [commitRecord],
  );

  const setCompletedAt = useCallback(
    (completedAt: string) => {
      const currentRecord = sessionRef.current.records[currentIndexRef.current];
      commitRecord(
        markPassAComplete(
          currentRecord,
          currentRecord.reviewer_name,
          completedAt || null,
          currentRecord.pass_a_complete,
        ),
        false,
      );
    },
    [commitRecord],
  );

  const togglePassAComplete = useCallback(() => {
    const currentRecord = sessionRef.current.records[currentIndexRef.current];
    const nextComplete = !currentRecord.pass_a_complete;
    if (nextComplete && (currentRecord.reviewer_name.trim() === "" || !currentRecord.completed_at)) {
      setMessage({
        kind: "error",
        text: "Reviewer name and Pass A date are required before marking Pass A complete.",
      });
      return;
    }
    if (nextComplete && currentRecord.outcome !== "CLEAN" && currentRecord.note.trim() === "") {
      setMessage({
        kind: "error",
        text: `A reviewer note is required before completing an ${currentRecord.outcome} window.`,
      });
      return;
    }
    commitRecord(
      markPassAComplete(
        currentRecord,
        currentRecord.reviewer_name,
        currentRecord.completed_at,
        nextComplete,
      ),
    );
  }, [commitRecord]);

  const undo = useCallback(() => {
    setUndoStack((stack) => {
      if (stack.length === 0) {
        setMessage({ kind: "info", text: "Nothing left to undo in this session." });
        return stack;
      }
      const previous = stack[stack.length - 1];
      setSession(previous);
      setPicker(null);
      setFocusedEndpointId(null);
      setPendingExcluded(false);
      setMessage({ kind: "info", text: "Undid the last edit in this window." });
      return stack.slice(0, -1);
    });
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPicker(null);
        return;
      }
      const target = event.target as HTMLElement | null;
      const typing =
        target !== null &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);
      if (typing || picker) {
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        goTo(currentIndexRef.current - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        goTo(currentIndexRef.current + 1);
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        undo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goTo, picker, undo]);

  const handleExport = useCallback(() => {
    const result = validateSessionInput(session, payload.packet_sha256, payload.records);
    if (!result.ok) {
      setMessage({
        kind: "error",
        text: `Export blocked: ${result.errors.slice(0, 3).join(" ")}`,
      });
      return;
    }
    const exportSession = buildExportSession(result.session, new Date().toISOString());
    const json = `${JSON.stringify(exportSession, null, 2)}\n`;
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `phase2j-review-session-${payload.packet_sha256.slice(0, 8)}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 2000);
    setMessage({
      kind: "info",
      text: "Review session exported. This is review material, not final gold.",
    });
  }, [session, payload.packet_sha256, payload.records]);

  const handleImportFile = useCallback(
    async (file: File) => {
      try {
        const text = await file.text();
        const parsed: unknown = JSON.parse(text);
        const result = validateSessionInput(parsed, payload.packet_sha256, payload.records);
        if (!result.ok) {
          setMessage({
            kind: "error",
            text: `Import rejected: ${result.errors.slice(0, 3).join(" ")}`,
          });
          return;
        }
        setSession(result.session);
        setUndoStack([]);
        setCurrentIndex(0);
        setPicker(null);
        setFocusedEndpointId(null);
        setPendingExcluded(false);
        setMessage({
          kind: "info",
          text: "Backup imported and replacing the local review session.",
        });
      } catch {
        setMessage({ kind: "error", text: "Import rejected: the file is not valid JSON." });
      }
    },
    [payload.packet_sha256, payload.records],
  );

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void handleImportFile(file);
      }
      event.target.value = "";
    },
    [handleImportFile],
  );

  const overviewClass = (item: SessionRecord): string => {
    switch (item.window_status) {
      case "AMBIGUOUS":
        return styles.cellAmbiguous;
      case "EXCLUDED":
        return styles.cellExcluded;
      case "IN_REVIEW":
        return item.pass_a_complete ? styles.cellComplete : styles.cellInReview;
      case "UNREVIEWED":
        return styles.cellUnreviewed;
    }
  };

  const overviewLabel = (item: SessionRecord): string => {
    switch (item.window_status) {
      case "AMBIGUOUS":
        return "Ambiguous";
      case "EXCLUDED":
        return "Excluded";
      case "IN_REVIEW":
        return item.pass_a_complete ? "Pass A complete" : "In review";
      case "UNREVIEWED":
        return "Unreviewed";
    }
  };

  const requiresNote = record.outcome !== "CLEAN";
  const outcomeHelp: Record<WindowOutcome, string> = {
    CLEAN: "Usable Bronze window: mark every existing source mention as an exact Bronze span.",
    AMBIGUOUS:
      "Some literal spans may remain unresolved; no inference from outside knowledge. Add a reviewer note explaining the ambiguity.",
    EXCLUDED:
      "Wholly unreliable ASR or context-truncated window: no endpoints may be accepted and a reviewer note is required.",
  };

  return (
    <div className={styles.container}>
      <div className={styles.inner}>
        <header className={styles.topbar}>
          <Link href="/" className={styles.backLink} aria-label="Back to query home">
            ← Home
          </Link>
          <div className={styles.titleBlock}>
            <h1 className={styles.title}>Phase 2J · Bronze endpoint review</h1>
            <p className={styles.subtitle}>
              Pass A review of the 30 locked Bronze windows. An endpoint is the exact Bronze
              mention, not a resolved identity: do not repair ASR or infer from outside knowledge.
            </p>
          </div>
          <div className={styles.actions}>
            <button
              type="button"
              className={styles.actionButton}
              onClick={() => fileInputRef.current?.click()}
            >
              Import backup
            </button>
            <button type="button" className={styles.exportButton} onClick={handleExport}>
              Export review
            </button>
          </div>
        </header>

        <section className={styles.progressBlock} aria-label="Session progress">
          <div className={styles.progressText}>
            <span>
              Window {currentIndex + 1} of {session.records.length}
            </span>
            <span>
              {progress.pass_a_complete} Pass A complete · {progress.endpoints} endpoints
            </span>
          </div>
          <div
            className={styles.progressTrack}
            role="progressbar"
            aria-label="Review deck progress"
            aria-valuemin={1}
            aria-valuemax={session.records.length}
            aria-valuenow={currentIndex + 1}
          >
            <div
              className={styles.progressFill}
              style={{ width: `${((currentIndex + 1) / session.records.length) * 100}%` }}
            />
          </div>
          <div className={styles.overview} role="group" aria-label="Completion overview">
            {session.records.map((item, index) => (
              <button
                key={item.window_id}
                type="button"
                className={`${styles.overviewCell} ${overviewClass(item)} ${
                  index === currentIndex ? styles.overviewCurrent : ""
                }`}
                onClick={() => goTo(index)}
                title={`Window ${index + 1}: ${overviewLabel(item)}`}
                aria-label={`Go to window ${index + 1}, ${overviewLabel(item)}`}
                aria-current={index === currentIndex ? "true" : undefined}
              >
                {index + 1}
              </button>
            ))}
          </div>
        </section>

        <nav className={styles.deckControls} aria-label="Window navigation">
          <button
            type="button"
            className={styles.navButton}
            onClick={() => goTo(currentIndex - 1)}
            disabled={currentIndex === 0}
            aria-label="Previous window"
          >
            ← Prev
          </button>
          <span className={styles.deckPosition}>{overviewLabel(record)}</span>
          <button
            type="button"
            className={styles.navButton}
            onClick={() => goTo(currentIndex + 1)}
            disabled={currentIndex === session.records.length - 1}
            aria-label="Next window"
          >
            Next →
          </button>
        </nav>

        <main className={styles.windowCard}>
          <div className={styles.windowMeta}>
            <span className={styles.metaTag}>Window {record.record_index}</span>
            <span className={styles.metaTag}>{record.bronze_char_length} chars</span>
            <span className={styles.metaTag}>{record.tokens.length} tokens</span>
          </div>
          <p
            ref={sourceRef}
            className={styles.sourceText}
            onMouseUp={handleSelection}
            onTouchEnd={handleTouchSelection}
            aria-label={`Bronze window ${record.record_index} source text`}
          >
            {record.tokens.map((token, index) => {
              const endpoint = endpointByToken.get(token.token_index);
              const tokenClasses = [styles.token];
              if (endpoint) {
                tokenClasses.push(styles.tokenAccepted);
                if (endpoint.endpoint_id === focusedEndpointId) {
                  tokenClasses.push(styles.tokenFocused);
                }
                if (revealed.has(endpoint.endpoint_id)) {
                  tokenClasses.push(styles.tokenReveal);
                }
              }
              const gap =
                index < record.tokens.length - 1
                  ? record.bronze_text.slice(token.end, record.tokens[index + 1].start)
                  : record.bronze_text.slice(token.end);
              return (
                <Fragment key={token.token_index}>
                  <span className={tokenClasses.join(" ")} data-token-index={token.token_index}>
                    {token.text}
                  </span>
                  {gap.length > 0 ? <span className={styles.gap}>{gap}</span> : null}
                </Fragment>
              );
            })}
          </p>
          <p className={styles.ruleNote}>
            Endpoint = exact Bronze mention, not a resolved identity. Do not repair ASR or infer
            from outside knowledge.
          </p>
          <p className={styles.selectionHint}>
            Drag across the sentence to select an exact mention, then choose a node type.
          </p>

          <details className={styles.legend} open>
            <summary className={styles.legendSummary}>Endpoint type legend</summary>
            <p className={styles.legendIntro}>
              Choose the role the exact phrase plays in this sentence. Examples are fabricated
              guidance, not suggested answers for the current window.
            </p>
            <div className={styles.legendGrid}>
              {ENDPOINT_TYPES.map((type) => {
                const guide = TYPE_GUIDE[type];
                return (
                  <div className={styles.legendItem} key={type}>
                    <div>
                      <strong className={styles.legendType}>{guide.label}</strong>
                      <span className={styles.legendCode}>{type}</span>
                    </div>
                    <p className={styles.legendDescription}>{guide.description}</p>
                    <p className={styles.legendExample}>{guide.example}</p>
                  </div>
                );
              })}
            </div>
            <p className={styles.legendRule}>
              One span, one type. Keep actor and action separate when they do not overlap—for
              example, <code>he</code> as Entity and <code>didn't W</code> as Action. Do not also
              add <code>W</code> inside that selected action.
            </p>
          </details>

          <section
            className={styles.rail}
            aria-label={`Accepted endpoints in window ${record.record_index}`}
          >
            <h2 className={styles.railHeading}>
              Accepted endpoints <span className={styles.countPill}>{record.endpoints.length}</span>
            </h2>
            {record.endpoints.length === 0 ? (
              <p className={styles.railEmpty}>No endpoints accepted yet.</p>
            ) : (
              <ol className={styles.railList}>
                {record.endpoints.map((endpoint) => (
                  <li key={endpoint.endpoint_id} className={styles.railRow}>
                    <button
                      type="button"
                      className={styles.railMain}
                      onClick={() => focusEndpoint(endpoint)}
                      aria-label={`Show span ${endpoint.exact_bronze_text}`}
                    >
                      <span className={styles.railPhrase}>“{endpoint.exact_bronze_text}”</span>
                      <span className={styles.railType}>{endpoint.node_type}</span>
                      <span className={styles.railRange}>
                        tok {endpoint.token_start}–{endpoint.token_end} · ch {endpoint.char_start}–
                        {endpoint.char_end}
                      </span>
                    </button>
                    <button
                      type="button"
                      className={styles.removeButton}
                      onClick={() => removeEndpoint(endpoint.endpoint_id)}
                      aria-label={`Remove endpoint ${endpoint.exact_bronze_text}`}
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </main>

        <section className={styles.statusPanel} aria-label="Window outcome">
          <h2>Window outcome</h2>
          <div className={styles.outcomeRow} role="radiogroup" aria-label="Outcome">
            {OUTCOMES.map((outcome) => (
              <button
                key={outcome}
                type="button"
                role="radio"
                aria-checked={record.outcome === outcome}
                className={`${styles.outcomeButton} ${
                  record.outcome === outcome ? styles.outcomeActive : ""
                }`}
                onClick={() => changeOutcome(outcome)}
              >
                {outcome}
              </button>
            ))}
          </div>
          <p className={styles.outcomeHelp}>{outcomeHelp[record.outcome]}</p>
          <label className={styles.noteLabel} htmlFor="window-note">
            Reviewer note{" "}
            {requiresNote ? <span className={styles.required}>required</span> : null}
          </label>
          <textarea
            id="window-note"
            className={styles.noteInput}
            value={record.note}
            onChange={(event) => setNote(event.target.value)}
            placeholder={
              record.outcome === "CLEAN"
                ? "Optional context for the next reviewer."
                : "Explain why this window is AMBIGUOUS or EXCLUDED."
            }
          />
          {requiresNote && record.note.trim() === "" ? (
            <p className={styles.requiredNote}>
              A reviewer note is required for {record.outcome.toLowerCase()} windows before export.
            </p>
          ) : null}

          <div className={styles.passRow}>
            <div className={styles.passFields}>
              <label className={styles.fieldLabel}>
                Reviewer name
                <input
                  className={styles.fieldInput}
                  value={record.reviewer_name}
                  onChange={(event) => setReviewer(event.target.value)}
                  placeholder="Your name"
                />
              </label>
              <label className={styles.fieldLabel}>
                Pass A date
                <input
                  type="date"
                  className={styles.fieldInput}
                  value={record.completed_at ?? ""}
                  onChange={(event) => setCompletedAt(event.target.value)}
                />
              </label>
            </div>
            <button
              type="button"
              className={styles.completeButton}
              onClick={togglePassAComplete}
              aria-pressed={record.pass_a_complete}
            >
              {record.pass_a_complete ? "Pass A complete" : "Mark Pass A complete"}
            </button>
          </div>
          <p className={styles.passNote}>
            Pass B is a later blinded audit and is not part of this route. Clean completed windows
            stay IN_REVIEW; ambiguous and excluded outcomes retain their explicit status. This
            review session is not final gold.
          </p>
        </section>

        {pendingExcluded ? (
          <div className={styles.confirmPanel} role="alertdialog" aria-label="Confirm exclusion">
            <p>
              Excluding this window removes {record.endpoints.length} accepted endpoint(s) locally.
              Continue?
            </p>
            <div className={styles.confirmActions}>
              <button type="button" className={styles.confirmButton} onClick={confirmExcluded}>
                Confirm exclusion
              </button>
              <button
                type="button"
                className={styles.cancelButton}
                onClick={() => setPendingExcluded(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : null}

        {message ? (
          <p
            role={message.kind === "error" ? "alert" : "status"}
            className={message.kind === "error" ? styles.messageError : styles.messageInfo}
          >
            {message.text}
          </p>
        ) : null}

        {picker ? (
          <>
            <div className={styles.pickerBackdrop} onMouseDown={closePicker} aria-hidden="true" />
            <div
              ref={pickerRef}
              className={styles.picker}
              style={{
                top: picker.position.top,
                left: picker.position.left,
              }}
              role="menu"
              aria-label="Endpoint type picker"
              aria-describedby="picker-note"
            >
              <p className={styles.pickerPhrase}>“{picker.phrase}”</p>
              <div className={styles.pickerGrid}>
                {ENDPOINT_TYPES.map((type) => (
                  <button
                    key={type}
                    type="button"
                    role="menuitem"
                    className={styles.pickerButton}
                    onClick={() => acceptEndpoint(type)}
                  >
                    <span className={styles.pickerButtonLabel}>{TYPE_GUIDE[type].label}</span>
                    <span className={styles.pickerButtonExample}>{TYPE_GUIDE[type].example}</span>
                  </button>
                ))}
              </div>
              <p id="picker-note" className={styles.pickerNote}>
                Accepting the exact Bronze span as this node type. Escape cancels.
              </p>
            </div>
          </>
        ) : null}

        <input
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          className={styles.hiddenInput}
          onChange={handleFileChange}
          tabIndex={-1}
          aria-label="Import a prior review session backup"
        />

        <footer className={styles.footer}>
          Review material only — not final gold. Nothing here is written to the canonical packet.
        </footer>
      </div>
    </div>
  );
}
