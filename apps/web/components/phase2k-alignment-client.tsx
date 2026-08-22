"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from "react";
import Link from "next/link";
import styles from "../app/phase2k-align/align.module.css";
import {
  addSpan,
  ALIGNMENT_DECISION_STATES,
  buildDecisionsMap,
  buildSessionExport,
  buildSessionFromPacket,
  bytesToHex,
  completeItem,
  DECISIONS_FILENAME,
  decisionsMapErrors,
  findExactOccurrences,
  itemMissingFields,
  removeSpan,
  sanitizePacket,
  setAllReviewers,
  setItemNotes,
  setItemReviewer,
  setItemState,
  summarizeProgress,
  uncompleteItem,
  utf8Bytes,
  validateSessionInput,
  type AlignmentPacket,
  type AlignmentSession,
  type DecisionState,
  type PolishedSpan,
  type Sha256Digest,
} from "../lib/phase2k-alignment";

const SESSION_PREFIX = "phase2k-alignment-session:v1";

type Message = { kind: "error" | "info"; text: string };

const STATE_HINTS: Record<DecisionState, string> = {
  ALIGNED: "one or more exact polished spans",
  ABSENT: "no span; target absent from the polished text",
  AMBIGUOUS: "optional spans; unresolved",
  MULTIPLE_CANDIDATES: "two or more exact polished spans",
};

function downloadJson(value: unknown, filename: string) {
  const json = `${JSON.stringify(value, null, 2)}\n`;
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function Phase2KAlignmentClient() {
  const [packet, setPacket] = useState<AlignmentPacket | null>(null);
  const [session, setSession] = useState<AlignmentSession | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [message, setMessage] = useState<Message | null>(null);
  const [selection, setSelection] = useState<{ start: number; end: number } | null>(null);
  const [exactTargets, setExactTargets] = useState<PolishedSpan[] | null>(null);
  const [manualStart, setManualStart] = useState("");
  const [manualEnd, setManualEnd] = useState("");

  const packetInputRef = useRef<HTMLInputElement>(null);
  const sessionInputRef = useRef<HTMLInputElement>(null);
  const sessionRef = useRef<AlignmentSession | null>(null);
  const currentIndexRef = useRef(0);
  const packetRef = useRef<AlignmentPacket | null>(null);
  const polishedRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  useEffect(() => {
    packetRef.current = packet;
  }, [packet]);

  const webDigest = useCallback<Sha256Digest>(async (bytes) => {
    const subtle = globalThis.crypto?.subtle;
    if (!subtle) {
      throw new Error("Web Crypto is unavailable in this browser");
    }
    return bytesToHex(await subtle.digest("SHA-256", bytes as unknown as BufferSource));
  }, []);

  const storageKey = useMemo(
    () => (packet ? `${SESSION_PREFIX}:${packet.content_sha256}` : null),
    [packet],
  );

  const item = session?.items[currentIndex] ?? null;
  const progress = useMemo(() => (session ? summarizeProgress(session) : null), [session]);
  const selectedSlice =
    item && selection
      ? item.representation.polished_text.slice(selection.start, selection.end)
      : null;

  // Restore the autosave bound to the exact packet hash after a packet load.
  useEffect(() => {
    if (!packet || !storageKey) {
      return;
    }
    let restored: AlignmentSession | null = null;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw) {
        const parsed: unknown = JSON.parse(raw);
        const result = validateSessionInput(parsed, packet);
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
      // Corrupt or unavailable storage: start a fresh session.
    }
    setSession(restored ?? buildSessionFromPacket(packet));
    setCurrentIndex(0);
    setExactTargets(null);
    setSelection(null);
    setHydrated(true);
  }, [packet, storageKey]);

  // Autosave once the packet-bound session is hydrated.
  useEffect(() => {
    if (!session || !packet || !hydrated || !storageKey) {
      return;
    }
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(session));
    } catch {
      // Storage may be unavailable (private mode / quota); review still works.
    }
  }, [session, packet, hydrated, storageKey]);

  const handlePacketFile = useCallback(
    async (file: File) => {
      try {
        const text = await file.text();
        const parsed: unknown = JSON.parse(text);
        const sanitized = await sanitizePacket(parsed, webDigest);
        setPacket(sanitized);
        setHydrated(false);
        setMessage({
          kind: "info",
          text: `Blank alignment packet accepted: ${sanitized.items.length} targets bound to ${sanitized.content_sha256.slice(0, 12)}….`,
        });
      } catch (error) {
        setMessage({
          kind: "error",
          text: `Packet rejected: ${error instanceof Error ? error.message : "invalid JSON"}`,
        });
      }
    },
    [webDigest],
  );

  const handlePacketInput = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void handlePacketFile(file);
      }
      event.target.value = "";
    },
    [handlePacketFile],
  );

  const handleSessionInput = useCallback(
    async (file: File) => {
      const currentPacket = packetRef.current;
      if (!currentPacket) {
        return;
      }
      try {
        const text = await file.text();
        const parsed: unknown = JSON.parse(text);
        const result = validateSessionInput(parsed, currentPacket);
        if (!result.ok) {
          setMessage({
            kind: "error",
            text: `Backup rejected: ${result.errors.slice(0, 2).join(" ")}`,
          });
          return;
        }
        setSession(result.session);
        setCurrentIndex(0);
        setExactTargets(null);
        setSelection(null);
        setMessage({ kind: "info", text: "Backup imported and replacing the local session." });
      } catch {
        setMessage({ kind: "error", text: "Backup rejected: the file is not valid JSON." });
      }
    },
    [],
  );

  const handleSessionInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void handleSessionInput(file);
      }
      event.target.value = "";
    },
    [handleSessionInput],
  );

  const goTo = useCallback((index: number) => {
    const total = sessionRef.current?.items.length ?? 0;
    if (total === 0) {
      return;
    }
    setCurrentIndex(Math.min(Math.max(index, 0), total - 1));
    setExactTargets(null);
    setSelection(null);
    setMessage(null);
  }, []);

  const goToNextIncomplete = useCallback(() => {
    const currentSession = sessionRef.current;
    if (!currentSession) {
      return;
    }
    const start = currentIndexRef.current;
    for (let offset = 1; offset <= currentSession.items.length; offset += 1) {
      const index = (start + offset) % currentSession.items.length;
      const candidate = currentSession.items[index];
      if (!candidate.decision.complete) {
        goTo(index);
        return;
      }
    }
    setMessage({ kind: "info", text: "Every target is already complete." });
  }, [goTo]);

  const commitState = useCallback((state: DecisionState | null) => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    try {
      const next = setItemState(currentSession, currentItem.alignment_id, state);
      setSession(next);
      setExactTargets(null);
      if (state === "ABSENT" || state === null) {
        setSelection(null);
      }
    } catch (error) {
      setMessage({
        kind: "error",
        text: error instanceof Error ? error.message : "Could not set that state.",
      });
    }
  }, []);

  const setReviewer = useCallback((reviewer: string) => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    setSession(setItemReviewer(currentSession, currentItem.alignment_id, reviewer));
  }, []);

  const applyReviewerToAll = useCallback(() => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem || currentItem.decision.reviewer.trim() === "") {
      setMessage({
        kind: "error",
        text: "Enter a reviewer name first, then apply it to every target.",
      });
      return;
    }
    setSession(setAllReviewers(currentSession, currentItem.decision.reviewer));
    setMessage({
      kind: "info",
      text: `Reviewer “${currentItem.decision.reviewer}” applied to all ${currentSession.items.length} targets.`,
    });
  }, []);

  const setNotes = useCallback((text: string) => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    const lines = text.split("\n").filter((line) => line !== "");
    setSession(setItemNotes(currentSession, currentItem.alignment_id, lines));
  }, []);

  const addSelectedSpan = useCallback(() => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem || !selection) {
      return;
    }
    if (selection.start >= selection.end) {
      setMessage({ kind: "error", text: "Select a non-empty substring in the polished text first." });
      return;
    }
    try {
      const next = addSpan(currentSession, currentItem.alignment_id, {
        start: selection.start,
        end: selection.end,
        text: currentItem.representation.polished_text.slice(selection.start, selection.end),
      });
      setSession(next);
      setSelection(null);
      setExactTargets(null);
      setMessage({
        kind: "info",
        text: `Added selected span ${selection.start}:${selection.end}.`,
      });
    } catch (error) {
      setMessage({
        kind: "error",
        text: error instanceof Error ? error.message : "Could not add that span.",
      });
    }
  }, [selection]);

  const addOccurrence = useCallback((span: PolishedSpan) => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    try {
      const next = addSpan(currentSession, currentItem.alignment_id, span);
      setSession(next);
      setExactTargets(null);
      setMessage({
        kind: "info",
        text: `Added exact occurrence ${span.start}:${span.end} “${span.text}”.`,
      });
    } catch (error) {
      setMessage({
        kind: "error",
        text: error instanceof Error ? error.message : "Could not add that occurrence.",
      });
    }
  }, []);

  const findExactTarget = useCallback(() => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    const needle = currentItem.bronze_target.evaluation_text;
    const occurrences = findExactOccurrences(
      currentItem.representation.polished_text,
      needle,
    );
    if (occurrences.length === 0) {
      setExactTargets(null);
      setMessage({
        kind: "error",
        text: `“${needle}” does not occur in the polished text; use manual offsets if needed.`,
      });
      return;
    }
    if (occurrences.length === 1) {
      addOccurrence(occurrences[0]);
      return;
    }
    setExactTargets(occurrences);
    setMessage({
      kind: "info",
      text: `${occurrences.length} exact occurrences found — choose one.`,
    });
  }, [addOccurrence]);

  const addManualSpan = useCallback(() => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    const start = Number(manualStart);
    const end = Number(manualEnd);
    if (!Number.isInteger(start) || !Number.isInteger(end)) {
      setMessage({ kind: "error", text: "Manual offsets must be integers." });
      return;
    }
    const polished = currentItem.representation.polished_text;
    try {
      const next = addSpan(currentSession, currentItem.alignment_id, {
        start,
        end,
        text: polished.slice(start, end),
      });
      setSession(next);
      setManualStart("");
      setManualEnd("");
      setExactTargets(null);
      setMessage({ kind: "info", text: `Added span ${start}:${end} after exact-slice validation.` });
    } catch (error) {
      setMessage({
        kind: "error",
        text: error instanceof Error ? error.message : "Could not add that span.",
      });
    }
  }, [manualStart, manualEnd]);

  const removeSpanAt = useCallback((index: number) => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    try {
      setSession(removeSpan(currentSession, currentItem.alignment_id, index));
    } catch (error) {
      setMessage({
        kind: "error",
        text: error instanceof Error ? error.message : "Could not remove that span.",
      });
    }
  }, []);

  const toggleComplete = useCallback(() => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    if (currentItem.decision.complete) {
      setSession(uncompleteItem(currentSession, currentItem.alignment_id));
      setMessage({
        kind: "info",
        text: "Completion retracted; the generated timestamp was cleared.",
      });
      return;
    }
    const result = completeItem(
      currentSession,
      currentItem.alignment_id,
      new Date().toISOString(),
    );
    if (!result.ok) {
      setMessage({
        kind: "error",
        text: `Cannot complete: ${result.errors.slice(0, 4).join("; ")}`,
      });
      return;
    }
    setSession(result.session);
    setMessage({ kind: "info", text: "Target marked complete." });
  }, []);

  const exportSession = useCallback(() => {
    const currentPacket = packetRef.current;
    const currentSession = sessionRef.current;
    if (!currentPacket || !currentSession) {
      return;
    }
    const result = validateSessionInput(currentSession, currentPacket);
    if (!result.ok) {
      setMessage({
        kind: "error",
        text: `Backup export blocked: ${result.errors.slice(0, 2).join(" ")}`,
      });
      return;
    }
    const exportValue = buildSessionExport(result.session, new Date().toISOString());
    downloadJson(
      exportValue,
      `phase2k-alignment-session-${currentPacket.content_sha256.slice(0, 8)}.json`,
    );
    setMessage({
      kind: "info",
      text: "Alignment session exported. This is review material, not final decisions.",
    });
  }, []);

  const exportDecisions = useCallback(() => {
    const currentSession = sessionRef.current;
    if (!currentSession) {
      return;
    }
    const errors = decisionsMapErrors(currentSession);
    if (errors.length > 0) {
      setMessage({
        kind: "error",
        text: `Final export blocked — ${errors.length} problem(s): ${errors.slice(0, 2).join(" ")}`,
      });
      return;
    }
    const map = buildDecisionsMap(currentSession);
    downloadJson(map, DECISIONS_FILENAME);
    setMessage({
      kind: "info",
      text: `Compact decisions map exported as ${DECISIONS_FILENAME} for the finalizer.`,
    });
  }, []);

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing =
        target !== null &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);
      if (typing || !sessionRef.current) {
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        goTo(currentIndexRef.current - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        goTo(currentIndexRef.current + 1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goTo]);

  if (!packet || !session) {
    return (
      <div className={styles.container}>
        <div className={styles.emptyInner}>
          <header className={styles.emptyHeader}>
            <Link href="/" className={styles.backLink} aria-label="Back to query home">
              ← Home
            </Link>
            <div className={styles.titleBlock}>
              <h1 className={styles.title}>Phase 2K · Downstream semantic-target alignment</h1>
              <p className={styles.subtitle}>
                Post-review target alignment for the sealed Phase 2K D representations. This
                workspace is explicitly model/scorer blind: no predictions, scores, or model
                results are ever loaded or shown.
              </p>
            </div>
          </header>
          <section className={styles.loadPanel}>
            <h2>Load the blank alignment packet</h2>
            <p>
              Select the official <code>phase2k-downstream-alignment-packet-v1.json</code> from
              this machine. The packet is validated strictly in your browser (schema, canonical
              content hash, every text hash, boundary rule, blank decisions), kept locally, and
              never uploaded.
            </p>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={() => packetInputRef.current?.click()}
            >
              Choose packet JSON
            </button>
            <input
              ref={packetInputRef}
              type="file"
              accept="application/json,.json"
              className={styles.hiddenInput}
              onChange={handlePacketInput}
              tabIndex={-1}
              aria-label="Load the blank Phase 2K downstream alignment packet JSON"
            />
          </section>
          {message ? (
            <p
              role={message.kind === "error" ? "alert" : "status"}
              className={message.kind === "error" ? styles.messageError : styles.messageInfo}
            >
              {message.text}
            </p>
          ) : null}
          <footer className={styles.footer}>
            Alignment material only — no decision, reviewer, or timestamp is fabricated, and
            nothing here writes the canonical packet.
          </footer>
        </div>
      </div>
    );
  }

  const missingHint = item ? itemMissingFields(item) : [];
  const corrected = item?.bronze_target.correction_status === "TERMINAL_PUNCTUATION_DROPPED";

  return (
    <div className={styles.container}>
      <div className={styles.inner}>
        <header className={styles.topbar}>
          <Link href="/" className={styles.backLink} aria-label="Back to query home">
            ← Home
          </Link>
          <div className={styles.titleBlock}>
            <h1 className={styles.title}>Phase 2K · Semantic-target alignment</h1>
            <p className={styles.subtitle}>
              Post-human-review target alignment bound to {` ${packet.content_sha256.slice(0, 12)}… `}
              and autosaved locally. No model, scorer, or prediction artifact is loaded or shown.
            </p>
          </div>
          <div className={styles.actions}>
            <button
              type="button"
              className={styles.actionButton}
              onClick={() => sessionInputRef.current?.click()}
            >
              Import backup
            </button>
            <button type="button" className={styles.actionButton} onClick={exportSession}>
              Export session
            </button>
            <button
              type="button"
              className={styles.exportButton}
              onClick={exportDecisions}
              disabled={progress?.complete !== progress?.total}
            >
              Export decisions
            </button>
          </div>
        </header>

        <section className={styles.progressBlock} aria-label="Session progress">
          <div className={styles.progressText}>
            <span>
              Target {currentIndex + 1} of {session.items.length}
            </span>
            <span>
              {progress
                ? `${progress.complete} complete · ${progress.ready} ready · ${progress.in_progress} in progress`
                : ""}
            </span>
          </div>
          <div
            className={styles.progressTrack}
            role="progressbar"
            aria-label="Alignment completion progress"
            aria-valuemin={0}
            aria-valuemax={session.items.length}
            aria-valuenow={progress?.complete ?? 0}
          >
            <div
              className={styles.progressFill}
              style={{
                width: `${((progress?.complete ?? 0) / Math.max(session.items.length, 1)) * 100}%`,
              }}
            />
          </div>
        </section>

        <section className={styles.listSection} aria-label="Target list">
          <div className={styles.listHeader}>
            <span className={styles.listTitle}>All targets (packet order)</span>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={goToNextIncomplete}
              disabled={progress?.complete === progress?.total}
            >
              Next incomplete ↓
            </button>
          </div>
          <div className={styles.targetList}>
            {session.items.map((entry, index) => {
              const decision = entry.decision;
              const classes = [
                styles.targetRow,
                decision.complete
                  ? styles.rowComplete
                  : decision.state !== null ||
                      decision.reviewer.trim() !== "" ||
                      decision.notes.length > 0 ||
                      decision.polished_spans.length > 0
                    ? styles.rowInProgress
                    : styles.rowUntouched,
                index === currentIndex ? styles.rowCurrent : "",
              ].join(" ");
              return (
                <button
                  key={entry.alignment_id}
                  type="button"
                  className={classes}
                  onClick={() => goTo(index)}
                  title={`Target ${index + 1} (${entry.window_id}): ${decision.state ?? "no state"} — ${decision.complete ? "complete" : "incomplete"}`}
                  aria-label={`Go to target ${index + 1}, ${decision.complete ? "complete" : "incomplete"}`}
                  aria-current={index === currentIndex ? "true" : undefined}
                >
                  <span className={styles.targetIndex}>{index + 1}</span>
                  <span className={styles.targetState}>
                    {decision.state ?? "—"}
                    {entry.bronze_target.correction_status === "TERMINAL_PUNCTUATION_DROPPED"
                      ? " ·"
                      : ""}
                  </span>
                  <span className={styles.targetWindow}>{entry.window_id}</span>
                </button>
              );
            })}
          </div>
        </section>

        <nav className={styles.deckControls} aria-label="Target navigation">
          <button
            type="button"
            className={styles.navButton}
            onClick={() => goTo(currentIndex - 1)}
            disabled={currentIndex === 0}
            aria-label="Previous target"
          >
            ← Prev
          </button>
          <span className={styles.deckPosition}>
            {item?.decision.complete ? "Complete" : "In progress"}
          </span>
          <button
            type="button"
            className={styles.navButton}
            onClick={() => goTo(currentIndex + 1)}
            disabled={currentIndex === session.items.length - 1}
            aria-label="Next target"
          >
            Next →
          </button>
        </nav>

        {item ? (
          <main className={styles.itemWorkspace} key={item.alignment_id}>
            <div className={styles.itemMeta}>
              <span className={styles.metaTag}>Window {item.window_id}</span>
              <span className={styles.metaTag}>{item.endpoint_id}</span>
              <span className={styles.metaTag}>
                Node {item.node_type ?? "null"}
              </span>
              <span
                className={[
                  styles.metaTag,
                  corrected ? styles.metaTagCorrected : "",
                ].join(" ")}
              >
                {corrected
                  ? `Terminal punctuation dropped (${item.bronze_target.dropped_text})`
                  : "Boundary unchanged"}
              </span>
            </div>

            <section className={styles.targetPanel} aria-label="Bronze target">
              <div className={styles.targetBlock}>
                <h2 className={styles.sectionLabel}>Original Bronze target</h2>
                <p className={styles.offsetLine}>
                  local {item.bronze_target.original_start}–{item.bronze_target.original_end}
                  {" · "}source {item.bronze_target.source_absolute_start}–
                  {item.bronze_target.source_absolute_end}
                </p>
                <p className={styles.targetText}>{item.bronze_target.original_text}</p>
              </div>
              <div className={styles.targetBlock}>
                <h2 className={styles.sectionLabel}>Evaluation target</h2>
                <p className={styles.offsetLine}>
                  local {item.bronze_target.evaluation_start}–{item.bronze_target.evaluation_end}
                </p>
                <p className={styles.targetText}>
                  {item.bronze_target.evaluation_text}
                  {corrected ? (
                    <span className={styles.droppedMark}>
                      {" "}
                      <s>{item.bronze_target.dropped_text}</s>
                    </span>
                  ) : null}
                </p>
                {corrected ? (
                  <p className={styles.correctionNote}>
                    Correction: the evaluation span drops exactly one terminal{" "}
                    {item.bronze_target.dropped_text === "." ? "period" : "comma"} from the raw
                    Bronze target (start unchanged, end −1, text without the terminal
                    punctuation).
                  </p>
                ) : null}
              </div>
            </section>

            <section className={styles.representationPanel} aria-label="Sealed representation">
              <h2 className={styles.sectionLabel}>Clean target transcript</h2>
              <p className={styles.transcriptText}>{item.representation.clean_target_transcript}</p>
            </section>

            <section className={styles.polishPanel} aria-label="Polished text alignment">
              <div className={styles.polishHeader}>
                <h2 className={styles.sectionLabel}>Polished text</h2>
                <span className={styles.polishHint}>Read-only — select an exact substring</span>
              </div>
              <textarea
                ref={polishedRef}
                className={styles.polishInput}
                value={item.representation.polished_text}
                readOnly
                spellCheck={false}
                onSelect={(event) => {
                  const target = event.currentTarget;
                  const start = target.selectionStart;
                  const end = target.selectionEnd;
                  setSelection(start !== end ? { start, end } : null);
                }}
                aria-label="Polished target transcript (read-only, select a span)"
              />
              <div className={styles.polishToolbar}>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={addSelectedSpan}
                  disabled={selectedSlice === null}
                  aria-label={
                    selectedSlice === null
                      ? "Add selected span (select text first)"
                      : `Add selected span ${JSON.stringify(selectedSlice)}`
                  }
                >
                  Add selected span
                </button>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={findExactTarget}
                >
                  Find exact target
                </button>
                <span className={styles.selectionPreview}>
                  {selectedSlice === null
                    ? "No selection"
                    : `Selected ${selection?.start}:${selection?.end} “${selectedSlice}”`}
                </span>
              </div>
              <div className={styles.manualRow} aria-label="Manual span fallback">
                <label className={styles.manualLabel}>
                  Start
                  <input
                    className={styles.manualInput}
                    type="number"
                    min={0}
                    value={manualStart}
                    onChange={(event) => setManualStart(event.target.value)}
                    aria-label="Manual span start offset"
                  />
                </label>
                <label className={styles.manualLabel}>
                  End
                  <input
                    className={styles.manualInput}
                    type="number"
                    min={1}
                    value={manualEnd}
                    onChange={(event) => setManualEnd(event.target.value)}
                    aria-label="Manual span end offset"
                  />
                </label>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={addManualSpan}
                >
                  Add at offsets
                </button>
                <span className={styles.manualHint}>
                  Must slice the exact polished text; the slice is validated on add.
                </span>
              </div>
              {exactTargets ? (
                <div className={styles.occurrenceChooser} aria-label="Exact occurrence choices">
                  <p className={styles.occurrenceNote}>
                    {exactTargets.length} exact occurrences of “{item.bronze_target.evaluation_text}”
                    — choose one:
                  </p>
                  {exactTargets.map((occurrence) => (
                    <button
                      key={`${occurrence.start}:${occurrence.end}`}
                      type="button"
                      className={styles.occurrenceButton}
                      onClick={() => addOccurrence(occurrence)}
                    >
                      {occurrence.start}:{occurrence.end} “{occurrence.text}”
                    </button>
                  ))}
                </div>
              ) : null}
            </section>

            <section className={styles.spansPanel} aria-label="Selected polished spans">
              <h2 className={styles.sectionLabel}>
                Selected spans ({item.decision.polished_spans.length})
              </h2>
              {item.decision.polished_spans.length === 0 ? (
                <p className={styles.noSpans}>No spans selected for this target yet.</p>
              ) : (
                <ul className={styles.spanList}>
                  {item.decision.polished_spans.map((span, index) => (
                    <li key={`${span.start}:${span.end}`} className={styles.spanRow}>
                      <span className={styles.spanOffsets}>
                        {span.start}–{span.end}
                      </span>
                      <span className={styles.spanText}>“{span.text}”</span>
                      <button
                        type="button"
                        className={styles.removeButton}
                        onClick={() => removeSpanAt(index)}
                        aria-label={`Remove span ${span.start} to ${span.end}`}
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className={styles.finishPanel} aria-label="Decision and completion">
              <div className={styles.stateBlock} role="radiogroup" aria-label="Alignment state">
                <h2 className={styles.sectionLabel}>Alignment state</h2>
                <div className={styles.stateButtons}>
                  {ALIGNMENT_DECISION_STATES.map((state) => (
                    <button
                      key={state}
                      type="button"
                      role="radio"
                      aria-checked={item.decision.state === state}
                      className={[
                        styles.stateButton,
                        item.decision.state === state ? styles.stateSelected : "",
                      ].join(" ")}
                      onClick={() =>
                        commitState(item.decision.state === state ? null : state)
                      }
                    >
                      {state}
                    </button>
                  ))}
                  <button
                    type="button"
                    className={styles.secondaryButton}
                    onClick={() => commitState(null)}
                    disabled={item.decision.state === null}
                  >
                    Clear state
                  </button>
                </div>
                <p className={styles.stateHint}>
                  {item.decision.state
                    ? `${item.decision.state}: ${STATE_HINTS[item.decision.state]}.`
                    : "Choose a state first; spans are added from the polished text."}
                </p>
              </div>
              <div className={styles.reviewerRow}>
                <label className={styles.fieldLabel} htmlFor="alignment-reviewer">
                  Reviewer name
                  <input
                    id="alignment-reviewer"
                    className={styles.fieldInput}
                    value={item.decision.reviewer}
                    onChange={(event) => setReviewer(event.target.value)}
                    placeholder="Your name"
                  />
                </label>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={applyReviewerToAll}
                  disabled={item.decision.reviewer.trim() === ""}
                >
                  Apply to all targets
                </button>
              </div>
              <label className={styles.fieldLabel} htmlFor="alignment-notes">
                Notes (one per line)
                <textarea
                  id="alignment-notes"
                  className={styles.noteInput}
                  value={item.decision.notes.join("\n")}
                  onChange={(event) => setNotes(event.target.value)}
                  placeholder="Optional notes for the finalizer."
                />
              </label>
              <div className={styles.completeRow}>
                <button
                  type="button"
                  className={styles.completeButton}
                  onClick={toggleComplete}
                  aria-pressed={item.decision.complete}
                >
                  {item.decision.complete ? "Mark incomplete" : "Mark target complete"}
                </button>
                <span className={styles.finishStatus}>
                  {item.decision.complete
                    ? `Completed ${item.decision.completed_at}`
                    : missingHint.length === 0
                      ? "Ready to complete"
                      : `Needs: ${missingHint.slice(0, 3).join("; ")}`}
                </span>
              </div>
              <p className={styles.finishNote}>
                completed_at is generated only when you explicitly mark the target complete. Any
                edit to a completed target retracts completion until you re-confirm it. Two
                endpoint IDs in the same window can never share the exact same span.
              </p>
            </section>
          </main>
        ) : null}

        {message ? (
          <p
            role={message.kind === "error" ? "alert" : "status"}
            className={message.kind === "error" ? styles.messageError : styles.messageInfo}
          >
            {message.text}
          </p>
        ) : null}

        <input
          ref={packetInputRef}
          type="file"
          accept="application/json,.json"
          className={styles.hiddenInput}
          onChange={handlePacketInput}
          tabIndex={-1}
          aria-label="Load a different blank Phase 2K downstream alignment packet JSON"
        />
        <input
          ref={sessionInputRef}
          type="file"
          accept="application/json,.json"
          className={styles.hiddenInput}
          onChange={handleSessionInputChange}
          tabIndex={-1}
          aria-label="Import a prior Phase 2K alignment session backup"
        />

        <footer className={styles.footer}>
          Scorer/model-blind alignment material only — no predictions, model results, or
          semantic extraction is loaded, displayed, or generated, and nothing here writes the
          canonical packet.
        </footer>
      </div>
    </div>
  );
}
