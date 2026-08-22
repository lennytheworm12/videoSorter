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
import styles from "../app/phase2k-review/review.module.css";
import {
  buildReviewsMap,
  buildSessionExport,
  buildSessionFromPacket,
  completeItem,
  NOT_APPLICABLE,
  reviewsMapErrors,
  sanitizePacket,
  SCORE_FIELDS,
  setAllReviewers,
  setItemNotes,
  setItemReviewer,
  setItemScore,
  summarizeProgress,
  uncompleteItem,
  validateSessionInput,
  type ReviewPacket,
  type ReviewSession,
  type ScoreField,
  type ScoreValue,
} from "../lib/phase2k-review";

const SESSION_PREFIX = "phase2k-review-session:v1";

type Message = { kind: "error" | "info"; text: string };

const FIELD_LABELS: Record<ScoreField, string> = {
  coached_actor: "Coached actor",
  opponent_entity: "Opponent / entity",
  pronouns: "Pronouns",
  ability_ownership: "Ability ownership",
  core_action: "Core action",
  condition: "Condition",
  consequence: "Consequence",
  causality: "Causality",
  standalone_coaching_claim: "Standalone coaching claim",
  asr_repair_correctness: "ASR repair correctness",
  entity_binding_correctness: "Entity binding correctness",
  meaning_preservation: "Meaning preservation",
  unsupported_invention: "Unsupported invention",
  remaining_ambiguity: "Remaining ambiguity",
};

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the file."));
    reader.readAsText(file);
  });
}

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

export function Phase2KReviewClient() {
  const [packet, setPacket] = useState<ReviewPacket | null>(null);
  const [session, setSession] = useState<ReviewSession | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [message, setMessage] = useState<Message | null>(null);
  const [revealed, setRevealed] = useState<{ itemId: string; field: ScoreField } | null>(null);

  const packetInputRef = useRef<HTMLInputElement>(null);
  const sessionInputRef = useRef<HTMLInputElement>(null);
  const sessionRef = useRef<ReviewSession | null>(null);
  const currentIndexRef = useRef(0);
  const packetRef = useRef<ReviewPacket | null>(null);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  useEffect(() => {
    packetRef.current = packet;
  }, [packet]);

  const storageKey = useMemo(
    () => (packet ? `${SESSION_PREFIX}:${packet.content_sha256}` : null),
    [packet],
  );

  const item = session?.items[currentIndex] ?? null;
  const progress = useMemo(() => (session ? summarizeProgress(session) : null), [session]);

  // Restore the autosave bound to the exact packet hash after a packet load.
  useEffect(() => {
    if (!packet || !storageKey) {
      return;
    }
    let restored: ReviewSession | null = null;
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

  const handlePacketFile = useCallback(async (file: File) => {
    try {
      const text = await readFileAsText(file);
      const parsed: unknown = JSON.parse(text);
      const sanitized = sanitizePacket(parsed);
      setPacket(sanitized);
      setHydrated(false);
      setMessage({
        kind: "info",
        text: `Blank packet accepted: ${sanitized.review_items.length} blinded items bound to ${sanitized.content_sha256.slice(0, 12)}….`,
      });
    } catch (error) {
      setMessage({
        kind: "error",
        text: `Packet rejected: ${error instanceof Error ? error.message : "invalid JSON"}`,
      });
    }
  }, []);

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
      if (!packetRef.current) {
        return;
      }
      try {
        const text = await readFileAsText(file);
        const parsed: unknown = JSON.parse(text);
        const result = validateSessionInput(parsed, packetRef.current);
        if (!result.ok) {
          setMessage({
            kind: "error",
            text: `Backup rejected: ${result.errors.slice(0, 2).join(" ")}`,
          });
          return;
        }
        setSession(result.session);
        setCurrentIndex(0);
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
    setMessage(null);
  }, []);

  const commitScore = useCallback((field: ScoreField, value: ScoreValue | null) => {
    const currentPacket = packetRef.current;
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentPacket || !currentSession || !currentItem) {
      return;
    }
    try {
      const next = setItemScore(
        currentSession,
        currentPacket.rubric,
        currentItem.review_item_id,
        field,
        value,
      );
      setSession(next);
      if (value !== null) {
        setRevealed({ itemId: currentItem.review_item_id, field });
        window.setTimeout(() => setRevealed(null), 700);
      }
    } catch (error) {
      setMessage({
        kind: "error",
        text: error instanceof Error ? error.message : "Could not set that score.",
      });
    }
  }, []);

  const handleScoreRowKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (/^[0-5]$/.test(event.key)) {
        event.preventDefault();
        const field = (event.currentTarget.dataset.field ?? "") as ScoreField;
        commitScore(field, Number(event.key));
      }
    },
    [commitScore],
  );

  const setNotes = useCallback((text: string) => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    const lines = text.split("\n").filter((line) => line !== "");
    setSession(
      setItemNotes(
        currentSession,
        currentItem.review_item_id,
        lines,
      ),
    );
  }, []);

  const setReviewer = useCallback((reviewer: string) => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    setSession(setItemReviewer(currentSession, currentItem.review_item_id, reviewer));
  }, []);

  const applyReviewerToAll = useCallback(() => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem || currentItem.reviewer.trim() === "") {
      setMessage({
        kind: "error",
        text: "Enter a reviewer name first, then apply it to every item.",
      });
      return;
    }
    setSession(setAllReviewers(currentSession, currentItem.reviewer));
    setMessage({
      kind: "info",
      text: `Reviewer “${currentItem.reviewer}” applied to all ${currentSession.items.length} items.`,
    });
  }, []);

  const toggleComplete = useCallback(() => {
    const currentSession = sessionRef.current;
    const currentItem = currentSession?.items[currentIndexRef.current];
    if (!currentSession || !currentItem) {
      return;
    }
    if (currentItem.complete) {
      setSession(uncompleteItem(currentSession, currentItem.review_item_id));
      setMessage({
        kind: "info",
        text: "Completion retracted; the generated timestamp was cleared.",
      });
      return;
    }
    const result = completeItem(
      currentSession,
      currentItem.review_item_id,
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
    setMessage({ kind: "info", text: "Item marked complete." });
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
      `phase2k-review-session-${currentPacket.content_sha256.slice(0, 8)}.json`,
    );
    setMessage({
      kind: "info",
      text: "Review session exported. This is review material, not final gold.",
    });
  }, []);

  const exportReviewsMap = useCallback(() => {
    const currentPacket = packetRef.current;
    const currentSession = sessionRef.current;
    if (!currentPacket || !currentSession) {
      return;
    }
    const errors = reviewsMapErrors(currentSession, currentPacket.rubric);
    if (errors.length > 0) {
      setMessage({
        kind: "error",
        text: `Final export blocked — ${errors.length} problem(s): ${errors.slice(0, 2).join(" ")}`,
      });
      return;
    }
    const map = buildReviewsMap(currentSession, currentPacket.rubric);
    downloadJson(
      map,
      `phase2k-human-reviews-${currentPacket.content_sha256.slice(0, 8)}.json`,
    );
    setMessage({
      kind: "info",
      text: "Final reviews map exported for the finalizer.",
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
              <h1 className={styles.title}>Phase 2K · Blinded semantic-recoverability review</h1>
              <p className={styles.subtitle}>
                Score each blinded presentation for how well the coached action and its meaning
                survive in the presented text. Conditions, radii, models, and the label mapping
                never reach this workspace.
              </p>
            </div>
          </header>
          <section className={styles.loadPanel}>
            <h2>Load the blank review packet</h2>
            <p>
              Select the official <code>phase2k-human-review-packet-v2.json</code> from this
              machine. The packet is validated strictly in your browser, kept locally, and never
              uploaded. The separate mapping file is never read.
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
              aria-label="Load the blank Phase 2K human-review packet JSON"
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
            Review material only — scores are never fabricated and nothing is written upstream.
          </footer>
        </div>
      </div>
    );
  }

  const scoredCount = item
    ? SCORE_FIELDS.filter((field) => item.scores[field] !== null).length
    : 0;
  const missingHint = item ? SCORE_FIELDS.filter((field) => item.scores[field] === null) : [];

  return (
    <div className={styles.container}>
      <div className={styles.inner}>
        <header className={styles.topbar}>
          <Link href="/" className={styles.backLink} aria-label="Back to query home">
            ← Home
          </Link>
          <div className={styles.titleBlock}>
            <h1 className={styles.title}>Phase 2K · Blinded review</h1>
            <p className={styles.subtitle}>
              Score semantic recoverability from the presented text alone. The packet is bound to
              {` ${packet.content_sha256.slice(0, 12)}… `}
              and autosaves locally.
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
              onClick={exportReviewsMap}
              disabled={progress?.complete !== progress?.total}
            >
              Export reviews map
            </button>
          </div>
        </header>

        <section className={styles.progressBlock} aria-label="Session progress">
          <div className={styles.progressText}>
            <span>
              Item {currentIndex + 1} of {session.items.length}
            </span>
            <span>
              {progress ? `${progress.complete} complete · ${progress.ready} ready` : ""}
            </span>
          </div>
          <div
            className={styles.progressTrack}
            role="progressbar"
            aria-label="Review completion progress"
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
          <div className={styles.overview} role="group" aria-label="Item completion overview">
            {session.items.map((entry, index) => {
              const classes = [
                styles.overviewCell,
                entry.complete
                  ? styles.cellComplete
                  : entry.reviewer.trim() !== "" ||
                      entry.notes.length > 0 ||
                      SCORE_FIELDS.some((field) => entry.scores[field] !== null)
                    ? styles.cellInProgress
                    : styles.cellUntouched,
                index === currentIndex ? styles.overviewCurrent : "",
              ].join(" ");
              return (
                <button
                  key={entry.review_item_id}
                  type="button"
                  className={classes}
                  onClick={() => goTo(index)}
                  title={`Item ${index + 1}: ${entry.complete ? "complete" : "not complete"}`}
                  aria-label={`Go to item ${index + 1}, ${entry.complete ? "complete" : "incomplete"}`}
                  aria-current={index === currentIndex ? "true" : undefined}
                >
                  {index + 1}
                </button>
              );
            })}
          </div>
        </section>

        <nav className={styles.deckControls} aria-label="Item navigation">
          <button
            type="button"
            className={styles.navButton}
            onClick={() => goTo(currentIndex - 1)}
            disabled={currentIndex === 0}
            aria-label="Previous item"
          >
            ← Prev
          </button>
          <span className={styles.deckPosition}>
            {item?.complete ? "Complete" : "In progress"}
          </span>
          <button
            type="button"
            className={styles.navButton}
            onClick={() => goTo(currentIndex + 1)}
            disabled={currentIndex === session.items.length - 1}
            aria-label="Next item"
          >
            Next →
          </button>
        </nav>

        {item ? (
          <main className={styles.itemWorkspace} key={item.review_item_id}>
            <div className={styles.itemMeta}>
              <span className={styles.metaTag}>Window {item.window_id}</span>
              <span className={styles.metaTag}>Item {item.blinded_label}</span>
              <span className={styles.metaTag}>
                {scoredCount} / {SCORE_FIELDS.length} scored
              </span>
            </div>

            <section className={styles.presentationPanel} aria-label="Review presentation">
              {item.presentation.sections.map((section) => (
                <div key={section.id} className={styles.sectionBlock}>
                  <h2 className={styles.sectionLabel}>
                    {section.id === "primary" ? "Primary text" : "Supplement"}
                  </h2>
                  <p className={styles.sectionText}>{section.text}</p>
                </div>
              ))}
            </section>

            <section className={styles.scoreSection} aria-label="Score fields">
              <h2 className={styles.scoreHeading}>Semantic recoverability scores</h2>
              {SCORE_FIELDS.map((field) => {
                const entry = packet.rubric[field];
                const value = item.scores[field];
                const isRevealed = revealed?.itemId === item.review_item_id && revealed.field === field;
                return (
                  <div
                    key={field}
                    className={styles.scoreRow}
                    data-field={field}
                    onKeyDown={handleScoreRowKeyDown}
                    role="radiogroup"
                    aria-label={`${FIELD_LABELS[field]} score`}
                  >
                    <div className={styles.scoreInfo}>
                      <span className={styles.scoreName}>{FIELD_LABELS[field]}</span>
                      <span className={styles.scoreDesc}>{entry.description}</span>
                      {entry.direction === "lower_is_better" ? (
                        <span className={styles.scoreHint}>lower is better</span>
                      ) : null}
                    </div>
                    <div className={styles.scoreButtons}>
                      {[0, 1, 2, 3, 4, 5].map((score) => {
                        const selected = value === score;
                        const classes = [
                          styles.scoreButton,
                          selected ? styles.scoreSelected : "",
                          isRevealed && value === score ? styles.scoreReveal : "",
                        ].join(" ");
                        return (
                          <button
                            key={score}
                            type="button"
                            role="radio"
                            aria-checked={selected}
                            className={classes}
                            onClick={() => commitScore(field, selected ? null : score)}
                          >
                            {score}
                          </button>
                        );
                      })}
                      {entry.not_applicable_allowed ? (
                        <button
                          type="button"
                          role="radio"
                          aria-checked={value === NOT_APPLICABLE}
                          className={`${styles.naButton} ${
                            value === NOT_APPLICABLE ? styles.scoreSelected : ""
                          }`}
                          onClick={() =>
                            commitScore(
                              field,
                              value === NOT_APPLICABLE ? null : NOT_APPLICABLE,
                            )
                          }
                        >
                          N/A
                        </button>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </section>

            <section className={styles.finishPanel} aria-label="Item completion">
              <div className={styles.reviewerRow}>
                <label className={styles.fieldLabel} htmlFor="reviewer-name">
                  Reviewer name
                  <input
                    id="reviewer-name"
                    className={styles.fieldInput}
                    value={item.reviewer}
                    onChange={(event) => setReviewer(event.target.value)}
                    placeholder="Your name"
                  />
                </label>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={applyReviewerToAll}
                  disabled={item.reviewer.trim() === ""}
                >
                  Apply to all items
                </button>
              </div>
              <label className={styles.fieldLabel} htmlFor="item-notes">
                Notes (one per line)
                <textarea
                  id="item-notes"
                  className={styles.noteInput}
                  value={item.notes.join("\n")}
                  onChange={(event) => setNotes(event.target.value)}
                  placeholder="Optional notes for the finalizer."
                />
              </label>
              <div className={styles.completeRow}>
                <button
                  type="button"
                  className={styles.completeButton}
                  onClick={toggleComplete}
                  aria-pressed={item.complete}
                >
                  {item.complete ? "Mark incomplete" : "Mark item complete"}
                </button>
                <span className={styles.finishStatus}>
                  {item.complete
                    ? `Completed ${item.completed_at}`
                    : missingHint.length === 0 && item.reviewer.trim() !== ""
                      ? "Ready to complete"
                      : `${missingHint.length} score(s) and a reviewer still needed`}
                </span>
              </div>
              <p className={styles.finishNote}>
                completed_at is generated only when you explicitly mark the item complete. A
                score or reviewer change retracts completion until you re-confirm it.
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
          aria-label="Load a different blank Phase 2K human-review packet JSON"
        />
        <input
          ref={sessionInputRef}
          type="file"
          accept="application/json,.json"
          className={styles.hiddenInput}
          onChange={handleSessionInputChange}
          tabIndex={-1}
          aria-label="Import a prior Phase 2K review session backup"
        />

        <footer className={styles.footer}>
          Blinded review material only — condition/radius/mapping/model data is never loaded or
          shown, and nothing here writes the canonical packet.
        </footer>
      </div>
    </div>
  );
}
