"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import Link from "next/link";
import styles from "../app/phase2k-audit/audit.module.css";
import {
  AUDIT_DECISIONS,
  AUDIT_CATEGORIES,
  buildCompletedAudit,
  buildSession,
  bytesToHex,
  flattenOperations,
  isOperationComplete,
  setSessionAttestation,
  setSessionCorrection,
  setSessionDecision,
  setSessionTaxonomy,
  STATEMENT_ATTESTATION_FIELDS,
  summarizeProgress,
  validateBlankTemplate,
  validateSessionInput,
  verifyTemplateContentHash,
  type AuditCategory,
  type AuditDecision,
  type AuditSession,
  type AuditTemplate,
  type BindingOperation,
  type RepairOperation,
  type Sha256Digest,
  type StatementOperation,
  type StatementAttestationField,
} from "../lib/phase2k-audit";

const DECISION_LABEL: Record<AuditDecision, string> = {
  APPROVE: "Approve",
  REJECT: "Reject",
  AMBIGUOUS: "Ambiguous",
};

const CATEGORY_LABEL: Record<AuditCategory, string> = {
  mechanical_repairs: "Mechanical repairs",
  contextual_repairs: "Contextual repairs",
  entity_bindings: "Entity bindings",
  pronoun_bindings: "Pronoun bindings",
  reference_bindings: "Reference bindings",
  ability_bindings: "Ability bindings",
  polished_statements: "Polished statements",
};

const ATTESTATION_LABEL: Record<StatementAttestationField, string> = {
  supported: "Supported by source evidence",
  uncertainty_preserved: "Uncertainty preserved",
  negation_preserved: "Negation preserved",
  modality_preserved: "Modality preserved",
  causality_invented: "Causality invented",
  source_detail_dropped: "Source detail dropped",
};

type Message = { kind: "error" | "info"; text: string };

function spanSummary(span: unknown): { text: string; meta: string } {
  if (span === null || typeof span !== "object" || Array.isArray(span)) {
    return { text: "", meta: "malformed evidence span" };
  }
  const record = span as Record<string, unknown>;
  const parts: string[] = [];
  if (typeof record.segment_id === "string" && record.segment_id.length > 0) {
    parts.push(`segment ${record.segment_id}`);
  }
  const sourceStart = typeof record.source_absolute_start === "number" ? record.source_absolute_start : null;
  const sourceEnd = typeof record.source_absolute_end === "number" ? record.source_absolute_end : null;
  if (sourceStart !== null && sourceEnd !== null) {
    parts.push(`source ${sourceStart}–${sourceEnd}`);
  }
  const localStart = typeof record.target_local_start === "number" ? record.target_local_start : null;
  const localEnd = typeof record.target_local_end === "number" ? record.target_local_end : null;
  if (localStart !== null && localEnd !== null) {
    parts.push(`target ${localStart}–${localEnd}`);
  }
  return {
    text: typeof record.text === "string" ? record.text : "",
    meta: parts.join(" · "),
  };
}

export function Phase2KAuditClient() {
  const [template, setTemplate] = useState<AuditTemplate | null>(null);
  const [session, setSession] = useState<AuditSession | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [message, setMessage] = useState<Message | null>(null);
  const [completedHash, setCompletedHash] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const templateInputRef = useRef<HTMLInputElement>(null);
  const sessionInputRef = useRef<HTMLInputElement>(null);

  const flattened = useMemo(
    () => (template === null ? [] : flattenOperations(template)),
    [template],
  );
  const progress = useMemo(
    () =>
      template === null || session === null
        ? null
        : summarizeProgress(template, session),
    [template, session],
  );
  const current = flattened[currentIndex] ?? null;
  const currentSessionOperation =
    session === null || current === null
      ? null
      : session.operations.find(
          (operation) => operation.operation_id === current.operation_id,
        ) ?? null;
  const repairOperation =
    current !== null &&
    (current.category === "mechanical_repairs" ||
      current.category === "contextual_repairs")
      ? (current.operation as RepairOperation)
      : null;
  const bindingOperation =
    current !== null && current.category.endsWith("bindings")
      ? (current.operation as BindingOperation)
      : null;
  const statementOperation =
    current !== null && current.category === "polished_statements"
      ? (current.operation as StatementOperation)
      : null;

  const webDigest = useCallback<Sha256Digest>(async (bytes) => {
    const subtle = globalThis.crypto?.subtle;
    if (!subtle) {
      throw new Error("Web Crypto is unavailable in this browser");
    }
    return bytesToHex(
      await subtle.digest("SHA-256", bytes as unknown as BufferSource),
    );
  }, []);

  const sessionStorageKey = useCallback(
    (templateValue: AuditTemplate) =>
      `phase2k-audit-session:v1:${templateValue.content_sha256}`,
    [],
  );

  const handleTemplateFile = useCallback(
    async (file: File) => {
      try {
        const text = await file.text();
        const parsed: unknown = JSON.parse(text);
        const result = validateBlankTemplate(parsed);
        if (!result.ok) {
          setMessage({
            kind: "error",
            text: `Template rejected: ${result.errors.slice(0, 3).join(" ")}`,
          });
          return;
        }
        if (!(await verifyTemplateContentHash(result.template, webDigest))) {
          setMessage({
            kind: "error",
            text: "Template rejected: content_sha256 does not match the canonical content.",
          });
          return;
        }
        const nextTemplate = result.template;
        setTemplate(nextTemplate);
        setCompletedHash(null);
        setCurrentIndex(0);
        let restored: AuditSession | null = null;
        let restoreError: string | null = null;
        try {
          const raw = window.localStorage.getItem(sessionStorageKey(nextTemplate));
          if (raw !== null) {
            const parsedSession: unknown = JSON.parse(raw);
            const sessionResult = validateSessionInput(parsedSession, nextTemplate);
            if (sessionResult.ok) {
              restored = sessionResult.session;
              setMessage({
                kind: "info",
                text: "Restored your saved review for this template from this browser.",
              });
            } else {
              restoreError = `Saved review for this template was ignored: ${sessionResult.errors[0]}`;
            }
          }
        } catch {
          // Corrupt autosave; start a fresh session below.
        }
        if (restored === null) {
          setSession(buildSession(nextTemplate));
          setMessage(
            restoreError === null
              ? { kind: "info", text: "Blank template validated and loaded." }
              : { kind: "error", text: restoreError },
          );
        } else {
          setSession(restored);
        }
      } catch {
        setMessage({
          kind: "error",
          text: "Template rejected: the file is not valid JSON.",
        });
      }
    },
    [sessionStorageKey, webDigest],
  );

  const handleTemplateChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void handleTemplateFile(file);
      }
      event.target.value = "";
    },
    [handleTemplateFile],
  );

  // Autosave bound to the template content hash + records hash.
  useEffect(() => {
    if (template === null || session === null || !hydrated) {
      return;
    }
    try {
      window.localStorage.setItem(
        sessionStorageKey(template),
        JSON.stringify(session),
      );
    } catch {
      // Private mode / quota; review still works in memory.
    }
  }, [template, session, hydrated, sessionStorageKey]);

  useEffect(() => {
    setHydrated(true);
  }, []);

  const goTo = useCallback(
    (index: number) => {
      setCurrentIndex(Math.max(0, Math.min(flattened.length - 1, index)));
    },
    [flattened.length],
  );

  const jumpToWindow = useCallback(
    (windowId: string) => {
      const index = flattened.findIndex((item) => item.window_id === windowId);
      if (index >= 0) {
        setCurrentIndex(index);
      }
    },
    [flattened],
  );

  const jumpToCategory = useCallback(
    (windowId: string, category: AuditCategory) => {
      const index = flattened.findIndex(
        (item) => item.window_id === windowId && item.category === category,
      );
      if (index >= 0) {
        setCurrentIndex(index);
      }
    },
    [flattened],
  );

  const handleSessionImport = useCallback(
    async (file: File) => {
      if (template === null) {
        return;
      }
      try {
        const text = await file.text();
        const parsed: unknown = JSON.parse(text);
        const result = validateSessionInput(parsed, template);
        if (!result.ok) {
          setMessage({
            kind: "error",
            text: `Session import rejected: ${result.errors.slice(0, 3).join(" ")}`,
          });
          return;
        }
        setSession(result.session);
        setCurrentIndex(0);
        setCompletedHash(null);
        setMessage({ kind: "info", text: "Session imported and replacing the local review." });
      } catch {
        setMessage({ kind: "error", text: "Session import rejected: the file is not valid JSON." });
      }
    },
    [template],
  );

  const handleSessionChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void handleSessionImport(file);
      }
      event.target.value = "";
    },
    [handleSessionImport],
  );

  const downloadJson = useCallback((filename: string, value: unknown) => {
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
  }, []);

  const handleSessionExport = useCallback(() => {
    if (template === null || session === null) {
      return;
    }
    downloadJson(
      `phase2k-audit-session-${template.content_sha256.slice(0, 8)}.json`,
      session,
    );
    setMessage({
      kind: "info",
      text: "Review session exported. This is a work-in-progress backup, not the completed audit.",
    });
  }, [template, session, downloadJson]);

  const handleCompletedExport = useCallback(async () => {
    if (template === null || session === null || progress === null) {
      return;
    }
    if (progress.remaining > 0) {
      setMessage({
        kind: "error",
        text: `Completed audit export requires every operation; ${progress.remaining} remain.`,
      });
      return;
    }
    const result = await buildCompletedAudit(template, session, webDigest);
    if (!result.ok) {
      setMessage({
        kind: "error",
        text: `Export blocked: ${result.errors.slice(0, 3).join(" ")}`,
      });
      return;
    }
    downloadJson(
      `phase2k-completed-transformation-audit-${template.content_sha256.slice(0, 8)}.json`,
      result.completed,
    );
    setCompletedHash(result.completed.content_sha256);
    setMessage({
      kind: "info",
      text: "Completed transformation audit exported for scripts/finalize_phase2k_human_review.py.",
    });
  }, [template, session, progress, webDigest, downloadJson]);

  const windows = useMemo(() => {
    const seen: string[] = [];
    for (const item of flattened) {
      if (!seen.includes(item.window_id)) {
        seen.push(item.window_id);
      }
    }
    return seen;
  }, [flattened]);

  if (template === null || session === null) {
    return (
      <div className={styles.container}>
        <div className={styles.inner}>
          <header className={styles.topbar}>
            <div className={styles.titleBlock}>
              <p className={styles.eyebrow}>Phase 2K · Live outputs</p>
              <h1 className={styles.title}>Transformation audit review</h1>
              <p className={styles.subtitle}>
                Operation-level, downstream-result-blind review of the live
                transformation audit. Load the blank template packet from your
                local file; it is validated strictly and never fetched from a
                server path.
              </p>
            </div>
          </header>
          {message !== null ? (
            <p
              className={
                message.kind === "error" ? styles.errorBanner : styles.infoBanner
              }
              role="status"
            >
              {message.text}
            </p>
          ) : null}
          <section className={styles.landing}>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={() => templateInputRef.current?.click()}
            >
              Load blank transformation-audit template
            </button>
            <p className={styles.landingNote}>
              Expects{" "}
              <code>phase2k-transformation-audit-packet-v2</code> with release
              gate <code>AWAITING_HUMAN_REVIEW</code>. Completed or malformed
              files are rejected.
            </p>
          </section>
          <input
            ref={templateInputRef}
            type="file"
            accept="application/json,.json"
            className={styles.hiddenInput}
            onChange={handleTemplateChange}
            tabIndex={-1}
            aria-label="Load a blank Phase 2K transformation-audit template"
          />
          <footer className={styles.footer}>
            <Link href="/" className={styles.backLink}>
              Home
            </Link>
            <span>No downstream results are shown; nothing is fabricated.</span>
          </footer>
        </div>
      </div>
    );
  }

  const sessionOperationById = new Map(
    session.operations.map((operation) => [operation.operation_id, operation]),
  );
  const currentSessionOp = current
    ? (sessionOperationById.get(current.operation_id) ?? null)
    : null;

  return (
    <div className={styles.container}>
      <div className={styles.inner}>
        <header className={styles.topbar}>
          <div className={styles.titleBlock}>
            <p className={styles.eyebrow}>Phase 2K · Live outputs</p>
            <h1 className={styles.title}>Transformation audit review</h1>
            <p className={styles.bindingLine}>
              Template{" "}
              <code>{template.content_sha256.slice(0, 16)}…</code> · records{" "}
              <code>{template.binding.records_sha256.slice(0, 16)}…</code>
            </p>
          </div>
          <div className={styles.actions}>
            <button type="button" className={styles.actionButton} onClick={handleSessionExport}>
              Export session
            </button>
            <button
              type="button"
              className={styles.actionButton}
              onClick={() => sessionInputRef.current?.click()}
            >
              Import session
            </button>
            <button
              type="button"
              className={styles.exportButton}
              onClick={() => void handleCompletedExport()}
              disabled={progress === null || progress.remaining > 0}
            >
              Export completed audit
            </button>
          </div>
        </header>

        {message !== null ? (
          <p
            className={
              message.kind === "error" ? styles.errorBanner : styles.infoBanner
            }
            role="status"
          >
            {message.text}
          </p>
        ) : null}

        {progress !== null ? (
          <section className={styles.progressBlock} aria-label="Review progress">
            <div className={styles.progressText}>
              <span>
                {progress.completed} / {progress.total} operations complete
              </span>
              <span>{progress.remaining} remaining</span>
            </div>
            <div className={styles.progressTrack}>
              <div
                className={styles.progressFill}
                style={{ width: `${progress.total === 0 ? 0 : (progress.completed / progress.total) * 100}%` }}
              />
            </div>
          </section>
        ) : null}

        <nav className={styles.navSection} aria-label="Window navigation">
          <div className={styles.navLabel}>Windows</div>
          <div className={styles.chipRow}>
            {windows.map((windowId) => (
              <button
                key={windowId}
                type="button"
                className={`${styles.chip} ${
                  current?.window_id === windowId ? styles.chipActive : ""
                }`}
                onClick={() => jumpToWindow(windowId)}
              >
                {windowId}
              </button>
            ))}
          </div>
          {current !== null ? (
            <>
              <div className={styles.navLabel}>Categories in window</div>
              <div className={styles.chipRow}>
                {AUDIT_CATEGORIES.filter(
                  (category) =>
                    current.window.operations[category].length > 0,
                ).map((category) => {
                  const counts = progress?.by_category[category];
                  return (
                    <button
                      key={category}
                      type="button"
                      className={`${styles.chip} ${
                        current.category === category ? styles.chipActive : ""
                      }`}
                      onClick={() => jumpToCategory(current.window_id, category)}
                    >
                      {CATEGORY_LABEL[category]}
                      {counts !== undefined
                        ? ` ${counts.completed}/${counts.total}`
                        : ""}
                    </button>
                  );
                })}
              </div>
            </>
          ) : null}
        </nav>

        {current !== null && currentSessionOp !== null ? (
          <main className={styles.stage}>
            <header className={styles.stageHeader}>
              <div>
                <p className={styles.stageEyebrow}>
                  Operation {current.ordinal + 1} of {flattened.length}
                </p>
                <h2 className={styles.stageTitle}>
                  {current.window_id} · {CATEGORY_LABEL[current.category]}
                </h2>
              </div>
              <div className={styles.stageNav}>
                <button
                  type="button"
                  className={styles.navButton}
                  onClick={() => goTo(currentIndex - 1)}
                  disabled={currentIndex === 0}
                >
                  Previous
                </button>
                <button
                  type="button"
                  className={styles.navButton}
                  onClick={() => goTo(currentIndex + 1)}
                  disabled={currentIndex >= flattened.length - 1}
                >
                  Next
                </button>
              </div>
            </header>

            <section className={styles.evidenceSection}>
              <div className={styles.sectionLabel}>Bronze target</div>
              <blockquote className={styles.bronzeQuote}>
                {current.window.bronze_target.text}
              </blockquote>
              <p className={styles.metaLine}>
                sha256 {current.window.bronze_target.text_sha256.slice(0, 16)}…
                · source{" "}
                {current.window.bronze_target.source_absolute_start}–
                {current.window.bronze_target.source_absolute_end}
              </p>
            </section>

            <section className={styles.proposalSection}>
              <div className={styles.sectionLabel}>Operation proposal</div>
              {current.category === "mechanical_repairs" ||
              current.category === "contextual_repairs" ? (
                <div className={styles.proposalGrid}>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Type</span>
                    <span className={styles.fieldValue}>
                      {repairOperation?.repair_type}
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Confidence</span>
                    <span className={styles.fieldValue}>
                      {repairOperation?.confidence}
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Original</span>
                    <span className={styles.fieldValue}>
                      “{repairOperation?.original_text}”
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Replacement</span>
                    <span className={styles.fieldValue}>
                      “{repairOperation?.replacement}”
                    </span>
                  </div>
                </div>
              ) : current.category.endsWith("bindings") ? (
                <div className={styles.proposalGrid}>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Slot</span>
                    <span className={styles.fieldValue}>
                      {bindingOperation?.slot}
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Mention</span>
                    <span className={styles.fieldValue}>
                      “{bindingOperation?.mention.text}” (target{" "}
                      {bindingOperation?.mention.target_local_start}–
                      {bindingOperation?.mention.target_local_end}
                      , source{" "}
                      {bindingOperation?.mention.source_absolute_start}–
                      {bindingOperation?.mention.source_absolute_end})
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Resolved candidate</span>
                    <span className={styles.fieldValue}>
                      “{bindingOperation?.resolved_candidate}”
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Status</span>
                    <span className={styles.fieldValue}>
                      {bindingOperation?.resolved_status}
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Human resolution</span>
                    <span className={styles.fieldValue}>
                      {bindingOperation?.human_resolvable_required
                        ? "required"
                        : "not required"}
                    </span>
                  </div>
                </div>
              ) : (
                <div className={styles.proposalGrid}>
                  <div className={styles.fieldWide}>
                    <span className={styles.fieldLabel}>Statement</span>
                    <span className={styles.statementText}>
                      {statementOperation?.text}
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Support mode</span>
                    <span className={styles.fieldValue}>
                      {statementOperation?.support_mode}
                    </span>
                  </div>
                  <div className={styles.field}>
                    <span className={styles.fieldLabel}>Reconstruction ops</span>
                    <span className={styles.fieldValue}>
                      {statementOperation &&
                      statementOperation.reconstruction_operation_ids.length > 0
                        ? statementOperation.reconstruction_operation_ids.join(", ")
                        : "—"}
                    </span>
                  </div>
                  {statementOperation?.unchanged_source_quote !== null &&
                  statementOperation !== null ? (
                    <div className={styles.fieldWide}>
                      <span className={styles.fieldLabel}>
                        Unchanged source quote
                      </span>
                      <span className={styles.fieldValue}>
                        “{statementOperation.unchanged_source_quote?.text}” (
                        target{" "}
                        {statementOperation.unchanged_source_quote?.target_local_start}–
                        {statementOperation.unchanged_source_quote?.target_local_end}
                        , source{" "}
                        {statementOperation.unchanged_source_quote?.source_absolute_start}–
                        {statementOperation.unchanged_source_quote?.source_absolute_end})
                      </span>
                    </div>
                  ) : null}
                </div>
              )}

              <div className={styles.sectionLabel}>Exact evidence spans</div>
              {(repairOperation?.evidence_spans ??
                bindingOperation?.evidence_spans ??
                statementOperation?.evidence_spans ?? []).length === 0 ? (
                <p className={styles.metaLine}>None.</p>
              ) : (
                <ul className={styles.spanList}>
                  {(repairOperation?.evidence_spans ??
                    bindingOperation?.evidence_spans ??
                    statementOperation?.evidence_spans ?? []).map((span, index) => {
                    const summary = spanSummary(span);
                    return (
                      <li key={index} className={styles.spanItem}>
                        {summary.text.length > 0 ? (
                          <span className={styles.spanText}>
                            “{summary.text}”
                          </span>
                        ) : null}
                        {summary.meta.length > 0 ? (
                          <span className={styles.spanMeta}>{summary.meta}</span>
                        ) : null}
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>

            {(current.window.first_failure !== null ||
              current.window.first_reconstruction_failure !== null) && (
              <section className={styles.failureSection}>
                <div className={styles.sectionLabel}>Window pipeline failures</div>
                {current.window.first_failure !== null ? (
                  <p className={styles.metaLine}>
                    first failure · {current.window.first_failure.stage} ·{" "}
                    {current.window.first_failure.error}
                  </p>
                ) : null}
                {current.window.first_reconstruction_failure !== null ? (
                  <p className={styles.metaLine}>
                    first reconstruction failure ·{" "}
                    {current.window.first_reconstruction_failure.error}
                  </p>
                ) : null}
              </section>
            )}

            <section className={styles.decisionSection}>
              <div className={styles.sectionLabel}>Human decision</div>
              <div className={styles.decisionRow} role="group" aria-label="Operation decision">
                {AUDIT_DECISIONS.map((decision) => (
                  <button
                    key={decision}
                    type="button"
                    className={`${styles.decisionButton} ${
                      currentSessionOp.decision === decision
                        ? styles.decisionActive
                        : ""
                    }`}
                    aria-pressed={currentSessionOp.decision === decision}
                    onClick={() =>
                      setSession(
                        setSessionDecision(session, current.operation_id, decision),
                      )
                    }
                  >
                    {DECISION_LABEL[decision]}
                  </button>
                ))}
              </div>

              {currentSessionOp.decision === "REJECT" ? (
                <div className={styles.rejectPanel}>
                  <div className={styles.field}>
                    <label className={styles.fieldLabel} htmlFor="audit-taxonomy">
                      Error taxonomy (required for REJECT)
                    </label>
                    <select
                      id="audit-taxonomy"
                      className={styles.selectInput}
                      value={currentSessionOp.error_taxonomy ?? ""}
                      onChange={(event) =>
                        setSession(
                          setSessionTaxonomy(
                            session,
                            current.operation_id,
                            event.target.value === "" ? null : event.target.value,
                          ),
                        )
                      }
                    >
                      <option value="">Select taxonomy…</option>
                      {template.error_taxonomy.map((taxonomy) => (
                        <option key={taxonomy} value={taxonomy}>
                          {taxonomy}
                        </option>
                      ))}
                    </select>
                  </div>
                  {(current.category === "mechanical_repairs" ||
                    current.category === "contextual_repairs") && (
                    <div className={styles.field}>
                      <label
                        className={styles.fieldLabel}
                        htmlFor="audit-correction"
                      >
                        Corrected replacement (optional)
                      </label>
                      <input
                        id="audit-correction"
                        className={styles.textInput}
                        value={currentSessionOp.corrected_replacement ?? ""}
                        onChange={(event) =>
                          setSession(
                            setSessionCorrection(
                              session,
                              current.operation_id,
                              event.target.value.length === 0
                                ? null
                                : event.target.value,
                            ),
                          )
                        }
                        placeholder="Exact corrected replacement text"
                      />
                    </div>
                  )}
                </div>
              ) : null}

              {current.category === "polished_statements" ? (
                <div className={styles.attestationSection}>
                  <div className={styles.sectionLabel}>
                    Statement attestations (all six explicit)
                  </div>
                  {STATEMENT_ATTESTATION_FIELDS.map((field) => (
                    <div className={styles.attestationRow} key={field}>
                      <span className={styles.attestationLabel}>
                        {ATTESTATION_LABEL[field]}
                      </span>
                      <div
                        className={styles.attestationControls}
                        role="group"
                        aria-label={ATTESTATION_LABEL[field]}
                      >
                        <button
                          type="button"
                          className={`${styles.ternaryButton} ${
                            currentSessionOp[field] === true
                              ? styles.ternaryActive
                              : ""
                          }`}
                          aria-pressed={currentSessionOp[field] === true}
                          onClick={() =>
                            setSession(
                              setSessionAttestation(
                                session,
                                current.operation_id,
                                field,
                                true,
                              ),
                            )
                          }
                        >
                          Yes
                        </button>
                        <button
                          type="button"
                          className={`${styles.ternaryButton} ${
                            currentSessionOp[field] === false
                              ? styles.ternaryActive
                              : ""
                          }`}
                          aria-pressed={currentSessionOp[field] === false}
                          onClick={() =>
                            setSession(
                              setSessionAttestation(
                                session,
                                current.operation_id,
                                field,
                                false,
                              ),
                            )
                          }
                        >
                          No
                        </button>
                        <button
                          type="button"
                          className={styles.ternaryUnset}
                          onClick={() =>
                            setSession(
                              setSessionAttestation(
                                session,
                                current.operation_id,
                                field,
                                null,
                              ),
                            )
                          }
                        >
                          Unset
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}

              <p className={styles.completionNote}>
                {isOperationComplete(template, session, current.operation_id)
                  ? "This operation is complete."
                  : "This operation is not complete yet."}
              </p>
            </section>
          </main>
        ) : null}

        {completedHash !== null ? (
          <p className={styles.hashLine}>
            Last completed-audit export canonical SHA-256:{" "}
            <code>{completedHash}</code>
          </p>
        ) : null}

        <input
          ref={sessionInputRef}
          type="file"
          accept="application/json,.json"
          className={styles.hiddenInput}
          onChange={handleSessionChange}
          tabIndex={-1}
          aria-label="Import a prior Phase 2K audit session backup"
        />

        <footer className={styles.footer}>
          <Link href="/" className={styles.backLink}>
            Home
          </Link>
          <span>
            Session autosaves locally, bound to template and records hashes.
            Decisions and attestations are explicit; nothing is fabricated.
          </span>
        </footer>
      </div>
    </div>
  );
}
