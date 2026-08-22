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
import styles from "../app/phase2j-adjudicate/adjudicate.module.css";
import {
  AUDIT_CHECKS,
  buildAdjudicationExport,
  buildAdjudicationState,
  deriveResolvedEndpoints,
  emptyAuditChecks,
  auditChecksComplete,
  isWindowResolved,
  keepPassAChoices,
  parseAuditChecks,
  summarizeAdjudicationProgress,
  unresolvedComponents,
  validateAdjudicationExport,
  validateAdjudicationState,
  type AuditCheckKey,
  type AuditChecks,
  type AdjudicationComponent,
  type AdjudicationOutcome,
  type AdjudicationPayload,
  type AdjudicationRecord,
  type AdjudicationRecordState,
  type AdjudicationState,
  type ComponentDecision,
} from "../lib/phase2j-adjudication";
import {
  deriveEndpointSpan,
  ENDPOINT_TYPES,
  snapCharRangeToTokens,
  type EndpointType,
} from "../lib/phase2j-review";

const OUTCOMES: readonly AdjudicationOutcome[] = ["CLEAN", "AMBIGUOUS", "EXCLUDED"];

const TYPE_GUIDE: Record<EndpointType, { label: string; example: string }> = {
  ENTITY: { label: "Entity", example: "he · you · the enemy" },
  ABILITY_OR_RESOURCE: { label: "Ability / resource", example: "W · Flash · mana" },
  EVENT: { label: "Event", example: "the wave crashes" },
  ACTION: { label: "Action", example: "push the wave" },
  STATE: { label: "State", example: "you are low" },
  OUTCOME: { label: "Outcome", example: "you are dead" },
  QUANTITY: { label: "Quantity", example: "100 HP" },
  TIME: { label: "Time", example: "after level six" },
  LOCATION_OR_SPACE: { label: "Location / space", example: "under tower" },
  UNDETERMINED: { label: "Undetermined", example: "span clear, type not" },
};

const CLASSIFICATION_LABEL: Record<AdjudicationComponent["classification"], string> = {
  EXACT_AGREEMENT: "Exact agreement",
  TYPE_DISAGREEMENT: "Type disagreement",
  BOUNDARY_DISAGREEMENT: "Boundary disagreement",
  SOL_ONLY: "Sol only",
  HUMAN_ONLY: "Human only",
};

type Message = { kind: "error" | "info"; text: string };

type PickerState = {
  componentId: string;
  tokenStart: number | null;
  tokenEnd: number | null;
  nodeType: EndpointType | null;
};

function charOffsetAt(root: HTMLElement, container: Node, offset: number): number {
  const prefix = document.createRange();
  prefix.selectNodeContents(root);
  prefix.setEnd(container, offset);
  return prefix.toString().length;
}

export function Phase2JAdjudicationClient({
  payload,
}: {
  payload: AdjudicationPayload;
}) {
  const [state, setState] = useState<AdjudicationState>(() => buildAdjudicationState(payload));
  const [auditChecks, setAuditChecks] = useState<AuditChecks>(() => emptyAuditChecks());
  const [hydrated, setHydrated] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [message, setMessage] = useState<Message | null>(null);
  const [pendingKeepPassA, setPendingKeepPassA] = useState(false);
  const [pendingReset, setPendingReset] = useState(false);
  const [showHumanOverlay, setShowHumanOverlay] = useState(true);
  const [showSolOverlay, setShowSolOverlay] = useState(true);
  const [picker, setPicker] = useState<PickerState | null>(null);
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);

  const sourceRef = useRef<HTMLParagraphElement>(null);
  const pickerSourceRef = useRef<HTMLParagraphElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const stateRef = useRef(state);
  const auditChecksRef = useRef(auditChecks);
  const currentIndexRef = useRef(currentIndex);
  const pickerRefState = useRef(picker);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    auditChecksRef.current = auditChecks;
  }, [auditChecks]);

  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  useEffect(() => {
    pickerRefState.current = picker;
  }, [picker]);

  const storageKey = `phase2j-adjudication-state:v1:${payload.adjudication_packet_sha256}`;
  const auditStorageKey = `phase2j-adjudication-audit:v1:${payload.adjudication_packet_sha256}`;
  const record = payload.records[currentIndex];
  const stateRecord = state.records[currentIndex];
  const progress = useMemo(() => summarizeAdjudicationProgress(payload, state), [payload, state]);

  // Hydrate the local autosave bound to the exact adjudication packet hash.
  useEffect(() => {
    let restored: AdjudicationState | null = null;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw) {
        const parsed: unknown = JSON.parse(raw);
        const result = validateAdjudicationState(parsed, payload);
        if (result.ok) {
          restored = result.state;
        } else {
          setMessage({
            kind: "error",
            text: `Saved adjudication from this browser was ignored: ${result.errors[0]}`,
          });
        }
      }
    } catch {
      // Corrupt autosave: start fresh; nothing writes the canonical packet.
    }
    if (restored) {
      setState(restored);
      setMessage({ kind: "info", text: "Restored your saved adjudication from this browser." });
    }
    setHydrated(true);
  }, [storageKey, payload]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    try {
      const persisted: AdjudicationState = {
        schema_version: state.schema_version,
        adjudication_packet_sha256: state.adjudication_packet_sha256,
        reviewer_name: state.reviewer_name,
        records: state.records,
      };
      window.localStorage.setItem(storageKey, JSON.stringify(persisted));
    } catch {
      // Storage may be unavailable (private mode / quota); adjudication still works.
    }
  }, [state, hydrated, storageKey]);

  // The Pass B audit attestation is stored separately so the component
  // autosave schema (v1) stays unchanged and existing decisions survive.
  useEffect(() => {
    let restored: AuditChecks | null = null;
    try {
      const raw = window.localStorage.getItem(auditStorageKey);
      if (raw) {
        restored = parseAuditChecks(JSON.parse(raw));
      }
    } catch {
      // Corrupt attestation storage: start with all five unchecked.
    }
    if (restored) {
      setAuditChecks(restored);
    }
  }, [auditStorageKey]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    try {
      window.localStorage.setItem(auditStorageKey, JSON.stringify(auditChecks));
    } catch {
      // Storage may be unavailable; adjudication still works.
    }
  }, [auditChecks, hydrated, auditStorageKey]);

  const setAuditCheck = useCallback((key: AuditCheckKey, checked: boolean) => {
    setAuditChecks((previous) => ({ ...previous, [key]: checked }));
  }, []);

  const commitState = useCallback((nextState: AdjudicationState) => {
    setState(nextState);
  }, []);

  const commitRecord = useCallback((nextRecord: AdjudicationRecordState) => {
    const index = currentIndexRef.current;
    setState({
      ...stateRef.current,
      records: stateRef.current.records.map((existing, existingIndex) =>
        existingIndex === index ? nextRecord : existing,
      ),
    });
  }, []);

  const goTo = useCallback((index: number) => {
    const total = stateRef.current.records.length;
    const next = Math.min(Math.max(index, 0), total - 1);
    setCurrentIndex(next);
    setPicker(null);
    setPendingKeepPassA(false);
    setPendingReset(false);
  }, []);

  const setDecision = useCallback(
    (componentId: string, decision: ComponentDecision) => {
      const currentRecord = stateRef.current.records[currentIndexRef.current];
      commitRecord({
        ...currentRecord,
        decisions: { ...currentRecord.decisions, [componentId]: decision },
      });
    },
    [commitRecord],
  );

  const changeOutcome = useCallback(
    (outcome: AdjudicationOutcome) => {
      const currentRecord = stateRef.current.records[currentIndexRef.current];
      commitRecord({ ...currentRecord, outcome });
      if (outcome !== "CLEAN" && currentRecord.note.trim() === "") {
        setMessage({
          kind: "info",
          text: `${outcome} windows require a note before they can be exported.`,
        });
      }
    },
    [commitRecord],
  );

  const setNote = useCallback(
    (note: string) => {
      const currentRecord = stateRef.current.records[currentIndexRef.current];
      commitRecord({ ...currentRecord, note });
    },
    [commitRecord],
  );

  const setReviewer = useCallback(
    (reviewerName: string) => {
      setState({ ...stateRef.current, reviewer_name: reviewerName });
    },
    [],
  );

  const confirmKeepPassA = useCallback(() => {
    const currentRecord = stateRef.current.records[currentIndexRef.current];
    const payloadRecord = payload.records[currentIndexRef.current];
    commitRecord(keepPassAChoices(payloadRecord, currentRecord));
    setPendingKeepPassA(false);
    setMessage({
      kind: "info",
      text: "Window resolved to your Pass A choices; every Sol-only proposal was rejected explicitly.",
    });
  }, [commitRecord, payload.records]);

  const confirmReset = useCallback(() => {
    setState(buildAdjudicationState(payload));
    setCurrentIndex(0);
    setPendingReset(false);
    setPicker(null);
    setMessage({ kind: "info", text: "Adjudication reset to the clean packet state." });
  }, [payload]);

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
    const currentRecord = stateRef.current.records[currentIndexRef.current];
    if (currentRecord.outcome === "EXCLUDED") {
      return;
    }
    const range = selection.getRangeAt(0);
    if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) {
      return;
    }
    const first = charOffsetAt(root, range.startContainer, range.startOffset);
    const second = charOffsetAt(root, range.endContainer, range.endOffset);
    const snapped = snapCharRangeToTokens(
      payload.records[currentIndexRef.current].tokens,
      Math.min(first, second),
      Math.max(first, second),
    );
    if (!snapped) {
      return;
    }
    openPickerForSpan(snapped.token_start, snapped.token_end);
    selection.removeAllRanges();
  }, [payload.records]);

  const openPickerForSpan = useCallback(
    (tokenStart: number, tokenEnd: number) => {
      const currentRecord = stateRef.current.records[currentIndexRef.current];
      const payloadRecord = payload.records[currentIndexRef.current];
      const overlapping = payloadRecord.components.filter((component) => {
        const endpoints = [
          ...component.human_endpoint_ids.map((id) =>
            payloadRecord.human_endpoints.find((endpoint) => endpoint.endpoint_id === id),
          ),
          ...component.sol_endpoint_ids.map((id) =>
            payloadRecord.sol_endpoints.find((endpoint) => endpoint.endpoint_id === id),
          ),
        ].filter((endpoint) => endpoint !== undefined);
        return endpoints.some((endpoint) => {
          const span = deriveEndpointSpan(payloadRecord, endpoint.token_start, endpoint.token_end);
          return span && tokenStart <= endpoint.token_end && endpoint.token_start <= tokenEnd;
        });
      });
      if (currentRecord.outcome === "EXCLUDED") {
        setMessage({
          kind: "error",
          text: "EXCLUDED windows cannot keep endpoints. Switch the outcome back to CLEAN first.",
        });
        return;
      }
      if (overlapping.length === 0) {
        setMessage({
          kind: "error",
          text: "Custom spans must overlap an existing Human or Sol alternative so they can be attached to a component.",
        });
        return;
      }
      if (overlapping.length > 1) {
        setMessage({
          kind: "error",
          text: "That span overlaps more than one component; choose a span that touches exactly one decision.",
        });
        return;
      }
      setPicker({
        componentId: overlapping[0].component_id,
        tokenStart,
        tokenEnd,
        nodeType: null,
      });
      window.setTimeout(() => {
        pickerRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
      }, 0);
    },
    [payload.records],
  );

  const openCustomPicker = useCallback(
    (component: AdjudicationComponent, fixedSpan: boolean) => {
      const payloadRecord = payload.records[currentIndexRef.current];
      if (fixedSpan) {
        const source =
          component.human_endpoint_ids.length > 0
            ? payloadRecord.human_endpoints.find(
                (endpoint) => endpoint.endpoint_id === component.human_endpoint_ids[0],
              )
            : payloadRecord.sol_endpoints.find(
                (endpoint) => endpoint.endpoint_id === component.sol_endpoint_ids[0],
              );
        setPicker({
          componentId: component.component_id,
          tokenStart: source?.token_start ?? null,
          tokenEnd: source?.token_end ?? null,
          nodeType: null,
        });
      } else {
        setPicker({
          componentId: component.component_id,
          tokenStart: null,
          tokenEnd: null,
          nodeType: null,
        });
      }
      window.setTimeout(() => {
        pickerRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
      }, 0);
    },
    [payload.records],
  );

  const handlePickerSelection = useCallback(() => {
    const root = pickerSourceRef.current;
    const selection = window.getSelection?.();
    if (
      !root ||
      !selection ||
      selection.isCollapsed ||
      selection.rangeCount === 0
    ) {
      return;
    }
    const range = selection.getRangeAt(0);
    if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) {
      return;
    }
    const first = charOffsetAt(root, range.startContainer, range.startOffset);
    const second = charOffsetAt(root, range.endContainer, range.endOffset);
    const snapped = snapCharRangeToTokens(
      payload.records[currentIndexRef.current].tokens,
      Math.min(first, second),
      Math.max(first, second),
    );
    if (!snapped) {
      return;
    }
    setPicker((previous) =>
      previous ? { ...previous, tokenStart: snapped.token_start, tokenEnd: snapped.token_end } : previous,
    );
  }, [payload.records]);

  const applyCustomDecision = useCallback(() => {
    if (!picker || picker.tokenStart === null || picker.tokenEnd === null || !picker.nodeType) {
      setMessage({ kind: "error", text: "Choose an exact span and an endpoint type first." });
      return;
    }
    const payloadRecord = payload.records[currentIndexRef.current];
    const currentRecord = stateRef.current.records[currentIndexRef.current];
    const span = deriveEndpointSpan(payloadRecord, picker.tokenStart, picker.tokenEnd);
    if (!span) {
      setMessage({ kind: "error", text: "That span is outside the Bronze window." });
      return;
    }
    const decision: ComponentDecision = {
      kind: "CUSTOM",
      token_start: picker.tokenStart,
      token_end: picker.tokenEnd,
      node_type: picker.nodeType,
    };
    const derived = deriveResolvedEndpoints(payloadRecord, {
      ...currentRecord,
      decisions: { ...currentRecord.decisions, [picker.componentId]: decision },
    });
    if (derived.errors.length > 0) {
      setMessage({
        kind: "error",
        text: `Custom span rejected: ${derived.errors[0]}`,
      });
      return;
    }
    setDecision(picker.componentId, decision);
    setPicker(null);
    setMessage({
      kind: "info",
      text: `Custom endpoint “${span.exact_bronze_text}” (${picker.nodeType}) attached to ${CLASSIFICATION_LABEL[
        payloadRecord.components.find((component) => component.component_id === picker.componentId)!
          .classification
      ].toLowerCase()}.`,
    });
  }, [picker, payload.records, setDecision]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPicker(null);
        setPendingKeepPassA(false);
        setPendingReset(false);
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
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goTo, picker]);

  const handleExport = useCallback(() => {
    const result = buildAdjudicationExport(
      payload,
      stateRef.current,
      stateRef.current.reviewer_name,
      new Date().toISOString(),
      auditChecksRef.current,
    );
    if (!result.ok) {
      setMessage({
        kind: "error",
        text: `Export blocked: ${result.errors.slice(0, 3).join(" ")}`,
      });
      return;
    }
    const json = `${JSON.stringify(result.export, null, 2)}\n`;
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `phase2j-adjudication-export-${payload.adjudication_packet_sha256.slice(0, 8)}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 2000);
    setMessage({
      kind: "info",
      text: "Adjudication export downloaded with the Pass B audit attestation. This is REVIEW MATERIAL, not final gold; canonical import/final validation is a separate step.",
    });
  }, [payload]);

  const handleImportFile = useCallback(
    async (file: File) => {
      try {
        const text = await file.text();
        const parsed: unknown = JSON.parse(text);
        const result = validateAdjudicationExport(parsed, payload);
        if (!result.ok) {
          setMessage({
            kind: "error",
            text: `Import rejected: ${result.errors.slice(0, 3).join(" ")}`,
          });
          return;
        }
        setState(result.state);
        setAuditChecks(result.audit_checks);
        setCurrentIndex(0);
        setPicker(null);
        setMessage({
          kind: "info",
          text: "Adjudication export imported (including the Pass B audit attestation) and replacing the local session.",
        });
      } catch {
        setMessage({ kind: "error", text: "Import rejected: the file is not valid JSON." });
      }
    },
    [payload],
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

  const humanByToken = useMemo(() => {
    const map = new Map<number, AdjudicationComponent["human_endpoint_ids"]>();
    for (const component of record.components) {
      for (const endpointId of component.human_endpoint_ids) {
        const endpoint = record.human_endpoints.find((item) => item.endpoint_id === endpointId);
        if (!endpoint) {
          continue;
        }
        for (let index = endpoint.token_start; index <= endpoint.token_end; index += 1) {
          map.set(index, [...(map.get(index) ?? []), endpointId]);
        }
      }
    }
    return map;
  }, [record]);

  const solByToken = useMemo(() => {
    const map = new Map<number, AdjudicationComponent["sol_endpoint_ids"]>();
    for (const component of record.components) {
      for (const endpointId of component.sol_endpoint_ids) {
        const endpoint = record.sol_endpoints.find((item) => item.endpoint_id === endpointId);
        if (!endpoint) {
          continue;
        }
        for (let index = endpoint.token_start; index <= endpoint.token_end; index += 1) {
          map.set(index, [...(map.get(index) ?? []), endpointId]);
        }
      }
    }
    return map;
  }, [record]);

  const currentUnresolved = useMemo(
    () => unresolvedComponents(record, stateRecord),
    [record, stateRecord],
  );
  const currentWindowResolved = useMemo(
    () => isWindowResolved(record, stateRecord),
    [record, stateRecord],
  );
  const currentDerived = useMemo(
    () => deriveResolvedEndpoints(record, stateRecord),
    [record, stateRecord],
  );
  const currentDerivedIssues = currentDerived.errors.filter(
    (error) => !error.includes("unresolved"),
  );

  const humanSetFor = useCallback(
    (component: AdjudicationComponent) =>
      component.human_endpoint_ids
        .map((id) => record.human_endpoints.find((endpoint) => endpoint.endpoint_id === id))
        .filter((endpoint) => endpoint !== undefined),
    [record],
  );

  const solSetFor = useCallback(
    (component: AdjudicationComponent) =>
      component.sol_endpoint_ids
        .map((id) => record.sol_endpoints.find((endpoint) => endpoint.endpoint_id === id))
        .filter((endpoint) => endpoint !== undefined),
    [record],
  );

  const renderDecisionControls = useCallback(
    (component: AdjudicationComponent) => {
      const decision = stateRecord.decisions[component.component_id];
      const human = humanSetFor(component);
      const sol = solSetFor(component);
      const resolved = decision !== undefined || component.classification === "EXACT_AGREEMENT";
      const option = (key: string, label: string, target: ComponentDecision) => (
        <button
          key={key}
          type="button"
          className={`${styles.decisionButton} ${
            decision?.kind === target.kind ? styles.decisionActive : ""
          }`}
          aria-pressed={decision?.kind === target.kind}
          onClick={() => setDecision(component.component_id, target)}
        >
          {label}
        </button>
      );

      return (
        <div className={styles.decisionPanel}>
          <div className={styles.decisionHeader}>
            <span className={styles.classification}>{CLASSIFICATION_LABEL[component.classification]}</span>
            <span className={resolved ? styles.resolvedChip : styles.unresolvedChip}>
              {resolved ? "Resolved" : "Unresolved"}
            </span>
          </div>

          {component.classification === "EXACT_AGREEMENT" ? (
            <p className={styles.exactAgreement}>
              Both sides agree on the exact Bronze mention and type. Agreement is evidence, not
              proof: the shared endpoint is pre-resolved to keep, and you may keep, edit, or drop
              it explicitly.
            </p>
          ) : null}

          <div className={styles.sideRow}>
            <div className={`${styles.sideBlock} ${styles.humanSide}`}>
              <span className={styles.sideLabel}>Human Pass A</span>
              {human.length === 0 ? (
                <span className={styles.sideEmpty}>no human endpoint here</span>
              ) : (
                human.map((endpoint) => (
                  <span key={endpoint.endpoint_id} className={styles.sideItem}>
                    <span className={styles.sidePhrase}>“{endpoint.exact_bronze_text}”</span>
                    <span className={styles.sideType}>{endpoint.node_type}</span>
                  </span>
                ))
              )}
            </div>
            <div className={`${styles.sideBlock} ${styles.solSide}`}>
              <span className={styles.sideLabel}>Sol proposal · second opinion</span>
              {sol.length === 0 ? (
                <span className={styles.sideEmpty}>no Sol proposal here</span>
              ) : (
                sol.map((endpoint) => (
                  <span key={endpoint.endpoint_id} className={styles.sideItem}>
                    <span className={styles.sidePhrase}>“{endpoint.exact_bronze_text}”</span>
                    <span className={styles.sideType}>
                      {endpoint.node_type ?? "type unspecified"}
                    </span>
                    {endpoint.sol_rationale ? (
                      <span className={styles.sideRationale}>{endpoint.sol_rationale}</span>
                    ) : null}
                  </span>
                ))
              )}
            </div>
          </div>

          <div className={styles.decisionOptions} role="group" aria-label={`Decide ${component.component_id}`}>
            {component.classification === "EXACT_AGREEMENT" ? (
              <>
                {option("keep", "Keep shared endpoint", { kind: "KEEP_HUMAN_SET" })}
                <button
                  type="button"
                  className={styles.decisionButton}
                  onClick={() => openCustomPicker(component, true)}
                >
                  Custom span/type…
                </button>
                {option("drop", "Drop", { kind: "DROP" })}
              </>
            ) : null}
            {component.classification === "TYPE_DISAGREEMENT" ? (
              <>
                {option("human", `Keep Human type (${human[0]?.node_type ?? "?"})`, { kind: "KEEP_HUMAN_SET" })}
                {option("sol", `Choose Sol type (${sol[0]?.node_type ?? "?"})`, { kind: "KEEP_SOL_SET" })}
                <button
                  type="button"
                  className={styles.decisionButton}
                  onClick={() => openCustomPicker(component, true)}
                >
                  Custom type…
                </button>
                {option("drop", "Drop", { kind: "DROP" })}
              </>
            ) : null}
            {component.classification === "BOUNDARY_DISAGREEMENT" ? (
              <>
                {option("human", `Keep Human set (${human.length})`, { kind: "KEEP_HUMAN_SET" })}
                {option("sol", `Keep Sol set (${sol.length})`, { kind: "KEEP_SOL_SET" })}
                <button
                  type="button"
                  className={styles.decisionButton}
                  onClick={() => openCustomPicker(component, false)}
                >
                  Custom span…
                </button>
                {option("drop", "Drop", { kind: "DROP" })}
              </>
            ) : null}
            {component.classification === "SOL_ONLY" ? (
              <>
                {option("accept", "Accept", { kind: "KEEP_SOL_SET" })}
                <button
                  type="button"
                  className={styles.decisionButton}
                  onClick={() => openCustomPicker(component, false)}
                >
                  Custom edit…
                </button>
                {option("reject", "Reject", { kind: "DROP" })}
              </>
            ) : null}
            {component.classification === "HUMAN_ONLY" ? (
              <>
                {option("keep", "Keep", { kind: "KEEP_HUMAN_SET" })}
                <button
                  type="button"
                  className={styles.decisionButton}
                  onClick={() => openCustomPicker(component, false)}
                >
                  Custom edit…
                </button>
                {option("drop", "Drop", { kind: "DROP" })}
              </>
            ) : null}
          </div>
        </div>
      );
    },
    [stateRecord, humanSetFor, solSetFor, setDecision, openCustomPicker],
  );

  const overviewClass = (index: number): string => {
    const item = payload.records[index];
    const itemState = state.records[index];
    if (!itemState) {
      return styles.cellUnresolved;
    }
    if (itemState.outcome === "AMBIGUOUS") {
      return styles.cellAmbiguous;
    }
    if (itemState.outcome === "EXCLUDED") {
      return styles.cellExcluded;
    }
    if (isWindowResolved(item, itemState)) {
      return styles.cellResolved;
    }
    const resolvedCount = item.components.filter((component) =>
      component.classification === "EXACT_AGREEMENT" ||
      itemState.decisions[component.component_id] !== undefined,
    ).length;
    return resolvedCount > 0 ? styles.cellPartial : styles.cellUnresolved;
  };

  const overviewLabel = (index: number): string => {
    const item = payload.records[index];
    const itemState = state.records[index];
    if (!itemState) {
      return "No state";
    }
    if (itemState.outcome === "AMBIGUOUS") {
      return "Ambiguous";
    }
    if (itemState.outcome === "EXCLUDED") {
      return "Excluded";
    }
    return isWindowResolved(item, itemState) ? "Resolved" : "In progress";
  };

  const tokenClass = (tokenIndex: number): string => {
    const classes = [styles.token];
    const hasHuman = showHumanOverlay && humanByToken.has(tokenIndex);
    const hasSol = showSolOverlay && solByToken.has(tokenIndex);
    if (hasHuman && hasSol) {
      classes.push(styles.tokenBoth);
    } else if (hasHuman) {
      classes.push(styles.tokenHuman);
    } else if (hasSol) {
      classes.push(styles.tokenSol);
    }
    if (hoveredToken === tokenIndex) {
      classes.push(styles.tokenHover);
    }
    return classes.join(" ");
  };

  const pickerRecord = payload.records[currentIndex];
  const pickerSpan =
    picker && picker.tokenStart !== null && picker.tokenEnd !== null
      ? deriveEndpointSpan(pickerRecord, picker.tokenStart, picker.tokenEnd)
      : null;
  const requiresNote = stateRecord.outcome !== "CLEAN";

  return (
    <div className={styles.container}>
      <div className={styles.inner}>
        <header className={styles.topbar}>
          <Link href="/phase2j-review/" className={styles.backLink} aria-label="Back to Pass A review">
            ← Pass A review
          </Link>
          <div className={styles.titleBlock}>
            <h1 className={styles.title}>Phase 2J · Human vs Sol adjudication</h1>
            <p className={styles.subtitle}>
              Your Pass A review is complete, so Sol&apos;s sealed second opinion may now be
              revealed for explicit adjudication. Sol is a navigation/audit proposal, never gold
              and never auto-promoted. The export below remains REVIEW MATERIAL until a separately
              validated canonical import/finalizer runs.
            </p>
          </div>
          <div className={styles.actions}>
            <label className={styles.fieldLabel}>
              Reviewer
              <input
                className={styles.fieldInput}
                value={state.reviewer_name}
                onChange={(event) => setReviewer(event.target.value)}
                placeholder="Your name"
                aria-label="Reviewer name"
              />
            </label>
            <button
              type="button"
              className={styles.actionButton}
              onClick={() => fileInputRef.current?.click()}
            >
              Import export
            </button>
            <button type="button" className={styles.exportButton} onClick={handleExport}>
              Export review material
            </button>
            <button
              type="button"
              className={styles.actionButton}
              onClick={() => setPendingReset(true)}
            >
              Reset
            </button>
            <input
              ref={fileInputRef}
              className={styles.hiddenInput}
              type="file"
              accept="application/json"
              onChange={handleFileChange}
            />
          </div>
        </header>

        {message ? (
          <div className={message.kind === "error" ? styles.messageError : styles.messageInfo} role="status">
            {message.text}
          </div>
        ) : null}

        {pendingReset ? (
          <section className={styles.confirmPanel} aria-label="Confirm reset">
            <p>Reset this browser&apos;s adjudication to the clean packet state? Exports already downloaded are unaffected.</p>
            <div className={styles.confirmActions}>
              <button type="button" className={styles.confirmButton} onClick={confirmReset}>
                Reset adjudication
              </button>
              <button
                type="button"
                className={styles.cancelButton}
                onClick={() => setPendingReset(false)}
              >
                Cancel
              </button>
            </div>
          </section>
        ) : null}

        <section className={styles.progressBlock} aria-label="Adjudication progress">
          <div className={styles.progressText}>
            <span>
              Window {currentIndex + 1} of {payload.records.length}
            </span>
            <span>
              {progress.resolved_windows}/{progress.windows} windows · {progress.resolved_components}/
              {progress.components} components resolved
            </span>
          </div>
          <div
            className={styles.progressTrack}
            role="progressbar"
            aria-label="Adjudication deck progress"
            aria-valuemin={0}
            aria-valuemax={progress.components}
            aria-valuenow={progress.resolved_components}
          >
            <div
              className={styles.progressFill}
              style={{ width: `${(progress.resolved_components / Math.max(progress.components, 1)) * 100}%` }}
            />
          </div>
          <div className={styles.overview} role="group" aria-label="Window overview">
            {payload.records.map((item, index) => (
              <button
                key={item.window_id}
                type="button"
                className={`${styles.overviewCell} ${overviewClass(index)} ${
                  index === currentIndex ? styles.overviewCurrent : ""
                }`}
                onClick={() => goTo(index)}
                title={`Window ${index + 1}: ${overviewLabel(index)}`}
                aria-label={`Go to window ${index + 1}, ${overviewLabel(index)}`}
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
          <span className={styles.deckPosition}>
            {overviewLabel(currentIndex)} · {currentUnresolved.length} unresolved component
            {currentUnresolved.length === 1 ? "" : "s"}
          </span>
          <button
            type="button"
            className={`${styles.navButton} ${currentWindowResolved ? styles.navEmphasized : ""}`}
            onClick={() => goTo(currentIndex + 1)}
            disabled={currentIndex === payload.records.length - 1}
            aria-label="Next window"
          >
            Next →
          </button>
        </nav>

        <main className={styles.windowCard}>
          <div className={styles.windowMeta}>
            <span className={styles.metaTag}>Window {record.record_index}</span>
            <span className={styles.metaTag} title={record.window_id}>
              {record.window_id.length > 52 ? `${record.window_id.slice(0, 52)}…` : record.window_id}
            </span>
            <span className={styles.metaTag}>{record.bronze_char_length} chars</span>
            <span className={styles.metaTag}>
              Pass A: {record.human_endpoints.length} endpoints
            </span>
            <span className={styles.metaTag}>
              Sol: {record.sol_endpoints.length} proposals
            </span>
          </div>

          <div className={styles.sourceBlock}>
            <div className={styles.sourceHeading}>
              <span className={styles.sourceTitle}>Bronze source</span>
              <div className={styles.overlayToggles}>
                <button
                  type="button"
                  className={`${styles.toggleButton} ${showHumanOverlay ? styles.toggleHumanOn : ""}`}
                  aria-pressed={showHumanOverlay}
                  onClick={() => setShowHumanOverlay((value) => !value)}
                >
                  Human overlay
                </button>
                <button
                  type="button"
                  className={`${styles.toggleButton} ${showSolOverlay ? styles.toggleSolOn : ""}`}
                  aria-pressed={showSolOverlay}
                  onClick={() => setShowSolOverlay((value) => !value)}
                >
                  Sol overlay
                </button>
              </div>
            </div>
            <p
              ref={sourceRef}
              className={styles.sourceText}
              onMouseUp={handleSelection}
              aria-label={`Bronze window ${record.record_index} source text`}
            >
              {record.tokens.map((token, index) => {
                const gap =
                  index < record.tokens.length - 1
                    ? record.bronze_text.slice(token.end, record.tokens[index + 1].start)
                    : record.bronze_text.slice(token.end);
                return (
                  <Fragment key={token.token_index}>
                    <span
                      className={tokenClass(token.token_index)}
                      data-token-index={token.token_index}
                      onMouseEnter={() => setHoveredToken(token.token_index)}
                      onMouseLeave={() => setHoveredToken(null)}
                      onFocus={() => setHoveredToken(token.token_index)}
                      onBlur={() => setHoveredToken(null)}
                      tabIndex={-1}
                    >
                      {token.text}
                    </span>
                    {gap.length > 0 ? <span className={styles.gap}>{gap}</span> : null}
                  </Fragment>
                );
              })}
            </p>
            <p className={styles.selectionHint}>
              Drag across the Bronze text to create a custom span for one component.
            </p>
          </div>

          <section className={styles.componentsSection} aria-label="Disagreement components">
            <div className={styles.railHeading}>
              <span>Decisions in this window</span>
              <span className={styles.countPill}>
                {record.components.length - currentUnresolved.length} / {record.components.length} resolved
              </span>
            </div>
            {record.components.map(renderDecisionControls)}
          </section>
        </main>

        <section className={styles.statusPanel} aria-label="Window outcome">
          <h2>Window outcome</h2>
          <div className={styles.outcomeRow}>
            {OUTCOMES.map((outcome) => (
              <button
                key={outcome}
                type="button"
                className={`${styles.outcomeButton} ${
                  stateRecord.outcome === outcome ? styles.outcomeActive : ""
                }`}
                aria-pressed={stateRecord.outcome === outcome}
                onClick={() => changeOutcome(outcome)}
              >
                {outcome}
              </button>
            ))}
          </div>
          <p className={styles.outcomeHelp}>
            {stateRecord.outcome === "CLEAN"
              ? "Every component below must be resolved before export."
              : `${stateRecord.outcome} resolves the window: a note is required, and ${stateRecord.outcome === "EXCLUDED" ? "all endpoints are cleared" : "unresolved components may remain marked"}.`}
          </p>
          <label className={styles.noteLabel} htmlFor={`note-${record.record_index}`}>
            Window note {requiresNote ? <span className={styles.required}>· required</span> : null}
          </label>
          <textarea
            id={`note-${record.record_index}`}
            className={styles.noteInput}
            value={stateRecord.note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="Explain ambiguity, exclusion, or any custom decisions…"
          />
          {requiresNote && stateRecord.note.trim() === "" ? (
            <p className={styles.requiredNote}>{stateRecord.outcome} windows require a note before export.</p>
          ) : null}
          {currentDerivedIssues.length > 0 ? (
            <p className={styles.requiredNote}>
              Resolved set problem: {currentDerivedIssues[0]}
            </p>
          ) : null}
          <div className={styles.passRow}>
            <button
              type="button"
              className={styles.completeButton}
              onClick={() => setPendingKeepPassA(true)}
              aria-pressed={pendingKeepPassA}
            >
              Keep my Pass A choices
            </button>
            <span className={styles.passNote}>
              Resolves every remaining decision to your Pass A endpoint set and explicitly rejects
              Sol-only proposals in this window.
            </span>
          </div>
          {pendingKeepPassA ? (
            <section className={styles.confirmPanel} aria-label="Confirm Pass A resolution">
              <p>
                Confirm: resolve every remaining decision in this window to Human Pass A and reject
                every Sol-only proposal?
              </p>
              <div className={styles.confirmActions}>
                <button type="button" className={styles.confirmButton} onClick={confirmKeepPassA}>
                  Confirm Pass A choices
                </button>
                <button
                  type="button"
                  className={styles.cancelButton}
                  onClick={() => setPendingKeepPassA(false)}
                >
                  Cancel
                </button>
              </div>
            </section>
          ) : null}
        </section>

        <section className={styles.auditSection} aria-label="Pass B final audit attestation">
          <h2 className={styles.auditTitle}>Pass B final audit</h2>
          <p className={styles.auditIntro}>
            Checking all five boxes attests that you completed the source-grounded Pass B audit
            across all {payload.records.length} windows: verifying boundaries, omissions, roles,
            duplicates, and ambiguity against Bronze. This is an audit attestation, not model
            approval — no model output is endorsed by checking these boxes. All five are required
            before the REVIEW MATERIAL export can be produced.
          </p>
          <div className={styles.auditGrid}>
            {AUDIT_CHECKS.map((key) => (
              <label key={key} className={styles.auditCheck}>
                <input
                  type="checkbox"
                  className={styles.auditCheckInput}
                  checked={auditChecks[key]}
                  onChange={(event) => setAuditCheck(key, event.target.checked)}
                />
                <span className={styles.auditCheckLabel}>{key}</span>
              </label>
            ))}
          </div>
          {!auditChecksComplete(auditChecks) ? (
            <p className={styles.auditWarning}>
              All five Pass B audit checks must be true before export.
            </p>
          ) : null}
        </section>

        <footer className={styles.footer}>
          Human Pass A remains the baseline; Sol is a second opinion. No model data, scores, or
          predictions are stored here. Exports carry the Pass B audit attestation and remain
          REVIEW MATERIAL, not canonical gold.
        </footer>
      </div>

      {picker ? (
        <div className={styles.pickerBackdrop} onClick={() => setPicker(null)}>
          <div
            ref={pickerRef}
            className={styles.picker}
            role="dialog"
            aria-modal="true"
            aria-label="Custom endpoint editor"
            onClick={(event) => event.stopPropagation()}
          >
            <h2 className={styles.pickerTitle}>Custom endpoint</h2>
            <p className={styles.pickerIntro}>
              Select an exact Bronze span, then choose a type. The span must touch exactly one
              component.
            </p>
            <p
              ref={pickerSourceRef}
              className={styles.pickerSource}
              onMouseUp={handlePickerSelection}
            >
              {pickerRecord.tokens.map((token, index) => {
                const gap =
                  index < pickerRecord.tokens.length - 1
                    ? pickerRecord.bronze_text.slice(token.end, pickerRecord.tokens[index + 1].start)
                    : pickerRecord.bronze_text.slice(token.end);
                const selected =
                  picker.tokenStart !== null &&
                  picker.tokenEnd !== null &&
                  token.token_index >= picker.tokenStart &&
                  token.token_index <= picker.tokenEnd;
                return (
                  <Fragment key={token.token_index}>
                    <span className={`${styles.pickerToken} ${selected ? styles.pickerTokenSelected : ""}`}>
                      {token.text}
                    </span>
                    {gap.length > 0 ? <span className={styles.gap}>{gap}</span> : null}
                  </Fragment>
                );
              })}
            </p>
            <p className={styles.pickerSpan}>
              {pickerSpan
                ? `Span: “${pickerSpan.exact_bronze_text}”`
                : "No span selected yet — drag across the text above."}
            </p>
            <fieldset className={styles.typeFieldset}>
              <legend>Endpoint type</legend>
              <div className={styles.pickerGrid}>
                {ENDPOINT_TYPES.map((nodeType) => (
                  <button
                    key={nodeType}
                    type="button"
                    className={`${styles.pickerButton} ${
                      picker.nodeType === nodeType ? styles.pickerButtonActive : ""
                    }`}
                    aria-pressed={picker.nodeType === nodeType}
                    onClick={() => setPicker({ ...picker, nodeType })}
                  >
                    <span className={styles.pickerButtonLabel}>{TYPE_GUIDE[nodeType].label}</span>
                    <span className={styles.pickerButtonExample}>{TYPE_GUIDE[nodeType].example}</span>
                  </button>
                ))}
              </div>
            </fieldset>
            <div className={styles.confirmActions}>
              <button
                type="button"
                className={styles.confirmButton}
                onClick={applyCustomDecision}
                disabled={!pickerSpan || !picker.nodeType}
              >
                Apply custom endpoint
              </button>
              <button
                type="button"
                className={styles.cancelButton}
                onClick={() => setPicker(null)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
