"use client";

import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { basePath, hasSupabaseConfig, supabase } from "../lib/supabase";

type QueryResponse = {
  answer: string;
  sources: string[];
  normalized_question: string;
  metadata: {
    game: string;
    subject?: string | null;
    role?: string | null;
    reasoning?: string | null;
  };
};

const queryApiUrl = process.env.NEXT_PUBLIC_QUERY_API_URL ?? "http://localhost:8000";
const authRequired = (process.env.NEXT_PUBLIC_REQUIRE_AUTH ?? "false").toLowerCase() === "true";

type AuthStatus = "loading" | "signed_out" | "signed_in";

type ListNode = {
  text: string;
  children: ListNode[];
};

type ParsedSource = {
  metrics: string | null;
  category: string | null;
  text: string;
};

const GAME_GUIDES = {
  lol: {
    title: "League query guide",
    concise: [
      "How do I lane as Riven into Ambessa?",
      "What is Aatrox's win condition into Darius?",
      "How should I teamfight as Kai'Sa?"
    ],
    detailed: [
      "How do I play Riven into Ambessa in detail?",
      "Give me an in-depth guide for Aatrox into Darius.",
      "How should I play Jinx step by step from lane to teamfights?"
    ],
    notes: [
      "Use exact champion names when possible for the best matchup retrieval.",
      "Add keywords like `in detail`, `in-depth`, `step by step`, or `full guide` when you want a longer answer.",
      "Leave those detail keywords out when you want a shorter, more concise response."
    ]
  },
  aoe2: {
    title: "AoE2 query guide",
    concise: [
      "How should I open with Franks?",
      "What is the core identity of Malay?",
      "How do I defend early pressure with better micro?"
    ],
    detailed: [
      "How should I play Khmer in detail?",
      "Give me an in-depth guide for Byzantines.",
      "How do I play Hindustanis step by step from opening through win condition?"
    ],
    notes: [
      "Use exact civilization names for the best cross-reference and matchup retrieval.",
      "Add `in detail`, `in-depth`, `step by step`, or `full guide` when you want the fuller civilization overview path.",
      "Ask shorter direct questions when you want a concise answer focused on one phase, unit, or problem."
    ]
  }
} as const;

function renderInlineMarkdown(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g).filter(Boolean);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={index}>{part.slice(1, -1)}</em>;
    }
    return <span key={index}>{part}</span>;
  });
}

function renderAnswer(answer: string): ReactNode[] {
  const lines = answer.split(/\r?\n/);
  const nodes: ReactNode[] = [];
  let paragraphLines: string[] = [];
  let listItems: { depth: number; text: string }[] = [];

  function flushParagraph() {
    if (paragraphLines.length === 0) return;
    nodes.push(
      <p key={`p-${nodes.length}`}>
        {renderInlineMarkdown(paragraphLines.join(" "))}
      </p>
    );
    paragraphLines = [];
  }

  function buildList(items: { depth: number; text: string }[], keyPrefix: string): ReactNode {
    const root: { children: ListNode[] } = { children: [] };
    const stack = [{ depth: -1, node: root }];

    for (const item of items) {
      while (stack.length > 1 && item.depth <= stack[stack.length - 1].depth) {
        stack.pop();
      }
      const next: ListNode = { text: item.text, children: [] };
      stack[stack.length - 1].node.children.push(next);
      stack.push({ depth: item.depth, node: next });
    }

    const renderList = (entries: ListNode[], depth: number): ReactNode => (
      <ul className="answerList" key={`${keyPrefix}-${depth}-${entries.length}`}>
        {entries.map((entry, index) => (
          <li key={`${keyPrefix}-${depth}-${index}`}>
            <span>{renderInlineMarkdown(entry.text)}</span>
            {entry.children.length > 0 ? renderList(entry.children, depth + 1) : null}
          </li>
        ))}
      </ul>
    );

    return renderList(root.children, 0);
  }

  function flushList() {
    if (listItems.length === 0) return;
    nodes.push(buildList(listItems, `list-${nodes.length}`));
    listItems = [];
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (/^---+$/.test(trimmed)) {
      flushParagraph();
      flushList();
      nodes.push(<hr className="answerRule" key={`rule-${nodes.length}`} />);
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = Math.min(headingMatch[1].length + 1, 6);
      const title = headingMatch[2];
      if (level === 2) nodes.push(<h2 key={`h-${nodes.length}`}>{renderInlineMarkdown(title)}</h2>);
      else if (level === 3) nodes.push(<h3 key={`h-${nodes.length}`}>{renderInlineMarkdown(title)}</h3>);
      else if (level === 4) nodes.push(<h4 key={`h-${nodes.length}`}>{renderInlineMarkdown(title)}</h4>);
      else nodes.push(<p className="answerHeading" key={`h-${nodes.length}`}>{renderInlineMarkdown(title)}</p>);
      continue;
    }

    const listMatch = rawLine.match(/^(\s*)([-*])\s+(.+)$/);
    if (listMatch) {
      flushParagraph();
      const depth = Math.floor(listMatch[1].length / 2);
      listItems.push({ depth, text: listMatch[3].trim() });
      continue;
    }

    flushList();
    paragraphLines.push(trimmed);
  }

  flushParagraph();
  flushList();
  return nodes;
}

function parseSourceLine(line: string): ParsedSource {
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

function siteUrl() {
  if (typeof window === "undefined") return undefined;
  const suffix = basePath || "/";
  return new URL(suffix, window.location.origin).toString();
}

function cleanAuthCallbackUrl() {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  const authParams = [
    "code",
    "error",
    "error_code",
    "error_description",
    "state",
  ];
  let mutated = false;
  for (const key of authParams) {
    if (url.searchParams.has(key)) {
      url.searchParams.delete(key);
      mutated = true;
    }
  }
  if (!mutated) return;
  const nextPath = `${url.pathname}${url.search}${url.hash}`;
  window.history.replaceState({}, "", nextPath);
}

export function QueryClient() {
  const abortRef = useRef<AbortController | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [authStatus, setAuthStatus] = useState<AuthStatus>("loading");
  const [email, setEmail] = useState("");
  const [game, setGame] = useState<"aoe2" | "lol">("aoe2");
  const [question, setQuestion] = useState("how should I play Khmer in detail");
  const [splitDetail, setSplitDetail] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSendingMagicLink, setIsSendingMagicLink] = useState(false);
  const [isRefreshingAuth, setIsRefreshingAuth] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const [showLennyPhoto, setShowLennyPhoto] = useState(true);
  const guide = GAME_GUIDES[game];

  useEffect(() => {
    if (!authRequired) {
      setAuthStatus("signed_in");
      return;
    }
    if (!supabase) {
      setAuthStatus("signed_out");
      return;
    }
    const client = supabase;

    let active = true;

    async function hydrateSession() {
      setAuthStatus("loading");
      setError(null);

      const currentUrl = typeof window === "undefined" ? null : new URL(window.location.href);
      const callbackError = currentUrl?.searchParams.get("error_description") ?? currentUrl?.searchParams.get("error");
      if (callbackError && active) {
        setError(decodeURIComponent(callbackError.replace(/\+/g, " ")));
      }

      const code = currentUrl?.searchParams.get("code");
      if (code) {
        const { error: exchangeError } = await client.auth.exchangeCodeForSession(code);
        if (!active) return;
        if (exchangeError) {
          setError(exchangeError.message);
        } else {
          cleanAuthCallbackUrl();
        }
      }

      const { data, error: sessionError } = await client.auth.getSession();
      if (!active) return;
      if (sessionError) {
        setError(sessionError.message);
      }
      setSession(data.session);
      setAuthStatus(data.session ? "signed_in" : "signed_out");
    }

    hydrateSession().catch((err: unknown) => {
      if (!active) return;
      setError(err instanceof Error ? err.message : String(err));
      setAuthStatus("signed_out");
    });

    const { data } = client.auth.onAuthStateChange((_event, nextSession) => {
      if (!active) return;
      setSession(nextSession);
      setAuthStatus(nextSession ? "signed_in" : "signed_out");
    });
    return () => {
      active = false;
      data.subscription.unsubscribe();
    };
  }, []);

  async function refreshExistingSession() {
    const client = supabase;
    if (!client) {
      setError("Supabase env vars are missing.");
      return;
    }

    setIsRefreshingAuth(true);
    setError(null);
    setNotice(null);

    try {
      const currentUrl = typeof window === "undefined" ? null : new URL(window.location.href);
      const code = currentUrl?.searchParams.get("code");
      if (code) {
        const { error: exchangeError } = await client.auth.exchangeCodeForSession(code);
        if (exchangeError) {
          setError(exchangeError.message);
          return;
        }
        cleanAuthCallbackUrl();
      }

      const { data, error: sessionError } = await client.auth.getSession();
      if (sessionError) {
        setError(sessionError.message);
        return;
      }
      if (data.session) {
        setSession(data.session);
        setAuthStatus("signed_in");
        setNotice("Existing session restored.");
        return;
      }
      setNotice("No saved session found in this browser yet. Use the email link once first.");
    } finally {
      setIsRefreshingAuth(false);
    }
  }

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (game !== "aoe2") {
      setSplitDetail(false);
    }
  }, [game]);

  async function signIn() {
    setError(null);
    setNotice(null);
    const client = supabase;
    if (!client) {
      setError("Supabase env vars are missing.");
      return;
    }
    if (isSendingMagicLink) return;

    setIsSendingMagicLink(true);
    const { error: signInError } = await client.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: siteUrl() }
    });
    setIsSendingMagicLink(false);
    if (signInError) {
      const message = signInError.message.toLowerCase().includes("rate limit")
        ? "Email rate limit exceeded. Use Google sign-in or wait a minute before requesting another link."
        : signInError.message;
      setError(message);
      return;
    }
    setNotice("Magic link sent. Open it once in this browser to create a reusable session.");
  }

  async function askQuestion() {
    if (isSubmitting) return;
    const trimmedQuestion = question.trim();
    if (trimmedQuestion.length < 2) return;

    setIsSubmitting(true);
    setError(null);
    setResult(null);
    setShowSources(false);
    const token = authRequired ? session?.access_token : null;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${queryApiUrl}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {})
        },
        body: JSON.stringify({
          game,
          question: trimmedQuestion,
          show_sources: true,
          split_detail: splitDetail
        }),
        signal: controller.signal
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      setResult((await response.json()) as QueryResponse);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setIsSubmitting(false);
    }
  }

  if (authRequired && !hasSupabaseConfig) {
    return (
      <main className="shell">
        <section className="panel">
          <p className="eyebrow">Configuration needed</p>
          <h1>Set Supabase frontend environment variables.</h1>
          <p>
            Add <code>NEXT_PUBLIC_SUPABASE_URL</code>, <code>NEXT_PUBLIC_SUPABASE_ANON_KEY</code>,
            and <code>NEXT_PUBLIC_QUERY_API_URL</code> before using the private web UI.
          </p>
        </section>
      </main>
    );
  }

  if (authRequired && authStatus === "loading") {
    return (
      <main className="shell">
        <section className="panel auth authLoading">
          <p className="eyebrow">Lenny&apos;s wise game wizard</p>
          <h1>Checking your saved session.</h1>
          <p>
            If you already signed in on this browser, Lenny should let you straight back in without another email.
          </p>
        </section>
      </main>
    );
  }

  if (authRequired && (authStatus !== "signed_in" || !session)) {
    return (
      <main className="shell">
        <section className="panel auth">
          <p className="eyebrow">Lenny&apos;s wise game wizard</p>
          <h1>Sign in to ask Lenny for matchup and strategy help.</h1>
          <p className="authLead">
            Send yourself one magic link, click it once in this browser, then use Log in to restore the saved
            session without resending email.
          </p>
          <div className="authStack">
            <div className="authRow">
              <input
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@example.com"
                type="email"
              />
              <button disabled={isSendingMagicLink || email.trim().length < 3} onClick={signIn}>
                {isSendingMagicLink ? "Sending..." : "Send magic link"}
              </button>
              <button className="ghost authLoginButton" disabled={isRefreshingAuth} onClick={refreshExistingSession}>
                {isRefreshingAuth ? "Checking..." : "Log in"}
              </button>
            </div>
          </div>
          {notice && <p className="notice">{notice}</p>}
          {error && <p className="notice error">{error}</p>}
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <section className="hero">
        <div className="heroCopy">
          <p className="eyebrow">Lenny&apos;s wise game wizard</p>
          <img
            alt="Lenny's wise game wizard poster"
            className="heroPoster"
            src={`${basePath || ""}/attendee-rules.jpg`}
          />
          <p>
            Query League of Legends coaching data and Age of Empires II guides from one private interface,
            with structured answers and source evidence when you want to inspect it.
          </p>
          {authRequired ? (
            <div className="heroActions">
              <button className="ghost" onClick={() => supabase?.auth.signOut()}>Sign out</button>
            </div>
          ) : null}
        </div>
        <div className="heroPhoto">
          {showLennyPhoto ? (
            <img
              alt="Lenny the plush game wizard"
              src={`${basePath || ""}/lenny.png`}
              onError={() => setShowLennyPhoto(false)}
            />
          ) : (
            <div className="heroFallback">
              <p className="eyebrow">Lenny Portrait</p>
              <strong>Add `apps/web/public/lenny.png`</strong>
              <span>The page is already wired to show the photo once the file is in place.</span>
            </div>
          )}
        </div>
      </section>

      <section className="panel queryPanel">
        <div className="controls">
          <label>
            Game
            <select value={game} onChange={(event) => setGame(event.target.value as "aoe2" | "lol")}>
              <option value="aoe2">Age of Empires II</option>
              <option value="lol">League of Legends</option>
            </select>
          </label>
          {game === "aoe2" && (
            <label className="toggle">
              <input
                type="checkbox"
                checked={splitDetail}
                onChange={(event) => setSplitDetail(event.target.checked)}
              />
              Test split detailed AoE2 answer
            </label>
          )}
        </div>
        <details className="tipsPanel">
          <summary>How to query better</summary>
          <div className="tipsBody">
            <div>
              <p className="tipsLabel">{guide.title}</p>
              <ul className="tipsList">
                {guide.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </div>
            <div className="tipsExamples">
              <div>
                <p className="tipsLabel">Concise examples</p>
                <ul className="tipsList">
                  {guide.concise.map((example) => (
                    <li key={example}>
                      <button
                        className="exampleButton"
                        type="button"
                        onClick={() => setQuestion(example)}
                      >
                        {example}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="tipsLabel">Detailed examples</p>
                <ul className="tipsList">
                  {guide.detailed.map((example) => (
                    <li key={example}>
                      <button
                        className="exampleButton"
                        type="button"
                        onClick={() => setQuestion(example)}
                      >
                        {example}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </details>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          rows={4}
          placeholder="Ask how to play Malay, Aatrox into Darius, Franks vs Hindustanis..."
        />
        <button
          className="primary"
          disabled={isSubmitting || question.trim().length < 2}
          onClick={() => {
            askQuestion().catch((err: unknown) => {
              if (err instanceof DOMException && err.name === "AbortError") return;
              setError(err instanceof Error ? err.message : String(err));
            });
          }}
        >
          {isSubmitting ? "Querying..." : "Ask"}
        </button>
        {error && <p className="notice error">{error}</p>}
      </section>

      {result && (
        <section className="answerGrid">
          <article className="panel answer">
            <p className="eyebrow">Answer</p>
            <p className="normalized">{result.normalized_question}</p>
            <div className="answerText">{renderAnswer(result.answer)}</div>
          </article>
          <aside className="panel sources">
            <div className="sourcesHeader">
              <div>
                <p className="eyebrow">Evidence</p>
                <p className="sourcesIntro">
                  Inspect the rows behind the answer when you want to audit where it came from.
                </p>
              </div>
              <button
                className="ghost sourceToggle"
                onClick={() => setShowSources((current) => !current)}
                type="button"
              >
                {showSources ? "Hide sources" : `Show sources (${result.sources.length})`}
              </button>
            </div>
            {showSources ? (
              result.sources.length === 0 ? (
                <p>No source rows returned.</p>
              ) : (
                <div className="sourceList">
                  {result.sources.map((line, index) => {
                    const source = parseSourceLine(line);
                    return (
                      <article className="sourceItem" key={`${index}-${line}`}>
                        <div className="sourceTopline">
                          {source.category ? <span className="sourceCategory">{source.category}</span> : null}
                          {source.metrics ? <span className="sourceMetrics">{source.metrics}</span> : null}
                        </div>
                        <p>{source.text}</p>
                      </article>
                    );
                  })}
                </div>
              )
            ) : null}
          </aside>
        </section>
      )}
    </main>
  );
}
