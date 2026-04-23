"use client";

import { useEffect, useState, useTransition } from "react";
import type { Session } from "@supabase/supabase-js";
import { hasSupabaseConfig, supabase } from "../lib/supabase";

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

export function QueryClient() {
  const [session, setSession] = useState<Session | null>(null);
  const [email, setEmail] = useState("");
  const [game, setGame] = useState<"aoe2" | "lol">("aoe2");
  const [question, setQuestion] = useState("how should I play Khmer in detail");
  const [splitDetail, setSplitDetail] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    if (!supabase) return;
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
    });
    return () => data.subscription.unsubscribe();
  }, []);

  async function signIn() {
    setError(null);
    if (!supabase) {
      setError("Supabase env vars are missing.");
      return;
    }
    const { error: signInError } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: window.location.origin }
    });
    if (signInError) setError(signInError.message);
    else setError("Magic link sent. Check your email.");
  }

  async function askQuestion() {
    setError(null);
    setResult(null);
    const token = session?.access_token;
    const response = await fetch(`${queryApiUrl}/api/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      },
      body: JSON.stringify({
        game,
        question,
        show_sources: true,
        split_detail: splitDetail
      })
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    setResult((await response.json()) as QueryResponse);
  }

  if (!hasSupabaseConfig) {
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

  if (!session) {
    return (
      <main className="shell">
        <section className="panel auth">
          <p className="eyebrow">Private knowledge base</p>
          <h1>Sign in to query your coaching data.</h1>
          <div className="authRow">
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              type="email"
            />
            <button onClick={signIn}>Send magic link</button>
          </div>
          {error && <p className="notice">{error}</p>}
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">videoSorter</p>
          <h1>Ask the game knowledge base.</h1>
          <p>
            Query League of Legends coaching data and Age of Empires II guides from one private interface.
          </p>
        </div>
        <button className="ghost" onClick={() => supabase?.auth.signOut()}>Sign out</button>
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
          <label className="toggle">
            <input
              type="checkbox"
              checked={splitDetail}
              onChange={(event) => setSplitDetail(event.target.checked)}
            />
            Test split detailed AoE2 answer
          </label>
        </div>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          rows={4}
          placeholder="Ask how to play Malay, Aatrox into Darius, Franks vs Hindustanis..."
        />
        <button
          className="primary"
          disabled={isPending || question.trim().length < 2}
          onClick={() => {
            startTransition(() => {
              askQuestion().catch((err: unknown) => {
                setError(err instanceof Error ? err.message : String(err));
              });
            });
          }}
        >
          {isPending ? "Querying..." : "Ask"}
        </button>
        {error && <p className="notice error">{error}</p>}
      </section>

      {result && (
        <section className="answerGrid">
          <article className="panel answer">
            <p className="eyebrow">Answer</p>
            <p className="normalized">{result.normalized_question}</p>
            <div className="answerText">{result.answer}</div>
          </article>
          <aside className="panel sources">
            <p className="eyebrow">Sources</p>
            {result.sources.length === 0 ? (
              <p>No source rows returned.</p>
            ) : (
              <details open>
                <summary>{result.sources.length} source lines</summary>
                <pre>{result.sources.join("\n")}</pre>
              </details>
            )}
          </aside>
        </section>
      )}
    </main>
  );
}
