"use client";

import { useState, type FormEvent } from "react";
import { api, errorMessage } from "@/lib/api";

export default function RagSearchPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.ragSearch(query);
      setResult(res.result);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>RAG Search</h1>
      <p className="hint">
        Calls the same <code>/rag/search</code> endpoint the live two-way
        agent uses as a tool — filtered by <code>RAG_MIN_SCORE</code>, so an
        unrelated question should come back as &quot;no relevant
        material&quot; rather than a hallucinated answer. If nothing was ever
        ingested (<code>python scripts/ingest_docs.py</code>), every question
        will return that fallback.
      </p>
      <form onSubmit={handleSubmit} className="form">
        <label>
          Question
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            required
            minLength={1}
            maxLength={512}
            placeholder="What's the fee for the digital marketing course?"
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {result !== null && (
        <section className="card">
          <h2>Result</h2>
          <p>{result || <em>(empty)</em>}</p>
        </section>
      )}
    </div>
  );
}
