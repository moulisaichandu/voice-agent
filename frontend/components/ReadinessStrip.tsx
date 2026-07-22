"use client";

import { useState } from "react";
import { api, errorMessage, type Readiness } from "@/lib/api";
import { formatRelativeTime } from "@/lib/format";
import { Button } from "./ui/Button";
import { ArrowPathIcon } from "./ui/icons";
import { Spinner } from "./ui/Spinner";
import { StatusBadge } from "./ui/StatusBadge";

/** "Can this system dial right now, and if not why" — built after a session
 * where every real blocker (a dead tunnel, ElevenLabs billing, a stale
 * PUBLIC_BASE_URL) was invisible until a call actually failed. Backed by
 * GET/POST /admin/readiness{,/refresh}; see app/admin/system.py. */
export function ReadinessStrip({
  readiness,
  onRefreshed,
}: {
  readiness: Readiness | null;
  onRefreshed: (r: Readiness) => void;
}) {
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRefresh() {
    setRefreshing(true);
    setError(null);
    try {
      onRefreshed(await api.refreshReadiness());
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setRefreshing(false);
    }
  }

  if (!readiness) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-neutral-200 p-4 text-sm text-neutral-500 dark:border-neutral-800 dark:text-neutral-400">
        <Spinner /> Checking readiness…
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900/40">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <StatusBadge
          tone={readiness.can_dial ? "good" : "critical"}
          label={readiness.can_dial ? "Ready to dial" : "Cannot dial right now"}
        />
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-500 dark:text-neutral-400">
            Checked {formatRelativeTime(readiness.checked_at)}
          </span>
          <Button variant="secondary" onClick={handleRefresh} disabled={refreshing}>
            {refreshing ? <Spinner className="h-3.5 w-3.5" /> : <ArrowPathIcon className="h-3.5 w-3.5" />}
            Re-check
          </Button>
        </div>
      </div>
      {error && <p className="mt-2 text-sm text-status-critical">{error}</p>}
      <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {readiness.checks.map((c) => (
          <li
            key={c.key}
            className="flex flex-col gap-1 rounded-md border border-neutral-200 p-2.5 dark:border-neutral-800"
          >
            <StatusBadge tone={c.status} label={c.label} />
            <span className="text-xs text-neutral-500 dark:text-neutral-400">{c.detail}</span>
            {c.cached && (
              <span className="text-[11px] text-neutral-400 dark:text-neutral-600">
                cached — {formatRelativeTime(c.checked_at)}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
