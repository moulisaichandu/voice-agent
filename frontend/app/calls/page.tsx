"use client";

import { useCallback, useEffect, useState } from "react";
import { api, errorMessage, type Call } from "@/lib/api";
import { CallTable } from "@/components/CallTable";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { fieldControlClass } from "@/components/ui/Field";

const LIMIT_OPTIONS = [25, 50, 100, 200];

export default function CallsPage() {
  const [calls, setCalls] = useState<Call[]>([]);
  const [limit, setLimit] = useState(50);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setCalls(await api.recentCalls(limit));
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">Calls</h1>

      <Card
        title="Call history"
        description="Most recent calls across every campaign, newest first. Click a row to view its transcript."
        action={
          <div className="flex items-center gap-2">
            <select
              className={fieldControlClass}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            >
              {LIMIT_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  Last {n}
                </option>
              ))}
            </select>
            <Button variant="secondary" onClick={refresh} disabled={loading}>
              Refresh
            </Button>
          </div>
        }
      >
        {error && <p className="mb-3 text-sm text-status-critical">{error}</p>}
        <CallTable calls={calls} loading={loading} />
      </Card>
    </div>
  );
}
