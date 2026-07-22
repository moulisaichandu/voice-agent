"use client";

import { useCallback, useEffect, useState } from "react";
import { api, errorMessage, type Call, type Readiness, type Stats } from "@/lib/api";
import { CallTable } from "@/components/CallTable";
import { ReadinessStrip } from "@/components/ReadinessStrip";
import { StatTile } from "@/components/StatTile";
import { Card } from "@/components/ui/Card";
import { usePolling } from "@/lib/usePolling";

const POLL_INTERVAL_MS = 10000;

// Fixed order regardless of which statuses currently have leads — a pipeline
// stage reads as "0 right now", not as absent, and the order should always
// read left-to-right as a lead's actual lifecycle (see LEAD_STATUS_META).
const PIPELINE_STAGES: { key: string; label: string }[] = [
  { key: "pending", label: "Pending" },
  { key: "queued", label: "Queued" },
  { key: "calling", label: "Calling" },
  { key: "done", label: "Done" },
  { key: "failed", label: "Failed" },
  { key: "dnd", label: "Do not call" },
];

export default function DashboardPage() {
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [recentCalls, setRecentCalls] = useState<Call[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [r, s, c] = await Promise.all([
        api.readiness(),
        api.stats(),
        api.recentCalls(10),
      ]);
      setReadiness(r);
      setStats(s);
      setRecentCalls(c);
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  usePolling(refresh, POLL_INTERVAL_MS, true);

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">Dashboard</h1>

      {error && <p className="text-sm text-status-critical">{error}</p>}

      <ReadinessStrip readiness={readiness} onRefreshed={setReadiness} />

      <Card title="Lead pipeline" description="Across active campaigns only — deactivated test campaigns don't count.">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {PIPELINE_STAGES.map((stage) => (
            <StatTile
              key={stage.key}
              label={stage.label}
              value={loading ? "…" : (stats?.leads_by_status[stage.key] ?? 0)}
            />
          ))}
        </div>
      </Card>

      <Card title="Today">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <StatTile label="Calls today" value={loading ? "…" : (stats?.calls_today ?? 0)} />
          <StatTile
            label="With transcript"
            value={loading ? "…" : (stats?.calls_today_with_transcript ?? 0)}
            hint="Two-way calls that produced a recorded turn"
          />
        </div>
      </Card>

      <Card title="Recent calls" description="Most recent 10, across every campaign.">
        <CallTable calls={recentCalls} loading={loading} />
      </Card>
    </div>
  );
}
