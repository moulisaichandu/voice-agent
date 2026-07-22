"use client";

import { useEffect, useState } from "react";
import { api, errorMessage, type ConfigVar, type SheetsStatus } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Table, TableMessageRow, Td, Th } from "@/components/ui/Table";
import { formatDateTime } from "@/lib/format";

export default function SettingsPage() {
  const [variables, setVariables] = useState<ConfigVar[] | null>(null);
  const [sheets, setSheets] = useState<SheetsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    try {
      const [cfg, sh] = await Promise.all([api.config(), api.sheetsStatus()]);
      setVariables(cfg.variables);
      setSheets(sh);
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">Settings</h1>

      {error && <p className="text-sm text-status-critical">{error}</p>}

      <Card title="Sheets sync">
        {!sheets ? (
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            {loading ? "Loading…" : "—"}
          </p>
        ) : (
          <div className="flex flex-col gap-2 text-sm">
            <StatusBadge
              tone={!sheets.configured ? "warning" : sheets.last_error ? "critical" : "good"}
              label={
                !sheets.configured
                  ? "Not configured"
                  : sheets.last_error
                    ? "Last sync failed"
                    : "Syncing normally"
              }
            />
            {sheets.configured && (
              <>
                <p className="text-neutral-600 dark:text-neutral-400">
                  Last sync: {formatDateTime(sheets.last_sync_at)}
                  {sheets.last_synced_count !== null &&
                    ` — ${sheets.last_synced_count} row(s) written`}
                </p>
                {sheets.last_error && (
                  <p className="text-status-critical">{sheets.last_error}</p>
                )}
              </>
            )}
            {!sheets.configured && (
              <p className="text-neutral-500 dark:text-neutral-400">
                GOOGLE_SERVICE_ACCOUNT_FILE / GOOGLE_SHEET_ID are not set — transcripts stay
                in Supabase only. Supabase remains the source of truth regardless; Sheets is
                a human-friendly mirror.
              </p>
            )}
          </div>
        )}
      </Card>

      <Card
        title="Configuration"
        description="Presence only for secrets — never a value. Behavioral settings show their live value."
        action={
          <Button variant="secondary" onClick={refresh} disabled={loading}>
            Refresh
          </Button>
        }
      >
        <Table>
          <thead>
            <tr>
              <Th>Variable</Th>
              <Th>Set</Th>
              <Th>Value</Th>
            </tr>
          </thead>
          <tbody>
            {loading && <TableMessageRow colSpan={3}>Loading…</TableMessageRow>}
            {!loading && variables?.length === 0 && (
              <TableMessageRow colSpan={3}>No config variables reported.</TableMessageRow>
            )}
            {!loading &&
              variables?.map((v) => (
                <tr key={v.name}>
                  <Td className="font-mono text-xs">{v.name}</Td>
                  <Td>
                    <StatusBadge
                      tone={v.is_set ? "good" : "warning"}
                      label={v.is_set ? "Set" : "Not set"}
                    />
                  </Td>
                  <Td className="font-mono text-xs">
                    {v.value === null ? <span className="text-neutral-400 dark:text-neutral-600">—</span> : String(v.value)}
                  </Td>
                </tr>
              ))}
          </tbody>
        </Table>
      </Card>
    </div>
  );
}
