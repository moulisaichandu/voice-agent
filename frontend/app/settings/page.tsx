"use client";

import { useEffect, useState } from "react";
import {
  api,
  errorMessage,
  type ConfigVar,
  type SheetsStatus,
  type SheetsSyncResult,
} from "@/lib/api";
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
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<SheetsSyncResult | null>(null);

  async function syncNow() {
    setSyncing(true);
    setSyncResult(null);
    try {
      const result = await api.syncSheets();
      setSyncResult(result);
      // Re-read the status so the badge and "last sync" line agree with the
      // run that just happened rather than the previous sweep.
      setSheets(await api.sheetsStatus());
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setSyncing(false);
    }
  }

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
                  {sheets.last_dialable_count !== null &&
                    `, ${sheets.last_dialable_count} dialable`}
                </p>
                {/* Imported but undialable is the quiet failure: the rows are
                    there, the dialer refuses every one of them for want of
                    consent, and the sync still reads as successful. */}
                {sheets.last_synced_count !== null &&
                  sheets.last_synced_count > 0 &&
                  sheets.last_dialable_count === 0 && (
                    <p className="text-status-warning">
                      No imported lead can be dialled — check the Consent Basis and
                      Consent At columns.
                    </p>
                  )}
                {sheets.last_skips && Object.keys(sheets.last_skips).length > 0 && (
                  <p className="text-neutral-500 dark:text-neutral-400">
                    Skipped:{" "}
                    {Object.entries(sheets.last_skips)
                      .map(([reason, n]) => `${n} ${reason}`)
                      .join(", ")}
                  </p>
                )}
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

            <div className="flex items-center gap-3 pt-1">
              <Button onClick={syncNow} disabled={syncing || !sheets.configured}>
                {syncing ? "Syncing…" : "Sync now"}
              </Button>
              {syncResult && (
                <span
                  className={
                    syncResult.error ? "text-status-critical" : "text-neutral-600 dark:text-neutral-400"
                  }
                >
                  {syncResult.error
                    ? syncResult.error
                    : `${syncResult.synced ?? 0} imported, ${syncResult.dialable ?? 0} dialable`}
                </span>
              )}
            </div>
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
