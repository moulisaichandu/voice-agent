"use client";

import { Fragment, useState } from "react";
import type { Call } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import { CallStatusBadge } from "./ui/StatusBadge";
import { Table, TableMessageRow, Td, Th } from "./ui/Table";
import { TranscriptView } from "./TranscriptView";

export function CallTable({ calls, loading = false }: { calls: Call[]; loading?: boolean }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const colCount = 4;

  function toggle(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <Table>
      <thead>
        <tr>
          <Th>Status</Th>
          <Th>Turns</Th>
          <Th>Summary</Th>
          <Th>Started</Th>
        </tr>
      </thead>
      <tbody>
        {loading && <TableMessageRow colSpan={colCount}>Loading…</TableMessageRow>}
        {!loading && calls.length === 0 && (
          <TableMessageRow colSpan={colCount}>No calls yet.</TableMessageRow>
        )}
        {!loading &&
          calls.map((c) => {
            const hasTranscript = !!c.transcript?.length;
            const isOpen = expanded.has(c.call_id);
            return (
              <Fragment key={c.call_id}>
                <tr
                  className={hasTranscript ? "cursor-pointer hover:bg-neutral-50 dark:hover:bg-neutral-900/60" : ""}
                  onClick={() => hasTranscript && toggle(c.call_id)}
                  title={hasTranscript ? (isOpen ? "Click to hide transcript" : "Click to view transcript") : undefined}
                >
                  <Td>
                    <CallStatusBadge status={c.status} />
                  </Td>
                  <Td>{c.turns ?? "—"}</Td>
                  <Td className="max-w-xs truncate">{c.summary ?? "—"}</Td>
                  <Td>{formatDateTime(c.started_at)}</Td>
                </tr>
                {hasTranscript && isOpen && (
                  <tr>
                    <td colSpan={colCount} className="border-b border-neutral-100 p-0 dark:border-neutral-900">
                      <TranscriptView turns={c.transcript!} />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
      </tbody>
    </Table>
  );
}
