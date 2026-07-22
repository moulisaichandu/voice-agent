"use client";

import Link from "next/link";
import type { Lead } from "@/lib/api";
import { ConfirmButton } from "./ConfirmButton";
import { LeadStatusBadge } from "./ui/StatusBadge";
import { Table, TableMessageRow, Td, Th } from "./ui/Table";

type Props = {
  leads: Lead[];
  loading?: boolean;
  /** Show which campaign each lead belongs to — for the cross-campaign
   * lookup page; redundant on a single campaign's own detail page. */
  showCampaignColumn?: boolean;
  /** Present only when the caller wants the do-not-call action available
   * (the campaign detail page has it; a read-only history view wouldn't). */
  onDoNotCall?: (lead: Lead) => void;
};

export function LeadTable({ leads, loading = false, showCampaignColumn = false, onDoNotCall }: Props) {
  const colCount = 5 + (showCampaignColumn ? 1 : 0) + (onDoNotCall ? 1 : 0);

  return (
    <Table>
      <thead>
        <tr>
          <Th>Name</Th>
          <Th>Phone</Th>
          {showCampaignColumn && <Th>Campaign</Th>}
          <Th>Status</Th>
          <Th>Attempts</Th>
          <Th>Consent</Th>
          {onDoNotCall && <Th />}
        </tr>
      </thead>
      <tbody>
        {loading && <TableMessageRow colSpan={colCount}>Loading…</TableMessageRow>}
        {!loading && leads.length === 0 && (
          <TableMessageRow colSpan={colCount}>No leads yet.</TableMessageRow>
        )}
        {!loading &&
          leads.map((l) => (
            <tr key={l.lead_id}>
              <Td>{l.name ?? "—"}</Td>
              <Td className="font-mono text-xs">{l.phone_e164}</Td>
              {showCampaignColumn && (
                <Td className="font-mono text-xs">
                  {l.campaign_id ? (
                    <Link href={`/campaigns/${l.campaign_id}`} className="hover:underline">
                      {l.campaign_id.slice(0, 8)}…
                    </Link>
                  ) : (
                    "—"
                  )}
                </Td>
              )}
              <Td>
                <LeadStatusBadge status={l.status} />
              </Td>
              <Td>{l.attempts}</Td>
              <Td>{l.consent_basis ?? "none"}</Td>
              {onDoNotCall && (
                <Td>
                  {l.status !== "dnd" && (
                    <ConfirmButton
                      variant="danger"
                      consequence={`Permanently opt ${l.name ?? l.phone_e164} out of every future call. This cannot be undone from here.`}
                      onConfirm={() => onDoNotCall(l)}
                    >
                      Do not call
                    </ConfirmButton>
                  )}
                </Td>
              )}
            </tr>
          ))}
      </tbody>
    </Table>
  );
}
