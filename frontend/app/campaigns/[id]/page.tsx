"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";
import { useParams } from "next/navigation";
import {
  api,
  errorMessage,
  languageLabel,
  type Call,
  type Campaign,
  type Lead,
  type Readiness,
} from "@/lib/api";
import { ConfirmButton } from "@/components/ConfirmButton";
import { CallTable } from "@/components/CallTable";
import { LeadsUpload } from "@/components/LeadsUpload";
import { LeadTable } from "@/components/LeadTable";
import { ReadinessStrip } from "@/components/ReadinessStrip";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Field, fieldControlClass } from "@/components/ui/Field";
import { usePolling } from "@/lib/usePolling";

const POLL_INTERVAL_MS = 3000;

export default function CampaignDetailPage() {
  const params = useParams<{ id: string }>();
  const campaignId = params.id;

  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [leads, setLeads] = useState<Lead[]>([]);
  const [calls, setCalls] = useState<Call[]>([]);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [togglingActive, setTogglingActive] = useState(false);

  const [phone, setPhone] = useState("");
  const [leadName, setLeadName] = useState("");
  const [recordConsent, setRecordConsent] = useState(true);
  const [leadError, setLeadError] = useState<string | null>(null);
  const [addingLead, setAddingLead] = useState(false);

  const [tickResult, setTickResult] = useState<string | null>(null);
  const [tickIsError, setTickIsError] = useState(false);
  const [triggering, setTriggering] = useState(false);

  const refresh = useCallback(async () => {
    try {
      // Campaign is fetched by listing (no single-campaign GET endpoint
      // exists) — cheap enough at this project's scale, and it's what
      // confirms the id in the URL actually resolves to something.
      const [all, l, c] = await Promise.all([
        api.listCampaigns(true),
        api.listLeads(campaignId),
        api.listCalls(campaignId),
      ]);
      setCampaign(all.find((x) => x.campaign_id === campaignId) ?? null);
      setLeads(l);
      setCalls(c);
      setLoadError(null);
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }, [campaignId]);

  async function refreshReadiness() {
    try {
      setReadiness(await api.readiness());
    } catch {
      // Non-fatal — the strip just stays in its "checking" state; the page's
      // other data doesn't depend on this.
    }
  }

  useEffect(() => {
    refresh();
    refreshReadiness();
    // campaignId comes from the route and won't change without a remount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [campaignId]);

  // Auto-refresh only while something is actually in flight — a lead sitting
  // 'pending' isn't going to change on its own, so polling then would just
  // be noise (and unnecessary backend load).
  const anyInFlight = leads.some((l) => l.status === "queued" || l.status === "calling");
  usePolling(refresh, POLL_INTERVAL_MS, anyInFlight);

  async function handleAddLead(e: FormEvent) {
    e.preventDefault();
    setLeadError(null);
    setAddingLead(true);
    try {
      await api.addLead(campaignId, {
        phone,
        name: leadName || undefined,
        consent_basis: recordConsent ? "explicit" : undefined,
        consent_at: recordConsent ? new Date().toISOString() : undefined,
      });
      setPhone("");
      setLeadName("");
      await refresh();
    } catch (e) {
      setLeadError(errorMessage(e));
    } finally {
      setAddingLead(false);
    }
  }

  async function handleDoNotCall(lead: Lead) {
    try {
      await api.doNotCall(lead.lead_id);
      await refresh();
    } catch (e) {
      setLoadError(errorMessage(e));
    }
  }

  async function handleToggleActive() {
    if (!campaign) return;
    setTogglingActive(true);
    try {
      await api.setCampaignActive(campaign.campaign_id, !campaign.active);
      await refresh();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setTogglingActive(false);
    }
  }

  async function handleTrigger() {
    setTriggering(true);
    setTickResult(null);
    setTickIsError(false);
    try {
      const result = await api.triggerTick(campaignId);
      setTickResult(
        `Queued ${result.queued} lead(s) from this campaign. ` +
          (result.queued > 0
            ? "The in-process worker picks these up within a couple of seconds."
            : "Nothing was dial-eligible right now (outside calling hours, DND, " +
              "missing consent, or attempts exhausted) — check the readiness panel above.")
      );
      await Promise.all([refresh(), refreshReadiness()]);
    } catch (e) {
      setTickIsError(true);
      setTickResult(errorMessage(e));
    } finally {
      setTriggering(false);
    }
  }

  const dialEligibleCount = leads.filter(
    (l) => l.status === "pending" && !l.dnd && l.consent_basis
  ).length;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
          {campaign?.name ?? "Campaign"}
        </h1>
        <p className="font-mono text-xs text-neutral-500 dark:text-neutral-400">{campaignId}</p>
      </div>

      {campaign && (
        <div className="flex flex-wrap items-center gap-3">
          <Badge tone={campaign.mode === "twoway" ? "accent" : "neutral"}>{campaign.mode}</Badge>
          <Badge tone={campaign.language === "auto" ? "neutral" : "accent"}>
            {languageLabel(campaign.language)}
          </Badge>
          <Badge tone={campaign.active ? "good" : "neutral"}>
            {campaign.active ? "active" : "inactive"}
          </Badge>
          <Button variant="secondary" onClick={handleToggleActive} disabled={togglingActive}>
            {togglingActive ? "…" : campaign.active ? "Deactivate campaign" : "Activate campaign"}
          </Button>
        </div>
      )}

      <ReadinessStrip readiness={readiness} onRefreshed={setReadiness} />

      <Card
        title="Import leads from a file"
        description="CSV or Excel. Re-uploading the same file is safe — leads are matched by phone number and updated, not duplicated."
      >
        <LeadsUpload campaignId={campaignId} onImported={refresh} />
      </Card>

      <Card title="Add a single lead">
        <form onSubmit={handleAddLead} className="flex max-w-md flex-col gap-4">
          <Field label="Phone">
            <input
              className={fieldControlClass}
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+91XXXXXXXXXX or 0XXXXXXXXXX"
              required
            />
          </Field>
          <Field label="Name">
            <input
              className={fieldControlClass}
              value={leadName}
              onChange={(e) => setLeadName(e.target.value)}
            />
          </Field>
          <Field label="Record explicit consent right now" inline>
            <input
              type="checkbox"
              checked={recordConsent}
              onChange={(e) => setRecordConsent(e.target.checked)}
            />
          </Field>
          <p className="text-xs text-neutral-500 dark:text-neutral-400">
            Required for this lead to ever be dial-eligible (app/compliance/consent.py).
          </p>
          {leadError && <p className="text-sm text-status-critical">{leadError}</p>}
          <Button type="submit" disabled={addingLead}>
            {addingLead ? "Adding…" : "Add lead"}
          </Button>
        </form>
      </Card>

      <Card title="Trigger a call now">
        <p className="mb-3 text-sm text-neutral-500 dark:text-neutral-400">
          Runs the exact same campaign_tick the scheduler runs automatically —
          calling hours, DND, and consent are all still enforced; there is no
          test-only bypass. {dialEligibleCount} lead(s) here currently look
          dial-eligible on the surface (final gates are re-checked at dial time).
        </p>
        <ConfirmButton
          onConfirm={handleTrigger}
          disabled={triggering}
          consequence={
            readiness && !readiness.can_dial
              ? "The readiness panel above reports this system cannot dial right now — this will still run, and will queue nothing until that's resolved."
              : "This places REAL phone calls to this campaign's dial-eligible leads if live credentials are configured."
          }
        >
          {triggering ? "Triggering…" : "Trigger campaign tick now"}
        </ConfirmButton>
        {tickResult && (
          <p className={`mt-3 text-sm ${tickIsError ? "text-status-critical" : ""}`}>{tickResult}</p>
        )}
      </Card>

      <Card title="Leads">
        {loadError && <p className="mb-3 text-sm text-status-critical">{loadError}</p>}
        <LeadTable leads={leads} loading={loading} onDoNotCall={handleDoNotCall} />
      </Card>

      <Card title="Calls">
        <CallTable calls={calls} loading={loading} />
      </Card>

      <Button variant="secondary" onClick={refresh} className="self-start">
        Refresh
      </Button>
    </div>
  );
}
