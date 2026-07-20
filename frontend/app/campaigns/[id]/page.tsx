"use client";

import { useEffect, useState, type FormEvent } from "react";
import { useParams } from "next/navigation";
import { api, errorMessage, type Call, type Lead } from "@/lib/api";

export default function CampaignDetailPage() {
  const params = useParams<{ id: string }>();
  const campaignId = params.id;

  const [leads, setLeads] = useState<Lead[]>([]);
  const [calls, setCalls] = useState<Call[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [phone, setPhone] = useState("");
  const [leadName, setLeadName] = useState("");
  const [recordConsent, setRecordConsent] = useState(true);
  const [leadError, setLeadError] = useState<string | null>(null);
  const [addingLead, setAddingLead] = useState(false);

  const [tickResult, setTickResult] = useState<string | null>(null);
  const [tickIsError, setTickIsError] = useState(false);
  const [triggering, setTriggering] = useState(false);

  async function refresh() {
    setLoading(true);
    setLoadError(null);
    try {
      const [l, c] = await Promise.all([
        api.listLeads(campaignId),
        api.listCalls(campaignId),
      ]);
      setLeads(l);
      setCalls(c);
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // campaignId comes from the route and won't change without a remount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [campaignId]);

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
              "missing consent, or attempts exhausted) — refresh below to check.")
      );
      await refresh();
    } catch (e) {
      setTickIsError(true);
      setTickResult(errorMessage(e));
    } finally {
      setTriggering(false);
    }
  }

  return (
    <div>
      <h1>Campaign</h1>
      <p className="mono">{campaignId}</p>

      <section className="card">
        <h2>Add a lead</h2>
        <form onSubmit={handleAddLead} className="form">
          <label>
            Phone
            <input
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+91XXXXXXXXXX or 0XXXXXXXXXX"
              required
            />
          </label>
          <label>
            Name
            <input value={leadName} onChange={(e) => setLeadName(e.target.value)} />
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={recordConsent}
              onChange={(e) => setRecordConsent(e.target.checked)}
            />
            Record explicit consent right now — required for this lead to ever
            be dial-eligible (app/compliance/consent.py)
          </label>
          {leadError && <p className="error">{leadError}</p>}
          <button type="submit" disabled={addingLead}>
            {addingLead ? "Adding…" : "Add lead"}
          </button>
        </form>
      </section>

      <section className="card">
        <h2>Trigger a call now</h2>
        <p className="hint">
          Runs the exact same campaign_tick the scheduler runs automatically —
          calling hours, DND, and consent are all still enforced, there is no
          test-only bypass.
        </p>
        <div className="warning-banner">
          Scoped to <strong>this campaign only</strong>. If real ElevenLabs
          credentials and a linked phone number are configured in the
          backend&apos;s .env, clicking this <strong>places real phone
          calls</strong> to this campaign&apos;s dial-eligible leads.
        </div>
        <button onClick={handleTrigger} disabled={triggering}>
          {triggering ? "Triggering…" : "Trigger campaign tick now"}
        </button>
        {tickResult && <p className={tickIsError ? "error" : ""}>{tickResult}</p>}
      </section>

      <section className="card">
        <h2>Leads</h2>
        {loading && <p>Loading…</p>}
        {loadError && <p className="error">{loadError}</p>}
        {!loading && !loadError && (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Phone</th>
                <th>Status</th>
                <th>Attempts</th>
                <th>Consent</th>
              </tr>
            </thead>
            <tbody>
              {leads.map((l) => (
                <tr key={l.lead_id}>
                  <td>{l.name ?? "—"}</td>
                  <td className="mono">{l.phone_e164}</td>
                  <td>
                    <span className={`badge badge-${l.status}`}>{l.status}</span>
                  </td>
                  <td>{l.attempts}</td>
                  <td>{l.consent_basis ?? "none"}</td>
                </tr>
              ))}
              {leads.length === 0 && (
                <tr>
                  <td colSpan={5}>No leads yet — add one above.</td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <h2>Calls</h2>
        {!loading && !loadError && (
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Turns</th>
                <th>Summary</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {calls.map((c) => (
                <tr key={c.call_id}>
                  <td>{c.status ?? "—"}</td>
                  <td>{c.turns ?? "—"}</td>
                  <td>{c.summary ?? "—"}</td>
                  <td>{c.started_at ?? "—"}</td>
                </tr>
              ))}
              {calls.length === 0 && (
                <tr>
                  <td colSpan={4}>No calls yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        )}
        {calls
          .filter((c) => c.transcript && c.transcript.length > 0)
          .map((c) => (
            <div key={c.call_id} className="transcript">
              <h3 className="mono">Transcript — {c.call_id}</h3>
              {c.transcript!.map((t, i) => (
                <p key={i} className={`turn turn-${t.role}`}>
                  <strong>{t.role}:</strong> {t.text}
                </p>
              ))}
            </div>
          ))}
      </section>

      <button className="secondary" onClick={refresh}>
        Refresh
      </button>
    </div>
  );
}
