"use client";

import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { api, errorMessage, type Campaign, type CampaignMode } from "@/lib/api";

export default function CampaignsPage() {
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [includeInactive, setIncludeInactive] = useState(false);

  const [name, setName] = useState("");
  const [mode, setMode] = useState<CampaignMode>("twoway");
  const [agentId, setAgentId] = useState("");
  const [script, setScript] = useState("");
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function refresh(inactive = includeInactive) {
    setLoading(true);
    setLoadError(null);
    try {
      setCampaigns(await api.listCampaigns(inactive));
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh(includeInactive);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [includeInactive]);

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    setFormError(null);
    setSubmitting(true);
    try {
      await api.createCampaign({
        name,
        mode,
        agent_id: agentId,
        script: script.trim() || undefined,
      });
      setName("");
      setAgentId("");
      setScript("");
      await refresh();
    } catch (e) {
      setFormError(errorMessage(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <h1>Campaigns</h1>

      <section className="card">
        <h2>Create a campaign</h2>
        <form onSubmit={handleCreate} className="form">
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} required />
          </label>
          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value as CampaignMode)}>
              <option value="twoway">Two-way (RAG conversation)</option>
              <option value="oneway">One-way (message + hang up)</option>
            </select>
          </label>
          <label>
            ElevenLabs agent ID
            <input
              value={agentId}
              onChange={(e) => setAgentId(e.target.value)}
              placeholder="agent_..."
              required
            />
          </label>
          <label>
            Script{" "}
            {mode === "oneway"
              ? "(required — first sentence must disclose AI, e.g. contain the word AI)"
              : "(optional for two-way)"}
            <textarea
              value={script}
              onChange={(e) => setScript(e.target.value)}
              rows={3}
              placeholder="This is an AI voice assistant calling on behalf of Digital Brolly..."
            />
          </label>
          {formError && <p className="error">{formError}</p>}
          <button type="submit" disabled={submitting}>
            {submitting ? "Creating…" : "Create campaign"}
          </button>
        </form>
      </section>

      <section className="card">
        <h2>Existing campaigns</h2>
        <label className="checkbox" style={{ marginBottom: "0.75rem" }}>
          <input
            type="checkbox"
            checked={includeInactive}
            onChange={(e) => setIncludeInactive(e.target.checked)}
          />
          Include inactive (a dev DB that&apos;s run the integration suite holds
          hundreds of throwaway test campaigns)
        </label>
        {loading && <p>Loading…</p>}
        {loadError && <p className="error">{loadError}</p>}
        {!loading && !loadError && campaigns.length === 0 && (
          <p>No campaigns yet — create one above.</p>
        )}
        {campaigns.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Mode</th>
                <th>Agent</th>
                <th>Active</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {campaigns.map((c) => (
                <tr key={c.campaign_id}>
                  <td>{c.name}</td>
                  <td>{c.mode}</td>
                  <td className="mono">{c.agent_id}</td>
                  <td>{c.active ? "yes" : "no"}</td>
                  <td>
                    <Link href={`/campaigns/${c.campaign_id}`}>View →</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
