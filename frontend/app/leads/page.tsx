"use client";

import { useState, type FormEvent } from "react";
import { api, errorMessage, type Lead } from "@/lib/api";
import { LeadTable } from "@/components/LeadTable";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Field, fieldControlClass } from "@/components/ui/Field";

export default function LeadsPage() {
  const [phone, setPhone] = useState("");
  const [results, setResults] = useState<Lead[] | null>(null);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResults(await api.searchLeadsByPhone(phone));
      setSearched(true);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleDoNotCall(lead: Lead) {
    try {
      await api.doNotCall(lead.lead_id);
      setResults(await api.searchLeadsByPhone(phone));
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">Leads</h1>

      <Card title="Find a lead by phone number" description="Matches across every campaign — any format works (09876543210, +919876543210, ...).">
        <form onSubmit={handleSearch} className="flex max-w-md flex-col gap-4">
          <Field label="Phone">
            <input
              className={fieldControlClass}
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+91XXXXXXXXXX or 0XXXXXXXXXX"
              required
            />
          </Field>
          {error && <p className="text-sm text-status-critical">{error}</p>}
          <Button type="submit" disabled={loading} className="self-start">
            {loading ? "Searching…" : "Search"}
          </Button>
        </form>
      </Card>

      {searched && (
        <Card title="Results">
          {results && results.length === 0 ? (
            <EmptyState title="No lead found with that phone number in any campaign." />
          ) : (
            <LeadTable leads={results ?? []} showCampaignColumn onDoNotCall={handleDoNotCall} />
          )}
        </Card>
      )}
    </div>
  );
}
