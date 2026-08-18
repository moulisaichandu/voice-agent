"use client";

import Link from "next/link";
import { useEffect, useState, type FormEvent } from "react";
import {
  api,
  CAMPAIGN_LANGUAGES,
  errorMessage,
  languageLabel,
  type Campaign,
  type CampaignLanguage,
  type CampaignMode,
  type ConsentBasis,
  type LeadImportResult,
} from "@/lib/api";
import { ImportSummary, LeadsUpload } from "@/components/LeadsUpload";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Field, fieldControlClass } from "@/components/ui/Field";
import { Table, TableMessageRow, Td, Th } from "@/components/ui/Table";

/** The campaign's name comes from the file it was created with — "july-batch"
 * from "july-batch.csv". Names are not unique in the schema, but two campaigns
 * both called "leads" are impossible to tell apart in a list, so a repeat gets
 * a numeric suffix. The name still matters beyond display: app/sheets/sync.py
 * matches the Google Sheet's Campaign column to a campaign BY NAME. */
function uniqueCampaignName(filename: string, existing: Campaign[]): string {
  const base = filename.replace(/\.[^./\\]+$/, "").trim() || "Leads";
  const taken = new Set(existing.map((c) => c.name));
  if (!taken.has(base)) return base;
  for (let n = 2; n < 500; n++) {
    if (!taken.has(`${base} (${n})`)) return `${base} (${n})`;
  }
  return `${base} (${Date.now()})`;
}

export default function CampaignsPage() {
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [includeInactive, setIncludeInactive] = useState(false);
  const [togglingId, setTogglingId] = useState<string | null>(null);

  const [mode, setMode] = useState<CampaignMode>("twoway");
  const [language, setLanguage] = useState<CampaignLanguage>("auto");
  const [script, setScript] = useState("");
  const [leadsFile, setLeadsFile] = useState<File | null>(null);
  const [consentBasis, setConsentBasis] = useState<ConsentBasis | undefined>();
  const [formError, setFormError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [importResult, setImportResult] = useState<LeadImportResult | null>(null);
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
    setNotice(null);
    setImportResult(null);
    if (!leadsFile) {
      setFormError("Choose a leads file — it's what names the campaign.");
      return;
    }
    setSubmitting(true);
    try {
      const campaign = await api.createCampaign({
        name: uniqueCampaignName(leadsFile.name, campaigns),
        mode,
        language,
        script: script.trim() || undefined,
      });

      // The campaign exists from here on. A failing upload must NOT read as
      // "campaign creation failed", or the operator creates a duplicate.
      if (leadsFile) {
        try {
          setImportResult(
            await api.uploadLeads(campaign.campaign_id, leadsFile, consentBasis)
          );
          setNotice(`Created “${campaign.name}” and imported its leads.`);
        } catch (uploadErr) {
          setFormError(
            `Campaign “${campaign.name}” was created, but the leads file failed ` +
              `to import: ${errorMessage(uploadErr)} — open the campaign and try ` +
              `the file again.`
          );
        }
      } else {
        setNotice(`Created “${campaign.name}”. Open it to add leads.`);
      }

      setScript("");
      setLeadsFile(null);
      await refresh();
    } catch (e) {
      setFormError(errorMessage(e));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleToggleActive(campaign: Campaign) {
    setTogglingId(campaign.campaign_id);
    try {
      await api.setCampaignActive(campaign.campaign_id, !campaign.active);
      await refresh();
    } catch (e) {
      setLoadError(errorMessage(e));
    } finally {
      setTogglingId(null);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">Campaigns</h1>

      <Card
        title="Create a campaign"
        description="Drop in your leads file. The campaign takes its name from the file, and calls run top to bottom in the order the rows appear."
      >
        <form onSubmit={handleCreate} className="flex max-w-lg flex-col gap-4">
          <Field
            label="Language"
            hint="Every lead in this file is called in this language. A Language column in the file overrides it for that row. Telugu and Tinglish use the Sarvam voice engine because ElevenLabs does not support Telugu. The Voice column shows which backend will carry the call."
          >
            <select
              className={fieldControlClass}
              value={language}
              onChange={(e) => setLanguage(e.target.value as CampaignLanguage)}
            >
              {CAMPAIGN_LANGUAGES.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
          </Field>

          <LeadsUpload
            onChange={(file, basis) => {
              setLeadsFile(file);
              setConsentBasis(basis);
            }}
          />

          <details className="rounded-md border border-neutral-200 p-3 dark:border-neutral-800">
            <summary className="cursor-pointer text-sm font-medium text-neutral-700 dark:text-neutral-300">
              Advanced
            </summary>
            <div className="mt-4 flex flex-col gap-4">
              <Field
                label="Mode"
                hint="Two-way lets the lead ask questions, answered from your course documents. One-way plays a message and hangs up."
              >
                <select
                  className={fieldControlClass}
                  value={mode}
                  onChange={(e) => setMode(e.target.value as CampaignMode)}
                >
                  <option value="twoway">Two-way (RAG conversation)</option>
                  <option value="oneway">One-way (message + hang up)</option>
                </select>
              </Field>
              {mode === "oneway" && (
                <Field
                  label="Script"
                  hint="Required for one-way, and its opening must disclose that the caller is an AI — enforced, not advisory."
                >
                  <textarea
                    className={fieldControlClass}
                    value={script}
                    onChange={(e) => setScript(e.target.value)}
                    rows={3}
                    placeholder="This is an AI voice assistant calling on behalf of Digital Brolly..."
                  />
                </Field>
              )}
              <p className="text-xs text-neutral-500 dark:text-neutral-400">
                The ElevenLabs agent comes from your{" "}
                <code className="font-mono">.env</code> automatically, based on the mode.
              </p>
            </div>
          </details>

          {formError && <p className="text-sm text-status-critical">{formError}</p>}
          {notice && <p className="text-sm text-neutral-700 dark:text-neutral-300">{notice}</p>}
          <Button type="submit" disabled={submitting} className="self-start">
            {submitting ? "Creating…" : "Create campaign"}
          </Button>
          {importResult && <ImportSummary result={importResult} />}
        </form>
      </Card>

      <Card title="Existing campaigns">
        <label className="mb-3 flex items-center gap-2 text-sm text-neutral-600 dark:text-neutral-400">
          <input
            type="checkbox"
            checked={includeInactive}
            onChange={(e) => setIncludeInactive(e.target.checked)}
          />
          Include inactive (a dev DB that&apos;s run the integration suite holds
          hundreds of throwaway test campaigns)
        </label>
        {loadError && <p className="mb-3 text-sm text-status-critical">{loadError}</p>}
        <Table>
          <thead>
            <tr>
              <Th>Name</Th>
              <Th>Mode</Th>
              <Th>Language</Th>
              <Th>Voice</Th>
              <Th>Agent</Th>
              <Th>Active</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {loading && <TableMessageRow colSpan={7}>Loading…</TableMessageRow>}
            {!loading && campaigns.length === 0 && (
              <TableMessageRow colSpan={7}>No campaigns yet — create one above.</TableMessageRow>
            )}
            {!loading &&
              campaigns.map((c) => (
                <tr key={c.campaign_id}>
                  <Td>{c.name}</Td>
                  <Td>
                    <Badge tone={c.mode === "twoway" ? "accent" : "neutral"}>{c.mode}</Badge>
                  </Td>
                  <Td>
                    <Badge tone={c.language === "auto" ? "neutral" : "accent"}>
                      {languageLabel(c.language)}
                    </Badge>
                  </Td>
                  <Td className="text-xs text-neutral-500 dark:text-neutral-400">
                    {c.voice_backend}
                  </Td>
                  <Td className="font-mono text-xs">{c.agent_id}</Td>
                  <Td>
                    <Badge tone={c.active ? "good" : "neutral"}>{c.active ? "active" : "inactive"}</Badge>
                  </Td>
                  <Td>
                    <div className="flex items-center gap-3">
                      <Link href={`/campaigns/${c.campaign_id}`} className="text-sm text-accent hover:underline">
                        View →
                      </Link>
                      <Button
                        variant="secondary"
                        onClick={() => handleToggleActive(c)}
                        disabled={togglingId === c.campaign_id}
                      >
                        {togglingId === c.campaign_id
                          ? "…"
                          : c.active
                            ? "Deactivate"
                            : "Activate"}
                      </Button>
                    </div>
                  </Td>
                </tr>
              ))}
          </tbody>
        </Table>
      </Card>
    </div>
  );
}
