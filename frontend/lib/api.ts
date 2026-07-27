// lib/api.ts — thin fetch wrapper over the backend's /admin routes.
// No auth header is sent unless NEXT_PUBLIC_API_AUTH_TOKEN is set, matching
// the backend's APP_AUTH_TOKEN: unset on both sides = open for local dev.

// Same-origin by design: /api/backend/* is proxied to the real FastAPI server
// by next.config.js's rewrite (see the reasoning in that file). Requests from
// the browser are therefore never cross-origin, so the console doesn't depend
// on the backend's ALLOWED_ORIGINS listing this exact port. Point BACKEND_URL
// (server-side, in .env.local) at the backend instead of changing this.
export const API_BASE = "/api/backend";

const AUTH_TOKEN = process.env.NEXT_PUBLIC_API_AUTH_TOKEN;

export type CampaignMode = "oneway" | "twoway";

/** Mirrors app/languages.py's TOKENS and migrations/0003's CHECK constraint.
 * "auto" sends no override to ElevenLabs and uses whatever the agent is
 * configured with — the behaviour of every campaign created before this
 * existed.
 *
 * TypeScript can't import from Python, so this union is a hand-kept copy.
 * If you add or rename a token here, also update the other three places that
 * must change together — tests/unit/test_languages.py pins the first two
 * against each other, but this union isn't reachable from Python:
 *   - app/languages.py's LANGUAGES dict (the source of truth)
 *   - app/db/models.py's CampaignLanguage Literal
 *   - migrations/0003_campaign_language.sql's two CHECK constraints
 */
export type CampaignLanguage = "auto" | "en" | "te" | "tinglish" | "hi" | "hinglish";

export const CAMPAIGN_LANGUAGES: { value: CampaignLanguage; label: string }[] = [
  { value: "auto", label: "Agent default" },
  { value: "en", label: "English" },
  { value: "te", label: "Telugu" },
  { value: "tinglish", label: "Tinglish (Telugu + English)" },
  { value: "hi", label: "Hindi" },
  { value: "hinglish", label: "Hinglish (Hindi + English)" },
];

export function languageLabel(value: string): string {
  return CAMPAIGN_LANGUAGES.find((l) => l.value === value)?.label ?? value;
}

export type Campaign = {
  campaign_id: string;
  name: string;
  mode: CampaignMode;
  agent_id: string;
  script: string | null;
  language: CampaignLanguage;
  max_attempts: number;
  active: boolean;
  created_at: string;
  /** Which voice backend actually carries this campaign, computed server-side
   * by app/db/models.py's Campaign.voice_backend. Not derived from `language`
   * here: the Telugu mapping is an operator decision (TELUGU_BACKEND in .env)
   * and a browser cannot see .env, so a client-side table would show a
   * rolled-back deployment running on a backend it no longer uses. */
  voice_backend: string;
};

export type Lead = {
  lead_id: string;
  sheet_row: number | null;
  name: string | null;
  phone_e164: string;
  language_pref: string;
  campaign_id: string | null;
  consent_basis: string | null;
  consent_at: string | null;
  dnd: boolean;
  status: string;
  attempts: number;
  last_called_at: string | null;
  created_at: string;
};

export type TranscriptTurn = { role: "agent" | "lead"; text: string };

export type Call = {
  call_id: string;
  lead_id: string | null;
  campaign_id: string | null;
  el_conversation_id: string | null;
  provider_call_id: string | null;
  mode: string | null;
  status: string | null;
  turns: number | null;
  started_at: string | null;
  ended_at: string | null;
  transcript: TranscriptTurn[] | null;
  summary: string | null;
  created_at: string;
};

export type ConsentBasis = "explicit" | "inferred";

export type LeadImportError = { row_number: number; reason: string };

export type LeadImportResult = {
  received: number;
  imported: number;
  skipped: number;
  /** Of those imported, how many pass the same consent check the dialer
   * applies. A file with no consent information imports fine and dials
   * nothing — this is how that becomes visible instead of silent. */
  dialable: number;
  errors: LeadImportError[];
};

export type ReadinessTone = "good" | "warning" | "serious" | "critical";

export type ReadinessCheck = {
  key: string;
  status: ReadinessTone;
  label: string;
  detail: string;
  cached: boolean;
  checked_at: string;
};

export type Readiness = {
  can_dial: boolean;
  checked_at: string;
  checks: ReadinessCheck[];
};

export type Stats = {
  leads_by_status: Record<string, number>;
  calls_today: number;
  calls_today_with_transcript: number;
};

export type ConfigVar = {
  name: string;
  is_set: boolean;
  value: string | number | boolean | null;
};

export type SheetsStatus = {
  configured: boolean;
  last_sync_at: string | null;
  last_synced_count: number | null;
  last_error: string | null;
};

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // A FormData body must NOT carry a hand-set Content-Type: the browser has to
  // generate `multipart/form-data; boundary=...` itself, and an explicit
  // application/json here makes the server fail to find any boundary at all.
  const isFormData =
    typeof FormData !== "undefined" && init?.body instanceof FormData;
  const headers: Record<string, string> = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(init?.headers as Record<string, string> | undefined),
  };
  if (AUTH_TOKEN) headers["Authorization"] = `Bearer ${AUTH_TOKEN}`;

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    // Requests go through this Next server's proxy (next.config.js), so a
    // throw here means the proxy couldn't reach the backend — not a CORS
    // rejection, which this same branch used to report misleadingly as
    // "is the backend running?".
    throw new ApiError(
      0,
      `The console's proxy could not reach the backend. Check it's up ` +
        `(docker compose ps, or curl http://localhost:8091/health) and that ` +
        `BACKEND_URL in frontend/.env.local points at it. Note BACKEND_URL is ` +
        `read at startup — restart 'npm run dev' after changing it.`
    );
  }

  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body;
    } catch {
      // non-JSON error body — fall back to statusText
    }
    throw new ApiError(
      res.status,
      typeof detail === "string" ? detail : JSON.stringify(detail)
    );
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  listCampaigns: (includeInactive = false) =>
    request<Campaign[]>(
      `/admin/campaigns${includeInactive ? "?include_inactive=true" : ""}`
    ),

  createCampaign: (body: {
    name: string;
    /** Defaults to twoway server-side — the safer of the two. Still admin-set
     * at creation time, never inferred per call. */
    mode?: CampaignMode;
    /** Omit to use the agent configured for the mode in the backend's .env.
     * Kept optional rather than removed so an explicit id still works. */
    agent_id?: string;
    script?: string;
    /** Omit for "auto" — no override, the agent's own language. */
    language?: CampaignLanguage;
    max_attempts?: number;
  }) =>
    request<Campaign>("/admin/campaigns", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  setCampaignActive: (campaignId: string, active: boolean) =>
    request<Campaign>(`/admin/campaigns/${campaignId}`, {
      method: "PATCH",
      body: JSON.stringify({ active }),
    }),

  listLeads: (campaignId: string) =>
    request<Lead[]>(`/admin/campaigns/${campaignId}/leads`),

  addLead: (
    campaignId: string,
    body: {
      phone: string;
      name?: string;
      language_pref?: string;
      consent_basis?: "explicit" | "inferred";
      consent_at?: string;
    }
  ) =>
    request<Lead>(`/admin/campaigns/${campaignId}/leads`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  uploadLeads: (campaignId: string, file: File, consentBasis?: ConsentBasis) => {
    const form = new FormData();
    form.append("file", file);
    // Only sent when the operator actually affirmed it. Omitting the field
    // means those rows import with no consent and are never dialled, which is
    // the safe default — see the endpoint's docstring.
    if (consentBasis) form.append("consent_basis", consentBasis);
    return request<LeadImportResult>(
      `/admin/campaigns/${campaignId}/leads/upload`,
      { method: "POST", body: form }
    );
  },

  searchLeadsByPhone: (phone: string) =>
    request<Lead[]>(`/admin/leads?phone=${encodeURIComponent(phone)}`),

  leadCalls: (leadId: string) => request<Call[]>(`/admin/leads/${leadId}/calls`),

  doNotCall: (leadId: string) =>
    request<Lead>(`/admin/leads/${leadId}/do-not-call`, { method: "POST" }),

  listCalls: (campaignId: string) =>
    request<Call[]>(`/admin/campaigns/${campaignId}/calls`),

  recentCalls: (limit = 50) => request<Call[]>(`/admin/calls?limit=${limit}`),

  // Scoped to one campaign on purpose — see the backend route's docstring.
  // The unscoped /admin/trigger-tick exists but is deliberately not exposed
  // in the UI: a "dial now" button on a campaign page must not call another
  // campaign's leads.
  triggerTick: (campaignId: string) =>
    request<{ queued: number }>(`/admin/campaigns/${campaignId}/trigger-tick`, {
      method: "POST",
    }),

  readiness: () => request<Readiness>("/admin/readiness"),
  refreshReadiness: () =>
    request<Readiness>("/admin/readiness/refresh", { method: "POST" }),

  stats: () => request<Stats>("/admin/stats"),

  config: () => request<{ variables: ConfigVar[] }>("/admin/config"),

  sheetsStatus: () => request<SheetsStatus>("/admin/sheets-status"),

};

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
