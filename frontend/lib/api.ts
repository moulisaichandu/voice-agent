// lib/api.ts — thin fetch wrapper over the backend's /admin and /rag routes.
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

export type Campaign = {
  campaign_id: string;
  name: string;
  mode: CampaignMode;
  agent_id: string;
  script: string | null;
  max_attempts: number;
  active: boolean;
  created_at: string;
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

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
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
    mode: CampaignMode;
    agent_id: string;
    script?: string;
    max_attempts?: number;
  }) =>
    request<Campaign>("/admin/campaigns", {
      method: "POST",
      body: JSON.stringify(body),
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

  listCalls: (campaignId: string) =>
    request<Call[]>(`/admin/campaigns/${campaignId}/calls`),

  triggerTick: () =>
    request<{ queued: number }>("/admin/trigger-tick", { method: "POST" }),

  ragSearch: (query: string) =>
    request<{ result: string }>("/rag/search", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),
};

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
