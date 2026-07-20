/**
 * The browser never calls the FastAPI backend directly — it calls this Next
 * server at /api/backend/*, which proxies to BACKEND_URL server-side.
 *
 * That's deliberate: a direct browser->FastAPI call is cross-origin (console
 * on :3200, backend on :8091), so it only works if the backend's
 * ALLOWED_ORIGINS happens to list this exact console port. When it doesn't,
 * the browser blocks the response and the failure surfaces as an opaque
 * "failed to fetch" that looks identical to the backend being down. Proxying
 * makes every request same-origin, so the console works regardless of how
 * ALLOWED_ORIGINS is configured, and one less thing has to be kept in sync.
 *
 * Server-to-server callers (the ElevenLabs RAG tool, the transcript webhook)
 * still hit the backend directly and are unaffected by any of this — CORS is
 * a browser mechanism only.
 */
const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8091";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: "/api/backend/:path*", destination: `${BACKEND_URL}/:path*` },
    ];
  },
};

module.exports = nextConfig;
