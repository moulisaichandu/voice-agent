const BACKEND_URL = (process.env.BACKEND_URL ?? "http://localhost:8091").replace(/\/$/, "");
// BACKEND_AUTH_TOKEN is the supported name. The legacy fallback lets an
// existing local console keep working during migration; it is read only in
// this server route and is never attached by browser JavaScript. Remove the
// NEXT_PUBLIC variable from frontend/.env.local after moving its value.
const BACKEND_AUTH_TOKEN =
  process.env.BACKEND_AUTH_TOKEN ?? process.env.NEXT_PUBLIC_API_AUTH_TOKEN;

// Mirrors MAX_UPLOAD_BYTES in app/config.py. The backend's own check protects
// FastAPI, but the browser posts to THIS route, so without a guard here a
// multi-GB upload is buffered in the Next server and OOMs the console before
// FastAPI ever sees it. Keep the two in step; override both if you raise one.
const MAX_UPLOAD_BYTES = Number(process.env.MAX_UPLOAD_BYTES ?? 10 * 1024 * 1024);

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "transfer-encoding",
]);

async function proxy(
  request: Request,
  { params }: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await params;
  const incomingUrl = new URL(request.url);
  const target = `${BACKEND_URL}/${path.map(encodeURIComponent).join("/")}${incomingUrl.search}`;
  const headers = new Headers(request.headers);

  for (const name of HOP_BY_HOP_HEADERS) headers.delete(name);
  headers.delete("authorization");
  if (BACKEND_AUTH_TOKEN) {
    headers.set("authorization", `Bearer ${BACKEND_AUTH_TOKEN}`);
  }

  const hasBody = request.method !== "GET" && request.method !== "HEAD";

  // Reject on the declared length BEFORE touching the body. A client that lies
  // about content-length still cannot blow memory here, because the body is
  // streamed through rather than buffered.
  if (hasBody) {
    const declared = Number(request.headers.get("content-length") ?? "0");
    if (Number.isFinite(declared) && declared > MAX_UPLOAD_BYTES) {
      return Response.json(
        {
          detail:
            `the uploaded file is too large; maximum is ${MAX_UPLOAD_BYTES} bytes`,
        },
        { status: 413 },
      );
    }
  }

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      // Streamed, not buffered: `await request.arrayBuffer()` materialised the
      // WHOLE multipart body in this process first, so the backend's upload cap
      // protected FastAPI while the console itself could be OOMed by a single
      // large POST. `duplex` is required by undici whenever body is a stream
      // and is not yet in the DOM RequestInit type.
      body: hasBody ? request.body : undefined,
      ...(hasBody ? { duplex: "half" } : {}),
      redirect: "manual",
    } as RequestInit & { duplex?: "half" });
    const responseHeaders = new Headers(upstream.headers);
    for (const name of HOP_BY_HOP_HEADERS) responseHeaders.delete(name);
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch {
    return Response.json(
      { detail: "The backend proxy could not reach FastAPI." },
      { status: 502 },
    );
  }
}

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const PUT = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
