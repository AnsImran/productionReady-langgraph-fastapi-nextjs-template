import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { SERVER_FASTAPI_BASE_URL } from "@/lib/config";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const hopByHopHeaders = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

type HandlerParams = { params: Promise<{ path?: string[] }> };

function buildUpstreamUrl(request: NextRequest, pathSegments: string[] = []) {
  const base =
    SERVER_FASTAPI_BASE_URL.endsWith("/")
      ? SERVER_FASTAPI_BASE_URL
      : `${SERVER_FASTAPI_BASE_URL}/`;

  const targetPath = pathSegments.join("/");
  const upstreamUrl = new URL(targetPath, base);
  upstreamUrl.search = request.nextUrl.search;

  return upstreamUrl;
}

function buildForwardHeaders(request: NextRequest) {
  const forwarded = new Headers(request.headers);
  forwarded.delete("host");
  forwarded.delete("content-length");
  forwarded.set("x-forwarded-host", request.headers.get("host") ?? "");
  forwarded.set("x-forwarded-proto", request.nextUrl.protocol.replace(":", ""));

  return forwarded;
}

function sanitizeResponseHeaders(source: Headers) {
  const sanitized = new Headers();
  source.forEach((value, key) => {
    if (hopByHopHeaders.has(key.toLowerCase())) {
      return;
    }
    sanitized.set(key, value);
  });
  sanitized.delete("content-length");
  return sanitized;
}

async function proxy(request: NextRequest, { params }: HandlerParams) {
  const resolvedParams = await params;
  const pathSegments = resolvedParams.path ?? [];

  if (!SERVER_FASTAPI_BASE_URL) {
    return NextResponse.json(
      { error: "FASTAPI_BASE_URL is not configured on the server." },
      { status: 500 }
    );
  }

  const upstreamUrl = buildUpstreamUrl(request, pathSegments);
  const init: RequestInit = {
    method: request.method,
    headers: buildForwardHeaders(request),
    redirect: "manual",
  };

  if (!["GET", "HEAD"].includes(request.method.toUpperCase())) {
    init.body = request.body;
    // @ts-expect-error Node.js streaming fetch requires duplex for request bodies.
    init.duplex = "half";
  }

  try {
    const upstreamResponse = await fetch(upstreamUrl, init);
    const responseHeaders = sanitizeResponseHeaders(upstreamResponse.headers);

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("FastAPI proxy error:", {
      error,
      path: upstreamUrl.pathname,
      host: request.headers.get("host"),
    });

    return NextResponse.json(
      { error: "Unable to reach FastAPI service." },
      { status: 502 }
    );
  }
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const HEAD = proxy;
export const OPTIONS = proxy;
