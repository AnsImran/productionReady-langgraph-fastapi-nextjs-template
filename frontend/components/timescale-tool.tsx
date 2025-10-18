"use client";

import type { ToolUIPart } from "ai";
import type { ReactNode } from "react";
import { useState } from "react";
import { Tool, ToolContent, ToolHeader, ToolInput, ToolOutput } from "./elements/tool";
import { Badge } from "./ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";

export type TimescaleToolPart = {
  type: string;
  state?: string;
  toolCallId?: string;
  toolName?: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
};

type TimescaleDoc = {
  id: string;
  title?: string;
  snippet: string;
  url?: string;
  mainService?: string;
  subService?: string;
  score?: number;
  metadata?: Record<string, unknown>;
};

type ParsedTimescaleOutput = {
  docs: TimescaleDoc[];
  rawFallback?: string;
};

const METADATA_HIDE_KEYS = new Set([
  "title",
  "url",
  "doc_url",
  "main_service_name",
  "sub_service_name",
]);

export function TimescaleTool({ part }: { part: TimescaleToolPart }) {
  const headerState = normalizeState(part.state);
  const headerType =
    part.type === "dynamic-tool"
      ? `tool-${part.toolName ?? "get_docs_pgvector"}`
      : part.type;

  const { docs, rawFallback } = parseTimescaleOutput(part.output);

  const errorText =
    headerState === "output-error"
      ? part.errorText ?? deriveErrorText(part.output)
      : undefined;

  let outputNode: ReactNode | undefined;
  if (docs.length > 0) {
    outputNode = <TimescaleDocList docs={docs} />;
  } else if (rawFallback) {
    outputNode = (
      <pre className="whitespace-pre-wrap text-xs">{rawFallback}</pre>
    );
  } else if (headerState === "output-streaming") {
    outputNode = (
      <div className="text-xs text-muted-foreground">
        Streaming documentation results…
      </div>
    );
  }

  const hasInput = part.input !== null && part.input !== undefined;
  const shouldShowInput =
    hasInput &&
    (headerState === "input-available" ||
      headerState === "output-streaming" ||
      headerState === "output-available");

  const shouldShowOutput = Boolean(outputNode || errorText);

  return (
    <Tool defaultOpen={false} key={part.toolCallId ?? headerType}>
      <ToolHeader state={headerState as any} type={headerType as any} />
      <ToolContent>
        {shouldShowInput && (
          <ToolInput input={part.input as ToolUIPart["input"]} />
        )}
        {shouldShowOutput && (
          <ToolOutput
            className="pt-0"
            errorText={errorText as any}
            output={outputNode}
          />
        )}
        {!shouldShowOutput && (
          <div className="px-4 pb-4 text-xs text-muted-foreground">
            Searching documentation…
          </div>
        )}
      </ToolContent>
    </Tool>
  );
}

function TimescaleDocList({ docs }: { docs: TimescaleDoc[] }) {
  return (
    <div className="flex flex-col gap-3">
      {docs.map((doc, index) => (
        <TimescaleDocCard doc={doc} key={doc.id} order={index + 1} />
      ))}
    </div>
  );
}

function TimescaleDocCard({ doc, order }: { doc: TimescaleDoc; order: number }) {
  const [open, setOpen] = useState(false);

  const metaEntries =
    doc.metadata && Object.keys(doc.metadata).length > 0
      ? Object.entries(doc.metadata).filter(
          ([key, value]) => !METADATA_HIDE_KEYS.has(key) && value != null,
        )
      : [];

  return (
    <Card>
      <CardHeader
        className="cursor-pointer gap-2"
        onClick={() => setOpen((prev) => !prev)}
        role="button"
      >
        <CardTitle className="flex items-start justify-between gap-3 text-base font-semibold">
          <span className="flex-1">
            {doc.title ?? `Result ${order}`}
          </span>
          {typeof doc.score === "number" && (
            <Badge variant="outline" className="shrink-0">
              Score {doc.score.toFixed(2)}
            </Badge>
          )}
        </CardTitle>
        <CardDescription className="flex flex-wrap items-center gap-2 text-xs">
          <Badge variant="secondary">Result {order}</Badge>
          {doc.mainService && (
            <Badge variant="secondary">{doc.mainService}</Badge>
          )}
          {doc.subService && (
            <Badge variant="secondary">{doc.subService}</Badge>
          )}
          <span className="ml-auto text-muted-foreground">
            {open ? "Hide details" : "Show details"}
          </span>
        </CardDescription>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p className="whitespace-pre-wrap text-foreground">{doc.snippet}</p>
          {doc.url && (
            <a
              className="inline-flex items-center gap-1 text-xs font-medium text-primary"
              href={doc.url}
              rel="noreferrer"
              target="_blank"
            >
              View full documentation →
            </a>
          )}
          {metaEntries.length > 0 && (
            <dl className="grid gap-2 text-xs sm:grid-cols-2">
              {metaEntries.slice(0, 8).map(([key, value]) => (
                <div key={key} className="flex flex-col gap-0.5">
                  <dt className="font-medium text-muted-foreground uppercase tracking-wide">
                    {key}
                  </dt>
                  <dd className="text-foreground">{formatMetaValue(value)}</dd>
                </div>
              ))}
            </dl>
          )}
        </CardContent>
      )}
    </Card>
  );
}

function parseTimescaleOutput(output: unknown): ParsedTimescaleOutput {
  if (output == null) {
    return { docs: [] };
  }

  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return { docs: [] };

    try {
      const parsed = JSON.parse(trimmed);
      return parseTimescaleOutput(parsed);
    } catch {
      const docsFromText = extractDocsFromText(trimmed);
      return {
        docs: docsFromText,
        rawFallback: docsFromText.length === 0 ? trimmed : undefined,
      };
    }
  }

  if (Array.isArray(output)) {
    const docs = output.flatMap((entry, index) =>
      collectDocs(entry, index)
    );

    return {
      docs,
      rawFallback:
        docs.length === 0
          ? safeStringify(output)
          : undefined,
    };
  }

  if (typeof output === "object") {
    const docs = collectDocs(output, 0);

    return {
      docs,
      rawFallback:
        docs.length === 0 ? safeStringify(output) : undefined,
    };
  }

  return {
    docs: [
      {
        id: "item-1",
        snippet: String(output),
      },
    ],
  };
}

function collectDocs(entry: unknown, index: number): TimescaleDoc[] {
  if (entry == null) return [];

  if (Array.isArray(entry)) {
    return entry.flatMap((item, idx) => collectDocs(item, idx));
  }

  if (typeof entry !== "object") {
    return [
      {
        id: `item-${index + 1}`,
        snippet: String(entry),
      },
    ];
  }

  const record = entry as Record<string, unknown>;

  const nestedKeys = ["documents", "docs", "items", "results", "chunks", "data"];
  for (const key of nestedKeys) {
    if (key in record) {
      const nested = record[key];
      const docs = collectDocs(nested, 0);
      if (docs.length > 0) {
        return docs;
      }
    }
  }

  return [buildDocFromRecord(record, index)];
}

function extractDocsFromText(text: string): TimescaleDoc[] {
  const blocks =
    text.match(/\[\d+\][\s\S]*?(?=(?:\n{2,}\[\d+\])|$)/g) ?? [];

  if (blocks.length === 0) {
    return [];
  }

  return blocks.map((block, idx) => {
    const identifierMatch = block.match(/\[(\d+)\]/);
    const idFromText = identifierMatch
      ? `item-${identifierMatch[1]}`
      : `item-${idx + 1}`;

    const jsonMatch = block.match(/\{[\s\S]*?\}/);

    let metadata: Record<string, unknown> | undefined;
    if (jsonMatch) {
      try {
        metadata = JSON.parse(jsonMatch[0]);
      } catch {
        metadata = undefined;
      }
    }

    const contentWithoutJson = jsonMatch
      ? block.replace(jsonMatch[0], "")
      : block;

    const snippet = contentWithoutJson
      .replace(/\[\d+\]\s*/, "")
      .trim();

    return buildDocFromRecord(
      {
        id: idFromText,
        snippet,
        metadata,
      },
      idx
    );
  });
}

function buildDocFromRecord(
  record: Record<string, unknown>,
  index: number
): TimescaleDoc {
  const metadata = normalizeMeta(record.metadata ?? record.meta);

  const id =
    firstString(
      record.id,
      record.doc_id,
      record.document_id,
      metadata?.id,
      metadata?.doc_id
    ) ?? `item-${index + 1}`;

  const title = firstString(record.title, metadata?.title);
  const url = firstString(
    record.url,
    record.doc_url,
    metadata?.url,
    metadata?.doc_url
  );

  const mainService = firstString(
    record.main_service_name,
    metadata?.main_service_name
  );
  const subService = firstString(
    record.sub_service_name,
    metadata?.sub_service_name
  );

  const score = firstNumber(record.score ?? metadata?.score);

  const snippet = deriveSnippet(record, metadata);

  const combinedMeta = collectMetadata(record, metadata);

  return {
    id,
    title,
    url,
    mainService,
    subService,
    score,
    snippet,
    metadata: Object.keys(combinedMeta).length ? combinedMeta : undefined,
  };
}

function deriveSnippet(
  record: Record<string, unknown>,
  metadata?: Record<string, unknown>
): string {
  const source =
    firstAvailable(
      record.snippet,
      record.content,
      record.text,
      record.body,
      metadata?.snippet,
      metadata?.content
    ) ?? record;

  if (typeof source === "string") {
    return source.trim();
  }

  if (Array.isArray(source)) {
    return source.map(formatMetaValue).join("\n");
  }

  if (typeof source === "object" && source != null) {
    return safeStringify(source) ?? "";
  }

  return String(source ?? "");
}

function collectMetadata(
  record: Record<string, unknown>,
  metadata?: Record<string, unknown>
) {
  const meta: Record<string, unknown> = {};
  const combined = {
    ...metadata,
    ...record,
  };

  for (const [key, value] of Object.entries(combined)) {
    if (value == null) continue;
    if (METADATA_HIDE_KEYS.has(key)) continue;
    if (
      [
        "id",
        "snippet",
        "content",
        "text",
        "body",
        "metadata",
        "meta",
        "score",
      ].includes(key)
    ) {
      continue;
    }
    if (typeof value === "string" && !value.trim()) continue;
    meta[key] = value;
  }

  return meta;
}

function normalizeState(state?: string) {
  if (!state) return "input-streaming";
  if (state === "input-in-progress") return "input-streaming";
  return state;
}

function deriveErrorText(output: unknown): string | undefined {
  if (!output) return undefined;

  if (typeof output === "string") return output;

  if (typeof output === "object" && "error" in (output as Record<string, unknown>)) {
    const error = (output as Record<string, unknown>).error;
    return typeof error === "string" ? error : safeStringify(error);
  }

  return safeStringify(output);
}

function normalizeMeta(meta: unknown): Record<string, unknown> | undefined {
  if (!meta || typeof meta !== "object" || Array.isArray(meta)) {
    return undefined;
  }
  return meta as Record<string, unknown>;
}

function safeStringify(value: unknown): string | undefined {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return undefined;
  }
}

function firstString(...values: Array<unknown>): string | undefined {
  for (const value of values) {
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed) return trimmed;
    }
  }
  return undefined;
}

function firstNumber(...values: Array<unknown>): number | undefined {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return undefined;
}

function firstAvailable(...values: Array<unknown>): unknown {
  for (const value of values) {
    if (value == null) continue;
    if (typeof value === "string") {
      if (value.trim()) return value;
      continue;
    }
    return value;
  }
  return undefined;
}

function formatMetaValue(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 160 ? `${trimmed.slice(0, 157)}...` : trimmed;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map(formatMetaValue).filter(Boolean).join(", ");
  }
  if (typeof value === "object") {
    return safeStringify(value) ?? String(value);
  }
  return String(value);
}
