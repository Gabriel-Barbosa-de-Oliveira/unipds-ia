import assert from "node:assert/strict";
import { test } from "node:test";

import type { TraceEvent } from "../agents/types.ts";
import { buildRequestRecord, restoreTrace, toStoredTraceEvents, type ChatMetrics } from "./request-record.ts";

const STARTED_AT = new Date("2026-10-09T14:00:00.000Z");

const METRICS: ChatMetrics = {
  llmCalls: 3,
  latencyMs: 900,
  promptTokens: 1200,
  tokenSource: "real",
  modelUsed: "modelo-a",
  historyMessages: 2,
  contextBreakdown: { system: 0, summary: 0, currentMessage: 5, conversationHistory: 10, recalledFacts: 0, total: 15 },
  contextTrimmed: { historyMessages: 0, recalledFacts: 0 },
};

const TRACE: TraceEvent[] = [
  { type: "route", at: 0, node: "roteador", route: "react", reason: "direta", source: "router" },
  { type: "fallback", at: 1, node: "roteador", from: "a", to: "b", reason: "429" },
  { type: "action", at: 2, node: "react", tool: "list_alerts", args: { status: "firing" } },
  { type: "observation", at: 3, node: "react", result: [{ id: "alert-1" }] },
  { type: "answer", at: 4, node: "react", content: "há 1 alerta" },
];

test("buildRequestRecord com outcome ok preenche rota e métricas", () => {
  const record = buildRequestRecord({
    requestId: "req-1",
    conversationId: "conv-1",
    userId: "gabriel",
    startedAt: STARTED_AT,
    durationMs: 912.4,
    outcome: "ok",
    route: { route: "react", reason: "direta", source: "router" },
    metrics: METRICS,
  });

  assert.deepEqual(record, {
    requestId: "req-1",
    conversationId: "conv-1",
    userId: "gabriel",
    startedAt: "2026-10-09T14:00:00.000Z",
    durationMs: 912,
    outcome: "ok",
    errorType: null,
    route: "react",
    routeSource: "router",
    llmCalls: 3,
    promptTokens: 1200,
    tokenSource: "real",
    modelUsed: "modelo-a",
    historyMessages: 2,
    context: { breakdown: METRICS.contextBreakdown, trimmed: METRICS.contextTrimmed },
  });
});

test("buildRequestRecord com timeout ou erro deixa rota e métricas nulas e guarda o tipo do erro", () => {
  const timeout = buildRequestRecord({
    requestId: "req-2",
    conversationId: "conv-1",
    startedAt: STARTED_AT,
    durationMs: 180_000,
    outcome: "timeout",
    errorType: "ChatTimeoutError",
  });
  assert.equal(timeout.errorType, "ChatTimeoutError");
  assert.equal(timeout.userId, null);
  for (const key of ["route", "routeSource", "llmCalls", "promptTokens", "tokenSource", "modelUsed", "historyMessages", "context"] as const) {
    assert.equal(timeout[key], null, key);
  }

  const failed = buildRequestRecord({ requestId: "req-3", conversationId: null, startedAt: STARTED_AT, durationMs: 5, outcome: "error" });
  assert.equal(failed.errorType, "Error");
});

test("toStoredTraceEvents usa at como posição e guarda o evento completo", () => {
  const rows = toStoredTraceEvents(TRACE);
  assert.deepEqual(rows[2], { position: 2, type: "action", node: "react", payload: TRACE[2] });
  assert.equal(toStoredTraceEvents([{ type: "thought", at: 0, content: "x" }])[0]!.node, null);
});

test("restoreTrace ordena por posição, devolve o trace idêntico e não muta a entrada", () => {
  const shuffled = [...toStoredTraceEvents(TRACE)].reverse();
  const snapshot = structuredClone(shuffled);

  assert.deepEqual(restoreTrace(shuffled), TRACE);
  assert.deepEqual(shuffled, snapshot);
});
