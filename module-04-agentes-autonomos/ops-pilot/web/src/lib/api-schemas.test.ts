import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  ApiErrorSchema,
  ChatAwaitingApprovalSchema,
  ChatOkSchema,
  DecisionOkSchema,
  TraceEventSchema,
} from "./api-schemas.ts";

const metrics = {
  llmCalls: 2,
  latencyMs: 120,
  promptTokens: 300,
  tokenSource: "real",
  modelUsed: "openai/gpt-4o-mini",
  historyMessages: 0,
  contextBreakdown: { total: 300 },
  contextTrimmed: { historyMessages: 0, recalledFacts: 0 },
};
const route = { route: "react", reason: "pergunta direta", source: "router" };

const knownEvents = [
  { type: "route", at: 0, node: "roteador", route: "react", reason: "x", source: "router" },
  { type: "thought", at: 1, node: "react", content: "vou listar" },
  { type: "plan", at: 2, steps: ["listar", "resumir"] },
  { type: "action", at: 3, node: "react", tool: "list_alerts", args: { status: "firing" } },
  { type: "observation", at: 4, node: "react", result: "[]" },
  { type: "critique", at: 5, node: "reflect", content: "ok" },
  { type: "fallback", at: 6, node: "react", from: "a", to: "b", reason: "429" },
  { type: "answer", at: 7, node: "react", content: "nada disparando" },
];

describe("TraceEventSchema", () => {
  test("aceita os 8 tipos conhecidos, com e sem node", () => {
    for (const event of knownEvents) {
      assert.deepEqual(TraceEventSchema.parse(event), event);
    }
  });

  test("aceita tipo desconhecido pelo ramo genérico, preservando os campos (FR-011)", () => {
    const parsed = TraceEventSchema.parse({ type: "novo", at: 3, foo: 1 });
    assert.deepEqual(parsed, { type: "novo", at: 3, foo: 1, unknown: true });
  });

  test("rejeita evento sem at", () => {
    assert.equal(TraceEventSchema.safeParse({ type: "thought", content: "x" }).success, false);
  });
});

describe("respostas da API", () => {
  test("ChatOkSchema aceita o 200 do contrato", () => {
    const body = { requestId: "r1", answer: "ok", trace: knownEvents, route, conversationId: "c1", metrics };
    const parsed = ChatOkSchema.parse(body);
    assert.equal(parsed.trace.length, 8);
    assert.equal(parsed.metrics.historyMessages, 0);
  });

  test("ChatAwaitingApprovalSchema aceita o 202 e não exige answer", () => {
    const body = {
      requestId: "r1",
      status: "awaiting_approval",
      approval: {
        id: "a1",
        tool: "resolve_incident",
        args: { id: "INC-42" },
        summary: "Resolver o incidente INC-42",
        reason: null,
        expiresAt: "2026-10-09T12:15:00.000Z",
      },
      trace: [],
      route,
      conversationId: "c1",
      metrics,
    };
    assert.equal(ChatAwaitingApprovalSchema.parse(body).approval.tool, "resolve_incident");
    assert.equal(ChatOkSchema.safeParse(body).success, false);
  });

  test("DecisionOkSchema aceita o 200 da decisão com rota e métricas nulas", () => {
    const body = {
      requestId: "r2",
      answer: "Incidente INC-42 resolvido.",
      trace: [{ type: "answer", at: 0, node: "aprovacao", content: "Incidente INC-42 resolvido." }],
      route: null,
      metrics: null,
      conversationId: "c1",
      approval: { id: "a1", status: "approved" },
    };
    assert.equal(DecisionOkSchema.parse(body).approval.status, "approved");
  });

  test("ApiErrorSchema aceita erros com e sem requestId", () => {
    assert.equal(ApiErrorSchema.parse({ requestId: "r1", error: "timeout", timeoutMs: 1000 }).error, "timeout");
    assert.equal(ApiErrorSchema.parse({ error: "internal_error" }).requestId, undefined);
  });
});
