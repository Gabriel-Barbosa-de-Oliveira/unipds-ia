import assert from "node:assert/strict";
import { test } from "node:test";

import { formatMetrics, formatTrace, formatTraceEvent, tagTrace } from "./trace.ts";
import type { TraceEvent } from "./types.ts";

const FIXTURE: TraceEvent[] = [
  { type: "thought", at: 0, content: "Preciso listar os alertas em firing" },
  { type: "plan", at: 1, steps: ["Listar alertas firing", "Responder com o resultado"] },
  { type: "action", at: 2, tool: "list_alerts", args: { status: "firing" } },
  { type: "observation", at: 3, result: [{ id: "alert-1", status: "firing" }] },
  { type: "critique", at: 4, content: "Resultado consistente com o pedido" },
  { type: "answer", at: 5, content: "Há 1 alerta em firing: alert-1" },
];

test("formatTraceEvent formata cada tipo de evento de forma determinística", () => {
  assert.equal(formatTraceEvent(FIXTURE[0]!), "[thought] Preciso listar os alertas em firing");
  assert.equal(
    formatTraceEvent(FIXTURE[1]!),
    "[plan] 1) Listar alertas firing; 2) Responder com o resultado",
  );
  assert.equal(
    formatTraceEvent(FIXTURE[2]!),
    '[action] tool=list_alerts args={"status":"firing"}',
  );
  assert.equal(
    formatTraceEvent(FIXTURE[3]!),
    '[observation] result=[{"id":"alert-1","status":"firing"}]',
  );
  assert.equal(formatTraceEvent(FIXTURE[4]!), "[critique] Resultado consistente com o pedido");
  assert.equal(formatTraceEvent(FIXTURE[5]!), "[answer] Há 1 alerta em firing: alert-1");
});

test("formatTrace junta os eventos em ordem, uma linha por evento", () => {
  const formatted = formatTrace(FIXTURE);
  const lines = formatted.split("\n");

  assert.equal(lines.length, FIXTURE.length);
  assert.equal(lines[0], "[thought] Preciso listar os alertas em firing");
  assert.equal(lines[lines.length - 1], "[answer] Há 1 alerta em firing: alert-1");
});

test("formatMetrics formata llmCalls, latencyMs e o modelo usado", () => {
  assert.equal(
    formatMetrics({ llmCalls: 3, latencyMs: 120, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" }),
    "llmCalls=3 latencyMs=120 model=fake-model",
  );
});

test("formatTraceEvent formata o evento route com rota, origem e motivo", () => {
  assert.equal(
    formatTraceEvent({ type: "route", at: 0, route: "planExecute", reason: "várias etapas", source: "router" }),
    "[route] planExecute (router): várias etapas",
  );
});

test("formatTraceEvent prefixa o nó quando o evento traz node", () => {
  assert.equal(formatTraceEvent({ type: "answer", at: 0, node: "react", content: "ok" }), "react │ [answer] ok");
  assert.equal(
    formatTraceEvent({ type: "route", at: 0, node: "roteador", route: "react", reason: "direta", source: "override" }),
    "roteador │ [route] react (override): direta",
  );
});

test("tagTrace carimba node, reindexa at a partir do offset e não muta a entrada", () => {
  const input: TraceEvent[] = [
    { type: "thought", at: 1_700_000_000_000, content: "a" },
    { type: "answer", at: 42, node: "reflect", content: "b" },
  ];
  const snapshot = structuredClone(input);

  const tagged = tagTrace(input, "planExecute", 3);

  assert.deepEqual(tagged, [
    { type: "thought", at: 3, node: "planExecute", content: "a" },
    { type: "answer", at: 4, node: "planExecute", content: "b" },
  ]);
  assert.deepEqual(input, snapshot);
});

test("formatTraceEvent formata o evento fallback de modelo, com e sem node (013)", () => {
  assert.equal(formatTraceEvent({ type: "fallback", at: 0, from: "a", to: "b", reason: "429" }), "[fallback] a → b: 429");
  assert.equal(
    formatTraceEvent({ type: "fallback", at: 0, node: "react", from: "a", to: "b", reason: "429" }),
    "react │ [fallback] a → b: 429",
  );
});
