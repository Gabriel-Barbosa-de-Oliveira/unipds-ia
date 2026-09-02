import assert from "node:assert/strict";
import { test } from "node:test";

import { formatMetrics, formatTrace, formatTraceEvent } from "./trace.ts";
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

test("formatMetrics formata llmCalls e latencyMs", () => {
  assert.equal(formatMetrics({ llmCalls: 3, latencyMs: 120 }), "llmCalls=3 latencyMs=120");
});
