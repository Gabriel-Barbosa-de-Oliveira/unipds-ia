import assert from "node:assert/strict";
import { test } from "node:test";

import type { TraceEvent } from "../agents/types.ts";
import { createLogger, errorTypeOf, formatLogLine, traceToLogEvents } from "./logger.ts";

const NOW = new Date("2026-10-09T00:00:00.000Z");

test("formatLogLine produz uma linha JSON com ts, level e event", () => {
  const line = formatLogLine({ event: "persistence.failed", requestId: "r1", errorType: "SqliteError" }, NOW);

  assert.ok(!line.includes("\n"));
  assert.deepEqual(JSON.parse(line), {
    ts: "2026-10-09T00:00:00.000Z",
    level: "error",
    event: "persistence.failed",
    requestId: "r1",
    errorType: "SqliteError",
  });
});

test("formatLogLine usa o nível fixo por tipo de evento", () => {
  const levelOf = (line: string) => (JSON.parse(line) as { level: string }).level;
  assert.equal(levelOf(formatLogLine({ event: "request.lookup", requestId: "r", found: true }, NOW)), "info");
  assert.equal(
    levelOf(formatLogLine({ event: "request.rejected", requestId: "r", status: 400, errorCode: "invalid_body" }, NOW)),
    "warn",
  );
});

test("traceToLogEvents deriva rota, fallback e tool só com metadados, na ordem do trace", () => {
  const trace: TraceEvent[] = [
    { type: "route", at: 0, node: "roteador", route: "react", reason: "motivo do roteador", source: "router" },
    { type: "thought", at: 1, node: "react", content: "pensamento" },
    { type: "action", at: 2, node: "react", tool: "list_alerts", args: { service: "checkout" } },
    { type: "observation", at: 3, node: "react", result: [{ id: "alert-1" }] },
    { type: "fallback", at: 4, node: "react", from: "a", to: "b", reason: "429 com eco do prompt" },
    { type: "critique", at: 5, node: "reflect", content: "crítica" },
    { type: "answer", at: 6, node: "react", content: "resposta" },
  ];

  const events = traceToLogEvents("r1", trace);

  assert.deepEqual(events, [
    { event: "route.chosen", requestId: "r1", node: "roteador", position: 0, route: "react", source: "router" },
    { event: "tool.called", requestId: "r1", node: "react", position: 2, tool: "list_alerts" },
    { event: "model.fallback", requestId: "r1", node: "react", position: 4, from: "a", to: "b" },
  ]);
  for (const event of events) {
    for (const forbidden of ["reason", "args", "result", "content"]) {
      assert.ok(!(forbidden in event), `${event.event} não pode ter ${forbidden}`);
    }
  }
});

test("errorTypeOf devolve só o nome do erro", () => {
  assert.equal(errorTypeOf(new TypeError("conteúdo sensível")), "TypeError");
  assert.equal(errorTypeOf("texto"), "Error");
});

test("createLogger escreve uma linha formatada por evento", () => {
  const lines: string[] = [];
  const logger = createLogger((line) => lines.push(line), () => NOW);

  logger.log({ event: "request.lookup", requestId: "r1", found: false });
  logger.log({ event: "request.lookup", requestId: "r2", found: true });

  assert.equal(lines.length, 2);
  assert.deepEqual(JSON.parse(lines[1]!), { ts: NOW.toISOString(), level: "info", event: "request.lookup", requestId: "r2", found: true });
});

test("eventos de aprovação (015) só levam metadados", () => {
  const requested = JSON.parse(
    formatLogLine({ event: "approval.requested", requestId: "r1", approvalId: "a1", tool: "resolve_incident" }, NOW),
  );
  assert.deepEqual(Object.keys(requested).sort(), ["approvalId", "event", "level", "requestId", "tool", "ts"]);

  const decided = JSON.parse(
    formatLogLine(
      { event: "approval.decided", requestId: "r2", approvalId: "a1", decision: "approved", outcome: "executed" },
      NOW,
    ),
  );
  assert.equal(decided.level, "info");
  assert.equal("args" in decided || "reason" in decided, false);
});

test("handoff da equipe (017) vira team.handoff sem a instrução", () => {
  const brief = "MARCADOR-SECRETO levante os alertas do checkout";
  const trace: TraceEvent[] = [
    { type: "handoff", at: 3, node: "react", role: "supervisor", from: "supervisor", to: "analista", brief },
  ];
  const [event] = traceToLogEvents("r1", trace);
  assert.deepEqual(event, { event: "team.handoff", requestId: "r1", node: "react", position: 3, from: "supervisor", to: "analista" });
  assert.equal(formatLogLine(event!, NOW).includes("MARCADOR-SECRETO"), false);
});
