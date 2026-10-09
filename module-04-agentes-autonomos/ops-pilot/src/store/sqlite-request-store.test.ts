import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { after, test } from "node:test";

import type { TraceEvent } from "../agents/types.ts";
import { buildRequestRecord } from "../domain/request-record.ts";
import { SqliteRequestStore } from "./sqlite-request-store.ts";

const TRACE: TraceEvent[] = [
  { type: "route", at: 0, node: "roteador", route: "react", reason: "direta", source: "router" },
  { type: "fallback", at: 1, node: "roteador", from: "a", to: "b", reason: "429" },
  { type: "action", at: 2, node: "react", tool: "list_alerts", args: { status: "firing" } },
  { type: "observation", at: 3, node: "react", result: [{ id: "alert-1", nested: { ok: true } }] },
  { type: "answer", at: 4, node: "react", content: "há 1 alerta" },
];

function okRecord(requestId: string) {
  return buildRequestRecord({
    requestId,
    conversationId: "conv-1",
    userId: "gabriel",
    startedAt: new Date("2026-10-09T14:00:00.000Z"),
    durationMs: 900,
    outcome: "ok",
    route: { route: "react", reason: "direta", source: "router" },
    metrics: {
      llmCalls: 3,
      latencyMs: 880,
      promptTokens: 1200,
      tokenSource: "real",
      modelUsed: "modelo-a",
      historyMessages: 0,
      contextBreakdown: { system: 0, summary: 0, currentMessage: 5, conversationHistory: 0, recalledFacts: 0, total: 5 },
      contextTrimmed: { historyMessages: 0, recalledFacts: 0 },
    },
  });
}

function rawDb(store: SqliteRequestStore): DatabaseSync {
  return (store as unknown as { db: DatabaseSync }).db;
}

test("save seguido de find devolve o mesmo registro e o trace idêntico", async () => {
  const store = new SqliteRequestStore(":memory:");
  const record = okRecord("req-1");

  await store.save(record, TRACE);

  assert.deepEqual(await store.find("req-1"), { request: record, trace: TRACE });
});

test("find de id inexistente devolve undefined", async () => {
  assert.equal(await new SqliteRequestStore(":memory:").find("nao-existe"), undefined);
});

test("registro de timeout com trace vazio volta com trace []", async () => {
  const store = new SqliteRequestStore(":memory:");
  const record = buildRequestRecord({
    requestId: "req-timeout",
    conversationId: "conv-1",
    startedAt: new Date(),
    durationMs: 20,
    outcome: "timeout",
    errorType: "ChatTimeoutError",
  });

  await store.save(record, []);

  assert.deepEqual(await store.find("req-timeout"), { request: record, trace: [] });
});

test("save é atômico: falha no meio não deixa registro parcial", async () => {
  const store = new SqliteRequestStore(":memory:");
  // Duas posições iguais violam UNIQUE(request_id, position) no 2º evento.
  const broken: TraceEvent[] = [
    { type: "thought", at: 0, content: "a" },
    { type: "thought", at: 0, content: "b" },
  ];

  await assert.rejects(store.save(okRecord("req-x"), broken));
  assert.equal(await store.find("req-x"), undefined);
});

const tempFile = join(tmpdir(), `opspilot-requests-${process.pid}-${Date.now()}.db`);
after(() => rmSync(tempFile, { force: true }));

test("registros sobrevivem a uma nova instância sobre o mesmo arquivo (DDL idempotente)", async () => {
  await new SqliteRequestStore(tempFile).save(okRecord("req-durable"), TRACE);

  const reopened = new SqliteRequestStore(tempFile);
  assert.deepEqual((await reopened.find("req-durable"))?.trace, TRACE);
});

test("CHECK da coluna requests.outcome rejeita valor fora do domínio, mesmo via SQL direto", () => {
  const db = rawDb(new SqliteRequestStore(":memory:"));

  assert.throws(() => {
    db.prepare(
      `INSERT INTO requests (id, started_at, duration_ms, outcome) VALUES ('bad', '2026-01-01T00:00:00.000Z', 1, 'partial')`,
    ).run();
  });
});

test("listSince devolve só os registros a partir do instante, em ordem cronológica", async () => {
  const store = new SqliteRequestStore(":memory:");
  const at = (iso: string, id: string) =>
    buildRequestRecord({ requestId: id, conversationId: "c", startedAt: new Date(iso), durationMs: 1, outcome: "error" });

  await store.save(at("2026-10-08T23:59:59.999Z", "antes"), []);
  await store.save(at("2026-10-09T10:00:00.000Z", "depois"), []);
  await store.save(at("2026-10-09T00:00:00.000Z", "no-limite"), []);

  const records = await store.listSince(new Date("2026-10-09T00:00:00.000Z"));

  assert.deepEqual(records.map((record) => record.requestId), ["no-limite", "depois"]);
});

test("trace da equipe (017) com handoff e role volta idêntico", async () => {
  const store = new SqliteRequestStore(":memory:");
  const trace: TraceEvent[] = [
    { type: "handoff", at: 0, node: "react", role: "supervisor", from: "supervisor", to: "analista", brief: "levante" },
    { type: "action", at: 1, node: "react", role: "analista", tool: "list_alerts", args: {} },
    { type: "handoff", at: 2, node: "react", role: "supervisor", from: "supervisor", to: "done", brief: "fim" },
    { type: "answer", at: 3, node: "react", role: "supervisor", content: "fim" },
  ];
  await store.save(okRecord("req-team"), trace);
  assert.deepEqual((await store.find("req-team"))?.trace, trace);
});

/** DDL de `requests` antes da 017 — sem 'team' no CHECK de `route`. */
const LEGACY_DDL = `
  CREATE TABLE requests (
    id TEXT PRIMARY KEY, conversation_id TEXT, user_id TEXT, started_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    outcome TEXT NOT NULL CHECK (outcome IN ('ok', 'timeout', 'error')), error_type TEXT,
    route TEXT CHECK (route IS NULL OR route IN ('react', 'planExecute', 'reflect')),
    route_source TEXT CHECK (route_source IS NULL OR route_source IN ('router', 'override', 'fallback')),
    llm_calls INTEGER, prompt_tokens INTEGER,
    token_source TEXT CHECK (token_source IS NULL OR token_source IN ('real', 'estimated', 'mixed')),
    model_used TEXT, history_messages INTEGER, context_json TEXT
  );
  CREATE TABLE trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL REFERENCES requests(id),
    position INTEGER NOT NULL, type TEXT NOT NULL, node TEXT, payload_json TEXT NOT NULL,
    UNIQUE (request_id, position)
  );
  CREATE INDEX idx_requests_conversation ON requests(conversation_id);
  CREATE INDEX idx_requests_started_at ON requests(started_at);
`;

test("banco antigo é migrado para aceitar a rota team, sem perder registros (017)", async () => {
  const legacyFile = join(tmpdir(), `opspilot-legacy-${process.pid}-${Date.now()}.db`);
  after(() => rmSync(legacyFile, { force: true }));

  // Banco criado com o DDL antigo e um registro gravado direto, antes de qualquer store novo abrir o arquivo.
  const legacy = new DatabaseSync(legacyFile);
  legacy.exec(LEGACY_DDL);
  legacy
    .prepare(`INSERT INTO requests (id, conversation_id, started_at, duration_ms, outcome, route, route_source)
              VALUES ('req-antigo', 'conv-1', '2026-10-09T14:00:00.000Z', 5, 'ok', 'react', 'router')`)
    .run();
  legacy
    .prepare("INSERT INTO trace_events (request_id, position, type, node, payload_json) VALUES ('req-antigo', 0, ?, ?, ?)")
    .run(TRACE[0]!.type, TRACE[0]!.node ?? null, JSON.stringify(TRACE[0]));
  legacy.close();

  const store = new SqliteRequestStore(legacyFile);
  const teamRecord = buildRequestRecord({
    requestId: "req-team-route",
    conversationId: "conv-1",
    startedAt: new Date("2026-10-09T15:00:00.000Z"),
    durationMs: 10,
    outcome: "ok",
    route: { route: "team", reason: "investigar e agir", source: "router" },
  });
  await store.save(teamRecord, []);

  assert.equal((await store.find("req-team-route"))?.request.route, "team");
  const old = await store.find("req-antigo");
  assert.equal(old?.request.route, "react");
  assert.deepEqual(old?.trace, [TRACE[0]]);

  const db = rawDb(store);
  const indexes = (db.prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'requests'").all() as { name: string }[])
    .map((row) => row.name)
    .filter((name) => name.startsWith("idx_"));
  assert.deepEqual(indexes.sort(), ["idx_requests_conversation", "idx_requests_started_at"]);
  assert.deepEqual(db.prepare("PRAGMA foreign_key_check").all(), []);

  // Reabrir não migra de novo e continua aceitando 'team'.
  const reopened = new SqliteRequestStore(legacyFile);
  assert.equal((await reopened.find("req-team-route"))?.request.route, "team");
  assert.throws(() =>
    rawDb(reopened)
      .prepare(`INSERT INTO requests (id, started_at, duration_ms, outcome, route) VALUES ('x', '2026-01-01', 1, 'ok', 'outra')`)
      .run(),
  );
});
