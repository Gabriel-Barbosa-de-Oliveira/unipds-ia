import assert from "node:assert/strict";
import { test } from "node:test";

import { buildPendingAction } from "../domain/approval.ts";
import { SqliteApprovalStore } from "./sqlite-approval-store.ts";

const NOW = new Date("2026-10-09T12:00:00.000Z");
const IN_TIME = new Date("2026-10-09T12:05:00.000Z");
const LATE = new Date("2026-10-09T12:20:00.000Z");

function action(id = "a1") {
  return buildPendingAction({
    id,
    requestId: "r1",
    conversationId: "c1",
    userId: "gabriel",
    tool: "open_incident",
    args: { title: "Latência", service: "checkout-api", severity: "high", nested: { ok: true } },
    reason: "p95 alto",
    now: NOW,
    ttlMs: 15 * 60_000,
  });
}

test("create + find fazem round-trip completo, inclusive args", async () => {
  const store = new SqliteApprovalStore(":memory:");
  await store.create(action());
  assert.deepEqual(await store.find("a1"), action());
  assert.equal(await store.find("nao-existe"), undefined);
});

test("decide grava a decisão uma vez; a segunda é recusada como já decidida", async () => {
  const store = new SqliteApprovalStore(":memory:");
  await store.create(action());

  const first = await store.decide("a1", "approved", IN_TIME, "r-dec-1");
  assert.equal(first.ok, true);
  assert.equal(first.ok && first.action.status, "approved");
  assert.equal(first.ok && first.action.decidedAt, IN_TIME.toISOString());
  assert.equal(first.ok && first.action.decisionRequestId, "r-dec-1");

  const second = await store.decide("a1", "denied", IN_TIME, "r-dec-2");
  assert.deepEqual(second.ok ? undefined : second.reason, "already_decided");
  assert.equal((await store.find("a1"))?.status, "approved");
});

test("decide depois do prazo é recusado como expirado e nada muda", async () => {
  const store = new SqliteApprovalStore(":memory:");
  await store.create(action());
  const result = await store.decide("a1", "approved", LATE, "r-dec");
  assert.equal(result.ok ? undefined : result.reason, "expired");
  assert.equal((await store.find("a1"))?.status, "pending");
});

test("decide de id inexistente → not_found", async () => {
  const store = new SqliteApprovalStore(":memory:");
  assert.deepEqual(await store.decide("x", "denied", IN_TIME, "r"), { ok: false, reason: "not_found" });
});

test("decisões concorrentes: exatamente uma vence", async () => {
  const store = new SqliteApprovalStore(":memory:");
  await store.create(action());
  const results = await Promise.all([
    store.decide("a1", "approved", IN_TIME, "r-1"),
    store.decide("a1", "denied", IN_TIME, "r-2"),
    store.decide("a1", "approved", IN_TIME, "r-3"),
  ]);
  assert.equal(results.filter((result) => result.ok).length, 1);
});
