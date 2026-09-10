import assert from "node:assert/strict";
import { test } from "node:test";

import { buildSeedState } from "../domain/seed-data.ts";
import { InMemoryOpsStore } from "./ops-store.memory.ts";

test("InMemoryOpsStore inicia com o dataset canônico (5 serviços, 6 alertas, 3 runbooks, 0 incidentes)", async () => {
  const store = new InMemoryOpsStore();

  assert.deepEqual(store.getState(), buildSeedState());
});

test("reset() volta ao estado canônico depois de uma mutação (abrir incidente)", async () => {
  const store = new InMemoryOpsStore();
  await store.openIncident({ title: "x", service: "checkout-api", severity: "high" });
  assert.equal((await store.listIncidents("all")).length, 1);

  store.reset();

  assert.equal((await store.listIncidents("all")).length, 0);
  assert.deepEqual(store.getState(), buildSeedState());
});

test("reset() duas vezes seguidas produz estados idênticos, independente de mutações no meio (SC-005)", async () => {
  const store = new InMemoryOpsStore();

  store.reset();
  const first = store.getState();

  await store.openIncident({ title: "y", service: "payments-api", severity: "critical" });
  store.reset();
  const second = store.getState();

  assert.deepEqual(first, second);
});
