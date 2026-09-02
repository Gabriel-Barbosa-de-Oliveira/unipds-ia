import assert from "node:assert/strict";
import { test } from "node:test";

import { IncidentNotFoundError, InvalidSeverityError, ServiceNotFoundError } from "./errors.ts";
import { buildSeedState } from "./seed-data.ts";
import { listAlerts, openIncident, resolveIncident } from "./ops-store.ts";

test("listAlerts sem status retorna todos os alertas semeados", () => {
  const state = buildSeedState();

  const alerts = listAlerts(state);

  assert.equal(alerts.length, 6);
});

test("listAlerts filtra por status", () => {
  const state = buildSeedState();

  const firing = listAlerts(state, "firing");
  const resolved = listAlerts(state, "resolved");

  assert.equal(firing.length, 3);
  assert.ok(firing.every((alert) => alert.status === "firing"));
  assert.equal(resolved.length, 3);
  assert.ok(resolved.every((alert) => alert.status === "resolved"));
});

test("openIncident cria um incidente open para um serviço e severidade válidos, sem mutar o estado anterior", () => {
  const state = buildSeedState();

  const { state: nextState, incident } = openIncident(
    state,
    { title: "Checkout fora do ar", service: "checkout-api", severity: "high" },
    { id: "inc-1", now: "2026-01-02T00:00:00.000Z" },
  );

  assert.equal(incident.status, "open");
  assert.equal(incident.resolvedAt, null);
  assert.equal(incident.serviceId, "svc-checkout-api");
  assert.equal(state.incidents.length, 0);
  assert.equal(nextState.incidents.length, 1);
});

test("resolveIncident transiciona um incidente existente para resolved", () => {
  const seeded = buildSeedState();
  const { state: withIncident, incident: opened } = openIncident(
    seeded,
    { title: "Checkout fora do ar", service: "checkout-api", severity: "high" },
    { id: "inc-1", now: "2026-01-02T00:00:00.000Z" },
  );

  const { incident: resolved } = resolveIncident(withIncident, opened.id, {
    now: "2026-01-02T01:00:00.000Z",
  });

  assert.equal(resolved.status, "resolved");
  assert.equal(resolved.resolvedAt, "2026-01-02T01:00:00.000Z");
});

test("openIncident com service inexistente lança ServiceNotFoundError", () => {
  const state = buildSeedState();

  assert.throws(
    () =>
      openIncident(
        state,
        { title: "X", service: "servico-que-nao-existe", severity: "high" },
        { id: "inc-1", now: "2026-01-02T00:00:00.000Z" },
      ),
    ServiceNotFoundError,
  );
});

test("openIncident com severity fora do enum lança InvalidSeverityError", () => {
  const state = buildSeedState();

  assert.throws(
    () =>
      openIncident(
        state,
        { title: "X", service: "checkout-api", severity: "catastrophic" },
        { id: "inc-1", now: "2026-01-02T00:00:00.000Z" },
      ),
    InvalidSeverityError,
  );
});

test("resolveIncident com id inexistente lança IncidentNotFoundError", () => {
  const state = buildSeedState();

  assert.throws(
    () => resolveIncident(state, "inc-nao-existe", { now: "2026-01-02T00:00:00.000Z" }),
    IncidentNotFoundError,
  );
});

test("resolveIncident sobre incidente já resolved é idempotente, sem erro", () => {
  const seeded = buildSeedState();
  const { state: withIncident, incident: opened } = openIncident(
    seeded,
    { title: "Checkout fora do ar", service: "checkout-api", severity: "high" },
    { id: "inc-1", now: "2026-01-02T00:00:00.000Z" },
  );
  const { state: withResolved } = resolveIncident(withIncident, opened.id, {
    now: "2026-01-02T01:00:00.000Z",
  });

  const { incident: resolvedAgain } = resolveIncident(withResolved, opened.id, {
    now: "2026-01-02T02:00:00.000Z",
  });

  assert.equal(resolvedAgain.status, "resolved");
  // Idempotente: o timestamp original de resolução é preservado, não sobrescrito.
  assert.equal(resolvedAgain.resolvedAt, "2026-01-02T01:00:00.000Z");
});
