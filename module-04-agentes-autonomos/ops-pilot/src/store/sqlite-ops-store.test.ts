import assert from "node:assert/strict";
import { test } from "node:test";

import type { DatabaseSync } from "node:sqlite";

import { IncidentNotFoundError, InvalidSeverityError, ServiceNotFoundError } from "../domain/errors.ts";
import { SqliteOpsStore, seedCanonicalScenario } from "./sqlite-ops-store.ts";

function seeded(): SqliteOpsStore {
  const store = new SqliteOpsStore(":memory:");
  seedCanonicalScenario(store);
  return store;
}

function rawDb(store: SqliteOpsStore): DatabaseSync {
  return (store as unknown as { db: DatabaseSync }).db;
}

test("seedCanonicalScenario semeia 5 serviços, 6 alertas (3 firing, 3 resolved) e 3 runbooks", async () => {
  const store = seeded();

  const alerts = await store.listAlerts();
  const firing = await store.listAlerts("firing");
  const resolved = await store.listAlerts("resolved");
  const incidents = await store.listIncidents();

  assert.equal(alerts.length, 6);
  assert.equal(firing.length, 3);
  assert.equal(resolved.length, 3);
  assert.equal(incidents.length, 0);
});

test("seedCanonicalScenario é idempotente: rodar duas vezes não duplica nem falha", async () => {
  const store = new SqliteOpsStore(":memory:");
  seedCanonicalScenario(store);
  seedCanonicalScenario(store);

  const alerts = await store.listAlerts();
  assert.equal(alerts.length, 6);
});

test("openIncident cria um incidente open para um serviço e severidade válidos", async () => {
  const store = seeded();

  const incident = await store.openIncident({
    title: "Checkout fora do ar",
    service: "checkout-api",
    severity: "high",
  });

  assert.equal(incident.status, "open");
  assert.equal(incident.resolvedAt, null);
  assert.equal(incident.summary, null);
  assert.equal(incident.title, "Checkout fora do ar");
});

test("openIncident lança InvalidSeverityError para severidade fora do enum", async () => {
  const store = seeded();

  await assert.rejects(
    () => store.openIncident({ title: "x", service: "checkout-api", severity: "sev2" }),
    InvalidSeverityError,
  );
});

test("openIncident lança ServiceNotFoundError para serviço inexistente", async () => {
  const store = seeded();

  await assert.rejects(
    () => store.openIncident({ title: "x", service: "servico-fantasma", severity: "high" }),
    ServiceNotFoundError,
  );
});

test("resolveIncident marca como resolved, preenche resolvedAt e aceita summary opcional", async () => {
  const store = seeded();
  const incident = await store.openIncident({ title: "x", service: "payments-api", severity: "critical" });

  const resolved = await store.resolveIncident(incident.id, "mitigado via rollback");

  assert.equal(resolved.status, "resolved");
  assert.ok(resolved.resolvedAt);
  assert.equal(resolved.summary, "mitigado via rollback");
});

test("resolveIncident é idempotente: resolver de novo não sobrescreve o summary já gravado", async () => {
  const store = seeded();
  const incident = await store.openIncident({ title: "x", service: "payments-api", severity: "critical" });
  await store.resolveIncident(incident.id, "primeiro resumo");

  const secondCall = await store.resolveIncident(incident.id, "segundo resumo, deveria ser ignorado");

  assert.equal(secondCall.summary, "primeiro resumo");
});

test("resolveIncident lança IncidentNotFoundError para id inexistente", async () => {
  const store = seeded();

  await assert.rejects(() => store.resolveIncident("id-que-nao-existe"), IncidentNotFoundError);
});

test("listIncidents filtra por open, resolved e all; lista vazia é resultado válido", async () => {
  const store = seeded();
  const open = await store.openIncident({ title: "x", service: "checkout-api", severity: "low" });
  const toResolve = await store.openIncident({ title: "y", service: "auth-service", severity: "medium" });
  await store.resolveIncident(toResolve.id);

  const allIncidents = await store.listIncidents("all");
  const openIncidents = await store.listIncidents("open");
  const resolvedIncidents = await store.listIncidents("resolved");
  const noFilter = await store.listIncidents();

  assert.equal(allIncidents.length, 2);
  assert.equal(noFilter.length, 2);
  assert.deepEqual(
    openIncidents.map((incident) => incident.id),
    [open.id],
  );
  assert.deepEqual(
    resolvedIncidents.map((incident) => incident.id),
    [toResolve.id],
  );
});

test("listIncidents retorna lista vazia quando não há incidentes semeados", async () => {
  const store = seeded();

  assert.deepEqual(await store.listIncidents("open"), []);
});

test("getRunbook retorna o conteúdo do runbook de um serviço com runbook cadastrado", async () => {
  const store = seeded();

  const runbook = await store.getRunbook("checkout-api");

  assert.ok(runbook);
  assert.match(runbook.content, /checkout/i);
});

test("getRunbook retorna null (não é erro) para um serviço existente sem runbook", async () => {
  const store = seeded();

  const runbook = await store.getRunbook("inventory-service");

  assert.equal(runbook, null);
});

test("getRunbook lança ServiceNotFoundError para um serviço inexistente", async () => {
  const store = seeded();

  await assert.rejects(() => store.getRunbook("servico-fantasma"), ServiceNotFoundError);
});

test("CHECK da coluna incidents.severity rejeita valor fora do domínio, mesmo via SQL direto", () => {
  const store = seeded();
  const db = rawDb(store);

  assert.throws(() => {
    db.prepare(
      `INSERT INTO incidents (id, title, service_id, severity, status, created_at, resolved_at, summary)
       VALUES ('bad-1', 'x', 'svc-checkout-api', 'sev2', 'open', '2026-01-01T00:00:00.000Z', NULL, NULL)`,
    ).run();
  });
});

test("CHECK da coluna incidents.status rejeita valor fora do domínio, mesmo via SQL direto", () => {
  const store = seeded();
  const db = rawDb(store);

  assert.throws(() => {
    db.prepare(
      `INSERT INTO incidents (id, title, service_id, severity, status, created_at, resolved_at, summary)
       VALUES ('bad-2', 'x', 'svc-checkout-api', 'high', 'mitigated', '2026-01-01T00:00:00.000Z', NULL, NULL)`,
    ).run();
  });
});

test("CHECK da coluna alerts.status rejeita valor fora do domínio, mesmo via SQL direto", () => {
  const store = seeded();
  const db = rawDb(store);

  assert.throws(() => {
    db.prepare(
      "INSERT INTO alerts (id, service_id, title, status, created_at) VALUES ('bad-3', 'svc-checkout-api', 'x', 'ack', '2026-01-01T00:00:00.000Z')",
    ).run();
  });
});

test("US1: incidente aberto e resolvido sobrevive à destruição e recriação da instância (mesmo arquivo)", async (t) => {
  const { mkdtempSync, rmSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");

  const dir = mkdtempSync(join(tmpdir(), "opspilot-restart-"));
  const dbPath = join(dir, "opspilot.db");
  t.after(() => rmSync(dir, { recursive: true, force: true }));

  let incidentId: string;
  {
    const beforeRestart = new SqliteOpsStore(dbPath);
    seedCanonicalScenario(beforeRestart);
    const opened = await beforeRestart.openIncident({
      title: "Latência alta",
      service: "checkout-api",
      severity: "high",
    });
    const resolved = await beforeRestart.resolveIncident(opened.id, "mitigado via rollback");
    incidentId = resolved.id;
    // `beforeRestart` sai de escopo aqui — simula o processo sendo encerrado.
  }

  const afterRestart = new SqliteOpsStore(dbPath);
  const [persisted] = await afterRestart.listIncidents("all");

  assert.equal(persisted?.id, incidentId);
  assert.equal(persisted?.status, "resolved");
  assert.equal(persisted?.summary, "mitigado via rollback");
  assert.ok(persisted?.resolvedAt);

  const services = await afterRestart.getRunbook("checkout-api");
  assert.ok(services, "serviços/runbooks semeados antes do reinício também continuam persistidos");
});

test("DDL é idempotente: construir duas instâncias sobre o mesmo arquivo não falha", async (t) => {
  const { mkdtempSync, rmSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");

  const dir = mkdtempSync(join(tmpdir(), "opspilot-test-"));
  const dbPath = join(dir, "opspilot.db");
  t.after(() => rmSync(dir, { recursive: true, force: true }));

  const first = new SqliteOpsStore(dbPath);
  seedCanonicalScenario(first);

  const second = new SqliteOpsStore(dbPath);
  const alerts = await second.listAlerts();

  assert.equal(alerts.length, 6);
});
