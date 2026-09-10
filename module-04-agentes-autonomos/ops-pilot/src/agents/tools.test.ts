import assert from "node:assert/strict";
import { test } from "node:test";

import { seedCanonicalScenario, SqliteOpsStore } from "../store/sqlite-ops-store.ts";
import { createOpsTools } from "./tools.ts";

function seededTools() {
  const store = new SqliteOpsStore(":memory:");
  seedCanonicalScenario(store);
  const tools = createOpsTools(store);
  return {
    store,
    listAlertsTool: tools.find((t) => t.name === "list_alerts")!,
    openIncidentTool: tools.find((t) => t.name === "open_incident")!,
    resolveIncidentTool: tools.find((t) => t.name === "resolve_incident")!,
    listIncidentsTool: tools.find((t) => t.name === "list_incidents")!,
    consultarRunbookTool: tools.find((t) => t.name === "consultar_runbook")!,
  };
}

test("list_alerts sem status retorna todos os alertas semeados", async () => {
  const { listAlertsTool } = seededTools();

  const result = JSON.parse(await listAlertsTool.invoke({}));

  assert.equal(result.length, 6);
});

test("list_alerts filtra por status", async () => {
  const { listAlertsTool } = seededTools();

  const firing = JSON.parse(await listAlertsTool.invoke({ status: "firing" }));

  assert.equal(firing.length, 3);
  assert.ok(firing.every((alert: { status: string }) => alert.status === "firing"));
});

test("open_incident cria um incidente open para serviço e severidade válidos", async () => {
  const { openIncidentTool } = seededTools();

  const result = JSON.parse(
    await openIncidentTool.invoke({ title: "Checkout fora do ar", service: "checkout-api", severity: "high" }),
  );

  assert.equal(result.status, "open");
  assert.equal(result.resolvedAt, null);
});

test("open_incident retorna ServiceNotFoundError estruturado para serviço inexistente", async () => {
  const { openIncidentTool } = seededTools();

  const result = JSON.parse(
    await openIncidentTool.invoke({ title: "x", service: "servico-fantasma", severity: "high" }),
  );

  assert.equal(result.error, "ServiceNotFoundError");
  assert.equal(result.service, "servico-fantasma");
});

test("resolve_incident resolve um incidente aberto, com summary opcional", async () => {
  const { openIncidentTool, resolveIncidentTool } = seededTools();
  const opened = JSON.parse(
    await openIncidentTool.invoke({ title: "x", service: "payments-api", severity: "critical" }),
  );

  const resolved = JSON.parse(
    await resolveIncidentTool.invoke({ id: opened.id, summary: "mitigado via rollback" }),
  );

  assert.equal(resolved.status, "resolved");
  assert.equal(resolved.summary, "mitigado via rollback");
});

test("resolve_incident retorna IncidentNotFoundError estruturado para id inexistente", async () => {
  const { resolveIncidentTool } = seededTools();

  const result = JSON.parse(await resolveIncidentTool.invoke({ id: "id-que-nao-existe" }));

  assert.equal(result.error, "IncidentNotFoundError");
});

test("list_incidents filtra por open, resolved e all", async () => {
  const { openIncidentTool, resolveIncidentTool, listIncidentsTool } = seededTools();
  const open = JSON.parse(
    await openIncidentTool.invoke({ title: "x", service: "checkout-api", severity: "low" }),
  );
  const toResolve = JSON.parse(
    await openIncidentTool.invoke({ title: "y", service: "auth-service", severity: "medium" }),
  );
  await resolveIncidentTool.invoke({ id: toResolve.id });

  const openList = JSON.parse(await listIncidentsTool.invoke({ status: "open" }));
  const resolvedList = JSON.parse(await listIncidentsTool.invoke({ status: "resolved" }));
  const allList = JSON.parse(await listIncidentsTool.invoke({ status: "all" }));

  assert.deepEqual(
    openList.map((i: { id: string }) => i.id),
    [open.id],
  );
  assert.deepEqual(
    resolvedList.map((i: { id: string }) => i.id),
    [toResolve.id],
  );
  assert.equal(allList.length, 2);
});

test("list_incidents sem status retorna todos; lista vazia é resultado válido (não erro)", async () => {
  const { listIncidentsTool } = seededTools();

  const result = JSON.parse(await listIncidentsTool.invoke({}));

  assert.deepEqual(result, []);
});

test("consultar_runbook retorna o conteúdo do runbook de um serviço com runbook cadastrado", async () => {
  const { consultarRunbookTool } = seededTools();

  const result = JSON.parse(await consultarRunbookTool.invoke({ service: "checkout-api" }));

  assert.equal(result.service, "checkout-api");
  assert.match(result.runbook, /checkout/i);
});

test("consultar_runbook retorna runbook: null (não é erro) para serviço sem runbook cadastrado", async () => {
  const { consultarRunbookTool } = seededTools();

  const result = JSON.parse(await consultarRunbookTool.invoke({ service: "inventory-service" }));

  assert.equal(result.runbook, null);
  assert.equal(result.error, undefined);
});

test("consultar_runbook retorna ServiceNotFoundError estruturado para serviço inexistente", async () => {
  const { consultarRunbookTool } = seededTools();

  const result = JSON.parse(await consultarRunbookTool.invoke({ service: "servico-fantasma" }));

  assert.equal(result.error, "ServiceNotFoundError");
});
