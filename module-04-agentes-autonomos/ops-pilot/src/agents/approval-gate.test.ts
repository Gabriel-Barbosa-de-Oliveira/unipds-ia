import assert from "node:assert/strict";
import { test } from "node:test";

import { InMemoryOpsStore } from "../services/ops-store.memory.ts";
import {
  createApprovalGate,
  createGatedOpsTools,
  executeGatedAction,
  parseGatedArgs,
} from "./approval-gate.ts";
import { createOpsTools } from "./tools.ts";

async function openIncidentId(store: InMemoryOpsStore): Promise<string> {
  const [incident] = await store.listIncidents("open");
  if (incident) {
    return incident.id;
  }
  return (await store.openIncident({ title: "Teste", service: "checkout-api", severity: "high" })).id;
}

function gatedTool(store: InMemoryOpsStore, name: string) {
  const gate = createApprovalGate();
  const tools = createGatedOpsTools(store, gate);
  const found = tools.find((candidate) => candidate.name === name);
  assert.ok(found, name);
  return { gate, tool: found, tools };
}

test("resolve_incident com porta só registra a proposta; o incidente continua aberto", async () => {
  const store = new InMemoryOpsStore();
  const id = await openIncidentId(store);
  const { gate, tool } = gatedTool(store, "resolve_incident");

  const observation = JSON.parse((await tool.invoke({ id })) as string);

  assert.equal(observation.status, "awaiting_approval");
  assert.deepEqual(gate.proposed, { tool: "resolve_incident", args: { id } });
  const stillOpen = await store.listIncidents("open");
  assert.ok(stillOpen.some((incident) => incident.id === id));
});

test("open_incident com porta não cria incidente", async () => {
  const store = new InMemoryOpsStore();
  const before = (await store.listIncidents("all")).length;
  const { gate, tool } = gatedTool(store, "open_incident");

  await tool.invoke({ title: "Latência", service: "checkout-api", severity: "critical" });

  assert.equal(gate.proposed?.tool, "open_incident");
  assert.equal((await store.listIncidents("all")).length, before);
});

test("segunda chamada com porta na mesma requisição é rejeitada e não troca a proposta", async () => {
  const store = new InMemoryOpsStore();
  const { gate, tools } = gatedTool(store, "resolve_incident");
  const resolve = tools.find((candidate) => candidate.name === "resolve_incident")!;
  const open = tools.find((candidate) => candidate.name === "open_incident")!;

  await resolve.invoke({ id: "INC-1" });
  const second = JSON.parse((await open.invoke({ title: "x", service: "checkout-api", severity: "low" })) as string);

  assert.equal(second.status, "rejected");
  assert.deepEqual(gate.proposed, { tool: "resolve_incident", args: { id: "INC-1" } });
});

test("ferramentas de leitura executam normalmente", async () => {
  const store = new InMemoryOpsStore();
  const { gate, tools } = gatedTool(store, "list_alerts");
  const alerts = JSON.parse((await tools.find((candidate) => candidate.name === "list_alerts")!.invoke({})) as string);
  assert.equal(alerts.length, (await store.listAlerts()).length);
  assert.equal(gate.proposed, undefined);
});

test("mesmos nomes, descrições e ordem de createOpsTools", () => {
  const store = new InMemoryOpsStore();
  const original = createOpsTools(store);
  const gated = createGatedOpsTools(store, createApprovalGate());
  assert.deepEqual(
    gated.map((candidate) => [candidate.name, candidate.description]),
    original.map((candidate) => [candidate.name, candidate.description]),
  );
});

test("parseGatedArgs valida com os schemas das ferramentas", () => {
  assert.deepEqual(parseGatedArgs("resolve_incident", { id: "INC-1" }), { id: "INC-1" });
  assert.throws(() => parseGatedArgs("resolve_incident", { id: "" }));
  assert.throws(() => parseGatedArgs("open_incident", { title: "x", service: "y", severity: "enorme" }));
});

test("executeGatedAction executa a ação aprovada e converte erro de domínio", async () => {
  const store = new InMemoryOpsStore();
  const id = await openIncidentId(store);

  const resolved = (await executeGatedAction(store, "resolve_incident", { id })) as { id: string; status: string };
  assert.equal(resolved.status, "resolved");

  const missing = await executeGatedAction(store, "resolve_incident", { id: "INC-404" });
  assert.deepEqual(missing, { error: "IncidentNotFoundError", id: "INC-404" });
});
