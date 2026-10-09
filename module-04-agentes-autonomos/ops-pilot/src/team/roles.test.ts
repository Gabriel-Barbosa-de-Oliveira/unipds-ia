import assert from "node:assert/strict";
import { test } from "node:test";

import { createApprovalGate, createGatedOpsTools } from "../agents/approval-gate.ts";
import { InMemoryOpsStore } from "../services/ops-store.memory.ts";
import { AnalystReportSchema, PlanSchema, selectRoleTools } from "./roles.ts";

const tools = createGatedOpsTools(new InMemoryOpsStore(), createApprovalGate());
const names = (role: Parameters<typeof selectRoleTools>[1]) => selectRoleTools(tools, role).map((tool) => tool.name);

test("analista recebe exatamente as 3 ferramentas de leitura", () => {
  assert.deepEqual(names("analista"), ["list_alerts", "list_incidents", "consultar_runbook"]);
});

test("planejador não recebe nenhuma ferramenta", () => {
  assert.deepEqual(names("planejador"), []);
});

test("executor recebe exatamente abrir e resolver incidente", () => {
  assert.deepEqual(names("executor"), ["open_incident", "resolve_incident"]);
});

test("ferramenta ausente é erro de composição", () => {
  assert.throws(() => selectRoleTools(tools.filter((tool) => tool.name !== "list_alerts"), "analista"), /list_alerts/);
});

test("relatório do analista só guarda fatos: recomendações são descartadas no parse", () => {
  const parsed = AnalystReportSchema.parse({
    facts: [{ statement: "alert-1 firing (critical)", source: "list_alerts", recommendation: "abra um incidente" }],
    recommendation: "reinicie o serviço",
  });
  assert.deepEqual(parsed, { facts: [{ statement: "alert-1 firing (critical)", source: "list_alerts" }] });
  assert.equal(AnalystReportSchema.safeParse({ facts: [{ statement: "x", source: "chute" }] }).success, false);
});

test("plano tem de 1 a 8 passos", () => {
  assert.equal(PlanSchema.safeParse({ steps: [] }).success, false);
  assert.equal(PlanSchema.safeParse({ steps: Array.from({ length: 9 }, (_, i) => `p${i}`) }).success, false);
  assert.equal(PlanSchema.safeParse({ steps: ["abrir incidente"] }).success, true);
});
