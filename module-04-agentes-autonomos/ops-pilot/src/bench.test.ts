import assert from "node:assert/strict";
import { test } from "node:test";

import {
  answerMentionsCount,
  firingAlerts,
  formatSummaryTable,
  newIncidents,
  parseArgs,
  SCENARIOS,
  type BenchRow,
} from "./bench.ts";
import { buildSeedState } from "./domain/seed-data.ts";
import type { Incident, OpsState } from "./domain/ops-store.ts";

function scenarioCheck(id: "C1" | "C2" | "C3") {
  const scenario = SCENARIOS.find((s) => s.id === id);
  if (!scenario) {
    throw new Error(`cenário ${id} não encontrado`);
  }
  return scenario.check;
}

function incident(overrides: Partial<Incident> & Pick<Incident, "id" | "serviceId" | "createdAt">): Incident {
  return {
    title: "título",
    severity: "high",
    status: "open",
    resolvedAt: null,
    ...overrides,
  };
}

test("answerMentionsCount reconhece dígito com limite de palavra", () => {
  assert.ok(answerMentionsCount("há 3 alertas disparando", 3));
  assert.ok(!answerMentionsCount("há 13 alertas disparando", 3));
  assert.ok(!answerMentionsCount("há 2 alertas disparando", 3));
});

test("answerMentionsCount reconhece número por extenso em português", () => {
  assert.ok(answerMentionsCount("restaram dois alertas", 2));
  assert.ok(!answerMentionsCount("restaram dois alertas", 3));
});

test("firingAlerts filtra apenas alertas com status firing", () => {
  const state = buildSeedState();
  const firing = firingAlerts(state);
  assert.equal(firing.length, 3);
  assert.ok(firing.every((alert) => alert.status === "firing"));
});

test("newIncidents retorna apenas incidentes ausentes em before, ordenados por createdAt", () => {
  const before = buildSeedState();
  const kept = incident({ id: "inc-kept", serviceId: "svc-auth-service", createdAt: "2026-01-01T00:00:00.000Z" });
  const before2: OpsState = { ...before, incidents: [kept] };

  const later = incident({ id: "inc-later", serviceId: "svc-checkout-api", createdAt: "2026-01-02T01:00:00.000Z" });
  const earlier = incident({ id: "inc-earlier", serviceId: "svc-payments-api", createdAt: "2026-01-02T00:00:00.000Z" });
  const after: OpsState = { ...before2, incidents: [kept, later, earlier] };

  const created = newIncidents(before2, after);

  assert.deepEqual(created.map((i) => i.id), ["inc-earlier", "inc-later"]);
});

test("formatSummaryTable alinha colunas na ordem cenário, estratégia, acerto, llmCalls, latencyMs", () => {
  const rows: BenchRow[] = [
    { scenario: "C1", strategy: "react", pass: true, reason: "ok", llmCalls: 2, latencyMs: 150 },
    { scenario: "C1", strategy: "plan-and-execute", pass: false, reason: "falhou", llmCalls: 5, latencyMs: 900 },
  ];

  const table = formatSummaryTable(rows);
  const lines = table.split("\n");

  assert.match(lines[0] ?? "", /cenário\s+estratégia\s+acerto\s+llmCalls\s+latencyMs/);
  assert.match(lines[2] ?? "", /C1\s+react\s+OK\s+2\s+150/);
  assert.match(lines[3] ?? "", /C1\s+plan-and-execute\s+FALHOU\s+5\s+900/);
});

test("parseArgs usa defaults quando nenhuma flag é passada", () => {
  const args = parseArgs([]);
  assert.equal(args.scenario, undefined);
  assert.equal(args.noReplanner, false);
  assert.equal(args.maxIterations, 8);
});

test("parseArgs lê --scenario (normalizado para maiúsculo), --no-replanner e --max-iterations", () => {
  const args = parseArgs(["--scenario", "c2", "--no-replanner", "--max-iterations", "4"]);
  assert.equal(args.scenario, "C2");
  assert.equal(args.noReplanner, true);
  assert.equal(args.maxIterations, 4);
});

test("checkC1 aprova quando o store fica inalterado e a resposta menciona a contagem de firing", () => {
  const before = buildSeedState();
  const check = scenarioCheck("C1");

  const result = check(before, before, "Há 3 alertas críticos disparando.");

  assert.equal(result.pass, true);
});

test("checkC1 reprova quando o store foi mutado numa pergunta somente leitura", () => {
  const before = buildSeedState();
  const mutated: OpsState = {
    ...before,
    incidents: [incident({ id: "inc-1", serviceId: "svc-checkout-api", createdAt: "2026-01-02T00:00:00.000Z" })],
  };
  const check = scenarioCheck("C1");

  const result = check(before, mutated, "Há 3 alertas críticos disparando.");

  assert.equal(result.pass, false);
});

test("checkC1 reprova quando a resposta não menciona a contagem correta", () => {
  const before = buildSeedState();
  const check = scenarioCheck("C1");

  const result = check(before, before, "Não há alertas críticos.");

  assert.equal(result.pass, false);
});

test("checkC2 aprova quando checkout (resolved) e payments (open) são abertos nessa ordem", () => {
  const before = buildSeedState();
  const checkout = incident({
    id: "inc-checkout",
    serviceId: "svc-checkout-api",
    createdAt: "2026-01-02T00:00:00.000Z",
    status: "resolved",
    resolvedAt: "2026-01-02T00:05:00.000Z",
  });
  const payments = incident({
    id: "inc-payments",
    serviceId: "svc-payments-api",
    createdAt: "2026-01-02T00:01:00.000Z",
    status: "open",
  });
  const after: OpsState = { ...before, incidents: [checkout, payments] };
  const check = scenarioCheck("C2");

  const result = check(before, after, "Incidentes abertos.");

  assert.equal(result.pass, true);
});

test("checkC2 reprova quando falta o incidente de payments-api", () => {
  const before = buildSeedState();
  const checkout = incident({
    id: "inc-checkout",
    serviceId: "svc-checkout-api",
    createdAt: "2026-01-02T00:00:00.000Z",
    status: "resolved",
    resolvedAt: "2026-01-02T00:05:00.000Z",
  });
  const after: OpsState = { ...before, incidents: [checkout] };
  const check = scenarioCheck("C2");

  const result = check(before, after, "Incidente aberto.");

  assert.equal(result.pass, false);
});

test("checkC2 reprova quando o segundo incidente (payments) também foi resolvido", () => {
  const before = buildSeedState();
  const checkout = incident({
    id: "inc-checkout",
    serviceId: "svc-checkout-api",
    createdAt: "2026-01-02T00:00:00.000Z",
    status: "resolved",
    resolvedAt: "2026-01-02T00:05:00.000Z",
  });
  const payments = incident({
    id: "inc-payments",
    serviceId: "svc-payments-api",
    createdAt: "2026-01-02T00:01:00.000Z",
    status: "resolved",
    resolvedAt: "2026-01-02T00:06:00.000Z",
  });
  const after: OpsState = { ...before, incidents: [checkout, payments] };
  const check = scenarioCheck("C2");

  const result = check(before, after, "Incidentes resolvidos.");

  assert.equal(result.pass, false);
});

test("checkC3 aprova quando 1 incidente novo mira um alerta firing e a resposta reporta o restante", () => {
  const before = buildSeedState();
  const opened = incident({
    id: "inc-oldest",
    serviceId: "svc-checkout-api",
    createdAt: "2026-01-02T00:00:00.000Z",
  });
  const after: OpsState = { ...before, incidents: [opened] };
  const check = scenarioCheck("C3");

  const result = check(before, after, "Abri um incidente; restaram 2 alertas disparando.");

  assert.equal(result.pass, true);
});

test("checkC3 reprova quando nenhum incidente novo foi aberto", () => {
  const before = buildSeedState();
  const check = scenarioCheck("C3");

  const result = check(before, before, "Restaram 3 alertas.");

  assert.equal(result.pass, false);
});

test("checkC3 reprova quando o incidente aberto não corresponde a um serviço com alerta firing", () => {
  const before = buildSeedState();
  const opened = incident({
    id: "inc-wrong",
    serviceId: "svc-auth-service",
    createdAt: "2026-01-02T00:00:00.000Z",
  });
  const after: OpsState = { ...before, incidents: [opened] };
  const check = scenarioCheck("C3");

  const result = check(before, after, "Restaram 2 alertas.");

  assert.equal(result.pass, false);
});
