import assert from "node:assert/strict";
import { test } from "node:test";

import { buildSupervisorMessages, resolveSupervisorDecision, SUPERVISOR_PROMPT, TEAM_MAX_TURNS } from "./supervisor.ts";

test("decisão válida para um papel vira route", () => {
  assert.deepEqual(resolveSupervisorDecision({ decided: { next: "analista", brief: "levante os alertas" } }, 0), {
    kind: "route",
    to: "analista",
    brief: "levante os alertas",
  });
});

test("done vira resposta final com o brief", () => {
  assert.deepEqual(resolveSupervisorDecision({ decided: { next: "done", brief: "Nada disparando." } }, 2), {
    kind: "done",
    answer: "Nada disparando.",
  });
});

test("decisão inválida vira abort controlado", () => {
  for (const decided of [{ next: "redator", brief: "x" }, { next: "analista", brief: "  " }, "texto", null, {}]) {
    const outcome = resolveSupervisorDecision({ decided }, 0);
    assert.equal(outcome.kind, "abort", JSON.stringify(decided));
    assert.match(outcome.kind === "abort" ? outcome.reason : "", /decisão inválida/);
  }
});

test("falha do modelo vira abort com o motivo", () => {
  const outcome = resolveSupervisorDecision({ error: new Error("429") }, 0);
  assert.deepEqual(outcome, { kind: "abort", reason: "supervisor falhou (429)" });
});

test("no teto, só done é aceito", () => {
  assert.deepEqual(resolveSupervisorDecision({ decided: { next: "analista", brief: "de novo" } }, TEAM_MAX_TURNS), {
    kind: "abort",
    reason: `teto de ${TEAM_MAX_TURNS} passagens atingido`,
  });
  assert.equal(resolveSupervisorDecision({ decided: { next: "done", brief: "fim" } }, TEAM_MAX_TURNS).kind, "done");
});

test("mensagens: prompt com papéis e critérios + quadro com o contador de passagens", () => {
  const [system, user] = buildSupervisorMessages("## Pedido\nx", 2);
  assert.deepEqual(system, ["system", SUPERVISOR_PROMPT]);
  for (const role of ["analista", "planejador", "executor", "done"]) {
    assert.ok(SUPERVISOR_PROMPT.includes(role));
  }
  assert.equal(user?.[0], "user");
  assert.match(user?.[1] ?? "", /Passagens usadas: 2 de 6/);
});
