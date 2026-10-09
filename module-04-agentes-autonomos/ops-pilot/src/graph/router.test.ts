import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { ROUTE_NAMES } from "../agents/types.ts";
import { UnknownStrategyError } from "../domain/errors.ts";
import { buildRouterMessages, parseRouteName, resolveRouteDecision, ROUTE_TABLE } from "./router.ts";

describe("parseRouteName", () => {
  test("aceita as rotas e os nomes legados das estratégias", () => {
    assert.equal(parseRouteName("react"), "react");
    assert.equal(parseRouteName("planExecute"), "planExecute");
    assert.equal(parseRouteName("plan-and-execute"), "planExecute");
    assert.equal(parseRouteName("reflect"), "reflect");
    assert.equal(parseRouteName("reflection"), "reflect");
  });

  test("lança UnknownStrategyError com o valor recebido para nomes desconhecidos", () => {
    for (const name of ["nao-existe", "", "React", "toString", "__proto__"]) {
      assert.throws(
        () => parseRouteName(name),
        (error: unknown) => error instanceof UnknownStrategyError && error.strategy === name,
      );
    }
  });
});

describe("buildRouterMessages", () => {
  test("devolve system + user com o prompt recebido", () => {
    const messages = buildRouterMessages("pergunta");
    assert.equal(messages.length, 2);
    assert.equal(messages[0]![0], "system");
    assert.deepEqual(messages[1], ["user", "pergunta"]);
  });

  test("o system contém a tabela de rotas com uma linha por rota (FR-004)", () => {
    const system = buildRouterMessages("x")[0]![1];
    assert.ok(system.includes("| Rota | Quando usar | Exemplos |"));
    for (const route of ROUTE_NAMES) {
      assert.ok(system.includes(`| ${route} |`), `faltou a linha da rota ${route}`);
    }
  });
});

describe("resolveRouteDecision", () => {
  test("decisão válida do modelo vira source router", () => {
    assert.deepEqual(resolveRouteDecision({ decided: { route: "planExecute", reason: "várias etapas" } }), {
      route: "planExecute",
      reason: "várias etapas",
      source: "router",
    });
  });

  test("override vence a decisão do modelo", () => {
    assert.deepEqual(resolveRouteDecision({ override: "reflect", decided: { route: "react", reason: "x" } }), {
      route: "reflect",
      reason: "Estratégia informada pelo cliente",
      source: "override",
    });
  });

  test("erro do roteador vira fallback para react com a causa no motivo", () => {
    const decision = resolveRouteDecision({ error: new Error("boom") });
    assert.equal(decision.route, "react");
    assert.equal(decision.source, "fallback");
    assert.ok(decision.reason.startsWith("Fallback:"));
    assert.ok(decision.reason.includes("boom"));
  });

  test("decisão nula, rota desconhecida ou motivo vazio viram fallback para react", () => {
    for (const decided of [null, undefined, { route: "outra", reason: "x" }, { route: "react", reason: "   " }]) {
      const decision = resolveRouteDecision({ decided });
      assert.equal(decision.route, "react");
      assert.equal(decision.source, "fallback");
      assert.ok(decision.reason.length > 0);
    }
  });
});

describe("rota team (017)", () => {
  test("team e equipe resolvem para a rota da equipe", () => {
    assert.equal(parseRouteName("team"), "team");
    assert.equal(parseRouteName("equipe"), "team");
    assert.throws(() => parseRouteName("time"), UnknownStrategyError);
  });

  test("a tabela do roteador oferece a rota team, e o roteador pode escolhê-la", () => {
    assert.ok(ROUTE_TABLE.some((entry) => entry.route === "team"));
    assert.match(buildRouterMessages("x")[0]![1], /\| team \|/);
    assert.deepEqual(resolveRouteDecision({ decided: { route: "team", reason: "investigar e agir" } }), {
      route: "team",
      reason: "investigar e agir",
      source: "router",
    });
  });
});
