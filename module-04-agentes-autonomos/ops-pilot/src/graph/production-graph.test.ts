import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { ROUTE_NAMES, type ReasoningStrategy, type RouteName, type RunResult } from "../agents/types.ts";
import { buildContext, DEFAULT_CONTEXT_BUDGET, type ContextInput } from "../context/context-builder.ts";
import { createProductionGraph, type ProductionRunResult } from "./production-graph.ts";
import type { DecideRoute } from "./router.ts";

type FakeStrategy = ReasoningStrategy & { calls: number; inputs: string[] };

function fakeStrategy(name: string, llmCalls: number): FakeStrategy {
  const fake: FakeStrategy = {
    name,
    calls: 0,
    inputs: [],
    async run(input: string): Promise<RunResult> {
      fake.calls += 1;
      fake.inputs.push(input);
      return {
        answer: `resposta ${name}`,
        // `at` arbitrário de propósito (como o planner usa Date.now()) — o grafo reindexa.
        trace: [
          { type: "thought", at: 1_700_000_000_000, content: `pensando em ${name}` },
          { type: "answer", at: 99, content: `resposta ${name}` },
        ],
        metrics: { llmCalls, latencyMs: 1, promptTokens: 100, tokenSource: "real", modelUsed: "fake-model" },
      };
    },
  };
  return fake;
}

type FakeRouter = DecideRoute & { calls: number };

function countingRouter(decide: DecideRoute): FakeRouter {
  const router: FakeRouter = Object.assign(
    (prompt: string) => {
      router.calls += 1;
      return decide(prompt);
    },
    { calls: 0 },
  );
  return router;
}

function fixedRouter(route: string, reason = "motivo fake"): FakeRouter {
  return countingRouter(async () => ({ decided: { route, reason }, tokenUsage: { promptTokens: 10, source: "real" }, fallbacks: [] }));
}

function failingRouter(): FakeRouter {
  return countingRouter(async () => {
    throw new Error("openrouter fora do ar");
  });
}

function setup(decideRoute: DecideRoute) {
  const strategies: Record<RouteName, FakeStrategy> = {
    react: fakeStrategy("react", 2),
    planExecute: fakeStrategy("planExecute", 4),
    reflect: fakeStrategy("reflect", 3),
  };
  const strategyForCalls: RouteName[] = [];
  const graph = createProductionGraph({
    decideRoute,
    strategyFor: (route) => {
      strategyForCalls.push(route);
      return strategies[route];
    },
  });
  return { graph, strategies, strategyForCalls };
}

const CONTEXT: ContextInput = {
  message: "quais alertas estão firing?",
  window: [
    { role: "user", content: "oi" },
    { role: "assistant", content: "olá" },
  ],
  memories: [{ fact: "o time de plantão é o SRE", score: 0.9 }],
};

/** Invariantes de trace do grafo (data-model.md). */
function assertTraceInvariants(result: ProductionRunResult): void {
  const routeEvents = result.trace.filter((event) => event.type === "route");
  assert.equal(routeEvents.length, 1, "exatamente um evento route");
  assert.equal(result.trace[0]?.type, "route", "route vem primeiro");
  assert.equal(result.trace[0]?.node, "roteador");
  for (const event of result.trace) {
    assert.ok(event.node, `evento sem node: ${JSON.stringify(event)}`);
  }
  // Logo após o route podem vir trocas de modelo do roteador (013); depois, só a estratégia escolhida.
  const routerFallbacks = result.trace.slice(1).findIndex((event) => !(event.type === "fallback" && event.node === "roteador"));
  const strategyStart = routerFallbacks === -1 ? result.trace.length : 1 + routerFallbacks;
  for (const event of result.trace.slice(strategyStart)) {
    assert.equal(event.node, result.route.route, "só a estratégia escolhida produz eventos");
  }
  assert.deepEqual(
    result.trace.map((event) => event.at),
    result.trace.map((_event, index) => index),
  );
  assert.ok(result.route.reason.trim().length > 0);
}

describe("createProductionGraph — roteamento (US1)", () => {
  for (const route of ROUTE_NAMES) {
    test(`rota ${route}: executa só a estratégia escolhida, com o contexto montado`, async () => {
      const { graph, strategies, strategyForCalls } = setup(fixedRouter(route));

      const result = await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET });

      for (const name of ROUTE_NAMES) {
        assert.equal(strategies[name].calls, name === route ? 1 : 0, `chamadas de ${name}`);
      }
      assert.deepEqual(strategyForCalls, [route]);
      assert.deepEqual(strategies[route].inputs, [buildContext(CONTEXT, DEFAULT_CONTEXT_BUDGET).prompt]);
      assert.equal(result.answer, `resposta ${route}`);
      assert.deepEqual(result.route, { route, reason: "motivo fake", source: "router" });
      assert.equal(result.context.prompt, buildContext(CONTEXT, DEFAULT_CONTEXT_BUDGET).prompt);
    });
  }

  test("métricas somam o roteador (+1 chamada, tokens) às da estratégia", async () => {
    const { graph } = setup(fixedRouter("planExecute"));

    const result = await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET });

    assert.equal(result.metrics.llmCalls, 1 + 4);
    assert.equal(result.metrics.promptTokens, 10 + 100);
    assert.equal(result.metrics.tokenSource, "real");
    assert.equal(typeof result.metrics.latencyMs, "number");
  });

  test("roteador que lança cai em react com source fallback, sem propagar o erro", async () => {
    const { graph, strategies } = setup(failingRouter());

    const result = await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET });

    assert.equal(result.route.route, "react");
    assert.equal(result.route.source, "fallback");
    assert.ok(result.route.reason.includes("openrouter fora do ar"));
    assert.equal(strategies.react.calls, 1);
    assert.equal(result.metrics.llmCalls, 1 + 2);
  });

  test("decisão inválida do roteador cai em react com source fallback", async () => {
    const { graph, strategies } = setup(fixedRouter("rota-inventada"));

    const result = await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET });

    assert.deepEqual([result.route.route, result.route.source], ["react", "fallback"]);
    assert.equal(strategies.react.calls, 1);
  });

  test("erro da estratégia escolhida propaga sem tentar outra rota", async () => {
    const graph = createProductionGraph({
      decideRoute: fixedRouter("react"),
      strategyFor: () => ({
        name: "quebrada",
        run: () => Promise.reject(new Error("estratégia falhou")),
      }),
    });

    await assert.rejects(() => graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET }), /estratégia falhou/);
  });
});

describe("createProductionGraph — invariantes do trace (US2)", () => {
  for (const route of ROUTE_NAMES) {
    test(`rota ${route}: um route primeiro, node em todo evento, at sequencial`, async () => {
      const { graph } = setup(fixedRouter(route));
      const result = await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET });

      assertTraceInvariants(result);
      assert.deepEqual(result.trace[0], {
        type: "route",
        at: 0,
        node: "roteador",
        route,
        reason: "motivo fake",
        source: "router",
      });
    });
  }

  test("fallback também respeita as invariantes", async () => {
    const { graph } = setup(failingRouter());
    assertTraceInvariants(await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET }));
  });
});

describe("createProductionGraph — override (US3)", () => {
  for (const override of ["planExecute", "reflect"] as const) {
    test(`override ${override}: não consulta o roteador e marca source override`, async () => {
      const router = fixedRouter("react");
      const { graph, strategies } = setup(router);

      const result = await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET, override });

      assert.equal(router.calls, 0);
      assert.equal(strategies[override].calls, 1);
      assert.equal(strategies.react.calls, 0);
      assert.deepEqual(result.route, {
        route: override,
        reason: "Estratégia informada pelo cliente",
        source: "override",
      });
      assert.equal(result.metrics.llmCalls, strategies[override] === strategies.planExecute ? 4 : 3);
      assert.equal(result.metrics.promptTokens, 100);
      assertTraceInvariants(result);
    });
  }
});

describe("createProductionGraph — resiliência de modelo (013)", () => {
  test("fallbacks de modelo do roteador vêm logo depois do route, com node roteador", async () => {
    const router = countingRouter(async () => ({
      decided: { route: "react", reason: "direta" },
      tokenUsage: { promptTokens: 10, source: "real" },
      fallbacks: [{ from: "a", to: "b", reason: "429" }],
    }));
    const { graph } = setup(router);

    const result = await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET });

    assert.deepEqual(result.trace[1], { type: "fallback", at: 1, node: "roteador", from: "a", to: "b", reason: "429" });
    assert.equal(result.trace[2]?.node, "react");
    assertTraceInvariants(result);
  });

  test("fallback ocorrido dentro da estratégia sai com o node da estratégia; modelUsed vem da estratégia", async () => {
    const strategy: ReasoningStrategy = {
      name: "react",
      run: async () => ({
        answer: "ok",
        trace: [
          { type: "fallback", at: 0, from: "a", to: "b", reason: "429" },
          { type: "answer", at: 1, content: "ok" },
        ],
        metrics: { llmCalls: 3, latencyMs: 1, promptTokens: 5, tokenSource: "real", modelUsed: "b" },
      }),
    };
    const graph = createProductionGraph({ decideRoute: fixedRouter("react"), strategyFor: () => strategy });

    const result = await graph.run({ context: CONTEXT, budget: DEFAULT_CONTEXT_BUDGET });

    assert.deepEqual(result.trace[1], { type: "fallback", at: 1, node: "react", from: "a", to: "b", reason: "429" });
    assert.equal(result.metrics.modelUsed, "b");
    assertTraceInvariants(result);
  });
});
