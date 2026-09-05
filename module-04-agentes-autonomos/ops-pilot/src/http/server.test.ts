import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { after, before, describe, test } from "node:test";

import type { Express } from "express";

import type { RunOptions, RunResult, ReasoningStrategy } from "../agents/types.ts";
import { UnknownStrategyError } from "../domain/errors.ts";
import { createApp } from "./server.ts";

function fakeStrategy(name: string, result: RunResult): ReasoningStrategy & { calls: number } {
  const strategy = {
    name,
    calls: 0,
    async run(_input: string, _options?: RunOptions): Promise<RunResult> {
      strategy.calls += 1;
      return result;
    },
  };
  return strategy;
}

function startServer(app: Express): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  return new Promise((resolve) => {
    const server = app.listen(0, () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        baseUrl: `http://127.0.0.1:${port}`,
        close: () => new Promise((res) => server.close(() => res())),
      });
    });
  });
}

describe("POST /chat", () => {
  describe("User Story 1 — estratégia padrão", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;

    before(async () => {
      fake = fakeStrategy("fake-default", {
        answer: "há 3 alertas firing",
        trace: [{ type: "answer", at: 0, content: "há 3 alertas firing" }],
        metrics: { llmCalls: 1, latencyMs: 5 },
      });
      const app = createApp({ resolveStrategy: () => fake });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("retorna 200 com answer/trace/metrics ao enviar apenas message", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quais alertas estão firing?" }),
      });

      assert.equal(response.status, 200);
      const body = await response.json();
      assert.deepEqual(body, {
        answer: "há 3 alertas firing",
        trace: [{ type: "answer", at: 0, content: "há 3 alertas firing" }],
        metrics: { llmCalls: 1, latencyMs: 5 },
      });
      assert.equal(fake.calls, 1);
    });

    test("retorna 400 quando message está ausente, sem chamar a estratégia", async () => {
      const callsBefore = fake.calls;

      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      assert.equal(response.status, 400);
      const body = (await response.json()) as { error: string; issues: unknown[] };
      assert.equal(body.error, "invalid_body");
      assert.ok(Array.isArray(body.issues));
      assert.equal(fake.calls, callsBefore);
    });

    test("retorna 400 quando message é vazio", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "" }),
      });

      assert.equal(response.status, 400);
      const body = (await response.json()) as { error: string };
      assert.equal(body.error, "invalid_body");
    });
  });

  describe("User Story 2 — estratégia explícita e estratégia desconhecida", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let reactFake: ReturnType<typeof fakeStrategy>;
    let planFake: ReturnType<typeof fakeStrategy>;

    before(async () => {
      reactFake = fakeStrategy("fake-react", {
        answer: "resposta react",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1 },
      });
      planFake = fakeStrategy("fake-plan-and-execute", {
        answer: "resposta plan-and-execute",
        trace: [],
        metrics: { llmCalls: 2, latencyMs: 2 },
      });

      const app = createApp({
        resolveStrategy: (name) => {
          if (name === undefined || name === "react") return reactFake;
          if (name === "plan-and-execute") return planFake;
          throw new UnknownStrategyError(name);
        },
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("usa a estratégia explícita informada, não a padrão", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "abra um incidente", strategy: "plan-and-execute" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as RunResult;
      assert.equal(body.answer, "resposta plan-and-execute");
      assert.equal(planFake.calls, 1);
      assert.equal(reactFake.calls, 0);
    });

    test("retorna 422 para estratégia desconhecida, sem chamar nenhuma estratégia", async () => {
      const reactCallsBefore = reactFake.calls;
      const planCallsBefore = planFake.calls;

      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "oi", strategy: "nao-existe" }),
      });

      assert.equal(response.status, 422);
      const body = (await response.json()) as { error: string; strategy: string };
      assert.equal(body.error, "unknown_strategy");
      assert.equal(body.strategy, "nao-existe");
      assert.equal(reactFake.calls, reactCallsBefore);
      assert.equal(planFake.calls, planCallsBefore);
    });
  });

  describe("User Story 3 — reflect", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let baseFake: ReturnType<typeof fakeStrategy>;
    let reflectedFake: ReturnType<typeof fakeStrategy>;

    before(async () => {
      baseFake = fakeStrategy("fake-base", {
        answer: "resposta sem reflection",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1 },
      });
      reflectedFake = fakeStrategy("reflect:fake-base", {
        answer: "resposta com reflection",
        trace: [{ type: "critique", at: 0, content: "aprovado" }],
        metrics: { llmCalls: 2, latencyMs: 3 },
      });

      const app = createApp({
        resolveStrategy: (_name, reflect) => (reflect ? reflectedFake : baseFake),
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("reflect: true usa a estratégia refletida em vez da base", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quais alertas estão firing?", reflect: true }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as RunResult;
      assert.equal(body.answer, "resposta com reflection");
      assert.equal(reflectedFake.calls, 1);
      assert.equal(baseFake.calls, 0);
    });

    test("sem reflect (padrão false) usa a estratégia base", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quais alertas estão firing?" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as RunResult;
      assert.equal(body.answer, "resposta sem reflection");
      assert.equal(baseFake.calls, 1);
    });
  });

  describe("User Story 4 — timeout", () => {
    let baseUrl: string;
    let close: () => Promise<void>;

    before(async () => {
      const neverResolvingFake: ReasoningStrategy = {
        name: "fake-never-resolves",
        run: () => new Promise(() => {}),
      };

      const app = createApp({
        resolveStrategy: () => neverResolvingFake,
        timeoutMs: 20,
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("retorna 504 quando a estratégia ultrapassa timeoutMs", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quais alertas estão firing?" }),
      });

      assert.equal(response.status, 504);
      const body = (await response.json()) as { error: string; timeoutMs: number };
      assert.equal(body.error, "timeout");
      assert.equal(body.timeoutMs, 20);
    });
  });

  describe("Isolamento entre requisições concorrentes (FR-010)", () => {
    let baseUrl: string;
    let close: () => Promise<void>;

    before(async () => {
      const app = createApp({
        resolveStrategy: (name) => {
          if (name === "slow") {
            return {
              name: "fake-slow",
              run: async () => {
                await new Promise((resolve) => setTimeout(resolve, 30));
                return { answer: "resposta lenta", trace: [], metrics: { llmCalls: 1, latencyMs: 30 } };
              },
            };
          }
          return {
            name: "fake-fast",
            run: async () => ({ answer: "resposta rápida", trace: [], metrics: { llmCalls: 1, latencyMs: 0 } }),
          };
        },
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("duas requisições concorrentes recebem, cada uma, sua própria resposta", async () => {
      const [slowResponse, fastResponse] = await Promise.all([
        fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "pergunta lenta", strategy: "slow" }),
        }),
        fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "pergunta rápida", strategy: "fast" }),
        }),
      ]);

      const [slowBody, fastBody] = await Promise.all([
        slowResponse.json() as Promise<RunResult>,
        fastResponse.json() as Promise<RunResult>,
      ]);

      assert.equal(slowResponse.status, 200);
      assert.equal(fastResponse.status, 200);
      assert.equal(slowBody.answer, "resposta lenta");
      assert.equal(fastBody.answer, "resposta rápida");
    });
  });
});
