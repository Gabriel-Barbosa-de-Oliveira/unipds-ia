import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { after, before, describe, test } from "node:test";

import type { Express } from "express";

import type { RouteDecision, RouteName, RunOptions, RunResult, ReasoningStrategy } from "../agents/types.ts";
import { estimateTokens, type ContextBreakdown } from "../context/tokens.ts";
import { composePrompt, type ConversationMessage } from "../domain/conversation.ts";
import { ConversationNotFoundError, UnknownStrategyError } from "../domain/errors.ts";
import { composeWithFacts } from "../domain/memory.ts";
import type { RecallMatch, MemoryStore } from "../memory/memory-store.ts";
import type { ConversationStore } from "../services/conversation-store.repository.ts";
import type { DecideRoute } from "../graph/router.ts";
import { createLogger } from "../obs/logger.ts";
import { SqliteRequestStore } from "../store/sqlite-request-store.ts";
import { createApp, type CreateAppOptions } from "./server.ts";

/** Corpo de resposta real do endpoint após 009 — `RunResult` com `conversationId` e `metrics.historyMessages`/`metrics.contextBreakdown`. */
type ChatResponseBody = Omit<RunResult, "metrics"> & {
  requestId: string;
  conversationId: string;
  route: RouteDecision;
  metrics: RunResult["metrics"] & {
    historyMessages: number;
    contextBreakdown: ContextBreakdown;
    contextTrimmed: { historyMessages: number; recalledFacts: number };
  };
};

/** Roteador fake (012): sempre escolhe `route`, sem rede; conta as chamadas. */
function fixedRouter(route: RouteName = "react"): DecideRoute & { calls: number } {
  const router = Object.assign(
    async () => {
      router.calls += 1;
      return { decided: { route, reason: "fake" }, tokenUsage: { promptTokens: 0, source: "real" as const }, fallbacks: [] };
    },
    { calls: 0 },
  );
  return router;
}

/**
 * `createApp` com roteador fake (nenhum teste chama o OpenRouter), registros num SQLite em memória
 * (nunca cria `./data/opspilot.db`) e logger silencioso (014) por padrão.
 */
function createTestApp(options: CreateAppOptions = {}): Express {
  return createApp({
    decideRoute: fixedRouter(),
    requestStore: new SqliteRequestStore(":memory:"),
    logger: { log() {} },
    ...options,
  });
}

function fakeStrategy(name: string, result: RunResult): ReasoningStrategy & { calls: number; lastInput?: string } {
  const strategy = {
    name,
    calls: 0,
    lastInput: undefined as string | undefined,
    async run(input: string, _options?: RunOptions): Promise<RunResult> {
      strategy.calls += 1;
      strategy.lastInput = input;
      return result;
    },
  };
  return strategy;
}

/** `ConversationStore` em memória, isolado por instância — usado só pelos testes, sem SQLite real. */
function fakeConversationStore(
  seed: Record<string, ConversationMessage[]> = {},
): ConversationStore & { conversations: Map<string, ConversationMessage[]> } {
  const conversations = new Map<string, ConversationMessage[]>(Object.entries(seed));
  let nextId = 0;

  return {
    conversations,
    async create(): Promise<string> {
      const id = `conv-${nextId++}`;
      conversations.set(id, []);
      return id;
    },
    async append(conversationId: string, messages: ConversationMessage[]): Promise<void> {
      const existing = conversations.get(conversationId);
      if (!existing) {
        throw new ConversationNotFoundError(conversationId);
      }
      conversations.set(conversationId, [...existing, ...messages]);
    },
    async lastMessages(conversationId: string, limit: number): Promise<ConversationMessage[]> {
      const existing = conversations.get(conversationId);
      if (!existing) {
        throw new ConversationNotFoundError(conversationId);
      }
      return existing.slice(-limit);
    },
  };
}

/** `MemoryStore` em memória, isolado por instância — usado só pelos testes, sem SQLite/modelo real. */
function fakeMemoryStore(
  seed: Record<string, RecallMatch[]> = {},
): MemoryStore & { rememberCalls: { userId: string; fact: string }[]; forgetCalls: { userId: string; description: string }[] } {
  const facts = new Map<string, RecallMatch[]>(Object.entries(seed));

  return {
    rememberCalls: [],
    forgetCalls: [],
    async remember(userId: string, fact: string) {
      this.rememberCalls.push({ userId, fact });
      const existing = facts.get(userId) ?? [];
      facts.set(userId, [...existing, { fact, score: 1 }]);
      return { stored: true, id: `mem-${existing.length}` };
    },
    async recall(userId: string, _query: string, limit = 3) {
      return (facts.get(userId) ?? []).slice(0, limit);
    },
    async forget(userId: string, description: string) {
      this.forgetCalls.push({ userId, description });
      return { removed: false };
    },
  };
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
        metrics: { llmCalls: 1, latencyMs: 5, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("retorna 200 com answer/trace/metrics/conversationId ao enviar apenas message", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quais alertas estão firing?" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as ChatResponseBody;
      assert.equal(body.answer, "há 3 alertas firing");
      assert.deepEqual(body.trace, [
        { type: "route", at: 0, node: "roteador", route: "react", reason: "fake", source: "router" },
        { type: "answer", at: 1, node: "react", content: "há 3 alertas firing" },
      ]);
      assert.deepEqual(body.route, { route: "react", reason: "fake", source: "router" });
      // latencyMs passa a medir o grafo inteiro (012), então não é mais o valor fixo da fake.
      const { latencyMs, ...metrics } = body.metrics;
      assert.equal(typeof latencyMs, "number");
      assert.deepEqual(metrics, {
        llmCalls: 1 + 1, // roteador + estratégia (012)
        modelUsed: "fake-model", // modelo da estratégia (013)
        promptTokens: 0,
        tokenSource: "real",
        historyMessages: 0,
        contextTrimmed: { historyMessages: 0, recalledFacts: 0 },
        contextBreakdown: {
          system: 0,
          summary: 0,
          currentMessage: estimateTokens("quais alertas estão firing?"),
          conversationHistory: 0,
          recalledFacts: 0,
          total: estimateTokens("quais alertas estão firing?"),
        },
      });
      assert.equal(typeof body.conversationId, "string");
      assert.ok(body.conversationId.length > 0);
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      planFake = fakeStrategy("fake-plan-and-execute", {
        answer: "resposta plan-and-execute",
        trace: [],
        metrics: { llmCalls: 2, latencyMs: 2, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });

      const app = createTestApp({
        resolveStrategy: (name) => {
          if (name === undefined || name === "react") return reactFake;
          if (name === "plan-and-execute") return planFake;
          throw new UnknownStrategyError(name);
        },
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      reflectedFake = fakeStrategy("reflect:fake-base", {
        answer: "resposta com reflection",
        trace: [{ type: "critique", at: 0, content: "aprovado" }],
        metrics: { llmCalls: 2, latencyMs: 3, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });

      const app = createTestApp({
        resolveStrategy: (_name, reflect) => (reflect ? reflectedFake : baseFake),
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
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

      const app = createTestApp({
        resolveStrategy: () => neverResolvingFake,
        timeoutMs: 20,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
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
      const app = createTestApp({
        resolveStrategy: (name) => {
          if (name === "plan-and-execute") {
            return {
              name: "fake-slow",
              run: async () => {
                await new Promise((resolve) => setTimeout(resolve, 30));
                return { answer: "resposta lenta", trace: [], metrics: { llmCalls: 1, latencyMs: 30, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" } };
              },
            };
          }
          return {
            name: "fake-fast",
            run: async () => ({ answer: "resposta rápida", trace: [], metrics: { llmCalls: 1, latencyMs: 0, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" } }),
          };
        },
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("duas requisições concorrentes recebem, cada uma, sua própria resposta", async () => {
      const [slowResponse, fastResponse] = await Promise.all([
        fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "pergunta lenta", strategy: "planExecute" }),
        }),
        fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "pergunta rápida", strategy: "react" }),
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

  describe("User Story 5 (006) — conversa nova, continuada e id desconhecido", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;
    let conversationStore: ReturnType<typeof fakeConversationStore>;

    before(async () => {
      fake = fakeStrategy("fake-conversation", {
        answer: "Gabriel, entendido!",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      conversationStore = fakeConversationStore({
        "conv-existente": [
          { role: "user", content: "me chame de Gabriel" },
          { role: "assistant", content: "Combinado, Gabriel!" },
        ],
      });
      const app = createTestApp({ resolveStrategy: () => fake, conversationStore, memoryStore: fakeMemoryStore() });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("[US1] mensagem sem conversationId cria uma conversa nova e reporta historyMessages: 0", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "primeira mensagem" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as ChatResponseBody;
      assert.equal(typeof body.conversationId, "string");
      assert.equal(body.metrics.historyMessages, 0);
      assert.equal(fake.lastInput, "primeira mensagem");
    });

    test("[US1] mensagem com conversationId existente inclui o histórico na composição do prompt", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "qual é o meu nome?", conversationId: "conv-existente" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as ChatResponseBody;
      assert.equal(body.conversationId, "conv-existente");
      assert.equal(body.metrics.historyMessages, 2);
      assert.ok(fake.lastInput?.includes("me chame de Gabriel"));
      assert.ok(fake.lastInput?.includes("qual é o meu nome?"));
    });

    test("[US1] conversationId desconhecido retorna 404 sem chamar a estratégia", async () => {
      const callsBefore = fake.calls;

      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "oi", conversationId: "id-que-nao-existe" }),
      });

      assert.equal(response.status, 404);
      const body = (await response.json()) as { error: string; conversationId: string };
      assert.equal(body.error, "conversation_not_found");
      assert.equal(body.conversationId, "id-que-nao-existe");
      assert.equal(fake.calls, callsBefore);
    });

    test("[US1] duas conversas diferentes nunca compartilham histórico entre si", async () => {
      const first = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "me chame de Ana" }),
      });
      const { conversationId: conversationA } = (await first.json()) as { conversationId: string };

      const second = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "qual é o meu nome?" }),
      });
      const secondBody = (await second.json()) as ChatResponseBody;

      assert.notEqual(secondBody.conversationId, conversationA);
      assert.equal(secondBody.metrics.historyMessages, 0);
      assert.ok(!fake.lastInput?.includes("Ana"));
    });
  });

  describe("User Story 6 (006) — conversas longas e historyMessages nunca excede 12", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;

    before(async () => {
      fake = fakeStrategy("fake-long-conversation", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });

      const longHistory: ConversationMessage[] = Array.from({ length: 15 }, (_, index) => ({
        role: index % 2 === 0 ? "user" : "assistant",
        content: `mensagem antiga ${index}`,
      }));

      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore({ "conv-longa": longHistory }),
        memoryStore: fakeMemoryStore(),
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("[US2][US3] conversa com mais de 12 mensagens responde 200 e historyMessages é exatamente 12, nunca 15", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "mensagem nova", conversationId: "conv-longa" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as ChatResponseBody;
      assert.equal(body.metrics.historyMessages, 12);
      // As 3 mensagens mais antigas (0, 1, 2) foram descartadas pelo corte de 12.
      assert.ok(!fake.lastInput?.includes("mensagem antiga 0"));
      assert.ok(fake.lastInput?.includes("mensagem antiga 14"));
    });
  });

  describe("User Story 7 (006) — auditoria de historyMessages em diferentes tamanhos de conversa", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;

    before(async () => {
      fake = fakeStrategy("fake-audit", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });

      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore({
          "conv-poucas": [
            { role: "user", content: "oi" },
            { role: "assistant", content: "olá!" },
          ],
        }),
        memoryStore: fakeMemoryStore(),
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("[US3] conversa nova reporta historyMessages: 0", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "primeira mensagem" }),
      });

      const body = (await response.json()) as ChatResponseBody;
      assert.equal(body.metrics.historyMessages, 0);
    });

    test("[US3] conversa com poucas mensagens reporta a quantidade exata", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "mais uma", conversationId: "conv-poucas" }),
      });

      const body = (await response.json()) as ChatResponseBody;
      assert.equal(body.metrics.historyMessages, 2);
    });
  });

  describe("User Story 1 (007) — memória semântica via userId", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;
    let memoryStore: ReturnType<typeof fakeMemoryStore>;
    let lastExtraTools: { length: number; names: string[] } | undefined;

    before(async () => {
      fake = fakeStrategy("fake-memory", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      memoryStore = fakeMemoryStore({
        gabriel: [{ fact: "Gabriel cuida de pagamentos", score: 1 }],
      });

      const app = createTestApp({
        resolveStrategy: (_name, _reflect, extraTools) => {
          lastExtraTools = extraTools
            ? { length: extraTools.length, names: extraTools.map((t) => t.name) }
            : undefined;
          return fake;
        },
        conversationStore: fakeConversationStore(),
        memoryStore,
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("[US1] mensagem sem userId não chama recall nem disponibiliza tools de memória", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quais alertas estão firing?" }),
      });

      assert.equal(response.status, 200);
      assert.equal(lastExtraTools, undefined);
      assert.ok(!fake.lastInput?.includes("Gabriel cuida de pagamentos"));
    });

    test("[US1] mensagem com userId injeta os fatos recuperados no prompt e disponibiliza remember_fact/forget_fact", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quem cuida do checkout?", userId: "gabriel" }),
      });

      assert.equal(response.status, 200);
      assert.ok(fake.lastInput?.includes("Gabriel cuida de pagamentos"));
      assert.deepEqual(lastExtraTools, { length: 2, names: ["remember_fact", "forget_fact"] });
    });

    test("[US1] userId sem fatos registrados nunca recebe fatos de outro userId", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quem cuida do checkout?", userId: "ana" }),
      });

      assert.equal(response.status, 200);
      assert.ok(!fake.lastInput?.includes("Gabriel cuida de pagamentos"));
    });
  });

  describe("User Story 1 (008) — refletor de aprendizado disparado via userId", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;
    let memoryStore: ReturnType<typeof fakeMemoryStore>;
    let reflectCalls: { store: MemoryStore; userId: string; message: string }[];

    before(async () => {
      fake = fakeStrategy("fake-reflector", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      memoryStore = fakeMemoryStore();
      reflectCalls = [];

      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore(),
        memoryStore,
        reflectAndRemember: async (store, userId, message) => {
          reflectCalls.push({ store, userId, message });
        },
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("[US1] mensagem com userId aciona o refletor com userId e message desta requisição", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "eu cuido de pagamentos", userId: "gabriel" }),
      });

      assert.equal(response.status, 200);
      assert.equal(reflectCalls.length, 1);
      assert.equal(reflectCalls[0]!.userId, "gabriel");
      assert.equal(reflectCalls[0]!.message, "eu cuido de pagamentos");
      assert.equal(reflectCalls[0]!.store, memoryStore);
    });

    test("[US1] mensagem sem userId nunca aciona o refletor", async () => {
      const callsBefore = reflectCalls.length;

      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quais alertas estão firing?" }),
      });

      assert.equal(response.status, 200);
      assert.equal(reflectCalls.length, callsBefore);
    });
  });

  describe("User Story 3 (008) — resposta nunca é afetada pelo refletor", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;

    before(async () => {
      fake = fakeStrategy("fake-reflector-lento", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });

      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
        reflectAndRemember: () => new Promise(() => {}),
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("[US3] refletor que nunca resolve não atrasa a resposta", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "eu cuido de pagamentos", userId: "gabriel" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as RunResult;
      assert.equal(body.answer, "ok");
    });
  });

  describe("User Story 3 (008) — falha do refletor não vira erro HTTP", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;

    before(async () => {
      fake = fakeStrategy("fake-reflector-falho", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });

      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
        reflectAndRemember: async () => {
          throw new Error("falha simulada no refletor");
        },
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("[US3] refletor que rejeita não vira erro HTTP nem impede a resposta", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "eu cuido de pagamentos", userId: "gabriel" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as RunResult;
      assert.equal(body.answer, "ok");
    });
  });

  describe("User Story 1 (009) — promptTokens/tokenSource reais repassados pela resposta", () => {
    let baseUrl: string;
    let close: () => Promise<void>;

    before(async () => {
      const fake = fakeStrategy("fake-tokens-real", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 2, latencyMs: 4, promptTokens: 187, tokenSource: "real", modelUsed: "fake-model" },
      });

      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });
      ({ baseUrl, close } = await startServer(app));
    });

    after(() => close());

    test("[US1] resposta repassa promptTokens/tokenSource exatamente como recebido da estratégia", async () => {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "quais alertas estão firing?" }),
      });

      assert.equal(response.status, 200);
      const body = (await response.json()) as ChatResponseBody;
      assert.equal(body.metrics.promptTokens, 187);
      assert.equal(body.metrics.tokenSource, "real");
    });
  });

  // Com `strategy` informado (override, 012) o roteador não roda, então as métricas são só as da estratégia.
  describe("User Story 2 (009) — tokenSource estimated/mixed repassados sem reinterpretação", () => {
    test("[US2] tokenSource estimated é repassado sem virar real", async () => {
      const fake = fakeStrategy("fake-tokens-estimated", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 4, promptTokens: 50, tokenSource: "estimated", modelUsed: "fake-model" },
      });
      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });
      const { baseUrl, close } = await startServer(app);

      try {
        const response = await fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "oi", strategy: "react" }),
        });

        assert.equal(response.status, 200);
        const body = (await response.json()) as ChatResponseBody;
        assert.equal(body.metrics.promptTokens, 50);
        assert.equal(body.metrics.tokenSource, "estimated");
      } finally {
        await close();
      }
    });

    test("[US2] tokenSource mixed é repassado sem virar real nem estimated", async () => {
      const fake = fakeStrategy("fake-tokens-mixed", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 2, latencyMs: 4, promptTokens: 73, tokenSource: "mixed", modelUsed: "fake-model" },
      });
      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });
      const { baseUrl, close } = await startServer(app);

      try {
        const response = await fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "oi", strategy: "react" }),
        });

        assert.equal(response.status, 200);
        const body = (await response.json()) as ChatResponseBody;
        assert.equal(body.metrics.promptTokens, 73);
        assert.equal(body.metrics.tokenSource, "mixed");
      } finally {
        await close();
      }
    });
  });

  describe("User Story 3 (009) — contextBreakdown via /chat", () => {
    test("[US3] histórico e fatos presentes produzem partes > 0 com total igual à soma exata", async () => {
      const fake = fakeStrategy("fake-breakdown-full", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      const conversationStore = fakeConversationStore({
        "conv-breakdown": [
          { role: "user", content: "me chame de Gabriel" },
          { role: "assistant", content: "Combinado, Gabriel!" },
        ],
      });
      const memoryStore = fakeMemoryStore({
        gabriel: [{ fact: "Gabriel cuida de pagamentos", score: 1 }],
      });
      const app = createTestApp({ resolveStrategy: () => fake, conversationStore, memoryStore });
      const { baseUrl, close } = await startServer(app);

      try {
        const response = await fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: "quem cuida do checkout?",
            userId: "gabriel",
            conversationId: "conv-breakdown",
          }),
        });

        assert.equal(response.status, 200);
        const body = (await response.json()) as ChatResponseBody;
        const { contextBreakdown } = body.metrics;
        assert.equal(body.metrics.historyMessages, 2);
        assert.deepEqual(body.metrics.contextTrimmed, { historyMessages: 0, recalledFacts: 0 });
        assert.ok(contextBreakdown.conversationHistory > 0);
        assert.ok(contextBreakdown.recalledFacts > 0);
        assert.ok(contextBreakdown.currentMessage > 0);
        assert.equal(
          contextBreakdown.total,
          contextBreakdown.system +
            contextBreakdown.summary +
            contextBreakdown.currentMessage +
            contextBreakdown.conversationHistory +
            contextBreakdown.recalledFacts,
        );
      } finally {
        await close();
      }
    });

    test("[US3] sem histórico nem userId, histórico/fatos aparecem como 0 explícito e total = currentMessage", async () => {
      const fake = fakeStrategy("fake-breakdown-empty", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });
      const { baseUrl, close } = await startServer(app);

      try {
        const response = await fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "oi" }),
        });

        assert.equal(response.status, 200);
        const body = (await response.json()) as ChatResponseBody;
        const { contextBreakdown } = body.metrics;
        assert.equal(contextBreakdown.conversationHistory, 0);
        assert.equal(contextBreakdown.recalledFacts, 0);
        assert.equal(contextBreakdown.system, 0);
        assert.equal(contextBreakdown.summary, 0);
        assert.equal(contextBreakdown.total, contextBreakdown.currentMessage);
        assert.equal(contextBreakdown.currentMessage, estimateTokens("oi"));
      } finally {
        await close();
      }
    });

    test("/chat uses the shared builder and reports counts after trimming", async () => {
      const history = [
        { role: "user" as const, content: "mensagem antiga longa" },
        { role: "assistant" as const, content: "resposta recente longa" },
      ];
      const fake = fakeStrategy("fake-context-budget", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore({ "conv-budget": history }),
        memoryStore: fakeMemoryStore({
          gabriel: [
            { fact: "fato um", score: 0.9 },
            { fact: "fato dois", score: 0.8 },
          ],
        }),
        contextBudget: { summary: 200, window: 6, memories: 0 },
      });
      const { baseUrl, close } = await startServer(app);

      try {
        const response = await fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "atual", conversationId: "conv-budget", userId: "gabriel" }),
        });

        assert.equal(response.status, 200);
        const body = (await response.json()) as ChatResponseBody;
        assert.ok(body.metrics.contextBreakdown.conversationHistory <= 6);
        assert.equal(body.metrics.contextBreakdown.recalledFacts, 0);
        assert.equal(body.metrics.contextTrimmed.historyMessages, 1);
        assert.equal(body.metrics.contextTrimmed.recalledFacts, 2);
        assert.equal(body.metrics.historyMessages + body.metrics.contextTrimmed.historyMessages, history.length);
        assert.ok(!fake.lastInput?.includes("mensagem antiga longa"));
        assert.ok(!fake.lastInput?.includes("fato um"));
        assert.ok(!fake.lastInput?.includes("fato dois"));
      } finally {
        await close();
      }
    });

    test("/chat applies the default 1200-token window budget to long histories", async () => {
      const history: ConversationMessage[] = Array.from({ length: 12 }, (_, index) => ({
        role: index % 2 === 0 ? "user" : "assistant",
        content: `historical-${index}: ${"x".repeat(600)}`,
      }));
      const fake = fakeStrategy("fake-default-window-budget", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
      });
      const app = createTestApp({
        resolveStrategy: () => fake,
        conversationStore: fakeConversationStore({ "conv-long-budget": history }),
        memoryStore: fakeMemoryStore(),
      });
      const { baseUrl, close } = await startServer(app);

      try {
        const response = await fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: "current", conversationId: "conv-long-budget" }),
        });

        assert.equal(response.status, 200);
        const body = (await response.json()) as ChatResponseBody;
        assert.ok(body.metrics.contextBreakdown.conversationHistory <= 1200);
        assert.ok(body.metrics.contextTrimmed.historyMessages > 0);
        assert.equal(
          body.metrics.historyMessages + body.metrics.contextTrimmed.historyMessages,
          history.length,
        );
        assert.ok(!fake.lastInput?.includes("historical-0:"));
        assert.ok(fake.lastInput?.includes("historical-11:"));
      } finally {
        await close();
      }
    });

    test("/chat composes the same legacy context for react and plan-and-execute with reflection", async () => {
      const history = [
        { role: "user" as const, content: "histórico" },
        { role: "assistant" as const, content: "resposta" },
      ];
      const facts = [
        { fact: "fato relevante", score: 0.9 },
        { fact: "outro fato", score: 0.8 },
      ];
      const expected = composeWithFacts(facts.map(({ fact }) => fact), composePrompt(history, "mensagem atual"));

      for (const strategy of ["react", "plan-and-execute"]) {
        const fake = fakeStrategy(`fake-${strategy}`, {
          answer: "ok",
          trace: [],
          metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
        });
        const app = createTestApp({
          resolveStrategy: () => fake,
          conversationStore: fakeConversationStore({ "conv-shared": history }),
          memoryStore: fakeMemoryStore({ gabriel: facts }),
        });
        const { baseUrl, close } = await startServer(app);

        try {
          const response = await fetch(`${baseUrl}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message: "mensagem atual",
              conversationId: "conv-shared",
              userId: "gabriel",
              strategy,
              reflect: true,
            }),
          });

          assert.equal(response.status, 200);
          assert.equal(fake.lastInput, expected);
        } finally {
          await close();
        }
      }
    });
  });
});

describe("POST /chat — grafo unificado (012)", () => {
  async function postChat(
    app: Express,
    body: Record<string, unknown>,
  ): Promise<{ status: number; body: ChatResponseBody; requestIdHeader: string | null }> {
    const { baseUrl, close } = await startServer(app);
    try {
      const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return {
        status: response.status,
        body: (await response.json()) as ChatResponseBody,
        requestIdHeader: response.headers.get("x-request-id"),
      };
    } finally {
      await close();
    }
  }

  function strategiesByName() {
    const result = (answer: string, llmCalls: number): RunResult => ({
      answer,
      trace: [
        { type: "thought", at: 0, content: `pensando: ${answer}` },
        { type: "answer", at: 1, content: answer },
      ],
      metrics: { llmCalls, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
    });
    const fakes = {
      react: fakeStrategy("fake-react", result("resposta react", 1)),
      plan: fakeStrategy("fake-plan", result("resposta plan", 3)),
      reflected: fakeStrategy("fake-reflect", result("resposta reflect", 2)),
    };
    const resolveCalls: { name: string | undefined; reflect: boolean | undefined }[] = [];
    const resolveStrategy: NonNullable<CreateAppOptions["resolveStrategy"]> = (name, reflect) => {
      resolveCalls.push({ name, reflect });
      if (reflect) return fakes.reflected;
      if (name === "plan-and-execute") return fakes.plan;
      return fakes.react;
    };
    return { fakes, resolveCalls, resolveStrategy };
  }

  describe("User Story 1 (012) — roteamento automático", () => {
    test("sem strategy, executa só a estratégia escolhida pelo roteador", async () => {
      const { fakes, resolveStrategy } = strategiesByName();
      const router = fixedRouter("planExecute");
      const app = createTestApp({
        resolveStrategy,
        decideRoute: router,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });

      const { status, body } = await postChat(app, { message: "triar todos os alertas críticos" });

      assert.equal(status, 200);
      assert.equal(body.answer, "resposta plan");
      assert.equal(router.calls, 1);
      assert.equal(fakes.plan.calls, 1);
      assert.equal(fakes.react.calls, 0);
      assert.equal(fakes.reflected.calls, 0);
      assert.equal(body.metrics.llmCalls, 1 + 3);
    });

    test("roteador que falha responde 200 via react com source fallback", async () => {
      const { fakes, resolveStrategy } = strategiesByName();
      const app = createTestApp({
        resolveStrategy,
        decideRoute: async () => {
          throw new Error("openrouter fora do ar");
        },
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });

      const { status, body } = await postChat(app, { message: "oi" });

      assert.equal(status, 200);
      assert.equal(body.answer, "resposta react");
      assert.equal(fakes.react.calls, 1);
      assert.equal(body.route.route, "react");
      assert.equal(body.route.source, "fallback");
    });
  });

  describe("User Story 2 (012) — route na resposta e node em todo evento", () => {
    test("body.route espelha o evento route e todo evento traz node", async () => {
      const { resolveStrategy } = strategiesByName();
      const app = createTestApp({
        resolveStrategy,
        decideRoute: fixedRouter("reflect"),
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });

      const { body } = await postChat(app, { message: "resumo do incidente para o pós-mortem" });

      const routeEvent = body.trace[0];
      assert.equal(routeEvent?.type, "route");
      if (routeEvent?.type !== "route") return;
      const { type: _type, at: _at, node, ...decision } = routeEvent;
      assert.equal(node, "roteador");
      assert.deepEqual(body.route, decision);
      assert.deepEqual(body.route, { route: "reflect", reason: "fake", source: "router" });
      assert.ok(body.trace.every((event) => event.node !== undefined));
      assert.deepEqual(
        body.trace.slice(1).map((event) => event.node),
        ["reflect", "reflect"],
      );
    });
  });

  describe("User Story 3 (012) — override", () => {
    test("strategy plan-and-execute vira override: roteador não é chamado", async () => {
      const { fakes, resolveStrategy } = strategiesByName();
      const router = fixedRouter("react");
      const app = createTestApp({
        resolveStrategy,
        decideRoute: router,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });

      const { status, body } = await postChat(app, { message: "oi", strategy: "plan-and-execute" });

      assert.equal(status, 200);
      assert.equal(router.calls, 0);
      assert.equal(fakes.plan.calls, 1);
      assert.deepEqual(body.route, {
        route: "planExecute",
        reason: "Estratégia informada pelo cliente",
        source: "override",
      });
      assert.equal(body.metrics.llmCalls, 3);
    });

    test("strategy reflection (ou reflect) resolve reflection sobre react", async () => {
      for (const strategy of ["reflection", "reflect"]) {
        const { fakes, resolveCalls, resolveStrategy } = strategiesByName();
        const app = createTestApp({
          resolveStrategy,
          conversationStore: fakeConversationStore(),
          memoryStore: fakeMemoryStore(),
        });

        const { status, body } = await postChat(app, { message: "oi", strategy });

        assert.equal(status, 200);
        assert.deepEqual(resolveCalls, [{ name: "react", reflect: true }]);
        assert.equal(fakes.reflected.calls, 1);
        assert.equal(body.route.route, "reflect");
        assert.equal(body.route.source, "override");
      }
    });

    test("strategy react com reflect: true decora a rota react", async () => {
      const { resolveCalls, resolveStrategy } = strategiesByName();
      const app = createTestApp({
        resolveStrategy,
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
      });

      const { status } = await postChat(app, { message: "oi", strategy: "react", reflect: true });

      assert.equal(status, 200);
      assert.deepEqual(resolveCalls, [{ name: "react", reflect: true }]);
    });

    test("strategy desconhecida → 422 sem chamar roteador, estratégia nem conversationStore", async () => {
      const { fakes, resolveCalls, resolveStrategy } = strategiesByName();
      const router = fixedRouter("react");
      const conversationStore = fakeConversationStore();
      const app = createTestApp({
        resolveStrategy,
        decideRoute: router,
        conversationStore,
        memoryStore: fakeMemoryStore(),
      });

      const response = await postChat(app, { message: "oi", strategy: "nao-existe" });

      assert.equal(response.status, 422);
      assert.deepEqual(response.body as unknown, {
        requestId: response.requestIdHeader,
        error: "unknown_strategy",
        strategy: "nao-existe",
      });
      assert.equal(router.calls, 0);
      assert.deepEqual(resolveCalls, []);
      assert.equal(fakes.react.calls + fakes.plan.calls + fakes.reflected.calls, 0);
      assert.equal(conversationStore.conversations.size, 0);
    });
  });
});

describe("Trace persistido e logs JSON (014)", () => {
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
  const MARKER = "MARCADOR-SECRETO-123";

  interface Call {
    status: number;
    body: Record<string, unknown>;
    requestIdHeader: string | null;
  }

  async function withServer<T>(app: Express, run: (baseUrl: string) => Promise<T>): Promise<T> {
    const { baseUrl, close } = await startServer(app);
    try {
      return await run(baseUrl);
    } finally {
      await close();
    }
  }

  async function post(baseUrl: string, body: unknown): Promise<Call> {
    const response = await fetch(`${baseUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return {
      status: response.status,
      body: (await response.json()) as Record<string, unknown>,
      requestIdHeader: response.headers.get("x-request-id"),
    };
  }

  async function lookup(baseUrl: string, id: string): Promise<{ status: number; body: Record<string, unknown> }> {
    const response = await fetch(`${baseUrl}/requests/${id}`);
    return { status: response.status, body: (await response.json()) as Record<string, unknown> };
  }

  function okStrategy(content = "há 1 alerta firing"): ReasoningStrategy {
    return {
      name: "fake",
      run: async () => ({
        answer: content,
        trace: [
          { type: "action", at: 0, tool: "list_alerts", args: { status: "firing" } },
          { type: "observation", at: 1, result: [{ id: "alert-1" }] },
          { type: "answer", at: 2, content },
        ],
        metrics: { llmCalls: 2, latencyMs: 1, promptTokens: 40, tokenSource: "real", modelUsed: "fake-model" },
      }),
    };
  }

  function collector() {
    const lines: string[] = [];
    return { lines, logger: createLogger((line) => lines.push(line)) };
  }

  describe("Foundational + User Story 2 — X-Request-Id e requestId em toda resposta", () => {
    test("200 traz cabeçalho UUID igual ao requestId do corpo; pedidos concorrentes têm ids distintos", async () => {
      const app = createTestApp({ resolveStrategy: () => okStrategy(), conversationStore: fakeConversationStore(), memoryStore: fakeMemoryStore() });
      await withServer(app, async (baseUrl) => {
        const [a, b] = await Promise.all([post(baseUrl, { message: "oi" }), post(baseUrl, { message: "oi" })]);
        for (const call of [a, b]) {
          assert.equal(call.status, 200);
          assert.match(call.requestIdHeader ?? "", UUID);
          assert.equal(call.body.requestId, call.requestIdHeader);
        }
        assert.notEqual(a.requestIdHeader, b.requestIdHeader);
      });
    });

    test("o X-Request-Id enviado pelo cliente é ignorado", async () => {
      const app = createTestApp({ resolveStrategy: () => okStrategy(), conversationStore: fakeConversationStore(), memoryStore: fakeMemoryStore() });
      await withServer(app, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Request-Id": "do-cliente" },
          body: JSON.stringify({ message: "oi" }),
        });
        assert.match(response.headers.get("x-request-id") ?? "", UUID);
      });
    });

    const errorCases: { name: string; status: number; options: CreateAppOptions; body: unknown }[] = [
      { name: "400 corpo inválido", status: 400, options: {}, body: {} },
      { name: "404 conversa desconhecida", status: 404, options: {}, body: { message: "oi", conversationId: "nao-existe" } },
      { name: "422 estratégia desconhecida", status: 422, options: {}, body: { message: "oi", strategy: "nao-existe" } },
      {
        name: "504 timeout",
        status: 504,
        options: { timeoutMs: 20, resolveStrategy: () => ({ name: "lenta", run: () => new Promise(() => {}) }) },
        body: { message: "oi" },
      },
      {
        name: "500 erro interno",
        status: 500,
        options: { resolveStrategy: () => ({ name: "quebrada", run: () => Promise.reject(new Error("boom")) }) },
        body: { message: "oi" },
      },
    ];

    for (const errorCase of errorCases) {
      test(`${errorCase.name}: requestId no corpo igual ao cabeçalho`, async () => {
        const app = createTestApp({
          resolveStrategy: () => okStrategy(),
          conversationStore: fakeConversationStore(),
          memoryStore: fakeMemoryStore(),
          ...errorCase.options,
        });
        await withServer(app, async (baseUrl) => {
          const call = await post(baseUrl, errorCase.body);
          assert.equal(call.status, errorCase.status);
          assert.match(call.requestIdHeader ?? "", UUID);
          assert.equal(call.body.requestId, call.requestIdHeader);
        });
      });
    }
  });

  describe("User Story 1 — reabrir registro + trace por GET /requests/:id", () => {
    test("POST 200 seguido de GET devolve métricas e trace idêntico ao original", async () => {
      const app = createTestApp({ resolveStrategy: () => okStrategy(), conversationStore: fakeConversationStore(), memoryStore: fakeMemoryStore() });
      await withServer(app, async (baseUrl) => {
        const chat = await post(baseUrl, { message: "quais alertas?", userId: "gabriel" });
        const found = await lookup(baseUrl, chat.requestIdHeader!);
        const request = found.body.request as Record<string, unknown>;
        const metrics = chat.body.metrics as Record<string, unknown>;

        assert.equal(found.status, 200);
        assert.deepEqual(found.body.trace, chat.body.trace);
        assert.equal(request.requestId, chat.requestIdHeader);
        assert.equal(request.outcome, "ok");
        assert.equal(request.errorType, null);
        assert.equal(request.conversationId, chat.body.conversationId);
        assert.equal(request.userId, "gabriel");
        assert.equal(request.route, "react");
        assert.equal(request.routeSource, "router");
        assert.equal(request.llmCalls, metrics.llmCalls);
        assert.equal(request.modelUsed, "fake-model");
        assert.deepEqual(request.context, { breakdown: metrics.contextBreakdown, trimmed: metrics.contextTrimmed });
      });
    });

    test("override é registrado com routeSource override (FR-012)", async () => {
      const app = createTestApp({ resolveStrategy: () => okStrategy(), conversationStore: fakeConversationStore(), memoryStore: fakeMemoryStore() });
      await withServer(app, async (baseUrl) => {
        const chat = await post(baseUrl, { message: "oi", strategy: "plan-and-execute" });
        const request = (await lookup(baseUrl, chat.requestIdHeader!)).body.request as Record<string, unknown>;
        assert.deepEqual([request.route, request.routeSource], ["planExecute", "override"]);
      });
    });

    test("timeout e erro interno são registrados com outcome, errorType e trace vazio", async () => {
      const cases = [
        { outcome: "timeout", errorType: "ChatTimeoutError", options: { timeoutMs: 20, resolveStrategy: () => ({ name: "lenta", run: () => new Promise<RunResult>(() => {}) }) } },
        { outcome: "error", errorType: "TypeError", options: { resolveStrategy: () => ({ name: "quebrada", run: () => Promise.reject(new TypeError("boom")) }) } },
      ];
      for (const item of cases) {
        const app = createTestApp({ conversationStore: fakeConversationStore(), memoryStore: fakeMemoryStore(), ...item.options });
        await withServer(app, async (baseUrl) => {
          const chat = await post(baseUrl, { message: "oi" });
          const found = await lookup(baseUrl, chat.requestIdHeader!);
          const request = found.body.request as Record<string, unknown>;
          assert.equal(found.status, 200);
          assert.deepEqual([request.outcome, request.errorType, request.route, request.llmCalls], [item.outcome, item.errorType, null, null]);
          assert.deepEqual(found.body.trace, []);
        });
      }
    });

    test("id inexistente ou malformado → 404 request_not_found", async () => {
      const app = createTestApp({ conversationStore: fakeConversationStore(), memoryStore: fakeMemoryStore() });
      await withServer(app, async (baseUrl) => {
        for (const id of [crypto.randomUUID(), "nao-e-uuid"]) {
          const found = await lookup(baseUrl, id);
          assert.equal(found.status, 404);
          assert.deepEqual(found.body, { error: "request_not_found", requestId: id });
        }
      });
    });

    test("rejeição antes de executar (422) não é persistida", async () => {
      const app = createTestApp({ conversationStore: fakeConversationStore(), memoryStore: fakeMemoryStore() });
      await withServer(app, async (baseUrl) => {
        const chat = await post(baseUrl, { message: "oi", strategy: "nao-existe" });
        assert.equal((await lookup(baseUrl, chat.requestIdHeader!)).status, 404);
      });
    });

    test("falha ao gravar não altera a resposta e vira persistence.failed (FR-008)", async () => {
      const { lines, logger } = collector();
      const app = createTestApp({
        resolveStrategy: () => okStrategy(),
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
        logger,
        requestStore: {
          save: () => Promise.reject(new RangeError("disco cheio")),
          find: async () => undefined,
        },
      });
      await withServer(app, async (baseUrl) => {
        const chat = await post(baseUrl, { message: "oi" });
        assert.equal(chat.status, 200);
        assert.equal(chat.body.answer, "há 1 alerta firing");
        const failure = lines.map((line) => JSON.parse(line) as Record<string, unknown>).find((line) => line.event === "persistence.failed");
        assert.deepEqual(failure && { requestId: failure.requestId, errorType: failure.errorType }, {
          requestId: chat.requestIdHeader,
          errorType: "RangeError",
        });
      });
    });
  });

  describe("User Story 3 — logs JSON, uma linha por evento, só metadados", () => {
    test("200 emite received → route.chosen → tool.called → request.completed, todos com o mesmo requestId", async () => {
      const { lines, logger } = collector();
      const app = createTestApp({ resolveStrategy: () => okStrategy(), conversationStore: fakeConversationStore(), memoryStore: fakeMemoryStore(), logger });
      await withServer(app, async (baseUrl) => {
        const chat = await post(baseUrl, { message: "oi" });
        const events = lines.map((line) => JSON.parse(line) as Record<string, unknown>);

        assert.deepEqual(
          events.map((event) => event.event),
          ["request.received", "route.chosen", "tool.called", "request.completed"],
        );
        assert.ok(events.every((event) => event.requestId === chat.requestIdHeader));
        const completed = events.at(-1)!;
        assert.equal(completed.node, "resposta");
        assert.equal(completed.llmCalls, (chat.body.metrics as Record<string, unknown>).llmCalls);
        assert.equal(completed.modelUsed, "fake-model");
        assert.equal(completed.traceEvents, (chat.body.trace as unknown[]).length);
      });
    });

    test("400 e 422 viram request.rejected; 500 vira request.failed sem a mensagem do erro", async () => {
      const { lines, logger } = collector();
      const app = createTestApp({
        resolveStrategy: () => ({ name: "quebrada", run: () => Promise.reject(new Error(`falhou com ${MARKER}`)) }),
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
        logger,
      });
      await withServer(app, async (baseUrl) => {
        await post(baseUrl, {});
        await post(baseUrl, { message: "oi", strategy: "nao-existe" });
        await post(baseUrl, { message: "oi" });
      });
      const events = lines.map((line) => JSON.parse(line) as Record<string, unknown>);
      const pick = (name: string) => events.filter((event) => event.event === name).map(({ status, errorCode, errorType }) => ({ status, errorCode, errorType }));

      assert.deepEqual(pick("request.rejected"), [
        { status: 400, errorCode: "invalid_body", errorType: undefined },
        { status: 422, errorCode: "unknown_strategy", errorType: undefined },
      ]);
      assert.deepEqual(pick("request.failed"), [{ status: 500, errorCode: undefined, errorType: "Error" }]);
      assert.ok(lines.every((line) => !line.includes(MARKER)));
    });

    test("teste-âncora (SC-004): nenhum conteúdo de conversa vaza para os logs", async () => {
      const { lines, logger } = collector();
      const leaky: ReasoningStrategy = {
        name: "vazadora",
        run: async () => ({
          answer: `resposta ${MARKER}`,
          trace: [
            { type: "thought", at: 0, content: `pensando ${MARKER}` },
            { type: "action", at: 1, tool: "list_alerts", args: { service: MARKER } },
            { type: "observation", at: 2, result: [{ note: MARKER }] },
            { type: "fallback", at: 3, from: "a", to: "b", reason: `429 ${MARKER}` },
            { type: "answer", at: 4, content: `resposta ${MARKER}` },
          ],
          metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 1, tokenSource: "real", modelUsed: "b" },
        }),
      };
      const app = createTestApp({
        resolveStrategy: () => leaky,
        decideRoute: async () => ({ decided: { route: "react", reason: `motivo ${MARKER}` }, tokenUsage: { promptTokens: 0, source: "real" }, fallbacks: [] }),
        conversationStore: fakeConversationStore(),
        memoryStore: fakeMemoryStore(),
        logger,
      });

      await withServer(app, async (baseUrl) => {
        const chat = await post(baseUrl, { message: `pergunta ${MARKER}`, userId: "gabriel" });
        assert.equal(chat.status, 200);
        assert.ok(JSON.stringify(chat.body).includes(MARKER), "a resposta HTTP mantém o conteúdo");
        assert.ok(JSON.stringify((await lookup(baseUrl, chat.requestIdHeader!)).body.trace).includes(MARKER), "o banco guarda o payload completo");
      });

      assert.ok(lines.length >= 5);
      for (const line of lines) {
        assert.ok(!line.includes("\n"));
        JSON.parse(line);
        assert.ok(!line.includes(MARKER), `vazou: ${line}`);
      }
    });
  });
});
