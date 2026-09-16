import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { after, before, describe, test } from "node:test";

import type { Express } from "express";

import type { RunOptions, RunResult, ReasoningStrategy } from "../agents/types.ts";
import { estimateTokens, type ContextBreakdown } from "../context/tokens.ts";
import type { ConversationMessage } from "../domain/conversation.ts";
import { ConversationNotFoundError, UnknownStrategyError } from "../domain/errors.ts";
import type { RecallMatch, MemoryStore } from "../memory/memory-store.ts";
import type { ConversationStore } from "../services/conversation-store.repository.ts";
import { createApp } from "./server.ts";

/** Corpo de resposta real do endpoint após 009 — `RunResult` com `conversationId` e `metrics.historyMessages`/`metrics.contextBreakdown`. */
type ChatResponseBody = Omit<RunResult, "metrics"> & {
  conversationId: string;
  metrics: RunResult["metrics"] & { historyMessages: number; contextBreakdown: ContextBreakdown };
};

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
        metrics: { llmCalls: 1, latencyMs: 5, promptTokens: 0, tokenSource: "real" },
      });
      const app = createApp({
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
      assert.deepEqual(body.trace, [{ type: "answer", at: 0, content: "há 3 alertas firing" }]);
      assert.deepEqual(body.metrics, {
        llmCalls: 1,
        latencyMs: 5,
        promptTokens: 0,
        tokenSource: "real",
        historyMessages: 0,
        contextBreakdown: {
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });
      planFake = fakeStrategy("fake-plan-and-execute", {
        answer: "resposta plan-and-execute",
        trace: [],
        metrics: { llmCalls: 2, latencyMs: 2, promptTokens: 0, tokenSource: "real" },
      });

      const app = createApp({
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });
      reflectedFake = fakeStrategy("reflect:fake-base", {
        answer: "resposta com reflection",
        trace: [{ type: "critique", at: 0, content: "aprovado" }],
        metrics: { llmCalls: 2, latencyMs: 3, promptTokens: 0, tokenSource: "real" },
      });

      const app = createApp({
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

      const app = createApp({
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
      const app = createApp({
        resolveStrategy: (name) => {
          if (name === "slow") {
            return {
              name: "fake-slow",
              run: async () => {
                await new Promise((resolve) => setTimeout(resolve, 30));
                return { answer: "resposta lenta", trace: [], metrics: { llmCalls: 1, latencyMs: 30, promptTokens: 0, tokenSource: "real" } };
              },
            };
          }
          return {
            name: "fake-fast",
            run: async () => ({ answer: "resposta rápida", trace: [], metrics: { llmCalls: 1, latencyMs: 0, promptTokens: 0, tokenSource: "real" } }),
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

  describe("User Story 5 (006) — conversa nova, continuada e id desconhecido", () => {
    let baseUrl: string;
    let close: () => Promise<void>;
    let fake: ReturnType<typeof fakeStrategy>;
    let conversationStore: ReturnType<typeof fakeConversationStore>;

    before(async () => {
      fake = fakeStrategy("fake-conversation", {
        answer: "Gabriel, entendido!",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });
      conversationStore = fakeConversationStore({
        "conv-existente": [
          { role: "user", content: "me chame de Gabriel" },
          { role: "assistant", content: "Combinado, Gabriel!" },
        ],
      });
      const app = createApp({ resolveStrategy: () => fake, conversationStore, memoryStore: fakeMemoryStore() });
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });

      const longHistory: ConversationMessage[] = Array.from({ length: 15 }, (_, index) => ({
        role: index % 2 === 0 ? "user" : "assistant",
        content: `mensagem antiga ${index}`,
      }));

      const app = createApp({
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });

      const app = createApp({
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });
      memoryStore = fakeMemoryStore({
        gabriel: [{ fact: "Gabriel cuida de pagamentos", score: 1 }],
      });

      const app = createApp({
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });
      memoryStore = fakeMemoryStore();
      reflectCalls = [];

      const app = createApp({
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });

      const app = createApp({
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });

      const app = createApp({
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
        metrics: { llmCalls: 2, latencyMs: 4, promptTokens: 187, tokenSource: "real" },
      });

      const app = createApp({
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

  describe("User Story 2 (009) — tokenSource estimated/mixed repassados sem reinterpretação", () => {
    test("[US2] tokenSource estimated é repassado sem virar real", async () => {
      const fake = fakeStrategy("fake-tokens-estimated", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 4, promptTokens: 50, tokenSource: "estimated" },
      });
      const app = createApp({
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
        metrics: { llmCalls: 2, latencyMs: 4, promptTokens: 73, tokenSource: "mixed" },
      });
      const app = createApp({
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
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
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
      const app = createApp({ resolveStrategy: () => fake, conversationStore, memoryStore });
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
        assert.ok(contextBreakdown.conversationHistory > 0);
        assert.ok(contextBreakdown.recalledFacts > 0);
        assert.ok(contextBreakdown.currentMessage > 0);
        assert.equal(
          contextBreakdown.total,
          contextBreakdown.currentMessage + contextBreakdown.conversationHistory + contextBreakdown.recalledFacts,
        );
      } finally {
        await close();
      }
    });

    test("[US3] sem histórico nem userId, histórico/fatos aparecem como 0 explícito e total = currentMessage", async () => {
      const fake = fakeStrategy("fake-breakdown-empty", {
        answer: "ok",
        trace: [],
        metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" },
      });
      const app = createApp({
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
        assert.equal(contextBreakdown.total, contextBreakdown.currentMessage);
        assert.equal(contextBreakdown.currentMessage, estimateTokens("oi"));
      } finally {
        await close();
      }
    });
  });
});
