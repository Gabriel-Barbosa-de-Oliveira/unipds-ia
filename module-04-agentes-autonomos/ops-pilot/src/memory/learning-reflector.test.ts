import assert from "node:assert/strict";
import { test } from "node:test";

import { createMemoryTools, SqliteMemoryStore, type MemoryStore, type RecallMatch } from "./memory-store.ts";
import { reflectAndRemember, type LearningVerdict } from "./learning-reflector.ts";

/** `MemoryStore` em memória, isolado por instância — usado só pelos testes, sem SQLite/modelo real. */
function fakeMemoryStore(): MemoryStore & { rememberCalls: { userId: string; fact: string }[] } {
  const facts = new Map<string, RecallMatch[]>();

  return {
    rememberCalls: [],
    async remember(userId: string, fact: string) {
      this.rememberCalls.push({ userId, fact });
      const existing = facts.get(userId) ?? [];
      facts.set(userId, [...existing, { fact, score: 1 }]);
      return { stored: true, id: `mem-${existing.length}` };
    },
    async recall(userId: string, _query: string, limit = 3) {
      return (facts.get(userId) ?? []).slice(0, limit);
    },
    async forget() {
      return { removed: false };
    },
  };
}

test("reflectAndRemember: hasLearning true com fact chama store.remember com userId e fact corretos", async () => {
  const store = fakeMemoryStore();
  const distillFn = async (): Promise<LearningVerdict> => ({ hasLearning: true, fact: "cuida de pagamentos" });

  await reflectAndRemember(store, "gabriel", "eu cuido de pagamentos", distillFn);

  assert.deepEqual(store.rememberCalls, [{ userId: "gabriel", fact: "cuida de pagamentos" }]);
});

test("reflectAndRemember: hasLearning false nunca chama store.remember", async () => {
  const store = fakeMemoryStore();
  const distillFn = async (): Promise<LearningVerdict> => ({ hasLearning: false });

  await reflectAndRemember(store, "gabriel", "qual o status do serviço?", distillFn);

  assert.deepEqual(store.rememberCalls, []);
});

test("reflectAndRemember: hasLearning true sem fact nunca chama store.remember", async () => {
  const store = fakeMemoryStore();
  const distillFn = async (): Promise<LearningVerdict> => ({ hasLearning: true });

  await reflectAndRemember(store, "gabriel", "mensagem qualquer", distillFn);

  assert.deepEqual(store.rememberCalls, []);
});

test("reflectAndRemember: distillFn rejeitando é absorvido, nunca chama store.remember, nunca rejeita", async () => {
  const store = fakeMemoryStore();
  const distillFn = async (): Promise<LearningVerdict> => {
    throw new Error("falha simulada na destilação");
  };

  await assert.doesNotReject(reflectAndRemember(store, "gabriel", "mensagem qualquer", distillFn));
  assert.deepEqual(store.rememberCalls, []);
});

test("reflectAndRemember: store.remember rejeitando é absorvido, nunca rejeita", async () => {
  const store: MemoryStore = {
    async remember() {
      throw new Error("falha simulada no store");
    },
    async recall() {
      return [];
    },
    async forget() {
      return { removed: false };
    },
  };
  const distillFn = async (): Promise<LearningVerdict> => ({ hasLearning: true, fact: "cuida de pagamentos" });

  await assert.doesNotReject(reflectAndRemember(store, "gabriel", "eu cuido de pagamentos", distillFn));
});

test("[US2] fato gravado por reflectAndRemember é removível por forget_fact, exatamente como um fato manual (FR-008)", async () => {
  const fakeEmbed = async (text: string): Promise<Float32Array> => {
    // Vetor determinístico e forjado: mesmo texto -> mesmo vetor; usado só para exercitar o
    // caminho de ponta a ponta (grava via reflectAndRemember, remove via forget_fact), sem rede.
    const vector = new Float32Array(4).fill(0);
    for (let i = 0; i < text.length; i += 1) {
      const slot = i % 4;
      vector[slot] = (vector[slot] ?? 0) + text.charCodeAt(i);
    }
    const norm = Math.hypot(...vector) || 1;
    return vector.map((value) => value / norm);
  };

  const store = new SqliteMemoryStore(":memory:", fakeEmbed);
  const userId = "gabriel";
  const distillFn = async (): Promise<LearningVerdict> => ({
    hasLearning: true,
    fact: "eu sou o responsável pelo serviço de pagamentos",
  });

  await reflectAndRemember(store, userId, "eu sou o responsável pelo serviço de pagamentos", distillFn);

  const beforeForget = await store.recall(userId, "eu sou o responsável pelo serviço de pagamentos");
  assert.equal(beforeForget.length, 1);

  const [, forgetFactTool] = createMemoryTools(store, userId);
  const observation = JSON.parse(
    (await forgetFactTool!.invoke({ description: "eu sou o responsável pelo serviço de pagamentos" })) as string,
  ) as { removed: boolean; fact?: string };

  assert.equal(observation.removed, true);
  assert.equal(observation.fact, "eu sou o responsável pelo serviço de pagamentos");

  const afterForget = await store.recall(userId, "eu sou o responsável pelo serviço de pagamentos");
  assert.equal(afterForget.length, 0);
});
