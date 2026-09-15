import assert from "node:assert/strict";
import { test } from "node:test";

import { embed } from "./embeddings.ts";
import { createMemoryTools, SqliteMemoryStore } from "./memory-store.ts";

/**
 * `embed` fake e determinístico: cada texto conhecido mapeia para um vetor fixo, forjado à mão,
 * para exercitar dedup/isolamento/limite sem tocar o modelo real nem a rede (research.md item 4).
 * Textos desconhecidos caem em um vetor "neutro" (ortogonal aos demais).
 */
const VECTORS: Record<string, number[]> = {
  "eu cuido de pagamentos": [1, 0, 0],
  "sou eu quem cuida de pagamentos por aqui": [0.99, 0.14, 0], // quase idêntico ao de cima (score > 0.92)
  "quem cuida do checkout?": [0.95, 0.31, 0], // relevante o suficiente para o forget encontrar (score >= 0.3)
  "eu cuido de autenticação": [0, 1, 0],
  "assunto totalmente diferente": [0, 0, 1],
  // Três fatos angularmente espaçados (pairwise <= 0.866, nunca deduplicam entre si) mas todos
  // relevantes (score >= 0.5) para a mesma consulta ampla — usados no teste de `limit`.
  "consulta ampla sobre pagamentos": [1, 0, 0],
  "fato A sobre pagamentos": [1, 0, 0],
  "fato B relacionado a pagamentos": [0.866, 0.5, 0],
  "fato C tangencialmente sobre pagamentos": [0.5, 0.866, 0],
};

async function fakeEmbed(text: string): Promise<Float32Array> {
  const raw = VECTORS[text] ?? [0, 0, 0.5];
  const vector = new Float32Array(raw);
  const norm = Math.sqrt(vector.reduce((acc, v) => acc + v * v, 0));
  return norm === 0 ? vector : vector.map((v) => v / norm);
}

test("remember não duplica um fato quase idêntico (score > 0.92) do mesmo userId", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);

  const first = await store.remember("gabriel", "eu cuido de pagamentos");
  const second = await store.remember("gabriel", "sou eu quem cuida de pagamentos por aqui");

  assert.equal(first.stored, true);
  assert.equal(second.stored, false);

  const recalled = await store.recall("gabriel", "eu cuido de pagamentos", 10);
  assert.equal(recalled.length, 1);
});

test("remember grava um fato novo quando o mais parecido existente tem score <= 0.92", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);

  await store.remember("gabriel", "eu cuido de pagamentos");
  const result = await store.remember("gabriel", "eu cuido de autenticação");

  assert.equal(result.stored, true);
  // Os dois fatos são semanticamente ortogonais entre si — confirma cada um separadamente
  // (uma única consulta não seria relevante o suficiente, score >= 0.3, para ambos ao mesmo tempo).
  assert.equal((await store.recall("gabriel", "eu cuido de pagamentos", 10)).length, 1);
  assert.equal((await store.recall("gabriel", "eu cuido de autenticação", 10)).length, 1);
});

test("recall nunca mistura fatos de dois userIds diferentes", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);

  await store.remember("gabriel", "eu cuido de pagamentos");
  await store.remember("ana", "eu cuido de autenticação");

  const recalledForGabriel = await store.recall("gabriel", "eu cuido de pagamentos", 10);
  const recalledForAna = await store.recall("ana", "eu cuido de pagamentos", 10);

  assert.equal(recalledForGabriel.length, 1);
  assert.equal(recalledForAna.length, 0);
});

test("recall nunca retorna mais que o limit informado", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);

  // Três fatos angularmente espaçados o suficiente para não deduplicarem entre si (pairwise <
  // 0.92), mas todos relevantes (score >= 0.5) para a mesma consulta ampla.
  await store.remember("gabriel", "fato A sobre pagamentos");
  await store.remember("gabriel", "fato B relacionado a pagamentos");
  await store.remember("gabriel", "fato C tangencialmente sobre pagamentos");

  const recalled = await store.recall("gabriel", "consulta ampla sobre pagamentos", 2);

  assert.equal(recalled.length, 2);
  assert.equal(recalled[0]?.fact, "fato A sobre pagamentos");
  assert.equal(recalled[1]?.fact, "fato B relacionado a pagamentos");
});

test("recall descarta fatos com score abaixo de 0.3 e retorna lista vazia sem fatos relevantes", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);

  await store.remember("gabriel", "eu cuido de autenticação");

  const recalled = await store.recall("gabriel", "eu cuido de pagamentos", 10);

  assert.deepEqual(recalled, []);
});

test("recall para userId sem nenhum fato retorna lista vazia, não erro", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);

  const recalled = await store.recall("ninguem", "qualquer pergunta", 10);

  assert.deepEqual(recalled, []);
});

test("forget remove o fato de maior score quando >= 0.3 e reporta qual foi removido", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);
  await store.remember("gabriel", "eu cuido de pagamentos");

  const result = await store.forget("gabriel", "quem cuida do checkout?");

  assert.equal(result.removed, true);
  assert.equal(result.fact, "eu cuido de pagamentos");
  assert.deepEqual(await store.recall("gabriel", "eu cuido de pagamentos", 10), []);
});

test("forget não remove nada quando nenhum fato atinge o limiar de confiança", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);
  await store.remember("gabriel", "eu cuido de autenticação");

  const result = await store.forget("gabriel", "assunto totalmente diferente");

  assert.equal(result.removed, false);
  assert.equal((await store.recall("gabriel", "eu cuido de autenticação", 10)).length, 1);
});

test("createMemoryTools: remember_fact delega a store.remember com o userId da closure, nunca do schema", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);
  const [rememberFactTool] = createMemoryTools(store, "gabriel");

  const observation = await rememberFactTool!.invoke({ fact: "eu cuido de pagamentos" });
  const parsed = JSON.parse(observation as string);

  assert.equal(parsed.stored, true);
  assert.equal((await store.recall("gabriel", "eu cuido de pagamentos", 10)).length, 1);
  assert.equal((await store.recall("outra-pessoa", "eu cuido de pagamentos", 10)).length, 0);
});

test("createMemoryTools: remember_fact chamado duas vezes com fatos quase iguais só grava uma vez (US2)", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);
  const [rememberFactTool] = createMemoryTools(store, "gabriel");

  const first = JSON.parse((await rememberFactTool!.invoke({ fact: "eu cuido de pagamentos" })) as string);
  const second = JSON.parse(
    (await rememberFactTool!.invoke({ fact: "sou eu quem cuida de pagamentos por aqui" })) as string,
  );

  assert.equal(first.stored, true);
  assert.equal(second.stored, false);
  assert.equal((await store.recall("gabriel", "eu cuido de pagamentos", 10)).length, 1);
});

test("createMemoryTools: forget_fact delega a store.forget com o userId da closure", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);
  await store.remember("gabriel", "eu cuido de pagamentos");
  const [, forgetFactTool] = createMemoryTools(store, "gabriel");

  const observation = await forgetFactTool!.invoke({ description: "quem cuida do checkout?" });
  const parsed = JSON.parse(observation as string);

  assert.equal(parsed.removed, true);
  assert.equal(parsed.fact, "eu cuido de pagamentos");
});

test("createMemoryTools: forget_fact não remove nada quando a descrição não corresponde com confiança suficiente (US3)", async () => {
  const store = new SqliteMemoryStore(":memory:", fakeEmbed);
  await store.remember("gabriel", "eu cuido de autenticação");
  const [, forgetFactTool] = createMemoryTools(store, "gabriel");

  const observation = await forgetFactTool!.invoke({ description: "assunto totalmente diferente" });
  const parsed = JSON.parse(observation as string);

  assert.equal(parsed.removed, false);
  assert.equal((await store.recall("gabriel", "eu cuido de autenticação", 10)).length, 1);
});

// Único teste desta suíte com o modelo real — mais lento, precisa de rede na 1ª execução
// (research.md item 4/9). Prova a promessa central da feature: recall sem palavra em comum.
test("recall (modelo real) encontra um fato relevante sem nenhuma palavra em comum com a pergunta", async () => {
  const store = new SqliteMemoryStore(":memory:", embed);

  await store.remember("gabriel", "eu sou o responsável pelo serviço de pagamentos");
  const recalled = await store.recall("gabriel", "quem cuida do checkout financeiro?");

  assert.ok(recalled.length > 0, "esperava encontrar o fato relevante, mas recall retornou vazio");
  assert.equal(recalled[0]?.fact, "eu sou o responsável pelo serviço de pagamentos");
  assert.ok(recalled[0]!.score >= 0.3);
});
