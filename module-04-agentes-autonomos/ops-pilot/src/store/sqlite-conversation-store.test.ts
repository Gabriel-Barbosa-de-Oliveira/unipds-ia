import assert from "node:assert/strict";
import { test } from "node:test";

import { ConversationNotFoundError } from "../domain/errors.ts";
import { SqliteConversationStore } from "./sqlite-conversation-store.ts";

test("create() gera ids distintos a cada chamada", async () => {
  const store = new SqliteConversationStore(":memory:");

  const idA = await store.create();
  const idB = await store.create();

  assert.notEqual(idA, idB);
});

test("append seguido de lastMessages retorna as mensagens na ordem de inserção", async () => {
  const store = new SqliteConversationStore(":memory:");
  const conversationId = await store.create();

  await store.append(conversationId, [
    { role: "user", content: "me chame de Gabriel" },
    { role: "assistant", content: "Combinado, Gabriel!" },
  ]);

  const messages = await store.lastMessages(conversationId, 12);

  assert.deepEqual(messages, [
    { role: "user", content: "me chame de Gabriel" },
    { role: "assistant", content: "Combinado, Gabriel!" },
  ]);
});

test("lastMessages com conversationId desconhecido lança ConversationNotFoundError", async () => {
  const store = new SqliteConversationStore(":memory:");

  await assert.rejects(() => store.lastMessages("id-que-nao-existe", 12), ConversationNotFoundError);
});

test("append com conversationId desconhecido lança ConversationNotFoundError, sem criar a conversa implicitamente", async () => {
  const store = new SqliteConversationStore(":memory:");

  await assert.rejects(
    () => store.append("id-que-nao-existe", [{ role: "user", content: "oi" }]),
    ConversationNotFoundError,
  );
});

test("conversa recém-criada sem mensagens retorna lista vazia de lastMessages", async () => {
  const store = new SqliteConversationStore(":memory:");
  const conversationId = await store.create();

  const messages = await store.lastMessages(conversationId, 12);

  assert.deepEqual(messages, []);
});

test("lastMessages nunca retorna mais que o limite, mesmo com mais de 12 mensagens, e mantém a ordem cronológica", async () => {
  const store = new SqliteConversationStore(":memory:");
  const conversationId = await store.create();

  for (let turn = 0; turn < 8; turn += 1) {
    await store.append(conversationId, [
      { role: "user", content: `pergunta ${turn}` },
      { role: "assistant", content: `resposta ${turn}` },
    ]);
  }

  const messages = await store.lastMessages(conversationId, 12);

  assert.equal(messages.length, 12);
  // 8 turnos = 16 mensagens; as últimas 12 descartam os 2 primeiros turnos (turnos 0 e 1).
  assert.deepEqual(messages[0], { role: "user", content: "pergunta 2" });
  assert.deepEqual(messages[messages.length - 1], { role: "assistant", content: "resposta 7" });
});

test("duas conversas distintas nunca misturam mensagens entre si, mesmo com append intercalado", async () => {
  const store = new SqliteConversationStore(":memory:");
  const conversationA = await store.create();
  const conversationB = await store.create();

  await store.append(conversationA, [{ role: "user", content: "sou da conversa A" }]);
  await store.append(conversationB, [{ role: "user", content: "sou da conversa B" }]);
  await store.append(conversationA, [{ role: "assistant", content: "resposta A" }]);

  const messagesA = await store.lastMessages(conversationA, 12);
  const messagesB = await store.lastMessages(conversationB, 12);

  assert.deepEqual(messagesA, [
    { role: "user", content: "sou da conversa A" },
    { role: "assistant", content: "resposta A" },
  ]);
  assert.deepEqual(messagesB, [{ role: "user", content: "sou da conversa B" }]);
});
