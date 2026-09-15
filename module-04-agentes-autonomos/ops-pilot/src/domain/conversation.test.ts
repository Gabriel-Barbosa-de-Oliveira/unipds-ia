import assert from "node:assert/strict";
import { test } from "node:test";

import { composePrompt, type ConversationMessage } from "./conversation.ts";

test("composePrompt sem histórico retorna a mensagem original, sem texto extra", () => {
  const result = composePrompt([], "quais alertas estão firing?");

  assert.equal(result, "quais alertas estão firing?");
});

test("composePrompt com histórico parcial inclui as mensagens anteriores, em ordem, antes da nova mensagem", () => {
  const history: ConversationMessage[] = [
    { role: "user", content: "me chame de Gabriel" },
    { role: "assistant", content: "Combinado, Gabriel!" },
  ];

  const result = composePrompt(history, "qual é o meu nome?");

  const indexUser = result.indexOf("me chame de Gabriel");
  const indexAssistant = result.indexOf("Combinado, Gabriel!");
  const indexNew = result.indexOf("qual é o meu nome?");

  assert.ok(indexUser >= 0 && indexAssistant >= 0 && indexNew >= 0);
  assert.ok(indexUser < indexAssistant, "mensagem do usuário deve vir antes da resposta do copiloto");
  assert.ok(indexAssistant < indexNew, "histórico deve vir antes da nova mensagem");
});

test("composePrompt com 12 mensagens de histórico inclui todas — o corte é responsabilidade de quem busca o histórico", () => {
  const history: ConversationMessage[] = Array.from({ length: 12 }, (_, index) => ({
    role: index % 2 === 0 ? "user" : "assistant",
    content: `mensagem ${index}`,
  }));

  const result = composePrompt(history, "mensagem nova");

  for (let index = 0; index < 12; index += 1) {
    assert.ok(result.includes(`mensagem ${index}`), `deveria conter mensagem ${index}`);
  }
  assert.ok(result.includes("mensagem nova"));
});

test("composePrompt é determinística — mesma entrada produz a mesma saída", () => {
  const history: ConversationMessage[] = [{ role: "user", content: "oi" }];

  assert.equal(composePrompt(history, "tudo bem?"), composePrompt(history, "tudo bem?"));
});
