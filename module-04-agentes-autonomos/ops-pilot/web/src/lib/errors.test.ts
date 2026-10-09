import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { z } from "zod";

import { toUiError, unavailableReason, type UiError } from "./errors.ts";

function assertHuman(error: UiError): void {
  for (const text of [error.title, error.detail]) {
    assert.doesNotMatch(text, /[{}]|\bat \w+ \(|Error:/, `texto técnico vazou: ${text}`);
  }
}

describe("toUiError — respostas HTTP", () => {
  const cases: [number, unknown, string, UiError["action"]][] = [
    [400, { requestId: "r1", error: "invalid_body", issues: [] }, "Não deu para enviar essa mensagem", "retry"],
    [404, { requestId: "r1", error: "conversation_not_found" }, "Essa conversa não existe mais", "new_conversation"],
    [422, { requestId: "r1", error: "unknown_strategy" }, "Estratégia inválida", "none"],
    [504, { requestId: "r1", error: "timeout", timeoutMs: 180000 }, "O copiloto demorou demais para responder", "retry"],
    [500, { requestId: "r1", error: "internal_error" }, "O copiloto falhou ao responder", "retry"],
    [502, "<html>bad gateway</html>", "O copiloto falhou ao responder", "retry"],
    [409, { requestId: "r1", error: "approval_already_decided" }, "Ação já decidida", "none"],
    [410, { requestId: "r1", error: "approval_expired" }, "Ação expirada", "none"],
    [404, { requestId: "r1", error: "approval_not_found" }, "Ação não encontrada", "none"],
  ];

  for (const [status, body, title, action] of cases) {
    test(`${status} ${JSON.stringify(body).slice(0, 40)} → "${title}"`, () => {
      const error = toUiError({ status, body });
      assert.equal(error.title, title);
      assert.equal(error.action, action);
      assert.equal(error.status, status);
      assertHuman(error);
    });
  }

  test("propaga o requestId do corpo quando existe", () => {
    assert.equal(toUiError({ status: 500, body: { requestId: "abc", error: "internal_error" } }).requestId, "abc");
    assert.equal("requestId" in toUiError({ status: 502, body: "x" }), false);
  });
});

describe("toUiError — exceções", () => {
  test("falha de rede/CORS sugere abrir as configurações e cita a origem", () => {
    const error = toUiError({ exception: new TypeError("Failed to fetch") }, { origin: "http://127.0.0.1:5173" });
    assert.equal(error.action, "open_settings");
    assert.equal(error.title, "Não foi possível falar com a API");
    assert.match(error.detail, /http:\/\/127\.0\.0\.1:5173/);
    assert.match(error.detail, /OPSPILOT_CORS_ORIGINS/);
  });

  test("abort local equivale a tempo esgotado", () => {
    const abort = new DOMException("aborted", "AbortError");
    assert.equal(toUiError({ exception: abort }).title, "O copiloto demorou demais para responder");
    assert.equal(toUiError({ exception: abort }).action, "retry");
  });

  test("resposta fora do schema vira 'formato inesperado'", () => {
    const zodError = z.object({ a: z.string() }).safeParse({}).error;
    const error = toUiError({ exception: zodError });
    assert.equal(error.title, "A API respondeu num formato inesperado");
    assertHuman(error);
  });
});

describe("unavailableReason", () => {
  test("409/410/404 tornam o cartão indisponível; demais são recuperáveis", () => {
    assert.equal(unavailableReason(toUiError({ status: 409, body: {} })), "Já decidida");
    assert.equal(unavailableReason(toUiError({ status: 410, body: {} })), "Expirou");
    assert.equal(unavailableReason(toUiError({ status: 404, body: {} })), "Não encontrada");
    assert.equal(unavailableReason(toUiError({ status: 500, body: {} })), undefined);
    assert.equal(unavailableReason(toUiError({ exception: new TypeError("x") })), undefined);
  });
});
