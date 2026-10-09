import assert from "node:assert/strict";
import { describe, test } from "node:test";

import type { ChatAwaitingApproval, ChatOk, DecisionOk } from "./api-schemas.ts";
import { chatReducer, initialConversation, type ConversationState } from "./conversation.ts";
import { toUiError } from "./errors.ts";

const metrics = { llmCalls: 1, latencyMs: 10, promptTokens: 5, tokenSource: "real", modelUsed: "m" };
const route = { route: "react", reason: "x", source: "router" };

function ok(conversationId = "c1", answer = "resposta"): ChatOk {
  return { requestId: "r1", answer, trace: [], route, conversationId, metrics };
}

function awaiting(conversationId = "c1"): ChatAwaitingApproval {
  return {
    requestId: "r2",
    status: "awaiting_approval",
    approval: { id: "a1", tool: "resolve_incident", args: { id: "INC-1" }, summary: "Resolver o incidente INC-1", reason: null, expiresAt: "2026-10-09T12:15:00.000Z" },
    trace: [],
    route,
    conversationId,
    metrics,
  };
}

const decision: DecisionOk = {
  requestId: "r3",
  answer: "Incidente INC-1 resolvido.",
  trace: [],
  route: null,
  metrics: null,
  conversationId: "c1",
  approval: { id: "a1", status: "approved" },
};

function sent(text = "oi"): ConversationState {
  return chatReducer(initialConversation, { type: "send", id: "u1", text });
}

describe("chatReducer — envio (US1)", () => {
  test("send adiciona a mensagem da pessoa e entra em sending", () => {
    const state = sent("  quais alertas?  ");
    assert.equal(state.pending, "sending");
    assert.deepEqual(state.items, [{ kind: "user", id: "u1", text: "quais alertas?", status: "sending" }]);
  });

  test("send vazio ou com resposta pendente devolve o mesmo estado", () => {
    assert.equal(chatReducer(initialConversation, { type: "send", id: "u1", text: "   " }), initialConversation);
    const state = sent();
    assert.equal(chatReducer(state, { type: "send", id: "u2", text: "outra" }), state);
  });

  test("received entrega a mensagem, adiciona a resposta e adota o conversationId", () => {
    const state = chatReducer(sent(), { type: "received", id: "a1", result: ok("c1") });
    assert.equal(state.pending, "idle");
    assert.equal(state.conversationId, "c1");
    assert.deepEqual(state.items[0], { kind: "user", id: "u1", text: "oi", status: "delivered" });
    assert.equal(state.items[1]?.kind, "assistant");
  });

  test("conversationId já adotado não muda com respostas seguintes", () => {
    let state = chatReducer(sent(), { type: "received", id: "a1", result: ok("c1") });
    state = chatReducer(state, { type: "send", id: "u2", text: "e agora?" });
    state = chatReducer(state, { type: "received", id: "a2", result: ok("outra") });
    assert.equal(state.conversationId, "c1");
  });

  test("failed marca a mensagem como falha e adiciona o erro com o texto para reenviar", () => {
    const error = toUiError({ status: 500, body: { error: "internal_error" } });
    const state = chatReducer(sent("oi"), { type: "failed", id: "e1", error });
    assert.equal(state.pending, "idle");
    assert.equal(state.items[0]?.kind === "user" && state.items[0].status, "failed");
    assert.deepEqual(state.items[1], { kind: "error", id: "e1", error, retryText: "oi", userItemId: "u1" });
  });

  test("retry remove a falha e reenvia o mesmo texto", () => {
    const error = toUiError({ status: 504, body: { error: "timeout" } });
    let state = chatReducer(sent("oi"), { type: "failed", id: "e1", error });
    state = chatReducer(state, { type: "retry", errorItemId: "e1", id: "u2" });
    assert.equal(state.pending, "sending");
    assert.deepEqual(state.items, [{ kind: "user", id: "u2", text: "oi", status: "sending" }]);
  });

  test("conversa inexistente mantém o conversationId até o reset", () => {
    let state = chatReducer(sent(), { type: "received", id: "a1", result: ok("c1") });
    state = chatReducer(state, { type: "send", id: "u2", text: "e?" });
    state = chatReducer(state, { type: "failed", id: "e1", error: toUiError({ status: 404, body: { error: "conversation_not_found" } }) });
    assert.equal(state.conversationId, "c1");
    assert.deepEqual(chatReducer(state, { type: "reset" }), initialConversation);
  });
});

describe("chatReducer — aprovação (US3)", () => {
  test("202 adiciona o cartão e bloqueia novos envios", () => {
    const state = chatReducer(sent("resolva"), { type: "received", id: "p1", result: awaiting("c9") });
    assert.equal(state.pending, "awaiting_decision");
    assert.equal(state.conversationId, "c9");
    const card = state.items[1];
    assert.equal(card?.kind, "approval");
    assert.equal(card?.kind === "approval" && card.run.answer, null);
    assert.equal(chatReducer(state, { type: "send", id: "u2", text: "outra" }), state);
  });

  test("decisão aprovada libera o envio e anexa a resposta logo após o cartão", () => {
    let state = chatReducer(sent("resolva"), { type: "received", id: "p1", result: awaiting() });
    state = chatReducer(state, { type: "approvalUpdated", itemId: "p1", action: { type: "decide", decision: "approve" } });
    assert.equal(state.pending, "awaiting_decision");
    state = chatReducer(state, { type: "approvalUpdated", itemId: "p1", action: { type: "succeeded", status: "approved" } });
    state = chatReducer(state, { type: "decisionReceived", itemId: "p1", id: "d1", data: decision });
    assert.equal(state.pending, "idle");
    assert.deepEqual(
      state.items.map((item) => item.kind),
      ["user", "approval", "assistant"],
    );
    const answer = state.items[2];
    assert.equal(answer?.kind === "assistant" && answer.run.answer, "Incidente INC-1 resolvido.");
    assert.equal(answer?.kind === "assistant" && answer.run.route, null);
  });

  test("falha de rede devolve o cartão a pending e mantém o bloqueio", () => {
    let state = chatReducer(sent("resolva"), { type: "received", id: "p1", result: awaiting() });
    state = chatReducer(state, { type: "approvalUpdated", itemId: "p1", action: { type: "decide", decision: "deny" } });
    state = chatReducer(state, {
      type: "approvalUpdated",
      itemId: "p1",
      action: { type: "failed", error: toUiError({ exception: new TypeError("x") }) },
    });
    const card = state.items[1];
    assert.equal(card?.kind === "approval" && card.card.status, "pending");
    assert.equal(state.pending, "awaiting_decision");
  });

  test("cartão indisponível (expirou) libera o envio", () => {
    let state = chatReducer(sent("resolva"), { type: "received", id: "p1", result: awaiting() });
    state = chatReducer(state, { type: "approvalUpdated", itemId: "p1", action: { type: "decide", decision: "approve" } });
    state = chatReducer(state, {
      type: "approvalUpdated",
      itemId: "p1",
      action: { type: "failed", error: toUiError({ status: 410, body: { error: "approval_expired" } }) },
    });
    assert.equal(state.pending, "idle");
  });
});
