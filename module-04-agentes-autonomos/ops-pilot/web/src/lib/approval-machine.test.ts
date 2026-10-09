import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { approvalReducer, initialApprovalCard, isAwaitingDecision, type ApprovalCardState } from "./approval-machine.ts";
import { toUiError } from "./errors.ts";

const submitting: ApprovalCardState = { status: "submitting", decision: "approve" };

describe("approvalReducer", () => {
  test("pending → submitting ao decidir", () => {
    assert.deepEqual(approvalReducer(initialApprovalCard, { type: "decide", decision: "deny" }), {
      status: "submitting",
      decision: "deny",
    });
  });

  test("submitting → approved | denied no sucesso", () => {
    assert.deepEqual(approvalReducer(submitting, { type: "succeeded", status: "approved" }), { status: "approved" });
    assert.deepEqual(approvalReducer(submitting, { type: "succeeded", status: "denied" }), { status: "denied" });
  });

  test("409/410/404 → indisponível com o motivo", () => {
    const cases = [
      [409, "Já decidida"],
      [410, "Expirou"],
      [404, "Não encontrada"],
    ] as const;
    for (const [status, reason] of cases) {
      const error = toUiError({ status, body: { error: "x" } });
      assert.deepEqual(approvalReducer(submitting, { type: "failed", error }), { status: "unavailable", reason });
    }
  });

  test("rede ou 5xx → volta a pending com o erro, permitindo tentar de novo", () => {
    const network = toUiError({ exception: new TypeError("Failed to fetch") });
    const next = approvalReducer(submitting, { type: "failed", error: network });
    assert.equal(next.status, "pending");
    assert.equal(next.status === "pending" ? next.error : undefined, network);

    const server = toUiError({ status: 500, body: { error: "internal_error" } });
    assert.equal(approvalReducer(submitting, { type: "failed", error: server }).status, "pending");
  });

  test("decidir fora de pending não muda nada (sem decisão em dobro)", () => {
    const states: ApprovalCardState[] = [
      submitting,
      { status: "approved" },
      { status: "denied" },
      { status: "unavailable", reason: "Expirou" },
    ];
    for (const state of states) {
      assert.equal(approvalReducer(state, { type: "decide", decision: "approve" }), state);
    }
  });

  test("resultados fora de submitting são ignorados", () => {
    assert.equal(approvalReducer(initialApprovalCard, { type: "succeeded", status: "approved" }), initialApprovalCard);
  });

  test("isAwaitingDecision só em pending/submitting", () => {
    assert.equal(isAwaitingDecision(initialApprovalCard), true);
    assert.equal(isAwaitingDecision(submitting), true);
    assert.equal(isAwaitingDecision({ status: "approved" }), false);
    assert.equal(isAwaitingDecision({ status: "unavailable", reason: "Expirou" }), false);
  });
});
