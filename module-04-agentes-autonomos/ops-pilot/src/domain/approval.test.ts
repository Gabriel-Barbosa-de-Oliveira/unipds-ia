import assert from "node:assert/strict";
import { describe, test } from "node:test";

import type { TraceEvent } from "../agents/types.ts";
import {
  approvalAnswer,
  approvalTrace,
  buildPendingAction,
  effectiveStatus,
  isGatedTool,
  reasonFromTrace,
  summarizeAction,
} from "./approval.ts";

const NOW = new Date("2026-10-09T12:00:00.000Z");

function pending(overrides: Partial<Parameters<typeof buildPendingAction>[0]> = {}) {
  return buildPendingAction({
    id: "a1",
    requestId: "r1",
    conversationId: "c1",
    tool: "resolve_incident",
    args: { id: "INC-42" },
    reason: null,
    now: NOW,
    ttlMs: 15 * 60_000,
    ...overrides,
  });
}

describe("buildPendingAction", () => {
  test("nasce pending, com expiração a partir do ttl e sem decisão", () => {
    assert.deepEqual(pending({ userId: "gabriel" }), {
      id: "a1",
      requestId: "r1",
      conversationId: "c1",
      userId: "gabriel",
      tool: "resolve_incident",
      args: { id: "INC-42" },
      reason: null,
      status: "pending",
      createdAt: "2026-10-09T12:00:00.000Z",
      expiresAt: "2026-10-09T12:15:00.000Z",
      decidedAt: null,
      decisionRequestId: null,
    });
    assert.equal(pending().userId, null);
  });
});

describe("effectiveStatus", () => {
  test("pending antes de expirar continua pending; depois vira expired", () => {
    const action = pending();
    assert.equal(effectiveStatus(action, new Date("2026-10-09T12:14:59.999Z")), "pending");
    assert.equal(effectiveStatus(action, new Date("2026-10-09T12:15:00.000Z")), "expired");
  });

  test("decisões tomadas não expiram", () => {
    const later = new Date("2026-10-10T00:00:00.000Z");
    assert.equal(effectiveStatus({ ...pending(), status: "approved" }, later), "approved");
    assert.equal(effectiveStatus({ ...pending(), status: "denied" }, later), "denied");
  });
});

describe("isGatedTool", () => {
  test("só abrir e resolver incidente exigem aprovação", () => {
    assert.equal(isGatedTool("open_incident"), true);
    assert.equal(isGatedTool("resolve_incident"), true);
    assert.equal(isGatedTool("list_alerts"), false);
    assert.equal(isGatedTool("consultar_runbook"), false);
  });
});

describe("summarizeAction", () => {
  test("descreve a ação em português", () => {
    assert.equal(summarizeAction("resolve_incident", { id: "INC-42" }), "Resolver o incidente INC-42");
    assert.equal(
      summarizeAction("open_incident", { severity: "critical", service: "checkout-api", title: "Latência" }),
      "Abrir incidente critical em checkout-api: Latência",
    );
  });
});

describe("reasonFromTrace", () => {
  test("último pensamento antes da última ação com porta", () => {
    const trace: TraceEvent[] = [
      { type: "thought", at: 0, content: "vou listar" },
      { type: "action", at: 1, tool: "list_incidents", args: {} },
      { type: "observation", at: 2, result: "[]" },
      { type: "thought", at: 3, content: "o rollback resolveu, posso fechar" },
      { type: "action", at: 4, tool: "resolve_incident", args: { id: "INC-42" } },
      { type: "observation", at: 5, result: "{}" },
      { type: "thought", at: 6, content: "depois da ação" },
    ];
    assert.equal(reasonFromTrace(trace), "o rollback resolveu, posso fechar");
  });

  test("sem pensamento antes da ação → null", () => {
    assert.equal(reasonFromTrace([{ type: "action", at: 0, tool: "open_incident", args: {} }]), null);
    assert.equal(reasonFromTrace([]), null);
  });
});

describe("approvalAnswer", () => {
  test("aprovada com sucesso", () => {
    assert.equal(
      approvalAnswer({ tool: "resolve_incident", args: { id: "INC-42" }, decision: "approved", result: { id: "INC-42" } }),
      "Incidente INC-42 resolvido.",
    );
    assert.equal(
      approvalAnswer({
        tool: "open_incident",
        args: { title: "Latência", service: "checkout-api", severity: "high" },
        decision: "approved",
        result: { id: "INC-7", severity: "high" },
      }),
      "Incidente INC-7 aberto em checkout-api (high): Latência.",
    );
  });

  test("aprovada mas a ferramenta falhou com erro de domínio", () => {
    assert.equal(
      approvalAnswer({
        tool: "resolve_incident",
        args: { id: "INC-404" },
        decision: "approved",
        result: { error: "IncidentNotFoundError", id: "INC-404" },
      }),
      "Não foi possível resolver o incidente INC-404: o incidente não existe.",
    );
  });

  test("negada", () => {
    assert.equal(
      approvalAnswer({ tool: "resolve_incident", args: { id: "INC-42" }, decision: "denied" }),
      "Ação cancelada: resolver o incidente INC-42. Nada foi executado.",
    );
  });
});

describe("approvalTrace", () => {
  test("aprovada: action, observation e answer no nó aprovacao", () => {
    assert.deepEqual(
      approvalTrace({ tool: "resolve_incident", args: { id: "INC-42" }, decision: "approved", result: { id: "INC-42" } }),
      [
        { type: "action", at: 0, node: "aprovacao", tool: "resolve_incident", args: { id: "INC-42" } },
        { type: "observation", at: 1, node: "aprovacao", result: { id: "INC-42" } },
        { type: "answer", at: 2, node: "aprovacao", content: "Incidente INC-42 resolvido." },
      ],
    );
  });

  test("negada: só a resposta", () => {
    assert.deepEqual(approvalTrace({ tool: "resolve_incident", args: { id: "INC-42" }, decision: "denied" }), [
      { type: "answer", at: 0, node: "aprovacao", content: "Ação cancelada: resolver o incidente INC-42. Nada foi executado." },
    ]);
  });
});
