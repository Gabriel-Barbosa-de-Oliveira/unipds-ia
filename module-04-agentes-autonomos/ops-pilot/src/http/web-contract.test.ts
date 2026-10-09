import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { describe, test } from "node:test";

import type { Express } from "express";

import type { RunResult } from "../agents/types.ts";
import { InMemoryOpsStore } from "../services/ops-store.memory.ts";
import { SqliteApprovalStore } from "../store/sqlite-approval-store.ts";
import { SqliteConversationStore } from "../store/sqlite-conversation-store.ts";
import { SqliteRequestStore } from "../store/sqlite-request-store.ts";
import {
  ApiErrorSchema,
  ChatAwaitingApprovalSchema,
  ChatOkSchema,
  DecisionOkSchema,
} from "../../web/src/lib/api-schemas.ts";
import { createApp, type CreateAppOptions } from "./server.ts";

/**
 * Contrato API ↔ war room (spec 015, research.md item 10): as respostas reais do `createApp` têm
 * que passar nos schemas zod que o `web/` usa. Se a API mudar o formato, este teste quebra aqui.
 */

const metrics = { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real" as const, modelUsed: "fake-model" };

function app(run: CreateAppOptions["resolveStrategy"], opsStore = new InMemoryOpsStore()): Express {
  return createApp({
    resolveStrategy: run,
    decideRoute: async () => ({
      decided: { route: "react", reason: "fake" },
      tokenUsage: { promptTokens: 0, source: "real" },
      fallbacks: [],
    }),
    conversationStore: new SqliteConversationStore(":memory:"),
    memoryStore: { remember: async () => ({ stored: true, id: "m" }), recall: async () => [], forget: async () => ({ removed: false }) },
    requestStore: new SqliteRequestStore(":memory:"),
    approvalStore: new SqliteApprovalStore(":memory:"),
    opsStore,
    logger: { log() {} },
  });
}

async function withServer<T>(express: Express, fn: (baseUrl: string) => Promise<T>): Promise<T> {
  const server = await new Promise<ReturnType<Express["listen"]>>((resolve) => {
    const listening = express.listen(0, () => resolve(listening));
  });
  try {
    return await fn(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

async function post(url: string, body: unknown): Promise<{ status: number; body: unknown }> {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return { status: response.status, body: await response.json() };
}

describe("contrato API ↔ war room", () => {
  test("200 do /chat (com trace de todos os tipos) passa no ChatOkSchema", async () => {
    const strategy = (): RunResult => ({
      answer: "nada disparando",
      trace: [
        { type: "thought", at: 0, content: "vou listar" },
        { type: "plan", at: 1, steps: ["listar"] },
        { type: "action", at: 2, tool: "list_alerts", args: { status: "firing" } },
        { type: "observation", at: 3, result: "[]" },
        { type: "critique", at: 4, content: "ok" },
        { type: "fallback", at: 5, from: "a", to: "b", reason: "429" },
        { type: "answer", at: 6, content: "nada disparando" },
      ],
      metrics,
    });
    await withServer(app(() => ({ name: "fake", run: async () => strategy() })), async (baseUrl) => {
      const { status, body } = await post(`${baseUrl}/chat`, { message: "o que está disparando?" });
      assert.equal(status, 200);
      const parsed = ChatOkSchema.parse(body);
      assert.equal(parsed.trace.some((event) => "unknown" in event), false, "todo evento da API é um tipo conhecido do web");
    });
  });

  test("400 do /chat passa no ApiErrorSchema", async () => {
    await withServer(app(() => ({ name: "fake", run: async () => ({ answer: "", trace: [], metrics }) })), async (baseUrl) => {
      const { status, body } = await post(`${baseUrl}/chat`, {});
      assert.equal(status, 400);
      assert.equal(ApiErrorSchema.parse(body).error, "invalid_body");
    });
  });

  test("202 do /chat e 200 da decisão passam nos schemas de aprovação", async () => {
    const opsStore = new InMemoryOpsStore();
    const [open] = await opsStore.listIncidents("open");
    const id = open?.id ?? "INC-1";
    const gated = app(
      (_name, _reflect, _extra, baseTools) => ({
        name: "fake",
        async run() {
          await baseTools?.find((candidate) => candidate.name === "resolve_incident")?.invoke({ id });
          return { answer: "x", trace: [{ type: "action", at: 0, tool: "resolve_incident", args: { id } }], metrics };
        },
      }),
      opsStore,
    );

    await withServer(gated, async (baseUrl) => {
      const awaiting = await post(`${baseUrl}/chat`, { message: "resolva" });
      assert.equal(awaiting.status, 202);
      const parsed = ChatAwaitingApprovalSchema.parse(awaiting.body);

      const decided = await post(`${baseUrl}/approvals/${parsed.approval.id}`, { decision: "approve" });
      assert.equal(decided.status, 200);
      assert.equal(DecisionOkSchema.parse(decided.body).approval.status, "approved");

      const again = await post(`${baseUrl}/approvals/${parsed.approval.id}`, { decision: "approve" });
      assert.equal(again.status, 409);
      assert.equal(ApiErrorSchema.parse(again.body).error, "approval_already_decided");
    });
  });
});
