import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { ModelUsageTracker, summarizeModelUsage, withModelFallbacks, type ModelCallEntry } from "./model-usage.ts";
import type { TraceEvent } from "./types.ts";

const CONFIG = { primary: "a", fallback: "b" };

const start = (runId: string, model: string): ModelCallEntry => ({ kind: "start", runId, model });
const end = (runId: string): ModelCallEntry => ({ kind: "end", runId });
const error = (runId: string, message: string): ModelCallEntry => ({ kind: "error", runId, error: message });

describe("summarizeModelUsage", () => {
  test("caminho feliz: sem fallback, modelUsed é o principal", () => {
    assert.deepEqual(summarizeModelUsage([start("1", "a"), end("1")], CONFIG), { fallbacks: [], modelUsed: "a" });
  });

  test("retry no principal não é fallback (FR-008)", () => {
    const log = [start("1", "a"), error("1", "429"), start("2", "a"), end("2")];
    assert.deepEqual(summarizeModelUsage(log, CONFIG), { fallbacks: [], modelUsed: "a" });
  });

  test("principal esgota as tentativas e a reserva responde: 1 fallback com o último erro", () => {
    const log = [
      start("1", "a"),
      error("1", "429 primeiro"),
      start("2", "a"),
      error("2", "429 Rate limit exceeded"),
      start("3", "b"),
      end("3"),
    ];
    assert.deepEqual(summarizeModelUsage(log, CONFIG), {
      fallbacks: [{ from: "a", to: "b", reason: "429 Rate limit exceeded" }],
      modelUsed: "b",
    });
  });

  test("duas chamadas independentes que caem na reserva geram 2 fallbacks", () => {
    const log = [start("1", "a"), error("1", "x"), start("2", "b"), end("2"), start("3", "a"), error("3", "y"), start("4", "b"), end("4")];
    const summary = summarizeModelUsage(log, CONFIG);
    assert.deepEqual(summary.fallbacks.map((f) => f.reason), ["x", "y"]);
  });

  test("modelUsed é o modelo do último end, mesmo depois de usar a reserva antes", () => {
    const log = [start("1", "a"), error("1", "x"), start("2", "b"), end("2"), start("3", "a"), end("3")];
    assert.deepEqual(summarizeModelUsage(log, CONFIG), {
      fallbacks: [{ from: "a", to: "b", reason: "x" }],
      modelUsed: "a",
    });
  });

  test("log vazio: modelUsed é o principal", () => {
    assert.deepEqual(summarizeModelUsage([], CONFIG), { fallbacks: [], modelUsed: "a" });
  });

  test("sem reserva configurada nunca há fallback", () => {
    const log = [start("1", "a"), error("1", "x"), start("2", "a"), end("2")];
    assert.deepEqual(summarizeModelUsage(log, { primary: "a" }), { fallbacks: [], modelUsed: "a" });
  });

  test("motivo longo é truncado em 200 caracteres e motivo vazio vira texto padrão", () => {
    const long = summarizeModelUsage([start("1", "a"), error("1", "x".repeat(500)), start("2", "b"), end("2")], CONFIG);
    assert.equal(long.fallbacks[0]!.reason.length, 200);

    const empty = summarizeModelUsage([start("1", "a"), error("1", "  "), start("2", "b"), end("2")], CONFIG);
    assert.ok(empty.fallbacks[0]!.reason.length > 0);
  });
});

describe("withModelFallbacks", () => {
  const TRACE: TraceEvent[] = [
    { type: "action", at: 0, tool: "list_alerts", args: {} },
    { type: "observation", at: 1, result: [] },
    { type: "answer", at: 2, content: "ok" },
  ];

  test("insere antes do último answer e reindexa at", () => {
    const merged = withModelFallbacks(TRACE, [{ from: "a", to: "b", reason: "429" }]);
    assert.deepEqual(
      merged.map((event) => [event.type, event.at]),
      [
        ["action", 0],
        ["observation", 1],
        ["fallback", 2],
        ["answer", 3],
      ],
    );
    assert.deepEqual(merged[2], { type: "fallback", at: 2, from: "a", to: "b", reason: "429" });
  });

  test("sem answer, insere no fim", () => {
    const merged = withModelFallbacks(TRACE.slice(0, 2), [{ from: "a", to: "b", reason: "x" }]);
    assert.equal(merged.at(-1)?.type, "fallback");
    assert.equal(merged.at(-1)?.at, 2);
  });

  test("sem fallbacks devolve um trace igual e nunca muta a entrada", () => {
    const snapshot = structuredClone(TRACE);
    assert.deepEqual(withModelFallbacks(TRACE, []), TRACE);
    withModelFallbacks(TRACE, [{ from: "a", to: "b", reason: "x" }]);
    assert.deepEqual(TRACE, snapshot);
  });
});

describe("ModelUsageTracker", () => {
  test("registra início (com o modelo de invocation_params), erro e fim por runId", () => {
    const tracker = new ModelUsageTracker();
    tracker.handleChatModelStart({ lc: 1, type: "not_implemented", id: [] }, [], "r1", undefined, {
      invocation_params: { model: "a" },
    });
    tracker.handleLLMError(new Error("429"), "r1");
    tracker.handleChatModelStart({ lc: 1, type: "not_implemented", id: [] }, [], "r2", undefined, {
      invocation_params: { model: "b" },
    });
    tracker.handleLLMEnd({ generations: [] }, "r2");

    assert.deepEqual(tracker.log, [start("r1", "a"), error("r1", "429"), start("r2", "b"), end("r2")]);
  });
});
