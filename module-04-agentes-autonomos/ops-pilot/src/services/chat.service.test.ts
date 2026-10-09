import assert from "node:assert/strict";
import { test } from "node:test";

import { ChatTimeoutError } from "../domain/errors.ts";
import type { ReasoningStrategy, RunResult } from "../agents/types.ts";
import { runWithTimeout, withTimeout } from "./chat.service.ts";

function delayedStrategy(result: RunResult, delayMs: number): ReasoningStrategy {
  return {
    name: "delayed",
    run: () =>
      new Promise((resolve) => {
        setTimeout(() => resolve(result), delayMs);
      }),
  };
}

const FAKE_RESULT: RunResult = {
  answer: "ok",
  trace: [],
  metrics: { llmCalls: 1, latencyMs: 1, promptTokens: 0, tokenSource: "real", modelUsed: "fake-model" },
};

test("runWithTimeout resolve normalmente quando a estratégia termina antes do limite", async () => {
  const strategy = delayedStrategy(FAKE_RESULT, 5);
  const result = await runWithTimeout(strategy, "oi", undefined, 200);
  assert.deepEqual(result, FAKE_RESULT);
});

test("runWithTimeout rejeita com ChatTimeoutError quando a estratégia não termina a tempo", async () => {
  const strategy = delayedStrategy(FAKE_RESULT, 200);
  await assert.rejects(() => runWithTimeout(strategy, "oi", undefined, 10), ChatTimeoutError);
});

test("runWithTimeout limpa o timer interno ao resolver com sucesso (sem handle pendente)", async () => {
  const originalClearTimeout = globalThis.clearTimeout;
  let clearCalls = 0;
  globalThis.clearTimeout = ((...args: Parameters<typeof clearTimeout>) => {
    clearCalls += 1;
    return originalClearTimeout(...args);
  }) as typeof clearTimeout;

  try {
    await runWithTimeout(delayedStrategy(FAKE_RESULT, 5), "oi", undefined, 200);
    assert.equal(clearCalls, 1);
  } finally {
    globalThis.clearTimeout = originalClearTimeout;
  }
});

test("withTimeout resolve com o valor de run() quando termina antes do limite", async () => {
  assert.equal(await withTimeout(() => Promise.resolve(42), 200), 42);
});

test("withTimeout propaga o erro de run()", async () => {
  await assert.rejects(() => withTimeout(() => Promise.reject(new Error("falhou")), 200), /falhou/);
});

test("withTimeout rejeita com ChatTimeoutError quando run() não termina a tempo", async () => {
  await assert.rejects(
    () => withTimeout(() => new Promise((resolve) => setTimeout(resolve, 100)), 10),
    (error: unknown) => error instanceof ChatTimeoutError && error.timeoutMs === 10,
  );
});
