import assert from "node:assert/strict";
import { test } from "node:test";

import { ChatTimeoutError } from "../domain/errors.ts";
import type { ReasoningStrategy, RunResult } from "../agents/types.ts";
import { runWithTimeout } from "./chat.service.ts";

function delayedStrategy(result: RunResult, delayMs: number): ReasoningStrategy {
  return {
    name: "delayed",
    run: () =>
      new Promise((resolve) => {
        setTimeout(() => resolve(result), delayMs);
      }),
  };
}

const FAKE_RESULT: RunResult = { answer: "ok", trace: [], metrics: { llmCalls: 1, latencyMs: 1 } };

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
