import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { buildRequestRecord, type RequestRecord } from "./request-record.ts";
import {
  computeStats,
  loadModelPrices,
  parseStatsWindow,
  percentile,
  promptPriceOf,
  UNKNOWN_GROUP,
} from "./request-stats.ts";

function ok(route: "react" | "planExecute" | "reflect", modelUsed: string, promptTokens: number, durationMs: number): RequestRecord {
  return buildRequestRecord({
    requestId: crypto.randomUUID(),
    conversationId: "c",
    startedAt: new Date("2026-10-09T12:00:00.000Z"),
    durationMs,
    outcome: "ok",
    route: { route, reason: "x", source: "router" },
    metrics: {
      llmCalls: 1,
      latencyMs: durationMs,
      promptTokens,
      tokenSource: "real",
      modelUsed,
      historyMessages: 0,
      contextBreakdown: { system: 0, summary: 0, currentMessage: 1, conversationHistory: 0, recalledFacts: 0, total: 1 },
      contextTrimmed: { historyMessages: 0, recalledFacts: 0 },
    },
  });
}

function failed(outcome: "timeout" | "error", durationMs: number): RequestRecord {
  return buildRequestRecord({
    requestId: crypto.randomUUID(),
    conversationId: "c",
    startedAt: new Date("2026-10-09T12:00:00.000Z"),
    durationMs,
    outcome,
    errorType: outcome === "timeout" ? "ChatTimeoutError" : "Error",
  });
}

describe("parseStatsWindow", () => {
  test("aceita minutos, horas e dias", () => {
    assert.equal(parseStatsWindow("30m"), 30 * 60_000);
    assert.equal(parseStatsWindow("24h"), 24 * 3_600_000);
    assert.equal(parseStatsWindow("7d"), 7 * 86_400_000);
  });

  test("rejeita formato inválido, zero e janela acima de 90d", () => {
    for (const value of ["", "24", "h", "0h", "-1h", "1.5h", "24x", "91d", "24h "]) {
      assert.equal(parseStatsWindow(value), undefined, value);
    }
  });
});

describe("preços", () => {
  test("loadModelPrices lê o JSON da env; ausente ou inválido vira tabela vazia", () => {
    assert.deepEqual(loadModelPrices({ OPSPILOT_MODEL_PRICES: '{"openai/gpt-4o-mini":0.15}' }), { "openai/gpt-4o-mini": 0.15 });
    for (const raw of [undefined, "", "não é json", '{"a":-1}', '{"a":"caro"}', "[1]"]) {
      assert.deepEqual(loadModelPrices({ OPSPILOT_MODEL_PRICES: raw }), {}, String(raw));
    }
  });

  test("modelos :free custam 0; sem preço configurado é undefined", () => {
    assert.equal(promptPriceOf("meta-llama/llama-3.1-8b-instruct:free", {}), 0);
    assert.equal(promptPriceOf("openai/gpt-4o-mini", { "openai/gpt-4o-mini": 0.15 }), 0.15);
    assert.equal(promptPriceOf("openai/gpt-4o-mini", {}), undefined);
    assert.equal(promptPriceOf("toString", {}), undefined);
  });
});

describe("percentile (nearest-rank)", () => {
  test("p50 e p95 sobre valores ordenados; null sem amostras", () => {
    const values = Array.from({ length: 20 }, (_, i) => (i + 1) * 100);
    assert.equal(percentile(values, 50), 1000);
    assert.equal(percentile(values, 95), 1900);
    assert.equal(percentile([42], 95), 42);
    assert.equal(percentile([], 50), null);
  });
});

describe("computeStats", () => {
  const PRICES = { "openai/gpt-4o-mini": 0.15 };

  test("janela vazia: zeros e latências nulas", () => {
    assert.deepEqual(computeStats([], PRICES), {
      total: 0,
      errors: 0,
      tokens: 0,
      costUsd: null,
      latencyMs: { p50: null, p95: null },
      timeouts: 0,
      errorRate: 0,
      unpricedModels: [],
      byRoute: {},
      byModel: {},
    });
  });

  test("agrega total, erros, tokens, custo e p50/p95, no total, por rota e por modelo", () => {
    const records = [
      ok("react", "openai/gpt-4o-mini", 1_000_000, 100),
      ok("react", "openai/gpt-4o-mini", 1_000_000, 300),
      ok("planExecute", "meta-llama/llama-3.1-8b-instruct:free", 500_000, 900),
      ok("reflect", "anthropic/sem-preco", 200, 500),
      failed("timeout", 20_000),
      failed("error", 50),
    ];

    const stats = computeStats(records, PRICES);

    assert.equal(stats.total, 6);
    assert.equal(stats.errors, 2);
    assert.equal(stats.timeouts, 1);
    assert.equal(stats.errorRate, 0.3333);
    assert.equal(stats.tokens, 2_500_200);
    // 2 × 1M tokens × 0.15 + :free (0); o modelo sem preço fica de fora e é sinalizado.
    assert.equal(stats.costUsd, 0.3);
    assert.deepEqual(stats.unpricedModels, ["anthropic/sem-preco"]);
    // latências ordenadas: 50, 100, 300, 500, 900, 20000
    assert.deepEqual(stats.latencyMs, { p50: 300, p95: 20_000 });

    assert.deepEqual(Object.keys(stats.byRoute), ["desconhecido", "planExecute", "react", "reflect"]);
    assert.deepEqual(stats.byRoute.react, {
      total: 2,
      errors: 0,
      tokens: 2_000_000,
      costUsd: 0.3,
      latencyMs: { p50: 100, p95: 300 },
    });
    assert.deepEqual(stats.byRoute[UNKNOWN_GROUP], {
      total: 2,
      errors: 2,
      tokens: 0,
      costUsd: null,
      latencyMs: { p50: 50, p95: 20_000 },
    });

    assert.equal(stats.byModel["meta-llama/llama-3.1-8b-instruct:free"]!.costUsd, 0);
    assert.equal(stats.byModel["anthropic/sem-preco"]!.costUsd, null);
    assert.equal(stats.byModel["openai/gpt-4o-mini"]!.total, 2);
  });
});
