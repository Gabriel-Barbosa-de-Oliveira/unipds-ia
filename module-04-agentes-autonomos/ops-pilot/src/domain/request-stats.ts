import { z } from "zod";

import type { RequestRecord } from "./request-record.ts";

const UNIT_MS = { m: 60_000, h: 3_600_000, d: 86_400_000 } as const;

/** Janela máxima aceita em `?since=` — 90 dias. */
const MAX_WINDOW_MS = 90 * UNIT_MS.d;

const windowSchema = z
  .string()
  .regex(/^[1-9]\d*[mhd]$/, "use <n>m, <n>h ou <n>d (ex.: 30m, 24h, 7d)")
  .transform((value) => Number(value.slice(0, -1)) * UNIT_MS[value.slice(-1) as keyof typeof UNIT_MS])
  .refine((ms) => ms <= MAX_WINDOW_MS, "janela máxima é 90d");

export const DEFAULT_STATS_WINDOW = "24h";

/** Converte `30m` / `24h` / `7d` em milissegundos; `undefined` quando inválido. Pura. */
export function parseStatsWindow(value: string): number | undefined {
  const parsed = windowSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

/** Preço de prompt por modelo, em USD por 1 milhão de tokens. */
export type ModelPrices = Readonly<Record<string, number>>;

const pricesSchema = z.record(z.string(), z.number().nonnegative());

/**
 * Lê `OPSPILOT_MODEL_PRICES` (JSON `{ "modelo": usdPorMilhaoDeTokensDePrompt }`). Pura. Ausente ou
 * inválido vira tabela vazia — o custo desses modelos fica de fora e é sinalizado, nunca inventado.
 */
export function loadModelPrices(env: Readonly<Record<string, string | undefined>>): ModelPrices {
  const raw = env.OPSPILOT_MODEL_PRICES;
  if (!raw) {
    return {};
  }
  try {
    const parsed = pricesSchema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : {};
  } catch {
    return {};
  }
}

/** USD por 1M tokens de prompt; modelos `:free` do OpenRouter custam 0; `undefined` = sem preço. */
export function promptPriceOf(model: string, prices: ModelPrices): number | undefined {
  if (model.endsWith(":free")) {
    return 0;
  }
  return Object.hasOwn(prices, model) ? prices[model] : undefined;
}

/** Percentil por nearest-rank sobre valores já ordenados; `null` sem amostras. Pura. */
export function percentile(sorted: readonly number[], p: number): number | null {
  if (sorted.length === 0) {
    return null;
  }
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(Math.max(rank, 1), sorted.length) - 1]!;
}

export interface StatsGroup {
  total: number;
  errors: number;
  tokens: number;
  /** Soma só dos modelos com preço conhecido; `null` quando nenhuma requisição do grupo tem preço. */
  costUsd: number | null;
  latencyMs: { p50: number | null; p95: number | null };
}

export interface RequestStats extends StatsGroup {
  timeouts: number;
  errorRate: number;
  /** Modelos usados na janela sem preço configurado — o custo deles não está em `costUsd`. */
  unpricedModels: string[];
  byRoute: Record<string, StatsGroup>;
  byModel: Record<string, StatsGroup>;
}

/** Chave de agrupamento para requisições sem rota/modelo (timeout ou erro antes da resposta). */
export const UNKNOWN_GROUP = "desconhecido";

function roundUsd(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

function costOf(record: RequestRecord, prices: ModelPrices): number | undefined {
  if (record.modelUsed === null || record.promptTokens === null) {
    return undefined;
  }
  const price = promptPriceOf(record.modelUsed, prices);
  return price === undefined ? undefined : (record.promptTokens / 1_000_000) * price;
}

function summarize(records: readonly RequestRecord[], prices: ModelPrices): StatsGroup {
  const latencies = records.map((record) => record.durationMs).sort((a, b) => a - b);
  const costs = records.map((record) => costOf(record, prices)).filter((cost): cost is number => cost !== undefined);

  return {
    total: records.length,
    errors: records.filter((record) => record.outcome !== "ok").length,
    tokens: records.reduce((sum, record) => sum + (record.promptTokens ?? 0), 0),
    costUsd: costs.length > 0 ? roundUsd(costs.reduce((sum, cost) => sum + cost, 0)) : null,
    latencyMs: { p50: percentile(latencies, 50), p95: percentile(latencies, 95) },
  };
}

function groupBy(
  records: readonly RequestRecord[],
  keyOf: (record: RequestRecord) => string | null,
  prices: ModelPrices,
): Record<string, StatsGroup> {
  const groups = new Map<string, RequestRecord[]>();
  for (const record of records) {
    const key = keyOf(record) ?? UNKNOWN_GROUP;
    groups.set(key, [...(groups.get(key) ?? []), record]);
  }
  return Object.fromEntries([...groups].sort(([a], [b]) => a.localeCompare(b)).map(([key, items]) => [key, summarize(items, prices)]));
}

/**
 * Agrega as execuções da janela: total, erros (timeout + erro), tokens de prompt, custo,
 * latência p50/p95 (ms, por nearest-rank) — no total, por rota e por modelo. Pura.
 */
export function computeStats(records: readonly RequestRecord[], prices: ModelPrices): RequestStats {
  const overall = summarize(records, prices);
  const unpriced = new Set(
    records
      .map((record) => record.modelUsed)
      .filter((model): model is string => model !== null && promptPriceOf(model, prices) === undefined),
  );

  return {
    ...overall,
    timeouts: records.filter((record) => record.outcome === "timeout").length,
    errorRate: overall.total === 0 ? 0 : Math.round((overall.errors / overall.total) * 10_000) / 10_000,
    unpricedModels: [...unpriced].sort(),
    byRoute: groupBy(records, (record) => record.route, prices),
    byModel: groupBy(records, (record) => record.modelUsed, prices),
  };
}
