import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { Serialized } from "@langchain/core/load/serializable";
import type { BaseMessage } from "@langchain/core/messages";

import type { ModelConfig } from "./model.ts";
import type { ModelFallback, TraceEvent } from "./types.ts";

const MAX_REASON_LENGTH = 200;

export type ModelCallEntry =
  | { kind: "start"; runId: string; model: string }
  | { kind: "end"; runId: string }
  | { kind: "error"; runId: string; error: string };

export interface ModelUsageSummary {
  fallbacks: ModelFallback[];
  modelUsed: string;
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Callback que registra cada chamada de chat model (início, fim, erro) com o id do modelo — o
 * `withFallbacks` não avisa quando troca de modelo, então a troca é deduzida deste log por
 * `summarizeModelUsage` (research.md item 5).
 */
export class ModelUsageTracker extends BaseCallbackHandler {
  name = "model-usage-tracker";
  readonly log: ModelCallEntry[] = [];

  override handleChatModelStart(
    _llm: Serialized,
    _messages: BaseMessage[][],
    runId: string,
    _parentRunId?: string,
    extraParams?: Record<string, unknown>,
  ): void {
    const invocationParams = extraParams?.invocation_params as { model?: unknown } | undefined;
    this.log.push({ kind: "start", runId, model: String(invocationParams?.model ?? "desconhecido") });
  }

  override handleLLMEnd(_output: unknown, runId: string): void {
    this.log.push({ kind: "end", runId });
  }

  override handleLLMError(error: unknown, runId: string): void {
    this.log.push({ kind: "error", runId, error: messageOf(error) });
  }
}

function truncate(text: string): string {
  const trimmed = text.trim() || "erro sem mensagem";
  return trimmed.length > MAX_REASON_LENGTH ? `${trimmed.slice(0, MAX_REASON_LENGTH - 1)}…` : trimmed;
}

/**
 * Deduz do log as trocas principal → reserva e o modelo que produziu a resposta final. Pura.
 * Retry no mesmo modelo não é troca (FR-008); `modelUsed` é o modelo do último `end` (FR-009).
 */
export function summarizeModelUsage(log: readonly ModelCallEntry[], config: ModelConfig): ModelUsageSummary {
  const modelByRun = new Map<string, string>();
  const fallbacks: ModelFallback[] = [];
  let lastPrimaryError: string | undefined;
  let modelUsed = config.primary;

  for (const entry of log) {
    if (entry.kind === "start") {
      modelByRun.set(entry.runId, entry.model);
      if (config.fallback && entry.model === config.fallback && lastPrimaryError !== undefined) {
        fallbacks.push({ from: config.primary, to: config.fallback, reason: truncate(lastPrimaryError) });
        lastPrimaryError = undefined;
      }
      continue;
    }

    const model = modelByRun.get(entry.runId);
    if (entry.kind === "error") {
      if (model === config.primary) {
        lastPrimaryError = entry.error;
      }
      continue;
    }

    if (model !== undefined) {
      modelUsed = model;
      if (model === config.primary) {
        lastPrimaryError = undefined;
      }
    }
  }

  return { fallbacks, modelUsed };
}

/**
 * Insere os eventos `fallback` imediatamente antes do último `answer` (ou no fim, sem `answer`) e
 * reindexa `at`. Pura — a resposta continua sendo o último evento do trace (research.md item 6).
 */
export function withModelFallbacks(trace: readonly TraceEvent[], fallbacks: readonly ModelFallback[]): TraceEvent[] {
  if (fallbacks.length === 0) {
    return trace.map((event) => ({ ...event }));
  }

  const lastAnswerIndex = trace.map((event) => event.type).lastIndexOf("answer");
  const insertAt = lastAnswerIndex === -1 ? trace.length : lastAnswerIndex;
  const fallbackEvents: TraceEvent[] = fallbacks.map((fallback) => ({ type: "fallback", at: 0, ...fallback }));
  const merged = [...trace.slice(0, insertAt), ...fallbackEvents, ...trace.slice(insertAt)];

  return merged.map((event, index) => ({ ...event, at: index }));
}
