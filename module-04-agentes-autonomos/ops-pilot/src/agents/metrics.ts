import { BaseCallbackHandler } from "@langchain/core/callbacks/base";

import type { UsageCollector } from "../context/tokens.ts";
import type { Metrics } from "./types.ts";

/**
 * Callback handler compartilhado que conta chamadas ao modelo. Usado por ambas as estratégias
 * (ReAct e Plan-and-Execute) para que a definição de "chamada de LLM" seja idêntica entre elas.
 */
export class LlmCallCounter extends BaseCallbackHandler {
  name = "llm-call-counter";
  calls = 0;

  override handleLLMStart(): void {
    this.calls += 1;
  }
}

/** Inicia um cronômetro; a função retornada dá o tempo decorrido em ms desde a chamada. */
export function startTimer(): () => number {
  const startedAt = Date.now();
  return () => Date.now() - startedAt;
}

export function buildMetrics(counter: LlmCallCounter, usageCollector: UsageCollector, latencyMs: number): Metrics {
  const { promptTokens, source } = usageCollector.tokenUsage;
  return { llmCalls: counter.calls, latencyMs, promptTokens, tokenSource: source };
}
