import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { Serialized } from "@langchain/core/load/serializable";
import type { LLMResult } from "@langchain/core/outputs";

/** Estimativa rápida de tokens sem tokenizador real — ~4 caracteres por token (research.md item 2). */
export function estimateTokens(text: string): number {
  return Math.max(0, Math.ceil(text.length / 4));
}

export type TokenSource = "real" | "estimated" | "mixed";

export interface TokenUsage {
  promptTokens: number;
  source: TokenSource;
}

/** Combina dois `TokenUsage` já finalizados (ex.: tentativa + crítica em reflection.ts, research.md item 6). */
export function mergeTokenUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
  return {
    promptTokens: a.promptTokens + b.promptTokens,
    source: a.source === b.source ? a.source : "mixed",
  };
}

interface OpenAiTokenUsage {
  promptTokens?: number;
}

function realPromptTokensOf(output: LLMResult): number | undefined {
  const tokenUsage = (output.llmOutput as { tokenUsage?: OpenAiTokenUsage } | undefined)?.tokenUsage;
  return typeof tokenUsage?.promptTokens === "number" ? tokenUsage.promptTokens : undefined;
}

/**
 * Callback LangChain que agrega o uso de tokens de prompt de todas as chamadas de uma execução.
 * `handleLLMStart` dispara para toda chamada de chat model — mesmo fallback que já faz
 * `LlmCallCounter` (`src/agents/metrics.ts`) funcionar hoje (research.md item 3) — e guarda uma
 * estimativa por `runId`; `handleLLMEnd` usa o uso real relatado pelo provedor quando presente
 * (`output.llmOutput.tokenUsage.promptTokens`, campo do `@langchain/openai`, research.md item 1),
 * senão cai para a estimativa guardada.
 */
export class UsageCollector extends BaseCallbackHandler {
  name = "usage-collector";

  private readonly pendingEstimates = new Map<string, number>();
  private promptTokensTotal = 0;
  private sawReal = false;
  private sawEstimated = false;

  override handleLLMStart(_llm: Serialized, prompts: string[], runId: string): void {
    this.pendingEstimates.set(runId, estimateTokens(prompts.join("\n")));
  }

  override handleLLMEnd(output: LLMResult, runId: string): void {
    const estimate = this.pendingEstimates.get(runId) ?? 0;
    this.pendingEstimates.delete(runId);

    const real = realPromptTokensOf(output);
    if (real !== undefined) {
      this.promptTokensTotal += real;
      this.sawReal = true;
    } else {
      this.promptTokensTotal += estimate;
      this.sawEstimated = true;
    }
  }

  get tokenUsage(): TokenUsage {
    const source: TokenSource = this.sawReal && this.sawEstimated ? "mixed" : this.sawReal ? "real" : "estimated";
    return { promptTokens: this.promptTokensTotal, source };
  }
}

export interface ContextBreakdown {
  currentMessage: number;
  conversationHistory: number;
  recalledFacts: number;
  total: number;
}

/**
 * Decompõe o contexto composto pelo controller HTTP (mensagem atual, histórico, fatos
 * lembrados) — `total` é sempre a soma das partes, por construção (FR-006, research.md item 7).
 * Pura: cada parte usa `estimateTokens` sobre o texto bruto (nunca a string final formatada com
 * rótulos que `composePrompt`/`composeWithFacts` adicionam).
 */
export function buildContextBreakdown(parts: {
  currentMessage: string;
  historyTexts: readonly string[];
  factTexts: readonly string[];
}): ContextBreakdown {
  const currentMessage = estimateTokens(parts.currentMessage);
  const conversationHistory = estimateTokens(parts.historyTexts.join("\n"));
  const recalledFacts = estimateTokens(parts.factTexts.join("\n"));

  return {
    currentMessage,
    conversationHistory,
    recalledFacts,
    total: currentMessage + conversationHistory + recalledFacts,
  };
}
