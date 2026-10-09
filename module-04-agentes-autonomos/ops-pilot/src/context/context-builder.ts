import { z } from "zod";

import { composePrompt, type ConversationMessage } from "../domain/conversation.ts";
import { composeWithFacts } from "../domain/memory.ts";
import { buildContextBreakdown, estimateTokens, type ContextBreakdown } from "./tokens.ts";

export interface ContextBudget {
  readonly summary: number;
  readonly window: number;
  readonly memories: number;
}

export const DEFAULT_CONTEXT_BUDGET: ContextBudget = { summary: 200, window: 1200, memories: 300 };

export interface ScoredMemory {
  readonly fact: string;
  readonly score: number;
}

export interface ContextInput {
  readonly message: string;
  readonly system?: string;
  readonly summary?: string;
  readonly window?: readonly ConversationMessage[];
  readonly memories?: readonly ScoredMemory[];
}

export interface BuiltContext {
  readonly prompt: string;
  readonly window: ConversationMessage[];
  readonly memories: ScoredMemory[];
  readonly summary: string;
  readonly breakdown: ContextBreakdown;
  readonly trimmed: { readonly historyMessages: number; readonly recalledFacts: number };
}

const configuredBudgetSchema = z.string().trim().min(1).pipe(z.coerce.number().int().min(0));

function configuredBudget(
  value: string | undefined,
  fallback: number,
): number {
  const parsed = configuredBudgetSchema.safeParse(value);
  return parsed.success ? parsed.data : fallback;
}

/** Loads each section limit independently; invalid or absent values use that section's default. */
export function loadContextBudget(env: Readonly<Record<string, string | undefined>>): ContextBudget {
  return {
    summary: configuredBudget(env.CONTEXT_BUDGET_SUMMARY, DEFAULT_CONTEXT_BUDGET.summary),
    window: configuredBudget(env.CONTEXT_BUDGET_WINDOW, DEFAULT_CONTEXT_BUDGET.window),
    memories: configuredBudget(env.CONTEXT_BUDGET_MEMORIES, DEFAULT_CONTEXT_BUDGET.memories),
  };
}

export function trimSummary(summary: string, budget: number): string {
  if (estimateTokens(summary) <= budget) {
    return summary;
  }
  return summary.slice(0, budget * 4);
}

export function trimWindow(window: readonly ConversationMessage[], budget: number): ConversationMessage[] {
  const kept = [...window];
  while (kept.length > 0 && estimateTokens(kept.map((message) => message.content).join("\n")) > budget) {
    kept.shift();
  }
  return kept;
}

export function trimMemories(memories: readonly ScoredMemory[], budget: number): ScoredMemory[] {
  const kept = [...memories].sort((a, b) => b.score - a.score);
  while (kept.length > 0 && estimateTokens(kept.map((memory) => memory.fact).join("\n")) > budget) {
    kept.pop();
  }
  return kept;
}

/** The single context composition point shared by chat, arena, bench, and all reasoning strategies. */
export function buildContext(
  input: ContextInput,
  budget: ContextBudget = DEFAULT_CONTEXT_BUDGET,
): BuiltContext {
  const sourceWindow = input.window ?? [];
  const sourceMemories = input.memories ?? [];
  const window = trimWindow(sourceWindow, budget.window);
  const memories = trimMemories(sourceMemories, budget.memories);
  const summary = trimSummary(input.summary ?? "", budget.summary);
  const system = input.system ?? "";
  const message = input.message;

  const composedPrompt = composeWithFacts(
    memories.map((memory) => memory.fact),
    composePrompt(window, message),
  );
  const summaryBlock = summary ? `Resumo da conversa até aqui:\n${summary}` : "";
  const prompt = [system, summaryBlock, composedPrompt].filter((block) => block.length > 0).join("\n\n");
  const breakdown = buildContextBreakdown({
    system,
    summary,
    currentMessage: message,
    historyTexts: window.map((item) => item.content),
    factTexts: memories.map((item) => item.fact),
  });

  return {
    prompt,
    window,
    memories,
    summary,
    breakdown,
    trimmed: {
      historyMessages: sourceWindow.length - window.length,
      recalledFacts: sourceMemories.length - memories.length,
    },
  };
}