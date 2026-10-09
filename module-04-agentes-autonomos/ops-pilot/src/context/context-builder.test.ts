import assert from "node:assert/strict";
import { test } from "node:test";

import { composePrompt, type ConversationMessage } from "../domain/conversation.ts";
import { composeWithFacts } from "../domain/memory.ts";
import {
  buildContext,
  DEFAULT_CONTEXT_BUDGET,
  loadContextBudget,
  trimMemories,
  trimSummary,
  trimWindow,
  type ScoredMemory,
} from "./context-builder.ts";
import { estimateTokens } from "./tokens.ts";

const makeMessage = (content: string, role: ConversationMessage["role"] = "user"): ConversationMessage => ({ role, content });

test("trimSummary preserves text within budget and truncates the beginning when over budget", () => {
  assert.equal(trimSummary("short", 2), "short");
  assert.equal(trimSummary("0123456789", 2), "01234567");
  assert.ok(estimateTokens(trimSummary("0123456789", 2)) <= 2);
  assert.equal(trimSummary("summary", 0), "");
});

test("trimWindow removes the oldest complete messages and preserves chronological order", () => {
  const input = [makeMessage("1111"), makeMessage("2222", "assistant"), makeMessage("3333"), makeMessage("4444", "assistant")];

  assert.deepEqual(trimWindow(input, 5), input);
  assert.deepEqual(trimWindow(input, 3), input.slice(2));
  assert.deepEqual(trimWindow([makeMessage("too long")], 1), []);
  assert.deepEqual(trimWindow(input, 0), []);
});

test("trimMemories sorts by score, removes lowest scores first and preserves input order for ties", () => {
  const input: ScoredMemory[] = [
    { fact: "medium", score: 0.7 },
    { fact: "low", score: 0.5 },
    { fact: "high", score: 0.9 },
  ];

  assert.deepEqual(trimMemories(input, 100), [input[2], input[0], input[1]]);
  assert.deepEqual(trimMemories(input, 1), [input[2]]);
  assert.deepEqual(trimMemories([{ fact: "first", score: 1 }, { fact: "second", score: 1 }], 100), [
    { fact: "first", score: 1 },
    { fact: "second", score: 1 },
  ]);
  assert.deepEqual(trimMemories(input, 0), []);
});

test("buildContext: tetos baixos cortam na ordem certa e mantêm system/message intocados", () => {
  const system = "SYSTEM LONGO que não pode ser cortado";
  const message = "MENSAGEM ATUAL MUITO LONGA e intocável";
  const window = [makeMessage("1111"), makeMessage("2222", "assistant"), makeMessage("3333"), makeMessage("4444", "assistant")];
  const memories: ScoredMemory[] = [
    { fact: "low", score: 0.5 },
    { fact: "high", score: 0.9 },
    { fact: "medium", score: 0.7 },
  ];
  const result = buildContext(
    { system, message, summary: "01234567890123456789", window, memories },
    { summary: 2, window: 3, memories: 1 },
  );

  assert.deepEqual(result.window, window.slice(2));
  assert.deepEqual(result.memories, [memories[1]]);
  assert.equal(result.summary, "01234567");
  assert.deepEqual(result.trimmed, { historyMessages: 2, recalledFacts: 2 });
  assert.ok(result.prompt.includes(system));
  assert.ok(result.prompt.includes(message));
  assert.ok(!result.prompt.includes("1111"));
  assert.ok(!result.prompt.includes("2222"));
  assert.ok(!result.prompt.includes("- low"));
  assert.ok(!result.prompt.includes("- medium"));
  assert.deepEqual(result.breakdown, {
    system: estimateTokens(system),
    summary: 2,
    currentMessage: estimateTokens(message),
    conversationHistory: estimateTokens("3333\n4444"),
    recalledFacts: estimateTokens("high"),
    total:
      estimateTokens(system) +
      2 +
      estimateTokens(message) +
      estimateTokens("3333\n4444") +
      estimateTokens("high"),
  });
});

test("buildContext without optional sections preserves the legacy composition", () => {
  const message = "atual";
  const history = [makeMessage("antes"), makeMessage("resposta", "assistant")];
  const memories: ScoredMemory[] = [{ fact: "fato", score: 1 }];
  const cases = [
    { window: [], memories: [] },
    { window: history, memories: [] },
    { window: history, memories },
  ];

  for (const input of cases) {
    assert.equal(
      buildContext({ message, ...input }).prompt,
      composeWithFacts(input.memories.map(({ fact }) => fact), composePrompt(input.window, message)),
    );
  }
});

test("buildContext keeps section budgets independent and orders non-empty blocks", () => {
  const result = buildContext(
    {
      system: "system",
      summary: "summary",
      message: "current",
      window: [makeMessage("history")],
      memories: [{ fact: "memory", score: 1 }],
    },
    { summary: 10, window: 0, memories: 1 },
  );

  assert.deepEqual(result.window, []);
  assert.deepEqual(result.memories, []);
  assert.equal(result.prompt, "system\n\nResumo da conversa até aqui:\nsummary\n\ncurrent");
});

test("loadContextBudget uses defaults and validates each environment value independently", () => {
  assert.deepEqual(loadContextBudget({}), DEFAULT_CONTEXT_BUDGET);
  assert.deepEqual(loadContextBudget({ CONTEXT_BUDGET_WINDOW: "50" }), { summary: 200, window: 50, memories: 300 });
  assert.deepEqual(loadContextBudget({ CONTEXT_BUDGET_WINDOW: "0" }).window, 0);
  assert.deepEqual(
    loadContextBudget({ CONTEXT_BUDGET_SUMMARY: "", CONTEXT_BUDGET_WINDOW: "abc", CONTEXT_BUDGET_MEMORIES: "12" }),
    { summary: 200, window: 1200, memories: 12 },
  );

  for (const invalid of ["", "abc", "-5", "1.5"]) {
    assert.equal(loadContextBudget({ CONTEXT_BUDGET_WINDOW: invalid }).window, DEFAULT_CONTEXT_BUDGET.window);
  }
});
