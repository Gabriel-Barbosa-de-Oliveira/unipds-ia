import assert from "node:assert/strict";
import { test } from "node:test";

import type { Serialized } from "@langchain/core/load/serializable";
import type { LLMResult } from "@langchain/core/outputs";

import { buildContextBreakdown, estimateTokens, mergeTokenUsage, UsageCollector, type TokenUsage } from "./tokens.ts";

const FAKE_LLM = {} as Serialized;

function fakeLLMResult(tokenUsage: { promptTokens: number } | undefined): LLMResult {
  return {
    generations: [],
    llmOutput: tokenUsage ? { tokenUsage } : {},
  };
}

test("estimateTokens: string vazia retorna 0", () => {
  assert.equal(estimateTokens(""), 0);
});

test("estimateTokens: string de 4 caracteres retorna 1", () => {
  assert.equal(estimateTokens("abcd"), 1);
});

test("estimateTokens: string de 5 caracteres arredonda para cima (2)", () => {
  assert.equal(estimateTokens("abcde"), 2);
});

test("estimateTokens: nunca retorna negativo", () => {
  assert.ok(estimateTokens("") >= 0);
});

test("mergeTokenUsage: real + real soma e mantém source real", () => {
  const a: TokenUsage = { promptTokens: 10, source: "real" };
  const b: TokenUsage = { promptTokens: 5, source: "real" };
  assert.deepEqual(mergeTokenUsage(a, b), { promptTokens: 15, source: "real" });
});

test("mergeTokenUsage: estimated + estimated soma e mantém source estimated", () => {
  const a: TokenUsage = { promptTokens: 10, source: "estimated" };
  const b: TokenUsage = { promptTokens: 5, source: "estimated" };
  assert.deepEqual(mergeTokenUsage(a, b), { promptTokens: 15, source: "estimated" });
});

test("mergeTokenUsage: real + estimated soma e vira mixed", () => {
  const a: TokenUsage = { promptTokens: 10, source: "real" };
  const b: TokenUsage = { promptTokens: 5, source: "estimated" };
  assert.deepEqual(mergeTokenUsage(a, b), { promptTokens: 15, source: "mixed" });
});

test("mergeTokenUsage: estimated + real também vira mixed", () => {
  const a: TokenUsage = { promptTokens: 5, source: "estimated" };
  const b: TokenUsage = { promptTokens: 10, source: "real" };
  assert.deepEqual(mergeTokenUsage(a, b), { promptTokens: 15, source: "mixed" });
});

test("UsageCollector: uma chamada com tokenUsage real presente -> source real, promptTokens exato", () => {
  const collector = new UsageCollector();

  collector.handleLLMStart(FAKE_LLM, ["mensagem de teste"], "run-1");
  collector.handleLLMEnd(fakeLLMResult({ promptTokens: 42 }), "run-1");

  assert.deepEqual(collector.tokenUsage, { promptTokens: 42, source: "real" });
});

test("UsageCollector: uma chamada sem tokenUsage -> usa a estimativa capturada em handleLLMStart, source estimated", () => {
  const collector = new UsageCollector();
  const prompt = "mensagem de teste com oito caracteres exatos";

  collector.handleLLMStart(FAKE_LLM, [prompt], "run-1");
  collector.handleLLMEnd(fakeLLMResult(undefined), "run-1");

  assert.deepEqual(collector.tokenUsage, { promptTokens: estimateTokens(prompt), source: "estimated" });
});

test("UsageCollector: duas chamadas, uma real e uma sem uso relatado -> soma e source mixed", () => {
  const collector = new UsageCollector();
  const prompt = "segunda chamada sem uso relatado";

  collector.handleLLMStart(FAKE_LLM, ["primeira chamada"], "run-1");
  collector.handleLLMEnd(fakeLLMResult({ promptTokens: 100 }), "run-1");

  collector.handleLLMStart(FAKE_LLM, [prompt], "run-2");
  collector.handleLLMEnd(fakeLLMResult(undefined), "run-2");

  assert.deepEqual(collector.tokenUsage, { promptTokens: 100 + estimateTokens(prompt), source: "mixed" });
});

test("UsageCollector: handleLLMEnd para runId desconhecido não lança erro, usa estimativa 0", () => {
  const collector = new UsageCollector();

  assert.doesNotThrow(() => collector.handleLLMEnd(fakeLLMResult(undefined), "run-nunca-iniciado"));
  assert.deepEqual(collector.tokenUsage, { promptTokens: 0, source: "estimated" });
});

test("buildContextBreakdown: partes vazias produzem todos os campos zero", () => {
  const breakdown = buildContextBreakdown({ currentMessage: "", historyTexts: [], factTexts: [] });

  assert.deepEqual(breakdown, { currentMessage: 0, conversationHistory: 0, recalledFacts: 0, total: 0 });
});

test("buildContextBreakdown: total é sempre a soma exata das partes (partes não-vazias)", () => {
  const breakdown = buildContextBreakdown({
    currentMessage: "quem cuida do checkout financeiro?",
    historyTexts: ["me chame de Gabriel", "Combinado, Gabriel!"],
    factTexts: ["Gabriel cuida de pagamentos", "Gabriel prefere respostas curtas"],
  });

  assert.equal(breakdown.total, breakdown.currentMessage + breakdown.conversationHistory + breakdown.recalledFacts);
  assert.ok(breakdown.currentMessage > 0);
  assert.ok(breakdown.conversationHistory > 0);
  assert.ok(breakdown.recalledFacts > 0);
});

test("buildContextBreakdown: só mensagem atual presente -> histórico e fatos zero, total igual à mensagem", () => {
  const breakdown = buildContextBreakdown({ currentMessage: "oi", historyTexts: [], factTexts: [] });

  assert.equal(breakdown.conversationHistory, 0);
  assert.equal(breakdown.recalledFacts, 0);
  assert.equal(breakdown.total, breakdown.currentMessage);
});
