import assert from "node:assert/strict";
import { test } from "node:test";

import { tool } from "@langchain/core/tools";
import { z } from "zod";

import { UnknownStrategyError } from "../domain/errors.ts";
import { planAndExecuteStrategy } from "./plan-and-execute.ts";
import { reactStrategy } from "./react.ts";
import { resolveStrategy } from "./index.ts";

test("resolveStrategy(undefined) retorna a estratégia padrão (react)", () => {
  assert.equal(resolveStrategy(undefined), reactStrategy);
});

test("resolveStrategy('plan-and-execute') retorna a estratégia correspondente", () => {
  assert.equal(resolveStrategy("plan-and-execute"), planAndExecuteStrategy);
});

test("resolveStrategy('react') retorna a estratégia react", () => {
  assert.equal(resolveStrategy("react"), reactStrategy);
});

test("resolveStrategy lança UnknownStrategyError para um nome desconhecido", () => {
  assert.throws(() => resolveStrategy("nao-existe"), UnknownStrategyError);
});

test("resolveStrategy('react', true) retorna a estratégia decorada com reflection", () => {
  const strategy = resolveStrategy("react", true);
  assert.equal(strategy.name, "reflect:react");
});

test("resolveStrategy('plan-and-execute', true) retorna a estratégia decorada com reflection", () => {
  const strategy = resolveStrategy("plan-and-execute", true);
  assert.equal(strategy.name, "reflect:plan-and-execute");
});

test("resolveStrategy(name, false) retorna a estratégia base, sem decoração", () => {
  assert.equal(resolveStrategy("react", false), reactStrategy);
});

const fakeExtraTool = tool(async () => "ok", {
  name: "fake_extra_tool",
  description: "tool extra usada só para testar composição por requisição",
  schema: z.object({}),
});

test("resolveStrategy sem extraTools continua retornando exatamente o singleton (react)", () => {
  assert.equal(resolveStrategy("react", false, undefined), reactStrategy);
});

test("resolveStrategy com extraTools vazio continua retornando exatamente o singleton (plan-and-execute)", () => {
  assert.equal(resolveStrategy("plan-and-execute", false, []), planAndExecuteStrategy);
});

test("resolveStrategy com extraTools não vazio compõe uma estratégia nova, distinta do singleton", () => {
  const strategy = resolveStrategy("react", false, [fakeExtraTool]);

  assert.notEqual(strategy, reactStrategy);
  assert.equal(strategy.name, "react");
});

test("resolveStrategy com extraTools e reflect:true decora a estratégia nova, não o singleton", () => {
  const strategy = resolveStrategy("plan-and-execute", true, [fakeExtraTool]);

  assert.equal(strategy.name, "reflect:plan-and-execute");
});
