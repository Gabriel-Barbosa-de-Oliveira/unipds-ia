import assert from "node:assert/strict";
import { test } from "node:test";

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
