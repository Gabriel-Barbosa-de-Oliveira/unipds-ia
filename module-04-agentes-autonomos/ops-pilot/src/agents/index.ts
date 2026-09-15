import type { StructuredToolInterface } from "@langchain/core/tools";

import { UnknownStrategyError } from "../domain/errors.ts";
import { createPlanAndExecuteStrategy, planAndExecuteStrategy } from "./plan-and-execute.ts";
import { createReactStrategy, reactStrategy } from "./react.ts";
import { withReflection } from "./reflection.ts";
import { opsTools } from "./tools.ts";
import type { ReasoningStrategy } from "./types.ts";

export type BaseStrategyName = "react" | "plan-and-execute";

export const DEFAULT_STRATEGY_NAME: BaseStrategyName = "react";

/** Registro de estratégias base disponíveis ao endpoint HTTP (nome -> estratégia). */
export const STRATEGIES: Record<BaseStrategyName, ReasoningStrategy> = {
  react: reactStrategy,
  "plan-and-execute": planAndExecuteStrategy,
};

function isBaseStrategyName(name: string): name is BaseStrategyName {
  return name in STRATEGIES;
}

/**
 * Resolve um nome de estratégia (ou o padrão, quando omitido) para a `ReasoningStrategy`
 * executável correspondente. Quando `extraTools` é informado e não vazio, compõe uma estratégia
 * nova por requisição sobre `[...opsTools, ...extraTools]` (mesmas fábricas já usadas por
 * `bench.ts`) em vez do singleton — usado por `007-semantic-memory` para disponibilizar
 * `remember_fact`/`forget_fact` escopadas a um `userId` específico (research.md item 7). Quando
 * omitido, comportamento idêntico ao de antes: mesmo singleton, nenhuma IO extra.
 */
export function resolveStrategy(
  name?: string,
  reflect?: boolean,
  extraTools?: StructuredToolInterface[],
): ReasoningStrategy {
  const resolvedName = name ?? DEFAULT_STRATEGY_NAME;

  if (!isBaseStrategyName(resolvedName)) {
    throw new UnknownStrategyError(resolvedName);
  }

  const strategy =
    extraTools && extraTools.length > 0
      ? buildStrategyWithExtraTools(resolvedName, extraTools)
      : STRATEGIES[resolvedName];

  return reflect ? withReflection(strategy) : strategy;
}

function buildStrategyWithExtraTools(
  name: BaseStrategyName,
  extraTools: StructuredToolInterface[],
): ReasoningStrategy {
  const tools = [...opsTools, ...extraTools];
  return name === "react" ? createReactStrategy(tools) : createPlanAndExecuteStrategy(tools);
}
