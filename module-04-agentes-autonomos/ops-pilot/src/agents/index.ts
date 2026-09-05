import { UnknownStrategyError } from "../domain/errors.ts";
import { planAndExecuteStrategy } from "./plan-and-execute.ts";
import { reactStrategy } from "./react.ts";
import { withReflection } from "./reflection.ts";
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
 * executável correspondente. Pura: nenhuma IO, mesma entrada sempre produz a mesma resolução.
 */
export function resolveStrategy(name?: string, reflect?: boolean): ReasoningStrategy {
  const resolvedName = name ?? DEFAULT_STRATEGY_NAME;

  if (!isBaseStrategyName(resolvedName)) {
    throw new UnknownStrategyError(resolvedName);
  }

  const strategy = STRATEGIES[resolvedName];
  return reflect ? withReflection(strategy) : strategy;
}
