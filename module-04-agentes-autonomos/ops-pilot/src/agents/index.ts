import type { StructuredToolInterface } from "@langchain/core/tools";

import { UnknownStrategyError } from "../domain/errors.ts";
import { createPlanAndExecuteStrategy, planAndExecuteStrategy } from "./plan-and-execute.ts";
import { createReactStrategy, reactStrategy } from "./react.ts";
import { withReflection } from "./reflection.ts";
import { createTeamStrategy } from "../team/index.ts";
import { opsTools } from "./tools.ts";
import type { ReasoningStrategy, RouteName } from "./types.ts";

export type BaseStrategyName = "react" | "plan-and-execute";

/**
 * Estratégia usada por `resolveStrategy(undefined)`. Desde a 012 o /chat não depende mais dela: a
 * rota vem do roteador do grafo de produção (`src/graph/`), que usa `react` como fallback.
 */
export const DEFAULT_STRATEGY_NAME: BaseStrategyName = "react";

/** Registro de estratégias base (nome -> estratégia), resolvidas pelo grafo via `strategyForRoute`. */
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
 * omitido, comportamento idêntico ao de antes: mesmo singleton, nenhuma IO extra. `baseTools`
 * substitui `opsTools` — o /chat passa as ferramentas com porta de aprovação (015).
 */
export function resolveStrategy(
  name?: string,
  reflect?: boolean,
  extraTools?: StructuredToolInterface[],
  baseTools: StructuredToolInterface[] = opsTools,
): ReasoningStrategy {
  const resolvedName = name ?? DEFAULT_STRATEGY_NAME;

  // Equipe (017): sempre por requisição, sobre as ferramentas recebidas — que precisam ter a porta
  // de aprovação (a montagem falha se não tiverem). Memória do usuário fica fora dos papéis.
  if (resolvedName === "team") {
    return createTeamStrategy(baseTools);
  }

  if (!isBaseStrategyName(resolvedName)) {
    throw new UnknownStrategyError(resolvedName);
  }

  const customTools = baseTools !== opsTools || (extraTools !== undefined && extraTools.length > 0);
  const strategy = customTools
    ? buildStrategyWithTools(resolvedName, [...baseTools, ...(extraTools ?? [])])
    : STRATEGIES[resolvedName];

  return reflect ? withReflection(strategy) : strategy;
}

function buildStrategyWithTools(name: BaseStrategyName, tools: StructuredToolInterface[]): ReasoningStrategy {
  return name === "react" ? createReactStrategy(tools) : createPlanAndExecuteStrategy(tools);
}

/**
 * Resolve a rota escolhida pelo grafo de produção (spec 012) para a estratégia executável. `reflect`
 * é reflection sobre react; a flag `reflect` decora as demais rotas e é ignorada em `reflect` (não
 * há reflection duplo — research.md item 7). `resolve` é injetável para os fakes do /chat.
 */
export function strategyForRoute(
  route: RouteName,
  reflect?: boolean,
  extraTools?: StructuredToolInterface[],
  resolve: typeof resolveStrategy = resolveStrategy,
  baseTools: StructuredToolInterface[] = opsTools,
): ReasoningStrategy {
  if (route === "reflect") {
    return resolve("react", true, extraTools, baseTools);
  }
  if (route === "team") {
    return resolve("team", false, extraTools, baseTools);
  }
  return resolve(route === "planExecute" ? "plan-and-execute" : "react", reflect, extraTools, baseTools);
}
