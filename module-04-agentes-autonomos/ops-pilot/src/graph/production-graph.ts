import { Annotation, END, START, StateGraph } from "@langchain/langgraph";

import { startTimer } from "../agents/metrics.ts";
import { tagTrace } from "../agents/trace.ts";
import type {
  ModelFallback,
  ProductionTraceEvent,
  ReasoningStrategy,
  RouteDecision,
  RouteName,
  RunResult,
} from "../agents/types.ts";
import {
  buildContext,
  type BuiltContext,
  type ContextBudget,
  type ContextInput,
} from "../context/context-builder.ts";
import { mergeTokenUsage, type TokenUsage } from "../context/tokens.ts";
import { resolveRouteDecision, type DecideRoute } from "./router.ts";

export interface ProductionGraphDeps {
  decideRoute: DecideRoute;
  strategyFor: (route: RouteName) => ReasoningStrategy;
}

export interface ProductionInput {
  context: ContextInput;
  budget: ContextBudget;
  /** Rota informada pelo cliente — quando presente, o roteador não é consultado (FR-008). */
  override?: RouteName;
}

export type ProductionRunResult = RunResult & {
  trace: ProductionTraceEvent[];
  route: RouteDecision;
  context: BuiltContext;
};

interface RouterUsage {
  llmCalls: number;
  tokenUsage?: TokenUsage;
}

const replace = <T>() => ({ reducer: (_prev: T, next: T) => next });

const GraphState = Annotation.Root({
  contextInput: Annotation<ContextInput>,
  budget: Annotation<ContextBudget>,
  override: Annotation<RouteName | undefined>(replace<RouteName | undefined>()),
  built: Annotation<BuiltContext>,
  decision: Annotation<RouteDecision>,
  routerUsage: Annotation<RouterUsage>,
  strategyResult: Annotation<RunResult>,
  trace: Annotation<ProductionTraceEvent[]>({ reducer: (prev, next) => prev.concat(next), default: () => [] }),
  result: Annotation<ProductionRunResult>,
});

type GraphStateType = typeof GraphState.State;

/** Soma as métricas do roteador às da estratégia (FR-013); `latencyMs` cobre o grafo inteiro. */
function combineMetrics(router: RouterUsage, strategy: RunResult["metrics"], latencyMs: number): RunResult["metrics"] {
  const strategyUsage: TokenUsage = { promptTokens: strategy.promptTokens, source: strategy.tokenSource };
  const usage = router.tokenUsage ? mergeTokenUsage(router.tokenUsage, strategyUsage) : strategyUsage;
  return {
    llmCalls: router.llmCalls + strategy.llmCalls,
    latencyMs,
    promptTokens: usage.promptTokens,
    tokenSource: usage.source,
    // O roteador não produz a resposta final — `modelUsed` é o da estratégia (013).
    modelUsed: strategy.modelUsed,
  };
}

/**
 * Grafo de produção (spec 012): contexto → roteador → uma estratégia (react | planExecute | reflect)
 * → resposta. As estratégias existentes rodam sem alteração dentro dos nós; o grafo só carimba
 * `node`, reindexa `at` e soma as métricas. Toda IO entra por `deps` (research.md item 9).
 */
export function createProductionGraph(deps: ProductionGraphDeps) {
  async function contextNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
    return { built: buildContext(state.contextInput, state.budget) };
  }

  async function routerNode(state: GraphStateType): Promise<Partial<GraphStateType>> {
    let decision: RouteDecision;
    let routerUsage: RouterUsage;
    let fallbacks: ModelFallback[] = [];

    if (state.override) {
      decision = resolveRouteDecision({ override: state.override });
      routerUsage = { llmCalls: 0 };
    } else {
      try {
        const routed = await deps.decideRoute(state.built.prompt);
        decision = resolveRouteDecision({ decided: routed.decided });
        routerUsage = { llmCalls: 1, tokenUsage: routed.tokenUsage };
        fallbacks = routed.fallbacks;
      } catch (error) {
        decision = resolveRouteDecision({ error });
        routerUsage = { llmCalls: 1 };
      }
    }

    return {
      decision,
      routerUsage,
      // `route` fica em trace[0] (012); trocas de modelo do roteador (013) vêm logo depois.
      trace: [
        { type: "route", at: 0, node: "roteador", ...decision },
        ...fallbacks.map((fallback, index): ProductionTraceEvent => ({
          type: "fallback",
          at: 1 + index,
          node: "roteador",
          ...fallback,
        })),
      ],
    };
  }

  function strategyNode(route: RouteName) {
    return async (state: GraphStateType): Promise<Partial<GraphStateType>> => {
      const result = await deps.strategyFor(route).run(state.built.prompt);
      return { strategyResult: result, trace: tagTrace(result.trace, route, state.trace.length) };
    };
  }

  function answerNode(latency: () => number) {
    return async (state: GraphStateType): Promise<Partial<GraphStateType>> => ({
      result: {
        answer: state.strategyResult.answer,
        trace: state.trace,
        metrics: combineMetrics(state.routerUsage, state.strategyResult.metrics, latency()),
        route: state.decision,
        context: state.built,
      },
    });
  }

  return {
    async run(input: ProductionInput): Promise<ProductionRunResult> {
      const elapsed = startTimer();

      const graph = new StateGraph(GraphState)
        .addNode("contexto", contextNode)
        .addNode("roteador", routerNode)
        .addNode("react", strategyNode("react"))
        .addNode("planExecute", strategyNode("planExecute"))
        .addNode("reflect", strategyNode("reflect"))
        .addNode("resposta", answerNode(elapsed))
        .addEdge(START, "contexto")
        .addEdge("contexto", "roteador")
        .addConditionalEdges("roteador", (state: GraphStateType) => state.decision.route, {
          react: "react",
          planExecute: "planExecute",
          reflect: "reflect",
        })
        .addEdge("react", "resposta")
        .addEdge("planExecute", "resposta")
        .addEdge("reflect", "resposta")
        .addEdge("resposta", END)
        .compile();

      const state = await graph.invoke({
        contextInput: input.context,
        budget: input.budget,
        override: input.override,
      });
      return state.result;
    },
  };
}
