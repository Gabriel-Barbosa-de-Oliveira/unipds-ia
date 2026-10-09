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
import { buildRequestRecord, chatMetricsOf } from "../domain/request-record.ts";
import { errorTypeOf, traceToLogEvents, type Logger } from "../obs/logger.ts";
import type { RequestStore } from "../services/request-store.repository.ts";
import { resolveRouteDecision, type DecideRoute } from "./router.ts";

export interface ProductionGraphDeps {
  decideRoute: DecideRoute;
  strategyFor: (route: RouteName) => ReasoningStrategy;
  /** Onde o nó `resposta` grava registro + trace (spec 014). Ausente: nada é gravado. */
  requestStore?: RequestStore;
  /** Logger JSON do nó `resposta` (spec 014). Ausente: nada é logado. */
  logger?: Logger;
  now?: () => Date;
}

/** Identificação da requisição HTTP que disparou a execução (spec 014). */
export interface RequestContext {
  requestId: string;
  conversationId: string | null;
  userId?: string;
  startedAt: Date;
  /**
   * true quando quem chamou já desistiu (ex.: timeout do /chat, que grava o próprio registro). A
   * execução não é cancelada, então o nó `resposta` precisa saber que não deve gravar nem logar.
   */
  abandoned?: () => boolean;
}

export interface ProductionInput {
  context: ContextInput;
  budget: ContextBudget;
  /** Rota informada pelo cliente — quando presente, o roteador não é consultado (FR-008). */
  override?: RouteName;
  request?: RequestContext;
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
  request: Annotation<RequestContext | undefined>(replace<RequestContext | undefined>()),
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

  /**
   * Nó `resposta`: consolida o resultado e, quando há requisição associada, grava registro +
   * trace numa transação e emite os logs de metadados (spec 014). Falha de gravação nunca derruba
   * a resposta (FR-008) — só vira `persistence.failed`.
   */
  function answerNode(latency: () => number) {
    const now = deps.now ?? (() => new Date());

    return async (state: GraphStateType): Promise<Partial<GraphStateType>> => {
      const result: ProductionRunResult = {
        answer: state.strategyResult.answer,
        trace: state.trace,
        metrics: combineMetrics(state.routerUsage, state.strategyResult.metrics, latency()),
        route: state.decision,
        context: state.built,
      };

      const request = state.request;
      if (request && !request.abandoned?.()) {
        const durationMs = now().getTime() - request.startedAt.getTime();
        await persist(
          buildRequestRecord({
            requestId: request.requestId,
            conversationId: request.conversationId,
            userId: request.userId,
            startedAt: request.startedAt,
            durationMs,
            outcome: "ok",
            route: result.route,
            metrics: chatMetricsOf(result.metrics, result.context),
          }),
          result.trace,
        );

        for (const event of traceToLogEvents(request.requestId, result.trace)) {
          deps.logger?.log(event);
        }
        deps.logger?.log({
          event: "request.completed",
          requestId: request.requestId,
          node: "resposta",
          durationMs,
          route: result.route.route,
          llmCalls: result.metrics.llmCalls,
          promptTokens: result.metrics.promptTokens,
          tokenSource: result.metrics.tokenSource,
          modelUsed: result.metrics.modelUsed,
          traceEvents: result.trace.length,
        });
      }

      return { result };
    };
  }

  async function persist(record: Parameters<RequestStore["save"]>[0], trace: ProductionTraceEvent[]): Promise<void> {
    if (!deps.requestStore) {
      return;
    }
    try {
      await deps.requestStore.save(record, trace);
    } catch (error) {
      deps.logger?.log({ event: "persistence.failed", requestId: record.requestId, errorType: errorTypeOf(error) });
    }
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
        .addNode("team", strategyNode("team"))
        .addNode("resposta", answerNode(elapsed))
        .addEdge(START, "contexto")
        .addEdge("contexto", "roteador")
        .addConditionalEdges("roteador", (state: GraphStateType) => state.decision.route, {
          react: "react",
          planExecute: "planExecute",
          reflect: "reflect",
          team: "team",
        })
        .addEdge("react", "resposta")
        .addEdge("planExecute", "resposta")
        .addEdge("reflect", "resposta")
        .addEdge("team", "resposta")
        .addEdge("resposta", END)
        .compile();

      const state = await graph.invoke({
        contextInput: input.context,
        budget: input.budget,
        override: input.override,
        request: input.request,
      });
      return state.result;
    },
  };
}
