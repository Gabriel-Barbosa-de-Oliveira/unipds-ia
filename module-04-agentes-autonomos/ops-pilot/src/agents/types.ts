import type { TokenSource } from "../context/tokens.ts";

export type ToolName =
  | "list_alerts"
  | "open_incident"
  | "resolve_incident"
  | "list_incidents"
  | "consultar_runbook";

/** Rotas que o roteador do grafo de produção pode escolher (spec 012). */
export const ROUTE_NAMES = ["react", "planExecute", "reflect", "team"] as const;
export type RouteName = (typeof ROUTE_NAMES)[number];

/** Origem da decisão de rota: modelo, override do cliente ou fallback por falha do roteador. */
export type RouteSource = "router" | "override" | "fallback";

export interface RouteDecision {
  route: RouteName;
  reason: string;
  source: RouteSource;
}

/** Nós do grafo de produção (`src/graph/production-graph.ts`); nós de estratégia têm o nome da rota. `aprovacao` marca o trace de uma decisão humana (015), fora do grafo. */
export type GraphNode = "contexto" | "roteador" | RouteName | "resposta" | "aprovacao";

/** Papéis do modo equipe (spec 017). O supervisor só aparece em `role` e como origem de `handoff`. */
export const TEAM_ROLES = ["analista", "planejador", "executor"] as const;
export type TeamRole = (typeof TEAM_ROLES)[number];

/** Destino de uma passagem: um papel ou `done` (encerramento pelo supervisor). */
export type HandoffTarget = TeamRole | "done";

/**
 * `node` é opcional porque arena/bench rodam estratégias fora do grafo; no grafo é sempre presente.
 * `role` só existe em eventos produzidos dentro da equipe (017).
 */
export type TraceEvent = (
  | { type: "thought"; at: number; content: string }
  | {
      type: "action";
      at: number;
      tool: ToolName;
      args: Record<string, unknown>;
    }
  | { type: "observation"; at: number; result: unknown }
  | { type: "plan"; at: number; steps: string[] }
  | { type: "critique"; at: number; content: string }
  | { type: "answer"; at: number; content: string }
  | { type: "route"; at: number; route: RouteName; reason: string; source: RouteSource }
  | { type: "fallback"; at: number; from: string; to: string; reason: string }
  | { type: "handoff"; at: number; from: "supervisor"; to: HandoffTarget; brief: string }
) & { node?: GraphNode; role?: TeamRole | "supervisor" };

/** Troca do modelo principal para o de reserva numa chamada ao modelo (spec 013). */
export interface ModelFallback {
  from: string;
  to: string;
  reason: string;
}

/** Evento de trace produzido pelo grafo de produção — `node` garantido (FR-006). */
export type ProductionTraceEvent = TraceEvent & { node: GraphNode };

export interface Metrics {
  llmCalls: number;
  latencyMs: number;
  promptTokens: number;
  tokenSource: TokenSource;
  /** Modelo que produziu a resposta final (spec 013, FR-009). */
  modelUsed: string;
}

export interface RunResult {
  answer: string;
  trace: TraceEvent[];
  metrics: Metrics;
}

export interface RunOptions {
  maxIterations?: number;
  /** Aplica-se apenas à estratégia plan-and-execute: pula o replanner e executa o plano inicial até o fim. */
  noReplanner?: boolean;
}

export interface ReasoningStrategy {
  readonly name: string;
  run(input: string, options?: RunOptions): Promise<RunResult>;
}
