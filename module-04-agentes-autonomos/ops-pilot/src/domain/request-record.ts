import type { GraphNode, Metrics, RouteDecision, RouteName, RouteSource, TraceEvent } from "../agents/types.ts";
import type { ContextBreakdown, TokenSource } from "../context/tokens.ts";

export type RequestOutcome = "ok" | "timeout" | "error";

export interface ContextTrimmed {
  historyMessages: number;
  recalledFacts: number;
}

/** Métricas como saem na resposta do /chat: as do grafo + as de contexto. */
export type ChatMetrics = Metrics & {
  historyMessages: number;
  contextBreakdown: ContextBreakdown;
  contextTrimmed: ContextTrimmed;
};

/** Junta as métricas do grafo às de contexto, no formato da resposta do /chat. Pura. */
export function chatMetricsOf(
  metrics: Metrics,
  context: { window: readonly unknown[]; breakdown: ContextBreakdown; trimmed: ContextTrimmed },
): ChatMetrics {
  return {
    ...metrics,
    historyMessages: context.window.length,
    contextBreakdown: context.breakdown,
    contextTrimmed: context.trimmed,
  };
}

/** Uma execução do /chat, como fica persistida (spec 014, data-model.md). */
export interface RequestRecord {
  requestId: string;
  conversationId: string | null;
  userId: string | null;
  startedAt: string;
  durationMs: number;
  outcome: RequestOutcome;
  errorType: string | null;
  route: RouteName | null;
  routeSource: RouteSource | null;
  llmCalls: number | null;
  promptTokens: number | null;
  tokenSource: TokenSource | null;
  modelUsed: string | null;
  historyMessages: number | null;
  context: { breakdown: ContextBreakdown; trimmed: ContextTrimmed } | null;
}

export interface RequestRecordInput {
  requestId: string;
  conversationId: string | null;
  userId?: string | null;
  startedAt: Date;
  durationMs: number;
  outcome: RequestOutcome;
  errorType?: string;
  route?: RouteDecision;
  metrics?: ChatMetrics;
}

/** Monta o registro de uma requisição. Pura — métricas e rota só existem quando houve resposta. */
export function buildRequestRecord(input: RequestRecordInput): RequestRecord {
  const { metrics, route } = input;

  return {
    requestId: input.requestId,
    conversationId: input.conversationId,
    userId: input.userId ?? null,
    startedAt: input.startedAt.toISOString(),
    durationMs: Math.max(0, Math.round(input.durationMs)),
    outcome: input.outcome,
    errorType: input.outcome === "ok" ? null : (input.errorType ?? "Error"),
    route: route?.route ?? null,
    routeSource: route?.source ?? null,
    llmCalls: metrics?.llmCalls ?? null,
    promptTokens: metrics?.promptTokens ?? null,
    tokenSource: metrics?.tokenSource ?? null,
    modelUsed: metrics?.modelUsed ?? null,
    historyMessages: metrics?.historyMessages ?? null,
    context: metrics ? { breakdown: metrics.contextBreakdown, trimmed: metrics.contextTrimmed } : null,
  };
}

export interface StoredTraceEvent {
  position: number;
  type: TraceEvent["type"];
  node: GraphNode | null;
  payload: TraceEvent;
}

/** Converte o trace em linhas de `trace_events`: posição = `at`, payload = evento completo. Pura. */
export function toStoredTraceEvents(trace: readonly TraceEvent[]): StoredTraceEvent[] {
  return trace.map((event) => ({
    position: event.at,
    type: event.type,
    node: event.node ?? null,
    payload: event,
  }));
}

/** Reconstrói o trace na ordem original (por `position`), sem mutar a entrada (FR-007). Pura. */
export function restoreTrace(rows: readonly StoredTraceEvent[]): TraceEvent[] {
  return [...rows].sort((a, b) => a.position - b.position).map((row) => row.payload);
}
