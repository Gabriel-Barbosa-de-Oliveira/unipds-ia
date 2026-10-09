import type { GraphNode, Metrics, ProductionTraceEvent, TraceEvent } from "./types.ts";

/**
 * Formata um único evento de trace como uma linha legível. Função pura, determinística. Eventos
 * vindos do grafo de produção (com `node`) ganham o nó como prefixo.
 */
export function formatTraceEvent(event: TraceEvent): string {
  const line = formatEventBody(event);
  return event.node ? `${event.node} │ ${line}` : line;
}

function formatEventBody(event: TraceEvent): string {
  switch (event.type) {
    case "thought":
      return `[thought] ${event.content}`;
    case "plan":
      return `[plan] ${event.steps.map((step, index) => `${index + 1}) ${step}`).join("; ")}`;
    case "action":
      return `[action] tool=${event.tool} args=${JSON.stringify(event.args)}`;
    case "observation":
      return `[observation] result=${JSON.stringify(event.result)}`;
    case "critique":
      return `[critique] ${event.content}`;
    case "answer":
      return `[answer] ${event.content}`;
    case "route":
      return `[route] ${event.route} (${event.source}): ${event.reason}`;
    case "fallback":
      return `[fallback] ${event.from} → ${event.to}: ${event.reason}`;
  }
}

/**
 * Carimba `node` em cada evento e reindexa `at` sequencialmente a partir de `offset`. Pura: devolve
 * cópias, sem mutar a entrada (research.md item 5).
 */
export function tagTrace(
  events: readonly TraceEvent[],
  node: GraphNode,
  offset: number,
): ProductionTraceEvent[] {
  return events.map((event, index) => ({ ...event, node, at: offset + index }));
}

/** Formata uma sequência completa de trace, uma linha por evento, na ordem recebida. */
export function formatTrace(trace: readonly TraceEvent[]): string {
  return trace.map(formatTraceEvent).join("\n");
}

/** Formata as métricas de uma execução como uma linha legível. */
export function formatMetrics(metrics: Metrics): string {
  return `llmCalls=${metrics.llmCalls} latencyMs=${metrics.latencyMs} model=${metrics.modelUsed}`;
}
