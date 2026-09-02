import type { Metrics, TraceEvent } from "./types.ts";

/** Formata um único evento de trace como uma linha legível. Função pura, determinística. */
export function formatTraceEvent(event: TraceEvent): string {
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
  }
}

/** Formata uma sequência completa de trace, uma linha por evento, na ordem recebida. */
export function formatTrace(trace: readonly TraceEvent[]): string {
  return trace.map(formatTraceEvent).join("\n");
}

/** Formata as métricas de uma execução como uma linha legível. */
export function formatMetrics(metrics: Metrics): string {
  return `llmCalls=${metrics.llmCalls} latencyMs=${metrics.latencyMs}`;
}
