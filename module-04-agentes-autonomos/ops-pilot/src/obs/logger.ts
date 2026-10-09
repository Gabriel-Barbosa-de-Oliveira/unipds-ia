import type { GraphNode, RouteName, RouteSource, TraceEvent } from "../agents/types.ts";
import type { TokenSource } from "../context/tokens.ts";

/**
 * Eventos de log do OpsPilot (spec 014). União FECHADA e só com metadados: não há campo livre
 * (`message`, `payload`, `details`), então conteúdo de conversa — mensagem, resposta, args e
 * resultados de tools, motivo do roteador, `error.message` — não tem por onde entrar (FR-011).
 */
export type LogEvent =
  | {
      event: "request.received";
      requestId: string;
      method: string;
      path: string;
      hasConversationId: boolean;
      hasUserId: boolean;
      strategyOverride: RouteName | null;
    }
  | {
      event: "request.rejected";
      requestId: string;
      status: number;
      errorCode: "invalid_body" | "unknown_strategy" | "conversation_not_found";
    }
  | { event: "route.chosen"; requestId: string; node: GraphNode | null; position: number; route: RouteName; source: RouteSource }
  | { event: "model.fallback"; requestId: string; node: GraphNode | null; position: number; from: string; to: string }
  | { event: "tool.called"; requestId: string; node: GraphNode | null; position: number; tool: string }
  | {
      event: "request.completed";
      requestId: string;
      node: "resposta";
      durationMs: number;
      route: RouteName;
      llmCalls: number;
      promptTokens: number;
      tokenSource: TokenSource;
      modelUsed: string;
      traceEvents: number;
    }
  | { event: "request.failed"; requestId: string; status: number; errorType: string; durationMs: number }
  | { event: "persistence.failed"; requestId: string; errorType: string }
  | { event: "request.lookup"; requestId: string; found: boolean };

export type LogLevel = "info" | "warn" | "error";

const LEVELS: Record<LogEvent["event"], LogLevel> = {
  "request.received": "info",
  "request.rejected": "warn",
  "route.chosen": "info",
  "model.fallback": "warn",
  "tool.called": "info",
  "request.completed": "info",
  "request.failed": "error",
  "persistence.failed": "error",
  "request.lookup": "info",
};

/** Uma linha JSON autocontida: `ts`, `level`, `event` e os metadados do evento. Pura. */
export function formatLogLine(event: LogEvent, now: Date): string {
  const { event: type, ...fields } = event;
  return JSON.stringify({ ts: now.toISOString(), level: LEVELS[type], event: type, ...fields });
}

/**
 * Deriva os logs de rota, troca de modelo e tool de um trace já produzido. Pura. Cada evento é
 * montado campo a campo (nunca por spread do evento de trace): `reason`, `args`, `result` e
 * `content` ficam de fora; os demais tipos de evento são só conteúdo e não geram log.
 */
export function traceToLogEvents(requestId: string, trace: readonly TraceEvent[]): LogEvent[] {
  const events: LogEvent[] = [];

  for (const item of trace) {
    const base = { requestId, node: item.node ?? null, position: item.at };
    switch (item.type) {
      case "route":
        events.push({ event: "route.chosen", ...base, route: item.route, source: item.source });
        break;
      case "fallback":
        events.push({ event: "model.fallback", ...base, from: item.from, to: item.to });
        break;
      case "action":
        events.push({ event: "tool.called", ...base, tool: item.tool });
        break;
      default:
        break;
    }
  }

  return events;
}

/** Nome do erro para log — nunca a mensagem, que pode ecoar conteúdo da conversa. */
export function errorTypeOf(error: unknown): string {
  return error instanceof Error && error.name ? error.name : "Error";
}

export interface Logger {
  log(event: LogEvent): void;
}

/** Logger padrão: uma linha JSON por evento no stdout. `write`/`now` injetáveis em testes. */
export function createLogger(
  write: (line: string) => void = (line) => process.stdout.write(`${line}\n`),
  now: () => Date = () => new Date(),
): Logger {
  return {
    log(event) {
      write(formatLogLine(event, now()));
    },
  };
}
