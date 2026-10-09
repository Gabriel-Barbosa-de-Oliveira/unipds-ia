import type { TraceEvent } from "./api-schemas.ts";
import type { ChatRun } from "./conversation.ts";

/** Acima disso, argumentos e resultados aparecem recolhidos (FR-013). */
export const LONG_CONTENT_CHARS = 600;

export type TraceKind =
  | "route"
  | "thought"
  | "plan"
  | "action"
  | "observation"
  | "critique"
  | "fallback"
  | "answer"
  | "handoff"
  | "unknown";

export type TraceBody =
  | { kind: "text"; text: string }
  | { kind: "steps"; steps: string[] }
  | { kind: "code"; title?: string; text: string; long: boolean }
  | { kind: "route"; route: string; reason: string; source: string }
  | { kind: "fallback"; from: string; to: string; reason: string }
  | { kind: "handoff"; from: string; to: string; brief: string };

export interface TraceView {
  kind: TraceKind;
  label: string;
  icon: TraceKind;
  node: string | null;
  /** Papel da equipe que produziu o evento (017), já traduzido; `null` fora da equipe. */
  role: string | null;
  at: number;
  body: TraceBody;
}

const LABELS: Record<Exclude<TraceKind, "unknown">, string> = {
  route: "Rota",
  thought: "Pensamento",
  plan: "Plano",
  action: "Ação",
  observation: "Observação",
  critique: "Crítica",
  fallback: "Troca de modelo",
  answer: "Resposta",
  handoff: "Passagem",
};

const ROLE_LABELS: Record<string, string> = {
  analista: "Analista",
  planejador: "Planejador",
  executor: "Executor",
  supervisor: "Supervisor",
  done: "Fim",
};

export function roleLabel(role: string): string {
  return ROLE_LABELS[role] ?? role;
}

const SOURCE_LABELS: Record<string, string> = {
  router: "decidida pelo roteador",
  override: "escolhida pelo cliente",
  fallback: "fallback após falha do roteador",
};

/** Formata qualquer valor para leitura humana; strings que são JSON viram JSON indentado. */
export function prettyValue(value: unknown): string {
  if (typeof value === "string") {
    try {
      const parsed: unknown = JSON.parse(value);
      return typeof parsed === "object" && parsed !== null ? JSON.stringify(parsed, null, 2) : value;
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2) ?? String(value);
}

function code(text: string, title?: string): TraceBody {
  return title === undefined
    ? { kind: "code", text, long: text.length > LONG_CONTENT_CHARS }
    : { kind: "code", title, text, long: text.length > LONG_CONTENT_CHARS };
}

function bodyOf(event: TraceEvent): { kind: TraceKind; body: TraceBody } {
  if ("unknown" in event) {
    const { unknown: _marker, at: _at, node: _node, role: _role, ...rest } = event;
    return { kind: "unknown", body: code(prettyValue(rest)) };
  }
  switch (event.type) {
    case "route":
      return { kind: "route", body: { kind: "route", route: event.route, reason: event.reason, source: sourceLabel(event.source) } };
    case "thought":
      return { kind: "thought", body: { kind: "text", text: event.content } };
    case "plan":
      return { kind: "plan", body: { kind: "steps", steps: event.steps } };
    case "action":
      return { kind: "action", body: code(prettyValue(event.args), event.tool) };
    case "observation":
      return { kind: "observation", body: code(prettyValue(event.result)) };
    case "critique":
      return { kind: "critique", body: { kind: "text", text: event.content } };
    case "fallback":
      return { kind: "fallback", body: { kind: "fallback", from: event.from, to: event.to, reason: event.reason } };
    case "answer":
      return { kind: "answer", body: { kind: "text", text: event.content } };
    case "handoff":
      return { kind: "handoff", body: { kind: "handoff", from: roleLabel(event.from), to: roleLabel(event.to), brief: event.brief } };
  }
}

/** Como cada evento do trace aparece no painel de raciocínio (FR-010, FR-011). Pura. */
export function toTraceView(event: TraceEvent): TraceView {
  const { kind, body } = bodyOf(event);
  const label = kind === "unknown" ? `Evento: ${event.type}` : LABELS[kind];
  const role = typeof event.role === "string" ? roleLabel(event.role) : null;
  return { kind, label, icon: kind, node: event.node ?? null, role, at: event.at, body };
}

export function sourceLabel(source: string): string {
  return SOURCE_LABELS[source] ?? source;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }
  return `${(ms / 1000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} s`;
}

export interface RunSummary {
  requestId: string;
  route: { route: string; reason: string; source: string } | null;
  metrics: { label: string; value: string }[] | null;
}

/** Cabeçalho do painel: rota, métricas e id (FR-012). Rota e métricas somem quando `null`. */
export function summarizeRun(run: ChatRun): RunSummary {
  const metrics = run.metrics;
  return {
    requestId: run.requestId,
    route: run.route ? { route: run.route.route, reason: run.route.reason, source: sourceLabel(run.route.source) } : null,
    metrics: metrics
      ? [
          { label: "Chamadas ao modelo", value: String(metrics.llmCalls) },
          { label: "Tempo", value: formatDuration(metrics.latencyMs) },
          { label: "Modelo", value: metrics.modelUsed },
          {
            label: "Tokens de prompt",
            value: `${metrics.promptTokens.toLocaleString("pt-BR")}${metrics.tokenSource === "real" ? "" : " (estimado)"}`,
          },
        ]
      : null,
  };
}
