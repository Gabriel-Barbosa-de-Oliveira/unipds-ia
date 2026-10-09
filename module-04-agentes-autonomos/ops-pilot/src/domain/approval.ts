import type { ProductionTraceEvent, TraceEvent } from "../agents/types.ts";

/**
 * Aprovação humana de ações que mudam a produção (spec 015, data-model.md). Tudo aqui é puro: o
 * IO (gravar, executar a ferramenta) fica no store e no controller.
 */

/** Ferramentas que nunca executam sem decisão humana (Princípio VI). */
export const GATED_TOOLS = ["open_incident", "resolve_incident"] as const;
export type GatedToolName = (typeof GATED_TOOLS)[number];

export function isGatedTool(name: string): name is GatedToolName {
  return (GATED_TOOLS as readonly string[]).includes(name);
}

export type ApprovalStatus = "pending" | "approved" | "denied";
export type EffectiveApprovalStatus = ApprovalStatus | "expired";
export type ApprovalDecision = "approved" | "denied";

export interface PendingAction {
  id: string;
  requestId: string;
  conversationId: string;
  userId: string | null;
  tool: GatedToolName;
  args: Record<string, unknown>;
  reason: string | null;
  status: ApprovalStatus;
  createdAt: string;
  expiresAt: string;
  decidedAt: string | null;
  decisionRequestId: string | null;
}

export interface PendingActionInput {
  id: string;
  requestId: string;
  conversationId: string;
  userId?: string | null;
  tool: GatedToolName;
  args: Record<string, unknown>;
  reason: string | null;
  now: Date;
  ttlMs: number;
}

export function buildPendingAction(input: PendingActionInput): PendingAction {
  return {
    id: input.id,
    requestId: input.requestId,
    conversationId: input.conversationId,
    userId: input.userId ?? null,
    tool: input.tool,
    args: input.args,
    reason: input.reason,
    status: "pending",
    createdAt: input.now.toISOString(),
    expiresAt: new Date(input.now.getTime() + input.ttlMs).toISOString(),
    decidedAt: null,
    decisionRequestId: null,
  };
}

/** `expired` é derivado, nunca gravado: pendente com prazo vencido (research.md item 3). */
export function effectiveStatus(action: PendingAction, now: Date): EffectiveApprovalStatus {
  if (action.status === "pending" && new Date(action.expiresAt).getTime() <= now.getTime()) {
    return "expired";
  }
  return action.status;
}

const text = (value: unknown, fallback = "?"): string =>
  typeof value === "string" && value.length > 0 ? value : fallback;

/** Descrição curta da ação, em português, para o cartão de aprovação. */
export function summarizeAction(tool: GatedToolName, args: Record<string, unknown>): string {
  if (tool === "resolve_incident") {
    return `Resolver o incidente ${text(args.id)}`;
  }
  return `Abrir incidente ${text(args.severity)} em ${text(args.service)}: ${text(args.title)}`;
}

/** Último `thought` antes da última ação com porta — o "motivo" mostrado no cartão. */
export function reasonFromTrace(trace: readonly TraceEvent[]): string | null {
  let lastGated = -1;
  trace.forEach((event, index) => {
    if (event.type === "action" && isGatedTool(event.tool)) {
      lastGated = index;
    }
  });
  for (let index = lastGated - 1; index >= 0; index -= 1) {
    const event = trace[index];
    if (event?.type === "thought") {
      return event.content;
    }
  }
  return null;
}

const DOMAIN_ERROR_DETAILS: Record<string, string> = {
  IncidentNotFoundError: "o incidente não existe",
  ServiceNotFoundError: "o serviço não existe",
  InvalidSeverityError: "a severidade é inválida",
};

const lowerFirst = (value: string) => value.charAt(0).toLowerCase() + value.slice(1);

export interface DecisionOutcomeInput {
  tool: GatedToolName;
  args: Record<string, unknown>;
  decision: ApprovalDecision;
  /** Resultado da ferramenta (incidente ou erro estruturado); ausente quando negada. */
  result?: unknown;
}

function errorCodeOf(result: unknown): string | undefined {
  if (result && typeof result === "object" && "error" in result && typeof result.error === "string") {
    return result.error;
  }
  return undefined;
}

/** A ação aprovada foi executada sem erro de domínio. */
export function decisionSucceeded(input: DecisionOutcomeInput): boolean {
  return input.decision === "approved" && errorCodeOf(input.result) === undefined;
}

/** Resposta final da decisão, sem chamar o modelo (research.md item 2). */
export function approvalAnswer(input: DecisionOutcomeInput): string {
  const summary = summarizeAction(input.tool, input.args);
  if (input.decision === "denied") {
    return `Ação cancelada: ${lowerFirst(summary)}. Nada foi executado.`;
  }

  const code = errorCodeOf(input.result);
  if (code !== undefined) {
    const detail = DOMAIN_ERROR_DETAILS[code] ?? "a operação falhou";
    return `Não foi possível ${lowerFirst(summary)}: ${detail}.`;
  }

  const result = (input.result ?? {}) as Record<string, unknown>;
  if (input.tool === "resolve_incident") {
    return `Incidente ${text(result.id, text(input.args.id))} resolvido.`;
  }
  return `Incidente ${text(result.id)} aberto em ${text(input.args.service)} (${text(result.severity, text(input.args.severity))}): ${text(input.args.title)}.`;
}

/** Trace da decisão, no nó `aprovacao`, para o "ver raciocínio" funcionar igual ao do /chat. */
export function approvalTrace(input: DecisionOutcomeInput): ProductionTraceEvent[] {
  const answer: ProductionTraceEvent = { type: "answer", at: 0, node: "aprovacao", content: approvalAnswer(input) };
  if (input.decision === "denied") {
    return [answer];
  }
  return [
    { type: "action", at: 0, node: "aprovacao", tool: input.tool, args: input.args },
    { type: "observation", at: 1, node: "aprovacao", result: input.result },
    { ...answer, at: 2 },
  ];
}
