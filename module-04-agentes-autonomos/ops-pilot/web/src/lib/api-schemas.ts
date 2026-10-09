import { z } from "zod";

/**
 * Schemas das respostas da API OpsPilot (specs/015-war-room-web/contracts/http.md). Toda resposta
 * passa por aqui antes de virar estado: para o navegador a API é entrada externa (Princípio II).
 * O trace espelha `TraceEvent` de `src/agents/types.ts`; `src/http/web-contract.test.ts` garante
 * que os dois lados não divergem.
 */

export const GraphNodeSchema = z.string();

export const RouteDecisionSchema = z.object({
  route: z.string(),
  reason: z.string(),
  source: z.string(),
});

export const MetricsSchema = z
  .object({
    llmCalls: z.number(),
    latencyMs: z.number(),
    promptTokens: z.number(),
    tokenSource: z.string(),
    modelUsed: z.string(),
  })
  .passthrough();

// `role` só existe em eventos da equipe (spec 017).
const base = { at: z.number(), node: GraphNodeSchema.optional(), role: z.string().optional() };

const KnownTraceEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("thought"), content: z.string(), ...base }),
  z.object({ type: z.literal("plan"), steps: z.array(z.string()), ...base }),
  z.object({ type: z.literal("action"), tool: z.string(), args: z.record(z.unknown()), ...base }),
  z.object({ type: z.literal("observation"), result: z.unknown(), ...base }),
  z.object({ type: z.literal("critique"), content: z.string(), ...base }),
  z.object({ type: z.literal("answer"), content: z.string(), ...base }),
  z.object({ type: z.literal("route"), route: z.string(), reason: z.string(), source: z.string(), ...base }),
  z.object({ type: z.literal("fallback"), from: z.string(), to: z.string(), reason: z.string(), ...base }),
  z.object({ type: z.literal("handoff"), from: z.string(), to: z.string(), brief: z.string(), ...base }),
]);

/** Evento de um tipo que a war room não conhece: aparece de forma genérica (FR-011). */
export const UnknownTraceEventSchema = z
  .object({ type: z.string(), ...base })
  .passthrough()
  .transform((event) => ({ ...event, unknown: true as const }));

export const TraceEventSchema = z.union([KnownTraceEventSchema, UnknownTraceEventSchema]);

export const ChatOkSchema = z.object({
  requestId: z.string(),
  answer: z.string(),
  trace: z.array(TraceEventSchema),
  route: RouteDecisionSchema,
  conversationId: z.string(),
  metrics: MetricsSchema,
});

export const ApprovalSchema = z.object({
  id: z.string(),
  tool: z.string(),
  args: z.record(z.unknown()),
  summary: z.string(),
  reason: z.string().nullable(),
  expiresAt: z.string(),
});

/** 202 do `/chat`: a ação aguarda aprovação humana e nada foi executado. */
export const ChatAwaitingApprovalSchema = z.object({
  requestId: z.string(),
  status: z.literal("awaiting_approval"),
  approval: ApprovalSchema,
  trace: z.array(TraceEventSchema),
  route: RouteDecisionSchema,
  conversationId: z.string(),
  metrics: MetricsSchema,
});

/** 200 de `POST /approvals/:id`: mesmo formato do 200 do `/chat`, sem rota nem métricas. */
export const DecisionOkSchema = z.object({
  requestId: z.string(),
  answer: z.string(),
  trace: z.array(TraceEventSchema),
  route: z.null(),
  metrics: z.null(),
  conversationId: z.string(),
  approval: z.object({ id: z.string(), status: z.enum(["approved", "denied"]) }),
});

export const ApiErrorSchema = z
  .object({
    requestId: z.string().optional(),
    error: z.string(),
  })
  .passthrough();

export type RouteDecision = z.infer<typeof RouteDecisionSchema>;
export type Metrics = z.infer<typeof MetricsSchema>;
export type TraceEvent = z.infer<typeof TraceEventSchema>;
export type KnownTraceEvent = z.infer<typeof KnownTraceEventSchema>;
export type ChatOk = z.infer<typeof ChatOkSchema>;
export type Approval = z.infer<typeof ApprovalSchema>;
export type ChatAwaitingApproval = z.infer<typeof ChatAwaitingApprovalSchema>;
export type DecisionOk = z.infer<typeof DecisionOkSchema>;
export type ApiError = z.infer<typeof ApiErrorSchema>;
