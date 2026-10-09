import { z } from "zod";

import type { TeamRole } from "../agents/types.ts";

/** Teto de turnos de papel por execução (spec 017, FR-005). */
export const TEAM_MAX_TURNS = 6;

export const SUPERVISOR_NEXT = ["analista", "planejador", "executor", "done"] as const;

/** Decisão estruturada do supervisor: próximo papel (ou `done`) e a instrução — ou a resposta final. */
export const nextSchema = z.object({
  next: z.enum(SUPERVISOR_NEXT),
  brief: z
    .string()
    .trim()
    .min(1)
    .describe("Instrução de trabalho para o próximo papel (nó) ou resposta final ao plantonista se done"),
});

export type SupervisorDecision = z.infer<typeof nextSchema>;

export type SupervisorOutcome =
  | { kind: "route"; to: TeamRole; brief: string }
  | { kind: "done"; answer: string }
  | { kind: "abort"; reason: string };

export const SUPERVISOR_PROMPT = [
  "Você é o SUPERVISOR de uma equipe de plantão do OpsPilot. A cada passo, leia o quadro e escolha",
  "quem trabalha a seguir, com uma instrução curta e específica (brief).",
  "",
  "Papéis disponíveis:",
  "- analista: levanta FATOS com ferramentas de leitura (alertas, incidentes, runbooks). Não propõe nada.",
  "- planejador: transforma os fatos num plano de passos. Não tem ferramentas.",
  "- executor: abre ou resolve incidentes, sempre com aprovação humana. Não consulta nada.",
  "- done: encerra. O brief passa a ser a resposta final ao plantonista, em português.",
  "",
  "Critérios de sequência:",
  "1. Comece pelo analista, a menos que o quadro já tenha os fatos necessários.",
  "2. Só chame o planejador quando houver fatos; só chame o executor quando houver plano com ação.",
  "3. Pedido só de consulta: depois do analista, encerre com done.",
  "4. A resposta final (done) usa SOMENTE o que está no quadro; não invente fatos.",
  `5. Você tem no máximo ${TEAM_MAX_TURNS} passagens para papéis; não repita um papel sem motivo novo.`,
].join("\n");

/** Mensagens do supervisor: prompt fixo + o quadro como texto (o estado da equipe). Pura. */
export function buildSupervisorMessages(blackboardText: string, turn: number, cap = TEAM_MAX_TURNS): [string, string][] {
  return [
    ["system", SUPERVISOR_PROMPT],
    ["user", `${blackboardText}\n\nPassagens usadas: ${turn} de ${cap}.`],
  ];
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Valida a decisão crua do modelo e aplica o teto. Pura: decisão inválida ou falha do modelo viram
 * `abort` (encerramento controlado), nunca exceção.
 */
export function resolveSupervisorDecision(
  input: { decided?: unknown; error?: unknown },
  turn: number,
  cap = TEAM_MAX_TURNS,
): SupervisorOutcome {
  if (input.error !== undefined) {
    return { kind: "abort", reason: `supervisor falhou (${describeError(input.error)})` };
  }

  const parsed = nextSchema.safeParse(input.decided);
  if (!parsed.success) {
    return { kind: "abort", reason: "decisão inválida do supervisor" };
  }

  const { next, brief } = parsed.data;
  if (next === "done") {
    return { kind: "done", answer: brief };
  }
  if (turn >= cap) {
    return { kind: "abort", reason: `teto de ${cap} passagens atingido` };
  }
  return { kind: "route", to: next, brief };
}
