import type { HandoffTarget } from "../agents/types.ts";

/**
 * Quadro compartilhado da equipe (spec 017, data-model.md). Imutável: toda escrita devolve uma
 * cópia. Cada campo tem um único escritor — fatos (analista), plano (planejador), resultados
 * (executor), passagens (supervisor) — e é o grafo da equipe quem chama cada função.
 */

export type AnalystSource = "list_alerts" | "list_incidents" | "consultar_runbook" | "pedido";

export interface Fact {
  statement: string;
  source: AnalystSource;
}

export interface Outcome {
  summary: string;
  /** O executor propôs uma ação, que aguarda aprovação humana. */
  proposal: boolean;
}

export interface Handoff {
  from: "supervisor";
  to: HandoffTarget;
  brief: string;
}

export interface Blackboard {
  request: string;
  facts: Fact[];
  plan: string[] | null;
  outcomes: Outcome[];
  handoffs: Handoff[];
}

const MAX_RENDERED_FACTS = 20;
const MAX_ITEM_CHARS = 500;

export function createBlackboard(request: string): Blackboard {
  return { request, facts: [], plan: null, outcomes: [], handoffs: [] };
}

export function addFacts(board: Blackboard, facts: readonly Fact[]): Blackboard {
  const seen = new Set(board.facts.map((fact) => `${fact.source}\u0000${fact.statement}`));
  const fresh: Fact[] = [];
  for (const fact of facts) {
    const key = `${fact.source}\u0000${fact.statement}`;
    if (!seen.has(key)) {
      seen.add(key);
      fresh.push({ statement: fact.statement, source: fact.source });
    }
  }
  return { ...board, facts: [...board.facts, ...fresh] };
}

export function setPlan(board: Blackboard, steps: readonly string[]): Blackboard {
  return { ...board, plan: [...steps] };
}

export function addOutcome(board: Blackboard, outcome: Outcome): Blackboard {
  return { ...board, outcomes: [...board.outcomes, { ...outcome }] };
}

export function addHandoff(board: Blackboard, handoff: Handoff): Blackboard {
  return { ...board, handoffs: [...board.handoffs, { ...handoff }] };
}

const clip = (text: string) => (text.length > MAX_ITEM_CHARS ? `${text.slice(0, MAX_ITEM_CHARS)}…` : text);

/** O quadro como texto para os prompts do supervisor e dos papéis. Pura, com tamanho limitado. */
export function renderBlackboard(board: Blackboard): string {
  const facts = board.facts.slice(-MAX_RENDERED_FACTS);
  const lines = [
    "## Pedido do plantonista",
    clip(board.request),
    "",
    "## Fatos (analista)",
    ...(facts.length > 0 ? facts.map((fact) => `- ${clip(fact.statement)} [${fact.source}]`) : ["(nenhum ainda)"]),
    "",
    "## Plano (planejador)",
    ...(board.plan ? board.plan.map((step, index) => `${index + 1}. ${clip(step)}`) : ["(nenhum ainda)"]),
    "",
    "## Resultados (executor)",
    ...(board.outcomes.length > 0
      ? board.outcomes.map((outcome) => `- ${outcome.proposal ? "[aguardando aprovação] " : ""}${clip(outcome.summary)}`)
      : ["(nenhum ainda)"]),
    "",
    "## Passagens já feitas",
    ...(board.handoffs.length > 0
      ? board.handoffs.map((handoff, index) => `${index + 1}. ${handoff.from} → ${handoff.to}: ${clip(handoff.brief)}`)
      : ["(nenhuma ainda)"]),
  ];
  return lines.join("\n");
}

/** Resposta quando a equipe encerra sem decisão final do supervisor (teto, decisão inválida). */
export function fallbackAnswer(board: Blackboard, reason: string): string {
  const parts = [`A equipe encerrou antes de concluir (${reason}).`];
  if (board.facts.length > 0) {
    parts.push(["Fatos levantados:", ...board.facts.map((fact) => `- ${fact.statement}`)].join("\n"));
  }
  if (board.plan) {
    parts.push(["Plano:", ...board.plan.map((step, index) => `${index + 1}. ${step}`)].join("\n"));
  }
  if (board.outcomes.length > 0) {
    parts.push(["Resultados:", ...board.outcomes.map((outcome) => `- ${outcome.summary}`)].join("\n"));
  }
  if (parts.length === 1) {
    parts.push("Nenhum fato foi levantado ainda; tente reformular o pedido.");
  }
  return parts.join("\n\n");
}
