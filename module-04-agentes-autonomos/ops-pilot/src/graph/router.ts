import { z } from "zod";

import { createModel, loadModelConfig } from "../agents/model.ts";
import { ModelUsageTracker, summarizeModelUsage } from "../agents/model-usage.ts";
import { ROUTE_NAMES, type ModelFallback, type RouteDecision, type RouteName } from "../agents/types.ts";
import { UsageCollector, type TokenUsage } from "../context/tokens.ts";
import { UnknownStrategyError } from "../domain/errors.ts";

/** Rota usada quando o roteador falha ou devolve uma decisão inválida (FR-011). */
export const FALLBACK_ROUTE: RouteName = "react";

export const OVERRIDE_REASON = "Estratégia informada pelo cliente";

/** Tabela de rotas embutida no prompt do roteador (FR-004, research.md item 3). */
export const ROUTE_TABLE: readonly { route: RouteName; whenToUse: string; examples: readonly string[] }[] = [
  {
    route: "react",
    whenToUse: "Consulta direta ou ação única sobre alertas/incidentes.",
    examples: ["quais alertas estão firing?", "abra um incidente para o checkout", "resolva o incidente INC-3"],
  },
  {
    route: "planExecute",
    whenToUse: "Pedido com várias etapas dependentes ou em lote.",
    examples: [
      "triar todos os alertas críticos e abrir um incidente para cada",
      "resolver tudo que estiver aberto do serviço payments",
    ],
  },
  {
    route: "reflect",
    whenToUse: "Precisão crítica: a resposta precisa ser revisada contra os fatos antes de sair.",
    examples: [
      "escreva o resumo do incidente INC-3 para o pós-mortem",
      "confirme, cruzando os dados, se o alerta de latência já foi tratado",
    ],
  },
];

export const routeSchema = z.object({
  route: z.enum(ROUTE_NAMES),
  reason: z.string().trim().min(1).describe("Uma frase justificando a escolha"),
});

/** Nomes aceitos no `strategy` do /chat; inclui os nomes legados das estratégias (SC-007). */
const ROUTE_ALIASES: Readonly<Record<string, RouteName>> = {
  react: "react",
  planExecute: "planExecute",
  "plan-and-execute": "planExecute",
  reflect: "reflect",
  reflection: "reflect",
};

/** Converte o `strategy` do /chat numa rota; lança `UnknownStrategyError` para nomes desconhecidos (FR-010). */
export function parseRouteName(name: string): RouteName {
  const route = Object.hasOwn(ROUTE_ALIASES, name) ? ROUTE_ALIASES[name] : undefined;
  if (!route) {
    throw new UnknownStrategyError(name);
  }
  return route;
}

function routeTableMarkdown(): string {
  const rows = ROUTE_TABLE.map(
    (entry) => `| ${entry.route} | ${entry.whenToUse} | ${entry.examples.map((example) => `"${example}"`).join("; ")} |`,
  );
  return ["| Rota | Quando usar | Exemplos |", "|---|---|---|", ...rows].join("\n");
}

export const SYSTEM_PROMPT = [
  "Você é o roteador do OpsPilot, um copiloto de plantão. Leia o pedido do plantonista e escolha",
  "exatamente UMA rota de raciocínio da tabela abaixo. Explique o motivo em uma frase.",
  "",
  routeTableMarkdown(),
  "",
  "Na dúvida entre react e outra rota, prefira react (mais barata).",
].join("\n");

/** Monta as mensagens (system + user) enviadas ao roteador. Função pura, determinística. */
export function buildRouterMessages(prompt: string): [string, string][] {
  return [
    ["system", SYSTEM_PROMPT],
    ["user", prompt],
  ];
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Aplica, nesta ordem: override do cliente > decisão válida do modelo > fallback para `react`.
 * Pura — a saída crua do modelo é validada aqui com `routeSchema` (FR-008, FR-009, FR-011).
 */
export function resolveRouteDecision(input: {
  override?: RouteName;
  decided?: unknown;
  error?: unknown;
}): RouteDecision {
  if (input.override) {
    return { route: input.override, reason: OVERRIDE_REASON, source: "override" };
  }

  if (input.error !== undefined) {
    return { route: FALLBACK_ROUTE, reason: `Fallback: roteador falhou (${describeError(input.error)})`, source: "fallback" };
  }

  const parsed = routeSchema.safeParse(input.decided);
  if (!parsed.success) {
    return { route: FALLBACK_ROUTE, reason: "Fallback: decisão do roteador inválida", source: "fallback" };
  }

  return { route: parsed.data.route, reason: parsed.data.reason, source: "router" };
}

export type DecideRoute = (
  prompt: string,
) => Promise<{ decided: unknown; tokenUsage: TokenUsage; fallbacks: ModelFallback[] }>;

/**
 * Roteador real sobre o modelo resiliente (013) — única IO deste módulo; o modelo só é criado na
 * chamada. Devolve também as trocas principal → reserva ocorridas, para o trace do grafo.
 */
export function createModelRouter(): DecideRoute {
  return async (prompt) => {
    const usageCollector = new UsageCollector();
    const modelTracker = new ModelUsageTracker();
    const modelConfig = loadModelConfig(process.env);
    const decided = await createModel(
      (model) => model.withStructuredOutput<z.infer<typeof routeSchema>>(routeSchema),
      modelConfig,
    ).invoke(buildRouterMessages(prompt), { callbacks: [usageCollector, modelTracker] });
    const { fallbacks } = summarizeModelUsage(modelTracker.log, modelConfig);
    return { decided, tokenUsage: usageCollector.tokenUsage, fallbacks };
  };
}
