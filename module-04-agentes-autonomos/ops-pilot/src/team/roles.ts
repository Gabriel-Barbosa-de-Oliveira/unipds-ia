import type { StructuredToolInterface } from "@langchain/core/tools";
import { z } from "zod";

import type { TeamRole, ToolName } from "../agents/types.ts";

/**
 * Ferramentas de cada papel (spec 017, data-model.md) — a lista é a garantia: o papel recebe
 * exatamente estas, nem mais nem menos.
 */
export const TEAM_ROLE_TOOLS: Readonly<Record<TeamRole, readonly ToolName[]>> = {
  analista: ["list_alerts", "list_incidents", "consultar_runbook"],
  planejador: [],
  executor: ["open_incident", "resolve_incident"],
};

/** Seleciona as ferramentas do papel; falta de alguma é erro de composição. Pura. */
export function selectRoleTools(all: readonly StructuredToolInterface[], role: TeamRole): StructuredToolInterface[] {
  return TEAM_ROLE_TOOLS[role].map((name) => {
    const tool = all.find((candidate) => candidate.name === name);
    if (!tool) {
      throw new Error(`Ferramenta "${name}" ausente para o papel ${role}`);
    }
    return tool;
  });
}

/**
 * Relatório do analista: só fatos com origem. Não existe campo para recomendação — chaves extras
 * são descartadas pelo parse (`strip`), então nada além de fatos chega ao quadro (FR-007).
 */
export const AnalystReportSchema = z.object({
  facts: z
    .array(
      z.object({
        statement: z.string().trim().min(1).describe("Um fato observado, em tópico telegráfico"),
        source: z
          .enum(["list_alerts", "list_incidents", "consultar_runbook", "pedido"])
          .describe("De onde o fato veio: a ferramenta consultada ou o próprio pedido"),
      }),
    )
    .max(20),
});

export const PlanSchema = z.object({
  steps: z.array(z.string().trim().min(1)).min(1).max(8).describe("Passos ordenados do plano"),
});

export const ANALYST_PROMPT = [
  "Você é o ANALISTA do plantão. Sua única função: produzir um diagnóstico FACTUAL do estado atual com",
  "as ferramentas de leitura. Liste: alertas disparando (com data e severidade), incidentes recentes",
  "(abertos e resolvidos), runbooks relevantes, status dos provedores, janelas e fatos conhecidos do time.",
  "NÃO proponha soluções, NÃO abra nem resolva nada.",
  "Formato: tópicos telegráficos. Seja cético: se um dado não está nas observações, não afirme.",
].join("\n");

export const ANALYST_EXTRACTION_PROMPT = [
  "Extraia do diagnóstico abaixo SOMENTE os fatos observados, um por item, com a origem de cada um.",
  "Descarte opiniões, recomendações e propostas. Se nada foi observado, devolva a lista vazia.",
].join("\n");

export const PLANNER_PROMPT = [
  "Você é o PLANEJADOR do plantão. Sem ferramentas: a partir SOMENTE dos fatos do quadro e da instrução",
  "do supervisor, escreva um plano curto de passos ordenados (no máximo 8). Ações sobre incidentes serão",
  "feitas pelo executor, com aprovação humana. Não invente fatos.",
].join("\n");

export const EXECUTOR_PROMPT = [
  "Você é o EXECUTOR do plantão. Execute SOMENTE o que a instrução do supervisor e o plano do quadro pedem,",
  "usando as ferramentas de incidente. Toda ação passa por aprovação humana: quando a ferramenta responder",
  "que a ação aguarda aprovação, NÃO tente de novo e informe isso. Não consulte nem invente dados.",
].join("\n");
