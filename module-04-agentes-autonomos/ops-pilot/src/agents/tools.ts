import { tool, type StructuredToolInterface } from "@langchain/core/tools";
import { z } from "zod";

import { IncidentNotFoundError, InvalidSeverityError, ServiceNotFoundError } from "../domain/errors.ts";
import type { OpsStoreRepository } from "../services/ops-store.repository.ts";
import { SqliteOpsStore } from "../store/sqlite-ops-store.ts";

/**
 * Shapes zod reaproveitados tanto pelas tools do LangChain abaixo quanto pelo servidor MCP
 * (`src/mcp/server.ts`) — única fonte de verdade para os schemas de `list_alerts`,
 * `open_incident` e `resolve_incident` (ver specs/005-mcp-server/research.md item 2).
 */
export const listAlertsShape = {
  status: z
    .enum(["firing", "resolved"])
    .optional()
    .describe("Filtra por status do alerta; omita para listar todos os alertas."),
};

export const openIncidentShape = {
  title: z.string().min(1).describe("Título curto descrevendo o problema do incidente."),
  service: z.string().min(1).describe("Nome do serviço afetado, ex.: \"checkout-api\"."),
  severity: z
    .enum(["low", "medium", "high", "critical"])
    .describe("Severidade do incidente, do menos ao mais grave."),
};

export const resolveIncidentShape = {
  id: z.string().min(1).describe("Id do incidente a resolver, retornado por open_incident ou list_incidents."),
  summary: z
    .string()
    .min(1)
    .optional()
    .describe("Resumo opcional do que foi feito para resolver o incidente, para consulta futura."),
};

export function toStructuredError(error: unknown, extra: Record<string, unknown>): Record<string, unknown> {
  if (error instanceof ServiceNotFoundError) {
    return { error: "ServiceNotFoundError", ...extra };
  }
  if (error instanceof IncidentNotFoundError) {
    return { error: "IncidentNotFoundError", ...extra };
  }
  if (error instanceof InvalidSeverityError) {
    return { error: "InvalidSeverityError", ...extra };
  }
  // Falha de infraestrutura não recuperável: não é um erro de domínio esperado, não deve
  // virar uma observação silenciosa — propaga para interromper o `run` (per contrato).
  throw error;
}

/**
 * Constrói as tools operacionais fechadas sobre `store`. Fábrica pura (nenhuma IO própria) —
 * quem decide qual `OpsStoreRepository` compor é o chamador (`opsTools` abaixo para o caminho
 * padrão de produção; `src/bench.ts` monta a sua própria composição sobre um mock em memória
 * isolado, ver specs/004-ops-persistence/research.md item 2).
 */
export function createOpsTools(store: OpsStoreRepository): StructuredToolInterface[] {
  const listAlertsTool = tool(
    async ({ status }: { status?: "firing" | "resolved" }) => {
      const alerts = await store.listAlerts(status);
      return JSON.stringify(alerts);
    },
    {
      name: "list_alerts",
      description:
        "Consulta (somente leitura) os alertas de monitoramento. Use quando o plantonista perguntar o " +
        "que está disparando, o estado dos serviços, ou pedir a lista de alertas. Não abre nem resolve " +
        "nada — para incidentes já abertos/resolvidos pelo próprio copiloto, use list_incidents.",
      schema: z.object(listAlertsShape),
    },
  );

  const openIncidentTool = tool(
    async ({
      title,
      service,
      severity,
    }: {
      title: string;
      service: string;
      severity: "low" | "medium" | "high" | "critical";
    }) => {
      try {
        const incident = await store.openIncident({ title, service, severity });
        return JSON.stringify(incident);
      } catch (error) {
        return JSON.stringify(toStructuredError(error, { service }));
      }
    },
    {
      name: "open_incident",
      description:
        "Cria um incidente NOVO para um serviço. Use somente quando o plantonista pedir explicitamente " +
        "para abrir/registrar um incidente — nunca para consultar incidentes que já existem (isso é " +
        "list_incidents) nem para saber os passos de mitigação de um serviço (isso é consultar_runbook).",
      schema: z.object(openIncidentShape),
    },
  );

  const resolveIncidentTool = tool(
    async ({ id, summary }: { id: string; summary?: string }) => {
      try {
        const incident = await store.resolveIncident(id, summary);
        return JSON.stringify(incident);
      } catch (error) {
        return JSON.stringify(toStructuredError(error, { id }));
      }
    },
    {
      name: "resolve_incident",
      description:
        "Resolve um incidente já existente pelo id. Use quando o plantonista pedir para fechar/resolver " +
        "um incidente que já foi aberto — para descobrir esse id primeiro, use list_incidents.",
      schema: z.object(resolveIncidentShape),
    },
  );

  const listIncidentsTool = tool(
    async ({ status }: { status?: "open" | "resolved" | "all" }) => {
      const incidents = await store.listIncidents(status);
      return JSON.stringify(incidents);
    },
    {
      name: "list_incidents",
      description:
        "Consulta (somente leitura) os incidentes já geridos pelo próprio copiloto — o que está aberto " +
        "agora, o que já foi resolvido, ou o histórico completo. Use para responder \"o que está aberto\" " +
        "ou achar o id de um incidente para resolve_incident. Nunca use para abrir um incidente novo " +
        "(isso é open_incident).",
      schema: z.object({
        status: z
          .enum(["open", "resolved", "all"])
          .optional()
          .describe("Filtra por status do incidente; omita ou use \"all\" para listar todos."),
      }),
    },
  );

  const consultarRunbookTool = tool(
    async ({ service }: { service: string }) => {
      try {
        const runbook = await store.getRunbook(service);
        return JSON.stringify({ service, runbook: runbook?.content ?? null });
      } catch (error) {
        return JSON.stringify(toStructuredError(error, { service }));
      }
    },
    {
      name: "consultar_runbook",
      description:
        "Consulta os passos de mitigação recomendados (runbook) para um serviço. Use durante um " +
        "alerta/incidente para saber como agir. Não abre, resolve nem lista incidentes — se o serviço " +
        "não tiver runbook cadastrado, retorna runbook: null (não é um erro).",
      schema: z.object({
        service: z
          .string()
          .min(1)
          .describe("Nome do serviço (o mesmo usado em open_incident), ex.: \"checkout-api\"."),
      }),
    },
  );

  return [listAlertsTool, openIncidentTool, resolveIncidentTool, listIncidentsTool, consultarRunbookTool];
}

/**
 * Composição padrão usada por `react.ts`/`plan-and-execute.ts` — o único ponto onde o adaptador
 * de produção (`SqliteOpsStore`, lido de `OPSPILOT_DB`) é instanciado para as tools do copiloto.
 */
export const opsTools = createOpsTools(new SqliteOpsStore());
