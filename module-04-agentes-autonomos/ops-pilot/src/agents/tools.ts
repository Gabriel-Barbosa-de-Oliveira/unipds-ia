import { tool } from "@langchain/core/tools";
import { z } from "zod";

import { IncidentNotFoundError, InvalidSeverityError, ServiceNotFoundError } from "../domain/errors.ts";
import { store } from "../services/ops-store.memory.ts";

function toStructuredError(error: unknown, extra: Record<string, unknown>): Record<string, unknown> {
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

export const listAlertsTool = tool(
  async ({ status }: { status?: "firing" | "resolved" }) => {
    const alerts = await store.listAlerts(status);
    return JSON.stringify(alerts);
  },
  {
    name: "list_alerts",
    description:
      "Lista os alertas de monitoramento. Use quando o plantonista perguntar o que está disparando, " +
      "o estado dos serviços, ou pedir a lista de alertas. status: firing | resolved (omita para listar todos).",
    schema: z.object({
      status: z.enum(["firing", "resolved"]).optional(),
    }),
  },
);

export const openIncidentTool = tool(
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
      "Abre um novo incidente para um serviço. Use quando o plantonista pedir para abrir/registrar " +
      "um incidente. severity: low | medium | high | critical.",
    schema: z.object({
      title: z.string().min(1),
      service: z.string().min(1),
      severity: z.enum(["low", "medium", "high", "critical"]),
    }),
  },
);

export const resolveIncidentTool = tool(
  async ({ id }: { id: string }) => {
    try {
      const incident = await store.resolveIncident(id);
      return JSON.stringify(incident);
    } catch (error) {
      return JSON.stringify(toStructuredError(error, { id }));
    }
  },
  {
    name: "resolve_incident",
    description:
      "Resolve um incidente existente pelo id. Use quando o plantonista pedir para fechar/resolver um incidente.",
    schema: z.object({
      id: z.string().min(1),
    }),
  },
);

/** Conjunto fixo de tools operacionais usado por todas as estratégias de raciocínio. */
export const opsTools = [listAlertsTool, openIncidentTool, resolveIncidentTool];
