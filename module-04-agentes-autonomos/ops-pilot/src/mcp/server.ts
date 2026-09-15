import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

import { listAlertsShape, openIncidentShape, resolveIncidentShape, toStructuredError } from "../agents/tools.ts";
import type { OpsStoreRepository } from "../services/ops-store.repository.ts";
import { SqliteOpsStore } from "../store/sqlite-ops-store.ts";

/**
 * Monta o servidor MCP `opspilot`, registrando as tools sobre `store`. Fábrica pura (nenhuma IO
 * própria) — o entrypoint abaixo decide qual `OpsStoreRepository` compor, nos mesmos moldes de
 * `createOpsTools` (`src/agents/tools.ts`).
 */
export function createMcpServer(store: OpsStoreRepository): McpServer {
  const server = new McpServer({ name: "opspilot", version: "0.1.0" });

  server.registerTool(
    "list_alerts",
    {
      description:
        "Consulta (somente leitura) os alertas de monitoramento. Use quando o plantonista perguntar o " +
        "que está disparando, o estado dos serviços, ou pedir a lista de alertas. Não abre nem resolve " +
        "nada.",
      inputSchema: listAlertsShape,
    },
    async ({ status }) => ({
      content: [{ type: "text", text: JSON.stringify(await store.listAlerts(status)) }],
    }),
  );

  server.registerTool(
    "open_incident",
    {
      description:
        "Cria um incidente NOVO para um serviço. Use somente quando o plantonista pedir explicitamente " +
        "para abrir/registrar um incidente — nunca para consultar incidentes que já existem.",
      inputSchema: openIncidentShape,
    },
    async ({ title, service, severity }) => {
      try {
        const incident = await store.openIncident({ title, service, severity });
        return { content: [{ type: "text", text: JSON.stringify(incident) }] };
      } catch (error) {
        return {
          content: [{ type: "text", text: JSON.stringify(toStructuredError(error, { service })) }],
          isError: true,
        };
      }
    },
  );

  server.registerTool(
    "resolve_incident",
    {
      description:
        "Resolve um incidente já existente pelo id. Use quando o plantonista pedir para fechar/resolver " +
        "um incidente que já foi aberto.",
      inputSchema: resolveIncidentShape,
    },
    async ({ id, summary }) => {
      try {
        const incident = await store.resolveIncident(id, summary);
        return { content: [{ type: "text", text: JSON.stringify(incident) }] };
      } catch (error) {
        return {
          content: [{ type: "text", text: JSON.stringify(toStructuredError(error, { id })) }],
          isError: true,
        };
      }
    },
  );

  return server;
}

const server = createMcpServer(new SqliteOpsStore());
await server.connect(new StdioServerTransport());
// stderr, nunca stdout — no transporte stdio, o stdout é o canal do protocolo MCP.
console.error("OpsPilot MCP server pronto (stdio)");
