import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import type { DatabaseSync } from "node:sqlite";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import { seedCanonicalScenario, SqliteOpsStore } from "../store/sqlite-ops-store.ts";

const REPO_ROOT = path.resolve(import.meta.dirname, "../..");

function inheritedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined) {
      env[key] = value;
    }
  }
  return env;
}

/**
 * Sobe o processo REAL do servidor MCP (`npm run mcp` equivalente) contra um SQLite temporário
 * já semeado com o cenário canônico, e conecta um `Client` MCP via stdio — exercita de ponta a
 * ponta o binário real, o script npm e o canal stdio (ver specs/005-mcp-server/research.md §5).
 */
async function connectToSeededServer(): Promise<{ client: Client; cleanup: () => Promise<void> }> {
  const dir = await mkdtemp(path.join(tmpdir(), "opspilot-mcp-"));
  const dbPath = path.join(dir, "opspilot.db");

  const store = new SqliteOpsStore(dbPath);
  seedCanonicalScenario(store);
  (store as unknown as { db: DatabaseSync }).db.close();

  const transport = new StdioClientTransport({
    command: process.execPath,
    args: ["--import", "tsx", "src/mcp/server.ts"],
    cwd: REPO_ROOT,
    env: { ...inheritedEnv(), OPSPILOT_DB: dbPath },
  });

  const client = new Client({ name: "opspilot-test-client", version: "1.0.0" });
  await client.connect(transport);

  return {
    client,
    async cleanup() {
      await client.close();
      await rm(dir, { recursive: true, force: true });
    },
  };
}

type CallToolResult = Awaited<ReturnType<Client["callTool"]>>;

function firstText(result: CallToolResult): string {
  const content = (result as { content?: Array<{ type: string; text?: string }> }).content;
  assert.ok(Array.isArray(content), "esperava um CallToolResult com content (não um resultado de task)");
  const first = content[0];
  assert.ok(first && first.type === "text" && typeof first.text === "string", "esperava content[0] de texto");
  return first.text as string;
}

// User Story 1 — Consultar alertas via cliente MCP

test("tools/list inclui list_alerts com status opcional (firing|resolved)", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const { tools } = await client.listTools();
    const listAlerts = tools.find((t) => t.name === "list_alerts");

    assert.ok(listAlerts, "list_alerts deveria estar em tools/list");
    assert.deepEqual(listAlerts?.inputSchema.properties?.status, {
      type: "string",
      enum: ["firing", "resolved"],
      description: "Filtra por status do alerta; omita para listar todos os alertas.",
    });
  } finally {
    await cleanup();
  }
});

test("list_alerts sem argumentos retorna todos os alertas semeados; com status filtra", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const all = JSON.parse(firstText(await client.callTool({ name: "list_alerts", arguments: {} })));
    const firing = JSON.parse(
      firstText(await client.callTool({ name: "list_alerts", arguments: { status: "firing" } })),
    );

    assert.equal(all.length, 6);
    assert.equal(firing.length, 3);
    assert.ok(firing.every((alert: { status: string }) => alert.status === "firing"));
  } finally {
    await cleanup();
  }
});

// User Story 2 — Abrir um incidente via cliente MCP

test("tools/list inclui open_incident com title/service/severity obrigatórios", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const { tools } = await client.listTools();
    const openIncident = tools.find((t) => t.name === "open_incident");

    assert.ok(openIncident, "open_incident deveria estar em tools/list");
    assert.deepEqual(openIncident?.inputSchema.required?.sort(), ["service", "severity", "title"]);
  } finally {
    await cleanup();
  }
});

test("open_incident cria um incidente open para um serviço existente", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const result = await client.callTool({
      name: "open_incident",
      arguments: { title: "Checkout fora do ar", service: "checkout-api", severity: "high" },
    });
    const incident = JSON.parse(firstText(result));

    assert.equal(result.isError, undefined);
    assert.equal(incident.status, "open");
    assert.equal(incident.resolvedAt, null);
  } finally {
    await cleanup();
  }
});

test("open_incident retorna ServiceNotFoundError estruturado para serviço inexistente", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const result = await client.callTool({
      name: "open_incident",
      arguments: { title: "x", service: "servico-fantasma", severity: "high" },
    });
    const error = JSON.parse(firstText(result));

    assert.equal(result.isError, true);
    assert.equal(error.error, "ServiceNotFoundError");
    assert.equal(error.service, "servico-fantasma");
  } finally {
    await cleanup();
  }
});

// User Story 3 — Resolver um incidente via cliente MCP

test("tools/list inclui resolve_incident com id obrigatório e summary opcional", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const { tools } = await client.listTools();
    const resolveIncident = tools.find((t) => t.name === "resolve_incident");

    assert.ok(resolveIncident, "resolve_incident deveria estar em tools/list");
    assert.deepEqual(resolveIncident?.inputSchema.required, ["id"]);
    assert.ok(resolveIncident?.inputSchema.properties?.summary, "summary deveria ser uma propriedade opcional");
  } finally {
    await cleanup();
  }
});

test("resolve_incident resolve um incidente aberto, com summary opcional", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const opened = JSON.parse(
      firstText(
        await client.callTool({
          name: "open_incident",
          arguments: { title: "x", service: "payments-api", severity: "critical" },
        }),
      ),
    );

    const resolved = JSON.parse(
      firstText(
        await client.callTool({
          name: "resolve_incident",
          arguments: { id: opened.id, summary: "mitigado via rollback" },
        }),
      ),
    );

    assert.equal(resolved.status, "resolved");
    assert.equal(resolved.summary, "mitigado via rollback");
  } finally {
    await cleanup();
  }
});

test("resolve_incident retorna IncidentNotFoundError estruturado para id inexistente", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const result = await client.callTool({ name: "resolve_incident", arguments: { id: "id-que-nao-existe" } });
    const error = JSON.parse(firstText(result));

    assert.equal(result.isError, true);
    assert.equal(error.error, "IncidentNotFoundError");
  } finally {
    await cleanup();
  }
});

test("tools/list expõe exatamente as 3 tools do servidor opspilot (FR-012/SC-004)", async () => {
  const { client, cleanup } = await connectToSeededServer();
  try {
    const { tools } = await client.listTools();

    assert.deepEqual(
      tools.map((t) => t.name).sort(),
      ["list_alerts", "open_incident", "resolve_incident"],
    );
  } finally {
    await cleanup();
  }
});
