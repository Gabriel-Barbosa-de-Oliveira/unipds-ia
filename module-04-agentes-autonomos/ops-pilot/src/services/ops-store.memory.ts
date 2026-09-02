import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { z } from "zod";

import type {
  Alert,
  AlertStatus,
  Incident,
  IncidentSeverity,
  OpenIncidentInput,
  OpsState,
} from "../domain/ops-store.ts";
import { INCIDENT_SEVERITIES, listAlerts, openIncident, resolveIncident } from "../domain/ops-store.ts";
import { buildSeedState } from "../domain/seed-data.ts";
import type { OpsStoreRepository } from "./ops-store.repository.ts";

const PROJECT_ROOT = fileURLToPath(new URL("../../", import.meta.url));

/**
 * Caminho do arquivo JSON usado como base de dados enquanto a feature não depende de um MySQL
 * real (ver `npm run seed`). Ponto único de configuração — trocar por uma URL de conexão real
 * mais tarde não deve exigir mudanças no domínio nem nas tools.
 */
export const DEFAULT_DATA_FILE = join(PROJECT_ROOT, "data", "ops-store.json");

const OpsStateSchema = z.object({
  services: z.array(z.object({ id: z.string(), name: z.string() })),
  alerts: z.array(
    z.object({
      id: z.string(),
      serviceId: z.string(),
      title: z.string(),
      status: z.enum(["firing", "resolved"]),
      createdAt: z.string(),
    }),
  ),
  incidents: z.array(
    z.object({
      id: z.string(),
      title: z.string(),
      serviceId: z.string(),
      severity: z.enum(INCIDENT_SEVERITIES as [IncidentSeverity, ...IncidentSeverity[]]),
      status: z.enum(["open", "resolved"]),
      createdAt: z.string(),
      resolvedAt: z.string().nullable(),
    }),
  ),
});

/** Lê o estado persistido do arquivo JSON, ou parte do dataset canônico se ele ainda não existir. */
export function readOpsStateFile(filePath: string): OpsState {
  if (!existsSync(filePath)) {
    return buildSeedState();
  }
  return OpsStateSchema.parse(JSON.parse(readFileSync(filePath, "utf-8")));
}

/** Grava o estado no arquivo JSON, criando o diretório pai se necessário. */
export function writeOpsStateFile(filePath: string, state: OpsState): void {
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(state, null, 2)}\n`, "utf-8");
}

/**
 * Adaptador do OpsStoreRepository apoiado em um arquivo JSON local (`data/ops-store.json`),
 * usado no lugar do MySQL enquanto os testes/validação manual desta feature não dependem de um
 * banco real: lê o estado existente do arquivo na inicialização (ou o dataset canônico, se o
 * arquivo ainda não existir) e persiste cada mutação de volta, para que execuções separadas da
 * arena (`npm run arena`) enxerguem os incidentes criados por execuções anteriores. É o
 * adaptador padrão usado por `src/agents/tools.ts` e pela arena.
 */
export class InMemoryOpsStore implements OpsStoreRepository {
  private state: OpsState;

  constructor(private readonly filePath: string = DEFAULT_DATA_FILE) {
    this.state = readOpsStateFile(filePath);
  }

  async listAlerts(status?: AlertStatus): Promise<Alert[]> {
    return listAlerts(this.state, status);
  }

  async openIncident(input: OpenIncidentInput): Promise<Incident> {
    const { state, incident } = openIncident(this.state, input, {
      id: crypto.randomUUID(),
      now: new Date().toISOString(),
    });
    this.state = state;
    this.persist();
    return incident;
  }

  async resolveIncident(id: string): Promise<Incident> {
    const { state, incident } = resolveIncident(this.state, id, {
      now: new Date().toISOString(),
    });
    this.state = state;
    this.persist();
    return incident;
  }

  /** Restaura o store para o dataset semeado, descartando incidentes criados. */
  reset(): void {
    this.state = buildSeedState();
    this.persist();
  }

  private persist(): void {
    writeOpsStateFile(this.filePath, this.state);
  }
}

/** Instância padrão compartilhada, usada por `src/agents/tools.ts` e por `src/arena.ts`. */
export const store: OpsStoreRepository = new InMemoryOpsStore();
