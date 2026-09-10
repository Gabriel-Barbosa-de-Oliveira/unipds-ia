import type {
  Alert,
  AlertStatus,
  Incident,
  IncidentStatusFilter,
  OpenIncidentInput,
  OpsState,
  Runbook,
} from "../domain/ops-store.ts";
import {
  getRunbookForService,
  listAlerts,
  listIncidents,
  openIncident,
  resolveIncident,
} from "../domain/ops-store.ts";
import { buildSeedState } from "../domain/seed-data.ts";
import type { OpsStoreRepository } from "./ops-store.repository.ts";

/**
 * Adaptador do `OpsStoreRepository` mantido inteiramente em memória (sem IO) — reservado a
 * testes e ao bench (`src/bench.ts`), onde reprodutibilidade entre execuções e isolamento do
 * banco real (`SqliteOpsStore`) importam mais do que fidelidade ao SQL de produção
 * (spec 004 Assumptions; research.md item 2). Cada instância começa no dataset canônico
 * (`buildSeedState()`) e só é afetada por chamadas feitas na própria instância.
 */
export class InMemoryOpsStore implements OpsStoreRepository {
  private state: OpsState;

  constructor() {
    this.state = buildSeedState();
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
    return incident;
  }

  async resolveIncident(id: string, summary?: string): Promise<Incident> {
    const { state, incident } = resolveIncident(this.state, id, {
      now: new Date().toISOString(),
      summary,
    });
    this.state = state;
    return incident;
  }

  async listIncidents(status?: IncidentStatusFilter): Promise<Incident[]> {
    return listIncidents(this.state, status);
  }

  async getRunbook(service: string): Promise<Runbook | null> {
    return getRunbookForService(this.state, service);
  }

  /** Restaura o store para o dataset semeado, descartando incidentes criados. */
  reset(): void {
    this.state = buildSeedState();
  }

  /** Retorna o estado atual do store. Usado pelo bench para conferir o resultado das estratégias. */
  getState(): OpsState {
    return this.state;
  }
}
