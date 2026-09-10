import type {
  Alert,
  AlertStatus,
  Incident,
  IncidentStatusFilter,
  OpenIncidentInput,
  Runbook,
} from "../domain/ops-store.ts";

/**
 * Contrato comum usado pelas tools (`src/agents/tools.ts`) para acessar o store operacional,
 * independente do adaptador concreto (in-memory ou SQLite).
 */
export interface OpsStoreRepository {
  listAlerts(status?: AlertStatus): Promise<Alert[]>;
  openIncident(input: OpenIncidentInput): Promise<Incident>;
  resolveIncident(id: string, summary?: string): Promise<Incident>;
  listIncidents(status?: IncidentStatusFilter): Promise<Incident[]>;
  getRunbook(service: string): Promise<Runbook | null>;
}
