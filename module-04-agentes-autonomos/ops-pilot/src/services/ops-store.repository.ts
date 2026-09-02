import type { Alert, AlertStatus, Incident, OpenIncidentInput } from "../domain/ops-store.ts";

/**
 * Contrato comum usado pelas tools (`src/agents/tools.ts`) para acessar o store operacional,
 * independente do adaptador concreto (in-memory ou Sequelize/MySQL).
 */
export interface OpsStoreRepository {
  listAlerts(status?: AlertStatus): Promise<Alert[]>;
  openIncident(input: OpenIncidentInput): Promise<Incident>;
  resolveIncident(id: string): Promise<Incident>;
}
