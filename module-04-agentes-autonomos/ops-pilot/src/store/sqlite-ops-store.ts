import { DatabaseSync } from "node:sqlite";

import { IncidentNotFoundError, InvalidSeverityError, ServiceNotFoundError } from "../domain/errors.ts";
import {
  INCIDENT_SEVERITIES,
  type Alert,
  type AlertStatus,
  type Incident,
  type IncidentSeverity,
  type IncidentStatus,
  type IncidentStatusFilter,
  type OpenIncidentInput,
  type Runbook,
} from "../domain/ops-store.ts";
import { buildSeedState } from "../domain/seed-data.ts";
import type { OpsStoreRepository } from "../services/ops-store.repository.ts";

const DEFAULT_DB_PATH = "./data/opspilot.db";

interface ServiceRow {
  id: string;
  name: string;
}

interface AlertRow {
  id: string;
  service_id: string;
  title: string;
  status: AlertStatus;
  created_at: string;
}

interface IncidentRow {
  id: string;
  title: string;
  service_id: string;
  severity: IncidentSeverity;
  status: IncidentStatus;
  created_at: string;
  resolved_at: string | null;
  summary: string | null;
}

interface RunbookRow {
  id: string;
  service_id: string;
  content: string;
}

function toAlert(row: AlertRow): Alert {
  return {
    id: row.id,
    serviceId: row.service_id,
    title: row.title,
    status: row.status,
    createdAt: row.created_at,
  };
}

function toIncident(row: IncidentRow): Incident {
  return {
    id: row.id,
    title: row.title,
    serviceId: row.service_id,
    severity: row.severity,
    status: row.status,
    createdAt: row.created_at,
    resolvedAt: row.resolved_at,
    summary: row.summary,
  };
}

/**
 * Implementação de `OpsStoreRepository` sobre `node:sqlite` (`DatabaseSync`) — o adaptador de
 * persistência real do OpsPilot (constitution v1.1.0). O construtor roda uma DDL idempotente
 * (`CREATE TABLE IF NOT EXISTS`, com `CHECK` em todo campo de valor fechado); toda query usa
 * prepared statements, nunca SQL concatenado com valor de entrada.
 */
const DDL = `
  CREATE TABLE IF NOT EXISTS services (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
  );

  CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    service_id TEXT NOT NULL REFERENCES services(id),
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('firing', 'resolved')),
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    service_id TEXT NOT NULL REFERENCES services(id),
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    summary TEXT
  );

  CREATE TABLE IF NOT EXISTS runbooks (
    id TEXT PRIMARY KEY,
    service_id TEXT NOT NULL UNIQUE REFERENCES services(id),
    content TEXT NOT NULL
  );
`;

export class SqliteOpsStore implements OpsStoreRepository {
  private readonly path: string;
  private connection: DatabaseSync | undefined;

  constructor(path: string = process.env.OPSPILOT_DB ?? DEFAULT_DB_PATH) {
    this.path = path;
  }

  /**
   * Abre a conexão e roda a DDL (idempotente) só na primeira operação real sobre a instância —
   * nunca na construção. Assim, compor `new SqliteOpsStore()` (ex.: `opsTools` em
   * `src/agents/tools.ts`) nunca toca o arquivo em disco por si só; só o primeiro uso efetivo.
   */
  private get db(): DatabaseSync {
    if (!this.connection) {
      this.connection = new DatabaseSync(this.path);
      this.connection.exec(DDL);
    }
    return this.connection;
  }

  async listAlerts(status?: AlertStatus): Promise<Alert[]> {
    const rows = status
      ? (this.db.prepare("SELECT * FROM alerts WHERE status = ?").all(status) as unknown as AlertRow[])
      : (this.db.prepare("SELECT * FROM alerts").all() as unknown as AlertRow[]);
    return rows.map(toAlert);
  }

  async openIncident(input: OpenIncidentInput): Promise<Incident> {
    if (!INCIDENT_SEVERITIES.includes(input.severity as IncidentSeverity)) {
      throw new InvalidSeverityError(input.severity);
    }

    const service = this.findServiceRowByName(input.service);
    if (!service) {
      throw new ServiceNotFoundError(input.service);
    }

    const id = crypto.randomUUID();
    const now = new Date().toISOString();

    this.db
      .prepare(
        `INSERT INTO incidents (id, title, service_id, severity, status, created_at, resolved_at, summary)
         VALUES (?, ?, ?, ?, 'open', ?, NULL, NULL)`,
      )
      .run(id, input.title, service.id, input.severity, now);

    return this.getIncidentOrThrow(id);
  }

  async resolveIncident(id: string, summary?: string): Promise<Incident> {
    const existing = this.getIncidentRow(id);
    if (!existing) {
      throw new IncidentNotFoundError(id);
    }

    if (existing.status === "resolved") {
      return toIncident(existing);
    }

    const resolvedAt = new Date().toISOString();
    const nextSummary = summary ?? existing.summary;

    this.db
      .prepare("UPDATE incidents SET status = 'resolved', resolved_at = ?, summary = ? WHERE id = ?")
      .run(resolvedAt, nextSummary, id);

    return this.getIncidentOrThrow(id);
  }

  async listIncidents(status?: IncidentStatusFilter): Promise<Incident[]> {
    const rows =
      status && status !== "all"
        ? (this.db
            .prepare("SELECT * FROM incidents WHERE status = ? ORDER BY created_at")
            .all(status) as unknown as IncidentRow[])
        : (this.db.prepare("SELECT * FROM incidents ORDER BY created_at").all() as unknown as IncidentRow[]);
    return rows.map(toIncident);
  }

  async getRunbook(service: string): Promise<Runbook | null> {
    const serviceRow = this.findServiceRowByName(service);
    if (!serviceRow) {
      throw new ServiceNotFoundError(service);
    }

    const row = this.db.prepare("SELECT * FROM runbooks WHERE service_id = ?").get(serviceRow.id) as
      | RunbookRow
      | undefined;

    return row ? { id: row.id, serviceId: row.service_id, content: row.content } : null;
  }

  /** Semeia o cenário canônico (ver `seedCanonicalScenario`) — idempotente via `INSERT OR IGNORE`. */
  seed(): void {
    const state = buildSeedState();

    const insertService = this.db.prepare("INSERT OR IGNORE INTO services (id, name) VALUES (?, ?)");
    for (const service of state.services) {
      insertService.run(service.id, service.name);
    }

    const insertAlert = this.db.prepare(
      "INSERT OR IGNORE INTO alerts (id, service_id, title, status, created_at) VALUES (?, ?, ?, ?, ?)",
    );
    for (const alert of state.alerts) {
      insertAlert.run(alert.id, alert.serviceId, alert.title, alert.status, alert.createdAt);
    }

    const insertRunbook = this.db.prepare(
      "INSERT OR IGNORE INTO runbooks (id, service_id, content) VALUES (?, ?, ?)",
    );
    for (const runbook of state.runbooks) {
      insertRunbook.run(runbook.id, runbook.serviceId, runbook.content);
    }
  }

  private findServiceRowByName(name: string): ServiceRow | undefined {
    return this.db.prepare("SELECT * FROM services WHERE lower(name) = lower(?)").get(name.trim()) as
      | ServiceRow
      | undefined;
  }

  private getIncidentRow(id: string): IncidentRow | undefined {
    return this.db.prepare("SELECT * FROM incidents WHERE id = ?").get(id) as IncidentRow | undefined;
  }

  private getIncidentOrThrow(id: string): Incident {
    const row = this.getIncidentRow(id);
    if (!row) {
      throw new IncidentNotFoundError(id);
    }
    return toIncident(row);
  }
}

/**
 * Semeia o cenário canônico ("Mercadinho": `buildSeedState()` — services, alerts, runbooks;
 * nunca incidents) em `store`. Idempotente: reexecutar sobre um banco já semeado não duplica
 * nem falha. Reaproveitado por `src/scripts/seed.ts` e pelos testes sobre `":memory:"`.
 */
export function seedCanonicalScenario(store: SqliteOpsStore): void {
  store.seed();
}
