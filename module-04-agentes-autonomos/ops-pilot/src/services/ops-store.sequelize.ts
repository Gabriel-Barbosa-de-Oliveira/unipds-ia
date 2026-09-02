import {
  INCIDENT_SEVERITIES,
  type Alert,
  type AlertStatus,
  type Incident,
  type IncidentSeverity,
  type OpenIncidentInput,
} from "../domain/ops-store.ts";
import { IncidentNotFoundError, InvalidSeverityError, ServiceNotFoundError } from "../domain/errors.ts";
import { AlertModel } from "../models/sequelize/alert.model.ts";
import { IncidentModel } from "../models/sequelize/incident.model.ts";
import { ServiceModel } from "../models/sequelize/service.model.ts";
import type { OpsStoreRepository } from "./ops-store.repository.ts";

function toAlert(row: AlertModel): Alert {
  return {
    id: row.id,
    serviceId: row.serviceId,
    title: row.title,
    status: row.status,
    createdAt: new Date(row.createdAt).toISOString(),
  };
}

function toIncident(row: IncidentModel): Incident {
  return {
    id: row.id,
    title: row.title,
    serviceId: row.serviceId,
    severity: row.severity,
    status: row.status,
    createdAt: new Date(row.createdAt).toISOString(),
    resolvedAt: row.resolvedAt ? new Date(row.resolvedAt).toISOString() : null,
  };
}

/**
 * Adaptador Sequelize/MySQL do OpsStoreRepository (stack obrigatória da constitution).
 * Usado pelo script de seed e por uma futura integração de persistência real; não é o
 * adaptador padrão das tools/arena/testes desta feature (ver `ops-store.memory.ts`).
 */
export class SequelizeOpsStore implements OpsStoreRepository {
  async listAlerts(status?: AlertStatus): Promise<Alert[]> {
    const rows = await AlertModel.findAll({ where: status ? { status } : {} });
    return rows.map(toAlert);
  }

  async openIncident(input: OpenIncidentInput): Promise<Incident> {
    if (!INCIDENT_SEVERITIES.includes(input.severity as IncidentSeverity)) {
      throw new InvalidSeverityError(input.severity);
    }

    const service = await ServiceModel.findOne({ where: { name: input.service.trim() } });
    if (!service) {
      throw new ServiceNotFoundError(input.service);
    }

    const now = new Date();
    const row = await IncidentModel.create({
      id: crypto.randomUUID(),
      title: input.title,
      serviceId: service.id,
      severity: input.severity as IncidentSeverity,
      status: "open",
      createdAt: now.toISOString(),
      resolvedAt: null,
    });

    return toIncident(row);
  }

  async resolveIncident(id: string): Promise<Incident> {
    const row = await IncidentModel.findByPk(id);
    if (!row) {
      throw new IncidentNotFoundError(id);
    }

    if (row.status === "resolved") {
      return toIncident(row);
    }

    row.status = "resolved";
    row.resolvedAt = new Date().toISOString();
    await row.save();

    return toIncident(row);
  }
}
