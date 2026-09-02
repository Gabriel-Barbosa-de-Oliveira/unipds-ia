import { IncidentNotFoundError, InvalidSeverityError, ServiceNotFoundError } from "./errors.ts";

export type AlertStatus = "firing" | "resolved";
export type IncidentSeverity = "low" | "medium" | "high" | "critical";
export type IncidentStatus = "open" | "resolved";

export const INCIDENT_SEVERITIES: readonly IncidentSeverity[] = [
  "low",
  "medium",
  "high",
  "critical",
];

export interface Service {
  readonly id: string;
  readonly name: string;
}

export interface Alert {
  readonly id: string;
  readonly serviceId: string;
  readonly title: string;
  readonly status: AlertStatus;
  readonly createdAt: string;
}

export interface Incident {
  readonly id: string;
  readonly title: string;
  readonly serviceId: string;
  readonly severity: IncidentSeverity;
  readonly status: IncidentStatus;
  readonly createdAt: string;
  readonly resolvedAt: string | null;
}

export interface OpsState {
  readonly services: readonly Service[];
  readonly alerts: readonly Alert[];
  readonly incidents: readonly Incident[];
}

export interface OpenIncidentInput {
  readonly title: string;
  readonly service: string;
  readonly severity: string;
}

export interface OpenIncidentContext {
  readonly id: string;
  readonly now: string;
}

export interface ResolveIncidentContext {
  readonly now: string;
}

/** Retorna os alertas do estado, opcionalmente filtrados por status. Lista vazia é um resultado válido. */
export function listAlerts(state: OpsState, status?: AlertStatus): Alert[] {
  if (!status) {
    return [...state.alerts];
  }
  return state.alerts.filter((alert) => alert.status === status);
}

function findServiceByName(state: OpsState, name: string): Service | undefined {
  const target = name.trim().toLowerCase();
  return state.services.find((service) => service.name.toLowerCase() === target);
}

/** Abre um novo incidente. Não muta `state`; retorna o próximo estado e o incidente criado. */
export function openIncident(
  state: OpsState,
  input: OpenIncidentInput,
  ctx: OpenIncidentContext,
): { state: OpsState; incident: Incident } {
  if (!INCIDENT_SEVERITIES.includes(input.severity as IncidentSeverity)) {
    throw new InvalidSeverityError(input.severity);
  }

  const service = findServiceByName(state, input.service);
  if (!service) {
    throw new ServiceNotFoundError(input.service);
  }

  const incident: Incident = {
    id: ctx.id,
    title: input.title,
    serviceId: service.id,
    severity: input.severity as IncidentSeverity,
    status: "open",
    createdAt: ctx.now,
    resolvedAt: null,
  };

  return {
    state: { ...state, incidents: [...state.incidents, incident] },
    incident,
  };
}

/**
 * Resolve um incidente existente. Idempotente: resolver um incidente já `resolved`
 * retorna o mesmo incidente sem erro (não muta `state`).
 */
export function resolveIncident(
  state: OpsState,
  id: string,
  ctx: ResolveIncidentContext,
): { state: OpsState; incident: Incident } {
  const existing = state.incidents.find((incident) => incident.id === id);
  if (!existing) {
    throw new IncidentNotFoundError(id);
  }

  if (existing.status === "resolved") {
    return { state, incident: existing };
  }

  const resolved: Incident = { ...existing, status: "resolved", resolvedAt: ctx.now };
  return {
    state: {
      ...state,
      incidents: state.incidents.map((incident) => (incident.id === id ? resolved : incident)),
    },
    incident: resolved,
  };
}
