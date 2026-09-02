export class ServiceNotFoundError extends Error {
  readonly service: string;

  constructor(service: string) {
    super(`Service not found: ${service}`);
    this.name = "ServiceNotFoundError";
    this.service = service;
  }
}

export class IncidentNotFoundError extends Error {
  readonly id: string;

  constructor(id: string) {
    super(`Incident not found: ${id}`);
    this.name = "IncidentNotFoundError";
    this.id = id;
  }
}

export class InvalidSeverityError extends Error {
  readonly severity: string;

  constructor(severity: string) {
    super(`Invalid severity: ${severity}`);
    this.name = "InvalidSeverityError";
    this.severity = severity;
  }
}
