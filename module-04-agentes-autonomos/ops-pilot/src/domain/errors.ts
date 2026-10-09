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

export class UnknownStrategyError extends Error {
  readonly strategy: string;

  constructor(strategy: string) {
    super(`Unknown strategy: ${strategy}`);
    this.name = "UnknownStrategyError";
    this.strategy = strategy;
  }
}

export class ConversationNotFoundError extends Error {
  readonly conversationId: string;

  constructor(conversationId: string) {
    super(`Conversation not found: ${conversationId}`);
    this.name = "ConversationNotFoundError";
    this.conversationId = conversationId;
  }
}

export class ChatTimeoutError extends Error {
  readonly timeoutMs: number;

  constructor(timeoutMs: number) {
    super(`Chat execution exceeded timeout of ${timeoutMs}ms`);
    this.name = "ChatTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}

export class ApprovalNotFoundError extends Error {
  readonly id: string;

  constructor(id: string) {
    super(`Approval not found: ${id}`);
    this.name = "ApprovalNotFoundError";
    this.id = id;
  }
}

export class ApprovalAlreadyDecidedError extends Error {
  readonly id: string;
  readonly status: "approved" | "denied";

  constructor(id: string, status: "approved" | "denied") {
    super(`Approval already decided: ${id} (${status})`);
    this.name = "ApprovalAlreadyDecidedError";
    this.id = id;
    this.status = status;
  }
}

export class ApprovalExpiredError extends Error {
  readonly id: string;
  readonly expiresAt: string;

  constructor(id: string, expiresAt: string) {
    super(`Approval expired: ${id} at ${expiresAt}`);
    this.name = "ApprovalExpiredError";
    this.id = id;
    this.expiresAt = expiresAt;
  }
}
