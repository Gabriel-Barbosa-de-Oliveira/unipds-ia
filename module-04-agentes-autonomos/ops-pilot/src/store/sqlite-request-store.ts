import { DatabaseSync } from "node:sqlite";

import type { GraphNode, RouteName, RouteSource, TraceEvent } from "../agents/types.ts";
import type { TokenSource } from "../context/tokens.ts";
import {
  restoreTrace,
  toStoredTraceEvents,
  type RequestOutcome,
  type RequestRecord,
  type StoredTraceEvent,
} from "../domain/request-record.ts";
import type { RequestStore } from "../services/request-store.repository.ts";

const DEFAULT_DB_PATH = "./data/opspilot.db";

/**
 * Execuções do /chat (`requests`) e seus traces (`trace_events`) — spec 014, data-model.md.
 * Colunas tipadas para o que se filtra/agrega; o evento completo em `payload_json`, para que o
 * trace reconstruído seja idêntico ao original mesmo com variantes de evento novas.
 */
const DDL = `
  CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    user_id TEXT,
    started_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    outcome TEXT NOT NULL CHECK (outcome IN ('ok', 'timeout', 'error')),
    error_type TEXT,
    route TEXT CHECK (route IS NULL OR route IN ('react', 'planExecute', 'reflect')),
    route_source TEXT CHECK (route_source IS NULL OR route_source IN ('router', 'override', 'fallback')),
    llm_calls INTEGER,
    prompt_tokens INTEGER,
    token_source TEXT CHECK (token_source IS NULL OR token_source IN ('real', 'estimated', 'mixed')),
    model_used TEXT,
    history_messages INTEGER,
    context_json TEXT
  );

  CREATE TABLE IF NOT EXISTS trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL REFERENCES requests(id),
    position INTEGER NOT NULL,
    type TEXT NOT NULL,
    node TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE (request_id, position)
  );

  CREATE INDEX IF NOT EXISTS idx_requests_conversation ON requests(conversation_id);
`;

interface RequestRow {
  id: string;
  conversation_id: string | null;
  user_id: string | null;
  started_at: string;
  duration_ms: number;
  outcome: RequestOutcome;
  error_type: string | null;
  route: RouteName | null;
  route_source: RouteSource | null;
  llm_calls: number | null;
  prompt_tokens: number | null;
  token_source: TokenSource | null;
  model_used: string | null;
  history_messages: number | null;
  context_json: string | null;
}

interface TraceRow {
  position: number;
  type: TraceEvent["type"];
  node: GraphNode | null;
  payload_json: string;
}

function toRecord(row: RequestRow): RequestRecord {
  return {
    requestId: row.id,
    conversationId: row.conversation_id,
    userId: row.user_id,
    startedAt: row.started_at,
    durationMs: row.duration_ms,
    outcome: row.outcome,
    errorType: row.error_type,
    route: row.route,
    routeSource: row.route_source,
    llmCalls: row.llm_calls,
    promptTokens: row.prompt_tokens,
    tokenSource: row.token_source,
    modelUsed: row.model_used,
    historyMessages: row.history_messages,
    context: row.context_json ? (JSON.parse(row.context_json) as RequestRecord["context"]) : null,
  };
}

function toStored(row: TraceRow): StoredTraceEvent {
  return { position: row.position, type: row.type, node: row.node, payload: JSON.parse(row.payload_json) as TraceEvent };
}

export class SqliteRequestStore implements RequestStore {
  private readonly path: string;
  private connection: DatabaseSync | undefined;

  constructor(path: string = process.env.OPSPILOT_DB ?? DEFAULT_DB_PATH) {
    this.path = path;
  }

  /** Conexão e DDL (idempotente) só no primeiro uso — mesmo padrão de `SqliteConversationStore`. */
  private get db(): DatabaseSync {
    if (!this.connection) {
      this.connection = new DatabaseSync(this.path);
      this.connection.exec(DDL);
    }
    return this.connection;
  }

  async save(record: RequestRecord, trace: readonly TraceEvent[]): Promise<void> {
    const db = this.db;
    db.exec("BEGIN");
    try {
      db.prepare(
        `INSERT INTO requests (id, conversation_id, user_id, started_at, duration_ms, outcome, error_type, route,
           route_source, llm_calls, prompt_tokens, token_source, model_used, history_messages, context_json)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        record.requestId,
        record.conversationId,
        record.userId,
        record.startedAt,
        record.durationMs,
        record.outcome,
        record.errorType,
        record.route,
        record.routeSource,
        record.llmCalls,
        record.promptTokens,
        record.tokenSource,
        record.modelUsed,
        record.historyMessages,
        record.context ? JSON.stringify(record.context) : null,
      );

      const insertEvent = db.prepare(
        "INSERT INTO trace_events (request_id, position, type, node, payload_json) VALUES (?, ?, ?, ?, ?)",
      );
      for (const event of toStoredTraceEvents(trace)) {
        insertEvent.run(record.requestId, event.position, event.type, event.node, JSON.stringify(event.payload));
      }

      db.exec("COMMIT");
    } catch (error) {
      db.exec("ROLLBACK");
      throw error;
    }
  }

  async find(requestId: string): Promise<{ request: RequestRecord; trace: TraceEvent[] } | undefined> {
    const row = this.db.prepare("SELECT * FROM requests WHERE id = ?").get(requestId) as unknown as
      | RequestRow
      | undefined;
    if (!row) {
      return undefined;
    }

    const traceRows = this.db
      .prepare("SELECT position, type, node, payload_json FROM trace_events WHERE request_id = ? ORDER BY position")
      .all(requestId) as unknown as TraceRow[];

    return { request: toRecord(row), trace: restoreTrace(traceRows.map(toStored)) };
  }
}
