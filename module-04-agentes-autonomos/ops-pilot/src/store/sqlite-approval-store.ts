import { DatabaseSync } from "node:sqlite";

import {
  effectiveStatus,
  type ApprovalDecision,
  type ApprovalStatus,
  type GatedToolName,
  type PendingAction,
} from "../domain/approval.ts";
import type { ApprovalStore, DecideResult } from "../services/approval-store.repository.ts";

const DEFAULT_DB_PATH = "./data/opspilot.db";

/** Ações que mudam a produção aguardando decisão humana — spec 015, data-model.md. */
const DDL = `
  CREATE TABLE IF NOT EXISTS pending_actions (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    user_id TEXT,
    tool TEXT NOT NULL CHECK (tool IN ('open_incident', 'resolve_incident')),
    args_json TEXT NOT NULL,
    reason TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'denied')),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    decided_at TEXT,
    decision_request_id TEXT
  );

  CREATE INDEX IF NOT EXISTS idx_pending_actions_conversation ON pending_actions(conversation_id);
`;

interface PendingActionRow {
  id: string;
  request_id: string;
  conversation_id: string;
  user_id: string | null;
  tool: GatedToolName;
  args_json: string;
  reason: string | null;
  status: ApprovalStatus;
  created_at: string;
  expires_at: string;
  decided_at: string | null;
  decision_request_id: string | null;
}

function toAction(row: PendingActionRow): PendingAction {
  return {
    id: row.id,
    requestId: row.request_id,
    conversationId: row.conversation_id,
    userId: row.user_id,
    tool: row.tool,
    args: JSON.parse(row.args_json) as Record<string, unknown>,
    reason: row.reason,
    status: row.status,
    createdAt: row.created_at,
    expiresAt: row.expires_at,
    decidedAt: row.decided_at,
    decisionRequestId: row.decision_request_id,
  };
}

export class SqliteApprovalStore implements ApprovalStore {
  private readonly path: string;
  private connection: DatabaseSync | undefined;

  constructor(path: string = process.env.OPSPILOT_DB ?? DEFAULT_DB_PATH) {
    this.path = path;
  }

  /** Conexão e DDL (idempotente) só no primeiro uso — mesmo padrão de `SqliteRequestStore`. */
  private get db(): DatabaseSync {
    if (!this.connection) {
      this.connection = new DatabaseSync(this.path);
      this.connection.exec(DDL);
    }
    return this.connection;
  }

  async create(action: PendingAction): Promise<void> {
    this.db
      .prepare(
        `INSERT INTO pending_actions (id, request_id, conversation_id, user_id, tool, args_json, reason, status,
           created_at, expires_at, decided_at, decision_request_id)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        action.id,
        action.requestId,
        action.conversationId,
        action.userId,
        action.tool,
        JSON.stringify(action.args),
        action.reason,
        action.status,
        action.createdAt,
        action.expiresAt,
        action.decidedAt,
        action.decisionRequestId,
      );
  }

  async find(id: string): Promise<PendingAction | undefined> {
    const row = this.db.prepare("SELECT * FROM pending_actions WHERE id = ?").get(id) as PendingActionRow | undefined;
    return row ? toAction(row) : undefined;
  }

  /**
   * Um único UPDATE condicional decide a corrida (duplo clique, duas abas): só uma decisão encontra
   * a linha pendente e no prazo. Sem efeito → relê para classificar o motivo (research.md item 3).
   */
  async decide(id: string, decision: ApprovalDecision, now: Date, decisionRequestId: string): Promise<DecideResult> {
    const nowIso = now.toISOString();
    const { changes } = this.db
      .prepare(
        `UPDATE pending_actions SET status = ?, decided_at = ?, decision_request_id = ?
         WHERE id = ? AND status = 'pending' AND expires_at > ?`,
      )
      .run(decision, nowIso, decisionRequestId, id, nowIso);

    const action = await this.find(id);
    if (!action) {
      return { ok: false, reason: "not_found" };
    }
    if (Number(changes) === 1) {
      return { ok: true, action };
    }
    return { ok: false, reason: effectiveStatus(action, now) === "expired" ? "expired" : "already_decided", action };
  }
}
