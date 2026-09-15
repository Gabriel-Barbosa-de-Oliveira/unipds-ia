import { DatabaseSync } from "node:sqlite";

import { ConversationNotFoundError } from "../domain/errors.ts";
import type { ConversationMessage, ConversationRole } from "../domain/conversation.ts";
import type { ConversationStore } from "../services/conversation-store.repository.ts";

const DEFAULT_DB_PATH = "./data/opspilot.db";

interface MessageRow {
  role: ConversationRole;
  content: string;
}

/**
 * Implementação de `ConversationStore` sobre `node:sqlite` (`DatabaseSync`) — mesmo padrão de
 * `SqliteOpsStore` (`004-ops-persistence`), mas com conexão própria e independente sobre duas
 * tabelas novas (`conversations`, `messages`): conversa é um bounded context distinto de dados
 * operacionais (research.md item 2).
 */
const DDL = `
  CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
`;

export class SqliteConversationStore implements ConversationStore {
  private readonly path: string;
  private connection: DatabaseSync | undefined;

  constructor(path: string = process.env.OPSPILOT_DB ?? DEFAULT_DB_PATH) {
    this.path = path;
  }

  /** Conexão e DDL (idempotente) só são abertas no primeiro método realmente chamado — mesmo padrão de `SqliteOpsStore`. */
  private get db(): DatabaseSync {
    if (!this.connection) {
      this.connection = new DatabaseSync(this.path);
      this.connection.exec(DDL);
    }
    return this.connection;
  }

  async create(): Promise<string> {
    const id = crypto.randomUUID();
    const now = new Date().toISOString();

    this.db.prepare("INSERT INTO conversations (id, created_at) VALUES (?, ?)").run(id, now);

    return id;
  }

  async append(conversationId: string, messages: ConversationMessage[]): Promise<void> {
    this.assertConversationExists(conversationId);

    const insert = this.db.prepare(
      "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
    );

    for (const message of messages) {
      insert.run(conversationId, message.role, message.content, new Date().toISOString());
    }
  }

  async lastMessages(conversationId: string, limit: number): Promise<ConversationMessage[]> {
    this.assertConversationExists(conversationId);

    const rows = this.db
      .prepare(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
      )
      .all(conversationId, limit) as unknown as MessageRow[];

    return rows.reverse().map((row) => ({ role: row.role, content: row.content }));
  }

  private assertConversationExists(conversationId: string): void {
    const row = this.db.prepare("SELECT id FROM conversations WHERE id = ?").get(conversationId);
    if (!row) {
      throw new ConversationNotFoundError(conversationId);
    }
  }
}
