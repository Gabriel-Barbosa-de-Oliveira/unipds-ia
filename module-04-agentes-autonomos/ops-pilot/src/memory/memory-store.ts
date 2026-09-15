import { DatabaseSync } from "node:sqlite";

import { tool, type StructuredToolInterface } from "@langchain/core/tools";
import { z } from "zod";

import { bufferToFloatArray, floatArrayToBuffer, selectTopMatches } from "../domain/memory.ts";
import { embed as embedDefault } from "./embeddings.ts";

const DEFAULT_DB_PATH = "./data/opspilot.db";
const DEDUP_THRESHOLD = 0.92;
const RECALL_MIN_SCORE = 0.3;
const DEFAULT_RECALL_LIMIT = 3;

export interface RememberResult {
  stored: boolean;
  id?: string;
}

export interface RecallMatch {
  fact: string;
  score: number;
}

export interface ForgetResult {
  removed: boolean;
  fact?: string;
}

/**
 * Contrato consumido pelo controller HTTP (`src/http/server.ts`) e pelas tools do agente
 * (`createMemoryTools`, abaixo) — independente do adaptador concreto, mesmo padrão de
 * `ConversationStore`/`OpsStoreRepository`.
 */
export interface MemoryStore {
  remember(userId: string, fact: string): Promise<RememberResult>;
  recall(userId: string, query: string, limit?: number): Promise<RecallMatch[]>;
  forget(userId: string, description: string): Promise<ForgetResult>;
}

interface MemoryRow {
  id: string;
  fact: string;
  embedding: Uint8Array;
}

const DDL = `
  CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
  );
`;

/**
 * Implementação de `MemoryStore` sobre `node:sqlite` (`DatabaseSync`) — conexão própria e
 * independente das de `SqliteOpsStore`/`SqliteConversationStore` (research.md item 2). Compara
 * fatos por significado (produto escalar sobre embeddings já normalizados, research.md item 2),
 * nunca por texto.
 */
export class SqliteMemoryStore implements MemoryStore {
  private readonly path: string;
  private readonly embedFn: (text: string) => Promise<Float32Array>;
  private connection: DatabaseSync | undefined;

  constructor(
    path: string = process.env.OPSPILOT_DB ?? DEFAULT_DB_PATH,
    embedFn: (text: string) => Promise<Float32Array> = embedDefault,
  ) {
    this.path = path;
    this.embedFn = embedFn;
  }

  /** Conexão e DDL (idempotente) só são abertas no primeiro método realmente chamado — mesmo padrão de `SqliteOpsStore`. */
  private get db(): DatabaseSync {
    if (!this.connection) {
      this.connection = new DatabaseSync(this.path);
      this.connection.exec(DDL);
    }
    return this.connection;
  }

  async remember(userId: string, fact: string): Promise<RememberResult> {
    const embedding = await this.embedFn(fact);
    const existing = this.allForUser(userId);

    const [closest] = selectTopMatches(existing, embedding, { limit: 1, minScore: 0 });
    if (closest && closest.score > DEDUP_THRESHOLD) {
      return { stored: false };
    }

    const id = crypto.randomUUID();
    const now = new Date().toISOString();
    this.db
      .prepare("INSERT INTO memories (id, user_id, fact, embedding, created_at) VALUES (?, ?, ?, ?, ?)")
      .run(id, userId, fact, floatArrayToBuffer(embedding), now);

    return { stored: true, id };
  }

  async recall(userId: string, query: string, limit: number = DEFAULT_RECALL_LIMIT): Promise<RecallMatch[]> {
    const embedding = await this.embedFn(query);
    const candidates = this.allForUser(userId);

    return selectTopMatches(candidates, embedding, { limit, minScore: RECALL_MIN_SCORE }).map((match) => ({
      fact: match.item.fact,
      score: match.score,
    }));
  }

  async forget(userId: string, description: string): Promise<ForgetResult> {
    const embedding = await this.embedFn(description);
    const candidates = this.allForUser(userId);

    const [best] = selectTopMatches(candidates, embedding, { limit: 1, minScore: RECALL_MIN_SCORE });
    if (!best) {
      return { removed: false };
    }

    this.db.prepare("DELETE FROM memories WHERE id = ?").run(best.item.id);
    return { removed: true, fact: best.item.fact };
  }

  private allForUser(userId: string): { item: { id: string; fact: string }; embedding: Float32Array }[] {
    const rows = this.db
      .prepare("SELECT id, fact, embedding FROM memories WHERE user_id = ?")
      .all(userId) as unknown as MemoryRow[];

    return rows.map((row) => ({
      item: { id: row.id, fact: row.fact },
      embedding: bufferToFloatArray(row.embedding),
    }));
  }
}

/**
 * Tools do agente (LangChain) para `remember`/`forget` — `recall` não é tool (research.md item
 * 5), é chamado diretamente pelo controller. `userId` é capturado por closure, nunca um campo do
 * schema zod que o modelo preenche (research.md item 6) — elimina por construção o risco de um
 * `userId` errado vazar/apagar fato de outra pessoa.
 */
export function createMemoryTools(store: MemoryStore, userId: string): StructuredToolInterface[] {
  const rememberFactTool = tool(
    async ({ fact }: { fact: string }) => {
      const result = await store.remember(userId, fact);
      return JSON.stringify(result);
    },
    {
      name: "remember_fact",
      description:
        "Registra um fato novo sobre quem está conversando, para lembrar em conversas futuras. " +
        "Use quando a pessoa contar algo sobre si mesma ou sua forma de trabalhar que valha a pena " +
        "lembrar depois. Se um fato essencialmente igual já existir, não duplica (retorna stored: false).",
      schema: z.object({
        fact: z.string().min(1).describe("O fato a ser lembrado, em linguagem natural."),
      }),
    },
  );

  const forgetFactTool = tool(
    async ({ description }: { description: string }) => {
      const result = await store.forget(userId, description);
      return JSON.stringify(result);
    },
    {
      name: "forget_fact",
      description:
        "Remove um fato previamente registrado sobre quem está conversando. Use quando a pessoa " +
        "pedir explicitamente para esquecer algo que contou antes. Identifica o fato pela " +
        "descrição dada — se nada corresponder com confiança suficiente, não remove nada " +
        "(retorna removed: false).",
      schema: z.object({
        description: z.string().min(1).describe("Descrição do fato a esquecer."),
      }),
    },
  );

  return [rememberFactTool, forgetFactTool];
}
