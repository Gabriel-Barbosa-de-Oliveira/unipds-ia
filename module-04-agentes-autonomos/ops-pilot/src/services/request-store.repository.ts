import type { TraceEvent } from "../agents/types.ts";
import type { RequestRecord } from "../domain/request-record.ts";

/**
 * Contrato de persistência das execuções do /chat (spec 014), independente do adaptador concreto
 * — mesmo padrão de `ConversationStore`.
 */
export interface RequestStore {
  /** Grava o registro e o trace numa única transação: ou entram os dois, ou nenhum. */
  save(record: RequestRecord, trace: readonly TraceEvent[]): Promise<void>;

  /** Registro + trace ordenado pela posição original; `undefined` quando o id não existe. */
  find(requestId: string): Promise<{ request: RequestRecord; trace: TraceEvent[] } | undefined>;
}
