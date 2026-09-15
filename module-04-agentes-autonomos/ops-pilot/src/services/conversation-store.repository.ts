import type { ConversationMessage } from "../domain/conversation.ts";

export type { ConversationMessage };

/**
 * Contrato usado pelo controller HTTP (`src/http/server.ts`) para persistir e recuperar o
 * histórico de uma conversa, independente do adaptador concreto (mesmo padrão de
 * `OpsStoreRepository`, `004-ops-persistence`).
 *
 * `append`/`lastMessages` MUST lançar `ConversationNotFoundError` (`src/domain/errors.ts`)
 * quando `conversationId` não corresponde a nenhuma conversa existente — nunca criam a
 * conversa implicitamente (research.md item 4).
 */
export interface ConversationStore {
  /** Cria uma conversa nova e retorna seu identificador. Nunca falha por conflito de id. */
  create(): Promise<string>;

  /** Grava as mensagens, na ordem do array, ao final do histórico de `conversationId`. */
  append(conversationId: string, messages: ConversationMessage[]): Promise<void>;

  /**
   * Retorna até `limit` mensagens mais recentes de `conversationId`, em ordem cronológica
   * (mais antiga primeiro). Conversa existente sem mensagens retorna lista vazia (não é erro).
   */
  lastMessages(conversationId: string, limit: number): Promise<ConversationMessage[]>;
}
