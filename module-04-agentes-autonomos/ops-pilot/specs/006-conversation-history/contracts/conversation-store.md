# Contract: `ConversationStore` (`src/services/conversation-store.repository.ts`) e `SqliteConversationStore` (`src/store/sqlite-conversation-store.ts`)

Ver [data-model.md](../data-model.md) para a forma das entidades e [research.md](../research.md) para as decisões por trás deste contrato. Mesmo padrão de `OpsStoreRepository`/`SqliteOpsStore` (`004-ops-persistence`): a interface vive em `src/services/`, a implementação concreta em `src/store/`.

## Interface `ConversationStore`

```ts
interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

interface ConversationStore {
  create(): Promise<string>;
  append(conversationId: string, messages: ConversationMessage[]): Promise<void>;
  lastMessages(conversationId: string, limit: number): Promise<ConversationMessage[]>;
}
```

## Construção (`SqliteConversationStore`)

```ts
new SqliteConversationStore(path?: string)
```

- `path` default: `process.env.OPSPILOT_DB ?? "./data/opspilot.db"` — mesmo arquivo usado por `SqliteOpsStore`, em uma conexão `DatabaseSync` própria e independente (research.md item 2).
- `":memory:"` abre um banco novo e vazio, isolado por instância — usado por testes.
- Conexão e DDL (`conversations`, `messages`, `CREATE TABLE IF NOT EXISTS`) são lazy, disparadas só no primeiro método chamado sobre a instância — mesmo padrão de `SqliteOpsStore`.

## Métodos

Todos usam prepared statements; nenhuma query concatena valor de entrada na string SQL.

### `create(): Promise<string>`

Gera um `id` novo (`crypto.randomUUID()`) e insere uma `Conversation` vazia. Nunca falha por colisão de id.

### `append(conversationId: string, messages: ConversationMessage[]): Promise<void>`

`conversationId` desconhecido → `ConversationNotFoundError` (nunca cria a conversa implicitamente). Insere cada mensagem de `messages`, na ordem do array, associada a `conversationId`; a ordem de inserção define a ordem de leitura (chave autoincrementada interna, nunca `created_at` — research.md item 7). Usada pelo controller com exatamente dois elementos por turno: `[{role:"user", content: message}, {role:"assistant", content: answer}]`.

### `lastMessages(conversationId: string, limit: number): Promise<ConversationMessage[]>`

`conversationId` desconhecido → `ConversationNotFoundError`. Conversa conhecida: retorna até `limit` mensagens mais recentes, em ordem cronológica (mais antiga primeiro) — nunca em ordem reversa, já que o consumidor (composição do prompt) precisa da sequência real da conversa. Conversa sem mensagens (recém-criada) → lista vazia, não erro.

## Erros de domínio

| Classe | Quando |
|---|---|
| `ConversationNotFoundError` (nova, `src/domain/errors.ts`) | `append`/`lastMessages` chamados com um `conversationId` que não corresponde a nenhuma `Conversation` existente |

Traduzido para HTTP `404` exclusivamente em `src/http/server.ts` (`errorMiddleware`) — nunca dentro do store ou do controller antes desse ponto (Princípio III).
