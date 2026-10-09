# Data Model: Trace Persistido e Logs Estruturados

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**:
> - **Sucesso é gravado no nó `resposta`.** Quem grava o registro e o trace e emite os logs de trace e o `request.completed` (com `node: "resposta"`) é o nó `resposta` do grafo (`answerNode` em `src/graph/production-graph.ts`), com `requestStore` e `logger` injetados via `ProductionGraphDeps`. O controller continua gravando timeout e erro de execução, que nunca chegam a esse nó.
> - **Requisição abandonada.** O `RequestContext.abandoned()` evita que um grafo que já deu timeout grave ou logue depois.
> - **`userId` no registro.** O registro ganhou `userId` (coluna `user_id`).
> - **`request.completed` sem `status`.** O evento não tem `status`, porque o grafo não conhece HTTP.

## RequestRecord (`src/domain/request-record.ts`)

| Campo | Tipo | Regra |
|---|---|---|
| `requestId` | `string` (UUID) | gerado pelo servidor |
| `conversationId` | `string \| null` | `null` quando a falha ocorreu antes de existir conversa (não acontece hoje, porque erros pré-execução não são gravados) |
| `startedAt` | `string` (ISO-8601) | início do processamento |
| `durationMs` | `number` ≥ 0 | do recebimento até a gravação |
| `outcome` | `"ok" \| "timeout" \| "error"` | |
| `errorType` | `string \| null` | `error.name` quando `outcome !== "ok"`; `null` caso contrário |
| `route` | `RouteName \| null` | `null` em timeout ou erro |
| `routeSource` | `RouteSource \| null` | idem |
| `llmCalls`, `promptTokens` | `number \| null` | do `metrics` final; `null` em timeout ou erro |
| `tokenSource` | `TokenSource \| null` | idem |
| `modelUsed` | `string \| null` | idem |
| `historyMessages` | `number \| null` | idem |
| `context` | `{ breakdown, trimmed } \| null` | `metrics.contextBreakdown` e `metrics.contextTrimmed` |

**Invariante**: com `outcome === "ok"`, todas as métricas são não nulas e `errorType` é nulo.

## StoredTraceEvent

| Campo | Tipo | Regra |
|---|---|---|
| `position` | `number` | igual a `event.at` (já sequencial de `0` a `n-1` pelo grafo da 012) |
| `type` | `TraceEvent["type"]` | |
| `node` | `GraphNode \| null` | |
| `payload` | `TraceEvent` | o evento completo, sem alteração |

`restoreTrace(rows)` ordena por `position` e devolve os `payload`, produzindo um trace idêntico ao original (FR-007).

## Tabelas SQLite (`src/store/sqlite-request-store.ts`)

```sql
CREATE TABLE IF NOT EXISTS requests (
  id TEXT PRIMARY KEY,
  conversation_id TEXT,
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
```

## RequestStore (`src/services/request-store.repository.ts`)

```ts
interface RequestStore {
  /** Grava o registro e o trace numa única transação. */
  save(record: RequestRecord, trace: readonly TraceEvent[]): Promise<void>;
  /** Devolve `undefined` quando o id não existe. O trace vem ordenado por `position`. */
  find(requestId: string): Promise<{ request: RequestRecord; trace: TraceEvent[] } | undefined>;
}
```

## LogEvent (`src/obs/logger.ts`)

É uma união fechada. Os campos abaixo são **todos** os que existem, e não há campo livre.

| `event` | `level` | Campos |
|---|---|---|
| `request.received` | info | `requestId`, `method`, `path`, `hasConversationId`, `hasUserId`, `strategyOverride` (nome da rota ou `null`) |
| `request.rejected` | warn | `requestId`, `status`, `errorCode` (`invalid_body` \| `unknown_strategy` \| `conversation_not_found`) |
| `route.chosen` | info | `requestId`, `node`, `position`, `route`, `source` |
| `model.fallback` | warn | `requestId`, `node`, `position`, `from`, `to` |
| `tool.called` | info | `requestId`, `node`, `position`, `tool` |
| `request.completed` | info | `requestId`, `status`, `durationMs`, `route`, `llmCalls`, `promptTokens`, `tokenSource`, `modelUsed`, `traceEvents` |
| `request.failed` | error | `requestId`, `status`, `errorType`, `durationMs` |
| `persistence.failed` | error | `requestId`, `errorType` |
| `request.lookup` | info | `requestId`, `found` |

**Linha emitida**: `{"ts":"<ISO>","level":"<level>","event":"<event>",...campos}`, um único `JSON.stringify` seguido de `\n`.

**Nunca aparecem no log**: `message`, `answer`, `content`, `args`, `result`, `reason`, `steps`, histórico, memórias, `error.message` ou `error.stack`.
