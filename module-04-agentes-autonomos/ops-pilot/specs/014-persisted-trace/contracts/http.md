# Contract: HTTP (`src/http/server.ts`)

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**:
> - **Sucesso é gravado no nó `resposta`.** Quem grava o registro e o trace e emite os logs de trace e o `request.completed` (com `node: "resposta"`) é o nó `resposta` do grafo (`answerNode` em `src/graph/production-graph.ts`), com `requestStore` e `logger` injetados via `ProductionGraphDeps`. O controller continua gravando timeout e erro de execução, que nunca chegam a esse nó.
> - **Requisição abandonada.** O `RequestContext.abandoned()` evita que um grafo que já deu timeout grave ou logue depois.
> - **`userId` no registro.** O registro ganhou `userId` (coluna `user_id`).
> - **`request.completed` sem `status`.** O evento não tem `status`, porque o grafo não conhece HTTP.

## `POST /chat` (delta sobre a 012/013)

Toda resposta passa a ter:

- o cabeçalho `X-Request-Id: <uuid>`;
- o campo `requestId` no corpo JSON, com o mesmo valor do cabeçalho.

O `X-Request-Id` enviado pelo cliente é ignorado.

| Status | Corpo (o que muda) | Persistido? |
|---|---|---|
| 200 | `{ requestId, answer, route, trace, conversationId, metrics }` | sim, com `outcome: "ok"` e o trace completo |
| 400 | `{ requestId, error: "invalid_body", issues }` | não |
| 404 | `{ requestId, error: "conversation_not_found", conversationId }` | não |
| 422 | `{ requestId, error: "unknown_strategy", strategy }` | não |
| 504 | `{ requestId, error: "timeout", timeoutMs }` | sim, com `outcome: "timeout"` e trace vazio |
| 500 | `{ requestId, error: "internal_error" }` | sim, com `outcome: "error"`, `errorType` e trace vazio |

Se a gravação falhar, a resposta não muda: o log registra `persistence.failed`.

## `GET /requests/:id` (novo)

### 200

```json
{
  "request": {
    "requestId": "7b1e…",
    "conversationId": "c9a0…",
    "startedAt": "2026-10-09T14:03:11.204Z",
    "durationMs": 2310,
    "outcome": "ok",
    "errorType": null,
    "route": "react",
    "routeSource": "router",
    "llmCalls": 3,
    "promptTokens": 1450,
    "tokenSource": "real",
    "modelUsed": "openai/gpt-4o-mini",
    "historyMessages": 0,
    "context": { "breakdown": { "...": 0 }, "trimmed": { "historyMessages": 0, "recalledFacts": 0 } }
  },
  "trace": [
    { "type": "route", "at": 0, "node": "roteador", "route": "react", "reason": "…", "source": "router" },
    { "type": "action", "at": 1, "node": "react", "tool": "list_alerts", "args": { "status": "firing" } },
    { "type": "observation", "at": 2, "node": "react", "result": [] },
    { "type": "answer", "at": 3, "node": "react", "content": "…" }
  ]
}
```

O `trace` é **deep-equal** ao `trace` da resposta 200 original do `/chat` (FR-007).

### 404

```json
{ "error": "request_not_found", "requestId": "<o id pedido>" }
```

Vale tanto para um id inexistente quanto para um id com formato inválido (não UUID).
