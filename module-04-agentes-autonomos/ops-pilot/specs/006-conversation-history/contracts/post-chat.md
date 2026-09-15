# Contract: `POST /chat` (`src/http/server.ts`, `createApp()`) — após `006-conversation-history`

Estende o contrato definido em [`003-chat-endpoint/contracts/post-chat.md`](../../003-chat-endpoint/contracts/post-chat.md) (permanece a referência histórica daquela feature) com suporte a conversa persistente. Nenhum campo existente é removido ou renomeado — apenas adições.

## Requisição

```http
POST /chat
Content-Type: application/json

{
  "message": "e o status desse incidente?",
  "strategy": "react",
  "reflect": false,
  "conversationId": "b0a3c1f2-..."
}
```

| Campo | Obrigatório | Tipo | Padrão | Novo? |
|---|---|---|---|---|
| `message` | Sim | string não vazia | — (400 se ausente/vazio/tipo errado) | Não |
| `strategy` | Não | string, DEVE corresponder a um nome registrado (`react`, `plan-and-execute`) | `react` | Não |
| `reflect` | Não | boolean | `false` | Não |
| `conversationId` | Não | string | — (nenhum: inicia conversa nova) | **Sim** |

## Respostas

### `200 OK` — sucesso

```json
{
  "answer": "o incidente inc-123 continua aberto, severidade alta.",
  "trace": [ { "type": "thought", "at": 0, "content": "..." }, "..." ],
  "metrics": { "llmCalls": 2, "latencyMs": 850, "historyMessages": 4 },
  "conversationId": "b0a3c1f2-..."
}
```

Frente ao contrato de `003-chat-endpoint`: `metrics` ganha `historyMessages` (quantidade de mensagens de histórico consideradas nesta resposta, 0–12) e o corpo ganha `conversationId` (o informado na requisição, ou um novo, gerado quando nenhum foi informado). `answer`, `trace`, `metrics.llmCalls` e `metrics.latencyMs` continuam com o mesmo significado.

### `400 Bad Request` — corpo inválido

Sem mudança frente a `003-chat-endpoint`. `conversationId`, quando informado, deve ser string — um `conversationId` de tipo errado (ex.: número) cai neste caso, antes de qualquer resolução de conversa.

### `404 Not Found` — conversa desconhecida (novo)

```json
{ "error": "conversation_not_found", "conversationId": "id-que-nao-existe" }
```

Disparado quando `conversationId` é informado mas não corresponde a nenhuma conversa existente (FR-007). Nenhuma execução de raciocínio ocorre — a resolução da conversa acontece antes de qualquer chamada à estratégia, no mesmo espírito de `400`/`422` (nunca gasta uma chamada ao modelo por um erro de fronteira já detectável).

### `422 Unprocessable Entity` — estratégia desconhecida

Sem mudança frente a `003-chat-endpoint`.

### `504 Gateway Timeout` — tempo excedido

Sem mudança frente a `003-chat-endpoint`. Uma conversa nova pode já ter sido criada no armazenamento antes do timeout (ver research.md item 6) — a resposta de erro não inclui `conversationId`, então esse id nunca chega a quem chamou.

### `500 Internal Server Error` — falha não classificada

Sem mudança frente a `003-chat-endpoint`.

## Regras do contrato

- Todas as regras já estabelecidas em `003-chat-endpoint` continuam valendo (nenhum dos erros previsíveis dispara execução de raciocínio; `strategy` ausente equivale a `"react"`; `reflect: true` nunca é, por si só, erro; isolamento entre requisições concorrentes).
- Duas conversas diferentes nunca compartilham histórico, mesmo sob requisições concorrentes (spec FR-009, SC-005).
- O histórico usado na composição do prompt é sempre, no máximo, as 12 mensagens mais recentes da conversa referenciada — independentemente de quantas mensagens a conversa já acumulou no total (spec FR-004).
- A mensagem gravada no histórico após cada turno é o texto literal de `message` e de `answer` — nunca o prompt já composto com histórico (research.md item 3).
