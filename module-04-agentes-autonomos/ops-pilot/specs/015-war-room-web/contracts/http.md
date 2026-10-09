# Contrato HTTP: War Room Web

Este contrato altera e amplia a API descrita nas specs 003, 006, 012 e 014. Os campos não citados aqui não mudam.

## CORS (todas as rotas)

Configuração: `OPSPILOT_CORS_ORIGINS="https://exemplo.com,http://localhost:5173"`. A comparação é exata, sem curinga. Sem a variável, o padrão é `http://localhost:5173`.

| Requisição | Origem permitida | Origem não permitida | Sem `Origin` |
|---|---|---|---|
| Qualquer, exceto `OPTIONS` | `Access-Control-Allow-Origin: <origin>`, `Vary: Origin`, `Access-Control-Expose-Headers: X-Request-Id` | sem cabeçalhos CORS | sem cabeçalhos CORS |
| `OPTIONS` (preflight) | **204** + os cabeçalhos acima + `Access-Control-Allow-Methods: GET, POST, OPTIONS`, `Access-Control-Allow-Headers: Content-Type`, `Access-Control-Max-Age: 600` | **204** sem cabeçalhos CORS | passa adiante, como hoje |

`Access-Control-Allow-Credentials` nunca é enviado.

## `POST /chat`

O corpo da requisição não muda (`message`, `strategy?`, `reflect?`, `conversationId?`, `userId?`).

### 200: resposta final (não muda)

```json
{
  "requestId": "uuid",
  "answer": "string",
  "trace": [TraceEvent],
  "route": { "route": "react|planExecute|reflect", "reason": "string", "source": "router|override|fallback" },
  "conversationId": "string",
  "metrics": { "llmCalls": 0, "latencyMs": 0, "promptTokens": 0, "tokenSource": "real|estimated|mixed", "modelUsed": "string", "historyMessages": 0, "contextBreakdown": {}, "contextTrimmed": {} }
}
```

### 202: ação aguardando aprovação (novo)

O servidor responde 202 quando o agente chamou `open_incident` ou `resolve_incident`. **Nada foi executado.**

```json
{
  "requestId": "uuid",
  "status": "awaiting_approval",
  "approval": {
    "id": "uuid",
    "tool": "resolve_incident",
    "args": { "id": "INC-42", "summary": "rollback aplicado" },
    "summary": "Resolver o incidente INC-42",
    "reason": "O alerta de checkout voltou ao normal após o rollback.",
    "expiresAt": "2026-10-09T12:15:00.000Z"
  },
  "trace": [TraceEvent],
  "route": { "route": "react", "reason": "string", "source": "router" },
  "conversationId": "string",
  "metrics": { "...": "igual ao 200" }
}
```

- `summary` é texto em português gerado por uma função pura a partir de `tool` + `args` (por exemplo, "Abrir incidente critical em checkout-api: Latência alta").
- `reason` pode ser `null`.
- Não há campo `answer` no 202. O texto que o modelo escreveu depois da pausa é descartado, porque poderia afirmar que a ação aconteceu.
- O cabeçalho `X-Request-Id` é igual ao `requestId`, como no 200.
- O histórico da conversa recebe `user: <mensagem>` e `assistant: "Aguardando aprovação: <summary>"`.

As respostas de erro (400, 404, 422, 500, 504) não mudam.

## `POST /approvals/:id` (novo)

```json
{ "decision": "approve" | "deny" }
```

O corpo é validado com zod: `z.object({ decision: z.enum(["approve", "deny"]) })`. O `:id` precisa ser um UUID.

### 200: decisão aplicada

O formato é o mesmo do 200 do `/chat`, com `route: null` e `metrics: null`, mais o campo `approval`:

```json
{
  "requestId": "uuid (novo, da decisão)",
  "answer": "Incidente INC-42 resolvido.",
  "trace": [
    { "type": "action", "at": 0, "node": "aprovacao", "tool": "resolve_incident", "args": { "id": "INC-42" } },
    { "type": "observation", "at": 1, "node": "aprovacao", "result": { "id": "INC-42", "status": "resolved" } },
    { "type": "answer", "at": 2, "node": "aprovacao", "content": "Incidente INC-42 resolvido." }
  ],
  "route": null,
  "metrics": null,
  "conversationId": "string",
  "approval": { "id": "uuid", "status": "approved" }
}
```

- **Negar**: `trace` tem só o `answer` ("Ação cancelada: resolver o incidente INC-42. Nada foi executado."), e `approval.status` é `"denied"`.
- **Aprovar com erro de domínio** (por exemplo, incidente inexistente): a resposta continua 200 e `approval.status` é `"approved"`. A `observation` traz `{"error":"IncidentNotFoundError","id":"INC-42"}` e o `answer` explica a falha ("Não foi possível resolver o incidente INC-42: o incidente não existe.").
- O histórico da conversa recebe `assistant: <answer>`.
- A decisão é gravada como uma requisição nova (014), com `outcome: "ok"` e trace `node: "aprovacao"`.

### Erros

| Status | Corpo | Quando |
|---|---|---|
| 400 | `{ requestId, error: "invalid_body", issues }` | `decision` ausente ou inválido |
| 404 | `{ requestId, error: "approval_not_found", approvalId }` | `:id` não é UUID ou não existe |
| 409 | `{ requestId, error: "approval_already_decided", approvalId, status }` | já foi aprovada ou negada |
| 410 | `{ requestId, error: "approval_expired", approvalId, expiresAt }` | passou de `expiresAt` |
| 500 | `{ requestId, error: "internal_error" }` | falha de infraestrutura |

## `GET /stats` (não muda)

A engrenagem usa `GET /stats?since=1h` como teste de conexão. 200 significa conectado.

## Logs (014)

Há dois eventos novos, só com metadados: `approval.requested` (`requestId`, `approvalId`, `tool`) e `approval.decided` (`requestId`, `approvalId`, `decision`, `outcome: "executed" | "cancelled" | "failed"`). Eles nunca levam `args` nem `reason`.
