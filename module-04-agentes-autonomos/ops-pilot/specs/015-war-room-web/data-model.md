# Data Model: War Room Web

O modelo tem duas metades: o que a API passa a guardar (ação pendente) e o estado que a war room mantém no navegador. Os formatos HTTP estão em [contracts/http.md](./contracts/http.md).

## API

### PendingAction (nova tabela `pending_actions`)

É uma ação que muda a produção e que o agente propôs durante um `/chat`. Ela espera decisão humana.

| Campo | Tipo (domínio) | Coluna SQLite | Regras |
|---|---|---|---|
| `id` | `string` (UUID) | `id TEXT PRIMARY KEY` | Gerado pelo servidor (`randomUUID`) |
| `requestId` | `string` | `request_id TEXT NOT NULL` | Execução do `/chat` que propôs a ação (014) |
| `conversationId` | `string` | `conversation_id TEXT NOT NULL` | A decisão é anexada a esta conversa |
| `userId` | `string \| null` | `user_id TEXT` | |
| `tool` | `GatedToolName` = `"open_incident" \| "resolve_incident"` | `tool TEXT NOT NULL CHECK (tool IN ('open_incident','resolve_incident'))` | |
| `args` | `Record<string, unknown>` | `args_json TEXT NOT NULL` | Já validado pelo schema da ferramenta. É validado de novo antes de executar |
| `reason` | `string \| null` | `reason TEXT` | Último `thought` do trace antes da ação, se houver |
| `status` | `"pending" \| "approved" \| "denied"` | `status TEXT NOT NULL CHECK (status IN ('pending','approved','denied'))` | `expired` é derivado, nunca gravado |
| `createdAt` | ISO string | `created_at TEXT NOT NULL` | |
| `expiresAt` | ISO string | `expires_at TEXT NOT NULL` | `createdAt + approvalTtlMs` (padrão 15 min) |
| `decidedAt` | ISO string \| null | `decided_at TEXT` | |
| `decisionRequestId` | `string \| null` | `decision_request_id TEXT` | `requestId` da requisição de decisão |

Índice: `idx_pending_actions_conversation ON pending_actions(conversation_id)`.

**Estado efetivo** (função pura `effectiveStatus(action, now)`): `pending` com `expiresAt <= now` vira `expired`. Os demais estados não mudam.

**Transições**:

```text
pending ──aprovar (antes de expirar)──▶ approved   (executa a ferramenta)
pending ──negar   (antes de expirar)──▶ denied     (nada executa)
pending ──tempo passa─────────────────▶ expired    (derivado; nenhuma decisão aceita)
approved | denied | expired ── qualquer decisão ──▶ erro (409 / 410)
```

A transição é gravada com `UPDATE pending_actions SET status=?, decided_at=?, decision_request_id=? WHERE id=? AND status='pending' AND expires_at > ?`. Se nenhuma linha for afetada, a linha é relida e o servidor devolve `ApprovalNotFoundError` (404), `ApprovalAlreadyDecidedError` (409) ou `ApprovalExpiredError` (410).

### ApprovalGate (em memória, por requisição)

`{ proposed?: { tool: GatedToolName; args: Record<string, unknown> } }`. É preenchido pela primeira chamada com porta. As chamadas seguintes recebem `rejected`. Ele vive só durante uma requisição do `/chat`.

### Erros de domínio novos (`src/domain/errors.ts`)

| Classe | Campos | HTTP (na borda) |
|---|---|---|
| `ApprovalNotFoundError` | `id` | 404 `approval_not_found` |
| `ApprovalAlreadyDecidedError` | `id`, `status` | 409 `approval_already_decided` |
| `ApprovalExpiredError` | `id`, `expiresAt` | 410 `approval_expired` |

### Mudanças em tipos existentes

- `GraphNode` (`src/agents/types.ts`) ganha `"aprovacao"`. Não precisa de migração, porque `trace_events.node` não tem CHECK.
- `RequestRecord` não muda. A requisição de decisão é gravada com `outcome: "ok"`, sem `route` e sem `metrics`. Os dois já são opcionais em `buildRequestRecord`.

## Web (`web/src/lib/`)

Todo o estado abaixo é derivado das respostas da API validadas com zod ([contracts/http.md](./contracts/http.md)) e é manipulado por reducers puros.

### Settings

| Campo | Tipo | Padrão | Validação | Guardado em |
|---|---|---|---|---|
| `apiUrl` | `string` | `http://localhost:3000` | URL absoluta `http:`/`https:`. A barra final é removida na normalização | `localStorage["opspilot.apiUrl"]` |
| `theme` | `"system" \| "light" \| "dark"` | `"system"` | enum | `localStorage["opspilot.theme"]` |

### Conversation

`{ conversationId: string | null; items: ConversationItem[]; pending: "idle" | "sending" | "awaiting_decision" }`

- Começa com `conversationId = null`. O `conversationId` da primeira resposta 200/202 é adotado, e a partir daí ele é sempre enviado.
- "Nova conversa" volta ao estado inicial.
- `pending = "awaiting_decision"` enquanto existe um cartão com estado `pending`. Nesse estado o envio fica bloqueado (FR-018).

### ConversationItem (união discriminada)

| `kind` | Campos | Estados |
|---|---|---|
| `user` | `id`, `text`, `sentAt` | `sending → delivered \| failed` |
| `assistant` | `id`, `run: ChatRun` | |
| `approval` | `id`, `approval: Approval`, `run: ChatRun` (a execução que gerou o 202) | ver ApprovalCard |
| `error` | `id`, `error: UiError`, `retryText` | |

### ChatRun

`{ requestId; answer; trace: TraceEvent[]; route: RouteDecision | null; metrics: Metrics | null }`. É o que o "ver raciocínio" abre. `route` e `metrics` são `null` nas respostas de decisão.

### TraceEvent (espelho tipado de `src/agents/types.ts`)

É uma união discriminada por `type`, com `at: number` e `node?: GraphNode | string`:

| `type` | Campos | Exibição (`toTraceView`) |
|---|---|---|
| `route` | `route`, `reason`, `source` | Rótulo "Rota", ícone de bifurcação, `route` + motivo + origem |
| `thought` | `content` | "Pensamento", ícone de balão |
| `plan` | `steps: string[]` | "Plano", lista numerada |
| `action` | `tool`, `args` | "Ação", nome da ferramenta + argumentos (recolhível) |
| `observation` | `result: unknown` | "Observação", resultado formatado (recolhível acima de ~600 caracteres) |
| `critique` | `content` | "Crítica" |
| `fallback` | `from`, `to`, `reason` | "Troca de modelo", `from → to` |
| `answer` | `content` | "Resposta" |
| *(desconhecido)* | `type: string` + resto | "Evento: `<type>`", JSON bruto legível (FR-011) |

### ApprovalCard (máquina de estados pura, `approval-machine.ts`)

```text
pending ──clicar Aprovar/Negar──▶ submitting(decision)
submitting ──200──▶ approved | denied        (anexa item assistant com a resposta)
submitting ──409/410/404──▶ unavailable      (mostra motivo; não aceita clique)
submitting ──rede/5xx──▶ pending + error     (permite tentar de novo)
approved | denied | unavailable ── clique ──▶ (ignorado)
```

### UiError (tradução pura em `errors.ts`)

`{ title: string; detail: string; requestId?: string; action: "retry" | "open_settings" | "new_conversation" | "none" }`

| Origem | `title` | `action` |
|---|---|---|
| 400 `invalid_body` | "Não deu para enviar essa mensagem" | `retry` |
| 404 `conversation_not_found` | "Essa conversa não existe mais" | `new_conversation` |
| 422 `unknown_strategy` | "Estratégia inválida" | `none` |
| 504 `timeout` | "O copiloto demorou demais para responder" | `retry` |
| 500 `internal_error` / outro 5xx | "O copiloto falhou ao responder" | `retry` |
| `TypeError` de rede / CORS | "Não foi possível falar com a API" (o texto sugere conferir o endereço e as origens permitidas) | `open_settings` |
| Abort local (190s) | igual a `timeout` | `retry` |
| Resposta fora do schema | "A API respondeu num formato inesperado" | `retry` |
