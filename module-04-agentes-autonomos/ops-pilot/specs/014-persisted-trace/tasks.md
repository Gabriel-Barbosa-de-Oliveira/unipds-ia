---

description: "Task list for 014-persisted-trace"
---

# Tasks: Trace Persistido e Logs Estruturados

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**:
> - **Sucesso é gravado no nó `resposta`.** Quem grava o registro e o trace e emite os logs de trace e o `request.completed` (com `node: "resposta"`) é o nó `resposta` do grafo (`answerNode` em `src/graph/production-graph.ts`), com `requestStore` e `logger` injetados via `ProductionGraphDeps`. O controller continua gravando timeout e erro de execução, que nunca chegam a esse nó.
> - **Requisição abandonada.** O `RequestContext.abandoned()` evita que um grafo que já deu timeout grave ou logue depois.
> - **`userId` no registro.** O registro ganhou `userId` (coluna `user_id`).
> - **`request.completed` sem `status`.** O evento não tem `status`, porque o grafo não conhece HTTP.

**Input**: Design documents from `/specs/014-persisted-trace/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos. A constitution (Princípio V, NON-NEGOTIABLE) exige teste para toda lógica nova, e a FR-013 pede testes sem rede para montagem do registro, seleção de metadados de log e ordenação do trace.

**Organization**: as tarefas estão agrupadas por user story (spec.md), para que cada uma possa ser implementada e testada de forma independente.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivo diferente e sem dependência de tarefa ainda não concluída).
- **[Story]**: a user story à qual a tarefa pertence (US1–US3).

## Path Conventions

Projeto único, com `src/` na raiz. Os testes ficam ao lado do código (`*.test.ts`).

---

## Phase 1: Setup

**Purpose**: inicialização do projeto.

Fase vazia: nenhuma dependência nova (`node:sqlite`, `node:crypto`, Express e zod já estão em uso).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: o identificador por requisição, que o registro (US1), a correlação (US2) e os logs (US3) usam, e a preparação da suíte HTTP.

**⚠️ CRITICAL**: nenhuma user story começa antes desta fase terminar.

- [X] T001 Em `src/http/server.ts`:
  - criar a função `assignRequestId(req, res, next)`, que gera `crypto.randomUUID()`, guarda em `res.locals.requestId` e chama `res.setHeader("X-Request-Id", id)`. O `X-Request-Id` recebido do cliente é ignorado (research.md item 1);
  - registrar com `app.post("/chat", assignRequestId, handler)`.

  Ainda sem mudar os corpos de resposta.
- [X] T002 Em `src/http/server.test.ts`:
  - `createTestApp` passa a injetar por padrão as opções que as fases seguintes criam (`requestStore: new SqliteRequestStore(":memory:")` e `logger: silentLogger()`). Enquanto elas não existem, deixar só o comentário do ponto de extensão;
  - adicionar o teste: `POST /chat` 200 → `response.headers.get("x-request-id")` é um UUID v4;
  - dois `POST` concorrentes (`Promise.all`) → ids diferentes (FR-002).

**Checkpoint**: todo `/chat` tem `X-Request-Id`, e a suíte está verde.

---

## Phase 3: User Story 1 - Reabrir depois o que aconteceu numa resposta (Priority: P1) 🎯 MVP

**Goal**: o registro e o trace de cada execução são gravados no SQLite e consultáveis em `GET /requests/:id`.

**Independent Test**: um `POST /chat` seguido de `GET /requests/<X-Request-Id>` devolve as métricas e um trace deep-equal ao da resposta original. Isso vale também com uma nova instância do store sobre o mesmo arquivo.

### Tests for User Story 1

- [X] T003 [P] [US1] Criar `src/domain/request-record.test.ts` (deve falhar até a T005):
  - `buildRequestRecord` com `outcome: "ok"`, `route` e `metrics` (incluindo `contextBreakdown` e `contextTrimmed`) → todas as colunas preenchidas, `errorType: null` e `context: { breakdown, trimmed }`;
  - com `outcome: "timeout"`, `errorType: "ChatTimeoutError"` e sem métricas → métricas e rota `null`;
  - com `outcome: "error"` → `errorType` igual ao informado;
  - `toStoredTraceEvents(trace)` → `position` igual a `at`, `type`, `node` (ou `null`) e `payload` igual ao evento original;
  - `restoreTrace(linhas embaralhadas)` → ordena por `position` e devolve um trace deep-equal ao original (FR-007).
- [X] T004 [P] [US1] Criar `src/store/sqlite-request-store.test.ts` (deve falhar até a T007), seguindo o estilo de `src/store/sqlite-conversation-store.test.ts`:
  - em `:memory:`, `save(record, trace)` seguido de `find(id)` devolve `{ request: record, trace }` deep-equal ao gravado, com trace de vários tipos (`route`, `fallback`, `action`, `observation`, `answer`, com `node`);
  - `find` de um id inexistente → `undefined`;
  - registro de timeout com trace vazio → `trace: []`;
  - persistência entre instâncias: arquivo temporário em `os.tmpdir()`, `new SqliteRequestStore(path)`, depois `save`, nova instância e `find`, com remoção do arquivo no `after` (SC-002);
  - duas instâncias sobre o mesmo arquivo não falham (DDL idempotente);
  - `CHECK` de `outcome` rejeita valor fora do domínio via SQL direto, como o teste equivalente de `sqlite-ops-store.test.ts`.

### Implementation for User Story 1

- [X] T005 [US1] Criar `src/domain/request-record.ts` (puro, sem IO), conforme [data-model.md](./data-model.md):
  - `export type RequestOutcome = "ok" | "timeout" | "error"`;
  - `export interface RequestRecord { … }`;
  - `export function buildRequestRecord(input: { requestId; conversationId: string | null; startedAt: Date; durationMs: number; outcome; errorType?: string; route?: RouteDecision; metrics?: ChatMetrics }): RequestRecord`, onde `ChatMetrics = Metrics & { historyMessages; contextBreakdown; contextTrimmed }`;
  - `export interface StoredTraceEvent { position: number; type: TraceEvent["type"]; node: GraphNode | null; payload: TraceEvent }`;
  - `export function toStoredTraceEvents(trace)`;
  - `export function restoreTrace(rows)`, que ordena uma cópia, sem mutar a entrada.

  Rodar a T003 até ficar verde.
- [X] T006 [P] [US1] Criar `src/services/request-store.repository.ts` com a interface `RequestStore { save(record, trace): Promise<void>; find(requestId): Promise<{ request: RequestRecord; trace: TraceEvent[] } | undefined> }` e um JSDoc no padrão de `conversation-store.repository.ts`.
- [X] T007 [US1] (depende de T005, T006) Criar `src/store/sqlite-request-store.ts`:
  - `export class SqliteRequestStore implements RequestStore`, com o mesmo padrão de `SqliteConversationStore`: `constructor(path = process.env.OPSPILOT_DB ?? "./data/opspilot.db")`, conexão lazy e o DDL do data-model;
  - `save` faz `BEGIN`, o `INSERT` em `requests` (com `context_json = JSON.stringify(record.context)` ou `NULL`), um `INSERT` por evento de `toStoredTraceEvents(trace)` (`payload_json = JSON.stringify(payload)`) e `COMMIT`, com `ROLLBACK` e relançamento em caso de erro;
  - `find` faz `SELECT` do registro e, se ele existir, `SELECT position, type, node, payload_json … WHERE request_id = ? ORDER BY position`, depois `restoreTrace`, e mapeia snake_case para camelCase.

  Rodar a T004 até ficar verde.
- [X] T008 [US1] (depende de T001, T007) Em `src/http/server.ts`:
  - `CreateAppOptions` ganha `requestStore?: RequestStore` (padrão `new SqliteRequestStore()`) e `now?: () => Date` (padrão `() => new Date()`);
  - no handler, `startedAt = now()` no início;
  - **depois** do grafo, e no `catch` para `ChatTimeoutError` e erros inesperados (**não** para `UnknownStrategyError`, `ConversationNotFoundError` nem 400), montar `buildRequestRecord(...)` e chamar `await persist(record, trace)`;
  - `persist` envolve `requestStore.save` em `try/catch` e nunca relança (FR-008). O log de falha entra na US3; por enquanto o `catch` fica vazio, com um comentário apontando a T018;
  - no sucesso, o trace é o `trace` do resultado e o desfecho é `ok`; no timeout e no erro, o trace é `[]` com `errorType = error.name`.

  Reestruturar o `try/catch` do handler para classificar os erros antes de chamar `next(error)`.
- [X] T009 [US1] (depende de T008) Em `src/http/server.ts`, criar `app.get("/requests/:id", …)`:
  - validar com `z.string().uuid().safeParse(req.params.id)`; inválido → `404 { error: "request_not_found", requestId: req.params.id }`;
  - `requestStore.find(id)`; `undefined` → o mesmo 404; encontrado → `200 { request, trace }` ([contracts/http.md](./contracts/http.md)).
- [X] T010 [US1] (depende de T009) Em `src/http/server.test.ts`:
  - o `createTestApp` passa a injetar `requestStore: new SqliteRequestStore(":memory:")` (um por app);
  - nova describe "User Story 1 (014) — trace persistido", com estes casos:
    - **(a)** `POST /chat` 200 e depois `GET /requests/<x-request-id>` → 200, `body.trace` deep-equal ao `trace` do POST, `request.outcome === "ok"`, `request.route === body.route.route`, `request.modelUsed === body.metrics.modelUsed` e `request.conversationId === body.conversationId`;
    - **(b)** override (`strategy: "plan-and-execute"`) → `request.routeSource === "override"` (FR-012);
    - **(c)** timeout (estratégia que nunca resolve, `timeoutMs: 20`) → 504, e o `GET` devolve `outcome: "timeout"`, `errorType: "ChatTimeoutError"` e `trace: []`;
    - **(d)** estratégia que lança `new Error("boom")` → 500, e o registro tem `outcome: "error"` e `errorType: "Error"`;
    - **(e)** `GET /requests/<uuid aleatório>` → 404 `request_not_found`;
    - **(f)** `GET /requests/nao-e-uuid` → 404;
    - **(g)** 422 de estratégia desconhecida → o `GET` com o id do cabeçalho dá 404 (não é persistido);
    - **(h)** um `requestStore` fake cujo `save` lança → a resposta 200 continua idêntica (FR-008).

**Checkpoint**: toda execução do `/chat` pode ser reaberta pelo id. Esse é o MVP.

---

## Phase 4: User Story 2 - Correlacionar resposta, registro e logs (Priority: P1)

**Goal**: `requestId` no corpo de **toda** resposta do `/chat`, igual ao cabeçalho.

**Independent Test**: para os casos 200, 400, 404, 422, 504 e 500, `body.requestId === headers["x-request-id"]`.

### Tests for User Story 2

- [X] T011 [P] [US2] (depende de T002) Em `src/http/server.test.ts`, nova describe "User Story 2 (014) — requestId em toda resposta", parametrizada pelos 6 cenários:
  - 200;
  - 400 sem `message`;
  - 404 com `conversationId` desconhecido;
  - 422 com `strategy` inválida;
  - 504 com `timeoutMs` baixo;
  - 500 com uma estratégia que lança.

  Em todos, `body.requestId` é um UUID igual ao cabeçalho `x-request-id`.

### Implementation for User Story 2

- [X] T012 [US2] (depende de T001, T011) Em `src/http/server.ts`:
  - a resposta 200 ganha `requestId: res.locals.requestId` como **primeiro** campo;
  - o ramo 400 (`invalid_body`) inclui `requestId`;
  - o `errorMiddleware` inclui `requestId: res.locals.requestId` em todos os corpos (404, 422, 504 e 500).

  Ajustar os asserts `deepEqual` de corpos de erro que já existem no `server.test.ts` (ex.: o 422 `{ error, strategy }` da describe 012 override) para incluir `requestId: response.headers.get("x-request-id")`.

**Checkpoint**: resposta, registro e (na US3) logs compartilham o mesmo id.

---

## Phase 5: User Story 3 - Logs JSON, uma linha por evento, só metadados (Priority: P2)

**Goal**: `src/obs/logger.ts`, com um tipo fechado de eventos, ligado ao `/chat` e ao `GET /requests/:id`. Conteúdo da conversa nunca entra nos logs.

**Independent Test**: capturar as linhas de uma requisição. Todas passam por `JSON.parse`, todas têm o mesmo `requestId`, e o marcador `MARCADOR-SECRETO-123` não aparece em nenhuma.

### Tests for User Story 3

- [X] T013 [P] [US3] Criar `src/obs/logger.test.ts` (deve falhar até a T015):
  - `formatLogLine(event, new Date("2026-10-09T00:00:00.000Z"))`: `JSON.parse` funciona, não contém `\n`, `ts` é o ISO informado, `level` está correto por tipo (tabela do data-model) e `event` é o tipo;
  - `traceToLogEvents("id", trace)` com `route` (com `reason`), `fallback` (com `reason`), `action` (com `args`), `thought`, `observation`, `critique` e `answer`. O resultado tem só `route.chosen`, `model.fallback` e `tool.called`, nessa ordem, com `node` e `position`, e **sem** as chaves `reason`, `args`, `content` e `result`;
  - `createLogger(write, now)`: cada `log(event)` chama `write` uma vez, com a linha formatada.
- [X] T014 [P] [US3] (depende de T010) Em `src/http/server.test.ts`, nova describe "User Story 3 (014) — logs JSON":
  - **(a)** com um logger coletor, um `POST /chat` 200 emite, em ordem, `request.received`, os eventos derivados do trace (`route.chosen` e um `tool.called` vindo da fake) e `request.completed`, todos com o mesmo `requestId` do cabeçalho; e `request.completed` traz `llmCalls`, `modelUsed` e `traceEvents` iguais aos da resposta;
  - **(b)** 400 → `request.rejected` com `errorCode: "invalid_body"`; 422 → `errorCode: "unknown_strategy"`;
  - **(c)** 500 → `request.failed` com `errorType` e **sem** a mensagem do erro;
  - **(d)** um `save` que lança → `persistence.failed`;
  - **(e) teste-âncora (SC-004)**:
    - a mensagem do usuário contém `MARCADOR-SECRETO-123`;
    - a fake de estratégia devolve `answer`, `thought`, `action.args`, `observation.result` e um `fallback.reason` com o marcador;
    - o roteador fake devolve `reason` com o marcador;
    - **nenhuma** linha coletada contém o marcador.

### Implementation for User Story 3

- [X] T015 [US3] (depende de T013) Criar `src/obs/logger.ts` conforme [contracts/logger.md](./contracts/logger.md) e data-model.md § LogEvent:
  - `export type LogEvent`, uma união discriminada por `event`, com exatamente os campos da tabela;
  - `export function formatLogLine(event, now)`, puro, que calcula o `level` por um mapa fixo `event → level`;
  - `export function traceToLogEvents(requestId, trace)`, puro, montando **explicitamente** cada evento campo a campo (nunca com spread do evento de trace);
  - `export interface Logger { log(event: LogEvent): void }`;
  - `export function createLogger(write = (line) => process.stdout.write(`${line}\n`), now = () => new Date()): Logger`.
- [X] T016 [US3] (depende de T015, T012) Em `src/http/server.ts`:
  - `CreateAppOptions` ganha `logger?: Logger` (padrão `createLogger()`);
  - no `/chat`:
    - `request.received` no início, com `hasConversationId`, `hasUserId` e `strategyOverride` (o nome da rota se o override for válido, senão `null`);
    - no sucesso, `traceToLogEvents(requestId, result.trace)` seguido de `request.completed`, com `durationMs = now() - startedAt`;
  - no ramo 400 e no `errorMiddleware`:
    - 404 e 422 → `request.rejected`;
    - 504 e 500 → `request.failed`, com `errorType` igual a `error.name` (ou `"Error"`);
  - **remover** o `console.error("Erro inesperado no /chat:", error)`;
  - `persist` registra `persistence.failed` no `catch`, completando a T008;
  - `GET /requests/:id` registra `request.lookup` com `{ requestId, found }`.
- [X] T017 [US3] (depende de T016) Em `src/http/server.test.ts`, o `createTestApp` passa a injetar `logger: { log() {} }` por padrão (silencioso), e as describes da T014 sobrescrevem com o coletor. Rodar a T014 até ficar verde.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T018 [P] Em `specs/003-chat-endpoint/quickstart.md`, acrescentar uma nota curta sobre a 014: `X-Request-Id`/`requestId` em toda resposta, `GET /requests/:id` e logs JSON no stdout (sem conteúdo de conversa). O "Troubleshooting 500" passa a indicar o log `request.failed` e a consulta `GET /requests/<id>`, em vez do `console.error`.
- [X] T019 [P] Conferir que `.gitignore` já cobre `data/`, onde fica o SQLite com os registros. Já cobre; não mudar.
- [X] T020 Rodar `npm run typecheck` e `npm test`; os dois precisam ficar verdes (Princípio V).
- [ ] T021 Validar manualmente os cenários 2–4 de [quickstart.md](./quickstart.md). Isso exige `OPENROUTER_API_KEY` no ambiente, sem nunca ler `.env`. **Pendente**: `OPENROUTER_API_KEY` não está configurada no ambiente desta execução.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup**: vazia.
- **Foundational**: T001 → T002. Bloqueia as stories.
- **US1**: a T003 roda em paralelo com a T004 e a T006; T005 → T007 → T008 → T009 → T010.
- **US2**: a T011 pode começar logo depois da Foundational; a T012 depende da T001 e da T011 (e mexe em `server.ts`, então deve vir depois da T009 se rodar junto com a US1).
- **US3**: a T013 pode começar a qualquer momento; T015 → T016 (depois da T012) → T017; a T014 depende da T010.
- **Polish**: depois de todas as stories.

### User Story Dependencies

- **US1 (P1)**: depende só da Foundational. Entrega o MVP.
- **US2 (P1)**: depende só da Foundational. Conflita em arquivo com a US1 (`server.ts`), então as tarefas de implementação rodam em sequência.
- **US3 (P2)**: usa o `requestId` (Foundational) e a persistência (US1) para o `persistence.failed`. A parte pura (T013, T015) é independente.

## Parallel Opportunities

- **US1**: T003 ∥ T004 ∥ T006.
- **Transversal**: T013 e T015 (logger puro) ∥ toda a US1.
- **US2**: T011 ∥ US1 (só testes).
- **Polish**: T018 ∥ T019.

### Parallel Example: User Story 1

```bash
Task: "T003 [US1] testes de request-record em src/domain/request-record.test.ts"
Task: "T004 [US1] testes do SqliteRequestStore em src/store/sqlite-request-store.test.ts"
Task: "T006 [US1] interface RequestStore em src/services/request-store.repository.ts"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Foundational (T001–T002).
2. US1 (T003–T010): registro e trace persistidos e consultáveis.
3. **STOP and VALIDATE**: `POST` → `GET` com trace idêntico; timeout e erro registrados; 404 para id desconhecido.

### Incremental Delivery

Foundational → US1 (persistência e consulta) → US2 (`requestId` em todo corpo) → US3 (logs JSON) → Polish. Cada passo deixa `typecheck` e `test` verdes.
