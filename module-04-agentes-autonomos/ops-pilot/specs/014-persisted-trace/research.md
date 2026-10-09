# Research: Trace Persistido e Logs Estruturados

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**:
> - **Sucesso é gravado no nó `resposta`.** Quem grava o registro e o trace e emite os logs de trace e o `request.completed` (com `node: "resposta"`) é o nó `resposta` do grafo (`answerNode` em `src/graph/production-graph.ts`), com `requestStore` e `logger` injetados via `ProductionGraphDeps`. O controller continua gravando timeout e erro de execução, que nunca chegam a esse nó.
> - **Requisição abandonada.** O `RequestContext.abandoned()` evita que um grafo que já deu timeout grave ou logue depois.
> - **`userId` no registro.** O registro ganhou `userId` (coluna `user_id`).
> - **`request.completed` sem `status`.** O evento não tem `status`, porque o grafo não conhece HTTP.

Base: o `/chat` em `src/http/server.ts` (012/013), `SqliteConversationStore` (padrão de persistência com `DatabaseSync`, conexão lazy e DDL idempotente) e `ConversationStore` (padrão de repositório em `src/services/`). Não ficou nenhum NEEDS CLARIFICATION em aberto.

## 1. Identificador da requisição

- **Decision**: um middleware Express aplicado só às rotas do chat e da consulta gera `crypto.randomUUID()` por requisição. Ele guarda o valor em `res.locals.requestId`, define o cabeçalho `X-Request-Id` **antes** de qualquer resposta e inclui `requestId` em todo corpo JSON do `/chat`: 200, 400, 404, 422, 504 e 500. O `errorMiddleware` e o ramo 400 de validação passam a incluir o campo. Um `X-Request-Id` recebido do cliente é ignorado (premissa da spec).
- **Rationale**: o cabeçalho fica no lugar certo (antes de qualquer `res.json`), não depende de cada ramo lembrar de defini-lo e cobre também os erros (FR-001).
- **Alternatives considered**: `AsyncLocalStorage` para propagar o id até o logger. Não é necessário, porque o controller já tem o id em mãos e os logs de trace são derivados no fim (item 5). Fica para o futuro.

## 2. Persistência: tabelas `requests` e `trace_events`

- **Decision**: a nova classe `SqliteRequestStore` (`src/store/sqlite-request-store.ts`) implementa `RequestStore` (`src/services/request-store.repository.ts`). O padrão é o mesmo do `SqliteConversationStore`: caminho `OPSPILOT_DB`, conexão lazy, DDL `CREATE TABLE IF NOT EXISTS` com `CHECK` nos domínios fechados e `:memory:` nos testes.
  - **`requests`**: as colunas tipadas são as que dão para filtrar e agregar (desfecho, rota, `llm_calls`, `prompt_tokens`, `model_used`, `duration_ms`). As métricas de contexto (`contextBreakdown`, `contextTrimmed`) vão como JSON numa coluna `context_json`, porque o formato delas já mudou em várias features.
  - **`trace_events`**: uma linha por evento, com `position` (o `at`), `type` e `node` em colunas e o evento completo em `payload_json`. `UNIQUE(request_id, position)`.
  - **Gravação**: `save(record, trace)` grava as duas tabelas numa única transação (`BEGIN` … `COMMIT`, com `ROLLBACK` no erro).
- **Rationale**: as colunas tipadas permitem consultas futuras ("quantos fallbacks hoje?"). O payload JSON completo garante FR-007 (trace idêntico) sem mapear cada variante de `TraceEvent`, e isso sobrevive a variantes novas, como as das features 012 e 013.
- **Alternatives considered**:
  - uma tabela só, com o trace inteiro em JSON. É mais simples, mas perde a consulta por tipo e por nó e não atende ao pedido explícito ("trace_events (node, payloads)");
  - colunas por campo de cada variante. São frágeis a cada evento novo.

## 3. Quando e como gravar

- **Decision**: o controller grava **depois** de ter o resultado, e **antes** de responder:
  - **sucesso**: o registro vai com `outcome: "ok"`, métricas, rota e trace;
  - **`ChatTimeoutError`**: `outcome: "timeout"`, sem métricas e com trace vazio;
  - **erro inesperado (500)**: `outcome: "error"` e `errorType = error.name`, também com trace vazio.

  Erros **antes** da execução não são gravados (premissa da spec): 400 de corpo inválido, 422 de estratégia desconhecida e 404 de conversa desconhecida.

  A gravação fica em `try/catch`. Se ela falhar, gera o log `persistence.failed` e a resposta segue normalmente (FR-008). A chamada usa `await`, porque o `DatabaseSync` é síncrono e leva poucos milissegundos (SC-006). Assim o `GET` logo depois do `POST` encontra o registro de forma determinística, sem corrida.
- **Rationale**: é simples e determinístico em teste, e cumpre FR-003, FR-004 e FR-008.
- **Alternatives considered**: gravar de forma fire-and-forget. Isso introduz corrida nos testes de consulta logo depois do chat e não ganha nada mensurável com SQLite local.

## 4. Montagem do registro (pura)

- **Decision**: `src/domain/request-record.ts` exporta funções puras:
  - `buildRequestRecord({ requestId, conversationId, startedAt, durationMs, outcome, route?, metrics?, errorType? })`, que devolve um `RequestRecord`;
  - `toStoredTraceEvents(trace)`, que devolve `{ position, type, node, payload }[]` com `position = event.at`;
  - `restoreTrace(rows)`, que ordena por `position` e devolve os payloads.

  O repositório SQLite só serializa.
- **Rationale**: Princípio IV, e FR-013 pede teste sem rede. Toda a lógica que decide o que vai para onde é testável sem banco.

## 5. Logger JSON (`src/obs/logger.ts`)

- **Decision**:
  - **Tipo do evento**: `LogEvent` é uma **união discriminada fechada**, cujos campos são só metadados (tabela no data-model). **Não existe campo livre** (`details`, `payload`, `message`), e por isso conteúdo de conversa não tem como entrar por tipo (FR-011). Um `errorType` carrega só `error.name`, nunca a `message`, que pode ecoar conteúdo.
  - **`formatLogLine(event, now)`**: função pura que devolve `JSON.stringify({ ts: now.toISOString(), level, event: type, ...campos })`, numa única linha.
  - **`createLogger(write = (line) => process.stdout.write(line + "\n"))`**: devolve `{ log(event) }`. O `write` é injetável nos testes.
  - **`traceToLogEvents(requestId, trace)`**: função pura que deriva os eventos de log do trace já produzido:
    - `route` vira `route.chosen` com `{ route, source, node }`, **sem** `reason`;
    - `fallback` vira `model.fallback` com `{ from, to, node }`, **sem** `reason`, porque a mensagem de erro do provedor pode ecoar o prompt;
    - `action` vira `tool.called` com `{ tool, node }`, **sem** `args`;
    - os demais tipos (thought, observation, plan, critique, answer) não geram log, porque são só conteúdo.
  - **O controller emite**:
    - `request.received`;
    - `request.rejected` para 400/404/422, com `status` e `errorCode`;
    - os eventos derivados do trace;
    - `request.completed` com `{ status, durationMs, llmCalls, promptTokens, tokenSource, modelUsed, traceEvents }`;
    - `request.failed` para 500/504, com `{ status, errorType }`;
    - `persistence.failed` com `{ errorType }`.
  - O `console.error("Erro inesperado no /chat:", error)` atual é **substituído** por `request.failed`. Ele imprimia o objeto de erro inteiro, que pode conter conteúdo. O stack completo deixa de ir para o log.
- **Rationale**: a garantia de "só metadados" vem do tipo e não da disciplina de quem chama. Um teste com texto marcador confirma (SC-004).
- **Trade-off conhecido**: os logs de rota, fallback e tool são emitidos **ao fim** da requisição (derivados do trace), não em tempo real. O `ts` é o da emissão; a ordem relativa é dada por `position`. Emitir em tempo real exigiria callbacks dentro do grafo e das estratégias, o que está fora do escopo.
- **Alternatives considered**: `pino`. Ele daria níveis e performance, mas é uma dependência nova (mudança de stack exige amendment) e não resolve sozinho a restrição de metadados.

## 6. `GET /requests/:id`

- **Decision**: valida o id com zod (`z.string().uuid()`). Um id inválido ou inexistente devolve `404 { error: "request_not_found", requestId }`. Um id encontrado devolve `200 { request: RequestRecord, trace: TraceEvent[] }`, com o trace na ordem de `position`. A consulta não ganha um `requestId` próprio (ela não é uma execução), mas gera o log `request.lookup` com `{ found }`.
- **Rationale**: FR-005 e FR-006, mais o edge case "formato inválido vira não encontrado" (não vaza a diferença).

## 7. Injeção e testes

- **Decision**: `CreateAppOptions` ganha `requestStore?: RequestStore` (padrão `new SqliteRequestStore()`), `logger?: Logger` (padrão `createLogger()`) e `now?: () => Date`. Os testes HTTP usam `new SqliteRequestStore(":memory:")` (o DDL é exercitado de verdade) e um logger que coleta as linhas num array. A suíte atual ganha um `requestStore` em memória e um logger silencioso no helper `createTestApp`, para não sujar a saída nem criar `./data/opspilot.db`.
