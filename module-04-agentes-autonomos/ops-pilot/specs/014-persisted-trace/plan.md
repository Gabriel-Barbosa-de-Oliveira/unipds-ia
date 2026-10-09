# Implementation Plan: Trace Persistido e Logs Estruturados

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**:
> - **Sucesso é gravado no nó `resposta`.** Quem grava o registro e o trace e emite os logs de trace e o `request.completed` (com `node: "resposta"`) é o nó `resposta` do grafo (`answerNode` em `src/graph/production-graph.ts`), com `requestStore` e `logger` injetados via `ProductionGraphDeps`. O controller continua gravando timeout e erro de execução, que nunca chegam a esse nó.
> - **Requisição abandonada.** O `RequestContext.abandoned()` evita que um grafo que já deu timeout grave ou logue depois.
> - **`userId` no registro.** O registro ganhou `userId` (coluna `user_id`).
> - **`request.completed` sem `status`.** O evento não tem `status`, porque o grafo não conhece HTTP.

**Branch**: `014-persisted-trace` | **Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/014-persisted-trace/spec.md`

## Summary

**Identificador.** Um middleware gera um UUID por requisição do `/chat`, que sai no cabeçalho `X-Request-Id` e no `requestId` de todo corpo de resposta, inclusive nos erros.

**Persistência.** Depois da execução, o controller grava no SQLite (`OPSPILOT_DB`) um registro em `requests` e o trace em `trace_events`. Cada evento tem posição, tipo, nó e o payload completo. A gravação é uma única transação e cobre sucesso, timeout e erro interno. Uma falha de gravação não afeta a resposta.

**Consulta.** `GET /requests/:id` devolve o registro e o trace na ordem original.

**Logs.** `src/obs/logger.ts` emite uma linha JSON por evento, com um tipo `LogEvent` fechado que só tem metadados. Os eventos de rota, troca de modelo e ferramenta são derivados do trace por uma função pura, sem `reason`, `args` nem conteúdo.

Os detalhes estão em [research.md](./research.md).

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS

**Primary Dependencies**: Express ^4.19, zod ^3.23, `node:sqlite` (`DatabaseSync`), `node:crypto` (`randomUUID`). Nenhuma dependência nova.

**Storage**: SQLite local via `OPSPILOT_DB` (padrão `./data/opspilot.db`), com duas tabelas novas, `requests` e `trace_events` ([data-model.md](./data-model.md)). Os testes usam `:memory:`.

**Testing**: `node:test` via `tsx`. O store roda de verdade em `:memory:`, o logger coleta linhas num array, e não há rede.

**Target Platform**: servidor Node (API Express)

**Project Type**: web-service

**Performance Goals**: a gravação síncrona leva poucos milissegundos, menos de 5% da latência típica (SC-006)

**Constraints**: nenhum conteúdo de conversa nos logs (FR-011), garantido por tipo e por teste-âncora. O `.env` não é lido.

**Scale/Scope**: 4 módulos novos (`request-record.ts`, `request-store.repository.ts`, `sqlite-request-store.ts`, `logger.ts`), 1 rota nova, `server.ts` alterado

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Status | Como |
|---|---|---|
| I. Camadas explícitas | ✅ | O repositório (`services/request-store.repository.ts`) tem um adaptador SQLite em `store/`. O registro é montado no domínio (`domain/request-record.ts`, puro). O controller orquestra. |
| II. Validação na fronteira | ✅ | O `:id` do `GET /requests/:id` é validado com zod (`uuid`). O corpo do `/chat` continua validado como hoje. |
| III. Erros de domínio | ✅ | Não há erro de domínio novo. "Não encontrado" vira `undefined` no repositório e 404 na borda. O `errorType` é o `name` das classes de erro que já existem. |
| IV. Funções puras | ✅ | `buildRequestRecord`, `toStoredTraceEvents`, `restoreTrace`, `formatLogLine` e `traceToLogEvents` são puras. O IO fica no store e no `write` do logger. |
| V. Teste obrigatório | ✅ | 4 suítes novas mais a extensão do `server.test.ts`, incluindo o teste-âncora de privacidade. |
| VI. Segurança | ✅ | Os logs só têm metadados (tipo fechado). O `console.error` com o objeto de erro inteiro é removido. O id é gerado pelo servidor, e o enviado pelo cliente é ignorado, para evitar sobrescrita. |
| VII. Spec antes de código | ✅ | spec → plan → tasks. |
| VIII. Pequeno e reversível | ✅ | Os incrementos são: domínio puro → store → logger → id no `/chat` → persistência → `GET` → logs. |
| Stack | ✅ | SQLite via `node:sqlite`, Express e zod. Nenhuma lib de log nova. |

**Re-check pós-design**: ✅ sem violações.

## Project Structure

### Documentation (this feature)

```text
specs/014-persisted-trace/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── http.md
│   └── logger.md
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```text
src/
├── domain/
│   ├── request-record.ts            # NOVO: RequestRecord, buildRequestRecord, toStoredTraceEvents, restoreTrace (puras)
│   └── request-record.test.ts       # NOVO
├── services/
│   └── request-store.repository.ts  # NOVO: interface RequestStore
├── store/
│   ├── sqlite-request-store.ts      # NOVO: requests + trace_events, transação, :memory: em testes
│   └── sqlite-request-store.test.ts # NOVO
├── obs/
│   ├── logger.ts                    # NOVO: LogEvent (união fechada), formatLogLine, traceToLogEvents, createLogger
│   └── logger.test.ts               # NOVO
└── http/
    ├── server.ts                    # ALTERADO: middleware de requestId, gravação, GET /requests/:id, logs; remove console.error
    └── server.test.ts               # ALTERADO: createTestApp com store :memory: e logger silencioso; testes novos
```

**Structure Decision**: o projeto segue as camadas que já existem (domain → services → store → http). `src/obs/` é um diretório novo, só para observabilidade, conforme o pedido.

## Complexity Tracking

Sem violações.
