---

description: "Task list template for feature implementation"
---

# Tasks: Endpoint HTTP de Chat

**Input**: Design documents from `/specs/003-chat-endpoint/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: solicitados explicitamente pelo feature description ("Teste de integração com estrategia fake determinista, sem rede") — todas as tasks de teste abaixo são obrigatórias, não opcionais.

**Organization**: Tasks agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivos diferentes, sem dependência entre si)
- **[Story]**: A qual user story esta task pertence (US1, US2, US3, US4)
- Caminhos de arquivo exatos incluídos em cada descrição

## Path Conventions

Projeto único — `src/` na raiz do repositório, conforme [plan.md](./plan.md#project-structure).

---

## Phase 1: Setup

**Purpose**: Preparar o diretório da nova camada HTTP

- [X] T001 Create `src/http/` directory for the new HTTP layer (fica vazio até a Phase 2)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Infraestrutura sem valor de negócio isolado, mas necessária para qualquer user story

**⚠️ CRITICAL**: Nenhuma user story pode começar antes desta fase estar completa

- [X] T002 [P] Create `src/agents/index.ts` with the `STRATEGIES` registry (`react`, `plan-and-execute`) and `DEFAULT_STRATEGY_NAME = "react"` — apenas o registro estático, sem função de resolução ainda (per [contracts/strategy-registry.md](./contracts/strategy-registry.md))
- [X] T003 [P] Create `src/http/server.ts` with `createApp(options?: { resolveStrategy?, timeoutMs? }): Express`: instancia o app, `express.json()`, o schema zod `ChatRequestSchema` para `{ message, strategy?, reflect? }`, e um middleware de erro genérico que mapeia qualquer erro não tratado para `500 { error: "internal_error" }` (per [contracts/post-chat.md](./contracts/post-chat.md)) — ainda sem a rota `/chat`
- [X] T004 Wire `src/index.ts` (hoje vazio) to call `createApp().listen(PORT)`, lendo `PORT` do ambiente (padrão 3000) e logando ao subir — depends on T003

**Checkpoint**: `npm run dev` sobe o servidor; nenhuma rota responde ainda.

---

## Phase 3: User Story 1 - Obter uma resposta via API para uma pergunta operacional (Priority: P1) 🎯 MVP

**Goal**: `POST /chat` com apenas `message` retorna `200` com a resposta da estratégia padrão (react).

**Independent Test**: enviar `{ message }` (sem `strategy`/`reflect`) para uma instância `createApp({ resolveStrategy: <fake> })` com uma estratégia fake determinística e verificar `200 { answer, trace, metrics }`.

### Tests for User Story 1 ⚠️

> Escrever estas tests PRIMEIRO — devem falhar antes da implementação abaixo.

- [X] T005 [US1] Integration test in `src/http/server.test.ts`: `POST /chat` com apenas `message`, usando `createApp({ resolveStrategy: <fake determinística injetada> })`, verifica `200` e o corpo `{ answer, trace, metrics }` batendo com o `RunResult` da fake
- [X] T006 [US1] Integration test in `src/http/server.test.ts`: `POST /chat` com `message` ausente/vazio verifica `400 { error: "invalid_body", issues }`, sem que a estratégia fake seja chamada

### Implementation for User Story 1

- [X] T007 [US1] Add `resolveStrategy(name?: string): ReasoningStrategy` to `src/agents/index.ts`: retorna `STRATEGIES[name ?? DEFAULT_STRATEGY_NAME]` (tratamento de nome desconhecido chega na US2) — depends on T002
- [X] T008 [US1] Add the `POST /chat` route to `src/http/server.ts`: valida o corpo com `ChatRequestSchema` (400 se inválido), resolve a estratégia via `resolveStrategy` (real por padrão, ou a função injetada em `options.resolveStrategy`), chama `strategy.run(message)`, responde `200` com o `RunResult` sem transformação — depends on T003, T007

**Checkpoint**: User Story 1 completa e testável de forma independente — caminho feliz com estratégia padrão e rejeição de corpo inválido, sem rede.

---

## Phase 4: User Story 2 - Escolher explicitamente a estratégia de raciocínio (Priority: P2)

**Goal**: `strategy` explícito é respeitado; nome desconhecido retorna `422` sem executar nenhum raciocínio.

**Independent Test**: enviar a mesma pergunta duas vezes com `strategy` diferentes (fakes) e conferir que cada resposta reflete a estratégia pedida; enviar um nome inexistente e conferir `422` sem nenhuma chamada de execução.

### Tests for User Story 2 ⚠️

- [X] T009 [P] [US2] Unit test in `src/agents/index.test.ts`: `resolveStrategy("plan-and-execute")` retorna a estratégia correspondente; `resolveStrategy("nao-existe")` lança `UnknownStrategyError`
- [X] T010 [P] [US2] Integration test in `src/http/server.test.ts`: `POST /chat` com `strategy` explícito (fake distinta da padrão) usa essa estratégia; `POST /chat` com `strategy` desconhecido retorna `422 { error: "unknown_strategy", strategy }` sem que nenhuma estratégia fake seja invocada

### Implementation for User Story 2

- [X] T011 [US2] Add `UnknownStrategyError` to `src/domain/errors.ts` (mesmo padrão de `ServiceNotFoundError`/`IncidentNotFoundError`)
- [X] T012 [P] [US2] Extend `resolveStrategy` in `src/agents/index.ts` to throw `UnknownStrategyError(name)` quando `name` é informado e não está em `STRATEGIES` — depends on T007, T011
- [X] T013 [P] [US2] Add `UnknownStrategyError` handling to the error middleware in `src/http/server.ts` (→ `422 { error: "unknown_strategy", strategy }`) — depends on T003, T011

**Checkpoint**: User Stories 1 e 2 funcionam juntas e de forma independente.

---

## Phase 5: User Story 3 - Pedir uma resposta autocriticada via reflection (Priority: P2)

**Goal**: `reflect: true` decora a estratégia resolvida com `withReflection` (feature 002) antes de executá-la.

**Independent Test**: enviar a mesma pergunta com e sem `reflect: true` (resolvedores fake distintos injetados) e conferir que a versão com `reflect` usa a estratégia "refletida".

### Tests for User Story 3 ⚠️

- [X] T014 [P] [US3] Unit test in `src/agents/index.test.ts`: `resolveStrategy("react", true)` retorna uma estratégia cujo `name` é `"reflect:react"` (composição via `withReflection`)
- [X] T015 [P] [US3] Integration test in `src/http/server.test.ts`: `POST /chat` com `reflect: true` (resolvedor fake que devolve uma estratégia "refletida" distinta da base) usa a estratégia refletida na resposta

### Implementation for User Story 3

- [X] T016 [US3] Extend `resolveStrategy` in `src/agents/index.ts` to accept `reflect?: boolean` e envolver a estratégia resolvida com `withReflection` (já existente, feature 002) quando `true` — depends on T012

**Checkpoint**: User Stories 1, 2 e 3 funcionam juntas e de forma independente.

---

## Phase 6: User Story 4 - Confiar que a requisição nunca fica pendente indefinidamente (Priority: P3)

**Goal**: execução que ultrapassa `timeoutMs` responde `504` sem deixar a conexão pendente.

**Independent Test**: configurar `createApp({ timeoutMs: <pequeno> })` com uma estratégia fake que nunca resolve dentro dessa janela e verificar `504` em uma margem curta.

### Tests for User Story 4 ⚠️

- [X] T017 [P] [US4] Unit test in `src/services/chat.service.test.ts`: `runWithTimeout` resolve normalmente quando a estratégia termina antes de `timeoutMs`; rejeita com `ChatTimeoutError` quando não termina; não deixa nenhum timer pendente em nenhum dos dois casos
- [X] T018 [P] [US4] Integration test in `src/http/server.test.ts`: `createApp({ timeoutMs: 20, resolveStrategy: <fake que nunca resolve> })` retorna `504 { error: "timeout", timeoutMs: 20 }`

### Implementation for User Story 4

- [X] T019 [US4] Add `ChatTimeoutError` to `src/domain/errors.ts`
- [X] T020 [US4] Create `runWithTimeout(strategy, input, options, timeoutMs)` in `src/services/chat.service.ts` per [contracts/strategy-registry.md](./contracts/strategy-registry.md) — depends on T019
- [X] T021 [US4] Wire `runWithTimeout` into the `POST /chat` route in `src/http/server.ts` (substitui a chamada direta `strategy.run(message)` introduzida em T008) com `timeoutMs` padrão `180000`, sobrescrevível via `createApp(options)` — depends on T008, T020
- [X] T022 [US4] Add `ChatTimeoutError` handling to the error middleware in `src/http/server.ts` (→ `504 { error: "timeout", timeoutMs }`) — depends on T003, T019

**Checkpoint**: todas as 4 user stories funcionam juntas e de forma independente — feature completa.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Garantias que atravessam todas as user stories

- [X] T023 [P] Integration test in `src/http/server.test.ts`: duas requisições `POST /chat` concorrentes com fakes/respostas diferentes recebem, cada uma, seu próprio `trace`/`metrics` (FR-010)
- [X] T024 Run `npm run typecheck` and `npm test`; corrigir qualquer falha nos arquivos novos/alterados desta feature
- [ ] T025 Executar os passos 3–5 do [quickstart.md](./quickstart.md) manualmente contra um `npm run dev` real (requer `.env` com `OPENROUTER_API_KEY`/`OPENROUTER_MODEL`) para confirmar ponta a ponta o caminho feliz, a combinação estratégia explícita + reflect, e os dois erros síncronos (400/422) — **pendente**: requer `.env` com credencial real, que este agente não lê/possui (Principle VI da constitution); executar manualmente

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: sem dependências — pode começar imediatamente
- **Foundational (Phase 2)**: depende do Setup — BLOQUEIA todas as user stories
- **User Stories (Phase 3-6)**: todas dependem da Foundational completa
  - Podem prosseguir em paralelo (se houver mais de um desenvolvedor) ou sequencialmente em ordem de prioridade (US1 → US2 → US3 → US4)
  - US2/US3 estendem a mesma função `resolveStrategy` criada na US1 (T007) — na prática, sequenciais entre si mesmo que "independentes" em termos de valor de negócio
  - US4 estende a mesma rota `POST /chat` criada na US1 (T008) — mesma observação
- **Polish (Phase 7)**: depende de todas as user stories desejadas estarem completas

### User Story Dependencies

- **US1 (P1)**: depende apenas da Foundational
- **US2 (P2)**: depende da Foundational; estende código criado pela US1 (`resolveStrategy`), mas é testável e entrega valor de forma independente
- **US3 (P2)**: depende da Foundational; estende `resolveStrategy` (código já tocado pela US2), testável de forma independente
- **US4 (P3)**: depende da Foundational; estende a rota `POST /chat` (código já tocado pela US1), testável de forma independente

### Parallel Opportunities

- T002 e T003 (Foundational) — arquivos diferentes, sem dependência entre si
- T009 e T010 (US2, testes) — arquivos diferentes
- T012 e T013 (US2, implementação) — arquivos diferentes, ambos dependem só de T011
- T014 e T015 (US3, testes) — arquivos diferentes
- T017 e T018 (US4, testes) — arquivos diferentes

---

## Parallel Example: Foundational

```bash
Task: "Create src/agents/index.ts with STRATEGIES registry and DEFAULT_STRATEGY_NAME"
Task: "Create src/http/server.ts with createApp(options?), express.json(), ChatRequestSchema and generic error middleware"
```

## Parallel Example: User Story 2 (implementação)

```bash
Task: "Extend resolveStrategy in src/agents/index.ts to throw UnknownStrategyError for unknown names"
Task: "Add UnknownStrategyError handling to the error middleware in src/http/server.ts"
```

---

## Implementation Strategy

### MVP First (User Story 1 apenas)

1. Completar Phase 1: Setup
2. Completar Phase 2: Foundational (CRÍTICO — bloqueia todas as stories)
3. Completar Phase 3: User Story 1
4. **PARAR e VALIDAR**: rodar `npm test` e confirmar que `POST /chat` funciona com a estratégia padrão e rejeita corpo inválido
5. Deploy/demo se pronto

### Incremental Delivery

1. Setup + Foundational → servidor sobe, sem rotas
2. + US1 → caminho feliz com estratégia padrão + corpo inválido (400) → **MVP**
3. + US2 → estratégia explícita + estratégia desconhecida (422)
4. + US3 → `reflect: true`
5. + US4 → timeout (504)
6. + Polish → isolamento de requisições concorrentes, `typecheck`/`test` verdes, validação manual via quickstart

---

## Notes

- `[P]` = arquivos diferentes, sem dependência entre as tasks marcadas
- `[Story]` mapeia cada task à sua user story para rastreabilidade
- Todas as tasks de teste usam estratégias fake injetadas via `createApp({ resolveStrategy, timeoutMs })` — nenhum teste desta feature depende de rede ou de `OPENROUTER_API_KEY`/`OPENROUTER_MODEL` (exceto a validação manual do quickstart em T025, que é deliberadamente fora da suíte automatizada)
- Confirmar que os testes falham antes da implementação correspondente (US1-US4 seguem TDD)
- Fazer commit após cada task ou grupo lógico de tasks
- Parar em qualquer checkpoint para validar a story de forma independente antes de seguir
