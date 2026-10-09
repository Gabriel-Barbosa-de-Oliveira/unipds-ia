---

description: "Task list for 013-model-resilience"
---

# Tasks: Resiliência de Modelo (retry + modelo de reserva)

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**: a fábrica se chama `createModel(build?, config?, deps?)` (não `resilientRunnable`), e o modelo base se chama `baseModel(id)` (não `createChatModel`). São **`stopAfterAttempt: 2`** (1 tentativa + 1 retry) **no principal e também na reserva**, compostos como `primary.withFallbacks([backup])`. Os números de tentativas citados abaixo (3 no principal, sem retry na reserva) foram substituídos por esses.

**Input**: Design documents from `/specs/013-model-resilience/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos. A constitution (Princípio V, NON-NEGOTIABLE) exige teste para toda lógica nova, e a FR-013 pede testes sem rede para classificação de erro, reserva válida, evento de fallback e `modelUsed`.

**Organization**: as tarefas estão agrupadas por user story (spec.md), para que cada uma possa ser implementada e testada de forma independente.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivo diferente e sem dependência de tarefa ainda não concluída).
- **[Story]**: a user story à qual a tarefa pertence (US1–US3).

## Path Conventions

Projeto único, com `src/` na raiz. Os testes ficam ao lado do código (`*.test.ts`).

**Modelos fake**: use `FakeStreamingChatModel` de `@langchain/core/utils/testing`, que tem `bindTools`, ou uma subclasse mínima de `BaseChatModel` que lê um roteiro de falhas. Cada instância recebe um `modelId` e devolve `invocationParams()` contendo `{ model: modelId }`, para que o `ModelUsageTracker` veja o nome. Os erros transitórios simulados são objetos `Error` com `status = 429`, e os não transitórios têm `status = 400`.

---

## Phase 1: Setup

**Purpose**: inicialização do projeto.

Fase vazia: nenhuma dependência nova. `@langchain/core` já traz `withRetry`, `withFallbacks`, `RunnableBinding` e os modelos fake de teste.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: a configuração e a classificação de erros (puras), mais os tipos compartilhados.

**⚠️ CRITICAL**: nenhuma user story começa antes desta fase terminar.

- [X] T001 [P] Criar `src/agents/model.test.ts` (deve falhar até a T002) com os testes de:
  - `loadModelConfig`:
    - `{ OPENROUTER_MODEL: "a" }` → `{ primary: "a" }`;
    - `{ OPENROUTER_MODEL: "a", OPENROUTER_MODEL_FALLBACK: "b" }` → `{ primary: "a", fallback: "b" }`;
    - `OPENROUTER_MODEL_FALLBACK` com valor `""`, `"   "` ou `"a"` (igual ao principal) → sem `fallback`;
    - `" b "` → `"b"`;
    - sem `OPENROUTER_MODEL` → lança `Error` com a mensagem `"OPENROUTER_MODEL não configurada"`.
  - `isTransientModelError`:
    - `true` para `{status: 429}`, `{status: 408}`, `{status: 500}`, `{status: 503}`, `{ response: { status: 502 } }`, `{ code: "ECONNRESET" }`, `{ code: "ETIMEDOUT" }`, `{ name: "APIConnectionTimeoutError" }`, `new Error("fetch failed")` e `new Error("Rate limit exceeded")`;
    - `false` para `{status: 400}`, `{status: 401}`, `{status: 404}`, `new Error("schema inválido")`, `"texto"` e `undefined`.
- [X] T002 Reescrever o topo de `src/agents/model.ts` (research.md itens 1–2, [contracts/model-factory.md](./contracts/model-factory.md)):
  - `export interface ModelConfig { primary: string; fallback?: string }`;
  - `export function loadModelConfig(env)`, com zod (`z.string().trim().min(1)` no principal e `z.string().trim().optional()` na reserva). Reserva vazia ou igual ao principal vira `undefined`;
  - `export function isTransientModelError(error: unknown): boolean`, puro, conforme a tabela do research.md item 2;
  - `export function createChatModel(modelId: string): ChatOpenAI`, com OpenRouter, `temperature: 0`, **`maxRetries: 0`** e a mesma checagem de `OPENROUTER_API_KEY` de hoje.

  **Manter** `createModel()` por enquanto, delegando para `createChatModel(loadModelConfig(process.env).primary)`; ele é removido na T014. Rodar a T001 até ficar verde.
- [X] T003 [P] Estender `src/agents/types.ts` (data-model.md):
  - nova variante de `TraceEvent` `{ type: "fallback"; at: number; from: string; to: string; reason: string }` (dentro da mesma união que recebe `& { node?: GraphNode }`);
  - `Metrics` ganha `modelUsed: string`;
  - exportar `interface ModelFallback { from: string; to: string; reason: string }`.

  Corrigir o que o `tsc` apontar:
  - em `src/agents/trace.ts`, o caso `"fallback"` → `` `[fallback] ${event.from} → ${event.to}: ${event.reason}` ``;
  - em `src/agents/metrics.ts`, `buildMetrics(counter, usageCollector, latencyMs, modelUsed: string)`. Os chamadores passam, provisoriamente, `loadModelConfig(process.env).primary`; isso é substituído na US2;
  - em `src/agents/reflection.ts`, `src/graph/production-graph.ts` e nos fixtures de teste com `Metrics` literal (`src/http/server.test.ts` com cerca de 24, `src/graph/production-graph.test.ts`, `src/agents/trace.test.ts`, `src/services/chat.service.test.ts`, `src/agents/reflection.test.ts` e outros apontados pelo `tsc`), acrescentar `modelUsed: "fake-model"`;
  - no grafo, `combineMetrics` usa `modelUsed: strategy.modelUsed`.

  `npm run typecheck` e `npm test` verdes.

**Checkpoint**: configuração, classificação e tipos prontos; a suíte continua verde.

---

## Phase 3: User Story 1 - Continuar respondendo quando o principal falha (Priority: P1) 🎯 MVP

**Goal**: toda chamada ao modelo passa por `build(principal).withRetry(...).withFallbacks([build(reserva)])`.

**Independent Test**: com modelos fake que falham por roteiro, uma falha transitória isolada é absorvida pelo principal, e falhas persistentes ou não transitórias resultam em resposta da reserva. Sem reserva, o erro propaga.

### Tests for User Story 1

- [X] T004 [P] [US1] (depende de T002) Estender `src/agents/model.test.ts` com `resilientRunnable(build, config, deps)`. Usar `deps.createChatModel` devolvendo fakes por `modelId` que contam chamadas, e `build = (m) => m` (o modelo puro). Para os testes não esperarem o backoff real, injetar `deps.retryOptions = { minTimeout: 0, maxTimeout: 0 }` (repassado ao `withRetry` como `onFailedAttempt` + config do p-retry; se a versão do core não aceitar, usar `stopAfterAttempt` e aceitar cerca de 3 s). Cenários do contrato:
  - **(a)** caminho feliz: 1 chamada ao principal e 0 à reserva;
  - **(b)** 429 e depois sucesso: 2 chamadas ao principal, resposta do principal;
  - **(c)** 429 sempre: 3 chamadas ao principal e 1 à reserva, resposta da reserva;
  - **(d)** 400: 1 chamada ao principal e 1 à reserva;
  - **(e)** sem `fallback` e 429 sempre: rejeita depois de 3 chamadas;
  - **(f)** principal e reserva falhando: rejeita com o erro da reserva, que teve 1 chamada.
- [X] T005 [P] [US1] (depende de T002) Em `src/agents/model.test.ts`, testar `toolCallingModel(tools, config, deps)`: `createReactAgent({ llm: toolCallingModel(tools, …), tools })` compila e responde com `FakeStreamingChatModel`, sem lançar `must define bindTools method` (research.md item 4). Com o principal falhando (400), a resposta vem da reserva.

### Implementation for User Story 1

- [X] T006 [US1] (depende de T002) Em `src/agents/model.ts`, implementar:
  - `export interface ResilienceDeps { createChatModel?: (id: string) => BaseChatModel; stopAfterAttempt?: number; retryOptions?: … }`;
  - `export function resilientRunnable<T>(build, config = loadModelConfig(process.env), deps = {})`, que faz:
    ```ts
    const primary = build(make(config.primary)).withRetry({
      stopAfterAttempt: deps.stopAfterAttempt ?? 3,
      onFailedAttempt: (error) => {
        if (!isTransientModelError(error)) throw error;
      },
    });
    return config.fallback ? primary.withFallbacks([build(make(config.fallback))]) : primary;
    ```
  - `export function toolCallingModel(tools, config?, deps?)`:
    - `const specs = (make(config.primary).bindTools(tools) as RunnableBinding).kwargs.tools`;
    - `return new RunnableBinding({ bound: resilientRunnable((m) => m.bindTools!(tools), config, deps), kwargs: { tools: specs }, config: {} })`;
    - comentário explicando o `_shouldBindTools` do LangGraph 0.2.74.

  Rodar a T004 e a T005 até ficarem verdes.
- [X] T007 [P] [US1] (depende de T006) Em `src/agents/react.ts`, trocar `llm: createModel()` por `llm: toolCallingModel(tools)`.
- [X] T008 [P] [US1] (depende de T006) Em `src/agents/plan-and-execute.ts`, dentro de `buildGraph`:
  - `plannerModel = resilientRunnable((m) => m.withStructuredOutput(PlanSchema))`;
  - `replannerModel = resilientRunnable((m) => m.withStructuredOutput(ReplanSchema))`;
  - `executorAgent = createReactAgent({ llm: toolCallingModel(tools), tools })`;
  - remover `const model = createModel()`.
- [X] T009 [P] [US1] (depende de T006) Em `src/agents/reflection.ts` (`critique`), trocar para `resilientRunnable((m) => m.withStructuredOutput(verdictSchema)).invoke(...)`.
- [X] T010 [P] [US1] (depende de T006) Em `src/graph/router.ts` (`createModelRouter`), trocar para `resilientRunnable((m) => m.withStructuredOutput(routeSchema)).invoke(...)`. A falha final continua caindo no fallback de rota da 012.
- [X] T011 [P] [US1] (depende de T006) Em `src/memory/learning-reflector.ts`, trocar para `resilientRunnable((m) => m.withStructuredOutput(learningSchema)).invoke(...)`.
- [X] T012 [US1] (depende de T007–T011) Confirmar com `grep -rn "createModel()" src` que não há mais nenhum uso. Rodar `npm run typecheck` e `npm test`.

**Checkpoint**: o copiloto absorve falhas do principal e usa a reserva. Esse é o MVP.

---

## Phase 4: User Story 2 - Saber quando a reserva foi usada e qual modelo respondeu (Priority: P1)

**Goal**: evento `fallback` no trace a cada troca e `metrics.modelUsed` em toda resposta.

**Independent Test**: com logs sintéticos, `summarizeModelUsage` distingue retry de fallback e aponta o modelo do último `end`. Nos traces das estratégias e do grafo, os eventos `fallback` aparecem na posição certa.

### Tests for User Story 2

- [X] T013 [P] [US2] (depende de T003) Criar `src/agents/model-usage.test.ts` (deve falhar até a T015):
  - `summarizeModelUsage(log, { primary: "a", fallback: "b" })`:
    - `[start a, end a]` → `{ fallbacks: [], modelUsed: "a" }`;
    - `[start a, error a, start a, end a]` (retry) → `fallbacks: []`, `modelUsed: "a"`;
    - `[start a, error a(429), start a, error a(429), start b, end b]` → 1 fallback `{ from: "a", to: "b", reason: contém "429" }`, `modelUsed: "b"`;
    - duas chamadas independentes que caem na reserva → 2 fallbacks;
    - `[start a, error a, start b, end b, start a, end a]` → 1 fallback e `modelUsed: "a"` (o último `end`);
    - log vazio → `modelUsed: "a"`;
    - um `reason` com mais de 200 caracteres é truncado.
  - `withModelFallbacks(trace, fallbacks)`:
    - insere antes do último `answer` e reindexa `at` em `0..n-1`;
    - sem `answer`, insere no fim;
    - com `fallbacks` vazio, devolve um trace igual;
    - não muta a entrada.
  - `ModelUsageTracker`: `handleChatModelStart` com `extraParams.invocation_params.model`, depois `handleLLMError` e `handleLLMEnd` com o mesmo `runId`, produz as entradas esperadas.
- [X] T014 [P] [US2] (depende de T003) Estender `src/agents/trace.test.ts`:
  - `formatTraceEvent({ type: "fallback", at: 0, from: "a", to: "b", reason: "429" })` → `"[fallback] a → b: 429"`;
  - com `node: "react"` → `"react │ [fallback] a → b: 429"`;
  - `formatMetrics` → `"llmCalls=3 latencyMs=120 model=fake-model"`.

### Implementation for User Story 2

- [X] T015 [US2] (depende de T013) Criar `src/agents/model-usage.ts` (research.md item 5, data-model.md):
  - `export type ModelCallEntry` e `export class ModelUsageTracker extends BaseCallbackHandler`, com `name = "model-usage-tracker"`, `readonly log: ModelCallEntry[] = []`:
    - `handleChatModelStart(_llm, _messages, runId, _parentRunId, extraParams)` → `start` com `model = String(extraParams?.invocation_params?.model ?? "desconhecido")`;
    - `handleLLMEnd(_output, runId)` → `end`;
    - `handleLLMError(error, runId)` → `error` com a mensagem;
  - `export function summarizeModelUsage(log, config): { fallbacks: ModelFallback[]; modelUsed: string }`, pura;
  - `export function withModelFallbacks(trace, fallbacks): TraceEvent[]`, pura.
- [X] T016 [US2] (depende de T014, T003) Em `src/agents/trace.ts`, ajustar `formatMetrics` para incluir `` ` model=${metrics.modelUsed}` ``. O caso `"fallback"` da T003 já cobre `formatTraceEvent`.
- [X] T017 [US2] (depende de T015, T007) Em `src/agents/react.ts`:
  - criar `const tracker = new ModelUsageTracker()` e incluir nos `callbacks`;
  - fazer `const usage = summarizeModelUsage(tracker.log, loadModelConfig(process.env))` e, nos dois caminhos (normal e `GraphRecursionError`), `trace = withModelFallbacks(trace, usage.fallbacks)` e `buildMetrics(..., usage.modelUsed)`.
- [X] T018 [US2] (depende de T015, T008) Em `src/agents/plan-and-execute.ts`: tracker passado em todos os `callbacks` (planner, executor e replanner) e, no `run`, `withModelFallbacks(result.trace, usage.fallbacks)` mais `buildMetrics(..., usage.modelUsed)`.
- [X] T019 [US2] (depende de T015, T009) Em `src/agents/reflection.ts`:
  - `critique` passa a usar também um `ModelUsageTracker` e devolve `fallbacks`;
  - em `runReflectionLoop`, os fallbacks do crítico entram antes do evento `critique` da rodada, e o `at` segue sequencial;
  - `withReflection` devolve `metrics.modelUsed` = o `modelUsed` da última tentativa (guardar no `ReflectionResult`).

  Atualizar os fakes de `CritiqueFn` em `src/agents/reflection.test.ts` para devolver `fallbacks: []`, e acrescentar um teste em que o crítico devolve 1 fallback e ele aparece imediatamente antes do `critique` correspondente.
- [X] T020 [US2] (depende de T015, T010) Em `src/graph/router.ts`: `DecideRoute` passa a devolver `{ decided, tokenUsage, fallbacks: ModelFallback[] }`. O `createModelRouter` usa o tracker e `summarizeModelUsage(...).fallbacks`.

  Em `src/graph/production-graph.ts`, o nó `roteador` emite `[routeEvent, ...fallbacks.map((f, i) => ({ type: "fallback", at: 1 + i, node: "roteador", ...f }))]`. No `catch`, sem fallbacks. O nó de estratégia continua usando `tagTrace(result.trace, route, state.trace.length)`, que já carimba `node` da estratégia também nos `fallback` vindos dela.
- [X] T021 [US2] (depende de T020) Atualizar os testes:
  - `src/graph/production-graph.test.ts`:
    - fakes de `DecideRoute` com `fallbacks: []`;
    - `assertTraceInvariants` passa a aceitar, logo depois do `route`, eventos `fallback` com `node: "roteador"` (data-model.md, invariantes 2–3);
    - novo teste: um roteador fake com 1 fallback → `trace[1]` é `{ type: "fallback", node: "roteador", at: 1, ... }`, e os eventos da estratégia começam no índice 2;
    - novo teste: uma estratégia fake cujo trace contém um `fallback` → ele sai com `node` da estratégia;
    - `result.metrics.modelUsed === strategyResult.metrics.modelUsed`.
  - `src/http/server.test.ts`: o helper `fixedRouter` devolve `fallbacks: []`, e um teste confirma `body.metrics.modelUsed === "fake-model"`.

**Checkpoint**: as trocas de modelo ficam visíveis no trace, e toda resposta informa `modelUsed`.

---

## Phase 5: User Story 3 - Configurar a reserva sem mudar código (Priority: P2)

**Goal**: a reserva é ligada e desligada só por `OPENROUTER_MODEL_FALLBACK`.

**Independent Test**: com `resilientRunnable` usando `config` padrão (lido do `process.env` em cada chamada), definir e remover a variável muda o comportamento sem mudar o código.

- [X] T022 [US3] (depende de T006) Em `src/agents/model.test.ts`: com `process.env.OPENROUTER_MODEL`/`OPENROUTER_MODEL_FALLBACK` setados e restaurados no `after`, e com `deps.createChatModel` fake, o `resilientRunnable(build)` **sem** `config` explícito usa a reserva da env. Remover a variável (ou deixá-la igual ao principal) faz o erro propagar. Isso confirma que a config é lida a cada montagem, sem cache no import (SC-006).
- [ ] T023 [US3] Documentar `OPENROUTER_MODEL_FALLBACK` (opcional, mesmo provedor/credencial e vazio = desligado) no `specs/013-model-resilience/quickstart.md` (já feito) e acrescentar a linha `OPENROUTER_MODEL_FALLBACK=` em `.env.example`. **Manual (pendente)**: o guardrail do agente bloqueia o acesso a arquivos `.env*`, então quem faz essa edição é a pessoa usuária.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T024 [P] ~~Remover `createModel()`~~: ele foi mantido como nome da fábrica resiliente, seguindo o esboço. Atualizar o JSDoc do topo do arquivo descrevendo a fábrica resiliente.
- [X] T025 [P] Atualizar a nota da 012 em `specs/003-chat-endpoint/quickstart.md` com uma linha sobre `metrics.modelUsed` e os eventos `fallback`, linkando `specs/013-model-resilience/contracts/post-chat.md`.
- [X] T026 Rodar `npm run typecheck` e `npm test`; os dois precisam ficar verdes (Princípio V).
- [ ] T027 Validar manualmente os cenários 2–5 de [quickstart.md](./quickstart.md). Isso exige `OPENROUTER_API_KEY` no ambiente, sem nunca ler `.env`. **Pendente**: `OPENROUTER_API_KEY` não está configurada no ambiente desta execução.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup**: vazia.
- **Foundational**: a T001 roda em paralelo com a T003; a T002 depende da T001. A fase bloqueia as stories.
- **US1**: a T004 roda em paralelo com a T005; a T006 vem depois; T007 a T011 rodam em paralelo (arquivos diferentes); T012 fecha a fase.
- **US2**: a T013 roda em paralelo com a T014 (só precisam da T003). T015 → T017 a T020, cada uma depois da migração correspondente da US1. T021 vem por último.
- **US3**: a T022 depende só da T006. A T023 é manual.
- **Polish**: depois de todas as stories.

### User Story Dependencies

- **US1 (P1)**: independente depois da Foundational. Entrega o MVP.
- **US2 (P1)**: usa os call sites já migrados na US1 para plugar o tracker. A parte pura (T013, T015) pode começar junto com a US1.
- **US3 (P2)**: só testa o que a T002 e a T006 já entregam e documenta a variável.

## Parallel Opportunities

- **Foundational**: T001 ∥ T003.
- **US1**: T004 ∥ T005; depois T007 ∥ T008 ∥ T009 ∥ T010 ∥ T011.
- **US2**: T013 ∥ T014 (podem rodar durante a US1).
- **Polish**: T024 ∥ T025.

### Parallel Example: User Story 1

```bash
# Depois de T006:
Task: "T007 [US1] react.ts usa toolCallingModel"
Task: "T008 [US1] plan-and-execute.ts usa resilientRunnable/toolCallingModel"
Task: "T009 [US1] crítico do reflection usa resilientRunnable"
Task: "T010 [US1] roteador usa resilientRunnable"
Task: "T011 [US1] learning-reflector usa resilientRunnable"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Foundational (T001–T003).
2. US1 (T004–T012): retry e reserva em todos os call sites.
3. **STOP and VALIDATE**: os cenários (a)–(f) da T004 estão verdes e o `createReactAgent` funciona com o modelo resiliente.

### Incremental Delivery

Foundational → US1 (resiliência) → US2 (visibilidade) → US3 (configuração) → Polish. Cada passo deixa `typecheck` e `test` verdes.
