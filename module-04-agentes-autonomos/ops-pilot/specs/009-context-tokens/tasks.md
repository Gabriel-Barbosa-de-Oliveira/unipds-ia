---

description: "Task list template for feature implementation"
---

# Tasks: Medição de Contexto

**Input**: Design documents from `/specs/009-context-tokens/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos — a constitution (Teste Obrigatório, NON-NEGOTIABLE) pede testes explicitamente para toda lógica nova.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivo diferente, sem dependência de tarefa ainda não concluída)
- **[Story]**: A qual user story a tarefa pertence (US1–US3)
- Caminhos de arquivo exatos em cada descrição

## Path Conventions

Projeto único — `src/` na raiz do repositório (ver plan.md § Project Structure). Sem `tests/` separado: testes ficam ao lado do código (`*.test.ts`), convenção já usada pelas features 001–008.

---

## Phase 1: Setup

**Purpose**: Inicialização de projeto.

Nenhuma tarefa de setup necessária nesta feature — nenhuma dependência de pacote nova (research.md: reaproveita `@langchain/core`, já presente), nenhuma tabela nova. A fundação começa diretamente na Phase 2.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: O módulo `context/tokens.ts` completo — `estimateTokens`, `UsageCollector` (real + fallback estimado), `mergeTokenUsage`, `buildContextBreakdown` — base sobre a qual as 3 user stories se apoiam. Nenhuma delas é testável de ponta a ponta antes desta fase estar completa.

**⚠️ CRITICAL**: Nenhuma user story começa antes desta fase estar completa.

- [X] T001 [P] Criar `src/context/tokens.ts`: `estimateTokens(text: string): number` — `Math.ceil(text.length / 4)`, nunca negativo (research.md item 2). `type TokenSource = "real" | "estimated" | "mixed"`; `interface TokenUsage { promptTokens: number; source: TokenSource }`; `mergeTokenUsage(a: TokenUsage, b: TokenUsage): TokenUsage` — soma `promptTokens`; `source` é `a.source` quando `a.source === b.source`, senão `"mixed"` (research.md item 6). `class UsageCollector extends BaseCallbackHandler` (`@langchain/core/callbacks/base`) — `handleLLMStart(llm: Serialized, prompts: string[], runId: string)` guarda `estimateTokens(prompts.join("\n"))` num `Map<string, number>` por `runId`; `handleLLMEnd(output: LLMResult, runId: string)` lê `output.llmOutput?.tokenUsage?.promptTokens`: se for `number`, soma ao total acumulado e marca que houve uso real; senão, soma a estimativa guardada em `handleLLMStart` para aquele `runId` (ou `0` se não houver, ex.: `runId` desconhecido) e marca que houve estimativa; getter `tokenUsage: TokenUsage` retorna `{ promptTokens: <total acumulado>, source: <"real" se só houve real, "estimated" se só houve estimativa, "mixed" se houve as duas> }` (research.md itens 1, 3, 4). `interface ContextBreakdown { currentMessage: number; conversationHistory: number; recalledFacts: number; total: number }`; `buildContextBreakdown(parts: { currentMessage: string; historyTexts: readonly string[]; factTexts: readonly string[] }): ContextBreakdown` — `currentMessage = estimateTokens(parts.currentMessage)`; `conversationHistory = estimateTokens(parts.historyTexts.join("\n"))`; `recalledFacts = estimateTokens(parts.factTexts.join("\n"))`; `total = currentMessage + conversationHistory + recalledFacts` (research.md item 7, por construção).
- [X] T002 (depende de T001) Criar `src/context/tokens.test.ts`: `estimateTokens` — `""` → `0`; string de 4 caracteres → `1`; string de 5 caracteres → `2` (arredonda para cima); nunca negativo. `mergeTokenUsage` — real+real soma e mantém `"real"`; estimated+estimated soma e mantém `"estimated"`; real+estimated (e vice-versa) soma e vira `"mixed"`. `UsageCollector` — construir `LLMResult`/`Serialized` forjados à mão (sem rede, sem chamar `createModel`): (a) uma chamada (`handleLLMStart` seguido de `handleLLMEnd`) com `output.llmOutput.tokenUsage.promptTokens` presente → `tokenUsage` final `{ promptTokens: <valor exato>, source: "real" }`; (b) uma chamada com `output.llmOutput` sem `tokenUsage` → usa a estimativa capturada em `handleLLMStart`, `source: "estimated"`; (c) duas chamadas com `runId`s diferentes, uma real e uma sem uso relatado → `promptTokens` é a soma das duas, `source: "mixed"`; (d) `handleLLMEnd` chamado para um `runId` sem `handleLLMStart` correspondente não lança erro (estimativa tratada como `0`). `buildContextBreakdown` — partes vazias (`""`, `[]`, `[]`) → todos os campos `0`; partes não-vazias → `total` sempre igual à soma de `currentMessage + conversationHistory + recalledFacts`, para várias combinações.

**Checkpoint**: `context/tokens.ts` completo e testado, sem rede — as user stories podem começar.

---

## Phase 3: User Story 1 - Ver o uso real de tokens de cada resposta do copiloto (Priority: P1) 🎯 MVP

**Goal**: `Metrics` ganha `promptTokens`/`tokenSource`, agregados entre **todas** as chamadas ao modelo feitas para uma resposta (incl. chamadas de crítica quando `reflect: true`), via `UsageCollector` passado como callback em cada ponto de chamada; `POST /chat` expõe esses campos na resposta.

**Independent Test**: Com uma estratégia fake cujo `RunResult.metrics` já traz `promptTokens`/`tokenSource` fixos, confirmar que a resposta de `/chat` os repassa sem alteração; com `reflection.test.ts`, confirmar que `runReflectionLoop` combina corretamente o `tokenUsage` da tentativa com o da crítica.

- [X] T003 [P] [US1] (depende de T001) Estender `src/agents/types.ts`: `Metrics` ganha `promptTokens: number` e `tokenSource: TokenSource` (tipo importado de `../context/tokens.ts`).
- [X] T004 [US1] (depende de T001, T003) Estender `src/agents/metrics.ts`: `buildMetrics(counter: LlmCallCounter, usageCollector: UsageCollector, latencyMs: number): Metrics` — inclui `promptTokens`/`tokenSource` lidos de `usageCollector.tokenUsage` no objeto retornado.
- [X] T005 [P] [US1] (depende de T004) Estender `src/agents/react.ts`: criar `const usageCollector = new UsageCollector();` junto ao `counter` já existente em `run()`; adicionar `usageCollector` ao array `callbacks` já passado a `agent.stream(...)`; usar `buildMetrics(counter, usageCollector, elapsed())` nos dois `return` (sucesso e `GraphRecursionError`).
- [X] T006 [P] [US1] (depende de T004) Estender `src/agents/plan-and-execute.ts`: criar `usageCollector` junto ao `counter` em `createPlanAndExecuteStrategy` (passado para `buildGraph`); adicionar `usageCollector` ao array `callbacks` nas 3 chamadas existentes (`planner`, `executor`, `replanner`); usar `buildMetrics(counter, usageCollector, elapsed())` no retorno final.
- [X] T007 [US1] (depende de T001, T003) Estender `src/agents/reflection.ts`: `critique(input, result)` cria seu próprio `UsageCollector`, passa `{ callbacks: [usageCollector] }` ao `.invoke(...)` existente (hoje sem nenhum `callbacks`), e passa a retornar `{ verdict: Verdict; tokenUsage: TokenUsage }` em vez de só `Verdict`; `CritiqueFn` (tipo) e `runReflectionLoop` atualizados de acordo — `ReflectionResult` ganha `tokenUsage: TokenUsage`, combinando a cada iteração `{ promptTokens: attempt.metrics.promptTokens, source: attempt.metrics.tokenSource }` (da tentativa) com o `tokenUsage` retornado pela crítica via `mergeTokenUsage`, acumulando entre iterações (mesmo espírito de `llmCalls += attempt.metrics.llmCalls; llmCalls += 1`); `withReflection`'s `run()` usa `result.tokenUsage.promptTokens`/`result.tokenUsage.source` para preencher `promptTokens`/`tokenSource` no `Metrics` final retornado.
- [X] T008 [P] [US1] (depende de T007) Estender `src/agents/reflection.test.ts`: fakes de `runAttempt` retornam `metrics.promptTokens`/`metrics.tokenSource` fixos; fakes de `critiqueFn` retornam `{ verdict, tokenUsage }`; confirmar que `runReflectionLoop` combina corretamente os dois via `mergeTokenUsage`, inclusive no caso com regeneração (mais de uma tentativa e mais de uma crítica, valores acumulando entre iterações).
- [X] T009 [P] [US1] (depende de T003) Estender `src/http/server.test.ts`: injetar uma estratégia fake cujo `RunResult.metrics` inclui `promptTokens: number` e `tokenSource: "real"` fixos; confirmar que a resposta `200` de `POST /chat` repassa esses valores em `metrics` sem alteração — nenhuma mudança de produção necessária em `server.ts` além do spread já existente (`metrics: { ...result.metrics, historyMessages }`, `006`).

**Checkpoint**: User Story 1 completa e testável de forma independente — `promptTokens`/`tokenSource` reais disponíveis via `/chat`, inclusive com reflection. MVP entregável.

---

## Phase 4: User Story 2 - Estimativa de tokens quando o uso real não está disponível (Priority: P2)

**Goal**: Confirmar de ponta a ponta (via `/chat`) que quando uma estratégia relata `tokenSource` `"estimated"` ou `"mixed"` (porque alguma chamada ao modelo não relatou uso real — já implementado e testado unitariamente em `UsageCollector`, T001/T002), a resposta ainda traz `promptTokens` preenchido, nunca ausente, e o `tokenSource` correspondente é repassado sem confundir com `"real"`.

**Independent Test**: Com uma estratégia fake cujo `RunResult.metrics` traz `tokenSource: "estimated"` (e depois `"mixed"`), confirmar que a resposta de `/chat` repassa `promptTokens`/`tokenSource` corretamente para os dois casos.

- [X] T010 [P] [US2] (depende de T009) Estender `src/http/server.test.ts`: dois casos adicionais — (a) estratégia fake com `metrics.tokenSource: "estimated"`, confirmar que a resposta repassa `promptTokens`/`tokenSource: "estimated"` sem alteração; (b) estratégia fake com `metrics.tokenSource: "mixed"`, mesma confirmação. Nenhum dos dois casos é tratado de forma diferente de `"real"` pelo controller (FR-003/FR-004 — o controller nunca reinterpreta a origem, só repassa).

**Checkpoint**: User Stories 1 e 2 funcionam de forma independente.

---

## Phase 5: User Story 3 - Detalhamento de onde vem o contexto enviado ao modelo (Priority: P3)

**Goal**: `POST /chat` calcula e expõe `metrics.contextBreakdown` (mensagem atual, histórico, fatos lembrados, total), via `buildContextBreakdown` chamado no controller HTTP a partir das peças já disponíveis antes de compor o prompt final.

**Independent Test**: Requisição com histórico de conversa e fatos de memória presentes produz um `contextBreakdown` com partes `> 0` e `total` igual à soma; requisição sem nenhum dos dois produz zeros explícitos.

- [X] T011 [US3] (depende de T001) Estender `src/http/server.ts`: antes de `composePrompt`/`composeWithFacts`, chamar `buildContextBreakdown({ currentMessage: parsed.data.message, historyTexts: history.map((m) => m.content), factTexts: recalledFacts })`; incluir o resultado na resposta: `metrics: { ...result.metrics, historyMessages: history.length, contextBreakdown }`.
- [X] T012 [P] [US3] (depende de T011) Estender `src/http/server.test.ts`: (a) requisição com histórico (via `conversationStore` fake já existente) e fatos (via `memoryStore` fake já existente) produz `contextBreakdown.conversationHistory > 0` e `contextBreakdown.recalledFacts > 0`, com `contextBreakdown.total` igual à soma exata das 3 partes; (b) requisição sem `conversationId`/histórico e sem `userId`/fatos produz `contextBreakdown.conversationHistory === 0` e `contextBreakdown.recalledFacts === 0` (presentes, nunca `undefined`), com `total === currentMessage`.

**Checkpoint**: Todas as 3 user stories funcionam de forma independente.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Confirmar não-regressão e aderência final aos contratos desta feature.

- [X] T013 [P] Rodar `npm run typecheck` e `npm test` e confirmar que as suítes já existentes (001–008) continuam passando sem alteração de comportamento observável — em particular, que `answer`/`trace`/raciocínio permanecem idênticos a antes desta feature (FR-008).
- [X] T014 Rodar manualmente `specs/009-context-tokens/quickstart.md` de ponta a ponta contra `npm run dev` real (`OPENROUTER_API_KEY`/`OPENROUTER_MODEL` configurados) — os 4 passos: `promptTokens`/`tokenSource: "real"` numa resposta simples; `contextBreakdown` com histórico e fatos presentes, `total` correto; zeros explícitos sem histórico/fatos; limpeza dos dados de teste no SQLite.
- [X] T015 [P] Revisar `specs/009-context-tokens/contracts/post-chat.md` e `contracts/tokens.md` contra a implementação final — conferir assinaturas (`estimateTokens`, `UsageCollector`, `mergeTokenUsage`, `buildContextBreakdown`, `Metrics`) e corrigir qualquer divergência encontrada.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: nenhuma tarefa — Foundational começa direto.
- **Foundational (Phase 2)**: BLOQUEIA todas as user stories — `context/tokens.ts` precisa existir e estar testado antes de qualquer wiring.
- **User Stories (Phase 3–5)**: todas dependem da Foundational completa.
  - US1 (P1) não depende de US2/US3.
  - US2 (P2) depende só de US1 ter estabelecido o padrão de repasse em `server.test.ts` (T009) — a lógica de fallback em si já está pronta e testada desde a Foundational (T001/T002); não depende do wiring real em `react.ts`/`plan-and-execute.ts`/`reflection.ts` (T005–T008), só do tipo `Metrics` (T003).
  - US3 (P3) depende só da Foundational (T001) — totalmente independente de US1/US2, já que `contextBreakdown` não usa `UsageCollector`/`promptTokens`.
- **Polish (Phase 6)**: depende de todas as user stories desejadas estarem completas.

### Dentro de cada User Story

- US1: T003 → T004 (depende de T003) → T005/T006 (paralelas entre si, dependem de T004) e T007 (depende de T001/T003, não de T004) → T008 (depende de T007); T009 depende só de T003 (usa estratégia fake, não a wiring real) — pode ser feita em paralelo com T004–T008.
- US2: T010 depende só de T009 (estende o mesmo bloco de teste) — não depende de T005–T008.
- US3: T011 depende só de T001 → T012 depende de T011.

### Parallel Opportunities

- Dentro da Foundational: só T001, depois T002 (depende de T001).
- T005 e T006 (US1) são paralelizáveis entre si (arquivos diferentes: `react.ts` vs `plan-and-execute.ts`), ambas dependendo só de T004.
- T009 (US1, `server.test.ts`) pode ser feita em paralelo com T004–T008, já que usa uma estratégia fake e depende só de T003.
- T010 (US2) e T011–T012 (US3) podem ser feitas em paralelo com T004–T008 (US1), já que dependem só da Foundational (T001) e, no caso de T010, de T009.
- T013 e T015 (Polish) são paralelizáveis entre si.

---

## Parallel Example: User Story 1 (depois da Foundational)

```bash
# Depois de T001-T004, em paralelo:
Task: "Wiring de UsageCollector em react.ts (T005)"
Task: "Wiring de UsageCollector em plan-and-execute.ts (T006)"
Task: "Teste de repasse via estratégia fake em server.test.ts (T009, só depende de T003)"
```

## Parallel Example: User Story 2 + User Story 3 (depois de T001, T009)

```bash
Task: "Casos estimated/mixed em server.test.ts (T010, US2)"
Task: "Wiring + teste de contextBreakdown em server.ts/server.test.ts (T011-T012, US3)"
```

---

## Implementation Strategy

### MVP First (User Story 1 apenas)

1. Completar Phase 2 (Foundational) — `context/tokens.ts` pronto e testado.
2. Completar Phase 3 (US1: T003–T009).
3. **PARAR e VALIDAR**: rodar `npm test` e os passos 1–2 de `quickstart.md` (testes determinísticos; `promptTokens`/`tokenSource: "real"` numa resposta real de `/chat`).
4. Nesse ponto, `npm run dev` já expõe uso real de tokens (incl. reflection) — MVP entregável.

### Incremental Delivery

1. Foundational → `context/tokens.ts` pronto e testado.
2. US1 → validar independentemente → `promptTokens`/`tokenSource` reais via `/chat` (MVP).
3. US2 → validar independentemente → fallback estimado confirmado de ponta a ponta.
4. US3 → validar independentemente → `contextBreakdown` disponível via `/chat`.
5. Polish → confirma não-regressão (001–008), valida o quickstart completo, fecha os contratos.

---

## Notes

- `[P]` = arquivos diferentes, sem dependência de tarefa incompleta.
- Rótulo `[US#]` mapeia a tarefa à user story correspondente da spec.
- Cada user story é completável e testável de forma independente — US2/US3 dependem só da Foundational (e, no caso de US2, do padrão de teste estabelecido por US1 em T009), nunca do wiring real em `react.ts`/`plan-and-execute.ts`/`reflection.ts`.
- Commitar após cada tarefa ou grupo lógico pequeno (constitution: Pequeno e Reversível).
- `npm run typecheck` e `npm test` devem ficar verdes ao final de cada fase — nenhuma tarefa desta feature depende de rede ou de `OPENROUTER_API_KEY`, exceto a validação manual de `quickstart.md` (T014).
