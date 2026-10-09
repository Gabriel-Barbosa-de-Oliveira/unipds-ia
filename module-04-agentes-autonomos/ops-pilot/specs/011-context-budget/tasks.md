---

description: "Task list for 010-context-budget"
---

# Tasks: Orçamento de Contexto por Seção

**Input**: Design documents from `/specs/010-context-budget/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos — a constitution (Teste Obrigatório, NON-NEGOTIABLE) exige teste para toda lógica nova, e o usuário pediu explicitamente o teste "tetos baixos cortam na ordem certa".

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivo diferente, sem dependência de tarefa ainda não concluída)
- **[Story]**: A qual user story a tarefa pertence (US1–US3)
- Caminhos de arquivo exatos em cada descrição

## Path Conventions

Projeto único — `src/` na raiz. Testes ao lado do código (`*.test.ts`), convenção das features 001–009.

---

## Phase 1: Setup

**Purpose**: Inicialização de projeto.

Nenhuma tarefa de setup — nenhuma dependência nova (`zod` já presente), nenhuma tabela nova.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: `ContextBreakdown` passa a conhecer as seções `system` e `summary`, base usada pelo builder (US1) e pelas métricas (US3).

**⚠️ CRITICAL**: Nenhuma user story começa antes desta fase estar completa.

- [X] T001 Estender `src/context/tokens.ts`: `interface ContextBreakdown` ganha `system: number` e `summary: number` (antes de `currentMessage`); `buildContextBreakdown(parts)` aceita `system?: string` e `summary?: string` (default `""`), calcula `system = estimateTokens(parts.system ?? "")`, `summary = estimateTokens(parts.summary ?? "")`, e `total = system + summary + currentMessage + conversationHistory + recalledFacts` (data-model.md § ContextBreakdown). Atualizar o JSDoc para mencionar as partes novas.
- [X] T002 (depende de T001) Estender `src/context/tokens.test.ts`: os asserts existentes de `buildContextBreakdown` ganham `system: 0, summary: 0`; novo caso com `system`/`summary` não vazios confirmando `total` igual à soma das 5 partes. Ajustar também `src/http/server.test.ts`: todo `deepEqual`/objeto esperado de `contextBreakdown` ganha `system: 0, summary: 0` (ex.: o bloco por volta da linha 142 e os cenários de "User Story 3 (009)"), para manter a suíte verde.

**Checkpoint**: `npm test` e `npm run typecheck` verdes com o breakdown estendido.

---

## Phase 3: User Story 1 - Contexto nunca estoura um teto previsível por seção (Priority: P1) 🎯 MVP

**Goal**: `src/context/context-builder.ts` monta o `input` de todas as estratégias aplicando os tetos padrão (resumo 200, janela 1200, memórias 300), com system e mensagem intocáveis; `/chat`, `arena` e `bench` passam a usá-lo.

**Independent Test**: `context-builder.test.ts` — com tetos baixos, só as mensagens mais antigas e as memórias de menor score saem, o resumo é truncado e system/mensagem permanecem íntegros; sem corte, a string é idêntica à composição anterior.

### Tests for User Story 1

- [X] T003 [P] [US1] (depende de T001) Criar `src/context/context-builder.test.ts` (escrito antes da implementação, deve falhar até T004) cobrindo [contracts/context-builder.md](./contracts/context-builder.md):
  - `trimSummary`: texto dentro do teto → inalterado; acima → `slice(0, budget * 4)` (mantém o início) e `estimateTokens(resultado) ≤ budget`; `budget = 0` → `""`.
  - `trimWindow`: janela dentro do teto → inalterada; acima → remove as mais antigas, mantém ordem cronológica, `estimateTokens(contents.join("\n")) ≤ budget`; mensagem mais recente sozinha maior que o teto → `[]`; `budget = 0` → `[]`.
  - `trimMemories`: entrada fora de ordem → saída em score desc; acima do teto → remove as de menor score; empate de score mantém a ordem de entrada (determinístico); `budget = 0` → `[]`.
  - `buildContext` — **teste-âncora "tetos baixos cortam na ordem certa"**: budget `{ summary: 2, window: <≈2 mensagens>, memories: <≈1 fato> }`, entrada com `system` e `message` longos (maiores que qualquer teto), resumo de 20 caracteres, 4 mensagens (`m1`..`m4`, cronológicas) e 3 memórias com scores `0.5`, `0.9`, `0.7` (fora de ordem). Esperado: `window` = `[m3, m4]`; `memories` = só a de score `0.9`; `summary` com 8 caracteres; `trimmed = { historyMessages: 2, recalledFacts: 2 }`; `prompt` contém `system` e `message` exatamente; `prompt` não contém `m1`, `m2`, nem os fatos de score `0.5`/`0.7`; `breakdown` pós-corte com `total` igual à soma das partes.
  - `buildContext` — regressão FR-011: sem `system`/`summary` e com tetos padrão, `prompt === composeWithFacts(facts, composePrompt(history, message))` para (a) só mensagem, (b) mensagem + histórico, (c) mensagem + histórico + memórias.
  - `buildContext` — independência entre seções: memórias cortadas não liberam espaço para a janela (e vice-versa).
  - `buildContext` — ordem dos blocos: `system`, depois `"Resumo da conversa até aqui:\n" + summary`, depois o bloco composto, separados por `"\n\n"`; blocos vazios omitidos.

### Implementation for User Story 1

- [X] T004 [US1] (depende de T001) Criar `src/context/context-builder.ts` (funções puras, sem IO, sem `process.env`), conforme [data-model.md](./data-model.md) e [contracts/context-builder.md](./contracts/context-builder.md):
  - `interface ContextBudget { readonly summary: number; readonly window: number; readonly memories: number }`; `export const DEFAULT_CONTEXT_BUDGET: ContextBudget = { summary: 200, window: 1200, memories: 300 }`.
  - `interface ScoredMemory { readonly fact: string; readonly score: number }`; `interface ContextInput { message: string; system?: string; summary?: string; window?: readonly ConversationMessage[]; memories?: readonly ScoredMemory[] }` (`ConversationMessage` de `../domain/conversation.ts`); `interface BuiltContext { prompt: string; window: ConversationMessage[]; memories: ScoredMemory[]; summary: string; breakdown: ContextBreakdown; trimmed: { historyMessages: number; recalledFacts: number } }`.
  - `trimSummary(summary, budget)`: retorna `summary` se `estimateTokens(summary) <= budget`, senão `summary.slice(0, budget * 4)` (research.md item 3).
  - `trimWindow(window, budget)`: copia e remove `[0]` enquanto `estimateTokens(kept.map(m => m.content).join("\n")) > budget` e houver itens (research.md itens 4–5).
  - `trimMemories(memories, budget)`: `[...memories].sort((a, b) => b.score - a.score)` (estável) e remove o último enquanto `estimateTokens(kept.map(m => m.fact).join("\n")) > budget` (research.md item 6).
  - `buildContext(input, budget = DEFAULT_CONTEXT_BUDGET)`: aplica os três cortes de forma independente; `breakdown = buildContextBreakdown({ system, summary: keptSummary, currentMessage: message, historyTexts: keptWindow contents, factTexts: keptMemories facts })`; `trimmed` = diferença de tamanhos; `prompt` = blocos `[system, keptSummary ? "Resumo da conversa até aqui:\n" + keptSummary : "", composeWithFacts(keptFacts, composePrompt(keptWindow, message))]` filtrando vazios e unidos com `"\n\n"` (research.md item 7). JSDoc explicando que é o ponto único de montagem do `input` de todas as estratégias (research.md item 1).
- [X] T005 [US1] (depende de T004) Alterar `src/http/server.ts`: remover o uso direto de `composePrompt`/`composeWithFacts`/`buildContextBreakdown`; manter `RecallMatch` completos do `memoryStore.recall(...)` (fato + score) em vez de mapear só `fact`; montar `const built = buildContext({ message, window: history, memories: recalled }, DEFAULT_CONTEXT_BUDGET)`; passar `built.prompt` a `runWithTimeout(strategy, ...)`; `metrics.contextBreakdown = built.breakdown`. `reflectAndRemember` e `conversationStore.append` inalterados (continuam usando a mensagem original).
- [X] T006 [P] [US1] (depende de T004) Alterar `src/arena.ts`: `strategy.run(buildContext({ message: args.input }).prompt, { maxIterations: args.maxIterations })` (research.md item 10 — saída idêntica).
- [X] T007 [P] [US1] (depende de T004) Alterar `src/bench.ts` (`runOne`): `STRATEGIES[strategyName].run(buildContext({ message: scenario.input }).prompt, options)`.
- [X] T008 [US1] (depende de T005) Estender `src/http/server.test.ts`: (a) com estratégia fake que registra o `input` recebido, confirmar que sem corte o `input` é igual a `composeWithFacts(facts, composePrompt(history, message))` (regressão FR-011) para `strategy` `react` e `plan-and-execute` e com `reflect: true` — mesmo builder para todas; (b) com histórico grande o bastante para exceder 1200 tokens estimados (ex.: 12 mensagens de ~600 caracteres), confirmar que `contextBreakdown.conversationHistory ≤ 1200` e que o `input` recebido não contém as mensagens mais antigas.

**Checkpoint**: tetos padrão aplicados no `/chat`, arena e bench; `npm test` verde.

---

## Phase 4: User Story 2 - Ajustar os tetos sem mudar código (Priority: P2)

**Goal**: Tetos configuráveis por `CONTEXT_BUDGET_SUMMARY`/`_WINDOW`/`_MEMORIES`, validados com zod, com fallback por campo; `createApp` lê uma vez e aceita override.

**Independent Test**: `loadContextBudget` com envs válidas/inválidas/ausentes; `/chat` com `contextBudget` injetado aplica os tetos configurados.

### Tests for User Story 2

- [X] T009 [P] [US2] (depende de T004) Estender `src/context/context-builder.test.ts` com `loadContextBudget`: `{}` → `DEFAULT_CONTEXT_BUDGET`; `{ CONTEXT_BUDGET_WINDOW: "50" }` → `window: 50`, demais padrão; `"0"` → `0` (desliga a seção); `""`, `"abc"`, `"-5"`, `"1.5"` → padrão daquele campo (cada campo independente, ex.: um inválido e outro válido no mesmo `env`); nunca lança.

### Implementation for User Story 2

- [X] T010 [US2] (depende de T004) Adicionar a `src/context/context-builder.ts`: `export function loadContextBudget(env: Readonly<Record<string, string | undefined>>): ContextBudget` — schema zod por campo: `z.string().trim().min(1).pipe(z.coerce.number().int().min(0))`; `safeParse` de `env.CONTEXT_BUDGET_SUMMARY`/`_WINDOW`/`_MEMORIES`, usando o padrão de `DEFAULT_CONTEXT_BUDGET` em caso de falha ou ausência (research.md item 8 — string vazia nunca vira `0`). Pura: recebe `env` por parâmetro.
- [X] T011 [US2] (depende de T005, T010) Alterar `src/http/server.ts`: `CreateAppOptions` ganha `contextBudget?: ContextBudget` (JSDoc: "Sobrescreve os tetos de contexto — usado por testes; padrão `loadContextBudget(process.env)`"); em `createApp`, `const contextBudget = options.contextBudget ?? loadContextBudget(process.env)` (lido uma vez); `buildContext(..., contextBudget)` no handler.
- [X] T012 [US2] (depende de T011) Estender `src/http/server.test.ts`: `createApp({ contextBudget: { summary: 200, window: <baixo>, memories: 0 }, ... })` com histórico e `MemoryStore` fake retornando fatos → `contextBreakdown.conversationHistory ≤ window`, `contextBreakdown.recalledFacts === 0`, e o `input` recebido pela estratégia fake não contém os fatos nem as mensagens mais antigas.

**Checkpoint**: tetos ajustáveis por ambiente; `npm test` verde.

---

## Phase 5: User Story 3 - Saber o que foi cortado (Priority: P3)

**Goal**: `metrics.contextTrimmed = { historyMessages, recalledFacts }` em toda resposta `200`; `metrics.historyMessages` conta as mensagens efetivamente enviadas.

**Independent Test**: resposta de `/chat` com corte informa as quantidades removidas; sem corte, zeros.

### Tests for User Story 3

- [X] T013 [P] [US3] (depende de T011) Estender `src/http/server.test.ts`: (a) sem corte → `metrics.contextTrimmed` deep-equal `{ historyMessages: 0, recalledFacts: 0 }` e `historyMessages` igual ao número de mensagens carregadas; (b) com `contextBudget` baixo → `contextTrimmed.historyMessages > 0`, `contextTrimmed.recalledFacts` igual à quantidade de fatos removidos, e `historyMessages + contextTrimmed.historyMessages` igual ao número de mensagens carregadas. Atualizar os `deepEqual` existentes de `body.metrics` para incluir `contextTrimmed: { historyMessages: 0, recalledFacts: 0 }`.

### Implementation for User Story 3

- [X] T014 [US3] (depende de T011) Alterar `src/http/server.ts`: resposta passa a ter `metrics: { ...result.metrics, historyMessages: built.window.length, contextBreakdown: built.breakdown, contextTrimmed: built.trimmed }` (contrato [post-chat.md](./contracts/post-chat.md); research.md item 9).

**Checkpoint**: todas as user stories funcionais e testadas.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T015 Rodar `npm run typecheck` e `npm test` — ambos verdes (Princípio V).
- [X] T016 [P] Atualizar o JSDoc de `buildContextBreakdown` em `src/context/tokens.ts`, que hoje cita `composePrompt`/`composeWithFacts` como compositores, para apontar `buildContext` (`src/context/context-builder.ts`) como o compositor atual.
- [ ] T017 Validar manualmente os cenários 2–6 de [quickstart.md](./quickstart.md) (requer `OPENROUTER_API_KEY` já presente no ambiente; nunca ler `.env`). Pendente: a API key não está configurada nesta execução.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: vazio.
- **Foundational (Phase 2)**: T001 → T002. Bloqueia todas as stories.
- **US1 (Phase 3)**: depende da Phase 2. T003 ∥ T004; T005, T006, T007 dependem de T004; T008 depende de T005.
- **US2 (Phase 4)**: T009/T010 dependem só de T004 (podem começar junto com o resto da US1); T011 depende de T005 + T010; T012 depende de T011.
- **US3 (Phase 5)**: depende de T011 (usa `contextBudget` injetado nos testes).
- **Polish (Phase 6)**: depois de todas as stories.

### User Story Dependencies

- **US1 (P1)**: independente — entrega o MVP com os tetos padrão.
- **US2 (P2)**: estende o builder da US1 com a leitura de env; testável sozinha via `loadContextBudget`.
- **US3 (P3)**: só reporta o `trimmed` que o builder da US1 já calcula; precisa do override da US2 para os testes HTTP de corte.

### Within Each User Story

- Testes escritos primeiro e falhando antes da implementação.
- `context-builder.ts` antes de `server.ts`/`arena.ts`/`bench.ts`.
- Cada tarefa cabe em um commit (Princípio VIII).

## Parallel Opportunities

- **US1**: T003 (teste) ∥ T004 (builder); depois T006 (arena) ∥ T007 (bench) ∥ T005 (server).
- **US2**: T009 ∥ T010 podem rodar em paralelo com T005–T008 (arquivos e funções diferentes de `server.ts`).
- **Polish**: T016 ∥ T015.

### Parallel Example: User Story 1

```bash
# Depois de T004:
Task: "T006 [US1] arena.ts usa buildContext"
Task: "T007 [US1] bench.ts usa buildContext"
Task: "T005 [US1] server.ts usa buildContext"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 2 (T001–T002).
2. Phase 3 (T003–T008) — tetos padrão aplicados para todas as estratégias.
3. **STOP and VALIDATE**: teste-âncora "tetos baixos cortam na ordem certa" verde + regressão FR-011.

### Incremental Delivery

1. Foundational → US1 (MVP: tetos fixos) → US2 (tetos por env) → US3 (métricas de corte) → Polish.
2. Cada story é aditiva; com tetos padrão o comportamento observável só muda quando há corte real.
