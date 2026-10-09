# Implementation Plan: Orçamento de Contexto por Seção

**Branch**: `010-context-budget` | **Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/010-context-budget/spec.md`

## Summary

Cria `src/context/context-builder.ts`, o ponto único que monta o `input` entregue a `ReasoningStrategy.run(input)` — e, portanto, o prompt de **todas** as estratégias (react, plan-and-execute, com/sem reflection), já que todas recebem esse mesmo `input` (research.md item 1). O builder recebe as seções brutas (system, resumo, janela de mensagens, memórias com score, mensagem atual) e um `ContextBudget` (`summary`/`window`/`memories`, em tokens estimados por `estimateTokens` da `009`), aplica os cortes com funções puras — resumo truncado mantendo o início, janela perdendo mensagens inteiras a partir da mais antiga, memórias perdendo itens inteiros a partir do menor score — e compõe o texto final **reaproveitando** `composePrompt`/`composeWithFacts` (`006`/`007`), de modo que sem corte (e sem system/resumo) a string é byte a byte a mesma de hoje (FR-011). Os tetos vêm de `CONTEXT_BUDGET_SUMMARY`/`_WINDOW`/`_MEMORIES`, lidos uma vez em `createApp` por `loadContextBudget(env)` (zod, valor inválido → padrão 200/1200/300). O controller `/chat` passa a delegar a composição ao builder e devolve em `metrics` o `contextBreakdown` pós-corte (agora com `system`/`summary`) e `contextTrimmed` (mensagens/memórias removidas). `arena.ts`/`bench.ts` passam a usar o mesmo builder (só com a mensagem — saída idêntica).

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS (sem mudança de runtime)

**Primary Dependencies**: nenhuma nova — `zod` (já usado na fronteira) para validar as variáveis de ambiente; `estimateTokens`/`ContextBreakdown` de `src/context/tokens.ts` (`009`); `composePrompt` (`src/domain/conversation.ts`) e `composeWithFacts` (`src/domain/memory.ts`).

**Storage**: N/A — nenhuma tabela nova; o builder opera sobre dados já carregados (`ConversationStore.lastMessages`, `MemoryStore.recall`).

**Testing**: `node:test` via `tsx`. Novo: `src/context/context-builder.test.ts` — `loadContextBudget` (ausente/válido/inválido/vazio/negativo/zero), `trimSummary`, `trimWindow` (corta as mais antigas, preserva ordem, mensagem única maior que o teto → janela vazia), `trimMemories` (corta menor score, desempate estável), `buildContext` (system/mensagem intocáveis mesmo enormes; sem corte ⇒ string idêntica a `composeWithFacts(facts, composePrompt(history, msg))`; **tetos baixos cortam na ordem certa** — cenário pedido pelo usuário; seções independentes, sem "empréstimo"). Estendido: `src/http/server.test.ts` (asserts de `contextBreakdown` ganham `system`/`summary`; novos cenários com `contextBudget` baixo injetado via `createApp` verificando `contextTrimmed` e o `input` recebido pela estratégia fake).

**Target Platform**: processo Node.js server-side (`npm run dev`); `arena`/`bench` também passam pelo builder, sem mudança de saída.

**Project Type**: projeto único — 1 arquivo novo de produção em `src/context/` (pedido explicitamente: `context-builder.ts`) + 1 de teste.

**Performance Goals**: cortes são O(n) sobre ≤ 12 mensagens e ≤ 3 memórias, em memória — custo desprezível frente à chamada ao modelo.

**Constraints**: system e mensagem atual nunca cortados (FR-003); cada seção cortável ≤ seu teto, medido com a mesma convenção do `contextBreakdown` (FR-008, research.md item 4); determinístico (FR-009); sem corte ⇒ saída idêntica à atual (FR-011); env inválida nunca derruba requisição (US2 cenário 3).

**Scale/Scope**: 1 módulo novo; `server.ts` troca 2 linhas de composição por 1 chamada ao builder e passa a manter `score` das memórias; `ContextBreakdown` ganha 2 partes (`system`, `summary`); `metrics` ganha `contextTrimmed`; `CreateAppOptions` ganha `contextBudget?`; `arena.ts`/`bench.ts` ganham 1 linha cada.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | `context-builder.ts` é lógica pura sem IO; o controller (`http/server.ts`) busca histórico/memórias (IO via stores, como hoje) e só delega a composição ao builder. Estratégias continuam recebendo uma `string` — não conhecem o builder. |
| II. Validação na Fronteira | Variáveis de ambiente são entrada externa: `loadContextBudget` valida cada `CONTEXT_BUDGET_*` com zod (inteiro ≥ 0), caindo no padrão quando inválido. `ChatRequestSchema` inalterado. |
| III. Erros de Domínio | Nenhum erro novo: estourar teto não é falha, é o caso que dispara o corte; env inválida cai no padrão (decisão explícita da spec, US2 cenário 3). |
| IV. Funções Puras | `loadContextBudget(env)`, `trimSummary`, `trimWindow`, `trimMemories` e `buildContext` são puras — recebem `env`/dados por parâmetro (nunca leem `process.env` diretamente); a leitura de `process.env` fica em `createApp`. |
| V. Teste Obrigatório | `context-builder.test.ts` cobre toda a lógica, incluindo o teste pedido ("tetos baixos cortam na ordem certa"); `server.test.ts` cobre o contrato. Tudo offline. |
| VI. Segurança por Padrão | Nada lê `.env`; só `process.env` já carregado pelo processo. Nenhum dado novo exposto além de contagens. |
| VII. Spec Antes de Código | Segue `specs/010-context-budget/spec.md`, sem `[NEEDS CLARIFICATION]`. |
| VIII. Pequeno e Reversível | Módulo isolado + troca pontual no controller; com tetos altos o comportamento é o atual. Mudança de contrato aditiva (`contextBreakdown.system/summary`, `contextTrimmed`). |

Nenhuma violação — **Complexity Tracking** não se aplica.

**Re-check pós-design (Phase 1)**: data-model e contratos mantêm todas as funções puras e a leitura de env restrita a `createApp`; nenhuma violação nova.

## Project Structure

### Documentation (this feature)

```text
specs/010-context-budget/
├── plan.md              # This file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   ├── context-builder.md
│   └── post-chat.md
└── tasks.md             # Phase 2 (/speckit-tasks — NÃO criado aqui)
```

### Source Code (repository root)

```text
src/
├── context/
│   ├── tokens.ts                     # [alterado] ContextBreakdown ganha system/summary; buildContextBreakdown aceita as novas partes (opcionais, default vazio)
│   ├── tokens.test.ts                # [alterado] cobre as novas partes no total
│   ├── context-builder.ts            # [NOVO] ContextBudget, DEFAULT_CONTEXT_BUDGET, loadContextBudget, trimSummary, trimWindow, trimMemories, buildContext
│   └── context-builder.test.ts       # [NOVO] inclui "tetos baixos cortam na ordem certa"
├── http/
│   ├── server.ts                     # [alterado] CreateAppOptions.contextBudget; mantém score do recall; usa buildContext; metrics.contextTrimmed
│   └── server.test.ts                # [alterado] breakdown com system/summary; cenários de corte via contextBudget injetado
├── arena.ts, bench.ts                # [alterado] input = buildContext({ message }).prompt (saída idêntica)
└── domain/, agents/, memory/, store/, services/, mcp/  # [inalterados]
```

**Structure Decision**: Projeto único, mesma estrutura das features 001–009. O builder fica em `src/context/` (pedido explícito e coeso com `tokens.ts`), reaproveitando `composePrompt`/`composeWithFacts` em vez de duplicar formatação.

## Complexity Tracking

Sem violações a justificar.
