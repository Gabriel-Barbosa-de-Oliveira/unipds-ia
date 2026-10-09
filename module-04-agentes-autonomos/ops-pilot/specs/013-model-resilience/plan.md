# Implementation Plan: Resiliência de Modelo (retry + modelo de reserva)

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**: a fábrica se chama `createModel(build?, config?, deps?)` (não `resilientRunnable`), e o modelo base se chama `baseModel(id)` (não `createChatModel`). São **`stopAfterAttempt: 2`** (1 tentativa + 1 retry) **no principal e também na reserva**, compostos como `primary.withFallbacks([backup])`. Os números de tentativas citados abaixo (3 no principal, sem retry na reserva) foram substituídos por esses.

**Branch**: `013-model-resilience` | **Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/013-model-resilience/spec.md`

## Summary

A fábrica de modelo (`src/agents/model.ts`) passa a montar todo runnable de modelo como `build(principal).withRetry(...).withFallbacks([build(reserva)])`.

**Retry.** São no máximo 3 tentativas, e só para erros transitórios.

**Reserva.** O modelo de reserva vem de `OPENROUTER_MODEL_FALLBACK`. Sem ela, o fluxo usa só o principal com retry.

**Ordem de composição.** O `build` (`withStructuredOutput` ou `bindTools`) é aplicado em cada ramo **antes** da composição, porque `withFallbacks` não tem esses métodos. Para o `createReactAgent`, o resultado é embrulhado num `RunnableBinding` com `kwargs.tools`, para que o agente não tente religar as tools.

**Trace e métricas.** Um `ModelUsageTracker` (callback) registra as chamadas, e a função pura `summarizeModelUsage` gera os eventos `fallback` do trace e o `metrics.modelUsed`. Os 6 call sites passam a usar a fábrica (ver [research.md](./research.md)).

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS

**Primary Dependencies**: `@langchain/core` 0.3.80 (`withRetry`, `withFallbacks`, `RunnableBinding`, `BaseCallbackHandler`), `@langchain/openai` 0.3.17 (`ChatOpenAI`, com `maxRetries: 0`), `@langchain/langgraph` 0.2.74 (`createReactAgent`), zod

**Storage**: N/A

**Testing**: `node:test` via `tsx`. Modelos fake com falhas por roteiro (subclasse de `FakeStreamingChatModel`/`BaseChatModel` de `@langchain/core/utils/testing`), sem rede.

**Target Platform**: servidor Node (API Express) e os scripts arena/bench

**Project Type**: web-service

**Performance Goals**: o caminho feliz não muda (1 chamada, sem espera extra; SC-005). O pior caso antes da reserva fica em ≤ 10 s (2 retries com backoff de ~1 s e ~2 s; SC-004).

**Constraints**: o `.env` nunca é lido pelo código (FR-012). A configuração vem do `process.env` injetado pelos scripts npm. O teto de 180 s continua valendo.

**Scale/Scope**: 2 módulos novos ou ampliados (`model.ts`, `model-usage.ts`) e 6 call sites migrados; `types.ts`, `trace.ts`, `metrics.ts`, `router.ts` e `production-graph.ts` também mudam

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Status | Como |
|---|---|---|
| I. Camadas explícitas | ✅ | A fábrica de modelo é infraestrutura de agente. Nenhum IO novo entra no domínio. |
| II. Validação na fronteira | ✅ | As variáveis de ambiente são validadas com zod em `loadModelConfig`. |
| III. Erros de domínio | ✅ | Não há erro novo. A falha da reserva propaga como hoje (500/504 na borda). A falha do roteador continua no fallback de rota da 012. |
| IV. Funções puras | ✅ | `loadModelConfig`, `isTransientModelError`, `summarizeModelUsage`, `withModelFallbacks` e `formatTraceEvent` são puras. O IO fica no `ChatOpenAI` e no callback. |
| V. Teste obrigatório | ✅ | Suítes novas: `model.test.ts` e `model-usage.test.ts`. Suítes estendidas: trace, grafo e server. Tudo sem rede. |
| VI. Segurança | ✅ | Nenhum segredo novo. A reserva usa a mesma `OPENROUTER_API_KEY`. O `.env` não é lido. |
| VII. Spec antes de código | ✅ | spec → plan → tasks. |
| VIII. Pequeno e reversível | ✅ | Os incrementos são: config/classificação puras → fábrica → tracker → migração de cada call site → trace/métricas. |
| Stack | ✅ | Só primitivos do LangChain que já estão na stack. Nenhuma dependência nova. |

**Re-check pós-design**: ✅ nenhum princípio violado. O `RunnableBinding` com `kwargs.tools` (research.md item 4) é um ajuste localizado de compatibilidade, coberto por teste, e não viola nenhum princípio.

## Project Structure

### Documentation (this feature)

```text
specs/013-model-resilience/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── model-factory.md
│   └── post-chat.md
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```text
src/
├── agents/
│   ├── model.ts               # REESCRITO: loadModelConfig, isTransientModelError, createChatModel, resilientRunnable, toolCallingModel
│   ├── model.test.ts          # NOVO
│   ├── model-usage.ts         # NOVO: ModelUsageTracker, summarizeModelUsage, withModelFallbacks
│   ├── model-usage.test.ts    # NOVO
│   ├── types.ts               # ALTERADO: evento "fallback", Metrics.modelUsed
│   ├── trace.ts / trace.test.ts   # ALTERADO: formatação de fallback, model= em formatMetrics
│   ├── metrics.ts             # ALTERADO: buildMetrics(..., modelUsed)
│   ├── react.ts               # ALTERADO: toolCallingModel + tracker
│   ├── plan-and-execute.ts    # ALTERADO: resilientRunnable (planner/replanner), toolCallingModel (executor) + tracker
│   └── reflection.ts          # ALTERADO: crítico via resilientRunnable + tracker; modelUsed da última tentativa
├── graph/
│   ├── router.ts              # ALTERADO: createModelRouter via resilientRunnable; DecideRoute devolve fallbacks
│   └── production-graph.ts(+test)  # ALTERADO: eventos fallback do roteador após o route; modelUsed da estratégia
├── memory/
│   └── learning-reflector.ts  # ALTERADO: resilientRunnable
└── http/server.test.ts        # ALTERADO: fixtures com modelUsed; fakes de DecideRoute com fallbacks: []
```

**Structure Decision**: o layout atual se mantém. O registro de uso de modelo fica num módulo separado (`model-usage.ts`) para que `model.ts` contenha só a montagem dos runnables.

## Complexity Tracking

Sem violações.
