# Implementation Plan: Grafo Unificado com Roteador de Estratégia

> **Nota de implementação (nomes finais)**: seguindo o esboço passado no `/speckit-implement`, o código usa `src/graph/production-graph.ts` e `src/graph/router.ts` (não `src/agents/`); nós `contexto`, `roteador`, `react`, `planExecute`, `reflect`, `resposta`; rotas `react | planExecute | reflect`. O `strategy` do `/chat` aceita também os nomes legados `plan-and-execute` e `reflection` como aliases. Onde este documento usa os nomes antigos (`context`/`router`/`answer` como nós, `plan-and-execute`/`reflection` como rotas), leia com esse mapeamento.

**Branch**: `012-unified-graph` | **Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/012-unified-graph/spec.md`

## Summary

O `/chat` passa a executar um único grafo LangGraph, `src/agents/production-graph.ts`, com os nós `context` → `router` → (`react` | `plan-and-execute` | `reflection`) → `answer`.

**Roteador.** O `router` decide a rota com `withStructuredOutput({ route, reason })` a partir de um prompt que contém a tabela de rotas. Quando o cliente informa `strategy`, ela vira override: o modelo não é consultado. Se o roteador falhar ou devolver algo inválido, o fluxo cai em `react` (fallback).

**Trace.** O trace ganha um evento `route` (rota, motivo, origem), e todo evento passa a carregar `node`.

**Estratégias.** As três estratégias existentes são reaproveitadas sem alteração. Os nós do grafo apenas as embrulham, carimbam `node`, reindexam `at` e somam as métricas.

As regras de decisão e de trace ficam em funções puras testadas sem rede (ver [research.md](./research.md)).

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS

**Primary Dependencies**: `@langchain/langgraph` ^0.2 (`StateGraph`, `addConditionalEdges`), `@langchain/core` ^0.3, `@langchain/openai` ^0.3 (OpenRouter via `createModel()`), zod ^3.23, Express ^4.19

**Storage**: N/A, sem persistência nova (histórico e memórias continuam no SQLite já existente, carregados pelo controller)

**Testing**: `node:test` via `tsx` (`npm test`) e `tsc --noEmit` (`npm run typecheck`); roteador e estratégias fake injetados, sem rede

**Target Platform**: servidor Node (API Express local)

**Project Type**: web-service (API única + scripts CLI arena/bench)

**Performance Goals**: no máximo +1 chamada ao modelo por resposta por causa do roteamento (SC-006); 0 chamadas extras no override

**Constraints**: o timeout de 180 s cobre o grafo inteiro; o formato da resposta muda só por acréscimo (SC-007); o contexto montado é idêntico ao da 011

**Scale/Scope**: 1 grafo, 6 nós, 3 rotas; cerca de 2 arquivos novos (`router.ts`, `production-graph.ts`) e 5 alterados

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Status | Como |
|---|---|---|
| I. Camadas explícitas | ✅ | O IO (histórico, memórias) continua no controller/service. O grafo recebe dados já carregados. O nó `context` só chama a função pura `buildContext`. |
| II. Validação na fronteira | ✅ | O `strategy` do `/chat` é validado no controller (`parseRouteName`, que gera 422). A saída do modelo é validada com o zod `RouteDecisionSchema` antes de virar `RouteDecision`. |
| III. Erros de domínio | ✅ | Reaproveita `UnknownStrategyError` e `ChatTimeoutError`. A falha do roteador não é erro (vira fallback). A tradução para HTTP continua só no middleware. |
| IV. Funções puras | ✅ | `buildRouterMessages`, `resolveRouteDecision`, `parseRouteName`, `tagTrace` e `formatTraceEvent` são puras. O IO se limita a `createModelRouter` e às estratégias já existentes. |
| V. Teste obrigatório | ✅ | Suítes novas: `router.test.ts` e `production-graph.test.ts`. Suítes estendidas: `trace.test.ts` e `server.test.ts`. |
| VI. Segurança | ✅ | Nenhum segredo novo e `.env` não é lido. O roteador não amplia as tools disponíveis: só escolhe entre estratégias já existentes. |
| VII. Spec antes de código | ✅ | spec → plan (este documento) → tasks. |
| VIII. Pequeno e reversível | ✅ | Os incrementos são independentes: tipos/trace → roteador → grafo → controller. As estratégias existentes ficam intocadas. |
| Stack | ✅ | Usa só LangGraph/LangChain/zod/Express, que já estão na stack. |

**Re-check pós-design**: ✅ nenhum princípio violado e nenhuma entrada em Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/012-unified-graph/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── post-chat.md
│   └── production-graph.md
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```text
src/
├── agents/
│   ├── types.ts                  # ALTERADO: RouteName, RouteSource, RouteDecision, GraphNode, evento "route", node?, ProductionTraceEvent
│   ├── trace.ts                  # ALTERADO: formata "route" + prefixo de node; tagTrace (pura)
│   ├── trace.test.ts             # ALTERADO
│   ├── router.ts                 # NOVO: ROUTE_TABLE, RouteDecisionSchema, buildRouterMessages, resolveRouteDecision, parseRouteName, createModelRouter
│   ├── router.test.ts            # NOVO
│   ├── production-graph.ts       # NOVO: StateGraph context → router → {react|plan-and-execute|reflection} → answer
│   ├── production-graph.test.ts  # NOVO
│   ├── index.ts                  # ALTERADO: strategyForRoute (reflection = withReflection(react))
│   └── react.ts / plan-and-execute.ts / reflection.ts   # sem mudança
├── services/
│   └── chat.service.ts           # ALTERADO: withTimeout<T> genérico; runWithTimeout delega para ele
└── http/
    ├── server.ts                 # ALTERADO: valida override, executa o grafo, expõe `route`; opção decideRoute para testes
    └── server.test.ts            # ALTERADO
```

**Structure Decision**: o projeto único segue o layout existente em `src/`. O grafo e o roteador ficam em `src/agents/`, junto das estratégias. O controller HTTP continua sendo o único ponto que faz IO de conversa/memória e que traduz erros.

## Complexity Tracking

Sem violações.
