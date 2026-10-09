# Data Model: Grafo Unificado com Roteador de Estratégia

> **Nota de implementação (nomes finais)**: seguindo o esboço passado no `/speckit-implement`, o código usa `src/graph/production-graph.ts` e `src/graph/router.ts` (não `src/agents/`); nós `contexto`, `roteador`, `react`, `planExecute`, `reflect`, `resposta`; rotas `react | planExecute | reflect`. O `strategy` do `/chat` aceita também os nomes legados `plan-and-execute` e `reflection` como aliases. Onde este documento usa os nomes antigos (`context`/`router`/`answer` como nós, `plan-and-execute`/`reflection` como rotas), leia com esse mapeamento.

Esta feature não tem persistência nova: tudo é em memória, durante uma requisição. Os tipos novos e alterados vivem em `src/agents/types.ts`, a menos que haja indicação diferente.

## RouteName

`"react" | "plan-and-execute" | "reflection"`

- É o conjunto fechado das rotas que o roteador pode escolher e dos valores aceitos no override do `/chat`.
- `parseRouteName(name: string): RouteName` lança `UnknownStrategyError` para qualquer outro valor (o 422 atual continua igual).

## RouteSource

`"router" | "override" | "fallback"`

| Valor | Quando |
|---|---|
| `router` | O modelo devolveu uma decisão válida pelo schema. |
| `override` | O `/chat` recebeu um `strategy` válido. O roteador não é chamado. |
| `fallback` | O roteador lançou erro, devolveu `null` ou devolveu uma decisão inválida. Nesse caso a rota é sempre `react`. |

## RouteDecision

| Campo | Tipo | Regra |
|---|---|---|
| `route` | `RouteName` | obrigatório |
| `reason` | `string` | não vazio (`min(1)` após trim) |
| `source` | `RouteSource` | obrigatório |

- O schema zod de saída do modelo (`RouteDecisionSchema`) cobre só `{ route, reason }`. O `source` é atribuído por `resolveRouteDecision`, que é uma função pura.

## GraphNode

`"context" | "router" | "react" | "plan-and-execute" | "reflection" | "answer"`

Um `GraphNode` de estratégia tem o mesmo nome que a `RouteName` correspondente.

## TraceEvent (alterado)

- Nova variante: `{ type: "route"; at: number; route: RouteName; reason: string; source: RouteSource; node?: GraphNode }`.
- Todas as variantes existentes (`thought`, `action`, `observation`, `plan`, `critique`, `answer`) ganham `node?: GraphNode`.
- `ProductionTraceEvent = TraceEvent & { node: GraphNode }` é o tipo devolvido pelo grafo. Nele `node` é sempre presente.

### Invariantes do trace do grafo

1. Há exatamente um evento `route`, com `node: "router"`.
2. O evento `route` vem antes de qualquer evento com `node` de estratégia.
3. Todos os eventos depois do `route` têm o mesmo `node`, igual a `decision.route`. Só uma estratégia roda.
4. Os valores de `at` são `0..n-1`, sequenciais e na ordem do array.

## ProductionGraphState (estado do LangGraph)

| Campo | Tipo | Escrito por |
|---|---|---|
| `contextInput` | `ContextInput` | entrada |
| `budget` | `ContextBudget` | entrada |
| `override` | `RouteName \| undefined` | entrada |
| `built` | `BuiltContext` | `context` |
| `decision` | `RouteDecision` | `router` |
| `routerUsage` | `{ llmCalls: number; tokenUsage?: TokenUsage }` | `router` (0 chamadas no override) |
| `strategyResult` | `RunResult` | nó de estratégia |
| `trace` | `ProductionTraceEvent[]` (reducer concat) | `router`, nó de estratégia |
| `result` | `ProductionRunResult` | `answer` |

## ProductionRunResult

`RunResult & { trace: ProductionTraceEvent[]; route: RouteDecision; context: BuiltContext }`

- `metrics.llmCalls` = chamadas do roteador + chamadas da estratégia.
- `metrics.promptTokens` / `metrics.tokenSource` = `mergeTokenUsage(roteador, estratégia)`.
- `metrics.latencyMs` mede o grafo inteiro.

## Transições

```text
START → context → router ─┬─ route=react            → react            ─┐
                          ├─ route=plan-and-execute → plan-and-execute ─┼→ answer → END
                          └─ route=reflection       → reflection       ─┘
```
