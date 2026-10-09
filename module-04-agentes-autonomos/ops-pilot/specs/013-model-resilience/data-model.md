# Data Model: Resiliência de Modelo

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**: a fábrica se chama `createModel(build?, config?, deps?)` (não `resilientRunnable`), e o modelo base se chama `baseModel(id)` (não `createChatModel`). São **`stopAfterAttempt: 2`** (1 tentativa + 1 retry) **no principal e também na reserva**, compostos como `primary.withFallbacks([backup])`. Os números de tentativas citados abaixo (3 no principal, sem retry na reserva) foram substituídos por esses.

Esta feature não tem persistência. Tudo é em memória, durante uma requisição ou execução.

## ModelConfig (`src/agents/model.ts`)

| Campo | Tipo | Origem | Regra |
|---|---|---|---|
| `primary` | `string` | `OPENROUTER_MODEL` | obrigatório (trim, min 1); sem ele, o erro é o mesmo de hoje |
| `fallback` | `string \| undefined` | `OPENROUTER_MODEL_FALLBACK` | trim; vazio, ausente ou `=== primary` vira `undefined` |

## ModelCallLog (`src/agents/model-usage.ts`)

Lista de entradas que o `ModelUsageTracker` registra, na ordem dos callbacks:

```ts
type ModelCallEntry =
  | { kind: "start"; runId: string; model: string }
  | { kind: "end"; runId: string }
  | { kind: "error"; runId: string; error: string };
```

`end` e `error` herdam o `model` do `start` que tem o mesmo `runId`.

## ModelFallback

| Campo | Tipo | Regra |
|---|---|---|
| `from` | `string` | sempre o `primary` |
| `to` | `string` | sempre o `fallback` |
| `reason` | `string` | resumo do último erro do principal antes da troca, truncado em 200 caracteres e nunca vazio |

## ModelUsageSummary

`summarizeModelUsage(log, config): { fallbacks: ModelFallback[]; modelUsed: string }`. É pura e segue estas regras:

1. Um `start` no `fallback` precedido por pelo menos um `error` no `primary` desde o último `start` no `fallback` gera 1 `ModelFallback`.
2. Vários `error` seguidos de `start` no `primary` (retry) **não** geram fallback.
3. `modelUsed` é o modelo do último `end`. Se não houver nenhum `end`, é `config.primary`.

## TraceEvent (alterado, `src/agents/types.ts`)

Nova variante: `{ type: "fallback"; at: number; from: string; to: string; reason: string; node?: GraphNode }`.

**Posição no trace de uma estratégia**: os eventos `fallback` entram imediatamente antes do último evento `answer`, ou no fim quando não há `answer`. Os valores de `at` são reindexados (`withModelFallbacks`, pura).

**Invariantes do grafo de produção (ajuste da 012)**:

1. `trace[0]` é o único `route` e tem `node: "roteador"`. Sem mudança.
2. Fallbacks do roteador vêm logo depois do `route`, com `node: "roteador"`.
3. Os demais eventos têm `node` igual a `route.route`. Isso inclui os `fallback` que aconteceram dentro da estratégia.
4. Os valores de `at` vão de `0` a `n-1`.

## Metrics (alterado)

`+ modelUsed: string`, **obrigatório**.

| Origem | `modelUsed` |
|---|---|
| react / plan-and-execute | `summarizeModelUsage(tracker.log, config).modelUsed` |
| reflection | o `modelUsed` da última tentativa |
| grafo de produção | o `modelUsed` da estratégia (o roteador não conta) |
| `/chat` | repassa o valor do grafo |

`formatMetrics`: `llmCalls=N latencyMs=N model=<id>`.

## DecideRoute (alterado, `src/graph/router.ts`)

Retorno: `{ decided: unknown; tokenUsage: TokenUsage; fallbacks: ModelFallback[] }`. Os fakes de teste devolvem `fallbacks: []`.
