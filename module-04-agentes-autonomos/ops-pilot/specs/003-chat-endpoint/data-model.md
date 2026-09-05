# Data Model: Endpoint HTTP de Chat

Entidades derivadas do [spec.md](./spec.md) (seção Key Entities) e das decisões de [research.md](./research.md). Nenhuma destas entidades é persistida — são construções efêmeras de uma única requisição HTTP. `RunResult`, `ReasoningStrategy`, `TraceEvent` e `Metrics` já existem (feature 001, [../001-reasoning-strategies-core/data-model.md](../001-reasoning-strategies-core/data-model.md)) e são reaproveitados sem alteração. Campos e tipos são descritos de forma independente de implementação (a codificação TypeScript/zod concreta é detalhe de tarefa).

## ChatRequest (corpo de `POST /chat`, validado na fronteira)

| Campo | Tipo | Regras |
|---|---|---|
| `message` | string | Obrigatório, não vazio (FR-001, FR-002). A pergunta/instrução operacional enviada ao raciocínio. |
| `strategy` | string | Opcional. Quando ausente, o sistema usa a estratégia padrão (`react`, FR-003). Quando presente e não corresponde a nenhuma estratégia registrada, a requisição é rejeitada antes de qualquer execução (FR-004). |
| `reflect` | boolean | Opcional, padrão `false`. Quando `true`, a estratégia resolvida (padrão ou explícita) é executada envolvida pela camada de reflection já existente, em vez da estratégia base isolada (FR-006). |

Corpo que não seja um objeto JSON válido, ou que viole qualquer uma das regras acima, é rejeitado por inteiro — nenhum campo é processado parcialmente (FR-002).

## ChatResponse (corpo de uma resposta `200`)

É o `RunResult` já definido pela feature 001 — nenhum campo novo, nenhuma transformação:

| Campo | Tipo | Regras |
|---|---|---|
| `answer` | string | Resposta final produzida pela estratégia resolvida (com ou sem reflection) — FR-007. |
| `trace` | TraceEvent[] | Histórico completo de eventos de raciocínio da execução, na mesma forma já produzida por `react.ts`/`plan-and-execute.ts`/`reflection.ts` — inclui eventos `critique` quando `reflect: true` (FR-007). |
| `metrics` | Metrics | `{ llmCalls, latencyMs }` da execução completa — com `reflect: true`, já inclui as chamadas extras do ciclo de crítica, permitindo comparação direta com uma execução sem reflection (FR-007, SC-005). |

## Erro de resposta (corpo de uma resposta `400`, `422`, `504` ou `500`)

Não é uma entidade de domínio nova — é o formato usado pelo controller para comunicar cada categoria de falha (ver taxonomia completa em [research.md](./research.md#7-taxonomia-de-erro--status-http)):

| Campo | Presente em | Descrição |
|---|---|---|
| `error` | todos | Identificador curto e estável da categoria (`"invalid_body"`, `"unknown_strategy"`, `"timeout"`, `"internal_error"`) — permite ao cliente diferenciar programaticamente a causa (FR-009). |
| `issues` | `400` | Lista de problemas de validação do zod (campo, mensagem), permitindo ao cliente corrigir exatamente o que falhou (FR-002, SC-002). |
| `strategy` | `422` | Nome da estratégia desconhecida que foi solicitada (SC-003). |
| `timeoutMs` | `504` | Valor do limite de tempo configurado que foi ultrapassado (FR-008, SC-004). |

`500` não expõe nenhum detalhe interno além do identificador genérico — a causa completa fica apenas no log do servidor (Principle VI da constitution).

## Estratégia de Raciocínio Registrada

Já modelada pela feature 001 (`ReasoningStrategy`: `{ name, run(input, options) }`). Esta feature não estende o tipo — apenas adiciona um mapa nome→estratégia (`react`, `plan-and-execute`) e uma função de resolução que, dado um nome opcional e um flag de reflection, produz a `ReasoningStrategy` a ser executada (base ou decorada com `withReflection`, feature 002).

## Relacionamentos

```text
ChatRequest { message, strategy?, reflect? }
        │
        ▼ (validação zod na fronteira — controller)
        │
        ▼ resolveStrategy(strategy, reflect)              ──throws──> UnknownStrategyError (422)
        │
        ▼ ReasoningStrategy (base ou "reflect:" + base)
        │
        ▼ runWithTimeout(strategy, message, timeoutMs)     ──throws──> ChatTimeoutError (504)
        │
        ▼ RunResult { answer, trace, metrics }
        │
        ▼ ChatResponse (200) — RunResult sem transformação
```

### Transições de erro

```text
corpo inválido           ──> 400 (nunca chega a resolveStrategy)
estratégia desconhecida  ──> 422 (nunca chega a runWithTimeout)
timeout                  ──> 504 (execução em andamento é descartada do ponto de vista do cliente)
qualquer outra falha     ──> 500 (mensagem genérica)
sucesso                  ──> 200 { answer, trace, metrics }
```
