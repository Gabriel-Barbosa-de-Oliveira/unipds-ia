# Contract: registro de estratégias e execução com timeout

Definidos em `src/agents/index.ts` e `src/services/chat.service.ts`. Assinaturas ilustrativas (a codificação final de tipos é detalhe de tarefa):

```ts
import type { ReasoningStrategy, RunOptions, RunResult } from "./types.ts";

// src/agents/index.ts
const STRATEGIES: Record<"react" | "plan-and-execute", ReasoningStrategy>;
const DEFAULT_STRATEGY_NAME: "react";

// Lança UnknownStrategyError quando `name` é informado e não está em STRATEGIES.
// Nunca faz IO — apenas consulta o registro estático e, se reflect, decora com withReflection (já existente).
function resolveStrategy(name: string | undefined, reflect: boolean | undefined): ReasoningStrategy;

// src/services/chat.service.ts
// Lança ChatTimeoutError se strategy.run(...) não resolver dentro de timeoutMs.
// Sempre limpa o timer interno (sucesso, erro da estratégia, ou timeout) — nenhum handle solto.
function runWithTimeout(
  strategy: ReasoningStrategy,
  input: string,
  options: RunOptions | undefined,
  timeoutMs: number,
): Promise<RunResult>;
```

## Regras do contrato

- `resolveStrategy(undefined, reflect)` DEVE se comportar como `resolveStrategy(DEFAULT_STRATEGY_NAME, reflect)` (FR-003).
- `resolveStrategy(name, reflect)` com `name` fora de `STRATEGIES` DEVE lançar `UnknownStrategyError(name)` **de forma síncrona** (antes de qualquer `await`) — o controller nunca precisa esperar IO para descobrir que a estratégia é desconhecida (FR-004).
- `resolveStrategy(name, true)` DEVE retornar `withReflection(STRATEGIES[name ?? DEFAULT_STRATEGY_NAME])` — mesma composição já usada por `src/arena.ts` para suas entradas `reflect:*`, sem duplicar a lógica de `withReflection` em si (reaproveitada da feature 002).
- `resolveStrategy` é pura: chamadas repetidas com os mesmos argumentos retornam uma estratégia equivalente (mesmo `name`), sem efeito colateral observável e sem depender de `process.env` ou de rede.
- `runWithTimeout` é a única função desta feature que introduz um efeito de tempo real (`setTimeout`); ela DEVE limpar o timer em todos os desfechos (sucesso, erro propagado pela estratégia, ou timeout) para nunca manter um handle aberto além do necessário — relevante tanto em produção quanto para os testes de integração não travarem/vazarem timers.
- `runWithTimeout` NUNCA tenta cancelar `strategy.run(...)` em andamento quando o timeout vence — apenas para de aguardá-la do lado do chamador (ver research.md item 2). Qualquer erro que `strategy.run(...)` lance **depois** do timeout já ter vencido é ignorado (a promise retornada por `runWithTimeout` já foi rejeitada com `ChatTimeoutError`).
- Nenhuma das duas funções conhece HTTP/status code — a tradução `UnknownStrategyError → 422` e `ChatTimeoutError → 504` acontece exclusivamente no middleware de erro de `src/http/server.ts` (Principle III da constitution).
