# Quickstart: validar a resiliência de modelo

## Pré-requisitos

- `npm install` já rodado.
- Para os passos manuais: `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` no ambiente do processo e, opcionalmente, `OPENROUTER_MODEL_FALLBACK`. O agente nunca lê o `.env`; quem define as variáveis é você.

## 1. Gates automáticos (sem rede)

```bash
npm run typecheck
npm test
```

O que as suítes novas e alteradas cobrem:

- `src/agents/model.test.ts`:
  - `loadModelConfig`: reserva ausente, vazia, com espaços ou igual ao principal vira `undefined`;
  - `isTransientModelError`: tabela de erros transitórios e não transitórios;
  - `resilientRunnable`, com modelos fake que falham por roteiro: caminho feliz, retry absorvido, fallback por erro transitório persistente, fallback imediato por erro não transitório, sem reserva → erro, reserva falhando → erro;
  - `toolCallingModel` aceito por `createReactAgent` sem tentar religar as tools.
- `src/agents/model-usage.test.ts`: `summarizeModelUsage` (retry não é fallback; `modelUsed` é o modelo do último `end`) e `withModelFallbacks` (os eventos entram antes do último `answer` e o `at` é reindexado).
- `src/agents/trace.test.ts`: formatação de `fallback` e `model=` em `formatMetrics`.
- `src/graph/production-graph.test.ts`: fallbacks do roteador logo depois do `route`, invariantes ajustadas e `metrics.modelUsed` vindo da estratégia.
- `src/http/server.test.ts`: `metrics.modelUsed` presente.

## 2. Caminho feliz (manual)

```bash
npm run dev
curl -s localhost:3000/chat -H 'content-type: application/json' \
  -d '{"message":"quais alertas estão firing?"}' | jq '.metrics.modelUsed, [.trace[] | select(.type=="fallback")]'
```

Esperado: `modelUsed` igual a `$OPENROUTER_MODEL` e uma lista vazia de fallbacks.

## 3. Forçar a reserva (manual)

Suba a API com um modelo principal inexistente, por exemplo `OPENROUTER_MODEL=nao/existe` (erro 4xx, não transitório), e uma reserva válida em `OPENROUTER_MODEL_FALLBACK`. Repita o `curl` do passo 2.

Esperado:

- `200`;
- `modelUsed` igual à reserva;
- pelo menos um evento `fallback` com `from: "nao/existe"`;
- a resposta chega sem a espera dos retries, porque erro 4xx não é retentado.

## 4. Sem reserva (manual)

Mesmo cenário do passo 3, mas sem `OPENROUTER_MODEL_FALLBACK`. Esperado: `500 internal_error`, igual ao comportamento de hoje.

## 5. Arena

```bash
npm run arena -- --input "quais alertas estão firing?"
```

Esperado: a linha de métricas mostra `model=<id>`. Com a reserva forçada (passo 3), o trace mostra `[fallback] nao/existe → <reserva>: ...`.
