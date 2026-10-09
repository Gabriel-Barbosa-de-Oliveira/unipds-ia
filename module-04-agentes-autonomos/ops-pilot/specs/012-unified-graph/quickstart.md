# Quickstart: validar o grafo unificado

> **Nota de implementação (nomes finais)**: seguindo o esboço passado no `/speckit-implement`, o código usa `src/graph/production-graph.ts` e `src/graph/router.ts` (não `src/agents/`); nós `contexto`, `roteador`, `react`, `planExecute`, `reflect`, `resposta`; rotas `react | planExecute | reflect`. O `strategy` do `/chat` aceita também os nomes legados `plan-and-execute` e `reflection` como aliases. Onde este documento usa os nomes antigos (`context`/`router`/`answer` como nós, `plan-and-execute`/`reflection` como rotas), leia com esse mapeamento.

## Pré-requisitos

- `npm install` já rodado.
- Para os passos manuais: `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` configurados no ambiente (o agente não lê `.env`), e `npm run seed` executado.

## 1. Gates automáticos (sem rede)

```bash
npm run typecheck
npm test
```

Resultado esperado: tudo verde. As suítes novas e alteradas cobrem:

- `src/agents/router.test.ts`: a tabela aparece no prompt e `resolveRouteDecision` cobre override, decisão válida e fallback (erro, `null`, rota inválida, motivo vazio).
- `src/agents/production-graph.test.ts`: com um roteador e estratégias fake, só a estratégia escolhida roda, há exatamente um `route` e ele vem primeiro, todo evento tem `node`, `at` é sequencial, o override não chama o roteador e as métricas são somadas.
- `src/agents/trace.test.ts`: formatação do `route` e do prefixo de `node`.
- `src/http/server.test.ts`: com e sem `strategy`, `strategy: "reflection"` aceito, estratégia desconhecida → 422 e campo `route` na resposta.

## 2. Roteamento automático (manual)

```bash
npm run dev
curl -s localhost:3000/chat -H 'content-type: application/json' \
  -d '{"message":"quais alertas estão firing?"}' | jq '.route, [.trace[] | {node, type}]'
```

Resultado esperado: `route.source == "router"`, um único evento `route` com `node: "router"` no índice 0 e os demais eventos com `node` igual a `route.route`.

Repita com um pedido de várias etapas ("triar todos os alertas críticos e abrir um incidente para cada") e com um pedido em que a precisão é crítica ("escreva o resumo do incidente X para o pós-mortem, revisando os fatos"). As rotas esperadas são `plan-and-execute` e `reflection`, respectivamente (SC-004: pelo menos 80% de acerto numa amostra de referência).

## 3. Override

```bash
curl -s localhost:3000/chat -H 'content-type: application/json' \
  -d '{"message":"quais alertas estão firing?","strategy":"plan-and-execute"}' | jq '.route, .metrics.llmCalls'
```

Resultado esperado: `source: "override"`, `route: "plan-and-execute"`, e nenhuma chamada a mais por roteamento.

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:3000/chat -H 'content-type: application/json' \
  -d '{"message":"oi","strategy":"nao-existe"}'
```

Resultado esperado: `422`.

## 4. Fallback

Coberto pelos testes automáticos do passo 1 (roteador fake que lança erro ou devolve uma rota inválida → `200` com `source: "fallback"` e `route: "react"`).
