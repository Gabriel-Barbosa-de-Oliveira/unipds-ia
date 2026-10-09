# Quickstart: validar o Orçamento de Contexto

## Pré-requisitos

- `npm install` feito; Node 24.
- Para o teste manual: `OPENROUTER_API_KEY`/`OPENROUTER_MODEL` já configurados no ambiente (nunca ler `.env`).

## 1. Testes automatizados (offline)

```bash
npm run typecheck
npm test
```

Esperado: verdes, incluindo `src/context/context-builder.test.ts` com o cenário "tetos baixos cortam na ordem certa" ([contrato](./contracts/context-builder.md#teste-âncora-tetos-baixos-cortam-na-ordem-certa)).

## 2. Padrões sem env (regressão)

```bash
npm run dev
curl -s localhost:3000/chat -H 'content-type: application/json' -d '{"message":"quais alertas estão firing?"}' | jq .metrics
```

Esperado: `contextTrimmed = { historyMessages: 0, recalledFacts: 0 }`; `contextBreakdown.system = 0`, `summary = 0`.

## 3. Teto baixo da janela corta as mais antigas

```bash
CONTEXT_BUDGET_WINDOW=20 npm run dev
```

Envie 4–5 mensagens na mesma conversa (reusando `conversationId`). Esperado nas últimas respostas: `contextBreakdown.conversationHistory ≤ 20`, `contextTrimmed.historyMessages > 0`, e `historyMessages` menor que o total de mensagens anteriores. A resposta não deve lembrar o conteúdo da primeira mensagem.

## 4. Teto zero desliga memórias

```bash
CONTEXT_BUDGET_MEMORIES=0 npm run dev
```

Com um `userId` que tenha fatos lembrados: `contextBreakdown.recalledFacts = 0` e `contextTrimmed.recalledFacts` = número de fatos recuperados.

## 5. Env inválida cai no padrão

```bash
CONTEXT_BUDGET_WINDOW=abc npm run dev
```

Esperado: servidor sobe normalmente e o comportamento é o mesmo do passo 2 (teto 1200).

## 6. arena/bench inalterados

```bash
npm run bench
```

Esperado: mesmos resultados de antes da feature (o builder só repassa a mensagem).
