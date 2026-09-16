# Quickstart: Validar a Medição de Contexto

Guia de validação ponta a ponta para esta feature após a implementação (`/speckit-tasks` → `/speckit-implement`). Contrato completo em [contracts/post-chat.md](./contracts/post-chat.md) e [contracts/tokens.md](./contracts/tokens.md); entidades em [data-model.md](./data-model.md). Pressupõe as features 001–008 já implementadas.

## Pré-requisitos

- Node 24 LTS e dependências instaladas (`npm install`) — nenhuma dependência nova.
- `.env` local com `OPENROUTER_API_KEY`/`OPENROUTER_MODEL` (necessário para os passos 2–4, que sobem o servidor real e chamam o modelo).

## 1. Rodar os testes determinísticos (offline)

```sh
npm run typecheck
npm test
```

**Esperado**: `src/context/tokens.test.ts` (`estimateTokens` com strings conhecidas; `UsageCollector` com `LLMResult`/prompts forjados — real, estimado e misto; `mergeTokenUsage`; `buildContextBreakdown` com partes vazias e não-vazias, `total` sempre igual à soma) roda 100% offline. Extensões de `src/agents/reflection.test.ts` (crítica agora retorna `tokenUsage` também) e `src/http/server.test.ts` (`metrics.promptTokens`/`tokenSource`/`contextBreakdown` presentes e coerentes) continuam sem rede, com fakes injetados.

## 2. Confirmar `promptTokens` real numa resposta simples

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "quais alertas estão firing?"}' | jq '.metrics'
```

**Esperado** (User Story 1, SC-001): `metrics.promptTokens` é um número maior que zero e `metrics.tokenSource` é `"real"` (o provedor configurado relata uso nas respostas não-streaming usadas pelo projeto).

## 3. Confirmar o detalhamento de contexto com histórico e fatos

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "eu sou o responsável pelo serviço de pagamentos", "userId": "gabriel-quickstart-009"}' | jq -r .conversationId
```

Guarde o `conversationId` retornado (`CONV_ID`) e envie uma segunda mensagem na mesma conversa e para a mesma pessoa:

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "quem cuida do checkout financeiro?", "userId": "gabriel-quickstart-009", "conversationId": "CONV_ID"}' | jq '.metrics.contextBreakdown'
```

**Esperado** (User Story 3, SC-003, SC-004): `contextBreakdown.conversationHistory` e `contextBreakdown.recalledFacts` são ambos maiores que zero (há histórico da primeira troca e o fato foi lembrado); `contextBreakdown.total` é exatamente `currentMessage + conversationHistory + recalledFacts`.

## 4. Confirmar zeros explícitos sem histórico/fatos

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "oi"}' | jq '.metrics.contextBreakdown'
```

**Esperado** (FR-007): `conversationHistory` e `recalledFacts` aparecem como `0` (nunca ausentes/omitidos), e `total` é igual a `currentMessage`.

## 5. Limpeza

```sh
node --import tsx -e '
import { DatabaseSync } from "node:sqlite";
const db = new DatabaseSync("./data/opspilot.db");
db.prepare("DELETE FROM memories WHERE user_id LIKE ?").run("gabriel-quickstart-009%");
'
```

> `npm run bench`/`npm run arena` não passam por `/chat` e ficam fora do escopo desta instrumentação (mesmo raciocínio de `006`/`007`/`008`, spec Assumptions).
