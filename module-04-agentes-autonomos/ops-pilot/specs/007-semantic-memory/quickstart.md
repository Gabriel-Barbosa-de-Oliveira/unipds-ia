# Quickstart: Validar a Memória Semântica

Guia de validação ponta a ponta para esta feature após a implementação (`/speckit-tasks` → `/speckit-implement`). Contrato completo em [contracts/post-chat.md](./contracts/post-chat.md) e [contracts/memory-store.md](./contracts/memory-store.md); entidades em [data-model.md](./data-model.md). Pressupõe as features 001–006 já implementadas.

## Pré-requisitos

- Node 24 LTS e dependências instaladas (`npm install`, incluindo `@huggingface/transformers` novo).
- `.env` local com `OPENROUTER_API_KEY`/`OPENROUTER_MODEL` (necessário só para os passos 3–5, que sobem o servidor real e chamam o modelo de raciocínio).
- Acesso de rede na **primeira** execução de qualquer teste/uso real da memória (download único do checkpoint ONNX `onnx-community/all-MiniLM-L6-v2-ONNX`, cacheado depois em `.cache/transformers`) — ver research.md item 4 e item 9.

## 1. Rodar os testes determinísticos (offline, exceto o teste dedicado do item 2)

```sh
npm run typecheck
npm test
```

**Esperado**: `src/domain/memory.test.ts` (produto escalar, seleção top-3/limiar 0.3, dedup > 0.92, serialização do vetor — tudo com vetores forjados, instantâneo) e a maior parte de `src/memory/memory-store.test.ts` (dedup, isolamento entre `userId`s, limite de 3, forget não encontrado) rodam com um `embed` fake injetado, sem rede. `src/memory/embeddings.test.ts` (e o teste dedicado de recall genuíno dentro de `memory-store.test.ts`) usam o modelo real — mais lentos, e a primeira execução em uma máquina nova baixa o checkpoint.

## 2. Confirmar a promessa central: recall sem palavra em comum

Este é o único cenário que precisa do modelo real (não dá para simular com vetor forjado, ver research.md item 4) — já coberto por um teste automatizado, mas também verificável manualmente:

```sh
node --import tsx -e '
import { SqliteMemoryStore } from "./src/memory/memory-store.ts";
const store = new SqliteMemoryStore(":memory:");
await store.remember("gabriel", "eu sou o responsável pelo serviço de pagamentos");
const results = await store.recall("gabriel", "quem cuida do checkout financeiro?");
console.log(results);
'
```

**Esperado**: a lista retornada não é vazia — o fato registrado aparece com `score >= 0.3`, mesmo sem nenhuma palavra em comum entre a pergunta e o fato (SC-001).

## 3. Registrar e recuperar um fato via `/chat`

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "eu sou o responsável pelo serviço de pagamentos", "userId": "gabriel"}' | jq
```

Em outra chamada (pode ser em qualquer conversa, `conversationId` novo ou omitido — memória semântica não depende de conversa):

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "quem cuida do checkout financeiro?", "userId": "gabriel"}' | jq -r .answer
```

**Esperado** (User Story 1, SC-001): a segunda resposta reflete o fato registrado na primeira, mesmo sem palavras em comum com a pergunta.

## 4. Deduplicação

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "sou eu quem cuida de pagamentos por aqui", "userId": "gabriel"}' | jq -r .answer
```

**Esperado** (User Story 2, SC-002): nenhum fato novo é registrado — é reconhecido como essencialmente o mesmo já dito no passo 3 (score de similaridade > 0.92 contra o fato existente).

## 5. Esquecer um fato

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "pode esquecer que eu cuido de pagamentos, mudei de time", "userId": "gabriel"}' | jq -r .answer

curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "quem cuida do checkout financeiro?", "userId": "gabriel"}' | jq -r .answer
```

**Esperado** (User Story 3, SC-004): a segunda resposta não menciona mais Gabriel como responsável por pagamentos — o fato foi removido.

## 6. Isolamento entre pessoas

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "eu sou o responsável pelo serviço de autenticação", "userId": "ana"}' | jq -r .answer

curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "quem cuida da autenticação?", "userId": "gabriel"}' | jq -r .answer
```

**Esperado** (SC-005): a resposta para `"gabriel"` nunca menciona o fato registrado por `"ana"`.

> Sem `userId` na requisição, o comportamento é idêntico ao de `006-conversation-history` — nenhum recall, nenhuma tool de memória disponível (contracts/post-chat.md).
