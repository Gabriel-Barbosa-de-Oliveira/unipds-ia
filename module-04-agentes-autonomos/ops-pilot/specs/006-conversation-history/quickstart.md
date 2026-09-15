# Quickstart: Validar a Conversa Persistente

Guia de validação ponta a ponta para esta feature após a implementação (`/speckit-tasks` → `/speckit-implement`). Contrato completo em [contracts/post-chat.md](./contracts/post-chat.md) e [contracts/conversation-store.md](./contracts/conversation-store.md); entidades em [data-model.md](./data-model.md). Pressupõe as features 001–003 já implementadas (endpoint `/chat` funcionando sem conversa).

## Pré-requisitos

- Node 24 LTS e dependências instaladas (`npm install`).
- `.env` local com `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` (necessário só para os passos 3–6, que sobem o servidor real e chamam o modelo; os passos 1–2 não precisam de rede).

## 1. Rodar os testes determinísticos (sem rede)

```sh
npm run typecheck
npm test
```

**Esperado**: `src/domain/conversation.test.ts` (composição pura do prompt com histórico), `src/store/sqlite-conversation-store.test.ts` (`create`/`append`/`lastMessages` sobre `":memory:"`, limite de 12, `ConversationNotFoundError`, isolamento entre conversas) e as extensões de `src/http/server.test.ts` (conversa nova, conversa continuada, `404` para id desconhecido, `metrics.historyMessages`) passam sem qualquer chamada de rede.

## 2. Semear o dataset canônico (se ainda não estiver semeado)

```sh
npm run seed
```

**Esperado**: mesmo comportamento das features anteriores — conversas não fazem parte do cenário canônico (`seed-data.ts`), então este passo não cria nem afeta nenhuma conversa.

## 3. Iniciar uma conversa nova

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "me chame de Gabriel"}' | jq
```

**Esperado** (User Story 1, SC-002): resposta `200` cujo corpo inclui `conversationId` (novo, nenhum foi informado na requisição) e `metrics.historyMessages: 0` (nenhuma mensagem anterior). Guarde o `conversationId` retornado para o próximo passo.

## 4. Continuar a mesma conversa

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "qual é o meu nome?", "conversationId": "<conversationId do passo 3>"}' | jq
```

**Esperado** (User Story 1, SC-001): `answer` reflete corretamente o nome informado no passo 3 (o copiloto tem acesso à mensagem anterior); `metrics.historyMessages` reporta as mensagens anteriores da conversa (2: a mensagem do usuário e a resposta do copiloto do passo 3); `conversationId` na resposta é o mesmo enviado na requisição.

## 5. Identificador de conversa desconhecido

```sh
curl -s -w "\nHTTP %{http_code}\n" -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "oi", "conversationId": "id-que-nao-existe"}'
```

**Esperado** (FR-007): `404` com `{"error": "conversation_not_found", "conversationId": "id-que-nao-existe"}`; nenhuma chamada ao modelo ocorre (resposta instantânea, sem o atraso típico de uma chamada real).

## 6. Isolamento entre conversas concorrentes

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "me chame de Ana"}' | jq -r .conversationId
# guarde como CONV_A

curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "qual é o meu nome?", "conversationId": "'"$CONV_B_OU_INEXISTENTE"'"}' | jq
```

**Esperado** (SC-005): perguntar "qual é o meu nome?" em uma conversa diferente de `CONV_A` nunca retorna "Ana" — cada `conversationId` mantém seu próprio histórico, isolado dos demais.

> O cenário de mais de 12 mensagens (User Story 2, SC-003) não é prático de reproduzir manualmente aqui (exigiria 13+ chamadas reais ao modelo) — é coberto pelo teste de integração de `src/store/sqlite-conversation-store.test.ts` (`lastMessages` nunca retorna mais que `limit`) e de `src/http/server.test.ts` com uma estratégia fake e um `ConversationStore` fake pré-populado.
