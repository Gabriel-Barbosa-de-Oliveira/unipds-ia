# Quickstart: Validar o Refletor de Aprendizado

Guia de validação ponta a ponta para esta feature após a implementação (`/speckit-tasks` → `/speckit-implement`). Contrato completo em [contracts/post-chat.md](./contracts/post-chat.md) e [contracts/learning-reflector.md](./contracts/learning-reflector.md); entidades em [data-model.md](./data-model.md). Pressupõe as features 001–007 já implementadas (em especial `007-semantic-memory`, da qual esta feature depende diretamente).

## Pré-requisitos

- Node 24 LTS e dependências instaladas (`npm install`) — nenhuma dependência nova.
- `.env` local com `OPENROUTER_API_KEY`/`OPENROUTER_MODEL` (necessário para os passos 2–5, que sobem o servidor real e chamam o modelo tanto para a resposta quanto para a destilação de aprendizado).

## 1. Rodar os testes determinísticos (offline)

```sh
npm run typecheck
npm test
```

**Esperado**: `src/memory/learning-reflector.test.ts` (lógica de decisão de `reflectAndRemember` com `distillFn` fake — hasLearning true/false, fact ausente, falha do `distillFn`, falha de `store.remember`) e as extensões de `src/http/server.test.ts` (refletor disparado só com `userId`, resposta não bloqueada mesmo com um fake que nunca resolve, falha do refletor não vira erro HTTP) rodam sem rede, mesmo padrão já usado pelas features anteriores.

## 2. Aprendizado automático, sem pedir para lembrar

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "eu sou o responsável pelo serviço de pagamentos", "userId": "gabriel"}' | jq -r .answer
```

A resposta chega normalmente (sem atraso perceptível). Aguarde alguns segundos para o refletor terminar em background, depois pergunte algo relacionado, com palavras diferentes:

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "quem cuida do checkout financeiro?", "userId": "gabriel"}' | jq -r .answer
```

**Esperado** (User Story 1, SC-001): a segunda resposta reflete o fato, mesmo sem a pessoa ter pedido explicitamente "lembre disso" na primeira mensagem.

## 3. Pedido pontual não vira aprendizado

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "qual o status do serviço de pagamentos agora?", "userId": "gabriel"}' > /dev/null

curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "o que você sabe sobre mim?", "userId": "gabriel"}' | jq -r .answer
```

**Esperado** (SC-005): nenhum fato novo derivado da pergunta pontual do passo anterior aparece — só o que já havia sido aprendido antes (passo 2), se houver.

## 4. Segredo nunca é aprendido

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "minha chave de API do provedor de alertas é sk-abandonei-1234", "userId": "gabriel"}' > /dev/null
```

Aguarde e confirme que nada foi registrado (ex.: consultando o banco local, ou perguntando algo que só recuperaria esse conteúdo se tivesse sido salvo):

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "qual é a minha chave de API?", "userId": "gabriel"}' | jq -r .answer
```

**Esperado** (SC-002): o copiloto não tem esse dado guardado — nunca foi persistido pelo refletor.

## 5. Desfazer um aprendizado automático

```sh
curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "pode esquecer que eu cuido de pagamentos, mudei de time", "userId": "gabriel"}' | jq -r .answer

curl -s -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"message": "quem cuida do checkout financeiro?", "userId": "gabriel"}' | jq -r .answer
```

**Esperado** (User Story 2, SC-004): a segunda resposta não menciona mais Gabriel como responsável por pagamentos — o fato aprendido automaticamente no passo 2 foi removido pela mesma via (`forget_fact`) já usada para fatos ensinados manualmente em `007`.

> Sem `userId` na requisição, nenhum refletor é acionado — comportamento idêntico ao de antes desta feature (contracts/post-chat.md).
