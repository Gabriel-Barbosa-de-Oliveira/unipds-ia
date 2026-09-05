# Quickstart: Validar o Endpoint HTTP de Chat

Guia de validação ponta a ponta para esta feature após a implementação (`/speckit-tasks` → `/speckit-implement`). Detalhes de contrato ficam em [contracts/](./contracts/); modelo de dados em [data-model.md](./data-model.md). Pressupõe as features 001 e 002 já implementadas.

## Pré-requisitos

- Node 24 LTS e dependências instaladas (`npm install`).
- `.env` local com `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` (necessário só para os passos 3–5, que sobem o servidor real e chamam o modelo; os passos 1–2 não precisam de rede).

## 1. Rodar os testes determinísticos (sem rede)

```sh
npm run typecheck
npm test
```

**Esperado**: `src/agents/index.test.ts`, `src/services/chat.service.test.ts` e `src/http/server.test.ts` passam sem qualquer chamada de rede ou credencial configurada — cobrindo com fakes injetados: resolução da estratégia padrão e explícita, erro de estratégia desconhecida, composição com `reflect: true`, timeout disparando `ChatTimeoutError`, e o endpoint completo respondendo 200/400/422/504 conforme [contracts/post-chat.md](./contracts/post-chat.md). Valida FR-001 a FR-010 e SC-001 a SC-004.

## 2. Semear o dataset canônico (se ainda não estiver semeado)

```sh
npm run seed
```

**Esperado**: mesmo comportamento já validado pelas features anteriores — necessário apenas se as tools de domínio forem exercitadas via o adaptador MySQL durante uma chamada real ao endpoint (passos 3–5).

## 3. Subir o servidor e chamar `/chat` com a estratégia padrão

```sh
npm run dev
```

Em outro terminal:

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "quais alertas estão firing?"}' | jq
```

**Esperado** (User Story 1, SC-001): resposta `200` contendo `answer`, `trace` e `metrics`, sem precisar informar `strategy` nem `reflect` — a estratégia padrão (`react`) é usada.

## 4. Escolher estratégia explicitamente e ativar reflection

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "abra um incidente de severidade alta para checkout-api", "strategy": "plan-and-execute", "reflect": true}' | jq
```

**Esperado** (User Story 2, User Story 3, SC-005): resposta `200` cujo `trace` inclui ao menos um evento `critique` (evidência de reflection) e cujas `metrics.llmCalls` são maiores do que uma chamada equivalente com `reflect: false` — comparável inspecionando as duas respostas lado a lado.

## 5. Confirmar os três tipos de erro

```sh
# corpo inválido — 400
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" -d '{}'

# estratégia desconhecida — 422
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" -d '{"message": "oi", "strategy": "nao-existe"}'
```

**Esperado** (SC-002, SC-003): `400` para corpo sem `message`; `422` para estratégia desconhecida — em nenhum dos dois casos o servidor faz qualquer chamada ao modelo (confirmável pela resposta instantânea, sem o atraso típico de uma chamada real).

> O cenário de timeout (`504`, SC-004) não é prático de reproduzir manualmente aqui (exigiria 180s reais) — é coberto pelo teste de integração do passo 1 com um `timeoutMs` reduzido injetado via `createApp({ timeoutMs })`.
