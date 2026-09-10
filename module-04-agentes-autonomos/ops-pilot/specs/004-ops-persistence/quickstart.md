# Quickstart: Validar a Persistência Real de Operações

Guia de validação ponta a ponta para esta feature após a implementação (`/speckit-tasks` → `/speckit-implement`). Contratos em [contracts/](./contracts/); modelo de dados em [data-model.md](./data-model.md). Pressupõe as features 001–003 já implementadas.

## Pré-requisitos

- Node 24 LTS e dependências instaladas (`npm install`).
- Nenhuma credencial de banco é necessária — `node:sqlite` é nativo do runtime. `.env` só é necessário (`OPENROUTER_API_KEY`/`OPENROUTER_MODEL`) para os passos 4–5, que chamam o modelo real.

## 1. Rodar os testes determinísticos (sem rede, sem arquivo compartilhado)

```sh
npm run typecheck
npm test
```

**Esperado**: `src/store/sqlite-ops-store.test.ts` e `src/agents/tools.test.ts` (novos) passam instanciando `SqliteOpsStore(":memory:")` por teste — seed, abrir/listar/resolver incidentes, filtros de `list_incidents`, `consultar_runbook`, e violação de `CHECK` rejeitada na gravação. Nenhum teste depende de `data/opspilot.db` nem de rede. Valida FR-001 a FR-011 e SC-001, SC-004, SC-006.

## 2. Semear o banco real e confirmar a persistência entre execuções

```sh
rm -f ./data/opspilot.db   # começa de um estado limpo (data/ já é ignorado pelo git)
npm run seed
```

**Esperado**: cria `./data/opspilot.db` com o cenário canônico (5 serviços, 6 alertas, 3 runbooks, 0 incidentes). Rodar `npm run seed` novamente não duplica nem falha (idempotente — SC-005).

## 3. Subir a API e confirmar que os dados sobrevivem a um reinício

```sh
npm run dev
```

Em outro terminal:

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "abra um incidente de severidade alta para checkout-api com o título \"latência alta\""}' | jq
```

Anote o `id` do incidente na resposta, pare o servidor (`Ctrl+C`) e suba de novo (`npm run dev`):

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "quais incidentes estão abertos?"}' | jq
```

**Esperado** (User Story 1, SC-001): o incidente aberto antes do reinício continua aparecendo na lista de abertos, com o mesmo `id`.

## 4. Listar incidentes por status e consultar um runbook

```sh
curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "resolva esse incidente com o resumo \"mitigado via rollback\" e me diga quantos incidentes resolvidos existem"}' | jq

curl -s -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "qual o runbook do checkout-api?"}' | jq
```

**Esperado** (User Story 2, User Story 3, SC-002, SC-003): a primeira resposta reflete o incidente resolvido (com resumo); a segunda traz o conteúdo do runbook de `checkout-api` (semeado no passo 2).

## 5. Confirmar que o cenário de bench continua reprodutível e isolado do banco real

```sh
npm run bench -- --scenario C1
```

**Esperado** (User Story 4, SC-005): o bench roda contra seu próprio mock em memória (não contra `./data/opspilot.db`) — rodar duas vezes seguidas produz o mesmo `before`/`after` por cenário, mesmo que passos 2–4 já tenham criado incidentes no banco real.
