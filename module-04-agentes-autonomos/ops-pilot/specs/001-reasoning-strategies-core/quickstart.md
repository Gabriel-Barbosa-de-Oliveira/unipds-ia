# Quickstart: Validar o Núcleo de Raciocínio

Guia de validação ponta a ponta para esta feature após a implementação (`/speckit-tasks` → `/speckit-implement`). Detalhes de contrato ficam em [contracts/](./contracts/); modelo de dados em [data-model.md](./data-model.md).

## Pré-requisitos

- Node 24 LTS e dependências instaladas (`npm install`).
- `.env` local (nunca lido pelo agente/Claude — só pela aplicação em runtime) com:
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_MODEL`
  - Variáveis de conexão MySQL, apenas se for validar o adaptador Sequelize (o caminho padrão de tools/arena/testes usa o adaptador in-memory e **não exige MySQL**, ver [research.md](./research.md) item 1).

## 1. Rodar os testes determinísticos (sem rede)

```sh
npm run typecheck
npm test
```

**Esperado**: `ops-store.test.ts` (domínio puro) e o teste de formatação de trace passam sem qualquer chamada de rede ou credencial configurada — validam FR-014 e SC-002/SC-005.

## 2. Semear o dataset canônico (opcional — necessário só para exercitar o adaptador MySQL)

```sh
npm run seed
```

**Esperado**: o script popula (ou restaura) 5 serviços e 6 alertas (3 `firing`, 3 `resolved`) via Sequelize/MySQL, de forma idempotente — reexecutar produz o mesmo estado (SC-006).

## 3. Rodar uma única estratégia via arena

```sh
npm run arena -- --input "quais alertas estão firing?" --strategies react
```

**Esperado** (SC-001, SC-002):
- Bloco `react` impresso com o trace completo (raciocínio → ação `list_alerts` com `{ status: "firing" }` → observação com os 3 alertas → resposta final).
- Métricas (`llmCalls`, `latencyMs`) impressas junto ao bloco.
- Resposta final lista exatamente os 3 alertas em estado `firing` do dataset semeado.

## 4. Comparar as duas estratégias na mesma pergunta

```sh
npm run arena -- --input "abra um incidente de severidade alta para o serviço checkout-api" --strategies react,plan-and-execute --max-iterations 8
```

**Esperado** (SC-003, SC-004):
- Um bloco por estratégia, cada um com seu próprio trace e métricas.
- A estratégia `plan-and-execute` mostra ao menos um evento `plan` antes de qualquer `action`.
- Um resumo comparativo final (nome × `llmCalls` × `latencyMs`) é impresso.
- Ambas as execuções terminam dentro do limite de 8 passos — nenhuma trava ou roda indefinidamente.

## 5. Validar erro estruturado em entrada inválida (User Story 3)

```sh
npm run arena -- --input "resolva o incidente inc-nao-existe" --strategies react
```

**Esperado** (SC-005): o trace mostra um evento `action` para `resolve_incident` seguido de um evento `observation` com um erro estruturado (`IncidentNotFoundError`), e a resposta final comunica a falha de forma clara — sem exceção não tratada nem travamento da CLI.
