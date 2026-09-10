# Data Model: Persistência Real de Operações

Estende as entidades já definidas em [specs/001-reasoning-strategies-core/data-model.md](../001-reasoning-strategies-core/data-model.md) (`Service`, `Alert`, `Incident`) e acrescenta `Runbook`. Campos e tipos são descritos de forma independente de implementação; o mapeamento para colunas SQL concretas (nomes de coluna, `CHECK`, índices) é detalhe de tarefa, guiado pelas decisões de [research.md](./research.md).

> Nota de nomenclatura: o pedido original desta feature chama o repositório existente de "OpsStore". O nome atual no código é `OpsStoreRepository` (`src/services/ops-store.repository.ts`, introduzido na feature 001) — não é renomeado por esta feature (evita um rename sem ganho funcional em todo import existente); os métodos novos abaixo são adições a essa mesma interface.

## Service

Sem mudança de forma frente à feature 001. Ganha uma segunda associação (`Runbook`, opcional).

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Identificador estável, gerado no seed |
| `name` | string | Único; usado por `open_incident(service)` e agora também por `consultar_runbook(service)` |

## Alert

Sem mudança de forma frente à feature 001.

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Identificador estável, gerado no seed |
| `serviceId` | string (FK → Service.id) | Deve referenciar um `Service` existente |
| `title` | string | Descrição curta do alerta |
| `status` | enum: `firing` \| `resolved` | Valor fechado — rejeitado na gravação se fora do enum (FR-006), não só na validação de entrada |
| `createdAt` | datetime | Timestamp de criação (seed) |

## Incident

Ganha um campo novo (`summary`, opcional). `resolvedAt` já existia (feature 001); nesta feature passa a ser, junto com `summary`, um par preenchido pelo mesmo evento de resolução.

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Gerado na criação (`open_incident`) |
| `title` | string | Obrigatório, não vazio |
| `serviceId` | string (FK → Service.id) | Obrigatório; inexistente → `ServiceNotFoundError` |
| `severity` | enum: `low` \| `medium` \| `high` \| `critical` | Valor fechado — rejeitado na gravação se fora do enum (FR-006) |
| `status` | enum: `open` \| `resolved` | Valor fechado — rejeitado na gravação se fora do enum (FR-006); criado como `open` |
| `createdAt` | datetime | Timestamp de criação |
| `resolvedAt` | datetime \| null | Preenchido quando `status` vira `resolved`; `null` enquanto `open` |
| `summary` | string \| null | **Novo.** Opcional; preenchido apenas quando quem resolve o incidente informa um resumo do que foi feito (spec Assumptions); ausência não é erro |

### State Transitions (Incident)

Sem mudança frente à feature 001 — `open → resolved` continua o único caminho, `resolve_incident` continua idempotente sobre um incidente já `resolved`. A única adição é que o evento de transição pode opcionalmente carregar um `summary`, gravado junto com `resolvedAt`.

## Runbook (novo)

Conjunto de passos de mitigação recomendados para um serviço. Dado de referência, semeado; nenhuma tool desta feature o modifica (somente leitura via `consultar_runbook`).

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Identificador estável, gerado no seed |
| `serviceId` | string (FK → Service.id, no máximo 1 runbook por serviço) | Obrigatório; deve referenciar um `Service` existente |
| `content` | string | Passos de mitigação recomendados, em texto |

**Regra de leitura**: `consultar_runbook(service)` resolve o nome do serviço primeiro (mesma resolução por nome já usada por `open_incident`) — nome desconhecido → `ServiceNotFoundError`; serviço conhecido sem `Runbook` associado → resultado "sem runbook" (não é um erro, ver spec FR-005); serviço com `Runbook` → retorna o `content`.

## IncidentStatusFilter (contrato de leitura, não persistido)

Parâmetro de filtro de `list_incidents` — não é um valor armazenável de `Incident.status` (esse continua sendo só `open` \| `resolved`; "all" nunca é gravado nem participa do `CHECK` da coluna).

| Valor | Significado |
|---|---|
| `open` | Apenas incidentes com `status = "open"` |
| `resolved` | Apenas incidentes com `status = "resolved"` |
| `all` (ou omitido) | Todos os incidentes, sem filtro |

**Regra de leitura**: mesmo padrão de `list_alerts` — lista vazia é um resultado válido, nunca um erro.

## Cenário Operacional Canônico ("Mercadinho")

Não é uma entidade persistida à parte — é o conjunto inicial das quatro tabelas acima, com uma única definição de origem (`src/domain/seed-data.ts`, ver [research.md](./research.md) item 3), reaproveitada tanto pelo mock em memória quanto pelo seed do SQLite.

| Serviço | Alertas | Runbook |
|---|---|---|
| `checkout-api` | 1 firing, 1 resolved | Sim |
| `payments-api` | 1 firing | Sim |
| `inventory-service` | 1 firing | Não |
| `notifications-service` | 1 resolved | Não |
| `auth-service` | 1 resolved | Sim |

Total: 5 serviços, 6 alertas (3 `firing`, 3 `resolved`), 3 runbooks, 0 incidentes — idêntico ao dataset já usado pelas features 001–003, apenas com runbooks acrescentados.

## Relacionamentos

```text
Service 1 ── * Alert        (um serviço tem zero ou mais alertas)
Service 1 ── * Incident     (um serviço tem zero ou mais incidentes)
Service 1 ── 0..1 Runbook   (um serviço tem no máximo um runbook)
```

## Repositório (`OpsStoreRepository`) — superfície após esta feature

| Método | Novo nesta feature? | Regras |
|---|---|---|
| `listAlerts(status?)` | Não | Inalterado (feature 001) |
| `openIncident(input)` | Não | Inalterado (feature 001) |
| `resolveIncident(id, summary?)` | Alterado | `summary` é um parâmetro novo, opcional — chamadas existentes com só `id` continuam válidas (FR-011) |
| `listIncidents(status?)` | **Sim** | Ver `IncidentStatusFilter` acima (FR-003) |
| `getRunbook(service)` | **Sim** | Ver regra de leitura de `Runbook` acima (FR-004, FR-005) |
