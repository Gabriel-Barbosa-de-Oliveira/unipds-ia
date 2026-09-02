# Data Model: Núcleo de Raciocínio (Reasoning Strategies Core)

Entidades derivadas do [spec.md](./spec.md) (seção Key Entities) e das decisões de [research.md](./research.md). Campos e tipos são descritos de forma independente de implementação (a codificação zod/Sequelize concreta é detalhe de tarefa, não deste documento).

## Service

Serviço operacional monitorado. Dado de referência, semeado; não é mutado por nenhuma tool desta feature.

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Identificador estável, gerado no seed |
| `name` | string | Único; é o valor usado por `open_incident(service)` e como referência de `Alert`/`Incident` |

## Alert

Sinal de monitoramento associado a um `Service`. Dado de referência, semeado; nenhuma tool desta feature muda seu status (apenas leitura via `list_alerts`).

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Identificador estável, gerado no seed |
| `serviceId` | string (FK → Service.id) | Deve referenciar um `Service` existente |
| `title` | string | Descrição curta do alerta |
| `status` | enum: `firing` \| `resolved` | Estado no momento do seed (3 `firing`, 3 `resolved` no dataset canônico) |
| `createdAt` | datetime | Timestamp de criação (seed) |

**Regra de leitura**: `list_alerts(status?)` retorna todos os alertas quando `status` é omitido, ou apenas os que casam com o `status` informado (lista vazia é um resultado válido, não erro — ver spec Edge Cases).

## Incident

Registro de incidente aberto por uma estratégia de raciocínio em nome do operador. Único tipo de dado mutável por esta feature (`open_incident` cria, `resolve_incident` transiciona).

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Gerado na criação (`open_incident`); é o identificador usado por `resolve_incident(id)` |
| `title` | string | Obrigatório, não vazio |
| `serviceId` | string (FK → Service.id) | Obrigatório; DEVE referenciar um `Service` existente — caso contrário `ServiceNotFoundError` (FR-008, User Story 3) |
| `severity` | enum: `low` \| `medium` \| `high` \| `critical` | Obrigatório; valor fora do enum → `InvalidSeverityError` (FR-008) |
| `status` | enum: `open` \| `resolved` | Criado como `open`; `resolve_incident` transiciona para `resolved` |
| `createdAt` | datetime | Timestamp de criação |
| `resolvedAt` | datetime \| null | Preenchido quando `status` vira `resolved`; `null` enquanto `open` |

### State Transitions (Incident)

```text
(open_incident) → open → (resolve_incident) → resolved
```

- `open → resolved`: único caminho de transição; disparado por `resolve_incident(id)`.
- `resolve_incident` sobre um `id` inexistente → `IncidentNotFoundError` (FR-008, sem efeito colateral).
- `resolve_incident` sobre um incidente já `resolved`: idempotente — retorna o incidente já resolvido sem erro (não há um estado "reaberto" nesta feature).

## ReasoningStrategy (contrato de execução, não persistido)

Representa uma abordagem de raciocínio nomeada; ver [contracts/reasoning-strategy.md](./contracts/reasoning-strategy.md) para a assinatura completa.

| Campo | Tipo | Regras |
|---|---|---|
| `name` | string | Identificador da estratégia (`"react"`, `"plan-and-execute"`) usado pela arena (`--strategies`) |

## TraceEvent (contrato de execução, não persistido)

Um passo dentro de uma execução. União discriminada por `type`.

| Campo comum | Tipo | Regras |
|---|---|---|
| `type` | enum: `thought` \| `action` \| `observation` \| `plan` \| `critique` \| `answer` | Discriminante da união |
| `at` | number (ms desde epoch ou desde o início do run) | Usado para ordenar/formatar o trace |

| Campo específico de `action` | Tipo | Regras |
|---|---|---|
| `tool` | string | Nome da tool invocada (`list_alerts` \| `open_incident` \| `resolve_incident`) |
| `args` | object | Argumentos passados à tool (já validados por zod no momento em que o evento é registrado) |

| Campo específico de `observation` | Tipo | Regras |
|---|---|---|
| `result` | object | Retorno da tool em caso de sucesso, ou forma de erro estruturado (`{ error: string, ... }`) em caso de falha |

| Campo específico de `plan` | Tipo | Regras |
|---|---|---|
| `steps` | string[] | Lista ordenada de passos restantes (apenas Plan-and-Execute) |

## Metrics (contrato de execução, não persistido)

| Campo | Tipo | Regras |
|---|---|---|
| `llmCalls` | number | Contagem de chamadas ao modelo da fábrica única durante o `run` (FR-003) |
| `latencyMs` | number | Tempo total decorrido do início ao fim do `run` (FR-003) |

## Relacionamentos

```text
Service 1 ── * Alert       (um serviço tem zero ou mais alertas)
Service 1 ── * Incident    (um serviço tem zero ou mais incidentes)
ReasoningStrategy.run() → produz 1 sequência de TraceEvent + 1 Metrics (efêmero, não persistido nesta feature — spec Assumptions)
```
