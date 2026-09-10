# Contract: `SqliteOpsStore` (`src/store/sqlite-ops-store.ts`)

Implementação de `OpsStoreRepository` (`src/services/ops-store.repository.ts`) sobre `node:sqlite` (`DatabaseSync`). Ver [data-model.md](../data-model.md) para a forma das entidades e [research.md](../research.md) para as decisões por trás deste contrato.

## Construção

```ts
new SqliteOpsStore(path?: string)
```

- `path` default: `process.env.OPSPILOT_DB ?? "./data/opspilot.db"`.
- `":memory:"` abre um banco novo e vazio, isolado por instância — usado por testes (nunca `OPSPILOT_DB`).
- A conexão e a DDL das 4 tabelas (`services`, `alerts`, `incidents`, `runbooks`, com `CREATE TABLE IF NOT EXISTS`) são adiadas (lazy) até a primeira operação real sobre a instância — nunca na construção. Isso garante que simplesmente compor `new SqliteOpsStore()` (ex.: a composição padrão em `src/agents/tools.ts`) nunca toca o arquivo em disco por si só; só o primeiro método chamado (incluindo `seed()`) dispara a abertura + DDL. A DDL em si é idempotente: reabrir um arquivo já existente, ou criar várias instâncias `":memory:"` em testes distintos, nunca falha nem duplica schema.
- A inicialização **não** semeia dados — seed é um passo explícito (`seedCanonicalScenario`, ver abaixo), para não sobrescrever silenciosamente um banco de produção já em uso.

## `CHECK` constraints (DDL)

| Tabela | Coluna | Valores permitidos |
|---|---|---|
| `alerts` | `status` | `firing`, `resolved` |
| `incidents` | `severity` | `low`, `medium`, `high`, `critical` |
| `incidents` | `status` | `open`, `resolved` |

Uma tentativa de `INSERT`/`UPDATE` com um valor fora dessas listas falha na camada de armazenamento (exceção do driver), independentemente de a chamada ter passado pela validação zod das tools ou não (FR-006, SC-004).

## Métodos (`OpsStoreRepository`)

Todos usam **prepared statements** (`db.prepare(sql)` com parâmetros ligados por `?`); nenhuma query concatena valor de entrada na string SQL.

### `listAlerts(status?: AlertStatus): Promise<Alert[]>`

Sem mudança de contrato frente à feature 001. `SELECT` com ou sem `WHERE status = ?` conforme `status` seja informado.

### `openIncident(input: OpenIncidentInput): Promise<Incident>`

Sem mudança de contrato frente à feature 001. Resolve o serviço por nome (`SELECT` em `services`); `INSERT` em `incidents` com `status = 'open'`, `resolved_at = NULL`, `summary = NULL`.

### `resolveIncident(id: string, summary?: string): Promise<Incident>`

**Alterado**: ganha o parâmetro opcional `summary`. Idempotente como antes (resolver um incidente já `resolved` retorna o registro existente sem erro e sem sobrescrever um `summary` já gravado). Quando `summary` é informado na primeira resolução, é gravado junto com `resolved_at` no mesmo `UPDATE`.

### `listIncidents(status?: IncidentStatus | "all"): Promise<Incident[]>` (novo)

`SELECT` em `incidents` filtrando por `status` quando informado e diferente de `"all"`; sem filtro (todas as linhas) quando omitido ou `"all"`. Lista vazia é um retorno válido, nunca uma exceção.

### `getRunbook(service: string): Promise<Runbook | null>` (novo)

Resolve o serviço por nome (mesma resolução de `openIncident`) — nome sem correspondência em `services` lança `ServiceNotFoundError`. Serviço encontrado: `SELECT` em `runbooks` por `service_id`; retorna o `Runbook` correspondente ou `null` se não houver linha (não é um erro).

## `seedCanonicalScenario(store: SqliteOpsStore): void`

Função exportada por `src/store/sqlite-ops-store.ts`, reaproveitada por `npm run seed` (`src/scripts/seed.ts`) e pelos testes que precisam do cenário semeado sobre `":memory:"`. Itera o mesmo dataset de `src/domain/seed-data.ts` (`buildSeedState()`, estendido com `runbooks`) e insere cada registro (`services`, `alerts`, `runbooks`; nunca `incidents` — o cenário canônico começa sempre com 0 incidentes) usando `INSERT OR IGNORE` casado pelo `id` estável de cada registro. Reexecutar sobre um banco já semeado é um no-op para os registros existentes — nenhuma duplicata, nenhum erro de `UNIQUE`/`PRIMARY KEY` (FR-007, SC-005).
