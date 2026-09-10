# Implementation Plan: Persistência Real de Operações

**Branch**: `004-ops-persistence` | **Date**: 2026-09-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-ops-persistence/spec.md`

## Summary

Substitui o adaptador Sequelize/MySQL (já revogado pela constitution v1.1.0) por `SqliteOpsStore` (`src/store/sqlite-ops-store.ts`), implementação de `OpsStoreRepository` sobre `node:sqlite` (`DatabaseSync`): 4 tabelas (`services`, `alerts`, `incidents`, `runbooks`) com DDL idempotente e `CHECK` em todo campo de valor fechado, caminho configurável via `OPSPILOT_DB` (default `./data/opspilot.db`), `":memory:"` em testes. O cenário canônico ("Mercadinho": 5 serviços, 6 alertas, 3 runbooks) continua definido uma única vez em `src/domain/seed-data.ts` e é reaproveitado tanto pelo mock em memória (agora usado só por testes e pelo bench) quanto pelo seed do SQLite. Duas tools novas (`list_incidents`, `consultar_runbook`) e uma alterada (`resolve_incident` ganha `summary` opcional) se somam às três já existentes, todas com descriptions revisadas (quando usar cada uma, `describe()` em todo campo, enums). A composição do store nas tools/estratégias passa a ser explícita via fábricas (`createOpsTools`, `createReactStrategy`, `createPlanAndExecuteStrategy`), preservando os singletons já exportados hoje — nenhuma mudança em `agents/index.ts`, `http/server.ts`, `src/index.ts`, `arena.ts`; só `bench.ts` passa a construir explicitamente seu próprio mock em memória para manter os cenários reprodutíveis e isolados do banco real (ver [research.md](./research.md) item 2).

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS (mesmo runtime das features 001–003; `node:sqlite`/`DatabaseSync` é nativo do runtime já mandatado, sem flag experimental necessária)

**Primary Dependencies**: nenhuma dependência de pacote nova — `node:sqlite` é builtin. Remove `sequelize` e `mysql2` do `package.json` (adaptador Sequelize/MySQL é código morto frente à constitution v1.1.0, ver [research.md](./research.md) item 1). `zod` continua sendo o validador na fronteira das tools, sem mudança de versão.

**Storage**: SQLite via `node:sqlite` (`DatabaseSync`) — arquivo local (`OPSPILOT_DB`, default `./data/opspilot.db`) para execução real (`npm run dev`, `npm run seed`); `":memory:"` para testes. `InMemoryOpsStore` (JS puro, sem SQL) permanece como segundo adaptador de `OpsStoreRepository`, reservado a `bench.ts` (reprodutibilidade sem tocar o banco real).

**Testing**: `node:test` via `tsx` (`npm test`); dois arquivos de teste novos — `src/store/sqlite-ops-store.test.ts` (DDL/seed/CRUD/filtros/`CHECK` sobre `":memory:"`) e `src/agents/tools.test.ts` (as 5 tools sobre um `SqliteOpsStore(":memory:")` semeado) — mais extensões pontuais em `src/domain/ops-store.test.ts` (novas funções puras `listIncidents`, `getRunbookForService`) e `src/domain/seed-data.ts`'s dataset (runbooks). Nenhum teste depende de rede nem de `data/opspilot.db`.

**Target Platform**: processo Node.js server-side (mesmo runtime das features anteriores) — `npm run dev`/`npm run bench`/`npm run seed` passam a operar sobre o novo adaptador (dev/seed) ou sobre o mock em memória (bench), nunca sobre MySQL.

**Project Type**: projeto único (extensão aditiva de `src/domain/`, `src/agents/tools.ts`; novo diretório `src/store/`; remoção de `src/models/sequelize/` e `src/services/ops-store.sequelize.ts`)

**Performance Goals**: não é caminho de alto throughput — mesmo perfil de custo por chamada ao modelo já documentado em 001–003; o que importa é que cada leitura/escrita no SQLite seja síncrona e rápida o suficiente para não ser perceptível frente à latência de uma chamada ao modelo (não há meta numérica própria desta feature).

**Constraints**: `CHECK` constraints recusam gravação de valor fora do domínio mesmo por um caminho que não passe pela validação zod das tools (FR-006); DDL e seed idempotentes (FR-002, FR-007); nenhuma query concatena valor de entrada em SQL — só prepared statements com parâmetros ligados; `bench.ts` nunca opera sobre `OPSPILOT_DB` (FR-010, isolamento de testes/bench).

**Scale/Scope**: 4 tabelas, 2 tools novas + 1 alterada + 2 inalteradas, 1 adaptador novo substituindo 1 removido; nenhuma estratégia de raciocínio nova, nenhuma mudança de contrato HTTP (`POST /chat` inalterado).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | `src/store/sqlite-ops-store.ts` é a única camada que fala `node:sqlite`; `src/domain/ops-store.ts` ganha `listIncidents`/`getRunbookForService` como funções puras sobre `OpsState`, sem IO; `src/agents/tools.ts` continua só delegando a `OpsStoreRepository`, nunca SQL direto. |
| II. Validação na Fronteira | As 2 tools novas e o campo `summary` de `resolve_incident` têm schema zod (enums para `status`/valores fechados, `.describe()` em todo campo) validado antes de qualquer chamada ao store — mesmo padrão das 3 tools existentes. |
| III. Erros de Domínio | `consultar_runbook` reaproveita `ServiceNotFoundError` já existente (nenhuma classe nova necessária); "sem runbook" é modelado como valor (`null`), não como erro, evitando um erro de domínio artificial para um caso que não é falha. |
| IV. Funções Puras | `listIncidents`/`getRunbookForService` (domínio) são puras, mesmo padrão de `listAlerts`/`openIncident`/`resolveIncident`; todo efeito colateral (arquivo SQLite) fica isolado em `src/store/sqlite-ops-store.ts`. |
| V. Teste Obrigatório | Nenhuma lógica nova (DDL idempotente, seed idempotente, `CHECK`, as 2 tools novas, `summary` opcional) entra sem teste — ver Technical Context/Testing; `npm test`/`npm run typecheck` continuam gates obrigatórios. |
| VI. Segurança por Padrão | `data/` já está no `.gitignore` (nenhuma mudança necessária); `OPSPILOT_DB` é um caminho de arquivo local, não um segredo; nenhum código desta feature lê `.env` diretamente (lido só pelo runtime via `--env-file-if-exists`, como hoje). |
| VII. Spec Antes de Código | Este plano segue `specs/004-ops-persistence/spec.md`, validado e sem `[NEEDS CLARIFICATION]` pendente. |
| VIII. Pequeno e Reversível | Cada decisão de research.md (remoção do Sequelize, fábricas de composição, dataset único, idempotência, `list_incidents`/`consultar_runbook`, descriptions, testes isolados) é independente o suficiente para virar uma ou poucas tarefas pequenas em `/speckit-tasks`; nenhuma reescreve `react.ts`/`plan-and-execute.ts` além de extrair o literal existente para uma fábrica com o mesmo comportamento. |

Nenhuma violação identificada — **Complexity Tracking** não se aplica (tabela deixada vazia).

## Project Structure

### Documentation (this feature)

```text
specs/004-ops-persistence/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   ├── store.md
│   └── tools.md
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── domain/
│   ├── ops-store.ts                  # [alterado] + Runbook, + Incident.summary, + listIncidents, + getRunbookForService, resolveIncident ganha summary?
│   ├── ops-store.test.ts             # [alterado] + testes das funções novas
│   ├── seed-data.ts                  # [alterado] + RUNBOOKS (checkout-api, payments-api, auth-service), buildSeedState() inclui runbooks
│   └── errors.ts                     # [existente, inalterado] — ServiceNotFoundError reaproveitado por consultar_runbook
├── services/
│   ├── ops-store.repository.ts       # [alterado] + listIncidents, + getRunbook; resolveIncident ganha summary?
│   ├── ops-store.memory.ts           # [alterado] deixa de ler/escrever data/ops-store.json — estado só em processo (this.state), reservado a testes e a bench.ts
│   ├── ops-store.sequelize.ts        # [REMOVIDO]
├── models/sequelize/                 # [REMOVIDO — connection.ts, alert.model.ts, incident.model.ts, service.model.ts]
├── store/                            # [NOVO diretório]
│   ├── sqlite-ops-store.ts           # [NOVO] SqliteOpsStore + seedCanonicalScenario(store)
│   └── sqlite-ops-store.test.ts      # [NOVO] DDL idempotente, seed idempotente, CRUD, filtros, CHECK — tudo sobre ":memory:"
├── agents/
│   ├── tools.ts                      # [alterado] createOpsTools(store) (fábrica) + opsTools (composição padrão sobre SqliteOpsStore); + listIncidentsTool, + consultarRunbookTool; descriptions revisadas nas 5 tools
│   ├── tools.test.ts                 # [NOVO] as 5 tools sobre SqliteOpsStore(":memory:") semeado
│   ├── react.ts                      # [alterado] extrai o literal existente para createReactStrategy(tools); export const reactStrategy = createReactStrategy(opsTools) mantém a mesma referência
│   ├── plan-and-execute.ts           # [alterado] mesma extração: createPlanAndExecuteStrategy(tools); planAndExecuteStrategy inalterado como singleton
│   ├── types.ts                      # [alterado] ToolName ganha "list_incidents" | "consultar_runbook"
│   └── index.ts                      # [existente, inalterado] — STRATEGIES/resolveStrategy continuam apontando para os mesmos singletons
├── bench.ts                          # [alterado] constrói seu próprio InMemoryOpsStore + createOpsTools/createReactStrategy/createPlanAndExecuteStrategy, em vez de importar o store de ops-store.memory.ts
├── scripts/
│   └── seed.ts                       # [alterado] semeia SqliteOpsStore (OPSPILOT_DB) via seedCanonicalScenario, em vez de escrever data/ops-store.json
└── http/server.ts, index.ts, agents/reflection.ts, services/chat.service.ts  # [existentes, inalterados]
```

**Structure Decision**: Projeto único (Option 1), mesma estrutura das features 001–003. O único diretório novo é `src/store/` (adaptador concreto de persistência, caminho pedido explicitamente na feature) — `src/services/` continua abrigando a interface (`ops-store.repository.ts`) e o mock em memória, sem mover esses dois arquivos para `src/store/`: mover arquivos sem mudança de comportamento não teria ganho funcional e infla o diff. `src/models/sequelize/` e `src/services/ops-store.sequelize.ts` são removidos por inteiro — nenhum caminho de composição os instancia depois desta feature.

## Complexity Tracking

*Nenhuma violação da Constitution Check — tabela não aplicável.*
