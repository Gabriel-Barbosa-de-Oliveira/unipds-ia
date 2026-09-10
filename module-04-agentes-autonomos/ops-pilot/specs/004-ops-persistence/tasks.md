---

description: "Task list template for feature implementation"
---

# Tasks: Persistência Real de Operações

**Input**: Design documents from `/specs/004-ops-persistence/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos — a spec e a constitution (Teste Obrigatório) pedem testes explicitamente para esta feature.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivo diferente, sem dependência de tarefa ainda não concluída)
- **[Story]**: A qual user story a tarefa pertence (US1–US4)
- Caminhos de arquivo exatos em cada descrição

## Path Conventions

Projeto único — `src/` na raiz do repositório (ver plan.md § Project Structure). Sem `tests/` separado: testes ficam ao lado do código (`*.test.ts`), convenção já usada pelas features 001–003.

---

## Phase 1: Setup

**Purpose**: Remover a stack de persistência revogada antes de construir a nova, para não ter dois adaptadores incompatíveis coexistindo.

- [X] T001 Remover o adaptador Sequelize/MySQL: apagar `src/services/ops-store.sequelize.ts` e o diretório `src/models/sequelize/` (`connection.ts`, `alert.model.ts`, `incident.model.ts`, `service.model.ts`); remover `sequelize` e `mysql2` de `package.json` (`dependencies`) e rodar `npm install` para atualizar `package-lock.json`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Modelo de domínio estendido, contrato do repositório estendido, e os dois adaptadores (`SqliteOpsStore` novo, `InMemoryOpsStore` simplificado) conformes a esse contrato — base sobre a qual toda user story se apoia.

**⚠️ CRITICAL**: Nenhuma user story começa antes desta fase estar completa.

- [X] T002 [P] Estender `src/domain/ops-store.ts`: tipo `Runbook` (`id`, `serviceId`, `content`), tipo `IncidentStatusFilter` (`IncidentStatus | "all"`), `Incident.summary: string | null`, `ResolveIncidentContext` ganha `summary?: string`, `resolveIncident()` grava `summary` só na primeira resolução (nunca sobrescreve num `resolve` repetido); novas funções puras `listIncidents(state, status?)` (mesmo padrão de `listAlerts`) e `getRunbookForService(state, serviceName)` (lança `ServiceNotFoundError` se o serviço não existe; retorna `null` se existe mas não tem runbook).
- [X] T003 [P] (depende de T002) Estender `src/domain/seed-data.ts`: adicionar `RUNBOOKS` para `checkout-api`, `payments-api` e `auth-service`, e incluir `runbooks` em `OpsState`/`buildSeedState()`.
- [X] T004 [P] (depende de T002) Estender `src/services/ops-store.repository.ts`: `OpsStoreRepository` ganha `listIncidents(status?: IncidentStatusFilter): Promise<Incident[]>` e `getRunbook(service: string): Promise<Runbook | null>`; `resolveIncident` passa a aceitar `(id: string, summary?: string)`.
- [X] T005 [P] (depende de T003, T004) Simplificar `src/services/ops-store.memory.ts`: remover `readOpsStateFile`/`writeOpsStateFile`/`DEFAULT_DATA_FILE` (sem mais arquivo `data/ops-store.json`) — `InMemoryOpsStore` mantém o estado só em um campo de instância, semeado via `buildSeedState()`; implementar `listIncidents`/`getRunbook`/`resolveIncident(id, summary?)` para satisfazer o `OpsStoreRepository` estendido.
- [X] T006 [P] (depende de T003, T004) Criar `src/store/sqlite-ops-store.ts`: classe `SqliteOpsStore` sobre `node:sqlite` (`DatabaseSync`); construtor com DDL idempotente (`CREATE TABLE IF NOT EXISTS`) para `services`, `alerts`, `incidents`, `runbooks`, com `CHECK` em `alerts.status`, `incidents.severity`, `incidents.status`; caminho default `process.env.OPSPILOT_DB ?? "./data/opspilot.db"`; implementar `listAlerts`, `openIncident`, `resolveIncident(id, summary?)` via prepared statements (sem SQL concatenado); exportar `seedCanonicalScenario(store)` (insere `buildSeedState()` — services, alerts, runbooks — via `INSERT OR IGNORE` casado por id, idempotente).
- [X] T007 [P] (depende de T006) Criar `src/store/sqlite-ops-store.test.ts`: seed idempotente (rodar `seedCanonicalScenario` duas vezes sobre o mesmo `":memory:"`, contagem de linhas inalterada), `listAlerts`/`openIncident` (serviço/severidade válidos e inválidos), `resolveIncident` (com e sem `summary`, idempotência ao resolver duas vezes), e um `CHECK` rejeitando um valor fora do domínio inserido diretamente por SQL (bypassando os métodos do repositório) — tudo sobre `new SqliteOpsStore(":memory:")`.
- [X] T008 (depende de T004, T006) Reescrever `src/agents/tools.ts`: extrair `createOpsTools(store: OpsStoreRepository)` (fábrica pura) para `list_alerts`/`open_incident`/`resolve_incident` (este último ganha o campo opcional `summary`, com `.describe()` próprio); exportar `opsTools = createOpsTools(new SqliteOpsStore())` como composição padrão; revisar as descriptions dessas 3 tools conforme `contracts/tools.md` (quando usar cada uma frente às demais, `.describe()` em todo campo).
- [X] T009 [P] (depende de T008) Criar `src/agents/tools.test.ts`: exercitar `list_alerts`/`open_incident`/`resolve_incident` (incluindo `summary`) chamando `createOpsTools(...)` sobre um `SqliteOpsStore(":memory:")` semeado com `seedCanonicalScenario`.
- [X] T010 (depende de T006) Reescrever `src/scripts/seed.ts`: construir `new SqliteOpsStore(process.env.OPSPILOT_DB)` e chamar `seedCanonicalScenario`, removendo o caminho de seed em arquivo JSON.

**Checkpoint**: Fundação pronta — `npm run typecheck`/`npm test` verdes; as user stories podem começar.

---

## Phase 3: User Story 1 - Dados operacionais sobrevivem a reinícios do processo (Priority: P1) 🎯 MVP

**Goal**: Provar que a persistência via `SqliteOpsStore` é real — dados sobrevivem à destruição e recriação do processo, não apenas a chamadas sucessivas dentro do mesmo processo. A composição padrão (`opsTools`, `npm run seed`) já foi ligada ao `SqliteOpsStore` na fase Foundational (T006–T010); esta fase adiciona a prova específica de sobrevivência a reinício pedida pela spec.

**Independent Test**: Abrir e resolver um incidente contra uma instância de `SqliteOpsStore` apontando para um arquivo, descartar essa instância, abrir uma nova instância apontando para o mesmo arquivo, e confirmar que o incidente e seu status continuam idênticos.

- [X] T011 [P] [US1] Adicionar a `src/store/sqlite-ops-store.test.ts` um teste de persistência entre instâncias: gravar via um `SqliteOpsStore(path)` sobre um arquivo temporário, descartar a referência, ler via um segundo `SqliteOpsStore(path)` sobre o mesmo caminho, e confirmar que serviços/alertas/incidente (com `status`, `resolvedAt`, `summary`) são idênticos (SC-001).

**Checkpoint**: User Story 1 completa e testável de forma independente — reiniciar o processo não perde dado algum.

---

## Phase 4: User Story 2 - Consultar incidentes existentes filtrando por status (Priority: P2)

**Goal**: Nova capacidade de leitura (`list_incidents`) sobre os incidentes já geridos pelo próprio sistema, filtrando por `open`/`resolved`/`all`.

**Independent Test**: Abrir dois incidentes, resolver um, pedir a lista filtrando por `open`, depois por `resolved`, depois por `all`, e confirmar que cada resposta corresponde exatamente ao filtro.

- [X] T012 [P] [US2] Adicionar `listIncidents(status?)` a `SqliteOpsStore` (já implementado em T006, junto com a criação da classe — a interface estendida em T004 exigiu conformidade imediata) (`src/store/sqlite-ops-store.ts`) — `SELECT` com `WHERE status = ?` só quando `status` é `"open"`/`"resolved"`; sem filtro quando omitido ou `"all"`.
- [X] T013 [P] [US2] Adicionar `listIncidents(status?)` a `InMemoryOpsStore` (já implementado em T005, pelo mesmo motivo de conformidade de interface) (`src/services/ops-store.memory.ts`), delegando à função pura `listIncidents` do domínio.
- [X] T014 [US2] (depende de T012, T013) Adicionar `listIncidentsTool` (`list_incidents`) a `createOpsTools` em `src/agents/tools.ts` — `status: z.enum(["open", "resolved", "all"]).optional()` com `.describe()`, description deixando claro que é só leitura e distinta de `open_incident`; adicionar `"list_incidents"` a `ToolName` em `src/agents/types.ts`; registrar no array `opsTools`.
- [X] T015 [P] [US2] Estender `src/store/sqlite-ops-store.test.ts` com testes de `listIncidents` (já coberto em T007: "listIncidents filtra por open, resolved e all") (`open`, `resolved`, `all`, e lista vazia quando não há correspondência).
- [X] T016 [P] [US2] Estender `src/agents/tools.test.ts` com testes de `list_incidents` (os três filtros e o caso de lista vazia, confirmando que não é tratado como erro).

**Checkpoint**: User Stories 1 e 2 funcionam de forma independente.

---

## Phase 5: User Story 3 - Consultar o runbook de um serviço durante um incidente (Priority: P2)

**Goal**: Nova capacidade de leitura (`consultar_runbook`) trazendo os passos de mitigação de um serviço.

**Independent Test**: Pedir o runbook de um serviço com runbook cadastrado (conteúdo correto retornado); de um serviço sem runbook (ausência sinalizada, não é erro); e de um serviço inexistente (erro de serviço desconhecido).

- [X] T017 [P] [US3] Adicionar `getRunbook(service)` a `SqliteOpsStore` (já implementado em T006) (`src/store/sqlite-ops-store.ts`) — resolve o serviço por nome (`ServiceNotFoundError` se não existe), `SELECT` em `runbooks` por `service_id`, retorna o `Runbook` ou `null`.
- [X] T018 [P] [US3] Adicionar `getRunbook(service)` a `InMemoryOpsStore` (já implementado em T005) (`src/services/ops-store.memory.ts`), delegando a `getRunbookForService` do domínio.
- [X] T019 [US3] (depende de T017, T018) Adicionar `consultarRunbookTool` (`consultar_runbook`) a `createOpsTools` em `src/agents/tools.ts` — campo `service` com `.describe()`, erro estruturado `ServiceNotFoundError`, sucesso como `{ service, runbook: string | null }`; adicionar `"consultar_runbook"` a `ToolName`; registrar no array `opsTools`.
- [X] T020 [P] [US3] Estender `src/store/sqlite-ops-store.test.ts` com testes de `getRunbook` (já coberto em T007) (serviço com runbook, serviço sem runbook, serviço inexistente).
- [X] T021 [P] [US3] Estender `src/agents/tools.test.ts` com testes de `consultar_runbook` (encontrado, ausente, serviço desconhecido).

**Checkpoint**: User Stories 1, 2 e 3 funcionam de forma independente.

---

## Phase 6: User Story 4 - Restaurar um cenário operacional conhecido de forma reprodutível (Priority: P3)

**Goal**: `bench.ts` (e qualquer teste/demonstração futura) roda contra um mock em memória isolado do banco real, reprodutível entre execuções — sem o risco (identificado em research.md item 2) de resetar um store desconectado do que as tools realmente usam.

**Independent Test**: Rodar o processo de restauração do cenário duas vezes seguidas, com incidentes criados entre as duas, e confirmar que o resultado (serviços/alertas/runbooks) é idêntico nas duas vezes e nenhum incidente sobrevive.

- [X] T022 [P] [US4] Extrair `createReactStrategy(tools)` em `src/agents/react.ts`, mantendo `export const reactStrategy = createReactStrategy(opsTools)` como a mesma referência de singleton já usada por `agents/index.ts`/testes existentes.
- [X] T023 [P] [US4] Extrair `createPlanAndExecuteStrategy(tools)` em `src/agents/plan-and-execute.ts`, mantendo `export const planAndExecuteStrategy = createPlanAndExecuteStrategy(opsTools)` como o mesmo singleton.
- [X] T024 [US4] (depende de T022, T023) Reescrever `src/bench.ts` para construir seu próprio `InMemoryOpsStore`, seu próprio `createOpsTools(...)` e suas próprias estratégias via `createReactStrategy`/`createPlanAndExecuteStrategy`, em vez de importar o `store`/os singletons compartilhados — `runOne`/`main` resetam e inspecionam essa instância dedicada.
- [X] T025 [P] [US4] Criar `src/services/ops-store.memory.test.ts`: o estado semeado corresponde a `buildSeedState()` (5 serviços/6 alertas/3 runbooks/0 incidentes); mutar (abrir incidente) e depois `reset()` volta ao estado canônico; duas chamadas consecutivas a `reset()` produzem estados idênticos (SC-005), sem tocar disco nem rede.

**Checkpoint**: Todas as 4 user stories funcionam de forma independente; `bench.ts` nunca mais toca `OPSPILOT_DB`.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Confirmar não-regressão e aderência final aos contratos desta feature.

- [X] T026 [P] Rodar `npm run typecheck` e `npm test` e confirmar que as suítes já existentes (`src/domain/ops-store.test.ts`, `src/agents/index.test.ts`, `src/agents/reflection.test.ts`, `src/agents/trace.test.ts`, `src/http/server.test.ts`, `src/services/chat.service.test.ts`, `src/bench.test.ts`) continuam passando sem alteração de comportamento observável (FR-011).
- [X] T027 Rodar manualmente `specs/004-ops-persistence/quickstart.md` (passos 1-2 completos via CLI; 3-4 validados diretamente via `createOpsTools`/`SqliteOpsStore` — mesma mecânica do endpoint, sem depender do LLM real; composição de `bench.ts` (passo 5) verificada por import — execução completa dos passos 3-5 via `/chat`/`npm run bench` reais requer credenciais de OpenRouter em `.env`, não lido nesta sessão) do início ao fim (seed, persistência a reinício, `list_incidents`, `consultar_runbook`, isolamento do bench) e corrigir qualquer divergência encontrada.
- [X] T028 [P] Revisar as descriptions finais das 5 tools em `src/agents/tools.ts` contra `specs/004-ops-persistence/contracts/tools.md` — conferido: todas declaram quando usar frente às demais, todo campo tem `.describe()` próprio, todo valor fechado usa `z.enum`; nenhum ajuste necessário em `src/agents/tools.ts` contra `specs/004-ops-persistence/contracts/tools.md` (quando usar cada uma, cobertura de `.describe()`, uso de enum) e ajustar qualquer inconsistência.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: sem dependências — pode começar imediatamente.
- **Foundational (Phase 2)**: depende do Setup — BLOQUEIA todas as user stories.
- **User Stories (Phase 3–6)**: todas dependem da Foundational completa.
  - US1 (P1) não depende de US2/US3/US4.
  - US2 (P2) e US3 (P2) não dependem entre si nem de US1 — podem ser feitas em paralelo ou em qualquer ordem após a Foundational.
  - US4 (P3) não depende de US2/US3; depende só da Foundational (usa `opsTools`/`createOpsTools` já existentes desde T008).
- **Polish (Phase 7)**: depende de todas as user stories desejadas estarem completas.

### Dentro de cada User Story

- US1: T011 é a única tarefa — depende só da Foundational.
- US2: T012/T013 (métodos do store) antes de T014 (tool); T015/T016 (testes) depois de T014.
- US3: T017/T018 (métodos do store) antes de T019 (tool); T020/T021 (testes) depois de T019.
- US4: T022/T023 (fábricas) antes de T024 (rewire do bench); T025 (teste) independe de T024 (testa `InMemoryOpsStore` diretamente).

### Parallel Opportunities

- Dentro da Foundational: T002 primeiro; depois T003 e T004 em paralelo; depois T005 e T006 em paralelo; T007 após T006; T009 após T008; T010 após T006.
- US2 e US3 são totalmente independentes entre si — dois desenvolvedores podem pegar uma cada, em paralelo, assim que a Foundational fechar.
- Dentro de US2: T012/T013 em paralelo; dentro de US3: T017/T018 em paralelo.
- US4 (T022/T023) pode começar em paralelo com US2/US3, já que só depende da Foundational.

---

## Parallel Example: Foundational

```bash
# Depois de T002 (tipos de domínio):
Task: "Estender src/domain/seed-data.ts com RUNBOOKS (T003)"
Task: "Estender src/services/ops-store.repository.ts com listIncidents/getRunbook (T004)"

# Depois de T003 + T004:
Task: "Simplificar src/services/ops-store.memory.ts (T005)"
Task: "Criar src/store/sqlite-ops-store.ts (T006)"
```

## Parallel Example: User Story 2 + User Story 3

```bash
# Depois da Foundational, dois desenvolvedores em paralelo:
Task: "listIncidents em SqliteOpsStore + InMemoryOpsStore + list_incidents tool (T012-T016)"
Task: "getRunbook em SqliteOpsStore + InMemoryOpsStore + consultar_runbook tool (T017-T021)"
```

---

## Implementation Strategy

### MVP First (User Story 1 apenas)

1. Completar Phase 1 (Setup) e Phase 2 (Foundational) — é aqui que a troca de adaptador (Sequelize/MySQL → SQLite) realmente acontece.
2. Completar Phase 3 (US1: T011).
3. **PARAR e VALIDAR**: rodar `npm test` e os passos 1–3 de `quickstart.md` (seed, reiniciar, confirmar persistência).
4. Nesse ponto, `npm run dev` já serve `/chat` com persistência real — MVP entregável.

### Incremental Delivery

1. Setup + Foundational → base pronta (adaptador real, sem Sequelize).
2. US1 → validar independentemente → persistência real comprovada (MVP).
3. US2 → validar independentemente → `list_incidents` disponível no copiloto.
4. US3 → validar independentemente → `consultar_runbook` disponível no copiloto.
5. US4 → validar independentemente → bench voltou a ser reprodutível, isolado do banco real.
6. Polish → confirma não-regressão e fecha o quickstart completo.

---

## Notes

- `[P]` = arquivos diferentes, sem dependência de tarefa incompleta.
- Rótulo `[US#]` mapeia a tarefa à user story correspondente da spec.
- Cada user story deve ser completável e testável de forma independente.
- Commitar após cada tarefa ou grupo lógico pequeno (constitution: Pequeno e Reversível).
- `npm run typecheck` e `npm test` devem ficar verdes ao final de cada fase, não só no final da feature.
