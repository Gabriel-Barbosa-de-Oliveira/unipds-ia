---

description: "Task list for 017-team-mode"
---

# Tasks: Modo Equipe

> **Nota de implementação (`/speckit-implement`, seguindo os exemplos da pessoa usuária)**:
> - **`done` no lugar de `FINISH`/`fim`.** O `nextSchema` é `{ next: "analista" | "planejador" | "executor" | "done", brief }`, como no exemplo, e o `handoff` grava `to: "done"`. Na war room, `done` aparece como "Fim". O `contracts/http.md` foi ajustado.
> - **Prompt do analista:** é o texto fornecido, com a acentuação corrigida (`ANALYST_PROMPT` em `src/team/roles.ts`).
> - **Os papéis devolvem só a contribuição deles** (`facts`, `steps` ou `summary`), e é o grafo da equipe que escreve no quadro. Assim, "cada papel só escreve o seu campo" vale por construção, e não por convenção (refina o research.md item 3).
> - **`createTeamStrategy` com todas as `deps` injetadas não exige o env do modelo**, porque o `loadModelConfig` só roda quando algum papel real é usado. Isso permite testes sem `OPENROUTER_MODEL`.
> - **`resolveStrategy("team")` monta a equipe por requisição** sobre as `baseTools`. Sem as ferramentas com porta (por exemplo, `strategyForRoute("team")` com as `opsTools` padrão), a montagem lança `TeamToolsNotGatedError`. O teste disso está em `src/agents/index.test.ts`.
> - **Migração** testada sobre um arquivo com o DDL antigo e um registro gravado direto por SQL. O registro e o trace antigos continuam iguais, os índices são recriados, `foreign_key_check` volta vazio, a migração não roda de novo ao reabrir e o CHECK segue rejeitando rotas desconhecidas.
> - **Contraste de `--trace-handoff`:** 5,92:1 no claro e 7,09:1 no escuro.

**Input**: Design documents from `/specs/017-team-mode/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos. A constitution (Princípio V, NON-NEGOTIABLE) exige teste para toda lógica nova. Todos os testes rodam sem rede: supervisor e papéis entram por `deps` injetáveis (research.md item 11).

**Organization**: as tarefas estão agrupadas por user story (spec.md).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivo diferente e sem dependência de tarefa ainda não concluída).
- **[Story]**: a user story à qual a tarefa pertence (US1–US4).

## Path Conventions

API em `src/` (testes ao lado, `*.test.ts`) e war room em `web/src/`.

---

## Phase 1: Setup

**Purpose**: nenhuma dependência nova. Só a pasta.

- [X] T001 Criar o diretório `src/team/` com um `src/team/index.ts` inicial que só exporta os tipos públicos (`TeamRole`), para os imports das fases seguintes resolverem.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: os tipos do trace (`handoff` e `role`) e o papel, que todas as stories usam.

**⚠️ CRITICAL**: nenhuma user story começa antes desta fase terminar.

- [X] T002 Em `src/agents/types.ts`:
  - exportar `TEAM_ROLES = ["analista", "planejador", "executor"] as const` e `type TeamRole`;
  - acrescentar à união `TraceEvent` a variante `{ type: "handoff"; at: number; from: "supervisor"; to: TeamRole | "fim"; brief: string }`;
  - acrescentar à interseção comum o campo `role?: TeamRole | "supervisor"`, ao lado de `node?`.

  Não mexer ainda em `RouteName`/`GraphNode`, que ficam para a US4. Reexportar `TeamRole` em `src/team/index.ts`.
- [X] T003 Em `src/agents/trace.ts`, acrescentar o `case "handoff"` em `formatEventBody` (`[handoff] supervisor → <to>: <brief>`), e prefixar `role` quando houver (`<node> │ <role> │ …`). Em `src/agents/trace.test.ts`, cobrir o handoff e o prefixo de papel.
- [X] T004 Rodar `npm run typecheck && npm test` e corrigir os `switch` exaustivos sobre `TraceEvent["type"]` que o compilador apontar, por exemplo em `src/obs/logger.ts` (`traceToLogEvents`, que por enquanto ignora `handoff`; o log vem na US3) e em `src/store/sqlite-request-store.ts`. Tudo verde.

**Checkpoint**: o trace aceita `handoff` e `role`, e nenhum comportamento muda.

---

## Phase 3: User Story 1 - Uma equipe de especialistas resolve o pedido (Priority: P1) 🎯 MVP

**Goal**: o supervisor coordena analista, planejador e executor sobre o quadro compartilhado e entrega uma resposta final. A ordem dos turnos, o teto e o aborto funcionam.

**Independent Test**: `createTeamStrategy(tools, fakeDeps).run("…")` com um supervisor fake roteirizado (analista → planejador → FINISH) produz um trace com handoffs na ordem, eventos com `role`, o quadro preenchido e a resposta = brief do FINISH.

### Tests for User Story 1

- [X] T005 [P] [US1] Criar `src/team/blackboard.test.ts` (deve falhar até a T008):
  - `createBlackboard("pedido")` começa vazio;
  - `addFacts` acrescenta e ignora duplicados (mesmo `statement` e `source`) sem mutar a entrada;
  - `setPlan` substitui;
  - `addOutcome` acrescenta;
  - `addHandoff` acrescenta;
  - `renderBlackboard` mostra o pedido, os fatos com origem, o plano numerado, os resultados e as passagens, limita aos últimos 20 fatos e trunca itens em 500 caracteres;
  - `fallbackAnswer(bb, "teto de 6 passagens atingido")` cita o motivo e resume fatos, plano e resultados, e vale também com o quadro vazio.
- [X] T006 [P] [US1] Criar `src/team/supervisor.test.ts` (deve falhar até a T009):
  - `resolveSupervisorDecision({ next: "analista", brief: "x" }, turn = 0, cap = 6)` → `{ kind: "route", to: "analista", brief: "x" }`;
  - `{ next: "FINISH", brief: "resposta" }` → `{ kind: "finish", answer: "resposta" }`;
  - `next` desconhecido, `brief` vazio ou objeto quebrado → `{ kind: "abort", reason: /decisão inválida/ }`;
  - `{ error }` → `abort` com a mensagem;
  - `turn === cap` e `next !== "FINISH"` → `abort("teto de 6 passagens atingido")`;
  - `turn === cap` com `FINISH` → `finish`.
- [X] T007 [P] [US1] Criar `src/team/team-graph.test.ts` (deve falhar até a T010), com `deps` fakes: `decide` roteirizado por array; `runAnalyst`, `runPlanner` e `runExecutor` que devolvem `{ trace, patch }` fixos. Casos:
  1. Roteiro analista → planejador → FINISH:
     - o trace tem `handoff(→analista)`, os eventos do analista com `role: "analista"`, `handoff(→planejador)`, `plan` com `role: "planejador"`, `handoff(→fim)` e `answer` com `role: "supervisor"`;
     - `at` é sequencial a partir de 0;
     - a resposta é o brief do FINISH;
     - cada papel recebe o brief certo e o quadro com o que veio antes.
  2. Só consulta: analista → FINISH, e o planejador e o executor nunca são chamados.
  3. Supervisor que sempre manda "analista": para em 6 turnos, termina com `handoff(→fim)` com o motivo do teto e a resposta = `fallbackAnswer`.
  4. Decisão inválida no 2º passo: aborta com `fallbackAnswer`.
  5. `decide` lança: aborta, sem propagar a exceção.
  6. O mesmo papel chamado duas vezes gera duas passagens.

### Implementation for User Story 1

- [X] T008 [US1] Criar `src/team/blackboard.ts` (puro), conforme o [data-model.md](./data-model.md#blackboard): tipos `Blackboard`, `Fact`, `Outcome`, `Handoff` e `AnalystSource`, e as funções `createBlackboard`, `addFacts`, `setPlan`, `addOutcome`, `addHandoff`, `renderBlackboard` e `fallbackAnswer`.
- [X] T009 [US1] Criar `src/team/supervisor.ts`:
  - `SupervisorDecisionSchema` (zod, `next` enum de 4 valores, `brief` com `trim().min(1)`);
  - `TEAM_MAX_TURNS = 6`;
  - `type SupervisorOutcome`;
  - `resolveSupervisorDecision(input: { decided?: unknown; error?: unknown }, turn, cap)`, que é pura;
  - `buildSupervisorMessages(request, blackboardText, turn, cap)`, pura: o system prompt descreve os papéis e seus limites, diz que no `FINISH` o brief é a resposta final em PT baseada só no quadro, e que deve encerrar cedo quando o pedido for só de consulta.
- [X] T010 [US1] Criar `src/team/team-graph.ts`:
  - `interface TeamDeps { decide(messages): Promise<unknown>; runAnalyst(input); runPlanner(input); runExecutor(input) }`, onde cada `run*` recebe `{ request, brief, blackboard }` e devolve `{ trace: TraceEvent[]; blackboard: Blackboard }`;
  - `StateGraph` com os nós `supervisor`, `analista`, `planejador` e `executor`, e as arestas de [data-model.md](./data-model.md#estado-da-equipe-stategraph);
  - `tagRole(events, role, offset)`, pura, carimba `role` e reindexa `at`;
  - `recursionLimit = 2 * TEAM_MAX_TURNS + 4`;
  - `runTeam(request, deps, callbacks?)` → `{ answer, trace, blackboard }`.

  O `handoff` é emitido pelo nó `supervisor` (`role: "supervisor"`) antes de cada papel e no encerramento (`to: "fim"`).
- [X] T011 [US1] Em `src/team/index.ts`, criar `createTeamStrategy(tools, deps?: Partial<TeamDeps>): ReasoningStrategy` (`name: "team"`), cujo `run` chama `runTeam` e monta o `RunResult` com `buildMetrics` sobre `LlmCallCounter`, `UsageCollector` e `ModelUsageTracker` criados por execução e passados em `callbacks` (research.md item 10). Nesta fase, as `deps` são obrigatórias: o padrão real entra na US2.

**Checkpoint**: a T007 passa. A equipe coordena, encerra e aborta só com fakes.

---

## Phase 4: User Story 2 - Cada papel só faz o que lhe cabe (Priority: P1)

**Goal**: ferramentas e escrita no quadro limitadas por construção. Executor sempre com porta, e a proposta encerra a equipe.

**Independent Test**: inspecionar as ferramentas entregues a cada papel. Montar a equipe com `createOpsTools(store)` lança `TeamToolsNotGatedError`. Um executor fake que chama o `resolve_incident` com porta deixa o incidente aberto, encerra a equipe e preenche o `gate`.

### Tests for User Story 2

- [X] T012 [P] [US2] Criar `src/team/roles.test.ts` (deve falhar até a T015):
  - `selectRoleTools(createGatedOpsTools(store, gate), "analista")` → exatamente `["list_alerts", "list_incidents", "consultar_runbook"]`;
  - `"planejador"` → `[]`;
  - `"executor"` → exatamente `["open_incident", "resolve_incident"]`;
  - faltar uma ferramenta da lista → erro;
  - `AnalystReportSchema` aceita `{ facts: [{ statement, source: "list_alerts" }] }` e **remove** chaves extras como `recommendation` (o resultado do parse não as contém);
  - `PlanSchema` rejeita 0 ou mais de 8 passos.
- [X] T013 [P] [US2] Em `src/agents/approval-gate.test.ts`, acrescentar: `isApprovalGated` é `true` para `open_incident`/`resolve_incident` de `createGatedOpsTools` e `false` para os de `createOpsTools` e para as ferramentas de leitura.
- [X] T014 [P] [US2] Em `src/team/team-graph.test.ts`, acrescentar: um executor fake cujo trace tem `observation` com `result.status === "awaiting_approval"` → a equipe encerra logo depois do executor, sem chamar `decide` de novo, com `handoff(→fim, "ação aguardando aprovação")` e um `outcome.proposal === true` no quadro. Testar também `hasProposal(trace)`, que é pura, com a observação em objeto e em string JSON.

### Implementation for User Story 2

- [X] T015 [US2] Criar `src/team/roles.ts`:
  - `TEAM_ROLE_TOOLS` (constante do [data-model.md](./data-model.md#papéis)) e `selectRoleTools(all, role)`;
  - `AnalystReportSchema` (`facts` com no máximo 20; `source` enum com as 3 ferramentas e `"pedido"`; zod `.strip()` padrão, que descarta chaves extras) e `PlanSchema` (1–8 passos);
  - os prompts de cada papel: o analista só relata fatos, sem recomendar; o planejador planeja a partir dos fatos, sem ferramentas; o executor executa só o que o plano e o brief pedem.
- [X] T016 [US2] Em `src/agents/approval-gate.ts`, registrar num `WeakSet` as instâncias criadas por `createGatedOpsTools` para `open_incident`/`resolve_incident` e exportar `isApprovalGated(tool)`. Em `src/domain/errors.ts`, criar `TeamToolsNotGatedError(toolNames: string[])`.
- [X] T017 [US2] Em `src/team/team-graph.ts`, criar `hasProposal(trace)`, pura, e o encerramento por proposta no nó `executor`: `addOutcome({ proposal: true })` mais `handoff(→fim)` e aresta para `END`. Sem proposta, `addOutcome({ proposal: false, summary: lastAnswer(trace) ?? "" })`.
- [X] T018 [US2] Criar `src/team/members.ts` com as implementações reais, as únicas com IO:
  - `createModelSupervisor()`: `createModel((m) => m.withStructuredOutput(SupervisorDecisionSchema))`;
  - `createAnalyst(tools)`: `createReactAgent` com `toolCallingModel(tools)` e depois a extração com `createModel((m) => m.withStructuredOutput(AnalystReportSchema))` sobre a transcrição. Só os `facts` vão para `addFacts`, e o trace do turno vem de `messagesToTrace` mais o fallback de modelo;
  - `createPlanner()`: `withStructuredOutput(PlanSchema)`, sem `bindTools`, com trace `[{ type: "plan", steps }]`;
  - `createExecutor(tools)`: `createReactAgent` só com as ferramentas do executor.

  Todos aceitam `callbacks` para as métricas.
- [X] T019 [US2] Em `src/team/index.ts`, `createTeamStrategy(tools, deps?)` passa a:
  1. validar com `selectRoleTools` que as ferramentas do executor satisfazem `isApprovalGated`, e lançar `TeamToolsNotGatedError` se não;
  2. completar as `deps` ausentes com `members.ts`.

  Em `src/team/team-graph.test.ts` (ou num novo `src/team/index.test.ts`), testar que `createTeamStrategy(createOpsTools(new InMemoryOpsStore()))` lança e que `createTeamStrategy(createGatedOpsTools(store, gate), fakeDeps)` monta.

**Checkpoint**: as T012–T014 passam. Nenhum papel recebe ferramenta fora da lista, e a equipe não monta sem porta.

---

## Phase 5: User Story 3 - Ver quem passou a vez para quem no "ver raciocínio" (Priority: P2)

**Goal**: handoffs e papéis gravados, logados (só metadados) e exibidos na war room.

**Independent Test**: o trace com `handoff`/`role` passa por `save` → `find` idêntico. O log de `handoff` não contém o brief. O `trace-view` produz "Passagem" com origem, destino e instrução.

### Tests for User Story 3

- [X] T020 [P] [US3] Em `src/store/sqlite-request-store.test.ts`, acrescentar o round-trip de um trace com `handoff` e eventos com `role` (deep-equal).
- [X] T021 [P] [US3] Em `src/obs/logger.test.ts`, acrescentar: `traceToLogEvents` mapeia `handoff` para `{ event: "team.handoff", requestId, node, position, from, to }`, e a linha formatada **não contém** o texto do brief (teste-âncora com um brief marcador).
- [X] T022 [P] [US3] Em `web/src/lib/api-schemas.test.ts` e `web/src/lib/trace-view.test.ts`:
  - o schema aceita `handoff` e `role`;
  - `toTraceView` do handoff → `label: "Passagem"`, `icon: "handoff"`, `body: { kind: "handoff", from: "Supervisor", to: "Analista", brief }`, com destino `"fim"` → `"Fim"`;
  - `role: "executor"` → `view.role === "Executor"`;
  - sem `role` → `null`;
  - o conjunto de ícones dos tipos conhecidos continua distinto (9).

### Implementation for User Story 3

- [X] T023 [US3] Em `src/obs/logger.ts`, acrescentar ao `LogEvent` a variante `team.handoff` (nível `info`) e o mapeamento em `traceToLogEvents`, sem `brief` ([data-model.md](./data-model.md#log-014)).
- [X] T024 [P] [US3] Em `web/src/lib/api-schemas.ts`:
  - ramo `handoff` (`from`, `to`, `brief`) na união conhecida;
  - `role: z.string().optional()` na base comum.

  Em `web/src/lib/trace-view.ts`:
  - `TraceKind` ganha `"handoff"`, e `TraceBody` ganha `{ kind: "handoff"; from; to; brief }`;
  - `TraceView` ganha `role: string | null`;
  - `ROLE_LABELS` (`analista` → `Analista` etc., `supervisor` → `Supervisor`, `fim` → `Fim`).
- [X] T025 [P] [US3] Em `web/src/components/Icon.tsx`, acrescentar o ícone `handoff`, duas setas opostas como o `fallback`, mas com traço distinto. Em `web/src/styles/tokens.css`, acrescentar `--trace-handoff` nos dois temas e a classe `.trace-handoff` em `web/src/styles/app.css`. Rodar de novo o script de contraste da 015 (texto ≥ 4.5:1 em todas as superfícies).
- [X] T026 [US3] Em `web/src/components/TraceEventItem.tsx`:
  - badge do papel (`view.role`) ao lado do badge do nó;
  - corpo `handoff`: `<strong>{from} → {to}</strong>` mais o brief em `pre-wrap`, conforme o [contracts/web-ui.md](./contracts/web-ui.md).

**Checkpoint**: as T020–T022 passam, e o "ver raciocínio" mostra as passagens.

---

## Phase 6: User Story 4 - A equipe como rota escolhível (Priority: P2)

**Goal**: rota `team` no roteador, no override, no grafo de produção e no SQLite.

**Independent Test**: `POST /chat` com `strategy: "team"` e `deps` fakes injetadas via `resolveStrategy` → 200 com `route.route === "team"` e source `override`, e o `GET /requests/:id` gravado com `route: "team"` num banco migrado. Um executor fake que propõe → 202.

### Tests for User Story 4

- [X] T027 [P] [US4] Em `src/store/sqlite-request-store.test.ts`, acrescentar a migração:
  - criar num arquivo temporário (`tmpdir`) as tabelas com o **DDL antigo** (`route IN ('react','planExecute','reflect')`), inserir um registro com trace;
  - abrir `new SqliteRequestStore(path)` e gravar um registro com `route: "team"`;
  - conferir que gravou, que o registro antigo e o trace dele continuam iguais e que os índices existem;
  - reabrir o store, mostrando que a migração não roda de novo (idempotente).
- [X] T028 [P] [US4] Em `src/graph/router.test.ts`:
  - `parseRouteName("team")` e `parseRouteName("equipe")` → `"team"`;
  - `ROUTE_TABLE` tem `team`;
  - `resolveRouteDecision({ decided: { route: "team", reason: "x" } })` → `team`.

  Em `src/agents/index.test.ts`, conferir que `strategyForRoute("team", …, baseTools)` resolve uma estratégia `name: "team"`.
- [X] T029 [P] [US4] Em `src/http/server.test.ts`, criar o bloco `describe("rota team (017)")`:
  - `strategy: "team"` com um `resolveStrategy` fake que devolve `createTeamStrategy(baseTools, fakeDeps)` → 200, `route.route === "team"`, trace com `handoff` e `role`;
  - um executor fake que invoca o `resolve_incident` das `baseTools` → 202, com o incidente ainda aberto;
  - `strategy: "time"` → 422 (regressão).

### Implementation for User Story 4

- [X] T030 [US4] Em `src/agents/types.ts`, acrescentar `"team"` a `ROUTE_NAMES`, o que faz `GraphNode` incluí-lo. Em `src/graph/router.ts`:
  - linha `team` na `ROUTE_TABLE` ("Investigar e agir de forma coordenada: levantar fatos, planejar e executar com aprovação", com exemplos);
  - aliases `team` e `equipe`.
- [X] T031 [US4] Em `src/agents/index.ts`:
  - `resolveStrategy("team", _reflect, _extraTools, baseTools)` → `createTeamStrategy(baseTools)`. Não há singleton, e a memória do usuário fica fora (premissa da spec);
  - `strategyForRoute("team")` repassa as `baseTools`.

  Em `src/graph/production-graph.ts`:
  - `.addNode("team", strategyNode("team"))`;
  - `team: "team"` no mapa condicional;
  - `.addEdge("team", "resposta")`.
- [X] T032 [US4] Em `src/store/sqlite-request-store.ts`:
  - o DDL novo aceita `'team'` no CHECK de `route`;
  - criar `migrateRouteCheck(db)`, chamada no getter `db` logo após o `exec(DDL)`. Se o `sql` de `requests` em `sqlite_master` não contém `'team'`, ela segue o procedimento do research.md item 9 (`PRAGMA foreign_keys=OFF` fora da transação, reconstrução, recriar os dois índices, `foreign_key_check`, `COMMIT`, `foreign_keys=ON`).
- [X] T033 [US4] Em `src/http/web-contract.test.ts`, acrescentar um 200 da rota `team` (equipe com `deps` fakes) validado por `ChatOkSchema`. Nenhum evento pode cair no ramo desconhecido do web.

**Checkpoint**: as T027–T029 passam. A rota `team` funciona por override e pelo roteador, e grava em bancos antigos.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T034 [P] Documentar a rota `team` e os papéis no `README.md` do OpsPilot, numa seção curta "Rotas de raciocínio", e no `CLAUDE.md` (a rota `team` e a pasta `src/team`).
- [X] T035 Rodar `npm run typecheck`, `npm test`, `npm --prefix web run typecheck`, `npm --prefix web test` e `npm --prefix web run build`. Tudo verde (Princípio V, SC-006).
- [ ] T036 **Pendente** (exige `OPENROUTER_API_KEY` e backup do `data/opspilot.db`). Validar as seções 2–6 do [quickstart.md](./quickstart.md) com o modelo real. Isso exige `OPENROUTER_API_KEY` no ambiente, sem nunca ler `.env`, e fazer backup do `data/opspilot.db` antes da migração.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (T001)** → **Foundational (T002 → T003 → T004)** → stories.
- **US1**: T005 ∥ T006 ∥ T007 (testes) → T008 → T009 → T010 → T011.
- **US2**: depende da US1 (`team-graph.ts`, `index.ts`). T012 ∥ T013 ∥ T014 → T015 → T016 → T017 → T018 → T019.
- **US3**: a parte da API (T020, T021, T023) depende só da Foundational. A parte web (T022, T024–T026) também. Pode correr em paralelo com a US1 e a US2.
- **US4**: depende da US2, porque `createTeamStrategy` real e com porta é o que a rota usa. T027 ∥ T028 ∥ T029 → T030 → T031 → T032 → T033.
- **Polish**: depois de tudo.

### User Story Dependencies

- **US1 (P1)**: Foundational. É o MVP técnico, testável só com fakes.
- **US2 (P1)**: US1.
- **US3 (P2)**: Foundational, independente das outras.
- **US4 (P2)**: US1 + US2. A US3 é recomendada antes, para a war room já mostrar as passagens.

## Parallel Opportunities

- **US1**: T005 ∥ T006 ∥ T007.
- **US2**: T012 ∥ T013 ∥ T014.
- **US3**: inteira em paralelo com US1/US2. Dentro dela, T020 ∥ T021 ∥ T022 e T024 ∥ T025.
- **US4**: T027 ∥ T028 ∥ T029.

### Parallel Example: User Story 1

```bash
Task: "T005 [US1] testes do quadro em src/team/blackboard.test.ts"
Task: "T006 [US1] testes do supervisor em src/team/supervisor.test.ts"
Task: "T007 [US1] testes do loop da equipe em src/team/team-graph.test.ts"
```

### Parallel Example: User Story 3

```bash
Task: "T021 [US3] teste-âncora do log team.handoff em src/obs/logger.test.ts"
Task: "T024 [US3] schema e trace-view do handoff em web/src/lib/"
Task: "T025 [US3] ícone e token --trace-handoff em web/src/"
```

---

## Implementation Strategy

### MVP First

1. Setup + Foundational (T001–T004).
2. US1 (T005–T011): a equipe coordena com fakes.
3. US2 (T012–T019): limites por construção e papéis reais. **Sem a US2 a equipe não pode ir para produção** (Princípio VI), então o MVP entregável é US1 + US2.
4. **STOP and VALIDATE**: suítes `src/team/*`.

### Incremental Delivery

Foundational → US1 → US2 → US3 (trace, log e war room; pode adiantar em paralelo) → US4 (rota e migração; a partir daqui a equipe fica acessível pelo `/chat`) → Polish. A equipe só fica acessível pela API na US4, então até lá nada muda para quem usa o OpsPilot.
