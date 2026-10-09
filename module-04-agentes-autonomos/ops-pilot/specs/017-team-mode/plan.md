# Implementation Plan: Modo Equipe

**Branch**: `017-team-mode` | **Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/017-team-mode/spec.md`

## Summary

**A equipe.** Uma `ReasoningStrategy` nova, `createTeamStrategy`, em `src/team/`, com um `StateGraph` interno. Um supervisor decide com `withStructuredOutput({ next, brief })` sobre um quadro (blackboard) guardado no estado, e três papéis com ferramentas fixadas por construção:
- **analista**: só as 3 ferramentas de leitura. Escreve no quadro só fatos estruturados, então não consegue propor;
- **planejador**: nenhuma ferramenta, só um plano estruturado;
- **executor**: só `open_incident` e `resolve_incident`, e só nas instâncias com porta da 015. Ferramenta sem porta faz a montagem falhar.

Cada passagem vira um evento `handoff` no trace, e todo evento da equipe leva o campo `role`. O teto de 6 turnos, uma decisão inválida ou uma proposta do executor encerram a equipe de forma controlada.

**A rota.** `team` entra no roteador, nos aliases e no grafo de produção (nó `team`). O `requests.route` do SQLite é migrado para aceitar `team`, e o log ganha `team.handoff`, sem a instrução.

**A war room.** Mostra a "Passagem" com visual próprio e o badge do papel em cada evento.

Os detalhes estão em [research.md](./research.md).

## Technical Context

**Language/Version**: TypeScript ESM `strict`, Node 24 LTS (API e web)

**Primary Dependencies**: `@langchain/langgraph` 0.2.74 (`StateGraph`, `createReactAgent`), `@langchain/core` (`withStructuredOutput` via `createModel`, 013), zod e `node:sqlite`. Na web, React e zod. Nenhuma dependência nova.

**Storage**: SQLite (`OPSPILOT_DB`). Migração idempotente do CHECK de `requests.route` (research.md item 9). O `trace_events` não muda.

**Testing**: `node:test` via `tsx`, sem rede. A equipe recebe `deps` injetáveis (supervisor e papéis fakes, research.md item 11). O store roda em `:memory:`, e a migração é testada sobre um banco criado com o DDL antigo. Na web, testes puros de schema e `trace-view`.

**Target Platform**: API Node (Express) e war room em navegador

**Project Type**: web-service + aplicação web

**Performance Goals**: a equipe faz no máximo 6 turnos, cerca de 2 a 3 chamadas ao modelo por turno, mais 1 por decisão. Ela respeita o timeout do `/chat` (180s). Não há meta de latência além disso: é a rota cara, escolhida para pedidos complexos.

**Constraints**:
- Nenhuma ação sem a porta de aprovação (FR-010).
- O analista não consegue propor (FR-007).
- Logs sem brief (FR-015).
- As rotas existentes não mudam (SC-006).
- O `.env` não é lido.

**Scale/Scope**:
- `src/team/`: 6 módulos (`blackboard`, `roles`, `supervisor`, `members`, `team-graph`, `index`), mais testes.
- Alterados: `agents/types.ts`, `agents/approval-gate.ts`, `agents/index.ts`, `graph/router.ts`, `graph/production-graph.ts`, `obs/logger.ts`, `store/sqlite-request-store.ts`, `domain/errors.ts` e `http/web-contract.test.ts`.
- Web: `api-schemas`, `trace-view`, `TraceEventItem`, `Icon` e `tokens.css`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Status | Como |
|---|---|---|
| I. Camadas explícitas | ✅ | O quadro, a decisão do supervisor, a seleção de ferramentas por papel, a detecção de proposta, a resposta de fallback e o `tagRole` são puros em `src/team/`. O IO (modelo, ferramentas) entra por `deps`. O controller (`server.ts`) não muda além do que a rota já faz. |
| II. Validação na fronteira | ✅ | Toda saída de modelo passa por zod: `SupervisorDecisionSchema`, `AnalystReportSchema` e `PlanSchema`. O `strategy: "team"` é validado por `parseRouteName`. |
| III. Erros de domínio | ✅ | `TeamToolsNotGatedError` é uma classe e vira 500 na borda. Decisão inválida não é erro: vira um `abort` controlado, que é pura. |
| IV. Funções puras | ✅ | Veja o item I. O loop do `StateGraph` só orquestra. |
| V. Teste obrigatório | ✅ | Suítes para o quadro, os papéis (ferramentas exatas), o supervisor (resolve/abort/teto), o loop com fakes, a migração, os logs, o `/chat` com `team` (200 e 202), o contrato API↔web e o `trace-view`. |
| VI. Segurança | ✅ | Limites por construção: o executor só recebe instâncias com porta (checagem por identidade) e o analista não tem campo para propostas. Logs sem brief. Este é o princípio que a feature mais exercita. |
| VII. Spec antes de código | ✅ | spec → plan → tasks. |
| VIII. Pequeno e reversível | ✅ | Incrementos: tipos e migração → quadro e papéis → supervisor → loop → rota → logs → web. Cada um vira um commit verde. |
| Stack | ✅ | Só LangGraph, zod e SQLite já adotados. |

**Re-check pós-design (Fase 1)**: os contratos mantêm todos os itens. Não há violações.

## Project Structure

### Documentation (this feature)

```text
specs/017-team-mode/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── http.md          # rota team, trace com handoff/role
│   └── web-ui.md        # evento "Passagem" e badge de papel
├── checklists/requirements.md
└── tasks.md             # /speckit.tasks
```

### Source Code (repository root)

```text
src/
├── team/                               # NOVO
│   ├── blackboard.ts (+ .test.ts)      # Blackboard, addFacts/setPlan/addOutcome/addHandoff, renderBlackboard, fallbackAnswer
│   ├── roles.ts (+ .test.ts)           # TeamRole, TEAM_ROLE_TOOLS, selectRoleTools, schemas Analyst/Plan, prompts
│   ├── supervisor.ts (+ .test.ts)      # SupervisorDecisionSchema, resolveSupervisorDecision, createModelSupervisor
│   ├── members.ts                      # runAnalyst / runPlanner / runExecutor reais (IO: modelo + agentes)
│   ├── team-graph.ts (+ .test.ts)      # StateGraph da equipe; tagRole; hasProposal; teto
│   └── index.ts                        # createTeamStrategy(tools, deps?) → ReasoningStrategy
├── agents/
│   ├── types.ts                        # RouteName/GraphNode += "team"; TraceEvent += handoff; role?
│   ├── approval-gate.ts                # + isApprovalGated (WeakSet das instâncias com porta)
│   └── index.ts                        # STRATEGY "team"; strategyForRoute("team")
├── graph/
│   ├── router.ts                       # ROUTE_TABLE/ALIASES += team
│   └── production-graph.ts             # nó "team"
├── domain/errors.ts                    # + TeamToolsNotGatedError
├── obs/logger.ts                       # + team.handoff (sem brief)
├── store/sqlite-request-store.ts       # DDL com 'team' + migrateRouteCheck
└── http/web-contract.test.ts           # + 200 da equipe com handoff

web/src/
├── lib/api-schemas.ts / trace-view.ts  # handoff + role (+ testes)
├── components/TraceEventItem.tsx, Icon.tsx
└── styles/tokens.css                   # --trace-handoff
```

**Structure Decision**: a equipe fica isolada em `src/team/`, como pedido, e entra no sistema só como mais uma estratégia (`agents/index.ts`) e mais uma rota (`graph/`). A borda HTTP não ganha código novo: a rota passa pelo `strategy` e pelo roteador, e o 202 vem da porta que já existe.

## Complexity Tracking

Sem violações.
