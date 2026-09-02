---
description: "Task list template for feature implementation"
---

# Tasks: Núcleo de Raciocínio (Reasoning Strategies Core)

**Input**: Design documents from `/specs/001-reasoning-strategies-core/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: A especificação pede explicitamente testes determinísticos e sem rede para o store e a formatação de trace (FR-014) — esses testes estão incluídos abaixo. Nenhum teste automatizado é gerado para as estratégias de raciocínio ou para a arena, pois ambas exigem um modelo real via OpenRouter; sua verificação é feita por validação manual guiada por [quickstart.md](./quickstart.md).

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivos diferentes, sem dependência de tarefa incompleta)
- **[Story]**: A qual user story a tarefa pertence (US1, US2, US3)
- Caminhos de arquivo exatos em cada descrição

## Path Conventions

Projeto único (Node/TypeScript ESM). Testes ficam colocados ao lado do módulo testado como `*.test.ts` (convenção já usada em `package.json`: `"test": "node --import tsx --test \"src/**/*.test.ts\""`) — não há diretório `tests/` separado.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Preparar scripts e configuração de ambiente antes de qualquer código de domínio/agente

- [X] T001 [P] Adicionar o script `"seed": "tsx src/scripts/seed.ts"` em `package.json`
- [ ] T002 [P] Documentar as variáveis de ambiente exigidas (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, e as variáveis de conexão MySQL: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`) em `.env.example`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Infraestrutura compartilhada que TODA user story depende — tipos comuns, domínio puro, os dois adaptadores de store, fábrica de modelo, tools e formatação de trace

**⚠️ CRITICAL**: Nenhuma user story pode começar antes desta fase estar completa

- [X] T003 [P] Definir `ReasoningStrategy`, `TraceEvent` (união `thought`/`action`/`observation`/`plan`/`critique`/`answer`), `Metrics` e `RunResult` em `src/agents/types.ts` (per [contracts/reasoning-strategy.md](./contracts/reasoning-strategy.md))
- [X] T004 [P] Definir as classes de erro de domínio `ServiceNotFoundError`, `IncidentNotFoundError`, `InvalidSeverityError` em `src/domain/errors.ts`
- [X] T005 Implementar os tipos `Service`, `Alert`, `Incident` e as funções puras `listAlerts`, `openIncident`, `resolveIncident` em `src/domain/ops-store.ts` (mesmo estado de entrada → mesmo estado de saída; usa as classes de `src/domain/errors.ts` de T004; regras de validação/transição conforme [data-model.md](./data-model.md)) — depende de T004
- [X] T006 [P] Escrever testes determinísticos (sem rede) do caminho feliz do store em `src/domain/ops-store.test.ts`: listar todos os alertas, listar por `status`, abrir incidente com serviço/severidade válidos, resolver incidente existente — depende de T005
- [X] T007 [P] Implementar o builder puro do dataset canônico (5 serviços; 6 alertas: 3 `firing`, 3 `resolved`) em `src/domain/seed-data.ts` — depende de T005 (usa os tipos `Service`/`Alert`)
- [X] T008 [P] Definir a interface `OpsStoreRepository` (`listAlerts`, `openIncident`, `resolveIncident`) em `src/services/ops-store.repository.ts` — depende de T005
- [X] T009 Implementar o adaptador in-memory `src/services/ops-store.memory.ts` (implementa `OpsStoreRepository`, mantém estado em processo inicializado a partir de `seed-data.ts`, delega a mutação às funções puras de `ops-store.ts`) — depende de T005, T007, T008
- [X] T010 [P] Implementar a fábrica de conexão Sequelize/MySQL em `src/models/sequelize/connection.ts` (lê as variáveis de ambiente documentadas em T002)
- [X] T011 Definir os modelos Sequelize `Service`, `Alert`, `Incident` em `src/models/sequelize/service.model.ts`, `src/models/sequelize/alert.model.ts`, `src/models/sequelize/incident.model.ts` — depende de T010
- [X] T012 Implementar o adaptador `src/services/ops-store.sequelize.ts` (implementa `OpsStoreRepository` sobre os modelos Sequelize; traduz ausência de linha para `ServiceNotFoundError`/`IncidentNotFoundError`) — depende de T008, T011
- [X] T013 Implementar o script de seed idempotente `src/scripts/seed.ts` (grava o dataset canônico de `seed-data.ts` via `ops-store.sequelize.ts`; reexecução produz o mesmo estado) — depende de T012, T007, T001
- [X] T014 [P] Implementar a fábrica única de modelo `createReasoningModel()` em `src/agents/model.ts` (`ChatOpenAI` com `baseURL` do OpenRouter, `apiKey: process.env.OPENROUTER_API_KEY`, `model: process.env.OPENROUTER_MODEL`, `temperature: 0`)
- [X] T015 [P] Implementar o helper compartilhado de métricas em `src/agents/metrics.ts` (contagem de `llmCalls` e medição de `latencyMs` no ponto de entrada/saída comum de `run()`, usado igualmente por ambas as estratégias) — depende de T003
- [X] T016 Implementar as tools LangChain com schemas zod (`list_alerts`, `open_incident`, `resolve_incident`) em `src/agents/tools.ts` (valida args na fronteira antes de delegar ao `OpsStoreRepository` injetado, padrão = adaptador in-memory de T009; mapeia erros de domínio para observações estruturadas per [contracts/tools.md](./contracts/tools.md)) — depende de T003, T004, T008, T009
- [X] T017 [P] Implementar os helpers puros de formatação/serialização de trace em `src/agents/trace.ts` (usados pela arena e pelos testes) — depende de T003
- [X] T018 [P] Escrever testes determinísticos (sem rede) da formatação de trace em `src/agents/trace.test.ts` (fixture de `TraceEvent[]` → saída formatada esperada) — depende de T017

**Checkpoint**: Fundação pronta — tipos, domínio puro, os dois adaptadores de store, seed, fábrica de modelo, tools e formatação de trace disponíveis para qualquer user story.

---

## Phase 3: User Story 1 - Obter uma resposta auditável para uma pergunta operacional (Priority: P1) 🎯 MVP

**Goal**: Uma pergunta operacional em linguagem natural é respondida por uma estratégia de raciocínio, retornando resposta final + trace completo + métricas.

**Independent Test**: Rodar `npm run arena -- --input "quais alertas estão firing?" --strategies react` contra o dataset semeado e verificar que a resposta lista exatamente os 3 alertas `firing`, com trace completo (raciocínio → ação → observação → resposta) e métricas (`llmCalls`, `latencyMs`) reportadas.

### Implementation for User Story 1

- [X] T019 [US1] Implementar a estratégia ReAct em `src/agents/react.ts`: `createReactAgent` (`@langchain/langgraph/prebuilt`) com as tools de `src/agents/tools.ts` e o modelo de `src/agents/model.ts`; capturar o stream de eventos e mapear para `TraceEvent[]` (mensagens → `thought`/`answer`, tool calls → `action`, resultados de tool → `observation`); usar `src/agents/metrics.ts` para `llmCalls`/`latencyMs`; expor como `ReasoningStrategy` (`name: "react"`) — depende de T003, T014, T015, T016, T017
- [X] T020 [US1] Implementar a arena `src/arena.ts` (v1): validar `--input`/`--strategies`/`--max-iterations` com zod (per [contracts/arena-cli.md](./contracts/arena-cli.md)), registrar a estratégia `react`, executar cada estratégia solicitada via `run()`, imprimir por estratégia um bloco identificado com o trace formatado (`src/agents/trace.ts`) e as métricas — depende de T019, T017
- [X] T021 [US1] Validar manualmente via [quickstart.md](./quickstart.md) passo 3: `npm run arena -- --input "quais alertas estão firing?" --strategies react` — confirmar resposta com os 3 alertas corretos e trace/métricas completos — depende de T020

**Checkpoint**: User Story 1 completa e testável de forma independente — uma estratégia de raciocínio responde a uma pergunta operacional com trace e métricas via `npm run arena`.

---

## Phase 4: User Story 2 - Comparar estratégias de raciocínio lado a lado (Priority: P2)

**Goal**: A mesma pergunta operacional roda em duas ou mais estratégias de uma vez, com trace e métricas de cada uma exibidos lado a lado.

**Independent Test**: Rodar `npm run arena -- --input "..." --strategies react,plan-and-execute --max-iterations 8` e verificar blocos de trace/métricas identificáveis por estratégia mais um resumo comparativo final, sem repetir a pergunta manualmente.

### Implementation for User Story 2

- [X] T022 [US2] Implementar a estratégia Plan-and-Execute em `src/agents/plan-and-execute.ts`: `StateGraph` com nós `planner` (saída estruturada: lista ordenada de passos), `executor` (executa um passo por vez usando as tools de `src/agents/tools.ts`) e `replanner` (revisa os passos restantes após cada execução, decide finalizar); contador de passos no estado do grafo com corte físico em 8 (FR-005); usar `src/agents/model.ts` e `src/agents/metrics.ts`; expor como `ReasoningStrategy` (`name: "plan-and-execute"`) — depende de T003, T014, T015, T016, T017
- [X] T023 [US2] Registrar a estratégia `plan-and-execute` na tabela de estratégias disponíveis de `src/arena.ts` — depende de T020, T022
- [X] T024 [US2] Adicionar a `src/arena.ts` o resumo comparativo final (nome × `llmCalls` × `latencyMs`) impresso quando duas ou mais estratégias são executadas — depende de T020
- [X] T025 [US2] Validar manualmente via [quickstart.md](./quickstart.md) passo 4: `npm run arena -- --input "abra um incidente de severidade alta para o serviço checkout-api" --strategies react,plan-and-execute --max-iterations 8` — confirmar um bloco por estratégia, um evento `plan` antes de qualquer `action` no bloco `plan-and-execute`, e o resumo comparativo final — depende de T023, T024

**Checkpoint**: User Stories 1 e 2 funcionam independentemente — a arena compara as duas estratégias na mesma pergunta.

---

## Phase 5: User Story 3 - Ações operacionais controladas e à prova de falha (Priority: P3)

**Goal**: Entradas inválidas para as ações operacionais produzem erros estruturados e auditáveis, e toda estratégia para de forma controlada ao atingir seu limite de passos, sem travar nem silenciar falhas.

**Independent Test**: (a) Testes automatizados chamando `openIncident`/`resolveIncident` diretamente com entradas inválidas, sem envolver nenhuma estratégia, verificando erros estruturados; (b) `npm run arena -- --input "resolva o incidente inc-nao-existe" --strategies react` mostrando uma observação de erro estruturado no trace, sem exceção não tratada.

### Implementation for User Story 3

- [X] T026 [P] [US3] Adicionar testes de caminho negativo, determinísticos e sem rede, em `src/domain/ops-store.test.ts`: `openIncident` com `service` inexistente → `ServiceNotFoundError`; `openIncident` com `severity` fora do enum → erro de validação; `resolveIncident` com `id` inexistente → `IncidentNotFoundError`; `resolveIncident` sobre incidente já `resolved` → idempotente, sem erro — depende de T005, T006
- [X] T027 [US3] Em `src/agents/react.ts`, aplicar o `maxIterations` recebido em `run(input, options)` como limite de passos/recursão do agente; ao atingir o limite sem resposta final, retornar o trace parcial mais um evento `answer` indicando explicitamente que o limite foi atingido (FR-006) — depende de T019
- [X] T028 [US3] Em `src/agents/plan-and-execute.ts`, respeitar um `options.maxIterations` recebido em `run()` como limite adicional (nunca acima do corte físico de 8 de T022); ao atingir o limite sem resposta final, retornar o trace parcial mais um evento `answer` indicando explicitamente que o limite foi atingido (FR-006) — depende de T022
- [X] T029 [US3] Validar manualmente via [quickstart.md](./quickstart.md) passo 5: `npm run arena -- --input "resolva o incidente inc-nao-existe" --strategies react` — confirmar par `action`/`observation` com `IncidentNotFoundError` estruturado e nenhuma exceção não tratada — depende de T027

**Checkpoint**: As três user stories funcionam de forma independente — guardrails comprovados tanto por testes automatizados do store quanto por uma execução real com entrada inválida e por limite de passos respeitado nas duas estratégias.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Fechar os gates de qualidade do projeto e a documentação após todas as user stories

- [X] T030 [P] Rodar `npm run typecheck` e `npm test`; corrigir quaisquer falhas de tipo ou teste nos módulos novos
- [X] T031 [P] Adicionar `npm run seed` à seção "Comandos" de `CLAUDE.md`
- [ ] T032 Rodar a validação completa de [quickstart.md](./quickstart.md) (passos 1–5) como passe final de aceitação

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: sem dependências — pode começar imediatamente
- **Foundational (Phase 2)**: depende de Setup — BLOQUEIA todas as user stories
- **User Stories (Phase 3+)**: todas dependem da conclusão de Foundational
  - US1 (P1) não depende de US2/US3
  - US2 (P2) reaproveita `src/arena.ts` criado em US1 (T020) — depende de US1 estar implementada, mas é testável e entregável de forma independente assim que T023–T025 estiverem prontas
  - US3 (P3) reaproveita `src/agents/react.ts` (T019) e `src/agents/plan-and-execute.ts` (T022) — depende de US1/US2 existirem como arquivos, mas seus critérios de aceitação (erros estruturados, corte por limite) são verificáveis de forma isolada
- **Polish (Phase 6)**: depende de todas as user stories desejadas estarem completas

### Within Each User Story

- Estratégia de raciocínio antes da integração na arena
- Integração na arena antes da validação manual via quickstart

### Parallel Opportunities

- Setup: T001 e T002 em paralelo
- Foundational: T003, T004 em paralelo no início; depois T006, T007, T008 em paralelo (todos dependem só de T005); T010, T014, T015, T017 em paralelo entre si; T018 em paralelo após T017
- User Story 3: T026 pode rodar em paralelo com T027/T028 (arquivo diferente)
- Polish: T030 e T031 em paralelo

---

## Parallel Example: Foundational

```bash
# Após T005 (domínio) estar pronto, disparar em paralelo:
Task: "Testes de caminho feliz do store em src/domain/ops-store.test.ts"
Task: "Builder do dataset canônico em src/domain/seed-data.ts"
Task: "Interface OpsStoreRepository em src/services/ops-store.repository.ts"

# Em paralelo, independentes de tudo acima:
Task: "Fábrica de conexão Sequelize/MySQL em src/models/sequelize/connection.ts"
Task: "Fábrica única de modelo em src/agents/model.ts"
Task: "Helper de métricas em src/agents/metrics.ts"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Completar Phase 1: Setup
2. Completar Phase 2: Foundational (CRITICAL — bloqueia todas as stories)
3. Completar Phase 3: User Story 1
4. **PARAR e VALIDAR**: rodar o teste independente de US1 (quickstart passo 3)
5. Nesse ponto já existe um copiloto funcional respondendo perguntas operacionais com trace e métricas via `npm run arena --strategies react`

### Incremental Delivery

1. Setup + Foundational → fundação pronta
2. US1 → validar independentemente → MVP demonstrável
3. US2 → validar independentemente → comparação de estratégias demonstrável
4. US3 → validar independentemente → guardrails comprovados (testes + execução real)
5. Polish → gates de qualidade (`typecheck`, `test`) e quickstart completo

---

## Notes

- [P] = arquivos diferentes, sem dependência de tarefa incompleta
- [Story] mapeia a tarefa à user story correspondente para rastreabilidade
- Nenhum teste automatizado cobre `react.ts`/`plan-and-execute.ts`/`arena.ts` diretamente (exigem OpenRouter real) — sua verificação é a validação manual via quickstart, conforme decidido em [research.md](./research.md) item 7
- Commitar após cada tarefa ou grupo lógico de tarefas (Constitution Principle VIII — Pequeno e Reversível)
- `npm run typecheck` e `npm test` DEVEM ficar verdes antes de qualquer commit (Constitution Principle V)
- **Desvio temporário de T013**: `npm run seed` grava o dataset canônico em `data/ops-store.json` (via `src/services/ops-store.memory.ts`) em vez de usar o adaptador Sequelize/MySQL, para permitir validação manual sem um banco real. O adaptador `ops-store.sequelize.ts` permanece implementado e intacto para quando um MySQL real estiver disponível.
