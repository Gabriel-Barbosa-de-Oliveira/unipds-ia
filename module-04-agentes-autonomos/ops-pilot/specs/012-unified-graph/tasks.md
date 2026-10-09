---

description: "Task list for 012-unified-graph"
---

# Tasks: Grafo Unificado com Roteador de Estratégia

> **Nota de implementação**: os nomes finais seguem o esboço do `/speckit-implement`. O grafo e o roteador ficam em `src/graph/` (testes em `src/graph/*.test.ts`), os nós são `contexto`, `roteador`, `react`, `planExecute`, `reflect` e `resposta`, e as rotas são `react | planExecute | reflect`, com `plan-and-execute`/`reflection` aceitos como aliases no `/chat`. O schema se chama `routeSchema`.

**Input**: Design documents from `/specs/012-unified-graph/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos. A constitution (Princípio V, NON-NEGOTIABLE) exige teste para toda lógica nova, e a FR-015 pede testes sem rede para roteamento, override, fallback e carimbo de `node`.

**Organization**: as tarefas estão agrupadas por user story (spec.md), para que cada uma possa ser implementada e testada de forma independente.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivo diferente e sem dependência de tarefa ainda não concluída).
- **[Story]**: a user story à qual a tarefa pertence (US1–US3).
- Cada descrição traz o caminho exato do arquivo.

## Path Conventions

Projeto único, com `src/` na raiz. Os testes ficam ao lado do código (`*.test.ts`), como nas features 001–011.

---

## Phase 1: Setup

**Purpose**: inicialização do projeto.

Fase vazia: nenhuma dependência ou configuração nova. `@langchain/langgraph`, `@langchain/openai` e zod já estão no `package.json`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: os tipos e utilitários puros que o roteador, o grafo e o controller compartilham.

**⚠️ CRITICAL**: nenhuma user story começa antes desta fase terminar.

- [X] T001 Estender `src/agents/types.ts` conforme [data-model.md](./data-model.md):
  - exportar `type RouteName = "react" | "plan-and-execute" | "reflection"`, `const ROUTE_NAMES = ["react", "plan-and-execute", "reflection"] as const`, `type RouteSource = "router" | "override" | "fallback"`, `interface RouteDecision { route: RouteName; reason: string; source: RouteSource }` e `type GraphNode = "context" | "router" | RouteName | "answer"`;
  - adicionar `node?: GraphNode` a **todas** as variantes de `TraceEvent` e a nova variante `{ type: "route"; at: number; route: RouteName; reason: string; source: RouteSource; node?: GraphNode }`;
  - exportar `type ProductionTraceEvent = TraceEvent & { node: GraphNode }`.

  Rodar `npm run typecheck`. O `switch` de `formatTraceEvent` em `src/agents/trace.ts` vai falhar por falta do caso `"route"`; por ora, adicionar um caso provisório `return \`[route] ${event.route}\`` (o formato final é a T013).
- [X] T002 [P] (depende de T001) Adicionar a `src/agents/trace.ts` a função pura `export function tagTrace(events: readonly TraceEvent[], node: GraphNode, offset: number): ProductionTraceEvent[]`. Ela devolve cópias com `node` carimbado e `at = offset + index`, sem mutar a entrada (research.md item 5).

  Em `src/agents/trace.test.ts`: testar o carimbo em todos os eventos, a reindexação a partir de `offset`, que `node` preexistente é sobrescrito e que a entrada original não muda.
- [X] T003 [P] (depende de T001) Criar `src/agents/router.ts` só com a parte pura de validação de nome: `export function parseRouteName(name: string): RouteName`, que devolve o nome se ele estiver em `ROUTE_NAMES` e senão lança `new UnknownStrategyError(name)` (de `src/domain/errors.ts`).

  Em `src/agents/router.test.ts` (novo): `"react"`, `"plan-and-execute"` e `"reflection"` são aceitos; `"nao-existe"`, `""` e `"React"` lançam `UnknownStrategyError` com `.strategy` igual ao valor recebido.
- [X] T004 [P] (depende de T001) Adicionar a `src/agents/index.ts` a função `export function strategyForRoute(route: RouteName, reflect?: boolean, extraTools?: StructuredToolInterface[]): ReasoningStrategy` (research.md item 7):
  - `route === "reflection"` → `resolveStrategy("react", true, extraTools)`;
  - qualquer outra rota → `resolveStrategy(route, reflect, extraTools)`.

  Em `src/agents/index.test.ts`: `strategyForRoute("reflection").name === "reflect:react"`; `strategyForRoute("reflection", true).name === "reflect:react"` (não aplica reflection duplo); `strategyForRoute("plan-and-execute", true).name === "reflect:plan-and-execute"`; `strategyForRoute("react").name === "react"`.
- [X] T005 [P] Generalizar `src/services/chat.service.ts` (research.md item 8):
  - adicionar `export function withTimeout<T>(run: () => Promise<T>, timeoutMs: number): Promise<T>`, com a mesma semântica atual (rejeita com `ChatTimeoutError` e sempre faz `clearTimeout`);
  - reescrever `runWithTimeout(strategy, input, options, timeoutMs)` como `withTimeout(() => strategy.run(input, options), timeoutMs)`.

  Em `src/services/chat.service.test.ts`: casos de `withTimeout` para resolve, reject propagado e timeout. Os casos existentes de `runWithTimeout` continuam verdes.

**Checkpoint**: `npm run typecheck` e `npm test` verdes. Os tipos e os utilitários puros estão prontos.

---

## Phase 3: User Story 1 - O copiloto escolhe sozinho a estratégia certa (Priority: P1) 🎯 MVP

**Goal**: sem `strategy`, o `/chat` passa por context → router → uma estratégia → answer, e o roteador decide a rota com `{ route, reason }`. Se o roteador falhar, a rota é `react` (fallback).

**Independent Test**: com um `decideRoute` fake que devolve `plan-and-execute` e estratégias fake, uma chamada ao `/chat` sem `strategy` executa só a fake de plan-and-execute e devolve 200 no formato atual.

### Tests for User Story 1

- [X] T006 [P] [US1] (depende de T003) Estender `src/agents/router.test.ts` (escrito antes da T008, deve falhar até ela) cobrindo [contracts/production-graph.md](./contracts/production-graph.md):
  - `buildRouterMessages("pergunta")` devolve `[["system", …], ["user", "pergunta"]]`;
  - o system contém um cabeçalho de tabela markdown (`| Rota |`) e uma linha para cada item de `ROUTE_NAMES` (FR-004);
  - `resolveRouteDecision`:
    - `{ decided: { route: "plan-and-execute", reason: "várias etapas" } }` → `source: "router"`;
    - `{ override: "reflection", decided: { route: "react", reason: "x" } }` → `{ route: "reflection", source: "override", reason: "Estratégia informada pelo cliente" }`;
    - `{ error: new Error("boom") }` → `{ route: "react", source: "fallback" }`, com `reason` começando com `"Fallback:"` e contendo `"boom"`;
    - `{ decided: null }`, `{ decided: { route: "outra", reason: "x" } }` e `{ decided: { route: "react", reason: "   " } }` → `source: "fallback"`, `route: "react"`.
- [X] T007 [P] [US1] (depende de T002, T004) Criar `src/agents/production-graph.test.ts` (deve falhar até a T010). Usar estratégias fake que registram `calls` e o `input` recebido, e cujo `run` devolve traces com `at` arbitrário (ex.: `Date.now()`), e um `decideRoute` fake que devolve uma rota fixa e `tokenUsage { promptTokens: 10, source: "real" }`. Cobrir:
  - **(a)** para cada uma das 3 rotas, só a fake daquela rota é chamada (FR-002);
  - **(b)** o `input` recebido pela estratégia é igual a `buildContext(context, budget).prompt` (FR-012);
  - **(c)** `result.answer` é o da estratégia, e `result.metrics.llmCalls` é a soma de 1 (roteador) com o valor da estratégia; `promptTokens` é a soma via `mergeTokenUsage`;
  - **(d)** com um `decideRoute` que lança, a resposta vem com `result.route.source === "fallback"` e foi executada pela fake de `react`;
  - **(e)** `strategyFor` é chamado uma única vez, com a rota escolhida.

### Implementation for User Story 1

- [X] T008 [US1] (depende de T003) Completar `src/agents/router.ts` (research.md itens 3–4):
  - `export const ROUTE_TABLE: readonly { route: RouteName; whenToUse: string; examples: string[] }[]`, com o conteúdo da tabela do research.md item 3;
  - `export const RouteDecisionSchema = z.object({ route: z.enum(ROUTE_NAMES), reason: z.string().trim().min(1) })`;
  - `export function buildRouterMessages(prompt: string): [string, string][]`, pura. O system é em pt-BR ("Você é o roteador do OpsPilot… escolha exatamente uma rota… explique o motivo em uma frase") e inclui a tabela markdown `| Rota | Quando usar | Exemplos |` gerada de `ROUTE_TABLE`;
  - `export function resolveRouteDecision(input: { override?: RouteName; decided?: unknown; error?: unknown }): RouteDecision`, pura, com a ordem override > `RouteDecisionSchema.safeParse(decided)` válido > fallback `{ route: "react", reason: \`Fallback: ${causa}\`, source: "fallback" }`;
  - `export type DecideRoute = (prompt: string) => Promise<{ decided: unknown; tokenUsage: TokenUsage }>`;
  - `export function createModelRouter(): DecideRoute`, que usa `createModel().withStructuredOutput(RouteDecisionSchema).invoke(buildRouterMessages(prompt), { callbacks: [usageCollector] })`, no mesmo padrão de `critique` em `src/agents/reflection.ts`. `createModel()` só é chamado dentro da função devolvida (lazy), nunca no import.
- [X] T009 [US1] (depende de T008) Criar em `src/agents/production-graph.ts` as interfaces `ProductionGraphDeps { decideRoute: DecideRoute; strategyFor: (route: RouteName) => ReasoningStrategy }`, `ProductionInput { context: ContextInput; budget: ContextBudget; override?: RouteName }` e `ProductionRunResult = RunResult & { trace: ProductionTraceEvent[]; route: RouteDecision; context: BuiltContext }`.

  Criar também o estado do grafo com `Annotation.Root` conforme [data-model.md](./data-model.md#productiongraphstate-estado-do-langgraph): `trace` com reducer de concat e os demais campos com reducer de substituição, no mesmo estilo de `PlanExecuteState` em `src/agents/plan-and-execute.ts`.
- [X] T010 [US1] (depende de T002, T009) Implementar `export function createProductionGraph(deps): { run(input): Promise<ProductionRunResult> }` em `src/agents/production-graph.ts` (research.md itens 1, 2, 6):
  - **nó `context`**: `built = buildContext(state.contextInput, state.budget)`;
  - **nó `router`**: com `state.override`, não chama `deps.decideRoute` e `routerUsage = { llmCalls: 0 }`. Sem override, faz `try { const r = await deps.decideRoute(state.built.prompt); decision = resolveRouteDecision({ decided: r.decided }); routerUsage = { llmCalls: 1, tokenUsage: r.tokenUsage } } catch (error) { decision = resolveRouteDecision({ error }); routerUsage = { llmCalls: 1 } }`. Emite `trace: [{ type: "route", at: 0, node: "router", ...decision }]`;
  - **nós `react`, `plan-and-execute` e `reflection`**: uma fábrica `strategyNode(route)` que faz `const r = await deps.strategyFor(route).run(state.built.prompt)` e devolve `{ strategyResult: r, trace: tagTrace(r.trace, route, state.trace.length) }`;
  - **nó `answer`**: monta `result` com `answer = strategyResult.answer`, `trace = state.trace` e `route = state.decision`. As métricas são `llmCalls = router + estratégia`, `promptTokens/tokenSource = mergeTokenUsage(...)` (só com a parte da estratégia se o roteador não tiver `tokenUsage`) e `latencyMs` vindo de `startTimer()`;
  - **arestas**: `START → context → router`, `addConditionalEdges("router", (s) => s.decision.route, ["react", "plan-and-execute", "reflection"])`, cada estratégia → `answer` → `END`.

  O `run(input)` dá `invoke` no grafo e devolve `state.result`. Exceções da estratégia propagam sem tratamento. Rodar a T007 até ficar verde.
- [X] T011 [US1] (depende de T004, T005, T010) Alterar `src/http/server.ts` para executar o grafo:
  - `CreateAppOptions` ganha `decideRoute?: DecideRoute`, com JSDoc "Sobrescreve o roteador — usado por testes; padrão `createModelRouter()`";
  - no handler, depois de carregar o histórico e as memórias como hoje, fazer `const graph = createProductionGraph({ decideRoute, strategyFor: (route) => route === "reflection" ? resolveStrategy("react", true, extraTools) : resolveStrategy(route, parsed.data.reflect, extraTools) })`. Isso replica `strategyForRoute`, mas por meio do `resolveStrategy` injetável, o que mantém os fakes dos testes;
  - `const result = await withTimeout(() => graph.run({ context: { message, window: history, memories: recalled }, budget: contextBudget, override }), timeoutMs)`;
  - as métricas de contexto (`historyMessages`, `contextBreakdown`, `contextTrimmed`) passam a vir de `result.context`.

  Nesta tarefa, `override` é sempre `undefined` (a US3 liga o override). Remover o `buildContext` direto do handler.
- [X] T012 [US1] (depende de T011) Atualizar `src/http/server.test.ts`:
  - criar o helper `fixedRouter(route: RouteName = "react"): DecideRoute`, que devolve `{ decided: { route, reason: "fake" }, tokenUsage: { promptTokens: 0, source: "real" } }`, e passar `decideRoute: fixedRouter()` em **todo** `createApp(...)` existente, para que nenhum teste chame o OpenRouter;
  - ajustar os asserts afetados: `body.trace` passa a ter o evento `route` em `[0]` e `node` nos demais (ex.: o `deepEqual` da "User Story 1 — estratégia padrão" vira `[{ type: "route", at: 0, node: "router", route: "react", reason: "fake", source: "router" }, { type: "answer", at: 1, node: "react", content: "há 3 alertas firing" }]`), e `metrics.llmCalls` soma +1 do roteador;
  - nova describe "User Story 1 (012) — roteamento automático": com `fixedRouter("plan-and-execute")` e `resolveStrategy` que devolve fakes por nome, sem `strategy`, roda só a fake de plan-and-execute; com um `decideRoute` que lança, responde 200 e executa a fake de react.

**Checkpoint**: sem `strategy`, o `/chat` é roteado automaticamente, e testes e typecheck estão verdes. Esse é o MVP.

---

## Phase 4: User Story 2 - Entender a rota e a etapa de cada passo (Priority: P1)

**Goal**: o trace tem exatamente um `route` (rota, motivo, origem) antes dos eventos de estratégia, e todo evento tem `node`. Isso aparece legível no terminal e no topo da resposta HTTP.

**Independent Test**: rodar o grafo com fakes e checar as invariantes de trace do [data-model.md](./data-model.md#invariantes-do-trace-do-grafo). `formatTrace` mostra a rota e o nó.

### Tests for User Story 2

- [X] T013 [P] [US2] (depende de T001) Estender `src/agents/trace.test.ts`:
  - `formatTraceEvent({ type: "route", at: 0, route: "plan-and-execute", reason: "várias etapas", source: "router" })` → `"[route] plan-and-execute (router): várias etapas"`;
  - um evento com `node: "react"` é prefixado com `"react │ "` (ex.: `"react │ [answer] ok"`);
  - os eventos sem `node` (o `FIXTURE` existente) continuam com a saída idêntica à de hoje (regressão de arena/bench).
- [X] T014 [P] [US2] (depende de T010) Estender `src/agents/production-graph.test.ts` com as invariantes, para cada rota e para o fallback:
  - exatamente 1 evento `type === "route"`, em `trace[0]` e com `node === "router"`;
  - todo evento tem `node` definido (FR-006/SC-002);
  - todo evento depois do índice 0 tem `node === result.route.route`;
  - `trace.map((e) => e.at)` é `[0, 1, …, n-1]`;
  - `reason` não é vazio.

### Implementation for User Story 2

- [X] T015 [US2] (depende de T013) Em `src/agents/trace.ts`, `formatTraceEvent`:
  - caso `"route"` → `` `[route] ${event.route} (${event.source}): ${event.reason}` ``, substituindo o provisório da T001;
  - quando `event.node` estiver definido, prefixar a linha com `` `${event.node} │ ` ``.

  Manter a função pura. Rodar a T013 até ficar verde.
- [X] T016 [US2] (depende de T011) Em `src/http/server.ts`, a resposta 200 ganha o campo aditivo `route: result.route` ([contracts/post-chat.md](./contracts/post-chat.md)). Atualizar o tipo `ChatResponseBody` usado em `src/http/server.test.ts`.

  Em `src/http/server.test.ts`: `body.route` deep-equal ao evento `route` do trace, sem os campos `type`, `at` e `node`, e todo `body.trace[i].node` definido.

**Checkpoint**: o trace é auditável (rota, motivo, origem e nó). US1 e US2 funcionam juntas.

---

## Phase 5: User Story 3 - Forçar uma estratégia específica (Priority: P2)

**Goal**: um `strategy` válido no `/chat` vira override. O roteador não é chamado, o `route.source` é `"override"`, `"reflection"` é aceito e uma estratégia desconhecida continua dando 422.

**Independent Test**: com `strategy: "react"` e um `decideRoute` fake que contaria chamadas e escolheria `plan-and-execute`, só a fake de react roda, o roteador tem 0 chamadas e `route.source === "override"`.

### Tests for User Story 3

- [X] T017 [P] [US3] (depende de T010) Estender `src/agents/production-graph.test.ts`:
  - com `override: "plan-and-execute"`, o `decideRoute` fake tem `calls === 0`, só a fake de plan-and-execute roda, `result.route` é `{ route: "plan-and-execute", source: "override", reason: "Estratégia informada pelo cliente" }` e `metrics.llmCalls` é igual ao da estratégia (sem +1, SC-003/SC-006);
  - o mesmo vale para `override: "reflection"`.
- [X] T018 [P] [US3] (depende de T012) Estender `src/http/server.test.ts`, na describe "User Story 2 — estratégia explícita e estratégia desconhecida" e numa describe nova "User Story 3 (012) — override":
  - `strategy: "plan-and-execute"` com `decideRoute` contador → roteador com 0 chamadas e `body.route.source === "override"`;
  - `strategy: "reflection"` → 200, com `resolveStrategy` chamado com `("react", true, …)`;
  - `strategy: "nao-existe"` → 422 `unknown_strategy`, com nenhuma estratégia e nenhum roteador chamado;
  - `strategy: "react", reflect: true` → `resolveStrategy("react", true, …)`;
  - ajustar o fake existente `resolveStrategy: (name) => …`, que hoje trata `name === undefined`: agora o nome sempre chega definido.

### Implementation for User Story 3

- [X] T019 [US3] (depende de T003, T011) Em `src/http/server.ts`, logo depois do `safeParse` e **antes** de qualquer IO de conversa ou memória, fazer `const override = parsed.data.strategy !== undefined ? parseRouteName(parsed.data.strategy) : undefined`. O `UnknownStrategyError` lançado segue para `next(error)` → 422 pelo middleware existente (FR-010).

  Passar `override` para `graph.run(...)`. O `ChatRequestSchema` mantém `strategy: z.string().optional()`, porque a validação semântica fica em `parseRouteName` para preservar o corpo do 422 atual.

**Checkpoint**: as três user stories funcionam de forma independente e em conjunto.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T020 [P] Atualizar o JSDoc de `resolveStrategy` / `DEFAULT_STRATEGY_NAME` em `src/agents/index.ts`. `react` deixa de ser o "padrão do endpoint" e passa a ser a rota de fallback do roteador; o `/chat` agora resolve via `strategyForRoute`/grafo.
- [X] T021 [P] Acrescentar uma nota em `specs/003-chat-endpoint/quickstart.md` (o projeto não tem README; esse é o guia do `/chat`) dizendo que, a partir da 012, `strategy` é um override opcional, que `reflection` é aceito, que a resposta traz `route` e que o trace tem `node`. Linkar `specs/012-unified-graph/contracts/post-chat.md`.
- [X] T022 Rodar `npm run typecheck` e `npm test`; os dois precisam ficar verdes (Princípio V).
- [ ] T023 Validar manualmente os cenários 2–3 de [quickstart.md](./quickstart.md). Isso exige `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` já presentes no ambiente, sem nunca ler `.env`. Registrar, numa amostra de pelo menos 5 perguntas com rota esperada, a taxa de acerto do roteador (SC-004 ≥ 80%) e ajustar `ROUTE_TABLE` se ficar abaixo. **Pendente**: `OPENROUTER_API_KEY` não está configurada no ambiente desta execução.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: vazia.
- **Foundational (Phase 2)**: a T001 vem primeiro. T002, T003 e T004 dependem dela e podem rodar em paralelo entre si. A T005 é independente. Esta fase bloqueia todas as stories.
- **US1 (Phase 3)**: T006 (testes do roteador) em paralelo com T007 (testes do grafo); T008 → T009 → T010 → T011 → T012.
- **US2 (Phase 4)**: a T013 depende só da T001 e pode começar cedo; a T014 depende da T010; T015 depende da T013; T016 depende da T011.
- **US3 (Phase 5)**: a T017 depende da T010; a T018 depende da T012; a T019 depende das T003 e T011.
- **Polish (Phase 6)**: depois de todas as stories.

### User Story Dependencies

- **US1 (P1)**: independente depois da Phase 2. Entrega o MVP: roteamento automático com fallback.
- **US2 (P1)**: o grafo da US1 já emite `route`/`node`. A US2 acrescenta as garantias por teste, a formatação legível e o campo `route` na resposta HTTP. As partes de `trace.ts` (T013/T015) dá para fazer em paralelo com a US1.
- **US3 (P2)**: o nó `router` já trata `override` desde a T010. A US3 liga o override no controller (T019) e cobre com testes.

### Within Each User Story

- Os testes são escritos primeiro e falham antes da implementação.
- O que é puro (`router.ts`, `trace.ts`) vem antes do grafo, e o grafo vem antes de `server.ts`.
- Cada tarefa cabe em um commit (Princípio VIII).

## Parallel Opportunities

- **Phase 2**: T002 ∥ T003 ∥ T004 ∥ T005 (arquivos diferentes, depois da T001).
- **US1**: T006 ∥ T007.
- **US2**: T013 ∥ T014. A T013/T015 pode correr em paralelo com toda a US1.
- **US3**: T017 ∥ T018.
- **Polish**: T020 ∥ T021.

### Parallel Example: Phase 2 + User Story 1

```bash
# Depois de T001:
Task: "T002 tagTrace em src/agents/trace.ts"
Task: "T003 parseRouteName em src/agents/router.ts"
Task: "T004 strategyForRoute em src/agents/index.ts"
Task: "T005 withTimeout em src/services/chat.service.ts"

# Início da US1:
Task: "T006 [US1] testes do roteador em src/agents/router.test.ts"
Task: "T007 [US1] testes do grafo em src/agents/production-graph.test.ts"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 2 (T001–T005).
2. Phase 3 (T006–T012): grafo de produção com roteador e fallback, ligado ao `/chat`.
3. **STOP and VALIDATE**: sem `strategy`, só a estratégia escolhida roda, a falha do roteador continua dando 200 e a suíte está verde.

### Incremental Delivery

1. Foundational → US1 (roteamento automático) → US2 (trace auditável + `route` na resposta) → US3 (override) → Polish.
2. Cada story é aditiva. O formato da resposta só ganha campos (SC-007).
