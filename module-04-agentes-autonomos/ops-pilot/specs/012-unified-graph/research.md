# Research: Grafo Unificado com Roteador de Estratégia

> **Nota de implementação (nomes finais)**: seguindo o esboço passado no `/speckit-implement`, o código usa `src/graph/production-graph.ts` e `src/graph/router.ts` (não `src/agents/`); nós `contexto`, `roteador`, `react`, `planExecute`, `reflect`, `resposta`; rotas `react | planExecute | reflect`. O `strategy` do `/chat` aceita também os nomes legados `plan-and-execute` e `reflection` como aliases. Onde este documento usa os nomes antigos (`context`/`router`/`answer` como nós, `plan-and-execute`/`reflection` como rotas), leia com esse mapeamento.

Todas as decisões abaixo partem do código atual: `src/agents/{react,plan-and-execute,reflection,index,types,trace}.ts`, `src/services/chat.service.ts` e `src/http/server.ts`. Não ficou nenhum NEEDS CLARIFICATION em aberto.

## 1. Forma do grafo de produção

- **Decision**: `src/agents/production-graph.ts` monta um `StateGraph` do LangGraph com seis nós: `context` → `router` → (`react` | `plan-and-execute` | `reflection`) → `answer` → END. A escolha do nó de estratégia é feita por `addConditionalEdges("router", (s) => s.decision.route, [...])`. Cada nó de estratégia chama o `ReasoningStrategy.run(...)` já existente.
- **Rationale**: as estratégias já são `ReasoningStrategy` testadas (o plan-and-execute é inclusive um subgrafo próprio). Embrulhá-las como nós mantém o comportamento delas idêntico (FR-012/SC-007) e torna o roteamento uma aresta condicional explícita, que dá para inspecionar e testar.
- **Alternatives considered**:
  - Achatar react e plan-and-execute em nós de baixo nível (agent/tools/planner/executor) dentro de um grafo único: o trace fica mais granular, mas obriga a reescrever duas estratégias já estáveis, e o reflection vira um ciclo difícil de compor. Rejeitado (Princípio VIII, pequeno e reversível).
  - Um `switch` imperativo em `chat.service.ts` sem LangGraph: é mais simples, mas contraria o pedido explícito de "nós" e perde a topologia declarativa.

## 2. Nó `context`: onde fica o IO

- **Decision**: o nó `context` só chama a função pura `buildContext(...)` sobre um `ContextInput` já carregado (mensagem, janela, memórias) e o `ContextBudget`. Quem continua carregando histórico e memórias é o controller (`server.ts`), que passa esses dados ao grafo.
- **Rationale**: Princípios I e IV. O IO de SQLite e embeddings continua nas camadas de service/model, o nó fica determinístico e o resultado do contexto é byte a byte o mesmo de hoje (FR-012).
- **Alternatives considered**: injetar `conversationStore`/`memoryStore` no grafo. Isso espalha IO pelo agente e complica os fakes de teste. Rejeitado.

## 3. Roteador com saída estruturada

- **Decision**: em `src/agents/router.ts`, `createModel().withStructuredOutput(RouteDecisionSchema)`, com `RouteDecisionSchema = z.object({ route: z.enum(["react","plan-and-execute","reflection"]), reason: z.string().min(1) })`. O system prompt traz uma **tabela markdown** de rotas (rota | quando usar | exemplos), gerada por uma função pura `buildRouterMessages(prompt)` a partir de uma constante `ROUTE_TABLE`. O roteador recebe o prompt já montado pelo nó `context`, para que perguntas de continuação ("e agora resolve esse") tenham histórico.
- **Rationale**: esse é o mesmo padrão já usado no crítico do reflection e no planner (`withStructuredOutput` + zod). A tabela como dado (`ROUTE_TABLE`) deixa o prompt testável (FR-004) e permite reaproveitar a mesma fonte no README/arena.
- **Tabela de rotas (conteúdo inicial)**:

  | Rota | Quando usar |
  |---|---|
  | `react` | Consulta direta ou ação única: listar alertas, ver um incidente, abrir/resolver um incidente específico. |
  | `plan-and-execute` | Pedido com várias etapas dependentes ou em lote: "triar todos os alertas críticos e abrir incidentes", "resolver tudo que estiver aberto do serviço X". |
  | `reflection` | A precisão é crítica e o erro custa caro: resumo para comunicação/pós-mortem, decisão que depende de cruzar várias observações, pedido explícito de revisão. |

- **Alternatives considered**: classificação por palavras-chave/regex. É barata, mas frágil para linguagem natural e não gera `reason`. Rejeitada como estratégia principal (a validação pura de fallback continua existindo).

## 4. Override, fallback e validação (lógica pura)

- **Decision**: a função pura `resolveRouteDecision({ override, decided, error })` devolve um `RouteDecision` com `source`:
  - `override` válido → `{ route: override, reason: "Estratégia informada pelo cliente", source: "override" }`. O roteador **não** é chamado (o nó `router` checa o override antes de invocar o modelo).
  - decisão do modelo válida pelo schema → `source: "router"`.
  - erro, `null` ou decisão inválida → `{ route: "react", reason: "Fallback: <motivo>", source: "fallback" }`.
  
  A validação do override acontece **antes** do grafo, no controller, via `parseRouteName`, que lança `UnknownStrategyError` (o 422 atual continua igual, FR-010). Override passa a aceitar também `"reflection"`.
- **Rationale**: concentra todas as regras de FR-008/FR-009/FR-011 numa função pura e trivialmente testável (Princípio IV/V), sem rede.
- **Alternatives considered**: deixar o override passar pelo roteador como "dica". Isso gasta uma chamada extra e contraria o SC-003. Rejeitado.

## 5. Campo `node` e evento `route` no trace

- **Decision**:
  - `TraceEvent` ganha a variante `{ type: "route"; at; route: RouteName; reason: string; source: RouteSource }`.
  - Todas as variantes ganham `node?: GraphNode`, onde `GraphNode = "context" | "router" | "react" | "plan-and-execute" | "reflection" | "answer"`.
  - O campo é **opcional no tipo**, porque as estratégias rodadas diretamente por arena/bench não passam pelo grafo. **No grafo é garantido**: cada nó de estratégia carimba `node` em todos os eventos que recebe, via a função pura `tagTrace(events, node, offset)`, que também reindexa `at` de forma sequencial.
  - O tipo de saída do grafo é `ProductionTraceEvent = TraceEvent & { node: GraphNode }`, para que o compilador garanta o campo (FR-006).
- **Rationale**: muda o mínimo nas estratégias (nenhuma delas precisa conhecer o grafo) e garante FR-006/SC-002 por tipo e por teste. Reindexar `at` também corrige a inconsistência atual (o planner usa `Date.now()` e os demais usam índice).
- **Alternatives considered**: tornar `node` obrigatório em `TraceEvent`. Isso força arena, bench e testes existentes a inventar um nó. Rejeitado.

## 6. Nó `answer` e métricas

- **Decision**: o nó `answer` não chama o modelo e não emite evento novo (a resposta final já vem como evento `answer` da estratégia, carimbado com o nó da estratégia). Ele consolida o `RunResult`: `answer`, trace completo (`route` primeiro, FR-007) e métricas somadas (`llmCalls` do roteador + estratégia, `promptTokens` via `mergeTokenUsage`, `latencyMs` do grafo inteiro). O objeto `route` (a `RouteDecision`) também sai no topo da resposta HTTP como campo **aditivo**.
- **Rationale**: atende FR-013/SC-006 (no máximo +1 chamada) sem duplicar eventos `answer`.
- **Alternatives considered**: o nó `answer` reemitir um evento `answer` próprio. Isso duplica o conteúdo e confunde `lastAnswer`. Rejeitado.

## 7. `reflect` e a rota `reflection`

- **Decision**: a rota `reflection` executa `withReflection(react)`, mantendo o padrão atual de `reflect: true`. A flag `reflect` do `/chat` continua valendo e decora a rota efetiva quando ela é `react` ou `plan-and-execute`; se a rota já for `reflection`, a flag é ignorada (não dá para fazer reflection duplo). A resolução fica em `strategyForRoute(route, reflect, extraTools)` em `agents/index.ts`, reaproveitando `resolveStrategy`.
- **Rationale**: mantém a compatibilidade e evita estratégias aninhadas sem sentido.

## 8. Timeout

- **Decision**: generalizar `chat.service.ts` com `withTimeout<T>(run: () => Promise<T>, timeoutMs)`. `runWithTimeout` passa a delegar para ela, e o controller envolve a execução inteira do grafo (roteador incluído).
- **Rationale**: o teto de 180 s cobre o fluxo inteiro (premissa da spec), e o teste existente de `runWithTimeout` continua válido.

## 9. Testabilidade sem rede

- **Decision**: `createProductionGraph(deps)` recebe `decideRoute: (prompt) => Promise<{ decision; tokenUsage }>` e `strategyFor: (route) => ReasoningStrategy` por injeção. `createApp` ganha a opção `decideRoute` e mantém `resolveStrategy` para os fakes já existentes. O roteador real (`createModelRouter()`) só é instanciado no default.
- **Rationale**: o mesmo padrão de injeção que já existe em `CreateAppOptions`. Os testes de grafo, roteador puro, trace e servidor rodam sem OpenRouter.
