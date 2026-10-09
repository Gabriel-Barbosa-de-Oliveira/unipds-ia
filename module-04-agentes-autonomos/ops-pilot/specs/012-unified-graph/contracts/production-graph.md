# Contract: roteador e grafo de produção

> **Nota de implementação (nomes finais)**: seguindo o esboço passado no `/speckit-implement`, o código usa `src/graph/production-graph.ts` e `src/graph/router.ts` (não `src/agents/`); nós `contexto`, `roteador`, `react`, `planExecute`, `reflect`, `resposta`; rotas `react | planExecute | reflect`. O `strategy` do `/chat` aceita também os nomes legados `plan-and-execute` e `reflection` como aliases. Onde este documento usa os nomes antigos (`context`/`router`/`answer` como nós, `plan-and-execute`/`reflection` como rotas), leia com esse mapeamento.

As assinaturas abaixo são ilustrativas. A codificação final é detalhe de tarefa.

```ts
// src/agents/router.ts
export const ROUTE_TABLE: readonly { route: RouteName; whenToUse: string; examples: string[] }[];
export const RouteDecisionSchema: z.ZodObject<{ route: z.ZodEnum<[...RouteName]>; reason: z.ZodString }>;

/** Pura. O system prompt contém a tabela markdown gerada a partir de ROUTE_TABLE. */
export function buildRouterMessages(prompt: string): [string, string][];

/** Pura. Aplica override > decisão válida > fallback(react). */
export function resolveRouteDecision(input: {
  override?: RouteName;
  decided?: unknown;          // saída crua do modelo (validada aqui com RouteDecisionSchema)
  error?: unknown;
}): RouteDecision;

/** Lança UnknownStrategyError para nomes fora de RouteName. */
export function parseRouteName(name: string): RouteName;

export type DecideRoute = (prompt: string) => Promise<{ decided: unknown; tokenUsage: TokenUsage }>;
/** Implementação real: createModel().withStructuredOutput(RouteDecisionSchema). Única IO deste módulo. */
export function createModelRouter(): DecideRoute;

// src/agents/trace.ts (ou production-graph.ts)
/** Pura. Carimba `node` e reindexa `at` a partir de `offset`. */
export function tagTrace(events: readonly TraceEvent[], node: GraphNode, offset: number): ProductionTraceEvent[];

// src/agents/production-graph.ts
export interface ProductionGraphDeps {
  decideRoute: DecideRoute;
  strategyFor: (route: RouteName) => ReasoningStrategy;
}
export interface ProductionInput {
  context: ContextInput;
  budget: ContextBudget;
  override?: RouteName;
}
export function createProductionGraph(deps: ProductionGraphDeps): {
  run(input: ProductionInput): Promise<ProductionRunResult>;
};

// src/agents/index.ts
/** route=reflection → withReflection(react). reflect=true decora react/plan-and-execute. */
export function strategyForRoute(route: RouteName, reflect?: boolean, extraTools?: StructuredToolInterface[]): ReasoningStrategy;

// src/services/chat.service.ts
export function withTimeout<T>(run: () => Promise<T>, timeoutMs: number): Promise<T>;
```

## Regras

- O nó `router` com `override` definido **não** chama `decideRoute`.
- Exceções de `decideRoute` nunca escapam do nó `router`: viram `source: "fallback"`.
- Exceções da estratégia escolhida propagam como hoje. Não há retry em outra rota.
- `strategyFor` é chamado só para a rota escolhida.
- Toda saída de `run` satisfaz as invariantes de trace de [data-model.md](../data-model.md#invariantes-do-trace-do-grafo).
- `formatTraceEvent` formata o evento `route` como `[route] <route> (<source>): <reason>` e, quando há `node`, prefixa a linha com `<node> │ `.
