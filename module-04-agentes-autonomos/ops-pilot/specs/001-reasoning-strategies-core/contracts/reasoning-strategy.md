# Contract: ReasoningStrategy interface

Interface comum implementada por `src/agents/react.ts` e `src/agents/plan-and-execute.ts`, definida em `src/agents/types.ts`. Assinatura ilustrativa (a codificação final de tipos é detalhe de tarefa):

```ts
type TraceEvent =
  | { type: "thought"; at: number; content: string }
  | { type: "action"; at: number; tool: "list_alerts" | "open_incident" | "resolve_incident"; args: Record<string, unknown> }
  | { type: "observation"; at: number; result: unknown }
  | { type: "plan"; at: number; steps: string[] }
  | { type: "critique"; at: number; content: string }
  | { type: "answer"; at: number; content: string };

interface Metrics {
  llmCalls: number;
  latencyMs: number;
}

interface RunResult {
  answer: string;
  trace: TraceEvent[];
  metrics: Metrics;
}

interface ReasoningStrategy {
  readonly name: string;
  run(input: string, options?: { maxIterations?: number }): Promise<RunResult>;
}
```

## Regras do contrato

- `run` NUNCA lança para condições de negócio esperadas (tool inválida, limite de passos atingido) — essas condições viram eventos `observation`/`answer` no trace e o `RunResult` é retornado normalmente. `run` só rejeita a Promise para falhas de infraestrutura não recuperáveis (ex.: `OPENROUTER_API_KEY` ausente).
- `trace` é ordenado cronologicamente (`at` crescente) e contém, no mínimo, um evento `answer` final quando a estratégia conclui normalmente.
- Ao atingir `maxIterations` (ou o padrão de 8 passos para `plan-and-execute`, FR-005) sem produzir uma resposta final, `run` retorna um `RunResult` com o trace parcial e um `answer` que indica explicitamente que o limite foi atingido (FR-006) — nunca um array de trace vazio nesse caso.
- `metrics.llmCalls` conta toda invocação do modelo criado por `src/agents/model.ts` durante aquele `run`, independentemente de quantos passos/tools foram executados.
- Todo evento `action` tem um evento `observation` correspondente subsequente no trace antes de qualquer novo `action` ou do `answer` final (FR-013) — nunca uma ação sem sua observação registrada.
