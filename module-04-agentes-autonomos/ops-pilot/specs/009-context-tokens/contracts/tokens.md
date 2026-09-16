# Contract: `src/context/tokens.ts`

Ver [data-model.md](../data-model.md) para a forma de `TokenUsage`/`ContextBreakdown` e [research.md](../research.md) para as decisões por trás deste contrato.

## `estimateTokens(text: string): number`

```ts
function estimateTokens(text: string): number;
```

Pura. `Math.ceil(text.length / 4)` — string vazia retorna `0`, nunca negativo (research.md item 2).

## `TokenUsage` e `mergeTokenUsage`

```ts
type TokenSource = "real" | "estimated" | "mixed";

interface TokenUsage {
  promptTokens: number;
  source: TokenSource;
}

function mergeTokenUsage(a: TokenUsage, b: TokenUsage): TokenUsage;
```

`mergeTokenUsage` é pura: `promptTokens` soma; `source` é `a.source` quando igual a `b.source`, senão `"mixed"` (research.md item 6). Usada por `src/agents/reflection.ts` para combinar o `TokenUsage` da estratégia envolvida com o da chamada de crítica.

## `UsageCollector` — callback LangChain (`extends BaseCallbackHandler`)

```ts
class UsageCollector extends BaseCallbackHandler {
  name: "usage-collector";
  handleLLMStart(llm: Serialized, prompts: string[], runId: string): void;
  handleLLMEnd(output: LLMResult, runId: string): void;
  get tokenUsage(): TokenUsage;
}
```

- Uma instância por execução de estratégia (mesmo ciclo de vida de `LlmCallCounter`, `src/agents/metrics.ts`) — nunca compartilhada entre requisições.
- `handleLLMStart`: guarda `estimateTokens(prompts.join("\n"))` associado ao `runId` daquela chamada (dispara para toda chamada de chat model — research.md item 3).
- `handleLLMEnd`: se `output.llmOutput?.tokenUsage?.promptTokens` é um `number`, soma esse valor real ao total acumulado e marca a chamada como real; senão, soma a estimativa guardada em `handleLLMStart` para aquele `runId` e marca como estimada.
- `tokenUsage`: `{ promptTokens: <soma acumulada>, source: <"real" se todas reais, "estimated" se nenhuma, "mixed" caso contrário> }`.

### Uso (por call site — 5 no total, research.md item 5)

| Call site | Arquivo | Mudança |
|---|---|---|
| `react.ts` — `agent.stream(...)` | `src/agents/react.ts` | `callbacks: [counter, usageCollector]` (adiciona `usageCollector` ao array já existente) |
| `plan-and-execute.ts` — planner | `src/agents/plan-and-execute.ts` | idem |
| `plan-and-execute.ts` — executor | `src/agents/plan-and-execute.ts` | idem |
| `plan-and-execute.ts` — replanner | `src/agents/plan-and-execute.ts` | idem |
| `reflection.ts#critique` | `src/agents/reflection.ts` | **novo**: hoje não passa nenhum `callbacks` — ganha `{ callbacks: [usageCollector] }` |

## `buildMetrics` (`src/agents/metrics.ts`) — assinatura estendida

```ts
function buildMetrics(counter: LlmCallCounter, usageCollector: UsageCollector, latencyMs: number): Metrics;
```

`Metrics` (`src/agents/types.ts`) ganha dois campos, lidos de `usageCollector.tokenUsage`:

```ts
interface Metrics {
  llmCalls: number;
  latencyMs: number;
  promptTokens: number;   // [NOVO]
  tokenSource: TokenSource; // [NOVO]
}
```

## `ContextBreakdown` e `buildContextBreakdown`

```ts
interface ContextBreakdown {
  currentMessage: number;
  conversationHistory: number;
  recalledFacts: number;
  total: number;
}

function buildContextBreakdown(parts: {
  currentMessage: string;
  historyTexts: readonly string[];
  factTexts: readonly string[];
}): ContextBreakdown;
```

Pura. `total = currentMessage + conversationHistory + recalledFacts`, sempre (research.md item 7). Chamada só em `src/http/server.ts`, nunca dentro de uma `ReasoningStrategy` — as estratégias não sabem que existe histórico/fatos por trás do `input` que recebem (mesmo desacoplamento de `006-conversation-history`/`007-semantic-memory`).
