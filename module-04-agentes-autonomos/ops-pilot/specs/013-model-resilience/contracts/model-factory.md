# Contract: fábrica de modelo resiliente (`src/agents/model.ts`, `src/agents/model-usage.ts`)

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**: a fábrica se chama `createModel(build?, config?, deps?)` (não `resilientRunnable`), e o modelo base se chama `baseModel(id)` (não `createChatModel`). São **`stopAfterAttempt: 2`** (1 tentativa + 1 retry) **no principal e também na reserva**, compostos como `primary.withFallbacks([backup])`. Os números de tentativas citados abaixo (3 no principal, sem retry na reserva) foram substituídos por esses.

As assinaturas abaixo são ilustrativas. A codificação final é detalhe de tarefa.

```ts
// src/agents/model.ts
export interface ModelConfig { primary: string; fallback?: string }

/** Pura. Lança Error("OPENROUTER_MODEL não configurada") quando primary falta, como hoje. */
export function loadModelConfig(env: Readonly<Record<string, string | undefined>>): ModelConfig;

/** Pura. true para 429/408/5xx, erros de rede/timeout; false para o resto (ver research.md item 2). */
export function isTransientModelError(error: unknown): boolean;

/** ChatOpenAI sobre OpenRouter, temperature 0, maxRetries 0. Lança se OPENROUTER_API_KEY faltar. */
export function createChatModel(modelId: string): ChatOpenAI;

export interface ResilienceDeps {
  createChatModel: (modelId: string) => BaseChatModel;
  /** Padrão 3 (1 tentativa + 2 retries). */
  stopAfterAttempt?: number;
}

/**
 * build(primary).withRetry({ stopAfterAttempt, onFailedAttempt: relança se !isTransientModelError })
 *   .withFallbacks([build(fallback)])   // só se config.fallback
 */
export function resilientRunnable<T>(
  build: (model: BaseChatModel) => Runnable<BaseLanguageModelInput, T>,
  config?: ModelConfig,          // padrão: loadModelConfig(process.env)
  deps?: ResilienceDeps,
): Runnable<BaseLanguageModelInput, T>;

/** RunnableBinding com kwargs.tools sobre resilientRunnable((m) => m.bindTools(tools)), aceito por createReactAgent. */
export function toolCallingModel(
  tools: StructuredToolInterface[],
  config?: ModelConfig,
  deps?: ResilienceDeps,
): RunnableBinding<BaseLanguageModelInput, AIMessageChunk>;

// src/agents/model-usage.ts
export class ModelUsageTracker extends BaseCallbackHandler { readonly log: ModelCallEntry[] }
export function summarizeModelUsage(log: readonly ModelCallEntry[], config: ModelConfig): ModelUsageSummary;
/** Pura. Insere fallbacks antes do último `answer` e reindexa `at`. */
export function withModelFallbacks(trace: readonly TraceEvent[], fallbacks: readonly ModelFallback[]): TraceEvent[];
```

## Regras

- Todo call site de modelo usa `resilientRunnable` ou `toolCallingModel` (FR-001). Nenhum chama `new ChatOpenAI` diretamente.
- No caminho feliz, há exatamente 1 chamada ao principal e 0 à reserva (SC-005).
- Com erro transitório seguido de sucesso, há 2 chamadas ao principal e nenhum fallback.
- Com erro transitório persistente, há 3 chamadas ao principal e 1 à reserva, gerando 1 fallback.
- Com erro não transitório, há 1 chamada ao principal e 1 à reserva, gerando 1 fallback.
- Sem reserva, o erro do principal propaga depois das tentativas, e o tipo de erro é o mesmo de hoje.
- Erro na reserva propaga, sem retry na reserva.
- `formatTraceEvent({ type: "fallback", from: "a", to: "b", reason: "429" })` → `"[fallback] a → b: 429"` (com prefixo `node │ ` quando houver `node`).
- `formatMetrics` → `"llmCalls=N latencyMs=N model=<modelUsed>"`.
