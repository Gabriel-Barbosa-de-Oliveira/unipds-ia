# Research: Resiliência de Modelo

> **Nota de implementação (seguindo o esboço do `/speckit-implement`)**: a fábrica se chama `createModel(build?, config?, deps?)` (não `resilientRunnable`), e o modelo base se chama `baseModel(id)` (não `createChatModel`). São **`stopAfterAttempt: 2`** (1 tentativa + 1 retry) **no principal e também na reserva**, compostos como `primary.withFallbacks([backup])`. Os números de tentativas citados abaixo (3 no principal, sem retry na reserva) foram substituídos por esses.

Base: `@langchain/core` 0.3.80, `@langchain/langgraph` 0.2.74, `@langchain/openai` 0.3.17 (versões instaladas). Todos os pontos que chamam o modelo hoje fazem `createModel()`: `src/agents/react.ts`, `src/agents/plan-and-execute.ts` (planner, replanner e executor), `src/agents/reflection.ts` (crítico), `src/graph/router.ts` e `src/memory/learning-reflector.ts`. Não ficou nenhum NEEDS CLARIFICATION em aberto.

## 1. Configuração: `OPENROUTER_MODEL_FALLBACK`

- **Decision**: uma função pura `loadModelConfig(env)` em `src/agents/model.ts` lê os dados com zod a partir do `env` recebido por parâmetro (o projeto nunca lê o `.env` diretamente; o processo já recebe as variáveis via `--env-file-if-exists` nos scripts npm). Ela devolve `{ primary: string; fallback?: string }`:
  - `OPENROUTER_MODEL` é obrigatório, e a mensagem de erro é a mesma de hoje;
  - `OPENROUTER_MODEL_FALLBACK` passa por `trim`; vazio, ausente ou igual a `primary` vira `undefined` (FR-005, US3-3).
- **Rationale**: é o mesmo padrão de `loadContextBudget(env)` da 011. Por ser pura, dá para testar sem mexer no `process.env`.
- **Alternatives considered**: uma lista de reservas (`OPENROUTER_MODEL_FALLBACKS=a,b`). O `withFallbacks` aceitaria, mas a spec pede uma única reserva. Deixado para o futuro.

## 2. Retry no principal: `withRetry`

- **Decision**: no principal, usar `.withRetry({ stopAfterAttempt: 3, onFailedAttempt })`. O `onFailedAttempt` **relança** o erro quando `isTransientModelError(error)` é falso, o que interrompe o `p-retry` (FR-003). A espera padrão do `p-retry` usado pelo `RunnableRetry` é exponencial (cerca de 1 s e depois 2 s), o que dá uns 3 s extras no pior caso, abaixo dos 10 s do SC-004.
- `isTransientModelError(error)` (pura) é verdadeira para:
  - status 429, 408 ou ≥ 500 (`error.status` / `error.response.status`);
  - os códigos `ECONNRESET`, `ETIMEDOUT`, `ECONNREFUSED`, `EAI_AGAIN` e `UND_ERR_*`;
  - os nomes `APIConnectionError`, `APIConnectionTimeoutError`, `TimeoutError` e `AbortError` vindo do cliente HTTP;
  - mensagens contendo `fetch failed`, `rate limit` ou `timeout` (sem diferenciar maiúsculas).

  Qualquer outra coisa é falsa: 400/401/403/404, erro de validação, `Error` genérico.
- **Rationale**: é o que o usuário pediu (`withRetry` no principal), e a classificação fica pura e testável.
- **Alternatives considered**: o `maxRetries` nativo do `ChatOpenAI`. Ele já existe (padrão 6, dentro do cliente), mas fica invisível ao LangChain, não tem classificação nossa e soma com o nosso retry. **Decisão complementar**: fixar `maxRetries: 0` no `ChatOpenAI` para que exista uma única política de retry e o SC-004 seja previsível.

## 3. Reserva: `withFallbacks([reserva])`

- **Decision**: `primaryRunnable.withRetry(...).withFallbacks([fallbackRunnable])`, sem retry na reserva. O `withFallbacks` trata qualquer erro por padrão, então falhas não transitórias também caem na reserva (FR-004). Sem reserva configurada, a função devolve só `primaryRunnable.withRetry(...)` (FR-006).
- **Ponto crítico, a ordem de composição**: `withFallbacks` devolve um `RunnableWithFallbacks`, que **não** tem `bindTools` nem `withStructuredOutput`. Por isso a resiliência é aplicada **depois** do bind, sobre o runnable já especializado. A fábrica recebe um `build(model)`:

  ```ts
  resilientRunnable((m) => m.withStructuredOutput(schema))   // roteador, crítico, planner, replanner, refletor
  resilientRunnable((m) => m.bindTools(tools))               // modelo do agente ReAct
  ```

  Cada ramo (principal e reserva) faz o próprio bind, com o mesmo `build`.
- **Rationale**: é a única forma de usar `withRetry`/`withFallbacks` (o que o usuário pediu) preservando a saída estruturada e o tool calling.
- **Alternatives considered**: uma subclasse de `BaseChatModel` com retry/fallback dentro de `_generate`. Integraria sem nenhum ajuste, mas descarta os primitivos pedidos. Rejeitada.

## 4. `createReactAgent` com o modelo resiliente

- **Problema**: `createReactAgent` (0.2.74, `_shouldBindTools`) chama `llm.bindTools(tools)`, a menos que `llm` seja um `RunnableBinding` cujo `kwargs.tools` já contenha as mesmas tools. Um `RunnableWithFallbacks` não é `RunnableBinding` nem tem `bindTools`, e falharia com `llm ... must define bindTools method`.
- **Decision**: a função `toolCallingModel(tools)` em `model.ts` devolve `new RunnableBinding({ bound: resilientRunnable((m) => m.bindTools(tools)), kwargs: { tools: <specs do primary.bindTools(tools).kwargs.tools> }, config: {} })`. Com isso, o `_shouldBindTools` vê o binding com as tools certas e não tenta religar. Na invocação, o `kwargs.tools` (idêntico ao de cada ramo) é repassado como opção e não muda nada. Esse é o **único** lugar com esse ajuste, e ele é coberto por um teste que monta `createReactAgent` sobre ele com `FakeStreamingChatModel` (que tem `bindTools`), sem rede.
- **Alternatives considered**:
  - atualizar o LangGraph para uma versão em que o `llm` pode ser uma função. Isso muda a stack fora do escopo;
  - reescrever o loop ReAct. Muito mais código. Rejeitada.

## 5. Detectar fallback e `modelUsed`: callback + função pura

- **Decision**: o `withFallbacks` não emite um "evento de fallback"; ele só tenta o próximo runnable. A detecção é feita por observação:
  - **`ModelUsageTracker extends BaseCallbackHandler`** (em `src/agents/model-usage.ts`) registra, numa lista, `{ kind: "start" | "end" | "error", runId, model, error? }`. O `model` vem de `extraParams.invocation_params.model` em `handleChatModelStart`, e o `runId` liga `end`/`error` ao `start`. Ele é passado em `callbacks`, ao lado do `LlmCallCounter`/`UsageCollector` que já existem.
  - **`summarizeModelUsage(log, config)`** (pura) devolve `{ fallbacks: ModelFallback[]; modelUsed: string }`:
    - **fallback**: um `start` com `model === config.fallback` depois de um `error` com `model === config.primary` gera `{ from: primary, to: fallback, reason: resumo do último erro do principal }`. Retries do principal (vários `error` seguidos de `start` no principal) **não** geram fallback (FR-008);
    - **`modelUsed`**: o `model` do último `end` bem-sucedido. Sem nenhum `end`, `config.primary`.
- **Rationale**: as estratégias continuam sem conhecer a fábrica. A lógica que importa é pura e testada com logs sintéticos (FR-013), e o callback é trivial.
- **Alternatives considered**: um `RunnableLambda` em volta de cada ramo, empurrando eventos para um sink. Exigiria passar o sink por todas as estratégias. Rejeitado.

## 6. Onde o evento entra no trace

- **Decision**: novo `TraceEvent` `{ type: "fallback"; at; from: string; to: string; reason: string; node? }`. Cada estratégia insere os eventos de `summarizeModelUsage(...).fallbacks` com a função pura `withModelFallbacks(trace, fallbacks)`. Os eventos entram **antes do último evento `answer`** (ou no fim, se não houver `answer`), e o `at` é reindexado. Assim a resposta continua sendo o último evento e o `lastAnswer` não muda.
  - **React / plan-and-execute**: um tracker por `run()`.
  - **Reflection**: concatena os traces das tentativas (que já trazem os próprios fallbacks), e o crítico ganha um tracker próprio cujos fallbacks entram antes do evento `critique` da rodada.
  - **Roteador (012)**: o `DecideRoute` passa a devolver também `fallbacks` e `modelUsed`. O nó `roteador` emite os eventos `fallback` **depois** do evento `route`, com `node: "roteador"`, para manter o `route` em `trace[0]` (invariante da 012).
- **Rationale**: o evento `fallback` (troca de modelo) é um `type` distinto da `source: "fallback"` do evento `route` (troca de rota), o que deixa os dois conceitos distinguíveis (edge case da spec).
- **Ajuste nas invariantes da 012**: "todo evento depois do índice 0 tem `node === route`" passa a ser "todo evento depois do índice 0 tem `node === route` **ou** é `type: "fallback"` com `node: "roteador"`". O teste correspondente precisa ser atualizado.

## 7. `metrics.modelUsed`

- **Decision**: `Metrics` ganha `modelUsed: string` **obrigatório** (SC-003 por tipo).
  - `buildMetrics(counter, usageCollector, latencyMs, modelUsed)`.
  - Na reflection, é o `modelUsed` da última tentativa (a que produziu a resposta).
  - No grafo, é o `modelUsed` da estratégia, porque o roteador não produz a resposta final.
  - `formatMetrics` passa a mostrar `model=<id>`.
  - Os fixtures de teste com `Metrics` literal (cerca de 27) ganham `modelUsed`.
- **Alternatives considered**: deixar o campo opcional. Isso evitaria tocar nos fixtures, mas perderia a garantia de tipo de que toda resposta informa o modelo. Rejeitado.

## 8. API da fábrica e compatibilidade

- **Decision**: `src/agents/model.ts` exporta:
  - `loadModelConfig(env)` (pura);
  - `createChatModel(modelId)`: um `ChatOpenAI` com `maxRetries: 0`, temperatura 0 e OpenRouter;
  - `resilientRunnable(build, config?, deps?)`;
  - `toolCallingModel(tools, config?, deps?)`.

  O `deps` (`createChatModel`, `onFailedAttempt` e o `isTransient`) é injetável para os testes usarem fakes. `createModel()` deixa de ser usado pelos call sites e é removido, junto com a atualização dos 6 usos.
- **Rationale**: há uma única fábrica (FR-001), e o injetável segue o padrão do projeto.

## 9. Teto de tempo

- **Decision**: nada muda. O `withTimeout` da 012 envolve o grafo inteiro. O retry de cerca de 3 s mais a reserva cabem nos 180 s, e se o teto vencer continua valendo o 504 atual.
