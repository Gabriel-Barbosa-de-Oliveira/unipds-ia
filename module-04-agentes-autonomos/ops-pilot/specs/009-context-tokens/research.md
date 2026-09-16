# Research: Medição de Contexto

## 1. Como obter o uso real de tokens do LangChain

**Decision**: um novo callback (`UsageCollector`, `src/context/tokens.ts`, `extends BaseCallbackHandler` — mesma classe-base já usada por `LlmCallCounter`, `src/agents/metrics.ts`) implementa `handleLLMEnd(output: LLMResult, runId: string)` e lê `output.llmOutput?.tokenUsage?.promptTokens` — campo populado por `@langchain/openai`/`ChatOpenAI` (confirmado em `node_modules/@langchain/openai/dist/chat_models.cjs`, ex.: `tokenUsage: { completionTokens, promptTokens, totalTokens }`) sempre que o provedor (OpenRouter, compatível com a API da OpenAI) retorna uso na resposta.

**Rationale**: é o mesmo mecanismo de callback que o projeto já usa para contar `llmCalls` (`LlmCallCounter#handleLLMStart`) — reaproveita a mesma infraestrutura (`callbacks: [...]` já passado em todo `invoke`/`stream`), sem precisar inspecionar o valor de retorno de cada chamada (que, com `withStructuredOutput` no modo padrão do projeto — sem `includeRaw` —, é só o objeto já parseado, sem acesso à mensagem bruta onde o uso viveria).

**Alternatives considered**: ler `usage_metadata` da `AIMessage` retornada por `agent.stream(...)`/`.invoke(...)`. Rejeitada como fonte única: funciona para o retorno de `react.ts` (que já tem acesso às `BaseMessage[]`), mas não para `plan-and-execute.ts`/`reflection.ts`, que usam `withStructuredOutput(...).invoke(...)` sem `includeRaw: true` — mudar isso alteraria o tipo de retorno consumido por código já existente (`result.steps`, `verdict.approved`), um raio de mudança maior que necessário. O callback funciona identicamente nos dois casos, sem tocar nenhum retorno existente.

## 2. `estimateTokens` — fórmula e onde vive

**Decision**: `estimateTokens(text: string): number` em `src/context/tokens.ts`, pura: `Math.ceil(text.length / 4)` — exatamente a razão "chars/4" pedida no input original.

**Rationale**: heurística padrão de mercado para estimativa rápida de tokens sem tokenizador real (aprox. 4 caracteres por token em inglês/português); suficiente para o propósito desta feature (monitoramento, não faturamento exato — spec Assumptions). Função pura, testável com strings forjadas, sem rede.

## 3. Fallback: quando o uso real não vem, estimar a partir do prompt capturado em `handleLLMStart`

**Decision**: `UsageCollector` também implementa `handleLLMStart(llm: Serialized, prompts: string[], runId: string)`, guardando `estimateTokens(prompts.join("\n"))` num `Map<runId, number>` por chamada em andamento. Quando `handleLLMEnd` chega para aquele `runId`: se `output.llmOutput?.tokenUsage?.promptTokens` é um `number`, soma esse valor real ao total e marca a chamada como `"real"`; caso contrário, soma a estimativa guardada e marca como `"estimated"`.

**Rationale**: `handleLLMStart` dispara para **toda** chamada de chat model — confirmado em `node_modules/@langchain/core/dist/callbacks/manager.cjs`: `handleChatModelStart` cai para `handler.handleLLMStart?.(llm, [messageString], ...)` quando o handler não implementa `handleChatModelStart`, onde `messageString` é a serialização das mensagens enviadas — é por isso que `LlmCallCounter` (que só implementa `handleLLMStart`) já conta corretamente as chamadas de hoje, mesmo `ChatOpenAI` sendo um chat model. `UsageCollector` usa exatamente esse mesmo fallback para capturar o texto do prompt de toda chamada, sem precisar implementar `handleChatModelStart` separadamente.

**Alternatives considered**: estimar a partir do `input`/prompt já conhecido em cada call site (`react.ts`, `plan-and-execute.ts`), passado explicitamente para `UsageCollector`. Rejeitada: exigiria threading manual do texto de prompt por 5 pontos de chamada diferentes (planner/executor/replanner/react/critique), cada um com sua própria forma de montar o prompt; capturar via `handleLLMStart` é automático e uniforme, sem esse acoplamento.

## 4. `source: "real" | "estimated" | "mixed"` em vez de dois números sempre separados

**Decision**: `TokenUsage { promptTokens: number; source: "real" | "estimated" | "mixed" }` — `promptTokens` é sempre a soma (real quando disponível, estimado quando não); `source` é `"real"` só quando **todas** as chamadas contribuíram com uso real, `"estimated"` quando **nenhuma** contribuiu, `"mixed"` quando parte veio real e parte estimada.

**Rationale**: satisfaz FR-004 (nunca deixar a origem ambígua) com um contrato simples — um número mais um rótulo — em vez de expor `realPromptTokens`/`estimatedPromptTokens` sempre separados, o que forçaria todo consumidor da métrica a somar os dois na maioria dos casos (o caso comum, com o provedor atual, é `source: "real"` sempre — OpenRouter/API compatível com OpenAI retorna uso em modo não-streaming, que é o modo usado por este projeto). `"mixed"` continua honesto sobre o caso raro, sem inventar uma terceira fonte que pareça "real" ou "estimada" quando não é nenhuma das duas isoladamente.

**Alternatives considered**: sempre expor os dois números (`real`/`estimated`) lado a lado, sem um total único. Rejeitada por verbosidade desnecessária frente ao caso comum, e por não corresponder ao pedido original ("promptTokens real") — um campo, não dois.

## 5. `reflection.ts#critique` precisa ganhar o mesmo callback (hoje não passa nenhum)

**Decision**: `critique()` cria seu próprio `UsageCollector` (e continua sem precisar de `LlmCallCounter`, já que `llmCalls` da crítica já é contado manualmente em `runReflectionLoop`) e passa `{ callbacks: [usageCollector] }` para `createModel().withStructuredOutput(verdictSchema).invoke(...)`. `critique()` passa a retornar `{ verdict: Verdict; tokenUsage: TokenUsage }` em vez de só `Verdict`; `runReflectionLoop` combina `tokenUsage` de cada tentativa (`attempt.metrics.tokenUsage`, da estratégia envolvida) com o da própria crítica via `mergeTokenUsage` (research.md item 6), acumulando entre iterações — mesmo espírito já usado para `llmCalls` (`llmCalls += attempt.metrics.llmCalls; llmCalls += 1`).

**Rationale**: spec (US1, cenário 2) exige explicitamente que a contagem reflita **todas** as chamadas de uma resposta, inclusive quando `reflect: true` aciona uma ou mais chamadas de crítica/regeneração — sem essa mudança, o token accounting ficaria incompleto exatamente no caso que a spec pede para cobrir.

**Alternatives considered**: deixar a crítica fora da contagem de tokens (só `llmCalls` continuaria contando-a). Rejeitada: violaria diretamente o cenário 2 da User Story 1 do spec.

## 6. `mergeTokenUsage` — combinando dois `TokenUsage` já finalizados

**Decision**: função pura em `src/context/tokens.ts`: `mergeTokenUsage(a: TokenUsage, b: TokenUsage): TokenUsage` — soma `promptTokens`; `source` é `a.source` quando `a.source === b.source`, senão `"mixed"`.

**Rationale**: usada por `reflection.ts` para combinar o `tokenUsage` já calculado da tentativa da estratégia envolvida com o da própria chamada de crítica (research.md item 5) — dois valores já processados pelo `UsageCollector` de cada lado, não um acumulador em construção (por isso não reaproveita a mesma lógica interna da classe, que lida com estado parcial por `runId`).

## 7. `contextBreakdown` é calculado no controller HTTP, não nas estratégias

**Decision**: `buildContextBreakdown({ currentMessage: string; historyTexts: readonly string[]; factTexts: readonly string[] }): ContextBreakdown` — função pura em `src/context/tokens.ts`, chamada em `src/http/server.ts` a partir das mesmas três peças que o controller já tem separadas (`parsed.data.message`, `history.map(m => m.content)`, `recalledFacts`) **antes** de compor o prompt final (`composePrompt`/`composeWithFacts`). Cada parte usa `estimateTokens` sobre o texto bruto (conteúdo das mensagens/fatos concatenado, não a string final já formatada com rótulos como "Histórico da conversa até aqui:"). `ContextBreakdown.total` é definido por construção como a soma das três partes — nunca medido de forma independente.

**Rationale**: só o controller conhece a decomposição em histórico/fatos/mensagem atual — as estratégias (`react.ts`/`plan-and-execute.ts`) só recebem um `input: string` já composto (research.md de `006-conversation-history`, item 1: nenhuma estratégia sabe que existe histórico por trás). Calcular ali evita introduzir esse conhecimento nas estratégias. Definir `total` como a soma das partes (em vez de comparar contra `estimateTokens` da string final formatada) satisfaz FR-006 **por construção**, sem depender dos rótulos exatos que `composePrompt`/`composeWithFacts` adicionam (que mudam a contagem de caracteres por um valor pequeno e não fazem parte do que se quer decompor).

**Alternatives considered**: medir `estimateTokens` sobre a string final já composta (`prompt`, depois de `composePrompt`+`composeWithFacts`) e reconciliar contra a soma das partes. Rejeitada: os rótulos/formatação adicionados por essas funções (cabeçalhos, quebras de linha) fariam a soma das partes nunca bater exatamente com o total medido independentemente, exigindo um "resto"/ajuste artificial só para fechar a conta — complexidade desnecessária frente ao que a spec pede (entender a proporção de cada parte, não uma contagem byte-exata).

## 8. `contextBreakdown.total` é uma quantidade diferente de `metrics.promptTokens` — e isso é documentado, não reconciliado

**Decision**: `metrics.promptTokens` (real/estimado, research.md itens 1–4) e `metrics.contextBreakdown.total` (research.md item 7) **não** precisam ser iguais, e nenhuma lógica tenta forçá-los a bater.

**Rationale**: `metrics.promptTokens` reflete tudo que o provedor efetivamente processou/cobrou — inclui overhead de sistema, definição de tools (schemas JSON de `list_alerts`/`open_incident`/etc.), formatação interna do LangGraph/ReAct — nada disso o controller HTTP compõe ou controla diretamente. `contextBreakdown` decompõe só o que o próprio projeto monta (histórico + fatos + mensagem atual). São medidas de coisas diferentes por design; documentar essa diferença evita que alguém interprete uma divergência entre os dois números como bug.

## 9. Nenhuma tabela nova, nenhuma mudança em `ChatRequestSchema`

**Decision**: toda a mudança de contrato é aditiva em `metrics` na resposta de `POST /chat` (`metrics.promptTokens`, `metrics.tokenSource`, `metrics.contextBreakdown`) — nenhum campo novo de requisição, nenhuma persistência nova.

**Rationale**: a feature é puramente observacional (spec FR-008) — não há necessidade de armazenar nada; expandir só a resposta já entregue é a mudança de menor raio possível (Princípio VIII).
