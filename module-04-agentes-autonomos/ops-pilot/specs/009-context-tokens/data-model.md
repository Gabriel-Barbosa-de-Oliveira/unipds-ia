# Data Model: Medição de Contexto

Nenhuma tabela nova, nenhuma persistência — as duas entidades desta feature são valores transientes, calculados por resposta de `/chat` e nunca gravados (research.md item 9).

## TokenUsage

| Campo | Tipo | Regras |
|---|---|---|
| `promptTokens` | number | Soma dos tokens de prompt de todas as chamadas ao modelo feitas para produzir uma resposta — real quando o provedor relata, estimado (`estimateTokens`) quando não (research.md itens 1–4) |
| `source` | `"real" \| "estimated" \| "mixed"` | `"real"`: todas as chamadas relataram uso real. `"estimated"`: nenhuma relatou (todas caíram no fallback). `"mixed"`: parte real, parte estimada (research.md item 4) |

Produzido por `UsageCollector` (`src/context/tokens.ts`), um callback LangChain acumulado ao longo de uma execução (`handleLLMStart`/`handleLLMEnd`), e combinado entre a estratégia envolvida e a chamada de crítica de `reflection.ts` (quando `reflect: true`) via `mergeTokenUsage` (research.md itens 5–6).

### Regra de agregação

1. A cada chamada ao modelo (`handleLLMStart`), guarda uma estimativa (`estimateTokens` sobre o prompt capturado) associada ao `runId` daquela chamada.
2. Ao final da mesma chamada (`handleLLMEnd`), usa o uso real relatado (`output.llmOutput?.tokenUsage?.promptTokens`) se presente; senão, usa a estimativa guardada no passo 1.
3. `promptTokens` acumula o valor de cada chamada (real ou estimado); `source` reflete se todas, nenhuma, ou só parte das chamadas tiveram uso real.
4. Quando `reflect: true` aciona uma ou mais chamadas de crítica, o `TokenUsage` da estratégia envolvida é combinado (`mergeTokenUsage`) com o de cada chamada de crítica — o resultado final cobre **todas** as chamadas feitas para aquela resposta, não só a primeira (spec FR-002, US1 cenário 2).

## ContextBreakdown

| Campo | Tipo | Regras |
|---|---|---|
| `currentMessage` | number | `estimateTokens` sobre a mensagem atual da pessoa (`parsed.data.message`) |
| `conversationHistory` | number | `estimateTokens` sobre o conteúdo bruto das mensagens de histórico já carregadas para esta requisição (`006-conversation-history`) — `0` quando não há histórico |
| `recalledFacts` | number | `estimateTokens` sobre o texto bruto dos fatos recuperados para esta requisição (`007-semantic-memory`) — `0` quando não há `userId`/fatos |
| `total` | number | `currentMessage + conversationHistory + recalledFacts` — definido por construção, nunca medido de forma independente (research.md item 7) |

Produzido por `buildContextBreakdown` (`src/context/tokens.ts`), chamado em `src/http/server.ts` a partir das mesmas três peças que o controller já tem disponíveis antes de compor o prompt final — nenhuma das estratégias de raciocínio (`react.ts`/`plan-and-execute.ts`) precisa saber dessa decomposição.

### Regra de escrita: `buildContextBreakdown({ currentMessage, historyTexts, factTexts })`

1. `currentMessage = estimateTokens(currentMessage)`.
2. `conversationHistory = estimateTokens(historyTexts.join("\n"))` — lista vazia produz `""`, logo `0`.
3. `recalledFacts = estimateTokens(factTexts.join("\n"))` — lista vazia produz `""`, logo `0`.
4. `total = currentMessage + conversationHistory + recalledFacts` (FR-006, por construção).

## Relacionamento com `Metrics` (`src/agents/types.ts`) e a resposta de `POST /chat`

```text
UsageCollector (por execução de estratégia) ──> Metrics.promptTokens / Metrics.tokenSource
buildContextBreakdown (no controller HTTP)  ──> resposta de /chat: metrics.contextBreakdown
```

`TokenUsage` vira dois campos apostos em `Metrics` (`promptTokens`, `tokenSource`) — não um objeto aninhado, para manter o formato já usado por `llmCalls`/`latencyMs`. `ContextBreakdown` vira um campo aninhado único (`metrics.contextBreakdown`), já que suas 4 sub-partes só fazem sentido juntas. Ver [contracts/post-chat.md](./contracts/post-chat.md) para o formato completo da resposta.
