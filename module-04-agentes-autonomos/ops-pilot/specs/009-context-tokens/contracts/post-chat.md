# Contract: `POST /chat` (`src/http/server.ts`, `createApp()`) — após `009-context-tokens`

Estende o contrato de [`008-learning-reflector/contracts/post-chat.md`](../../008-learning-reflector/contracts/post-chat.md). **Nenhum campo de requisição muda** — a mudança é só em `metrics`, na resposta.

## Requisição

Idêntica ao contrato de `008` — mesmos campos (`message`, `strategy`, `reflect`, `conversationId`, `userId`).

## Resposta (`200`)

```json
{
  "answer": "...",
  "trace": [...],
  "conversationId": "...",
  "metrics": {
    "llmCalls": 2,
    "latencyMs": 842,
    "historyMessages": 3,
    "promptTokens": 187,
    "tokenSource": "real",
    "contextBreakdown": {
      "currentMessage": 9,
      "conversationHistory": 42,
      "recalledFacts": 15,
      "total": 66
    }
  }
}
```

| Campo (`metrics`) | Tipo | Desde | Regra |
|---|---|---|---|
| `llmCalls`, `latencyMs` | number | `001` | Inalterados |
| `historyMessages` | number | `006` | Inalterado |
| `promptTokens` | number | **`009`** | Soma dos tokens de prompt de todas as chamadas ao modelo feitas para esta resposta — real quando o provedor relata, estimado quando não (data-model.md § TokenUsage) |
| `tokenSource` | `"real" \| "estimated" \| "mixed"` | **`009`** | Origem de `promptTokens` — nunca ambígua (FR-004) |
| `contextBreakdown` | object | **`009`** | Decomposição do contexto composto pelo próprio projeto (mensagem atual, histórico, fatos) — ver abaixo |
| `contextBreakdown.currentMessage` | number | **`009`** | `estimateTokens` sobre a mensagem desta requisição |
| `contextBreakdown.conversationHistory` | number | **`009`** | `estimateTokens` sobre o histórico carregado; `0` sem histórico |
| `contextBreakdown.recalledFacts` | number | **`009`** | `estimateTokens` sobre os fatos recuperados; `0` sem `userId`/fatos |
| `contextBreakdown.total` | number | **`009`** | Soma das três partes acima — sempre exata, por construção (FR-006) |

## Regras do contrato

- `promptTokens`/`tokenSource`/`contextBreakdown` estão presentes em **toda** resposta `200` de `/chat` que aciona o modelo de raciocínio — nunca omitidos (SC-002).
- `contextBreakdown.total` nunca diverge da soma de `currentMessage + conversationHistory + recalledFacts` (FR-006, SC-003) — é a mesma soma, não um valor medido à parte.
- `contextBreakdown.total` **não** é comparável a `promptTokens` — medem coisas diferentes (research.md item 8): `promptTokens` inclui tudo que o provedor processou (system prompt interno, schemas de tools); `contextBreakdown` decompõe só o que o projeto compõe (histórico + fatos + mensagem atual).
- Quando `reflect: true` aciona chamadas de crítica adicionais, `promptTokens` cobre todas elas, não só a primeira chamada da estratégia base (FR-002, US1 cenário 2).
- A medição nunca altera `answer` nem o comportamento do raciocínio (FR-008) — puramente informativa.
- Métricas de requisições concorrentes nunca se misturam (FR-009) — cada `UsageCollector`/`buildContextBreakdown` é isolado por requisição, mesmo padrão já usado por `LlmCallCounter`.
