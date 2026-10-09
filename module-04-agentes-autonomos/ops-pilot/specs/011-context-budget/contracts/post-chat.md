# Contract: `POST /chat` — após `010-context-budget`

Estende [`009-context-tokens/contracts/post-chat.md`](../../009-context-tokens/contracts/post-chat.md). **Requisição inalterada.**

## `createApp` (`src/http/server.ts`)

`CreateAppOptions` ganha:

| Campo | Tipo | Padrão |
|---|---|---|
| `contextBudget` | `ContextBudget` | `loadContextBudget(process.env)`, lido uma vez na criação do app |

## Resposta (`200`)

```json
{
  "answer": "...",
  "trace": [...],
  "conversationId": "...",
  "metrics": {
    "llmCalls": 2,
    "latencyMs": 842,
    "promptTokens": 187,
    "tokenSource": "real",
    "historyMessages": 8,
    "contextBreakdown": {
      "system": 0,
      "summary": 0,
      "currentMessage": 9,
      "conversationHistory": 1190,
      "recalledFacts": 15,
      "total": 1214
    },
    "contextTrimmed": { "historyMessages": 4, "recalledFacts": 0 }
  }
}
```

| Campo (`metrics`) | Desde | Regra |
|---|---|---|
| `historyMessages` | 006 | Mensagens de histórico **efetivamente enviadas** (pós-corte) |
| `contextBreakdown.system` | **010** | 0 enquanto o `/chat` não fornece system |
| `contextBreakdown.summary` | **010** | 0 enquanto não houver gerador de resumo |
| `contextBreakdown.conversationHistory` | 009 | Pós-corte; ≤ `contextBudget.window` |
| `contextBreakdown.recalledFacts` | 009 | Pós-corte; ≤ `contextBudget.memories` |
| `contextBreakdown.total` | 009 | Soma de todas as partes |
| `contextTrimmed.historyMessages` | **010** | Mensagens removidas pelo teto (0 sem corte) |
| `contextTrimmed.recalledFacts` | **010** | Memórias removidas pelo teto (0 sem corte) |

## Regras

- O `input` passado a `strategy.run` é `buildContext(...).prompt` — mesma montagem para qualquer `strategy`/`reflect`.
- Sem corte, o `input` é idêntico ao enviado antes desta feature.
- `contextTrimmed` está presente em toda resposta `200`, mesmo com zeros.
