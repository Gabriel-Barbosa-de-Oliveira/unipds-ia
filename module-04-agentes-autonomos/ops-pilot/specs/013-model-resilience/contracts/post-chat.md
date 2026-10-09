# Contract: `POST /chat` (delta sobre a 012)

A requisição e os erros continuam iguais. A resposta `200` muda só por acréscimo:

- `metrics.modelUsed: string` passa a estar sempre presente.
- O `trace` pode conter eventos `{ "type": "fallback", "at", "node", "from", "to", "reason" }`.

Exemplo de uma resposta em que o principal falhou numa chamada da estratégia:

```json
{
  "answer": "Há 3 alertas firing...",
  "route": { "route": "react", "reason": "Consulta direta.", "source": "router" },
  "trace": [
    { "type": "route", "at": 0, "node": "roteador", "route": "react", "reason": "Consulta direta.", "source": "router" },
    { "type": "action", "at": 1, "node": "react", "tool": "list_alerts", "args": { "status": "firing" } },
    { "type": "observation", "at": 2, "node": "react", "result": [] },
    { "type": "fallback", "at": 3, "node": "react", "from": "openai/gpt-4o-mini", "to": "meta-llama/llama-3.1-70b-instruct", "reason": "429 Rate limit exceeded" },
    { "type": "answer", "at": 4, "node": "react", "content": "Há 3 alertas firing..." }
  ],
  "metrics": { "llmCalls": 6, "latencyMs": 5400, "promptTokens": 2100, "tokenSource": "real", "modelUsed": "meta-llama/llama-3.1-70b-instruct", "...": "campos de contexto inalterados" }
}
```

- O `llmCalls` conta todas as tentativas, incluindo os retries e a reserva. O `LlmCallCounter` já conta cada `handleLLMStart`.
- Se o principal e a reserva falharem, o resultado é o mesmo de hoje: `500 internal_error`, ou `504` se o teto de tempo vencer antes.
- Se o roteador falhar mesmo depois da reserva, vale o fallback de rota da 012 (`route.source: "fallback"`). Os eventos `fallback` de modelo do roteador aparecem logo depois do `route`, com `node: "roteador"`.
