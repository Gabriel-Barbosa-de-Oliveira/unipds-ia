# Contract: `POST /chat` (delta sobre 003/011)

Este documento só descreve o que muda. Tudo o que não está aqui segue igual ao que `specs/011-context-budget/contracts/post-chat.md` define.

## Requisição

| Campo | Obrigatório | Tipo | Mudança |
|---|---|---|---|
| `strategy` | Não | `"react" \| "planExecute" \| "plan-and-execute" \| "reflect" \| "reflection"` (os dois últimos de cada par são aliases) | Quando é **omitido**, a rota passa a ser decidida pelo roteador (antes o padrão era `react`). Quando é **informado**, vira override: o roteador não é chamado e o trace marca `source: "override"`. Agora `"reflection"` também é aceito. |
| `reflect` | Não | boolean | Sem mudança no significado. Decora a rota efetiva `react`/`planExecute` e é ignorado quando a rota é `reflect`. |

## `200 OK` (campos aditivos)

```json
{
  "answer": "Há 3 alertas firing...",
  "route": { "route": "react", "reason": "Consulta direta de alertas.", "source": "router" },
  "trace": [
    { "type": "route", "at": 0, "node": "roteador", "route": "react", "reason": "Consulta direta de alertas.", "source": "router" },
    { "type": "action", "at": 1, "node": "react", "tool": "list_alerts", "args": { "status": "firing" } },
    { "type": "observation", "at": 2, "node": "react", "result": [] },
    { "type": "answer", "at": 3, "node": "react", "content": "Há 3 alertas firing..." }
  ],
  "conversationId": "…",
  "metrics": {
    "llmCalls": 3,
    "latencyMs": 2100,
    "promptTokens": 1450,
    "tokenSource": "real",
    "historyMessages": 0,
    "contextBreakdown": { },
    "contextTrimmed": { }
  }
}
```

- `route` é um campo **novo** no topo da resposta e tem o mesmo conteúdo do evento `route`.
- Todo item de `trace` tem `node`.
- O trace tem exatamente um evento `type: "route"`, e ele vem antes de todos os eventos de estratégia.
- `metrics.llmCalls` e `metrics.promptTokens` incluem o roteador. No override, o roteador conta 0 chamadas.

### Override

Para a requisição `{ "message": "...", "strategy": "plan-and-execute" }`, o evento de rota fica assim:

```json
{ "type": "route", "at": 0, "node": "roteador", "route": "planExecute", "reason": "Estratégia informada pelo cliente", "source": "override" }
```

### Fallback

Quando o roteador falha ou devolve uma decisão inválida, a resposta continua sendo `200`, com o evento de rota assim:

```json
{ "type": "route", "at": 0, "node": "roteador", "route": "react", "reason": "Fallback: <causa>", "source": "fallback" }
```

## Erros

Sem mudança. Uma `strategy` fora dos valores aceitos continua resultando em `422 { "error": "unknown_strategy", "strategy": "<valor>" }`, sem executar nada. O `504` agora cobre o grafo inteiro, roteador incluído.
