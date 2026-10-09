# Contrato HTTP: Modo Equipe

Este contrato estende o da 015 ([`../015-war-room-web/contracts/http.md`](../../015-war-room-web/contracts/http.md)). Os formatos de resposta não mudam; só aparecem valores novos.

## `POST /chat`

- `strategy` aceita `"team"` e o alias `"equipe"`. Um nome desconhecido continua dando 422 `unknown_strategy`.
- `route.route` pode ser `"team"`, com `source` `"router"`, `"override"` ou `"fallback"`.
- O 202 da equipe tem o mesmo formato do 202 atual. Ele aparece quando o executor propõe uma ação.

### Exemplo de `trace` da equipe (200)

```json
[
  { "type": "route", "at": 0, "node": "roteador", "route": "team", "reason": "Investigar e agir", "source": "router" },
  { "type": "handoff", "at": 1, "node": "team", "role": "supervisor", "from": "supervisor", "to": "analista", "brief": "Levante alertas e incidentes do checkout-api" },
  { "type": "action", "at": 2, "node": "team", "role": "analista", "tool": "list_alerts", "args": { "status": "firing" } },
  { "type": "observation", "at": 3, "node": "team", "role": "analista", "result": [] },
  { "type": "handoff", "at": 4, "node": "team", "role": "supervisor", "from": "supervisor", "to": "planejador", "brief": "Proponha passos com base nos fatos" },
  { "type": "plan", "at": 5, "node": "team", "role": "planejador", "steps": ["Abrir incidente high para checkout-api"] },
  { "type": "handoff", "at": 6, "node": "team", "role": "supervisor", "from": "supervisor", "to": "done", "brief": "Não há alertas disparando; nada a fazer." },
  { "type": "answer", "at": 7, "node": "team", "role": "supervisor", "content": "Não há alertas disparando; nada a fazer." }
]
```

**Regras**:
- Todo evento com `node: "team"` tem `role`.
- Todo turno de papel é precedido por um `handoff` para esse papel.
- A execução termina com um `handoff` para `"done"`. Quando a equipe encerra normalmente, vem em seguida um `answer` com `role: "supervisor"`. Quando encerra por proposta do executor, a resposta é 202 e não tem `answer`.

## `GET /requests/:id`

O trace devolvido é idêntico ao da resposta original, inclusive `role`, `from`, `to` e `brief`.

## `GET /stats`

`byRoute` passa a ter a chave `team` quando houver execuções da equipe.
