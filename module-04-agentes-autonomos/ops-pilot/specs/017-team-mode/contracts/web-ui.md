# Contrato de UI: passagem no "ver raciocínio"

Este contrato estende o [`../015-war-room-web/contracts/web-ui.md`](../../015-war-room-web/contracts/web-ui.md#painel-de-raciocínio-dialog).

## Evento `handoff`

```text
┌▌ ⇄ Passagem   [team] [Supervisor]                       #2 ┐
│  Supervisor → Analista                                      │
│  "Levante alertas e incidentes do checkout-api"             │
└─────────────────────────────────────────────────────────────┘
```

- Rótulo "Passagem", com o ícone `handoff` (`aria-hidden`) e a cor `--trace-handoff` na borda e no rótulo. A cor nunca é o único sinal.
- O corpo mostra `<origem> → <destino>` em destaque, seguido da instrução. Com destino `fim`, o texto é "Supervisor → Fim".
- A instrução aparece como texto puro (`pre-wrap`), nunca como HTML.

## Papel em todos os eventos

- Eventos com `role` ganham um badge com o papel em PT ("Analista", "Planejador", "Executor", "Supervisor"), ao lado do badge do nó.
- Eventos sem `role` (todas as outras rotas) não mudam.

## Tokens

`--trace-handoff` nos temas claro e escuro, com contraste de texto ≥ 4.5:1 sobre todas as superfícies (o script de contraste usado na 015 roda de novo).
