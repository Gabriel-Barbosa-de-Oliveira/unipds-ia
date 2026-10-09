# Contrato de UI: War Room Web

Este é o contrato do que a war room mostra e de como cada área se comporta. Ele segue `.github/instructions/design.instructions.md` e é verificado pelo [quickstart.md](../quickstart.md).

## Layout

```text
┌───────────────────────────────────────────────────────────┐
│ header: h1 "OpsPilot · War room"   [Nova conversa] [⚙]    │
├───────────────────────────────────────────────────────────┤
│ main (role=log, aria-live=polite)                         │
│   item user          (alinhado à direita)                 │
│   item assistant     texto + [Ver raciocínio]             │
│   item approval      cartão Aprovar / Negar               │
│   item error         mensagem + ação                      │
├───────────────────────────────────────────────────────────┤
│ composer: <label> textarea  [Enviar]  (ação primária)     │
└───────────────────────────────────────────────────────────┘
painéis laterais (dialog modal): Raciocínio | Configurações
  ≥ 768px: gaveta à direita, 480px · < 768px: tela cheia
```

- Há um único `h1`. Os painéis usam `h2` e as seções internas, `h3`.
- A ação primária é "Enviar". "Nova conversa" e a engrenagem são ações secundárias (ghost).
- O espaçamento usa só tokens `--space-1..8` = 4, 8, 12, 16, 24, 32, 48, 64 px.
- A largura máxima da conversa é 760px, centralizada. Não há rolagem horizontal a partir de 360px.

## Tokens (`web/src/styles/tokens.css`)

As cores são semânticas, definidas em `:root[data-theme="light"]` e `:root[data-theme="dark"]`. Quando o tema é `system`, `data-theme` é resolvido por `prefers-color-scheme` e acompanha mudanças do sistema.

`--color-bg`, `--color-surface`, `--color-surface-raised`, `--color-text`, `--color-text-muted`, `--color-border`, `--color-primary`, `--color-primary-text`, `--color-focus`, `--color-danger`, `--color-warning`, `--color-success`, `--color-info`, e uma cor por tipo de evento de trace (`--trace-route`, `--trace-thought`, `--trace-plan`, `--trace-action`, `--trace-observation`, `--trace-critique`, `--trace-fallback`, `--trace-answer`).

O contraste de texto é ≥ 4.5:1 e o de bordas e ícones de UI é ≥ 3:1, nos dois temas. No tema escuro o fundo não é `#000` e o texto não é `#fff`.

## Áreas

### Conversa (`main`)

| Estado | Conteúdo |
|---|---|
| vazio | `h2` "Pronto para o plantão" + texto curto + 3 exemplos clicáveis ("Quais alertas estão disparando?", "Quais incidentes estão abertos?", "Qual o runbook do checkout-api?"). Clicar preenche o composer e envia. |
| enviando | item `user` + indicador "Pensando… 12s" (skeleton com `aria-busy`). Não aparece se a resposta chegar em menos de 300ms. |
| sucesso | item `assistant`: texto em `pre-wrap` + botão "Ver raciocínio" (`aria-haspopup="dialog"`). |
| erro | item `error`: ícone + `title` + `detail` + "ID da requisição: …" (copiável, se houver) + botão da ação (`Tentar novamente` / `Abrir configurações` / `Nova conversa`). |

### Composer

- Tem `<label>` visível ("Mensagem"). `Enter` envia e `Shift+Enter` quebra linha.
- O envio fica desabilitado com o texto vazio, durante `sending` e durante `awaiting_decision`. No último caso aparece a nota: "Decida a ação pendente antes de enviar outra mensagem".
- O texto digitado é preservado quando o envio falha ou está bloqueado.

### Cartão de aprovação

```text
┌ ⚠ Ação aguardando aprovação ─────────────────── expira 12:15 ┐
│ Resolver o incidente INC-42                                   │
│ Motivo: O alerta de checkout voltou ao normal…                │
│ ▸ Detalhes (resolve_incident · id, summary)                   │
│                                  [Negar]   [Aprovar]          │
└───────────────────────────────────────────────────────────────┘
```

- `role="group"` com `aria-labelledby` apontando para o título. A chegada do cartão é anunciada pela região `aria-live`.
- O estado é comunicado por ícone, texto e cor, nunca só por cor: pendente (⚠ "Aguardando"), enviando (spinner, "Enviando decisão…"), aprovada (✓ "Aprovada"), negada (✕ "Negada"), indisponível ("Já decidida" / "Expirou").
- "Aprovar" é a ação primária do cartão e "Negar" é secundária. Os dois têm 44×44px no mínimo.
- Os botões ficam desabilitados fora do estado `pending`. Em caso de erro de rede, o cartão volta a `pending` com a mensagem inline.
- Há um botão "Ver raciocínio" para o trace da execução que gerou o 202.

### Painel de raciocínio (dialog)

- Título `h2` "Raciocínio". O cabeçalho mostra a rota (badge + motivo + origem), as métricas (chamadas ao modelo, tempo, modelo, tokens) e o ID da requisição com o botão "Copiar". Rota e métricas são omitidas quando `null`.
- O corpo é uma lista ordenada (`<ol>`) de eventos. Cada evento mostra o ícone e o rótulo do tipo, o badge do nó (`contexto`, `roteador`, `react`, `planExecute`, `reflect`, `resposta`, `aprovacao`) e o conteúdo conforme o [data-model.md](../data-model.md#traceevent-espelho-tipado-de-srcagentstypests).
- Conteúdo longo (`args`, `result` acima de ~600 caracteres) fica em `<details>`.
- Com o trace vazio, aparece um estado vazio: "Essa execução não registrou eventos."
- O foco vai para o título ao abrir e fica preso no painel. `Esc` fecha e o foco volta ao botão de origem.

### Painel de configurações (dialog)

- Título `h2` "Configurações". O campo "Endereço da API" (`type="url"`) tem um texto de ajuda e o erro inline via `aria-describedby`.
- Botões: "Salvar" (primário), "Restaurar padrão" e "Cancelar".
- Depois de salvar, aparece o status da conexão: "Testando…" → "✓ Conectado" ou "✕ Não foi possível conectar". O endereço fica salvo mesmo que o teste falhe.
- Tema: grupo de rádio "Seguir o sistema / Claro / Escuro".

## Teclado e leitor de tela

- A ordem de tabulação é header → conversa (botões dos itens) → composer. `:focus-visible` usa `--color-focus` com 2px de contorno e 2px de offset.
- Todos os botões só com ícone têm `aria-label` (por exemplo, a engrenagem: "Configurações").
- Com `prefers-reduced-motion: reduce`, transições e o shimmer do skeleton são removidos.
