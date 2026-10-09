# Quickstart: War Room Web

Este é o roteiro de validação de ponta a ponta. Os formatos estão em [contracts/http.md](./contracts/http.md) e o comportamento da tela, em [contracts/web-ui.md](./contracts/web-ui.md).

## Pré-requisitos

- Node 24 LTS. Rode `npm install` na raiz e em `web/`.
- Rode `npm run seed` para restaurar o dataset canônico. Ele tem incidentes abertos que podem ser resolvidos.
- Tenha as variáveis do OpenRouter no ambiente (não leia nem commite o `.env`).

## 1. Gates automáticos

```bash
npm run typecheck && npm test                    # API, incluindo src/http/web-contract.test.ts
npm --prefix web run typecheck && npm --prefix web test
npm --prefix web run build                       # gera web/dist com assets sob /opspilot/
```

Resultado esperado: tudo verde e `web/dist/index.html` referenciando `/opspilot/assets/...`.

## 2. Subir API e war room

```bash
npm run dev                                      # API em http://localhost:3000
npm --prefix web run dev                         # war room em http://localhost:5173/opspilot/
```

## 3. CORS (US5)

```bash
# origem permitida (padrão): cabeçalhos presentes, preflight 204
curl -si -X OPTIONS http://localhost:3000/chat -H 'Origin: http://localhost:5173' \
  -H 'Access-Control-Request-Method: POST' | grep -iE '^HTTP|access-control'
# origem não permitida: 204 sem Access-Control-Allow-Origin
curl -si -X OPTIONS http://localhost:3000/chat -H 'Origin: https://malicioso.example' \
  -H 'Access-Control-Request-Method: POST' | grep -iE '^HTTP|access-control'
```

Abra a war room por `http://127.0.0.1:5173/opspilot/`, que é outra origem e não está na lista. Ao enviar uma mensagem, o erro deve ser "Não foi possível falar com a API", com o atalho para as configurações.

## 4. Chat e raciocínio (US1, US2)

1. Abra `http://localhost:5173/opspilot/` e confira o estado vazio com os 3 exemplos.
2. Clique em "Quais alertas estão disparando?". Deve aparecer "Pensando… Ns" e depois a resposta.
3. Pergunte "e quais deles são do checkout?" e confira que a resposta usa o contexto anterior.
4. Clique em "Ver raciocínio". Confira a rota com o motivo, as métricas, o ID copiável e os eventos na ordem, cada um com tipo e nó. A `observation` longa deve estar recolhida.
5. Aperte `Esc`. O painel fecha e o foco volta ao botão.
6. Recarregue a página. A tela volta ao estado vazio e o endereço da API continua o mesmo.

## 5. Aprovação (US3)

1. Envie "Liste os incidentes abertos" e anote um id.
2. Envie "Resolva o incidente <id>, o rollback resolveu". Deve aparecer o cartão "Ação aguardando aprovação", e o composer fica bloqueado com a nota.
3. Envie "Liste os incidentes abertos" pelo `curl` (ou numa outra aba). O incidente **ainda está aberto**.
4. Clique em "Aprovar". O cartão vai para "Aprovada", aparece a resposta "Incidente <id> resolvido." e o "Ver raciocínio" mostra `action → observation → answer` com o nó `aprovacao`.
5. Repita com outro incidente e clique em "Negar". O cartão vai para "Negada", aparece a resposta de cancelamento, e o incidente continua aberto.
6. Decisão duplicada:
   ```bash
   curl -s -X POST http://localhost:3000/approvals/<id-já-decidido> -H 'Content-Type: application/json' \
     -d '{"decision":"approve"}'
   ```
   Deve voltar `409 approval_already_decided`.
7. Expiração: suba a API com `OPSPILOT_APPROVAL_TTL_MS=5000`, gere um cartão, espere 6s e aprove. O cartão deve ir para "Expirou" e nada deve ser executado.

## 6. Configurações (US4)

1. Clique na engrenagem e digite `localhost:3000` (sem esquema). O erro deve aparecer junto ao campo, e o endereço anterior continua valendo.
2. Digite `http://localhost:3999` e salve. Deve aparecer "Não foi possível conectar". Ao enviar uma mensagem, o erro deve oferecer "Abrir configurações".
3. Clique em "Restaurar padrão" e salve. Deve aparecer "Conectado". Recarregue a página e confira que o endereço persiste.
4. Troque o tema para "Escuro" e depois para "Seguir o sistema", alterando o tema do sistema operacional. A war room deve acompanhar.

## 7. Publicação sob `/opspilot/` (US5)

```bash
npm --prefix web run preview                     # serve web/dist em http://localhost:4173/opspilot/
```

Adicione `http://localhost:4173` em `OPSPILOT_CORS_ORIGINS`, reinicie a API, abra a war room, recarregue a página e mande uma mensagem. Não pode aparecer nenhum 404 de asset no console.

## 8. Acessibilidade e responsividade

- Rode o axe DevTools nos dois temas, com o painel de raciocínio aberto e com um cartão pendente. O resultado esperado é 0 violações AA.
- Percorra todo o fluxo da seção 5 só com o teclado.
- No DevTools, use largura de 360px. Não pode haver rolagem horizontal, e os painéis devem ocupar a tela inteira.
- Ative `prefers-reduced-motion` e confira que o skeleton fica sem animação.
