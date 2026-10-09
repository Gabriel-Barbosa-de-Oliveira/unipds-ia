# Quickstart: Modo Equipe

Contratos: [contracts/http.md](./contracts/http.md) e [contracts/web-ui.md](./contracts/web-ui.md).

## 1. Gates (sem rede)

```bash
npm run typecheck && npm test                              # API: blackboard, supervisor, papéis, loop, migração, logs, /chat team
npm --prefix web run typecheck && npm --prefix web test    # war room: schema e trace-view do handoff
```

## 2. Migração do banco existente

```bash
cp data/opspilot.db /tmp/opspilot-antes.db                 # backup
npm run dev                                                # a primeira gravação migra o CHECK de requests.route
```

Mande qualquer pergunta com `"strategy":"team"`. Em seguida, `GET /requests/<id>` deve devolver `route: "team"`, sem `persistence.failed` no log. Os registros antigos continuam consultáveis.

## 3. Equipe ponta a ponta (exige OpenRouter no ambiente e `npm run seed`)

```bash
curl -s localhost:3000/chat -H 'Content-Type: application/json' \
  -d '{"message":"Quais alertas estão disparando no checkout-api?","strategy":"team"}' | jq '.route, [.trace[] | {type, role, to}]'
```

Resultado esperado:
- `route.route` é `"team"`;
- a sequência começa com `handoff → analista`, seguido das ações de leitura do analista;
- termina com `handoff → fim` e `answer` com `role: "supervisor"`;
- não aparece planejador nem executor (pedido só de consulta).

```bash
curl -s localhost:3000/chat -H 'Content-Type: application/json' \
  -d '{"message":"O checkout está lento: investigue e abra um incidente se for o caso","strategy":"team"}' -o /tmp/team.json -w '%{http_code}\n'
```

Resultado esperado: **202**, com `approval.tool: "open_incident"`. O trace mostra analista → planejador → executor, e o incidente **não** foi criado. Aprovar pela war room ou com `POST /approvals/:id` cria o incidente.

## 4. Limites dos papéis

Nos testes (`src/team/*.test.ts`), conferir que:
- as ferramentas do analista são exatamente as 3 de leitura, as do planejador são nenhuma e as do executor são exatamente `open_incident` e `resolve_incident`, nas instâncias com porta;
- montar a equipe com `createOpsTools(store)`, que não tem porta, lança `TeamToolsNotGatedError`.

## 5. War room

Abra "ver raciocínio" numa resposta da equipe. Confira que cada turno é precedido por uma "Passagem" (`Supervisor → Analista` + a instrução), que cada evento mostra o badge do papel, e que isso vale nos dois temas e com leitor de tela, porque o rótulo é texto.

## 6. Logs

Na saída do `npm run dev`, as linhas `team.handoff` devem ter `from`, `to` e `position`, e **nunca** o texto do brief.
