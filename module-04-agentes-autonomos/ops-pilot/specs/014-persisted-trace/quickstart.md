# Quickstart: validar o trace persistido e os logs JSON

## Pré-requisitos

- `npm install` já rodado.
- Para os passos manuais: `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` no ambiente do processo. O agente nunca lê o `.env`. O `OPSPILOT_DB` é opcional (padrão `./data/opspilot.db`).

## 1. Gates automáticos (sem rede)

```bash
npm run typecheck
npm test
```

O que as suítes novas cobrem:

- `src/domain/request-record.test.ts`: montagem do registro para `ok`, `timeout` e `error`; `toStoredTraceEvents` e `restoreTrace` fazem o caminho de ida e volta e ordenam por `position`.
- `src/store/sqlite-request-store.test.ts`: em `:memory:`, `save` + `find` devolvem o mesmo registro e o mesmo trace; um id inexistente devolve `undefined`; o DDL é idempotente; os registros sobrevivem a uma nova instância sobre o mesmo arquivo temporário (SC-002).
- `src/obs/logger.test.ts`: uma linha por evento, o `JSON.parse` funciona, e `traceToLogEvents` não inclui `reason`, `args` nem `content`.
- `src/http/server.test.ts`:
  - `X-Request-Id` igual a `body.requestId` em 200, 400, 422, 504 e 500;
  - dois pedidos produzem ids diferentes;
  - `GET /requests/:id` devolve um trace deep-equal ao da resposta original;
  - 404 para id inexistente ou malformado;
  - timeout é persistido como `outcome: "timeout"`;
  - uma falha de gravação não altera a resposta e gera `persistence.failed`;
  - teste-âncora com `MARCADOR-SECRETO-123` ausente de todos os logs.

## 2. Ponta a ponta (manual)

```bash
npm run dev > /tmp/opspilot.log &
curl -si localhost:3000/chat -H 'content-type: application/json' \
  -d '{"message":"quais alertas estão firing?"}' | tee /tmp/resp.txt | grep -i x-request-id
ID=$(grep -i x-request-id /tmp/resp.txt | awk '{print $2}' | tr -d '\r')
curl -s localhost:3000/requests/$ID | jq '.request.outcome, (.trace | length)'
grep "\"requestId\":\"$ID\"" /tmp/opspilot.log | jq -c '{event, node, route, tool}'
```

Esperado:

- o cabeçalho `X-Request-Id` está presente;
- a consulta devolve `"ok"` e o mesmo número de eventos da resposta;
- cada linha de log é um JSON válido com o mesmo `requestId`;
- nenhuma linha contém o texto da pergunta ou da resposta.

## 3. Persistência entre reinícios

Reinicie a API (`npm run dev`) e repita o `curl …/requests/$ID`. Esperado: o mesmo registro.

## 4. Não encontrado

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:3000/requests/nao-existe
```

Esperado: `404`.
