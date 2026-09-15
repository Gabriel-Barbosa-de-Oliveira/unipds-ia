# Quickstart: validar o servidor MCP do OpsPilot

Pré-requisitos: feature implementada (`/speckit.implement`), dependências instaladas
(`npm install`, incluindo `@modelcontextprotocol/sdk` — ver [research.md](./research.md)).

## 1. Rodar a suíte automatizada (cobre FR-012 / SC-004)

```bash
npm test
```

O teste `src/mcp/server.test.ts` sobe o processo real (`npm run mcp` equivalente, com
`OPSPILOT_DB=":memory:"`), conecta um `Client` MCP via `StdioClientTransport` e valida que
`client.listTools()` retorna exatamente `list_alerts`, `open_incident` e `resolve_incident` — ver
[contracts/mcp-tools.md](./contracts/mcp-tools.md). Qualquer escrita indevida no stdout (ex.: um
`console.log` esquecido) quebra o handshake e faz esse teste falhar.

```bash
npm run typecheck
```

## 2. Rodar o servidor manualmente

```bash
OPSPILOT_DB=":memory:" npm run --silent mcp
```

O processo do **servidor** (`src/mcp/server.ts`) nunca escreve em stdout — só a linha de
prontidão em stderr. **Atenção**: `npm run mcp` **sem** `--silent` faz o próprio `npm` imprimir um
banner (`> ops-pilot@0.1.0 mcp\n> node ...`) em stdout antes de executar o script — isso não vem
do código do servidor, mas quebraria o handshake de qualquer cliente MCP real apontado para esse
comando. Por isso, qualquer cliente MCP (passo 3) deve ser configurado para rodar o comando
`node` diretamente (ou `npm run --silent mcp`), nunca `npm run mcp` puro. Use `Ctrl+C` para
encerrar.

## 3. Validar manualmente com um cliente MCP (opcional)

Qualquer cliente MCP compatível com stdio pode ser apontado para o comando de inicialização do
servidor — **`node --env-file-if-exists=.env --import tsx src/mcp/server.ts`** (com `OPSPILOT_DB`
configurado para o banco desejado), e não `npm run mcp` puro, pelo motivo explicado no passo 2.
Passos para validar os três cenários de `spec.md`:

1. **US1 — listar alertas**: chamar a tool `list_alerts` (sem argumentos e depois com
   `{ "status": "firing" }`) e comparar o resultado com o mesmo dado hoje pelo chat do OpsPilot
   para o mesmo banco (`SC-002`).
2. **US2 — abrir incidente**: chamar `open_incident` com `title`, `service` (de um serviço
   existente) e `severity`; confirmar que o incidente retornado aparece depois via
   `list_incidents` no chat, ou consultando o mesmo `OPSPILOT_DB` (`SC-003`). Repetir com um
   `service` inexistente e confirmar o erro estruturado `ServiceNotFoundError`.
3. **US3 — resolver incidente**: chamar `resolve_incident` com o `id` retornado no passo
   anterior e um `summary` opcional; confirmar que o incidente aparece como resolvido nas demais
   interfaces do OpsPilot. Repetir com um `id` inexistente e confirmar o erro estruturado
   `IncidentNotFoundError`.

## Critério de pronto

- `npm test` e `npm run typecheck` verdes.
- Os 3 cenários manuais acima (ou seus equivalentes automatizados, se adicionados em
  `/speckit.tasks`) produzem os resultados descritos nas Acceptance Scenarios de `spec.md`.
- Nenhum byte fora do protocolo aparece no stdout do processo em nenhum dos passos acima.
