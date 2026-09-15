# Implementation Plan: MCP Server para OpsPilot

**Branch**: `005-mcp-server` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-mcp-server/spec.md`

## Summary

Expor um servidor MCP (`opspilot`) via transporte stdio, em `src/mcp/server.ts`, com exatamente
três tools (`list_alerts`, `open_incident`, `resolve_incident`) que reaproveitam, sem duplicação,
os mesmos schemas zod e o mesmo `OpsStoreRepository`/`SqliteOpsStore` já usados pelas tools do
agente de chat (`src/agents/tools.ts`). O único ajuste estrutural necessário no código existente é
extrair os schemas zod hoje definidos inline em `tools.ts` para constantes exportadas, para que
`src/mcp/server.ts` os importe em vez de redeclará-los — essa é a "única fonte de verdade" pedida.
O servidor NUNCA escreve em stdout (canal do protocolo MCP); todo diagnóstico vai para stderr. Um
teste automatizado sobe o processo real via `StdioClientTransport` do `@modelcontextprotocol/sdk`
e valida a lista de tools exposta.

## Technical Context

**Language/Version**: TypeScript ESM `strict`, Node 24 LTS (mesmo runtime do restante do projeto)

**Primary Dependencies**: `@modelcontextprotocol/sdk` (novo, para `McpServer`/`StdioServerTransport`/
`StdioClientTransport`/`Client`); `zod` (já existente — mesmos schemas reaproveitados de
`src/agents/tools.ts`); `@langchain/core` NÃO é dependência do servidor MCP (as tools MCP não
passam pelo wrapper `tool()` do LangChain, só reaproveitam os schemas zod e o `OpsStoreRepository`)

**Storage**: SQLite via `node:sqlite` (`DatabaseSync`), através do `SqliteOpsStore` já existente
(`src/store/sqlite-ops-store.ts`), lido do mesmo `OPSPILOT_DB`/`./data/opspilot.db` padrão — nenhum
mecanismo de persistência novo

**Testing**: `node:test` via `tsx` (`src/mcp/server.test.ts`), usando `StdioClientTransport` do SDK
para spawnar o processo real (`node --env-file-if-exists=.env --import tsx src/mcp/server.ts`, com
`OPSPILOT_DB=":memory:"`) e um `Client` MCP chamando `listTools()` — exercita o binário real e o
canal stdio de ponta a ponta, não só uma composição em memória

**Target Platform**: processo local de linha de comando (Node 24), spawnado por um cliente MCP
(ex.: Claude Desktop, outro agente) via stdio — sem HTTP, sem rede

**Project Type**: single project — novo módulo dentro do repositório existente do OpsPilot

**Performance Goals**: N/A (não é serviço de alta concorrência; um processo por cliente MCP
conectado, latência dominada pelas mesmas operações de SQLite já usadas pelo chat)

**Constraints**: zero bytes em stdout fora de frames válidos do protocolo MCP (FR-008); schemas de
validação idênticos aos já usados pelo agente de chat, sem duplicação (FR-003); nenhuma tool além
das três pedidas (`list_incidents`/`consultar_runbook` ficam de fora, conforme Assumptions do spec)

**Scale/Scope**: 3 tools; 1 arquivo de entrypoint novo (`src/mcp/server.ts`) + 1 teste + um ajuste
mínimo de exports em `src/agents/tools.ts`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Avaliação |
|---|---|
| I. Camadas Explícitas | Pass — `src/mcp/server.ts` é uma nova borda de transporte (equivalente a `src/http/server.ts`), que só chama o `OpsStoreRepository` já existente; nenhuma regra de negócio nova é escrita no domínio. |
| II. Validação na Fronteira | Pass — o MCP é uma fronteira externa (cliente fora do processo); a validação continua sendo feita pelos mesmos schemas zod, agora exportados e reaproveitados em vez de duplicados. |
| III. Erros de Domínio | Pass — os mesmos erros de domínio (`ServiceNotFoundError`, `IncidentNotFoundError`, `InvalidSeverityError`) são capturados e traduzidos na borda do servidor MCP (`isError: true` + conteúdo estruturado), nunca dentro do domínio. |
| IV. Funções Puras | Pass — o registro das tools é uma função de fábrica pura (`createMcpServer(store)`, nos moldes de `createOpsTools(store)`); o único efeito colateral (stdio, `process`) fica isolado no bloco de entrypoint do arquivo. |
| V. Teste Obrigatório | Pass — FR-012 exige teste automatizado (`src/mcp/server.test.ts`, coberto pelo glob `src/**/*.test.ts` já usado por `npm test`); `npm run typecheck` continua obrigatório. |
| VI. Segurança por Padrão | Pass — nenhum segredo novo, nenhuma leitura direta de `.env` pelo código (o carregamento continua via `--env-file-if-exists=.env` no script npm, como os demais); nenhuma ação destrutiva nova além das já existentes (abrir/resolver incidente já é possível hoje via chat). |
| VII. Spec Antes de Código | Pass — este plano é gerado a partir de `spec.md` já revisado, seguindo `/speckit.specify` → `/speckit.plan` → `/speckit.tasks` → `/speckit.implement`. |
| VIII. Pequeno e Reversível | Pass — escopo cabe em poucos commits pequenos (extrair schemas; criar `server.ts`; criar teste; script npm) — detalhado em `/speckit.tasks`. |

Nenhuma violação → tabela de Complexity Tracking não se aplica.

## Project Structure

### Documentation (this feature)

```text
specs/005-mcp-server/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   └── mcp-tools.md
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── agents/
│   ├── tools.ts             # ajuste: extrai os 3 schemas zod reaproveitados para constantes exportadas
│   └── tools.test.ts        # inalterado
├── mcp/
│   ├── server.ts            # NOVO: McpServer + StdioServerTransport, registra list_alerts/open_incident/resolve_incident
│   └── server.test.ts       # NOVO: spawna o processo real (StdioClientTransport) e valida tools/list
├── domain/                  # inalterado (Alert, Incident, erros de domínio)
├── services/                # inalterado (OpsStoreRepository)
└── store/                   # inalterado (SqliteOpsStore)
```

**Structure Decision**: projeto único (sem frontend/backend separados). A feature adiciona um
módulo de transporte novo (`src/mcp/`), no mesmo nível de `src/http/`, e reaproveita as camadas
`domain/`/`services/`/`store/` já existentes sem alteração de contrato. Os testes ficam colocados
junto ao código (`*.test.ts`), seguindo a convenção já usada em todo o projeto — não é criado um
diretório `tests/` separado.

## Complexity Tracking

*Sem violações de constitution — tabela não aplicável.*
