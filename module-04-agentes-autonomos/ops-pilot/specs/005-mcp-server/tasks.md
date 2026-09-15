---

description: "Task list template for feature implementation"
---

# Tasks: MCP Server para OpsPilot

**Input**: Design documents from `/specs/005-mcp-server/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/mcp-tools.md, quickstart.md

**Tests**: incluídos — Constitution V (Teste Obrigatório, NON-NEGOTIABLE) exige teste para toda
lógica nova, e FR-012/SC-004 exigem explicitamente um teste automatizado sobre o servidor real.

**Organization**: tarefas agrupadas por user story (spec.md), na ordem de prioridade P1 → P2 → P3.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivos diferentes, sem dependência entre si)
- **[Story]**: a qual user story a tarefa pertence (US1, US2, US3)
- Caminhos de arquivo são sempre absolutos-relativos ao repo (`src/...`)

## Path Conventions

Projeto único (ver plan.md → Project Structure): `src/mcp/server.ts` (novo), ajuste mínimo em
`src/agents/tools.ts`, testes colocados junto ao código (`*.test.ts`), sem diretório `tests/`.

---

## Phase 1: Setup

**Purpose**: preparar dependência e script npm do servidor MCP, sem ainda tocar em código.

- [X] T001 Adicionar `@modelcontextprotocol/sdk` como `dependency` (não `devDependency` — é usada
      em runtime pelo entrypoint) em `package.json` e rodar `npm install`.
- [X] T002 Adicionar o script `"mcp": "node --env-file-if-exists=.env --import tsx src/mcp/server.ts"`
      em `package.json`, no mesmo padrão já usado por `dev`/`arena`/`bench`/`seed` — não o literal
      `"tsx src/mcp/server.ts"` do pedido original, para garantir que `OPSPILOT_DB` seja carregado
      antes de `SqliteOpsStore` ler `process.env.OPSPILOT_DB` (ver research.md §6).

**Checkpoint**: dependência instalada, script `npm run mcp` existe (ainda sem `src/mcp/server.ts`).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: extrair a fonte única de verdade (schemas zod + tradução de erro) que as 3 tools do
servidor MCP vão reaproveitar, e criar o esqueleto do servidor.

**⚠️ CRITICAL**: nenhuma user story pode começar antes desta fase estar completa.

- [X] T003 [P] Em `src/agents/tools.ts`, extrair os três schemas zod hoje definidos inline nos
      `tool(...)` de `list_alerts`, `open_incident` e `resolve_incident` para constantes
      exportadas em formato de **shape** (não `z.object(...)`), preservando exatamente os mesmos
      campos, tipos e `.describe(...)`:
      `export const listAlertsShape = { status: z.enum(["firing", "resolved"]).optional().describe(...) };`
      `export const openIncidentShape = { title: z.string().min(1).describe(...), service: z.string().min(1).describe(...), severity: z.enum(["low","medium","high","critical"]).describe(...) };`
      `export const resolveIncidentShape = { id: z.string().min(1).describe(...), summary: z.string().min(1).optional().describe(...) };`
      Atualizar os três `tool(...)` correspondentes para usar `schema: z.object(listAlertsShape)`
      (idem para os outros dois) em vez do objeto inline — sem mudança de comportamento (ver
      research.md §2).
- [X] T004 Em `src/agents/tools.ts`, exportar a função `toStructuredError` (hoje privada do
      módulo) para que `src/mcp/server.ts` possa reaproveitá-la sem duplicar o mapeamento de
      erros de domínio (ver research.md §4). Depende de T003 (mesmo arquivo).
- [X] T005 Rodar `npm test` e `npm run typecheck` e confirmar que `src/agents/tools.test.ts` e o
      resto da suíte continuam passando sem nenhuma mudança de comportamento após T003/T004
      (refactor puro). Depende de T003, T004.
- [X] T006 [P] Criar o esqueleto de `src/mcp/server.ts`: função exportada
      `createMcpServer(store: OpsStoreRepository): McpServer` que instancia
      `new McpServer({ name: "opspilot", version: "0.1.0" })` (ainda sem nenhuma tool registrada);
      no topo do arquivo, bloco de execução que compõe `createMcpServer(new SqliteOpsStore())` e
      faz `await server.connect(new StdioServerTransport())`, seguido de
      `console.error("OpsPilot MCP server pronto (stdio)")` — **nunca** `console.log`/`console.info`
      neste arquivo nem em nenhum caminho que ele exercite (FR-008/REGRA CRÍTICA). Pode rodar em
      paralelo a T003–T005 (arquivo diferente).

**Checkpoint**: `npm run mcp` sobe um servidor `opspilot` válido via stdio, ainda sem tools —
fundação pronta para as user stories.

---

## Phase 3: User Story 1 - Consultar alertas via cliente MCP (Priority: P1) 🎯 MVP

**Goal**: um cliente MCP consegue listar os alertas (com e sem filtro de status) via `list_alerts`.

**Independent Test**: subir o servidor real, chamar `list_alerts` com e sem `status`, e comparar
o resultado com o mesmo dataset semeado que `src/agents/tools.test.ts` já usa.

### Implementation for User Story 1

- [X] T007 [US1] Em `src/mcp/server.ts`, registrar a tool `list_alerts` dentro de
      `createMcpServer`, importando `listAlertsShape` de `../agents/tools.ts`:
      `server.registerTool("list_alerts", { description: "<mesma descrição de tools.ts>", inputSchema: listAlertsShape }, async ({ status }) => ({ content: [{ type: "text", text: JSON.stringify(await store.listAlerts(status)) }] }))`.
- [X] T008 [US1] Criar `src/mcp/server.test.ts` com a infraestrutura de teste compartilhada pelas
      3 user stories: um helper que cria um arquivo SQLite temporário, semeia com
      `new SqliteOpsStore(tmpPath)` + `seedCanonicalScenario` (de `../store/sqlite-ops-store.ts`),
      spawna o processo real do servidor via `StdioClientTransport`
      (`@modelcontextprotocol/sdk/client/stdio.js`, `command: "node"`,
      `args: ["--import", "tsx", "src/mcp/server.ts"]`, `env: { ...process.env, OPSPILOT_DB: tmpPath }`,
      `cwd:` raiz do repo) e conecta um `Client`
      (`@modelcontextprotocol/sdk/client/index.js`); e um cleanup que fecha a conexão e apaga o
      arquivo temporário (e eventuais `-journal`/`-wal`) ao final de cada teste. Primeiro teste:
      `client.listTools()` inclui uma tool `list_alerts` cujo `inputSchema` declara `status` como
      enum opcional `["firing","resolved"]` (usar `.find`, não comparar tamanho da lista ainda —
      só a US3 fecha a asserção de "exatamente 3 tools").
- [X] T009 [US1] Em `src/mcp/server.test.ts`, adicionar testes chamando
      `client.callTool({ name: "list_alerts", arguments: {} })` (espera os 6 alertas semeados) e
      `client.callTool({ name: "list_alerts", arguments: { status: "firing" } })` (espera os 3
      alertas com status `firing`) — mesmos números já validados em
      `src/agents/tools.test.ts` para o mesmo dataset canônico (SC-002).

**Checkpoint**: US1 funcional de ponta a ponta — um cliente MCP já consegue listar alertas.

---

## Phase 4: User Story 2 - Abrir um incidente via cliente MCP (Priority: P2)

**Goal**: um cliente MCP registra um novo incidente (`open_incident`) e recebe o incidente criado.

**Independent Test**: com o servidor rodando, chamar `open_incident` com dados válidos e conferir
que o incidente criado é retornado com `status: "open"`; chamar com serviço inexistente e conferir
o erro estruturado.

### Implementation for User Story 2

- [X] T010 [US2] Em `src/mcp/server.ts`, registrar a tool `open_incident` dentro de
      `createMcpServer`, importando `openIncidentShape` e `toStructuredError` de
      `../agents/tools.ts`, espelhando o corpo de `openIncidentTool` em `tools.ts`: `try` chama
      `store.openIncident({ title, service, severity })` e retorna
      `{ content: [{ type: "text", text: JSON.stringify(incident) }] }`; `catch` retorna
      `{ content: [{ type: "text", text: JSON.stringify(toStructuredError(error, { service })) }], isError: true }`.
- [X] T011 [US2] Em `src/mcp/server.test.ts`, adicionar testes: (a) `client.listTools()` inclui
      `open_incident` com `title`/`service`/`severity` obrigatórios (`severity` enum
      `["low","medium","high","critical"]`); (b) `callTool open_incident` com um serviço semeado
      existente retorna um incidente com `status: "open"` e `resolvedAt: null`; (c) `callTool
      open_incident` com um serviço inexistente retorna `isError: true` e conteúdo
      `{ error: "ServiceNotFoundError", service }`.

**Checkpoint**: US1 e US2 funcionam de forma independente via MCP.

---

## Phase 5: User Story 3 - Resolver um incidente via cliente MCP (Priority: P3)

**Goal**: um cliente MCP resolve (`resolve_incident`) um incidente já aberto, com resumo opcional.

**Independent Test**: abrir um incidente (US2) e resolvê-lo pelo id retornado; conferir também o
erro estruturado para um id inexistente.

### Implementation for User Story 3

- [X] T012 [US3] Em `src/mcp/server.ts`, registrar a tool `resolve_incident` dentro de
      `createMcpServer`, importando `resolveIncidentShape` e reaproveitando `toStructuredError`,
      espelhando o corpo de `resolveIncidentTool` em `tools.ts`: `try` chama
      `store.resolveIncident(id, summary)` e retorna o incidente atualizado como texto JSON;
      `catch` retorna `isError: true` com `JSON.stringify(toStructuredError(error, { id }))`.
- [X] T013 [US3] Em `src/mcp/server.test.ts`, adicionar testes: (a) `client.listTools()` inclui
      `resolve_incident` com `id` obrigatório e `summary` opcional; (b) abrir um incidente via
      `open_incident` e resolvê-lo via `resolve_incident` com um `summary`, conferindo
      `status: "resolved"` e o `summary` salvo; (c) `callTool resolve_incident` com um `id`
      inexistente retorna `isError: true` e `{ error: "IncidentNotFoundError", id }`; (d)
      **asserção final de FR-012/SC-004**: `client.listTools()` retorna **exatamente** 3 tools,
      com os nomes `list_alerts`, `open_incident` e `resolve_incident` — nem mais, nem menos.

**Checkpoint**: as 3 user stories funcionam de ponta a ponta; FR-012/SC-004 cobertos pelo teste
automatizado que sobe o processo real.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: validação final de ponta a ponta, sem lógica nova.

- [X] T014 Rodar manualmente `OPSPILOT_DB=":memory:" npm run mcp`, confirmar que nada é impresso
      em stdout (só a linha de prontidão em stderr), e encerrar com Ctrl+C — valida FR-008
      manualmente, seguindo `quickstart.md` passo 2.
- [X] T015 Rodar `npm test` e `npm run typecheck` uma última vez, confirmando toda a suíte
      (incluindo `src/mcp/server.test.ts` e `src/agents/tools.test.ts`) verde — gate de conclusão
      da feature (Constitution V).

**Checkpoint**: feature completa, testada e alinhada com spec.md/plan.md/quickstart.md.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: sem dependências — pode começar imediatamente.
- **Foundational (Phase 2)**: depende de Setup — BLOQUEIA todas as user stories.
- **User Stories (Phase 3-5)**: todas dependem de Foundational; entre si, US2 reaproveita a
  infraestrutura de teste criada em US1 (T008) e US3 reaproveita `open_incident` (US2) dentro do
  próprio teste de resolução — por isso, apesar de cada uma ser independentemente demonstrável,
  a ordem de implementação recomendada é sequencial (P1 → P2 → P3), não paralela entre stories.
- **Polish (Phase 6)**: depende de todas as user stories desejadas estarem completas.

### Dentro de cada User Story

- Registro da tool em `src/mcp/server.ts` antes dos testes que a exercitam (mesmo arquivo de
  teste, `src/mcp/server.test.ts`, editado incrementalmente a cada fase — não há tarefas `[P]`
  dentro das fases 3-5, pois todas tocam os mesmos dois arquivos).

### Parallel Opportunities

- T003 (schemas em `tools.ts`) e T006 (esqueleto de `server.ts`) podem rodar em paralelo —
  arquivos diferentes.
- Nenhuma outra tarefa é paralelizável: Foundational tem uma cadeia de dependência em `tools.ts`
  (T003 → T004 → T005), e as fases 3-5 editam sequencialmente os mesmos dois arquivos
  (`src/mcp/server.ts` e `src/mcp/server.test.ts`).

---

## Parallel Example: Foundational

```bash
# T003 e T006 podem ser feitos ao mesmo tempo (arquivos diferentes):
Task: "Extrair schemas zod para constantes exportadas em src/agents/tools.ts"
Task: "Criar esqueleto de createMcpServer em src/mcp/server.ts"
```

---

## Implementation Strategy

### MVP First (User Story 1 apenas)

1. Completar Phase 1 (Setup) e Phase 2 (Foundational).
2. Completar Phase 3 (US1 — `list_alerts`).
3. **PARAR e VALIDAR**: rodar `src/mcp/server.test.ts`, confirmar `list_alerts` funcionando via
   MCP de ponta a ponta.
4. Esse já é um MVP demonstrável: um cliente MCP consegue consultar alertas do OpsPilot.

### Incremental Delivery

1. Setup + Foundational → fundação pronta (servidor `opspilot` sobe, sem tools).
2. US1 (`list_alerts`) → MVP, testável isoladamente.
3. US2 (`open_incident`) → adiciona escrita, testável isoladamente (não quebra US1).
4. US3 (`resolve_incident`) → fecha o ciclo de vida do incidente; fase final do teste valida a
   lista completa de 3 tools (FR-012).
5. Polish → validação manual da regra crítica de stdout + gate final de `npm test`/`typecheck`.

## Notes

- Todas as tarefas de código tocam apenas 2 arquivos novos/ajustados
  (`src/mcp/server.ts`, `src/mcp/server.test.ts`) e 1 arquivo existente ajustado minimamente
  (`src/agents/tools.ts`) — consistente com Constitution VIII (Pequeno e Reversível).
- Nenhuma lógica de negócio nova é escrita: toda validação e toda regra de domínio já existem em
  `src/agents/tools.ts`/`src/services/ops-store.repository.ts`/`src/store/sqlite-ops-store.ts` e
  são apenas reaproveitadas (Constitution I, II, III, IV).
- Commit sugerido por tarefa ou por pequeno grupo de tarefas relacionadas (ex.: T001+T002 juntos,
  T003+T004+T005 juntos, um commit por user story).
