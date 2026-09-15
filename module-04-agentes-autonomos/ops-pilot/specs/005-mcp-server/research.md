# Phase 0 Research: MCP Server para OpsPilot

Nenhum `NEEDS CLARIFICATION` restou no Technical Context do plano — todas as decisões abaixo
foram fechadas consultando a documentação oficial do `@modelcontextprotocol/typescript-sdk`
(via context7, versão estável `v1.29.0`, o pacote publicado como `@modelcontextprotocol/sdk`) e o
código já existente do projeto.

## 1. Como montar o servidor e o transporte stdio

- **Decision**: usar a API de alto nível `McpServer` de `@modelcontextprotocol/sdk/server/mcp.js`
  junto com `StdioServerTransport` de `@modelcontextprotocol/sdk/server/stdio.js`:
  ```ts
  const server = new McpServer({ name: "opspilot", version: "<pkg version>" });
  // registrar tools...
  const transport = new StdioServerTransport();
  await server.connect(transport);
  ```
- **Rationale**: é o padrão canônico documentado pelo próprio SDK para "local integrations where
  the client spawns the server as a child process" — exatamente o caso de uso pedido (transporte
  stdio, nome do server "opspilot").
- **Alternatives considered**: API de baixo nível (`Server` + handlers manuais de
  `ListToolsRequestSchema`/`CallToolRequestSchema`) — rejeitada por exigir reimplementar validação
  e serialização que `McpServer.registerTool` já resolve, aumentando superfície de bug sem
  necessidade (o projeto não precisa de nenhum recurso fora do que a API de alto nível oferece).

## 2. Como reaproveitar exatamente os mesmos schemas zod das tools existentes

- **Decision**: extrair, dentro de `src/agents/tools.ts`, os três schemas hoje definidos inline
  (`listAlertsTool`, `openIncidentTool`, `resolveIncidentTool`) para constantes exportadas (ex.:
  `listAlertsSchema`, `openIncidentSchema`, `resolveIncidentSchema`), como **shapes** zod (o mesmo
  formato de objeto já usado, ex. `{ status: z.enum([...]).optional()... }`), e importar essas
  mesmas constantes tanto na construção das tools do LangChain (`tool(fn, { schema: z.object(shape) })`)
  quanto no `registerTool` do MCP (`{ inputSchema: shape }`).
- **Rationale**: a API estável do SDK (`v1.29.0`) espera em `inputSchema` um **`ZodRawShape`**
  (um objeto simples de campos zod, ex. `{ a: z.number() }`), não uma instância de `z.object(...)`
  — todos os exemplos da documentação da v1.29.0 (`calculate`, `calculate-bmi`, etc.) usam esse
  formato. Como as tools do LangChain (`@langchain/core`'s `tool()`) esperam um `ZodObject` em
  `schema`, o denominador comum que evita duplicar a definição é guardar o **shape** como fonte
  única e derivar `z.object(shape)` no lado do LangChain. Isso cumpre literalmente o pedido do
  usuário ("os mesmos schemas zod das tools existentes — uma única fonte de verdade") sem duas
  definições divergentes.
- **Alternatives considered**: manter os schemas inline em `tools.ts` e reescrever schemas
  equivalentes em `src/mcp/server.ts` — rejeitada explicitamente pelo pedido do usuário (FR-003) e
  pelo risco real de as duas definições divergirem ao longo do tempo (ex.: alguém adiciona uma
  severidade nova só de um lado).

## 3. Como reaproveitar o mesmo armazenamento (OpsStore)

- **Decision**: `src/mcp/server.ts` expõe uma função de fábrica pura,
  `createMcpServer(store: OpsStoreRepository)`, nos mesmos moldes de `createOpsTools(store)` em
  `tools.ts`; o entrypoint do processo compõe `createMcpServer(new SqliteOpsStore())` (mesma
  composição de produção já usada por `opsTools`) e conecta o `StdioServerTransport`.
- **Rationale**: mantém a Camada Explícita (Constitution I) e a Função Pura (Constitution IV) —
  a fábrica é testável isoladamente com qualquer `OpsStoreRepository` (ex.: um store em memória
  seedado), e o único efeito colateral (conectar ao stdio real) fica isolado no bloco de execução
  do arquivo.
- **Alternatives considered**: instanciar `SqliteOpsStore` diretamente dentro da função de
  registro das tools — rejeitada por acoplar a lógica de composição das tools ao adaptador
  concreto de persistência, dificultando testes em memória.

## 4. Como traduzir erros de domínio para o protocolo MCP

- **Decision**: cada handler de tool captura os mesmos erros de domínio já tratados em
  `toStructuredError` (`ServiceNotFoundError`, `IncidentNotFoundError`, `InvalidSeverityError`) e
  retorna `{ content: [{ type: "text", text: JSON.stringify({ error: ..., ... }) }], isError: true }`
  — reaproveitando a mesma função `toStructuredError` já exportada/adaptada de `tools.ts`, em vez
  de recriar o mapeamento de erros.
- **Rationale**: é o mecanismo padrão do SDK para "falha de chamada de tool" (`isError: true`
  + conteúdo descrevendo o erro), documentado como o contrato oficial de error handling do
  `McpServer`; qualquer erro não capturado (falha de infraestrutura) continua sendo lançado e
  vira uma falha de protocolo em vez de um erro silencioso, preservando o mesmo contrato já usado
  no chat (comentário em `tools.ts`: "Falha de infraestrutura não recuperável... propaga para
  interromper o `run`").
- **Alternatives considered**: deixar o SDK converter exceções automaticamente (o `McpServer`
  também captura exceções não tratadas e as transforma em `isError: true` com `err.message`) —
  rejeitada como único mecanismo porque perderia a estrutura JSON (`{ error: "ServiceNotFoundError", service }`)
  que o chat já consome hoje; capturar explicitamente preserva o mesmo formato estruturado em
  ambas as interfaces.

## 5. Como garantir e testar "zero bytes em stdout fora do protocolo"

- **Decision**: (a) nenhuma chamada a `console.log`/`console.info`/`console.debug` em
  `src/mcp/server.ts` nem em qualquer módulo que ele importe no caminho de execução do servidor;
  qualquer diagnóstico usa `console.error` (stderr). (b) O teste automatizado
  (`src/mcp/server.test.ts`) usa `StdioClientTransport` do SDK para **spawnar o processo real**
  (`node --env-file-if-exists=.env --import tsx src/mcp/server.ts`, com `OPSPILOT_DB=":memory:"`)
  e um `Client` MCP chamando `client.listTools()`. Como o handshake MCP e o `listTools()` dependem
  de o stdout carregar *apenas* frames JSON-RPC válidos, qualquer escrita indevida no stdout
  (ex.: um `console.log` esquecido, ou de uma dependência transitiva) quebra o parsing e faz o
  teste falhar de forma determinística — não é necessário um assert manual de "stdout está vazio",
  o próprio handshake já expõe a violação.
- **Rationale**: testar contra o binário real (em vez de só uma composição em memória via
  `InMemoryTransport`) é a única forma de validar de ponta a ponta a regra crítica pedida pelo
  usuário, incluindo o próprio script `npm run mcp` e o carregamento de env antes de o servidor
  subir.
- **Alternatives considered**: testar só em memória com `InMemoryTransport` (também documentado
  pelo SDK) — mais rápido, mas não exercita o processo real nem o script npm, então não prova a
  regra crítica de stdout; por isso fica descartado como *único* teste, embora possa ser usado
  como complemento mais granular se necessário durante `/speckit.tasks`.

## 6. Script npm e carregamento de variáveis de ambiente

- **Decision**: `"mcp": "node --env-file-if-exists=.env --import tsx src/mcp/server.ts"` —
  mesmo padrão já usado por `dev`, `arena`, `bench` e `seed` no `package.json` atual, em vez do
  `"tsx src/mcp/server.ts"` literal sugerido inicialmente pelo usuário (que ele mesmo já
  antecipou precisar ajustar: "se precisar de env, alterar o script e carregar elas antes").
- **Rationale**: `SqliteOpsStore` lê `process.env.OPSPILOT_DB` no momento da construção
  (`src/store/sqlite-ops-store.ts:115`); o env precisa estar carregado antes desse ponto, e
  `--env-file-if-exists=.env` é o mecanismo padrão do projeto para isso (Node 24 nativo, sem
  dependência extra como `dotenv`).
- **Alternatives considered**: carregar `.env` manualmente dentro de `server.ts` (ex.: lendo e
  fazendo parse do arquivo) — rejeitado pela Constitution VI ("`.env` NUNCA é lido pelo agente ou
  por ferramentas automatizadas" no código da aplicação) e por duplicar um mecanismo que o
  runtime Node já oferece nativamente e que o projeto já usa em todos os outros scripts.

## Dependência nova a adicionar

- `@modelcontextprotocol/sdk` (dependency, não devDependency — é usada em runtime pelo
  entrypoint do servidor) — instalação e versão exata ficam para `/speckit.tasks`/`/speckit.implement`.
