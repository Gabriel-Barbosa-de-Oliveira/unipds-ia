# Phase 0 Research: Persistência Real de Operações

## 1. Substituir Sequelize/MySQL por SQLite nativo (`node:sqlite`)

**Decision**: A constitution do projeto (v1.1.0) já define SQLite via `node:sqlite` (`DatabaseSync`) como a stack obrigatória de persistência, no lugar de Sequelize + MySQL. Esta feature remove o adaptador `src/services/ops-store.sequelize.ts`, os modelos `src/models/sequelize/*` e a fábrica de conexão `connection.ts`, e as dependências `sequelize`/`mysql2` do `package.json` — código morto frente à nova stack, não apenas uma adição paralela. `node:sqlite` é nativo do runtime (Node 24 LTS já mandatado), então nenhuma dependência de pacote nova é introduzida (`DatabaseSync` emite um `ExperimentalWarning` no stderr, mas funciona sem flag `--experimental-sqlite` em Node 24; nenhum script do `package.json` precisa mudar por causa disso).

**Rationale**: Manter os dois adaptadores lado a lado violaria Pequeno e Reversível (código para uma stack que a própria constitution já revogou) e confundiria qual é o adaptador "real" a partir de agora. Como `node:sqlite` é builtin, a objeção original (`specs/001.../research.md` item 1) contra "Sequelize com SQLite in-memory para testes" — que rejeitava SQLite por exigir uma dependência de driver nova (`sqlite3`) — deixa de se aplicar.

**Alternatives considered**:
- *Manter `SequelizeOpsStore` como adaptador alternativo, não removê-lo*: rejeitado — nunca é instanciado por nenhum caminho de composição depois desta feature, e referencia uma stack que a constitution não mandata mais; é peso morto que ainda precisaria compilar/tipar a cada `npm run typecheck`.
- *Usar `sqlite3`/`better-sqlite3` (pacotes npm) em vez de `node:sqlite`*: rejeitado — adiciona dependência nativa/binária externa quando o runtime já mandatado (Node 24) resolve o mesmo problema sem nenhuma dependência nova.

## 2. Composição do store nas tools/estratégias — real (SQLite) vs. mock (in-memory)

**Decision**: `src/agents/tools.ts` deixa de importar um `store` singleton fixo e passa a exportar uma fábrica pura `createOpsTools(store: OpsStoreRepository)`, que constrói as 5 tools (as 3 existentes + as 2 novas) fechadas sobre o `store` recebido. O módulo continua exportando `opsTools` como o conjunto padrão, agora construído sobre uma instância de `SqliteOpsStore` (lida de `OPSPILOT_DB`, default `./data/opspilot.db`) — esse é o único ponto de composição do adaptador real, preservando o formato de import já usado hoje por `react.ts`/`plan-and-execute.ts`/`http/server.ts`/`src/index.ts` (nenhum desses arquivos muda). `react.ts` e `plan-and-execute.ts` passam a expor também uma fábrica (`createReactStrategy(tools)`, `createPlanAndExecuteStrategy(tools)`) por trás do singleton já exportado (`export const reactStrategy = createReactStrategy(opsTools)`), sem mudar a referência do singleton (`agents/index.test.ts` depende de igualdade referencial com `reactStrategy`/`planAndExecuteStrategy` — precisa continuar válida). `src/bench.ts` para de importar o `store` de `ops-store.memory.ts` para resetar/inspecionar; passa a construir sua própria `InMemoryOpsStore` e suas próprias estratégias via as novas fábricas, garantindo que o store que ele reseta/inspeciona (`before`/`after`) é exatamente o mesmo que as tools da estratégia mutam durante `run()`.

**Rationale**: Hoje `bench.ts` e `agents/tools.ts` compartilham o mesmo singleton importado de `ops-store.memory.ts` — se o padrão de `tools.ts` simplesmente trocasse para `SqliteOpsStore` sem mais nada, `bench.ts` continuaria resetando/lendo um `InMemoryOpsStore` desconectado do store real que as estratégias de fato usam, quebrando silenciosamente todo `before`/`after` do bench (FR-007, User Story 4 da spec). Fábricas pequenas ao lado dos singletons já existentes resolvem isso sem exigir mudar a assinatura de `ReasoningStrategy`/`RunOptions` nem tocar `reflection.ts`, `chat.service.ts`, `http/server.ts`, `agents/index.ts` ou `arena.ts` — nenhum deles for a afetado, e nenhum teste existente que depende da identidade dos singletons quebra (FR-011).

**Alternatives considered**:
- *Selecionar o adaptador por variável de ambiente dentro do próprio `ops-store.memory.ts`/`tools.ts` (ex.: `NODE_ENV === "test"`)*: implícito e frágil — o bench não roda com `NODE_ENV=test`, então ainda pegaria o adaptador real; e testes que quisessem o adaptador real de propósito (ex.: os testes do próprio `SqliteOpsStore`) ficariam presos à mesma variável. Rejeitado por esconder a composição em vez de deixá-la explícita, como a spec pede.
- *Adicionar um `store`/`tools` opcional em `RunOptions`*: mudaria o contrato de `ReasoningStrategy` (usado por `reflection.ts`, `chat.service.ts`, `http/server.ts`) só para resolver uma necessidade do bench; maior raio de mudança para o mesmo resultado.
- *Fazer `bench.ts` chamar as tools HTTP via `createApp` real*: mediria uma camada a mais (HTTP) sem necessidade — o bench já invoca `ReasoningStrategy.run(...)` diretamente hoje; manter esse padrão.

## 3. Dataset canônico único, reaproveitado por dois destinos

**Decision**: `src/domain/seed-data.ts` continua sendo a única fonte de verdade do cenário canônico (agora chamado "Mercadinho": os mesmos 5 serviços/6 alertas já existentes, mais `RUNBOOKS` para `checkout-api`, `payments-api` e `auth-service`) e passa a expor também esses runbooks via `buildSeedState()`. `InMemoryOpsStore` (usado por testes que não precisam de SQL e pelo bench) consome `buildSeedState()` diretamente como objetos JS. O seed do `SqliteOpsStore` (script `npm run seed` e testes de `":memory:"`) itera o mesmo `buildSeedState()` e faz `INSERT` de cada registro nas 4 tabelas — nenhuma segunda definição do dataset em SQL literal.

**Rationale**: Evita duas fontes de verdade divergentes para "o cenário canônico" (uma em JS, outra em SQL) — o requisito de reprodutibilidade (FR-007, SC-005) depende de que ambos os destinos derivem exatamente do mesmo dado.

**Alternatives considered**:
- *Arquivo `.sql` de seed separado*: duplicaria o dataset em dois formatos que podem divergir silenciosamente; rejeitado.

## 4. Idempotência de DDL e de seed

**Decision**: O construtor de `SqliteOpsStore` executa `CREATE TABLE IF NOT EXISTS` para as 4 tabelas a cada instanciação (idempotente por natureza — reabrir o mesmo arquivo ou um novo `":memory:"` nunca falha nem duplica schema). A função de seed (`seedCanonicalScenario(store)`, reaproveitada por `npm run seed` e pelos testes) verifica, por `id`, se cada registro já existe antes de inserir (`INSERT OR IGNORE`, casado com os `id`s estáveis já usados por `buildSeedState()`) — reexecutar o seed sobre um banco já semeado não duplica linhas nem falha por violar `UNIQUE`.

**Rationale**: Atende literalmente ao pedido ("DDL idempotente no construtor", "Seed idempotente") e à User Story 4 (restaurar o cenário sem acumular duplicatas entre execuções sucessivas do bench/testes).

**Alternatives considered**:
- *`DROP TABLE` + recriar a cada seed*: mais simples de implementar, mas destrói incidentes reais criados por execução ao vivo caso o seed seja rodado por engano contra o banco de produção (`OPSPILOT_DB` sem override) — mais perigoso do que necessário para o mesmo resultado em teste/bench, que já usam `":memory:"`/instância descartável.

## 5. `list_incidents` e `consultar_runbook`: domínio, repositório e tool

**Decision**: `src/domain/ops-store.ts` ganha `listIncidents(state, status?: IncidentStatus | "all")` (mesmo padrão de `listAlerts`: filtro opcional, lista vazia é resultado válido) e `getRunbookForService(state, serviceName)` (retorna o `Runbook` do serviço, `null` se o serviço existe mas não tem runbook, lança `ServiceNotFoundError` se o nome não corresponde a nenhum serviço — mesma função `findServiceByName` já usada por `openIncident`). `OpsStoreRepository` ganha os métodos equivalentes assíncronos (`listIncidents`, `getRunbook`), implementados por `InMemoryOpsStore` e `SqliteOpsStore`. `resolveIncident` (domínio, repositório, tool) ganha um `summary?: string` opcional, gravado apenas quando informado (`Incident.summary: string | null`).

**Rationale**: Mantém a mesma forma (funções puras no domínio, IO só no adaptador) já estabelecida para `listAlerts`/`openIncident`/`resolveIncident` em `specs/001.../data-model.md` — nenhum padrão novo introduzido, apenas estendido.

**Alternatives considered**:
- *`consultar_runbook` retornar erro quando não há runbook*: rejeitado pela spec (FR-005) — ausência de runbook é distinguível de um serviço inexistente, não é uma falha.

## 6. Descrições das tools (contrato de function-calling)

**Decision**: Todas as 5 descriptions de tool em `src/agents/tools.ts` são revisadas para: (a) declarar explicitamente quando usar cada uma frente às demais — em especial, `open_incident` passa a deixar claro que é só para *criar* um incidente novo, nunca para consultar o que já existe (isso é `list_incidents`); (b) todo campo de input ganha `.describe(...)` próprio (hoje nenhum campo tem); (c) todo campo de valor fechado usa `z.enum([...])` (já é o caso dos existentes; os dois novos seguem o mesmo padrão — `status` de `list_incidents` como `z.enum(["open", "resolved", "all"])`).

**Rationale**: Atende FR-008/FR-009 da spec — a ambiguidade mais provável para o modelo por trás do copiloto é confundir "abrir um incidente" com "consultar incidentes/runbook já existentes"; descrições explícitas de quando usar cada tool reduzem essa ambiguidade sem precisar de lógica de desambiguação em código.

**Alternatives considered**:
- *Um único tool "incidents" com um campo `action` (create/list/resolve)*: consolidaria 3 tools em 1, mas cada `tool()` do LangChain já carrega seu próprio schema/descrição — um campo `action` reintroduziria a mesma ambiguidade de desambiguação, só que dentro de um schema em vez de entre nomes de tool; rejeitado.

## 7. Testes isolados sobre `":memory:"`

**Decision**: `src/store/sqlite-ops-store.test.ts` (novo) e `src/agents/tools.test.ts` (novo) instanciam `new SqliteOpsStore(":memory:")` a cada teste (nunca um arquivo compartilhado, nunca `OPSPILOT_DB`), chamam `seedCanonicalScenario(store)` quando o teste precisa do cenário semeado, e descartam a instância ao final do teste (sem `afterEach` compartilhado entre arquivos). Os testes das tools existentes (hoje inexistentes como suíte própria — cobertos apenas indiretamente) passam a viver em `tools.test.ts`, exercitando as 5 tools por cima de um `SqliteOpsStore(":memory:")` semeado, em vez de qualquer mock ad hoc.

**Rationale**: Atende literalmente FR-010 (testes isolados, sem storage compartilhado) e SC-006; testar as tools contra o adaptador real (em memória) em vez de um dublê garante que a revisão de descriptions (item 6) e os `CHECK` do schema (item 4) sejam exercitados pelo mesmo caminho de código usado em produção.

**Alternatives considered**:
- *Testar as tools contra `InMemoryOpsStore`*: mais rápido, mas não exercitaria os `CHECK`/tipos de coluna reais do adaptador que passa a ser o padrão de produção; `InMemoryOpsStore` fica reservado para bench (item 2), onde fidelidade ao SQL real não é o que está sendo avaliado.
