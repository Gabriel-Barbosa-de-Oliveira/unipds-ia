# Research: War Room Web

Decisões da Fase 0 do `/speckit.plan`. Cada item segue o formato Decisão / Racional / Alternativas.

## 1. Como pausar uma ação que muda a produção

**Decisão**: usar uma **porta de aprovação por requisição** (`ApprovalGate`). Para cada `/chat`, o servidor monta as ferramentas `open_incident` e `resolve_incident` em versão "com porta" (`createGatedOpsTools(store, gate)`). Quando o agente chama uma delas, a ferramenta **não executa nada**: ela registra a chamada proposta (nome + argumentos já validados pelo schema da ferramenta) no `gate` e devolve ao modelo uma observação fixa (`{"status":"awaiting_approval", ...}`) dizendo que a ação ficou aguardando aprovação humana e que não deve tentar de novo. Uma segunda chamada com porta na mesma requisição devolve `{"status":"rejected"}`, porque só existe uma ação pendente por vez. Ao fim do grafo, se o `gate` tiver uma proposta, o controller grava a ação pendente e responde **202** em vez de 200.

**Racional**:
- O `ToolNode` do `createReactAgent` (`@langchain/langgraph` 0.2.74) usa `handleToolErrors = true` por padrão: qualquer exceção lançada pela ferramenta vira um `ToolMessage` para o modelo, exceto `GraphInterrupt` (`node_modules/@langchain/langgraph/dist/prebuilt/tool_node.js:177-183`). Uma exceção de domínio do tipo "ApprovalRequired" seria engolida e não chegaria ao controller.
- O `interrupt()` + checkpointer do LangGraph exigiria checkpointer persistente, `thread_id` e retomada dentro de três estratégias diferentes (react, plan-and-execute, reflection). Isso muda todas as estratégias e o grafo de produção. A porta funciona igual nas três sem alterar nenhuma.
- O padrão já existe no projeto: as ferramentas de memória (007) também são montadas por requisição (`createMemoryTools(memoryStore, userId)`).

**Alternativas consideradas**:
- `interrupt()` do LangGraph com checkpointer em SQLite: retomada fiel do raciocínio, mas é uma mudança grande e arriscada. Fica para uma feature futura, se a retomada com o modelo virar requisito.
- Lançar erro de domínio pela ferramenta com `handleToolErrors: false`: quebra o tratamento atual de erros das ferramentas e não funciona no plan-and-execute, que roda subagentes próprios.

## 2. O que acontece ao aprovar ou negar

**Decisão**: a decisão é **determinística, sem nova chamada ao modelo**.
- **Aprovar**: o servidor executa exatamente a ferramenta e os argumentos guardados (validados de novo com os mesmos shapes zod de `tools.ts`) direto no `OpsStoreRepository`. A resposta final é montada por uma função pura (`approvalAnswer`), por exemplo: "Incidente INC-42 resolvido.". Erros de domínio (`IncidentNotFoundError` etc.) viram uma resposta de falha legível, via `toStructuredError`.
- **Negar**: nada é executado. A resposta é "Ação cancelada: `resolve_incident` não foi executada."
- Nos dois casos a resposta tem **o mesmo formato do 200 do `/chat`** (`answer`, `trace`, `route`, `metrics`, `conversationId`, `requestId`). O trace tem os eventos `action` → `observation` → `answer` (ou só `answer`, na negação), todos com `node: "aprovacao"`. Assim o "ver raciocínio" da war room funciona sem caso especial.

**Racional**: o que a pessoa aprova é exatamente o que roda, sem a chance de o modelo decidir outra coisa numa segunda passada. Também é mais barato (zero chamadas ao modelo), testável sem rede e coerente com o Princípio VI.

**Alternativas consideradas**: rodar o grafo de novo com uma "pré-aprovação" da ação. O modelo pode escolher argumentos diferentes e cair num novo 202 sem fim. Os passos que o plano previa depois da ação se perdem na abordagem escolhida. Isso fica registrado como limitação: a pessoa pode pedir o próximo passo numa nova mensagem.

## 3. Ciclo de vida e concorrência da ação pendente

**Decisão**: uma nova tabela `pending_actions` no SQLite (detalhes em [data-model.md](./data-model.md)), com estados `pending → approved | denied | expired`. A decisão é gravada com um `UPDATE ... WHERE id = ? AND status = 'pending' AND expires_at > ?` atômico. Se nenhuma linha for afetada, o servidor relê a linha para devolver 404 (não existe), 409 (já decidida) ou 410 (expirou). A validade padrão é de **15 minutos**, injetável em `createApp` (`approvalTtlMs`) e configurável por `OPSPILOT_APPROVAL_TTL_MS`. Expirar é derivado, não um job: uma linha `pending` com `expires_at` no passado é tratada como `expired` na leitura.

**Racional**: o `UPDATE` condicional garante "no máximo uma decisão" (FR-019) mesmo com duplo clique ou duas abas, sem lock na aplicação. Expirar na leitura evita processos em segundo plano.

**Alternativas**: guardar só em memória (perde as ações pendentes ao reiniciar e não é coerente com o SQLite do projeto); usar um job de expiração (complexidade sem ganho).

## 4. Execução que gera 202 no registro (014)

**Decisão**: a execução que termina em 202 é gravada pelo nó `resposta` como hoje, com `outcome: "ok"`, porque o grafo terminou normalmente. O trace dela contém o `action` com porta e a `observation` `awaiting_approval`. A ação pendente referencia o `requestId` dessa execução. A decisão gera **uma requisição nova** (novo `requestId`), gravada pelo controller com `buildRequestRecord` e o trace de `node: "aprovacao"`.

**Racional**: a coluna `outcome` tem `CHECK (outcome IN ('ok','timeout','error'))`. Como a criação é `CREATE TABLE IF NOT EXISTS`, um valor novo exigiria migração de bancos existentes. O vínculo pela tabela `pending_actions` já deixa a pausa auditável.

## 5. Nó `aprovacao` no tipo do trace

**Decisão**: estender `GraphNode` com `"aprovacao"`. A coluna `trace_events.node` não tem `CHECK`, então não precisa de migração.

## 6. CORS

**Decisão**: um middleware próprio, sem dependência nova, sobre uma função pura `corsHeadersFor(origin, allowlist)`.
- Lista de origens permitidas em `OPSPILOT_CORS_ORIGINS` (separadas por vírgula, comparação exata de origem, sem curinga). Sem configuração, a lista é `["http://localhost:5173"]` (FR-028), a origem do dev server do Vite.
- Para origem permitida: `Access-Control-Allow-Origin: <origin>`, `Vary: Origin`, `Access-Control-Expose-Headers: X-Request-Id` (FR-027). No preflight (`OPTIONS`), o servidor também responde `Allow-Methods: GET, POST, OPTIONS`, `Allow-Headers: Content-Type` e `Max-Age: 600`, com status 204.
- Para origem não permitida: nenhum cabeçalho CORS. O navegador bloqueia (FR-026). O preflight responde 204 sem cabeçalhos.
- Sem cabeçalho `Origin` (curl, testes, MCP): o servidor não faz nada.
- Sem `Allow-Credentials`, porque não há cookies nem autenticação (spec, Assumptions).

**Racional**: são cerca de 30 linhas, testáveis como função pura, e evitam adicionar o pacote `cors` à stack obrigatória. A origem tem que ser exata porque a API mexe em incidentes de produção.

**Alternativas**: o pacote `cors` (dependência nova para pouca lógica); `*` (inaceitável pelo Princípio VI).

## 7. Stack e estrutura do `web/`

**Decisão**: `web/` como **pacote npm próprio** (`web/package.json`), com Vite 7, React 19, TypeScript 5 `strict` e zod 3. CSS puro com custom properties como tokens, sem lib de UI nem de CSS. Também sem router: é uma tela só, com painéis.

**Racional**:
- Pacote separado: as dependências do frontend não entram na API, e o `npm test` da raiz continua cobrindo só `src/`.
- Sem router: a war room é uma única tela (chat + painel de raciocínio + painel de configurações). Recarregar sob `/opspilot/` sempre cai no `index.html`, o que satisfaz o FR-025 sem configurar fallback no servidor estático.
- CSS puro com tokens implementa diretamente as instruções de design (escala de 4px, tokens semânticos, dark mode por `data-theme`).

**Alternativas**: workspace npm na raiz (acopla o lockfile da API ao do front); Tailwind (dependência e configuração extras, e os tokens já resolvem o problema); React Router (sem rotas reais para servir).

## 8. Caminho base `/opspilot/`

**Decisão**: `base: "/opspilot/"` no `vite.config.ts`. O dev server também serve em `http://localhost:5173/opspilot/`. Todos os assets usam caminhos relativos ao base, gerados pelo Vite.

## 9. Testes do frontend sem sair da constitution

**Decisão**: toda a lógica do `web/` fica em **módulos puros** em `web/src/lib/`, testados com `node:test` via `tsx`, igual à API (`web/package.json` → `"test": "node --import tsx --test \"src/**/*.test.ts\""`). Os módulos são: os schemas zod da API, a tradução de erros, o reducer da conversa, a máquina de estados do cartão de aprovação, a validação de URL, o modelo de exibição do trace e o tema. Os componentes React ficam finos, sem lógica além de chamar essas funções, e não têm teste automatizado nesta versão. A validação deles é pelo [quickstart.md](./quickstart.md), incluindo a checagem de acessibilidade com axe DevTools.

**Racional**: cumpre o Princípio V ("lógica nova nasce com teste") e o Princípio IV (funções puras) sem adicionar Vitest nem jsdom à stack. Mantém um único runner de testes no repositório.

**Alternativas**: Vitest + Testing Library (dependências e um segundo runner); Playwright (pesado para esta fase).

## 10. "Trace tipado" no navegador

**Decisão**: `web/src/lib/api-schemas.ts` define com zod uma união discriminada que espelha `TraceEvent` (`src/agents/types.ts`): `thought | plan | action | observation | critique | answer | route | fallback`, com `node` opcional, mais um **ramo genérico** (`{ type: string } & passthrough`) para tipos desconhecidos (FR-011). Toda resposta da API passa por esse schema antes de virar estado (Princípio II: a resposta da API é entrada externa para o navegador). Uma função pura `toTraceView(event)` transforma cada evento em `{ kind, label, icon, node, body }` para o painel.

**Contra divergência**: um teste na raiz (`src/http/web-contract.test.ts`) monta o `createApp` com fakes, faz `/chat` (200 e 202) e `/approvals/:id`, e valida os corpos com os schemas do `web/`. Se a API mudar o formato, o teste da API quebra. O `zod` importado pelo arquivo do `web/` resolve para o `node_modules` da raiz, a mesma versão maior.

## 11. Teste de conexão (engrenagem)

**Decisão**: `GET {apiUrl}/stats?since=1h` com timeout de 5s. Um 200 conta como "conectado". Erro de rede ou CORS, ou status diferente de 200, conta como "não foi possível conectar". Não é criado endpoint novo (Assumptions da spec).

## 12. Tempo de espera do `/chat` no navegador

**Decisão**: o navegador aborta com `AbortController` em **190s**, logo acima do teto de 180s da API, para que o 504 da API chegue antes do abort local. O indicador de "pensando" mostra o tempo decorrido a cada segundo.

## 13. Formatação da resposta do copiloto

**Decisão**: a resposta é exibida como **texto puro** com `white-space: pre-wrap`, preservando quebras de linha e listas simples. Não há renderização de Markdown nem HTML.

**Racional**: elimina XSS (a resposta vem de um modelo que lê dados de alertas) e não traz dependência. Markdown pode entrar depois com um sanitizador.

## 14. Persistência no navegador

**Decisão**: `localStorage` só para `opspilot.apiUrl` e `opspilot.theme`, com toda leitura e escrita em `try/catch` e volta para o padrão em caso de falha. A conversa não é persistida (Assumptions da spec).

## 15. Constitution: stack de frontend

**Decisão**: a stack obrigatória da constitution não prevê frontend. A **primeira tarefa** da implementação é um amendment **MINOR (1.1.0 → 1.2.0)** que adiciona uma subseção "Frontend (`web/`)": Vite + React + TypeScript `strict`, zod na fronteira com a API, testes `node:test` via `tsx` para a lógica pura e CSS com tokens conforme `.github/instructions/design.instructions.md`. O `CLAUDE.md` ganha os comandos do `web/`.

**Racional**: a seção de Governance exige amendment para qualquer mudança de stack. É uma adição, não uma redefinição, por isso o bump é MINOR.
