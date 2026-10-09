---

description: "Task list for 015-war-room-web"
---

# Tasks: War Room Web

> **Nota de implementação (seguindo o `/speckit-implement`)**:
> - **`parseGatedArgs`/`executeGatedAction` ficam em `src/agents/approval-gate.ts`, não no domínio.** Os dois reusam os shapes zod de `src/agents/tools.ts`; importá-los em `src/domain/` inverteria a direção das camadas (Princípio I). O domínio (`src/domain/approval.ts`) segue puro e sem zod.
> - **`tsconfig.json` da raiz: `rootDir` passou de `src` para `.`.** O `src/http/web-contract.test.ts` importa `web/src/lib/api-schemas.ts`; com `noEmit`, o `rootDir` não afeta saída nenhuma.
> - **`@vitejs/plugin-react@^5`.** A versão 6 exige Vite 8; o plano fixa Vite 7.
> - **Schemas de 202/decisão, `approval-machine` e o item `approval` da conversa nasceram junto da base (Fase 2/US1)**, para não reescrever os mesmos arquivos na US3. Os testes correspondentes (T053–T056) estão nas mesmas suítes.
> - **Contraste verificado por script** sobre os tokens: menor razão de texto 5,06:1 nos dois temas. Cores de trace e de sucesso do tema claro foram escurecidas para isso.

**Input**: Design documents from `/specs/015-war-room-web/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos. A constitution (Princípio V, NON-NEGOTIABLE) exige teste para toda lógica nova. Na API, os testes ficam em `src/**/*.test.ts` (`npm test`). No web, só os módulos puros de `web/src/lib/` têm teste (`npm --prefix web test`), conforme research.md item 9. Os componentes React são validados pelo [quickstart.md](./quickstart.md).

**Organization**: as tarefas estão agrupadas por user story (spec.md), para que cada uma possa ser implementada e testada de forma independente.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivo diferente e sem dependência de tarefa ainda não concluída).
- **[Story]**: a user story à qual a tarefa pertence (US1–US5).

## Path Conventions

- **API**: `src/` na raiz, com testes ao lado do código (`*.test.ts`).
- **War room**: pacote separado em `web/`. A lógica pura fica em `web/src/lib/` (com `*.test.ts` ao lado), o IO em `web/src/api/` e a visão em `web/src/components/`.

---

## Phase 1: Setup

**Purpose**: autorizar a stack de frontend e criar o pacote `web/`.

- [X] T001 Fazer o amendment da constitution em `.specify/memory/constitution.md` (research.md item 15):
  - bump MINOR 1.1.0 → 1.2.0, com `Last Amended: 2026-10-09`;
  - novo Sync Impact Report no topo;
  - na seção "Stack Tecnológica Obrigatória", adicionar a subseção "Frontend (`web/`)" com: Vite + React + TypeScript `strict` como pacote npm separado; zod validando toda resposta da API; lógica em funções puras testadas com `node:test` via `tsx`; CSS com tokens conforme `.github/instructions/design.instructions.md`;
  - em "Fluxo de Desenvolvimento & Quality Gates", acrescentar `npm --prefix web run typecheck` e `npm --prefix web test` como gates.
- [X] T002 Atualizar `CLAUDE.md`:
  - na Stack, uma linha "Frontend em `web/` (Vite + React + TS)";
  - em Comandos, `npm --prefix web run dev|build|preview|test|typecheck`;
  - corrigir o caminho das instruções de design citado, se houver.
- [X] T003 Criar `web/package.json`:
  - `"name": "ops-pilot-web"`, `"private": true` e `"type": "module"`;
  - scripts: `dev: "vite"`, `build: "tsc --noEmit && vite build"`, `preview: "vite preview"`, `test: "node --import tsx --test \"src/**/*.test.ts\""`, `typecheck: "tsc --noEmit"`;
  - deps: `react@^19`, `react-dom@^19` e `zod@^3.23`;
  - devDeps: `vite@^7`, `@vitejs/plugin-react`, `typescript@^5.5`, `@types/react`, `@types/react-dom` e `tsx@^4.19`.

  Depois, rodar `npm --prefix web install`.
- [X] T004 [P] Criar `web/tsconfig.json`:
  - `strict`, `target: "ES2022"`, `module: "ESNext"` e `moduleResolution: "Bundler"`;
  - `jsx: "react-jsx"`, `lib: ["ES2022","DOM","DOM.Iterable"]`, `allowImportingTsExtensions: true`, `noEmit: true` e `include: ["src","vite.config.ts"]`.

  Seguir o estilo de imports com extensão `.ts`/`.tsx` usado na API.
- [X] T005 [P] Criar `web/vite.config.ts` com `base: "/opspilot/"` (research.md item 8), `plugins: [react()]` e `server.port: 5173`.
- [X] T006 [P] Criar `web/index.html`:
  - `lang="pt-BR"`, `<title>OpsPilot · War room</title>` e `meta viewport`;
  - um script inline mínimo que aplica `data-theme` antes do primeiro paint, lendo `localStorage["opspilot.theme"]` dentro de `try/catch` e caindo em `prefers-color-scheme`;
  - `<div id="root">` e `<script type="module" src="/src/main.tsx">`.
- [X] T007 [P] Garantir que `.gitignore` cubra `web/node_modules/` e `web/dist/`. O `node_modules/` genérico já cobre o primeiro. Acrescentar `dist/` se faltar.

**Checkpoint**: `npm --prefix web run typecheck` passa num `main.tsx` vazio, e a constitution está na versão 1.2.0.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: o que todas as stories usam: CORS na API, schemas e cliente HTTP do web, tradução de erros, configurações base, tokens visuais e o componente de diálogo.

**⚠️ CRITICAL**: nenhuma user story começa antes desta fase terminar.

### CORS na API (pré-requisito para o navegador falar com a API)

- [X] T008 [P] Criar `src/domain/cors.test.ts` (deve falhar até a T009):
  - `parseAllowedOrigins(undefined)` → `["http://localhost:5173"]`;
  - `parseAllowedOrigins(" https://a.com , http://localhost:5173 ,")` → lista sem espaços nem vazios;
  - `corsHeadersFor("https://a.com", ["https://a.com"], { preflight: false })` → `Access-Control-Allow-Origin`, `Vary: Origin` e `Access-Control-Expose-Headers: X-Request-Id`;
  - `corsHeadersFor` com `preflight: true` → acrescenta `Allow-Methods: GET, POST, OPTIONS`, `Allow-Headers: Content-Type` e `Max-Age: 600`;
  - origem fora da lista → `{}`;
  - `origin` `undefined` → `{}`;
  - nunca devolve `Access-Control-Allow-Credentials`.
- [X] T009 Criar `src/domain/cors.ts` (puro) com `DEFAULT_CORS_ORIGINS`, `parseAllowedOrigins(raw?: string): string[]` e `corsHeadersFor(origin: string | undefined, allowlist: readonly string[], opts: { preflight: boolean }): Record<string, string>`, conforme [contracts/http.md](./contracts/http.md#cors-todas-as-rotas).
- [X] T010 Em `src/http/server.ts`:
  - adicionar `corsOrigins?: string[]` em `CreateAppOptions`, com padrão `parseAllowedOrigins(process.env.OPSPILOT_CORS_ORIGINS)`;
  - registrar, antes de `express.json()`, um middleware que aplica `corsHeadersFor` em toda resposta;
  - para `OPTIONS` com cabeçalho `Origin`, responder 204 e encerrar (com ou sem cabeçalhos, conforme a allowlist). Sem `Origin`, chamar `next()`.
- [X] T011 Em `src/http/server.test.ts`, criar o bloco `describe("CORS")`:
  - preflight `OPTIONS /chat` com origem permitida → 204 com os cabeçalhos;
  - com origem não permitida → 204 sem `access-control-allow-origin`;
  - `POST /chat` (fake) com origem permitida → `access-control-allow-origin` igual à origem e `access-control-expose-headers` contendo `X-Request-Id`;
  - sem `Origin` → nenhum cabeçalho CORS (regressão das suítes existentes).

### Base do web

- [X] T012 [P] Criar `web/src/lib/api-schemas.test.ts` (deve falhar até a T013), com fixtures copiadas de [contracts/http.md](./contracts/http.md):
  - `TraceEventSchema` aceita os 8 tipos conhecidos (com e sem `node`);
  - aceita tipo desconhecido (`{type:"novo", at:3, foo:1}`) pelo ramo genérico, preservando os campos;
  - rejeita `at` ausente;
  - `ChatOkSchema` aceita o 200 de exemplo;
  - `ApiErrorSchema` aceita `{requestId, error:"timeout", timeoutMs}` e `{error:"internal_error"}`.
- [X] T013 Criar `web/src/lib/api-schemas.ts` (research.md item 10) com:
  - `GraphNodeSchema`, `RouteDecisionSchema`, `MetricsSchema` (`.passthrough()` para os campos de contexto);
  - `TraceEventSchema`: `z.union([z.discriminatedUnion("type", [8 ramos]), UnknownTraceEventSchema])`, onde o desconhecido é `z.object({type: z.string(), at: z.number()}).passthrough()`;
  - `ChatOkSchema` e `ApiErrorSchema` (`{ requestId?: string, error: string }.passthrough()`), mais os tipos inferidos exportados.

  Os schemas de 202 e de decisão entram na US3.
- [X] T014 [P] Criar `web/src/lib/errors.test.ts` (deve falhar até a T015), cobrindo cada linha da tabela UiError do [data-model.md](./data-model.md#uierror-tradução-pura-em-errorsts):
  - status + corpo → `title` e `action` esperados, com `requestId` propagado;
  - `TypeError("Failed to fetch")` → `open_settings`;
  - `DOMException` de nome `AbortError` → mesmo resultado de timeout;
  - `ZodError` → "formato inesperado";
  - nenhum `title` ou `detail` contém JSON cru ou stack.
- [X] T015 Criar `web/src/lib/errors.ts` com `type UiError` e `toUiError(input: { status: number; body: unknown } | { exception: unknown }): UiError` (puro).
- [X] T016 [P] Criar `web/src/lib/settings.test.ts` (deve falhar até a T017):
  - `normalizeApiUrl("http://localhost:3000/")` → `ok("http://localhost:3000")`;
  - `" https://api.x.com/base/ "` → sem espaços e sem barra final, preservando o caminho;
  - `"localhost:3000"`, `"ftp://x"` e `""` → `err("…mensagem humana…")`;
  - `parseStoredSettings({apiUrl:"lixo", theme:"roxo"})` → padrões;
  - `resolveTheme("system", true)` → `"dark"` e `resolveTheme("light", true)` → `"light"`.
- [X] T017 Criar `web/src/lib/settings.ts` (puro) com:
  - `DEFAULT_API_URL = "http://localhost:3000"`, `ThemePreference`, `SettingsSchema` (zod);
  - `normalizeApiUrl(raw): { ok: true; value } | { ok: false; error }`;
  - `parseStoredSettings(raw: unknown): Settings` e `resolveTheme(pref, prefersDark): "light" | "dark"`.
- [X] T018 [P] Criar `web/src/api/storage.ts` com `loadSettings(): Settings` e `saveSettings(s: Settings): void` sobre `localStorage` (chaves `opspilot.apiUrl` e `opspilot.theme`). Toda leitura e escrita fica em `try/catch`, e a leitura usa `parseStoredSettings` (research.md item 14).
- [X] T019 Criar `web/src/api/client.ts` com `postChat(apiUrl, body, signal?)`:
  - `fetch` `POST {apiUrl}/chat` com JSON e `AbortController` de 190s combinado ao `signal` externo (research.md item 12);
  - em 200, faz parse com `ChatOkSchema` e devolve `{ kind: "ok", data }`;
  - fora de 2xx, devolve `{ kind: "error", error: toUiError({status, body}) }`;
  - exceção (rede, abort, zod) → `{ kind: "error", error: toUiError({exception}) }`;
  - o `requestId` vem do corpo ou, na falta dele, do cabeçalho `X-Request-Id`.

  Nunca lança.
- [X] T020 [P] Criar `web/src/styles/tokens.css` conforme [contracts/web-ui.md](./contracts/web-ui.md#tokens-webstylestokenscss):
  - `--space-1..8` (4, 8, 12, 16, 24, 32, 48, 64 px), escala tipográfica 12/14/16/20/24/32 e `--radius-*`;
  - cores semânticas e `--trace-*` em `:root[data-theme="light"]` e `:root[data-theme="dark"]`, com contraste AA (texto ≥ 4.5:1, UI ≥ 3:1), sem `#000` nem `#fff` no tema escuro;
  - `@media (prefers-reduced-motion: reduce)` zerando transições e animações.
- [X] T021 [P] Criar `web/src/styles/app.css` com:
  - reset mínimo, `body` com `background: var(--color-bg)` e `color: var(--color-text)`, corpo 16px e line-height 1.5;
  - `:focus-visible` com contorno `--color-focus` de 2px e offset de 2px;
  - classes de botão `.btn-primary`, `.btn-secondary` e `.btn-ghost` com alvo mínimo de 44×44px, layout com `gap`, coluna central de 760px no máximo e sem rolagem horizontal a partir de 360px.

  Só tokens, nenhum valor mágico.
- [X] T022 [P] Criar `web/src/components/Dialog.tsx`: diálogo modal acessível, usado pelas US2 e US4.
  - `role="dialog"`, `aria-modal` e `aria-labelledby` no `h2`;
  - foco no título ao abrir, foco preso no painel e `Esc` fecha;
  - devolve o foco ao elemento que abriu;
  - gaveta de 480px à direita a partir de 768px e tela cheia abaixo disso.
- [X] T023 Criar `web/src/main.tsx` e um `web/src/App.tsx` mínimo:
  - `App` lê `loadSettings()`, aplica `data-theme` via `resolveTheme` e escuta `matchMedia("(prefers-color-scheme: dark)")` quando o tema é `system`;
  - renderiza o header (`h1` "OpsPilot · War room") e um `main` vazio;
  - importa `tokens.css` e `app.css`.

**Checkpoint**: `npm test`, `npm --prefix web test` e os dois `typecheck` ficam verdes. `npm --prefix web run dev` abre `http://localhost:5173/opspilot/` com o header nos dois temas.

---

## Phase 3: User Story 1 - Conversar com o copiloto pelo navegador (Priority: P1) 🎯 MVP

**Goal**: enviar mensagens ao `/chat` pela war room, com continuidade da conversa, estados vazio, carregando e erro, e "Tentar novamente".

**Independent Test**: com a API no ar, enviar duas mensagens em que a segunda depende da primeira. As duas respostas aparecem em ordem e a segunda usa o contexto. Com a API derrubada, o erro aparece em linguagem humana com a ação correta.

### Tests for User Story 1

- [X] T024 [P] [US1] Criar `web/src/lib/conversation.test.ts` (deve falhar até a T025):
  - `send` adiciona um item `user` em `sending` e coloca `pending: "sending"`;
  - `send` com texto vazio ou só espaços, ou com `pending !== "idle"`, devolve o mesmo estado (referência igual);
  - `received(ok)` marca o `user` como `delivered`, adiciona `assistant` com o `run`, adota o `conversationId` (só se ainda era `null`) e volta a `idle`;
  - `failed(error, text)` marca o `user` como `failed`, adiciona o item `error` com `retryText` e volta a `idle`;
  - `retry(itemId)` remove o item de erro e reenvia o texto;
  - `reset` volta ao estado inicial;
  - `failed` com `action: "new_conversation"` mantém o `conversationId` até `reset`.
- [X] T025 [US1] Criar `web/src/lib/conversation.ts` (puro) com `ConversationState`, `ConversationItem` (`user | assistant | error`; o `approval` entra na US3), `ChatRun` e `initialConversation`. Também `chatReducer(state, action)` para as ações `send | received | failed | retry | reset`. Os ids vêm de um `makeId` injetado na ação, para o reducer continuar puro.
- [X] T026 [P] [US1] Criar `src/http/web-contract.test.ts` (API). Montar `createApp` com o fake de estratégia e os stores `:memory:` no mesmo padrão de `src/http/server.test.ts`, e fazer `POST /chat`. Validar o corpo 200 com `ChatOkSchema` e o corpo 400 com `ApiErrorSchema`, importados de `../../web/src/lib/api-schemas.ts` (research.md item 10). O teste deve passar sem rede.

### Implementation for User Story 1

- [X] T027 [P] [US1] Criar `web/src/components/EmptyState.tsx`: `h2` "Pronto para o plantão", texto curto e 3 botões de exemplo ([contracts/web-ui.md](./contracts/web-ui.md#conversa-main)). Clicar chama `onPick(text)`.
- [X] T028 [P] [US1] Criar `web/src/components/ThinkingIndicator.tsx`: skeleton com `aria-busy` e o texto "Pensando… {n}s", atualizado a cada 1s. Ele só aparece depois de 300ms e respeita o movimento reduzido.
- [X] T029 [P] [US1] Criar `web/src/components/MessageItem.tsx` com os itens `user` (alinhado à direita, com o estado `failed` indicado por ícone e texto) e `assistant` (texto em `white-space: pre-wrap`, sem `dangerouslySetInnerHTML`, research.md item 13). O `assistant` aceita um slot `actions`, onde a US2 coloca o "Ver raciocínio".
- [X] T030 [P] [US1] Criar `web/src/components/ErrorItem.tsx`: ícone, `title`, `detail`, "ID da requisição" com botão copiar (`navigator.clipboard`, com falha silenciosa) e o botão da `action` (`Tentar novamente` / `Abrir configurações` / `Nova conversa` / nenhum), via callbacks.
- [X] T031 [P] [US1] Criar `web/src/components/Composer.tsx`:
  - `<label>` "Mensagem" visível e `textarea`; `Enter` envia e `Shift+Enter` quebra linha;
  - "Enviar" é a ação primária, desabilitada se o texto estiver vazio ou se `disabled`, com `disabledReason` exibido como nota;
  - o texto só é limpo quando o envio é aceito pelo reducer.
- [X] T032 [US1] Criar `web/src/components/ConversationLog.tsx`: `main` com `role="log"` e `aria-live="polite"`. Mostra `EmptyState` quando não há itens e `ThinkingIndicator` quando `pending === "sending"`. Rola até o fim quando chega item novo, sem animação se o movimento for reduzido.
- [X] T033 [US1] Integrar em `web/src/App.tsx`:
  - `useReducer(chatReducer)`;
  - ao enviar, chamar `postChat(settings.apiUrl, { message, conversationId? })` e despachar `received` ou `failed`;
  - "Nova conversa" no header (botão ghost) despacha `reset`;
  - ligar `retry` e a ação `new_conversation` do `ErrorItem`;
  - um único request por vez (o reducer já bloqueia, e o botão também).
- [X] T034 [US1] Criar `web/src/components/Header.tsx` (extraído do `App`): `h1`, "Nova conversa" e o botão de engrenagem com `aria-label="Configurações"` (abre o painel na US4; até lá, `onOpenSettings` opcional). Usar o componente no `App.tsx`.

**Checkpoint**: os cenários 4.1–4.3 e 4.6 do [quickstart.md](./quickstart.md) passam. A US1 é entregável sozinha (MVP).

---

## Phase 4: User Story 2 - Ver o raciocínio por trás de uma resposta (Priority: P1)

**Goal**: o "Ver raciocínio" abre o trace tipado da execução, com formato por tipo, nó, rota, métricas e ID.

**Independent Test**: uma pergunta que use ferramenta → "Ver raciocínio" mostra todos os eventos na ordem, cada tipo com rótulo e ícone próprios e o nó indicado. `Esc` fecha e devolve o foco.

### Tests for User Story 2

- [X] T035 [P] [US2] Criar `web/src/lib/trace-view.test.ts` (deve falhar até a T036):
  - para cada um dos 8 tipos, `toTraceView` devolve `kind`, `label` em PT (Rota / Pensamento / Plano / Ação / Observação / Crítica / Troca de modelo / Resposta) e `icon` distintos;
  - `plan` → `body.steps` na ordem;
  - `action` → `body.tool` e `body.argsText` (JSON indentado);
  - `observation` com resultado em string JSON → objeto formatado;
  - `observation` com mais de 600 caracteres → `long: true`;
  - `fallback` → `"from → to"`;
  - tipo desconhecido → `kind: "unknown"`, `label: "Evento: novo"` e `body.raw` com JSON legível;
  - `node` ausente → `node: null`;
  - `summarizeRun(run)` com `route` e `metrics` `null` → seções omitidas.
- [X] T036 [US2] Criar `web/src/lib/trace-view.ts` (puro) com `toTraceView(event: TraceEvent): TraceView`, `summarizeRun(run: ChatRun)` (rota, motivo, origem, métricas formatadas e ID) e o limite `LONG_CONTENT_CHARS = 600`.

### Implementation for User Story 2

- [X] T037 [P] [US2] Criar `web/src/components/TraceEventItem.tsx`: um `<li>` com ícone (`aria-hidden`) e rótulo visível do tipo, badge do nó, borda à esquerda com a cor `--trace-<kind>` (a cor nunca é o único sinal) e o corpo por `kind`. O plano é um `<ol>`, `args`/`result` ficam em `<pre>`, e conteúdo `long` fica dentro de `<details><summary>Mostrar conteúdo</summary>`.
- [X] T038 [US2] Criar `web/src/components/ReasoningPanel.tsx` sobre o `Dialog`, com o título `h2` "Raciocínio":
  - cabeçalho com a rota (badge + motivo + origem), as métricas (chamadas ao modelo, tempo, modelo, tokens) e o ID com "Copiar";
  - lista `<ol>` de `TraceEventItem`;
  - estado vazio "Essa execução não registrou eventos." quando o trace estiver vazio.
- [X] T039 [US2] Em `web/src/components/MessageItem.tsx` e `web/src/App.tsx`:
  - botão secundário "Ver raciocínio" (`aria-haspopup="dialog"`) em cada item `assistant`;
  - o `App` guarda `openRun: ChatRun | null` e o elemento de origem, e renderiza o `ReasoningPanel`;
  - ao fechar, o foco volta ao botão.

**Checkpoint**: os cenários 4.4–4.5 do [quickstart.md](./quickstart.md) passam.

---

## Phase 5: User Story 3 - Aprovar ou negar uma ação antes que ela aconteça (Priority: P2)

**Goal**: `open_incident` e `resolve_incident` não executam sem decisão humana. A API responde 202 e expõe `POST /approvals/:id`, e a war room mostra o cartão Aprovar/Negar.

**Independent Test**: "resolva o incidente X" → cartão de aprovação, e o incidente continua aberto. Aprovar → resolvido e com resposta. Negar → nada muda. Uma decisão repetida → 409.

### Tests for User Story 3 (API)

- [X] T040 [P] [US3] Criar `src/domain/approval.test.ts` (deve falhar até a T043):
  - `effectiveStatus` (pending antes e depois de `expiresAt`, approved e denied);
  - `summarizeAction("resolve_incident", {id:"INC-42"})` → "Resolver o incidente INC-42";
  - `summarizeAction("open_incident", {severity:"critical", service:"checkout-api", title:"Latência"})` → "Abrir incidente critical em checkout-api: Latência";
  - `reasonFromTrace(trace)` → o último `thought` antes do último `action` com porta, ou `null`;
  - `approvalAnswer` para approved com sucesso, approved com `{error:"IncidentNotFoundError"}` e denied;
  - `approvalTrace(...)` → `action, observation, answer` com `node: "aprovacao"` e `at` 0..2, ou só `answer` quando negada;
  - `parseGatedArgs(tool, args)` valida com os shapes de `src/agents/tools.ts` e rejeita args inválidos.
- [X] T041 [P] [US3] Criar `src/store/sqlite-approval-store.test.ts` (deve falhar até a T045), em `:memory:` e no estilo de `src/store/sqlite-request-store.test.ts`:
  - `create` + `find` fazem round-trip de `args`;
  - `decide(id, "approved", now, decisionRequestId)` → `{ ok: true }`, e o status fica gravado;
  - um segundo `decide` → `{ ok: false, reason: "already_decided", status }`;
  - `decide` depois de `expiresAt` → `{ ok: false, reason: "expired" }`;
  - id inexistente → `{ ok: false, reason: "not_found" }`;
  - dois `decide` concorrentes (`Promise.all`) → exatamente um `ok`.
- [X] T042 [P] [US3] Criar `src/agents/approval-gate.test.ts` (deve falhar até a T046), usando `new InMemoryOpsStore()` de `src/services/ops-store.memory.ts`:
  - invocar o `resolve_incident` com porta → devolve JSON com `status: "awaiting_approval"`, o `gate.proposed` fica preenchido e o incidente no store continua aberto;
  - uma segunda chamada com porta → `status: "rejected"`, e `gate.proposed` não muda;
  - `list_alerts`, `list_incidents` e `consultar_runbook` executam normalmente;
  - os nomes das 5 ferramentas são iguais aos de `createOpsTools`.

### Implementation for User Story 3 (API)

- [X] T043 [US3] Criar `src/domain/approval.ts` (puro) com:
  - `GATED_TOOLS = ["open_incident","resolve_incident"] as const`, `GatedToolName`, `isGatedTool`;
  - `PendingAction`, `ApprovalStatus`, `EffectiveApprovalStatus`;
  - `effectiveStatus`, `summarizeAction`, `reasonFromTrace`, `parseGatedArgs` (reusando `openIncidentShape` e `resolveIncidentShape` de `src/agents/tools.ts`), `approvalAnswer` e `approvalTrace`;
  - em `src/agents/types.ts`, adicionar `"aprovacao"` a `GraphNode`.
- [X] T044 [US3] Em `src/domain/errors.ts`, adicionar `ApprovalNotFoundError(id)`, `ApprovalAlreadyDecidedError(id, status)` e `ApprovalExpiredError(id, expiresAt)` no mesmo estilo das classes existentes. Criar também `src/services/approval-store.repository.ts`:

  ```ts
  interface ApprovalStore {
    create(a: PendingAction): Promise<void>;
    find(id): Promise<PendingAction | undefined>;
    decide(id, status: "approved" | "denied", now: Date, decisionRequestId: string): Promise<
      | { ok: true; action: PendingAction }
      | { ok: false; reason: "not_found" | "already_decided" | "expired"; action?: PendingAction }
    >;
  }
  ```
- [X] T045 [US3] Criar `src/store/sqlite-approval-store.ts` com `SqliteApprovalStore`, seguindo `src/store/sqlite-request-store.ts`:
  - `DatabaseSync` com caminho de `OPSPILOT_DB` ou do construtor, e `CREATE TABLE IF NOT EXISTS pending_actions` + índice conforme o [data-model.md](./data-model.md#pendingaction-nova-tabela-pending_actions);
  - `decide` com um único `UPDATE ... WHERE id=? AND status='pending' AND expires_at > ?` e `changes === 0` → relê para classificar o motivo.
- [X] T046 [US3] Criar `src/agents/approval-gate.ts` com `interface ApprovalGate { proposed?: { tool: GatedToolName; args: Record<string, unknown> } }`, `createApprovalGate()` e `createGatedOpsTools(store, gate): StructuredToolInterface[]`. A função reaproveita as ferramentas de leitura de `createOpsTools(store)` e substitui `open_incident`/`resolve_incident` por versões com o mesmo `name`, `description` e `schema` que só registram a proposta e devolvem a observação fixa (research.md item 1).
- [X] T047 [US3] Em `src/agents/index.ts`:
  - `resolveStrategy(name, reflect, extraTools, baseTools = opsTools)`, que compõe uma estratégia nova quando `baseTools !== opsTools` ou quando `extraTools` não está vazio;
  - `strategyForRoute(route, reflect, extraTools, resolve, baseTools)` repassa `baseTools`;
  - `src/agents/index.test.ts` continua verde.
- [X] T048 [US3] Em `src/obs/logger.ts`, adicionar ao `LogEvent` os eventos `approval.requested` (`requestId`, `approvalId`, `tool`) e `approval.decided` (`requestId`, `approvalId`, `decision`, `outcome: "executed" | "cancelled" | "failed"`), com nível `info`. Atualizar `src/obs/logger.test.ts` com um caso que confirme que não saem `args` nem `reason`.
- [X] T049 [US3] Em `src/http/server.ts`, fazer o 202 no `/chat`:
  - `CreateAppOptions` ganha `opsStore?: OpsStoreRepository` (padrão `new SqliteOpsStore()`), `approvalStore?: ApprovalStore` (padrão `new SqliteApprovalStore()`) e `approvalTtlMs?: number` (padrão `Number(process.env.OPSPILOT_APPROVAL_TTL_MS) || 900_000`);
  - por requisição, criar `gate` e `baseTools = createGatedOpsTools(opsStore, gate)` e repassá-los via `strategyFor`;
  - depois do grafo, se `gate.proposed` existir: `approvalStore.create(...)` (com `reason = reasonFromTrace(result.trace)` e `expiresAt = now + ttl`), anexar à conversa `user` + `assistant: "Aguardando aprovação: <summary>"`, logar `approval.requested` e responder **202** no formato de [contracts/http.md](./contracts/http.md#202-ação-aguardando-aprovação-novo), sem `answer`;
  - sem proposta, o comportamento 200 não muda.
- [X] T050 [US3] Em `src/http/server.ts`, criar a rota `POST /approvals/:id` (com `assignRequestId`):
  - validar `:id` (UUID, inválido → 404) e o corpo com `z.object({ decision: z.enum(["approve","deny"]) })`, inválido → 400 `invalid_body`;
  - chamar `approvalStore.decide`; em caso de falha, lançar o erro de domínio correspondente;
  - aprovado: `parseGatedArgs` e executar `opsStore.resolveIncident`/`openIncident`, convertendo erro de domínio com `toStructuredError`;
  - montar `approvalTrace`/`approvalAnswer`, gravar com `requestStore.save(buildRequestRecord({ outcome: "ok", ... }), trace)` (falha de gravação só vira log), anexar `assistant: answer` à conversa e logar `approval.decided`;
  - responder 200 com `{ requestId, answer, trace, route: null, metrics: null, conversationId, approval: { id, status } }`;
  - no `createErrorMiddleware`, mapear os três erros novos para 404, 409 e 410, com os corpos do contrato.
- [X] T051 [US3] Em `src/http/server.test.ts`, criar o bloco `describe("aprovação")`. Usar um fake de estratégia que chama `baseTools.find(t => t.name === "resolve_incident").invoke({id})`, mais `new InMemoryOpsStore()` e `SqliteApprovalStore(":memory:")`:
  - `/chat` → 202 com `approval.summary`, sem `answer`, e o incidente continua aberto;
  - `approve` → 200, incidente resolvido, trace com `node: "aprovacao"` e `GET /requests/:novoId` encontra o registro;
  - `deny` → 200 e o incidente continua aberto;
  - repetir a decisão → 409;
  - com `approvalTtlMs: 1` e `now` avançado → 410;
  - id inexistente → 404;
  - `decision: "talvez"` → 400;
  - o histórico da conversa contém as mensagens de aguardo e de decisão;
  - os logs não contêm `args`.
- [X] T052 [US3] Em `src/http/web-contract.test.ts`, acrescentar a validação do 202 com `ChatAwaitingApprovalSchema` e do 200 de decisão com `DecisionOkSchema` (depende da T053).

### Tests for User Story 3 (web)

- [X] T053 [P] [US3] Em `web/src/lib/api-schemas.ts` e `web/src/lib/api-schemas.test.ts`:
  - adicionar `ApprovalSchema` (`id`, `tool`, `args`, `summary`, `reason` nulo e `expiresAt`);
  - adicionar `ChatAwaitingApprovalSchema` (`status: z.literal("awaiting_approval")`, sem `answer`);
  - adicionar `DecisionOkSchema` (como o `ChatOkSchema`, com `route` e `metrics` nulos e `approval: {id, status}`);
  - testar com as fixtures do contrato.
- [X] T054 [P] [US3] Criar `web/src/lib/approval-machine.test.ts` (deve falhar até a T055) cobrindo as transições do [data-model.md](./data-model.md#approvalcard-máquina-de-estados-pura-approval-machinets):
  - `pending → submitting` em `decide`;
  - `submitting → approved | denied` em `succeeded`;
  - `submitting → unavailable("Já decidida" | "Expirou" | "Não encontrada")` em 409, 410 ou 404;
  - `submitting → pending + error` em rede ou 5xx;
  - `decide` fora de `pending` → mesmo estado (referência igual).
- [X] T055 [US3] Criar `web/src/lib/approval-machine.ts` (puro) com `ApprovalCardState` e `approvalReducer`. Em `web/src/lib/errors.ts`, tratar 409, 410 e 404 de aprovação como `unavailableReason`. Atualizar `errors.test.ts`.
- [X] T056 [US3] Em `web/src/lib/conversation.ts` e `conversation.test.ts`:
  - o item `approval` passa a existir;
  - `received(awaiting)` adiciona o cartão, adota o `conversationId` e coloca `pending: "awaiting_decision"`;
  - `send` em `awaiting_decision` é ignorado;
  - `approvalUpdated(itemId, cardAction)` delega ao `approvalReducer`;
  - quando o cartão sai de `pending`/`submitting`, `pending` volta a `idle`;
  - `decisionReceived(itemId, run)` anexa o `assistant`.

### Implementation for User Story 3 (web)

- [X] T057 [US3] Em `web/src/api/client.ts`:
  - `postChat` passa a reconhecer o 202 (`ChatAwaitingApprovalSchema`) e devolver `{ kind: "awaiting", data }`;
  - novo `postDecision(apiUrl, approvalId, decision)` → `{ kind: "ok", data: DecisionOk } | { kind: "error", status, error }`, com timeout de 30s.

  Nunca lança.
- [X] T058 [US3] Criar `web/src/components/ApprovalCard.tsx` conforme [contracts/web-ui.md](./contracts/web-ui.md#cartão-de-aprovação):
  - `role="group"` com `aria-labelledby`;
  - título "Ação aguardando aprovação" com ⚠ e o horário de expiração;
  - `summary`, "Motivo:" e `<details>` com a ferramenta e os `args`;
  - "Negar" (secundário) e "Aprovar" (primário) com no mínimo 44×44px, desabilitados fora de `pending`;
  - estados com ícone e texto; erro inline em caso de falha;
  - "Ver raciocínio" do 202.
- [X] T059 [US3] Integrar em `web/src/App.tsx` e `web/src/components/ConversationLog.tsx`:
  - renderizar os itens `approval`;
  - nos cliques, despachar `approvalUpdated(decide)`, chamar `postDecision` e despachar `succeeded` + `decisionReceived` ou `failed`;
  - passar `disabledReason="Decida a ação pendente antes de enviar outra mensagem"` ao `Composer` em `awaiting_decision`.

**Checkpoint**: a seção 5 do [quickstart.md](./quickstart.md) passa, e `npm test` cobre 202, 200, 409, 410, 404 e 400.

---

## Phase 6: User Story 4 - Apontar a war room para a API certa (Priority: P2)

**Goal**: a engrenagem abre as configurações. O endereço da API é validado, testado, salvo e restaurável, e o tema pode ser escolhido.

**Independent Test**: um endereço inválido é rejeitado inline. Um válido é aceito, testado e lembrado depois de recarregar a página. "Restaurar padrão" funciona.

### Tests for User Story 4

- [X] T060 [P] [US4] Em `web/src/lib/settings.test.ts`, acrescentar `settingsFormReducer`: `edit`, `submit` com URL inválida (erro de campo, mantendo `saved`), `submit` válido (`saved` atualizado e `connection: "testing"`), `pingResult(true|false)` → `connected` ou `unreachable`, e `restoreDefault` → `DEFAULT_API_URL` no campo.

### Implementation for User Story 4

- [X] T061 [US4] Em `web/src/lib/settings.ts`, criar `SettingsFormState` e `settingsFormReducer` (puro). Em `web/src/api/client.ts`, criar `ping(apiUrl): Promise<boolean>`: `GET {apiUrl}/stats?since=1h` com timeout de 5s, `true` só em 200 e nunca lança (research.md item 11).
- [X] T062 [US4] Criar `web/src/components/SettingsPanel.tsx` sobre o `Dialog`, com o título `h2` "Configurações":
  - campo `type="url"` "Endereço da API" com texto de ajuda e erro via `aria-describedby`;
  - "Salvar" (primário), "Restaurar padrão" e "Cancelar";
  - status da conexão em `role="status"`: "Testando…", "✓ Conectado" ou "✕ Não foi possível conectar";
  - `fieldset` de tema com rádios "Seguir o sistema / Claro / Escuro".
- [X] T063 [US4] Integrar em `web/src/App.tsx`:
  - a engrenagem do `Header` abre o `SettingsPanel`;
  - ao salvar, `saveSettings` e depois `ping`; a mudança de tema aplica `data-theme` na hora e persiste;
  - a ação `open_settings` do `ErrorItem` abre o painel.

**Checkpoint**: a seção 6 do [quickstart.md](./quickstart.md) passa.

---

## Phase 7: User Story 5 - Caminho próprio e API de outra origem (Priority: P3)

**Goal**: a war room funciona publicada sob `/opspilot/` e fala com uma API de outra origem só quando autorizada.

**Independent Test**: `npm --prefix web run build && npm --prefix web run preview`, com a origem do preview na allowlist → tudo carrega e funciona, inclusive ao recarregar a página. Uma origem fora da lista → erro com o atalho para as configurações.

- [X] T064 [P] [US5] Criar `web/src/lib/base-path.test.ts` (puro, usando o HTML gerado). Depois de `npm --prefix web run build`, o teste lê `web/dist/index.html` e verifica que todo `src`/`href` de asset começa com `/opspilot/`. Se `dist/` não existir, o teste é pulado (`t.skip`) com a instrução de rodar o build primeiro.
- [X] T065 [US5] Em `web/src/lib/errors.ts` e `errors.test.ts`, deixar explícito no `detail` do erro de rede/CORS: "Confira o endereço da API nas configurações e se esta origem (`<location.origin>`) está em OPSPILOT_CORS_ORIGINS". A origem entra como parâmetro, para manter a função pura.
- [X] T066 [US5] Documentar `OPSPILOT_CORS_ORIGINS` e `OPSPILOT_APPROVAL_TTL_MS` (com os padrões) em `specs/003-chat-endpoint/quickstart.md`, numa nota "015", e em `CLAUDE.md`, na seção de Comandos e env.

**Checkpoint**: as seções 3 e 7 do [quickstart.md](./quickstart.md) passam.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T067 [P] Revisar `web/src/styles/*.css` e os componentes contra `.github/instructions/design.instructions.md`:
  - nenhum valor fora da escala de espaço e nenhuma cor hex fora de `tokens.css` (`grep -nE '#[0-9a-fA-F]{3,6}|[0-9]+px' web/src --include='*.tsx' --include='app.css'`);
  - um `h1` só e níveis de título sem pular.
- [ ] T068 [P] **Pendente (verificação visual)**: o CSS já foi escrito para isso (coluna fluida, `overflow-wrap`, diálogo em tela cheia abaixo de 768px, alvos de `--target-min`), mas ninguém abriu a tela a 360px ainda. Checagem de responsividade a 360px: sem rolagem horizontal, painéis em tela cheia e alvos de 44px. Ajustar `web/src/styles/app.css` se preciso.
- [X] T069 Rodar `npm run typecheck`, `npm test`, `npm --prefix web run typecheck`, `npm --prefix web test` e `npm --prefix web run build`. Todos precisam ficar verdes (Princípio V).
- [ ] T070 **Pendente**: só o smoke test automatizado foi feito (CORS permitido/negado, `/stats`, 404 de aprovação, war room e assets servidos sob `/opspilot/` no `vite preview`); o fluxo com o modelo exige `OPENROUTER_API_KEY`. Validar manualmente todo o [quickstart.md](./quickstart.md), incluindo o axe DevTools nos dois temas (0 violações AA) e o fluxo da seção 5 só com teclado. Isso exige `OPENROUTER_API_KEY` no ambiente, sem nunca ler `.env`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup**: T001 → T002. T003 vem antes de T004–T006 (pacote criado e instalado). A T007 é independente.
- **Foundational**: depende do Setup.
  - CORS: T008 ∥ T009 → T010 → T011.
  - Web: T012 → T013. T014 → T015. T016 → T017 → T018. T013 + T015 → T019. T020 ∥ T021 ∥ T022. T017 + T018 → T023.
  - Bloqueia todas as stories.
- **US1**: T024 → T025. A T026 depende da T013. T027–T031 em paralelo. T032 → T033 → T034.
- **US2**: depende da US1 (`MessageItem` e `ChatRun`). T035 → T036 → T037 → T038 → T039.
- **US3**: a parte da API é independente do web até a T052. T040 ∥ T041 ∥ T042 → T043 → T044 → T045 → T046 → T047 → T048 → T049 → T050 → T051. T053 → T052. Na parte web: T053 ∥ T054 → T055 → T056 → T057 → T058 → T059. Depende da US1 (conversa) e usa o `ReasoningPanel` da US2 para o "Ver raciocínio" do cartão.
- **US4**: depende só da Foundational e do `Header` (T034). T060 → T061 → T062 → T063.
- **US5**: depende da Foundational (CORS) e da US1 (o fluxo a validar). A T064 depende de existir um build.
- **Polish**: depois de todas as stories.

### User Story Dependencies

- **US1 (P1)**: só Foundational. Entrega o MVP.
- **US2 (P1)**: usa o `MessageItem` e o `ChatRun` da US1.
- **US3 (P2)**: a API só depende da Foundational. O web depende da US1 e, para o "Ver raciocínio" do cartão, da US2.
- **US4 (P2)**: só da Foundational + T034. Pode rodar em paralelo à US2 e à US3.
- **US5 (P3)**: Foundational + US1.

## Parallel Opportunities

- **Setup**: T004 ∥ T005 ∥ T006 ∥ T007.
- **Foundational**: o bloco CORS (T008–T011) ∥ o bloco web (T012–T023). Dentro do web, os pares teste/implementação T012/T014/T016 e os estilos T020 ∥ T021 ∥ T022.
- **US1**: T024 ∥ T026 ∥ T027 ∥ T028 ∥ T029 ∥ T030 ∥ T031.
- **US3**: T040 ∥ T041 ∥ T042. A API inteira (T040–T051) ∥ a US2 e a US4 no web.
- **Polish**: T067 ∥ T068.

### Parallel Example: User Story 3

```bash
Task: "T040 [US3] testes do domínio de aprovação em src/domain/approval.test.ts"
Task: "T041 [US3] testes do SqliteApprovalStore em src/store/sqlite-approval-store.test.ts"
Task: "T042 [US3] testes da porta de aprovação em src/agents/approval-gate.test.ts"
Task: "T054 [US3] testes da máquina do cartão em web/src/lib/approval-machine.test.ts"
```

### Parallel Example: User Story 1

```bash
Task: "T024 [US1] testes do chatReducer em web/src/lib/conversation.test.ts"
Task: "T026 [US1] teste de contrato API↔web em src/http/web-contract.test.ts"
Task: "T027 [US1] EmptyState em web/src/components/EmptyState.tsx"
Task: "T029 [US1] MessageItem em web/src/components/MessageItem.tsx"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Setup (T001–T007): constitution 1.2.0 e o pacote `web/`.
2. Foundational (T008–T023): CORS, schemas, cliente, erros, configurações base e tokens.
3. US1 (T024–T034): chat de ponta a ponta.
4. **STOP and VALIDATE**: os cenários 4.1–4.3 e 4.6 do quickstart.

### Incremental Delivery

Foundational → US1 (chat) → US2 (raciocínio) → US3 (aprovação: primeiro a API, depois o cartão) → US4 (engrenagem) → US5 (publicação) → Polish. Cada passo deixa os quatro gates (`typecheck` e `test` da API e do web) verdes e cabe em commits pequenos (Princípio VIII).

### Nota de segurança

Até a US3 terminar, o `/chat` continua executando `open_incident`/`resolve_incident` direto, como hoje. Se a war room for usada por outras pessoas antes disso, priorizar a parte da API da US3 (T040–T051) logo depois da Foundational.
