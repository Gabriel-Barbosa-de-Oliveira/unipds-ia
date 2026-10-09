# Implementation Plan: War Room Web

**Branch**: `015-war-room-web` | **Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/015-war-room-web/spec.md`

## Summary

**War room (`web/`).** É uma SPA de uma tela em Vite + React + TypeScript, publicada sob `/opspilot/`. A tela tem chat com o `/chat`, um painel "ver raciocínio" que mostra o trace com formato próprio por tipo de evento, um cartão Aprovar/Negar quando a API responde 202, e uma engrenagem para configurar o endereço da API e o tema. Toda a lógica (schemas zod da API, reducer da conversa, máquina do cartão, tradução de erros, modelo de exibição do trace, validação de URL) fica em funções puras testadas com `node:test`. Os componentes React são finos. O visual segue `.github/instructions/design.instructions.md`.

**API.**
1. **Porta de aprovação.** `open_incident` e `resolve_incident` passam a só registrar a proposta num `ApprovalGate` por requisição, sem executar. O `/chat` responde 202 com a ação pendente gravada em `pending_actions`.
2. **`POST /approvals/:id`.** Aplica a decisão de forma atômica e determinística, sem chamar o modelo, e responde no mesmo formato do 200 do `/chat`.
3. **CORS.** Um middleware próprio com allowlist exata (`OPSPILOT_CORS_ORIGINS`) que expõe o `X-Request-Id`.

Os detalhes estão em [research.md](./research.md).

## Technical Context

**Language/Version**: TypeScript `strict` (ESM). A API roda em Node 24 LTS e a war room em navegadores evergreen. A build e os testes do `web/` também rodam em Node 24.

**Primary Dependencies**:
- **API**: Express ^4.19, zod ^3.23, `node:sqlite`, `node:crypto`. Nenhuma dependência nova.
- **Web** (pacote novo `web/package.json`): `react` ^19, `react-dom` ^19, `zod` ^3.23. Em dev: `vite` ^7, `@vitejs/plugin-react`, `typescript` ^5.5, `@types/react`, `@types/react-dom`, `tsx`.

**Storage**: na API, uma tabela nova `pending_actions` no SQLite (`OPSPILOT_DB`), descrita em [data-model.md](./data-model.md). No web, `localStorage` só para `apiUrl` e `theme`.

**Testing**: `node:test` via `tsx` nos dois pacotes. Na API, o store roda em `:memory:` e as estratégias são fakes que chamam as ferramentas recebidas. O teste de contrato `src/http/web-contract.test.ts` valida as respostas reais do `createApp` contra os schemas do `web/`. No web, só os módulos puros de `web/src/lib/` têm teste automatizado. A UI é validada pelo [quickstart.md](./quickstart.md), incluindo o axe.

**Target Platform**: API em servidor Node. War room em navegadores evergreen (Chrome, Firefox, Safari e Edge, nas duas últimas versões), de 360px até desktop.

**Project Type**: web-service + aplicação web (frontend separado em `web/`)

**Performance Goals**: interface pronta em até 2s numa conexão comum (SC-007). Bundle JS gzip menor que 100 kB, já que só React e zod são dependências de runtime.

**Constraints**:
- Sem `*` no CORS e sem credenciais.
- Nenhuma ação que muda a produção roda sem decisão humana registrada.
- Resposta do modelo exibida como texto puro, sem HTML nem Markdown.
- WCAG 2.1 AA.
- O `.env` não é lido.

**Scale/Scope**:
- API: 5 módulos novos (`domain/approval.ts`, `domain/cors.ts`, `services/approval-store.repository.ts`, `store/sqlite-approval-store.ts`, `agents/approval-gate.ts`), 1 rota nova, `server.ts`, `agents/index.ts`, `agents/types.ts`, `domain/errors.ts` e `obs/logger.ts` alterados.
- Web: cerca de 8 módulos puros e cerca de 10 componentes.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Status | Como |
|---|---|---|
| I. Camadas explícitas | ✅ | API: as regras da aprovação (estado efetivo, resumo, resposta, validação de args) ficam em `domain/approval.ts`, que é puro. O contrato fica em `services/approval-store.repository.ts`, o adaptador SQLite em `store/` e a orquestração no controller (`server.ts`). Web: `lib/` (modelo puro) → `api/` (IO com `fetch`) → `components/` (visão). |
| II. Validação na fronteira | ✅ | O corpo de `/approvals/:id` e o `:id` são validados com zod. Os `args` guardados são revalidados com os shapes de `tools.ts` antes de executar. No web, toda resposta da API e toda entrada de configuração passam por zod antes de virar estado. |
| III. Erros de domínio | ✅ | `ApprovalNotFoundError`, `ApprovalAlreadyDecidedError` e `ApprovalExpiredError` são traduzidos para 404, 409 e 410 só no middleware de erro. No web, `toUiError` é a única tradução de falha para mensagem. |
| IV. Funções puras | ✅ | `effectiveStatus`, `summarizeAction`, `approvalAnswer`, `approvalTrace`, `corsHeadersFor`, `parseAllowedOrigins`, `chatReducer`, `approvalMachine`, `toTraceView`, `toUiError`, `normalizeApiUrl` e `resolveTheme` são puras. O IO fica no store, no `fetch` e no `localStorage`. |
| V. Teste obrigatório | ✅ | Há suítes novas para domínio, store, porta, CORS, rota de aprovação, 202 no `/chat`, contrato API↔web e cada módulo puro do web. O `npm test` da raiz e o do `web/` ficam verdes. |
| VI. Segurança | ✅ | Este é o ponto central da feature: ações destrutivas passam por uma guarda estrutural (porta + decisão atômica), não pelo prompt. O CORS usa allowlist exata. Os logs de aprovação não levam `args` nem `reason`. A resposta do modelo nunca é interpretada como HTML. |
| VII. Spec antes de código | ✅ | spec → plan → tasks. A clarificação da FR-019 está registrada na spec. |
| VIII. Pequeno e reversível | ✅ | Incrementos: amendment → CORS → porta + 202 → `/approvals` → esqueleto do web → chat → raciocínio → aprovação → configurações → polimento de acessibilidade. |
| Stack obrigatória | ⚠️ → ✅ | A constitution não prevê frontend. Isso é resolvido com o **amendment MINOR 1.2.0** como primeira tarefa (research.md item 15), antes de qualquer código em `web/`. A API não ganha dependência. |

**Re-check pós-design (Fase 1)**: os contratos e o modelo de dados mantêm os princípios acima. Nenhuma violação nova. O único item condicionado é o amendment da stack, que está registrado em Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/015-war-room-web/
├── spec.md
├── plan.md              # este arquivo
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── http.md          # CORS, /chat 202, POST /approvals/:id
│   └── web-ui.md        # layout, estados, painéis, a11y
├── checklists/requirements.md
└── tasks.md             # /speckit.tasks
```

### Source Code (repository root)

```text
src/                                   # API (existente)
├── agents/
│   ├── approval-gate.ts               # NOVO: ApprovalGate + createGatedOpsTools(store, gate)
│   ├── approval-gate.test.ts
│   ├── index.ts                       # resolveStrategy aceita baseTools (padrão opsTools)
│   └── types.ts                       # GraphNode += "aprovacao"
├── domain/
│   ├── approval.ts                    # NOVO (puro): PendingAction, effectiveStatus, summarizeAction,
│   │                                  #   approvalAnswer, approvalTrace, reasonFromTrace
│   ├── approval.test.ts
│   ├── cors.ts                        # NOVO (puro): parseAllowedOrigins, corsHeadersFor
│   ├── cors.test.ts
│   └── errors.ts                      # + ApprovalNotFound/AlreadyDecided/Expired
├── services/
│   └── approval-store.repository.ts   # NOVO: interface ApprovalStore
├── store/
│   ├── sqlite-approval-store.ts       # NOVO: tabela pending_actions, decide() atômico
│   └── sqlite-approval-store.test.ts
├── obs/logger.ts                      # + approval.requested / approval.decided
└── http/
    ├── server.ts                      # CORS, 202 no /chat, POST /approvals/:id, erros novos
    ├── server.test.ts                 # + cenários 202 / approvals / CORS
    └── web-contract.test.ts           # NOVO: respostas reais validadas pelos schemas do web/

web/                                   # NOVO pacote
├── package.json                       # dev, build, preview, test, typecheck
├── tsconfig.json
├── vite.config.ts                     # base: "/opspilot/"
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx                        # composição: header, conversa, composer, painéis
    ├── lib/                           # PURO + testes node:test
    │   ├── api-schemas.ts             # zod: ChatOk, ChatAwaitingApproval, Decision, TraceEvent, ApiError
    │   ├── api-schemas.test.ts
    │   ├── trace-view.ts              # toTraceView(event) → { kind, label, icon, node, body, long }
    │   ├── trace-view.test.ts
    │   ├── conversation.ts            # chatReducer + ações
    │   ├── conversation.test.ts
    │   ├── approval-machine.ts        # transições do cartão
    │   ├── approval-machine.test.ts
    │   ├── errors.ts                  # toUiError(status, body | exception)
    │   ├── errors.test.ts
    │   ├── settings.ts                # normalizeApiUrl, SettingsSchema, resolveTheme
    │   └── settings.test.ts
    ├── api/
    │   ├── client.ts                  # postChat, postDecision, ping (fetch + AbortController + zod)
    │   └── storage.ts                 # localStorage com try/catch
    ├── components/
    │   ├── Header.tsx  Composer.tsx  ConversationLog.tsx  EmptyState.tsx
    │   ├── MessageItem.tsx  ErrorItem.tsx  ApprovalCard.tsx  ThinkingIndicator.tsx
    │   ├── Dialog.tsx                 # focus trap, Esc, retorno de foco
    │   ├── ReasoningPanel.tsx  TraceEventItem.tsx  SettingsPanel.tsx
    └── styles/
        ├── tokens.css                 # escala de espaço, tipografia, cores semânticas light/dark
        └── app.css
```

**Structure Decision**: a API continua com o layout atual em `src/` (projeto único, MVC por pastas). A war room é um **pacote npm separado em `web/`** (research.md item 7), com `lib/` puro, `api/` de IO e `components/` de visão. Os dois só compartilham o contrato HTTP, protegido pelo `web-contract.test.ts`. Os scripts da raiz não mudam. O `CLAUDE.md` passa a documentar `npm --prefix web run dev|build|test|typecheck`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Stack de frontend (Vite + React) fora da stack obrigatória | Pedido explícito da feature. A war room é uma interface interativa (chat, painéis, estados) | HTML estático com JS puro: reimplementaria estado e componentes à mão, com mais código e menos testável. Resolvido por amendment 1.2.0, não ignorado. |
| Segundo `package.json` (`web/`) | Isola as dependências e o build do frontend da API | Workspace npm na raiz: acopla o lockfile e as instalações da API ao front sem ganho. |
| Resposta da decisão sem chamar o modelo (perde os passos seguintes do plano) | Garante que roda exatamente o que foi aprovado (Princípio VI) | Reexecutar o grafo com pré-aprovação: o modelo pode escolher outros argumentos e gerar 202 em loop. |
