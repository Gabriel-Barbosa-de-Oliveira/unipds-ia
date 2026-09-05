# Implementation Plan: Endpoint HTTP de Chat

**Branch**: `003-chat-endpoint` | **Date**: 2026-09-03 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-chat-endpoint/spec.md`

## Summary

Primeiro endpoint HTTP do projeto: `POST /chat` recebe `{ message, strategy?, reflect? }` (validado com zod na fronteira), resolve `strategy` contra um registro central de estratégias de raciocínio já existentes (`react` padrão, `plan-and-execute`), aplica opcionalmente a camada de reflection (`withReflection`, feature 002) já disponível sobre a estratégia resolvida quando `reflect: true`, executa com um teto de 180s, e responde `200 { answer, trace, metrics }` reaproveitando o `RunResult` já definido pela feature 001 sem nenhuma transformação. Erros de fronteira (corpo inválido → 400, estratégia desconhecida → 422) e de tempo excedido (→ 504) são modelados como falhas previsíveis e traduzidos para status HTTP somente no controller, nunca dentro da resolução de estratégia ou da execução em si. A resolução de nome→estratégia (`src/agents/index.ts`) e a execução com timeout (`src/services/chat.service.ts`) são totalmente injetáveis na criação do app Express, permitindo um teste de integração determinístico do endpoint com uma estratégia fake e sem nenhuma chamada de rede.

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS (mesmo runtime das features 001/002)

**Primary Dependencies**: `express` (já presente no `package.json`, servidor HTTP mandatado pela constitution — nenhuma versão nova), `zod` (validação do corpo da requisição na fronteira, mesmo padrão já usado em `arena.ts`/`reflection.ts`) — reaproveita `src/agents/react.ts`, `plan-and-execute.ts` e `reflection.ts` das features 001/002 sem nenhuma alteração; nenhuma dependência nova é introduzida

**Storage**: N/A — o endpoint não lê nem escreve estado operacional diretamente; as tools de domínio (`open_incident`, etc.) continuam encapsuladas inteiramente dentro das estratégias já existentes, inalteradas por esta feature

**Testing**: `node:test` via `tsx` (`npm test`); o teste de integração do endpoint sobe `createApp(...)` com um resolvedor de estratégia **totalmente injetado** (fake determinística — nunca chama `withReflection`/`createModel` reais) e um `timeoutMs` pequeno e configurável, em porta efêmera (`listen(0)`), usando `fetch` nativo do Node — nenhuma dependência de teste HTTP nova (ex.: `supertest`) é adicionada; `resolveStrategy` (`src/agents/index.ts`) e `runWithTimeout` (`src/services/chat.service.ts`) também ganham testes unitários próprios com fakes

**Target Platform**: processo Node.js server-side (dev local/CI) — primeira superfície HTTP real do projeto; `npm run dev` passa a de fato subir um servidor ouvindo requisições

**Project Type**: projeto único (extensão aditiva de `src/agents/`, novos `src/http/` e `src/services/`)

**Performance Goals**: não é caminho de alto throughput — mesma natureza de custo por chamada ao modelo já documentada em 001/002; o que importa é o teto de 180s por requisição ser sempre respeitado do ponto de vista do cliente (FR-008)

**Constraints**: `timeoutMs` tem padrão 180000 mas DEVE ser configurável na criação do app (necessário para testes rápidos e determinísticos, FR-008); corpo inválido e estratégia desconhecida NUNCA disparam execução de raciocínio (FR-002, FR-004); cada requisição é isolada — nenhum estado é compartilhado entre requisições concorrentes, cada chamada a `strategy.run(...)` é independente (FR-010); a tradução de erro de domínio para status HTTP acontece exclusivamente no controller (`src/http/server.ts`), nunca em `agents/index.ts` ou `chat.service.ts`

**Scale/Scope**: primeiro endpoint HTTP do projeto; expõe as 2 estratégias base já existentes (`react`, `plan-and-execute`) mais a decoração opcional de reflection sobre qualquer uma delas — nenhuma estratégia de raciocínio nova é criada por esta feature

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | `src/http/server.ts` (controller) só valida, delega e traduz erro→status; `src/agents/index.ts` resolve nome→estratégia (nenhuma IO própria, só compõe `ReasoningStrategy`s já existentes); `src/services/chat.service.ts` é a única camada que introduz o efeito de tempo (timeout) sobre a execução. Nenhuma dessas camadas repete lógica de raciocínio já implementada em `react.ts`/`plan-and-execute.ts`/`reflection.ts`. |
| II. Validação na Fronteira | O corpo de `POST /chat` é validado com zod (`ChatRequestSchema: { message, strategy?, reflect? }`) antes de qualquer resolução de estratégia ou execução — corpo inválido nunca alcança `resolveStrategy` nem `strategy.run(...)` (FR-002). |
| III. Erros de Domínio | `UnknownStrategyError` e `ChatTimeoutError` (novas classes em `src/domain/errors.ts`, mesmo padrão de `ServiceNotFoundError`/`IncidentNotFoundError`) são lançadas por `agents/index.ts`/`chat.service.ts` e traduzidas para 422/504 **somente** no middleware de erro do controller — nenhuma das duas camadas internas conhece HTTP. |
| IV. Funções Puras | `resolveStrategy(name, reflect)` é pura dado o registro estático de estratégias (mesma entrada → mesma estratégia resolvida, sem IO); `runWithTimeout` isola o único efeito colateral real (o timer) atrás de uma função pequena e testável com fakes controláveis por tempo. |
| V. Teste Obrigatório | `agents/index.test.ts` (resolução de nome/erro de estratégia desconhecida/composição com reflection), `services/chat.service.test.ts` (timeout dispara `ChatTimeoutError`, execução rápida não dispara), e `http/server.test.ts` (integração do endpoint completo: 200 com estratégia padrão, 200 com estratégia explícita, 200 com reflect, 400 corpo inválido, 422 estratégia desconhecida, 504 timeout) — todos com fakes injetados, sem rede; `npm test`/`npm run typecheck` continuam gates obrigatórios. |
| VI. Segurança por Padrão | Nenhum segredo novo é introduzido; o endpoint não expõe `.env` nem detalhes de erro interno ao cliente (erro 500 genérico e sem stack trace na resposta; detalhe completo só no log de servidor). |
| VII. Spec Antes de Código | Este plano segue `specs/003-chat-endpoint/spec.md`, já validado e sem `[NEEDS CLARIFICATION]` pendente, e antecede `/speckit-tasks` + `/speckit-implement`. |
| VIII. Pequeno e Reversível | Mudança fica contida em 3 arquivos novos pequenos (`agents/index.ts`, `services/chat.service.ts`, `http/server.ts`) + seus testes, uma extensão pequena de `domain/errors.ts` (2 classes) e o preenchimento de `src/index.ts` (hoje vazio); `src/arena.ts` e todos os arquivos das features 001/002 permanecem intocados (ver research.md item 6). |

Nenhuma violação identificada — **Complexity Tracking** não se aplica (tabela deixada vazia).

## Project Structure

### Documentation (this feature)

```text
specs/003-chat-endpoint/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md         # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── agents/
│   ├── react.ts                      # [existente, inalterado]
│   ├── plan-and-execute.ts           # [existente, inalterado]
│   ├── reflection.ts                 # [existente, inalterado] withReflection reaproveitado como está
│   ├── types.ts                      # [existente, inalterado] ReasoningStrategy/RunResult reaproveitados como resposta HTTP
│   └── index.ts                      # [NOVO] STRATEGIES (react, plan-and-execute) + resolveStrategy(name?, reflect?)
├── services/
│   ├── ops-store.repository.ts       # [existente, inalterado]
│   ├── ops-store.memory.ts           # [existente, inalterado]
│   ├── ops-store.sequelize.ts        # [existente, inalterado]
│   ├── chat.service.ts               # [NOVO] runWithTimeout(strategy, input, options, timeoutMs)
│   └── chat.service.test.ts          # [NOVO]
├── domain/
│   └── errors.ts                     # [alterado] + UnknownStrategyError, + ChatTimeoutError
├── http/                             # [NOVO diretório]
│   ├── server.ts                     # [NOVO] createApp(options?) — rota POST /chat, validação zod, middleware de erro
│   └── server.test.ts                # [NOVO] teste de integração, estratégia fake injetada, sem rede
└── index.ts                          # [alterado, hoje vazio] createApp().listen(PORT)
```

**Structure Decision**: Projeto único (Option 1), mesma estrutura das features 001/002. Dois diretórios novos e pequenos (`src/http/`, mais o arquivo `src/agents/index.ts` como registro) cobrem inteiramente a superfície HTTP; nenhum arquivo de `src/agents/react.ts`, `plan-and-execute.ts`, `reflection.ts` ou `src/arena.ts` é modificado — esta feature apenas expõe, via HTTP, capacidades que essas features já entregaram.

## Complexity Tracking

*Nenhuma violação da Constitution Check — tabela não aplicável.*
