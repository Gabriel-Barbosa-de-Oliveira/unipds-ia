# Implementation Plan: Conversa Persistente

**Branch**: `006-conversation-history` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/006-conversation-history/spec.md`

## Summary

Adiciona continuidade de conversa ao `POST /chat`: uma nova classe `SqliteConversationStore` (tabelas `conversations`/`messages`, sobre `node:sqlite`, mesmo padrão de `SqliteOpsStore` da feature 004) persiste os turnos de cada conversa; o corpo da requisição ganha `conversationId` opcional — omitido inicia conversa nova, informado continua uma existente (id desconhecido → `404`); o controller resolve as últimas 12 mensagens da conversa, as compõe com a nova mensagem em uma única string (função pura em `src/domain/conversation.ts`) que passa a ser o `input` já hoje aceito por `ReasoningStrategy.run(...)` — nenhuma mudança em `react.ts`, `plan-and-execute.ts`, `reflection.ts` ou `message-trace.ts` (ver [research.md](./research.md) item 1). Ao final de cada turno, a mensagem crua da pessoa de plantão e a resposta final do copiloto são gravadas na conversa. A resposta `200` passa a incluir `conversationId` e `metrics.historyMessages` (quantidade de mensagens de histórico usadas), sem alterar `answer`/`trace`/`metrics.llmCalls`/`metrics.latencyMs` nem o contrato de erro já existente (`400`/`422`/`504`/`500`); um `404` novo cobre conversa desconhecida.

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS (mesmo runtime das features 001–005; nenhuma mudança de runtime)

**Primary Dependencies**: nenhuma dependência de pacote nova — `node:sqlite` (`DatabaseSync`) já é builtin e já usado por `SqliteOpsStore` (004). `zod` continua o validador na fronteira (`ChatRequestSchema` ganha `conversationId` opcional).

**Storage**: SQLite via `node:sqlite` (`DatabaseSync`) — mesmo arquivo (`OPSPILOT_DB`, default `./data/opspilot.db`) já usado pelos dados operacionais, em uma conexão própria e independente (`SqliteConversationStore`, duas tabelas novas: `conversations`, `messages`); `":memory:"` em testes, mesmo padrão de inicialização lazy da DDL que `SqliteOpsStore`.

**Testing**: `node:test` via `tsx` (`npm test`). Novos: `src/domain/conversation.test.ts` (composição pura do prompt), `src/store/sqlite-conversation-store.test.ts` (`create`/`append`/`lastMessages` sobre `":memory:"`: limite de 12, `ConversationNotFoundError`, isolamento entre conversas). Estendidos: `src/http/server.test.ts` (conversa nova vs. continuada, `404` para id desconhecido, `metrics.historyMessages`, isolamento via um `ConversationStore` fake injetável — mesmo padrão de `resolveStrategy` fake já usado nesse arquivo). Nenhum teste depende de rede nem de `data/opspilot.db`.

**Target Platform**: processo Node.js server-side (mesmo runtime das features anteriores) — `npm run dev` passa a operar sobre o novo `SqliteConversationStore` além do `SqliteOpsStore` já existente; `npm run arena`/`npm run bench` são chamadas diretas às estratégias (não passam pelo `/chat`) e ficam fora do escopo desta feature (spec Assumptions).

**Project Type**: projeto único (extensão aditiva de `src/domain/`, `src/services/`, `src/store/`, `src/http/server.ts`; nenhum diretório novo além de arquivos dentro dos já existentes)

**Performance Goals**: mesmo perfil das features anteriores — não é caminho de alto throughput; a única meta observável é que compor/persistir histórico não introduza atraso perceptível frente à latência de uma chamada ao modelo (sem meta numérica própria desta feature, mesmo critério usado em 004).

**Constraints**: histórico usado na composição do prompt é sempre, no máximo, as 12 mensagens mais recentes (FR-004), independentemente do tamanho total da conversa; `conversationId` desconhecido nunca inicia conversa silenciosa — sempre `404` explícito (FR-007); nenhuma mudança de comportamento para requisições sem `conversationId` além do campo adicional na resposta (FR-008); nenhuma query concatena valor de entrada em SQL — só prepared statements, mesmo padrão de `SqliteOpsStore`.

**Scale/Scope**: 2 tabelas novas, 1 classe de store nova (`SqliteConversationStore`) + 1 interface nova (`ConversationStore`), 1 classe de erro nova (`ConversationNotFoundError`), 1 função pura nova (composição de prompt), 1 campo de requisição novo e 2 campos de resposta novos no `/chat`; nenhuma estratégia de raciocínio nova, nenhuma tool nova, nenhuma mudança em `agents/types.ts`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | `SqliteConversationStore` (`src/store/`) é a única camada que fala `node:sqlite` para conversas; `ConversationStore` (`src/services/`) é o contrato consumido pelo controller; a composição de prompt (`src/domain/conversation.ts`) é pura, sem IO; `src/http/server.ts` só traduz `ConversationNotFoundError` para `404`, nunca embute lógica de composição. |
| II. Validação na Fronteira | `ChatRequestSchema` (zod) ganha `conversationId: z.string().optional()`, validado antes de qualquer resolução de conversa ou chamada à estratégia. |
| III. Erros de Domínio | `ConversationNotFoundError` (classe nova) modela "conversa desconhecida"; a tradução para `404` acontece exclusivamente em `errorMiddleware` (`src/http/server.ts`), nunca dentro de `SqliteConversationStore` ou do controller antes desse ponto. |
| IV. Funções Puras | A composição do histórico com a nova mensagem (`src/domain/conversation.ts`) é uma função pura — mesma entrada (histórico + mensagem), mesma saída; todo efeito colateral (leitura/escrita de mensagens) fica isolado em `SqliteConversationStore`. |
| V. Teste Obrigatório | Nenhuma lógica nova (composição de prompt, DDL/CRUD do store de conversas, resolução de `conversationId` no controller, `historyMessages`) entra sem teste — ver Technical Context/Testing; `npm test`/`npm run typecheck` continuam gates obrigatórios. |
| VI. Segurança por Padrão | Mesmo arquivo `OPSPILOT_DB` já coberto por `.gitignore` (`data/`); nenhum código desta feature lê `.env` diretamente; nenhuma ação destrutiva envolvida (conversas nunca são apagadas por esta feature). |
| VII. Spec Antes de Código | Este plano segue `specs/006-conversation-history/spec.md`, validado e sem `[NEEDS CLARIFICATION]` pendente. |
| VIII. Pequeno e Reversível | Cada decisão de research.md (store separado, string composta em vez de mudar a assinatura de `run()`, mensagem crua persistida em vez do prompt composto, `historyMessages` fora de `agents/types.ts`) minimiza o diff sobre código já estável (001–005); nenhuma delas exige tocar `react.ts`, `plan-and-execute.ts`, `reflection.ts`, `message-trace.ts`, `agents/types.ts`, `bench.ts` ou `arena.ts`. |

Nenhuma violação identificada — **Complexity Tracking** não se aplica (tabela deixada vazia).

## Project Structure

### Documentation (this feature)

```text
specs/006-conversation-history/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/            # Phase 1 output (/speckit-plan command)
│   ├── post-chat.md
│   └── conversation-store.md
└── tasks.md              # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── domain/
│   ├── conversation.ts               # [NOVO] ConversationMessage (tipo) + composePrompt(history, input) — pura
│   ├── conversation.test.ts          # [NOVO] testes de composePrompt (histórico vazio, parcial, 12 mensagens)
│   └── errors.ts                     # [alterado] + ConversationNotFoundError
├── services/
│   └── conversation-store.repository.ts   # [NOVO] interface ConversationStore (create/append/lastMessages)
├── store/
│   ├── sqlite-conversation-store.ts       # [NOVO] SqliteConversationStore — tabelas conversations/messages
│   └── sqlite-conversation-store.test.ts  # [NOVO] create/append/lastMessages sobre ":memory:", limite 12, ConversationNotFoundError, isolamento
├── http/
│   ├── server.ts                     # [alterado] ChatRequestSchema + conversationId; CreateAppOptions + conversationStore; resolução de conversa, composição, append pós-turno, resposta com conversationId/historyMessages; errorMiddleware + 404
│   └── server.test.ts                # [alterado] + cenários: conversa nova, conversa continuada, 404 id desconhecido, historyMessages, isolamento (ConversationStore fake injetável)
└── domain/ops-store.ts, agents/*, services/chat.service.ts, store/sqlite-ops-store.ts, index.ts, bench.ts, arena.ts, mcp/*  # [existentes, inalterados]
```

**Structure Decision**: Projeto único (Option 1), mesma estrutura das features 001–005. Nenhum diretório novo — apenas arquivos novos dentro de `src/domain/`, `src/services/` e `src/store/`, seguindo exatamente a divisão Model (`store/`)/contrato (`services/`)/domínio puro (`domain/`) já estabelecida por `004-ops-persistence` para dados operacionais, agora replicada para o bounded context de conversas. `src/mcp/` (feature 005, integração MCP) permanece fora de escopo (spec Assumptions) — nenhum arquivo ali é tocado.

## Complexity Tracking

*Nenhuma violação da Constitution Check — tabela não aplicável.*
