# Implementation Plan: Memória Semântica

**Branch**: `007-semantic-memory` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/007-semantic-memory/spec.md`

## Summary

Adiciona memória de longo prazo, por pessoa (`userId`), ao copiloto: um `SqliteMemoryStore` novo (`src/memory/memory-store.ts`, tabela `memories`) registra (`remember`), recupera (`recall`) e remove (`forget`) fatos em linguagem natural, comparando-os por **significado** — via embeddings locais (`onnx-community/all-MiniLM-L6-v2-ONNX`, `@huggingface/transformers`, pooling mean + normalize, singleton lazy em `src/memory/embeddings.ts`) e produto escalar, nunca por correspondência de texto. `POST /chat` ganha `userId` opcional: quando presente, o controller busca (`recall`, automático, não é tool) até 3 fatos relevantes daquele `userId` e os injeta no prompt antes de chamar a estratégia (função pura `composeWithFacts`, `src/domain/memory.ts`), e disponibiliza duas tools novas ao modelo (`remember_fact`, `forget_fact`, escopadas àquele `userId` por closure — nunca por parâmetro que o modelo preenche). Ausência de `userId` preserva exatamente o comportamento de `006-conversation-history`. `resolveStrategy` (`src/agents/index.ts`) ganha um terceiro parâmetro opcional (`extraTools`) para compor essas tools por requisição, sem tocar `react.ts`/`plan-and-execute.ts`/`reflection.ts` (ver [research.md](./research.md) item 7).

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS (mesmo runtime das features 001–006; nenhuma mudança de runtime)

**Primary Dependencies**: **nova** — `@huggingface/transformers` (transformers.js), única dependência de produto adicionada desde a criação do projeto; roda o modelo ONNX localmente em Node, sem servidor de embedding externo (research.md item 1). `zod` continua o validador na fronteira (`ChatRequestSchema` ganha `userId` opcional).

**Storage**: SQLite via `node:sqlite` (`DatabaseSync`) — mesmo arquivo (`OPSPILOT_DB`) já usado pelas features 004/006, em uma conexão própria e independente (`SqliteMemoryStore`, uma tabela nova: `memories`, com `embedding BLOB`); `":memory:"` em testes. Checkpoint ONNX do modelo de embedding cacheado em `./.cache/transformers` (`.gitignore` novo), fora do SQLite.

**Testing**: `node:test` via `tsx` (`npm test`). Novos: `src/domain/memory.test.ts` (produto escalar, top-3/limiar 0.3, dedup > 0.92, serialização — vetores forjados, instantâneo), `src/memory/embeddings.test.ts` (modelo real: dimensão do vetor, normalização) e `src/memory/memory-store.test.ts` (a maioria com `embed` fake injetado; um teste dedicado com o `embed` real provando recall sem palavra em comum — research.md item 4). Estendidos: `src/agents/index.test.ts` (`resolveStrategy` com `extraTools`) e `src/http/server.test.ts` (recall automático injetado no prompt, tools de memória disponíveis só com `userId`, isolamento entre `userId`s, requisição sem `userId` inalterada frente a `006`).

**Target Platform**: processo Node.js server-side (mesmo runtime das features anteriores) — `npm run dev` passa a operar sobre o novo `SqliteMemoryStore`/`embeddings.ts` além dos stores já existentes; `npm run arena`/`npm run bench` não passam por `/chat` e ficam fora do escopo (mesmo raciocínio de `006`).

**Project Type**: projeto único — um diretório novo, `src/memory/` (pedido explicitamente pelo usuário: `embeddings.ts`, `memory-store.ts`), mais um arquivo de domínio puro novo (`src/domain/memory.ts`) e extensões pontuais a `src/agents/index.ts`/`src/http/server.ts`.

**Performance Goals**: cada `embed()` (inferência local, CPU) custa dezenas a poucas centenas de ms — aceitável frente à latência já dominante de uma chamada ao modelo de raciocínio (segundos); `recall`/`remember`/`forget` fazem no máximo 1 chamada de embedding por invocação mais uma varredura em memória dos vetores já existentes do `userId` (sem índice vetorial dedicado — volume esperado por pessoa não justifica essa complexidade adicional nesta fase).

**Constraints**: dedup só descarta um fato quando o mais parecido já registrado (do mesmo `userId`) tem score `> 0.92` (FR-003); recall/forget nunca consideram fato com score `< 0.3` relevante o suficiente (FR-005, FR-009); fatos de um `userId` nunca influenciam nem são visíveis para outro (FR-007, SC-005); nenhuma query concatena valor de entrada em SQL — só prepared statements, mesmo padrão das features 004/006; requisição sem `userId` é bit-a-bit igual ao contrato de `006` (nenhuma chamada de embedding, nenhuma tool de memória disponível).

**Scale/Scope**: 1 tabela nova (`memories`), 1 dependência de pacote nova, 2 arquivos novos em `src/memory/` + 1 em `src/domain/`, 2 tools novas (`remember_fact`, `forget_fact`), 1 parâmetro novo opcional em `resolveStrategy`, 1 campo de requisição novo (`userId`) no `/chat` — nenhum campo de resposta novo, nenhuma mudança em `agents/types.ts`/`react.ts`/`plan-and-execute.ts`/`reflection.ts`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | `src/memory/embeddings.ts` e `src/memory/memory-store.ts` são a única camada que fala com o modelo de embedding/SQLite para memória; `src/domain/memory.ts` é puro (produto escalar, seleção top-k, serialização); `src/http/server.ts` só orquestra (chama `recall`, compõe o prompt, escolhe as tools), nunca embute lógica de similaridade. |
| II. Validação na Fronteira | `ChatRequestSchema` ganha `userId: z.string().optional()`, validado antes de qualquer recall/tool; `remember_fact`/`forget_fact` têm schema zod próprio (`fact`/`description`, `.describe()`). |
| III. Erros de Domínio | Nenhuma classe de erro nova é necessária — "nada encontrado" (`forget` sem correspondência, `recall` vazio) é modelado como valor (`{ removed: false }`, `[]`), não como erro, mesmo padrão já usado por `getRunbook`/`listIncidents` (FR-005, FR-009). |
| IV. Funções Puras | `dotProduct`, seleção top-k/limiar, `floatArrayToBuffer`/`bufferToFloatArray`, `composeWithFacts` (`src/domain/memory.ts`) são puras — mesma entrada, mesma saída; todo efeito colateral (inferência do modelo, SQLite) isolado em `src/memory/`. |
| V. Teste Obrigatório | Nenhuma lógica nova entra sem teste — ver Technical Context/Testing; a divisão entre testes com `embed` fake (rápidos, maioria) e o único teste com o modelo real (research.md item 4) é uma decisão deliberada para manter `npm test` majoritariamente offline sem abrir mão de provar a promessa central da feature. |
| VI. Segurança por Padrão | `userId` nunca é preenchido pelo modelo (research.md item 6) — elimina por construção o risco de um `userId` errado vazar/apagar fato de outra pessoa; `.cache/transformers` (checkpoint do modelo) e `data/` (SQLite) seguem fora do controle de versão; nenhum código desta feature lê `.env` diretamente. |
| VII. Spec Antes de Código | Este plano segue `specs/007-semantic-memory/spec.md`, validado e sem `[NEEDS CLARIFICATION]` pendente. |
| VIII. Pequeno e Reversível | Cada decisão de research.md (tools por requisição via parâmetro aditivo em `resolveStrategy`, recall automático fora do laço de tools, `userId` só por closure, teste real isolado a um único arquivo) minimiza o raio de mudança sobre `react.ts`/`plan-and-execute.ts`/`reflection.ts`/`agents/types.ts` — nenhum desses é tocado. |

Nenhuma violação identificada — **Complexity Tracking** não se aplica (tabela deixada vazia).

## Project Structure

### Documentation (this feature)

```text
specs/007-semantic-memory/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/            # Phase 1 output (/speckit-plan command)
│   ├── post-chat.md
│   └── memory-store.md
└── tasks.md              # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── domain/
│   ├── memory.ts                     # [NOVO] dotProduct, floatArrayToBuffer/bufferToFloatArray, seleção top-k/limiar, composeWithFacts(facts, input) — tudo puro
│   ├── memory.test.ts                 # [NOVO] testes de memory.ts com vetores forjados
│   └── errors.ts                      # [existente, inalterado] — nenhuma classe de erro nova necessária
├── memory/                            # [NOVO diretório, pedido explicitamente]
│   ├── embeddings.ts                  # [NOVO] embed(text) — singleton lazy do pipeline onnx-community/all-MiniLM-L6-v2-ONNX
│   ├── embeddings.test.ts             # [NOVO] modelo real: dimensão do vetor, normalização
│   ├── memory-store.ts                # [NOVO] interface MemoryStore + SqliteMemoryStore (tabela memories) + createMemoryTools(store, userId)
│   └── memory-store.test.ts           # [NOVO] maioria com embed fake; 1 teste dedicado com embed real (recall sem palavra em comum)
├── agents/
│   ├── index.ts                       # [alterado] resolveStrategy(name?, reflect?, extraTools?) — aditivo, comportamento inalterado quando extraTools é omitido
│   ├── index.test.ts                  # [alterado] + testes de resolveStrategy com extraTools
│   ├── react.ts, plan-and-execute.ts, reflection.ts, types.ts, tools.ts  # [existentes, inalterados]
├── http/
│   ├── server.ts                      # [alterado] ChatRequestSchema + userId; CreateAppOptions + memoryStore; recall automático + composeWithFacts antes de resolveStrategy/runWithTimeout quando userId presente
│   └── server.test.ts                 # [alterado] + cenários: recall injetado no prompt, tools de memória só com userId, isolamento entre userId, requisição sem userId idêntica a 006
└── store/sqlite-ops-store.ts, sqlite-conversation-store.ts, services/*, mcp/*, bench.ts, arena.ts  # [existentes, inalterados]
```

**Structure Decision**: Projeto único (Option 1), mesma estrutura das features 001–006. Único diretório novo: `src/memory/` — pedido explicitamente pelo usuário (`embeddings.ts`, `memory-store.ts`) e coeso (ambos os arquivos só existem em função da memória semântica); a interface `MemoryStore` e a fábrica `createMemoryTools` vivem dentro de `memory-store.ts` em vez de arquivos `services`/`agents` separados, mantendo o footprint nos 2 arquivos pedidos mais o mínimo necessário de lógica pura isolada (`src/domain/memory.ts`, exigido pelo Princípio IV).
