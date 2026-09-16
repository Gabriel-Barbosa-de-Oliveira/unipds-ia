# Implementation Plan: Refletor de Aprendizado

**Branch**: `008-learning-reflector` | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/008-learning-reflector/spec.md`

## Summary

Adiciona aprendizado automático de fatos duráveis ao copiloto, sobre a infraestrutura de memória semântica já existente (`007-semantic-memory`): depois de cada resposta de `POST /chat` com `userId` informado, um novo módulo (`src/memory/learning-reflector.ts`) chama `createModel().withStructuredOutput({ hasLearning, fact })` sobre a última mensagem da pessoa e, quando um fato durável elegível é identificado (nunca um pedido pontual, nunca um segredo — julgamento do próprio modelo via prompt), registra-o via `MemoryStore.remember` — o mesmo método já usado pela tool `remember_fact`. O disparo não é aguardado pela resposta HTTP (fire-and-forget, com absorção total de falhas) e só ocorre quando `userId` está presente, mesma precondição já usada por `recall`/`remember_fact`/`forget_fact`. Nenhuma tool nova é criada: `forget_preference`, citado no pedido original, é servido pela tool `forget_fact` já existente, que já remove qualquer fato — automático ou manual — pela mesma via (ver [research.md](./research.md) item 2).

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS (mesmo runtime das features 001–007; nenhuma mudança de runtime)

**Primary Dependencies**: nenhuma nova — reaproveita `@langchain/core`/`@langchain/openai` (já usados por `agents/model.ts`/`agents/reflection.ts` para `withStructuredOutput`) e `zod` (schema do veredito), mesmo padrão de `reflection.ts`.

**Storage**: SQLite via `node:sqlite`, mesma tabela `memories` de `007-semantic-memory` — nenhuma tabela nova, nenhuma migração (research.md item 5). O único efeito de escrita continua sendo `MemoryStore.remember(userId, fact)`, já existente.

**Testing**: `node:test` via `tsx` (`npm test`). Novo: `src/memory/learning-reflector.test.ts` (lógica de decisão de `reflectAndRemember` com `distillFn` fake injetado — mesmo padrão de `reflection.test.ts#runReflectionLoop`, sem exercitar `distillLearning`/`createModel` diretamente, research.md item 4). Estendido: `src/http/server.test.ts` (refletor disparado só com `userId` presente; resposta não bloqueada mesmo com um fake de refletor que nunca resolve — mesmo padrão de `neverResolvingFake` já usado para a estratégia; falha do refletor injetado não vira erro HTTP nem afeta a resposta).

**Target Platform**: processo Node.js server-side (mesmo runtime das features anteriores) — `npm run dev` passa a disparar, em background, uma chamada de modelo extra por requisição de `/chat` com `userId`; `npm run bench`/`npm run arena` não passam por `/chat` e ficam fora do escopo (mesmo raciocínio de `006`/`007`).

**Project Type**: projeto único — 1 arquivo novo (`src/memory/learning-reflector.ts`) mais extensões pontuais a `src/http/server.ts`; nenhum diretório novo.

**Performance Goals**: a chamada de destilação (`withStructuredOutput`) roda em paralelo à resposta já entregue — não soma à latência percebida pela pessoa usuária (SC-003); soma, sim, uma chamada de modelo adicional por requisição de `/chat` com `userId` (custo/quota do provedor), aceito como parte do pedido original ("após cada resposta").

**Constraints**: refletor só roda com `userId` presente (research.md item 1); resposta HTTP nunca aguarda o refletor (FR-005); falha do refletor é sempre absorvida, nunca vira erro HTTP nem log ruidoso de crash (FR-006, research.md item 3); classificação de "fato durável" vs. "pedido pontual" vs. "segredo" é delegada ao modelo via prompt — nenhuma heurística determinística no domínio (FR-002, FR-003, research.md item 6); nenhuma tool nova é exposta ao modelo (research.md item 2).

**Scale/Scope**: 1 arquivo novo (`src/memory/learning-reflector.ts`), 1 arquivo de teste novo, 1 campo novo opcional em `CreateAppOptions` (`reflectAndRemember`), ~10 linhas novas no handler de `POST /chat` — nenhuma tabela nova, nenhuma tool nova, nenhum campo novo em `ChatRequestSchema`/resposta HTTP, nenhuma dependência de pacote nova.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | `src/memory/learning-reflector.ts` é a única camada que fala com o modelo de destilação e decide chamar `store.remember`; `src/http/server.ts` só orquestra o disparo (fire-and-forget, condicionado a `userId`), sem lógica de decisão própria embutida no controller. |
| II. Validação na Fronteira | Nenhum campo novo de requisição HTTP — `reflectAndRemember` consome `parsed.data.message`, já validado por `ChatRequestSchema` (`007`); o schema zod de `LearningVerdict` (`hasLearning`/`fact`) valida a saída estruturada do modelo, mesma fronteira já usada por `verdictSchema` em `reflection.ts`. |
| III. Erros de Domínio | Nenhuma classe de erro nova — falhas do refletor nunca cruzam a borda HTTP (absorvidas internamente, FR-006, research.md item 3), mesmo espírito de "nada encontrado" já modelado como valor em `007` (`{ removed: false }`, `[]`). |
| IV. Funções Puras | A decisão central de `reflectAndRemember` ("se `hasLearning` e houver `fact`, chama `store.remember`; senão, não") é isolada do IO de `distillFn`, que é injetável — mesmo padrão de `runReflectionLoop` recebendo `critiqueFn` por parâmetro; todo efeito colateral real (chamada ao modelo, escrita no store) fica encapsulado em `distillLearning`/`MemoryStore.remember`, já existentes ou explicitamente isolados. |
| V. Teste Obrigatório | `learning-reflector.test.ts` cobre a lógica de decisão com `distillFn` fake (hasLearning true/false, fact ausente, falha de `distillFn`, falha de `store.remember` — todas absorvidas); `server.test.ts` cobre o disparo condicionado a `userId` e o não-bloqueio da resposta (fake que nunca resolve) — nenhum teste depende de rede ou de `OPENROUTER_API_KEY`. |
| VI. Segurança por Padrão | FR-003 (nunca registrar segredo) é responsabilidade do prompt de `distillLearning`, delegando ao julgamento do modelo (research.md item 6) — mesma abordagem do pedido original; nenhum código desta feature lê `.env` diretamente; nenhum log grava o texto bruto de mensagens além do já feito hoje pelo resto do sistema. |
| VII. Spec Antes de Código | Este plano segue `specs/008-learning-reflector/spec.md`, validado e sem `[NEEDS CLARIFICATION]` pendente. |
| VIII. Pequeno e Reversível | Maior decisão de `research.md` (item 2) é reaproveitar a tool `forget_fact` já existente em vez de criar uma `forget_preference` duplicada — reduz o raio de mudança a 1 arquivo novo + edições pontuais em `server.ts`/`server.test.ts`, nenhuma tabela nova, nenhuma tool nova, nenhuma mudança de contrato HTTP observável. |

Nenhuma violação identificada — **Complexity Tracking** não se aplica (tabela deixada vazia).

## Project Structure

### Documentation (this feature)

```text
specs/008-learning-reflector/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md         # Phase 1 output (/speckit-plan command)
├── quickstart.md         # Phase 1 output (/speckit-plan command)
├── contracts/            # Phase 1 output (/speckit-plan command)
│   ├── post-chat.md
│   └── learning-reflector.md
└── tasks.md              # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── memory/
│   ├── learning-reflector.ts       # [NOVO] LearningVerdict (zod), distillLearning(message), reflectAndRemember(store, userId, message, distillFn?)
│   ├── learning-reflector.test.ts  # [NOVO] reflectAndRemember com distillFn fake — sem rede
│   ├── memory-store.ts             # [existente, inalterado] MemoryStore, SqliteMemoryStore, createMemoryTools (remember_fact/forget_fact reaproveitados)
│   └── embeddings.ts               # [existente, inalterado]
├── agents/
│   ├── model.ts                    # [existente, inalterado] createModel() — reaproveitado por learning-reflector.ts
│   ├── reflection.ts, react.ts, plan-and-execute.ts, types.ts, tools.ts, index.ts  # [existentes, inalterados]
├── http/
│   ├── server.ts                   # [alterado] CreateAppOptions + reflectAndRemember; dispara reflect (fire-and-forget, condicionado a userId) após montar `result`, antes de responder
│   └── server.test.ts              # [alterado] + cenários: refletor disparado só com userId, resposta não bloqueada (fake nunca resolve), falha do refletor não vira erro HTTP
└── domain/, store/, services/, mcp/, bench.ts, arena.ts  # [existentes, inalterados]
```

**Structure Decision**: Projeto único (Option 1), mesma estrutura das features 001–007. Único arquivo novo de produção: `src/memory/learning-reflector.ts` — vive em `src/memory/` (não em `src/agents/`) porque seu colaborador principal é `MemoryStore` (mesmo diretório de `memory-store.ts`), ainda que reaproveite `createModel()` de `agents/` (mesma relação cross-diretório que `bench.ts`/`arena.ts` já têm com `agents/`). Nenhuma tool nova, nenhuma tabela nova, nenhum diretório novo.
