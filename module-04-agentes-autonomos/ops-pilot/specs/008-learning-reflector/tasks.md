---

description: "Task list template for feature implementation"
---

# Tasks: Refletor de Aprendizado

**Input**: Design documents from `/specs/008-learning-reflector/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos — a constitution (Teste Obrigatório, NON-NEGOTIABLE) pede testes explicitamente para toda lógica nova.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivo diferente, sem dependência de tarefa ainda não concluída)
- **[Story]**: A qual user story a tarefa pertence (US1–US3)
- Caminhos de arquivo exatos em cada descrição

## Path Conventions

Projeto único — `src/` na raiz do repositório (ver plan.md § Project Structure). Sem `tests/` separado: testes ficam ao lado do código (`*.test.ts`), convenção já usada pelas features 001–007.

---

## Phase 1: Setup

**Purpose**: Inicialização de projeto.

Nenhuma tarefa de setup necessária nesta feature — nenhuma dependência de pacote nova (research.md: reaproveita `@langchain/core`/`@langchain/openai`/`zod` já presentes), nenhuma tabela nova. A fundação começa diretamente na Phase 2.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: O módulo `learning-reflector.ts` completo (schema do veredito, chamada ao modelo, e a lógica de decisão testável por injeção) — base sobre a qual as 3 user stories se apoiam. Nenhuma delas é testável de ponta a ponta antes desta fase estar completa.

**⚠️ CRITICAL**: Nenhuma user story começa antes desta fase estar completa.

- [X] T001 [P] Criar `src/memory/learning-reflector.ts`: `learningSchema` (zod) `{ hasLearning: boolean, fact?: string }` e `type LearningVerdict = z.infer<typeof learningSchema>`; `async function distillLearning(message: string): Promise<LearningVerdict>` — chama `createModel().withStructuredOutput(learningSchema).invoke([["system", DISTILL_PROMPT], ["user", message]])` (mesmo padrão de `src/agents/reflection.ts#critique`), lançando erro se o modelo não retornar um veredito válido; `DISTILL_PROMPT` instrui: identificar fatos duráveis sobre a pessoa ou seu contexto de trabalho; nunca tratar pedido pontual/pergunta como aprendizado; nunca incluir segredo/credencial em `fact`, mesmo ao lado de um fato genuíno (quando inseparável, `hasLearning: false`) — research.md item 6. `type DistillFn = (message: string) => Promise<LearningVerdict>`; `async function reflectAndRemember(store: MemoryStore, userId: string, message: string, distillFn: DistillFn = distillLearning): Promise<void>` — chama `distillFn(message)`; se `hasLearning === true` e `fact` presente, chama `store.remember(userId, fact)`; **nunca rejeita**: qualquer erro de `distillFn` ou `store.remember` é capturado e vira só um `console.error` de diagnóstico (FR-006, research.md item 3).
- [X] T002 (depende de T001) Criar `src/memory/learning-reflector.test.ts`: testar **apenas** `reflectAndRemember` com um `distillFn` fake injetado (nunca `distillLearning`/`createModel` diretamente — research.md item 4, mesmo padrão de `reflection.test.ts#runReflectionLoop`). Casos: `distillFn` resolve `{ hasLearning: true, fact: "..." }` → `store.remember(userId, fact)` é chamado com os argumentos corretos; `distillFn` resolve `{ hasLearning: false }` → `store.remember` nunca é chamado; `distillFn` resolve `{ hasLearning: true }` sem `fact` → `store.remember` nunca é chamado; `distillFn` rejeita → `reflectAndRemember` não rejeita, `store.remember` nunca é chamado; `distillFn` resolve `hasLearning: true` com `fact`, mas `store.remember` rejeita → `reflectAndRemember` não rejeita. Usar um `MemoryStore` fake simples (mesmo padrão de `fakeMemoryStore` em `src/http/server.test.ts`) para capturar as chamadas a `remember`.

**Checkpoint**: `reflectAndRemember` completo e testado, sem nenhuma rede — as user stories podem começar.

---

## Phase 3: User Story 1 - Aprendizado automático de fatos duráveis a partir da conversa (Priority: P1) 🎯 MVP

**Goal**: `POST /chat`, quando `userId` está presente, dispara `reflectAndRemember` sobre a última mensagem da pessoa depois de calcular a resposta da estratégia, sem aguardar esse disparo antes de responder.

**Independent Test**: Injetar um `reflectAndRemember` fake via `createApp(...)` e confirmar que uma requisição com `userId` o aciona com `(memoryStore, userId, message)` corretos; confirmar que uma requisição sem `userId` nunca o aciona.

- [X] T003 [US1] Estender `src/http/server.ts`: `CreateAppOptions` ganha `reflectAndRemember?: typeof reflectAndRemember` (default: a função real de `src/memory/learning-reflector.ts`). No handler de `POST /chat`, depois que `result` já foi calculado e **antes** de montar/enviar a resposta: se `parsed.data.userId` está presente, `void reflect(memoryStore, userId, parsed.data.message).catch(() => {})` — onde `reflect = options.reflectAndRemember ?? reflectAndRemember` — chamada **não aguardada** (contracts/learning-reflector.md). Quando `userId` está ausente, nada é disparado (comportamento idêntico ao de antes desta feature).
- [X] T004 [P] [US1] (depende de T003) Estender `src/http/server.test.ts`: injetar um `reflectAndRemember` fake determinístico via `createApp({ reflectAndRemember: fake, memoryStore })` que só registra suas chamadas (sem lógica real). Casos: (a) requisição com `userId` e `message` faz o fake ser chamado exatamente uma vez, com `userId` e `message` (o texto desta requisição) corretos; (b) requisição sem `userId` nunca aciona o fake.

**Checkpoint**: User Story 1 completa e testável de forma independente — aprendizado automático disparado via `/chat`. MVP entregável.

---

## Phase 4: User Story 2 - Desfazer um aprendizado automático (Priority: P2)

**Goal**: Confirmar que a tool `forget_fact` já existente (`007-semantic-memory`, disponível sempre que `userId` está presente) remove, pela descrição, um fato gravado por `reflectAndRemember` exatamente como remove um fato ensinado manualmente — sem nenhuma tool nova (research.md item 2).

**Independent Test**: Gravar um fato via `reflectAndRemember` (com um `distillFn` fake que resolve `hasLearning: true`), depois invocar a tool `forget_fact` (de `createMemoryTools`) descrevendo esse fato, e confirmar que ele deixa de ser recuperável.

- [X] T005 [P] [US2] (depende de T001, e de `createMemoryTools`/`SqliteMemoryStore` já existentes de `007`) Estender `src/memory/learning-reflector.test.ts` com um teste de integração: sobre `new SqliteMemoryStore(":memory:", fakeEmbed)` real, chamar `reflectAndRemember(store, userId, message, fakeDistillFnQueRetornaHasLearningTrue)` para gravar um fato; em seguida, invocar a tool `forget_fact` (de `createMemoryTools(store, userId)`) com uma descrição correspondente; confirmar que a observação retornada é `{ removed: true, fact }` e que uma chamada seguinte a `store.recall(userId, ...)` não retorna mais esse fato (FR-008 — remoção funciona igual para fatos automáticos e manuais).

**Checkpoint**: User Stories 1 e 2 funcionam de forma independente.

---

## Phase 5: User Story 3 - A resposta ao usuário nunca é afetada pelo processo de aprendizado (Priority: P3)

**Goal**: Confirmar que o disparo feito em T003 nunca atrasa nem quebra a resposta de `POST /chat`, mesmo quando o processo de reflexão está lento ou falha.

**Independent Test**: Injetar um `reflectAndRemember` fake que nunca resolve (ou que rejeita) via `createApp(...)` e confirmar que `POST /chat` com `userId` ainda responde `200` normalmente, sem esperar por ele e sem expor erro nenhum.

- [X] T006 [P] [US3] (depende de T003) Estender `src/http/server.test.ts`: injetar um `reflectAndRemember` fake que retorna uma `Promise` que nunca resolve (mesmo padrão de `neverResolvingFake` já usado no arquivo para a estratégia de raciocínio). Confirmar que uma requisição `POST /chat` com `userId` ainda responde `200` dentro do tempo esperado do teste — a resposta não espera essa promessa.
- [X] T007 [P] [US3] (depende de T003) Estender `src/http/server.test.ts`: injetar um `reflectAndRemember` fake que rejeita (`Promise.reject(new Error("falha simulada"))`). Confirmar que `POST /chat` com `userId` ainda responde `200`, sem status `500` nem exceção não tratada no processo de teste — prova a camada `.catch(() => {})` do controller (research.md item 3), independente de a implementação injetada respeitar ou não o contrato de "nunca rejeita" de `reflectAndRemember`.

**Checkpoint**: Todas as 3 user stories funcionam de forma independente.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Confirmar não-regressão e aderência final aos contratos desta feature.

- [X] T008 [P] Rodar `npm run typecheck` e `npm test` e confirmar que as suítes já existentes (001–007) continuam passando sem alteração de comportamento observável — em particular, que uma requisição a `/chat` sem `userId` continua produzindo `answer`/`trace`/`metrics`/`conversationId` idênticos a antes desta feature.
- [X] T009 Rodar manualmente `specs/008-learning-reflector/quickstart.md` de ponta a ponta contra `npm run dev` real (`OPENROUTER_API_KEY`/`OPENROUTER_MODEL` configurados) — os 5 passos: aprendizado automático sem pedir para lembrar; pedido pontual não vira aprendizado; segredo nunca é aprendido; desfazer um aprendizado automático via `forget_fact`; confirmar que sem `userId` nada é acionado.
- [X] T010 [P] Revisar `specs/008-learning-reflector/contracts/post-chat.md` e `contracts/learning-reflector.md` contra a implementação final — conferido: assinaturas de `distillLearning`/`reflectAndRemember`/`CreateAppOptions.reflectAndRemember` batem exatamente com o código (`src/memory/learning-reflector.ts`, `src/http/server.ts`); nenhuma divergência encontrada, nenhuma correção necessária.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: nenhuma tarefa — Foundational começa direto.
- **Foundational (Phase 2)**: BLOQUEIA todas as user stories — `reflectAndRemember` precisa existir e estar testado antes de qualquer wiring ou teste de integração.
- **User Stories (Phase 3–5)**: todas dependem da Foundational completa.
  - US1 (P1) não depende de US2/US3.
  - US2 (P2) depende só da Foundational (T001) e de `createMemoryTools`/`SqliteMemoryStore` já existentes (`007`) — não depende do wiring de US1 (T003).
  - US3 (P3) depende do wiring de US1 (T003) existir, porque testa exatamente o comportamento não-bloqueante desse ponto de disparo — única dependência entre stories nesta feature.
- **Polish (Phase 6)**: depende de todas as user stories desejadas estarem completas.

### Dentro de cada User Story

- US1: T003 (server.ts) → T004 (teste, depende de T003).
- US2: T005 depende só de T001 (Foundational) — pode ser feita a qualquer momento depois da Foundational, inclusive em paralelo com US1.
- US3: T006 e T007 dependem de T003 (US1) — não podem começar antes do wiring existir, mas são independentes entre si e da T004.

### Parallel Opportunities

- Dentro da Foundational: T001 sozinha, depois T002 (depende de T001).
- T005 (US2) pode ser feita em paralelo com T003–T004 (US1), já que depende só da Foundational.
- T006 e T007 (US3) podem ser feitas em paralelo entre si assim que T003 (US1) estiver pronta, mesmo que T004 ainda não tenha sido escrita.
- T008 e T010 (Polish) são paralelizáveis entre si.

---

## Parallel Example: User Story 2 + User Story 3 (depois de T001–T003)

```bash
# Depois da Foundational (T001) e do wiring de US1 (T003), em paralelo:
Task: "Teste de integração forget_fact sobre fato aprendido automaticamente (T005, US2)"
Task: "Teste de resposta não-bloqueada com refletor lento/falho (T006-T007, US3)"
```

---

## Implementation Strategy

### MVP First (User Story 1 apenas)

1. Completar Phase 2 (Foundational) — `reflectAndRemember` pronto e testado com fakes.
2. Completar Phase 3 (US1: T003–T004).
3. **PARAR e VALIDAR**: rodar `npm test` e os passos 1–2 de `quickstart.md` (testes determinísticos; aprendizado automático via `/chat` real).
4. Nesse ponto, `npm run dev` já aprende fatos automaticamente via `/chat` — MVP entregável (esquecer via `forget_fact` e a garantia de não-bloqueio já funcionam por baixo, mesmo antes de US2/US3 formalizarem os testes específicos).

### Incremental Delivery

1. Foundational → `reflectAndRemember` pronto e testado.
2. US1 → validar independentemente → aprendizado automático funcionando via `/chat` (MVP).
3. US2 → validar independentemente → remoção confirmada para fatos aprendidos automaticamente.
4. US3 → validar independentemente → não-bloqueio e absorção de falha confirmados.
5. Polish → confirma não-regressão (001–007) e fecha os contratos.

---

## Notes

- `[P]` = arquivos diferentes, sem dependência de tarefa incompleta.
- Rótulo `[US#]` mapeia a tarefa à user story correspondente da spec.
- Cada user story deve ser completável e testável de forma independente (exceto a dependência documentada de US3 sobre o wiring de US1, inerente ao que US3 verifica).
- Commitar após cada tarefa ou grupo lógico pequeno (constitution: Pequeno e Reversível).
- `npm run typecheck` e `npm test` devem ficar verdes ao final de cada fase — nenhuma tarefa desta feature depende de rede ou de `OPENROUTER_API_KEY` (diferente de `007`), exceto a validação manual de `quickstart.md` (T009).
