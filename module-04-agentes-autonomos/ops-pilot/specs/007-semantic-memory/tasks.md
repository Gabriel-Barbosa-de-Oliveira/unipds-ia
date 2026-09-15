---

description: "Task list template for feature implementation"
---

# Tasks: Memória Semântica

**Input**: Design documents from `/specs/007-semantic-memory/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos — a constitution (Teste Obrigatório, NON-NEGOTIABLE) pede testes explicitamente para toda lógica nova.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivo diferente, sem dependência de tarefa ainda não concluída)
- **[Story]**: A qual user story a tarefa pertence (US1–US3)
- Caminhos de arquivo exatos em cada descrição

## Path Conventions

Projeto único — `src/` na raiz do repositório (ver plan.md § Project Structure). Sem `tests/` separado: testes ficam ao lado do código (`*.test.ts`), convenção já usada pelas features 001–006.

---

## Phase 1: Setup

**Purpose**: Trazer a única dependência nova desta feature antes de qualquer código depender dela.

- [X] T001 Adicionar `@huggingface/transformers` a `dependencies` em `package.json` e rodar `npm install` (atualiza `package-lock.json`); adicionar `.cache/` ao `.gitignore` (cache local do checkpoint ONNX, research.md item 9).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Lógica pura de similaridade/serialização, o singleton de embedding, e o `SqliteMemoryStore` completo (remember/recall/forget + as duas tools do agente) — base sobre a qual as 3 user stories se apoiam. Nenhuma delas é testável de ponta a ponta antes desta fase estar completa.

**⚠️ CRITICAL**: Nenhuma user story começa antes desta fase estar completa.

- [X] T002 [P] Criar `src/domain/memory.ts`: `dotProduct(a: Float32Array, b: Float32Array): number` (produto escalar puro); `floatArrayToBuffer(vector: Float32Array): Buffer` / `bufferToFloatArray(buffer: Buffer): Float32Array` (serialização pura para a coluna `BLOB`, research.md item 3); `selectTopMatches<T>(candidates: { item: T; embedding: Float32Array }[], query: Float32Array, opts: { limit: number; minScore: number }): { item: T; score: number }[]` (ordena por `dotProduct` decrescente, descarta `score < minScore`, retorna no máximo `limit`); `composeWithFacts(facts: readonly string[], input: string): string` (sem fatos retorna `input` inalterado; com fatos, prefixa um bloco "fatos conhecidos" antes de `input` — mesmo espírito de `composePrompt`, `006-conversation-history`). Tudo puro, determinístico.
- [X] T003 [P] (depende de T002) Criar `src/domain/memory.test.ts`: `dotProduct` com vetores conhecidos (ortogonais → 0, idênticos normalizados → 1); `selectTopMatches` respeita `limit`, descarta abaixo de `minScore`, ordena corretamente; `floatArrayToBuffer`/`bufferToFloatArray` fazem round-trip exato; `composeWithFacts` sem fatos retorna `input` inalterado, com fatos inclui todos antes de `input`.
- [X] T004 [P] Criar `src/memory/embeddings.ts`: `embed(text: string): Promise<Float32Array>` — singleton lazy do `pipeline("feature-extraction", "onnx-community/all-MiniLM-L6-v2-ONNX", { cache_dir: "./.cache/transformers" })` de `@huggingface/transformers` (só instanciado no primeiro `embed()`, nunca no import do módulo); chama o pipeline com `{ pooling: "mean", normalize: true }`; retorna sempre um `Float32Array` de 384 posições.
- [X] T005 (depende de T004) Criar `src/memory/embeddings.test.ts`: `embed("qualquer texto")` retorna um `Float32Array` de exatamente 384 posições; o vetor é normalizado (`dotProduct(v, v)` de `src/domain/memory.ts` é aproximadamente `1`, com tolerância). Usa o modelo real — mais lento, precisa de rede na 1ª execução (research.md item 4/9).
- [X] T006 (depende de T002, T004) Criar `src/memory/memory-store.ts`: interface `MemoryStore` (`remember(userId, fact): Promise<{stored, id?}>`, `recall(userId, query, limit?): Promise<{fact, score}[]>`, `forget(userId, description): Promise<{removed, fact?}>`); classe `SqliteMemoryStore implements MemoryStore` sobre `node:sqlite` (`DatabaseSync`), construtor `(path?, embedFn?)` — `path` default `process.env.OPSPILOT_DB ?? "./data/opspilot.db"`, `embedFn` default `embed` de `embeddings.ts`; DDL idempotente para `memories` (`id TEXT PRIMARY KEY`, `user_id TEXT NOT NULL`, `fact TEXT NOT NULL`, `embedding BLOB NOT NULL`, `created_at TEXT NOT NULL`); `remember` usa `selectTopMatches`/`dotProduct` contra os fatos do mesmo `userId` — maior score `> 0.92` não grava; `recall` usa `selectTopMatches` com `limit` (padrão 3) e `minScore: 0.3`; `forget` usa a mesma busca sem limite de quantidade, remove o de maior score `>= 0.3`; todas as queries com prepared statements. Também exportar `createMemoryTools(store: MemoryStore, userId: string): StructuredToolInterface[]` — `remember_fact` (`{ fact: string }`) e `forget_fact` (`{ description: string }`), ambas fechadas por closure sobre `store` e `userId` (nunca campo do schema zod, research.md item 6), delegando a `store.remember`/`store.forget` e retornando a observação como JSON string.
- [X] T007 (depende de T006) Criar `src/memory/memory-store.test.ts` sobre `new SqliteMemoryStore(":memory:", fakeEmbed)`: com um `embedFn` fake e determinístico (vetores forjados à mão) — `remember` não duplica um fato com score forjado `> 0.92` contra um já existente do mesmo `userId`, mas grava um com score `<= 0.92`; `recall` nunca mistura fatos de dois `userId`s diferentes, nunca retorna mais que `limit` (padrão 3), descarta score `< 0.3`, retorna `[]` para `userId` sem fatos; `forget` remove o fato de maior score quando `>= 0.3` e reporta qual foi removido, e retorna `removed: false` sem apagar nada quando nada atinge o limiar; `remember_fact`/`forget_fact` (de `createMemoryTools`) delegam corretamente a `store.remember`/`store.forget` com o `userId` da closure, nunca um `userId` vindo do schema. Adicionar também **um teste dedicado usando o `embed` real** (sem `fakeEmbed`, importado de `embeddings.ts`): registrar um fato e recuperá-lo com uma pergunta sem nenhuma palavra em comum, confirmando `score >= 0.3` (spec SC-001, User Story 1) — mais lento, precisa de rede na 1ª execução.

**Checkpoint**: Fundação pronta — `npm run typecheck`/`npm test` verdes (a menos do tempo/rede da 1ª execução dos testes com modelo real); as user stories podem começar.

---

## Phase 3: User Story 1 - O copiloto lembra fatos contados antes, mesmo com outras palavras (Priority: P1) 🎯 MVP

**Goal**: `POST /chat` aceita `userId` opcional; quando presente, recupera automaticamente (antes de chamar a estratégia) até 3 fatos relevantes daquele `userId` e os injeta no prompt, e disponibiliza `remember_fact` ao modelo — tudo escopado a esse `userId`.

**Independent Test**: Enviar uma mensagem com `userId` registrando um fato (via o modelo decidindo chamar `remember_fact`, ou diretamente pelo store em teste), depois perguntar algo relacionado com palavras totalmente diferentes e confirmar que o fato influencia a resposta; confirmar que sem `userId` o comportamento é idêntico ao de `006`.

- [X] T008 [US1] Estender `src/agents/index.ts`: `resolveStrategy(name?: string, reflect?: boolean, extraTools?: StructuredToolInterface[]): ReasoningStrategy` — quando `extraTools` é informado e não vazio, construir a estratégia via `createReactStrategy([...opsTools, ...extraTools])` ou `createPlanAndExecuteStrategy([...opsTools, ...extraTools])` (conforme `resolvedName`) em vez do singleton `STRATEGIES[resolvedName]`; quando omitido, comportamento idêntico ao atual (mesmo singleton). `reflect` continua se aplicando por cima do resultado, como hoje.
- [X] T009 [P] [US1] (depende de T008) Estender `src/agents/index.test.ts`: `resolveStrategy("react", false, [umaToolExtra])` produz uma estratégia cujas tools incluem a extra (verificável rodando a estratégia contra um cenário que só é resolvido se a tool extra existir, ou inspecionando a composição); `resolveStrategy("react")` (sem terceiro argumento) continua retornando exatamente o singleton `reactStrategy` de antes (mesma referência ou mesmo comportamento observável).
- [X] T010 [US1] (depende de T006, T008) Estender `src/http/server.ts`: `ChatRequestSchema` ganha `userId: z.string().optional()`; `CreateAppOptions` ganha `memoryStore?: MemoryStore` (default: uma instância real de `SqliteMemoryStore`, mesmo padrão de `conversationStore`); no handler de `POST /chat`, quando `parsed.data.userId` está presente: `const recalled = await memoryStore.recall(userId, parsed.data.message, 3)`, compor `composeWithFacts(recalled.map(r => r.fact), promptComHistorico)` (por cima do resultado de `composePrompt`, `006`) como o `input` final; construir `resolveStrategy(parsed.data.strategy, parsed.data.reflect, createMemoryTools(memoryStore, userId))`. Quando `userId` está ausente, comportamento idêntico ao de `006` (sem recall, sem tools extras, `resolveStrategy` chamado com dois argumentos).
- [X] T011 [P] [US1] (depende de T010) Estender `src/http/server.test.ts`: injetar um `MemoryStore` fake determinístico via `createApp({ memoryStore })` (mesmo padrão de `conversationStore` fake já usado no arquivo). Casos: (a) mensagem sem `userId` produz exatamente o mesmo resultado que sem esta feature — nenhuma chamada a `recall`, resposta idêntica ao contrato de `006`; (b) mensagem com `userId` cujo fake `recall` retorna fatos pré-configurados faz a estratégia fake receber um `input` que inclui esses fatos; (c) dois `userId`s diferentes, exercitados na mesma suíte, nunca compartilham fatos entre si (fake escopado por `userId`).

**Checkpoint**: User Story 1 completa e testável de forma independente — memória semântica funcionando via `/chat`. MVP entregável.

---

## Phase 4: User Story 2 - Não acumular fatos duplicados quando algo é repetido (Priority: P2)

**Goal**: Confirmar que a tool `remember_fact` (disponível desde US1) delega corretamente à deduplicação já implementada em `SqliteMemoryStore.remember` (T006/T007) — nenhuma lógica nova de dedup, só a garantia de que o caminho tool → store está correto.

**Independent Test**: Chamar a tool `remember_fact` duas vezes com fatos essencialmente iguais (via um `MemoryStore` fake ou real) e confirmar que a segunda chamada reporta `stored: false`, sem duplicar.

- [X] T012 [P] [US2] Estender `src/memory/memory-store.test.ts` (se o cenário de T007 ainda não cobrir explicitamente pela tool, e não só pelo método do store) com um teste que invoca a tool `remember_fact` (de `createMemoryTools`) duas vezes seguidas com fatos quase idênticos para o mesmo `userId`, confirmando que a segunda invocação retorna `{ stored: false }` na observação JSON, e que só um registro existe em `store.recall` ao final.

**Checkpoint**: User Stories 1 e 2 funcionam de forma independente.

---

## Phase 5: User Story 3 - Pedir para o copiloto esquecer um fato específico (Priority: P3)

**Goal**: Confirmar que a tool `forget_fact` (disponível desde US1) delega corretamente à remoção semântica já implementada em `SqliteMemoryStore.forget` (T006/T007).

**Independent Test**: Registrar um fato, chamar a tool `forget_fact` descrevendo-o, e confirmar que uma chamada seguinte de `recall` não o retorna mais; chamar `forget_fact` com uma descrição que não corresponde a nada e confirmar que nada é removido.

- [X] T013 [P] [US3] Estender `src/memory/memory-store.test.ts` com um teste que invoca a tool `forget_fact` (de `createMemoryTools`) para remover um fato previamente registrado (via `remember_fact` ou diretamente pelo store), confirmando a observação `{ removed: true, fact }` e que `recall` deixa de retorná-lo; e outro teste chamando `forget_fact` com uma descrição sem correspondência suficiente, confirmando `{ removed: false }` e que nenhum fato existente foi apagado.

**Checkpoint**: Todas as 3 user stories funcionam de forma independente.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Confirmar não-regressão, validar a promessa central com o modelo real de ponta a ponta, e aderência final aos contratos desta feature.

- [X] T014 [P] Rodar `npm run typecheck` e `npm test` e confirmar que as suítes já existentes (001–006) continuam passando sem alteração de comportamento observável — em particular, que uma requisição a `/chat` sem `userId` continua produzindo `answer`/`trace`/`metrics`/`conversationId` idênticos a antes desta feature.
- [X] T015 Rodar manualmente `specs/007-semantic-memory/quickstart.md` de ponta a ponta contra `npm run dev` real — executado por completo, sem divergências: passo 2 (recall direto no store: fato recuperado com `score` 0.394, zero palavras em comum); passo 3 (via `/chat`: o copiloto registrou o fato e, na pergunta seguinte sobre "checkout financeiro", respondeu citando o fato de pagamentos); passo 4 (dedup: o copiloto reconheceu que o fato já existia e o store continuou com exatamente 1 registro); passo 5 (forget: fato removido; a pergunta seguinte deixou de citá-lo); passo 6 (isolamento: o fato de `ana` nunca apareceu na resposta a `gabriel`; estado final do store confirma `gabriel: []`, `ana: 1 fato`).
- [X] T016 [P] Revisar `specs/007-semantic-memory/contracts/post-chat.md` e `contracts/memory-store.md` contra a implementação final — conferido: assinaturas de `embed`/`remember`/`recall`/`forget`/`createMemoryTools`/`resolveStrategy`, limiares (`DEDUP_THRESHOLD = 0.92`, `RECALL_MIN_SCORE = 0.3`, `RECALL_LIMIT = 3`) e nomes de tool/campo batem com o código. Única correção necessária foi o id do checkpoint, ajustado durante T004 em research.md/plan.md/quickstart.md/contracts (`Xenova/all-MiniLM-L6-v2` → `onnx-community/all-MiniLM-L6-v2-ONNX`, o exemplo oficial da versão instalada da biblioteca).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: sem dependências — pode começar imediatamente.
- **Foundational (Phase 2)**: depende do Setup (precisa da dependência instalada) — BLOQUEIA todas as user stories.
- **User Stories (Phase 3–5)**: todas dependem da Foundational completa.
  - US1 (P1) não depende de US2/US3.
  - US2 (P2) depende das tools existirem (T006, criadas na Foundational) e do wiring de US1 (T010) só indiretamente — na prática só precisa de T006, mas seu teste reaproveita `createMemoryTools` já testado em T007.
  - US3 (P3) mesma relação de US2, sobre `forget_fact`.
- **Polish (Phase 6)**: depende de todas as user stories desejadas estarem completas.

### Dentro de cada User Story

- US1: T008 (resolveStrategy) → T009 (teste) e T010 (server.ts, depende de T008) → T011 (teste, depende de T010).
- US2: T012 depende só de T006/T007 (Foundational) — pode ser escrita a qualquer momento depois da Foundational, inclusive em paralelo com US1.
- US3: T013 mesma relação de US2.

### Parallel Opportunities

- Dentro da Foundational: T002 e T004 em paralelo (arquivos diferentes, sem dependência mútua); T003 após T002; T005 após T004; T006 após T002+T004; T007 após T006.
- T012 (US2) e T013 (US3) podem ser feitas em paralelo entre si e em paralelo com T008–T011 (US1) por pessoas diferentes, já que todas dependem só da Foundational (T006/T007), não uma da outra.
- T009 e T011 estendem arquivos diferentes (`agents/index.test.ts` vs `http/server.test.ts`) — paralelizáveis entre si, mas cada uma depende da tarefa de implementação correspondente (T008/T010).

---

## Parallel Example: Foundational

```bash
# Em paralelo, logo no início (depois de T001):
Task: "Criar src/domain/memory.ts (T002)"
Task: "Criar src/memory/embeddings.ts (T004)"

# Depois de T002 + T004:
Task: "Criar src/memory/memory-store.ts (T006)"
```

## Parallel Example: User Story 1 + User Story 2 + User Story 3

```bash
# Depois da Foundational, três pessoas em paralelo:
Task: "resolveStrategy(extraTools) + wiring de /chat (T008-T011, US1)"
Task: "Teste de dedup via remember_fact (T012, US2)"
Task: "Teste de remoção via forget_fact (T013, US3)"
```

---

## Implementation Strategy

### MVP First (User Story 1 apenas)

1. Completar Phase 1 (Setup) e Phase 2 (Foundational) — dependência instalada, store completo (remember/recall/forget + tools) já testado, inclusive a promessa central (recall sem palavra em comum) provada com o modelo real.
2. Completar Phase 3 (US1: T008–T011).
3. **PARAR e VALIDAR**: rodar `npm test` e os passos 2–3 de `quickstart.md` (recall sem palavra em comum, via store e via `/chat`).
4. Nesse ponto, `npm run dev` já serve `/chat` com memória semântica — MVP entregável (dedup e forget já funcionam por baixo, mesmo antes de US2/US3 formalizarem os testes específicos das tools).

### Incremental Delivery

1. Setup + Foundational → base pronta (store completo, promessa central provada).
2. US1 → validar independentemente → memória semântica funcionando via `/chat` (MVP).
3. US2 → validar independentemente → deduplicação confirmada no caminho da tool.
4. US3 → validar independentemente → esquecer confirmado no caminho da tool.
5. Polish → confirma não-regressão (001–006), valida o quickstart completo com o modelo real, e fecha os contratos.

---

## Notes

- `[P]` = arquivos diferentes, sem dependência de tarefa incompleta.
- Rótulo `[US#]` mapeia a tarefa à user story correspondente da spec.
- Cada user story deve ser completável e testável de forma independente.
- Commitar após cada tarefa ou grupo lógico pequeno (constitution: Pequeno e Reversível).
- `npm run typecheck` e `npm test` devem ficar verdes ao final de cada fase — os dois testes que dependem do modelo real (`T005`, parte de `T007`) só são lentos/precisam de rede na primeira execução em cada máquina (research.md item 4/9), não a cada execução.
