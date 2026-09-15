---

description: "Task list template for feature implementation"
---

# Tasks: Conversa Persistente

**Input**: Design documents from `/specs/006-conversation-history/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos — a constitution (Teste Obrigatório, NON-NEGOTIABLE) pede testes explicitamente para toda lógica nova.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivo diferente, sem dependência de tarefa ainda não concluída)
- **[Story]**: A qual user story a tarefa pertence (US1–US3)
- Caminhos de arquivo exatos em cada descrição

## Path Conventions

Projeto único — `src/` na raiz do repositório (ver plan.md § Project Structure). Sem `tests/` separado: testes ficam ao lado do código (`*.test.ts`), convenção já usada pelas features 001–004.

---

## Phase 1: Setup

**Purpose**: Inicialização de projeto.

Nenhuma tarefa de setup necessária nesta feature — nenhuma dependência de pacote nova, nenhum código morto a remover antes de começar (diferente de `004-ops-persistence`). A fundação começa diretamente na Phase 2.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Modelo de domínio da conversa, contrato do repositório e o adaptador SQLite conforme esse contrato — base sobre a qual as 3 user stories se apoiam. Nenhuma delas é testável de ponta a ponta antes desta fase estar completa.

**⚠️ CRITICAL**: Nenhuma user story começa antes desta fase estar completa.

- [X] T001 [P] Criar `src/domain/conversation.ts`: tipo `ConversationMessage` (`{ role: "user" | "assistant"; content: string }`) e função pura `composePrompt(history: readonly ConversationMessage[], input: string): string` — histórico vazio retorna só `input`; histórico não vazio formata cada turno anterior (identificando quem falou) seguido da nova mensagem, de forma determinística (mesma entrada sempre produz a mesma string).
- [X] T002 [P] (depende de T001) Criar `src/domain/conversation.test.ts`: `composePrompt` com histórico vazio (retorna exatamente `input`, sem texto extra), com histórico parcial (2-3 mensagens, ordem preservada), e com 12 mensagens (todas incluídas, nenhuma omitida pela própria função — o corte para 12 é responsabilidade de quem busca o histórico, não da composição).
- [X] T003 [P] Adicionar `ConversationNotFoundError` a `src/domain/errors.ts` — mesmo padrão de `IncidentNotFoundError`/`ServiceNotFoundError` (classe com campo `conversationId`, `name` próprio).
- [X] T004 (depende de T001, T003) Criar `src/services/conversation-store.repository.ts`: interface `ConversationStore` com `create(): Promise<string>`, `append(conversationId: string, messages: ConversationMessage[]): Promise<void>` e `lastMessages(conversationId: string, limit: number): Promise<ConversationMessage[]>` (reexportando `ConversationMessage` de `src/domain/conversation.ts`); documentar no próprio arquivo que `append`/`lastMessages` devem lançar `ConversationNotFoundError` para um `conversationId` desconhecido.
- [X] T005 (depende de T004) Criar `src/store/sqlite-conversation-store.ts`: classe `SqliteConversationStore implements ConversationStore` sobre `node:sqlite` (`DatabaseSync`); DDL idempotente (`CREATE TABLE IF NOT EXISTS`) para `conversations` (`id TEXT PRIMARY KEY`, `created_at TEXT NOT NULL`) e `messages` (`id INTEGER PRIMARY KEY AUTOINCREMENT`, `conversation_id TEXT NOT NULL REFERENCES conversations(id)`, `role TEXT NOT NULL CHECK (role IN ('user','assistant'))`, `content TEXT NOT NULL`, `created_at TEXT NOT NULL`); caminho default `process.env.OPSPILOT_DB ?? "./data/opspilot.db"`, conexão lazy (mesmo padrão de `SqliteOpsStore`, `004-ops-persistence`); `create()` gera `crypto.randomUUID()` e insere a conversa; `append`/`lastMessages` verificam a existência da conversa primeiro (`SELECT` em `conversations`) e lançam `ConversationNotFoundError` se não encontrada; `lastMessages` ordena por `id` (nunca `created_at` — ver research.md item 7) e retorna no máximo `limit` mensagens mais recentes, em ordem cronológica; todas as queries via prepared statements.
- [X] T006 [P] (depende de T005) Criar `src/store/sqlite-conversation-store.test.ts` sobre `new SqliteConversationStore(":memory:")`: `create()` gera ids distintos a cada chamada; `append` seguido de `lastMessages` retorna as mensagens na ordem de inserção; `append`/`lastMessages` com `conversationId` desconhecido lançam `ConversationNotFoundError`; conversa recém-criada sem mensagens retorna lista vazia de `lastMessages`; uma conversa com mais de 12 mensagens — `lastMessages(id, 12)` retorna exatamente as 12 mais recentes, nunca mais, em ordem cronológica; duas conversas distintas (dois `create()`) nunca misturam mensagens entre si mesmo com `append` intercalado.

**Checkpoint**: Fundação pronta — `npm run typecheck`/`npm test` verdes; as user stories podem começar.

---

## Phase 3: User Story 1 - Continuar uma conversa ao longo de várias mensagens (Priority: P1) 🎯 MVP

**Goal**: `POST /chat` aceita `conversationId` opcional, resolve/cria a conversa, compõe o histórico com a nova mensagem antes de chamar a estratégia, e grava o turno ao final — o copiloto passa a responder considerando o que foi dito antes na mesma conversa.

**Independent Test**: Enviar uma mensagem sem `conversationId`, guardar o id devolvido, enviar uma segunda mensagem referenciando esse id e algo dito na primeira, e confirmar que a resposta reflete esse contexto; confirmar que um `conversationId` desconhecido retorna erro claro, e que duas conversas nunca compartilham histórico.

- [X] T007 [US1] Estender `src/http/server.ts`: `ChatRequestSchema` ganha `conversationId: z.string().optional()`; `CreateAppOptions` ganha `conversationStore?: ConversationStore` (default: uma instância real de `SqliteConversationStore`, mesmo padrão de `resolveStrategy`/`resolveStrategyDefault`); no handler de `POST /chat`, após validar o corpo: resolver a conversa (`conversationId` informado → `conversationStore.lastMessages(conversationId, 12)` como histórico; omitido → `conversationStore.create()` e histórico vazio), compor `composePrompt(history, parsed.data.message)` (`src/domain/conversation.ts`) como o `input` passado a `runWithTimeout(strategy, ..., options, timeoutMs)`; após a estratégia responder com sucesso, `conversationStore.append(conversationId, [{ role: "user", content: parsed.data.message }, { role: "assistant", content: result.answer }])`; responder `200` com `{ ...result, conversationId, metrics: { ...result.metrics, historyMessages: history.length } }`.
- [X] T008 (depende de T007) Estender `errorMiddleware` em `src/http/server.ts`: tratar `ConversationNotFoundError` (importada de `src/domain/errors.ts`) retornando `404 { error: "conversation_not_found", conversationId: error.conversationId }`.
- [X] T009 [P] [US1] (depende de T007, T008) Estender `src/http/server.test.ts`: injetar um `ConversationStore` fake determinístico via `createApp({ conversationStore })` (mesmo padrão do `fakeStrategy`/`resolveStrategy` já usado no arquivo). Casos: (a) mensagem sem `conversationId` recebe `200` com um `conversationId` novo e `metrics.historyMessages: 0`; (b) mensagem com `conversationId` de uma conversa existente (fake pré-configurado com histórico) faz a estratégia fake receber um `input` que inclui esse histórico, e a resposta reporta `conversationId` igual ao enviado; (c) `conversationId` desconhecido retorna `404 { error: "conversation_not_found", conversationId }` sem chamar a estratégia (contador de chamadas da fake permanece `0`); (d) duas conversas diferentes, exercitadas na mesma suíte, nunca compartilham histórico entre si.

**Checkpoint**: User Story 1 completa e testável de forma independente — conversas persistem e continuam corretamente. MVP entregável.

---

## Phase 4: User Story 2 - Retomar uma conversa longa sem perder desempenho (Priority: P2)

**Goal**: Provar que o limite de 12 mensagens (já implementado em `SqliteConversationStore.lastMessages`, T005/T006) se comporta corretamente end-to-end através do `/chat` — uma conversa com muito mais de 12 mensagens continua respondendo normalmente, sem erro e sem o prompt crescer sem limite.

**Independent Test**: Conduzir (via um `ConversationStore` fake ou real) uma conversa com mais de 12 mensagens trocadas e confirmar que uma nova mensagem continua respondendo `200`, com `metrics.historyMessages` nunca excedendo 12.

- [X] T010 [P] [US2] Estender `src/store/sqlite-conversation-store.test.ts` (já coberto em T006 — o teste "lastMessages nunca retorna mais que o limite..." já simula 8 turnos/16 mensagens) com uma simulação de conversa real turno a turno: 8 chamadas de `append` (16 mensagens no total) sobre a mesma conversa, confirmando a cada passo que `lastMessages(id, 12)` nunca retorna mais que 12 e sempre reflete as mensagens mais recentes na ordem correta.
- [X] T011 [US2] (depende de T009) Estender `src/http/server.test.ts` com um `ConversationStore` fake cujo `lastMessages` retorna 15 mensagens pré-populadas para uma conversa existente: confirmar que a resposta a uma nova mensagem nessa conversa é `200`, `metrics.historyMessages` é exatamente `12` (nunca `15`), e o `input` recebido pela estratégia fake tem tamanho limitado (não contém as 15 mensagens inteiras sem corte).

**Checkpoint**: User Stories 1 e 2 funcionam de forma independente.

---

## Phase 5: User Story 3 - Auditar quanto contexto foi usado em cada resposta (Priority: P3)

**Goal**: Confirmar que `metrics.historyMessages` reporta, para qualquer conversa, exatamente a quantidade de mensagens de histórico usadas naquela resposta específica — capacidade já exposta pelo wiring de US1 (T007), formalizada aqui com os casos de auditoria pedidos pela spec.

**Independent Test**: Enviar mensagens em conversas com diferentes quantidades de histórico acumulado (nenhuma, poucas, mais de 12) e confirmar que `metrics.historyMessages` corresponde exatamente ao que foi incluído em cada caso.

- [X] T012 [P] [US3] Estender `src/http/server.test.ts` com os três casos de auditoria da spec (reaproveitando os fakes de T009/T011): conversa nova (sem histórico) → `metrics.historyMessages === 0`; conversa com poucas mensagens (ex.: 2, fake pré-populado) → `metrics.historyMessages === 2`; conversa com mais de 12 (fake com 15) → `metrics.historyMessages === 12`.

**Checkpoint**: Todas as 3 user stories funcionam de forma independente.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Confirmar não-regressão e aderência final aos contratos desta feature.

- [X] T013 [P] Rodar `npm run typecheck` e `npm test` e confirmar que as suítes já existentes (`src/domain/ops-store.test.ts`, `src/agents/index.test.ts`, `src/agents/reflection.test.ts`, `src/store/sqlite-ops-store.test.ts`, `src/services/chat.service.test.ts`, `src/mcp/server.test.ts`) continuam passando sem alteração de comportamento observável — em particular, que uma requisição a `/chat` sem `conversationId` continua produzindo `answer`/`trace`/`metrics.llmCalls`/`metrics.latencyMs` idênticos a antes desta feature, apenas com os dois campos novos adicionados (FR-008).
- [X] T014 Rodar manualmente `specs/006-conversation-history/quickstart.md`: passos 1–2 completos (`npm run typecheck`/`npm test`, 114 testes verdes); passos 3–6 executados de ponta a ponta contra `npm run dev` real (credenciais já presentes em `.env`) — conversa nova (`historyMessages: 0`), conversa continuada (copiloto lembrou "Gabriel", `historyMessages: 2`), `conversationId` desconhecido (`404 conversation_not_found` instantâneo) e isolamento entre conversas (segunda conversa nunca viu "Ana") — todos os resultados exatamente como esperado, nenhuma divergência encontrada.
- [X] T015 [P] Revisar `specs/006-conversation-history/contracts/post-chat.md` e `contracts/conversation-store.md` contra a implementação final (assinaturas, mensagens de erro, nomes de campo) — conferido: `404 { error: "conversation_not_found", conversationId }`, `metrics.historyMessages`, e as assinaturas de `create`/`append`/`lastMessages` batem exatamente com `src/http/server.ts`/`src/store/sqlite-conversation-store.ts`; nenhum ajuste necessário.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: não aplicável — sem tarefas.
- **Foundational (Phase 2)**: sem dependências externas — BLOQUEIA todas as user stories.
- **User Stories (Phase 3–5)**: todas dependem da Foundational completa.
  - US1 (P1) não depende de US2/US3.
  - US2 (P2) depende do wiring de US1 (T007–T009) para seu teste end-to-end (T011), mas não depende de US3.
  - US3 (P3) depende do wiring de US1 (T007) e reaproveita fakes de US1/US2 em seus testes; não depende de US2 para existir, só para um dos seus casos de teste ser mais rico.
- **Polish (Phase 6)**: depende de todas as user stories desejadas estarem completas.

### Dentro de cada User Story

- US1: T007 (wiring) → T008 (erro 404, mesmo arquivo) → T009 (testes).
- US2: T010 (teste de store, independente) pode rodar em paralelo com T007–T009; T011 depende de T009 (estende a mesma suíte/fakes).
- US3: T012 depende de T009 (e, para o caso de 12, também se beneficia de T011 já existir, mas pode ser escrito de forma independente com seu próprio fake).

### Parallel Opportunities

- Dentro da Foundational: T001 e T003 em paralelo (arquivos diferentes, sem dependência mútua); T002 após T001; T004 após T001+T003; T005 após T004; T006 após T005.
- T010 (US2, arquivo de store) pode ser feito em paralelo com T007–T009 (US1, arquivo de HTTP) por duas pessoas diferentes, já que ambos só dependem da Foundational.
- T009, T011, T012 estendem o mesmo arquivo (`src/http/server.test.ts`) em sequência — não são paralelizáveis entre si, mesmo pertencendo a stories diferentes.

---

## Parallel Example: Foundational

```bash
# Em paralelo, logo no início:
Task: "Criar src/domain/conversation.ts (T001)"
Task: "Adicionar ConversationNotFoundError a src/domain/errors.ts (T003)"

# Depois de T001 + T003:
Task: "Criar src/services/conversation-store.repository.ts (T004)"
```

## Parallel Example: User Story 1 + User Story 2

```bash
# Depois da Foundational, duas pessoas em paralelo:
Task: "Wiring de conversationId/histórico em src/http/server.ts + testes (T007-T009, US1)"
Task: "Teste de conversa longa (>12 mensagens) em src/store/sqlite-conversation-store.test.ts (T010, US2)"
```

---

## Implementation Strategy

### MVP First (User Story 1 apenas)

1. Completar Phase 2 (Foundational) — store, domínio e erro de conversa prontos.
2. Completar Phase 3 (US1: T007–T009).
3. **PARAR e VALIDAR**: rodar `npm test` e os passos 1, 3 e 4 de `quickstart.md` (conversa nova, conversa continuada).
4. Nesse ponto, `npm run dev` já serve `/chat` com continuidade de conversa — MVP entregável.

### Incremental Delivery

1. Foundational → base pronta (store de conversas, sem tocar o HTTP ainda).
2. US1 → validar independentemente → continuidade de conversa funcionando (MVP).
3. US2 → validar independentemente → conversas longas confirmadamente seguras.
4. US3 → validar independentemente → `historyMessages` auditável em qualquer resposta.
5. Polish → confirma não-regressão (001–005) e fecha o quickstart completo.

---

## Notes

- `[P]` = arquivos diferentes, sem dependência de tarefa incompleta.
- Rótulo `[US#]` mapeia a tarefa à user story correspondente da spec.
- Cada user story deve ser completável e testável de forma independente.
- Commitar após cada tarefa ou grupo lógico pequeno (constitution: Pequeno e Reversível).
- `npm run typecheck` e `npm test` devem ficar verdes ao final de cada fase, não só no final da feature.
