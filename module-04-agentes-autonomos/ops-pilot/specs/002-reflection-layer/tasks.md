---

description: "Task list template for feature implementation"
---

# Tasks: Camada de Reflection

**Input**: Design documents from `/specs/002-reflection-layer/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Constitution Principle V (Teste Obrigatório) exige teste para toda lógica nova. `runReflectionLoop` e os helpers puros (`buildRetryInput`, `buildCritiquePrompt`) são determinísticos e testados com fakes, sem rede — incluídos abaixo. `withReflection` em si (a casca que instancia o crítico real via `createModel()`) não recebe teste automatizado, pelo mesmo motivo já registrado em [../001-reasoning-strategies-core/research.md](../001-reasoning-strategies-core/research.md) item 7 para `react.ts`/`plan-and-execute.ts`: exige um modelo real via OpenRouter; sua verificação é manual, guiada por [quickstart.md](./quickstart.md).

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e teste independentes de cada uma.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Pode rodar em paralelo (arquivos diferentes, sem dependência de tarefa incompleta)
- **[Story]**: A qual user story a tarefa pertence (US1, US2, US3)
- Caminhos de arquivo exatos em cada descrição

## Path Conventions

Projeto único (Node/TypeScript ESM), mesma convenção da feature 001: testes ficam colocados ao lado do módulo testado como `*.test.ts`. Toda a feature cabe em um arquivo novo (`src/agents/reflection.ts` + `src/agents/reflection.test.ts`) e uma extensão pequena de `src/arena.ts` — nenhum arquivo de `src/domain/`, `src/services/` ou `src/models/` é tocado.

---

## Phase 1: Setup (Shared Infrastructure)

Nenhuma tarefa de setup nova é necessária. Esta feature reaproveita 100% da infraestrutura já criada pela feature 001 (dependências do `package.json`, `OPENROUTER_API_KEY`/`OPENROUTER_MODEL`, fábrica única de modelo, dataset semeado) — nenhuma dependência, script ou variável de ambiente nova é introduzida (ver [plan.md](./plan.md) Technical Context).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Schema do crítico, helpers puros de formatação e o esqueleto da orquestração do ciclo de reflection — usados por todas as user stories

**⚠️ CRITICAL**: Nenhuma user story pode começar antes desta fase estar completa

- [X] T001 Definir `CritiqueSchema` (zod: `{ approved: boolean, feedback: string não vazia }`) e os helpers puros `buildRetryInput(originalInput, previousAnswer, feedback)` e `buildCritiquePrompt(input, trace, answer)` (reaproveitando `formatTrace` de `src/agents/trace.ts`) em `src/agents/reflection.ts` (per [contracts/reflection-strategy.md](./contracts/reflection-strategy.md))
- [X] T002 [P] Testes determinísticos e sem rede de `buildRetryInput`/`buildCritiquePrompt` em `src/agents/reflection.test.ts` (fixtures fixas → string esperada) — depende de T001
- [X] T003 [P] Implementar `runReflectionLoop(runAttempt, critique, input, options, maxReflections)` em `src/agents/reflection.ts`: roda a 1ª tentativa via `runAttempt`, avalia via `critique`, e enquanto reprovado e o nº de regenerações já feitas for menor que `maxReflections`, gera uma nova tentativa com `buildRetryInput` e repete a crítica; concatena e reindexa (`at`) o trace de todas as tentativas com um evento `critique` após cada uma; soma `llmCalls` de todas as tentativas mais 1 por crítica; função pura/determinística — não instancia modelo nem lê `process.env` — depende de T001

**Checkpoint**: Schema do crítico, helpers puros e o laço de orquestração pura disponíveis para qualquer user story.

---

## Phase 3: User Story 1 - Obter uma resposta autocriticada e mais confiável (Priority: P1) 🎯 MVP

**Goal**: Uma estratégia de raciocínio existente, decorada com reflection, tem sua resposta avaliada por um crítico e, se reprovada, regenerada com o feedback no contexto — parando na primeira aprovação.

**Independent Test**: Envolver uma única estratégia com `withReflection` e verificar que a estratégia base é executada, um crítico avalia a resposta, uma reprovação dispara uma nova tentativa com o feedback incorporado, e a execução para assim que uma tentativa é aprovada.

### Tests for User Story 1

- [X] T004 [US1] Teste determinístico de `runReflectionLoop` em `src/agents/reflection.test.ts`: fake `critique` aprova já na 1ª tentativa — resultado final é a tentativa 1 (nenhuma regeneração), exatamente 1 evento `critique` no trace, `llmCalls` = tentativa + 1 (Acceptance Scenario 1 e 3) — depende de T003
- [X] T005 [US1] Teste determinístico de `runReflectionLoop` em `src/agents/reflection.test.ts`: fake `critique` reprova a 1ª tentativa e aprova a 2ª — confirma que a 2ª chamada de `runAttempt` recebe o input construído por `buildRetryInput(input, tentativa1.answer, feedback1)`, que o trace final concatena tentativa 1 → critique 1 → tentativa 2 → critique 2 em ordem crescente de `at`, e que `llmCalls` soma as 2 tentativas mais as 2 críticas (Acceptance Scenario 2 e 3) — depende de T004

### Implementation for User Story 1

- [X] T006 [US1] Implementar `withReflection(strategy, opts?: { maxReflections?: number })` em `src/agents/reflection.ts`: cria o crítico real (`createModel().withStructuredOutput(CritiqueSchema)`, prompt via `buildCritiquePrompt`), usa `strategy.run` como `runAttempt`, aplica `opts?.maxReflections ?? 2` e delega a `runReflectionLoop`; retorna um `ReasoningStrategy` com `name: "reflect:" + strategy.name` — nunca lança para reprovação/esgotamento (FR-011), apenas propaga falhas de infraestrutura — depende de T003
- [X] T007 [US1] Registrar `"reflect:react"` em `STRATEGY_NAMES` e `STRATEGIES` de `src/arena.ts` (`withReflection(reactStrategy)`) — depende de T006
- [X] T008 [US1] Validar manualmente via [quickstart.md](./quickstart.md): `npm run arena -- --input "quais alertas estão firing?" --strategies reflect:react` — confirmar uma tentativa completa de `react` seguida de ao menos um evento `critique` no trace, e que a execução termina com uma resposta final — depende de T007

**Checkpoint**: User Story 1 completa e testável de forma independente — uma estratégia decorada com reflection responde a uma pergunta operacional com o ciclo de crítica/regeneração funcionando.

---

## Phase 4: User Story 2 - Comparar estratégias com e sem reflection na arena (Priority: P2)

**Goal**: A arena permite selecionar `reflect:react`/`reflect:plan-and-execute` ao lado de suas versões base e comparar trace, resposta e métricas lado a lado, incluindo o custo extra de chamadas de modelo introduzido pela reflection.

**Independent Test**: Selecionar uma estratégia base e sua contraparte com reflection na arena, rodar a mesma pergunta operacional, e verificar que a saída identifica trace/métricas de cada uma separadamente, incluindo o custo adicional em chamadas de modelo da versão com reflection.

### Implementation for User Story 2

- [X] T009 [US2] Registrar `"reflect:plan-and-execute"` em `STRATEGY_NAMES` e `STRATEGIES` de `src/arena.ts` (`withReflection(planAndExecuteStrategy)`), ao lado de `reflect:react` (T007) — depende de T006
- [X] T010 [US2] Validar manualmente via [quickstart.md](./quickstart.md): `npm run arena -- --input "abra um incidente de severidade alta para o serviço checkout-api" --strategies react,reflect:react` — confirmar blocos identificáveis por estratégia, `reflect:react.metrics.llmCalls` maior que `react.metrics.llmCalls`, e um resumo comparativo final listando ambas (SC-004) — depende de T007
- [X] T011 [US2] Validar manualmente: `npm run arena -- --input "quais alertas estão firing?" --strategies plan-and-execute,reflect:plan-and-execute` — confirmar que a versão com reflection de Plan-and-Execute também produz evento(s) `critique` e uma resposta final, provando que o decorador se aplica a cada estratégia base de forma independente (FR-009) — depende de T009

**Checkpoint**: User Stories 1 e 2 funcionam de forma independente — a arena compara as quatro estratégias (`react`, `plan-and-execute`, `reflect:react`, `reflect:plan-and-execute`) sem repetir a pergunta manualmente.

---

## Phase 5: User Story 3 - Confiar que a reflection tem custo e tempo limitados (Priority: P3)

**Goal**: Nenhuma execução com reflection ultrapassa `maxReflections + 1` tentativas; ao esgotar o limite sem aprovação, a execução para de forma controlada com o histórico completo de críticas auditável no trace; falhas de infraestrutura continuam propagando, nunca são mascaradas.

**Independent Test**: Configurar (via fake) um crítico que nunca aprova e verificar que o número de tentativas nunca ultrapassa o limite configurado, que a execução termina com uma resposta (sem travar, sem lançar erro) e que o trace mostra o histórico completo de críticas.

### Tests for User Story 3

- [X] T012 [US3] Teste determinístico de `runReflectionLoop` em `src/agents/reflection.test.ts`: fake `critique` sempre reprova — confirma no máximo `maxReflections + 1` tentativas avaliadas (3 no padrão), o mesmo número de eventos `critique` no trace (um por tentativa), a resposta final é a da última tentativa (não aprovada), e nenhum erro é lançado (Acceptance Scenario 1 e 2 de US3, SC-002, SC-003) — depende de T003
- [X] T013 [US3] Teste determinístico de `runReflectionLoop` em `src/agents/reflection.test.ts` com `maxReflections = 0`: fake `critique` reprova a 1ª tentativa — confirma que nenhuma regeneração ocorre (1 tentativa, 1 evento `critique`), resposta final = tentativa 1 não aprovada (edge case da spec) — depende de T003
- [X] T014 [US3] Teste determinístico de `runReflectionLoop` em `src/agents/reflection.test.ts`: um fake `runAttempt` (ou `critique`) que rejeita a Promise faz `runReflectionLoop` propagar o mesmo erro, em vez de mascará-lo ou travar (FR-011) — depende de T003

### Implementation for User Story 3

- [X] T015 [US3] Validar manualmente a propagação de falha de infraestrutura real: rodar `npm run arena -- --input "quais alertas estão firing?" --strategies reflect:react` com `OPENROUTER_API_KEY` temporariamente ausente do ambiente — confirmar que a CLI encerra com um erro claro e código de saída não-zero, sem travar — depende de T006

**Checkpoint**: As três user stories funcionam de forma independente — custo e tempo da reflection comprovadamente limitados e auditáveis, tanto por testes automatizados quanto por uma execução real com falha de infraestrutura.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Fechar os gates de qualidade do projeto após todas as user stories

- [X] T016 [P] Rodar `npm run typecheck` e `npm test`; corrigir quaisquer falhas de tipo ou teste em `src/agents/reflection.ts`/`src/agents/reflection.test.ts`/`src/arena.ts`
- [X] T017 Rodar a validação completa de [quickstart.md](./quickstart.md) (passos 1–5) como passe final de aceitação — depende de T008, T010, T011, T015

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: sem tarefas — nada bloqueia o início da Phase 2
- **Foundational (Phase 2)**: BLOQUEIA todas as user stories
- **User Stories (Phase 3+)**: todas dependem da conclusão de Foundational
  - US1 (P1) não depende de US2/US3
  - US2 (P2) reaproveita `withReflection` (T006) e o registro de `reflect:react` na arena (T007), ambos criados em US1 — depende de US1 estar implementada, mas é testável e entregável de forma independente assim que T009–T011 estiverem prontas
  - US3 (P3) reaproveita `runReflectionLoop` (T003, Foundational) para seus próprios testes de forma isolada (T012–T014 não dependem de US1/US2); apenas a validação manual de falha de infraestrutura (T015) depende de `withReflection` (T006, US1)
- **Polish (Phase 6)**: depende de todas as user stories desejadas estarem completas

### Within Each User Story

- Testes de `runReflectionLoop` (fakes, sem rede) antes da implementação de `withReflection` (real, com modelo)
- `withReflection` antes do registro na arena
- Registro na arena antes da validação manual

### Parallel Opportunities

- Foundational: T002 e T003 em paralelo assim que T001 estiver pronto (arquivos diferentes — `reflection.test.ts` vs `reflection.ts` —, ambos dependem só de T001)
- Polish: T016 pode ser feito antes ou em paralelo da preparação de T017 (T017 é o passe manual final e depende de todas as validações anteriores)
- Esta feature tem um raio de arquivos pequeno e concentrado (essencialmente `src/agents/reflection.ts` + seu teste, mais uma extensão pequena de `src/arena.ts`) — ao contrário da feature 001, há pouca superfície para paralelismo real além do par acima; a maioria das tarefas edita o mesmo arquivo em sequência e por isso não é marcada `[P]`

---

## Parallel Example: Foundational

```bash
# Após T001 (schema + helpers puros) estar pronto, disparar em paralelo:
Task: "Testes determinísticos de buildRetryInput/buildCritiquePrompt em src/agents/reflection.test.ts"
Task: "Implementar runReflectionLoop em src/agents/reflection.ts"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Completar Phase 1: Setup (nada a fazer)
2. Completar Phase 2: Foundational (CRITICAL — bloqueia todas as stories)
3. Completar Phase 3: User Story 1
4. **PARAR e VALIDAR**: rodar o teste independente de US1 (T008)
5. Nesse ponto já existe `reflect:react` funcional na arena, com ciclo de crítica/regeneração completo

### Incremental Delivery

1. Setup + Foundational → schema do crítico, helpers e laço de orquestração prontos
2. US1 → validar independentemente → `reflect:react` demonstrável
3. US2 → validar independentemente → comparação `react` × `reflect:react` × `plan-and-execute` × `reflect:plan-and-execute` demonstrável
4. US3 → validar independentemente → custo/tempo limitados e falhas de infraestrutura comprovadamente não mascaradas
5. Polish → gates de qualidade (`typecheck`, `test`) e quickstart completo

---

## Notes

- [P] = arquivos diferentes, sem dependência de tarefa incompleta
- [Story] mapeia a tarefa à user story correspondente para rastreabilidade
- `withReflection` em si não tem teste automatizado (exige modelo real via OpenRouter) — sua verificação é a validação manual via quickstart, mesma decisão já tomada para `react.ts`/`plan-and-execute.ts` na feature 001
- Commitar após cada tarefa ou grupo lógico de tarefas (Constitution Principle VIII — Pequeno e Reversível)
- `npm run typecheck` e `npm test` DEVEM ficar verdes antes de qualquer commit (Constitution Principle V)
