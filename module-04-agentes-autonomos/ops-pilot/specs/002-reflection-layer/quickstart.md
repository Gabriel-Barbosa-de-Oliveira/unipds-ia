# Quickstart: Validar a Camada de Reflection

Guia de validação ponta a ponta para esta feature após a implementação (`/speckit-tasks` → `/speckit-implement`). Detalhes de contrato ficam em [contracts/](./contracts/); modelo de dados em [data-model.md](./data-model.md). Pressupõe a feature 001 já implementada (ver [../001-reasoning-strategies-core/quickstart.md](../001-reasoning-strategies-core/quickstart.md) para os pré-requisitos gerais de ambiente/`.env`).

## Pré-requisitos

- Node 24 LTS e dependências instaladas (`npm install`).
- `.env` local com `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` (necessário só para os passos 3–4, que chamam o modelo real; o passo 1 não precisa de rede).

## 1. Rodar os testes determinísticos (sem rede)

```sh
npm run typecheck
npm test
```

**Esperado**: `reflection.test.ts` passa sem qualquer chamada de rede ou credencial configurada, cobrindo com fakes: aprovação já na 1ª tentativa (nenhuma regeneração), reprovação seguida de aprovação, esgotamento de `maxReflections` (padrão 2 e customizado), `maxReflections = 0`, e a formatação de `buildRetryInput`/`buildCritiquePrompt` — valida FR-001 a FR-008, FR-011, FR-012 e SC-001/SC-002/SC-003.

## 2. Semear o dataset canônico (se ainda não estiver semeado)

```sh
npm run seed
```

**Esperado**: mesmo comportamento já validado pela feature 001 — necessário apenas se for exercitar o adaptador MySQL; o caminho padrão de arena/tools usa o adaptador in-memory.

## 3. Comparar uma estratégia base e sua versão com reflection

```sh
npm run arena -- --input "abra um incidente de severidade alta para o serviço checkout-api" \
  --strategies react,reflect:react
```

**Esperado** (User Story 1, User Story 2, SC-004):
- Bloco `react`: trace e métricas de uma execução única, sem eventos `critique` (comportamento inalterado da feature 001).
- Bloco `reflect:react`: mesmo trace inicial de uma tentativa de `react`, seguido de ao menos um evento `critique` antes da resposta final ser considerada definitiva.
- `reflect:react.metrics.llmCalls` é maior que `react.metrics.llmCalls` (ao menos +1, pela avaliação crítica) — visível no resumo comparativo final.

## 4. Observar uma regeneração completa

```sh
npm run arena -- --input "quais alertas estão firing?" --strategies reflect:plan-and-execute
```

**Esperado** (User Story 1, User Story 3, SC-003):
- Se o crítico aprovar já na 1ª tentativa: o trace mostra exatamente 1 evento `critique` e nenhuma tentativa adicional (SC-001).
- Se o crítico reprovar: o trace mostra o padrão `tentativa 1 → critique → tentativa 2 → critique → ...` até uma aprovação ou até 3 tentativas no total (`maxReflections` padrão 2) — nunca mais que isso (SC-002).
- Em qualquer um dos dois casos, a execução termina com uma resposta final (nunca trava, nunca lança erro não tratado) — SC-005.

## 5. Confirmar que as estratégias base continuam inalteradas

```sh
npm run arena -- --input "quais alertas estão firing?" --strategies react,plan-and-execute
```

**Esperado**: comportamento idêntico ao já validado pelo quickstart da feature 001 — nenhum evento `critique` aparece, nenhuma métrica extra é somada; confirma que `withReflection` é puramente aditivo (spec Assumptions).
