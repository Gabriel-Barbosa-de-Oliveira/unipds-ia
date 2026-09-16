# Implementation Plan: Medição de Contexto

**Branch**: `009-context-tokens` | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/009-context-tokens/spec.md`

## Summary

Instrumenta `POST /chat` com medição de tokens de prompt e de composição de contexto. Um novo módulo, `src/context/tokens.ts`, expõe `estimateTokens` (heurística `chars/4`, pura) e `UsageCollector` (callback LangChain — `extends BaseCallbackHandler`, mesmo mecanismo já usado por `LlmCallCounter`) que agrega o uso real de tokens relatado pelo provedor (`handleLLMEnd`, `output.llmOutput?.tokenUsage?.promptTokens`, campo do `@langchain/openai`) e cai para a estimativa quando o real não vem. `UsageCollector` é passado como callback adicional em todo ponto de chamada ao modelo — `react.ts`, os três nós de `plan-and-execute.ts`, e `reflection.ts#critique` (que hoje não passa nenhum callback; esta é a mudança que faz o token accounting cobrir também as chamadas de crítica/regeneração). `Metrics` (`src/agents/types.ts`) ganha `promptTokens`/`tokenSource`; a resposta de `/chat` ganha também `metrics.contextBreakdown` — uma decomposição (mensagem atual/histórico/fatos lembrados), calculada só no controller HTTP via `buildContextBreakdown` (também em `context/tokens.ts`), já que só ele conhece essas três peças separadamente antes de compor o prompt final. Nenhuma tabela nova, nenhum campo novo de requisição — tudo aditivo em `metrics`.

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS (mesmo runtime das features 001–008; nenhuma mudança de runtime)

**Primary Dependencies**: nenhuma nova — reaproveita `@langchain/core` (`BaseCallbackHandler`, `LLMResult`, `Serialized`, mesmos tipos já usados por `agents/metrics.ts`).

**Storage**: N/A — nenhuma tabela nova, nenhuma persistência (research.md item 9); toda medição é transiente, calculada por resposta.

**Testing**: `node:test` via `tsx` (`npm test`). Novo: `src/context/tokens.test.ts` (`estimateTokens` com strings conhecidas — vazia, curta, múltiplos de 4; `UsageCollector#handleLLMStart`/`#handleLLMEnd` com `LLMResult`/`Serialized` forjados à mão, cobrindo real/estimado/misto entre múltiplas chamadas simuladas; `mergeTokenUsage` com as 3 combinações de `source`; `buildContextBreakdown` com partes vazias e não-vazias, confirmando `total` sempre igual à soma — tudo offline, sem rede). Estendidos: `src/agents/reflection.test.ts` (`critique`/`runReflectionLoop` combinando `tokenUsage` de tentativa + crítica via fakes), `src/http/server.test.ts` (`metrics.promptTokens`/`tokenSource`/`contextBreakdown` presentes e coerentes com um `MemoryStore`/estratégia fake).

**Target Platform**: processo Node.js server-side (mesmo runtime das features anteriores) — `npm run dev` passa a expor `promptTokens`/`tokenSource`/`contextBreakdown` em toda resposta real de `/chat`; `npm run bench`/`npm run arena` não passam por `/chat` e ficam fora do escopo (mesmo raciocínio de `006`/`007`/`008`).

**Project Type**: projeto único — 1 diretório novo (`src/context/`, pedido explicitamente pelo usuário: `tokens.ts`), extensões pontuais a `react.ts`/`plan-and-execute.ts`/`reflection.ts`/`agents/types.ts`/`agents/metrics.ts`/`http/server.ts`.

**Performance Goals**: `UsageCollector`/`estimateTokens` são operações em memória sobre texto já disponível (sem I/O, sem chamada extra ao modelo) — custo desprezível frente à latência já dominante da chamada real ao modelo de raciocínio.

**Constraints**: nenhuma mudança de comportamento observável em `answer`/raciocínio (FR-008) — puramente aditivo em `metrics`; `promptTokens`/`tokenSource`/`contextBreakdown` presentes em toda resposta `200` que aciona o modelo, nunca omitidos (SC-002); `contextBreakdown.total` sempre igual à soma das partes, por construção (FR-006); `contextBreakdown` e `promptTokens` medem coisas diferentes e não são reconciliados entre si (research.md item 8); métricas de requisições concorrentes nunca se misturam (FR-009, mesmo padrão de isolamento já usado por `LlmCallCounter`).

**Scale/Scope**: 1 arquivo novo de produção (`src/context/tokens.ts`) + 1 de teste; 5 call sites ganham `usageCollector` no array de `callbacks` (react.ts ×1, plan-and-execute.ts ×3, reflection.ts ×1 — este último ganhando `callbacks` pela primeira vez); `Metrics` ganha 2 campos; a resposta de `/chat` ganha 1 campo aninhado (`contextBreakdown`) — nenhuma tabela nova, nenhum campo novo de requisição, nenhuma dependência de pacote nova.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | `src/context/tokens.ts` é a única camada que fala com o callback do LangChain para medir tokens/contexto; `src/agents/metrics.ts` só monta o `Metrics` final a partir do que `UsageCollector` calculou (mesmo papel que já tem hoje para `llmCalls`); `src/http/server.ts` só chama `buildContextBreakdown` e anexa o resultado à resposta, sem lógica de estimativa própria. |
| II. Validação na Fronteira | Nenhum campo novo de requisição — `ChatRequestSchema` inalterado. A fronteira relevante aqui é com o LangChain/provedor do modelo, não com HTTP/CLI; `UsageCollector` valida o formato do uso real com um `typeof === "number"` antes de usá-lo, caindo para a estimativa em qualquer outro caso. |
| III. Erros de Domínio | Nenhuma classe de erro nova — ausência de uso real não é um erro, é só o caso que aciona a estimativa (mesmo espírito de "nada encontrado" já modelado como valor em features anteriores, não como exceção). |
| IV. Funções Puras | `estimateTokens`, `mergeTokenUsage` e `buildContextBreakdown` são puras — mesma entrada, mesma saída, testáveis com valores forjados; o único efeito colateral (ouvir callbacks do LangChain) fica isolado em `UsageCollector`, mesmo padrão já usado por `LlmCallCounter`. |
| V. Teste Obrigatório | `context/tokens.test.ts` cobre toda a lógica pura e o callback (com `LLMResult`/`Serialized` forjados, sem rede); extensões de `reflection.test.ts`/`server.test.ts` cobrem a composição entre chamadas e o contrato de `/chat` — nenhum teste novo depende de `OPENROUTER_API_KEY`. |
| VI. Segurança por Padrão | Nenhum código desta feature lê `.env`; a medição nunca altera `answer` nem decisões do raciocínio (FR-008) — estritamente informativa, sem novo vetor de risco. |
| VII. Spec Antes de Código | Este plano segue `specs/009-context-tokens/spec.md`, validado e sem `[NEEDS CLARIFICATION]` pendente. |
| VIII. Pequeno e Reversível | Toda mudança de contrato é aditiva (campos novos em `metrics`, nenhum removido/renomeado); os 5 call sites que ganham `usageCollector` são alterações de uma linha cada; `contextBreakdown.total` definido por construção (research.md item 7) evita a complexidade de reconciliar contra `promptTokens` real (research.md item 8) — menor raio de mudança possível para entregar as 3 user stories. |

Nenhuma violação identificada — **Complexity Tracking** não se aplica (tabela deixada vazia).

## Project Structure

### Documentation (this feature)

```text
specs/009-context-tokens/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md         # Phase 1 output (/speckit-plan command)
├── quickstart.md         # Phase 1 output (/speckit-plan command)
├── contracts/            # Phase 1 output (/speckit-plan command)
│   ├── post-chat.md
│   └── tokens.md
└── tasks.md              # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── context/                          # [NOVO diretório, pedido explicitamente]
│   ├── tokens.ts                     # [NOVO] estimateTokens, TokenUsage/TokenSource, mergeTokenUsage, UsageCollector, ContextBreakdown, buildContextBreakdown
│   └── tokens.test.ts                # [NOVO] toda a lógica pura + UsageCollector com LLMResult/Serialized forjados
├── agents/
│   ├── types.ts                      # [alterado] Metrics ganha promptTokens/tokenSource
│   ├── metrics.ts                    # [alterado] buildMetrics(counter, usageCollector, latencyMs)
│   ├── react.ts                      # [alterado] callbacks: [counter, usageCollector]
│   ├── plan-and-execute.ts           # [alterado] callbacks: [counter, usageCollector] nos 3 nós (planner/executor/replanner)
│   ├── reflection.ts                 # [alterado] critique() ganha callbacks (novo) e retorna { verdict, tokenUsage }; runReflectionLoop combina via mergeTokenUsage
│   ├── reflection.test.ts            # [alterado] cobre a combinação de tokenUsage entre tentativa e crítica
│   ├── tools.ts, model.ts, message-trace.ts, index.ts  # [existentes, inalterados]
├── http/
│   ├── server.ts                     # [alterado] chama buildContextBreakdown antes de compor o prompt; metrics.contextBreakdown na resposta
│   └── server.test.ts                # [alterado] + cenários de metrics.promptTokens/tokenSource/contextBreakdown
└── domain/, memory/, store/, services/, mcp/, bench.ts, arena.ts  # [existentes, inalterados]
```

**Structure Decision**: Projeto único (Option 1), mesma estrutura das features 001–008. Único diretório novo: `src/context/` — pedido explicitamente pelo usuário (`tokens.ts`) e coeso (toda a lógica de medição de tokens/contexto vive nele); `Metrics`/`buildMetrics` (já existentes) são estendidos em vez de duplicados, mantendo um único lugar onde o formato final de métricas de uma execução é montado.
