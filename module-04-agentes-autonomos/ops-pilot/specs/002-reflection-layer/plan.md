# Implementation Plan: Camada de Reflection

**Branch**: `002-reflection-layer` | **Date**: 2026-09-03 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-reflection-layer/spec.md`

## Summary

`withReflection(strategy, opts)` decora qualquer `ReasoningStrategy` já existente (`react`, `plan-and-execute`) sem alterar sua implementação: executa a estratégia base, submete a resposta a um crítico (mesmo modelo da fábrica única, saída estruturada `{ approved, feedback }` validada com zod) que avalia a resposta contra as observações do trace daquela tentativa e, se reprovar, gera uma nova tentativa passando o feedback como contexto adicional na entrada da estratégia base. O ciclo para na primeira aprovação ou ao esgotar `maxReflections` (padrão 2, contando regenerações extras após a 1ª tentativa — até 3 tentativas no total). Cada avaliação crítica vira um evento `critique` no trace (tipo já existente em `TraceEvent`); as métricas somam `llmCalls` de todas as tentativas mais uma chamada por crítica, e a latência é medida de ponta a ponta no wrapper. A orquestração do ciclo (`runReflectionLoop`) é escrita como função pura recebendo as chamadas de tentativa/crítica como parâmetros injetados, isolando toda IO real (modelo, estratégia) na fábrica `withReflection`, o que permite testes determinísticos sem rede. A arena ganha `reflect:react` e `reflect:plan-and-execute` como estratégias adicionais, ao lado das duas já existentes.

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS (mesmo runtime da feature 001)

**Primary Dependencies**: `@langchain/core` (tipos de mensagem/callback já usados), `@langchain/openai` (via `createModel()` já existente, reaproveitado para o crítico com `withStructuredOutput`), `zod` (schema `{ approved: boolean, feedback: string }` do crítico, validado na fronteira modelo→domínio, mesmo padrão de `PlanSchema`/`ReplanSchema` em `plan-and-execute.ts`) — nenhuma dependência nova é introduzida

**Storage**: N/A — esta feature não introduz nem muta dados operacionais; reflection opera inteiramente sobre a resposta/trace já produzidos pela estratégia envolvida

**Testing**: `node:test` via `tsx` (`npm test`); a orquestração do ciclo de reflection (`runReflectionLoop`) e os construtores de prompt (`buildRetryInput`, `buildCritiquePrompt`) são testados com funções de tentativa/crítica falsas (fakes assíncronos determinísticos) — nenhum teste chama o modelo real ou depende de rede, mesma convenção da feature 001

**Target Platform**: processo Node.js server-side (dev local/CI); acionado via CLI (`npm run arena`) com os novos nomes de estratégia — nenhuma superfície HTTP nova

**Project Type**: projeto único (extensão de `src/agents/` e `src/arena.ts` já existentes)

**Performance Goals**: não é caminho de alto throughput; o que importa é o número de tentativas ficar estritamente limitado (`maxReflections + 1`), nunca um ciclo sem fim

**Constraints**: `maxReflections` DEVE ser configurável na construção do decorador e usar 2 como padrão (FR-005); cada tentativa individual continua respeitando o `maxIterations` já suportado pela estratégia base (FR-012); falhas de infraestrutura (crítico ou estratégia base) DEVEM propagar como erro, nunca ser mascaradas (FR-011); toda crítica realizada DEVE virar um evento `critique` no trace, em ordem cronológica (FR-007)

**Scale/Scope**: mesma escala de demonstração/avaliação da feature 001 (dataset semeado de 5 serviços/6 alertas); no pior caso (rejeição repetida), até `maxReflections + 1` execuções completas da estratégia base por chamada — padrão 3 no total

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | `runReflectionLoop`, `buildRetryInput` e `buildCritiquePrompt` (orquestração/formatação) ficam isolados de qualquer IO real; `withReflection` é a única função que instancia o modelo do crítico e invoca a estratégia base, mesmo padrão de separação já usado entre `react.ts`/`plan-and-execute.ts` (estratégia com IO) e `trace.ts` (helpers puros). |
| II. Validação na Fronteira | A saída do crítico (texto gerado pelo modelo) é uma fronteira de confiança como qualquer entrada externa — validada com um schema zod (`CritiqueSchema: { approved: boolean, feedback: string }`) via `withStructuredOutput` antes de qualquer decisão de fluxo, mesmo padrão de `PlanSchema`/`ReplanSchema`. |
| III. Erros de Domínio | Nenhuma classe de erro de domínio nova é necessária — reflection não introduz validação de entrada de usuário nova; falhas de infraestrutura (crítico ou estratégia indisponível) propagam como já ocorre em `react.ts`/`plan-and-execute.ts` (apenas `GraphRecursionError` é tratado especificamente por elas; o restante propaga). |
| IV. Funções Puras | `runReflectionLoop(runAttempt, critique, input, options, maxReflections)`, `buildRetryInput(...)` e `buildCritiquePrompt(...)` são determinísticas dado o mesmo input e as mesmas funções injetadas — nenhuma chama `createModel()` nem faz IO diretamente; `withReflection` é a única casca impura que as conecta ao mundo real. |
| V. Teste Obrigatório | `reflection.test.ts` cobre: aprovação na 1ª tentativa (nenhuma regeneração), reprovação seguida de aprovação, esgotamento de `maxReflections` (padrão e customizado), `maxReflections = 0`, e a formatação de `buildRetryInput`/`buildCritiquePrompt` — tudo com fakes, sem rede; `npm test`/`npm run typecheck` continuam gates obrigatórios. |
| VI. Segurança por Padrão | O crítico reaproveita a fábrica única `createModel()` (mesmas env vars já cobertas por essa fábrica, nenhum segredo novo); reflection nunca chama tools operacionais diretamente — toda ação sobre alertas/incidentes continua exclusivamente dentro da estratégia base decorada, sob as mesmas validações zod já existentes. |
| VII. Spec Antes de Código | Este plano segue `specs/002-reflection-layer/spec.md`, já validado e sem `[NEEDS CLARIFICATION]` pendente, e antecede `/speckit-tasks` + `/speckit-implement`. |
| VIII. Pequeno e Reversível | Mudança fica contida em um arquivo novo (`src/agents/reflection.ts` + seu teste) e uma extensão pequena e aditiva de `src/arena.ts` (novas entradas no registro de estratégias); nenhum arquivo da feature 001 é reescrito. |

Nenhuma violação identificada — **Complexity Tracking** não se aplica (tabela deixada vazia).

## Project Structure

### Documentation (this feature)

```text
specs/002-reflection-layer/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── agents/
│   ├── model.ts                      # [existente, inalterado] fábrica única do modelo — reaproveitada pelo crítico
│   ├── types.ts                      # [existente, inalterado] ReasoningStrategy/TraceEvent já incluem "critique"
│   ├── metrics.ts                    # [existente, inalterado] LlmCallCounter/startTimer/buildMetrics — reaproveitados
│   ├── trace.ts                      # [existente, inalterado] formatTrace — reaproveitado para montar o prompt do crítico
│   ├── react.ts                      # [existente, inalterado] decorável via withReflection sem modificação
│   ├── plan-and-execute.ts           # [existente, inalterado] decorável via withReflection sem modificação
│   ├── reflection.ts                 # [NOVO] withReflection(strategy, opts) + runReflectionLoop (puro) +
│   │                                  #        buildRetryInput/buildCritiquePrompt (puros) + wiring do crítico real
│   └── reflection.test.ts            # [NOVO] testes determinísticos do ciclo de reflection (fakes, sem rede)
└── arena.ts                          # [alterado] STRATEGY_NAMES/STRATEGIES ganham "reflect:react" e
                                       #            "reflect:plan-and-execute" ao lado das duas existentes
```

**Structure Decision**: Projeto único (Option 1), mesma estrutura da feature 001. Nenhum diretório novo é criado — a feature inteira cabe em um módulo novo (`src/agents/reflection.ts` + teste) e uma extensão aditiva de `src/arena.ts`. Nenhum arquivo de `src/domain/`, `src/services/` ou `src/models/` é tocado, pois reflection não introduz nem muda dados operacionais — apenas decora estratégias de raciocínio já existentes.

## Complexity Tracking

*Nenhuma violação da Constitution Check — tabela não aplicável.*
