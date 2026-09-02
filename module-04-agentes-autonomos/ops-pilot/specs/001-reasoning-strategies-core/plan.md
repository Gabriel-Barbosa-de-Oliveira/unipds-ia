# Implementation Plan: Núcleo de Raciocínio (Reasoning Strategies Core)

**Branch**: `001-reasoning-strategies-core` | **Date**: 2026-09-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-reasoning-strategies-core/spec.md`

## Summary

O núcleo de raciocínio do OpsPilot expõe uma interface comum `ReasoningStrategy` (nome + `run(input)` → resposta, trace tipado, métricas) implementada por duas estratégias — ReAct (agente pré-construído do LangGraph) e Plan-and-Execute (grafo planner/executor/replanner, máx. 8 passos) — sobre três ferramentas operacionais validadas na fronteira com zod (`list_alerts`, `open_incident`, `resolve_incident`). Uma fábrica única (`src/agents/model.ts`) cria o modelo de raciocínio via OpenRouter (`OPENROUTER_API_KEY`/`OPENROUTER_MODEL`, `temperature: 0`). O armazenamento operacional segue a separação de camadas da constitution: lógica de domínio pura sobre um adaptador in-memory (usado por padrão pelas tools, pela arena e pelos testes — determinístico, sem rede) e um adaptador Sequelize + MySQL (stack obrigatória do projeto), ambos atrás da mesma interface de repositório, alimentados por um script de seed reprodutível (5 serviços, 6 alertas: 3 firing, 3 resolved). Uma arena mínima (`src/arena.ts`, flags `--strategies` e `--max-iterations`) roda a mesma entrada em uma ou mais estratégias e imprime trace e métricas lado a lado.

## Technical Context

**Language/Version**: TypeScript ESM `strict` sobre Node 24 LTS

**Primary Dependencies**: `@langchain/core`, `@langchain/langgraph` (agente ReAct pré-construído + `StateGraph` para Plan-and-Execute), `@langchain/openai` (cliente compatível OpenAI apontado para o baseURL do OpenRouter), `zod` (validação na fronteira: args de tool e flags de CLI), `sequelize` + `mysql2` (persistência mandatória), `express` (já presente no shell da app, não afetado por esta feature)

**Storage**: MySQL via Sequelize para o adaptador de persistência durável (`Service`, `Alert`, `Incident`); um adaptador in-memory equivalente (mesma interface de repositório) é o padrão usado pelas tools/arena/testes desta feature, por ser determinístico e não depender de rede — ver [research.md](./research.md) para a decisão completa

**Testing**: `node:test` via `tsx` (`npm test`), arquivos `*.test.ts` colocados ao lado do módulo testado (convenção já usada no projeto), cobrindo o store puro e a formatação de trace — determinístico e sem rede (FR-014)

**Target Platform**: processo Node.js server-side (dev local / CI); esta feature é acionada via CLI (`npm run arena`) e por chamadas internas de outras camadas — nenhuma superfície HTTP nova é exigida pelo spec

**Project Type**: projeto único (extensão do backend Node/Express/TypeScript já existente em `ops-pilot`)

**Performance Goals**: não é um caminho de alto throughput; o que importa é a estratégia terminar dentro do limite de passos configurado (não corridas ilimitadas), não requisições/segundo

**Constraints**: estratégias DEVEM respeitar um limite configurável de iterações/passos (padrão 8 para Plan-and-Execute, FR-005/FR-006); toda entrada de tool DEVE ser validada com zod antes de qualquer mutação no store (FR-008); lógica de store e formatação de trace DEVEM ser testáveis sem rede (FR-014)

**Scale/Scope**: escala de demonstração/avaliação — 5 serviços, 6 alertas semeados, contagem pequena e não limitada de incidentes; uma execução por vez (spec Assumptions), sem requisito de concorrência

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como esta feature cumpre |
|---|---|
| I. Camadas Explícitas | Lógica de domínio (regras sobre alertas/incidentes) fica em `src/domain/` como funções puras; toda IO (LLM, MySQL, seed) fica isolada em `src/services/`, `src/models/` e `src/agents/`. `src/agents/tools.ts` delega a mutação de estado ao service layer, nunca manipula IO diretamente. |
| II. Validação na Fronteira | Args de cada tool (`list_alerts`, `open_incident`, `resolve_incident`) e as flags da CLI da arena são validados com schemas zod antes de qualquer efeito. |
| III. Erros de Domínio | `ServiceNotFoundError`, `IncidentNotFoundError`, `InvalidSeverityError` (classes) são lançados pelo domínio/service; a tradução para uma observação de erro no trace acontece na borda (`src/agents/tools.ts`), nunca dentro da lógica pura. |
| IV. Funções Puras | `src/domain/ops-store.ts` contém apenas funções puras (mesmo estado de entrada → mesmo estado de saída); efeitos colaterais (mutação persistida, IO) ficam nos adaptadores de `src/services/`. |
| V. Teste Obrigatório | Store puro e formatação de trace ganham testes `node:test` determinísticos e sem rede antes de qualquer PR; `npm test`/`npm run typecheck` seguem como gate. |
| VI. Segurança por Padrão | Nenhum segredo é commitado; `OPENROUTER_API_KEY` e credenciais MySQL são lidos de `process.env` em runtime pela aplicação (não pelo agente/Claude); `resolve_incident`/`open_incident` só afetam o store semeado de demonstração. |
| VII. Spec Antes de Código | Este plano segue o spec já aprovado (`spec.md`) e antecede `/speckit-tasks` + `/speckit-implement`. |
| VIII. Pequeno e Reversível | Detalhamento em tarefas pequenas e commitáveis fica a cargo de `/speckit-tasks`; nenhuma tarefa aqui exige uma mudança monolítica. |

Nenhuma violação identificada — **Complexity Tracking** não se aplica (tabela deixada vazia).

## Project Structure

### Documentation (this feature)

```text
specs/001-reasoning-strategies-core/
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
├── index.ts                          # Express entrypoint existente (não alterado por esta feature)
├── agents/
│   ├── model.ts                      # Fábrica única do modelo (OpenRouter via @langchain/openai, temperature 0)
│   ├── types.ts                      # ReasoningStrategy, TraceEvent (union), Metrics, RunResult
│   ├── tools.ts                      # zod schemas + tools LangChain: list_alerts/open_incident/resolve_incident
│   ├── react.ts                      # Estratégia ReAct (createReactAgent do LangGraph) + captura de trace
│   ├── plan-and-execute.ts           # Estratégia Plan-and-Execute (StateGraph: planner/executor/replanner, máx. 8 passos)
│   └── trace.ts                      # Helpers puros de formatação/serialização de trace (usados por arena e testes)
├── domain/
│   ├── ops-store.ts                  # Tipos + funções puras: listAlerts, openIncident, resolveIncident sobre estado imutável
│   ├── ops-store.test.ts             # Testes determinísticos do domínio (sem rede)
│   └── errors.ts                     # Classes de erro de domínio (ServiceNotFoundError, IncidentNotFoundError, InvalidSeverityError)
├── services/
│   ├── ops-store.repository.ts       # Interface de repositório comum (contrato) usada pelas tools
│   ├── ops-store.memory.ts           # Adaptador in-memory (padrão para tools/arena/testes)
│   └── ops-store.sequelize.ts        # Adaptador Sequelize + MySQL (stack obrigatória)
├── models/
│   └── sequelize/
│       ├── connection.ts             # Instância Sequelize (MySQL) a partir de env vars
│       ├── service.model.ts
│       ├── alert.model.ts
│       └── incident.model.ts
├── scripts/
│   └── seed.ts                       # Script de seed idempotente (5 serviços, 6 alertas: 3 firing, 3 resolved)
└── arena.ts                          # CLI: roda 1+ estratégias sobre o mesmo input, flags --strategies/--max-iterations
```

**Structure Decision**: Projeto único (Option 1), estendendo o backend Node/Express/TypeScript existente do `ops-pilot`. Nenhum diretório `tests/` separado é introduzido — o projeto já usa arquivos `*.test.ts` colocados ao lado do código testado (`npm test` roda `src/**/*.test.ts`), convenção mantida aqui. Os caminhos de arquivo explicitamente citados no pedido original (`src/agents/model.ts`, `src/agents/tools.ts`, `src/agents/react.ts`, `src/agents/plan-and-execute.ts`, `src/arena.ts`) são respeitados literalmente; os demais caminhos (`src/domain/`, `src/services/`, `src/models/sequelize/`, `src/scripts/seed.ts`) implementam a separação MVC exigida pela constitution para sustentar esses arquivos sem IO direto no domínio.

## Complexity Tracking

*Nenhuma violação da Constitution Check — tabela não aplicável.*
