# OpsPilot

Copiloto de plantão que gerencia alertas e incidentes de produção. A API é um agente LangChain/LangGraph rodando sobre OpenRouter.

## Stack

- Node 24 LTS
- TypeScript ESM, `strict`
- zod na fronteira (HTTP/CLI)
- Testes com `node:test` via `tsx`
- Express como servidor HTTP
- Sequelize + MySQL como banco

## Comandos

- `npm run dev` — inicia a API (`src/index.ts`)
- `npm run arena` — roda `src/arena.ts` (`--input`, `--strategies`, `--max-iterations`)
- `npm run seed` — semeia/restaura o MySQL com o dataset canônico (`src/scripts/seed.ts`)
- `npm run bench` — roda `src/bench.ts`
- `npm test` — roda os testes (`node --import tsx --test`)
- `npm run typecheck` — `tsc --noEmit`

## Convenções

- Camadas padrão MVC (Model, Service, Controller)
- Entrada externa é sempre validada com zod
- Erros de domínio são classes, traduzidas na borda
- Lógica nova nasce com teste; typecheck e teste sempre verdes
- Nunca commitar segredos nem ler `.env`
- Sempre utilize funções puras

## Fluxo

Seguir o fluxo do Spec Kit (GitHub Copilot): `/speckit.specify` → `/speckit.plan` → `/speckit.tasks` → `/speckit.implement`. Specs devem ser versionadas.
