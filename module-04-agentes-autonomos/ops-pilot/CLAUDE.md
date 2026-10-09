# OpsPilot

Copiloto de plantão que gerencia alertas e incidentes de produção. A API é um agente LangChain/LangGraph rodando sobre OpenRouter.

## Stack

- Node 24 LTS
- TypeScript ESM, `strict`
- zod na fronteira (HTTP/CLI)
- Testes com `node:test` via `tsx`
- Express como servidor HTTP
- SQLite via `node:sqlite` (`DatabaseSync`) como banco — arquivo local (caminho via env `OPSPILOT_DB`), `:memory:` em testes
- Frontend em `web/` (Vite + React + TS), pacote npm separado — design em `.github/instructions/design.instructions.md`

## Comandos

- `npm run dev` — inicia a API (`src/index.ts`)
- `npm run arena` — roda `src/arena.ts` (`--input`, `--strategies`, `--max-iterations`)
- `npm run seed` — semeia/restaura o SQLite com o dataset canônico (`src/scripts/seed.ts`)
- `npm run bench` — roda `src/bench.ts`
- `npm test` — roda os testes (`node --import tsx --test`)
- `npm run typecheck` — `tsc --noEmit`
- `npm --prefix web run dev|build|preview|test|typecheck` — war room em `http://localhost:5173/opspilot/`
- Publicação: push em `master` que toca `web/` publica no GitHub Pages via `.github/workflows/ops-pilot-web-pages.yml` (na raiz do repositório `unipds-ia`); PR só checa

## Env da API

- `OPSPILOT_CORS_ORIGINS` — origens do navegador autorizadas, separadas por vírgula (padrão `http://localhost:5173`)
- `OPSPILOT_APPROVAL_TTL_MS` — validade de uma ação aguardando aprovação (padrão 900000 = 15 min)
- `OPSPILOT_WEB_BASE` — caminho base do build do `web/` (padrão `/opspilot/`; o Pages usa `/<repo>/opspilot/`)

## Convenções

- Camadas padrão MVC (Model, Service, Controller)
- Entrada externa é sempre validada com zod
- Erros de domínio são classes, traduzidas na borda
- Lógica nova nasce com teste; typecheck e teste sempre verdes
- Nunca commitar segredos nem ler `.env`
- Sempre utilize funções puras

## Fluxo

Seguir o fluxo do Spec Kit (GitHub Copilot): `/speckit.specify` → `/speckit.plan` → `/speckit.tasks` → `/speckit.implement`. Specs devem ser versionadas.
