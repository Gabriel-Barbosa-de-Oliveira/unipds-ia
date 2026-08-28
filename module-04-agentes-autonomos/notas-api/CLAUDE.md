# notas-api

## Stack

- Node 24 LTS
- TypeScript ESM, `strict`
- zod na fronteira (HTTP/CLI)
- Testes com `node:test` via `tsx`
- Sem framework HTTP — `node:http` de propósito

## Comandos

- `npm run dev` — API em `localhost:3000`
- `npm run cli` — executa a CLI
- `npm test` — roda os testes
- `npm run typecheck` — verifica os tipos, sem emitir arquivos

## Estrutura

- `src/domain` — tipos + schemas zod, sem IO
- `src/store` — persistência in-memory atrás de uma interface
- `src/service` — regras de negócio
- `src/http` — camada HTTP
- `src/cli.ts` — entrada da CLI
- `specs/` — artefatos do fluxo spec-driven

## Convenções

- Camadas não pulam: `http`/`cli` → `service` → `store`
- Entrada externa é sempre validada com zod
- Erros de domínio são classes, traduzidas na borda
- Lógica nova nasce com teste; typecheck e teste sempre verdes
- Nunca commitar segredos nem ler `.env`

## Fluxo de trabalho

`/especificar` → `/planejar` → `/tarefas` → implementar, com revisão humana entre as fases. Artefatos em `specs/`.
