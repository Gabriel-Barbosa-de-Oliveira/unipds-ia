# notas-api

Esqueleto de projeto Node.js + TypeScript para uma API de notas.

## Stack

- Node.js com ES Modules (`"type": "module"`)
- TypeScript (target `ES2022`, `module`/`moduleResolution` `NodeNext`, `strict`)
- [tsx](https://github.com/privatenumber/tsx) para rodar TypeScript diretamente, sem etapa de build
- [zod](https://zod.dev) para validação de dados
- Test runner nativo do Node (`node --test`)

## Estrutura

```
notas-api/
├── src/            # código-fonte (ainda vazio)
├── dist/           # saída da build (gerado, ignorado no git)
├── package.json
├── tsconfig.json
└── .gitignore
```

## Pré-requisitos

- Node.js 20+ (necessário para `module`/`moduleResolution: NodeNext` e para o test runner nativo)
- npm

## Como instalar

```bash
cd module-04-agentes-autonomos/notas-api
npm install
```

## Scripts disponíveis

| Script              | Comando                                      | Descrição                                    |
| ------------------- | --------------------------------------------- | --------------------------------------------- |
| `npm run dev`       | `tsx src/index.ts`                            | Executa a API em modo desenvolvimento          |
| `npm run cli`       | `tsx src/cli.ts`                              | Executa a CLI do projeto                       |
| `npm test`          | `node --import tsx --test "src/**/*.test.ts"` | Roda os testes                                 |
| `npm run typecheck` | `tsc --noEmit`                                | Verifica os tipos sem gerar arquivos de saída  |

## Status atual

Este é apenas o esqueleto do projeto (configuração, dependências e scripts). A pasta `src/` ainda não contém código — `npm run typecheck` e os demais scripts só funcionarão plenamente após a criação de `src/index.ts`, `src/cli.ts` e dos arquivos de teste.
