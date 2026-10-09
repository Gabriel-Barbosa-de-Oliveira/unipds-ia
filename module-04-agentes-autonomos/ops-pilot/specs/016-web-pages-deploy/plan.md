# Implementation Plan: Publicação da War Room no GitHub Pages

**Branch**: `016-web-pages-deploy` | **Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/016-web-pages-deploy/spec.md`

## Summary

**Workflow.** Um workflow na raiz do monorepo (`.github/workflows/ops-pilot-web-pages.yml`), filtrado por mudanças em `ops-pilot/web/`, tem dois jobs:
- `check`, só com leitura: `npm ci`, typecheck, testes, build com o caminho base do Pages e `upload-pages-artifact`;
- `deploy`, com `pages: write` + `id-token: write` e ambiente `github-pages`: `deploy-pages`.

Os PRs rodam só o `check`. A fila de publicação não cancela o deploy em andamento. As actions oficiais são fixadas por SHA.

**War room.** O caminho base deixa de ser fixo: `resolveBasePath(OPSPILOT_WEB_BASE)`, com `/opspilot/` no local e `/unipds-ia/opspilot/` no Pages. O artifact põe a war room em `opspilot/`, com um redirecionamento na raiz do site.

**Documentação.** O `ops-pilot` ganha um README (rodar local, variáveis, publicação, ativação única do Pages, conexão com a API via CORS), e o README da raiz ganha o link.

Os detalhes estão em [research.md](./research.md).

## Technical Context

**Language/Version**: YAML do GitHub Actions; TypeScript `strict` no `web/` (Node 24 LTS no runner)

**Primary Dependencies**: `actions/checkout` v7.0.1, `actions/setup-node` v7.1.0, `actions/upload-pages-artifact` v5.0.0 e `actions/deploy-pages` v5.0.1, todas fixadas por SHA (research.md item 5). Nenhuma dependência npm nova.

**Storage**: N/A (o artifact do Pages é gerenciado pelo GitHub)

**Testing**: `node:test` via `tsx` para `resolveBasePath` e para a validação do `dist/` com o base do Pages, rodados no próprio CI. O workflow é validado pelo [quickstart.md](./quickstart.md), com `actionlint` opcional no local.

**Target Platform**: GitHub Actions (`ubuntu-latest`) → GitHub Pages (site de projeto)

**Project Type**: CI/CD + configuração de build da aplicação web existente

**Performance Goals**: da integração ao site no ar em até 10 minutos (SC-001). O alvo é menos de 3 minutos, com o cache npm.

**Constraints**: no máximo 3 permissões e nenhuma escrita no repositório (SC-004), sem segredos (FR-012), `npm ci` com lockfile (FR-009) e o dev local sem mudança (`/opspilot/`).

**Scale/Scope**: 1 workflow novo, 1 função pura (+ teste), `vite.config.ts` e `base-path.test.ts` ajustados, 1 README novo, 1 README atualizado e o `.gitignore` (`_site/`)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Status | Como |
|---|---|---|
| I. Camadas explícitas | ✅ | A lógica nova (`resolveBasePath`) fica em `web/src/lib/`, que é puro. O workflow só orquestra comandos que já existem. |
| II. Validação na fronteira | ✅ | `OPSPILOT_WEB_BASE` é entrada externa do build e é validada por `resolveBasePath`, que faz o build falhar com valor inválido. |
| III. Erros de domínio | ✅ | Não há domínio novo. A falha de configuração aborta o build com mensagem clara. |
| IV. Funções puras | ✅ | `resolveBasePath` e `localAssetPaths` são puras. |
| V. Teste obrigatório | ✅ | `resolveBasePath` tem teste. O CI passa a executar os gates do `web/` em todo PR e antes de toda publicação. |
| VI. Segurança | ✅ | Permissões mínimas e por job, checagem separada da publicação, actions oficiais fixadas por SHA, `persist-credentials: false` e nenhum segredo. O Pages só aceita deploy do `master`. |
| VII. Spec antes de código | ✅ | spec → plan → tasks. |
| VIII. Pequeno e reversível | ✅ | Incrementos: base configurável → workflow → README. Cada um vira um commit. Remover o workflow desfaz a publicação. |
| Stack | ✅ | O GitHub Actions é infraestrutura de CI, não stack de runtime. Sem amendment (research.md item 11). |

**Re-check pós-design (Fase 1)**: os contratos mantêm todos os itens. Não há violações.

## Project Structure

### Documentation (this feature)

```text
specs/016-web-pages-deploy/
├── spec.md
├── plan.md              # este arquivo
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── workflow.md      # gatilhos, permissões, jobs, passos
│   └── readme.md        # seções dos READMEs
├── checklists/requirements.md
└── tasks.md             # /speckit.tasks
```

### Source Code (repository root)

```text
unipds-ia/                                        # raiz do repositório git
├── .github/workflows/
│   └── ops-pilot-web-pages.yml                   # NOVO
├── README.md                                     # ATUALIZADO: link para o OpsPilot
└── module-04-agentes-autonomos/ops-pilot/
    ├── README.md                                 # NOVO
    ├── CLAUDE.md                                 # + linha sobre a publicação e OPSPILOT_WEB_BASE
    └── web/
        ├── vite.config.ts                        # base: resolveBasePath(process.env.OPSPILOT_WEB_BASE)
        ├── .gitignore                            # NOVO: _site/
        └── src/lib/
            ├── base-path.ts                      # + resolveBasePath
            └── base-path.test.ts                 # testa resolveBasePath; valida dist/ com o base efetivo
```

**Structure Decision**: o workflow fica na raiz, como o GitHub exige, e aponta para `ops-pilot/web` via `working-directory` e `paths`. A mudança de código fica restrita ao `web/`. A API não muda.

## Complexity Tracking

Sem violações.
