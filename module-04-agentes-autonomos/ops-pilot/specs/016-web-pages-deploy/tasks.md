---

description: "Task list for 016-web-pages-deploy"
---

# Tasks: Publicação da War Room no GitHub Pages

> **Nota de implementação (`/speckit-implement`)**:
> - **O workflow foi escrito numa passada só (T005–T011)**, já com gates, gatilho de PR, concorrência, `persist-credentials: false` e as actions fixadas por SHA, para nunca existir uma versão sem gates ou com tags mutáveis. Os SHAs foram reconferidos com `git ls-remote` antes de gravar.
> - **`OPSPILOT_WEB_BASE` é `env` do job `check` inteiro**, não só do passo de build. Assim, o `npm test` pós-build valida o `dist/` com o mesmo caminho, sem repetir o valor.
> - **T012:** o `grep` mostra só `contents: read`, `pages: write` e `id-token: write`, sem `secrets.` nem `write-all`. O YAML é válido (Ruby `YAML.load_file`), e o `actionlint` 1.7.12 passou sem erros (binário oficial baixado para o scratchpad da sessão, fora do repositório).
> - **`resolveBasePath` também rejeita `\`**, além do que estava na T003, para não aceitar separador de Windows num caminho de URL.

**Input**: Design documents from `/specs/016-web-pages-deploy/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Incluídos só para a lógica nova (`resolveBasePath` e a validação do `dist/`), conforme o Princípio V. O workflow e os READMEs são validados por inspeção e pelo [quickstart.md](./quickstart.md).

**Organization**: as tarefas estão agrupadas por user story (spec.md).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivo diferente e sem dependência de tarefa ainda não concluída).
- **[Story]**: a user story à qual a tarefa pertence (US1–US4).

## Path Conventions

- **`<repo>/`** é a raiz do repositório git `unipds-ia`. **`ops-pilot/`** é `<repo>/module-04-agentes-autonomos/ops-pilot/`.
- O workflow fica em `<repo>/.github/workflows/ops-pilot-web-pages.yml`, o único caminho que o GitHub lê.

---

## Phase 1: Setup

**Purpose**: preparar o `web/` para o build do Pages.

- [X] T001 Criar `ops-pilot/web/.gitignore` com `_site/`, a pasta do artifact montada no CI (research.md item 7). Conferir com `git check-ignore -v module-04-agentes-autonomos/ops-pilot/web/_site/x` a partir de `<repo>/`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: o caminho base configurável, sem o qual o build publicado sob `/unipds-ia/opspilot/` quebraria todos os assets.

**⚠️ CRITICAL**: o workflow (US1) depende desta fase.

- [X] T002 Em `ops-pilot/web/src/lib/base-path.test.ts`, acrescentar o `describe("resolveBasePath")` (deve falhar até a T003):
  - `undefined` e `""` → `"/opspilot/"`;
  - `"/unipds-ia/opspilot/"` → igual;
  - `"unipds-ia/opspilot"` → `"/unipds-ia/opspilot/"`, porque normaliza as barras;
  - `"//x//"` → `"/x/"`;
  - `"http://x/opspilot/"`, `"/a/../b/"`, `"/com espaço/"` e `"/x?q/"` → lançam `Error` com mensagem citando `OPSPILOT_WEB_BASE`.

  No mesmo arquivo, trocar o teste "vite.config.ts usa o mesmo caminho base" (hoje procura a string literal) por um que confira que o `vite.config.ts` importa e chama `resolveBasePath(process.env.OPSPILOT_WEB_BASE)`. Trocar também o teste do `dist/`, que passa a esperar o prefixo `resolveBasePath(process.env.OPSPILOT_WEB_BASE)`, para funcionar com o base local e com o do Pages.
- [X] T003 Em `ops-pilot/web/src/lib/base-path.ts`, criar `export function resolveBasePath(raw?: string): string` (pura), conforme o [data-model.md](./data-model.md#configuração-nova):
  - vazio → `BASE_PATH`;
  - senão: `trim`, rejeitar esquema (`/^[a-z]+:/i`), `..`, espaços, `?` e `#`;
  - colapsar barras repetidas e garantir `/` no início e no fim.

  O erro deve dizer, em PT, o valor recebido e o formato esperado (`/repo/opspilot/`). Atualizar o comentário de `BASE_PATH`: "padrão local; a publicação usa OPSPILOT_WEB_BASE".
- [X] T004 Em `ops-pilot/web/vite.config.ts`, trocar `base: "/opspilot/"` por `base: resolveBasePath(process.env.OPSPILOT_WEB_BASE)`, importando de `./src/lib/base-path.ts`, e atualizar o comentário (spec 016, research.md item 6). Rodar `npm --prefix web run typecheck && npm --prefix web test`, depois `OPSPILOT_WEB_BASE=/unipds-ia/opspilot/ npm --prefix web run build` seguido de `OPSPILOT_WEB_BASE=/unipds-ia/opspilot/ npm --prefix web test`, e `npm --prefix web run build` voltando a `/opspilot/`. Tudo deve ficar verde.

**Checkpoint**: o dev local continua em `/opspilot/`, e o build com `OPSPILOT_WEB_BASE` gera assets sob `/unipds-ia/opspilot/`.

---

## Phase 3: User Story 1 - War room publicada automaticamente (Priority: P1) 🎯 MVP

**Goal**: push em `master` que toca `web/` → war room no ar em `…/unipds-ia/opspilot/`. Também há disparo manual.

**Independent Test**: com o Pages ativado, um push com uma mudança visível em `web/` aparece no endereço público, inclusive ao recarregar. Um push só na API não dispara nada.

- [X] T005 [US1] Criar `<repo>/.github/workflows/ops-pilot-web-pages.yml` conforme o [contracts/workflow.md](./contracts/workflow.md):
  - `name: "OpsPilot web → Pages"`;
  - `on.push` com `branches: [master]` e `paths` (`module-04-agentes-autonomos/ops-pilot/web/**` e o próprio arquivo), mais `on.workflow_dispatch`;
  - `permissions: { contents: read }` no topo.

  Job `check`:
  - `runs-on: ubuntu-latest`, `timeout-minutes: 10` e `defaults.run.working-directory: module-04-agentes-autonomos/ops-pilot/web`;
  - passos: checkout, setup-node (`node-version: 24`, `cache: npm`, `cache-dependency-path` do lockfile do web) e `npm ci`;
  - `npm run build` com `env.OPSPILOT_WEB_BASE: /${{ github.event.repository.name }}/opspilot/`;
  - montar `_site/`: `mkdir -p _site/opspilot && cp -R dist/. _site/opspilot/`, mais o `index.html` de redirecionamento da T006;
  - `actions/upload-pages-artifact` com `path: module-04-agentes-autonomos/ops-pilot/web/_site` (o input `path` não segue o `working-directory`).

  Job `deploy`:
  - `needs: check` e `if: github.ref == 'refs/heads/master'`;
  - `permissions: { pages: write, id-token: write }`;
  - `environment: { name: github-pages, url: ${{ steps.deployment.outputs.page_url }} }`;
  - `actions/deploy-pages` com `id: deployment`.

  Nesta tarefa as actions ainda podem usar tags (`@v7`/`@v5`); a fixação por SHA vem na T011.
- [X] T006 [US1] No passo de montagem do `_site/` em `<repo>/.github/workflows/ops-pilot-web-pages.yml`, gerar `_site/index.html`. É um HTML mínimo em `pt-BR` com `<meta http-equiv="refresh" content="0; url=./opspilot/">`, `<title>OpsPilot · War room</title>` e um link visível "Abrir a war room" para `./opspilot/` (research.md item 7). Gerar via `cat <<'EOF'`, sem arquivo extra no repositório.
- [X] T007 [US1] Em `<repo>/.github/workflows/ops-pilot-web-pages.yml`, acrescentar ao `deploy` a `concurrency: { group: pages-ops-pilot, cancel-in-progress: false }`, para que uma publicação em andamento termine e a mais recente fique no ar (FR-005, research.md item 4).

**Checkpoint**: o YAML descreve a publicação completa. Validação real nas seções 2, 3 e 5 do [quickstart.md](./quickstart.md), depois do push.

---

## Phase 4: User Story 2 - Nada quebrado vai para o ar (Priority: P1)

**Goal**: typecheck, testes e build antes de publicar; PR roda a checagem sem publicar.

**Independent Test**: um PR com teste quebrado falha em "npm test", sem job `deploy`. Corrigido, fica verde e continua sem `deploy`.

- [X] T008 [US2] Em `<repo>/.github/workflows/ops-pilot-web-pages.yml`, no job `check`:
  - entre `npm ci` e o build, adicionar os passos `npm run typecheck` e `npm test` (FR-007);
  - depois do build com o base do Pages, rodar `npm test` de novo com o mesmo `env.OPSPILOT_WEB_BASE`, para o `base-path.test.ts` validar o `dist/` publicado (SC-006).
- [X] T009 [US2] Em `<repo>/.github/workflows/ops-pilot-web-pages.yml`:
  - adicionar `on.pull_request` com os mesmos `paths` (FR-008);
  - condicionar a montagem do `_site/` e o `upload-pages-artifact` a `if: github.event_name != 'pull_request'`;
  - mudar o `if` do `deploy` para `github.event_name != 'pull_request' && github.ref == 'refs/heads/master'`;
  - adicionar ao `check` a `concurrency: { group: ops-pilot-web-check-${{ github.ref }}, cancel-in-progress: true }`.

**Checkpoint**: seção 4 do [quickstart.md](./quickstart.md).

---

## Phase 5: User Story 3 - Publicação com o mínimo de acesso (Priority: P2)

**Goal**: 3 permissões no total, nenhuma escrita no repositório, sem segredos e actions imutáveis.

**Independent Test**: o `grep` da seção 6 do [quickstart.md](./quickstart.md) mostra só `contents: read`, `pages: write` e `id-token: write`, sem `secrets.` nem `write-all`.

- [X] T010 [US3] Em `<repo>/.github/workflows/ops-pilot-web-pages.yml`, adicionar `persist-credentials: false` ao checkout e `timeout-minutes: 10` ao `deploy`, mais um comentário curto acima de cada `permissions` explicando o porquê (Princípio VI).
- [X] T011 [US3] Em `<repo>/.github/workflows/ops-pilot-web-pages.yml`, fixar as 4 actions pelo SHA completo, com a tag em comentário, conforme a tabela do research.md item 5:
  - `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1`;
  - `actions/setup-node@949feb2413d6458794dcd2491c4babbbce0c15c1 # v7.1.0`;
  - `actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9 # v5.0.0`;
  - `actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346 # v5.0.1`.

  Antes de gravar, reconferir cada SHA com `git ls-remote https://github.com/<action> refs/tags/<tag>`.
- [X] T012 [US3] Verificar o arquivo final com o `grep` da seção 6 do [quickstart.md](./quickstart.md), validar a sintaxe do YAML (`ruby -ryaml -e 'YAML.load_file(ARGV[0])' <arquivo>` ou `python3 -c 'import yaml…'`) e rodar `actionlint`, se estiver disponível. Registrar o resultado na nota de implementação do `tasks.md`.

**Checkpoint**: SC-004 conferido por inspeção.

---

## Phase 6: User Story 4 - Saber como acessar, publicar e conectar à API (Priority: P2)

**Goal**: README do OpsPilot completo e link a partir da raiz.

**Independent Test**: uma pessoa nova segue só o README e troca uma mensagem com a war room publicada usando a API local (seção 7 do quickstart).

- [X] T013 [P] [US4] Criar `ops-pilot/README.md` com as 7 seções do [contracts/readme.md](./contracts/readme.md), em PT e com os valores reais:
  - URL `https://gabriel-barbosa-de-oliveira.github.io/unipds-ia/opspilot/`;
  - origem `https://gabriel-barbosa-de-oliveira.github.io`;
  - nome do workflow "OpsPilot web → Pages";
  - padrões de `OPSPILOT_DB` (`./data/opspilot.db`), `OPSPILOT_CORS_ORIGINS` (`http://localhost:5173`) e `OPSPILOT_APPROVAL_TTL_MS` (`900000`);
  - incluir `npm run seed` como passo obrigatório antes do primeiro uso, já que um banco sem seed faz "o serviço não existe" em toda ação;
  - sem segredos nem conteúdo de `.env`.
- [X] T014 [P] [US4] Atualizar `<repo>/README.md`: manter as duas linhas existentes e acrescentar a seção "Projetos em destaque", com o OpsPilot (link relativo `module-04-agentes-autonomos/ops-pilot/` e link da war room publicada).
- [X] T015 [P] [US4] Em `ops-pilot/CLAUDE.md`, na seção de comandos/env, acrescentar `OPSPILOT_WEB_BASE` (caminho base do build do web; padrão `/opspilot/`, e o Pages usa `/<repo>/opspilot/`) e uma linha dizendo que o push em `master` que toca `web/` publica no Pages via `.github/workflows/ops-pilot-web-pages.yml` (na raiz do repositório).

**Checkpoint**: seção 7 do [quickstart.md](./quickstart.md).

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T016 **Parcial**: os 4 gates passaram (API 344/344, web 69/69). O build com `OPSPILOT_WEB_BASE=/unipds-ia/opspilot/` mais o `npm test` passaram na T004. A montagem do `_site/` não foi simulada localmente, porque o `rm -rf _site` foi negado na sessão; o passo está coberto pelo `actionlint` e é validado na primeira execução real. Rodar os gates: `npm run typecheck`, `npm test`, `npm --prefix web run typecheck`, `npm --prefix web test` e `npm --prefix web run build`. Também simular o CI localmente: `OPSPILOT_WEB_BASE=/unipds-ia/opspilot/ npm --prefix web run build && OPSPILOT_WEB_BASE=/unipds-ia/opspilot/ npm --prefix web test`, mais os comandos de montagem do `_site/` copiados do workflow, conferindo `_site/index.html` e `_site/opspilot/index.html`. Apagar `_site/` e restaurar o `dist/` local no fim.
- [ ] T017 **Pendente** (exige push, ativação do Pages e PR de teste). Validar no GitHub as seções 2–7 do [quickstart.md](./quickstart.md). Isso exige push em `master`, a ativação do Pages pela pessoa que administra o repositório e um PR de teste. **Não fazer push nem abrir PR sem pedido explícito.**

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (T001)**: independente.
- **Foundational**: T002 → T003 → T004. Bloqueia a US1, porque o workflow usa `OPSPILOT_WEB_BASE`.
- **US1**: T005 → T006 → T007, todas no mesmo arquivo e em sequência.
- **US2**: depois da US1 (mesmo arquivo). T008 → T009.
- **US3**: depois da US2 (mesmo arquivo). T010 → T011 → T012.
- **US4**: T013 ∥ T014 ∥ T015, independentes do workflow. Podem começar logo depois do Setup.
- **Polish**: depois de todas.

### User Story Dependencies

- **US1 (P1)**: Foundational.
- **US2 (P1)**: US1 (estende o mesmo workflow).
- **US3 (P2)**: US1 + US2 (endurece o arquivo final).
- **US4 (P2)**: nenhuma. Só documenta o que as outras entregam.

## Parallel Opportunities

- A US4 inteira (T013 ∥ T014 ∥ T015) roda em paralelo com Foundational, US1, US2 e US3.
- T001 ∥ T002.

### Parallel Example: User Story 4

```bash
Task: "T013 [US4] README do OpsPilot em module-04-agentes-autonomos/ops-pilot/README.md"
Task: "T014 [US4] link no README.md da raiz do repositório"
Task: "T015 [US4] OPSPILOT_WEB_BASE e publicação em module-04-agentes-autonomos/ops-pilot/CLAUDE.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Setup + Foundational (T001–T004): base configurável com testes.
2. US1 (T005–T007): workflow que publica.
3. **STOP and VALIDATE** (exige push): seções 2, 3 e 5 do quickstart.

Na prática, US1, US2 e US3 mexem no mesmo arquivo e devem ir juntas no primeiro push. Publicar sem os gates (US2) ou sem a fixação por SHA (US3) vai contra os Princípios V e VI. O MVP entregável é então T001–T012, com a US4 em paralelo.

### Incremental Delivery

Base configurável → workflow (publicação → gates → endurecimento) → READMEs → validação local completa. Cada passo deixa os gates verdes e cabe num commit.
