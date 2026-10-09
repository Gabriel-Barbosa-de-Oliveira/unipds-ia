# Research: Publicação da War Room no GitHub Pages

Decisões da Fase 0 do `/speckit.plan`. Cada item segue o formato Decisão / Racional / Alternativas.

## 1. Onde fica o workflow num monorepo

**Decisão**: `.github/workflows/ops-pilot-web-pages.yml` na **raiz do repositório** (`unipds-ia/`), com `defaults.run.working-directory: module-04-agentes-autonomos/ops-pilot/web` e filtro `paths` limitado a esse diretório e ao próprio arquivo do workflow.

**Racional**: o GitHub só lê workflows em `<raiz>/.github/workflows/`. O `ops-pilot/.github/` existente guarda só as instruções de design do Copilot e não é lido pelo Actions. O filtro por `paths` cumpre o FR-001 e o FR-002 sem lógica extra.

**Alternativas**: `paths-ignore` (inverteria a regra, porque qualquer pasta nova do monorepo passaria a disparar a publicação); um job de detecção de mudança (complexidade desnecessária quando `paths` resolve).

## 2. Gatilhos

**Decisão**:
- `push` em `master`, com `paths: [module-04-agentes-autonomos/ops-pilot/web/**, .github/workflows/ops-pilot-web-pages.yml]`: checagem e publicação.
- `pull_request` com os mesmos `paths`: só a checagem (FR-008).
- `workflow_dispatch`: publicação manual (FR-003). O job de publicação só roda quando `github.ref == 'refs/heads/master'`, mesmo que alguém dispare a partir de outro ramo.

**Racional**: o ambiente `github-pages` já vem protegido para aceitar só deploys do ramo padrão. A condição no job explicita isso e evita uma execução que falharia no fim.

## 3. Jobs e permissões

**Decisão**: dois jobs.

| Job | Quando | Permissões | Passos |
|---|---|---|---|
| `check` | sempre | `contents: read` (herdado do topo) | checkout → setup-node 24 com cache npm → `npm ci` → `typecheck` → `test` → `build` (com o base do Pages) → monta `_site/` → `upload-pages-artifact` (só fora de PR) |
| `deploy` | `needs: check`, e só em `push`/`workflow_dispatch` no `master` | `pages: write`, `id-token: write` | `deploy-pages` com `environment: github-pages` e `url: ${{ steps.deployment.outputs.page_url }}` |

No topo: `permissions: contents: read`. O job `deploy` sobrescreve com `pages: write` e `id-token: write`, que são o mínimo exigido pelo `deploy-pages` (README oficial da v5.0.1, seção "Security considerations"). O total é 3 permissões, e nenhuma escreve no repositório (SC-004).

**Racional**:
- Separar a checagem (que roda código do PR e instala dependências) da publicação (que tem permissões de escrita) limita o estrago de uma dependência comprometida: o código instalado nunca roda com `pages: write`.
- PRs de fork recebem token só de leitura e não têm o job `deploy` (FR-011, US3 cenário 2).
- O `upload-pages-artifact` não precisa de permissão extra, porque usa o artifact da própria execução.

**Alternativas**: um único job com tudo (o código do build rodaria com `pages: write`); `permissions: write-all` (viola o Princípio VI).

## 4. Concorrência

**Decisão**: `concurrency: { group: pages-ops-pilot, cancel-in-progress: false }` só no job `deploy`. A checagem de PR fica livre, com `concurrency: { group: ops-pilot-web-${{ github.ref }}, cancel-in-progress: true }` no job `check`.

**Racional**: o GitHub mantém no máximo uma execução pendente por grupo, e uma pendente mais nova substitui a mais antiga. Uma publicação em andamento termina, sem ser cortada no meio, e a mais recente fica no ar (FR-005). Essa é a configuração recomendada nos starter workflows de Pages. Nas checagens, cancelar a execução antiga do mesmo ramo economiza minutos sem risco.

## 5. Versões das actions e fixação por SHA

**Decisão**: actions oficiais, fixadas pelo **SHA completo do commit**, com a tag num comentário. Os valores foram resolvidos com `git ls-remote` em 2026-10-09:

| Action | Tag | SHA |
|---|---|---|
| `actions/checkout` | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| `actions/setup-node` | v7.1.0 | `949feb2413d6458794dcd2491c4babbbce0c15c1` |
| `actions/upload-pages-artifact` | v5.0.0 | `fc324d3547104276b827a68afc52ff2a11cc49c9` |
| `actions/deploy-pages` | v5.0.1 | `368f82528645a54fb793d4d04e342629a3f51346` |

O `checkout` usa `persist-credentials: false`, porque nenhum passo seguinte usa git autenticado.

**Racional**: uma tag pode ser movida, e o SHA não (Princípio VI, cadeia de suprimentos). O comentário com a tag mantém a leitura humana e permite atualização pelo Dependabot.

**Alternativas**: `@v5` (mais simples, mas mutável); um `actions/configure-pages` extra (desnecessário: só serve para habilitar o Pages, o que exige permissão de administrador, ou para descobrir o base path, que já vem de `github.event.repository.name`).

## 6. Caminho base configurável

**Decisão**:
- `web/src/lib/base-path.ts` ganha `resolveBasePath(raw?: string): string` (pura). Sem valor, devolve `/opspilot/`. Com valor, normaliza para começar e terminar com `/` e rejeita o que não for um caminho (`http://…`, `..`, espaços).
- `vite.config.ts` usa `base: resolveBasePath(process.env.OPSPILOT_WEB_BASE)`.
- No workflow: `OPSPILOT_WEB_BASE: /${{ github.event.repository.name }}/opspilot/`. Hoje isso é `/unipds-ia/opspilot/`, e acompanha uma renomeação do repositório automaticamente (edge case da spec).
- O `base-path.test.ts` deixa de procurar a string literal no `vite.config.ts`. Ele passa a testar `resolveBasePath` e a validar o `dist/` contra `resolveBasePath(process.env.OPSPILOT_WEB_BASE)`, então funciona no CI e no local.

**Racional**: o desenvolvimento local (`http://localhost:5173/opspilot/`) não muda, e a publicação ganha o prefixo do site de projeto.

**Alternativas**: base relativo (`./`) quebraria assets em rotas aninhadas no futuro e exigiria configuração extra no dev; trocar o padrão para `/unipds-ia/opspilot/` amarraria o dev local ao nome do repositório.

## 7. Estrutura do artifact

**Decisão**: o artifact é `_site/` (em `web/`), montado assim:

```text
_site/
├── index.html          # redireciona para ./opspilot/ (meta refresh + link visível)
└── opspilot/           # conteúdo de web/dist
```

O site do repositório responde em `/unipds-ia/`, então a war room fica em `/unipds-ia/opspilot/`, e quem abrir `/unipds-ia/` é levado para ela. `_site/` entra no `.gitignore`.

**Racional**: o artifact do Pages é a raiz do site do repositório, e o subdiretório reproduz o segmento `/opspilot/` da spec 015. A página de redirecionamento evita um 404 na raiz e tem link visível, por acessibilidade e para o caso de o refresh estar bloqueado.

## 8. Node e instalação reprodutível

**Decisão**: `node-version: 24` (constitution), `cache: npm` com `cache-dependency-path: module-04-agentes-autonomos/ops-pilot/web/package-lock.json` e `npm ci`. O lockfile do `web/` já existe e não está ignorado (conferido com `git check-ignore`).

**Racional**: `npm ci` falha se o lockfile divergir do `package.json`, o que cumpre o FR-009.

## 9. Ativação do Pages

**Decisão**: não automatizar. O README documenta o passo único: **Settings → Pages → Build and deployment → Source: GitHub Actions**. Sem isso, o `deploy-pages` falha com mensagem de Pages não habilitado, e o README cita essa mensagem.

**Racional**: habilitar o Pages por API exige token com permissão de administrador, o que contraria o FR-010 e o FR-012.

## 10. Conectar a war room publicada a uma API

**Decisão**: o README documenta:
- `OPSPILOT_CORS_ORIGINS=https://gabriel-barbosa-de-oliveira.github.io` na API (é a origem, sem caminho; pode ser combinada com `http://localhost:5173` por vírgula);
- o endereço da API se configura na engrenagem;
- de página HTTPS para `http://localhost:3000`: o Chrome e o Firefox tratam `localhost` como contexto confiável. O Chrome pode pedir permissão de acesso à rede local na primeira vez. O Safari pode bloquear conteúdo misto, e nesse caso o caminho é usar a war room local ou uma API com HTTPS.

**Racional**: nenhuma mudança de código é necessária, porque a mensagem de erro de rede da 015 já cita a origem e o `OPSPILOT_CORS_ORIGINS`.

## 11. Constitution

**Decisão**: sem amendment. O GitHub Actions não entra na stack de runtime. O workflow executa os gates que a constitution já exige para o `web/` (Princípio V) e não usa segredos (Princípio VI).
