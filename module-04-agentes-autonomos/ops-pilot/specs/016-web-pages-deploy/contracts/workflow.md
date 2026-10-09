# Contrato: workflow `ops-pilot-web-pages`

**Arquivo**: `.github/workflows/ops-pilot-web-pages.yml`, na raiz do repositório `unipds-ia`.

Este contrato é verificado por inspeção, ou seja, lendo o YAML, e pelo [quickstart.md](../quickstart.md).

## Gatilhos

| Evento | Filtro | Jobs que rodam |
|---|---|---|
| `push` | `branches: [master]` e `paths: [module-04-agentes-autonomos/ops-pilot/web/**, .github/workflows/ops-pilot-web-pages.yml]` | `check` → `deploy` |
| `pull_request` | mesmos `paths` | `check` (sem upload nem deploy) |
| `workflow_dispatch` | — | `check` → `deploy` (o `deploy` só se `github.ref == refs/heads/master`) |

## Permissões

```yaml
permissions:
  contents: read          # topo: vale para todos os jobs

jobs:
  deploy:
    permissions:
      pages: write        # criar o deploy do Pages
      id-token: write     # OIDC exigido pelo deploy-pages
```

Não há outras permissões, `secrets.*` nem `write-all`. O total é 3 permissões distintas (SC-004).

## Job `check`

- `runs-on: ubuntu-latest`, `timeout-minutes: 10`.
- `defaults.run.working-directory: module-04-agentes-autonomos/ops-pilot/web`.
- `concurrency: { group: ops-pilot-web-check-${{ github.ref }}, cancel-in-progress: true }`.
- Passos, nesta ordem:
  1. `actions/checkout@<sha> # v7.0.1`, com `persist-credentials: false`.
  2. `actions/setup-node@<sha> # v7.1.0`, com `node-version: 24`, `cache: npm` e `cache-dependency-path: module-04-agentes-autonomos/ops-pilot/web/package-lock.json`.
  3. `npm ci`.
  4. `npm run typecheck`.
  5. `npm test`.
  6. `npm run build`, com `env: OPSPILOT_WEB_BASE: /${{ github.event.repository.name }}/opspilot/`.
  7. `npm test` de novo, para o teste do caminho base validar o `dist/` gerado com o base do Pages.
  8. Montar `_site/`: `opspilot/` recebe o conteúdo de `dist/`, e `index.html` redireciona.
  9. `actions/upload-pages-artifact@<sha> # v5.0.0`, com `path: module-04-agentes-autonomos/ops-pilot/web/_site`. **Só** quando `github.event_name != 'pull_request'`.

## Job `deploy`

- `needs: check`.
- `if: github.event_name != 'pull_request' && github.ref == 'refs/heads/master'`.
- `runs-on: ubuntu-latest`, `timeout-minutes: 10`.
- `concurrency: { group: pages-ops-pilot, cancel-in-progress: false }`.
- `environment: { name: github-pages, url: ${{ steps.deployment.outputs.page_url }} }`.
- Passo único: `actions/deploy-pages@<sha> # v5.0.1`, com `id: deployment`.

## Saídas observáveis

- **Sucesso**: o ambiente `github-pages` mostra a URL `https://gabriel-barbosa-de-oliveira.github.io/unipds-ia/`, que redireciona para `…/opspilot/` (FR-006).
- **Falha**: a execução fica vermelha no passo que falhou, e o site continua com a versão anterior (FR-007).
