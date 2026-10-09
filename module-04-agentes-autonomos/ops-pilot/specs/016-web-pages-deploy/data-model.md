# Data Model: Publicação da War Room no GitHub Pages

A feature não persiste dados. As "entidades" da spec viram estes elementos do GitHub, e só há uma entrada de configuração nova.

## Publicação → execução do workflow `ops-pilot-web-pages`

| Atributo (spec) | Onde fica |
|---|---|
| Gatilho | `github.event_name`: `push`, `pull_request` ou `workflow_dispatch` |
| Versão de origem | `github.sha` |
| Resultado e etapa que falhou | status de cada job e passo na aba Actions |
| Endereço público | `steps.deployment.outputs.page_url` (mostrado no ambiente `github-pages`) |

**Estados**:

```text
disparada ─▶ check (typecheck → test → build → artifact)
               ├─ falhou ─▶ FIM (nada publicado; versão anterior no ar)
               └─ ok ─┬─ pull_request ─▶ FIM (sem publicar)
                      └─ master ─▶ deploy (fila "pages-ops-pilot")
                                     ├─ falhou ─▶ FIM (versão anterior no ar)
                                     └─ ok ─▶ site publicado = esta versão
```

## Site publicado → artifact `github-pages`

Só uma versão fica no ar, substituída a cada `deploy` bem-sucedido. A estrutura está no [research.md](./research.md#7-estrutura-do-artifact): `index.html` (redirecionamento) mais `opspilot/` (a war room).

## Configuração nova

| Nome | Onde | Valor | Validação |
|---|---|---|---|
| `OPSPILOT_WEB_BASE` | env do build do `web/` | padrão `/opspilot/`; no workflow, `/<repo>/opspilot/` | `resolveBasePath`: precisa ser um caminho absoluto, com `/` no início e no fim, sem esquema e sem `..`. Valor inválido faz o build falhar com mensagem clara |
