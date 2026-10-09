# Quickstart: Publicação da War Room no GitHub Pages

Este é o roteiro de validação. O formato do workflow está em [contracts/workflow.md](./contracts/workflow.md).

## 1. Local, antes de subir

```bash
cd module-04-agentes-autonomos/ops-pilot/web
npm test                                                  # resolveBasePath + build local (/opspilot/)
OPSPILOT_WEB_BASE=/unipds-ia/opspilot/ npm run build
OPSPILOT_WEB_BASE=/unipds-ia/opspilot/ npm test           # dist/ validado com o base do Pages
OPSPILOT_WEB_BASE=x npm run build                         # deve falhar com mensagem clara
```

O YAML também pode ser validado localmente com `actionlint`, se você o tiver instalado: `actionlint .github/workflows/ops-pilot-web-pages.yml` deve passar sem erros.

## 2. Ativação (uma vez, por quem administra o repositório)

Settings → Pages → Build and deployment → Source: **GitHub Actions**.

## 3. Publicação automática (US1)

1. Faça um push em `master` com uma mudança visível em `web/`, por exemplo o texto do estado vazio.
2. Em Actions, confira: o `check` passa por todos os passos, e o `deploy` mostra a URL do ambiente.
3. Abra `https://gabriel-barbosa-de-oliveira.github.io/unipds-ia/`. A página deve redirecionar para `/opspilot/` e mostrar a mudança.
4. Recarregue a página. No DevTools, não pode haver nenhum 404 de asset (SC-006).
5. Faça um push que toca só em `src/` (API). **Nenhuma** execução deve ser disparada (SC-003).

## 4. Nada quebrado vai para o ar (US2)

1. Abra um PR que quebra um teste de `web/src/lib/`. O `check` deve falhar em "npm test", e não pode haver job `deploy`.
2. Corrija o teste. O `check` fica verde no PR, ainda sem `deploy`.

## 5. Publicação manual

Actions → **OpsPilot web → Pages** → Run workflow → `master`. Os dois jobs devem passar e o site deve ser republicado.

## 6. Permissões (US3)

Revise o YAML. Só podem aparecer `contents: read` no topo e `pages: write` + `id-token: write` no `deploy`. Também não pode haver nenhum `secrets.` nem `write-all`:

```bash
grep -nE "permissions|contents:|pages:|id-token:|secrets\.|write-all" .github/workflows/ops-pilot-web-pages.yml
```

## 7. README (US4)

Siga só o `ops-pilot/README.md`:
1. Suba a API local com `OPSPILOT_CORS_ORIGINS=https://gabriel-barbosa-de-oliveira.github.io,http://localhost:5173`.
2. Abra a war room publicada e confira na engrenagem o endereço `http://localhost:3000` ("Conectado").
3. Envie "Quais alertas estão disparando?" e confira que a resposta aparece (SC-005).
