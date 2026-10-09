# Contrato: READMEs

## `module-04-agentes-autonomos/ops-pilot/README.md` (novo)

Escrito em português, para quem chega ao projeto. Seções, nesta ordem:

1. **OpsPilot**: uma frase sobre o que é e um link para a war room publicada.
2. **Rodar localmente**: pré-requisitos (Node 24), `npm install` na raiz e em `web/`, `npm run seed` (o banco local precisa do dataset), `npm run dev` (API em `:3000`) e `npm --prefix web run dev` (war room em `http://localhost:5173/opspilot/`). Uma nota dizendo que as variáveis do OpenRouter vêm do ambiente e que o `.env` nunca é commitado.
3. **Variáveis da API**: tabela com `OPSPILOT_DB`, `OPSPILOT_CORS_ORIGINS` e `OPSPILOT_APPROVAL_TTL_MS` (com os padrões).
4. **Testes e qualidade**: os 4 gates (`typecheck` e `test` da API e do web).
5. **War room publicada (GitHub Pages)**:
   - endereço: `https://gabriel-barbosa-de-oliveira.github.io/unipds-ia/opspilot/`;
   - como funciona: o push em `master` que toca `web/` roda checagem e publicação, e o PR roda só a checagem; link para o workflow;
   - **ativação única**: Settings → Pages → Source: GitHub Actions, com a mensagem de erro típica quando falta;
   - **publicar manualmente**: Actions → "OpsPilot web → Pages" → Run workflow (ramo `master`);
   - **conferir uma publicação**: o ambiente `github-pages` mostra a URL e o commit.
6. **Conectar a war room publicada à API**: `OPSPILOT_CORS_ORIGINS=https://gabriel-barbosa-de-oliveira.github.io,http://localhost:5173`, a engrenagem para trocar o endereço e uma nota sobre HTTPS → `http://localhost` por navegador.
7. **Fluxo de desenvolvimento**: Spec Kit (`specs/`), com links para a constitution e o `CLAUDE.md`.

## `README.md` da raiz (atualizar)

Mantém as duas linhas existentes e acrescenta a seção **Projetos em destaque**, com uma linha do OpsPilot: link para `module-04-agentes-autonomos/ops-pilot/` e para a war room publicada.
