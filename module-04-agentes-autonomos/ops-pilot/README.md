# OpsPilot

OpsPilot é um copiloto de plantão que gerencia alertas e incidentes de produção. A API é um agente LangChain/LangGraph que roda sobre o OpenRouter. A **war room** é a interface web: um chat com o copiloto, o raciocínio de cada resposta e a aprovação humana de ações.

**War room publicada:** <https://gabriel-barbosa-de-oliveira.github.io/unipds-ia/opspilot/>

## Rodar localmente

Pré-requisito: Node 24 LTS.

```bash
npm install                 # API
npm --prefix web install    # war room
npm run seed                # obrigatório antes do primeiro uso (veja abaixo)
npm run dev                 # API em http://localhost:3000
npm --prefix web run dev    # war room em http://localhost:5173/opspilot/
```

- **`npm run seed`** cria o dataset canônico no SQLite: 5 serviços, 6 alertas e 3 runbooks. Sem ele o banco fica vazio, e toda ação falha com "o serviço não existe". O seed só reescreve serviços, alertas, incidentes e runbooks. Conversas e o histórico de requisições são mantidos.
- O modelo é configurado por variáveis de ambiente (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL` e, opcionalmente, `OPENROUTER_MODEL_FALLBACK` e `OPENROUTER_BASE_URL`). O `npm run dev` carrega um `.env` local, se existir. **Nunca commite o `.env` nem segredos.**

## Rotas de raciocínio

O roteador escolhe uma rota por pedido. O cliente também pode forçar a rota com `"strategy"` no `POST /chat`.

| Rota | Quando |
|---|---|
| `react` | Consulta direta ou ação única |
| `planExecute` | Várias etapas dependentes ou em lote |
| `reflect` | Precisão crítica (a resposta é revisada antes de sair) |
| `team` (ou `equipe`) | Investigar e agir de forma coordenada |

**Modo equipe (`team`).** Um supervisor decide, a cada passo, quem trabalha em seguida e com qual instrução. Ele lê um quadro compartilhado e encerra com a resposta final. Os papéis têm limites fixos no código:

- **analista**: só consulta alertas, incidentes e runbooks, e registra no quadro só fatos com origem, sem propostas;
- **planejador**: não tem ferramentas e escreve o plano a partir dos fatos;
- **executor**: só abre e resolve incidentes, sempre pela aprovação humana (resposta 202). A equipe nem monta se essas ferramentas não tiverem a aprovação.

Cada passagem aparece como "Passagem" no "ver raciocínio" da war room. A equipe tem no máximo 6 passagens por pedido. O código está em [`src/team/`](src/team/) e a spec em [`specs/017-team-mode/`](specs/017-team-mode/).

## Variáveis da API

| Variável | Para quê | Padrão |
|---|---|---|
| `PORT` | Porta HTTP | `3000` |
| `OPSPILOT_DB` | Arquivo SQLite | `./data/opspilot.db` |
| `OPSPILOT_CORS_ORIGINS` | Origens do navegador autorizadas, separadas por vírgula (comparação exata) | `http://localhost:5173` |
| `OPSPILOT_APPROVAL_TTL_MS` | Validade de uma ação aguardando aprovação | `900000` (15 min) |
| `OPSPILOT_MODEL_PRICES` | Preços por modelo, usados no `GET /stats` | — |

### Aprovação humana

Para abrir ou resolver um incidente, a API não executa nada sozinha. O `POST /chat` responde **202** com a ação pendente, e a ação só roda depois de `POST /approvals/:id` com `{"decision":"approve"}`. Com `"deny"`, ela é cancelada. Na war room, isso aparece como um cartão Aprovar/Negar. O contrato completo está em [`specs/015-war-room-web/contracts/http.md`](specs/015-war-room-web/contracts/http.md).

## Testes e qualidade

```bash
npm run typecheck && npm test                              # API
npm --prefix web run typecheck && npm --prefix web test    # war room
```

Os quatro precisam estar verdes antes de qualquer commit (veja a [constitution](.specify/memory/constitution.md)).

## War room publicada (GitHub Pages)

A war room publicada fica em <https://gabriel-barbosa-de-oliveira.github.io/unipds-ia/opspilot/>. A raiz `…/unipds-ia/` redireciona para ela.

**Como funciona.** O workflow [`OpsPilot web → Pages`](../../.github/workflows/ops-pilot-web-pages.yml) fica na raiz do repositório e roda assim:

| Evento | O que acontece |
|---|---|
| Push em `master` que toca `ops-pilot/web/` (ou o próprio workflow) | Typecheck, testes, build e **publicação** |
| Pull request que toca os mesmos caminhos | Typecheck, testes e build, **sem publicar** |
| Disparo manual | Igual ao push, a partir de `master` |
| Mudança só na API ou em outros projetos | Nada |

Se qualquer etapa falhar, a publicação não acontece e a versão anterior continua no ar. O workflow só tem as permissões necessárias (`contents: read`, mais `pages: write` e `id-token: write` no job de publicação), não usa segredos e fixa as actions oficiais pelo SHA do commit.

**Ativação (uma vez).** Quem administra o repositório vai em **Settings → Pages → Build and deployment → Source** e escolhe **GitHub Actions**. Sem isso, o job "Publicar no Pages" falha com um erro dizendo que o Pages não está habilitado ou que o site não foi encontrado.

**Publicar manualmente.** Vá em **Actions → OpsPilot web → Pages → Run workflow** e escolha o ramo `master`.

**Conferir uma publicação.** Em **Actions**, a execução mostra a URL no job "Publicar no Pages". A página de **Environments → github-pages** do repositório mostra o commit que está no ar.

**Build com outro caminho base.** O build lê `OPSPILOT_WEB_BASE`, cujo padrão é `/opspilot/`. O workflow usa `/<repo>/opspilot/`. Para reproduzir localmente:

```bash
OPSPILOT_WEB_BASE=/unipds-ia/opspilot/ npm --prefix web run build
OPSPILOT_WEB_BASE=/unipds-ia/opspilot/ npm --prefix web test   # confere que todo asset usa esse caminho
```

## Conectar a war room publicada à API

A war room publicada é só um site estático. A API roda onde você quiser, por exemplo na sua máquina.

1. Suba a API liberando a origem do site publicado. A origem não leva caminho e pode ser combinada com a origem local:

   ```bash
   OPSPILOT_CORS_ORIGINS=https://gabriel-barbosa-de-oliveira.github.io,http://localhost:5173 npm run dev
   ```

2. Abra a war room publicada e clique na **engrenagem**. Confira o endereço da API (o padrão é `http://localhost:3000`) e salve. O status deve mostrar "Conectado".
3. Mande uma mensagem, por exemplo "Quais alertas estão disparando?".

**HTTPS → `http://localhost`.** O site publicado é HTTPS. O Chrome e o Firefox aceitam chamar `http://localhost` a partir dele. O Chrome pode pedir permissão de acesso à rede local na primeira vez. O Safari pode bloquear como conteúdo misto. Nesse caso, use a war room local (`npm --prefix web run dev`) ou uma API com HTTPS.

Se aparecer "Não foi possível falar com a API", confira o endereço na engrenagem e se a origem está em `OPSPILOT_CORS_ORIGINS`.

## Fluxo de desenvolvimento

As mudanças seguem o Spec Kit: `/speckit.specify` → `/speckit.plan` → `/speckit.tasks` → `/speckit.implement`, com as specs versionadas em [`specs/`](specs/). Os princípios estão na [constitution](.specify/memory/constitution.md), e as convenções do dia a dia, no [`CLAUDE.md`](CLAUDE.md). O design da war room segue as [instruções de design](.github/instructions/design.instructions.md).
