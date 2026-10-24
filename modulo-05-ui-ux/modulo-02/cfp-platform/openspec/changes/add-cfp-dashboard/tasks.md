# Tasks

## 1. Endpoint de listagem no `api`

- [x] 1.1 Adicionar `findAll(): SpeakerDTO[]` ao `SubmissionsService`, devolvendo uma cópia do array em ordem de inserção; estender `submissions.service.spec.ts` cobrindo lista vazia, ordem de inserção após dois `create` e que alterar o array retornado não afeta a chamada seguinte; verificar com `nx test api` verde
- [x] 1.2 Adicionar `@Get()` ao `SubmissionsController` retornando `this.submissionsService.findAll()`; estender `submissions.controller.spec.ts` (mesmo módulo de teste + `supertest`) com um `describe('GET /submissions')` cobrindo: `200` com `[]` antes de qualquer envio (módulo novo por teste ou `beforeEach`); dois `POST` válidos seguidos de `GET` → `200` com os dois `SpeakerDTO` em ordem e iguais às respostas do `POST`; um `POST` rejeitado com `400` não aparece no `GET`; verificar com `nx test api` verde
- [x] 1.3 Verificar manualmente com `nx serve api`: `curl -X POST localhost:3000/api/submissions` com payload válido e depois `curl localhost:3000/api/submissions` → `200` com o item enviado

## 2. Design Tokens e estilos compartilhados

- [x] 2.1 Declarar em `frontend/src/styles.scss`, sob `:root`, os tokens da tabela da decisão 5 do design.md, com os valores literais atuais; adicionar a classe global `.button-link` com o mesmo visual do `button` do formulário (padding `0.625rem 1.25rem`, peso 600, raio, fundo primário, texto `--cfp-color-on-primary`, sem sublinhado, `display: inline-block`, `align-self: flex-start`) e `:focus-visible` com `--cfp-focus-ring` e offset `2px`; verificar com `nx build frontend` sem erros nem avisos de budget
- [x] 2.2 Trocar todos os valores fixos de cor, raio, fonte e foco de `submission-form.component.scss` por `var(--cfp-…)`; verificar que `grep -E '#[0-9a-fA-F]{3,6}' frontend/src/app/submissions/submission-form.component.scss` não retorna nada, que `nx test frontend` continua verde e, com `nx serve frontend`, que o formulário (estado inicial, erro de campo, botão desabilitado/habilitado, foco por teclado) está visualmente igual ao de antes

## 3. `CfpDashboardComponent`

- [x] 3.1 Criar `frontend/src/app/submissions/cfp-dashboard.component.ts` (Standalone, OnPush, seletor `app-cfp-dashboard`, imports `RouterLink`) com `inject(HttpClient)`, `speakers: WritableSignal<SpeakerDTO[]>`, `status` (`'loading' | 'loaded' | 'error'`), `isEmpty` e `statusMessage` como `computed`, e o `GET /api/submissions` disparado no construtor com `takeUntilDestroyed()`, conforme as decisões 2 e 3 do design.md; verificar com `nx build frontend` sem erros
- [x] 3.2 Criar o template `cfp-dashboard.component.html` conforme a decisão 4 do design.md: `section` com `aria-labelledby` e `aria-busy`, `h1` "Propostas enviadas", região viva `role="status" aria-live="polite"`, tabela com `caption`, `th scope="col"` (Nome, E-mail, Título da palestra, GDE), linhas via `@for ... track speaker.id`, GDE como "Sim"/"Não", estado vazio com convite para enviar a primeira proposta, e link `.button-link` `routerLink="/"` "Enviar nova proposta"; verificar pelos testes da 3.4
- [x] 3.3 Criar `cfp-dashboard.component.scss` usando apenas tokens (host com a mesma fonte/margem/padding do formulário e `max-width: 48rem`, `.table-wrapper` com `overflow-x: auto`, tabela `border-collapse: collapse` com bordas em `--cfp-color-border`, cabeçalho peso 600, mensagem de erro em `--cfp-color-danger`); verificar que não há hex no arquivo e que `nx build frontend` não emite aviso de budget
- [x] 3.4 Criar `cfp-dashboard.component.spec.ts` (Jest, `provideHttpClient()`, `provideHttpClientTesting()`, `provideRouter([])`, `HttpTestingController.verify()` no `afterEach`) cobrindo os cenários da spec: estado inicial `loading` com `speakers()` vazio, `aria-busy="true"` e região viva com "Carregando"; exatamente um `GET /api/submissions`; resposta com duas propostas → duas linhas na ordem, GDE "Sim"/"Não", `caption` presente e todos os `th` com `scope="col"`; resposta `[]` → sem tabela e mensagem de lista vazia; resposta `500` → `status()` `error`, sem tabela e região viva com a mensagem de erro; link de volta com `href="/"`; verificar com `nx test frontend` verde

## 4. Roteamento e navegação

- [x] 4.1 Adicionar a rota `{ path: 'dashboard', title: 'Propostas enviadas', loadComponent: ... }` em `frontend/src/app/app.routes.ts`, no padrão lazy da rota existente; verificar com `nx build frontend` (gera um chunk lazy para o dashboard) e abrindo `http://localhost:4200/dashboard` com `nx serve frontend`
- [x] 4.2 Adicionar ao `SubmissionFormComponent` o import de `RouterLink` e, no template, fora do `<form>` e abaixo da região viva, o link `<a class="button-link" routerLink="/dashboard">Ver propostas enviadas</a>` (envolver form + link num elemento se o layout exigir, sem mudar o `form` existente); adicionar `provideRouter([])` ao `TestBed` de `submission-form.component.spec.ts` e um teste verificando o link com `href="/dashboard"`; verificar com `nx test frontend` verde

## 5. Verificação integrada

- [x] 5.1 Com `nx serve frontend` (sobe o `api` via `dependsOn`): abrir `/`, enviar duas propostas (uma GDE), ativar "Ver propostas enviadas" → URL `/dashboard` sem recarregar e tabela com as duas linhas; ativar "Enviar nova proposta" → volta ao formulário; percorrer ambas as telas só com teclado confirmando o mesmo contorno de foco; reduzir a janela para ~360px e confirmar que a tabela rola horizontalmente sem quebrar a página; parar o `api` e recarregar `/dashboard` → mensagem de erro anunciada
- [x] 5.2 Rodar `nx run-many -t lint test build -p api frontend shared-types` e verificar tudo verde
