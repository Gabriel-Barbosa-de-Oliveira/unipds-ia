# Proposal

## Why

O `cfp-platform` já recebe propostas via `POST /api/submissions`, mas elas ficam presas no repositório em memória do `api`: não há como ver o que foi enviado. Um dashboard de listagem fecha o ciclo do Call for Papers e permite conferir as submissões sem inspecionar o servidor.

## What Changes

- Nova rota `GET /api/submissions` no `api` (NestJS), no controller de submissões já existente, retornando `SpeakerDTO[]` com todas as propostas recebidas.
- Novo `CfpDashboardComponent` no `frontend` (Angular, Standalone, OnPush), que consome a rota via `HttpClient` e mantém a lista num `WritableSignal<SpeakerDTO[]>`, com estados de carregamento, lista vazia e erro anunciados por região viva.
- Listagem em HTML semântico (`<table>` com `<caption>` e cabeçalhos `scope="col"`), seguindo a identidade visual do formulário existente.
- As cores, raios, tipografia e foco hoje fixos no SCSS do formulário viram Design Tokens (CSS custom properties globais), consumidos pelo formulário e pelo dashboard — sem mudança visual no formulário.
- Nova rota `path: 'dashboard'` no roteador do `frontend`, um link de navegação no formulário para o dashboard (estilo do botão primário) e um link de volta no dashboard.

## Capabilities

### New Capabilities
- `cfp-submission-listing`: consulta das propostas de palestra já enviadas — resposta do endpoint de listagem e comportamento observável do dashboard (estados de carregamento, vazio, erro, dados exibidos e navegação de/para o formulário).

### Modified Capabilities
<!-- Nenhuma: openspec/specs/ está vazio (a change anterior, add-cfp-feature, foi arquivada sem specs). -->

## Impact

- **Código**: `api/src/app/submissions/` (novo método no service e handler `@Get()` no controller, mais testes); `frontend/src/app/submissions/` (novo componente e testes), `frontend/src/app/app.routes.ts` (rota `dashboard`), `frontend/src/styles.scss` (tokens), `submission-form.component.{html,scss,ts,spec.ts}` (link de navegação, uso dos tokens, `provideRouter` nos testes).
- **API**: novo `GET /api/submissions` (prefixo global `api` e proxy `/api` do frontend já existentes). O `POST` atual não muda.
- **Dependências**: nenhuma nova.
- **Privacidade**: o endpoint expõe nome e e-mail de todos os palestrantes sem autenticação (ver design.md, Riscos).
- **Fora do escopo**: autenticação/autorização, paginação, filtros, ordenação configurável, edição/remoção e persistência em banco.
- **Nomenclatura**: o pedido cita `CfpController` e `CfpFormComponent`; no código eles se chamam `SubmissionsController` e `SubmissionFormComponent`. Esta change estende esses existentes em vez de criar novos com outro nome.
