# Tasks

## 1. Contrato em `shared-types`

- [x] 1.1 Criar `shared-types/src/lib/event.dto.ts` com `EventDTO` e `CreateEvent` (decisão 6 do design.md) e exportar em `shared-types/src/index.ts`; verificar com `nx build frontend` e `nx build api` sem erros de tipo depois das tarefas que os consomem

## 2. `POST /api/events` no `api`

- [x] 2.1 Criar `api/src/app/events/events.service.ts` com `create()` em memória e `id` por `randomUUID()`, no padrão de `SubmissionsService`; adicionar `events.service.spec.ts` cobrindo `id` gerado e dados preservados; verificar com `nx test api` verde
- [x] 2.2 Criar `create-event.body.ts` (`trim` em `name` e `address`, `@MaxLength(120)` e `@MaxLength(200)`, `@IsInt()` + `@Min(1)` em `capacity`, `@IsDateString({ strict: true })` + `@Matches` em `date`), `events.controller.ts` (`@Controller('events')`, `@Post()`) e `events.module.ts`, e importar `EventsModule` em `app.module.ts`
- [x] 2.3 Adicionar `events.controller.spec.ts` no padrão de `submissions.controller.spec.ts` (`Test.createTestingModule`, `createValidationPipe()`, `supertest`), cobrindo todos os cenários dos requisitos "API cria eventos válidos" e "API rejeita eventos inválidos"; verificar com `nx test api` verde

## 3. Tela `/event/new` no `frontend`

- [x] 3.1 Criar `frontend/src/app/events/event.service.ts` com `create(event: CreateEvent)` chamando `POST /api/events`; adicionar `event.service.spec.ts` no padrão de `submission.service.spec.ts`; verificar com `nx test frontend` verde
- [x] 3.2 Criar `EventFormComponent` (`event-form.component.{ts,html,scss}`, Standalone, OnPush, Signals, tokens `--cfp-*`) com ids `event-name`, `event-address`, `event-capacity` e `event-date`, atributos `name` iguais aos campos do contrato, atributos nativos de validação, erros em `#event-<campo>-error` e `.field-error`, e status em `.form-status`, seguindo as decisões 4, 5 e 6 do design.md
- [x] 3.3 Adicionar `event-form.component.spec.ts` cobrindo os cenários dos requisitos "Tela de cadastro acessível pela rota event/new", "Formulário valida os campos ao enviar" e "Formulário informa o resultado do envio", incluindo `capacity` numérico no corpo do `POST`; verificar com `nx test frontend` verde
- [x] 3.4 Adicionar a rota `path: 'event/new'` com `loadComponent` e título "Cadastrar evento" em `app.routes.ts`; verificar com `nx build frontend` sem erros nem avisos de budget, e com `nx run frontend:serve` abrindo `/event/new`

## 4. Cypress em `frontend-e2e`

- [x] 4.1 Em `nx.json`, trocar os alvos do `@nx/cypress/plugin` para `targetName: "e2e-cypress"` e `ciTargetName: "e2e-cypress-ci"`; criar `frontend-e2e/cypress.config.ts` (decisão 2 do design.md) e `frontend-e2e/src/support/e2e.ts`, e adicionar `"types": ["cypress", "node"]` em `frontend-e2e/tsconfig.json`
- [x] 4.2 Verificar com `nx show project frontend-e2e --json` que `e2e` continua usando Playwright e que `e2e-cypress` usa Cypress, e com `nx run frontend-e2e:e2e --list` (ou comando equivalente do Playwright) que nenhum arquivo `.cy.ts` é listado

## 5. Suíte `event-registration.cy.ts`

- [x] 5.1 Criar `frontend-e2e/src/e2e/event-registration.cy.ts` com o cenário de sucesso: `cy.visit('/event/new')`, preencher `input[name="name"]`, `input[name="address"]`, `input[name="capacity"]` e `input[name="date"]` (data como `2027-03-15`), clicar em `button[type="submit"]` e verificar a mensagem em `.form-status--success` e os campos vazios
- [x] 5.2 Adicionar o cenário de erro: `cy.visit('/event/new')`, clicar em `button[type="submit"]` com o formulário vazio e verificar com `cy.contains` as quatro mensagens, com `cy.get('input[name="<campo>"]:invalid').should('exist')` o estado nativo dos quatro campos, e que `.form-status--success` não existe
- [x] 5.3 Verificar com `grep -nE "cy\.intercept|cy\.prompt|@testing-library|data-cy" frontend-e2e/src/e2e/event-registration.cy.ts` que nada é encontrado, e que `package.json` não ganhou dependências

## 6. Verificação integrada

- [x] 6.1 Executar `env -u ELECTRON_RUN_AS_NODE npx nx run frontend-e2e:e2e-cypress` e confirmar os dois cenários verdes, subindo `frontend` e `api` pelo preset
- [x] 6.2 Quebrar temporariamente a mensagem de sucesso do `EventFormComponent`, confirmar que o cenário de sucesso falha, e reverter
- [x] 6.3 Executar `nx run-many -t test -p api frontend shared-types` e `nx run-many -t build -p api frontend`, confirmando que tudo passa sem regressões
