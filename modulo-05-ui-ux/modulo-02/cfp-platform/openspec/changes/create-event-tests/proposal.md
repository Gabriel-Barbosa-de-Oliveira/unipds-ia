# Proposal

## Why

O pedido é cobrir a tela de Cadastro de Eventos (`/event/new`) com testes E2E tradicionais em Cypress, mas essa tela não existe: o `frontend` só tem o formulário de proposta de palestra (`/`) e o dashboard (`/dashboard`), e o `api` não tem nenhum recurso de eventos. Por decisão do usuário, esta change entrega a fatia vertical completa (tela, API e testes), para que o cenário de sucesso exercite o sistema real em vez de um endpoint simulado.

## What Changes

- Novos tipos `EventDTO` e `CreateEvent` em `shared-types`.
- Novo `POST /api/events` no `api` (NestJS), com validação e armazenamento em memória, no mesmo padrão de `submissions`.
- Nova rota `event/new` no `frontend`, com `EventFormComponent` (campos Nome, Endereço, Capacidade e Data), que chama a API e mostra sucesso ou erro.
- Cypress configurado no projeto `frontend-e2e`, ao lado do Playwright existente, com o alvo `e2e-cypress`.
- Nova suíte `frontend-e2e/src/e2e/event-registration.cy.ts`, com cenário de sucesso e de erro, usando somente a sintaxe tradicional do Cypress e seletores CSS convencionais, sem bibliotecas externas de IA.

## Capabilities

### New Capabilities
- `event-registration`: cadastro de eventos — contrato de criação na API, comportamento observável da tela `/event/new` (validação, sucesso e falha) e sua cobertura E2E.

### Modified Capabilities
<!-- Nenhuma: openspec/specs/ está vazio. -->

## Impact

- **Código**: `shared-types/src/` (novos tipos); `api/src/app/events/` (novo módulo) e `api/src/app/app.module.ts`; `frontend/src/app/events/` (novo componente e serviço) e `frontend/src/app/app.routes.ts`; `frontend-e2e/` (`cypress.config.ts`, `src/support/e2e.ts`, `src/e2e/event-registration.cy.ts`, `tsconfig.json`); `nx.json` (nome dos alvos do plugin do Cypress).
- **API**: novo `POST /api/events`, pelo prefixo global `api` e pelo proxy `/api` já existentes. As rotas atuais não mudam.
- **Dependências**: nenhuma além de `cypress` e `@nx/cypress`, já presentes na árvore de trabalho junto com o registro do plugin em `nx.json`. Essas alterações ainda não foram commitadas e são pré-requisito desta change.
- **Caminho do teste**: o pedido cita `apps/frontend-e2e/...` como exemplo; este workspace não tem pasta `apps/`, então o arquivo fica em `frontend-e2e/src/e2e/`.
- **Fora do escopo**: listagem, edição e remoção de eventos; link de navegação para a tela (acesso pela URL); persistência em banco; autenticação; migrar ou corrigir os testes Playwright existentes.
