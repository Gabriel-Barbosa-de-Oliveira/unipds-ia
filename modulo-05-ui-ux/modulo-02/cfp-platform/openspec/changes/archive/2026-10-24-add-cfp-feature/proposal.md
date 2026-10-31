# Proposal

## Why

A plataforma `cfp-platform` ainda não recebe submissões: o `api` e o `frontend` são esqueletos gerados pelo Nx e a lib `shared-types` exporta um `SpeakerDTO` que ninguém consome. Sem um fluxo de Call for Papers, palestrantes não têm como enviar propostas, e o contrato compartilhado não é exercitado por nenhum dos lados.

## What Changes

- Novo endpoint HTTP no `api` (NestJS) que recebe uma submissão de palestra, valida estritamente o corpo da requisição e rejeita payloads inválidos com `400 Bad Request`.
- Novo formulário de submissão no `frontend` (Angular, Standalone Components) com estado gerenciado por Signals, botão de envio bloqueado enquanto o formulário for inválido ou durante o envio, e semântica WAI-ARIA (rótulos, erros anunciados, estado de envio).
- `api` e `frontend` passam a importar o contrato `SpeakerDTO` da lib `shared-types`, criando as arestas de dependência no grafo do Nx.
- Testes unitários com Jest obrigatórios nos dois lados: rejeição de payloads inválidos no `api`; estado inicial do Signal e bloqueio do botão de envio no `frontend`.
- Ajuste do runner de testes do `frontend` para Jest (hoje usa `@angular/build:unit-test` com Vitest).

## Capabilities

### New Capabilities
- `cfp-submission`: submissão de propostas de palestra por palestrantes — contrato de dados, regras de validação, respostas do endpoint e comportamento observável do formulário (estado inicial, bloqueio de envio, acessibilidade).

### Modified Capabilities
<!-- Nenhuma: openspec/specs/ está vazio. -->

## Impact

- **Código**: `api/src` (novo módulo de submissões, `ValidationPipe` global em `main.ts`), `frontend/src` (novo componente de formulário, rota, `provideHttpClient`), `shared-types/src` (contrato consumido pelos dois apps).
- **API**: novo `POST /api/submissions` (prefixo global `api` já existente; proxy `/api` do frontend já aponta para `localhost:3000`).
- **Dependências novas**: `class-validator`, `class-transformer` (runtime do `api`); `supertest` + tipos (testes do `api`); `jest-preset-angular` (testes do `frontend`).
- **Grafo Nx**: `api -> shared-types` e `frontend -> shared-types` passam a existir, então `nx affected` propaga mudanças no contrato para ambos os apps.
- **Fora do escopo**: persistência em banco, autenticação, listagem/edição/revisão de submissões.
