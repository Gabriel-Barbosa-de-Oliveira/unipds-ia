# Design

## Context

- `frontend`: rotas `''` e `dashboard`, sem `/event/new`. `SubmissionFormComponent` é o padrão a seguir (Standalone, OnPush, estado em `signal()`, `novalidate`, serviço `providedIn: 'root'`, testes Jest + `HttpTestingController`). Os Design Tokens `--cfp-*` estão em `styles.scss`.
- `api`: `submissions/` tem módulo, controller, service em memória e body com `class-validator`. `createValidationPipe()` usa `whitelist`, `forbidNonWhitelisted` e `enableImplicitConversion: false`, ou seja, números precisam chegar como número.
- `frontend-e2e` é um projeto Playwright. `example.spec.ts` espera o texto `Welcome`, que não existe mais na página. `cypress` 16.2.1 e `@nx/cypress` 23.3.0 estão instalados, mas não há `cypress.config.*` no workspace.
- `nx.json` registra `@nx/playwright/plugin` e `@nx/cypress/plugin`, ambos com `targetName: "e2e"`.
- `frontend:serve` depende de `api:serve`, então um único comando sobe as duas aplicações, com `/api` encaminhado pelo proxy.
- No terminal integrado do VS Code, `ELECTRON_RUN_AS_NODE=1` faz `cypress verify` falhar com `bad option`. Sem a variável, o binário é verificado com sucesso.

## Goals / Non-Goals

**Goals:**
- Teste E2E real, passando pelo Angular, pelo proxy e pelo NestJS.
- Cypress e Playwright convivendo em `frontend-e2e`, sem um sobrescrever o alvo do outro.
- Seguir os padrões já usados em `submissions/`.

**Non-Goals:**
- Extrair um formulário ou validação genéricos a partir de `SubmissionFormComponent`.
- Alterar comportamento ou testes do formulário de propostas.
- Corrigir `example.spec.ts`.

## Decisions

### 1. Cypress em `frontend-e2e`, com alvo próprio

Adicionar `cypress.config.ts` em `frontend-e2e/` e renomear, em `nx.json`, os alvos do plugin do Cypress para `targetName: "e2e-cypress"` e `ciTargetName: "e2e-cypress-ci"`.

Sem a renomeação, os dois plugins inferem `e2e` no mesmo projeto e um sobrescreve o outro. Um projeto `frontend-cypress-e2e` separado evitaria o conflito, mas contraria o pedido de usar o projeto E2E gerado. Remover o Playwright também está fora do que foi pedido. O Playwright só procura `*.spec`/`*.test`, então não executa arquivos `.cy.ts` mesmo com `testDir: './src'`.

### 2. Configuração pelo preset do Nx

`cypress.config.ts` usa `nxE2EPreset(__filename, { cypressDir: 'src', webServerCommands: { default: 'npx nx run frontend:serve' } })` e `baseUrl: 'http://localhost:4200'`. O preset define `src/**/*.cy.{js,jsx,ts,tsx}` como padrão de specs e `src/support/e2e.{js,ts}` como arquivo de suporte, que precisa existir, e reaproveita um servidor que já esteja rodando. `tsconfig.json` do projeto recebe `"types": ["cypress", "node"]`. Isso não colide com o Playwright, porque `example.spec.ts` importa `test` e `expect` explicitamente.

### 3. Testes contra a API real, sem `cy.intercept`

A suíte usa só `cy.visit`, `cy.get`, `cy.contains`, `should`, `type` e `click`. `cy.intercept` ficaria fora da regra de sintaxe pedida e esconderia a integração que justificou criar a API. Como o armazenamento é em memória e não há listagem, repetir a suíte não interfere nas asserções.

### 4. Validação: `novalidate`, atributos nativos e mensagens inline

O formulário mantém `novalidate`, como o de propostas, e os campos recebem `required`, `min="1"` e `step="1"`. Os balões nativos do navegador não fazem parte do DOM, então `cy.contains` não os enxerga. Além disso, o texto deles varia com o idioma do navegador. Por isso a suíte verifica duas coisas:

- as mensagens exibidas na tela, com `cy.contains`, que é o que o pedido chama de "aparecem na tela";
- o estado nativo, com `cy.get('input[name="name"]:invalid')`, que continua válido mesmo com `novalidate`.

Essa combinação atende às duas leituras de "mensagens nativas" sem depender do idioma.

### 5. Botão habilitado com o formulário vazio

Diferente do formulário de propostas, que desabilita o botão enquanto está inválido, aqui ele só fica desabilitado durante o envio. Caso contrário, o cenário de erro não conseguiria clicar para submeter. `onSubmit` marca todos os campos como tocados e retorna sem chamar a API se houver erro.

### 6. Contrato e nomes

- `EventDTO { id, name, address, capacity: number, date: string }` e `CreateEvent = Omit<EventDTO, 'id'>`. O nome `EventDTO` evita colidir com o tipo global `Event`.
- `date` trafega como `YYYY-MM-DD`, o valor do `<input type="date">`, validado por `@IsDateString({ strict: true })`, em que `strict` rejeita `2027-02-30`, e por `@Matches`, que recusa outras formas ISO 8601, como data com hora.
- `capacity` é mantido como texto no signal do input e convertido com `Number()` no envio, porque a API não faz conversão implícita.
- Limites de 120 e 200 caracteres para `name` e `address`, iguais aos de `CreateSubmissionBody`.
- Datas passadas são aceitas: o pedido não define essa regra, e bloqueá-las faria uma data fixa no teste expirar com o tempo.
- `toErrorMessage` é reescrita no componente, porque os textos são outros e extrair uma função para dois usos não compensa agora.

## Risks / Trade-offs

- **[`ELECTRON_RUN_AS_NODE=1` no terminal do VS Code impede o Cypress de abrir]** → Executar a suíte com `env -u ELECTRON_RUN_AS_NODE`, como indicado nas tarefas.
- **[Dois frameworks E2E no mesmo projeto aumentam a manutenção]** → Alvos separados e sufixos de arquivo distintos. Consolidar fica para uma decisão futura.
- **[Primeira carga do `frontend:serve` pode ser lenta]** → O preset espera a `baseUrl` responder antes de iniciar a suíte.
- **[Tela sem link de navegação]** → Limitação aceita: o pedido só define o acesso pela URL.
- **[Eventos cadastrados não podem ser consultados e somem ao reiniciar o `api`]** → Limitação aceita neste escopo, igual à de `submissions` em memória.
