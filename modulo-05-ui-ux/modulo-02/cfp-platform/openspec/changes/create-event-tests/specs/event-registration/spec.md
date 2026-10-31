# Spec Delta

## Purpose

Permite cadastrar eventos com nome, endereço, capacidade e data, pela API e pela tela `/event/new`, com o fluxo principal coberto por testes E2E em Cypress.

## ADDED Requirements

### Requirement: API cria eventos válidos
O `api` SHALL aceitar `POST /api/events` com `name`, `address`, `capacity` e `date` válidos e responder `201 Created` com um `EventDTO` contendo um `id` gerado e os dados recebidos, removendo espaços nas pontas de `name` e `address`.

#### Scenario: Evento válido
- **WHEN** o cliente envia `{ "name": "Angular Summit", "address": "Av. Paulista, 1000", "capacity": 200, "date": "2027-03-15" }`
- **THEN** a resposta é `201` com esses mesmos valores e um `id` não vazio

#### Scenario: Espaços nas pontas
- **WHEN** o cliente envia `name` e `address` válidos com espaços nas pontas
- **THEN** a resposta é `201` com `name` e `address` sem esses espaços

### Requirement: API rejeita eventos inválidos
O `api` SHALL responder `400 Bad Request`, sem armazenar o evento, quando `name` ou `address` estiverem vazios ou acima de 120 e 200 caracteres, respectivamente, quando `capacity` não for um inteiro maior ou igual a 1, quando `date` não for uma data `YYYY-MM-DD` válida, ou quando houver campos desconhecidos.

#### Scenario: Campo obrigatório vazio
- **WHEN** o cliente envia `name` composto apenas de espaços
- **THEN** a resposta é `400`

#### Scenario: Capacidade inválida
- **WHEN** o cliente envia `capacity` igual a `0`, `1.5` ou `"200"`
- **THEN** a resposta é `400`

#### Scenario: Data inválida
- **WHEN** o cliente envia `date` igual a `2027-02-30`
- **THEN** a resposta é `400`

#### Scenario: Campo desconhecido
- **WHEN** o cliente envia um campo fora do contrato
- **THEN** a resposta é `400`

### Requirement: Tela de cadastro acessível pela rota event/new
O `frontend` SHALL exibir, na rota `/event/new`, um formulário com título de documento próprio, um `h1` e os campos Nome (`name`), Endereço (`address`), Capacidade (`capacity`, numérico) e Data (`date`, tipo data), cada um com rótulo associado, além de um botão de envio.

#### Scenario: Acesso direto à rota
- **WHEN** o usuário abre `/event/new`
- **THEN** a página exibe os quatro campos vazios, com `name` e seletor `id` estáveis, e o botão de envio habilitado

### Requirement: Formulário valida os campos ao enviar
O `frontend` SHALL marcar os quatro campos como obrigatórios com atributos nativos de validação e, ao enviar o formulário com dados inválidos, SHALL exibir na tela a mensagem de erro de cada campo inválido, sem chamar a API.

#### Scenario: Envio do formulário vazio
- **WHEN** o usuário clica no botão de envio com o formulário vazio
- **THEN** a tela exibe "Nome é obrigatório.", "Endereço é obrigatório.", "Capacidade é obrigatória." e "Data é obrigatória.", cada campo fica em estado nativo `:invalid`, e nenhuma requisição é feita

#### Scenario: Capacidade não positiva
- **WHEN** o usuário preenche a capacidade com `0` e envia
- **THEN** a tela exibe "Capacidade deve ser um número inteiro maior que zero." e nenhuma requisição é feita

#### Scenario: Erro some após correção
- **WHEN** o usuário corrige um campo que exibia erro
- **THEN** a mensagem de erro desse campo deixa de ser exibida

### Requirement: Formulário informa o resultado do envio
O `frontend` SHALL enviar os dados válidos para `POST /api/events`, com `capacity` numérico, desabilitar o botão durante o envio e anunciar o resultado em uma região viva. Em caso de sucesso SHALL limpar o formulário; em caso de falha SHALL preservar os dados digitados.

#### Scenario: Envio bem-sucedido
- **WHEN** o usuário envia dados válidos com nome "Angular Summit" e a API responde `201`
- **THEN** a tela exibe `Evento "Angular Summit" cadastrado com sucesso.` e os campos ficam vazios

#### Scenario: API recusa o evento
- **WHEN** a API responde `400` com uma mensagem de validação
- **THEN** a tela exibe "O servidor recusou o evento:" seguido da mensagem, e os dados digitados são mantidos

#### Scenario: API indisponível
- **WHEN** a requisição falha por qualquer outro motivo
- **THEN** a tela exibe "Não foi possível cadastrar o evento. Tente novamente." e os dados digitados são mantidos

### Requirement: Cadastro coberto por testes E2E em Cypress
O projeto `frontend-e2e` SHALL conter `src/e2e/event-registration.cy.ts`, executado por `nx run frontend-e2e:e2e-cypress` contra o `frontend` e o `api` reais, usando somente comandos tradicionais do Cypress e seletores CSS por `class`, `id` ou `name`, sem bibliotecas externas de IA e sem simular a API.

#### Scenario: Cenário de sucesso automatizado
- **WHEN** a suíte visita `/event/new`, preenche os quatro campos com `cy.get()` e clica no botão de envio
- **THEN** a suíte verifica com `should` ou `cy.contains()` a mensagem de sucesso e os campos vazios

#### Scenario: Cenário de erro automatizado
- **WHEN** a suíte visita `/event/new` e clica no botão de envio com o formulário vazio
- **THEN** a suíte verifica com `should` ou `cy.contains()` as quatro mensagens de erro e o estado nativo `:invalid` dos campos

#### Scenario: Playwright continua disponível
- **WHEN** o Cypress é configurado em `frontend-e2e`
- **THEN** o alvo `e2e` continua executando o Playwright, e `e2e-cypress` executa o Cypress
