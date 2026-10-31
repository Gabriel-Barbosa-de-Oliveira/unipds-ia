# Spec Delta

## Purpose

Permite consultar as propostas de palestra já enviadas ao Call for Papers, tanto pela API quanto por um dashboard no frontend.

## ADDED Requirements

### Requirement: Endpoint lista as propostas enviadas
O `api` SHALL responder `GET /api/submissions` com `200 OK` e um array JSON de `SpeakerDTO` (`id`, `name`, `email`, `talkTitle`, `isGDE`) contendo todas as propostas aceitas desde o início do processo, na ordem em que foram recebidas.

#### Scenario: Nenhuma proposta enviada
- **WHEN** nenhuma proposta foi aceita e o cliente chama `GET /api/submissions`
- **THEN** a resposta é `200` com o corpo `[]`

#### Scenario: Propostas enviadas aparecem na listagem
- **WHEN** duas propostas válidas são aceitas por `POST /api/submissions` e em seguida o cliente chama `GET /api/submissions`
- **THEN** a resposta é `200` com um array de dois itens, na ordem de envio, cada um idêntico ao `SpeakerDTO` devolvido pelo respectivo `POST`

#### Scenario: Propostas rejeitadas não aparecem
- **WHEN** um `POST /api/submissions` é rejeitado com `400` e em seguida o cliente chama `GET /api/submissions`
- **THEN** a proposta rejeitada não consta no array

### Requirement: Dashboard acessível pela rota dashboard
O `frontend` SHALL exibir o dashboard de propostas na rota `/dashboard`, com título de documento próprio e um cabeçalho de nível 1.

#### Scenario: Acesso direto à rota
- **WHEN** o usuário abre `/dashboard`
- **THEN** a página exibe o dashboard com um `h1` descrevendo as propostas enviadas e o título do documento muda para o do dashboard

### Requirement: Dashboard carrega as propostas ao abrir
Ao ser aberto, o dashboard SHALL buscar a listagem em `GET /api/submissions` uma vez e anunciar o carregamento numa região viva educada (`role="status"`, `aria-live="polite"`).

#### Scenario: Carregamento em andamento
- **WHEN** o dashboard é aberto e a resposta ainda não chegou
- **THEN** a região viva anuncia que as propostas estão sendo carregadas e a área da listagem fica com `aria-busy="true"`

### Requirement: Dashboard exibe as propostas em tabela semântica
Com a resposta recebida e não vazia, o dashboard SHALL exibir uma tabela com legenda (`caption`) e uma coluna por campo: Nome, E-mail, Título da palestra e GDE, com cabeçalhos de coluna marcados como tal e uma linha por proposta, na ordem recebida. O campo `isGDE` SHALL aparecer como "Sim" ou "Não".

#### Scenario: Duas propostas recebidas
- **WHEN** a API responde com duas propostas, uma com `isGDE: true` e outra com `isGDE: false`
- **THEN** a tabela exibe duas linhas de dados com nome, e-mail e título de cada proposta, e a coluna GDE mostra "Sim" na primeira e "Não" na segunda

#### Scenario: Estrutura acessível da tabela
- **WHEN** a tabela é exibida
- **THEN** ela tem um `caption` e cada cabeçalho de coluna é um `th` com `scope="col"`

### Requirement: Dashboard informa quando não há propostas
Quando a API responde com um array vazio, o dashboard SHALL exibir uma mensagem de lista vazia no lugar da tabela, com um convite para enviar a primeira proposta.

#### Scenario: Resposta vazia
- **WHEN** a API responde `200` com `[]`
- **THEN** nenhuma tabela é exibida e a página mostra que ainda não há propostas enviadas

### Requirement: Dashboard informa falha de carregamento
Quando a requisição de listagem falha, o dashboard SHALL anunciar na região viva uma mensagem de erro em português, sem exibir a tabela.

#### Scenario: Erro do servidor
- **WHEN** a API responde com status `500` ou a requisição falha por rede
- **THEN** a região viva anuncia que não foi possível carregar as propostas e nenhuma tabela é exibida

### Requirement: Navegação entre formulário e dashboard
O formulário de submissão SHALL ter um link de navegação para `/dashboard`, e o dashboard SHALL ter um link de volta para o formulário (`/`). Ambos SHALL ser links (`<a href>`) navegáveis por teclado, com o estilo visual do botão primário do formulário.

#### Scenario: Do formulário para o dashboard
- **WHEN** o usuário ativa o link de navegação do formulário
- **THEN** o app navega para `/dashboard` sem recarregar a página

#### Scenario: Do dashboard para o formulário
- **WHEN** o usuário ativa o link de volta no dashboard
- **THEN** o app navega para `/` e exibe o formulário de submissão

### Requirement: Identidade visual compartilhada
O dashboard SHALL usar as mesmas cores, tipografia, raio de borda e indicador de foco do formulário, a partir de uma única fonte de Design Tokens. A adoção dos tokens MUST NOT alterar a aparência atual do formulário.

#### Scenario: Formulário inalterado após a adoção dos tokens
- **WHEN** o formulário é exibido depois da mudança
- **THEN** as cores do botão primário, borda dos campos, mensagens de erro e sucesso, e o contorno de foco têm os mesmos valores computados de antes

#### Scenario: Foco visível no dashboard
- **WHEN** o usuário navega por teclado até o link de volta no dashboard
- **THEN** ele recebe o mesmo contorno de foco usado nos controles do formulário
