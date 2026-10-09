# Feature Specification: Publicação da War Room no GitHub Pages

**Feature Branch**: `016-web-pages-deploy`

**Created**: 2026-10-09

**Status**: Draft

**Input**: User description: "Deploy do web/ no Pages via Actions: upload-pages-artifact + deploy-pages, permissions, README (atualizar)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - War room publicada automaticamente a cada mudança (Priority: P1)

Quem mantém o OpsPilot altera a war room e integra a mudança no ramo principal. Sem nenhum passo manual, a versão nova fica disponível num endereço público fixo, e qualquer pessoa de plantão abre a war room pelo navegador sem precisar rodar nada na própria máquina.

**Why this priority**: é o objetivo da feature. Hoje a war room só existe para quem roda o servidor de desenvolvimento. Sem publicação automática, cada atualização depende de alguém lembrar de publicar à mão.

**Independent Test**: dá para testar integrando uma mudança visível na war room (por exemplo, um texto do estado vazio) no ramo principal e conferindo, depois de alguns minutos, que o endereço público mostra a mudança, inclusive ao recarregar a página.

**Acceptance Scenarios**:

1. **Given** uma mudança na war room integrada no ramo principal, **When** a publicação termina, **Then** o endereço público serve a versão nova, com todos os recursos da página carregando sem erro.
2. **Given** a war room publicada, **When** a pessoa recarrega a página ou abre o endereço direto, **Then** a página continua funcionando.
3. **Given** uma mudança no ramo principal que não toca a war room (por exemplo, só na API), **When** ela é integrada, **Then** nenhuma publicação nova é disparada.
4. **Given** a necessidade de republicar sem mudança de código, **When** quem mantém dispara a publicação manualmente, **Then** a war room é republicada a partir do ramo principal.
5. **Given** duas integrações seguidas em pouco tempo, **When** as publicações se sobrepõem, **Then** a versão que fica no ar é a mais recente, e uma publicação nunca corrompe a outra.

---

### User Story 2 - Nada quebrado vai para o ar (Priority: P1)

Antes de publicar, a mesma checagem de qualidade exigida no desenvolvimento roda automaticamente (verificação de tipos, testes e build da war room). Se qualquer uma falhar, a publicação não acontece e a versão anterior continua no ar. Quem mantém vê qual etapa falhou.

**Why this priority**: a war room é usada durante incidentes. Uma versão quebrada no ar durante um plantão é pior do que uma versão um pouco desatualizada. Também é o que a constitution exige (Princípio V).

**Independent Test**: dá para testar abrindo uma mudança com um teste da war room falhando e conferindo que a execução para na etapa de testes, que nada é publicado e que o endereço público continua com a versão anterior.

**Acceptance Scenarios**:

1. **Given** uma mudança com teste falhando, **When** a publicação é disparada, **Then** ela para antes de publicar, o resultado aparece como falha com a etapa identificada, e a versão anterior continua no ar.
2. **Given** uma mudança com erro de tipos ou de build, **When** a publicação é disparada, **Then** o comportamento é o mesmo do cenário 1.
3. **Given** uma proposta de mudança (pull request) que toca a war room, **When** ela é aberta ou atualizada, **Then** a checagem de qualidade roda e mostra o resultado na proposta, sem publicar nada.

---

### User Story 3 - Publicação com o mínimo de acesso (Priority: P2)

O processo de publicação só tem as permissões necessárias para ler o código e publicar o site. Ele não pode alterar o repositório, abrir ou mexer em issues, nem acessar segredos que não usa. A checagem de pull requests roda só com leitura.

**Why this priority**: o OpsPilot opera sobre incidentes de produção (Princípio VI). A automação de publicação é uma porta de entrada comum para ataques à cadeia de suprimentos e deve ser fechada por padrão.

**Independent Test**: dá para testar lendo a configuração da automação e conferindo que as permissões listadas são exatamente leitura do conteúdo e escrita do site publicado (mais a identidade necessária para publicar), e que a checagem de pull request só tem leitura.

**Acceptance Scenarios**:

1. **Given** a configuração da publicação, **When** ela é revisada, **Then** ela declara só as permissões necessárias e nenhuma permissão de escrita no repositório.
2. **Given** um pull request vindo de um fork, **When** a checagem roda, **Then** ela não tem permissão de publicar nem acesso a segredos.

---

### User Story 4 - Saber como acessar, publicar e conectar à API (Priority: P2)

Quem chega ao projeto encontra no README como abrir a war room publicada, como ela é publicada, o que fazer na primeira vez (ativar a publicação no repositório), como apontá-la para uma API e o que a API precisa liberar (a origem do site publicado). O README do repositório aponta para o do OpsPilot.

**Why this priority**: a war room publicada só funciona se a API aceitar chamadas vindas do endereço público. Sem essa instrução, quem abre o site vê só "não foi possível falar com a API".

**Independent Test**: dá para testar com uma pessoa que nunca usou o projeto seguindo só o README: ela consegue abrir a war room publicada, rodar a API local liberando a origem publicada e trocar uma mensagem.

**Acceptance Scenarios**:

1. **Given** o README do OpsPilot, **When** uma pessoa nova o lê, **Then** ela encontra o endereço público, o passo único de ativação da publicação, como publicar manualmente e como conferir uma publicação.
2. **Given** a war room publicada e a API rodando localmente, **When** a pessoa segue o README para liberar a origem publicada na API e configurar o endereço da API na engrenagem, **Then** a conversa funciona.
3. **Given** o README da raiz do repositório, **When** alguém o lê, **Then** encontra um link para o OpsPilot e para a war room publicada.

---

### Edge Cases

- Primeira execução com a publicação ainda não ativada no repositório: a execução falha com mensagem que indica o passo de ativação, e o README explica esse passo.
- Mudança só em documentação da war room (sem código): a publicação dispara por estar dentro da war room. O custo é aceitável e a versão no ar não muda de comportamento.
- Mudança na própria configuração da automação: dispara a publicação, para validar a configuração nova.
- A war room publicada tenta falar com a API padrão local, por HTTP, a partir de uma página HTTPS. Os navegadores principais tratam o endereço local como confiável, mas alguns podem bloquear ou pedir permissão de acesso à rede local. O README documenta isso e a mensagem de erro existente já orienta conferir o endereço e as origens permitidas.
- O endereço público inclui o nome do repositório no caminho. Se o repositório for renomeado, o caminho muda e a publicação seguinte precisa refletir isso automaticamente.
- Falha temporária do serviço de publicação: a execução aparece como falha, a versão anterior continua no ar, e dá para republicar manualmente.

## Requirements *(mandatory)*

### Functional Requirements

**Publicação**

- **FR-001**: Toda integração no ramo principal que altere a war room (ou a configuração da própria publicação) MUST disparar uma publicação automática.
- **FR-002**: Integrações que não alterem a war room MUST NOT disparar publicação.
- **FR-003**: Quem mantém o projeto MUST conseguir disparar uma publicação manual a partir do ramo principal.
- **FR-004**: A war room publicada MUST funcionar sob o caminho público do site, com todos os recursos carregando e com recarga e acesso direto funcionando.
- **FR-005**: Publicações sobrepostas MUST resultar na versão mais recente no ar. Uma publicação em andamento MUST NOT ser interrompida no meio a ponto de deixar o site inconsistente.
- **FR-006**: Ao fim de uma publicação bem-sucedida, o endereço público MUST ficar visível no resultado da execução.

**Qualidade antes de publicar**

- **FR-007**: A publicação MUST rodar, antes de publicar, a verificação de tipos, os testes e o build da war room, e MUST NOT publicar se qualquer um falhar.
- **FR-008**: Pull requests que alterem a war room MUST rodar a mesma checagem de qualidade, sem publicar.
- **FR-009**: As dependências usadas na checagem e no build MUST ser exatamente as travadas no repositório (instalação reprodutível).

**Permissões**

- **FR-010**: A publicação MUST declarar só as permissões necessárias: leitura do conteúdo do repositório, escrita no site publicado e a identidade exigida para publicar. Nenhuma outra.
- **FR-011**: A checagem de pull requests MUST rodar só com leitura do conteúdo.
- **FR-012**: A automação MUST NOT usar segredos.

**Documentação**

- **FR-013**: O OpsPilot MUST ter um README com: o que é, como rodar a API e a war room localmente, o endereço público da war room, como a publicação funciona e quando dispara, o passo único de ativação no repositório, como publicar manualmente, e como conectar a war room publicada a uma API (configurar a origem permitida na API e o endereço na engrenagem).
- **FR-014**: O README da raiz do repositório MUST apontar para o README do OpsPilot e para a war room publicada.

### Key Entities

- **Publicação**: uma execução que pega a war room de uma versão do ramo principal, checa, constrói e põe no ar. Tem gatilho (integração ou manual), versão de origem, resultado (publicada ou falhou, com a etapa) e endereço público.
- **Site publicado**: a versão da war room que está no ar. Só uma por vez, substituída a cada publicação bem-sucedida.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Uma mudança integrada na war room fica disponível no endereço público em até 10 minutos, sem nenhum passo manual.
- **SC-002**: 100% das execuções com teste, verificação de tipos ou build falhando terminam sem publicar, e a versão anterior continua acessível.
- **SC-003**: 0 publicações disparadas por mudanças que não tocam a war room.
- **SC-004**: A automação declara no máximo 3 permissões, e nenhuma delas é de escrita no repositório.
- **SC-005**: Uma pessoa nova, seguindo só o README, abre a war room publicada e troca uma mensagem com uma API local em até 15 minutos.
- **SC-006**: A war room publicada carrega sem nenhum recurso com erro de "não encontrado", inclusive depois de recarregar a página.

## Assumptions

- **Ferramentas definidas pela pessoa usuária**: GitHub Actions com as actions oficiais `actions/upload-pages-artifact` e `actions/deploy-pages`, com `permissions` explícitas. O detalhamento fica no `/speckit.plan`.
- **Repositório monorepo**: o OpsPilot vive em `module-04-agentes-autonomos/ops-pilot/` dentro do repositório `unipds-ia`. A automação fica na raiz do repositório (é onde o GitHub procura) e filtra pelas mudanças dentro de `ops-pilot/web/`.
- **Endereço público**: site de projeto do GitHub Pages, `https://gabriel-barbosa-de-oliveira.github.io/unipds-ia/opspilot/`. O site do repositório fica sob `/unipds-ia/`, e a war room mantém o segmento `/opspilot/` da spec 015. O caminho base da war room passa a ser configurável no build: `/opspilot/` continua o padrão local, e a publicação usa `/unipds-ia/opspilot/`.
- **O repositório não publica outro site no Pages**: nenhum workflow nem site existe hoje. Esta automação passa a ser a dona do site Pages do repositório.
- **Ramo principal**: `master`.
- **Ativação única**: a fonte do Pages precisa ser "GitHub Actions" nas configurações do repositório. Isso é feito uma vez por quem administra o repositório e fica documentado no README. A automação não tenta ativar sozinha, porque isso exigiria permissão administrativa.
- **API não é publicada**: só a war room (site estático). A API continua rodando onde quem opera escolher. A origem do site publicado (`https://gabriel-barbosa-de-oliveira.github.io`, sem caminho) precisa estar em `OPSPILOT_CORS_ORIGINS`.
- **Endereço padrão da API continua local** (`http://localhost:3000`). Quem usa a versão publicada ajusta na engrenagem se a API estiver em outro lugar.
- **README**: o OpsPilot ainda não tem README próprio, então ele é criado. O README da raiz (hoje com duas linhas) é atualizado com o link.
- **Fora do escopo**: publicar a API, domínio próprio, ambientes de prévia por pull request e publicação de outros projetos do monorepo.
- **Dependências**: spec 015 (war room em `web/` com build estático e caminho base).
