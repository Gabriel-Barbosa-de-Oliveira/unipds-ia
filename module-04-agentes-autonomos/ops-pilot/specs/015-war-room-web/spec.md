# Feature Specification: War Room Web

**Feature Branch**: `015-war-room-web`

**Created**: 2026-10-09

**Status**: Draft

**Input**: User description: "War room web/ (Vite+react+TS) com as instructions de design: chat -> /chat, com "ver raciocínio" abrindo o trace tipado. 202 vira cartão aprovar/negar; engrenagem com URL da API; base /opspilot/; CORS"

## Clarifications

### Session 2026-10-09

- Q: O lado da API do fluxo de aprovação (202 + decisão) entra nesta feature? → A: Sim (opção A, recomendada). Adotada por padrão quando o `/speckit.plan` foi rodado sem resposta. Revisar se outra opção for preferida.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Conversar com o copiloto pelo navegador (Priority: P1)

Durante um incidente, a pessoa de plantão abre a war room no navegador e conversa com o copiloto como num chat: escreve uma pergunta ("quais alertas críticos estão abertos?"), vê que o copiloto está trabalhando e recebe a resposta na mesma conversa. As perguntas seguintes continuam a mesma conversa, então o copiloto lembra do que já foi dito.

**Why this priority**: é o núcleo da feature. Hoje só dá para falar com o copiloto por chamadas manuais à API. Sem o chat no navegador, nada mais da war room tem valor.

**Independent Test**: dá para testar abrindo a war room com a API no ar, mandando duas mensagens seguidas em que a segunda depende da primeira ("e quais deles são do checkout?"), e conferindo que as duas respostas aparecem em ordem e que a segunda usa o contexto da primeira.

**Acceptance Scenarios**:

1. **Given** a war room aberta e a API disponível, **When** a pessoa envia uma mensagem, **Then** a mensagem aparece na conversa na hora, um indicador de "pensando" aparece enquanto a resposta não chega, e a resposta do copiloto aparece em seguida.
2. **Given** uma conversa já iniciada, **When** a pessoa envia outra mensagem, **Then** ela continua a mesma conversa (o copiloto recebe o histórico) em vez de começar outra.
3. **Given** a conversa vazia, **When** a war room é aberta, **Then** aparece um estado vazio que explica o que dá para perguntar e oferece exemplos clicáveis.
4. **Given** uma resposta em andamento, **When** a pessoa tenta enviar outra mensagem, **Then** o envio fica bloqueado até a resposta chegar, sem perder o que foi digitado.
5. **Given** a API devolve erro (corpo inválido, tempo esgotado, falha interna ou API fora do ar), **When** a resposta chega, **Then** a conversa mostra uma mensagem humana do que aconteceu, o identificador da requisição (quando houver) e um botão "Tentar novamente", sem texto técnico cru.
6. **Given** a pessoa quer começar do zero, **When** ela escolhe "Nova conversa", **Then** a conversa é limpa e a próxima mensagem abre uma conversa nova.

---

### User Story 2 - Ver o raciocínio por trás de uma resposta (Priority: P1)

Antes de agir com base numa resposta, a pessoa de plantão quer saber como o copiloto chegou nela. Cada resposta tem um "ver raciocínio" que abre o trace daquela execução, evento a evento, com cada tipo de evento mostrado de forma própria: a rota escolhida e o motivo, os pensamentos, o plano em passos, as ferramentas chamadas com seus argumentos, o que cada ferramenta devolveu, as críticas da reflexão, as trocas de modelo e a resposta final. Cada evento mostra a etapa do fluxo em que aconteceu. Junto vão as métricas da execução (chamadas ao modelo, tempo, modelo usado).

**Why this priority**: em incidente real, confiar cegamente numa resposta é perigoso. O trace já existe na API, mas sem interface ninguém olha para ele. Ver o raciocínio é o que diferencia a war room de um chat genérico.

**Independent Test**: dá para testar mandando uma pergunta que faça o copiloto usar uma ferramenta, abrindo "ver raciocínio" e conferindo que todos os eventos devolvidos pela API aparecem, na ordem, cada tipo com seu formato visual e com a etapa indicada.

**Acceptance Scenarios**:

1. **Given** uma resposta recebida, **When** a pessoa clica em "ver raciocínio", **Then** abre um painel com todos os eventos do trace daquela resposta, na ordem em que aconteceram, sem sair da conversa.
2. **Given** um trace com eventos de tipos diferentes, **When** o painel é exibido, **Then** cada tipo (rota, pensamento, plano, ação, observação, crítica, troca de modelo, resposta) tem rótulo, ícone e formato próprios: o plano aparece como lista numerada, a ação mostra o nome da ferramenta e os argumentos, a observação mostra o resultado de forma legível e recolhível quando for longo.
3. **Given** o painel aberto, **When** a pessoa lê um evento, **Then** ela vê em que etapa do fluxo ele aconteceu (contexto, roteador, estratégia, resposta).
4. **Given** o painel aberto, **When** a pessoa olha o topo, **Then** vê a rota escolhida, as métricas da execução e o identificador da requisição, com opção de copiá-lo.
5. **Given** o painel aberto, **When** a pessoa aperta Esc ou fecha o painel, **Then** ele fecha e o foco volta para o botão "ver raciocínio" que o abriu.
6. **Given** um trace com um tipo de evento que a war room não conhece, **When** o painel é exibido, **Then** o evento aparece de forma genérica (tipo e conteúdo bruto legível) em vez de sumir ou quebrar o painel.

---

### User Story 3 - Aprovar ou negar uma ação antes que ela aconteça (Priority: P2)

Quando o copiloto quer fazer algo que muda o estado da produção (por exemplo, abrir ou resolver um incidente), a API não executa direto: responde que a ação está aguardando aprovação humana. Na war room, isso vira um cartão na conversa que mostra o que o copiloto quer fazer, em linguagem clara (ação, alvo, argumentos e o motivo), com dois botões: "Aprovar" e "Negar". Aprovando, a ação é executada e a resposta final aparece na conversa. Negando, a ação não acontece e o copiloto confirma que ela foi cancelada.

**Why this priority**: é a guarda humana para ações com efeito em produção (Princípio VI da constitution). Depende de o chat (US1) já funcionar e do contrato de aprovação existir na API.

**Independent Test**: dá para testar pedindo algo como "resolva o incidente INC-42", conferindo que aparece um cartão de aprovação em vez de uma resposta final e que nada mudou nos dados. Depois, aprovando, a mudança acontece e a resposta aparece. Repetindo o pedido e negando, nada muda.

**Acceptance Scenarios**:

1. **Given** uma mensagem que leva o copiloto a uma ação que muda a produção, **When** a API responde que a ação aguarda aprovação, **Then** a conversa mostra um cartão de aprovação com a ação, o alvo, os argumentos e os botões "Aprovar" e "Negar", e nenhuma resposta final.
2. **Given** um cartão de aprovação pendente, **When** a pessoa clica em "Aprovar", **Then** os dois botões ficam desabilitados com indicador de progresso, a decisão é enviada, e a resposta final do copiloto aparece logo abaixo, com seu próprio "ver raciocínio".
3. **Given** um cartão de aprovação pendente, **When** a pessoa clica em "Negar", **Then** a ação não é executada, o cartão passa para o estado "Negada" e o copiloto confirma o cancelamento na conversa.
4. **Given** um cartão já decidido, **When** a pessoa volta a ele, **Then** o cartão mostra a decisão tomada (aprovada ou negada) e não aceita outro clique.
5. **Given** a decisão falha ao ser enviada (rede ou API), **When** o erro chega, **Then** o cartão volta a ficar pendente com uma mensagem de erro e permite tentar de novo, sem decidir em dobro.
6. **Given** uma aprovação pendente, **When** a pessoa tenta enviar nova mensagem, **Then** a war room avisa que existe uma ação aguardando decisão e pede para decidir antes.

---

### User Story 4 - Apontar a war room para a API certa (Priority: P2)

A war room pode falar com a API local, de staging ou de produção. Um ícone de engrenagem abre as configurações, onde a pessoa vê e muda o endereço da API. O endereço fica guardado no navegador para as próximas visitas. Ao salvar, a war room confere se a API responde e mostra se conectou ou não.

**Why this priority**: sem isso a war room só funciona com a API num endereço fixo. É pequeno, mas necessário para usar fora da máquina de desenvolvimento.

**Independent Test**: dá para testar abrindo as configurações, trocando o endereço por um válido e por um inválido, e conferindo que o válido é aceito, lembrado depois de recarregar a página e usado nas mensagens seguintes, e que o inválido é rejeitado com mensagem clara.

**Acceptance Scenarios**:

1. **Given** a war room aberta, **When** a pessoa clica na engrenagem, **Then** abre um painel de configurações com o endereço da API em uso preenchido.
2. **Given** o painel de configurações, **When** a pessoa digita um endereço mal formado e tenta salvar, **Then** o campo mostra o erro junto dele e o endereço anterior continua em uso.
3. **Given** um endereço válido, **When** a pessoa salva, **Then** a war room testa a conexão e mostra "conectado" ou "não foi possível conectar". O endereço fica salvo mesmo que a conexão falhe, com o aviso visível.
4. **Given** um endereço salvo, **When** a página é recarregada, **Then** a war room continua usando o endereço salvo.
5. **Given** nenhum endereço salvo, **When** a war room abre pela primeira vez, **Then** ela usa um endereço padrão e as configurações permitem restaurá-lo.
6. **Given** a API do endereço configurado está fora do ar, **When** a pessoa envia uma mensagem, **Then** o erro sugere conferir o endereço e oferece atalho para abrir as configurações.

---

### User Story 5 - Acessar a war room por um caminho próprio e falar com a API de outra origem (Priority: P3)

A war room é publicada sob o caminho `/opspilot/` de um domínio (por exemplo, `https://exemplo.com/opspilot/`), e não na raiz. Ela abre e funciona normalmente nesse caminho, inclusive ao recarregar a página ou ao abrir um link direto. A API fica em outra origem, e a API aceita chamadas vindas do navegador apenas das origens autorizadas.

**Why this priority**: é requisito de publicação. Não muda o que a pessoa faz, mas sem isso a war room não funciona fora do ambiente de desenvolvimento.

**Independent Test**: dá para testar publicando a war room sob `/opspilot/`, abrindo e recarregando a página e conferindo que tudo carrega. Depois, apontando para uma API em outra origem, as mensagens funcionam se a origem estiver autorizada e são recusadas se não estiver.

**Acceptance Scenarios**:

1. **Given** a war room publicada sob `/opspilot/`, **When** a pessoa abre o endereço, **Then** a interface e todos os seus recursos carregam sem erros.
2. **Given** a war room publicada sob `/opspilot/`, **When** a pessoa recarrega a página, **Then** ela continua funcionando.
3. **Given** a war room numa origem autorizada pela API, **When** ela envia uma mensagem, **Then** a API aceita a chamada do navegador e a resposta chega normalmente, com o identificador da requisição legível.
4. **Given** a war room numa origem não autorizada, **When** ela envia uma mensagem, **Then** o navegador bloqueia a resposta e a war room mostra um erro que sugere conferir o endereço e a configuração de origens permitidas da API.

---

### Edge Cases

- Resposta muito longa ou com quebras de linha e listas: aparece formatada e legível, sem estourar a largura no celular.
- Trace vazio (por exemplo, numa falha): "ver raciocínio" mostra um estado vazio explicando que essa execução não registrou eventos.
- Observação de ferramenta com resultado grande (lista de dezenas de alertas): aparece recolhida, com opção de expandir.
- A requisição demora até o teto da API (minutos): o indicador de "pensando" continua, mostra há quanto tempo está esperando e, no tempo esgotado, mostra erro com "Tentar novamente".
- A pessoa recarrega a página no meio da conversa: a conversa visível some (não é guardada no navegador) e a war room começa com o estado vazio. Ver Assumptions.
- A conversa referenciada não existe mais na API (resposta "conversa não encontrada"): a war room avisa e oferece começar uma nova conversa.
- Duplo clique em "Aprovar"/"Negar" ou em "Enviar": só uma requisição sai.
- Aprovação pendente que a API não reconhece mais (já decidida ou expirada): o cartão mostra que a ação não está mais disponível e não executa nada.
- Endereço da API com ou sem barra no final: os dois funcionam igual.
- Tema do sistema muda com a war room aberta: o tema acompanha, a não ser que a pessoa tenha escolhido um manualmente.
- Uso só com teclado e leitor de tela: novas respostas e novos cartões de aprovação são anunciados, e todo controle é alcançável.

## Requirements *(mandatory)*

### Functional Requirements

**Chat**

- **FR-001**: A war room MUST permitir enviar uma mensagem de texto ao copiloto e mostrar a resposta na conversa, na ordem em que foram trocadas.
- **FR-002**: A war room MUST manter a continuidade da conversa: depois da primeira resposta, as mensagens seguintes MUST seguir na mesma conversa da API até a pessoa escolher "Nova conversa".
- **FR-003**: A war room MUST mostrar a mensagem enviada imediatamente e um indicador de "pensando" com tempo decorrido até a resposta chegar.
- **FR-004**: A war room MUST impedir envio de mensagem vazia e envio duplicado enquanto uma resposta está pendente, preservando o texto digitado.
- **FR-005**: A war room MUST tratar os quatro estados de toda área que carrega dados: carregando, vazio, erro e sucesso.
- **FR-006**: A war room MUST traduzir cada erro da API (corpo inválido, estratégia desconhecida, conversa não encontrada, tempo esgotado, erro interno, falha de rede/origem bloqueada) numa mensagem humana com ação sugerida, mostrando o identificador da requisição quando existir e nunca exibindo stack trace ou resposta crua.
- **FR-007**: A war room MUST oferecer "Tentar novamente" para mensagens que falharam, reenviando o mesmo texto.

**Raciocínio (trace)**

- **FR-008**: Toda resposta do copiloto MUST ter uma ação "ver raciocínio" que abre o trace daquela execução sem sair da conversa.
- **FR-009**: O painel de raciocínio MUST mostrar todos os eventos do trace na ordem recebida, cada um com a etapa do fluxo em que aconteceu.
- **FR-010**: O painel MUST dar a cada tipo de evento conhecido (rota, pensamento, plano, ação, observação, crítica, troca de modelo, resposta) um rótulo, um ícone e um formato próprios, e cada tipo MUST ser distinguível sem depender só de cor.
- **FR-011**: O painel MUST mostrar eventos de tipo desconhecido de forma genérica, sem quebrar nem omitir.
- **FR-012**: O painel MUST mostrar a rota escolhida (e o motivo), as métricas da execução e o identificador da requisição, com ação para copiá-lo.
- **FR-013**: Resultados de ferramenta ou argumentos longos MUST aparecer recolhidos, com opção de expandir.

**Aprovação humana**

- **FR-014**: Quando a API responder que uma ação aguarda aprovação humana, a war room MUST mostrar um cartão de aprovação na conversa, no lugar de uma resposta final.
- **FR-015**: O cartão MUST mostrar, em linguagem clara, a ação pretendida, o alvo, os argumentos e o motivo (quando informado), com os botões "Aprovar" e "Negar".
- **FR-016**: Ao aprovar ou negar, a war room MUST enviar a decisão uma única vez, desabilitar os botões durante o envio e mostrar o resultado na conversa: a resposta final, se aprovada, ou a confirmação de cancelamento, se negada.
- **FR-017**: Um cartão decidido MUST exibir a decisão e não aceitar nova decisão. Se o envio da decisão falhar, o cartão MUST voltar a ficar pendente com mensagem de erro.
- **FR-018**: Enquanto houver aprovação pendente, a war room MUST impedir novas mensagens e explicar por quê.
- **FR-019**: O lado da API faz parte desta feature. A API MUST interceptar as ações que mudam a produção antes de executá-las, responder "aceito, aguardando aprovação" com a ação pendente, receber a decisão (aprovar ou negar) e só então executar ou cancelar. Uma ação pendente aceita no máximo uma decisão e expira se ninguém decidir a tempo.

**Configurações**

- **FR-020**: A war room MUST ter um ícone de engrenagem que abre as configurações, contendo o endereço da API em uso.
- **FR-021**: A war room MUST validar o endereço informado (URL absoluta http/https) antes de salvar, com erro junto ao campo, mantendo o endereço anterior em caso de erro.
- **FR-022**: A war room MUST guardar o endereço no navegador, usá-lo nas visitas seguintes e permitir restaurar o endereço padrão.
- **FR-023**: Ao salvar, a war room MUST testar a conexão com a API e mostrar o resultado, sem impedir o salvamento se o teste falhar.
- **FR-024**: As configurações MUST permitir escolher o tema (claro, escuro ou seguir o sistema), guardado no navegador.

**Publicação e acesso**

- **FR-025**: A war room MUST funcionar publicada sob o caminho `/opspilot/`, incluindo recarregar a página e abrir links diretos.
- **FR-026**: A API MUST aceitar chamadas vindas do navegador apenas de origens autorizadas por configuração, e MUST recusar as demais.
- **FR-027**: A API MUST permitir que o navegador leia o cabeçalho com o identificador da requisição nas respostas para origens autorizadas.
- **FR-028**: Com a configuração de origens vazia, a API MUST autorizar só a origem padrão de desenvolvimento local da war room.

**Design e acessibilidade** (conforme `.github/instructions/design.instructions.md`)

- **FR-029**: A war room MUST seguir as diretrizes de design do projeto: hierarquia com um título principal e uma ação primária por área, espaçamento só da escala definida, estados vazios e de erro explícitos, cores por tokens semânticos com tema claro e escuro.
- **FR-030**: A war room MUST atender WCAG 2.1 AA: contraste mínimo, operação completa por teclado com foco visível, rótulos em todos os controles, alvos de toque de 44×44px, anúncio de novas respostas e cartões de aprovação para leitores de tela, e respeito à preferência de movimento reduzido.
- **FR-031**: A war room MUST ser usável em telas de celular (a partir de 360px de largura) sem rolagem horizontal; no celular, o painel de raciocínio ocupa a tela inteira.

### Key Entities

- **Mensagem**: um item da conversa. Tem autor (pessoa ou copiloto), texto, momento e estado (enviando, entregue, falhou). Mensagens do copiloto guardam a execução que as produziu.
- **Execução**: o resultado de uma ida ao copiloto. Tem identificador da requisição, resposta, rota escolhida (com motivo e origem), métricas (chamadas ao modelo, tempo, modelo usado, tokens) e trace.
- **Evento de trace**: um passo da execução. Tem tipo (rota, pensamento, plano, ação, observação, crítica, troca de modelo, resposta), posição na sequência, etapa do fluxo e conteúdo próprio do tipo (por exemplo, ferramenta e argumentos para ação, lista de passos para plano).
- **Aprovação pendente**: uma ação que o copiloto quer executar e que espera decisão humana. Tem identificador, ação, alvo, argumentos, motivo e estado (pendente, enviando, aprovada, negada, indisponível).
- **Conversa**: a sequência de mensagens com um identificador da API que garante a continuidade.
- **Configurações**: endereço da API e tema escolhido, guardados por navegador.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Uma pessoa de plantão que nunca usou a war room consegue fazer uma pergunta e receber a resposta em menos de 30 segundos a partir da abertura da página, sem contar o tempo de resposta do copiloto.
- **SC-002**: A partir de qualquer resposta, a pessoa chega ao raciocínio completo em 1 clique, e 100% dos eventos devolvidos pela API aparecem no painel, na ordem.
- **SC-003**: 100% das ações que mudam a produção passam por um cartão de aprovação antes de serem executadas, e nenhuma ação negada altera dados.
- **SC-004**: Todo erro da API ou de rede resulta numa mensagem compreensível com ação sugerida. Nenhuma tela mostra texto técnico cru (stack trace, JSON de erro).
- **SC-005**: Trocar o endereço da API leva menos de 1 minuto, e o endereço sobrevive a recarregar a página.
- **SC-006**: A war room passa numa verificação de acessibilidade WCAG 2.1 AA sem violações nos dois temas, e todos os fluxos (enviar, ver raciocínio, aprovar/negar, configurar) podem ser concluídos só com teclado.
- **SC-007**: A interface aparece pronta para uso em até 2 segundos numa conexão comum, publicada sob `/opspilot/`.

## Assumptions

- **Stack definida pela pessoa usuária**: a war room vive em `web/`, como aplicação separada da API, feita com Vite + React + TypeScript. A constitution lista só a stack da API; incluir uma stack de frontend pode exigir amendment, o que deve ser avaliado no `/speckit.plan`.
- **Público**: pessoas de plantão e quem opera o copiloto, num computador ou celular, em sessões curtas durante incidentes.
- **Sem autenticação nesta versão**: a war room não tem login; o acesso é controlado pelo lugar onde ela é publicada e pela lista de origens permitidas da API.
- **Conversa não persiste no navegador**: recarregar a página começa uma conversa nova na tela. A conversa continua existindo na API. Retomar conversas antigas fica fora do escopo.
- **Endereço padrão da API**: o endereço local de desenvolvimento (`http://localhost:3000`) quando nada foi salvo.
- **Teste de conexão**: usa um endpoint de leitura que já existe na API (por exemplo, o de estatísticas), sem criar endpoint novo só para isso.
- **Ações que exigem aprovação**: as que mudam o estado da produção, ou seja, abrir e resolver incidentes. Leituras (listar alertas, listar incidentes, consultar runbook) não exigem.
- **"202"**: na descrição, "202" é a resposta da API que indica "aceito, aguardando aprovação humana", diferente da resposta de sucesso com a resposta final.
- **Origens permitidas da API**: configuradas por variável de ambiente, como lista de origens. Sem configuração, só a origem de desenvolvimento local da war room é aceita (FR-028).
- **Fora do escopo**: streaming da resposta token a token, painel de estatísticas, consulta de execuções antigas por identificador, internacionalização (a interface é em português) e suporte a navegadores antigos.
- **Dependências**: o `/chat` existente (spec 003) com conversas (006), trace por etapa do grafo (012) e identificador da requisição (014).
