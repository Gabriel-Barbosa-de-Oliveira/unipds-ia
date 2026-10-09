# Feature Specification: Modo Equipe

**Feature Branch**: `017-team-mode`

**Created**: 2026-10-09

**Status**: Draft

**Input**: User description: "Modo equipe em src/team: supervisor com withStructuredOutput({ next, brief }) sobre um blackboard no estado. Papéis: analista (só leitura, não propõe), planejador (sem tools), executor (incidentes, sem bypass). Evento "handoff" no trace, renderizado no "ver raciocínio". Rota "team""

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Uma equipe de especialistas resolve o pedido (Priority: P1)

Durante um incidente complexo, a pessoa de plantão pede algo que exige investigar, decidir e agir, por exemplo: "o checkout está lento, descubra o que está acontecendo e abra um incidente se for o caso". Em vez de um único agente fazer tudo, o copiloto trabalha como uma equipe coordenada por um supervisor:
- o **analista** levanta os fatos (alertas, incidentes, runbooks) e só relata o que encontrou;
- o **planejador** transforma os fatos num plano de ação, sem consultar nem alterar nada;
- o **executor** executa as ações sobre incidentes.

O supervisor decide, a cada passo, quem trabalha em seguida e com qual instrução. Todos leem e escrevem num quadro compartilhado. Ao terminar, a pessoa recebe uma resposta única e coerente.

**Why this priority**: é o núcleo da feature. Separar papéis reduz erros típicos de um agente único, como agir antes de investigar ou misturar fatos com suposições. Também deixa claro quem fez o quê.

**Independent Test**: dá para testar mandando um pedido de investigação e ação com a rota de equipe forçada e conferindo que o analista trabalha antes do planejador, que o planejador trabalha antes do executor, e que a resposta final cita os fatos levantados e o que foi feito ou proposto.

**Acceptance Scenarios**:

1. **Given** um pedido que exige investigar e agir, **When** a equipe trabalha, **Then** o supervisor passa a vez para pelo menos o analista e, quando há ação, para o planejador e o executor, e a resposta final resume fatos, plano e resultado.
2. **Given** um pedido só de consulta ("quais alertas estão disparando no checkout?"), **When** a equipe trabalha, **Then** o supervisor pode encerrar logo depois do analista, sem acionar planejador nem executor.
3. **Given** qualquer papel em ação, **When** ele trabalha, **Then** ele vê a instrução do supervisor e tudo o que os papéis anteriores registraram no quadro compartilhado.
4. **Given** a equipe trabalhando, **When** o supervisor decide encerrar, **Then** a pessoa recebe uma resposta final baseada no que está no quadro compartilhado, sem inventar fatos que não estão lá.

---

### User Story 2 - Cada papel só faz o que lhe cabe (Priority: P1)

As responsabilidades são garantidas pelo sistema, não só pedidas no texto:
- o analista só consegue consultar e, ao registrar no quadro, não pode registrar propostas de ação;
- o planejador não tem acesso a nenhuma ferramenta;
- o executor só tem as ações sobre incidentes, e toda ação dele passa pela aprovação humana existente. Ele não tem caminho para executar sem aprovação.

**Why this priority**: o OpsPilot mexe em incidentes de produção (Princípio VI). Se os papéis forem só sugestão, um analista ou planejador "criativo" poderia agir. Garantir os limites por construção é o que torna o modo equipe seguro.

**Independent Test**: dá para testar inspecionando, por papel, quais ferramentas cada um recebe (analista só leitura, planejador nenhuma, executor só incidentes) e simulando um executor que tenta resolver um incidente: o resultado precisa ser o cartão de aprovação, com o incidente ainda aberto.

**Acceptance Scenarios**:

1. **Given** o analista em ação, **When** ele tenta usar uma ação que muda a produção, **Then** essa ação não está disponível para ele.
2. **Given** o analista registra algo no quadro, **When** o registro é gravado, **Then** ele fica no formato de fatos (o que foi observado e de onde veio), sem campo para propostas nem recomendações.
3. **Given** o planejador em ação, **When** ele trabalha, **Then** não tem nenhuma ferramenta disponível e produz um plano a partir do quadro.
4. **Given** o executor propõe abrir ou resolver um incidente, **When** a proposta acontece, **Then** a execução para e a resposta é "aguardando aprovação", com o mesmo cartão Aprovar/Negar de hoje. Nada é executado antes da decisão humana.
5. **Given** o executor em ação, **When** ele tenta consultar alertas ou runbooks, **Then** essas ferramentas não estão disponíveis para ele (consulta é papel do analista).

---

### User Story 3 - Ver quem passou a vez para quem no "ver raciocínio" (Priority: P2)

No painel "ver raciocínio" da war room, cada passagem de vez aparece como um evento próprio, que mostra quem passou para quem e com qual instrução. Os eventos de cada papel (pensamentos, consultas, plano, proposta) aparecem identificados pelo papel que os produziu. Assim, a pessoa acompanha a conversa interna da equipe na ordem em que aconteceu.

**Why this priority**: sem isso, o trace da equipe vira uma lista longa sem dono, e a pessoa não consegue avaliar se a equipe investigou antes de agir. Depende das US1 e US2.

**Independent Test**: dá para testar abrindo "ver raciocínio" numa resposta da equipe e conferindo que há um evento de passagem antes de cada turno de papel, cada um com destino e instrução, e que os demais eventos mostram o papel de origem.

**Acceptance Scenarios**:

1. **Given** uma resposta produzida pela equipe, **When** a pessoa abre "ver raciocínio", **Then** cada passagem de vez aparece como um evento "Passagem" com origem, destino e instrução, e com visual próprio, distinto dos outros tipos.
2. **Given** o mesmo painel, **When** a pessoa lê os eventos de um turno, **Then** cada evento mostra qual papel o produziu (analista, planejador, executor ou supervisor).
3. **Given** a decisão final do supervisor, **When** o painel é exibido, **Then** o encerramento também aparece como passagem, para "fim".
4. **Given** um registro gravado de uma execução da equipe, **When** ele é consultado depois pelo identificador da requisição, **Then** as passagens e os papéis aparecem iguais aos da resposta original.

---

### User Story 4 - A equipe como rota escolhível (Priority: P2)

O modo equipe é uma rota a mais do copiloto. O roteador pode escolhê-lo sozinho para pedidos que pedem investigação e ação coordenadas, e o cliente pode forçá-lo pelo nome. As demais rotas continuam iguais.

**Why this priority**: sem rota, o modo equipe não é acessível. Fica atrás das demais histórias porque o valor está nelas.

**Independent Test**: dá para testar forçando a rota da equipe pelo nome e conferindo que a rota registrada é a da equipe, e mandando um pedido típico de investigação e ação sem forçar nada, para ver se o roteador a escolhe.

**Acceptance Scenarios**:

1. **Given** um pedido com a rota da equipe forçada pelo nome, **When** a resposta chega, **Then** a rota registrada é a da equipe, com a origem "escolhida pelo cliente".
2. **Given** um pedido do tipo "investigue e aja", **When** o roteador decide, **Then** a equipe é uma das rotas que ele pode escolher, com motivo.
3. **Given** um nome de rota desconhecido, **When** ele é enviado, **Then** a resposta continua sendo "estratégia desconhecida", como hoje.

---

### Edge Cases

- O supervisor fica alternando entre papéis sem convergir: existe um teto de passagens. Ao atingi-lo, a equipe encerra de forma controlada, com uma resposta que diz isso e resume o que estiver no quadro.
- O supervisor devolve uma decisão inválida (papel inexistente, sem instrução, formato quebrado): a equipe encerra de forma controlada, com a resposta baseada no quadro, e o evento registra o motivo.
- O supervisor manda o executor agir sem plano no quadro: o executor recebe a instrução e o quadro como está. A aprovação humana continua sendo a guarda final, e o cartão mostra a ação proposta.
- O executor propõe uma segunda ação na mesma execução: vale a regra de hoje, só uma ação pendente por vez. A segunda é recusada e o executor é avisado.
- A equipe passa da aprovação (resposta "aguardando aprovação"): a decisão segue exatamente o fluxo atual. Aprovar executa a ação guardada, sem retomar a equipe.
- O mesmo papel é chamado mais de uma vez (ex.: analista de novo depois do plano): é permitido, e cada turno aparece com sua própria passagem.
- A rota da equipe estoura o tempo máximo de resposta: vale o mesmo tempo esgotado das outras rotas.
- O analista tenta escrever uma recomendação no quadro: o registro é rejeitado ou reduzido aos fatos, e o texto de recomendação não chega ao planejador.

## Requirements *(mandatory)*

### Functional Requirements

**Equipe e coordenação**

- **FR-001**: O copiloto MUST ter um modo equipe, composto por um supervisor e três papéis: analista, planejador e executor.
- **FR-002**: A cada passo, o supervisor MUST escolher, de forma estruturada, o próximo papel (ou o encerramento) e uma instrução curta para ele.
- **FR-003**: A equipe MUST manter um quadro compartilhado durante a execução, com o pedido original, os fatos do analista, o plano do planejador e os resultados ou propostas do executor. O supervisor e todos os papéis MUST ler o quadro antes de agir.
- **FR-004**: Ao encerrar, a equipe MUST produzir uma única resposta final para a pessoa, baseada no conteúdo do quadro.
- **FR-005**: A equipe MUST ter um teto de passagens por execução e encerrar de forma controlada ao atingi-lo. O mesmo vale para uma decisão inválida do supervisor.

**Limites dos papéis (garantidos pelo sistema)**

- **FR-006**: O analista MUST ter acesso apenas às consultas (alertas, incidentes, runbooks) e MUST NOT ter acesso a ações que mudam a produção.
- **FR-007**: O que o analista registra no quadro MUST ser só fatos com origem. O formato do registro MUST NOT ter campo para propostas, planos ou recomendações.
- **FR-008**: O planejador MUST NOT ter acesso a nenhuma ferramenta.
- **FR-009**: O executor MUST ter acesso apenas às ações sobre incidentes (abrir e resolver), e MUST NOT ter acesso às consultas.
- **FR-010**: Toda ação do executor MUST passar pela aprovação humana existente (resposta "aguardando aprovação" e decisão posterior). O modo equipe MUST NOT oferecer caminho alternativo de execução.

**Rastreabilidade**

- **FR-011**: Cada passagem de vez MUST gerar um evento de passagem no trace, com origem, destino (papel ou fim) e instrução.
- **FR-012**: Cada evento produzido dentro da equipe MUST indicar o papel que o produziu.
- **FR-013**: Os eventos de passagem e a identificação de papel MUST ser gravados e recuperados junto com o resto do trace, iguais aos da resposta original.
- **FR-014**: O "ver raciocínio" da war room MUST exibir o evento de passagem com rótulo, ícone e formato próprios (origem → destino, mais a instrução), e MUST mostrar o papel de cada evento da equipe. O rótulo não pode depender só de cor.
- **FR-015**: Os logs MUST registrar as passagens só com metadados (origem, destino, posição), nunca a instrução.

**Rota**

- **FR-016**: O modo equipe MUST ser uma rota do copiloto, escolhível pelo roteador (com motivo) e forçável pelo cliente pelo nome.
- **FR-017**: As rotas existentes, os registros, as métricas, a aprovação humana e o formato das respostas MUST continuar funcionando sem mudança para quem não usa a rota da equipe.
- **FR-018**: As métricas da execução da equipe MUST somar as chamadas ao modelo de todos os papéis e do supervisor.

### Key Entities

- **Supervisor**: coordena a equipe. A cada passo, lê o quadro e decide o próximo papel (ou o fim) e a instrução.
- **Papel**: analista, planejador ou executor. Cada um tem ferramentas permitidas fixas e um tipo de contribuição ao quadro (fatos, plano, resultado ou proposta).
- **Quadro compartilhado**: estado da execução visível a todos. Tem o pedido, a lista de fatos (com origem), o plano (passos), os resultados e propostas do executor, e o histórico de passagens.
- **Passagem**: decisão do supervisor. Tem origem, destino (papel ou fim), instrução e posição na execução.
- **Evento de passagem**: a passagem registrada no trace, com a etapa da equipe.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Em 100% das execuções da equipe que terminam em ação, o analista trabalhou antes do executor.
- **SC-002**: 0 ações sobre incidentes executadas pela equipe sem decisão humana registrada.
- **SC-003**: 0 ferramentas fora da lista permitida disponíveis para cada papel, verificado para os três papéis.
- **SC-004**: Toda execução da equipe termina, por encerramento do supervisor, teto de passagens ou decisão inválida, sem ficar em loop.
- **SC-005**: A partir de "ver raciocínio", a pessoa identifica, em até 1 minuto, a sequência de papéis acionados e a instrução de cada passagem.
- **SC-006**: As rotas existentes mantêm 100% dos testes atuais passando sem alteração de comportamento.

## Assumptions

- **Termos pedidos pela pessoa usuária**: o código vive em `src/team`. O supervisor decide com saída estruturada `{ next, brief }` (`withStructuredOutput`). O quadro compartilhado ("blackboard") faz parte do estado do grafo. O evento chama `handoff` e a rota chama `team`. O detalhamento técnico fica no `/speckit.plan`.
- **Resposta final**: produzida pelo supervisor ao encerrar, a partir do quadro. Nenhum papel extra de "redator".
- **Teto de passagens**: 6 por execução (padrão), suficiente para analista → planejador → executor com uma volta extra.
- **Ferramentas do analista**: listar alertas, listar incidentes e consultar runbook. "Não propõe" significa que o registro do analista no quadro só aceita fatos, e o texto dele não chega como proposta ao planejador.
- **Executor**: só abrir e resolver incidente, sempre pela porta de aprovação da spec 015. Como a porta só aceita uma ação pendente por execução, a equipe propõe no máximo uma ação por pedido.
- **Aprovação**: a decisão (aprovar ou negar) segue o fluxo atual e não retoma a equipe depois.
- **Memória do usuário** (ferramentas de lembrar e esquecer, spec 007): fica fora dos papéis da equipe nesta versão.
- **Roteador**: a tabela de rotas ganha a linha "team" para pedidos de investigar e agir de forma coordenada. Na dúvida, o roteador continua preferindo a rota mais barata.
- **Fora do escopo**: papéis configuráveis pelo usuário, execução paralela de papéis, retomar a equipe após a aprovação e uma interface própria para a equipe (além do "ver raciocínio").
- **Dependências**: grafo de produção com roteador (012), trace persistido e logs (014), aprovação humana e war room (015).
