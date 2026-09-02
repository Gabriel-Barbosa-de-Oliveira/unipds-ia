# Feature Specification: Núcleo de Raciocínio (Reasoning Strategies Core)

**Feature Branch**: `001-reasoning-strategies-core`

**Created**: 2026-09-01

**Status**: Draft

**Input**: User description: "Núcle de raciocinio do OpsPilot: Interface comum ReasoningStrategy (name + run(input) -> answer, trace, metrics; trace com eventos tipados thought/action/observation/plan/critique/answer; metrics com chamadas de LLM e latência); fábrica única de modelo (OpenRouter, temperature 0); ferramentas mock sobre store in-memory pré-populado (5 serviços, 6 alertas: 3 firing, 3 resolved, com script de seed); tools list_alerts/open_incident/resolve_incident; estratégia ReAct; estratégia Plan-and-Execute (planner/executor/replanner, máx. 8 passos); limite de iterações e contagem de chamadas de LLM em todas as estratégias; arena mínima que roda 1+ estratégias sobre o mesmo input e imprime traces e métricas (flags --strategies e --max-iterations); testes determinísticos e sem rede para o store e a formatação de trace."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Obter uma resposta auditável para uma pergunta operacional (Priority: P1)

Um operador de plantão (ou alguém validando o copiloto) faz uma pergunta operacional em linguagem natural sobre o estado atual dos alertas e incidentes (ex.: "quais alertas estão disparando agora?" ou "abra um incidente de severidade alta para o serviço X"). O núcleo de raciocínio processa a pergunta, decide quais ações operacionais executar, e devolve uma resposta final acompanhada do passo a passo completo (raciocínio, ações tomadas, resultados observados) e de métricas de execução.

**Why this priority**: Sem essa capacidade básica — responder com resposta + trace auditável + métricas — não existe copiloto. É o valor central da feature e a base sobre a qual tudo mais (comparação de estratégias, guardrails) se apoia.

**Independent Test**: Pode ser testado isoladamente enviando uma pergunta operacional a uma única estratégia de raciocínio contra o conjunto de dados semeado e verificando que a resposta final está correta, que o trace contém a sequência completa de passos até a resposta, e que as métricas (nº de chamadas ao modelo, latência) são reportadas.

**Acceptance Scenarios**:

1. **Given** o conjunto de dados semeado (5 serviços, 6 alertas: 3 firing, 3 resolved), **When** o operador pergunta quais alertas estão disparando, **Then** a resposta final lista exatamente os 3 alertas em estado "firing", e o trace mostra cada passo (raciocínio, ação executada com seus argumentos, observação retornada) que levou a essa resposta.
2. **Given** o conjunto de dados semeado, **When** o operador pede para abrir um incidente para um serviço existente com uma severidade válida, **Then** o sistema executa a ação de abertura, o trace registra a ação e sua observação (incidente criado), e a resposta final confirma a criação com os dados do incidente.
3. **Given** qualquer execução concluída, **When** o resultado é retornado, **Then** ele inclui métricas com, no mínimo, o número de chamadas feitas ao modelo de raciocínio e o tempo total de execução.

---

### User Story 2 - Comparar estratégias de raciocínio lado a lado (Priority: P2)

Um desenvolvedor quer decidir qual abordagem de raciocínio (agir-e-observar passo a passo, ou planejar-executar-replanejar) funciona melhor para o OpsPilot. Ele roda a mesma pergunta operacional contra duas ou mais estratégias de uma só vez e compara, lado a lado, o trace completo e as métricas de cada uma.

**Why this priority**: Depende da User Story 1 já existir (uma estratégia funcionando com trace e métricas), mas é o que dá valor de decisão: sem comparação, não há como saber qual estratégia é mais confiável, rápida ou econômica antes de colocar uma delas em produção.

**Independent Test**: Pode ser testado isoladamente escolhendo duas estratégias e uma pergunta operacional, executando a comparação, e verificando que a saída apresenta trace e métricas de cada estratégia individualmente, sem que a execução de uma interfira na outra.

**Acceptance Scenarios**:

1. **Given** duas ou mais estratégias de raciocínio disponíveis, **When** o desenvolvedor solicita a execução comparativa com uma mesma pergunta operacional, **Then** o sistema executa a pergunta em cada estratégia selecionada e apresenta o trace e as métricas de cada execução de forma identificável (qual trace pertence a qual estratégia).
2. **Given** uma execução comparativa em andamento, **When** o desenvolvedor especifica um limite de passos/iterações diferente do padrão, **Then** todas as estratégias selecionadas respeitam esse limite durante a execução.
3. **Given** uma execução comparativa concluída, **When** o desenvolvedor revisa a saída, **Then** consegue identificar, sem re-executar nada, qual estratégia usou menos chamadas ao modelo e qual foi mais rápida.

---

### User Story 3 - Ações operacionais controladas e à prova de falha (Priority: P3)

Um operador confia que qualquer ação tomada pelo copiloto (listar alertas, abrir incidente, resolver incidente) passa por um conjunto fixo e validado de ações — nunca uma ação livre e não auditável — e que entradas inválidas (ex.: id de incidente inexistente, severidade mal formada) resultam em um erro claro registrado no trace, não em uma falha silenciosa ou trava do sistema.

**Why this priority**: É uma camada de segurança e confiabilidade sobre as User Stories 1 e 2. Importante para produção, mas o sistema já entrega valor demonstrável sem ela (em um ambiente de teste/demo controlado), por isso vem depois.

**Independent Test**: Pode ser testado isoladamente chamando cada ação operacional com entradas válidas e inválidas (sem envolver nenhuma estratégia de raciocínio) e verificando que toda entrada inválida produz um erro estruturado e nenhuma entrada trava a execução.

**Acceptance Scenarios**:

1. **Given** o conjunto de dados semeado, **When** uma estratégia tenta resolver um incidente com um id que não existe, **Then** a ação retorna um erro claro, esse erro é registrado como observação no trace, e a estratégia consegue prosseguir ou finalizar de forma controlada em vez de travar.
2. **Given** o conjunto de dados semeado, **When** uma estratégia tenta abrir um incidente com severidade ou serviço inválidos, **Then** a ação rejeita a entrada com um erro claro antes de qualquer efeito colateral no conjunto de dados.
3. **Given** uma estratégia em execução, **When** o limite máximo de passos/iterações é atingido sem uma resposta final, **Then** a execução para de forma controlada e retorna o trace parcial e uma indicação de que o limite foi atingido, em vez de continuar indefinidamente.

---

### Edge Cases

- O que acontece quando a pergunta operacional não corresponde a nenhuma ação disponível (ex.: pergunta fora do domínio de alertas/incidentes)? O sistema deve responder de forma clara que não consegue executar a solicitação, sem inventar uma ação.
- O que acontece quando nenhum alerta corresponde ao filtro solicitado (ex.: perguntar por alertas "firing" quando todos estão resolvidos)? Deve ser tratado como resultado válido (lista vazia), não como erro.
- O que acontece quando o modelo de raciocínio subjacente está indisponível ou mal configurado? A execução deve falhar rapidamente com um erro claro, em vez de ficar pendurada esperando resposta.
- O que acontece quando a estratégia Plan-and-Execute atinge o limite de 8 passos sem esgotar o plano? A execução para de forma controlada, mesmo que o plano ainda tenha passos restantes.
- O que acontece quando o conjunto de dados precisa ser reiniciado ao estado inicial entre execuções (ex.: para comparar estratégias em igualdade de condições)? Deve existir uma forma de restaurar o conjunto de dados semeado (5 serviços, 6 alertas: 3 firing, 3 resolved) de forma reprodutível.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema DEVE oferecer uma forma comum de invocar qualquer estratégia de raciocínio com uma pergunta/instrução operacional e receber de volta uma resposta final, um trace completo dos passos intermediários e métricas de execução.
- **FR-002**: O trace de uma execução DEVE registrar cada passo como um de um conjunto fixo de tipos de evento: raciocínio/pensamento, ação executada (com o nome da ação e seus argumentos), observação (resultado de uma ação), atualização de plano, crítica/revisão, e resposta final.
- **FR-003**: As métricas de uma execução DEVEM incluir, no mínimo, o número de chamadas feitas ao modelo de raciocínio subjacente e o tempo total decorrido para produzir a resposta.
- **FR-004**: O sistema DEVE oferecer pelo menos duas estratégias de raciocínio distintas: uma que alterna raciocínio e ação passo a passo, e outra que primeiro produz um plano de passos, executa um passo por vez, e revisa o plano restante após cada passo.
- **FR-005**: A estratégia de planejar-executar-revisar DEVE encerrar quando não restarem passos no plano ou ao atingir um máximo de 8 passos executados, o que ocorrer primeiro.
- **FR-006**: Toda estratégia de raciocínio DEVE respeitar um limite configurável de iterações/passos e, ao atingi-lo sem uma resposta final, encerrar de forma controlada retornando a melhor resposta disponível junto com o trace parcial, em vez de continuar indefinidamente.
- **FR-007**: O sistema DEVE expor um conjunto fixo de ações operacionais controladas que as estratégias de raciocínio podem usar: listar alertas (com filtro opcional por status), abrir um novo incidente para um serviço, e resolver um incidente existente por id.
- **FR-008**: Toda entrada para as ações operacionais (ex.: filtro de status de alerta, título/serviço/severidade de incidente, id de incidente) DEVE ser validada antes da execução; entrada inválida DEVE produzir um erro claro e estruturado em vez de falha silenciosa ou travamento.
- **FR-009**: As ações operacionais DEVEM operar sobre um conjunto de dados inicial que inclua múltiplos serviços e múltiplos alertas em estado ativo ("firing") e resolvido, para que as estratégias possam ser exercitadas contra cenários mistos realistas.
- **FR-010**: O sistema DEVE oferecer uma forma de semear/restaurar o conjunto de dados operacional para esse estado inicial conhecido sob demanda, independente da execução de qualquer estratégia de raciocínio.
- **FR-011**: O sistema DEVE oferecer um modo de comparação ("arena") que executa a mesma pergunta/instrução operacional em uma ou mais estratégias de raciocínio selecionadas e apresenta o trace completo e as métricas de cada estratégia lado a lado.
- **FR-012**: O modo de comparação DEVE permitir que quem o invoca escolha quais estratégias executar e sobrescreva o limite máximo de iterações/passos para aquela execução.
- **FR-013**: Para toda ação tomada por uma estratégia de raciocínio, o sistema DEVE registrar tanto os argumentos da ação quanto a observação resultante, de modo que o caminho de decisão completo seja auditável após o fato.
- **FR-014**: A lógica do conjunto de dados operacional (armazenamento e comportamento das ações) e a formatação do trace DEVEM ser verificáveis automaticamente sem depender de acesso à rede ou de um modelo de raciocínio real.

### Key Entities

- **Estratégia de Raciocínio**: uma abordagem de raciocínio identificada por nome; recebe uma pergunta/instrução operacional e produz uma resposta final, um trace e métricas.
- **Evento de Trace**: um passo dentro da execução de uma estratégia; possui um tipo (raciocínio, ação, observação, plano, crítica, resposta) e detalhes específicos do tipo (uma ação carrega o nome da ação e seus argumentos).
- **Métricas de Execução**: resumo quantitativo de uma execução (número de chamadas ao modelo, latência total).
- **Serviço**: um serviço operacional monitorado, identificado por nome.
- **Alerta**: um sinal de monitoramento associado a um serviço, com um status (disparando/resolvido).
- **Incidente**: um registro aberto por um operador (ou pela estratégia em seu nome) associado a um serviço, com título, severidade e status (aberto/resolvido).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Dado o conjunto de dados semeado, uma pergunta sobre "quais alertas estão disparando" é respondida corretamente com os 3 alertas certos em 100% das execuções.
- **SC-002**: Para qualquer execução concluída, é possível revisar o passo a passo completo entre a pergunta e a resposta final sem nenhuma lacuna — cada ação executada tem uma observação correspondente registrada.
- **SC-003**: 100% das execuções, em ambas as estratégias, terminam (com resposta final ou parada controlada por limite) dentro do limite de passos configurado — nenhuma execução roda de forma ilimitada.
- **SC-004**: Um desenvolvedor consegue comparar 2 ou mais estratégias sobre a mesma pergunta operacional em uma única execução, vendo o trace e as métricas de cada uma sem precisar repetir a pergunta manualmente.
- **SC-005**: 100% das entradas inválidas para ações operacionais (ex.: id de incidente inexistente, severidade mal formada) resultam em um erro claro registrado no trace, sem travamentos ou falhas não tratadas.
- **SC-006**: O conjunto de dados pode ser restaurado ao seu estado inicial conhecido (5 serviços, 6 alertas: 3 firing, 3 resolved) em um único passo, de forma reprodutível.

## Assumptions

- A "pergunta/instrução operacional" recebida por uma estratégia de raciocínio é texto em linguagem natural (ex.: "quais alertas estão firing?", "abra um incidente para o serviço X com severidade alta"); esta especificação não prescreve a superfície de invocação (CLI, chamada interna, etc.) usada para chegar até o núcleo de raciocínio.
- As três ações operacionais controladas (listar alertas, abrir incidente, resolver incidente) são o conjunto completo em escopo para esta feature; novas ações ficam fora de escopo.
- O conjunto de dados inicial (5 serviços, 6 alertas: 3 firing, 3 resolved) é um conjunto de demonstração/teste, não dados reais de produção.
- "Chamadas ao modelo" nas métricas conta chamadas ao modelo de raciocínio subjacente, independente de qual estratégia está em uso.
- As estratégias de raciocínio operam sobre o mesmo conjunto de dados semeável; isolamento entre execuções simultâneas não é um requisito desta feature (uma execução por vez é aceitável nesta fase).
- Autenticação/autorização sobre quem pode disparar ações de incidente está fora de escopo desta feature (assume-se que é tratada por uma camada de API já existente ou futura).
- O modo de comparação ("arena") é uma ferramenta para desenvolvedores/operadores avaliarem estratégias, não uma funcionalidade voltada ao usuário final de produção.
- Severidades de incidente seguem uma escala padrão do domínio de operações (ex.: baixa/média/alta/crítica); o valor exato da escala pode ser refinado na fase de planejamento.
- Resultados de execuções (trace e métricas) são retornados/exibidos por execução; persistência histórica de execuções passadas não é um requisito desta feature.
