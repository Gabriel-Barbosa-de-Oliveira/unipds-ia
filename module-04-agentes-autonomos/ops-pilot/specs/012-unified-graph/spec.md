# Feature Specification: Grafo Unificado com Roteador de Estratégia

**Feature Branch**: `012-unified-graph`

**Created**: 2026-10-09

**Status**: Draft

**Input**: User description: "Grafo unificado: production-graph.ts: nós contexto, roteador, as 3 estratégias como nós e resposta. Roteador: withStructuredOutput (route, reason); tabela no prompt; evento \"route\" e campo node em todo evento de trace. /chat: strategy opcional (se vier, é override no trace)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - O copiloto escolhe sozinho a estratégia certa para cada pergunta (Priority: P1)

A pessoa de plantão manda uma mensagem ao copiloto sem dizer qual estratégia de raciocínio usar. O copiloto monta o contexto, decide qual das três estratégias disponíveis (react, plan-and-execute, reflection) é mais adequada àquela pergunta, executa essa estratégia e devolve a resposta — tudo num único fluxo de produção, com a decisão e o motivo registrados no trace.

**Why this priority**: é o núcleo da feature. Hoje a pessoa precisa conhecer as estratégias e escolher uma (ou cair sempre no padrão react), o que gera respostas lentas/caras para perguntas simples ou rasas para pedidos de várias etapas. Um fluxo único que escolhe a rota sozinho elimina essa decisão do usuário.

**Independent Test**: pode ser testado isoladamente enviando ao copiloto, sem indicar estratégia, perguntas típicas de cada perfil (consulta simples, pedido de várias etapas, pedido onde a precisão é crítica) com um roteador simulado/determinístico, e confirmando que a estratégia executada é a que o roteador escolheu e que a resposta é retornada normalmente.

**Acceptance Scenarios**:

1. **Given** uma mensagem sem estratégia indicada, **When** o copiloto responde, **Then** o fluxo passa, nesta ordem, por montagem de contexto → roteamento → exatamente uma estratégia → resposta.
2. **Given** que o roteador escolhe a rota "plan-and-execute" para uma mensagem, **When** o copiloto responde, **Then** a resposta é produzida pela estratégia plan-and-execute e nenhuma outra estratégia é executada.
3. **Given** qualquer uma das três rotas escolhida, **When** a resposta é devolvida, **Then** o formato da resposta (resposta, trace, métricas, identificador de conversa) é o mesmo que já existe hoje.
4. **Given** a montagem de contexto já existente (histórico, memórias, orçamento por seção), **When** o fluxo unificado é usado, **Then** o contexto entregue à estratégia escolhida é o mesmo que seria entregue hoje para a mesma entrada.

---

### User Story 2 - Entender por que o copiloto seguiu aquela rota e em que etapa cada passo aconteceu (Priority: P1)

Quem opera ou depura o copiloto quer ler o trace de uma resposta e saber (a) qual rota foi escolhida, (b) por quê, e (c) em qual etapa do fluxo cada evento do trace foi produzido (contexto, roteador, qual estratégia, resposta).

**Why this priority**: sem isso, a escolha automática vira uma caixa-preta — impossível auditar decisões ruins do roteador ou comparar rotas. É condição para confiar no roteamento automático em produção.

**Independent Test**: pode ser testado isoladamente executando o fluxo com um roteador e estratégias simulados e inspecionando o trace retornado: deve conter exatamente um evento de rota com rota e motivo, e todo evento deve indicar a etapa que o produziu.

**Acceptance Scenarios**:

1. **Given** uma resposta produzida pelo fluxo unificado, **When** o trace é inspecionado, **Then** há exatamente um evento do tipo "route" contendo a rota escolhida e o motivo (texto não vazio).
2. **Given** uma resposta produzida pelo fluxo unificado, **When** o trace é inspecionado, **Then** todo evento do trace — inclusive os pensamentos, ações, observações, planos, críticas e a resposta final vindos das estratégias — informa a etapa (nó) do fluxo que o gerou.
3. **Given** um trace, **When** os eventos são lidos em ordem, **Then** o evento "route" aparece antes de qualquer evento produzido pela estratégia escolhida.
4. **Given** a formatação legível de trace já existente (usada no terminal/arena), **When** um trace com evento "route" é formatado, **Then** a rota e o motivo aparecem numa linha legível.

---

### User Story 3 - Forçar uma estratégia específica quando necessário (Priority: P2)

Quem usa o endpoint de chat (ex.: para testes, comparação ou porque sabe o que quer) pode continuar informando uma estratégia. Quando informa, essa estratégia é usada no lugar da escolha do roteador, e o trace deixa claro que a rota veio de um override, não de uma decisão automática.

**Why this priority**: preserva compatibilidade com quem já usa o campo de estratégia e mantém um caminho de controle manual, mas o valor principal da feature (roteamento automático) não depende disso.

**Independent Test**: pode ser testado isoladamente chamando o chat com uma estratégia explícita e um roteador simulado que escolheria outra rota, e confirmando que a estratégia explícita foi executada, que o roteador não foi consultado e que o evento "route" marca a origem como override.

**Acceptance Scenarios**:

1. **Given** uma requisição de chat com estratégia "react" informada, **When** o copiloto responde, **Then** a estratégia react é executada independentemente do que o roteador escolheria.
2. **Given** uma requisição com estratégia informada, **When** o trace é inspecionado, **Then** o evento "route" indica a rota informada e que sua origem foi override (e não decisão do roteador).
3. **Given** uma requisição com estratégia informada, **When** o copiloto responde, **Then** o roteador não é consultado (nenhuma chamada extra ao modelo é gasta com roteamento).
4. **Given** uma requisição com uma estratégia inexistente, **When** o copiloto recebe a requisição, **Then** a resposta é o mesmo erro de estratégia desconhecida que já existe hoje.
5. **Given** uma requisição sem estratégia informada, **When** o copiloto responde, **Then** a rota é decidida pelo roteador (e não mais por um padrão fixo).

---

### Edge Cases

- Roteador devolve uma rota fora das três conhecidas ou uma resposta malformada: o fluxo não falha; cai na rota padrão (react) e registra no evento "route" que houve fallback, com motivo explicando a falha.
- Roteador falha (erro de rede/modelo) ou excede tempo: mesmo comportamento de fallback para a rota padrão, com motivo indicando a falha; o teto de tempo total da requisição continua valendo.
- Roteador devolve motivo vazio: tratado como resposta malformada (fallback).
- Override com `reflect` também informado: mantém-se o comportamento atual de combinar reflection sobre a estratégia base indicada; o evento "route" registra a rota efetiva.
- A estratégia escolhida falha no meio da execução: o erro se propaga como hoje (sem tentar outra rota automaticamente).
- Mensagem ambígua que caberia em mais de uma rota: o roteador escolhe uma só; o motivo explica a escolha.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema MUST oferecer um único fluxo de produção composto pelas etapas: contexto → roteador → uma das três estratégias (react, plan-and-execute, reflection) → resposta.
- **FR-002**: Cada execução MUST percorrer exatamente uma estratégia; nenhuma estratégia não escolhida pode ser executada.
- **FR-003**: O roteador MUST produzir uma decisão estruturada com dois campos: rota (uma das três estratégias) e motivo (texto não vazio).
- **FR-004**: As instruções do roteador MUST conter uma tabela que descreve, para cada rota, quando usá-la (perfil de pergunta adequado) — para que a decisão seja consistente e auditável.
- **FR-005**: O trace de toda execução do fluxo unificado MUST conter exatamente um evento do tipo "route" com a rota efetiva, o motivo e a origem da decisão (roteador, override ou fallback).
- **FR-006**: Todo evento de trace produzido no fluxo unificado MUST indicar a etapa (nó) que o gerou.
- **FR-007**: O evento "route" MUST aparecer no trace antes de qualquer evento da estratégia executada.
- **FR-008**: O campo de estratégia do chat MUST continuar opcional; quando informado e válido, MUST ser usado como rota (override), sem consultar o roteador.
- **FR-009**: Quando a estratégia não é informada, a rota MUST ser decidida pelo roteador (substituindo o padrão fixo atual).
- **FR-010**: Uma estratégia informada que não existe MUST continuar resultando no erro de estratégia desconhecida já existente.
- **FR-011**: Se o roteador falhar ou devolver decisão inválida, o sistema MUST usar a rota padrão (react) e registrar a origem como fallback, com motivo descritivo.
- **FR-012**: A etapa de contexto MUST reutilizar a montagem de contexto já existente (histórico, memórias, orçamento por seção), sem mudar seu resultado para uma mesma entrada.
- **FR-013**: O formato da resposta do chat (resposta, trace, métricas, identificador de conversa e métricas de contexto) MUST permanecer compatível; as métricas MUST incluir as chamadas ao modelo e tokens gastos pelo roteador quando ele for consultado.
- **FR-014**: A formatação legível de trace MUST suportar o evento "route" e exibir a etapa de cada evento.
- **FR-015**: A lógica de decisão que não depende do modelo (validação da rota, aplicação de override/fallback, marcação de etapa nos eventos) MUST ser coberta por testes que não acessam rede.

### Key Entities

- **Decisão de rota**: resultado do roteamento — rota (react | plan-and-execute | reflection), motivo (texto) e origem (roteador | override | fallback).
- **Evento de trace**: evento já existente (pensamento, ação, observação, plano, crítica, resposta) acrescido do novo tipo "route" e de um campo que identifica a etapa (nó) que o produziu.
- **Etapa (nó) do fluxo**: contexto, roteador, react, plan-and-execute, reflection, resposta.
- **Tabela de rotas**: descrição, por rota, do perfil de pergunta para o qual ela é indicada; faz parte das instruções do roteador.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% das respostas do chat produzidas sem estratégia informada trazem no trace exatamente um evento "route" com rota válida e motivo não vazio.
- **SC-002**: 100% dos eventos de trace de qualquer resposta identificam a etapa que os produziu.
- **SC-003**: 100% das requisições com estratégia válida informada executam essa estratégia, sem nenhuma chamada de roteamento ao modelo, e marcam a origem como override.
- **SC-004**: Numa amostra de perguntas de referência com rota esperada conhecida (ex.: o dataset do arena/bench), o roteador acerta a rota esperada em pelo menos 80% dos casos.
- **SC-005**: Falhas do roteador nunca resultam em erro para quem pergunta: 100% delas terminam numa resposta via rota padrão com origem fallback registrada.
- **SC-006**: O custo adicional do roteamento automático é de no máximo uma chamada extra ao modelo por resposta.
- **SC-007**: Clientes atuais do chat continuam funcionando sem alteração (mesmos campos de requisição e de resposta, apenas acrescidos).

## Assumptions

- As "3 estratégias" são react, plan-and-execute e reflection; a rota "reflection" corresponde à crítica/reflexão aplicada sobre a estratégia base padrão (react), como já é feito hoje com a opção de reflexão.
- A rota padrão para fallback é react, mantendo o padrão atual do sistema.
- No override, o roteador não é consultado — a origem "override" no evento "route" e um motivo fixo indicando que a rota foi informada pelo cliente bastam para auditoria.
- A opção de reflexão já existente no chat continua aceita e se combina com a rota efetiva como hoje; não faz parte do roteamento automático.
- O teto de tempo total por requisição já existente cobre o fluxo inteiro, incluindo o roteador.
- Os pontos de entrada de comparação (arena/bench) continuam podendo executar estratégias diretamente; adotá-los no fluxo unificado não é requisito desta feature.
- O roteador usa o mesmo provedor/modelo já configurado para as estratégias.
