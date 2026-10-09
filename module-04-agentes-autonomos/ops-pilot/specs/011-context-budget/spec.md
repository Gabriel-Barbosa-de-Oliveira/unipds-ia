# Feature Specification: Orçamento de Contexto por Seção

**Feature Branch**: `010-context-budget`

**Created**: 2026-10-09

**Status**: Draft

**Input**: User description: "ContextBuilder com orçamento por seção: src/context/context-builder.ts monta o prompt de TODAS as estratégias com teto por seção via env CONTEXT_BUDGET_*: system e mensagem intocáveis, resumo 200, janela 1200 (corta as mais antigas), memórias 300 (corta menor score). Teste: tetos baixos cortam na ordem certa"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Contexto enviado ao modelo nunca estoura um teto previsível por seção (Priority: P1)

A pessoa que opera o copiloto quer que o contexto enviado ao modelo em cada resposta tenha um tamanho limitado e previsível, seção por seção (resumo da conversa, janela de mensagens recentes, memórias lembradas), para que conversas longas ou pessoas com muitos fatos lembrados não façam o custo e a latência crescerem sem controle — independentemente de qual estratégia de raciocínio estiver em uso.

**Why this priority**: é o núcleo da feature. A medição de contexto (`009-context-tokens`) mostra de onde vem o volume; esta feature é o que efetivamente o limita. Sem tetos, cada nova mensagem numa conversa longa aumenta o prompt de forma indefinida.

**Independent Test**: pode ser testado isoladamente montando o contexto com uma janela de conversa e um conjunto de memórias maiores que seus tetos e confirmando que cada seção resultante fica dentro do seu teto, enquanto a instrução de sistema e a mensagem atual permanecem idênticas às originais.

**Acceptance Scenarios**:

1. **Given** uma conversa cujas mensagens recentes somam mais que o teto da janela, **When** o contexto é montado, **Then** a seção de janela fica dentro do teto e as mensagens removidas são sempre as mais antigas.
2. **Given** um conjunto de memórias lembradas que soma mais que o teto de memórias, **When** o contexto é montado, **Then** a seção de memórias fica dentro do teto e as memórias removidas são sempre as de menor relevância (score).
3. **Given** um resumo da conversa maior que o teto de resumo, **When** o contexto é montado, **Then** o resumo é encurtado para caber no teto.
4. **Given** uma instrução de sistema e uma mensagem atual de qualquer tamanho, **When** o contexto é montado, **Then** ambas aparecem no contexto exatamente como foram fornecidas — nunca cortadas, mesmo que sejam maiores que qualquer teto.
5. **Given** qualquer estratégia de raciocínio disponível (ex.: react, plan-and-execute, com ou sem reflection), **When** uma resposta é produzida, **Then** o contexto usado foi montado pelo mesmo mecanismo com os mesmos tetos.

---

### User Story 2 - Ajustar os tetos sem mudar código (Priority: P2)

A pessoa que opera o copiloto quer ajustar o teto de cada seção por configuração do ambiente — apertando para reduzir custo ou afrouxando para dar mais memória à conversa — sem precisar alterar ou republicar código.

**Why this priority**: os valores padrão (resumo 200, janela 1200, memórias 300) são um bom ponto de partida, mas o equilíbrio ideal entre custo e qualidade varia por ambiente/modelo; ajustá-lo deve ser barato.

**Independent Test**: pode ser testado isoladamente configurando tetos diferentes dos padrões e confirmando que o contexto montado respeita os tetos configurados; e removendo a configuração para confirmar que os padrões voltam a valer.

**Acceptance Scenarios**:

1. **Given** nenhum teto configurado no ambiente, **When** o contexto é montado, **Then** valem os padrões: resumo 200, janela 1200, memórias 300 tokens.
2. **Given** um teto configurado para uma seção, **When** o contexto é montado, **Then** essa seção respeita o teto configurado e as demais seguem seus padrões.
3. **Given** um teto configurado com valor inválido (não numérico, negativo ou vazio), **When** o contexto é montado, **Then** o padrão daquela seção é usado no lugar, sem derrubar a requisição.

---

### User Story 3 - Saber o que foi cortado (Priority: P3)

A pessoa que opera o copiloto quer ver, numa resposta, que o contexto foi reduzido pelos tetos e o quanto ficou de cada seção, para diagnosticar respostas que "esqueceram" algo dito no início de uma conversa longa.

**Why this priority**: torna o corte auditável; depende do corte existir (User Story 1) e reaproveita o detalhamento de contexto já existente.

**Independent Test**: pode ser testado isoladamente enviando uma requisição cujo contexto excede algum teto e confirmando que as métricas da resposta refletem o tamanho já cortado de cada seção e indicam quantos itens foram removidos.

**Acceptance Scenarios**:

1. **Given** uma requisição em que alguma seção foi cortada, **When** a resposta é entregue, **Then** o detalhamento de contexto reflete o tamanho final (pós-corte) de cada seção e informa quantas mensagens e quantas memórias foram removidas.
2. **Given** uma requisição em que nada excedeu os tetos, **When** a resposta é entregue, **Then** o detalhamento indica zero itens removidos.

---

### Edge Cases

- Uma única mensagem da janela (a mais recente) já é maior que o teto da janela inteira: ela também é removida — a janela pode ficar vazia; a mensagem atual (fora da janela) continua intocada.
- Teto configurado como zero para uma seção: a seção inteira fica vazia (desligada), sem erro.
- Memórias com o mesmo score: o desempate é determinístico (mantém-se a ordem original de relevância), de modo que a mesma entrada sempre produz o mesmo contexto.
- Seção vazia na entrada (sem resumo, sem histórico, sem memórias — ex.: primeira mensagem sem pessoa identificada): a seção simplesmente não aparece no contexto e conta como zero, sem erro.
- Corte de uma seção nunca "empresta" sobra para outra: cada seção é limitada apenas pelo seu próprio teto.
- Mensagem atual enorme: nenhum corte é aplicado a ela; o contexto total pode exceder a soma dos tetos — os tetos limitam apenas as seções cortáveis.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema DEVE montar o contexto enviado ao modelo a partir de seções nomeadas: instrução de sistema, resumo da conversa, janela de mensagens recentes, memórias lembradas e mensagem atual.
- **FR-002**: O mesmo mecanismo de montagem DEVE ser usado por todas as estratégias de raciocínio disponíveis — nenhuma estratégia monta seu contexto de conversa/memória por conta própria.
- **FR-003**: A instrução de sistema e a mensagem atual DEVEM ser incluídas integralmente, sem qualquer corte, independentemente de tamanho.
- **FR-004**: O resumo da conversa DEVE ser limitado a um teto (padrão 200 tokens); quando excedido, o resumo é encurtado até caber.
- **FR-005**: A janela de mensagens recentes DEVE ser limitada a um teto (padrão 1200 tokens); quando excedido, mensagens inteiras são removidas a partir da mais antiga até a janela caber, preservando a ordem cronológica das restantes.
- **FR-006**: As memórias lembradas DEVEM ser limitadas a um teto (padrão 300 tokens); quando excedido, memórias inteiras são removidas a partir da de menor score até a seção caber, com desempate determinístico.
- **FR-007**: Cada teto DEVE ser configurável por variável de ambiente própria da família `CONTEXT_BUDGET_*` (uma por seção cortável); valores ausentes ou inválidos caem no padrão da seção.
- **FR-008**: O tamanho de cada seção DEVE ser medido com a mesma unidade de tokens já usada na medição de contexto (`009-context-tokens`), para que tetos e métricas sejam comparáveis.
- **FR-009**: A montagem DEVE ser determinística: a mesma entrada e os mesmos tetos produzem sempre exatamente o mesmo contexto.
- **FR-010**: O detalhamento de contexto retornado nas métricas de `/chat` DEVE refletir os tamanhos pós-corte e informar quantas mensagens da janela e quantas memórias foram removidas.
- **FR-011**: Quando nenhuma seção excede seu teto, o contexto montado DEVE conter o mesmo conteúdo que seria enviado antes desta feature (nenhuma perda de informação sem necessidade).

### Key Entities *(include if feature involves data)*

- **Seção de contexto**: parte nomeada do prompt (sistema, resumo, janela, memórias, mensagem atual), com seu conteúdo, tamanho em tokens e indicação de ser cortável ou intocável.
- **Orçamento de contexto**: conjunto de tetos por seção cortável (resumo, janela, memórias), com padrões e possibilidade de sobrescrita por ambiente.
- **Contexto montado**: resultado final entregue à estratégia de raciocínio, acompanhado do tamanho final de cada seção e da contagem de itens removidos.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Em 100% das respostas, cada seção cortável do contexto (resumo, janela, memórias) fica dentro do seu teto configurado.
- **SC-002**: Em 100% das respostas, a instrução de sistema e a mensagem atual chegam ao modelo idênticas às originais.
- **SC-003**: Com tetos baixos, os itens removidos são sempre, e só, as mensagens mais antigas da janela e as memórias de menor score — verificável por um conjunto de casos de teste com ordem esperada conhecida.
- **SC-004**: Numa conversa com 50 trocas, o tamanho do contexto enviado ao modelo deixa de crescer a cada nova mensagem, estabilizando abaixo da soma dos tetos mais sistema e mensagem atual.
- **SC-005**: Alterar um teto pela configuração do ambiente passa a valer na próxima reinicialização, sem alteração de código.

## Assumptions

- **Unidade de medida**: "200 / 1200 / 300" são tokens estimados pela mesma função de estimativa da feature `009-context-tokens` (caracteres ÷ 4), não tokens reais do provedor — o corte acontece antes da chamada ao modelo, quando o uso real ainda não existe.
- **Nomes das variáveis**: `CONTEXT_BUDGET_SUMMARY`, `CONTEXT_BUDGET_WINDOW`, `CONTEXT_BUDGET_MEMORIES`. Sistema e mensagem atual não têm variável, pois são intocáveis. Os valores são lidos uma vez, na inicialização.
- **Resumo da conversa**: o projeto ainda não gera resumo de conversa; a seção existe no montador desde já e fica vazia (zero) enquanto nenhum resumo for fornecido. Gerar o resumo está fora do escopo desta feature.
- **Corte do resumo**: o resumo, por ser um texto único, é encurtado mantendo seu início (o fim é descartado). Janela e memórias são cortadas apenas em itens inteiros — nunca uma mensagem ou memória pela metade.
- **Instrução de sistema**: as instruções de sistema já definidas por cada estratégia continuam sendo as mesmas; o montador as trata como intocáveis e não altera seu conteúdo.
- **Rótulos de seção**: o tamanho de cada seção é medido sobre o texto bruto de seus itens (mesma convenção de `009`), não sobre os rótulos/cabeçalhos de formatação adicionados ao montar o prompt.
- **Escopo de uso**: aplica-se ao fluxo de `/chat`, que é onde existem janela de conversa e memórias. `npm run bench`/`npm run arena` usam o mesmo montador, mas sem histórico/memórias as seções cortáveis ficam vazias e o comportamento é inalterado.
- **Fonte das mensagens**: a janela continua recebendo as mensagens recentes já buscadas hoje (limite de quantidade existente); o teto em tokens é aplicado por cima desse limite.
