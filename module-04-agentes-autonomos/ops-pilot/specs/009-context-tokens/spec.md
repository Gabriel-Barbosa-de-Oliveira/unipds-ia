# Feature Specification: Medição de Contexto

**Feature Branch**: `009-context-tokens`

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "Instrumente a medição de contexto: src/context/tokens.ts com estimateTokens (chars/4) e o usage real do LangChain; métricas do /chat com promptTokens real e contextBreakdown"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ver o uso real de tokens de cada resposta do copiloto (Priority: P1)

A pessoa que opera/monitora o copiloto quer saber, para cada resposta de `/chat`, quantos tokens de prompt foram efetivamente consumidos — usando o número relatado pelo próprio provedor do modelo, não uma suposição — para acompanhar custo e uso de contexto ao longo do tempo.

**Why this priority**: é o motivo central da feature — sem uma medida real e confiável, qualquer decisão sobre custo ou tamanho de contexto (ex.: vale a pena reduzir histórico? os fatos lembrados estão pesando muito?) fica baseada em suposição, não em dado.

**Independent Test**: pode ser testado isoladamente enviando uma requisição normal a `/chat` e confirmando que a resposta inclui a contagem real de tokens de prompt relatada pelo modelo para aquela requisição.

**Acceptance Scenarios**:

1. **Given** uma requisição normal a `/chat` que aciona pelo menos uma chamada ao modelo de raciocínio, **When** a resposta é entregue, **Then** ela inclui a contagem real de tokens de prompt relatada pelo provedor do modelo para essa requisição.
2. **Given** uma requisição que aciona múltiplas chamadas ao modelo (ex.: com reflection ativado, ou várias chamadas de ferramenta), **When** a resposta é entregue, **Then** a contagem de tokens de prompt reflete o total de todas as chamadas feitas para produzir essa resposta, não só a primeira.

---

### User Story 2 - Estimativa de tokens quando o uso real não está disponível (Priority: P2)

Quando o provedor do modelo não informa quantos tokens de prompt foram usados em alguma chamada, a pessoa que opera o copiloto ainda vê uma estimativa — calculada a partir do tamanho do texto enviado — em vez de nenhuma informação.

**Why this priority**: depende de já existir a métrica real (User Story 1), mas garante que a ausência ocasional de dado do provedor nunca deixa quem opera o copiloto "no escuro" sobre o tamanho do contexto usado.

**Independent Test**: pode ser testado isoladamente simulando uma chamada ao modelo sem informação de uso real e confirmando que a métrica retornada é a estimativa, claramente identificada como tal (nunca confundida com um valor real).

**Acceptance Scenarios**:

1. **Given** o modelo não relata uso real de tokens para uma chamada, **When** a resposta de `/chat` é entregue, **Then** a métrica retornada usa uma estimativa calculada a partir do tamanho do texto enviado, identificada como estimativa.
2. **Given** o modelo relata uso real de tokens, **When** a resposta é entregue, **Then** a métrica reportada é o valor real — a estimativa nunca substitui um valor real disponível.

---

### User Story 3 - Detalhamento de onde vem o contexto enviado ao modelo (Priority: P3)

A pessoa que opera o copiloto quer entender, para uma resposta específica, de onde veio o contexto enviado ao modelo — quanto veio da mensagem atual, quanto do histórico da conversa, quanto de fatos lembrados sobre a pessoa — para diagnosticar respostas lentas, caras, ou aparentemente distraídas por contexto irrelevante.

**Why this priority**: depende de já existir alguma medida de tamanho de contexto (User Stories 1/2), mas é o que torna essa medida acionável — sem saber de onde vem o volume, não dá para saber o que reduzir.

**Independent Test**: pode ser testado isoladamente enviando uma requisição com histórico de conversa e fatos de memória registrados, e confirmando que a resposta detalha, por parte, quanto cada uma contribuiu para o contexto total enviado.

**Acceptance Scenarios**:

1. **Given** uma requisição a `/chat` com histórico de conversa e fatos de memória presentes, **When** a resposta é entregue, **Then** ela inclui um detalhamento por parte (pelo menos: mensagem atual, histórico de conversa, fatos lembrados) de quanto cada uma contribuiu para o tamanho do contexto.
2. **Given** uma requisição sem histórico de conversa nem fatos de memória (ex.: primeira mensagem, sem pessoa identificada), **When** a resposta é entregue, **Then** o detalhamento mostra que praticamente todo o contexto veio da mensagem atual, e as demais partes aparecem com valor zero, nunca omitidas.
3. **Given** o detalhamento de uma resposta, **When** as partes são somadas, **Then** a soma corresponde ao tamanho total de contexto relatado para essa resposta (real ou estimado).

---

### Edge Cases

- Mensagem muito curta (ex.: "oi"): a estimativa ainda é calculada normalmente, sem erro, mesmo resultando em um número pequeno.
- Requisição que usa uma estratégia de raciocínio diferente da padrão (ex.: plan-and-execute, ou com reflection ativado): a contagem de tokens e o detalhamento ainda refletem a chamada real feita, incluindo todas as sub-chamadas envolvidas em produzir aquela resposta.
- Requisição sem pessoa identificada (sem memória semântica ativa), mas com histórico de conversa: o detalhamento mostra a parte de fatos lembrados como zero e a parte de histórico com seu valor real.
- Duas requisições concorrentes: a métrica de tokens/contexto de uma nunca aparece misturada com a da outra.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema DEVE calcular e incluir, em toda resposta de `/chat` que aciona o modelo de raciocínio, uma contagem de tokens de prompt.
- **FR-002**: Quando o provedor do modelo relata o uso real de tokens de prompt para uma chamada, o sistema DEVE usar esse valor real na métrica reportada, agregando entre todas as chamadas feitas para produzir aquela resposta.
- **FR-003**: Quando o provedor do modelo não relata uso real de tokens (para parte ou toda a resposta), o sistema DEVE calcular uma estimativa a partir do tamanho do texto enviado, em vez de omitir a métrica.
- **FR-004**: A métrica de tokens retornada DEVE indicar claramente se é baseada em uso real ou em estimativa, sem misturar as duas fontes de forma que pareçam a mesma coisa.
- **FR-005**: O sistema DEVE detalhar o contexto enviado ao modelo em partes nomeadas e reconhecíveis — no mínimo: mensagem atual da pessoa, histórico de conversa e fatos lembrados — com o tamanho contribuído por cada uma.
- **FR-006**: A soma das partes do detalhamento de contexto DEVE corresponder ao tamanho total de contexto relatado para aquela resposta.
- **FR-007**: Partes do contexto que não se aplicam a uma requisição (ex.: sem histórico, sem fatos lembrados) DEVEM aparecer no detalhamento com valor zero, nunca omitidas silenciosamente.
- **FR-008**: A medição de contexto/tokens NUNCA DEVE alterar a resposta dada à pessoa usuária (`answer`) nem o comportamento do raciocínio do copiloto — é estritamente informativa.
- **FR-009**: As métricas de tokens/contexto de requisições concorrentes NUNCA DEVEM se misturar entre si — cada resposta reflete só a sua própria requisição.

### Key Entities *(include if feature involves data)*

- **Uso de tokens**: contagem de tokens de prompt associada a uma resposta de `/chat`, com indicação de origem — real (relatado pelo provedor do modelo) ou estimado (calculado a partir do tamanho do texto enviado).
- **Detalhamento de contexto**: conjunto de partes nomeadas (mensagem atual, histórico de conversa, fatos lembrados) com o tamanho que cada uma contribuiu para o contexto total enviado ao modelo numa resposta.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Em 100% das respostas de `/chat` em que o provedor do modelo relata uso real, a contagem de tokens de prompt reportada corresponde exatamente ao valor relatado pelo provedor.
- **SC-002**: Em 100% das respostas de `/chat`, alguma contagem de tokens de prompt (real ou estimada) está presente — nunca ausente.
- **SC-003**: Em 100% das respostas, a soma das partes do detalhamento de contexto é igual ao total de tokens de prompt relatado para essa resposta.
- **SC-004**: Uma pessoa consegue identificar, a partir de uma única resposta de `/chat`, qual parte do contexto mais contribuiu para o tamanho do prompt, sem precisar de nenhuma ferramenta externa de análise.

## Assumptions

- Esta feature se aplica apenas ao endpoint `/chat` (mesmo escopo de `006`/`007`/`008`) — `npm run bench`/`npm run arena` não passam por essa instrumentação.
- "Uso real" é o valor relatado pelo provedor do modelo de raciocínio através do framework de orquestração já usado pelo projeto; quando esse valor não está disponível para uma chamada específica, considera-se "não disponível" e a estimativa é usada no lugar dela, só para essa chamada.
- A estimativa (quando usada) é calculada a partir da contagem de caracteres do texto dividida por 4 — aproximação comum e suficiente para efeitos de monitoramento, não para faturamento exato.
- As partes do detalhamento de contexto (FR-005) cobrem, no mínimo, mensagem atual, histórico de conversa e fatos lembrados; outras partes que eventualmente componham o prompt podem ser incluídas como uma parte adicional, mas não são o foco desta feature.
- Quando uma resposta envolve múltiplas chamadas ao modelo (ex.: reflection, chamadas de ferramenta dentro do raciocínio), a contagem de tokens de prompt reportada é a soma de todas essas chamadas — mesmo espírito já usado pela métrica de quantidade de chamadas ao modelo já existente.
