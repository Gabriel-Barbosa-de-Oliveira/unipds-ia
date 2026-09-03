# Feature Specification: Camada de Reflection

**Feature Branch**: `002-reflection-layer`

**Created**: 2026-09-03

**Status**: Draft

**Input**: User description: "Camada Reflection: withReflection(strategy, opts) decora qualquer Reasoning Strategy: executa a base; um crítico (mesmo modelo, saída estruturada { approved, feedback }) avalia a resposta contra as observações do trace; se reprovar, regenera com o feedback no contexto; para em approved ou maxReflections (default 2). Event \"critique\" no trace; métricas somam as chamadas extras. Arena: reflect:react e reflect:plan-and-execute"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Obter uma resposta autocriticada e mais confiável (Priority: P1)

Um desenvolvedor quer que as respostas produzidas por uma estratégia de raciocínio existente (ReAct ou Plan-and-Execute) passem por uma revisão automática antes de serem consideradas finais: um crítico avalia se a resposta é de fato sustentada pelas observações coletadas durante a execução e, se não for, a estratégia tenta novamente levando em conta o motivo da reprovação.

**Why this priority**: É o valor central da feature — sem o ciclo crítica → regeneração, não existe "reflection", apenas a estratégia base já existente. Todo o resto (comparação na arena, limites de custo) depende deste comportamento existir primeiro.

**Independent Test**: Pode ser testado isoladamente envolvendo uma única estratégia de raciocínio com a camada de reflection, enviando uma pergunta operacional, e verificando que: (a) a estratégia base é executada, (b) um crítico avalia a resposta produzida, (c) se o crítico reprovar, uma nova tentativa é gerada incorporando o feedback, e (d) a execução para assim que o crítico aprovar.

**Acceptance Scenarios**:

1. **Given** uma estratégia de raciocínio já existente envolvida pela camada de reflection, **When** a resposta da primeira tentativa é aprovada pelo crítico, **Then** a execução termina com essa resposta, sem nenhuma tentativa adicional.
2. **Given** uma estratégia de raciocínio envolvida pela camada de reflection, **When** o crítico reprova a primeira resposta, **Then** o sistema gera uma nova tentativa que recebe o feedback da reprovação como parte do seu contexto de entrada.
3. **Given** uma execução com reflection em andamento, **When** o crítico aprova uma tentativa (seja a primeira ou uma tentativa regenerada), **Then** a execução para imediatamente e não são feitas tentativas adicionais.

---

### User Story 2 - Comparar estratégias com e sem reflection na arena (Priority: P2)

Um desenvolvedor quer decidir se vale a pena pagar o custo extra da reflection: ele roda a mesma pergunta operacional contra a versão base de uma estratégia e contra sua versão com reflection (ex.: `react` vs `reflect:react`, ou `plan-and-execute` vs `reflect:plan-and-execute`) e compara trace, resposta final e métricas lado a lado.

**Why this priority**: Depende da User Story 1 já existir (o ciclo de reflection funcionando), mas é o que permite decidir, com dados, se a reflection compensa em confiabilidade o custo adicional de chamadas ao modelo — sem isso, a feature fica isolada e não é avaliável frente ao que já existe.

**Independent Test**: Pode ser testado isoladamente selecionando uma estratégia base e sua contraparte com reflection na arena, executando a mesma pergunta operacional em ambas, e verificando que a saída identifica claramente qual trace/métrica pertence a cada uma, incluindo o custo adicional (chamadas de modelo) da versão com reflection.

**Acceptance Scenarios**:

1. **Given** a arena com as estratégias `reflect:react` e `reflect:plan-and-execute` disponíveis, **When** o desenvolvedor as seleciona junto de suas versões base para a mesma pergunta, **Then** a arena executa todas as estratégias selecionadas e apresenta trace e métricas de cada uma de forma identificável.
2. **Given** uma execução comparativa incluindo uma estratégia com reflection, **When** o desenvolvedor revisa o resumo comparativo de métricas, **Then** o número de chamadas ao modelo da versão com reflection reflete tanto as chamadas da(s) tentativa(s) da estratégia base quanto as chamadas extras feitas pelo crítico.

---

### User Story 3 - Confiar que a reflection tem custo e tempo limitados (Priority: P3)

Um desenvolvedor confia que a camada de reflection nunca entra em um ciclo de regeneração sem fim: existe um número máximo configurável de reflections (2 por padrão) e, ao atingi-lo sem aprovação, a execução para de forma controlada retornando a última resposta produzida, com o histórico completo de críticas disponível para auditoria.

**Why this priority**: É uma camada de segurança/previsibilidade de custo sobre as User Stories 1 e 2. O sistema já entrega valor demonstrável sem essa garantia explícita (em uso manual/controlado), mas ela é necessária antes de confiar na feature em qualquer uso repetido ou automatizado.

**Independent Test**: Pode ser testado isoladamente configurando um crítico que nunca aprova (ou usando um cenário onde a reprovação é esperada) e verificando que o número de tentativas/regenerações nunca ultrapassa o limite configurado, que a execução termina com uma resposta (não trava, não lança erro) e que o trace mostra o histórico completo de críticas que levaram à parada.

**Acceptance Scenarios**:

1. **Given** um limite máximo de reflections configurado, **When** o crítico reprova repetidamente até esse limite ser atingido, **Then** a execução para, retorna a última resposta produzida (mesmo não aprovada) e o trace contém uma crítica registrada para cada tentativa avaliada.
2. **Given** uma execução com reflection concluída (por aprovação ou por limite atingido), **When** o resultado é revisado, **Then** as métricas relatadas incluem todas as chamadas ao modelo feitas em todas as tentativas da estratégia base somadas a todas as chamadas feitas pelo crítico.
3. **Given** o limite máximo de reflections não é informado explicitamente, **When** a execução com reflection ocorre, **Then** o sistema aplica o padrão de 2 reflections.

---

### Edge Cases

- O que acontece quando o crítico aprova já na primeira tentativa? Nenhuma regeneração deve ocorrer, mas a crítica de aprovação ainda deve ser registrada no trace e contar como uma chamada extra ao modelo nas métricas.
- O que acontece quando a estratégia base, por si só, já para de forma controlada por ter atingido seu próprio limite interno de iterações (sem produzir uma resposta "normal")? O crítico ainda deve avaliar essa resposta de parada controlada como qualquer outra resposta.
- O que acontece quando o crítico (ou o modelo subjacente) falha por um problema de infraestrutura (ex.: modelo indisponível) durante a avaliação? A execução com reflection deve falhar rapidamente com um erro claro, em vez de mascarar a falha ou travar — mesmo comportamento de falha de infraestrutura já adotado pelas estratégias base.
- O que acontece quando `maxReflections` é configurado como 0? Nenhuma regeneração deve ser permitida; a execução resulta na primeira resposta produzida pela estratégia base, com o histórico de crítica disponível de acordo com o mesmo comportamento aplicado a qualquer outro limite atingido.
- O que acontece ao comparar, na arena, uma estratégia base e sua versão com reflection lado a lado? Ambas devem operar sobre o mesmo conjunto de dados semeado sem que a execução de uma interfira na da outra, da mesma forma que duas estratégias base já se comportam hoje na arena.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema DEVE oferecer uma forma de envolver ("decorar") qualquer estratégia de raciocínio existente com uma camada de reflection, sem exigir alteração na implementação da estratégia envolvida.
- **FR-002**: Após a estratégia envolvida produzir uma resposta, o sistema DEVE submeter essa resposta a uma avaliação crítica que julga se ela é sustentada pelas observações registradas no trace daquela tentativa, produzindo uma decisão (aprovada ou não) e uma justificativa/feedback explicando o motivo.
- **FR-003**: Quando a avaliação crítica reprovar a resposta, o sistema DEVE gerar uma nova tentativa executando novamente a estratégia envolvida, incorporando o feedback da reprovação ao contexto dessa nova tentativa.
- **FR-004**: O ciclo de crítica e possível regeneração DEVE se repetir até que uma resposta seja aprovada ou até que o número máximo de reflections configurado seja atingido, o que ocorrer primeiro.
- **FR-005**: O número máximo de reflections DEVE ser configurável por execução e assumir o valor padrão de 2 quando não informado. `maxReflections` conta regenerações extras após a primeira tentativa — a primeira tentativa nunca conta como uma reflection. Com o padrão de 2, uma execução avalia no máximo 3 tentativas no total (1 tentativa inicial + até 2 regenerações).
- **FR-006**: Ao atingir o número máximo de reflections sem uma resposta aprovada, o sistema DEVE encerrar de forma controlada e retornar a última resposta produzida, junto com o histórico completo do que foi tentado, em vez de falhar ou continuar indefinidamente.
- **FR-007**: Toda avaliação crítica realizada DEVE ser registrada como um evento próprio no trace da execução, em ordem cronológica junto com os demais eventos da estratégia envolvida, de modo que todo o histórico de reflection seja auditável após o fato.
- **FR-008**: As métricas de uma execução com reflection DEVEM incluir a soma de todas as chamadas ao modelo feitas em todas as tentativas da estratégia envolvida mais todas as chamadas feitas pela avaliação crítica — nunca reportar apenas as chamadas da última tentativa ou omitir as chamadas do crítico.
- **FR-009**: A camada de reflection DEVE poder ser aplicada a cada estratégia de raciocínio existente de forma independente, expondo cada versão envolvida sob um nome distinto e identificável, sem remover ou alterar a estratégia base original.
- **FR-010**: O modo de comparação (arena) DEVE permitir selecionar as versões com reflection das estratégias existentes (ReAct e Plan-and-Execute) ao lado de — ou no lugar de — suas versões base, usando os mesmos parâmetros de entrada e limite de iterações já suportados.
- **FR-011**: A camada de reflection NUNCA DEVE lançar erro para uma reprovação do crítico ou para o esgotamento do número máximo de reflections — essas são condições de negócio esperadas que resultam em um resultado normal com o trace e a resposta correspondentes; falhas de infraestrutura não recuperáveis continuam propagando erro, como já ocorre nas estratégias base.
- **FR-012**: O limite máximo de iterações/passos já respeitado pela estratégia base DEVE continuar sendo respeitado individualmente em cada tentativa executada pela camada de reflection.

### Key Entities

- **Camada de Reflection**: um decorador que envolve uma Estratégia de Raciocínio existente, adicionando um ciclo de autocrítica e regeneração sem alterar a estratégia original.
- **Avaliação Crítica (Critique)**: o resultado de uma avaliação sobre uma tentativa de resposta; possui uma decisão (aprovada/reprovada) e um feedback textual explicando o motivo.
- **Tentativa (Attempt)**: uma execução completa da estratégia envolvida dentro de um ciclo de reflection, incluindo seu próprio trace e resposta; a primeira tentativa e cada regeneração subsequente são todas tentativas.
- **Histórico de Reflection**: a sequência ordenada de tentativas e avaliações críticas realizadas durante uma execução com reflection, do início até a aprovação ou o esgotamento do limite.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Quando o crítico aprova a primeira tentativa, o custo adicional da reflection fica limitado a exatamente uma avaliação crítica extra — nenhuma regeneração é executada.
- **SC-002**: 100% das execuções com reflection terminam respeitando o número máximo de reflections configurado — nenhuma execução realiza mais tentativas do que o limite permite.
- **SC-003**: Para qualquer execução com reflection concluída, um revisor consegue determinar, olhando apenas para o trace, quantas tentativas foram feitas, qual foi o feedback de cada avaliação crítica, e se a execução parou por aprovação ou por esgotamento do limite.
- **SC-004**: Um desenvolvedor consegue comparar, em uma única execução da arena, uma estratégia base e sua versão com reflection sobre a mesma pergunta operacional, e identificar o custo adicional em chamadas de modelo introduzido pela reflection.
- **SC-005**: 100% das execuções com reflection produzem uma resposta final (por aprovação ou por parada controlada no limite) — nenhuma execução trava ou fica pendente indefinidamente.

## Assumptions

- O crítico usa o mesmo modelo de raciocínio subjacente já configurado para as estratégias (mesma fábrica de modelo), apenas com uma chamada adicional dedicada à avaliação — não é necessário um modelo/serviço externo separado.
- A avaliação crítica tem acesso à resposta final da tentativa e às observações registradas no trace daquela tentativa (mas não precisa reavaliar raciocínio/pensamentos intermediários que não geraram observação).
- O feedback de uma reprovação é incorporado à tentativa seguinte por meio do contexto de entrada fornecido à estratégia envolvida, já que a interface de estratégias existente não possui um canal dedicado separado para isso.
- As estratégias base (`react`, `plan-and-execute`) permanecem disponíveis e inalteradas na arena; as versões com reflection (`reflect:react`, `reflect:plan-and-execute`) são adicionadas como opções novas, não substituem as existentes.
- O conjunto de dados operacional semeado (serviços, alertas, incidentes) já definido pela feature de núcleo de raciocínio é reutilizado sem alterações; esta feature não introduz novas ações operacionais nem novos dados.
- Assim como nas estratégias base, uma execução com reflection por vez é aceitável nesta fase; isolamento entre execuções simultâneas continua fora de escopo.
