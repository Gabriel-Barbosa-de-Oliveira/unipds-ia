# Feature Specification: Refletor de Aprendizado

**Feature Branch**: `008-learning-reflector`

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "Refletor de aprendizado: após cada resposta, um withStructuredOutput({ hasLearning, fact }) lê a ultima mensagem do usuário e destila fatos duráveis (nunca pedido pontual, nunca segredo) → memories.remember assíncrono; tool forget_preference"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Aprendizado automático de fatos duráveis a partir da conversa (Priority: P1)

A pessoa de plantão conta algo sobre si mesma ou sua forma de trabalhar durante a conversa normal
com o copiloto (sem pedir explicitamente "lembre disso"). Depois que o copiloto responde, o
sistema analisa o que a pessoa disse e, se identificar um fato durável, guarda esse fato
automaticamente na memória dela — assim como já acontece quando o fato é ensinado explicitamente.

**Why this priority**: é o motivo central da feature — sem aprendizado automático, a pessoa
continuaria precisando pedir explicitamente para o copiloto lembrar de cada coisa, o que é
exatamente o atrito que esta feature elimina.

**Independent Test**: pode ser testado isoladamente enviando uma mensagem que contém um fato
durável (sem pedir para lembrar), confirmando que a resposta do copiloto é entregue normalmente e
que, em seguida, o fato passa a ser recuperável em uma pergunta futura relacionada.

**Acceptance Scenarios**:

1. **Given** uma mensagem da pessoa contendo um fato durável nunca registrado antes, **When** o
   copiloto termina de responder, **Then** o fato é identificado e registrado na memória dessa
   pessoa, sem que ela precise pedir explicitamente.
2. **Given** uma mensagem que é apenas um pedido pontual ou uma pergunta (ex.: "abra um incidente
   para o alerta X", "qual o status do serviço Y?"), **When** o copiloto termina de responder,
   **Then** nenhum fato novo é registrado.
3. **Given** uma mensagem contendo informação sensível do tipo credencial (senha, token, chave de
   API), mesmo que junto de algo que pareça um fato durável, **When** o copiloto termina de
   responder, **Then** nenhuma informação dessa mensagem é registrada na memória.
4. **Given** um fato já registrado (ensinado manualmente ou aprendido automaticamente antes),
   **When** a pessoa menciona novamente algo essencialmente igual, **Then** nenhum registro
   duplicado é criado.

---

### User Story 2 - Desfazer um aprendizado automático (Priority: P2)

A pessoa de plantão percebe que o copiloto aprendeu sozinho algo que não é mais válido (ou que ela
não queria que ficasse guardado) e pede para esquecer essa preferência, descrevendo-a em palavras
próprias.

**Why this priority**: depende de já existir aprendizado automático (User Story 1), mas é o que dá
controle sobre o que fica guardado — sem isso, um fato capturado automaticamente e incorreto (ou
indesejado) influenciaria respostas futuras indefinidamente, sem a pessoa ter como corrigir.

**Independent Test**: pode ser testado isoladamente deixando um fato ser aprendido automaticamente,
depois pedindo para o copiloto esquecer essa preferência descrevendo-a, e confirmando que uma
pergunta relacionada deixa de recuperar esse fato.

**Acceptance Scenarios**:

1. **Given** um fato aprendido automaticamente em uma conversa anterior, **When** a pessoa pede
   para esquecer essa preferência descrevendo-a, **Then** o fato correspondente é removido e não é
   mais recuperado em perguntas futuras.
2. **Given** um pedido para esquecer algo que não corresponde a nenhum fato registrado com
   confiança suficiente, **When** o pedido é feito, **Then** o sistema informa que não encontrou
   correspondência, em vez de remover algo por engano.
3. **Given** um fato ensinado manualmente (não capturado pelo aprendizado automático), **When** a
   pessoa pede para esquecê-lo pela mesma via, **Then** ele também é removido normalmente — a forma
   de esquecer não depende de como o fato foi originalmente aprendido.

---

### User Story 3 - A resposta ao usuário nunca é afetada pelo processo de aprendizado (Priority: P3)

Independentemente do que acontece durante a análise da mensagem em busca de fatos duráveis (falha,
lentidão, resultado inesperado), a pessoa de plantão recebe a resposta do copiloto normalmente, sem
atraso perceptível e sem erros visíveis relacionados a esse processo.

**Why this priority**: é uma garantia de confiabilidade que sustenta as duas histórias anteriores —
sem ela, a tentativa de aprender automaticamente poderia degradar a experiência principal do
copiloto (responder sobre alertas e incidentes), o que anularia o valor da feature.

**Independent Test**: pode ser testado isoladamente simulando uma falha ou demora no processo de
identificação de fatos e confirmando que a resposta ao usuário é entregue normalmente, no tempo
esperado, mesmo assim.

**Acceptance Scenarios**:

1. **Given** o processo de identificação de fatos falha por qualquer motivo, **When** a pessoa
   envia uma mensagem, **Then** a resposta do copiloto é entregue normalmente e nenhum erro é
   exposto à pessoa por causa dessa falha.
2. **Given** o processo de identificação de fatos está em andamento, **When** a resposta do
   copiloto já está pronta, **Then** a resposta é entregue sem esperar esse processo terminar.

---

### Edge Cases

- O que acontece quando a mensagem da pessoa não contém fato algum (só uma pergunta ou pedido de
  ação)? Nada é registrado.
- O que acontece quando a mensagem mistura um fato genuíno com uma credencial ou dado sensível de
  forma inseparável? Nada da mensagem é registrado — nunca se arrisca vazar o dado sensível para
  preservar o fato.
- O que acontece se o pedido de esquecer (User Story 2) não corresponder com confiança suficiente a
  nenhum fato guardado, incluindo fatos aprendidos automaticamente? O sistema informa que não
  encontrou correspondência, sem remover nada.
- O que acontece se a mesma pessoa mencionar o mesmo fato em conversas diferentes, em momentos
  diferentes? Apenas um registro existe ao final (deduplicação já garantida pela memória
  semântica existente).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Depois de cada resposta do copiloto, o sistema DEVE analisar a última mensagem da
  pessoa usuária para identificar se ela contém um fato durável sobre a pessoa ou seu contexto de
  trabalho.
- **FR-002**: O sistema DEVE distinguir fatos duráveis (informação que permanece válida além da
  conversa atual) de pedidos pontuais ou perguntas (que só fazem sentido no momento da conversa
  atual), e considerar elegível para aprendizado apenas o primeiro caso.
- **FR-003**: O sistema NUNCA DEVE registrar na memória informação identificada como segredo ou
  credencial (ex.: senha, token, chave de API), mesmo quando aparece ao lado de um fato durável
  genuíno na mesma mensagem.
- **FR-004**: Quando um fato durável elegível é identificado, o sistema DEVE registrá-lo na
  memória da pessoa usuária, reaproveitando as garantias de deduplicação já existentes na memória
  semântica (fatos essencialmente iguais não geram novos registros).
- **FR-005**: A identificação e o registro do fato DEVEM ocorrer de forma assíncrona em relação à
  resposta já entregue à pessoa usuária — a entrega da resposta não pode esperar, ser atrasada nem
  falhar por causa desse processo.
- **FR-006**: Se a identificação ou o registro do fato falhar por qualquer motivo, o sistema DEVE
  absorver essa falha silenciosamente do ponto de vista da pessoa usuária, mantendo um registro do
  erro para diagnóstico.
- **FR-007**: O sistema DEVE oferecer à pessoa usuária uma forma de pedir para esquecer uma
  preferência ou fato previamente aprendido, descrevendo-o em palavras próprias.
- **FR-008**: A remoção de um fato DEVE funcionar da mesma forma independentemente de o fato ter
  sido aprendido automaticamente ou ensinado explicitamente pela pessoa.
- **FR-009**: Quando o pedido de esquecer não corresponder a nenhum fato registrado com confiança
  suficiente, o sistema DEVE informar que nada foi encontrado, em vez de remover algo por engano.
- **FR-010**: Fatos aprendidos automaticamente DEVEM permanecer isolados por pessoa usuária — nunca
  acessíveis ou removíveis por outra pessoa, seguindo o mesmo isolamento já garantido pela memória
  semântica existente.

### Key Entities *(include if feature involves data)*

- **Reflexão de aprendizado**: resultado da análise de uma mensagem da pessoa usuária após cada
  resposta do copiloto — indica se havia um fato aprendível (sim/não) e, quando sim, o texto do
  fato já destilado (sem o restante da conversa). É um resultado transitório por turno, usado
  apenas para decidir se um novo fato deve ser registrado; não é, por si só, um registro
  persistente adicional.
- **Fato de memória**: mesma entidade já definida pela memória semântica existente (fato associado
  a uma pessoa usuária, recuperável por relevância). Esta feature passa a alimentá-la também de
  forma automática, além da forma manual já existente.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Fatos duráveis mencionados espontaneamente em conversa ficam disponíveis para
  perguntas futuras sem que a pessoa precise pedir explicitamente para o copiloto lembrar, em pelo
  menos 90% de um conjunto representativo de mensagens contendo fatos claros.
- **SC-002**: Nenhuma informação do tipo credencial (senha, token, chave de API) mencionada em
  conversa aparece registrada na memória, em 100% de um conjunto de casos de teste com esse tipo
  de conteúdo.
- **SC-003**: O tempo até a pessoa usuária receber a resposta do copiloto não aumenta de forma
  perceptível em relação ao cenário sem esta feature.
- **SC-004**: Uma pessoa consegue remover um fato aprendido automaticamente apenas descrevendo-o, e
  essa remoção se reflete em perguntas subsequentes em 100% dos casos testados.
- **SC-005**: Mensagens que são apenas pedidos pontuais ou perguntas não geram nenhum registro
  novo de memória, em 100% de um conjunto de casos de teste sem fatos duráveis.

## Assumptions

- Reaproveita a `MemoryStore` e o isolamento por pessoa usuária já implementados na feature de
  memória semântica existente, incluindo o critério de deduplicação e a forma de recuperação por
  relevância já em uso.
- "Segredo/credencial" inclui, no mínimo, senhas, tokens de acesso e chaves de API; a classificação
  de uma informação como segredo é feita pelo próprio processo de destilação do fato, sem depender
  de uma lista fixa e exaustiva de padrões no domínio.
- Quando uma mensagem mistura um fato genuíno com conteúdo sensível de forma inseparável, nenhum
  fato é registrado — prioriza-se nunca vazar um dado sensível, mesmo ao custo de perder um fato
  válido nessa ocasião.
- O gatilho da reflexão é cada resposta do fluxo de conversa já existente; não há reflexão fora
  desse fluxo (ex.: não há varredura de histórico antigo).
- A forma de "esquecer uma preferência" reaproveita a mesma capacidade de remoção por descrição já
  disponível para fatos ensinados manualmente, agora também nomeada e aplicável explicitamente a
  fatos aprendidos automaticamente.
- Uma falha no processo de reflexão não gera nova tentativa automática para a mesma mensagem; se o
  mesmo fato for mencionado novamente em uma mensagem futura, ele pode ser capturado nessa ocasião.
