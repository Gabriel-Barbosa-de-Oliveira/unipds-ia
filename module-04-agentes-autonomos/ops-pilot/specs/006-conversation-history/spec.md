# Feature Specification: Conversa Persistente

**Feature Branch**: `006-conversation-history`

**Created**: 2026-09-15

**Status**: Draft

**Input**: User description: "Conversa persistente:
- ConversationStore (append/lastMessages/create) + tabela messages como no SqliteOpsStore
- /chat: conversationId opcional, devolvido na resposta
- 12 últimas mensagens no prompt via composição; métrica historyMessages"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Continuar uma conversa ao longo de várias mensagens (Priority: P1)

A pessoa de plantão conversa com o copiloto em várias trocas de mensagem (ex.: "me chame de Gabriel", depois "abra um incidente médio no catalog") e o copiloto responde à segunda mensagem já sabendo o que foi dito na primeira, sem que a pessoa precise repetir o contexto.

**Why this priority**: é o motivo central da feature — hoje cada chamada ao copiloto é isolada; sem continuidade de conversa, nenhuma das capacidades abaixo (retomar depois, auditar quanto histórico foi usado) tem valor prático.

**Independent Test**: pode ser testado isoladamente enviando uma primeira mensagem, guardando o identificador de conversa devolvido, enviando uma segunda mensagem referenciando esse identificador e algo dito na primeira (ex.: um nome informado), e verificando que a resposta reflete esse contexto anterior.

**Acceptance Scenarios**:

1. **Given** nenhuma conversa prévia, **When** a pessoa de plantão envia uma mensagem sem identificador de conversa, **Then** o copiloto responde normalmente e a resposta inclui um identificador de conversa novo.
2. **Given** um identificador de conversa de uma troca anterior, **When** a pessoa de plantão envia uma nova mensagem referenciando esse identificador, **Then** o copiloto responde levando em conta o que foi dito nas mensagens anteriores dessa mesma conversa.
3. **Given** duas conversas diferentes em andamento, **When** mensagens são enviadas para cada uma, **Then** o histórico de uma conversa nunca aparece nem influencia a resposta da outra.

---

### User Story 2 - Retomar uma conversa longa sem perder desempenho (Priority: P2)

Uma conversa que já acumulou muitas mensagens ao longo de um plantão continua respondendo de forma ágil e coerente, sem que o copiloto precise reprocessar a conversa inteira a cada nova pergunta.

**Why this priority**: depende da User Story 1 já existir, mas é o que torna a continuidade de conversa sustentável durante um plantão longo, em vez de degradar com o tempo.

**Independent Test**: pode ser testado isoladamente conduzindo uma conversa com mais de 12 trocas de mensagem e verificando que o copiloto continua respondendo com base no contexto mais recente, sem erro e sem tempo de resposta crescente proporcional ao tamanho total da conversa.

**Acceptance Scenarios**:

1. **Given** uma conversa com mais de 12 mensagens já trocadas, **When** uma nova mensagem é enviada, **Then** o copiloto responde normalmente, considerando o contexto mais recente da conversa.
2. **Given** a mesma conversa longa, **When** a resposta é produzida, **Then** é possível saber quantas mensagens de histórico foram efetivamente consideradas para aquela resposta.

---

### User Story 3 - Auditar quanto contexto foi usado em cada resposta (Priority: P3)

Quem avalia ou depura o comportamento do copiloto (durante desenvolvimento, comparação de estratégias, ou investigação de uma resposta inesperada) consegue ver quantas mensagens de histórico entraram na composição daquela resposta específica.

**Why this priority**: não é percebido pela pessoa de plantão durante o uso normal; é o que dá visibilidade e confiança sobre o comportamento da continuidade de conversa introduzida pelas stories anteriores.

**Independent Test**: pode ser testado isoladamente enviando mensagens em conversas com diferentes quantidades de histórico acumulado (nenhuma, poucas, mais de 12) e verificando que o número de mensagens de histórico reportado corresponde ao que de fato foi incluído em cada caso.

**Acceptance Scenarios**:

1. **Given** uma conversa nova, sem mensagens anteriores, **When** a primeira mensagem é enviada, **Then** a resposta indica zero mensagens de histórico consideradas.
2. **Given** uma conversa com poucas mensagens (menos de 12), **When** uma nova mensagem é enviada, **Then** a resposta indica exatamente a quantidade de mensagens anteriores existentes.
3. **Given** uma conversa com mais de 12 mensagens, **When** uma nova mensagem é enviada, **Then** a resposta indica no máximo 12 mensagens de histórico consideradas.

---

### Edge Cases

- O que acontece quando é informado um identificador de conversa que não corresponde a nenhuma conversa conhecida pelo sistema?
- O que acontece quando duas mensagens são enviadas para a mesma conversa ao mesmo tempo (concorrência)?
- O que acontece quando a mensagem enviada é a primeira de uma conversa nova (sem identificador informado)?
- O que acontece com o histórico de uma conversa quando o processo da aplicação é reiniciado no meio dela?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema MUST persistir de forma durável as mensagens de cada conversa (tanto as enviadas pela pessoa de plantão quanto as respostas do copiloto), de modo que sobrevivam a reinícios do processo da aplicação.
- **FR-002**: O endpoint de conversa MUST aceitar um identificador de conversa opcional; quando informado e correspondente a uma conversa existente, o sistema MUST continuar essa conversa; quando omitido, o sistema MUST iniciar uma nova conversa.
- **FR-003**: Toda resposta do endpoint de conversa MUST incluir o identificador da conversa correspondente (o informado na requisição, ou um novo gerado quando nenhum foi informado), permitindo que quem chama continue a mesma conversa na próxima mensagem.
- **FR-004**: Ao processar uma nova mensagem de uma conversa existente, o sistema MUST considerar as mensagens anteriores dessa conversa como contexto, limitado às 12 mensagens mais recentes.
- **FR-005**: Toda resposta do endpoint de conversa MUST reportar quantas mensagens de histórico foram efetivamente consideradas no processamento daquela resposta específica.
- **FR-006**: Ao final de cada troca, o sistema MUST registrar tanto a mensagem da pessoa de plantão quanto a resposta final do copiloto no histórico da conversa, para que fiquem disponíveis em trocas futuras.
- **FR-007**: Quando um identificador de conversa informado não corresponde a nenhuma conversa conhecida, o sistema MUST responder com um erro que deixe isso claro, em vez de iniciar silenciosamente uma conversa vazia sob esse identificador.
- **FR-008**: Uma requisição ao endpoint de conversa sem identificador de conversa informado MUST continuar produzindo o mesmo comportamento de resposta já existente (resposta, trace, métricas), com o acréscimo do identificador da nova conversa.
- **FR-009**: O histórico de cada conversa MUST permanecer isolado das demais — nenhuma mensagem de uma conversa MUST aparecer no contexto de outra conversa.

### Key Entities

- **Conversa**: sequência de trocas de mensagem entre uma pessoa de plantão e o copiloto ao longo do tempo; identificada por um identificador único; agrupa suas mensagens em ordem cronológica.
- **Mensagem**: uma unidade de diálogo dentro de uma conversa — quem a originou (pessoa de plantão ou copiloto) e seu conteúdo; pertence a exatamente uma conversa e tem uma posição definida na ordem cronológica dela.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Uma pessoa de plantão faz uma pergunta de acompanhamento que depende do que disse antes (ex.: um nome ou serviço mencionado anteriormente) e o copiloto responde considerando corretamente esse contexto, sem que ela precise repeti-lo.
- **SC-002**: 100% das respostas do endpoint de conversa incluem um identificador que permite retomar exatamente a mesma conversa na mensagem seguinte.
- **SC-003**: Conversas com mais de 12 mensagens trocadas continuam funcionando sem erro e sem degradação perceptível de tempo de resposta em relação a uma conversa mais curta.
- **SC-004**: Para qualquer resposta do copiloto, é possível determinar exatamente quantas mensagens de histórico foram usadas para produzi-la.
- **SC-005**: Mensagens de duas conversas diferentes, mesmo enviadas na mesma janela de tempo, nunca se misturam no histórico usada para gerar uma resposta.

## Assumptions

- A persistência de conversas reaproveita a mesma abordagem de armazenamento já adotada para dados operacionais (feature 004) — dentro do mesmo armazenamento local durável do processo, sem exigir um serviço externo adicional.
- "Últimas 12 mensagens" conta tanto mensagens da pessoa de plantão quanto respostas do copiloto, na ordem cronológica em que ocorreram — não apenas as mensagens de um dos dois lados.
- A resposta ao endpoint de conversa passa a incluir o identificador da conversa como um campo adicional, estendendo o formato de resposta já existente (feature 003) sem remover nem renomear os campos atuais (`answer`, `trace`, `metrics`).
- Não há autenticação ou controle de acesso por conversa além do já aplicado ao copiloto como um todo — está fora do escopo desta feature.
- O histórico completo de cada conversa é mantido indefinidamente no armazenamento durável; apenas a quantidade usada na composição do prompt enviado ao modelo é limitada às 12 mensagens mais recentes.
- Esta feature cobre o endpoint HTTP de conversa; a integração equivalente pelo servidor MCP (feature 005), se desejada, é tratada como uma extensão futura fora deste escopo.
