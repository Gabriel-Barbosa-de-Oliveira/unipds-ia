# Feature Specification: Trace Persistido e Logs Estruturados

**Feature Branch**: `014-persisted-trace`

**Created**: 2026-10-09

**Status**: Draft

**Input**: User description: "Trace persistido + logs JSON: /chat: requestid no corpo e no header X-Request-Id. SQLite: requests (métricas) e trace_events (node, payloads). src/obs/logger.ts: 1 linha JSON por evento, só metadados. GET /requests/:id registro + trace ordenados."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reabrir depois o que aconteceu numa resposta (Priority: P1)

Uma pessoa de plantão reclama que o copiloto deu uma resposta estranha há uma hora. Quem opera o copiloto pega o identificador daquela requisição e consulta o registro completo: quando foi, quais métricas teve (chamadas ao modelo, tokens, latência, modelo usado, rota escolhida, se terminou bem ou com erro) e o trace inteiro, evento a evento, na ordem em que aconteceu e com a etapa (nó) de cada um.

**Why this priority**: é o núcleo da feature. Hoje o trace só existe na resposta HTTP; se ninguém guardou essa resposta, a investigação é impossível. Sem registro durável não há auditoria de decisões do agente sobre incidentes reais.

**Independent Test**: dá para testar fazendo uma requisição ao chat, guardando o identificador devolvido e, depois, consultando esse identificador: o registro e o trace devolvidos devem ser iguais aos da resposta original, na mesma ordem, mesmo depois de reiniciar o serviço.

**Acceptance Scenarios**:

1. **Given** uma resposta do chat concluída com sucesso, **When** o identificador dessa requisição é consultado, **Then** o retorno traz as métricas da requisição e todos os eventos do trace, na mesma ordem e com o mesmo conteúdo da resposta original.
2. **Given** um registro gravado, **When** o serviço é reiniciado e o identificador é consultado de novo, **Then** o mesmo registro é devolvido.
3. **Given** um identificador que não existe, **When** ele é consultado, **Then** o retorno é "não encontrado", sem detalhes internos.
4. **Given** uma requisição que terminou em erro de execução (tempo esgotado ou falha interna), **When** o identificador é consultado, **Then** o registro existe, indica o desfecho (erro e tipo de erro) e traz o que houver de trace (possivelmente vazio).

---

### User Story 2 - Correlacionar a resposta com o registro e com os logs (Priority: P1)

Toda resposta do chat, de sucesso ou de erro, carrega um identificador único da requisição, tanto no corpo quanto num cabeçalho padrão. Esse mesmo identificador aparece no registro persistido e em todas as linhas de log daquela requisição, para que quem opera consiga ir da resposta (ou de uma reclamação) ao registro e aos logs sem adivinhação.

**Why this priority**: sem um identificador compartilhado, o registro persistido e os logs viram ilhas. É a chave que liga a US1 à US3.

**Independent Test**: dá para testar fazendo requisições de sucesso e de erro (corpo inválido, estratégia desconhecida, falha interna) e confirmando que todas trazem o identificador no corpo e no cabeçalho, que os dois são iguais e que dois pedidos nunca recebem o mesmo identificador.

**Acceptance Scenarios**:

1. **Given** qualquer requisição ao chat, **When** a resposta é devolvida (sucesso ou erro), **Then** ela traz o identificador da requisição no corpo e no cabeçalho, com o mesmo valor.
2. **Given** duas requisições, mesmo simultâneas, **When** as respostas chegam, **Then** os identificadores são diferentes.
3. **Given** uma resposta de sucesso, **When** seu identificador é consultado, **Then** o registro correspondente é encontrado.

---

### User Story 3 - Logs estruturados, uma linha por evento, sem dados sensíveis (Priority: P2)

Quem opera o copiloto quer que o serviço emita logs legíveis por máquina: uma linha por acontecimento relevante (requisição recebida, rota escolhida, troca de modelo, ferramenta chamada, resposta concluída, erro), cada uma com o identificador da requisição, o horário, a etapa e metadados (duração, contagens, nomes). Os logs nunca podem conter o conteúdo das conversas (mensagens, respostas, argumentos e resultados de ferramentas), porque podem ser enviados a sistemas de log com acesso mais amplo.

**Why this priority**: logs permitem acompanhar o serviço em tempo real e agregar números (latência, taxa de fallback) sem consultar o banco. Mas o diagnóstico detalhado já é coberto pela US1, então esta história é complementar.

**Independent Test**: dá para testar capturando a saída de log de uma requisição e confirmando que cada linha é um objeto estruturado válido e independente, que todas trazem o identificador da requisição e que nenhuma contém o texto da mensagem, da resposta ou dos argumentos e resultados de ferramentas.

**Acceptance Scenarios**:

1. **Given** uma requisição ao chat, **When** ela é processada, **Then** cada evento relevante gera exatamente uma linha de log estruturada e autocontida.
2. **Given** qualquer linha de log de uma requisição, **When** ela é inspecionada, **Then** ela tem o identificador da requisição, o horário, o tipo de evento e, quando aplicável, a etapa.
3. **Given** uma requisição cuja mensagem e resposta contêm um texto marcador conhecido, **When** todos os logs são inspecionados, **Then** o marcador não aparece em nenhuma linha.
4. **Given** uma falha interna, **When** ela é logada, **Then** a linha traz o tipo do erro, mas não o conteúdo da conversa.

---

### Edge Cases

- Falha ao gravar o registro (ex.: banco indisponível): a resposta ao usuário não pode falhar nem atrasar por causa disso; a falha de gravação é logada (só com metadados).
- Requisição rejeitada antes de executar (corpo inválido, estratégia desconhecida): recebe identificador e gera log, mas não gera registro persistido, porque não houve execução nem trace.
- Tempo esgotado: o registro é gravado com desfecho "timeout" e o trace fica vazio, porque o trace parcial não é recuperável da execução abandonada.
- Conversa e requisição são coisas diferentes: uma conversa tem várias requisições. O registro guarda o identificador da conversa para permitir a ligação.
- Payloads grandes de eventos (ex.: observação com lista longa) são guardados por inteiro no registro persistido; os logs nunca os incluem.
- O identificador consultado com formato inválido é tratado como "não encontrado".

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Toda resposta do chat (sucesso ou erro) MUST carregar um identificador único da requisição no corpo e no cabeçalho de resposta `X-Request-Id`, com valores iguais.
- **FR-002**: O identificador MUST ser gerado pelo servidor e ser único entre requisições, inclusive simultâneas.
- **FR-003**: Toda requisição do chat que chegou a executar (sucesso, tempo esgotado ou falha interna) MUST gerar um registro durável com: identificador, conversa, momento de início, duração, desfecho (sucesso/tipo de erro), rota escolhida e sua origem, e as métricas da resposta (chamadas ao modelo, tokens, origem dos tokens, modelo usado e métricas de contexto).
- **FR-004**: Cada evento do trace de uma requisição bem-sucedida MUST ser gravado de forma durável e associado ao registro da requisição, com posição, tipo, etapa (nó) e o conteúdo completo do evento.
- **FR-005**: O sistema MUST permitir consultar um registro pelo identificador, devolvendo o registro e o trace completo ordenado pela posição original.
- **FR-006**: A consulta de um identificador inexistente MUST devolver "não encontrado".
- **FR-007**: O trace devolvido pela consulta MUST ser idêntico, em conteúdo e ordem, ao trace devolvido na resposta original do chat.
- **FR-008**: A gravação do registro MUST NOT alterar o conteúdo da resposta do chat nem fazê-la falhar; falhas de gravação MUST ser logadas.
- **FR-009**: O serviço MUST emitir uma linha de log estruturada e autocontida por evento relevante: requisição recebida, requisição rejeitada, rota escolhida, troca de modelo, ação de ferramenta, resposta concluída, erro e falha de gravação.
- **FR-010**: Toda linha de log MUST conter o identificador da requisição (quando houver), o horário, o nível e o tipo do evento; e, quando aplicável, a etapa e metadados numéricos ou nominais (duração, contagens, nome da ferramenta, rota, modelos).
- **FR-011**: As linhas de log MUST NOT conter o conteúdo da conversa: mensagem do usuário, resposta, pensamentos, argumentos ou resultados de ferramentas, motivo textual do roteador, histórico ou memórias.
- **FR-012**: O identificador e o registro MUST funcionar para todas as rotas e estratégias e com override de estratégia, sem diferença de comportamento.
- **FR-013**: A lógica de montagem do registro, a seleção dos metadados de log (o que entra e o que fica de fora) e a ordenação do trace MUST ser cobertas por testes sem rede.

### Key Entities

- **Registro de requisição**: uma execução do chat. Tem identificador, conversa, início, duração, desfecho, rota e sua origem, e as métricas.
- **Evento de trace persistido**: um evento do trace de uma requisição. Tem o identificador da requisição, a posição, o tipo, a etapa e o conteúdo completo.
- **Linha de log**: um acontecimento do serviço. Tem horário, nível, tipo, identificador da requisição e metadados, sem conteúdo de conversa.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% das respostas do chat (sucesso e erro) trazem um identificador de requisição, igual no corpo e no cabeçalho.
- **SC-002**: 100% das requisições que executaram podem ser reabertas pelo identificador, inclusive depois de reiniciar o serviço, com trace idêntico ao original.
- **SC-003**: Quem opera localiza o registro de uma resposta problemática a partir do identificador em menos de 1 minuto, sem acesso à resposta original.
- **SC-004**: 0 linhas de log contendo conteúdo de conversa, verificado por um teste com texto marcador.
- **SC-005**: 100% das linhas de log são objetos estruturados válidos, um por linha, e todas as linhas de uma requisição compartilham o mesmo identificador.
- **SC-006**: Persistir o registro não acrescenta tempo perceptível à resposta (menos de 5% da latência no caso típico).

## Assumptions

- O identificador é sempre gerado pelo servidor (um UUID). Um `X-Request-Id` enviado pelo cliente é ignorado, para evitar colisão ou sobrescrita de registros. Aceitar o identificador do cliente fica para o futuro.
- O registro e o trace ficam no mesmo banco local já usado pelo projeto, com o mesmo caminho configurável e banco em memória nos testes.
- Não há expiração automática de registros nesta feature. A política de retenção fica para o futuro.
- A consulta de registros segue o mesmo nível de acesso do chat hoje (sem autenticação própria), o que é adequado ao uso local e interno do copiloto.
- O conteúdo completo dos eventos (inclusive argumentos e resultados de ferramentas) é guardado no banco, porque é o que permite o diagnóstico. A restrição de "só metadados" vale para os logs.
- Os logs vão para a saída padrão do processo, uma linha por evento. A coleta e o envio para outro sistema ficam fora do escopo.
- Requisições rejeitadas antes de executar (400/422) não geram registro persistido, só log.
- Arena e bench não gravam registros nem logs desta feature, porque rodam fora do chat.
