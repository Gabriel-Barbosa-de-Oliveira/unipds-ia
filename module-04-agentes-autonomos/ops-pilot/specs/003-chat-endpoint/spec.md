# Feature Specification: Endpoint HTTP de Chat

**Feature Branch**: `003-chat-endpoint`

**Created**: 2026-09-03

**Status**: Draft

**Input**: User description: "POST /chat em src/http/server.ts ou (padrão do express) : body { message, strategy?, reflect?} validado com zod; default react. 200 { answer, trace, metrics}; 400 body invalido (issues do zodiac); 422 estrategia desconhecido; timeout 180s -> 504. Registry em src/agents/index.ts (nome -> estrategia; reflect aplica withReflection). Teste de integração com estrategia fake determinista, sem rede"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Obter uma resposta via API para uma pergunta operacional (Priority: P1)

Um cliente da API (dashboard interno, CLI, outro serviço) envia uma pergunta operacional em texto livre e recebe de volta a resposta final do copiloto, sem precisar saber qual estratégia de raciocínio está sendo usada por baixo — o sistema aplica a estratégia padrão automaticamente.

**Why this priority**: É o caminho de valor central do endpoint — sem ele, não existe forma de consumir o agente via HTTP. Todo o resto (escolha de estratégia, reflection, tratamento de erros) é refinamento sobre este fluxo básico.

**Independent Test**: Pode ser testado isoladamente enviando uma requisição com apenas o campo de mensagem preenchido e verificando que a resposta contém a resposta final, o histórico de raciocínio (trace) e as métricas de execução, usando a estratégia padrão.

**Acceptance Scenarios**:

1. **Given** o endpoint disponível, **When** um cliente envia uma requisição contendo apenas a pergunta operacional, **Then** o sistema executa a estratégia de raciocínio padrão e responde com sucesso contendo resposta final, trace e métricas.
2. **Given** uma resposta bem-sucedida, **When** o cliente inspeciona o corpo da resposta, **Then** encontra a resposta em texto, a sequência de eventos que levou a ela, e métricas quantificando o custo da execução (ao menos número de chamadas ao modelo e tempo decorrido).

---

### User Story 2 - Escolher explicitamente a estratégia de raciocínio (Priority: P2)

Um cliente que já sabe qual estratégia de raciocínio quer usar (por exemplo, para comparar comportamento ou por já ter validado uma estratégia específica para seu caso de uso) informa o nome da estratégia na requisição e recebe a resposta produzida por ela.

**Why this priority**: Depende do fluxo básico da User Story 1 já funcionar, mas é o que torna o endpoint útil para além do caso trivial — sem ele, clientes ficam presos à estratégia padrão mesmo quando uma alternativa já disponível no sistema seria mais adequada.

**Independent Test**: Pode ser testado isoladamente enviando a mesma pergunta duas vezes, cada vez informando uma estratégia diferente dentre as já suportadas pelo sistema, e verificando que a resposta de cada requisição reflete a estratégia solicitada.

**Acceptance Scenarios**:

1. **Given** o endpoint disponível, **When** um cliente informa o nome de uma estratégia suportada pelo sistema, **Then** o sistema executa essa estratégia (e não a padrão) e responde com o resultado dela.
2. **Given** um cliente que informa o nome de uma estratégia que não corresponde a nenhuma estratégia suportada, **When** a requisição é processada, **Then** o sistema não executa nenhum raciocínio e responde com um erro que identifica que a estratégia é desconhecida.

---

### User Story 3 - Pedir uma resposta autocriticada via reflection (Priority: P2)

Um cliente que precisa de mais confiança na resposta final (por exemplo, para decisões operacionais mais sensíveis) sinaliza que quer a camada de reflection ativada, e a resposta retornada já passou pelo ciclo de autocrítica e regeneração antes de ser considerada final.

**Why this priority**: Depende das User Stories 1 e 2 (execução básica e seleção de estratégia), mas expõe uma capacidade já existente no sistema (a camada de reflection) através da API, o que amplia o valor do endpoint sem introduzir lógica de raciocínio nova.

**Independent Test**: Pode ser testado isoladamente enviando a mesma pergunta com e sem o sinalizador de reflection ativado (usando a mesma estratégia base) e verificando que, com reflection ativado, o trace retornado contém evidência do ciclo de crítica e as métricas refletem o custo adicional.

**Acceptance Scenarios**:

1. **Given** um cliente que ativa reflection em conjunto com uma estratégia (padrão ou explícita), **When** a requisição é processada, **Then** o sistema executa essa estratégia envolvida pela camada de reflection e responde com o resultado já autocriticado.
2. **Given** reflection ativado, **When** o cliente compara as métricas da resposta com uma execução equivalente sem reflection, **Then** as métricas com reflection refletem as chamadas adicionais feitas pelo ciclo de crítica.

---

### User Story 4 - Confiar que a requisição nunca fica pendente indefinidamente (Priority: P3)

Um cliente que integra o endpoint em um fluxo automatizado confia que uma execução que está demorando demais (por exemplo, por lentidão do modelo subjacente) não deixa a conexão aberta para sempre — depois de um tempo máximo, ele recebe um erro claro e pode decidir como reagir (repetir, alertar, etc.).

**Why this priority**: É uma garantia de previsibilidade operacional sobre as demais histórias. O endpoint já entrega valor sem ela em uso manual e supervisionado, mas ela é necessária antes de confiar o endpoint a qualquer chamada automatizada.

**Independent Test**: Pode ser testado isoladamente configurando uma execução que ultrapassa o tempo máximo permitido e verificando que o cliente recebe um erro de tempo excedido dentro de uma margem razoável após esse limite, em vez de a conexão ficar pendente.

**Acceptance Scenarios**:

1. **Given** uma requisição em processamento, **When** o tempo de execução ultrapassa o limite máximo permitido, **Then** o sistema responde com um erro de tempo excedido e encerra a execução, em vez de manter o cliente esperando indefinidamente.

---

### Edge Cases

- O que acontece quando o corpo da requisição não é um JSON válido, ou está ausente por completo?
- O que acontece quando o campo de mensagem está presente mas vazio, ou tem um tipo diferente de texto?
- O que acontece quando o cliente informa `reflect` junto com uma estratégia desconhecida — o erro de estratégia desconhecida deve ser identificado antes de qualquer tentativa de execução?
- O que acontece quando o modelo subjacente falha por um motivo que não é tempo excedido (ex.: erro de comunicação com o provedor)?
- O que acontece quando dois clientes fazem requisições simultâneas — cada execução deve ser isolada, sem misturar trace ou métricas de uma requisição com as de outra?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema MUST expor um endpoint HTTP que aceita requisições contendo uma pergunta operacional em texto (campo de mensagem obrigatório) e, opcionalmente, o nome de uma estratégia de raciocínio e um sinalizador para ativar reflection.
- **FR-002**: O sistema MUST validar o corpo da requisição antes de iniciar qualquer execução; requisições com corpo inválido (campo de mensagem ausente, vazio ou de tipo incorreto, ou corpo mal formado) MUST ser rejeitadas com um erro que identifica especificamente o(s) campo(s) inválido(s), sem executar nenhum raciocínio.
- **FR-003**: Quando o nome da estratégia não é informado, o sistema MUST usar a estratégia padrão do sistema (ReAct).
- **FR-004**: Quando o nome da estratégia informado não corresponde a nenhuma estratégia conhecida pelo sistema, o sistema MUST rejeitar a requisição com um erro que identifica que a estratégia é desconhecida, sem executar nenhum raciocínio.
- **FR-005**: O sistema MUST manter uma correspondência central entre nomes de estratégia e as estratégias de raciocínio executáveis, de forma que novas estratégias possam se tornar acessíveis pelo endpoint sem exigir mudanças na lógica de validação ou resposta do endpoint.
- **FR-006**: Quando o cliente ativa o sinalizador de reflection, o sistema MUST executar a estratégia selecionada (padrão ou explícita) envolvida pela camada de autocrítica e regeneração já existente no sistema, em vez da estratégia base isolada.
- **FR-007**: Em uma execução bem-sucedida, o sistema MUST responder com a resposta final, o histórico completo de eventos de raciocínio (trace) e as métricas de execução (incluindo ao menos número de chamadas ao modelo e tempo decorrido).
- **FR-008**: O sistema MUST impor um tempo máximo de execução por requisição de 180 segundos; ao ultrapassar esse limite, MUST encerrar a execução e responder ao cliente com um erro de tempo excedido, em vez de manter a conexão pendente.
- **FR-009**: Erros de validação de corpo, de estratégia desconhecida e de tempo excedido MUST resultar em códigos de erro distintos entre si, permitindo que o cliente diferencie programaticamente a causa da falha.
- **FR-010**: Cada requisição MUST ser processada de forma isolada — o trace e as métricas retornados correspondem exclusivamente à execução daquela requisição, mesmo sob requisições concorrentes.

### Key Entities

- **Requisição de Chat**: representa o pedido de um cliente ao endpoint; contém a pergunta operacional em texto, o nome opcional da estratégia de raciocínio desejada, e um sinalizador opcional indicando se a camada de reflection deve ser aplicada.
- **Resposta de Chat**: representa o resultado de uma execução bem-sucedida; contém a resposta final em texto, o trace (sequência ordenada de eventos que documentam o raciocínio) e as métricas da execução (ao menos chamadas ao modelo e latência).
- **Estratégia de Raciocínio Registrada**: representa uma estratégia de raciocínio conhecida pelo sistema e acessível via nome através do endpoint; inclui tanto as estratégias base quanto suas variantes decoradas com reflection.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Um cliente consegue obter uma resposta completa (resposta final, trace e métricas) para uma pergunta operacional válida enviando apenas a mensagem, sem precisar conhecer ou informar nenhum outro parâmetro.
- **SC-002**: 100% das requisições com corpo de mensagem inválido recebem um erro que identifica exatamente qual campo falhou, sem que nenhuma execução de raciocínio seja iniciada.
- **SC-003**: 100% das requisições que referenciam uma estratégia inexistente recebem um erro distinguível dos demais tipos de erro, sem que nenhuma execução de raciocínio seja iniciada.
- **SC-004**: Nenhuma requisição permanece pendente além do tempo máximo configurado; ao ultrapassá-lo, o cliente recebe uma resposta de erro de tempo excedido.
- **SC-005**: Um cliente consegue comparar, para a mesma pergunta e estratégia base, o custo adicional (em chamadas ao modelo) de ativar reflection frente a não ativar, apenas inspecionando as métricas retornadas pelas duas execuções.

## Assumptions

- O endpoint é consumido por clientes internos/confiáveis (dashboard, CLI, outros serviços do próprio ecossistema OpsPilot); autenticação/autorização do chamador está fora do escopo desta feature.
- "Estratégia desconhecida" e "corpo inválido" são tratados como categorias de erro distintas porque têm causas diferentes (requisição estruturalmente inválida vs. requisição bem formada mas referenciando um recurso inexistente) — cada uma recebe um código de erro HTTP diferente, seguindo convenções REST padrão.
- O conjunto de estratégias suportadas (e suas variantes com reflection) já existe no sistema antes desta feature; esta feature apenas as expõe via HTTP, sem adicionar nenhuma estratégia de raciocínio nova.
- Falhas do provedor do modelo subjacente que não sejam tempo excedido (ex.: erro de rede/API do provedor) são tratadas como erro genérico de execução, distinto do erro de tempo excedido e dos erros de validação de entrada.
- Os testes automatizados desta feature (incluindo o teste de integração do endpoint) rodam contra uma estratégia de teste determinística registrada apenas em ambiente de teste, sem depender de chamadas de rede a um provedor de modelo real — consistente com o princípio de Teste Obrigatório da constitution do projeto.
