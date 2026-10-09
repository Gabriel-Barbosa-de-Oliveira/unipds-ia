# Feature Specification: Resiliência de Modelo (retry + modelo de reserva)

**Feature Branch**: `013-model-resilience`

**Created**: 2026-10-09

**Status**: Draft

**Input**: User description: "Resiliência de modelo: .env: OPENROUTER_MODEL_FALLBACK; fábrica model.ts: withRetry no primário; withFallbacks([reserva]); Trace: evento \"fallback\"; metrics.modelUsed."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - O copiloto continua respondendo quando o modelo principal falha (Priority: P1)

A pessoa de plantão faz uma pergunta ao copiloto bem no momento em que o modelo principal está instável (limite de requisições, erro temporário do provedor, indisponibilidade). Em vez de receber um erro, ela recebe a resposta: o sistema tenta de novo o modelo principal algumas vezes e, se ele continuar falhando, passa a usar um modelo de reserva configurado.

**Why this priority**: é o núcleo da feature. O copiloto é usado durante incidentes, que é exatamente quando provedores costumam estar sobrecarregados. Uma falha transitória do modelo hoje vira erro para a pessoa de plantão.

**Independent Test**: dá para testar simulando um modelo principal que falha de forma transitória e confirmando que (a) uma falha seguida de sucesso é absorvida por nova tentativa no principal e (b) falhas persistentes resultam em resposta produzida pelo modelo de reserva.

**Acceptance Scenarios**:

1. **Given** um modelo principal que falha uma vez de forma transitória e depois responde, **When** o copiloto atende a pergunta, **Then** a resposta é produzida pelo modelo principal, sem uso da reserva.
2. **Given** um modelo principal que falha de forma transitória em todas as tentativas e um modelo de reserva configurado, **When** o copiloto atende a pergunta, **Then** a resposta é produzida pelo modelo de reserva e a pessoa recebe a resposta normalmente.
3. **Given** que o modelo principal e o de reserva falham, **When** o copiloto atende a pergunta, **Then** a pessoa recebe o mesmo tipo de erro que já recebe hoje para falhas do modelo, sem esperar indefinidamente.
4. **Given** que nenhum modelo de reserva está configurado, **When** o modelo principal falha em todas as tentativas, **Then** o comportamento é igual ao de hoje (erro), apenas depois das novas tentativas no principal.
5. **Given** qualquer etapa do copiloto que consulta o modelo (roteador, estratégias de raciocínio, crítico do reflection, planejador, refletor de aprendizado), **When** o modelo principal falha, **Then** essa etapa recebe a mesma proteção de novas tentativas e reserva.

---

### User Story 2 - Saber quando a reserva foi usada e qual modelo respondeu (Priority: P1)

Quem opera ou depura o copiloto quer ver, em cada resposta, qual modelo de fato a produziu e, quando houve troca para a reserva, um registro explícito no trace dizendo que houve fallback, de qual modelo para qual e por quê.

**Why this priority**: sem isso, a troca de modelo fica invisível. Respostas de qualidade diferente ou custos inesperados ficam impossíveis de explicar, e não há como saber com que frequência o principal está falhando.

**Independent Test**: dá para testar forçando o uso da reserva e verificando que a resposta informa o modelo usado e que o trace tem um evento de fallback; e, no caminho feliz, verificando que o modelo usado é o principal e que não há evento de fallback.

**Acceptance Scenarios**:

1. **Given** uma resposta produzida só pelo modelo principal, **When** as métricas são inspecionadas, **Then** o modelo usado é o principal e o trace não tem nenhum evento de fallback.
2. **Given** uma resposta em que alguma chamada ao modelo precisou da reserva, **When** o trace é inspecionado, **Then** há um evento do tipo "fallback" para cada troca, com o modelo de origem, o modelo de destino e o motivo (a falha do principal).
3. **Given** uma resposta em que a reserva foi usada, **When** as métricas são inspecionadas, **Then** o modelo usado informado é o de reserva.
4. **Given** um evento de fallback no trace, **When** o trace é formatado para leitura no terminal, **Then** a troca aparece numa linha legível.
5. **Given** que o modelo principal precisou de novas tentativas mas acabou respondendo, **When** o trace é inspecionado, **Then** não há evento de fallback (nova tentativa no mesmo modelo não é troca).

---

### User Story 3 - Configurar o modelo de reserva sem mudar código (Priority: P2)

Quem opera o copiloto quer definir qual é o modelo de reserva por configuração do ambiente, do mesmo jeito que já define o modelo principal, e poder desligar a reserva simplesmente não configurando-a.

**Why this priority**: a escolha da reserva muda com preço e disponibilidade dos provedores, e trocar deve ser barato. Mas o valor central (não falhar) já vem da US1 com qualquer reserva definida.

**Independent Test**: dá para testar configurando uma reserva e confirmando que ela é usada nas falhas; depois removendo a configuração e confirmando que o sistema volta a falhar como hoje após as novas tentativas.

**Acceptance Scenarios**:

1. **Given** uma reserva configurada no ambiente, **When** o principal falha persistentemente, **Then** a reserva configurada é usada.
2. **Given** nenhuma reserva configurada (ou configuração vazia), **When** o sistema inicia, **Then** ele funciona normalmente, só com novas tentativas no principal.
3. **Given** uma reserva configurada igual ao modelo principal, **When** o sistema roda, **Then** isso é tratado como "sem reserva" (não faz sentido cair no mesmo modelo).

---

### Edge Cases

- Falha não transitória do principal (ex.: requisição inválida, credencial inválida): o sistema não deve gastar novas tentativas no mesmo modelo, mas pode cair na reserva.
- Falha no meio de uma execução com várias chamadas ao modelo (ex.: plan-and-execute): só a chamada que falhou usa a reserva; as chamadas seguintes voltam a tentar o principal primeiro. O trace pode ter mais de um evento de fallback, e o modelo usado reflete o que produziu a resposta final.
- O roteador do grafo de produção (012) já tem fallback de rota próprio: se o roteador falhar mesmo depois da reserva de modelo, continua valendo o fallback de rota para `react`. O evento de fallback de modelo e a origem `fallback` do evento `route` são conceitos distintos e precisam ser distinguíveis no trace.
- Novas tentativas não podem fazer a requisição estourar o teto de tempo total já existente; se o teto vencer, a resposta é o timeout atual.
- A reserva também precisa conseguir produzir saída estruturada (roteador, crítico, planejador); se ela não conseguir, a falha é tratada como falha da reserva.
- Arena e bench, que rodam estratégias fora do `/chat`, também usam a fábrica de modelo e por isso ganham a mesma resiliência; o trace deles mostra o evento de fallback.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Toda chamada ao modelo feita pelo copiloto MUST passar pela mesma fábrica de modelo, que aplica novas tentativas no modelo principal e, se configurado, o modelo de reserva.
- **FR-002**: Falhas transitórias do modelo principal (limite de requisições, erro temporário do provedor, falha de rede, tempo esgotado da chamada) MUST ser tentadas de novo, com um número máximo pequeno e fixo de novas tentativas e espera crescente entre elas.
- **FR-003**: Falhas não transitórias do modelo principal MUST NOT ser tentadas de novo no mesmo modelo.
- **FR-004**: Quando o modelo principal esgota as tentativas (ou falha de forma não transitória) e há reserva configurada, a chamada MUST ser refeita no modelo de reserva.
- **FR-005**: O modelo de reserva MUST ser configurável por variável de ambiente própria, ao lado da que já define o modelo principal; ausente, vazia ou igual ao principal significa "sem reserva".
- **FR-006**: Sem reserva configurada, o comportamento final MUST ser igual ao de hoje (erro propagado), exceto pelas novas tentativas no principal.
- **FR-007**: Cada troca do principal para a reserva MUST gerar um evento de trace do tipo "fallback" com o modelo de origem, o modelo de destino e o motivo resumido da falha.
- **FR-008**: Novas tentativas no mesmo modelo MUST NOT gerar evento de fallback.
- **FR-009**: As métricas de toda resposta MUST incluir o modelo usado, que é o identificador do modelo que produziu a resposta final.
- **FR-010**: No grafo de produção (012), eventos de fallback MUST seguir as regras de trace já existentes: indicar o nó em que ocorreram e manter a ordem e a numeração sequencial.
- **FR-011**: A formatação legível de trace MUST suportar o evento de fallback.
- **FR-012**: Os segredos e a configuração de modelos MUST continuar vindo do ambiente do processo; o sistema MUST NOT ler o arquivo de configuração local diretamente.
- **FR-013**: A lógica de decisão (o que é falha transitória, se há reserva válida, montagem do evento de fallback, cálculo do modelo usado) MUST ser coberta por testes que não acessam rede.

### Key Entities

- **Configuração de modelos**: modelo principal (obrigatório, já existente) e modelo de reserva (opcional).
- **Evento de fallback**: registro no trace de uma troca de modelo, com o modelo de origem, o modelo de destino, o motivo e, no grafo de produção, o nó.
- **Modelo usado**: identificador do modelo que produziu a resposta final, informado nas métricas da resposta.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Com reserva configurada, 100% das perguntas feitas durante uma indisponibilidade simulada do modelo principal recebem resposta, em vez de erro.
- **SC-002**: Falhas transitórias isoladas do principal (uma falha seguida de sucesso) não aparecem para a pessoa de plantão: 100% delas resultam em resposta do próprio principal, sem evento de fallback.
- **SC-003**: 100% das respostas informam o modelo usado, e 100% das respostas em que houve troca de modelo têm pelo menos um evento de fallback no trace.
- **SC-004**: O tempo extra causado pelas novas tentativas no principal antes de cair na reserva fica abaixo de 10 segundos no pior caso.
- **SC-005**: No caminho feliz (principal responde de primeira), o número de chamadas ao modelo e o tempo de resposta não mudam em relação a hoje.
- **SC-006**: Trocar ou desligar o modelo de reserva exige só mudança de configuração do ambiente, sem alteração de código.

## Assumptions

- O modelo de reserva roda no mesmo provedor e com as mesmas credenciais do principal. Só o identificador do modelo muda.
- São 2 tentativas por modelo (1 tentativa + 1 retry), com espera exponencial curta, tanto no principal quanto na reserva. Se a reserva também falhar, o erro propaga. (Ajustado no `/speckit-implement`, seguindo o esboço da pessoa usuária.)
- Cada chamada ao modelo decide sozinha se cai na reserva. Não existe "disjuntor" que mantenha a reserva ligada entre chamadas ou requisições; isso fica para uma feature futura.
- O "modelo usado" é um único identificador: o do modelo que produziu a resposta final. O detalhe de trocas em chamadas intermediárias fica nos eventos de fallback do trace.
- O fallback de rota do roteador (012) não muda. O evento de fallback de modelo é um tipo de evento novo e separado.
- O teto de tempo total por requisição já existente (180 s) continua valendo e cobre as novas tentativas e a reserva.
- A variável de ambiente se chama `OPENROUTER_MODEL_FALLBACK`, seguindo o padrão de `OPENROUTER_MODEL`.
