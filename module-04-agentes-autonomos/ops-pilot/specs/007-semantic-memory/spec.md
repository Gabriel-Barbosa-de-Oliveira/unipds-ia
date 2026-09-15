# Feature Specification: Memória Semântica

**Feature Branch**: `007-semantic-memory`

**Created**: 2026-09-15

**Status**: Draft

**Input**: User description: "Memória semântica: MemoryStore por userId - remember (dedup > 0.92), recall top-3 por produto escalar (min 0.3), forget; tabela memories, embedding all-MiniLM-L6-v2 local em BLOB;
/chat ganha userId e injeta o recall no prompt; teste: recall acha fato sem palavra em comum.
user: @huggingface/transformers com pooling: mean + normalize: true e lazy singleton src/memory/embeddings.ts e src/memory/memory-store.ts.
As colunas de memories (id, user_id, fact, embedding, created_at)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - O copiloto lembra fatos contados antes, mesmo com outras palavras (Priority: P1)

A pessoa de plantão conta algo sobre si mesma ou sua forma de trabalhar (ex.: "eu sou o responsável pelo serviço de pagamentos") e, em uma pergunta futura — mesmo formulada com palavras completamente diferentes (ex.: "quem cuida do checkout financeiro?") — o copiloto reconhece que esse fato é relevante e usa essa informação na resposta, sem que a pessoa precise repeti-la.

**Why this priority**: é o motivo central da feature — memória que só reconhece a repetição exata das mesmas palavras não seria diferente de uma busca de texto simples, e não entregaria valor real num plantão onde a mesma informação é mencionada de formas variadas ao longo do tempo.

**Independent Test**: pode ser testado isoladamente registrando um fato com um conjunto de palavras, depois perguntando algo relacionado a esse fato usando um conjunto de palavras totalmente diferente (nenhuma palavra em comum), e confirmando que o fato é recuperado e influencia a resposta.

**Acceptance Scenarios**:

1. **Given** um fato registrado para uma pessoa, **When** essa pessoa faz uma pergunta relacionada ao fato usando palavras diferentes das usadas ao registrá-lo, **Then** o fato é recuperado e considerado na resposta.
2. **Given** nenhum fato relevante registrado para o assunto perguntado, **When** a pessoa faz uma pergunta, **Then** nenhum fato é injetado na resposta, e isso não é tratado como erro.
3. **Given** fatos registrados para duas pessoas diferentes, **When** uma delas faz uma pergunta, **Then** apenas os fatos dela próprios podem ser recuperados — nunca os de outra pessoa.

---

### User Story 2 - Não acumular fatos duplicados quando algo é repetido (Priority: P2)

Quando a pessoa de plantão menciona novamente algo que já havia contado antes — mesmo com palavras um pouco diferentes —, o copiloto reconhece que é essencialmente a mesma informação e não guarda um registro repetido.

**Why this priority**: depende de já existir um mecanismo de comparação de fatos (User Story 1), mas é o que mantém a memória útil e enxuta ao longo do tempo — sem isso, a mesma informação se acumularia indefinidamente e prejudicaria a qualidade do que é recuperado depois.

**Independent Test**: pode ser testado isoladamente registrando um fato, depois tentando registrar essencialmente a mesma informação com palavras quase idênticas, e confirmando que apenas um registro existe ao final.

**Acceptance Scenarios**:

1. **Given** um fato já registrado, **When** uma informação essencialmente igual é registrada novamente, **Then** nenhum registro novo é criado.
2. **Given** um fato já registrado, **When** uma informação relacionada mas genuinamente diferente é registrada, **Then** um novo registro é criado normalmente.

---

### User Story 3 - Pedir para o copiloto esquecer um fato específico (Priority: P3)

A pessoa de plantão pede para o copiloto esquecer algo que contou antes (ex.: porque deixou de ser verdade), descrevendo o que deve ser esquecido, e o copiloto remove o fato correspondente da memória.

**Why this priority**: depende de já haver fatos registrados (User Story 1), mas é o que dá controle sobre a própria memória — sem isso, uma informação desatualizada ficaria influenciando respostas futuras indefinidamente.

**Independent Test**: pode ser testado isoladamente registrando um fato, pedindo para esquecê-lo descrevendo-o, e confirmando que uma pergunta relacionada deixa de recuperar esse fato.

**Acceptance Scenarios**:

1. **Given** um fato registrado, **When** a pessoa pede para esquecer esse fato (descrevendo-o), **Then** o fato correspondente é removido e não é mais recuperado em perguntas futuras.
2. **Given** um pedido para esquecer algo que não corresponde a nenhum fato registrado com confiança suficiente, **When** o pedido é feito, **Then** o sistema informa que não encontrou um fato correspondente, em vez de remover algo por engano.

---

### Edge Cases

- O que acontece quando a pessoa de plantão ainda não tem nenhum fato registrado e faz uma pergunta qualquer?
- O que acontece quando duas informações são parecidas o suficiente para ficar na fronteira entre "duplicata" e "fato novo"?
- O que acontece quando um pedido de esquecer é ambíguo o suficiente para corresponder a mais de um fato registrado?
- O que acontece com a memória de uma pessoa quando o processo da aplicação é reiniciado?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema MUST permitir registrar um fato associado a uma pessoa de plantão específica, de forma que sobreviva a reinícios do processo da aplicação.
- **FR-002**: O sistema MUST reconhecer quando duas formulações diferentes descrevem essencialmente a mesma informação, com base no significado, não apenas nas palavras usadas.
- **FR-003**: Antes de registrar um fato novo, o sistema MUST verificar se já existe um fato suficientemente parecido (mesmo significado) para essa mesma pessoa e, nesse caso, MUST NOT criar um registro duplicado.
- **FR-004**: O sistema MUST permitir recuperar, para uma pergunta específica de uma pessoa, os fatos previamente registrados dessa mesma pessoa que sejam relevantes para o assunto perguntado — mesmo quando a pergunta não compartilha nenhuma palavra com o fato registrado.
- **FR-005**: A recuperação de fatos MUST retornar no máximo 3 fatos por pergunta, e MUST omitir qualquer fato cuja relevância para a pergunta seja considerada baixa demais para ser útil.
- **FR-006**: Quando fatos relevantes são recuperados para uma pergunta, o sistema MUST considerá-los ao formular a resposta.
- **FR-007**: Os fatos de uma pessoa MUST permanecer isolados dos de qualquer outra pessoa — nenhuma pergunta de uma pessoa MUST recuperar fatos registrados por outra.
- **FR-008**: O sistema MUST permitir que a pessoa de plantão peça para esquecer um fato específico, identificando-o pela descrição do que deve ser esquecido; um fato removido MUST deixar de ser recuperável em perguntas futuras.
- **FR-009**: Quando um pedido de esquecer não corresponde com confiança suficiente a nenhum fato registrado, o sistema MUST informar isso claramente em vez de remover um fato por engano.
- **FR-010**: A pessoa de plantão à qual um fato pertence MUST ser identificada de forma explícita em cada interação com o copiloto que envolva registrar ou recuperar memória.

### Key Entities

- **Fato (memória)**: uma informação registrada sobre uma pessoa de plantão específica, expressa em linguagem natural; pertence a exatamente uma pessoa; comparável a outros fatos por significado (não apenas por texto), tanto para evitar duplicatas quanto para ser recuperado por perguntas relacionadas.
- **Pessoa de plantão (dona da memória)**: identifica a quem um fato pertence; nenhum fato é compartilhado entre pessoas diferentes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Uma pergunta formulada sem nenhuma palavra em comum com um fato registrado anteriormente ainda assim recupera esse fato quando ele é relevante para a pergunta.
- **SC-002**: Repetir essencialmente a mesma informação múltiplas vezes, com formulações diferentes, nunca resulta em mais de um fato armazenado para ela.
- **SC-003**: Uma pergunta cujo assunto não tem nenhum fato relevante registrado nunca recebe fatos irrelevantes injetados na resposta.
- **SC-004**: Um fato removido a pedido da pessoa de plantão nunca mais aparece em respostas futuras dela.
- **SC-005**: Fatos de uma pessoa nunca aparecem nas respostas dadas a outra pessoa, mesmo quando ambas perguntam sobre assuntos semelhantes.

## Assumptions

- Identificar a pessoa de plantão (`userId`) é um campo adicional e opcional na conversa com o copiloto — quando omitido, nenhuma memória é registrada ou recuperada, e o comportamento existente do copiloto permanece inalterado (extensão aditiva sobre o contrato já existente, mesmo espírito da feature `006-conversation-history`).
- "Suficientemente parecido" (duplicata, FR-003) e "relevância baixa demais" (FR-005) são limiares de similaridade de significado definidos internamente pelo sistema — o pedido original já fixa esses limiares (deduplicação acima de 0.92; recuperação a partir de 0.3, numa escala onde 1.0 é a maior similaridade possível), preservados aqui como a calibração inicial.
- Identificar qual fato remover em um pedido de "esquecer" (FR-008/FR-009) usa o mesmo mecanismo de comparação por significado da recuperação (User Story 1) — o fato mais parecido com a descrição dada, desde que a semelhança seja alta o suficiente para reduzir o risco de remover o fato errado.
- Memória semântica é um mecanismo independente do histórico de conversa já existente (`006-conversation-history`): um fato registrado permanece disponível entre conversas diferentes da mesma pessoa, não apenas dentro de uma única conversa.
- Não há um limite explícito de quantos fatos uma pessoa pode acumular ao longo do tempo — está fora do escopo desta feature.
- Não há autenticação ou verificação de identidade além do identificador informado — confiar que quem envia um `userId` é de fato essa pessoa está fora do escopo desta feature, mesmo padrão já assumido pelas features anteriores para o copiloto como um todo.
