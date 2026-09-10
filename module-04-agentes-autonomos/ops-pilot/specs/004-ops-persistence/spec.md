# Feature Specification: Persistência Real de Operações

**Feature Branch**: `004-ops-persistence`

**Created**: 2026-09-10

**Status**: Draft

**Input**: User description: "Persistencia real de operações
- SqliteOpsStore (src/store/sqlite-ops-store.ts) implementa a interface OpsStore existente via node:sqlite (DatabaseSync): caminho em OPSPILOT_DB (default ./data/opspilot.db); \":memory:\" nos testes
- 4 tabelas - services; alerts; incidents; runbooks - espelhando os tipos atuais do domain (incidentes ganham resolved_at e summary, anuláveis); DDL idempotente no construtor; CHECK em todo campo de domínio fechado (tier, severity, status)
- Seed idempotente = cenário Mercadinho do mock (5 serviços, 6 alertas: 3 firing, 3 resolved, runbooks de checkout/payments/auth)
- Prepared statements em toda query; sem SQL concatenado
- Tools novas: list_incidents (status open | resolved | all) e consultar_runbook(service) - descrições pelas 6 regras
- Composição injeta SqliteOpsStore; mock in memory fica para testes e para o bench (cenários possam ser reproduzidos)
- data/ no .gitignore
- Revisar descrições src/agents/tools.ts pelas 6 regras (divida do open_incident: quando usar; describe() em todo campo; enums)
- Testes: \":memory:\" seed, abrir/listar/resolver, filtros e CHECKs; testes das tools existentes passam a rodar sobre \":memory:\""

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Dados operacionais sobrevivem a reinícios do processo (Priority: P1)

A equipe de plantão confia que o histórico de serviços, alertas e incidentes não desaparece quando o processo da API é reiniciado (deploy, crash, restart manual) — a persistência é real, não um estado que existe apenas enquanto o processo está de pé.

**Why this priority**: é o motivo central desta feature — sem persistência durável, nenhuma das capacidades abaixo (consultar histórico, confiar no status de um incidente, comparar execuções) tem valor real fora de uma sessão isolada.

**Independent Test**: pode ser testado isoladamente abrindo um incidente, resolvendo-o, reiniciando o processo da aplicação, e verificando que o incidente e seu status atual continuam consultáveis exatamente como estavam antes do reinício.

**Acceptance Scenarios**:

1. **Given** serviços, alertas e incidentes registrados, **When** o processo da API é reiniciado, **Then** os mesmos dados continuam disponíveis sem exigir nenhum passo manual de reconfiguração.
2. **Given** um incidente aberto e depois resolvido, **When** o processo reinicia, **Then** o incidente aparece como resolvido, com o horário de resolução preservado.

---

### User Story 2 - Consultar incidentes existentes filtrando por status (Priority: P2)

Durante um plantão, a pessoa de plantão pergunta ao copiloto quais incidentes estão abertos (ou já resolvidos, ou todos) para priorizar o que precisa de atenção, sem precisar lembrar ids ou vasculhar canais separados.

**Why this priority**: depende da persistência durável da User Story 1, mas é o que dá visibilidade sobre o que o próprio sistema já gerencia — hoje o copiloto lista alertas, mas não os incidentes que ele mesmo abre e resolve.

**Independent Test**: pode ser testado isoladamente abrindo dois incidentes, resolvendo um deles, e pedindo a lista filtrando por "abertos", depois por "resolvidos", depois por "todos", verificando que cada resposta reflete exatamente o filtro pedido.

**Acceptance Scenarios**:

1. **Given** incidentes abertos e resolvidos, **When** a pessoa de plantão pede a lista de incidentes abertos, **Then** recebe somente os incidentes com status aberto.
2. **Given** os mesmos incidentes, **When** pede a lista de resolvidos, **Then** recebe somente os resolvidos; **When** pede "todos", **Then** recebe o conjunto completo.
3. **Given** nenhum incidente aberto no momento, **When** pede os abertos, **Then** recebe uma lista vazia, não um erro.

---

### User Story 3 - Consultar o runbook de um serviço durante um incidente (Priority: P2)

Ao lidar com um alerta ou incidente de um serviço específico, a pessoa de plantão pede ao copiloto o runbook daquele serviço para saber os passos de mitigação recomendados, sem sair da conversa para procurar em outro lugar.

**Why this priority**: depende de já haver um serviço identificado (geralmente via alerta ou incidente já visível), mas é o que conecta o copiloto ao conhecimento operacional acumulado da equipe.

**Independent Test**: pode ser testado isoladamente pedindo o runbook de um serviço com runbook cadastrado e verificando que o conteúdo retornado é o daquele serviço; e pedindo o runbook de um serviço sem runbook cadastrado, verificando que a ausência é sinalizada claramente.

**Acceptance Scenarios**:

1. **Given** um serviço com runbook cadastrado, **When** a pessoa de plantão pede o runbook desse serviço, **Then** recebe o conteúdo do runbook correspondente.
2. **Given** um serviço existente sem runbook cadastrado, **When** pede o runbook desse serviço, **Then** recebe uma resposta que deixa claro que não há runbook, e não um erro genérico.
3. **Given** um nome de serviço que não existe no sistema, **When** pede o runbook, **Then** recebe um erro que identifica que o serviço é desconhecido.

---

### User Story 4 - Restaurar um cenário operacional conhecido de forma reprodutível (Priority: P3)

A equipe que testa o copiloto (manualmente, em testes automatizados, ou comparando estratégias de raciocínio) restaura os dados operacionais para o mesmo cenário canônico sempre que precisa, garantindo que execuções diferentes sejam comparáveis entre si.

**Why this priority**: não é usado durante um plantão real; sustenta a confiabilidade do desenvolvimento e da avaliação do próprio copiloto. As stories anteriores já entregam valor de plantão sem ela.

**Independent Test**: pode ser testado isoladamente executando o processo de restauração do cenário duas vezes seguidas, com incidentes criados entre as duas execuções, e verificando que o resultado final (serviços, alertas, runbooks) é idêntico nas duas vezes e nenhum incidente sobrevive à restauração.

**Acceptance Scenarios**:

1. **Given** um estado qualquer, incluindo incidentes criados durante testes, **When** o cenário canônico é restaurado, **Then** o conjunto de serviços, alertas e runbooks volta a ser exatamente o conjunto canônico, e nenhum incidente de execuções anteriores permanece.
2. **Given** o processo de restauração executado repetidamente, **When** o resultado é comparado entre execuções, **Then** o resultado é idêntico (mesmo conjunto, mesmos identificadores), sem registros duplicados.

---

### Edge Cases

- O que acontece quando alguém tenta registrar um incidente, alerta ou serviço com um valor fora do conjunto de valores conhecidos para um campo de domínio fechado (ex.: severidade, status), inclusive por um caminho que não passe pela validação de entrada usual?
- O que acontece quando dois pedidos concorrentes tentam resolver o mesmo incidente ao mesmo tempo?
- O que acontece quando a lista de incidentes é pedida antes de qualquer incidente ter sido aberto?
- O que acontece na primeira execução do sistema, antes de o armazenamento de dados ter sido inicializado alguma vez?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema MUST persistir de forma durável serviços, alertas, incidentes e runbooks, de modo que os dados sobrevivam a reinícios do processo da aplicação.
- **FR-002**: O sistema MUST inicializar automaticamente a estrutura de armazenamento necessária na primeira execução, sem exigir passo manual de configuração por parte de quem opera o sistema.
- **FR-003**: O sistema MUST expor, através do copiloto, uma forma de listar incidentes filtrando por status (abertos, resolvidos, ou todos), retornando lista vazia — não erro — quando não houver correspondência.
- **FR-004**: O sistema MUST expor, através do copiloto, uma forma de consultar o runbook associado a um serviço a partir do nome do serviço.
- **FR-005**: Quando o serviço informado na consulta de runbook não existe no sistema, o sistema MUST responder com um erro que identifica que o serviço é desconhecido; quando o serviço existe mas não possui runbook cadastrado, o sistema MUST responder de um jeito que essa ausência seja distinguível de um erro.
- **FR-006**: O sistema MUST recusar, na camada de armazenamento, a gravação de qualquer registro cujo campo de valor fechado (ex.: severidade de incidente, status de incidente/alerta) esteja fora do conjunto de valores válidos do domínio — mesmo que a tentativa de gravação venha de um caminho que não passe pela validação de entrada usual.
- **FR-007**: O sistema MUST fornecer um processo de restauração para um cenário operacional canônico (mesmo conjunto de serviços, alertas e runbooks) que produz exatamente o mesmo resultado toda vez que é executado, independentemente de incidentes criados desde a última restauração.
- **FR-008**: As ferramentas do copiloto que operam sobre dados persistidos (listar alertas, abrir incidente, resolver incidente, listar incidentes, consultar runbook) MUST ter suas descrições revisadas para deixar explícito, para o modelo por trás do copiloto, quando cada uma deve ser usada em relação às demais — em particular, distinguir quando abrir um novo incidente é apropriado versus quando apenas consultar incidentes ou runbooks já existentes é o que foi pedido.
- **FR-009**: Todo campo de entrada de cada ferramenta do copiloto MUST ter uma descrição própria explicando seu significado, e todo campo cujo valor pertence a um conjunto fechado e conhecido MUST ser restrito a esse conjunto na própria definição da ferramenta, não apenas documentado em texto livre.
- **FR-010**: Os testes automatizados desta feature (armazenamento, cenário canônico, ferramentas do copiloto que dependem de dados persistidos) MUST rodar isolados uns dos outros, sem depender de nenhum arquivo ou serviço de armazenamento compartilhado no ambiente onde rodam.
- **FR-011**: O sistema MUST continuar aceitando as operações já existentes sobre alertas e incidentes (listar alertas, abrir incidente, resolver incidente, incluindo o comportamento idempotente de resolver um incidente já resolvido) sem alteração de comportamento observável para quem consome o copiloto.

### Key Entities

- **Serviço**: unidade monitorada (ex.: checkout, pagamentos, autenticação); identificado por um nome único; pode ter um runbook associado.
- **Alerta**: sinal de monitoramento vinculado a um serviço, com um status pertencente a um conjunto fechado (disparado/resolvido).
- **Incidente**: registro de um problema aberto para um serviço, com severidade (conjunto fechado de níveis) e status (aberto/resolvido); ao ser resolvido, guarda o momento da resolução e, opcionalmente, um resumo do que foi feito.
- **Runbook**: conjunto de passos de mitigação recomendados associado a um serviço; nem todo serviço tem um runbook cadastrado.
- **Cenário Operacional Canônico**: conjunto fixo e conhecido de serviços, alertas e runbooks usado como ponto de partida reproduzível para testes, demonstrações e comparação de estratégias de raciocínio.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Um incidente aberto e depois resolvido continua consultável com o status e o horário de resolução corretos após um reinício completo do processo da aplicação.
- **SC-002**: Uma pessoa de plantão obtém a lista de incidentes abertos, resolvidos, ou todos, em uma única pergunta ao copiloto, sem precisar saber identificadores específicos.
- **SC-003**: Uma pessoa de plantão obtém o runbook de um serviço específico dentro da própria conversa com o copiloto, sem precisar sair para procurar em outro lugar.
- **SC-004**: 100% das tentativas de gravar um valor fora do conjunto de valores válidos para um campo fechado do domínio são recusadas antes de se tornarem parte dos dados persistidos, independentemente do caminho de código que as originou.
- **SC-005**: Restaurar o cenário canônico duas vezes seguidas produz exatamente o mesmo conjunto de dados nas duas vezes, permitindo comparar execuções de teste ou benchmark entre si.
- **SC-006**: A suíte de testes automatizados da feature roda do início ao fim sem exigir nenhum arquivo ou serviço de armazenamento pré-existente no ambiente, e sem que uma execução deixe resíduo que afete a próxima.

## Assumptions

- O "cenário operacional canônico" reaproveita o dataset já usado hoje pelo sistema (5 serviços, 6 alertas: 3 disparados e 3 resolvidos), adicionando runbooks para os serviços de checkout, pagamentos e autenticação; os demais serviços permanecem sem runbook cadastrado.
- Nem todo incidente resolvido tem um resumo — é um campo opcional; sua ausência não é tratada como erro.
- A camada de persistência é interna ao processo da aplicação (armazenamento local gerenciado pela própria aplicação), sem exigir um serviço de banco de dados externo a ser instalado ou administrado separadamente, consistente com a stack tecnológica já definida pela constitution do projeto.
- Consultas ao runbook e à lista de incidentes não exigem autenticação ou autorização além da já aplicada ao copiloto como um todo — está fora do escopo desta feature.
- Quando duas requisições tentam resolver o mesmo incidente ao mesmo tempo, uma delas prevalece e a outra observa o incidente já resolvido (comportamento idempotente já existente), sem erro nem estado inconsistente.
- Os cenários de teste e de benchmark usados no dia a dia de desenvolvimento continuam podendo rodar sobre um armazenamento apenas em memória, sem tocar no armazenamento durável usado em execução real.
