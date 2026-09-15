# Feature Specification: MCP Server para OpsPilot

**Feature Branch**: `005-mcp-server`

**Created**: 2026-09-15

**Status**: Draft

**Input**: User description: "MCP server do OpsPilot: src/mcp/server.ts com @modelcontextprotocol/sdk, transport stdio, expondo list_alerts, open_incident e resolve_incident - reutilizando o mesmo OpsStore e os mesmos schemas zod das tools existentes (uma unica fonte de verdade). Nome do server: opspilot. Script npm: mcp = \"tsx src/mcp/server.ts\" (se precisar de env, alterar o script e carregar elas antes). REGRA CRÍTICA: nenhum console.log no server - no stdio o stdout é o canal do protocolo; diagnóstico vai para o stderr. Test: sobre o server e valida o list de tools"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Consultar alertas via cliente MCP (Priority: P1)

Um cliente MCP (ex.: um assistente de IA ou ferramenta de agente conectada via stdio) se conecta
ao servidor OpsPilot e consulta os alertas de monitoramento atuais, com a opção de filtrar por
status, exatamente como já é possível fazer hoje pelo chat do copiloto.

**Why this priority**: É a operação mais básica e somente-leitura; sem ela nenhuma outra
interação via MCP faz sentido, e é a mais simples de entregar e validar isoladamente.

**Independent Test**: Conectar um cliente MCP ao servidor, chamar a tool `list_alerts` (com e
sem filtro de status) e confirmar que o resultado bate com o que `list_alerts` já retorna hoje
pelo agente do chat, para o mesmo estado do banco.

**Acceptance Scenarios**:

1. **Given** o servidor MCP rodando e alertas existentes no banco, **When** o cliente chama
   `list_alerts` sem argumentos, **Then** recebe a lista completa de alertas.
2. **Given** o servidor MCP rodando, **When** o cliente chama `list_alerts` com
   `status: "firing"`, **Then** recebe somente os alertas com esse status.

---

### User Story 2 - Abrir um incidente via cliente MCP (Priority: P2)

Um cliente MCP registra um novo incidente para um serviço, informando título, serviço e
severidade, e recebe de volta o incidente criado — a mesma operação que hoje só existe pelo chat
do copiloto.

**Why this priority**: É a principal ação de escrita do plantão e o motivo prático de existir um
servidor MCP (permitir que outras ferramentas/agentes abram incidentes), mas depende da leitura
(US1) já existir como referência de comportamento esperado.

**Independent Test**: Com o servidor MCP rodando, chamar `open_incident` com dados válidos e
confirmar que o incidente retornado aparece depois em `list_alerts`/no restante do sistema (ex.:
via `list_incidents` do chat ou consulta direta ao banco) com os mesmos dados.

**Acceptance Scenarios**:

1. **Given** um serviço conhecido, **When** o cliente chama `open_incident` com título, serviço e
   severidade válidos, **Then** recebe o incidente criado (com id) e ele passa a existir no
   mesmo armazenamento usado pelo restante do OpsPilot.
2. **Given** um serviço que não existe, **When** o cliente chama `open_incident` para esse
   serviço, **Then** recebe um erro estruturado indicando o problema, sem o processo do servidor
   travar ou encerrar.

---

### User Story 3 - Resolver um incidente via cliente MCP (Priority: P3)

Um cliente MCP resolve um incidente já aberto (pelo próprio MCP ou pelo chat), informando o id e,
opcionalmente, um resumo do que foi feito.

**Why this priority**: Fecha o ciclo de vida do incidente, mas só tem valor depois que
abrir/listar incidentes já funciona (US1/US2) — sem essas, não há como obter um id válido para
testar.

**Independent Test**: Abrir um incidente (via US2 ou diretamente no store), chamar
`resolve_incident` com o id retornado e confirmar que o incidente passa a aparecer como
resolvido nas demais interfaces do OpsPilot.

**Acceptance Scenarios**:

1. **Given** um incidente aberto, **When** o cliente chama `resolve_incident` com o id e um
   resumo, **Then** recebe o incidente atualizado com status resolvido e o resumo salvo.
2. **Given** um id de incidente inexistente, **When** o cliente chama `resolve_incident` com
   esse id, **Then** recebe um erro estruturado indicando que o incidente não foi encontrado,
   sem o processo do servidor travar ou encerrar.

---

### Edge Cases

- O que acontece se o cliente enviar argumentos que não batem com o schema esperado de uma tool
  (ex.: severidade fora do enum, título vazio)? O servidor deve rejeitar a chamada de forma
  estruturada, sem travar o processo.
- O que acontece se qualquer código do servidor tentar escrever no stdout (ex.: um
  `console.log` esquecido ou de uma dependência)? Isso corrompe o canal do protocolo MCP para
  todas as chamadas seguintes — é a falha mais crítica possível desta feature.
- O que acontece se a variável de ambiente do caminho do banco (`OPSPILOT_DB`) não estiver
  configurada quando o servidor sobe? Deve seguir o mesmo comportamento padrão já usado pelas
  outras interfaces do OpsPilot (mesmo arquivo/local padrão), sem exigir configuração adicional.
- O que acontece se dois clientes MCP diferentes (ou um cliente MCP e o chat) operam sobre o
  mesmo incidente ao mesmo tempo (ex.: ambos tentam resolver)? O resultado deve ser consistente
  com o comportamento já existente do armazenamento compartilhado (uma das chamadas resolve, a
  outra recebe o estado já resolvido ou um erro de "não encontrado" se aplicável).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema DEVE expor um servidor MCP chamado "opspilot", acessível via transporte
  stdio.
- **FR-002**: O servidor MCP DEVE expor exatamente três tools: `list_alerts`, `open_incident` e
  `resolve_incident`.
- **FR-003**: O schema de entrada e as regras de validação de cada tool exposta pelo MCP DEVEM
  ser idênticos aos das tools equivalentes já usadas pelo agente de chat (mesma fonte de
  verdade) — nenhuma regra de validação ou lógica de negócio pode ser duplicada ou reescrita
  para o servidor MCP.
- **FR-004**: `list_alerts` DEVE suportar filtro opcional por status do alerta, com o mesmo
  comportamento já existente hoje.
- **FR-005**: `open_incident` DEVE exigir título, serviço e severidade, e retornar o incidente
  criado; DEVE rejeitar entradas inválidas (ex.: serviço desconhecido, severidade fora do
  conjunto permitido) da mesma forma que a tool equivalente já existente.
- **FR-006**: `resolve_incident` DEVE exigir o id do incidente, aceitar um resumo opcional, e
  retornar o incidente atualizado; DEVE rejeitar um id inexistente da mesma forma que a tool
  equivalente já existente.
- **FR-007**: Todas as três tools DEVEM operar sobre o mesmo armazenamento (mesma fonte de
  dados) já usado pelas demais interfaces do OpsPilot — uma ação feita via MCP deve ser visível
  para o chat e vice-versa, sem estado duplicado.
- **FR-008**: O processo do servidor MCP NUNCA DEVE escrever no canal de saída padrão (stdout)
  além das mensagens do próprio protocolo; qualquer diagnóstico, log ou aviso DEVE ir para o
  canal de erro padrão (stderr).
- **FR-009**: O sistema DEVE poder ser iniciado por um comando dedicado (`npm run mcp`).
- **FR-010**: O sistema DEVE carregar toda configuração de ambiente necessária (ex.: caminho do
  banco) antes do servidor começar a atender chamadas.
- **FR-011**: Erros de validação de negócio (ex.: serviço inexistente, severidade inválida,
  incidente não encontrado) DEVEM ser retornados ao cliente MCP como uma resposta de erro da
  chamada da tool, e NUNCA DEVEM derrubar o processo do servidor.
- **FR-012**: DEVE existir um teste automatizado que sobe o servidor MCP e valida a lista de
  tools expostas (nomes e formato de schema esperados).

### Key Entities *(include if feature involves data)*

- **Alerta**: evento de monitoramento com um status (ex.: disparado/resolvido); já existente no
  domínio do OpsPilot, apenas consultado (leitura) por esta feature.
- **Incidente**: registro de um problema em um serviço, com título, serviço, severidade e status
  (aberto/resolvido); já existente no domínio do OpsPilot — esta feature permite criá-lo e
  resolvê-lo também via MCP, usando as mesmas regras já aplicadas pelo chat.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Um cliente MCP compatível com o protocolo consegue se conectar ao servidor e
  descobrir as 3 tools (`list_alerts`, `open_incident`, `resolve_incident`) com seus schemas de
  entrada, sem nenhuma configuração manual além de apontar para o comando de inicialização.
- **SC-002**: Em 100% das consultas, o resultado de `list_alerts` via MCP é idêntico ao
  resultado da mesma consulta feita pelo agente de chat, para o mesmo estado do banco.
- **SC-003**: Um incidente aberto ou resolvido via MCP fica visível, sem nenhuma divergência,
  em qualquer outra interface do OpsPilot que consulte o mesmo armazenamento.
- **SC-004**: Em 100% das execuções do teste automatizado, nenhum byte é escrito no stdout do
  processo além de mensagens válidas do protocolo MCP.
- **SC-005**: A suíte de testes referente ao servidor MCP roda e passa via `npm test` sem
  intervenção manual.

## Assumptions

- O cliente MCP é qualquer cliente compatível com o protocolo conectando via stdio (ex.: Claude
  Desktop, outro agente/runtime) — nenhum cliente específico é o alvo exclusivo desta feature.
- O servidor MCP reutiliza a mesma composição de armazenamento (`OpsStoreRepository` /
  `SqliteOpsStore`, lida a partir de `OPSPILOT_DB`) já usada pelas tools do agente de chat —
  nenhum mecanismo de persistência novo é introduzido.
- Ficam fora do escopo desta feature as tools `list_incidents` e `consultar_runbook`: o pedido
  do usuário lista explicitamente apenas `list_alerts`, `open_incident` e `resolve_incident`.
- Autenticação/autorização de clientes MCP está fora do escopo — mesmo modelo de confiança de
  rodar o processo localmente, sem controle de acesso adicional.
- Erros de negócio (ex.: `ServiceNotFoundError`, `InvalidSeverityError`, `IncidentNotFoundError`)
  são reportados ao cliente MCP como falha da chamada da tool (não como exceção que derruba o
  processo), espelhando o tratamento já existente nas tools do agente de chat.
