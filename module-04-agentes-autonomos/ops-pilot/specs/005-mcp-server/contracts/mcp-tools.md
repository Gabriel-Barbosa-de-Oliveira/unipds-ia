# Contract: Tools do servidor MCP `opspilot`

Transporte: **stdio**. Nome do servidor: **`opspilot`**. Todas as três tools abaixo usam,
literalmente (mesma constante importada, não uma cópia), os schemas zod já definidos para as
tools equivalentes do agente de chat em `src/agents/tools.ts` — ver
[research.md §2](../research.md).

## `list_alerts`

- **Descrição**: consulta somente leitura dos alertas de monitoramento; mesmo comportamento da
  tool `list_alerts` já usada pelo chat.
- **Input** (`ListAlertsInput`):
  | Campo | Tipo | Obrigatório | Descrição |
  |---|---|---|---|
  | `status` | `"firing" \| "resolved"` | não | filtra por status; omitido = todos os alertas |
- **Output (sucesso)**: `content: [{ type: "text", text: JSON.stringify(Alert[]) }]`
- **Output (erro)**: nenhum erro de domínio esperado para esta tool (é somente leitura); falhas de
  infraestrutura não são capturadas e derrubam a chamada como erro de protocolo.
- **Acceptance ref**: User Story 1, cenários 1-2 (`spec.md`).

## `open_incident`

- **Descrição**: cria um novo incidente para um serviço; mesmo comportamento da tool
  `open_incident` já usada pelo chat.
- **Input** (`OpenIncidentInput`):
  | Campo | Tipo | Obrigatório | Descrição |
  |---|---|---|---|
  | `title` | string (mín. 1 char) | sim | título curto do problema |
  | `service` | string (mín. 1 char) | sim | nome do serviço afetado |
  | `severity` | `"low" \| "medium" \| "high" \| "critical"` | sim | severidade do incidente |
- **Output (sucesso)**: `content: [{ type: "text", text: JSON.stringify(Incident) }]`, com
  `Incident.status === "open"` e `Incident.resolvedAt === null`.
- **Output (erro)**: `isError: true`, `content: [{ type: "text", text: JSON.stringify({ error: "ServiceNotFoundError", service }) }]`
  quando o serviço não existe (reaproveita `toStructuredError`).
- **Acceptance ref**: User Story 2, cenários 1-2 (`spec.md`).

## `resolve_incident`

- **Descrição**: resolve um incidente já existente pelo id; mesmo comportamento da tool
  `resolve_incident` já usada pelo chat.
- **Input** (`ResolveIncidentInput`):
  | Campo | Tipo | Obrigatório | Descrição |
  |---|---|---|---|
  | `id` | string (mín. 1 char) | sim | id do incidente a resolver |
  | `summary` | string (mín. 1 char) | não | resumo opcional da resolução |
- **Output (sucesso)**: `content: [{ type: "text", text: JSON.stringify(Incident) }]`, com
  `Incident.status === "resolved"` e `Incident.summary` igual ao informado (se houver).
- **Output (erro)**: `isError: true`, `content: [{ type: "text", text: JSON.stringify({ error: "IncidentNotFoundError", id }) }]`
  quando o id não existe (reaproveita `toStructuredError`).
- **Acceptance ref**: User Story 3, cenários 1-2 (`spec.md`).

## Contrato de descoberta (`tools/list`)

Um cliente MCP conectado ao servidor `opspilot` via stdio, ao chamar `listTools()`, DEVE receber
exatamente 3 tools, com esses nomes e esses schemas de entrada (convertidos para JSON Schema pelo
SDK a partir dos mesmos shapes zod acima) — nem mais, nem menos. Esta é a asserção central do
teste automatizado exigido por FR-012 (`src/mcp/server.test.ts`).
