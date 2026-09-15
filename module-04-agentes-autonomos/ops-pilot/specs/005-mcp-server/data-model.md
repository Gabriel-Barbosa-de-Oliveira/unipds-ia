# Data Model: MCP Server para OpsPilot

Esta feature **não introduz nenhuma entidade de domínio nova**. `Alert` e `Incident` já existem em
`src/domain/ops-store.ts` e são apenas expostos, sem alteração de forma, através de um transporte
adicional (MCP/stdio) sobre o mesmo `OpsStoreRepository` já usado pelo chat.

## Entidades reaproveitadas (referência, sem alteração)

### Alert

| Campo | Tipo | Observação |
|---|---|---|
| `id` | string | já existente |
| `service` | string | já existente |
| `title` | string | já existente |
| `status` | `"firing" \| "resolved"` | já existente |

Somente leitura via `list_alerts` — nenhum campo novo, nenhuma regra de transição de estado
introduzida por esta feature.

### Incident

| Campo | Tipo | Observação |
|---|---|---|
| `id` | string | gerado por `open_incident` |
| `title` | string | obrigatório em `open_incident` |
| `service` | string | obrigatório em `open_incident`; deve existir (senão `ServiceNotFoundError`) |
| `severity` | `"low" \| "medium" \| "high" \| "critical"` | obrigatório em `open_incident` |
| `status` | `"open" \| "resolved"` | `"open"` na criação; `"resolved"` após `resolve_incident` |
| `resolvedAt` | string \| null | `null` até ser resolvido |
| `summary` | string \| undefined | opcional, definido em `resolve_incident` |

Criado/atualizado via `open_incident`/`resolve_incident` — mesmas regras de validação e as mesmas
transições de estado (`open` → `resolved`) já aplicadas hoje pelo chat.

## Formas de entrada/saída específicas do MCP (não são novas entidades — são o contrato de tool)

Estas formas já existem como schemas zod em `src/agents/tools.ts` (ver
[research.md §2](./research.md)) e são apenas reaproveitadas, não redefinidas:

- **ListAlertsInput**: `{ status?: "firing" | "resolved" }`
- **OpenIncidentInput**: `{ title: string (min 1); service: string (min 1); severity: "low"|"medium"|"high"|"critical" }`
- **ResolveIncidentInput**: `{ id: string (min 1); summary?: string (min 1) }`
- **ToolErrorOutput** (forma comum de erro estruturado, reaproveitada de `toStructuredError`):
  `{ error: "ServiceNotFoundError" | "IncidentNotFoundError" | "InvalidSeverityError", ...contexto }`,
  entregue como conteúdo de texto (`JSON.stringify(...)`) de um `CallToolResult` com `isError: true`.

Detalhe completo de request/response por tool está em
[contracts/mcp-tools.md](./contracts/mcp-tools.md).
