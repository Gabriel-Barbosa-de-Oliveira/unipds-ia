# Contract: Operational tools (`src/agents/tools.ts`) — mudanças desta feature

Estende [specs/001-reasoning-strategies-core/contracts/tools.md](../../001-reasoning-strategies-core/contracts/tools.md), que continua valendo para `list_alerts`/`open_incident` sem mudança. Esta feature: (a) adiciona `list_incidents` e `consultar_runbook`; (b) estende `resolve_incident` com `summary` opcional; (c) revisa as descriptions das 5 tools pelas regras da seção [Regra comum de descrições](#regra-comum-de-descrições).

## `resolve_incident` (alterado)

**Input schema (zod, ilustrativo)**:

```ts
z.object({
  id: z.string().min(1).describe("Id do incidente a resolver, retornado por open_incident ou list_incidents"),
  summary: z.string().min(1).optional().describe(
    "Resumo opcional do que foi feito para resolver o incidente, para consulta futura"
  ),
});
```

**Output (sucesso)**: o `Incident` atualizado (`status: "resolved"`, `resolvedAt` preenchido, `summary` preenchido só se informado). Idempotente como antes — resolver um incidente já `resolved` retorna o incidente existente sem erro e sem sobrescrever um `summary` já gravado, mesmo que um novo `summary` seja informado na chamada repetida.

**Erros estruturados**: inalterados (`IncidentNotFoundError`).

## `list_incidents` (novo)

**Input schema (zod, ilustrativo)**:

```ts
z.object({
  status: z.enum(["open", "resolved", "all"]).optional().describe(
    "Filtra por status do incidente; omita ou use \"all\" para listar todos"
  ),
});
```

**Output (sucesso)**: array de `Incident` (pode ser vazio — resultado válido, não erro).

**Erros**: nenhum além de falha de validação zod.

**Quando usar** (na description da tool): consultar incidentes já existentes — o que está aberto agora, o que já foi resolvido, ou o histórico completo. Nunca para abrir um incidente novo (isso é `open_incident`).

## `consultar_runbook` (novo)

**Input schema (zod, ilustrativo)**:

```ts
z.object({
  service: z.string().min(1).describe(
    "Nome do serviço (o mesmo usado em open_incident), ex.: \"checkout-api\""
  ),
});
```

**Output (sucesso)**: `{ service: string, runbook: string | null }` — `runbook: null` quando o serviço existe mas não tem runbook cadastrado (não é um erro; distinguível pelo campo, não por lançar exceção).

**Erros estruturados**:

| Condição | Erro de domínio | Exemplo de shape na observação |
|---|---|---|
| `service` não corresponde a nenhum `Service` semeado | `ServiceNotFoundError` | `{ error: "ServiceNotFoundError", service: "<valor recebido>" }` |

**Quando usar** (na description da tool): obter os passos de mitigação recomendados para um serviço durante um alerta/incidente. Não abre nem resolve nada.

## Regra comum de descrições

Toda tool desta lista (as 3 já existentes e as 2 novas) segue, a partir desta feature:

1. A description da tool declara explicitamente **quando usar** essa tool frente às demais — em particular, `open_incident` deixa claro que serve só para *criar* um incidente novo (não para consultar/listar; isso é `list_incidents`), e `list_alerts`/`list_incidents`/`consultar_runbook` deixam claro que são somente leitura.
2. Todo campo do schema zod tem seu próprio `.describe(...)` — nenhum campo sem descrição própria, mesmo quando o nome já parece autoexplicativo.
3. Todo campo cujo valor pertence a um conjunto fechado e conhecido usa `z.enum([...])` na definição do schema (nunca `z.string()` livre documentando os valores só em texto).

## Regra comum (herdada da feature 001)

Toda chamada de tool, com sucesso ou erro estruturado, continua gerando exatamente um evento `action` seguido de exatamente um evento `observation` no trace — nunca uma exceção não tratada que interrompa o `run` (ver [specs/001-reasoning-strategies-core/contracts/reasoning-strategy.md](../../001-reasoning-strategies-core/contracts/reasoning-strategy.md)). `list_incidents` e `consultar_runbook` seguem a mesma convenção de erro estruturado das tools existentes — nunca lançam para fora de `toStructuredError`.
