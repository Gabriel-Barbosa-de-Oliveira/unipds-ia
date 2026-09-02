# Contract: Operational tools (`src/agents/tools.ts`)

Três tools LangChain expostas às estratégias de raciocínio, cada uma com um schema zod de entrada validado antes de qualquer chamada ao service layer (`OpsStoreRepository`, ver [data-model.md](../data-model.md)).

## `list_alerts`

**Input schema (zod, ilustrativo)**:

```ts
z.object({
  status: z.enum(["firing", "resolved"]).optional(),
});
```

**Output (sucesso)**: array de `Alert` (pode ser vazio — resultado válido, não erro; ver spec Edge Cases).

**Erros**: nenhum além de falha de validação zod (não há como um `status` inválido chegar ao service, já que o enum barra na fronteira).

## `open_incident`

**Input schema (zod, ilustrativo)**:

```ts
z.object({
  title: z.string().min(1),
  service: z.string().min(1),
  severity: z.enum(["low", "medium", "high", "critical"]),
});
```

**Output (sucesso)**: o `Incident` criado (`status: "open"`, `resolvedAt: null`).

**Erros estruturados** (retornados como observação, nunca como exceção não tratada até a estratégia):

| Condição | Erro de domínio | Exemplo de shape na observação |
|---|---|---|
| `service` não corresponde a nenhum `Service` semeado | `ServiceNotFoundError` | `{ error: "ServiceNotFoundError", service: "<valor recebido>" }` |
| `severity` fora do enum | rejeitado na validação zod (não chega ao domínio) | erro de validação zod formatado |

## `resolve_incident`

**Input schema (zod, ilustrativo)**:

```ts
z.object({
  id: z.string().min(1),
});
```

**Output (sucesso)**: o `Incident` atualizado (`status: "resolved"`, `resolvedAt` preenchido). Se já estava `resolved`, retorna o mesmo incidente sem erro (idempotente — ver [data-model.md](../data-model.md)).

**Erros estruturados**:

| Condição | Erro de domínio | Exemplo de shape na observação |
|---|---|---|
| `id` não corresponde a nenhum `Incident` existente | `IncidentNotFoundError` | `{ error: "IncidentNotFoundError", id: "<valor recebido>" }` |

## Regra comum

Toda chamada de tool, com sucesso ou erro estruturado, gera exatamente um evento `action` (com `tool` + `args` já validados) seguido de exatamente um evento `observation` (com o resultado ou o erro estruturado) no trace da estratégia que a invocou — nunca uma exceção não tratada que interrompa o `run` (ver [contracts/reasoning-strategy.md](./reasoning-strategy.md)).
