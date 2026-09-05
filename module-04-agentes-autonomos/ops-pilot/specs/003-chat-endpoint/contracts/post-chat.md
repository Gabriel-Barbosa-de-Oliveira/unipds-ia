# Contract: `POST /chat` (`src/http/server.ts`, `createApp()`)

## Requisição

```http
POST /chat
Content-Type: application/json

{
  "message": "quais alertas estão firing?",
  "strategy": "react",
  "reflect": false
}
```

| Campo | Obrigatório | Tipo | Padrão |
|---|---|---|---|
| `message` | Sim | string não vazia | — (400 se ausente/vazio/tipo errado) |
| `strategy` | Não | string, DEVE corresponder a um nome registrado (`react`, `plan-and-execute`) | `react` |
| `reflect` | Não | boolean | `false` |

## Respostas

### `200 OK` — sucesso

```json
{
  "answer": "há 3 alertas firing.",
  "trace": [ { "type": "thought", "at": 0, "content": "..." }, "..." ],
  "metrics": { "llmCalls": 2, "latencyMs": 850 }
}
```

Corpo idêntico ao `RunResult` já produzido por `strategy.run(...)` (feature 001) — nenhum campo é adicionado, removido ou renomeado pelo endpoint.

### `400 Bad Request` — corpo inválido

```json
{ "error": "invalid_body", "issues": [ { "path": ["message"], "message": "message é obrigatório e não pode ser vazio" } ] }
```

Disparado pela validação zod do corpo, antes de qualquer resolução de estratégia ou execução (FR-002). `issues` reflete diretamente `parsed.error.issues` do zod.

### `422 Unprocessable Entity` — estratégia desconhecida

```json
{ "error": "unknown_strategy", "strategy": "nome-que-nao-existe" }
```

Disparado quando `strategy` é informado mas não corresponde a nenhum nome do registro (FR-004). Nenhuma execução de raciocínio ocorre.

### `504 Gateway Timeout` — tempo excedido

```json
{ "error": "timeout", "timeoutMs": 180000 }
```

Disparado quando a execução ultrapassa `timeoutMs` (padrão 180000ms, FR-008). O cliente recebe a resposta imediatamente ao atingir o limite; a execução em andamento não é mais aguardada (ver research.md item 2).

### `500 Internal Server Error` — falha não classificada

```json
{ "error": "internal_error" }
```

Qualquer falha que não seja corpo inválido, estratégia desconhecida ou timeout (ex.: credencial do provedor do modelo ausente, erro de rede do provedor). Nenhum detalhe interno é incluído na resposta; o erro completo é logado no servidor.

## Regras do contrato

- Nenhum dos três erros previsíveis (400, 422, 504) dispara execução de raciocínio — validação e resolução de estratégia acontecem estritamente antes de qualquer chamada a `strategy.run(...)` (FR-002, FR-004).
- `strategy` ausente é equivalente a `strategy: "react"` — nunca um erro (FR-003).
- `reflect: true` NUNCA é, por si só, motivo de erro — aplica-se a qualquer estratégia resolvida, base ou explícita (FR-006).
- Cada requisição é processada de forma isolada: `trace` e `metrics` da resposta pertencem exclusivamente àquela requisição, mesmo sob requisições concorrentes (FR-010) — não há estado compartilhado entre chamadas a `POST /chat`.
- `createApp(options?)` aceita `resolveStrategy` e `timeoutMs` como overrides para permitir testes determinísticos sem rede (ver [strategy-registry.md](./strategy-registry.md) e research.md item 4); em produção (`src/index.ts`), o app é criado sem overrides, usando os padrões reais.
