# Contract: `POST /chat` (`src/http/server.ts`, `createApp()`) — após `007-semantic-memory`

Estende o contrato de [`006-conversation-history/contracts/post-chat.md`](../../006-conversation-history/contracts/post-chat.md) (permanece a referência histórica daquela feature). Nenhum campo existente é removido, renomeado ou tem seu significado alterado — apenas uma adição.

## Requisição

```http
POST /chat
Content-Type: application/json

{
  "message": "quem cuida do checkout financeiro?",
  "conversationId": "b0a3c1f2-...",
  "userId": "gabriel"
}
```

| Campo | Obrigatório | Tipo | Padrão | Novo? |
|---|---|---|---|---|
| `message`, `strategy`, `reflect`, `conversationId` | — | — | — | Não (inalterados desde `003`/`006`) |
| `userId` | Não | string | — (nenhum: memória semântica desativada nesta requisição) | **Sim** |

## Comportamento quando `userId` é informado

1. Antes de chamar a estratégia de raciocínio, o sistema recupera (`recall`) até 3 fatos previamente registrados para esse `userId` relevantes para `message`, e os disponibiliza ao modelo como contexto — mesmo sem nenhuma palavra em comum com `message` (busca por significado, não por texto).
2. Durante o raciocínio, o modelo pode decidir registrar (`remember_fact`) um fato novo mencionado por quem está conversando, ou esquecer (`forget_fact`) um fato existente, sempre escopados a esse mesmo `userId` — nunca a outro.
3. Nada disso é visível como campo novo na resposta (`200`) — `answer`/`trace`/`metrics`/`conversationId` mantêm exatamente o formato de `006-conversation-history`; a diferença é só o conteúdo de `answer`, que passa a poder refletir fatos lembrados de conversas/sessões anteriores da mesma pessoa.

## Respostas

Sem mudança de forma frente a `006-conversation-history` — `200`/`400`/`404`/`422`/`504`/`500` continuam exatamente como documentado lá. `userId` de tipo errado (ex.: número) cai em `400 invalid_body`, antes de qualquer recall/tool.

## Regras do contrato

- Uma requisição sem `userId` se comporta de forma idêntica ao contrato anterior (`006`) — nenhuma chamada de embedding, nenhuma tool de memória disponível, nenhuma mudança observável (spec Assumptions).
- Fatos de um `userId` nunca influenciam a resposta de outro `userId`, mesmo em requisições concorrentes (spec FR-007, SC-005).
- `userId` e `conversationId` são independentes entre si — memória semântica (por pessoa, entre conversas) e histórico de conversa (por conversa) podem ser usados juntos, cada um com seu próprio identificador, ou isoladamente.
