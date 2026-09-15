# Data Model: Conversa Persistente

Entidades novas desta feature — não estendem nenhuma entidade de `001-reasoning-strategies-core`/`004-ops-persistence` (bounded context distinto, ver [research.md](./research.md) item 2). Campos e tipos são descritos de forma independente de implementação; o mapeamento para colunas SQL concretas é detalhe de tarefa.

## Conversation

Agrupa uma sequência ordenada de mensagens entre uma pessoa de plantão e o copiloto.

| Campo | Tipo | Regras |
|---|---|---|
| `id` | string | Identificador estável, gerado em `create()` (`crypto.randomUUID()`) |
| `createdAt` | datetime | Timestamp de criação |

## ConversationMessage

Uma unidade de diálogo dentro de uma `Conversation`.

| Campo | Tipo | Regras |
|---|---|---|
| `conversationId` | string (FK → Conversation.id) | Obrigatório; deve referenciar uma `Conversation` existente |
| `role` | enum: `user` \| `assistant` | Valor fechado — quem originou a mensagem |
| `content` | string | Conteúdo literal do turno — a mensagem crua enviada pela pessoa de plantão, ou a resposta final (`answer`) do copiloto; nunca o prompt já composto com histórico (ver research.md item 3) |
| *(ordem)* | inteiro autoincrementado, interno | Não exposto pela API; garante ordem cronológica exata mesmo entre mensagens gravadas no mesmo milissegundo (ver research.md item 7) |

### Regra de leitura: `lastMessages(conversationId, limit)`

- `conversationId` desconhecido (nenhuma `Conversation` com esse id) → `ConversationNotFoundError`.
- `conversationId` conhecido, sem mensagens ainda → lista vazia (não é erro; é o estado normal logo após `create()`).
- `conversationId` conhecido, com mensagens → as últimas `limit` mensagens, em ordem cronológica (mais antiga primeiro), nunca mais que `limit` mesmo que a conversa tenha um histórico maior.

### Regra de escrita: `append(conversationId, messages)`

- `conversationId` desconhecido → `ConversationNotFoundError` (mesma regra de leitura; nunca cria a conversa implicitamente).
- Cada chamada grava um ou mais `ConversationMessage` na ordem em que aparecem no array — usada pelo controller para gravar, em uma única chamada, o par `[mensagem da pessoa de plantão, resposta do copiloto]` ao final de cada turno.

## Relacionamentos

```text
Conversation 1 ── * ConversationMessage   (uma conversa tem zero ou mais mensagens)
```

## Repositório (`ConversationStore`) — superfície desta feature

| Método | Regras |
|---|---|
| `create(): Promise<string>` | Cria uma `Conversation` nova e retorna seu `id`; nunca falha por conflito de id (gerado internamente) |
| `append(conversationId: string, messages: ConversationMessage[]): Promise<void>` | Ver regra de escrita acima |
| `lastMessages(conversationId: string, limit: number): Promise<ConversationMessage[]>` | Ver regra de leitura acima |

## Contrato HTTP (`POST /chat`) — campos novos

Ver [contracts/post-chat.md](./contracts/post-chat.md) para o contrato completo do endpoint após esta feature.

| Campo | Onde | Regra |
|---|---|---|
| `conversationId` (requisição) | body, opcional | Quando informado, deve corresponder a uma `Conversation` existente (senão `404`); quando omitido, uma `Conversation` nova é criada |
| `conversationId` (resposta) | body, em toda resposta `200` | O id efetivamente usado — o informado na requisição, ou o recém-criado |
| `metrics.historyMessages` (resposta) | body, em toda resposta `200` | Quantidade de mensagens de histórico (0 a 12) efetivamente incluídas na composição do prompt desta resposta |
